"""Placement must never change numerics: VRAM / host / disk / tiny-arena runs are bit-identical.

Also exercises the step cache and block-major micro-batching through the real streaming path.
"""

from __future__ import annotations

import pytest
import torch

from selas.convert import ConvertOptions, convert_t5, t5_view
from selas.container import Container
from selas.models.flux import FluxRunner, main_order, mod_order
from selas.models.t5 import T5Config, T5Encoder
from selas.sampling import get_schedule
from selas.sources import TensorSource
from selas.stepcache import FULL, StepCache, StepCacheConfig
from selas.store import DISK, HOST, VRAM, WeightStore

from .conftest import TINY_FLUX, build_flux, needs_cuda, save_st, tiny_inputs

pytestmark = [needs_cuda, pytest.mark.cuda]
DEV = torch.device("cuda", 0) if torch.cuda.is_available() else None


def _run(c, tiers, dtype=torch.float16, steps=3, arena=None, staging=0, cache_cfg=None, micro=None, batch=2, host_res=False):
    cfg = TINY_FLUX
    x, txt, pooled, img_ids, txt_ids = tiny_inputs(cfg, batch=batch)
    x = x.to(DEV)
    names = main_order(cfg) + mod_order(cfg)
    if arena is None:
        arena = 3 * max(c.units[n].nbytes for n in names)
    sigmas = get_schedule(steps, x.shape[1])
    with WeightStore(c, tiers, DEV, dtype, arena_bytes=arena, staging_bytes=staging) as st:
        runner = FluxRunner(st, cfg, dtype, micro_batch=micro)
        cond = runner.prepare(sigmas, 3.5, pooled, txt, img_ids, txt_ids)
        streamed = {n for n in runner.main if runner.is_streamed(n)}
        cache = StepCache(cache_cfg, steps, streamed, torch.bfloat16, host_residuals=host_res) if cache_cfg else None
        decisions = []
        for s in range(steps):
            pred, info = runner.step(s, x, cond, cache)
            decisions.append(info.decision)
            x = x + (sigmas[s + 1] - sigmas[s]) * pred
        runner.end()
        torch.cuda.synchronize()
        if cache is not None:
            cache.close()
        stats = st.stats
    return x.cpu(), decisions, stats


def _tiers(c, mode):
    cfg = TINY_FLUX
    t = {"globals": VRAM}
    for i, n in enumerate(main_order(cfg) + mod_order(cfg)):
        if mode == "mixed":
            t[n] = (VRAM, HOST, DISK)[i % 3]
        else:
            t[n] = mode
    return t


@pytest.fixture(scope="module")
def tiny_container(tmp_path_factory):
    from .conftest import flux_state

    tmp = tmp_path_factory.mktemp("flux")
    return build_flux(tmp, flux_state(TINY_FLUX, seed=0), dtype=torch.float16)


@pytest.mark.parametrize("mode", [HOST, DISK, "mixed"])
def test_placement_is_bit_exact(tiny_container, mode):
    ref, _, _ = _run(tiny_container, _tiers(tiny_container, VRAM))
    out, _, stats = _run(tiny_container, _tiers(tiny_container, mode))
    assert torch.isfinite(ref).all()
    assert torch.equal(out, ref)
    assert stats.units_streamed > 0


def test_minimal_arena_and_staging_are_bit_exact(tiny_container):
    c = tiny_container
    names = main_order(TINY_FLUX) + mod_order(TINY_FLUX)
    biggest = max(c.units[n].nbytes for n in names)
    ref, _, _ = _run(c, _tiers(c, VRAM))
    out, _, _ = _run(c, _tiers(c, "mixed"), arena=biggest, staging=0)  # one unit in flight at a time
    assert torch.equal(out, ref)


def test_cyclic_prefetch_crosses_step_boundaries(tiny_container):
    c = tiny_container
    _, _, stats = _run(c, _tiers(c, HOST), steps=4)
    n_main = len(main_order(TINY_FLUX))
    # every step streams every main unit once; prefetch may run at most an arena's worth ahead
    assert stats.units_streamed >= 4 * n_main


def test_stream_order_violation_is_detected(tiny_container):
    c = tiny_container
    with WeightStore(c, _tiers(c, HOST), DEV, torch.float16, arena_bytes=4 * max(u.nbytes for u in c.units.values())) as st:
        stream = st.stream(main_order(TINY_FLUX), cyclic=True)
        with pytest.raises(RuntimeError, match="order violated"):
            stream.acquire("double.1")
        stream.close()


def test_cache_with_zero_threshold_is_exact(tiny_container):
    c = tiny_container
    ref, _, _ = _run(c, _tiers(c, "mixed") | {"double.0": VRAM}, steps=5)
    out, dec, _ = _run(c, _tiers(c, "mixed") | {"double.0": VRAM}, steps=5,
                       cache_cfg=StepCacheConfig("fbcache", threshold=0.0))
    assert all(d == FULL for d in dec)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("policy,predict", [("fbcache", "reuse"), ("fbcache", "linear"), ("tiered", "reuse"), ("tiered", "linear")])
def test_cache_policies_run_and_stay_close(tiny_container, policy, predict):
    c = tiny_container
    tiers = _tiers(c, "mixed") | {"double.0": VRAM}
    ref, _, _ = _run(c, tiers, steps=8)
    cfg = StepCacheConfig(policy, threshold=10.0, predict=predict, warmup=2, tail_full=1, max_consecutive=2)
    out, dec, stats = _run(c, tiers, steps=8, cache_cfg=cfg)
    assert any(d != FULL for d in dec)
    assert torch.isfinite(out).all()
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.5  # random tiny weights: just guard against nonsense


@pytest.mark.parametrize("predict", ["reuse", "linear"])
def test_tiered_host_residuals_match_device_residuals(tiny_container, predict):
    """Residuals parked in pinned RAM (D2H on a side stream, H2D on reuse) must equal VRAM-kept ones."""
    c = tiny_container
    tiers = _tiers(c, HOST) | {"double.0": VRAM}
    cfg = StepCacheConfig("tiered", threshold=10.0, predict=predict, warmup=2, tail_full=1, max_consecutive=2)
    dev, dec, _ = _run(c, tiers, steps=10, cache_cfg=cfg)
    host, dec2, _ = _run(c, tiers, steps=10, cache_cfg=cfg, host_res=True)
    assert dec == dec2 and dec.count("hybrid") >= 3
    assert torch.equal(dev, host)


def test_skipped_steps_skip_transfers(tiny_container):
    c = tiny_container
    tiers = _tiers(c, HOST) | {"double.0": VRAM}
    _, _, full_stats = _run(c, tiers, steps=6)
    cfg = StepCacheConfig("fbcache", threshold=10.0, warmup=1, tail_full=1, max_consecutive=10)
    _, dec, cache_stats = _run(c, tiers, steps=6, cache_cfg=cfg)
    assert dec.count(FULL) == 2
    assert cache_stats.h2d_bytes[HOST] < full_stats.h2d_bytes[HOST]


def test_micro_batching_matches_full_batch(tiny_container):
    c = tiny_container
    full, _, _ = _run(c, _tiers(c, "mixed"), batch=3, dtype=torch.float32)
    micro, _, _ = _run(c, _tiers(c, "mixed"), batch=3, micro=1, dtype=torch.float32)
    torch.testing.assert_close(micro, full, rtol=1e-4, atol=1e-5)


def test_nf4_container_streams_bit_exact(tmp_path):
    from .conftest import flux_state

    c = build_flux(tmp_path, flux_state(TINY_FLUX, seed=3), dtype=torch.float16, quant="nf4")
    assert c.units["double.0"].tensors["img_attn.qkv.weight"].codec == "nf4"
    ref, _, _ = _run(c, _tiers(c, VRAM))
    out, _, _ = _run(c, _tiers(c, "mixed"))
    assert torch.equal(out, ref)


def test_t5_streamed_matches_resident(tmp_path):
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.T5Config(vocab_size=128, d_model=64, d_kv=16, num_heads=4, d_ff=128, num_layers=4,
                                   feed_forward_proj="gated-gelu", dropout_rate=0.0)
    torch.manual_seed(0)
    sd = {k: v.clone().to(torch.bfloat16) for k, v in transformers.T5EncoderModel(hf_cfg).state_dict().items()}
    path = save_st(tmp_path / "t5.safetensors", sd)
    convert_t5(t5_view(TensorSource([path])), tmp_path / "t5", ConvertOptions(quant_report=False))
    c = Container(tmp_path / "t5")
    ids = torch.randint(0, 128, (2, 24))
    blocks = [n for n in c.units if n != "globals"]
    arena = max(c.units[n].nbytes for n in blocks)
    outs = []
    for tier in (VRAM, HOST, DISK):
        with WeightStore(c, {"globals": VRAM, **{n: tier for n in blocks}}, DEV, torch.float32, arena_bytes=arena) as st:
            outs.append(T5Encoder(st, T5Config.from_dict(c.config)).encode(ids).cpu())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
