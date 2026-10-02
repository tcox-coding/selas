"""Learned activation reserve: prediction from measured peaks, recording, and the out-of-memory fallback."""

from __future__ import annotations

import json

import pytest
import torch

from selas.pipeline import ACT_HEADROOM, ACT_HEADROOM_EXACT, ACT_MARGIN, ACT_MARGIN_EXACT, learned_activation

from .conftest import TINY_FLUX, build_flux, flux_state, needs_cuda, tiny_inputs


def _est(o: dict) -> int:  # toy analytic model: proportional to tokens x batch
    return (o["l_img"] + o["l_txt"]) * o["micro"] * 1000


def test_no_observations_means_no_prediction():
    assert learned_activation([], {"l_img": 4096, "l_txt": 512, "batch": 1, "micro": 1}, _est) is None


def test_exact_shape_uses_its_measurement():
    obs = [{"l_img": 4096, "l_txt": 512, "batch": 1, "micro": 1, "bytes": 600 << 20},
           {"l_img": 1024, "l_txt": 512, "batch": 1, "micro": 1, "bytes": 100 << 20}]
    got = learned_activation(obs, {"l_img": 4096, "l_txt": 512, "batch": 1, "micro": 1}, _est)
    assert got == int((600 << 20) * ACT_MARGIN_EXACT + ACT_HEADROOM_EXACT)


def test_other_shapes_scale_from_the_nearest_observation():
    obs = [{"l_img": 4096, "l_txt": 512, "batch": 1, "micro": 1, "bytes": 600 << 20},
           {"l_img": 256, "l_txt": 512, "batch": 1, "micro": 1, "bytes": 90 << 20}]
    shape = {"l_img": 4096, "l_txt": 512, "batch": 2, "micro": 2}  # 2x the 1024² estimate
    assert learned_activation(obs, shape, _est) == int((600 << 20) * 2 * ACT_MARGIN + ACT_HEADROOM)
    assert learned_activation(obs, shape, _est, safety=1.5) == int(1.5 * ((600 << 20) * 2 * ACT_MARGIN + ACT_HEADROOM))


@pytest.fixture
def tiny_engine(tmp_path, monkeypatch):
    from selas.pipeline import FluxEngine, RuntimeOptions

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = tmp_path / "model"
    model.mkdir()
    build_flux(model, flux_state(TINY_FLUX, seed=0), dtype=torch.float16, name="transformer")
    (model / "model.json").write_text(json.dumps({
        "format": "selas-model", "components": {"transformer": "transformer"},
        "defaults": {"steps": 3, "guidance": 3.5, "max_t5_tokens": 8, "shift": True},
    }))
    eng = FluxEngine(model, RuntimeOptions(keep_loaded=True))
    _, txt, pooled, _, _ = tiny_inputs(TINY_FLUX, batch=1)
    yield eng, {"p": (txt[0], pooled[0])}
    eng.close()


def _profile(eng) -> dict:
    return json.loads((eng.dir / ".selas" / "profile.json").read_text())[eng._gpu_key]


@needs_cuda
def test_reserve_is_learned_from_a_run(tiny_engine):
    from selas.pipeline import Job
    from selas.stepcache import StepCacheConfig

    eng, embeds = tiny_engine
    jobs = [Job("p", seed=0, width=64, height=64, steps=3)]
    first = eng.make_plan(64, 64, 1, 3, StepCacheConfig())
    assert eng.reserve_source == "estimated" and "(estimated)" in first.notes[-1]
    est, _ = eng._activation_reserve(16, 8, 1, 1)
    eng.denoise(jobs, embeds, StepCacheConfig())
    obs = _profile(eng)["activation"]
    assert len(obs) == 1 and obs[0]["l_img"] == 16 and obs[0]["bytes"] >= 0
    second = eng.make_plan(64, 64, 1, 3, StepCacheConfig())
    learned, source = eng._activation_reserve(16, 8, 1, 1)
    assert source == "learned" and "(learned)" in second.notes[-1]
    assert learned < est


@needs_cuda
def test_oom_with_learned_reserve_retries_on_the_estimate(tiny_engine, monkeypatch):
    from selas.models.flux import FluxRunner
    from selas.pipeline import Job
    from selas.stepcache import StepCacheConfig

    eng, embeds = tiny_engine
    jobs = [Job("p", seed=0, width=64, height=64, steps=3)]
    eng.denoise(jobs, embeds, StepCacheConfig())  # learn once (this plan itself was estimated)
    eng.unload()  # the next run re-plans, now with the learned reserve
    real_step, raised = FluxRunner.step, []

    def flaky(self, *a, **k):
        if not raised:
            raised.append(eng.reserve_source)
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_step(self, *a, **k)

    monkeypatch.setattr(FluxRunner, "step", flaky)
    lat, stats = eng._denoise_safe(jobs, embeds, StepCacheConfig())
    assert raised == ["learned"] and eng.reserve_source == "estimated"
    assert torch.isfinite(lat).all() and len(stats["step_s"]) == 3
    assert _profile(eng)["activation_safety"] == pytest.approx(1.5)
    eng.make_plan(64, 64, 1, 3, StepCacheConfig())
    assert eng.reserve_source == "learned"  # back to learning, with the wider margin


@needs_cuda
def test_repeat_measurements_keep_the_worst_case(tiny_engine):
    eng, _ = tiny_engine
    shape = {"l_img": 16, "l_txt": 8, "batch": 1, "micro": 1}
    eng._learn_activation(shape, 5 << 20)
    eng._learn_activation(shape, 3 << 20)
    eng._learn_activation({**shape, "batch": 2, "micro": 2}, 9 << 20)
    obs = {(o["batch"], o["bytes"]) for o in _profile(eng)["activation"]}
    assert obs == {(1, 5 << 20), (2, 9 << 20)}


@needs_cuda
def test_oom_with_estimated_reserve_is_not_swallowed(tiny_engine, monkeypatch):
    from selas.models.flux import FluxRunner
    from selas.pipeline import Job
    from selas.stepcache import StepCacheConfig

    eng, embeds = tiny_engine

    def oom(self, *a, **k):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(FluxRunner, "step", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        eng._denoise_safe([Job("p", seed=0, width=64, height=64, steps=3)], embeds, StepCacheConfig())


@needs_cuda
def test_warm_up_compiles_everything_the_run_needs(tiny_engine):
    """Kernels compile while the weights load; compiling mid-run would stall step 1 and skip learning."""
    from selas.models.flux import compile_count
    from selas.pipeline import Job
    from selas.stepcache import StepCacheConfig

    eng, embeds = tiny_engine
    if not eng.kernels.compiled:
        pytest.skip("torch.compile unavailable")
    eng.ensure_loaded(64, 64, 1, 3, StepCacheConfig())
    n = compile_count()
    eng.denoise([Job("p", seed=0, width=64, height=64, steps=3)], embeds, StepCacheConfig())
    assert compile_count() == n


@needs_cuda
def test_compute_calibration_is_kept_per_numerics_mode(tiny_engine):
    """--fp16-accum runs are ~1.6x faster: their calibration must not overwrite the default's."""
    from selas.pipeline import Job
    from selas.stepcache import StepCacheConfig

    eng, embeds = tiny_engine
    jobs = [Job("p", seed=0, width=64, height=64, steps=3)]
    eng.denoise(jobs, embeds, StepCacheConfig())
    default = {k for k in _profile(eng) if k.startswith(("double", "single"))}
    assert default == {eng._calib_key("double"), eng._calib_key("single")}
    eng.fp16_accum = True
    eng.denoise(jobs, embeds, StepCacheConfig())
    mode = "fused" if eng.kernels.compiled else "eager"
    added = {k for k in _profile(eng) if k.startswith(("double", "single"))} - default
    assert added == {f"double|{mode}+acc16", f"single|{mode}+acc16"}
