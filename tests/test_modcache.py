"""Modulation-vector cache: exact reuse, unit skipping, batch independence, eviction."""

from __future__ import annotations

import json
import os

import pytest
import torch

from selas.modcache import ModCache
from selas.store import ABSENT

from .conftest import TINY_FLUX, build_flux, flux_state, needs_cuda, tiny_inputs


def test_key_depends_on_every_input(tmp_path):
    c = ModCache(tmp_path, "abc", torch.float16)
    pooled, sig = torch.randn(768), [1.0, 0.5, 0.0]
    k = c.key(pooled, 3.5, sig)
    assert k == c.key(pooled.clone(), 3.5, list(sig))
    assert len({k, c.key(pooled + 1e-3, 3.5, sig), c.key(pooled, 4.0, sig), c.key(pooled, 3.5, [1.0, 0.4, 0.0]),
                ModCache(tmp_path, "abd", torch.float16).key(pooled, 3.5, sig),
                ModCache(tmp_path, "abc", torch.bfloat16).key(pooled, 3.5, sig)}) == 6


def test_roundtrip_and_eviction(tmp_path):
    entry = {"double.0": torch.randn(4, 1, 12, dtype=torch.float16), "single.0": torch.randn(4, 1, 3, dtype=torch.float16)}
    c = ModCache(tmp_path, "abc", torch.float16, limit_bytes=10**9)
    c.save("k1", entry)
    got = c.load("k1")
    assert set(got) == set(entry) and all(torch.equal(got[n], entry[n][:, 0]) for n in entry)
    size = (tmp_path / ".selas" / "mods" / "k1.safetensors").stat().st_size
    small = ModCache(tmp_path, "abc", torch.float16, limit_bytes=int(2.5 * size))
    small.save("k2", entry)
    os.utime(tmp_path / ".selas" / "mods" / "k1.safetensors", (1, 1))  # k1 least recently used
    small.save("k3", entry)
    assert not small.has("k1") and small.has("k2") and small.has("k3")


def test_disabled_cache_never_hits(tmp_path):
    c = ModCache(tmp_path, "abc", torch.float16, enabled=False)
    c.save("k", {"a": torch.zeros(2, 1, 3)})
    assert not c.has("k") and c.load("k") is None


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model = tmp_path / "model"
    model.mkdir()
    build_flux(model, flux_state(TINY_FLUX, seed=0), dtype=torch.float16, name="transformer")
    (model / "model.json").write_text(json.dumps({
        "format": "selas-model", "components": {"transformer": "transformer"},
        "defaults": {"steps": 3, "guidance": 3.5, "max_t5_tokens": 8, "shift": True},
    }))
    return model


def _embeds():
    _, txt, pooled, _, _ = tiny_inputs(TINY_FLUX, batch=2)
    return {"a": (txt[0], pooled[0]), "b": (txt[1], pooled[1])}


def _run(model, prompts, **rt):
    from selas.pipeline import FluxEngine, Job, RuntimeOptions
    from selas.stepcache import StepCacheConfig

    eng = FluxEngine(model, RuntimeOptions(**rt))
    try:
        lat, stats = eng.denoise([Job(p, seed=i, width=64, height=64, steps=3) for i, p in enumerate(prompts)],
                                 _embeds(), StepCacheConfig())
        absent = sorted(n for n, t in eng.store.tier.items() if t == ABSENT)
    finally:
        eng.close()
    return lat, stats, absent


@needs_cuda
def test_cached_run_is_exact_and_skips_the_modulation_units(model_dir):
    lat1, st1, absent1 = _run(model_dir, ["a"], keep_loaded=True)
    assert (st1["mods_cached"], st1["mods_computed"]) == (0, 1) and absent1 == []
    lat2, st2, absent2 = _run(model_dir, ["a"], keep_loaded=True)
    assert (st2["mods_cached"], st2["mods_computed"]) == (1, 0)
    assert absent2 and all(n.endswith(".mod") for n in absent2)  # never loaded
    assert st2["prologue_h2d"]["host"] == st2["prologue_h2d"]["disk"] == 0
    assert torch.equal(lat1, lat2)


@needs_cuda
def test_entries_do_not_depend_on_the_batch(model_dir, tmp_path):
    _run(model_dir, ["a"], keep_loaded=True)
    root = model_dir / ".selas" / "mods"
    alone = {p.name: p.read_bytes() for p in root.glob("*.safetensors")}
    for p in root.glob("*.safetensors"):
        p.unlink()
    _, st, _ = _run(model_dir, ["b", "a"], keep_loaded=True)
    assert st["mods_computed"] == 2
    batched = {p.name: p.read_bytes() for p in root.glob("*.safetensors")}
    assert len(batched) == 2 and set(alone) <= set(batched)
    assert all(batched[n] == alone[n] for n in alone)  # same bits as when computed alone


@needs_cuda
def test_partial_hit_computes_only_the_missing_images(model_dir):
    _run(model_dir, ["a"], keep_loaded=True)
    lat, st, absent = _run(model_dir, ["a", "b"], keep_loaded=True)
    assert (st["mods_cached"], st["mods_computed"]) == (1, 1) and absent == []
    lat_cold, _, _ = _run(model_dir, ["a", "b"], keep_loaded=True, prompt_cache=False)
    assert torch.equal(lat, lat_cold)


@needs_cuda
def test_store_with_modulation_units_is_reused_not_reloaded(model_dir):
    from selas.pipeline import FluxEngine, RuntimeOptions
    from selas.stepcache import StepCacheConfig

    eng = FluxEngine(model_dir, RuntimeOptions(keep_loaded=True))
    try:
        eng.ensure_loaded(64, 64, 1, 3, StepCacheConfig(), need_mods=True)
        full = eng.store
        eng.ensure_loaded(64, 64, 1, 3, StepCacheConfig(), need_mods=False)
        assert eng.store is full  # a superset: no reload
        eng.unload()
        eng.ensure_loaded(64, 64, 1, 3, StepCacheConfig(), need_mods=False)
        lean = eng.store
        assert any(t == ABSENT for t in lean.tier.values())
        eng.ensure_loaded(64, 64, 1, 3, StepCacheConfig(), need_mods=True)
        assert eng.store is not lean and ABSENT not in eng.store.tier.values()
    finally:
        eng.close()
