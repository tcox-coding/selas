"""Tests on a real converted FLUX.1-dev (opt-in).

    SELAS_TEST_MODEL=/path/to/converted/model  pytest tests/test_real_model.py
    SELAS_TEST_SOURCE=/path/to/original.safetensors   # enables the bitsandbytes cross-check
    SELAS_TEST_OUT=/path/to/dir                       # keep the generated images
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from selas.container import Container
from selas.keymap import bfl_from_diffusers
from selas.models.flux import FluxConfig, FluxRunner, double_block, main_order, mod_order, rope_tables, single_block
from selas.sampling import get_schedule, image_ids, pack, get_noise, text_ids
from selas.store import DISK, VRAM, WeightStore

from .conftest import needs_cuda, real_model_dir

MODEL = real_model_dir()
pytestmark = [
    needs_cuda,
    pytest.mark.realmodel,
    pytest.mark.skipif(MODEL is None, reason="set SELAS_TEST_MODEL to a converted FLUX model"),
]
DEV = torch.device("cuda", 0) if torch.cuda.is_available() else None
F32 = torch.float32


def _subset(c: Container, names):
    c.units = {k: c.units[k] for k in names}
    return c


def _to_diffusers(view_tensors: dict, kind: str, cfg: FluxConfig) -> dict:
    d, m = cfg.hidden, cfg.mlp_hidden
    if kind == "double":
        table, bp, dp = bfl_from_diffusers(1, 0), "double_blocks.0.", "transformer_blocks.0."
    else:
        table, bp, dp = bfl_from_diffusers(0, 1), "single_blocks.0.", "single_transformer_blocks.0."
    out = {}
    for bkey, (op, dkeys) in table.items():
        if not bkey.startswith(bp):
            continue
        t = view_tensors[bkey[len(bp):]]
        dkeys = [k[len(dp):] for k in dkeys]
        if op == "cat":
            sizes = [d, d, d] if len(dkeys) == 3 else [d, d, d, m]
            out.update(dict(zip(dkeys, t.split(sizes, dim=0))))
        else:
            out[dkeys[0]] = t
    return out


def test_real_blocks_match_diffusers():
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    c = Container(MODEL / "transformer")
    cfg = FluxConfig.from_dict(c.config)
    names = ["double.0", "double.0.mod", "single.0", "single.0.mod"]
    _subset(c, names)
    g = torch.Generator().manual_seed(0)
    lt, h = 16, 128  # 128x128 px -> 8x8 = 64 image tokens
    li = (h // 16) ** 2
    img = torch.randn(1, li, cfg.hidden, generator=g).to(DEV)
    txt = torch.randn(1, lt, cfg.hidden, generator=g).to(DEV)
    vec = torch.randn(1, cfg.hidden, generator=g).to(DEV)
    ids = torch.cat((text_ids(lt), image_ids(h, h))).to(DEV)
    cos, sin = rope_tables(ids, cfg.axes_dim, cfg.theta)
    rot = tf.FluxPosEmbed(theta=int(cfg.theta), axes_dim=list(cfg.axes_dim))(ids)
    with torch.no_grad(), WeightStore(c, {n: VRAM for n in names}, DEV, F32) as st:
        sv = F.silu(vec)
        dm = st.view("double.0.mod")
        mod = torch.cat((dm.linear(sv, "img_mod.lin"), dm.linear(sv, "txt_mod.lin")), dim=-1)
        oi, ot = double_block(st.view("double.0"), img, txt, mod, cos, sin, cfg, F32, False)
        w = {n: st.view("double.0").get(n, F32) for n in st.view("double.0").spec.tensors}
        w |= {n: dm.get(n, F32) for n in dm.spec.tensors}
        blk = tf.FluxTransformerBlock(dim=cfg.hidden, num_attention_heads=cfg.heads, attention_head_dim=cfg.head_dim).to(DEV).eval()
        blk.load_state_dict(_to_diffusers(w, "double", cfg))
        et, hi = blk(hidden_states=img, encoder_hidden_states=txt, temb=vec, image_rotary_emb=rot)
        torch.testing.assert_close(oi, hi, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ot, et, rtol=1e-3, atol=1e-3)
        del blk, w

        x = torch.cat((txt, img), dim=1)
        sm = st.view("single.0.mod")
        ox = single_block(st.view("single.0"), x, sm.linear(sv, "modulation.lin"), cos, sin, cfg, F32, False)
        w = {n: st.view("single.0").get(n, F32) for n in st.view("single.0").spec.tensors}
        w |= {n: sm.get(n, F32) for n in sm.spec.tensors}
        sb = tf.FluxSingleTransformerBlock(dim=cfg.hidden, num_attention_heads=cfg.heads, attention_head_dim=cfg.head_dim).to(DEV).eval()
        sb.load_state_dict(_to_diffusers(w, "single", cfg))
        if "encoder_hidden_states" in inspect.signature(sb.forward).parameters:
            e2, h2 = sb(hidden_states=img, encoder_hidden_states=txt, temb=vec, image_rotary_emb=rot)
            ref = torch.cat((e2, h2), dim=1)
        else:
            ref = sb(hidden_states=x, temb=vec, image_rotary_emb=rot)
        torch.testing.assert_close(ox, ref, rtol=1e-3, atol=1e-3)


def _real_embedding(prompt: str = "a photo of a red fox sitting in fresh snow, golden hour"):
    """(T5 txt [L, 4096], CLIP pooled [768]) from the model's own encoders (prompt-cached)."""
    import json

    from selas.text import TextEncoders

    te = TextEncoders(MODEL, json.loads((MODEL / "model.json").read_text()), DEV)
    try:
        return te.encode([prompt])[prompt]
    finally:
        te.close()


def test_real_placements_bit_exact():
    c = Container(MODEL / "transformer")
    cfg = FluxConfig.from_dict(c.config)
    dtype = torch.float16 if torch.cuda.get_device_capability(DEV) < (8, 0) else torch.bfloat16
    main, mods = main_order(cfg), mod_order(cfg)
    free = torch.cuda.mem_get_info(DEV)[0]
    budget = free - c.units["globals"].nbytes - 3 * 2**30
    resident, used = set(), 0
    for n in main:  # a "first N resident" placement that fits whatever VRAM is free
        if used + c.units[n].nbytes > budget:
            break
        resident.add(n)
        used += c.units[n].nbytes
    arena = 3 * max(c.units[n].nbytes for n in main + mods)
    # Real conditioning: a random pooled vector lies off CLIP's output manifold and drives the
    # modulations to ~5e4 (also in fp32), which overflows any fp16 forward.
    txt, pooled = _real_embedding()
    txt, pooled = txt[None, :64], pooled[None]
    h = 256
    x0 = pack(get_noise(0, h, h)).to(DEV)
    sigmas = get_schedule(2, x0.shape[1])
    outs = []
    for tiers in ({"globals": VRAM, **{n: (VRAM if n in resident else DISK) for n in main + mods}},
                  {"globals": VRAM, **{n: DISK for n in main + mods}}):
        with WeightStore(c, tiers, DEV, dtype, arena_bytes=arena) as st:
            r = FluxRunner(st, cfg, dtype)
            cond = r.prepare(sigmas, 3.5, pooled, txt, image_ids(h, h), text_ids(64))
            x = x0.clone()
            for s in range(2):
                pred, _ = r.step(s, x, cond)
                x = x + (sigmas[s + 1] - sigmas[s]) * pred
            r.end()
            outs.append(x.cpu())
    assert torch.isfinite(outs[0]).all()
    assert torch.equal(outs[0], outs[1])


@pytest.mark.slow
def test_real_generation_end_to_end(tmp_path):
    from selas.pipeline import FluxEngine, Job, RuntimeOptions

    out_dir = Path(os.environ.get("SELAS_TEST_OUT", tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = FluxEngine(MODEL, RuntimeOptions(profile=True))
    try:
        res = eng.generate([Job("a photo of a red fox sitting in fresh snow, golden hour", seed=0, width=512, height=512, steps=20)])
    finally:
        eng.close()
    img = res[0].image
    img.save(out_dir / "selas_e2e_512.png")
    import numpy as np

    a = np.asarray(img, dtype=np.float32)
    assert a.shape == (512, 512, 3)
    assert a.std() > 10, "image is (nearly) constant"
    st = res[0].stats
    assert st["mean_full_step_s"] > 0 and all(d == "full" for d in st["decisions"])


def test_nf4_import_matches_bitsandbytes():
    src = os.environ.get("SELAS_TEST_SOURCE")
    if not src:
        pytest.skip("set SELAS_TEST_SOURCE to the original NF4 checkpoint")
    bnbf = pytest.importorskip("bitsandbytes.functional")
    from selas.codecs import decode
    from selas.sources import SourceView, TensorSource

    s = TensorSource([src])
    prefix = s.find_prefix("double_blocks.0.img_attn.qkv.weight")
    v = SourceView(s, prefix)
    key = "double_blocks.3.img_mlp.0.weight"
    if not v.is_nf4(key):
        pytest.skip("source is not bnb-NF4")
    full = prefix + key
    qs = {k[len(full) + 1:]: s.get(k) for k in s.keys() if k.startswith(full + ".")}
    state = bnbf.QuantState.from_dict(qs, device=DEV)
    ref = bnbf.dequantize_4bit(s.get(full).to(DEV), state)
    e = v.encoded_nf4(key, "w")
    ours = decode("nf4", {k: t.to(DEV) for k, t in e.parts.items()}, e.meta, e.shape, F32)
    assert torch.equal(ours.to(ref.dtype), ref)
