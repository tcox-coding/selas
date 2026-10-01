"""Our functional models vs reference implementations on tiny random weights (fp32, CPU).

FLUX transformer and VAE vs diffusers; T5 and CLIP vs transformers.
"""

from __future__ import annotations

import pytest
import torch

from selas.container import Container
from selas.convert import ConvertOptions, clip_view, convert_clip, convert_t5, convert_vae, t5_view, vae_view
from selas.keymap import bfl_from_diffusers
from selas.models.clip import ClipConfig, ClipTextEncoder
from selas.models.flux import FluxRunner
from selas.models.t5 import T5Config, T5Encoder
from selas.models.vae import VaeConfig, VaeDecoder
from selas.sampling import get_schedule
from selas.sources import TensorSource
from selas.store import VRAM, WeightStore

from .conftest import TINY_FLUX, build_flux, save_st, tiny_inputs

pytestmark = pytest.mark.oracle
CPU = torch.device("cpu")
OPTS = ConvertOptions(dtype=torch.float32, quant_report=False)


def bfl_to_diffusers(sd: dict, cfg) -> dict:
    d, m = cfg.hidden, cfg.mlp_hidden
    out = {}
    for bkey, (op, dkeys) in bfl_from_diffusers(cfg.depth_double, cfg.depth_single, cfg.guidance_embed).items():
        t = sd[bkey]
        if op == "id":
            out[dkeys[0]] = t
        elif op == "swap":
            a, b = t.chunk(2, dim=0)
            out[dkeys[0]] = torch.cat((b, a), dim=0)
        else:
            sizes = [d, d, d] if len(dkeys) == 3 else [d, d, d, m]
            for k, part in zip(dkeys, t.split(sizes, dim=0)):
                out[k] = part
    return out


def test_flux_matches_diffusers(tmp_path, tiny_sd):
    diffusers = pytest.importorskip("diffusers")
    cfg = TINY_FLUX
    ref = diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=cfg.in_channels, num_layers=cfg.depth_double, num_single_layers=cfg.depth_single,
        attention_head_dim=cfg.head_dim, num_attention_heads=cfg.heads, joint_attention_dim=cfg.context_dim,
        pooled_projection_dim=cfg.vec_dim, guidance_embeds=True, axes_dims_rope=tuple(cfg.axes_dim),
    ).eval()
    missing, unexpected = ref.load_state_dict(bfl_to_diffusers(tiny_sd, cfg), strict=False)
    assert not unexpected and not missing, (missing, unexpected)

    c = build_flux(tmp_path, tiny_sd)
    x, txt, pooled, img_ids, txt_ids = tiny_inputs(cfg)
    sigmas = get_schedule(4, x.shape[1])
    with WeightStore(c, {}, CPU, torch.float32) as st:
        runner = FluxRunner(st, cfg, torch.float32)
        cond = runner.prepare(sigmas, 3.5, pooled, txt, img_ids, txt_ids)
        for s in (0, 2):
            ours, _ = runner.step(s, x, cond)
            with torch.no_grad():
                theirs = ref(hidden_states=x, encoder_hidden_states=txt, pooled_projections=pooled,
                             timestep=torch.full((x.shape[0],), sigmas[s]), img_ids=img_ids, txt_ids=txt_ids,
                             guidance=torch.full((x.shape[0],), 3.5), return_dict=False)[0]
            torch.testing.assert_close(ours, theirs, rtol=2e-4, atol=2e-5)
        runner.end()


def test_flux_from_diffusers_layout_converts_identically(tmp_path, tiny_sd):
    """A diffusers-format checkpoint must produce byte-identical units to the BFL one."""
    from selas.convert import convert_flux, flux_view

    cfg = TINY_FLUX
    a = build_flux(tmp_path, tiny_sd, name="bfl")
    p = save_st(tmp_path / "diff.safetensors", bfl_to_diffusers(tiny_sd, cfg))
    convert_flux(flux_view(TensorSource([p])), tmp_path / "diff", OPTS, axes_dim=cfg.axes_dim)
    b = Container(tmp_path / "diff")
    assert a.config == b.config
    assert [u.hash for u in a.units.values()] == [u.hash for u in b.units.values()]


def test_t5_matches_transformers(tmp_path):
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.T5Config(vocab_size=128, d_model=32, d_kv=8, num_heads=4, d_ff=64, num_layers=2,
                                   feed_forward_proj="gated-gelu", dropout_rate=0.0, layer_norm_epsilon=1e-6)
    torch.manual_seed(0)
    ref = transformers.T5EncoderModel(hf_cfg).eval()
    with torch.no_grad():  # non-trivial norm weights
        for n, p in ref.named_parameters():
            if "layer_norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    sd = {k: v.clone() for k, v in ref.state_dict().items()}
    path = save_st(tmp_path / "t5.safetensors", sd)
    convert_t5(t5_view(TensorSource([path])), tmp_path / "t5", OPTS)
    c = Container(tmp_path / "t5")
    ids = torch.randint(0, 128, (3, 40))
    with WeightStore(c, {}, CPU, torch.float32) as st:
        ours = T5Encoder(st, T5Config.from_dict(c.config)).encode(ids, micro_batch=2)
    with torch.no_grad():
        theirs = ref(input_ids=ids).last_hidden_state
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)


def test_clip_matches_transformers(tmp_path):
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.CLIPTextConfig(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                         num_attention_heads=2, max_position_embeddings=77, hidden_act="quick_gelu",
                                         eos_token_id=999, bos_token_id=998, pad_token_id=999)
    torch.manual_seed(0)
    ref = transformers.CLIPTextModel(hf_cfg).eval()
    sd = {k: v.clone() for k, v in ref.state_dict().items() if not k.endswith("position_ids")}
    path = save_st(tmp_path / "clip.safetensors", sd)
    convert_clip(clip_view(TensorSource([path])), tmp_path / "clip", OPTS)
    c = Container(tmp_path / "clip")
    ids = torch.randint(0, 990, (2, 77))
    ids[:, 0] = 998
    ids[0, 10:] = 999
    ids[1, 30:] = 999
    with WeightStore(c, {}, CPU, torch.float32) as st:
        hidden, pooled = ClipTextEncoder(st, ClipConfig.from_dict(c.config)).encode(ids)
    with torch.no_grad():
        out = ref(input_ids=ids)
    torch.testing.assert_close(hidden, out.last_hidden_state, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(pooled, out.pooler_output, rtol=1e-4, atol=1e-5)


def _diffusers_vae_to_ldm(sd: dict, n_levels: int) -> dict:
    out = {}
    for k, v in sd.items():
        if not k.startswith("decoder."):
            continue
        k2 = k[len("decoder."):]
        k2 = k2.replace("mid_block.resnets.0.", "mid.block_1.").replace("mid_block.resnets.1.", "mid.block_2.")
        if k2.startswith("mid_block.attentions.0."):
            rest = k2[len("mid_block.attentions.0."):]
            rest = {"group_norm.weight": "norm.weight", "group_norm.bias": "norm.bias"}.get(rest, rest)
            for a, b in (("to_q.", "q."), ("to_k.", "k."), ("to_v.", "v."), ("to_out.0.", "proj_out.")):
                if rest.startswith(a):
                    rest = b + rest[len(a):]
            if rest.endswith(".weight") and v.dim() == 2:
                v = v[:, :, None, None]
            k2 = "mid.attn_1." + rest
        if k2.startswith("up_blocks."):
            parts = k2.split(".")
            lvl = n_levels - 1 - int(parts[1])
            if parts[2] == "resnets":
                k2 = f"up.{lvl}.block.{parts[3]}." + ".".join(parts[4:])
            else:  # upsamplers.0.conv.*
                k2 = f"up.{lvl}.upsample." + ".".join(parts[4:])
        k2 = k2.replace("conv_shortcut", "nin_shortcut").replace("conv_norm_out", "norm_out")
        out["decoder." + k2] = v
    return out


def test_vae_decoder_matches_diffusers(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 64), layers_per_block=1, latent_channels=16, norm_num_groups=32,
        use_quant_conv=False, use_post_quant_conv=False,
    ).eval()
    sd = _diffusers_vae_to_ldm(ref.state_dict(), 2)
    path = save_st(tmp_path / "ae.safetensors", sd)
    convert_vae(vae_view(TensorSource([path])), tmp_path / "vae", OPTS)
    c = Container(tmp_path / "vae")
    cfg = VaeConfig.from_dict(c.config)
    assert cfg.ch == 32 and cfg.ch_mult == (1, 2) and cfg.num_res_blocks == 1
    z = torch.randn(1, 16, 8, 12)
    with WeightStore(c, {"decoder": VRAM}, CPU, torch.float32) as st:
        dec = VaeDecoder(st, cfg)
        ours = dec.decode(z)
        tiled = dec.decode_tiled(z, tile=6, overlap=2)
    with torch.no_grad():
        theirs = ref.decode(z / cfg.scale_factor + cfg.shift_factor).sample.clamp(-1, 1)
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)
    assert tiled.shape == ours.shape and torch.isfinite(tiled).all()
