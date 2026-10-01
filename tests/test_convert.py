"""Converting a Forge-style all-in-one checkpoint (bnb NF4 transformer + fp8 T5 + CLIP + VAE)."""

from __future__ import annotations

import json

import pytest
import torch

from selas.codecs import encode_nf4
from selas.container import Container
from selas.convert import ConvertOptions, convert_model

from .conftest import TINY_FLUX, flux_state, save_st


def _bnb_pack(prefix: str, name: str, t: torch.Tensor) -> dict:
    """Serialize like bitsandbytes' packed QuantState (Forge NF4 v2: fp32 absmax, no nesting)."""
    e = encode_nf4(name, t)
    state = json.dumps({"quant_type": "nf4", "blocksize": 64, "dtype": "bfloat16", "shape": list(t.shape)}).encode()
    return {
        prefix + name: e.parts["packed"].reshape(-1, 1),
        prefix + name + ".absmax": e.parts["absmax"],
        prefix + name + ".quant_map": torch.tensor(e.meta["code"], dtype=torch.float32),
        prefix + name + ".quant_state.bitsandbytes__nf4": torch.tensor(list(state), dtype=torch.uint8),
    }


def _tiny_t5_clip_vae():
    transformers = pytest.importorskip("transformers")
    diffusers = pytest.importorskip("diffusers")
    from .test_oracles import _diffusers_vae_to_ldm

    torch.manual_seed(0)
    t5 = transformers.T5EncoderModel(transformers.T5Config(vocab_size=64, d_model=32, d_kv=8, num_heads=4, d_ff=64, num_layers=2,
                                                           feed_forward_proj="gated-gelu")).state_dict()
    clip = transformers.CLIPTextModel(transformers.CLIPTextConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                                                                  num_hidden_layers=1, num_attention_heads=1)).state_dict()
    vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                                  up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(32, 64), layers_per_block=1,
                                  latent_channels=16, use_quant_conv=False, use_post_quant_conv=False).state_dict()
    return t5, clip, _diffusers_vae_to_ldm(vae, 2)


def test_forge_all_in_one(tmp_path):
    t5, clip, vae = _tiny_t5_clip_vae()
    sd = flux_state(TINY_FLUX, seed=5)
    aio = {}
    pre = "model.diffusion_model."
    for k, v in sd.items():
        if k.endswith(".weight") and v.dim() == 2:
            aio.update(_bnb_pack(pre, k, v))
        elif k.endswith(".bias"):
            aio[pre + k] = v.to(torch.bfloat16)
        else:
            aio[pre + k] = v
    for k, v in t5.items():
        if k == "encoder.embed_tokens.weight":
            continue
        aio["text_encoders.t5xxl.transformer." + k] = v.to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else v.half()
    for k, v in clip.items():
        if not k.endswith("position_ids"):
            aio["text_encoders.clip_l.transformer." + k] = v.half()
    for k, v in vae.items():
        aio["vae." + k] = v
    path = save_st(tmp_path / "aio.safetensors", aio)

    out = convert_model(tmp_path / "model", [str(path)], opts=ConvertOptions(dtype=torch.float16, quant_report=False),
                        axes_dim=TINY_FLUX.axes_dim)
    info = json.loads((out / "model.json").read_text())
    assert info["variant"] == "dev" and set(info["components"]) == {"transformer", "t5", "clip", "vae"}

    c = Container(out / "transformer")
    assert c.config["source_quant"] == "nf4"
    u = c.units["double.1"]
    t = u.tensors["img_attn.qkv.weight"]
    assert t.codec == "nf4" and t.shape == (3 * TINY_FLUX.hidden, TINY_FLUX.hidden)
    buf = torch.empty(u.nbytes, dtype=torch.uint8)
    c.read_unit_into(u, buf)
    from selas.container import part_view

    packed = part_view(buf, t.parts["packed"])
    assert torch.equal(packed, aio[pre + "double_blocks.1.img_attn.qkv.weight"].reshape(-1))  # imported bit-exactly
    assert u.tensors["img_attn.qkv.bias"].stored_dtype == "float16"
    assert u.tensors["img_attn.norm.query_norm.scale"].stored_dtype == "float32"
    assert "img_mod.lin.weight" in c.units["double.1.mod"].tensors

    t5c = Container(out / "t5")
    assert t5c.config["num_layers"] == 2 and t5c.config["num_heads"] == 4
    assert {"globals", "block.0", "block.1"} == set(t5c.units)
    assert Container(out / "clip").config["layers"] == 1
    assert Container(out / "vae").config["ch_mult"] == [1, 2]
