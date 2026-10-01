"""Shared fixtures: tiny random checkpoints in real key layouts, converted into containers."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selas.container import Container  # noqa: E402
from selas.convert import ConvertOptions, convert_flux, flux_view  # noqa: E402
from selas.models.flux import FluxConfig  # noqa: E402
from selas.sources import TensorSource  # noqa: E402

TINY_FLUX = FluxConfig(
    hidden=64, heads=4, mlp_hidden=256, depth_double=2, depth_single=3, in_channels=64,  # mlp_ratio 4 like diffusers
    context_dim=32, vec_dim=24, axes_dim=(4, 6, 6), guidance_embed=True,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _lin(sd: dict, g: torch.Generator, name: str, i: int, o: int, bias: bool = True) -> None:
    sd[name + ".weight"] = torch.randn(o, i, generator=g) / math.sqrt(i)
    if bias:
        sd[name + ".bias"] = torch.randn(o, generator=g) * 0.02


def flux_state(cfg: FluxConfig = TINY_FLUX, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random FLUX weights with BFL key names (fp32)."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    d, m, hd = cfg.hidden, cfg.mlp_hidden, cfg.head_dim
    _lin(sd, g, "img_in", cfg.in_channels, d)
    _lin(sd, g, "txt_in", cfg.context_dim, d)
    _lin(sd, g, "time_in.in_layer", 256, d)
    _lin(sd, g, "time_in.out_layer", d, d)
    _lin(sd, g, "vector_in.in_layer", cfg.vec_dim, d)
    _lin(sd, g, "vector_in.out_layer", d, d)
    if cfg.guidance_embed:
        _lin(sd, g, "guidance_in.in_layer", 256, d)
        _lin(sd, g, "guidance_in.out_layer", d, d)

    def norms(prefix):
        sd[prefix + "query_norm.scale"] = 1 + 0.1 * torch.randn(hd, generator=g)
        sd[prefix + "key_norm.scale"] = 1 + 0.1 * torch.randn(hd, generator=g)

    for i in range(cfg.depth_double):
        p = f"double_blocks.{i}."
        for s in ("img", "txt"):
            _lin(sd, g, p + f"{s}_mod.lin", d, 6 * d)
            _lin(sd, g, p + f"{s}_attn.qkv", d, 3 * d)
            _lin(sd, g, p + f"{s}_attn.proj", d, d)
            _lin(sd, g, p + f"{s}_mlp.0", d, m)
            _lin(sd, g, p + f"{s}_mlp.2", m, d)
            norms(p + f"{s}_attn.norm.")
    for i in range(cfg.depth_single):
        p = f"single_blocks.{i}."
        _lin(sd, g, p + "modulation.lin", d, 3 * d)
        _lin(sd, g, p + "linear1", d, 3 * d + m)
        _lin(sd, g, p + "linear2", d + m, d)
        norms(p + "norm.")
    _lin(sd, g, "final_layer.linear", d, cfg.in_channels)
    _lin(sd, g, "final_layer.adaLN_modulation.1", d, 2 * d)
    return sd


def save_st(path: Path, sd: dict[str, torch.Tensor]) -> Path:
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()}, str(path))  # clone: no shared storage
    return path


def build_flux(tmp: Path, sd: dict, cfg: FluxConfig = TINY_FLUX, dtype=torch.float32, quant="none", name="flux", src_dtype=None) -> Container:
    if src_dtype is not None:
        sd = {k: v.to(src_dtype) for k, v in sd.items()}
    path = save_st(tmp / f"{name}.safetensors", sd)
    out = tmp / name
    convert_flux(flux_view(TensorSource([path])), out, ConvertOptions(dtype=dtype, quant=quant, quant_report=False), axes_dim=cfg.axes_dim)
    return Container(out)


def tiny_inputs(cfg: FluxConfig = TINY_FLUX, batch: int = 2, height: int = 64, width: int = 64, l_txt: int = 8, seed: int = 1):
    from selas.sampling import get_noise, image_ids, pack, text_ids

    g = torch.Generator().manual_seed(seed)
    txt = torch.randn(batch, l_txt, cfg.context_dim, generator=g)
    pooled = torch.randn(batch, cfg.vec_dim, generator=g)
    x = torch.cat([pack(get_noise(seed + i, height, width)) for i in range(batch)])
    return x, txt, pooled, image_ids(height, width), text_ids(l_txt)


@pytest.fixture
def tiny_sd():
    return flux_state(TINY_FLUX, seed=0)


def real_model_dir() -> Path | None:
    p = os.environ.get("SELAS_TEST_MODEL")
    return Path(p) if p else None
