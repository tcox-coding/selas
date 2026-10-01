"""Convert FLUX checkpoints into selas containers.

Supported sources (auto-detected by key layout):

* transformer: BFL single file (``flux1-dev.safetensors``), Forge/bnb NF4
  all-in-one (``model.diffusion_model.*`` with bnb NF4 quant states), fp8
  ComfyUI files (unscaled), diffusers ``transformer/`` folders;
* T5-XXL: HF ``T5EncoderModel`` keys (``encoder.block.*``, ``shared.weight``),
  bare or under a prefix such as ``text_encoders.t5xxl.transformer.``;
* CLIP-L: HF ``CLIPTextModel`` keys (``text_model.*``);
* VAE: BFL/LDM ``ae.safetensors`` layout (``decoder.mid.block_1...``).

Layout of the output directory::

    model.json  transformer/  t5/  clip/  vae/  tokenizers/{t5,clip}/
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from . import __version__
from .codecs import EncodedTensor, decode, encode_int8_row, encode_nf4, encode_raw
from .container import ContainerWriter
from .models.clip import ClipConfig
from .models.flux import FluxConfig
from .models.t5 import T5Config
from .models.vae import VaeConfig
from .sources import DiffusersFluxView, SourceView, TensorSource
from .util import human_bytes, log, warn, write_json_atomic

FLOAT8 = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n))


@dataclass
class ConvertOptions:
    dtype: torch.dtype = torch.float16  # storage dtype for unquantized float transformer weights
    quant: str = "none"  # none | int8 | nf4 (transformer linear weights only)
    hash_units: bool = True
    quant_report: bool = True


class _QuantStats:
    def __init__(self):
        self.err2 = 0.0
        self.ref2 = 0.0
        self.worst = (0.0, "")
        self.n = 0

    def add(self, name: str, ref: torch.Tensor, enc: EncodedTensor):
        deq = decode(enc.codec, enc.parts, enc.meta, enc.shape, torch.float32)
        r = ref.float()
        e2 = float((deq - r).pow(2).sum())
        r2 = float(r.pow(2).sum())
        self.err2 += e2
        self.ref2 += r2
        self.n += 1
        rel = (e2 / r2) ** 0.5 if r2 > 0 else 0.0
        if rel > self.worst[0]:
            self.worst = (rel, name)

    def summary(self) -> str:
        if not self.n:
            return ""
        tot = (self.err2 / self.ref2) ** 0.5 if self.ref2 else 0.0
        return f"quantization: {self.n} tensors, relative RMS error {tot:.4%} overall, worst {self.worst[0]:.4%} ({self.worst[1]})"


# --------------------------------------------------------------------------- detection


def flux_view(src: TensorSource):
    p = src.find_prefix("double_blocks.0.img_attn.qkv.weight")
    if p is not None:
        return SourceView(src, p)
    p = src.find_prefix("transformer_blocks.0.attn.to_q.weight")
    if p is not None:
        return DiffusersFluxView(src, p)
    return None


def t5_view(src: TensorSource) -> SourceView | None:
    p = src.find_prefix("encoder.block.0.layer.0.SelfAttention.q.weight")
    return SourceView(src, p) if p is not None else None


def clip_view(src: TensorSource) -> SourceView | None:
    # HF/Forge files nest it under ``text_model.``; transformers 5 state dicts do not.
    p = src.find_prefix("embeddings.token_embedding.weight")
    if p is None or not src.has(p + "encoder.layers.0.self_attn.q_proj.weight"):
        return None
    return SourceView(src, p)


def vae_view(src: TensorSource) -> SourceView | None:
    p = src.find_prefix("decoder.mid.block_1.conv1.weight")
    if p is not None:
        return SourceView(src, p + "decoder.")
    if src.find_prefix("decoder.mid_block.resnets.0.conv1.weight") is not None:
        raise NotImplementedError("diffusers-format VAE found; pass the BFL ae.safetensors (LDM layout) instead")
    return None


def _indices(keys, pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return sorted({int(m.group(1)) for k in keys if (m := rx.match(k))})


# --------------------------------------------------------------------------- FLUX


def infer_flux_config(v, axes_dim=None) -> FluxConfig:
    keys = v.keys()
    hidden, in_ch = v.shape("img_in.weight")[:2]
    head_dim = v.shape("double_blocks.0.img_attn.norm.query_norm.scale")[0]
    nd = len(_indices(keys, r"double_blocks\.(\d+)\."))
    ns = len(_indices(keys, r"single_blocks\.(\d+)\."))
    if axes_dim is None:
        if head_dim != 128:
            raise ValueError(f"cannot infer RoPE axes for head_dim {head_dim}; pass axes_dim")
        axes_dim = (16, 56, 56)
    if sum(axes_dim) != head_dim:
        raise ValueError(f"axes_dim {axes_dim} must sum to head_dim {head_dim}")
    return FluxConfig(
        hidden=hidden,
        heads=hidden // head_dim,
        mlp_hidden=v.shape("double_blocks.0.img_mlp.0.weight")[0],
        depth_double=nd,
        depth_single=ns,
        in_channels=in_ch,
        context_dim=v.shape("txt_in.weight")[1],
        vec_dim=v.shape("vector_in.in_layer.weight")[1],
        axes_dim=tuple(axes_dim),
        guidance_embed=v.has("guidance_in.in_layer.weight"),
    )


def flux_layout(cfg: FluxConfig) -> list[tuple[str, dict, str, list[str]]]:
    """(unit name, attrs, source key prefix, tensor names relative to that prefix)."""

    def lin(*names):
        return [f"{n}.{p}" for n in names for p in ("weight", "bias")]

    units = []
    g = lin("img_in", "txt_in", "time_in.in_layer", "time_in.out_layer", "vector_in.in_layer", "vector_in.out_layer",
            "final_layer.linear", "final_layer.adaLN_modulation.1")
    if cfg.guidance_embed:
        g += lin("guidance_in.in_layer", "guidance_in.out_layer")
    units.append(("globals", {"group": "globals", "kind": "globals"}, "", g))
    for i in range(cfg.depth_double):
        names = lin("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2", "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")
        names += [f"{s}_attn.norm.{q}_norm.scale" for s in ("img", "txt") for q in ("query", "key")]
        units.append((f"double.{i}", {"group": "main", "kind": "double", "index": i}, f"double_blocks.{i}.", names))
    for i in range(cfg.depth_single):
        names = lin("linear1", "linear2") + ["norm.query_norm.scale", "norm.key_norm.scale"]
        units.append((f"single.{i}", {"group": "main", "kind": "single", "index": i}, f"single_blocks.{i}.", names))
    for i in range(cfg.depth_double):
        units.append((f"double.{i}.mod", {"group": "mod", "kind": "double_mod", "index": i}, f"double_blocks.{i}.", lin("img_mod.lin", "txt_mod.lin")))
    for i in range(cfg.depth_single):
        units.append((f"single.{i}.mod", {"group": "mod", "kind": "single_mod", "index": i}, f"single_blocks.{i}.", lin("modulation.lin")))
    return units


def _encode_flux_tensor(v, key: str, name: str, group: str, opts: ConvertOptions, stats: _QuantStats | None) -> EncodedTensor:
    if v.is_nf4(key):
        return v.encoded_nf4(key, name)
    t = v.get(key)
    if name.endswith("_norm.scale"):
        return encode_raw(name, t, torch.float32)
    if name.endswith(".bias"):
        return encode_raw(name, t, opts.dtype)
    if t.dtype in FLOAT8:
        if opts.quant == "none":
            return encode_raw(name, t)  # keep fp8 storage; cast to the compute dtype at use
        t = t.to(torch.float32)
    if opts.quant == "none" or group == "globals" or t.dim() != 2:
        return encode_raw(name, t, opts.dtype)
    enc = encode_int8_row(name, t) if opts.quant == "int8" else encode_nf4(name, t)
    if stats is not None:
        stats.add(name, t, enc)
    return enc


def convert_flux(v, out_dir: Path, opts: ConvertOptions, axes_dim=None) -> FluxConfig:
    if any(k.endswith(".scale_weight") for k in v.keys()):
        raise NotImplementedError("scaled-fp8 FLUX checkpoints are not supported yet")
    cfg = infer_flux_config(v, axes_dim)
    src_q = "nf4" if any(v.is_nf4(k) for k in ("double_blocks.0.img_attn.qkv.weight",)) else None
    if src_q and opts.quant not in ("none", src_q):
        warn(f"source is already {src_q}; --quant {opts.quant} ignored (selas never re-quantizes a quantized source)")
    log(f"FLUX: {cfg.depth_double} double + {cfg.depth_single} single blocks, hidden {cfg.hidden}, "
        f"{'dev (guidance)' if cfg.guidance_embed else 'schnell'}, source {src_q or 'float'}")
    stats = _QuantStats() if (opts.quant != "none" and opts.quant_report and not src_q) else None
    w = ContainerWriter(out_dir, "flux-transformer", {**cfg.to_dict(), "source_quant": src_q, "quant": src_q or opts.quant},
                        hash_units=opts.hash_units)
    try:
        layout = flux_layout(cfg)
        for k, (unit, attrs, prefix, names) in enumerate(layout):
            enc = []
            for n in names:
                key = prefix + n
                if not v.has(key):
                    raise KeyError(f"missing FLUX tensor {key}")
                enc.append(_encode_flux_tensor(v, key, n, attrs["group"], opts, stats))
            size = w.add_unit(unit, enc, **attrs)
            if k % 10 == 0 or k == len(layout) - 1:
                log(f"  [{k + 1}/{len(layout)}] {unit} {human_bytes(size)}")
    except BaseException:
        w.abort()
        raise
    w.close({"selas_version": __version__})
    if stats is not None:
        log(stats.summary())
    return cfg


# --------------------------------------------------------------------------- T5 / CLIP / VAE


def convert_t5(v: SourceView, out_dir: Path, opts: ConvertOptions) -> T5Config:
    keys = v.keys()
    layers = _indices(keys, r"encoder\.block\.(\d+)\.")
    emb = "shared.weight" if v.has("shared.weight") else "encoder.embed_tokens.weight"
    if not v.has(emb):
        raise KeyError("T5 embedding (shared.weight) not found")
    rel = v.shape("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    heads = rel[1]
    cfg = T5Config(
        d_model=v.shape(emb)[1], num_heads=heads,
        d_kv=v.shape("encoder.block.0.layer.0.SelfAttention.q.weight")[0] // heads,
        num_layers=len(layers), num_buckets=rel[0],
    )

    # fp32 sources are halved to bf16 (T5-XXL: 19 -> 9.5 GB; compute stays fp32) unless fp32 storage was asked for.
    down = None if opts.dtype == torch.float32 else torch.bfloat16

    def enc(name, key):
        t = v.get(key)
        return encode_raw(name, t, down if t.dtype == torch.float32 and t.dim() == 2 else None)

    w = ContainerWriter(out_dir, "t5-encoder", cfg.to_dict(), hash_units=opts.hash_units)
    try:
        w.add_unit("globals", [enc("shared.weight", emb), enc("final_layer_norm.weight", "encoder.final_layer_norm.weight")], group="globals")
        for i in layers:
            p = f"encoder.block.{i}."
            names = sorted(k[len(p) :] for k in keys if k.startswith(p))
            w.add_unit(f"block.{i}", [enc(n, p + n) for n in names], group="main", kind="t5_block", index=i)
    except BaseException:
        w.abort()
        raise
    w.close({"selas_version": __version__, "source_dtype": str(v.dtype(emb))})
    log(f"T5: {cfg.num_layers} blocks, d_model {cfg.d_model}, stored as {v.dtype('encoder.block.0.layer.0.SelfAttention.q.weight')}")
    return cfg


def convert_clip(v: SourceView, out_dir: Path, opts: ConvertOptions) -> ClipConfig:
    keys = [k for k in v.keys() if not k.endswith("position_ids")]
    hidden = v.shape("embeddings.token_embedding.weight")[1]
    cfg = ClipConfig(
        hidden=hidden, heads=hidden // 64, layers=len(_indices(keys, r"encoder\.layers\.(\d+)\.")),
        max_positions=v.shape("embeddings.position_embedding.weight")[0],
    )
    w = ContainerWriter(out_dir, "clip-text", cfg.to_dict(), hash_units=opts.hash_units)
    try:
        w.add_unit("all", [encode_raw(k, v.get(k)) for k in sorted(keys)], group="globals")
    except BaseException:
        w.abort()
        raise
    w.close({"selas_version": __version__})
    log(f"CLIP: {cfg.layers} layers, hidden {cfg.hidden}")
    return cfg


def convert_vae(v: SourceView, out_dir: Path, opts: ConvertOptions) -> VaeConfig:
    keys = v.keys()
    ch = v.shape("conv_out.weight")[1]
    z = v.shape("conv_in.weight")[1]
    levels = _indices(keys, r"up\.(\d+)\.")
    nblocks = len(_indices([k for k in keys if k.startswith("up.0.")], r"up\.0\.block\.(\d+)\."))
    mult = tuple(v.shape(f"up.{i}.block.0.conv2.weight")[0] // ch for i in levels)
    if z != 16:
        raise NotImplementedError(f"VAE with {z} latent channels is not a FLUX autoencoder")
    cfg = VaeConfig(ch=ch, ch_mult=mult, num_res_blocks=nblocks - 1, z_channels=z)
    w = ContainerWriter(out_dir, "vae-decoder", cfg.to_dict(), hash_units=opts.hash_units)
    try:
        w.add_unit("decoder", [encode_raw(k, v.get(k)) for k in sorted(keys)], group="globals")
    except BaseException:
        w.abort()
        raise
    w.close({"selas_version": __version__})
    log(f"VAE decoder: ch {ch}, mult {mult}")
    return cfg


# --------------------------------------------------------------------------- tokenizers & driver


def _copy_tokenizer(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"tokenizer directory {src} not found")
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.stat().st_size < 64 << 20:
            shutil.copy2(f, dst / f.name)


def _find_tokenizer(sources: list[str], sub: str) -> Path | None:
    for s in sources:
        base = Path(s)
        for root in (base if base.is_dir() else base.parent, (base if base.is_dir() else base.parent).parent):
            cand = root / sub
            if (cand / "tokenizer_config.json").exists():
                return cand
    return None


def convert_model(
    out: str | os.PathLike,
    flux: list[str],
    t5: list[str] | None = None,
    clip: list[str] | None = None,
    vae: list[str] | None = None,
    t5_tokenizer: str | None = None,
    clip_tokenizer: str | None = None,
    opts: ConvertOptions | None = None,
    components: tuple[str, ...] = ("transformer", "t5", "clip", "vae"),
    axes_dim=None,
) -> Path:
    opts = opts or ConvertOptions()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src_main = TensorSource(flux)
    srcs = {"t5": TensorSource(t5) if t5 else src_main, "clip": TensorSource(clip) if clip else src_main,
            "vae": TensorSource(vae) if vae else src_main}
    info = {"format": "selas-model", "version": 1, "family": "flux", "selas_version": __version__,
            "components": {}, "tokenizers": {}, "sources": {"flux": flux, "t5": t5, "clip": clip, "vae": vae}}

    if "transformer" in components:
        v = flux_view(src_main)
        if v is None:
            raise KeyError(f"no FLUX transformer found in {flux}")
        cfg = convert_flux(v, out / "transformer", opts, axes_dim)
        info["components"]["transformer"] = "transformer"
        info["variant"] = "dev" if cfg.guidance_embed else "schnell"
    if "t5" in components:
        v = t5_view(srcs["t5"])
        if v is None:
            raise KeyError("no T5 encoder found (pass --t5)")
        convert_t5(v, out / "t5", opts)
        info["components"]["t5"] = "t5"
    if "clip" in components:
        v = clip_view(srcs["clip"])
        if v is None:
            raise KeyError("no CLIP text encoder found (pass --clip)")
        convert_clip(v, out / "clip", opts)
        info["components"]["clip"] = "clip"
    if "vae" in components:
        v = vae_view(srcs["vae"])
        if v is None:
            raise KeyError("no FLUX VAE found (pass --vae)")
        convert_vae(v, out / "vae", opts)
        info["components"]["vae"] = "vae"

    all_srcs = list(flux) + list(t5 or []) + list(clip or [])
    t5_tok = Path(t5_tokenizer) if t5_tokenizer else _find_tokenizer(all_srcs, "tokenizer_2")
    clip_tok = Path(clip_tokenizer) if clip_tokenizer else _find_tokenizer(all_srcs, "tokenizer")
    for key, tok in (("t5", t5_tok), ("clip", clip_tok)):
        if tok is None:
            warn(f"no {key} tokenizer given (--{key}-tokenizer); generation will need one in {out / 'tokenizers' / key}")
            continue
        _copy_tokenizer(tok, out / "tokenizers" / key)
    info["tokenizers"] = {"t5": "tokenizers/t5", "clip": "tokenizers/clip"}

    prev = {}
    try:
        import json

        prev = json.loads((out / "model.json").read_text())
    except (OSError, ValueError):
        pass
    for k in ("components", "tokenizers"):
        info[k] = {**prev.get(k, {}), **info[k]}
    info.setdefault("variant", prev.get("variant", "dev"))
    dev = info["variant"] == "dev"
    info["defaults"] = {"steps": 28 if dev else 4, "guidance": 3.5 if dev else 0.0, "max_t5_tokens": 512 if dev else 256, "shift": dev}
    write_json_atomic(out / "model.json", info)
    log(f"wrote {out}")
    return out
