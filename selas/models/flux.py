"""FLUX.1 (MMDiT) as plain functions over unit views, plus the streaming runner.

Weights are never module parameters: each block function receives a
:class:`~selas.store.UnitView` and reads its tensors as views into whatever
buffer the unit currently lives in (resident buffer or streaming arena).

Numerics (default, "exact" mode):
* matmuls and attention in the compute dtype (fp16 on Turing, bf16 on Ampere+);
* residual streams, LayerNorm, QK-RMSNorm, RoPE, modulation arithmetic in fp32;
* in fp16, Linear outputs that feed a residual are clamped to the fp16 range
  (the overflow guard diffusers applies to FLUX in fp16).

Runner-level optimizations, all exact:
* **modulation hoisting** — every block's adaLN Linear depends only on
  ``vec = f(t, guidance, pooled_clip)``; all steps' modulations are computed in
  one prologue pass so the 27 % of FLUX weights that are modulation layers are
  streamed once per image instead of once per step;
* ``txt_in``, RoPE tables and time/guidance embeddings computed once;
* block-major micro-batching: several images pass through a block while it is
  loaded, ``micro_batch`` at a time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..store import VRAM, UnitView, WeightStore

FP16_MAX = 65504.0


# --------------------------------------------------------------------------- config


@dataclass
class FluxConfig:
    hidden: int = 3072
    heads: int = 24
    mlp_hidden: int = 12288
    depth_double: int = 19
    depth_single: int = 38
    in_channels: int = 64
    context_dim: int = 4096
    vec_dim: int = 768
    axes_dim: tuple[int, ...] = (16, 56, 56)
    theta: float = 10000.0
    guidance_embed: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @classmethod
    def from_dict(cls, d: dict) -> "FluxConfig":
        keys = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in keys}
        if "axes_dim" in kw:
            kw["axes_dim"] = tuple(kw["axes_dim"])
        return cls(**kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes_dim"] = list(self.axes_dim)
        return d


def double_names(cfg: FluxConfig) -> list[str]:
    return [f"double.{i}" for i in range(cfg.depth_double)]


def single_names(cfg: FluxConfig) -> list[str]:
    return [f"single.{i}" for i in range(cfg.depth_single)]


def main_order(cfg: FluxConfig) -> list[str]:
    return double_names(cfg) + single_names(cfg)


def mod_order(cfg: FluxConfig) -> list[str]:
    return [n + ".mod" for n in main_order(cfg)]


# --------------------------------------------------------------------------- primitives


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0, time_factor: float = 1000.0) -> torch.Tensor:
    t = time_factor * t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def rope_tables(ids: torch.Tensor, axes_dim, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [L, head_dim/2] (fp32) for position ids [L, n_axes]; computed in fp64 like BFL."""
    angles = []
    for i, d in enumerate(axes_dim):
        scale = torch.arange(0, d, 2, dtype=torch.float64, device=ids.device) / d
        omega = 1.0 / (theta**scale)
        angles.append(ids[:, i].double()[:, None] * omega[None])
    ang = torch.cat(angles, dim=-1)
    return ang.cos().float(), ang.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs (x[2i], x[2i+1]); x fp32 [B, H, L, Dh]."""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


def qk_rms(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * scale


def _heads(qkv: torch.Tensor, heads: int):
    b, l, _ = qkv.shape
    x = qkv.view(b, l, 3, heads, -1).permute(2, 0, 3, 1, 4)
    return x[0], x[1], x[2]


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    b, h, l, d = q.shape
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, l, h * d)


def _res(y: torch.Tensor, guard: bool) -> torch.Tensor:
    y = y.float()
    return y.clamp_(-FP16_MAX, FP16_MAX) if guard else y


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    return (_ln(x) * (1 + scale) + shift).to(dt)


def mlp_embed(w: UnitView, prefix: str, x: torch.Tensor) -> torch.Tensor:
    return w.linear(F.silu(w.linear(x, prefix + ".in_layer")), prefix + ".out_layer")


# --------------------------------------------------------------------------- blocks


def double_block(w: UnitView, img, txt, mod, cos, sin, cfg: FluxConfig, dt, guard: bool):
    """img [B, Li, D] fp32, txt [B, Lt, D] fp32, mod [B, 12D] (img 6D | txt 6D)."""
    m = mod.float().unsqueeze(1).chunk(12, dim=-1)
    i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = m[:6]
    t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = m[6:]
    lt = txt.shape[1]
    f32 = torch.float32

    iq, ik, iv = _heads(w.linear(_modulate(img, i_sh1, i_sc1, dt), "img_attn.qkv"), cfg.heads)
    tq, tk, tv = _heads(w.linear(_modulate(txt, t_sh1, t_sc1, dt), "txt_attn.qkv"), cfg.heads)
    q = torch.cat((qk_rms(tq, w.get("txt_attn.norm.query_norm.scale", f32)), qk_rms(iq, w.get("img_attn.norm.query_norm.scale", f32))), dim=2)
    k = torch.cat((qk_rms(tk, w.get("txt_attn.norm.key_norm.scale", f32)), qk_rms(ik, w.get("img_attn.norm.key_norm.scale", f32))), dim=2)
    del iq, ik, tq, tk
    q = apply_rope(q, cos, sin).to(dt)
    k = apply_rope(k, cos, sin).to(dt)
    v = torch.cat((tv, iv), dim=2)
    attn = _attention(q, k, v)
    del q, k, v, iv, tv
    t_attn, i_attn = attn[:, :lt], attn[:, lt:]

    img = img + i_g1 * _res(w.linear(i_attn, "img_attn.proj"), guard)
    h = w.linear(_modulate(img, i_sh2, i_sc2, dt), "img_mlp.0")
    h = w.linear(F.gelu(h, approximate="tanh"), "img_mlp.2")
    img = img + i_g2 * _res(h, guard)

    txt = txt + t_g1 * _res(w.linear(t_attn, "txt_attn.proj"), guard)
    h = w.linear(_modulate(txt, t_sh2, t_sc2, dt), "txt_mlp.0")
    h = w.linear(F.gelu(h, approximate="tanh"), "txt_mlp.2")
    txt = txt + t_g2 * _res(h, guard)
    return img, txt


def single_block(w: UnitView, x, mod, cos, sin, cfg: FluxConfig, dt, guard: bool):
    """x [B, Lt+Li, D] fp32, mod [B, 3D]."""
    sh, sc, g = mod.float().unsqueeze(1).chunk(3, dim=-1)
    f32 = torch.float32
    h = w.linear(_modulate(x, sh, sc, dt), "linear1")
    qkv, mlp = h.split([3 * cfg.hidden, cfg.mlp_hidden], dim=-1)
    q, k, v = _heads(qkv, cfg.heads)
    q = apply_rope(qk_rms(q, w.get("norm.query_norm.scale", f32)), cos, sin).to(dt)
    k = apply_rope(qk_rms(k, w.get("norm.key_norm.scale", f32)), cos, sin).to(dt)
    # v is a strided view into linear1's output; keep SDPA on its memory-efficient kernel
    attn = _attention(q, k, v.contiguous())
    del q, k, v, qkv
    out = w.linear(torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=2), "linear2")
    del h, mlp, attn
    return x + g * _res(out, guard)


def final_layer(w: UnitView, img, fmod, dt):
    sh, sc = fmod.float().unsqueeze(1).chunk(2, dim=-1)
    return w.linear(_modulate(img, sh, sc, dt), "final_layer.linear").float()


# --------------------------------------------------------------------------- cost model


def unit_flops(kind: str, cfg: FluxConfig, l_img: int, l_txt: int, batch: int) -> tuple[float, float]:
    """(linear FLOPs, attention FLOPs) of one unit for one step."""
    d, m = cfg.hidden, cfg.mlp_hidden
    seq = l_img + l_txt
    attn = 4.0 * batch * seq * seq * d
    if kind == "double":  # each token passes through one stream's weights
        params = d * 3 * d + d * d + d * m + m * d
        return 2.0 * batch * seq * params, attn
    if kind == "single":
        params = d * (3 * d + m) + (d + m) * d
        return 2.0 * batch * seq * params, attn
    return 0.0, 0.0


def activation_reserve(cfg: FluxConfig, l_img: int, l_txt: int, batch: int, micro: int, decode_bytes: int) -> int:
    """Rough peak activation memory for one block forward plus persistent state."""
    d = cfg.hidden
    seq = l_img + l_txt
    per_tok = 56  # bytes per (token, channel) at the single-block peak (fp32 residual + fp16 temps + fp32 RoPE)
    block = micro * seq * d * per_tok
    state = batch * seq * d * 4 * 3  # img/txt/x fp32 + one in-flight copy
    return int(block + state + 2 * decode_bytes + 512 * 2**20)


# --------------------------------------------------------------------------- runner


@dataclass
class FluxConditioning:
    sigmas: list[float]
    mods: dict[str, torch.Tensor]  # main unit -> [S, B, k*D] (compute dtype)
    final_mod: torch.Tensor  # [S, B, 2D]
    txt: torch.Tensor  # [B, Lt, D] fp32 (txt_in output)
    cos: torch.Tensor
    sin: torch.Tensor
    l_txt: int


@dataclass
class StepInfo:
    decision: str = "full"
    distance: float | None = None


class FluxRunner:
    def __init__(self, store: WeightStore, cfg: FluxConfig, dtype: torch.dtype, micro_batch: int | None = None):
        self.store = store
        self.cfg = cfg
        self.dt = dtype
        self.guard = dtype == torch.float16
        self.g = store.view("globals")
        self.main = main_order(cfg)
        self.doubles = double_names(cfg)
        self.singles = single_names(cfg)
        self.mod_units = mod_order(cfg)
        self.micro_batch = micro_batch
        self._stream = None

    # ------------------------------------------------------------------ prologue
    @torch.no_grad()
    def prepare(self, sigmas, guidance: float, pooled: torch.Tensor, txt: torch.Tensor, img_ids: torch.Tensor, txt_ids: torch.Tensor) -> FluxConditioning:
        """Everything that does not depend on the latent: done once per image batch."""
        cfg, g, dt = self.cfg, self.g, self.dt
        dev = self.store.device
        b = pooled.shape[0]
        t = torch.tensor(list(sigmas[:-1]), dtype=torch.float32, device=dev)  # [S]
        vec = mlp_embed(g, "time_in", timestep_embedding(t, 256).to(dt)).float()[:, None, :]  # [S, 1, D]
        if cfg.guidance_embed:
            gv = torch.full((b,), float(guidance), dtype=torch.float32, device=dev)
            vec = vec + mlp_embed(g, "guidance_in", timestep_embedding(gv, 256).to(dt)).float()[None]
        vec = vec + mlp_embed(g, "vector_in", pooled.to(dev, dt)).float()[None]  # [S, B, D]
        svec = F.silu(vec).to(dt)

        mods: dict[str, torch.Tensor] = {}
        with self.store.stream(self.mod_units, cyclic=False) as st:
            for name in self.mod_units:
                w = st.acquire(name)
                if name.startswith("double."):
                    out = torch.cat((w.linear(svec, "img_mod.lin"), w.linear(svec, "txt_mod.lin")), dim=-1)
                else:
                    out = w.linear(svec, "modulation.lin")
                st.release(name)
                mods[name.removesuffix(".mod")] = out
        final_mod = g.linear(svec, "final_layer.adaLN_modulation.1")
        txt_h = g.linear(txt.to(dev, dt), "txt_in").float()
        ids = torch.cat((txt_ids, img_ids), dim=0).to(dev)
        cos, sin = rope_tables(ids, cfg.axes_dim, cfg.theta)
        return FluxConditioning(list(sigmas), mods, final_mod, txt_h, cos, sin, txt.shape[1])

    # ------------------------------------------------------------------ main loop
    def begin(self) -> None:
        if self._stream is None:
            self._stream = self.store.stream(self.main, cyclic=True)

    def end(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _timed(self, kind: str):
        if not self.store.profile or self.store.device.type != "cuda":
            return None
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        return kind, e0

    def _untimed(self, tok) -> None:
        if tok is None:
            return
        kind, e0 = tok
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        self.store.stats.compute_events.setdefault(kind, []).append((e0, e1))

    def _chunks(self, b: int):
        mb = self.micro_batch or b
        return [(a, min(b, a + mb)) for a in range(0, b, mb)]

    def _double(self, name: str, img, txt, cond: FluxConditioning, s: int):
        w = self._stream.acquire(name)
        tok = self._timed("double")
        mod = cond.mods[name][s]
        chunks = self._chunks(img.shape[0])
        if len(chunks) == 1:
            img, txt = double_block(w, img, txt, mod, cond.cos, cond.sin, self.cfg, self.dt, self.guard)
        else:
            outs = [double_block(w, img[a:e], txt[a:e], mod[a:e], cond.cos, cond.sin, self.cfg, self.dt, self.guard) for a, e in chunks]
            img = torch.cat([o[0] for o in outs])
            txt = torch.cat([o[1] for o in outs])
        self._untimed(tok)
        self._stream.release(name)
        return img, txt

    def _single(self, name: str, x, cond: FluxConditioning, s: int):
        w = self._stream.acquire(name)
        tok = self._timed("single")
        mod = cond.mods[name][s]
        chunks = self._chunks(x.shape[0])
        if len(chunks) == 1:
            x = single_block(w, x, mod, cond.cos, cond.sin, self.cfg, self.dt, self.guard)
        else:
            x = torch.cat([single_block(w, x[a:e], mod[a:e], cond.cos, cond.sin, self.cfg, self.dt, self.guard) for a, e in chunks])
        self._untimed(tok)
        self._stream.release(name)
        return x

    @torch.no_grad()
    def step(self, s: int, latents: torch.Tensor, cond: FluxConditioning, cache=None) -> tuple[torch.Tensor, StepInfo]:
        """Velocity prediction for step ``s``. latents: packed [B, Li, 64] fp32."""
        from ..stepcache import FULL, HYBRID, SKIP

        if self._stream is None:
            self.begin()
        dt = self.dt
        sigma = cond.sigmas[s]
        info = StepInfo()
        img = self.g.linear(latents.to(dt), "img_in").float()
        txt = cond.txt
        lt = cond.l_txt

        probe = self.doubles[0]
        if cache is not None and cache.active and self.is_streamed(probe):
            # a skipped step would consume the probe but not the units after it, breaking stream order
            raise ValueError(f"step caching needs the probe block {probe} to be VRAM-resident")
        img_in = img
        img, txt = self._double(probe, img, txt, cond, s)
        decision = FULL
        if cache is not None and cache.active:
            decision, info.distance = cache.decide(s, sigma, img - img_in)
        info.decision = decision
        del img_in

        if decision == SKIP:
            img = img + cache.tail_residual(sigma)
        else:
            img_probe = img
            hybrid = decision == HYBRID
            for name in self.doubles[1:]:
                if hybrid and cache.substitutes(name):
                    ri, rt = cache.block_residual(name, sigma)
                    img = img + ri
                    txt = txt + rt
                    continue
                i0, t0 = img, txt
                img, txt = self._double(name, img, txt, cond, s)
                if decision == FULL and cache is not None and cache.records(name):
                    cache.put_block(name, sigma, (img - i0, txt - t0))
            x = torch.cat((txt, img), dim=1)
            del txt, img
            for name in self.singles:
                if hybrid and cache.substitutes(name):
                    x = x + cache.block_residual(name, sigma)[0]
                    continue
                x0 = x
                x = self._single(name, x, cond, s)
                if decision == FULL and cache is not None and cache.records(name):
                    cache.put_block(name, sigma, (x - x0,))
            img = x[:, lt:]
            del x
            if decision == FULL and cache is not None and cache.active:
                cache.put_tail(sigma, img - img_probe)
        out = final_layer(self.g, img, cond.final_mod[s], dt)
        return out, info

    def is_streamed(self, name: str) -> bool:
        return self.store.tier[name] != VRAM
