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

Kernel fusion (``Kernels(compiled=True)``, the default on CUDA) runs the
elementwise code between matmuls as ``torch.compile``-generated kernels: the same
fp32 math, not bitwise equal to eager. Placements stay bit-identical to each
other within either mode.
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


def mlp_embed(w: UnitView, prefix: str, x: torch.Tensor) -> torch.Tensor:
    return w.linear(F.silu(w.linear(x, prefix + ".in_layer")), prefix + ".out_layer")


# --------------------------------------------------------------------------- kernels
#
# Everything between the matmuls and the attention is elementwise or a small
# per-row reduction, and memory-bound: in eager PyTorch it was ~25 % of a 1024²
# step on an RTX 2080 Ti. The blocks are therefore written as matmuls/attention
# plus a few elementwise *segments*; ``Kernels`` holds the segments, either as the
# plain eager functions below or fused by ``torch.compile`` (one Triton kernel per
# segment, same fp32 math, results differ from eager only by rounding).


def _mod_in(x, shift, scale, dt):
    """LayerNorm + modulation, cast for the next matmul."""
    return (_ln(x) * (1 + scale) + shift).to(dt)


def _gated_add(x, g, y, guard: bool):
    """Gated residual update."""
    return x + g * _res(y, guard)


def _gated_add_mod(x, g, y, shift, scale, dt, guard: bool):
    x = _gated_add(x, g, y, guard)
    return x, _mod_in(x, shift, scale, dt)


def _qkv_double(tqkv, iqkv, tqs, tks, iqs, iks, cos, sin, heads: int, dt):
    """Split heads, QK-RMSNorm per stream, join txt+img, RoPE: (q, k, v) for attention."""
    tq, tk, tv = _heads(tqkv, heads)
    iq, ik, iv = _heads(iqkv, heads)
    q = apply_rope(torch.cat((qk_rms(tq, tqs), qk_rms(iq, iqs)), dim=2), cos, sin).to(dt)
    k = apply_rope(torch.cat((qk_rms(tk, tks), qk_rms(ik, iks)), dim=2), cos, sin).to(dt)
    return q, k, torch.cat((tv, iv), dim=2)


def _qkv_single(qkv, qs, ks, cos, sin, heads: int, dt):
    q, k, v = _heads(qkv, heads)
    # v is a strided view into linear1's output; keep SDPA on its memory-efficient kernel
    return apply_rope(qk_rms(q, qs), cos, sin).to(dt), apply_rope(qk_rms(k, ks), cos, sin).to(dt), v.contiguous()


def _gelu(h):
    return F.gelu(h, approximate="tanh")


def _attn_mlp(attn, mlp):
    """Single block: attention output and activated MLP branch, side by side for linear2."""
    return torch.cat((attn, _gelu(mlp)), dim=2)


SEGMENTS = (_mod_in, _gated_add, _gated_add_mod, _qkv_double, _qkv_single, _gelu, _attn_mlp)


class Kernels:
    """The elementwise segments of a block, eager or compiled."""

    def __init__(self, compiled: bool = False):
        self.compiled = compiled
        for fn in SEGMENTS:
            setattr(self, fn.__name__.lstrip("_"), _compile(fn) if compiled else fn)


def _compile(fn):
    """``torch.compile(fn)``, falling back to ``fn`` for good if compilation fails."""
    from ..util import warn

    state = {"fn": torch.compile(fn, dynamic=False, fullgraph=True)}

    def call(*args):
        try:
            return state["fn"](*args)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            if state["fn"] is fn:
                raise
            warn(f"torch.compile failed for {fn.__name__} ({type(e).__name__}: {str(e).splitlines()[0][:200]}); "
                 "using eager kernels for it")
            state["fn"] = fn
            return fn(*args)

    call.__name__ = fn.__name__
    return call


def compile_count() -> int:
    """Frames torch.compile has compiled in this process so far (0 if the counter is unavailable)."""
    try:
        from torch._dynamo.utils import counters

        return int(counters["stats"]["unique_graphs"])
    except Exception:  # pragma: no cover
        return 0


def compile_available(device: torch.device) -> bool:
    """torch.compile's GPU backend needs Triton (and a CUDA device)."""
    if torch.device(device).type != "cuda":
        return False
    try:
        from torch.utils._triton import has_triton

        return has_triton()
    except Exception:  # pragma: no cover
        return False


EAGER = Kernels()


class _ShapeOnly:
    """Stands in for a :class:`UnitView`: right output shapes, no weights, no matmuls."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], dt: torch.dtype, device: torch.device):
        self.shapes, self.dt, self.device = shapes, dt, device

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(self.shapes[name], dtype=dtype or self.dt, device=self.device)

    def linear(self, x: torch.Tensor, prefix: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(*x.shape[:-1], self.shapes[prefix + ".weight"][0], dtype=x.dtype, device=x.device)


@torch.no_grad()
def warm_kernels(k: Kernels, cfg: FluxConfig, shapes: dict[str, dict[str, tuple]], l_img: int, l_txt: int, batch: int,
                 dt: torch.dtype, device: torch.device) -> None:
    """Compile ``k``'s segments for these shapes by running the blocks once on zeros.

    The blocks' own code makes the calls, so arguments match the real run exactly
    (shapes, dtypes, strides) and the real run hits the compiled code. Meant for a
    background thread while the weights load: compiling is CPU work, loading is I/O.
    """
    guard = dt == torch.float16
    z = lambda *s, d=torch.float32: torch.zeros(*s, dtype=d, device=device)  # noqa: E731
    d, lt = cfg.hidden, l_txt
    img, txt = z(batch, l_img, d), z(batch, lt, d)
    cos, sin = z(lt + l_img, cfg.head_dim // 2), z(lt + l_img, cfg.head_dim // 2)  # distinct: guards check identity
    double_block(_ShapeOnly(shapes["double"], dt, device), img, txt, z(batch, 12 * d, d=dt), cos, sin, cfg, dt, guard, k)
    x = torch.cat((txt, img), 1)
    single_block(_ShapeOnly(shapes["single"], dt, device), x, z(batch, 3 * d, d=dt), cos, sin, cfg, dt, guard, k)
    final_layer(_ShapeOnly(shapes["globals"], dt, device), x[:, lt:], z(batch, 2 * d, d=dt), dt, k)
    torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- blocks


def double_block(w: UnitView, img, txt, mod, cos, sin, cfg: FluxConfig, dt, guard: bool, k: Kernels = EAGER):
    """img [B, Li, D] fp32, txt [B, Lt, D] fp32, mod [B, 12D] (img 6D | txt 6D)."""
    m = mod.float().unsqueeze(1).chunk(12, dim=-1)
    i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = m[:6]
    t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = m[6:]
    lt = txt.shape[1]
    f32 = torch.float32

    iqkv = w.linear(k.mod_in(img, i_sh1, i_sc1, dt), "img_attn.qkv")
    tqkv = w.linear(k.mod_in(txt, t_sh1, t_sc1, dt), "txt_attn.qkv")
    q, kk, v = k.qkv_double(tqkv, iqkv, w.get("txt_attn.norm.query_norm.scale", f32), w.get("txt_attn.norm.key_norm.scale", f32),
                            w.get("img_attn.norm.query_norm.scale", f32), w.get("img_attn.norm.key_norm.scale", f32),
                            cos, sin, cfg.heads, dt)
    del iqkv, tqkv
    attn = _attention(q, kk, v)
    del q, kk, v
    t_attn, i_attn = attn[:, :lt], attn[:, lt:]

    img, h = k.gated_add_mod(img, i_g1, w.linear(i_attn, "img_attn.proj"), i_sh2, i_sc2, dt, guard)
    h = w.linear(h, "img_mlp.0")
    img = k.gated_add(img, i_g2, w.linear(k.gelu(h), "img_mlp.2"), guard)
    txt, h = k.gated_add_mod(txt, t_g1, w.linear(t_attn, "txt_attn.proj"), t_sh2, t_sc2, dt, guard)
    h = w.linear(h, "txt_mlp.0")
    txt = k.gated_add(txt, t_g2, w.linear(k.gelu(h), "txt_mlp.2"), guard)
    return img, txt


def single_block(w: UnitView, x, mod, cos, sin, cfg: FluxConfig, dt, guard: bool, k: Kernels = EAGER):
    """x [B, Lt+Li, D] fp32, mod [B, 3D]."""
    sh, sc, g = mod.float().unsqueeze(1).chunk(3, dim=-1)
    f32 = torch.float32
    h = w.linear(k.mod_in(x, sh, sc, dt), "linear1")
    qkv, mlp = h.split([3 * cfg.hidden, cfg.mlp_hidden], dim=-1)
    q, kk, v = k.qkv_single(qkv, w.get("norm.query_norm.scale", f32), w.get("norm.key_norm.scale", f32), cos, sin, cfg.heads, dt)
    attn = _attention(q, kk, v)
    del q, kk, v, qkv
    out = w.linear(k.attn_mlp(attn, mlp), "linear2")
    del h, mlp, attn
    return k.gated_add(x, g, out, guard)


def final_layer(w: UnitView, img, fmod, dt, k: Kernels = EAGER):
    sh, sc = fmod.float().unsqueeze(1).chunk(2, dim=-1)
    return w.linear(k.mod_in(img, sh, sc, dt), "final_layer.linear").float()


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
    def __init__(self, store: WeightStore, cfg: FluxConfig, dtype: torch.dtype, micro_batch: int | None = None,
                 kernels: Kernels = EAGER):
        self.store = store
        self.k = kernels
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
    def svec(self, sigmas, guidance: float, pooled: torch.Tensor) -> torch.Tensor:
        """silu(vec) [S, 1, D] for one image (``pooled`` [1, 768]): timestep + guidance + CLIP embeddings.

        Resident weights only, so cheap. Computed per image, never batched, so an image's
        modulation vectors are the same bits whatever batch it is in (see :mod:`selas.modcache`).
        """
        cfg, g, dt = self.cfg, self.g, self.dt
        dev = self.store.device
        t = torch.tensor(list(sigmas[:-1]), dtype=torch.float32, device=dev)  # [S]
        vec = mlp_embed(g, "time_in", timestep_embedding(t, 256).to(dt)).float()[:, None, :]  # [S, 1, D]
        if cfg.guidance_embed:
            gv = torch.full((1,), float(guidance), dtype=torch.float32, device=dev)
            vec = vec + mlp_embed(g, "guidance_in", timestep_embedding(gv, 256).to(dt)).float()[None]
        vec = vec + mlp_embed(g, "vector_in", pooled.to(dev, dt)).float()[None]
        return F.silu(vec).to(dt)

    @torch.no_grad()
    def modulations(self, svecs: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Every main unit's modulation vectors [S, 1, kD], for each image's ``svec``.

        The one part of the prologue that needs the modulation units (27 % of the
        weights): they are streamed once, however many images there are.
        """
        outs: list[dict[str, torch.Tensor]] = [{} for _ in svecs]
        with self.store.stream(self.mod_units, cyclic=False) as st:
            for name in self.mod_units:
                w = st.acquire(name)
                main = name.removesuffix(".mod")
                for out, sv in zip(outs, svecs):
                    if name.startswith("double."):
                        out[main] = torch.cat((w.linear(sv, "img_mod.lin"), w.linear(sv, "txt_mod.lin")), dim=-1)
                    else:
                        out[main] = w.linear(sv, "modulation.lin")
                st.release(name)
        return outs

    @torch.no_grad()
    def prepare(self, sigmas, guidance: float, pooled: torch.Tensor, txt: torch.Tensor, img_ids: torch.Tensor,
                txt_ids: torch.Tensor, mods: dict[str, torch.Tensor] | None = None) -> FluxConditioning:
        """Everything that does not depend on the latent: done once per image batch.

        ``mods``: the batch's modulation vectors ({main unit: [S, B, kD]}) if already
        known, e.g. from the cache; otherwise they are computed here.
        """
        cfg, g, dt = self.cfg, self.g, self.dt
        dev = self.store.device
        svecs = [self.svec(sigmas, guidance, pooled[i : i + 1]) for i in range(pooled.shape[0])]
        if mods is None:
            per_image = self.modulations(svecs)
            mods = {n: torch.cat([m[n] for m in per_image], dim=1) for n in self.main}
        final_mod = torch.cat([g.linear(sv, "final_layer.adaLN_modulation.1") for sv in svecs], dim=1)
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
            img, txt = double_block(w, img, txt, mod, cond.cos, cond.sin, self.cfg, self.dt, self.guard, self.k)
        else:
            outs = [double_block(w, img[a:e], txt[a:e], mod[a:e], cond.cos, cond.sin, self.cfg, self.dt, self.guard, self.k)
                    for a, e in chunks]
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
            x = single_block(w, x, mod, cond.cos, cond.sin, self.cfg, self.dt, self.guard, self.k)
        else:
            x = torch.cat([single_block(w, x[a:e], mod[a:e], cond.cos, cond.sin, self.cfg, self.dt, self.guard, self.k)
                           for a, e in chunks])
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
        out = final_layer(self.g, img, cond.final_mod[s], dt, self.k)
        return out, info

    def is_streamed(self, name: str) -> bool:
        return self.store.tier[name] != VRAM
