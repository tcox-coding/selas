"""Weight codecs: how a tensor is stored in a container and decoded at use time.

A codec turns a tensor into one or more *parts* (plain arrays written into the
container) plus a small JSON ``meta`` dict, and turns device views of those
parts back into a tensor in the compute dtype.

Decoding happens just in time, one Linear at a time, so a quantized unit is
never fully materialized in the compute dtype.

* ``raw``      — stored as-is in any dtype (fp32/fp16/bf16/fp8). Decoding to the
                 same dtype is a zero-copy view; otherwise a cast.
* ``int8_row`` — symmetric per-output-channel absmax int8 (weight-only).
* ``nf4``      — bitsandbytes-compatible NF4: blockwise (default 64) absmax in
                 fp32, two 4-bit codes per byte, first element in the high nibble.
                 Forge/bnb NF4 checkpoints import bit-exactly.
"""

from __future__ import annotations

import os
import threading
from typing import NamedTuple

import torch

from .util import warn

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
for _name in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _name):
        DTYPES[_name] = getattr(torch, _name)


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def to_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r} (torch {torch.__version__})") from None


# The NF4 code book (QLoRA / bitsandbytes), identical to the quant_map shipped
# in bnb NF4 checkpoints.
NF4_CODE: tuple[float, ...] = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


class EncodedTensor(NamedTuple):
    name: str
    codec: str
    shape: tuple[int, ...]
    meta: dict
    parts: dict[str, torch.Tensor]  # CPU, contiguous

    @property
    def nbytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parts.values())


# --------------------------------------------------------------------------- encode


def encode_raw(name: str, t: torch.Tensor, dtype: torch.dtype | None = None) -> EncodedTensor:
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return EncodedTensor(name, "raw", tuple(t.shape), {}, {"data": t.contiguous()})


def encode_int8_row(name: str, t: torch.Tensor) -> EncodedTensor:
    if t.dim() < 2:
        raise ValueError(f"int8_row needs a matrix, got {tuple(t.shape)} for {name}")
    w = t.float().reshape(t.shape[0], -1)
    absmax = w.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return EncodedTensor(
        name, "int8_row", tuple(t.shape), {}, {"q": q.reshape(t.shape).contiguous(), "scale": scale.contiguous()}
    )


def encode_nf4(name: str, t: torch.Tensor, blocksize: int = 64, code: tuple[float, ...] = NF4_CODE) -> EncodedTensor:
    flat = t.float().reshape(-1)
    n = flat.numel()
    if n % blocksize or blocksize % 2:
        raise ValueError(f"nf4 needs numel divisible by blocksize ({n} % {blocksize}) for {name}")
    blocks = flat.view(-1, blocksize)
    absmax = blocks.abs().amax(dim=1)
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normed = (blocks / safe[:, None]).reshape(-1)
    code_t = torch.tensor(code, dtype=torch.float32)
    mids = (code_t[1:] + code_t[:-1]) / 2
    # bucketize(right=False): mids[i-1] < x <= mids[i] -> i ; ties round to the lower code like bnb.
    idx = torch.bucketize(normed, mids).to(torch.uint8)
    packed = (idx[0::2] << 4) | idx[1::2]
    return EncodedTensor(
        name,
        "nf4",
        tuple(t.shape),
        {"blocksize": blocksize, "code": list(code)},
        {"packed": packed.contiguous(), "absmax": absmax.contiguous()},
    )


def nf4_import(
    name: str,
    packed: torch.Tensor,
    absmax: torch.Tensor,
    shape: tuple[int, ...],
    blocksize: int,
    code: list[float] | tuple[float, ...],
) -> EncodedTensor:
    """Wrap an existing bnb NF4 tensor without touching its bits."""
    n = 1
    for s in shape:
        n *= int(s)
    if n % blocksize:
        raise NotImplementedError(f"{name}: partial final NF4 block ({n} % {blocksize}) is not supported")
    packed = packed.reshape(-1).contiguous()
    if packed.dtype != torch.uint8 or packed.numel() * 2 != n:
        raise ValueError(f"{name}: packed NF4 data has {packed.numel()} bytes for {n} elements")
    absmax = absmax.reshape(-1).float().contiguous()
    if absmax.numel() != n // blocksize:
        raise ValueError(f"{name}: expected {n // blocksize} absmax values, got {absmax.numel()}")
    return EncodedTensor(
        name, "nf4", tuple(int(s) for s in shape), {"blocksize": int(blocksize), "code": [float(c) for c in code]},
        {"packed": packed, "absmax": absmax},
    )


# --------------------------------------------------------------------------- decode

_LUT_LOCK = threading.Lock()
_LUT_CACHE: dict[tuple, torch.Tensor] = {}
_CODE_CACHE: dict[tuple, torch.Tensor] = {}


def _nf4_lut(code: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """[256, 2] fp32 table: byte -> (code[hi nibble], code[lo nibble])."""
    key = (code, str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        c = torch.tensor(code, dtype=torch.float32)
        b = torch.arange(256)
        lut = torch.stack((c[b >> 4], c[b & 15]), dim=1).to(device)
        with _LUT_LOCK:
            _LUT_CACHE[key] = lut
    return lut


def _nf4_code(code: tuple[float, ...], device: torch.device) -> torch.Tensor:
    key = (code, str(device))
    t = _CODE_CACHE.get(key)
    if t is None:
        t = torch.tensor(code, dtype=torch.float32, device=device)
        with _LUT_LOCK:
            _CODE_CACHE[key] = t
    return t


def nf4_dequant_torch(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    code: tuple[float, ...],
    shape: tuple[int, ...],
    dtype: torch.dtype,
    chunk_bytes: int = 1 << 23,
) -> torch.Tensor:
    """Reference/fallback NF4 decode: value = code[nibble] * absmax[block] (fp32), then cast."""
    n_bytes = packed.numel()
    out = torch.empty(n_bytes * 2, dtype=dtype, device=packed.device)
    lut = _nf4_lut(code, packed.device)
    bpb = blocksize // 2  # bytes per block
    step = max(bpb, (chunk_bytes // bpb) * bpb)
    for s in range(0, n_bytes, step):
        e = min(n_bytes, s + step)
        vals = lut.index_select(0, packed[s:e].to(torch.int32))  # [k, 2] fp32
        vals = vals.view(-1, blocksize)
        vals.mul_(absmax[s // bpb : e // bpb].unsqueeze(1))
        out[2 * s : 2 * e].copy_(vals.view(-1))
    return out.view(shape)


# Optional Triton fast path ------------------------------------------------------

_TRITON = {"kernel": None, "checked": {}, "disabled": os.environ.get("SELAS_NO_TRITON", "") not in ("", "0")}


def _build_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _nf4_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_bytes, BYTES_PER_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_bytes
        byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
        scale = tl.load(absmax_ptr + offs // BYTES_PER_BLOCK, mask=mask, other=0.0)
        hi = tl.load(code_ptr + (byte >> 4), mask=mask, other=0.0) * scale
        lo = tl.load(code_ptr + (byte & 15), mask=mask, other=0.0) * scale
        tl.store(out_ptr + 2 * offs, hi.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + 2 * offs + 1, lo.to(out_ptr.dtype.element_ty), mask=mask)

    def run(packed, absmax, blocksize, code_t, n_bytes, out):
        BLOCK = 1024
        grid = (triton.cdiv(n_bytes, BLOCK),)
        _nf4_kernel[grid](packed, absmax, code_t, out, n_bytes, BYTES_PER_BLOCK=blocksize // 2, BLOCK=BLOCK)

    return run


def _triton_ready(device: torch.device) -> bool:
    if _TRITON["disabled"] or device.type != "cuda":
        return False
    key = str(device)
    ok = _TRITON["checked"].get(key)
    if ok is not None:
        return ok
    ok = False
    try:
        if _TRITON["kernel"] is None:
            _TRITON["kernel"] = _build_triton_kernel()
        # Self-check against the torch path; both compute fp32 products then cast,
        # so they must agree bit for bit.
        g = torch.Generator(device="cpu").manual_seed(0)
        packed = torch.randint(0, 256, (4096,), dtype=torch.uint8, generator=g).to(device)
        absmax = (torch.rand(128, generator=g) + 0.01).to(device)
        for dt in (torch.float16, torch.float32):
            ref = nf4_dequant_torch(packed, absmax, 64, NF4_CODE, (8192,), dt)
            out = torch.empty(8192, dtype=dt, device=device)
            _TRITON["kernel"](packed, absmax, 64, _nf4_code(NF4_CODE, device), 4096, out)
            if not torch.equal(ref, out):
                raise RuntimeError("triton NF4 kernel disagrees with reference")
        ok = True
    except Exception as e:  # pragma: no cover - depends on toolchain
        warn(f"Triton NF4 kernel unavailable ({type(e).__name__}: {e}); using the PyTorch fallback")
    _TRITON["checked"][key] = ok
    return ok


def nf4_dequant(packed, absmax, blocksize, code, shape, dtype) -> torch.Tensor:
    code = tuple(code)
    if packed.is_cuda and _triton_ready(packed.device):
        out = torch.empty(packed.numel() * 2, dtype=dtype, device=packed.device)
        _TRITON["kernel"](packed, absmax, blocksize, _nf4_code(code, packed.device), packed.numel(), out)
        return out.view(shape)
    return nf4_dequant_torch(packed, absmax, blocksize, code, shape, dtype)


def decode(codec: str, parts: dict[str, torch.Tensor], meta: dict, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if codec == "raw":
        x = parts["data"]
        return x if x.dtype == dtype else x.to(dtype)
    if codec == "int8_row":
        q, s = parts["q"], parts["scale"]
        w = q.to(dtype)
        w.mul_(s.to(dtype).view(-1, *([1] * (q.dim() - 1))))
        return w
    if codec == "nf4":
        return nf4_dequant(parts["packed"], parts["absmax"], int(meta["blocksize"]), meta["code"], tuple(shape), dtype)
    raise ValueError(f"unknown codec {codec!r}")


def decoded_is_view(codec: str, stored_dtype: str | None, dtype: torch.dtype) -> bool:
    return codec == "raw" and stored_dtype is not None and to_dtype(stored_dtype) == dtype
