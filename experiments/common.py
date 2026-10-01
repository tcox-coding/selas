"""Shared helpers for the experiment scripts.

Run from the repo root with the project venv, e.g.::

    .venv/bin/python -m experiments.placement --model models/flux1-dev

Each script writes a JSON record to ``experiments/results/`` and images (if any)
to ``outputs/experiments/``.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
IMAGES = ROOT / "outputs" / "experiments"

PROMPTS = [
    "a photo of a red fox sitting in fresh snow, golden hour",
    "a cozy reading nook with a large window, rain outside, warm lamp light, film photo",
    "a detailed ink illustration of a lighthouse on a cliff during a storm",
]


def embed(eng, prompts: list[str]) -> dict:
    """prompt -> (txt [L, 4096], pooled [768]) via the model's own encoders (prompt-cached)."""
    from selas.text import TextEncoders

    te = TextEncoders(eng.dir, eng.info, eng.device)
    try:
        return te.encode(prompts)
    finally:
        te.close()
        torch.cuda.empty_cache()


def psnr(a, b) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        return {"gpu": out[0], "vram_used_mib": int(out[1]), "temp_c": int(out[2]), "sm_mhz": int(out[3])}
    except Exception:
        return {}


def save_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {
        "experiment": name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        **gpu_state(),
        **data,
    }
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(rec, indent=1, default=str))
    print(f"wrote {path}")
    return path


def mean_after_first(xs: list[float]) -> float:
    xs = xs[1:] or xs
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------- page cache


def evict(path: str | os.PathLike) -> None:
    """Drop a file's clean pages from the page cache (no root needed)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def resident_fraction(path: str | os.PathLike, ranges: list[tuple[int, int]] | None = None) -> float:
    """Fraction of the file's pages (or of the given (offset, length) ranges) in the page cache (mincore)."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    page = mmap.PAGESIZE
    size = os.path.getsize(path)
    ranges = ranges or [(0, size)]
    hit = total = 0
    with open(path, "rb") as f:
        for off, length in ranges:
            start = off // page * page
            length += off - start
            mm = mmap.mmap(f.fileno(), length, access=mmap.ACCESS_COPY, offset=start)
            try:
                buf = (ctypes.c_char * length).from_buffer(mm)
                n = (length + page - 1) // page
                vec = (ctypes.c_ubyte * n)()
                if libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(length), vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
                del buf
                hit += int((np.frombuffer(vec, dtype=np.uint8) & 1).sum())
                total += n
            finally:
                mm.close()
    return hit / total if total else 0.0
