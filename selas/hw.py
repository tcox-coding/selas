"""Hardware discovery, memory budgets and quick micro-benchmarks.

Measured numbers are cached in ``~/.cache/selas/hw.json`` so the planner has real
bandwidth/throughput figures without re-measuring on every run: GPU figures per
GPU, disk read bandwidth per block device (``disks``).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .util import GiB, MiB, human_bytes, log, meminfo, read_json, user_cache_dir, warn, write_json_atomic


@dataclass
class DeviceInfo:
    name: str
    index: int
    capability: tuple[int, int]
    total: int
    free: int

    @property
    def bf16_native(self) -> bool:
        return self.capability >= (8, 0)


@dataclass
class HwProfile:
    """Throughput figures the planner uses. Defaults are conservative guesses."""

    h2d_bw: float = 11e9  # bytes/s pinned host -> device
    disk_bw: float = 2.0e9  # bytes/s; fallback only — the engine measures the model's disk (disk_bandwidth)
    matmul_flops: float = 30e12  # sustained fp16/bf16 GEMM
    attn_flops: float = 18e12  # SDPA effective
    mem_bw: float = 400e9  # effective elementwise device bandwidth
    dequant_bw: float = 150e9  # decoded output bytes/s for quantized codecs
    measured: tuple[str, ...] = ()


def cuda_device(index: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("selas needs a CUDA GPU for generation (torch.cuda.is_available() is False)")
    return torch.device("cuda", torch.cuda.current_device() if index is None else index)


def device_info(device: torch.device) -> DeviceInfo:
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return DeviceInfo(props.name, device.index or 0, (props.major, props.minor), int(total), int(free))


def default_compute_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 0):
        return torch.bfloat16
    return torch.float16


def ram_available() -> tuple[int, int]:
    mi = meminfo()
    total = mi.get("MemTotal", 0)
    avail = mi.get("MemAvailable", mi.get("MemFree", 0))
    return total, avail


def ram_budget(requested_gb: float | None, margin: int = 4 * GiB) -> int:
    """Pinned-RAM budget: what the user asked for, never more than available - margin."""
    total, avail = ram_available()
    if total == 0:  # no /proc/meminfo: trust an explicit request, otherwise pin nothing
        return int(requested_gb * GiB) if requested_gb is not None else 0
    if requested_gb is not None:  # explicit request may dip into the safety margin, not below 1 GiB free
        return max(0, min(int(requested_gb * GiB), avail - 1 * GiB))
    return max(0, min(avail - margin, int(total * 0.85)))


def vram_budget(device: torch.device, requested_gb: float | None, margin: int = 384 * MiB) -> int:
    """VRAM selas may use (weights + arena + activations)."""
    free, _ = torch.cuda.mem_get_info(device)
    usable = max(0, free - margin)
    if requested_gb is not None:
        usable = min(usable, int(requested_gb * GiB))
    return usable


# ----------------------------------------------------------------------------- benchmarks


def _defaults_for(cap: tuple[int, int]) -> HwProfile:
    p = HwProfile()
    if cap >= (9, 0):
        p.matmul_flops, p.attn_flops, p.mem_bw = 400e12, 250e12, 2000e9
    elif cap >= (8, 9):
        p.matmul_flops, p.attn_flops, p.mem_bw = 90e12, 60e12, 700e9
    elif cap >= (8, 0):
        p.matmul_flops, p.attn_flops, p.mem_bw = 60e12, 40e12, 700e9
    return p


def measure_h2d(device: torch.device, nbytes: int = 256 * MiB, reps: int = 3) -> float:
    from .store import PinnedPool  # local import: store imports hw

    pool = PinnedPool(nbytes)
    try:
        src = pool.take(nbytes)
        dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
        stream = torch.cuda.Stream(device)
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
            stream.synchronize()
            best = max(best, nbytes / (time.perf_counter() - t0))
        del dst, src
        return best
    finally:
        pool.close()


def measure_matmul(device: torch.device, dtype: torch.dtype, n: int = 4096, reps: int = 5) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.mm(a, b)
    torch.cuda.synchronize(device)
    return 2 * n**3 * reps / (time.perf_counter() - t0)


def measure_attention(device: torch.device, dtype: torch.dtype, L: int = 4096, H: int = 24, D: int = 128, reps: int = 3) -> float:
    q = torch.randn(1, H, L, D, device=device, dtype=dtype)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize(device)
    return 4 * L * L * H * D * reps / (time.perf_counter() - t0)


def measure_disk(path: str, nbytes: int = 1 * GiB, direct: bool = False, cold: bool = True,
                 max_seconds: float | None = None) -> float:
    """Sequential read bandwidth of the device holding ``path``, in 256 MiB requests like the engine's.

    ``direct`` reads with O_DIRECT (raises OSError if refused); otherwise, if ``cold``,
    the range is evicted from the page cache first so the disk is measured, not RAM.
    An untimed read first wakes the device: an idle NVMe's first request is slow.
    Stops early after ``max_seconds``.
    """
    from .store import PinnedPool

    if direct and not hasattr(os, "O_DIRECT"):
        raise OSError("O_DIRECT is not available on this platform")
    size = os.path.getsize(path)
    chunk = 256 * MiB
    warm = min(64 * MiB, size // 2) // 4096 * 4096
    nbytes = min(nbytes, size - warm) // 4096 * 4096
    if nbytes <= 0:
        return 0.0
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
    pool = PinnedPool(max(min(chunk, nbytes), warm), pin=False)
    try:
        buf = memoryview(pool.take(pool.nbytes).numpy())
        if cold and not direct:
            os.posix_fadvise(fd, 0, warm + nbytes, os.POSIX_FADV_DONTNEED)
        if warm:
            os.preadv(fd, [buf[:warm]], 0)
        t0 = time.perf_counter()
        pos = 0
        while pos < nbytes:
            got = os.preadv(fd, [buf[: min(chunk, nbytes - pos)]], warm + pos)
            if got <= 0:
                break
            pos += got
            if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                break
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
        pool.close()
    return pos / dt if dt > 0 else 0.0


def disk_key(path: str | os.PathLike) -> str:
    """Name of the block device holding ``path`` (e.g. ``nvme0n1p3``), the disk-bandwidth cache key."""
    dev = os.stat(path).st_dev
    major, minor = os.major(dev), os.minor(dev)
    try:
        for line in Path(f"/sys/dev/block/{major}:{minor}/uevent").read_text().splitlines():
            if line.startswith("DEVNAME="):
                return line.split("=", 1)[1]
    except OSError:
        pass
    return f"dev{major}:{minor}"  # e.g. btrfs/overlay (anonymous device numbers)


def disk_bandwidth(path: str | os.PathLike, direct: bool = True, remeasure: bool = False) -> float | None:
    """Cold read bandwidth of the device holding ``path``, in the read mode the engine will use.

    Measured once per (device, mode) — at most 1 GiB or ~3 s of reads — and cached in
    ``hw.json``. If O_DIRECT is refused the buffered rate is measured and cached under the
    O_DIRECT key too, as the engine falls back the same way. None if it cannot be measured.
    """
    dev = disk_key(path)
    key = f"{dev}|{'direct' if direct else 'buffered'}"
    cache = user_cache_dir() / "hw.json"
    entry = (read_json(cache, {}) or {}).get("disks", {}).get(key)
    if entry and not remeasure:
        return float(entry["read_bw"])
    used_direct = direct
    try:
        try:
            bw = measure_disk(str(path), direct=direct, max_seconds=3.0)
        except OSError:
            if not direct:
                raise
            used_direct = False
            bw = measure_disk(str(path), direct=False, max_seconds=3.0)
    except OSError as e:
        warn(f"could not measure disk bandwidth for {path} ({e}); assuming {human_bytes(HwProfile.disk_bw)}/s")
        return None
    if bw <= 0:
        return None
    db = read_json(cache, {}) or {}  # re-read: keep entries other processes wrote meanwhile
    db.setdefault("disks", {})[key] = {"read_bw": bw, "o_direct": used_direct, "file": str(Path(path).resolve()),
                                       "date": time.strftime("%Y-%m-%d")}
    try:
        write_json_atomic(cache, db)
    except OSError:
        pass
    log(f"measured disk read {human_bytes(bw)}/s on {dev} ({'O_DIRECT' if used_direct else 'buffered, cold'}); cached")
    return bw


def _profile_key(dev: DeviceInfo) -> str:
    return f"{dev.name}|cc{dev.capability[0]}{dev.capability[1]}|torch{torch.__version__}"


def load_profile(device: torch.device, quick_measure: bool = True) -> HwProfile:
    """Cached profile for this GPU; measures H2D and GEMM throughput (~0.3 s) if missing."""
    dev = device_info(device)
    path = user_cache_dir() / "hw.json"
    db = read_json(path, {}) or {}
    entry = db.get(_profile_key(dev))
    prof = _defaults_for(dev.capability)
    if entry:
        for k, v in entry.items():
            # disk_bw is per disk now (disk_bandwidth); older `selas bench` runs stored it per GPU
            if hasattr(prof, k) and k not in ("measured", "disk_bw"):
                setattr(prof, k, float(v))
        prof.measured = tuple(m for m in entry.get("measured", ()) if m != "disk_bw")
    missing = [k for k in ("h2d_bw", "matmul_flops") if k not in prof.measured]
    if quick_measure and missing:
        dt = default_compute_dtype(device)
        try:
            if "h2d_bw" in missing:
                prof.h2d_bw = measure_h2d(device)
            if "matmul_flops" in missing:
                prof.matmul_flops = measure_matmul(device, dt)
                prof.attn_flops = measure_attention(device, dt)
            prof.measured = tuple(sorted(set(prof.measured) | {"h2d_bw", "matmul_flops", "attn_flops"}))
            save_profile(device, prof)
            log(f"measured H2D {human_bytes(prof.h2d_bw)}/s, GEMM {prof.matmul_flops / 1e12:.1f} TFLOP/s, "
                f"attention {prof.attn_flops / 1e12:.1f} TFLOP/s ({dt})")
        except Exception as e:  # pragma: no cover
            log(f"quick hardware measurement failed ({e}); using defaults")
    return prof


def save_profile(device: torch.device, prof: HwProfile) -> None:
    dev = device_info(device)
    path = user_cache_dir() / "hw.json"
    db = read_json(path, {}) or {}
    d = asdict(prof)
    d["measured"] = list(prof.measured)
    db[_profile_key(dev)] = d
    try:
        write_json_atomic(path, db)
    except OSError:
        pass
