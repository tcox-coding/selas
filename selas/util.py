"""Small shared helpers: logging, byte formatting, alignment, /proc/meminfo, JSON io."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

KiB = 1024
MiB = 1024**2
GiB = 1024**3

_T0 = time.perf_counter()
VERBOSITY = 1


def log(msg: str, level: int = 1) -> None:
    if level <= VERBOSITY:
        print(f"[selas {time.perf_counter() - _T0:7.2f}s] {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    log(f"warning: {msg}", 0)


def human_bytes(n: float) -> str:
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{sign}{n:.0f} {unit}" if unit == "B" else f"{sign}{n:.2f} {unit}"
        n /= 1024
    raise AssertionError("unreachable")


def align_up(n: int, a: int) -> int:
    return (int(n) + a - 1) // a * a


def meminfo() -> dict[str, int]:
    """Return /proc/meminfo in bytes (empty dict where unavailable)."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                val = int(parts[0])
                if len(parts) > 1 and parts[1] == "kB":
                    val *= 1024
                out[key] = val
    except OSError:
        pass
    return out


def sha256_hex(*items: Any) -> str:
    h = hashlib.sha256()
    for it in items:
        if isinstance(it, bytes):
            h.update(it)
        else:
            h.update(repr(it).encode())
        h.update(b"\0")
    return h.hexdigest()


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json_atomic(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def user_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "selas"


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"
