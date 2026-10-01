"""The selas container: one contiguous, aligned byte range per *unit*.

Layout of a container directory::

    manifest.json   # component, config, units -> tensors -> parts (offsets relative to the unit)
    weights.bin     # units back to back, each 4 KiB aligned and padded to 4 KiB

Loading a unit is a single ``pread`` and a single host->device copy; inside the
unit every tensor part starts on a 256-byte boundary so it can be viewed in
place with its own dtype (``uint8_buffer[off:off+n].view(dtype).view(shape)``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from .codecs import EncodedTensor, dtype_name, to_dtype
from .util import align_up, human_bytes, warn, write_json_atomic

FORMAT = "selas-container"
VERSION = 1
ALIGN = 4096
PART_ALIGN = 256
DATA_FILE = "weights.bin"
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class PartSpec:
    offset: int  # relative to the unit start
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class TensorSpec:
    name: str
    codec: str
    shape: tuple[int, ...]
    meta: dict
    parts: dict[str, PartSpec]

    @property
    def stored_dtype(self) -> str | None:
        p = self.parts.get("data")
        return p.dtype if p is not None else None

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts.values())


@dataclass
class UnitSpec:
    name: str
    offset: int  # absolute offset in weights.bin
    nbytes: int  # padded size
    tensors: dict[str, TensorSpec]
    attrs: dict = field(default_factory=dict)
    hash: str | None = None

    @property
    def group(self) -> str:
        return self.attrs.get("group", "main")

    @property
    def kind(self) -> str:
        return self.attrs.get("kind", self.group)


def part_view(buf: torch.Tensor, part: PartSpec) -> torch.Tensor:
    """View one part of a unit that lives in the 1-D uint8 tensor ``buf``."""
    raw = buf[part.offset : part.offset + part.nbytes]
    return raw.view(to_dtype(part.dtype)).view(part.shape)


class Container:
    def __init__(self, path: str | os.PathLike):
        self.dir = Path(path)
        mpath = self.dir / MANIFEST
        raw = mpath.read_bytes()
        m = json.loads(raw)
        if m.get("format") != FORMAT:
            raise ValueError(f"{self.dir} is not a selas container")
        if int(m.get("version", 0)) != VERSION:
            raise ValueError(f"{self.dir}: container version {m.get('version')} unsupported (want {VERSION})")
        self.manifest = m
        self.id = hashlib.sha256(raw).hexdigest()[:16]
        self.component: str = m["component"]
        self.config: dict = m.get("config", {})
        self.data_path = self.dir / m.get("data_file", DATA_FILE)
        self.units: dict[str, UnitSpec] = {}
        for u in m["units"]:
            tensors = {}
            for t in u["tensors"]:
                parts = {
                    k: PartSpec(int(p["offset"]), int(p["nbytes"]), p["dtype"], tuple(int(s) for s in p["shape"]))
                    for k, p in t["parts"].items()
                }
                tensors[t["name"]] = TensorSpec(t["name"], t["codec"], tuple(int(s) for s in t["shape"]), t.get("meta", {}), parts)
            self.units[u["name"]] = UnitSpec(
                u["name"], int(u["offset"]), int(u["nbytes"]), tensors, dict(u.get("attrs", {})), u.get("hash")
            )
        self._fd: int | None = None
        self._fd_direct: int | None = None
        self.direct_failed = False

    # ------------------------------------------------------------------ io
    def _get_fd(self, direct: bool) -> int:
        if direct and not self.direct_failed and hasattr(os, "O_DIRECT"):
            if self._fd_direct is None:
                try:
                    self._fd_direct = os.open(self.data_path, os.O_RDONLY | os.O_DIRECT)
                except OSError as e:
                    self.direct_failed = True
                    warn(f"O_DIRECT unavailable for {self.data_path} ({e}); using buffered reads")
            if self._fd_direct is not None:
                return self._fd_direct
        if self._fd is None:
            self._fd = os.open(self.data_path, os.O_RDONLY)
        return self._fd

    def read_unit_into(self, spec: UnitSpec, dst: torch.Tensor, direct: bool = False) -> int:
        """Read a whole unit into ``dst`` (1-D uint8 CPU tensor, >= spec.nbytes). Thread-safe."""
        if dst.dtype != torch.uint8 or dst.device.type != "cpu" or dst.numel() < spec.nbytes:
            raise ValueError("destination must be a uint8 CPU tensor large enough for the unit")
        mv = memoryview(dst.numpy())
        fd = self._get_fd(direct)
        n = spec.nbytes
        pos = 0
        chunk = 1 << 30
        while pos < n:
            want = min(chunk, n - pos)
            try:
                got = os.preadv(fd, [mv[pos : pos + want]], spec.offset + pos)
            except OSError:
                if fd == self._fd_direct:  # alignment or fs refused O_DIRECT: fall back for good
                    self.direct_failed = True
                    warn(f"O_DIRECT read failed in {self.data_path}; using buffered reads from now on")
                    fd = self._get_fd(False)
                    continue
                raise
            if got <= 0:
                raise IOError(f"short read in {self.data_path} at {spec.offset + pos}")
            pos += got
        return n

    def close(self) -> None:
        for name in ("_fd", "_fd_direct"):
            fd = getattr(self, name)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, None)

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ queries
    def total_bytes(self, group: str | None = None) -> int:
        return sum(u.nbytes for u in self.units.values() if group is None or u.group == group)

    def verify(self, names: Iterable[str] | None = None) -> list[str]:
        """Re-hash units; return the names that do not match their recorded hash."""
        bad = []
        names = list(names) if names is not None else list(self.units)
        buf = torch.empty(max(self.units[n].nbytes for n in names), dtype=torch.uint8)
        for n in names:
            u = self.units[n]
            if not u.hash:
                continue
            self.read_unit_into(u, buf)
            if hashlib.blake2b(buf[: u.nbytes].numpy(), digest_size=16).hexdigest() != u.hash:
                bad.append(n)
        return bad

    def describe(self) -> str:
        groups: dict[str, int] = {}
        for u in self.units.values():
            groups[u.group] = groups.get(u.group, 0) + u.nbytes
        g = ", ".join(f"{k} {human_bytes(v)}" for k, v in groups.items())
        return f"{self.component}: {len(self.units)} units, {human_bytes(self.total_bytes())} ({g})"


class ContainerWriter:
    """Streams units into a new container without holding the model in memory."""

    def __init__(self, out_dir: str | os.PathLike, component: str, config: dict, hash_units: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.component = component
        self.config = dict(config)
        self.hash_units = hash_units
        self._tmp = self.dir / (DATA_FILE + ".partial")
        self._f = open(self._tmp, "wb")
        self._offset = 0
        self._units: list[dict] = []
        self._names: set[str] = set()

    def add_unit(self, name: str, tensors: list[EncodedTensor], **attrs) -> int:
        if name in self._names:
            raise ValueError(f"duplicate unit {name}")
        self._names.add(name)
        cursor = 0
        tspecs = []
        chunks: list[tuple[int, torch.Tensor]] = []
        seen = set()
        for et in tensors:
            if et.name in seen:
                raise ValueError(f"duplicate tensor {et.name} in unit {name}")
            seen.add(et.name)
            parts = {}
            for pname, arr in et.parts.items():
                arr = arr.detach().contiguous().cpu()
                nbytes = arr.numel() * arr.element_size()
                cursor = align_up(cursor, PART_ALIGN)
                parts[pname] = {"offset": cursor, "nbytes": nbytes, "dtype": dtype_name(arr.dtype), "shape": list(arr.shape)}
                chunks.append((cursor, arr))
                cursor += nbytes
            tspecs.append({"name": et.name, "codec": et.codec, "shape": list(et.shape), "meta": et.meta, "parts": parts})
        size = align_up(max(cursor, 1), ALIGN)
        h = hashlib.blake2b(digest_size=16) if self.hash_units else None
        pos = 0
        for off, arr in chunks:
            if off > pos:
                pad = bytes(off - pos)
                self._f.write(pad)
                if h:
                    h.update(pad)
                pos = off
            b = arr.reshape(-1).view(torch.uint8).numpy()
            self._f.write(memoryview(b))
            if h:
                h.update(b)
            pos += b.nbytes
        if size > pos:
            pad = bytes(size - pos)
            self._f.write(pad)
            if h:
                h.update(pad)
        self._units.append(
            {"name": name, "offset": self._offset, "nbytes": size, "attrs": attrs, "hash": h.hexdigest() if h else None, "tensors": tspecs}
        )
        self._offset += size
        return size

    def close(self, extra: dict | None = None) -> Path:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        os.replace(self._tmp, self.dir / DATA_FILE)
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "component": self.component,
            "config": self.config,
            "data_file": DATA_FILE,
            "align": ALIGN,
            "units": self._units,
        }
        if extra:
            manifest["extra"] = extra
        write_json_atomic(self.dir / MANIFEST, manifest)
        return self.dir

    def abort(self) -> None:
        try:
            self._f.close()
        finally:
            try:
                os.unlink(self._tmp)
            except OSError:
                pass
