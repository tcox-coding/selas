"""Tiered weight store and the streaming executor.

Every unit of a container lives in exactly one tier:

* ``vram`` — loaded once into its own device buffer; access is free.
* ``host`` — page-locked RAM; streamed to the GPU by DMA on a copy stream.
* ``disk`` — the container on NVMe; a reader thread ``pread``s it into a pinned
  staging ring ahead of time, then it is streamed like a host unit.

Non-resident units flow through a VRAM *ring arena* (see :mod:`selas.arena`)
in a fixed, known order (:class:`UnitStream`). Because the order is known in
advance — FLUX runs the same 57 blocks every step — prefetching is exact and
continues across step boundaries.

Synchronization is GPU-side only on the hot path:

* copy stream records ``ready`` after a unit's H2D copy; the compute stream
  waits on it in :meth:`UnitStream.acquire`;
* the compute stream records ``done`` after the last kernel that read a unit
  (:meth:`UnitStream.release`); the copy stream waits on it before reusing the
  arena region.

Placement never changes numerics: a unit's bytes are identical in every tier
and the compute code receives the same views either way.
"""

from __future__ import annotations

import gc
import mmap
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .arena import RingArena, RingEntry
from .codecs import decode
from .container import Container, UnitSpec, part_view
from .util import align_up, human_bytes, log, warn

VRAM, HOST, DISK = "vram", "host", "disk"
TIERS = (VRAM, HOST, DISK)


# --------------------------------------------------------------------------- pinned host memory


def _cuda_ok(err) -> bool:
    try:
        return int(err) == 0
    except Exception:  # pragma: no cover
        return str(err).endswith("success")


class PinnedPool:
    """One page-aligned anonymous mapping, page-locked with ``cudaHostRegister``.

    PyTorch's pinned allocator rounds every allocation up to a power of two, which
    would waste up to 2x RAM for multi-GiB weight sets; registering our own mapping
    pins exactly what we use. Slices are handed out bump-allocator style.
    """

    def __init__(self, nbytes: int, pin: bool = True):
        self.nbytes = align_up(max(int(nbytes), 4096), 4096)
        self._mm = mmap.mmap(-1, self.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.base = torch.frombuffer(self._mm, dtype=torch.uint8)
        self.pinned = False
        if pin and torch.cuda.is_available():
            try:
                err = torch.cuda.cudart().cudaHostRegister(self.base.data_ptr(), self.nbytes, 0)
                self.pinned = _cuda_ok(err)
            except Exception as e:  # pragma: no cover
                warn(f"cudaHostRegister unavailable ({e})")
            if not self.pinned:
                warn(f"could not page-lock {human_bytes(self.nbytes)} of host memory; host->device copies will be slower")
        self._cursor = 0

    def take(self, n: int, align: int = 4096) -> torch.Tensor:
        off = align_up(self._cursor, align)
        end = off + int(n)
        if end > self.nbytes:
            raise MemoryError(f"pinned pool exhausted ({human_bytes(end)} > {human_bytes(self.nbytes)})")
        self._cursor = end
        return self.base[off:end]

    @property
    def used(self) -> int:
        return self._cursor

    def close(self) -> None:
        if self.base is None:
            return
        if self.pinned:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.base.data_ptr())
            except Exception:  # pragma: no cover
                pass
            self.pinned = False
        self.base = None
        gc.collect()
        try:
            self._mm.close()
        except BufferError:  # a slice is still referenced; the mapping is freed when it is collected
            pass


# --------------------------------------------------------------------------- views


class UnitView:
    """Typed access to the tensors of one unit living in a uint8 device buffer."""

    __slots__ = ("spec", "buf", "dtype", "_parts", "_views")

    def __init__(self, spec: UnitSpec, buf: torch.Tensor, dtype: torch.dtype):
        self.spec = spec
        self.buf = buf
        self.dtype = dtype
        self._parts: dict[str, dict[str, torch.Tensor]] = {}
        self._views: dict[tuple, torch.Tensor] = {}

    def has(self, name: str) -> bool:
        return name in self.spec.tensors

    def parts(self, name: str) -> dict[str, torch.Tensor]:
        p = self._parts.get(name)
        if p is None:
            ts = self.spec.tensors[name]
            p = {k: part_view(self.buf, ps) for k, ps in ts.parts.items()}
            self._parts[name] = p
        return p

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Decoded tensor in ``dtype`` (default: the compute dtype).

        Zero-copy for raw tensors already in that dtype; otherwise a fresh
        temporary that dies with the caller's reference (JIT decode).
        """
        dtype = dtype or self.dtype
        key = (name, dtype)
        hit = self._views.get(key)
        if hit is not None:
            return hit
        ts = self.spec.tensors[name]
        parts = self.parts(name)
        out = decode(ts.codec, parts, ts.meta, ts.shape, dtype)
        if ts.codec == "raw" and out.data_ptr() == parts["data"].data_ptr():
            self._views[key] = out
        return out

    def raw(self, name: str, part: str = "data") -> torch.Tensor:
        return self.parts(name)[part]

    def codec(self, name: str) -> str:
        return self.spec.tensors[name].codec

    def linear(self, x: torch.Tensor, prefix: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        w = self.get(prefix + ".weight", dtype)
        bname = prefix + ".bias"
        b = self.get(bname, dtype) if bname in self.spec.tensors else None
        return F.linear(x, w, b)


# --------------------------------------------------------------------------- stats


@dataclass
class StreamStats:
    h2d_bytes: dict = field(default_factory=lambda: {HOST: 0, DISK: 0})
    disk_read_bytes: int = 0
    units_streamed: int = 0
    stall_events: list = field(default_factory=list)
    compute_events: dict = field(default_factory=dict)  # kind -> list[(start, end)]

    def reset(self) -> None:
        self.h2d_bytes = {HOST: 0, DISK: 0}
        self.disk_read_bytes = 0
        self.units_streamed = 0
        self.stall_events = []
        self.compute_events = {}

    def stall_seconds(self) -> float:
        tot = 0.0
        for a, b in self.stall_events:
            b.synchronize()
            tot += a.elapsed_time(b) / 1e3
        return tot

    def compute_seconds(self) -> dict[str, tuple[float, int]]:
        out = {}
        for kind, evs in self.compute_events.items():
            tot = 0.0
            for a, b in evs:
                b.synchronize()
                tot += a.elapsed_time(b) / 1e3
            out[kind] = (tot, len(evs))
        return out


# --------------------------------------------------------------------------- store


class WeightStore:
    """Owns one container's units across the VRAM / host / disk tiers."""

    def __init__(
        self,
        container: Container,
        tiers: dict[str, str],
        device: torch.device,
        dtype: torch.dtype,
        arena_bytes: int = 0,
        staging_bytes: int = 0,
        direct_io: bool = False,
        profile: bool = False,
        label: str = "",
    ):
        self.c = container
        self.units = container.units
        self.device = torch.device(device)
        self.dtype = dtype
        self.direct_io = direct_io
        self.profile = profile
        self.label = label or container.component
        self.stats = StreamStats()
        self._active: UnitStream | None = None
        self.tier: dict[str, str] = {}
        for name in self.units:
            t = tiers.get(name, DISK)
            if t not in TIERS:
                raise ValueError(f"bad tier {t!r} for {name}")
            self.tier[name] = t
        if self.device.type != "cuda":  # CPU execution (tests): everything is "resident"
            self.tier = {n: VRAM for n in self.units}

        vram_units = [n for n, t in self.tier.items() if t == VRAM]
        host_units = [n for n, t in self.tier.items() if t == HOST]
        disk_units = [n for n, t in self.tier.items() if t == DISK]
        if disk_units:
            staging_bytes = max(staging_bytes, 2 * max(self.units[n].nbytes for n in disk_units))
        else:
            staging_bytes = 0
        self.staging_bytes = staging_bytes
        load_bytes = max((self.units[n].nbytes for n in vram_units), default=0) if self.device.type == "cuda" else 0
        shared = max(staging_bytes, load_bytes)
        host_bytes = sum(align_up(self.units[n].nbytes, 4096) for n in host_units)

        self.pool: PinnedPool | None = None
        self.host: dict[str, torch.Tensor] = {}
        self.views: dict[str, UnitView] = {}
        self.staging: torch.Tensor | None = None
        self.arena_buf: torch.Tensor | None = None
        self.arena: RingArena | None = None
        self.copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

        t0 = time.perf_counter()
        try:
            self._load(vram_units, host_units, disk_units, host_bytes, shared, arena_bytes)
        except BaseException:
            self.close()  # never leak page-locked memory or device buffers on a failed load
            raise

        vb = sum(self.units[n].nbytes for n in vram_units)
        db = sum(self.units[n].nbytes for n in disk_units)
        log(
            f"{self.label}: vram {len(vram_units)} units {human_bytes(vb)}, host {len(host_units)} units "
            f"{human_bytes(host_bytes)}, disk {len(disk_units)} units {human_bytes(db)}, "
            f"arena {human_bytes(self.arena_buf.numel() if self.arena_buf is not None else 0)}, "
            f"staging {human_bytes(self.staging_bytes)} (loaded in {time.perf_counter() - t0:.1f}s)"
        )

    def _load(self, vram_units, host_units, disk_units, host_bytes: int, shared: int, arena_bytes: int) -> None:
        direct_io, dtype, staging_bytes = self.direct_io, self.dtype, self.staging_bytes
        if self.device.type == "cuda" and (host_bytes or shared):
            self.pool = PinnedPool(host_bytes + shared)
            for n in host_units:
                buf = self.pool.take(self.units[n].nbytes)
                self.c.read_unit_into(self.units[n], buf, direct=direct_io)
                self.host[n] = buf
            if shared:
                scratch = self.pool.take(shared)
                self.staging = scratch[:staging_bytes] if staging_bytes else None
            else:
                scratch = None
        else:
            scratch = None

        for n in vram_units:
            spec = self.units[n]
            if self.device.type == "cuda":
                assert scratch is not None
                self.c.read_unit_into(spec, scratch, direct=direct_io)
                dev = torch.empty(spec.nbytes, dtype=torch.uint8, device=self.device)
                dev.copy_(scratch[: spec.nbytes])  # synchronous: scratch is reused next iteration
            else:
                dev = torch.empty(spec.nbytes, dtype=torch.uint8)
                self.c.read_unit_into(spec, dev)
            self.views[n] = UnitView(spec, dev, dtype)

        if arena_bytes and self.device.type == "cuda" and (host_units or disk_units):
            max_streamed = max(self.units[n].nbytes for n in host_units + disk_units)
            arena_bytes = max(int(arena_bytes), max_streamed)
            self.arena_buf = torch.empty(arena_bytes, dtype=torch.uint8, device=self.device)
            self.arena = RingArena(arena_bytes)
        elif (host_units or disk_units) and self.device.type == "cuda":
            raise ValueError(f"{self.label}: streamed units need a non-zero arena")

    # ------------------------------------------------------------------ api
    def view(self, name: str) -> UnitView:
        """A resident unit's view (raises for streamed units)."""
        try:
            return self.views[name]
        except KeyError:
            raise KeyError(f"{name} is not VRAM-resident (tier {self.tier.get(name)})") from None

    def stream(self, order: list[str], cyclic: bool) -> "UnitStream":
        return UnitStream(self, order, cyclic)

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.views.clear()
        self.host.clear()
        self.staging = None
        self.arena_buf = None
        self.arena = None
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- disk reader


class DiskReader:
    """Reads disk-tier units, in stream order, into a pinned FIFO staging ring."""

    def __init__(self, store: WeightStore, order: list[str], cyclic: bool):
        assert store.staging is not None
        self.store = store
        self.order = order
        self.cyclic = cyclic
        self.ring = RingArena(store.staging.numel())
        self.cond = threading.Condition()
        self.ready: dict[int, tuple[torch.Tensor, RingEntry]] = {}
        self.stop = False
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name=f"selas-disk-{store.label}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        st = self.store
        n = len(self.order)
        seq = 0
        try:
            while not self.stop and (self.cyclic or seq < n):
                spec = st.units[self.order[seq % n]]
                with self.cond:
                    while True:
                        if self.stop:
                            return
                        r = self.ring.try_alloc(spec.nbytes)
                        if r is not None:
                            break
                        self.cond.wait(0.05)
                entry, tok = r
                if tok is not None:  # wait until the H2D copy that last read this region has finished
                    while not tok.query():
                        if self.stop:
                            return
                        time.sleep(0.0002)
                dst = st.staging[entry.off : entry.off + spec.nbytes]
                st.c.read_unit_into(spec, dst, direct=st.direct_io)
                st.stats.disk_read_bytes += spec.nbytes
                with self.cond:
                    self.ready[seq] = (dst, entry)
                    self.cond.notify_all()
                seq += 1
        except BaseException as e:  # surfaced to the consumer
            with self.cond:
                self.error = e
                self.cond.notify_all()

    def peek(self, seq: int, block: bool) -> torch.Tensor | None:
        with self.cond:
            while seq not in self.ready:
                if self.error is not None:
                    raise RuntimeError(f"disk reader failed: {self.error!r}") from self.error
                if not block:
                    return None
                self.cond.wait(0.1)
            return self.ready[seq][0]

    def consumed(self, seq: int, copy_done: torch.cuda.Event) -> None:
        with self.cond:
            _, entry = self.ready.pop(seq)
            self.ring.release(entry, copy_done)
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        self.thread.join()


# --------------------------------------------------------------------------- stream


@dataclass
class _Flight:
    seq: int
    name: str
    entry: RingEntry
    ready: torch.cuda.Event
    view: UnitView


class UnitStream:
    """Consumes units in a fixed (optionally cyclic) order, prefetching non-resident ones.

    Callers iterate over *all* units and call ``acquire``/``release`` around each;
    resident units pass straight through. Streamed units must be consumed in
    order, but whole stretches may be skipped as long as nothing is consumed out
    of order (e.g. a cached step that touches no streamed unit leaves the
    prefetched units in place for the next step).
    """

    def __init__(self, store: WeightStore, order: list[str], cyclic: bool):
        if store._active is not None:
            raise RuntimeError(f"{store.label}: another stream is still open")
        self.store = store
        self.cyclic = cyclic
        self.order = [n for n in order if store.tier[n] != VRAM]
        self.n = len(self.order)
        self.issue_seq = 0
        self.consume_seq = 0
        self.disk_seq = 0
        self.flights: deque[_Flight] = deque()
        self.closed = False
        if self.n and store.arena is None:
            raise RuntimeError(f"{store.label}: no arena for streamed units")
        if store.arena is not None:
            store.arena.reset()
        disk_order = [n for n in self.order if store.tier[n] == DISK]
        self.reader = DiskReader(store, disk_order, cyclic) if disk_order else None
        store._active = self
        if self.n:
            self.pump()

    # ------------------------------------------------------------------ internals
    def _limit(self) -> float:
        return float("inf") if self.cyclic else self.n

    def pump(self, need: bool = False) -> None:
        """Issue as many H2D copies as arena space and host data allow."""
        st = self.store
        while self.issue_seq < self._limit():
            name = self.order[self.issue_seq % self.n]
            spec = st.units[name]
            must = need and self.issue_seq == self.consume_seq
            tier = st.tier[name]
            if tier == HOST:
                src = st.host[name]
            else:
                src = self.reader.peek(self.disk_seq, block=must)
                if src is None:
                    return
            r = st.arena.try_alloc(spec.nbytes)
            if r is None:
                if must:  # cannot happen: everything before this unit has been released
                    raise RuntimeError("arena exhausted with nothing in flight")
                return
            entry, wait_tok = r
            dst = st.arena_buf[entry.off : entry.off + spec.nbytes]
            with torch.cuda.stream(st.copy_stream):
                if wait_tok is not None:
                    st.copy_stream.wait_event(wait_tok)
                dst.copy_(src[: spec.nbytes], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(st.copy_stream)
            if tier == DISK:
                self.reader.consumed(self.disk_seq, ready)
                self.disk_seq += 1
            st.stats.h2d_bytes[tier] += spec.nbytes
            st.stats.units_streamed += 1
            self.flights.append(_Flight(self.issue_seq, name, entry, ready, UnitView(spec, dst, st.dtype)))
            self.issue_seq += 1
            need = False

    # ------------------------------------------------------------------ api
    def acquire(self, name: str) -> UnitView:
        st = self.store
        if st.tier[name] == VRAM:
            return st.views[name]
        if self.closed:
            raise RuntimeError("stream is closed")
        if not self.flights:
            self.pump(need=True)
        f = self.flights[0]
        if f.name != name:
            raise RuntimeError(f"{st.label}: stream order violated — expected {f.name}, got {name}")
        cur = torch.cuda.current_stream(st.device)
        if st.profile:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(cur)
            cur.wait_event(f.ready)
            e1.record(cur)
            st.stats.stall_events.append((e0, e1))
        else:
            cur.wait_event(f.ready)
        return f.view

    def release(self, name: str) -> None:
        st = self.store
        if st.tier[name] == VRAM:
            return
        f = self.flights.popleft()
        if f.name != name:
            raise RuntimeError(f"{st.label}: release order violated — expected {f.name}, got {name}")
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(st.device))
        st.arena.release(f.entry, done)
        self.consume_seq += 1
        self.pump()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.reader is not None:
            self.reader.close()
        if self.store.device.type == "cuda":
            torch.cuda.synchronize(self.store.device)
        self.flights.clear()
        if self.store.arena is not None:
            self.store.arena.reset()
        self.store._active = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
