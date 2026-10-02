"""Tiered weight store and the streaming executor.

Every unit of a container lives in exactly one tier:

* ``vram`` — loaded once into its own device buffer; access is free.
* ``host`` — page-locked RAM; streamed to the GPU by DMA on a copy stream.
* ``disk`` — the container on NVMe; a reader thread ``pread``s it into a pinned
  staging ring ahead of time, then it is streamed like a host unit.
* ``absent`` — not loaded at all (e.g. modulation units when every image's
  modulation vectors are cached); using it is an error.

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
import os
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

VRAM, HOST, DISK, ABSENT = "vram", "host", "disk", "absent"
TIERS = (VRAM, HOST, DISK, ABSENT)


# --------------------------------------------------------------------------- pinned host memory


def _cuda_ok(err) -> bool:
    try:
        return int(err) == 0
    except Exception:  # pragma: no cover
        return str(err).endswith("success")


def _prefault(buf: torch.Tensor, threads: int = 8, min_bytes: int = 256 << 20) -> None:
    """Touch one byte per page of a fresh mapping, from several threads (torch releases the GIL)."""
    n = buf.numel()
    k = min(threads, os.cpu_count() or 1, max(1, n // min_bytes))
    step = align_up(-(-n // k), 4096)
    parts = [buf[a : min(n, a + step) : 4096] for a in range(0, n, step)]
    if len(parts) == 1:
        parts[0].fill_(0)
        return
    ts = [threading.Thread(target=p.fill_, args=(0,)) for p in parts]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


_releasing: list[threading.Thread] = []


def _release_later(fn) -> None:
    t = threading.Thread(target=fn, name="selas-unmap")  # not a daemon: the interpreter waits for it
    t.start()
    _releasing.append(t)


def wait_released() -> None:
    """Wait for pools being unmapped in the background (so RAM is not held twice)."""
    while _releasing:
        _releasing.pop().join()


class PinnedPool:
    """One page-aligned anonymous mapping, page-locked with ``cudaHostRegister``.

    PyTorch's pinned allocator rounds every allocation up to a power of two, which
    would waste up to 2x RAM for multi-GiB weight sets; registering our own mapping
    pins exactly what we use. Slices are handed out bump-allocator style.

    Registering faults in and zeroes every page first, single-threaded (~0.6 s/GiB);
    the pages are faulted in parallel beforehand instead (~0.1 s/GiB on 8 threads),
    which leaves registration only the pinning (~0.2 s/GiB). ``pin=True`` pins the
    whole pool now; with ``pin=False`` the owner pins slices as it fills them
    (:meth:`register`), each slice a separate range: the driver lock is then held
    in short pieces, and a copy must not span two ranges. (Transparent huge pages
    would make pinning cheaper still, but on a fragmented desktop their compaction
    stalls made loading slower.)
    """

    def __init__(self, nbytes: int, pin: bool = True):
        self.nbytes = align_up(max(int(nbytes), 4096), 4096)
        if _releasing:  # a small pool (e.g. the VAE's) may overlap an old pool's release; a large one waits
            from .hw import ram_available

            if ram_available()[1] < self.nbytes + (2 << 30):
                wait_released()
        self._mm = mmap.mmap(-1, self.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.base = torch.frombuffer(self._mm, dtype=torch.uint8)
        self._regs: list[int] = []  # start addresses of registered ranges
        self._lock = threading.Lock()
        self._warned = False
        if pin and torch.cuda.is_available():
            _prefault(self.base)
            self.register(self.base)
        self._cursor = 0

    @property
    def pinned(self) -> bool:
        return bool(self._regs)

    def register(self, t: torch.Tensor) -> bool:
        """Page-lock a slice of the pool (already faulted in, ideally) as its own range."""
        try:
            ok = _cuda_ok(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), align_up(t.numel(), 4096), 0))
        except Exception as e:  # pragma: no cover
            ok = False
            warn(f"cudaHostRegister unavailable ({e})")
        with self._lock:
            if ok:
                self._regs.append(t.data_ptr())
            elif not self._warned:
                self._warned = True
                warn(f"could not page-lock host memory ({human_bytes(t.numel())}); host->device copies will be slower")
        return ok

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

    def close(self, unmap_async: bool = False) -> None:
        """Unpin (cheap: ~0.02 s/GiB) and unmap (frees the pages: ~0.12 s/GiB). With ``unmap_async``
        the unmapping runs on a background thread; it involves no CUDA calls, so GPU work proceeds."""
        if self.base is None:
            return
        with self._lock:
            regs, self._regs = self._regs, []
        for ptr in regs:
            try:
                torch.cuda.cudart().cudaHostUnregister(ptr)
            except Exception:  # pragma: no cover
                pass
        self.base = None
        if unmap_async:
            _release_later(self._unmap)
        else:
            self._unmap()

    def _unmap(self, chunk: int = 64 << 20) -> None:
        gc.collect()
        # Free the pages in chunks first: munmap holds the address-space lock for writing the whole
        # time (~2 s for 15 GiB), stalling every page fault and pinning elsewhere in the process (the
        # VAE loading meanwhile); MADV_DONTNEED frees the same pages under the read lock, a chunk at a
        # time. A slice still referenced somewhere would read zeros afterwards, never fault.
        if hasattr(mmap, "MADV_DONTNEED"):
            try:
                for off in range(0, self.nbytes, chunk):
                    self._mm.madvise(mmap.MADV_DONTNEED, off, min(chunk, self.nbytes - off))
            except (OSError, ValueError):
                pass
        try:
            self._mm.close()
        except BufferError:  # a slice is still referenced; the mapping is freed when it is collected
            pass


class _Prefaulter:
    """Faults in a fresh mapping front to back on several threads, ahead of a reader:
    ``wait(end)`` returns once ``[0, end)`` is faulted in."""

    def __init__(self, buf: torch.Tensor, threads: int = 8, chunk: int = 64 << 20):
        self.buf, self.chunk = buf, chunk
        self.n = buf.numel()
        self.nchunks = -(-self.n // chunk)
        self.next = 0
        self.done = [False] * self.nchunks
        self.prefix = 0  # chunks [0, prefix) are done
        self.stop = False
        self.cond = threading.Condition()
        k = min(threads, os.cpu_count() or 1, self.nchunks)
        self.threads = [threading.Thread(target=self._work, name="selas-prefault", daemon=True) for _ in range(k)]
        for t in self.threads:
            t.start()

    def _work(self) -> None:
        while True:
            with self.cond:
                if self.stop or self.next >= self.nchunks:
                    return
                i = self.next
                self.next += 1
            a = i * self.chunk
            self.buf[a : min(self.n, a + self.chunk) : 4096].fill_(0)
            with self.cond:
                self.done[i] = True
                while self.prefix < self.nchunks and self.done[self.prefix]:
                    self.prefix += 1
                self.cond.notify_all()

    def wait(self, end: int) -> None:
        with self.cond:
            while self.prefix * self.chunk < end and self.prefix < self.nchunks and not self.stop:
                self.cond.wait(0.1)

    def close(self) -> None:
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        for t in self.threads:
            t.join()


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
    """Owns one container's units across the VRAM / host / disk tiers.

    On CUDA, construction only allocates: the units are read on a background
    thread in ``order`` (their first use; the rest after), so compute starts while
    the weights are still loading. :meth:`view` and the streams wait for each unit
    as needed (the CPU until it is read, the GPU until it is copied);
    :meth:`wait_loaded` waits for all of them.
    """

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
        order: list[str] | None = None,
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
            self.tier = {n: (ABSENT if t == ABSENT else VRAM) for n, t in self.tier.items()}
        first = [n for n in dict.fromkeys(order or []) if n in self.units]
        self.order = first + [n for n in self.units if n not in set(first)]  # load order

        vram_units = [n for n in self.order if self.tier[n] == VRAM]
        host_units = [n for n in self.order if self.tier[n] == HOST]
        disk_units = [n for n in self.order if self.tier[n] == DISK]
        if disk_units:
            staging_bytes = max(staging_bytes, 2 * max(self.units[n].nbytes for n in disk_units))
        else:
            staging_bytes = 0
        self.staging_bytes = staging_bytes
        self.host_bytes = sum(align_up(self.units[n].nbytes, 4096) for n in host_units)

        self.pool: PinnedPool | None = None
        self.host: dict[str, torch.Tensor] = {}
        self.views: dict[str, UnitView] = {}
        self.staging: torch.Tensor | None = None
        self.arena_buf: torch.Tensor | None = None
        self.arena: RingArena | None = None
        self.copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._ready: dict[str, threading.Event] = {}  # unit -> read (host: and pinned; vram: copy issued)
        self._copied: dict[str, torch.cuda.Event] = {}  # vram unit -> its H2D copy, on the load stream
        self._loader: threading.Thread | None = None
        self._load_error: BaseException | None = None
        self._stop = False

        self._t0 = time.perf_counter()
        try:
            self._setup(vram_units, host_units, disk_units, arena_bytes)
        except BaseException:
            self.close()  # never leak page-locked memory or device buffers on a failed load
            raise
        if self._loader is None:
            self._log_loaded()

    def _summary(self) -> str:
        vb = sum(self.units[n].nbytes for n, t in self.tier.items() if t == VRAM)
        db = sum(self.units[n].nbytes for n, t in self.tier.items() if t == DISK)
        count = lambda tier: sum(1 for t in self.tier.values() if t == tier)  # noqa: E731
        return (f"{self.label}: vram {count(VRAM)} units {human_bytes(vb)}, host {count(HOST)} units "
                f"{human_bytes(self.host_bytes)}, disk {count(DISK)} units {human_bytes(db)}, "
                f"arena {human_bytes(self.arena_buf.numel() if self.arena_buf is not None else 0)}, "
                f"staging {human_bytes(self.staging_bytes)}")

    def _log_loaded(self, background: bool = False) -> None:
        log(f"{self._summary()} (loaded in {time.perf_counter() - self._t0:.1f}s{', in the background' if background else ''})")

    def _setup(self, vram_units, host_units, disk_units, arena_bytes: int) -> None:
        if self.device.type != "cuda":
            for n in vram_units:
                dev = torch.empty(self.units[n].nbytes, dtype=torch.uint8)
                self.c.read_unit_into(self.units[n], dev)
                self.views[n] = UnitView(self.units[n], dev, self.dtype)
            return
        if self.host_bytes or self.staging_bytes:
            self.pool = PinnedPool(self.host_bytes + self.staging_bytes, pin=False)
            for n in host_units:  # in load order: the prefaulter runs ahead of the reader
                self.host[n] = self.pool.take(self.units[n].nbytes)
            if self.staging_bytes:
                self.staging = self.pool.take(self.staging_bytes)
                _prefault(self.staging)
                self.pool.register(self.staging)
        for n in vram_units:
            dev = torch.empty(self.units[n].nbytes, dtype=torch.uint8, device=self.device)
            self.views[n] = UnitView(self.units[n], dev, self.dtype)
        if arena_bytes and (host_units or disk_units):
            max_streamed = max(self.units[n].nbytes for n in host_units + disk_units)
            arena_bytes = max(int(arena_bytes), max_streamed)
            self.arena_buf = torch.empty(arena_bytes, dtype=torch.uint8, device=self.device)
            self.arena = RingArena(arena_bytes)
        elif host_units or disk_units:
            raise ValueError(f"{self.label}: streamed units need a non-zero arena")
        todo = [n for n in self.order if self.tier[n] in (VRAM, HOST)]
        if not todo:
            return
        self._ready = {n: threading.Event() for n in todo}
        self._loader = threading.Thread(target=self._load, args=(todo,), name=f"selas-load-{self.label}", daemon=True)
        self._loader.start()

    # ------------------------------------------------------------------ background loading
    def _load(self, todo: list[str]) -> None:
        """Loader thread: read every VRAM/host unit in load order.

        VRAM units go through a two-slot pinned scratch (read one while the other is
        copied). Host units are read straight into their pool slice, which a second
        thread then pins; the pages were faulted in ahead of the reads by a third.
        """
        units, direct = self.units, self.direct_io
        vram = [n for n in todo if self.tier[n] == VRAM]
        host = [n for n in todo if self.tier[n] == HOST]
        scratch_pool = prefaulter = None
        pin_q: deque = deque()
        pin_cond = threading.Condition()
        pinner = None

        def pin_loop() -> None:
            while True:
                with pin_cond:
                    while not pin_q:
                        pin_cond.wait()
                    name = pin_q.popleft()
                if name is None:
                    return
                if not self._stop:
                    self.pool.register(self.host[name])
                self._ready[name].set()

        try:
            with torch.cuda.device(self.device):
                load_stream = torch.cuda.Stream(self.device)
                if vram:
                    slot = max(units[n].nbytes for n in vram)
                    scratch_pool = PinnedPool(slot * min(2, len(vram)))
                    slots = [scratch_pool.base[i * slot : (i + 1) * slot] for i in range(min(2, len(vram)))]
                    slot_done: list[torch.cuda.Event | None] = [None] * len(slots)
                if host:
                    prefaulter = _Prefaulter(self.pool.base[: self.host_bytes])
                    pinner = threading.Thread(target=pin_loop, name=f"selas-pin-{self.label}", daemon=True)
                    pinner.start()
                k = 0
                for name in todo:
                    if self._stop:
                        break
                    spec = units[name]
                    if self.tier[name] == VRAM:
                        i, k = k % len(slots), k + 1
                        if slot_done[i] is not None:
                            slot_done[i].synchronize()  # the copy that last read this slot
                        self.c.read_unit_into(spec, slots[i], direct=direct)
                        with torch.cuda.stream(load_stream):
                            self.views[name].buf.copy_(slots[i][: spec.nbytes], non_blocking=True)
                            ev = torch.cuda.Event()
                            ev.record(load_stream)
                        slot_done[i] = self._copied[name] = ev
                        self._ready[name].set()
                    else:
                        dst = self.host[name]
                        prefaulter.wait(dst.data_ptr() - self.pool.base.data_ptr() + dst.numel())
                        self.c.read_unit_into(spec, dst, direct=direct)
                        with pin_cond:
                            pin_q.append(name)
                            pin_cond.notify()
                if pinner is not None:
                    with pin_cond:
                        pin_q.append(None)
                        pin_cond.notify()
                    pinner.join()
                    pinner = None
                for ev in self._copied.values():
                    ev.synchronize()
            if not self._stop:
                self._log_loaded(background=True)
        except BaseException as e:  # surfaced to whoever waits for a unit
            self._load_error = e
        finally:
            if pinner is not None:
                with pin_cond:
                    pin_q.append(None)
                    pin_cond.notify()
                pinner.join()
            if prefaulter is not None:
                prefaulter.close()
            if scratch_pool is not None:
                try:
                    load_stream.synchronize()  # after an error too: no copy may still read the scratch
                except Exception:  # pragma: no cover
                    pass
                scratch_pool.close()
            for ev in self._ready.values():  # wake every waiter; they check _load_error
                ev.set()

    def _check_loading(self) -> None:
        if self._load_error is not None:
            raise RuntimeError(f"{self.label}: loading weights failed: {self._load_error!r}") from self._load_error

    @property
    def loading(self) -> bool:
        return self._loader is not None and self._loader.is_alive()

    def is_ready(self, name: str) -> bool:
        ev = self._ready.get(name)
        if ev is None:
            return True
        if not ev.is_set():
            return False
        self._check_loading()  # a failed loader sets every event: the unread units hold zeros
        return True

    def wait_unit(self, name: str) -> None:
        """Block until ``name`` is loaded; for a VRAM unit, also order the current stream after its copy."""
        ev = self._ready.get(name)
        if ev is not None and not ev.is_set():
            ev.wait()
        self._check_loading()
        done = self._copied.get(name)
        if done is not None:
            torch.cuda.current_stream(self.device).wait_event(done)

    def wait_loaded(self) -> None:
        if self._loader is not None:
            self._loader.join()
        self._check_loading()

    # ------------------------------------------------------------------ api
    def view(self, name: str) -> UnitView:
        """A resident unit's view (raises for streamed units); waits until it is loaded."""
        try:
            v = self.views[name]
        except KeyError:
            raise KeyError(f"{name} is not VRAM-resident (tier {self.tier.get(name)})") from None
        self.wait_unit(name)
        return v

    def stream(self, order: list[str], cyclic: bool) -> "UnitStream":
        return UnitStream(self, order, cyclic)

    def close(self, release_host_async: bool = False) -> None:
        """Free everything. ``release_host_async``: unmap the host pool on a background
        thread (seconds for a large pool); VRAM and the pinning are released on return."""
        self._stop = True
        if self._loader is not None:
            self._loader.join()  # finishes the read in flight, then stops
            self._loader = None
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
            self.pool.close(unmap_async=release_host_async)
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
        absent = [n for n in order if store.tier[n] == ABSENT]
        if absent:
            raise RuntimeError(f"{store.label}: cannot stream units that were not loaded: {', '.join(absent[:4])}")
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
                if not st.is_ready(name):  # still loading: issue it later, or wait if it is needed now
                    if not must:
                        return
                    st.wait_unit(name)
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
            return st.view(name)
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
