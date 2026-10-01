"""Byte-granular FIFO ring allocator.

Units are streamed in a fixed order and consumed in the same order, so a ring
buffer is the natural allocator: allocate at ``head`` (wrapping to 0 when the
tail of the buffer is too small), free from the front.

Freeing is *deferred*: ``release(entry, token)`` marks an entry reusable once
``token`` (typically a CUDA event recorded after the last kernel that read the
entry) has completed. ``try_alloc`` reclaims released entries lazily and hands
the caller the newest token it reclaimed, which the caller must wait on before
writing into the new region. Tokens are produced in FIFO order on a single
stream, so waiting on the newest one covers all older ones.

This module has no torch dependency so its invariants can be tested on the CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class RingEntry:
    off: int
    size: int
    token: Any = None
    released: bool = False


class RingArena:
    def __init__(self, capacity: int, align: int = 4096):
        if capacity <= 0:
            raise ValueError("ring capacity must be positive")
        self.align = align
        self.capacity = capacity // align * align
        if self.capacity <= 0:
            raise ValueError("ring capacity smaller than alignment")
        self._entries: deque[RingEntry] = deque()
        self._head = 0
        self._pending: Any = None

    # ------------------------------------------------------------------ internals
    def _round(self, size: int) -> int:
        return max(self.align, (int(size) + self.align - 1) // self.align * self.align)

    def _fit(self, size: int) -> int | None:
        if not self._entries:
            self._head = 0
            return 0 if size <= self.capacity else None
        tail = self._entries[0].off
        head = self._head
        if tail < head:  # used region is [tail, head): free space is [head, cap) and [0, tail)
            if head + size <= self.capacity:
                return head
            if size <= tail:
                return 0
            return None
        # head <= tail: wrapped, free space is [head, tail)
        if head + size <= tail:
            return head
        return None

    # ------------------------------------------------------------------ api
    def try_alloc(self, size: int) -> tuple[RingEntry, Any] | None:
        """Allocate ``size`` bytes or return None if the live entries do not leave room.

        Returns ``(entry, wait_token)``; ``wait_token`` is None or a token the
        caller must wait on before writing the region.
        """
        size = self._round(size)
        if size > self.capacity:
            raise ValueError(f"allocation of {size} bytes exceeds ring capacity {self.capacity}")
        while True:
            off = self._fit(size)
            if off is not None:
                entry = RingEntry(off, size)
                self._entries.append(entry)
                self._head = off + size
                tok, self._pending = self._pending, None
                return entry, tok
            if self._entries and self._entries[0].released:
                e = self._entries.popleft()
                if e.token is not None:
                    self._pending = e.token
                continue
            return None

    def release(self, entry: RingEntry, token: Any = None) -> None:
        if entry.released:
            raise RuntimeError("ring entry released twice")
        entry.token = token
        entry.released = True

    def reset(self) -> None:
        """Forget everything. Only valid when no region is in use by anyone."""
        self._entries.clear()
        self._head = 0
        self._pending = None

    # ------------------------------------------------------------------ introspection
    def __len__(self) -> int:
        return len(self._entries)
