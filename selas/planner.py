"""Placement planner: which units live in VRAM, pinned RAM, or on disk.

The access pattern of a diffusion transformer is a *cyclic scan*: the same units
in the same order every step. LRU/FIFO caching gets a 0 % hit rate on a cyclic
scan larger than the cache; the optimal policy (Belady) is to keep a fixed
subset resident and stream the rest through a small FIFO buffer. The planner
picks that subset with a small pipeline simulator:

* one compute stream (units run back to back),
* one H2D copy engine, FIFO, gated by a byte-accurate ring arena that frees a
  unit's bytes when its compute finishes,
* one disk reader, FIFO, gated by a staging ring that frees when the unit's H2D
  copy finishes.

The simulator runs several cycles so cross-step prefetch is accounted for, and
reports the steady-state cycle time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .store import DISK, HOST, VRAM
from .util import align_up, fmt_seconds, human_bytes


@dataclass
class UnitCost:
    name: str
    nbytes: int
    compute_s: float
    kind: str = "unit"


@dataclass
class Plan:
    tiers: dict[str, str]
    arena_bytes: int
    staging_bytes: int
    step_s: float
    compute_s: float
    prologue_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)

    def bytes_in(self, tier: str, names=None) -> int:
        names = self.tiers.keys() if names is None else names
        return sum(self.sizes.get(n, 0) for n in names if self.tiers.get(n) == tier)

    def count(self, tier: str, names=None) -> int:
        names = self.tiers.keys() if names is None else names
        return sum(1 for n in names if self.tiers.get(n) == tier)

    def describe(self, cycle_order: list[str] | None = None) -> str:
        cyc = cycle_order or list(self.tiers)
        lines = [
            f"vram  {self.count(VRAM, cyc):3d} units {human_bytes(self.bytes_in(VRAM, cyc)):>11}  + arena {human_bytes(self.arena_bytes)}",
            f"host  {self.count(HOST, cyc):3d} units {human_bytes(self.bytes_in(HOST, cyc)):>11}  + staging {human_bytes(self.staging_bytes)}",
            f"disk  {self.count(DISK, cyc):3d} units {human_bytes(self.bytes_in(DISK, cyc)):>11}",
            f"predicted step {fmt_seconds(self.step_s)} (pure compute {fmt_seconds(self.compute_s)}, "
            f"exposed transfer {fmt_seconds(max(0.0, self.step_s - self.compute_s))})",
        ]
        if self.prologue_s:
            lines.append(f"predicted prologue {fmt_seconds(self.prologue_s)} (once per image batch)")
        if cycle_order:
            lines.append("map   " + "".join({VRAM: "V", HOST: "h", DISK: "d"}[self.tiers[n]] for n in cycle_order))
        lines += self.notes
        return "\n".join(lines)


def simulate(
    order: list[str],
    tiers: dict[str, str],
    size: dict[str, int],
    comp: dict[str, float],
    arena: int,
    staging: int,
    bw_h2d: float,
    bw_disk: float,
    cycles: int = 3,
) -> tuple[float, float]:
    """Return (steady-state seconds per cycle, pure compute seconds per cycle)."""
    if not order:
        return 0.0, 0.0
    t = 0.0
    copy_free = 0.0
    disk_free = 0.0
    dev_q: deque = deque()
    dev_used = 0
    stg_q: deque = deque()
    stg_used = 0
    ends = []
    n = len(order)
    for i in range(n * cycles):
        name = order[i % n]
        tier = tiers[name]
        if tier == VRAM:
            t += comp[name]
        else:
            nb = size[name]
            host_ready = 0.0
            if tier == DISK:
                start = disk_free
                while stg_used + nb > staging and stg_q:
                    b, ce = stg_q.popleft()
                    stg_used -= b
                    start = max(start, ce)
                disk_free = start + nb / bw_disk
                host_ready = disk_free
            cstart = max(copy_free, host_ready)
            while dev_used + nb > arena and dev_q:
                b, ce = dev_q.popleft()
                dev_used -= b
                cstart = max(cstart, ce)
            copy_free = cstart + nb / bw_h2d
            if tier == DISK:
                stg_q.append((nb, copy_free))
                stg_used += nb
            t = max(t, copy_free) + comp[name]
            dev_q.append((nb, t))
            dev_used += nb
        if (i + 1) % n == 0:
            ends.append(t)
    per_cycle = ends[-1] - ends[-2] if len(ends) >= 2 else ends[-1]
    return per_cycle, sum(comp[x] for x in order)


def simulate_once(order, tiers, size, comp, arena, staging, bw_h2d, bw_disk) -> float:
    return simulate(order, tiers, size, comp, arena, staging, bw_h2d, bw_disk, cycles=1)[0]


def _spread_pick(candidates: list[str], chosen: set[str], order: list[str]) -> str:
    """Candidate farthest (cyclically) from already-chosen units: spreads residency so
    resident compute is interleaved with streamed transfers."""
    if not chosen:
        return candidates[len(candidates) // 2]
    pos = {n: i for i, n in enumerate(order)}
    n = len(order)
    cpos = sorted(pos[c] for c in chosen)

    def dist(x: str) -> int:
        p = pos[x]
        return min(min((p - q) % n, (q - p) % n) for q in cpos)

    return max(candidates, key=lambda x: (dist(x), -pos[x]))


def make_plan(
    cycle: list[UnitCost],
    once: list[UnitCost],
    vram_avail: int,
    ram_avail: int,
    bw_h2d: float,
    bw_disk: float,
    force_vram: set[str] | frozenset = frozenset(),
    arena_multiples: tuple[float, ...] = (2.0, 3.0, 4.0),
    min_arena: int = 0,
) -> Plan:
    """Plan tiers for a cyclic schedule (``cycle``) plus once-per-run units (``once``).

    ``vram_avail`` covers resident units *and* the arena; ``ram_avail`` covers
    host-tier units *and* the disk staging ring.
    """
    order = [u.name for u in cycle]
    size = {u.name: align_up(u.nbytes, 4096) for u in cycle + once}
    comp = {u.name: u.compute_s for u in cycle + once}
    kind = {u.name: u.kind for u in cycle + once}
    forced = [n for n in order if n in force_vram]
    forced_bytes = sum(size[n] for n in forced)
    free_units = [n for n in order if n not in force_vram]
    once_names = [u.name for u in once]
    max_free = max((size[n] for n in free_units), default=0)
    max_once = max((size[n] for n in once_names), default=0)

    best: Plan | None = None
    tried = set()
    # Everything resident: no arena needed for the cycle (only for once-units).
    candidates_arena = [0] + sorted({align_up(int(m * max_free), 4096) for m in arena_multiples if max_free})
    for arena_cycle in candidates_arena:
        arena = max(arena_cycle, align_up(2 * max_once, 4096) if max_once else 0, min_arena)
        if (arena, arena_cycle == 0) in tried:
            continue
        tried.add((arena, arena_cycle == 0))
        budget = vram_avail - arena - forced_bytes
        if budget < 0:
            continue
        tiers = {n: HOST for n in order}
        for n in forced:
            tiers[n] = VRAM
        if arena_cycle == 0:
            if sum(size[n] for n in free_units) > budget:
                continue
            for n in free_units:
                tiers[n] = VRAM
        else:
            _greedy_resident(order, tiers, size, comp, kind, free_units, budget, arena, bw_h2d, bw_disk)
        # Host tier: whatever does not fit in RAM goes to disk (needs staging). Room for the
        # staging ring that disk-resident once-units would need is reserved up front.
        once_stage = 2 * max_once if once_names else 0
        cycle_staging = _fit_host(order, tiers, size, comp, kind, ram_avail - once_stage, arena, bw_h2d, bw_disk)
        step_s, comp_s = simulate(order, tiers, size, comp, arena, cycle_staging, bw_h2d, bw_disk)
        # Once-units (prologue): host if leftover RAM, else disk.
        used_ram = sum(size[n] for n in order if tiers[n] == HOST) + max(cycle_staging, once_stage)
        once_tiers = {}
        for n in once_names:
            if used_ram + size[n] <= ram_avail:
                once_tiers[n] = HOST
                used_ram += size[n]
            else:
                once_tiers[n] = DISK
        once_disk = [n for n, t in once_tiers.items() if t == DISK]
        staging = max(cycle_staging, 2 * max(size[n] for n in once_disk)) if once_disk else cycle_staging
        all_tiers = {**tiers, **once_tiers}
        prologue_s = simulate_once(once_names, all_tiers, size, comp, arena, staging, bw_h2d, bw_disk) if once_names else 0.0
        plan = Plan(all_tiers, arena, staging, step_s, comp_s, prologue_s, sizes=dict(size))
        if best is None or plan.step_s < best.step_s * 0.995 or (
            abs(plan.step_s - best.step_s) <= best.step_s * 0.005 and plan.arena_bytes < best.arena_bytes
        ):
            best = plan
    if best is None:
        need = forced_bytes + align_up(2 * max(max_free, max_once), 4096)
        raise MemoryError(
            f"not enough VRAM to run: need at least {human_bytes(need)} for forced-resident units and a "
            f"2-unit arena, have {human_bytes(vram_avail)}"
        )
    return best


def _greedy_resident(order, tiers, size, comp, kind, free_units, budget, arena, bw_h2d, bw_disk) -> None:
    """Make units resident one at a time, best simulated gain per byte first.

    Candidates: per unit kind, the not-yet-resident unit farthest from existing
    residents. Keeps filling while VRAM remains (residency is never harmful),
    ordering by gain/byte and breaking ties toward low compute-per-byte kinds.
    """
    used = 0
    chosen = {n for n in order if tiers[n] == VRAM}
    staging = 0  # residency decisions assume host-tier streaming
    base, _ = simulate(order, tiers, size, comp, arena, staging, bw_h2d, bw_disk)
    while True:
        remaining = [n for n in free_units if tiers[n] != VRAM and used + size[n] <= budget]
        if not remaining:
            return
        by_kind: dict[str, list[str]] = {}
        for n in remaining:
            by_kind.setdefault(kind[n], []).append(n)
        best = None
        for k, cands in by_kind.items():
            c = _spread_pick(cands, chosen, order)
            tiers[c] = VRAM
            s, _ = simulate(order, tiers, size, comp, arena, staging, bw_h2d, bw_disk)
            tiers[c] = HOST
            gain = (base - s) / size[c]
            ratio = comp[c] / size[c]  # lower = worse to stream
            key = (round(gain * 1e12, 6), -ratio)
            if best is None or key > best[0]:
                best = (key, c, s)
        _, c, s = best
        tiers[c] = VRAM
        chosen.add(c)
        used += size[c]
        base = s


def _fit_host(order, tiers, size, comp, kind, ram_avail, arena, bw_h2d, bw_disk) -> int:
    """Demote host-tier cycle units to disk until they fit in RAM; returns the staging size."""
    host = [n for n in order if tiers[n] == HOST]
    if not host:
        return 0
    if sum(size[n] for n in host) <= ram_avail:
        return 0
    staging = 0
    while True:
        host = [n for n in order if tiers[n] == HOST]
        disk = [n for n in order if tiers[n] == DISK]
        staging = 2 * max((size[n] for n in disk), default=0)
        if sum(size[n] for n in host) + staging <= ram_avail or not host:
            return staging
        base, _ = simulate(order, tiers, size, comp, arena, max(staging, 1), bw_h2d, bw_disk)
        by_kind: dict[str, list[str]] = {}
        for n in host:
            by_kind.setdefault(kind[n], []).append(n)
        best = None
        for k, cands in by_kind.items():
            # spread disk units out as well: pick the host unit farthest from other disk units
            c = _spread_pick(cands, set(disk), order)
            tiers[c] = DISK
            st = max(staging, 2 * size[c])
            s, _ = simulate(order, tiers, size, comp, arena, st, bw_h2d, bw_disk)
            tiers[c] = HOST
            loss = (s - base) / size[c]
            if best is None or loss < best[0]:
                best = (loss, c)
        tiers[best[1]] = DISK
