"""O_DIRECT vs buffered reads for the disk tier (docs/PLAN.md §7).

1. Microbenchmark: read the transformer's streamed units in stream order into a
   pinned buffer, single-threaded like the engine's DiskReader: buffered from a
   cold page cache (evicted with posix_fadvise), buffered warm, and O_DIRECT.
   Page-cache residency is checked with mincore.
2. Engine A/B at 512² (transfer-bound, so disk speed shows) with ``--ram-gb 6``:
   buffered vs O_DIRECT, (a) unconstrained — buffered reads may be served by the
   page cache after the first step, as the disk set fits in free RAM — and
   (b) inside a user cgroup whose MemoryMax leaves only ~1.5 GiB for page cache,
   the situation the disk tier exists for (RAM really is short).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import ROOT, evict, resident_fraction, save_result


def microbench(model: Path, reps: int) -> list[dict]:
    from selas.container import Container
    from selas.models.flux import FluxConfig, main_order
    from selas.store import PinnedPool

    c = Container(model / "transformer")
    units = [c.units[n] for n in main_order(FluxConfig.from_dict(c.config))[15:]]  # the default plan's streamed set
    total = sum(u.nbytes for u in units)
    ranges = [(u.offset, u.nbytes) for u in units]
    pool = PinnedPool(max(u.nbytes for u in units))
    buf = pool.take(max(u.nbytes for u in units))

    def read_all(direct: bool) -> tuple[float, float]:
        t0, c0 = time.perf_counter(), time.process_time()
        for u in units:
            c.read_unit_into(u, buf, direct=direct)
        return total / (time.perf_counter() - t0), (time.process_time() - c0) / (total / 2**30)

    rows = []
    try:
        for rep in range(reps):
            evict(c.data_path)
            f0 = resident_fraction(c.data_path, ranges)
            bw_cold, cpu_cold = read_all(False)
            f1 = resident_fraction(c.data_path, ranges)
            bw_warm, cpu_warm = read_all(False)
            evict(c.data_path)
            bw_dir, cpu_dir = read_all(True)
            f2 = resident_fraction(c.data_path, ranges)
            assert not c.direct_failed, "O_DIRECT fell back to buffered"
            row = {"rep": rep, "bytes": total, "buffered_cold_bw": bw_cold, "buffered_warm_bw": bw_warm, "direct_bw": bw_dir,
                   "cpu_s_per_gib": {"buffered_cold": cpu_cold, "buffered_warm": cpu_warm, "direct": cpu_dir},
                   "resident_before_cold": f0, "resident_after_cold": f1, "resident_after_direct": f2}
            rows.append(row)
            print(f"rep {rep}: {total / 2**30:.2f} GiB  buffered cold {bw_cold / 2**30:.2f} GiB/s "
                  f"(cache {f0:.0%} -> {f1:.0%})  warm {bw_warm / 2**30:.2f} GiB/s  O_DIRECT {bw_dir / 2**30:.2f} GiB/s "
                  f"(cache after {f2:.0%})  CPU s/GiB cold {cpu_cold:.3f} warm {cpu_warm:.3f} direct {cpu_dir:.3f}", flush=True)
    finally:
        pool.close()
        c.close()
    return rows


def child(a) -> None:
    """One engine run; writes stats JSON to ``a.out``."""
    from selas.container import Container
    from selas.pipeline import FluxEngine, Job, RuntimeOptions
    from selas.store import DISK

    from .common import PROMPTS

    eng = FluxEngine(a.model, RuntimeOptions(ram_gb=a.ram_gb, direct_io=bool(a.direct), profile=True))
    try:
        res = eng.generate([Job(PROMPTS[0], seed=0, width=a.res, height=a.res, steps=a.steps)])
        st = res[0].stats
        disk_units = [n for n, t in eng.plan.tiers.items() if t == DISK]
        direct_failed = eng.tc.direct_failed
    finally:
        eng.close()
    c = Container(Path(a.model) / "transformer")
    ranges = [(c.units[n].offset, c.units[n].nbytes) for n in disk_units]
    out = {"direct": bool(a.direct), "mean_step_s": st["mean_full_step_s"], "step_s": st["step_s"],
           "stall_s_total": st["stall_s"], "disk_bytes_per_step": st["h2d_bytes"]["disk"] / st["steps"],
           "resident_after": resident_fraction(c.data_path, ranges), "direct_failed": direct_failed,
           "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    Path(a.out).write_text(json.dumps(out))


def engine_run(a, direct: bool, mem_max: int | None) -> dict:
    from selas.container import Container

    evict(Container(Path(a.model) / "transformer").data_path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = [sys.executable, "-m", "experiments.disk_io", "--child", "--model", a.model, "--res", str(a.res),
           "--steps", str(a.steps), "--ram-gb", str(a.ram_gb), "--direct", str(int(direct)), "--out", out]
    if mem_max:
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem_max}", "-p", "MemorySwapMax=0"] + cmd
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"child failed ({p.returncode}):\n{p.stderr[-3000:]}")
    rec = json.loads(Path(out).read_text())
    os.unlink(out)
    rec["mem_max"] = mem_max
    steps = rec["step_s"]
    print(f"  {'O_DIRECT' if direct else 'buffered':8s} {'limit ' + format(mem_max / 2**30, '.1f') + ' GiB' if mem_max else 'no limit':>14}: "
          f"step {rec['mean_step_s']:.3f} s (first {steps[0]:.2f}, last {steps[-1]:.2f}), stall {rec['stall_s_total']:.2f} s total, "
          f"disk {rec['disk_bytes_per_step'] / 2**30:.2f} GiB/step, disk set cached after {rec['resident_after']:.0%}, "
          f"maxrss {rec['maxrss'] / 2**30:.1f} GiB", flush=True)
    assert not rec["direct_failed"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--ram-gb", type=float, default=6.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--cache-headroom-gb", type=float, default=1.5)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--direct", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        child(a)
        return
    model = Path(a.model)
    print("microbenchmark", flush=True)
    micro = microbench(model, a.reps)
    print("engine, no memory limit", flush=True)
    free = []
    for _ in range(a.reps):
        free += [engine_run(a, False, None), engine_run(a, True, None)]
    rss = max(r["maxrss"] for r in free)
    limit = int(rss + a.cache_headroom_gb * 2**30)
    print(f"engine, cgroup MemoryMax {limit / 2**30:.1f} GiB (peak RSS {rss / 2**30:.1f} GiB + {a.cache_headroom_gb} GiB for page cache)", flush=True)
    limited = []
    for _ in range(a.reps):
        for direct in (False, True):
            try:
                limited.append(engine_run(a, direct, limit))
            except RuntimeError as e:  # e.g. OOM-killed inside the cgroup: record it, keep the other results
                print(f"  {'O_DIRECT' if direct else 'buffered'} under the limit failed: {str(e)[-300:]}", flush=True)
                limited.append({"direct": direct, "error": str(e)[-2000:]})

    def agg(rows, direct):
        rs = [r for r in rows if r["direct"] == direct and "error" not in r]
        return sum(r["mean_step_s"] for r in rs) / len(rs) if rs else None

    summary = {"micro_buffered_cold_gibs": sum(r["buffered_cold_bw"] for r in micro) / len(micro) / 2**30,
               "micro_direct_gibs": sum(r["direct_bw"] for r in micro) / len(micro) / 2**30,
               "free_buffered_step_s": agg(free, False), "free_direct_step_s": agg(free, True),
               "limited_buffered_step_s": agg(limited, False), "limited_direct_step_s": agg(limited, True),
               "mem_limit": limit}
    print(json.dumps(summary, indent=1))
    save_result(f"disk_io_{a.res}", {"model": a.model, "ram_gb": a.ram_gb, "steps": a.steps, "summary": summary,
                                     "microbench": micro, "engine_free": free, "engine_limited": limited,
                                     "mean_after_first_note": "engine mean_step_s excludes the first step"})


if __name__ == "__main__":
    main()
