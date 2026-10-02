"""Planner placement vs naive "first N blocks resident" vs "stream everything".

Hypotheses (docs/PLAN.md §7):
  * streaming is fully hidden at >= 768² on PCIe 3.0 (also measured here at 512²);
  * the planner's (spread) resident set beats "first N resident" at the same VRAM.

Every A/B pair gets the same VRAM for weights: identical arena, and "first N" gets
the largest prefix of the block order whose bytes do not exceed the planner's
resident bytes. Modulation units sit on the disk tier for every placement (they
only feed the prologue, which is excluded from step timing). Pairs run in ABBA
order to cancel thermal drift.
"""

from __future__ import annotations

import argparse
import time

import torch

from selas.models.flux import FluxRunner, main_order, mod_order
from selas.pipeline import FluxEngine, RuntimeOptions
from selas.planner import simulate
from selas.sampling import get_noise, get_schedule, image_ids, pack, text_ids
from selas.stepcache import StepCacheConfig
from selas.store import DISK, HOST, VRAM, WeightStore
from selas.util import align_up, human_bytes

from .common import PROMPTS, embed, gpu_state, mean_after_first, save_result


def timed_run(eng, cycle_tiers, arena, staging, res, steps, txt, pooled, label) -> dict:
    cfg = eng.cfg
    tiers = {"globals": VRAM, **cycle_tiers, **{m: DISK for m in mod_order(cfg)}}
    l_img = (res // 16) ** 2
    sig = get_schedule(steps, l_img)
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    with WeightStore(eng.tc, tiers, eng.device, eng.dtype, arena_bytes=arena, staging_bytes=staging,
                     profile=True, label=label) as st:
        st.wait_loaded()  # the store loads in the background; time the whole load
        load_s = time.perf_counter() - t0
        r = FluxRunner(st, cfg, eng.dtype)
        cond = r.prepare(sig, 3.5, pooled[None], txt[None], image_ids(res, res), text_ids(txt.shape[0]))
        x = pack(get_noise(0, res, res)).to(eng.device)
        comp = torch.cuda.current_stream(eng.device)
        torch.cuda.synchronize(eng.device)
        torch.cuda.reset_peak_memory_stats(eng.device)
        step_s, stall_s, h2d = [], [], []
        r.begin()
        try:
            for s in range(steps):
                st.stats.reset()
                ts = time.perf_counter()
                pred, _ = r.step(s, x, cond)
                x = x + (sig[s + 1] - sig[s]) * pred
                comp.synchronize()
                step_s.append(time.perf_counter() - ts)
                stall_s.append(st.stats.stall_seconds())
                h2d.append(sum(st.stats.h2d_bytes.values()))
        finally:
            r.end()
        peak = torch.cuda.max_memory_allocated(eng.device)
        assert torch.isfinite(x).all(), "non-finite latents"
    out = {
        "label": label, "res": res, "load_s": load_s, "step_s": step_s, "stall_s": stall_s,
        "mean_step_s": mean_after_first(step_s), "mean_stall_s": mean_after_first(stall_s),
        "h2d_per_step": mean_after_first(h2d), "peak_vram": peak, **gpu_state(),
    }
    print(f"  {label:<10} {res}²  step {out['mean_step_s']:.3f} s  stall {out['mean_stall_s'] * 1e3:7.1f} ms/step  "
          f"h2d {human_bytes(out['h2d_per_step'])}/step  peak {human_bytes(peak)}  ({out.get('temp_c')}°C {out.get('sm_mhz')} MHz)",
          flush=True)
    return out


def predicted(eng, cycle_tiers, arena, res) -> float:
    main = main_order(eng.cfg)
    l_img, l_txt = (res // 16) ** 2, int(eng.defaults.get("max_t5_tokens", 512))
    size = {n: align_up(eng.tc.units[n].nbytes, 4096) for n in main}
    comp = {n: eng._unit_seconds(n, l_img, l_txt, 1) for n in main}
    return simulate(main, cycle_tiers, size, comp, arena, 0, eng.hw.h2d_bw, eng.hw.disk_bw)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0, 5.0], help="VRAM GiB (0 = auto)")
    ap.add_argument("--steps", type=int, default=0, help="0: 8 at <=768², else 6")
    a = ap.parse_args()

    eng = FluxEngine(a.model, RuntimeOptions())
    cfg, units = eng.cfg, eng.tc.units
    main_units = main_order(cfg)
    txt, pooled = embed(eng, PROMPTS[:1])[PROMPTS[0]]
    max_mod = max(units[m].nbytes for m in mod_order(cfg))
    staging = 2 * max_mod
    runs, configs = [], []
    for res in a.res:
        steps = a.steps or (8 if res <= 768 else 6)
        for budget in a.budgets:
            eng.rt.vram_gb = budget or None
            plan = eng.make_plan(res, res, 1, steps, StepCacheConfig())
            auto = {n: plan.tiers[n] for n in main_units}
            res_bytes = sum(units[n].nbytes for n in main_units if auto[n] == VRAM)
            first, used = {}, 0
            for n in main_units:
                fits = used + units[n].nbytes <= res_bytes and all(t == VRAM for t in first.values())
                first[n] = VRAM if fits else HOST
                used += units[n].nbytes if fits else 0
            arena = plan.arena_bytes
            cfg_rec = {
                "res": res, "budget_gb": budget or "auto", "arena": arena,
                "auto_map": "".join({VRAM: "V", HOST: "h", DISK: "d"}[auto[n]] for n in main_units),
                "auto_resident_bytes": res_bytes, "first_n": sum(1 for t in first.values() if t == VRAM),
                "first_resident_bytes": used,
                "pred_auto_s": predicted(eng, auto, arena, res), "pred_first_s": predicted(eng, first, arena, res),
            }
            configs.append(cfg_rec)
            print(f"{res}² budget {budget or 'auto'}: auto {cfg_rec['auto_map']} ({human_bytes(res_bytes)}), "
                  f"first:{cfg_rec['first_n']} ({human_bytes(used)}), arena {human_bytes(arena)}; predicted "
                  f"{cfg_rec['pred_auto_s']:.3f} vs {cfg_rec['pred_first_s']:.3f} s", flush=True)
            for label, tiers in (("auto", auto), ("first:N", first), ("first:N", first), ("auto", auto)):
                rec = timed_run(eng, tiers, arena, staging, res, steps, txt, pooled, label)
                runs.append({**rec, "budget_gb": budget or "auto"})
        # Everything streamed through a 2-block arena: the minimum-VRAM configuration.
        allh = {n: HOST for n in main_units}
        arena = align_up(2 * max(units[n].nbytes for n in main_units), 4096)
        rec = timed_run(eng, allh, arena, staging, res, steps, txt, pooled, "stream-all")
        runs.append({**rec, "budget_gb": "stream-all", "pred_s": predicted(eng, allh, arena, res), "arena": arena})
    eng.close()

    print("\nsummary (mean step after the first; ABBA pairs averaged)")
    for c in configs:
        sel = [r for r in runs if r["res"] == c["res"] and r["budget_gb"] == c["budget_gb"]]
        for lab in ("auto", "first:N"):
            rs = [r for r in sel if r["label"] == lab]
            c[f"{lab}_step_s"] = sum(r["mean_step_s"] for r in rs) / len(rs)
            c[f"{lab}_stall_s"] = sum(r["mean_stall_s"] for r in rs) / len(rs)
        print(f"  {c['res']}² budget {c['budget_gb']}: auto {c['auto_step_s']:.3f} s (stall {c['auto_stall_s'] * 1e3:.0f} ms) "
              f"vs first:{c['first_n']} {c['first:N_step_s']:.3f} s (stall {c['first:N_stall_s'] * 1e3:.0f} ms) "
              f"-> {100 * (c['first:N_step_s'] / c['auto_step_s'] - 1):+.1f}% for first:N")
    for r in runs:
        if r["label"] == "stream-all":
            print(f"  {r['res']}² stream-all: {r['mean_step_s']:.3f} s (stall {r['mean_stall_s'] * 1e3:.0f} ms), "
                  f"predicted {r['pred_s']:.3f} s, peak VRAM {human_bytes(r['peak_vram'])}")
    save_result("placement", {"model": a.model, "configs": configs, "runs": runs})


if __name__ == "__main__":
    main()
