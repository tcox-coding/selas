"""Block-major micro-batching at 512²: images/s vs batch size (docs/PLAN.md §7).

At 512² a step does ~1,250 FLOPs per weight byte, below the ~2,500 this
GPU + PCIe 3.0 needs to hide streaming, so batch 1 is transfer-bound. A batch
of B images passes every block while it is loaded, multiplying FLOPs per
streamed byte by B. ``micro`` < B splits the batch inside each block forward
(less activation memory, same weight traffic).

Runs through the engine (planner, prefetch, learned calibration). The learned
calibration file is restored afterwards; the scale each run learned is
recorded.
"""

from __future__ import annotations

import argparse
import shutil

from selas.pipeline import FluxEngine, Job, RuntimeOptions
from selas.store import DISK, HOST, VRAM
from selas.util import read_json

from .common import IMAGES, PROMPTS, save_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--configs", default="1:1,2:2,4:4,4:1,8:8,8:2", help="batch:micro,...")
    ap.add_argument("--name", default="", help="result file suffix (e.g. 'reverse' for a drift-cancelling second pass)")
    a = ap.parse_args()

    eng = FluxEngine(a.model, RuntimeOptions(profile=True))
    prof = eng.dir / ".selas" / "profile.json"
    backup = prof.with_suffix(".json.bak")
    if prof.exists():
        shutil.copy2(prof, backup)
    IMAGES.mkdir(parents=True, exist_ok=True)
    runs = []
    try:
        for spec in a.configs.split(","):
            b, micro = (int(v) for v in spec.split(":"))
            eng.rt.micro_batch = micro
            jobs = [Job(PROMPTS[0], seed=i, width=a.res, height=a.res, steps=a.steps) for i in range(b)]
            res = eng.generate(jobs, batch_size=b)
            res[0].image.save(IMAGES / f"batch{b}_micro{micro}_{a.res}.png")
            st = res[0].stats
            main = [n for n in eng.plan.tiers if n.startswith(("double.", "single.")) and not n.endswith(".mod")]
            learned = read_json(prof, {}).get(eng._gpu_key, {})
            rec = {
                "batch": b, "micro": micro, "res": a.res, "steps": a.steps,
                "mean_step_s": st["mean_full_step_s"], "step_s": st["step_s"],
                "stall_s_total": st["stall_s"], "prologue_s": st["prologue_s"],
                "s_per_image_step": st["mean_full_step_s"] / b,
                "images_per_min_denoise": 60 * b / (st["mean_full_step_s"] * a.steps),
                "h2d_per_step": sum(st["h2d_bytes"].values()) / st["steps"],
                "peak_vram": st["peak_vram"], "predicted_step_s": st["predicted_step_s"],
                "map": "".join({VRAM: "V", HOST: "h", DISK: "d"}[eng.plan.tiers[n]] for n in main),
                "learned_scale": {k: v.get("scale") for k, v in learned.items()},
            }
            runs.append(rec)
            print(f"batch {b} micro {micro}: step {rec['mean_step_s']:.3f} s -> {rec['s_per_image_step']:.3f} s/image-step, "
                  f"{rec['images_per_min_denoise']:.2f} img/min (denoise), stall {rec['stall_s_total']:.2f} s total, "
                  f"resident {rec['map'].count('V')}/57, scale {rec['learned_scale']}", flush=True)
    finally:
        eng.close()
        if backup.exists():
            shutil.move(backup, prof)
    base = next((r["s_per_image_step"] for r in runs if r["batch"] == 1), runs[0]["s_per_image_step"])
    for r in runs:
        r["speedup_vs_b1"] = base / r["s_per_image_step"]
    save_result(f"batching_{a.res}{'_' + a.name if a.name else ''}", {"model": a.model, "configs": a.configs, "runs": runs})


if __name__ == "__main__":
    main()
