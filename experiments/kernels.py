"""Kernel options: eager vs fused (torch.compile) vs fused + fp16 accumulation (docs/PLAN.md §7).

One engine, weights loaded once; only the kernels change between runs. Timing:
prompt 0 in ABBA order (eager, fused, acc16, acc16, fused, eager) so GPU warm-up
and thermal throttling cancel. Quality: every prompt once per config, PSNR of
the decoded image against the eager one (eager is the reference numerics).
"""

from __future__ import annotations

import argparse

import torch

from selas.models.flux import Kernels, compile_available
from selas.pipeline import FluxEngine, Job, RuntimeOptions

from .common import IMAGES, PROMPTS, ROOT, gpu_state, psnr, save_result

CONFIGS = {"eager": (False, False), "fused": (True, False), "fused+acc16": (True, True)}


def set_config(eng: FluxEngine, name: str) -> None:
    compiled, acc16 = CONFIGS[name]
    eng.kernels = Kernels(compiled=compiled) if compiled != eng.kernels.compiled else eng.kernels
    if eng.runner is not None:
        eng.runner.k = eng.kernels
    eng.fp16_accum = acc16


def run(eng, prompt_i: int, name: str, res: int, steps: int) -> dict:
    set_config(eng, name)
    r = eng.generate([Job(PROMPTS[prompt_i], seed=0, width=res, height=res, steps=steps)])[0]
    path = IMAGES / f"kernels_{name}_p{prompt_i}_{res}.png"
    r.image.save(path)
    st = r.stats
    rec = {"config": name, "prompt": prompt_i, "image": str(path.relative_to(ROOT)), "step_s": st["step_s"],
           "mean_step_s": st["mean_full_step_s"], "first_step_s": st["step_s"][0], "peak_vram": st["peak_vram"],
           **gpu_state(), "_img": r.image}
    print(f"{name:12s} p{prompt_i}: mean step {rec['mean_step_s']:.3f} s (first {rec['first_step_s']:.2f} s), "
          f"peak VRAM {rec['peak_vram'] / 2**30:.2f} GiB, {rec.get('temp_c')} °C {rec.get('sm_mhz')} MHz", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    eng = FluxEngine(a.model, RuntimeOptions(keep_loaded=True))
    assert compile_available(eng.device) and eng.dtype == torch.float16
    names = list(CONFIGS)
    rows = []
    try:
        for name in names + names[::-1]:
            rows.append(run(eng, 0, name, a.res, a.steps))
        for i in range(1, len(PROMPTS)):
            for name in names:
                rows.append(run(eng, i, name, a.res, a.steps))
    finally:
        eng.close()
    ref = {r["prompt"]: r["_img"] for r in rows if r["config"] == "eager"}
    for r in rows:
        r["psnr"] = psnr(r["_img"], ref[r["prompt"]])
    summary = []
    print(f"\n{'config':<12} {'mean step (ABBA, p0)':>21} {'vs eager':>9}  PSNR vs eager per prompt (dB)")
    base = None
    for name in names:
        t = [r["mean_step_s"] for r in rows if r["config"] == name and r["prompt"] == 0]
        mean_t = sum(t) / len(t)
        base = base or mean_t
        ps = [next(r["psnr"] for r in rows if r["config"] == name and r["prompt"] == i) for i in range(len(PROMPTS))]
        summary.append({"config": name, "mean_step_s": mean_t, "speedup": base / mean_t, "psnr": ps})
        print(f"{name:<12} {mean_t:>19.3f} s {base / mean_t:>8.2f}x  {[round(p, 2) for p in ps]}")
    for r in rows:
        r.pop("_img")
    save_result(f"kernels_{a.res}", {"model": a.model, "steps": a.steps, "prompts": PROMPTS, "summary": summary, "runs": rows})


if __name__ == "__main__":
    main()
