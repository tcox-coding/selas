"""Per-step error of each kernel option against an fp32 reference (docs/PLAN.md §7).

Image PSNR against the eager image mixes two things: how wrong a step is, and how
chaotic the prompt's trajectory is (a rounding-level change can tip some prompts
into a different but equally valid image). This measures the first alone: take
the eager trajectory's latents at a few steps, run *one* step from each with
eager / fused / fused+acc16 (fp16 compute) and with fp32 compute and eager
kernels as the reference, and report the relative L2 error of the velocity.
"""

from __future__ import annotations

import argparse

import torch

from selas.models.flux import EAGER, Kernels
from selas.pipeline import FluxEngine, RuntimeOptions
from selas.sampling import get_noise, get_schedule, image_ids, pack, text_ids
from selas.stepcache import StepCacheConfig

from .common import PROMPTS, embed, save_result


def predictions(eng: FluxEngine, txt, pooled, res: int, steps: int, at: list[int], variants: dict, trajectory=None):
    """{variant: {step: velocity}} from the latents in ``trajectory`` (or, if None, record eager's)."""
    eng.ensure_loaded(res, res, 1, steps, StepCacheConfig())
    eng.store.wait_loaded()
    eng._join_warm()  # no compiling inside the comparison
    r = eng.runner
    l_img = (res // 16) ** 2
    sig = get_schedule(steps, l_img, shift=True)
    cond = r.prepare(sig, 3.5, pooled[None], txt[None], image_ids(res, res), text_ids(txt.shape[0]))
    if trajectory is None:
        trajectory = {}
        x = pack(get_noise(0, res, res)).to(eng.device)
        r.k = EAGER
        for s in range(max(at) + 1):
            if s in at:
                trajectory[s] = x.clone()
            pred, _ = r.step(s, x, cond)
            x = x + (sig[s + 1] - sig[s]) * pred
    out = {}
    for name, (kernels, acc16) in variants.items():
        r.k = kernels
        prev = torch.backends.cuda.matmul.allow_fp16_accumulation
        torch.backends.cuda.matmul.allow_fp16_accumulation = acc16
        try:
            out[name] = {s: r.step(s, trajectory[s], cond)[0].float().cpu() for s in at}
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = prev
    r.end()
    return out, trajectory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--at", type=int, nargs="+", default=[0, 5, 10, 15, 20, 26])
    ap.add_argument("--prompts", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    fused = Kernels(compiled=True)
    eng16 = FluxEngine(a.model, RuntimeOptions(keep_loaded=True))
    embeds = embed(eng16, PROMPTS)
    preds, trajs = {}, {}
    try:
        for i in a.prompts:
            txt, pooled = embeds[PROMPTS[i]]
            preds[i], trajs[i] = predictions(eng16, txt, pooled, a.res, a.steps, a.at,
                                             {"eager": (EAGER, False), "fused": (fused, False), "fused+acc16": (fused, True)})
    finally:
        eng16.close()
    eng32 = FluxEngine(a.model, RuntimeOptions(keep_loaded=True, dtype="fp32", compile=False))
    try:
        for i in a.prompts:
            txt, pooled = embeds[PROMPTS[i]]
            ref, _ = predictions(eng32, txt, pooled, a.res, a.steps, a.at, {"fp32": (EAGER, False)}, trajs[i])
            preds[i]["fp32"] = ref["fp32"]
    finally:
        eng32.close()
    rows = []
    print(f"\nrelative L2 error of one step's velocity vs fp32 (eager kernels), {a.res}², steps {a.at}")
    for name in ("eager", "fused", "fused+acc16"):
        errs = {i: [float((preds[i][name][s] - preds[i]["fp32"][s]).norm() / preds[i]["fp32"][s].norm()) for s in a.at]
                for i in a.prompts}
        flat = [e for v in errs.values() for e in v]
        rows.append({"config": name, "rel_err": errs, "mean_rel_err": sum(flat) / len(flat), "max_rel_err": max(flat)})
        print(f"{name:12s} mean {rows[-1]['mean_rel_err']:.2e}  max {rows[-1]['max_rel_err']:.2e}  "
              + "  ".join(f"p{i}: " + " ".join(f"{e:.1e}" for e in v) for i, v in errs.items()))
    for i in a.prompts:  # how far apart the fp16 variants are from each other, for scale
        d = [float((preds[i]["fused"][s] - preds[i]["eager"][s]).norm() / preds[i]["eager"][s].norm()) for s in a.at]
        print(f"p{i} fused vs eager: " + " ".join(f"{e:.1e}" for e in d))
    save_result(f"kernel_error_{a.res}", {"model": a.model, "steps": a.steps, "at": a.at, "prompts": [PROMPTS[i] for i in a.prompts],
                                          "summary": rows})


if __name__ == "__main__":
    main()
