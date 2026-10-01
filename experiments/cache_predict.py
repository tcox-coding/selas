"""Step cache: linear residual extrapolation vs reuse at an equal skip rate (docs/PLAN.md §7).

For each prompt: an exact run (no cache), then fbcache at each threshold with
``predict=reuse`` and ``predict=linear``. The skip decision depends on the probe
distance, which can drift once predictions differ; when the free-running linear
schedule differs from reuse's, linear is also run on reuse's exact schedule
(replayed), so the comparison at equal skip rate is exact. Quality is PSNR of the
decoded image vs the exact image.
"""

from __future__ import annotations

import argparse

import selas.pipeline as pipeline
from selas.pipeline import FluxEngine, Job, RuntimeOptions
from selas.stepcache import FULL, StepCache, StepCacheConfig

from .common import IMAGES, PROMPTS, ROOT, psnr, save_result


def replaying(schedules: list[list[str]]):
    """A StepCache class whose i-th instance follows ``schedules[i]`` instead of its threshold."""
    queue = list(schedules)

    class Replay(StepCache):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.schedule = queue.pop(0)

        def decide(self, step, sigma, probe_residual):
            d = None
            if self.ref_probe is not None:
                ref = self.ref_probe
                d = float((probe_residual - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12))
            dec = self.schedule[step]
            if dec == FULL:
                self.ref_probe = probe_residual.detach().float()
                self.consecutive = 0
            else:
                self.consecutive += 1
            self.log.append((step, dec, d))
            return dec, d

    return Replay


def run(eng, prompts, res, steps, cfg: StepCacheConfig, tag: str, replay=None) -> list[dict]:
    jobs = [Job(p, seed=0, width=res, height=res, steps=steps) for p in prompts]
    pipeline.StepCache = replaying(replay) if replay else StepCache
    try:
        results = eng.generate(jobs, cfg, batch_size=1)
    finally:
        pipeline.StepCache = StepCache
    out = []
    for i, r in enumerate(results):
        path = IMAGES / f"cache_{tag}_p{i}_{res}.png"
        r.image.save(path)
        st = r.stats
        out.append({"tag": tag, "prompt": i, "image": str(path.relative_to(ROOT)), "decisions": st["decisions"],
                    "skipped": sum(d != "full" for d in st["decisions"]),
                    "denoise_s": st["prologue_s"] + sum(st["step_s"]), "_img": r.image})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux1-dev")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.08, 0.15])
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    eng = FluxEngine(a.model, RuntimeOptions())
    rows = []
    try:
        exact = run(eng, PROMPTS, a.res, a.steps, StepCacheConfig(), "exact")
        rows += exact
        for thr in a.thresholds:
            reuse = run(eng, PROMPTS, a.res, a.steps, StepCacheConfig("fbcache", threshold=thr, predict="reuse"), f"reuse{thr}")
            lin = run(eng, PROMPTS, a.res, a.steps, StepCacheConfig("fbcache", threshold=thr, predict="linear"), f"linear{thr}")
            rows += reuse + lin
            if any(r["decisions"] != l["decisions"] for r, l in zip(reuse, lin)):
                rows += run(eng, PROMPTS, a.res, a.steps, StepCacheConfig("fbcache", threshold=thr, predict="linear"),
                            f"linear{thr}-replay", replay=[r["decisions"] for r in reuse])
    finally:
        eng.close()
    for r in rows:
        r["psnr"] = psnr(r["_img"], exact[r["prompt"]]["_img"]) if r["tag"] != "exact" else float("inf")
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    print(f"\n{'config':<22} {'skipped/28 per prompt':<24} {'PSNR per prompt (dB)':<28} mean PSNR  denoise s")
    summary = []
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        mean_psnr = sum(r["psnr"] for r in rs) / len(rs)
        mean_t = sum(r["denoise_s"] for r in rs) / len(rs)
        summary.append({"tag": t, "skipped": [r["skipped"] for r in rs], "psnr": [r["psnr"] for r in rs],
                        "mean_psnr": mean_psnr, "mean_denoise_s": mean_t})
        print(f"{t:<22} {str([r['skipped'] for r in rs]):<24} {str([round(r['psnr'], 2) for r in rs]):<28} "
              f"{mean_psnr:8.2f}  {mean_t:8.1f}")
    for r in rows:
        r.pop("_img")
    save_result(f"cache_predict_{a.res}", {"model": a.model, "steps": a.steps, "prompts": PROMPTS,
                                           "summary": summary, "runs": rows})


if __name__ == "__main__":
    main()
