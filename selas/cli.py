"""``selas`` command line: info | convert | plan | generate | bench | verify | compare."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from . import util
from .util import GiB, human_bytes, log


def _add_runtime(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("runtime / placement")
    g.add_argument("--model", required=True, help="converted model directory")
    g.add_argument("--vram-gb", type=float, help="VRAM budget (default: free VRAM minus a margin)")
    g.add_argument("--ram-gb", type=float, help="pinned-RAM budget (default: available RAM minus 4 GiB)")
    g.add_argument("--reserve-gb", type=float, help="override the activation-memory reserve (normally learned from earlier runs)")
    g.add_argument("--placement", default="auto", help="auto | vram | host | disk | first:N (A/B baselines)")
    g.add_argument("--micro-batch", type=int, help="images per block forward (block-major batching)")
    g.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="compute dtype")
    g.add_argument("--direct-io", action=argparse.BooleanOptionalAction, default=True,
                   help="O_DIRECT reads for the disk tier (default; --no-direct-io uses the page cache)")
    g.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="fuse the elementwise kernels with torch.compile (default on CUDA; --no-compile runs eager PyTorch)")
    g.add_argument("--fp16-accum", action="store_true",
                   help="fp16 matmuls accumulate in fp16: ~1.6x faster matmuls on GeForce cards, approximate")
    g.add_argument("--profile", action="store_true", help="measure transfer stalls and per-block compute")
    g.add_argument("--device", type=int, help="CUDA device index")


def _add_cache(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("step cache (approximate, opt-in)")
    g.add_argument("--cache", default="none", choices=["none", "fbcache", "tiered"])
    g.add_argument("--cache-threshold", type=float, default=0.08)
    g.add_argument("--cache-predict", default="linear", choices=["reuse", "linear"],
                   help="residual on approximated steps: extrapolate from the last two full steps (default) or reuse the last")
    g.add_argument("--cache-warmup", type=int, default=1)
    g.add_argument("--cache-tail", type=int, default=1, help="final steps always computed")
    g.add_argument("--cache-max-consecutive", type=int, default=3)


def _cache_cfg(a):
    from .stepcache import StepCacheConfig

    return StepCacheConfig(a.cache, a.cache_threshold, a.cache_predict, a.cache_warmup, a.cache_tail, a.cache_max_consecutive)


def _runtime(a):
    from .pipeline import RuntimeOptions

    return RuntimeOptions(
        device=a.device, dtype=a.dtype, vram_gb=a.vram_gb, ram_gb=a.ram_gb, reserve_gb=a.reserve_gb,
        direct_io=a.direct_io, profile=a.profile, micro_batch=a.micro_batch, placement=a.placement,
        compile=a.compile, fp16_accum=a.fp16_accum,
        vae_tile=getattr(a, "vae_tile", "auto"), prompt_cache=not getattr(a, "no_prompt_cache", False),
    )


# --------------------------------------------------------------------------- commands


def cmd_info(a) -> int:
    import torch

    from .hw import device_info, ram_available

    print(f"selas {__version__} | torch {torch.__version__} (CUDA {torch.version.cuda}) | python {sys.version.split()[0]}")
    total, avail = ram_available()
    print(f"RAM   {human_bytes(total)} total, {human_bytes(avail)} available")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            d = device_info(torch.device("cuda", i))
            print(f"GPU{i}  {d.name} sm_{d.capability[0]}{d.capability[1]}  {human_bytes(d.free)} free / {human_bytes(d.total)}"
                  f"  native bf16: {'yes' if d.bf16_native else 'no (compute in fp16)'}")
    else:
        print("GPU   none visible to torch")
    try:
        import triton  # noqa: F401

        print(f"triton {triton.__version__} (NF4 fast path; self-checked on first use)")
    except Exception:
        print("triton not available (NF4 uses the PyTorch fallback)")
    if a.model:
        from .container import Container
        from .util import read_json

        info = read_json(Path(a.model) / "model.json", {})
        print(f"model {a.model}: FLUX.1-{info.get('variant', '?')}, defaults {info.get('defaults')}")
        for comp, sub in info.get("components", {}).items():
            c = Container(Path(a.model) / sub)
            print(f"  {comp:12s} {c.describe()}  [{c.config.get('quant', '')}]")
            c.close()
    return 0


def cmd_convert(a) -> int:
    import torch

    from .convert import ConvertOptions, convert_model

    if a.dtype == "auto":
        dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)) else torch.float16
    else:
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    opts = ConvertOptions(dtype=dt, quant=a.quant, hash_units=not a.no_hash)
    comps = tuple(a.only.split(",")) if a.only else ("transformer", "t5", "clip", "vae")
    convert_model(a.out, a.flux, a.t5, a.clip, a.vae, a.t5_tokenizer, a.clip_tokenizer, opts, comps)
    return 0


def cmd_plan(a) -> int:
    from .models.flux import main_order
    from .pipeline import FluxEngine

    eng = FluxEngine(a.model, _runtime(a))
    try:
        plan = eng.make_plan(a.width, a.height, a.batch_size, a.steps or int(eng.defaults.get("steps", 28)), _cache_cfg(a))
        print(plan.describe(main_order(eng.cfg)))
        print("legend: V = VRAM-resident, h = pinned RAM (streamed), d = disk (streamed)")
    finally:
        eng.close()
    return 0


def _read_prompts(a) -> list[str]:
    prompts = list(a.prompt or [])
    if a.prompt_file:
        with open(a.prompt_file) as f:
            prompts += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not prompts:
        raise SystemExit("no prompt given (-p or --prompt-file)")
    return prompts


def cmd_generate(a) -> int:
    from PIL.PngImagePlugin import PngInfo

    from .pipeline import FluxEngine, Job, png_metadata

    prompts = _read_prompts(a)
    jobs = [Job(p, a.seed + k, a.width, a.height, a.steps, a.guidance) for p in prompts for k in range(a.count)]
    cache_cfg = _cache_cfg(a)
    log(f"{len(jobs)} image(s), step cache: {cache_cfg.describe()}")
    eng = FluxEngine(a.model, _runtime(a))
    try:
        results = eng.generate(jobs, cache_cfg, batch_size=a.batch_size)
    finally:
        eng.close()
    out = Path(a.out)
    single = len(results) == 1 and out.suffix.lower() == ".png"
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    numerics = f", Kernels: {'fused' if eng.kernels.compiled else 'eager'}"
    if eng.fp16_accum:
        numerics += ", Matmul accumulation: fp16 (approximate)"
    for i, r in enumerate(results):
        path = out if single else out / f"selas_{r.job.seed}_{i:03d}.png"
        meta = PngInfo()
        for k, v in png_metadata(r.job, eng.defaults, cache_cfg, numerics).items():
            meta.add_text(k, v)
        r.image.save(path, pnginfo=meta)
        print(path)
    return 0


def cmd_bench(a) -> int:
    import torch

    from .hw import (HwProfile, default_compute_dtype, disk_bandwidth, load_profile, measure_attention, measure_h2d,
                     measure_matmul, save_profile)

    dev = torch.device("cuda", a.device or 0)
    prof: HwProfile = load_profile(dev, quick_measure=False)
    dt = default_compute_dtype(dev)
    prof.h2d_bw = measure_h2d(dev, 1 * GiB)
    print(f"H2D pinned       {human_bytes(prof.h2d_bw)}/s")
    prof.matmul_flops = measure_matmul(dev, dt, 8192)
    print(f"GEMM {str(dt)[6:]:9s}  {prof.matmul_flops / 1e12:.1f} TFLOP/s")
    prof.attn_flops = measure_attention(dev, dt)
    print(f"SDPA {str(dt)[6:]:9s}  {prof.attn_flops / 1e12:.1f} TFLOP/s (L=4096, 24 heads)")
    measured = {"h2d_bw", "matmul_flops", "attn_flops"}
    if a.model:
        from .util import read_json

        info = read_json(Path(a.model) / "model.json", {})
        path = Path(a.model) / info["components"]["transformer"] / "weights.bin"
        for direct in (False, True):  # cold reads; also refreshes the per-disk cache the engine uses
            bw = disk_bandwidth(path, direct=direct, remeasure=True)
            print(f"disk {'O_DIRECT' if direct else 'buffered'} {'' if direct else '(cold)':6s} "
                  f"{human_bytes(bw) + '/s' if bw else 'unavailable'}")
    prof.measured = tuple(sorted(set(prof.measured) | measured))
    save_profile(dev, prof)
    print("saved to ~/.cache/selas/hw.json")
    return 0


def cmd_verify(a) -> int:
    from .container import Container
    from .util import read_json

    info = read_json(Path(a.model) / "model.json", {})
    bad_total = 0
    for comp, sub in info.get("components", {}).items():
        c = Container(Path(a.model) / sub)
        bad = c.verify()
        bad_total += len(bad)
        print(f"{comp:12s} {'OK' if not bad else 'CORRUPT: ' + ', '.join(bad)}")
        c.close()
    return 1 if bad_total else 0


def cmd_compare(a) -> int:
    import numpy as np
    from PIL import Image

    x = np.asarray(Image.open(a.a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(a.b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape:
        print(f"size mismatch {x.shape} vs {y.shape}")
        return 1
    mse = float(((x - y) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(f"PSNR {psnr:.2f} dB | mean abs diff {np.abs(x - y).mean():.3f} | max abs diff {np.abs(x - y).max():.0f}")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="selas", description="Tiered-memory FLUX inference: run models larger than your VRAM.")
    ap.add_argument("--version", action="version", version=f"selas {__version__}")
    ap.add_argument("-q", "--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="hardware and model summary")
    p.add_argument("--model")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("convert", help="convert checkpoints into a selas model directory")
    p.add_argument("--flux", nargs="+", required=True, help="transformer checkpoint(s) or folder; may be an all-in-one file")
    p.add_argument("--t5", nargs="+", help="T5-XXL checkpoint(s) (default: look inside --flux)")
    p.add_argument("--clip", nargs="+", help="CLIP-L checkpoint (default: look inside --flux)")
    p.add_argument("--vae", nargs="+", help="ae.safetensors (default: look inside --flux)")
    p.add_argument("--t5-tokenizer", help="directory with the T5 tokenizer (tokenizer.json or spiece.model)")
    p.add_argument("--clip-tokenizer", help="directory with the CLIP tokenizer (vocab.json + merges.txt)")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="storage dtype for float transformer weights")
    p.add_argument("--quant", default="none", choices=["none", "int8", "nf4"], help="weight-only quantization (APPROXIMATE)")
    p.add_argument("--only", help="comma-separated subset of transformer,t5,clip,vae")
    p.add_argument("--no-hash", action="store_true", help="skip per-unit BLAKE2 hashes")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("plan", help="show the placement plan and predicted step time")
    _add_runtime(p)
    _add_cache(p)
    p.add_argument("-W", "--width", type=int, default=1024)
    p.add_argument("-H", "--height", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("generate", help="generate images")
    _add_runtime(p)
    _add_cache(p)
    p.add_argument("-p", "--prompt", action="append")
    p.add_argument("--prompt-file")
    p.add_argument("-W", "--width", type=int, default=1024)
    p.add_argument("-H", "--height", type=int, default=1024)
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=1, help="images per prompt (seeds seed, seed+1, ...)")
    p.add_argument("--batch-size", type=int, default=1, help="images advanced together through each loaded block")
    p.add_argument("-o", "--out", default="outputs")
    p.add_argument("--vae-tile", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--no-prompt-cache", action="store_true")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("bench", help="measure H2D, GEMM, attention and disk throughput for the planner")
    p.add_argument("--model")
    p.add_argument("--device", type=int)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("verify", help="re-hash every unit of a converted model")
    p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("compare", help="PSNR between two images")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.quiet:
        util.VERBOSITY = 0
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # compiled kernels: a cold compile is ~16 s per new image size, a cached one ~3 s (hidden behind
    # loading); PyTorch's default cache lives in /tmp, which the OS may clean
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(util.user_cache_dir() / "inductor"))
    return int(a.fn(a) or 0)


__all__ = ["main", "build_parser"]
