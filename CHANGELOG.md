# Changelog

## Unreleased

Faster steps and faster loading; measurements in `docs/PLAN.md` §7.

- Fused elementwise kernels (default on CUDA with Triton): the code between the matmuls runs as
  `torch.compile`-generated kernels, compiled in the background while the weights load. 1.20× per
  1024² step on an RTX 2080 Ti and as accurate as eager PyTorch (same per-step error against fp32),
  though not bitwise equal: `--no-compile` restores the previous numerics.
- `--fp16-accum` (opt-in, approximate): fp16 matmuls accumulate in fp16. 1.59× per step on GeForce
  Turing, 2.6× the per-step error. Named in the PNG metadata, as is the kernel mode.
- Loading pins RAM ~2× faster: pages are pre-faulted on 8 threads and registered while the weights are
  read, so loading the default 1024² plan is disk-bound (15.9 → 8.7 s).
- Modulation vectors are cached on disk (`<model>/.selas/mods`, kept under 4 GiB, least recently used
  first), keyed by each image's pooled prompt vector, guidance and step schedule. A new seed for a cached
  prompt neither loads nor streams the modulation layers (27 % of the weights): at 1024² loading takes
  6.0 instead of 7.7 s, and the first step ends ~2 s sooner. Exact: vectors are computed per image, so cached,
  fresh and batched runs give identical bits. `--no-prompt-cache` turns it off with the prompt cache.
- Denoising starts while the weights load: a background thread reads them in order of first use, and the
  prologue and the first step wait for each unit only until it arrives. The first step of a 1024² image
  ends 1.3–1.9 s sooner. A failed read is raised to whatever waits for the weights.
- The pinned pool is freed in the background, without blocking the VAE.
- The learned compute calibration is kept per numerics mode; runs that compile kernels mid-run do not
  update the learned activation memory or calibration.
- The `selas` CLI keeps compiled kernels in `~/.cache/selas/inductor` (PyTorch's default is under `/tmp`).
- New experiments: `kernels` (speed and image PSNR per kernel mode), `kernel_error` (per-step error
  against an fp32 reference).

## 0.1.0 — 2026-10-01

First release.

- Tiered weight placement for FLUX.1 across VRAM, pinned RAM and NVMe, chosen by a pipeline simulator
  (Belady-style resident set + FIFO ring arena). Placement never changes numerics: outputs are bit-identical.
- Continuous cross-step prefetch from pinned RAM and from disk (background reader, O_DIRECT by default).
- Modulation hoisting: the timestep/prompt-only layers (27 % of FLUX) are computed once per image.
- Self-calibration: per-disk read bandwidth, per-block compute time and activation memory are measured
  during normal runs and cached; an out-of-memory error under a learned reserve retries on the estimate.
- Container format with per-unit hashes; converters for BFL single files, diffusers folders and
  bitsandbytes NF4 all-in-one checkpoints (imported bit-exactly); optional int8 / NF4 quantization.
- Functional T5-XXL (streamed), CLIP-L and VAE decoder (with tiling); prompt-embedding cache.
- Optional, approximate step caching (`fbcache`, experimental `tiered`) with linear residual forecasting.
- Block-major batching and micro-batching.
- Tests: oracles against diffusers/transformers, placement bit-exactness, gated real-model tests.
- Measurements and scripts for every design hypothesis (`docs/PLAN.md` §7, `experiments/`).
