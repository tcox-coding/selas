# Changelog

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
