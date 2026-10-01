# selas — design plan

> **σέλας** (Greek: *light, brightness*). A tiered-memory inference engine for
> diffusion transformers. Run image models that do not fit in VRAM by *placing*
> their weights across VRAM, pinned RAM and NVMe — the way
> [colibri](https://github.com/JustVugg/colibri) does for trillion-parameter
> MoE LLMs — plus caching tricks that only exist because diffusion is iterative.

Target model: **FLUX.1-dev** (12B MMDiT). Target machine for development: RTX
2080 Ti (11 GB, Turing, PCIe 3.0 x16), 31 GB RAM, NVMe.

---

## 1. What colibri does, and why it does not transfer directly

colibri's thesis: a model doesn't need to *fit* in fast memory, it needs to be
*placed*. Weights are data staged across a VRAM / RAM / NVMe hierarchy, "a JIT
for weights". It works for MoE LLMs because:

| colibri relies on…                         | …which for a diffusion transformer is |
|--------------------------------------------|---------------------------------------|
| **Sparsity** — ~5% of params active per token (routed experts) | **Dense** — every block runs on every step |
| Routing has *learnable* structure (heat → LRU + pins) | Execution order is **fully deterministic** and **cyclic**: blocks 0…56, then 0…56 again, N times |
| Router lookahead prefetch (~72% predictable) | **100% predictable** — the whole schedule is known before step 1 |
| Decode = 1 token per weight read → I/O bound | 1,500–4,600 tokens per weight read → often **compute bound** |
| KV-cache persistence across turns          | Nothing persists across steps … except that consecutive steps are *very similar* |

So the shape of the solution changes, but the philosophy carries over intact:

* **Placement only decides speed, never semantics.** The default path is exact:
  a block computed from VRAM, from pinned RAM or from disk runs identical
  kernels on identical bytes and must give **bit-identical** output (tested).
* Every approximation (quantized containers, step caching) is **opt-in, named
  in the run banner, and measured** against the exact path.
* Every optimization is a hypothesis until an end-to-end A/B on real hardware
  says otherwise.

---

## 2. Key observations about FLUX.1-dev

Computed from the architecture (hidden 3072, 19 double + 38 single blocks):

| quantity | value |
|---|---|
| total params | 11.90 B (22.2 GiB fp16) |
| double block (excl. modulation) | 226.5 M params → 432 MiB fp16 |
| single block (excl. modulation) | 113.3 M params → 216 MiB fp16 |
| **modulation (adaLN) linears** | **3.23 B params = 27 % of the model** |
| per-step weight set w/o modulation | 16.0 GiB fp16 |
| FLOPs/step @ 512² (L=1536) | 19.8 T linear + 1.7 T attention |
| FLOPs/step @ 1024² (L=4608) | 59.5 T linear + 14.9 T attention |
| FLOPs per weight byte @ 512² / 768² / 1024² | 1,248 / 2,435 / 4,320 |

Three consequences drive the whole design:

**O1 — Streaming is cheap for DiTs at high resolution.** A 2080 Ti sustains
~30 TFLOP/s fp16 and ~12 GB/s H2D, so it needs ≈2,500 FLOPs per transferred
byte to hide the copy. At ≥768² FLUX exceeds that: if prefetch is pipelined
correctly, *streaming every block from RAM every step costs almost nothing*.
At 512² it is ~2× transfer-bound, which is where residency and caching matter.

**O2 — 27 % of the weights only depend on things known before sampling
starts.** Every block's modulation is `Linear(silu(vec))` where
`vec = f(timestep, guidance, pooled CLIP)`. The timestep schedule is fixed, so
*all* modulation vectors for *all* steps can be computed in one up-front pass
("hoisting"), streaming those 6 GiB **once per image instead of once per step**.
The result is ~2 MB per step per image. This is exact.

**O3 — Double blocks are the worst bytes to stream.** Each token only passes
through one of the two streams of a double block, so a double block has the
same FLOPs as a single block but twice the bytes. Residency should go to double
blocks first; single blocks hide their own transfer twice as well.

---

## 3. Architecture

```
            ┌──────────────── selas generate ────────────────┐
 prompts ──►│ Text phase      CLIP-L (resident, fp32)        │
            │                 T5-XXL (streamed blocks, fp32) │◄── prompt-embedding cache (disk)
            ├────────────────────────────────────────────────┤
            │ Prologue        vec_t for every step t          │
            │ (once/image)    hoisted modulation: stream the  │
            │                 57 "mod" units once → ~2 MB/step│
            ├────────────────────────────────────────────────┤
            │ Denoise loop    cyclic schedule of 57 "main"    │
            │  (N steps)      units through the TieredStore   │◄── step cache (opt-in)
            ├────────────────────────────────────────────────┤
            │ Decode          VAE decoder (resident, fp32,    │
            │                 tiled when VRAM is short)       │
            └────────────────────────────────────────────────┘

TieredStore (one per component)
  VRAM tier ── resident units, loaded once
  HOST tier ── page-locked RAM (one mmap'd region, cudaHostRegister'ed)
  DISK tier ── selas container on NVMe, pread (optionally O_DIRECT)
                  │ reader thread, FIFO staging ring (pinned)
                  ▼
  copy stream ── H2D into a VRAM *ring arena* (byte-granular, FIFO)
                  │ CUDA events: copy-done → compute waits; compute-done → copy may overwrite
                  ▼
  compute stream ─ functional block code reads weights as views into the arena
```

### 3.1 Container format (`selas convert`)

colibri converts models into its own container so an expert is one `pread`.
selas does the same at **unit** granularity (a unit = everything one block
needs):

* `manifest.json` + one `weights.bin`; each unit is a contiguous 4 KiB-aligned
  byte range, each tensor part inside it is 256-byte aligned.
* Loading a unit = one read + one `cudaMemcpyAsync`; the block's tensors are
  zero-copy *views* into that byte range on the GPU.
* Units for FLUX: `globals` (embedders + final layer, always resident),
  `double.{i}` / `single.{i}` (per-step "main" units), `double.{i}.mod` /
  `single.{i}.mod` (prologue-only modulation units).
* Each tensor has a **codec**: `raw` (any dtype incl. fp8), `int8_row`
  (per-output-channel absmax), `nf4` (bitsandbytes-compatible, blocksize 64).
  Codecs decode *just in time*, one Linear at a time, so a quantized unit never
  exists fully dequantized.
* Sources: BFL single-file (`flux1-dev.safetensors`), diffusers transformer
  folders, Forge all-in-one NF4 checkpoints (bnb NF4 imported bit-exactly),
  HF/ComfyUI T5 & CLIP files, BFL `ae.safetensors`.
* Units carry a BLAKE2 hash (`selas verify`).

### 3.2 Placement: Belady, not LRU

The access pattern is a cyclic scan of 57 units. **LRU (and FIFO) achieve a 0 %
hit rate on a cyclic scan larger than the cache**, so colibri's LRU/heat
machinery would be actively harmful. With perfect knowledge of the future, the
optimal policy (Belady/MIN) for a cyclic scan is to *pin a fixed subset and
stream the rest through a small FIFO buffer* — which is exactly the
VRAM-resident set + ring arena.

The **planner** chooses that subset with a pipeline simulator:

* cost model per unit: bytes, compute time (FLOP model calibrated by a quick
  matmul benchmark, later replaced by measured per-block times recorded in the
  model's `.selas/profile.json` — the analogue of colibri's `.coli_usage`
  learning), H2D bandwidth (measured), disk bandwidth;
* simulator: one copy engine, one disk reader, byte-accurate ring arena and
  staging ring, FIFO frees on compute completion; steady state measured over
  the 2nd/3rd cycle so cross-step prefetch is included;
* greedy: repeatedly make resident the unit with the best simulated
  gain-per-byte (candidates per unit kind, spread evenly through the cycle so
  resident compute "covers" streamed transfers), then fit the remaining main
  units into the pinned-RAM budget (overflow → disk, chosen by least loss per
  byte), then modulation units into leftover RAM;
* tries several arena sizes (2–4× the largest streamed unit) and keeps the best;
* prints the predicted step time and its breakdown; after a run, prints the
  measured one next to it.

### 3.3 Streaming executor

* **Ring arena**: one VRAM buffer, byte-granular FIFO allocation with wrap.
  A slot is reclaimed by recording a CUDA event after the last kernel that used
  it; the copy stream waits on that event before overwriting. No CPU syncs on
  the hot path.
* **Prefetch is continuous across step boundaries**: the schedule is cyclic,
  so while step *t* runs its last blocks, step *t+1*'s first streamed blocks
  are already arriving.
* **Disk tier**: a reader thread `pread`s units in schedule order into a pinned
  staging ring; the H2D copy is issued as soon as both the bytes and arena
  space are ready. Reads use `O_DIRECT` by default (measured 3.28 vs 2.19 GiB/s cold and
  1.9× faster end to end when RAM is short, §7); `--no-direct-io` uses the page cache.
* Pinned memory is one `mmap` region registered with `cudaHostRegister`
  (PyTorch's pinned allocator rounds to powers of two, which would waste up to
  2× RAM at these sizes).

### 3.4 Exact compute path on Turing

* Turing has no native bf16, so compute is **fp16 with an fp32 residual
  stream**: matmuls/attention in fp16, residual adds, norms, RoPE and the
  sampler in fp32; Linear outputs feeding a residual are clamped to the fp16
  range (the same overflow guard diffusers uses for FLUX in fp16). On Ampere+
  the default is bf16. The banner states which.
* T5-XXL runs in **fp32** (it overflows in fp16), streamed block-by-block like
  the transformer; CLIP and VAE in fp32.
* Attention via PyTorch SDPA (memory-efficient kernel on Turing).

### 3.5 Caching — the answer to "can previously-used nodes be cached?"

Yes, at four levels:

| level | what is cached | exact? |
|---|---|---|
| **Weights** | resident set (Belady), host tier, OS page cache | exact |
| **Condition-only nodes** | modulation vectors for all steps, `txt_in(T5)`, RoPE tables, time/guidance embeddings — computed once per image | exact |
| **Prompts** | T5 + CLIP outputs on disk keyed by prompt + model hash: re-seeding a prompt never loads T5 | exact |
| **Activations across steps** | block residuals reused on steps where the model's output barely changes | **approximate, opt-in** |

Activation caching policies (`--cache`):

* `fbcache` — *first-block probe*: always compute block 0 (kept resident),
  compare its residual with the last fully computed step
  (`mean|Δr| / mean|r|`); under the threshold, skip every other block and add
  the cached tail residual. In a streaming engine a skipped step saves not only
  compute but **all of that step's PCIe/disk traffic**, and the prefetched
  units stay in the arena for the next step (FIFO order is preserved).
* `tiered` *(experimental, novel)* — **residency-aware caching**. On a
  probe-approved step, *resident* blocks are recomputed (cheap: no transfer)
  while *streamed* blocks are replaced by their cached residuals (moving a
  ~28 MB activation instead of a 216–432 MB weight block, or nothing at all).
  The cost of recomputing a block depends on its tier, so the cache should be
  spent where the weights are expensive to move.
* `--cache-predict linear` (the default since the §7 measurement) — instead of
  reusing the last residual, extrapolate it linearly in σ from the last two full
  steps (a first-order TaylorSeer-style forecast). `--cache-predict reuse` keeps
  the FBCache behaviour.
* Guards: warm-up steps, final steps always computed, max consecutive skips.

### 3.6 Block-major batching

colibri's "batch-union" reads each expert once for many tokens. selas's analogue
is **block-major execution**: several images (seeds/prompts) are advanced
through each block while it is loaded, so one transfer serves N images. With
`--micro-batch m` they are run *m at a time inside the block*, so activation
memory stays at one micro-batch while the transfer cost is still divided by N.
This turns a transfer-bound 512² run into a compute-bound one.

---

## 4. Memory budget on the development machine

FLUX.1-dev fp16 (exact), 11 GB card, ~14 GB free RAM:

| where | what |
|---|---|
| VRAM ~5–6 GiB | globals + ~12 double blocks resident |
| VRAM ~1.3 GiB | ring arena (3 × 432 MiB) |
| VRAM ~1–2 GiB | activations @1024² (+ step-cache residuals if enabled) |
| pinned RAM ~10 GiB | remaining main units |
| NVMe | the rest + all modulation units (read once per image) |

Predicted (to be verified): 1024² compute-bound, ~2.5 s/step; 512²
transfer-bound ~1 s/step, compute-bound with `--batch-size 2`.

The NF4 checkpoint already on disk (6.4 GiB transformer) fits in VRAM almost
entirely and exercises the codec/JIT-dequant path rather than streaming.

---

## 5. Implementation layout

```
selas/
  cli.py          selas info | convert | plan | generate | bench | verify | compare
  hw.py           device/RAM detection, budgets, micro-benchmarks, GPU + per-disk profile cache
  codecs.py       raw / int8_row / nf4 encode+decode (+ optional Triton NF4 kernel w/ self-check)
  container.py    container format reader/writer
  sources.py      safetensors sources, prefix detection, bnb-NF4 import, diffusers→BFL adapter
  keymap.py       diffusers ↔ BFL FLUX key mapping
  convert.py      source checkpoints → containers + tokenizers + model.json
  arena.py        byte-granular FIFO ring allocator (pure Python, unit-tested)
  store.py        pinned pool, WeightStore, UnitView, UnitStream (prefetch), DiskReader
  planner.py      pipeline simulator + greedy placement
  stepcache.py    fbcache / tiered policies, residual predictors
  sampling.py     FLUX schedule, noise, latent packing
  text.py         tokenizers, CLIP/T5 orchestration, prompt-embedding cache
  pipeline.py     end-to-end generation; learned step-time and activation-memory calibration
  models/flux.py  functional FLUX blocks + streaming runner (hoisting, step cache, micro-batching)
  models/t5.py    functional T5 encoder (streamed)
  models/clip.py  functional CLIP-L text encoder
  models/vae.py   functional AE decoder (+ tiling)
tests/            unit, oracle (diffusers/transformers on tiny random models), streaming exactness,
                  and gated real-model tests (SELAS_TEST_* env vars)
experiments/      the measurement scripts behind §7, raw results in experiments/results/
```

## 6. Test plan

1. **Unit**: codecs (NF4 vs reference formula, int8 error bounds), container
   round-trip, ring arena invariants (randomized), planner properties,
   step-cache decisions.
2. **Oracle** (tiny random weights, fp32): FLUX transformer vs diffusers
   `FluxTransformer2DModel`; T5 vs `transformers.T5EncoderModel`; CLIP vs
   `CLIPTextModel`; VAE decoder vs diffusers `AutoencoderKL`.
3. **Placement exactness** (GPU): all-VRAM vs all-host vs all-disk vs mixed with
   a tiny arena → **bit-identical** outputs; hoisted modulation vs inline.
4. **Real FLUX.1-dev** (gated): per-block oracle vs diffusers on real (dequantized)
   weights; NF4 import vs bitsandbytes (if installed); resident vs streamed
   bit-exactness on 2 steps; full 1024² generation; step-cache PSNR vs exact;
   predicted vs measured step time.

## 7. Hypotheses (colibri-style: measure, publish negatives)

All runs: FLUX.1-dev fp16 container unless noted, RTX 2080 Ti 11 GB, PCIe 3.0 x16, 31 GB RAM,
NVMe, 2026-10-01. Scripts in `experiments/` (one per row), raw records in `experiments/results/`.
The GPU ran at 81–86 °C and throttled (SM clock 1350–1550 MHz under load), so absolute step
times drift by ~5 %; A/B comparisons were run in ABBA or forward+reverse order to cancel it.

| hypothesis | experiment | result |
|---|---|---|
| Streaming is fully hidden at ≥768² on PCIe 3.0 | measured stall time per step vs resolution | **confirmed at 1024², not at 512²**. 1024²: 0.1 ms stall/step at every placement; **streaming all 57 blocks through a 2-block arena runs at 3.07 s/step with 1.60 GiB peak VRAM** vs 2.97 s with 7.5 GiB (≈ +3 %, within thermal noise). 512²: 84 ms stall/step at the auto plan (0.885 s/step); streaming everything costs +63 % (1.44 s, 666 ms stall) |
| Belady placement + planner beats naive "first N blocks resident" | A/B step time, same VRAM (same arena, ≤ same resident bytes), ABBA | **confirmed where transfer-bound, moot where compute-bound**. 512²: first-N is +20 % slower at the auto budget (1.066 vs 0.885 s; stall 315 vs 84 ms) and +11 % at 5 GiB. 1024²: +3 % and −0.1 % (noise). The simulator predicted the 512² gaps (+18 %, +6 %) |
| `tiered` caching gives better quality per second than `fbcache` when transfer-bound | PSNR/LPIPS vs exact at matched wall time | **refuted on this setup**: same skip decisions, fp16 1024² 28 steps → fbcache 39.05 dB in 79.8 s vs tiered 38.71 dB in 86.9 s; NF4 at a 2.5 GiB budget → 38.36 vs 38.33 dB at equal time. Mixing fresh resident-block residuals with stale streamed ones is not better than reusing the whole tail consistently |
| Linear residual extrapolation beats reuse at equal skip rate | fbcache, 3 prompts, 1024², 28 steps, PSNR vs exact | **confirmed, modestly**. Threshold 0.08 (10–11/28 skipped, identical schedules): linear 32.37 dB vs reuse 30.36 dB mean (+2.0; per prompt +3.5, +0.1, +2.5). Threshold 0.15 (16/28 skipped), linear replaying reuse's exact schedule: 23.64 vs 23.01 dB (+0.6; one prompt −0.5). Same wall time |
| Block-major micro-batching makes 512² compute-bound | 512², batch × micro, 10 steps, forward + reverse order | **true but nearly irrelevant here**: batch 1 is only marginally transfer-bound (22–93 ms stall/step) because the planner keeps ~18–20 blocks resident and spread. Batch 2: −8 % time per image (0.812 vs 0.878 s/image-step); batch 4: −7 %; batch 8: −2 %. Micro-batch 1 inside batch 4 gains nothing (0.882), so the gain is GEMM size, not hidden transfers |
| O_DIRECT helps on this NVMe | microbenchmark + engine A/B (512², `--ram-gb 6`, 4.85–5.06 GiB/step from disk), page cache evicted per run | **confirmed**. Cold reads: 3.28 GiB/s O_DIRECT vs 2.19 GiB/s buffered, at ⅓ the CPU (0.16 vs 0.47 s/GiB). Engine with free RAM: equal (1.49 vs 1.54 s/step; buffered is served by the page cache after step 1). Engine inside a cgroup capping page cache at ~1.5 GiB (RAM really short, the case the disk tier exists for): **O_DIRECT 1.47 s/step vs buffered 2.78 s/step (1.9×)** |

### 7.1 Measured so far

FLUX.1-dev, prompt "a photo of a red fox sitting in fresh snow, golden hour", seed 0,
RTX 2080 Ti 11 GB (no other GPU workloads), 31 GB RAM, PCIe 3.0 x16. All placements of the same
job produce **pixel-identical** images (verified at 1024²: all-VRAM vs 4 GiB vs 2.5 GiB budgets).

| model | size | budget | placement (V/h/d of 57 blocks) | step | peak VRAM |
|---|---|---|---|---|---|
| NF4 (Forge) | 6.2 GiB | auto | 57 / 0 / 0 | 2.68 s | 5.33 GiB |
| NF4 | | `--vram-gb 4` | 17 / 40 / 0 | 2.82 s | 2.96 GiB |
| NF4 | | `--vram-gb 2.5` | 5 / 52 / 0 | 2.98 s | **1.47 GiB** |
| fp16 (exact) | 22.2 GiB | auto | 15 / 42 / 0 | 2.73 s | 7.76 GiB |
| fp16 (exact) | | `--ram-gb 6 --direct-io` | 15 / 22 / 20 | 2.87 s | 7.72 GiB |

Step cache at 1024², 28 steps: `fbcache` (threshold 0.08) approximates 10–11/28 steps and
cuts denoise time ~35 % (86 → 56 s). Quality depends strongly on the prompt: 39 dB on the fox
(the prompt first measured), but 28 and 24 dB on an interior and an ink illustration, 30.4 dB
mean with `reuse` (32.4 with `linear`, now the default). Visually, 0.08 keeps the composition but softens fine
line work and can swap small objects (a bookshelf became a plant); 0.15 (16/28 skipped, −55 %
time, ~23 dB) changes content (roof colours, outbuildings). Side-by-sides:
[interior](images/cache_compare_interior.jpg), [ink illustration](images/cache_compare_ink.jpg) (exact, reuse 0.08, linear 0.08, reuse 0.15).

The planner's analytic step prediction was 25–40 % low before calibration (it ignores
elementwise ops); after the learned per-block calibration (`<model>/.selas/profile.json`)
it was within 1–15 %. The learned scale is one scalar per block kind, overwritten by every
run; across the batching runs it ranged 1.37–1.66 with batch shape and GPU temperature, so
predictions carry that much noise. Disk bandwidth was a 2 GB/s default until
`selas bench` ran, against this NVMe's 3.5 GB/s O_DIRECT rate (a 768² `--ram-gb 6` plan predicted
2.83 s/step, measured 1.62 s). It is now measured automatically: the first time an engine opens a
model on a block device, ≤ 1 GiB of cold reads in the engine's read mode, cached per device and
mode in `~/.cache/selas/hw.json` (`disks`); the same plan now predicts 1.85 s. The
activation reserve was estimated at ~1.4–1.7 GiB at 1024² against a measured peak of 0.79 GiB
(allocator-reserved bytes above weights, arena, hoisted modulation and cache buffers), so a
2 GiB budget needed a hand-set `--reserve-gb`. It is now learned too: every run records its
peak per shape (tokens, batch, micro-batch) in `<model>/.selas/profile.json`; a repeated shape
reserves peak × 1.05 + 64 MiB, another shape scales the nearest measurement by the analytic
model's ratio with × 1.15 + 256 MiB. If a learned reserve runs out of VRAM, the batch is retried
once on the analytic estimate and the margins widen ×1.5. Result: the default 1024² plan reserves
0.89 instead of 1.40 GiB (one more resident block), and `--vram-gb 2` alone runs full fp16
FLUX.1-dev at 1024² in a 2,030 MiB process (incl. CUDA context), 2.74 s/step.

Bugs found by the real-model run: random pooled vectors are off CLIP's output manifold and
drive FLUX modulations to ~5·10⁴ (in fp32 too), overflowing any fp16 forward — tests now use
real prompt embeddings and the engine fails fast on non-finite latents; the `tiered` planner
could not place residuals when VRAM was short — they now go to pinned RAM.

## 8. Future directions

* Lossless exponent coding of bf16 weights (DFloat11-style) to cut exact-mode
  transfer ~30 %, decoded on GPU at the same JIT point as NF4.
* LoRA applied as a low-rank delta at the JIT-decode point (no merged copy).
* Precision ladder: early high-noise steps from a resident NF4 copy, late
  steps from the exact streamed weights.
* Multi-GPU: place units across devices; second GPU as a VRAM tier.
