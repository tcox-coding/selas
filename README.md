<p align="center">
  <img src="docs/images/logo.png" width="360" alt="selas logo">
</p>

<h1 align="center">selas</h1>

<p align="center"><b>Run FLUX.1 at full precision on a GPU that is too small for it.</b></p>

selas is a tiered-memory inference engine for diffusion transformers. Instead of requiring the model to
*fit* in VRAM, it *places* the weights across VRAM, pinned RAM and NVMe and streams what is not resident
just in time. The idea comes from [colibri](https://github.com/JustVugg/colibri), which does this for huge
mixture-of-experts LLMs. Diffusion needs a different design, plus some tricks that only work because
diffusion is iterative.

<p align="center">
  <img src="docs/images/fox.jpg" width="420" alt="A red fox sitting in snow at golden hour">
  <br><sub>FLUX.1-dev at full precision, 1024², 28 steps, generated on an 11 GB RTX 2080 Ti.</sub>
</p>

## Results

FLUX.1-dev, full precision (fp16, 22 GiB transformer), 1024², on an RTX 2080 Ti (11 GB, PCIe 3.0):

| | VRAM used (whole process) | time per step |
|---|---|---|
| everything on the GPU (what it would take) | ~23 GiB, ~32 GiB with T5 | — |
| `selas generate` (default) | 7.9 GiB | 2.64 s |
| `selas generate --vram-gb 2` | **2.0 GiB** | 2.74 s (+4 %) |

* At 1024² the GPU waits on weight transfers for about 0.1 ms per step: streaming is almost free.
* With only 6 GiB of pinned RAM, the rest is read from NVMe each step: 2.87 s/step.
* A 4-bit NF4 conversion fits entirely in VRAM (5.3 GiB allocated) and runs at 2.68 s/step.
* At 512² a step is too short to hide transfers, and streaming everything is 63 % slower than the default.

All placements give **bit-identical** images. The tests and the measurements check this. Every
approximation is opt-in and named in the output. [docs/PLAN.md](docs/PLAN.md) has the design and every
measurement, including the hypotheses that did not work out.

## How it works

1. **A diffusion transformer is a cyclic scan.** FLUX runs the same 57 blocks in the same order every step,
   so LRU caching gets 0 % hits. The optimal policy is to keep a fixed subset in VRAM and stream the rest
   through a small ring buffer. A pipeline simulator picks the subset and spreads it, so resident blocks
   compute while streamed ones arrive. At 512² this is 20 % faster than keeping the first blocks resident.
2. **Prefetch never stops.** The schedule is known before step 1, so copies run continuously, across step
   boundaries, from pinned RAM and from NVMe (O_DIRECT, read by a background thread).
3. **27 % of FLUX depends only on the timestep and the prompt.** selas computes those modulation layers for
   every step up front. They are read once per image instead of once per step. This is exact.
4. **It calibrates itself.** Disk bandwidth, per-block compute time and activation memory are measured
   during normal runs and cached. The planner's predictions improve after the first run, and the VRAM it
   sets aside for activations shrinks to what the model really needs.
5. **Caching at every level.** Prompt embeddings are cached on disk, so a new seed never loads T5. Optionally,
   steps that barely differ from the last one can reuse its result (`--cache fbcache`). This is ~35 % faster
   and approximate.

## Requirements

* Linux, an NVIDIA GPU with CUDA, Python ≥ 3.10, PyTorch ≥ 2.4.
* RAM for the weights that are not in VRAM. For full-precision FLUX.1-dev at minimal VRAM that is ~16 GB of
  pinned RAM. With less, the rest streams from disk (an NVMe is recommended).
* Disk space for the converted model: ~32 GB at full precision, ~11 GB for NF4.
* Developed and measured on an RTX 2080 Ti (Turing: fp16 compute with an fp32 residual stream). Ampere and
  newer use bf16. They should work but are untested.

## Install

```bash
git clone <repository-url> selas && cd selas
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or -e ".[test]" to run the tests
```

## Get the weights

FLUX.1-dev is gated. Accept its license on the
[model page](https://huggingface.co/black-forest-labs/FLUX.1-dev), log in, and download the single-file
transformer, the autoencoder, the text encoders and the tokenizers (~34 GB):

```bash
hf auth login
hf download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev \
  --include "flux1-dev.safetensors" --include "ae.safetensors" \
  --include "text_encoder/*" --include "text_encoder_2/*" --include "tokenizer/*" --include "tokenizer_2/*"
```

selas does not include or redistribute any weights. The weights are covered by the
[FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

## Convert

selas reads models from its own container format: units aligned for direct I/O, hashed, and laid out in
execution order. Conversion takes about 90 s for the full model.

```bash
selas convert --flux checkpoints/FLUX.1-dev/flux1-dev.safetensors \
  --t5 checkpoints/FLUX.1-dev/text_encoder_2 --clip checkpoints/FLUX.1-dev/text_encoder \
  --vae checkpoints/FLUX.1-dev/ae.safetensors \
  --t5-tokenizer checkpoints/FLUX.1-dev/tokenizer_2 --clip-tokenizer checkpoints/FLUX.1-dev/tokenizer \
  --out models/flux1-dev
```

Other sources work too: diffusers `transformer/` folders, and all-in-one bitsandbytes NF4 checkpoints such as
Forge's `flux1-dev-bnb-nf4-v2.safetensors`, which is imported bit-exactly. An all-in-one file has no
tokenizers, so pass `--t5-tokenizer` and `--clip-tokenizer`. `--quant int8|nf4` quantizes while converting
(approximate; the error is reported).

## Generate

```bash
selas plan     --model models/flux1-dev -W 1024 -H 1024    # show the placement and predicted step time
selas generate --model models/flux1-dev -p "a red fox in fresh snow" -W 1024 -H 1024 --seed 0 -o fox.png
selas generate --model models/flux1-dev -p "a red fox in fresh snow" --vram-gb 2 --vae-tile on -o fox.png
```

| flag | effect |
|---|---|
| `--vram-gb`, `--ram-gb` | budgets (default: what is free, minus a margin) |
| `--steps`, `--guidance`, `--seed`, `--count`, `--prompt-file` | the usual; `--count N` makes N seeds per prompt |
| `--batch-size N` | denoise N images together: one weight transfer serves all of them |
| `--cache fbcache` | step caching, **approximate**: ~35 % faster at the default `--cache-threshold 0.08`; quality varies by prompt |
| `--vae-tile on` | decode in tiles: less VRAM, slightly different pixels |
| `--no-direct-io` | read the disk tier through the page cache instead of O_DIRECT |
| `--reserve-gb` | override the learned activation reserve |
| `--placement first:N\|host\|disk\|vram` | fixed placements, for comparing against the planner |
| `--profile` | measure transfer stalls and per-block compute |

Other commands: `selas info` (hardware and model summary), `selas bench --model …` (re-measures GPU and disk
throughput), `selas verify --model …` (re-hashes every unit), `selas compare a.png b.png` (PSNR).

## Limitations

* Only FLUX.1-dev has been tested. FLUX.1-schnell converts but is untested. No other model families yet.
* Linux and a single GPU only.
* Measured on one machine (RTX 2080 Ti).
* The step cache is lossy. At the default threshold it averaged ~32 dB PSNR against the exact image over
  three prompts, ranging from 26 to 43 dB. It keeps the composition but can soften fine detail or change
  small objects.
* `--cache tiered` (recompute resident blocks, reuse streamed ones) is experimental and measured no better
  than `fbcache`.

## Experiments

`experiments/` holds the scripts behind the measurements in [docs/PLAN.md](docs/PLAN.md) §7: `placement`,
`batching`, `cache_predict` and `disk_io`. They write raw results to `experiments/results/`.

```bash
python -m experiments.placement --model models/flux1-dev
```

## Tests

```bash
pip install -e ".[test]"
pytest -m "not realmodel"            # unit, oracle and streaming tests (CUDA tests skip without a GPU)
SELAS_TEST_MODEL=models/flux1-dev pytest tests/test_real_model.py
```

* **Oracle tests** compare the FLUX, T5, CLIP and VAE implementations against diffusers/transformers on
  tiny random models.
* **Streaming tests** require bit-identical results across VRAM, host, disk and minimal-arena placements.
* **Real-model tests** compare blocks against diffusers on real weights, check placement exactness, and run
  an end-to-end generation. With `SELAS_TEST_SOURCE` pointing at an NF4 checkpoint they also check the NF4
  import against bitsandbytes.

## Acknowledgements

* [colibri](https://github.com/JustVugg/colibri) for the idea that a model needs to be placed, not to fit.
* [Black Forest Labs](https://blackforestlabs.ai) for FLUX.1.
* [diffusers](https://github.com/huggingface/diffusers) and
  [transformers](https://github.com/huggingface/transformers), used as reference implementations in the tests.
* First-block caching follows FBCache/TeaCache; the linear residual forecast follows TaylorSeer.

## License

MIT, see [LICENSE](LICENSE). Model weights are covered by their own licenses.
