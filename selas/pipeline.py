"""End-to-end FLUX generation on a tiered weight store.

Phases (each phase frees the GPU for the next):

1. text   — prompt cache, else CLIP (resident) + T5 (streamed from disk, fp32);
2. plan   — placement for this resolution/batch (see :mod:`selas.planner`);
3. denoise — prologue (hoisted modulations) then N steps over the cyclic stream;
4. decode — VAE (resident, fp32; tiled when VRAM is short).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .container import Container
from .codecs import decoded_is_view
from .hw import cuda_device, default_compute_dtype, device_info, disk_bandwidth, load_profile, ram_budget, vram_budget
from .models.flux import FluxConfig, FluxRunner, activation_reserve, main_order, mod_order, unit_flops
from .models.vae import VaeConfig, VaeDecoder
from .planner import Plan, UnitCost, make_plan
from .sampling import get_noise, get_schedule, image_ids, latent_hw, pack, text_ids, unpack
from .stepcache import StepCache, StepCacheConfig
from .store import DISK, HOST, VRAM, WeightStore
from .text import TextEncoders
from .util import GiB, MiB, align_up, fmt_seconds, human_bytes, log, read_json, warn, write_json_atomic

DTYPE_NAMES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SHAPE_KEYS = ("l_img", "l_txt", "batch", "micro")
# Learned activation reserve = measured peak * margin + headroom (both times a safety factor that
# grows after an out-of-memory error). A shape measured before repeats its peak almost exactly;
# one scaled from another shape through the analytic model gets a wider margin.
ACT_MARGIN_EXACT, ACT_HEADROOM_EXACT = 1.05, 64 * MiB
ACT_MARGIN, ACT_HEADROOM = 1.15, 256 * MiB


def learned_activation(observations: list[dict], shape: dict, estimate, safety: float = 1.0) -> int | None:
    """Activation reserve from measured peaks, or None without any.

    ``observations``: ``{l_img, l_txt, batch, micro, bytes}`` records of past runs;
    ``estimate(shape)``: the analytic model, used only for relative scaling. Uses the
    observation of the same shape if there is one, else the one closest in estimated
    size, scaled by the estimate's ratio.
    """
    if not observations:
        return None
    for o in observations:
        if all(o[k] == shape[k] for k in SHAPE_KEYS):
            return int(safety * (o["bytes"] * ACT_MARGIN_EXACT + ACT_HEADROOM_EXACT))
    target = estimate(shape)

    def ratio(o: dict) -> float:
        return target / max(1, estimate(o))

    best = min(observations, key=lambda o: abs(math.log(ratio(o))))
    return int(safety * (best["bytes"] * ratio(best) * ACT_MARGIN + ACT_HEADROOM))


@dataclass
class RuntimeOptions:
    device: int | None = None
    dtype: str = "auto"  # auto | fp16 | bf16 | fp32
    vram_gb: float | None = None
    ram_gb: float | None = None
    reserve_gb: float | None = None  # override the (learned) activation reserve
    direct_io: bool = True  # O_DIRECT for the disk tier; falls back to buffered reads (with a warning) if refused
    profile: bool = False
    micro_batch: int | None = None
    vae_tile: str = "auto"  # auto | on | off
    prompt_cache: bool = True
    placement: str = "auto"  # auto | vram | host | disk | first:N
    keep_loaded: bool = False


@dataclass
class Job:
    prompt: str
    seed: int = 0
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    guidance: float | None = None


@dataclass
class Result:
    job: Job
    image: object  # PIL.Image.Image
    stats: dict = field(default_factory=dict)


class FluxEngine:
    def __init__(self, model_dir: str | Path, rt: RuntimeOptions | None = None):
        self.rt = rt or RuntimeOptions()
        self.dir = Path(model_dir)
        info = read_json(self.dir / "model.json")
        if not info or info.get("format") != "selas-model":
            raise FileNotFoundError(f"{self.dir} is not a selas model (run `selas convert` first)")
        self.info = info
        self.defaults = info.get("defaults", {})
        self.device = cuda_device(self.rt.device)
        if self.rt.dtype == "auto":
            self.dtype = default_compute_dtype(self.device)
        else:
            self.dtype = DTYPE_NAMES[self.rt.dtype]
        self.hw = load_profile(self.device)
        self.tc = Container(self.dir / info["components"]["transformer"])
        disk_bw = disk_bandwidth(self.tc.data_path, direct=self.rt.direct_io)  # measured once per disk, then cached
        if disk_bw:
            self.hw.disk_bw = disk_bw
        self.cfg = FluxConfig.from_dict(self.tc.config)
        self.store: WeightStore | None = None
        self.runner: FluxRunner | None = None
        self.plan: Plan | None = None
        self.reserve_source = "estimated"  # where the last plan's activation reserve came from
        self._force_estimate = False  # set for one re-plan after an out-of-memory error
        self.residual_host = False  # tiered cache: block residuals in pinned RAM (set by make_plan)
        self._plan_key = None
        dev = device_info(self.device)
        self._gpu_key = f"{dev.name}|{self.dtype}"
        log(f"{dev.name} (sm_{dev.capability[0]}{dev.capability[1]}), {human_bytes(dev.free)} free of {human_bytes(dev.total)}; "
            f"compute {str(self.dtype).removeprefix('torch.')}{' + fp32 residual stream' if self.dtype != torch.float32 else ''}"
            f"{' (no native bf16 on this GPU)' if not dev.bf16_native and self.dtype == torch.float16 else ''}")
        log(self.tc.describe())

    # ------------------------------------------------------------------ cost model
    def _learned(self) -> dict:
        d = read_json(self.dir / ".selas" / "profile.json", {}) or {}
        return d.get(self._gpu_key, {})

    def _estimate(self, kind: str, l_img: int, l_txt: int, batch: int, spec) -> float:
        hw, cfg = self.hw, self.cfg
        lin, att = unit_flops(kind, cfg, l_img, l_txt, batch)
        t = lin / hw.matmul_flops + att / hw.attn_flops
        t += batch * (l_img + l_txt) * cfg.hidden * 100 / hw.mem_bw
        esize = torch.finfo(self.dtype).bits // 8
        dec = 0
        for ts in spec.tensors.values():
            if len(ts.shape) == 2 and not decoded_is_view(ts.codec, ts.stored_dtype, self.dtype):
                n = 1
                for s in ts.shape:
                    n *= s
                dec += n * esize
        return t + dec / hw.dequant_bw

    def _unit_seconds(self, name: str, l_img: int, l_txt: int, batch: int) -> float:
        spec = self.tc.units[name]
        kind = spec.kind
        if kind in ("double", "single"):
            est = self._estimate(kind, l_img, l_txt, batch, spec)
            scale = self._learned().get(kind, {}).get("scale")
            return est * scale if scale else est
        # modulation units: one [S*B, D] x [D, kD] matmul per step count — negligible
        n = sum(1 for _ in spec.tensors)
        return 1e-4 * n

    def _hoisted_bytes(self, steps: int, batch: int) -> int:
        """Modulation vectors for every step, computed in the prologue and kept in VRAM."""
        cfg, d = self.cfg, self.cfg.hidden
        esize = torch.finfo(self.dtype).bits // 8
        return steps * batch * (cfg.depth_double * 12 * d + cfg.depth_single * 3 * d + 2 * d) * esize

    def _cache_bytes(self, cache: StepCacheConfig, batch: int, l_img: int) -> int:
        """Step-cache buffers: probe reference + kept tail residual(s), fp32."""
        if not cache.active:
            return 0
        keep = 2 if cache.predict == "linear" else 1
        return (1 + keep) * batch * l_img * self.cfg.hidden * 4

    def _activation_reserve(self, l_img: int, l_txt: int, batch: int, micro: int) -> tuple[int, str]:
        """(bytes, source): peak memory of one block forward + persistent state, beyond weights,
        arena, hoisted modulation and cache buffers. Learned from past runs when possible."""
        dec = self._decode_temp_bytes()

        def estimate(o: dict) -> int:
            return activation_reserve(self.cfg, o["l_img"], o["l_txt"], o["batch"], o["micro"], dec)

        shape = {"l_img": l_img, "l_txt": l_txt, "batch": batch, "micro": micro}
        learned = self._learned()
        if not self._force_estimate:
            got = learned_activation(learned.get("activation", []), shape, estimate, learned.get("activation_safety", 1.0))
            if got is not None:
                return got, "learned"
        return estimate(shape), "estimated"

    def _learn_activation(self, shape: dict, nbytes: int) -> None:
        path = self.dir / ".selas" / "profile.json"
        db = read_json(path, {}) or {}
        entry = db.setdefault(self._gpu_key, {})
        key = tuple(shape[k] for k in SHAPE_KEYS)
        old = [o for o in entry.get("activation", []) if tuple(o[k] for k in SHAPE_KEYS) == key]
        if old:  # keep the worst case seen for this shape (e.g. with and without the step cache)
            nbytes = max(int(nbytes), int(old[0]["bytes"]))
        obs = [o for o in entry.get("activation", []) if tuple(o[k] for k in SHAPE_KEYS) != key]
        entry["activation"] = (obs + [{**shape, "bytes": int(nbytes)}])[-16:]
        try:
            write_json_atomic(path, db)
        except OSError:
            pass

    def _activation_oom(self) -> None:
        """A learned reserve ran out of VRAM: widen its margins for the future."""
        path = self.dir / ".selas" / "profile.json"
        db = read_json(path, {}) or {}
        entry = db.setdefault(self._gpu_key, {})
        entry["activation_safety"] = min(3.0, float(entry.get("activation_safety", 1.0)) * 1.5)
        try:
            write_json_atomic(path, db)
        except OSError:
            pass

    def _decode_temp_bytes(self) -> int:
        esize = torch.finfo(self.dtype).bits // 8
        worst = 0
        for u in self.tc.units.values():
            for ts in u.tensors.values():
                if len(ts.shape) == 2 and not decoded_is_view(ts.codec, ts.stored_dtype, self.dtype):
                    worst = max(worst, ts.shape[0] * ts.shape[1] * esize)
        return worst

    # ------------------------------------------------------------------ planning
    def make_plan(self, width: int, height: int, batch: int, steps: int, cache: StepCacheConfig) -> Plan:
        cfg, units = self.cfg, self.tc.units
        h, w = latent_hw(height, width)
        l_img = (h // 2) * (w // 2)
        l_txt = int(self.defaults.get("max_t5_tokens", 512))
        micro = min(self.rt.micro_batch or batch, batch)
        vram = vram_budget(self.device, self.rt.vram_gb)
        if self.rt.reserve_gb is not None:
            act, self.reserve_source = int(self.rt.reserve_gb * GiB), "set by --reserve-gb"
        else:
            act, self.reserve_source = self._activation_reserve(l_img, l_txt, batch, micro)
        d = cfg.hidden
        reserve = act + self._hoisted_bytes(steps, batch) + self._cache_bytes(cache, batch, l_img)
        main, mods = main_order(cfg), mod_order(cfg)
        res_bytes = {n: batch * (l_img + l_txt) * d * 2 for n in main}  # tiered: bf16 residual per streamed block
        keep = 2 if cache.predict == "linear" else 1
        avail = vram - reserve - units["globals"].nbytes
        ram = ram_budget(self.rt.ram_gb)
        force = {main[0]} if cache.active else set()

        def plan_with(avail_bytes: int, ram_bytes: int = ram) -> Plan:
            if self.rt.placement != "auto":
                return self._manual_plan(main, mods, avail_bytes, l_img, l_txt, batch, force)
            cycle = [UnitCost(n, units[n].nbytes, self._unit_seconds(n, l_img, l_txt, micro) * (batch / micro), units[n].kind) for n in main]
            once = [UnitCost(n, units[n].nbytes, self._unit_seconds(n, l_img, l_txt, batch), units[n].kind) for n in mods]
            return make_plan(cycle, once, avail_bytes, ram_bytes, self.hw.h2d_bw, self.hw.disk_bw, force_vram=force)

        def residuals(p: Plan) -> int:  # tiered: cached residual bytes for p's streamed blocks
            return keep * sum(res_bytes[n] for n in main if p.tiers[n] != VRAM)

        plan = plan_with(avail)
        self.residual_host = False
        if cache.policy == "tiered" and residuals(plan):
            plan, self.residual_host = self._plan_tiered(plan, plan_with, residuals, avail, ram)
        plan.tiers["globals"] = VRAM
        plan.sizes["globals"] = units["globals"].nbytes
        plan.notes.append(f"budget: {human_bytes(vram)} VRAM usable, reserve {human_bytes(act)} for activations "
                          f"({self.reserve_source}) + {human_bytes(reserve - act)} hoisted/cache, "
                          f"{human_bytes(ram)} pinned RAM; tokens {l_img}+{l_txt}, batch {batch} (micro {micro})")
        return plan

    @staticmethod
    def _plan_tiered(base: Plan, plan_with, residuals, avail: int, ram: int) -> tuple[Plan, bool]:
        """Place the tiered cache's per-block residuals: (plan, residuals_in_host_ram).

        In VRAM they cost weight residency (more streaming on every full step); in
        pinned RAM they cost one small H2D copy per substituted block on cached
        steps. VRAM wins only if it barely slows full steps. Streaming more blocks
        needs more residuals, so the VRAM option is iterated to a fixed point.
        """
        dev, need = None, residuals(base)
        for _ in range(8):
            if need >= avail:
                break
            try:
                p = plan_with(avail - need)
            except MemoryError:
                break
            if residuals(p) <= need:
                dev = p
                break
            need = residuals(p)
        host_ram = ram - residuals(base)
        host = None
        if host_ram > 0:
            host = plan_with(avail, host_ram)  # VRAM placement as in ``base``; host-tier units may shift to disk
        if dev is not None and (host is None or dev.step_s <= host.step_s * 1.02):
            dev.notes.append(f"tiered cache: block residuals in VRAM ({human_bytes(residuals(dev))})")
            return dev, False
        if host is None:
            raise MemoryError(f"tiered cache: {human_bytes(residuals(base))} of block residuals fit neither VRAM nor "
                              f"RAM ({human_bytes(ram)}); use --cache fbcache")
        host.notes.append(f"tiered cache: block residuals in pinned RAM ({human_bytes(residuals(host))}); "
                          f"VRAM option {'predicted ' + fmt_seconds(dev.step_s) + '/step' if dev else 'does not fit'}")
        return host, True

    def _manual_plan(self, main, mods, avail, l_img, l_txt, batch, force) -> Plan:
        units = self.tc.units
        mode = self.rt.placement
        tiers = {}
        if mode == "vram":
            tiers = {n: VRAM for n in main}
        elif mode in ("host", "disk"):
            tiers = {n: (VRAM if n in force else (HOST if mode == "host" else DISK)) for n in main}
        elif mode.startswith("first:"):
            k = int(mode.split(":", 1)[1])
            tiers = {n: (VRAM if (i < k or n in force) else HOST) for i, n in enumerate(main)}
        else:
            raise ValueError(f"unknown placement {mode!r}")
        for n in mods:
            tiers[n] = DISK if mode == "disk" else HOST
        streamed = [units[n].nbytes for n in main + mods if tiers[n] != VRAM]
        arena = align_up(3 * max(streamed), 4096) if streamed else 0
        staging = 2 * max((units[n].nbytes for n in main + mods if tiers[n] == DISK), default=0)
        p = Plan(tiers, arena, staging, 0.0, 0.0, sizes={n: units[n].nbytes for n in main + mods})
        p.notes.append(f"manual placement '{mode}' (planner bypassed)")
        return p

    # ------------------------------------------------------------------ loading
    def ensure_loaded(self, width: int, height: int, batch: int, steps: int, cache: StepCacheConfig) -> None:
        key = (width, height, batch, steps, cache.policy, cache.predict, self.rt.micro_batch)
        if self._plan_key == key and self.store is not None:
            return
        self.unload()  # plan against the VRAM/RAM that will actually be free, not what the old store holds
        plan = self.make_plan(width, height, batch, steps, cache)
        log("placement plan:\n  " + plan.describe(main_order(self.cfg)).replace("\n", "\n  "))
        self.plan = plan
        self.store = WeightStore(self.tc, plan.tiers, self.device, self.dtype, arena_bytes=plan.arena_bytes,
                                 staging_bytes=plan.staging_bytes, direct_io=self.rt.direct_io, label="flux")
        self.runner = FluxRunner(self.store, self.cfg, self.dtype, micro_batch=self.rt.micro_batch)
        self._plan_key = key

    def unload(self) -> None:
        if self.runner is not None:
            self.runner.end()
        if self.store is not None:
            self.store.close()
        self.store = None
        self.runner = None
        self._plan_key = None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ generation
    def denoise(self, jobs: list[Job], embeds: dict, cache_cfg: StepCacheConfig, on_step=None) -> tuple[torch.Tensor, dict]:
        """Run one batch (same size/steps/guidance). Returns (latents [B,16,h,w] CPU fp32, stats)."""
        j0 = jobs[0]
        width, height = j0.width, j0.height
        steps = j0.steps or int(self.defaults.get("steps", 28))
        guidance = j0.guidance if j0.guidance is not None else float(self.defaults.get("guidance", 3.5))
        b = len(jobs)
        self.ensure_loaded(width, height, b, steps, cache_cfg)
        runner, store = self.runner, self.store
        h, w = latent_hw(height, width)
        l_img = (h // 2) * (w // 2)
        sigmas = get_schedule(steps, l_img, shift=bool(self.defaults.get("shift", True)))
        txt = torch.stack([embeds[j.prompt][0] for j in jobs])
        pooled = torch.stack([embeds[j.prompt][1] for j in jobs])
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()  # so the reserved-memory peak below measures what this run really needs
        base_reserved = torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        t0 = time.perf_counter()
        store.stats.reset()
        cond = runner.prepare(sigmas, guidance, pooled, txt, image_ids(height, width), text_ids(txt.shape[1]))
        torch.cuda.synchronize(self.device)
        t_prologue = time.perf_counter() - t0
        prologue_bytes = dict(store.stats.h2d_bytes)

        x = torch.cat([pack(get_noise(j.seed, height, width)) for j in jobs]).to(self.device)
        streamed = {n for n in runner.main if runner.is_streamed(n)}
        cache = StepCache(cache_cfg, steps, streamed, torch.float32 if self.dtype == torch.float32 else torch.bfloat16,
                          host_residuals=self.residual_host)
        step_times, decisions = [], []
        learn_step = 1 if steps > 2 else None
        store.stats.reset()
        compute = torch.cuda.current_stream(self.device)
        runner.begin()
        try:
            for s in range(steps):
                store.profile = self.rt.profile or s == learn_step
                if s == learn_step:
                    store.stats.compute_events = {}  # calibrate on this step alone
                ts = time.perf_counter()
                pred, info = runner.step(s, x, cond, cache)
                x = x + (sigmas[s + 1] - sigmas[s]) * pred
                # wait for compute only: the copy stream keeps prefetching the next step's blocks
                compute.synchronize()
                if not bool(torch.isfinite(x).all()):  # tiny reduction; fail now rather than decode a black image
                    hint = " (fp16 overflow: retry with --dtype fp32)" if self.dtype == torch.float16 else ""
                    raise FloatingPointError(f"non-finite latents after step {s + 1}/{steps} in {self.dtype}{hint}")
                dt_s = time.perf_counter() - ts
                step_times.append(dt_s)
                decisions.append(info.decision)
                if s == learn_step:
                    self._learn(store.stats.compute_seconds(), l_img, txt.shape[1], b)
                if on_step:
                    on_step(s, steps, dt_s, info)
                else:
                    extra = "" if info.decision == "full" else f" [{info.decision} d={info.distance:.3f}]"
                    log(f"step {s + 1:2d}/{steps} {fmt_seconds(dt_s)}{extra}", 1)
        finally:
            store.profile = self.rt.profile
            runner.end()
            summary = cache.summary()
            cache.close()
        stall = store.stats.stall_seconds() if self.rt.profile else None
        if cache_cfg.policy != "tiered" or self.residual_host:  # VRAM-kept tiered residuals would skew it
            act = (torch.cuda.max_memory_reserved(self.device) - base_reserved
                   - self._hoisted_bytes(steps, b) - self._cache_bytes(cache_cfg, b, l_img))
            micro = min(self.rt.micro_batch or b, b)
            self._learn_activation({"l_img": l_img, "l_txt": txt.shape[1], "batch": b, "micro": micro}, max(0, act))
        lat = unpack(x, height, width).float().cpu()
        full = [t for t, d in zip(step_times, decisions) if d == "full"]
        stats = {
            "steps": steps,
            "prologue_s": t_prologue,
            "prologue_h2d": prologue_bytes,
            "step_s": step_times,
            "mean_full_step_s": sum(full[1:] or full) / max(1, len(full[1:] or full)),
            "predicted_step_s": self.plan.step_s,
            "h2d_bytes": dict(store.stats.h2d_bytes),
            "disk_read_bytes": store.stats.disk_read_bytes,
            "stall_s": stall,
            "peak_vram": torch.cuda.max_memory_allocated(self.device),
            "cache": summary,
            "decisions": decisions,
        }
        return lat, stats

    def _learn(self, kind_times: dict, l_img: int, l_txt: int, batch: int) -> None:
        """Calibrate the planner's cost model with measured per-block compute times."""
        path = self.dir / ".selas" / "profile.json"
        db = read_json(path, {}) or {}
        entry = db.get(self._gpu_key, {})
        for kind in ("double", "single"):
            if kind not in kind_times:
                continue
            secs, n = kind_times[kind]
            if not n:
                continue
            name = f"{kind}.0"
            micro = min(self.rt.micro_batch or batch, batch)
            est = self._estimate(kind, l_img, l_txt, micro, self.tc.units[name]) * (batch / micro)
            if est > 0:
                entry[kind] = {"scale": (secs / n) / est, "seconds": secs / n, "tokens": l_img + l_txt, "batch": batch}
        db[self._gpu_key] = entry
        try:
            write_json_atomic(path, db)
        except OSError:
            pass

    def decode(self, latents: list[torch.Tensor], sizes: list[tuple[int, int]]) -> list:
        from PIL import Image

        vc = Container(self.dir / self.info["components"]["vae"])
        out = []
        with WeightStore(vc, {"decoder": VRAM}, self.device, torch.float32, label="vae") as st:
            dec = VaeDecoder(st, VaeConfig.from_dict(vc.config))
            for lat, (wpx, hpx) in zip(latents, sizes):
                img = self._decode_one(dec, lat.to(self.device), wpx, hpx)
                arr = ((img[0].permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                out.append(Image.fromarray(arr))
        vc.close()
        return out

    def _decode_one(self, dec: VaeDecoder, lat: torch.Tensor, wpx: int, hpx: int) -> torch.Tensor:
        mode = self.rt.vae_tile
        free, _ = torch.cuda.mem_get_info(self.device)
        need = hpx * wpx * 256 * 4 * 3  # rough fp32 peak of the full-resolution up blocks
        tiled = mode == "on" or (mode == "auto" and need > free * 0.9)
        if not tiled:
            try:
                return dec.decode(lat)
            except torch.cuda.OutOfMemoryError:
                if mode == "off":
                    raise
                torch.cuda.empty_cache()
                warn("VAE decode ran out of memory; retrying tiled")
        return dec.decode_tiled(lat)

    def generate(self, jobs: list[Job], cache_cfg: StepCacheConfig | None = None, batch_size: int = 1, on_step=None) -> list[Result]:
        cache_cfg = cache_cfg or StepCacheConfig()
        t_all = time.perf_counter()
        for j in jobs:  # FLUX works on a 16-pixel grid (8x VAE x 2x2 patches)
            w16, h16 = max(16, j.width // 16 * 16), max(16, j.height // 16 * 16)
            if (w16, h16) != (j.width, j.height):
                warn(f"{j.width}x{j.height} is not a multiple of 16; using {w16}x{h16}")
                j.width, j.height = w16, h16
        tenc = TextEncoders(self.dir, self.info, self.device, direct_io=self.rt.direct_io, use_cache=self.rt.prompt_cache)
        t0 = time.perf_counter()
        embeds = tenc.encode([j.prompt for j in jobs])
        tenc.close()
        torch.cuda.empty_cache()
        t_text = time.perf_counter() - t0

        groups: dict[tuple, list[Job]] = {}
        for j in jobs:
            groups.setdefault((j.width, j.height, j.steps, j.guidance), []).append(j)
        pending: list[tuple[Job, torch.Tensor, dict]] = []
        for gjobs in groups.values():
            for i in range(0, len(gjobs), batch_size):
                batch = gjobs[i : i + batch_size]
                lat, stats = self._denoise_safe(batch, embeds, cache_cfg, on_step)
                self._report(stats)
                for k, j in enumerate(batch):
                    pending.append((j, lat[k : k + 1], stats))
        if not self.rt.keep_loaded:
            self.unload()
        t0 = time.perf_counter()
        images = self.decode([p[1] for p in pending], [(p[0].width, p[0].height) for p in pending])
        t_vae = time.perf_counter() - t0
        log(f"text {fmt_seconds(t_text)}, VAE {fmt_seconds(t_vae)}, total {fmt_seconds(time.perf_counter() - t_all)}")
        return [Result(j, im, {**st, "text_s": t_text, "vae_s": t_vae}) for (j, _, st), im in zip(pending, images)]

    def _denoise_safe(self, jobs, embeds, cache_cfg, on_step=None):
        """``denoise``; if a *learned* activation reserve runs out of VRAM, widen its margin
        and retry once with the conservative estimate instead of failing the batch."""
        try:
            return self.denoise(jobs, embeds, cache_cfg, on_step)
        except torch.cuda.OutOfMemoryError:
            if self.reserve_source != "learned":
                raise
        # Retry outside the handler: the traceback would keep the failed attempt's tensors alive.
        warn("out of VRAM with the learned activation reserve; retrying with the conservative estimate")
        self._activation_oom()
        self.unload()
        self._force_estimate = True
        try:
            return self.denoise(jobs, embeds, cache_cfg, on_step)
        finally:
            self._force_estimate = False
            self._plan_key = None  # the next batch plans with the (re-)learned reserve again

    def _report(self, st: dict) -> None:
        h2d = st["h2d_bytes"]
        n = max(1, st["steps"])
        msg = (f"denoise: prologue {fmt_seconds(st['prologue_s'])}, mean full step {fmt_seconds(st['mean_full_step_s'])} "
               f"(planner predicted {fmt_seconds(st['predicted_step_s'])}), streamed/step host {human_bytes(h2d[HOST] / n)} "
               f"disk {human_bytes(h2d[DISK] / n)}, peak VRAM {human_bytes(st['peak_vram'])}")
        if st.get("stall_s") is not None:
            msg += f", compute stalled on transfers {fmt_seconds(st['stall_s'])} total"
        log(msg)
        log(st["cache"])

    def close(self) -> None:
        self.unload()
        self.tc.close()


def png_metadata(job: Job, defaults: dict, cache_cfg: StepCacheConfig, extra: str = "") -> dict[str, str]:
    from . import __version__

    steps = job.steps or defaults.get("steps")
    guidance = job.guidance if job.guidance is not None else defaults.get("guidance")
    return {
        "parameters": f"{job.prompt}\nSteps: {steps}, Seed: {job.seed}, Size: {job.width}x{job.height}, "
                      f"Guidance: {guidance}, Engine: selas {__version__}, Step cache: {cache_cfg.describe()}{extra}"
    }


__all__ = ["FluxEngine", "Job", "Result", "RuntimeOptions", "png_metadata"]
