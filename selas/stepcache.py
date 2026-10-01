"""Activation caching across denoising steps (opt-in, approximate).

Consecutive denoising steps run the same blocks on slightly different inputs,
so on many steps the transformer's *residual* (output minus input) barely
changes. A cheap probe — the first double block, which is always kept
VRAM-resident — measures how much the step differs from the last fully
computed one::

    d = mean|r0_now - r0_ref| / mean|r0_ref|      (r0 = probe block residual)

Policies:

* ``fbcache`` — if ``d < threshold``: skip *all* remaining blocks and add the
  cached tail residual (first-block cache). In a streaming engine a skipped
  step saves every byte of that step's PCIe/disk traffic too.
* ``tiered`` (experimental) — residency-aware: if ``d < threshold``, recompute
  the VRAM-resident blocks (no transfer cost) and replace only the *streamed*
  blocks by their cached per-block residuals. Compute is spent where weights are
  cheap to reach; transfers are skipped where they are expensive. When VRAM is
  short the residuals live in pinned host RAM instead: a substituted block then
  costs one small H2D copy of its residual (~28 MB at 1024²) instead of its
  weights (60-230 MB) plus its compute.

Residual prediction:

* ``linear`` (default) — first-order extrapolation in sigma from the last two
  full steps (a TaylorSeer-style forecast). Measured +2.0 dB PSNR over
  ``reuse`` at the same skip schedule (docs/PLAN.md §7); falls back to reuse
  until two full steps exist. Keeps two residuals, so ``tiered`` needs 2x memory.
* ``reuse`` — the last fully computed residual (FBCache/TeaCache behaviour).

Guards: the first ``warmup`` and last ``tail_full`` steps are always computed,
and at most ``max_consecutive`` steps in a row are approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .store import PinnedPool
from .util import align_up

FULL, SKIP, HYBRID = "full", "skip", "hybrid"
POLICIES = ("none", "fbcache", "tiered")


@dataclass
class StepCacheConfig:
    policy: str = "none"
    threshold: float = 0.08
    predict: str = "linear"  # linear | reuse
    warmup: int = 1
    tail_full: int = 1
    max_consecutive: int = 3

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"cache policy must be one of {POLICIES}")
        if self.predict not in ("reuse", "linear"):
            raise ValueError("cache prediction must be 'reuse' or 'linear'")

    @property
    def active(self) -> bool:
        return self.policy != "none"

    def describe(self) -> str:
        if not self.active:
            return "off (exact)"
        return (f"{self.policy} threshold={self.threshold} predict={self.predict} warmup={self.warmup} "
                f"tail_full={self.tail_full} max_consecutive={self.max_consecutive} (APPROXIMATE)")


@dataclass
class _Hist:
    items: list = field(default_factory=list)  # [(sigma, tensors tuple, ready event | None)] newest last

    def push(self, sigma: float, tensors: tuple, keep: int, ready=None) -> None:
        self.items.append((sigma, tensors, ready))
        del self.items[:-keep]

    def predict(self, sigma: float, mode: str) -> tuple:
        if not self.items:
            raise RuntimeError("no cached residual available")
        s1, r1, _ = self.items[-1]
        if mode == "linear" and len(self.items) >= 2:
            s0, r0, _ = self.items[-2]
            if s1 != s0:
                a = (sigma - s1) / (s1 - s0)
                return tuple(x1.float() + a * (x1.float() - x0.float()) for x1, x0 in zip(r1, r0))
        return tuple(x.float() for x in r1)


class StepCache:
    """Per-generation cache state. ``streamed`` = names of non-resident main units."""

    def __init__(self, cfg: StepCacheConfig, num_steps: int, streamed: set[str] | frozenset = frozenset(),
                 store_dtype: torch.dtype = torch.bfloat16, host_residuals: bool = False):
        self.cfg = cfg
        self.num_steps = num_steps
        self.streamed = frozenset(streamed)
        self.store_dtype = store_dtype
        self.keep = 2 if cfg.predict == "linear" else 1
        self.host = host_residuals  # tiered: per-block residuals in pinned RAM instead of VRAM
        self._pool: PinnedPool | None = None
        self._d2h: torch.cuda.Stream | None = None
        self._dev: torch.device | None = None
        self.ref_probe: torch.Tensor | None = None
        self.tail = _Hist()
        self.blocks: dict[str, _Hist] = {}
        self.consecutive = 0
        self.log: list[tuple[int, str, float | None]] = []

    @property
    def active(self) -> bool:
        return self.cfg.active

    # ------------------------------------------------------------------ decisions
    def decide(self, step: int, sigma: float, probe_residual: torch.Tensor) -> tuple[str, float | None]:
        cfg = self.cfg
        d = None
        decision = FULL
        forced = (
            step < cfg.warmup
            or step >= self.num_steps - cfg.tail_full
            or self.ref_probe is None
            or self.consecutive >= cfg.max_consecutive
        )
        if self.ref_probe is not None:
            ref = self.ref_probe
            d = float((probe_residual - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12))
        if not forced and d is not None and d < cfg.threshold:
            decision = SKIP if cfg.policy == "fbcache" else HYBRID
        if decision == FULL:
            self.ref_probe = probe_residual.detach().float()
            self.consecutive = 0
        else:
            self.consecutive += 1
        self.log.append((step, decision, d))
        return decision, d

    # ------------------------------------------------------------------ fbcache
    def put_tail(self, sigma: float, residual: torch.Tensor) -> None:
        self.tail.push(sigma, (residual.detach().float(),), self.keep)

    def tail_residual(self, sigma: float) -> torch.Tensor:
        return self.tail.predict(sigma, self.cfg.predict)[0]

    # ------------------------------------------------------------------ tiered
    def records(self, name: str) -> bool:
        return self.cfg.policy == "tiered" and name in self.streamed

    def substitutes(self, name: str) -> bool:
        if self.cfg.policy != "tiered" or name not in self.streamed:
            return False
        if name not in self.blocks:
            raise RuntimeError(f"tiered cache has no residual for streamed unit {name}")
        return True

    def put_block(self, name: str, sigma: float, residuals: tuple) -> None:
        h = self.blocks.setdefault(name, _Hist())
        rs = tuple(r.detach().to(self.store_dtype) for r in residuals)
        if not self.host or rs[0].device.type != "cuda":
            h.push(sigma, rs, self.keep)
            return
        # Host tier: once the history is full, overwrite the oldest entry's buffers.
        bufs = h.items[0][1] if len(h.items) >= self.keep else tuple(self._pinned(r) for r in rs)
        cur = torch.cuda.current_stream(rs[0].device)
        if self._d2h is None:
            self._dev = rs[0].device
            self._d2h = torch.cuda.Stream(self._dev)
        # Waiting for the compute stream also orders this write after any pending read of ``bufs``.
        self._d2h.wait_stream(cur)
        with torch.cuda.stream(self._d2h):
            for b, r in zip(bufs, rs):
                b.copy_(r, non_blocking=True)
                r.record_stream(self._d2h)  # keep the device tensor alive until the copy ran
            ready = torch.cuda.Event()
            ready.record(self._d2h)
        h.push(sigma, bufs, self.keep, ready)

    def _pinned(self, like: torch.Tensor) -> torch.Tensor:
        """A pinned host buffer shaped like ``like``, from one pool sized for every streamed block."""
        nbytes = like.numel() * like.element_size()
        if self._pool is None:
            per_block = 2 * align_up(nbytes, 4096) + 4096  # double blocks hold img + txt residuals
            self._pool = PinnedPool(self.keep * len(self.streamed) * per_block)
        try:
            raw = self._pool.take(nbytes)
        except MemoryError:  # estimate was short: fall back to PyTorch's pinned allocator
            return torch.empty(like.shape, dtype=like.dtype, pin_memory=True)
        return raw.view(like.dtype).view(like.shape)

    def block_residual(self, name: str, sigma: float) -> tuple:
        h = self.blocks[name]
        if not any(ev is not None for _, _, ev in h.items):
            return h.predict(sigma, self.cfg.predict)
        dev = self._dev
        cur = torch.cuda.current_stream(dev)
        items = []
        for s, ts, ev in h.items:
            cur.wait_event(ev)
            items.append((s, tuple(t.to(dev, non_blocking=True) for t in ts), None))
        return _Hist(items).predict(sigma, self.cfg.predict)

    def close(self) -> None:
        """Free cached residuals (and the pinned pool, once pending copies are done)."""
        if self._d2h is not None:
            self._d2h.synchronize()
        self.blocks.clear()
        self.tail = _Hist()
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------ reporting
    def summary(self) -> str:
        if not self.active:
            return "step cache off"
        n = len(self.log)
        approx = sum(1 for _, dec, _ in self.log if dec != FULL)
        dists = " ".join(f"{dec[0]}{'' if d is None else f'{d:.3f}'}" for _, dec, d in self.log)
        return f"step cache: {approx}/{n} steps approximated ({self.cfg.policy}); per-step: {dists}"
