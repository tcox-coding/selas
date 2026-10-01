"""FLUX rectified-flow sampling helpers (schedule, noise, latent packing), after BFL's sampling.py."""

from __future__ import annotations

import math

import torch


def time_shift(mu: float, sigma: float, t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def lin_mu(seq_len: int, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> float:
    m = (y2 - y1) / (x2 - x1)
    return m * seq_len + (y1 - m * x1)


def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15, shift: bool = True) -> list[float]:
    """num_steps + 1 sigmas from 1 to 0 (resolution-dependent shift for dev)."""
    ts = [1.0 - i / num_steps for i in range(num_steps + 1)]
    if shift:
        mu = lin_mu(image_seq_len, y1=base_shift, y2=max_shift)
        ts = [time_shift(mu, 1.0, t) for t in ts]
    return ts


def latent_hw(height: int, width: int) -> tuple[int, int]:
    """Latent (16-channel, /8) spatial size; FLUX needs multiples of 16 pixels."""
    return 2 * math.ceil(height / 16), 2 * math.ceil(width / 16)


def get_noise(seed: int, height: int, width: int, channels: int = 16) -> torch.Tensor:
    """Deterministic noise generated on the CPU in fp32 (identical on every machine)."""
    h, w = latent_hw(height, width)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(1, channels, h, w, generator=g, dtype=torch.float32)


def pack(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)(W/2), C*4]  ('b c (h ph) (w pw) -> b (h w) (c ph pw)')."""
    b, c, h, w = x.shape
    return x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`pack` for an image of ``height`` x ``width`` pixels."""
    h, w = latent_hw(height, width)
    b, _, cc = x.shape
    c = cc // 4
    return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def image_ids(height: int, width: int) -> torch.Tensor:
    """Position ids [L, 3] for the packed latent grid: (0, row, col)."""
    h, w = latent_hw(height, width)
    h, w = h // 2, w // 2
    ids = torch.zeros(h, w, 3)
    ids[..., 1] = torch.arange(h)[:, None]
    ids[..., 2] = torch.arange(w)[None, :]
    return ids.reshape(h * w, 3)


def text_ids(length: int) -> torch.Tensor:
    return torch.zeros(length, 3)
