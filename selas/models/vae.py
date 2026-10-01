"""FLUX autoencoder decoder (BFL ``ae.py`` / LDM key layout) as functions over one resident unit.

Runs in fp32 by default. ``decode_tiled`` decodes overlapping latent tiles and
blends them linearly for when the full-resolution activations do not fit; the
mid-block attention then sees a tile instead of the whole image, so tiled output
is close to but not identical to untiled output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..store import UnitView, WeightStore


@dataclass
class VaeConfig:
    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    out_ch: int = 3
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159

    @classmethod
    def from_dict(cls, d: dict) -> "VaeConfig":
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "ch_mult" in kw:
            kw["ch_mult"] = tuple(kw["ch_mult"])
        return cls(**kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ch_mult"] = list(self.ch_mult)
        return d


class VaeDecoder:
    def __init__(self, store: WeightStore, cfg: VaeConfig, dtype: torch.dtype = torch.float32):
        self.store = store
        self.cfg = cfg
        self.dt = dtype

    def _w(self) -> UnitView:
        return self.store.view("decoder")

    def _conv(self, w: UnitView, x, p, padding=1):
        return F.conv2d(x, w.get(p + ".weight", self.dt), w.get(p + ".bias", self.dt), padding=padding)

    def _gn(self, w: UnitView, x, p):
        return F.group_norm(x, 32, w.get(p + ".weight", self.dt), w.get(p + ".bias", self.dt), eps=1e-6)

    def _resnet(self, w: UnitView, x, p):
        h = F.silu(self._gn(w, x, p + ".norm1"))
        h = self._conv(w, h, p + ".conv1")
        h = F.silu(self._gn(w, h, p + ".norm2"))
        h = self._conv(w, h, p + ".conv2")
        if w.has(p + ".nin_shortcut.weight"):
            x = self._conv(w, x, p + ".nin_shortcut", padding=0)
        return x + h

    def _attn(self, w: UnitView, x, p):
        b, c, hh, ww = x.shape
        h = self._gn(w, x, p + ".norm")

        def proj(n):
            return self._conv(w, h, f"{p}.{n}", padding=0).reshape(b, 1, c, hh * ww).transpose(2, 3)

        a = F.scaled_dot_product_attention(proj("q"), proj("k"), proj("v"))
        a = a.transpose(2, 3).reshape(b, c, hh, ww)
        return x + self._conv(w, a, p + ".proj_out", padding=0)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, 16, h, w] (sampler latents) -> image [B, 3, 8h, 8w] in [-1, 1] (fp32)."""
        cfg = self.cfg
        w = self._w()
        z = z.to(self.store.device, torch.float32) / cfg.scale_factor + cfg.shift_factor
        h = self._conv(w, z.to(self.dt), "conv_in")
        h = self._resnet(w, h, "mid.block_1")
        h = self._attn(w, h, "mid.attn_1")
        h = self._resnet(w, h, "mid.block_2")
        for lvl in reversed(range(len(cfg.ch_mult))):
            for i in range(cfg.num_res_blocks + 1):
                h = self._resnet(w, h, f"up.{lvl}.block.{i}")
            if lvl != 0:
                h = F.interpolate(h, scale_factor=2.0, mode="nearest")
                h = self._conv(w, h, f"up.{lvl}.upsample.conv")
        h = F.silu(self._gn(w, h, "norm_out"))
        h = self._conv(w, h, "conv_out")
        return h.float().clamp_(-1, 1)

    @torch.no_grad()
    def decode_tiled(self, z: torch.Tensor, tile: int = 64, overlap: int = 16) -> torch.Tensor:
        """Decode overlapping ``tile``x``tile`` latent tiles and blend them."""
        b, _, h, w = z.shape
        if h <= tile and w <= tile:
            return self.decode(z)
        up = 2 ** (len(self.cfg.ch_mult) - 1)
        stride = tile - overlap
        out = torch.zeros(b, 3, h * up, w * up, dtype=torch.float32, device=self.store.device)
        weight = torch.zeros(1, 1, h * up, w * up, dtype=torch.float32, device=self.store.device)
        ys = list(range(0, max(h - tile, 0) + 1, stride))
        xs = list(range(0, max(w - tile, 0) + 1, stride))
        if ys[-1] + tile < h:
            ys.append(h - tile)
        if xs[-1] + tile < w:
            xs.append(w - tile)
        ramp = overlap * up
        for y in ys:
            for x in xs:
                th, tw = min(tile, h - y), min(tile, w - x)
                dec = self.decode(z[:, :, y : y + th, x : x + tw])
                mask = torch.ones(1, 1, th * up, tw * up, device=self.store.device)
                if ramp:
                    r = torch.linspace(0, 1, ramp + 2, device=self.store.device)[1:-1]
                    if y > 0:
                        mask[:, :, :ramp, :] *= r.view(1, 1, -1, 1)
                    if y + th < h:
                        mask[:, :, -ramp:, :] *= r.flip(0).view(1, 1, -1, 1)
                    if x > 0:
                        mask[:, :, :, :ramp] *= r.view(1, 1, 1, -1)
                    if x + tw < w:
                        mask[:, :, :, -ramp:] *= r.flip(0).view(1, 1, 1, -1)
                out[:, :, y * up : (y + th) * up, x * up : (x + tw) * up] += dec * mask
                weight[:, :, y * up : (y + th) * up, x * up : (x + tw) * up] += mask
        return (out / weight.clamp_min(1e-6)).clamp_(-1, 1)
