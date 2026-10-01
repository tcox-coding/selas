"""T5 v1.1 encoder (T5-XXL for FLUX) as functions over unit views, streamed block by block.

Runs in fp32: T5 activations overflow fp16, and Turing has no native bf16.
Weights stay in their stored dtype (fp8/bf16/fp16) and are upcast one Linear
at a time. The container has units ``globals`` (shared embedding + final norm)
and ``block.{i}`` (HF names relative to ``encoder.block.{i}.``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..store import UnitView, WeightStore


@dataclass
class T5Config:
    d_model: int = 4096
    num_heads: int = 64
    d_kv: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    max_distance: int = 128
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> "T5Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def relative_position_bucket(rel: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
    """Bidirectional T5 bucketing (HF ``_relative_position_bucket``)."""
    num_buckets //= 2
    buckets = (rel > 0).to(torch.long) * num_buckets
    rel = rel.abs()
    max_exact = num_buckets // 2
    is_small = rel < max_exact
    large = max_exact + (
        torch.log(rel.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
    return buckets + torch.where(is_small, rel, large)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


def _embed(g: UnitView, ids: torch.Tensor) -> torch.Tensor:
    name = "shared.weight"
    if g.codec(name) == "raw":
        table = g.raw(name)  # gather rows in the stored dtype, then upcast only those
        rows = table.view(torch.uint8).view(table.shape[0], -1).index_select(0, ids.reshape(-1))
        return rows.view(table.dtype).view(*ids.shape, table.shape[1]).float()
    return F.embedding(ids, g.get(name, torch.float32))


class T5Encoder:
    def __init__(self, store: WeightStore, cfg: T5Config):
        self.store = store
        self.cfg = cfg
        self.blocks = [f"block.{i}" for i in range(cfg.num_layers)]

    def _block(self, w: UnitView, h: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        f32 = torch.float32
        b, l, _ = h.shape
        x = _rms(h, w.get("layer.0.layer_norm.weight", f32), cfg.eps)

        def proj(n):
            return w.linear(x, f"layer.0.SelfAttention.{n}", f32).view(b, l, cfg.num_heads, cfg.d_kv).transpose(1, 2)

        a = F.scaled_dot_product_attention(proj("q"), proj("k"), proj("v"), attn_mask=bias, scale=1.0)
        h = h + w.linear(a.transpose(1, 2).reshape(b, l, cfg.num_heads * cfg.d_kv), "layer.0.SelfAttention.o", f32)
        del a
        x = _rms(h, w.get("layer.1.layer_norm.weight", f32), cfg.eps)
        ff = F.gelu(w.linear(x, "layer.1.DenseReluDense.wi_0", f32), approximate="tanh")
        ff = ff * w.linear(x, "layer.1.DenseReluDense.wi_1", f32)
        return h + w.linear(ff, "layer.1.DenseReluDense.wo", f32)

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, micro_batch: int = 4) -> torch.Tensor:
        """ids [B, L] -> last hidden state [B, L, d_model] fp32 (no attention mask, like FLUX).

        Block-major: every prompt passes through a block while it is loaded,
        ``micro_batch`` prompts at a time, so N prompts cost one streaming pass.
        """
        cfg = self.cfg
        dev = self.store.device
        f32 = torch.float32
        ids = ids.to(dev)
        b, l = ids.shape
        g = self.store.view("globals")
        h = _embed(g, ids)
        bias = None
        with self.store.stream(self.blocks, cyclic=False) as st:
            for name in self.blocks:
                w = st.acquire(name)
                if bias is None:
                    pos = torch.arange(l, device=dev)
                    buckets = relative_position_bucket(pos[None, :] - pos[:, None], cfg.num_buckets, cfg.max_distance)
                    table = w.get("layer.0.SelfAttention.relative_attention_bias.weight", f32)  # [buckets, H]
                    bias = F.embedding(buckets, table).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, H, L, L]
                h = torch.cat([self._block(w, h[a : a + micro_batch], bias) for a in range(0, b, micro_batch)])
                st.release(name)
        return _rms(h, g.get("final_layer_norm.weight", f32), cfg.eps)
