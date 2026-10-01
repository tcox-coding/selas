"""CLIP-L text encoder (HF ``CLIPTextModel``) as functions over one resident unit, fp32.

FLUX uses the *pooled* output: the final-layer-normed hidden state at the EOS
token (the first occurrence of the largest id, as HF does for the original CLIP
tokenizer, where EOS 49407 is the largest id). No text projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..store import WeightStore


@dataclass
class ClipConfig:
    hidden: int = 768
    heads: int = 12
    layers: int = 12
    max_positions: int = 77
    eps: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict) -> "ClipConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


class ClipTextEncoder:
    def __init__(self, store: WeightStore, cfg: ClipConfig):
        self.store = store
        self.cfg = cfg

    @torch.no_grad()
    def encode(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids [B, L] -> (last_hidden_state [B, L, D], pooled [B, D]) in fp32."""
        cfg = self.cfg
        f32 = torch.float32
        w = self.store.view("all")
        ids = ids.to(self.store.device)
        b, l = ids.shape
        hd = cfg.hidden // cfg.heads
        h = F.embedding(ids, w.get("embeddings.token_embedding.weight", f32))
        h = h + w.get("embeddings.position_embedding.weight", f32)[:l][None]

        def ln(x, p):
            return F.layer_norm(x, (cfg.hidden,), w.get(p + ".weight", f32), w.get(p + ".bias", f32), cfg.eps)

        for i in range(cfg.layers):
            p = f"encoder.layers.{i}"
            x = ln(h, p + ".layer_norm1")

            def proj(n):
                return w.linear(x, f"{p}.self_attn.{n}", f32).view(b, l, cfg.heads, hd).transpose(1, 2)

            a = F.scaled_dot_product_attention(proj("q_proj"), proj("k_proj"), proj("v_proj"), is_causal=True)
            h = h + w.linear(a.transpose(1, 2).reshape(b, l, cfg.hidden), p + ".self_attn.out_proj", f32)
            x = ln(h, p + ".layer_norm2")
            x = w.linear(x, p + ".mlp.fc1", f32)
            x = x * torch.sigmoid(1.702 * x)  # quick_gelu
            h = h + w.linear(x, p + ".mlp.fc2", f32)
        h = ln(h, "final_layer_norm")
        eos = ids.to(torch.int64).argmax(dim=-1)
        pooled = h[torch.arange(b, device=h.device), eos]
        return h, pooled
