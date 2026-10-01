"""Reading tensors out of checkpoints without loading whole files.

* :class:`TensorSource` — one or more safetensors files (mmap'd by safetensors).
* :class:`SourceView`  — the keys under a prefix (``model.diffusion_model.``,
  ``text_encoders.t5xxl.transformer.``, ...), with bnb-NF4 awareness.
* :class:`DiffusersFluxView` — a diffusers FluxTransformer2DModel presented under
  BFL key names (qkv/linear1 fused on the fly).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable

import torch
from safetensors import safe_open

from .codecs import EncodedTensor, nf4_import
from .keymap import bfl_from_diffusers

NF4_STATE = ".quant_state.bitsandbytes__nf4"

_ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}
if hasattr(torch, "float8_e4m3fn"):
    _ST_DTYPES["F8_E4M3"] = torch.float8_e4m3fn
    _ST_DTYPES["F8_E5M2"] = torch.float8_e5m2


def expand_paths(paths: Iterable[str | os.PathLike]) -> list[str]:
    out = []
    for p in paths:
        p = os.fspath(p)
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.safetensors")))
            if not files:
                raise FileNotFoundError(f"no .safetensors files in {p}")
            out += files
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


class TensorSource:
    def __init__(self, paths: Iterable[str | os.PathLike]):
        self.paths = expand_paths(paths)
        self._handles = []
        self._where: dict[str, object] = {}
        for p in self.paths:
            h = safe_open(p, framework="pt", device="cpu")
            self._handles.append(h)
            for k in h.keys():
                self._where[k] = h

    def keys(self) -> list[str]:
        return list(self._where)

    def has(self, k: str) -> bool:
        return k in self._where

    def get(self, k: str) -> torch.Tensor:
        return self._where[k].get_tensor(k)

    def shape(self, k: str) -> tuple[int, ...]:
        return tuple(self._where[k].get_slice(k).get_shape())

    def dtype(self, k: str) -> torch.dtype:
        return _ST_DTYPES[self._where[k].get_slice(k).get_dtype()]

    def find_prefix(self, suffix: str) -> str | None:
        """Prefix p such that p + suffix is a key (shortest match wins)."""
        hits = [k[: -len(suffix)] for k in self._where if k.endswith(suffix)]
        return min(hits, key=len) if hits else None


class SourceView:
    """Keys of ``src`` under ``prefix``, with the prefix stripped."""

    def __init__(self, src: TensorSource, prefix: str):
        self.src = src
        self.prefix = prefix

    def keys(self) -> list[str]:
        p = self.prefix
        return [k[len(p) :] for k in self.src.keys() if k.startswith(p)]

    def has(self, k: str) -> bool:
        return self.src.has(self.prefix + k)

    def get(self, k: str) -> torch.Tensor:
        return self.src.get(self.prefix + k)

    def dtype(self, k: str) -> torch.dtype:
        return self.src.dtype(self.prefix + k)

    def is_nf4(self, k: str) -> bool:
        return self.has(k + NF4_STATE)

    def nf4_state(self, k: str) -> dict:
        raw = self.get(k + NF4_STATE)
        return json.loads(bytes(raw.reshape(-1).tolist()).decode("utf-8"))

    def shape(self, k: str) -> tuple[int, ...]:
        if self.is_nf4(k):
            return tuple(int(s) for s in self.nf4_state(k)["shape"])
        return self.src.shape(self.prefix + k)

    def encoded_nf4(self, k: str, name: str) -> EncodedTensor:
        st = self.nf4_state(k)
        if st.get("quant_type", "nf4") != "nf4":
            raise NotImplementedError(f"{k}: bnb quant type {st.get('quant_type')} not supported")
        absmax = self.get(k + ".absmax")
        if self.has(k + ".nested_absmax"):  # double-quantized statistics (bnb compress_statistics=True)
            nested_absmax = self.get(k + ".nested_absmax").float()
            nested_code = self.get(k + ".nested_quant_map").float()
            nb = int(st["nested_blocksize"])
            q = absmax.reshape(-1).to(torch.long)
            scale = nested_absmax.repeat_interleave(nb)[: q.numel()]
            absmax = nested_code[q] * scale + float(st.get("nested_offset", 0.0))
        code = self.get(k + ".quant_map").float().tolist()
        return nf4_import(name, self.get(k), absmax, tuple(st["shape"]), int(st["blocksize"]), code)


class DiffusersFluxView:
    """A diffusers-format FLUX transformer exposed with BFL key names."""

    def __init__(self, src: TensorSource, prefix: str = ""):
        self.src = src
        self.prefix = prefix
        keys = [k[len(prefix) :] for k in src.keys() if k.startswith(prefix)]
        nd = len({k.split(".")[1] for k in keys if k.startswith("transformer_blocks.")})
        ns = len({k.split(".")[1] for k in keys if k.startswith("single_transformer_blocks.")})
        guidance = any(k.startswith("time_text_embed.guidance_embedder.") for k in keys)
        if any(k.endswith(NF4_STATE) for k in keys):
            raise NotImplementedError("bnb-quantized diffusers FLUX checkpoints are not supported; use a BFL-format file")
        self.map = bfl_from_diffusers(nd, ns, guidance)
        missing = [d for _, ds in self.map.values() for d in ds if not src.has(prefix + d)]
        if missing:
            raise KeyError(f"diffusers FLUX checkpoint is missing {len(missing)} expected keys, e.g. {missing[:3]}")

    def keys(self) -> list[str]:
        return list(self.map)

    def has(self, k: str) -> bool:
        return k in self.map

    def is_nf4(self, k: str) -> bool:
        return False

    def dtype(self, k: str) -> torch.dtype:
        return self.src.dtype(self.prefix + self.map[k][1][0])

    def shape(self, k: str) -> tuple[int, ...]:
        op, ds = self.map[k]
        shapes = [self.src.shape(self.prefix + d) for d in ds]
        if op == "cat":
            return (sum(s[0] for s in shapes),) + tuple(shapes[0][1:])
        return shapes[0]

    def get(self, k: str) -> torch.Tensor:
        op, ds = self.map[k]
        ts = [self.src.get(self.prefix + d) for d in ds]
        if op == "cat":
            return torch.cat(ts, dim=0)
        if op == "swap":
            a, b = ts[0].chunk(2, dim=0)
            return torch.cat((b, a), dim=0)
        return ts[0]
