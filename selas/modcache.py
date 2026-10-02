"""Hoisted modulation vectors, cached on disk per image conditioning.

FLUX's adaLN modulations depend only on the pooled CLIP vector, the guidance
value and the timestep schedule, not on the latent (see
:meth:`selas.models.flux.FluxRunner.prepare`). Computing them needs the
modulation layers: 27 % of the weights, 6 GiB at fp16, streamed once per image
batch. Once an image's vectors are cached, a new seed with the same prompt, size
and step count needs none of those weights, and when every image of a run is
cached they are not even loaded.

Entries are exact: vectors are computed per image (never batched), so the same
inputs give the same bits in any batch. An entry is ~2 MiB per step (59 MiB for
28 steps); the directory is kept under a size limit, least recently used first.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from .util import GiB, log, sha256_hex, user_cache_dir

VERSION = "selas-mods-v1"


class ModCache:
    def __init__(self, model_dir: Path, container_id: str, dtype: torch.dtype, enabled: bool = True,
                 limit_bytes: int = 4 * GiB):
        self.container_id = container_id
        self.dtype = dtype
        self.enabled = enabled
        self.limit = limit_bytes
        self.root: Path | None = None
        if not enabled:
            return
        for root in (Path(model_dir) / ".selas" / "mods", user_cache_dir() / "mods" / container_id):
            try:
                root.mkdir(parents=True, exist_ok=True)
                probe = root / ".w"
                probe.touch()
                probe.unlink()
                self.root = root
                break
            except OSError:
                continue
        if self.root is None:
            self.enabled = False

    def key(self, pooled: torch.Tensor, guidance: float, sigmas) -> str:
        """``pooled``: one image's CLIP vector [768]; ``sigmas``: the full schedule."""
        digest = hashlib.sha256(pooled.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
        return sha256_hex(VERSION, self.container_id, str(self.dtype), float(guidance),
                          [float(s) for s in sigmas], digest)[:32]

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        return self.enabled and self._path(key).exists()

    def load(self, key: str) -> dict[str, torch.Tensor] | None:
        """{main unit: [S, kD]} (CPU), or None."""
        if not self.enabled:
            return None
        from safetensors.torch import load_file

        path = self._path(key)
        try:
            d = load_file(str(path))
            os.utime(path)  # recently used
        except Exception:
            return None
        return d

    def save(self, key: str, mods: dict[str, torch.Tensor]) -> None:
        """``mods``: {main unit: [S, 1, kD]} for one image."""
        if not self.enabled:
            return
        from safetensors.torch import save_file

        path = self._path(key)
        tmp = path.with_name(f"{key}.{os.getpid()}.tmp")
        try:
            save_file({n: t[:, 0].contiguous().cpu() for n, t in mods.items()}, str(tmp))
            os.replace(tmp, path)
        except OSError as e:
            log(f"modulation cache: could not write {path.name} ({e})")
            tmp.unlink(missing_ok=True)
            return
        self._evict()

    def _evict(self) -> None:
        assert self.root is not None
        entries = []
        for p in self.root.glob("*.safetensors"):
            try:
                st = p.stat()
                entries.append((st.st_mtime, st.st_size, p))
            except OSError:
                continue
        total = sum(e[1] for e in entries)
        for _, size, p in sorted(entries):
            if total <= self.limit:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass
