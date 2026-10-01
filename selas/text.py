"""Prompt -> (T5 sequence embedding, CLIP pooled vector), with an on-disk cache.

The cache is keyed by the prompt and the exact T5/CLIP containers, so
re-running a prompt with new seeds, sizes or step counts never loads T5-XXL.
Entries live next to the model in ``<model>/.selas/prompts/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .container import Container
from .models.clip import ClipConfig, ClipTextEncoder
from .models.t5 import T5Config, T5Encoder
from .store import DISK, VRAM, WeightStore
from .util import align_up, log, sha256_hex, user_cache_dir


def _clip_bpe(path: Path):
    """CLIP BPE built directly with `tokenizers`, mirroring transformers' CLIPConverter."""
    import json

    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import BPE

    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    lines = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
    merges = [tuple(ln.split()) for ln in lines if ln and not ln.startswith("#version")]
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, continuing_subword_prefix="", end_of_word_suffix="</w>",
                        fuse_unk=False, unk_token="<|endoftext|>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(r"\s+"), " "), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""), behavior="removed", invert=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.RobertaProcessing(
        sep=("<|endoftext|>", vocab["<|endoftext|>"]), cls=("<|startoftext|>", vocab["<|startoftext|>"]),
        add_prefix_space=False, trim_offsets=False)
    return tok, vocab["<|endoftext|>"]


def load_tokenizer(path: str | os.PathLike):
    """(tokenizer, pad_id): transformers' AutoTokenizer if it loads, else a `tokenizers` fallback."""
    path = Path(path)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        return tok, tok.pad_token_id
    except Exception as e:  # version quirks: fall back to the tokenizers library directly
        log(f"AutoTokenizer failed for {path} ({type(e).__name__}); using the tokenizers fallback")
    if (path / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        pad = tok.token_to_id("<pad>")
        return tok, 0 if pad is None else pad
    if (path / "vocab.json").exists() and (path / "merges.txt").exists():
        return _clip_bpe(path)
    raise FileNotFoundError(f"no usable tokenizer files in {path}")


def tokenize(tok_pad, prompts: list[str], max_length: int) -> torch.Tensor:
    tok, pad_id = tok_pad
    if hasattr(tok, "encode_batch"):  # tokenizers.Tokenizer
        tok.enable_truncation(max_length)
        tok.enable_padding(length=max_length, pad_id=pad_id)
        return torch.tensor([e.ids for e in tok.encode_batch(prompts)], dtype=torch.long)
    enc = tok(prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    return enc["input_ids"]


class TextEncoders:
    def __init__(self, model_dir: Path, info: dict, device: torch.device, direct_io: bool = True, use_cache: bool = True):
        self.dir = Path(model_dir)
        self.info = info
        self.device = device
        self.direct_io = direct_io
        self.use_cache = use_cache
        comps = info["components"]
        self.t5_c = Container(self.dir / comps["t5"])
        self.clip_c = Container(self.dir / comps["clip"])
        self.max_t5 = int(info.get("defaults", {}).get("max_t5_tokens", 512))
        self.max_clip = int(self.clip_c.config.get("max_positions", 77))
        cache_root = self.dir / ".selas" / "prompts"
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            probe = cache_root / ".w"
            probe.touch()
            probe.unlink()
        except OSError:
            cache_root = user_cache_dir() / "prompts" / self.t5_c.id
            cache_root.mkdir(parents=True, exist_ok=True)
        self.cache_root = cache_root

    def _key(self, prompt: str) -> str:
        return sha256_hex("selas-prompt-v1", prompt, self.t5_c.id, self.clip_c.id, self.max_t5, self.max_clip)[:32]

    def _cache_path(self, prompt: str) -> Path:
        return self.cache_root / f"{self._key(prompt)}.safetensors"

    def encode(self, prompts: list[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Unique prompt -> (txt [L, 4096] fp32 CPU, pooled [768] fp32 CPU)."""
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        todo = []
        for p in dict.fromkeys(prompts):
            path = self._cache_path(p)
            if self.use_cache and path.exists():
                try:
                    d = load_file(str(path))
                    out[p] = (d["txt"], d["pooled"])
                    continue
                except Exception:
                    pass
            todo.append(p)
        if out:
            log(f"prompt cache: {len(out)} hit(s), {len(todo)} miss(es)")
        if not todo:
            return out
        tok_dir = self.dir / self.info["tokenizers"]["clip"]
        clip_ids = tokenize(load_tokenizer(tok_dir), todo, self.max_clip)
        t5_ids = tokenize(load_tokenizer(self.dir / self.info["tokenizers"]["t5"]), todo, self.max_t5)

        with WeightStore(self.clip_c, {"all": VRAM}, self.device, torch.float32, label="clip") as st:
            _, pooled = ClipTextEncoder(st, ClipConfig.from_dict(self.clip_c.config)).encode(clip_ids)
            pooled = pooled.cpu()

        # T5-XXL: a single pass, so residency buys nothing — stream every block from disk.
        t5c = self.t5_c
        blocks = [n for n in t5c.units if n != "globals"]
        tiers = {"globals": VRAM, **{n: DISK for n in blocks}}
        max_block = max(t5c.units[n].nbytes for n in blocks)
        arena = align_up(3 * max_block, 4096)
        with WeightStore(t5c, tiers, self.device, torch.float32, arena_bytes=arena, staging_bytes=3 * max_block,
                         direct_io=self.direct_io, label="t5") as st:
            txt = T5Encoder(st, T5Config.from_dict(t5c.config)).encode(t5_ids).cpu()

        for i, p in enumerate(todo):
            out[p] = (txt[i].contiguous(), pooled[i].contiguous())
            if self.use_cache:
                try:
                    save_file({"txt": out[p][0], "pooled": out[p][1]}, str(self._cache_path(p)), metadata={"prompt": p[:2000]})
                except OSError:
                    pass
        return out

    def close(self) -> None:
        self.t5_c.close()
        self.clip_c.close()
