"""Kernel plumbing that does not need a GPU: compile fallback and the pinned pool's page pre-faulting."""

from __future__ import annotations

import torch

from selas.models import flux
from selas.store import _prefault


def test_failed_compile_falls_back_to_eager(monkeypatch):
    def broken_compile(fn, **kw):
        def raise_(*a):
            raise RuntimeError("inductor exploded")
        return raise_

    monkeypatch.setattr(torch, "compile", broken_compile)
    k = flux.Kernels(compiled=True)
    x, y, g = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 1, 8)
    assert torch.equal(k.gated_add(x, g, y, True), flux.EAGER.gated_add(x, g, y, True))
    assert torch.equal(k.gated_add(x, g, y, True), flux.EAGER.gated_add(x, g, y, True))  # stays on eager


def test_errors_in_eager_kernels_still_raise(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)  # "compiled" is eager: nothing to fall back to
    k = flux.Kernels(compiled=True)
    try:
        k.gated_add(torch.randn(2, 3), torch.randn(4), torch.randn(5), False)
    except RuntimeError:
        return
    raise AssertionError("shape error was swallowed")


def test_prefault_touches_every_page_without_changing_size():
    buf = torch.ones(5 * 4096 + 17, dtype=torch.uint8)
    _prefault(buf, threads=3, min_bytes=4096)
    assert buf.numel() == 5 * 4096 + 17
    assert int(buf[::4096].sum()) == 0 and int(buf.sum()) == buf.numel() - 6  # one byte per page, nothing else


def test_compile_count_sees_compilations():
    before = flux.compile_count()
    f = torch.compile(lambda x: x * 3 + 1, backend="eager")
    f(torch.randn(4))
    assert flux.compile_count() > before
    f(torch.randn(4))  # cached: no new graph
    n = flux.compile_count()
    f(torch.randn(4))
    assert flux.compile_count() == n
