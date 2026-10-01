import random

import pytest

from selas.arena import RingArena


def _overlap(a, b):
    return a.off < b.off + b.size and b.off < a.off + a.size


def test_fifo_wrap_and_tokens():
    r = RingArena(10 * 4096)
    e1, t = r.try_alloc(4 * 4096)
    assert e1.off == 0 and t is None
    e2, _ = r.try_alloc(4 * 4096)
    assert e2.off == 4 * 4096
    assert r.try_alloc(4 * 4096) is None  # only 2 pages left at the end, nothing released
    r.release(e1, "tok1")
    e3, t = r.try_alloc(4 * 4096)  # wraps into e1's space and must wait on e1's token
    assert e3.off == 0 and t == "tok1"
    r.release(e2, "tok2")
    r.release(e3, "tok3")
    e4, t = r.try_alloc(10 * 4096)  # everything released: whole ring, newest reclaimed token
    assert e4.off == 0 and t == "tok3"


def test_alloc_after_full_release_always_succeeds():
    r = RingArena(64 * 4096)
    live = []
    for size in (5, 17, 30, 12):
        e, _ = r.try_alloc(size * 4096)
        live.append(e)
    for e in live:
        r.release(e, object())
    e, tok = r.try_alloc(64 * 4096)
    assert e.off == 0 and tok is not None


def test_too_big_raises():
    r = RingArena(8 * 4096)
    with pytest.raises(ValueError):
        r.try_alloc(9 * 4096)


@pytest.mark.parametrize("seed", range(20))
def test_randomized_invariants(seed):
    rnd = random.Random(seed)
    cap = 50 * 4096
    r = RingArena(cap)
    live = []  # FIFO of unreleased entries
    released_tokens = []
    for step in range(2000):
        if live and (rnd.random() < 0.45 or len(live) > 6):
            e = live.pop(0)
            tok = ("t", step)
            r.release(e, tok)
            released_tokens.append(tok)
            continue
        size = rnd.randint(1, 20) * 4096 - rnd.randint(0, 4095)
        got = r.try_alloc(size)
        if got is None:
            assert live, "allocation failed with nothing live"
            continue
        e, tok = got
        assert e.size >= size and e.size % 4096 == 0
        assert 0 <= e.off and e.off + e.size <= cap
        for o in live:
            assert not _overlap(e, o), "new allocation overlaps a live entry"
        if tok is not None:
            assert tok in released_tokens
        live.append(e)
