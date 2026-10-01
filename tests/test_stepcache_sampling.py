import math

import pytest
import torch

from selas.sampling import get_schedule, image_ids, latent_hw, pack, time_shift, unpack
from selas.stepcache import FULL, HYBRID, SKIP, StepCache, StepCacheConfig


def test_schedule_endpoints_and_shift():
    s = get_schedule(28, 4096)
    assert len(s) == 29 and s[0] == 1.0 and s[-1] == 0.0
    assert all(a > b for a, b in zip(s, s[1:]))
    mu = 1.15  # seq 4096 -> max_shift
    assert s[14] == pytest.approx(math.exp(mu) / (math.exp(mu) + (1 / 0.5 - 1)))
    assert get_schedule(4, 256, shift=False) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert time_shift(1.0, 1.0, 0.0) == 0.0


def test_pack_unpack_roundtrip():
    x = torch.randn(2, 16, 24, 40)
    p = pack(x)
    assert p.shape == (2, 12 * 20, 64)
    assert torch.equal(unpack(p, 24 * 8, 40 * 8), x)
    ids = image_ids(24 * 8, 40 * 8)
    assert ids.shape == (240, 3) and ids[21].tolist() == [0.0, 1.0, 1.0]
    assert latent_hw(1000, 1000) == (126, 126)


def _cache(policy="fbcache", **kw):
    return StepCache(StepCacheConfig(policy=policy, **kw), num_steps=10, streamed={"double.3"}, store_dtype=torch.float32)


def test_cache_warmup_tail_and_threshold():
    c = _cache(threshold=0.1, warmup=2, tail_full=2, max_consecutive=10)
    r = torch.ones(4, 4)
    decisions = [c.decide(s, 1 - s / 10, r * (1 + 0.01 * s))[0] for s in range(10)]
    assert decisions[:2] == [FULL, FULL]
    assert decisions[-2:] == [FULL, FULL]
    assert all(d == SKIP for d in decisions[2:8])


def test_cache_max_consecutive():
    c = _cache(threshold=1.0, warmup=1, tail_full=0, max_consecutive=2)
    r = torch.ones(3)
    ds = [c.decide(s, 0.5, r)[0] for s in range(7)]
    assert ds == [FULL, SKIP, SKIP, FULL, SKIP, SKIP, FULL]


def test_cache_large_change_forces_full():
    c = _cache(threshold=0.05)
    c.decide(0, 1.0, torch.ones(8))
    d, dist = c.decide(1, 0.9, torch.full((8,), 2.0))
    assert d == FULL and dist == pytest.approx(1.0)


def test_tiered_policy_hybrid_and_substitution():
    c = _cache("tiered", threshold=0.5)
    c.decide(0, 1.0, torch.ones(2))
    assert c.records("double.3") and not c.records("double.1")
    c.put_block("double.3", 1.0, (torch.ones(2), torch.zeros(2)))
    d, _ = c.decide(1, 0.9, torch.ones(2))
    assert d == HYBRID
    assert c.substitutes("double.3") and not c.substitutes("double.1")
    ri, rt = c.block_residual("double.3", 0.9)
    assert torch.equal(ri, torch.ones(2))


def test_linear_prediction_extrapolates():
    c = StepCache(StepCacheConfig(policy="fbcache", predict="linear"), 10)
    c.put_tail(1.0, torch.tensor([1.0]))
    c.put_tail(0.8, torch.tensor([2.0]))
    # slope -5 per unit sigma: at 0.6 -> 3.0
    assert c.tail_residual(0.6).item() == pytest.approx(3.0)
    c2 = StepCache(StepCacheConfig(policy="fbcache", predict="reuse"), 10)
    c2.put_tail(1.0, torch.tensor([1.0]))
    c2.put_tail(0.8, torch.tensor([2.0]))
    assert c2.tail_residual(0.6).item() == 2.0


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        StepCacheConfig(policy="lru")
