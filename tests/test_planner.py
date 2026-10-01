import pytest

from selas.planner import UnitCost, make_plan, simulate
from selas.store import DISK, HOST, VRAM

MB = 2**20


def flux_like(n_double=6, n_single=12, double_mb=432, single_mb=216, compute=0.02):
    """Doubles: twice the bytes of singles for the same compute (as in FLUX)."""
    cyc = [UnitCost(f"double.{i}", double_mb * MB, compute, "double") for i in range(n_double)]
    cyc += [UnitCost(f"single.{i}", single_mb * MB, compute, "single") for i in range(n_single)]
    once = [UnitCost(f"double.{i}.mod", 216 * MB, 1e-4, "double_mod") for i in range(n_double)]
    return cyc, once


def test_simulate_all_resident_is_pure_compute():
    order = ["a", "b", "c"]
    tiers = {n: VRAM for n in order}
    step, comp = simulate(order, tiers, {n: MB for n in order}, {n: 0.1 for n in order}, 0, 0, 1e9, 1e9)
    assert step == pytest.approx(0.3) and comp == pytest.approx(0.3)


def test_simulate_streaming_hidden_when_bandwidth_is_high():
    order = [f"u{i}" for i in range(10)]
    tiers = {n: HOST for n in order}
    size = {n: 100 * MB for n in order}
    comp = {n: 0.05 for n in order}
    step, c = simulate(order, tiers, size, comp, 3 * 100 * MB, 0, 1e12, 1e12)
    assert step == pytest.approx(c, rel=0.01)
    slow, _ = simulate(order, tiers, size, comp, 3 * 100 * MB, 0, 1e9, 1e9)  # 0.1 s per copy > 0.05 compute
    assert slow == pytest.approx(10 * 100 * MB / 1e9, rel=0.02)


def test_disk_tier_slower_than_host():
    order = [f"u{i}" for i in range(8)]
    size = {n: 100 * MB for n in order}
    comp = {n: 0.01 for n in order}
    host, _ = simulate(order, {n: HOST for n in order}, size, comp, 300 * MB, 0, 10e9, 1e9)
    disk, _ = simulate(order, {n: DISK for n in order}, size, comp, 300 * MB, 300 * MB, 10e9, 1e9)
    assert disk > host


def test_plan_respects_budgets_and_forced_units():
    cyc, once = flux_like()
    vram = 4 * 1024 * MB
    ram = 3 * 1024 * MB
    p = make_plan(cyc, once, vram, ram, 11e9, 2e9, force_vram={"double.0"})
    names = [u.name for u in cyc]
    assert p.tiers["double.0"] == VRAM
    assert p.bytes_in(VRAM, names) + p.arena_bytes <= vram
    assert p.bytes_in(HOST) + p.staging_bytes <= ram
    assert set(p.tiers) == {u.name for u in cyc + once}


def test_more_vram_never_predicts_slower():
    cyc, once = flux_like()
    prev = None
    for gb in (2, 3, 4, 6, 8):
        p = make_plan(cyc, once, gb * 1024 * MB, 64 * 1024 * MB, 11e9, 2e9)
        if prev is not None:
            assert p.step_s <= prev * 1.01
        prev = p.step_s


def test_transfer_bound_prefers_double_blocks_for_residency():
    cyc, once = flux_like(compute=0.005)  # transfer-bound: residency matters
    p = make_plan(cyc, once, 3 * 1024 * MB, 64 * 1024 * MB, 11e9, 2e9)
    res = [n for n in p.tiers if p.tiers[n] == VRAM]
    assert res and all(n.startswith("double.") for n in res)


def test_everything_fits_means_no_cycle_streaming():
    cyc, once = flux_like(n_double=2, n_single=2)
    p = make_plan(cyc, once, 64 * 1024 * MB, 64 * 1024 * MB, 11e9, 2e9)
    assert all(p.tiers[u.name] == VRAM for u in cyc)
    assert p.arena_bytes >= 2 * 216 * MB  # still needs room to stream modulation units once


def test_insufficient_vram_raises():
    cyc, once = flux_like()
    with pytest.raises(MemoryError):
        make_plan(cyc, once, 100 * MB, 64 * 1024 * MB, 11e9, 2e9)


def test_ram_overflow_goes_to_disk_with_staging():
    cyc, once = flux_like()
    p = make_plan(cyc, once, 2 * 1024 * MB, 1024 * MB, 11e9, 2e9)
    assert p.count(DISK, [u.name for u in cyc]) > 0
    assert p.staging_bytes >= 2 * 216 * MB
