"""Disk bandwidth: measured once per device and read mode, then served from the cache."""

from __future__ import annotations

import errno
import json
import os

import pytest

from selas import hw


@pytest.fixture
def data_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = tmp_path / "weights.bin"
    path.write_bytes(os.urandom(48 << 20))
    return path


def _cached(tmp_path) -> dict:
    return json.loads((tmp_path / "cache" / "selas" / "hw.json").read_text())["disks"]


def test_measure_disk_cold_reads(data_file):
    assert hw.measure_disk(str(data_file), direct=False) > 0
    try:
        assert hw.measure_disk(str(data_file), direct=True) > 0
    except OSError:
        pytest.skip("filesystem refuses O_DIRECT")


def test_measured_once_then_cached(data_file, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hw, "measure_disk", lambda path, direct=False, **kw: calls.append(direct) or 3e9)
    assert hw.disk_bandwidth(data_file, direct=True) == 3e9
    assert hw.disk_bandwidth(data_file, direct=True) == 3e9
    assert calls == [True]
    key = f"{hw.disk_key(data_file)}|direct"
    assert _cached(tmp_path)[key]["o_direct"] is True
    assert hw.disk_bandwidth(data_file, direct=False) == 3e9  # the buffered mode has its own entry
    assert calls == [True, False]
    monkeypatch.setattr(hw, "measure_disk", lambda path, direct=False, **kw: 1e9)
    assert hw.disk_bandwidth(data_file, direct=True, remeasure=True) == 1e9
    assert hw.disk_bandwidth(data_file, direct=True) == 1e9


def test_refused_o_direct_falls_back_to_buffered(data_file, tmp_path, monkeypatch):
    calls = []

    def fake(path, direct=False, **kw):
        calls.append(direct)
        if direct:
            raise OSError(errno.EINVAL, "O_DIRECT refused")
        return 5e8

    monkeypatch.setattr(hw, "measure_disk", fake)
    assert hw.disk_bandwidth(data_file, direct=True) == 5e8
    assert hw.disk_bandwidth(data_file, direct=True) == 5e8  # cached under the requested mode: no re-probe
    assert calls == [True, False]
    assert _cached(tmp_path)[f"{hw.disk_key(data_file)}|direct"]["o_direct"] is False


def test_unmeasurable_returns_none(data_file, monkeypatch):
    def fail(path, direct=False, **kw):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(hw, "measure_disk", fail)
    assert hw.disk_bandwidth(data_file, direct=False) is None


def test_legacy_per_gpu_disk_bw_is_ignored(tmp_path, monkeypatch):
    """Old `selas bench` stored disk_bw in the GPU entry; the fallback must stay the documented default."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    dev = hw.DeviceInfo("FakeGPU", 0, (7, 5), 11 << 30, 10 << 30)
    monkeypatch.setattr(hw, "device_info", lambda device: dev)
    path = tmp_path / "cache" / "selas" / "hw.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({hw._profile_key(dev): {"h2d_bw": 1.2e10, "disk_bw": 1.7e9,
                                                         "measured": ["h2d_bw", "matmul_flops", "disk_bw"]}}))
    prof = hw.load_profile(None, quick_measure=False)
    assert prof.h2d_bw == 1.2e10
    assert prof.disk_bw == hw.HwProfile.disk_bw
    assert "disk_bw" not in prof.measured
