"""Defaults that encode measured decisions (docs/PLAN.md §7), and their opt-outs."""

from __future__ import annotations

from selas.cli import _cache_cfg, _runtime, build_parser
from selas.pipeline import RuntimeOptions
from selas.stepcache import StepCacheConfig


def test_measured_defaults():
    assert RuntimeOptions().direct_io is True
    assert StepCacheConfig().predict == "linear"
    for cmd in ("generate", "plan"):
        a = build_parser().parse_args([cmd, "--model", "m"])
        assert _runtime(a).direct_io is True
        assert _cache_cfg(a).predict == "linear"


def test_opt_outs():
    a = build_parser().parse_args(["generate", "--model", "m", "--no-direct-io", "--cache-predict", "reuse"])
    assert _runtime(a).direct_io is False
    assert _cache_cfg(a).predict == "reuse"
