"""Tier-9 synthetic toy: reserve subsystem in isolation.

See ``tests/fixtures/flex_toy_reserve.py`` for the construction and
closed-form derivation.
"""
from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool

from flex_toy_reserve import data, expected_obj


@pytest.mark.smoke
def test_flex_toy_reserve_obj():
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal
    expected = expected_obj()
    rel = abs(sol.obj - expected) / max(1.0, abs(expected))
    assert rel < 1e-9, (
        f"obj mismatch: polar_high={sol.obj!r}, expected={expected!r}, rel={rel!r}"
    )


@pytest.mark.smoke
def test_flex_toy_reserve_vq_binds():
    """vq_reserve must take the value 0.6 — the slack must be priced
    because reservation can't be fully met given the maxToSink coupling."""
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal
    vq = sol.value("vq_reserve")
    assert vq.height == 2, f"expected 2 (r, ud, g, d, t) rows, got {vq.height}"
    for v in vq["value"].to_list():
        assert abs(v - 0.6) < 1e-9, (
            f"vq_reserve should be 0.6 (= (50-20)/50), got {v}"
        )
