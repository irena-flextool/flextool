"""Tier-9 synthetic toy: delayed-process (hydro chain) in isolation.

See ``tests/fixtures/flex_toy_delay.py`` for the construction and
closed-form derivation.
"""
from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool

from flex_toy_delay import data, expected_obj


@pytest.mark.smoke
def test_flex_toy_delay_obj():
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal
    expected = expected_obj()
    rel = abs(sol.obj - expected) / max(1.0, abs(expected))
    assert rel < 1e-9, (
        f"obj mismatch: flexpy={sol.obj!r}, expected={expected!r}, rel={rel!r}"
    )


@pytest.mark.smoke
def test_flex_toy_delay_time_shift_correct():
    """Verify the input/output time-shift by 1 step:

    * Source-side v_flow at t01 = 0.1 (delivers demand at t02).
    * Sink-side v_flow at t01 = 0   (no demand at t01).
    """
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal

    flow = sol.value("v_flow")

    # Source-side at t01 (input edge from water_upstream into p_d) = 0.1.
    src_t01 = flow.filter(
        (flow["source"] == "water_upstream")
        & (flow["sink"] == "p_d")
        & (flow["t"] == "t01")
    )["value"][0]
    assert abs(src_t01 - 0.1) < 1e-9, (
        f"source-side v_flow at t01 should be 0.1 (delivers demand at t02), "
        f"got {src_t01}"
    )

    # Sink-side at t01 (output edge p_d → water_downstream) = 0.
    snk_t01 = flow.filter(
        (flow["source"] == "p_d")
        & (flow["sink"] == "water_downstream")
        & (flow["t"] == "t01")
    )["value"][0]
    assert abs(snk_t01) < 1e-9, (
        f"sink-side v_flow at t01 should be 0 (no demand at t01), got {snk_t01}"
    )
