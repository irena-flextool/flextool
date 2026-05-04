"""Tier-9 synthetic toy: capacity_margin in isolation.

See ``tests/fixtures/flex_toy_capacity_margin.py`` for the toy
construction and the closed-form derivation.

This test is the canonical regression for the ``capacityMargin``
constraint emission and the ``vq_capacity_margin`` slack-penalty obj
term.  Any wrong factor on the slack term, any wrong sign on the
inflow contribution, or a missing ``inv_group_cap`` cancellation will
fail this test with a single-line obj mismatch — pointing directly at
the broken term.
"""
from __future__ import annotations

import pytest

from flexpy import Problem
from flextool.engine_polars import build_flextool

from flex_toy_capacity_margin import data, expected_obj


@pytest.mark.smoke
def test_flex_toy_capacity_margin_obj():
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
def test_flex_toy_capacity_margin_vq_solves_to_gap():
    """vq_capacity_margin must equal floor − inflow_per_step at every t.

    With no producers and inv_group_cap = 1.0, the constraint reduces
    to vq_cap[g, p2020] >= 100 − 10 = 90 at each timestep.  The single
    vq variable picks up the binding value 90.0.
    """
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal
    vq = sol.value("vq_capacity_margin")
    assert vq.height == 1, f"expected 1 (g, d) row, got {vq.height}"
    val = float(vq["value"][0])
    assert abs(val - 90.0) < 1e-9, (
        f"vq_capacity_margin should be 100−10=90, got {val}"
    )
