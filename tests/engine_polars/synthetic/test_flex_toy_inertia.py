"""Tier-9 synthetic toy: inertia constraint in isolation.

See ``tests/fixtures/flex_toy_inertia.py`` for the construction and
closed-form derivation.
"""
from __future__ import annotations

import pytest

from polar_high_opt import Problem
from flextool.engine_polars import build_flextool

from flex_toy_inertia import data, expected_obj


@pytest.mark.smoke
def test_flex_toy_inertia_obj():
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
def test_flex_toy_inertia_no_slack_and_constraint_emitted():
    """vq_inertia must be 0 (constraint binds with margin), and the
    inertia_constraint must emit exactly |groupInertia| · |dt| rows."""
    d = data()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal

    # vq_inertia values — all zero since LHS = 500 ≥ RHS = 200.
    vq = sol.value("vq_inertia")
    assert vq.height == 2, f"expected 2 (g, d, t) rows, got {vq.height}"
    for v in vq["value"].to_list():
        assert abs(v) < 1e-9, f"vq_inertia should be 0 (constraint not binding), got {v}"

    # Constraint emission: 1 group × 2 timesteps × 1 period = 2 rows.
    rows = pb.cstr_row_count("inertia_constraint")
    assert rows == 2, f"expected 2 inertia_constraint rows, got {rows}"
