"""Tier-9 synthetic toy: §8.1 existing-entity fixed-cost tripwire.

See ``tests/fixtures/flex_toy_existing_fixed_cost.py`` for the
construction and closed-form derivation.

The §8.1 constant is gated by the
``include_existing_fixed_cost`` flag on ``build_flextool``.  Default
False so flexpy's ``sol.obj`` matches flextool's published v_obj
parquet (which doesn't include the constant — the AMPL→HiGHS bridge
drops it).  This test enables the flag to verify the closed-form
derivation that includes §8.1.
"""
from __future__ import annotations

from polar_high_opt import Problem
from flextool.engine_polars import build_flextool

from flex_toy_existing_fixed_cost import data, expected_obj


def test_flex_toy_existing_fixed_cost_obj():
    d = data()
    pb = Problem()
    build_flextool(pb, d, include_existing_fixed_cost=True)
    sol = pb.solve()
    assert sol.optimal
    expected = expected_obj()
    rel = abs(sol.obj - expected) / max(1.0, abs(expected))
    assert rel < 1e-9, (
        f"obj mismatch: flexpy={sol.obj!r}, expected={expected!r}, rel={rel!r}"
    )
