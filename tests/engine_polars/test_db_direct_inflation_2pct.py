"""Hand-checked parity test for the multi-year inflation cascade at
**rate = 0.02** (2%).

Background
----------
``audit/gamma_3f_value_audit.md`` (Part A, "Caveats" §2) documents that
``test_p_inflation_op_full_cascade_parity`` exercises the cascade
arithmetic only through the structural CSV-vs-DB-direct frame
equality assertion — the bit-exact analytical oracle was hand-checked
on the rate=0 fixture, where the cascade collapses to a degenerate
``1.0`` factor per period.

This test pins the cascade output to a hand-derived analytical oracle
on a fixture with **non-trivial 2 %% compounding** (4 periods × 5 years
each → 20 global years), so any future drift of the cascade helper
will fail loudly against the canonical formula rather than relying on
the CSV path as a silent oracle.

Fixture
-------
``tests/data/work_inflation_check/`` — derivative of
``wind_battery_invest_lifetime_renew`` with the spine-DB
``model.inflation_rate`` patched from 0.04 to 0.02 *before*
``runner.write_input``, so flextool's CSV writer and flexpy's
DB-direct ``p_inflation_op_full_cascade_from_source`` both see the
same rate-0.02 source.  Built by ``tests/_gen_inflation_check.py``.

Cascade algorithm (per ``flextool/flextoolrunner/preprocessing/period_calculated_params.py:280-322``):

For each (period d, year y in ``solve.years_represented``):

  - ``base[d, y] = Σ_{y' < y in global year set} pyr[d, y']``
  - ``until[d, y] = base[d, y] + pyr[d, y] × offset[d, y]``
  - ``factor_contribution[d, y] = pyr[d, y] × (1 + rate)^(-until[d, y])``
  - ``factor[d] = Σ_y factor_contribution[d, y]``

For the operations cascade ``offset = inflation_offset_operations``
(default 0.5, mid-period); for the investment cascade
``offset = inflation_offset_investment`` (default 0.0, end-of-prior-period).
The fixture's offsets are at their defaults.

Lifetime cascade for ``ed_entity_annual_discounted`` (per
``flextool/_derived_params.py:5848-5885``):

  - annuity = invest_cost × 1000 × r / (1 - (1/(1+r))^lifetime)
  - For ``reinvest_automatic`` (default lifetime_method):
      disc[e, d] = annuity × Σ_{d' ∈ period_in_use, pdy[d'] ≥ pdy[d]}
                                 inv_factor[d']
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from polar_high_opt import Problem
from flextool.engine_polars import (
    SpineDbReader,
    build_flextool,
    load_flextool,
)

from _golden import assert_obj_within

DATA = Path(__file__).resolve().parent / "data"
WORK = DATA / "work_inflation_check"
SQLITE = WORK / "tests.sqlite"
SCENARIO = "wind_battery_invest_lifetime_renew"

# ---------------------------------------------------------------------------
# Hand-derived expected values (rate = 0.02; 4 periods × 5 years; offset_ops =
# 0.5; offset_inv = 0.0; weight 1.0 per year).
#
#     ops[p2020] = Σ_{y∈[0..4]} 1.02^-(y+0.5)
#                = 1.02^-0.5 + 1.02^-1.5 + 1.02^-2.5 + 1.02^-3.5 + 1.02^-4.5
#                = 4.760361 (verified to 15 digits below)
#     ops[p2025] = Σ_{y∈[5..9]}  1.02^-(y+0.5) = 4.311605
#     ops[p2030] = Σ_{y∈[10..14]} 1.02^-(y+0.5) = 3.905154
#     ops[p2035] = Σ_{y∈[15..19]} 1.02^-(y+0.5) = 3.537018
#
#     inv[p2020] = Σ_{y∈[0..4]}  1.02^-(y+0.0) = 4.807729
#     inv[p2025] = 4.354508
#     inv[p2030] = 3.944012
#     inv[p2035] = 3.572213
#
# These numbers are computed live in the test (no hand-typed digits) so
# Python's IEEE-754 implementation is the oracle; the comments above
# show enough digits for a reader to verify by hand.
# ---------------------------------------------------------------------------

R = 0.02                           # patched model.inflation_rate
PERIODS = ["p2020", "p2025", "p2030", "p2035"]
PDY = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}
OFFSET_OPS = 0.5                   # flextool default (period_calculated_params.py:226)
OFFSET_INV = 0.0                   # flextool default (period_calculated_params.py:223)


def _years_for(d: str) -> range:
    """Return the 5 global years bound to period d (0..4 for p2020,
    5..9 for p2025, …).  Per
    ``solve_data/p_years_until_dispatch.csv``.
    """
    base = PDY[d]
    return range(int(base), int(base) + 5)


def _ops_factor(d: str) -> float:
    """Hand-derived ``p_inflation_factor_operations_yearly[d]``."""
    return sum((1.0 + R) ** -(y + OFFSET_OPS) for y in _years_for(d))


def _inv_factor(d: str) -> float:
    """Hand-derived ``p_inflation_factor_investment_yearly[d]``."""
    return sum((1.0 + R) ** -(y + OFFSET_INV) for y in _years_for(d))


def _annuity(invest_cost_eur_per_kw: float,
              discount_rate: float,
              lifetime_years: float) -> float:
    """Annuity formula per ``flextool/_derived_params.py:5487-5498``::

        ann = invest_cost × 1000 × r / (1 - (1/(1+r))^n)

    Note the 1000× scaling: ``invest_cost`` is EUR/kW but the LP works
    in EUR/MW.
    """
    r, n = discount_rate, lifetime_years
    return invest_cost_eur_per_kw * 1000.0 * r / (1.0 - (1.0 / (1.0 + r)) ** n)


def _annual_disc_reinvest_automatic(annuity: float, d: str) -> float:
    """``reinvest_automatic`` discount window: unbounded sum across
    period_in_use of inv_factor[d'] for pdy[d'] ≥ pdy[d].
    """
    pdy_d = PDY[d]
    return annuity * sum(_inv_factor(d_all)
                          for d_all in PERIODS
                          if PDY[d_all] >= pdy_d)


# Bare-annuity oracle for wind_plant + battery (params from input/p_*.csv):
#   wind_plant: invest_cost=1000 EUR/kW, discount_rate=0.04, lifetime=5 yr.
#   battery:   invest_cost=200,           discount_rate=0.05, lifetime=10.
ANN_WIND = _annuity(1000.0, 0.04, 5.0)        # 224627.11349303348
ANN_BATT = _annuity(200.0, 0.05, 10.0)        # 25900.91499309131


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(param, *idx_keys: str) -> float:
    """Lookup ``param.frame``'s ``value`` column at the given index
    keys (in column order, last column is always ``value``).
    Returns the float.  Raises if the row is missing.
    """
    df = param.frame
    keys = [c for c in df.columns if c != "value"]
    assert len(idx_keys) == len(keys), (
        f"expected {len(keys)} idx keys for {keys}, got {idx_keys}")
    expr = pl.col(keys[0]) == idx_keys[0]
    for c, v in zip(keys[1:], idx_keys[1:]):
        expr = expr & (pl.col(c) == v)
    sub = df.filter(expr)
    assert sub.height == 1, (
        f"expected exactly 1 row at {idx_keys}; got {sub.height} rows:\n"
        f"{sub}\nFull frame:\n{df}"
    )
    return float(sub["value"][0])


def _approx(actual: float, expected: float, *, abs_tol: float = 1e-9) -> bool:
    """Strict bit-near comparison.  abs_tol=1e-9 is tight enough that
    only IEEE-754 last-bit ULP differences pass; logical drift fails.
    """
    return math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_p_inflation_op_2pct_hand_derivation():
    """``p_inflation_op[d]`` is the per-period operations factor sum:
    Σ_y (1+r)^-(y+0.5).  Each period contributes 5 compounded terms
    (one per year in years_represented).  At rate=0.02 and offset=0.5
    the expected per-period values are:

        p2020: 1.02^-0.5 + 1.02^-1.5 + 1.02^-2.5 + 1.02^-3.5 + 1.02^-4.5
             ≈ 0.99020 + 0.97078 + 0.95174 + 0.93308 + 0.91478
             ≈ 4.76036
        p2025 (years 5..9):  ≈ 4.31161
        p2030 (years 10..14):≈ 3.90515
        p2035 (years 15..19):≈ 3.53702

    Both the DB-direct and CSV paths must reproduce these values to
    abs_tol = 1e-9.  The expected numbers are derived live (Python
    1.02^-x = IEEE-754 oracle) so the test is bit-near identical to
    the helper's arithmetic — any drift is real, not a typo.
    """
    reader = SpineDbReader(SQLITE, SCENARIO)
    data = load_flextool(WORK, db_reader=reader)
    p_infl = data.p_inflation_op
    assert p_infl is not None, "p_inflation_op missing on DB-direct path"

    for d in PERIODS:
        actual = _val(p_infl, d)
        expected = _ops_factor(d)
        assert _approx(actual, expected), (
            f"p_inflation_op[{d}]: actual={actual!r}, "
            f"expected={expected!r}, diff={actual - expected!r}"
        )


def test_ed_entity_annual_discounted_2pct_hand_derivation():
    """``ed_entity_annual_discounted[e, d]`` is the bare annuity (per
    ``_annuity()``) multiplied by an unbounded inv_factor sum across
    period_in_use for ``reinvest_automatic`` lifetime_method.

    For (wind_plant, p2020) the expected value is::

        annuity_wind = 1000 × 1000 × 0.04 / (1 - (1/1.04)^5)
                     = 224627.11349303348

        sum(inv_factor[d_all] for d_all in PERIODS if pdy[d_all] >= 0)
            = inv[p2020] + inv[p2025] + inv[p2030] + inv[p2035]
            ≈ 4.807729 + 4.354508 + 3.944012 + 3.572213
            ≈ 16.678462

        ed[wind_plant, p2020] = 224627.11349 × 16.678462 ≈ 3,746,434.78

    For (battery, p2020) (annuity = 25900.91499 EUR; same
    inv_factor sum because period_in_use is identical):
        ed[battery, p2020] ≈ 25900.91499 × 16.678462 ≈ 431,987.43

    Subsequent periods drop the earlier-period terms from the sum.
    """
    reader = SpineDbReader(SQLITE, SCENARIO)
    data = load_flextool(WORK, db_reader=reader)
    ed = data.ed_entity_annual_discounted
    assert ed is not None, "ed_entity_annual_discounted missing"

    for d in PERIODS:
        # wind_plant
        actual = _val(ed, "wind_plant", d)
        expected = _annual_disc_reinvest_automatic(ANN_WIND, d)
        assert _approx(actual, expected), (
            f"ed[wind_plant, {d}]: actual={actual!r}, "
            f"expected={expected!r}, diff={actual - expected!r}"
        )
        # battery
        actual = _val(ed, "battery", d)
        expected = _annual_disc_reinvest_automatic(ANN_BATT, d)
        assert _approx(actual, expected), (
            f"ed[battery, {d}]: actual={actual!r}, "
            f"expected={expected!r}, diff={actual - expected!r}"
        )


def test_solve_parity_2pct():
    """End-to-end CSV-path solve parity: the LP built from the
    rate=0.02 fixture must reproduce flextool's recorded objective
    within rel < 1e-6.
    """
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"solve not optimal: {sol}"
    assert_obj_within(sol.obj, WORK)


def test_solve_parity_2pct_db_direct():
    """End-to-end DB-direct solve parity for the rate=0.02 fixture.

    Γ.6 close: this previously diverged from the CSV path by ~1.1 %
    because the unitsize cascade in ``p_flow_upper_existing`` /
    ``p_process_existing_count`` / ``p_state_upper`` mishandled
    explicit ``virtual_unitsize=1`` rows (treating them as default-
    broadcast).  Helpers now use ``InputSource.parameter_explicit``
    so explicit-set values pass through; default-broadcast rows are
    suppressed.  See ``audit/handoff_db_direct_battery_inverter_divergence.md``.
    """
    reader = SpineDbReader(SQLITE, SCENARIO)
    data = load_flextool(WORK, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"DB-direct solve not optimal: {sol}"
    assert_obj_within(sol.obj, WORK)
