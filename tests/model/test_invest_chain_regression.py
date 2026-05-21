"""Invest-chain regression fixture — Pre-work 0b (R1 + R2).

Context
-------
The Phase E-h post-mortem (``specs/phase_e_h_post_mortem.md``) traced the
Rivendell ``B0_base_hourly_rp`` regression to a fixture-coverage gap:
**no fixture combined invest-chain + non-trivial
``p_entity_all_existing`` + end-to-end LP solve + value assertion** under
the (then-new) ``csv_emission_disabled`` gate.  19 commits stayed green
before a user surfaced the failure from the CLI.

The Step-2.5 / 2.6 FlexDataProvider migration consumed the
``csv_emission_disabled`` distinction (every cascade run now goes
through the in-memory Provider; there is no "gated" vs "default" mode
to switch).  But the underlying coverage requirement — assert
**invest-chain frame state** end-to-end at high precision against a
golden, not just objective parity at ``rel_tol=1e-6`` — still applies.

This module provides:

* **R1** — A native-cascade end-to-end run of
  ``5weeks_invest_fullYear_dispatch_coal_wind`` (the invest-chain
  fixture that the post-mortem flagged as canonical) asserted against
  ``golden_obj.json`` at ``rel_tol=1e-9``.  The fixture's two
  sub-solves (``invest_1year_5weeks`` then ``y2020_fullYear_dispatch``)
  exercise the lifetime / invest cumulative path that's
  structurally absent from ``work_base`` and ``work_fullYear_roll``.

* **R2** — Frame-level equality assertion on
  ``provider.get("solve_data/p_entity_all_existing")`` against a checked-in
  expected snapshot.  This catches the kind of single-row shift that
  feasibly produces a different LP but masks behind objective parity
  at ``rel_tol=1e-6``.  Per the post-mortem, the original Rivendell
  bug shifted a single ``(entity, period)`` row 500× and still gave a
  feasible LP for most other scenarios.

The companion ``tests/test_scenarios.py::test_scenario`` already runs
golden-CSV parity at ``rel_tol=1e-4``; this test tightens the gate at
the in-memory frame layer, which is where the original regression
lived.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl
import pytest

from polar_high import Problem

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from flextool.engine_polars import (  # noqa: E402
    build_flextool,
    run_chain_from_db,
)
from flextool.engine_polars.scaling import (  # noqa: E402
    USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN,
    recommend_user_bound_scale_from_lp,
)


SCENARIO = "5weeks_invest_fullYear_dispatch_coal_wind"

# Sub-solves the fixture's solve chain produces, in cascade order.
SUB_SOLVES = ("invest_1year_5weeks", "y2020_fullYear_dispatch")

# The post-mortem's R1 target: ``rel_tol=1e-9`` against v3.32.0 golden.
# The fixture's ``golden_obj.json`` carries the captured objective; we
# tighten from the file's recorded ``rel_tolerance=1e-6`` to ``1e-9``
# because the native cascade matches the golden to ~1e-15 today (see
# the file docstring's ``probe`` block in commit history).
OBJ_REL_TOL = 1e-9

_FIXTURE_DIR = (
    _TESTS_DIR
    / "engine_polars"
    / "data"
    / "work_5weeks_invest_fullYear_dispatch_coal_wind"
)

# R2 expected frame.  Hand-derived from the fixture spec:
#   - ``coal_market`` / ``west`` are markets / nodes — ``existing=0``.
#   - ``coal_plant`` ships with ``existing=500`` MW at ``p2020``.
#   - ``wind_plant`` ships with ``existing=1000`` MW at ``p2020``.
#
# Both sub-solves see the same frame because the invest chain in this
# fixture is single-period (invest in y2020, no chained carry-over).
# A future ``work_invest_chain_minimal`` fixture (post-mortem fallback
# option) would extend this to 2 periods with non-trivial ``p2030``
# carry-over.
_EXPECTED_PAE_ROWS = [
    {"entity": "coal_market", "period": "p2020", "value": "0.0"},
    {"entity": "west",        "period": "p2020", "value": "0.0"},
    {"entity": "coal_plant",  "period": "p2020", "value": "500.0"},
    {"entity": "wind_plant",  "period": "p2020", "value": "1000.0"},
]


pytestmark = pytest.mark.solver


@pytest.fixture(scope="module")
def invest_chain_steps(test_db_url, tmp_path_factory):  # noqa: ANN001
    """Run the invest-chain cascade ONCE per module and yield the steps.

    Both R1 and R2 hit the same cascade; sharing the run keeps this
    test file's per-CI cost at one ``run_chain_from_db`` invocation.
    Uses ``keep_solutions=True`` so each step retains its Provider for
    frame-equality checks (R2).
    """
    work = tmp_path_factory.mktemp("invest_chain_regression_work")
    steps = run_chain_from_db(
        test_db_url,
        SCENARIO,
        work_folder=work,
        keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for {SCENARIO!r}"
    return steps


def _golden_obj() -> float:
    with (_FIXTURE_DIR / "golden_obj.json").open() as fh:
        return float(json.load(fh)["obj"])


def test_invest_chain_objective_matches_golden(invest_chain_steps):
    """R1 — last-sub-solve objective matches v3.32.0 golden at 1e-9.

    The cascade's terminal sub-solve produces the full-horizon dispatch
    objective; the invest sub-solve's objective is a different number
    (the investment + first-week dispatch) and is covered by the
    per-sub-solve parquet test in
    ``test_flex_5weeks_invest_fullYear_dispatch_coal_wind_per_sub_solve.py``.
    """
    last = next(reversed(invest_chain_steps.values()))
    assert last.solution is not None and last.solution.optimal, (
        f"Last sub-solve of {SCENARIO!r} did not produce an optimal LP"
    )
    golden = _golden_obj()
    # ``step.obj`` is the un-scaled (model-cost) objective populated by
    # ``OrchestrationStep`` — ``step.solution.obj`` is the raw LP
    # solver value, which may be scaled by the ``scale_the_objective``
    # heuristic and is several orders of magnitude smaller.  Match
    # against the un-scaled value, which is what the golden was
    # captured from.
    assert last.obj is not None, (
        f"Last sub-solve of {SCENARIO!r} has no unscaled objective"
    )
    obj = float(last.obj)
    rel = abs(obj - golden) / max(1.0, abs(golden))
    assert rel < OBJ_REL_TOL, (
        f"objective parity broken on {SCENARIO!r}: "
        f"native={obj!r} golden={golden!r} rel={rel!r} tol={OBJ_REL_TOL}"
    )


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_invest_chain_pae_frame_equality(invest_chain_steps, sub_solve):
    """R2 — Provider's ``p_entity_all_existing`` matches expected exactly.

    Frame-level equality (not just objective parity) catches single-row
    shifts in the invest chain that the Phase E-h post-mortem
    identified as the most insidious regression class — they still
    produce a feasible LP and pass loose objective gates.
    """
    step = invest_chain_steps[sub_solve]
    provider = step.flex_data_provider
    assert provider is not None, (
        f"sub-solve {sub_solve!r}: flex_data_provider unexpectedly None "
        f"(keep_solutions=True should retain it)"
    )
    frame = provider.get("solve_data/p_entity_all_existing")
    assert frame is not None, (
        f"sub-solve {sub_solve!r}: provider has no "
        f"'solve_data/p_entity_all_existing' key — invest-chain cascade did not "
        f"populate it"
    )

    expected = pl.DataFrame(
        _EXPECTED_PAE_ROWS,
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )

    # Sort both sides on (entity, period) to make the comparison
    # insertion-order-independent — the Provider's writer ordering is
    # an internal detail; what matters is the *set* of (entity,
    # period, value) tuples.
    actual_sorted = frame.sort(["entity", "period"])
    expected_sorted = expected.sort(["entity", "period"])

    assert actual_sorted.equals(expected_sorted), (
        f"sub-solve {sub_solve!r}: p_entity_all_existing frame drift\n"
        f"--- actual ---\n{actual_sorted}\n"
        f"--- expected ---\n{expected_sorted}"
    )


# ---------------------------------------------------------------------------
# R7 — LP-bound-range smoke
# ---------------------------------------------------------------------------
#
# Phase E-h's bug surfaced as a -10 → -19 shift in ``user_bound_scale``;
# the underlying cause was that the recommendation heuristic anchored to
# ``abs_max`` and collapsed the bottom end of the bound range below
# HiGHS' practical precision (~1e-8).  Current heuristic uses
# geometric-midpoint centering with a clamp at
# ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX] = [-10, 0]`` — see
# ``flextool/engine_polars/scaling.py::recommend_user_bound_scale_from_lp``.
#
# This smoke asserts:
#
# 1. The post-build LP bound spread (decades of |bound|) stays under a
#    generous ceiling.  A real regression that pushed Rivendell into
#    presolve-infeasible territory would manifest here as a sudden jump
#    from ~3 decades to ~10+ decades.
# 2. The recommended ``user_bound_scale`` lands in the legitimate
#    post-clamp range ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.
#    The pre-fix bug recommended ``-19`` (5 decades past the floor);
#    this assertion pins that failure mode at the LP-formulation layer
#    rather than waiting for HiGHS to declare the model infeasible.
#
# The post-mortem text suggests requiring ``[-12, -8]`` for invest
# fixtures, but that range presumes a Rivendell-sized LP whose
# bound-spread exceeds the 6-decade threshold the recommender treats
# as "no scaling needed".  This test's fixture has a 3.4-decade
# spread, so the recommender legitimately returns 0.  The relevant
# invariant is the clamp range, not a particular non-zero value;
# Rivendell-shape coverage is deferred to R6 (see post-mortem).

# Generous ceiling: keep some headroom over the fixture's observed
# 3.4-decade spread.  10 decades would still be well below the
# pathological Rivendell case (the bug shifted the spread to 13+
# decades).  Re-run this test locally before tightening.
_LP_BOUND_SPREAD_DECADES_MAX = 10.0


def _finite_positive(rng) -> list[float]:
    """Return finite, strictly-positive endpoints of a ``(lo, hi)``."""
    if rng is None:
        return []
    try:
        lo, hi = rng
    except (TypeError, ValueError):
        return []
    out: list[float] = []
    for x in (lo, hi):
        if x is None:
            continue
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and xf > 0.0:
            out.append(xf)
    return out


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_invest_chain_lp_bound_range_smoke(invest_chain_steps, sub_solve):
    """R7 — LP bound range fits in <= 10 decades and recommended
    ``user_bound_scale`` is within the post-clamp range.

    Rebuilds the LP in a fresh ``polar_high.Problem`` from the
    step's ``flex_data`` so we can call ``peek_lp_ranges`` post-build
    (the recommendation API the cascade itself uses at
    ``_orchestration.py:873``).
    """
    step = invest_chain_steps[sub_solve]
    assert step.flex_data is not None, (
        f"sub-solve {sub_solve!r}: flex_data unexpectedly None "
        f"(keep_solutions=True should retain it)"
    )

    pb = Problem()
    build_flextool(pb, step.flex_data)
    lp_ranges = pb.peek_lp_ranges()

    positive_bounds: list[float] = []
    for key in ("row_bound", "col_bound"):
        positive_bounds.extend(_finite_positive(lp_ranges.get(key)))
    assert positive_bounds, (
        f"sub-solve {sub_solve!r}: peek_lp_ranges returned no finite "
        f"positive bounds — LP appears unbounded or malformed: "
        f"{lp_ranges!r}"
    )

    spread = math.log10(max(positive_bounds)) - math.log10(min(positive_bounds))
    assert spread <= _LP_BOUND_SPREAD_DECADES_MAX, (
        f"sub-solve {sub_solve!r}: LP bound spread blew up to "
        f"{spread:.2f} decades (ceiling {_LP_BOUND_SPREAD_DECADES_MAX}). "
        f"Phase E-h's pre-fix Rivendell case had spread ~13 decades and "
        f"shifted user_bound_scale to -19 → presolve-infeasible. "
        f"row_bound={lp_ranges.get('row_bound')!r} "
        f"col_bound={lp_ranges.get('col_bound')!r}"
    )

    scale = recommend_user_bound_scale_from_lp(lp_ranges)
    assert USER_BOUND_SCALE_MIN <= scale <= USER_BOUND_SCALE_MAX, (
        f"sub-solve {sub_solve!r}: recommend_user_bound_scale_from_lp "
        f"returned {scale}, outside the post-clamp range "
        f"[{USER_BOUND_SCALE_MIN}, {USER_BOUND_SCALE_MAX}]. "
        f"The geometric-midpoint heuristic in scaling.py is supposed "
        f"to enforce this clamp — a breach here means either the "
        f"clamp regressed or the recommender bypassed it."
    )


# ---------------------------------------------------------------------------
# R7 — also smoke ``work_base`` (the trivial-shape anchor)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_steps(test_db_url, tmp_path_factory):  # noqa: ANN001
    """Run the ``base`` scenario once per module for the work_base smoke."""
    work = tmp_path_factory.mktemp("invest_chain_regression_base_work")
    steps = run_chain_from_db(
        test_db_url,
        "base",
        work_folder=work,
        keep_solutions=True,
    )
    assert steps, "run_chain_from_db returned no steps for 'base'"
    return steps


def test_work_base_lp_bound_range_smoke(base_steps):
    """R7 (work_base anchor) — LP bound range stays sane on the
    trivial-shape canary fixture.

    The post-mortem flags ``work_base`` as the universal default-mode
    canary; the LP-bound smoke applies here too so any future
    seed-funnel-style regression that distorts even the simplest LP
    surfaces at the formulation layer instead of HiGHS-presolve.
    """
    last = next(reversed(base_steps.values()))
    assert last.flex_data is not None, (
        "work_base last step has flex_data=None (keep_solutions=True "
        "should retain it)"
    )
    pb = Problem()
    build_flextool(pb, last.flex_data)
    lp_ranges = pb.peek_lp_ranges()

    positive_bounds: list[float] = []
    for key in ("row_bound", "col_bound"):
        positive_bounds.extend(_finite_positive(lp_ranges.get(key)))
    assert positive_bounds, (
        f"work_base: peek_lp_ranges returned no finite positive "
        f"bounds: {lp_ranges!r}"
    )
    spread = math.log10(max(positive_bounds)) - math.log10(min(positive_bounds))
    assert spread <= _LP_BOUND_SPREAD_DECADES_MAX, (
        f"work_base LP bound spread blew up to {spread:.2f} decades "
        f"(ceiling {_LP_BOUND_SPREAD_DECADES_MAX}). "
        f"row_bound={lp_ranges.get('row_bound')!r} "
        f"col_bound={lp_ranges.get('col_bound')!r}"
    )

    scale = recommend_user_bound_scale_from_lp(lp_ranges)
    assert USER_BOUND_SCALE_MIN <= scale <= USER_BOUND_SCALE_MAX, (
        f"work_base: recommend_user_bound_scale_from_lp returned "
        f"{scale}, outside the post-clamp range "
        f"[{USER_BOUND_SCALE_MIN}, {USER_BOUND_SCALE_MAX}]"
    )
