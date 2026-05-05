"""Γ.8.E — override-chain parity for the CSV-retirement Step 3 surface.

Validates that ``load_flextool(workdir, db_reader=SpineDbReader(...))``
produces a ``FlexData`` whose LP solves to the same objective as the
default CSV path.  This is the gating contract for CSV retirement
Step 3 — when ``load_flextool`` auto-constructs a SpineDbReader, the
DB-direct apply chain (``apply_direct_params → apply_projection_params
→ apply_derived_a..g``) must populate every load-bearing field
correctly so the LP doesn't regress.

The Γ.6.D handoff (``audit/handoff_csv_retirement.md``) and the
Γ.8.E dispatch identified six fixtures that previously regressed:

1. ``work_base_weighted`` — was 6% obj gap (p_rp_cost_weight returned
   trivial 1.0 instead of reading ``timeset.timeset_weights``).
2. ``work_fullYear_roll`` — was 80% obj gap (p_inflation_op returned
   1.0 instead of reading ``solve.years_represented`` for rolling
   solves which key by parent solve name).
3. ``test_cost_aggregation_semantics::TestRpCostWeightFactor::
   test_rp_cost_weight_factor_is_non_trivial`` — same as 1.
4-6. ``work_multi_fullYear_battery_nested_*`` — investigated and
     verified clean already.

Γ.8.E commits:
* ``ce180421`` p_rp_cost_weight wires to timeset_weights.
* ``151109e6`` p_inflation_op handles multi-year-per-period.

This test loops the regression fixtures and asserts CSV-path obj ==
DB-path obj at ``rel < 1e-6``.  Adding a fixture here is a one-line
extension to ``REGRESSION_FIXTURES``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import (
    SpineDbReader,
    build_flextool,
    load_flextool,
)


DATA = Path(__file__).resolve().parent / "data"


# (work_dirname, scenario_name)
REGRESSION_FIXTURES: list[tuple[str, str]] = [
    ("work_base", "base"),
    ("work_base_weighted", "base_weighted"),
    ("work_fullYear_roll", "fullYear_roll"),
    ("work_multi_year", "multi_year"),
    ("work_multi_year_one_solve", "multi_year_one_solve"),
    ("work_multi_fullYear_battery_nested_24h_invest_one_solve",
     "multi_fullYear_battery_nested_24h_invest_one_solve"),
    ("work_multi_fullYear_battery_nested_multi_invest",
     "multi_fullYear_battery_nested_multi_invest"),
    ("work_multi_fullYear_battery_nested_sample_invest_one_solve",
     "multi_fullYear_battery_nested_sample_invest_one_solve"),
    ("work_test_a_lot", "test_a_lot"),
]


@pytest.mark.parametrize(
    "work_name,scenario",
    REGRESSION_FIXTURES,
    ids=[f"{w}::{s}" for w, s in REGRESSION_FIXTURES],
)
def test_override_chain_db_reader_solve_parity(work_name: str,
                                                  scenario: str) -> None:
    """``load_flextool(workdir, db_reader=...)`` solves to the same obj
    as the default CSV path.  Regression guard for the override-chain
    helpers used by CSV-retirement Step 3.
    """
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip(f"{work_name}: no tests.sqlite")

    # Path A — the default CSV loader.
    data_csv = load_flextool(work)
    pb_csv = Problem()
    build_flextool(pb_csv, data_csv)
    sol_csv = pb_csv.solve()
    assert sol_csv.optimal, f"{work_name}: CSV-path LP not optimal"

    # Path B — load via DB-direct override (Step 3 surface).
    reader = SpineDbReader(f"sqlite:///{sqlite}", scenario)
    data_db = load_flextool(work, db_reader=reader)
    pb_db = Problem()
    build_flextool(pb_db, data_db)
    sol_db = pb_db.solve()
    assert sol_db.optimal, f"{work_name}: DB-path LP not optimal"

    rel = abs(sol_csv.obj - sol_db.obj) / max(1.0, abs(sol_csv.obj))
    assert rel < 1e-6, (
        f"{work_name}: CSV={sol_csv.obj}, DB={sol_db.obj}, "
        f"rel={rel} — override-chain regression"
    )


def test_p_rp_cost_weight_non_trivial_on_base_weighted() -> None:
    """Direct unit test on the helper: ``p_rp_cost_weight_from_source``
    must produce non-uniform weights for ``work_base_weighted``.

    Companion to ``test_cost_aggregation_semantics::
    TestRpCostWeightFactor::test_rp_cost_weight_factor_is_non_trivial``.
    Without the Γ.8.E wire-up the helper returns the trivial 1.0
    default and the LP reproduces only by coincidence.
    """
    from flextool.engine_polars._derived_params import (
        p_rp_cost_weight_from_source,
    )
    work = DATA / "work_base_weighted"
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("work_base_weighted not present")
    reader = SpineDbReader(f"sqlite:///{sqlite}", "base_weighted")
    data = load_flextool(work)
    rp = p_rp_cost_weight_from_source(reader, data.dt, "y2020_2day_dispatch")
    assert rp is not None
    uniques = sorted(rp.frame["value"].unique().to_list())
    # Fixture has 4 distinct weight blocks (0.4, 0.8, 1.2, 1.6).
    assert len(uniques) > 1, (
        f"helper produced trivial weights {uniques}; "
        f"timeset_weights wire-up regression"
    )


def test_p_inflation_op_uses_years_represented() -> None:
    """Direct unit test on the helper: ``p_inflation_op_from_source``
    must read ``solve.years_represented`` when ``inflation_rate=0``,
    not return the trivial 1.0.  Specifically: rolling-solve fixtures
    key the DB parameter on the parent solve name; the helper must
    fall back via stripping ``_roll_<N>``.
    """
    from flextool.engine_polars._derived_params import (
        p_inflation_op_from_source,
    )
    work = DATA / "work_fullYear_roll"
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("work_fullYear_roll not present")
    reader = SpineDbReader(f"sqlite:///{sqlite}", "fullYear_roll")
    data = load_flextool(work)
    # The active solve in the workdir is the per-roll name.
    infl = p_inflation_op_from_source(
        reader, data.dt, "dispatch_fullYear_roll_roll_71",
    )
    assert infl is not None
    # fullYear_roll uses 5 years per period.
    uniques = sorted(infl.frame["value"].unique().to_list())
    assert uniques == [5.0], (
        f"helper produced {uniques}; expected [5.0] via "
        f"years_represented parent-solve fallback"
    )
