"""Γ.8.E — override-chain parity for the CSV-retirement Step 3 surface.

Validates that the production loader call (``load_flextool(folder,
provider=..., db_reader=SpineDbReader(...), handoff=...)``) produces a
``FlexData`` whose LP solves to the cascade's own objective.  The
DB-direct apply chain (``apply_direct_params → apply_projection_params
→ apply_derived_a..g``) must populate every load-bearing field
correctly so the LP doesn't regress.

The Γ.6.D handoff (``audit/handoff_csv_retirement.md``) and the
Γ.8.E dispatch identified six fixtures that previously regressed:

1. ``work_base_weighted`` — was 6% obj gap (p_timestep_weight returned
   trivial 1.0 instead of reading ``timeset.timeset_weights``).
2. ``work_fullYear_roll`` — was 80% obj gap (p_inflation_op returned
   1.0 instead of reading ``solve.years_represented`` for rolling
   solves which key by parent solve name).
3. ``test_cost_aggregation_semantics::TestTimestepWeightFactor::
   test_timestep_weight_factor_is_non_trivial`` — same as 1.
4-6. ``work_multi_fullYear_battery_nested_*`` — investigated and
     verified clean already.

Γ.8.E commits:
* ``ce180421`` p_timestep_weight wires to timeset_weights.
* ``151109e6`` p_inflation_op handles multi-year-per-period.

This test loops the regression fixtures and asserts that the LP rebuilt
from the post-solve provider through the production loader call solves
to the cascade's own objective at ``rel 1e-9``.  (It used to compare a
CSV load with a DB-direct load of the same, DB-derived snapshot, which
could no longer fail.)  Adding a fixture here is a one-line extension
to ``REGRESSION_SCENARIOS``.
"""
from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario


REGRESSION_SCENARIOS: list[str] = [
    "base",
    "base_weighted",
    "fullYear_roll",
    "multi_year",
    "multi_year_one_solve",
    "multi_fullYear_battery_nested_24h_invest_one_solve",
    "multi_fullYear_battery_nested_multi_invest",
    "multi_fullYear_battery_nested_sample_invest_one_solve",
    "test_a_lot",
]


@pytest.mark.parametrize("scenario", REGRESSION_SCENARIOS)
def test_override_chain_db_reader_solve_parity(
    scenario: str, scenario_run,
) -> None:
    """The LP rebuilt from the last step's post-solve provider through the
    production loader call (``load_scenario``: provider + DB reader +
    prior handoff) solves to the cascade's own objective (rel 1e-9).
    Regression guard for the override-chain helpers on the fixtures that
    once regressed.
    """
    run = scenario_run(scenario)
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"{scenario}: rebuilt LP not optimal"
    assert run.last_obj is not None, f"{scenario}: the cascade recorded no objective"
    assert sol.obj == pytest.approx(run.last_obj, rel=1e-9, abs=1e-9), (
        f"{scenario}: rebuilt={sol.obj}, cascade={run.last_obj} — override-chain regression"
    )


def test_p_timestep_weight_non_trivial_on_base_weighted(
    scenario_run,
) -> None:
    """Direct unit test on the helper: ``p_timestep_weight_from_source``
    must produce non-uniform weights for ``work_base_weighted``.

    Companion to ``test_cost_aggregation_semantics::
    TestTimestepWeightFactor::test_timestep_weight_factor_is_non_trivial``.
    Without the Γ.8.E wire-up the helper returns the trivial 1.0
    default and the LP reproduces only by coincidence.
    """
    from flextool.engine.preprocess.params._derived import (
        p_timestep_weight_from_source,
    )
    run = scenario_run("base_weighted")
    reader = run.reader()
    data = load_scenario(run)
    rp = p_timestep_weight_from_source(reader, data.dt, "y2020_2day_dispatch")
    assert rp is not None
    uniques = sorted(rp.frame["value"].unique().to_list())
    # Fixture has 4 distinct weight blocks (0.4, 0.8, 1.2, 1.6).
    assert len(uniques) > 1, (
        f"helper produced trivial weights {uniques}; "
        f"timeset_weights wire-up regression"
    )


def test_p_inflation_op_uses_years_represented(scenario_run) -> None:
    """Direct unit test on the helper: ``p_inflation_op_from_source``
    must read ``solve.years_represented`` when ``inflation_rate=0``,
    not return the trivial 1.0.  Specifically: rolling-solve fixtures
    key the DB parameter on the parent solve name; the helper must
    fall back via stripping ``_roll_<N>``.
    """
    from flextool.engine.preprocess.params._derived import (
        p_inflation_op_from_source,
    )
    run = scenario_run("fullYear_roll")
    reader = run.reader()
    data = load_scenario(run)
    # The active solve of the last step is the per-roll name.
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
