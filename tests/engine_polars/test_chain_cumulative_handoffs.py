"""Diagnostic for the chain-cumulative carriers populated by
:func:`flextool.input.build_handoff_from_solution` through ``run_chain``.

Exercises the chain-cumulative carrier SHAPE — state passed solve-to-
solve — on the carriers the 4-solve invest+lifetime-renew fixture
actually populates:

* ``realized_invest`` / ``realized_existing`` — per-(entity, period)
  chain-cumulative invest history (grow by 4 rows per realised period).
* ``roll_end_state`` — per-node end-of-roll storage state (battery).
* ``cum_sim_hours`` — per-period simulated-hour totals (grows by one
  row per realised period).

This fixture has no commodity ladder, no CO2 method, no divest, and no
nested storage fixing, so the remaining ``SolveHandoff`` fields
(``cumulative_commodity``, ``cumulative_co2``, ``divest_cumulative``,
``fix_storage``) stay ``None``.  Explicit ``is None`` assertions on
those serve as a bidirectional regression guard: any future change
that starts populating them — or stops populating the carriers
asserted positive — surfaces here.

Carrier exercised: chain-cumulative invest accumulators on a
4-solve invest+lifetime-renew fixture.
"""
from __future__ import annotations

import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


SCENARIO_NAME = "wind_battery_invest_lifetime_renew_4solve"


def test_chain_cumulative_handoffs_accumulate_monotonically(
    scenario_workdir,
) -> None:
    work = scenario_workdir(SCENARIO_NAME)
    db_path = work / "tests.sqlite"

    sols = run_chain_from_db(db_path, scenario_name=SCENARIO_NAME)
    chain_order = list(sols)
    assert chain_order == [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ], f"chain order changed unexpectedly: {chain_order}"

    prev_keys: set[tuple[str, str]] = set()
    prev_height = 0
    prev_csh_height = 0
    for sub in chain_order:
        h = sols[sub].handoff

        # Populated-by-design carriers.
        assert h.realized_invest is not None, (
            f"{sub}: realized_invest unexpectedly None")
        assert h.realized_existing is not None, (
            f"{sub}: realized_existing unexpectedly None")
        assert h.roll_end_state is not None, (
            f"{sub}: roll_end_state unexpectedly None (battery storage)")
        assert h.cum_sim_hours is not None, (
            f"{sub}: cum_sim_hours unexpectedly None")

        # Unpopulated-by-design carriers (no commodity ladder, no CO2 method,
        # no divest, no nested storage fixing in this fixture).  Bidirectional
        # regression guard: surface any future flip to populated.
        assert h.cumulative_commodity is None, (
            f"{sub}: cumulative_commodity populated unexpectedly "
            f"(fixture has no commodity ladder)")
        assert h.cumulative_co2 is None, (
            f"{sub}: cumulative_co2 populated unexpectedly "
            f"(fixture has no CO2 method)")
        assert h.divest_cumulative is None, (
            f"{sub}: divest_cumulative populated unexpectedly "
            f"(fixture has no divest)")
        assert not hasattr(h, "fix_storage"), (
            f"{sub}: wide fix_storage field unexpectedly resurrected — "
            f"Phase 4.1l retired it; data now flows via the three narrow "
            f"fix_storage_{{quantity,price,usage}} fields")

        ri = h.realized_invest
        re = h.realized_existing

        # Unique (entity, period) keys — no duplicate-row leakage from
        # the chain-cumulation step.
        assert ri.unique(["entity", "period"]).height == ri.height, (
            f"{sub}: realized_invest has duplicate (entity, period) rows")
        assert re.unique(["entity", "period"]).height == re.height, (
            f"{sub}: realized_existing has duplicate (entity, period) rows")

        # Strict-growth invariant: each subsequent solve adds rows.
        assert ri.height > prev_height, (
            f"{sub}: realized_invest height {ri.height} did not exceed "
            f"prior solve's {prev_height} — chain-cumulative state should "
            f"strictly grow as more periods are realised.")

        # Cross-solve carryover: every (entity, period) from the prior
        # solve's handoff persists in this solve's handoff.
        cur_keys = {(r["entity"], r["period"]) for r in ri.iter_rows(named=True)}
        missing = prev_keys - cur_keys
        assert not missing, (
            f"{sub}: prior-solve realized_invest keys missing from current "
            f"handoff: {sorted(missing)[:5]} ... — chain-cumulative carrier "
            f"must persist all earlier contributions.")
        prev_keys = cur_keys
        prev_height = ri.height

        # roll_end_state — battery node only, per-roll v_state at last (d, t).
        rs = h.roll_end_state
        assert "node" in rs.columns and "value" in rs.columns, (
            f"{sub}: roll_end_state missing expected columns; got {rs.columns}")
        assert rs.height >= 1, (
            f"{sub}: roll_end_state empty — battery should produce a row")

        # cum_sim_hours — chain-cumulative per-period; one row added per
        # realised period and non-decreasing per-period values.
        csh = h.cum_sim_hours
        assert csh.height > prev_csh_height, (
            f"{sub}: cum_sim_hours height {csh.height} did not exceed "
            f"prior solve's {prev_csh_height} — one row should be added "
            f"per newly realised period.")
        # Phase 4.1a — canonical column is ``p_ladder_cum_sim_hours``
        # (matches the ``solve_data/`` Provider key schema; was ``value``
        # before the handoff carriers were migrated to canonical names).
        cum_col = "p_ladder_cum_sim_hours"
        assert (csh[cum_col].to_list() == sorted(csh[cum_col].to_list())
                or csh.height == 1), (
            f"{sub}: cum_sim_hours per-period values not monotone")
        prev_csh_height = csh.height
