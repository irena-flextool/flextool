"""Chain-runner parity for in-memory handoff threading.

Companion to ``test_flex_chain_wind_battery_invest_lifetime_renew_4solve.py``
asserting that the native cascade threads each sub-solve's polar_high
``SolveHandoff`` to the next via the in-memory handoff path — making
the runner a TRUE standalone driver of a multi-solve flextool scenario.

If parity holds at machine precision across all sub-solves of all
covered scenarios, then flextool's per-sub-solve snapshots are
needed only for STRUCTURE (entity sets, methods, profiles, …) and
all multi-solve STATE flows in-memory between polar_high invocations.

Δ.12e — migrated from the legacy ``run_chain(work, use_handoff_overlay=True)``
file-symlink driver to the native ``run_chain_from_db`` cascade.  The
native path always wires the handoff in-memory through flextool's
``preprocessing_solve_time`` consume hook, so the legacy
``use_handoff_overlay`` knob is structurally unnecessary on native
(it's the *only* mode now).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


SCENARIOS = [
    "wind_battery_invest_lifetime_renew_4solve",
    "multi_year",
    "5weeks_invest_fullYear_dispatch_coal_wind",
]


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_chain_with_apply_handoff(
    scenario_name: str, scenario_workdir
) -> None:
    work = scenario_workdir(scenario_name)
    db_path = work / "tests.sqlite"

    sols = run_chain_from_db(
        db_path, scenario_name=scenario_name, keep_solutions=True,
    )
    assert sols, f"{scenario_name}: chain produced no sub-solves"

    for sub_solve, step in sols.items():
        assert step.solution.optimal, (
            f"{scenario_name}/{sub_solve}: polar_high LP not optimal")
        parq = work / "output_raw" / f"v_obj__{sub_solve}.parquet"
        assert parq.exists(), (
            f"{scenario_name}/{sub_solve}: missing reference {parq.name}")
        ft = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft) / max(1.0, abs(ft))
        assert rel < 1e-6, (
            f"{scenario_name}/{sub_solve}: polar_high={step.solution.obj}, "
            f"flextool={ft}, rel={rel}"
        )
