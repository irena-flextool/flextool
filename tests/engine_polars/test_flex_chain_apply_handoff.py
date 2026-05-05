"""Chain-runner parity for in-memory handoff threading.

Companion to ``test_flex_chain_wind_battery_invest_lifetime_renew_4solve.py``
asserting that the native cascade threads each sub-solve's flexpy
``SolveHandoff`` to the next via the in-memory handoff path — making
the runner a TRUE standalone driver of a multi-solve flextool scenario.

If parity holds at machine precision across all sub-solves of all
covered scenarios, then flextool's per-sub-solve snapshots are
needed only for STRUCTURE (entity sets, methods, profiles, …) and
all multi-solve STATE flows in-memory between flexpy invocations.

Δ.12e — migrated from the legacy ``run_chain(work, use_handoff_overlay=True)``
file-symlink driver to the native ``run_chain_from_db`` cascade.  The
native path always wires the handoff in-memory through flextool's
``preprocessing_solve_time`` consume hook, so the legacy
``use_handoff_overlay`` knob is structurally unnecessary on native
(it's the *only* mode now).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


DATA = Path(__file__).resolve().parent / "data"


# (work_dirname, scenario_name).  The scenario inside each fixture's
# ``tests.sqlite`` happens to be the same string as the work-dir suffix
# for these three; explicit mapping documented for clarity.
SCENARIOS = [
    ("work_wind_battery_invest_lifetime_renew_4solve",
     "wind_battery_invest_lifetime_renew_4solve"),
    ("work_multi_year", "multi_year"),
    ("work_5weeks_invest_fullYear_dispatch_coal_wind",
     "5weeks_invest_fullYear_dispatch_coal_wind"),
]


@pytest.mark.parametrize("scenario,scenario_name", SCENARIOS)
def test_chain_with_apply_handoff(scenario: str, scenario_name: str) -> None:
    work = DATA / scenario
    if not work.exists():
        pytest.skip(f"fixture {work} not present")
    db_path = work / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"DB {db_path} not present")

    sols = run_chain_from_db(db_path, scenario_name=scenario_name)
    assert sols, f"{scenario}: chain produced no sub-solves"

    for sub_solve, step in sols.items():
        assert step.solution.optimal, (
            f"{scenario}/{sub_solve}: flexpy LP not optimal")
        parq = work / "output_raw" / f"v_obj__{sub_solve}.parquet"
        assert parq.exists(), (
            f"{scenario}/{sub_solve}: missing reference {parq.name}")
        ft = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft) / max(1.0, abs(ft))
        assert rel < 1e-6, (
            f"{scenario}/{sub_solve}: flexpy={step.solution.obj}, "
            f"flextool={ft}, rel={rel}"
        )
