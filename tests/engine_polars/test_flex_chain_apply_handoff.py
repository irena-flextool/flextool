"""Chain-runner parity with ``use_handoff_overlay=True``.

Companion to ``test_flex_chain_wind_battery_invest_lifetime_renew_4solve.py``
which validates the chain runner's default mode (snapshot CSVs as the
source of truth for handoff state).  This module flips
``use_handoff_overlay=True`` so the chain runner passes each
sub-solve's prior flexpy ``SolveHandoff`` to ``load_flextool(...,
handoff=)`` — Δ.11 construct-with-handoff path — making the runner a
TRUE standalone driver of a multi-solve flextool scenario.

If parity holds at machine precision across all sub-solves of all
covered scenarios, then flextool's per-sub-solve snapshots are
needed only for STRUCTURE (entity sets, methods, profiles, …) and
all multi-solve STATE flows in-memory between flexpy invocations.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import run_chain


DATA = Path(__file__).resolve().parent / "data"


SCENARIOS = [
    "work_wind_battery_invest_lifetime_renew_4solve",
    "work_multi_year",
    "work_5weeks_invest_fullYear_dispatch_coal_wind",
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_chain_with_apply_handoff(scenario: str) -> None:
    work = DATA / scenario
    if not work.exists():
        pytest.skip(f"fixture {work} not present")

    sols = run_chain(work, use_handoff_overlay=True)
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
