"""End-to-end chain runner parity for ``wind_battery_invest_lifetime_renew_4solve``.

Whereas ``test_flex_wind_battery_invest_lifetime_renew_4solve.py`` loads each
per-sub-solve snapshot independently, this test exercises the native
cascade (``run_chain_from_db``) which iterates the chain in order and
threads an in-memory handoff between sub-solves.

The parity assertion is the same — flexpy's objective should match
flextool's parquet at every sub-solve to machine precision — but the
driver is now the chain runner rather than ad-hoc per-sub-solve
fixture wiring.

Δ.12e — migrated from the legacy file-symlink ``run_chain(work)`` driver
to the native ``run_chain_from_db(db, scenario)`` entry point.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars import run_chain_from_db
import pytest

pytestmark = pytest.mark.solver


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_wind_battery_invest_lifetime_renew_4solve"
)


def test_chain_runs_end_to_end() -> None:
    if not WORK.exists():
        pytest.skip(f"fixture {WORK} not present")
    db_path = WORK / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"DB {db_path} not present")

    sols = run_chain_from_db(
        db_path, scenario_name="wind_battery_invest_lifetime_renew_4solve",
        keep_solutions=True,
    )
    assert list(sols) == [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ]

    for sub_solve, step in sols.items():
        assert step.solution.optimal, (
            f"sub-solve {sub_solve}: flexpy LP not optimal")
        parq = WORK / "output_raw" / f"v_obj__{sub_solve}.parquet"
        ft = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft) / max(1.0, abs(ft))
        assert rel < 1e-6, (
            f"{sub_solve}: flexpy={step.solution.obj}, "
            f"flextool={ft}, rel={rel}"
        )
