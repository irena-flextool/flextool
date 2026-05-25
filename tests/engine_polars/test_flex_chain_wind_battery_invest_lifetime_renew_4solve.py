"""End-to-end chain runner parity for ``wind_battery_invest_lifetime_renew_4solve``.

Whereas ``test_flex_wind_battery_invest_lifetime_renew_4solve.py`` loads each
per-sub-solve snapshot independently, this test exercises the native
cascade (``run_chain_from_db``) which iterates the chain in order and
threads an in-memory handoff between sub-solves.

The parity assertion is the same — polar_high's objective should match
flextool's parquet at every sub-solve to machine precision — but the
driver is now the chain runner rather than ad-hoc per-sub-solve
fixture wiring.

Δ.12e — migrated from the legacy file-symlink ``run_chain(work)`` driver
to the native ``run_chain_from_db(db, scenario)`` entry point.
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars import run_chain_from_db
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "wind_battery_invest_lifetime_renew_4solve"


def test_chain_runs_end_to_end(scenario_workdir) -> None:
    work = scenario_workdir(SCENARIO)
    db_path = work / "tests.sqlite"

    sols = run_chain_from_db(
        db_path, scenario_name=SCENARIO,
        keep_solutions=True,
    )
    assert list(sols) == [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ]

    for sub_solve, step in sols.items():
        assert step.solution.optimal, (
            f"sub-solve {sub_solve}: polar_high LP not optimal")
        parq = work / "output_raw" / f"v_obj__{sub_solve}.parquet"
        ft = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft) / max(1.0, abs(ft))
        assert rel < 1e-6, (
            f"{sub_solve}: polar_high={step.solution.obj}, "
            f"flextool={ft}, rel={rel}"
        )
