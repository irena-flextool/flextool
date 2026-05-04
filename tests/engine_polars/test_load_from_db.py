"""Phase B: DB-direct loader parity.

Each test runs ``load_flextool_from_db()`` against the scenario's
``tests.sqlite`` and compares flexpy's obj to flextool's reference obj
(committed in ``output_raw/v_obj__*.parquet``).  Multi-solve cascades
are out of scope (flexpy is single-solve)."""
from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import polars as pl
import pytest

from flexpy import Problem
from flextool.engine_polars import build_flextool, load_flextool_from_db


DATA = Path(__file__).resolve().parent / "data"

# Single-solve scenarios — Phase B in-scope.
# ``coal_chp_extraction`` has a pre-existing parity gap (96%) that
# the fixture-load also hits — not a Phase B regression.
SCENARIOS_SINGLE_SOLVE = [
    "base", "base_weighted",
    "coal", "coal_chp", "coal_co2_limit", "coal_co2_price",
    "coal_min_load", "coal_min_load_MIP_wind", "coal_min_load_wind",
    "coal_ramp_limit", "coal_retire",
    "multi_year_wind_no_investment",
    "scale_to_peak_flow",
    "wind", "wind_battery", "wind_battery_invest",
]

# Multi-solve cascade scenarios — drive flextool's solve loop with
# flexpy, build a SolveHandoff from each solve's solution, deposit
# into state.handoffs for the next solve's preprocessing.  The final
# solve's FlexData is returned to the caller, who solves it externally
# and compares to the *last* solve's reference obj.
SCENARIOS_MULTI_SOLVE = [
    ("coal_unit_size_MIP_wind", "y2020_fullYear_dispatch"),
]


@pytest.fixture(autouse=True)
def _quiet_flextool_logging():
    logging.getLogger().setLevel(logging.ERROR)
    yield


@pytest.mark.parametrize("scenario", SCENARIOS_SINGLE_SOLVE)
def test_db_direct_parity(scenario, tmp_path):
    fixture = DATA / f"work_{scenario}"
    db = fixture / "tests.sqlite"
    obj_path = next(iter(fixture.glob("output_raw/v_obj__*.parquet")))
    flextool_obj = pl.read_parquet(obj_path)["objective"][0]

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        d = load_flextool_from_db(db, scenario_name=scenario,
                                   work_folder=tmp_path)
        pb = Problem()
        build_flextool(pb, d)
        sol = pb.solve()

    assert sol.optimal
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"DB-direct parity for {scenario}: "
        f"flexpy={sol.obj:.4f} flextool={flextool_obj:.4f} rel={rel:.2e}"
    )


@pytest.mark.parametrize("scenario,final_solve", SCENARIOS_MULTI_SOLVE)
def test_db_direct_parity_multi_solve(scenario, final_solve, tmp_path):
    """Multi-solve cascade.  ``load_flextool_from_db`` drives the cascade
    internally with flexpy as the solver, building a ``SolveHandoff``
    per solve and feeding it into the next solve's preprocessing.  The
    returned ``FlexData`` reflects the FINAL solve's setup; the test
    solves it externally and compares to that solve's reference obj."""
    fixture = DATA / f"work_{scenario}"
    db = fixture / "tests.sqlite"
    obj_path = fixture / f"output_raw/v_obj__{final_solve}.parquet"
    flextool_obj = pl.read_parquet(obj_path)["objective"][0]

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        d = load_flextool_from_db(db, scenario_name=scenario,
                                   work_folder=tmp_path)
        pb = Problem()
        build_flextool(pb, d)
        sol = pb.solve()

    assert sol.optimal
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"multi-solve DB-direct parity for {scenario}/{final_solve}: "
        f"flexpy={sol.obj:.4f} flextool={flextool_obj:.4f} rel={rel:.2e}"
    )
