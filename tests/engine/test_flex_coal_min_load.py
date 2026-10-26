"""polar_high ``coal_min_load`` scenarios — unit commitment with linear
v_online, min_load, startup cost.  Parity vs flextool."""

import pytest
import polars as pl

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import ScenarioRun, load_scenario

pytestmark = pytest.mark.solver


def _parity(run: ScenarioRun) -> None:
    work = run.work_folder
    d = load_scenario(run)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6


def test_coal_min_load_parity(scenario_run):
    _parity(scenario_run("coal_min_load"))


def test_coal_min_load_wind_parity(scenario_run):
    _parity(scenario_run("coal_min_load_wind"))


def test_coal_min_load_MIP_wind_parity(scenario_run):
    """Integer v_online — exercises engine MIP support."""
    _parity(scenario_run("coal_min_load_MIP_wind"))
