"""polar_high ``coal_min_load`` scenarios — unit commitment with linear
v_online, min_load, startup cost.  Parity vs flextool."""

from pathlib import Path

import pytest
import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool

pytestmark = pytest.mark.solver


def _parity(work: Path) -> None:
    d = load_flextool(work)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6


def test_coal_min_load_parity(scenario_workdir):
    _parity(scenario_workdir("coal_min_load"))


def test_coal_min_load_wind_parity(scenario_workdir):
    _parity(scenario_workdir("coal_min_load_wind"))


def test_coal_min_load_MIP_wind_parity(scenario_workdir):
    """Integer v_online — exercises engine MIP support."""
    _parity(scenario_workdir("coal_min_load_MIP_wind"))
