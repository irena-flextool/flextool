"""``coal_co2_limit`` scenario — adds a per-period CO2 cap that
binds, forcing additional slack.  Tests the optional ``co2_max_period``
constraint in flex_coal_model."""

import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "coal_co2_limit"

def test_coal_co2_limit_parity(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.flow_from_co2_capped is not None, \
        "fixture should carry a CO2 cap"
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
