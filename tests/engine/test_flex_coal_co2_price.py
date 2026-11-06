"""Stage-3+: ``coal_co2_price`` scenario — adds a CO2 price on the
gas commodity.  Tests that the optional CO2 term in flex_coal_model
is wired correctly: parity vs flextool's recorded v_obj."""

import polars as pl

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "coal_co2_price"

def test_coal_co2_price_parity(scenario_run):
    run = scenario_run(SCENARIO)
    work = run.work_folder
    data = load_scenario(run)
    assert data.flow_from_co2_priced is not None, \
        "fixture should carry CO2 price data"

    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()

    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
