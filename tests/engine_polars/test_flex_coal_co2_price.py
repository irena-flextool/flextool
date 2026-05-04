"""Stage-3+: ``coal_co2_price`` scenario — adds a CO2 price on the
gas commodity.  Tests that the optional CO2 term in flex_coal_model
is wired correctly: parity vs flextool's recorded v_obj."""

from pathlib import Path

import polars as pl

from flexpy import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool


WORK = Path(__file__).resolve().parent / "data" / "work_coal_co2_price"


def test_coal_co2_price_parity():
    data = load_flextool(WORK)
    assert data.flow_from_co2_priced is not None, \
        "fixture should carry CO2 price data"

    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()

    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
