"""flexpy ``coal_ramp_limit`` scenario — sink-side ramp_speed_up
limits how fast v_flow can change between consecutive timesteps.
Parity vs flextool."""

from pathlib import Path

import pytest
import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_coal_ramp_limit"


def test_coal_ramp_limit_parity():
    d = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
