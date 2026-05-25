"""``wind`` scenario — single VRE process with profile_flow_upper_limit.
First parity test for the ``profile_flow_upper`` feature."""

import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "wind"


def test_wind_parity(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.process_profile_upper is not None and data.process_profile_upper.height > 0
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
