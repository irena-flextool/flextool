"""``wind_battery`` scenario — wind plant + battery storage with
bind_within_timeblock cyclic state and fix_start.  First parity test
for storage features (v_state, state-change in nodeBalance, source-
side flows in nodeBalance, maxState, storage_state_start_binding,
self-discharge)."""

import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "wind_battery"

def test_wind_battery_parity(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.nodeState is not None and data.nodeState.height > 0, \
        "fixture should carry a storage node"
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
