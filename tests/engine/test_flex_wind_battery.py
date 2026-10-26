"""``wind_battery`` scenario — wind plant + battery storage with
bind_within_timeblock cyclic state and fix_start.  First parity test
for storage features (v_state, state-change in nodeBalance, source-
side flows in nodeBalance, maxState, storage_state_start_binding,
self-discharge)."""

import polars as pl

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario
import pytest

pytestmark = pytest.mark.solver


SCENARIO = "wind_battery"

def test_wind_battery_parity(scenario_run):
    run = scenario_run(SCENARIO)
    work = run.work_folder
    data = load_scenario(run)
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
