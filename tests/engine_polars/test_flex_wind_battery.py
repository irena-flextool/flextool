"""``wind_battery`` scenario — wind plant + battery storage with
bind_within_timeset cyclic state and fix_start.  First parity test
for storage features (v_state, state-change in nodeBalance, source-
side flows in nodeBalance, maxState, storage_state_start_binding,
self-discharge)."""

from pathlib import Path
import polars as pl

from polar_high_opt import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool


WORK = Path(__file__).resolve().parent / "data" / "work_wind_battery"


def test_wind_battery_parity():
    data = load_flextool(WORK)
    assert data.nodeState is not None and data.nodeState.height > 0, \
        "fixture should carry a storage node"
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
