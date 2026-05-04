"""``wind`` scenario — single VRE process with profile_flow_upper_limit.
First parity test for the ``profile_flow_upper`` feature."""

from pathlib import Path
import polars as pl

from flexpy import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool


WORK = Path(__file__).resolve().parent / "data" / "work_wind"


def test_wind_parity():
    data = load_flextool(WORK)
    assert data.process_profile_upper is not None and data.process_profile_upper.height > 0
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
