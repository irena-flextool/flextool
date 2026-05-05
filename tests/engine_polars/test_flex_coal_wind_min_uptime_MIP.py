"""flexpy ``coal_wind_min_uptime_MIP`` parity (integer UC + min uptime)."""

from pathlib import Path

import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_coal_wind_min_uptime_MIP"


def test_coal_wind_min_uptime_MIP_parity():
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    obj_path = next(WORK.glob("output_raw/v_obj__*.parquet"))
    flextool_obj = pl.read_parquet(obj_path)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
