"""flexpy ``wind_battery_invest_lifetime_renew`` parity."""

from pathlib import Path

import polars as pl

from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool


WORK = Path(__file__).resolve().parent / "data" / "work_wind_battery_invest_lifetime_renew"


def test_wind_battery_invest_lifetime_renew_parity():
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    obj_path = next(WORK.glob("output_raw/v_obj__*.parquet"))
    flextool_obj = pl.read_parquet(obj_path)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
