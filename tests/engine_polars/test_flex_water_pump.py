"""flexpy water_pump parity."""
from pathlib import Path
import polars as pl
from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_water_pump"


def test_water_pump_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
