"""flexpy multi_year_wind_no_investment parity — multi-period dispatch
with no investment decisions (existing-only)."""
from pathlib import Path
import polars as pl
from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_multi_year_wind_no_investment"


def _flextool_obj():
    sc_file = WORK / "solve_data" / "solve_current.csv"
    if sc_file.exists():
        solve = pl.read_csv(sc_file)["solve"][0]
        parq = WORK / "output_raw" / f"v_obj__{solve}.parquet"
        if parq.exists():
            return pl.read_parquet(parq)["objective"][0]
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(sorted(parq)[-1])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_multi_year_wind_no_investment_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
