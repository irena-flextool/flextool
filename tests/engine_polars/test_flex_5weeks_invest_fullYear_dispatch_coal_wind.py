"""flexpy 5weeks_invest_fullYear_dispatch_coal_wind parity (dispatch solve)."""
from pathlib import Path
import polars as pl
from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_5weeks_invest_fullYear_dispatch_coal_wind"


def test_5weeks_invest_fullYear_dispatch_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    # Γ.4: prefer golden_obj.json if present, fall back to parquet.
    # On this fixture the golden is the canonical answer because the
    # parquet was produced from a buggy CSV (e_invest_total empty
    # despite Spine having unit.invest_max_total > 0; see Γ.3.G stanza).
    from _golden import assert_obj_within
    assert_obj_within(sol.obj, WORK,
                       parquet_glob="v_obj__y2020_fullYear_dispatch.parquet")
