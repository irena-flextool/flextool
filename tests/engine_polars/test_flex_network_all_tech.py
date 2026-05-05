"""flexpy network_all_tech parity — multi-feature kitchen-sink (DR +
water_pump-style storage + fusion + delays + reserves + cumulative
invest).  Closed by the merge step 6+7 fixes (bind_forward_only /
bind_within_solve in nodeBalance, p_flow_upper_existing for direct
processes, availability factor on maxToSink RHS,
bind_forward_only+fix_start in-balance term)."""
from pathlib import Path
import polars as pl
from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_network_all_tech"


def _flextool_obj():
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(parq[0])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_network_all_tech_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
