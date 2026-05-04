"""flexpy ``network_wind_coal_battery_fullYear_invest`` parity —
multi-node fixture where flextool's preprocessing emits a loose
``p_flow_max`` (existing + invest_max_total) even on entities whose
``ed_invest`` is empty.  Closed by routing maxToSink RHS through
``p_flow_upper_existing`` for direct processes, which matches the
.mod's actual RHS expansion (existing-only when no v_invest exists).
"""
from pathlib import Path
import polars as pl
from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_network_wind_coal_battery_fullYear_invest"


def _flextool_obj():
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(parq[0])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_network_wind_coal_battery_fullYear_invest_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
