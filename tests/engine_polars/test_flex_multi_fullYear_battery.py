"""flexpy ``multi_fullYear_battery`` parity — closed by mirroring the
.mod's bind_forward_only + fix_start in-balance start term
(mod:2197-2203) inside flexpy's nodeBalance.  Sign convention had to
flip because flexpy's nodeBalance has state_change as
``(v_state_lag - v_state_now)`` while the .mod uses
``(v_state_now - v_state_lag)``."""
from pathlib import Path
import polars as pl
from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_multi_fullYear_battery"


def _flextool_obj():
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(parq[0])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_multi_fullYear_battery_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
