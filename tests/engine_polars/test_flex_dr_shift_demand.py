"""flexpy dr_shift_demand parity — closed once bind_within_solve was
wired into the nodeBalance state-change term in model.py.  The
fixture's ``dr_storage`` node uses bind_within_solve (cyclic via
``t_previous_within_solve``); without that mapping flexpy was
treating dr_storage as a regular flow-only node, leaving the load-
shift unconstrained and producing an obj 25% too high (slack-priced
demand violations replaced what should have been zero-cost shifts).
"""
from pathlib import Path
import polars as pl
from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_dr_shift_demand"


def test_dr_shift_demand_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
