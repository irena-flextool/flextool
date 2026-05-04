"""flexpy parity for delay-path with non-default source flow coef.

Derivative of ``water_pump_delayed`` with
``p_process_source_flow_coefficient(water_pump, water_source) = 2.0``.
This exercises the .mod's source-coefficient multiplier on delayed
source flows (flextool.mod:2573) — a code path that no other fixture
combines with delays.  Without the matching multiplier in
``flextool/_delay.py::delayed_input_expr``, the LP solutions diverge.
"""
from pathlib import Path
import polars as pl
from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool

WORK = Path(__file__).resolve().parent / "data" / "work_delay_source_coef"


def test_delay_source_coef_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
