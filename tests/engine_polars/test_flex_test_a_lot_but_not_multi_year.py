"""flexpy parity for the ``test_a_lot_but_not_multi_year`` kitchen-sink
fixture.  Closed by adding the availability factor on
``maxToSink_online`` RHS — the .mod's online-process flow bound is
``v_online × max_cap × availability × unitsize`` (mod:3015-3026), but
flexpy was emitting ``v_flow ≤ v_online × 1`` (no availability), which
under-tightens the v_flow bound at hours where availability < 1 and
v_online < 1 simultaneously.  That extra slack lets the LP run the
unit beyond its de-rated peak, allowing it to invest less.  Closing
this on test_a_lot* moved the gap from -1.21% to machine epsilon on
this fixture.  See ``flextool/model.py`` _add_online_block."""
from pathlib import Path
import polars as pl
from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_test_a_lot_but_not_multi_year"


def _flextool_obj() -> float:
    parq = sorted(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(parq[-1])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_test_a_lot_but_not_multi_year_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}")
