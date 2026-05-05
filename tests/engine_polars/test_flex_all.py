"""flexpy 'all' scenario parity — kitchen-sink integration test.

Closed by fixing a cross-join bug in ``build_flextool``'s maxToSink RHS
construction: when broadcasting ``p_flow_upper_existing`` (no ``t`` dim)
over the timeline, a cross-join with ``d.dt`` produced duplicate rows
whenever a ``t`` label was reused across periods (e.g. ``t0001`` in
both ``p2020`` and ``p2025``), inflating the RHS by the duplication
factor.  Replaced with an inner-join on ``d``.
"""
from pathlib import Path
import polars as pl
from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_all"


def _flextool_obj():
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        # If multi-solve, prefer the one matching solve_data's solve_current.
        return pl.read_parquet(sorted(parq)[-1])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_all_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
