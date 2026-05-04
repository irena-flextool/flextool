"""flexpy parity for the three nested rolling-horizon battery
fixtures.  Each fixture's ``solve_data/`` reflects the LAST sub-solve
of a 72-step rolling-horizon dispatch chain
(``dispatch_fullYear_roll_roll_71``).  The .mod features needed to
reproduce that LP single-shot in flexpy are:

* ``p_roll_continue_state`` — handed-off state at the boundary
  (mod:2196).
* ``node_balance_fix_quantity_eq_lower`` — upper-level "anchor" pin
  on v_state at the last timestep (mod:2760).

Both are gated on ``p_nested_model['solveFirst'] == 0`` (i.e. this is
not the first sub-solve), which is the case for these fixtures.
"""
from pathlib import Path
import polars as pl
import pytest
from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool


def _flextool_obj(work: Path) -> float:
    """Pick the parquet matching ``solve_current.csv``'s solve name —
    in multi-solve fixtures the solve_data/ snapshot is keyed to the
    specific sub-solve we want to reproduce, not whichever
    ``v_obj__*.parquet`` lexically sorts first."""
    sc_file = work / "solve_data" / "solve_current.csv"
    if sc_file.exists():
        sc = pl.read_csv(sc_file)
        if sc.height > 0:
            solve = sc["solve"][0]
            parq = work / "output_raw" / f"v_obj__{solve}.parquet"
            if parq.exists():
                return pl.read_parquet(parq)["objective"][0]
    parq_list = sorted(work.glob("output_raw/v_obj__*.parquet"))
    if parq_list:
        return pl.read_parquet(parq_list[-1])["objective"][0]
    return pl.read_csv(work / "output_raw" / "v_obj.csv")["objective"][0]


@pytest.mark.parametrize("fixture", [
    "work_multi_fullYear_battery_nested_24h_invest_one_solve",
    "work_multi_fullYear_battery_nested_multi_invest",
    "work_multi_fullYear_battery_nested_sample_invest_one_solve",
])
def test_multi_fullYear_battery_nested_parity(fixture):
    work = Path(__file__).resolve().parent / "data" / fixture
    data = load_flextool(work)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj(work)
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"obj mismatch ({fixture}): flexpy={sol.obj}, "
        f"flextool={flextool_obj}, rel={rel}")
