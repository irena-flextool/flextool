"""flexpy ``fullYear_roll`` parity — last-roll snapshot.

flextool runs this scenario as 72 sequential rolling-horizon dispatch
solves. The fixture preserves the solve_data for the *final* roll
(roll_71), with `p_roll_continue_state.csv` carrying the handed-off
state from roll_70. Since for this scenario the battery state hands
off as 0, single-solve parity on this snapshot reproduces the LP that
flextool solved at roll_71.
"""

from pathlib import Path

import polars as pl

from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_fullYear_roll"


def test_fullYear_roll_last_roll_parity():
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__dispatch_fullYear_roll_roll_71.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
