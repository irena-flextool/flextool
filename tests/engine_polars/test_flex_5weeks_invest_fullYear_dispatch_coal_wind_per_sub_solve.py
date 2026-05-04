"""flexpy ``5weeks_invest_fullYear_dispatch_coal_wind`` — per-sub-solve parity.

The fixture has 2 chained sub-solves
(``invest_1year_5weeks`` then ``y2020_fullYear_dispatch``).
The companion ``test_flex_5weeks_invest_fullYear_dispatch_coal_wind.py``
test only validates the LAST sub-solve.

Per-sub-solve snapshots produced by ``tests/_gen_input.py``'s
``SolverRunner.run`` hook let us validate every sub-solve
independently using the existing single-solve
``load_flextool`` + ``build_flextool`` path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_5weeks_invest_fullYear_dispatch_coal_wind"
)

SUB_SOLVES = ["invest_1year_5weeks", "y2020_fullYear_dispatch"]


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_5weeks_invest_fullYear_dispatch_per_sub_solve(sub_solve):
    sub_dir = WORK / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        pytest.skip(
            f"sub-solve fixture {sub_dir} not present "
            f"(regen via tests/_gen_input.py "
            f"5weeks_invest_fullYear_dispatch_coal_wind)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        for child in ("input", "output_raw"):
            src = WORK / child
            if src.exists():
                os.symlink(src, td / child)
        os.symlink(sub_dir, td / "solve_data")

        data = load_flextool(td)
        pb = Problem()
        build_flextool(pb, data)
        sol = pb.solve()

    assert sol.optimal, f"sub-solve {sub_solve}: flexpy LP not optimal"
    parq = WORK / "output_raw" / f"v_obj__{sub_solve}.parquet"
    flextool_obj = pl.read_parquet(parq)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"sub-solve {sub_solve}: flexpy={sol.obj}, "
        f"flextool={flextool_obj}, rel={rel}"
    )
