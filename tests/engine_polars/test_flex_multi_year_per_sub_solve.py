"""polar_high ``multi_year`` — per-sub-solve parity.

The ``multi_year`` fixture has 4 chained sub-solves
(``y2020_5week``, ``y2025_5week``, ``y2030_5week``, ``y2035_5week``).
The companion ``test_flex_multi_year.py`` test only validates the
LAST sub-solve (the snapshot left in ``solve_data/`` after generation).

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

from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool

pytestmark = pytest.mark.solver


SCENARIO = "multi_year"

SUB_SOLVES = ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_multi_year_per_sub_solve(sub_solve, scenario_workdir):
    work = scenario_workdir(SCENARIO)
    sub_dir = work / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        pytest.skip(
            f"sub-solve fixture {sub_dir} not present "
            f"(regen via tests/_gen_input.py multi_year)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        for child in ("input", "output_raw"):
            src = work / child
            if src.exists():
                os.symlink(src, td / child)
        os.symlink(sub_dir, td / "solve_data")

        data = load_flextool(td)
        pb = Problem()
        build_flextool(pb, data)
        sol = pb.solve()

    assert sol.optimal, f"sub-solve {sub_solve}: polar_high LP not optimal"
    parq = work / "output_raw" / f"v_obj__{sub_solve}.parquet"
    flextool_obj = pl.read_parquet(parq)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"sub-solve {sub_solve}: polar_high={sol.obj}, "
        f"flextool={flextool_obj}, rel={rel}"
    )
