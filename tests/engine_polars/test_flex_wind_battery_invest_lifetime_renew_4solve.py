"""flexpy ``wind_battery_invest_lifetime_renew_4solve`` — per-sub-solve parity.

The fixture has 4 chained sub-solves
(``y2020_5week``, ``y2025_5week``, ``y2030_5week``, ``y2035_5week``).
Today flexpy doesn't orchestrate the chain end-to-end (no handoff
capture / replay between sub-solves); instead, this test loads each
sub-solve's snapshot independently and verifies flexpy matches
flextool's parquet at THAT sub-solve.

Per-sub-solve snapshots are produced by ``tests/_gen_input.py``'s
hook around ``SolverRunner.run``; see that script for details.

Status (2026-05-01)
-------------------
All four sub-solves pass at parity.  Flexpy reads the multi-solve
handoff state (``p_entity_previously_invested_capacity``,
``p_entity_invested``, ``p_entity_divested``) from the snapshot's
``solve_data/`` dir and threads it into the cumulative invest/divest
caps (mod:3597-3623), so each sub-solve closes at machine precision
when given its peer snapshot.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_wind_battery_invest_lifetime_renew_4solve"
)

SUB_SOLVES = ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]


@pytest.mark.parametrize(
    "sub_solve",
    [pytest.param(name, id=name) for name in SUB_SOLVES],
)
def test_wind_battery_invest_lifetime_renew_4solve_per_sub_solve(sub_solve):
    sub_dir = WORK / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        pytest.skip(
            f"sub-solve fixture {sub_dir} not present "
            f"(regen via tests/_gen_input.py "
            f"wind_battery_invest_lifetime_renew_4solve)"
        )

    # Build a temporary work dir whose ``solve_data/`` is the requested
    # sub-solve's snapshot; ``input/`` and ``output_raw/`` are shared
    # across sub-solves so we just symlink them.
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
