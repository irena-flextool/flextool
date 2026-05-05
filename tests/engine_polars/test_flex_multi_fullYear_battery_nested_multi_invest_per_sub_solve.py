"""flexpy ``multi_fullYear_battery_nested_multi_invest`` — per-sub-solve parity.

The fixture has 80 chained sub-solves across 3 nested levels:

* 4 invest_5weeks_p20XX solves   — outer-level investment LPs.
* 4 storage_fullYear_6h_p20XX solves — middle-level annual storage anchors.
* 72 dispatch_fullYear_roll_roll_<i> solves — inner rolling-horizon dispatch.

The companion ``test_flex_multi_fullYear_battery_nested.py`` validates
only the LAST dispatch sub-solve.

Per-sub-solve snapshots produced by ``tests/_gen_input.py``'s
``SolverRunner.run`` hook let us spot-check every nesting level
independently.  We pick a representative subset (all 4 invest sub-solves,
all 4 storage sub-solves, and dispatch rolls 0 / middle / last) rather
than all 80 to keep the test wall-time bounded.

Status (2026-05-01)
-------------------
* invest_5weeks_p20XX           — all 4 PASS at machine precision.
* storage_fullYear_6h_p20XX     — all 4 PASS.  Closed by two related
  fixes: (1) the input-loader's multi-resolution block-rewrite no
  longer fires on single-block fixtures (which spuriously promoted
  every nodeBalance entity into nodeStateBlock and forced v_state to
  be constant across the period via the cyclic
  stateConstantWithinBlock_eq chain), and (2) flow / slack terms in
  ``nodeBalance_eq`` are now multiplied by ``p_step_duration`` so the
  per-step balance has correct dimensions for fixtures whose default
  step duration ≠ 1 (storage_fullYear_6h: 6h per step).
* dispatch_fullYear_roll_roll_X — sampled 0, middle, last; all PASS.
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


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_multi_fullYear_battery_nested_multi_invest"
)

# (sub_solve_name, xfail_reason_or_None)
_INVEST = [
    ("invest_5weeks_p2020", None),
    ("invest_5weeks_p2025", None),
    ("invest_5weeks_p2030", None),
    ("invest_5weeks_p2035", None),
]
_STORAGE = [
    ("storage_fullYear_6h_p2020", None),
    ("storage_fullYear_6h_p2025", None),
    ("storage_fullYear_6h_p2030", None),
    ("storage_fullYear_6h_p2035", None),
]
_DISPATCH_SAMPLE = [
    ("dispatch_fullYear_roll_roll_0", None),
    ("dispatch_fullYear_roll_roll_36", None),  # middle of 0..71
    ("dispatch_fullYear_roll_roll_71", None),  # last
]
SUB_SOLVES = _INVEST + _STORAGE + _DISPATCH_SAMPLE


@pytest.mark.parametrize(
    "sub_solve,xfail_reason",
    [pytest.param(name, reason, id=name) for name, reason in SUB_SOLVES],
)
def test_multi_fullYear_battery_nested_multi_invest_per_sub_solve(
    sub_solve, xfail_reason, request
):
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    sub_dir = WORK / f"solve_data_{sub_solve}"
    if not sub_dir.exists():
        pytest.skip(
            f"sub-solve fixture {sub_dir} not present (regen via "
            f"tests/_gen_input.py multi_fullYear_battery_nested_multi_invest)"
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
