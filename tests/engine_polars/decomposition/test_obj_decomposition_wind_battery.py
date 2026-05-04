"""Tier-8 closed-form decomposition test #22 — ``wind_battery`` fixture.

Active features: slack + nodeBalance.state_change (bind_within_timeset)
+ self_discharge.  No commodity, no co2, no invest.  The state cycling
itself doesn't have an obj term unless ``use_reference_price`` is set;
it influences obj only via the slack values it pins.

The decomposed obj must equal ``sol.obj`` to ~1e-9 rel.  A failure
indicates a missing obj term tied to storage (e.g., §10.1
``storage_state_reference_price``), or a slack-side bug.
"""

from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.decomposition._components import total_decomposed_obj
from tests.engine_polars.perturbation._harness import solve_full


WORK = Path(__file__).resolve().parents[1] / "data" / "work_wind_battery"


def test_obj_decomposition_wind_battery():
    data = load_flextool(WORK)
    assert data.nodeState is not None and data.nodeState.height > 0, (
        "wind_battery fixture should carry a storage node")
    pb, sol = solve_full(data)
    assert sol.optimal, "LP did not solve to optimality"

    total, components = total_decomposed_obj(data, sol)
    rel = abs(total - sol.obj) / max(1.0, abs(sol.obj))

    if rel >= 1e-9:
        nz = {k: v for k, v in components.items() if abs(v) > 1.0}
        pytest.fail(
            f"obj decomposition mismatch:\n"
            f"  decomposed = {total!r}\n"
            f"  sol.obj    = {sol.obj!r}\n"
            f"  rel diff   = {rel!r}\n"
            f"Components (|v|>1):\n"
            + "\n".join(f"  {k}: {v:,.4f}" for k, v in sorted(nz.items())))
