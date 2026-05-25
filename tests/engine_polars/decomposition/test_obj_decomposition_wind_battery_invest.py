"""Tier-8 closed-form decomposition test #23 — ``wind_battery_invest``
fixture.

Active features: slack + storage + invest_n (battery node).
``invest_p`` is also active on ``battery_inverter`` but the fixture's
solution drives it to 0 (the user constraint pins it via the kW/kWh
relation).  The decomposed obj must equal ``sol.obj`` to ~1e-9 rel.

A failure here pinpoints either an annuity / lifetime-fixed-cost
sign error, a missing divest accounting branch, or a missing operational
term that's small in dispatch fixtures but large under invest.
"""

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.decomposition._components import total_decomposed_obj
from tests.engine_polars.perturbation._harness import solve_full


SCENARIO = "wind_battery_invest"

def test_obj_decomposition_wind_battery_invest(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.nd_invest_set is not None and data.nd_invest_set.height > 0, (
        "wind_battery_invest fixture should carry node-side invest")
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
