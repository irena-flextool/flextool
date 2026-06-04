"""Tier-8 closed-form decomposition test #24 — ``coal_chp`` fixture.

Exercises the **indirect process** path (CHP unit with one input flow
and two output flows on the noEff side).  Active features:
slack + commodity-buy-noEff + maxFlow (no online, no invest, no co2).

If a future polar_high patch adds a per-output-flow obj term and forgets
to gate it on the indirect topology, this test will catch the
double-count immediately.
"""

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.decomposition._components import total_decomposed_obj
from tests.engine_polars.perturbation._harness import solve_full


SCENARIO = "coal_chp"

def test_obj_decomposition_coal_chp(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.process_indirect is not None, (
        "coal_chp fixture should carry an indirect process")
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
