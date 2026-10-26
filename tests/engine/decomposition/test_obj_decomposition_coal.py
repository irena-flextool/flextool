"""Tier-8 closed-form decomposition test #21 — ``coal`` fixture.

Active features in ``work_coal/``: slack + commodity-buy-eff + maxFlow
(no co2, no online, no invest).  The decomposed obj must equal
``sol.obj`` to ~1e-9 rel.

A failure prints the per-component dict, so any double-count or
sign-flip on the §1, §2, or §6 sections of ``audit/objective_audit.md``
is immediately localised.
"""

import pytest

from tests._support.load import load_scenario
from tests.engine.decomposition._components import total_decomposed_obj
from tests.engine.perturbation._harness import solve_full


SCENARIO = "coal"


def test_obj_decomposition_coal(scenario_run):
    run = scenario_run(SCENARIO)
    data = load_scenario(run)
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
