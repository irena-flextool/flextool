"""polar_high ``5weeks_invest_fullYear_dispatch_coal_wind`` — per-sub-solve parity.

The fixture has 2 chained sub-solves
(``invest_1year_5weeks`` then ``y2020_fullYear_dispatch``).
The companion ``test_flex_5weeks_invest_fullYear_dispatch_coal_wind.py``
test only validates the LAST sub-solve.

Every sub-solve is rebuilt here through the production loader call
(:func:`tests._support.load.load_scenario` with ``step=``: that step's
post-solve provider, the DB reader and the handoff the step consumed),
solved, and compared with the objective the cascade got for that step.
Each step has its own provider (``keep_solutions=True``), so the invest
step is rebuilt from its own frames, not the dispatch step's.
"""

from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario

pytestmark = pytest.mark.solver


SCENARIO = "5weeks_invest_fullYear_dispatch_coal_wind"

SUB_SOLVES = ["invest_1year_5weeks", "y2020_fullYear_dispatch"]


@pytest.fixture(scope="module")
def run(scenario_run):
    return scenario_run(SCENARIO)


def test_5weeks_invest_fullYear_dispatch_sub_solves_are_the_cascade_steps(run):
    """The parametrisation below covers every sub-solve the cascade ran, in order."""
    assert [s.solve_name for s in run.steps] == SUB_SOLVES


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_5weeks_invest_fullYear_dispatch_per_sub_solve(sub_solve, run):
    step = run.step(sub_solve)
    assert step.obj is not None, f"sub-solve {sub_solve}: the cascade recorded no objective"
    data = load_scenario(run, step=sub_solve)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()

    assert sol.optimal, f"sub-solve {sub_solve}: polar_high LP not optimal"
    rel = abs(sol.obj - step.obj) / max(1.0, abs(step.obj))
    assert rel < 1e-9, (
        f"sub-solve {sub_solve}: rebuilt={sol.obj}, cascade={step.obj}, rel={rel}"
    )
