"""polar_high ``multi_year`` — per-sub-solve parity.

The ``multi_year`` fixture has 4 chained sub-solves
(``y2020_5week``, ``y2025_5week``, ``y2030_5week``, ``y2035_5week``).
The companion ``test_flex_multi_year.py`` test only validates the
LAST sub-solve.

Every sub-solve is rebuilt here through the production loader call
(:func:`tests._support.load.load_scenario` with ``step=``: that step's
post-solve provider, the DB reader and the handoff the step consumed),
solved, and compared with the objective the cascade got for that step.
Each step has its own provider (``keep_solutions=True``), so a non-last
step is rebuilt from its own frames, not the last step's.
"""

from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario

pytestmark = pytest.mark.solver


SCENARIO = "multi_year"

SUB_SOLVES = ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]


@pytest.fixture(scope="module")
def run(scenario_run):
    return scenario_run(SCENARIO)


def test_multi_year_sub_solves_are_the_cascade_steps(run):
    """The parametrisation below covers every sub-solve the cascade ran, in order."""
    assert [s.solve_name for s in run.steps] == SUB_SOLVES


@pytest.mark.parametrize("sub_solve", SUB_SOLVES)
def test_multi_year_per_sub_solve(sub_solve, run):
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
