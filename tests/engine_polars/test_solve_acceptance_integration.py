"""Integration: the accept/reject gate wired into the cascade.

Unit coverage of the policy lives in ``test_solve_acceptance.py``.  Here we
drive the full ``run_chain_from_db`` cascade on the smallest fixture and
patch the gate's ``classify_acceptance`` to force each branch, proving the
control-flow contract:

* an ACCEPTED near-optimal solve does NOT abort — the cascade runs to
  completion and writes its solution (the behaviour the crossover-off case
  depends on); and
* a REJECTED solve still aborts cleanly with ``FlexToolSolveError``.
"""

from __future__ import annotations

import pytest
import spinedb_api as api

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars._solve_acceptance import (
    Acceptance,
    classify_acceptance as _real_classify,
)
from flextool.engine_polars._solve_state import FlexToolSolveError

pytestmark = pytest.mark.solver


def _first_scenario(sqlite) -> str:
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        return sorted(s.name for s in db.query(db.scenario_sq).all())[0]


def test_near_optimal_accept_does_not_abort_cascade(
    scenario_workdir, monkeypatch
) -> None:
    """Forcing the near-optimal accept branch must let the cascade complete
    and keep the solution — it must not raise or discard outputs (the
    behaviour the crossover-off case depends on)."""
    sqlite = scenario_workdir("base") / "tests.sqlite"
    scenario = _first_scenario(sqlite)

    seen: list[bool] = []

    def _fake(sol, *, ranges_post, solve_name):
        # Run the real classifier for side-effect-free validation, then
        # override to the NEAR-optimal accept branch so the fall-through is
        # exercised even though the genuine solve certifies kOptimal.
        _real_classify(sol, ranges_post=ranges_post, solve_name=solve_name)
        seen.append(True)
        return Acceptance(
            accepted=True,
            near_optimal=True,
            message=f"Accepted near-optimal solve for {solve_name}: forced",
            scaling_hint=None,
        )

    monkeypatch.setattr(
        "flextool.engine_polars._orchestration.classify_acceptance", _fake
    )

    steps = run_chain_from_db(sqlite, scenario, keep_solutions=True)

    # Gate reached, and the near-optimal accept did NOT abort: every solve
    # ran to completion with its solution retained.
    assert seen, "acceptance gate was never reached"
    assert len(steps) >= 1
    for name, step in steps.items():
        assert step.solution is not None, f"{name}: solution discarded"
        assert step.obj is not None, f"{name}: obj is None"


def test_reject_still_raises(scenario_workdir, monkeypatch) -> None:
    """Forcing a reject must abort the cascade with FlexToolSolveError and
    an honest message — no silent success."""
    sqlite = scenario_workdir("base") / "tests.sqlite"
    scenario = _first_scenario(sqlite)

    def _fake(sol, *, ranges_post, solve_name):
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=f"forced reject for {solve_name}",
            scaling_hint=None,
        )

    monkeypatch.setattr(
        "flextool.engine_polars._orchestration.classify_acceptance", _fake
    )

    with pytest.raises(FlexToolSolveError):
        run_chain_from_db(sqlite, scenario, keep_solutions=True)
