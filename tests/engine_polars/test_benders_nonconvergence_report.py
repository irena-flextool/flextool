"""Non-convergence reporting for the Benders cascade.

A Benders-decomposed solve that finds a FEASIBLE incumbent but never closes
the optimality gap to tolerance is NOT infeasible: every subproblem/master LP
solved to optimality and the written outputs are a valid feasible plan.  The
CLI must therefore

  * SUCCEED (exit 0) so the parquet / results-DB outputs are consumed, and
  * emit a LOUD warning naming the actual gap vs the required tolerance,

while a genuine failure (an LP HiGHS could not solve, or a Benders solve with
no feasible incumbent at all) must still exit 1.

These pin the ``_scan_cascade_optimality`` policy + the warning banner without
running a full solve.
"""
from __future__ import annotations

import logging
import math
import re

from flextool.cli.cmd_run_flextool import (
    _benders_nonconvergence_banner,
    _scan_cascade_optimality,
)
from flextool.engine_polars._benders import BendersResult
from flextool.engine_polars._orchestration import OrchestrationStep

# Model-instance vocabulary that must not leak into a user-facing diagnostic
# (mirrors ``test_benders_failure_messages``): FlexTool class names only.
_FORBIDDEN = re.compile(r"\b(trade|pipe|pipeline|line|region|regions)\b", re.I)


def _step(name, *, optimal, is_benders=False, obj=1.0,
          gap=None, tol=None, iters=None):
    return OrchestrationStep(
        solve_name=name,
        solution=None,
        handoff=None,
        obj=obj,
        optimal=optimal,
        is_benders=is_benders,
        benders_gap=gap,
        benders_tol=tol,
        benders_iterations=iters,
    )


def test_benders_result_carries_convergence_contract():
    # The tol / max_iters the loop ran under survive onto the result so the
    # CLI can report "gap X vs required tol Y".
    r = BendersResult(
        converged=False, iterations=20, total_objective=1.0,
        lower_bound=0.9, upper_bound=1.0, gap=0.1, region_costs={},
        invest={}, tol=0.001, max_iters=20,
    )
    assert r.tol == 0.001
    assert r.max_iters == 20


def test_nonconverged_benders_with_incumbent_succeeds(caplog):
    steps = {
        "lt_rp": _step("lt_rp", optimal=False, is_benders=True, obj=0.751,
                       gap=0.05, tol=0.001, iters=20),
        "dispatch": _step("dispatch", optimal=True),
    }
    with caplog.at_level(logging.ERROR):
        code, last = _scan_cascade_optimality(steps)
    # Feasible incumbent → run succeeds so the written outputs are consumed.
    assert code == 0
    assert last.solve_name == "dispatch"
    # ...but a loud warning fired naming the gap and the required tolerance.
    assert "did NOT meet its convergence tolerance" in caplog.text
    assert "5%" in caplog.text          # gap reached, 0.05 -> 5%
    assert "0.1%" in caplog.text        # tolerance, 0.001 -> 0.1%


def test_genuine_infeasible_exits_one(caplog):
    steps = {"mono": _step("mono", optimal=False, is_benders=False)}
    with caplog.at_level(logging.ERROR):
        code, _ = _scan_cascade_optimality(steps)
    assert code == 1
    assert "did not solve to optimality" in caplog.text


def test_benders_without_incumbent_exits_one():
    # No feasible incumbent (best_UB stayed +inf) → genuine failure.
    steps = {
        "lt_rp": _step("lt_rp", optimal=False, is_benders=True,
                       obj=math.inf, gap=math.inf, tol=0.001, iters=20),
    }
    code, _ = _scan_cascade_optimality(steps)
    assert code == 1


def test_all_optimal_exits_zero():
    steps = {
        "a": _step("a", optimal=True),
        "b": _step("b", optimal=True),
    }
    code, last = _scan_cascade_optimality(steps)
    assert code == 0
    assert last.solve_name == "b"


def test_banner_content_and_vocabulary():
    step = _step("lt_rp", optimal=False, is_benders=True, obj=0.751,
                 gap=0.05, tol=0.001, iters=20)
    msg = _benders_nonconvergence_banner("lt_rp", step)
    # Reports the numbers a user needs to act.
    assert "5%" in msg
    assert "0.1%" in msg
    assert "20" in msg
    assert "0.751" in msg
    # States the results are usable and were written.
    assert "feasible" in msg.lower()
    assert "written" in msg.lower()
    assert "not certified optimal" in msg.lower()
    # Stays in FlexTool class vocabulary — no model-instance leakage.
    assert not _FORBIDDEN.search(msg), _FORBIDDEN.search(msg)


def test_banner_handles_missing_metrics():
    # A Benders step that somehow lost its gap/tol/iters still renders.
    step = _step("lt_rp", optimal=False, is_benders=True, obj=0.751)
    msg = _benders_nonconvergence_banner("lt_rp", step)
    assert "unknown" in msg
