"""The note on what a not-clean HiGHS outcome means (``engine/solve/_solver_outcome.py``, W-0540..W-0542)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flextool import run_warnings
from flextool.engine.solve._solver_outcome import (
    IPM_RELATIVE_ONLY, RELATIVE_TOLERANCES, interpret_optimal, note_solver_outcome, snapshot_highs_outcome,
)

_TOL = {"primal_feasibility_tolerance": 1e-7, "dual_feasibility_tolerance": 1e-7,
        "primal_residual_tolerance": 1e-7, "dual_residual_tolerance": 1e-7}


def _facts(**over):
    """A clean IPM-without-crossover LP outcome (every check computed and within tolerance), changed by *over*."""
    facts = {"model_status": "kOptimal", "valid": True, "ipm_iteration_count": 29, "crossover_iteration_count": 0,
             "mip_node_count": -1, **_TOL}
    for stem in ("primal_infeasibilities", "dual_infeasibilities", "primal_residual_errors", "dual_residual_errors"):
        facts[f"num_{stem}"] = 0
        facts[f"num_relative_{stem}"] = 0
    for stem in ("primal_infeasibility", "dual_infeasibility", "primal_residual_error", "dual_residual_error"):
        facts[f"max_{stem}"] = 1e-9
        facts[f"max_relative_{stem}"] = 1e-10
    facts.update(over)
    return facts


def test_a_clean_outcome_gets_no_note():
    assert interpret_optimal(_facts(), is_mip=False) is None
    # crossover ran: a vertex HiGHS certified
    assert interpret_optimal(_facts(crossover_iteration_count=12, num_dual_residual_errors=-1), is_mip=False) is None


def test_relative_violation_gives_the_factor_of_the_worst_check():
    """HiGHS: "Model status is Optimal since relative violation of tolerances is no more than 2.96"."""
    note = interpret_optimal(_facts(
        num_relative_dual_infeasibilities=1, max_relative_dual_infeasibility=2.96e-7,
        num_relative_dual_residual_errors=3, max_relative_dual_residual_error=2.01e-7), is_mip=False)
    assert note.code == RELATIVE_TOLERANCES and note.fmt == {"factor": "2.96"}
    assert "1 relative dual infeasibilities up to 2.96e-07" in note.detail and "3 relative dual residual" in note.detail


def test_absolute_only_violation_after_ipm_without_crossover():
    """The "IPM imprecise ... Model status changed from Unknown to Optimal" case: 672 absolute dual residual
    errors, none relative."""
    note = interpret_optimal(_facts(num_dual_residual_errors=672, max_dual_residual_error=1.13e-7), is_mip=False)
    assert note.code == IPM_RELATIVE_ONLY and note.fmt == {}
    assert note.detail.startswith("672 dual residual errors up to 1.13e-07")


def test_user_scaled_solve_whose_residual_checks_are_missing():
    """HiGHS recomputes its info without the residual checks after removing a user scaling (-1 counts): the facts
    cannot tell, so the IPM note is given."""
    note = interpret_optimal(_facts(num_primal_residual_errors=-1, num_dual_residual_errors=-1,
                                    num_relative_primal_residual_errors=-1, num_relative_dual_residual_errors=-1),
                             is_mip=False)
    assert note.code == IPM_RELATIVE_ONLY and "primal residual errors / dual residual errors" in note.detail


@pytest.mark.parametrize("facts, is_mip", [
    (None, False),
    (_facts(model_status="kUnknown", num_dual_residual_errors=5), False),
    (_facts(num_dual_residual_errors=5), True),
])
def test_no_note_without_facts_for_another_status_or_a_mip(facts, is_mip):
    assert interpret_optimal(facts, is_mip=is_mip) is None


def _sol(facts):
    return SimpleNamespace(pre_unscale_highs_outcome=facts, highs=None, _vars={})


def test_once_per_complete_solve_and_next_to_the_solver_output(coded_log):
    run_warnings.start_run()
    noted: set = set()
    acc = SimpleNamespace(accepted=True, near_optimal=False, model_status="kOptimal", message="")
    worse = _facts(num_relative_dual_infeasibilities=7, max_relative_dual_infeasibility=1.19e-7)
    with run_warnings.collecting():  # the per-roll scope of the cascade: the note does not wait for its end
        note_solver_outcome(_sol(worse), acc, solve_name="st_roll_roll_3", solver="highs", noted=noted)
        printed = coded_log()
    assert "Note: solve 'st_roll' optimal by HiGHS at up to 1.19x its relative tolerances [W-0540]" in printed
    assert "W-0540 in 'st_roll_roll_3': 7 relative dual infeasibilities up to 1.19e-07" in printed
    note_solver_outcome(_sol(worse), acc, solve_name="st_roll_roll_4", solver="highs", noted=noted)
    assert "Note:" not in coded_log()
    note_solver_outcome(_sol(_facts(num_dual_residual_errors=3)), acc, solve_name="st_roll_roll_5", solver="highs",
                        noted=noted)
    assert "used interior point without crossover; HiGHS warnings expected [W-0541]" in coded_log()
    # not HiGHS, not accepted, a clean solve: nothing
    note_solver_outcome(_sol(worse), acc, solve_name="other", solver="gurobi", noted=noted)
    rejected = SimpleNamespace(accepted=False, near_optimal=False, model_status="kInfeasible", message="")
    note_solver_outcome(_sol(worse), rejected, solve_name="other", solver="highs", noted=noted)
    note_solver_outcome(_sol(_facts()), acc, solve_name="other", solver="highs", noted=noted)
    assert coded_log() == ""


def test_near_optimal_acceptance(coded_log):
    run_warnings.start_run()
    acc = SimpleNamespace(accepted=True, near_optimal=True, model_status="kUnknown",
                          message="Accepted near-optimal solve for inv: ...")
    note_solver_outcome(_sol(None), acc, solve_name="inv", solver="highs", noted=set())
    printed = coded_log()
    assert "Note: solve 'inv' not certified optimal by HiGHS; its feasible solution is used [W-0542]" in printed
    assert "W-0542 in 'inv': Accepted near-optimal solve for inv" in printed


def test_snapshot_reads_a_live_handle():
    highspy = pytest.importorskip("highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    h.addVars(2, np.array([0.0, 0.0]), np.array([inf, inf]))
    h.changeColsCost(2, np.array([0, 1], dtype=np.int32), np.array([1.0, 2.0]))
    h.addRow(1.0, inf, 2, np.array([0, 1], dtype=np.int32), np.array([1.0, 1.0]))
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    h.run()
    facts = snapshot_highs_outcome(h)
    assert facts["model_status"] == "kOptimal" and facts["run_crossover"] == "off"
    assert facts["primal_feasibility_tolerance"] == 1e-7 and "num_relative_dual_residual_errors" in facts
    assert snapshot_highs_outcome(None) is None
