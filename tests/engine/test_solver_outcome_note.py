"""The note on what a not-clean HiGHS outcome means (``engine/solve/_solver_outcome.py``, W-0540..W-0542).

W-0540 carries HiGHS's own factor X ("Model status is Optimal since relative violation of tolerances is no more
than X"): the oracle of the integration tests is HiGHS's log of the same solve, no stored numbers.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flextool import run_warnings
from flextool.engine.solve import _solver_outcome
from flextool.engine.solve._solver_outcome import (
    IPM_RELATIVE_ONLY, RELATIVE_TOLERANCES, interpret_optimal, note_solver_outcome, snapshot_highs_outcome,
)

_TOL = {"primal_feasibility_tolerance": 1e-7, "dual_feasibility_tolerance": 1e-7,
        "primal_residual_tolerance": 1e-7, "dual_residual_tolerance": 1e-7, "optimality_tolerance": 1e-7,
        "kkt_tolerance": 1e-7}
_IPX = ("Note: these HiGHS IPX warnings ('imprecise', 'Unwelcome IPX status', 'Model status changed from Unknown "
        "to Optimal') are expected in this solver mode and the solve was accepted by FlexTool's checks [W-0541]")


def _facts(**over):
    """A clean IPM-without-crossover LP outcome (every check computed and within tolerance, no basis, unscaled:
    X from HiGHS's info), changed by *over*."""
    facts = {"model_status": "kOptimal", "valid": True, "ipm_iteration_count": 29, "crossover_iteration_count": 0,
             "mip_node_count": -1, "basis_validity": 0, "primal_dual_objective_error": 1e-10,
             "kkt_source": "highs-info", **_TOL}
    for stem in ("primal_infeasibilities", "dual_infeasibilities", "primal_residual_errors", "dual_residual_errors"):
        facts[f"num_{stem}"] = 0
        facts[f"num_relative_{stem}"] = 0
    for stem in ("primal_infeasibility", "dual_infeasibility", "primal_residual_error", "dual_residual_error"):
        facts[f"max_{stem}"] = 1e-9
        facts[f"max_relative_{stem}"] = 1e-10
    facts.update(over)
    return facts


def _basis(**over):
    """A crossover (vertex) outcome: a valid basis, so HiGHS's X takes the absolute infeasibilities and the P-D
    error; the residuals are not computed (count -1)."""
    return _facts(crossover_iteration_count=12, basis_validity=1, num_primal_residual_errors=-1,
                  num_dual_residual_errors=-1, num_relative_primal_residual_errors=-1,
                  num_relative_dual_residual_errors=-1, max_primal_residual_error=math.inf,
                  max_dual_residual_error=math.inf, max_relative_primal_residual_error=math.inf,
                  max_relative_dual_residual_error=math.inf, **over)


def _codes(notes):
    return [(n.code, {k: v for k, v in n.fmt.items() if k != "rolls"}) for n in notes]


def test_a_clean_outcome_gets_only_the_ipx_note():
    """W-0541 at every interior-point solve without crossover, whatever X is; nothing after crossover."""
    assert _codes(interpret_optimal(_facts(), is_mip=False)) == [(IPM_RELATIVE_ONLY, {})]
    assert interpret_optimal(_basis(), is_mip=False) == []


def test_relative_violation_gives_the_factor_of_the_worst_check():
    """HiGHS: "Model status is Optimal since relative violation of tolerances is no more than 2.96"."""
    notes = interpret_optimal(_facts(
        num_relative_dual_infeasibilities=1, max_relative_dual_infeasibility=2.96e-7,
        num_relative_dual_residual_errors=3, max_relative_dual_residual_error=2.01e-7), is_mip=False)
    assert _codes(notes) == [(IPM_RELATIVE_ONLY, {}), (RELATIVE_TOLERANCES, {"factor": "2.96"})]
    detail = notes[1].detail
    assert "relative dual infeasibility 2.96e-07 = 2.96x its tolerance" in detail
    assert "relative dual residual error 2.01e-07 = 2.01x" in detail and "(no basis: relative measures)" in detail


def test_basis_solve_with_only_relative_counters_gets_no_w0540():
    """Today's false positive: with a basis HiGHS judges the absolute infeasibilities, not the relative ones."""
    facts = _basis(num_relative_dual_infeasibilities=7, max_relative_dual_infeasibility=1.19e-7)
    assert interpret_optimal(facts, is_mip=False) == []


def test_basis_solve_with_an_absolute_infeasibility():
    notes = interpret_optimal(_basis(num_primal_infeasibilities=1, max_primal_infeasibility=3e-7), is_mip=False)
    assert _codes(notes) == [(RELATIVE_TOLERANCES, {"factor": "3"})]
    assert "(basis: absolute infeasibilities and P-D error)" in notes[0].detail


def test_no_basis_only_the_pd_error():
    notes = interpret_optimal(_facts(primal_dual_objective_error=5e-7), is_mip=False)
    assert _codes(notes) == [(IPM_RELATIVE_ONLY, {}), (RELATIVE_TOLERANCES, {"factor": "5"})]
    assert "primal dual objective error 5e-07 = 5x its tolerance" in notes[1].detail


def test_kkt_tolerance_replaces_all_five():
    """``kkt_tolerance`` other than 1e-7 replaces the five tolerances (HiGHS ``HighsSolution.cpp:83-89``)."""
    tol = _solver_outcome._tolerances(_facts(kkt_tolerance=1e-9, primal_feasibility_tolerance=1e-6))
    assert (tol.primal_feasibility, tol.dual_feasibility, tol.primal_residual, tol.dual_residual,
            tol.optimality) == (1e-9,) * 5
    notes = interpret_optimal(_facts(kkt_tolerance=1e-9, max_relative_dual_residual_error=4e-9), is_mip=False)
    assert notes[-1].fmt == {"factor": "4", "rolls": ""}


def test_user_scaled_solve_takes_x_from_the_kkt_report():
    facts = _facts(user_objective_scale=-5, kkt_source="polar-high", kkt_x=2.0104, basis_valid=False,
                   kkt_terms=(("max_relative_dual_residual_error", 2.0104),), max_relative_dual_residual_error=2.0104e-7)
    assert _codes(interpret_optimal(facts, is_mip=False)) == [(IPM_RELATIVE_ONLY, {}),
                                                              (RELATIVE_TOLERANCES, {"factor": "2.01"})]


def test_user_scaled_solve_without_a_kkt_report_falls_back_to_the_ipx_note():
    """No report (HiGHS other than 1.15.x): no X, so no W-0540, whatever the (unscaled) info counters say."""
    facts = _facts(user_objective_scale=-5, kkt_source=None, kkt_unavailable="no KKT report (HiGHS 1.16.0)",
                   num_relative_dual_infeasibilities=9, max_relative_dual_infeasibility=5e-6)
    notes = interpret_optimal(facts, is_mip=False)
    assert _codes(notes) == [(IPM_RELATIVE_ONLY, {})] and "no KKT report" in notes[0].detail
    assert interpret_optimal(_basis(kkt_source=None), is_mip=False) == []


@pytest.mark.parametrize("facts, is_mip", [
    (None, False),
    (_facts(model_status="kUnknown", primal_dual_objective_error=5e-6), False),
    (_facts(primal_dual_objective_error=5e-6), True),
])
def test_no_note_without_facts_for_another_status_or_a_mip(facts, is_mip):
    assert interpret_optimal(facts, is_mip=is_mip) == []


def _sol(facts):
    return SimpleNamespace(pre_unscale_highs_outcome=facts, highs=None, _vars={})


_ACC = SimpleNamespace(accepted=True, near_optimal=False, model_status="kOptimal", message="")


def test_once_per_complete_solve_and_next_to_the_solver_output(coded_log):
    run_warnings.start_run()
    noted: set = set()
    worse = _facts(num_relative_dual_infeasibilities=7, max_relative_dual_infeasibility=1.19e-7)
    with run_warnings.collecting():  # the per-roll scope of the cascade: the note does not wait for its end
        note_solver_outcome(_sol(worse), _ACC, solve_name="st_roll_roll_3", solver="highs", noted=noted)
        printed = coded_log()
    lines = [line for line in printed.splitlines() if line.startswith("Note:")]
    # W-0541 first: it explains the IPX warnings above both
    assert lines == [_IPX, "Note: solve 'st_roll' optimal by HiGHS at up to 1.19x its relative tolerances [W-0540]"]
    assert "W-0540 in 'st_roll_roll_3': relative dual infeasibility 1.19e-07 = 1.19x its tolerance" in printed
    note_solver_outcome(_sol(worse), _ACC, solve_name="st_roll_roll_4", solver="highs", noted=noted)
    assert "Note:" not in coded_log()
    # not HiGHS, not accepted, a clean solve: nothing
    note_solver_outcome(_sol(worse), _ACC, solve_name="other", solver="gurobi", noted=noted)
    rejected = SimpleNamespace(accepted=False, near_optimal=False, model_status="kInfeasible", message="")
    note_solver_outcome(_sol(worse), rejected, solve_name="other", solver="highs", noted=noted)
    note_solver_outcome(_sol(_facts()), _ACC, solve_name="other", solver="highs", noted=noted)
    assert "Note:" not in coded_log()


def test_rolls_give_one_line_with_their_number_and_the_largest_factor(coded_log):
    """W2 follow-up: the rolls of a rolling solve gather their W-0540 (``pending``); the line, printed when the
    rolls end, says in how many rolls it happened and gives the largest factor, the worst roll first in the debug
    log.  A solve that is not rolling prints at once, as before."""
    from flextool.engine.solve._solver_outcome import flush_solver_outcome_notes

    run_warnings.start_run()
    noted: set = set()
    pending: dict = {}

    def facts(x):
        return _facts(num_relative_dual_infeasibilities=1, max_relative_dual_infeasibility=x * 1e-7)

    for k, x in ((1, 1.19), (2, None), (3, 3.51), (4, 2.01)):
        f = facts(x) if x is not None else _facts()
        note_solver_outcome(_sol(f), _ACC, solve_name="st_roll", sub_solve=f"st_roll_roll_{k}", solver="highs",
                            noted=noted, pending=pending)
    note_solver_outcome(_sol(facts(1.5)), _ACC, solve_name="invest", solver="highs", noted=noted, pending=pending)
    printed = coded_log()
    assert [line for line in printed.splitlines() if "[W-0540]" in line] == [
        "Note: solve 'invest' optimal by HiGHS at up to 1.5x its relative tolerances [W-0540]"]
    flush_solver_outcome_notes(pending, noted, keep={"st_roll"})  # its rolls go on
    assert "[W-0540]" not in coded_log()
    flush_solver_outcome_notes(pending, noted)
    printed = coded_log()
    assert [line for line in printed.splitlines() if "[W-0540]" in line] == [
        "Note: solve 'st_roll' optimal by HiGHS at up to 3.51x its relative tolerances in 3 rolls [W-0540]"]
    details = next(line for line in printed.splitlines() if line.startswith("W-0540 in 'st_roll_roll_3'"))
    assert details.index("st_roll_roll_3") < details.index("st_roll_roll_4") < details.index("st_roll_roll_1")
    assert pending == {}
    # A later instance of the same solve (nested): the debug log only.
    note_solver_outcome(_sol(facts(9.0)), _ACC, solve_name="st_roll", sub_solve="st_roll_roll_9", solver="highs",
                        noted=noted, pending=pending)
    flush_solver_outcome_notes(pending, noted)
    assert "Note:" not in coded_log()


def _w0540(text: str) -> list[str]:
    return [line for line in text.splitlines() if "[W-0540]" in line]


def _replay(order, coded_log) -> list[list[str]]:
    """Replay the cascade's W-0540 handling over *order* (``(sub-solve, complete solve, X or None)`` in run
    order), as ``native_run_model`` and ``_PolarHighCascadeSolver.run`` do it: the last index of each complete
    solve from the order, the note at the end of each sub-solve, then the flush that keeps the solves still
    running.  The W-0540 lines printed at each step."""
    from flextool.engine.solve._solver_outcome import flush_solver_outcome_notes, still_running

    run_warnings.start_run()
    noted: set = set()
    pending: dict = {}
    last = {complete: i for i, (_name, complete, _x) in enumerate(order)}
    printed = []
    for i, (name, complete, x) in enumerate(order):
        f = (_facts(num_relative_dual_infeasibilities=1, max_relative_dual_infeasibility=x * 1e-7)
             if x is not None else _facts())
        note_solver_outcome(_sol(f), _ACC, solve_name=complete, sub_solve=name, solver="highs", noted=noted,
                            pending=pending)
        flush_solver_outcome_notes(pending, noted, keep=still_running(last, i + 1))
        printed.append(coded_log())
    assert pending == {}
    return printed


def _line(solve: str, factor: str, rolls: str = "") -> str:
    return f"Note: solve '{solve}' optimal by HiGHS at up to {factor}x its relative tolerances{rolls} [W-0540]"


def test_interleaved_and_nested_rolls_print_after_their_last_roll(coded_log):
    """A complete solve's W-0540 prints once, right after its last sub-solve in the run (the order is known
    before the run).  Expected lines from that rule, by hand:

    * interleaved (rolling ``uc`` and its realization ``uc__realized_dispatch``): last ``uc`` = step 4, last
      dispatch = step 5.  ``uc`` had X in rolls 0 (3.0) and 1 (5.0): at step 4 "up to 5x ... in 2 rolls";
      dispatch rolls 1 (2.0) and 2 (4.0): at step 5 "up to 4x ... in 2 rolls"; nothing before.
    * nested (rolling ``P`` contains rolling ``C``, renamed per period, then a single ``S``): last P = 3, last
      C = 5, last S = 6.  P: X only in roll 0 (1.5): step 3 "up to 1.5x ... in 1 roll" (its own last roll,
      although C still runs).  C: 2.5, 7.0, 1.2 in three rolls: step 5 "up to 7x ... in 3 rolls".  S, not a
      roll: step 6 "up to 3x" without a roll count.
    """
    U, D = "uc", "uc__realized_dispatch"
    steps = _replay([(f"{U}_roll_0", U, 3.0), (f"{D}_roll_0", D, None), (f"{U}_roll_1", U, 5.0),
                     (f"{D}_roll_1", D, 2.0), (f"{U}_roll_2", U, None), (f"{D}_roll_2", D, 4.0)], coded_log)
    assert [_w0540(text) for text in steps] == [
        [], [], [], [], [_line(U, "5", " in 2 rolls")], [_line(D, "4", " in 2 rolls")]]

    steps = _replay([("P_roll_0", "P", 1.5), ("C_p2020_roll_0", "C", 2.5), ("C_p2020_roll_1", "C", None),
                     ("P_roll_1", "P", None), ("C_p2025_roll_0", "C", 7.0), ("C_p2025_roll_1", "C", 1.2),
                     ("S", "S", 3.0)], coded_log)
    assert [_w0540(text) for text in steps] == [
        [], [], [], [_line("P", "1.5", " in 1 roll")], [], [_line("C", "7", " in 3 rolls")], [_line("S", "3")]]
    details = next(line for line in steps[5].splitlines() if line.startswith("W-0540 in 'C_"))
    assert details.index("C_p2025_roll_0") < details.index("C_p2020_roll_0") < details.index("C_p2025_roll_1")

    # A renamed single instance (not a roll) prints at once, without "rolls".
    steps = _replay([("X_p2020", "X", 2.0), ("Y", "Y", None)], coded_log)
    assert [_w0540(text) for text in steps] == [[_line("X", "2")], []]


def test_the_generic_ipx_note_is_printed_once_per_run(coded_log):
    """W-0541 (HiGHS's IPX warnings are expected): at the first interior-point solve without crossover, also a
    clean one; the later solves of the run, also other complete solves, go to the debug log.  Not for simplex."""
    run_warnings.start_run()
    noted: set = set()
    note_solver_outcome(_sol(_basis()), _ACC, solve_name="simplex", solver="highs", noted=noted)
    assert coded_log() == ""
    for solve in ("invest", "dispatch_roll_1", "dispatch_roll_2", "other"):
        note_solver_outcome(_sol(_facts()), _ACC, solve_name=solve, solver="highs", noted=noted)
    assert [line for line in coded_log().splitlines() if line.startswith("Note:")] == [_IPX]
    assert run_warnings.tally() == {}                         # a note, not a warning


def test_near_optimal_acceptance(coded_log):
    run_warnings.start_run()
    acc = SimpleNamespace(accepted=True, near_optimal=True, model_status="kUnknown",
                          message="Accepted near-optimal solve for inv: ...")
    note_solver_outcome(_sol(None), acc, solve_name="inv", solver="highs", noted=set())
    printed = coded_log()
    assert "Note: solve 'inv' not certified optimal by HiGHS; its feasible solution is used [W-0542]" in printed
    assert "W-0542 in 'inv': Accepted near-optimal solve for inv" in printed


# ---------------------------------------------------------------------------
# Real HiGHS: the factor is HiGHS's logged X
# ---------------------------------------------------------------------------

_X_LINE = re.compile(r"since relative violation of tolerances is no more than\s+(\S+)")


def _highs_x(log: str) -> float | None:
    """HiGHS's X of the full LP (the line after ``Performed postsolve``; a "Before postsolve" one concerns the
    presolved LP only), or ``None`` when HiGHS printed none."""
    block = log.split("After solving the user-scaled model")[0].rsplit("Performed postsolve", 1)[-1]
    found = _X_LINE.findall(block)
    return float(found[-1]) if found else None


def _same_3_digits(note_factor: str, logged: float) -> bool:
    """The note's ``%.3g`` *note_factor* is HiGHS's ``%8.3g`` *logged* (up to a half-unit rounding tie)."""
    return abs(float(note_factor) - logged) <= 0.5 * 10 ** (math.floor(math.log10(abs(logged))) - 2) * 1.0001


def _small_lp(seed: int):
    """polar-high's small KKT test LP: 400 columns x 300 ``>=`` rows, positive costs, mixed upper bounds."""
    highspy = pytest.importorskip("highspy")
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    rng = np.random.default_rng(seed)
    n, m = 400, 300
    for _ in range(n):
        h.addVar(0, rng.choice([10.0, 1e4, inf]))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), rng.uniform(0.1, 5, n) * rng.choice([1, 1e3], n))
    for _ in range(m):
        idx = rng.choice(n, 10, replace=False).astype(np.int32)
        h.addRow(rng.uniform(1, 100), inf, 10, idx, rng.uniform(0.01, 30, 10))
    return h


def _solve_logged(h, **options) -> str:
    import highspy

    log: list[str] = []
    h.setCallback(lambda _t, msg, *_a: log.append(msg), None)
    h.startCallback(highspy.cb.kCallbackLogging)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    for k, v in options.items():
        h.setOptionValue(k, v)
    h.run()
    return "".join(log)


_HIGHS_115 = False
try:
    import highspy as _highspy

    _HIGHS_115 = _highspy.Highs().version().startswith("1.15.") and _highspy.Highs().version() != "1.15.0"
except ImportError:  # pragma: no cover
    pass
requires_highs_115 = pytest.mark.skipif(not _HIGHS_115, reason="polar-high's KKT report: HiGHS 1.15.x")


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "user_scaled"])
def test_the_factor_is_the_x_highs_logged(scaled):
    """Seed 2 of polar-high's small LP, IPM without crossover at a loose ``ipm_optimality_tolerance``: HiGHS logs
    "no more than X" and W-0540 gives that X (unscaled: from HiGHS's info; user-scaled: from the KKT report, as
    HiGHS's info is recomputed in the unscaled space).  At the default tolerance HiGHS logs no X and there is no
    W-0540."""
    if scaled and not _HIGHS_115:
        pytest.skip("polar-high's KKT report: HiGHS 1.15.x")
    scale = {"user_bound_scale": -3, "user_objective_scale": -5} if scaled else {}
    h = _small_lp(2)
    log = _solve_logged(h, ipm_optimality_tolerance=1e-5, **scale)
    logged = _highs_x(log)
    facts = snapshot_highs_outcome(h)
    assert facts["kkt_source"] == ("polar-high" if scaled else "highs-info")
    notes = interpret_optimal(facts, is_mip=False)
    assert logged is not None and logged > 1
    assert [n.code for n in notes] == [IPM_RELATIVE_ONLY, RELATIVE_TOLERANCES]
    assert _same_3_digits(notes[1].fmt["factor"], logged)

    h = _small_lp(2)
    log = _solve_logged(h, **scale)
    assert _highs_x(log) is None
    assert [n.code for n in interpret_optimal(snapshot_highs_outcome(h), is_mip=False)] == [IPM_RELATIVE_ONLY]


@requires_highs_115
@pytest.mark.parametrize("report", ["none", "raises"])
def test_user_scaled_without_a_report_falls_back(monkeypatch, report):
    """``kkt_report`` ``None`` (HiGHS other than 1.15.x) or failing: no W-0540 although HiGHS logged an X; the
    once-per-run W-0541 stays."""
    import polar_high

    def no_report(h):
        if report == "raises":
            raise RuntimeError("boom")
        return None

    monkeypatch.setattr(polar_high, "kkt_report", no_report)
    h = _small_lp(2)
    log = _solve_logged(h, ipm_optimality_tolerance=1e-5, user_bound_scale=-3, user_objective_scale=-5)
    assert _highs_x(log) is not None
    facts = snapshot_highs_outcome(h)
    assert facts["kkt_source"] is None and "kkt_x" not in facts
    assert facts["kkt_unavailable"].startswith("no KKT report (HiGHS 1." if report == "none" else "KKT report failed")
    assert [n.code for n in interpret_optimal(facts, is_mip=False)] == [IPM_RELATIVE_ONLY]


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
    assert facts["kkt_tolerance"] == 1e-7 and facts["user_objective_scale"] == 0
    assert snapshot_highs_outcome(None) is None


def test_save_memory_solve_has_no_snapshot_and_no_report(monkeypatch):
    """A ``save_memory`` (or cold) HiGHS solve runs in a child process: ``Solution.highs`` is the shim, which has
    no ``getInfo``, so no snapshot and no ``kkt_report`` (its +2.5 GB transient on the largest LPs)."""
    import polar_high

    from flextool.engine.solve._subprocess_solve import _SolHighsShim

    def must_not_run(h):
        raise AssertionError("kkt_report called")

    monkeypatch.setattr(polar_high, "kkt_report", must_not_run)
    shim = _SolHighsShim(col_names=["x"], row_names=[], col_value=np.array([0.0]), col_dual=np.array([0.0]),
                         row_dual=np.array([]), objective=0.0)
    assert snapshot_highs_outcome(shim) is None


# ---------------------------------------------------------------------------
# A FlexTool rolling solve, IPM without crossover
# ---------------------------------------------------------------------------


def test_rolling_ipm_solve_notes_follow_highs_log(tmp_path_factory, monkeypatch, capfd, coded_log):
    """The rolling dispatch of ``tests.json`` with ``solver=ipm``, ``run_crossover=off``, ``presolve=off`` and a
    loose ``ipm_optimality_tolerance``: every roll gets W-0540 iff HiGHS's log of that roll has an X line, with
    that X; W-0541 is printed once in the run, W-0540 once for the solve."""
    from tests.engine._unit_db import b64, make_db, run

    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    seen: list[list] = []
    real = _solver_outcome.interpret_optimal

    def spy(facts, *, is_mip):
        notes = real(facts, is_mip=is_mip)
        seen.append(notes)
        return notes

    monkeypatch.setattr(_solver_outcome, "interpret_optimal", spy)
    roll = "dispatch_fullYear_roll"
    pmap = b64({"index_type": "str", "index_name": "period", "data": [["p2020", "2day"]]}, "map")
    args = {"index_type": "str", "index_name": "x", "data": [
        ["solver", "ipm"], ["presolve", "off"], ["run_crossover", "off"], ["ipm_optimality_tolerance", "1e-5"]]}
    values = [["solve", roll, "period_timeset", pmap],
              ["solve", roll, "realized_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
              ["solve", roll, "rolling_solve_horizon", b64(8.0, "float")],
              ["solve", roll, "rolling_solve_jump", b64(4.0, "float")],
              ["solve", roll, "solver_arguments", b64(args, "map")]]
    url, scen = make_db(tmp_path_factory, alt="ipm_roll", values=values,
                        base_chain=["init", "west", "coal", "fullYear", roll])
    run(tmp_path_factory, url, scen)
    stdout = capfd.readouterr().out
    logged = [_highs_x(block) for block in stdout.split("Running HiGHS")[1:]]
    assert len(logged) == len(seen) > 1
    assert any(x is not None for x in logged) and any(x is None for x in logged)
    for x, notes in zip(logged, seen):
        w0540 = [n for n in notes if n.code == RELATIVE_TOLERANCES]
        assert [n.code for n in notes][0] == IPM_RELATIVE_ONLY
        if x is None:
            assert w0540 == []
        else:
            assert len(w0540) == 1 and _same_3_digits(w0540[0].fmt["factor"], x), (x, w0540)
    printed = [line for line in coded_log().splitlines() if line.startswith("Note:") and "[W-054" in line]
    hits = [x for x in logged if x is not None]
    worst = max(hits)
    rolls = f"{len(hits)} roll{'s' if len(hits) != 1 else ''}"
    assert printed == [_IPX, f"Note: solve '{roll}' optimal by HiGHS at up to {worst:.3g}x its relative "
                             f"tolerances in {rolls} [W-0540]"]


def test_interleaved_separate_dispatch_notes_follow_highs_log(tmp_path_factory, monkeypatch, capfd, coded_log):
    """``rolling_separate_dispatch.json`` (rolling ``1week_rolling``, its rolls interleaved with those of the
    realization solve ``1week_rolling__realized_dispatch``) as an LP (``startup_method linear``: a MIP gets no
    W-0540), IPM without crossover, presolve off, a loose ``ipm_optimality_tolerance`` (the sibling inherits the
    arguments).  Oracle: the X of each roll in HiGHS's own log, grouped by the ``Solve start:`` lines.  Each
    complete solve gets one W-0540 line, after the HiGHS output of its last roll and before the next
    ``Solve start:``, with the largest X of its group and the number of its rolls that had an X."""
    import json
    import logging
    import sys

    from tests.db_utils import json_to_db
    from tests.engine._unit_db import b64
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "rolling_separate_dispatch.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    for row in data["parameter_values"]:
        if row[0] == "unit" and row[2] == "startup_method":
            row[3] = b64("linear", "str")
    args = {"index_type": "str", "index_name": "x", "data": [
        ["solver", "ipm"], ["presolve", "off"], ["run_crossover", "off"], ["ipm_optimality_tolerance", "1e-5"]]}
    data["parameter_values"].append(["solve", "1week_rolling", "solver_arguments", b64(args, "map"),
                                     "1week_rolling"])
    tmp = tmp_path_factory.mktemp("interleaved_w0540")
    (tmp / "fixture.json").write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(tmp / "fixture.json", tmp / "fixture.sqlite")
    migrate_database(url)
    capfd.readouterr()
    # The coded lines into the same captured stream as HiGHS's log, as on the console of a run.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger(run_warnings.LOGGER_NAME).addHandler(console)
    try:
        assert run_chain_from_db(url, "rolling_separate_dispatch", work_folder=tmp / "work", keep_solutions=True)
    finally:
        logging.getLogger(run_warnings.LOGGER_NAME).removeHandler(console)
    stdout = capfd.readouterr().out

    # One segment per sub-solve: "<complete>, i/n", its HiGHS output, then what printed before the next start.
    segments = stdout.split("Solve start: ")[1:]
    uc, disp = "1week_rolling", "1week_rolling__realized_dispatch"
    names = [segment.split(",", 1)[0] for segment in segments]
    assert names == [uc, disp] * 3
    xs: dict[str, list[float]] = {uc: [], disp: []}
    for name, segment in zip(names, segments):
        x = _highs_x(segment.split("Running HiGHS", 1)[1])
        if x is not None:
            xs[name].append(x)
    if not (xs[uc] and xs[disp]):
        pytest.skip(f"HiGHS logged no X in every solve on this platform: {xs}")

    def expected(name: str) -> str:
        count = len(xs[name])
        return _line(name, f"{max(xs[name]):.3g}", f" in {count} roll{'s' if count != 1 else ''}")

    found = [_w0540(segment) for segment in segments]
    assert found == [[], [], [], [], [expected(uc)], [expected(disp)]]
    for k in (4, 5):  # after the HiGHS output of that roll
        assert segments[k].index(found[k][0]) > segments[k].rindex("Running HiGHS")
