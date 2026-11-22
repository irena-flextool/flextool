"""The scaling range guard (:mod:`flextool.engine.autoscale._guard`): policy and messages.

Unit level: synthetic :class:`RangeReport` pools judged under a Layer 3 plan, no solve.  Every expected
number below was derived by hand from the spec's formulas (window ``[10·tol, 1e18·tol]``, warn above 13
decades) and checked by a second, independent derivation; none was copied from the code's output.

* A — HiGHS, tol 1e-7, pool 0.5 … 1e18: 18.3 decades, window [1e-6, 1e11], N_min_hard = -18 >
  N_max = -24 → error; the unscalable exponent min(-18, 6) = -18 puts the large end at 3.81e12; fitting
  the large end (N = -24) allows a violation of 1e-7·2^24 = 1.68, 3.36x the smallest limit.
* B — HiGHS, tol 1e-7, pool 1.6e-5 … 1.3e9: 13.91 decades → warning; hard window [-4, 6].
* C — Gurobi (default tolerance 1e-6, N = 0), pool 0.5 … 1e18: window [1e-5, 1e12], the large end is
  outside → a solver-level error that only warns.  D — with FeasibilityTol 1e-9: window [1e-8, 1e9].
* E — manual N = -10, tol 1e-7, pool 1e-4 … 1e3: S = [9.77e-8, 0.977], below 1e-6 → error; an exponent in
  [-6, 26] fits.
* F — roll recheck, tol 1e-8: roll 1 pool 1 … 1e12 gets N = -19 (smallest-side, raised to the 100·tol
  preference); roll 2 pool 1e-3 … 1e12 puts 1e-3·2^-19 = 1.91e-9 below 1e-7, so it gets its own N = -9.
"""
from __future__ import annotations

import math

import pytest
from polar_high import Problem
from polar_high.autoscale import RangeReport, ScalingConfig, recommend_scaling

from flextool.engine import _behaviour_changes
from flextool.engine.autoscale import _guard
from flextool.engine.autoscale._report import format_console_summary, render_guard
from flextool.engine.input._solve_state import FlexToolConfigError, ScalingRangeError
from flextool.engine.solve._solver_dispatch import CommercialOption

NAN = (math.nan, math.nan)


@pytest.fixture(autouse=True)
def _fresh_run(monkeypatch):
    """Each test is its own run: no warning remembered from another, no strict mode from the shell."""
    _behaviour_changes.start_run(None)
    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    yield
    _behaviour_changes.start_run(None)


def _report(b_lo: float, b_hi: float, *, cost=(1.0, 10.0), matrix=(1.0, 10.0)) -> RangeReport:
    return RangeReport(
        matrix=matrix, cost=cost, bound=NAN, rhs=(b_lo, b_hi), cross_group_max_ratio=math.nan,
        trigger=False, bound_rhs_meaningful=(b_lo, b_hi), bound_rhs_residue=(0, 0.0),
        bound_rhs_beyond_infinite=0, matrix_continuous=matrix, cost_continuous=cost,
        infinite_bound=1e20,
    )


def _problem(tol: float | None = None) -> Problem:
    pb = Problem()
    if tol is not None:
        pb.set_solver_options({"primal_feasibility_tolerance": tol, "dual_feasibility_tolerance": tol})
    return pb


def _guard_of(report, *, tol=None, cfg=None, mode="error", solver="highs", commercial=None, pre=None):
    pb = _problem(tol)
    plan = recommend_scaling(report, cfg or ScalingConfig(), primal_feasibility_tolerance=tol or 1e-7)
    out = _guard.run_guard(pb, ranges=report, ranges_pre=pre, plan=plan, solve_name="s1", mode=mode,
                           solver=solver, commercial_options=commercial)
    return plan, out


def _text(outcome, group="bound_rhs") -> str:
    (msg,) = [m for m in outcome.messages if m.group == group]
    return msg.text


# --------------------------------------------------------------------- bound / RHS


def test_unscalable_spread_is_an_error_with_numbers_and_override():
    plan, out = _guard_of(_report(0.5, 1e18))
    assert plan.user_bound_scale == -18 and plan.bound_tag.startswith("unscalable")
    assert out.level == "error" and out.verdict.window == pytest.approx((1e-6, 1e11))
    text = _text(out)
    for part in ("Solve s1 cannot be scaled", "span 18.3 decades", "(0.5 … 1e+18)", "at most 17 fit",
                 "primal feasibility tolerance 1e-07, HiGHS default",
                 "Fitting the largest value, the smallest limit could be violated by up to 1.68 (model "
                 "units), 3.36x its size.",
                 "user_bound_scale -18", "3.81e+12",
                 "To solve anyway: --scaling-guard warn (FLEXTOOL_SCALING_GUARD=warn; in the GUI: Solver "
                 "options, Scaling guard: Warn)."):
        assert part in text, part
    with pytest.raises(ScalingRangeError) as exc:
        _guard.enforce(out, base_solve="s1")
    assert isinstance(exc.value, FlexToolConfigError)
    assert "cannot be scaled" in str(exc.value)


def test_error_under_warn_prints_once_and_goes_on(coded_log):
    _plan, out = _guard_of(_report(0.5, 1e18), mode="warn")
    _guard.enforce(out, base_solve="s1")
    _guard.enforce(out, base_solve="s1")  # a later roll of the same solve
    printed = coded_log()
    assert printed.count("W-0544 s1: Solve s1 cannot be scaled") == 1
    assert printed.count("Warning: limits and bounds do not fit the solver's accurate range; solved anyway "
                         "('s1') [W-0544]") == 1


def test_spread_above_13_decades_warns_inside_the_window(coded_log):
    plan, out = _guard_of(_report(1.6e-5, 1.3e9))
    assert out.level == "warn"
    assert -4 <= plan.user_bound_scale <= 6
    b = out.verdict.finding("bound_rhs")
    assert b.numbers["decades"] == pytest.approx(13.91, abs=0.005)
    assert "span 13.9 decades" in _text(out) and "more than 13" in _text(out)
    _guard.enforce(out, base_solve="s1")  # a warning never raises
    printed = coded_log()
    assert "W-0543 s1: Solve s1: limits and bounds span 13.9 decades" in printed
    assert ("Warning: limits and bounds span over 13 decades; small limits may be violated ('s1') [W-0543]"
            in printed)


def test_off_runs_no_verdict_but_layer3_still_clips():
    plan_off, out = _guard_of(_report(1.6e-5, 1.3e9), mode="off")
    assert out.level == "off" and out.verdict is None and out.messages == ()
    _guard.enforce(out, base_solve="s1")
    plan_on, _ = _guard_of(_report(1.6e-5, 1.3e9), mode="error")
    assert plan_off.user_bound_scale == plan_on.user_bound_scale


def test_manual_exponent_outside_the_window_is_an_error_naming_it():
    plan, out = _guard_of(_report(1e-4, 1e3), cfg=ScalingConfig(user_bound_scale=-10))
    assert plan.bound_source == "manual" and plan.user_bound_scale == -10
    assert out.level == "error"
    text = _text(out)
    assert "user_bound_scale -10 (--user-bound-scale / FLEXTOOL_USER_BOUND_SCALE)" in text
    assert "9.77e-08 … 0.977" in text and "[-6, 26] fits" in text


def test_strict_raises_only_for_a_layer2_caused_error(monkeypatch, coded_log):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    # Layer 2 caused it: the pre-Layer-2 pool fitted.
    _plan, caused = _guard_of(_report(0.5, 1e18), mode="warn", pre=_report(1.0, 1e3))
    assert caused.strict_raise
    with pytest.raises(ScalingRangeError, match="FLEXTOOL_AUTOSCALE_STRICT=1"):
        _guard.enforce(caused, base_solve="s1")
    # Not caused by Layer 2 (already as wide before): under warn it only warns, even strict.
    _plan, own = _guard_of(_report(0.5, 1e18), mode="warn", pre=_report(0.5, 1e18))
    assert not own.strict_raise
    _guard.enforce(own, base_solve="s1")
    assert "s1: Solve s1 cannot be scaled" in coded_log()
    # Under off a Layer-2-caused error still raises with strict.
    _plan, off = _guard_of(_report(0.5, 1e18), mode="off", pre=_report(1.0, 1e3))
    with pytest.raises(ScalingRangeError):
        _guard.enforce(off, base_solve="s1")


# --------------------------------------------------------------------- commercial solvers


def test_commercial_solver_is_judged_unscaled_and_only_warns(coded_log):
    _plan, out = _guard_of(_report(0.5, 1e18), solver="gurobi")
    assert out.verdict.bound_scale == 0 and out.verdict.window == pytest.approx((1e-5, 1e12))
    assert out.verdict.finding("bound_rhs").level == "error"  # what the solver gets
    assert out.level == "warn"
    assert "gurobi gets no FlexTool scaling, so this is a warning only" in _text(out)
    _guard.enforce(out, base_solve="s1")  # mode error, but no raise
    printed = coded_log()
    assert "W-0544 s1: Solve s1 (gurobi) cannot be scaled" in printed and "('s1') [W-0544]" in printed


@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress", "copt"])
def test_commercial_tolerance_from_the_option_file(tmp_path, solver):
    from flextool.engine.solve._solver_dispatch import resolve_commercial_options
    from flextool.run_options import RunOptions

    name = _guard.COMMERCIAL_TOLERANCE_PARAMETERS[solver][0]
    opt = tmp_path / f"{solver}.opt"
    opt.write_text(f"{name} 1e-9\n")
    opts = resolve_commercial_options(solver, opt_file=opt, solver_config=None,
                                      run_options=RunOptions.from_env({}))
    tol_p, _tol_d, source = _guard.commercial_tolerances(solver, opts)
    assert tol_p == pytest.approx(1e-9) and str(opt) in source
    _plan, out = _guard_of(_report(0.5, 1e18), solver=solver, commercial=opts)
    assert out.verdict.window == pytest.approx((1e-8, 1e9))
    assert out.level == "warn"


def test_commercial_default_tolerance_without_the_parameter():
    opts = [CommercialOption("MIPGap", "0.01", "gurobi.opt")]
    assert _guard.commercial_tolerances("gurobi", opts)[:2] == (1e-6, 1e-6)


# --------------------------------------------------------------------- cost and matrix


def test_cost_spread_warns_only(coded_log):
    _plan, out = _guard_of(_report(1.0, 1e3, cost=(1e-8, 1e8)))
    cost = out.verdict.finding("cost")
    assert cost.level == "warn" and "spread" in cost.kind
    assert out.level == "warn"
    assert "the objective costs span 16.0 decades" in _text(out, "cost")
    _guard.enforce(out, base_solve="s1")
    printed = coded_log()
    assert "W-0545 s1: Solve s1: the objective costs span" in printed and "('s1') [W-0545]" in printed


def test_matrix_warns_only_for_dropped_or_refused_terms():
    _plan, dropped = _guard_of(_report(1.0, 1e3, matrix=(1e-10, 1.0)))
    assert dropped.verdict.finding("matrix").kind == "dropped" and dropped.level == "warn"
    assert "HiGHS drops them from the model" in _text(dropped, "matrix")
    _plan, refused = _guard_of(_report(1.0, 1e3, matrix=(1.0, 1e16)))
    assert refused.verdict.finding("matrix").kind == "refused" and refused.level == "warn"
    assert "HiGHS refuses such a model" in _text(refused, "matrix")
    # A wide spread alone (15 decades) is reported, not warned.
    _plan, wide = _guard_of(_report(1.0, 1e3, matrix=(1e-3, 1e12)))
    assert wide.level == "ok" and wide.messages == ()
    assert render_guard(wide)["findings"]["matrix"]["numbers"]["decades"] == pytest.approx(15.0)


# --------------------------------------------------------------------- report


def test_yaml_section_and_console_flag():
    _plan, out = _guard_of(_report(0.5, 1e18))
    tree = render_guard(out)
    assert tree["verdict"] == "error" and tree["mode"] == "error" and tree["solver"] == "highs"
    assert tree["primal_tolerance"] == 1e-7 and tree["tolerance_source"] == "HiGHS default"
    assert tree["window"] == {"lo": pytest.approx(1e-6), "hi": pytest.approx(1e11)}
    b = tree["findings"]["bound_rhs"]
    assert b["level"] == "error" and b["numbers"]["n_used"] == -18.0
    ranges = _report(0.5, 1e18)
    ranges = RangeReport(**{**{f: getattr(ranges, f) for f in ("matrix", "cost", "bound", "rhs",
                                                               "cross_group_max_ratio")}, "trigger": True})
    line = format_console_summary(ranges_pre=ranges, ranges_post=None, layer2_plan=None, layer3_plan=None,
                                  threshold_decades=6.0, guard=out)
    assert line.rstrip().endswith("guard=error")
    quiet = format_console_summary(ranges_pre=_report(1.0, 10.0), ranges_post=None, layer2_plan=None,
                                   layer3_plan=None, threshold_decades=6.0, guard=out)
    assert quiet.endswith(" guard=error")
    assert "guard=" not in format_console_summary(ranges_pre=_report(1.0, 10.0), ranges_post=None,
                                                  layer2_plan=None, layer3_plan=None,
                                                  threshold_decades=6.0)


# --------------------------------------------------------------------- roll recheck


def _hook_ranges(b_lo: float, b_hi: float, tol: float) -> dict:
    """A ``pre_run_check`` ranges dict (row bounds only, as polar-high passes them)."""
    return {
        "matrix": (1.0, 10.0), "cost": (1.0, 10.0), "col_bound": None, "row_bound": (b_lo, b_hi),
        "col_bound_meaningful": None, "row_bound_meaningful": (b_lo, b_hi),
        "bound_rhs_residue": (0, 0.0), "bound_rhs_beyond_infinite": 0, "residue_floor": 10 * tol,
        "infinite_bound": 1e20, "matrix_continuous": (1.0, 10.0), "integer_matrix": None,
        "cost_continuous": (1.0, 10.0), "integer_cost": None, "integer_bound": None,
    }


def _first_roll_plan(tol: float):
    roll1 = _guard.report_from_hook_ranges(_hook_ranges(1.0, 1e12, tol))
    return recommend_scaling(roll1, ScalingConfig(), primal_feasibility_tolerance=tol)


def test_roll_recheck_moves_the_exponent_when_the_cached_one_leaves_the_window(coded_log):
    tol = 1e-8
    plan = _first_roll_plan(tol)
    assert plan.user_bound_scale == -19 and plan.bound_tag.startswith("smallest-side")
    hook = _guard.RollRecheck(_problem(tol), plan=plan, config=ScalingConfig(), mode="error",
                              solve_name="s1, roll 2", base_solve="s1", layer2_applied=False)
    # Roll 1's own ranges: nothing to change.
    assert hook(_hook_ranges(1.0, 1e12, tol), None) is None
    assert hook.outcome.level == "ok" and hook.plan.user_bound_scale == -19
    # Roll 2: 1e-3 · 2^-19 = 1.91e-9 < 1e-7 → this roll gets N = -9; 15 decades warn.
    out = hook(_hook_ranges(1e-3, 1e12, tol), None)
    assert out == {"user_bound_scale": -9}
    assert hook.plan.user_bound_scale == -9 and hook.outcome.adjusted_bound_scale == -9
    assert hook.outcome.level == "warn"
    printed = coded_log()
    assert "Solve s1, roll 2: limits and bounds span 15.0 decades" in printed
    assert "('s1, roll 2') [W-0543]" in printed
    assert "This roll uses user_bound_scale -9 instead of -19" in printed


def test_roll_recheck_raises_when_no_exponent_fits():
    tol = 1e-8
    hook = _guard.RollRecheck(_problem(tol), plan=_first_roll_plan(tol), config=ScalingConfig(),
                              mode="error", solve_name="s1, roll 2", base_solve="s1",
                              layer2_applied=False)
    with pytest.raises(ScalingRangeError, match=r"Solve s1, roll 2 cannot be scaled"):
        hook(_hook_ranges(1e-6, 1e12, tol), None)


def test_roll_recheck_under_off_still_clips_silently(coded_log):
    tol = 1e-8
    hook = _guard.RollRecheck(_problem(tol), plan=_first_roll_plan(tol), config=ScalingConfig(),
                              mode="off", solve_name="s1, roll 2", base_solve="s1",
                              layer2_applied=False)
    assert hook(_hook_ranges(1e-3, 1e12, tol), None) == {"user_bound_scale": -9}
    assert hook.outcome.level == "off"
    assert coded_log() == ""


def test_roll_recheck_keeps_a_manual_exponent_and_judges_it():
    tol = 1e-7
    cfg = ScalingConfig(user_bound_scale=-10)
    plan = recommend_scaling(_guard.report_from_hook_ranges(_hook_ranges(1.0, 1e3, tol)), cfg,
                             primal_feasibility_tolerance=tol)
    hook = _guard.RollRecheck(_problem(tol), plan=plan, config=cfg, mode="error",
                              solve_name="s1, roll 2", base_solve="s1", layer2_applied=False)
    with pytest.raises(ScalingRangeError, match="user_bound_scale -10"):
        hook(_hook_ranges(1e-4, 1e3, tol), None)
    assert hook.plan.user_bound_scale == -10


# --------------------------------------------------------------------- solver dispatch


def _tiny_lp(b_lo: float, b_hi: float) -> Problem:
    import polars as pl
    from polar_high import Param

    pb = Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=10.0)
    pb.add_cstr("cap", over=pl.DataFrame({"i": [0]}), sense="<=", lhs_terms={"x": x},
                rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [0], "value": [b_hi]}))})
    pb.add_cstr("need", over=pl.DataFrame({"i": [1]}), sense=">=", lhs_terms={"x": x},
                rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [1], "value": [b_lo]}))})
    pb.set_objective(Param(("i",), pl.DataFrame({"i": [0, 1], "value": [1.0, 1.0]})) * x, sense="min")
    pb.set_solver_options({"output_flag": False, "primal_feasibility_tolerance": 1e-8})
    return pb


def test_in_process_dispatch_runs_the_roll_recheck():
    """``run_one_solve`` (in-process HiGHS) hands the recheck to ``Problem.solve``: roll 2's ranges
    (1e-3 … 1e12) move the cached -19 to -9 and HiGHS solves with it."""
    from flextool.engine.input._solve_config import SolverConfig
    from flextool.engine.solve._solver_dispatch import run_one_solve

    tol = 1e-8
    pb = _tiny_lp(1e-3, 1e12)
    hook = _guard.RollRecheck(pb, plan=_first_roll_plan(tol), config=ScalingConfig(), mode="warn",
                              solve_name="s1, roll 2", base_solve="s1", layer2_applied=False)
    sol = run_one_solve(pb, SolverConfig(), pre_run_check=hook)
    assert hook.called and hook.plan.user_bound_scale == -9
    assert sol.optimal and sol.solve_diagnostics().user_bound_scale == -9


def test_in_process_dispatch_stops_before_highs_runs():
    from flextool.engine.input._solve_config import SolverConfig
    from flextool.engine.solve._solver_dispatch import run_one_solve

    tol = 1e-8
    pb = _tiny_lp(1e-6, 1e12)
    hook = _guard.RollRecheck(pb, plan=_first_roll_plan(tol), config=ScalingConfig(), mode="error",
                              solve_name="s1, roll 2", base_solve="s1", layer2_applied=False)
    with pytest.raises(ScalingRangeError, match="Solve s1, roll 2 cannot be scaled"):
        run_one_solve(pb, SolverConfig(), pre_run_check=hook)


# --------------------------------------------------------------------- message details


def test_warning_states_the_violation_of_the_smallest_and_of_every_limit():
    """B at tol 1e-7: Layer 3 aims the small end at 100·tol = 1e-5: N = ceil(log2(1e-5/1.6e-5)) = 0, and
    the large end fits (1.3e9 ≤ 1e11).  At N = 0 the solver may miss a limit by tol = 1e-7 (model units),
    1e-7 / 1.6e-5 = 0.625 % of the smallest one and less of every larger one."""
    plan, out = _guard_of(_report(1.6e-5, 1.3e9))
    assert plan.user_bound_scale == 0
    assert ("With user_bound_scale 0, the smallest limit could be violated by up to 1e-07 (model units); "
            "every limit by at most 0.625 %.") in _text(out)


def _entry(model: float, count: int, i: int) -> dict:
    return {"family": "cap", "component": "rhs", "model": model, "post": model, "dims": {"i": i},
            "count_within_factor": count, "factor": 10.0}


def _located_text(monkeypatch, located: dict) -> str:
    monkeypatch.setattr(_guard, "_locate", lambda _pb, f: located if f.group == "bound_rhs" else {})
    _plan, out = _guard_of(_report(0.5, 1e18))
    return _text(out)


def test_extremes_list_only_the_entries_within_10x(monkeypatch):
    """Five entries named at the small end, two of them within 10x: only those two are listed; the large
    end with one value within 10x reads as one line and ends with the hint."""
    text = _located_text(monkeypatch, {
        "max": [_entry(1e18, 1, 0), _entry(1e3, 1, 1)],
        "min": [_entry(0.5, 2, 10), _entry(0.6, 2, 11), _entry(9.0, 2, 12), _entry(20.0, 2, 13),
                _entry(40.0, 2, 14)],
    })
    lines = text.splitlines()
    (big,) = [ln for ln in lines if ln.startswith("  Largest:")]
    assert big == ('  Largest:  1e+18 (model units) in cap (constraint right-hand side) at i=0 — '
                   'a stand-in for "no limit"? Use inf.')
    k = next(n for n, ln in enumerate(lines) if ln.startswith("  Smallest:"))
    assert lines[k] == ("  Smallest: 0.5 (model units) in cap (constraint right-hand side), "
                        "2 values within 10x of it, e.g. at:")
    assert lines[k + 1:k + 3] == ["      i=10", "      i=11"]
    assert not lines[k + 3].startswith("      i=")


def test_a_single_value_within_10x_reads_as_one_line(monkeypatch):
    text = _located_text(monkeypatch, {"min": [_entry(0.5, 1, 10), _entry(30.0, 1, 11)]})
    (small,) = [ln for ln in text.splitlines() if ln.startswith("  Smallest:")]
    assert small == "  Smallest: 0.5 (model units) in cap (constraint right-hand side) at i=10"
    assert "i=11" not in text and "values within" not in text


def test_the_no_limit_hint_follows_a_list_on_its_own_line(monkeypatch):
    text = _located_text(monkeypatch, {"max": [_entry(1e18, 3, 0), _entry(5e17, 3, 1), _entry(2e17, 3, 2)]})
    lines = text.splitlines()
    k = next(n for n, ln in enumerate(lines) if ln.startswith("  Largest:"))
    assert lines[k] == ("  Largest:  1e+18 (model units) in cap (constraint right-hand side), "
                        "3 values within 10x of it, e.g. at:")
    assert lines[k + 1:k + 5] == ["      i=0", "      i=1", "      i=2",
                                  '      A stand-in for "no limit"? Use inf.']


def test_a_warning_layer2_narrowed_from_beyond_17_decades_says_basic_would_stop():
    """Before Layer 2: 0.5 … 1e18 = 18.3 decades (> 17, so --scaling basic stops); after it 13.9 → warn."""
    pb = _problem()
    report = _report(1.6e-5, 1.3e9)
    plan = recommend_scaling(report, ScalingConfig(), primal_feasibility_tolerance=1e-7)

    def text_with(pre):
        out = _guard.run_guard(pb, ranges=report, ranges_pre=pre, plan=plan, solve_name="s1", mode="error",
                               layer2_applied=True)
        assert out.level == "warn"
        return _text(out)

    assert ("Layer 2 narrowed the span from 18.3 decades; under --scaling basic this would stop."
            in text_with(_report(0.5, 1e18)))
    # 1e-3 … 1e12 = 15 decades before Layer 2: basic would fit them, so no such line.
    assert "--scaling basic" not in text_with(_report(1e-3, 1e12))


# --------------------------------------------------------------------- once-per-run keys


def test_a_later_error_or_new_exponent_is_not_silenced_under_warn(coded_log):
    from dataclasses import replace

    _plan, warn = _guard_of(_report(1.6e-5, 1.3e9), mode="warn")
    _plan, error = _guard_of(_report(0.5, 1e18), mode="warn")
    _guard.enforce(warn, base_solve="s1")
    _guard.enforce(error, base_solve="s1")  # same solve and group, now an error: printed
    _guard.enforce(replace(warn, adjusted_bound_scale=-9), base_solve="s1")  # a roll's own exponent
    _guard.enforce(replace(warn, adjusted_bound_scale=-9), base_solve="s1")  # the same again: silent
    _guard.enforce(warn, base_solve="s1")
    printed = coded_log()
    assert printed.count("s1: Solve s1: limits and bounds span 13.9 decades") == 2
    assert printed.count("s1: Solve s1 cannot be scaled") == 1


def test_the_adjusted_exponent_log_is_once_per_run(caplog):
    import logging

    tol = 1e-8

    def adjust():
        hook = _guard.RollRecheck(_problem(tol), plan=_first_roll_plan(tol), config=ScalingConfig(),
                                  mode="off", solve_name="s1, roll 2", base_solve="s1",
                                  layer2_applied=False)
        assert hook(_hook_ranges(1e-3, 1e12, tol), None) == {"user_bound_scale": -9}

    with caplog.at_level(logging.INFO, logger=_guard.__name__):
        adjust()
        adjust()
        _behaviour_changes.start_run(None)  # a new run logs it again
        adjust()
    logged = [r for r in caplog.records if "leaves the tolerance window" in r.getMessage()]
    assert len(logged) == 2


def test_warn_mode_report_goes_to_the_run_log_not_the_console(caplog):
    """Review M2 (owner): under ``--scaling-guard warn`` the console gets the one coded line and the run's
    ``_run.log`` the whole report (without ``--debug``); with the debug log on, the debug log has it instead."""
    import logging

    from flextool import run_record, run_warnings

    caplog.set_level(logging.INFO, logger=run_warnings.LOGGER_NAME)
    logged: list[str] = []
    run_record.set_log_only_sink(logged.append)
    try:
        _plan, out = _guard_of(_report(0.5, 1e18), mode="warn")
        _guard.enforce(out, base_solve="s1")
        console = [r.getMessage() for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
        assert console == ["Warning: limits and bounds do not fit the solver's accurate range; solved anyway "
                           "('s1') [W-0544]"]
        text = "\n".join(logged)
        assert text.startswith("W-0544 s1: Solve s1 cannot be scaled")
        for part in ("span 18.3 decades", "the smallest limit could be violated by up to 1.68"):
            assert part in text, part
        logged.clear()
        caplog.set_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME)
        _behaviour_changes.start_run(None)
        _guard.enforce(out, base_solve="s1")
        assert logged == []
    finally:
        run_record.set_log_only_sink(None)
