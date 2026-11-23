"""The realization-roll plan (``solve._realization_tail``): the gate, ``roll_tail_plan``, the rows each roll reads,
the plan errors and G2b.

The plan is not wired yet (``_run_model._roll_tail_plans`` returns ``{}``), so these tests call the builder directly.
Every expected value is derived by hand from the roll windows below (comment at each test): a roll realizes its
jump ``[s_k, s_{k+1})`` and looks ahead from ``s_{k+1}``, the first step of its horizon after the jump, with the
forecasts made there (weight > 0, not realized, in database order).  The jump reads the one realized branch
flattened into one timeline: its analysis times nearest to the roll start s_k first, the earlier one on a tie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_config import SolveConfig
from flextool.engine.input._solve_state import ActiveTimeEntry, FlexToolConfigError
from flextool.engine.preprocess.emit._solve_writers import derive_solve_branch_weight
from flextool.engine.solve import _realization_tail as rt
from flextool.engine.solve._realization_tail import RollTail, RollWindow

S = "s"


def _cfg(rows, *, method="within_solve", mode="rolling_window", model_solve=None, **extra) -> SolveConfig:
    """A bare SolveConfig with the fields the gate reads; solve ``s`` realizes period ``p1``."""
    sc = object.__new__(SolveConfig)
    sc.model_solve = model_solve if model_solve is not None else {"flextool": [S]}
    sc.solve_modes = {S: mode}
    sc.stochastic_branches = {S: rows}
    sc.dispatch_realization_method = {} if method is None else {S: method}
    sc.stochastic_invest_method = {}
    sc.contains_solves = {}
    sc.realized_dispatch_parent = {}
    sc.derived_solves = {}
    sc.realized_periods = {S: [(S, "p1")]}
    sc.timesets_used_by_solves = {S: [("p1", "ts")]}
    sc.rolling_times = {S: [2, 4, -1]}
    for key, value in extra.items():
        setattr(sc, key, value)
    return sc


def _steps(cells):
    out: dict = {}
    for i, (p, t) in enumerate(cells):
        out.setdefault(p, []).append(ActiveTimeEntry(t, i, "1"))
    return out


def _rolls(cells, jump, horizon, *, solve=S, name=None):
    """The rolls of *cells* as ``create_rolling_solves`` makes them (1 h steps): roll k starts at step k * jump,
    holds ``horizon`` steps (fewer at the end) and realizes ``jump`` steps (the rest at the end)."""
    out = []
    for k, start in enumerate(range(0, len(cells), jump)):
        out.append(RollWindow((name or f"{solve}_roll_{{k}}").format(k=k), solve,
                              _steps(cells[start:start + horizon]), _steps(cells[start:start + jump])))
    return out


def _p1(n):
    return [("p1", f"t{i}") for i in range(1, n + 1)]


def _row(branch, step, weight=1.0, realized="no", period="p1"):
    return (period, branch, step, realized, weight)


# t1..t6, realized R and forecasts A (0.25), B (0.75) made at t1, t3, t5, a zero-weight Z at t3.
ROWS = [_row("R", t, realized="yes") for t in ("t1", "t3", "t5")] \
    + [_row("A", t, 0.25) for t in ("t1", "t3", "t5")] \
    + [_row("B", t, 0.75) for t in ("t1", "t3", "t5")] + [_row("Z", "t3", 0.0)]


def test_within_solve_rolling():
    # Jump 2, horizon 4 on t1..t6: roll 0 = t1-t4, jump t1-t2, tail from t3; roll 1 = t3-t6, jump t3-t4, tail from
    # t5; roll 2 = t5-t6 = its jump (the last roll): no tail.  Every tail starts inside p1: split.  R is made at t1,
    # t3, t5 (positions 0, 2, 4 on t1..t6): roll 0 (s = t1) has distances 0 / 2 / 4 -> t1, t3, t5; roll 1 (s = t3)
    # 2 / 0 / 2 -> t3, then the tie t1 (earlier) before t5; roll 2 (s = t5) 4 / 2 / 0 -> t5, t3, t1.  The last roll
    # keeps its realized branch.
    plan = rt.roll_tail_plan(_cfg(ROWS), _rolls(_p1(6), 2, 4), _p1(6))
    assert plan == {
        "s_roll_0": RollTail(("A", "B"), "split", False, (("p1", "t1"), ("p1", "t2")), ("p1", "t3"),
                             "R", ("t1", "t3", "t5")),
        "s_roll_1": RollTail(("A", "B"), "split", False, (("p1", "t3"), ("p1", "t4")), ("p1", "t5"),
                             "R", ("t3", "t1", "t5")),
        "s_roll_2": RollTail((), "deterministic", True, (("p1", "t5"), ("p1", "t6")), None,
                             "R", ("t5", "t3", "t1")),
    }


def test_roll_branch_rows_are_the_authored_rows_at_the_jump_end():
    # Roll 0 reads the forecast rows of its cohort made at t3, in database order (A, B; Z has weight 0; R is
    # realized, read through the plan), not re-dated; the last roll reads none.
    cfg = _cfg(ROWS)
    rolls = {S: _rolls(_p1(6), 2, 4)}
    plan = rt.roll_tail_plan(cfg, rolls[S], _p1(6))
    rows = rt.roll_branch_rows(cfg, rolls, plan)
    assert rows == {
        "s_roll_0": [_row("A", "t3", 0.25), _row("B", "t3", 0.75)],
        "s_roll_1": [_row("A", "t5", 0.25), _row("B", "t5", 0.75)],
        "s_roll_2": [],
    }
    assert cfg.stochastic_branches[S] == ROWS  # never changed


def test_separate_dispatch_rolling_plans_the_realization_rolls():
    # The windows are those of the stochastic rolls, named by the realization solve's rolls; the same tails as
    # test_within_solve_rolling.
    sib = f"{S}__realized_dispatch"
    cfg = _cfg(ROWS, method="separate_dispatch")
    plan = rt.roll_tail_plan(cfg, _rolls(_p1(6), 2, 4, name=sib + "_roll_{k}"), _p1(6))
    assert [(n, t.cohort, t.shape, t.tail_start) for n, t in plan.items()] == [
        (sib + "_roll_0", ("A", "B"), "split", ("p1", "t3")),
        (sib + "_roll_1", ("A", "B"), "split", ("p1", "t5")),
        (sib + "_roll_2", (), "deterministic", None),
    ]


def test_separate_dispatch_needs_forecasts_at_the_roll_start():
    # Forecasts made at t3 and t5 only: solve (1) of roll 0 branches at its first step t1, where none is made.
    rows = [r for r in ROWS if r[2] != "t1"]
    with pytest.raises(FlexToolConfigError, match=r"Solve 's', roll 0 \(time steps t1 to t2\): no forecast .* "
                                                  r"made at t1, the start of the roll.*period 'p1'"):
        rt.roll_tail_plan(_cfg(rows, method="separate_dispatch"), _rolls(_p1(6), 2, 4), _p1(6))
    # within_solve reads no forecast at the roll start: the same rows plan.
    assert rt.roll_tail_plan(_cfg(rows), _rolls(_p1(6), 2, 4), _p1(6))["s_roll_0"].cohort == ("A", "B")


def test_jump_of_one_step():
    # Jump 1, horizon 2 on t1..t3: roll 0 = t1-t2 (jump t1, tail t2), roll 1 = t2-t3 (jump t2, tail t3), roll 2 = t3
    # (no tail).  Forecasts and R at every step: realized times by distance from t1 / t2 / t3 (t1 before t3 on the
    # tie of roll 1).
    rows = [_row("R", t, realized="yes") for t in ("t1", "t2", "t3")] + [_row("A", t, 0.5) for t in ("t1", "t2", "t3")] \
        + [_row("B", t, 0.5) for t in ("t1", "t2", "t3")]
    plan = rt.roll_tail_plan(_cfg(rows), _rolls(_p1(3), 1, 2), _p1(3))
    assert plan == {
        "s_roll_0": RollTail(("A", "B"), "split", False, (("p1", "t1"),), ("p1", "t2"), "R", ("t1", "t2", "t3")),
        "s_roll_1": RollTail(("A", "B"), "split", False, (("p1", "t2"),), ("p1", "t3"), "R", ("t2", "t1", "t3")),
        "s_roll_2": RollTail((), "deterministic", True, (("p1", "t3"),), None, "R", ("t3", "t2", "t1")),
    }


def test_seam_and_multi_period_jump():
    # p1 = t1-t2, p2 = t3-t6.  Jump 2: roll 0 ends with p1, so s_1 = t3 is p2's first step: seam.  Jump 3: roll 0
    # holds p1 and t3 of p2, so s_1 = t4 lies inside p2: split (only the period of s_1 splits).
    cells = [("p1", "t1"), ("p1", "t2")] + [("p2", f"t{i}") for i in range(3, 7)]
    rows = [_row(b, t, w, period="p2") for t in ("t3", "t4") for b, w in (("A", 0.5), ("B", 0.5))] \
        + [_row("R", "t1", realized="yes")]
    cfg = _cfg(rows)
    seam = rt.roll_tail_plan(cfg, _rolls(cells, 2, 4)[:1], cells)["s_roll_0"]
    assert (seam.shape, seam.tail_start, seam.cohort) == ("seam", ("p2", "t3"), ("A", "B"))
    split = rt.roll_tail_plan(cfg, _rolls(cells, 3, 4)[:1], cells)["s_roll_0"]
    assert (split.shape, split.tail_start, split.jump_steps) == (
        "split", ("p2", "t4"), (("p1", "t1"), ("p1", "t2"), ("p2", "t3")))


def test_one_forecast_is_a_cohort_of_one():
    rows = [_row("R", "t1", realized="yes"), _row("A", "t3", 1.0), _row("R", "t3", realized="yes")]
    plan = rt.roll_tail_plan(_cfg(rows), _rolls(_p1(4), 2, 4), _p1(4))
    assert plan["s_roll_0"].cohort == ("A",)


def test_no_forecast_at_the_jump_end():
    # T-w3: forecasts made at t1 and t3 only, jump 1, horizon 3: roll 0 (jump t1) looks ahead from t2, where no
    # forecast is made.
    rows = [_row("R", t, realized="yes") for t in ("t1", "t3")] + [_row("A", t, 0.5) for t in ("t1", "t3")] \
        + [_row("B", t, 0.5) for t in ("t1", "t3")]
    with pytest.raises(FlexToolConfigError, match=r"Solve 's', roll 0 \(time steps t1 to t1\): no forecast .* "
                                                  r"made at t2, the end of its jump.*analysis_time t2 for "
                                                  r"period 'p1'"):
        rt.roll_tail_plan(_cfg(rows), _rolls(_p1(3), 1, 3), _p1(3))


def test_zero_weight_and_realized_rows_are_no_forecast():
    # At t3 only R (realized) and Z (weight 0): no forecast is made there.
    rows = [_row("R", "t1", realized="yes"), _row("R", "t3", realized="yes"), _row("Z", "t3", 0.0)]
    with pytest.raises(FlexToolConfigError, match="made at t3, the end of its jump"):
        rt.roll_tail_plan(_cfg(rows), _rolls(_p1(4), 2, 4), _p1(4))


@pytest.mark.parametrize("method", ["within_solve", "separate_dispatch"])
def test_rolling_solve_needs_a_realized_branch(method):
    # W3, both methods: forecasts only (made at every roll start, so separate_dispatch's roll-start check would pass).
    rows = [r for r in ROWS if r[3] != "yes"]
    with pytest.raises(FlexToolConfigError, match=f"Solve 's' is a rolling stochastic solve with "
                                                  f"dispatch_realization_method {method}, but its stochastic_branches "
                                                  "have no branch with realized: yes"):
        rt.roll_tail_plan(_cfg(rows, method=method), _rolls(_p1(6), 2, 4), _p1(6))


def test_realized_series_authored_once():
    # D7: R made once at t1 covers the whole timeline; forecasts A / B at t1, t3, t5.  Every roll (also the last)
    # reads R through the one tier t1; no row a roll reads is a realized row.
    rows = [_row("R", "t1", realized="yes")] + [r for r in ROWS if r[3] != "yes"]
    cfg = _cfg(rows)
    rolls = {S: _rolls(_p1(6), 2, 4)}
    plan = rt.roll_tail_plan(cfg, rolls[S], _p1(6))
    assert [(n, t.tail_start, t.realized, t.realized_times) for n, t in plan.items()] == [
        ("s_roll_0", ("p1", "t3"), "R", ("t1",)),
        ("s_roll_1", ("p1", "t5"), "R", ("t1",)),
        ("s_roll_2", None, "R", ("t1",)),
    ]
    assert all(r[3] != "yes" for rs in rt.roll_branch_rows(cfg, rolls, plan).values() for r in rs)


def test_two_realized_branches():
    # R realized at t1, Q realized at t3: two realized series cannot be read as one timeline.
    rows = [_row("R", "t1", realized="yes"), _row("Q", "t3", realized="yes")] + [r for r in ROWS if r[3] != "yes"]
    with pytest.raises(FlexToolConfigError, match=r"Solve 's': the realized: yes rows of its stochastic_branches "
                                                  r"name more than one branch \(R, Q\)"):
        rt.roll_tail_plan(_cfg(rows), _rolls(_p1(6), 2, 4), _p1(6))


@pytest.mark.parametrize("cell", [("p9", "t1"), ("p1", "t9")])
def test_realized_row_off_the_timeline(cell):
    # The only realized row is made at a period, or a step, that is not on the timeline t1..t6 of p1.
    rows = [_row("R", cell[1], realized="yes", period=cell[0])] + [r for r in ROWS if r[3] != "yes"]
    with pytest.raises(FlexToolConfigError, match="Solve 's' is a rolling stochastic solve with "
                                                  "dispatch_realization_method within_solve, but no realized: yes "
                                                  "row .* starts at a time step of the timeline of its periods"):
        rt.roll_tail_plan(_cfg(rows), _rolls(_p1(6), 2, 4), _p1(6))


def test_error_names_the_solve_not_its_2d_map_copy():
    rows = [r for r in ROWS if r[3] != "yes"]
    cfg = _cfg(rows, map2d_duplicate_of={S: ("authored", "x")})
    with pytest.raises(FlexToolConfigError, match="Solve 'authored' is a rolling"):
        rt.roll_tail_plan(cfg, _rolls(_p1(6), 2, 4), _p1(6))


def test_jump_not_at_the_start_of_the_horizon_is_an_internal_error():
    # Roll 0's horizon t1-t4 with the jump t2-t3: not a prefix of the horizon.
    roll = RollWindow("s_roll_0", S, _steps(_p1(4)), _steps(_p1(3)[1:]))
    with pytest.raises(RuntimeError, match="Internal error: the jump of roll 's_roll_0' is not the start of its "
                                           "horizon"):
        rt.roll_tail_plan(_cfg(ROWS), [roll], _p1(6))


@pytest.mark.parametrize("case", [
    "deterministic_absent", "deterministic_explicit", "single_within_solve", "single_separate_dispatch",
    "recourse", "author_declared", "nested", "contains", "no_branches",
])
def test_gate_rows_get_no_plan(case):
    # A rolling solve with forecasts and a realized branch at every step: only the gate decides.
    rows = [_row(b, f"t{i}", w, r) for i in range(1, 7) for b, w, r in (("R", 1.0, "yes"), ("A", 0.5, "no"),
                                                                      ("B", 0.5, "no"))]
    kw: dict = {}
    if case == "deterministic_absent":
        kw["method"] = None
    elif case == "deterministic_explicit":
        kw["method"] = "deterministic"
    elif case == "single_within_solve":
        kw["mode"] = "single_solve"
    elif case == "single_separate_dispatch":
        kw.update(mode="single_solve", method="separate_dispatch")
    elif case == "recourse":
        kw["stochastic_invest_method"] = {S: "recourse"}
    elif case == "nested":
        kw.update(model_solve={"flextool": ["outer"]}, contains_solves={"outer": [S]})
    elif case == "contains":
        kw["contains_solves"] = {S: ["inner"]}
    elif case == "no_branches":
        rows = []
    cfg = _cfg(rows, **kw)
    if case == "author_declared":
        # A deterministic rolling solve right after s with the same rolls that realizes p1
        # (``_realization.configured_realization_solve``).
        cfg.model_solve = {"flextool": [S, "r"]}
        cfg.solve_modes["r"] = "rolling_window"
        cfg.rolling_times["r"] = [2, 4, -1]
        cfg.realized_periods["r"] = [("r", "p1")]
    assert rt.realization_roll_method(cfg, S) is None
    assert rt.roll_tail_plan(cfg, _rolls(_p1(6), 2, 4), _p1(6)) == {}


def test_gate_engine_realization_solve():
    # The engine-added realization solve (a derived solve) is no author-declared realization: s keeps its answer,
    # and the realization solve takes it, for separate_dispatch only.
    sib = f"{S}__realized_dispatch"
    for method, sibling_answer in (("separate_dispatch", "separate_dispatch"), ("within_solve", None)):
        cfg = _cfg(ROWS, method=method, model_solve={"flextool": [S, sib]})
        cfg.solve_modes[sib] = "rolling_window"
        cfg.rolling_times[sib] = [2, 4, -1]
        cfg.realized_periods[sib] = [(sib, "p1")]
        cfg.realized_dispatch_parent = {sib: S}
        cfg.derived_solves = {sib: (S, frozenset())}
        assert rt.realization_roll_method(cfg, S) == method
        assert rt.realization_roll_method(cfg, sib) == sibling_answer
        # Not derived: an author-declared realization solve; then neither has realization rolls.
        cfg.derived_solves = {}
        assert rt.realization_roll_method(cfg, S) is None
        assert rt.realization_roll_method(cfg, sib) is None


def _weights(periods):
    return pl.DataFrame({"branch": periods, "p_branch_weight_input": ["0.5"] * len(periods)},
                        schema={"branch": pl.Utf8, "p_branch_weight_input": pl.Utf8})


def test_g2b_every_forecast_has_its_weight():
    tail = RollTail(("A", "B"), "split", False, (("p1", "t1"),), ("p1", "t2"), "R", ("t1",))
    sbtb = [("p1_A", "A"), ("p1_B", "B"), ("p1", "R")]
    rt.check_tail_weights("s_roll_0", tail, sbtb, _weights(["p1_A", "p1_B", "p1"]))
    with pytest.raises(RuntimeError, match="Internal error: solve_branch_weight on period 'p1_B' of roll "
                                           "'s_roll_0' is not path-aware"):
        rt.check_tail_weights("s_roll_0", tail, sbtb, _weights(["p1_A", "p1"]))
    # A forecast without any branch period reads no weight either.
    with pytest.raises(RuntimeError, match="period 'p1_B'"):
        rt.check_tail_weights("s_roll_0", tail, sbtb[:1], _weights(["p1_A"]))
    # A deterministic roll has no forecast to weigh.
    rt.check_tail_weights("s_roll_1", RollTail((), "deterministic", True, (("p1", "t2"),), None, "R", ("t1",)), [],
                          _weights([]))


def test_branch_weight_reader_takes_the_roll_rows():
    # Fallback (no plan rows): the complete solve's rows made at the fan start; with plan rows: those rows.
    active = {"p1_A": [ActiveTimeEntry("t3", 0, "1")], "p1_B": [ActiveTimeEntry("t3", 0, "1")]}
    sbtb = [("p1_A", "A"), ("p1_B", "B")]
    pb = [("p1", "p1"), ("p1", "p1_A"), ("p1", "p1_B")]
    authored = {S: [_row("A", "t3", 0.25), _row("B", "t3", 0.75)]}
    today = derive_solve_branch_weight(S, active, sbtb, ("p1", "t3"), pb, authored)
    assert today.rows() == [("p1_A", "0.25"), ("p1_B", "0.75")]
    planned = derive_solve_branch_weight(S, active, sbtb, ("p1", "t3"), pb, authored,
                                         [_row("A", "t3", 0.4), _row("B", "t3", 0.6)])
    assert planned.rows() == [("p1_A", "0.4"), ("p1_B", "0.6")]
    cfg = _cfg(ROWS)
    cfg.roll_branch_rows = {"s_roll_0": []}
    assert rt.plan_branch_rows(cfg, "s_roll_0") == []
    assert rt.plan_branch_rows(cfg, "s_roll_1") is None


# ---------------------------------------------------------------------------
# The windows of a real run (the seam's input), stopped before the stochastic expansion.
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class _Stop(BaseException):
    """Ends the run at the seam (not an Exception, so no handler of the run takes it)."""


def _seam_input(tmp_path, monkeypatch, fixture: str, scenario: str):
    tests_dir = str(Path(__file__).resolve().parents[2])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from db_utils import json_to_db
    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _run_model
    from flextool.update_flextool.db_migration import migrate_database

    url = json_to_db(FIXTURES / fixture, tmp_path / "db.sqlite")
    migrate_database(url)
    seen: dict = {}

    def capture(solve_cfg, rolls, timelines):
        assert _run_model.__dict__["_roll_tail_plans"] is capture
        seen["cfg"], seen["rolls"], seen["timelines"] = solve_cfg, rolls, timelines
        raise _Stop

    default = _run_model._roll_tail_plans
    monkeypatch.setattr(_run_model, "_roll_tail_plans", capture)
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(_Stop):
        run_chain_from_db(url, scenario, work_folder=work, warm=False)
    assert default(seen["cfg"], seen["rolls"], seen["timelines"]) == {}  # not wired
    return seen["cfg"], seen["rolls"], seen["timelines"]


def _hours(start, end):
    return tuple(("period1", f"t{i:04d}") for i in range(start, end + 1))


def test_seam_windows_rolling_separate_dispatch(tmp_path, monkeypatch):
    # 72 h, jump 24, horizon 48: rolls start at t0001, t0025, t0049; the realization solve's roll k gets the window
    # of stochastic roll k (its own is cut to the jump).  Forecasts upper / lower / mid (DB order) at each roll
    # start: rolls 0-1 look ahead from t0025 / t0049, roll 2 (t0049-t0072) has no tail.  "realized" is made at
    # t0001 / t0025 / t0049; by distance from the UC rolls' starts t0001 / t0025 / t0049 (24 h apart, the earlier
    # one on a tie): (t0001, t0025, t0049), (t0025, t0001, t0049), (t0049, t0025, t0001).
    cfg, rolls, timelines = _seam_input(tmp_path, monkeypatch, "rolling_separate_dispatch.json",
                                        "rolling_separate_dispatch")
    assert list(rolls) == list(timelines) == ["1week_rolling"]
    sib = "1week_rolling__realized_dispatch_roll_{}"
    assert [w.name for w in rolls["1week_rolling"]] == [sib.format(k) for k in range(3)]
    plan = rt.roll_tail_plan(cfg, rolls["1week_rolling"], timelines["1week_rolling"])
    assert plan == {
        sib.format(0): RollTail(("upper", "lower", "mid"), "split", False, _hours(1, 24), ("period1", "t0025"),
                                "realized", ("t0001", "t0025", "t0049")),
        sib.format(1): RollTail(("upper", "lower", "mid"), "split", False, _hours(25, 48), ("period1", "t0049"),
                                "realized", ("t0025", "t0001", "t0049")),
        sib.format(2): RollTail((), "deterministic", True, _hours(49, 72), None,
                                "realized", ("t0049", "t0025", "t0001")),
    }
    rows = rt.roll_branch_rows(cfg, rolls, plan)[sib.format(0)]
    assert [(r[1], r[2], r[3]) for r in rows] == [
        ("upper", "t0025", "no"), ("lower", "t0025", "no"), ("mid", "t0025", "no")]


def test_seam_windows_rolling_within_solve(tmp_path, monkeypatch):
    # 168 h, jump 24, horizon 96: 7 rolls starting at t0001 + 24k; roll k looks ahead from t0025 + 24k for
    # k = 0..5 (roll 3 ends at t0073 + 95 h = t0168; the horizon is cut at t0168 from roll 4 on, still past the jump);
    # roll 6 (t0145-t0168) has no tail.  The spawned realization solve of today has no realization rolls.  "realized"
    # is made at t0001 + 24j (j = 0..6): roll 3 (s = t0073) reads t0073, then t0049 / t0097 (24 h), t0025 / t0121
    # (48 h), t0001 / t0145 (72 h), the earlier one first on each tie.
    cfg, rolls, timelines = _seam_input(tmp_path, monkeypatch, "stochastics.json", "1_week_rolling_wind")
    assert list(rolls) == ["1week_rolling"]
    assert rt.realization_roll_method(cfg, "1week_rolling__realized_dispatch") is None
    plan = rt.roll_tail_plan(cfg, rolls["1week_rolling"], timelines["1week_rolling"])
    assert {t.realized for t in plan.values()} == {"realized"}
    assert plan["1week_rolling_roll_3"].realized_times == (
        "t0073", "t0049", "t0097", "t0025", "t0121", "t0001", "t0145")
    expected = {f"1week_rolling_roll_{k}": (("upper", "lower", "mid"), "split", False,
                                            _hours(1 + 24 * k, 24 + 24 * k), ("period1", f"t{25 + 24 * k:04d}"))
                for k in range(6)}
    expected["1week_rolling_roll_6"] = ((), "deterministic", True, _hours(145, 168), None)
    assert {n: (t.cohort, t.shape, t.deterministic, t.jump_steps, t.tail_start) for n, t in plan.items()} == expected
