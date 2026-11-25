"""The realization roll wired in (design rev 7.1 §2.6, §8.2 T-w1 / T-w3 / T-w4, §8.3 jump = 1): no test seam.

A rolling ``within_solve`` solve is one solve per roll (no realization solve is added); a rolling
``separate_dispatch`` solve is solve (1) then the realization roll, which looks ahead after its jump with the
forecasts made at the jump end.  Models: ``test_split_roll_lp._tw1`` (T-w1) and ``test_trunk_sites._spec``.
Every expected value is derived by hand in the docstrings.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as srl
from tests.engine.realization_roll import test_trunk_sites as ts

pack = scm.pack


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _set_solve(spec: dict, name: str, value) -> dict:
    for row in spec["parameter_values"]:
        if row[:3] == ["solve", "st", name]:
            row[3] = value
            return spec
    spec["parameter_values"].append(["solve", "st", name, value, "init"])
    return spec


def _state(step) -> float:
    return srl._value(step.solution.value("v_state"), n="elec", d="p1", t="t01")


@pytest.mark.parametrize(("w_b", "state", "objectives"), [
    (0.5, 1.0, [87_600.0, 0.0]),
    (0.25, 0.0, [76_650.0, 131_400.0])])
def test_within_solve_is_one_solve_per_roll(tmp_path, w_b, state, objectives):
    """T-w1 (jump 1, two rolls): 2 solves, no realization solve (today 4).  Factor 8760 / 2 = 4380 per step.

    R_0 = p1 t01 on R (wind 0) + p1_A / p1_B t02 (forecasts made at t02: A wind 1, B wind 0, weights 1 - w_B, w_B).
    gas covers t01 at 10 and can charge 1 MWh more (2 MW); storing saves 30 at t02 only under B.  w_B 0.5: 15 > 10,
    the realized storage at the end of t01 is 1, cost 20 x 4380 = 87 600; R_1 (t02, deterministic on R) discharges:
    0.  w_B 0.25: 7.5 < 10, storage 0, cost (10 + 0.25 x 30) x 4380 = 76 650; R_1 buys 1 MWh at 30: 131 400.
    No price is asserted (gas is capped: the t01 price lies anywhere in [10, 15])."""
    built, steps = scm.run_spec(srl._tw1(w_b), tmp_path)
    names = [s for s, _, _ in built]
    assert len(names) == 2 and not any("realized_dispatch" in s for s in names), names
    first = list(steps.values())[0]
    assert _state(first) == pytest.approx(state, abs=1e-9)
    assert srl._objectives(steps) == pytest.approx(objectives, rel=1e-9, abs=1e-6)


def test_separate_dispatch_realization_looks_ahead_jump_one(tmp_path):
    """T-w1 as ``separate_dispatch`` (w_B 0.5), jump 1, with forecasts also made at t01 (A wind 0 at t01, 1 at t02;
    B wind 0 at both; weights 0.5 / 0.5) for solve (1) of roll 0.  4 solves: (1) and the realization roll per roll.

    Solve (1) roll 0: both forecasts from t01, storage per forecast (no unit online, so nothing is tied): A buys 1
    MWh at 10; B also stores 1 MWh at 10 for t02 (instead of 30): 0.5 x 10 + 0.5 x 20 = 15 -> 65 700.  Realization
    roll 0: no _fix_in_uc unit, so nothing is pinned: R_0 of T-w1, 87 600, realized storage 1.  Solve (1) roll 1
    (t02, from the realized storage 1): discharge, 0.  Realization roll 1 (t02, deterministic on R): 0."""
    spec = srl._tw1(0.5)
    _set_solve(spec, "dispatch_realization_method", pack("separate_dispatch", "str"))
    _set_solve(spec, "stochastic_branches", srl._fan([
        ("A", {"t01": 0.5, "t02": 0.5}, "no"), ("B", {"t01": 0.5, "t02": 0.5}, "no"),
        ("R", {"t01": 0.0, "t02": 0.0}, "yes")]))
    for row in spec["parameter_values"]:
        if row[:3] == ["profile", "wp", "profile"]:
            row[3] = srl._series({"A": {"t01": {"t01": 0.0, "t02": 1.0}, "t02": {"t02": 1.0}},
                                  "B": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}},
                                  "R": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}}})
    built, steps = scm.run_spec(spec, tmp_path)
    names = [s for s, _, _ in built]
    assert len(names) == 4 and ["realized_dispatch" in s for s in names] == [False, True, False, True], names
    assert srl._objectives(steps) == pytest.approx([65_700.0, 87_600.0, 0.0, 0.0], rel=1e-9, abs=1e-6)
    assert _state(list(steps.values())[1]) == pytest.approx(1.0, abs=1e-9)


def test_a_jump_ending_without_forecasts_stops(tmp_path):
    """T-w3: test_trunk_sites' model, jump 1 / horizon 2, forecasts made at t01 and t03 only: roll 0 (t01-t02)
    ends its jump at t02, where no forecast is made."""
    spec = ts._spec(real_at=ts.OVERLAP, forecast_at={"t01": ts.STEPS, "t03": ts.STEPS[2:]}, rolling=(1.0, 2.0))
    with pytest.raises(FlexToolConfigError, match=r"Solve 'st', roll 0 \(time steps t01 to t01\): no forecast of "
                                                  r"stochastic_branches is made at t02, the end of its jump"):
        scm.run_spec(spec, tmp_path)


def test_within_solve_without_a_realized_branch_stops(tmp_path):
    """T-w4 (W3): T-w1 without its ``realized: yes`` row."""
    spec = _set_solve(srl._tw1(0.5), "stochastic_branches",
                      srl._fan([("A", {"t02": 0.5}, "no"), ("B", {"t02": 0.5}, "no")]))
    with pytest.raises(FlexToolConfigError, match=r"Solve 'st' is a rolling stochastic solve with "
                                                  r"dispatch_realization_method within_solve, but its "
                                                  r"stochastic_branches have no branch with realized: yes"):
        scm.run_spec(spec, tmp_path)


# ---------------------------------------------------------------------------
# W-0635 / W-0636 (design rev 7.1 §6.2): announced for a listed solve that runs with the new realization, only
# after the checks of §6.4 have passed; a stopped solve never reaches the notice.
# ---------------------------------------------------------------------------

def _listed(monkeypatch, solve: str = "st") -> None:
    """The database carries both markers (v83 part 3) listing *solve*."""
    from flextool.engine import _behaviour_changes

    marker = '{"since": 83, "solves": ["%s"]}' % solve
    markers = {"rolling_within_solve_one_solve": marker, "separate_dispatch_realization_looks_ahead": marker}
    monkeypatch.setattr(_behaviour_changes, "read_markers_from_url", lambda url: dict(markers))


def _separate(spec: dict, forecasts_at_t01: bool = True) -> dict:
    """*spec* (T-w1) as ``separate_dispatch``; with *forecasts_at_t01* also the forecasts made at t01 that solve (1)
    of roll 0 branches into (``test_separate_dispatch_realization_looks_ahead_jump_one``)."""
    _set_solve(spec, "dispatch_realization_method", pack("separate_dispatch", "str"))
    if forecasts_at_t01:
        _set_solve(spec, "stochastic_branches", srl._fan([
            ("A", {"t01": 0.5, "t02": 0.5}, "no"), ("B", {"t01": 0.5, "t02": 0.5}, "no"),
            ("R", {"t01": 0.0, "t02": 0.0}, "yes")]))
        for row in spec["parameter_values"]:
            if row[:3] == ["profile", "wp", "profile"]:
                row[3] = srl._series({"A": {"t01": {"t01": 0.0, "t02": 1.0}, "t02": {"t02": 1.0}},
                                      "B": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}},
                                      "R": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}}})
    return spec


def test_listed_solves_get_their_notice(tmp_path, monkeypatch):
    """T-w1 ``within_solve`` runs (2 rolls): W-0635 names 'st' once, no W-0636.  As ``separate_dispatch`` (forecasts
    made at t01 and t02): W-0636 names 'st', no W-0635."""
    from tests._support.coded_warnings import capture_coded

    _listed(monkeypatch)
    with capture_coded() as got:
        scm.run_spec(srl._tw1(0.5), tmp_path / "within")
    assert got.names("W-0635") == ["st"] and not got.names("W-0636")
    with capture_coded() as got:
        scm.run_spec(_separate(srl._tw1(0.5)), tmp_path / "separate")
    assert got.names("W-0636") == ["st"] and not got.names("W-0635")


def test_a_stopped_solve_gets_no_notice(tmp_path, monkeypatch):
    """The listed solve stops with a §6.4 error before it runs, so neither notice is announced: ``within_solve``
    without its ``realized: yes`` row (W3); ``separate_dispatch`` without forecasts made at roll 0's start t01
    (T-w1's fan is made at t02 only); ``separate_dispatch`` with ``fix_storage_periods`` (its realization solve
    would re-float the pinned storage)."""
    from tests._support.coded_warnings import capture_coded

    _listed(monkeypatch)
    no_realized = _set_solve(srl._tw1(0.5), "stochastic_branches",
                             srl._fan([("A", {"t02": 0.5}, "no"), ("B", {"t02": 0.5}, "no")]))
    fix_storage = _set_solve(_separate(srl._tw1(0.5)), "fix_storage_periods", scm.array(["p1"]))
    cases = [
        (no_realized, r"^Solve 'st' is a rolling stochastic solve with dispatch_realization_method within_solve, "
                      r"but its stochastic_branches have no branch with realized: yes"),
        (_separate(srl._tw1(0.5), forecasts_at_t01=False),
         r"^Solve 'st', roll 0 \(time steps t01 to t01\): no forecast of stochastic_branches is made at t01, the "
         r"start of the roll"),
        (fix_storage, r"^Solve 'st' needs an engine-added realization solve .* but also authors fix_storage_periods"),
    ]
    for i, (spec, error) in enumerate(cases):
        with capture_coded() as got, pytest.raises(FlexToolConfigError, match=error):
            scm.run_spec(spec, tmp_path / str(i))
        assert not got.names("W-0635") and not got.names("W-0636"), i


def test_rolling_separate_dispatch_with_delays_gets_the_notice_without_look_ahead(tmp_path, monkeypatch, coded_log):
    """A rolling ``separate_dispatch`` solve in a model with a delayed unit keeps today's realization (D12: no
    realization roll, the jump is realized without look-ahead): note W-0594 names the unit.  Its solve (1) changed
    (it shares only the ``_fix_in_uc`` commitment), so W-0636 names the solve, and its full notice says that this
    solve's realization dispatches each jump alone.  test_trunk_sites' model (jump 2, horizon 4, forecasts and the
    realized row made at t01, t03, t05, the roll starts) with ``gen`` fed from ``fuel`` with a 1 h delay, as
    ``test_split_roll_lp.test_a_delay_keeps_todays_realization``."""
    from tests._support.coded_warnings import capture_coded

    _listed(monkeypatch)
    spec = ts._spec(real_at=ts.OVERLAP, forecast_at=ts.OVERLAP)
    spec["entities"] += [["node", "fuel", None], ["unit__inputNode", ["gen", "fuel"], None]]
    spec["entity_alternatives"].append(["node", ["fuel"], "init", True])
    spec["parameter_values"] += [*srl._node("fuel", "balance", pack(100.0, "float")),
                                 ["unit", "gen", "delay", pack(1.0, "float"), "init"]]
    _set_solve(spec, "dispatch_realization_method", pack("separate_dispatch", "str"))
    coded_log.notices()
    with capture_coded() as got:
        built, _steps = scm.run_spec(spec, tmp_path)
    assert any("__realized_dispatch" in s for s, _, _ in built)
    assert got.names("W-0594") == ["gen"] and got.names("W-0636") == ["st"] and not got.names("W-0635")
    (text,) = [n for n in coded_log.notices() if "separate_dispatch" in n]
    assert ("Affected solves: st. Solve 'st' keeps the earlier realization: its realization solve dispatches each "
            "jump alone, without the look-ahead; only its first solve changed.") in text


@pytest.mark.parametrize("stop", ["fix_storage_periods", "realized_row_where_no_roll_fans_out"])
def test_a_delayed_solve_stopped_before_it_runs_gets_no_w0594(tmp_path, stop):
    """The delayed-unit case above, stopped before any roll: by ``fix_storage_periods`` (the realization solve it
    needs would re-float the pinned storage; refused while the realization solve is added) or by a ``realized: yes``
    row at t02, where no roll fans out (the solve keeps the earlier realization, so nothing reads that row: R1,
    refused by the stochastic expansion).  W-0594, which says how that realization solve dispatches, is noted with
    the realization notices after those checks, so neither stopped run gets it."""
    from tests._support.coded_warnings import capture_coded

    real_at = dict(ts.OVERLAP, t02=ts.STEPS[1:]) if stop != "fix_storage_periods" else ts.OVERLAP
    spec = ts._spec(real_at=real_at, forecast_at=ts.OVERLAP)
    spec["entities"] += [["node", "fuel", None], ["unit__inputNode", ["gen", "fuel"], None]]
    spec["entity_alternatives"].append(["node", ["fuel"], "init", True])
    spec["parameter_values"] += [*srl._node("fuel", "balance", pack(100.0, "float")),
                                 ["unit", "gen", "delay", pack(1.0, "float"), "init"]]
    _set_solve(spec, "dispatch_realization_method", pack("separate_dispatch", "str"))
    if stop == "fix_storage_periods":
        _set_solve(spec, "fix_storage_periods", scm.array(["p1"]))
        error = r"^Solve 'st' needs an engine-added realization solve .* but also authors fix_storage_periods"
    else:
        error = (r"^Roll 'st_roll_0' of rolling solve 'st' opens its stochastic fan at period 'p1' \(analysis_time "
                 r"'t01'\)\. The stochastic_branches of solve 'st' also mark a realized branch later in its horizon")
    with capture_coded() as got, pytest.raises(FlexToolConfigError, match=error):
        scm.run_spec(spec, tmp_path)
    assert not got.names("W-0594")


@pytest.mark.parametrize("markers", ["another_solve", "no_marker"])
def test_an_unlisted_solve_gets_no_notice(tmp_path, monkeypatch, markers):
    """The same two runs as :func:`test_listed_solves_get_their_notice` (``within_solve``, then ``separate_dispatch``
    with forecasts made at t01 and t02), but the database's markers list only a solve 'other' (or the database
    carries no marker: it was made at 83, nothing was upgraded): 'st' runs with the new realization in both, yet is
    in no list, so neither W-0635 nor W-0636 is announced."""
    from flextool.engine import _behaviour_changes
    from tests._support.coded_warnings import capture_coded

    if markers == "another_solve":
        _listed(monkeypatch, solve="other")
    else:
        monkeypatch.setattr(_behaviour_changes, "read_markers_from_url", lambda url: {})
    for name, spec in (("within", srl._tw1(0.5)), ("separate", _separate(srl._tw1(0.5)))):
        with capture_coded() as got:
            _built, steps = scm.run_spec(spec, tmp_path / name)
        assert steps and all(s.solution.optimal for s in steps.values()), name
        assert not got.names("W-0635") and not got.names("W-0636"), name


def test_listed_rolling_within_solve_with_delays_gets_w0594_not_w0635(tmp_path, monkeypatch):
    """test_trunk_sites' model (``within_solve``, jump 2, horizon 4, forecasts and the realized row made at t01, t03,
    t05, the roll starts; listed in both markers) with ``gen`` fed from ``fuel`` with a 1 h delay, as
    ``test_split_roll_lp.test_a_delay_keeps_todays_realization``: D12 gives no solve a realization roll in a model
    with delayed units, so 'st' keeps the earlier realization (a realization solve after each roll, as before 4.7.0)
    and its realization did not change: no W-0635.  It is no ``separate_dispatch`` solve: no W-0636.  W-0594 names
    the delayed unit ``gen`` instead."""
    from tests._support.coded_warnings import capture_coded

    _listed(monkeypatch)
    spec = ts._spec(real_at=ts.OVERLAP, forecast_at=ts.OVERLAP)
    spec["entities"] += [["node", "fuel", None], ["unit__inputNode", ["gen", "fuel"], None]]
    spec["entity_alternatives"].append(["node", ["fuel"], "init", True])
    spec["parameter_values"] += [*srl._node("fuel", "balance", pack(100.0, "float")),
                                 ["unit", "gen", "delay", pack(1.0, "float"), "init"]]
    with capture_coded() as got:
        built, _steps = scm.run_spec(spec, tmp_path)
    assert any("__realized_dispatch" in s for s, _, _ in built)
    assert got.names("W-0594") == ["gen"]
    assert not got.names("W-0635") and not got.names("W-0636")


def test_listed_single_separate_dispatch_gets_w0636(tmp_path, monkeypatch, coded_log):
    """T-sd2 (``test_solve_shapes``): a single ``separate_dispatch`` solve, listed.  Its solve (1) changed (D3: only
    the ``_fix_in_uc`` commitment is shared), so W-0636 names 'st'; it is no rolling solve, so W-0635 is not
    announced and its full notice carries no 'keeps the earlier realization' sentence (that is for the rolling solves
    without realization rolls)."""
    from tests._support.coded_warnings import capture_coded
    from tests.engine.realization_roll import test_solve_shapes as ss

    _listed(monkeypatch)
    spec = ss._spec(ss.SD2_PERIODS, mode="single_solve", method="separate_dispatch", realized=["p1"], fan=ss.SD2_FAN,
                    wind=ss.SD2_WIND, gas={"t01": 10.0, "t02": 30.0})
    coded_log.notices()
    with capture_coded() as got:
        built, _steps = scm.run_spec(spec, tmp_path)
    assert len(built) == 2 and "__realized_dispatch" in built[1][0]
    assert got.names("W-0636") == ["st"] and not got.names("W-0635")
    (text,) = [n for n in coded_log.notices() if "separate_dispatch" in n]
    assert "Affected solves: st." in text and "keeps the earlier realization" not in text


def test_rolling_separate_dispatch_with_invest_branching_is_refused(tmp_path, monkeypatch):
    """T-w1 as ``separate_dispatch`` with ``invest_solve_branching_method`` investments_after_reveal_period: the
    setting applies to investment solves only, and a rolling solve is never one, so the run stops before any solve
    with an error that names 'st', says it is a rolling solve and points to ``dispatch_realization_method``
    (``SolveConfig.check_invest_branching``).  The delayed-unit case above keeps the rolling realization without
    look-ahead and its notice."""
    _listed(monkeypatch)
    spec = _set_solve(_separate(srl._tw1(0.5)), "invest_solve_branching_method",
                      pack("investments_after_reveal_period", "str"))
    with pytest.raises(FlexToolConfigError) as err:
        scm.run_spec(spec, tmp_path)
    text = str(err.value)
    assert "Solve 'st' sets invest_solve_branching_method investments_after_reveal_period" in text
    assert "it is a rolling solve" in text and "dispatch_realization_method" in text
