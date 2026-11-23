"""``dispatch_realization_method`` ``deterministic`` (the default since 4.7.0) on a solve with ``stochastic_branches``.

The solve ignores the forecasts and reads only its ``realized: yes`` branch (one W-0593 line), flattened into one
timeline: each step reads the realized series that covers it, where several do the one made nearest to the solve's
(a roll's) start.  So the LP of each build is the LP of the same model written by hand without
``stochastic_branches``, with that roll's flattened realized series as a plain time series (the independent oracle:
no stochastic data in it at all).  Without a ``realized: yes`` branch, or at a step no realized series covers, the
run stops with an error naming the solve.  Without ``stochastic_branches`` the members give the same LP.

Model: one balance node ``elec`` (demand 100 MW, penalty 10 000 EUR/MWh), ``gen`` (200 MW, 10 EUR/MWh) and ``wind``
(100 MW, upper-limit profile ``wp``, in the group ``sg`` with ``include_stochastics``), one period ``p1`` on 1 h
steps t01..t06.  The forecasts ``high`` (0.9 everywhere) and ``low`` (0.05 everywhere) differ from the realized
series ``real``: the series made at analysis time ``at`` is ``REAL[t] + 0.01 * position of at`` (t01: +0, t03:
+0.02, t05: +0.04), with REAL = 0.3, 0.5, 0.2, 0.6, 0.4, 0.1, so a build tells which analysis time each step read.
"""
from __future__ import annotations

import logging

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _solve_copy_models as scm

STEPS = ("t01", "t02", "t03", "t04", "t05", "t06")
REAL = dict(zip(STEPS, (0.3, 0.5, 0.2, 0.6, 0.4, 0.1)))
FORECAST = {"high": 0.9, "low": 0.05}
#: Forecasts made at t01, t03 and t05, each up to the end of the timeline (they overlap).
OVERLAP = {at: STEPS[STEPS.index(at):] for at in ("t01", "t03", "t05")}
#: Forecasts made at t01, t03 and t05, each up to the next one (no overlap).
ABUTTING = {"t01": ("t01", "t02"), "t03": ("t03", "t04"), "t05": ("t05", "t06")}


def _real(at: str, t: str) -> float:
    """The realized value at *t* of the series made at analysis time *at*."""
    return round(REAL[t] + 0.01 * STEPS.index(at), 6)


def _made_at(at: str) -> dict[str, float]:
    """The plain profile of the realized series made at *at*, over the whole timeline (an oracle profile)."""
    return {t: _real(at, t) for t in STEPS}


def _spec(*, stochastic: bool, rolling: bool | tuple[float, float] = False, realized_row: bool = True,
          method: str | None = None, forecasts: "dict[str, tuple[str, ...]] | None" = None,
          plain: "dict[str, float] | None" = None, start: str = "t01") -> dict:
    """The model; *stochastic*: ``stochastic_branches`` with the forecasts made at the analysis times of
    *forecasts* (``{analysis time: steps it covers}``, default t01 over all steps) and the profile as a branch map,
    else *plain* (default the series made at t01) as a plain profile.  *rolling*: ``(jump, horizon)``, ``True`` =
    ``(2, 4)``.  *start*: the first step of the timeset (it runs to t06)."""
    forecasts = forecasts or {"t01": STEPS}
    plain = plain or _made_at("t01")
    from flextool.update_flextool.fixture_schema import schema_sections

    pack, pmap, nested = scm.pack, scm.pmap, scm.nested
    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", None]]
    spec["entities"] = [["model", "flextool", None], ["node", "elec", None], ["timeline", "tl", None],
                        ["timeset", "ts", None], ["unit", "gen", None], ["unit", "wind", None],
                        ["group", "sg", None], ["profile", "wp", None], ["solve", "st", None],
                        ["unit__outputNode", ["gen", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                        ["group__unit", ["sg", "wind"], None], ["unit__node__profile", ["wind", "elec", "wp"], None]]
    spec["entity_alternatives"] = [["node", ["elec"], "init", True], ["unit", ["gen"], "init", True],
                                   ["unit", ["wind"], "init", True]]
    times = tuple(forecasts)
    if stochastic:
        branches = [(b, nested([(at, nested([(t, v) for t in steps])) for at, steps in forecasts.items()]))
                    for b, v in FORECAST.items()]
        branches.append(("real", nested([(at, nested([(t, _real(at, t)) for t in steps]))
                                         for at, steps in forecasts.items()])))
        profile = pack(pmap(branches), "map")
    else:
        profile = pack(pmap(list(plain.items()), index_name="time"), "map")
    pv = [
        ["model", "flextool", "solves", scm.array(["st"]), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in STEPS], index_name="time"), "map"),
         "init"],
        ["timeset", "ts", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts", "timeset_duration", pack(pmap([(start, 7.0 - int(start[1:]))], index_name="time"),
                                                    "map"), "init"],
        ["node", "elec", "node_type", pack("balance", "str"), "init"],
        ["node", "elec", "penalty_up", pack(10000.0, "float"), "init"],
        ["node", "elec", "penalty_down", pack(10000.0, "float"), "init"],
        ["node", "elec", "inflow", pack(-100.0, "float"), "init"],
        ["unit", "gen", "efficiency", pack(1.0, "float"), "init"],
        ["unit", "gen", "existing", pack(200.0, "float"), "init"],
        ["unit__outputNode", ["gen", "elec"], "other_operational_cost", pack(10.0, "float"), "init"],
        ["unit", "wind", "efficiency", pack(1.0, "float"), "init"],
        ["unit", "wind", "existing", pack(100.0, "float"), "init"],
        ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
        ["profile", "wp", "profile", profile, "init"],
        ["group", "sg", "include_stochastics", pack("yes", "str"), "init"],
        ["solve", "st", "solve_mode", pack("rolling_window" if rolling else "single_solve", "str"), "init"],
        ["solve", "st", "period_timeset", pack(pmap([("p1", "ts")], index_name="period"), "map"), "init"],
        ["solve", "st", "realized_periods", scm.array(["p1"]), "init"],
        ["solve", "st", "years_represented", pack(pmap([("p1", 1.0)], index_name="period"), "map"), "init"],
    ]
    if rolling:
        jump, horizon = (2.0, 4.0) if rolling is True else rolling
        pv += [["solve", "st", "rolling_solve_jump", pack(jump, "float"), "init"],
               ["solve", "st", "rolling_solve_horizon", pack(horizon, "float"), "init"]]
    if stochastic:
        rows = [(b, w, "no") for b, w in (("high", 1.0), ("low", 3.0))]
        if realized_row:
            rows.append(("real", 0.0, "yes"))
        fan = pmap([("p1", nested([(b, nested([(at, nested([(yes, w)])) for at in times])) for b, w, yes in rows]))])
        pv.append(["solve", "st", "stochastic_branches", pack(fan, "map"), "init"])
    if method is not None:
        pv.append(["solve", "st", "dispatch_realization_method", pack(method, "str"), "init"])
    spec["parameter_values"] = pv
    return spec


def _w0593(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[W-0593]" in r.getMessage()]


def _assert_same_lp(case: list, oracle: list) -> None:
    """Tier A: the same builds, each MPS byte for byte (after the canonical diff, which names a difference)."""
    assert [s for s, _, _ in case] == [s for s, _, _ in oracle]
    assert not scm.lp_differences(case, oracle, rtol=0.0)
    for (solve, a, off_a), (_, b, off_b) in zip(case, oracle):
        assert a.read_bytes() == b.read_bytes(), solve
        assert off_a == off_b, solve


def test_deterministic_solve_reads_the_realized_branch(tmp_path_factory, caplog) -> None:
    """The single stochastic solve under the default method builds the oracle's LP (one build ``st``, no
    ``st__realized_dispatch``), with one W-0593 line naming the solve."""
    caplog.set_level(logging.WARNING)
    case, _ = scm.run_spec(_spec(stochastic=True), tmp_path_factory.mktemp("case"))
    lines = _w0593(caplog)
    oracle, _ = scm.run_spec(_spec(stochastic=False), tmp_path_factory.mktemp("oracle"))
    assert [s for s, _, _ in case] == ["st"]
    _assert_same_lp(case, oracle)
    assert lines == ["Warning: solve 'st' is deterministic: only its realized: yes branch is used [W-0593]"], lines


# Rolling cases: (jump, horizon), the forecasts, and the analysis time whose series each roll reads, hand-derived
# from the rule "each step reads the nearest analysis time (to the roll's start, in steps; a tie: the earlier)
# whose realized series covers it":
# * overlap, jump 2: rolls start t01 / t03 / t05; each start is an analysis time (distance 0) whose series covers
#   the roll's whole horizon, so roll k reads the series made at its own start.
# * overlap, jump 3: rolls start t01 / t04.  Roll 2 (t04..t06): t03 and t05 are both one step away, the earlier
#   (t03) wins, and its series covers t04..t06; it never reads the series made at t05, nor 0 at t04.
# * once: only t01's forecasts, over the whole timeline: every roll reads them (rolls 2 and 3 have no forecast of
#   their own).
ROLLING_CASES = {
    "overlap_jump2": ((2.0, 4.0), OVERLAP, ["t01", "t03", "t05"]),
    "overlap_jump3": ((3.0, 4.0), OVERLAP, ["t01", "t03"]),
    "once": ((2.0, 4.0), {"t01": STEPS}, ["t01", "t01", "t01"]),
}


@pytest.mark.parametrize("case_id", list(ROLLING_CASES))
def test_rolling_deterministic_solve_reads_the_nearest_realized_series(tmp_path_factory, case_id) -> None:
    """Roll k of the stochastic rolling solve builds roll k of the oracle whose plain profile is the series that
    roll reads (``ROLLING_CASES``); one oracle run per distinct series."""
    rolling, forecasts, reads = ROLLING_CASES[case_id]
    case, _ = scm.run_spec(_spec(stochastic=True, rolling=rolling, forecasts=forecasts),
                           tmp_path_factory.mktemp("case"))
    assert [s for s, _, _ in case] == ["st"] * len(reads)
    oracles = {at: scm.run_spec(_spec(stochastic=False, rolling=rolling, plain=_made_at(at)),
                                tmp_path_factory.mktemp(f"oracle_{at}"))[0] for at in dict.fromkeys(reads)}
    for k, at in enumerate(reads):
        _assert_same_lp([case[k]], [oracles[at][k]])


def test_rolling_deterministic_solve_reads_one_flattened_timeline(tmp_path_factory) -> None:
    """Forecasts that abut (t01: t01..t02, t03: t03..t04, t05: t05..t06): every roll reads the one flattened
    series -- t01's values at t01 / t02, t03's at t03 / t04, t05's at t05 / t06 (the only series covering each
    step) -- even where its own start's series ends inside its horizon."""
    flattened = {t: _real(at, t) for at, steps in ABUTTING.items() for t in steps}
    case, _ = scm.run_spec(_spec(stochastic=True, rolling=True, forecasts=ABUTTING), tmp_path_factory.mktemp("case"))
    oracle, _ = scm.run_spec(_spec(stochastic=False, rolling=True, plain=flattened), tmp_path_factory.mktemp("oracle"))
    assert [s for s, _, _ in case] == ["st"] * 3
    _assert_same_lp(case, oracle)


def test_realized_series_made_before_the_solve_starts(tmp_path_factory) -> None:
    """The timeset starts at t03 (t03..t06) and the only realized series is made at t01 (two steps before the
    solve's first step, off its active steps) over t01..t06: it covers every step, so the solve reads its values at
    t03..t06 (0.22, 0.62, 0.42, 0.12 -- REAL + 0 for t01) -- the LP of the plain profile made at t01 on the same
    timeset."""
    case, _ = scm.run_spec(_spec(stochastic=True, start="t03"), tmp_path_factory.mktemp("case"))
    oracle, _ = scm.run_spec(_spec(stochastic=False, start="t03"), tmp_path_factory.mktemp("oracle"))
    assert [s for s, _, _ in case] == ["st"]
    _assert_same_lp(case, oracle)


def test_rolling_reads_the_nearer_later_series_first(tmp_path_factory) -> None:
    """Rolls t01..t03 and t04..t06 (jump 3, horizon 3); series made at t01 (over t01..t06) and at t05 (over
    t05..t06).  Roll 2 starts at t04: t05 is one step away, t01 three, so t05's series is read first where it
    covers (t05: 0.4 + 0.04 = 0.44, t06: 0.1 + 0.04 = 0.14) and t01's at t04 (0.6).  Roll 1 reads t01's (distance
    0).  A rule "the latest made at or before the start" would read t01's 0.4 / 0.1 at t05 / t06 instead."""
    forecasts = {"t01": STEPS, "t05": ("t05", "t06")}
    flattened = {t: _real("t05" if t in ("t05", "t06") else "t01", t) for t in STEPS}
    case, _ = scm.run_spec(_spec(stochastic=True, rolling=(3.0, 3.0), forecasts=forecasts),
                           tmp_path_factory.mktemp("case"))
    oracle, _ = scm.run_spec(_spec(stochastic=False, rolling=(3.0, 3.0), plain=flattened),
                             tmp_path_factory.mktemp("oracle"))
    assert [s for s, _, _ in case] == ["st"] * 2
    _assert_same_lp(case, oracle)


def test_step_no_realized_series_covers_is_an_error(tmp_path_factory) -> None:
    """A gap: t03's series covers only t03, so t04 (in the first roll's horizon) has no realized value."""
    gap = {"t01": ("t01", "t02"), "t03": ("t03",), "t05": ("t05", "t06")}
    with pytest.raises(FlexToolConfigError, match=(
            r"profile 'wp', parameter 'profile': solve 'st[^']*' has stochastic_branches and "
            r"dispatch_realization_method deterministic, .* No realized series covers time step 't04'")):
        scm.run_spec(_spec(stochastic=True, rolling=True, forecasts=gap), tmp_path_factory.mktemp("gap"))


def test_explicit_deterministic_is_the_default(tmp_path_factory) -> None:
    """An explicit ``deterministic`` gives the LP of the absent value (the shipped default)."""
    explicit, _ = scm.run_spec(_spec(stochastic=True, method="deterministic"), tmp_path_factory.mktemp("explicit"))
    oracle, _ = scm.run_spec(_spec(stochastic=False), tmp_path_factory.mktemp("oracle"))
    _assert_same_lp(explicit, oracle)


def test_within_solve_still_solves_the_forecasts(tmp_path_factory) -> None:
    """``within_solve`` keeps the forecasts (the branch periods p1_high / p1_low) and the realization solve."""
    built, steps = scm.run_spec(_spec(stochastic=True, method="within_solve"), tmp_path_factory.mktemp("within"))
    assert [s for s, _, _ in built] == ["st", "st__realized_dispatch"]
    cols = set(scm.lp_snapshot().parse_mps(built[0][1]).cols)
    assert any("p1_high" in c for c in cols) and any("p1_low" in c for c in cols)


def test_forecasts_only_fan_under_deterministic_is_an_error(tmp_path_factory) -> None:
    with pytest.raises(FlexToolConfigError, match=(
            r"Solve 'st' has stochastic_branches and dispatch_realization_method deterministic \(the default "
            r"since FlexTool 4\.7\.0\), but no branch with realized: yes\.")):
        scm.run_spec(_spec(stochastic=True, realized_row=False), tmp_path_factory.mktemp("noreal"))


def test_within_solve_gives_the_same_lp_without_branches(tmp_path_factory) -> None:
    """Without ``stochastic_branches`` an explicit ``within_solve`` builds the LP of the default.  (``separate_dispatch``
    needs a ``realized: yes`` branch: the run stops, ``test_w4a_dispatch_realization_guard``.)"""
    explicit, _ = scm.run_spec(_spec(stochastic=False, method="within_solve"), tmp_path_factory.mktemp("within"))
    default, _ = scm.run_spec(_spec(stochastic=False), tmp_path_factory.mktemp("default"))
    _assert_same_lp(explicit, default)


def test_resolver_and_guard() -> None:
    from types import SimpleNamespace

    from flextool.engine.input._solve_config import SolveConfig
    from flextool.engine.solve._orchestration import _dispatch_realization_raw_guard

    cfg = SimpleNamespace(dispatch_realization_method={"a": "Deterministic", "b": " within_solve ", "c": "",
                                                       "d": "separate_dispatch"})
    resolve = SolveConfig.dispatch_realization_method_for
    assert [resolve(cfg, s) for s in "abcde"] == [
        "deterministic", "within_solve", "deterministic", "separate_dispatch", "deterministic"]
    for s in "abc":
        _dispatch_realization_raw_guard(cfg, s)
    cfg.dispatch_realization_method["x"] = "determinstic"
    with pytest.raises(FlexToolConfigError, match="Supported: 'deterministic' \\(default\\), 'within_solve' and "
                                                  "'separate_dispatch'"):
        _dispatch_realization_raw_guard(cfg, "x")
