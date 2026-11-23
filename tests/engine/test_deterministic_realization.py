"""``dispatch_realization_method`` ``deterministic`` (the default since 4.7.0) on a solve with ``stochastic_branches``.

The solve ignores the forecasts and solves once with the data of its ``realized: yes`` branch (one W-0593 line), so
its LP is the LP of the same model written by hand without ``stochastic_branches``, with the realized branch's
data as a plain time series (the independent oracle: no stochastic data in it at all).  Without a ``realized: yes``
branch the run stops with an error naming the solve.  Without ``stochastic_branches`` the members give the same LP.

Model: one balance node ``elec`` (demand 100 MW, penalty 10 000 EUR/MWh), ``gen`` (200 MW, 10 EUR/MWh) and ``wind``
(100 MW, upper-limit profile ``wp``, in the group ``sg`` with ``include_stochastics``), one period ``p1`` on 1 h
steps t01..t06.  The forecasts ``high`` (0.9 everywhere) and ``low`` (0.05 everywhere) differ from the realized
series ``real`` = 0.3, 0.5, 0.2, 0.6, 0.4, 0.1, which every forecast made at t01 / t03 / t05 repeats from its
analysis time on.
"""
from __future__ import annotations

import logging

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _solve_copy_models as scm

STEPS = ("t01", "t02", "t03", "t04", "t05", "t06")
REAL = dict(zip(STEPS, (0.3, 0.5, 0.2, 0.6, 0.4, 0.1)))
FORECAST = {"high": 0.9, "low": 0.05}


def _spec(*, stochastic: bool, rolling: bool = False, realized_row: bool = True, method: str | None = None) -> dict:
    """The model; *stochastic*: ``stochastic_branches`` (the forecasts made at t01, and at t03 / t05 when
    *rolling*) and the profile as a branch map, else the realized series as a plain profile."""
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
    times = ("t01", "t03", "t05") if rolling else ("t01",)
    if stochastic:
        branches = [(b, nested([(at, nested([(t, v) for t in STEPS[STEPS.index(at):]])) for at in times]))
                    for b, v in FORECAST.items()]
        branches.append(("real", nested([(at, nested([(t, REAL[t]) for t in STEPS[STEPS.index(at):]]))
                                         for at in times])))
        profile = pack(pmap(branches), "map")
    else:
        profile = pack(pmap(list(REAL.items()), index_name="time"), "map")
    pv = [
        ["model", "flextool", "solves", scm.array(["st"]), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in STEPS], index_name="time"), "map"),
         "init"],
        ["timeset", "ts", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts", "timeset_duration", pack(pmap([("t01", 6.0)], index_name="time"), "map"), "init"],
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
        pv += [["solve", "st", "rolling_solve_jump", pack(2.0, "float"), "init"],
               ["solve", "st", "rolling_solve_horizon", pack(4.0, "float"), "init"]]
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


@pytest.mark.parametrize("rolling", [False, True], ids=["single", "rolling"])
def test_deterministic_solve_reads_the_realized_branch(tmp_path_factory, caplog, rolling) -> None:
    """The stochastic solve under the default method builds the oracle's LP in every roll (single: one build ``st``,
    no ``st__realized_dispatch``; rolling: three rolls reading the forecast made at each roll's start), with
    one W-0593 line naming the solve."""
    caplog.set_level(logging.WARNING)
    case, _ = scm.run_spec(_spec(stochastic=True, rolling=rolling), tmp_path_factory.mktemp("case"))
    lines = _w0593(caplog)
    oracle, _ = scm.run_spec(_spec(stochastic=False, rolling=rolling), tmp_path_factory.mktemp("oracle"))
    expected = ["st"] * (3 if rolling else 1)
    assert [s for s, _, _ in case] == expected
    _assert_same_lp(case, oracle)
    assert len(lines) == 1 and "solve 'st' is deterministic" in lines[0], lines


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
