"""A period is dispatched only after its investments are realized (decision D11).

The run stops before anything solves when a solve realizes the dispatch of a period (``realized_periods``) that an
investment solve lists in ``invest_periods`` while no investment solve up to and including it realized the period's
investments.  Before, the solve dispatched the period on the capacity that existed before those investments.

Integration cases on ``stoch_two_period_invest`` (periods p2035, p2040; the fan opens at p2035): a dispatch solve
``disp`` (no investments, ``realized_periods`` as given) is appended after the investment solve ``stoch_2p_inv``,
and a realized branch ``real`` (the data of ``high``) is added to both fans, so that each case ran to the end before
the check (``disp`` dispatching on the capacity that existed before the period's investments).

* (a) ``after_reveal``: ``investments_after_reveal_period``, realized investments [p2035]; ``disp`` realizes p2040,
  whose investments are per forecast: refused.
* (b) ``recourse``: ``investments_at_reveal_period`` with the reveal at the first period, nothing realized;
  ``disp`` realizes p2035: refused.
* (c) ``det`` (no ``stochastic_branches``): ``stoch_2p_inv`` invests in p2035 and p2040 and realizes p2035 (the
  investments and the dispatch); ``disp`` realizing p2040 is refused, ``disp`` realizing p2035 runs.
* D12: ``y2020_2029_2x5y`` (``tests.json``) dispatches p2025, whose investments it decides itself but does not
  realize: it runs, with warning W-0641.
* (f) the engine's realization solve (``w4b_auto_dispatch``: ``stoch_rd`` invests, ``stoch_rd__realized_dispatch``
  dispatches its realized periods after it) runs.

The pure check (``check_dispatch_after_investments``) on hand rows: (d) no investments, (e) a year-by-year chain (and
the same solve dispatching its own look-ahead period: W-0641), a nested solve renamed per period, a realization
solve whose parent did not realize the investments of a period it dispatches.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._realized_overlap import check_dispatch_after_investments
from tests._support.coded_warnings import capture_coded

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"
W4B_FIXTURE = ROOT / "tests" / "fixtures" / "w4b_auto_dispatch.json"
INVEST = "stoch_2p_inv"
DISP = "disp"
TESTS_JSON = ROOT / "tests" / "fixtures" / "tests.json"
W0641 = "Warning: solve '{solve}' dispatches {periods} on investments it does not realize [W-0641]"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _enc(value) -> list:
    kind = "str" if isinstance(value, str) else "array" if "value_type" in value else "map"
    return [base64.b64encode(json.dumps(value).encode()).decode("ascii"), kind]


def _arr(periods: list[str]) -> dict:
    return {"value_type": "str", "data": periods}


def _fan_row(realized: str, weight: float) -> dict:
    """The ``{t0001: {realized: weight}}`` part of a fan row (the fan opens at the first step of p2035)."""
    return {"type": "map", "index_type": "str", "data": [
        ["t0001", {"type": "map", "index_type": "str", "data": [[realized, weight]]}]]}


def _with_dispatch_solve(tmp_path: Path, disp_realized: list[str]) -> Path:
    """The fixture with ``disp`` after ``stoch_2p_inv`` in ``model.solves`` and the scenario ``det``."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = data["parameter_values"]
    rows += [["solve", DISP, r[2], r[3], "init"] for r in list(rows)
             if r[0] == "solve" and r[1] == INVEST and r[4] == "init"
             and r[2] in ("period_timeset", "solve_mode", "years_represented")]
    rows.append(["solve", DISP, "realized_periods", _enc(_arr(disp_realized)), "init"])
    for r in rows:
        if r[0] == "model" and r[2] == "solves":
            r[3] = _enc(_arr([INVEST, DISP]))
    data["entities"].append(["solve", DISP, None])
    # A realized branch ``real`` with the data of ``high``: the investment solve realizes what it commits in its fan,
    # and ``disp`` dispatches it (deterministic: the realized branch only).  So without the check both runs go on.
    for r in rows:
        if r[4] == "stoch" and r[2] in ("profile", "stochastic_branches") and r[1] in ("wprof", INVEST):
            value = json.loads(base64.b64decode(r[3][0]))
            high = next(row for row in value["data"] if row[0] == "high") if r[2] == "profile" else None
            if high is not None:
                value["data"].append(["real", high[1]])
            else:
                value["data"][0][1]["data"].append(["real", _fan_row("yes", 0.0)])
            r[3] = _enc(value)
    rows.append(["solve", DISP, "stochastic_branches",
                 _enc({"index_type": "str", "data": [["p2035", {"type": "map", "index_type": "str", "data": [
                     ["real", _fan_row("yes", 1.0)]]}]]}), "stoch"])
    # det: no stochastic_branches; the investment solve realizes p2035 only.
    data["alternatives"].append(["det", "realize p2035"])
    data["scenarios"].append(["det", False, "deterministic invest + dispatch"])
    data["scenario_alternatives"] += [["det", "init", "base"], ["det", "base", "det"], ["det", "det", None]]
    rows += [["solve", INVEST, par, _enc(_arr(["p2035"])), "det"]
             for par in ("realized_periods", "realized_invest_periods")]
    out = tmp_path / "with_disp.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def _run(fixture_json: Path, scenario: str, tmp_path: Path):
    tests_dir = ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    url = json_to_db(fixture_json, tmp_path / f"{fixture_json.stem}.sqlite")
    migrate_database(url)
    work = tmp_path / "work"
    work.mkdir()
    assert os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1"
    return run_chain_from_db(url, scenario, work_folder=work, warm=False, keep_solutions=True)


@pytest.mark.parametrize(("scenario", "realized", "period", "method"), [
    ("after_reveal", ["p2040"], "p2040", "investments_after_reveal_period: its investments after the reveal "
                                         "period 'p2035' are per forecast branch"),
    ("recourse", ["p2035"], "p2035", "investments_at_reveal_period: its investments from the reveal period "
                                     "'p2035' on are per forecast branch"),
    ("det", ["p2040"], "p2040", None),
])
def test_dispatch_before_the_investments_are_realized_is_refused(tmp_path, scenario, realized, period,
                                                                    method) -> None:
    with pytest.raises(FlexToolConfigError) as err:
        _run(_with_dispatch_solve(tmp_path, realized), scenario, tmp_path)
    msg = str(err.value)
    assert msg.startswith(f"Solve '{DISP}' realizes the dispatch of period '{period}' (realized_periods) before its "
                          f"investments are realized: solve '{INVEST}' lists it in invest_periods")
    assert f"add '{period}' to the realized_invest_periods of an investment solve that runs no later" in msg
    assert "leave it out of the realized_periods of this solve" in msg
    if method is None:
        assert "invest_solve_branching_method" not in msg
    else:
        assert f"Solve '{INVEST}' sets invest_solve_branching_method {method}" in msg
    # Before anything solved: no solve wrote a result.
    assert not list((tmp_path / "work" / "output_raw").glob("v_obj__*"))


@pytest.mark.solver
def test_dispatch_of_a_realized_investment_period_runs(tmp_path) -> None:
    """(c) passing: ``disp`` dispatches p2035, whose investments ``stoch_2p_inv`` realized: no W-0641 either."""
    with capture_coded() as got:
        steps = _run(_with_dispatch_solve(tmp_path, ["p2035"]), "det", tmp_path)
    assert set(steps) == {INVEST, DISP}
    assert all(step.optimal for step in steps.values())
    assert "W-0641" not in got.codes()


@pytest.mark.solver
def test_a_solve_dispatching_its_own_unrealized_investments_runs_with_a_warning(tmp_path) -> None:
    """D12 on ``y2020_2029_2x5y`` (``tests.json``): one solve invests in p2020 and p2025, realizes the investments
    of p2020 only and dispatches both.  Its own ``invest_periods`` count as decided for its dispatch: it runs, with
    one W-0641 naming the solve and p2025."""
    with capture_coded() as got:
        steps = _run(TESTS_JSON, "y2020_2029_2x5y", tmp_path)
    assert set(steps) == {"y2020_2029_2x5y"}
    assert got.lines("W-0641") == [W0641.format(solve="y2020_2029_2x5y", periods="p2025")]


@pytest.mark.solver
def test_the_engine_realization_solve_runs(tmp_path) -> None:
    """(f): the realization solve drops the investments and dispatches its parent's realized periods after it."""
    steps = _run(W4B_FIXTURE, "w8_realized_dispatch", tmp_path)
    assert set(steps) == {"stoch_rd", "stoch_rd__realized_dispatch"}


def _cfg(solves: dict[str, dict[str, list[str]]], **extra) -> SimpleNamespace:
    """A solve configuration with the period lists of *solves* (``{solve: {field: [period]}}``), not rolling."""
    fields = ("invest_periods", "realized_invest_periods", "realized_periods")
    lists = {f: {s: [(s, p) for p in v.get(f, [])] for s, v in solves.items()} for f in fields}
    return SimpleNamespace(**lists, solve_modes={}, **extra)


def _order(solves: list[str], periods: tuple[str, ...] = ("p1", "p2")) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(solve, complete solve, periods)`` of single solves over *periods*, in run order."""
    return [(s, s, periods) for s in solves]


def test_a_nested_solve_renamed_per_period_is_checked_per_period() -> None:
    """``inv`` split by a 2-D map into ``inv_p1`` / ``inv_p2`` (each realizes its period), with ``disp`` contained
    in each and renamed ``disp_p1`` / ``disp_p2`` (``realized_periods`` [p1, p2] of ``disp``, one period each): run
    order inv_p1, disp_p1, inv_p2, disp_p2 passes; dispatching p2 in ``disp_p1`` would come before ``inv_p2``."""
    rows = {"inv_p1": {"invest_periods": ["p1"], "realized_invest_periods": ["p1"]},
            "inv_p2": {"invest_periods": ["p2"], "realized_invest_periods": ["p2"]},
            "disp": {"realized_periods": ["p1", "p2"]}}
    order = [("inv_p1", "inv_p1", ("p1",)), ("disp_p1", "disp", ("p1",)),
             ("inv_p2", "inv_p2", ("p2",)), ("disp_p2", "disp", ("p2",))]
    check_dispatch_after_investments(_cfg(rows), order)
    order[1] = ("disp_p1", "disp", ("p1", "p2"))
    cfg = _cfg(rows, invest_branching_method_for=lambda s: "operations_only", reveal_period=lambda s: None)
    with pytest.raises(FlexToolConfigError, match=r"^Solve 'disp_p1' realizes the dispatch of period 'p2' "):
        check_dispatch_after_investments(cfg, order)


def test_a_run_without_investments_passes() -> None:
    """(d): nothing is invested, so every dispatch passes."""
    check_dispatch_after_investments(_cfg({"a": {"realized_periods": ["p1"]}, "b": {"realized_periods": ["p2"]}}),
                                     _order(["a", "b"]))


def test_a_year_by_year_chain_passes_and_an_own_look_ahead_dispatch_warns() -> None:
    """(e): A invests in p1 and p2 (p2 is look-ahead) and realizes p1; B invests in and realizes p2: no line.  With A
    dispatching p2 as well, A decides p2's investments itself (D12): it runs, with one W-0641 naming A and p2."""
    chain = {"A": {"invest_periods": ["p1", "p2"], "realized_invest_periods": ["p1"], "realized_periods": ["p1"]},
             "B": {"invest_periods": ["p2"], "realized_invest_periods": ["p2"], "realized_periods": ["p2"]}}
    with capture_coded() as got:
        check_dispatch_after_investments(_cfg(chain), _order(["A", "B"]))
    assert "W-0641" not in got.codes()
    chain["A"]["realized_periods"] = ["p1", "p2"]
    with capture_coded() as got:
        check_dispatch_after_investments(_cfg(chain), _order(["A", "B"]))
    assert got.lines("W-0641") == [W0641.format(solve="A", periods="p2")]


def test_a_realization_solve_needs_the_investments_its_parent_realized() -> None:
    """A realization solve dispatches its parent's periods on the investments the parent realized (only those are
    handed on).  ``inv`` invests in p1 and p2, realizes the investments of p1 and dispatches both (W-0641 alone);
    its realization solve dispatching p2 is refused and named as the realization solve of ``inv``."""
    rows = {"inv": {"invest_periods": ["p1", "p2"], "realized_invest_periods": ["p1"],
                    "realized_periods": ["p1", "p2"]},
            "inv__realized_dispatch": {"realized_periods": ["p2"]}}
    cfg = _cfg(rows, realized_dispatch_parent={"inv__realized_dispatch": "inv"},
               invest_branching_method_for=lambda s: "operations_only", reveal_period=lambda s: None)
    with pytest.raises(FlexToolConfigError) as err:
        check_dispatch_after_investments(cfg, _order(["inv", "inv__realized_dispatch"]))
    assert str(err.value).startswith("The realization solve 'inv__realized_dispatch' of solve 'inv' realizes the "
                                     "dispatch of period 'p2' (realized_periods) before its investments are realized: "
                                     "solve 'inv' lists it in invest_periods")
    rows["inv"]["realized_invest_periods"] = ["p1", "p2"]
    with capture_coded() as got:
        check_dispatch_after_investments(_cfg(rows, realized_dispatch_parent={"inv__realized_dispatch": "inv"}),
                                         _order(["inv", "inv__realized_dispatch"]))
    assert "W-0641" not in got.codes()
