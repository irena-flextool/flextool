"""A stochastic forecast branch with the name of a period (item 267, owner 2026-11-15).

FlexTool allows a branch named like another period (a branch period is ``<period>_<branch>``, so the names stay
apart), but results and messages that name the two are easy to confuse, so the run warns once per solve and branch.
A branch named like the period where it fans out is an error at input time (before any solve), naming the solve,
the branch and the period.

Fixture (built from the schema JSON, never a checked-in sqlite): one node ``s`` (demand 1 per step), one unit
``src`` (10 MW), steps t0001..t0008, periods ``p1`` = t0001..t0004 and ``p2`` = t0005..t0008, one solve ``sol`` that
fans out at p1/t0001 into two forecast branches of weight 0.5.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import spinedb_api as api

import flextool
from flextool.engine.input._solve_state import FlexToolConfigError

SCHEMA_JSON = Path(flextool.__file__).parent / "schemas" / "spinedb_schema.json"
_counter = itertools.count()


def _m(d: dict, name: str) -> api.Map:
    return api.Map(list(d), list(d.values()), index_name=name)


def _run(tmp_path_factory, branches: tuple[str, str]):
    """Build the fixture with forecast *branches* at p1/t0001 and run it; ``{solve name: OrchestrationStep}``."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp(f"brname{next(_counter)}")
    db = work / "t.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db.resolve().as_posix()}"
    steps = [f"t{i:04d}" for i in range(1, 9)]
    ent = [("model", "flextool"), ("timeline", "tl"), ("node", "s"), ("unit", "src"),
           ("unit__outputNode", ("src", "s")), ("solve", "sol"), ("timeset", "ts1"), ("timeset", "ts2")]
    fan = _m({"p1": _m({b: _m({"t0001": _m({"no": 0.5}, "realized")}, "analysis_time") for b in branches},
                       "branch")}, "period")
    values = [
        ("model", "flextool", "solves", api.Array(["sol"])),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in steps}, "time")),
        ("node", "s", "node_type", "balance"), ("node", "s", "inflow", -1.0),
        ("node", "s", "penalty_up", 1000.0), ("node", "s", "penalty_down", 1000.0),
        ("unit", "src", "existing", 10.0), ("unit", "src", "virtual_unitsize", 1.0),
        ("unit", "src", "efficiency", 1.0),
        ("solve", "sol", "solve_mode", "single_solve"),
        ("solve", "sol", "period_timeset", _m({"p1": "ts1", "p2": "ts2"}, "period")),
        ("solve", "sol", "realized_periods", api.Array(["p1", "p2"])),
        ("solve", "sol", "years_represented", _m({"p1": 1.0, "p2": 1.0}, "period")),
        ("solve", "sol", "stochastic_branches", fan),
        ("timeset", "ts1", "timeline", "tl"), ("timeset", "ts1", "timeset_duration", _m({"t0001": 4.0}, "time")),
        ("timeset", "ts2", "timeline", "tl"), ("timeset", "ts2", "timeset_duration", _m({"t0005": 4.0}, "time")),
    ]
    with open_database_mapping(url) as m:
        _, err = api.import_data(
            m, alternatives=[("a", "")], scenarios=[("sc", False, "")],
            scenario_alternatives=[("sc", "a", None)], entities=ent,
            entity_alternatives=[("node", ("s",), "a", True), ("unit", ("src",), "a", True)],
            parameter_values=[(c, e, p, v, "a") for c, e, p, v in values])
        assert not err, err
        m.commit_session("fixture")
    return run_chain_from_db(url, "sc", work_folder=work / "w", keep_solutions=True)


def _warnings(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if "(stochastic_branches) named like" in ln]


def test_branch_named_like_a_period_warns_once_and_runs(tmp_path_factory, capsys) -> None:
    """Branch ``p2`` (the name of period p2) in the p1 fan: one warning naming the solve, the branch and the
    period; the run solves, and the branch's periods are ``p1_p2`` and ``p2_p2``."""
    steps = _run(tmp_path_factory, ("p2", "low"))
    warned = _warnings(capsys.readouterr().out)
    assert len(warned) == 1, warned
    assert "solve 'sol'" in warned[0] and "branch 'p2'" in warned[0] and "period(s) 'p1'" in warned[0]
    s = steps["sol"]
    assert s.solution is not None
    flow_periods = {str(d) for d in s.solution.value("v_flow")["d"].to_list()}
    assert {"p1_p2", "p2_p2", "p1_low", "p2_low"} <= flow_periods


def test_branch_names_that_are_not_period_names_do_not_warn(tmp_path_factory, capsys) -> None:
    _run(tmp_path_factory, ("high", "low"))
    assert _warnings(capsys.readouterr().out) == []


def test_branch_named_like_its_own_fan_period_stops_before_any_solve(tmp_path_factory, capsys) -> None:
    """Branch ``p1`` in the p1 fan: it would not be a member of the fan while the fan check counts it (its weight
    would disappear), so loading the solves stops with an error naming the solve, the branch and the period."""
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, ("p1", "low"))
    msg = str(err.value)
    assert "solve 'sol'" in msg and "branch 'p1'" in msg and "period 'p1'" in msg
    assert "where the branch fans out" in msg and "rename the branch 'p1'" in msg
    # Raised while the solves are loaded from the database (input time), not mid-run.
    assert any(e.name == "_check_branches_named_like_their_fan_period" for e in err.traceback)
    assert _warnings(capsys.readouterr().out) == []
