"""Data version 84: ``invest_solve_branching_method`` replaces ``stochastic_invest_method``, ``shared_invest_periods``
and ``shared_operation_periods`` (spec ``stochastic_controls`` revision 5, proposals rev 4, owner decision D8).

The rules each row derives from (written from the spec, not from the code):

* a ``recourse`` row becomes ``investments_after_reveal_period`` in its own alternative, and a ``none`` or null row
  of a solve that has a ``recourse`` row becomes ``operations_only``, so every scenario reads what it read;
* under ``recourse``, ``shared_invest_periods`` unset or ``[]`` (each branch invested on its own: wait-and-see)
  converts and the solve is listed in the marker (results change); ``[reveal]`` converts with results unchanged;
  any other list leaves the solve unconverted (printed, nothing written);
* a stochastic investment solve whose ``shared_operation_periods`` is unset or a non-empty list is listed (its
  operations were tied; now each branch operates on its own from the reveal); ``[]`` was already untied;
* a stochastic dispatch solve with ``shared_operation_periods`` is listed (no longer read); nothing written;
* ``dispatch_realization_method`` ``separate_dispatch`` on a solve that is an investment solve in every scenario
  where the row wins is removed (D8) and the solve listed; where a lower-ranked value other than ``within_solve``
  would then win (an explicit ``deterministic`` flattens an investment solve), the row becomes ``within_solve``
  instead; a row that also wins where the solve is a dispatch solve is kept (undecided).

The reveal period is the first period with a forecast row (``realized`` other than ``yes``).  Pure planner rows
first, then the step on a database at 83 and the head refresh at 84.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, Map, to_database

from flextool.engine.input._solve_config import INVEST_SOLVE_BRANCHING_METHODS
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

AFTER = "investments_after_reveal_period"
OPS = "operations_only"


def _branches(period: str = "p1") -> Map:
    """One forecast row at *period*: branch A, analysis time t1, realized no, weight 1."""
    return Map([period], [Map(["A"], [Map(["t1"], [Map(["no"], [1.0])])])])


def _arr(*periods: str) -> Array:
    return Array(list(periods), value_type=str)


#: Solve ``v``: a single investment solve over p1, p2 with forecasts from p1 (the reveal), operations already untied
#: (``shared_operation_periods`` []), so only the row under test decides what is listed.
BASE = {
    "solve_mode": {("v", "base"): "single_solve"},
    "invest_periods": {("v", "base"): _arr("p1", "p2")},
    "stochastic_branches": {("v", "base"): _branches()},
    "period_timeset": {("v", "base"): Map(["p1", "p2"], ["ts", "ts"])},
    "shared_operation_periods": {("v", "base"): _arr()},
}


def _values(**extra) -> dict:
    out = {k: dict(v) for k, v in BASE.items()}
    for name, rows in extra.items():
        out.setdefault(name, {}).update(rows)
    return out


ONE = {"S": ["base"]}


def test_v84_methods_are_the_engine_methods() -> None:
    assert m._V84_METHODS == INVEST_SOLVE_BRANCHING_METHODS


@pytest.mark.parametrize("shared, listed", [
    (None, True),            # unset: wait-and-see before -> listed
    (_arr(), True),          # []: the same
    (_arr("p1"), False),     # [reveal]: the same LP after -> converted, not listed
])
def test_recourse_converts(shared, listed) -> None:
    extra = {"stochastic_invest_method": {("v", "base"): "recourse"}}
    if shared is not None:
        extra["shared_invest_periods"] = {("v", "base"): shared}
    plan = m.plan_branching_methods(_values(**extra), ONE)
    assert plan.writes == (("v", "base", AFTER),)
    assert [c[:2] for c in plan.converted] == [("v", "base")]
    assert plan.undecided == ()
    assert ("v" in plan.listed) is listed


def test_recourse_with_another_shared_list_is_undecided() -> None:
    # [p2] is neither unset, [] nor the reveal period p1.
    plan = m.plan_branching_methods(_values(stochastic_invest_method={("v", "base"): "recourse"},
                                            shared_invest_periods={("v", "base"): _arr("p2")}), ONE)
    assert plan.writes == () and plan.converted == ()
    assert [u[0] for u in plan.undecided] == ["v"]


def test_none_rows_of_a_recourse_solve_become_operations_only() -> None:
    # base holds none, E recourse: S1 = [base] reads none, S2 = [base, E] reads recourse; each keeps its reading.
    plan = m.plan_branching_methods(
        _values(stochastic_invest_method={("v", "base"): "none", ("v", "E"): "recourse"},
                shared_invest_periods={("v", "E"): _arr("p1")}),
        {"S1": ["base"], "S2": ["base", "E"]})
    assert sorted(plan.writes) == [("v", "E", AFTER), ("v", "base", OPS)]


def test_none_alone_writes_nothing() -> None:
    plan = m.plan_branching_methods(_values(stochastic_invest_method={("v", "base"): "none"}), ONE)
    assert plan.writes == () and plan.listed == {}


@pytest.mark.parametrize("ops, listed", [(None, True), (_arr("p1", "p2"), True), (_arr(), False)])
def test_tied_operations_of_an_investment_solve_are_listed(ops, listed) -> None:
    values = _values()
    del values["shared_operation_periods"][("v", "base")]
    if ops is not None:
        values["shared_operation_periods"][("v", "base")] = ops
    plan = m.plan_branching_methods(values, ONE)
    assert plan.writes == ()
    assert ("v" in plan.listed) is listed


def test_dispatch_solve_with_shared_operation_periods_is_listed() -> None:
    # A rolling solve is a dispatch solve whatever its invest_periods.
    values = _values(solve_mode={("v", "base"): "rolling_window"})
    plan = m.plan_branching_methods(values, ONE)
    assert "dispatch solve" in plan.listed["v"]
    values["shared_operation_periods"] = {}
    assert m.plan_branching_methods(values, ONE).listed == {}


def test_separate_dispatch_on_an_investment_solve_is_removed() -> None:
    plan = m.plan_branching_methods(_values(dispatch_realization_method={("v", "base"): "separate_dispatch"}), ONE)
    assert plan.removed == (("v", "base"),) and plan.within_solve == ()
    assert "removed" in plan.listed["v"]


def test_separate_dispatch_over_a_lower_ranked_deterministic_becomes_within_solve() -> None:
    # S = [base, E]: E's separate_dispatch wins; removing it would let base's deterministic win (a flattened solve).
    plan = m.plan_branching_methods(
        _values(dispatch_realization_method={("v", "base"): "deterministic", ("v", "E"): "separate_dispatch"}),
        {"S": ["base", "E"]})
    assert plan.removed == () and plan.within_solve == (("v", "E"),)


def test_separate_dispatch_over_a_lower_ranked_within_solve_is_removed() -> None:
    plan = m.plan_branching_methods(
        _values(dispatch_realization_method={("v", "base"): "within_solve", ("v", "E"): "separate_dispatch"}),
        {"S": ["base", "E"]})
    assert plan.removed == (("v", "E"),) and plan.within_solve == ()


def test_separate_dispatch_winning_on_a_dispatch_solve_too_is_undecided() -> None:
    # S1 = [base]: an investment solve; S2 = [base, R]: R makes it rolling (a dispatch solve).  The row wins in both.
    plan = m.plan_branching_methods(
        _values(dispatch_realization_method={("v", "base"): "separate_dispatch"},
                solve_mode={("v", "R"): "rolling_window"}),
        {"S1": ["base"], "S2": ["base", "R"]})
    assert plan.removed == () and plan.within_solve == ()
    assert [u[0] for u in plan.undecided] == ["v"]


# --- the step on a database -------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v83_db(tmp_path_factory) -> Path:
    """The bundled v25 template migrated to 83, with one investment solve ``v``: forecasts from p1,
    ``stochastic_invest_method`` recourse and ``shared_invest_periods`` unset (wait-and-see), and
    ``dispatch_realization_method`` separate_dispatch."""
    from flextool._resources import package_data_path
    from flextool.update_flextool.initialize_database import initialize_database

    path = tmp_path_factory.mktemp("v83_db") / "v83.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(package_data_path("schemas/pre_v26/flextool_template_v25.json")), str(path))
        migrate_database(f"sqlite:///{path}", up_to=83)
    with DatabaseMapping(f"sqlite:///{path}") as db:
        db.add_item("entity", entity_class_name="solve", name="v")
        db.add_item("scenario", name="S")
        db.add_item("scenario_alternative", scenario_name="S", alternative_name="base", rank=1)
        for name, value in (("solve_mode", "single_solve"), ("invest_periods", _arr("p1", "p2")),
                            ("period_timeset", Map(["p1", "p2"], ["ts", "ts"])),
                            ("stochastic_branches", _branches()), ("stochastic_invest_method", "recourse"),
                            ("dispatch_realization_method", "separate_dispatch")):
            raw, typ = to_database(value)
            db.add_item("parameter_value", entity_class_name="solve", entity_byname=("v",),
                        parameter_definition_name=name, alternative_name="base", value=raw, type=typ)
        db.commit_session("v83 solve")
    return path


def _migrate_copy(src: Path, tmp_path: Path) -> tuple[str, str]:
    dst = tmp_path / "db.sqlite"
    shutil.copy(src, dst)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        migrate_database(f"sqlite:///{dst}")
    return f"sqlite:///{dst}", out.getvalue()


def _solve_values(db) -> dict:
    return {pv["parameter_definition_name"]: pv["parsed_value"]
            for pv in db.find_parameter_values(entity_class_name="solve")}


def test_step_converts_lists_and_removes(v83_db: Path, tmp_path: Path) -> None:
    url, printed = _migrate_copy(v83_db, tmp_path)
    with DatabaseMapping(url) as db:
        values = _solve_values(db)
        names = {d["name"] for d in db.get_items("parameter_definition", entity_class_name="solve")}
        marker = [md for md in db.get_items("metadata")
                  if md["name"] == m.BEHAVIOUR_CHANGE_MARKER_PREFIX + m.V84_MARKER]
    assert values["invest_solve_branching_method"] == AFTER
    assert "dispatch_realization_method" not in values
    assert not names & set(m._V84_OLD_PARAMETERS)
    assert "solve v [base]" in printed and "separate_dispatch removed" in printed
    (md,) = marker
    stored = json.loads(md["value"])
    assert stored["since"] == 84 and stored["solves"] == ["v"]
    # A second run converts nothing more.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        migrate_database(url)
    with DatabaseMapping(url) as db:
        assert _solve_values(db) == values


def test_head_refresh_adds_the_missing_v84_definition(v83_db: Path, tmp_path: Path) -> None:
    url, _printed = _migrate_copy(v83_db, tmp_path)
    path = Path(url.removeprefix("sqlite:///"))
    with DatabaseMapping(url) as db:
        (pdef,) = db.get_items("parameter_definition", entity_class_name="solve",
                               name="invest_solve_branching_method")
        db.remove_item("parameter_definition", pdef["id"])
        db.commit_session("a database at 84 without the v84 definition")
    assert m._head_schema_missing_ro(path)
    with contextlib.redirect_stdout(io.StringIO()):
        migrate_database(url)
    assert not m._head_schema_missing_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION
