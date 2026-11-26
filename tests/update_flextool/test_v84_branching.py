"""Data version 84: ``invest_solve_branching_method`` replaces ``stochastic_invest_method``, ``shared_invest_periods``
and ``shared_operation_periods``.

The rules (from the spec, not from the code): a solve is eligible when its own rows, in any alternative, can make it
a stochastic investment solve (a ``solve_mode`` other than ``rolling_window`` or none, a non-empty ``invest_periods``
or ``realized_invest_periods``, a ``stochastic_branches`` row with an entry).  On an eligible solve with a
``recourse`` row each ``recourse`` row becomes ``investments_after_reveal_period`` and each other row
``operations_only``, in its own alternative; nothing else is written (``dispatch_realization_method`` untouched).
The old definitions and their list go; the solves with a row of any old parameter are printed and listed in the
W-0637 marker.  Hand-built databases at 83, then the head refresh at 84.
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
from flextool.update_flextool import db_migration as dm
from flextool.update_flextool.db_migration import head, runner, v84_branching as m
from flextool.update_flextool.db_migration import migrate_database

AFTER = "investments_after_reveal_period"
OPS = "operations_only"


def _branches(period: str = "p1") -> Map:
    """One forecast row at *period*: branch A, analysis time t1, realized no, weight 1."""
    return Map([period], [Map(["A"], [Map(["t1"], [Map(["no"], [1.0])])])])


def _arr(*periods: str) -> Array:
    return Array(list(periods), value_type=str)


def test_v84_methods_are_the_engine_methods() -> None:
    assert m._V84_METHODS == INVEST_SOLVE_BRANCHING_METHODS


#: (solve, alternative, parameter, value) of the database at 83.  ``v``: an investment solve with forecasts,
#: ``recourse`` in base and ``none`` in ``high``, ``separate_dispatch`` (kept).  ``r``: the same but rolling
#: (not eligible).  ``d``: a dispatch solve with ``shared_operation_periods`` only.  ``w``: an investment solve with
#: forecasts and no old setting (not listed).
INVEST = (("invest_periods", _arr("p1", "p2")), ("period_timeset", Map(["p1", "p2"], ["ts", "ts"])),
          ("stochastic_branches", _branches()))
ROWS = (
    *[("v", "base", n, v) for n, v in (("solve_mode", "single_solve"), *INVEST, ("stochastic_invest_method", "recourse"),
                                       ("shared_invest_periods", _arr("p1")),
                                       ("dispatch_realization_method", "separate_dispatch"))],
    ("v", "high", "stochastic_invest_method", "none"),
    *[("r", "base", n, v) for n, v in (("solve_mode", "rolling_window"), *INVEST,
                                       ("stochastic_invest_method", "recourse"))],
    *[("d", "base", n, v) for n, v in (("solve_mode", "single_solve"), ("stochastic_branches", _branches()),
                                       ("shared_operation_periods", _arr("p1")))],
    *[("w", "base", n, v) for n, v in (("solve_mode", "single_solve"), *INVEST)],
)


@pytest.fixture(scope="module")
def v83_db(tmp_path_factory) -> Path:
    """The bundled v25 template migrated to 83, with the solves of :data:`ROWS` and the scenario ``S`` (base, high)."""
    from flextool._resources import package_data_path
    from flextool.update_flextool.initialize_database import initialize_database

    path = tmp_path_factory.mktemp("v83_db") / "v83.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(package_data_path("schemas/pre_v26/flextool_template_v25.json")), str(path))
        migrate_database(f"sqlite:///{path}", up_to=83)
    with DatabaseMapping(f"sqlite:///{path}") as db:
        db.add_item("alternative", name="high")
        for solve in dict.fromkeys(r[0] for r in ROWS):
            db.add_item("entity", entity_class_name="solve", name=solve)
        db.add_item("scenario", name="S")
        for rank, alt in enumerate(("base", "high"), start=1):
            db.add_item("scenario_alternative", scenario_name="S", alternative_name=alt, rank=rank)
        for solve, alt, name, value in ROWS:
            raw, typ = to_database(value)
            db.add_item("parameter_value", entity_class_name="solve", entity_byname=(solve,),
                        parameter_definition_name=name, alternative_name=alt, value=raw, type=typ)
        db.commit_session("v83 solves")
    return path


def _migrate_copy(src: Path, tmp_path: Path) -> tuple[str, str]:
    dst = tmp_path / "db.sqlite"
    shutil.copy(src, dst)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        migrate_database(f"sqlite:///{dst}")
    return f"sqlite:///{dst}", out.getvalue()


def _values(db, name: str) -> dict:
    return {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
            for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name=name)}


def test_step_writes_recourse_on_eligible_solves_and_lists_old_settings(v83_db: Path, tmp_path: Path) -> None:
    url, printed = _migrate_copy(v83_db, tmp_path)
    with DatabaseMapping(url) as db:
        written = _values(db, "invest_solve_branching_method")
        dispatch = _values(db, "dispatch_realization_method")
        names = {d["name"] for d in db.get_items("parameter_definition", entity_class_name="solve")}
        lists = {x["name"] for x in db.get_items("parameter_value_list")}
        marker = [md for md in db.get_items("metadata")
                  if md["name"] == dm.BEHAVIOUR_CHANGE_MARKER_PREFIX + m.V84_MARKER]
    assert written == {("v", "base"): AFTER, ("v", "high"): OPS}
    assert dispatch[("v", "base")] == "separate_dispatch"
    assert not names & set(m._V84_OLD_PARAMETERS) and m._V84_OLD_LIST not in lists
    (md,) = marker
    assert json.loads(md["value"]) == {"since": 84, "solves": ["d", "r", "v"]}
    (line,) = [x for x in printed.splitlines() if x.startswith("v84:")]
    assert line.endswith("check invest_solve_branching_method on solve d, r, v")
    # A second run changes nothing.
    with contextlib.redirect_stdout(io.StringIO()):
        migrate_database(url)
    with DatabaseMapping(url) as db:
        assert _values(db, "invest_solve_branching_method") == written


def test_head_refresh_adds_the_missing_v84_definition(v83_db: Path, tmp_path: Path) -> None:
    url, _printed = _migrate_copy(v83_db, tmp_path)
    path = Path(url.removeprefix("sqlite:///"))
    with DatabaseMapping(url) as db:
        (pdef,) = db.get_items("parameter_definition", entity_class_name="solve",
                               name="invest_solve_branching_method")
        db.remove_item("parameter_definition", pdef["id"])
        db.commit_session("a database at 84 without the v84 definition")
    assert head._head_schema_missing_ro(path)
    with contextlib.redirect_stdout(io.StringIO()):
        migrate_database(url)
    assert not head._head_schema_missing_ro(path)
    assert runner._current_version_ro(path, None) == FLEXTOOL_DB_VERSION
