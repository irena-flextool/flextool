"""Data version 83: the step ``_migrate_v83_schema_batch`` (parts 0, 1, 2 and 5) and the refresh of a database already
at 83 (plan §0 "Head-step rules at 83", §7 V1b).

Databases: the v82 schema (``v82_schema_db_url``: the bundled v25 template migrated to 82), the current schema
(``schema_db_url``, at 83), v75b (``_rerun.build_v75b``) and ``tests.json`` data imported into the v82 schema; all
built in ``tmp_path``.  Expected values come from the plan's rules: the members appended at the end in their plan order, the
shipped default ``deterministic``, the description texts of plan §2 (``test_v83_startup_method.NEW``, copied from
the plan), the hash of the empty-default names as ``list_default_restore.empty_hash`` defines it (sha1 of the sorted
names, newline-joined, first 12 hex), and the T1 model of C rev 3 §10.3 worked out by hand at the test.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import shutil
import sqlite3
from pathlib import Path

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, import_data, to_database
from spinedb_api.exception import NothingToCommit

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.fixture_schema import unpack_value

from tests.update_flextool._rerun import v75b_url
from tests.update_flextool._value_list_cases import add_commit, make_lost_defaults, repair_part_b, sql
from tests.update_flextool.test_v83_startup_method import NEW, OLD_UNIT_STARTUP

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SCHEMA = Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "spinedb_schema.json"
STARTUP_MEMBERS = ["no_startup", "linear", "linear_fix_in_uc", "integer", "integer_fix_in_uc"]
DISPATCH_MEMBERS = ["within_solve", "separate_dispatch", "deterministic"]
#: The ``model.discount_offset_operations`` description a development build of v82 shipped (no 0-1 range).
DEV_V82_OPERATIONS_TEXT = (
    "[years] Where in each year operating and fixed costs are weighed by social_discount_rate, counted from the start "
    "of the year: 0.5 (default) the middle of the year, standing for costs spread over the year. No effect when "
    "social_discount_rate is 0. Constant.")


def _shipped_texts() -> dict[tuple[str, str], str]:
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return {(r[0], r[1]): r[4] for r in data["parameter_definitions"]}


def _copy(url: str, folder: Path, name: str = "db.sqlite") -> str:
    path = folder / name
    shutil.copy2(url.removeprefix("sqlite:///"), path)
    return f"sqlite:///{path.resolve().as_posix()}"


def _path(url: str) -> Path:
    return Path(url.removeprefix("sqlite:///"))


def _migrate(url: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        migrate_database(url)
    return out.getvalue()


def _raw(url: str, statement: str, *args) -> None:
    con = sqlite3.connect(_path(url))
    try:
        con.execute(statement, args)
        con.commit()
    finally:
        con.close()


def _members(db, name: str) -> list:
    return [x["parsed_value"] for x in sorted(db.get_items("list_value", parameter_value_list_name=name),
                                               key=lambda x: x["index"])]


def _definition(db, cls: str, name: str) -> dict:
    return db.get_item("parameter_definition", entity_class_name=cls, name=name)


def _add_values(db) -> None:
    """A unit and a solve with values on the three lists v83 changes (none names a member a test removes)."""
    for cls, name, param, member in (("unit", "u1", "startup_method", "linear"),
                                     ("unit", "u1", "minimum_time_method", "min_uptime"),
                                     ("solve", "s1", "dispatch_realization_method", "separate_dispatch")):
        if not db.get_item("entity", entity_class_name=cls, name=name):
            db.add_entity(entity_class_name=cls, name=name)
        raw, typ = to_database(member)
        db.add_parameter_value(entity_class_name=cls, entity_byname=(name,), parameter_definition_name=param,
                               alternative_name="base", value=raw, type=typ)


def _values(db) -> dict:
    return {(x["entity_class_name"], x["entity_byname"], x["parameter_definition_name"], x["alternative_name"]):
            x["parsed_value"] for x in db.get_items("parameter_value")}


def _commit_count(url: str) -> int:
    con = sqlite3.connect(_path(url))
    try:
        return con.execute("SELECT COUNT(*) FROM \"commit\"").fetchone()[0]
    finally:
        con.close()


def _assert_second_refresh_commits_nothing(url: str, values_before: dict) -> None:
    """The values are unchanged, a direct second refresh has nothing to commit, and a second open adds no commit."""
    with DatabaseMapping(url) as db:
        assert _values(db) == values_before
        with pytest.raises(NothingToCommit):
            m._refresh_head_schema(db)
            db.commit_session("rerun")
    commits = _commit_count(url)
    assert "already up-to-date at version 83" in _migrate(url)
    assert _commit_count(url) == commits


# ─── v25 → 83 ────────────────────────────────────────────────────────────


def test_v25_to_83_lists_defaults_and_texts(v82_schema_db_url: str, tmp_path: Path) -> None:
    """The v25 template migrated to 82 and then to 83: no description names ``binary`` (CR-1 B1), the members and
    the defaults of plan §2 / §2a, and the file is current."""
    url = _copy(v82_schema_db_url, tmp_path)
    _migrate(url)
    with DatabaseMapping(url) as db:
        assert _definition(db, "model", "version")["parsed_value"] == 83.0
        assert [(d["entity_class_name"], d["name"]) for d in db.get_items("parameter_definition")
                if "binary" in (d["description"] or "")] == []
        assert _members(db, "startup_methods") == STARTUP_MEMBERS
        assert _members(db, "dispatch_realization_methods") == DISPATCH_MEMBERS
        assert _members(db, "minimum_time_methods") == ["min_uptime", "min_downtime", "both", "none"]
        assert _definition(db, "solve", "dispatch_realization_method")["parsed_value"] == "deterministic"
        assert _definition(db, "unit", "startup_method")["parsed_value"] == "no_startup"
        for key, text in NEW.items():
            assert _definition(db, *key)["description"] == text, key
    assert m._current_version_ro(_path(url), None) == FLEXTOOL_DB_VERSION


@pytest.mark.parametrize("default", ["null", "binary"])
def test_v25_to_83_startup_method_default(v82_schema_db_url: str, tmp_path: Path, default: str) -> None:
    """A ``NULL`` and a ``binary`` ``unit.startup_method`` default both end with the new description (by text alone,
    CR-1 B2); ``NULL`` stays ``NULL``, ``binary`` becomes ``integer_fix_in_uc``."""
    url = _copy(v82_schema_db_url, tmp_path)
    if default == "null":
        _raw(url, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL WHERE name = "
                  "'startup_method' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')")
    else:
        with DatabaseMapping(url) as db:
            raw, typ = to_database("binary")
            db.update_item("parameter_definition", id=_definition(db, "unit", "startup_method")["id"],
                           default_value=raw, default_type=typ)
            db.commit_session("binary default")
    with DatabaseMapping(url) as db:
        assert _definition(db, "unit", "startup_method")["description"] == OLD_UNIT_STARTUP
    _migrate(url)
    with DatabaseMapping(url) as db:
        pdef = _definition(db, "unit", "startup_method")
        assert pdef["description"] == NEW[("unit", "startup_method")]
        assert pdef["parsed_value"] == (None if default == "null" else "integer_fix_in_uc")


def test_v75_seed_to_83_writes_within_solve(tmp_path: Path, tmp_path_factory) -> None:
    """v75b (a stochastic solve before v76): ``stoch_2p`` has ``stochastic_branches`` in alternative
    ``stoch`` (scenarios stoch and stoch_horizon; not det) and no method, so ``within_solve`` is written into
    ``stoch`` and nowhere else; the ``within_solve`` default becomes ``deterministic``."""
    url = v75b_url(tmp_path_factory, tmp_path / "seed.sqlite")
    printed = _migrate(url)
    with DatabaseMapping(url) as db:
        written = {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
                   for pv in db.find_parameter_values(entity_class_name="solve",
                                                      parameter_definition_name="dispatch_realization_method")}
        assert written == {("stoch_2p", "stoch"): "within_solve"}
        assert _definition(db, "solve", "dispatch_realization_method")["parsed_value"] == "deterministic"
        assert _members(db, "dispatch_realization_methods") == DISPATCH_MEMBERS
    assert "  solve stoch_2p [stoch]\n" in printed


def test_v75_seed_to_83_lists_no_realization_notice(tmp_path: Path, tmp_path_factory) -> None:
    """v75b → 83, part 3: ``stoch_2p`` gets ``within_solve`` from part 5 (above), but its ``solve_mode`` is
    ``single_solve`` (alternative ``init``, in every scenario), so it is not rolling: a single ``within_solve`` solve
    keeps its separate realization and W-0635 does not list it; no solve has ``separate_dispatch``, so W-0636 lists
    nothing.  Neither marker is written (a marker is written only when its list is non-empty)."""
    url = v75b_url(tmp_path_factory, tmp_path / "seed.sqlite")
    _migrate(url)
    names = {f"flextool_behaviour_change:{m.V83_WITHIN_SOLVE_MARKER}",
             f"flextool_behaviour_change:{m.V83_SEPARATE_DISPATCH_MARKER}"}
    with DatabaseMapping(url) as db:
        assert db.get_item("parameter_value", entity_class_name="solve", entity_byname=("stoch_2p",),
                           parameter_definition_name="solve_mode", alternative_name="init")["parsed_value"] \
            == "single_solve"
        assert [md["name"] for md in db.get_items("metadata") if md["name"] in names] == []


# ─── refresh of a database at 83 ─────────────────────────────────────────


def test_old_texts_at_83_are_refreshed_once(schema_db_url: str, tmp_path: Path, monkeypatch) -> None:
    """A database at 83 carrying an old v83 text (``unit.startup_method``, with a ``NULL`` default) and a superseded
    v82 text (``model.discount_offset_operations``) is not current; one open replaces both texts (the default stays
    ``NULL``, an edited text is kept, the values are unchanged), and a second open is current, takes no lock and
    commits nothing (S5)."""
    url = _copy(schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        for (cls, name), text in ((("unit", "startup_method"), OLD_UNIT_STARTUP),
                                  (("model", "discount_offset_operations"), DEV_V82_OPERATIONS_TEXT),
                                  (("connection", "startup_method"), "my own text")):
            db.update_parameter_definition(id=_definition(db, cls, name)["id"], description=text)
        _add_values(db)
        db.commit_session("old texts")
        values_before = _values(db)
    _raw(url, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL WHERE name = "
              "'startup_method' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')")
    path = _path(url)
    assert m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) is None
    _migrate(url)
    shipped = _shipped_texts()
    with DatabaseMapping(url) as db:
        unit = _definition(db, "unit", "startup_method")
        assert unit["description"] == NEW[("unit", "startup_method")] and unit["parsed_value"] is None
        assert (_definition(db, "model", "discount_offset_operations")["description"]
                == shipped[("model", "discount_offset_operations")] != DEV_V82_OPERATIONS_TEXT)
        assert _definition(db, "connection", "startup_method")["description"] == "my own text"
    assert not m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION

    def no_lock(*_args, **_kwargs):
        raise AssertionError("a current database took the migration lock")

    # The package re-exports a function of the module's name: patch the module itself.
    monkeypatch.setattr(importlib.import_module("flextool.update_flextool.migrate_before_run"), "migration_lock",
                        no_lock)
    _assert_second_refresh_commits_nothing(url, values_before)


@pytest.mark.parametrize("name, member", [("startup_methods", "integer"), ("minimum_time_methods", "both"),
                                          ("dispatch_realization_methods", "deterministic")])
def test_missing_member_at_83_is_not_current(schema_db_url: str, tmp_path: Path, name: str, member: str) -> None:
    """A database stamped 83 without a member v83 adds is not current (N3); the refresh appends it (no value or
    default is written: a ``within_solve`` default stays, the values are unchanged), and a second refresh commits
    nothing."""
    url = _copy(schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        if name == "dispatch_realization_methods":
            raw, typ = to_database("within_solve")
            db.update_item("parameter_definition", id=_definition(db, "solve", "dispatch_realization_method")["id"],
                           default_value=raw, default_type=typ)
        before = _members(db, name)
        (item,) = [x for x in db.get_items("list_value", parameter_value_list_name=name) if x["parsed_value"] == member]
        db.remove_item("list_value", item["id"])
        _add_values(db)
        db.commit_session(f"no {member}")
        values_before = _values(db)
    path = _path(url)
    assert m._head_schema_missing_ro(path)
    assert m._current_version_ro(path, None) is None
    _migrate(url)
    with DatabaseMapping(url) as db:
        assert _members(db, name) == [x for x in before if x != member] + [member]
        if name == "dispatch_realization_methods":
            assert _definition(db, "solve", "dispatch_realization_method")["parsed_value"] == "within_solve"
    assert not m._head_schema_missing_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION
    _assert_second_refresh_commits_nothing(url, values_before)


def test_missing_v82_definition_at_83_is_added(schema_db_url: str, tmp_path: Path) -> None:
    """A database at 83 without ``unit.life_extension_cost`` (a v82 definition) is not current, and the materialized
    file is stale; the refresh adds the definition with its shipped fields and keeps an edited description."""
    from flextool.update_flextool.canonical_databases import _stale_version

    url = _copy(schema_db_url, tmp_path)
    path = _path(url)
    assert _stale_version(path) is None
    with DatabaseMapping(url) as db:
        db.remove_item("parameter_definition", _definition(db, "unit", "life_extension_cost")["id"])
        db.update_parameter_definition(id=_definition(db, "node", "life_extension_cost")["id"],
                                       description="my own text")
        db.commit_session("an older v83 database")
    assert m._head_schema_missing_ro(path) and _stale_version(path) == FLEXTOOL_DB_VERSION
    _migrate(url)
    with DatabaseMapping(url) as db:
        unit = _definition(db, "unit", "life_extension_cost")
        assert (unit["description"], tuple(unit["parameter_type_list"]), unit["parameter_group_name"],
                unit["default_value"]) == (_shipped_texts()[("unit", "life_extension_cost")], ("float", "1d_map"),
                                           "investment", None)
        assert _definition(db, "node", "life_extension_cost")["description"] == "my own text"
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION and _stale_version(path) is None


# ─── repair part B across the upgrade ────────────────────────────────────


def test_part_b_record_stays_valid_across_the_upgrade(v82_schema_db_url: str, tmp_path: Path) -> None:
    """A v82 database with four empty list defaults whose part B repair (by FlexTool 4.6.3) kept all four empty: its
    record carries the hash of those names.  Migrated to 83 the defaults stay empty (parts 2 and 5 leave a ``NULL``
    default), so the hash is the same and W-0491 stays off; the repair tool then runs at 83 and restores the shipped
    members, ``deterministic`` for ``dispatch_realization_method``."""
    names = ["solve.dispatch_realization_method", "solve.solve_mode", "unit.conversion_method",
             "unit.startup_method"]
    case = make_lost_defaults(tmp_path, names, schema_db=v82_schema_db_url)
    empty = hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]
    from flextool.update_flextool.value_list_history import REPAIR_COMMIT
    commit = add_commit(case.path, REPAIR_COMMIT, "2026-11-23")
    sql(case.path, "INSERT INTO metadata (name, value, commit_id) VALUES (?, ?, ?)",
        (ldr.PART_B_RECORD_KEY, ldr.part_b_record_value("2026-11-23", 0, 4, 0, empty), commit))
    url = f"sqlite:///{case.path.resolve().as_posix()}"
    _migrate(url)
    with DatabaseMapping(url) as db:
        assert _definition(db, "model", "version")["parsed_value"] == 83.0
    with contextlib.closing(sqlite3.connect(case.path)) as con:
        assert ldr.empty_shipped_defaults(con) == sorted(names)
        assert ldr.parse_part_b_record(con).empty == empty
    assert ldr.list_defaults_lost_ro(case.path) is None
    repaired = repair_part_b(case.path, tmp_path / "out" / "repaired.sqlite")
    with DatabaseMapping(f"sqlite:///{repaired.resolve().as_posix()}") as db:
        assert {n: _definition(db, *n.split("."))["parsed_value"] for n in names} == {
            "solve.dispatch_realization_method": "deterministic", "solve.solve_mode": "single_solve",
            "unit.conversion_method": "constant_efficiency", "unit.startup_method": "no_startup"}
    with contextlib.closing(sqlite3.connect(repaired)) as con:
        assert ldr.empty_shipped_defaults(con) == []
    assert ldr.list_defaults_lost_ro(repaired) is None


# ─── C M2: a migrated database keeps its result ──────────────────────────


@pytest.mark.solver
def test_migrated_min_downtime_unit_keeps_its_result(v82_schema_db_url: str, tmp_path: Path,
                                                     tmp_path_factory) -> None:
    """C rev 3 §10.3 M2 on ``tests.json`` data imported into the v82 schema: ``coal_plant`` (one 500 MW unit,
    efficiency 0.4, coal 20 €/MWh, so 50 €/MWh of fuel; ``west``: penalty_up 900, penalty_down 800 €/MWh) with
    ``startup_method linear``, ``min_load 0.5``, ``min_downtime 2`` and no method, demand 500 / 0 / 500 MW over three
    1 h steps.  Scenario ``m2_base``, and ``m2_off`` with ``no_startup`` on top.

    Base: t1 and t3 need the whole unit online (unmet demand costs 900 > 50).  Staying online at t2 (share x) costs
    250·x MW dumped at 800 + fuel 50, so 212 500·x; going off (1 - x) keeps it off at t3 (min down time 2 h), with
    500·(1 - x) MW unmet at 900 there.  x = 1: 212 500 + fuel at t3 25 000 = 237 500 < x = 0: 450 000.  So units online
    (1, 1, 1) and flow (500, 250, 500).  Off: no online variables, flow (500, 0, 500).  The upgrade writes
    ``min_downtime`` where ``linear`` is (m2) and ``none`` into ``off`` (C §6.3, the M1 row ``ft2_off``), so the
    migrated results equal the pre-C ones."""
    from tests.engine._unit_db import add_override_alternative, b64, flow, run, time_map, uc, utf8

    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    chain = ["init", "west", "coal"]
    values = uc("coal_plant", 0.5, "linear") + [
        ["unit", "coal_plant", "min_downtime", b64(2.0, "float")],
        ["timeset", "2day", "timeset_duration",
         b64({"index_type": "str", "index_name": "time", "data": [["t0001", 3.0]]}, "map")],
        ["node", "west", "inflow", time_map({"t0001": -500.0, "t0003": -500.0}, 0.0)]]
    base = add_override_alternative(data, "m2", values, chain)
    off = add_override_alternative(data, "off", [["unit", "coal_plant", "startup_method", b64("no_startup", "str")]],
                                   chain + ["m2"])
    alternatives = set(chain) | {"m2", "off"}
    url = _copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db, unparse_value=unpack_value,
            alternatives=[a for a in data["alternatives"] if a[0] in alternatives and a[0] != "Base"],
            scenarios=[s for s in data["scenarios"] if s[0] in (base, off)],
            scenario_alternatives=[s for s in data["scenario_alternatives"] if s[0] in (base, off)],
            entities=data["entities"],
            entity_alternatives=[e for e in data["entity_alternatives"] if e[2] in alternatives],
            parameter_values=[v for v in data["parameter_values"] if v[4] in alternatives])
        assert not errors, errors[:3]
        db.commit_session("tests.json data at 82")
    _migrate(url)
    with DatabaseMapping(url) as db:
        written = {pv["alternative_name"]: pv["parsed_value"]
                   for pv in db.find_parameter_values(entity_class_name="unit", entity_byname=("coal_plant",),
                                                      parameter_definition_name="minimum_time_method")}
    assert written == {"m2": "min_downtime", "off": "none"}

    def flows(steps) -> list[float]:
        (step,) = steps.values()
        return [round(x, 6) + 0.0 for x in flow(step, "coal_plant", "coal_market", "west").to_list()]

    from tests._support.coded_warnings import capture_coded
    with capture_coded() as got:
        base_steps = run(tmp_path_factory, url, base)
    # W-0634 names what the upgrade wrote on the units of the solve.
    assert got.names("W-0634") == ["coal_plant (alternative m2: min_downtime)", "coal_plant (alternative off: none)"]
    online = utf8(next(iter(base_steps.values())).solution.value("v_online_linear"))
    assert [round(x, 6) for x in online.filter(pl.col("p") == "coal_plant").sort("t")["value"]] == [1.0, 1.0, 1.0]
    assert flows(base_steps) == [500.0, 250.0, 500.0]
    assert flows(run(tmp_path_factory, url, off)) == [500.0, 0.0, 500.0]
