"""Databases shipped by old releases migrate cleanly (item 291).

Seeds: the definitions-only raw rows of four shipped databases (``data/release_seeds``, written by
``tools/make_release_seeds.py`` from the release tags; rebuilt here with ``sqlite3`` in ``tmp_path``).  What each one
carries that a synthetic seed cannot (plan §5): v3.0.0 has no ``model.version`` and the old ``object_class`` tables;
v3.10.0 and v3.22.0 store list defaults as their member's text (item 279) and ``b'null'`` defaults; v3.33.0 has lost
(``NULL``) list defaults and no ``none`` member in ``minimum_time_methods``.  None has an entity or a value.

Expected values: the facts above are read from the shipped files; the lists at 83 are the members the v83 parts
append or remove (``test_v83_*``); the part 1 rows are V4's ``CASES`` of ``test_v83_minimum_time_method`` (derived
there by hand: the seed's own scenarios hold no value of the added unit, so they change nothing).
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping

from tests.update_flextool._raw_seed import DATA_TABLES, build, read_seed
from tests.update_flextool._rerun import (Kill, already_applied_records, commit_messages, copy_db, diff,
                                          kill_after_commit, migrate, restamp_file, snapshot, url_of, version_of)
from tests.update_flextool.test_v83_dispatch_realization_method import _branches
from tests.update_flextool.test_v83_minimum_time_method import CASES, _fill, _marker, _methods

SEEDS = Path(__file__).parent / "data" / "release_seeds"
#: seed -> (its ``model.version`` as stored, None without one; definitions; an alternative named ``Base``; ``none``
#: in ``minimum_time_methods`` at 82: v27 adds it to the two older seeds, v3.22.0 ships it, v3.33.0 lost it and no
#: step before 83 adds it back).
TRAITS = {
    "v3_0_0": (None, 176, False, True),
    "v3_10_0": (b"22", 203, False, True),
    "v3_22_0": (b"28", 211, False, True),
    "v3_33_0": (b"51", 227, True, False),
}
#: seed -> (list defaults stored as their member's text, list defaults lost (``NULL``), how ``unit.startup_method``'s
#: default is stored): the plan §5 table, from the shipped files.
LIST_DEFAULTS = {
    "v3_0_0": (0, 18, "member id"),
    "v3_10_0": (31, 9, "text"),
    "v3_22_0": (30, 10, "text"),
    "v3_33_0": (0, 38, "lost"),
}
#: Part 1 rows added at 82 (plan §5): the M1 rows of C rev 3 §10.3 that the release seeds must also pass.
M1_ROWS = ("plain", "ft2_off", "explicit_some_scenarios", "masking_conflict", "pw_auto", "default_time")
#: The lists at 83 (V4: part 1 appends ``none``, part 2 the three startup members without ``binary``, part 5
#: ``deterministic``; part 4 deletes ``solver``).
LISTS_AT_83 = {
    "minimum_time_methods": ["min_uptime", "min_downtime", "both", "none"],
    "startup_methods": ["no_startup", "linear", "linear_fix_in_uc", "integer", "integer_fix_in_uc"],
    "dispatch_realization_methods": ["within_solve", "separate_dispatch", "deterministic"],
}


def _rows(seed: dict, table: str) -> list[dict]:
    columns = seed["tables"][table]["columns"]
    return [dict(zip(columns, row)) for row in seed["tables"][table]["rows"]]


def _payload(cell) -> bytes | None:
    return base64.b64decode(cell["b64"]) if isinstance(cell, dict) else cell


def _list_default(row: dict) -> str | None:
    """How a definition bound to a value list stores its default: ``member id``, ``text`` (item 279), ``lost``."""
    if row["parameter_value_list_id"] is None:
        return None
    if row["default_value"] is None:
        return "lost"
    if row["default_type"] == "list_value_ref":
        return "member id" if _payload(row["default_value"]).isdigit() else "text"
    return "other"


def _scenarios_holding(seed: dict, alternative: str) -> set[str]:
    """The scenarios of *seed* that hold *alternative* (v3.33.0 ships a ``Base`` in none of them)."""
    alt_ids = {a["id"] for a in _rows(seed, "alternative") if a["name"] == alternative}
    names = {s["id"]: s["name"] for s in _rows(seed, "scenario")}
    return {names[r["scenario_id"]] for r in _rows(seed, "scenario_alternative") if r["alternative_id"] in alt_ids}


def _members(db, name: str) -> list:
    return [x["parsed_value"] for x in sorted(db.get_items("list_value", parameter_value_list_name=name),
                                              key=lambda x: x["index"])]


@pytest.fixture(scope="module", params=sorted(TRAITS))
def release(request, tmp_path_factory) -> dict:
    """``{"name", "seed", "built", 82, 83, "records"}``: the seed rebuilt, migrated to 82, and migrated to 83 in one
    run (with the ``_already_applied`` records of that run)."""
    name = request.param
    folder = tmp_path_factory.mktemp(name)
    seed = read_seed(SEEDS / f"{name}.json")
    built = build(seed, folder / "built.sqlite")
    clean = copy_db(built, folder / "k83.sqlite")
    with already_applied_records() as records:
        migrate(clean)
    k82 = copy_db(built, folder / "k82.sqlite")
    migrate(k82, up_to=82)
    return {"name": name, "seed": seed, "built": built, 82: k82, 83: clean, "records": records}


def test_the_seed_is_the_shipped_definitions(release) -> None:
    """The stamp and the definition count of the shipped file, no entity or value, the commits kept with no user."""
    stamp, definitions, has_base, _none = TRAITS[release["name"]]
    seed = release["seed"]
    versions = [d["default_value"] for d in _rows(seed, "parameter_definition") if d["name"] == "version"]
    assert versions == ([] if stamp is None else [{"b64": base64.b64encode(stamp).decode()}])
    assert len(seed["tables"]["parameter_definition"]["rows"]) == definitions
    assert all(not seed["tables"][t]["rows"] for t in DATA_TABLES if t in seed["tables"])
    commits = _rows(seed, "commit")
    assert commits and {c["user"] for c in commits} == {""} and all(c["date"] for c in commits)
    assert any(a["name"] == "Base" for a in _rows(seed, "alternative")) == has_base
    classes = {c["id"]: c["name"] for c in _rows(seed, "entity_class")}
    kinds = {(classes[d["entity_class_id"]], d["name"]): _list_default(d) for d in _rows(seed, "parameter_definition")}
    text, lost, startup = LIST_DEFAULTS[release["name"]]
    assert (sum(k == "text" for k in kinds.values()), sum(k == "lost" for k in kinds.values())) == (text, lost)
    assert kinds[("unit", "startup_method")] == startup
    if release["name"] == "v3_33_0":
        assert all(kinds[(cls, "invest_method")] == "lost" for cls in ("unit", "node", "connection", "group"))


def test_a_clean_migration_reaches_the_current_lists(release) -> None:
    """One run to 83: nothing found already applied, the v83 lists, no ``solver`` list, no ``Base`` made where there
    is none (no entity: nothing for v56 to backfill), a shipped ``Base`` with its own scenario rows (v3.33.0: none)
    and no value."""
    path = release[83]
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert release["records"] == []
    with open_database_mapping(url_of(path)) as db:
        assert {name: _members(db, name) for name in LISTS_AT_83} == LISTS_AT_83
        assert not db.get_item("parameter_value_list", name="solver")
        assert bool(db.get_item("alternative", name="Base")) == TRAITS[release["name"]][2]
        held = {r["scenario_name"] for r in db.get_items("scenario_alternative", alternative_name="Base")}
        assert not db.find_parameter_values(alternative_name="Base")
    assert held == _scenarios_holding(release["seed"], "Base")


def test_minimum_time_methods_at_82(release) -> None:
    with open_database_mapping(url_of(release[82])) as db:
        assert ("none" in _members(db, "minimum_time_methods")) == TRAITS[release["name"]][3]


def test_staged_migration_equals_the_one_shot(release, tmp_path) -> None:
    path = copy_db(release[82], tmp_path / "db.sqlite")
    migrate(path)
    assert diff(snapshot(release[83]), snapshot(path)) == []


def test_v83_rerun_on_its_own_result_converges(release, tmp_path) -> None:
    """Content 83 under stamp 82 (a hand edit, or a kill after the v83 commit) migrated again equals content 83."""
    path = copy_db(release[83], tmp_path / "db.sqlite")
    restamp_file(path, 82)
    migrate(path)
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert diff(snapshot(release[83]), snapshot(path)) == []


def test_a_kill_after_the_v83_commit_converges(release, tmp_path) -> None:
    """The v83 step commits once, then the version commit: a kill right after the first leaves stamp 82, and the
    rerun converges."""
    assert len(commit_messages(release[82], 83)) == 2
    path = copy_db(release[82], tmp_path / "db.sqlite")
    with pytest.raises(Kill), kill_after_commit(1):
        migrate(path)
    assert version_of(path) == 82
    migrate(path)
    assert diff(snapshot(release[83]), snapshot(path)) == []


def _with_base_alternative(path: Path) -> None:
    """The rows' alternative ``base`` (the release seeds have none; ``_fill`` expects it)."""
    with open_database_mapping(url_of(path)) as db:
        db.add_alternative(name="base")
        db.commit_session("base")


def _m1_row(k82: Path, folder: Path, row: str) -> list:
    """The M1 *row* added to a copy of *k82* and migrated to 83: what differs from the row's expected methods,
    marker and unchanged ``startup_method`` values (empty when all match)."""
    values, scenarios, defaults, writes, undecided, _db = CASES[row]
    path = copy_db(k82, folder / f"{row}.sqlite")
    _with_base_alternative(path)
    with DatabaseMapping(url_of(path)) as db:
        _fill(db, {"u": values}, scenarios, defaults)
        own = _methods(db)
        startups = _startups(db)
        db.commit_session(f"M1 row {row}")
    migrate(path)
    expected = {"since": 83, "writes": [["u", a, method] for a, method in sorted(writes.items())],
                "undecided": [] if undecided is None else [["u", undecided[0], list(undecided[1])]]}
    with DatabaseMapping(url_of(path)) as db:
        found = (_methods(db), _marker(db), _startups(db))
    wanted = (own | {("u", a): method for a, method in writes.items()}, expected if writes or undecided else None,
              startups)
    return [] if found == wanted else [(row, wanted, found)]


def _startups(db) -> dict:
    return {(pv["entity_byname"], pv["alternative_name"]): pv["parsed_value"] for pv in
            db.find_parameter_values(entity_class_name="unit", parameter_definition_name="startup_method")}


def test_part_1_rows_added_at_82(release, tmp_path) -> None:
    """A unit of each M1 row added to the seed at 82 gets the row's ``minimum_time_method`` writes and marker at
    83; its ``startup_method`` values stay (linear and no_startup are not renamed).  A row with an explicit ``none``
    is left out where the list has no ``none`` at 82 (it cannot be stored; ``test_minimum_time_methods_at_82``)."""
    rows = [row for row in M1_ROWS if TRAITS[release["name"]][3]
            or "none" not in CASES[row][0].get("minimum_time_method", {}).values()]
    assert len(rows) == (6 if TRAITS[release["name"]][3] else 4)
    assert [failure for row in rows for failure in _m1_row(release[82], tmp_path, row)] == []


def test_a_stochastic_solve_without_a_method_gets_within_solve(release, tmp_path) -> None:
    """Part 5: a solve whose ``stochastic_branches`` in ``base`` wins in S, with no method, gets ``within_solve`` in
    ``base``; the definition default becomes ``deterministic``."""
    path = copy_db(release[82], tmp_path / "db.sqlite")
    _with_base_alternative(path)
    with DatabaseMapping(url_of(path)) as db:
        db.add_item("scenario", name="S")
        db.add_item("scenario_alternative", scenario_name="S", alternative_name="base", rank=1)
        db.add_item("entity", entity_class_name="solve", name="st")
        raw, typ = to_database(_branches())
        db.add_item("parameter_value", entity_class_name="solve", entity_byname=("st",),
                    parameter_definition_name="stochastic_branches", alternative_name="base", value=raw, type=typ)
        db.commit_session("stochastic solve")
    migrate(path)
    with DatabaseMapping(url_of(path)) as db:
        assert {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"] for pv in db.find_parameter_values(
            entity_class_name="solve", parameter_definition_name="dispatch_realization_method")} == {
            ("st", "base"): "within_solve"}
        assert db.get_item("parameter_definition", entity_class_name="solve",
                           name="dispatch_realization_method")["parsed_value"] == "deterministic"
