"""A database from a shipped file before the SpineDB entity schema (v3.0.0 .. v3.5.1) ends with the shipped
``active_by_default`` of ``constraint`` and the two reserve classes, and keeps which of their entities are on.

spinedb_api upgrades such a file when the migration opens it, and its first commit writes ``active_by_default`` 0 on
every class with an ``is_active`` definition whose default is not ``yes`` (and turns the ``is_active`` values into
``entity_alternative`` rows) behind the open session.  The migration session must read that, and v56 must keep "no
``is_active`` = off" across switching the classes on.

Expected values, by hand:

* the shipped schema (``flextool/schemas/spinedb_schema.json``): ``constraint``, ``reserve__upDown__unit__node`` and
  ``reserve__upDown__connection__node`` active by default; ``node``, ``unit``, ``connection`` not;
* v3.0.0 ``docs/reference.md``: ``is_active`` "(if not defined, then not active)", for the reserves "Empty indicates
  not allowed"; ``docs/index.md``: "``is_active`` must be defined in order to include the constraint";
* v56 (``_migrate_v56_reactivate_is_enabled_parameter``): an ``is_active`` value becomes ``is_enabled`` in its
  alternative; an entity of a class that v56 switches on gets ``Base`` ``is_enabled = no`` when it has no Base row;
  no ``entity_alternative`` row of the three classes is left; ``Base`` (made, the seed has none) is one rank below
  the lowest of every scenario (``_v56_ensure_base``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping

from tests.update_flextool._raw_seed import build, read_seed
from tests.update_flextool._rerun import diff, migrate, snapshot, url_of

SEED = Path(__file__).parent / "data" / "release_seeds" / "v3_0_0.json"
#: The shipped schema's ``active_by_default`` (``spinedb_schema.json``).
SHIPPED = {"constraint": True, "reserve__upDown__unit__node": True, "reserve__upDown__connection__node": True,
           "node": False, "unit": False, "connection": False}
#: Ids of the v3.0.0 seed: classes, the ``is_active`` definitions, its ``yes`` member, the alternative ``reserve``.
CLASS = {"node": 12, "unit": 13, "connection": 34, "reserve": 3, "upDown": 46, "constraint": 63,
         "reserve__upDown__unit__node": 54, "reserve__upDown__connection__node": 53}
IS_ACTIVE = {"constraint": 279, "reserve__upDown__unit__node": 254, "reserve__upDown__connection__node": 293}
YES_MEMBER_ID = b"45"
ALTERNATIVE_RESERVE = 293
COMMIT = 879
R_UP = ("primary", "up")


def _add_data(path: Path) -> None:
    """Two constraints, two unit reserves and two connection reserves: ``*_on`` with ``is_active = yes`` in
    ``reserve``, ``*_off`` with no ``is_active``.  Written into the pre-entity tables, as a v3.0.0 file holds them."""
    con = sqlite3.connect(path)
    try:
        ids = iter(range(9001, 9100))

        def obj(cls: str, name: str) -> int:
            eid = next(ids)
            con.execute("INSERT INTO entity (id, type_id, class_id, name, description, commit_id) "
                        "VALUES (?, 1, ?, ?, NULL, ?)", (eid, CLASS[cls], name, COMMIT))
            con.execute("INSERT INTO object (entity_id, type_id) VALUES (?, 1)", (eid,))
            return eid

        def rel(cls: str, name: str, members: list[tuple[int, str]]) -> int:
            eid = next(ids)
            con.execute("INSERT INTO entity (id, type_id, class_id, name, description, commit_id) "
                        "VALUES (?, 2, ?, ?, NULL, ?)", (eid, CLASS[cls], name, COMMIT))
            con.execute("INSERT INTO relationship (entity_id, entity_class_id, type_id) VALUES (?, ?, 2)",
                        (eid, CLASS[cls]))
            for dimension, (member, member_cls) in enumerate(members):
                con.execute("INSERT INTO relationship_entity (entity_id, entity_class_id, dimension, member_id, "
                            "member_class_id) VALUES (?, ?, ?, ?, ?)",
                            (eid, CLASS[cls], dimension, member, CLASS[member_cls]))
            return eid

        def is_active_yes(cls: str, eid: int) -> None:
            con.execute("INSERT INTO parameter_value (id, parameter_definition_id, entity_id, entity_class_id, type, "
                        "value, commit_id, alternative_id) VALUES (?, ?, ?, ?, 'list_value_ref', ?, ?, ?)",
                        (next(ids), IS_ACTIVE[cls], eid, CLASS[cls], YES_MEMBER_ID, COMMIT, ALTERNATIVE_RESERVE))

        reserve, up = obj("reserve", R_UP[0]), obj("upDown", R_UP[1])
        unit, node, connection = obj("unit", "u1"), obj("node", "n1"), obj("connection", "c1")
        for state in ("on", "off"):
            constraint = obj("constraint", f"c_{state}")
            unit_reserve = rel("reserve__upDown__unit__node", f"ru_{state}",
                               [(reserve, "reserve"), (up, "upDown"), (unit, "unit"), (node, "node")])
            connection_reserve = rel("reserve__upDown__connection__node", f"rc_{state}",
                                     [(reserve, "reserve"), (up, "upDown"), (connection, "connection"),
                                      (node, "node")])
            if state == "on":
                is_active_yes("constraint", constraint)
                is_active_yes("reserve__upDown__unit__node", unit_reserve)
                is_active_yes("reserve__upDown__connection__node", connection_reserve)
        con.commit()
    finally:
        con.close()


def _step_by_step(path: Path) -> None:
    """Each step in a migration of its own (the file reopened after each)."""
    for version in range(1, FLEXTOOL_DB_VERSION + 1):
        migrate(path, up_to=version)


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> dict:
    """``{"one_shot": path, "step": path}``: the v3.0.0 seed with :func:`_add_data`, migrated both ways."""
    folder = tmp_path_factory.mktemp("pre_entity")
    seed = read_seed(SEED)
    out = {}
    for name, run in (("one_shot", migrate), ("step", _step_by_step)):
        path = build(seed, folder / f"{name}.sqlite")
        _add_data(path)
        run(path)
        out[name] = path
    return out


@pytest.mark.parametrize("path_name", ["one_shot", "step"])
def test_the_classes_end_active_by_default_as_shipped(migrated, path_name) -> None:
    with open_database_mapping(url_of(migrated[path_name])) as db:
        assert {c: db.get_item("entity_class", name=c)["active_by_default"] for c in SHIPPED} == SHIPPED


@pytest.mark.parametrize("path_name", ["one_shot", "step"])
def test_the_entities_without_is_active_stay_off(migrated, path_name) -> None:
    """``*_on``: ``yes`` in ``reserve``, ``no`` in Base; ``*_off``: ``no`` in Base.  So in a scenario with
    ``reserve`` only the ``*_on`` entities are on; in one without it, none."""
    expected = {}
    for cls, prefix in (("constraint", "c"), ("reserve__upDown__unit__node", "ru"),
                        ("reserve__upDown__connection__node", "rc")):
        for state in ("on", "off"):
            byname = (f"{prefix}_{state}",) if cls == "constraint" else (
                R_UP[0], R_UP[1], "u1" if prefix == "ru" else "c1", "n1")
            expected[(cls, byname, "Base")] = "no"
            if state == "on":
                expected[(cls, byname, "reserve")] = "yes"
    with open_database_mapping(url_of(migrated[path_name])) as db:
        found = {(pv["entity_class_name"], tuple(pv["entity_byname"]), pv["alternative_name"]): pv["parsed_value"]
                 for cls in SHIPPED for pv in db.find_parameter_values(entity_class_name=cls,
                                                                        parameter_definition_name="is_enabled")}
        assert found == expected
        assert [ea for cls in SHIPPED if cls in IS_ACTIVE for ea in db.find_entity_alternatives(
            entity_class_name=cls)] == []
        ranks: dict[str, list] = {}
        for sa in db.get_items("scenario_alternative"):
            ranks.setdefault(sa["scenario_name"], []).append((sa["rank"], sa["alternative_name"]))
        assert ranks and all(min(r)[1] == "Base" for r in ranks.values())


def test_one_shot_equals_step_by_step(migrated) -> None:
    assert diff(snapshot(migrated["step"]), snapshot(migrated["one_shot"])) == []
