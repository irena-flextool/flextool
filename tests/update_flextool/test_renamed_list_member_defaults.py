"""A parameter default that names a value-list member which a migration step renames or removes.

A default refers to its list member.  After an in-place member rename the open database mapping still holds the old
name for the definition, so a later update of that definition in the same migration run failed ("default value ...
is not in the list") and left the database at the older version; after a member removal the reference dangled and
the default later read as whichever member took the freed id.  Each step now clears such a default before its
member change and sets the new member name after.

Every database is built from a JSON template (the bundled v25 template, or the v75 seed for a step above 75) and
migrated up to the version before the step; its definition default is then set to the old member and the database is
migrated to the current version (the default reads the new member, which is in the list), and once more (nothing to
migrate, nothing changes).  The v25 step is checked in one open mapping, as the migration runs it: a later update of
the definition succeeds.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, to_database

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
V25_TEMPLATE = package_data_path("schemas/pre_v26/flextool_template_v25.json")
V2_TEMPLATE = package_data_path("schemas/pre_v26/flextool_template_v2.json")

#: (step, class, parameter, old member, new member).  The new names are the steps' own renames: v55
#: storage_binding_methods, v56 storage_nested_fix_method no -> fix_nothing, v62 lagrangian -> benders /
#: lagrangian_region -> benders_regional, v77 solve_mode single_shot -> single_solve.  (The v80 capacity_margin_methods
#: defaults are in test_v80_limit_inf_descriptions.py.)
CASES = [
    (55, "node", "storage_binding_method", "bind_within_timeset", "bind_within_timeblock"),
    (55, "node", "storage_binding_method", "bind_using_blended_weights", "bind_within_solve_blended_weights"),
    (55, "node", "storage_binding_method", "bind_within_model", "bind_within_solve"),
    (56, "node", "storage_nested_fix_method", "no", "fix_nothing"),
    (62, "solve", "decomposition", "lagrangian", "benders"),
    (62, "group", "decomposition_method", "lagrangian_region", "benders_regional"),
    (77, "solve", "solve_mode", "single_shot", "single_solve"),
]


def _default(db, cls: str, name: str):
    pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
    return from_database(pdef["default_value"], pdef["default_type"])


def _members(db, cls: str, name: str) -> list:
    pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
    return [x["parsed_value"] for x in db.get_items("list_value",
                                                     parameter_value_list_name=pdef["parameter_value_list_name"])]


def _set_default(url: str, cls: str, name: str, value: str) -> None:
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
        assert value in _members(db, cls, name)
        val, typ = to_database(value)
        db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
        db.commit_session(f"{cls}.{name} default {value}")


def _snapshot(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {(d["entity_class_name"], d["name"]): (d["parsed_value"], d["parameter_value_list_name"], d["description"])
                for d in db.get_items("parameter_definition")}


@pytest.mark.parametrize(("step", "cls", "name", "old", "new"), CASES, ids=[f"v{c[0]}-{c[3]}" for c in CASES])
def test_old_member_default_reads_the_new_member(tmp_path_factory, step, cls, name, old, new) -> None:
    path = tmp_path_factory.mktemp(f"v{step}_{old}") / "db.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        if step - 1 >= 75:
            url = json_to_db(SEED, path)
        else:
            initialize_database(str(V25_TEMPLATE), str(path))
            url = "sqlite:///" + str(path)
        migrate_database(url, up_to=step - 1)
    _set_default(url, cls, name, old)
    with contextlib.redirect_stdout(io.StringIO()):
        migrate_database(url)
    with DatabaseMapping(url) as db:
        assert _default(db, "model", "version") == FLEXTOOL_DB_VERSION
        assert _default(db, cls, name) == new
        assert new in _members(db, cls, name) and old not in _members(db, cls, name)
    after = _snapshot(url)
    with contextlib.redirect_stdout(io.StringIO()):
        migrate_database(url)
    assert _snapshot(url) == after


def test_v25_renamed_storage_binding_default_survives_a_later_update(tmp_path_factory) -> None:
    """v25 renames the member ``bind_within_timeblock`` to ``bind_within_timeset``; a later step of the same run that
    updates ``node.storage_binding_method`` (v53 and v55 do) must not fail on the default."""
    path = tmp_path_factory.mktemp("v25") / "db.sqlite"
    url = "sqlite:///" + str(path)
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(V2_TEMPLATE), str(path))
        migrate_database(url, up_to=24)
    _set_default(url, "node", "storage_binding_method", "bind_within_timeblock")
    with DatabaseMapping(url) as db:
        m.update_timestructure(db)
        db.commit_session("v25 time structure")
        pdef = db.get_item("parameter_definition", entity_class_name="node", name="storage_binding_method")
        db.update_parameter_definition(id=pdef["id"], description="A later step's description.")
        db.commit_session("a later update of the definition")
        assert _default(db, "node", "storage_binding_method") == "bind_within_timeset"
        assert "bind_within_timeblock" not in _members(db, "node", "storage_binding_method")
