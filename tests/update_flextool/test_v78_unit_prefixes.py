"""Database migration to version 78 (a unit on every parameter description).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 77, then given the cases
the v78 step must handle: one definition deleted (it must stay absent), one
description edited by the user with its own bracket (kept), one edited without
a bracket (gets the prefix) and one starting with an old bracket followed by the
user's own text (only the bracket is rewritten).  The seed's
``unit__outputNode.capacity_min_coeff`` text predates the v70 one; a bracket the step
does not know is kept.  Every other description must
end up with the unit the shipped schema has.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
ABSENT = ("unit", "min_load")  # deleted before the step: it must not be re-created
OWN_BRACKET = ("node", "penalty_up", "[EUR/MWh] My own penalty note.")
NO_BRACKET = ("unit", "startup_method", "My own note.")
OLD_BRACKET = ("unit", "availability", "[e.g. 0.9 means 90%] My own availability note.")
UNKNOWN_BRACKET = ("unit__outputNode", "capacity_min_coeff")  # "[factor, default 1.0] ..." in the seed


def _descriptions(db) -> dict[tuple[str, str], str]:
    return {(d["entity_class_name"], d["name"]): d["description"] for d in db.get_items("parameter_definition")}


def _set(db, cls: str, name: str, text: str) -> None:
    pdef = db.item(db.mapped_table("parameter_definition"), entity_class_name=cls, name=name)
    db.update_parameter_definition(id=pdef["id"], description=text)


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v78") / "seed.sqlite")
    migrate_database(url, up_to=77)
    with DatabaseMapping(url) as db:
        for cls, name, text in (OWN_BRACKET, NO_BRACKET, OLD_BRACKET):
            _set(db, cls, name, text)
        db.remove_item("parameter_definition",
                       db.get_item("parameter_definition", entity_class_name=ABSENT[0], name=ABSENT[1])["id"])
        db.commit_session("pre-78 content")
        before = _descriptions(db)
    migrate_database(url)
    with DatabaseMapping(url) as db:
        after = _descriptions(db)
        version = db.get_item("parameter_definition", entity_class_name="model", name="version")
        m._migrate_v78_unit_prefixes(db)
        try:
            db.commit_session("re-run")
            rerun_idempotent = False
        except NothingToCommit:
            rerun_idempotent = True
    return {"before": before, "after": after, "version_description": version["description"],
            "rerun_idempotent": rerun_idempotent}


def _shipped() -> dict[tuple[str, str], str]:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        return {(row[0], row[1]): row[4] for row in json.load(f)["parameter_definitions"]}


def _bracket(text: str) -> str:
    return text[: text.index("]") + 1]


def test_units_match_the_shipped_schema(migrated) -> None:
    """The seed's texts after the bracket can be older than the shipped ones; its units cannot."""
    shipped = _shipped()
    edited = {OWN_BRACKET[:2], NO_BRACKET[:2], OLD_BRACKET[:2], UNKNOWN_BRACKET}
    for key, text in migrated["after"].items():
        if key in edited:
            continue
        assert _bracket(text) == _bracket(shipped[key]), key
        if key in m._V78_DESCRIPTIONS:
            assert text == shipped[key], key


def test_every_description_has_a_bracket(migrated) -> None:
    assert all(text.startswith("[") for text in migrated["after"].values())
    assert migrated["version_description"] == "[version number] Contains database version information."


def test_user_edits(migrated) -> None:
    after = migrated["after"]
    assert after[OWN_BRACKET[:2]] == OWN_BRACKET[2]
    assert after[NO_BRACKET[:2]] == "[choice] My own note."
    assert after[OLD_BRACKET[:2]] == "[-] e.g. 0.9 means 90%. My own availability note."
    assert after[UNKNOWN_BRACKET].startswith("[factor, default 1.0] ")
    assert after[UNKNOWN_BRACKET] == migrated["before"][UNKNOWN_BRACKET]


def test_absent_definition_is_not_created(migrated) -> None:
    assert ABSENT not in migrated["before"]
    assert ABSENT not in migrated["after"]


def test_rerun_is_idempotent(migrated) -> None:
    assert migrated["rerun_idempotent"]


def test_tables_are_disjoint() -> None:
    """A whole-text replacement, a bracket rewrite and a prefix never apply to the same definition."""
    texts, rewrites, prefixes = set(m._V78_DESCRIPTIONS), set(m._V78_BRACKET_REWRITES), set(m._V78_UNIT_PREFIXES)
    assert not texts & rewrites and not texts & prefixes and not rewrites & prefixes
    assert all(text.startswith("[") for text in m._V78_DESCRIPTIONS.values())
