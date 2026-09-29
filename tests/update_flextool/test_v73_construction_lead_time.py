"""Tests for the v73 database migration — construction lead time (Slice H).

v73 adds two per-entity parameters on each invest entity class
(``unit`` / ``connection`` / ``node``):

* ``construction_lead_time`` — ``[years]`` float/map, default unset (0);
* ``construction_lead_time_method`` — enum
  (``immediate`` / ``closest_seam`` / ``previous_seam`` / ``next_seam``),
  bound to the ``construction_lead_time_methods`` value list, default unset
  (null — so the reader can distinguish unset from explicit ``immediate``).

Both defaults are ``null`` and the migration adds definitions ONLY (never a
per-entity value), so every migrated entity resolves to unset -> ``L=0`` ->
``immediate`` -> byte-identical to prior behaviour.  Grouped under
``investment`` (mirrors ``lifetime``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.sync_master_json_template import (
    sync_master_template,
)

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = TEST_DIR / "fixtures"

_CLASSES = ("unit", "connection", "node")
_METHODS = ("immediate", "closest_seam", "previous_seam", "next_seam")


def test_v73_version_constant_is_at_least_73() -> None:
    """The engine must report a schema version >= 73 — the
    construction_lead_time lower bound."""
    assert FLEXTOOL_DB_VERSION >= 73


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON (at its old version, predating v73),
    migrate it to HEAD, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def _db_version(url: str) -> int:
    """Read the migrated ``model.version`` parameter-definition default."""
    with DatabaseMapping(url, create=False) as db:
        sq = db.object_parameter_definition_sq
        row = (
            db.query(sq)
            .filter(sq.c.object_class_name == "model")
            .filter(sq.c.parameter_name == "version")
            .one_or_none()
        )
        assert row is not None, "model.version parameter definition missing"
        return int(from_database(row.default_value, row.default_type))


# --- definitions on all three invest classes ---------------------------


def test_lead_time_definition_present(tmp_path: Path) -> None:
    """``construction_lead_time`` exists as an unset float/map param
    (no default, no value list) grouped under ``investment`` on every
    invest entity class."""
    db = _migrated_db(tmp_path)
    try:
        for cls in _CLASSES:
            pdef = db.get_parameter_definition_item(
                entity_class_name=cls, name="construction_lead_time",
            )
            assert pdef, f"construction_lead_time missing on {cls}"
            assert pdef["default_value"] is None
            assert pdef["default_type"] is None
            assert not pdef["parameter_value_list_name"]
            assert pdef["parameter_group_name"] == "investment"
    finally:
        db.close()


def test_method_definition_present(tmp_path: Path) -> None:
    """``construction_lead_time_method`` exists as an unset enum param
    (null default so unset != explicit immediate) bound to the
    ``construction_lead_time_methods`` value list, grouped under
    ``investment`` on every invest entity class."""
    db = _migrated_db(tmp_path)
    try:
        for cls in _CLASSES:
            pdef = db.get_parameter_definition_item(
                entity_class_name=cls, name="construction_lead_time_method",
            )
            assert pdef, f"construction_lead_time_method missing on {cls}"
            assert pdef["default_value"] is None, (
                "method default must be null, not literal 'immediate' — the "
                "reader needs unset != explicit immediate for the two-level "
                "default"
            )
            assert pdef["default_type"] is None
            assert (pdef["parameter_value_list_name"]
                    == "construction_lead_time_methods")
            assert pdef["parameter_group_name"] == "investment"
    finally:
        db.close()


def test_method_value_list_values(tmp_path: Path) -> None:
    """The value list carries exactly the four methods."""
    db = _migrated_db(tmp_path)
    try:
        got = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"]
            == "construction_lead_time_methods"
        }
        assert got == set(_METHODS), f"value list mismatch: {got}"
    finally:
        db.close()


def test_lead_time_parameter_types_ranked(tmp_path: Path) -> None:
    """``construction_lead_time`` allows float (rank 0) + map (rank 1) with
    DISTINCT ranks on every class — a collapsed rank silently drops the
    map row on re-import."""
    db = _migrated_db(tmp_path)
    try:
        for cls in _CLASSES:
            types = {
                (t["type"], t["rank"])
                for t in db.get_parameter_type_items()
                if t["entity_class_name"] == cls
                and t["parameter_definition_name"] == "construction_lead_time"
            }
            assert ("float", 0) in types, f"{cls}: float@0 missing {types}"
            assert ("map", 1) in types, f"{cls}: map@1 missing {types}"
            ranks = [r for _, r in types]
            assert len(ranks) == len(set(ranks)), (
                f"{cls}: duplicate parameter_type ranks {types}"
            )
    finally:
        db.close()


def test_migration_adds_no_per_entity_value(tmp_path: Path) -> None:
    """The v73 migration adds definitions ONLY — never a per-entity value —
    so every migrated entity resolves to unset -> immediate -> byte-parity."""
    db = _migrated_db(tmp_path)
    try:
        values = [
            v for v in db.get_parameter_value_items()
            if v["parameter_definition_name"] in (
                "construction_lead_time", "construction_lead_time_method")
        ]
        assert values == [], (
            "migration must not author any construction_lead_time value"
        )
    finally:
        db.close()


def _build_v72_invest_db(url: str) -> None:
    """Create a minimal v72 DB with the three invest entity classes carrying
    no construction_lead_time — so the v73 block is the ONLY path that can
    add it.  ``model.version`` default pinned to 72."""
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["unit", ()],
                ["connection", ()],
                ["node", ()],
            ],
            parameter_definitions=[
                ["model", "version", 72.0, None, "Database version."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["unit", "u1"],
                ["node", "n1"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v72 DB (invest classes, no lead time)")


def test_migration_reaches_v73_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v73 DB migrates to exactly v73, and re-running the v73 step is
    a no-op that keeps the parameters present with distinct type ranks."""
    db_path = tmp_path / "v72_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v72_invest_db(url)
    start_version = _db_version(url)
    assert start_version < 73, (
        "seed DB must start below v73 so the v73 block is exercised"
    )

    migrate_database(url, up_to=73)
    assert _db_version(url) == 73

    # Idempotent re-run of the v73 step.
    migrate_database(url, up_to=73)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        for cls in _CLASSES:
            pdef = db.get_parameter_definition_item(
                entity_class_name=cls, name="construction_lead_time",
            )
            assert pdef, f"lead_time lost on idempotent re-migration ({cls})"
            types = {
                (t["type"], t["rank"])
                for t in db.get_parameter_type_items()
                if t["entity_class_name"] == cls
                and t["parameter_definition_name"] == "construction_lead_time"
            }
            assert ("float", 0) in types and ("map", 1) in types, (
                f"{cls}: type ranks corrupted after idempotent re-run {types}"
            )


# --- the committed master template is up to date ------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="sync_master_template runs migrate_database, which leaves a sqlite "
    "engine handle open; the TemporaryDirectory cleanup then hits WinError 32 "
    "on Windows. The same verification runs on the ubuntu-only template-check "
    "CI job (sync_master_json_template --verify).",
)
def test_master_template_up_to_date() -> None:
    """``sync_master_template(verify_only=True)`` returns True — the
    spinedb_schema.json regen (with the v73 parameters) was committed."""
    assert sync_master_template(verify_only=True) is True
