"""Tests for the v72 database migration — the invest-NA window.

v72 adds ``solve.non_anticipativity_invest_periods`` — the Array-of-periods
knob for the standard two-stage stochastic-investment mode: the periods
over which ``v_invest`` / ``v_divest`` are TIED across stochastic branches
(a single shared here-and-now decision).

* unset (default) OR empty Array -> NO tie, byte-identical to prior
  behaviour (recourse stays per-branch, none stays shared);
* period list -> tie investment over those periods (shared first stage),
  recourse later.

The parameter is Array-shaped (``default_value`` / ``default_type`` both
``None``, no value list — mirrors ``realized_periods`` /
``non_anticipativity_periods``) and grouped under ``solve_advanced``.
Migrating a DB adds the definition ONLY (no per-solve value), so every
migrated solve resolves to unset -> no tie -> today's behaviour.
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


def test_v72_version_constant_is_at_least_72() -> None:
    """The engine must report a schema version >= 72 — the
    non_anticipativity_invest_periods lower bound."""
    assert FLEXTOOL_DB_VERSION >= 72


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON (at the fixture's old version, which
    predates v72), migrate it to HEAD, and return an open mapping."""
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


# --- the migration adds the Array definition (no value list, no default) ---


def test_invest_periods_definition_present(tmp_path: Path) -> None:
    """``solve.non_anticipativity_invest_periods`` exists as an unset Array
    param (no default, no value list) grouped under ``solve_advanced``."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="non_anticipativity_invest_periods",
        )
        assert pdef, "non_anticipativity_invest_periods not added by migration"
        assert pdef["default_value"] is None
        assert pdef["default_type"] is None
        assert not pdef["parameter_value_list_name"]
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_migration_adds_no_per_solve_value(tmp_path: Path) -> None:
    """The v72 migration adds the definition ONLY — never a per-solve
    value — so every migrated solve resolves to the unset (no-tie) case
    and stays byte-identical."""
    db = _migrated_db(tmp_path)
    try:
        values = [
            v for v in db.get_parameter_value_items()
            if v["parameter_definition_name"]
            == "non_anticipativity_invest_periods"
        ]
        assert values == [], (
            "migration must not author any "
            "non_anticipativity_invest_periods value"
        )
    finally:
        db.close()


def _build_v71_solve_db(url: str) -> None:
    """Create a minimal v71 DB (``model`` + ``solve`` classes) carrying no
    ``non_anticipativity_invest_periods`` — so the v72 block is the ONLY
    path that can add it.  ``model.version`` default pinned to 71 so
    ``migrate_database(..., up_to=72)`` runs only the v72 step."""
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
            ],
            parameter_definitions=[
                ["model", "version", 71.0, None, "Database version."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["solve", "s1"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v71 DB (model + solve, no NA invest periods)")


def test_migration_reaches_v72_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v72 DB migrates to exactly v72, and re-running the v72 step
    is a no-op that keeps the parameter present."""
    db_path = tmp_path / "v71_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v71_solve_db(url)
    start_version = _db_version(url)
    assert start_version < 72, (
        "seed DB must start below v72 so the v72 block is exercised"
    )

    migrate_database(url, up_to=72)
    assert _db_version(url) == 72

    # Idempotent re-run of the v72 step.
    migrate_database(url, up_to=72)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="non_anticipativity_invest_periods",
        )
        assert pdef, "parameter lost on idempotent re-migration"
        assert pdef["default_value"] is None
        assert pdef["default_type"] is None


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
    spinedb_schema.json regen (with the v72 parameter) was committed."""
    assert sync_master_template(verify_only=True) is True
