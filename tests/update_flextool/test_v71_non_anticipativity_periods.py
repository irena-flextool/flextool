"""Tests for the v71 database migration (Slice G).

v71 adds ``solve.non_anticipativity_periods`` — the Array-of-periods
window knob that decouples operational non-anticipativity (storage /
online / reserve dispatch tied across stochastic branches) from
``realized_periods``:

* unset (default) -> legacy ``realized_dispatch u fix_storage`` window,
  byte-identical to prior behaviour;
* explicitly empty Array -> operations free from t0 (two-stage capacity
  expansion);
* period list -> tie only over those periods' timesteps.

The parameter is Array-shaped (``default_value``/``default_type`` both
``None``, no value list — mirrors ``realized_periods`` /
``fix_storage_periods``) and grouped under ``solve_advanced``.  Migrating
a DB adds the definition only (no per-solve value), so every migrated
solve stays on the unset -> legacy-window path.
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


def test_v71_version_constant_is_at_least_71() -> None:
    """The engine must report a schema version >= 71 — the
    non_anticipativity_periods lower bound.  Later migrations keep raising
    the constant, so an exact-equality assertion would regress.
    """
    assert FLEXTOOL_DB_VERSION >= 71


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON (at the fixture's old version, which
    predates v71 so the parameter can ONLY appear via the v71 block),
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


# --- the migration adds the Array definition (no value list, no default) ---


def test_non_anticipativity_periods_definition_present(tmp_path: Path) -> None:
    """``solve.non_anticipativity_periods`` exists as an unset Array param
    (no default, no value list) grouped under ``solve_advanced``."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="non_anticipativity_periods",
        )
        assert pdef, "non_anticipativity_periods not added by migration"
        # Array param — no scalar default, no bound value list.
        assert pdef["default_value"] is None
        assert pdef["default_type"] is None
        assert not pdef["parameter_value_list_name"]
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_migration_adds_no_per_solve_value(tmp_path: Path) -> None:
    """The v71 migration adds the definition ONLY — never a per-solve
    value — so every migrated solve resolves to the unset (legacy-window)
    case and stays byte-identical."""
    db = _migrated_db(tmp_path)
    try:
        values = [
            v for v in db.get_parameter_value_items()
            if v["parameter_definition_name"] == "non_anticipativity_periods"
        ]
        assert values == [], (
            "migration must not author any non_anticipativity_periods value"
        )
    finally:
        db.close()


def _build_v70_solve_db(url: str) -> None:
    """Create a minimal v70 DB (``model`` + ``solve`` classes) carrying no
    ``non_anticipativity_periods`` — so the v71 block is the ONLY path that
    can add it.

    Built from scratch (never a checked-in .sqlite, never the
    ``test_fixtures`` corpus): that corpus is regenerated to HEAD (>= v71),
    so loading one would start AT v71 and never exercise the v71 block.
    ``model.version`` default is pinned to 70 so ``migrate_database(...,
    up_to=71)`` runs only the v71 step.  The v71 step is designed to
    tolerate such minimal DBs (it creates ``solve_advanced`` if absent).
    """
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
            ],
            parameter_definitions=[
                ["model", "version", 70.0, None, "Database version."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["solve", "s1"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v70 DB (model + solve, no NA periods)")


def test_migration_reaches_v71_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v71 DB migrates to exactly v71, and re-running the v71 step
    is a no-op that keeps the parameter present — the ``add_update_item``
    helpers tolerate re-adds.

    Seeded from scratch at v70 (see ``_build_v70_solve_db``) so it starts
    genuinely BELOW v71 and exercises the v71 migration block from below,
    independent of the auto-migrated ``test_fixtures`` corpus.
    """
    db_path = tmp_path / "v70_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v70_solve_db(url)
    start_version = _db_version(url)
    assert start_version < 71, (
        "seed DB must start below v71 so the v71 block is exercised"
    )

    migrate_database(url, up_to=71)
    assert _db_version(url) == 71

    # Idempotent re-run of the v71 step — no crash, parameter still present.
    migrate_database(url, up_to=71)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="non_anticipativity_periods",
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
    spinedb_schema.json regen (with the v71 parameter) was committed.
    This is the same assertion CI makes; pinning it locally catches a
    forgotten regen."""
    assert sync_master_template(verify_only=True) is True
