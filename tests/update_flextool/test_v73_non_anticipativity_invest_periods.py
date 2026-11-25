"""Tests for the v73 database migration — the invest-NA window.

v73 adds ``solve.non_anticipativity_invest_periods`` — the Array-of-periods
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

v75 renames it (``shared_*``) and v84 replaces it by
``solve.invest_solve_branching_method`` (``test_v84_branching``), so the
step is tested on a database migrated to 83 at most.
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




def test_v73_version_constant_is_at_least_73() -> None:
    """The engine must report a schema version >= 73 — the
    non_anticipativity_invest_periods lower bound."""
    assert FLEXTOOL_DB_VERSION >= 73


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """A minimal v72 database (:func:`_build_v72_solve_db`) migrated to 83, the last data version that has the
    parameter (v84 replaces it by ``invest_solve_branching_method``: ``test_v84_branching``), as an open mapping.
    The fixture corpus is at the head, so it cannot start below v73."""
    db_path = tmp_path / "v72.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v72_solve_db(url)
    migrate_database(url, up_to=83)
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
    """At 83 the investment shared-window param exists as an unset Array
    param (no default, no value list) grouped under ``solve_advanced``.
    Added at v73 as ``non_anticipativity_invest_periods`` and renamed to
    ``shared_invest_periods`` at v75 — the database migrates to 83, so it
    carries the current (renamed) name."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="shared_invest_periods",
        )
        assert pdef, "shared_invest_periods not present at 83"
        assert pdef["default_value"] is None
        assert pdef["default_type"] is None
        assert not pdef["parameter_value_list_name"]
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_migration_adds_no_per_solve_value(tmp_path: Path) -> None:
    """The v73 migration adds the definition ONLY — never a per-solve
    value — so every migrated solve resolves to the unset (no-tie) case
    and stays byte-identical.  (Param renamed to ``shared_invest_periods``
    at v75; the database is at 83.)"""
    db = _migrated_db(tmp_path)
    try:
        values = [
            v for v in db.get_parameter_value_items()
            if v["parameter_definition_name"]
            == "shared_invest_periods"
        ]
        assert values == [], (
            "migration must not author any "
            "shared_invest_periods value"
        )
    finally:
        db.close()


def _build_v72_solve_db(url: str) -> None:
    """Create a minimal v72 DB (``model`` + ``solve`` classes) carrying no
    ``non_anticipativity_invest_periods`` — so the v73 block is the ONLY
    path that can add it.  ``model.version`` default pinned to 72 so
    ``migrate_database(..., up_to=73)`` runs only the v73 step."""
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
            ],
            parameter_definitions=[
                ["model", "version", 72.0, None, "Database version."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["solve", "s1"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v72 DB (model + solve, no NA invest periods)")


def test_migration_reaches_v73_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v73 DB migrates to exactly v73, and re-running the v73 step
    is a no-op that keeps the parameter present."""
    db_path = tmp_path / "v72_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v72_solve_db(url)
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
    spinedb_schema.json regen (with the v73 parameter) was committed."""
    assert sync_master_template(verify_only=True) is True
