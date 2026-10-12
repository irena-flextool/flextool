"""Tests for the v76 database migration (W4 Slice 1 / W4a).

v76 adds ``solve.dispatch_realization_method`` — the opt-in enum selecting
how the stochastic branches of a solve are realized into a committed
dispatch (the 2-stage seed of a future multi-stage market structure):

1. Creates the value list ``dispatch_realization_methods`` ==
   {within_solve, separate_dispatch}.
2. Adds the ``solve.dispatch_realization_method`` definition (default
   ``"within_solve"``) bound to that value list, grouped under
   ``solve_advanced``.

``'within_solve'`` (default) is byte-identical to prior behaviour;
``'separate_dispatch'`` is hard-rejected by the W4a guard until a later
slice lands the separate-dispatch machinery.  The engine-side
read/resolver + guard are covered by
``tests/engine_polars/test_w4a_dispatch_realization_guard.py``.

This migration is a structural clone of the v71
``stochastic_invest_method`` line.
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


def test_v76_version_constant_is_at_least_76() -> None:
    """The engine must report a schema version >= 76 — the
    dispatch_realization_method lower bound.  Later migrations keep raising
    the constant, so an exact-equality assertion would regress.
    """
    assert FLEXTOOL_DB_VERSION >= 76


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it to HEAD, and return an open
    mapping carrying the v76 parameter."""
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


# --- M1 / M3: the migration adds the definition + value list ------------


def test_dispatch_realization_method_definition_present(tmp_path: Path) -> None:
    """``solve.dispatch_realization_method`` exists, defaults to
    ``"within_solve"``, is bound to ``dispatch_realization_methods`` and
    grouped ``solve_advanced``."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="dispatch_realization_method",
        )
        assert pdef, "dispatch_realization_method not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "within_solve"
        assert (
            pdef["parameter_value_list_name"]
            == "dispatch_realization_methods"
        )
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_dispatch_realization_methods_value_list(tmp_path: Path) -> None:
    """``dispatch_realization_methods`` == {within_solve, separate_dispatch}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"]
            == "dispatch_realization_methods"
        }
        assert values == {"within_solve", "separate_dispatch"}
    finally:
        db.close()


def _build_v75_solve_db(url: str) -> None:
    """Create a minimal v75 DB (``model`` + ``solve`` classes) carrying no
    ``dispatch_realization_method`` — so the v76 block is the ONLY path that
    can add it.

    Built from scratch (never a checked-in .sqlite, never the
    ``test_fixtures`` corpus): that corpus is regenerated to HEAD (>= v76),
    so loading one would start AT v76 and never exercise the v76 block.
    ``model.version`` default is pinned to 75 so ``migrate_database(...,
    up_to=76)`` runs only the v76 step, keeping this a faithful v76 test
    even as later migrations extend the chain.  The v76 step is designed
    to tolerate such minimal DBs (it creates ``solve_advanced`` if absent).
    """
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
            ],
            parameter_definitions=[
                ["model", "version", 75.0, None, "Database version."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["solve", "s1"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session(
            "Seed v75 DB (model + solve, no dispatch realization method)"
        )


def test_migration_reaches_v76_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v76 DB migrates to exactly v76, and re-running the v76 step
    is a no-op that keeps the parameter present — the
    ``add_value_list_manual`` / ``add_update_item`` helpers tolerate
    re-adds.

    Seeded from scratch at v75 (see ``_build_v75_solve_db``) so it starts
    genuinely BELOW v76 and exercises the v76 migration block from below,
    independent of the auto-migrated ``test_fixtures`` corpus.
    """
    db_path = tmp_path / "v75_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v75_solve_db(url)
    start_version = _db_version(url)
    assert start_version < 76, (
        "seed DB must start below v76 so the v76 block is exercised"
    )

    migrate_database(url, up_to=76)
    assert _db_version(url) == 76

    # Idempotent re-run of the v76 step — no crash, parameter still present.
    migrate_database(url, up_to=76)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="dispatch_realization_method",
        )
        assert pdef, "parameter lost on idempotent re-migration"
        assert (
            from_database(pdef["default_value"], pdef["default_type"])
            == "within_solve"
        )


# --- M2: the committed master template is up to date --------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="sync_master_template runs migrate_database, which leaves a sqlite "
    "engine handle open; the TemporaryDirectory cleanup then hits WinError 32 "
    "on Windows. The same verification runs on the ubuntu-only template-check "
    "CI job (sync_master_json_template --verify).",
)
def test_master_template_up_to_date() -> None:
    """``sync_master_template(verify_only=True)`` returns True — the
    spinedb_schema.json regen (with the v76 parameter) was committed.
    This is the same assertion CI makes; pinning it locally catches a
    forgotten regen."""
    assert sync_master_template(verify_only=True) is True
