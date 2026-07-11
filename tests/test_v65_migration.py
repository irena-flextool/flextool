"""Tests for the v65 database migration.

v65 makes two related schema changes in one migration:

1. Adds ``node.energy_margin`` (float, default ``1.0``) and
   ``node.energy_margin_method`` (enum ``energy_margin_methods`` =
   {none, inflow_multiplier}, default ``none``) — an investment-stage inflow
   multiplier that offsets representative-period VRE optimism.
2. Converts the old ``group.has_capacity_margin`` yes/no flag into
   ``group.capacity_margin_method`` (enum ``capacity_margin_methods`` =
   {none, manual}, default ``none``).  ``yes`` -> ``manual``; ``no`` /
   absent -> the default ``none`` (no row).  The old flag definition is
   removed.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v65_version_constant_is_at_least_65() -> None:
    """The engine must report a schema version >= 65 — the energy-margin /
    capacity-margin-method lower bound.  Later migrations keep raising the
    constant, so an exact-equality assertion would regress as the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 65


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def test_migration_adds_energy_margin_method(tmp_path: Path) -> None:
    """``node.energy_margin_method`` exists, defaults to ``none`` and is
    bound to the ``energy_margin_methods`` value list.
    """
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="energy_margin_method",
        )
        assert pdef, "energy_margin_method not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "none"
        assert pdef.extended().get("parameter_value_list_name") == (
            "energy_margin_methods"
        )
    finally:
        db.close()


def test_migration_adds_energy_margin(tmp_path: Path) -> None:
    """``node.energy_margin`` exists and defaults to ``1.0`` (float)."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="energy_margin",
        )
        assert pdef, "energy_margin not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == 1.0
        assert isinstance(default, float)
    finally:
        db.close()


def test_migration_adds_capacity_margin_method(tmp_path: Path) -> None:
    """``group.capacity_margin_method`` exists, defaults to ``none`` and is
    bound to the ``capacity_margin_methods`` value list.
    """
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="group",
            name="capacity_margin_method",
        )
        assert pdef, "capacity_margin_method not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "none"
        assert pdef.extended().get("parameter_value_list_name") == (
            "capacity_margin_methods"
        )
    finally:
        db.close()


def test_energy_margin_methods_value_list(tmp_path: Path) -> None:
    """``energy_margin_methods`` carries exactly {none, inflow_multiplier}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "energy_margin_methods"
        }
        assert values == {"none", "inflow_multiplier"}
    finally:
        db.close()


def test_capacity_margin_methods_value_list(tmp_path: Path) -> None:
    """``capacity_margin_methods`` carries exactly {none, manual}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "capacity_margin_methods"
        }
        assert values == {"none", "manual"}
    finally:
        db.close()


def test_has_capacity_margin_definition_removed(tmp_path: Path) -> None:
    """The old ``group.has_capacity_margin`` definition is gone after v65."""
    db = _migrated_db(tmp_path)
    try:
        stale = [
            p
            for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == "group"
            and p["name"] == "has_capacity_margin"
        ]
        assert stale == []
    finally:
        db.close()


def _build_v64_group_flag_db(url: str) -> None:
    """Create a minimal v64 DB carrying ``group.has_capacity_margin``.

    Built from scratch (never a checked-in .sqlite): one group with the
    flag ``yes`` in a non-Base alternative and one with ``no`` in Base.
    ``model.version`` default is pinned to 64 so ``migrate_database`` runs
    only the v65 step.
    """
    with DatabaseMapping(url, create=True) as db:
        count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["node", ()],
                ["group", ()],
            ],
            parameter_value_lists=[
                ["yes_no", "yes"],
                ["yes_no", "no"],
            ],
            parameter_definitions=[
                ["model", "version", 64.0, None, "Database version."],
                [
                    "group",
                    "has_capacity_margin",
                    "no",
                    "yes_no",
                    "Enforce a capacity margin (legacy yes/no flag).",
                ],
            ],
            alternatives=[["Base", ""], ["scen_a", ""]],
            entities=[
                ["group", "g_yes"],
                ["group", "g_no"],
            ],
            parameter_values=[
                ["group", "g_yes", "has_capacity_margin", "yes", "scen_a"],
                ["group", "g_no", "has_capacity_margin", "no", "Base"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v64 DB with has_capacity_margin flag")


def test_value_conversion_yes_to_manual(tmp_path: Path) -> None:
    """``has_capacity_margin='yes'`` becomes ``capacity_margin_method='manual'``
    in the same alternative; a ``'no'`` group gets no row (defaults to none).
    """
    db_path = tmp_path / "v64_flags.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v64_group_flag_db(url)

    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        # The old flag definition is gone.
        assert not [
            p
            for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == "group"
            and p["name"] == "has_capacity_margin"
        ]

        cm_rows = [
            pv
            for pv in db.get_parameter_value_items()
            if pv["entity_class_name"] == "group"
            and pv["parameter_definition_name"] == "capacity_margin_method"
        ]
        by_entity = {
            (pv["entity_byname"], pv["alternative_name"]): from_database(
                pv["value"], pv["type"]
            )
            for pv in cm_rows
        }
        # g_yes converted to 'manual' in the SAME alternative (scen_a).
        assert by_entity.get((("g_yes",), "scen_a")) == "manual"
        # g_no has NO capacity_margin_method row (defaults to none).
        assert not any(
            ent == ("g_no",) for (ent, _alt) in by_entity
        )
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running the migration over an already-current DB is a no-op: the
    new definitions stay single and at their defaults.
    """
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        for cls, name in (
            ("node", "energy_margin_method"),
            ("node", "energy_margin"),
            ("group", "capacity_margin_method"),
        ):
            matches = [
                p
                for p in db.get_parameter_definition_items()
                if p["entity_class_name"] == cls and p["name"] == name
            ]
            assert len(matches) == 1, f"{cls}.{name} duplicated after re-run"
    finally:
        db.close()
