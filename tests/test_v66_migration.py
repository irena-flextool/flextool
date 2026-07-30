"""Tests for the v66 database migration.

v66 makes the energy-margin lever symmetric:

1. Renames ``node.energy_margin`` -> ``node.energy_margin_multiplier``
   (a plain definition rename that preserves any FK'd parameter values).
2. Adds ``node.energy_margin_adder`` (float, default ``0.0``) — an additive
   investment-stage inflow adder [MWh].
3. Populates the previously-reserved ``energy_margin_methods`` value
   ``inflow_adder`` (list becomes {none, inflow_multiplier, inflow_adder}).
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v66_version_constant_is_at_least_66() -> None:
    """The engine must report a schema version >= 66 — the energy-margin-adder
    lower bound.  Later migrations keep raising the constant, so an exact
    equality assertion would regress as the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 66


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def test_energy_margin_definition_removed(tmp_path: Path) -> None:
    """The old ``node.energy_margin`` definition is gone after v66."""
    db = _migrated_db(tmp_path)
    try:
        stale = [
            p
            for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == "node" and p["name"] == "energy_margin"
        ]
        assert stale == []
    finally:
        db.close()


def test_energy_margin_multiplier_present(tmp_path: Path) -> None:
    """``node.energy_margin_multiplier`` exists and defaults to ``1.0``."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="energy_margin_multiplier",
        )
        assert pdef, "energy_margin_multiplier not present after migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == 1.0
        assert isinstance(default, float)
    finally:
        db.close()


def test_energy_margin_adder_present(tmp_path: Path) -> None:
    """``node.energy_margin_adder`` exists and defaults to ``0.0`` (float)."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="energy_margin_adder",
        )
        assert pdef, "energy_margin_adder not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == 0.0
        assert isinstance(default, float)
    finally:
        db.close()


def test_energy_margin_methods_value_list(tmp_path: Path) -> None:
    """``energy_margin_methods`` == {none, inflow_multiplier, inflow_adder}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "energy_margin_methods"
        }
        assert values == {"none", "inflow_multiplier", "inflow_adder"}
    finally:
        db.close()


def _build_v65_energy_margin_db(url: str) -> None:
    """Create a minimal v65 DB carrying ``node.energy_margin`` with a value.

    Built from scratch (never a checked-in .sqlite): one node with an
    explicit ``energy_margin`` value in a non-Base alternative.
    ``model.version`` default is pinned to 65 so ``migrate_database`` runs
    only the v66 step.
    """
    with DatabaseMapping(url, create=True) as db:
        count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["node", ()],
            ],
            parameter_value_lists=[
                ["energy_margin_methods", "none"],
                ["energy_margin_methods", "inflow_multiplier"],
            ],
            parameter_definitions=[
                ["model", "version", 65.0, None, "Database version."],
                [
                    "node",
                    "energy_margin",
                    1.0,
                    None,
                    "Investment-stage inflow multiplier (v65 name).",
                ],
            ],
            alternatives=[["Base", ""], ["scen_a", ""]],
            entities=[
                ["node", "n1"],
            ],
            parameter_values=[
                ["node", "n1", "energy_margin", 1.25, "scen_a"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v65 DB with node.energy_margin value")


def test_energy_margin_value_preserved_through_rename(tmp_path: Path) -> None:
    """A pre-existing ``energy_margin`` value survives the rename to
    ``energy_margin_multiplier`` in the same alternative.
    """
    db_path = tmp_path / "v65_energy_margin.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v65_energy_margin_db(url)

    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        # Old definition gone.
        assert not [
            p
            for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == "node" and p["name"] == "energy_margin"
        ]
        # Value carried onto energy_margin_multiplier, same alternative.
        rows = {
            (pv["entity_byname"], pv["alternative_name"]): from_database(
                pv["value"], pv["type"]
            )
            for pv in db.get_parameter_value_items()
            if pv["entity_class_name"] == "node"
            and pv["parameter_definition_name"] == "energy_margin_multiplier"
        }
        assert rows.get((("n1",), "scen_a")) == 1.25
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running the migration over an already-current DB is a no-op: the
    renamed / added definitions stay single and at their defaults.
    """
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        for cls, name in (
            ("node", "energy_margin_multiplier"),
            ("node", "energy_margin_adder"),
            ("node", "energy_margin_method"),
        ):
            matches = [
                p
                for p in db.get_parameter_definition_items()
                if p["entity_class_name"] == cls and p["name"] == name
            ]
            assert len(matches) == 1, f"{cls}.{name} duplicated after re-run"
    finally:
        db.close()
