"""Regression test for the v28 economic-parameter rename collision.

The v28 migration step renames the entity-level ``interest_rate``
parameter definition to ``discount_rate`` (on ``unit``, ``connection``,
``node``) and the model-level ``discount_rate`` to ``inflation_rate``.

It did so with a bare ``update_parameter_definition(name=...)`` guarded
only by ``if param:`` — i.e. "if the *source* name still exists, rename
it".  That guard is **not collision-safe**: it never checks whether the
*target* name is already occupied.  When a database reaches the v28 step
with BOTH the source and the target definition present, the rename hits
spinedb_api's unique-name constraint and the whole migration aborts with::

    SpineDBAPIError: there's already a parameter_definition with
    {'entity_class_name': 'unit', 'name': 'discount_rate'}

How the mixed state is reachable in the wild
--------------------------------------------
``migrate_database`` writes ``model.version`` exactly once, at the very
end of the loop — individual steps commit their schema changes
immediately but do not bump the stored version.  So if a migration is
interrupted / cancelled / crashes after a later step has already created
``discount_rate`` (the v28 rename itself, or the v48 ``add_update_item``
default-setter) but before the final version write lands, the database
keeps the new ``discount_rate`` definition while ``model.version`` still
reads its old pre-28 value.  A subsequent re-application of the v25
template (re-init / GUI "apply template" / re-import) restores
``interest_rate``.  The next ``migrate_database`` re-runs the v28 step
against a class that now has *both* names → collision.

This test reconstructs the exact mixed state at version 27 (the step
just before v28) and asserts the migration completes cleanly, leaving a
single, correctly-named ``discount_rate`` definition.
"""
from __future__ import annotations

import sys
from pathlib import Path

from spinedb_api import DatabaseMapping, import_data

from flextool._resources import package_data_path
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool import FLEXTOOL_DB_VERSION

_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

_V25_TEMPLATE = str(package_data_path("schemas/pre_v26/flextool_template_v25.json"))


def _param_def(db: DatabaseMapping, entity_class_name: str, name: str):
    defs = db.mapped_table("parameter_definition")
    try:
        return db.item(defs, entity_class_name=entity_class_name, name=name)
    except Exception:
        return None


def _stage_mixed_state_at_v27(db_path: Path) -> str:
    """Build a v25 DB, migrate to v27, then plant the collision:
    add ``discount_rate`` definitions alongside the still-present
    ``interest_rate`` on the entity classes, and an ``inflation_rate``
    alongside the model-level ``discount_rate``.  Leaves ``model.version``
    at 27 so the v28 step runs next.
    """
    url = "sqlite:///" + str(db_path.resolve())
    initialize_database(_V25_TEMPLATE, str(db_path))
    # Stop just before v28 — interest_rate is still present at v27.
    migrate_database(url, up_to=27)

    with DatabaseMapping(url) as db:
        # Sanity: pre-collision the entity classes have interest_rate only.
        for cls in ("unit", "connection", "node"):
            assert _param_def(db, cls, "interest_rate") is not None, (
                f"fixture invariant broken: {cls}.interest_rate missing at v27"
            )
            assert _param_def(db, cls, "discount_rate") is None
        # Plant the target names so the v28 rename collides.
        plant = [
            ["unit", "discount_rate", None, None, "planted collision"],
            ["connection", "discount_rate", None, None, "planted collision"],
            ["node", "discount_rate", None, None, "planted collision"],
            # model-level: discount_rate already exists (v25 template);
            # plant inflation_rate so that rename collides too.
            ["model", "inflation_rate", None, None, "planted collision"],
        ]
        import_data(db, object_parameters=plant)
        db.commit_session("Plant mixed interest/discount + discount/inflation state")
    return url


def test_v28_rename_is_collision_safe(tmp_path):
    """Migrating past v28 with both source and target names present must
    not raise, and must leave exactly one correctly-named definition.
    """
    db_path = tmp_path / "mixed_v27.sqlite"
    url = _stage_mixed_state_at_v27(db_path)

    # Before the fix this raises SpineDBAPIError at the v28 rename.
    migrate_database(url, up_to=28)

    with DatabaseMapping(url) as db:
        for cls in ("unit", "connection", "node"):
            assert _param_def(db, cls, "discount_rate") is not None, (
                f"{cls}.discount_rate should exist after v28"
            )
            assert _param_def(db, cls, "interest_rate") is None, (
                f"{cls}.interest_rate should be gone after v28 rename"
            )
        # model-level: discount_rate -> inflation_rate (target pre-existed).
        assert _param_def(db, "model", "inflation_rate") is not None
        assert _param_def(db, "model", "discount_rate") is None


def test_v28_collision_full_chain_reaches_current_version(tmp_path):
    """The collision-recovery path must not just survive v28 — the whole
    chain must still reach the current schema version.
    """
    db_path = tmp_path / "mixed_v27_full.sqlite"
    url = _stage_mixed_state_at_v27(db_path)

    migrate_database(url)  # all the way to FLEXTOOL_DB_VERSION

    with DatabaseMapping(url) as db:
        sq = db.object_parameter_definition_sq
        sp = (
            db.query(sq)
            .filter(sq.c.object_class_name == "model")
            .filter(sq.c.parameter_name == "version")
            .one_or_none()
        )
        from spinedb_api import from_database

        version = from_database(sp.default_value, sp.default_type)
        assert int(version) == FLEXTOOL_DB_VERSION
        # And the rename target survived to the end.
        assert _param_def(db, "unit", "discount_rate") is not None
        assert _param_def(db, "unit", "interest_rate") is None
