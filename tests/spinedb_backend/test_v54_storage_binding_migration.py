"""Phase 2 data migration for ``node.storage_binding_method``.

The 2026-04 list-valued design (now being reverted) silently flattened
array-typed ``storage_binding_method`` values into one row per array
element, which downstream additive logic in ``calc_storage_vre.py``
turned into double-counted state-change residuals.  v53 wired the
``storage_binding_methods`` parameter_value_list and Phase 1 added the
ingestion guard; the **v54 step exercised here** ports the data: every
array-valued row is rewritten as a single string by picking the
highest-priority element from a fixed list:

    bind_using_blended_weights >
    bind_intraperiod_blocks >
    bind_within_solve >
    bind_within_period >
    bind_within_timeset >
    bind_forward_only

Arrays containing only unknown method strings raise
``SpineDBAPIError`` naming the offending entity and contents — the
migration refuses to guess.  These tests pin both branches.

See ``_audit_reports/storage_binding_method_callsites.md`` §9 for the
H2_trade.sqlite case that motivated the priority ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from spinedb_api import (
    Array,
    DatabaseMapping,
    SpineDBAPIError,
    from_database,
    import_data,
    to_database,
)

# ``db_utils.json_to_db`` is the standard chain-integrity entry point
# used by ``test_v52_migration.py`` and other v* migration tests; it
# stages a known-good v25-era fixture so the migration chain can run
# without us hand-recreating the pre-v25 schema.
_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from db_utils import json_to_db  # noqa: E402

from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

_BASE_FIXTURE = _TESTS_ROOT / "fixtures" / "stochastics.json"


def _build_v53_db_with_array_value(
    db_path: Path,
    *,
    entity_name: str,
    array_members: list[str],
) -> str:
    """Create a fresh SQLite DB, migrate it up to v53, then plant an
    array-valued ``storage_binding_method`` row.

    Returns the SQLAlchemy-style ``sqlite:///`` URL.

    Mirrors the H2_trade.sqlite shape that Phase 2 must port: a node
    with a list of method strings under the ``storage_binding_method``
    parameter.  Migrating up to 53 first ensures the
    parameter_value_list wiring from Phase 1 is in place before we
    plant the array (so the planted row is exactly the legacy shape
    that the v54 step is supposed to find and rewrite).
    """
    # Stage a known-good pre-v52 fixture (mirrors test_v52_migration.py)
    # then migrate up to v52 only — NOT v53.  We stop short of v53
    # because by v22 (``update_timestructure``) the
    # ``storage_binding_methods`` parameter_value_list has already
    # been bound to ``node.storage_binding_method``, and the binding
    # causes ``import_data`` / ``add_update_item`` to reject any value
    # (including arrays) that isn't a list member.
    #
    # Real-world DBs (e.g. H2_trade.sqlite) carry array values because
    # they were edited through tools that bypassed the wiring or via
    # an inflight schema where the binding was temporarily absent.  To
    # faithfully reproduce that legacy shape we briefly **un-wire**
    # the list, plant the array, and let the migration chain (which
    # re-wires in the Phase 1 v53 step) handle the rest.  The v53
    # wiring step does NOT re-validate existing parameter_value rows,
    # so the planted array survives v53 and reaches the v54 step.
    url = json_to_db(_BASE_FIXTURE, db_path)
    migrate_database(url, up_to=52)

    with DatabaseMapping(url) as db:
        defs = list(db.find_parameter_definitions(
            entity_class_name="node", name="storage_binding_method",
        ))
        assert defs, "v52 fixture is missing node.storage_binding_method"
        db.update_parameter_definition(
            id=defs[0]["id"],
            parameter_value_list_name=None,
        )
        db.commit_session("temporarily un-wire value list to plant legacy array")

    bad_value = Array(array_members, value_type=str, index_name="i")
    val_bytes, val_type = to_database(bad_value)
    with DatabaseMapping(url) as db:
        db.add_update_item(
            "entity",
            entity_class_name="node",
            name=entity_name,
            entity_byname=(entity_name,),
        )
        db.add_update_item(
            "parameter_value",
            entity_class_name="node",
            entity_byname=(entity_name,),
            parameter_definition_name="storage_binding_method",
            alternative_name="Base",
            value=val_bytes,
            type=val_type,
        )
        db.commit_session("plant array value (legacy shape)")
    return url


def _read_storage_binding_value(url: str, entity_name: str):
    """Return ``(type, parsed_value)`` for the named node's
    ``storage_binding_method`` row, or ``(None, None)`` if absent."""
    with DatabaseMapping(url) as db:
        for pv in db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="storage_binding_method",
        ):
            if pv["entity_byname"] == (entity_name,):
                return pv["type"], pv["parsed_value"]
    return None, None


def test_v54_picks_highest_priority_element(tmp_path: Path) -> None:
    """The H2_trade pattern: an array containing
    ``["bind_within_period", "bind_using_blended_weights",
    "bind_within_solve"]`` must collapse to
    ``"bind_using_blended_weights"`` because it ranks highest in the
    priority list — even though it appears second in the array
    (priority is by *method*, not array position).
    """
    url = _build_v53_db_with_array_value(
        tmp_path / "v54_priority.sqlite",
        entity_name="ARG_H2",
        array_members=[
            "bind_within_period",
            "bind_using_blended_weights",
            "bind_within_solve",
        ],
    )

    # Sanity-check the pre-migration state matches what we planted.
    pre_type, _ = _read_storage_binding_value(url, "ARG_H2")
    assert pre_type == "array", (
        f"expected planted row to be type='array', got {pre_type!r}"
    )

    migrate_database(url, up_to=54)

    post_type, post_value = _read_storage_binding_value(url, "ARG_H2")
    assert post_type == "str", (
        "v54 migration must rewrite type to 'str', got "
        f"{post_type!r}"
    )
    assert post_value == "bind_using_blended_weights", (
        "v54 migration must pick the highest-priority element from the "
        "array; got "
        f"{post_value!r} (expected 'bind_using_blended_weights')"
    )


def test_v54_preserves_existing_scalar_values(tmp_path: Path) -> None:
    """Rows already at ``type=='str'`` must be left untouched — the
    v54 step is a no-op for the battery-shaped entries in
    H2_trade.sqlite (``ARG_Battery`` etc. at ``bind_within_timeset``).
    """
    url = json_to_db(_BASE_FIXTURE, tmp_path / "v54_scalar_preserve.sqlite")
    # For the scalar-preserve case we can stage at v53 (the wiring is
    # OK with member strings) or v52; staging at v53 verifies that the
    # post-wiring write path accepts a valid scalar.
    migrate_database(url, up_to=53)

    scalar_value, scalar_type = to_database("bind_within_timeset")
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            entities=[("node", "ARG_Battery")],
            parameter_values=[
                (
                    "node",
                    "ARG_Battery",
                    "storage_binding_method",
                    from_database(scalar_value, scalar_type),
                ),
            ],
        )
        assert not errors, f"data import errors: {errors}"
        db.commit_session("plant scalar value")

    migrate_database(url, up_to=54)

    post_type, post_value = _read_storage_binding_value(url, "ARG_Battery")
    assert post_type == "str"
    assert post_value == "bind_within_timeset", (
        "v54 must not mutate already-scalar rows; got "
        f"{post_value!r}"
    )


def test_v54_rejects_unknown_method_array(tmp_path: Path) -> None:
    """An array containing only strings outside the known method set
    must raise :class:`SpineDBAPIError` naming the entity and the
    unknown contents.  The migration refuses to guess.
    """
    url = _build_v53_db_with_array_value(
        tmp_path / "v54_unknown.sqlite",
        entity_name="MysteryNode",
        array_members=["bind_unknown_alpha", "bind_unknown_beta"],
    )

    with pytest.raises(SpineDBAPIError) as exc_info:
        migrate_database(url, up_to=54)

    msg = str(exc_info.value)
    assert "MysteryNode" in msg, (
        "unknown-method error must name the offending entity; got:\n"
        + msg
    )
    assert "bind_unknown_alpha" in msg, msg
    assert "bind_unknown_beta" in msg, msg
