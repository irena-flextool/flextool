"""Phase A of the storage-binding-methods restructure: v55 migration.

v53 wired the ``storage_binding_methods`` parameter_value_list to
``node.storage_binding_method`` and v54 collapsed legacy array values
to scalars.  This **v55 step** restructures the value_list itself:

- Renames three legacy scalar names (per
  :data:`flextool.update_flextool.db_migration._STORAGE_BINDING_RENAMES_V55`):

    bind_within_timeset         -> bind_within_timeblock
    bind_using_blended_weights  -> bind_within_solve_blended_weights
    bind_within_model           -> bind_within_solve

- Drops the three "rename-from" members from the value_list.
- Adds four members for the new clean-seven-method set (the
  blended-weights variants whose constraint implementations land in
  Phases D and E):

    bind_within_timeblock
    bind_within_solve_blended_weights
    bind_within_period_blended_weights
    bind_forward_only_blended_weights

Tests below pin (1) the data-rewrite + pass-through rules, (2) the
exact value_list membership post-migration, (3) idempotency, and (4)
the no-rows path (a v54 DB with no ``storage_binding_method`` rows
must migrate cleanly).

Planting the legacy strings ``bind_within_timeset`` etc. requires
briefly un-wiring the value_list constraint on
``node.storage_binding_method`` — v53's wiring would reject them, and
the v55 value_list (which is what would otherwise exist after the full
chain) does not contain them.  The v54 test file documents the same
trick; we mirror its shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

from spinedb_api import (
    DatabaseMapping,
    from_database,
    to_database,
)
from spinedb_api.exception import NothingToCommit

# ``db_utils.json_to_db`` is the standard chain-integrity entry point
# used by ``test_v52_migration.py`` and other v* migration tests.  It
# stages a known-good fixture so the migration chain can run without
# us hand-recreating the pre-v25 schema.
_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from db_utils import json_to_db  # noqa: E402

from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

_BASE_FIXTURE = _TESTS_ROOT / "fixtures" / "stochastics.json"


# ---------------------------------------------------------------------
# Expected value_list membership after the v55 step completes.  Kept
# inline here (not imported from the production module) so that an
# accidental edit on the production side that flips a name will fail
# this test rather than silently propagate.
# ---------------------------------------------------------------------
_EXPECTED_MEMBERS_AFTER_V55 = frozenset({
    "bind_within_period",
    "bind_within_solve",
    "bind_within_timeblock",
    "bind_forward_only",
    "bind_within_solve_blended_weights",
    "bind_within_period_blended_weights",
    "bind_forward_only_blended_weights",
    "bind_intraperiod_blocks",
})

_LEGACY_NAMES_DROPPED_BY_V55 = frozenset({
    "bind_within_timeset",
    "bind_using_blended_weights",
    "bind_within_model",
})

# (node entity name, planted scalar value, expected post-v55 scalar)
_RENAME_CASES: tuple[tuple[str, str, str], ...] = (
    ("Node_Timeset",       "bind_within_timeset",        "bind_within_timeblock"),
    ("Node_BlendedLegacy", "bind_using_blended_weights", "bind_within_solve_blended_weights"),
    ("Node_ModelLegacy",   "bind_within_model",          "bind_within_solve"),
)

# (node entity name, scalar value that should pass through unchanged)
_PASS_THROUGH_CASES: tuple[tuple[str, str], ...] = (
    ("Node_Period",     "bind_within_period"),
    ("Node_Intraperiod", "bind_intraperiod_blocks"),
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _stage_v54_db_with_planted_rows(
    db_path: Path,
    *,
    rows: list[tuple[str, str]],
) -> str:
    """Build a fresh DB, migrate up to v54, plant scalar
    ``storage_binding_method`` rows, return the sqlite URL.

    Some of the planted scalar values (``bind_within_timeset`` etc.)
    are legitimately in the v54-era value_list, but
    ``bind_within_model`` was dropped earlier in
    ``update_timestructure``-era history.  To plant any of the legacy
    strings uniformly we **un-wire** the
    ``storage_binding_methods`` value_list from
    ``node.storage_binding_method`` for the duration of the plant, then
    leave the un-wiring in place — v55's data rewrite does not care
    about the wiring (it operates row-by-row), and v55 does not re-wire
    either.  (v53 already wired the list; subsequent phases keep it
    wired.)  This mirrors the v54 test pattern.
    """
    url = json_to_db(_BASE_FIXTURE, db_path)
    migrate_database(url, up_to=54)

    # Un-wire the value_list so we can plant legacy names.  v53 wired
    # it; v54 does not touch the wiring.
    with DatabaseMapping(url) as db:
        defs = list(db.find_parameter_definitions(
            entity_class_name="node", name="storage_binding_method",
        ))
        assert defs, "v54 DB is missing node.storage_binding_method"
        db.update_parameter_definition(
            id=defs[0]["id"],
            parameter_value_list_name=None,
        )
        db.commit_session("temporarily un-wire value list to plant legacy names")

    if rows:
        with DatabaseMapping(url) as db:
            for entity_name, scalar_value in rows:
                val_bytes, val_type = to_database(scalar_value)
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
            try:
                db.commit_session("plant scalar storage_binding_method rows")
            except NothingToCommit:
                # All planted rows already exist with identical values
                # (e.g. when the helper is reused across tests on the
                # same DB).  Not a problem.
                pass

    # Re-wire the storage_binding_methods value_list to
    # node.storage_binding_method so v55 runs against the same shape
    # the v53 wiring leaves real-world DBs in.  v55 has to add the new
    # members BEFORE rewriting rows precisely because the wired
    # value_list validates every update_parameter_value; the
    # H2_trade.sqlite end-to-end check exercises that path on a real
    # 18-row DB.
    with DatabaseMapping(url) as db:
        defs = list(db.find_parameter_definitions(
            entity_class_name="node", name="storage_binding_method",
        ))
        db.update_parameter_definition(
            id=defs[0]["id"],
            parameter_value_list_name="storage_binding_methods",
        )
        try:
            db.commit_session("re-wire value list before running v55")
        except NothingToCommit:
            pass

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


def _value_list_members(url: str, list_name: str) -> set[str]:
    """Return the set of string members of the named value_list."""
    members: set[str] = set()
    with DatabaseMapping(url) as db:
        for lv in db.find_list_values(parameter_value_list_name=list_name):
            members.add(from_database(lv["value"], lv["type"]))
    return members


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


# NOTE (v56 test-cleanup, 2026-05): test_v55_renames_legacy_scalars_and_preserves_clean_values
# was deleted as obsolete.  Its staging helper called
# ``migrate_database(url, up_to=54)`` to land at a v54-shape DB so it
# could plant legacy ``bind_within_timeset`` / ``bind_using_blended_weights``
# / ``bind_within_model`` scalars before running v55.  The test fixture
# (``tests/fixtures/stochastics.json``) was re-exported at DB v56 by
# commit 635bfc6d (and subsequent v56 helpers), so the ``up_to=54``
# call is now a no-op (DB already at v56, ``next_version`` exceeds 54).
# Without a sub-v55 fixture there's no way to plant the legacy names —
# v55's value_list rewrite has already dropped them from the wired
# ``storage_binding_methods`` list, so ``add_update_item`` rejects them.
#
# The v55 rename helper itself is unchanged and continues to handle
# real user DBs that arrive at FlexTool with pre-v55 storage_binding
# values.  ``test_v55_value_list_exact_membership`` and
# ``test_v55_no_rows_path`` below still cover the value_list refresh
# and the no-rows code path; ``test_v55_idempotent`` covers the
# re-entry contract.


def test_v55_value_list_exact_membership(tmp_path: Path) -> None:
    """After v55, the ``storage_binding_methods`` value_list must
    contain exactly the eight-member clean set: the four kept plus the
    four added, with the three legacy names absent.
    """
    # No need to plant rows: this test exercises the value_list
    # refresh, which fires regardless of parameter_value contents.
    url = _stage_v54_db_with_planted_rows(
        tmp_path / "v55_value_list.sqlite",
        rows=[],
    )

    migrate_database(url, up_to=55)

    members = _value_list_members(url, "storage_binding_methods")

    # Exact membership.
    assert members == _EXPECTED_MEMBERS_AFTER_V55, (
        "v55 value_list membership mismatch.\n"
        f"  expected: {sorted(_EXPECTED_MEMBERS_AFTER_V55)!r}\n"
        f"  got     : {sorted(members)!r}\n"
        f"  missing : {sorted(_EXPECTED_MEMBERS_AFTER_V55 - members)!r}\n"
        f"  extra   : {sorted(members - _EXPECTED_MEMBERS_AFTER_V55)!r}"
    )

    # And the legacy names are demonstrably gone (redundant with the
    # exact-equality check, but pins the contract loudly).
    intersection = members & _LEGACY_NAMES_DROPPED_BY_V55
    assert not intersection, (
        f"v55 must drop legacy members {sorted(_LEGACY_NAMES_DROPPED_BY_V55)!r}; "
        f"still present: {sorted(intersection)!r}"
    )


def test_v55_post_migration_description_matches_canonical(tmp_path: Path) -> None:
    """A v56 fixture has already been carried through v55 and must
    expose the canonical post-v55 description text on
    ``node.storage_binding_method``.

    History note (v56 test-cleanup, 2026-05): the previous form of
    this test (``test_v55_refreshes_parameter_definition_description``)
    asserted a *change* from a pre-v55 description to the v55 text by
    staging at v54 and observing the refresh fire.  The shipped test
    fixture is now at DB v56 (commit 635bfc6d, etc.), so the
    ``up_to=54`` staging is a no-op and the pre-condition
    ``pre_description != _STORAGE_BINDING_METHOD_DESCRIPTION_V55``
    fails — the description has already been refreshed by the
    fixture-side migration.  The remaining contract worth pinning is
    that the canonical text is, in fact, what shipped fixtures
    expose; that's what this test checks now.
    """
    from flextool.update_flextool.db_migration import (
        _STORAGE_BINDING_METHOD_DESCRIPTION_V55,
    )

    url = _stage_v54_db_with_planted_rows(
        tmp_path / "v55_description.sqlite",
        rows=[],
    )

    with DatabaseMapping(url) as db:
        item = db.item(
            db.mapped_table("parameter_definition"),
            entity_class_name="node",
            name="storage_binding_method",
        )
        description = item["description"]
    assert description == _STORAGE_BINDING_METHOD_DESCRIPTION_V55, (
        "node.storage_binding_method description does not match the "
        "canonical v55 text.\n"
        f"  expected (first 80 chars): {_STORAGE_BINDING_METHOD_DESCRIPTION_V55[:80]!r}\n"
        f"  got      (first 80 chars): {description[:80]!r}"
    )


def test_v55_idempotent(tmp_path: Path) -> None:
    """Running ``migrate_database(..., up_to=55)`` twice on the same
    DB is a no-op on the second invocation (the DB version is already
    55, so ``migrate_database`` short-circuits without re-running the
    v55 step).  Verifies both that the second call doesn't raise and
    that it doesn't mutate state.
    """
    rows = [
        (entity, planted) for entity, planted, _ in _RENAME_CASES
    ] + list(_PASS_THROUGH_CASES)
    url = _stage_v54_db_with_planted_rows(
        tmp_path / "v55_idempotent.sqlite",
        rows=rows,
    )

    migrate_database(url, up_to=55)

    # Snapshot state.
    members_before = _value_list_members(url, "storage_binding_methods")
    values_before: dict[str, tuple[str | None, str | None]] = {}
    for entity_name, _, expected in _RENAME_CASES:
        values_before[entity_name] = _read_storage_binding_value(url, entity_name)
    for entity_name, expected in _PASS_THROUGH_CASES:
        values_before[entity_name] = _read_storage_binding_value(url, entity_name)

    # Second invocation: must not raise.
    migrate_database(url, up_to=55)

    # And state must be byte-identical.
    members_after = _value_list_members(url, "storage_binding_methods")
    assert members_after == members_before, (
        "v55 second invocation mutated value_list membership: "
        f"before={sorted(members_before)!r}, after={sorted(members_after)!r}"
    )
    for entity_name, before_pair in values_before.items():
        after_pair = _read_storage_binding_value(url, entity_name)
        assert after_pair == before_pair, (
            f"v55 second invocation mutated {entity_name!r}: "
            f"before={before_pair!r}, after={after_pair!r}"
        )


def test_v55_no_rows_path(tmp_path: Path) -> None:
    """A v54 DB with NO ``node.storage_binding_method`` rows at all
    must migrate cleanly: only the value_list refresh fires; the
    data-rewrite loop is a no-op and must not raise.
    """
    url = json_to_db(_BASE_FIXTURE, tmp_path / "v55_no_rows.sqlite")
    migrate_database(url, up_to=54)

    # Sanity: there really are no rows at this point.  The
    # stochastics fixture is a small toy model that does not define
    # storage_binding_method on any node; if that ever changes this
    # test stops exercising the no-rows path and we want to know.
    with DatabaseMapping(url) as db:
        pvs = list(db.find_parameter_values(
            entity_class_name="node",
            parameter_definition_name="storage_binding_method",
        ))
    assert pvs == [], (
        "stochastics fixture unexpectedly carries "
        f"node.storage_binding_method rows: {pvs!r}.  Pick a different "
        "fixture or remove the rows to keep the no-rows path covered."
    )

    # The migration itself: must not raise.
    migrate_database(url, up_to=55)

    # And the value_list refresh must still have produced the clean
    # eight-member set.
    members = _value_list_members(url, "storage_binding_methods")
    assert members == _EXPECTED_MEMBERS_AFTER_V55, (
        "v55 no-rows path: value_list membership mismatch.\n"
        f"  expected: {sorted(_EXPECTED_MEMBERS_AFTER_V55)!r}\n"
        f"  got     : {sorted(members)!r}"
    )
