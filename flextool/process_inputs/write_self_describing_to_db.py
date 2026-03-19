"""Write SheetData objects (from read_self_describing_excel) to a SpineDB.

The writer uses a two-pass import strategy:
  1. First pass  — sheets with 1-dimensional entity classes (node, unit, …)
  2. Second pass — sheets with multi-dimensional entity classes (unit__inputNode, …)

This guarantees that base entities exist before relationship entities reference them.

Entity classes and parameter definitions are assumed to already exist in the
target database (initialized from the FlexTool template).  This module only
adds entities, alternatives, entity_alternatives, and parameter_values.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from spinedb_api import Asterisk, DatabaseMapping, Map, SpineDBAPIError, to_database
from spinedb_api.exception import NothingToCommit

from flextool.process_inputs.read_self_describing_excel import SheetData

logger = logging.getLogger(__name__)

# The special pseudo-parameter that maps to entity_alternative rows.
_ENTITY_EXISTENCE = "entity existence"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_multidimensional(sheet: SheetData) -> bool:
    """Return True if the sheet's entity class has more than one dimension."""
    for ec in sheet.metadata.entity_classes:
        if len(ec.dimensions) > 1:
            return True
    return False


def _group_map_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into scalar records and grouped Map records.

    Scalar records have no ``index_value``.  Map records share the same
    (entity_class, entity_byname, param_name, alternative) key and are
    collected into a single Map value per group.

    Returns:
        (scalar_records, map_records) where each map_record is a dict with
        keys identical to an input record but ``value`` replaced by a
        :class:`spinedb_api.Map` built from all the grouped index entries.
    """
    scalars: list[dict[str, Any]] = []
    # key -> list of (index_value, value) in insertion order
    map_groups: dict[tuple, list[tuple[str, Any]]] = defaultdict(list)
    # key -> first record (used as template for the collapsed record)
    map_templates: dict[tuple, dict[str, Any]] = {}

    for rec in records:
        if rec["param_name"] == _ENTITY_EXISTENCE:
            scalars.append(rec)
            continue

        if rec.get("index_value") is not None:
            key = (
                rec["entity_class"],
                rec["entity_byname"],
                rec["param_name"],
                rec["alternative"],
            )
            map_groups[key].append((rec["index_value"], rec["value"]))
            if key not in map_templates:
                map_templates[key] = rec
        else:
            scalars.append(rec)

    # Build collapsed Map records
    collapsed: list[dict[str, Any]] = []
    for key, entries in map_groups.items():
        indexes = [e[0] for e in entries]
        values = [e[1] for e in entries]
        index_name = map_templates[key].get("index_name") or "index"
        map_value = Map(indexes=indexes, values=values, index_name=index_name)
        rec = dict(map_templates[key])
        rec["value"] = map_value
        rec["index_value"] = None  # no longer individual
        collapsed.append(rec)

    return scalars, collapsed


# ---------------------------------------------------------------------------
# Core writing logic for one sheet
# ---------------------------------------------------------------------------


def _write_sheet(
    sheet: SheetData,
    db: DatabaseMapping,
    alternatives_added: set[str],
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write a single SheetData into *db*.

    The caller-owned sets track what has already been added so that
    duplicates are silently skipped.
    """
    # -- Link sheets (entity relationships only, no parameters) -------------
    if sheet.link_entities:
        for ec in sheet.metadata.entity_classes:
            for byname in sheet.link_entities:
                key = (ec.class_name, byname)
                if key not in entities_added:
                    entities_added.add(key)
                    try:
                        db.add_entity(
                            entity_class_name=ec.class_name,
                            entity_byname=byname,
                        )
                    except SpineDBAPIError as exc:
                        logger.warning(
                            "Could not add link entity %s (%s): %s",
                            byname, ec.class_name, exc,
                        )
        return

    if not sheet.records:
        return

    # -- Alternatives -------------------------------------------------------
    for rec in sheet.records:
        alt = rec.get("alternative")
        if alt and alt not in alternatives_added:
            alternatives_added.add(alt)
            try:
                db.add_alternative(name=alt)
            except SpineDBAPIError as exc:
                logger.warning("Could not add alternative '%s': %s", alt, exc)

    # -- Entities -----------------------------------------------------------
    for rec in sheet.records:
        ec_name = rec.get("entity_class")
        byname = rec.get("entity_byname")
        if ec_name and byname:
            key = (ec_name, byname)
            if key not in entities_added:
                entities_added.add(key)
                try:
                    db.add_entity(
                        entity_class_name=ec_name,
                        entity_byname=byname,
                    )
                except SpineDBAPIError as exc:
                    logger.warning(
                        "Could not add entity %s (%s): %s",
                        byname, ec_name, exc,
                    )

    # -- Split records into scalars and maps --------------------------------
    scalars, maps = _group_map_records(sheet.records)

    # -- Entity existence (entity_alternative) ------------------------------
    for rec in scalars:
        if rec["param_name"] != _ENTITY_EXISTENCE:
            continue
        ec_name = rec["entity_class"]
        byname = rec["entity_byname"]
        alt = rec["alternative"]
        active = bool(rec["value"])
        ea_key = (ec_name, byname, alt)
        if ea_key not in entity_alts_added:
            entity_alts_added.add(ea_key)
            try:
                db.add_entity_alternative(
                    entity_class_name=ec_name,
                    entity_byname=byname,
                    alternative_name=alt,
                    active=active,
                )
            except SpineDBAPIError as exc:
                logger.warning(
                    "Could not add entity_alternative %s %s '%s': %s",
                    ec_name, byname, alt, exc,
                )

    # -- Scalar parameter values --------------------------------------------
    for rec in scalars:
        if rec["param_name"] == _ENTITY_EXISTENCE:
            continue
        value, type_ = to_database(rec["value"])
        try:
            db.add_parameter_value(
                entity_class_name=rec["entity_class"],
                entity_byname=rec["entity_byname"],
                parameter_definition_name=rec["param_name"],
                alternative_name=rec["alternative"],
                value=value,
                type=type_,
            )
        except SpineDBAPIError as exc:
            logger.warning(
                "Could not add parameter value %s.%s (%s, alt=%s): %s",
                rec["entity_class"], rec["param_name"],
                rec["entity_byname"], rec["alternative"], exc,
            )

    # -- Map parameter values -----------------------------------------------
    for rec in maps:
        value, type_ = to_database(rec["value"])
        try:
            db.add_parameter_value(
                entity_class_name=rec["entity_class"],
                entity_byname=rec["entity_byname"],
                parameter_definition_name=rec["param_name"],
                alternative_name=rec["alternative"],
                value=value,
                type=type_,
            )
        except SpineDBAPIError as exc:
            logger.warning(
                "Could not add map parameter %s.%s (%s, alt=%s): %s",
                rec["entity_class"], rec["param_name"],
                rec["entity_byname"], rec["alternative"], exc,
            )


# ---------------------------------------------------------------------------
# Ensure 0-dimensional entities referenced by multi-dim sheets
# ---------------------------------------------------------------------------


def _ensure_dimension_entities(
    data_sheets_nd: list[SheetData],
    db: DatabaseMapping,
    entities_added: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Create any missing 0-dim entities referenced as dimensions in multi-dim sheets.

    Entity classes like ``reserve``, ``upDown``, ``model``, ``group``, ``commodity``
    may have entities that only appear as dimension elements in multi-dimensional
    relationship entities (e.g. ``reserve__upDown__unit__node``).  They never get
    their own sheet, so they would be missing after pass 1.

    This function scans all multi-dim SheetData records *and* link_entities and
    adds the missing 0-dim entities before pass 2 creates the relationships.
    """
    for sheet in data_sheets_nd:
        for ec in sheet.metadata.entity_classes:
            if len(ec.dimensions) <= 1:
                continue

            # From regular records
            for rec in sheet.records:
                if rec.get("entity_class") != ec.class_name:
                    continue
                byname = rec.get("entity_byname")
                if not byname or len(byname) != len(ec.dimensions):
                    continue
                for dim_name, elem_name in zip(ec.dimensions, byname):
                    if not elem_name:
                        continue
                    key = (dim_name, (elem_name,))
                    if key not in entities_added:
                        entities_added.add(key)
                        try:
                            db.add_entity(
                                entity_class_name=dim_name,
                                entity_byname=(elem_name,),
                            )
                        except SpineDBAPIError:
                            pass  # already exists or class doesn't exist

            # From link_entities
            for byname in sheet.link_entities:
                if len(byname) != len(ec.dimensions):
                    continue
                for dim_name, elem_name in zip(ec.dimensions, byname):
                    if not elem_name:
                        continue
                    key = (dim_name, (elem_name,))
                    if key not in entities_added:
                        entities_added.add(key)
                        try:
                            db.add_entity(
                                entity_class_name=dim_name,
                                entity_byname=(elem_name,),
                            )
                        except SpineDBAPIError:
                            pass  # already exists or class doesn't exist


# ---------------------------------------------------------------------------
# Scenario sheet handling
# ---------------------------------------------------------------------------


def _write_scenarios(
    sheet: SheetData,
    db: DatabaseMapping,
    alternatives_added: set[str],
) -> None:
    """Handle a scenario sheet.

    Scenario sheets have records where entity_class == 'scenario'.
    Record structure (from parse_scenario_sheet):
      entity_class: 'scenario'
      entity_byname: ('scenario_name',)
      param_name: row label (e.g. 'base_alternative', 'alternative_1')
      alternative: the alternative name
      value: rank (int)
    """
    if not sheet.records:
        return

    # Group records by scenario name → list of (alternative_name, rank)
    scenario_alts: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for rec in sheet.records:
        scenario_name = rec.get("entity_byname")
        if scenario_name and isinstance(scenario_name, tuple):
            scenario_name = scenario_name[0]

        if not scenario_name:
            continue

        # The 'alternative' field holds the actual alternative name
        alt_name = rec.get("alternative")
        value = rec.get("value")

        if not alt_name or alt_name == _ENTITY_EXISTENCE:
            continue

        if isinstance(value, (int, float)):
            rank = int(value)
        else:
            rank = 0

        scenario_alts[scenario_name].append((alt_name, rank))

    # Write scenarios and their alternatives
    for scenario_name, alt_ranks in scenario_alts.items():
        try:
            db.add_scenario(name=scenario_name)
        except SpineDBAPIError as exc:
            logger.warning("Could not add scenario '%s': %s", scenario_name, exc)

        # Sort by rank before adding
        alt_ranks.sort(key=lambda x: x[1])
        for rank, (alt_name, _original_rank) in enumerate(alt_ranks):
            # Ensure the alternative exists
            if alt_name not in alternatives_added:
                alternatives_added.add(alt_name)
                try:
                    db.add_alternative(name=alt_name)
                except SpineDBAPIError as exc:
                    logger.warning(
                        "Could not add alternative '%s': %s", alt_name, exc,
                    )
            try:
                db.add_scenario_alternative(
                    scenario_name=scenario_name,
                    alternative_name=alt_name,
                    rank=rank,
                )
            except SpineDBAPIError as exc:
                logger.warning(
                    "Could not add scenario_alternative %s → %s (rank %d): %s",
                    scenario_name, alt_name, rank, exc,
                )


def _is_scenario_sheet(sheet: SheetData) -> bool:
    """Return True if the sheet looks like a scenario definition sheet."""
    name = sheet.sheet_name.lower()
    if "scenario" in name:
        return True
    # Also check entity class names
    for ec in sheet.metadata.entity_classes:
        if "scenario" in ec.class_name.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Purge helper
# ---------------------------------------------------------------------------


def _purge_database(db: DatabaseMapping) -> None:
    """Remove all entities, alternatives, scenarios, and parameter values."""
    try:
        db.remove_parameter_value(id=Asterisk)
        db.remove_entity(id=Asterisk)
        db.remove_alternative(id=Asterisk)
        db.remove_scenario(id=Asterisk)
        db.commit_session("Purged database before import")
    except NothingToCommit:
        pass
    except SpineDBAPIError as exc:
        raise RuntimeError(f"Failed to purge database: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_sheet_data_to_db(
    sheet_data_list: list[SheetData],
    db_url: str,
    purge_first: bool = True,
) -> None:
    """Write parsed SheetData objects into a SpineDB.

    Uses a two-pass strategy so that 1-dimensional entities are created
    before multi-dimensional (relationship) entities that reference them.

    Args:
        sheet_data_list: Output of :func:`read_self_describing_excel`.
        db_url: SQLAlchemy-style database URL, e.g.
            ``sqlite:///path/to/db.sqlite``.
        purge_first: If True, remove all existing data from the database
            before writing (keeps entity classes and parameter definitions).
    """
    with DatabaseMapping(db_url, create=False, upgrade=True) as db:
        if purge_first:
            _purge_database(db)

        # Tracking sets to avoid duplicate add calls
        alternatives_added: set[str] = set()
        entities_added: set[tuple[str, tuple[str, ...]]] = set()
        entity_alts_added: set[tuple[str, tuple[str, ...], str]] = set()

        # Separate scenario sheets from data sheets
        scenario_sheets: list[SheetData] = []
        data_sheets_1d: list[SheetData] = []
        data_sheets_nd: list[SheetData] = []

        for sheet in sheet_data_list:
            if _is_scenario_sheet(sheet):
                scenario_sheets.append(sheet)
            elif _is_multidimensional(sheet):
                data_sheets_nd.append(sheet)
            else:
                data_sheets_1d.append(sheet)

        # Pass 1: 1-dimensional entity classes
        for sheet in data_sheets_1d:
            logger.info(
                "Pass 1 — writing sheet '%s' (%d records, %d links)",
                sheet.sheet_name, len(sheet.records), len(sheet.link_entities),
            )
            _write_sheet(sheet, db, alternatives_added, entities_added, entity_alts_added)

        # Pass 1.5: ensure 0-dim entities referenced by multi-dim sheets
        _ensure_dimension_entities(data_sheets_nd, db, entities_added)

        # Pass 2: multi-dimensional entity classes
        for sheet in data_sheets_nd:
            logger.info(
                "Pass 2 — writing sheet '%s' (%d records, %d links)",
                sheet.sheet_name, len(sheet.records), len(sheet.link_entities),
            )
            _write_sheet(sheet, db, alternatives_added, entities_added, entity_alts_added)

        # Pass 3: scenario sheets
        for sheet in scenario_sheets:
            logger.info(
                "Writing scenario sheet '%s' (%d records)",
                sheet.sheet_name, len(sheet.records),
            )
            _write_scenarios(sheet, db, alternatives_added)

        # Commit everything
        try:
            db.commit_session("Imported data from self-describing Excel")
            logger.info("Successfully committed all imported data")
        except NothingToCommit:
            logger.info("No new data to commit")
        except SpineDBAPIError as exc:
            raise RuntimeError(f"Failed to commit imported data: {exc}") from exc
