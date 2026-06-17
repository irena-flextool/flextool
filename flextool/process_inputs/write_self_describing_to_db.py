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

from spinedb_api import Array, Asterisk, DatabaseMapping, Map, SpineDBAPIError, to_database
from spinedb_api.exception import NothingToCommit

from flextool.process_inputs.read_self_describing_excel import SheetData

logger = logging.getLogger(__name__)

# The special pseudo-parameter that maps to entity_alternative rows.
_ENTITY_EXISTENCE = "entity existence"

# Index-cell marker the exporter writes for a zero-length Array/Map value.
# Mirrors excel_writer.EMPTY_COLLECTION_SENTINEL — keep the two in sync.
_EMPTY_COLLECTION_SENTINEL = "(empty)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _natural_key(v: Any) -> tuple[int, Any]:
    """Sort key that orders integer-like strings numerically.

    Map/Array index values arrive as strings from the Excel reader.  A plain
    string sort orders an integer axis (e.g. ladder tiers) lexically —
    ``"1","10","11",…,"2"`` — silently reordering it versus the authored
    ``"1","2",…,"10","11"``.  Returns ``(0, int)`` for integer-like values so
    they sort numerically ahead of, and separately from, genuine strings
    ``(1, str)`` — keeping the order total and deterministic for mixed axes.
    """
    try:
        return (0, int(str(v)))
    except (TypeError, ValueError):
        return (1, str(v))


def _is_multidimensional(sheet: SheetData) -> bool:
    """Return True if the sheet's entity class has more than one dimension."""
    for ec in sheet.metadata.entity_classes:
        if len(ec.dimensions) > 1:
            return True
    return False


def _registry_facet_targets() -> "dict[tuple[str, str], list[tuple[str, tuple[str, ...], frozenset[str]]]]":
    """Return the registry's facet-target map.

    Keys are ``(entity_class, facet_key)`` — e.g. ``("commodity", "price")`` and
    ``("commodity", "quantity")``.  Values are lists of
    ``(canonical_param_name, outer_axis_names, facet_keys)`` tuples covering
    EVERY allowed shape with a facet leaf for that entity class.  The
    caller picks the right canonical name by matching the observed outer
    axis vocabulary to ``outer_axis_names`` (e.g. ``("tier",)`` vs
    ``("period", "tier")``).
    """
    try:
        from flextool.engine_polars._param_shapes import (
            LeafKind,
            PARAM_ALLOWED_SHAPES,
            facet_keys,
            shape_to_axes,
        )
    except Exception:  # pragma: no cover — registry must be importable
        return {}

    out: dict[tuple[str, str], list[tuple[str, tuple[str, ...], frozenset[str]]]] = {}
    axis_to_label = {"d": "period", "t": "time", "i": "tier"}
    for (ec, pname), shapes in PARAM_ALLOWED_SHAPES.items():
        for shape in shapes:
            axes = shape_to_axes(shape)
            if axes.leaf is not LeafKind.FACET_PRICE_QUANTITY:
                continue
            outer = tuple(
                axis_to_label.get(getattr(ax, "value", ""), getattr(ax, "value", ""))
                for ax in axes.map_levels
            )
            fkeys = facet_keys(axes.leaf)
            for fk in fkeys:
                out.setdefault((ec, fk), []).append((pname, outer, fkeys))
    return out


def _index_tuple(rec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Build the (axis_name, value) tuple for *rec* in left-to-right order.

    Combines ``extra_index_values`` (left of rightmost) with the
    ``(index_name, index_value)`` pair from the standard record fields.
    """
    parts: list[tuple[str, str]] = []
    for axis, val in rec.get("extra_index_values") or []:
        if val is not None:
            parts.append((axis, val))
    if rec.get("index_name") and rec.get("index_value") is not None:
        parts.append((rec["index_name"], rec["index_value"]))
    return tuple(parts)


def _combine_facet_records(
    records: list[dict[str, Any]],
    sheet_name: str = "",
) -> list[dict[str, Any]]:
    """Combine facet-leaf records emitted by the ladder reader into single
    nested-Map records per canonical parameter.

    The reader treats the two ``parameter: price`` / ``parameter: quantity``
    Excel columns as separate parameter records.  Spine encodes them as a
    single nested Map under a canonical name (e.g.
    ``commodity.price_ladder_cumulative``) — the registry tells us which.

    Each ladder parameter lives on its own dedicated sheet whose title IS the
    canonical parameter name, so ``sheet_name`` is the authoritative
    disambiguator: when two canonical params share the same outer-axes shape
    (e.g. both ``price_ladder_annual`` and ``price_ladder_cumulative`` allow a
    2d ``(tier,)`` Map), matching on ``outer_axes`` alone would silently pick
    whichever the registry happens to list first, relabelling the parameter.

    Records that don't match the registry's facet vocabulary pass through
    unchanged.
    """
    facet_targets = _registry_facet_targets()
    if not facet_targets:
        return records

    # Bucket key: (entity_class, entity_byname, alternative, outer_index_tuple)
    # where outer_index_tuple is the (axis, value)... part of the row.
    BucketKey = tuple[str, tuple, str, tuple[tuple[str, str], ...]]
    buckets: dict[BucketKey, dict[str, Any]] = {}
    # Track which record-list positions belong to each bucket, so we can
    # drop them in one go.
    bucket_indices: dict[BucketKey, list[int]] = {}
    # Per (entity_class, entity_byname, alt) — track the outer-axes
    # signature so all rows in one group share the same canonical param.
    group_axes: dict[tuple[str, tuple, str], tuple[str, ...]] = {}

    drop_indices: set[int] = set()
    out: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        pname = rec.get("param_name") or ""
        ec = rec.get("entity_class") or ""
        targets = facet_targets.get((ec, pname))
        if not targets:
            continue
        # This record is a facet-leaf candidate.  We must still check that
        # the index axes line up with a registered shape.
        idx_pairs = _index_tuple(rec)
        outer_axes = tuple(axis for axis, _ in idx_pairs)
        # Pick the registered (canonical_name, outer_axes, fkeys) entry
        # whose outer_axes matches our observed axes.  The facet axes
        # never appear in the index tuple (they're separate columns), so
        # equality is exact.  When several canonical params share the same
        # outer-axes shape, prefer the one whose name matches the source
        # sheet (the dedicated ladder sheet is named after its parameter);
        # fall back to the first outer-axes match only when the sheet name
        # is unknown or doesn't correspond to a candidate.
        match: tuple[str, tuple[str, ...], frozenset[str]] | None = None
        for cand in targets:
            if cand[1] == outer_axes and cand[0] == sheet_name:
                match = cand
                break
        if match is None:
            for cand in targets:
                if cand[1] == outer_axes:
                    match = cand
                    break
        if match is None:
            # No registered shape matches; treat as a normal record.
            continue

        canonical_name, _axes, fkeys = match
        alt = rec.get("alternative") or ""
        ent = rec.get("entity_byname") or ()
        group_key = (ec, ent, alt)
        prior = group_axes.get(group_key)
        if prior is not None and prior != outer_axes:
            # Same (entity, alt) carries TWO different outer-axes shapes.
            # The registry allows depth-2 and depth-3 to coexist on the
            # same entity (price_ladder_annual), but only one canonical
            # name per (entity, alt) — we refuse to merge silently in
            # that case.  Surface a warning and keep the records as-is.
            logger.warning(
                "Facet records for %s.%s under (%s, alt=%s) carry mixed "
                "index axes %s vs %s; skipping facet combination.",
                ec, canonical_name, ent, alt, prior, outer_axes,
            )
            continue
        group_axes[group_key] = outer_axes

        bkey: BucketKey = (ec, ent, alt, idx_pairs)
        bucket = buckets.get(bkey)
        if bucket is None:
            bucket = {
                "canonical_name": canonical_name,
                "facet_keys": fkeys,
                "facet_values": {},
                "template": rec,
            }
            buckets[bkey] = bucket
            bucket_indices[bkey] = []
        if pname in fkeys:
            bucket["facet_values"][pname] = rec.get("value")
            bucket_indices[bkey].append(i)
            drop_indices.add(i)

    if not buckets:
        return records

    # Group buckets by (entity_class, entity_byname, alt, canonical_name)
    # to build one Map per canonical record.
    GroupKey = tuple[str, tuple, str, str]
    grouped: dict[GroupKey, list[tuple[tuple[tuple[str, str], ...], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for bkey, bucket in buckets.items():
        ec, ent, alt, idx_pairs = bkey
        gkey = (ec, ent, alt, bucket["canonical_name"])
        grouped[gkey].append((idx_pairs, bucket))

    # Emit one combined record per group.
    template_alt: dict[GroupKey, dict[str, Any]] = {}
    for gkey, entries in grouped.items():
        ec, ent, alt, canonical_name = gkey
        # Determine outer-axes signature (uniform across entries within a
        # group — group_axes guarantees it).
        sample_pairs = entries[0][0]
        outer_axes = tuple(axis for axis, _ in sample_pairs)

        # Sort entries deterministically by outer index tuple values.
        # Numeric-aware: a 1-based integer tier axis must order 1,2,…,10,11
        # (numeric), not the lexical 1,10,11,…,2 a plain string sort yields —
        # the latter silently reorders the ladder versus its authored form
        # and breaks DB round-trip byte-parity.
        entries.sort(key=lambda e: tuple(_natural_key(v) for _, v in e[0]))

        # Build the nested Map.  Inner-leaf Map: facet -> value (axis=
        # "price" per the DB encoding hack).
        facet_keys_order = list(entries[0][1]["facet_keys"])
        # Canonical ordering: price, quantity
        if set(facet_keys_order) == {"price", "quantity"}:
            facet_keys_order = ["price", "quantity"]
        else:
            facet_keys_order.sort()

        def _leaf_map(facet_values: dict[str, Any]) -> Map:
            # The innermost Map's axis carries the facet key (price /
            # quantity).  The canonical DB omits ``index_name`` for this
            # level — Spine's encoder treats the facet axis as anonymous
            # because the keys themselves are the schema.  Leaving
            # ``index_name`` unset keeps the JSON byte-identical to the
            # canonical authoring form.
            idxs = facet_keys_order
            vals = [facet_values.get(k) for k in idxs]
            return Map(indexes=idxs, values=vals)

        if len(outer_axes) == 1:
            # Depth-2 Map(tier -> facet_map)
            inner_indexes = [pairs[0][1] for pairs, _ in entries]
            inner_values = [_leaf_map(b["facet_values"]) for _, b in entries]
            top_map = Map(
                indexes=inner_indexes,
                values=inner_values,
                index_name=outer_axes[0],
            )
        elif len(outer_axes) == 2:
            # Depth-3 Map(period -> Map(tier -> facet_map)).  Group by
            # first axis (period).
            from collections import defaultdict as _dd
            by_period: "dict[str, list[tuple[str, dict[str, Any]]]]" = _dd(list)
            for pairs, bucket in entries:
                p_val = pairs[0][1]
                t_val = pairs[1][1]
                by_period[p_val].append((t_val, bucket["facet_values"]))
            period_indexes = list(by_period.keys())  # insertion order
            period_values = []
            for p in period_indexes:
                inner_entries = by_period[p]
                tier_indexes = [t for t, _ in inner_entries]
                tier_values = [_leaf_map(fv) for _, fv in inner_entries]
                period_values.append(Map(
                    indexes=tier_indexes,
                    values=tier_values,
                    index_name=outer_axes[1],
                ))
            top_map = Map(
                indexes=period_indexes,
                values=period_values,
                index_name=outer_axes[0],
            )
        else:
            # Shouldn't happen for registered facet shapes.
            continue

        template = entries[0][1]["template"]
        combined = dict(template)
        combined["param_name"] = canonical_name
        combined["value"] = top_map
        combined["index_value"] = None
        combined["index_name"] = None
        combined.pop("extra_index_values", None)
        combined["data_type"] = "map"
        template_alt[gkey] = combined

    # Final output: original records minus dropped ones, plus combined.
    for i, rec in enumerate(records):
        if i in drop_indices:
            continue
        out.append(rec)
    out.extend(template_alt.values())
    return out


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
        rec = dict(map_templates[key])
        # When the source sheet's data type marks this parameter as an
        # array (e.g. ``"string (array)"`` / ``"boolean (array)"``), the
        # cells are array-expanded rows where the index cell holds the
        # array slot number OR the period token.  Reconstruct an Array
        # so the round-trip preserves the SpineDB ``"array"`` type — a
        # hard requirement for ``model.solves``, ``solve.realized_periods``
        # and friends, which downstream code (engine_polars._solve_config
        # ``params_to_dict``) only treats as ordered sequences when the
        # parsed value is an ``Array`` instance.  Index_name is preserved
        # from the source so the engine's axis labels stay intact.
        dtype = (rec.get("data_type") or "").lower()
        is_array_dtype = "(array)" in dtype or dtype.endswith(" array") or dtype == "array"
        if indexes and all(i == _EMPTY_COLLECTION_SENTINEL for i in indexes):
            # Sentinel row written by the exporter for a zero-length
            # Array/Map (excel_writer.EMPTY_COLLECTION_SENTINEL): rebuild an
            # empty collection so an explicit empty override (e.g.
            # solve.contains_solves = []) round-trips instead of collapsing
            # into "no value" and inheriting a lower-ranked alternative.
            if is_array_dtype:
                value = Array(values=[])
            else:
                value = Map(indexes=[], values=[])
        elif is_array_dtype:
            value = Array(values=values, index_name=index_name)
        else:
            value = Map(indexes=indexes, values=values, index_name=index_name)
        rec["value"] = value
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
    param_values_added: set[tuple[str, tuple[str, ...], str, str]],
) -> None:
    """Write a single SheetData into *db*.

    The caller-owned sets track what has already been added so that
    duplicates are silently skipped.

    ``param_values_added`` keys on
    ``(entity_class, entity_byname, param_name, alternative)`` — the unique
    identity of a Spine parameter_value.  A parameter that can be either
    constant or period-indexed (e.g. ``node.existing``) is split across a
    ``*_c`` constant sheet and a ``*_p`` periodic sheet by the exporter;
    older workbooks emitted the scalar value on BOTH.  The constant sheet is
    written first (pass-1 workbook order), so first-writer-wins keeps the
    correctly-typed float and silently drops the periodic-sheet duplicate
    instead of surfacing an alarming "already a parameter_value" warning.
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

    # -- Combine facet-leaf records (price_ladder_*) ------------------------
    # Reader emits ``price`` / ``quantity`` as separate per-row records;
    # the registry tells us they belong to a canonical Map-valued param
    # (e.g. ``commodity.price_ladder_cumulative``).  Combination must
    # happen BEFORE the generic Map grouper so the resulting top-level
    # Map flows through the scalar branch as a single rec with type=map.
    records = _combine_facet_records(sheet.records, sheet.sheet_name)

    # -- Split records into scalars and maps --------------------------------
    scalars, maps = _group_map_records(records)

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
        if not rec["param_name"]:
            continue  # entity-only record — no param to write
        pv_key = (
            rec["entity_class"], rec["entity_byname"],
            rec["param_name"], rec["alternative"],
        )
        if pv_key in param_values_added:
            continue  # already written by a sibling sheet (constant wins)
        param_values_added.add(pv_key)
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
        pv_key = (
            rec["entity_class"], rec["entity_byname"],
            rec["param_name"], rec["alternative"],
        )
        if pv_key in param_values_added:
            continue  # already written by a sibling sheet
        param_values_added.add(pv_key)
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
                    "Could not add scenario_alternative %s -> %s (rank %d): %s",
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


def _purge_database(db: DatabaseMapping, keep_entities: bool = False) -> None:
    """Remove data from the database before import.

    Args:
        keep_entities: If True, keep entities (useful when importing into a
            fresh template DB where structural entities like upDown should
            be preserved).  Alternatives, scenarios, and parameter values
            are always purged.
    """
    try:
        db.remove_parameter_value(id=Asterisk)
        if not keep_entities:
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
    keep_entities: bool = False,
) -> None:
    """Write parsed SheetData objects into a SpineDB.

    Uses a two-pass strategy so that 1-dimensional entities are created
    before multi-dimensional (relationship) entities that reference them.

    Args:
        sheet_data_list: Output of :func:`read_self_describing_excel`.
        db_url: SQLAlchemy-style database URL, e.g.
            ``sqlite:///path/to/db.sqlite``.
        purge_first: If True, remove existing data from the database
            before writing (keeps entity classes and parameter definitions).
        keep_entities: If True (and purge_first is True), keep existing
            entities during purge.  Useful when importing into a fresh
            template DB that has structural entities (like upDown).
    """
    with DatabaseMapping(db_url, create=False, upgrade=True) as db:
        if purge_first:
            _purge_database(db, keep_entities=keep_entities)

        # Tracking sets to avoid duplicate add calls
        alternatives_added: set[str] = set()
        entities_added: set[tuple[str, tuple[str, ...]]] = set()
        entity_alts_added: set[tuple[str, tuple[str, ...], str]] = set()
        param_values_added: set[tuple[str, tuple[str, ...], str, str]] = set()

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
            _write_sheet(
                sheet, db, alternatives_added, entities_added,
                entity_alts_added, param_values_added,
            )

        # Pass 1.5: ensure 0-dim entities referenced by multi-dim sheets
        _ensure_dimension_entities(data_sheets_nd, db, entities_added)

        # Pass 2: multi-dimensional entity classes
        for sheet in data_sheets_nd:
            logger.info(
                "Pass 2 — writing sheet '%s' (%d records, %d links)",
                sheet.sheet_name, len(sheet.records), len(sheet.link_entities),
            )
            _write_sheet(
                sheet, db, alternatives_added, entities_added,
                entity_alts_added, param_values_added,
            )

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
