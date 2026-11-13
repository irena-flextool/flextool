"""Write parsed old FlexTool data (from read_old_flextool) to a Spine database.

This module takes an OldFlexToolData instance and writes all entities, parameters,
relationships, and structural items (timeline, solve, model) into a SpineDB that
already has the FlexTool entity classes and parameter definitions (from the template).

Usage::

    from flextool.process_inputs.read_old_flextool import read_old_flextool
    from flextool.process_inputs.write_old_flextool_to_db import write_old_flextool_to_db

    data = read_old_flextool("old_model.xlsm")
    write_old_flextool_to_db(data, "sqlite:///flextool_input.sqlite")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import spinedb_api as api
from spinedb_api import Array, DatabaseMapping, Map, SpineDBAPIError
from flextool.update_flextool.db_handle import open_database_mapping
from spinedb_api.exception import NothingToCommit

from flextool.process_inputs.read_old_flextool import (
    DemandTimeSeries,
    OldFlexToolData,
    SensitivityOverride,
    TimeSeriesData,
    UnitInstance,
    UnitTimeSeries,
    UnitType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counters for summary logging
# ---------------------------------------------------------------------------

#: Report topics, in the order the report lists them (by FT2 sheet).
_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Time and costs", ("master",)),
    ("Units", ("unit_type", "units", "ts_unit")),
    ("Nodes and connections", ("gridNode", "nodeGroup", "nodeNode")),
    ("Time series", ("ts_cf", "ts_inflow")),
)


@dataclass
class UnmappedItem:
    """A FlexTool 2 input the importer could not translate (or that works
    differently in FlexTool 4).  ``reason`` is a plain sentence: what it means
    for the model and what to do."""
    sheet: str
    column: str
    entities: list[str]
    reason: str

    def line(self) -> str:
        where = f" ({', '.join(self.entities)})" if self.entities else ""
        return f"'{self.column}' on sheet {self.sheet}{where}: {self.reason}"

    @property
    def topic(self) -> str:
        if self.sheet.startswith("Sensitivity"):
            return "Sensitivities"
        for topic, sheets in _TOPICS:
            if self.sheet in sheets:
                return topic
        return "Other inputs"


class ImportStrictError(RuntimeError):
    """Raised by a ``strict`` import when a parameter value could not be written."""


class _Counters:
    """What was written, what failed and what could not be translated."""

    def __init__(self, strict: bool = False) -> None:
        self.entities: int = 0
        self.relationships: int = 0
        self.parameters: int = 0
        self.entity_alternatives: int = 0
        self.strict = strict
        #: Every ``_add_param`` write the database refused.
        self.failed: list[str] = []
        #: Inputs that have no FlexTool 4 counterpart (or lack their data).
        self.unmapped: list[UnmappedItem] = []
        #: How an input was interpreted where FlexTool 2 and 4 differ.
        self.assumptions: list[str] = []

    def summary(self) -> str:
        return (
            f"entities={self.entities}, relationships={self.relationships}, "
            f"parameters={self.parameters}, entity_alternatives={self.entity_alternatives}"
        )


@dataclass
class ImportReport:
    """End-of-import summary: what was written, failed and not translated."""
    written: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    unmapped: list[UnmappedItem] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    #: Behaviour-change markers the migration wrote and the import removed.
    markers_removed: list[str] = field(default_factory=list)

    def add(self, label: str, counters: _Counters) -> None:
        self.written.append(f"{label}: {counters.summary()}")
        self.failed.extend(counters.failed)
        self.unmapped.extend(counters.unmapped)
        self.assumptions.extend(counters.assumptions)

    def format(self) -> str:
        """The report as plain text for a reader who is not a modeller."""
        lines = ["FlexTool 2 import report", "", "Written to the database:"]
        lines += [f"  {w}" for w in self.written]
        if self.failed:
            lines += ["", f"{len(self.failed)} value(s) could not be written. "
                      "They are missing from the database; the list below "
                      "names them. Please report this, with the workbook, as "
                      "an importer bug."]
            lines += [f"  - {f}" for f in self.failed]
        lines.append("")
        if not self.unmapped and not self.assumptions:
            lines.append("Every FlexTool 2 input in the workbook was carried over.")
            return "\n".join(lines)
        lines.append(
            "These FlexTool 2 inputs were not carried over, or work "
            "differently in FlexTool 4, so the FlexTool 4 model differs from "
            "the FlexTool 2 model where listed. Each line says what it means "
            "and what to do.")
        topics = [t for t, _ in _TOPICS] + ["Sensitivities", "Other inputs"]
        for topic in topics:
            items = [u for u in self.unmapped if u.topic == topic]
            if not items:
                continue
            lines += ["", f"{topic}:"]
            lines += [f"  - {u.line()}" for u in items]
        if self.assumptions:
            lines += ["", "How some inputs were read:"]
            lines += [f"  - {a}" for a in self.assumptions]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _add_entity(
    db: DatabaseMapping,
    class_name: str,
    name: str,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Add a 0-dimensional entity and its entity_alternative."""
    key = (class_name, (name,))
    if key not in entities_added:
        entities_added.add(key)
        try:
            db.add_entity(entity_class_name=class_name, name=name)
            counters.entities += 1
        except SpineDBAPIError as exc:
            logger.debug("Entity %s '%s' may already exist: %s", class_name, name, exc)

    ea_key = (class_name, (name,), alt_name)
    if ea_key not in entity_alts_added:
        entity_alts_added.add(ea_key)
        try:
            db.add_entity_alternative(
                entity_class_name=class_name,
                entity_byname=(name,),
                alternative_name=alt_name,
                active=True,
            )
            counters.entity_alternatives += 1
        except SpineDBAPIError as exc:
            logger.debug(
                "Entity alternative %s '%s' alt='%s' may already exist: %s",
                class_name, name, alt_name, exc,
            )


def _add_relationship(
    db: DatabaseMapping,
    class_name: str,
    elements: tuple[str, ...],
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Add a multi-dimensional entity (relationship) and its entity_alternative."""
    key = (class_name, elements)
    if key not in entities_added:
        entities_added.add(key)
        try:
            db.add_entity(entity_class_name=class_name, entity_byname=elements)
            counters.relationships += 1
        except SpineDBAPIError as exc:
            logger.debug(
                "Relationship %s %s may already exist: %s", class_name, elements, exc,
            )

    ea_key = (class_name, elements, alt_name)
    if ea_key not in entity_alts_added:
        entity_alts_added.add(ea_key)
        try:
            db.add_entity_alternative(
                entity_class_name=class_name,
                entity_byname=elements,
                alternative_name=alt_name,
                active=True,
            )
            counters.entity_alternatives += 1
        except SpineDBAPIError as exc:
            logger.debug(
                "Relationship entity_alt %s %s alt='%s' may already exist: %s",
                class_name, elements, alt_name, exc,
            )


def _add_param(
    db: DatabaseMapping,
    class_name: str,
    entity_byname: tuple[str, ...],
    param_name: str,
    value: Any,
    alt_name: str,
    counters: _Counters,
) -> None:
    """Add a parameter value, converting via api.to_database()."""
    if value is None:
        return
    db_value, db_type = api.to_database(value)
    try:
        db.add_parameter_value(
            entity_class_name=class_name,
            entity_byname=entity_byname,
            parameter_definition_name=param_name,
            alternative_name=alt_name,
            value=db_value,
            type=db_type,
        )
        counters.parameters += 1
    except SpineDBAPIError as exc:
        message = (
            f"{class_name}.{param_name} {'/'.join(entity_byname)} "
            f"[{alt_name}]: {exc}"
        )
        if counters.strict:
            raise ImportStrictError(f"Could not add param {message}") from exc
        logger.warning("Could not add param %s", message)
        counters.failed.append(message)


def _add_param_if_set(
    db: DatabaseMapping,
    class_name: str,
    entity_byname: tuple[str, ...],
    param_name: str,
    value: Any,
    alt_name: str,
    counters: _Counters,
    skip_zero: bool = True,
) -> None:
    """Add a parameter value only if it is not None and (optionally) not zero."""
    if value is None:
        return
    if skip_zero and isinstance(value, (int, float)) and value == 0:
        return
    _add_param(db, class_name, entity_byname, param_name, value, alt_name, counters)


def _make_time_map(
    ts_data: dict[str, float],
    index_name: str = "time",
) -> Map:
    """Convert a {time_id: value} dict to a spinedb_api.Map."""
    indexes = list(ts_data.keys())
    values = list(ts_data.values())
    return Map(indexes, values, index_name=index_name)


def _get_unit_type(data: OldFlexToolData, unit: UnitInstance) -> UnitType | None:
    """Look up the UnitType matching a unit's unit_type field."""
    for ut in data.unit_types:
        if ut.name == unit.unit_type:
            return ut
    return None


def _get_unit_type_param(
    data: OldFlexToolData,
    unit: UnitInstance,
    param_key: str,
) -> float | None:
    """Look up a parameter value from the unit's unit_type template."""
    ut = _get_unit_type(data, unit)
    if ut is None:
        return None
    return ut.params.get(param_key)


def _get_unit_name(unit: UnitInstance) -> str:
    """Generate a unique unit name: {unit_type}_{output_node}."""
    return f"{unit.unit_type}_{unit.output_node}"


def _unit_has_storage(unit: UnitInstance) -> bool:
    """Return True if the unit has existing or investable storage capacity."""
    return (
        (unit.storage_mwh is not None and unit.storage_mwh != 0)
        or (unit.invested_storage_mwh is not None and unit.invested_storage_mwh > 0)
        or (unit.max_invest_mwh is not None and unit.max_invest_mwh > 0)
    )


def _ft2_capacity(capacity: float | None, invested: float | None) -> float | None:
    """FT4 ``existing`` of an FT2 capacity: ``capacity`` + ``invested capacity``.

    FT2 ``invested capacity (MW)`` / ``invested storage (MWh)`` is a parameter
    (``flexModel.mod``: C = capacity + invested capacity + v_invest), capacity
    already built, not a decision: it is in every capacity limit in every mode,
    and outside ``v_invest`` (the kW/kWh ratio and the unitGroup invest limits
    count v_invest only).  So it is FT4 existing capacity, not a forced
    investment (which a dispatch-only solve, or an entity without an
    investment cost, would never make).  ``None`` when both are empty.
    """
    if capacity is None and invested is None:
        return None
    return (capacity or 0.0) + (invested or 0.0)


def _storage_max_invest(
    max_mw: float | None, max_mwh: float | None, ratio: float | None,
) -> tuple[float, float]:
    """(MW, MWh) investment limits of an FT2 storage unit.

    ``flexModel.mod`` ``p_unit_max_invest_MW`` / ``p_unit_max_invest_MWh``:
    each is its own ``max invest`` when given, else the other one through the
    ``fixed kW/kWh ratio`` (MW = ratio x MWh), else 0.  (The unitGroup
    fallback is not per unit.)
    """
    mw = max_mw if max_mw is not None and max_mw > 0 else 0.0
    mwh = max_mwh if max_mwh is not None and max_mwh > 0 else 0.0
    has_ratio = ratio is not None and ratio > 0
    out_mw = mw if mw > 0 else (mwh * ratio if has_ratio and mwh > 0 else 0.0)
    out_mwh = mwh if mwh > 0 else (mw / ratio if has_ratio and mw > 0 else 0.0)
    return out_mw, out_mwh


def _get_unit_node_name(unit: UnitInstance) -> str:
    """Generate the input node name for a non-charger/discharger unit.

    Uses '_storage' if the unit has storage capacity or investment,
    '_node' otherwise (pure inflow / run-of-river).
    """
    suffix = "_storage" if _unit_has_storage(unit) else "_node"
    return f"{_get_unit_name(unit)}{suffix}"


def _get_master_param(data: OldFlexToolData, key: str, default: float = 0.0) -> float:
    """Get a master parameter with a default fallback.

    Tries the key as-is first, then with underscores replaced by spaces
    and vice versa, to handle naming variations between files.
    """
    if key in data.master.params:
        return data.master.params[key]
    alt_key = key.replace(" ", "_")
    if alt_key in data.master.params:
        return data.master.params[alt_key]
    alt_key = key.replace("_", " ")
    if alt_key in data.master.params:
        return data.master.params[alt_key]
    return default


def _find_demand_ts(
    data: OldFlexToolData, grid: str, node: str,
) -> DemandTimeSeries | None:
    """Find the demand time series for a given grid/node pair."""
    for dts in data.demand_ts:
        if dts.grid == grid and dts.node == node:
            return dts
    return None


def _find_inflow_profile(
    data: OldFlexToolData, profile_name: str,
) -> TimeSeriesData | None:
    """Find an inflow profile by name."""
    for ip in data.inflow_profiles:
        if ip.name == profile_name:
            return ip
    return None


def _find_cf_profile(
    data: OldFlexToolData, profile_name: str,
) -> TimeSeriesData | None:
    """Find a capacity factor profile by name."""
    for cp in data.cf_profiles:
        if cp.name == profile_name:
            return cp
    return None


def _find_unit_ts(
    data: OldFlexToolData,
    unit: UnitInstance,
    param_name: str,
) -> UnitTimeSeries | None:
    """Find a unit time series by grid/node/unit/param match."""
    for uts in data.unit_ts:
        if (
            uts.grid == unit.output_grid
            and uts.node == unit.output_node
            and uts.unit == unit.unit_type
            and uts.param_name == param_name
        ):
            return uts
    return None


def _fuel_market_name(fuel_name: str) -> str:
    """Return the market node name for a fuel."""
    return f"{fuel_name}_market"


#: FT2 ``units`` flag column → (``ts_unit`` parameter, FT4 ``profile_method``).
#: FT2 ``flexModel.mod`` (``fix_unit_generation``, ``min_generation_ts``,
#: ``upwardLimitNotOnline``) bounds the generation by the ts value times the
#: unit's capacity (existing + invested), and the IRENA FlexTool methodology
#: guide says so too: "fix_generation, min generation, max_generation (these
#: fix or limit the generation. Use values from 0-1 as a share of max
#: generation)".  An FT4 profile on ``unit__node__profile`` is the same bound:
#: profile × capacity on the flow to the node.
_GENERATION_LIMITS: tuple[tuple[str, str, str], ...] = (
    ("fix unit generation", "fix_generation", "fixed"),
    ("use min generation", "min_generation", "lower_limit"),
    ("use max generation", "max_generation", "upper_limit"),
)


def _unit_flag(unit: UnitInstance, column: str) -> float | None:
    """Value of a ``units`` column kept in ``extra_params`` (header case-insensitive)."""
    for header, value in unit.extra_params.items():
        if _norm_key(header) == column:
            return value
    return None


def _norm_key(key: str) -> str:
    """FT2 header normalisation: lower case, underscores as spaces."""
    return " ".join(str(key).strip().lower().replace("_", " ").split())


def _write_generation_limits(
    data: OldFlexToolData,
    db: DatabaseMapping,
    unit: UnitInstance,
    process: str,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """FT2 fix / min / max generation → a profile on the primary output.

    A flag of 1 in ``units`` uses the ``ts_unit`` series of the same
    grid/node/unit; the profile ``<process>_<param>`` gets that series and
    ``unit__node__profile(process, output node, profile)`` gets the method
    (see :data:`_GENERATION_LIMITS`).  Only the primary output: an output2
    follows through the CHP eq / gt / lt constraints.  A flag of 0 ignores
    the series, as FT2 does; a flag of 1 without a series is reported.
    """
    output_node = unit.output_node
    for column, ts_param, method in _GENERATION_LIMITS:
        if not _unit_flag(unit, column):
            continue
        if unit.input_grid and unit.input_node:
            # FT2 bounds v_convert, the INPUT flow of a conversion unit, and
            # its capacity is on the input side; fix_unit_generation is not
            # defined for conversion units at all.
            counters.unmapped.append(UnmappedItem(
                "units", column, [process], _CONVERSION_UNIT_LIMIT[column],
            ))
            continue
        if method == "upper_limit" and unit.cf_profile:
            counters.unmapped.append(UnmappedItem(
                "units", column, [process],
                "FlexTool 2 ignores the maximum generation of a unit with a "
                "capacity factor profile (the profile is its upper limit), so "
                "the import ignores it too. Nothing to do.",
            ))
            continue
        uts = _find_unit_ts(data, unit, ts_param)
        if uts is None or not uts.data:
            counters.unmapped.append(UnmappedItem(
                "units", column, [process],
                f"the column is 1, but sheet ts_unit has no '{ts_param}' "
                f"series for {unit.output_grid}/{output_node}/"
                f"{unit.unit_type}, so the unit got no limit. Add the series "
                "to the workbook and import again, or give the FlexTool 4 "
                "unit a profile by hand.",
            ))
            continue
        profile = f"{process}_{ts_param}"
        _add_entity(db, "profile", profile, alt_name, counters,
                    entities_added, entity_alts_added)
        _add_param(db, "profile", (profile,), "profile",
                   _make_time_map(uts.data), alt_name, counters)
        link = (process, output_node, profile)
        _add_relationship(db, "unit__node__profile", link, alt_name, counters,
                          entities_added, entity_alts_added)
        _add_param(db, "unit__node__profile", link, "profile_method", method,
                   alt_name, counters)
        if method == "upper_limit" and _unit_has_online(data, unit):
            counters.assumptions.append(
                f"{process}: FlexTool 2 also limits the online capacity by "
                "'max_generation'; FlexTool 4 limits only the generation. The "
                "minimum load, start-ups and reserves of the unit can differ "
                "a little where the series is below 1."
            )


def _unit_has_online(data: OldFlexToolData, unit: UnitInstance) -> bool:
    """FT2 ``unit_online`` (``flexModel.mod``): master ``use_online`` and a
    min load, min up / down time or ``use min online`` on the unit (dispatch
    runs only, which is where FT2 has online variables)."""
    if _get_master_param(data, "use online") != 1:
        return False
    return any(
        (_get_unit_type_param(data, unit, key) or 0) > 0
        for key in ("min load", "min uptime (h)", "min downtime (h)")
    ) or bool(_unit_flag(unit, "use min online"))


#: The generation-limit columns on a conversion unit (input from a node):
#: FT2 bounds v_convert, the INPUT flow, there, and its capacity is on the
#: input side; ``fix_unit_generation`` is not defined for conversion units.
_CONVERSION_UNIT_LIMIT: dict[str, str] = {
    "fix unit generation": (
        "FlexTool 2 has no fixed generation for a unit that takes its input "
        "from a node (a conversion unit such as a heat pump), so the column "
        "had no effect there and the import ignores it. Nothing to do."
    ),
    "use min generation": (
        "on a unit that takes its input from a node, FlexTool 2 sets the "
        "minimum on the input flow, not on the output. Not carried over: to "
        "keep it, give the FlexTool 4 unit a lower_limit profile on its input "
        "node by hand."
    ),
    "use max generation": (
        "on a unit that takes its input from a node, FlexTool 2 sets the "
        "maximum on the input flow, not on the output. Not carried over: to "
        "keep it, give the FlexTool 4 unit an upper_limit profile on its "
        "input node by hand."
    ),
}


# ---------------------------------------------------------------------------
# Inputs the importer does not translate (reported, never silently dropped)
# ---------------------------------------------------------------------------

#: master parameters the importer reads (normalised names).
_MASTER_MAPPED = frozenset({
    "co2 cost", "loss of load penalty", "loss of reserves penalty",
    "lack of inertia penalty", "lack of capacity penalty",
    "time period duration", "use capacity margin", "use online", "use ramps",
    "use non synchronous", "use inertia limit", "mode invest", "mode dispatch",
})

#: unit_type parameters the importer reads, or need not read (annuity: FT4
#: computes it from interest and lifetime; :func:`_annuity_mismatch_items`
#: reports an FT2 annuity that is not that value).
_UNIT_TYPE_MAPPED = frozenset({
    "efficiency", "min load", "eff at min load", "ramp up (p.u. per min)",
    "ramp down (p.u. per min)", "o&m cost/mwh", "availability", "max reserve",
    "inertia constant (mws/mw)", "fixed cost/kw/year", "inv.cost/kw",
    "inv.cost/kwh", "fixed kw/kwh ratio", "startup cost", "min uptime (h)",
    "min downtime (h)", "eff charge", "self discharge loss", "lifetime",
    "interest", "annuity", "non synchronous",
})

#: ``units`` columns kept in ``extra_params`` that the importer reads.
_UNITS_EXTRA_MAPPED = frozenset(column for column, _, _ in _GENERATION_LIMITS)

#: ``ts_unit`` parameters the importer reads.
_TS_UNIT_MAPPED = frozenset({"efficiency"} | {p for _, p, _ in _GENERATION_LIMITS})

#: Known FT2 inputs without an FT4 counterpart the v56 importer can write.
#: (sheet, normalised column) → plain-language reason with what to do.
_UNMAPPED: dict[tuple[str, str], str] = {
    ("master", "curtailment penalty"): (
        "FlexTool 2 charges this price for each MWh of wind, solar or inflow "
        "energy that is curtailed. FlexTool 4 has no curtailment cost, so "
        "curtailment is free there. Nothing to set; expect more curtailment "
        "where FlexTool 2 avoided it because of the cost."
    ),
    ("master", "reserve duration"): (
        "FlexTool 2 lets a storage hold reserve only while it has energy for "
        "this many hours. The import cannot write this. To keep it, set "
        "reserve_duration on the reserve__upDown__group entity of each "
        "reserve in FlexTool 4 by hand."
    ),
    ("unit_type", "conversion eff"): (
        "FlexTool 2 uses it as the output / input ratio of a unit that takes "
        "its input from a node (for example the COP of a heat pump). Not "
        "carried over: set the FlexTool 4 unit's efficiency to this value by "
        "hand (where the unit has no efficiency time series)."
    ),
    ("unit_type", "availability"): (
        "FlexTool 2 uses availability only in the capacity margin: a unit "
        "counts there with availability x capacity. It does not reduce the "
        "unit's generation. FlexTool 4 has no input for a unit's share in the "
        "capacity margin, so the import leaves availability out: generation "
        "is as in FlexTool 2, but in the capacity margin these units count "
        "with their full capacity. This matters only when the capacity margin "
        "is used; then raise the group's capacity_margin by the capacity the "
        "availability took away, or lower the units' capacity by hand."
    ),
    ("units", "use min online"): (
        "FlexTool 2 keeps at least the ts_unit 'min_online' share of the "
        "capacity online. FlexTool 4 has no time series for the minimum "
        "online capacity, so the unit can go offline there. To keep a "
        "minimum, give the unit a lower_limit profile on its output by hand "
        "(it limits the generation, not the online capacity)."
    ),
    ("units", "fueluse increase eq x output2"): (
        "FlexTool 2's model (the fuel use equation in flexModel.mod) adds "
        "this fuel only when output2 is on the unit's own output node, so in "
        "this workbook it adds no fuel in FlexTool 2 either, and the import "
        "ignores it. Nothing to do, unless your FlexTool 2 version did add "
        "the fuel: then compare the fuel use of the unit."
    ),
    ("gridNode", "import (mwh)"): (
        "FlexTool 2 adds the ts_import series, scaled to this annual energy, "
        "to the node. Not carried over, so the node gets no import. To keep "
        "it, add the import as a positive inflow of the node by hand (or as "
        "a unit with a fixed profile)."
    ),
}

_GENERIC_REASON = (
    "the importer does not read this column, so it is not in the FlexTool 4 "
    "model. Check whether it matters for your study; if it does, set the "
    "FlexTool 4 counterpart by hand."
)

#: Relative difference above which two time scalings count as different.
_SCALING_TOLERANCE = 0.01


def _is_cosmetic(column: str) -> bool:
    """Output-only FT2 columns (printing, colours): nothing to translate."""
    return column.startswith("print") or column.startswith("color")


def _nonzero(value: Any) -> bool:
    return value is not None and value != 0 and value != ""


def _time_in_years_item(data: OldFlexToolData) -> UnmappedItem | None:
    """``time_in_years`` that differs from the hours the time series covers.

    FT2 (``flexModel.mod``): ``time_in_use_in_years = |time_in_use| / |time|
    × time_in_years`` and ``scale_to_annual = (time_period_duration / 60) /
    time_in_use_in_years`` multiplies the operating costs (fuel, CO2, O&M,
    start-ups and every penalty) in the objective; investment annuities and
    fixed costs are annual and not scaled.  FT4 scales the operating costs of
    a period by ``1 / period_share`` with ``period_share = Σ step_duration /
    8760``.  Per MW in a step in use the two weights are
    ``dur·|time| / (|time_in_use|·time_in_years)`` (FT2) and
    ``8760 / |time_in_use|`` (FT4), so they agree exactly when
    ``time_in_years = |time| × dur / 8760``.  FT4 derives ``period_share``
    from the step durations only (``timeset_weights`` and representative
    period weights are normalised), so it has no input for this: report it.
    """
    tiy = _get_master_param(data, "time in years", default=1.0)
    n_steps = len(data.time_steps)
    if not n_steps or not tiy or tiy <= 0:
        return None
    duration = _get_master_param(data, "time period duration")
    duration_h = duration / 60.0 if duration > 0 else 1.0
    covered = n_steps * duration_h / 8760.0
    factor = tiy / covered
    if abs(factor - 1.0) <= _SCALING_TOLERANCE:
        return None
    heavier = "more" if factor > 1 else "less"
    return UnmappedItem(
        "master", f"time_in_years = {tiy:g}", [],
        f"FlexTool 2 counts the {n_steps} time steps of {duration_h:g} h as "
        f"{tiy:g} year(s), but they cover {covered:.4g} year(s). FlexTool 4 "
        "always uses the hours the time steps cover and scales their "
        "operating costs (fuel, CO2, O&M, start-ups, penalties) to a full "
        f"year. So FlexTool 4 weighs the operating costs {factor:.4g} times "
        f"as much as FlexTool 2 did ({heavier} than FlexTool 2), compared with "
        "the investment and fixed costs, and it will choose other "
        "investments. FlexTool 4 has no input that changes this. To get "
        "FlexTool 2's balance, multiply the investment and fixed costs "
        "(invest_cost and fixed_cost of the units, connections and storage "
        f"nodes) by {factor:.4g}; the reported investment and fixed costs are "
        f"then {factor:.4g} times FlexTool 2's.",
    )


def _profile_scaling_items(data: OldFlexToolData) -> list[UnmappedItem]:
    """cf and inflow profiles that FT2 rescales over the time steps in use.

    FT2 (``flexModel.mod`` ``p_scaleCf`` / ``p_scaleInflow``) multiplies a
    capacity factor or inflow series by ``mean over the whole series / mean
    over the steps in use``, so a sample of the year keeps the annual mean.
    FT4 uses the values as they are.  The steps in use are those of the run:
    ``in_use_invest`` in an investment run, ``in_use`` in a dispatch run.
    """
    steps = data.time_steps
    runs: list[tuple[str, set[str]]] = []
    if _get_master_param(data, "mode invest") == 1:
        runs.append(("investment", {t.time_id for t in steps if t.in_use_invest}))
    if _get_master_param(data, "mode dispatch") == 1 or not runs:
        runs.append(("dispatch", {t.time_id for t in steps if t.in_use}))
    items: list[UnmappedItem] = []
    for sheet, kind, profiles in (("ts_cf", "capacity factor", data.cf_profiles),
                                  ("ts_inflow", "inflow", data.inflow_profiles)):
        for profile in profiles:
            values = list(profile.data.values())
            if not values or not sum(values):
                continue
            mean_all = sum(values) / len(values)
            factors: dict[str, str] = {}
            for run, in_use in runs:
                used = [v for t, v in profile.data.items() if t in in_use]
                if not used or len(used) == len(values) or not sum(used):
                    continue
                factor = mean_all / (sum(used) / len(used))
                if abs(factor - 1.0) > _SCALING_TOLERANCE:
                    factors[run] = f"{factor:.4g}"
            if factors:
                if len(factors) > 1 and len(set(factors.values())) == 1:
                    by = f"{next(iter(factors.values()))} in both runs"
                else:
                    by = " and ".join(f"{f} in the {run} run"
                                      for run, f in factors.items())
                items.append(UnmappedItem(
                    sheet, profile.name, [],
                    f"FlexTool 2 multiplies this {kind} series by "
                    f"{by}, so that its mean over the time "
                    "steps in use equals its mean over the whole series. "
                    "FlexTool 4 uses the values as they are, so this source "
                    "gives a different amount of energy. To match FlexTool 2, "
                    "multiply the FlexTool 4 profile (or inflow) by that "
                    "factor.",
                ))
    return items


def _availability_item(data: OldFlexToolData) -> UnmappedItem | None:
    """Units whose FT2 ``availability`` < 1 (FT2: capacity margin only)."""
    names = []
    for unit in data.units:
        if unit.cf_profile:
            continue  # FT2 counts a cf unit by its generation, not availability
        avail = _get_unit_type_param(data, unit, "availability")
        if avail is not None and avail < 1.0:
            names.append(f"{_get_unit_name(unit)} {avail:g}")
    if not names:
        return None
    return UnmappedItem("unit_type", "availability", names,
                        _UNMAPPED[("unit_type", "availability")])


_INVESTED_CAPACITY_REASON = (
    "imported as existing capacity (added to 'capacity'), so it is in "
    "every solve, as in FlexTool 2. FlexTool 2 also adds its investment "
    "annuity to the total cost; FlexTool 4 does not, so the FlexTool 4 "
    "total cost is lower by that fixed amount. The dispatch and the other "
    "investments are the same. Nothing to do, unless you compare total "
    "costs."
)


def ft2_annuity(interest: float, lifetime: float) -> float:
    """FT2's annuity factor, ``-PMT(interest, lifetime, 1)`` in its workbook: ``r / (1 - (1 + r)^-n)``, and
    ``1 / n`` at an interest of 0 (Excel's PMT at rate 0).  The same as FlexTool 4's annuity of ``discount_rate``
    and ``lifetime``, so the importer maps ``interest`` to ``discount_rate`` (0 included) and drops the column."""
    if abs(interest) < 1e-9:
        return 1.0 / lifetime
    return interest / (1.0 - (1.0 + interest) ** -lifetime)


_ANNUITY_MISMATCH_REASON = (
    "The annuity in the workbook is not the one its interest and lifetime "
    "give, so it was typed over the formula. FlexTool 4 computes the annuity "
    "from 'interest' (as discount_rate) and 'lifetime' and ignores this "
    "column, so the investment costs differ from FlexTool 2. Set the interest "
    "or the lifetime to give the annuity you meant."
)


def _annuity_mismatch_items(data: OldFlexToolData) -> list[UnmappedItem]:
    """An FT2 ``annuity`` that differs from :func:`ft2_annuity` of its ``interest`` and ``lifetime`` (a value typed
    over the workbook formula): FT4 uses only interest and lifetime.  A blank annuity, interest or lifetime is not
    checked."""
    def mismatch(annuity: Any, interest: Any, lifetime: Any) -> bool:
        if not all(isinstance(v, (int, float)) for v in (annuity, interest, lifetime)) or lifetime <= 0:
            return False
        expected = ft2_annuity(float(interest), float(lifetime))
        return abs(float(annuity) - expected) > 1e-6 * max(1.0, abs(expected))

    items = []
    types = [ut.name for ut in data.unit_types
             if mismatch(ut.params.get("annuity"), ut.params.get("interest"), ut.params.get("lifetime"))]
    if types:
        items.append(UnmappedItem("unit_type", "annuity", types, _ANNUITY_MISMATCH_REASON))
    conns = [f"{c.node1}_{c.node2}" for c in data.connections if mismatch(c.annuity, c.interest, c.lifetime)]
    if conns:
        items.append(UnmappedItem("nodeNode", "annuity", conns, _ANNUITY_MISMATCH_REASON))
    return items


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _invested_capacity_items(data: OldFlexToolData) -> list[UnmappedItem]:
    """FT2 ``invested capacity (MW)`` / ``invested storage (MWh)``: imported as
    existing capacity (:func:`_ft2_capacity`); FT2 adds its annuity to the
    objective as a constant, FT4 does not.  Only where that annuity is not 0
    (an investment cost is given)."""
    items = []
    units_mw = [_get_unit_name(u) for u in data.units
                if _positive(u.invested_capacity_mw)
                and _positive(_get_unit_type_param(data, u, "inv.cost/kw"))]
    units_mwh = [_get_unit_name(u) for u in data.units
                 if _positive(u.invested_storage_mwh)
                 and _positive(_get_unit_type_param(data, u, "inv.cost/kwh"))]
    conns = [f"{c.node1}_{c.node2}" for c in data.connections
             if _positive(c.invested_capacity_mw) and _positive(c.invest_cost_per_kw)]
    for sheet, column, names in (("units", "invested capacity (MW)", units_mw),
                                 ("units", "invested storage (MWh)", units_mwh),
                                 ("nodeNode", "invested capacity (MW)", conns)):
        if names:
            items.append(UnmappedItem(sheet, column, names, _INVESTED_CAPACITY_REASON))
    return items


def _sensitivity_invested_capacity_items(
    sensitivities: dict[str, list[SensitivityOverride]], data: OldFlexToolData,
) -> list[UnmappedItem]:
    """:func:`_invested_capacity_items` for the invested capacity a
    sensitivity gives (the base import reports its own): an override > 0 of
    an entity with an investment cost (the scenario's unit_type / nodeNode
    cost override, else the base cost)."""
    found: dict[tuple[str, str], list[str]] = {}
    for scenario, overrides in sensitivities.items():
        for ov in overrides:
            param = ov.param_name.lower()
            if not _positive(ov.value) or "invested" not in param:
                continue
            if ov.section == "units":
                unit_type = ov.entity_ids.get("unittype")
                unit = next((u for u in data.units if u.unit_type == unit_type
                             and u.output_node == ov.entity_ids.get("output node")), None)
                if unit is None:
                    continue
                storage = "invested storage" in param
                cost_column = "inv.cost/kwh" if storage else "inv.cost/kw"
                cost = _get_unit_type_param(data, unit, cost_column)
                for other in overrides:
                    if (other.section == "unit_type"
                            and other.entity_ids.get("unit type") == unit_type
                            and _norm_key(other.param_name) == cost_column):
                        cost = other.value
                key = ("units", "invested storage (MWh)" if storage
                       else "invested capacity (MW)")
                name = _get_unit_name(unit)
            elif ov.section == "nodeNode" and "invested capacity" in param:
                node1, node2 = ov.entity_ids.get("node1"), ov.entity_ids.get("node2")
                conn = next((c for c in data.connections
                             if c.node1 == node1 and c.node2 == node2), None)
                cost = None if conn is None else conn.invest_cost_per_kw
                for other in overrides:
                    if (other.section == "nodeNode"
                            and other.entity_ids.get("node1") == node1
                            and other.entity_ids.get("node2") == node2
                            and ("inv.cost/kw" in other.param_name.lower()
                                 or "inv cost" in other.param_name.lower())):
                        cost = other.value
                key = ("nodeNode", "invested capacity (MW)")
                name = f"{node1}_{node2}"
            else:
                continue
            if _positive(cost):
                found.setdefault(key, []).append(f"{name} [{scenario}]")
    return [UnmappedItem(f"Sensitivity definitions / {sheet}", column, names,
                         _INVESTED_CAPACITY_REASON)
            for (sheet, column), names in found.items()]


def _collect_unmapped(data: OldFlexToolData) -> list[UnmappedItem]:
    """Every non-empty FT2 input the importer does not translate, by entity."""
    found: dict[tuple[str, str], UnmappedItem] = {}

    def note(sheet: str, column: str, entity: str | None,
             label: str | None = None) -> None:
        item = found.get((sheet, column))
        if item is None:
            item = found[(sheet, column)] = UnmappedItem(
                sheet, label or column, [],
                _UNMAPPED.get((sheet, column), _GENERIC_REASON))
        if entity is not None and entity not in item.entities:
            item.entities.append(entity)

    for key, value in data.master.params.items():
        column = _norm_key(key)
        if column in _MASTER_MAPPED or column == "time in years" or _is_cosmetic(column):
            continue
        if _nonzero(value):
            note("master", column, None)

    for ut in data.unit_types:
        for key, value in ut.params.items():
            column = _norm_key(key)
            if column in _UNIT_TYPE_MAPPED or _is_cosmetic(column):
                continue
            if _nonzero(value):
                note("unit_type", column, ut.name)

    for unit in data.units:
        name = _get_unit_name(unit)
        for key, value in unit.extra_params.items():
            column = _norm_key(key)
            if column in _UNITS_EXTRA_MAPPED or _is_cosmetic(column):
                continue
            if _nonzero(value):
                note("units", column, name)
        if _nonzero(unit.fueluse_increase_eq):
            note("units", "fueluse increase eq x output2", name)
        if _needs_charger_discharger(unit, data):
            for column, _, _ in _GENERATION_LIMITS:
                if _unit_flag(unit, column):
                    found.setdefault(("units", column + " (storage)"), UnmappedItem(
                        "units", column, [],
                        "the import does not carry the generation limits of a "
                        "storage unit (it becomes a charger and a discharger "
                        "in FlexTool 4). To keep the limit, give the "
                        "discharger a profile on its output node by hand.",
                    )).entities.append(name)

    for uts in data.unit_ts:
        param = _norm_key(uts.param_name).replace(" ", "_")
        # min_online: FT2 reads it only with the unit's 'use min online' flag,
        # and that flag is reported above (its reason covers the series).
        if param in _TS_UNIT_MAPPED or param == "min_online":
            continue
        if any(_nonzero(v) for v in uts.data.values()):
            note("ts_unit", uts.param_name,
                 f"{uts.grid}/{uts.node}/{uts.unit}")

    for gn in data.grid_nodes:
        if _nonzero(gn.import_mwh):
            note("gridNode", "import (mwh)", gn.node, label="import (MWh)")

    items = list(found.values())
    availability = _availability_item(data)
    if availability is not None:
        items.append(availability)
    items.extend(_profile_scaling_items(data))
    items.extend(_invested_capacity_items(data))
    items.extend(_annuity_mismatch_items(data))
    # time_in_years first: it changes the balance of every cost.
    tiy = _time_in_years_item(data)
    return ([tiy] if tiy is not None else []) + items


def _write_storage_start_end(
    db: DatabaseMapping,
    node: str,
    unit: UnitInstance,
    alt_name: str,
    counters: _Counters,
) -> None:
    """Storage start / finish of an FT2 unit → the FT4 storage node's methods.

    FlexTool 2 (``flexModel.mod``: ``stateStart``, ``stateEnd``,
    ``storageBalance``): a non-zero ``storage start`` / ``storage finish``
    fixes the state at the first / last step to that share of the storage
    capacity; 0 or empty fixes nothing.  The storage balance wraps from the
    last step to the first, except when both are set.  So:

    * both set → ``fix_start_end``, no wrap (``bind_forward_only``);
    * one set → ``fix_start`` / ``fix_end``, wrap (``bind_within_solve``);
    * none → ``fix_nothing``, wrap (``bind_within_solve``).

    The method is always written: the v56 default (``fix_start``) became
    ``fix_nothing`` at v77, so leaving it to the default would change what
    the FT2 input means after migration.
    """
    start = unit.storage_start if unit.storage_start else None
    finish = unit.storage_finish if unit.storage_finish else None
    if start is not None and finish is not None:
        method, binding = "fix_start_end", "bind_forward_only"
    elif start is not None:
        method, binding = "fix_start", "bind_within_solve"
    elif finish is not None:
        method, binding = "fix_end", "bind_within_solve"
    else:
        method, binding = "fix_nothing", "bind_within_solve"
    _add_param(db, "node", (node,), "storage_binding_method", binding,
               alt_name, counters)
    _add_param(db, "node", (node,), "storage_start_end_method", method,
               alt_name, counters)
    _add_param(db, "node", (node,), "storage_state_start", start,
               alt_name, counters)
    _add_param(db, "node", (node,), "storage_state_end", finish,
               alt_name, counters)


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------


def _write_commodities(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write commodity entities and fuel market nodes (Section 1)."""
    # Determine which fuels are actually used by units
    used_fuels: set[str] = set()
    for unit in data.units:
        if unit.fuel:
            used_fuels.add(unit.fuel)

    for fuel in data.fuels:
        if fuel.name not in used_fuels:
            continue

        # Create commodity entity
        _add_entity(db, "commodity", fuel.name, alt_name, counters,
                     entities_added, entity_alts_added)

        # Create fuel market node. It is a commodity source, so node_type
        # MUST be set to "commodity" — an unset node_type defaults to
        # "balance" in the engine (_mid_sets.py "default 'balance'"),
        # which would give the market no supply and starve the fuel.
        market_name = _fuel_market_name(fuel.name)
        _add_entity(db, "node", market_name, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (market_name,), "node_type", "commodity",
                   alt_name, counters)

        # Create commodity__node relationship
        _add_relationship(db, "commodity__node", (fuel.name, market_name),
                          alt_name, counters, entities_added, entity_alts_added)

        # Set commodity parameters
        _add_param(db, "commodity", (fuel.name,), "price", fuel.price_per_mwh,
                   alt_name, counters)
        _add_param_if_set(db, "commodity", (fuel.name,), "co2_content",
                          fuel.co2_content, alt_name, counters)

    logger.info("Wrote %d commodity/fuel-market entities.", len(used_fuels))


def _write_balance_nodes(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write balance nodes from grid_nodes (Section 2)."""
    lolp = _get_master_param(data, "loss of load penalty")

    for gn in data.grid_nodes:
        _add_entity(db, "node", gn.node, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (gn.node,), "node_type", "balance",
                   alt_name, counters)

        if lolp > 0:
            _add_param(db, "node", (gn.node,), "penalty_up", lolp,
                       alt_name, counters)
            _add_param(db, "node", (gn.node,), "penalty_down", lolp,
                       alt_name, counters)

        # Demand as negative inflow
        demand_ts = _find_demand_ts(data, gn.grid, gn.node)
        if demand_ts and demand_ts.data:
            neg_data = {k: -v for k, v in demand_ts.data.items()}
            inflow_map = _make_time_map(neg_data)
            _add_param(db, "node", (gn.node,), "inflow", inflow_map,
                       alt_name, counters)

        if gn.demand_mwh is not None and gn.demand_mwh != 0:
            _add_param(db, "node", (gn.node,), "inflow_method",
                       "scale_to_annual_flow", alt_name, counters)
            _add_param(db, "node", (gn.node,), "annual_flow",
                       gn.demand_mwh, alt_name, counters)

    logger.info("Wrote %d balance nodes.", len(data.grid_nodes))


def _has_eff_charge(unit: UnitInstance, data: OldFlexToolData) -> bool:
    """Return True if the unit's unit_type has an eff_charge parameter."""
    ut = _get_unit_type(data, unit)
    if ut is None:
        return False
    ec = ut.params.get("eff charge")
    return ec is not None and ec > 0


def _needs_charger_discharger(unit: UnitInstance, data: OldFlexToolData) -> bool:
    """Return True if the unit should be split into charger + discharger.

    Based on the existence of ``eff_charge`` in the unit_type, which indicates
    the unit has a charging/storage dimension (battery, pumped hydro, etc.).
    """
    return _has_eff_charge(unit, data)


def _write_storage_units(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write storage nodes with charger/discharger units (Section 3).

    For pure storage units (no fuel, no cf_profile, no inflow), creates:
    - A storage node with node_type='storage'
    - A charger unit: output_node → storage_node
    - A discharger unit: storage_node → output_node
    - Investment constraint linking charger and discharger capacities
    - Optional kW/kWh constraint linking discharger and storage node capacities
    """
    use_online = _get_master_param(data, "use online") == 1
    use_ramps = _get_master_param(data, "use ramps") == 1
    lolp = _get_master_param(data, "loss of load penalty")
    count = 0

    for unit in data.units:
        if not _needs_charger_discharger(unit, data):
            continue

        unit_name = _get_unit_name(unit)
        _get_unit_type(data, unit)
        storage_node = f"{unit_name}_storage"
        charger_name = f"{unit_name}_charger"
        discharger_name = f"{unit_name}_discharger"
        output_node = unit.output_node

        if output_node is None:
            logger.warning("Unit '%s' has no output node; skipping storage.", unit_name)
            continue

        capacity = _ft2_capacity(unit.capacity_mw, unit.invested_capacity_mw) or 0.0

        # ── Storage node ──────────────────────────────────────────
        _add_entity(db, "node", storage_node, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (storage_node,), "node_type", "storage",
                   alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "existing",
                          _ft2_capacity(unit.storage_mwh, unit.invested_storage_mwh),
                          alt_name, counters, skip_zero=False)
        _write_storage_start_end(db, storage_node, unit, alt_name, counters)

        if lolp > 0:
            _add_param(db, "node", (storage_node,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (storage_node,), "penalty_down", 0.0,
                   alt_name, counters)

        self_discharge = _get_unit_type_param(data, unit, "self discharge loss")
        _add_param_if_set(db, "node", (storage_node,), "self_discharge_loss",
                          self_discharge, alt_name, counters)

        # Storage node investment (inv.cost/kWh)
        inv_cost_kwh = _get_unit_type_param(data, unit, "inv.cost/kwh")
        _add_param_if_set(db, "node", (storage_node,), "invest_cost",
                          inv_cost_kwh, alt_name, counters)
        # invest_method / invest_max_total of the storage node are written
        # below with the unit's investment ('invested storage (MWh)' is in
        # existing, see _ft2_capacity).

        # Inflow on storage node (pumped hydro pattern)
        if unit.inflow_profile:
            inflow_ts = _find_inflow_profile(data, unit.inflow_profile)
            if inflow_ts and inflow_ts.data:
                ts_data = inflow_ts.data
                if unit.inflow_multiplier is not None and unit.inflow_multiplier != 0:
                    ts_data = {k: v * unit.inflow_multiplier for k, v in ts_data.items()}
                inflow_map = _make_time_map(ts_data)
                _add_param(db, "node", (storage_node,), "inflow",
                           inflow_map, alt_name, counters)
                _add_param(db, "node", (storage_node,), "inflow_method",
                           "use_original", alt_name, counters)

        # ── Charger unit ──────────────────────────────────────────
        _add_entity(db, "unit", charger_name, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "unit", (charger_name,), "conversion_method",
                   "constant_efficiency", alt_name, counters)
        _add_param(db, "unit", (charger_name,), "existing", capacity,
                   alt_name, counters)

        eff_charge = _get_unit_type_param(data, unit, "eff charge")
        _add_param(db, "unit", (charger_name,), "efficiency",
                   eff_charge if eff_charge is not None else 1.0, alt_name, counters)

        # FT2 'availability' enters only the capacity margin; it is reported,
        # not written as FT4 unit.availability (which derates every flow).

        # Charger relationships
        _add_relationship(db, "unit__inputNode", (charger_name, output_node),
                          alt_name, counters, entities_added, entity_alts_added)
        _add_relationship(db, "unit__outputNode", (charger_name, storage_node),
                          alt_name, counters, entities_added, entity_alts_added)

        # Charger ramps
        if use_ramps:
            ramp_up = _get_unit_type_param(data, unit, "ramp up (p.u. per min)")
            ramp_down = _get_unit_type_param(data, unit, "ramp down (p.u. per min)")
            if ramp_up is not None and ramp_up > 0:
                _add_param(db, "unit__outputNode",
                           (charger_name, storage_node),
                           "ramp_method", "ramp_limit", alt_name, counters)
                _add_param(db, "unit__outputNode",
                           (charger_name, storage_node),
                           "ramp_speed_up", ramp_up, alt_name, counters)
            if ramp_down is not None and ramp_down > 0:
                _add_param(db, "unit__outputNode",
                           (charger_name, storage_node),
                           "ramp_speed_down", ramp_down, alt_name, counters)

        # Charger non-synchronous / inertia on inputNode (the grid side)
        non_sync = _get_unit_type_param(data, unit, "non synchronous")
        if non_sync is not None and non_sync == 1:
            _add_param(db, "unit__inputNode",
                       (charger_name, output_node),
                       "is_non_synchronous", "yes", alt_name, counters)
        inertia_const = _get_unit_type_param(data, unit, "inertia constant (mws/mw)")
        _add_param_if_set(db, "unit__inputNode",
                          (charger_name, output_node),
                          "inertia_constant", inertia_const, alt_name, counters)

        # ── Discharger unit ───────────────────────────────────────
        _add_entity(db, "unit", discharger_name, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "unit", (discharger_name,), "existing", capacity,
                   alt_name, counters)

        # Discharger conversion method — use min_load_efficiency when applicable
        eff_val = _get_unit_type_param(data, unit, "efficiency")
        eff_at_min = _get_unit_type_param(data, unit, "eff at min load")
        min_load_val = _get_unit_type_param(data, unit, "min load")
        has_min_load = (
            min_load_val is not None and min_load_val > 0
            and eff_at_min is not None
            and eff_val is not None
            and eff_at_min != eff_val
        )
        if has_min_load:
            _add_param(db, "unit", (discharger_name,), "conversion_method",
                       "min_load_efficiency", alt_name, counters)
            _add_param(db, "unit", (discharger_name,), "startup_method",
                       "linear", alt_name, counters)
            _add_param(db, "unit", (discharger_name,), "min_load",
                       min_load_val, alt_name, counters)
            _add_param(db, "unit", (discharger_name,), "efficiency_at_min_load",
                       eff_at_min, alt_name, counters)
        else:
            _add_param(db, "unit", (discharger_name,), "conversion_method",
                       "constant_efficiency", alt_name, counters)

        _add_param(db, "unit", (discharger_name,), "efficiency",
                   eff_val if eff_val is not None else 1.0, alt_name, counters)

        # Discharger startup (only on discharger)
        startup_cost = _get_unit_type_param(data, unit, "startup cost")
        if use_online and startup_cost is not None and startup_cost > 0:
            _add_param(db, "unit", (discharger_name,), "startup_cost",
                       startup_cost, alt_name, counters)
            _add_param(db, "unit", (discharger_name,), "startup_method",
                       "linear", alt_name, counters)

        min_uptime = _get_unit_type_param(data, unit, "min uptime (h)")
        _add_param_if_set(db, "unit", (discharger_name,), "min_uptime",
                          min_uptime, alt_name, counters)
        min_downtime = _get_unit_type_param(data, unit, "min downtime (h)")
        _add_param_if_set(db, "unit", (discharger_name,), "min_downtime",
                          min_downtime, alt_name, counters)

        # Discharger relationships
        _add_relationship(db, "unit__inputNode", (discharger_name, storage_node),
                          alt_name, counters, entities_added, entity_alts_added)
        _add_relationship(db, "unit__outputNode", (discharger_name, output_node),
                          alt_name, counters, entities_added, entity_alts_added)
        _write_generation_limits(data, db, unit, discharger_name, alt_name,
                                 counters, entities_added, entity_alts_added)

        # Discharger O&M cost (only on discharger output)
        om_cost = _get_unit_type_param(data, unit, "o&m cost/mwh")
        _add_param_if_set(db, "unit__outputNode",
                          (discharger_name, output_node),
                          "other_operational_cost", om_cost, alt_name, counters)

        # Discharger non-synchronous / inertia on outputNode (the grid side)
        if non_sync is not None and non_sync == 1:
            _add_param(db, "unit__outputNode",
                       (discharger_name, output_node),
                       "is_non_synchronous", "yes", alt_name, counters)
        _add_param_if_set(db, "unit__outputNode",
                          (discharger_name, output_node),
                          "inertia_constant", inertia_const, alt_name, counters)

        # Discharger ramps
        if use_ramps:
            ramp_up = _get_unit_type_param(data, unit, "ramp up (p.u. per min)")
            ramp_down = _get_unit_type_param(data, unit, "ramp down (p.u. per min)")
            if ramp_up is not None and ramp_up > 0:
                _add_param(db, "unit__outputNode",
                           (discharger_name, output_node),
                           "ramp_method", "ramp_limit", alt_name, counters)
                _add_param(db, "unit__outputNode",
                           (discharger_name, output_node),
                           "ramp_speed_up", ramp_up, alt_name, counters)
            if ramp_down is not None and ramp_down > 0:
                _add_param(db, "unit__outputNode",
                           (discharger_name, output_node),
                           "ramp_speed_down", ramp_down, alt_name, counters)

        # ── Investment params ──────────────────────────────────────
        inv_cost_kw = _get_unit_type_param(data, unit, "inv.cost/kw")
        lifetime = _get_unit_type_param(data, unit, "lifetime")
        interest = _get_unit_type_param(data, unit, "interest")

        # Cost on discharger only; lifetime and interest on all three
        # (always set so sensitivities can enable investment later).  An
        # interest of 0 is written: FlexTool reads 0 as 0 % (straight-line
        # annuity), while no value is the 5 % default.
        _add_param_if_set(db, "unit", (discharger_name,), "invest_cost",
                          inv_cost_kw, alt_name, counters)
        _add_param_if_set(db, "unit", (discharger_name,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "unit", (discharger_name,), "discount_rate",
                          interest, alt_name, counters, skip_zero=False)
        _add_param_if_set(db, "unit", (charger_name,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "unit", (charger_name,), "discount_rate",
                          interest, alt_name, counters, skip_zero=False)
        _add_param_if_set(db, "node", (storage_node,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "discount_rate",
                          interest, alt_name, counters, skip_zero=False)

        # 'invested capacity (MW)' / 'invested storage (MWh)' are in existing
        # (_ft2_capacity); 'max invest' is the FT2 v_invest limit, the one
        # not given derived through the kW/kWh ratio (_storage_max_invest).
        kw_kwh_ratio = _get_unit_type_param(data, unit, "fixed kw/kwh ratio")
        max_mw, max_mwh = _storage_max_invest(
            unit.max_invest_mw, unit.max_invest_mwh, kw_kwh_ratio)
        if max_mw > 0:
            for target in (discharger_name, charger_name):
                _add_param(db, "unit", (target,), "invest_method",
                           "invest_total", alt_name, counters)
                _add_param(db, "unit", (target,), "invest_max_total",
                           max_mw, alt_name, counters)
        if max_mwh > 0:
            _add_param(db, "node", (storage_node,), "invest_method",
                       "invest_total", alt_name, counters)
            _add_param(db, "node", (storage_node,), "invest_max_total",
                       max_mwh, alt_name, counters)

        # ── Charger ↔ Discharger investment constraint ────────────
        # FT2 has one unit, so one v_invest: the charger invests what the
        # discharger invests.  Only v_invest (constraint_invested_capacity_coeff):
        # the existing capacities are equal already.
        charger_link = f"{unit_name}_charger_link"
        _add_entity(db, "constraint", charger_link, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "constraint", (charger_link,), "sense", "equal",
                   alt_name, counters)
        _add_param(db, "constraint", (charger_link,), "constant", 0.0,
                   alt_name, counters)
        _add_param(db, "unit", (charger_name,),
                   "constraint_invested_capacity_coeff",
                   Map([charger_link], [-1.0], index_name="constraint"),
                   alt_name, counters)
        discharger_coeff_indexes = [charger_link]
        discharger_coeff_values = [1.0]

        # ── kW/kWh constraint (discharger ↔ storage node) ────────
        # FT2 investStorage_kW_per_kWh: v_invest (MW) = ratio × v_investStorage
        # (MWh), on v_invest only ('invested ...' capacities are existing).
        if kw_kwh_ratio is not None and kw_kwh_ratio > 0:
            storage_link = f"{unit_name}_storage_link"
            _add_entity(db, "constraint", storage_link, alt_name, counters,
                         entities_added, entity_alts_added)
            _add_param(db, "constraint", (storage_link,), "sense", "equal",
                       alt_name, counters)
            _add_param(db, "constraint", (storage_link,), "constant", 0.0,
                       alt_name, counters)
            _add_param(db, "node", (storage_node,),
                       "constraint_invested_capacity_coeff",
                       Map([storage_link], [-kw_kwh_ratio], index_name="constraint"),
                       alt_name, counters)
            discharger_coeff_indexes.append(storage_link)
            discharger_coeff_values.append(1.0)

        _add_param(db, "unit", (discharger_name,),
                   "constraint_invested_capacity_coeff",
                   Map(discharger_coeff_indexes, discharger_coeff_values,
                       index_name="constraint"),
                   alt_name, counters)

        count += 1

    logger.info("Wrote %d storage charger/discharger pairs.", count)


def _write_units(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write unit entities and their relationships (Section 4)."""
    use_online = _get_master_param(data, "use online") == 1
    use_ramps = _get_master_param(data, "use ramps") == 1

    for unit in data.units:
        if unit.output_node is None:
            logger.warning("Unit type '%s' has no output node; skipping.", unit.unit_type)
            continue

        # Pure storage units are handled by _write_storage_units (charger/discharger)
        if _needs_charger_discharger(unit, data):
            continue

        unit_name = _get_unit_name(unit)
        ut = _get_unit_type(data, unit)

        # Create unit entity
        _add_entity(db, "unit", unit_name, alt_name, counters,
                     entities_added, entity_alts_added)

        # -- Determine conversion_method --
        has_cf = unit.cf_profile is not None
        min_load_val = ut.params.get("min load") if ut else None
        eff_val = ut.params.get("efficiency") if ut else None
        eff_at_min = ut.params.get("eff at min load") if ut else None
        has_min_load = (
            min_load_val is not None and min_load_val > 0
            and eff_at_min is not None
            and eff_val is not None
            and eff_at_min != eff_val
        )

        if has_cf:
            conversion_method = "none"
        elif has_min_load:
            conversion_method = "min_load_efficiency"
        else:
            conversion_method = "constant_efficiency"

        _add_param(db, "unit", (unit_name,), "conversion_method",
                   conversion_method, alt_name, counters)

        # -- existing capacity: ``capacity (MW)`` + ``invested capacity (MW)``
        # (see _ft2_capacity); ``max invest (MW)`` is the investment below. --
        _add_param_if_set(db, "unit", (unit_name,), "existing",
                          _ft2_capacity(unit.capacity_mw, unit.invested_capacity_mw),
                          alt_name, counters)

        # -- efficiency --
        # Check for time-varying efficiency first
        eff_ts = _find_unit_ts(data, unit, "efficiency")
        if eff_ts and eff_ts.data:
            eff_map = _make_time_map(eff_ts.data)
            _add_param(db, "unit", (unit_name,), "efficiency", eff_map,
                       alt_name, counters)
        else:
            eff_val = _get_unit_type_param(data, unit, "efficiency")
            _add_param_if_set(db, "unit", (unit_name,), "efficiency", eff_val,
                              alt_name, counters)

        # -- min_load and efficiency_at_min_load --
        if has_min_load:
            _add_param(db, "unit", (unit_name,), "min_load", min_load_val,
                       alt_name, counters)
            eff_at_min = _get_unit_type_param(data, unit, "eff at min load")
            _add_param_if_set(db, "unit", (unit_name,), "efficiency_at_min_load",
                              eff_at_min, alt_name, counters)

        # -- availability: FT2 uses it only in the capacity margin
        # (flexModel.mod capacityMarginNode / ...NodeGroup); FT4
        # unit.availability would derate dispatch, so it is reported instead
        # (_availability_item). --

        # -- startup parameters --
        startup_cost = _get_unit_type_param(data, unit, "startup cost")
        # min_load_efficiency always requires startup_method = "linear"
        needs_startup = (
            has_min_load
            or (use_online and startup_cost is not None and startup_cost > 0)
        )
        if needs_startup:
            if startup_cost is not None and startup_cost > 0:
                _add_param(db, "unit", (unit_name,), "startup_cost", startup_cost,
                           alt_name, counters)
            _add_param(db, "unit", (unit_name,), "startup_method", "linear",
                       alt_name, counters)

        # -- min_uptime / min_downtime --
        min_uptime = _get_unit_type_param(data, unit, "min uptime (h)")
        _add_param_if_set(db, "unit", (unit_name,), "min_uptime", min_uptime,
                          alt_name, counters)
        min_downtime = _get_unit_type_param(data, unit, "min downtime (h)")
        _add_param_if_set(db, "unit", (unit_name,), "min_downtime", min_downtime,
                          alt_name, counters)

        # -- fixed_cost, invest_cost, lifetime, interest_rate --
        fixed_cost = _get_unit_type_param(data, unit, "fixed cost/kw/year")
        _add_param_if_set(db, "unit", (unit_name,), "fixed_cost", fixed_cost,
                          alt_name, counters)

        invest_cost = _get_unit_type_param(data, unit, "inv.cost/kw")
        _add_param_if_set(db, "unit", (unit_name,), "invest_cost", invest_cost,
                          alt_name, counters)

        lifetime = _get_unit_type_param(data, unit, "lifetime")
        _add_param_if_set(db, "unit", (unit_name,), "lifetime", lifetime,
                          alt_name, counters)

        interest_rate = _get_unit_type_param(data, unit, "interest")
        _add_param_if_set(db, "unit", (unit_name,), "discount_rate", interest_rate,
                          alt_name, counters, skip_zero=False)

        # -- investment: FT2 v_invest <= max invest (MW) --
        has_max_invest = (
            unit.max_invest_mw is not None and unit.max_invest_mw > 0
        )
        if has_max_invest:
            _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                       alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_max_total",
                       unit.max_invest_mw, alt_name, counters)

        # -- Unit relationships --

        # a) Fuel input
        if unit.fuel:
            fuel_market = _fuel_market_name(unit.fuel)
            _add_relationship(db, "unit__inputNode", (unit_name, fuel_market),
                              alt_name, counters, entities_added, entity_alts_added)

        # b) Conversion input (e.g. heat pump)
        if unit.input_grid and unit.input_node:
            _add_relationship(db, "unit__inputNode", (unit_name, unit.input_node),
                              alt_name, counters, entities_added, entity_alts_added)

        # c) Primary output
        _add_relationship(db, "unit__outputNode", (unit_name, unit.output_node),
                          alt_name, counters, entities_added, entity_alts_added)
        _write_generation_limits(data, db, unit, unit_name, alt_name, counters,
                                 entities_added, entity_alts_added)

        # d) Secondary output (CHP, output2)
        if unit.output2_node:
            _add_relationship(db, "unit__outputNode", (unit_name, unit.output2_node),
                              alt_name, counters, entities_added, entity_alts_added)
            # FT2 output2 (flexModel.mod): it has no fuel (the fueluse
            # equation has only the primary v_gen), is not in the unit's
            # capacity limit, min load, ramps or reserves, and is tied to the
            # primary output only by the eq / gt / lt rows and the
            # output2_max_capacity_ratio cap (_write_chp_constraints).  FT4
            # conversion_flow_coeff = 0 is that edge: out of the conversion
            # equation and of the capacity, ramp and min-load constraints.
            _add_param(db, "unit__outputNode", (unit_name, unit.output2_node),
                       "conversion_flow_coeff", 0.0, alt_name, counters)

        # -- unit__outputNode parameters (on primary output) --
        om_cost = _get_unit_type_param(data, unit, "o&m cost/mwh")
        _add_param_if_set(db, "unit__outputNode",
                          (unit_name, unit.output_node),
                          "other_operational_cost", om_cost, alt_name, counters)

        non_sync = _get_unit_type_param(data, unit, "non synchronous")
        if non_sync is not None and non_sync == 1:
            _add_param(db, "unit__outputNode",
                       (unit_name, unit.output_node),
                       "is_non_synchronous", "yes", alt_name, counters)

        inertia_const = _get_unit_type_param(data, unit, "inertia constant (mws/mw)")
        _add_param_if_set(db, "unit__outputNode",
                          (unit_name, unit.output_node),
                          "inertia_constant", inertia_const, alt_name, counters)

        # Ramp parameters
        if use_ramps:
            ramp_up = _get_unit_type_param(data, unit, "ramp up (p.u. per min)")
            ramp_down = _get_unit_type_param(data, unit, "ramp down (p.u. per min)")
            if (ramp_up is not None and ramp_up > 0) or (ramp_down is not None and ramp_down > 0):
                _add_param(db, "unit__outputNode",
                           (unit_name, unit.output_node),
                           "ramp_method", "ramp_limit", alt_name, counters)
                _add_param_if_set(db, "unit__outputNode",
                                  (unit_name, unit.output_node),
                                  "ramp_speed_up", ramp_up, alt_name, counters)
                _add_param_if_set(db, "unit__outputNode",
                                  (unit_name, unit.output_node),
                                  "ramp_speed_down", ramp_down, alt_name, counters)

    logger.info("Wrote %d units.", len(data.units))


def _write_profiles(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write CF profiles and link them to units (Section 5)."""
    count = 0
    for profile_ts in data.cf_profiles:
        if not profile_ts.data:
            continue

        profile_name = profile_ts.name
        _add_entity(db, "profile", profile_name, alt_name, counters,
                     entities_added, entity_alts_added)

        profile_map = _make_time_map(profile_ts.data)
        _add_param(db, "profile", (profile_name,), "profile", profile_map,
                   alt_name, counters)

        # Find units that reference this profile
        for unit in data.units:
            if unit.cf_profile == profile_name and unit.output_node:
                unit_name = _get_unit_name(unit)
                _add_relationship(
                    db, "unit__node__profile",
                    (unit_name, unit.output_node, profile_name),
                    alt_name, counters, entities_added, entity_alts_added,
                )
                _add_param(
                    db, "unit__node__profile",
                    (unit_name, unit.output_node, profile_name),
                    "profile_method", "upper_limit", alt_name, counters,
                )
        count += 1

    logger.info("Wrote %d CF profiles.", count)


def _write_inflow_units(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Create dedicated inflow nodes for hydro/inflow units (Section 6).

    For each unit with an inflow profile, creates a dedicated input node:
    - Hydro_RES (has storage): node with node_type='storage' + existing
    - Hydro_ROR (no storage): node with node_type='balance' (use-it-or-lose-it)

    The inflow time series is set on this node and the unit draws from it
    via a unit__inputNode relationship.
    """
    lolp = _get_master_param(data, "loss of load penalty")
    count = 0
    for unit in data.units:
        if not unit.inflow_profile or not unit.output_node:
            continue

        # Units with eff_charge get their inflow on the storage node
        # via _write_storage_units (pumped hydro pattern)
        if _has_eff_charge(unit, data):
            continue

        inflow_ts = _find_inflow_profile(data, unit.inflow_profile)
        if inflow_ts is None or not inflow_ts.data:
            logger.warning(
                "Inflow profile '%s' not found for unit '%s'.",
                unit.inflow_profile, _get_unit_name(unit),
            )
            continue

        unit_name = _get_unit_name(unit)
        inflow_node = _get_unit_node_name(unit)

        # Apply inflow multiplier if set
        ts_data = inflow_ts.data
        if unit.inflow_multiplier is not None and unit.inflow_multiplier != 0:
            ts_data = {k: v * unit.inflow_multiplier for k, v in ts_data.items()}
        inflow_map = _make_time_map(ts_data)

        # Create the inflow node.  Always 'storage' — even with 0 capacity this
        # just creates an unused state variable but allows sensitivities to add
        # storage capacity later without needing to also change node_type.
        _add_entity(db, "node", inflow_node, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (inflow_node,), "node_type", "storage",
                   alt_name, counters)
        if lolp > 0:
            _add_param(db, "node", (inflow_node,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (inflow_node,), "penalty_down", 0.0,
                   alt_name, counters)
        _add_param_if_set(db, "node", (inflow_node,), "existing",
                          _ft2_capacity(unit.storage_mwh, unit.invested_storage_mwh),
                          alt_name, counters, skip_zero=False)
        _write_storage_start_end(db, inflow_node, unit, alt_name, counters)
        # Self-discharge loss from unit_type
        self_discharge = _get_unit_type_param(data, unit, "self discharge loss")
        _add_param_if_set(db, "node", (inflow_node,), "self_discharge_loss",
                          self_discharge, alt_name, counters)

        # Set inflow time series on the node
        _add_param(db, "node", (inflow_node,), "inflow", inflow_map,
                   alt_name, counters)
        _add_param(db, "node", (inflow_node,), "inflow_method",
                   "use_original", alt_name, counters)

        # Connect as unit__inputNode
        _add_relationship(db, "unit__inputNode", (unit_name, inflow_node),
                          alt_name, counters, entities_added, entity_alts_added)

        count += 1

    # Create standalone inflow nodes for profiles not referenced by any unit
    used_profiles = {unit.inflow_profile for unit in data.units if unit.inflow_profile}
    for inflow_ts in data.inflow_profiles:
        if inflow_ts.name in used_profiles:
            continue
        if not inflow_ts.data:
            continue

        node_name = f"{inflow_ts.name}_node"
        inflow_map = _make_time_map(inflow_ts.data)

        # Create the entity but set it inactive — it's not connected to any
        # unit, so having it active would cause infeasibilities/penalties.
        key = ("node", (node_name,))
        if key not in entities_added:
            entities_added.add(key)
            try:
                db.add_entity(entity_class_name="node", name=node_name)
                counters.entities += 1
            except SpineDBAPIError:
                pass
        ea_key = ("node", (node_name,), alt_name)
        if ea_key not in entity_alts_added:
            entity_alts_added.add(ea_key)
            try:
                db.add_entity_alternative(
                    entity_class_name="node",
                    entity_byname=(node_name,),
                    alternative_name=alt_name,
                    active=False,
                )
                counters.entity_alternatives += 1
            except SpineDBAPIError:
                pass

        _add_param(db, "node", (node_name,), "node_type", "storage",
                   alt_name, counters)
        if lolp > 0:
            _add_param(db, "node", (node_name,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (node_name,), "penalty_down", 0.0,
                   alt_name, counters)
        _add_param(db, "node", (node_name,), "storage_binding_method",
                   "bind_within_solve", alt_name, counters)
        _add_param(db, "node", (node_name,), "storage_start_end_method",
                   "fix_nothing", alt_name, counters)
        _add_param(db, "node", (node_name,), "inflow", inflow_map,
                   alt_name, counters)
        _add_param(db, "node", (node_name,), "inflow_method",
                   "use_original", alt_name, counters)
        count += 1
        logger.info("Created standalone inflow node '%s' (inactive — not connected to any unit).", node_name)

    logger.info("Wrote %d inflow nodes total.", count)


def _write_connections(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write nodeNode connections (Section 7)."""
    for conn in data.connections:
        conn_name = f"{conn.node1}_{conn.node2}"

        _add_entity(db, "connection", conn_name, alt_name, counters,
                     entities_added, entity_alts_added)

        # existing capacity: the larger direction + invested capacity
        # (see _ft2_capacity)
        cap = conn.cap_rightward_mw or 0.0
        if conn.cap_leftward_mw is not None:
            cap = max(cap, conn.cap_leftward_mw)
        _add_param_if_set(db, "connection", (conn_name,), "existing",
                          _ft2_capacity(cap, conn.invested_capacity_mw),
                          alt_name, counters)

        # efficiency = 1 - loss
        if conn.loss is not None:
            eff = 1.0 - conn.loss
            _add_param(db, "connection", (conn_name,), "efficiency", eff,
                       alt_name, counters)

        _add_param(db, "connection", (conn_name,), "transfer_method",
                   "regular", alt_name, counters)

        _add_param_if_set(db, "connection", (conn_name,), "invest_cost",
                          conn.invest_cost_per_kw, alt_name, counters)
        _add_param_if_set(db, "connection", (conn_name,), "lifetime",
                          conn.lifetime, alt_name, counters)
        _add_param_if_set(db, "connection", (conn_name,), "discount_rate",
                          conn.interest, alt_name, counters, skip_zero=False)

        has_max_invest = (
            conn.max_invest_mw is not None and conn.max_invest_mw > 0
        )
        if has_max_invest:
            _add_param(db, "connection", (conn_name,), "invest_method",
                       "invest_total", alt_name, counters)
            _add_param(db, "connection", (conn_name,), "invest_max_total",
                       conn.max_invest_mw, alt_name, counters)

        if conn.is_hvdc:
            _add_param(db, "connection", (conn_name,), "is_DC", "yes",
                       alt_name, counters)

        # connection__node__node relationship
        _add_relationship(db, "connection__node__node",
                          (conn_name, conn.node1, conn.node2),
                          alt_name, counters, entities_added, entity_alts_added)

    logger.info("Wrote %d connections.", len(data.connections))


def _write_groups(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write groups: grid groups, node groups, unit groups, CO2 group (Section 8)."""
    use_cap_margin = _get_master_param(data, "use capacity margin") == 1
    use_non_sync = _get_master_param(data, "use non synchronous") == 1
    use_inertia = _get_master_param(data, "use inertia limit") == 1
    lack_cap_penalty = _get_master_param(data, "lack of capacity penalty")
    lack_inertia_penalty = _get_master_param(data, "lack of inertia penalty")
    loss_reserves_penalty = _get_master_param(data, "loss of reserves penalty")

    # -- Grid groups --
    grids: dict[str, list[str]] = {}
    for gn in data.grid_nodes:
        grids.setdefault(gn.grid, []).append(gn.node)

    for grid_name, nodes in grids.items():
        _add_entity(db, "group", grid_name, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "group", (grid_name,), "output_nodeGroup_indicators", "yes",
                   alt_name, counters)
        _add_param(db, "group", (grid_name,), "output_nodeGroup_dispatch", "yes",
                   alt_name, counters)
        for node_name in nodes:
            _add_relationship(db, "group__node", (grid_name, node_name),
                              alt_name, counters, entities_added, entity_alts_added)

    logger.info("Wrote %d grid groups.", len(grids))

    # -- Node groups --
    # Build a lookup: node_group_name -> list of node names (from gridNode)
    ng_nodes: dict[str, list[str]] = {}
    for gn in data.grid_nodes:
        for ng_name in gn.node_groups:
            ng_nodes.setdefault(ng_name, []).append(gn.node)

    for ng in data.node_groups:
        has_constraints = (
            (ng.capacity_margin_mw is not None and ng.capacity_margin_mw > 0)
            or (ng.non_synchronous_share is not None and ng.non_synchronous_share > 0)
            or (ng.inertia_limit_mws is not None and ng.inertia_limit_mws > 0)
        )
        if not has_constraints:
            continue

        _add_entity(db, "group", ng.name, alt_name, counters,
                     entities_added, entity_alts_added)

        if use_cap_margin and ng.capacity_margin_mw is not None and ng.capacity_margin_mw > 0:
            _add_param(db, "group", (ng.name,), "capacity_margin",
                       ng.capacity_margin_mw, alt_name, counters)
            _add_param(db, "group", (ng.name,), "has_capacity_margin", "yes",
                       alt_name, counters)
            if lack_cap_penalty > 0:
                _add_param(db, "group", (ng.name,), "penalty_capacity_margin",
                           lack_cap_penalty, alt_name, counters)

        if use_non_sync and ng.non_synchronous_share is not None and ng.non_synchronous_share > 0:
            _add_param(db, "group", (ng.name,), "non_synchronous_limit",
                       ng.non_synchronous_share, alt_name, counters)
            _add_param(db, "group", (ng.name,), "has_non_synchronous", "yes",
                       alt_name, counters)
            if loss_reserves_penalty > 0:
                _add_param(db, "group", (ng.name,), "penalty_non_synchronous",
                           loss_reserves_penalty, alt_name, counters)

        if use_inertia and ng.inertia_limit_mws is not None and ng.inertia_limit_mws > 0:
            _add_param(db, "group", (ng.name,), "inertia_limit",
                       ng.inertia_limit_mws, alt_name, counters)
            _add_param(db, "group", (ng.name,), "has_inertia", "yes",
                       alt_name, counters)
            if lack_inertia_penalty > 0:
                _add_param(db, "group", (ng.name,), "penalty_inertia",
                           lack_inertia_penalty, alt_name, counters)

        # Link nodes to this group
        for node_name in ng_nodes.get(ng.name, []):
            _add_relationship(db, "group__node", (ng.name, node_name),
                              alt_name, counters, entities_added, entity_alts_added)

    logger.info("Wrote %d node groups.", len(data.node_groups))

    # -- Unit groups --
    # Build lookup: group_name -> list of UnitInstance
    ug_units: dict[str, list[UnitInstance]] = {}
    for unit in data.units:
        if unit.unit_group:
            ug_units.setdefault(unit.unit_group, []).append(unit)

    for ug in data.unit_groups:
        _add_entity(db, "group", ug.name, alt_name, counters,
                     entities_added, entity_alts_added)

        has_invest_limits = (
            (ug.max_invest_mw is not None and ug.max_invest_mw > 0)
            or (ug.min_invest_mw is not None and ug.min_invest_mw > 0)
        )
        if has_invest_limits:
            _add_param(db, "group", (ug.name,), "invest_method", "invest_total",
                       alt_name, counters)
        _add_param_if_set(db, "group", (ug.name,), "invest_max_total",
                          ug.max_invest_mw, alt_name, counters)
        _add_param_if_set(db, "group", (ug.name,), "invest_min_total",
                          ug.min_invest_mw, alt_name, counters)

        _add_param(db, "group", (ug.name,), "flow_aggregator", "yes",
                   alt_name, counters)

        # Link units to this group
        for unit in ug_units.get(ug.name, []):
            if unit.output_node is None:
                continue
            if _needs_charger_discharger(unit, data):
                # Storage units are represented as charger + discharger
                unit_name = _get_unit_name(unit)
                charger = f"{unit_name}_charger"
                discharger = f"{unit_name}_discharger"
                _add_relationship(db, "group__unit", (ug.name, charger),
                                  alt_name, counters, entities_added, entity_alts_added)
                _add_relationship(db, "group__unit__node",
                                  (ug.name, charger, unit.output_node),
                                  alt_name, counters, entities_added, entity_alts_added)
                _add_relationship(db, "group__unit", (ug.name, discharger),
                                  alt_name, counters, entities_added, entity_alts_added)
                _add_relationship(db, "group__unit__node",
                                  (ug.name, discharger, unit.output_node),
                                  alt_name, counters, entities_added, entity_alts_added)
            else:
                unit_name = _get_unit_name(unit)
                _add_relationship(db, "group__unit", (ug.name, unit_name),
                                  alt_name, counters, entities_added, entity_alts_added)
                _add_relationship(db, "group__unit__node",
                                  (ug.name, unit_name, unit.output_node),
                                  alt_name, counters, entities_added, entity_alts_added)

    logger.info("Wrote %d unit groups.", len(data.unit_groups))

    # -- CO2 group --
    co2_cost = _get_master_param(data, "co2 cost")
    if co2_cost > 0:
        _add_entity(db, "group", "co2_price", alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "group", ("co2_price",), "co2_method", "price",
                   alt_name, counters)
        _add_param(db, "group", ("co2_price",), "co2_price", co2_cost,
                   alt_name, counters)

        # Link all fuel market nodes
        used_fuels: set[str] = set()
        for unit in data.units:
            if unit.fuel:
                used_fuels.add(unit.fuel)
        for fuel_name in used_fuels:
            market_name = _fuel_market_name(fuel_name)
            _add_relationship(db, "group__node", ("co2_price", market_name),
                              alt_name, counters, entities_added, entity_alts_added)

        logger.info("Wrote CO2 group with cost=%.2f.", co2_cost)


def _find_reserve_node_ts(
    data: OldFlexToolData, node_name: str,
) -> TimeSeriesData | None:
    """Find a reserve time series for a specific node."""
    for ts in data.reserve_node_ts:
        if ts.name == node_name:
            return ts
    return None


def _find_reserve_group_ts(
    data: OldFlexToolData, group_name: str,
) -> TimeSeriesData | None:
    """Find a reserve time series for a specific group."""
    for ts in data.reserve_group_ts:
        if ts.name == group_name:
            return ts
    return None


def _get_reserve_method(
    use_ts_reserve: float | None,
    use_dynamic_reserve: float | None,
) -> str | None:
    """Map old boolean flags to the new reserve_method string.

    Returns None for "no_reserve" (both flags off) to signal the caller
    should skip creating this reserve.
    """
    ts = use_ts_reserve is not None and use_ts_reserve == 1
    dyn = use_dynamic_reserve is not None and use_dynamic_reserve == 1
    if ts and dyn:
        return "timeseries_and_dynamic"
    if ts:
        return "timeseries_only"
    if dyn:
        return "dynamic_only"
    return None  # no_reserve — skip


def _write_reserves(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write reserve entities, relationships, and parameters (Section 8b).

    Creates the "primary" reserve entity and builds reserve__upDown__group
    and reserve__upDown__unit__node relationships from old FlexTool reserve
    data (node groups, grid nodes, unit types, and reserve time series).
    """
    penalty_reserve = _get_master_param(data, "loss of reserves penalty")

    # 1. Create the "primary" reserve and "up" upDown entities
    #    (both exist in the template but are removed by purge)
    _add_entity(db, "reserve", "primary", alt_name, counters,
                entities_added, entity_alts_added)
    _add_entity(db, "upDown", "up", alt_name, counters,
                entities_added, entity_alts_added)

    # Build lookup: node_group_name -> list of node names (from gridNode)
    ng_nodes: dict[str, list[str]] = {}
    for gn in data.grid_nodes:
        for ng_name in gn.node_groups:
            ng_nodes.setdefault(ng_name, []).append(gn.node)

    # Track which nodes are covered by a group-level reserve
    # and which group each node belongs to (for unit matching)
    nodes_with_reserves: set[str] = set()
    node_to_reserve_group: dict[str, str] = {}
    reserve_count = 0

    # 2. Node groups with reserves
    for ng in data.node_groups:
        reserve_method = _get_reserve_method(ng.use_ts_reserve, ng.use_dynamic_reserve)
        if reserve_method is None:
            continue

        group_name = ng.name

        # The group entity may not exist yet (if it had no capacity/inertia
        # constraints, _write_groups would have skipped it).
        _add_entity(db, "group", group_name, alt_name, counters,
                    entities_added, entity_alts_added)

        # Ensure group__node links exist for this group
        for node_name in ng_nodes.get(group_name, []):
            _add_relationship(db, "group__node", (group_name, node_name),
                              alt_name, counters, entities_added, entity_alts_added)

        # Create reserve__upDown__group relationship
        rel_elements = ("primary", "up", group_name)
        _add_relationship(db, "reserve__upDown__group", rel_elements,
                          alt_name, counters, entities_added, entity_alts_added)

        # Set reserve_method
        _add_param(db, "reserve__upDown__group", rel_elements,
                   "reserve_method", reserve_method, alt_name, counters)

        # Set penalty_reserve from master params
        _add_param_if_set(db, "reserve__upDown__group", rel_elements,
                          "penalty_reserve", penalty_reserve, alt_name, counters)

        # Reserve time series for this group
        group_ts = _find_reserve_group_ts(data, group_name)
        if group_ts and group_ts.data:
            reservation_map = _make_time_map(group_ts.data)
            _add_param(db, "reserve__upDown__group", rel_elements,
                       "reservation", reservation_map, alt_name, counters)

        # Record which nodes are covered
        for node_name in ng_nodes.get(group_name, []):
            nodes_with_reserves.add(node_name)
            node_to_reserve_group[node_name] = group_name

        reserve_count += 1

    logger.info("Wrote %d node-group reserves.", reserve_count)

    # 3. Grid nodes with reserves (only if not already covered by a nodeGroup)
    node_reserve_count = 0
    for gn in data.grid_nodes:
        # Skip if this node is already in a reserve-enabled group
        if gn.node in nodes_with_reserves:
            continue

        reserve_method = _get_reserve_method(gn.use_ts_reserve, gn.use_dynamic_reserve)
        if reserve_method is None:
            continue

        # Create a single-node reserve group
        reserve_group_name = f"reserve_{gn.node}"

        _add_entity(db, "group", reserve_group_name, alt_name, counters,
                    entities_added, entity_alts_added)

        # Link node to the new group
        _add_relationship(db, "group__node", (reserve_group_name, gn.node),
                          alt_name, counters, entities_added, entity_alts_added)

        # Create reserve__upDown__group relationship
        rel_elements = ("primary", "up", reserve_group_name)
        _add_relationship(db, "reserve__upDown__group", rel_elements,
                          alt_name, counters, entities_added, entity_alts_added)

        # Set reserve_method
        _add_param(db, "reserve__upDown__group", rel_elements,
                   "reserve_method", reserve_method, alt_name, counters)

        # Set penalty_reserve
        _add_param_if_set(db, "reserve__upDown__group", rel_elements,
                          "penalty_reserve", penalty_reserve, alt_name, counters)

        # Reserve time series for this node
        node_ts = _find_reserve_node_ts(data, gn.node)
        if node_ts and node_ts.data:
            reservation_map = _make_time_map(node_ts.data)
            _add_param(db, "reserve__upDown__group", rel_elements,
                       "reservation", reservation_map, alt_name, counters)

        # Record this node as having reserves
        nodes_with_reserves.add(gn.node)
        node_to_reserve_group[gn.node] = reserve_group_name
        node_reserve_count += 1

    logger.info("Wrote %d single-node reserve groups.", node_reserve_count)

    # 5. Unit reserve participation
    unit_reserve_count = 0
    for unit in data.units:
        if unit.output_node is None:
            continue

        # Look up max_reserve from unit_type
        max_reserve = _get_unit_type_param(data, unit, "max reserve")
        if max_reserve is None or max_reserve == 0:
            continue

        # Determine which node to use: check output_node first, then input_node
        # (skip output2 — reserves are electrical grid only)
        reserve_node: str | None = None
        if unit.output_node in nodes_with_reserves:
            reserve_node = unit.output_node
        elif unit.input_node is not None and unit.input_node in nodes_with_reserves:
            reserve_node = unit.input_node

        if reserve_node is None:
            continue

        unit_name = _get_unit_name(unit)

        if _needs_charger_discharger(unit, data):
            # Both charger and discharger participate in reserves
            for sub_name in (f"{unit_name}_charger", f"{unit_name}_discharger"):
                rel_elements = ("primary", "up", sub_name, reserve_node)
                _add_relationship(db, "reserve__upDown__unit__node", rel_elements,
                                  alt_name, counters, entities_added, entity_alts_added)
                _add_param(db, "reserve__upDown__unit__node", rel_elements,
                           "max_share", max_reserve, alt_name, counters)
                _add_param_if_set(db, "reserve__upDown__unit__node", rel_elements,
                                  "increase_reserve_ratio",
                                  unit.reserve_increase_ratio, alt_name, counters)
                unit_reserve_count += 1
        else:
            rel_elements = ("primary", "up", unit_name, reserve_node)
            _add_relationship(db, "reserve__upDown__unit__node", rel_elements,
                              alt_name, counters, entities_added, entity_alts_added)
            _add_param(db, "reserve__upDown__unit__node", rel_elements,
                       "max_share", max_reserve, alt_name, counters)
            _add_param_if_set(db, "reserve__upDown__unit__node", rel_elements,
                              "increase_reserve_ratio",
                              unit.reserve_increase_ratio, alt_name, counters)
            unit_reserve_count += 1

    logger.info("Wrote %d unit reserve participations.", unit_reserve_count)


def _write_chp_constraints(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """FT2 output2 rows → FT4 user constraints (Section 9).

    FT2 (``flexModel.mod`` ``output2_equals_output1_times_x``,
    ``output2_greater_than_output1_times_x``, ``output2_less_than_...``,
    ``upwardLimitOutput2``), with C = capacity + invested capacity +
    v_invest of the primary output::

        output2  =, >=, <=  coeff × output1 + constant × C
        output2  <=         output2_max_capacity_ratio × C

    A row exists when its coeff or its constant is non-zero.  In FT4 the row
    is ``output2 − coeff × output1 − constant × C  {sense}  0``, with C the
    capacity in service in the solve: ``constraint_cumulative_pre_built_capacity_coeff``
    (existing capacity, including what an earlier solve of the chain
    invested) plus ``constraint_invested_capacity_coeff`` (what this solve
    invests), both −constant on the unit.  FT2 has no retirements, so the
    two are all of C.  Not ``constraint_invested_capacity_coeff`` alone: in
    the dispatch solve after the invest solve the invested MW are pre-built,
    and the row would lose them.

    Older workbooks (header ``output2 max capacity (MW)``, see
    ``UnitInstance.output2_constants_mw``) give the constants in MW:
    ``output2 − coeff × output1  {sense}  constant``, no capacity term, and
    ``output2 max capacity (MW)`` is a plain cap.  All coefficients of a unit
    go into one Map per edge.
    """
    count = 0
    mw_units: list[str] = []
    for unit in data.units:
        if not unit.output2_node or not unit.output_node:
            continue
        if _needs_charger_discharger(unit, data):
            continue

        unit_name = _get_unit_name(unit)
        constants_mw = unit.output2_constants_mw
        if constants_mw and any((unit.output2_eq_constant, unit.output2_gt_constant,
                                 unit.output2_lt_constant)):
            mw_units.append(unit_name)
        if not any((unit.output2_eq_coeff, unit.output2_eq_constant,
                    unit.output2_gt_coeff, unit.output2_gt_constant,
                    unit.output2_lt_coeff, unit.output2_lt_constant,
                    unit.output2_max_capacity_ratio, unit.output2_max_capacity)):
            counters.unmapped.append(UnmappedItem(
                "units", "output2 grid / node", [unit_name],
                "the unit has an output2 but no 'output2 eq / gt / lt' row and "
                "no output2 maximum, so its output2 is free: it uses no fuel and "
                f"has no limit (in FlexTool 4 only {unit.output2_node}'s "
                "balance and the solver's upper bound for flows limit it). "
                "FlexTool 2 behaves the same. If the unit should tie its "
                "output2 to its main output, add the rows in the workbook and "
                "import again.",
            ))
        primary_coeffs: dict[str, float] = {}
        secondary_coeffs: dict[str, float] = {}
        capacity_coeffs: dict[str, float] = {}

        def row(cname: str, sense: str, coeff: float, per_capacity: float,
                constant: float = 0.0) -> None:
            _add_entity(db, "constraint", cname, alt_name, counters,
                        entities_added, entity_alts_added)
            _add_param(db, "constraint", (cname,), "sense", sense,
                       alt_name, counters)
            _add_param(db, "constraint", (cname,), "constant", constant,
                       alt_name, counters)
            secondary_coeffs[cname] = 1.0
            if coeff:
                primary_coeffs[cname] = -coeff
            if per_capacity:
                capacity_coeffs[cname] = -per_capacity

        for suffix, sense, coeff, constant in (
            ("eq", "equal", unit.output2_eq_coeff, unit.output2_eq_constant),
            ("gt", "greater_than", unit.output2_gt_coeff, unit.output2_gt_constant),
            ("lt", "less_than", unit.output2_lt_coeff, unit.output2_lt_constant),
        ):
            if not coeff and not constant:
                continue
            if constants_mw:
                row(f"{unit_name}_{suffix}", sense, coeff or 0.0, 0.0,
                    constant=constant or 0.0)
            else:
                row(f"{unit_name}_{suffix}", sense, coeff or 0.0, constant or 0.0)
            count += 1
        if unit.output2_max_capacity_ratio:
            row(f"{unit_name}_output2_max", "less_than", 0.0,
                unit.output2_max_capacity_ratio)
            count += 1
        elif unit.output2_max_capacity:
            row(f"{unit_name}_output2_max", "less_than", 0.0, 0.0,
                constant=unit.output2_max_capacity)
            count += 1

        for cls, edge, param, coeffs in (
            ("unit__outputNode", (unit_name, unit.output_node),
             "constraint_flow_coeff", primary_coeffs),
            ("unit__outputNode", (unit_name, unit.output2_node),
             "constraint_flow_coeff", secondary_coeffs),
            ("unit", (unit_name,),
             "constraint_cumulative_pre_built_capacity_coeff", capacity_coeffs),
            ("unit", (unit_name,),
             "constraint_invested_capacity_coeff", capacity_coeffs),
        ):
            if not coeffs:
                continue
            names = list(coeffs)
            _add_param(db, cls, edge, param,
                       Map(names, [coeffs[n] for n in names], index_name="constraint"),
                       alt_name, counters)

    if mw_units:
        counters.assumptions.append(
            "This workbook has the older column 'output2 max capacity (MW)', so "
            "the 'output2 eq / gt / lt constant' values were read as MW (not as "
            f"shares of the unit's capacity): {', '.join(mw_units)}.")
    logger.info("Wrote %d CHP constraints.", count)


def _write_timeline_and_solve(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write timeline, timeset, solve, and model entities (Sections 10)."""
    if not data.time_steps:
        logger.warning("No time steps found; skipping timeline/solve creation.")
        return

    time_period_duration = _get_master_param(data, "time period duration")
    duration_hours = time_period_duration / 60.0 if time_period_duration > 0 else 1.0

    # -- Timeline --
    _add_entity(db, "timeline", "y2020", alt_name, counters,
                 entities_added, entity_alts_added)

    # timestep_duration: Map of ALL time step IDs -> duration_hours
    all_time_ids = [ts.time_id for ts in data.time_steps]
    all_durations = [duration_hours] * len(all_time_ids)
    timestep_map = Map(all_time_ids, all_durations, index_name="time")
    _add_param(db, "timeline", ("y2020",), "timestep_duration", timestep_map,
               alt_name, counters)

    # -- Timeset (dispatch) --
    _add_entity(db, "timeset", "dispatch_set", alt_name, counters,
                 entities_added, entity_alts_added)
    _add_param(db, "timeset", ("dispatch_set",), "timeline", "y2020",
               alt_name, counters)

    # Build timeset_duration from contiguous blocks of in_use=True (none when
    # no step is in use: only allowed where the modes do not use the set,
    # _require_time_in_use).
    dispatch_duration = _build_timeset_duration(data.time_steps, "dispatch", duration_hours)
    _add_param(db, "timeset", ("dispatch_set",), "timeset_duration",
               dispatch_duration, alt_name, counters)

    # -- Timeset (invest) if in_use_invest differs --
    has_invest_ts = any(ts.in_use_invest != ts.in_use for ts in data.time_steps)
    invest_set_name = "invest_set" if has_invest_ts else "dispatch_set"

    if has_invest_ts:
        _add_entity(db, "timeset", "invest_set", alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "timeset", ("invest_set",), "timeline", "y2020",
                   alt_name, counters)
        invest_duration = _build_timeset_duration(data.time_steps, "invest", duration_hours)
        _add_param(db, "timeset", ("invest_set",), "timeset_duration",
                   invest_duration, alt_name, counters)

    # -- Solve entities (always create both dispatch and invest) --
    mode_invest = _get_master_param(data, "mode invest") == 1
    mode_dispatch = _get_master_param(data, "mode dispatch") == 1

    # Dispatch solve
    _add_entity(db, "solve", "dispatch", alt_name, counters,
                 entities_added, entity_alts_added)
    _add_param(db, "solve", ("dispatch",), "solve_mode", "single_solve",
               alt_name, counters)
    _add_param(db, "solve", ("dispatch",), "period_timeset",
               Map(["p2020"], ["dispatch_set"], index_name="period"),
               alt_name, counters)
    _add_param(db, "solve", ("dispatch",), "realized_periods",
               Array(["p2020"]), alt_name, counters)
    _add_param(db, "solve", ("dispatch",), "solver", "highs",
               alt_name, counters)

    # Invest solve
    _add_entity(db, "solve", "invest", alt_name, counters,
                 entities_added, entity_alts_added)
    _add_param(db, "solve", ("invest",), "solve_mode", "single_solve",
               alt_name, counters)
    _add_param(db, "solve", ("invest",), "period_timeset",
               Map(["p2020"], [invest_set_name], index_name="period"),
               alt_name, counters)
    _add_param(db, "solve", ("invest",), "invest_periods",
               Array(["p2020"]), alt_name, counters)
    _add_param(db, "solve", ("invest",), "realized_invest_periods",
               Array(["p2020"]), alt_name, counters)
    # realized_periods on invest only when dispatch is not active
    # (when dispatch is active, realization happens in the dispatch solve)
    if not mode_dispatch:
        _add_param(db, "solve", ("invest",), "realized_periods",
                   Array(["p2020"]), alt_name, counters)
    _add_param(db, "solve", ("invest",), "solver", "highs",
               alt_name, counters)

    # Model solves based on active modes
    if mode_invest and mode_dispatch:
        # Invest nests dispatch
        _add_param(db, "solve", ("invest",), "contains_solves", "dispatch",
                   alt_name, counters)
        model_solves = Array(["invest"])
    elif mode_invest:
        # Invest only (no nested dispatch)
        model_solves = Array(["invest"])
    else:
        # Dispatch only (default)
        model_solves = Array(["dispatch"])

    # -- Model --
    _add_entity(db, "model", "flexTool", alt_name, counters,
                 entities_added, entity_alts_added)
    _add_param(db, "model", ("flexTool",), "solves", model_solves,
               alt_name, counters)

    modes = []
    if mode_invest:
        modes.append("invest")
    if mode_dispatch:
        modes.append("dispatch")
    logger.info(
        "Wrote timeline (%d steps), timeset, solve (%s), and model.",
        len(data.time_steps),
        "+".join(modes) if modes else "dispatch",
    )


def _build_timeset_duration(
    time_steps: list,
    mode: str,
    duration_hours: float,
) -> Map | None:
    """Build a timeset_duration Map from contiguous blocks of in_use steps.

    Args:
        time_steps: List of TimeStep objects.
        mode: "dispatch" (uses in_use) or "invest" (uses in_use_invest).
        duration_hours: Duration of each time step in hours.

    Returns:
        A Map where each key is the start time_id of a contiguous block
        and each value is the total duration of that block in hours;
        ``None`` when no step is in use.
    """

    starts: list[str] = []
    durations: list[float] = []
    block_start: str | None = None
    block_count: int = 0

    for ts in time_steps:
        in_use = ts.in_use if mode == "dispatch" else ts.in_use_invest
        if in_use:
            if block_start is None:
                block_start = ts.time_id
                block_count = 1
            else:
                block_count += 1
        else:
            if block_start is not None:
                starts.append(block_start)
                durations.append(block_count * duration_hours)
                block_start = None
                block_count = 0

    # Close last block if open
    if block_start is not None:
        starts.append(block_start)
        durations.append(block_count * duration_hours)

    if not starts:
        return None
    return Map(starts, durations, index_name="time")


def _require_time_in_use(data: OldFlexToolData, xlsm_path: str) -> None:
    """Refuse a workbook whose solves have no time step in use.

    FT2 (``flexModel.mod`` ``time_in_use``) solves the ``ts_time`` steps with
    ``in_use`` (dispatch) or, with ``mode_invest``, ``in_use_invest``; a
    workbook without any has nothing to solve (and FT4 no timeset).
    """
    mode_invest = _get_master_param(data, "mode invest") == 1
    mode_dispatch = _get_master_param(data, "mode dispatch") == 1
    needed = []
    if mode_dispatch or not mode_invest:
        needed.append(("in_use", "dispatch"))
    if mode_invest:
        needed.append(("in_use_invest", "invest"))
    for column, mode in needed:
        attr = "in_use" if mode == "dispatch" else "in_use_invest"
        if not any(getattr(ts, attr) for ts in data.time_steps):
            raise ValueError(
                f"'{xlsm_path}' has no time step in use: column '{column}' of "
                f"its 'ts_time' sheet has no 1 ({len(data.time_steps)} steps), "
                f"so the {mode} solve has nothing to solve, in FlexTool 2 as "
                "well. A FlexTool 2 master workbook can take ts_time from its "
                "time series folder instead (ts/<input file name>/ts_time.csv, "
                "see 'Time series filter' on its 'Settings and filters' sheet). "
                f"Mark the steps to solve with 1 in '{column}' and import again."
            )


# ---------------------------------------------------------------------------
# Purge helper
# ---------------------------------------------------------------------------


def _purge_database(db: DatabaseMapping) -> None:
    """Remove data from the database before import.

    Purges entities, alternatives, scenarios, and parameter values, but keeps
    entity classes and parameter definitions (from the template).
    """
    from spinedb_api import Asterisk

    try:
        db.remove_parameter_value(id=Asterisk)
        db.remove_entity(id=Asterisk)
        db.remove_alternative(id=Asterisk)
        db.remove_scenario(id=Asterisk)
        db.commit_session("Purged database before old FlexTool import")
    except NothingToCommit:
        pass
    except SpineDBAPIError as exc:
        raise RuntimeError(f"Failed to purge database: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_old_flextool_to_db(
    data: OldFlexToolData,
    db_url: str,
    alternative_name: str = "base",
    purge: bool = True,
    *,
    strict: bool = False,
) -> _Counters:
    """Write old FlexTool data to a Spine database.

    Args:
        data: Parsed old FlexTool data from read_old_flextool().
        db_url: Spine database URL (e.g., "sqlite:///path/to/db.sqlite").
            Must be a FlexTool database at the import schema version (56);
            see :func:`import_old_flextool_xlsm` for the full pipeline.
        alternative_name: Name of the alternative to create for this data.
        purge: If True, purge existing data before import.
        strict: Raise :class:`ImportStrictError` on the first parameter value
            the database refuses (otherwise it is collected and reported).

    Returns:
        The counters: what was written, failed writes and the inputs that
        were not translated.
    """
    logger.info("Opening database: %s", db_url)
    _require_import_version(db_url)

    with open_database_mapping(db_url, create=False, upgrade=True) as db:
        if purge:
            logger.info("Purging existing data...")
            _purge_database(db)

        # Tracking sets to avoid duplicate add calls
        entities_added: set[tuple[str, tuple[str, ...]]] = set()
        entity_alts_added: set[tuple[str, tuple[str, ...], str]] = set()
        counters = _Counters(strict=strict)
        counters.unmapped.extend(_collect_unmapped(data))

        # Add the alternative
        try:
            db.add_alternative(name=alternative_name)
        except SpineDBAPIError:
            pass  # May already exist

        # Write in dependency order:
        # 1. Commodities and fuel market nodes
        _write_commodities(data, db, alternative_name, counters,
                           entities_added, entity_alts_added)

        # 2. Balance nodes
        _write_balance_nodes(data, db, alternative_name, counters,
                             entities_added, entity_alts_added)

        # 3. Storage nodes and connections
        _write_storage_units(data, db, alternative_name, counters,
                                             entities_added, entity_alts_added)

        # 4. Units (depends on nodes existing)
        _write_units(data, db, alternative_name, counters,
                     entities_added, entity_alts_added)

        # 5. CF Profiles
        _write_profiles(data, db, alternative_name, counters,
                        entities_added, entity_alts_added)

        # 6. Inflow profiles
        _write_inflow_units(data, db, alternative_name, counters,
                               entities_added, entity_alts_added)

        # 7. Connections
        _write_connections(data, db, alternative_name, counters,
                           entities_added, entity_alts_added)

        # 8. Groups (grid, node, unit, CO2)
        _write_groups(data, db, alternative_name, counters,
                      entities_added, entity_alts_added)

        # 8b. Reserves (depends on groups existing)
        _write_reserves(data, db, alternative_name, counters,
                        entities_added, entity_alts_added)

        # 9. CHP constraints
        _write_chp_constraints(data, db, alternative_name, counters,
                               entities_added, entity_alts_added)

        # 10. Timeline, timeset, solve, model
        _write_timeline_and_solve(data, db, alternative_name, counters,
                                  entities_added, entity_alts_added)

        # 11. Create a scenario that uses our alternative
        db.add_scenario_item(name=alternative_name)
        db.add_scenario_alternative_item(
            scenario_name=alternative_name,
            alternative_name=alternative_name,
            rank=1,
        )
        logger.info("Created scenario '%s'.", alternative_name)

        # Commit everything
        try:
            db.commit_session("Import old FlexTool data")
            logger.info("Successfully committed all data. Summary: %s", counters.summary())
        except NothingToCommit:
            logger.info("No new data to commit.")
        except SpineDBAPIError as exc:
            raise RuntimeError(f"Failed to commit imported data: {exc}") from exc
    return counters


# ---------------------------------------------------------------------------
# Sensitivity import
# ---------------------------------------------------------------------------


def _unknown_sensitivity(
    counters: _Counters,
    scenario: str,
    section: str,
    param: str,
    entity: str | None,
) -> None:
    """Report a sensitivity override the importer does not translate."""
    if _is_cosmetic(_norm_key(param)):
        return
    entities = [f"{entity} [{scenario}]" if entity else f"[{scenario}]"]
    logger.warning("Sensitivity %s: %s '%s' is not translated", scenario,
                   section, param)
    column = _norm_key(param)
    reason = _UNMAPPED.get((section, column))
    if reason is None:
        reason = (
            "the sensitivity import does not translate this column (the base "
            "import may). Set the FlexTool 4 value by hand in alternative "
            f"'{scenario}'."
        )
    counters.unmapped.append(UnmappedItem(
        f"Sensitivity definitions / {section}", param, entities, reason,
    ))


def _build_existing_entity_set(db: DatabaseMapping) -> set[tuple[str, tuple[str, ...]]]:
    """Build a lookup set of (class_name, entity_byname) for all entities in the DB."""
    existing: set[tuple[str, tuple[str, ...]]] = set()
    for item in db.get_entity_items():
        class_name = item["entity_class_name"]
        byname = tuple(item["entity_byname"])
        existing.add((class_name, byname))
    return existing


def _entity_exists(
    existing_entities: set[tuple[str, tuple[str, ...]]],
    class_name: str,
    entity_byname: tuple[str, ...],
) -> bool:
    """Check if an entity exists in the lookup set."""
    return (class_name, entity_byname) in existing_entities


def _get_effective_master_params(
    data: OldFlexToolData,
    overrides: list[SensitivityOverride],
) -> dict[str, float]:
    """Merge base master params with all master overrides for a scenario.

    Returns a dict of all master params with overrides applied on top.
    """
    params = dict(data.master.params)
    for ov in overrides:
        if ov.section == "master" and isinstance(ov.value, (int, float)):
            params[ov.param_name] = float(ov.value)
    return params


def _apply_master_override(
    ov: SensitivityOverride,
    data: OldFlexToolData,
    effective_master: dict[str, float],
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a single master-section sensitivity override."""
    param = ov.param_name.lower().replace("_", " ")
    value = ov.value

    if "co2" in param and "cost" in param:
        # co2_cost -> group.co2_price on the co2_price group
        if _entity_exists(existing_entities, "group", ("co2_price",)):
            _add_param(db, "group", ("co2_price",), "co2_price", value, alt_name, counters)
        else:
            logger.warning("co2_price group not found; skipping co2_cost override.")

    elif "loss of load penalty" in param or "loss_of_load_penalty" in ov.param_name.lower():
        # Set penalty_up and penalty_down on ALL balance nodes
        for gn in data.grid_nodes:
            if _entity_exists(existing_entities, "node", (gn.node,)):
                _add_param(db, "node", (gn.node,), "penalty_up", value, alt_name, counters)
                _add_param(db, "node", (gn.node,), "penalty_down", value, alt_name, counters)

    elif "loss of reserves penalty" in param or "loss_of_reserves_penalty" in ov.param_name.lower():
        # Set penalty_reserve on all reserve groups
        for ng in data.node_groups:
            reserve_method = _get_reserve_method(ng.use_ts_reserve, ng.use_dynamic_reserve)
            if reserve_method is not None:
                rel = ("primary", "up", ng.name)
                if _entity_exists(existing_entities, "reserve__upDown__group", rel):
                    _add_param(db, "reserve__upDown__group", rel,
                               "penalty_reserve", value, alt_name, counters)

    elif "lack of inertia penalty" in param or "lack_of_inertia_penalty" in ov.param_name.lower():
        for ng in data.node_groups:
            if ng.inertia_limit_mws is not None and ng.inertia_limit_mws > 0:
                if _entity_exists(existing_entities, "group", (ng.name,)):
                    _add_param(db, "group", (ng.name,), "penalty_inertia",
                               value, alt_name, counters)

    elif "lack of capacity penalty" in param or "lack_of_capacity_penalty" in ov.param_name.lower():
        for ng in data.node_groups:
            if ng.capacity_margin_mw is not None and ng.capacity_margin_mw > 0:
                if _entity_exists(existing_entities, "group", (ng.name,)):
                    _add_param(db, "group", (ng.name,), "penalty_capacity_margin",
                               value, alt_name, counters)

    elif "use capacity margin" in param or "use_capacity_margin" in ov.param_name.lower():
        enabled = float(value) == 1 if isinstance(value, (int, float)) else False
        for ng in data.node_groups:
            if ng.capacity_margin_mw is not None and ng.capacity_margin_mw > 0:
                if _entity_exists(existing_entities, "group", (ng.name,)):
                    if enabled:
                        _add_param(db, "group", (ng.name,), "has_capacity_margin",
                                   "yes", alt_name, counters)
                        _add_param(db, "group", (ng.name,), "capacity_margin",
                                   ng.capacity_margin_mw, alt_name, counters)
                    else:
                        _add_param(db, "group", (ng.name,), "has_capacity_margin",
                                   "no", alt_name, counters)

    elif "use online" in param or "use_online" in ov.param_name.lower():
        # Affects startup_method on units with startup costs
        enabled = float(value) == 1 if isinstance(value, (int, float)) else False
        for unit in data.units:
            if unit.output_node is None:
                continue
            ut = _get_unit_type(data, unit)
            startup_cost = ut.params.get("startup cost") if ut else None
            if startup_cost is not None and startup_cost > 0:
                unit_name = _get_unit_name(unit)
                if _entity_exists(existing_entities, "unit", (unit_name,)):
                    if enabled:
                        _add_param(db, "unit", (unit_name,), "startup_method",
                                   "linear", alt_name, counters)
                        _add_param(db, "unit", (unit_name,), "startup_cost",
                                   startup_cost, alt_name, counters)
                    else:
                        _add_param(db, "unit", (unit_name,), "startup_method",
                                   "no_startup", alt_name, counters)

    elif "use ramps" in param or "use_ramps" in ov.param_name.lower():
        enabled = float(value) == 1 if isinstance(value, (int, float)) else False
        for unit in data.units:
            if unit.output_node is None:
                continue
            ut = _get_unit_type(data, unit)
            ramp_up = ut.params.get("ramp up (p.u. per min)") if ut else None
            ramp_down = ut.params.get("ramp down (p.u. per min)") if ut else None
            if (ramp_up and ramp_up > 0) or (ramp_down and ramp_down > 0):
                unit_name = _get_unit_name(unit)
                rel = (unit_name, unit.output_node)
                if _entity_exists(existing_entities, "unit__outputNode", rel):
                    if enabled:
                        _add_param(db, "unit__outputNode", rel,
                                   "ramp_method", "ramp_limit", alt_name, counters)
                    else:
                        _add_param(db, "unit__outputNode", rel,
                                   "ramp_method", "none", alt_name, counters)

    elif "use non synchronous" in param or "use_non_synchronous" in ov.param_name.lower():
        enabled = float(value) == 1 if isinstance(value, (int, float)) else False
        for ng in data.node_groups:
            if ng.non_synchronous_share is not None and ng.non_synchronous_share > 0:
                if _entity_exists(existing_entities, "group", (ng.name,)):
                    if enabled:
                        _add_param(db, "group", (ng.name,), "has_non_synchronous",
                                   "yes", alt_name, counters)
                        _add_param(db, "group", (ng.name,), "non_synchronous_limit",
                                   ng.non_synchronous_share, alt_name, counters)
                    else:
                        _add_param(db, "group", (ng.name,), "has_non_synchronous",
                                   "no", alt_name, counters)

    elif "use inertia limit" in param or "use_inertia_limit" in ov.param_name.lower():
        enabled = float(value) == 1 if isinstance(value, (int, float)) else False
        for ng in data.node_groups:
            if ng.inertia_limit_mws is not None and ng.inertia_limit_mws > 0:
                if _entity_exists(existing_entities, "group", (ng.name,)):
                    if enabled:
                        _add_param(db, "group", (ng.name,), "has_inertia",
                                   "yes", alt_name, counters)
                        _add_param(db, "group", (ng.name,), "inertia_limit",
                                   ng.inertia_limit_mws, alt_name, counters)
                    else:
                        _add_param(db, "group", (ng.name,), "has_inertia",
                                   "no", alt_name, counters)

    elif "mode invest" in param or "mode dispatch" in param:
        # Handled separately in _apply_mode_overrides
        pass

    else:
        _unknown_sensitivity(counters, alt_name, "master", ov.param_name, None)


def _apply_mode_overrides(
    effective_master: dict[str, float],
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply mode_invest / mode_dispatch overrides by computing effective mode."""
    mode_invest = False
    mode_dispatch = False
    for key, val in effective_master.items():
        key_lower = key.lower().replace("_", " ")
        if "mode invest" in key_lower and val == 1:
            mode_invest = True
        if "mode dispatch" in key_lower and val == 1:
            mode_dispatch = True

    if mode_invest and mode_dispatch:
        model_solves = Array(["invest"])
        if _entity_exists(existing_entities, "model", ("flexTool",)):
            _add_param(db, "model", ("flexTool",), "solves", model_solves, alt_name, counters)
        if _entity_exists(existing_entities, "solve", ("invest",)):
            _add_param(db, "solve", ("invest",), "contains_solves", "dispatch",
                       alt_name, counters)
    elif mode_invest:
        model_solves = Array(["invest"])
        if _entity_exists(existing_entities, "model", ("flexTool",)):
            _add_param(db, "model", ("flexTool",), "solves", model_solves, alt_name, counters)
    elif mode_dispatch:
        model_solves = Array(["dispatch"])
        if _entity_exists(existing_entities, "model", ("flexTool",)):
            _add_param(db, "model", ("flexTool",), "solves", model_solves, alt_name, counters)


def _apply_node_group_override(
    ov: SensitivityOverride,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a nodeGroup-section sensitivity override."""
    group_name = ov.entity_ids.get("nodeGroup")
    if not group_name:
        logger.warning("nodeGroup override missing 'nodeGroup' identifier; skipping.")
        return

    if not _entity_exists(existing_entities, "group", (group_name,)):
        logger.warning("Group '%s' not found in DB; skipping nodeGroup override.", group_name)
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    if "capacity margin" in param_lower:
        _add_param(db, "group", (group_name,), "capacity_margin", value, alt_name, counters)
        _add_param(db, "group", (group_name,), "has_capacity_margin", "yes", alt_name, counters)
    elif "non synchronous" in param_lower:
        _add_param(db, "group", (group_name,), "non_synchronous_limit", value, alt_name, counters)
        _add_param(db, "group", (group_name,), "has_non_synchronous", "yes", alt_name, counters)
    elif "inertia limit" in param_lower:
        _add_param(db, "group", (group_name,), "inertia_limit", value, alt_name, counters)
        _add_param(db, "group", (group_name,), "has_inertia", "yes", alt_name, counters)
    elif "use ts_reserve" in param_lower or "use dynamic reserve" in param_lower:
        # Need both flags to determine reserve method; just set the one we have
        # The reserve_method logic needs both flags, so we compute from the override
        use_ts = None
        use_dyn = None
        if "ts_reserve" in param_lower:
            use_ts = float(value) if isinstance(value, (int, float)) else None
        if "dynamic reserve" in param_lower:
            use_dyn = float(value) if isinstance(value, (int, float)) else None
        reserve_method = _get_reserve_method(use_ts, use_dyn)
        if reserve_method is not None:
            rel = ("primary", "up", group_name)
            if _entity_exists(existing_entities, "reserve__upDown__group", rel):
                _add_param(db, "reserve__upDown__group", rel,
                           "reserve_method", reserve_method, alt_name, counters)
    else:
        _unknown_sensitivity(counters, alt_name, "nodeGroup", ov.param_name,
                             ov.entity_ids.get("nodeGroup"))


def _apply_unit_type_override(
    ov: SensitivityOverride,
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a unit_type-section override to ALL units of that type."""
    type_name = ov.entity_ids.get("unit type")
    if not type_name:
        logger.warning("unit_type override missing 'unit type' identifier; skipping.")
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    # Find all units with this unit_type
    matching_units = [u for u in data.units if u.unit_type == type_name and u.output_node]

    if not matching_units:
        logger.warning("No units found for unit_type '%s'; skipping override.", type_name)
        return

    for unit in matching_units:
        unit_name = _get_unit_name(unit)
        is_storage = _needs_charger_discharger(unit, data)

        if is_storage:
            charger = f"{unit_name}_charger"
            discharger = f"{unit_name}_discharger"
            if not _entity_exists(existing_entities, "unit", (discharger,)):
                logger.warning("Unit '%s' (discharger) not found in DB; skipping.", discharger)
                continue
        else:
            if not _entity_exists(existing_entities, "unit", (unit_name,)):
                logger.warning("Unit '%s' not found in DB; skipping.", unit_name)
                continue

        if "efficiency" == param_lower:
            if is_storage:
                _add_param(db, "unit", (discharger,), "efficiency", value, alt_name, counters)
            else:
                _add_param(db, "unit", (unit_name,), "efficiency", value, alt_name, counters)
        elif "eff charge" in param_lower:
            if is_storage:
                _add_param(db, "unit", (charger,), "efficiency", value, alt_name, counters)
        elif "eff at min load" in param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "efficiency_at_min_load", value, alt_name, counters)
        elif "min load" == param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "min_load", value, alt_name, counters)
        elif "availability" == param_lower:
            # Capacity margin only in FT2; not FT4 unit.availability.
            counters.unmapped.append(UnmappedItem(
                "Sensitivity definitions / unit_type", ov.param_name,
                [f"{unit_name} [{alt_name}]"],
                _UNMAPPED[("unit_type", "availability")],
            ))
        elif "inv.cost/kwh" in param_lower:
            # The storage cost: on the storage node, as in the base import.
            if is_storage:
                _add_param(db, "node", (f"{unit_name}_storage",), "invest_cost",
                           value, alt_name, counters)
            else:
                _unknown_sensitivity(counters, alt_name, "unit_type",
                                     ov.param_name, unit_name)
        elif "inv.cost/kw" in param_lower or "inv cost" in param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "invest_cost", value, alt_name, counters)
        elif "fixed cost" in param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "fixed_cost", value, alt_name, counters)
        elif "lifetime" == param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "lifetime", value, alt_name, counters)
        elif "interest" == param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "discount_rate", value, alt_name, counters)
        elif "startup cost" in param_lower:
            target = discharger if is_storage else unit_name
            _add_param(db, "unit", (target,), "startup_cost", value, alt_name, counters)
        elif "o&m cost" in param_lower:
            if is_storage:
                rel = (discharger, unit.output_node)
            else:
                rel = (unit_name, unit.output_node)
            if _entity_exists(existing_entities, "unit__outputNode", rel):
                _add_param(db, "unit__outputNode", rel,
                           "other_operational_cost", value, alt_name, counters)
        elif "ramp up" in param_lower:
            if is_storage:
                for sub, node in ((charger, f"{unit_name}_storage"), (discharger, unit.output_node)):
                    rel = (sub, node)
                    if _entity_exists(existing_entities, "unit__outputNode", rel):
                        _add_param(db, "unit__outputNode", rel,
                                   "ramp_speed_up", value, alt_name, counters)
            else:
                rel = (unit_name, unit.output_node)
                if _entity_exists(existing_entities, "unit__outputNode", rel):
                    _add_param(db, "unit__outputNode", rel,
                               "ramp_speed_up", value, alt_name, counters)
        elif "ramp down" in param_lower:
            if is_storage:
                for sub, node in ((charger, f"{unit_name}_storage"), (discharger, unit.output_node)):
                    rel = (sub, node)
                    if _entity_exists(existing_entities, "unit__outputNode", rel):
                        _add_param(db, "unit__outputNode", rel,
                                   "ramp_speed_down", value, alt_name, counters)
            else:
                rel = (unit_name, unit.output_node)
                if _entity_exists(existing_entities, "unit__outputNode", rel):
                    _add_param(db, "unit__outputNode", rel,
                               "ramp_speed_down", value, alt_name, counters)
        elif "self discharge" in param_lower:
            storage_node = f"{unit_name}_storage"
            if _entity_exists(existing_entities, "node", (storage_node,)):
                _add_param(db, "node", (storage_node,), "self_discharge_loss",
                           value, alt_name, counters)
        elif "non synchronous" in param_lower:
            is_ns = "yes" if (isinstance(value, (int, float)) and float(value) == 1) else "no"
            if is_storage:
                for sub, node in ((charger, unit.output_node), (discharger, unit.output_node)):
                    rel = (sub, node)
                    if _entity_exists(existing_entities, "unit__outputNode", rel):
                        _add_param(db, "unit__outputNode", rel,
                                   "is_non_synchronous", is_ns, alt_name, counters)
            else:
                rel = (unit_name, unit.output_node)
                if _entity_exists(existing_entities, "unit__outputNode", rel):
                    _add_param(db, "unit__outputNode", rel,
                               "is_non_synchronous", is_ns, alt_name, counters)
        elif "inertia constant" in param_lower:
            if is_storage:
                for sub, node in ((charger, unit.output_node), (discharger, unit.output_node)):
                    rel = (sub, node)
                    if _entity_exists(existing_entities, "unit__outputNode", rel):
                        _add_param(db, "unit__outputNode", rel,
                                   "inertia_constant", value, alt_name, counters)
            else:
                rel = (unit_name, unit.output_node)
                if _entity_exists(existing_entities, "unit__outputNode", rel):
                    _add_param(db, "unit__outputNode", rel,
                               "inertia_constant", value, alt_name, counters)
        else:
            _unknown_sensitivity(counters, alt_name, "unit_type", ov.param_name,
                                 type_name)


def _apply_fuel_override(
    ov: SensitivityOverride,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a fuel-section sensitivity override."""
    fuel_name = ov.entity_ids.get("fuel")
    if not fuel_name:
        logger.warning("fuel override missing 'fuel' identifier; skipping.")
        return

    if not _entity_exists(existing_entities, "commodity", (fuel_name,)):
        logger.warning("Commodity '%s' not found in DB; skipping fuel override.", fuel_name)
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    if "price" in param_lower:
        _add_param(db, "commodity", (fuel_name,), "price", value, alt_name, counters)
    elif "co2" in param_lower:
        _add_param(db, "commodity", (fuel_name,), "co2_content", value, alt_name, counters)
    else:
        _unknown_sensitivity(counters, alt_name, "fuel", ov.param_name,
                             ov.entity_ids.get("fuel"))


def _apply_unit_group_override(
    ov: SensitivityOverride,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a unitGroup-section sensitivity override."""
    group_name = ov.entity_ids.get("unitGroup")
    if not group_name:
        logger.warning("unitGroup override missing 'unitGroup' identifier; skipping.")
        return

    if not _entity_exists(existing_entities, "group", (group_name,)):
        logger.warning("Group '%s' not found in DB; skipping unitGroup override.", group_name)
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    if "max invest" in param_lower:
        _add_param(db, "group", (group_name,), "invest_max_total", value, alt_name, counters)
        if isinstance(value, (int, float)) and float(value) > 0:
            _add_param(db, "group", (group_name,), "invest_method", "invest_total",
                       alt_name, counters)
    elif "min invest" in param_lower:
        _add_param(db, "group", (group_name,), "invest_min_total", value, alt_name, counters)
    else:
        _unknown_sensitivity(counters, alt_name, "unitGroup", ov.param_name,
                             ov.entity_ids.get("unitGroup"))



def _units_override_kind(param_lower: str) -> str | None:
    """The capacity or investment-limit column a units override sets (the
    order of the :func:`_apply_units_override` branches), else ``None``."""
    if param_lower in ("capacity (mw)", "capacity"):
        return "capacity"
    if "invested capacity" in param_lower:
        return "invested_mw"
    if "max invest (mw)" in param_lower and "mwh" not in param_lower:
        return "max_invest_mw"
    if "invested storage" in param_lower:
        return "invested_mwh"
    if "max invest (mwh)" in param_lower:
        return "max_invest_mwh"
    if "storage (mwh)" in param_lower or (
        "storage" in param_lower and "invest" not in param_lower
        and "start" not in param_lower and "finish" not in param_lower
    ):
        return "storage_mwh"
    return None


def _scenario_override(
    overrides: list[SensitivityOverride], ids: dict[str, str | None],
    kind: str, base: float | None,
) -> tuple[float | None, bool]:
    """(value, overridden) of a units capacity column in one scenario: its
    override for the unit with ``ids`` (unittype, output node), else ``base``."""
    for other in reversed(overrides):
        if (other.section == "units"
                and all(other.entity_ids.get(k) == v for k, v in ids.items())
                and _units_override_kind(other.param_name.lower()) == kind
                and isinstance(other.value, (int, float))):
            return float(other.value), True
    return base, False


def _apply_units_override(
    ov: SensitivityOverride,
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
    scenario_overrides: list[SensitivityOverride] | None = None,
) -> None:
    """Apply a units-section sensitivity override.

    ``capacity`` and ``invested capacity`` (``storage`` and ``invested
    storage``) are one FT4 ``existing`` (:func:`_ft2_capacity`): each takes
    the other from the scenario's own override, else from the base workbook,
    and the pair is written once.
    """
    unit_type = ov.entity_ids.get("unittype")
    output_node = ov.entity_ids.get("output node")

    if not unit_type or not output_node:
        logger.warning(
            "units override missing 'unittype' or 'output node'; skipping. IDs: %s",
            ov.entity_ids,
        )
        return

    unit_name = f"{unit_type}_{output_node}"

    # Identify the base unit and whether it's a pure storage pattern
    base_unit = next(
        (u for u in data.units
         if u.unit_type == unit_type and u.output_node == output_node),
        None,
    )
    is_storage = base_unit is not None and _needs_charger_discharger(base_unit, data)

    if is_storage:
        charger = f"{unit_name}_charger"
        discharger = f"{unit_name}_discharger"
        storage_node = f"{unit_name}_storage"
        if not _entity_exists(existing_entities, "unit", (discharger,)):
            logger.warning("Unit '%s' (discharger) not found in DB; skipping.", discharger)
            return
    else:
        if not _entity_exists(existing_entities, "unit", (unit_name,)):
            logger.warning("Unit '%s' not found in DB; skipping units override.", unit_name)
            return

    param_lower = ov.param_name.lower()
    value = ov.value
    ids = {"unittype": unit_type, "output node": output_node}
    overrides = scenario_overrides if scenario_overrides is not None else [ov]

    def capacity_pair(kind: str, other: str, base_other: float | None) -> float | None:
        """FT4 existing for a capacity / invested capacity override; None when
        the scenario's override of ``other`` writes it instead (it is later
        in the order of the branches: the capacity branch writes)."""
        other_value, other_overridden = _scenario_override(overrides, ids, other, base_other)
        if other_overridden and kind in ("invested_mw", "invested_mwh"):
            return None
        mine = float(value) if isinstance(value, (int, float)) else None
        return _ft2_capacity(mine, other_value)

    def base_of(attr: str) -> float | None:
        return None if base_unit is None else getattr(base_unit, attr)

    ratio = (None if base_unit is None
             else _get_unit_type_param(data, base_unit, "fixed kw/kwh ratio"))
    has_ratio = ratio is not None and ratio > 0

    def max_invest() -> tuple[float, float, bool, bool]:
        """The scenario's (MW, MWh) investment limits (_storage_max_invest)
        and whether the scenario overrides each ``max invest``."""
        mw, mw_ov = _scenario_override(overrides, ids, "max_invest_mw",
                                       base_of("max_invest_mw"))
        mwh, mwh_ov = _scenario_override(overrides, ids, "max_invest_mwh",
                                         base_of("max_invest_mwh"))
        max_mw, max_mwh = _storage_max_invest(mw, mwh, ratio)
        return max_mw, max_mwh, mw_ov, mwh_ov

    def write_invest_limit(class_name: str, targets: tuple[str, ...],
                           limit: float) -> None:
        for target in targets:
            _add_param(db, class_name, (target,), "invest_max_total", limit,
                       alt_name, counters)
            if limit > 0:
                _add_param(db, class_name, (target,), "invest_method",
                           "invest_total", alt_name, counters)

    def write_node_type(storage_target: str) -> None:
        """``storage`` while the node has capacity or can invest in it, else
        ``balance`` (no state variable).  Written once per scenario, by the
        last of its storage / invested storage / max invest (MWh) overrides."""
        last = None
        for other in overrides:
            if (other.section == "units"
                    and all(other.entity_ids.get(k) == v for k, v in ids.items())
                    and _units_override_kind(other.param_name.lower())
                    in ("storage_mwh", "invested_mwh", "max_invest_mwh")):
                last = other
        if last is not ov:
            return
        storage, _ = _scenario_override(overrides, ids, "storage_mwh",
                                        base_of("storage_mwh"))
        invested, _ = _scenario_override(overrides, ids, "invested_mwh",
                                         base_of("invested_storage_mwh"))
        _, max_mwh, _, _ = max_invest()
        has_storage = (_ft2_capacity(storage, invested) or 0.0) > 0 or max_mwh > 0
        _add_param(db, "node", (storage_target,), "node_type",
                   "storage" if has_storage else "balance", alt_name, counters)

    if "capacity (mw)" == param_lower or param_lower == "capacity" or (
        "invested capacity" in param_lower
    ):
        kind = "capacity" if "invested" not in param_lower else "invested_mw"
        other = "invested_mw" if kind == "capacity" else "capacity"
        base_other = (None if base_unit is None else
                      base_unit.invested_capacity_mw if kind == "capacity"
                      else base_unit.capacity_mw)
        existing = capacity_pair(kind, other, base_other)
        if existing is not None:
            for target in ((charger, discharger) if is_storage else (unit_name,)):
                _add_param(db, "unit", (target,), "existing", existing, alt_name, counters)
    elif "max invest (mw)" in param_lower and "mwh" not in param_lower:
        if is_storage:
            # FT2 derives the MWh limit from this one (kW/kWh ratio) when the
            # scenario and the base give no 'max invest (MWh)'.
            max_mw, max_mwh, _, mwh_ov = max_invest()
            write_invest_limit("unit", (charger, discharger), max_mw)
            if has_ratio and not mwh_ov and _entity_exists(
                    existing_entities, "node", (storage_node,)):
                write_invest_limit("node", (storage_node,), max_mwh)
        else:
            _add_param(db, "unit", (unit_name,), "invest_max_total", value, alt_name, counters)
            if isinstance(value, (int, float)) and float(value) > 0:
                _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                           alt_name, counters)
    elif "invested storage" in param_lower:
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        existing = capacity_pair(
            "invested_mwh", "storage_mwh",
            None if base_unit is None else base_unit.storage_mwh)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            if existing is not None:
                _add_param(db, "node", (storage_target,), "existing", existing,
                           alt_name, counters)
            write_node_type(storage_target)
    elif "max invest (mwh)" in param_lower:
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            max_mw, max_mwh, mw_ov, _ = max_invest()
            write_invest_limit("node", (storage_target,), max_mwh)
            write_node_type(storage_target)
            # FT2 derives the MW limit from this one (kW/kWh ratio) when the
            # scenario and the base give no 'max invest (MW)'.
            if is_storage and has_ratio and not mw_ov:
                write_invest_limit("unit", (charger, discharger), max_mw)
    elif "storage (mwh)" in param_lower or (
        "storage" in param_lower and "invest" not in param_lower and "start" not in param_lower and "finish" not in param_lower
    ):
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        existing = capacity_pair(
            "storage_mwh", "invested_mwh",
            None if base_unit is None else base_unit.invested_storage_mwh)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            _add_param(db, "node", (storage_target,), "existing", existing,
                       alt_name, counters)
            write_node_type(storage_target)
    elif "inv.cost/kw" in param_lower or "inv cost" in param_lower:
        target = discharger if is_storage else unit_name
        _add_param(db, "unit", (target,), "invest_cost", value, alt_name, counters)
    elif "efficiency" == param_lower:
        target = discharger if is_storage else unit_name
        _add_param(db, "unit", (target,), "efficiency", value, alt_name, counters)
    elif "eff charge" in param_lower:
        if is_storage:
            _add_param(db, "unit", (charger,), "efficiency", value, alt_name, counters)
    elif "min load" in param_lower:
        target = discharger if is_storage else unit_name
        _add_param(db, "unit", (target,), "min_load", value, alt_name, counters)
    elif "inflow multiplier" in param_lower:
        # Re-read the original inflow profile, apply the new multiplier,
        # and write to the unit's inflow node under this alternative
        multiplier = float(value) if value else 0.0
        if base_unit and base_unit.inflow_profile:
            inflow_ts = _find_inflow_profile(data, base_unit.inflow_profile)
            if inflow_ts and inflow_ts.data:
                scaled = {k: v * multiplier for k, v in inflow_ts.data.items()}
                inflow_map = _make_time_map(scaled)
                inflow_node = _get_unit_node_name(base_unit)
                if _entity_exists(existing_entities, "node", (inflow_node,)):
                    _add_param(db, "node", (inflow_node,), "inflow",
                               inflow_map, alt_name, counters)
                else:
                    logger.warning(
                        "Inflow node '%s' not found; cannot apply inflow multiplier.",
                        inflow_node,
                    )
            else:
                logger.warning(
                    "Inflow profile '%s' not found for unit '%s'; cannot apply multiplier.",
                    base_unit.inflow_profile, unit_name,
                )
        else:
            logger.warning(
                "No inflow profile for unit '%s'; cannot apply inflow multiplier.",
                unit_name,
            )
    else:
        _unknown_sensitivity(counters, alt_name, "units", ov.param_name,
                             unit_name)


def _apply_grid_node_override(
    ov: SensitivityOverride,
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
) -> None:
    """Apply a gridNode-section sensitivity override."""
    node_name = ov.entity_ids.get("node")
    if not node_name:
        logger.warning("gridNode override missing 'node' identifier; skipping.")
        return

    if not _entity_exists(existing_entities, "node", (node_name,)):
        logger.warning("Node '%s' not found in DB; skipping gridNode override.", node_name)
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    if "demand" in param_lower:
        _add_param(db, "node", (node_name,), "annual_flow", value, alt_name, counters)
    elif "capacity margin" in param_lower:
        # This applies to the node's group, find the group
        for gn in data.grid_nodes:
            if gn.node == node_name:
                for ng_name in gn.node_groups:
                    if _entity_exists(existing_entities, "group", (ng_name,)):
                        _add_param(db, "group", (ng_name,), "capacity_margin",
                                   value, alt_name, counters)
                        _add_param(db, "group", (ng_name,), "has_capacity_margin",
                                   "yes", alt_name, counters)
    elif "non synchronous" in param_lower:
        for gn in data.grid_nodes:
            if gn.node == node_name:
                for ng_name in gn.node_groups:
                    if _entity_exists(existing_entities, "group", (ng_name,)):
                        _add_param(db, "group", (ng_name,), "non_synchronous_limit",
                                   value, alt_name, counters)
                        _add_param(db, "group", (ng_name,), "has_non_synchronous",
                                   "yes", alt_name, counters)
    elif "use ts_reserve" in param_lower or "use dynamic reserve" in param_lower:
        use_ts = None
        use_dyn = None
        if "ts_reserve" in param_lower:
            use_ts = float(value) if isinstance(value, (int, float)) else None
        if "dynamic reserve" in param_lower:
            use_dyn = float(value) if isinstance(value, (int, float)) else None
        reserve_method = _get_reserve_method(use_ts, use_dyn)
        if reserve_method is not None:
            # Find the reserve group for this node
            for gn in data.grid_nodes:
                if gn.node == node_name:
                    for ng_name in gn.node_groups:
                        rel = ("primary", "up", ng_name)
                        if _entity_exists(existing_entities, "reserve__upDown__group", rel):
                            _add_param(db, "reserve__upDown__group", rel,
                                       "reserve_method", reserve_method, alt_name, counters)
    else:
        _unknown_sensitivity(counters, alt_name, "gridNode", ov.param_name,
                             ov.entity_ids.get("node"))


def _apply_node_node_override(
    ov: SensitivityOverride,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
    data: OldFlexToolData | None = None,
    overrides: list[SensitivityOverride] | None = None,
) -> None:
    """Apply a nodeNode-section sensitivity override."""
    node1 = ov.entity_ids.get("node1")
    node2 = ov.entity_ids.get("node2")

    if not node1 or not node2:
        logger.warning("nodeNode override missing 'node1' or 'node2'; skipping.")
        return

    conn_name = f"{node1}_{node2}"
    if not _entity_exists(existing_entities, "connection", (conn_name,)):
        logger.warning("Connection '%s' not found in DB; skipping nodeNode override.", conn_name)
        return

    param_lower = ov.param_name.lower()
    value = ov.value

    if "invested capacity" in param_lower or "cap.rightward" in param_lower or (
        "cap.leftward" in param_lower
    ):
        # One FT4 existing: the larger direction + invested capacity
        # (_ft2_capacity), each from the scenario's override, else the base.
        base = next((c for c in (data.connections if data else [])
                     if c.node1 == node1 and c.node2 == node2), None)
        values = {"cap.rightward": None if base is None else base.cap_rightward_mw,
                  "cap.leftward": None if base is None else base.cap_leftward_mw,
                  "invested capacity": None if base is None else base.invested_capacity_mw}
        last = None
        for other in overrides or [ov]:
            if other.section != "nodeNode" or other.entity_ids.get("node1") != node1 \
                    or other.entity_ids.get("node2") != node2:
                continue
            for key in values:
                if key in other.param_name.lower() and isinstance(other.value, (int, float)):
                    values[key] = float(other.value)
                    last = other
        if last is ov or last is None:
            directions = [v for v in (values["cap.rightward"], values["cap.leftward"])
                          if v is not None]
            _add_param(db, "connection", (conn_name,), "existing",
                       _ft2_capacity(max(directions) if directions else None,
                                     values["invested capacity"]),
                       alt_name, counters)
    elif "max invest" in param_lower:
        _add_param(db, "connection", (conn_name,), "invest_max_total", value, alt_name, counters)
        if isinstance(value, (int, float)) and float(value) > 0:
            _add_param(db, "connection", (conn_name,), "invest_method", "invest_total",
                       alt_name, counters)
    elif "loss" == param_lower:
        eff = 1.0 - float(value) if isinstance(value, (int, float)) else 1.0
        _add_param(db, "connection", (conn_name,), "efficiency", eff, alt_name, counters)
    elif "inv.cost/kw" in param_lower or "inv cost" in param_lower:
        _add_param(db, "connection", (conn_name,), "invest_cost", value, alt_name, counters)
    elif "lifetime" == param_lower:
        _add_param(db, "connection", (conn_name,), "lifetime", value, alt_name, counters)
    elif "interest" == param_lower:
        _add_param(db, "connection", (conn_name,), "discount_rate", value, alt_name, counters)
    else:
        _unknown_sensitivity(counters, alt_name, "nodeNode", ov.param_name,
                             conn_name)


def write_sensitivities_to_db(
    sensitivities: dict[str, list[SensitivityOverride]],
    data: OldFlexToolData,
    db_url: str,
    base_alternative: str = "base",
    *,
    strict: bool = False,
) -> _Counters:
    """Write sensitivity alternatives and scenarios on top of an already-imported DB.

    This does NOT purge the database. It layers sensitivity alternatives on top
    of the base import.  The database must still be at the import schema
    version (56): the overrides use v56 names and values, and only migration
    may carry them forward (:func:`import_old_flextool_xlsm` with ``master=``
    does both in order).

    Args:
        sensitivities: Dict mapping scenario_name -> list of SensitivityOverride.
        data: The base OldFlexToolData (needed for unit_type->unit mapping, etc.).
        db_url: Spine database URL.
        base_alternative: Name of the base alternative already in the DB.
        strict: Raise on the first parameter value the database refuses.

    Returns:
        The counters (written, failed writes, untranslated overrides).

    Raises:
        ValueError: the database is not at the import schema version.
    """
    counters = _Counters(strict=strict)
    _require_import_version(db_url)
    if not sensitivities:
        logger.info("No sensitivities to write.")
        return counters

    logger.info(
        "Writing %d sensitivity scenarios to: %s",
        len(sensitivities), db_url,
    )

    with open_database_mapping(db_url, create=False, upgrade=True) as db:
        entity_alts_added: set[tuple[str, tuple[str, ...], str]] = set()

        # Build entity existence lookup from DB
        existing_entities = _build_existing_entity_set(db)

        # Ensure base alternative exists
        try:
            db.add_alternative(name=base_alternative)
        except SpineDBAPIError:
            pass

        # Create a "base" scenario with just the base alternative
        try:
            db.add_scenario_item(name="base")
        except SpineDBAPIError:
            pass
        try:
            db.add_scenario_alternative_item(
                scenario_name="base",
                alternative_name=base_alternative,
                rank=1,
            )
        except SpineDBAPIError:
            pass

        for scenario_name, overrides in sensitivities.items():
            logger.info("Processing sensitivity scenario: '%s' (%d overrides)",
                        scenario_name, len(overrides))

            # 1. Create alternative for this scenario
            try:
                db.add_alternative(name=scenario_name)
            except SpineDBAPIError:
                pass

            # 2. Compute effective master params (base + all master overrides)
            effective_master = _get_effective_master_params(data, overrides)

            # 3. Check if mode overrides are present
            has_mode_override = any(
                ov.section == "master"
                and ("mode invest" in ov.param_name.lower().replace("_", " ")
                     or "mode dispatch" in ov.param_name.lower().replace("_", " "))
                for ov in overrides
            )

            # 4. Process each override
            for ov in overrides:
                if ov.section == "master":
                    _apply_master_override(
                        ov, data, effective_master, db, scenario_name,
                        counters, existing_entities,
                    )
                elif ov.section == "nodeGroup":
                    _apply_node_group_override(
                        ov, db, scenario_name, counters, existing_entities,
                    )
                elif ov.section == "gridNode":
                    _apply_grid_node_override(
                        ov, data, db, scenario_name, counters, existing_entities,
                    )
                elif ov.section == "unit_type":
                    _apply_unit_type_override(
                        ov, data, db, scenario_name, counters, existing_entities,
                    )
                elif ov.section == "fuel":
                    _apply_fuel_override(
                        ov, db, scenario_name, counters, existing_entities,
                    )
                elif ov.section == "unitGroup":
                    _apply_unit_group_override(
                        ov, db, scenario_name, counters, existing_entities,
                    )
                elif ov.section == "units":
                    _apply_units_override(
                        ov, data, db, scenario_name, counters, existing_entities,
                        entity_alts_added, overrides,
                    )
                elif ov.section == "nodeNode":
                    _apply_node_node_override(
                        ov, db, scenario_name, counters, existing_entities,
                        data, overrides,
                    )
                else:
                    _unknown_sensitivity(counters, scenario_name, ov.section,
                                         ov.param_name, None)

            # 5. Apply mode overrides if present
            if has_mode_override:
                _apply_mode_overrides(
                    effective_master, db, scenario_name, counters, existing_entities,
                )

            # 6. Create scenario with base + sensitivity alternatives
            try:
                db.add_scenario_item(name=scenario_name)
            except SpineDBAPIError:
                pass
            try:
                db.add_scenario_alternative_item(
                    scenario_name=scenario_name,
                    alternative_name=base_alternative,
                    rank=1,
                )
            except SpineDBAPIError:
                pass
            try:
                db.add_scenario_alternative_item(
                    scenario_name=scenario_name,
                    alternative_name=scenario_name,
                    rank=2,
                )
            except SpineDBAPIError:
                pass

            logger.info("Created scenario '%s' with alternatives: [%s, %s]",
                        scenario_name, base_alternative, scenario_name)

        counters.unmapped.extend(
            _sensitivity_invested_capacity_items(sensitivities, data))

        # Commit everything
        try:
            db.commit_session("Import sensitivity overrides")
            logger.info(
                "Successfully committed sensitivity data. Summary: %s",
                counters.summary(),
            )
        except NothingToCommit:
            logger.info("No new sensitivity data to commit.")
        except SpineDBAPIError as exc:
            raise RuntimeError(
                f"Failed to commit sensitivity data: {exc}"
            ) from exc
    return counters


# ---------------------------------------------------------------------------
# Full import pipeline: frozen-template init → write → migrate
# ---------------------------------------------------------------------------
#
# The FlexTool-2 importer hand-writes a *fixed* schema version — the one that
# was current when it was written (v56; see ``OLD_FLEXTOOL_IMPORT_SCHEMA``,
# whose ``model.version`` default is 56).  It is deliberately NOT kept in step
# with the live schema: the DB is initialised from the frozen template, the
# importer writes its v56 data, and ``migrate_database`` then carries the
# result forward to whatever the current ``FLEXTOOL_DB_VERSION`` is.
#
# This is what removes the maintenance burden — a future schema rename is
# handled by its migration step (``db_migration.py``), not by editing this
# importer.  Initialising from the *live* schema instead (the old behaviour)
# would silently drop every value whose name a later rename had changed,
# because ``_add_param`` skips unknown parameters.  The frozen template +
# always-migrate keeps the importer correct without touching it again.

#: Frozen schema the importer targets.  Snapshot of ``spinedb_schema.json``
#: taken at DB version 56; do NOT update it to track the live schema — that is
#: precisely what migration exists to do.
OLD_FLEXTOOL_IMPORT_SCHEMA = "schemas/old_flextool_import_template.json"
OLD_FLEXTOOL_IMPORT_SCHEMA_VERSION = 56


def _sqlite_path_from_url(db_url: str) -> str | None:
    """Return the filesystem path of a ``sqlite:///`` URL, else ``None``.

    ``sqlite:///rel.sqlite`` → ``rel.sqlite`` (relative); the absolute form
    ``sqlite:////abs/path.sqlite`` → ``/abs/path.sqlite`` (the 4th slash is
    the leading slash of the path).
    """
    from flextool.sqlite_url import SQLITE_PREFIX, sqlite_file_path

    if db_url.startswith(SQLITE_PREFIX):
        path = sqlite_file_path(db_url)
        return None if path is None else str(path)
    return None


#: :func:`_db_version` of a database that has data but no FlexTool schema.
_NOT_FLEXTOOL = -1


def _db_version(db_url: str) -> int | None:
    """``model.version`` of a FlexTool DB; ``None`` for a new (missing or
    empty) database; :data:`_NOT_FLEXTOOL` for a database with entity
    classes but no FlexTool version.

    A file that cannot be opened as a database is a ``ValueError``: the
    importer must not initialise over (and on failure delete) a user's file.
    """
    path = _sqlite_path_from_url(db_url)
    if path is not None:
        from pathlib import Path

        if not Path(path).exists() or Path(path).stat().st_size == 0:
            return None
    try:
        with open_database_mapping(db_url) as db:
            pdef = db.get_parameter_definition_item(
                entity_class_name="model", name="version"
            )
            if pdef:
                return int(api.from_database(pdef["default_value"],
                                             pdef["default_type"]))
            return _NOT_FLEXTOOL if db.get_items("entity_class") else None
    except Exception as exc:  # noqa: BLE001 - any open failure is reported
        raise ValueError(
            f"Could not open {db_url} as a database ({exc}). Give a new file "
            "name for the import."
        ) from exc


def _require_import_version(db_url: str) -> None:
    """Refuse a FlexTool DB that is not at the import schema version (56),
    and a database that is not a FlexTool database.

    The importer writes v56 names and values; only ``migrate_database`` may
    carry them to a later version (it renames parameters *and* rewrites
    values, e.g. v65 ``has_capacity_margin`` → ``capacity_margin_method``,
    v77 ``not_allowed`` → ``not_active``).  Writing v56 data into a later
    database would lose every renamed value with only a log line, so it is a
    hard error.
    """
    version = _db_version(db_url)
    if version is None or version == OLD_FLEXTOOL_IMPORT_SCHEMA_VERSION:
        return
    if version == _NOT_FLEXTOOL:
        raise ValueError(
            f"{db_url} is a database with data but not a FlexTool database, "
            "so the FlexTool 2 import will not write into it. Give a new file "
            "name for the import."
        )
    raise ValueError(
        f"{db_url} is a FlexTool database at version {version}. The FlexTool 2 "
        f"importer writes version {OLD_FLEXTOOL_IMPORT_SCHEMA_VERSION} data, "
        "which is then migrated, so it cannot write into a database at "
        "another version. Import into a new database instead "
        "(flextool-read-old BASE.xlsm sqlite:///new.sqlite "
        "--master MASTER.xlsm imports the sensitivities too)."
    )


def _require_master_workbook(master: str) -> None:
    """Refuse a ``master`` file without a ``Sensitivity definitions`` sheet."""
    import openpyxl

    wb = openpyxl.load_workbook(master, read_only=True)
    try:
        if "Sensitivity definitions" not in wb.sheetnames:
            raise ValueError(
                f"'{master}' has no 'Sensitivity definitions' sheet, so it is "
                "not a FlexTool 2 master workbook."
            )
    finally:
        wb.close()


def import_old_flextool_xlsm(
    xlsm_path: str,
    target_db_url: str,
    *,
    alternative_name: str = "base",
    purge: bool = True,
    migrate: bool = True,
    master: str | None = None,
    strict: bool = False,
) -> ImportReport:
    """Import an old-format FlexTool ``.xlsm`` into ``target_db_url``.

    The complete, single-source pipeline used by both the CLI and the GUI:

    1. Refuse a target that is a FlexTool DB at another version than the
       import schema (v56); initialise a new target from the **frozen** v56
       import template (``OLD_FLEXTOOL_IMPORT_SCHEMA``).
    2. Parse the workbook and write its data (this importer targets v56).
    3. With ``master``: write the sensitivities of the FT2 master workbook
       (its ``Sensitivity definitions`` sheet), still at v56.
    4. ``migrate_database`` the result forward to the current
       ``FLEXTOOL_DB_VERSION``, which absorbs every later schema change.
    5. Remove the behaviour-change markers the migration wrote: they describe
       changes between FlexTool 4 versions, and an FT2 import never ran on
       the old ones.  Only when the import owns the whole database (it
       initialised it, or purged it).

    ``migrate=False`` stops after step 3 (the database stays at v56).

    Returns:
        The :class:`ImportReport` (written, failed writes, inputs not
        translated, interpretation notes).
    """
    # Lazy imports: avoid an import-time cycle with ``update_flextool`` (which
    # imports back into ``process_inputs`` during migration) and keep the
    # heavyweight Excel reader off this module's import path.
    from flextool._resources import package_data_path
    from flextool.process_inputs.read_old_flextool import (
        read_old_flextool,
        read_old_flextool_sensitivities,
    )
    from flextool.update_flextool.db_migration import (
        BEHAVIOUR_CHANGE_MARKER_PREFIX,
        migrate_database,
        remove_behaviour_change_markers,
    )
    from flextool.update_flextool.initialize_database import initialize_database

    _require_import_version(target_db_url)

    # Read everything before creating or writing anything: a wrong file
    # (e.g. the master workbook as the input) fails here and leaves no DB.
    data = read_old_flextool(xlsm_path)
    _require_time_in_use(data, xlsm_path)
    sensitivities: dict[str, list[SensitivityOverride]] = {}
    if master:
        _require_master_workbook(master)
        sensitivities = read_old_flextool_sensitivities(master)

    initialised_here = False
    sqlite_path = _sqlite_path_from_url(target_db_url)
    from pathlib import Path

    # Delete on failure only a file this import created (never a user's file).
    created_file = sqlite_path is not None and not (
        Path(sqlite_path).exists() and Path(sqlite_path).stat().st_size > 0)
    if _db_version(target_db_url) is None:
        if sqlite_path is None:
            raise ValueError(
                "Cannot auto-initialise a non-sqlite target; create the "
                f"database first: {target_db_url}"
            )
        template = package_data_path(OLD_FLEXTOOL_IMPORT_SCHEMA)
        logger.info("Initialising %s from frozen import template (v%d)",
                    target_db_url, OLD_FLEXTOOL_IMPORT_SCHEMA_VERSION)
        initialize_database(str(template), sqlite_path)
        initialised_here = True

    try:
        report = ImportReport()
        report.add("base", write_old_flextool_to_db(
            data, target_db_url, alternative_name=alternative_name, purge=purge,
            strict=strict,
        ))
        if master:
            report.add("sensitivities", write_sensitivities_to_db(
                sensitivities, data, target_db_url,
                base_alternative=alternative_name, strict=strict,
            ))

        if migrate:
            logger.info("Migrating %s to the current schema version", target_db_url)
            migrate_database(target_db_url)
            if initialised_here or purge:
                with open_database_mapping(target_db_url) as db:
                    report.markers_removed = sorted(
                        str(md["name"]) for md in db.get_items("metadata")
                        if str(md["name"]).startswith(BEHAVIOUR_CHANGE_MARKER_PREFIX))
                    if remove_behaviour_change_markers(db):
                        db.commit_session(
                            "Removed behaviour-change markers (new FlexTool 2 import)"
                        )
    except BaseException:
        if initialised_here and created_file and sqlite_path is not None:
            # A half-written database created here is worse than none.
            Path(sqlite_path).unlink(missing_ok=True)
        raise
    return report
