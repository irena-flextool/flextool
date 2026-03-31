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
from typing import Any

import spinedb_api as api
from spinedb_api import Array, DatabaseMapping, Map, SpineDBAPIError
from spinedb_api.exception import NothingToCommit

from flextool.process_inputs.read_old_flextool import (
    DemandTimeSeries,
    GridNode,
    NodeGroup,
    NodeNodeConnection,
    OldFlexToolData,
    SensitivityOverride,
    TimeSeriesData,
    UnitGroup,
    UnitInstance,
    UnitTimeSeries,
    UnitType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counters for summary logging
# ---------------------------------------------------------------------------

class _Counters:
    """Simple counters for tracking what was written."""

    def __init__(self) -> None:
        self.entities: int = 0
        self.relationships: int = 0
        self.parameters: int = 0
        self.entity_alternatives: int = 0

    def summary(self) -> str:
        return (
            f"entities={self.entities}, relationships={self.relationships}, "
            f"parameters={self.parameters}, entity_alternatives={self.entity_alternatives}"
        )


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
        logger.warning(
            "Could not add param %s.%s %s: %s",
            class_name, param_name, entity_byname, exc,
        )


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

        # Create fuel market node (no balance — it's a commodity source)
        market_name = _fuel_market_name(fuel.name)
        _add_entity(db, "node", market_name, alt_name, counters,
                     entities_added, entity_alts_added)

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
        _add_param(db, "node", (gn.node,), "has_balance", "yes",
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
    - A storage node with has_balance + has_storage
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
        ut = _get_unit_type(data, unit)
        storage_node = f"{unit_name}_storage"
        charger_name = f"{unit_name}_charger"
        discharger_name = f"{unit_name}_discharger"
        output_node = unit.output_node

        if output_node is None:
            logger.warning("Unit '%s' has no output node; skipping storage.", unit_name)
            continue

        capacity = unit.capacity_mw or 0.0

        # ── Storage node ──────────────────────────────────────────
        _add_entity(db, "node", storage_node, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (storage_node,), "has_balance", "yes",
                   alt_name, counters)
        _add_param(db, "node", (storage_node,), "has_storage", "yes",
                   alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "existing",
                          unit.storage_mwh, alt_name, counters, skip_zero=False)
        _add_param(db, "node", (storage_node,), "storage_binding_method",
                   "bind_within_solve", alt_name, counters)

        if lolp > 0:
            _add_param(db, "node", (storage_node,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (storage_node,), "penalty_down", 0.0,
                   alt_name, counters)

        self_discharge = _get_unit_type_param(data, unit, "self discharge loss")
        _add_param_if_set(db, "node", (storage_node,), "self_discharge_loss",
                          self_discharge, alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "storage_state_start",
                          unit.storage_start, alt_name, counters, skip_zero=False)

        # Storage node investment (inv.cost/kWh)
        inv_cost_kwh = _get_unit_type_param(data, unit, "inv.cost/kwh")
        _add_param_if_set(db, "node", (storage_node,), "invest_cost",
                          inv_cost_kwh, alt_name, counters)
        if unit.max_invest_mwh is not None and unit.max_invest_mwh > 0:
            _add_param(db, "node", (storage_node,), "invest_max_total",
                       unit.max_invest_mwh, alt_name, counters)
            _add_param(db, "node", (storage_node,), "invest_method",
                       "invest_total", alt_name, counters)

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

        avail = _get_unit_type_param(data, unit, "availability")
        if avail is not None and avail < 1.0:
            _add_param(db, "unit", (charger_name,), "availability", avail,
                       alt_name, counters)

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

        if avail is not None and avail < 1.0:
            _add_param(db, "unit", (discharger_name,), "availability", avail,
                       alt_name, counters)

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
        # (always set so sensitivities can enable investment later)
        _add_param_if_set(db, "unit", (discharger_name,), "invest_cost",
                          inv_cost_kw, alt_name, counters)
        _add_param_if_set(db, "unit", (discharger_name,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "unit", (discharger_name,), "interest_rate",
                          interest, alt_name, counters)
        _add_param_if_set(db, "unit", (charger_name,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "unit", (charger_name,), "interest_rate",
                          interest, alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "lifetime",
                          lifetime, alt_name, counters)
        _add_param_if_set(db, "node", (storage_node,), "interest_rate",
                          interest, alt_name, counters)

        invested_mw = unit.invested_capacity_mw
        invested_mwh = unit.invested_storage_mwh
        has_forced_mw = invested_mw is not None and invested_mw > 0
        has_forced_mwh = invested_mwh is not None and invested_mwh > 0
        has_invest = unit.max_invest_mw is not None and unit.max_invest_mw > 0

        if has_forced_mw:
            # Forced investment on charger and discharger
            for target in (discharger_name, charger_name):
                _add_param(db, "unit", (target,), "invest_method",
                           "invest_total", alt_name, counters)
                _add_param(db, "unit", (target,), "invest_max_total",
                           invested_mw, alt_name, counters)
                _add_param(db, "unit", (target,), "invest_min_total",
                           invested_mw, alt_name, counters)
        elif has_invest:
            for target in (discharger_name, charger_name):
                _add_param(db, "unit", (target,), "invest_method",
                           "invest_total", alt_name, counters)
                _add_param(db, "unit", (target,), "invest_max_total",
                           unit.max_invest_mw, alt_name, counters)

        if has_forced_mwh:
            # Forced investment on storage node
            _add_param(db, "node", (storage_node,), "invest_method",
                       "invest_total", alt_name, counters)
            _add_param(db, "node", (storage_node,), "invest_max_total",
                       invested_mwh, alt_name, counters)
            _add_param(db, "node", (storage_node,), "invest_min_total",
                       invested_mwh, alt_name, counters)
        elif unit.max_invest_mwh is not None and unit.max_invest_mwh > 0:
            _add_param(db, "node", (storage_node,), "invest_method",
                       "invest_total", alt_name, counters)
            _add_param(db, "node", (storage_node,), "invest_max_total",
                       unit.max_invest_mwh, alt_name, counters)

        # ── Charger ↔ Discharger investment constraint ────────────
        # Always active — ties charger capacity to discharger capacity
        charger_link = f"{unit_name}_charger_link"
        deactivate_storage_link = has_forced_mw and has_forced_mwh

        _add_entity(db, "constraint", charger_link, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "constraint", (charger_link,), "sense", "equal",
                   alt_name, counters)
        _add_param(db, "constraint", (charger_link,), "constant", 0.0,
                   alt_name, counters)

        # Charger always gets charger_link coefficient
        _add_param(db, "unit", (charger_name,),
                   "constraint_capacity_coefficient",
                   Map([charger_link], [-1.0], index_name="constraint"),
                   alt_name, counters)

        # Build discharger coefficient Map (always includes charger_link)
        discharger_coeff_indexes = [charger_link]
        discharger_coeff_values = [1.0]

        # ── kW/kWh constraint (discharger ↔ storage node) ────────
        # Deactivated when both MW and MWh are forced (avoids rounding issues)
        kw_kwh_ratio = _get_unit_type_param(data, unit, "fixed kw/kwh ratio")
        if kw_kwh_ratio is not None and kw_kwh_ratio > 0:
            storage_link = f"{unit_name}_storage_link"
            _add_entity(db, "constraint", storage_link, alt_name, counters,
                         entities_added, entity_alts_added)
            _add_param(db, "constraint", (storage_link,), "sense", "equal",
                       alt_name, counters)
            _add_param(db, "constraint", (storage_link,), "constant", 0.0,
                       alt_name, counters)

            if deactivate_storage_link:
                try:
                    db.add_or_update_entity_alternative(
                        entity_class_name="constraint",
                        entity_byname=(storage_link,),
                        alternative_name=alt_name,
                        active=False,
                    )
                except SpineDBAPIError:
                    pass
            else:
                # Storage node coefficient — only when constraint is active
                _add_param(db, "node", (storage_node,),
                           "constraint_capacity_coefficient",
                           Map([storage_link], [-1.0], index_name="constraint"),
                           alt_name, counters)
                # Add to discharger Map
                discharger_coeff_indexes.append(storage_link)
                discharger_coeff_values.append(kw_kwh_ratio)

        # Discharger coefficient Map (charger_link always, storage_link when active)
        _add_param(db, "unit", (discharger_name,),
                   "constraint_capacity_coefficient",
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

        # -- existing capacity --
        existing = unit.capacity_mw if unit.capacity_mw is not None else unit.invested_capacity_mw
        _add_param_if_set(db, "unit", (unit_name,), "existing", existing,
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

        # -- availability --
        avail = _get_unit_type_param(data, unit, "availability")
        if avail is not None and avail < 1.0:
            _add_param(db, "unit", (unit_name,), "availability", avail,
                       alt_name, counters)

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
        _add_param_if_set(db, "unit", (unit_name,), "interest_rate", interest_rate,
                          alt_name, counters)

        # -- investment --
        has_forced_invest = (
            unit.invested_capacity_mw is not None and unit.invested_capacity_mw > 0
        )
        has_max_invest = (
            unit.max_invest_mw is not None and unit.max_invest_mw > 0
        )
        if has_forced_invest:
            _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                       alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_max_total",
                       unit.invested_capacity_mw, alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_min_total",
                       unit.invested_capacity_mw, alt_name, counters)
        elif has_max_invest:
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

        # d) Secondary output (CHP, output2)
        if unit.output2_node:
            _add_relationship(db, "unit__outputNode", (unit_name, unit.output2_node),
                              alt_name, counters, entities_added, entity_alts_added)
            # Set coefficient for secondary output capacity scaling
            # output2_max_capacity / existing gives the ratio of secondary to primary capacity
            if unit.output2_max_capacity and existing and existing > 0:
                coeff = unit.output2_max_capacity / existing
                _add_param(db, "unit__outputNode",
                           (unit_name, unit.output2_node),
                           "coefficient", coeff, alt_name, counters)

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
    - Hydro_RES (has storage): node with has_balance + has_storage + existing
    - Hydro_ROR (no storage): node with has_balance only (use-it-or-lose-it)

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

        # Create the inflow node
        _add_entity(db, "node", inflow_node, alt_name, counters,
                     entities_added, entity_alts_added)
        _add_param(db, "node", (inflow_node,), "has_balance", "yes",
                   alt_name, counters)
        if lolp > 0:
            _add_param(db, "node", (inflow_node,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (inflow_node,), "penalty_down", 0.0,
                   alt_name, counters)

        # Always set has_storage on inflow nodes — even with 0 capacity this
        # just creates an unused state variable but allows sensitivities to add
        # storage capacity later without needing to also set has_storage.
        _add_param(db, "node", (inflow_node,), "has_storage", "yes",
                   alt_name, counters)
        _add_param_if_set(db, "node", (inflow_node,), "existing",
                          unit.storage_mwh, alt_name, counters, skip_zero=False)
        _add_param(db, "node", (inflow_node,), "storage_binding_method",
                   "bind_within_solve", alt_name, counters)
        # Self-discharge loss from unit_type
        self_discharge = _get_unit_type_param(data, unit, "self discharge loss")
        _add_param_if_set(db, "node", (inflow_node,), "self_discharge_loss",
                          self_discharge, alt_name, counters)
        # Storage state start
        _add_param_if_set(db, "node", (inflow_node,), "storage_state_start",
                          unit.storage_start, alt_name, counters, skip_zero=False)

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

        _add_param(db, "node", (node_name,), "has_balance", "yes",
                   alt_name, counters)
        if lolp > 0:
            _add_param(db, "node", (node_name,), "penalty_up", lolp,
                       alt_name, counters)
        _add_param(db, "node", (node_name,), "penalty_down", 0.0,
                   alt_name, counters)
        _add_param(db, "node", (node_name,), "has_storage", "yes",
                   alt_name, counters)
        _add_param(db, "node", (node_name,), "storage_binding_method",
                   "bind_within_solve", alt_name, counters)
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

        # existing capacity
        cap = conn.cap_rightward_mw or 0.0
        if conn.cap_leftward_mw is not None:
            cap = max(cap, conn.cap_leftward_mw)
        _add_param_if_set(db, "connection", (conn_name,), "existing", cap,
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
        _add_param_if_set(db, "connection", (conn_name,), "interest_rate",
                          conn.interest, alt_name, counters)

        has_forced_invest = (
            conn.invested_capacity_mw is not None and conn.invested_capacity_mw > 0
        )
        has_max_invest = (
            conn.max_invest_mw is not None and conn.max_invest_mw > 0
        )
        if has_forced_invest:
            _add_param(db, "connection", (conn_name,), "invest_method",
                       "invest_total", alt_name, counters)
            _add_param(db, "connection", (conn_name,), "invest_max_total",
                       conn.invested_capacity_mw, alt_name, counters)
            _add_param(db, "connection", (conn_name,), "invest_min_total",
                       conn.invested_capacity_mw, alt_name, counters)
        elif has_max_invest:
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
        _add_param(db, "group", (grid_name,), "output_results", "yes",
                   alt_name, counters)
        _add_param(db, "group", (grid_name,), "output_node_flows", "yes",
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

        _add_param(db, "group", (ug.name,), "output_aggregate_flows", "yes",
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
    """Write CHP constraints for units with output2 (Section 9).

    Collects all constraint names/coefficients per (unit, node) pair and writes
    a single Map per unit__outputNode, so multiple constraints don't overwrite
    each other.
    """
    count = 0
    for unit in data.units:
        if not unit.output2_node or not unit.output_node:
            continue

        unit_name = _get_unit_name(unit)
        primary_node = unit.output_node
        secondary_node = unit.output2_node

        # Collect coefficients: {(unit_name, node): {constraint_name: coeff}}
        primary_coeffs: dict[str, float] = {}
        secondary_coeffs: dict[str, float] = {}

        # Equal constraint
        if unit.output2_eq_coeff is not None and unit.output2_eq_coeff != 0:
            cname = f"{unit_name}_eq"
            _add_entity(db, "constraint", cname, alt_name, counters,
                         entities_added, entity_alts_added)
            _add_param(db, "constraint", (cname,), "sense", "equal",
                       alt_name, counters)
            constant = unit.output2_eq_constant if unit.output2_eq_constant is not None else 0.0
            _add_param(db, "constraint", (cname,), "constant", constant,
                       alt_name, counters)
            primary_coeffs[cname] = 1.0
            secondary_coeffs[cname] = -unit.output2_eq_coeff
            count += 1

        # Greater-than constraint
        if unit.output2_gt_coeff is not None and unit.output2_gt_coeff != 0:
            cname = f"{unit_name}_gt"
            _add_entity(db, "constraint", cname, alt_name, counters,
                         entities_added, entity_alts_added)
            _add_param(db, "constraint", (cname,), "sense", "greater_than",
                       alt_name, counters)
            constant = unit.output2_gt_constant if unit.output2_gt_constant is not None else 0.0
            _add_param(db, "constraint", (cname,), "constant", constant,
                       alt_name, counters)
            primary_coeffs[cname] = 1.0
            secondary_coeffs[cname] = -unit.output2_gt_coeff
            count += 1

        # Less-than constraint
        if unit.output2_lt_coeff is not None and unit.output2_lt_coeff != 0:
            cname = f"{unit_name}_lt"
            _add_entity(db, "constraint", cname, alt_name, counters,
                         entities_added, entity_alts_added)
            _add_param(db, "constraint", (cname,), "sense", "less_than",
                       alt_name, counters)
            constant = unit.output2_lt_constant if unit.output2_lt_constant is not None else 0.0
            _add_param(db, "constraint", (cname,), "constant", constant,
                       alt_name, counters)
            primary_coeffs[cname] = 1.0
            secondary_coeffs[cname] = -unit.output2_lt_coeff
            count += 1

        # Write combined Maps for constraint_flow_coefficient
        if primary_coeffs:
            names = list(primary_coeffs.keys())
            pmap = Map(names, [primary_coeffs[n] for n in names],
                       index_name="constraint")
            _add_param(db, "unit__outputNode", (unit_name, primary_node),
                       "constraint_flow_coefficient", pmap, alt_name, counters)

            snames = list(secondary_coeffs.keys())
            smap = Map(snames, [secondary_coeffs[n] for n in snames],
                       index_name="constraint")
            _add_param(db, "unit__outputNode", (unit_name, secondary_node),
                       "constraint_flow_coefficient", smap, alt_name, counters)

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

    # Build timeset_duration from contiguous blocks of in_use=True
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
) -> Map:
    """Build a timeset_duration Map from contiguous blocks of in_use steps.

    Args:
        time_steps: List of TimeStep objects.
        mode: "dispatch" (uses in_use) or "invest" (uses in_use_invest).
        duration_hours: Duration of each time step in hours.

    Returns:
        A Map where each key is the start time_id of a contiguous block
        and each value is the total duration of that block in hours.
    """
    from flextool.process_inputs.read_old_flextool import TimeStep

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

    return Map(starts, durations, index_name="time")


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
) -> None:
    """Write old FlexTool data to a Spine database.

    Args:
        data: Parsed old FlexTool data from read_old_flextool().
        db_url: Spine database URL (e.g., "sqlite:///path/to/db.sqlite").
        alternative_name: Name of the alternative to create for this data.
        purge: If True, purge existing data before import.
    """
    logger.info("Opening database: %s", db_url)

    with DatabaseMapping(db_url, create=False, upgrade=True) as db:
        if purge:
            logger.info("Purging existing data...")
            _purge_database(db)

        # Tracking sets to avoid duplicate add calls
        entities_added: set[tuple[str, tuple[str, ...]]] = set()
        entity_alts_added: set[tuple[str, tuple[str, ...], str]] = set()
        counters = _Counters()

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


# ---------------------------------------------------------------------------
# Sensitivity import
# ---------------------------------------------------------------------------


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
                                   "ramp_method", "no_ramp", alt_name, counters)

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
        logger.warning("Unknown master sensitivity param: '%s'", ov.param_name)


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
        logger.warning("Unknown nodeGroup sensitivity param: '%s'", ov.param_name)


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
            if is_storage:
                _add_param(db, "unit", (charger,), "availability", value, alt_name, counters)
                _add_param(db, "unit", (discharger,), "availability", value, alt_name, counters)
            else:
                _add_param(db, "unit", (unit_name,), "availability", value, alt_name, counters)
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
            _add_param(db, "unit", (target,), "interest_rate", value, alt_name, counters)
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
            logger.warning(
                "Unknown unit_type sensitivity param: '%s' for type '%s'",
                ov.param_name, type_name,
            )


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
        logger.warning("Unknown fuel sensitivity param: '%s'", ov.param_name)


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
        logger.warning("Unknown unitGroup sensitivity param: '%s'", ov.param_name)



def _apply_units_override(
    ov: SensitivityOverride,
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
    forced_invest: dict[str, set[str]] | None = None,
) -> None:
    """Apply a units-section sensitivity override."""
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

    if "capacity (mw)" == param_lower or param_lower == "capacity":
        if is_storage:
            _add_param(db, "unit", (charger,), "existing", value, alt_name, counters)
            _add_param(db, "unit", (discharger,), "existing", value, alt_name, counters)
        else:
            _add_param(db, "unit", (unit_name,), "existing", value, alt_name, counters)
    elif "invested capacity" in param_lower:
        if is_storage:
            for target in (charger, discharger):
                _add_param(db, "unit", (target,), "invest_max_total", value, alt_name, counters)
                _add_param(db, "unit", (target,), "invest_min_total", value, alt_name, counters)
                _add_param(db, "unit", (target,), "invest_method", "invest_total",
                           alt_name, counters)
            if forced_invest is not None:
                forced_invest.setdefault(unit_name, set()).add("mw")
        else:
            _add_param(db, "unit", (unit_name,), "invest_max_total", value, alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_min_total", value, alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                       alt_name, counters)
    elif "max invest (mw)" in param_lower and "mwh" not in param_lower:
        if is_storage:
            for target in (charger, discharger):
                _add_param(db, "unit", (target,), "invest_max_total", value, alt_name, counters)
                if isinstance(value, (int, float)) and float(value) > 0:
                    _add_param(db, "unit", (target,), "invest_method", "invest_total",
                               alt_name, counters)
        else:
            _add_param(db, "unit", (unit_name,), "invest_max_total", value, alt_name, counters)
            if isinstance(value, (int, float)) and float(value) > 0:
                _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                           alt_name, counters)
    elif "invested storage" in param_lower:
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            _add_param(db, "node", (storage_target,), "invest_max_total", value,
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "invest_min_total", value,
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "invest_method", "invest_total",
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "has_balance", "yes",
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "has_storage", "yes",
                       alt_name, counters)
            if forced_invest is not None:
                forced_invest.setdefault(unit_name, set()).add("mwh")
    elif "max invest (mwh)" in param_lower:
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            _add_param(db, "node", (storage_target,), "has_balance", "yes",
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "has_storage", "yes",
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "invest_max_total", value,
                       alt_name, counters)
            _add_param(db, "node", (storage_target,), "invest_method", "invest_total",
                       alt_name, counters)
    elif "storage (mwh)" in param_lower or (
        "storage" in param_lower and "invest" not in param_lower and "start" not in param_lower and "finish" not in param_lower
    ):
        storage_target = storage_node if is_storage else _get_unit_node_name(base_unit)
        if _entity_exists(existing_entities, "node", (storage_target,)):
            _add_param(db, "node", (storage_target,), "existing", value,
                       alt_name, counters)
            if float(value) > 0:
                _add_param(db, "node", (storage_target,), "has_balance", "yes",
                           alt_name, counters)
                _add_param(db, "node", (storage_target,), "has_storage", "yes",
                           alt_name, counters)
            else:
                _add_param(db, "node", (storage_target,), "has_storage", None,
                           alt_name, counters)
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
        logger.warning("Unknown units sensitivity param: '%s' for unit '%s'",
                       ov.param_name, unit_name)


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
        logger.warning("Unknown gridNode sensitivity param: '%s'", ov.param_name)


def _apply_node_node_override(
    ov: SensitivityOverride,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    existing_entities: set[tuple[str, tuple[str, ...]]],
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

    if "invested capacity" in param_lower:
        _add_param(db, "connection", (conn_name,), "invest_max_total", value, alt_name, counters)
        _add_param(db, "connection", (conn_name,), "invest_min_total", value, alt_name, counters)
        _add_param(db, "connection", (conn_name,), "invest_method", "invest_total",
                   alt_name, counters)
    elif "cap.rightward" in param_lower or "cap.leftward" in param_lower:
        # Use the value as existing capacity
        _add_param(db, "connection", (conn_name,), "existing", value, alt_name, counters)
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
        _add_param(db, "connection", (conn_name,), "interest_rate", value, alt_name, counters)
    else:
        logger.warning("Unknown nodeNode sensitivity param: '%s' for connection '%s'",
                       ov.param_name, conn_name)


def write_sensitivities_to_db(
    sensitivities: dict[str, list[SensitivityOverride]],
    data: OldFlexToolData,
    db_url: str,
    base_alternative: str = "base",
) -> None:
    """Write sensitivity alternatives and scenarios on top of an already-imported DB.

    This does NOT purge the database. It layers sensitivity alternatives on top
    of the base import.

    Args:
        sensitivities: Dict mapping scenario_name -> list of SensitivityOverride.
        data: The base OldFlexToolData (needed for unit_type->unit mapping, etc.).
        db_url: Spine database URL.
        base_alternative: Name of the base alternative already in the DB.
    """
    if not sensitivities:
        logger.info("No sensitivities to write.")
        return

    logger.info(
        "Writing %d sensitivity scenarios to: %s",
        len(sensitivities), db_url,
    )

    with DatabaseMapping(db_url, create=False, upgrade=True) as db:
        counters = _Counters()
        entities_added: set[tuple[str, tuple[str, ...]]] = set()
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

            # Track forced investments per unit for constraint deactivation
            # Keys: unit_name, values: set of "mw" and/or "mwh"
            forced_invest: dict[str, set[str]] = {}

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
                        entity_alts_added, forced_invest,
                    )
                elif ov.section == "nodeNode":
                    _apply_node_node_override(
                        ov, db, scenario_name, counters, existing_entities,
                    )
                else:
                    logger.warning("Unknown sensitivity section: '%s'", ov.section)

            # 4b. Handle storage_link constraint activation/deactivation.
            #     The charger_link (charger ↔ discharger) always stays active.
            #     The storage_link (discharger ↔ storage node kW/kWh ratio) is
            #     deactivated when both MW and MWh are forced, to avoid rounding.
            for unit in data.units:
                if not _needs_charger_discharger(unit, data):
                    continue
                unit_name = _get_unit_name(unit)
                ut = _get_unit_type(data, unit)
                kw_kwh_ratio = ut.params.get("fixed kw/kwh ratio") if ut else None
                if not (kw_kwh_ratio is not None and kw_kwh_ratio > 0):
                    continue  # No storage_link to manage

                # Base forced state
                base_forced_mw = (
                    unit.invested_capacity_mw is not None
                    and unit.invested_capacity_mw > 0
                )
                base_forced_mwh = (
                    unit.invested_storage_mwh is not None
                    and unit.invested_storage_mwh > 0
                )
                base_both_forced = base_forced_mw and base_forced_mwh

                # Effective forced state (base + sensitivity overrides)
                sens_types = forced_invest.get(unit_name, set())
                eff_forced_mw = "mw" in sens_types or (
                    base_forced_mw and "mw" not in sens_types
                )
                eff_forced_mwh = "mwh" in sens_types or (
                    base_forced_mwh and "mwh" not in sens_types
                )
                eff_both_forced = eff_forced_mw and eff_forced_mwh

                if eff_both_forced == base_both_forced:
                    continue  # No change

                charger_link = f"{unit_name}_charger_link"
                storage_link = f"{unit_name}_storage_link"
                discharger = f"{unit_name}_discharger"
                storage_node = f"{unit_name}_storage"

                if eff_both_forced and not base_both_forced:
                    # Deactivate storage_link, rewrite discharger coeff without it
                    if _entity_exists(existing_entities, "constraint", (storage_link,)):
                        try:
                            db.add_or_update_entity_alternative(
                                entity_class_name="constraint",
                                entity_byname=(storage_link,),
                                alternative_name=scenario_name,
                                active=False,
                            )
                        except SpineDBAPIError:
                            pass

                    # Discharger: only charger_link (drop storage_link)
                    _add_param(db, "unit", (discharger,),
                               "constraint_capacity_coefficient",
                               Map([charger_link], [1.0], index_name="constraint"),
                               scenario_name, counters)
                    # Storage node: null
                    null_value, null_type = api.to_database(None)
                    if _entity_exists(existing_entities, "node", (storage_node,)):
                        try:
                            db.add_parameter_value(
                                entity_class_name="node",
                                entity_byname=(storage_node,),
                                parameter_definition_name="constraint_capacity_coefficient",
                                alternative_name=scenario_name,
                                value=null_value, type=null_type,
                            )
                        except SpineDBAPIError:
                            pass

                elif base_both_forced and not eff_both_forced:
                    # Re-activate storage_link, write full coefficients
                    if _entity_exists(existing_entities, "constraint", (storage_link,)):
                        try:
                            db.add_or_update_entity_alternative(
                                entity_class_name="constraint",
                                entity_byname=(storage_link,),
                                alternative_name=scenario_name,
                                active=True,
                            )
                        except SpineDBAPIError:
                            pass

                    # Discharger: both charger_link + storage_link
                    _add_param(db, "unit", (discharger,),
                               "constraint_capacity_coefficient",
                               Map([charger_link, storage_link],
                                   [1.0, kw_kwh_ratio],
                                   index_name="constraint"),
                               scenario_name, counters)
                    # Storage node coefficient
                    _add_param(db, "node", (storage_node,),
                               "constraint_capacity_coefficient",
                               Map([storage_link], [-1.0], index_name="constraint"),
                               scenario_name, counters)

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
