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


def _write_storage_nodes_and_connections(
    data: OldFlexToolData,
    db: DatabaseMapping,
    alt_name: str,
    counters: _Counters,
    entities_added: set[tuple[str, tuple[str, ...]]],
    entity_alts_added: set[tuple[str, tuple[str, ...], str]],
) -> None:
    """Write storage nodes and their connections (Section 3)."""
    count = 0
    for unit in data.units:
        # Units with inflow profiles get their storage handled by _write_inflow_units
        if unit.inflow_profile:
            continue

        has_storage = (
            (unit.storage_mwh is not None and unit.storage_mwh > 0)
            or (unit.storage_mwh is not None and unit.storage_mwh == 0
                and (unit.storage_start is not None or unit.storage_finish is not None))
        )
        if not has_storage:
            continue

        unit_name = _get_unit_name(unit)
        storage_node = f"{unit_name}_storage"
        conn_name = f"{unit_name}_conn"
        output_node = unit.output_node

        if output_node is None:
            logger.warning("Unit '%s' has storage but no output node; skipping storage.", unit_name)
            continue

        # Storage node
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

        # Self-discharge loss from unit_type
        self_discharge = _get_unit_type_param(data, unit, "self discharge loss")
        _add_param_if_set(db, "node", (storage_node,), "self_discharge_loss",
                          self_discharge, alt_name, counters)

        # Storage state start
        _add_param_if_set(db, "node", (storage_node,), "storage_state_start",
                          unit.storage_start, alt_name, counters, skip_zero=False)

        # Connection entity
        _add_entity(db, "connection", conn_name, alt_name, counters,
                     entities_added, entity_alts_added)

        capacity = unit.capacity_mw or unit.invested_capacity_mw or 0.0
        _add_param_if_set(db, "connection", (conn_name,), "existing", capacity,
                          alt_name, counters)

        eff_charge = _get_unit_type_param(data, unit, "eff charge")
        _add_param(db, "connection", (conn_name,), "efficiency",
                   eff_charge if eff_charge is not None else 1.0, alt_name, counters)
        _add_param(db, "connection", (conn_name,), "transfer_method",
                   "regular", alt_name, counters)

        # connection__node__node: (conn, output_node, storage_node)
        _add_relationship(db, "connection__node__node",
                          (conn_name, output_node, storage_node),
                          alt_name, counters, entities_added, entity_alts_added)

        count += 1

    logger.info("Wrote %d storage nodes and connections.", count)


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

        unit_name = _get_unit_name(unit)
        ut = _get_unit_type(data, unit)

        # Create unit entity
        _add_entity(db, "unit", unit_name, alt_name, counters,
                     entities_added, entity_alts_added)

        # -- Determine conversion_method --
        has_cf = unit.cf_profile is not None
        has_fuel = unit.fuel is not None
        min_load_val = ut.params.get("min load") if ut else None
        has_min_load = min_load_val is not None and min_load_val > 0
        is_conversion = (
            unit.input_grid is not None
            and unit.output_grid is not None
            and unit.input_grid != unit.output_grid
        )

        if has_cf:
            conversion_method = "none"
        elif has_fuel and has_min_load:
            conversion_method = "min_load_efficiency"
        elif has_fuel:
            conversion_method = "constant_efficiency"
        elif is_conversion:
            conversion_method = "constant_efficiency"
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

        # -- startup parameters (only if use_online) --
        startup_cost = _get_unit_type_param(data, unit, "startup cost")
        is_fork = unit.output2_node is not None
        # Fork units with min_load need startup_method even without startup_cost
        needs_startup = (
            use_online
            and (
                (startup_cost is not None and startup_cost > 0)
                or (is_fork and has_min_load)
            )
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
        if unit.max_invest_mw is not None and unit.max_invest_mw > 0:
            _add_param(db, "unit", (unit_name,), "invest_method", "invest_total",
                       alt_name, counters)
            _add_param(db, "unit", (unit_name,), "invest_max_total",
                       unit.max_invest_mw, alt_name, counters)
            cap = unit.capacity_mw if unit.capacity_mw is not None else 0.0
            if cap > 0:
                _add_param(db, "unit", (unit_name,), "virtual_unitsize", cap,
                           alt_name, counters)

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
    count = 0
    for unit in data.units:
        if not unit.inflow_profile or not unit.output_node:
            continue

        inflow_ts = _find_inflow_profile(data, unit.inflow_profile)
        if inflow_ts is None or not inflow_ts.data:
            logger.warning(
                "Inflow profile '%s' not found for unit '%s'.",
                unit.inflow_profile, _get_unit_name(unit),
            )
            continue

        unit_name = _get_unit_name(unit)
        inflow_node = f"{unit_name}_inflow"

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

        # Storage properties (Hydro_RES pattern)
        has_storage = unit.storage_mwh is not None and unit.storage_mwh > 0
        if has_storage:
            _add_param(db, "node", (inflow_node,), "has_storage", "yes",
                       alt_name, counters)
            _add_param(db, "node", (inflow_node,), "existing",
                       unit.storage_mwh, alt_name, counters)
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

    logger.info("Wrote %d inflow unit nodes.", count)


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

        if conn.max_invest_mw is not None and conn.max_invest_mw > 0:
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

        # Link units to this group
        for unit in ug_units.get(ug.name, []):
            if unit.output_node is None:
                continue
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

    # -- Solve (dispatch) --
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

    # -- Solve (invest) if mode_invest --
    mode_invest = _get_master_param(data, "mode invest") == 1
    if mode_invest:
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
        _add_param(db, "solve", ("invest",), "realized_periods",
                   Array(["p2020"]), alt_name, counters)
        _add_param(db, "solve", ("invest",), "contains_solves", "dispatch",
                   alt_name, counters)
        _add_param(db, "solve", ("invest",), "solver", "highs",
                   alt_name, counters)

    # -- Model --
    _add_entity(db, "model", "flexTool", alt_name, counters,
                 entities_added, entity_alts_added)
    if mode_invest:
        _add_param(db, "model", ("flexTool",), "solves",
                   Array(["invest"]), alt_name, counters)
    else:
        _add_param(db, "model", ("flexTool",), "solves",
                   Array(["dispatch"]), alt_name, counters)

    logger.info(
        "Wrote timeline (%d steps), timeset, solve%s, and model.",
        len(data.time_steps),
        "+invest" if mode_invest else "",
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
        _write_storage_nodes_and_connections(data, db, alternative_name, counters,
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
