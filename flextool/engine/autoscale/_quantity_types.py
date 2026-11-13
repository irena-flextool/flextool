"""Quantity-type lookup table for FlexTool parameters.

Derived from parameter_definition.description in
templates/input_data_template.sqlite (see specs/h2_trade_scaling_regression_handoff.md
and the autoscaler design).  This module maps every (parameter_name, entity_class)
pair declared in the template to a QuantityType so the autoscaler can apply a
semantically appropriate scaling factor per quantity.

UNKNOWN entries are intentional: the source description either gives no unit or
gives a context-dependent unit (e.g. [MW or MWh] on a group whose composition
decides the unit).  The autoscaler MUST refuse to scale UNKNOWN parameters and
instead surface them to the operator.
"""
from __future__ import annotations

from enum import Enum


class QuantityType(Enum):
    """Physical quantity carried by a FlexTool parameter value.

    Values are stable strings so they can appear in scaling-report CSVs without
    leaking Python identifiers.
    """

    ENERGY = "energy"                            # [MWh]
    POWER = "power"                              # [MW]
    APPARENT_POWER = "apparent_power"            # [MVA]
    INERTIA = "inertia"                          # [MWs]  synchronous inertia stock
    INERTIA_PER_CAPACITY = "inertia_per_capacity"  # [MWs/MW] (= seconds)
    CURRENCY = "currency"                        # [CUR]  reserved; no template parameter uses it directly
    PRICE_PER_ENERGY = "price_per_energy"        # [CUR/MWh]
    PRICE_PER_CAPACITY = "price_per_capacity"    # [CUR/kW], [CUR/kW/a], [CUR/MW]
    PRICE_PER_STORAGE = "price_per_storage"      # [CUR/kWh], [CUR/kWh/a]
    PRICE_PER_MASS = "price_per_mass"            # [CUR/tCO2]
    PRICE_PER_INERTIA = "price_per_inertia"      # [CUR/MWs per hour]
    EMISSION_MASS = "emission_mass"              # [tCO2], [tCO2/a]
    EMISSION_INTENSITY = "emission_intensity"    # [tCO2/MWh]
    FRACTION = "fraction"                        # [-]: shares, ratios, efficiencies, weights
    DURATION = "duration"                        # [hours], [years], "Hours."
    RAMP_RATE = "ramp_rate"                      # [share of capacity per minute]
    DIMENSIONLESS = "dimensionless"              # method choices, flags, name arrays, user-supplied coefficients
    UNKNOWN = "unknown"                          # unit cannot be determined from the description; requires operator action


PARAMETER_TYPES: dict[tuple[str, str], QuantityType] = {
    ('annual_flow', 'node'): QuantityType.ENERGY,
    # [MWh] Annual flow in energy units (always positive, the sign of inflow defines in/out). Inflow time series can be scaled to match annual flo...
    ('availability', 'connection'): QuantityType.FRACTION,
    # [e.g. 0.9 means 90%] Fraction of capacity available for connection flows. Constant or time.
    ('availability', 'node'): QuantityType.FRACTION,
    # [e.g. 0.9 means 90%] Fraction of capacity available for storage. Constant or time.
    ('availability', 'unit'): QuantityType.FRACTION,
    # [e.g. 0.9 means 90%] Fraction of capacity available for flows from/to the unit. For online units, the online variable is multiplied by the a...
    ('base_MVA', 'group'): QuantityType.APPARENT_POWER,
    # [MVA] Base power for the per-unit system used in DC power flow. Default 100. susceptance = base_MVA / reactance_pu.
    ('benders_max_iter', 'solve'): QuantityType.DIMENSIONLESS,
    # Benders max outer (master/subproblem) iterations. Solver-algorithm knob, never an LP coefficient.
    ('benders_tolerance', 'solve'): QuantityType.DIMENSIONLESS,
    # Benders relative optimality-gap convergence threshold. Solver-algorithm knob, never an LP coefficient.
    ('benders_in_out_weight', 'solve'): QuantityType.DIMENSIONLESS,
    # Benders in-out separation stabilization weight (lambda). Solver-algorithm knob, never an LP coefficient.
    ('candidate_precapacity_to_avoid_big_m', 'group'): QuantityType.POWER,
    # [MW] Small pre-existing capacity assigned to investment candidate connections in DC power flow groups that have zero existing capacity. This...
    ('capacity_margin', 'group'): QuantityType.POWER,
    # [MW] Capacity the group needs above its demand at every time step of an investment period (capacity_margin_method manual).
    ('capacity_margin_share', 'group'): QuantityType.FRACTION,
    # [-] Capacity margin as a share of the group's peak demand (capacity_margin_method share_of_peak).
    # Pre-v70 input-side names, carried during the migration window
    # (renamed to input_share_max / input_share_min by db_migration v70).
    ('capacity_max_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    ('capacity_max_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity (existing + invested - retired) available to this edge's upper cap (maxToSink / maxFro...
    ('capacity_min_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    ('capacity_min_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity imposed as a lower cap on this edge when online (combined multiplicatively with the un...
    # Legacy long-form aliases still present in the template DB until db_migration.py renames
    # max_capacity_coefficient → capacity_max_coeff and min_capacity_coefficient → capacity_min_coeff.
    ('constraint_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # Puts the connection's capacity in service in each period on the left side of the user constraint (existing plus investments still in service...
    ('constraint_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # Puts the node's storage capacity in service in each period on the left side of the user constraint (existing plus investments still in servi...
    ('constraint_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # Puts the unit's capacity in service in each period on the left side of the user constraint (existing plus investments still in service minus...
    ('constraint_in_construction_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # Puts the capacity in construction in each period on the left side of the user constraint. Capacity is in con...
    ('constraint_in_construction_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # Puts the storage capacity in construction in each period on the left side of the user constraint. Capacity...
    ('constraint_in_construction_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # Puts the capacity in construction in each period on the left side of the user constraint. Capacity is in con...
    ('constraint_online_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # Puts the number of units of the connection online in each time step on the left side of the user constraint.
    ('constraint_online_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # Puts the number of units online in each time step on the left side of the user constraint.
    ('constraint_retired_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # Puts the capacity retired in each period on the left side of the user constraint. Coefficient per MW.
    ('constraint_retired_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # Puts the storage capacity retired in each period on the left side of the user constraint. Coefficient per MWh.
    ('constraint_retired_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # Puts the capacity retired in each period on the left side of the user constraint. Coefficient per MW.
    ('max_capacity_coefficient', 'unit__inputNode'): QuantityType.FRACTION,
    ('max_capacity_coefficient', 'unit__outputNode'): QuantityType.FRACTION,
    ('min_capacity_coefficient', 'unit__inputNode'): QuantityType.FRACTION,
    ('min_capacity_coefficient', 'unit__outputNode'): QuantityType.FRACTION,
    ('co2_content', 'commodity'): QuantityType.EMISSION_INTENSITY,
    # [CO2 ton per MWh] Constant.
    ('co2_max_period', 'group'): QuantityType.EMISSION_MASS,
    # [tCO2/a] Largest annual CO2 emissions in each period. In a rolling solve, each roll may use the part of the limit its horizon reaches, minus...
    ('co2_max_total', 'group'): QuantityType.EMISSION_MASS,
    # [tCO2] Largest CO2 emissions over the whole model horizon. Each period counts its annual emissions times its years_represented. Rolling solv...
    ('co2_method', 'group'): QuantityType.DIMENSIONLESS,
    # Choice of the CO2 method: none, price, period, total, price_period, price_total, period_total, price_period_total
    ('co2_price', 'group'): QuantityType.PRICE_PER_MASS,
    # [CUR/ton] CO2 price for a group of nodes. Constant, period or time.
    ('construction_lead_time', 'connection'): QuantityType.DURATION,
    # [years] Construction lead time: capacity ordered in a period becomes available this many years later (snapped to a period boundary). Constant or period.
    ('construction_lead_time', 'node'): QuantityType.DURATION,
    # [years] Construction lead time: capacity ordered in a period becomes available this many years later (snapped to a period boundary). Constant or period.
    ('construction_lead_time', 'unit'): QuantityType.DURATION,
    # [years] Construction lead time: capacity ordered in a period becomes available this many years later (snapped to a period boundary). Constant or period.
    ('construction_lead_time_method', 'connection'): QuantityType.DIMENSIONLESS,
    # How yr(order)+construction_lead_time snaps to a period boundary (immediate/closest_seam/previous_seam/next_seam) — an enum string, no physical unit.
    ('construction_lead_time_method', 'node'): QuantityType.DIMENSIONLESS,
    # How yr(order)+construction_lead_time snaps to a period boundary (immediate/closest_seam/previous_seam/next_seam) — an enum string, no physical unit.
    ('construction_lead_time_method', 'unit'): QuantityType.DIMENSIONLESS,
    # How yr(order)+construction_lead_time snaps to a period boundary (immediate/closest_seam/previous_seam/next_seam) — an enum string, no physical unit.
    ('debug', 'model'): QuantityType.DIMENSIONLESS,
    # Instruction set for performing model debugging and testing — a string toggle, no physical unit.
    ('constraint_period_method', 'constraint'): QuantityType.DIMENSIONLESS,
    # Whether a user constraint has a row per time step or one annualised total per period — an enum string.
    ('constant', 'constraint'): QuantityType.DIMENSIONLESS,
    # Right-hand side of the user constraint. A period map gives each period its own value; it must cover every period of the solve. Constant or p...
    # User constraints aggregate LHS terms whose units depend on the user's coefficient choices; the RHS constant lives
    # in that same composite coordinate. Layer-2 does not apply a per-row factor — column scalers propagate via the LHS.
    ('constraint_cumulative_pre_built_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # Puts the capacity in service at the start of each period on the left side of the user constraint (before that period's investments and retir...
    ('constraint_cumulative_pre_built_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # Puts the capacity in service at the start of each period on the left side of the user constraint (before that period's investments and retir...
    ('constraint_cumulative_pre_built_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # Puts the capacity in service at the start of each period on the left side of the user constraint (before that period's investments and retir...
    # Legacy long-form names still present in the template DB until migration runs (db_migration.py renames *_coefficient → *_coeff).
    ('constraint_cumulative_pre_built_capacity_coefficient', 'connection'): QuantityType.DIMENSIONLESS,
    ('constraint_cumulative_pre_built_capacity_coefficient', 'node'): QuantityType.DIMENSIONLESS,
    ('constraint_cumulative_pre_built_capacity_coefficient', 'unit'): QuantityType.DIMENSIONLESS,
    ('constraint_flow_coeff', 'connection__node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow from the connection to a node ...
    ('constraint_flow_coeff', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow between unit and node in user-...
    ('constraint_flow_coeff', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow between unit and node in user-...
    # Legacy long-form names still present in the template DB until migration runs.
    ('constraint_flow_coefficient', 'connection__node'): QuantityType.DIMENSIONLESS,
    ('constraint_flow_coefficient', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    ('constraint_flow_coefficient', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    ('constraint_invested_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    ('constraint_invested_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    ('constraint_invested_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    # Legacy long-form names still present in the template DB until migration runs.
    ('constraint_invested_capacity_coefficient', 'connection'): QuantityType.DIMENSIONLESS,
    ('constraint_invested_capacity_coefficient', 'node'): QuantityType.DIMENSIONLESS,
    ('constraint_invested_capacity_coefficient', 'unit'): QuantityType.DIMENSIONLESS,
    ('constraint_state_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the storage state in user-defined const...
    ('constraint_state_coefficient', 'node'): QuantityType.DIMENSIONLESS,
    # Legacy long-form name still present in the template DB until migration runs (renamed to constraint_state_coeff).
    ('contains_solves', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of solves - used for nested solve sequencesArray of solves - used for nested solve sequences
    ('conversion_flow_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor, default 1] Fuel-equivalent energy per unit of this flow. It enters the unit's conversion equation, not the node balance. Applies to...
    ('conversion_flow_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor, default 1] Fuel-equivalent energy per unit of this flow. It enters the unit's conversion equation, not the node balance. Applies to...
    # Legacy long-form alias still present in the template DB until db_migration.py renames flow_coefficient → conversion_flow_coeff.
    ('flow_coefficient', 'unit__inputNode'): QuantityType.FRACTION,
    ('flow_coefficient', 'unit__outputNode'): QuantityType.FRACTION,
    ('conversion_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice of conversion method.
    ('cumulative_max_capacity', 'connection'): QuantityType.POWER,
    # [MW] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_max_capacity', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Maximum cumulative capacity for a group of entities (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_max_capacity', 'node'): QuantityType.ENERGY,
    # [MWh] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_max_capacity', 'unit'): QuantityType.POWER,
    # [MW] Maximum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_min_capacity', 'connection'): QuantityType.POWER,
    # [MW] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_min_capacity', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Minimum cumulative capacity for a group of entities (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_min_capacity', 'node'): QuantityType.ENERGY,
    # [MWh] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('cumulative_min_capacity', 'unit'): QuantityType.POWER,
    # [MW] Minimum cumulative capacity (considers existing, invested and retired capacity). Constant or period.
    ('decomposition', 'solve'): QuantityType.DIMENSIONLESS,
    # Decomposition scheme for this solve: 'none' (monolithic, default) or 'benders' (region decomposition). Solver-config choice, never an LP coefficient.
    ('decomposition_method', 'group'): QuantityType.DIMENSIONLESS,
    # Decomposition strategy to apply to this group. Currently supported: 'none' (no decomposition — default), 'benders_regional' (group is solve...
    ('delay', 'connection'): QuantityType.DURATION,
    # [hours] A time delay between the input node and the output node - works only with one-way connections (or units). Either a constant indicati...
    ('delay', 'unit'): QuantityType.DURATION,
    # [hours] A time delay between the input nodes and the output nodes. Either a constant indicating the time difference in hours or a map of tim...
    ('discount_rate', 'connection'): QuantityType.FRACTION,
    # [e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model...
    ('discount_rate', 'node'): QuantityType.FRACTION,
    # [e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model...
    ('discount_rate', 'unit'): QuantityType.FRACTION,
    # [e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk premium for this technology. When the model...
    ('efficiency', 'connection'): QuantityType.FRACTION,
    # [factor, typically 0-1] Efficiency of the connection. Constant, period, time or period+time.
    ('efficiency', 'unit'): QuantityType.FRACTION,
    # [factor] Efficiency of the unit at full load: outputs / inputs, each flow weighted by its conversion_flow_coeff (default 1, giving plain sum...
    ('efficiency_at_min_load', 'unit'): QuantityType.FRACTION,
    # [e.g. 0.4 means 40%] Efficiency of a min_load_efficiency unit at its minimum load. Fuel use is a straight line from minimum load (this effic...
    ('energy_margin_multiplier', 'node'): QuantityType.DIMENSIONLESS,
    # [factor] Multiplier applied to this node's inflow in the investment solve only, when energy_margin_method = 'inflow_multiplier'. Offsets rep...
    ('energy_margin', 'node'): QuantityType.DIMENSIONLESS,
    # Deprecated v65 name for energy_margin_multiplier; carried during the v65->v66 migration window so a pre-migration DB read is not silently un-scaled.
    ('energy_margin_adder', 'node'): QuantityType.ENERGY,
    # [MWh] Extra demand (negative inflow) added to this node in the investment solve only, when energy_margin_method = 'inflow_adder'. Raises ade...
    ('energy_margin_method', 'node'): QuantityType.DIMENSIONLESS,
    # Method choice {none, inflow_multiplier, inflow_adder} selecting whether/how the energy margin is applied.
    ('existing', 'connection'): QuantityType.POWER,
    # [MW or count] Existing capacity or number of units. If virtual_unitsize is given, then existing is the number of units (capacity = existing ...
    ('existing', 'node'): QuantityType.ENERGY,
    # [MWh or count] Existing storage capacity or number of storage units. If virtual_unitsize is given, then existing is the number of units (cap...
    ('existing', 'unit'): QuantityType.POWER,
    # [MW or count] Existing capacity or number of units. If virtual_unitsize is given, then existing is the number of units (capacity = existing ...
    ('exclude_entity_outputs', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean-style toggle: excludes node/unit/connection-level results while preserving group-level results.
    ('fix_storage_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of periods where the storage_values are fixed when the node has storage_include_solve_fix_method is set
    ('fixed_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Annual fixed cost. Constant or period.
    ('fixed_cost', 'node'): QuantityType.PRICE_PER_STORAGE,
    # [CUR/kWh] Annual fixed cost for storage. Constant or period.
    ('fixed_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Annual fixed cost. Constant or period.
    ('flow_aggregator', 'flowGroup'): QuantityType.DIMENSIONLESS,
    # Used with group_unit_node or group_connection_node to combine the flows when producing the dispatch output of a node group. Renamed from out...
    ('has_capacity_margin', 'group'): QuantityType.DIMENSIONLESS,
    # A flag whether the group of nodes has a capacity margin constraint in the investment mode.
    ('capacity_margin_method', 'group'): QuantityType.DIMENSIONLESS,
    # Whether the group needs a capacity margin in investment solves, and how the margin is given (none, manual, share_of_peak).
    ('has_inertia', 'group'): QuantityType.DIMENSIONLESS,
    # Whether the group has an inertia constraint. Without yes, inertia_limit is ignored.
    ('has_non_synchronous', 'group'): QuantityType.DIMENSIONLESS,
    # Whether the group has a non-synchronous share limit. Without yes, non_synchronous_limit is ignored.
    ('include_stochastics', 'group'): QuantityType.DIMENSIONLESS,
    # Includes the stochastic branches to be used for the nodes/units/connections in this group
    ('increase_reserve_ratio', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] The reserve requirement grows by this ratio times the flow between the connection and the node, in either direction, for upward and...
    ('increase_reserve_ratio', 'reserve__upDown__group'): QuantityType.FRACTION,
    # [factor] Only for dynamic reserve methods. The requirement grows by this ratio times the demand of the group's nodes in each step (MW, real ...
    ('increase_reserve_ratio', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] The reserve requirement grows by this ratio times the flow between the unit and the node, for upward and downward reserve alike. At...
    ('inertia_constant', 'unit__inputNode'): QuantityType.INERTIA_PER_CAPACITY,
    # [MWs/MW] Inertia constant for a synchronously connected unit to this node. Constant.
    ('inertia_constant', 'unit__outputNode'): QuantityType.INERTIA_PER_CAPACITY,
    # [MWs/MW] Inertia constant for a synchronously connected unit to this node. Constant.
    ('inertia_limit', 'group'): QuantityType.INERTIA,
    # [MWs] Minimum synchronous inertia in the group. Used only when has_inertia is yes. No value means no requirement. Constant or period.
    ('inflation_offset_investment', 'model'): QuantityType.DURATION,
    # [years] Offset for when investment costs occur within a year. Default 0 (beginning of year).
    ('inflation_offset_operations', 'model'): QuantityType.DURATION,
    # [years] Offset for when operational costs occur within a year. Default 0.5 (middle of year).
    ('inflation_rate', 'model'): QuantityType.FRACTION,
    # [e.g. 0.02 for 2%] Model-wide inflation rate applied to all future costs. When inputs are in real (constant-price) terms, set to 0. When inp...
    ('inflow', 'node'): QuantityType.ENERGY,
    # [MWh] Inflow into the node (negative is outflow). Constant or time.
    ('inflow_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how to treat inflow time series. Empty defaults to 'use_original', which does not scale the time series. 'no_inflow' ignores the infl...
    ('input_share_max', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Largest share of the unit's full-load input energy (flow x conversion_flow_coeff) that this input may supply, relativ...
    ('input_share_min', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Lower-limit counterpart of input_share_max (renamed from capacity_min_coeff in schema v70). Currently not applied by ...
    ('invest_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Investment cost for new 'virtual' capacity. Constant or period.
    ('invest_cost', 'node'): QuantityType.PRICE_PER_STORAGE,
    # [CUR/kWh] Investment cost for new storage capacity. Constant or period.
    ('invest_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Investment cost of the unit. Constant or period.
    ('invest_forced', 'node'): QuantityType.ENERGY,
    # (empty description) — inferred from ('existing', 'node') = [MWh]: forced storage-capacity investment.
    ('invest_max_period', 'connection'): QuantityType.POWER,
    # [MW] Largest allowed investment in each period. Counts only what is built in that period; existing capacity and retirements do not change it...
    ('invest_max_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Largest allowed investment in each period, summed over the group's units and connections (MW) and storage nodes (MWh). Constant ...
    ('invest_max_period', 'node'): QuantityType.ENERGY,
    # [MWh] Largest allowed storage investment in each period. Counts only what is built in that period; existing capacity and retirements do not ...
    ('invest_max_period', 'unit'): QuantityType.POWER,
    # [MW] Largest allowed investment in each period. Counts only what is built in that period; existing capacity and retirements do not change it...
    ('invest_max_total', 'connection'): QuantityType.POWER,
    # [MW] Largest allowed total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in ser...
    ('invest_max_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Largest allowed total investment of the group over all solves. Every investment counts; retirements do not reduce it. Constant.
    ('invest_max_total', 'node'): QuantityType.ENERGY,
    # [MWh] Largest allowed total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in se...
    ('invest_max_total', 'unit'): QuantityType.POWER,
    # [MW] Largest allowed total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in ser...
    ('invest_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: not_allowed, invest and retire indicate availability of investment and retirement. no_limit removes all limits ...
    ('invest_method', 'group'): QuantityType.DIMENSIONLESS,
    # How the group limits investment and retirement of its members. not_active: the group sets no limits. invest / retire / invest_retire: which ...
    ('invest_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: either not_allowed or then a combination of 1) investment or retirement and 2) investment limits per period, al...
    ('invest_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: not_allowed, invest and retire indicate availability of investment and retirement. no_limit removes all limits ...
    ('invest_min_period', 'connection'): QuantityType.POWER,
    # [MW] Smallest required investment in each period. Counts only what is built in that period; existing capacity and retirements do not change ...
    ('invest_min_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Smallest required investment in each period, summed over the group's units and connections (MW) and storage nodes (MWh). Constan...
    ('invest_min_period', 'node'): QuantityType.ENERGY,
    # [MWh] Smallest required storage investment in each period. Counts only what is built in that period; existing capacity and retirements do no...
    ('invest_min_period', 'unit'): QuantityType.POWER,
    # [MW] Smallest required investment in each period. Counts only what is built in that period; existing capacity and retirements do not change ...
    ('invest_min_total', 'connection'): QuantityType.POWER,
    # [MW] Smallest required total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in s...
    ('invest_min_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Smallest required total investment of the group over all solves. Every investment counts; retirements do not reduce it. Constant...
    ('invest_min_total', 'node'): QuantityType.ENERGY,
    # [MWh] Smallest required total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in ...
    ('invest_min_total', 'unit'): QuantityType.POWER,
    # [MW] Smallest required total investment over all solves. Every investment counts, also reinvestments and investments that are no longer in s...
    ('invest_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of periods where investments are allowed.
    ('is_DC', 'connection'): QuantityType.DIMENSIONLESS,
    # Whether the connection is DC. Its flow into a group with a non-synchronous limit counts as non-synchronous, in both directions. Default no.
    ('is_non_synchronous', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # Chooses whether the unit is synchronously connected to this node.
    ('is_non_synchronous', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # Chooses whether the unit is synchronously connected to this node.
    # v56 (fb444974) re-added is_enabled on these three classes; it is a
    # yes_no gate consumed at read time (_spinedb_reader.py:205 drops
    # is_enabled="no" entities) and never enters the LP as a coefficient,
    # so DIMENSIONLESS per the flag convention (is_DC / is_non_synchronous).
    ('is_enabled', 'constraint'): QuantityType.DIMENSIONLESS,
    # Flag whether the constraint is enabled; disabled rows are dropped.
    ('is_enabled', 'reserve__upDown__connection__node'): QuantityType.DIMENSIONLESS,
    # Flag whether the reserve participation is enabled; disabled rows dropped.
    ('is_enabled', 'reserve__upDown__unit__node'): QuantityType.DIMENSIONLESS,
    # Flag whether the reserve participation is enabled; disabled rows dropped.
    ('large_failure_ratio', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] With a large_failure method, the reserve must cover the loss of this connection's flow times this ratio. Its own reserve does not c...
    ('large_failure_ratio', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] With a large_failure method, the reserve must cover the loss of this unit's flow times this ratio. Its own reserve does not count. ...
    ('lifetime', 'connection'): QuantityType.DURATION,
    # [years] Used to calculate annuity together with interest rate. Constant or period.
    ('lifetime', 'node'): QuantityType.DURATION,
    # [years] Used to calculate annuity together with interest rate. Constant or period.
    ('lifetime', 'unit'): QuantityType.DURATION,
    # [years] Used to calculate annuity together with interest rate. Constant or period.
    ('lifetime_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice how the investments behave after unit runs out of lifetime. Automatic reinvestment causes the model to keep the capacity until the en...
    ('lifetime_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how the investments behave after unit runs out of lifetime. Automatic reinvestment causes the model to keep the capacity until the en...
    ('lifetime_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice how the investments behave after unit runs out of lifetime. Automatic reinvestment causes the model to keep the capacity until the en...
    ('max_cumulative_flow', 'flowGroup'): QuantityType.POWER,
    # [MW] Largest allowed average flow into the group. A constant gives one limit per solve; a period map gives one limit per period. Average flo...
    ('max_flow_for_unconstrained_variables', 'model'): QuantityType.POWER,
    # [MW] Upper bound assigned to LP variables that have no other cap (invest_no_limit / invest_retire_no_limit capacity; flows through edges who...
    ('max_instant_flow', 'flowGroup'): QuantityType.POWER,
    # [MW] Maximum instantenous flow for the aggregated flow of all group members. Constant or period.
    ('max_share', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] Largest share of the connection's built capacity that can be held as this reserve at this node. Constant.
    ('max_share', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] Largest share of the unit's built capacity at this node that can be held as this reserve. At an input node, the capacity is the uni...
    ('min_cumulative_flow', 'flowGroup'): QuantityType.POWER,
    # [MW] Smallest required average flow into the group. A constant gives one limit per solve; a period map gives one limit per period. Average f...
    ('min_downtime', 'unit'): QuantityType.DURATION,
    # [hours] Minimum time the unit stays offline after a stop. Needs minimum_time_method. The look-back stops at a gap in the timeline and at the...
    ('min_instant_flow', 'flowGroup'): QuantityType.POWER,
    # [MW] Minimum instantenous flow for the aggregated flow of all group members. Constant or period.
    ('min_load', 'unit'): QuantityType.FRACTION,
    # [0-1] Minimum load of the unit as a share of its online capacity. Applies only if the unit has online variables. No value means 0. Constant,...
    ('min_uptime', 'unit'): QuantityType.DURATION,
    # [hours] Minimum time the unit stays online after a start. Needs minimum_time_method. The look-back stops at a gap in the timeline and at the...
    ('minimum_time_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice between minimum up- and downtimes (none, min_downtime, min_uptime, both). Setting this to anything other than 'none' will activate on...
    ('new_stepduration', 'group'): QuantityType.DURATION,
    # [hours] Balances the energy of the group's nodes once per step of this length instead of every time step, e.g. daily hydrogen next to hourly...
    ('new_stepduration', 'solve'): QuantityType.DURATION,
    # Hours. Creates a new `timeline` from the old for this `solve` with this timestep duration. The new timeline will sum or average the other ti...
    ('node_type', 'node'): QuantityType.DIMENSIONLESS,
    # Role of this node in the LP.  'commodity' = price-exposed source/sink with no balance constraint (e.g. fuel imports, no storage); 'balance' ...
    ('retire_max_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Largest allowed retirement in each period, summed over the group's members. Needs a retire …_period option in the group's invest...
    ('retire_max_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Largest allowed total retirement of the group over all solves. Needs a retire …_total option in the group's invest_method. Const...
    ('retire_min_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Smallest required retirement in each period, summed over the group's members. Needs a retire …_period option in the group's inve...
    ('retire_min_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Smallest required total retirement of the group over all solves. Needs a retire …_total option in the group's invest_method. Con...
    ('shared_invest_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Periods where investments are the same in every stochastic branch (with recourse). No value or empty array: none.
    ('shared_operation_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Periods where operations are the same in every stochastic branch. No value: the committed time steps. Empty array: none.
    ('non_synchronous_limit', 'group'): QuantityType.FRACTION,
    # [share, e.g. 0.8] Largest share of non-synchronous supply in the group, measured against the real demand. Flows between the group's nodes do...
    ('other_operational_cost', 'connection'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, Period or time.
    ('other_operational_cost', 'unit__inputNode'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.
    ('other_operational_cost', 'unit__outputNode'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.
    ('output_connection__node__node_flow_t', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: per-timestep connection→node flows. Flag, no physical unit.
    ('output_connection_flow_separate', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: produce connection flows separately for each direction. Flag.
    ('output_horizon', 'model'): QuantityType.DIMENSIONLESS,
    # Debug flag. yes writes one raw parquet file per LP variable and constraint, with every row of every solve and a realized column. Normal outp...
    ('output_node_balance_t', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: per-timestep node balance (inflows/outflows) for diagnostics. Flag.
    ('print_dispatch', 'group'): QuantityType.DIMENSIONLESS,
    # Creates the timewise flow output for this node group (node-group dispatch table). Renamed from output_nodeGroup_dispatch.
    ('print_indicators', 'group'): QuantityType.DIMENSIONLESS,
    # Flag to output node-group indicator results for groups whose members are nodes (group__node). Renamed from output_nodeGroup_indicators.
    ('output_ramp_envelope', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: include the seven-parameter ramp-envelope diagnostic outputs. Flag.
    ('output_unit__node_flow_t', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: per-timestep unit→node flows. Flag.
    ('output_unit__node_ramp_t', 'model'): QuantityType.DIMENSIONLESS,
    # Boolean output toggle: per-timestep per-unit ramps. Flag.
    ('peak_inflow', 'node'): QuantityType.ENERGY,
    # [MWh] Highest flow for scaling the inflow. Used only with inflow_method scale_to_annual_and_peak_flow. Constant or period.
    ('penalty_capacity_margin', 'group'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Penalty for violating the capacity margin constraint. Uses operational discounting (not annualized over lifetime like investment co...
    ('penalty_down', 'node'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, Period or Time.
    ('penalty_inertia', 'group'): QuantityType.PRICE_PER_INERTIA,
    # [CUR/MWs] Penalty for violating the inertia constraint. Cost scales with the duration of the violation. Constant or period.
    ('penalty_method', 'node'): QuantityType.DIMENSIONLESS,
    # Per-node toggle: 'regular' keeps balance-slack (penalty) vars, 'off' removes them.
    ('penalty_non_synchronous', 'group'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Penalty for violating the non synchronous constraint. Constant or period.
    ('penalty_reserve', 'reserve__upDown__group'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Penalty for violating a reserve constraint. Constant.
    ('penalty_up', 'node'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, Period or Time.
    ('period_timeset', 'solve'): QuantityType.DIMENSIONLESS,
    # Map of periods with associated timesets that will be included in the solve. Index: period name, value: timeset name.
    ('periods_available', 'model'): QuantityType.DIMENSIONLESS,
    # (Optional) Array of periods available for the model. Periods that are in the data, but are not in period_timeset.
    ('price', 'commodity'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh or other unit] Price of the commodity. Constant, period or time.
    ('price_ladder_annual', 'commodity'): QuantityType.DIMENSIONLESS,
    # Stepped supply curve for price_method price_ladder_annual. Map of tier → price and quantity (optionally per period). Quantity is per year; a...
    ('price_ladder_cumulative', 'commodity'): QuantityType.DIMENSIONLESS,
    # Stepped supply curve for price_method='price_ladder_cumulative'.  2d map with rows 'tier,price,quantity' — one row per tier, giving the tier...
    ('price_method', 'commodity'): QuantityType.DIMENSIONLESS,
    # How the commodity's price enters the LP.  'price' = scalar or time-series price x flow (current behaviour); 'price_ladder_annual' = stepped ...
    ('profile', 'profile'): QuantityType.FRACTION,
    # [factor, typically 0-1] Profile values, e.g. the availability of wind power (capacity factor). Constant, period, time, period+time or stocha...
    ('profile_method', 'connection__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('profile_method', 'node__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('profile_method', 'unit__node__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('ramp_cost', 'unit__inputNode'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost per MW of change in the step-average flow, up or down, between a time step and the step before it. Charged per change...
    ('ramp_cost', 'unit__outputNode'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost per MW of change in the step-average flow, up or down, between a time step and the step before it. Charged per change...
    ('ramp_method', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # [choice] Choice of ramp method. ramp_limit limits the ramp speed. ramp_cost charges every change in the flow between consecutive time steps. both adds the limit as well.
    ('ramp_method', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # [choice] Choice of ramp method. ramp_limit limits the ramp speed. ramp_cost charges every change in the flow between consecutive time steps. both adds the limit as well.
    ('ramp_speed_down', 'unit__inputNode'): QuantityType.RAMP_RATE,
    # [share of capacity per minute] Largest decrease of the flow between two time steps. No limit across a gap in the timeline. Constant.
    ('ramp_speed_down', 'unit__outputNode'): QuantityType.RAMP_RATE,
    # [share of capacity per minute] Largest decrease of the flow between two time steps. No limit across a gap in the timeline. Constant.
    ('ramp_speed_up', 'unit__inputNode'): QuantityType.RAMP_RATE,
    # [share of capacity per minute] Largest increase of the flow between two time steps. No limit across a gap in the timeline. Constant.
    ('ramp_speed_up', 'unit__outputNode'): QuantityType.RAMP_RATE,
    # [share of capacity per minute] Largest increase of the flow between two time steps. No limit across a gap in the timeline. Constant.
    ('reactance', 'connection'): QuantityType.DIMENSIONLESS,
    # [p.u.] Per-unit reactance of the transmission line. Used for DC power flow when the connection is within a nodeGroup that has transfer_metho...
    ('realized_invest_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of the periods that will realize the investment decisions. If not used when the invest_periods exist. The realized_periods are used to...
    ('realized_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of periods that will be realized in the solve.
    ('reference_node', 'group'): QuantityType.DIMENSIONLESS,
    # Name of the reference bus node (angle fixed to zero) for DC power flow. Optional — if not specified, automatically selected as the node with...
    ('reliability', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] The share of the reservation that is counted to reserves (sometimes reserve sources are not fully trusted). Constant.
    ('reliability', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] The share of the reservation that is counted to reserves (sometimes reserve sources are not fully trusted). Constant.
    ('reservation', 'reserve__upDown__group'): QuantityType.POWER,
    # [MW] Amount of reserve to be reserved. Constant or time.
    ('reserve_duration', 'reserve__upDown__group'): QuantityType.DURATION,
    # [h] Time span over which committed reserve power must be sustainable from storage. Constant.
    ('reserve_method', 'reserve__upDown__group'): QuantityType.DIMENSIONLESS,
    # Choice of reserve method: no_reserve, timeseries_only, dynamic_only, large_failure_only or their combinations. Dynamic and large_failure req...
    ('retire_max_period', 'connection'): QuantityType.POWER,
    # [MW] Largest allowed retirement in each period. Counts only what is retired in that period; existing capacity and investments do not change ...
    ('retire_max_period', 'node'): QuantityType.ENERGY,
    # [MWh] Largest allowed retirement of storage capacity in each period. Counts only what is retired in that period; existing capacity and inves...
    ('retire_max_period', 'unit'): QuantityType.POWER,
    # [MW] Largest allowed retirement in each period. Counts only what is retired in that period; existing capacity and investments do not change ...
    ('retire_max_total', 'connection'): QuantityType.POWER,
    # [MW] Largest allowed total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('retire_max_total', 'node'): QuantityType.ENERGY,
    # [MWh] Largest allowed total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('retire_max_total', 'unit'): QuantityType.POWER,
    # [MW] Largest allowed total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('retire_min_period', 'connection'): QuantityType.POWER,
    # [MW] Smallest required retirement in each period. Counts only what is retired in that period; existing capacity and investments do not chang...
    ('retire_min_period', 'node'): QuantityType.ENERGY,
    # [MWh] Smallest required retirement of storage capacity in each period. Counts only what is retired in that period; existing capacity and inv...
    ('retire_min_period', 'unit'): QuantityType.POWER,
    # [MW] Smallest required retirement in each period. Counts only what is retired in that period; existing capacity and investments do not chang...
    ('retire_min_total', 'connection'): QuantityType.POWER,
    # [MW] Smallest required total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('retire_min_total', 'node'): QuantityType.ENERGY,
    # [MWh] Smallest required total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('retire_min_total', 'unit'): QuantityType.POWER,
    # [MW] Smallest required total retirement over all solves. Investments and existing capacity do not change it. Constant.
    ('rolling_duration', 'solve'): QuantityType.DURATION,
    # Hours (Optional). Duration of rolling, if not stated, assumed to be the whole timeline of the solve
    ('rolling_solve_horizon', 'solve'): QuantityType.DURATION,
    # Hours (Required if rolling_window solve). How long into the future the roll sees
    ('rolling_solve_jump', 'solve'): QuantityType.DURATION,
    # Hours, (Required if rolling_window solve). Interval between the start points of the rolls. Should be smaller than the horizon
    ('salvage_value', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Salvage value for retiring capacity. Constant or period.
    ('salvage_value', 'node'): QuantityType.PRICE_PER_STORAGE,
    # [CUR/kWh] Salvage value of the storage. Constant or period.
    ('salvage_value', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Salvage value of the unit. Constant or period.
    ('scaling', 'solve'): QuantityType.DIMENSIONLESS,
    # Numerical scaling of this solve's LP. full (default): FlexTool rescales variables, constraints and costs and sets the solver's scaling. basi...
    ('self_discharge_loss', 'node'): QuantityType.FRACTION,
    # [e.g. 0.01 means 1% per hour] Share of the stored energy lost each hour. Constant, period, time or period+time.
    ('sense', 'constraint'): QuantityType.DIMENSIONLESS,
    # The sense of the constraint ('greater_than', 'less_than' or 'equal').
    ('share_loss_of_load', 'group'): QuantityType.DIMENSIONLESS,
    # Force the upward slack of the nodes in this group to be equal or inflow (demand) weighted
    ('solve_mode', 'solve'): QuantityType.DIMENSIONLESS,
    # A single_solve or rolling_window for a set of rolling optimisation windows solved in a sequence.
    ('solver', 'solve'): QuantityType.DIMENSIONLESS,
    # Choice of solver. HIGHs used as default. GLPSOL is another open source option. Out of commercial solvers CPLEX has been tested.
    ('solver_arguments', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of text commands for passing command line arguments to a solver. Can be used to set additional solver parameters.
    # HiGHS-specific config strings ('simplex'/'ipm'/'choose', 'on'/'off') — all DIMENSIONLESS toggles, no physical unit.
    ('highs_method', 'solve'): QuantityType.DIMENSIONLESS,
    ('highs_parallel', 'solve'): QuantityType.DIMENSIONLESS,
    ('highs_presolve', 'solve'): QuantityType.DIMENSIONLESS,
    # Cross-solver configuration knobs normalised by polar-high. All are config (strings/maps/integers) or unitless
    # tolerances/counts; none carry a physical [MW/MWh/CUR] unit.
    ('solver_io_api', 'solve'): QuantityType.DIMENSIONLESS,
    # 'direct' | 'mps' | 'lp' — how the model is handed to the solver. String choice.
    ('solver_log_level', 'solve'): QuantityType.DIMENSIONLESS,
    # 'silent' | 'normal' | 'verbose' — solver-log verbosity. String choice.
    ('solver_mip_gap', 'solve'): QuantityType.FRACTION,
    # Relative MIP optimality gap (dimensionless ratio in [0, 1]); classified as FRACTION because it is a unitless share.
    ('solver_options', 'solve'): QuantityType.DIMENSIONLESS,
    # Map of solver-specific option name -> raw value. Composite, no physical unit.
    ('solver_threads', 'solve'): QuantityType.DIMENSIONLESS,
    # Maximum number of solver worker threads — an integer count, not a physical quantity.
    ('solver_time_limit', 'solve'): QuantityType.DURATION,
    # Wall-clock time limit in seconds. DURATION carries time semantics; autoscaler treats it like other durations.
    ('solver_precommand', 'solve'): QuantityType.DIMENSIONLESS,
    # Not used: FlexTool never runs solver_precommand (kept so older databases still read).
    ('use_row_scaling', 'solve'): QuantityType.DIMENSIONLESS,
    # Boolean-style toggle ('yes'/'no') enabling experimental automatic row scaling. Flag, no physical unit.
    ('solves', 'model'): QuantityType.DIMENSIONLESS,
    # Sequence of solves in the model. Array.
    ('small_number_threshold', 'model'): QuantityType.DIMENSIONLESS,
    # Config scalar: LP coefficients/RHS below this magnitude are floored to 0. Not an LP coefficient itself.
    ('startup_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of starting up one MW of capacity. Constant or period.
    ('startup_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of starting up one MW of capacity. Constant or period.
    ('startup_energy', 'unit__inputNode'): QuantityType.DURATION,
    # [MWh per MW started] Energy taken from the input node per MW started (MWh/MW = h). Constant or period.
    ('startup_input_method', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # [choice] just_input / startup_and_input / startup_only: the arc also, or only, supplies the start-up energy.
    ('startup_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Gives the connection online capacity in both directions. Flows and reserves are limited by the online capacity, and startup_cost is paid per...
    ('startup_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Whether the unit has online variables. linear: any share of the capacity can be online. binary: whole units only (slower to solve). Online s...
    ('stochastic_branches', 'solve'): QuantityType.DIMENSIONLESS,
    # Forecast branches of the solve (map: period, branch, analysis_time, realized → probability weight). Weights are normalised per start time. A...
    ('stochastic_invest_method', 'solve'): QuantityType.DIMENSIONLESS,
    # How stochastic branches invest. none (default): one shared investment plan for all branches. recourse: each branch also invests at and after...
    ('dispatch_realization_method', 'solve'): QuantityType.DIMENSIONLESS,
    # How the committed part of a stochastic solve is realized. within_solve (default): a realized: yes branch gets a realization solve automatica...
    ('reduce_unit_detail_method', 'solve'): QuantityType.DIMENSIONLESS,
    # Detail of units and connections this solve leaves out (none, integers, ..., online_ramps).
    ('reduce_system_detail_method', 'solve'): QuantityType.DIMENSIONLESS,
    # System detail this solve leaves out (reserves, inertia, non_synchronous, dc_power_flow, combinations).
    ('storage_binding_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how the storage state will be maintained over discontinuous timelines. Seven cycle-scope methods (the state-continuity family): 'bind...
    ('storage_nested_fix_method', 'node'): QuantityType.DIMENSIONLESS,
    # Used in nested solve sequences. Set this storage as a long term storage, which end state is passed to the lower level solves as a target. *F...
    ('storage_solve_horizon_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how to treat storage state at the end of time horizon of each solve. 'free' lets the model choose the end state. 'use_reference_value...
    ('storage_start_end_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice whether the start and end states of storage are fixed in the beginning and end of the whole model timeline (not between solves). Uses...
    ('storage_state_end', 'node'): QuantityType.FRACTION,
    # [0-1] Storage state at the end of the last solve, as a share of the storage capacity. Needed by fix_end and fix_start_end. Constant.
    ('storage_state_reference_price', 'node'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Price for the stored energy at the end of the solve horizon. Requires 'use_reference_price' in 'storage_solve_horizon_method'. Con...
    ('storage_state_reference_value', 'node'): QuantityType.FRACTION,
    # [0-1] Relative state of storage at then end of each solve (can be overwritten in the next solve).  Requires 'use_reference_value' in 'storag...
    ('storage_state_start', 'node'): QuantityType.FRACTION,
    # [0-1] Storage state at the start of the first solve, as a share of the storage capacity. Needed by fix_start and fix_start_end. Constant.
    ('timeline', 'timeset'): QuantityType.DIMENSIONLESS,
    # The name of the timeline that the timeset uses. (String)
    ('timeline_hole_multiplier', 'solve'): QuantityType.DIMENSIONLESS,
    # [unitless] Multiplier applied to the inverse-step-duration term in nodeBalance_eq and storage-binding constraints across timeline gaps (hole...
    ('timeset_duration', 'timeset'): QuantityType.DURATION,
    # Index: name of the the timestep that starts the timeset, value: duration of the block in timesteps
    ('timeset_weights', 'timeset'): QuantityType.FRACTION,
    # Weight of each time step (map, index time). Weights how much each step counts in costs, energy and demand scaling. Normalised per period. Fo...
    ('representative_period_weights', 'timeset'): QuantityType.FRACTION,
    # Representative-period weighting map (rank-2 nested Map: base period start timestep -> representative period start timestep -> weight).  Crea...
    ('timestep_duration', 'timeline'): QuantityType.DURATION,
    # Map of time steps in the timeline. Index: time step name, value: time step duration in hours.
    ('transfer_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice of transfer method. Options: regular (default), no_losses_no_variable_cost, variable_cost_only, exact, unidirectional. 'unidirectiona...
    ('transfer_method', 'group'): QuantityType.DIMENSIONLESS,
    # Override transfer_method for all connections within this nodeGroup. Options: use_connection_transfer_methods (default, no override), no_loss...
    ('unitsize', 'commodity'): QuantityType.DIMENSIONLESS,
    # Numeric scaling for the v_trade variable column (analogous to virtual_unitsize on process/node entities).  The variable is in user-MWh divid...
    # Enable automatic row scaling (experimental): derive node_capacity_for_scaling / group_capacity_for_scaling from connected-unit unitsizes (ro...
    ('unit_of_measure', 'node'): QuantityType.DIMENSIONLESS,
    # [text] Unit of measure of the node's amounts (m3, t, MWh_th; default MWh). A label only: not read by the LP.
    ('use_for_representative_periods', 'group'): QuantityType.DIMENSIONLESS,
    # yes_no flag marking a group as an aggregation unit for net-load representative-period selection (preprocessor-only; not read by the LP model).
    ('version', 'model'): QuantityType.DIMENSIONLESS,
    # Contains database version information.
    ('virtual_unitsize', 'connection'): QuantityType.POWER,
    # [MW] Size of one unit. With it, existing is the number of units. Binary startups commit whole units of this size. Investments and retirement...
    ('virtual_unitsize', 'node'): QuantityType.ENERGY,
    # [MWh] Size of one unit. With it, existing is the number of units. Investments and retirements are continuous; their limits are in MWh. No va...
    ('virtual_unitsize', 'unit'): QuantityType.POWER,
    # [MW] Size of one unit. With it, existing is the number of units. Binary startups commit whole units of this size. Investments and retirement...
    ('years_represented', 'solve'): QuantityType.DURATION,
    # Map to indicate how many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple p...
}


def lookup(parameter_name: str, entity_class: str) -> QuantityType:
    """Return the QuantityType for a (parameter_name, entity_class) key.

    Raises KeyError if no type is registered.  The autoscaler MUST refuse to run
    on a parameter whose type is not declared here — silently defaulting risks
    rescaling currency as energy or vice-versa.
    """
    return PARAMETER_TYPES[(parameter_name, entity_class)]


#: The unit brackets a parameter description may start with, per quantity
#: (the bracket is the description up to its first ``]``).  The units are the
#: default convention (MW, MWh, CUR, tCO2); FlexTool does not check them.
#: ``DIMENSIONLESS`` lists the markers of non-numeric parameters and the
#: words of use-dependent units; it never accepts ``[-]`` (a parameter with
#: a true ratio there needs a :data:`UNIT_OVERRIDES` entry).
QUANTITY_UNITS: dict[QuantityType, frozenset[str]] = {
    QuantityType.ENERGY: frozenset({"[MWh]"}),
    QuantityType.POWER: frozenset({"[MW]"}),
    QuantityType.APPARENT_POWER: frozenset({"[MVA]"}),
    QuantityType.INERTIA: frozenset({"[MWs]"}),
    QuantityType.INERTIA_PER_CAPACITY: frozenset({"[MWs/MW]"}),
    QuantityType.CURRENCY: frozenset({"[CUR]"}),
    QuantityType.PRICE_PER_ENERGY: frozenset({"[CUR/MWh]"}),
    QuantityType.PRICE_PER_CAPACITY: frozenset({"[CUR/kW]", "[CUR/kW/a]", "[CUR/MW]"}),
    QuantityType.PRICE_PER_STORAGE: frozenset({"[CUR/kWh]", "[CUR/kWh/a]"}),
    QuantityType.PRICE_PER_MASS: frozenset({"[CUR/tCO2]"}),
    QuantityType.PRICE_PER_INERTIA: frozenset({"[CUR/MWs per hour]"}),
    QuantityType.EMISSION_MASS: frozenset({"[tCO2]", "[tCO2/a]"}),
    QuantityType.EMISSION_INTENSITY: frozenset({"[tCO2/MWh]"}),
    QuantityType.FRACTION: frozenset({"[-]"}),
    QuantityType.DURATION: frozenset({"[hours]", "[years]"}),
    QuantityType.RAMP_RATE: frozenset({"[share of capacity per minute]"}),
    QuantityType.DIMENSIONLESS: frozenset({
        "[choice]", "[yes/no]", "[name]", "[text]", "[count]", "[version number]", "[p.u.]",
        "[unit of the constraint per MW]", "[unit of the constraint per MWh]", "[unit of the constraint per unit online]",
        "[the constraint's own unit]",
    }),
    QuantityType.UNKNOWN: frozenset({"[MW or MWh]"}),
}

#: Parameters whose user unit is not the one their autoscale quantity implies,
#: keyed like :data:`PARAMETER_TYPES` (parameter, class) -> the exact bracket.
UNIT_OVERRIDES: dict[tuple[str, str], str] = {
    # A scaling divisor for autoscale, but the user value is an amount of the commodity.
    ("unitsize", "commodity"): "[MWh]",
    # Price-ladder tiers carry two quantities: the tier price and its quantity.
    ("price_ladder_annual", "commodity"): "[CUR/MWh; MWh/a]",
    ("price_ladder_cumulative", "commodity"): "[CUR/MWh; MWh]",
    ("price", "commodity"): "[CUR/MWh or other unit]",
    # A capacity, or a number of units with virtual_unitsize.
    ("existing", "unit"): "[MW or count]",
    ("existing", "connection"): "[MW or count]",
    ("existing", "node"): "[MWh or count]",
    ("annual_flow", "node"): "[MWh/a]",
    # A reserve shortfall is paid per MW and hour.
    ("penalty_reserve", "reserve__upDown__group"): "[CUR/MW per hour]",
    ("timeset_duration", "timeset"): "[count of time steps]",
    # True ratios typed DIMENSIONLESS (never an LP coefficient, or not scaled as a fraction).
    ("energy_margin_multiplier", "node"): "[-]",
    ("benders_in_out_weight", "solve"): "[-]",
    ("benders_tolerance", "solve"): "[-]",
    ("stochastic_branches", "solve"): "[-]",
    ("timeline_hole_multiplier", "solve"): "[-]",
    # Compared with every LP coefficient and right-hand side in its own unit.
    ("small_number_threshold", "model"): "[unit of each LP coefficient]",
    # An energy per capacity started (MWh/MW = h, typed DURATION): the unit says what it is per.
    ("startup_energy", "unit__inputNode"): "[MWh per MW started]",
}


# Parameters whose unit is ``[MW or MWh]`` — the description is context-dependent
# on whether the group's members are units/connections (POWER) or nodes (ENERGY).
# Layer 2 must call :func:`resolve_group_capacity_type` with the group's member
# entity-class to pick the correct QuantityType at apply time.
_GROUP_CAPACITY_PARAMS: frozenset[str] = frozenset({
    "cumulative_max_capacity",
    "cumulative_min_capacity",
    "invest_max_period",
    "invest_max_total",
    "invest_min_period",
    "invest_min_total",
    "retire_max_period",
    "retire_max_total",
    "retire_min_period",
    "retire_min_total",
})


def resolve_group_capacity_type(parameter_name: str, member_class: str) -> QuantityType:
    """Resolve `[MW or MWh]` group-level capacity parameters at apply time.

    Group capacity parameters carry ``[MW or MWh]``: ``POWER`` when the group
    aggregates ``unit`` / ``connection`` capacity, ``ENERGY`` when it aggregates
    ``node`` storage capacity.  Raises ``ValueError`` if called for a parameter
    that is not on the context-dependent list, and ``ValueError`` for an
    unrecognised member_class.
    """
    if parameter_name not in _GROUP_CAPACITY_PARAMS:
        raise ValueError(
            f"resolve_group_capacity_type: {parameter_name!r} is not a "
            f"context-dependent group capacity parameter"
        )
    if member_class in ("unit", "connection"):
        return QuantityType.POWER
    if member_class == "node":
        return QuantityType.ENERGY
    raise ValueError(
        f"resolve_group_capacity_type: unrecognised member_class {member_class!r} "
        f"for parameter {parameter_name!r}"
    )
