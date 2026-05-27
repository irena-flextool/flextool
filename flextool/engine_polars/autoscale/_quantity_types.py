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
    PRICE_PER_CAPACITY = "price_per_capacity"    # [CUR/kW], [CUR/MW]
    PRICE_PER_STORAGE = "price_per_storage"      # [CUR/kWh]
    PRICE_PER_MASS = "price_per_mass"            # [CUR/ton]
    PRICE_PER_INERTIA = "price_per_inertia"      # [CUR/MWs]
    EMISSION_MASS = "emission_mass"              # [tCO2]
    EMISSION_INTENSITY = "emission_intensity"    # [CO2 ton per MWh]
    FRACTION = "fraction"                        # [factor], [0-1], [%], availability/efficiency-style
    DURATION = "duration"                        # [hours], [years], "Hours."
    RAMP_RATE = "ramp_rate"                      # [per unit / minute]
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
    ('candidate_precapacity_to_avoid_big_m', 'group'): QuantityType.POWER,
    # [MW] Small pre-existing capacity assigned to investment candidate connections in DC power flow groups that have zero existing capacity. This...
    ('capacity_margin', 'group'): QuantityType.POWER,
    # [MW] How much capacity a node group is required to have in addition to the peak net load in the investment time series. Used only by the inv...
    ('capacity_max_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity available to this edge's upper cap (maxToSink / maxFromSource / ramp). For extraction ...
    ('capacity_max_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity available to this edge's upper cap (maxToSink / maxFromSource / ramp). For extraction ...
    ('capacity_min_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity imposed as a lower cap on this edge when online (combined multiplicatively with the un...
    ('capacity_min_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor, default 1.0] Fraction of the unit's capacity imposed as a lower cap on this edge when online (combined multiplicatively with the un...
    ('co2_content', 'commodity'): QuantityType.EMISSION_INTENSITY,
    # [CO2 ton per MWh] Constant.
    ('co2_max_period', 'group'): QuantityType.EMISSION_MASS,
    # [tCO2] Annualized maximum limit for emitted CO2 in each period.
    ('co2_max_total', 'group'): QuantityType.EMISSION_MASS,
    # [tCO2] Maximum limit for emitted CO2 in the whole solve.
    ('co2_method', 'group'): QuantityType.DIMENSIONLESS,
    # Choice of the CO2 method: no_method, price, period, total, price_period, price_total, period_total, price_period_total
    ('co2_price', 'group'): QuantityType.PRICE_PER_MASS,
    # [CUR/ton] CO2 price for a group of nodes. Constant, period or time.
    ('constant', 'constraint'): QuantityType.DIMENSIONLESS,
    # A constant offset for a user constraint (typically zero). The constant will be on the right side of the equation.
    # User constraints aggregate LHS terms whose units depend on the user's coefficient choices; the RHS constant lives
    # in that same composite coordinate. Layer-2 does not apply a per-row factor — column scalers propagate via the LHS.
    ('constraint_cumulative_pre_built_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places the cumulative pre-built capacity at period d — data baseline...
    ('constraint_cumulative_pre_built_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places the cumulative pre-built capacity at period d — data baseline...
    ('constraint_cumulative_pre_built_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places the cumulative pre-built capacity at period d — data baseline...
    ('constraint_flow_coeff', 'connection__node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow from the connection to a node ...
    ('constraint_flow_coeff', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow between unit and node in user-...
    ('constraint_flow_coeff', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the flow between unit and node in user-...
    ('constraint_invested_capacity_coeff', 'connection'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    ('constraint_invested_capacity_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    ('constraint_invested_capacity_coeff', 'unit'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (index: constraint name, value: coefficient) that places v_invest[e, d] — new-build capacity decided in the current pe...
    ('constraint_state_coeff', 'node'): QuantityType.DIMENSIONLESS,
    # A map of coefficients (Index: constraint name, value: coefficient) to represent the participation of the storage state in user-defined const...
    ('contains_solves', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of solves - used for nested solve sequencesArray of solves - used for nested solve sequences
    ('conversion_flow_coeff', 'unit__inputNode'): QuantityType.FRACTION,
    # [factor] Energy-unit conversion factor for this flow in the node balance and conversion_indirect equations. Value of 0 removes the edge from...
    ('conversion_flow_coeff', 'unit__outputNode'): QuantityType.FRACTION,
    # [factor] Energy-unit conversion factor for this flow in the node balance and conversion_indirect equations. Value of 0 removes the edge from...
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
    ('decomposition_method', 'group'): QuantityType.DIMENSIONLESS,
    # Decomposition strategy to apply to this group. Currently supported: 'none' (no decomposition — default), 'lagrangian_region' (group is solve...
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
    # [factor, typically between 0-1] Efficiency of a connection. Constant or time.
    ('efficiency', 'unit'): QuantityType.FRACTION,
    # [factor] Efficiency of a unit. Constant or time.
    ('efficiency_at_min_load', 'unit'): QuantityType.FRACTION,
    # [e.g. 0.4 means 40%] Efficiency of the unit at minimum load. Applies only if the unit has an online variable. Constant.
    ('existing', 'connection'): QuantityType.POWER,
    # [MW] Existing capacity. Constant or Period
    ('existing', 'node'): QuantityType.ENERGY,
    # [MWh] Existing storage capacity. Constant or Period
    ('existing', 'unit'): QuantityType.POWER,
    # [MW] Existing capacity. Constant or Period
    ('fix_storage_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of periods where the storage_values are fixed when the node has storage_include_solve_fix_method is set
    ('fixed_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Annual fixed cost. Constant or period.
    ('fixed_cost', 'node'): QuantityType.PRICE_PER_STORAGE,
    # [CUR/kWh] Annual fixed cost for storage. Constant or period.
    ('fixed_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Annual fixed cost. Constant or period.
    ('flow_aggregator', 'group'): QuantityType.DIMENSIONLESS,
    # Used with group_unit_node or group_connection_node to combine the flows when producing the dispatch output of a node group. Renamed from out...
    ('has_capacity_margin', 'group'): QuantityType.DIMENSIONLESS,
    # A flag whether the group of nodes has a capacity margin constraint in the investment mode.
    ('has_inertia', 'group'): QuantityType.DIMENSIONLESS,
    # A flag whether the group of nodes has an inertia constraint active.
    ('has_non_synchronous', 'group'): QuantityType.DIMENSIONLESS,
    # A flag whether the group of nodes has the non-synchronous share constraint active.
    ('highs_method', 'solve'): QuantityType.DIMENSIONLESS,
    # HiGHS solver method ('simplex' or 'ipm' which is interior point method). Should use 'choose' for MIP models, since 'simplex' and 'ipm' will ...
    ('highs_parallel', 'solve'): QuantityType.DIMENSIONLESS,
    # HiGHS parallelises single solves or not ('on' or 'off'). It can be better to turn HiGHS parallel off when executing multiple scnearios in pa...
    ('highs_presolve', 'solve'): QuantityType.DIMENSIONLESS,
    # HiGHS uses presolve ('on') or not ('off'). Can have a large impact on solution time when solves are large.
    ('include_stochastics', 'group'): QuantityType.DIMENSIONLESS,
    # Includes the stochastic branches to be used for the nodes/units/connections in this group
    ('increase_reserve_ratio', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] The reserve requirement is increased by the flow from the connection to the node multiplied by this ratio. Constant.
    ('increase_reserve_ratio', 'reserve__upDown__group'): QuantityType.FRACTION,
    # [factor] The reserve is increased by the sum of demands from the group members multiplied by this ratio. Constant.
    ('increase_reserve_ratio', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] The reserve requirement is increased by generation from the unit to the node multiplied by this ratio. Constant.
    ('inertia_constant', 'unit__inputNode'): QuantityType.INERTIA_PER_CAPACITY,
    # [MWs/MW] Inertia constant for a synchronously connected unit to this node. Constant.
    ('inertia_constant', 'unit__outputNode'): QuantityType.INERTIA_PER_CAPACITY,
    # [MWs/MW] Inertia constant for a synchronously connected unit to this node. Constant.
    ('inertia_limit', 'group'): QuantityType.INERTIA,
    # [MWs] Minimum for synchronous inertia in the group of nodes. Constant or period.
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
    ('invest_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Investment cost for new 'virtual' capacity. Constant or period.
    ('invest_cost', 'node'): QuantityType.PRICE_PER_STORAGE,
    # [CUR/kWh] Investment cost for new storage capacity. Constant or period.
    ('invest_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Investment cost of the unit. Constant or period.
    ('invest_forced', 'node'): QuantityType.ENERGY,
    # (empty description) — inferred from ('existing', 'node') = [MWh]: forced storage-capacity investment.
    ('invest_max_period', 'connection'): QuantityType.POWER,
    # [MW] Maximum investment. Period.
    ('invest_max_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Maximum investment per period to the virtual capacity of a group of units or to the storage capacity of a group of nodes. Period...
    ('invest_max_period', 'node'): QuantityType.ENERGY,
    # [MWh] Maximum storage investment. Period.
    ('invest_max_period', 'unit'): QuantityType.POWER,
    # [MW] Maximum investment. Period.
    ('invest_max_total', 'connection'): QuantityType.POWER,
    # [MW] Maximum investment over all solves. Constant.
    ('invest_max_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Maximum investment to the virtual capacity of a group of units or to the storage capacity of a group of nodes. Total over all so...
    ('invest_max_total', 'node'): QuantityType.ENERGY,
    # [MWh] Maximum investment over all solves. Constant.
    ('invest_max_total', 'unit'): QuantityType.POWER,
    # [MW] Maximum investment over all solves. Constant.
    ('invest_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: not_allowed, invest and retire indicate availability of investment and retirement. no_limit removes all limits ...
    ('invest_method', 'group'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: not_allowed, invest and retire indicate availability of investment and retirement. no_limit removes all limits ...
    ('invest_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: either not_allowed or then a combination of 1) investment or retirement and 2) investment limits per period, al...
    ('invest_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice of investment method: not_allowed, invest and retire indicate availability of investment and retirement. no_limit removes all limits ...
    ('invest_min_period', 'connection'): QuantityType.POWER,
    # [MW] Minimum investment. Period.
    ('invest_min_period', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Minimum investment per period to the virtual capacity of a group of units or to the storage capacity of a group of nodes. Period...
    ('invest_min_period', 'node'): QuantityType.ENERGY,
    # [MWh] Minimum storage investment. Period.
    ('invest_min_period', 'unit'): QuantityType.POWER,
    # [MW] Minimum investment. Period.
    ('invest_min_total', 'connection'): QuantityType.POWER,
    # [MW] Minimum investment over all solves. Constant.
    ('invest_min_total', 'group'): QuantityType.UNKNOWN,
    # [MW or MWh] Minimum investment to the virtual capacity of a group of units or to the storage capacity of a group of nodes. Total over all so...
    ('invest_min_total', 'node'): QuantityType.ENERGY,
    # [MWh] Minimum investment over all solves. Constant.
    ('invest_min_total', 'unit'): QuantityType.POWER,
    # [MW] Minimum investment over all solves. Constant.
    ('invest_periods', 'solve'): QuantityType.DIMENSIONLESS,
    # Array of periods where investments are allowed.
    ('is_DC', 'connection'): QuantityType.DIMENSIONLESS,
    # A flag whether the connection is DC (the flow will not be counted as synchronous if there is a non_synchronous_limit). Default false.
    ('is_non_synchronous', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # Chooses whether the unit is synchronously connected to this node.
    ('is_non_synchronous', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # Chooses whether the unit is synchronously connected to this node.
    ('large_failure_ratio', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] Each connection using the N-1 failure method will have a separate constraint to require sufficient reserve to cover a failure of th...
    ('large_failure_ratio', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] Each unit using the N-1 failure method will have a separate constraint to require sufficient reserve to cover a failure of the unit...
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
    ('max_cumulative_flow', 'group'): QuantityType.POWER,
    # [MW] Maximum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multipli...
    ('max_flow_for_unconstrained_variables', 'model'): QuantityType.POWER,
    # [MW] Upper bound assigned to LP variables that have no other cap (invest_no_limit capacity; flows through edges whose capacity_max_coeff is ...
    ('max_instant_flow', 'group'): QuantityType.POWER,
    # [MW] Maximum instantenous flow for the aggregated flow of all group members. Constant or period.
    ('max_share', 'reserve__upDown__connection__node'): QuantityType.FRACTION,
    # [factor] Maximum ratio for the transfer of reserve to this node. Constant.
    ('max_share', 'reserve__upDown__unit__node'): QuantityType.FRACTION,
    # [factor] Maximum ratio for the transfer of reserve to this node. Constant.
    ('min_cumulative_flow', 'group'): QuantityType.POWER,
    # [MW] Minimum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multipli...
    ('min_downtime', 'unit'): QuantityType.DURATION,
    # [hours] Minimum time the unit must stay offline after shutting down. Requires minimum_time_method set to 'min_downtime' or 'both'. Constant.
    ('min_instant_flow', 'group'): QuantityType.POWER,
    # [MW] Minimum instantenous flow for the aggregated flow of all group members. Constant or period.
    ('min_load', 'unit'): QuantityType.FRACTION,
    # [0-1] Minimum load of the unit. Applies only if the unit has an online variable. With linear startups, it is the share of capacity started u...
    ('min_uptime', 'unit'): QuantityType.DURATION,
    # [hours] Minimum time the unit must stay online after starting up. Requires minimum_time_method set to 'min_uptime' or 'both'. Constant.
    ('minimum_time_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice between minimum up- and downtimes (none, min_downtime, min_uptime, both). Setting this to anything other than 'none' will activate on...
    ('new_stepduration', 'group'): QuantityType.DURATION,
    # Hours. Members of this group operate at this step duration. Overrides the solve-level new_stepduration for these entities. Used for multi-re...
    ('new_stepduration', 'solve'): QuantityType.DURATION,
    # Hours. Creates a new `timeline` from the old for this `solve` with this timestep duration. The new timeline will sum or average the other ti...
    ('node_type', 'node'): QuantityType.DIMENSIONLESS,
    # Role of this node in the LP.  'commodity' = price-exposed source/sink with no balance constraint (e.g. fuel imports, no storage); 'balance' ...
    ('non_synchronous_limit', 'group'): QuantityType.FRACTION,
    # [share, e.g. 0.8 means 80%] The maximum share of non-synchronous generation in the node group. Constant or period.
    ('other_operational_cost', 'connection'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, Period or time.
    ('other_operational_cost', 'unit__inputNode'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.
    ('other_operational_cost', 'unit__outputNode'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.
    ('output_connection__node__node_flow_t', 'model'): QuantityType.DIMENSIONLESS,
    # The flows between the nodes for each timestep.
    ('output_connection_flow_separate', 'model'): QuantityType.DIMENSIONLESS,
    # Produces the connection flows separately for both directions.
    ('output_flowGroup_indicators', 'group'): QuantityType.DIMENSIONLESS,
    # Flag to output flow-group indicator results for groups whose members are flows (group__unit__node or group__connection__node).
    ('output_horizon', 'model'): QuantityType.DIMENSIONLESS,
    # Outputs the flows in the horizons. Used for testing the model.
    ('output_nodeGroup_dispatch', 'group'): QuantityType.DIMENSIONLESS,
    # Creates the timewise flow output for this node group (node-group dispatch table). Renamed from output_node_flows.
    ('output_nodeGroup_indicators', 'group'): QuantityType.DIMENSIONLESS,
    # Flag to output node-group indicator results for groups whose members are nodes (group__node).
    ('output_unit__node_flow_t', 'model'): QuantityType.DIMENSIONLESS,
    # The flows from units to the nodes for each timestep.
    ('output_unit__node_ramp_t', 'model'): QuantityType.DIMENSIONLESS,
    # Produces the ramps of individual units for all timesteps.
    ('peak_inflow', 'node'): QuantityType.ENERGY,
    # [MWh] Highest flow for scaling the inflow. Used only with inflow_method scale_to_annual_and_peak_flow. Constant or period.
    ('penalty_capacity_margin', 'group'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/kW] Penalty for violating the capacity margin constraint. Uses operational discounting (not annualized over lifetime like investment co...
    ('penalty_down', 'node'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, Period or Time.
    ('penalty_inertia', 'group'): QuantityType.PRICE_PER_INERTIA,
    # [CUR/MWs] Penalty for violating the inertia constraint. Cost scales with the duration of the violation. Constant or period.
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
    # Stepped supply curve for price_method='price_ladder_annual'.  Two forms accepted: 2d map with rows 'tier,price,quantity' applies the same pe...
    ('price_ladder_cumulative', 'commodity'): QuantityType.DIMENSIONLESS,
    # Stepped supply curve for price_method='price_ladder_cumulative'.  2d map with rows 'tier,price,quantity' — one row per tier, giving the tier...
    ('price_method', 'commodity'): QuantityType.DIMENSIONLESS,
    # How the commodity's price enters the LP.  'price' = scalar or time-series price x flow (current behaviour); 'price_ladder_annual' = stepped ...
    ('profile', 'profile'): QuantityType.FRACTION,
    # [factor, typically between 0-1]. Availability time series for a fluctuating resource like wind power. Time.
    ('profile_method', 'connection__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('profile_method', 'node__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('profile_method', 'unit__node__profile'): QuantityType.DIMENSIONLESS,
    # Choice of profile method (upper_limit, lower_limit, fixed). Negative values also possible.
    ('ramp_cost', 'unit__inputNode'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of ramping the unit. Constant.
    ('ramp_cost', 'unit__outputNode'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of ramping the unit. Constant.
    ('ramp_method', 'unit__inputNode'): QuantityType.DIMENSIONLESS,
    # Choice of ramp method. 'ramp_limit' poses a limit on the speed of ramp. 'ramp_cost' poses a cost on ramping the flow (NOT FUNCTIONAL AS OF 1...
    ('ramp_method', 'unit__outputNode'): QuantityType.DIMENSIONLESS,
    # Choice of ramp methods.
    ('ramp_speed_down', 'unit__inputNode'): QuantityType.RAMP_RATE,
    # [per unit / minute] Maximum ramp down speed. Constant.
    ('ramp_speed_down', 'unit__outputNode'): QuantityType.RAMP_RATE,
    # [per unit  / minute] Maximum ramp down speed. Constant.
    ('ramp_speed_up', 'unit__inputNode'): QuantityType.RAMP_RATE,
    # [per unit / minute] Maximum ramp up speed. Constant.
    ('ramp_speed_up', 'unit__outputNode'): QuantityType.RAMP_RATE,
    # [per unit  / minute] Maximum ramp up speed. Constant.
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
    ('reserve_method', 'reserve__upDown__group'): QuantityType.DIMENSIONLESS,
    # Choice of reserve method: no_reserve, timeseries_only, dynamic_only, large_failure_only, timeseries_and_dynamic, timeseries_and_large_failur...
    ('retire_max_period', 'connection'): QuantityType.POWER,
    # [MW] Maximum retired capacity. Period.
    ('retire_max_period', 'node'): QuantityType.ENERGY,
    # [MWh] Maximum retired storage capacity. Period.
    ('retire_max_period', 'unit'): QuantityType.POWER,
    # [MW] Maximum retired capacity. Period.
    ('retire_max_total', 'connection'): QuantityType.POWER,
    # [MW] Maximum retired capacity over all solves. Constant.
    ('retire_max_total', 'node'): QuantityType.ENERGY,
    # [MWh] Maximum retired storage capacity over all solves. Constant.
    ('retire_max_total', 'unit'): QuantityType.POWER,
    # [MW] Maximum retired capacity over all solves. Constant.
    ('retire_min_period', 'connection'): QuantityType.POWER,
    # [MW] Maximum retired capacity. Period.
    ('retire_min_period', 'node'): QuantityType.ENERGY,
    # [MWh] Minimum retired storage capacity. Period.
    ('retire_min_period', 'unit'): QuantityType.POWER,
    # [MW] Minimum retired capacity. Period.
    ('retire_min_total', 'connection'): QuantityType.POWER,
    # [MW] Minimum retired capacity over all solves. Constant.
    ('retire_min_total', 'node'): QuantityType.ENERGY,
    # [MWh] Minimum retired storage capacity over all solves. Constant.
    ('retire_min_total', 'unit'): QuantityType.POWER,
    # [MW] Minimum retired capacity over all solves. Constant.
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
    ('self_discharge_loss', 'node'): QuantityType.FRACTION,
    # [e.g. 0.01 means 1% every hour] Loss of stored energy over time. Constant or time.
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
    ('solver_precommand', 'solve'): QuantityType.DIMENSIONLESS,
    # Additional command to execute before calling the solver. Can be used to e.g. reserve a floating license for a commercial solver.
    ('solves', 'model'): QuantityType.DIMENSIONLESS,
    # Sequence of solves in the model. Array.
    ('startup_cost', 'connection'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of starting up one MW of 'virtual' capacity. Constant.
    ('startup_cost', 'unit'): QuantityType.PRICE_PER_CAPACITY,
    # [CUR/MW] Cost of starting up one MW of 'virtual' capacity. Constant.
    ('startup_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice of startup method
    ('startup_method', 'unit'): QuantityType.DIMENSIONLESS,
    # Choice of startup method. Linear startup means that the unit can start partially (anything between 0 and full capacity) but will face startu...
    ('stochastic_branches', 'solve'): QuantityType.DIMENSIONLESS,
    # [4d-Map], Sets branches included in the solve. [Period, branch, start_time (time_step), realized (yes/no), weight (number)]. Only one of the...
    ('storage_binding_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how the storage state will be maintained over discontinuos timelines. The default value 'bind_forward_only' will bind forward over an...
    ('storage_nested_fix_method', 'node'): QuantityType.DIMENSIONLESS,
    # Used in nested solve sequences. Set this storage as a long term storage, which end state is passed to the lower level solves as a target. *F...
    ('storage_solve_horizon_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice how to treat storage state at the end of time horizon of each solve. 'free' lets the model choose the end state. 'use_reference_value...
    ('storage_start_end_method', 'node'): QuantityType.DIMENSIONLESS,
    # Choice whether the start and end states of storage are fixed in the beginning and end of the whole model timeline (not between solves). Uses...
    ('storage_state_end', 'node'): QuantityType.FRACTION,
    # [0-1] Relative state of storage at the end of the last model solve (overrides 'storage_state_end_reference'). Constant.
    ('storage_state_reference_price', 'node'): QuantityType.PRICE_PER_ENERGY,
    # [CUR/MWh] Price for the stored energy at the end of the solve horizon. Requires 'use_reference_price' in 'storage_solve_horizon_method'. Con...
    ('storage_state_reference_value', 'node'): QuantityType.FRACTION,
    # [0-1] Relative state of storage at then end of each solve (can be overwritten in the next solve).  Requires 'use_reference_value' in 'storag...
    ('storage_state_start', 'node'): QuantityType.FRACTION,
    # [0-1] Relative state of storage at the beginning of the first model solve (irrespective of when the model starts). Constant.
    ('timeline', 'timeset'): QuantityType.DIMENSIONLESS,
    # The name of the timeline that the timeset uses. (String)
    ('timeline_hole_multiplier', 'solve'): QuantityType.DIMENSIONLESS,
    # [unitless] Multiplier applied to the inverse-step-duration term in nodeBalance_eq and storage-binding constraints across timeline gaps (hole...
    ('timeset_duration', 'timeset'): QuantityType.DURATION,
    # Index: name of the the timestep that starts the timeset, value: duration of the block in timesteps
    ('timeset_weights', 'timeset'): QuantityType.FRACTION,
    # Per-timestep weight map (index: timestep name, value: float) applied to cost and slack terms in the objective. Use for non-RP models where t...
    ('timestep_duration', 'timeline'): QuantityType.DURATION,
    # Map of time steps in the timeline. Index: time step name, value: time step duration in hours.
    ('transfer_method', 'connection'): QuantityType.DIMENSIONLESS,
    # Choice of transfer method. Options: regular (default), no_losses_no_variable_cost, variable_cost_only, exact, unidirectional. 'unidirectiona...
    ('transfer_method', 'group'): QuantityType.DIMENSIONLESS,
    # Override transfer_method for all connections within this nodeGroup. Options: use_connection_transfer_methods (default, no override), no_loss...
    ('unitsize', 'commodity'): QuantityType.DIMENSIONLESS,
    # Numeric scaling for the v_trade variable column (analogous to virtual_unitsize on process/node entities).  The variable is in user-MWh divid...
    ('use_row_scaling', 'solve'): QuantityType.DIMENSIONLESS,
    # Enable automatic row scaling (experimental): derive node_capacity_for_scaling / group_capacity_for_scaling from connected-unit unitsizes (ro...
    ('version', 'model'): QuantityType.DIMENSIONLESS,
    # Contains database version information.
    ('virtual_unitsize', 'connection'): QuantityType.POWER,
    # [MW] Size of single connection - used for integer (lumped) investments.
    ('virtual_unitsize', 'node'): QuantityType.ENERGY,
    # [MWh] Size of a single storage unit - used for integer investments (lumped investments). If not given, assumed from the existing storage cap...
    ('virtual_unitsize', 'unit'): QuantityType.POWER,
    # [MW] Size of a single unit - used for integer investments (lumped investments), minimum loads and start-up costs. If not given, assumed from...
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
