# Connections

Connections can transfer energy between two nodes. Parameters for the connection are defined in the `connection` entity, but the two `nodes` it connects are defined by establishing an entity between `connection--leftNode--rightNode`.

## Defining how the connection functions

- `transfer_method` to define the way the connection transfers energy between the nodes
- `startup_method` gives the connection online capacity. *linear*: the online capacity is anything between 0 and the built capacity; *binary*: whole units of the connection (of `virtual_unitsize`, else the whole connection), computationally more demanding. The connection's flow in each direction is limited by its online capacity × availability, its reserve by its online capacity, and `startup_cost` is paid per MW brought online. Connections have no minimum load or minimum up/down times. Not available for connections in a `dc_power_flow_with_angles` group, other than `is_DC` connections (an error: switching a DC power-flow line off would need line-switching constraints).
- `invest_method` to define investment and retirement limits: either *not_allowed* or then a combination of 
    - *invest* and/or *retire* 
    - investment limits for each *period* and/or for all periods (*total*) or *no_limit* 
    - *cumulative_limits* considers investments, retirements and existing together, requires `cumulative_max_capacity` and/or `cumulative_min_capacity`
- `lifetime_method` to choose how the investments behave after unit runs out of lifetime. Automatic reinvestment (reinvest_automatic - default) causes the model to keep the capacity until the end of model horizon and applies the annualized investment cost until the end of model horizon without further choice by the model. Choice of reinvestment (reinvest_choice) removes the capacity at the end of the lifetime and the model needs to decide how much new capacity is to be built. One-shot investment (no_investment) allows the model to invest freely within the first-period lifetime window (subject to the usual invest_method / invest_max_period caps) but pins v_invest to zero for any period after that window — the asset retires once and cannot be rebuilt. Use for physical one-off additions such as plant refurbishments that cannot be repeated at the same cost. If there is a need to remove the possibility to invest after lifetime, then the investment limits can be used.

## Main data items for connections

- `existing` - [MW or count] Existing capacity or number of units. If `virtual_unitsize` is given, then `existing` is the number of units (capacity = `existing` × `virtual_unitsize`). Otherwise `existing` is the existing capacity of a single unit (typically in MW). Constant or period.
- `efficiency` - [factor, typically between 0-1] Efficiency of a connection. Constant, period, time or period+time.
- `delay` - [hours] Not yet applied to connections (planned). A connection `delay` is ignored: the connection is modelled exactly as if no delay were set (the flow arrives in the same time step) and the model prints a warning naming the connection. To model a delayed flow between two nodes, use a unit with `delay` instead.
- `startup_cost` - [CUR/MW] Cost of starting up one MW of capacity (used when `startup_method` activates online variables on the connection). Constant or period.
- `reactance` - [p.u.] Per-unit reactance of the transmission line. Used for DC power flow when the enclosing `group__node` has `transfer_method = dc_power_flow_with_angles`.
- `constraint_invested_capacity_coeff` - A map of coefficients (Index: constraint name, value: coefficient) putting the capacity that comes online in each period on the left side of the user-defined constraint. Without a construction lead time, this is the capacity invested in that period. Coefficient per MW. Renamed from `constraint_capacity_coefficient`; the old expression summed `v_invest` once per active investment period, giving incorrect results in multi-period models.
- `constraint_cumulative_pre_built_capacity_coeff` - A map of coefficients (Index: constraint name, value: coefficient). Puts the capacity in service at the start of each period on the left side of the user constraint. This is existing capacity plus capacity that came online in earlier periods, minus earlier retirements. Capacity coming online in the period itself is in the invested term. Enables learning-effect and period-over-period growth limits (pair with `constraint_invested_capacity_coeff` on the same constraint). Coefficient per MW.
- `constraint_capacity_coeff` - A map of coefficients (Index: constraint name, value: coefficient). Puts the connection's capacity in service in each period on the left side of the user constraint (existing plus investments still in service minus retirements). Coefficient per MW. Capacity in construction is not included.
- `constraint_retired_capacity_coeff` - A map of coefficients (Index: constraint name, value: coefficient). Puts the capacity retired in each period on the left side of the user constraint. Coefficient per MW.
- `constraint_in_construction_capacity_coeff` - A map of coefficients (Index: constraint name, value: coefficient). Puts the capacity in construction in each period on the left side of the user constraint. Capacity is in construction from the period of the investment decision until it comes online. It is zero without a construction lead time. Coefficient per MW.
- `other_operational_cost` - [CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, period, time or period+time.
- `fixed_cost` - [CUR/kW/a] Annual fixed cost. New capacity pays it only while it operates. Constant or period.
- `invest_cost` - [CUR/kW] Overnight investment cost of new capacity. It is paid as an annuity over the years the capacity operates. Construction time costs with their interest must be included here. Constant or period: the value of the period when the capacity is built (comes online) is used.
- `discount_rate` - [e.g. 0.05 equals 5%] Discount rate for investments (WACC). Reflects the financing cost and risk for this technology. Should be nominal when model `inflation_rate` > 0, real when `inflation_rate` = 0. Constant or period.
- `lifetime` - [years] Operating lifetime. It starts when the capacity starts to operate: with a construction lead time, after the construction. Used with the discount rate to calculate the annuity. What happens at the end of the lifetime depends on `lifetime_method`: *reinvest_automatic* keeps the capacity to the end of the model horizon, *reinvest_choice* retires it, *no_investment* retires it once. The `existing` capacity ends at the first year of the model run plus the `lifetime` of the first period (in every solve of a chain of solves). Constant or period.
- `construction_lead_time` - [years] Time from the investment decision until the capacity starts to operate. The start is rounded to the start of a period (see `construction_lead_time_method`). Investments whose rounded start is after the end of the solve or the model horizon are not offered, with a warning. Constant or period. Default 0. See [construction lead time](../how_to/investment.md#how-to-make-investments-storageunit).
- `construction_lead_time_method` - How the start of operation (decision year plus `construction_lead_time`) is rounded to the start of a period: *immediate* (no lead time, the capacity operates from the decision period), *closest_seam* (the nearest period start; a tie goes to the later one), *previous_seam* (the period start before), *next_seam* (the period start after). Without a value: *immediate* when there is no lead time, else *closest_seam*.
- other investment parameters: `invest_max_total`, `invest_max_period`, `invest_min_total`, `invest_min_period`, `salvage_value` (only `existing` capacity can be retired, see [retirements](units.md#retirements))
- `is_DC` - A flag whether the connection is DC: its flow into a node of a group with a *non_synchronous_limit* counts as non-synchronous (both directions; between two nodes of the same group it does not count). Default false. An `is_DC` connection inside a `dc_power_flow_with_angles` group is not a DC power-flow line: it keeps its own `transfer_method` and needs no `reactance`.
- `virtual_unitsize` - [MW] Size of one unit of the connection. With it, `existing` is the number of units. Binary startups commit whole units of this size. Investments and retirements are continuous; their limits are in MW. No value: the existing capacity, or 1000 MW without it.
- `availability` - [e.g. 0.9 means 90%] Fraction of capacity available for connection flows. Constant, period, time or period+time.

## Investment parameters for connections

These are the same as for units, see [here](units.md#investment-parameters-for-capacity-expansion)

## connection__node (per-endpoint properties)

On the membership entity between a `connection` and one of its endpoint `node`s:

- `constraint_flow_coeff` - A map of coefficients (Index: constraint name, value: coefficient) placing this connection-to-node flow variable on the left side of a user-defined `constraint`. Applied per timestep.
