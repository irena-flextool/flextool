# Model limitations

FlexTool is a linear (or mixed-integer) optimisation model of an energy system. Like every model, it leaves some things out and simplifies others. This page lists the limitations that matter most in planning studies, with the usual way around them. Read it before you trust a result that depends on one of these features.

## Emissions

- **CO2 is the only emission.** Each commodity has one constant `co2_content` [tCO2/MWh]; other pollutants are not tracked.

    The emissions of a group come from the fuel its units draw from the group's commodity nodes. A cap or a price
    for a sector therefore needs separate fuel nodes for that sector. The emissions of a unit with several
    products (combined heat and power, cogeneration with desalination) are not split between the products.

- **A CO2 limit has no slack.** If the system cannot meet `co2_max_period` or `co2_max_total`, the model leaves demand unserved at the node's `penalty_up` instead (or is infeasible when the node's `penalty_method` is *off*).

    Check the slack results of a run with a tight limit; see
    [How to compare emission trajectories](how_to/emissions.md#how-to-compare-emission-trajectories).

## Thermal units and unit commitment

- **Linear commitment is an approximation.** With `startup_method` *linear*, a part of a unit can be online: with `min_load` 0.5, a 500 MW unit can run at 25 MW with 50 MW (10 %) of it online.

    Minimum load and start-up costs apply only to the part that is online, so a large unit looks more flexible and
    cheaper to cycle than it is. *binary* commits whole units (a mixed-integer problem): it is exact, but slower.
    `virtual_unitsize` splits a fleet into smaller units to keep a *binary* model fast.

- **No start-up fuel.** A start-up costs only `startup_cost` [CUR/MW]: it uses no fuel or heat, and there is no shut-down cost and no hot, warm or cold start.
- **Fuel use is linear in the output.** With `conversion_method` *min_load_efficiency*, the fuel use is a straight line from minimum load to full load; there are no curved efficiency curves.
- **No parameter that keeps a unit online.** A unit can be made to produce (a profile, a flow group floor), but there is no parameter that fixes its online status alone.

    A user constraint on the units online (`constraint_online_coeff`) sets a minimum number of units online; see
    [How to make a unit run all the time](how_to/thermal_chp.md#how-to-make-a-unit-run-all-the-time-must-run).

## Inertia and reserves

- **Inertia of units without online variables comes from their flow.** A unit without online variables (no `startup_method` and no `minimum_time_method`) gives flow × `inertia_constant`, so a unit at half load gives half of its inertia.

    Give the synchronous units online variables (`startup_method` *linear* or *binary*) to count their inertia from
    the capacity online; see [How to set a minimum inertia](how_to/emissions.md#how-to-set-a-minimum-inertia).

- **The inertia limit is entered, not computed.** FlexTool reports the largest single flow into the group, but the `inertia_limit` does not follow it.
- **Upward reserve comes from online capacity.** A unit with online variables holds upward reserve only from the capacity it has online. There is no separate reserve from offline units that can start fast.
- **Reserve activation is not modelled.** The model holds the reserve capacity back; the energy of activating it and its cost are not part of the dispatch.

## Investment and adequacy

- **The dispatch does not feed back into the investment.** An investment solve does not see the results of the dispatch solves that follow it.

    If a dispatch check shows unserved energy or reserve shortfall, change the investment solve by hand: a better
    sample of representative periods, a `capacity_margin` or an energy margin, and run again; see
    [Several milestones](how_to/solve_sequences.md#several-milestones-representative-period-investment-then-a-full-year-dispatch-of-each).
    The net-load re-selection of representative periods is the one loop that FlexTool runs for you.

- **The capacity margin counts units at their capacity.** A unit counts with its full capacity (a unit with a profile at capacity × profile), whatever its `availability`.

    A unit that runs from a storage, such as a CSP turbine fed by its heat tank or a hydro turbine, counts in full
    even when the storage is empty. Connections do not count, so neither do imports nor a battery modelled with a
    connection.

- **Representative periods leave hours out.** An investment made on a sample of weeks cannot see the hours outside the sample; check the plan with a full-year dispatch.
- **Costs count only within the model horizon.** The annuity and the fixed cost of new capacity are paid in the periods it operates within the model; the years after the last period are not counted.

## Discounting

- **Only `inflation_rate` discounts between periods.** With the default 0, a cost in the last milestone weighs the same as in the first.

    In a model with real (constant-price) inputs, enter the social discount rate as `inflation_rate`. The
    per-technology `discount_rate` only turns investment costs into annuities; see
    [Discounting between milestone years](advanced_concepts.md#discounting-between-milestone-years).

## Storage

- **The standing loss is proportional to the content.** `self_discharge_loss` is a share of the stored energy per hour. A loss that depends on the size of the storage (for example the heat loss of a hot tank) cannot be given.

## User constraints and policy targets

- **Constraints on flows hold in every time step.** A user constraint with flow, state or online terms is written for each time step, and its `constant` is a constant or a value per period, not per time step.

    There is no user constraint on a period's total flows, such as "renewables make at least 30 % of the year's
    electricity". The cumulative limits of a flow group (`min_cumulative_flow`, `max_cumulative_flow`) are
    absolute amounts (an average flow in MW), not shares: compute the amount outside FlexTool. See
    [User constraints](reference/user_constraints.md#rows-per-period-or-per-time-step) and
    [Flow groups](reference/flow_groups.md#cumulative-and-instant-flow-limits).

## Networks

- **No AC power flow.** A connection moves energy with losses as a share of the flow, or the network uses a linear DC power flow (`dc_power_flow_with_angles`). Voltages, reactive power and AC losses are not modelled.

## Time, foresight and uncertainty

- **Perfect foresight within a solve.** A solve knows its whole horizon, so storage and investments are planned with full knowledge of the future. Rolling and nested solves limit what each solve sees; see [Solve sequences](how_to/solve_sequences.md).
- **Forecasts differ only in time series.** In a stochastic solve, the forecasts can differ only in the parameters marked "stochastic time" in the reference (profiles, inflow, availability and a few others); see [How to use stochastics](how_to/stochastics.md).

## Mathematics and solvers

- **Linear or mixed-integer only.** Every relation in the model is linear. A nonlinear effect has to be made linear (for example with several units or a profile) before it goes in.
- **Solver.** The free HiGHS solver comes with FlexTool; commercial solvers can be used through [Solvers](solvers/index.md). A large mixed-integer model can take long with any solver.
