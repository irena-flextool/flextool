# User constraints

A user constraint is a linear row that you write yourself. It ties flows, storage states, units online and capacities of nodes, units and connections together, for example a fixed ratio between two capacities or a minimum share of a flow.

## The `constraint` entity

- `constraint`: a user-defined linear constraint between flow, state, units-online and capacity variables on nodes, units and connections. The left-hand side is assembled by tagging the relevant entities with the matching coefficient parameter (`constraint_flow_coeff` on `unit__inputNode` / `unit__outputNode` / `connection__node`, `constraint_state_coeff` on `node`, `constraint_online_coeff` on `unit` / `connection`, `constraint_invested_capacity_coeff` / `constraint_in_construction_capacity_coeff` / `constraint_cumulative_pre_built_capacity_coeff` / `constraint_capacity_coeff` / `constraint_retired_capacity_coeff` on `node` / `unit` / `connection`); the constraint entity itself carries:

    - `sense` - The sense of the constraint: `greater_than`, `less_than`, or `equal`.
    - `constant` - Right-hand side of the user constraint (default 0). A period map gives each period its own value; it must cover every period of the solve (a missing period is an error). Constant or period.
    - `constraint_period_method` - `per_timestep` (default) or `period_total`: whether the constraint holds in each time step or once for the total of each period (see [Rows per period or per time step](#rows-per-period-or-per-time-step)).

  Capacity terms, all in MW (MWh for storage nodes) and multiplied by the coefficient: *invested* (`constraint_invested_capacity_coeff`) is the capacity that comes online in the period, *in construction* (`constraint_in_construction_capacity_coeff`) the capacity decided but not yet operating, *retired* (`constraint_retired_capacity_coeff`) what is retired in it, *pre-built* (`constraint_cumulative_pre_built_capacity_coeff`) the capacity in service at the start of the period and *capacity* (`constraint_capacity_coeff`) the capacity in service in the period — the same capacity that limits the flows. Capacity = pre-built + invested − retired holds with or without a construction lead time; capacity in construction is not in *capacity*.

  *Online* (`constraint_online_coeff`) is the number of units online in each time step (of `virtual_unitsize`; without it, the largest `existing` capacity, else 1000 MW).

  A constraint with only capacity terms is written once per period, otherwise once per time step (rows `process_constraint_{equal,less_than,greater_than}_period` with dimensions (constraint, period) in the `output_horizon` raw dump, instead of (constraint, period, time)). With `constraint_period_method` *period_total*, the constraint is written once per period over the period's total (rows `process_constraint_{equal,less_than,greater_than}_period_total`, dimensions (constraint, period)). For plain sums of capacity with a period trajectory, the group parameters (`invest_max_period`, `retire_max_period`, `cumulative_max_capacity` and their minimum counterparts) are simpler: user constraints are for weighted, cross-class and ratio forms.

## The coefficient parameters

**Map parameters.** Every coefficient is a map: the index is the name of a `constraint`, the value is the coefficient. One entity can take part in several constraints with different coefficients. An entity takes part in a constraint only if it has a coefficient for it.

**The row.** The model multiplies each variable by its coefficient and adds the terms up on the left-hand side. The `sense` compares that sum with the `constant`. The unit of the row is your choice: the coefficients carry it (for example, a coefficient of 1 on a flow keeps the row in MW).

| Parameter | Set on | Term it puts in the row | Unit of the variable |
|---|---|---|---|
| `constraint_flow_coeff` | `unit__inputNode`, `unit__outputNode`, `connection__node` | The flow between the unit or connection and the node, in each time step. | MW |
| `constraint_state_coeff` | `node` (storage) | The storage state of the node, in each time step. | MWh |
| `constraint_online_coeff` | `unit`, `connection` (with online variables) | The number of units online, in each time step. One unit is `virtual_unitsize`; without it, the largest `existing` capacity, else 1000 MW. Whole numbers with *binary* startup, continuous with *linear*. | count of units |
| `constraint_invested_capacity_coeff` | `node`, `unit`, `connection` | The capacity that comes online in the period. Without a construction lead time, this is the capacity invested in that period. | MW (MWh for a storage node) |
| `constraint_in_construction_capacity_coeff` | `node`, `unit`, `connection` | The capacity in construction in the period: decided, but not yet online. It is zero without a construction lead time. | MW (MWh for a storage node) |
| `constraint_retired_capacity_coeff` | `node`, `unit`, `connection` | The capacity retired in the period. | MW (MWh for a storage node) |
| `constraint_cumulative_pre_built_capacity_coeff` | `node`, `unit`, `connection` | The capacity in service at the start of the period: existing capacity plus capacity that came online in earlier periods, minus earlier retirements. | MW (MWh for a storage node) |
| `constraint_capacity_coeff` | `node`, `unit`, `connection` | The capacity in service in the period: existing plus investments still in service minus retirements. Capacity in construction is not included. | MW (MWh for a storage node) |

The full descriptions are with each entity: [nodes](nodes.md#data-for-nodes), [units](units.md#investment-parameters-for-capacity-expansion), [unit flows](units.md#properties-of-unit-inputnode-and-unit-outputnode-entities) and [connections](connections.md#investment-parameters-for-connections).

## Rows per period or per time step

**Capacity terms only.** A constraint that has only capacity terms is the same in every time step of a period. The model writes it once per period, and a period map in `constant` gives each period its own right-hand side.

**Flow, state or online terms.** A constraint with at least one `constraint_flow_coeff`, `constraint_state_coeff` or `constraint_online_coeff` is written once per time step. Any capacity terms in it take the value of their period in each of those time steps.

**Period total.** With `constraint_period_method` *period_total*, the constraint is written once per period, over the sum of all the time steps of the period. Use it for limits on energy over a period, such as "gas covers at most 15 % of the CSP output over the year" or "a unit runs at most 30 % of the year at full capacity". The `max_cumulative_flow` parameters of a group are a simpler choice for a limit on the flow of a group alone; they take the limit as an average MW.

## Period total: units and weighting

**Annual equivalent.** A *period_total* row is per year, like `co2_max_period`. Each time step's terms are multiplied by its duration (hours) and its timestep weight, summed over the period, and divided by the period's share of a year. So a flow term is in MWh per year, and the `constant` is per year too (for example MWh/yr). For a solve that covers a whole year, this is the plain total of the period. For representative periods, the timestep weights scale each modelled step to the time it stands for, so the row counts the whole year. A share constraint (`constant` 0) is the same whatever the scaling.

**Units online.** An online term counts unit-hours per year (units online × hours), for example "at least 4000 online hours per year". With *linear* startup the units online are continuous, so the row counts unit-hours of fractional units.

**Capacity terms.** Every capacity term (invested, in construction, retired, pre-built, capacity), with its existing capacity, is multiplied by the hours of the year that the period's steps stand for (8760 for a whole year; the model computes it from the step durations and weights). Each term is then its per-step quantity summed over the period, so the coefficients keep their per-step meaning: a flow − 0.3 × capacity ≤ 0 row caps the annual capacity factor of the unit at 30 %. A *period_total* constraint with only capacity terms is today's per-period row multiplied by the hours, so its `constant` is in MWh per year.

**State terms are not allowed.** A storage state summed over the time steps of a period is not an energy. A *period_total* constraint with a `constraint_state_coeff` stops the run with an error naming the constraint and the node; use *per_timestep* for constraints on storage states.

**Rolling solves.** A row needs the whole period. In a rolling solve, the row is written in a roll that realizes the whole period (its `rolling_solve_jump` is a multiple of the period length and starts at a period start); a period that a roll holds only in its look-ahead in part gets no row in that roll. When the rolls of a rolling solve do not each realize whole periods, that solve leaves the *period_total* constraints out and the run log warns, naming the constraints, the solve and the period. A typical case is an investment solve on representative periods followed by a rolling dispatch solve with daily rolls: the investment solve enforces the constraint, the dispatch does not. When no solve of the run enforces the constraint, the run stops with an error.

**Chains of solves.** Each solve writes the rows of its own whole periods, so an investment solve and a later dispatch solve both enforce the constraint, each on its own time steps. Because the row is annualised, the two are comparable. A *greater_than* row in a dispatch solve has no slack: if the investment solve built too little for it, the dispatch solve is infeasible (as with a *per_timestep* row).

**Stochastic solves.** Each forecast branch period has its own row: every forecast meets the limit on its own, without a probability weight.

**Dual.** The dual of a *period_total* row (in the `output_horizon` raw dump) is in objective units per unit of the `constant` per year.

## Example: storage size tied to turbine capacity

**Goal.** A pumped-hydro reservoir must hold 6 hours of its turbine capacity: storage capacity (MWh) = 6 × turbine capacity (MW). Both can be invested in.

**Data.**

- `constraint` *reservoir_6h*: `sense` *equal*, `constant` 0
- `node` *reservoir*: `constraint_capacity_coeff` Map (*reservoir_6h* | 1)
- `unit` *turbine*: `constraint_capacity_coeff` Map (*reservoir_6h* | -6)

**The row.** In each period: 1 × reservoir capacity (MWh) − 6 × turbine capacity (MW) = 0. The row has only capacity terms, so it is written once per period. It counts existing capacity, investments and retirements, so it holds for the whole capacity in service, not only for the new build. To tie only the new build, use `constraint_invested_capacity_coeff` instead.

## Example: minimum share of a flow

**Goal.** In every time step, at least 40 % of the electricity flowing into node *elec* comes from unit *wind*. The only other unit that feeds *elec* is *coal_plant*.

**The algebra.** wind ≥ 0.4 × (wind + coal) is the same as 0.6 × wind − 0.4 × coal ≥ 0.

**Data.**

- `constraint` *wind_share*: `sense` *greater_than*, `constant` 0
- `unit__outputNode` (*wind* | *elec*): `constraint_flow_coeff` Map (*wind_share* | 0.6)
- `unit__outputNode` (*coal_plant* | *elec*): `constraint_flow_coeff` Map (*wind_share* | -0.4)

**The row.** The row has flow terms, so it is written once per time step: the share must hold in each time step, not only on average. Every unit that feeds the node has to be in the constraint, or its flow is left out of the total.

## Example: a 15 % share over the year

**Goal.** Over each year, the gas backup *gas_backup* of a CSP plant produces at most 15 % of the CSP output. In a single time step it may produce more, as long as the yearly total stays within the share.

**The algebra.** gas ≤ 0.15 × csp over the year is the same as gas − 0.15 × csp ≤ 0, summed over the time steps.

**Data.**

- `constraint` *gas_share*: `sense` *less_than*, `constant` 0, `constraint_period_method` *period_total*
- `unit__outputNode` (*gas_backup* | *elec*): `constraint_flow_coeff` Map (*gas_share* | 1)
- `unit__outputNode` (*csp* | *elec*): `constraint_flow_coeff` Map (*gas_share* | -0.15)

**The row.** Once per period: Σ_t (gas − 0.15 × csp) × duration × weight / period share ≤ 0. For four 1-hour steps with CSP output 100, 100, 50 and 0 MW (250 MWh), gas may produce 0.15 × 250 = 37.5 MWh in total, in any of the steps. With *per_timestep*, gas could produce only 7.5 MW in the third step (15 % of 50 MW) and nothing when the CSP plant is off.

## Example: two of three units online

**Goal.** In every time step, at least two of the three units *g1*, *g2*, *g3* are online (for example, to keep spinning capacity), whatever the demand.

**Data.**

- `constraint` *two_online*: `sense` *greater_than*, `constant` 2
- `unit` *g1*, *g2*, *g3*: `startup_method` *binary*, `constraint_online_coeff` Map (*two_online* | 1)

**The row.** In each time step: online(g1) + online(g2) + online(g3) ≥ 2. The row counts units, so it is the same for one entity of three units: `existing` 3, `virtual_unitsize` 100 and `constraint_online_coeff` Map (*two_online* | 1). With *linear* startup, the count is continuous: ≥ 2 then means at least two units' worth of online capacity, with the minimum load applying to it, not two whole machines. A coefficient on a unit without online variables (`startup_method` unset and no `minimum_time_method`) stops the run with an error naming the unit and the constraint.
