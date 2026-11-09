# User constraints

## Additional entities for further functionality

- `constraint`: a user-defined linear constraint between flow, state, and capacity variables on nodes, units and connections. The left-hand side is assembled by tagging the relevant entities with the matching coefficient parameter (`constraint_flow_coeff` on `unit__inputNode` / `unit__outputNode` / `connection__node`, `constraint_state_coeff` on `node`, `constraint_invested_capacity_coeff` / `constraint_in_construction_capacity_coeff` / `constraint_cumulative_pre_built_capacity_coeff` / `constraint_capacity_coeff` / `constraint_retired_capacity_coeff` on `node` / `unit` / `connection`); the constraint entity itself carries:

    - `sense` - The sense of the constraint: `greater_than`, `less_than`, or `equal`.
    - `constant` - Right-hand side of the user constraint (default 0). A period map gives each period its own value; it must cover every period of the solve (a missing period is an error). Constant or period.

  Capacity terms, all in MW (MWh for storage nodes) and multiplied by the coefficient: *invested* (`constraint_invested_capacity_coeff`) is the capacity that comes online in the period, *in construction* (`constraint_in_construction_capacity_coeff`) the capacity decided but not yet operating, *retired* (`constraint_retired_capacity_coeff`) what is retired in it, *pre-built* (`constraint_cumulative_pre_built_capacity_coeff`) the capacity in service at the start of the period and *capacity* (`constraint_capacity_coeff`) the capacity in service in the period — the same capacity that limits the flows. Capacity = pre-built + invested − retired holds with or without a construction lead time; capacity in construction is not in *capacity*.

  A constraint with only capacity terms is written once per period, otherwise once per time step (rows `process_constraint_{equal,less_than,greater_than}_period` with dimensions (constraint, period) in the `output_horizon` raw dump, instead of (constraint, period, time)). For plain sums of capacity with a period trajectory, the group parameters (`invest_max_period`, `retire_max_period`, `cumulative_max_capacity` and their minimum counterparts) are simpler: user constraints are for weighted, cross-class and ratio forms.

