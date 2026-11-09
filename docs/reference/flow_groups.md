# Flow groups (`flowGroup`)

A `flowGroup` aggregates a set of *flows* — where each flow is a `(process, node)` arc. Members are declared through the 3-dim membership classes `flowGroup__unit__node` and `flowGroup__connection__node`. A `flowGroup` carries the flow-limit parameters and the `flow_aggregator` method; it does not carry any of the node-group / system-constraint / investment parameters (those stay on [`group`](groups.md)).

## Cumulative and instant flow limits

- `max_cumulative_flow` - [MW] Maximum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multiplied by the model duration to get the cumulative limit (e.g. by 8760 if a single year is modelled). Applied for each solve. Constant or period.
- `min_cumulative_flow` - [MW] Minimum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multiplied by the model duration to get the cumulative limit (e.g. by 8760 if a single year is modelled). Applied for each solve. Constant or period.
- `max_instant_flow` - [MW] Maximum instantenous flow for the aggregated flow of all group members. Constant, period, time or period+time.
- `min_instant_flow` - [MW] Minimum instantenous flow for the aggregated flow of all group members. Constant, period, time or period+time.

When authoring `min_instant_flow` / `max_instant_flow` as a Map, set the Map's `index_name` to `period` or `time` so the period-vs-time axis is unambiguous — see [the note on naming the Map index](entities_and_time.md#timesteps-and-periods) above.

## Flow aggregation method

- `flow_aggregator` — choice of flow-aggregation method for this flowGroup (default `none`). Values:
    - `none` (default) — no aggregated output; the flowGroup exists only for flow limits / investment.
    - `dispatch_plots_only` — member flows are shown as aggregated bands inside a node group's dispatch table (the table produced by `print_dispatch` on a `group`). Produces no output file of its own.
    - `standalone_aggregator_only` — emits the flowGroup's own outputs: the per-period `cumulative_flow` and `average_flow` totals, plus a signed net-flow timewise series (see [Results](../results.md)). Does not feed any node-group dispatch table.
    - `both` — both of the above.

**Double-counting guard.** If a flow arc belongs to two flowGroups that are *both* dispatch-bound (`dispatch_plots_only` or `both`), the arc is double-counted in the node-group dispatch balance; the model emits a loud validation warning in this case. For overlapping thematic aggregates, use `standalone_aggregator_only` — overlap is harmless there.

Some of the outputs are optional. They can be removed to speed up the post-processing of results. The user can enable/disable them by changing parameters of the the `model` entity:

- `output_node_balance_t`: Default: yes. Produces detailed inflows and outflows for all the nodes for all timesteps. Mainly useful to diagnose what is wrong with the model. 
- `output_connection__node__node_flow_t`: Default: yes. The flows between the nodes for each timestep.
- `output_unit__node_flow_t`: Default, yes. The flows from units to the nodes for each timestep.
- `output_ramp_envelope`: Default, no. Includes seven parameters that form the ramp room envelope. How much there is additional ramping capability in a given node. (Parameter node_ramp_t)
- `output_connection_flow_separate`: Default, no. Produces the connection flows separately for both directions.
- `output_unit__node_ramp_t`: Default, no. Produces the ramps of individual units for all timesteps.

Additionally a model level option to exclude all node, connection and unit level outputs and only leave group and model level results. This will significantly increase the speed of importing data to the result database.

- model: `exclude_entity_outputs`: Default, no. Excludes results on node, unit and connection level, but preserves group level results

A further debug option writes everything the LP holds over the model horizon to a separate raw dump: the look-ahead of each roll after `rolling_solve_jump` and the forecast branches of stochastic solves. The normal outputs keep only the realized results, with or without it.

- model: `output_horizon`: Default, no. Writes one raw parquet per LP variable and constraint, with every row of every solve and a `realized` column, to `output_parquet/<scenario>/horizon/` (see [Results](../results.md)). The normal outputs are unchanged.
