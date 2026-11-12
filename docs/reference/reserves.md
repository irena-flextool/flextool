# Reserves

The user defines reserve categories through `reserve` entity. Reserves are reservations of capacity (either upward or downward) and that capacity will not therefore be available for other use (flowing energy or commodities). There are three different ways how a reserve requirement can be calculated: timeseries, large_failure and dynamic. 

- Timeseries requires that the user provides a pre-defined time series for the amount of reserve to be procured in each time step. 
- Large_failure requires that the user defines the energy flows that could be the largest failure in the system. The highest possible failure (flow multiplied by `large_failure_ratio`) in each timestep will then set the reserve requirement for that timestep. Each failing unit or connection is checked separately, and the reserve it holds itself does not count towards covering its own failure. For upward reserve the failure is the loss of the flow into the group's nodes; for downward reserve it is the loss of a large consumption (flow out of the group's nodes). Every unit or connection given a `large_failure_ratio` adds one small reserve constraint per reserve group and time step (its failure against the group's total reserve minus its own reserve), so the model grows with the number of such units plus the number of reserve providers. Still give the ratio only to the units and connections whose failure can be the largest, since each one adds constraints. 
- Dynamic means that there is a requirement based on user chosen energy flows - each participating flow is multipled by `increase_reserve_ratio` and then summed to form the reserve requirement, for upward and downward reserve alike (the ratio stands for the variability of the flow, which can go either way). This can be useful for covering variability within timesteps. Also demand variability can be included through `increase_reserve_ratio` parameter in `reserve__upDown__group` entity: the requirement then also grows with the demand of the group's nodes (see below).

When the same reserve category (e.g. primary upward) has more than one of these (timeseries, large_failure and dynamic) in use, then the largest requirement will apply for that timestep. If they are separated into different reserves, then they need to be fulfilled separately.

Reserve requirement is defined for groups of nodes. This means that multiple nodes can have a common reserve requirement (but it is also possible to make a group with only one node). One node can be in multiple groups and therefore subject to multiple overlapping reserve requirements. Only units can generate reserve, but connections can move reserve from one node to another (therefore, there is no impact if the nodes are in the same reserve group, but it can be useful to import reserve from outside the group).

## Reserve groups

For `reserve__upDown__group` entities:

- `reserve_method` - Choice of reserve method (`no_reserve`, `timeseries_only`, `dynamic_only`, `large_failure_only` or their combinations). A dynamic or large_failure requirement needs no `reservation`: it applies once a unit or connection at a node of the group has an `increase_reserve_ratio` (dynamic; or the group has one) or a `large_failure_ratio` (large_failure) for this reserve.
- `reservation` - [MW] Amount of reserve required. Constant, period, time, period+time or stochastic time.
- `penalty_reserve` - [CUR/MW per hour] Penalty cost for not fulfilling the reserve requirement, per MW of shortfall and hour. Constant. Default 5000 (applied to every reserve group without a value of its own, so a reserve requirement is never free to leave unmet). The unmet part (reserve shortfall) can cover the whole requirement of every method and is always priced with this penalty: for a timeseries requirement the shortfall is at most `reservation`; for a dynamic requirement at most the sum of the participating flows' capacities × `increase_reserve_ratio` (at a node a unit consumes from, its full-load input), plus the group's demand term; for a large_failure requirement at most the largest failing unit's or connection's capacity × `large_failure_ratio` (summed over its flows into the group, or the full-load input it draws from the group for downward reserve) (capacity = existing plus the most that can be invested by that period; with unlimited investment `max_flow_for_unconstrained_variables` is used). A group combining methods adds these up.
- `increase_reserve_ratio` - [factor] For a group with a dynamic `reserve_method`: in each time step the requirement increases by this ratio times the demand of the group's nodes in that step (the negative part of each node's inflow, in MW: the step's inflow divided by the step duration; a node with positive inflow does not count and does not offset another node's demand). The extra demand of an `energy_margin` does not count: the requirement follows the real demand. Ignored (with a warning) for other methods. Constant.
- `reserve_duration` - [hours] Time span over which committed reserve power must be sustainable from storage. Constant. Absent or 0 means the reserve is instantaneous-power-only (legacy behaviour); no storage-energy coupling is added.

### Storage-backed reserve energy adequacy (`reserve_duration`)

By default a reserve reservation only withholds *power* (MW of headroom) from
the provider. `reserve_duration` additionally requires the provider to hold
enough *energy* to actually sustain that committed power for the given number
of hours, when the provider draws on (upward reserve) or charges into
(downward reserve) a storage node. This is a single-scalar simplification of
Backbone's duty-cycle formulation (self-discharge over the window is ignored
in this first version).

For each committed reserve MW the required energy is `reservation_MW ×
reserve_duration`, corrected by the provider's efficiency:

- **Upward / discharge** — the storage node is the provider's *source*, so the
  energy drawn is divided by the efficiency (`÷ η`): a lossy discharge must
  draw more stored energy to deliver the same MW. The stored energy floor is
  `v_state · unitsize ≥ Σ reservation_MW · reserve_duration / η`.
- **Downward / charge** — the storage node is the provider's *sink*, so the
  energy stored is multiplied by the efficiency (`× η`): a lossy charge stores
  less per MW of charging. The free-headroom bound is
  `v_state · unitsize + Σ reservation_MW · reserve_duration · η ≤ capacity`
  (with the same invest/divest capacity replication as the storage state
  bound).

The coupling is storage-only: reserve provided by non-storage units (e.g. a
gas turbine drawing on a fuel commodity) is unaffected. The obligation uses the
provider's full committed MW — the trust derating (`reliability`) is
deliberately excluded, since the physical energy required is independent of how
much of the reservation is counted towards the requirement.

Limitation under representative periods: the constraints reference the
intra-representative-period storage state (`v_state`) only. Under the
representative-period *blended-weights* storage modes the seasonal inter-period
state is bounded separately and is not aligned to the reserve's time index, so
`reserve_duration` couples reserve to intra-period state only — matching
FlexTool's own intra-period capacity approximation.

## Reserve provision by units

For `reserve__upDown__unit__node` entities:

- `max_share` - [factor] Maximum share of the unit's capacity at this node that can be held as this reserve: `max_share` × built capacity (existing + invested − retired), also for units with online variables (their online capacity limits the reserve separately, see below). At a node the unit consumes from, the capacity is the unit's full-load input there (for a unit with one input and one output: capacity / efficiency; for a unit with several inputs: `input_share_max` × full-load input). Without a value the unit offers no reserve. Constant.
- `reliability` - [factor] The share of the reservation that is counted to reserves (sometimes reserve sources are not fully trusted). Constant.
- `increase_reserve_ratio` - [factor] The reserve requirement is increased by the flow between the unit and the node multiplied by this ratio, for upward and downward reserve alike. At a node the unit consumes from, the flow is what the unit draws there (output / efficiency, as in the node balance). Constant.
- `large_failure_ratio` - [factor] Each unit using the N-1 failure method will have a separate constraint to require sufficient reserve to cover a failure of the unit generation (multiplied by this ratio); for downward reserve, the loss of the consumption the unit draws from the group's nodes (output / efficiency). The failing unit's own reserve does not count towards covering its own failure. With `virtual_unitsize`, the failure is the whole unit (all its units of `virtual_unitsize`), not one of them. Constant.

A unit can only offer reserve that reduces its flow as far as its flow allows: downward reserve at a node the unit feeds, and upward reserve at a node the unit consumes from (a consumption cut), cannot exceed the flow, and for an online unit cannot take the sum of its outputs below `min_load` × online capacity, nor an output with `capacity_min_coeff` below its own floor. The effective limit is the smaller of this and `max_share` × capacity. An upward reserve at the input node of a unit with one input and one output is converted to output flow by the unit's efficiency. A unit with several outputs or inputs that offers upward reserve at an input node is only kept from cutting that input below zero; together with downward reserve on its outputs it can then end up below `min_load`.

Reserve that raises a unit's flow needs headroom: upward reserve at a node the unit feeds and downward reserve at a node it consumes from (consuming more) cannot exceed the free capacity of that flow (built capacity; online capacity for a unit with online variables), so a unit at full consumption offers no downward reserve at its input, and an offline unit none. A downward reserve at the input node of a unit with one input and one output is converted to output flow by the unit's efficiency; such a unit with several outputs needs the headroom on each output. A unit with several inputs is only kept within that input's capacity, and its upward reserve at an output and downward reserve at an input (both "burn more") are each checked against the unit's capacity separately, not together. The effective limit is the smaller of these and `max_share` × capacity. A connection's reserve at a node is limited by its net position there: upward reserve by its free capacity towards the node plus the flow it sends away from the node (which it can stop), downward reserve by the flow it brings to the node plus its free capacity away from it (power at the node, after losses). A one-way connection can therefore offer upward reserve at its sending node only up to what it draws there. For a connection with online variables the online capacity counts; for an *exact* connection, the capacity of the direction it is set to.

A unit with online variables (`startup_method` `linear` or `binary`, or a `minimum_time_method` other than *none*) holds upward reserve only from its online capacity: at a node the unit feeds, its output plus the upward reserve it holds there cannot exceed online capacity × availability (× `capacity_max_coeff`), and for a unit with several outputs the sum of its outputs plus the upward reserve held at them cannot exceed online capacity × availability. An offline unit therefore offers no upward reserve, and an upward reserve requirement can make it worth committing a unit. This is a simplification: offline units are assumed unable to start within the reserve's activation time (reserve activation time versus unit start-up time is not modelled yet). Units without online variables are limited by their built capacity.

A unit with ramp limits (`ramp_method` `ramp_limit` or `both`) can hold upward reserve only within its upward ramp room of the time step after the change of its flow from the previous step, and downward reserve within its downward ramp room: the flow plus the upward reserve may rise from the previous step's flow by at most `ramp_speed_up × 60 × (average duration of the two steps) × capacity`, and the flow minus the downward reserve may fall by at most the same with `ramp_speed_down` (so a unit ramping down in a step can hold more upward reserve in it). "Consume more" downward reserve at a node the unit consumes from uses the ramp room of that input (an upward input ramp), converted by the efficiency on a unit with one input and one output; upward reserve there (consume less) uses the downward room. In the first step of a solve, and after a gap in the timeline, the reserve alone is limited to one step's room, using the online capacity for a unit with online variables. For units with online variables the reserve uses the ramp room of the unit's online capacity: a unit starting or stopping in a step is limited by its startup / shutdown output there instead (see the how-to on start-ups), not by any extra room from declaring the start or stop itself. The ramp room is per model step; the reserve's activation time is not considered.

## Reserve transfer by connections

For `reserve__upDown__connection__node` entities:

- `max_share` - [factor] Maximum share of the connection's built capacity (existing + invested − retired) that can be held as this reserve at this node. Constant.
- `reliability` - [factor] The share of the reservation that is counted to reserves (sometimes reserve sources are not fully trusted). Constant.
- `increase_reserve_ratio` - [factor] The reserve requirement is increased by the flow between the connection and the node, in either direction, multiplied by this ratio, for upward and downward reserve alike (at the node the connection takes from: the flow it draws there, including its losses; a *no_losses_no_variable_cost* or DC power-flow connection's reverse flow counts too). Constant.
- `large_failure_ratio` - [factor] Each connection using the N-1 failure method will have a separate constraint to require sufficient reserve to cover a failure of the connection (multiplied by this ratio): for upward reserve the loss of its flow into the group's nodes, for downward reserve the loss of the flow it draws from them, in either direction (a *no_losses_no_variable_cost* or DC power-flow connection's reverse flow counts too). The failing connection's own reserve does not count towards covering its own failure. With `virtual_unitsize`, the failure is the whole connection (all its units of `virtual_unitsize`), not one of them. Constant.
