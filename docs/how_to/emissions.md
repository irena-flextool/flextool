# Emissions and system limits

## How to add CO2 emissions, costs and limits

**(examples.sqlite scenario: coal_co2 )**
***init - west - coal - co2_price - co2_limit***

Carbon dioxide emissions are added to FlexTool by associating relevant `commodities` (e.g. *coal*) with a `co2_content` parameter (CO2 content (tons per MWh) of energy contained in the fuel). 
The other CO2 parameters are handeled through a group of nodes (Here *coal_price* or *coal_limit* `groups`). Therefore one needs to create a group and add all the nodes that supply these commodities (eg. coal_market) to a group with a `group_node` entity. (Here the entity *co2_price--coal_market*)
To set a price one needs to create set the co2_method parameter to *price* (constant) and create the `co2_price` parameter with the desired value. This price is added to the price of the commodity. 

Alternatively one can set a limit on the co2 used by setting the `co2_method` parameter to *period* or *total* and setting the `co2_max_period` (periodic map [tCO2/a], annual emissions in each period) parameter or `co2_max_total` (constant [tCO2], emissions over the whole solve: annual emissions times `years_represented`, summed over the periods) parameter. 

A flow into a node of the same `group` (instead of out of it) is treated as CO2 removal: it earns `co2_content × co2_price` as a credit against the price, and it is subtracted from the group's `co2_max_period` / `co2_max_total` emissions. This is how a direct-air-capture or bio-energy-with-carbon-capture process should be modelled — as a process whose output flow delivers into the group's CO2-content commodity node.

There methods can be combined by setting the `co2_method` to *price_period*, *price_total*, *period_total* or *price_period_total*.


![Add CO2](../img/tutorial/coal_co2.png)

## How to create a non-synchronous limit
**(non_sync_and_curtailment.sqlite)**
***(scenario: non_sync)***

Non-synchronous limit is a property of a `group` of nodes. It states that the non-synchronous flow to the group of nodes cannot exceed a set share of the input flows at any timestep. To demonstrate this, we have set a system with a coal plant, a wind plant and a single demand node. However, it can be done to a group of nodes with unlimited number of plants or connections connected. So, one can limit the non-synchronous share of individual nodes or of the whole system. The flows between the nodes of the group are excluded.

The non-synchronous limit is set to a `group` of nodes with one or multiple members. Note: These are set to the group with `group_node` entity, not with `group_node_unit` entity!

Create a group (here *nodeA_group*) and set a `group_node` entity (nodeA_group |nodeA). Then add parameters:

- `has_non_synchronous`: yes
- `non_synchronous_limit`: 0.5
- `penalty_non_synchronous`: 4000

This forces the non-sync flow to be at max 50% of the incoming flow to the nodeA. 
The penalty should be always set as in some cases there is no other way to keep the constraint feasible (and it will be difficult to find the reason why the model does not solve). The existence of the non-synchronous penalty in your results indicates that this constraint has been violated and you should investigate the reason for this in your system. If the `penalty_non_synchronous` is lower than the `upward_penalty` of the demand `node`, the system will prefer breaking the non-sync constraint instead of the `node` balance equation. In other words, it will not curtail the production of `profile` plants if the demand is not satisfied even though it will break the non-synchronous penalty. If it is higher, the curtailment will take place instead.

When having a connection to a storage from a node with this constraint, the best practice is to add this storage node to the group as well. This excludes the flows to and from the storage in the calculation of the non-synchronous ratio. However, if you want to have these flows in the calculation, you have two options depending on if you want the flow to be synchronous or not. If it is non-synchronous, just set the connection parameter `is_DC`: yes. If it is synchronous (e.g., some flywheels), an issue arises with the *reqular* connection as it now can reduce the ratio with equal simultaneous flow in both directions. To prevent the simultaneous flow to both directions you need to set the `transfer_method`: *exact*. This will increase the solving time significantly as it will use integer variables (so, better to add the storage node to the node group with a non-synchronous limit).

Then set which plants and connections are considered non-synchronous by adding parameters:

`unit_outputNode`:

- `is_non_synchronous`: yes 

`Connection_node_node`:

- `is_DC`: yes 

Here the (wind_plant|nodeA) relation has the `is_non_synchronous` parameter and battery connection `is_DC` parameter. Because the battery node is in *nodeA_group*, the battery connection runs inside the group and its `is_DC` has no effect; it would count if the battery node were left out of the group.

A connection between a node of the group and a node outside it counts in both directions: its flow into the group is infeed (non-synchronous if the connection is `is_DC`), its flow out of the group is supply. A *regular* connection (at the cost of its losses) and a *no_losses_no_variable_cost* connection (at no cost) can flow both ways at the same time, which raises the supply side and so loosens the limit. Use `transfer_method`: *exact* for a synchronous connection that crosses the boundary of a group with a non-synchronous limit. FlexTool warns about a *no_losses_no_variable_cost* connection across the boundary (not when it is `is_DC`: then flowing both ways only tightens the limit).

With several groups that have a non-synchronous limit, each group counts its own boundary: wind exported from one group over an AC connection arrives in the neighbouring group as synchronous infeed (it is not counted against that group's limit), while over an `is_DC` connection it counts as non-synchronous there too.

If you want summary metrics per flow in the results, create a separate `flowGroup` for each flow (or collection of flows) and add `flowGroup__unit__node` entities to it. The flowGroup needs the parameter:

- `flow_aggregator`: standalone_aggregator_only

This emits the flow-group's own outputs (per-period `cumulative_flow` MWh and `average_flow` MW, plus a signed net-flow timewise series) for the members. It is summary output — for per-timestep decomposition of a node's flows, use `print_dispatch` on a `group__node` group instead (see "How to create aggregate outputs").

Here we have *coal_flow* `flowGroup` with `flowGroup__unit__node` relation coal_flow|coal_plant|nodeA
and *wind_flow* `flowGroup` with `flowGroup__unit__node` relation wind_flow|wind_plant|nodeA.

![non-sync](../img/concept/non_sync.png)
