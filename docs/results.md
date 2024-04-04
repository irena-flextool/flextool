# Results

FlexTool outputs results typical to a planning model or a scheduling model, but it also tries to highlight potential flexibility issues in the system. The outputs from the latest run are initially CSV files and can befound in the folder 'output'. File 'summary_solve.csv' can give a quick overview of potential issues in the solve - it is a diagnostic file. The other files are all numerical results and will be imported to a Spine database by the FlexTool workflow.

- [Costs](#costs)
- [Prices](#prices)
- [Energy flows](#energy-flows)
- [Energy balance in nodes](#energy-balance-in-nodes)
- [Group results](#group-results)
- [Capacity and investment results](#capacity-and-investment-results)
- [CO2 emissions](#co2-emissions)
- [Reserves](#reserves)
- [Inertia and non-synchronous generation](#inertia-and-non-synchronous-generation)
- [Ramps](#ramps)
- [Slack and penalty values](#slack-and-penalty-values)

## Costs

- `model` object `cost_annualized` parameter - M[CUR] (millions of user chosen currency) includes annualized total cost as well as annualized costs divided into 
  - *unit investment/retirement* - M[CUR] cost of investing in unit capacity or benefits from salvaging unit capacity
  - *connection investment/retirement* - M[CUR] cost of investing in connection capacity or benefits from salvaging connection capacity
  - *storage investment/retirement* - M[CUR] cost of investing in storage capacity or benefits from salvaging storage capacity
  - *commodity* - M[CUR] cost of unit using commodity inputs or benefit of selling commodities (negative value)
  - *CO2* - M[CUR] cost of CO2 emissions caused by unit using commodities with CO2 content
  - *variable cost* - M[CUR] other variable operation and maintenance costs
  - *starts* - M[CUR] start up costs
  - *upward penalty* - M[CUR] cost of involuntary demand reduction
  - *downward penalty* - M[CUR] cost of involuntary demand increase
  - *inertia penalty* - M[CUR] cost of not meeting the inertia constraint
  - *non-synchronous penalty* - M[CUR] cost of not meeting the non-synchronous constraint
  - *capacity margin penalty* - M[CUR] cost of not meeting the capacity margin constraint
  - *upward reserve penalty* - M[CUR] cost of not meeting the upward reserve constraint
  - *downward reserve penalty* - M[CUR] cost of not meeting the downward reserve constraint
- `model` object `cost_t` parameter - M[CUR] similar as above but costs given for each timestep (no investment/retirement costs)
- `model` object `cost_discounted_solve` paramater - M[CUR] Costs for the solve considering discounting and years presented (scaled to all years presented). Divided like in `cost_annualized`
- `model` object `cost_discounted_total` parameter - M[CUR] Total costs for all the solves considering discounting and years presented (scaled to all years presented). Divided like in `cost_annualized`


## Prices

- `node` object `price_t` parameter - [CUR/MWh] each node that maintains an energy balance provides a price time series based on the marginal value of the balance constraint

## Energy flows

- `unit__node` relationship `flow_annualized` parameter - [MWh] cumulative flow from the node (if node is input) or to the node (if node is output) annualized.
Annualization scales the sum to correspond with full year time series.
- `unit__node` relationship `flow_t` parameter - [MW] flow from the node (if node is input) or to the node (if node is output)
- `connection__node__node` relationship `flow_annualized` parameter - [MWh] cumulative flow through the connection (left to right is positive) annualized
- `connection__node__node` relationship `flow_t` parameter - [MW] flow through the connection (left to right is positive)

### Optional output: output_connetion_flows_separate
- `connection__node__node` relationship `flow_to_first_node_annualized` parameter - [MWh] annualized cumulative flow through the connection only to the left (first) node.
- `connection__node__node` relationship `flow_to_second_node_annualized` parameter - [MWh] annualized cumulative flow through the connection only to the right (second) node.
- `connection__node__node` relationship `flow_to_first_node_t` parameter - [MW] flow through the connection to the left (first) node.
- `connection__node__node` relationship `flow_to_second_node_t` parameter - [MW] flow through the connection to the right (second) node.

## Capacity factors

- `unit__node` relationship `cf` parameter - [per unit] average capacity factor of the flow, i.e. the utilization rate of the flow from/to the unit. Average of flow [MWh/h] divided by capacity [MW] of the input or output to the unit.
- `connection` relationship `cf` parameter - [per unit] average capacity factor of the flow, i.e. the utilization rate of the connection where flows in both directions are considered as utilization. Average of the absolute flow [MWh/h] divided by the capacity of the connection.

## Energy balance in nodes

- `node` object `balance` parameter - [MWh] cumulative inputs (positive) and outputs (negative) to the node from all the possible sources (*from_units*, *from_connection*, *to_units*, *to_connections*, *state change* over the period, *self discharge* during the period, *upward slack* for involuntary demand reduction and *downward slack* for involuntary demand increase)
- `node` object `balance_t` parameter - [MW] same as above, but for each timestep
- `node` object `state_t` parameter - [MWh] storage state of the node in each timestep

## Unit online and startup

- `unit` object `online_average` parameter - [count] average online status of the unit (average number of units online during the period)
- `unit` object `online_t` parameter - [count] online status of the unit (number of units online in each timestep)
- `unit` object `startup_cumulative` parameter - [count] cumulative number of unit startups during the period

# Unit curtailment

- `unit` object `curtailment_share` parameter - [0-1] Share of curtailed production from potential production for periods
- `unit` object `curtailment_t` parameter - [MW] curtailed flow to the node

## Group results

- `group` object `indicator` parameter - gives a set of results for all `node` members of the `group`
  - *sum of annualized inflows* - [MWh] sum of `inflow` to the node which has been annualized (scaled to correspond to a year of timesteps)
  - *VRE share* - [0-1] how much the flows from VRE sources (inputs using  'upper limit' profile) are of the inflow
  - *curtailed VRE share* - [0-1] how much the unused flows from VRE sources would have been of the inflow
  - *upward slack share* - [0-1] upward slack in relation to the inflow
  - *downward slack share* - [0-1] downward slack in relation to the inflow
- `group` object `flow_annualized` parameter - [MWh] produces grouped and annualized flow results of the `node` members of the `group`
- `group` object `flow_t` parameter - [MW] produces grouped flow results to the `node` members of the `group`
- `group` object `sum_flow_annualized` parameter [MWh] Annualized sum of flows in the group (members from group__connection__node and group__unit__node). Annualization scales the sum to correspond with full year time series.
- `group` object `sum_flow_t` parameter [MW] Sum of flows in the group (members from group__connection__node and group__unit__node).
- `group` object `VRE_share_t` parameter - [0-1]  how much the flows from VRE sources (inputs using  'upper limit' profile) are of the inflow for each timestep

## Capacity and investment results

- `unit`, `connection` and `node` objects `capacity` parameter - [MW or MWh] include the following parameters
  - *existing* - [MW or MWh] capacity that was assumed to exist in the beginning of the solve
  - *invested* - [MW or MWh] capacity the model decided to invest for the given period
  - *retired* - [MW or MWh] capacity the model decided to retire in the beginning of the given period
  - *total* - [MW or MWh] sum of *existing*, *invested* and *retired* capacities
- `unit`, `connection` and `node` objects `invest_marginal` parameter - [CUR/MW or MWh] marginal cost to invest in one more MW or MWh of capacity (zero value means that the model has invested in optimal amount; negative value means that if the model would be able to invest more, it could reduce total cost by the stated amount per MW or MWh; positive value means the cost is higher than the benefit by the stated amount per MW or MWh)
- `group` parameter `slack_capacity_margin` - [MW or MWh] use of slack variable and the associated penalty cost to meet the capacity margin requirement in the period
- `group` parameter `slack_capacity_margin` - use of slack variable and the associated penalty cost to meet the capacity margin requirement in the period

## CO2 emissions

- `group` object `CO2_annualized` parameter - [Mt] how many million tons of CO2 annualized the units and connections in this group have generated (by using commodity with CO2 content) or removed.
- `unit` object `CO2_annualized` parameter - [Mt] how many million tons of CO2 annualized the unit has generated (by using commodity with CO2 content) or removed

## Reserves

- `unit__reserve__upDown__node` relationship `reservation_t` parameter - [MW] how much upward or downward reserve particular unit was providing to a particular node in given timestep
- `unit__reserve__upDown__node` relationship `reservation_average` parameter - [MW] how much upward or downward reserve particular unit was providing to a particular node in average during the period
- `group__reserve__upDown` relationship `slack_reserve_t` parameter - [MW] use of slack variable and the associated penalty cost to fulfill the upward or downward reserve requirement in each timestep
- `group__reserve__upDown` relationship `slack_reserve` parameter - [MW] cumulative use of slack variable and the associated penalty cost to fulfill the upward or downward reserve requirement during the period

## Inertia and non-synchronous generation

- `group` object `inertia_t` parameter - [MWs] the amount of inertia (MWs) in the group of nodes in each timestep
- `group` object `inertia_largest_flow_t` parameter - [MW] The largest individual flow coming into the group of nodes that *has_inertia*
- `group` object `inertia_unit_node_t` parameter - [MW] the amount of inertia between units and the nodes of the group
- `group` object `slack_inertia_t` parameter - [MWs] use of slack variable and the associated penalty cost to fulfill the inertia requirement in each timestep
- `group` object `slack_nonsync_t` parameter - [MWh] use of slack variable and the associated penalty cost to fulfill the non-synchronous share maximum share constraint in each timestep 

## Ramps

- `node` object `ramp_t` parameter - includes seven parameters that form the ramp room envelope (how much there is additional room to ramp in a give node)
  - *ramp* - [MW] the actual ramp in the node from previous timestep to this timestep
  - *units_up* - [MW] additional room for upward ramps from non-VRE units connected to the node
  - *VRE_up* - [MW] adds upward ramp room from VRE units on top of the ramp room from non-VRE units
  - *connections_up* - [MW] adds upward ramp room from connections on top of the previous ramp rooms (does not consider whether the connected node has ramp room, but is simply the available capacity in the connection)
  - *units_down* - [MW] additional room for downward ramps from non-VRE units connected to the node 
  - *VRE_down* - [MW] adds downward ramp room from VRE units on top of the ramp room from non-VRE units
  - *connections_down* - [MW] adds downward ramp room from connections on top of the previous ramp rooms (does not consider whether the connected node has ramp room, but is simply the available capacity in the connection)
- `unit__node` relationship `ramp_t` parameter - [MW] shows ramping of particular input or output flow between a unit and a node for each time step

## Slack and penalty values

Slack and penalty values are listed in various places above (costs, energy balance, reserves, inertia and non-sychronous generation).
