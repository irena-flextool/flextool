[Install](https://github.com/irena-flextool/flextool/tree/master#irena-flextool) | [Tutorial](https://irena-flextool.github.io/flextool) | [Results](https://irena-flextool.github.io/flextool/results) | [Reference](https://irena-flextool.github.io/flextool/reference) | [Data structure](https://irena-flextool.github.io/flextool/spine_database) | [Spine Toolbox interface](https://irena-flextool.github.io/flextool/spine_toolbox) | [Browser-interface](https://irena-flextool.github.io/flextool/browser_interface)

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

- `model` object `cost` parameter - includes annualized total cost as well as annualized costs divided into 
  - *unit investment/retirement* - cost of investing in unit capacity or benefits from salvaging unit capacity
  - *connection investment/retirement* - cost of investing in connection capacity or benefits from salvaging connection capacity
  - *storage investment/retirement* - cost of investing in storage capacity or benefits from salvaging storage capacity
  - *commodity* - cost of unit using commodity inputs or benefit of selling commodities (negative value)
  - *CO2* - cost of CO2 emissions caused by unit using commodities with CO2 content
  - *variable cost* - other variable operation and maintenance costs
  - *starts* - start up costs
  - *upward penalty* - cost of involuntary demand reduction
  - *downward penalty* - cost of involuntary demand increase
  - *inertia penalty* - cost of not meeting the inertia constraint
  - *non-synchronous penalty* - cost of not meeting the non-synchronous constraint
  - *capacity margin penalty* - cost of not meeting the capacity margin constraint
  - *upward reserve penalty* - cost of not meeting the upward reserve constraint
  - *downward reserve penalty* - cost of not meeting the downward reserve constraint
- `model` object `cost_t` parameter - similar as above but costs given for each timestep (no investment/retirement costs)

## Prices

- `node` object `price_t` parameter - each node that maintains an energy balance provides a price time series based on the marginal value of the balance constraint

## Energy flows

- `unit__node` relationship `flow` parameter - cumulative flow from the node (if node is input) or to the node (if node is output)
- `unit__node` relationship `flow_t` parameter - flow from the node (if node is input) or to the node (if node is output)
- `connection__node__node` relationship `flow` parameter - cumulative flow through the connection (left to right is positive)
- `connection__node__node` relationship `flow_t` parameter - flow through the connection (left to right is positive)

## Energy balance in nodes

- `node` object `balance` parameter - cumulative inputs (positive) and (outputs) to the node from all the possible sources (*from_units*, *from_connection*, *to_units*, *to_connections*, *state change* over the period, *self discharge* during the period, *upward slack* for involuntary demand reduction and *downward slack* for involuntary demand increase)
- `node` object `balance_t` parameter - same as above, but for each timestep
- `node` object `state_t` parameter - storage state of the node in each timestep (typically MWh).

## Unit online and startup

- `unit` object `online_average` parameter - average online status of the unit (average number of units online during the period)
- `unit` object `online_t` parameter - online status of the unit (number of units online in each timestep)
- `unit` object `startup_cumulative` parameter - cumulative number of unit startups during the period

## Group results

- `group` object `indicator` parameter - gives a set of results for all `node` members of the `group`
  - *sum of annualized inflows* - sum of `inflow` to the node which has been annualized (scaled to correspond to a year of timesteps)
  - *VRE share* - how much the flows from VRE sources (inputs using  'upper limit' profile) are of the inflow
  - *curtailed VRE share* - how much the unused flows from VRE sources would have been of the inflow
  - *upward slack share* - upward slack in relation to the inflow
  - *downward slack share* - downward slack in relation to the inflow

## Capacity and investment results

- `unit`, `connection` and `node` objects `capacity` parameter - include the following parameters
  - *existing* - capacity that was assumed to exist in the beginning of the solve
  - *invested* - capacity the model decided to invest for the given period
  - *retired* - capacity the model decided to retire in the beginning of the given period
  - *total* - sum of *existing*, *invested* and *retired* capacities
- `unit`, `connection` and `node` objects `invest_marginal` parameter - marginal cost to invest in one more MW or MWh of capacity (zero value means that the model has invested in optimal amount; negative value means that if the model would be able to invest more, it could reduce total cost by the stated amount per MW or MWh; positive value means the cost is higher than the benefit by the stated amount per MW or MWh)
- `group` parameter `slack_capacity_margin` - use of slack variable and the associated penalty cost to meet the capacity margin requirement in the period

## CO2 emissions

- `unit` object `co2` parameter - how many tons of CO2 the unit has generated (by using commodity with CO2 content) or removed

## Reserves

- `unit__reserve__upDown__node` relationship `reservation_t` parameter - how much upward or downward reserve particular unit was providing to a particular node in given timestep
- `unit__reserve__upDown__node` relationship `reservation_average` parameter - how much upward or downward reserve particular unit was providing to a particular node in average during the period
- `group__reserve__upDown` relationship `slack_reserve_t` parameter - use of slack variable and the associated penalty cost to fulfill the upward or downward reserve requirement in each timestep
- `group__reserve__upDown` relationship `slack_reserve` parameter - cumulative use of slack variable and the associated penalty cost to fulfill the upward or downward reserve requirement during the period

## Inertia and non-synchronous generation

- `group` object `inertia_t` parameter - the amount of inertia (MWs) in the group of nodes in each timestep
- `group` object `slack_inertia_t` parameter - use of slack variable and the associated penalty cost to fulfill the inertia requirement in each timestep
- `group` object `slack_nonsync_t` parameter - use of slack variable and the associated penalty cost to fulfill the non-synchronous share maximum share constraint in each timestep 

## Ramps

- `node` object `ramp_t` parameter - includes seven parameters that form the ramp room envelope (how much there is additional room to ramp in a give node)
  - *ramp* - the actual ramp in the node from previous timestep to this timestep
  - *units_up* - additional room for upward ramps from non-VRE units connected to the node
  - *VRE_up* - adds upward ramp room from VRE units on top of the ramp room from non-VRE units
  - *connections_up* - adds upward ramp room from connections on top of the previous ramp rooms (does not consider whether the connected node has ramp room, but is simply the available capacity in the connection)
  - *unis_down* - additional room for downward ramps from non-VRE units connected to the node 
  - *VRE_down* - adds downward ramp room from VRE units on top of the ramp room from non-VRE units
  - *connections_down* - adds downward ramp room from connections on top of the previous ramp rooms (does not consider whether the connected node has ramp room, but is simply the available capacity in the connection)
- `unit__node` relationship `ramp_t` parameter - shows ramping of particular input or output flow between a unit and a node for each time step

## Slack and penalty values

Slack and penalty values are listed in various places above (costs, energy balance, reserves, inertia and non-sychronous generation).
