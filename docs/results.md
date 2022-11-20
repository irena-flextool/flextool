FlexTool outputs results typical to a planning model or a scheduling model, but it also tries to highlight potential flexibility issues in the system. 
The outputs from the latest run are initially CSV files and can befound in the folder 'output'. File 'summary_solve.csv' can give a quick overview
of potential issues in the solve - it is a diagnostic file. The other files are all numerical results and will be imported to a Spine database by the FlexTool
workflow.

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

## Group results

- `group` object `indicator` parameter - gives a set of results for all `node` members of the `group`
  - *sum of annualized inflows* - sum of `inflow` to the node which has been annualized (scaled to correspond to a year of timesteps)
  - *VRE share* - how much the flows from VRE sources (inputs using  'upper limit' profile) are of the inflow
  - *curtailed VRE share* - how much the unused flows from VRE sources would have been of the inflow
  - *upward slack share* - upward slack in relation to the inflow
  - *downward slack share* - downward slack in relation to the inflow

## Capacity and investment results

## CO2 emissions

## Reserves

## Inertia and non-synchronous generation

## Penalty values

