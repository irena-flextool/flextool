# IRENA FlexTool user guide and documentation

IRENA FlexTool is an energy systems optimisation model developed for power and energy systems with high shares of wind and solar power. It can be used to find cost-effective sources of flexibility across the energy system to mitigate the increasing variability arising from the power systems. It can perform multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast and easy to use while including lot of functionality especially in the time scales where an energy balance is maintained between generation and consumption.

The instructions for installing IRENA FlexTool are [here](/readme.md).

# Essential objects for defining a power/energy system

- **Commodities**: fuels or other commodities that are either purchased or sold at a price outside of the model scope
- **Connections**: transmission lines or other transfer connections between nodes
- **Nodes**: maintain a balance between generation, consumption, transfers and storage state changes (nodes can also represent storages)
- **Profiles**: timeseries that can be used to constraint the behaviour of units, connections or storages
- **Units**: power plants or other conversion devices that take one or more inputs and turn them into one or more outputs
- **Constraints**:  
- **Groups**:

![Simple example grid](./simple_grid.png)

# Nodes

## Main definitions

The properties and operational characteristics of nodes can be set with the following parameters (available choices are marked in *italics*):

- **'name'** - unique name identifier (case sensitive)
- **'is_active'** - is the model/node/unit active in a specific scenario: *yes*, *no*
- **'has_balance'** - does the node maintain a balance for inputs and outputs: *yes*, *no*
- **'has_state'** - does the node represent a storage and therefore have a state: *yes*, *no*
- **'annual_flow'** -
- **'invest_method'** - choice of investment method
    - *only_invest* - 
    - *only_retire* - 
    - *invest_and_retire* - 
    - *not_allowed* - 
- **'inflow_method'** - choice of inflow method
    - use_original - time series from node
    - no_inflow - ignores any inserted inflow time series
    - scale_to_annual_flow - 

![image.png](./image.png)

## Data

Input data is set with the following parameters:

- **'invest_cost'** - investment cost. Constant or time series.
- **'inflow'** - Inflow into the node (negative is outflow). Constant or time series.
- **'penalty_down'** and **'penalty_up'** - penalty costs for violating the balance of the node (downwards or upwards)

# Commodities

Commodities include fuels, such as coal and natural gas, and are characterized by their price and CO2 content. Commodities are not directly injected to units (e.g. coal in the coal plant) so to be useful, they need to be assigned to commodity nodes (e.g. coal_node, gas_node). 

![image-1.png](./commodities.png)

# Connections

Connections have a name and a transfer capacity. Their operational characteristics include the transfer method, startup method, DC or AC and efficiency.

- **'left_node' and 'right_node'** -
- **'is_active'** -
- **'transfer_method'** - *regular*
- **'startup_method'** - *no_startup*
- **'invest_method'** -
- **'is_DC'** - *yes*, *no*
- **'startup_cost'** - SHOULD THIS BE IN DATA?

Investment parameters (for capacity expansion): investment method, investment cost, interest rate, lifetime. Retirement possible

# Generators
