![irenalogo](./irena_flextool_logo.png)

# IRENA FlexTool user guide and documentation

IRENA FlexTool is an energy systems optimisation model developed for power and energy systems with high shares of wind and solar power. It can be used to find cost-effective sources of flexibility across the energy system to mitigate the increasing variability arising from the power systems. It can perform multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast and easy to use while including lot of functionality especially in the time scales where an energy balance is maintained between generation and consumption.

The instructions for installing IRENA FlexTool are [here](https://github.com/irena-flextool/flextool/tree/master#irena-flextool).

# Essential objects for defining a power/energy system

- **nodes**: maintain a balance between generation, consumption, transfers and storage state changes (nodes can also represent storages)
- **units**: power plants or other conversion devices that take one or more inputs and turn them into one or more outputs
- **connections**: transmission lines or other transfer connections between nodes
- **commodities**: fuels or other commodities that are either purchased or sold at a price outside of the model scope
- **profiles**: timeseries that can be used to constraint the behaviour of units, connections or storages

See below for more detailed explanations.

![Simple example grid](./simple_grid.png)

# Essential objects to define model properties

- **model**: model defines the sequence of solves to be performed (e.g. first an investment solve and then a dispatch solve)
- **solve**: each solve is built from an array of periods (e.g. one period for 2025 and another for 2030). Periods use timeblocksets to connect with a timeline.
- **timeblockset**: timeblocksets are sets of timeblocks with a start (from timeline) and a duration (number of time steps)
- **timeline**: continuous timeline with a user-defined duration for each timestep. Timelines are used by time series data.
 
# Additional objects for further functionality
- **group**: include multiple objects in a group to define common constraints (e.g. minimum VRE share)
- **reserve**: to define reserves for power systems
- **constraint**: to create user defined constraints between flow, state, and capacity variables (for nodes, units and connections)
 
# Nodes

## Main definitions
## Functionality

The properties and operational characteristics of nodes can be set with the following parameters (available choices are marked in *italics*):

- **'name'** - unique name identifier (case sensitive)
- **'is_active'** - is the model/node/unit active in a specific scenario: *yes* (if not defined, then not active)
- **'has_balance'** - does the node maintain a balance for inputs and outputs: *yes* (if not defined, then balance is not maintained)
- **'has_state'** - does the node represent a storage and therefore have a state: *yes* (if not defined, then no storage)
- **'invest_method'** - choice of investment method
    - *only_invest* 
    - *only_retire* 
    - *invest_and_retire* 
    - *not_allowed* 
- **'inflow_method'** - choice of inflow method
    - *use_original* - time series from node
    - *no_inflow* - ignores any inserted inflow time series
    - *scale_to_annual_flow* - 

![image.png](./nodes.png)

## Data

Input data is set with the following parameters:

- **'invest_cost'** - investment cost. Constant or time series.
- **'annual_flow'** - annual flow in energy units (always positive, the sign of inflow defines in/out)
- **'inflow'** - Inflow into the node (negative is outflow). Constant or time series.
- **'penalty_down'** and **'penalty_up'** - penalty costs for violating the balance of the node (downwards or upwards)
- **'startup_cost'**
- **'annual_flow'**

# Commodities

Commodities are characterized by their price and CO2 content. Commodities are not directly injected to units (e.g. coal to the coal plant) so to be useful, they need to be assigned to commodity nodes (e.g. coal_node, gas_node). 

![image-1.png](./commodities.PNG)

# Connections

Connections have a name and a transfer capacity. Their operational characteristics include the transfer method, startup method, DC or AC and efficiency.

- **'left_node' and 'right_node'**
- **'is_active'**
- **'transfer_method'** - *regular*
- **'startup_method'** - *no_startup*
- **'invest_method'**
- **'is_DC'**
    - *yes*, *no*

Investment parameters (for capacity expansion): investment method, investment cost, interest rate, lifetime. Retirement possible

# Units

Units convert energy (or matter) from one form to another (e.g. open cycle gas turbine).

Units definition

- Unit names (e.g. coal_plant, hydro_plant, solar_pv), capacities.

variable_cost

Operational characteristics

- Energy conversion method (conversion_method), startup method (startup_method), minimum up/down time method (minimum_time_method)
- Technical: Minimum load (min_load), efficiency, efficiency at min load (efficiency_at_min_load), minimum up/down time
- Economic: Variable O&M cost, startup cost.
- is_active

Investment parameters (for capacity expansion)

- Investment method, investment cost, interest rate, lifetime
- Retirement possible

![image](./generators.png)

Generators are associated with nodes.

## Relationship of a unit to a node and determination of the type of relationship:

- If the unit’s outputs are flowing into the node, the node acts as output for the unit.
- If the unit’s inputs are flowing out of the node (into the unit), the node acts as input for the unit.
- Not all units necessary have an input node. E.g. VRE generators have only output nodes and their generation is driven by profiles (next slide).

## Relationship properties:

- Flow (from/to node) coefficient (accounts for efficiency of unit)
- Variable cost of flow

Generators are associated with nodes.

### Generators driven by profiles

Some generators (e.g. VRE) are not converting energy from one node to the other. Instead, their generation is determined (or limited) by a specific generation profile.

Association of profile-unit and determination of profile method.

- profile: solar_capacity_factor, ...
- profile_method: upper_limit, ...

# Defining a battery

In Flextool 3, batteries are modeled with 
- 1 battery node, which represents the storage capacity (MWh) of the battery (energy part of the battery)
- 1 connection which transfers energy to and from the battery node (power part of the battery)

Storage specific parameters of a node: has_state (has storage), existing (MWh), self_discharge_loss

- **self_discharge_loss**
- **virtual_unitsize**
- **transfer_method**
- **fixed_cost**
- **variable_cost**
- **efficiency**

