# IRENA FlexTool user guide and documentation

IRENA FlexTool is an energy systems optimisation model developed for power and energy systems with high shares of wind and solar power. It can be used to find cost-effective sources of flexibility across the energy system to mitigate the increasing variability arising from the power systems. It can perform multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast and easy to use while including lot of functionality especially in the time scales where an energy balance is maintained between generation and consumption.

The instructions for installing IRENA FlexTool are [here]</readme.md>.

# Essential objects for defining a power/energy system

Nodes: maintain a balance between generation, consumption, transfers and storage state changes (nodes can also represent storages)

Commodities: fuels or other commodities that are either purchased or sold at a price outside of the model scope

Units: power plants or other conversion devices that take one or more inputs and turn them into one or more outputs

Connections: transmission lines or other transfer connections between nodes

Profiles: timeseries that can be used to constraint the behaviour of units, connections or storages

![Simple example grid](./simple_grid.png)

# Nodes

## Main definitions

name - unique name to identify the name (case sensitive)
'is_active' - allows to make the node active in a specific scenarios
'has_balance' - does the node maintain a balance?
'has_state' - does the node represent a storage and therefore has a state 

## Data

'inflow' - Inflow into the node (negative is outflow). Constant or time.
