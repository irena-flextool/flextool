- [How to define model properties](#how-to-define-model-properties)
- [Additional objects for further functionality](#additional-objects-for-further-functionality)
- [Nodes](#nodes)
- [Units](#units)
- [Connections](#connections)
- [Commodities](#commodities)
- [Profiles](#profiles)

# How to define model properties

## Timesteps and periods

FlexTool has two different kinds of time varying parameters. The first one represents a regular timeline based on timesteps. The duration of each timestep can be defined by the user. There can be multiple timelines in the database - the user needs to define which timeline to use (and what parts of the timeline should be used, as will be explained later). The timestep names in the timeline are defined by the user - they can be abstract like 't0001' or follow a datetime format of choice. However, the timestep names between different timelines must remain unique.

The second time varying dimension is `period`, which is typically used to depict assumptions about the future. One model can include multiple `solves` that the model will solve in sequence (to allow multi-stage modelling). Each solve can include multiple `periods` (so that the user can change parameter values for different parts of the future).

A parameter of particular type can be either constant/time-varying or constant/period-based. For example `inflow` is either a constant or time-varying, but it cannot be period-based.

## Timeblocksets

Timeblocks pick one or more sections from the `timeline` to form a `timeblockset`. Each timeblock defines a start and a duration. The aim of timeblocksets is to allow the modeller to create models with represeantive periods often used in the investment planning.

## Definitions

- **model**: model defines the sequence of solves to be performed (e.g. first an investment solve and then a dispatch solve)
  - *solves*: sequence of solves in the model represented with an array of solve names.
- **solve**: each solve is built from an array of periods (e.g. one period for 2025 and another for 2030). Periods use timeblocksets to connect with a timeline.
  - *period_timeblockset*: map of periods with associated timeblocks that will be included in the solve. Index: period name, value: timeblockSet name.
  - *realized_periods*: these are the periods the model will 'realize' - i.e., what periods will be reported in the results from this solve
  - *invest_periods*: array of periods where investements are allowed in this solve (applies only to objects that can be invested in)
  - *discount_years*: how far in the future each period is from the start of this solve (in years). Index: period, value: years.
  - *solver*: choice of solver (a list of possible values)
  - *solve_mode*: a single shot or a rolling solve (not functional yet, always a single shot)
- **timeblockset**: timeblocksets are sets of timeblocks with a start (from timeline) and a duration (number of time steps)
  - *block_duration* a map with index *timestep_name* that starts the timeblock and value that defines the duration of the block (how many timesteps)
- **timeline**: continuous timeline with a user-defined duration for each timestep. Timelines are used by time series data.
  - *timestep_duration*: a map with *timestep_name* as an index and *duration* as a value.
  - *timeline_duration_in_years* Total duration of the timeline in years. Used to relate operational part of the model with the annualized part of the model.
- **timeblockset__timeline**: defines which timeline object particular timeblockset is using.


# Additional objects for further functionality
- **group**: include multiple objects in a group to define common constraints (e.g. minimum VRE share)
- **reserve**: to define reserves for power systems
- **constraint**: to create user defined constraints between flow, state, and capacity variables (for nodes, units and connections)

# Nodes

## Defining how the node functions

These parameters will define how the node will behave and use the data it is given (available choices are marked in *italics*):

- **'name'** - unique name identifier (case sensitive)
- **'is_active'** - is the model/node/unit active in a specific scenario: *yes* (if not defined, then not active)
- **'has_balance'** - does the node maintain a balance for inputs and outputs: *yes* (if not defined, then balance is not maintained)
- **'has_state'** - does the node represent a storage and therefore have a state: *yes* (if not defined, then no storage)
- **'invest_method'** - choice of investment method
    - *only_invest* 
    - *only_retire* 
    - *invest_and_retire* 
    - *not_allowed* 
- **'inflow_method'** - choice how to treat inflow time series
    - *use_original* - does not scale the original time series (no value defaults here)
    - *no_inflow* - ignores any inserted inflow time series
    - *scale_to_annual_flow* - will scale the time series to match the `annual_flow` so that the sum of inflow is multiplied by 8760/`hours_in_solve`
    - *scale_in_proprotion* - calculates a scaling factor by dividing `annual_flow` with the sum of time series inflow (after it has been annualized using `timeline_duration_in_years`)

![image.png](./nodes.png)

## Data for nodes

Input data is set with the following parameters:

- **'inflow'** - inflow into the node (negative is outflow). Constant or time series.
- **'annual_flow'** - annual flow in energy units (always positive, the sign of inflow defines in/out). Constant or period.
- **'existing'** - existing storage capacity (requires `has_state`). Constant.
- **'invest_cost'** - investment cost for new storage capacity. Constant or period.
- **'invest_max_total'** - maximum investment over all solves. Constant.
- **'lifetime'** - life time of the storage unit represented by the node. Constant or period.
- **'interest_rate'** - interest rate for investments. Constant or period.
- **'fixed_cost'** - annual fixed cost for storage. Constat or period.
- **'penalty_up'** - penalty cost for decreasing consumption in the node with a slack variable. Constant or time.
- **'penalty_down'** - penalty cost for increasing consumption in the node with a slack variable. Constant or time.

# Units

Units convert energy (or matter) from one form to another (e.g. open cycle gas turbine), but the can also have multiple inputs and/or outputs (e.g. combined heat and power plant). The input nodes are defined with the relationship `unit--inputNode` while the output nodes are defined through the relationship `unit--outputNode`.

## Defining how the unit functions

- `is_active` to state the alternative where the unit becomes active
- 'conversion_method' to define the way unit converts inputs to outputs 
- `startup_method` defines how the start-up mechanism is modelled
- `minimum_time_method` - not functional at the moment

## Main data items for units

- Capacity: `existing` (and the investment and retirement parameters below)
- Technical: `efficiency`, `min_load`, `efficiency_at_min_load`, `inertia_constant`
- Economic: `variable_cost`, `startup_cost`, `fixed_cost` (fuel cost comes through the use of fuel commodities)

## Investment parameters (for capacity expansion)

- investment/retirement method
- invest_cost, interest_rate, lifetime, invest_max_total, invest_max_period
- salvage_cost, retire_max_total, retire_max_period

![image](./generators.png)

## Relationship of a unit to a node and determination of the type of relationship

- If the unit’s outputs are flowing into the node, the node acts as output for the unit.
- If the unit’s inputs are flowing out of the node (into the unit), the node acts as input for the unit.
- Not all units necessary have an input or an output node. E.g. VRE generators have only output nodes and their generation is driven by profiles

## Properties of unit--inputNode and unit--outputNode relationships

- Flow (from/to node) coefficient (changes the accounts for efficiency of unit)
- Variable cost of the particular flow of unit--inputNode or unit--outputNode

## Units constrained by profiles

Some generators (e.g. VRE) are not converting energy from one node to the other. Instead, their generation is determined (or limited) by a specific generation profile set by a `profile` object with a `profile_method`, thats state whether the profile forces an *upper_limit*, *lower_limit* or *equal*ity. Finally `profile`object is given a `profile` time series (or it can also be a constant).

# Connections

Connections can have an `existing` transfer capacity as well as an opportunity to invest in new capacity and retire old capacity. The functional choices of connections include the `is_active`, `transfer_method`, `invest_method`, `startup_method` as well as a choice if the tranfer connection `is_DC`. Parameters for the connection are defined in the `connection` object, but the two `nodes` it connects are defined by establishing a relationship between `connection--leftNode--rightNode`.

# Commodities

Some `nodes` can act as a source or a sink of commodities instead of forcing a balance between inputs and outputs. To make that happen, commodities must have a `price` and be connected to those `nodes` that serve (or buy) that particular `commodity` at the given `price`. In other words, `commodity` is separate from `node` so that the user can use the same `commodity` properties for multiple nodes. Commodities can also have a `co2_content`. The `commodity` and its `nodes` are connected by establishin a new relationship between the `commodity` and each of its `nodes` (e.g. *coal--coal_market*).

![image-1.png](./commodities.PNG)

# Profiles

Profiles are time series of data that will be multiplied by capacity to form a constraint for a unit flow, a connection flow or a storage state. The constraint can be *upper_limit*, *lower_limit* or *fixed*.
