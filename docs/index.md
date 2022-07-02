![irenalogo](./irena_flextool_logo.png)

# IRENA FlexTool user guide and documentation

IRENA FlexTool is an energy systems optimisation model developed for power and energy systems with high shares of wind and solar power. It can be used to find cost-effective sources of flexibility across the energy system to mitigate the increasing variability arising from the power systems. It can perform multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast and easy to use while including lot of functionality especially in the time scales relevant for investment planning and operational scheduling of energy systems.

The instructions for installing IRENA FlexTool are [here](https://github.com/irena-flextool/flextool/tree/master#irena-flextool).

This user guide will build a small system step-by-step. After that, there is a reference section for model properties. The small system is also available in the repository ('Init.sqlite') and can be opened with Spine Toolbox database editor. It can also be run with IRENA FlexTool (in the Spine Toolbox workflow one can initialize the Input_data.sqlite with the Init.sqlite when testing the modelling framework). More information on how to set-up and use the Spine Toolbox front-end in [here](https://github.com/irena-flextool/flextool#irena-flextool-workflow-shortly-explained).

- [Building a small test system](#building-a-small-test-system)
- [More functionality](#more-functionality)
- [Essential objects for defining a power/energy system](#essential-objects-for-defining-a-powerenergy-system)
- [Essential objects to define model properties](#essential-objects-to-define-model-properties)
- [Additional objects for further functionality](#additional-objects-for-further-functionality)
- [Nodes](#nodes)
- [Commodities](#commodities)
- [Connections](#connections)
- [Units](#units)

# Building a small test system

## 1st step - a node with no units

At first the test system shows the parameters needed to establish a working model. However, this model has only a `node` (*west*) with demand, but no units to provide the demand. It will therefore use the slack variables and accept the penalty associated with them. All parameters here are part of the *init* `alternative` - they will be used whenever a `scenario` includes the *init* `alternative`.

![First_model](./first_model.png)

## 2nd step - add a coal unit

In the second step, a coal unit is added. It needs `efficiency` and capacity (`existing`), but it also needs a new `node` *coal_market* from which it will get the *coal* `commodity` which needs a parameter for `price`. All these new parameters are part of the 'coal' `alternative`. A scenario with the initial node and the coal unit is then built by including both *init* and *coal* `alternatives` in the *coal* `scenario`. There are some extra parameters related to investments that will be useful later.

![Add unit](./add_unit.png)

Furthermore, the model needs to know that there is a link between the *coal_market* and *coal_plant* as well as *coal_plant* and the `node` *west*. These are established as relationships between objects. `unit__inputNode` relationship will therefore have *coal_plant--coal_market* relationship and `unit__outputNode` will include *coal_plant--west* relationship.

## 3rd step - add a wind power plant

Next, a wind power plant is added. The parameters for this unit include `conversion_method`, `efficiency`, `existing` and `is_active`. Note that wind does not require a commodity, but instead uses a profile to limit the generation to the available wind. A *wind_profile* object is added to the `profile` object class and the parameter `profile` is given a map of values where each time step gets the maximum available capacity factor for the time step. On the bottom of the the figure, the relationship class `unit__node__profile` gets a new member *wind_plant, west, wind_profile*, which tells the model to connect the *wind_profile* with the flow going from the *wind_plant* to the *west* `node`. There is also a parameter `profile_method` given to *wind_plant, west, wind_profile* relationship with the choice *upper_limit* selected. Now the *wind_plant* must generate at or below its capacity factor.

![Add another unit](./add_unit2.png)

## 4th step - add a network

 A *network* `alternative` introduces two new `nodes` (*east* and *north*) and three new `connections` between `nodes` (*east_north*, *west_east* and *west_north*). The new nodes are kept simple: they just have a constant negative `inflow` (i.e. demand) and penalty values for violating their energy balance, which is also required through the `has_balance` parameter. The *north* `node` has the lowest upward penalty, so the model will prefer to use that whenever the *coal* and *wind* units cannot meet all the demand. Sometimes the `existing` capacity of the new `connections` will not be sufficient to carry all the needed power, since both generators are producing to the *west* `node`.

 ![Add network](./add_network.png)

 ## 5th step - add a reserve

Reserve requirement is defined for a group of nodes. Therefore, the first step is to add a new `group` called *electricity* with *west*, *east* and *north* as its members using the `group__node` relationship class. Then, a new reserve category called *primary* is added to the `reserve` object class. 

A relationship between *primary--up--electricity* in the `reserve__upDown__group` class allows to define the reserve parameters `reserve_method`, `reservation` and `penalty_reserve`. In this case the reserve requirement will be a constant even though the `reserve_method` is *timeseries_only*. The other alternative is dynamic reserves where the model calculates the reserve requirement from generation and loads according to user defined factors (`increase_reserve_ratio`). 

Parameters from the `reserve__upDown__unit__node` class will be used to define how different units can contribute to different reserves. Parameter `max_share` says how large share of the total capacity of the unit can contribute to this reserve category (e.g. *coal_plant*, in this example, has ramping restrictions and can only provide 1% of it's capacity to this upward primary reserve. Meanwhile, parameter `reliability` affects what portion of the reserved capacity actually contributes to the reserve (e.g. in this contrived example, *wind_plant* must reduce output by 20 MW to provide 10 MW of reserve). 

 ![Add a reserve](./reserves.png)

# More functionality

## Adding a battery : init - wind - battery

In the init.sqlite, there is a `scenario` *wind_battery* - the *wind_plant* alone is not able to meet the load in all conditions, but the *battery* will help it to improve the situation.

In FlexTool, only `nodes` can have storage. This means that `existing` capacity and all investment parameters for `nodes` refer to the amount of storage the `node` can have. In this example, a *battery* `node` is established to describe the storage properties of the *battery* (e.g. `existing` capacity and `self_discharge_loss` in each hour). 

Battery also needs charging and discharging capabilities. These could be presented either with a `connection` or by having a charging `unit` and a discarging `unit`. In here, we are using a `connection` called *batter_inverter*, since its more easy to prevent simultaneous charging and discharging that way (although, in a linear model, this cannot be fully prevented since that requires an integer variable). Please note that the `efficiency` parameter of the `connection` applies to both directions, so the round-trip `efficiency` will be `efficiency` squared.

The `transfer_method` can be used by all types of connections, but in this case it is best to choose *regular*, which tries to avoid simultaneous charging and discharing, but can still do it when the model needs to dissipate energy. *exact* method would prevent that, but it would require integer variables and make the storage computationally much more expensive. Model leakage will be reported in the results (forthcoming).

![Add a battery](./battery.png)

##  Adding battery investment capabilities : init - wind - battery - battery_invest

To make the *wind_battery* `scenario` more interesting, an option to invest in *battery* and *battery_inverter* will be added. It will also demonstrate how FlexTool can have more complicated constraints that the user defines through data. 

First, the investment parameters need to be included both for the *battery_inverter* and *battery* objects:

- `invest_method` - the modeller needs to choose between *only_invest*, *only_retire*, *invest_and_retire* or *not_allowed*
- `invest_cost` - overnight investment cost new capacity [currency/kW] for the *battery_inverter* and [currency/kWh] for the *battery*. Other one can be left empty or zero, since they will be tied together in the next phase. Here we will assume a fixed relation between kW and kWh for this battery technology, but for example flow batteries could have separate investments for storage and charging capacities.
- `invest_max_total` - maximum investment (power [MW] or energy [MWh]) to the virtual capacity of a group of units or to the storage capacity of a group of nodes. This should not be empty or zero, since then the model cannot invest in the technology.
- `interest_rate` - an interest rate [e.g. 0.05 means 5%] for the technology that is sufficient to cover capital costs assuming that the economic lifetime equals the technical lifetime
- `lifetime` - technical lifetime of the technology to calculate investment annuity (together with interest rate)

Second, we need to create a new constraint that will tie together the storage capacity of the *battery* and the charging/discharging capacity of the *battery_inverter*. A new `constraint` object *battery_tie_kW_kWh* is created and it is given parameters `constant`, `is_active` and `sense`. Constant could be left out, since it is zero, but `is_active` must be defined in order to include the constraint in the *battery_invest* `alternative`. The `sense` of the constraint must be *equal* to enforce the kw/kWh relation.

Third, both *battery_inverter* and *battery* will need a coefficient that will tell the model how they relate to each other. The equation has the capacity variables on the left side of the equation and the constant on the right side.

```
sum_i(`constraint_capacity_coefficient` * `invested_capacity`) = `constant` 
      where i is any unit, connection or node that is part of the constraint
```

If we now set `constraint_capacity_coefficient` for *battery* at 1 and for *battery_inverter* at -8, the equation will force *battery_inverter* `capacity`to be 8 times smaller than the *battery* `capacity`. The negative term can be seen to move to the right side of the equation, so that we have:

```1 x *battery* = 8 x *battery_inverter*, which can be true only if *battery_inverter* is 1/8 of *battery*```

`constraint_capacity_coefficient` is not a parameter with a single value, but a map type parameter (index: constraint name, value: coefficient). It allows the object to participate in multiple constraints.

Finally, FlexTool can mix three different types of constraint coefficients: `constraint_capacity_coefficient`, `constraint_state_coefficient` and `constraint_flow_coefficient` allowing the user to create custom constraints between any types of objects in the model for the main variables in the model (*flow*, *state* as well as *invest* and *divest*). So, the equation above is in full form:

```
  + sum_i [constraint_capacity_coefficient(i) * invested_capacity]
           where i contains [node, unit, connection] belonging to the constraint
  + sum_j [constraint_flow_coefficient(j) * invested_capacity]
           where j contains [unit--node, connection--node] belonging to the constraint
  + sum_k [constraint_state_coefficient(k) * `invested_capacity]
           where k contains [node] belonging to the constraint
  = 
  constant
```

![Add battery investments](./battery_invest.png)

## Combined heat and power (CHP) example : init - coal_chp - heat

This CHP plant is an another example where the user defined `constraint` (see the last equation in the previous example) is used to make something in the model to behave in the derised manner. In a backpressure CHP, heat and power outputs are fixed - increase one of them, and you must also increase the other. In an extraction CHP plants the relation is more complicated - there is an allowed operating area between heat and power. Both can be depicted in FlexTool, but here a backpressure example is given. An extraction plant would require two or more *greater_than* and/or *lesser_than* `constraints` to define an operating area.

First, a new *heat* `node` is added and it is given the necessary parameters. Then the *coal_chp* `unit` is made with a high efficiency, since CHP units  convert fuel energy to power and heat at such rates. In FlexTool, `efficiency` is a property of the unit - it demarcates at what rate the sum of inputs is converted to the sum of outputs. However, without any additional constraints, the `unit` is free to choose in what proportion to use inputs and in which proportion to use outputs. In units with only one input and output, this freedom does not exist, but in here, the *coal_chp* needs to be constrained. 

This is done by adding a new `constraint` *coal_chp_fix* where the heat and power co-efficients are fixed. As can be seen in the bottom part of the figure below, the `constraint_flow_coefficient` parameter for the *coal_chp--heat* and *coal_chp--west* is set as a map value where the `constraint` name matches with the *coal_chp_fix* `constraint` object name. The values are set so that the constraint equation forces the heat output to be twice as large as the electricity output. Again, the negative value moves the other variable to the right side of the equality, creating this:

```1 x *electricity* = 0.5 x *heat*, which is true only if *heat* is 2 x *electricity*```

![Add CHP](./coal_chp.png)

## Minimum load example : init - coal - coal_min_load

The next example is more simple. It adds a minimum load behavior to the *coal_plant* `unit`. Minimum load requires that the unit must have an online variable in addition to flow variables and therefore a `startup_method` needs to be defined and an optional `startup_cost` can be given. The options are *no_startup*, *linear* and *binary*. *binary* would require an integer variable so *linear* is chosen. However, this means that the unit can startup partially. The minimum online will still apply, but it is the minimum of the online capacity in any given moment (*flow* >= *min_load_efficiency* x *capacity_online*).

The online variable also allows to change the efficiency of the plant between the minimum and full loads. An unit with a part-load efficiency will obey the following equation:

```
  + sum_i[ input(i) * input_coefficient(i) ]
  =
  + sum_o[ output(o) * output_coefficient(o) ] * slope
  + online * section

where   slope = 1 / efficiency - section
  and section = 1 / efficiency 
                - ( 1 / efficiency - 1 / efficiency_at_min_load) / ( 1 - efficiency_at_min_load )
```

By default, `input_coefficient` and `output_coefficient` are 1, but if there is a need to tweak their relative contributions, these coefficients allow to do so (e.g. a coal plant might have lower efficieny when using lignite than when using brown coal).

![Add min_load](./coal_min_load.png)

## Adding coal CO2 emissions : init - coal - co2

Carbon dioxide emissions of e.g. coal production can be added as a `commodity` with the parameter `co2_content` (CO2 per energy produced). The `price` (price per energy produced) of the emission is linked to a `group`.

![Add CO2](./coal_co2.png)

## Full year model : init - fullYear

![fullYear](./fullYear.png)

## System with coal, wind, network, battery and CO2 over a full year : init - coal - wind - network - battery - co2 - fullYear

![Entity graph](./coal_wind_chp_battery_graph.png)

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
- **'annual_flow'** - annual flow in energy units (always positive, the sign of inflow defines in/out). 
- **'existing'** - existing storage capacity (requires `has_state`)
- **'invest_cost'** - investment cost for new storage capacity. Constant or time series.
- **'penalty_up'** - penalty cost for decreasing consumption in the node with a slack variable
- **'penalty_down'** - penalty cost for increasing consumption in the node with a slack variable

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
- 


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
- Not all units necessary have an input node. E.g. VRE generators have only output nodes and their generation is driven by profiles

## Relationship properties:

- Flow (from/to node) coefficient (accounts for efficiency of unit)
- Variable cost of flow

Generators are associated with nodes.

## Generators driven by profiles

Some generators (e.g. VRE) are not converting energy from one node to the other. Instead, their generation is determined (or limited) by a specific generation profile.

Association of profile-unit and determination of profile method.

- profile: solar_capacity_factor, ...
- profile_method: upper_limit, ...
