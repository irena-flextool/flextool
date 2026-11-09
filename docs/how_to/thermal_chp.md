# Thermal, CHP and reserves

## How to create combined heat and power (CHP) 

**(examples.sqlite scenario: coal_chp)**
***init - west - coal_chp - heat***

First, a new *heat* `node` is added with the necessary parameters. The `nodes` can be used for energy form of energy or matter, so the heat demand node does not differ from the electricity demand node. 

The *heat* `node` needs to be added to an alternative in the `Entity Alternative` sheet. 

The required parameters are:

- `node_type`: balance
- `inflow`: Map for the heat demand (negative) [MW]
- `penalty_up`: a large number to prefer not creating energy from nowhere
- `penalty_down`: a large number to prefer not creating energy from nowhere

The heating systems tend to have some level of storage capability, so one could also add storage parameters to the node as well, but here they are not used.

Then the *coal_chp* `unit` is made with a high `efficiency` parameter, since CHP units convert fuel energy to power and heat at high overall rates. In FlexTool, `efficiency` is a property of the unit - it demarcates at what rate the sum of inputs is converted to the sum of outputs. However, without any additional constraints, the `unit` is free to choose in what proportion to use inputs and in what proportion to use outputs. In units with only one input and output, this freedom does not exist, but in here, the *coal_chp* needs to be constrained as otherwise the unit could produce only electricity and no heat at 90% efficiency, which is not feasible. 

This CHP plant is an another example where the user defined `constraint` (see the last equation in the previous example) is used to achieve the desired behaviour. In a backpressure CHP, heat and power outputs are fixed - increase one of them, and you must also increase the other. In an extraction CHP plant the relation is more complicated - there is an allowed operating area between heat and power. Both can be represented in FlexTool, but here a backpressure example is given. An extraction plant would require two or more *greater_than* and/or *lesser_than* `constraints` to define an operating area.

For extraction CHP specifically, the `unit__inputNode` / `unit__outputNode` coefficients interact. `conversion_flow_coeff` multiplies into the edge's contribution in the conversion equation (`conversion_indirect`) — a higher coefficient means more input energy per unit of that edge's output, so a premium electricity output might use `conversion_flow_coeff = 2` while a byproduct heat output uses `0.2`. The unit capacity caps the sum of the outputs, and `capacity_max_coeff` on an output adds a per-output upper cap. With a minimum load (`min_load` plus a `startup_method`), the floor `electricity + heat >= min_load × online capacity` applies once, to the sum of the outputs, so both the pure-condensing mode (heat = 0) and the pure-heat mode (electricity = 0) remain available while online with the default coefficients. Only if an output must carry part of the minimum load on its own, set `capacity_min_coeff` on that output (e.g. 1 on electricity: `electricity >= min_load × online capacity`, which rules out the pure-heat mode). On the inputs, `input_share_max` limits the share of the full-load input a single input may supply (default 1: no effective limit) and `input_share_min` sets the smallest share of the current input an input must supply while the unit runs (default 0). Without the capacity-coefficient split, each output would be capped individually at `conversion_flow_coeff × capacity`, which prevents the model from using the full operating envelope.

Electricity and heat outputs are fixed by adding a new `constraint` *coal_chp_fix* where the heat and power co-efficients are fixed. You need to create the two entities `unit__outputNode`, for *coal_chp--heat* and *coal_chp--west*. As can be seen in the bottom part of the figure below, the `constraint_flow_coeff` parameter for the *coal_chp--heat* and *coal_chp--west* is set as a map value where the `constraint` name matches with the *coal_chp_fix* `constraint` entity name. The values are set so that the constraint equation forces the heat output to be twice as large as the electricity output.

Create constraint *coal_chp_fix*, add it to an alternative in the `Entity Alternative` sheet and add parameters:

- `sense`: equal
- `constant`: 0.0

Create `unit_outputNode` (coal_chp|heat):

- `constraint_flow_coeff` : *coal_chp_fix*, -0.5

Create `unit_outputNode` (coal_chp|west):

- `constraint_flow_coeff` : *coal_chp_fix*, 2


Again, the negative value can be turned positive by arranging it to the right side of the equality, creating this:

```
1 x *electricity* = 0.5 x *heat*, which is true only if *heat* is 2 x *electricity*
```

![Add CHP](../img/tutorial/coal_chp.png)

## How to add a reserve
**(examples.sqlite: scenario network_coal_wind_reserve)**

In FlexTool, reserves are defined for a group of nodes. If there is a need to have a reserve requirement for a single node, it needs its own group. Therefore, when creating a reserve, the first step is to add a new `group` (e.g. *electricity*) with all member nodes (e.g. *west*, *east* and *north*) using the `group__node` entity. Then, a new reserve categories can be added (e.g. *primary*) to the `reserve` entity. 
Finally, make sure there are *up* and *down* entities in the `UpDown' entity. These are hard-coded names in FlexTool and need to be used when creating reserves.

Next, the reserve requirement will be defined. An entity between the `reserve__upDown__group` class (e.g. *primary--up--electricity*) allows to define the reserve parameters `reserve_method`, `reservation` (i.e. the amount of reserve) and `penalty_reserve` (i.e. the penalty cost in case of lack of reserve). For example, a constant of 10 MW could be used. Even though the name of the `reserve_method` is *timeseries_only*, it can also accept a constant value - it's an exogenous reserve requirement whereas the other two reserve methods are endogenous. Dynamic reserve method calculates the reserve requirement from generation and loads according to user defined factors (`increase_reserve_ratio`). Largest failure method will force enough reserve to cope with a failure of the chosen unit and connection flows.

Parameters from the `reserve__upDown__unit__node` class should be used to define how different units can contribute to different reserves. Note that the entities in this class need to be added to the `Entity Alternative` sheet. Parameter `max_share` says how large a share of the unit's capacity at that node (its built capacity: existing + invested − retired; at a node the unit consumes from, its full-load input there) can be held for this reserve category (e.g. *coal_plant* may be limited by ramp constraint to provide only 1% of its capacity to an upward primary reserve.) Meanwhile, parameter `reliability` affects what portion of the reserved capacity actually contributes to the reserve (e.g. *wind_plant* may contribute only 80% of its generation to reserve due to uncertainty).

If the reserve is provided by a **battery** or other storage (a unit discharging from, or a connection/charger tied to, a storage node), set `reserve_duration` (hours) on the `reserve__upDown__group` entity so the storage actually holds enough energy to sustain the committed reserve power. For example, a 4-hour upward reserve from a battery uses `reserve_duration = 4`: the model then guarantees the battery's stored energy can back its reserve commitment for 4 hours (corrected by the round-trip efficiency). Without `reserve_duration` the reserve is instantaneous-power-only — the battery could commit reserve it has no energy to deliver. The coupling only applies to storage-backed providers; ordinary fuel-fed units are unaffected.

 ![Add a reserve](../img/concept/reserves.png)

## How to add a minimum load, start-up and ramp
**(ramp_and_start_up.sqlite)**

Some plants cannot vary their output freely, but have some cost and limits associated with it. In this example we will add the minimum load behaviour to a *coal_plant* `unit` and add the cost and limits for starting up and ramping the plant respectively.

Minimum load requires that the unit must have an online variable in addition to flow variables and therefore a `startup_method` needs to be defined and an optional `startup_cost` can be given. The `startup_cost` is added to the total cost every time the unit is required to start-up. Here we use a value 1000 (Currency/MW started up).

The options for the `startup_method` are *no_startup*, *linear* and *binary*. *binary* would require an integer variable so *linear* is chosen. However, this means that the unit can startup partially. The minimum online will still apply, but it is the minimum of the online capacity in any given moment (*sum of output flows* >= *min_load* x *capacity* x *online*), where 0 <= *online* <= 1. The minimum load applies once per unit, to the sum of its outputs; a unit with several inputs can run on any mix of them.

The units online in a time step are linked to the units online in the previous step along the timeline, in the same way as ramp limits: also from the last step of a period into a period whose time steps continue it on the same timeline, and into the first step of a forecast branch. A step without a previous step — the first step of a solve, the first step of a period that does not continue the previous one (e.g. a period that reuses the same timeset), the first step after a gap in the timeline and the first step of a representative period — is linked to the last step of its own stretch of the timeline (the steps up to the next gap or the end of the period): the commitment cycles within that stretch and never across a gap, a representative-period start or into another period. So a stretch that ends offline starts up (and pays `startup_cost`) at its first step when it needs the unit there. Only the start of a stretch wraps: the step before a gap has no next step, so no pre-shutdown output limit applies there. In a rolling solve the units online that the previous roll realized are carried to the steps of the next roll that continue its last realized step on the timeline (also when the next solve has longer time steps: it continues the finer step that ends where its previous step ends); a roll that starts after a gap, at a new year that reuses the timeset or at a representative-period start takes no carry and cycles within its stretch.

The online variable also allows to change the efficiency of the plant between the minimum and full loads. A unit with a part-load efficiency will obey the following equation:

```
  + sum_i[ input(i) * input_coefficient(i) ]
  =
  + sum_o[ output(o) * output_coefficient(o) ] * slope
  + online * section

where   slope = 1 / efficiency - section
  and section = 1 / efficiency 
                - ( 1 / efficiency - min_load / efficiency_at_min_load) / ( 1 - min_load )
```

By default, `input_coefficient` and `output_coefficient` are 1, but if there is a need to tweak their relative contributions, these coefficients allow to do so (e.g. a coal plant might have lower efficiency when using lignite than when using brown coal).

The input is required at different ouput levels is shown in the figure below, when Capacity = 100, Efficiency = 0.8, Minimum load = 0.6 and Efficiency at minimum load = 0.5.

![Min load figure](../img/tutorial/minimum_load.png)

Next we will add ramp limits. With the ramping limits, the user can force the change of a flow from a unit to be below a certain value each timestep. The ramping is an attribute of the flow. Therefore it does not require the minimum load behaviour and its parameters are added to the `unit_outputNode` entity:

- `ramp_method`: ramp_cost, ramp_limit or both. Only ramp limit is currently implemented (August 2023).
- `ramp_speed_up`: Limit on how fast the plant can ramp up. (fraction of unit / min) ie. Value 0.01 allows the change of 60% of capacity per hour; with 2-hour time steps, 120% (i.e. no limit). The change between two time steps of different length uses their average length.
- `ramp_speed_down`: Limit on how fast the plant can ramp down. (fraction of unit / min) ie. Value 0.01 allows the change of 60% of capacity per hour.

The first time step of a solve, the first step after a gap in the timeline and the first step of a representative period have no ramp limit (there is no previous step to ramp from). A rolling window ramps from the flow of the previous window where it continues the previous window's last realized step on the timeline; a solve with longer time steps continues the finer step that ends where its own previous step ends (by position in time). Unlike the online status, the ramp has no cyclic link from the first step of a stretch to its last step.

- `ramp_cost`: NOT YET IMPLEMENTED. The cost of ramping the capacity. [CUR/MW] 

![Add min_load](../img/tutorial/coal_min_load.png)

## How to add minimum uptime and downtime

Some power plants cannot be started and stopped freely due to thermal stress, safety requirements, or operational constraints. Minimum uptime forces a unit to stay online for a specified duration after starting up. Minimum downtime forces a unit to stay offline for a specified duration after shutting down.

To add minimum uptime and/or downtime constraints to a `unit`:

1. Set the `minimum_time_method` parameter on the unit:
    - *min_uptime* - only enforce minimum uptime
    - *min_downtime* - only enforce minimum downtime
    - *both* - enforce both constraints
    - *none* - no minimum time constraints (default)

2. Set the duration parameters (in hours):
    - `min_uptime` - e.g. 8 for an 8-hour minimum online time
    - `min_downtime` - e.g. 4 for a 4-hour minimum offline time

Setting `minimum_time_method` to anything other than *none* will automatically activate online variables for the unit (at least linear startup), even if `startup_method` is not explicitly defined. This means you do not need to separately set `startup_method` to *linear* or *binary*, although you can if you also want startup costs or binary commitment.

The minimum time constraints work with variable timestep durations. The constraint checks that the cumulative duration of consecutive online (or offline) timesteps meets the minimum requirement, rather than simply counting timesteps.

The look-back follows the timeline in the same way as ramp limits: it continues into a period whose time steps continue the previous period's on the same timeline and into the steps before a forecast branch starts, and stops at a gap in the timeline (the minimum time restarts after the gap), at the start of a representative period, at a period that reuses the same timeset as the previous one and at the first step of a solve. In rolling window solves, the start-ups and shut-downs that the previous roll realized count for the time steps of the next roll that continue its last realized step on the timeline, in the same way as the ramp continues: the same period, a period that continues it and every forecast branch that starts there (not a period that reuses the same timeset, a step after a gap or a representative-period start). So the solve horizon overlap does not need to cover the minimum times. A solve with longer time steps (for example a 6-hour storage roll after an hourly dispatch) continues the realized hourly step that ends where its own previous step ends, by position in time. A warning is printed only when a solve continues the timeline of an earlier solve but its first step neither follows nor ends at the earlier solve's last realized step (a hole or an overlap between the two), since no realized on-line history reaches it then.

If the unit also has ramp limits (`ramp_method` with `ramp_speed_up`, `ramp_speed_down`), the model additionally tightens the maximum output in the startup and pre-shutdown timesteps based on how fast the unit can ramp within one timestep. A unit that starts in a timestep reaches on average at most `min_load + ½ × ramp_speed_up × 60 × timestep duration` of the started capacity in that step (it synchronises at minimum load and then ramps up: half a step's ramp on average), and in the step before it shuts down at most `min_load + ½ × ramp_speed_down × 60 × timestep duration` (both capped at full output and by `availability`). A unit online for a single step gets the smaller of the two. On an input with a ramp limit of a unit with several inputs or outputs, the limit is on the input it draws (at minimum load it needs the input of its minimum-load output). Without ramp limits, the unit can reach full output immediately after starting.

Startups and shutdowns count exactly the units coming online and going offline. A unit with online variables ramps on its online capacity: the units online in both steps change by at most the ramp; a unit starting in a step reaches at most the startup-step output above, and a unit shutting down drops from at least its minimum load (and, in the step before, at most the pre-shutdown output). Starting and stopping units in the same step therefore gives no extra ramp room on a unit's output. On an input, or on one output of a unit with several outputs, a start/stop pair can still gain up to the minimum-load share of that flow, which the startup cost pays for.
