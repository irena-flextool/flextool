** Release 3.8.0 (29.8.2024)
**Bug fixes**
- Excel input file to use entity_alternative
- unit_curtailment_share fix to Excel input

**New features**
- Existing capacity can be set for periods
- Cumulative limits for investments in nodes, units and connections
- New node_type that has period limit for inflow
- New highs solver version

## Release 3.7.0 (24.6.2024)
**Bug fixes**
- Several bugs related to the migration to Spine Toolbox 0.8.
- CPLEX call without pre_command fixed
- Timeline aggregarion without all timesteps
- User constraint including capacity fixed to include unitsize
- User constraint including flows fixed to include step_duration
- Creating multiple invests with 2D maps fixed
- Update_flextool.py/ migrate_database.py crashing if nothing to commit 
- Nested structure:
  - Pass end state only from dispatch level
  - Storage fix_quantity overrides other storage options

**New features**

- Added non-anticipatory constraint to stochastic branches. The brnaches now start from the first timestep.
- Storage_reference_value: Now possible to set seperately for multiple end times as time step map.
- Nested structure storage_nested_fix_method: 
  - Added fix_usage
  - Fix_price changed to set the price to the objective function of the lower level solve


## Release 3.6.0 (15.5.2024)
**New features**

- Upgraded FlexTool to Spine Toolbox v0.8
- Entity alternative replaces is_active parameter


## Release 3.5.0 (29.4.2024)
**Bug fixes**
- Stochastic and rolling weights


## Release 3.4.0 (13.4.2024)
**Bug fixes**
- Inertia with MIP units added missing unitsize
- Result ordering improvements
- Fixing how flow variables are limited
- Min_load to work with multiple inputs/outputs¨
- Capacity limit includes the efficiency

**New features**

- Output of inertia seperately for units
- Default penalty values for reserve, inertia, capacity margin and non-synchronous limits
- Large model data processing speed-ups


## Release 3.3.0 (23.3.2024)
**Bug fixes**

-	Investments: maxState to only sum over the investments in this node, not all the nodes
-	Ramp constraint now correctly calculates the investments done in previous periods
-	Min and Max Invest/Divest constraints to only affect investment periods
-	Efficiency works with profile units
-	Allow upward and downward slacks to be more than the inflow
-	Potential VRE generation now includes efficiency and availability
-	Non-sync group constraint now excludes the flows inside the group
-	Two-way MIP connection to allow only one direction at a time
-	Connection variable cost to the objective function
-	Allow investing with just `invest_periods` without `realized_periods`
- unit_inputNode: `coefficient` applied correctly to units with multiple inputs 

**New features**

- Two-stage stochastic modelling of future uncertainty
-	Better infeasibility and required parameter checks to inform the user what is wrong with the model
-	Small changes to some result data parameters
-	Added availability (constant /timeseries) parameter for entities
-	Several outputs to be optional to hasten the data transfer.
-	Added group flow output parameter `output_aggregate_outputs` to give the flow of the desired nodes in a grouped format
-	Option for loss of load sharing `share_loss_of_load` between nodes in a group to better describe the system where loss of load is present
-	A workflow item to display the summary file from model runs
-	Database with pre-made time settings for the users to start with *time_settings_only.sqlite*
-	C02 max total costraint
-	Default values for: 
    -	Penalty values
    -	Efficiency
    -	Constraint constant
    -	Reserve reliability

**Documentation**

-	Introduction
-	Improved tutorial (including how-tos split of into their own section)
-	Installation / update instructions
-	Theory slides
-	How to section to guide on making specific parts of the model

Building parts of the model:

-	How to create basic temporal structures for your model
-	How to create a PV, wind or run-of-river hydro power plant
-	How to connect nodes in the same energy network
-	How to set the demand in a node
-	How to add a storage unit (battery)
-	How to make investments (storage/unit)
-	How to create combined heat and power (CHP)
-	How to create a hydro reservoir
-	How to create a hydro pump storage
-	How to add a reserve
-	How to add a minimum load, start-up and ramp
-	How to add CO2 emissions, costs and limits
-	How to create a non-synchronous limit
-	How to see the VRE curtailment and VRE share results for a node

Setting different solves:

-	How to run solves in a sequence (investment + dispatch)
-	How to create a multi-year model
-	How to use stochastics (represent uncertainty)

General:

-	How to use CPLEX as the solver
-	How to create aggregate outputs
-	How to enable/disable outputs
-	How to make the Flextool run faster


## Release 3.2.0 (23.8.2023)
**Bug fixes**
- Storage state, unit flows and ramps were not properly limited in multi-period investment models.
- Removed limits from vq_state (they were causing infeasibilities in some case).
- Added time changing efficiency to profile based flows.
- Flows from node to unit in unit__node are marked negative (were positive).

**New features**
- *Breaking change*: `realized_periods` (solve objects) is supplemented with a `realized_invest_periods` so that the user can state from which solves the results should take investment results and from which solves dispatch results.
- New parameters for solve objects: `solve_mode`, `rolling_duration`, `rolling_solve_horizon`, `rolling_solve_jump` and `rolling_start_time` that enable to build rolling window models. See [How to use a rolling window for a dispatch model](#How-to-use-a-rolling-window-for-a-dispatch-model).
- New parameter `contains_solves` that enables nesting solves inside solves (e.g. to calculate shadow values for long term storages or to implement rolling dispatch inside a multi-year investment model). See [How to use Nested Rolling window solves (investments and long term storage)](#How-to-use-nested-rolling-window-solves).
- New outputs: For groups: `VRE_share_t`. For unit__nodes (VRE units): `curtailment_share`, `curtailment_share_t`.
- Name changes to outputs: `flow` to `flow_annualized`, `sum_flow` to `sum_flow_annualized`.
- Add migration for results database (parameter descriptions can be migrated).
- Documentation updates: new how-to sections.
- Faster outputting of ramp results.
- update_flextool.py will leave the Toolbox project untouched.
- Disable execute project button from the FlexTool workflow.

## Release 3.1.4 (14.6.2023)
**Bug fixes**
- Cancelling of plot_results will not freeze Toolbox

**New features**
- Lifetime_method (reinvest_automatic and reinvest_choice) so that user can choose whether assets will be automically renewed at the end of lifetime or the model has to make that choice
- Commercial solver support (CPLEX)
- Database migration. init.sqlite and input_data.sqlite will be updated to the latest version when update_flextool.py is run. It is also possible to update any database using migrate_database.py. After `git pull`, run `python -m update_flextool.py`. In future, just `python -m update_flextool.py` is sufficient. Best to update Spine Toolbox before doing this.

## Release 3.1.3 (1.6.2023)
**Bug fixes**
- Use of capacity margin caused an infeasibility
- Certain inflow time series settings crashed the model (complained about ptNode[n, 'inflow', t])
- Investments did not consider lifetime correctly - units stayed in the system even after lifetime ended
- Lifetime is now calculated using years_represented
- Fixed how retirements work over multiple solves

**New features**
- Added support for commercial solvers (CPLEX explicitly at this point)


## Release 3.1.2 (23.5.2023)
**Bug fixes**
- Node prices were million times smaller than they should have been (bug introduced when scaling the model in 3.1.0)
- Non-synchronous limit was not working.

**New features**
- Documentation structure was improved
- Model assumes precedence between storage_start_end_method, storage_binding_method, and storage_solve_horizon_method (in that order)


## Release 3.1.1 (13.5.2023)

**Bug fixes**
- The calculation of the p_entity_max_capacity did not consider which investment method was actually active. It also limited capacity in situations where investment method was supposed to be 'invest_no_limit'.


## Release 3.1.0 (13.5.2023)

**Bug fixes**
- Division by zero capacity for units with cf
- Output fixed costs for existing units
- Fix output calculation for annualized fixed costs
- Fix error in importing CO2_methods for groups
- Add other_variable_cost for flows from unit to sink

**New features**
- New investment method with a limited horizon (model used to assume infinite horizon)
  - *Braking change*: new parameter 'years_represented', which replaces 'discount_years'. It is a map to indicate how many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple periods in one year). Index: period, value: years.
- Added sidebar to documentation (gh-pages branch)
- Default solver changed from GLPSOL to HiGHS
- Documentation improvements
- Scaling of the model resulting in very big performance improvement for linear problems
- Improved plotting for the web browser version
- In Spine Toolbox, plotting is based on the specification from the browser version
- More results: group flows over time
- More results: costs for whole model run
- More result plots

## Release 3.0.1 (17.3.2023)

**Bug fixes**
- CO2_method for groups was not imported correctly from Excel inputs
- Fix other_variable_cost in units for flows from unit to sink (source to unit was working)

**New features**
- Output fixed costs for existing units
- Added outputs to show average capacity factors for periods

## Release 3.0.0 (11.1.2023)

All planned features implemented



TEMPLATE

## Release (dd.mm.yyyy)

**Bug fixes**
- foo

**New features**
- bar
