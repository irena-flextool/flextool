** Release 3.2.0 (23.8.2023)
Bug fixes
- Storage state, unit flows and ramps were not properly limited in multi-period investment models

New features
- *Breaking change*: `realized_periods` (solve objects) is supplemented with a `realized_invest_periods` so that the user can state from which solves the results should take investment results and from which solves dispatch results.
- New parameters for solve objects: `solve_mode`, `rolling_duration`, `rolling_solve_horizon`, `rolling_solve_jump` and `rolling_start_time` that enable to build rolling window models. See [How to use a rolling window for a dispatch model](#How-to-use-a-rolling-window-for-a-dispatch-model).
- New parameter `contains_solves` that enables nesting solves inside solves (e.g. to calculate shadow values for long term storages or to implement rolling dispatch inside a multi-year investment model). See [How to use Nested Rolling window solves (investments and long term storage)](#How-to-use-nested-rolling-window-solves).
- New outputs: For groups: `VRE_share_t`. For unit__nodes (VRE units): `curtailment_share`, `curtailment_share_t`.
- Name changes to outputs: `flow` to `flow_annualized`, `sum_flow` to `sum_flow_annualized`.
- Add migration for results database (parameter descriptions can be migrated).

** Release 3.1.4 (14.6.2023)
Bug fixes 
- Cancelling of plot_results will not freeze Toolbox

New features 
- Lifetime_method (reinvest_automatic and reinvest_choice) so that user can choose whether assets will be automically renewed at the end of lifetime or the model has to make that choice
- Commercial solver support (CPLEX)
- Database migration. init.sqlite and input_data.sqlite will be updated to the latest version when update_flextool.py is run. It is also possible to update any database using migrate_database.py. After `git pull`, run `python -m update_flextool.py`. In future, just `python -m update_flextool.py` is sufficient. Best to update Spine Toolbox before doing this.

** Release 3.1.3 (1.6.2023)
Bug fixes
- Use of capacity margin caused an infeasibility
- Certain inflow time series settings crashed the model (complained about ptNode[n, 'inflow', t])
- Investments did not consider lifetime correctly - units stayed in the system even after lifetime ended
- Lifetime is now calculated using years_represented
- Fixed how retirements work over multiple solves

New features
- Added support for commercial solvers (CPLEX explicitly at this point)


** Release 3.1.2 (23.5.2023)
Bug fixes
- Node prices were million times smaller than they should have been (bug introduced when scaling the model in 3.1.0)
- Non-synchronous limit was not working.

New features
- Documentation structure was improved
- Model assumes precedence between storage_start_end_method, storage_binding_method, and storage_solve_horizon_method (in that order)


** Release 3.1.1 (13.5.2023)

Bug fixes
- The calculation of the p_entity_max_capacity did not consider which investment method was actually active. It also limited capacity in situations where investment method was supposed to be 'invest_no_limit'.


** Release 3.1.0 (13.5.2023)

Bug fixes
- Division by zero capacity for units with cf
- Output fixed costs for existing units
- Fix output calculation for annualized fixed costs
- Fix error in importing CO2_methods for groups
- Add other_variable_cost for flows from unit to sink
- 


New features
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


** Release 3.0.1 (17.3.2023)

Bug fixes
- CO2_method for groups was not imported correctly from Excel inputs
- Fix other_variable_cost in units for flows from unit to sink (source to unit was working)

New features
- Output fixed costs for existing units
- Added outputs to show average capacity factors for periods

** Release 3.0.0 (11.1.2023)

All planned features implemented


----
TEMPLATE
** Release (dd.mm.yyyy)

Bug fixes
- 

New features
- 
