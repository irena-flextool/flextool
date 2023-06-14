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
** Release 

Bug fixes
- 

New features
- 
