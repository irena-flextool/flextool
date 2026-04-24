## Release 3.28.0 (24.4.2026)
**Bug fixes**
- Default HiGHS to serial simplex (`parallel=off`, `threads=1`) to avoid non-determinism and occasional stalls on small models; defaults reinforced in `solver_runner` as belt-and-suspenders
- Restore 0.05 `discount_rate` default and related advisory defaults that had drifted
- Revert slack primary+escape split — back to single-variable slacks with penalty acting as a valve

**New features**
- HiGHS upgraded to 1.14 (`highspy>=1.14`)
- Auto-scaling of LP/MIP numerics: row scaling via `node_cap`/`group_cap`, objective scaling, and bound scaling with diagnostics printed
- `--auto-scale` CLI flag that gates objective/state auto-apply behaviour
- `--highs-threads` CLI flag
- HiGHS `mip_detect_symmetry` enabled — ~15× speed-up on unit-commitment with identical units
- `virtual_unitsize` documented as a speed lever for UC with identical units
- `unidirectional` transfer method for connections
- Cost-aggregation semantics fixed — weighting factors applied per variable class
- Divest salvage included in objective; `pre_existing` renamed to `all_existing`; storage valuation documented
- Auto-seed missing `output_info`, `output_settings` and `comparison_settings` databases on first run
- Seed `bin/highs.opt` from tracked template on first run
- `scaling_benchmark` harness relocated to `benchmarks/scaling`; unit tests moved to `tests/`
- `min_load_efficiency` section-term test promoted from xfail to passing

## Release 3.27.0 (22.4.2026)
**Bug fixes**
- Annualize `nodeGroup_flows_d_gpe` to match the `node_d_ep` convention
- Column misalignment in `ed_lifetime_fixed_cost[_divest].csv` writers
- Harden axis-manifest override against `None` active_scenarios
- Sum only over `stack_levels` in shared axis-bound resolution
- Reserve provision with a timeseries reserve requirement
- Reserve provision active only when there is demand
- Delete dead plot functions

**New features**
- Parameter groups — every parameter now assigned to a group; "Outputs" renamed to "output"
- Parameter-group colours re-tuned for readability in both light and dark theme
- Colour template infrastructure for plots: category-based colouring (costs, node_flows) and entity_class.group colouring (flowGroup)
- Cross-scenario axis-bounds manifest — viewer reads shared bounds so y-axes stay stable across scenarios
- Per-scenario axis manifest with subset filter
- Composite colour lookup for `nodeGroup_flows` plots
- Colour template plumbed into the batch render path
- Plot plan JSON cleanup (redundant timestep fields dropped)
- Group-output parameters renamed; `output_results` split; flow-group indicator stub added
- Alphabetical sorting of subplots
- Periods ordered left-to-right in vertical p-variant bar plots

## Release 3.26.0 (21.4.2026)
**Bug fixes**
- Negative-remaining infeasibility in cumulative price-ladder cap
- Bug in `realized_invest` uniqueness
- Parameter_type_list corrections across migrations and master template

**New features**
- Commodity price ladder — `price_method`, `unitsize`, `price_ladder` parameters for commodities
- `v_trade` variable (MWh × unitsize) with tier caps and objective routing
- Rolling cumulative-quota handoff for `price_ladder_cumulative` — per-period accumulators preserve remaining quota across rolls
- Two-roll cumulative-ladder validation scenario
- Rolling-aware `co2_max_total` with per-period accumulators
- Node-type consolidation aligned with the ladder work

## Release 3.25.0 (20.4.2026)
**Bug fixes**
- Align `fix_storage_*.csv` header between init and phase 3
- Carry fractional tail in `write_years_represented` (tested at R=2.5)
- `canonical_sort` sorts stably by `solve_pos` only, preserving within-solve order
- Drop `_round_to_sig` from the parquet read path
- Reindex `extract_variable` output to canonical phase-1 row order

**New features**
- Direct HiGHS → parquet extraction replaces the phase-3 glpsol reader round-trip for solver outputs and solve handoffs
- Phase-3 glpsol retirement — derived-parameter printfs moved pre-solve; readers repointed at `input/` and `solve_data/`
- Canonicalise row order by solve-creation order across all readers
- `has_balance`, `has_storage` and `node_type` consolidated into a single `node_type` enum
- Parquet `VariableSpec` column names aligned with CSV readers
- `empty_variable_frame` helper for same-shape empties

## Release 3.24.0 (19.4.2026)
**Bug fixes**
- `storage_state_start` first-timestep semantics corrected for cyclic bindings
- Goldens regenerated for the sink-flow coefficient flip

**New features**
- `coefficient` on capacity constraints split into `flow_coefficient`, `max_capacity_coefficient`, `min_capacity_coefficient`
- Sink-side `flow_coefficient` flipped from division to multiplication — both sides of the balance now use the same convention
- New `pre_built` capacity-coefficient term alongside the invested term
- Growth-cap recipe documented
- Lifetime-aware dispatch capacity bound
- `max_flow_for_unconstrained_variables` model parameter replaces the previous 1e6 literal
- CO2 duals exposed; `rp_cost_weight` threaded into constraints; horizon-vs-annual outputs distinguished
- `timeset_weights` wired through the runner (populates `rp_cost_weight.csv`)
- `no_investment` handled as a `lifetime_method`

## Release 3.23.0 (17.4.2026)
**Bug fixes**
- Pre-processing now also updates the solve `period_timeset`
- Step duration correctly applied in period aggregations of flow-derived outputs
- Reserve Excel handling fixed

**New features**
- Greedy convex-hull clustering for representative-period selection
- Representative-period storage binding via `bind_using_blended_weights` (renamed from `bind_using_rp_weights`)
- `bind_intraperiod_blocks` storage binding method for LTS-style intra-period blocks (blocks defined by gaps in timeline indices)
- Multi-year wind `no_investment` scenario test and goldens
- Base-weighted scenario test and goldens
- Delays for units and connections (constant or map)
- `5weeks_battery_intraperiod_blocks` scenario test and goldens

## Release 3.22.0 (7.4.2026)
**Bug fixes**
- Parquet loading strips the scenario column level
- `savefig` works for `Figure()` objects not registered with `pyplot`
- Threading errors from `plt.figure()` replaced with `Figure()` usage
- Dispatch colors and `nodeGroup` filtering
- Cancel pending `draw_idle` to prevent redundant re-renders
- Numerous canvas / toolbar jitter fixes (freeze/thaw around draws, layout locking)
- Legend clipping, tree navigation, canvas clearing
- Stacked-area plot axis limits
- Empty `years_represented` for rolling dispatch solves
- CO2 emissions parquet key mismatch
- Scenarios without connections

**New features**
- `PlotPlan` — pre-computed plot plans saved to disk
- Threaded figure building with prefetch cache and memory-based GB cache limit
- Time-series downsampling via `tsdownsample` with a numpy fallback
- Network graph visualisation from the Spine database
- Cross-scenario dispatch y-limits and column consistency
- Comparison-mode parquet pipeline
- Availability manifest for three-level variant display
- Redesigned three-level variant navigation (Shift+Up/Down to next row with data at focus column)
- `PlotPlan` unit tests
- YAML config restructured with entry names as top-level keys
- Per-variant `plot_name` and explicit `variant` field in configs
- Dispatch mode in Result Viewer
- `combine_scenario_parquets` and comparison checkboxes in the viewer
- Parquet file-size reduction by dropping DataFrame multi-index bloat

## Release 3.21.0 (1.4.2026)
**Bug fixes**
- Handle scenarios without connections
- Connection output for `method_2way_1var` (DC power flow)
- Guard against all-NaN values in plotting
- Reserve provision active only when there is demand for the reserve
- Mac glpsol binary naming and selection across platforms
- Fixed and cleaned-up test suite

**New features**
- DC power flow B-θ formulation in the GMPL model
- Method resolution moved from GMPL to Python
- DC power flow data pipeline and database migration
- Output processing and parquet passthrough for DC power flow
- MATPOWER import CLI wrapper
- PGLib-OPF IEEE 14-bus integration test
- 24-test DC power flow pytest suite
- glpsol built for macOS, Linux and Windows via GitHub Actions (including macos-15 arm64)

## Release 3.20.0 (29.3.2026)
**Bug fixes**
- Excel migration edge cases
- Direct execution from Excel (`.xlsx`) corrected
- GUI updated to handle different input Excel versions

**New features**
- FlexTool 2.0 importer — reads old FlexTool 2.0 Excel `.xlsm` files into a Spine database
- Import of sensitivity scenarios from FlexTool 2.0
- CLI entry point for FlexTool 2.0 sensitivity import
- Version-aware importer that handles older FlexTool 3.0 Excels as well
- `invest_method` support in the FT2 importer
- `_inflow` renamed to `_node`; `_storage` used where applicable
- `base` used instead of `Base` as the default alternative
- Stochastic sheets in Excel
- Improved Excel read/write roundtrip, including DB-to-Excel export
- Dialog to migrate Excel files
- Single source of truth for the FlexTool input DB version (`flextool/update_flextool/__init__.py`)

## Release 3.19.0 (14.3.2026)
**Bug fixes**
- Execution runs from the FlexTool root with isolated work folders
- Comparison Excel written to the correct directory
- All scenarios no longer produce identical results
- Output-actions column order, drag-select and spinbox sync
- Foreground-color crash on start
- Several threading and layout issues

**New features**
- Result Viewer window with scenario listbox, plot tree and variant panel
- `PlotCache` with memory-based GB limit
- `PlotCanvas` with PNG display wired into ResultViewer
- Dark mode with `sv_ttk` and a theme-toggle radio
- Font-relative and DPI-aware GUI sizing
- Plot-settings dialog with YAML config parsing
- Execution menu window with subprocess pool and parallel workers
- Database version checking and auto-upgrade from the GUI
- Custom file picker with last-modified column and sorting
- Visual feedback: green/grey/red highlights, boxed outputs, auto-check
- Keyboard shortcuts (including `Ctrl-A` to select all) and rearranged move arrows
- DB-editor integration with process tracking
- `--work-folder` support for parallel scenario execution
- `--parquet-base-dir` for comparison without a database
- `output_db_url` optional in `cmd_run_flextool`
- Dual variant state in the viewer: desired (dashed) + shown (solid)
- Animated hourglass spinner on output-action buttons
- Persisted checkboxes, View button, sizing adjustments
- Parent-directory button in the file-picker dialog

## Release 3.18.0 (27.2.2026)
**Bug fixes**
- Fix reading input files in `run_flextool.py`
- Many small issues surfaced by the refactor fixed so all examples execute

**New features**
- Major refactoring into deep modules — package structure reorganised for separation of concerns
- New subdirectories to keep the FlexTool root uncluttered
- Initialisation now also creates settings databases
- Updated `update_flextool.py`
- Walked back global `flextool` CLI commands — they do not work with multiple installs
- Added `output_info_template.sqlite`
- `flextool_location.txt` for runner discovery

## Release 3.17.0 (14.1.2026)
**Bug fixes**
- Float-to-string handling in the import spec
- Node-summary column alignment
- `nodeBalancePeriod` uses `pdtNodeInflow` (sum over period) instead of `annual_flow` (`annual_flow` should not carry a sign)
- Rolling-model fixes including `p_roll_continue` handling

**New features**
- Excel import specification
- Direct Excel-to-DB read pipeline
- Calamine engine for faster spreadsheet reading
- Full FlexTool execution from Python — no Toolbox required
- Removed data checks from later solves (they belong in the first solve)

## Release 3.16.0 (5.12.2025)
**Bug fixes**
- Fixed costs split off from investment costs and their calculation corrected
- `years_represented` with a discount factor >1 now calculated correctly
- Timestep durations checked to be positive and non-zero

**New features**
- Speed-up in Python output writing
- Result list refactored
- Old `write_outputs.mod` glpsol output code removed from `flextool.mod`
- Further plotting enhancements and minor refactoring
- `solve_progress.csv` with timings and scenario name
- Vertical plots, sums and means, better subplot spacing
- `write_outputs` can be used to plot a single ad-hoc time series

## Release 3.15.0 (2.12.2025)
**Bug fixes**
- `costs_discounted.csv` calculation for multi-year models
- Empty sets and rolling-model crashes
- Negative capacity units handling in outputs
- Inertia calculation in outputs
- Missing `fix_storage_time_lists`
- Several output-processing bugs (ramp, investment, storage, group results)

**New features**
- Python post-processing pipeline reaches parity with the old glpsol `write_outputs.mod` — all result CSVs replicated except node ramp envelopes
- Scenario-comparison specification (`scenario_comparison.json`) and combined-scenario results framework
- Removed redundant `fix_storage` constraint that used `dt_realized_dispatch`
- First ugly-duckling plotting on top of the Python pipeline

## Release 3.14.0 (6.8.2025)
**Bug fixes**
- Fixing problems with group results (wrong signs)
- Fix broken importer
- Fix calculation of VRE shares in the results
- Other operational costs for units had gone missing in 3.12.0 (22.5.). Now they are back.

**New features**
- There is a new migrate button in the workflow to update the active input database to the latest version.

## Release 3.13.0 (27.6.2025)
**Bug fixes**
- Added missing fixed costs to operational models
- Catching infeasible models better

**New features**
- Time structure has been simplified, see ´https://irena-flextool.github.io/flextool/reference/#how-to-define-the-temporal-properties-of-the-model'
- Changed the time structure to have timeset instead of timeblockSet
- Added periods_available for the model entity, to have periods in the data that are not used without domain errors
- Added assumptions about the model timestructure to ease the defining of it.

## Release 3.12.0 (22.5.2025)
**Bug fixes**
- Decimal points in node_balance_t were causing some grievance.
- Issues with lower case / upper case filenames when importing results
- Added missing parameters inertia_constant, ramp_cost, ramp_speed back to the model (caused by release 3.10.0)
- Fixed a output processing crash in a rolling model without VRE

**New features**
- Delays for units and one-way connections. New parameter delay, measured in time-steps (be careful), which can be a constant or a map of (integer timesteps, weight) where the weights should add to one. The map allows to spread the delay like it might happen in river systems.
- An order of magnitude speed-up in reading inputs

## Release 3.11.0 (4.3.2025)
**Bug fixes**
- Preventing crashes on curtailment calucaltions to units without capacity and MIP stochastic solving
- Correcting nested storage state passing

**New features**
- Timeseries prices for commodity and co2

## Release 3.10.0 (31.1.2025)
**Bug fixes**

- Prevent HiGHS from hanging after founding a solution
- Fix profile limits (issue when efficiency != 1)
- Made to work with new Toolbox DB API

**New features**

- New HiGHS versio (1.9.0)
- Script to see the differences in outputs (or inputs) between two model runs (meant for testing)
- FlexToolRunner.py reads input data directly from DB (speed-up)


## Release 3.9.0 (28.11.2024)
**Bug fixes**

- Fixing Excel input template

**New features**

- Updated documentation to match Toolbox changes
- New HiGHS version (1.8.1), should help with model getting stuck.
- Reorganising file structure to remove clutter from FlexTool root
- Some name changes in the workflow
- FlexTool has requirements.txt and pyproject.toml (allows pip install -e ., but not in pypi)
- Improved installation instructions (switched from miniconda to venv)
- Linux (Ubuntu tested) should work out of the box (but FlexTool execution only in work directory one scenario at a time!)


## Release 3.8.0 (29.8.2024)
**Bug fixes**

- Excel input file to use entity_alternative
- unit_curtailment_share fix to Excel input

**New features**

- Existing capacity can be set for periods
- Cumulative limits for investments in nodes, units and connections
- New node_type that has period limit for inflow
- New highs solver version
- All parameter now have valid types. These are not enforced, but Spine Toolbox highlights parameters with a wrong type.


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
