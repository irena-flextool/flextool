## Release 3.34.0 (12.5.2026) — Multi-solver support + documentation overhaul

This release ships the two user-facing additions on top of 3.33.0's engine swap: per-solve commercial-solver selection and a documentation refresh organised around the Spine Toolbox GUI as the primary FlexTool interface.

**Multi-solver support**
- New per-solve parameters (v52 migration) on the `solve` entity: `solver`, `solver_io_api`, `solver_options`, `solver_time_limit`, `solver_mip_gap`, `solver_threads`, `solver_log_level`. HiGHS remains the default — existing scenarios with no solver parameters are unaffected.
- Backends supported via [polar-high](https://pypi.org/project/polar-high/): **HiGHS** (default, ships with FlexTool), **Gurobi**, **CPLEX**, **Xpress**, **COPT**. Each commercial solver requires its own Python wrapper + license; FlexTool itself never imports the commercial wrappers and never inspects licenses (vendor's discovery path handles it).
- The three convenience knobs (`solver_time_limit`, `solver_mip_gap`, `solver_threads`) are normalised across solvers; FlexTool translates them to each backend's native parameter name. `solver_options` passes raw key/value pairs through unchanged.
- Cascade startup probes each available solver with a trivial 1-var LP and logs a per-solver license status line, e.g. `Solver license status: gurobi=licensed, cplex=licensed, xpress=licensed, copt=not-installed, highs=licensed`.
- Cold rebuilds in the cascade dispatch through `polar_high.solvers.solve(...)` and normalise the result via a new `LiteSolution` adapter so downstream output writers stay unchanged. Warm-start and Lagrangian decomposition remain HiGHS-only by polar-high design — selecting another solver on a warm cascade logs a warning and falls back to cold rebuilds; selecting another solver on a Lagrangian-decomposed scenario raises a clear configuration error.
- Per-solver documentation pages under `docs/solvers/` (install, licensing, common errors, how to set the solver in FlexTool).

**Documentation overhaul**
- Site reorganised around the Spine Toolbox GUI as the primary FlexTool interface; CLI flows are documented but no longer the default story.
- New developer guide and how-to recipes for the engine_polars-era codebase.
- `decomposition_method` parameter description tightened; superseded GUI screenshot removed.

**Bug fixes**
- `_writer_mid_sets.derive_commodity_node_co2` no longer crashes with `ComputeError: cannot compare string with numeric type (f64)` on customer DBs where `p_commodity.csv`'s value column is inferred as Float64 by polars. `_read_csv` now forces every column to Utf8 on read via `infer_schema_length=0`.
- `process_outputs.calc_storage_vre` no longer raises `KeyError` when `node_self_discharge_loss` is authored on nodes that have no `v_state` LP variable (supply-curve / commodity nodes). The self-discharge multiply is now restricted to the intersection of authored nodes and storage nodes.
- CLI cascade exception handler distinguishes `FlexToolUserError` (configuration problem — clean message, exit 1) from other exceptions (real flextool bug — full traceback). Users hitting unknown-solver / missing-license errors no longer see a stack trace.
- `update_flextool` now refreshes declared dependencies for editable installs too — `pip install -e .` with new core deps in `pyproject.toml` would previously be missed, causing `ModuleNotFoundError` on the next solver invocation. The `--upgrade` is now passed without `--upgrade-strategy=eager` so transitive dependencies aren't churned.

**Performance**
- Spine DB read: `SolveConfig.load_from_db_url` and `TimelineConfig.load_from_db_url` now pre-warm with `db.fetch_all("entity")` + `db.fetch_all("parameter_value")` before any `find_*` call, mirroring the legacy `FlexToolRunner.__init__` pattern. Measured 5.5–6.4× speedup on large customer DBs (~1.8 s saved per cascade run); sub-MB test DBs see no change.

**Other**
- v52 schema migration renames the legacy `solver` value list to `solvers` in place: members expanded from `[glpsol, highs, cplex]` to `[highs, gurobi, cplex, xpress, copt]`; pre-existing `solve.solver = "glpsol"` values are rewritten to `"highs"` (GLPK retired in Δ.22). Parameter_definition foreign key is preserved through the rename so no data is lost.

---

## Release 3.33.0 (12.5.2026) — `glpsol` retired; HiGHS via `polar-high` becomes the sole LP backend

This release closes out the largest architectural change since FlexTool went open source: the GLPK/GMPL pipeline that built the LP via `glpsol`, wrote MPS files to disk, and re-loaded them into HiGHS is gone.  The new path builds the LP in process via [`polar-high`](https://pypi.org/project/polar-high/) on top of the [`polars`](https://pola.rs/) DataFrame engine, hands it to HiGHS through `highspy`, and stays in memory for the entire solve → handoff → output-writer chain.

There are no binary build artefacts left to ship: `glpsol` is gone, the `bin/glpsol*` binaries that the previous releases bundled have been deleted, and the `flextool.mod` GMPL model file is deleted.  FlexTool is now a pure-Python install — `pip install flextool` is sufficient.

The version is held at 3.x rather than bumped to 4.0.0 until the new engine reaches full parity with the old one across the test suite.  A 4.0.0 semantic bump will follow once that bar is met; this release ships the backend swap behind the existing 3.x API surface.

**Breaking changes**
- `bin/glpsol*` removed; `flextool/flextool.mod` and `flextool/flextool_base.dat` removed
- `--engine=gmpl` hard-rejected with a clear retirement banner
- Legacy GMPL-pipeline CLI flags removed: `--use-old-raw-csv`, `--ipm`, `--auto-scale`, `--relax-feasibility`, `--glpsol-timing`, `--report-near-duplicates`
- The stubbed `flextool/flextoolrunner/lagrangian.py` (legacy GMPL coordinator) is deleted; `--decomposition lagrangian` now drives the native polars coordinator (see below)
- `--highs-threads` accepted as a no-op stub for GUI/Toolbox compat; the native cascade is single-threaded

**New solver backend (`engine_polars`)**
- Whole-system `FlexData` is built directly from a Spine DB or pre-staged workdir CSVs; the LP build, solve, and output handoff all happen in memory through `engine_polars.run_chain_from_db`
- HiGHS is consulted via `polar-high.Problem` / `WarmProblem` / `LagrangianProblem`; no MPS roundtrip
- New experimental `--fast-single-solve` CLI path for simple single-solve workloads — bypasses `write_input` entirely and reads inputs from Spine via `SpineDbReader`
- Native `engine_polars/_writer_*` ports of 150+ preprocessing sets and calculated parameters (Writer Phase 1-4: `L0-L9`, follow-ups 1-8, closeout, Phase 2 sub-dispatches 1-8, Phase 3 cascade adoption, Phase 4 Gap F handoff)
- Automatic LP scaling — analyser, scaling-report, two-sided cost-band guard, geometric-centering fallback, objective and bound scaling
- Δ.31: in-memory `FlexData` + `solution` threaded into `process_outputs.write_outputs` so the output writers no longer round-trip through `solve_data/*.csv`

**Spatial Lagrangian decomposition — native rewire**
- `--decomposition lagrangian` CLI rewired onto `engine_polars._lagrangian.solve_lagrangian`: per-region builds via `LagrangianProblem`, damped subgradient outer loop, primal averaging on the cross-region pipeline flows
- Smoke-test coverage on the LH2 three-region fixture pins the CLI contract within a 2 % gap-to-monolithic tolerance

**Flex-temporal decomposition** (carried over from `new-outputs` 3.29.0, fully integrated)
- Per-entity temporal blocks make mixed-resolution dispatch possible (hourly power + daily hydrogen in the same solve)
- `v50` + `v51` migrations land `solve.new_stepduration` and group-level `new_stepduration` + `decomposition_method`
- Constraints made block-aware: storage, conversion, flow capacity, DC flow, UC, ramp, profile, reserve; node balance generalised via overlap-set aggregation
- Output writers expand coarse-block variables back onto the fine timeline for the user

**Test infrastructure**
- Layer-1/2/3 test pyramid scaffolded, with golden objectives pinned on 10 scenarios and per-scenario timing budgets
- `@pytest.mark.solver` and `@pytest.mark.smoke` markers
- GitHub Actions CI on the smoke tier
- LH2 three-region JSON fixture + per-fixture native parity tests
- MPS-parity harness retired alongside the GMPL pipeline

**Packaging / installation**
- All binary dependencies retired — FlexTool is now a pure-Python install
- `polar-high` is now a core dependency (was an optional `engine-polars` extra)
- `pip install flextool` is sufficient on Linux, macOS, and Windows (HiGHS arrives via the `highspy` wheel)
- PyPI release in preparation (see `specs/pypi_release_checklist.md` for the remaining steps)

---

## Release 3.32.0 (5.5.2026)
**Bug fixes**
- Excel template link-sheet retention: drop link sheets where the link's own class has no surviving data (e.g. `connection_node` when `constraint` is unselected)
- `group.{include_stochastics, new_stepduration}` retagged from `model`/`timeline` to `solve_advanced`
- `timeset.{timeset_weights, representative_period_weights}` retagged to `solve_advanced`
- Quadratic linear scan in `blocks::derive_blocks::_assign_entity_to_group` replaced with a precomputed `(entity → group)` lookup

**New features**
- `--groups` CLI flag on `cmd_export_to_tabular` to restrict the Excel template to a parameter-group set; sheets with no surviving columns are dropped
- GUI parameter-group picker on "Add empty FlexTool input Excel" — DB-driven checkbox tree, required groups (`timeline`, `model`, `solve_basics`, `basics`) highlighted with hover tooltip; theme-aware colours
- Default-value row in v2 Excel sheets where any parameter on the sheet has a default (Map/Array defaults stringified)
- Representative-period clustering input drops scalar-valued profiles/inflows

**Note** — `new-outputs` also landed four hot-path optimisations on the legacy `flextool/flextoolrunner/preprocessing/*` writers (`is_static` fast-path on `PdtLookup`, sparse-emit + `default 0` on the six dense `pdt*` CSVs, `pandas.read_csv` swap in `_read_pdt_at_param`, sparse iteration in `pssdt_varCost_*`, the O(N²×T²) → O(N + |pairs|) fix on `node_inflow_scaling_params.has_node_time_inflow`, and side-node indexing for `process_arc_unions::write_node_group_dispatch_sets`).  Those edits target a code path that the `engine_polars` writer port (Writer Phase 1-4) is retiring, so they were not carried across in the merge — the performance lessons are preserved as actionable items for the native polars writers in `specs/sparse_writer_lessons_for_engine_polars.md`.

## Release 3.31.0 (4.5.2026)
**Bug fixes**
- Stochastic per-branch profile values previously lost in `pdtProfile` are restored
- Plot dataframe fragmentation in dispatch column-alignment fixed
- Dispatch view reads group flags from `nodeGroupDispatch` (not `nodeGroupIndicators`)
- `solve.timeline_hole_multiplier` now declared in the v44 schema (default 1.0)
- `db_migration` v42: robust legacy value-list cleanup via `find_parameter_value_lists`
- Migration steps tolerate `NothingToCommit` so a second run doesn't abort the chain
- RP clustering input: scalar-valued profiles/inflows dropped

**New features**
- `mod → Python` preprocessing migration: ~150+ sets and calculated parameters previously declared in `flextool.mod` now live under `flextool/flextoolrunner/preprocessing/`, written per-solve to `solve_data/`
- Migration scaffolding: phase-0 inventory + MPS parity harness + lint rules + DAG of derivations
- 70 incremental migration batches with MPS parity verified at 7-sig-fig precision across multiple baselines (rolling, multi-solve, contains_solve, h2_trade, test_a_lot)
- Per-class param taxonomy in `_param_taxonomy.py` (PROCESS_TIME_PARAM, NODE_PERIOD_PARAM, etc.)
- Order-determinism lint rule (`tests/test_preprocessing_ordered_set_lint.py`) — bare `set()` / set-literals / set-comprehensions blocked in `preprocessing/`
- `templates/examples.sqlite` rebuilt from `tests.json` and migrated to v51

## Release 3.30.0 (28.4.2026)
**Bug fixes**
- `read_highs_solution`: strip GLPSOL quotes from ISO 8601 timestamps in parsed variable names
- `handoff_writers`: load resolved-set CSVs from `solve_data/` (Phase A follow-up)
- `plot_canvas`: cap rendered DPI to keep figure pixmaps under X11's limit
- `RP`: representative-period indices sorted into chronological order
- `plot_bars`: bar thickness constant per row regardless of bar count
- GUI memory-watchdog: kill only when BOTH free-RAM reserve AND swap allowance are breached
- GUI: refuse to open the same input source twice (flash row red)

**New features**
- `result_viewer` Phases B–E: single rebuild path with filter-only tree toggles, per-scenario availability union, lazy plan-parquet union, comparison-only plans
- GUI per-job memory budget + watchdog + admission control + status bar
- GUI: drag-to-reorder + Alt+Up/Down on result-viewer scenarios tree
- GUI: shared `CheckTreeController` for every check tree + generation tokens for comparison-data updates
- GUI: View button always shown; opens single mode focused on the scenario
- GUI: persist per-variant duration with template default + clamp-safe save
- GUI: robust delete + prune dangling scenario state from `settings.yaml`
- `.mod`-resolved set files moved from `input/` to `solve_data/`; `set_` prefix dropped
- `--glpsol-timing` diagnostic flag for per-constraint matrix-gen timing
- `nodeBalance_eq` and `conversion_indirect` use tuple binding instead of overlap equality-filters

## Release 3.29.0 (24.4.2026)
**Bug fixes**
- v50 item-lookup bug fixed while restoring `unidirectional` in the master template
- `export_to_tabular` whitelists synced with master template schema
- Lagrangian: defensive name-map consistency check per iteration

**New features**
- **Flex-temporal decomposition**: per-entity temporal blocks for mixed-resolution dispatch (e.g. hourly power + daily hydrogen in the same solve)
- v50 migration: `new_stepduration` moved from `timeset` to `solve`
- v51 migration: group-level `new_stepduration` + `decomposition_method`
- `blocks.py`: derive per-entity blocks + overlap set + block predecessors + boundaries
- Generalised node balance via overlap-set (M-matrix) aggregation; constraints made block-aware (storage, conversion, flow capacity, DC flow, UC, ramp, profile, reserve)
- Output writers expand coarse-block variables back to the fine timeline for the user
- **Spatial Lagrangian decomposition**: regional input filter (cross-region flows as half-flows), `lagrangian.py` scaffolding + subgradient loop with primal recovery, CLI flag + docs
- LH2 three-region fixture + golden integration test (Lagrangian vs monolithic)
- `HighsModelHandle` persistence helper for repeated subgradient solves

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
- New parameters for solve objects: `solve_mode`, `rolling_duration`, `rolling_solve_horizon`, `rolling_solve_jump` and `rolling_start_time` that enable to build rolling window models. See [How to use a rolling window for a dispatch model](how_to.md#how-to-use-a-rolling-window-for-a-dispatch-model).
- New parameter `contains_solves` that enables nesting solves inside solves (e.g. to calculate shadow values for long term storages or to implement rolling dispatch inside a multi-year investment model). See [How to use Nested Rolling window solves (investments and long term storage)](how_to.md#how-to-use-nested-rolling-window-solves-investments-and-long-term-storage).
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
