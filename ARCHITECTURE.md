# FlexTool Architecture

## Project Purpose

FlexTool is an energy and power systems optimization model (IRENA FlexTool). It reads input data from a Spine database, generates a linear programming (LP) model, solves it using HiGHS (or GLPK/CPLEX), and writes results to parquet files, Excel, CSV, and plots. Input databases can be populated from tabular input files (Excel, ODS, CSV) or imported from old FlexTool v2 formats. A Tkinter GUI and Spine Toolbox integration provide user interfaces.

## Repository Layout

```
flextool/                          Python package (core library)
├── __init__.py                    Re-exports: FlexToolRunner, write_outputs, migrate_database,
│                                    initialize_database, update_flextool
├── cli/                           CLI entry points (argparse + delegation)
├── flextoolrunner/                Optimization engine (DB → LP → solver)
├── process_inputs/                Read tabular/Excel/old-format data → Spine DB
├── process_outputs/               Read solver CSV → post-process → write results
├── plot_outputs/                  Generate line/bar/stacked-area plots
├── scenario_comparison/           Multi-scenario analysis and dispatch plots
├── export_to_tabular/             Export Spine DB → Excel (.xlsx)
├── update_flextool/               DB schema migration, GitHub update, DB init
├── gui/                           Tkinter GUI application
├── helpers/                       Debugging/analysis utilities
├── flextool.mod                   GLPK/LP model definition file
├── flextool_base.dat              Base data file for solver
└── import_excel_input.json        Excel import specification

Root scripts (backward compat with Spine Toolbox):
├── run_flextool.py                → flextool.cli.cmd_run_flextool:main
├── write_outputs.py               → flextool.cli.cmd_write_outputs:main
├── scenario_results.py            → flextool.cli.cmd_scenario_results:main
├── migrate_database.py            → flextool.cli.cmd_migrate_database:main
├── read_tabular_input.py          → flextool.cli.cmd_read_tabular_input:main
├── execute_flextool_workflow.py   → flextool.cli.cmd_execute_flextool_workflow:main
└── update_flextool.py             → flextool.cli.cmd_update_flextool:main

bin/                               Solver binaries
├── highs / highs.exe              HiGHS LP/MIP solver
├── highs.opt                      HiGHS solver options
├── glpsol / glpsol.exe            GLPK LP/MIP solver
├── GLPK-master.zip                GLPK source archive
└── *.dll                          Windows runtime dependencies

templates/                         Example databases and configuration
├── examples.sqlite                Pre-populated example database
├── time_settings_only.sqlite      Minimal template with time settings
├── example_input_template.xlsx    Excel input template
├── default_plots.yaml             Default plot configuration
└── default_comparison_plots.yaml  Default comparison plot configuration

version/                           Database schema templates (JSON)
├── flextool_template_master.json  Master template for new input DBs
├── flextool_template_*.json       Schema migration step templates
├── output_settings_template.json  Output settings DB template
├── output_info_template.json      Output info DB template
└── comparison_settings_template.json  Comparison settings template

plot_settings/                     Plot configuration presets
├── single_dataset/default_result_plots.json
└── multiple_datasets/default_result_plots.json

test/                              Integration tests (scenario execution)
├── conftest.py                    Pytest fixtures
├── db_utils.py                    Test database utilities
├── test_scenarios.py              Scenario execution tests
├── scenarios.yaml                 Test scenario definitions
├── expected/                      Expected output CSVs per scenario
└── fixtures/tests.json            Test fixture data

tests/                             Unit and feature tests
├── test_bar_layout.py             Plot bar layout tests
├── test_comparison_excel.py       Excel comparison tests
├── test_export_to_tabular.py      Excel export tests
├── test_gui_startup.py            GUI startup tests
├── test_plot_results.py           Plot generation tests
├── test_self_describing_reader.py Self-describing Excel reader tests
├── test_xlsx_workflow.py          Excel input workflow tests
└── run_scenarios_tests.py         Scenario runner

execution_tests/                   Spine Toolbox execution tests
├── test_execution.py              Toolbox workflow execution tests
└── tool.py                        Test tool wrapper

docs/                              MkDocs documentation site
├── index.md                       Introduction
├── tutorial.md                    Getting started tutorial
├── reference.md                   Model parameters reference
├── results.md                     Model outputs documentation
├── how_to.md                      How-to guides
├── flextool_gui_interface.md      GUI documentation
├── spine_toolbox.md               Spine Toolbox interface docs
├── spine_database.md              Spine data editor docs
├── install_toolbox.md             Installation guide
├── interface_overview.md          Interface overview
├── browser_interface.md           Web interface docs (deprecated)
├── install_web_interface.md       Web interface install (deprecated)
└── *.png, *.pdf                   Screenshots and theory slides

.spinetoolbox/                     Spine Toolbox project integration
├── project.json                   Project configuration
└── specifications/                Tool/Importer/Exporter/Transformer specs

.github/workflows/                 CI/CD
└── toolbox-flextool-bundle.yml    Toolbox bundle workflow

Other root files:
├── pyproject.toml                 Build config, dependencies, entry points
├── requirements.txt               Pip requirements
├── mkdocs.yml                     Documentation site config
├── CITATION.cff                   Citation metadata
├── LICENSE.txt                    License (Apache 2.0)
├── README.md                      Project readme
└── RELEASE.md                     Release notes
```

## Module Details

### flextool/cli/ — CLI Entry Points

Each `cmd_*.py` file defines a `main()` function using argparse.

| File | Purpose |
|------|---------|
| `cmd_run_flextool.py` | Run FlexToolRunner + write_outputs for a scenario |
| `cmd_write_outputs.py` | Read solver CSV outputs → post-process → write parquet/Excel/plots |
| `cmd_scenario_results.py` | Load multiple scenario parquets → comparison plots/Excel |
| `cmd_read_tabular_input.py` | Parse specification + Excel/CSV → write Spine DB |
| `cmd_read_self_describing_tabular_input.py` | Read self-describing Excel with embedded metadata → Spine DB |
| `cmd_read_old_flextool.py` | Import old FlexTool v2 .xlsm files → Spine DB |
| `cmd_export_to_tabular.py` | Export Spine DB → Excel (.xlsx) |
| `cmd_execute_flextool_workflow.py` | Three-phase workflow: input → solve → output (subprocess) |
| `cmd_migrate_database.py` | Upgrade DB schema to latest version |
| `cmd_update_flextool.py` | Git pull + project migration |

### flextool/flextoolrunner/ — Optimization Engine

Reads Spine DB → writes LP model files → calls solver → coordinates solve sequence.

```
FlexToolRunner (flextoolrunner.py)    Main class: .write_input(), .run_model()
├── runner_state.py                   RunnerState, PathConfig dataclasses
├── db_reader.py                      Read spinedb_api: check_version, entities_to_dict, params_to_dict
├── input_writer.py                   Write input/ CSV files from DB data
├── orchestration.py                  Main solve loop (run_model entry point)
├── recursive_solves.py               Rolling/nested/recursive solve structure builder
├── stochastic.py                     Stochastic branch handling
├── solve_config.py                   SolveConfig: all solve-level parameters from DB
├── timeline_config.py                TimelineConfig: timeline definitions and timeset mappings
├── solve_writers.py                  Write solve_data/ CSV files for each solve step
└── solver_runner.py                  Invoke glpsol/HiGHS/CPLEX solver binaries
```

### flextool/process_inputs/ — Input Data Handling

Reads tabular data and writes to Spine DB format.

| File | Public API | Purpose |
|------|-----------|---------|
| `read_tabular_with_specification.py` | `TabularReader` | Reads Excel/ODS/CSV using JSON specification |
| `read_self_describing_excel.py` | — | Reads self-describing Excel with embedded metadata |
| `read_old_flextool.py` | — | Parses old FlexTool v2 .xlsm files → OldFlexToolData |
| `write_to_input_db.py` | `write_to_flextool_input_db` | Writes parsed tabular data to Spine DB |
| `write_self_describing_to_db.py` | — | Writes self-describing Excel data to DB |
| `write_old_flextool_to_db.py` | — | Writes old FlexTool data to DB |

### flextool/process_outputs/ — Output Processing

Three-layer architecture: I/O → calculations → output formatting.

```
I/O Layer:
  read_variables.py              Reads v_*.csv → SimpleNamespace v
  read_parameters.py             Reads p_*.csv → SimpleNamespace par
  read_sets.py                   Reads s_*.csv → SimpleNamespace s
  read_flextool_outputs.py       Backward-compat shim
  read_highs_solution.py         Direct HiGHS → parquet extractor
                                 (bypass for variable/dual CSV writes —
                                 see "Solver outputs" section below)

Post-processing Calculations:
  drop_levels.py                 Strips 'solve' level from time-indexed objects
  calc_capacity_flows.py         Capacity, online status, flow_dt, ramps
  calc_connections.py            Connection flows and losses
  calc_storage_vre.py            Storage state changes and VRE potential
  calc_slacks.py                 Reserve, slack, and inertia quantities
  calc_costs.py                  All cost aggregates
  calc_group_flows.py            Group-level flow aggregations
  process_results.py             Coordinator: calls all calc_* in order

Output Functions:
  out_capacity.py                Unit/connection/node capacity tables
  out_flows.py                   Unit flow, capacity-factor, VRE, ramp outputs
  out_group.py                   NodeGroup flow, inflow, VRE-share, indicator outputs
  out_node.py                    Node summary and additional-results outputs
  out_costs.py                   Cost summary, CO2, and generic outputs
  out_ancillary.py               Connection, reserve, inertia, slack, dual outputs
  write_outputs.py               ALL_OUTPUTS list + orchestrator
  result_writer.py               Backward-compat shim
  to_spine_db.py                 Writes result DataFrames to Spine DB
```

### flextool/plot_outputs/ — Visualization

Generates time-series line plots, bar charts, and stacked-area diagrams.

```
orchestrator.py                  plot_dict_of_dataframes() entry point
config.py                        PlotConfig dataclass, DIMENSION_RULES, PLOT_FIELD_NAMES
plot_lines.py                    Line/stacked-area plots
plot_bars.py                     Bar chart orchestration
plot_bars_detail.py              Bar rendering (grouped, stacked, simple)
axis_helpers.py                  Axis formatting, smart xticks
legend_helpers.py                Legend sizing, label formatting
subplot_helpers.py               Grid layout, data slicing
format_helpers.py                Value formatters, filename generation
perf.py                          Timing utilities
plot_functions.py                Backward-compat shim
```

### flextool/scenario_comparison/ — Multi-Scenario Analysis

Loads parquet files from multiple scenario folders, combines, and generates comparison plots.

```
data_models.py                   TimeSeriesResults, DispatchMappings dataclasses
db_reader.py                     Load scenario parquet files → TimeSeriesResults
dispatch_mappings.py             Load dispatch mapping parquets → DispatchMappings
config_io.py                     Parse/write dispatch config.yaml
config_builder.py                Build/update dispatch config from data
dispatch_data.py                 Prepare per-scenario dispatch DataFrames
dispatch_plots.py                Render dispatch stacked area plots
constants.py                     Colors and special column name lists
orchestrator.py                  Top-level run() function
scenario_comparison.py           Backward-compat shim
```

### flextool/export_to_tabular/ — Excel Export

Exports a Spine DB to Excel (.xlsx) in self-describing v2 or original v1 format.

| File | Purpose |
|------|---------|
| `export_to_excel.py` | Main orchestrator; v1 vs v2 format selection |
| `db_reader.py` | `DatabaseContents` dataclass and `read_database()` |
| `sheet_config.py` | `SheetSpec`, `build_sheet_specs()` — sheet specifications |
| `excel_writer.py` | Sheet writing functions (periodic, timeseries, etc.) |
| `formatting.py` | Cell formatting and data type conversions |
| `export_settings.yaml` | Export configuration |

### flextool/gui/ — Tkinter GUI

Desktop application for project management, scenario execution, and output visualization.

```
__main__.py                      Entry point → MainWindow
main_window.py                   MainWindow (tkinter.Tk root)
project_utils.py                 Create/list/rename projects
settings_io.py                   Load/save GlobalSettings, ProjectSettings
data_models.py                   GlobalSettings, ProjectSettings, ScenarioInfo dataclasses
input_sources.py                 InputSourceManager (file selection)
scenario_lists.py                AvailableScenarioManager, ExecutedScenarioManager
execution_manager.py             ExecutionManager (threaded subprocess execution)
execution_window.py              ExecutionWindow (progress tracking)
output_actions.py                OutputActionManager (plot/export actions)
output_log_window.py             OutputLogWindow (log display)
db_editor_integration.py         DbEditorManager (Spine DB editor)
db_version_check.py              Database version validation
error_handling.py                safe_callback decorator
platform_utils.py                Platform-specific file/app opening
config_parser.py                 Configuration parsing
dialogs/
├── add_dialog.py                Add project/scenario dialogs
├── plot_dialog.py               Plot configuration dialog
├── project_dialog.py            Project settings dialog
└── file_picker.py               File selection dialog
```

### flextool/update_flextool/ — Updates and Migration

| File | Public API | Purpose |
|------|-----------|---------|
| `self_update.py` | `update_flextool` | Git pull + project migration |
| `db_migration.py` | `migrate_database` | Upgrade DB schema using version/ templates |
| `initialize_database.py` | `initialize_database` | Create new DB from JSON template |

### flextool/helpers/ — Utilities

| File | Public API | Purpose |
|------|-----------|---------|
| `compare_files.py` | `compare_files` | CSV file comparison |
| `find_coefficients.py` | `find_largest_numbers` | LP coefficient analysis |
| `mps_matrix_to_csv.py` | `parse_mps_to_matrices` | MPS matrix parsing |
| `transform_toolbox_schema.py` | `convert_schema` | Spine Toolbox schema conversion |

## Installed Entry Points

From `pyproject.toml`:

| Command | Entry point |
|---------|------------|
| `flextool-gui` | `flextool.gui.__main__:main` |
| `flextool-read-old` | `flextool.cli.cmd_read_old_flextool:main` |

## Public APIs

```python
from flextool import FlexToolRunner, write_outputs, migrate_database, initialize_database, update_flextool
from flextool.process_outputs import write_outputs, read_variables, read_parameters, read_sets, post_process_results
from flextool.scenario_comparison import get_scenario_results
from flextool.plot_outputs import plot_dict_of_dataframes
from flextool.process_inputs import TabularReader, write_to_flextool_input_db
from flextool.export_to_tabular import export_to_excel
from flextool.update_flextool import migrate_database, initialize_database, update_flextool
from flextool.helpers import compare_files, find_largest_numbers, parse_mps_to_matrices, convert_schema
```

## Data Flow

```
┌──────────────────────────────────────────────────────┐
│                  INPUT DATA SOURCES                  │
│  Excel/ODS/CSV  │  Old FlexTool .xlsm  │  Spine DB  │
└────────┬────────────────┬───────────────────┬────────┘
         │                │                   │
    TabularReader    read_old_flextool   (direct use)
    read_self_describing_excel
         │                │                   │
         └───── write_*_to_db ────────────────┘
                          │
              ┌───────────▼────────────┐
              │    Spine Input DB      │
              │  (input_data.sqlite)   │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   FlexToolRunner    export_to_excel   DB Editor
   write_input()      (→ .xlsx)       (manual)
   run_model()
          │
          │ input/ CSVs → flextool.mod → LP
          │
    ┌─────▼──────────┐
    │  HiGHS / GLPK  │
    │  / CPLEX       │
    └─────┬──────────┘
          │ output/ CSVs
          │
   ┌──────▼──────────────────────────────────┐
   │           write_outputs                  │
   │  read CSV → calc_* → out_* → write      │
   └──┬───────────────────────────────────────┘
      │
      ├── parquet files (per output type)
      ├── Excel summary
      ├── PNG/SVG plots (via plot_outputs)
      └── Spine output DB
      │
      │ (multiple scenarios)
      │
   ┌──▼──────────────────┐
   │  scenario_comparison │
   │  combine parquets    │
   │  → comparison plots  │
   │  → dispatch plots    │
   │  → comparison Excel  │
   └──────────────────────┘
```

## Solver outputs: folder layout and two parallel pathways

**Folder convention** (the direction we're migrating toward):

- **`input/`** — files that do NOT change between solves in a single
  model run.  Written once up front by the Python input writer
  (entity definitions, parameters, sets that are pure input).
- **`solve_data/`** — files that DO change between solves (rolling
  window / nested / stochastic branches).  Written freshly for each
  solve.  Includes time-dependent parameters, the six solve-to-solve
  handoff files, and — once phase 3 retires — also the solver-output
  parquet files (variables / duals per solve).
- **`output/`** — user-facing aggregated outputs (Excel, parquet
  summaries, plots) produced by `write_outputs` after the solve loop
  completes.

- **`output_raw/`** — TRANSITIONAL.  Phase 3 still writes ~140 CSVs
  here; the new HiGHS → parquet pathway also writes here for now.
  Target state: gone.  Its contents get redistributed to the above
  three folders.

`glpsol` is invoked **twice** per solve:

- **Phase 1** — `--check --wfreemps flextool.mps`.  Reads the model +
  input CSVs and writes the MPS file.  This stays.  *Anything the
  model can compute before `solve;` — including all Category B
  parameters below — can be printf-written in this phase, eliminating
  the need to re-run glpsol after the solver.*
- **Phase 3** — runs AFTER HiGHS has solved, re-reads the model + the
  `.sol`, and executes all the `printf` statements at the end of
  `flextool.mod`.  This is what the new pathway is retiring.

Two pathways currently run in parallel after the solver returns:

1. **Legacy GLPSOL phase 3** — produces ~140 CSVs.  Per-file audit
   (see the classification appendix in the git history of this file;
   counts vary slightly by scenario):

   - **Category A — pure input dumps** (~76 files, all `set_*.csv`).
     Same content as the corresponding `input/` file.  Belongs in
     `input/` (one-time write) and the Python readers should look
     there, not in `output_raw/`.
   - **Category B — model-derived pre-solve** (~57 files: `p_*`,
     `pd_*`, `pdt_*`, `ed_*`).  Computed by GMPL `param := ...`
     declarations BEFORE `solve;`.  Split by update cadence:
     * time-indexed / period-indexed parameters that change each
       solve (e.g. `pdtProcess_slope`, `pdt_commodity_price` slices)
       → `solve_data/`.
     * static derivations (e.g. `p_entity_unitsize`,
       `ed_entity_annuity`, entity-level financial constants) →
       `input/` (written once on the first solve, unchanged after).
     Either way, moving the printfs above `solve;` in `flextool.mod`
     lets phase 1 produce them — no Python re-derivation required.
   - **Category C — solver-derived outputs** (~34 files: `v_*`,
     `vq_*`, `v_dual_*`, `v_obj`).  Require a solver run.  Go in
     `solve_data/` (per-solve) as parquet once phase 3 retires.  Most
     already covered by `VARIABLE_SPECS` in `read_highs_solution.py`;
     remaining: complex duals (`v_dual_node_balance` with inflation
     transform, `v_dual_reserve_balance`, CO2 duals with `/1000`),
     synthesised duals (`v_dual_invest_unit/_connection/_node`),
     scalar `v_obj`.

   The six solve-to-solve handoff CSVs are already written to
   `solve_data/` — they belong there by the convention above.  See
   the "Solve-to-solve handoff" subsection below.

2. **HiGHS → parquet** (new) — `flextool/process_outputs/read_highs_solution.py`
   reads variable names and solution arrays directly from the live
   `highspy.Highs` instance and writes one parquet file per
   `VariableSpec` entry in `VARIABLE_SPECS` (see the module docstring).
   Covers the ~26 "simple" variable and dual outputs (v_flow, v_state,
   v_ramp, v_reserve, v_online/startup/shutdown_{linear,integer},
   v_invest, v_divest, v_angle, vq_*, and the six period-only
   investment-cap duals).  Written in wide layout via
   `flextool.lean_parquet.write_lean_parquet` so the on-disk MultiIndex
   round-trips exactly and downstream code can load each file with
   `read_lean_parquet(path)` (same shape as `read_variables.v.*`).

   **HiGHS-only by construction** — the hook lives inside
   `solver_runner._run_highs` and reads the live `Highs` instance's
   `getSolution()`.  CPLEX takes a different code path; the hook cannot
   fire for CPLEX until someone verifies that the CPLEX → glpsol-format
   `.sol` round-trip preserves the same `col_value` / `row_dual`
   semantics.

The `--use-old-raw-csv` CLI flag on `run_flextool.py` disables the
new extractor (falls back to the pure-glpsol pathway) and is the
documented fallback while parquet coverage grows.

### Solve-to-solve handoff

These six files are written by phase 3 and read again by `flextool.mod`
on the *next* solve.  All six are now also written from the live
`Highs` instance by
`flextool.process_outputs.handoff_writers.write_all_handoffs`, called
AFTER phase 3 inside `solver_runner._run_highs_or_cplex`.  The
Python writer's output matches phase 3's byte-for-byte (verified
end-to-end on `wind_battery_invest`,
`multi_fullYear_battery_nested_24h_invest_one_solve`, `fullYear_roll`,
and `network_all_tech`; see `tests/test_handoff_writers.py`).

| File | Contains | `.sol` source |
|---|---|---|
| `solve_data/p_entity_period_existing_capacity.csv` | Cumulative capacity carried forward | `v_invest[e,d]` × unitsize, accumulated with prior file |
| `solve_data/p_entity_divested.csv` | Cumulative divestments | `v_divest[e,d]` × unitsize, summed across periods, plus prior |
| `solve_data/fix_storage_quantity.csv` | Storage state at boundary | `v_state[n,d,t]` × unitsize at fix-storage steps |
| `solve_data/fix_storage_price.csv` | Shadow price at boundary | `−nodeBalance_eq.dual / inflation × period_share / scale` |
| `solve_data/fix_storage_usage.csv` | Net flow through storage node | `(outflow − inflow) × step_duration`; outflow/inflow = `v_flow × unitsize` summed across connected processes.  Simplified formula — exact for method_nvar and simple method_1var_per_way; approximate when processes attached to the storage node use min_load_efficiency or non-unity unit coefficients |
| `solve_data/p_roll_continue_state.csv` | State at last realized step | `v_state[n,d,t]` × unitsize at last realized timestep |

For now `handoff_writers` reads Category-B parameters (unitsize,
pre-existing, inflation, period-share) from `output_raw/` — i.e. from
what phase 3 just wrote.  This is intentional during the transition:
phase 3 still runs unconditionally, and the Python writer overwrites
its output with values computed directly from the `.sol`, acting as a
live validator.  Once the Category B printfs move above `solve;` in
`flextool.mod`, `handoff_writers` will switch its parameter source to
`input/` + `solve_data/` (per the update-cadence split above) and
`output_raw/` can be dropped.

### Update cadence — where each printf should land

Combining Category A/B with a trace of each file's transitive
dependencies against `solve_data/*.csv` reads:

**Write-once (target: `input/`) — 76 files, depend only on
`input/` data.**  All the entity / node / process / commodity / group
sets (~36) and the static parameter dumps already gated by `if
p_model['solveFirst']` in `flextool.mod` (~40: `p_node`, `p_unit`,
`p_connection`, `p_entity_unitsize`, `p_entity_{all_existing,pre_existing,max_units}`,
the process coefficient / capacity parameters, `p_commodity_co2_content`,
`p_reserve_upDown_group_penalty`, `group_entity_invest`,
`set_process_{VRE,source_sink}`, `set_enable_optional_outputs`, …).

**Write-per-solve (target: `solve_data/`) — ~33 files, depend on at
least one `solve_data/*.csv` (typically `steps_in_use.csv` → `dt`,
`period_first_of_solve.csv`, `p_years_represented.csv`, or
`realized_dispatch.csv`).**

| Group | Representative files |
|---|---|
| Solver variables / duals (Cat. C) (~20) | `v_flow`, `v_ramp`, `v_reserve`, `v_state`, `v_{online,startup,shutdown}_{linear,integer}`, `v_angle`, `v_dual_node_balance`, `vq_state_{up,down}`, `vq_{reserve,inertia,non_synchronous,state_up_group}`, `v_obj` |
| Solve-specific derived params (~13) | `p_step_duration`, `p_rp_cost_weight`, `p_flow_{min,max}`, `pdtProcess_{slope,section,availability,source_sink_varCost}`, `pdtNode_{self_discharge_loss,penalty_up,penalty_down}`, `pdtNodeInflow`, `entity_all_capacity` |
| Period-scoped (~10) | `v_invest`, `v_divest`, `vq_capacity_margin`, `ed_entity_{annuity,annual_discounted,annual_divest_discounted}`, `p_inflation_factor_{operations,investment}_yearly`, `{node,group}_capacity_for_scaling`, `complete_period_share_of_year` |
| Solve-state sets (~10) | `set_period`, `set_d_{realize_invest,realize_dispatch_or_invest,realized_period}`, `set_dt{,t,ttdt,_realize_dispatch,_fix_storage_timesteps}`, `set_period_{in_use,first_of_solve,__time_first}`, `timeline_breaks` |
| Per-entity invest sets (3) | `set_ed_invest`, `set_ed_divest`, `set_edd_invest` |

Plus the **six solve-to-solve handoff files** already in
`solve_data/` (handoff subsection above).

### Retiring phase 3

The three steps below, in order, remove the need for the second
`glpsol` invocation on HiGHS runs:

1. **Move the 76 write-once printfs above `solve;`** in
   `flextool.mod`, redirected at `input/`.  No Python work — phase 1
   already runs the param/set derivations, so it just needs to emit
   them before hitting `solve;`.  Guard with `if p_model['solveFirst']`
   so only the first solve writes them.
2. **Move the ~13 write-per-solve derived-parameter printfs above
   `solve;`** (the derived-parameter subset — everything in the
   write-per-solve list EXCEPT the ~20 solver-output `v_*`/`vq_*`/
   `v_dual_*` group and the per-entity invest sets), redirected at
   `solve_data/`.  Phase 1 runs every solve, so this correctly
   re-emits them with the current solve's active periods.
3. **Finish Category C coverage in `read_highs_solution.py`** — DONE.
   Coverage for all solver-dependent outputs the downstream pipeline
   reads, via ``VARIABLE_SPECS`` and a set of custom writers:

   | Output | Pathway |
   |---|---|
   | `v_flow`, `v_ramp`, `v_reserve`, `v_state`, `v_online/startup/shutdown_{linear,integer}`, `v_angle`, `v_invest`, `v_divest`, `vq_*` | `VARIABLE_SPECS` — wide parquet per solve |
   | `v_dual_maxInvest_{period,total}`, `v_dual_maxCumulative`, `v_dual_maxInvestGroup_{period,total,cumulative}` | `VARIABLE_SPECS` — `source="row_dual"`, `value_scale=1e6` |
   | `v_dual_co2_max_{period,total}` | `VARIABLE_SPECS` — the `total` variant is the first use of `has_period=False` |
   | `v_obj` | `write_v_obj` — `h.getObjectiveValue() * 1e6`, scalar-per-solve |
   | `v_dual_invest_{unit,connection,node}` | `write_v_dual_invest_by_class` — `source="col_dual"` on `v_invest`, split by entity class loaded from `input/{process_unit,process_connection,node}.csv` |
   | `v_dual_node_balance` | `write_v_dual_node_balance` — `source="row_dual"` on `nodeBalance_eq` with `leading_ignore=1` (skip `c`) + `trailing_ignore=4` (skip `tp, tpwt, dp, tpws`), then `× −1e6 / inflation[period]` |
   | `v_dual_reserve_balance` | `write_v_dual_reserve_balance` — `reserveBalance_timeseries_eq` row duals aggregated across `method` level then `× period_share / inflation`.  **Simplified** — treats the model formula as if only the `timeseries_only` method were active.  Groups using `dynamic` or `n_1` reserve methods will be under-reported until the full `max()` of three constraints is ported. |
   | `nodeBalanceBlock_eq` aggregation for block-storage nodes (used in representative-period scenarios) | **NOT YET** in `write_v_dual_node_balance` — `nodeStateBlock` nodes see zeroes until ported |

   All solver-output files validated end-to-end against phase 3 on
   `multi_year_wind_growth_cap`, `test_a_lot`, `wind_battery_invest`,
   and `network_all_tech` — match within `%.8g` CSV format precision.

After steps 1-3, repoint `read_parameters.py` / `read_sets.py` at
`input/` + `solve_data/` instead of `output_raw/`, and phase 3 can be
removed from `solver_runner` for HiGHS.  `output_raw/` becomes empty
and the directory can be deleted.

### Extension pattern

To add a new variable or dual to the parquet pipeline, append a
`VariableSpec(name, col_names, has_time, is_dual, value_scale,
output_name)` to `VARIABLE_SPECS` in `read_highs_solution.py`.  No code
changes elsewhere.

## Numerical scaling

FlexTool's LP/MIP can accumulate coefficients spanning many decades when
users combine entities of different physical scale (e.g., a 10 kW building
heat pump alongside a 10 GW continental grid).  Left unchecked this
causes HiGHS to miss symmetry, presolve aggressively on the wrong rows,
slow down, or raise false-infeasibility warnings.  The scaling pipeline
runs on every solve and is layered so each mechanism targets one source
of spread.

### Layers (bottom up)

1. **Unitsize column normalisation (user convention).**  Every variable
   `v_flow`, `v_state`, `v_reserve`, `v_invest`, etc. is written in
   units of `p_entity_unitsize` per entity, so the raw values stay
   near O(1).  Predates the scaling project — this is the single
   biggest contributor to well-conditioned inputs.
2. **Two-tier slack convention (`flextool/SLACK_CONVENTION.md`).**
   Every `vq_*` is a primary slack `≤ K_rel` (bounded, scaler-relative,
   `K_rel = 1` default) plus an unbounded escape slack penalised at
   `× 1000`.  Keeps the slack column coefficients bounded against the
   row scaler; escape activity becomes a user diagnostic rather than
   false infeasibility.  Agents 2-4 implemented this.
3. **Row scaling (opt-in, `solve.use_row_scaling`).**  Node-balance
   and group-balance constraint rows get multiplied by
   `node_capacity_for_scaling` / `group_capacity_for_scaling` derived
   from the unitsizes of connected entities, rounded to powers of 10
   to preserve HiGHS symmetry detection.  Agent 5 added the flag and
   the formulas; the hardcoded `node_capacity_for_scaling := 1`
   stays active until the user opts in.
4. **Precision cleanup (always on, `--precision-digits`).**  Every
   numeric CSV cell is rounded to 10 significant figures before
   write.  Removes benign precision artifacts that would otherwise
   trip the near-duplicate detector and waste HiGHS scaling passes.
5. **ScaleAnalyzer (`flextool/flextoolrunner/scaling.py`).**  After
   the input CSVs are written but before the solver runs, a pure-
   stdlib analyzer walks the cost / flow / unitsize / penalty CSV
   families, computes per-family log10 spread stats, and recommends:
   * `use_row_scaling = "yes"` when unitsize spread > 3 decades;
   * `scale_the_objective = 10 ** -round(log10(rough_obj_estimate))`,
     clamped to `[1e-12, 1e0]`.
   Output: `solve_data/scaling_analysis.json`.
6. **`--auto-scale` application.**  When the `--auto-scale` CLI
   flag is set (or `FLEXTOOL_AUTO_SCALE=1`), the analyzer's row-
   scaling recommendation is applied if and only if the user has not
   explicitly set `solve.use_row_scaling`.  The objective scalar
   recommendation is not auto-applied (user-controlled only).
7. **Output un-scaling (`flextool/process_outputs/read_highs_solution.py`).**
   Every `VariableSpec` carries a `unscale_by` field; slack parquets,
   node-balance duals, and reserve-balance duals are multiplied back
   into the user's absolute units before parquet / CSV write.
   Downstream consumers see no change regardless of whether row
   scaling was active.
8. **Diagnostic report (`flextool/flextoolrunner/scaling_report.py`).**
   After the solve, `scaling_report.txt` is written next to the
   solve's `scaling_analysis.json`.  Nine sections (header, decisions,
   family ranges, bimodal detection, composite-scale mismatch,
   near-duplicate clusters, escape-tier slack activity, HiGHS matrix
   ranges, summary verdict).  A 3-10 line echo goes to stdout so the
   verdict is visible without opening the file.

### What the user sees

- On every solve: `scaling_analysis.json` + `scaling_report.txt` in
  `<work_folder>/solve_data/`.  A short stdout echo summarising the
  verdict (`well-scaled`, `acceptably`, or `poorly scaled`).
- On pathological inputs (composite-scale mismatch, bimodal cost
  family, escape-slack activity): the stdout echo expands to include
  the load-bearing diagnostic and its recommendation.
- Per-solve caching is keyed on the solve name — rolling-window
  solves reuse the cached ScaleTable with no CSV re-read.

### Where to look

- **User-facing guide**: `flextool/SCALING_USER_GUIDE.md`.
- **Slack implementation reference**: `flextool/SLACK_CONVENTION.md`.
- **Benchmark harness + validation**: `scaling_benchmark/README.md`
  and `scaling_benchmark/VALIDATION_REPORT.md`.
- **Design memo**:
  `~/.claude/projects/-home-jkiviluo-sources-flextool/memory/project_lp_scaling_2026-04.md`.

## Key Architectural Patterns

- **CLI → Core delegation**: All user-facing entry points (CLI scripts, GUI) delegate to core library modules.
- **State containers**: `RunnerState`, `SolveConfig`, `TimelineConfig`, `PathConfig` bundle related data for the solver pipeline.
- **Backward-compat shims**: `plot_functions.py`, `result_writer.py`, `read_flextool_outputs.py`, `scenario_comparison.py` re-export from refactored modules.
- **Dataclass-driven data flow**: `DatabaseContents`, `TimeSeriesResults`, `DispatchMappings`, `SheetSpec` define data shapes at module boundaries.
- **Multi-format I/O**: Inputs (CSV, Excel, ODS, old .xlsm), outputs (CSV, parquet, Excel, PNG/SVG), multiple solver backends.
- **GUI as subprocess orchestrator**: The Tkinter GUI spawns CLI commands as subprocesses rather than calling library functions directly.
