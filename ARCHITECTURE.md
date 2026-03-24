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

## Key Architectural Patterns

- **CLI → Core delegation**: All user-facing entry points (CLI scripts, GUI) delegate to core library modules.
- **State containers**: `RunnerState`, `SolveConfig`, `TimelineConfig`, `PathConfig` bundle related data for the solver pipeline.
- **Backward-compat shims**: `plot_functions.py`, `result_writer.py`, `read_flextool_outputs.py`, `scenario_comparison.py` re-export from refactored modules.
- **Dataclass-driven data flow**: `DatabaseContents`, `TimeSeriesResults`, `DispatchMappings`, `SheetSpec` define data shapes at module boundaries.
- **Multi-format I/O**: Inputs (CSV, Excel, ODS, old .xlsm), outputs (CSV, parquet, Excel, PNG/SVG), multiple solver backends.
- **GUI as subprocess orchestrator**: The Tkinter GUI spawns CLI commands as subprocesses rather than calling library functions directly.
