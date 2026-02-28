# FlexTool Architecture

## Project Purpose

FlexTool is an energy and power systems optimization model (IRENA FlexTool). It reads input data from a Spine database, generates a linear programming (LP) model, solves it using HiGHS, and writes results to parquet files, Excel, CSV, and plots.

## Module Map

```
flextool/
├── __init__.py                        Top-level re-exports
├── cli/                               CLI entry points (argparse + delegation)
│   ├── run_flextool.py                Run optimization for a scenario
│   ├── write_outputs.py               Process and write solver outputs
│   ├── scenario_results.py            Cross-scenario comparison analysis
│   ├── read_tabular_input.py          Import Excel/CSV data to Spine DB
│   ├── execute_flextool_workflow.py   Full workflow orchestration (subprocess)
│   ├── update_flextool.py             Update FlexTool from GitHub
│   └── migrate_database.py            Migrate database to latest schema
├── flextoolrunner/
│   └── flextoolrunner.py              Core model: reads DB → writes LP → calls HiGHS/GLPK
├── process_inputs/
│   └── read_tabular_with_specification.py  Reads Excel/CSV to Spine DB format
├── process_outputs/
│   ├── __init__.py                    Re-exports public API
│   │   ── I/O layer ──
│   ├── read_variables.py              Reads solver variable CSV files → SimpleNamespace v
│   ├── read_parameters.py             Reads solver parameter CSV files → SimpleNamespace par
│   ├── read_sets.py                   Reads solver set CSV files → SimpleNamespace s
│   ├── read_flextool_outputs.py       Backward-compat shim (re-exports read_* functions)
│   ├── to_spine_db.py                 Writes result DataFrames back to Spine DB
│   │   ── Post-processing calculations ──
│   ├── drop_levels.py                 Strips 'solve' level from all time-indexed objects
│   ├── calc_capacity_flows.py         Computes capacity, online status, flow_dt, ramps
│   ├── calc_connections.py            Computes connection flows and losses
│   ├── calc_storage_vre.py            Computes storage state changes and VRE potential
│   ├── calc_slacks.py                 Computes reserve, slack, and inertia quantities
│   ├── calc_costs.py                  Computes all cost aggregates
│   ├── calc_group_flows.py            Computes group-level flow aggregations
│   ├── process_results.py             Thin coordinator: calls all calc_* in order
│   │   ── Output functions ──
│   ├── out_capacity.py                unit/connection/node capacity tables
│   ├── out_flows.py                   unit flow, capacity-factor, VRE, ramp outputs
│   ├── out_group.py                   nodeGroup flow, inflow, VRE-share, indicator outputs
│   ├── out_node.py                    node summary and additional-results outputs
│   ├── out_costs.py                   cost summary, CO2, and generic outputs
│   ├── out_ancillary.py               connection, reserve, inertia, slack, dual outputs
│   ├── write_outputs.py               ALL_OUTPUTS list + orchestrator + __main__
│   └── result_writer.py               Backward-compat shim (re-exports write_outputs)
├── plot_outputs/
│   ├── plot_functions.py              Core matplotlib plotting functions
│   └── plot_results.py                Result-specific plot generation
├── update_flextool/
│   ├── update_flextool.py             Git pull + project migration
│   ├── migrate_database.py            Schema migration to latest version
│   └── initialize_database.py         Create new blank FlexTool database
├── helpers/
│   ├── compare_files.py               File comparison utilities
│   ├── find_coefficients.py           LP coefficient analysis
│   ├── mps_matrix_to_csv.py           MPS matrix parsing
│   └── transform_toolbox_schema.py    Spine Toolbox schema conversion
└── create_scenarios/
    └── scenario_results.py            Load, combine, and plot multi-scenario results
```

Root directory contains thin wrapper scripts for backward compatibility with Spine Toolbox:
- `run_flextool.py` → delegates to `flextool.cli.run_flextool:main`
- `write_outputs.py` → delegates to `flextool.cli.write_outputs:main`
- `scenario_results.py` → delegates to `flextool.cli.scenario_results:main`
- `migrate_database.py` → delegates to `flextool.cli.migrate_database:main`
- `read_tabular_input.py` → delegates to `flextool.cli.read_tabular_input:main`

## CLI Commands (after `pip install -e .`)

| Command | Entry point |
|---------|------------|
| `flextool` | `flextool.cli.run_flextool:main` |
| `flextool-write-outputs` | `flextool.cli.write_outputs:main` |
| `flextool-scenario-results` | `flextool.cli.scenario_results:main` |
| `flextool-read-tabular` | `flextool.cli.read_tabular_input:main` |
| `flextool-update` | `flextool.cli.update_flextool:main` |
| `flextool-migrate` | `flextool.cli.migrate_database:main` |
| `flextool-workflow` | `flextool.cli.execute_flextool_workflow:main` |

## Public APIs

```python
from flextool import FlexToolRunner, write_outputs, migrate_database
from flextool.process_outputs import write_outputs, read_variables, read_parameters, read_sets
from flextool.create_scenarios import get_scenario_results
from flextool.plot_outputs import plot_dict_of_dataframes
from flextool.process_inputs import TabularReader
from flextool.update_flextool import migrate_database, initialize_database, update_flextool
```

## Data Flow

```
Excel/CSV → read_tabular_input → Spine input DB
                                      ↓
                              FlexToolRunner
                              (reads DB, writes LP file)
                                      ↓
                              HiGHS/GLPK solver
                              (writes CSV output files)
                                      ↓
                              write_outputs
                              (reads CSV → parquet/Excel/plots)
                                      ↓
                              scenario_results
                              (combines multiple scenarios → comparison plots)
```
