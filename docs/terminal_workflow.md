# Running IRENA FlexTool from the terminal

IRENA FlexTool can be run directly from the command line without Spine Toolbox or a web browser. This is useful for scripting, automation, running on servers, or when you prefer a terminal-based workflow.

## Prerequisites

- **Python 3.11+** with a virtual environment (e.g. `~/venv-spi/`)
- **HiGHS** solver, called from Python via the `highspy` bindings (installed automatically with the FlexTool package)
- **FlexTool** installed: `pip install -e .` (add `[toolbox]` for Spine Toolbox integration)
- See the [install page](https://irena-flextool.github.io/flextool/install_toolbox/) for full setup instructions

Activate the virtual environment before running any commands:

```bash
source ~/venv-spi/bin/activate
```

## Quick start

Run the example scenario `base` from the bundled example database:

```bash
python execute_flextool_workflow.py \
    templates/examples.sqlite \
    output_info.sqlite \
    base
```

The database already exists, so the input-preparation phase is skipped automatically (it runs only when `--tabular-file-path` or `--csv-directory-path` is given). An input database older than this FlexTool is migrated first (a frozen one stops the workflow; see [Migrate database schema](#migrate-database-schema)). Phase 2 then runs the model and writes outputs in one pass. Results land in `output_plots/base/` and `output_parquet/base/`.

## Scripts overview

Root-level scripts are backward-compatible wrappers that delegate to modules under `flextool/cli/`. To run the
model, use the `flextool-run` console script:

```bash
# Console script (preferred for model runs)
flextool-run ...

# Root-level script (backward compat)
python run_flextool.py ...

# Module invocation
python -m flextool.cli.cmd_run_flextool ...
```

`flextool-run` runs the engine (`flextool.cli.cmd_run_flextool`) as a child process and writes everything the
engine prints into the run's log, `output_parquet/<scenario>/_run.log`, including the solver's own lines and a
crash dump. It takes the same arguments and exits with the engine's exit code. The other two forms run the engine
in the same process; their `_run.log` holds only FlexTool's own console output.

### Workflow and execution

| Script | CLI module | Purpose |
|---|---|---|
| `execute_flextool_workflow.py` | `flextool.cli.cmd_execute_flextool_workflow` | Unified entry point &mdash; runs the full workflow |
| `run_flextool.py` | `flextool.cli.cmd_run_flextool` (run it with `flextool-run`) | Runs the optimization model and writes outputs |
| `write_outputs.py` | `flextool.cli.cmd_write_outputs` | (Re-)generates plots, parquet, CSV, or Excel from results |
| `scenario_results.py` | `flextool.cli.cmd_scenario_results` | Compares results across multiple scenarios |

### Input preparation

| CLI module | Purpose |
|---|---|
| `flextool.cli.cmd_read_tabular_input` | Parse Excel/ODS/CSV using JSON specification &rarr; write Spine DB |
| `flextool.cli.cmd_read_self_describing_tabular_input` | Read self-describing Excel with embedded metadata &rarr; Spine DB |
| `flextool.cli.cmd_read_old_flextool` | Import old FlexTool v2 .xlsm files &rarr; Spine DB |

### Export and maintenance

| Script | CLI module | Purpose |
|---|---|---|
| `migrate_database.py` | `flextool.cli.cmd_migrate_database` | Upgrade DB schema to latest version |
| &mdash; | `flextool.cli.cmd_export_to_tabular` | Export Spine DB &rarr; Excel (.xlsx) |
| `update_flextool.py` | `flextool.cli.cmd_update_flextool` | Git pull + project migration |
| &mdash; | `flextool.cli.cmd_time_runs` (`flextool-time-runs`) | Time scenario runs phase by phase ([Check that exercises run fast enough](#check-that-exercises-run-fast-enough)) |

### Installed entry points

After `pip install -e .` or `pip install .`, these commands are available directly:

| Command | Entry point |
|---|---|
| `flextool-gui` | `flextool.gui.__main__:main` |
| `flextool-read-old` | `flextool.cli.cmd_read_old_flextool:main` |
| `flextool-migrate-database` | `flextool.cli.cmd_migrate_database:main` |
| `flextool-freeze` | `flextool.cli.cmd_freeze_database:main` |
| `flextool-project` | `flextool.cli.cmd_project_package:main` |
| `flextool-time-runs` | `flextool.cli.cmd_time_runs:main` |

---

## `execute_flextool_workflow.py` &mdash; full workflow

This is the recommended entry point. It orchestrates two phases:

1. **Input preparation** (optional) &mdash; convert tabular data (Excel/ODS/CSV) into a Spine database. Runs only when `--tabular-file-path` or `--csv-directory-path` is given; otherwise the existing input database is used as-is.
2. **Model execution + output write** &mdash; run the FlexTool optimization model and write results in the requested formats. Parquet is always produced; `--write-methods` selects which additional formats (plot, csv, excel, spinedb) to generate alongside.

The two phases are fused because the in-memory `FlexData` + HiGHS solution are gone once the solver exits &mdash; there is no separate standalone "write outputs" step. (Use `python run_flextool.py` followed by `python write_outputs.py --read-parquet-dir ...` if you need to re-render artefacts from a previous run's parquets without re-solving.)

### Usage

```
python execute_flextool_workflow.py INPUT_DB_URL OUTPUT_DB_URL SCENARIO_NAME [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `INPUT_DB_URL` | Input database URL or file path (e.g. `sqlite:///input.sqlite` or `input.sqlite`) |
| `OUTPUT_DB_URL` | Output database URL for storing result metadata |
| `SCENARIO_NAME` | Name of the scenario to execute |

### Options

| Flag | Description |
|---|---|
| `--tabular-file-path PATH` | Path to Excel/ODS input file (mutually exclusive with `--csv-directory-path`). Triggers Phase 1. |
| `--csv-directory-path PATH` | Path to directory containing CSV input files. Triggers Phase 1. |
| `--write-methods METHOD [...]` | Output formats to generate (default: `plot parquet csv`). Choices: `plot`, `parquet`, `excel`, `csv`, `spinedb`. Parquet is the canonical output and is always produced when this flag is omitted. `spinedb` dumps the processed results into a SpineDB using the FlexTool results schema (one Spine alternative per run, named after the scenario, so multiple runs coexist in one file). |
| `--output-subdir DIR` | Subdirectory for output files (default: scenario name) |
| `--output-config PATH` | Path to output configuration YAML (default: bundled `flextool/schemas/default_plots.yaml`) |
| `--debug` | Forward `--debug` to the model run (enables verbose memory checkpoints and per-solve CSV diagnostics) |

### Input sources

**From an Excel or ODS file:**

```bash
python execute_flextool_workflow.py \
    input.sqlite output_info.sqlite my_scenario \
    --tabular-file-path my_input.xlsx
```

**From a directory of CSV files:**

```bash
python execute_flextool_workflow.py \
    input.sqlite output_info.sqlite my_scenario \
    --csv-directory-path input_data/
```

**From an existing Spine database** (no `--tabular-file-path` / `--csv-directory-path` &rArr; Phase 1 is skipped automatically):

```bash
python execute_flextool_workflow.py \
    sqlite:///input.sqlite output_info.sqlite my_scenario
```

## `run_flextool.py` &mdash; model execution

Runs the FlexTool optimization model directly. This is called internally by `execute_flextool_workflow.py` but can also be used standalone. After solving, it writes outputs and records scenario information in the output database.

### Usage

```
flextool-run INPUT_DB_URL [OUTPUT_DB_URL] [options]
```

`python run_flextool.py` and `python -m flextool.cli.cmd_run_flextool` take the same arguments.

Each run writes two files into its `output_parquet/<subdir>/` folder: `_run_status.json` (running, succeeded,
failed, ...; each solve's outcome; the solver option files used: folder, and a SHA-256 of each; whether the
parquet files there belong to this run) and `_run.log` (the run's
console output). Under `flextool-run`, the GUI, the workflow runner and the calibrator the log also holds the
solver's native output and any crash dump.

### Positional arguments

| Argument | Description |
|---|---|
| `INPUT_DB_URL` | Input database URL |
| `OUTPUT_DB_URL` | Output database URL for result metadata (optional) |

### Options

**Run control and I/O paths:**

| Flag | Description |
|---|---|
| `--scenario-name NAME` | Scenario name to execute (if omitted, uses the active DB filter) |
| `--settings-db-url URL` | Settings DB consulted for post-processing defaults |
| `--migrate` | Migrate the input database to the current FlexTool version before the run, if it is older (see [Migrate database schema](#migrate-database-schema)). A frozen database is not migrated: the run stops with exit code 2. Without the flag, an older database is refused. |
| `--output-spreadsheet PATH` | Save results to a spreadsheet file |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv`, `spinedb` (default: `plot parquet`) |
| `--results-db-url URL` | Target SpineDB for the `spinedb` write-method (default: `<output-location>/results.sqlite`) |
| `--output-config PATH` | Path to output configuration YAML (default: the bundled `flextool/schemas/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--output-location DIR` | Override output location path |
| `--output-subdir NAME` | Subdirectory under `output_parquet/` etc. (default: scenario name) |
| `--flextool-location PATH` | Spine Toolbox shim: directs outputs to the FlexTool repo root rather than the work directory |
| `--work-folder DIR` | Working directory for intermediate files (default: CWD). Enables parallel scenario execution by isolating each run |
| `--only-first-file-per-plot` | Only produce the first file for each plot (quick overview mode) |
| `--write-plot-data` | Also write the data drawn in each plot as a CSV file next to it (same name, `.csv`); see [Results](results.md#plotted-data-as-csv) |
| `--input-source-file PATH` | The file the input database was made from (e.g. an Excel file), recorded in `_run_status.json` |
| `--run-launcher NAME` | Who started the run, recorded in `_run_status.json` (default `direct`; `flextool-run` adds `flextool-run` unless one is given) |

**Debug and diagnostics:**

| Flag | Description |
|---|---|
| `--debug` | Enable verbose logging, per-checkpoint memory trace, and a `solve_data/memory_diagnostics.csv` per-solve CSV |
| `--csv-dump` | Preserve cascade debug artefacts on disk: `input/`, `solve_data/`, `cross_solve/`, and `output_raw/` mirrors. Off by default (these directories are normally cleaned up after the run) |

**Scaling and solver tuning:**

Each of these flags has an environment-variable twin for runs without a command line (Spine Toolbox, a
script); a given flag wins over the variable, and the variable over the solve's database value. See
[Environment variables](env_vars.md). `flextool-run` passes them to the engine unchanged.

| Flag | Description |
|---|---|
| `--scaling {off,solver_only,basic,full}` | FlexTool autoscaler strategy (see [scaling.md](dev/scaling.md)). `off` disables all scaling including HiGHS' internal equilibration; `solver_only` keeps only HiGHS' default matrix scaling; `basic` adds FlexTool's Layer-1 range detection and Layer-3 `user_*_scale` recommendation without LP mutation; `full` (default) adds Layer-2 semantic per-type column/row/cost scaling. Env fallback: `FLEXTOOL_SCALING` |
| `--user-bound-scale N` | Pin HiGHS' `user_bound_scale` exponent (multiplies all column bounds and RHS by `2**N`). Use the value HiGHS suggests in its scaling warning; clamped to `[-10, 0]`. Overrides the Layer-3 auto-pick. Env fallback: `FLEXTOOL_USER_BOUND_SCALE` |
| `--presolve {on,off,choose}` | HiGHS `presolve` override. Default keeps the determinism-pinned `on`; `off` disables presolve (slower but useful for memory or numerical diagnostics). Env fallback: `FLEXTOOL_HIGHS_PRESOLVE` |
| `--threads N` | Threads for one FlexTool run: FlexTool's data processing and the solver. FlexTool's own processing (polars) uses at most N and at most 8 (`POLARS_MAX_THREADS` sets another limit below N); the solver gets N threads (HiGHS `threads=N` with its parallel simplex off; Gurobi, CPLEX, Xpress and COPT their thread count, over the option file and `solver_arguments`, with one printed line when a different value is replaced); a Benders solve runs at most N region solves at once. Default: the available cores, with the solvers' own thread settings and Benders as before. When you run several FlexTool runs at once (for example an AI agent or a script starting them in parallel), give each about cores / runs, e.g. `flextool-run input.sqlite --scenario-name base --threads 4`. Env fallback: `FLEXTOOL_THREADS` |
| `--highs-threads N` | Number of HiGHS solver threads (default `1`). Values >1 enable HiGHS parallel mode and trade determinism for wall-clock speedup; at most `--threads` when that is given. Env fallback: `FLEXTOOL_HIGHS_THREADS` |
| `--solver-time-limit SECONDS` | Time limit of each solve, on every solver (HiGHS `time_limit`, Gurobi `TimeLimit`, CPLEX `timelimit`, Xpress `MAXTIME`, COPT `TimeLimit`), a positive number (default: none; 0 or a negative value is ignored with a warning). Wins over a time limit in the solve's `solver_arguments` and in the solver's option file. A MIP stopped at the limit keeps its solution with a warning; an LP, or a solve without a solution, stops the run (see [Time limit](solvers/index.md#time-limit)). Env fallback: `FLEXTOOL_SOLVER_TIME_LIMIT` (`FLEXTOOL_HIGHS_TIME_LIMIT`: HiGHS only) |
| `--solver-mip-gap GAP` | MIP relative gap for every solve of the run, on every solver (HiGHS `mip_rel_gap`, Gurobi `MIPGap`, CPLEX `mip tolerances mipgap`, Xpress `MIPRELSTOP`, COPT `RelGap`). Unset: each solve's `solver_mip_gap`, else the gap in the solver's option file. Env fallback: `FLEXTOOL_SOLVER_MIP_GAP` (`FLEXTOOL_HIGHS_MIP_GAP`: HiGHS only) |
| `--solver-config-dir PATH` | A folder of the solver option files (`highs.opt`, `gurobi.opt`, ...); repeat the flag for several folders, highest first: each solver's file comes from the first folder that has it, else FlexTool's default, and only the folders given are read. Unset: `solver_config/` next to the input database; for a GUI project database the project's `solver_config/`, then the workspace's, file by file; then the Spine Toolbox project folder (`--project-folder-file`), then the current folder; FlexTool's defaults with a warning when none has an option file. The run header's `Solver config:` line names the folder of each file; the run copies the files to `<work folder>/solve_data/solver_config/` (see [Where the option files come from](solvers/index.md#where-the-option-files-come-from)). Env fallback: `FLEXTOOL_SOLVER_CONFIG_DIR` (several folders separated by `:`, on Windows `;`) |
| `--solver-log-level {silent,normal,verbose}` | Solver log output (default `normal`); `silent` also hides a commercial solver's log (still saved as `solve_data/subprocess/<solve>.log`); `verbose` is HiGHS only. Env fallback: `FLEXTOOL_SOLVER_LOG_LEVEL` |
| `--save-memory` | Opt-in peak-RSS reduction: after the LP matrix is built, drop polar-high's polars/numpy source, write the LP to a temp MPS file, and solve it in a separate HiGHS subprocess (via `cmd_solve_mps`) so the solver's working set lives outside the FlexTool address space. Frees ~5-10 GB on large models at ~+90 s I/O per sub-solve. Also disables warm-LP reuse across cascade iterations. See [architecture.md](dev/architecture.md). Env fallback: `FLEXTOOL_SAVE_MEMORY=1` |
| `--warm-start` | Reuse a cached HiGHS basis for solves of the same model structure, in-process and with `--save-memory` (Benders solves excepted); any mismatch falls back to a cold solve. Env fallback: `FLEXTOOL_WARM_START=1` |
| `--basis-cache-dir PATH` | Folder of the `--warm-start` basis cache (default `<work folder>/basis_cache`); share one folder to reuse bases between runs. Env fallback: `FLEXTOOL_BASIS_CACHE_DIR` |
| `--benders-workers N` | Threads for the region solves of a Benders solve (default, or `0`: the number of regions, at most cores minus one). Env fallback: `FLEXTOOL_BENDERS_WORKERS` |

**Decomposition (Benders):**

Benders decomposition is selected **per solve from the database**, not
via a CLI flag: set `solve.decomposition = benders` (plus the optional
`solve.benders_max_iter` / `benders_tolerance` knobs) and declare at
least two `group.decomposition_method = benders_regional` region groups.
A single chain can mix schemes. See
[decomposition.md](dev/decomposition.md). The one remaining CLI flag is
the filter-only inspector:

| Flag | Description |
|---|---|
| `--region GROUP_NAME` | Filter-only entry point: produce a per-region input directory `input_region_<GROUP>/` for Benders decomposition and exit without solving. Cross-region processes are replaced with import/export half-flows; coupling variables are listed in `solve_data/region_coupling.csv` |

### Example

```bash
flextool-run \
    sqlite:///templates/examples.sqlite \
    sqlite:///output_info.sqlite \
    --scenario-name base
```

Return codes: `0` = success, `1` = infeasible or unbounded, `-1` = failure.

## `write_outputs.py` &mdash; output generation

Generates plots, parquet files, CSV, or Excel outputs from existing model results. Useful for re-plotting with different settings without re-running the model.

### Usage

```
python write_outputs.py [options]
```

All arguments are optional. When run from the terminal, you typically provide `--scenario-name`.

### Key options

| Flag | Description |
|---|---|
| `--scenario-name NAME` | Scenario with raw outputs available (when re-plotting a single scenario from the terminal) |
| `--input-db-url URL` | Input DB with scenario filter (used when `run_flextool.py` chains into this command from Toolbox) |
| `--output-locations-db-url URL` | Output-locations DB holding paths of existing outputs (re-plotting from Toolbox) |
| `--settings-db-url URL` | Settings DB consulted for unset parameters |
| `--read-parquet-dir DIR` | Read from existing parquet files instead of raw CSVs (faster) |
| `--config-path PATH` | Output configuration YAML (default: the bundled `flextool/schemas/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv`, `spinedb` (default: `plot parquet excel`) |
| `--results-db-url URL` | Target SpineDB for the `spinedb` write-method (default: `<output-location>/results.sqlite`). Note: `spinedb` needs the native solve path, so it is skipped (with a warning) when re-rendering via `--read-parquet-dir`. |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--subdir DIR` | Subdirectory for outputs (default: scenario name) |
| `--output-location DIR` | Root directory for input/output locations (default: flextool root); overridden by `--output-locations-db-url` |
| `--plot-file-format {png,svg}` | File format for plots (default: `png`) |
| `--only-first-file-per-plot` | Only produce the first file for each plot (quick overview mode) |
| `--write-plot-data` | Also write the data drawn in each plot as a CSV file next to it (same name, `.csv`); see [Results](results.md#plotted-data-as-csv) |
| `--single-result KEY CSV_NAME PLOT_NAME PLOT_TYPE SUBPLOTS_PER_ROW LEGEND_POSITION` | Process a single result (overrides `--config-path`). Use `"null"` for None values |
| `--debug` | Enable debug output |

### Examples

**Re-plot from raw CSV outputs:**

```bash
python write_outputs.py --scenario-name base
```

**Re-plot from existing parquet files:**

```bash
python write_outputs.py \
    --scenario-name base \
    --read-parquet-dir output_parquet/base
```

**Generate only Excel output:**

```bash
python write_outputs.py \
    --scenario-name base \
    --write-methods excel
```

**Plot a different time range (rows 0 to 500):**

```bash
python write_outputs.py \
    --scenario-name base \
    --plot-rows 0 500
```

## `scenario_results.py` &mdash; cross-scenario comparison

Reads results from multiple scenarios and generates comparison plots and optional spreadsheets.

### Usage

```
python scenario_results.py [DB_URL] [options]
```

### Options

| Flag | Description |
|---|---|
| `--parquet-base-dir DIR` | Base directory containing per-scenario parquet subdirectories |
| `--alternatives NAME [...]` | Specify scenario/alternative names manually |
| `--dispatch-plots` | Generate dispatch area plots for nodes and node groups |
| `--write-to-xlsx` | Write combined results to Excel file |
| `--write-dispatch-xlsx` | Write dispatch data to Excel in the plot directory |
| `--write-to-ods` | Write combined results to ODS file |
| `--output-config-path PATH` | Plot configuration YAML (default: the bundled `flextool/schemas/default_plots.yaml`, which also holds the comparison plots) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--plot-dir DIR` | Directory for comparison plots (default: `output_plot_comparisons`) |
| `--excel-dir DIR` | Directory for comparison Excel files |
| `--plot-file-format FORMAT` | File format for plots: `png` or `svg` |
| `--shared-legend` | Use shared legend across subplots |
| `--only-first-file-per-plot` | Only produce the first file for each plot |
| `--write-plot-data` | Also write the data drawn in each plot, dispatch plots included, as a CSV file next to it; see [Results](results.md#plotted-data-as-csv) |
| `--show-plots` | Display plots interactively in addition to saving |

### Examples

**Generate dispatch plots for specific alternatives:**

```bash
python scenario_results.py sqlite:///output_info.sqlite \
    --dispatch-plots --alternatives base high_RE
```

**Compare scenarios from parquet directories (without output DB):**

```bash
python -m flextool.cli.cmd_scenario_results \
    --parquet-base-dir output_parquet \
    --alternatives base network_all_tech \
    --dispatch-plots --write-to-xlsx
```

---

## Input preparation commands

These commands convert tabular input data into a Spine database. They are called automatically by `execute_flextool_workflow.py` and by the GUI, but can also be run manually.

### Import self-describing Excel

The self-describing format embeds metadata (entity classes, parameter names, data types) directly in the Excel sheet headers. This is the recommended input format.

```bash
python -m flextool.cli.cmd_read_self_describing_tabular_input \
    my_input.xlsx sqlite:///input.sqlite
```

| Argument | Description |
|---|---|
| `xlsx_path` | Path to the self-describing Excel file |
| `target_db_url` | Target database URL |
| `--keep-entities` | Keep existing entities during purge (default: purge all) |
| `--no-purge` | Do not purge existing data before importing |

### Import with specification file

The older format uses a separate JSON specification file to define how Excel/ODS/CSV columns map to database entities.

```bash
python -m flextool.cli.cmd_read_tabular_input \
    sqlite:///input.sqlite \
    --tabular-file-path my_input.xlsx
```

| Argument | Description |
|---|---|
| `target_db_url` | URL to FlexTool input database |
| `--tabular-file-path PATH` | Path to Excel or ODS input file |
| `--csv-directory-path PATH` | Path to directory containing CSV input files |
| `--migration-follows` | Accept version mismatch (migration will run after import) |

### Import old FlexTool v2 files

Converts an old-format FlexTool 2 input data file (`.xlsm`, 18 fixed sheets) to a new FlexTool database. With `--master`, the sensitivities of the FlexTool 2 master workbook (the one with the run buttons and the `Sensitivity definitions` sheet) are imported in the same run, as alternatives and scenarios on top of `base`.

```bash
python -m flextool.cli.cmd_read_old_flextool \
    InputData/old_model.xlsm sqlite:///input.sqlite \
    --master flexTool.xlsm
```

| Argument | Description |
|---|---|
| `xlsm_path` | Path to the FlexTool 2 input data file (one of the master workbook's `InputData/*.xlsm`) |
| `target_db_url` | Target database URL. A new file is created. An existing FlexTool database is accepted only at version 56 (see below) |
| `--master PATH` | FlexTool 2 master workbook: import its sensitivities too |
| `--alternative-name NAME` | Name for the alternative (default: `base`) |
| `--no-purge` | Do not purge existing data before importing |
| `--no-migrate` | Leave the database at version 56 instead of migrating it |
| `--strict` | Stop with an error, and leave no database, when a value cannot be written (default: list it in the report) |
| `--verbose`, `-v` | Debug logging |

The importer writes the data at a fixed database version (56) and then migrates the database to the current version. The sensitivities are written before the migration, so give `--master` in the same command. A database that is already at another version is refused: writing version-56 data into it would lose every value whose name or meaning changed since. So is a database that has data but is not a FlexTool database. `flextool.cli.cmd_import_sensitivities MASTER BASE.xlsm sqlite:///db.sqlite` adds sensitivities later only to a database left at version 56 with `--no-migrate`; migrate it afterwards with `flextool-migrate-database db.sqlite`. Giving the master workbook as the input file stops with a message that names the right files. A database the command migrates carries no behaviour-change notices: they describe changes between FlexTool 4 versions, which an imported FlexTool 2 model never ran on. A database left at version 56 with `--no-migrate` and migrated later with `flextool-migrate-database` does carry them; they can be ignored for an imported FlexTool 2 model.

**Import report.** The run ends with a report written for a reader who is not a modeller: what was written, every value the database refused, and, grouped by topic (time and costs, units, nodes and connections, sensitivities), every FlexTool 2 input that was not carried over or works differently in FlexTool 4, with the entities that use it and what to do. Check it before using the model. The inputs it lists:

| FlexTool 2 input | What differs |
|---|---|
| master `time_in_years` that differs (by more than 1 %) from the hours the time series covers / 8760 | FlexTool 2 scales the operating costs by the time the series *represents* (`time_in_years`); FlexTool 4 scales them by the hours the time steps cover, to a full year, and has no input to change that. The ratio of operating to investment costs then differs by the factor the line gives (e.g. 365 for 24 one-hour steps with `time_in_years` = 1), and so do the investments. Listed first. To keep FlexTool 2's balance, multiply the investment and fixed costs by that factor. |
| cf and inflow series with steps not in use | FlexTool 2 scales each series so that its mean over the steps in use equals its mean over the whole series (separately for the investment and the dispatch run); FlexTool 4 uses the values as they are. The line gives the factor. |
| unit_type `availability` < 1 | In FlexTool 2 it enters only the capacity margin; it does not reduce generation. In FlexTool 4 `availability` limits generation and derates the unit in the capacity margin together, so it is not imported: generation is as in FlexTool 2, and in a capacity margin the units count with their full capacity. |
| master `curtailment_penalty` | FlexTool 4 has no curtailment cost. |
| master `reserve_duration` | FlexTool 4 has `reserve__upDown__group.reserve_duration`, which the version-56 importer cannot write. Set it after the import. |
| unit_type `conversion eff` | The output / input ratio of a unit that takes its input from a node (e.g. a heat pump COP). Set the unit's `efficiency` to it, where the unit has no efficiency time series. |
| units `use min online` (with its `ts_unit` `min_online`) | FlexTool 4 has no time-series floor on the online capacity. The series of a unit whose flag is 0 is not listed: FlexTool 2 ignores it too. |
| units `fueluse increase eq x output2` | FlexTool 2's fuel-use equation adds this fuel only when output2 is on the unit's own output node, so in a normal workbook it adds nothing in FlexTool 2 either; the import ignores it. |
| gridNode `import (MWh)` | The import series is not added to the node. |
| a generation-limit flag on a storage unit | Not carried over (the unit becomes a charger and a discharger). |
| any other non-zero column the importer does not read | Listed with a note to check whether it matters. |

Sensitivity overrides that the sensitivity import does not translate are listed too, with their scenario (alternative), where the value can be set by hand.

**Must-run and generation limits.** In FlexTool 2, the `units` flags `fix unit generation`, `use min generation` and `use max generation` use the `ts_unit` series `fix_generation`, `min_generation` and `max_generation` of the same grid / node / unit: the share (0-1) of the unit's capacity it must generate, at least or at most. Each becomes a FlexTool 4 profile `<unit>_<series>` (a time series) and a `unit__node__profile` (unit, output node, profile) with `profile_method` `fixed`, `lower_limit` or `upper_limit`. The limit is on the primary output only; an output2 follows through the CHP constraints. A flag of 0 ignores the series, as in FlexTool 2. The report lists a flag without its series, a `use max generation` on a unit with a cf profile (FlexTool 2 ignores it there) and a flag on a conversion unit (FlexTool 2 bounds its input flow). FlexTool 2's `max_generation` also caps the online capacity of a unit with online variables; the FlexTool 4 profile caps only the generation, so the minimum load and start-ups of such a unit can differ a little where the series is below 1 (the report notes these units).

**CHP (output2).** FlexTool 2's output2 (e.g. the heat of a CHP unit) uses no fuel and is outside the unit's capacity, minimum load and ramp limits; it is tied to the primary output only by the `output2 eq / gt / lt` rows and `output2_max_capacity_ratio`. In FlexTool 4 the output2 edge gets `conversion_flow_coeff` 0 (out of the conversion equation and of the capacity limits), and each FlexTool 2 row becomes a user `constraint` `<unit>_eq`, `_gt`, `_lt` or `_output2_max`: output2 − coeff × output1 {=, ≥, ≤} constant × capacity, where the capacity is the unit's capacity in service in each solve: `constraint_cumulative_pre_built_capacity_coeff` (existing capacity, including what an earlier solve invested) plus `constraint_invested_capacity_coeff` (what the solve invests), both on the unit. So an invested CHP keeps its output2 in the dispatch solve that follows the investment solve. In older workbooks, which have the column `output2 max capacity (MW)` in place of `output2 max capacity ratio`, the eq / gt / lt constants are MW: the rows are output2 − coeff × output1 {=, ≥, ≤} constant, with no capacity term, `output2 max capacity (MW)` becomes a plain upper limit in MW, and the report says the constants were read as MW. The report also lists an output2 without any row: its output2 is free (no fuel, no limit), as in FlexTool 2.

**Capacity.** In FlexTool 2, `invested capacity (MW)` and `invested storage (MWh)` are capacity already built: they count in every capacity limit, in every mode, and are not part of the investment decision. So `existing` is `capacity (MW)` + `invested capacity (MW)` (units, chargers and dischargers, connections; for a connection the larger direction) and `storage (MWh)` + `invested storage (MWh)` (storage nodes). `max invest (MW)` / `max invest (MWh)` is the investment limit (`invest_max_total`). A sensitivity of either column changes the sum. FlexTool 2 also adds the investment annuity of the invested capacity to its total cost as a fixed amount; FlexTool 4 does not, so the report lists these entities (those with an investment cost, also when a sensitivity gives the capacity). The `fixed kW/kWh ratio` ties the invested MW to the invested MWh (MW = ratio × MWh), as in FlexTool 2; when only one `max invest` is given, the other is derived through the ratio, as FlexTool 2 does. A storage node stays a storage while it has capacity or can invest in it; a sensitivity that leaves it neither (e.g. `invested storage (MWh)` and `max invest (MWh)` 0, no `storage (MWh)`) makes it a balance node.

**Time steps.** The steps with `in_use` 1 on the `ts_time` sheet form the dispatch timeset, those with `in_use_invest` 1 the investment timeset (as FlexTool 2's model). A workbook whose solves have no step in use is refused with a message, before any database is written: FlexTool 2 would solve nothing either. (A FlexTool 2 master workbook can take `ts_time` from its time series folder, `ts/<input file name>/ts_time.csv`; the importer reads only the workbook, so copy the column into it.)

**Storage start and finish.** A non-zero `storage start` / `storage finish` fixes the first / last storage state at that share of the storage capacity (`storage_start_end_method` `fix_start`, `fix_end` or `fix_start_end`); 0 or empty fixes nothing (`fix_nothing`). The storage state wraps from the last step to the first (`bind_within_solve`), except when both are set (`bind_forward_only`), as in FlexTool 2.

---

## Export and maintenance commands

### Export database to Excel

Exports a Spine database back to Excel (.xlsx) in self-describing format.

```bash
python -m flextool.cli.cmd_export_to_tabular \
    sqlite:///input.sqlite output.xlsx
```

| Argument | Description |
|---|---|
| `db_url` | URL to FlexTool input database |
| `output_path` | Output Excel file path (.xlsx) |
| `--include-advanced` | Include advanced sheets (solve sequences, stochastic data) |
| `--old-format` | Use the old v1 format instead of self-describing v2 |

### Migrate database schema

Upgrades a database to the latest FlexTool schema version using the templates bundled under `flextool/schemas/`.

```bash
flextool-migrate-database input.sqlite
# or: python -m flextool.cli.cmd_migrate_database input.sqlite
```

| Argument | Description |
|---|---|
| `filepath` | Path to the database file (absolute, or relative to the current folder) |
| `--copy-to PATH` | Copy the database to `PATH` and migrate the copy; the original is not changed. For a frozen database. |
| `--allow-migrate-frozen` | Migrate a frozen database in place; its frozen marker is kept with the new version. |

The model run (`flextool-run`, `run_flextool.py`, `python -m flextool.cli.cmd_run_flextool`) does not run a database that is not at the current FlexTool version: it stops with a message naming this command. With `--migrate` it migrates the database first, the same way as this command: a frozen database is not migrated, and the run stops with the frozen message and exit code 2. Runs started at the same time with `--migrate` on one database migrate it once: the migration holds a lock on `<database>.migrate.lock` next to the database (the file stays there; it can be deleted when no run is starting), and the other runs print "Waiting for another migration of <database> ...", wait for it and then find the database current. Every in-place migration takes this lock: this command, the Toolbox **Migrate database** tool, the FlexTool window's refresh, `--migrate` and `execute_flextool_workflow.py`. A database that is already current does not wait for the lock and gets no lock file. On Windows a run waits for the lock at most 10 minutes (set `FLEXTOOL_MIGRATION_LOCK_TIMEOUT` to a number of seconds to change it; it is not read on Linux or macOS) and then stops with a message naming the lock file. A file system without file locks (some network mounts) stops the migration with a message: copy the database to a local folder. On a network folder that several machines share, the lock may only coordinate runs on one machine (CIFS, NFS mounted with `local_lock`): start the migration from one machine first. The settings databases (`--settings-db-url`, the output database) are not migrated. `execute_flextool_workflow.py` migrates the input database before it runs the model, the FlexTool window migrates its input sources when it refreshes them, and the Spine Toolbox workflow runs its **Migrate database** tool before FlexTool.

A frozen database older than this FlexTool (or at the current version but with an older SpineDB schema) is refused with exit code 2 and a message naming the two options: `--copy-to NEW.sqlite` migrates a copy (the copy is not frozen; it records the label, the old version and the SHA-256 of the frozen file it was made from) and `--allow-migrate-frozen` migrates the frozen file itself. A frozen database that is already current, or newer than this FlexTool, is left unchanged, also with `--allow-migrate-frozen`. If FlexTool cannot read the file to check for the marker (another program holds a lock on it), it does not migrate it and says so. `flextool-update` skips frozen databases and lists them, and `execute_flextool_workflow.py` stops on one.

### Freeze a released database

A released model (a reference case, a study's final input) can be frozen so that this FlexTool and later versions do not migrate it in place without being told to (older FlexTool versions do not know the marker). The marker is a row in the database's `metadata` table, so it travels with the file; it does not change the model. Only a database at the current version can be frozen.

```bash
flextool-freeze input.sqlite --label "Reference case 1.0"
flextool-freeze input.sqlite --status
flextool-freeze input.sqlite --unfreeze
```

| Argument | Description |
|---|---|
| `database` | The `.sqlite` file (a path or a `sqlite:///` URL) |
| `--label LABEL` | Freeze with this label (at most 170 characters, no vertical bar) |
| `--status` | Print the data version, the frozen marker and, for a migrated copy, what it was migrated from |
| `--unfreeze` | Remove the frozen marker |

Freezing writes into the file, so take the SHA-256 of the released file after freezing it. Editing a frozen database (in the database editor, or by importing data into it) is not prevented.

### Sharing a project

`flextool-project` writes a project to one file, checks such a file and imports it (the FlexTool window does the same from **Manage projects… → Export… / Import…**).

```bash
flextool-project export UAE_ref --results WOM_1,WEM_1 -o UAE_ref.flextool.zip
flextool-project verify UAE_ref.flextool.zip
flextool-project import UAE_ref.flextool.zip --name UAE_ref_received
```

| Command / argument | Description |
|---|---|
| `export PROJECT` | The project folder, or the name of a project in `./projects` |
| `-o FILE` | The package file (default `./<Name>_<YYYYMMDD>.flextool.zip`) |
| `--results none\|all\|RUN,RUN` | The result runs (folders of `output_parquet/`) to include; default `none` |
| `--no-external` | Do not copy the input files that the project reads from outside its folder |
| `--keep-run-records` | Ship each run's `_run_status.json` and `_run.log` as they are (default: this computer's folder paths, user name and host name are replaced in the record and the log is left out) |
| `--force` | Replace the package file when it exists (without it, an existing file stops the export) |
| `--solver-config-dir DIR` | Where the `.opt` files are read from; repeat for several folders, highest first (per solver the first folder's file is shipped). Default: the project's `solver_config/`, then `$FLEXTOOL_SOLVER_CONFIG_DIR`, else `./solver_config`, as a FlexTool window run of the project reads them |
| `verify FILE` | Check every checksum and the versions, unpacking nothing; exit code 1 on a problem |
| `import FILE` | Unpack into `./projects/` (`--projects-dir DIR` for another folder) under the package's name or `--name NAME`; exit code 3 when the name is taken (the message suggests `<Name>_2`), 2 when the package is refused |

The package holds the input sources (external input files copied in under their own source numbers), `settings.yaml` without the machine-specific settings (`execution_limits`, `max_workers`, `scenario_resource_history`, `external_refs`), `plot_settings.yaml`, the solver option files the project's runs use (per solver the project's `solver_config/<solver>.opt`, else the workspace's; only the five solver files, and no unedited FlexTool template), and the chosen result runs, with `output_parquet/_axis_bounds.json` cut to those runs. `--results all` leaves out a run that is still being written, and naming such a run stops the export. A run's record (`_run_status.json`) and log (`_run.log`) show this computer's folder paths, user name and host name: by default the record has them replaced by `<project>`, `~`, `<path>/`, `<host>` and `<user>` and the log is left out (`--keep-run-records` keeps both as they are). Plots, CSV and Excel outputs, `results.sqlite` and the scratch folders are left out; `flextool-write-outputs` re-creates them from the results (`--write-methods spinedb` for `results.sqlite`). A `.sqlite` input with a `-journal` or `-wal` file next to it (a program may be writing to it) is copied through SQLite as a consistent database, others byte for byte; `-journal`, `-wal` and `-shm` files are never packaged, and a file that changes while it is copied stops the export. Names Windows cannot hold (`CON`, `NUL`, `COM1`…, `< > : " | ? *`, a trailing dot or space, names that differ only in case) stop the export, because a package must unpack on every system. `plot_settings.yaml` travels as it is, folder paths in it included. A solver `.opt` file with an option that names a file or folder by its path, or that connects the solver to another computer (a server, cloud or licence service, or its password), is left out of the package, with a warning that names the file and its lines; remove those lines to ship it. Its `manifest.json` records the FlexTool, polar-high, HiGHS, polars, spinedb_api and Python versions, the data version and frozen marker of each input database, the runs whose input changed after the run (`stale_runs`) and the SHA-256, size and modification time of each file. Export reads the databases read-only: nothing is migrated, and a frozen database stays frozen.

Import refuses member names that would leave the project folder or that Windows cannot hold, refuses a package that does not fit on the disk, checks every checksum before it writes anything, unpacks into a hidden `projects/.<name>.partial/` folder, gives each file its recorded modification time back and then renames the folder into place. A database newer than this FlexTool is refused; an older one is migrated when the project is first opened, with the usual consent for a frozen one. The shipped `.opt` files go to `solver_config/` in the project, where its runs use them ahead of the workspace's, file by file; the workspace's own are never overwritten. A shipped file with an option that names a path or another computer (a server, cloud or licence service) is kept as `<solver>.opt.review`, which no run reads, until you check it and rename it to `<solver>.opt`; the import message names its lines and says whether the shipped files differ from the workspace's (with the lines that differ). A project imported by an earlier FlexTool 4.4 development version keeps its options in `imported_solver_config/` until the FlexTool window first opens it, which renames the folder to `solver_config/`; a terminal run of such a project uses the workspace's options until then. It names any input left out of the package: its scenarios cannot be run again until that file is added. Errors such as a full disk or a folder that cannot be written are reported as one message (exit code 2); a stopped import leaves a hidden `projects/.<name>.partial/`, which the GUI does not list and the next import of that name removes. The checksums detect a file damaged on the way, not a deliberate change: the manifest is not signed.

**Reproducing a delivered run.** Send a project with the inputs and the one scenario run to be reproduced (`--results WOM_1`). The receiver runs `flextool-project verify`: it prints the versions the run was made with and one line, `pip install flextool==X polar-high==Y highspy==Z polars==W`, that installs the same FlexTool, solver interface, HiGHS and polars (the results depend on all four; import warns when the installed polar-high or highspy differs). The receiver then imports the package, makes the shipped solver options active as above, runs the scenario again, and compares the new results with the shipped `WOM_1` (whose `_run_status.json` records the verdict, the solver and the input data version). Because the source number is kept, the re-run writes to the same `WOM_1` folder name, so first move the shipped folder **outside** `output_parquet/` (a renamed folder inside it shows up as another run), or compare against a copy of it kept elsewhere.

### Check that exercises run fast enough

For a course, `flextool-time-runs` runs scenarios one after another on the trainer's (or a participant's)
computer and shows where each run's time goes. It prints the computer first (processor, cores, memory, system and
FlexTool version), so the tables of different laptops can be compared. Before each run it prints the run's number
and scenario, and the time when the run ends; the whole default set of examples takes several minutes.

```bash
flextool-time-runs                                  # every example and how-to scenario that comes with FlexTool
flextool-time-runs --source templates_examples --scenario coal "network_*"
flextool-time-runs --db projects/Course/input_sources/exercises.sqlite --scenario Ex1 Ex2 --target 30 --strict
```

Put a pattern in double quotes, as above: that works in bash, PowerShell and the Windows Command Prompt (which
keeps single quotes as part of the name). On Windows, open the terminal with the FlexTool environment active
(`.venv\Scripts\activate`, see [Installation](install_gui.md)) so that `flextool-time-runs` is found.

The table has one row per run, slowest first, in seconds; `!` marks a run over the target (default 60 s):

| Column | What it covers |
|---|---|
| `wall` | The whole run, started with the command and thread count the FlexTool window uses by default (see `--threads`) |
| `start-up` | Starting Python and loading FlexTool |
| `read input` | Reading the input database, preparing the input data and setting up the run |
| `preprocess` | Building each solve's data and its optimisation problem |
| `solve` | The solver |
| `outputs` | Writing the results (parquet) |
| `viewer` | The data the result viewer needs (also written with plots off) |
| `plots` | The scenario plots (png); zero with plots off |
| `untimed` | The rest. A Benders solve writes no per-solve problem-building, solving or result-writing times, so those land here |

| Argument | Description |
|---|---|
| `--db PATH` | Run the scenarios of this `.sqlite` input database (a path or a `sqlite:///` URL; can be given several times) instead of the examples. The runs read a copy of it made in the temporary folder, so the file itself is never opened by a run |
| `--source JSON` | Build and run a JSON database: a file, or a shipped one by name (`templates_examples`, `howto_demand`, …) |
| `--scenario NAME [...]` | Scenario names or patterns (`*`, `?`); default every scenario. A name that matches nothing is an error |
| `--plots off\|on\|both` | Write the scenario plots in the runs (default `off`); `both` runs each scenario twice |
| `--threads N` | Threads for each run, as **Threads per run** in the FlexTool window (default 1, the window's default). Give the value the course project uses |
| `--target SECONDS` | Mark runs longer than this (default 60) |
| `--strict` | Exit code 2 when a run is over the target. A failed run always gives exit code 1 |
| `--keep DIR` | Keep the runs' work folders, results and logs in `DIR` (default: a temporary folder that is removed) |
| `--csv FILE` | Also write the table to a CSV file |

The runs go one after another, because runs in parallel slow each other down. The run times come from each run's
`timings.csv` (in `output_csv/<scenario>/`), which every run writes. With `--plots on`, the runs draw the
default plot set, not only the output configurations a project has made active, so the `plots` and `viewer`
times can differ a little from that project's runs in the window.

Plots are off in the timed runs by default because they are the slowest part of a small model's run and an
exercise does not need them: participants see every plot in the result viewer, which draws them from the results.
Switch them off for an exercise project with the **Auto-gen** tick of **Scenario pngs** in the FlexTool window
(see [Outputs](flextool_gui_interface.md#outputs-unified-table); the tick is saved in the project's
`settings.yaml`, so a shared project keeps it), with `--write-methods parquet` on `flextool-run`, or with
`output-plot` in the Spine Toolbox settings database.

### Update FlexTool

Pulls the latest version from GitHub and migrates project databases.

```bash
python -m flextool.cli.cmd_update_flextool
```

---

## Worked examples

These examples use the bundled `templates/examples.sqlite` database and the `base` scenario.

### 1. Run a full workflow from an existing database

```bash
# Run model and generate outputs (plots, parquet, CSV)
# No --tabular-file-path / --csv-directory-path → Phase 1 is skipped automatically
python execute_flextool_workflow.py \
    templates/examples.sqlite \
    output_info.sqlite \
    base

# Output plots are saved to output_plots/base/
# Parquet files are saved to output_parquet/base/
# CSV files are saved to output_csv/base/
```

### 2. Import from Excel and run

```bash
# Self-describing format (recommended)
python execute_flextool_workflow.py \
    input.sqlite output_info.sqlite my_scenario \
    --tabular-file-path my_input.xlsx
```

### 3. Re-plot outputs from parquet files

After running the model at least once, you can regenerate plots from the saved parquet files without re-running the model:

```bash
python write_outputs.py \
    --scenario-name base \
    --read-parquet-dir output_parquet/base \
    --write-methods plot
```

### 4. Compare multiple scenarios

First, run two scenarios to populate the output database:

```bash
# Run first scenario
python execute_flextool_workflow.py \
    templates/examples.sqlite output_info.sqlite base

# Run second scenario
python execute_flextool_workflow.py \
    templates/examples.sqlite output_info.sqlite network_all_tech
```

Then compare them:

```bash
python scenario_results.py sqlite:///output_info.sqlite --dispatch-plots
```

Comparison plots are saved to `output_plot_comparisons/`.

### 5. Import and convert an old FlexTool v2 model

```bash
# Import the old-format file (creates converted.sqlite; --master adds the sensitivities)
python -m flextool.cli.cmd_read_old_flextool InputData/old_model.xlsm sqlite:///converted.sqlite \
    --master flexTool.xlsm

# Run a scenario from the converted database
python execute_flextool_workflow.py \
    converted.sqlite output_info.sqlite my_scenario
```

### 6. Export a database to Excel and re-import

```bash
# Export
python -m flextool.cli.cmd_export_to_tabular sqlite:///input.sqlite exported.xlsx

# Edit exported.xlsx in a spreadsheet editor...

# Re-import
python -m flextool.cli.cmd_read_self_describing_tabular_input \
    exported.xlsx sqlite:///input.sqlite
```

### 7. Run several scenarios in a batch

To run a set of scenarios (for example the sensitivities of
[How to run a sensitivity](how_to/basics.md#how-to-run-a-sensitivity-one-alternative-on-top-of-a-scenario)) without
the FlexTool window, loop over their names with `flextool-run`. Activate FlexTool's virtual environment first so
that `flextool-run` is found.

**Migrate the database first.** `flextool-run` refuses an input database that is not at the current FlexTool
version and names the command to run. Migrate it once before the loop:

```bash
flextool-migrate-database input.sqlite
```

or add `--migrate` to the `flextool-run` command; runs started at the same time then migrate it once.

**Bash (Linux, macOS):**

```bash
for s in coal coal_co2_price coal_fuel_price_high; do
    flextool-run sqlite:///input.sqlite sqlite:///output_info.sqlite --scenario-name "$s" --output-location results --work-folder "work/$s" --threads 2
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "Scenario $s did not finish (return code $status)"
    fi
done
```

**PowerShell (Windows):**

```powershell
foreach ($s in @("coal", "coal_co2_price", "coal_fuel_price_high")) {
    flextool-run "sqlite:///input.sqlite" "sqlite:///output_info.sqlite" --scenario-name "$s" --output-location "results" --work-folder "work\$s" --threads 2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Scenario $s did not finish (return code $LASTEXITCODE)"
    }
}
```

**Quoting.** Put each scenario name and path in double quotes: a name with spaces then stays one argument in both
shells. In a database URL, use forward slashes also on Windows, for example
`"sqlite:///C:/Users/me/flextool/input.sqlite"` (three slashes, then the drive).

**Return codes.** `0` is success, `1` an infeasible or unbounded model, `-1` another failure (bash shows it as
255). The loop goes on to the next scenario; each run's log is `_run.log` in its `output_parquet/<scenario>/`
folder, next to `_run_status.json`.

**Runs at the same time.** The loop above runs one scenario after another. To start several at once (for example
from several terminals), give each run its own `--work-folder` and about cores / runs `--threads` (see the
[`run_flextool.py` options](#run_flextoolpy-model-execution)). Leave the output database out of runs that start at the same time and compare them from
their parquet folders, as below.

**Compare the results** with `scenario_results.py`, from the output database or from the parquet folders:

```bash
python -m flextool.cli.cmd_scenario_results --parquet-base-dir results/output_parquet --alternatives coal coal_co2_price coal_fuel_price_high --write-to-xlsx
```

The workbook is written to `output_plot_comparisons/` under the parquet folder; `--excel-dir` and `--plot-dir`
choose other folders.

To check that the runs are fast enough for a course, time them with
[`flextool-time-runs`](#check-that-exercises-run-fast-enough).

## Output directory structure

After running the workflow, outputs are organized as follows:

```
<output location>/
  output_plots/<scenario>/          # PNG plot files
  output_parquet/<scenario>/        # Parquet data files
  output_csv/<scenario>/            # CSV data files
  output_excel/output_<scenario>.xlsx  # Excel workbook, one per scenario
  output_plot_comparisons/          # Cross-scenario comparison plots
  solve_data/                       # Solver progress and timing
```

The output location is the current directory, or the path given with `--output-location`. The FlexTool GUI writes into the project folder (`projects/<Name>/`), and its folder names carry the input source number (`<scenario>_<src#>`).

## Configuration

Output plots are controlled by YAML configuration files:

- `flextool/schemas/default_plots.yaml` &mdash; the bundled plot configuration, for single-scenario plots and cross-scenario comparison plots alike (each plot's `scenario_rule` decides where it is drawn)

The file defines which results to plot, plot types, layout, and styling. You can create custom configuration files and pass them via `--output-config` or `--output-config-path`. An old setting that points to `templates/default_plots.yaml` or `default_comparison_plots.yaml` uses the bundled file.

Dispatch plot colors and stacking order are defined in the project's `plot_settings.yaml` (`projects/<Name>/plot_settings.yaml`), which is created from the bundled `flextool/schemas/default_plot_settings.yaml` and can be edited via the GUI or any text editor.
