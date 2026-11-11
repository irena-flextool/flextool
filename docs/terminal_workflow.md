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

The database already exists, so the input-preparation phase is skipped automatically (it runs only when `--tabular-file-path` or `--csv-directory-path` is given). Phase 2 then runs the model and writes outputs in one pass. Results land in `output_plots/base/` and `output_parquet/base/`.

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

### Installed entry points

After `pip install -e .` or `pip install .`, these commands are available directly:

| Command | Entry point |
|---|---|
| `flextool-gui` | `flextool.gui.__main__:main` |
| `flextool-read-old` | `flextool.cli.cmd_read_old_flextool:main` |

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
failed, ...; each solve's outcome; whether the parquet files there belong to this run) and `_run.log` (the run's
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
| `--output-spreadsheet PATH` | Save results to a spreadsheet file |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv`, `spinedb` (default: `plot parquet`) |
| `--results-db-url URL` | Target SpineDB for the `spinedb` write-method (default: `<output-location>/results.sqlite`) |
| `--output-config PATH` | Path to output configuration YAML (default: `templates/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--output-location DIR` | Override output location path |
| `--output-subdir NAME` | Subdirectory under `output_parquet/` etc. (default: scenario name) |
| `--flextool-location PATH` | Spine Toolbox shim: directs outputs to the FlexTool repo root rather than the work directory |
| `--work-folder DIR` | Working directory for intermediate files (default: CWD). Enables parallel scenario execution by isolating each run |
| `--only-first-file-per-plot` | Only produce the first file for each plot (quick overview mode) |
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
| `--threads N` | Threads for one FlexTool run: FlexTool's data processing and the solver. FlexTool's own processing (polars) uses at most N and at most 8 (`POLARS_MAX_THREADS` sets another limit below N); HiGHS gets `threads=N` with its parallel simplex off; a Benders solve runs at most N region solves at once. Default: the available cores, with HiGHS and Benders as before. When you run several FlexTool runs at once (for example an AI agent or a script starting them in parallel), give each about cores / runs, e.g. `flextool-run input.sqlite --scenario-name base --threads 4`. Env fallback: `FLEXTOOL_THREADS` |
| `--highs-threads N` | Number of HiGHS solver threads (default `1`). Values >1 enable HiGHS parallel mode and trade determinism for wall-clock speedup; at most `--threads` when that is given. Env fallback: `FLEXTOOL_HIGHS_THREADS` |
| `--solver-time-limit SECONDS` | HiGHS wall-clock time limit per solve, a positive number (default: none; 0 or a negative value is ignored with a warning). Env fallback: `FLEXTOOL_HIGHS_TIME_LIMIT` |
| `--solver-mip-gap GAP` | HiGHS MIP relative gap for every solve of the run. Unset: each solve's `solver_mip_gap`, else `mip_rel_gap` in `solver_config/highs.opt`. Env fallback: `FLEXTOOL_HIGHS_MIP_GAP` |
| `--solver-log-level {silent,normal,verbose}` | HiGHS log output (default `normal`). Env fallback: `FLEXTOOL_SOLVER_LOG_LEVEL` |
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
| `--config-path PATH` | Output configuration YAML (default: `templates/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv`, `spinedb` (default: `plot parquet excel`) |
| `--results-db-url URL` | Target SpineDB for the `spinedb` write-method (default: `<output-location>/results.sqlite`). Note: `spinedb` needs the native solve path, so it is skipped (with a warning) when re-rendering via `--read-parquet-dir`. |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--subdir DIR` | Subdirectory for outputs (default: scenario name) |
| `--output-location DIR` | Root directory for input/output locations (default: flextool root); overridden by `--output-locations-db-url` |
| `--plot-file-format {png,svg}` | File format for plots (default: `png`) |
| `--only-first-file-per-plot` | Only produce the first file for each plot (quick overview mode) |
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
| `--output-config-path PATH` | Comparison plot configuration YAML (default: `templates/default_comparison_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--plot-dir DIR` | Directory for comparison plots (default: `output_plot_comparisons`) |
| `--excel-dir DIR` | Directory for comparison Excel files |
| `--plot-file-format FORMAT` | File format for plots: `png` or `svg` |
| `--shared-legend` | Use shared legend across subplots |
| `--only-first-file-per-plot` | Only produce the first file for each plot |
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

Converts old-format FlexTool .xlsm files (18 fixed sheets) to the current Spine database format.

```bash
python -m flextool.cli.cmd_read_old_flextool \
    old_model.xlsm sqlite:///input.sqlite
```

| Argument | Description |
|---|---|
| `xlsm_path` | Path to the old-format FlexTool Excel file |
| `target_db_url` | Target database URL |
| `--alternative-name NAME` | Name for the alternative (default: `base`) |
| `--no-purge` | Do not purge existing data before importing |

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
python -m flextool.cli.cmd_migrate_database input.sqlite
```

| Argument | Description |
|---|---|
| `filepath` | Path to the database file (absolute or relative to flextool folder) |

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
# Create a target database from the template
python -c "from flextool import initialize_database; initialize_database('schemas/spinedb_schema.json', 'converted.sqlite')"

# Import the old-format file
python -m flextool.cli.cmd_read_old_flextool old_model.xlsm sqlite:///converted.sqlite

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

## Output directory structure

After running the workflow, outputs are organized as follows:

```
flextool/
  output_plots/<scenario>/     # PNG plot files
  output_parquet/<scenario>/   # Parquet data files
  output_csv/<scenario>/       # CSV data files
  output_excel/<scenario>/     # Excel summary files
  output_plot_comparisons/     # Cross-scenario comparison plots
  solve_data/                  # Solver progress and timing
```

## Configuration

Output plots are controlled by YAML configuration files:

- `templates/default_plots.yaml` &mdash; single-scenario plot configuration
- `templates/default_comparison_plots.yaml` &mdash; cross-scenario comparison plot configuration

These files define which results to plot, plot types, layout, and styling. You can create custom configuration files and pass them via `--output-config` or `--output-config-path`.

Dispatch plot colors and stacking order are defined in `output_plot_comparisons/config.yaml`, which is auto-generated on first run and can be edited via the GUI or any text editor.
