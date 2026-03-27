# Running IRENA FlexTool from the terminal

IRENA FlexTool can be run directly from the command line without Spine Toolbox or a web browser. This is useful for scripting, automation, running on servers, or when you prefer a terminal-based workflow.

## Prerequisites

- **Python 3.11+** with a virtual environment (e.g. `~/venv-spi/`)
- **HiGHS solver** binary available in the `bin/` directory (included in the repository)
- **Dependencies** installed: `pip install -r requirements.txt`
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
    base \
    --skip-input-prep
```

This skips the input preparation phase (the database already exists) and runs the model followed by output generation. Results are written to `output_plots/base/` and `output_parquet/base/`.

## Scripts overview

Root-level scripts are backward-compatible wrappers that delegate to modules under `flextool/cli/`. Both invocation styles work:

```bash
# Root-level script (backward compat)
python run_flextool.py ...

# Module invocation (preferred)
python -m flextool.cli.cmd_run_flextool ...
```

### Workflow and execution

| Script | CLI module | Purpose |
|---|---|---|
| `execute_flextool_workflow.py` | `flextool.cli.cmd_execute_flextool_workflow` | Unified entry point &mdash; runs the full workflow |
| `run_flextool.py` | `flextool.cli.cmd_run_flextool` | Runs the optimization model and writes outputs |
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

This is the recommended entry point. It orchestrates three phases:

1. **Input preparation** &mdash; convert tabular data (Excel/ODS/CSV) into a Spine database
2. **Model execution** &mdash; run the FlexTool optimization model
3. **Output generation** &mdash; process results into plots, parquet files, CSV, or Excel

Each phase can be skipped independently.

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
| `--tabular-file-path PATH` | Path to Excel/ODS input file (mutually exclusive with `--csv-directory-path`) |
| `--csv-directory-path PATH` | Path to directory containing CSV input files |
| `--output-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv` (default: `plot parquet csv`) |
| `--output-subdir DIR` | Subdirectory for output files (default: scenario name) |
| `--output-config PATH` | Path to output configuration YAML (default: `templates/default_plots.yaml`) |
| `--skip-input-prep` | Skip input preparation (assumes database already exists) |
| `--skip-model-run` | Skip model execution (assumes model has already been run) |
| `--skip-output-write` | Skip output generation |
| `--debug` | Enable debug output |

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

**From an existing Spine database** (skip input prep):

```bash
python execute_flextool_workflow.py \
    sqlite:///input.sqlite output_info.sqlite my_scenario \
    --skip-input-prep
```

## `run_flextool.py` &mdash; model execution

Runs the FlexTool optimization model directly. This is called internally by `execute_flextool_workflow.py` but can also be used standalone. After solving, it writes outputs and records scenario information in the output database.

### Usage

```
python run_flextool.py INPUT_DB_URL [OUTPUT_DB_URL] [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `INPUT_DB_URL` | Input database URL |
| `OUTPUT_DB_URL` | Output database URL for result metadata (optional) |

### Options

| Flag | Description |
|---|---|
| `--scenario-name NAME` | Scenario name to execute (if omitted, uses database filter) |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `csv` (default: `plot parquet`) |
| `--output-config PATH` | Path to output configuration YAML (default: `templates/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use |
| `--plot-rows START END` | First and last row to plot in time series |
| `--output-location DIR` | Override output location path |
| `--work-folder DIR` | Working directory for intermediate files |
| `--only-first-file-per-plot` | Only produce the first file for each plot |
| `--debug` | Enable debug output |

### Example

```bash
python run_flextool.py \
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
| `--scenario-name NAME` | Scenario with raw outputs available |
| `--read-parquet-dir DIR` | Read from existing parquet files instead of raw CSVs (faster) |
| `--config-path PATH` | Output configuration YAML (default: `templates/default_plots.yaml`) |
| `--active-configs NAME [...]` | Which plot configuration sets to use (default: `default`) |
| `--write-methods METHOD [...]` | Output formats: `plot`, `parquet`, `excel`, `db`, `csv` (default: `plot parquet csv`) |
| `--plot-rows START END` | First and last row to plot in time series (default: `0 167`) |
| `--subdir DIR` | Subdirectory for outputs (default: scenario name) |
| `--output-location DIR` | Root directory for input/output locations (default: flextool root) |
| `--plot-file-format FORMAT` | File format for plots: `png` or `svg` |
| `--only-first-file-per-plot` | Only produce the first file for each plot |
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

Upgrades a database to the latest FlexTool schema version using the templates in `version/`.

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
python execute_flextool_workflow.py \
    templates/examples.sqlite \
    output_info.sqlite \
    base \
    --skip-input-prep

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
    templates/examples.sqlite output_info.sqlite base \
    --skip-input-prep

# Run second scenario
python execute_flextool_workflow.py \
    templates/examples.sqlite output_info.sqlite network_all_tech \
    --skip-input-prep
```

Then compare them:

```bash
python scenario_results.py sqlite:///output_info.sqlite --dispatch-plots
```

Comparison plots are saved to `output_plot_comparisons/`.

### 5. Import and convert an old FlexTool v2 model

```bash
# Create a target database from the template
python -c "from flextool import initialize_database; initialize_database('version/flextool_template_master.json', 'converted.sqlite')"

# Import the old-format file
python -m flextool.cli.cmd_read_old_flextool old_model.xlsm sqlite:///converted.sqlite

# Run a scenario from the converted database
python execute_flextool_workflow.py \
    converted.sqlite output_info.sqlite my_scenario \
    --skip-input-prep
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
