# FlexTool Architecture

## Project Purpose

IRENA FlexTool is an energy and power systems optimization model. It reads
input data from a Spine database, derives per-solve preprocessing tables in
polars, builds a linear/MIP program in memory via `polar_high`, solves it
with HiGHS in-process via `highspy`, and writes results to parquet (with
optional CSV, Excel, and plot renderings on top of the parquets).
Inputs can be authored directly in the Spine database, populated from
tabular input files (Excel / ODS / CSV), or imported from old FlexTool v2
`.xlsm` files. A Tkinter desktop application and a Spine Toolbox workflow
provide the two user-facing GUIs.

## Top-level data flow

```mermaid
flowchart LR
  excel["Excel / ODS / CSV<br/>+ old FlexTool .xlsm"] --> spdb
  manual["Spine DB editor"] --> spdb[("Spine input DB")]
  spdb --> backend["SpineDBBackend<br/>(EAV → polars frames)"]
  backend --> derive["input_derivation<br/>(per-solve preprocessing)"]
  derive --> provider["FlexDataProvider<br/>(in-memory transport)"]
  provider --> build["engine.build_flextool<br/>(polar_high.Problem)"]
  build --> solve["HiGHS via highspy<br/>(Problem / WarmProblem)"]
  solve --> handoff["SolveHandoff<br/>(Provider keys)"]
  handoff -->|next roll / sub-solve| derive
  solve --> outputs["process_outputs<br/>(read solution → calc_* → out_*)"]
  outputs --> parquet[("output_parquet/<scenario>/")]
  parquet --> renderers["CSV mirrors / Excel /<br/>PNG plots / GUI ResultViewer /<br/>scenario_comparison"]
```

Key invariants:

- **One backend.** `SpineDBBackend` is the only input backend. Everything
  upstream (Excel, ODS, CSV, old `.xlsm`) is converted to a Spine DB
  before the engine sees it.
- **Lazy polars, in-memory.** Frames stay as `LazyFrame` until they reach
  `polar_high`, which assembles the HiGHS matrix. Solver variables / duals
  are read back from the live `highspy.Highs` instance — no intermediate
  solver files.
- **Parquet is the canonical output.** CSV / Excel / PNG / GUI plots all
  read from the parquets the cascade just wrote. Scenario comparison loads
  multi-scenario parquet sets and emits combined renderings.

## CLI surface

```mermaid
flowchart LR
  root["Root wrapper scripts<br/>run_flextool.py, write_outputs.py,<br/>scenario_results.py, ..."] --> cli["flextool/cli/cmd_*"]
  cli --> engine["flextool/engine<br/>cascade + LP build + HiGHS solve"]
  cli --> backend["flextool/spinedb_backend"]
  cli --> derive["flextool/input_derivation"]
  cli --> decomp["flextool/decomposition"]
  cli --> inputs["flextool/process_inputs"]
  cli --> outputs["flextool/process_outputs"]
  cli --> plots["flextool/plot_outputs"]
  cli --> compare["flextool/scenario_comparison"]
  cli --> export["flextool/export_to_tabular"]
  cli --> update["flextool/update_flextool"]
  cli --> gui["flextool/gui"]
```

Each `cmd_*.py` is a thin argparse wrapper; the work lives in the linked
packages. Every CLI command also has an installed entry point in
`pyproject.toml` (the `flextool-*` names).

| CLI module | Entry point | Purpose |
|---|---|---|
| `cmd_run_flextool` | `flextool-run` | Run the cascade (`run_chain_from_db` / `run_orchestration`) plus output processing for a scenario. |
| `cmd_write_outputs` | `flextool-write-outputs` | Replay `process_outputs.write_outputs` over an existing parquet folder. |
| `cmd_scenario_results` | `flextool-scenario-results` | Combine parquets from multiple scenario folders into comparison plots / Excel. |
| `cmd_read_tabular_input` | `flextool-read-tabular-input` | Parse Excel / ODS / CSV via JSON spec → Spine DB. |
| `cmd_read_self_describing_tabular_input` | `flextool-read-self-describing-tabular-input` | Self-describing Excel (embedded metadata) → Spine DB. |
| `cmd_read_old_flextool` | `flextool-read-old` | Import old FlexTool v2 `.xlsm` files → Spine DB. |
| `cmd_read_matpower` | `flextool-read-matpower` | Import a MATPOWER case → Spine DB. |
| `cmd_import_sensitivities` | `flextool-import-sensitivities` | Import FlexTool v2 sensitivity scenarios into a version-56 import (`flextool-read-old --master` does both in one run). |
| `cmd_export_to_tabular` | `flextool-export-to-tabular` | Export Spine DB → Excel (`.xlsx`). |
| `cmd_execute_flextool_workflow` | `flextool-execute-workflow` | Two-phase workflow runner: optional input-prep (auto-triggered by `--tabular-file-path` / `--csv-directory-path`) followed by the fused model-run + output-write phase. |
| `cmd_migrate_database` | `flextool-migrate-database` | Upgrade an input DB to the current schema version. |
| `cmd_update_flextool` | `flextool-update` | `git pull` + project-side migration in one step. |
| `cmd_gui` (via `flextool.gui.__main__`) | `flextool-gui` | Launch the Tkinter desktop application. |

Two entry points are top-level modules outside `flextool/cli/` (`[tool.setuptools] py-modules`), because they
must act before `import flextool`: `flextool_run_supervisor` (`flextool-run`, see "Run record") and
`flextool_assistant_entry` (`flextool-assistant`, see "AI assistant tools").

### Runtime options

`flextool/run_options.py` holds the one resolver for every operator option that has more than one form, and
the registry of every environment variable FlexTool reads. Where a setting belongs:

- **database** — what changes the model or its answer (a solve's `scaling`, `solver_mip_gap`,
  `benders_in_out_weight`, ...);
- **CLI flag** — a choice for one run (threads, time limit, log level, `--save-memory`, `--benders-workers`);
- **environment variable** — process-level settings that must exist before import (allocator, polars build),
  developer / CI switches, and the twin of a flag for callers without a command line (Spine Toolbox, a Python
  console, tests).

The contract:

1. Several forms of one option resolve in the order **flag > environment variable > database > default**,
   through `resolve()` / `RunOptions`, one `OptionSpec` per option in `OPTIONS`.
2. Flags of registry options default to `None` (a `store_true` flag gives `True` or `None`), so an unset flag
   lets the environment variable through.
3. Nothing writes `os.environ` to carry an option. `cmd_run_flextool` builds `RunOptions.from_cli(args)` and
   passes it to `run_chain_from_db(..., run_options=)`, which stores it on `RunnerState.run_options`; a library
   caller that passes none gets `RunOptions.from_env()`, resolved once per run. The database tier is added per
   solve with `RunOptions.with_db` (e.g. `scaling`).
4. A child FlexTool process (GUI job, calibration iteration, workflow run) gets the parent's flag-given choices
   as flags (`RunOptions.to_argv()`) and inherits the parent's environment untouched. Benders region workers are
   threads and read the same `RunOptions`. The launcher label is a flag too (`--run-launcher`; the outermost one
   is kept). **One exception:** a *pipe owner* — the GUI's `_run_job` and the `flextool-run` supervisor
   (`flextool_run_supervisor.py`), which read the engine's output pipe and write `_run.log` — adds
   `FLEXTOOL_RUN_LOG_OWNER=1`, `PYTHONIOENCODING=utf-8` and `PYTHONUNBUFFERED=1` to the `env=` copy of its
   **direct** child. They describe the pipe, not a run choice. The engine reads and removes
   `FLEXTOOL_RUN_LOG_OWNER`, so no process it starts inherits it; the owner's own `os.environ` is not changed.
   The calibrator, the representative-period iteration and the workflow runner start the supervisor with
   `env=None`. **A second exception:** `FLEXTOOL_RUN_STOP_FILE` is set only by the GUI, in the `env=` of an
   auxiliary job's process (a calibration, a representative-period iteration), passed on by inheritance and read
   only by the `flextool-run` supervisor, which drops it from the engine's `env=`. It is the job's stop channel
   (see "Run record"), never a run choice.
5. Process-level variables (`MALLOC_ARENA_MAX`, `_RJEM_MALLOC_CONF`, `POLARS_*`) keep `setdefault` semantics: a
   value already in the environment wins.  The exception is `POLARS_MAX_THREADS` while `flextool/__init__.py`
   imports polars: see "Thread budget" below.
6. Solver options keep their own stack, the same for every solver (HiGHS: baseline < `highs.opt`; a commercial
   solver: `<solver>.opt`; then `solve.solver_mip_gap` < `solve.solver_arguments` < operator layer); the registry
   supplies only the operator layer (`_operator_highs_overrides`; `_solver_dispatch._operator_native_overrides`
   maps `--solver-time-limit` / `--solver-mip-gap` to the native names of `_PARAM_MAP`).
   The option files come from one folder per run, `flextool/_solver_config_source.py` (engine-free; not
   `flextool/solver_config.py`, which would shadow the template data folder):
   `resolve_solver_config_source` — explicit argument > `--solver-config-dir` > `FLEXTOOL_SOLVER_CONFIG_DIR` >
   the bounded database layout (next to the database; `<workspace>` of a `projects/<P>/input_sources/` database)
   > the Toolbox project folder > cwd > packaged templates. `snapshot_solver_option_files` reads them once at run
   start into `PathConfig.solver_option_files` (every solve reads its layer from there, never from disk) and
   writes the record `<work_folder>/solve_data/solver_config/`. The GUI and the calibrator pass the folder
   explicitly.
   Trust follows the source: the database layers (`solver_arguments`, `solver_mip_gap`) take only names in
   `engine/solve/solver_catalogue/<solver>.json` (`_solver_catalogue`, written by `tools/solver_catalogue.py` with
   a reviewed class per name: `plain` / `file` / `local_only` / `flextool`), checked for every solve before the
   first (`check_run_solver_options`) and again per solve (`check_database_solver_option`). HiGHS names the
   installed highspy knows as bool / int / double pass too. The per-user switch for unlisted names
   (`flextool/_local_solver_settings.py`: `FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`, else
   `user_config_dir()/solver_settings.json`) is never read from a database, project, workspace or flag. Option
   files and flags warn on an unlisted name and pass it.
   An option folder the run found by itself (tiers 3-5, and the GUI's `--workspace-solver-config-dir`) is trusted
   only when its parent is in `trusted_workspaces` of that settings file (`SolverConfigSource.trusted`); otherwise
   `SolverOptionFiles.untrusted_note()` is handed to `resolve_commercial_options` / `_resolve_effective_highs_options`
   and each file line goes through `check_database_solver_option(untrusted=...)`. Only `ensure_runtime_files`
   (when it creates `solver_config/` or finds only unedited templates, `holds_only_flextool_defaults`) and the
   GUI's explicit "trust this workspace" add to the list; nothing inside a workspace (`templates/project_folder.txt`
   included) is a trust anchor. `--solver-config-dir` / `FLEXTOOL_SOLVER_CONFIG_DIR` stay trusted.
7. The run header prints `RunOptions.summary_lines()` — each option set by a flag or an environment variable,
   with its source — and the thread budget (`Threads:`), with polars' and HiGHS' thread counts.

Every other variable is an `EnvKnob` in `ENV_KNOBS` with its audience (`user`, `dev`, `test`) and page:
[user page](../env_vars.md), [developer page](env_vars.md), [testing](testing.md#test-suite-environment-variables).
`tests/test_env_knob_registry.py` fails on an unregistered read, an undocumented registered variable, or a
documented variable that is no longer read.

## Repository layout

```
flextool/                          The installed Python package
├── __init__.py                    Public re-exports: write_outputs,
│                                    migrate_database, initialize_database,
│                                    update_flextool
├── cli/                           Argparse wrappers (cmd_*.py) +
│                                    cli/_timing.py (TimingRecorder)
├── spinedb_backend/               Spine DB → polars frames (the input API).
│                                    SpineDBBackend.find_entities /
│                                    parameter_values / default_values.
├── input_derivation/              Per-solve preprocessing computations
│                                    (the live successor to the retired
│                                    flextoolrunner/preprocessing).
├── engine/                        Optimization engine + cascade coordinator:
│                                    FlexData, FlexDataProvider, parameter
│                                    cascade + per-solve emitters,
│                                    build_flextool, run_orchestration /
│                                    run_chain_from_db, SolveHandoff,
│                                    auto-scaling.
├── decomposition/                 Region decomposition + region filter
│                                    (Benders sub-problems).
├── common_utils/                  Cross-package helpers (precision rounding).
├── process_inputs/                Tabular / Excel / old-format readers →
│                                    Spine DB writers.
├── process_outputs/               Solution → parquet + downstream CSV /
│                                    Excel rendering pipeline.
├── plot_outputs/                  PNG / SVG plot rendering + plot-plan
│                                    persistence (the viewer-readable
│                                    parquet plans).
├── scenario_comparison/           Multi-scenario parquet load + dispatch
│                                    & summary comparison rendering.
├── export_to_tabular/             Spine DB → Excel (.xlsx) exporter.
├── update_flextool/               DB schema migration chain, canonical-DB
│                                    generation, settings-DB initialisation,
│                                    repo self-update.
├── model_builder/                 Synthetic / mock-up scenario generator.
├── representative_periods/        Greedy convex-hull representative-period
│                                    clustering.
├── assistant/                     The AI assistant: read-only tools (registry,
│                                    input / result / diagnostics tools, docs
│                                    index), the connection server for agents,
│                                    and the chat (service backends, session,
│                                    store, terminal front end).
├── docs_build/                    Generated docs pages (formulation, results
│                                    catalogue) and the assistant's docs index.
├── gui/                           Tkinter desktop application (main_window,
│                                    result_viewer, plot_canvas,
│                                    network_graph, check_tree, dialogs/...,
│                                    assistant/ — the AI assistant's chat
│                                    window).
├── helpers/                       Small standalone diagnostic utilities
│                                    (file compare, LP coefficient inspector,
│                                    schema converters).
├── solver_config/                 Solver options templates (HiGHS + the
│                                    commercial solvers; shipped in the
│                                    wheel as *.opt.template, seeded into
│                                    the project-root solver_config/ on
│                                    first run).
├── schemas/                       Schema JSONs, YAML templates,
│                                    canonical-database JSONs, pre-v26
│                                    migration snapshots.
└── lean_parquet.py                Tiny parquet loader used by the GUI
                                     ResultViewer.

Top-level modules that run before `import flextool` (py-modules):
  flextool_run_supervisor.py (flextool-run), flextool_assistant_entry.py
  (flextool-assistant); build-time only: _flextool_build.py.

Root wrapper scripts (kept for Spine Toolbox workflow compatibility):
  run_flextool.py, write_outputs.py, scenario_results.py,
  migrate_database.py, read_tabular_input.py,
  execute_flextool_workflow.py, update_flextool.py

solver_config/                     Project-root user-editable solver options
                                     (highs.opt + gurobi/cplex/xpress/copt.opt
                                     — all gitignored; each seeded from its
                                     bundled *.opt.template on first run).

templates/                         Example databases and project assets
                                     (examples.sqlite, input templates).

migration_artifacts/               (none — the migration scratchpad was
                                     retired with the GMPL pipeline.)

tests/                             Test suite (single root):
  tests/                             — top-level unit & integration tests
  tests/engine/               — engine-internal unit tests (cluster,
                                       constraints, scaling, perturbation, ...)
  tests/spinedb_backend/             — backend unit tests
  tests/model/                       — model-level end-to-end suites
  tests/gui/                         — GUI startup tests
  tests/fixtures/*.json              — single authoritative source for all
                                       test data; sqlite DBs are built on
                                       demand
  tests/scenarios.yaml + tests/expected/  — the scenario integration suite
  tests/conftest.py                  — shared top-level fixtures
  tests/engine/conftest.py    — engine-internal fixtures
                                       (scenario_workdir, ...)

execution_tests/                   Spine Toolbox subprocess execution tests.

docs/                              MkDocs documentation site:
  docs/dev/                          — developer reference (this file,
                                       decomposition, scaling, slack convention)
  docs/*.md + assets                 — user-facing tutorials and reference.

.spinetoolbox/                     Spine Toolbox project integration.
.github/workflows/                 CI/CD (tests + verify hooks).
pyproject.toml, mkdocs.yml,        Build config, doc config, citation,
CITATION.cff, LICENSE.txt,         license, readme, release notes.
README.md, CHANGELOG.md
```

## Module details

### `flextool/spinedb_backend/` — the input API

`SpineDBBackend` is the sole materialiser between the Spine database and
the rest of the engine. It owns the EAV-to-tabular transformation: every
method returns a canonical-schema `polars.DataFrame`, and no method writes
to disk. The caller stack is:

```
SpineDB → SpineDBBackend → input_derivation → FlexDataProvider → cascade
```

Key surface:

- `SpineDBBackend.find_entities(entity_class, ...)` — entity rows.
- `SpineDBBackend.parameter_values(entity_class, parameter, ...)` —
  parameter rows with parsed values (Map / time-series / array / scalar).
- `SpineDBBackend.default_values(...)` — default-value lookups for
  cascade fallbacks.

A handful of small caching layers (parameter-value row cache, parsed-value
eviction, chunked accumulators) keep peak RSS bounded on continental-scale
inputs.

### `flextool/input_derivation/` — per-solve preprocessing

The per-solve derivation pipeline. Each module is a focused port of a
self-contained preprocessing step (commodity ladder, DC power flow, process
methods, period sets, ...). The package's `__init__.py` orchestrates the
three-step pattern:

1. Spec-driven entity / parameter spec loops over the backend.
2. Derivation helpers project the raw rows into FlexData fields.
3. Validators sanity-check shape / dtype invariants before handoff.

Output is an in-memory `FlexData` augmented with derived fields, exposed to
the engine through `FlexDataProvider`.

### `flextool/engine/` — the optimization engine

The dominant package. Builds a `polar_high.Problem` from `FlexData`, solves
it with HiGHS, and threads cross-solve state through `SolveHandoff` (in
memory; CSVs only emitted under `csv_dump=True` for debugging).

#### Public surface

Exported from `flextool.engine` (`__init__.py`):

| Symbol | Purpose |
|---|---|
| `FlexData` | Input dataclass — polars frames + `polar_high.Param`s. Sets unprefixed, parameters `p_*`. Full inventory in `_param_shapes.py`. |
| `FlexDataProvider` | In-memory transport surface (`preprocess/_flex_data_provider.py`; not re-exported). Keys are declared in `_provider_keys.py` (no `.csv` suffix). All cross-phase data moves through it. |
| `load_flextool(source, *, provider, db_reader)` | Build `FlexData` for one sub-solve from its `FlexDataProvider` + `SpineDbReader`; `source` (the work folder) only forms the provider keys. Never reads the work folder. |
| `build_flextool(m, d, ...)` | Feature-conditional LP/MIP construction on a `Problem` / `WarmProblem`. |
| `run_chain_from_db(...)` / `run_chain(...)` | Run a DB scenario end to end, threading `SolveHandoff` between sub-solves (`run_chain` is an older wrapper returning `ChainStep`s). |
| `run_orchestration(state, work_folder, ...)` | Native cascade coordinator: reads `RunnerState`, drives the rolling / nested / stochastic cascade, returns one `OrchestrationStep` per solve. |
| `SolveHandoff` | Cross-solve carrier (invest / divest / storage / roll-state / CO2 ladder / commodity ladder / cumulative sim-hours / history sets). |
| `SpineDbReader` / `InMemoryReader` / `CsvSource` / `FlexInputSource` / `InputSource` | Input adapters — production Spine DB, in-memory test fixture, or the work-folder path wrapper (`CsvSource`: the loader takes its directories as key paths and reads the provider). |

#### Solve modes

Same `build_flextool` body, three wrappers from `polar_high`:

- **Monolithic** — `polar_high.Problem`. One LP per call.
- **Warm-start cascade** — `polar_high.WarmProblem`. Reuses the built LP
  when the next sub-solve is identical (every `FlexData` field, labels
  included, and the solver configuration; `_reuse_blocker` in
  `solve/_warm.py`); rolls are rebuilt, because the output writer and the
  handoff read the LP by its labels. The future rolling-overlap work (adopt
  the new labels on the live LP and update only changing coefficients, the
  "rebind" feature item) would live here.
- **Benders (regional)** — `engine.solve._benders.solve_benders` driven by
  the orchestrator. Slices `FlexData` by region via `_region_filter`, then
  runs a **Benders decomposition**: a coordinating *master* (a persistent
  `polar_high.WarmProblem`) holds the inter-regional trade flows and the
  trade-connection investment plus one recourse variable `η_r` per region;
  each region is an operational *subproblem* solved with the master's trade
  pinned, returning duals that the master accumulates as optimality
  **cuts** (`WarmProblem.add_cut_row` / `add_recourse_col`). The master
  objective is a **valid lower bound** and the loop converges on the
  optimality gap. This replaces the earlier dual-subgradient scheme, which
  collapsed greenfield cross-region trade to autarky with an *invalid*
  bound. See [decomposition.md](decomposition.md).

#### Feature flags in `build_flextool`

`build_flextool` is a table of contents: `_prepare` → `_declare_variables`
→ `_shared_expressions` → one `_eq_<family>(b)` per core equation family
(plus the feature modules' `add_constraints(b)`) → `_obj_<family>(b)` /
`objective_terms(b)` summed into the objective. Optional features with
their own variables live in their own modules (`_online`, `_unit_floors`,
`_reserve`, `_group_slack`, `_cumulative_invest`, `_commodity_ladder`,
`_dc_power_flow`, `_delay`, `_non_anticipativity`). The `model/core.py` module
docstring maps every variable and equation family; details in
[engine.md](engine.md#lp-build-modelpy-and-the-feature-modules).

`_prepare` sets one `has_*` flag per feature on the `_Build` context from
its switch field on `FlexData`; each family function returns early when
its flag is off. Required-field lists are module-level tuples in
`model/core.py`:

| Flag | Switch field | Required tuple | Adds |
|---|---|---|---|
| (always) | — | `ALWAYS` | `vq_state_up/down`, `nodeBalance_eq`, balance slack penalty. |
| `has_proc` | `process_source_sink` | `PROCESSES` | `v_flow`, flow terms of `nodeBalance_eq`, `maxFlow`, `maxOutputSum` (indirect units), commodity cost. |
| `has_indirect` | `process_indirect` | `INDIRECT` | `conversion_indirect`, `minInputShare`. |
| `has_co2_price` | `flow_from_co2_priced` | — (warns and skips without price / content) | CO2 price term in the objective. |
| `has_co2_cap` / `has_co2_cap_total` | `flow_from_co2_capped[_noEff]` / `flow_from_co2_capped_total[_noEff]` | `CO2_CAP` / `CO2_CAP_TOTAL` | `co2_max_period` / `co2_max_total`. |
| `has_user_cstr` | `cdt_eq` / `cdt_le` / `cdt_ge` | `USER_CSTR` | `process_constraint_{equal,less_than,greater_than}[_period\|_period_total]` (`_period`: one row per period for constraints without flow or state terms; `_period_total`: one annualised row per period for `constraint_period_method` *period_total*, `constraint_period_total`). |
| `has_profile` | `process_profile_{upper,lower,fixed}` | `PROFILES` | `profile_flow_*` bounds on `v_flow`. |
| `has_storage` | `nodeState` | `STORAGE` | `v_state`, state change in `nodeBalance_eq`, `maxState`, start binding, state profiles, reference value. |
| `has_rp` | `nodeState_rp` | — | Representative-period inter-period state coupling (`_eq_rp_storage_chain`). |
| `has_nsb` | `nodeStateBlock` | — | `bind_intraperiod_blocks` nodes: `stateConstantWithinBlock_eq`, `nodeBalanceBlock_eq`. |
| `has_coarse` | `nodeCoarse` + `node_coarse_step` | — | Nodes of coarse `new_stepduration` groups: `nodeBalanceCoarse_eq` (the summed `nodeBalance_eq` terms per coarse step), `stateConstantCoarse_eq`. |
| `has_online_{lin,int}` | `process_online_{linear,integer}` | `ONLINE` | `_online`: `v_online`, startup / shutdown, uptime / downtime; `_unit_floors`: min-load floors. |
| `has_minload_eff` | `process_min_load_eff` | `MINLOAD_EFF` | min_load_efficiency section terms. |
| `has_startup_cost_{lin,int}` | `pdt_online_{linear,integer}` | `STARTUP_COST_*` | Startup-cost objective term. |
| `has_ramp` | any `process_source_sink_ramp_limit_*` | `RAMP` | Ramp-up / ramp-down constraints. |
| `has_{invest,divest}_{p,n}` | `pd_{invest,divest}_set` / `nd_{invest,divest}_set` | `INVEST` / `DIVEST` | `v_invest` / `v_divest`, investment maxima, annuities. |
| `has_reserve` | `reserve_upDown_group` | checked in `_reserve` | `_reserve`. |
| `has_capacity_margin` / `has_inertia` / `has_non_sync` | `_group_slack.has_*` | — | `_group_slack`. |
| `has_cumulative_invest` | `_cumulative_invest.has_feature` | — | `_cumulative_invest`. |
| `has_ladder` | `_commodity_ladder.has_feature` | — | `_commodity_ladder`. |

A solve's `reduce_unit_detail_method` / `reduce_system_detail_method` turns flags off by clearing their switch
fields before the build (`preprocess._reduce_detail`, see [engine.md](engine.md#solve-level-detail-reduction)):
`online*` clears the `has_online_*`, `has_minload_eff` and `has_startup_cost_*` switches, `*ramps` the `has_ramp`
switch and the ramp-cost set, `*min_limits` the min-load and min-time sets, `reserves` the `has_reserve` gate,
`inertia` / `non_synchronous` the group sets behind `has_inertia` / `has_non_sync`, and `dc_power_flow` the DC
sets. `integers*` keeps the online flags and sets `relax_integrality`, which `_online` and `_exact` read as
`integer=not d.relax_integrality`. The capacity margin is never cleared.

The DC power-flow, exact-connection, delay and non-anticipativity modules
gate on their own data (`_dc_power_flow.has_feature`, `_exact.has_feature`
(`process_exact`), `_delay.has_feature`, `period_branch_full` /
`pd_non_anticipativity`).

If a switch is set but a required field is missing, `build_flextool` raises
`ValueError` rather than silently degrading.

#### Block-COO dense-axis contract

`_prepare` (the first step of `build_flextool`) declares FlexTool's dense
trailing axes once, via
`m.declare_dense_axes(("d", "t"))` — `(period, timestep)`. This is an
unconditional call requiring **polar-high>=2.4.0** (the interim `hasattr`
capability guard has been dropped; the pin lives in `pyproject.toml`). With the
axes declared, polar-high's canonical-matrix build slices the pre-sorted dense
axis as contiguous numpy blocks and multiplies coefficient factors with ufuncs
(the block-COO slice path) instead of wide relational joins. `Sum`-wrapped
chains — FlexTool's `Sum`-heavy families (nodeBalance, reserves, co2) — evaluate
in one pass via a captured `SumBlockMeta` recipe with a relabel fast-path. The
block-COO output is bit-identical to the legacy polars path; the win is purely
the peak-memory and wall-time of the build.

#### Cross-solve state (Provider keys + `SolveHandoff`)

Inside the cascade everything moves through `FlexDataProvider`. The
relevant key families are declared in `engine/preprocess/_provider_keys.py`:

- `HANDOFF_*` — quantities the just-solved problem produces for the next
  iteration (`fix_storage_quantity/price/usage`, `roll_end_state`,
  `realized_invest`, `realized_existing`, `divest_cumulative`,
  `cumulative_co2`, `commodity_ladder_cumulative`, `cum_sim_hours`,
  `ed_history_realized_first`, `edd_history`).
- `INPUT_*` — backend-derived rows the cascade re-reads each iteration.
- `SOLVE_DATA_*` — per-solve sets and parameters that the derivation
  layer produces.

The Provider is the contract. Disk-side CSV mirrors in `input/` + `solve_data/`
exist only when `csv_dump=True` (a debug toggle; one folder, shared by every
sub-solve; a Benders solve dumps its undecomposed FlexData) and are not load-bearing for any downstream phase.

#### Output writers read only memory; one run per work folder

The output cascade (`flextool/process_outputs/`) has the same contract as the loader: every writer sources its
values from the sub-solve's `FlexData`, `FlexDataProvider` or `OutputWriterState`, never from `input/` or
`solve_data/` on the work folder (item 163, `specs/bugfix/c24/design_163.md`;
`test_output_writers_read_no_work_folder_inputs` in `tests/engine/test_meta_provider_invariants.py`, Rule 5). The
five handoff-mirror CSVs under `solve_data/` (`p_entity_divested`, `fix_storage_quantity`, `fix_storage_usage`,
`p_roll_continue_state`, `p_entity_period_existing_capacity`) are write-only `--csv-dump` debug output with no
reader anywhere in the engine; `period_capacity.csv` is the one mirror still written on every run (vestigial,
always header-only).

Because nothing reads the work folder back, **FlexTool supports exactly one run per work folder at a time**: a
second concurrent cascade into the same folder is not supported (its parquets would interleave with the first's,
and whichever run finishes first deletes the other's stash). `run_orchestration`'s `_bootstrap_dirs` clears the
*previous* run's own leftovers at the start of every cascade — `output_raw/` (`rmtree` + recreate: the per-roll
parquet stash, `_solve_order.txt`, `manifest.json`) and the `solve_data/` handoff-mirror basenames
(`handoff_writers.MIRROR_FILES`) — so a reused folder (the CLI-in-cwd rerun case, e.g. after a failed run or a
`--csv-dump` run) cannot leak a prior run's output into this one
(`union_realized_slice.has_persisted_slices` would otherwise union stale per-roll parquets into the fresh run).
`input/` and every other `solve_data/*` file (dump / debug output) are left alone. A symlinked `output_raw/` is
refused rather than followed. The GUI and Spine Toolbox already give each job / execution a fresh work folder, so
this clearing is a no-op there; it is the fix for the CLI case.

#### CSV dump (supported) and CSV reload (unsupported)

`--csv-dump` / the GUI "Full" debug level, `FlexData.dump_csvs` and
`FlexDataProvider.snapshot_processed_inputs` write a CSV dump for inspection. Both writers stamp the folder with
`_dump_meta.json` (`dump_format`, FlexTool version, schema version, git hash; `preprocess/_dump_stamp.py`). The
writers are supported; nothing in a run reads a dump back. The main loader is provider-only: it never reads `input/` or
`solve_data/` on disk, so a dump or stale CSVs in a reused work folder cannot change a run
(`test_loader_and_model_have_no_disk_arms` in `tests/engine/test_meta_provider_invariants.py`).

Reloading a model from a dump lives only in `flextool/engine/_csv_reload/`: `load_from_dump(dump_dir, *, db_url=None,
scenario=None, force=False)`, `copy_runner_meta`, and the offline output writer (`python -m
flextool.engine._csv_reload.offline_outputs`, outputs from an MPS + solution file). It is **unsupported**: it is not
kept in sync with the engine, it warns on every call, and it refuses a dump without a stamp or with another dump
format or schema version unless `force=True`. A reloaded model has no solve-to-solve carry and can differ from the
solved model. The stamp does not detect leftover CSVs from earlier runs in a reused folder (the CLI work folder
defaults to the current directory and is never cleaned), so reload only from a folder that holds a single dump. Nothing outside the package imports it (`test_csv_reload_is_a_leaf`), and its tests in
`tests/csv_reload/` run only with `pytest -m csv_reload`. Do not update the package when you change the engine;
whoever needs the reload brings it up to date first (and then updates `VERIFIED_FLEXTOOL` / `VERIFIED_SCHEMA` in
`_csv_reload/_stamp.py`).

The offline output writer replaces the removed `python -m flextool.process_outputs.read_highs_solution` and needs
`--dump`. It reads the solve's `scale_the_objective` from the dump's `solve_data/scale_the_objective.csv`: the
sub-solve provider holds the scale its LP was built with, so a provider snapshot (`--csv-dump`, a run's work folder)
carries it. `--scale` is an optional override (an MPS built at another scale); a missing or header-only file is an
error, never a silent default.

#### Package map

`flextool/engine/README.md` is the entry point: public API, the pipeline
step by step, and one line per folder. In short:

```
__init__.py           Public API (table above)
input/                Database and in-memory readers, SolveConfig, timeline,
                      RunnerState, parameter shapes (_param_shapes.py), constants
preprocess/           FlexData + load_flextool (flex_data.py), FlexDataProvider
                      and its keys, input writer, per-solve context
preprocess/params/    Parameter cascade: direct, projected, derived (NPV,
                      walks, profiles, blocks, branches, existing capacity)
preprocess/emit/      Per-solve sets and parameters written into the Provider
model/                The LP: core.py (build_flextool, table of contents +
                      core equations) and one module per optional feature
solve/                Orchestration (_orchestration.py), sub-solve expansion,
                      solver dispatch, warm start, Benders, SolveHandoff
output/               Solution → output parquet bundle; CSV dumps
_csv_reload/           Unsupported, opt-in CSV-dump reload + offline outputs (see "CSV dump … reload")
autoscale/            Auto-scaling pipeline + YAML audit report
```

Imports point down the pipeline (solve → model → preprocess → input); the
README lists the known exceptions.

#### Parameter shapes & `(d,t)` broadcasting

`_param_shapes.py` is the authoritative resolver for how a DB-authored
parameter maps onto the model axes. **Never detect a parameter's
period / time axis by column name.** Spine writes a Map's index column
under the author's `index_name`, defaulting to the silent `"x"` when
unset — so a column-name check like `("t","time","step")` misclassifies
a time series as a scalar and broadcasts it with a `how="cross"`
Cartesian product against the full `(d,t)` grid. On long timelines that
overflows polars' 2³² row index; on short ones it silently flattens the
time variation via a downstream `unique([...])`. Both bit us (the
`p_slope` blow-up). Route every param→`(d,t)` broadcast through
`resolve_param_shape` (DB-metadata shape + value-domain probing) →
`broadcast_to_period_time` (keeps the Param at its authored shape, no
cross-product) → `promote_param_to_dt` (narrowing inner-join on the
carried axis). A new parameter that needs this must be added to
`PARAM_ALLOWED_SHAPES`.

##### Silent-label Maps are resolved per entity, not per parameter

When the DB carries no explicit `index_name` on a Map (`"x"` or empty) and the registry admits both a
period and a time shape at that depth, `resolve_param_shape` first tries to disambiguate the *whole*
parameter at once, by value-domain probing the union of every entity's keys against the solve's
`(d, t)`. When that union probe stays ambiguous — no entity's keys overlap the solve's periods or
timesteps at all, or different entities' keys overlap different axes — the resolver does **not** fall
back to dropping the whole parameter. Instead each silent entity is classified on its own
(`_silent_entity_axes`): an entity never decides another entity's shape. An entity with no key in the
active solve has no value in that solve — it is removed from the frame the same way an entity that
authored nothing at all would be (so the parameter's own default applies, whatever that is for the
consuming producer); when *every* entity is removed this way, the frame falls through to the source's
own default-broadcast path (a scalar default spread over every entity, or nothing when there is none).
An entity whose keys match both axes raises, naming it — the same rule a single ambiguous row inside
one entity's own Map already raises on. See `tests/engine/test_param_shapes_silent_per_entity.py` and
`tests/engine/test_silent_label_params.py` (items #167 / #168 of `specs/github_issues.md`).

##### Every entity's index path is checked; unnamed maps of different depths are read per depth

Before any shape is resolved, `_check_entity_label_paths` checks each entity's own raw label path
(`parameter_entity_label_paths`), not only the parameter's combined one: a depth that is not the
`shape_rank` of an allowed shape, or an explicit name that no allowed shape of that depth has on that level
(`period` / `time`; for the price ladders, level 1 of a 3-level ladder is the period and the tier and
price / quantity levels may not be named `period` / `time`), raises `FlexToolConfigError` in plain words
(`allowed_shapes_in_words`): *"commodity 'coal', parameter 'price': the value has 2 index levels; this
parameter takes a constant, or a map over periods or over time (1 level)"*. A depth that stays ambiguous
between allowed shapes is accepted; value-domain probing decides it. `shape_rank(shape)` =
`len(_SHAPE_LABELS[shape])` is the one source of a shape's depth (also the schema's map rank,
`tests/engine/test_schema_shape_consistency.py`).

When a non-ladder parameter's entities have maps of more than one depth and some level is unnamed, the
reader's single positional unroll cannot place them (it puts every row into the deepest map's columns), so
`_resolve_by_depth` resolves each depth group on a `_view_without` the others (as `resolve_facet_groups`
does for ladders) with the usual rules, and unions the groups into one `[entities, period, t, value]` frame
(`_axis_frame_resolved`). It raises only where those rules cannot place a group: keys on both axes, or an
ambiguous group without a `period_filter`. With every level named, the reader places rows by label and no
split happens. When an unnamed map's shape was read from its depth or keys, `resolve_param_shape` prints one
note per (class, parameter) and run (`_behaviour_changes.warn_once`, `print(flush=True)`); ladders are left
out (their levels are placed by depth by definition).

`_param_shapes` is the **I/O contract only** — it validates user-authored
DB shapes and errors to the user on mismatch. Model-internal axes (e.g.
the rolling-solve `roll` axis) are deliberately excluded; they are never
DB-authored, never reach the resolver, and a stray one already fails
loudly via `_UnrecognisedIndex`. Document such internal axes in
`schemas/flextool_axis_contract.json`, not in the resolver's allow-lists.

##### Solve-step aggregation (`new_stepduration` of a solve)

A solve with `new_stepduration` combines the time steps of its timeline into longer steps, each labelled by
its first original (base) step. Every time series then needs one value per new step, and the rule is a
property of the parameter: `TIME_AGGREGATION` in `engine/input/_solve_step_aggregation.py` maps each
`(entity_class, parameter)` that can vary in time to `MEAN` (duration-weighted mean: prices, costs, limits,
availability, min load, penalties, reserves, self-discharge, weights, profiles, instant-flow limits),
`HARMONIC` (mean of 1/v: `efficiency`, `efficiency_at_min_load`, so the fuel use of a constant flow is
exact), `SUM` (`inflow`, `energy_margin_adder`: energy per base step) or `LAST` (`storage_state_reference_value`:
the state at the step's end). `NOT_TIME_SERIES` lists the schema's other map parameters; every map
parameter of the schema is in exactly one of the two (`tests/engine/test_time_aggregation_registry.py`).
**A new time-series parameter needs a `TIME_AGGREGATION` entry**; while a solve's steps are active, a
broadcast of a key without one raises.

One vectorised function, `aggregate_to_solve_steps`, does the arithmetic (a step whose base values are all
equal keeps the value itself; the means are exact, only the profile keeps its earlier 6-decimal rounding
(`ROUND6_KEYS`); `SUM` is `math.fsum`, so n equal values give n·v like a constant; `LAST` follows the timeline
position); `combine` is its per-step reference (`test_vectorised_kernel_equals_combine`). A roll aggregates
only the base steps of its own solve steps (`aggregate_if_active(steps=...)`). It serves three places:

- the provider's frames: `TimelineConfig.create_averaged_timeseries` writes `solve_data/pt_*`, `pbt_*`,
  `pdt_process`, `pdt_node`, `pdt_energy_margin_adder` and (verbatim) `pd_energy_margin_adder` (kinds by
  (file, parameter)) and
  `solve_data/new_step_base_steps` (base step, new step, count, duration, position); without
  `new_stepduration` the `input/` frames pass through verbatim. The per-solve emit folds read these
  `solve_data/` frames, never `input/` (see docs/dev/engine.md "Vectorized per-roll emit"), and
  `_run_model` mirrors every key the aggregation wrote onto the sub-solve provider;
- the DB-direct readers: `broadcast_to_period_time` aggregates the rows that carry `t`, keyed by
  `ResolvedShape.key`, after the null filter and before `filter_zero` and the axis cast;
- the readers outside the resolver (the period+time profile reader, `timeset_weights`).

`load_flextool` arms the base steps in a contextvar (`activate_solve_step_aggregation`) next to the
stochastic-time source; with no `new_stepduration` the frame is empty and nothing is aggregated, so the
LP of every uncoarsened solve is unchanged. The hook does not break the rule above: `t` stays the I/O axis
and only the values are grouped. The `SUM` kinds are scaled by their own readers (a constant counts once per
base step: the inflow scaling, the energy-margin adder), so a `SUM` key reaching the generic broadcast
raises.

##### Silent-default `"x"` index — readers outside the resolver

Not every DB read goes through `resolve_param_shape`. The **per-entity
cascades** in `_derived.py` / `_existing.py` /
`_npv.py` read raw entity params (`source.parameter(...)`) and
must detect the period axis themselves. The failure mode is identical to
the resolver's: Spine names a Map's index column with the author's
`index_name`, whose spinedb_api *default is the literal string `"x"`*
(not empty — real 2-D maps carry `index_name='x'`, see
`_spinedb_reader._discover_index_cols`). So a `if "period" in cols`
gate misfires on the common `"x"`-indexed map and the code either drops
the row or collapses it via a downstream `.unique(subset=["e"],
keep="last")` — for a *retiring* `existing` map (nameplate → 0 at expiry)
that keeps the last period's `0`, defaulting `unitsize` to `1000` and
turning `existing_count = capacity / unitsize` into a spurious fraction
(caps continuous online below 1; forbids integer online entirely).

**Safe idioms** (use one of these for any raw per-entity map read):

- period-agnostic aggregate — `group_by("e").agg(pl.col("value").max())`
  (used by `_entity_unitsize_lf`: the cascade only needs "non-zero", and
  the per-period MAX is non-zero iff any period is);
- explicit dual check — `period_col = "period" if "period" in cols else
  "x"` / `for cand in ("period","x")` (used by
  `_ed_explicit_period_param`, `_per_solve_sets`);
- per-row scalar detection — treat any single non-`name`/`value` column
  as the period dim and classify scalar-vs-map by that column's
  *per-row* null-ness, so a constant value sharing a class with period-Map
  siblings (surfaces as `x = null`) broadcasts instead of zeroing (used by
  `_per_entity_param_lf` in both the `existing` and `npv` mirrors).

**Audited state** (see the coverage test
`tests/engine/test_silent_default_index_pdt.py`):

- *Safe* — `availability`, `inflow`, `other_operational_cost`,
  `reservation` (resolver-routed); the `existing`/`lifetime`/
  `virtual_unitsize` per-period chains (`_per_entity_param_lf`,
  hardened); `_ed_explicit_period_param`, `_per_solve_sets`; `profile`
  in real solves (reads the engine-written canonical `pt_profile.csv` /
  `pbt_profile`, not the raw Map).
- *Profile tiers (item 73)* — `_profiles._classify_profile_rows` reads a
  1-D profile Map without an index name by its keys against the solve's
  `_param_shapes._period_time_vocab(dt)` (`_silent_map_tier`): periods
  only → period tier (`_profile_period_only_lf` reads the `x` column as
  the period), else time tier; the `InMemoryReader` column fallback uses
  the same rule. Unlike the resolver's silent drop of an entity whose keys
  miss the solve (`_silent_entity_axes`), the run is told:
  `validate_period_keys` warns once per run about a profile period map
  that misses periods of the scenario's solves and stops on keys that
  are periods and time steps of the model. `_profile_time_lf`'s raw
  fallback also reads `x`. A 2-D Map whose outer level has no index name
  is `period_time` by the same rule on its outer keys (item 223;
  `_profile_period_time_lf` reads `x` as the period and `x_2` / `t` as the
  time step, the column fallback likewise; under an outer level labelled
  `period` the reader names the unnamed inner level `x`, read as the time
  step on the rows whose `period` is set); other unnamed 2-D Maps stay
  `stochastic` (only 3-level Maps fill `pbt_profile`), and
  `validate_period_keys` stops the run on them (outer keys that are time
  steps, both kinds, or no period). Residual: the CSV `pdtProfile` seed
  has no period tier at all (item 222).
- *Guarded by schema* — `invest_max_total` and `virtual_unitsize` are
  declared `float`-only in `parameter_types` (no `map` row), so the
  `keep="last"` collapse in `_e_explicit_param` and in the
  `virtual_unitsize` sub-branch of `_entity_unitsize_lf` can only bite if
  a param is authored against its declared type. Value-type enforcement
  is not yet strict, so these are latent-until-misauthored, not safe by
  construction.
- *Separate gap* — `min_load` is declared map-capable (`map,1` / `map,3`)
  but `p_min_load_from_source` reads it as a scalar `Param(("p",))`, and
  `minFlow_minload` consumes only that scalar. A period/time-varying
  `min_load` is therefore flattened regardless of the index name — an
  engine "consumes scalar only" gap, not just an `"x"`-detection one.
  Route through the resolver if per-period `min_load` is ever wired.

### `flextool/process_inputs/` — tabular → Spine DB

Reads tabular data and writes Spine DB. The output of this package is the
Spine DB the engine consumes.

| File | Purpose |
|---|---|
| `read_tabular_with_specification.py` (`TabularReader`) | JSON-spec-driven reader for Excel / ODS / CSV. |
| `read_self_describing_excel.py` | Self-describing Excel with embedded metadata. |
| `read_old_flextool.py` | Parse old FlexTool v2 `.xlsm` → `OldFlexToolData`. |
| `read_matpower.py` | MATPOWER case → import structure. |
| `write_to_input_db.py` (`write_to_flextool_input_db`) | Tabular → Spine DB. |
| `write_self_describing_to_db.py` | Self-describing Excel → Spine DB. |
| `write_old_flextool_to_db.py` | Old FlexTool data → Spine DB. |
| `import_old_excel_input.json` / `import_excel_input.json` | Excel import specifications. |

### `flextool/process_outputs/` — solution → parquet → downstream

Three-layer architecture:

```
Solution extraction (in-memory, from highspy.Highs):
  read_highs_solution.py         VARIABLE_SPECS — variables, row duals,
                                 column duals harvested directly from
                                 the live solver instance.

Post-processing (parquet-backed calculations):
  drop_levels.py                 Strip 'solve' level from time-indexed objects.
  calc_capacity_flows.py         Capacity, online status, flow_dt, ramps.
  calc_connections.py            Connection flows and losses.
  calc_storage_vre.py            Storage state changes and VRE potential.
  calc_slacks.py                 Reserve, slack, inertia quantities.
  calc_costs.py                  All cost aggregates.
  calc_group_flows.py            Group-level flow aggregations.
  process_results.py             Coordinator: calls calc_* in order.
  handoff_writers.py             SolveHandoff CSV mirrors (csv_dump only).
  solve_order.py                 Solve-sequence reconstruction.

Output formatting:
  out_capacity.py                Unit / connection / node capacity tables.
  out_flows.py                   Unit flow, capacity-factor, VRE, ramp.
  out_group.py / out_flowgroup.py NodeGroup flow, inflow, VRE-share, indicators.
  out_node.py                    Node summary and additional-results outputs.
  out_costs.py                   Cost summary, CO2, generic outputs.
  out_ancillary.py               Connection, reserve, inertia, slack, dual.
  write_outputs.py               ALL_OUTPUTS list + orchestrator.
  to_spine_db.py                 Result DataFrames → Spine output DB.
```

The cascade produces `output_parquet/<scenario>/`. CSV / Excel / plot
renderers all consume those parquets — no separate solver-output
extraction step.

### `flextool/plot_outputs/` — PNG / SVG plots and plot plans

Static plots (lines, bars, stacked area) plus a persistence layer that
saves precomputed *plot plans* (one parquet per result key) so the GUI's
`ResultViewer` can re-render without recomputing pivots / pagination.

| File | Purpose |
|---|---|
| `orchestrator.py` | `plot_dict_of_dataframes`, `prepare_plot_data`, `compute_all_plot_plans`. |
| `config.py` | `PlotConfig`, `DIMENSION_RULES`, `PLOT_FIELD_NAMES`. |
| `plan.py` | `compute_plot_plans_for_result`, `save_plot_plan`, `load_plot_plan` (with `schema_version` invalidation). |
| `plot_lines.py` | Line and stacked-area rendering. |
| `plot_bars.py` / `plot_bars_detail.py` | Bar chart orchestration and rendering. |
| `axis_helpers.py`, `legend_helpers.py`, `subplot_helpers.py`, `format_helpers.py` | Layout, formatting, filename generation. |
| `color_template.py` | Stable color assignment across scenarios. |
| `shared_manifest.py`, `plan_union.py` | Per-scenario plan availability + cross-scenario union. |
| `perf.py` | Internal timing. |

### `flextool/scenario_comparison/` — multi-scenario analysis

Loads parquets from multiple scenario folders, builds combined views, and
renders comparison plots.

| File | Purpose |
|---|---|
| `data_models.py` | `TimeSeriesResults`, `DispatchMappings`. |
| `db_reader.py` | Load scenario parquet files into `TimeSeriesResults`. |
| `dispatch_mappings.py` | Load dispatch-mapping parquets. |
| `config_builder.py` | Discover dispatch entity/scenario names + assign palette colors (seeded into `plot_settings.yaml`). |
| `dispatch_data.py` / `dispatch_plots.py` | Per-scenario dispatch DataFrame prep and stacked-area rendering. |
| `constants.py` | Colors and special column-name lists. |
| `orchestrator.py` | Top-level `run()` function. |

### `flextool/export_to_tabular/` — Spine DB → Excel

Exports a Spine input DB to `.xlsx` in self-describing v2 or original v1.

| File | Purpose |
|---|---|
| `export_to_excel.py` | Orchestrator; v1 vs v2 format selection. |
| `db_reader.py` | `DatabaseContents` + `read_database()`. |
| `sheet_config.py` | `SheetSpec`, `build_sheet_specs()`. |
| `excel_writer.py` | Sheet writers (periodic, time-series, ...). |
| `formatting.py` | Cell formatting / type conversion. |
| `export_settings.yaml` | Export configuration. |

### `flextool/gui/` — Tkinter desktop application

Spawns CLI commands as subprocesses; never imports the engine directly.

```
__main__.py                      Entry point → MainWindow.
main_window.py                   MainWindow (tkinter.Tk root).
project_utils.py                 Create / list / rename projects.
settings_io.py                   Load / save GlobalSettings + ProjectSettings.
data_models.py                   Settings + ScenarioInfo dataclasses.
input_sources.py                 InputSourceManager (file selection).
scenario_lists.py                Available / executed scenario managers.
execution_manager.py             ExecutionManager (threaded subprocess pool).
execution_window.py              ExecutionWindow (progress).
output_actions.py                Plot / export action wiring.
output_log_window.py             Log display.
db_editor_integration.py         Spine DB editor launcher.
db_version_check.py              Input-DB version validation + auto-upgrade.
error_handling.py                safe_callback decorator.
platform_utils.py                Platform-specific file/app opening.
result_viewer.py                 ResultViewer (parquet → plot canvas).
plot_canvas.py                   Matplotlib canvas with downsampling.
plot_cache.py                    Plot cache.
plot_config_reader.py            Plot-config YAML parser.
network_graph.py                 Network-topology figure builder.
check_tree.py                    CheckTreeController.
hover_tooltip.py / tree_reorder.py / downsampling.py / cli_format.py / scenario_key.py / config_parser.py
dialogs/                         Add / project / plot / file-picker dialogs.
```

### `flextool/update_flextool/` — migration and updates

| File | Purpose |
|---|---|
| `self_update.py` (`update_flextool`) | `git pull` + project-side migration. |
| `db_migration.py` (`migrate_database`) | Walk the version chain to the current `FLEXTOOL_DB_VERSION`. |
| `initialize_database.py` (`initialize_database`) | Create new DB from JSON template. |
| `ensure_settings_db.py` | Idempotent settings-DB creation. |
| `test_fixtures.py` | Migrate `tests/fixtures/*.json` to current schema and sync their schema sections. |
| `fixture_schema.py` | Schema sections of fixture / canonical JSONs generated from `spinedb_schema.json`; sync rules and drift listing. |
| `canonical_databases.py` | Materialise / migrate canonical example DBs from JSON. |
| `generate_canonical.py` | Project authoritative fixtures into derived canonical DBs. |
| `extend_tests_fixture.py` | Agent-friendly YAML-delta workflow for append-only fixture additions. |

CI-enforced verify chain (`tests.yml::template-check`):

- `sync_master_json_template --verify` — schema parity.
- `test_fixtures verify` — `tests/fixtures/*.json` at current schema.
- `generate_canonical --verify` — generated canonical files in sync.
- `canonical_databases verify` — authoritative canonical files at current schema.

### `flextool/decomposition/` — Benders region helpers

`region_decomposition.py` and `region_filter.py` slice `FlexData` by region
group and prepare sub-problems for the Benders solve mode (the filter-only
`--region` CLI entry point that materialises one region's standalone input
directory). The in-engine slicer used by the live Benders driver lives at
`engine/solve/_region_filter.py`; the master + cut loop is in
`engine/solve/_benders.py`.

### `flextool/model_builder/` and `flextool/representative_periods/`

- **`model_builder`** — synthetic scenario generator used for benchmarking
  and stress-testing the engine. `build_model.py` is the entry point;
  `encoding.py`, `names.py`, `profiles.py` are helpers.
- **`representative_periods`** — greedy convex-hull clustering used at the
  input-preparation stage to pick representative weeks / days.
  `preprocess.preprocess_representative_periods` is the entry point.

#### Net-load clustering mode

By default the greedy hull clusters on the per-series normalized
profile/inflow feature stack (`preprocess._build_clustering_matrix`). The
`--netload-clustering` flag (schema wiring lives in
`representative_periods/netload_inputs.py` for the DB read and
`representative_periods/netload.py` for the pure math) instead clusters on a
real-MW **net-load** signal per aggregation unit `g`:

    net_load[g, h] = Σ_{demand node in g} demand_h
                     − Σ_{VRE unit in g} cap[u] · avail[profile(u)][h]

FlexTool's sign convention (shared with `force_include`) makes demand a
*negative* inflow, so `demand_h = −inflow_h + |scalar demand level|`. **VRE**
is any `unit__node__profile` arc whose `profile_method` is `upper_limit`;
that is the schema default, so an arc that never sets `profile_method`
explicitly (no row from the reader) is treated as VRE. Each aggregation
unit's series is min-max normalized to `[0, 1]` exactly as the default
feature builder does per feature.

An **aggregation unit** is either a **region group** or a single **node**.
Granularity is chosen by the schema-v68 `group.use_for_representative_periods`
(`yes_no`) flag: when one or more groups carry the flag `yes` the member
nodes' net load is summed into one regional signal per flagged group; when no
group carries the flag the selection falls back to **per-node** signals (each
node its own aggregation unit). On a pre-v68 database the flag parameter is
not yet defined — that genuinely-absent case is detected and quietly activates
the per-node fallback, while a read failure on a *defined* flag is allowed to
surface rather than silently degrade.

The iteration-0 VRE capacities come from a **demand-match** default: the
investable VRE of each aggregation unit is sized so its energy plus the
existing VRE energy just covers the unit's demand energy (pure energy balance,
curtailment ignored). `--vre-penetration` scales that energy-share target
(1.0 = full-energy match). The default (non-`--netload-clustering`) code path
is byte-parity with the prior behaviour.

#### Net-load iteration driver

`representative_periods/netload_iterate.py` (`run_netload_iteration`,
`NetloadIterConfig`; also a `main()`/`__main__` CLI) composes the single-shot
net-load selector into an *iterate-until-stable* loop that feeds each
iteration's solved investment capacities back into the next selection — sizing
the VRE fleet from a solve's realised investments (instead of the demand-match
default) sharpens which periods are net-load-critical.

- **Invest-only loop.** Iteration `k` in `range(iterations + 1)`. `k = 0` is
  the bootstrap: select on the demand-match caps and run an INVEST-ONLY solve.
  The bootstrap dispatch is **skipped** — default-cap investments are
  known-bad, so measuring their full-year cost is not worth the solve.
  `k = 1..n` select on the prior iteration's invested caps; if the selected
  representative set is unchanged from the previous iteration the loop
  converges and breaks before solving (**early-stop on stability**).
- **Keep-best (opt-in).** With `--keep-best`, each mature iteration (`k ≥ 1`)
  ALSO runs the full-year DISPATCH, reads its total system cost, and the loop
  keeps the representative set with the **lowest full-year dispatch cost**. A
  keep-best winner from an earlier iteration is deterministically
  re-materialised by re-running the selector on that iteration's input caps
  (greedy hull over a fixed matrix — no randomness). A final dispatch is run on
  the chosen set for output.
- **Stable-alternative-name orphan guard.** The full `alternative_name` is
  pinned once and passed as the explicit override to the selector every
  iteration, so the selector never derives a per-iteration name (whose
  force-suffix could shift with the forced count) and orphans timesets. A
  stable name lets the selector's purge+rewrite cleanly REPLACE the single
  net-load timeset each iteration, so N iterations leave exactly one.
- **`model.solves`-override subset mechanism.** `cmd_run_flextool` has no
  `--solves` flag; the DB-driven way to run a subset is the `model.solves`
  Array resolved under the scenario's alternative stack. The driver writes a
  dedicated `model.solves` OVERRIDE alternative at the top of the stack
  (mirroring `calibrate._db_alt`) — the invest solve(s) for an invest-only
  pass, the dispatch solves (or the scenario's full original chain) for a
  dispatch pass. A genuine subset run, not a silent full-chain fallback.
- **Fail-closed guards.** `_guard_no_timed_energy_margin` refuses to run when a
  node's `energy_margin_method` resolves to `inflow_adder` with a timed
  (2-D per-cell) `energy_margin_adder` — the timed adder and the net-load
  loop's cap feedback are mutually exclusive. Every iteration solve is
  `assess_solve`-gated on its required outputs; an untrusted solve raises
  `NetloadIterError` rather than feeding a bad cap set forward. Basis caches
  are cold per solve by default (the LP columns change with the representative
  set); `--warm-start` opts into one shared cache only when the grid is stable.

## Public API surface

```python
from flextool import (
    write_outputs,
    migrate_database,
    initialize_database,
    update_flextool,
)
from flextool.engine import (
    FlexData,
    FlexDataProvider,
    SolveHandoff,
    build_flextool,
    load_flextool,
    run_chain_from_db,
    run_orchestration,
)
from flextool.engine.input._db_loader import FlexToolRunner
from flextool.spinedb_backend import SpineDBBackend
from flextool.process_inputs import TabularReader, write_to_flextool_input_db
from flextool.process_outputs import write_outputs, post_process_results
from flextool.plot_outputs import plot_dict_of_dataframes
from flextool.scenario_comparison import get_scenario_results
from flextool.export_to_tabular import export_to_excel
from flextool.update_flextool import (
    migrate_database,
    initialize_database,
    update_flextool,
)
from flextool.helpers import (
    compare_files,
    find_largest_numbers,
    parse_mps_to_matrices,
    convert_schema,
)
```

## Solver layout

HiGHS is a regular Python dependency (`highspy`); FlexTool no longer
bundles solver binaries. The only solver-config artefact is the HiGHS
options file:

| Path | Tracked? | Role |
|---|---|---|
| `flextool/solver_config/highs.opt.template` | Yes (package data) | Template that ships in the wheel. |
| `solver_config/highs.opt` | No (gitignored) | User-editable runtime copy; seeded from the template on first run. |

`flextool._resources.package_data_path("solver_config/highs.opt.template")`
returns the template; `ensure_runtime_files` (GUI start, `flextool-update`)
copies it to the workspace's `solver_config/highs.opt` if absent. Users edit
the runtime copy; a run reads it once (see "Runtime options" item 6) and
`_solve_highs_options` layers the solve's settings and the CLI overrides on
it. The commercial templates (`gurobi`, `cplex`, `xpress`, `copt`) follow the
same pattern.

`--save-memory` (or `FLEXTOOL_SAVE_MEMORY=1`) is an opt-in peak-RSS
strategy that disables warm-start and round-trips each sub-solve through
an on-disk MPS file, trading ~5-10 GB peak RSS for slower per-solve
build time. The orchestrator emits a one-time warning when it disables
warm-start because of this flag.

**Solver hand-off.** A subprocess solve gets its merged options as a list of
`(name, value text)` pairs, never through a file it reads back: a commercial
script builder (`_subprocess_solve._SCRIPTS`) takes `commercial_params(options)`
and emits one argument or script line per pair; the `--save-memory` HiGHS
child (`cmd_solve_mps --options-json`) sets each pair with `setOptionValue`,
refusing an empty value as HiGHS' option-file reader does. `<solve>.opt` next
to the MPS is the record only. `tests/engine/test_solver_handoff_golden.py`
pins argv, scripts and HiGHS' resulting options for a corpus (golden made
before the hand-off was changed) and compares seeded random option lists with
the old write-and-read-back chain.

**Memory: polars allocator.** polars links jemalloc statically, and its
own decay settings in 1.x (`dirty_decay_ms:500,muzzy_decay_ms:-1`) leave
freed pages resident for good, which roughly doubles peak RSS on large
models (2.x writes `muzzy_decay_ms:1000`, which returns them after ~2 s).
FlexTool wants `dirty_decay_ms:1000,muzzy_decay_ms:0`
(`flextool/_allocator.py`, called first thing in `flextool/__init__.py`).
jemalloc reads `_RJEM_MALLOC_CONF` once, when `import polars` loads the
runtime library, so there are two paths:
- polars not loaded yet (CLI, GUI, a fresh Toolbox run): `setdefault` the
  variable; a value from the shell wins as a whole.
- polars loaded first (a Toolbox console, a notebook, a script): append
  FlexTool's values to the variable for child processes and write the two
  decay values at runtime with jemalloc's `mallctl` (`arenas.*` and every
  initialised `arena.<i>.*`), which also returns pages freed before
  flextool was imported. `_rjem_mallctl` is a local symbol, so its address
  comes from the library's ELF `.symtab` and the load base of its exported
  `PyInit_*`; the function's first bytes in memory must equal those in the
  file before it is called. A value set before polars was imported was
  already read by jemalloc and is left alone. Any failure (macOS Mach-O,
  stripped symbol table, a mismatch, a failing call) prints one warning
  on stderr with the remedy: import flextool before polars, or start
  Python with `_RJEM_MALLOC_CONF=dirty_decay_ms:1000,muzzy_decay_ms:0`.
  Windows (mimalloc) and builds without jemalloc are a silent no-op.

**Thread budget.** One number B per run (`--threads` > `FLEXTOOL_THREADS` > the available cores, by CPU
affinity) limits FlexTool's processing and the solver (`flextool/_thread_budget.py`). polars reads
`POLARS_MAX_THREADS` only at import, and `import flextool` imports it, so `flextool/__init__.py` runs, in this
order: the thread budget (`apply_before_polars`: when the process is a FlexTool command that accepts
`--threads` — `sys.orig_argv` holds `-m flextool.cli.cmd_run_flextool` / `cmd_solve_mps` /
`cmd_execute_flextool_workflow` / `flextool.calibrate` (`-m flextool.cli.cmd_run_flextool` is also the engine
that `flextool-run` supervises), or `sys.argv[0]` is a `flextool-run[.exe]` installed before the supervisor (which
ran the engine in-process) or the `cmd_run_flextool.py` file Spine Toolbox runs — the flag is read from the
command line, full name only; the assistant (`flextool-assistant`, `-m flextool_assistant_entry`) reads its
`--threads` the same way, and without it its budget is 1 (`command_default_threads`: it reads files while a model
may be running and must not compete with it; `FLEXTOOL_THREADS` does not raise it); then
`POLARS_MAX_THREADS = min(B, the user's POLARS_MAX_THREADS, else 8)`), the allocator config, `import polars`,
the rest of the package, and finally `restore_env`, which puts `POLARS_MAX_THREADS` back exactly as it was.
The environment therefore never carries B: a child gets `--threads` on its command line (workflow, calibrator,
GUI jobs, the `--save-memory` solve child). With a polar-high older than 3.12.2, whose import sets
`POLARS_MAX_THREADS=1` even with polars loaded, it is imported before the restore so that value is removed too.
When polars was imported before flextool, its pool cannot change; the header says so. HiGHS: a caller-set B
gives `threads=B` with the `parallel=off` pin kept (`highs_thread_options`); `--highs-threads N>1` keeps its
parallel-simplex opt-in, at most B. Benders: without `--benders-workers`, at most B region solves at once.

## Solver outputs: folder layout

```
work_folder/
├── input/, solve_data/        Debug-only mirrors of cascade state
│                                (csv_dump=True path). Not load-bearing.
├── output_parquet/<scenario>/ Canonical results — one parquet per output.
├── output_csv/<scenario>/     Optional CSV mirror of the parquets.
├── output_excel/              Optional summary workbook.
└── output_plots/              Optional PNG / SVG plots (default_plots.yaml).
```

Cross-solve state moves through `FlexDataProvider` keys; the `solve_data/`
CSVs exist only as a debugging aid.

## Run record

Every engine run (`cmd_run_flextool.main`, once its scenario and output folder are known) writes two files into
`output_parquet/<subdir>/` (`flextool/run_record/`):

- **`_run_status.json`** (schema `flextool.run_status`) — written by the **engine**, the only process that knows
  the verdicts: `RunRecorder` (`_status.py`) creates it as `running`, rewrites it atomically as solves start and
  are classified, and finishes it in `main()`'s `finally` (`succeeded` / `failed` / `interrupted`). It records each
  solve's model status and acceptance, its `reduced` detail levels (`reduce_*_detail_method`), the error, the input, the options and threads, the solver option folder and a hash of each option file
  (`solver_config`: `SolverOptionFiles.record()`), and `outputs.parquet_state`:
  a failed run keeps the earlier run's parquet, marked `previous_run`. `write_outputs` keeps both files when it
  empties the folder.
- **`_run.log`** — written by the **pipe owner**, the process that reads the engine's stdout + stderr pipe and so
  also sees native solver output and crash dumps: the GUI job thread (`ExecutionManager._run_job`) or the
  `flextool-run` supervisor (`flextool_run_supervisor.py`, also used by the workflow runner, the calibrator and the
  representative-period iteration). The owner sets `FLEXTOOL_RUN_LOG_OWNER=1` in its child's `env=` (the first
  exception in "Runtime options" item 4; `FLEXTOOL_RUN_STOP_FILE` below is the second); the engine then prints one marker line with the folder and `run_id`, and
  the owner (`RunLogOwner`, `_owner.py`) writes the log from there, hiding the marker. Without an owner (Spine
  Toolbox, `python -m flextool.cli.cmd_run_flextool`, a Python console) the engine copies its own Python streams
  (`_tee.py`, `log_capture: "python"`). The log keeps its first 64 KiB and last `FLEXTOOL_RUN_LOG_MAX_BYTES`.
- When the engine dies without finishing its status (a crash, a kill, a signal), the owner finishes it from the
  exit code (`classify_exit`: `killed`, `interrupted`, `crashed`). Without an owner a reader derives `abandoned` from
  a stale `running` (`effective_state`).
- A run the GUI stops is `killed` with the reason (`ExecutionManager._stop_job_locked`: the job is KILLED with its
  `kill_reason` before its process is killed). A calibration or representative-period job runs its engines under
  `flextool-run` supervisors, which own those records, so the GUI does not kill them with the rest: it gives the
  job's process `FLEXTOOL_RUN_STOP_FILE` (a file in a private temp folder, inherited by each supervisor), writes
  the reason into that file, kills the job's process and everything not under a supervisor, gives the
  supervisors 10 s to kill their engines and record `killed`, then kills what is left (`stop_aux_tree`).
  The stop thread is the file's only owner and removes its folder at the end. A job's worker finishes the record
  before the job's output folder is free for the next job, and closing FlexTool waits (at most 15 s) for those
  records.
- A GUI job whose engine ends before it starts its record (an early error, a kill in its first seconds) would leave
  the folder's previous `succeeded` record, and its results would still be offered. The GUI then writes a minimal
  `failed` / `killed` record (`record_run_without_marker`, `_owner.py`) whose `outputs` carry the previous ones
  (`carried_outputs`, the same rule the engine uses), with the engine's early output in `_run.log`. It writes only
  over an older finished record or a legacy folder with parquet; never into a missing folder, over a record that
  cannot be read, over another run's `running` record, or over a run that started at or after the launch. The run
  id on disk is read again just before the replace, and the log is written after the status, only while the status
  is still the GUI's.

The supervisor is a top-level module that never imports `flextool` (that imports polars): it loads `_owner.py` and
`env_check.py` by path, so `_owner.py` stays stdlib-only with no relative imports.

**Whether a folder's results may be offered** is one pure rule, `run_record.results_usable(status)` (with
`unusable_reason`): no record (a folder from before run records) → usable; otherwise only `succeeded` with
`parquet_state == "written"`. Every consumer calls it and none restates it:

- the GUI executed list (`ExecutedScenarioManager.scan_executed` → `ExecutedScenarioInfo.status` /
  `run_state` / `checkable`): a row that is not usable shows its state in the check column (✗ / … / –) with a
  tooltip, cannot be checked, is unchecked on a rescan, and its ▶ opens `_run.log`. The timestamp is the run's
  `finished_at` (else `started_at`), so sorting is unchanged. The list is refreshed once when a job's engine has
  written its record;
- the result viewer (`ResultViewer._scan_scenarios` drops such folders after the ordering merge) and the
  comparison (`OutputActionManager._build_comparison_cmd`; after a batch `ExecutionManager._run_post_execution`
  leaves out folders whose latest job failed). A new reader of run folders calls it too.

`ExecutionManager` also starts at most one job per output folder at a time (`_pick_next_pending`).

## AI assistant tools

`flextool/assistant/` holds read-only tools that let an AI assistant answer questions about one project: input data
by scenario, results, run logs and slack, and the documentation; the connection server that gives them to an AI
agent; and FlexTool's own chat with an AI service ("AI assistant chat" below). User page:
[The AI assistant](../ai_assistant.md).
`flextool/assistant/__init__.py` imports nothing; every module imports what it needs when first used.

- **Registry** (`registry.py`). Each tool module (`tools/orientation`, `discovery`, `inputs`, `results`,
  `diagnostics`) lists its `Tool`s in `TOOLS` (name, title, description ending with the data-not-instructions
  sentence, a closed JSON schema without ranges, handler, area, `small_tier`). `call_tool(ctx, name, args)` is the
  only way to run one and never raises: it clamps the caps arguments (`caps.py`: `--result-cap` presets and hard
  ceilings), validates against the schema with `jsonschema` (mcp does not validate on stdio), runs the handler and
  fits the payload with `fit_json` (compact JSON, whole trailing rows dropped with a `truncated` note). A
  `ToolError` carries a message key of `errors.MESSAGES`, the one table of user-facing texts; any other exception
  becomes the fault message, with the traceback on stderr only. `tools_for_tier("small")` is the basic set.
- **Redaction** (`redact.py`) runs once, in `fit_json`, on every string of every payload and every message: the
  project folder → `<project>`, home → `~`, host and user names, other absolute paths → their last part.
- **Inputs** (`inputs/_db.py`). One read-only, percent-encoded SQLite URI open per file (`mode=ro` counts only
  with `uri=true`; a raw `#` or `%` makes SQLite create another file), `sqlite_timeout=5`; a fixed list of item
  types is fetched and the mapping closed, so no file handle stays open. The scenario view (rank order,
  `entity_alternative`, `is_enabled`) is computed in our code; spinedb-api's scenario filter is only the oracle of
  its test. Input sources and their numbers come from `gui.input_sources` / `gui.settings_io` (Tk-free), read
  only; a spreadsheet is read through its conversion `intermediate/<stem>.sqlite`.
- **Results** (`results/`). A run folder is offered only when `run_record.results_usable` says so (see "Run
  record"); the message comes from `unusable_code`. Tables are read and aggregated wide (`flextool/lean_parquet.py`
  footer layout); only a reduced frame is unpivoted. `check_aggregation` refuses sums with no meaning (power over
  time, annual values over periods, levels, ratios). Units are worded only by `unit_wording.py`, as the default
  convention (units are not enforced by FlexTool). Comparison files are used while their `_metadata.json` members
  match the current successful runs, else the member folders.
- **Docs** (`docs_store.py`, `flextool/docs_build/`). The build step `_flextool_build.BuildPy` copies the curated
  Markdown (`docs_build/scope.CURATED_DOCS`) into `flextool/assistant/_docs/md/`; an editable install reads the
  clone's `docs/`. On first use the formulation page, the results catalogue and the BM25 index are generated by the
  installed FlexTool into the user cache folder, keyed by FlexTool version + source hash. The web docs generate the
  same two pages with the same generator (`tools/mkdocs_hooks.py`), plus the web-only parameter catalogue
  (`scope.WEB_ONLY_PAGES`, `docs_build/parameters.py`: one large table per class, which `describe_parameter`
  already serves, so it stays out of the assistant's index); `python -m flextool.docs_build --xlsx` writes it as an
  Excel data dictionary.

**The connection server for agents** (`flextool-assistant --mcp`, `mcp_server.py`, mcp 2.x low-level `Server`,
telemetry middleware removed). The protocol uses stdin and stdout, so `flextool_assistant_entry.py`, a
standard-library-only top-level module, keeps private duplicates of fd 0 and fd 1 for the protocol, then points
fd 1 and `sys.stdout` at stderr and fd 0 at the null device (on Windows also the Win32 standard handles, and
binary mode on the duplicates) **before** `import flextool`, whose dependencies and start-up checks may print.
The server's streams are explicit `anyio` wrappers over those duplicates (UTF-8, `\n`), so mcp's own stdio claim
is skipped. Tool calls run in a worker thread that takes the context's lock, so `ping` and `tools/list` answer
during a slow call. The server sends the instructions of `prompt.instructions` (data use, units convention,
vocabulary), lists the tools with read-only annotations, serves the docs pages as `flextool-docs:///<page>`
resources and the level texts (`prompt.ADDENDA`) as the prompts `basic`, `standard`, `advanced`. `--tier basic`
lists and runs only the basic set and defaults to the small result cap.

When `import flextool` stops with `CompatRuntimeRequired` (env_check: this processor needs the compatible polars
build), the entry module loads `mcp_server.py` by path (it imports only the standard library at module level) and
serves one tool, `flextool_setup_problem`, that returns the fix. The assistant options have no environment-variable
twins.

`--call TOOL [JSON | @file.json]` runs one tool through the same `call_tool` and prints its text (support and
testing). Tests: `tests/assistant/` (`test_server_stdio.py` drives the installed command over real stdio and fails
on any non-protocol byte on stdout); CI smoke: `.github/scripts/mcp_smoke.py` (wheel check in `publish.yml`, the
Toolbox bundle, the old-CPU emulation's degraded server).

### AI assistant chat

`flextool-assistant` without `--mcp` / `--call` is the chat (`chat_cli.py`); the same layers serve the FlexTool
window ("AI assistant in the FlexTool window" below). `--mcp` and `--call` import none of the service libraries (`anthropic`, `openai`, `keyring`)
nor the chat (`tests/assistant/test_import_hygiene.py`).

- **Neutral model** (`llm_adapter.py`): `Message`, `ToolCall`, `Request`, `Reply`, `Usage`, the events a front
  end shows (`TextDelta`, `ToolDone`, `Waiting`, `RetryWait`, …) and the `Backend` protocol. Data only one provider
  understands (Anthropic thinking blocks, Gemini thought signatures, OpenRouter reasoning details) travels in
  `provider_extra`, stamped with its origin (preset, base URL, model), and is replayed only to that origin; on a
  switch it is dropped and tool-call ids are re-mapped. Ollama's reasoning string (the preset's
  `reasoning_wire_key`, `delta.reasoning` joined) is kept as `provider_extra["reasoning_text"]`: never shown,
  saved with the conversation, and sent back as `reasoning` only on the assistant messages after the last user
  message (gpt-oss's template renders it only there; without it the model leaks its analysis into tool calls). It
  does not make a turn's history uneditable, and like all message data a compaction drops it — also one in the
  middle of a turn, after which this turn's earlier reasoning is not sent again.
- **Backends** (`backends/`): Anthropic Messages and OpenAI Chat Completions (also Mistral, OpenRouter, Gemini,
  Ollama, LM Studio and any compatible endpoint). The provider libraries are imported only in `make_backend`.
  `net.py` builds each library's HTTP client: headers rebuilt from an allow-list (the libraries' identification
  headers and the variables they read add nothing), our own proxy decision (a local endpoint is never proxied), no
  redirects, cancel by shutting the open sockets. Presets (`providers.py`) name no models; they say which request
  fields a service accepts and which privacy fields are sent (`store: false`, OpenRouter
  `provider.data_collection`).
- **Errors and retries** (`llm_errors.py`, `llm_retry.py`): every library error becomes one `LLMError` subclass
  with a plain-English message; our own retry loop (no library retries), never after the first streamed output.
- **Probes** (`probes.py`): before the first question, the model's context and tool support (Ollama `/api/show`,
  load, `/api/ps`; LM Studio; the OpenRouter model list; Anthropic limits), standard library only and proxy-less
  for local servers. A model Ollama forwards to another host (its cloud) gives a remote route.
- **Session** (`session.py`): the agent loop — rounds of tool calls per question capped by level, tool calls under
  the process-wide tool lock, the history kept valid for every provider after a cut-off, refusal, stop or error;
  compaction points (every `compaction_every` questions, a context overflow, a rejected signature, a resume with
  changed instructions) drop all provider data. `TurnWorker` runs turns in a daemon thread; a stop returns at once.
- **Prompt** (`prompt.py`): five byte-stable layers (core rules, orientation, documentation map, level text,
  front-end note) and the context block (project, input source, scenario) at the start of a user message; the
  agent connection's instructions reuse the same rules.
- **Levels** (`tiers.py`, standard library only): `resolve_tier` places a model from `TIER_RULES` (data, first
  match wins: small-model words, the reported or named size, family words, GPT major ≥ 5, a fallback guess by
  route) and caps it by the context window (`WINDOW_LEVELS`; below `MIN_WINDOW` it raises `WindowTooSmall`). The
  automatic level is the lower of the two; `allowed_levels` is it and below, or up to the window's level for an
  unrecognised model (`ALLOW_HIGHER_WHEN_UNRECOGNISED`). There is no bypass in shipped code: a stored or `--tier`
  level above the allowed ones is lowered with a notice (`-q` with `--tier` exits 2). Before a probe the front ends
  use `probes.cached_facts` (user cache); the settings dialog never probes. The level changes only the prompt
  (`prompt.ADDENDA`), the tool list (`registry.tools_for_tier`; basic = the `small_tier` tools, any other name
  refused by the session before the registry) and the budget — never what a tool returns. Agents choose a level
  with the connection prompts or `--mcp --tier` (not checked against their model).
- **Budgets** (`budget.py`, standard library only): `LevelBudget` per level (`LEVEL_BUDGETS`, the free plan's
  `FREE_TIER_BUDGETS`): rounds, tool calls per question and per round, the output limit (≤ window / 4 for a known
  window), planned calls and the prefix budget the shipped layers and schemas must fit
  (`tests/assistant/test_prompt_budget.py`). `resolve_budget` adds the window's class (`WINDOW_CLASSES`: result-cap
  ceiling, `fit_caps` for windows under 32 768, and the compaction rhythm; settings `compaction_every` /
  `keep_recent_turns` `null` = automatic). `approx_tokens` is a digit-aware estimate (`JSON_CHARS_PER_TOKEN`,
  pinned by `tests/assistant/llm/test_token_estimate.py` against real tokenizer counts) corrected per session by the
  reported prompt tokens (`TokenCorrection`); the session's pre-send fit compacts or, for origins with replay data,
  asks for an answer without tools, and sends nothing when the estimate plus the answer limit stays above 95 % of
  the window (a reported one, or the assumed local one). The free plan
  (`free_tier_on`: the setting, else on only for an OpenRouter `:free` id) lowers rounds and calls, caps results at
  medium and, in the FlexTool window, gives the chat's gated backend a limit of one request at a time; the first rate-limit reply on a route with it off
  shows the hint once (`settings.free_tier_suggested`).
- **Local files** (`_files.py`, `settings.py`, `store.py`, `usage.py`, `keys.py`, `endpoint.py`), in
  `platformdirs.user_config_dir("flextool")/assistant/`: settings, conversations per project (lock file per open
  conversation, purge by age at every start), the usage ledger, the key file (opt-in; the keyring first, the
  preset's environment variable before both) and `acknowledgements.json`. Every read-modify-write runs under a
  lock file; every time is UTC.
- **Data notice** (`endpoint.py`): the route (loopback, local network, remote; the proxy; the forwarded host), the
  indicator line, and the notice a remote route needs before the first question. An acceptance is kept per
  preset, normalised address, proxy host, forwarded host and notice wording.
- **Terminal** (`chat_cli.py`): the REPL (a terminal only) and `-q` (only the answer on stdout; exit codes 0 / 1 /
  2 / 3 / 130). The question and the probe run in daemon threads, so Ctrl-C stops them at once. The banner gives the
  level and why (`level_lines`); `/context` the window, the result cap and the compaction rhythm of the session's
  budget; `--free-tier on|off|auto` overrides `settings.free_tier` for the session.

### AI assistant in the FlexTool window

`flextool/gui/assistant/` is the window's front end on the same session, store, settings, keys, notices and ledger
(user page: [The AI assistant](../ai_assistant.md), "Chat in the FlexTool window"). Everything without Tk — the
worker, the provider gate, the Markdown parser, the window geometry, the strings — is tested in the sharded job
(`tests/assistant/gui_support/`); the Tk modules in the `gui-tests` job ([Testing](testing.md), "GUI tests").

- **Entry and lifecycle** (`entry.py`, `AssistantEntry`, created by `MainWindow`): "AI assistant…" + ⚙ in side-menu
  row 5. Building it only runs `importlib.util.find_spec`; the chat window, the dialogs, the SDKs, the keyring and
  the store are imported on the first click. It closes the chat window on a project switch, rename or deletion
  (one window per project) and waits for its workers (5 s; the Project dialog and the project combo refuse a
  rename or deletion while one still runs a tool call, `before_project_change`); `project_utils.rename_project`
  moves the saved chats (`store.rename_project`), `notify_assistant_project_deleted` deletes them
  (`store.forget_project`). At FlexTool close (`MainWindow._on_close`, before the settings save) it withdraws the
  windows, shuts the workers down, deletes the runtime file and purges, within 3 s; at start (after 3 s) it purges
  and removes runtime files of gone processes.
- **Threads**: one `AssistantWorker` daemon thread per chat tab owns its `Session`, conversation and lock; the Tk
  thread never calls the network, keyring, probes, store or ledger (except reading the settings and the welcome
  flag when a window opens). Worker → Tk only through queues drained by one `after(50)` poll. The process-wide
  `ProviderGate` (`gate.py`) lets `parallel_requests` (1–4) model requests per provider run at once (a free-plan
  chat's own `limit` is 1), gives slots in arrival order and shares a rate-limit cool-down between chats; tool
  calls run outside it, under the process-wide tool lock. A shut-down worker saves a checkpoint in which a running
  tool call is "Not run: FlexTool was closed…", so the file is always a resumable history. A project rename or
  deletion moves or deletes its chats on one FIFO daemon thread (`project_utils`); every store user waits for the
  queued hooks first (`wait_project_hooks`). The Tk thread's blocking waits (project change, close) pump
  `dooneevent(FILE_EVENTS | DONT_WAIT)`, so a worker's Tcl call (a finalizer collected on its thread) finishes;
  closed assistant dialogs retire their Tk variables on the Tk thread.
- **Window** (`chat_window.py`, `chat_tab.py`, `transcript.py`, `notices.py`, `settings_dialog.py`): chats as
  `ttk.Notebook` tabs with an overflow budget (the Chats menu shows the hidden ones), restored when the window
  opens again (`store.open_chats`); one data notice at a time; the size per monitor configuration in
  `GlobalSettings.assistant_window_geometry`; a key shield on every assistant Toplevel so the main window's
  `bind_all` shortcuts never fire from it. The settings' level rows list "Automatic" and only
  `TierChoice.allowed` (from cached facts; ttk cannot grey single entries) with `TierChoice.lines()`, and the free
  plan as a three-entry list (automatic / on / off → `free_tier` null / true / false); the worker's setup
  fingerprint includes `free_tier`, so a saved change applies from the next question.
- **What the window shows** (`gui_state.py`): `snapshot_from_main` reads the main window's checks and highlights
  and `ResultViewer.view_state()` on the Tk thread (each part on its own, about 2 ms); every Send takes its own
  snapshot (`ContextSnapshot.gui`), which the prompt renders as the context block. While a project is open,
  `ContextPublisher` polls it every second and hands a changed payload to the `RuntimeContextWriter`
  (`flextool/assistant/runtime_context.py`): one file per process,
  `<config>/assistant/runtime/<project key>.<pid>.json`, names and selections only, rewritten every 30 s while the
  poll ticks (a hung Tk thread's file goes stale), deleted by the writer thread at close. An agent's
  `get_gui_context` (`--mcp`, `--call`) reads the file of the window in use (`read_runtime_context`: process alive,
  written within 120 s; the latest focus `active_at`, then `changed_at`, then `written_at`); a live process with
  only a stale file reads as open but not answering (`responding: false`).

Tests: `tests/assistant/llm/` — no network (a socket guard refuses anything but loopback), synthetic provider
payloads through `httpx2.MockTransport`, a scripted `FakeBackend` for the session and the chat, loopback servers
for the probes and the cancel. Real provider replies are recorded by hand with `tools/record_llm_replies.py` on
the synthetic test project only, scrubbed, checked and replayed in CI ([Testing](testing.md), "Recorded provider
replies").

## Solve-to-solve handoff

When a solve is part of a rolling-horizon, nested, or sub-solve chain, the
just-solved problem produces a `SolveHandoff` (typed in-memory carrier).
Inside the cascade it's transported via the `HANDOFF_*` Provider keys —
zero disk round-trips. The `csv_dump=True` mirror files retain the same
names as before for debug inspection:

| Carrier (`SolveHandoff` field) | Debug-mirror file | Meaning |
|---|---|---|
| `realized_invest` | `p_entity_period_existing_capacity.csv` (period column) | New capacity built this solve, in absolute units. |
| `realized_existing` | `p_entity_period_existing_capacity.csv` (existing column) | Cumulative existing capacity history per `(entity, period)`. |
| `divest_cumulative` | `p_entity_divested.csv` | Cumulative divest per entity carried forward. |
| `roll_end_state` / `upward_roll_end_state` | `p_roll_continue_state.csv` | `v_state` at the end of the realized window; the start state of the sub-solves that continue from it, routed by `solve/_carry_routing.py` (the latest realised state at the sub-solve's nesting level or deeper; engine.md "Storage carry routing"). |
| `roll_end_online` | *(in-memory only)* | Committed jump-end `v_online` per online process (OC1 Carrier 1); the next roll's `online_change` first step (`dt_online_carry`) sees the prior commitment instead of the within-roll cyclic wrap, so a unit that stays committed is not re-charged a startup across the boundary. |
| `online_history` | *(in-memory only)* | The realized `v_startup`/`v_shutdown` that a minimum up / down time still covers after the solve: the last realized step and its predecessors along `dt_ramp_link`, each with `hours_to_end` (hours from its start to the end of the last realized step); inside a forecasts-only fan under the tied forecast's steps. Rebuilt into `uptime_carry`/`downtime_carry` at the next solve's steps that follow its `dt_ramp_carry` and injected on the `minimum_uptime` RHS / `minimum_downtime` LHS. |
| `fix_storage_quantity / price / usage` | `fix_storage_*.csv` | Parent-imposed storage quota at the boundary (any subset of the three metrics — each carries its own `(node, period, step)` index, so a separate timesteps carrier is not needed). |
| `cumulative_co2` | `co2_cum_realized_tonnes.csv` | Running CO2 totals across rolls. |
| `cumulative_commodity` | `commodity_ladder_cumulative.csv` | Per-tier commodity consumption (ladder pricing). |
| `cum_sim_hours` | `ladder_cum_sim_hours.csv` | Running simulated-hour total per period. |

Cross-boundary unit-commitment continuity (OC1 Carriers 1 & 2) is
**hard-enforced** and **timeline-aware**: a `min_uptime` / `min_downtime`
obligation opened near a solve's last realized step carries exactly where
`dt_ramp_carry` does — into the first steps of the next solve that continue
that step on the timeline (`jump == 1`): the same period, a period that
continues the previous one, and the forecast branches that start with such a
step. It does not carry into a period that does not continue the previous one
(investment years that reuse one timeset, e.g. 2020 → 2025), across a gap or
into a representative-block start. The consumer counts elapsed time from the
`dt_ramp_carry` roots along `dt_ramp_link` and adds each history step's
`hours_to_end` (`params/_mintime.py`); it raises when the history does not end
at the carried step (history and step handed over from different solves).
Time-keyed carries (ramp, online, min up / down) reach forecast branches
through the `dt_*_carry` roots (`_ramp_links._with_fan_members`); the
period-keyed `branch_periods_inherit_carry` (`params/_branch.py`) is
deliberately not applied to them — `hours_to_end` plus the roots is its
timeline counterpart, and a copy per branch would count twice. A solve whose
first step continues a step of another time resolution (a 6 h storage roll
after an hourly dispatch) has no `dt_ramp_carry` there: it takes neither the
ramp nor the min up / down carry, and the cut-short look-back warning names
the step. A node with the default `penalty_method`
(`regular`) absorbs any forced-on generation through its downward penalty
(`vq_state_down`, a feasible spill); a hard-balance node (`penalty_method =
off`) can instead become infeasible when a myopically-committed unit's
forced generation cannot be absorbed — that is the model correctly refusing
an un-absorbable commitment, resolved by allowing slack.

`fix_storage_timesteps`, `ed_history_realized_first`, and `edd_history`
used to be `SolveHandoff` fields too, populated only by the now-removed
`capture_post_solve()` disk-read constructor. They were retired when the
cascade fell through to Provider/CSV reads in all production paths; see
the docstring of `solve/_handoff.py` for the history.

## Solution extraction

`process_outputs.read_highs_solution.VARIABLE_SPECS` declares every
variable and dual to harvest from the live `highspy.Highs` instance:
`v_flow`, `v_state`, `v_invest`, `v_divest`, `vq_*`, period / total /
cumulative invest-cap duals, `co2_max_*` duals, the inflation-adjusted
`v_dual_node_balance`, and so on. Each `VariableSpec` carries an
`unscale_by` field so row / objective scaling is undone before the parquet
is written.

To add a new variable or dual: append a `VariableSpec` to `VARIABLE_SPECS`.
No other changes required.

## Numerical scaling

FlexTool's LP/MIP can accumulate coefficients spanning many decades when
users combine entities of different physical scale (10 kW heat pump
alongside 10 GW continental grid). The scaling pipeline is layered so each
mechanism targets one source of spread; see [scaling.md](scaling.md) for
the user-facing guide and [slack_convention.md](slack_convention.md) for
slack semantics.

### Layers (bottom up)

1. **Unitsize column normalisation.** Every variable is written in units
   of `p_entity_unitsize` per entity, so the raw values stay near `O(1)`.
2. **Single-variable slack convention.** Every `vq_*` is a single
   non-negative variable, relative to its row-scaler where one applies.
   See [slack_convention.md](slack_convention.md).
3. **Row scaling (opt-in, `--scaling` CLI flag / `FLEXTOOL_SCALING`).**
   Node-balance and
   group-balance constraint rows get multiplied by
   `node_capacity_for_scaling` / `group_capacity_for_scaling` derived
   from connected-entity unitsizes; rounded to powers of 10 to preserve
   HiGHS symmetry detection.
4. **Autoscale package** (`engine/autoscale/`). Three layers,
   on by default (gate via `--scaling=off` / `FLEXTOOL_SCALING=off`;
   intermediate modes are `solver_only`, `basic`, `full`):
   - **Layer 1 — detect** (`_ranges.py`). Walks the assembled LP and
     computes matrix / cost / bound / RHS log10 ranges plus a
     cross-group max-ratio. Always emitted to the YAML audit; the
     `trigger` flag tells the orchestrator whether Layers 2/3 should
     run. The range readout routes through polar-high's
     `bounded_coefficient_walk` (requires polar-high>=2.4.0): the
     constraint / column spine is iterated in fixed row-batches and the
     `(rid/col_id, coef)` product is rebuilt per batch, so the wide
     coefficient product is never materialised to read its min/max. The
     former size-blind family-row cap (`POLAR_HIGH_RANGES_MAX_FAMILY_ROWS`
     / `_skip_unbounded_over_cap`) is gone — every family's range is now
     bounded and folded into the report, so no shape is silently dropped.
   - **Layer 2 — semantic per-type scaling** (`_layer2.py`,
     `_quantity_types.py`). Buckets matrix columns by quantity-type
     (energy, capacity, monetary, …) and applies a power-of-2 scaler
     per type to compress the median-of-type onto O(1). Has structural
     limits: cannot compress within-type spread, and a guard
     prevents over-shrinking columns with mixed-magnitude
     coefficients on a single row. Like Layer 1, the magnitude-bucketing
     readout walks the spine in bounded row-batches via
     `bounded_coefficient_walk` (the log2-magnitude histogram reducer)
     rather than collecting the wide product — and with the family-row
     cap retired, every family's magnitudes are folded into the scaling
     decision (no silent coverage gap).

     **Registry contract — read this before adding any `add_var` /
     `add_cstr` / parameter.** Layer 2 looks up the quantity-type of every
     variable (`VARIABLE_FAMILIES`), constraint (`CONSTRAINT_FAMILIES`
     via `lookup_cstr`) and parameter (`PARAMETER_TYPES`) in
     `_layer2_types.py` / `_quantity_types.py`. These registries refuse
     to default — an unregistered name raises `KeyError` in
     `bucket_coefficients`. That `KeyError` is **caught** in
     `_orchestration._autoscale_apply_layer2_pre_solve` and the solve
     silently reverts to an **un-scaled LP** ("autoscale Layer 2 apply
     failed … reverting"), so a missing registration does not crash — it
     quietly disables autoscaling. So: every new `add_var(...)` /
     `add_cstr(...)` literal and every new schema parameter MUST get a
     registry entry (flags / `yes_no` gates → `DIMENSIONLESS`).
     `lookup_cstr` resolves dynamic (f-string) constraint names by
     exact-match → `_linear`/`_integer`/`_p`/`_n` suffix strip → longest
     prefix; dynamic names are invisible to the static-literal grep in
     `test_registry_coverage`, so pin them in
     `test_dynamic_constraint_names_resolve`. Run autoscale-exercising
     tests with `FLEXTOOL_AUTOSCALE_STRICT=1` to turn the silent degrade
     into a loud failure (off in production so a gap never blocks a user
     solve).
   - **Layer 3 — HiGHS native + bound-scale folding** (`_layer3.py`).
     Picks an integer `user_bound_scale` exponent (within HiGHS's safe
     range) and folds D's escape-tier valve into the same pass, so the
     LP arriving at HiGHS is comfortably within the solver's
     conditioning sweet spot.
5. **YAML audit** (`engine/autoscale/_report.py`). After every
   solve, writes `solve_data/autoscale_<solve>.yaml` covering Layer 1
   ranges, the Layer 2 plan (per-type exponents + post-L2 ranges),
   and the Layer 3 plan (`user_bound_scale`, escape-tier).
   A 1-line console summary echoes via `format_console_summary`; the
   non-optimal hint (`format_nonoptimal_hint`) fires only when a *rejected*
   solve's **post-autoscale** LP is still wide-ranged (i.e. the autoscaler
   could not tame the spread), and even then it is framed as a possibility
   — never as the certified cause. Accept/reject itself is decided by
   `engine/solve/_acceptance.py` from the solver's own diagnostics
   (see "Solve acceptance" below), not by the range report.
6. **Manual override.** `--user-bound-scale N` (or
   `FLEXTOOL_USER_BOUND_SCALE`) pins Layer 3's bound-scale exponent and
   disables the auto-pick. Useful when the audit hints at a value the
   recommender did not lock in.
7. **Output un-scaling** (`process_outputs/read_highs_solution.py`).
   Every `VariableSpec` carries an `unscale_by` field; the Layer 2 +
   Layer 3 reverse maps are applied in the writer so downstream
   consumers see absolute units regardless of which layers fired.

### Solve acceptance

The cascade does **not** gate on a bare `kOptimal`. An interior-point
solve run without crossover (the model generator's `run_crossover`
choice) can return a genuinely feasible, in-practice-optimal *primal* that
HiGHS refuses to *certify* — post-solve leaves the dual objective slightly
inconsistent, so HiGHS reports `kUnknown` rather than a failure. FlexTool
consumes the primal for every output, so such a solve is usable.

`engine/solve/_acceptance.py::classify_acceptance` decides from the
solver's own diagnostics (`polar_high.Solution.solve_diagnostics()`, a
snapshot of `getModelStatus()` + `getInfo()`):

- `kOptimal` → accept.
- a genuine failure status (`kInfeasible`, `kUnbounded`, `kTimeLimit`,
  `kSolveError`, …) → reject, naming the precise cause.
- `kUnknown` → accept as **near-optimal** iff the primal is feasible
  (HiGHS' own verdict + a scale-invariant relative check + a
  tolerance-derived absolute backstop) *and* `primal_dual_objective_error`
  is within the accepted optimality gap; otherwise reject.

Thresholds follow the solve's own `primal_feasibility_tolerance` (the
option value HiGHS ran with, read from the live handle; the Layer-2
unscale resets status and info, not options): primal-relative ceiling
`max(1e-6, 10 × tolerance)`, absolute margin `10×` the tolerance, and P-D
gap `1e-2` (the same order as the `mip_rel_gap` models routinely accept).
The dual side is judged by the P-D gap only, so there is no dual ceiling.
Each has a `FLEXTOOL_ACCEPT_*` env override; a set (positive)
`FLEXTOOL_ACCEPT_PRIMAL_REL` replaces the derived relative ceiling outright.
`FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0` restores the legacy hard-fail. A solve
accepted above the former fixed `1e-6` prints the
`acceptance_ceiling_follows_tolerance` behaviour-change notice. The subprocess/commercial path exposes no
queryable handle, so `solve_diagnostics()` returns `None` and the solve is
never near-optimal-accepted — it falls back to the strict `kOptimal` gate.

## Test layout and fixture architecture

### JSON-fixture single source of truth

`tests/fixtures/*.json` is the only authoritative source for test data and
canonical example databases. Schema migration covers them via
`flextool.update_flextool.test_fixtures.migrate_all`, and the CI verify
chain enforces drift detection.

**No test reads a checked-in `.sqlite`.** A test that needs a DB builds
one from its JSON / schema source under `tmp_path_factory` at session
start (e.g. `json_to_db(...)` for fixtures, or the `schema_db_url`
fixture which runs `initialize_database` on `spinedb_schema.json` once
per session). Checked-in DBs (`templates/*.sqlite`) are user-facing and
go stale between regenerations: a registry-coverage test that read
`templates/input_data_template.sqlite` stayed green through the v56
`is_enabled` add because the on-disk template lagged the schema. Build
from the source of truth so a schema change is caught the moment it
lands — and never mutate a user-facing DB from a test. Some downstream JSONs (e.g.
`templates_examples.json`, `howto_stochastics.json`) are *generated
projections* of authoritative fixtures via
`flextool.update_flextool.generate_canonical`; the remaining `howto_*.json`
files are individually authoritative.

For append-only edits to fixtures (entities / alternatives / scalar
parameter values / scenarios), use
`flextool.update_flextool.extend_tests_fixture` — it validates against
`spinedb_schema.json` with did-you-mean suggestions and rejects edits to
existing entries.

### Test directory layout

```
tests/
├── conftest.py                  Shared fixtures (scenario_workdir, ...).
├── fixtures/                    Authoritative JSON fixtures.
├── scenarios.yaml               Integration-suite scenario definitions.
├── expected/                    Golden CSVs (per-scenario, regenerated
│                                  when the suite genuinely changes).
├── engine/               Engine-internal unit tests.
├── spinedb_backend/             Backend unit tests.
├── model/                       Model-level end-to-end suites.
├── csv_reload/                  Unsupported CSV reload; run only with -m csv_reload.
├── gui/                         GUI startup tests.
├── decomposition/               Benders / region-filter tests.
└── test_*.py                    Top-level unit & integration tests.
```

`scenario_workdir(scenario_name, db_fixture="main")` (defined in
`tests/engine/conftest.py`) is a session-scoped fixture that
builds a per-scenario workdir on demand by running the full cascade with
`csv_dump=True`. It replaced the historical bulky CSV snapshot pattern
that previously sat under `tests/engine/data/work_*/` (~521 MB
gitignored); the directories that remain contain only a single
`golden_obj.json` golden file each. Tests do not depend on pre-built
fixture directories.

### Three-tier gate

- **Inner loop** — module-specific tests, run after every edit
  (typically `tests/engine/<area>/`).
- **Broader gate** — `tests/engine/` full sweep, once per commit
  batch.
- **Activation truth** — full `tests/` including the integration scenario
  suite, run rarely (overnight / release).

## Key architectural patterns

- **CLI → core delegation.** Every user-facing entry point (CLI script or
  GUI) is a thin wrapper that delegates to a core library module. The
  Tkinter GUI is special: it spawns CLI commands as subprocesses rather
  than calling library functions directly, so user runs stay isolated.
- **State containers.** `RunnerState`, `SolveConfig`, `TimelineConfig`,
  `PathConfig` bundle related state for the cascade. The Provider is the
  transport surface between phases.
- **Dataclass-driven module boundaries.** `FlexData`, `SolveHandoff`,
  `DatabaseContents`, `TimeSeriesResults`, `DispatchMappings`, `SheetSpec`
  define data shapes where packages meet.
- **Multi-format I/O at the edges only.** Inputs (Excel / ODS / CSV / old
  `.xlsm` / MATPOWER) convert to Spine DB; outputs render from parquets.
  The engine itself is single-source / single-sink.
- **Verify hooks instead of conventions.** CI verifies that schemas,
  canonical DBs, and test fixtures stay in sync — drift is a failed
  workflow, not a code-review catch.
