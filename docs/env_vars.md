# Environment variables

FlexTool takes its settings from three places:

- the **database**: what changes the model or its answer (for example a solve's `scaling` or `solver_mip_gap`);
- a **command-line flag** of `flextool-run`: a choice for one run (threads, time limit, `--save-memory`, ...);
- an **environment variable**: a setting for a whole process, and the twin of a flag where you have no command
  line — a Spine Toolbox tool, a Python console or script, a CI job.

When one option has several of these forms, the order is always
**flag > environment variable > database > default**. A flag you do not give lets the environment variable
through. A run never writes the variables of this page's first table, so a run started from the GUI, the
calibrator or a workflow inherits them exactly as you set them in the shell. The run header lists every option
set by a flag or an environment variable, and where it came from.

All variables are optional, and an empty value counts as unset. In the first table, a value FlexTool cannot
read (for example `FLEXTOOL_HIGHS_THREADS=many`) is ignored with one warning that names the variable.

## Twins of `flextool-run` flags

| Variable | Flag | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_THREADS` | `--threads` | positive integer (default: the available cores) | Threads for one FlexTool run: FlexTool's data processing and the solver. polars uses at most this many and at most 8; HiGHS gets `threads` set to it (its parallel simplex stays off); a Benders solve runs at most this many region solves at once. Unset: polars uses up to 8 threads and HiGHS and Benders keep their defaults. |
| `FLEXTOOL_HIGHS_THREADS` | `--highs-threads` | positive integer (default `1`) | HiGHS solver threads. More than 1 turns on HiGHS' parallel mode, which is faster but not bit-for-bit reproducible. |
| `FLEXTOOL_HIGHS_PRESOLVE` | `--presolve` | `on` / `off` / `choose` | HiGHS presolve. Unset keeps `on`. |
| `FLEXTOOL_SOLVER_TIME_LIMIT` | `--solver-time-limit` | seconds | Time limit of each solve, a positive number, on every solver (HiGHS `time_limit`, Gurobi `TimeLimit`, CPLEX `timelimit`, Xpress `MAXTIME`, COPT `TimeLimit`). Unset means no limit; 0 or a negative value is ignored with a warning, for the flag and the variable alike. |
| `FLEXTOOL_SOLVER_MIP_GAP` | `--solver-mip-gap` | relative gap, e.g. `0.005` | MIP gap for every solve of the run, on every solver (HiGHS `mip_rel_gap`, Gurobi `MIPGap`, CPLEX `mip tolerances mipgap`, Xpress `MIPRELSTOP`, COPT `RelGap`). Unset: each solve's `solver_mip_gap`, else the gap in the solver's option file. |
| `FLEXTOOL_HIGHS_TIME_LIMIT` | none | seconds | The former name of `FLEXTOOL_SOLVER_TIME_LIMIT`, now for HiGHS only: it never limits a commercial solver. The flag and `FLEXTOOL_SOLVER_TIME_LIMIT` win over it. |
| `FLEXTOOL_HIGHS_MIP_GAP` | none | relative gap | The former name of `FLEXTOOL_SOLVER_MIP_GAP`, now for HiGHS only. The flag and `FLEXTOOL_SOLVER_MIP_GAP` win over it. |
| `FLEXTOOL_SOLVER_LOG_LEVEL` | `--solver-log-level` | `silent` / `normal` / `verbose` | Solver log output (default `normal`); `silent` also hides a commercial solver's log (still saved as `solve_data/subprocess/<solve>.log`); `verbose` is HiGHS only. |
| `FLEXTOOL_SCALING` | `--scaling` | `off` / `solver_only` / `basic` / `full` | Autoscaler strategy for every solve; overrides each solve's `scaling` (default `full`). See [LP scaling pipeline](dev/scaling.md). |
| `FLEXTOOL_USER_BOUND_SCALE` | `--user-bound-scale` | integer in `[-10, 0]` | Pins HiGHS' `user_bound_scale` exponent; unset lets the autoscaler choose. |
| `FLEXTOOL_SAVE_MEMORY` | `--save-memory` | `1` | Solve each LP in a separate process to lower peak memory (slower). |
| `FLEXTOOL_WARM_START` | `--warm-start` | `1` | Reuse a cached HiGHS basis for solves of the same model structure, both in-process and with `--save-memory` (not for Benders solves). |
| `FLEXTOOL_BASIS_CACHE_DIR` | `--basis-cache-dir` | folder | Where the warm-start basis cache is kept (default `<work folder>/basis_cache`). Share one folder to reuse bases between runs. |
| `FLEXTOOL_SOLVER_CONFIG_DIR` | `--solver-config-dir` | folders, separated by `:` (Windows `;`) | Where a run takes its solver option files (`highs.opt`, `gurobi.opt`, ...) from: each solver's file from the first folder that has it, else FlexTool's default. For the FlexTool window it replaces the workspace's folder (a project's own `solver_config/` stays first). Wins over the search of a run without it: `solver_config/` next to the input database, the project's and then the workspace's of a GUI project database, in the Spine Toolbox project folder, in the current folder. The run copies the files it used to `<work folder>/solve_data/solver_config/`. See [Where the option files come from](solvers/index.md#where-the-option-files-come-from). |
| `FLEXTOOL_BENDERS_WORKERS` | `--benders-workers` | positive integer | Threads for the region solves of a Benders solve. Unset or `0`: the number of regions, at most the number of cores minus one. |

## Process and GUI settings

| Variable | Set to | What it does |
|---|---|---|
| `FLEXTOOL_NO_ENV_CHECK` | `1` | Skip FlexTool's start-up checks of the native libraries: the processor check at `import flextool` (which offers to install the compatible polars build on a processor without AVX2) and the GUI's start-up probe. With the checks skipped, polars may crash on such a processor. |
| `FLEXTOOL_NO_UPDATE_CHECK` | `1` | The GUI does not look for a newer FlexTool at start. |
| `FLEXTOOL_DPI` | DPI value, e.g. `144` | GUI screen resolution, when the automatic detection is wrong. |
| `FLEXTOOL_SLICE` | systemd slice name | Linux: GUI-started runs go into this systemd slice (needs `systemd-run`). |
| `MALLOC_ARENA_MAX` | integer | Linux: cap on glibc's malloc arenas. The FlexTool engine (what `flextool-run` runs) sets `4`, which keeps memory from fragmenting over long runs, unless you set a value. |
| `FLEXTOOL_RUN_LOG_MAX_BYTES` | bytes (default `4194304`, 4 MiB; at most `268435456`, 256 MiB) | Size limit of the run log `_run.log` in each run's `output_parquet/<run>/` folder. A longer log keeps its first 64 KiB and its last this many bytes, with a line that says how many bytes were left out between them. A larger value is taken as 256 MiB. The GUI's log window shows at most 2 MiB of a log: its start and its end, with a line that says how much is left out. |
| `FLEXTOOL_MIGRATION_LOCK_TIMEOUT` | seconds, a positive number (default `600`, 10 minutes) | Windows: how long a migration of an input database (`--migrate`, `flextool-migrate-database`, ...) waits for another migration of the same database to finish before it stops with a message naming the lock file `<database>.migrate.lock`. Not read on Linux or macOS, where the wait has no limit. See [Terminal workflow](terminal_workflow.md). |
| `POLARS_PREFER_PKG` | `64` / `32` / `compat` | polars' own: which polars build loads first. On a processor with AVX2, `import flextool` sets the fast build when the compatible build is installed too; a value you set is kept. |
| `POLARS_FORCE_PKG` | `64` / `32` / `compat` | polars' own: load only this build. Also skips FlexTool's processor check. |
| `POLARS_MAX_THREADS` | integer | polars' own: size of polars' thread pool. FlexTool uses it in place of its limit of 8, but never above the run's `--threads` / `FLEXTOOL_THREADS`. Set it before FlexTool starts; the run header prints the count polars uses. |

## AI assistant

The assistant has no variables of its own: its choices are flags of `flextool-assistant` and its settings. It
reads only the key of the AI service you chose, from that service's variable. A key in the variable wins over a
key stored in the system keyring or in the key file, and the chat says where its key came from.

| Variable | Set to | What it does |
|---|---|---|
| `OPENAI_API_KEY` | key | The key of OpenAI for the assistant chat. It wins over a key stored in the system keyring or the key file. |
| `ANTHROPIC_API_KEY` | key | The key of Anthropic for the assistant chat. It wins over a key stored in the system keyring or the key file. |
| `MISTRAL_API_KEY` | key | The key of Mistral for the assistant chat. It wins over a key stored in the system keyring or the key file. |
| `OPENROUTER_API_KEY` | key | The key of OpenRouter for the assistant chat. It wins over a key stored in the system keyring or the key file. |
| `GEMINI_API_KEY` | key | The key of Google Gemini for the assistant chat. It wins over a key stored in the system keyring or the key file. |
| `OLLAMA_CONTEXT_LENGTH` | tokens, e.g. `16384` | Ollama's own: the context length Ollama loads models with. The assistant needs at least 12288 (a smaller context sends no question); set it where the Ollama server starts and restart Ollama. |
| `PYTHON_KEYRING_BACKEND` | backend class | keyring's own: which keyring backend stores the keys. FlexTool does not use a backend that stores keys unencrypted. |

The provider libraries the assistant uses read more variables of their own: `OPENAI_BASE_URL`, `OPENAI_ORG_ID`,
`OPENAI_PROJECT_ID`, `OPENAI_CUSTOM_HEADERS`, `OPENAI_LOG`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_CUSTOM_HEADERS` and `ANTHROPIC_LOG`. FlexTool ignores them: it gives each library the address and the
key itself and sends only the request headers it needs, so these variables change nothing. Proxy variables
(`HTTPS_PROXY`, `NO_PROXY`, ...) and the system proxy settings apply only to services on the internet, never to
a model on this computer or your local network; the chat names the proxy it uses.

FlexTool also reads developer variables (diagnostics, profiling, Benders experiments, test switches). They are
described in the developer guide, are not meant for production runs and can change between releases.
