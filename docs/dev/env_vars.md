# Environment variables (developer)

Developer-only switches: diagnostics, profiling, experiments and checks. They are not part of the user
interface, their names and meaning can change between releases, and user pages do not link here. The variables
users set — the twins of `run_flextool` flags and the process / GUI settings — are on the user page
[Environment variables](../env_vars.md). Test-suite switches are in [Testing](testing.md).

Every variable FlexTool reads is registered in `flextool/run_options.py`: the twin of a flag in `OPTIONS`, every
other one in `ENV_KNOBS` with its audience (`user`, `dev`, `test`) and its page. Developer variables keep their
names (no `FLEXTOOL_DEV_*` prefix); the registry and this page mark them developer-only.
`tests/test_env_knob_registry.py` fails when FlexTool reads a variable that is not registered, when a registered
variable is missing from its page, or when a page lists one that is no longer read. A new variable therefore
needs an `ENV_KNOBS` entry and a row on its page.

How options are resolved (one resolver, flag > environment variable > database > default, nothing writes the
environment) is in [architecture.md, "Runtime options"](architecture.md#runtime-options).

For polar-high's own variables (`POLAR_HIGH_WRITE_MPS_PROFILE`, ...), see the
[polar-high environment-variables page](https://nodal-tools.github.io/polar-high/guide/env-vars/).

## Solve acceptance

`engine/solve/_acceptance.py` decides whether a solve that HiGHS did not certify optimal is accepted.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_ACCEPT_NEAR_OPTIMAL` | on | `0` | `0` rejects every solver status other than optimal (the old hard failure), a MIP stopped by a limit included, e.g. for strict test runs. On by default: an interior-point solve stopped without crossover (status Unknown) is accepted when the checks below pass. |
| `FLEXTOOL_ACCEPT_PD_GAP` | `1e-2` | positive float | Largest relative primal-dual gap of an accepted near-optimal solve. |
| `FLEXTOOL_ACCEPT_PRIMAL_REL` | `max(1e-6, 10 × primal_feasibility_tolerance)` | positive float | Largest relative primal infeasibility of an accepted near-optimal solve. Unset, it follows the tolerance the solve ran with; a set value replaces that derived ceiling (tighter or looser). |
| `FLEXTOOL_ACCEPT_PRIMAL_ABS_MARGIN` | `10` | positive float | Absolute primal infeasibility allowed, as a multiple of HiGHS' primal feasibility tolerance (HiGHS checks feasibility on the scaled LP, so the unscaled slack can exceed it). |

## Benders

Read by `engine/solve/_benders.py`. A malformed value is ignored with a warning.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_BENDERS_IN_OUT_WEIGHT` | the solve's `benders_in_out_weight` (`0`) | float in `[0, 1)` | Developer override of the in-out stabilization weight λ (weight of the interior centre in the separation point). `0` is off. No flag: it changes convergence, so production runs set it in the database. |
| `FLEXTOOL_BENDERS_TRUST_REGION_RADIUS` | unset (off) | positive float | Trust-region stabilizer: absolute box half-width on the coupling flows (normalised units). Not combined with in-out. |
| `FLEXTOOL_BENDERS_MAX_STALL` | `8` | positive integer | Iterations without progress before the stall guard stops the loop. The guard cannot be turned off. |
| `FLEXTOOL_BENDERS_CUT_COMPACT_AT` | unset / `0` (off) | positive integer | Master cut count at which the master is compacted to its binding cuts. |
| `FLEXTOOL_BENDERS_CUT_POLICY` | `slack` | `slack` / `dominance` | Compaction policy: drop cuts slack at the current optimum, or keep per recourse column the oldest maximum over the last trial points. |
| `FLEXTOOL_BENDERS_CUT_WINDOW` | `10` | positive integer | Trial-point window of the `dominance` policy. |
| `FLEXTOOL_BENDERS_REGION_DUALS` | `basic` | `basic` / `interior` | Duals for the cut slopes: vertex duals with crossover, or barrier duals without crossover (a central point of a degenerate dual face; no warm start). |
| `FLEXTOOL_BENDERS_VERBOSE` | unset | any value | Show the HiGHS log of every master and region solve (by default they are silent and the per-iteration bound log shows progress). |
| `FLEXTOOL_BENDERS_PIN_DIAGNOSTIC` | unset | parquet path | Pin diagnostic: instead of iterating, evaluate the decomposition once at the monolith's coupling flows (columns `p, source, sink, d, t, value`) and report whether a region's recourse blows up. |
| `FLEXTOOL_BENDERS_PIN_MONOLITH_OBJ` | unset | float (real units) | Monolith objective for the pin diagnostic's exactness check (Σ region cost + master ≈ monolith). |
| `FLEXTOOL_BENDERS_PIN_BLOWUP_MULT` | `100` | float | Pin diagnostic: a region whose cost exceeds this multiple of its stand-alone cost counts as blown up. |

## Checks and modes

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_AUTOSCALE_STRICT` | unset (off) | `1` | A variable, constraint or parameter missing from the autoscale registries raises instead of the solve silently falling back to an unscaled LP. Set it in tests that exercise autoscale (CLAUDE.md invariant 1). |
| `FLEXTOOL_AXIS_ENUMS` | unset (off) | `1` | Experimental axis-Enum mode for the whole run (axes typed as polars Enums instead of Utf8). Development and tests only. |
| `FLEXTOOL_FORCE_ROW_SCALING` | unset | `1` / `true` / `yes` / `on` | Turn on the legacy node / group balance row scaling (`p_use_row_scaling=1`); every solve has it off otherwise. Used by the un-scaling benchmark. |
| `FLEXTOOL_SKIP_SOLVER_PROBE` | unset | `1` | Skip the start-up probe of commercial solver licences (avoids the FICO Xpress Community licence warning). The test conftest sets it. |
| `FLEXTOOL_NATIVE_SCENARIO` | unset | scenario name | Scenario of `run_chain` when neither the work folder name (`work_<scenario>`) nor an argument gives it. |
| `FLEXTOOL_AUDIT_SOURCES` | unset (off) | `1` | Append every Provider entry with a `source` tag (entries injected between solves) to `<work_folder>/audit_sources.log`, per sub-solve. |
| `FLEXTOOL_LEVEL_RELEASE_AUDIT` | unset (off) | `1` | Record solves whose HiGHS instance is still alive after its level is finished (memory-release audit). |
| `FLEXTOOL_DUMP_CSVS` | unset (off) | `1` | With `--csv-dump`, also write the eight largest debug CSVs (`p_flow_max`, `pdtProcess*`, `pdtNode*`, `ptNode_inflow`), which are not read by any output step. |
| `FLEXTOOL_COLD_KEEP_PROVIDER` | unset (off) | `1` | Under `--save-memory`, keep the data provider between cold solves instead of rebuilding it (less database reading, higher steady memory). |
| `FLEXTOOL_RSS_BUDGET_MB` | unset (no eviction) | MB | RSS above which the data provider evicts cached frames (subject to their lifetime). |
| `FLEXTOOL_WARM_AB_MARGIN` | `0.10` | float | Warm-start first-transfer test: the basis is kept when the warm solve takes at most `1 + margin` times the cold solve; a negative value forces the "slower" verdict (tests). |
| `FLEXTOOL_DISABLE_AUTOSCALE_CACHE` | unset | `1` | Recompute the autoscale plan in every roll instead of reusing it (A/B check). |
| `FLEXTOOL_DISABLE_OUTPUT_HOIST` | unset | `1` | Fetch the solution arrays once per output instead of once per solve (A/B check). |

## Memory and timing diagnostics

All off by default, with no overhead when unset.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_MEMORY_DIAGNOSTICS` | unset (off) | `1` | Per-checkpoint memory recorder with `tracemalloc`; writes `solve_data/memory_diagnostics.csv`. `--debug` sets it. |
| `FLEXTOOL_MEMORY_VERBOSE` | unset (off) | `1` | Print every phase in the `[mem]` table, not only the main ones. `--debug` sets it. |
| `FLEXTOOL_MEM_SAMPLER` | unset (off) | `1` / `true` / `yes` | Background thread that samples RSS at a fixed interval; catches peaks between checkpoints. |
| `FLEXTOOL_MEM_SAMPLER_LOG` | `/tmp/flextool_mem_sampler_<pid>.log` | path | Sampler output. |
| `FLEXTOOL_MEM_SAMPLER_INTERVAL_MS` | `100` (20–10000) | milliseconds | Sampler interval. |
| `FLEXTOOL_PHASE_TIMING` | unset (off) | `1` | Per-iteration rows (`preprocessing` / `lp_build` / `solve` / `handoff`, `warm_used`) in `timings.csv`. `warm_used` is a flag in the `seconds` column (1 = warm start used), not a time. Always-on rows: `read_input`, `preprocessing_global`, `roll_setup`, `all_solves`, `write_outputs/*`, `total`. |
| `FLEXTOOL_PHASE_PROFILE` | unset (off) | `1` | One epoch-stamped RSS line per orchestration phase on stderr (aligns with the sampler log). |
| `FLEXTOOL_PYRAMID_PROFILE` | unset (off) | `1` | One `[pyramid profile]` RSS line per preprocessing emit on stderr. |
| `FLEXTOOL_AUTOSCALE_PROFILE` | unset (off) | `1` | RSS markers around the autoscale steps. |
| `FLEXTOOL_BUILD_PROFILE` | unset (off) | `1` | One RSS line per model-build step on stderr. |
| `FLEXTOOL_LOADFLEX_PROFILE` | unset (off) | `1` | One RSS line per data-load step on stderr. |
| `POLAR_HIGH_LAYER2_PROFILE` | unset (off) | `1` | One `[layer2-bucket profile]` line per term of the autoscale Layer 2 coefficient walk. Read by FlexTool's `engine/autoscale/_layer2.py`, not by polar-high. |
| `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS` | `1000000` | integer | Only with a polar-high that has no bounded coefficient walk: constraint families with more rows are skipped in Layer 2 bucketing. Read by FlexTool's `_layer2.py`. |

Example — find a transient RSS peak the per-phase recorder misses:

```bash
FLEXTOOL_MEM_SAMPLER=1 FLEXTOOL_MEM_SAMPLER_INTERVAL_MS=50 \
    python run_flextool.py input.sqlite output_info.sqlite --scenario-name base
```

## Run record

Each run writes `_run_status.json` and `_run.log` into its `output_parquet/<run>/` folder. `_run.log` is written by
the process that holds the run's output pipe, when there is one.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_RUN_LOG_OWNER` | unset | `1` | Set by the GUI and `flextool-run` in the environment of the engine run they start (only that direct child; the engine removes it, so no run it starts sees it). It tells the engine that its parent writes `_run.log` from the output pipe, with the native solver lines and crash dumps: the engine prints one marker line for the parent instead of copying its own console output into the log. Not for use by hand. |
| `FLEXTOOL_RUN_STOP_FILE` | unset | a file path | Set by the GUI in the environment of a calibration or representative-period job, and inherited by every `flextool-run` supervisor that job starts. When the file appears (the GUI writes the reason into it when the job is stopped), the supervisor kills its engine and records the run as `killed` with that reason; a supervisor that finds it before its engine starts runs nothing. The supervisor leaves it out of the engine's environment. Not for use by hand. |

## User folders

FlexTool keeps its per-user files (the AI assistant's settings, saved chats, usage ledger, key file and caches) in
the platform's user config and cache folders, found with `platformdirs`.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_USER_HOME` | unset (the platform's folders) | folder | FlexTool's user config folder becomes `<folder>/config` and its user cache folder `<folder>/cache`. The test suite sets it for every test process and every process a test starts (`tests/user_dirs_guard.py`), so no test reads or writes the real user folders. |

## Process level

Set when `flextool` is imported, with `setdefault` semantics: a value already in the environment wins.

| Variable | FlexTool sets | What it does |
|---|---|---|
| `POLARS_ENGINE_AFFINITY` | `in-memory` | polars' engine choice. FlexTool's results are verified only on the in-memory engine; the streaming engine changes row order and summation order. |
| `_RJEM_MALLOC_CONF` | `dirty_decay_ms:1000,muzzy_decay_ms:0` | Settings of polars' jemalloc allocator: return freed pages to the system, which roughly halves peak memory on long runs. When polars was imported before flextool, FlexTool sets the decay at run time instead. |
