# Environment variables

Reference for every environment variable FlexTool reads at runtime,
grouped by what they do. All variables are optional — defaults
preserve the user-facing behaviour and add zero overhead.

Two reasons to set any of them:

- **Functional toggles** — change what FlexTool does (autoscale mode,
  save-memory subprocess solver, HiGHS options).
- **Diagnostics** — record memory or timing for post-hoc analysis.

CLI flags on `run_flextool.py` take precedence and, where they map to
an env var, set it for the rest of the process. Direct env-var use is
the right path when you cannot pass a flag (Spine Toolbox execution,
GUI subprocess, CI matrix).

For the `polar_high` solver kernel's own env vars (profiling MPS
writes, autoscale range readout cap), see the
[polar-high environment-variables page](https://nodal-tools.github.io/polar-high/guide/env-vars/).

## Functional toggles

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_SCALING` | `full` | `off` / `solver_only` / `basic` / `full` | Autoscale mode. Equivalent to `--scaling <mode>`. `off` skips Layers 2/3 (Layer 1 always runs for the YAML audit). See [LP scaling pipeline](scaling.md). |
| `FLEXTOOL_USER_BOUND_SCALE` | unset | integer (e.g. `6`) | Pin the HiGHS Layer-3 `user_bound_scale` exponent instead of auto-picking. Equivalent to `--user-bound-scale N`. |
| `FLEXTOOL_SAVE_MEMORY` | unset (off) | `1` | Enable the polar-high subprocess-solver path (writes MPS, shells out, frees parent-process LP buffers before the solve). Equivalent to `--save-memory`. Trades wall time for peak RSS. |

## HiGHS tuning

Each maps to a HiGHS option threaded through
`_orchestration._finalise_highs_options`. Setting the env var is
equivalent to passing the matching CLI flag on `run_flextool.py`.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_HIGHS_TIME_LIMIT` | HiGHS default | seconds (float) | HiGHS `time_limit` option. Hard wall-clock cap per solve. |
| `FLEXTOOL_HIGHS_PRESOLVE` | HiGHS default (`on`) | `on` / `off` / `choose` | HiGHS `presolve` option. Equivalent to `--presolve`. Useful when comparing warm-vs-cold solves. |
| `FLEXTOOL_HIGHS_THREADS` | HiGHS default | positive integer | HiGHS `threads` option. Equivalent to `--highs-threads`. |

## Memory diagnostics

All diagnostics are off by default and impose zero overhead when
unset.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_MEMORY_DIAGNOSTICS` | unset (off) | `1` | Enable the per-checkpoint memory recorder. Activates `tracemalloc` (so `traced_peak` is meaningful) and writes `solve_data/memory_diagnostics.csv` for post-hoc analysis. `--debug` sets this. |
| `FLEXTOOL_MEMORY_VERBOSE` | unset (off) | `1` | Drop the whitelist filter on the standard `[mem]` log table so every phase checkpoint is printed. `--debug` sets this. |
| `FLEXTOOL_AUTOSCALE_PROFILE` | unset (off) | `1` | Per-substep RSS markers around the autoscale orchestration. Complements `POLAR_HIGH_RANGES_PROFILE` from polar-high — set both when range detection itself is suspect. |
| `FLEXTOOL_PYRAMID_PROFILE` | unset (off) | `1` | Wired through `_MemoryRecorder.checkpoint()`: after every `emit_*` call in `_emit_solve_time.run` (the per-solve preprocessing pyramid), emit a tab-separated stderr line in the polar-high `[pyramid profile] phase=… rss_gb=… delta_gb=… t_s=…` format (precedent: `POLAR_HIGH_WRITE_MPS_PROFILE`). Zero overhead when unset; combine with `FLEXTOOL_MEMORY_DIAGNOSTICS=1` to also persist a CSV trace. |
| `FLEXTOOL_PHASE_TIMING` | unset (off) | `1` | Per-iter `per_iter` rows added to the workdir's `timings.csv` covering `lp_build` / `solve` / `handoff` plus a `warm_used` marker. Use for rolling / warm-chain wall-time breakdowns. |
| `FLEXTOOL_MEM_SAMPLER` | unset (off) | `1` (or any truthy: `true`, `yes`) | Start an in-process daemon-thread RSS sampler at startup. Writes one row per sample to the log path below. Independent of the phase recorder — useful when the recorder's coarse checkpoints miss a transient peak. |
| `FLEXTOOL_MEM_SAMPLER_LOG` | `/tmp/flextool_mem_sampler_<pid>.log` | filesystem path | Sampler output path. Only consulted when the sampler is on. |
| `FLEXTOOL_MEM_SAMPLER_INTERVAL_MS` | `100` (clamped 20–10000) | integer milliseconds | Sampler cadence. Tighter intervals catch shorter peaks at the cost of more log volume. |

Example — find a transient RSS peak the per-phase recorder is missing:

```bash
FLEXTOOL_MEM_SAMPLER=1 FLEXTOOL_MEM_SAMPLER_INTERVAL_MS=50 \
    python run_flextool.py input.sqlite output_info.sqlite --scenario-name base
```

## Memory tuning

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `MALLOC_ARENA_MAX` | `4` (set by `flextool/cli/cmd_run_flextool.py` at startup) | integer (e.g. `1`, `2`, `8`) | Cap on per-thread glibc malloc arenas. Linux-only. Defaults to `4` because polars / Arrow allocate-and-free in patterns that otherwise fragment per-thread arenas into a large RSS footprint that is not reclaimed between solves. Lower (`1` or `2`) further trims long-running rolling-horizon RSS at a small allocator-contention cost; higher relaxes the cap if you suspect arena contention is dominating. Not a FlexTool-defined variable — the entry point sets a default via `os.environ.setdefault`, so any value already in the environment wins. |
| `FLEXTOOL_COLD_KEEP_PROVIDER` | unset (off) | `1` | Under `--save-memory`, keep the `flex_data_provider` alive across cold iterations instead of rebuilding from the Spine DB each time. Trades steady-state RSS for skipping a per-iter DB re-read. Off by default because peak RSS is the typical save-memory bottleneck. |
| `FLEXTOOL_RSS_BUDGET_MB` | unset (no eviction) | MB (float) | Soft RSS budget for the `flex_data_provider`'s lazy-frame eviction. When set, the provider drops cached frames once RSS crosses the threshold (subject to lifetime info). Useful on long-running workflows where intermediate frames accumulate; unset gives unbounded retention. |

## Precision cleanup

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_PRECISION_DIGITS` | unset (passthrough) | integer 1–15 | Round every numeric CSV cell written to `input/` / `solve_data/` to N significant figures. Env-var fallback for `--precision-digits N`. `0` or unset disables. CLI takes precedence. See [LP scaling pipeline § Precision rounding](scaling.md#precision-rounding). |
| `FLEXTOOL_REPORT_NEAR_DUPS` | unset (off) | `1` / `true` / `yes` / `on` | Run the near-duplicate parameter-value diagnostic. Off by default; equivalent to `--report-near-duplicates`. Surfaces clusters of values that are nearly-but-not-exactly equal — useful before tightening `FLEXTOOL_PRECISION_DIGITS`. |

## Niche / test hooks

These are documented for completeness. Most users will never set
them.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_NATIVE_SCENARIO` | unset | scenario name | Override for `run_chain`'s scenario-name resolution when neither the dir-name convention (`work_<scenario>`) nor an explicit kwarg applies. |
| `FLEXTOOL_SOLVER_CONFIG_DIR` | `<cwd>/solver_config` | directory path | Override the lookup directory for `highs.opt` and friends used by the subprocess-solve path. Matches the existing hook honoured by CI / tests. |
| `FLEXTOOL_DUMP_CSVS` | unset (off) | `1` | Restore the legacy debug-oracle CSV writes — emits the seven gigabyte-scale CSVs that `dump_csvs` skips by default. Used by the round-trip regression test; for normal runs the default (skip) saves significant disk I/O. |
| `FLEXTOOL_AUDIT_SOURCES` | unset (off) | `1` | Append every Provider key carrying a non-None `source` tag to `<work_folder>/audit_sources.log` per sub-solve. Captures externally-injected entries from the override translator. |
| `FLEXTOOL_FORCE_ROW_SCALING` | unset | `1` / `true` / `yes` / `on` | Test hook that forces `p_use_row_scaling=1`. Since the `solve.use_row_scaling` DB parameter was removed, every solve emits `p_use_row_scaling=0` by default and this hook is the only way to enable the legacy row-scaling family. Used by the Mode B un-scaling benchmark harness. |
| `FLEXTOOL_SKIP_SOLVER_PROBE` | unset | `1` | Skip the startup solver-license probe (avoids the FICO Xpress Community LicenseWarning). Set in the test conftest. |
| `FLEXTOOL_SLICE` | unset | systemd slice name | When set and `systemd-run` is available, place the worker process in the named cgroup slice for scheduling isolation. GUI-launched runs only. |
| `FLEXTOOL_DPI` | unset (auto-detect) | DPI value (e.g. `144`) | GUI-only. Explicit DPI override when desktop-environment auto-detection (`Xft.dpi`, `GDK_SCALE`, Windows ctypes query) gives the wrong value. |

## See also

- [LP scaling pipeline](scaling.md) — the full autoscale story for
  `FLEXTOOL_SCALING` and `FLEXTOOL_USER_BOUND_SCALE`.
- [polar-high — Environment variables](https://nodal-tools.github.io/polar-high/guide/env-vars/) —
  the kernel-side `POLAR_HIGH_*` profiling and tuning variables.
  `FLEXTOOL_SAVE_MEMORY=1` enables the path where
  `POLAR_HIGH_WRITE_MPS_PROFILE=1` is most informative.
