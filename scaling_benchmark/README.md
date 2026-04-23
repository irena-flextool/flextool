# FlexTool LP-scaling benchmark harness

Reproducible benchmark scenarios and a harness for the LP/MIP scaling work
described in
`~/.claude/projects/-home-jkiviluo-sources-flextool/memory/project_lp_scaling_2026-04.md`.
Every agent in the 11-agent sequential plan uses this to verify "no numerical
regression" before proceeding.

See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for the Agent 11 final
validation matrix (4 scenarios x 3 modes) and
[`baseline/CHANGELOG.md`](baseline/CHANGELOG.md) for the baseline refresh log.

## The four scenarios

Each lives under `scaling_benchmark/scenarios/<name>/` and has a `generate.py`
that writes `input.sqlite` next to itself, and declares a module-level
`SCENARIO_NAME` naming the scenario inside that DB that the harness should
execute.

| Name              | Source DB                      | Scenario in DB              | Shape (rows/cols/nnz) | Matrix range   |
|-------------------|--------------------------------|-----------------------------|-----------------------|----------------|
| `small_building`  | `templates/examples.sqlite`    | `wind_battery`              | ~430 / 340 / 770      | `[1, 1e3]`     |
| `medium_national` | `templates/examples.sqlite`    | `network_all_tech`          | ~2500 / 1900 / 5000   | `[0.5, 1e3]`   |
| `continental`     | `rivendell/rivendell.sqlite`   | `continental_benchmark`     | ~1000 / 560 / 2000    | `[1, 6e5]`     |
| `composite`       | `templates/examples.sqlite`    | `composite_benchmark`       | ~2900 / 2200 / 5700   | `[3e-3, 2e4]`  |

- **`small_building`** — reuses the existing `wind_battery` scenario (2-day
  48h dispatch, wind + battery + west node). Dispatch-only, single period.
- **`medium_national`** — reuses the existing `network_all_tech` scenario
  (multi-node network with coal, wind, CHP, EV, water pump, heat storage,
  demand response). Mix of LP and UC (MIP).
- **`continental`** — copies `rivendell/rivendell.sqlite` (14 nodes, 23
  units, 3 connections, 3-region system) and adds a
  `benchmark_cap_2_periods` alternative that trims `period_timeset`,
  `realized_periods`, `invest_periods`, and `years_represented` to the
  first 2 periods. Full 32-period rivendell already solves in ~2 s; the
  capped version produces a smaller matrix still representative of the
  structure.
- **`composite`** — the scenario the scaling work is primarily built for.
  Copies `templates/examples.sqlite`, layers a `composite_scales`
  alternative on top of `network_all_tech` that adds a tiny building node
  with 3 kW constant demand, a 0.01 MW heatpump, a tiny battery storage
  node (0.0025 MWh reference), and two oversized continental units
  (10 000 MW coal, 5 000 MW wind proxy). Matrix range spans ~7 orders
  of magnitude by construction.

## Usage

```bash
source ~/venv-spi/bin/activate

# Generate input.sqlite files (once, or after changing a generator)
python scaling_benchmark/run_benchmarks.py --generate

# Run all scenarios and (re)write baselines
python scaling_benchmark/run_benchmarks.py --write-baseline

# Run one scenario
python scaling_benchmark/run_benchmarks.py --scenario composite --write-baseline

# Compare a fresh run to an existing baseline (exit 2 if material delta)
python scaling_benchmark/run_benchmarks.py --scenario composite \
    --compare scaling_benchmark/baseline/composite.json
```

Flags:

- `--generate` regenerates `input.sqlite` files via the four
  `scenarios/*/generate.py`. Safe to run; they're deterministic.
- `--write-baseline` writes `scaling_benchmark/baseline/<scenario>.json`
  with `objective`, `matrix_range`, `cost_range`, `bound_range`,
  `rhs_range`, `rows/cols/nnz` (initial and post-presolve when HiGHS
  reports them), `matrix_range_from_mps` (independent scan of the emitted
  `flextool.mps`), `solve_wall_time_s` (HiGHS solver time as reported by
  `flextool`'s runner), `slack_totals` for all seven `vq_*` slacks,
  plus `timestamp` and `git_commit`.
- `--compare <baseline.json>` prints per-field deltas and exits 2 if any
  numerical field differs materially (wall time is reported but not
  flagged since it jitters).
- `--keep-work` keeps `scaling_benchmark/work/<name>/` (parquet,
  `HiGHS.log`, `flextool.mps`) for debugging.

## What the harness captures

- **Matrix range** — parsed from the `Coefficient ranges:` block in
  `HiGHS.log`. An independent scan of `flextool.mps` COLUMNS is also
  recorded as `matrix_range_from_mps` for cross-checks (values can differ
  slightly because HiGHS log ranges reflect the LP as HiGHS sees it after
  its own construction; MPS is the unfiltered emitted matrix).
- **Objective** — read from `output_raw/v_obj__*.parquet` (the HiGHS
  extractor) and summed across any rolling solves.
- **Slack totals** — summed absolute values from
  `output_raw/vq_*__*.parquet` across every shard/solve. Later agents
  watch these to catch behaviour changes from the slack-convention
  rewrite (Agents 2-4).

## Generated artefacts

- `scaling_benchmark/scenarios/<name>/input.sqlite` — per-scenario DB
  (not committed; `.gitignore`d).
- `scaling_benchmark/baseline/<name>.json` — committed baselines.
- `scaling_benchmark/work/<name>/` — ephemeral per-scenario work folder
  (HiGHS.log + MPS kept, output parquets dropped unless `--keep-work`).
