# Testing

FlexTool has a multi-layer test pyramid: fast unit/integration tests on
the `engine_polars` building blocks, scenario goldens built on demand
from JSON fixtures, and end-to-end scenario runs that compare output
CSVs against committed reference files. This page is a developer
pointer — where the tests live, how the JSON-fixture architecture
works, how the CI verify chain enforces drift detection, and how the
markers gate CI.

For the high-level test-layout inventory see
[`architecture.md`](architecture.md) ("Test layout and fixture
architecture"). For the engine internals being tested see
[`engine_polars.md`](engine_polars.md). For the JSON-fixture rollout
narrative see the `CHANGELOG.md` entry for v3.47.0.

## Layer 1 — engine_polars unit & integration

Lives under `tests/engine_polars/`. Covers the LP build path, derived
parameter expansion, writer port, scaling pipeline (see
[`scaling.md`](scaling.md)), decomposition (see
[`decomposition.md`](decomposition.md)), warm-start handoff, fast load,
stochastic / PBT pieces, and the Surface A audit subdirectories:

- `loaders/` — A-tier loader tests (`test_a01_a02_temporal_node.py`,
  `test_a03_a11_a12_process_topology.py`, …) covering temporal-node,
  topology, online-ramp, storage-handoff, invest, costs/CO2,
  user-cstr/stochastic, groups/reserve/ladder, delay/dc loaders.
- `constraints/` — B-tier constraint tests
  (`test_b01_b02_balance_storage.py` …
  `test_b09_b10_b11_dc_co2_user.py`, plus emission / storage-handoff
  wiring).
- `objective/` — B16–B20 objective-term tests (varcost, invest cost,
  slack & reserve penalties, delay/DR/ladder).

Each subdirectory has its own `conftest.py` and a
`test_fixtures_solve.py` that exercises the shared fixture solve
fixtures.

Tests range from millisecond-class unit tests (block layout, derived
params) to solver-touching parity tests. Synthetic flextool-flavoured
fixtures live in `tests/engine_polars/fixtures/` and are auto-added to
`sys.path` by `tests/engine_polars/conftest.py` so test modules can
`from flex_toy_<feature> import …` directly.

**Meta-provider invariants.**
`tests/engine_polars/test_meta_provider_invariants.py` enforces the
Provider-only cascade contract as executable tests, not convention:

- `test_no_disk_csv_reads_in_cascade` — fails if a cascade module
  re-introduces a `csv.reader` / `open(...).read()` against a CSV path
  without an explicit justification entry in the allowlist.
- `test_no_photocopier_lru_cache` — fails on module-level
  `@lru_cache` over frame-returning helpers (cross-test contamination
  trap).
- `test_no_module_level_frame_globals` — fails on module-level
  `pl.DataFrame` / `pl.LazyFrame` globals.
- `test_provider_or_shims_have_justification` — every disk read still
  in the cascade must be accompanied by a documented reason.
- `test_allowlists_cover_only_existing_files` — the allowlists
  themselves cannot drift past the files they reference.

A cascade module that reads from disk silently is a meta-test failure,
not a code-review catch.

## Layer 2 — scenario fixtures, goldens, and parity

### `tests/fixtures/*.json` is the single authoritative source

Every SQLite DB used by tests is built on demand from a JSON fixture.
The previous ~521 MB of tracked / gitignored SQLite + CSV fixtures has
collapsed to ~5 KB of JSON plus the on-demand cascade. Schema
migration covers the fixture set via
`flextool.update_flextool.test_fixtures.migrate_all`, and the CI
verify chain (see below) fails the build if any fixture drifts from
the current `FLEXTOOL_DB_VERSION`.

Canonical fixtures currently in `tests/fixtures/`:

| Fixture | Role |
|---|---|
| `tests.json` | Main fixture; `tests/scenarios.yaml` and most engine_polars tests build their work-folders from here. Also the source for the generated `templates_examples.json` canonical DB. |
| `stochastics.json` | Stochastic / forecast-branch scenarios. Source for the generated `howto_stochastics.json` canonical DB. |
| `stochastics_pbt_inflow.json` | PBT inflow scenarios (period-block-time stochastic inflow). |
| `lh2_three_region.json` | LH2 three-region monolithic + Lagrangian parity fixture (see [`decomposition.md`](decomposition.md)). |
| `h2_trade_parity.json` | H2-trade Lagrangian parity fixture. |
| `multi_ts_branch1.json` | Multi-timeseries branch-1 fixture. |
| `branch2_parent_period.json` | Branch-2 / parent-period stochastic fixture. |
| `case14_dc_power_flow.json` | PGLib IEEE case14 MATPOWER export (~8200 lines), used by `test_flex_dc_power_flow.py`. |

The `_augment_*.py` and `build_*.py` scripts alongside these JSONs are
one-shot builders / overlays kept for provenance, not runtime
dependencies.

### `scenario_workdir` — on-demand cascade fixture

`scenario_workdir(scenario_name, db_fixture="main")`, defined in
`tests/engine_polars/conftest.py`, is the session-scoped fixture that
replaced the historical `tests/engine_polars/data/work_*/` snapshot
pattern. For a requested `(scenario, db_fixture)` pair it:

1. Resolves `db_fixture` against the per-fixture session-scoped DB
   (built once per session from the matching JSON).
2. Runs the full cascade with `csv_dump=True` + `keep_solutions=True`
   in a `tmp_path_factory` workdir.
3. Snapshots the last orchestration step's `flex_data_provider` so the
   on-disk CSV layout matches what the legacy disk fixtures shipped.
4. Caches the resulting work-folder per `(scenario, db_fixture)` —
   each scenario pays its ~2 s cascade cost once per test session
   regardless of how many tests consume it.

Valid `db_fixture` keys map 1:1 to the JSONs listed above (`main`,
`stochastic`, `lh2`, `case14`, `h2_trade_parity`, `multi_ts_branch1`,
`stochastics_pbt_inflow`, `branch2_parent_period`).

```python
def test_foo(scenario_workdir):
    work = scenario_workdir("coal_ladder_annual", db_fixture="main")
    # work is a Path to a populated workdir with input/output CSVs.
```

The two `tests/engine_polars/data/work_<scenario>/` directories that
survive after Stage 4 (`work_lh2_three_region`, `work_test_a_lot`)
hold only a single `golden_obj.json` golden objective each. No
migrated test depends on the gitignored bulk CSV snapshot pattern;
the entry stays on `.gitignore` for back-compat only.

### `extend_tests_fixture` — agent-friendly append-only edits

`flextool.update_flextool.extend_tests_fixture` is a YAML-delta
workflow for *append-only* additions to `tests/fixtures/*.json` —
new entities, alternatives, scalar parameter values, and scenarios.
It exists because rewriting authoritative JSON by hand is risky:

- Validates every name against `spinedb_schema.json` with did-you-mean
  suggestions.
- Rejects edits to existing entries (append-only enforcement).
- Rejects structured (`Map` / time-series) values, pointing the caller
  at the SpineDB editor for those.
- Eight unit tests under `tests/spinedb_backend/` lock the contract.

Agents add to fixtures without rewriting them; structured / curated
edits remain a human-with-SpineDB-editor job.

### `generate_canonical` — projected canonical DBs

`flextool.update_flextool.generate_canonical` projects authoritative
fixtures into derived canonical JSONs. The recipes in
`flextool/schemas/canonical_databases/_recipes.yaml` declare two
projections today:

- `tests/fixtures/tests.json` → `templates_examples.json`
- `tests/fixtures/stochastics.json` → `howto_stochastics.json`

The remaining `howto_*.json` files plus
`templates_time_settings_only.json` stay individually authoritative —
purpose-specific minimal teaching examples whose curation goals
differ from test-coverage curation. `generate_canonical --verify`
fails when a projected canonical drifts from its source.

### Golden objectives & scenario goldens

`tests/scenarios.yaml` defines a parametrised end-to-end scenario
set that runs in `tests/test_scenarios.py`:

1. Build the session-scoped SQLite DB from
   `tests/fixtures/<fixture>.json`.
2. Run `FlexToolRunner` for the scenario in an isolated tmp workdir.
3. Compare selected output CSVs against
   `tests/expected/<scenario>/*.csv`.
4. If the scenario YAML entry sets `expected_objective`, assert that
   `summary_solve.csv` lands within `expected_objective_tolerance`
   (default `1e-3`, often tightened to `1e-5` for stable scenarios).
5. If `time_budget_seconds` is set, assert the full
   write-input → solve → write-outputs cycle finishes in budget.

Standalone golden integration tests live alongside (e.g.
`tests/test_lh2_three_region.py` pins the LH2 three-region monolithic
objective against `tests/expected/lh2_three_region/objective.json` at
relative tolerance `1e-4`). The same fixture also backs the native
Lagrangian gap-to-monolithic check — see
[`decomposition.md`](decomposition.md) and
[`slack_convention.md`](slack_convention.md).

## Layer 3 — full integration

The `tests/test_scenarios.py` driver above covers Layer 3 for the
in-tree scenarios (`tests/scenarios.yaml` parametrises it, goldens
live in `tests/expected/<scenario>/`). The separate `execution_tests/`
directory contains Spine Toolbox workflow execution tests
(`test_execution.py`, `tool.py`) that exercise the same scenarios
through the Toolbox DAG rather than the Python runner directly.

**Three-tier gate cadence** (matches the
[`architecture.md`](architecture.md) wording so contributors converge
on the same loop):

- **Inner loop** — module-specific tests after every edit, typically
  `pytest tests/engine_polars/<area>/`. Run continuously while
  iterating.
- **Broader gate** — `pytest tests/engine_polars/` full sweep, once
  per commit batch (~27 minutes on a recent reference run).
- **Activation truth** — `pytest tests/` (Layer 1 + scenario
  integration suite + `tests/model/` + `tests/gui/` +
  `tests/decomposition/` + `tests/spinedb_backend/`), run rarely
  (overnight / pre-release).

Inner loop ≠ confidence to merge; activation truth is the only gate
that signals end-to-end health.

## CI verify chain

`.github/workflows/tests.yml::template-check` enforces four drift
checks before any Linux/Windows/macOS test job runs. The same
commands are valid as pre-push sanity locally:

| Step | Command | What it checks |
|---|---|---|
| Master template parity | `python -m flextool.update_flextool.sync_master_json_template --verify` | `schemas/spinedb_schema.json` matches the migration chain (see [`db_schema.md`](db_schema.md)). |
| Test fixtures schema | `python -m flextool.update_flextool.test_fixtures verify` | All `tests/fixtures/*.json` are at the current `FLEXTOOL_DB_VERSION`. |
| Generated canonicals | `python -m flextool.update_flextool.generate_canonical --verify` | Generated canonical JSONs (`templates_examples.json`, `howto_stochastics.json`) match what `generate_canonical` would emit from the current authoritative sources. |
| Canonical schema | `python -m flextool.update_flextool.canonical_databases verify` | The authoritative canonical JSONs themselves are at the current schema. |

A drift in any of the four is a failed workflow, not a code-review
catch. See the [`architecture.md`](architecture.md) "verify hooks"
tail for the same chain in the higher-level inventory and
[`db_schema.md`](db_schema.md) for the migration-chain side of the
contract.

## Markers and gating

Defined in `pyproject.toml` and re-asserted in `tests/conftest.py`:

| Marker          | Meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `smoke`         | Layer-1 + scenarios on the per-commit CI gate (<1 min)        |
| `solver`        | Touches a real solver (HiGHS), 5–60 s each                    |
| `slow`          | >~30 s — nightly only                                         |
| `decomposition` | Tier 8 objective-decomposition parity tests                   |
| `perturbation`  | Tier 6 single-multiplier perturbation tests                   |
| `emission`      | Tier 7 MPS row-count emission tests                           |

The `smoke` marker is attached automatically to scenarios in
`scenarios.yaml` that set `smoke: true`.

`.github/workflows/ci.yml` defines the three-tier CI gate:

- **smoke** — every push and PR; runs
  `pytest -m "smoke or (not slow and not solver)"` with a 1-minute
  timeout.
- **full** — push to `master` and nightly cron; full Layer-1 plus
  Layer-2 MPS parity (30-minute timeout).
- **slow** — nightly only; `-m slow`.

## Running the tests

```bash
pytest -m smoke              # the CI smoke gate, fast
pytest -m "not solver"       # everything that doesn't touch a solver
pytest tests/engine_polars/  # broader gate — engine layer
pytest tests/                # activation truth (slow)
```

Regenerate a single scenario's golden CSVs in place:

```bash
pytest tests/test_scenarios.py --regenerate coal
```

This runs only the matching `scenario` parametrize-case, writes fresh
CSVs into `tests/expected/coal/`, and lets you commit the diff.

## Determinism guards

`engine_polars` emitters keep cross-solve determinism through explicit
polars sort keys on every set / parameter frame before it enters the
`polar_high.Problem`. The downstream LP row / column IDs are then
assigned in canonical order regardless of Python hash randomisation.

The autouse fixture `_reset_flextool_module_caches` in
`tests/conftest.py` clears the LP-scaling module cache around every
test so scenarios that share solve names cannot cross-contaminate.

## Adding a new test

- **Engine unit test.** Drop a `test_*.py` file into
  `tests/engine_polars/` (or the matching `loaders/` / `constraints/`
  / `objective/` subdirectory) — pytest auto-discovers.
- **New scenario golden.** Append a scenario entry to
  `tests/scenarios.yaml`, then
  `pytest tests/test_scenarios.py --regenerate <name>` and commit the
  resulting `tests/expected/<name>/` directory.
- **New fixture content (append-only).** Use
  `flextool.update_flextool.extend_tests_fixture` with a YAML delta
  rather than editing the JSON by hand.
- **Marker discipline.** Mark fast tests with `smoke` only if they
  belong in the per-commit gate; mark anything that builds and solves
  an LP with `solver`; mark anything over ~30 s with `slow`.

## Related pages

- [`architecture.md`](architecture.md) — pipeline end-to-end,
  Repository layout, verify-hook inventory
- [`engine_polars.md`](engine_polars.md) — the core under Layer 1
- [`db_schema.md`](db_schema.md) — schema migration chain that
  `sync_master_json_template --verify` enforces
- [`decomposition.md`](decomposition.md) — Lagrangian parity context
- `CHANGELOG.md` 3.47.0 entry — the JSON-fixture architecture rollout
- `CHANGELOG.md` 3.33.0 entry — the original test-pyramid rollout
