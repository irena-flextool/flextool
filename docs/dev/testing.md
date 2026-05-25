# Testing

FlexTool has a multi-layer test pyramid: fast unit/integration tests on
the `engine_polars` building blocks, golden-objective parity tests on a
curated scenario set, and end-to-end scenario runs that compare output
CSVs against committed reference files. This page is a developer
pointer — where the tests live, how to run them, and how the markers
gate CI.

For the engine internals being tested, see
[`engine_polars.md`](engine_polars.md). For the 3.33.0 narrative that
introduced this pyramid, see the corresponding entry in `RELEASE.md`.

## Layer 1 — engine_polars unit & integration

Lives under `tests/engine_polars/`. Covers the LP build path, derived
parameter expansion, writer port, scaling pipeline (see
[`scaling.md`](scaling.md)), decomposition (see
[`decomposition.md`](decomposition.md)), warm-start handoff, fast load,
and stochastic / PBT pieces. Tests range from millisecond-class unit
tests (block layout, derived params) to solver-touching parity tests.

Frozen fixtures live in `tests/engine_polars/data/` (input SQLite DBs,
expected scaling reports, expected solutions). Synthetic
flextool-flavoured fixtures live in `tests/engine_polars/fixtures/` and
are auto-added to `sys.path` by `tests/engine_polars/conftest.py` so
test modules can `from flex_toy_<feature> import …` directly.

## Layer 2 — scenario goldens & parity

`tests/scenarios.yaml` defines a parametrised end-to-end scenario set
that runs in `tests/test_scenarios.py`:

1. Load a session-scoped SQLite DB from `tests/fixtures/tests.json` or
   `tests/fixtures/stochastics.json`.
2. Run `FlexToolRunner` for the scenario in an isolated tmp workdir.
3. Compare selected CSVs against `tests/expected/<scenario>/*.csv`.
4. If the scenario YAML entry sets `expected_objective`, assert that
   `summary_solve.csv` lands within
   `expected_objective_tolerance` (default `1e-3`, often tightened to
   `1e-5` for stable scenarios).
5. If `time_budget_seconds` is set, assert the full
   write-input → solve → write-outputs cycle finishes in budget.

Standalone golden integration tests live alongside (e.g.
`tests/test_lh2_three_region.py` pins the LH2 three-region monolithic
objective against `tests/expected/lh2_three_region/objective.json` at
relative tolerance `1e-4`). The same fixture also backs the native
Lagrangian gap-to-monolithic check — see
[`decomposition.md`](decomposition.md) and
[`slack_convention.md`](slack_convention.md).

JSON fixtures under `tests/fixtures/` (`tests.json`,
`stochastics.json`, `lh2_three_region.json`, `h2_trade_parity.json`,
`stochastics_pbt_inflow.json`) are imported into a temp SQLite DB once
per session by `tests/conftest.py`.

## Layer 3 — full scenario execution

The `tests/test_scenarios.py` driver above also covers Layer 3 for the
in-tree scenarios. The separate `execution_tests/` directory contains
Spine Toolbox workflow execution tests (`test_execution.py`,
`tool.py`) that exercise the same scenarios through the Toolbox DAG
rather than the Python runner directly.

## Markers

Defined in `pyproject.toml` and re-asserted in `tests/conftest.py`:

| Marker          | Meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `smoke`         | Layer-1 + scenarios on the per-commit CI gate (<1 min)        |
| `solver`        | Touches a real solver (HiGHS), 5–60 s each                    |
| `slow`          | >~30 s — nightly only                                         |
| `decomposition` | Tier 8 objective-decomposition parity tests                   |
| `perturbation`  | Tier 6 single-multiplier perturbation tests                   |
| `emission`      | Tier 7 MPS row-count emission tests                           |

The smoke marker is attached automatically to scenarios in
`scenarios.yaml` that set `smoke: true`.

## Running tests locally

```bash
pytest -m smoke              # the CI smoke gate, fast
pytest -m "not solver"       # everything that doesn't touch a solver
pytest tests/engine_polars/  # just the engine layer
pytest tests/                # everything (slow)
```

Regenerate a single scenario's golden CSVs in place:

```bash
pytest tests/test_scenarios.py --regenerate coal
```

This runs only the matching `scenario` parametrize-case, writes fresh
CSVs into `tests/expected/coal/`, and lets you commit the diff.

## CI

`.github/workflows/ci.yml` defines a three-tier gate:

- **smoke** — every push and PR; runs `pytest -m "smoke or (not slow and not solver)"` with a 1-minute timeout.
- **full** — push to `master` and nightly cron; full Layer-1 plus
  Layer-2 MPS parity (30-minute timeout).
- **slow** — nightly only; `-m slow`.

A separate workflow `.github/workflows/tests.yml` runs the full
`pytest tests/` matrix on Linux + Windows + macOS for the canonical
3.12 Python, plus a `sync_master_json_template --verify` step that
asserts `schemas/spinedb_schema.json` matches the migration chain
(see [`db_schema.md`](db_schema.md)).

## Determinism guards

`engine_polars` emitters keep cross-solve determinism through explicit
polars sort keys on every set / parameter frame before it enters the
`polar_high.Problem`. The downstream LP row / column IDs are then
assigned in canonical order regardless of Python hash randomisation.

The autouse fixture `_reset_flextool_module_caches` in
`tests/conftest.py` clears the LP-scaling module cache around every
test so scenarios that share solve names cannot cross-contaminate.

## Adding a new test

- **Engine unit test:** drop a `test_*.py` file into
  `tests/engine_polars/` — pytest auto-discovers.
- **New scenario golden:** append a scenario entry to
  `tests/scenarios.yaml`, then
  `pytest tests/test_scenarios.py --regenerate <name>` and commit the
  resulting `tests/expected/<name>/` directory.
- **Marker discipline:** mark fast tests with `smoke` only if they
  belong in the per-commit gate; mark anything that builds and solves
  an LP with `solver`; mark anything over ~30 s with `slow`.

## Related pages

- [`architecture.md`](architecture.md) — what the pipeline being
  tested looks like end-to-end
- [`engine_polars.md`](engine_polars.md) — the core under Layer 1
- [`decomposition.md`](decomposition.md) — Lagrangian parity context
- `RELEASE.md` 3.33.0 entry — the test-pyramid rollout narrative
