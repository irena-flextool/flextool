# Testing

FlexTool has a multi-layer test pyramid: fast unit/integration tests on
the `engine` building blocks, scenario goldens built on demand
from JSON fixtures, and end-to-end scenario runs that compare output
CSVs against committed reference files. This page is a developer
pointer — where the tests live, how the JSON-fixture architecture
works, how the CI verify chain enforces drift detection, and how the
markers gate CI.

For the high-level test-layout inventory see
[`architecture.md`](architecture.md) ("Test layout and fixture
architecture"). For the engine internals being tested see
[`engine.md`](engine.md). For the JSON-fixture rollout
narrative see the `CHANGELOG.md` entry for v3.47.0.

## Layer 1 — engine unit & integration

Lives under `tests/engine/`. Covers the LP build path, derived
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
fixtures live in `tests/engine/fixtures/` and are auto-added to
`sys.path` by `tests/engine/conftest.py` so test modules can
`from flex_toy_<feature> import …` directly.

**Meta-provider invariants.**
`tests/engine/test_meta_provider_invariants.py` enforces the
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
| `tests.json` | Main fixture; `tests/scenarios.yaml` and most engine tests build their work-folders from here. Also the source for the generated `templates_examples.json` canonical DB. |
| `stochastics.json` | Stochastic / forecast-branch scenarios. Source for the generated `howto_stochastics.json` canonical DB. |
| `stochastics_pbt_inflow.json` | PBT inflow scenarios (period-block-time stochastic inflow). |
| `lh2_three_region.json` | LH2 three-region monolithic + Benders parity fixture (see [`decomposition.md`](decomposition.md)). |
| `h2_trade_parity.json` | H2-trade Benders parity fixture. |
| `multi_ts_branch1.json` | Multi-timeseries branch-1 fixture. |
| `branch2_parent_period.json` | Branch-2 / parent-period stochastic fixture. |
| `case14_dc_power_flow.json` | PGLib IEEE case14 MATPOWER export (~8200 lines), used by `test_flex_dc_power_flow.py`. |

The `_augment_*.py`, `build_*.py` and `regen_lh2_three_region.py` scripts
alongside these JSONs are builders / overlays kept for provenance, not
runtime dependencies.  They take the schema sections from
`spinedb_schema.json` (`fixture_schema.schema_sections`), and
`tests/update_flextool/test_fixture_builders.py` checks that each still
reproduces the data of its committed fixture.  Every fixture carries
exactly the schema's sections (`test_fixture_schema_drift.py`).

### `scenario_workdir` — on-demand cascade fixture

`scenario_workdir(scenario_name, db_fixture="main")`, defined in
`tests/conftest.py`, is the session-scoped fixture that
replaced the historical `tests/engine/data/work_*/` snapshot
pattern. It shares one cached run with `scenario_run` (which returns the
`tests._support.load.ScenarioRun` instead of the folder). For a requested
`(scenario, db_fixture)` pair it:

1. Resolves `db_fixture` against the per-fixture session-scoped DB
   (built once per session from the matching JSON).
2. Runs the full cascade (`tests._support.load.run_scenario`:
   `csv_dump=False`, `keep_solutions=True`) in a `tmp_path_factory`
   work folder.
3. Copies the DB to `tests.sqlite` and snapshots the last step's
   provider into `input/` + `solve_data/`, for tests that read those
   files. Nothing in a run reads them back: to get a sub-solve's
   `FlexData`, use `scenario_run` + `load_scenario` (the production
   loader call: the step's provider + a DB reader).
4. Caches the result per `(scenario, db_fixture)` —
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

The two `tests/engine/data/work_<scenario>/` directories that
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
Benders gap-to-monolithic check — see
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
  `pytest tests/engine/<area>/`. Run continuously while
  iterating.
- **Broader gate** — `pytest tests/engine/` full sweep, once
  per commit batch (~27 minutes on a recent reference run).
- **Activation truth** — `pytest tests/` (Layer 1 + scenario
  integration suite + `tests/model/` + `tests/gui/` +
  `tests/spinedb_backend/`), run rarely
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
| `decomposition` | Tier 8 objective-decomposition parity (`tests/test_scenarios.py`) |
| `perturbation`  | Tier 6 single-multiplier perturbation tests                   |
| `emission`      | Tier 7 MPS row-count emission tests                           |
| `csv_reload`    | CSV reload contract tests in `tests/csv_reload/` — opt-in     |
| `gui`           | Opens a real Tk window — opt-in with `--run-gui` (see "GUI tests" below) |

The `smoke` marker is attached automatically to scenarios in
`scenarios.yaml` that set `smoke: true`.

`csv_reload` is attached by `tests/csv_reload/conftest.py` to every test
in that folder: tests that load a CSV work folder (a provider snapshot, a
`dump_csvs` tree, hand-written CSVs) with the unsupported
`flextool.engine._csv_reload.load_from_dump` (a provider seeded from the
folder's CSVs), which production never does: `load_flextool` needs the
sub-solve's provider and a DB reader, and never reads the work folder. The default `addopts` (`-m 'not csv_reload'`)
deselects them, and because a command-line `-m` replaces that one, the
conftest also skips them unless the `-m` expression names `csv_reload`.
Run them with `pytest -m csv_reload tests/csv_reload/`.

`.github/workflows/ci.yml` defines the three-tier CI gate:

- **smoke** — every push and PR; runs
  `pytest -m "smoke or (not slow and not solver)"` with a 1-minute
  timeout.
- **full** — push to `master` and nightly cron; full Layer-1 plus
  Layer-2 MPS parity (30-minute timeout).
- **slow** — nightly only; `-m slow`.
- **gui-tests** — every push and PR, a required check: the Tk tests
  under xvfb (see "GUI tests" below).

### GUI tests

The tests that open a real Tk window are the modules listed in
`_GUI_DISPLAY_MODULES` (`tests/conftest.py`) and the tests marked `gui`
(`pytestmark = pytest.mark.gui`, e.g. every module of `tests/gui/assistant/`).
They are skipped unless `--run-gui` is given; `--gui-only` deselects
everything else.  Never run them on your own display — they move windows
and the keyboard focus. Run them under xvfb, with openbox as the window
manager as the CI job does (mapping, focus, grabs and the work area then
behave as on a desktop):

```bash
xvfb-run -a sh -c 'openbox & sleep 1; python -m pytest tests/gui --run-gui --gui-only -n 0'
```

The `gui-tests` job of `ci.yml` runs exactly this on Linux (Python 3.14,
`--timeout 300`). It runs serially: the focus is one per display, so tests
in parallel processes on one display take it from each other. A test that
checks where the focus went asks the window (`focus_lastfor()`), not the
display (`focus_get()`). A dialog is transient to its parent, and a window
manager never maps a transient of a withdrawn window: open dialogs over a
mapped window. The Tk-free modules of `tests/gui/` run in the sharded
Tests workflow. Windows and macOS Tk are checked by hand.

### Test-suite environment variables

Opt-in switches read by the test suite only (registered with audience `test` in `ENV_KNOBS`,
`flextool/run_options.py`). The runtime developer switches the suite also uses, such as
`FLEXTOOL_AUTOSCALE_STRICT` and `FLEXTOOL_SKIP_SOLVER_PROBE`, are in
[Environment variables (developer)](env_vars.md).

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `FLEXTOOL_NATIVE_PARITY_SWEEP` | unset (skipped) | `1` | Run the slow parity sweep of `tests/engine/test_orchestration.py` over every fixture (about 5 min per fixture). |
| `FLEXTOOL_WRITE_SOLVER_HANDOFF_GOLDEN` | unset | `1` | Rewrite `tests/engine/data/solver_handoff_golden.json` from the current code (`tests/engine/test_solver_handoff_golden.py`; the test then skips). Only when a change to what the solvers are handed is intended. |
| `FLEXTOOL_TEST_RSS_BUDGET_MB` | `1500` | MB | RSS budget of the input-derivation memory test `tests/spinedb_backend/test_memory_budget.py`, for machines with unrelated memory pressure. |
| `FLEXTOOL_TEST_KEYRING_LOG` | set by the suite | file | Set by the root `conftest.py` in the first process of a session and inherited by the xdist workers: every process appends a call that reached a system keyring backend, or a store or removal on the default keyring, to this file, and the session fails when it is not empty (`tests/user_dirs_guard.py`). Not for use by hand. |
| `FLEXTOOL_SHIM_RECORD` | set by the tests | file | Set by `tests/engine/commercial_shim/test_item230_commercial_shims.py` (opt-in, `-m commercial_shim`): each commercial-CLI shim in `tests/engine/commercial_shim/shims/` appends one JSON line per call to this file (arguments, stdin script, the parameters as the solver package reports them after the solve, status, MIP gap); the tests assert on it. Not for use by hand. |

## Running the tests

```bash
pytest -m smoke              # the CI smoke gate, fast
pytest -m "not solver"       # everything that doesn't touch a solver
pytest tests/engine/  # broader gate — engine layer
pytest tests/                # activation truth (slow)
```

Regenerate a single scenario's golden CSVs in place:

```bash
pytest tests/test_scenarios.py --regenerate coal
```

This runs only the matching `scenario` parametrize-case, writes fresh
CSVs into `tests/expected/coal/`, and lets you commit the diff.

## Determinism guards

`engine` emitters keep cross-solve determinism through explicit
polars sort keys on every set / parameter frame before it enters the
`polar_high.Problem`. The downstream LP row / column IDs are then
assigned in canonical order regardless of Python hash randomisation.

The autouse fixture `_reset_flextool_module_caches` in
`tests/conftest.py` clears the LP-scaling module cache around every
test so scenarios that share solve names cannot cross-contaminate.

## Adding a new test

- **Engine unit test.** Drop a `test_*.py` file into
  `tests/engine/` (or the matching `loaders/` / `constraints/`
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

## Recorded provider replies

The AI assistant's backend tests run on synthetic payloads (`tests/assistant/llm/payloads/`). The real answers of
each AI service are kept as recordings under `tests/assistant/llm/recorded/<preset>/<case>.json`, made by hand with
`tools/record_llm_replies.py` and checked in CI by `tests/assistant/llm/test_recorded_replies.py` (skipped while
there are none). A recording is made **only** against the synthetic test project that the tool builds itself
(`tests/fixtures/tests.json` with the runs of the assistant tests), never against a user's project.

To record one service:

1. Make its key findable as the chat finds it: the preset's environment variable (`GEMINI_API_KEY`, …) or
   `flextool-assistant --set-key --provider <preset>`. The key is never given on the command line. For a local
   server, start it with the model loaded the way users run it (for Ollama, `OLLAMA_CONTEXT_LENGTH=16384`).
2. Run, from the repository root:
   ```bash
   python tools/record_llm_replies.py --preset gemini --model <a model from flextool-assistant --list-models>
   python tools/record_llm_replies.py --preset ollama --model <a pulled model that supports tools>
   python tools/record_llm_replies.py --preset openai_compatible --base-url http://localhost:8000/v1 --model <name>
   ```
   The tool asks two fixed questions (one about input data, one about a result) through the real session, so the
   service is billed for a few requests. It writes the file only when both questions were answered, the model made
   at least two rounds of tool calls, the scrubbed recording passes the scrub check and it replays to the same
   answers; otherwise it says why and writes nothing (exit 1).
3. Look through the file before committing it, then run
   `pytest tests/assistant/llm/test_recorded_replies.py`.

What the scrub does before writing: no request header is recorded; message, call, response and request ids
(`chatcmpl-…`, `msg_…`, `toolu_…`, `gen-…`, `call_…`, `req_…`) become placeholders of the same format, the same
everywhere in the file; organisation and project ids, run ids, e-mail addresses and the key are replaced; the
FlexTool redaction removes the project folder, the home folder, user and host names and other absolute paths; the
address of a custom endpoint becomes `https://endpoint.example.test/v1`. Any other address than the preset's own
or one on this computer keeps its scheme, port and route kind with a placeholder host (`192.0.2.1` or
`[2001:db8::1]` on the local network, `endpoint.example.test` elsewhere) and the preset's path; a `remote_host`
outside the known services becomes `forwarded.example.test`. In every text, the top-level fields included, an IP
address outside loopback and the documentation ranges becomes a numbered documentation address (`192.0.2.N`,
`2001:db8::N`, never the endpoint's), and a host name of the local network (`*.lan`, `*.local`, `*.home`,
`*.internal`, `*.intranet`, `*.corp`, `*.localdomain`, `*.home.arpa`, or a one-label host after `://`) becomes
`host-N.example.test`; a one-label endpoint host is replaced only as a host (`://gpu`, `gpu:11434`), never inside a
word. Public host names in texts stay. Of the model server's own answers only the fields the probe reads are
recorded (Ollama `/api/show`: capabilities, `remote_host`, parameter size, architecture and context length;
`/api/ps`: the recorded model's name, context, size and graphics memory, no digest; LM Studio: the recorded model's
context, load state and capabilities), and a model list keeps only the recorded model. Signatures and thinking text stay, because the replay needs them. Response headers are kept only from an
allow-list (`content-type`, `retry-after`, `retry-after-ms`, `x-ratelimit-*`, `anthropic-ratelimit-*`).

What the scrub check (in the tool and in CI) refuses: a key-like text (`sk-`, `sk-ant-`, `sk-or-`, `AIza…`), a run
of 32 or more letters and digits outside a signature field, an absolute path, an e-mail address, a request header,
`x-stainless-*`, `openai-organization` or `openai-project` anywhere, a project name other than the fixture's, an
IP address outside loopback and the documentation ranges, a host name of the local network or a one-label host
after `://`, a `base_url` or `remote_host` outside the known services, and (in the tool) the endpoint's own host
left anywhere — a one-label one as a whole word. The check's patterns are its own and looser than the scrub's, and
it also reads each streamed answer joined, so an address the scrub missed, or one split over two pieces, is
refused rather than written.

When a recording settles a point the design marked unverified (for example the shape of Ollama's `remote_host`
or of LM Studio's model list), correct the code and its tests in the same change.

## Related pages

- [`architecture.md`](architecture.md) — pipeline end-to-end,
  Repository layout, verify-hook inventory
- [`engine.md`](engine.md) — the core under Layer 1
- [`db_schema.md`](db_schema.md) — schema migration chain that
  `sync_master_json_template --verify` enforces
- [`decomposition.md`](decomposition.md) — Benders parity context
- `CHANGELOG.md` 3.47.0 entry — the JSON-fixture architecture rollout
- `CHANGELOG.md` 3.33.0 entry — the original test-pyramid rollout
