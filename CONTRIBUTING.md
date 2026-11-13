# Contributing to IRENA FlexTool

## Development Setup

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/irena-flextool/flextool.git
   cd flextool
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   pip install pyyaml matplotlib
   ```
   The `[dev]` extra pulls in `pytest`, `pytest-xdist` and `pytest-timeout`.

3. Verify the setup by running the tests:
   ```bash
   pytest tests/
   ```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the codebase.

## Running Tests

Run the full test suite:
```bash
pytest tests/
```
The suite runs one worker process per physical core by default (`-n auto`, via
`pytest-xdist`). Pass `-n 0` to disable parallelism when you need `-s` / `--pdb`
for interactive debugging.
The suite and every FlexTool run it starts are single-threaded (`tests/conftest.py`
and the CI workflows set `FLEXTOOL_THREADS=1` and `POLARS_MAX_THREADS=1` unless
they are already set); thread-budget tests set their own environment.

Tests of the unsupported CSV reload are deselected by default; run them with
`pytest -m csv_reload tests/csv_reload/`.

Axis-Enum mode (`pl.Enum` axis columns instead of `pl.Utf8`) is experimental and
off in production; `FLEXTOOL_AXIS_ENUMS=1` turns it on for a whole run (see
`docs/dev/engine.md` "Enum-dtype axis convention" for its known gaps). Its LP
parity run (Enum LP byte-identical to the Utf8 LP, fixture by fixture) is
deselected by default; run it with
`pytest -m axis_enums tests/engine/axis_enums -n 8` (it also runs nightly in
the non-gating CI job `enum-mode`).

The heaviest tests (~1 h of CI worker time) carry the `nightly_heavy` marker.
A local `pytest tests/` runs them like any other test; only the PR "Tests"
workflow deselects them (`-m "not csv_reload and not axis_enums and not
nightly_heavy"` — a command-line `-m` replaces the `addopts` one), and its
`nightly-heavy` job runs them on Linux every night and on `workflow_dispatch`.
Mark a test `nightly_heavy` only when it costs minutes and a faster test of the
same area stays in the PR run.

The PR "Tests" workflow runs Linux 3.13 + 3.14, Windows 3.14 and macOS 3.14,
each with one xdist worker per logical core (`-n logical`: 4 on the Linux /
Windows runners, whose 4 vCPUs are 2 SMT cores — the default `-n auto` counts
physical cores — and 3 on macOS); `.github/scripts/peak_mem.py` prints each
job's cores and peak memory. Python 3.11 and 3.12 (still supported,
`requires-python >= 3.11`) run in `tests-old-python.yml`; test them locally
when a change touches version-specific code.
CI splits the PR run of each OS/Python cell into 4 shards with
`--shard K/N` (`tests/_ci_shard.py`): module-atomic (a module's tests stay in
one shard, so module-scoped fixtures are built once) and balanced by the
per-module worker seconds in `tests/ci_durations.json`. The shards partition
exactly what the same command selects unsharded (`tests/test_ci_shard.py`).
Reproduce a CI shard locally with e.g.
`pytest tests/ --shard 2/4 -m "not csv_reload and not axis_enums and not nightly_heavy"`.
When the shard times drift apart, refresh the weights from a few complete
Linux job logs:
```bash
gh api repos/:owner/:repo/actions/jobs/<job-id>/logs > job.log
python tools/ci_durations.py job1.log job2.log ...   # writes tests/ci_durations.json
```
Logs from the sharded PR jobs can be combined (each covers its own modules).

Run a specific test file:
```bash
pytest tests/test_scenarios.py -v
```

Run a single scenario test:
```bash
pytest tests/ -k "coal" -v
```

### LP snapshot gate (model-layer refactors)

A change to `flextool/engine/model/core.py` or its feature modules that should not change the model is checked with `tools/lp_snapshot.py`: it builds the LP of a fixed fixture set and writes named MPS files, then compares two snapshots byte for byte (or, with `--canonical`, up to row/column order). Emit order is part of the LP, so this also shows what reordering steps does.

```bash
python tools/lp_snapshot.py snapshot -j 8 _audit_reports/lp_base   # before the change
python tools/lp_snapshot.py snapshot -j 8 _audit_reports/lp_new    # after
python tools/lp_snapshot.py compare _audit_reports/lp_base _audit_reports/lp_new
```

### Shape-equivalence tests for parameters

A parameter that accepts more than one shape (constant, period map, time map, period×time, stochastic) must have a test that a constant and the equivalent map produce the same model — in a single solve, a rolling solve and a stochastic solve. Most shape bugs found so far (a constant silently dropped, a map on one entity wiping another entity's value, period values lost in rolls or branch periods) would have been caught by this one test type.

### Golden File Tests

Scenario tests compare solver outputs against golden files in `tests/expected/<scenario>/`. To regenerate golden files after intentional changes to the model or output processing:

```bash
# Regenerate one scenario
pytest tests/ --regenerate coal

# Regenerate all scenarios
for s in $(grep "^- scenario:" tests/scenarios.yaml | sed 's/- scenario: //'); do
  pytest tests/ --regenerate "$s"
done
```

Always inspect the changes in `tests/expected/` before committing to make sure the differences are expected.

## Updating the Data Structure

When the FlexTool database schema needs to change (new parameters, renamed entities, new value lists, etc.), follow this process:

1. **Edit `flextool/update_flextool/db_migration.py`** — add a new migration step at the end of the `migrate_database()` function. Increment the version number in `flextool/update_flextool/__init__.py`.

2. **Regenerate the master template** — this ensures `schemas/spinedb_schema.json` matches the latest schema:
   ```bash
   python -m flextool.update_flextool.sync_master_json_template
   ```
   This creates a temporary database from the current master template, runs all migrations on it, and exports the result back to JSON.

3. **Migrate the canonical example/template databases** — the `templates/` and `how to example databases/` SQLites live in git as JSON (`schemas/canonical_databases/*.json`). After bumping the schema, round-trip every JSON through the current migration chain; the command also replaces each JSON's schema sections (entity classes, value lists, parameter groups, definitions with their defaults and descriptions, parameter types) with those of `schemas/spinedb_schema.json`:
   ```bash
   python -m flextool.update_flextool.canonical_databases migrate-all
   ```
   This step is idempotent when no schema change is pending. The on-disk `.sqlite` files in `templates/` and `how to example databases/` are recreated on next `update_flextool()` call (or `pytest_configure` for tests) — you don't need to commit them; they're gitignored.  CI runs `python -m flextool.update_flextool.canonical_databases verify` to catch drift between committed canonical JSONs and the current schema.

4. **Migrate the test fixtures** — `tests/fixtures/*.json` use a base64-packed value format (consumed by `tests/db_utils.py`) and are migrated separately from the canonical databases.  Every fixture JSON with `parameter_definitions` is migrated and then gets the schema sections of `schemas/spinedb_schema.json` (a fixture without a `model.version` definition is only synced):
   ```bash
   python -m flextool.update_flextool.test_fixtures migrate-all
   ```
   Idempotent when no schema change is pending.  CI runs `python -m flextool.update_flextool.test_fixtures verify` to catch drift between committed fixtures and the current schema.

   A fixture's schema sections are never edited by hand and never copied from another fixture: a migration step only adds what is absent, so a copy that skips a change never catches up.  `tests/update_flextool/test_fixture_schema_drift.py` fails, naming each (class, parameter, field) difference, when a fixture or canonical JSON differs from the schema.  A migration that fails here with "not in the schema" or "not in the value list" left data the new schema cannot hold: fix the migration, not the fixture.  Fixture builder scripts (`tests/fixtures/build_*.py`, `regen_lh2_three_region.py`, `_augment_*.py`) take their sections from `flextool.update_flextool.fixture_schema.schema_sections` and write through `test_fixtures.write_built_fixture` / `migrate_fixture`; `tests/update_flextool/test_fixture_builders.py` checks that each still reproduces the data of its committed fixture.

5. **Verify** (optional, also run by CI):
   ```bash
   python -m flextool.update_flextool.sync_master_json_template --verify
   python -m flextool.update_flextool.test_fixtures verify
   python -m flextool.update_flextool.canonical_databases verify
   ```

6. **Regenerate user-facing example databases from test fixtures**:
   ```bash
   python -m flextool.update_flextool.generate_canonical
   ```
   This produces `flextool/schemas/canonical_databases/*.json` files
   that are *derivations* of a `tests/fixtures/*.json` source, per
   the recipes in `_recipes.yaml`.  Currently:
   `templates_examples.json` (broad subset of tests.json) and
   `howto_stochastics.json` (full mirror of stochastics.json).

   The remaining `canonical_databases/howto_*.json` and
   `templates_time_settings_only.json` files are *authoritative for
   themselves* — purpose-specific minimal examples curated per how-to
   doc, with no derivation relationship to tests/fixtures.  This is
   intentional: how-to scenarios are curated for teaching one concept,
   while tests.json scenarios are curated for test coverage; mixing
   the two curation goals would muddy both.  These files are kept in
   sync with schema migrations via step 3 (`canonical_databases
   migrate-all`).

   CI runs `python -m flextool.update_flextool.generate_canonical --verify`
   to catch drift on the generated files.

7. **Commit all files together** — the migration code, the regenerated master template, and any canonical-database / test-fixture JSON diffs should land in the same commit or pull request.

### Adding a new scenario via YAML delta

For small additions (a new alternative + scenario, a few scalar
parameter values), the SpineDB editor workflow can be heavy.  An
alternative path: write a YAML delta and apply it with
`extend_tests_fixture`:

```bash
python -m flextool.update_flextool.extend_tests_fixture path/to/delta.yaml
```

The delta describes new entities, alternatives, parameter values, and
scenarios; the script validates against `schemas/spinedb_schema.json`
and against the target fixture (append-only — name collisions are
rejected) before any write.  See examples in `tests/fixtures/_addenda/`
(TBD) for the YAML format.  Use `--validate` to dry-run.

Scope: **append-only, scalar parameter values only**.  Edits to
existing entries (rename, value change) go through a `db_migration.py`
step.  Complex value types (time-series, maps, arrays) go through the
SpineDB editor — the script will surface a clear error pointing you
there.

If the YAML sets `regenerate_canonical: true` and the target appears
as a `source:` in `schemas/canonical_databases/_recipes.yaml`, the
relevant canonical files are regenerated as part of the same run.

### Files involved

| File | Role |
|------|------|
| `flextool/update_flextool/__init__.py` | `FLEXTOOL_DB_VERSION` — bump by 1 for each migration |
| `flextool/update_flextool/db_migration.py` | Migration steps (one `elif next_version == N:` block per version) |
| `schemas/spinedb_schema.json` | Auto-generated — **never edit by hand** |
| `schemas/canonical_databases/*.json` | Canonical examples + templates — `templates_examples.json` is generated from `tests/fixtures/tests.json` by `generate_canonical` (per `schemas/canonical_databases/_recipes.yaml`); the rest are regenerated by `canonical_databases migrate-all` |
| `schemas/canonical_databases/_recipes.yaml` | Recipes mapping `tests/fixtures/*.json` sources to `canonical_databases/*.json` outputs |
| `tests/fixtures/*.json` | Test-suite database fixtures — regenerated by `test_fixtures migrate-all` |

### Migration code examples

The migration block uses the SpineDB API (`db` is a `DatabaseMapping`). Key imports already available in the file: `import_data`, `from_database`, `to_database`, `SpineDBAPIError`.

**Add a new parameter** (no value list):
```python
elif next_version == N:
    db.add_update_item("parameter_definition",
        entity_class_name="connection", name="reactance",
        parameter_type_list=("float",),
        description="[p.u.] Per-unit reactance of the transmission line.")
    db.commit_session("Added reactance parameter")
```

**Add a new parameter_value_list and use it on a parameter** (enum/dropdown):
```python
elif next_version == N:
    # 1. Add the value list entries (list name + each allowed value)
    add_value_list_manual(db, [
        ["my_methods", "option_a"],
        ["my_methods", "option_b"],
        ["my_methods", "option_c"],
    ])
    # 2. Create or update the parameter definition to use the list.
    #    default_value must be in database format — use to_database().
    default_val, default_type = to_database("option_a")
    db.add_update_item("parameter_definition",
        entity_class_name="group", name="my_param",
        default_value=default_val, default_type=default_type,
        parameter_value_list_name="my_methods",
        description="Choose one of: option_a, option_b, option_c.")
    db.commit_session("Added my_methods value list and my_param parameter")
```

Note: `add_value_list_manual` commits internally, so the value list is available when `add_update_item` references it.

**Add an entry to an existing parameter_value_list:**
```python
elif next_version == N:
    add_value_list_manual(db, [["existing_list_name", "new_value"]])
```

**Rename a parameter:**
```python
elif next_version == N:
    parameter_definitions = db.mapped_table("parameter_definition")
    param = db.item(parameter_definitions, entity_class_name="unit", name="old_name")
    if param:
        db.update_parameter_definition(id=param["id"], name="new_name",
            description="Updated description.")
    db.commit_session("Renamed old_name to new_name")
```

**Update only the description of an existing parameter:**
```python
elif next_version == N:
    db.update_item("parameter_definition",
        entity_class_name="node", name="penalty_up",
        description="[CUR/MWh] Updated description text.")
    db.commit_session("Updated penalty_up description")
```

### Common pitfalls

- **`NothingToCommit` error** — `db.commit_session()` raises this if nothing changed. This can happen when `add_update_item` silently finds no difference (e.g., `default_value` was passed as a plain string instead of using `to_database()`). Always convert default values with `to_database()`.
- **Value list must exist before referencing** — call `add_value_list_manual()` before `add_update_item()` that uses `parameter_value_list_name`.
- **`add_value_list_manual` commits internally** — don't call `db.commit_session()` right after it unless you have other uncommitted changes.
- **Entity classes** — the main entity classes are: `model`, `solve`, `node`, `unit`, `connection`, `commodity`, `group`, `constraint`, `profile`. Relationship classes include `unit__node`, `connection__node`, `reserve__upDown__group`, etc.

## Documentation style

The user docs live in `docs/` and are built with MkDocs Material (`mkdocs build --strict`). Write them like this:

- **Plain English.** Use short sentences and everyday words. Say what the model does, not how the code is organised.
- **Exact names.** Write parameter, method and table names exactly as in the data, in backticks (`conversion_flow_coeff`, `new_stepduration`). Link to a long parameter description instead of repeating it.
- **Short paragraphs.** Split a paragraph longer than 3–5 sentences. Start each new paragraph with a few **bolded words** that say what it is about.
- **Long list items.** Keep the bullet itself to one line, then continue in paragraphs indented by 4 spaces under it. Where breaking the item would mess up the indentation, use an admonition (`!!! note` or `!!! tip`) instead.
- **Blank line before a list.** A list that follows a paragraph line directly renders as part of that paragraph. Always leave a blank line between them, also before a nested list.
- **No version history.** Describe what the current version does. Mention an older version or data version only when it still matters to a current user, for example when the database migration silently changed a behaviour.

## Pull Requests

### Branching

- Create a feature branch from `main` for your work.
- Use descriptive branch names: `fix-reserve-output`, `add-ramp-cost`, `update-excel-export`.

### What a PR Should Contain

- **Focused changes** — one feature, one bug fix, or one refactoring per PR. Don't mix unrelated changes.
- **Tests** — add or update tests for any new or changed functionality. If you're fixing a bug, add a test that would have caught it.
- **Vectorized derive parity** — when you change a heavy preprocessing derive in `flextool/engine/preprocess/emit/`, gate the vectorized derive against a reference implementation with a `tests/engine/test_vectorize_<family>_parity.py` test, under a two-tier policy: **Tier A** strict byte-equality (`df_vec.equals(df_legacy)`) for pure lookup/coalesce families, **Tier B** tolerance (`rtol≈1e-12`, parsing both sides' rendered values back to float) for families that sum many terms. Run that per-family parity test in the inner loop.
- **Updated golden files** — if your change affects solver outputs, regenerate the affected golden files and include them in the PR.
- **Updated master template** — if your change modifies the database schema, include the regenerated `schemas/spinedb_schema.json`.
- **Clear description** — explain what the PR does and why. Reference any related issues.

### PR Checklist

- [ ] Lint passes (`ruff check .`; `ruff check --fix .` applies safe autofixes)
- [ ] Tests pass locally (`pytest tests/`)
- [ ] No unintended changes to golden files
- [ ] Master template is up-to-date (If schema changed. Check [CONTRIBUTING.md updating the data structure](#updating-the-data-structure))
- [ ] Code follows existing patterns and style
- [ ] A new schema map parameter is in `TIME_AGGREGATION` (how a solve's `new_stepduration` combines it: mean, harmonic mean, sum or last) or in `NOT_TIME_SERIES` (`flextool/engine/input/_solve_step_aggregation.py`); `tests/engine/test_time_aggregation_registry.py` checks it.
- [ ] New `add_var` / `add_cstr` / schema parameter is registered in the autoscale registries (`flextool/engine/autoscale/_layer2_types.py`, `_quantity_types.py`); ran the affected tests with `FLEXTOOL_AUTOSCALE_STRICT=1`. A missing entry silently degrades the solve to an un-scaled LP — see [architecture.md "Numerical scaling"](docs/dev/architecture.md).
- [ ] Any parameter broadcast over `(d,t)` goes through `_param_shapes` (`resolve_param_shape` / `broadcast_to_period_time` / `promote_param_to_dt`), never column-name axis detection — see [architecture.md "Parameter shapes & (d,t) broadcasting"](docs/dev/architecture.md).
- [ ] New / changed tests build their DB from JSON or schema under `tmp_path` (no checked-in `.sqlite` dependency).
- [ ] The CSV dump writers (`dump_csvs`, `snapshot_processed_inputs`) are supported; `flextool/engine/_csv_reload/` (CSV reload, offline outputs) is not kept in sync — do not update it or its `tests/csv_reload/` unless you need the reload, and never import it from outside the package.
- [ ] A changed heavy `preprocess/emit/` derive is gated by a `tests/engine/test_vectorize_<family>_parity.py` test against a reference implementation (Tier A `.equals` for pure lookup/coalesce families, Tier B `rtol≈1e-12` for many-term sums).

### Review Process

- PRs to `main` trigger CI tests on Linux, Windows, and macOS.
- At least one review approval is recommended before merging.
- Prefer merge commits over squash to preserve commit history.

## Releasing

A release is a version tag pushed to GitHub; `publish.yml` builds it and uploads it to PyPI.  Checklist:

1. **Version** — set `version` in `pyproject.toml` (the tag must match it).
2. **CHANGELOG** — rename `## Unreleased` to `## Release X.Y.Z (D.M.YYYY) — <short title>`.
3. **Behaviour-change registry** — in `flextool/engine/_behaviour_changes.py`, set `since="X.Y.Z"` and
   `published="D.M.YYYY"` (the heading's version and date) on every entry still `None`.  Runs on databases upgraded
   across a behaviour change print a notice naming that version and date.
   `tests/engine/test_behaviour_change_registry.py` fails while an entry is unfilled under a released CHANGELOG,
   or names a version/date with no matching heading; `publish.yml` runs it before building, so such a release
   never reaches PyPI.
4. **Tag** — an annotated tag `vX.Y.Z` on the release commit, pushed.
5. **polar-high** — the `polar-high>=` pin in `pyproject.toml` must name a version on PyPI; release polar-high
   first when the pin moved past its last release.  The pin is now `>=3.12.2` (the thread budget needs it):
   release polar-high 3.12.2 before FlexTool is pushed, or CI installs fail.
6. **Solver parameter lists** — after a `highspy` pin bump, or when a newer Gurobi, CPLEX, Xpress or COPT is
   reported, run `python tools/solver_catalogue.py --write` in a virtual environment with the solver packages
   (none needs a licence).  It merges the new names into `flextool/engine/solve/solver_catalogue/*.json` (names
   are never removed) and stops on every name without a reviewed class in `tools/solver_catalogue_review.json`:
   add each one there (a note for every text parameter and every class other than the name rules'), then write
   again.  `tests/engine/test_solver_catalogue.py` fails while the installed packages and the lists differ.

A change that alters what an existing model does gets a registry entry (summary, how to silence it) and a marker
(`metadata` named `flextool_behaviour_change:<id>`) written by the database migration on upgraded databases only;
the engine calls `_behaviour_changes.notice(...)` with the entities whose results change.

## Solver

FlexTool uses the HiGHS solver, called via the `highspy` Python package (installed automatically via pip).
The input database may set only the parameters in FlexTool's list of each solver's parameters
(`flextool/engine/solve/solver_catalogue/`, written by `tools/solver_catalogue.py`; see Releasing, step 6).

## CI Workflows

- **`tests.yml`** — push/PR to `main`: the suite minus `nightly_heavy` on Linux 3.13 + 3.14, Windows 3.14 and macOS
  3.14, 4 shards each (16 jobs: within the free plan's 20 concurrent jobs / 5 macOS jobs), `-n logical`; verifies the
  master template is up to date (3.14). Nightly: the `nightly_heavy` tests on Linux 3.13 + 3.14.
- **`tests-old-python.yml`** — the full sharded suite (nightly_heavy included) on Linux 3.11 and 3.12, on the 1st and
  15th of the month and on *Run workflow*. A scheduled run skips its tests when the default branch has no commits
  since the workflow's last successful run (compares `github.sha` with that run's head sha); a manual run always tests.
  Not a required check.
- **`ci.yml`** — smoke gate on every push/PR; slow tests nightly (the whole suite runs sharded in `tests.yml`); the
  axis-Enum mode LP parity run (`enum-mode`) nightly, non-gating. All on Python 3.14, like `canary.yml`, `docs.yml`,
  `publish.yml` and `old-cpu.yml`.
- **`canary.yml`** — weekly dependency canary (see below).
- **`toolbox-flextool-bundle.yml`** — manual (*Run workflow*, inputs: the Spine Toolbox ref and the FlexTool
  ref): builds the Windows zip with a frozen Spine Toolbox and FlexTool. The runner's Python (3.14) only builds the
  frozen Toolbox; the Python embedded in the bundle (`_internal/Python/python.exe`, same 3.14.x, which Toolbox runs
  the FlexTool tools with) gets FlexTool and all its dependencies installed with its own pip, and
  `update_flextool.py` runs with it. FlexTool is an editable install of the bundled clone (`<bundle>/flextool`);
  setuptools writes the clone's absolute path, so `flextool/update_flextool/bundle_install.py` rewrites the
  `__editable__.flextool-*.pth` to the path relative to site-packages (`../../../../flextool`) and deletes
  setuptools' finder module, and the bundle keeps working wherever it is unzipped or moved. The workflow runs it as
  a script after moving the clone into the bundle, and "Update FlexTool" (`self_update._reinstall_if_needed`) calls
  it after every editable reinstall; it acts only when the running Python is the bundle's `_internal/Python` next to
  the clone, so venvs are untouched (tests: `tests/update_flextool/test_bundle_install.py`). odfpy is published only as an sdist and the embedded Python cannot build it
  (its `._pth` file makes it ignore `PYTHONPATH`), so its wheel is built with the runner's Python first. The job
  ends with smoke steps: imports and `flextool.env_check` in the bundled Python, a solve of the `base` example, and
  `spinetoolbox.exe --list-items flextool` (headless). The embedded Python has no tkinter, so the FlexTool GUI does
  not run from the bundle. It also gets both polars runtimes (`polars[rtcompat]` of the installed polars, when the
  bundled FlexTool selects the default runtime on AVX2 processors); smoke steps check that the default runtime loads
  on the runner and that the compatible one imports and runs. No Visual C++ runtime is copied in: the libraries that
  need `msvcp140.dll` (highspy >= 1.15, numpy, pandas, pyarrow) carry their own, and the two that import the system
  one (pywin32's MAPI, debugpy) are removed. The bundle is slimmed before the smoke steps, which then test the slim
  bundle: FlexTool's `tests/` is left out of the clone (sparse checkout, so the self-update's `git restore` /
  `git pull` keep it out); in the bundled Python, debugpy is uninstalled (ipykernel's optional debugger; `pip check`
  is allowed to report it), and the step "Slim the bundled Python" deletes the packages' `tests`/`test` folders,
  `*.lib` import libraries and C headers, jedi's third-party stubs, unused pyarrow parts (Flight, Substrait, S3, C++
  tests), matplotlib's sample data and unused pywin32 parts (Pythonwin, help file, demos, ADO, ISAPI, MAPI); the
  frozen Toolbox is built without ipykernel, debugpy, jedi and parso (the spec's `excludes`, patched in the workflow;
  kernels run in the bundled Python) and loses Qt's software OpenGL renderer (`opengl32sw.dll`) and the same pyarrow
  parts. Kept on purpose: pip and setuptools (self-update reinstall, env_check fixes), ipykernel (Toolbox's Jupyter
  Console; FlexTool's tools run in the Basic Console, which needs only the standard library), jill (Toolbox's
  "Install Julia") and both polars runtimes. "Size report" steps print the total and the 20 largest entries before
  and after slimming, and the zipped size, also to the job summary. A smoke step starts a Jupyter kernel with the
  bundled Python and imports FlexTool in it.
- **`old-cpu.yml`** — nightly and manual, non-gating: FlexTool under qemu user-mode emulation of processors without
  AVX2 (`-cpu Nehalem`, `-cpu SandyBridge`), driven by `.github/scripts/old_cpu_emulation.sh`. It checks that polars'
  default runtime crashes there, that `env_check` and `import flextool` refuse it with the fix, that
  `update_flextool.py --skip-git` installs the compatible runtime, that the `base` example then solves, and that the
  AVX2 host still selects the default runtime. Locally (no root needed): `apt-get download qemu-user-static`,
  `dpkg-deb -x` it, then `QEMU=<dir>/usr/bin/qemu-x86_64-static bash .github/scripts/old_cpu_emulation.sh Nehalem
  <venv with FlexTool, default runtime only> <FlexTool source> <work dir>`.

## Dependency caps and the canary

Dependencies whose new releases have broken FlexTool before carry an upper
cap in `pyproject.toml`; the rest (polar-high, which we follow closely,
openpyxl, xlsxwriter, matplotlib, pyyaml, the GUI helpers) are uncapped.

| package | cap | why |
|---|---|---|
| `polars` | `<3` (next major) | majors change behaviour silently (2.0 broke the rolling warm-start) |
| `pandas` | `<4` (next major) | floor `>=2.2.3` is spinedb-api's own requirement |
| `spinedb-api` | `<0.37` (next minor) | 0.x: minor releases break the API (0.33, 0.34, 0.36); `>=0.36.5` for the `parameter_group` table |
| `highspy` | `<1.16` (next minor) | patch releases allowed; `>=1.15.1`: from 1.15 the Windows wheel ships its own `msvcp140.dll` (1.14.0 import-crashes on Windows 10 1809 / Server 2019, HiGHS#2964) |

When an older highspy crashes natively, `flextool/env_check.py` offers
`pip install "highspy>=1.15.1,<1.16"` (FlexTool's own requirement).

The weekly **canary** (`.github/workflows/canary.yml`, also runnable by hand
with *Run workflow*) lifts the caps with `.github/scripts/dependency_caps.py
lift`, installs the newest versions of everything, prints the installed
versions against the caps, and runs the whole test suite. When the
tests fail, or a capped package has a release past its cap, it opens (or
updates) the single issue **"canary: dependency drift"** (label `canary`);
a later clean run closes it. To raise a cap: wait for (or trigger) a green
canary with the new version installed, bump the cap in `pyproject.toml`, and
mention it in the CHANGELOG.
