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
   pip install -e .
   pip install pytest pyyaml matplotlib
   ```

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

Run a specific test file:
```bash
pytest tests/test_scenarios.py -v
```

Run a single scenario test:
```bash
pytest tests/ -k "coal" -v
```

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

3. **Migrate the canonical example/template databases** — the `templates/` and `how to example databases/` SQLites live in git as JSON (`schemas/canonical_databases/*.json`). After bumping the schema, round-trip every JSON through the current migration chain:
   ```bash
   python -m flextool.update_flextool.canonical_databases migrate-all
   ```
   This step is idempotent when no schema change is pending. The on-disk `.sqlite` files in `templates/` and `how to example databases/` are recreated on next `update_flextool()` call (or `pytest_configure` for tests) — you don't need to commit them; they're gitignored.  CI runs `python -m flextool.update_flextool.canonical_databases verify` to catch drift between committed canonical JSONs and the current schema.

4. **Migrate the test fixtures** — `tests/fixtures/*.json` use a base64-packed value format (consumed by `tests/db_utils.py`) and are migrated separately from the canonical databases:
   ```bash
   python -m flextool.update_flextool.test_fixtures migrate-all
   ```
   Idempotent when no schema change is pending.  CI runs `python -m flextool.update_flextool.test_fixtures verify` to catch drift between committed fixtures and the current schema.

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

## Pull Requests

### Branching

- Create a feature branch from `master` for your work.
- Use descriptive branch names: `fix-reserve-output`, `add-ramp-cost`, `update-excel-export`.

### What a PR Should Contain

- **Focused changes** — one feature, one bug fix, or one refactoring per PR. Don't mix unrelated changes.
- **Tests** — add or update tests for any new or changed functionality. If you're fixing a bug, add a test that would have caught it.
- **Updated golden files** — if your change affects solver outputs, regenerate the affected golden files and include them in the PR.
- **Updated master template** — if your change modifies the database schema, include the regenerated `schemas/spinedb_schema.json`.
- **Clear description** — explain what the PR does and why. Reference any related issues.

### PR Checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] No unintended changes to golden files
- [ ] Master template is up-to-date (If schema changed. Check [CONTRIBUTING.md updating the data structure](#updating-the-data-structure))
- [ ] Code follows existing patterns and style

### Review Process

- PRs to `master` trigger CI tests on Linux, Windows, and macOS.
- At least one review approval is recommended before merging.
- Prefer merge commits over squash to preserve commit history.

## Solver Binaries

FlexTool uses two solvers:
- **HiGHS** — called via the `highspy` Python package (installed automatically via pip).
- **GLPK (glpsol)** — a custom build from [mingodad/GLPK](https://github.com/mingodad/GLPK). Platform-specific binaries are in `bin/`. The macOS binary can be rebuilt via the `build-glpsol.yml` GitHub Actions workflow.

## CI Workflows

- **`tests.yml`** — runs pytest on all three platforms and verifies the master template is up to date. Triggered on push/PR to `master`.
- **`build-glpsol.yml`** — builds the glpsol binary for macOS arm64. Triggered manually.
