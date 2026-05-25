# Database schema and migration

FlexTool's input data lives in **Spine databases** (SQLite plus a thin
Python layer from `spinedb_api`). The schema — entity classes, parameter
definitions, value lists, alternatives — is **template-driven**: every
JSON file under [`version/`](https://github.com/irena-flextool/flextool/tree/master/version)
represents one snapshot of the schema. Migrations chain those snapshots
forward, so a user's database can always be brought up to the latest
version without losing data.

This page is a pointer for developers who need to read or modify the
schema. For the user-facing meaning of individual parameters, see
[`reference.md`](../reference.md). For where this DB sits in the
overall data flow, see [`architecture.md`](architecture.md).

## The current schema

The canonical, fully-populated snapshot is:

```
schemas/spinedb_schema.json
```

It lists every entity class, parameter definition, default value, value
list, alternative, and parameter-group assignment that FlexTool knows
about at the current version. New empty input databases are built from
this file. The GUI's parameter-group picker (see
[`excel_interface.md`](../excel_interface.md) for the same groupings
used in the Excel layer) reads its categorisation from here.

The current target version is the integer constant `FLEXTOOL_DB_VERSION`
in [`flextool/update_flextool/__init__.py`](https://github.com/irena-flextool/flextool/blob/master/flextool/update_flextool/__init__.py).
The same constant is read at write time by
`flextool/process_inputs/write_to_input_db.py`, so the schema version a
fresh write announces matches the migration target.

Output-side databases use a parallel set of templates:

- `flextool_template_results_master.json` — output DB schema
- `output_settings_template.json`, `output_info_template.json` — output config seeds
- `comparison_settings_template.json` — scenario-comparison config seed

## Migrations

`flextool/update_flextool/db_migration.py` walks a user's database
forward through versioned steps. Each step either:

- imports a small `flextool_template_v<N>.json` snapshot (e.g.
  `flextool_template_rolling_window.json`,
  `flextool_template_lifetime_method.json`), or
- runs a hand-written Python step that adds/removes/renames specific
  entity classes, parameter definitions, value lists, or alternatives.

The chain is **forward-only** — there are no down-migrations. Each
step bumps the `model.version` parameter stored in the DB so the
migration is resumable from where it left off.

!!! note "Idempotent commits"
    Migration steps wrap their `commit_session` calls in `_commit_step`,
    which swallows `spinedb_api.exception.NothingToCommit`. A step that
    was already hand-applied by an earlier partial run is treated as a
    no-op rather than a fatal abort, so the version bump still
    persists and later steps run normally.

Skim `migrate_database` in `db_migration.py` to see the full chain and
the per-version diffs.

## CLI: `flextool-migrate-database`

```
flextool-migrate-database path/to/input_data.sqlite
```

Implemented in [`flextool/cli/cmd_migrate_database.py`](https://github.com/irena-flextool/flextool/blob/master/flextool/cli/cmd_migrate_database.py).
Runs the migration chain idempotently against the given SQLite file.
Re-running on an already-current DB is safe — every step is a no-op.

## `initialize_database`

`flextool/update_flextool/initialize_database.py` creates a fresh empty
database from a JSON template (defaults to
`schemas/spinedb_schema.json`). Used by:

- the GUI's *Add empty input DB* action,
- `update_flextool` when `input_data.sqlite` is missing,
- session-scoped pytest fixtures that need a clean DB.

## `update_flextool` / `self_update.py`

The user-facing updater is implemented in
[`flextool/update_flextool/self_update.py`](https://github.com/irena-flextool/flextool/blob/master/flextool/update_flextool/self_update.py).
On each run it:

1. `git pull` the FlexTool source tree
2. `pip install --upgrade` to refresh Python dependencies
3. Run `migrate_database` against every `.sqlite` under `templates/`
   and against the user's `input_data.sqlite` if present
4. Initialise the input DB from the master template if it doesn't exist

This is what the GUI's *Update* button invokes.

## Adding a schema change

A typical workflow when introducing a new parameter or entity class:

1. Pick the next free version number `N` (one above the current
   `FLEXTOOL_DB_VERSION`).
2. Either add a small `schemas/pre_v26/flextool_template_v<N>.json` snapshot
   that imports cleanly via `spinedb_api.import_data`, or write a
   hand-coded `add_parameters_manual` / `remove_parameters_manual` /
   `add_value_list_manual` / `add_relationships_manual` step. Recent
   steps in `db_migration.py` are good templates.
3. Append the new step to the `while next_version <= new_version:`
   chain in `migrate_database`.
4. Bump `FLEXTOOL_DB_VERSION` in
   `flextool/update_flextool/__init__.py`.
5. Mirror the change into `schemas/spinedb_schema.json` so freshly
   initialised DBs land at the new schema directly. The
   `sync_master_json_template.py` helper (CI runs it with `--verify`)
   checks that master is consistent with the migration chain.
6. Add a test if the migration is non-trivial — see
   [`testing.md`](testing.md) for where to put it.

!!! tip
    Steps that touch existing rows should be tolerant of partial state:
    use `update_item` / `add_update_item` so re-applying the step on a
    DB that has already been migrated is a no-op.

## Related pages

- [`architecture.md`](architecture.md) — overall data flow from input
  DB through the engine to outputs
- [`reference.md`](../reference.md) — user-facing parameter docs that
  follow the schema groupings
- [`excel_interface.md`](../excel_interface.md) — where the
  parameter-group picker also surfaces
