# Data-structure and migration rules (`flextool/update_flextool/`)

- **Migration from any past version must work.** A step that crashes on a valid database is a bug and is fixed,
  even in a released step (hot-patch) — with a migration test.  A step is never loosened to tolerate a malformed
  source file (e.g. one hand-edited to carry both old and new names): fix that file instead.
- **One data version per release.** The version under development has one step (`_migrate_v<N>_schema_batch`),
  edited and extended in place until the release ships; never add a second step or a later version for the same
  release, and no code that repairs development-stamped databases.  Branch code migrates only copies in tmp
  folders, never a real database.  Repository databases (fixtures, canonical JSONs, seeds) are regenerated with the
  finished step before the release (`CONTRIBUTING.md` "Updating the Data Structure").
- **A schema rename** updates the engine readers, `PARAMETER_TYPES` (carry old + new names during the migration
  window), the migration step in `db_migration/`, and — for a parameter on an exported class — `split_params` in
  `flextool/export_to_tabular/export_settings.yaml` (or its exempt list; `TestSplitParamsSchemaSync` in
  `tests/test_export_to_tabular.py` catches drift).
- Versions ≤ 24 use the `_pre_v26_template` JSON path, ≥ 25 the add_update_item path; the one canonical v25 template
  is `flextool/schemas/pre_v26/flextool_template_v25.json`.
- `db_migration/` is a package: one module per data-version step (`v01_v25` .. `v83`, `v84_branching`), shared
  helpers in `common.py`, the walk over the steps in `runner.py` (one `elif next_version == N:` per version), the
  refresh of a database at the current version in `head.py`; the public names in `__init__.py`.  A new version is a
  new `v<N>.py` and one elif in `runner.py`.  A module split is proved with `tools/split_check.py`.
