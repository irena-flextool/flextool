## FlexTool 4.0.2 — results-database schema seeding fix

Patch release. No schema or model changes.

- **The results database is no longer seeded from an outdated schema.** On install/update, `flextool-update` created `templates/results_template.sqlite` and `results.sqlite` from the legacy pre-v26 template (old `object_parameters` / `relationship_parameters` format) rather than the current results schema, which could break the results-import / Spine Toolbox workflow. It now uses `schemas/spinedb_results_schema.json` (modern `entity_classes` / `parameter_definitions`), consistent with the output-write path's `ensure_results_db`. (#329)

**Full changelog:** see `CHANGELOG.md`.
