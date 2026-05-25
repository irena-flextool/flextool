# FlexTool axis contract

This directory holds two coupled JSON files that together define the
canonical FlexTool axis vocabularies for the `pl.Enum` dtype activation
(see `specs/enum_dtype_refactor_plan.md`):

| File | Purpose |
|---|---|
| `flextool_axis_contract.json` | Authoritative declaration of every axis the cascade casts to `pl.Enum`. Each axis row names the entity class / parameter map / synthetic-token list that supplies its vocabulary. |
| `flextool_axis_contract.schema.json` | JSON Schema (draft 2020-12) validating the structure of `flextool_axis_contract.json` itself. |

The contract is consumed by `flextool/spinedb_backend/_axis_enums.py`
(landing in Phase 1) at runtime to build the per-axis `pl.Enum` dtypes
that the Backend and the cascade cast against.

## Coupling with `spinedb_schema.json`

The contract migrates **hand-in-hand** with `spinedb_schema.json`.
Any schema change that adds, renames, or removes an entity class or a
parameter definition referenced by an axis row REQUIRES updating both
files in the same commit:

* If you add a new entity class that the cascade joins on, decide
  whether it deserves its own axis and add a row to `axes`.
* If you rename an entity class, update every `source` field that
  references it.
* If you remove an entity class, remove or repoint the axis row.
* If you rename or remove a parameter referenced under
  `source_type: parameter_keys` or `parameter_value_list`, update the
  `source.entity_class` / `source.parameter` / `source.parameters` field.

The `_template_master_version_compat` block at the top of
`flextool_axis_contract.json` pins the schema commit hash the contract
was last verified against. Bump that block whenever a schema-affecting
edit lands.

## Self-validation

`tests/spinedb_backend/test_axis_contract.py` validates the contract
against its own schema and cross-checks that every entity class /
parameter referenced by an axis row exists in `spinedb_schema.json`.
Run it after every contract edit:

```bash
python -m pytest tests/spinedb_backend/test_axis_contract.py -v
```

## Adding or changing an axis

1. Decide the `source_type`:
   - `entity_class` — one entity class, single member list.
   - `entity_class_union` — multiple entity classes unioned (e.g. process = unit ∪ connection).
   - `parameter_keys` — vocabulary lives in the keys of a parameter Map (or the values of an Array).
   - `parameter_value_list` — vocabulary is the values of a Spine `parameter_value_list`.
   - `synthetic` — fixed tokens introduced by cascade derivations (no DB source).
   - `union` — composite axis combining other axes' vocabularies.
2. Fill in the `source` field — string / list / object per `source_type`.
3. Add `column_synonyms` if the cascade uses friendly column names that should map to the same enum (mirrors `_AXIS_SYNONYMS` in `flextool/engine_polars/_axis_enums.py`).
4. Add `filter` if the vocabulary is intersected with another set (e.g. scenario filter, realized periods).
5. If introducing synthetic tokens that aren't from the DB, register them in `synthetic_token_allowlist` AND in the axis's `tokens` field (for `source_type: synthetic`) or `tokens_default_extension` (for hybrid axes).
6. Update the schema (`flextool_axis_contract.schema.json`) only if you need a new `source_type` enum value or a new optional field shape.
7. Run the self-check tests.

## Outstanding review notes

The `_review_notes` array at the bottom of `flextool_axis_contract.json`
tracks Phase 0 audit findings that need Phase 1 resolution decisions
(e.g. column-letter collisions between axes like `c` for both commodity
and constraint, or `b` for both block and branch). These notes are
informational; Phase 1's `build_axis_enums` is where the disambiguation
strategy is actually implemented.
