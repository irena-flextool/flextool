# v56 `parameter_type_list` + description-suffix audit — scoping report

Read-only investigation for the in-progress `Batch E — valid_types + description
hygiene` task (worktree `docs-v4-refresh`, branch `docs-v4-refresh`). This
report only scopes; no schema or code edits were made.

## Q1 — Is `flextool/schemas/flextool_axis_contract.json` the source of truth?

**Verdict: NO.** The lead is incorrect. `flextool_axis_contract.json` is a
contract for *axis enums* (LP-dimension vocabularies), not for
`parameter_type_list`.

Evidence:

- File structure (`/home/jkiviluo/.../flextool/schemas/flextool_axis_contract.json`,
  291 lines):
  - top-level keys: `_comment, _doc, _schema_version,
    _template_master_version_compat, axes, synthetic_token_allowlist,
    mixed_vocab_columns, non_dim_columns, _review_notes`.
  - Each `axes[]` record carries `{name, label, source_type, source, filter,
    tokens, column_synonyms, note}` — i.e. how to enumerate a `pl.Enum` axis
    vocabulary (n, p, c, d, t, branch, block, …).
  - **Nothing about Spine parameter `valid_types` / `parameter_type_list` /
    accepted Spine value types per parameter.**
- Loader `flextool/spinedb_backend/_axis_enums.py:187 load_axis_contract`
  returns an `AxisContract` dataclass containing `axes`,
  `synthetic_token_allowlist`, `mixed_vocab_columns`, `non_dim_columns` —
  again no parameter-type metadata.
- Companion `flextool/schemas/AXIS_CONTRACT.md` explicitly states the file
  is for "every axis the cascade casts to `pl.Enum`". The accompanying
  `flextool_axis_contract.schema.json` JSON-Schema validates the same axes
  shape.

**The actual source of truth for parameter type lists in v56 is the
`parameter_types` block of `flextool/schemas/spinedb_schema.json` itself.**
The schema JSON has these top-level keys: `entity_classes, entities,
parameter_value_lists, parameter_groups, parameter_definitions,
parameter_types, alternatives`.

- `parameter_definitions` row layout: `[entity_class, name, default_value,
  value_list_name, description, parameter_group_name]` — 6 columns,
  carries the description.
- `parameter_types` row layout: `[entity_class, name, type, depth]` — N
  rows per parameter, one row per accepted Spine value type. Encoding:
  - `float, depth=0`     ↔ scalar float.
  - `str, depth=0`       ↔ scalar string.
  - `array, depth=1`     ↔ Spine Array.
  - `map, depth=k`       ↔ k-d_map (`1d_map`, `2d_map`, `3d_map`, `4d_map`).

Spine-DB-API translates these rows into the `parameter_type_list` tuple
the runtime sees (e.g. rows `[(float,0), (map,1), (map,3)]` → tuple
`("float", "1d_map", "3d_map")`).

Coverage:
- 215 `parameter_definitions` rows.
- 207 of them have ≥1 `parameter_types` row.
- **8 definitions have NO `parameter_types` row at all** (under-specified — every
  type is implicitly accepted, which Spine treats as legacy behaviour):
  - `connection.constraint_cumulative_pre_built_capacity_coeff`
  - `node.constraint_cumulative_pre_built_capacity_coeff`
  - `unit.constraint_cumulative_pre_built_capacity_coeff`
  - `timeset.timeset_weights`
  - `unit__inputNode.capacity_max_coeff`
  - `unit__inputNode.capacity_min_coeff`
  - `unit__outputNode.capacity_max_coeff`
  - `unit__outputNode.capacity_min_coeff`

The `db_migration.py` runtime calls (`db.add_update_item("parameter_definition",
..., parameter_type_list=...)`) used to be the historical source-of-truth
during v23 → v55 migration, but for a freshly-imported v56 template the
SpineDB API populates `parameter_type_list` from the `parameter_types`
block at template-load time. The schema JSON is therefore both the
authoritative snapshot and the editable surface.

## Q2 — Other layers that add their own type constraints

Beyond Spine's `parameter_type_list` (declared in
`spinedb_schema.json::parameter_types`), three engine layers add narrower
constraints:

1. **`flextool/engine_polars/_param_shapes.py` — `PARAM_ALLOWED_SHAPES`**
   - Per-`(entity_class, parameter_name)` allow-list of `Shape` enums
     (`SCALAR | MAP_PERIOD | MAP_TIME | MAP_PERIOD_TIME | MAP_TIME_PERIOD`).
   - Covers only the parameters routed through `resolve_param_shape` +
     `broadcast_to_period[_time]` (the period/time family — `availability`,
     `co2_price`, `inflow`-adjacent, `other_operational_cost`, etc.).
   - Tighter than the schema. Notable disagreements:
     - `node.availability`, `unit.availability`,
       `reserve__upDown__group.reservation`,
       `unit__inputNode.other_operational_cost`,
       `unit__outputNode.other_operational_cost`,
       `node.storage_state_reference_value` — `PARAM_ALLOWED_SHAPES`
       includes `MAP_PERIOD_TIME` (a 2d_map[period,time]); the schema
       does NOT list `("map", 2)` for these.
     - `unit.other_operational_cost` — registered in `PARAM_ALLOWED_SHAPES`
       (all 4 shapes), but has **zero rows in `parameter_types`** (and
       wasn't in the earlier `unit__outputNode` audit either — it lives on
       `unit` itself per the Tier-3 silent-default migration).
     - `connection.other_operational_cost` — `PARAM_ALLOWED_SHAPES` says
       all 4 shapes; schema lists `("float", "1d_map")` only.

2. **`flextool/input_derivation/_specs.py` — `filter_in_type`**
   - Per `_PARAMETER_SPECS` entry, the list of Spine value-type codes the
     EAV → CSV materialiser accepts for that destination file (e.g.
     `commodity.price` → "1d_map" rows go to `pdt_commodity.csv`,
     "float"/"str" rows go to `p_commodity.csv`).
   - Doesn't *define* legality, it *partitions* values. Implicit
     constraint: every value type that arrives must be listed by at least
     one spec for the (class, param). A type missing from both `filter_in_type`
     and the schema's `parameter_types` would be silently dropped — a known
     hazard already enforced via `_backend.py:510-522` (single Nd_map per
     spec).
   - No disagreement with the schema for the canonical params; the
     engine and schema agree the spec set is exhaustive.

3. **`flextool/export_to_tabular/sheet_config.py:46 classify_param_types`**
   - Maps `parameter_type_list` tuples → set of sheet layouts
     `{constant, periodic, timeseries, stochastic}`.
   - This is the **mechanical decoder** the schema description suffixes
     should mirror:
     - `float`/`str` → `constant`
     - `array` → `constant`
     - `2d_map` → `constant`
     - `1d_map` w/o `3d_map` → `periodic`
     - `1d_map` w/ `3d_map` → `timeseries` (pure time-series — no
       constant fallback)
     - `3d_map` → `stochastic` + `timeseries`
     - `4d_map` → `stochastic`

4. **`flextool/spinedb_backend/_backend.py:510-522` — single-Nd_map rule**
   - At most one `Nd_map` per `filter_in_type` (write helper invariant).
   - Bears on `_specs.py` not on the schema directly.

5. **No `isinstance` / explicit type checks** in `_direct_params.py`,
   `_derived_params.py`, `_derived_arithmetic.py`, `_solve_config.py`,
   `_axis_enums.py` — those modules consume already-shaped polars frames
   produced by `_param_shapes.py` / the cascade and trust the Spine API
   payload.

**Net:** The schema's `parameter_types` block is the authoritative declaration
that the v56 user-facing layer (GUI, Excel roundtrip, docs) reads.
`_param_shapes.PARAM_ALLOWED_SHAPES` adds engine-side enforcement for the
narrower (period, time) family and currently is in mild disagreement with
the schema (engine accepts 2d_map for several params that schema doesn't
declare with `("map", 2)`). The implementation phase should widen the
schema to cover what the engine actually accepts, not narrow the engine.

## Q3 — Is `parameter_type_list ↔ description suffix` mechanical?

**Verdict: (b) mechanical for ~most rows, with a small set of exceptions.**

Statistics from `parameter_definitions` (215 rows) vs the convention in
`classify_param_types`:

| suffix family                   | count | dominant type-list                            |
|---------------------------------|------:|------------------------------------------------|
| `Constant.`                     | 41    | `('float',)`                                   |
| `Constant or period.`           | 36    | `('float', '1d_map')`                          |
| `Constant or Period` (cap.)     | 39    | `('float', '1d_map')` (case inconsistency)     |
| `Period.`                       | 14    | `('1d_map',)`                                  |
| `Time.`                         |  1    | `('1d_map', '3d_map')`                         |
| `Constant or time.`             |  9    | mixed: 6× w/ 3d_map (wrong), 3× w/o 3d_map     |
| `Constant, period or time.`     |  7    | `('float', '1d_map')` (missing 3d_map?)        |
| `Constant, Period or Time.`     |  7    | mixed: half w/ 3d_map (mis-cased + missing)    |

Mechanical rule (proposed, verified against the engine convention in
`classify_param_types` and `Shape` enum):

```
            tuple                          canonical suffix
─────────────────────────────────────────────────────────────────
("float",)                                 Constant.
("str",)                                   Constant.          (string-valued)
("float", "1d_map")  if 1d_map ⇒ period    Constant or period.
("float", "1d_map")  if 1d_map ⇒ time      Constant or time.
("1d_map",)          period-indexed        Period.
("1d_map",)          time-indexed          Time.
("float", "1d_map", "3d_map")              Constant, time or stochastic time.
("1d_map", "3d_map")                       Time or stochastic time.
("float", "1d_map", "2d_map", "3d_map")    Constant, period, time, period+time or stochastic time.
("2d_map",)                                Period and time.
("4d_map",)                                Stochastic-branch map.    (only `solve.stochastic_branches`)
("array",)                                 Array.                    (e.g. `solve.contains_solves`)
```

The "ambiguity" sits in `("float", "1d_map")` and `("1d_map",)` because
the schema doesn't carry per-Map index-name annotations — the engine
disambiguates at runtime via `_param_shapes.resolve_param_shape`. The
distinction is therefore *not* derivable from the type-list alone for these
families; it requires:

- the parameter's intended use (period-only invest budget vs. time
  profile), AND
- either (a) the existing description's "period" / "time" wording (when
  trustworthy), OR (b) `PARAM_ALLOWED_SHAPES` registry for the period/time
  family.

Conclusion: a script can mechanically derive ~85% of the canonical
suffixes (the `Constant.` family, the `Period.`-only family, the
`stochastic time` upgrade for any row with `3d_map`/`4d_map`, the
case-normalisation). The remaining 1d_map period-vs-time disambiguation
needs a one-time pass over the ambiguous rows with engine cross-check;
this is a small bounded set (≤50 rows across the families above).

## Q4 — Recommended implementation phase plan

Split into **3 commits**, no sub-agents (mechanical + bounded fix-ups):

### Commit 1 — Mechanical normalisation (single agent, single edit pass)

1. Write a small one-off helper (kept under
   `_audit_reports/scripts/normalise_param_descriptions.py`, or inline in
   the commit) that walks `parameter_definitions`, looks up
   `parameter_types`, and rewrites the trailing accepted-types sentence to
   the canonical form per the Q3 table.
2. Apply only the *non-ambiguous* rewrites:
   - any row with `3d_map` ⇒ ensure description ends with
     `"…stochastic time."` (the most-wrong class — 6 + 4 rows confirmed).
   - any row with `4d_map` ⇒ append `"Stochastic-branch map."` if absent.
   - any row with `("float",)` and a non-`Constant.` suffix ⇒ rewrite to
     `Constant.`.
   - any row with `("1d_map",)` only (no float, no 3d_map) ⇒
     `Period.` / `Time.` from current wording (keep period/time choice).
   - case-normalise `Constant or Period` → `Constant or period.`,
     `Constant, Period or Time.` → `Constant, period or time.` (7 rows).
3. Verify: load the schema with `json.load`, run the script,
   `json.dump(indent=2)`, diff. Should be a pure description-text patch.

No type-list (i.e. `parameter_types` block) edits in this commit.

### Commit 2 — Schema `parameter_types` completion

1. Add `parameter_types` rows for the **8 under-specified definitions**
   listed in Q1.
2. Add `("map", 2)` (2d_map) rows for the 5 params where
   `PARAM_ALLOWED_SHAPES` accepts `MAP_PERIOD_TIME` but the schema doesn't
   declare it:
   - `node.availability`, `unit.availability`,
     `reserve__upDown__group.reservation`,
     `unit__inputNode.other_operational_cost`,
     `unit__outputNode.other_operational_cost`,
     `node.storage_state_reference_value`,
     `commodity.price` (engine allows {SCALAR,MAP_PERIOD,MAP_TIME} — no
     2d_map needed; double-check before adding).
3. Add the missing `unit.other_operational_cost` rows
   (`PARAM_ALLOWED_SHAPES` declares 4 shapes; schema has 0 rows).
4. Verify by re-running the same script — every parameter should now
   have ≥1 `parameter_types` row and `PARAM_ALLOWED_SHAPES` ⊆ declared
   types for every entry.

### Commit 3 — Re-run description normaliser with the now-complete type lists

1. Re-run commit 1's script — the 5 params that gained `2d_map` rows now
   need their suffix updated to e.g.
   `Constant, period, time, period+time or stochastic time.` (or a
   shorter wording the team picks).
2. Manual review (15–25 rows) of the `("float","1d_map")` / `("1d_map",)`
   ambiguous-shape rows where the existing description says "period" but
   the engine accepts "time" too (or vice versa) — cross-check against
   `PARAM_ALLOWED_SHAPES` for the period/time family.

### Migration / verify trio?

- **No `migrate-all` step is needed.** `parameter_type_list` and
  description are stored on each `parameter_definition` row already in
  shipped DBs (carried since v23). Editing the schema JSON updates only
  the *template* shipped with the engine. Existing DBs created from
  older templates retain their stored values; the schema is consulted
  only when a fresh DB is bootstrapped from the template OR when the GUI
  shows the canonical metadata.
- However: any historical DB that lacks the new `("map", 2)` row for
  e.g. `node.availability` will silently reject a user-authored 2d_map.
  If the team wants the schema-template fix to propagate into existing
  fixtures (canonical_databases, test fixtures), add a one-line
  `_migrate_v57_align_parameter_types_to_engine(db)` migration that
  re-sets `parameter_type_list` on the affected rows. This is one
  ~30-line function, no risk to LP semantics. (Or defer to v57 once v56
  is locked.)

### Risk areas

- **Engine-narrowing risk = zero:** these edits only widen `parameter_types`
  (add accepted types) and refine descriptions; they don't remove any
  type the engine already supports.
- **GUI / export risk:** `sheet_config.classify_param_types` depends on
  `parameter_type_list`. Adding `2d_map` to a parameter changes the
  sheet-layout classification:
  - `("float", "1d_map", "2d_map", "3d_map")` ⇒
    `{constant, periodic, timeseries, stochastic}` — same set as
    `("float", "1d_map", "3d_map")`, so no new sheet appears. Safe.
  - Adding `2d_map` without `float/1d_map/3d_map` (won't happen here)
    would force the value onto a constant sheet — irrelevant for this
    audit.
- **Roundtrip xlsx**: Q1 affects only descriptions; the export pipeline
  reads `parameter_type_list` from the DB, not the schema text, so
  description-only edits are inert. The `parameter_types` additions in
  commit 2 take effect after a fresh template import.
- **db_migration.py**: contains historical, version-specific calls. The
  schema-template edits do NOT need a sibling db_migration step *for v56
  shipping*, but the migrate-all pipeline that takes a v55 DB to v56
  should pick up the new types. Check whether `_migrate_v56_*` already
  re-applies `parameter_type_list` for any affected param; if so, mirror
  the schema there. Otherwise note in commit 3 that a v57 migration may
  be needed for legacy DBs (deferrable).

### Tests / fixtures

- No goldens to regenerate — description text and accepted-type
  widening don't change LP outputs.
- `tests/spinedb_backend/test_axis_contract.py` validates the *axis*
  contract, not parameter_types. No update needed.
- Add a new sanity test:
  `tests/spinedb_backend/test_parameter_types_complete.py` that asserts
  - every `parameter_definitions` row has ≥1 `parameter_types` row.
  - `PARAM_ALLOWED_SHAPES[(cls, par)]` ⊆ declared types for every entry
    (i.e. engine never accepts a shape the schema doesn't declare).
  This guards the contract going forward.

## Q5 — Top-10 most-wrong rows

(Confidence: each entry has a direct engine call-site or contract-vs-schema
disagreement.)

| # | `<class>.<param>` | Current type-list | Current suffix | Proposed type-list | Proposed suffix | Engine evidence |
|---|---|---|---|---|---|---|
| 1 | `node.availability` | `('float','1d_map','3d_map')` | `Constant or time.` | `('float','1d_map','2d_map','3d_map')` | `Constant, time, period+time or stochastic time.` | `_param_shapes.py:193` declares `MAP_PERIOD_TIME` accepted; schema lacks `("map", 2)`. |
| 2 | `unit.availability` | `('float','1d_map','3d_map')` | `Constant or time.` | `('float','1d_map','2d_map','3d_map')` | `Constant, time, period+time or stochastic time.` | `_param_shapes.py:206`. |
| 3 | `connection.availability` | `('float','1d_map')` | `Constant or time.` | `('float','1d_map','2d_map','3d_map')` | `Constant, time, period+time or stochastic time.` | `_param_shapes.py:209` declares all 4 shapes; schema only has 2. |
| 4 | `unit.other_operational_cost` | `()` (no rows) | `Constant, Period or Time.` | `('float','1d_map','2d_map','3d_map')` | `Constant, period, time, period+time or stochastic time.` | `_param_shapes.py:240` declares 4 shapes; schema has zero `parameter_types` rows. |
| 5 | `unit__inputNode.other_operational_cost` | `('float','1d_map','3d_map')` | `Constant, Period or Time.` (cap.) | `('float','1d_map','2d_map','3d_map')` | `Constant, period, time, period+time or stochastic time.` | `_param_shapes.py:234`. |
| 6 | `unit__outputNode.other_operational_cost` | `('float','1d_map','3d_map')` | `Constant, Period or Time.` (cap.) | `('float','1d_map','2d_map','3d_map')` | `Constant, period, time, period+time or stochastic time.` | `_param_shapes.py:237`. |
| 7 | `node.inflow` | `('float','1d_map','3d_map')` | `Constant or time.` | (unchanged) | `Constant, time or stochastic time.` | `classify_param_types` says `3d_map ⇒ stochastic`; suffix currently omits it. |
| 8 | `unit.efficiency` | `('float','1d_map','3d_map')` | `Constant or time.` | (unchanged) | `Constant, time or stochastic time.` | Same — `3d_map` present, "stochastic" missing from suffix. |
| 9 | `unit.min_load` | `('float','1d_map','3d_map')` | `Constant or time.` | (unchanged) | `Constant, time or stochastic time.` | Same. |
| 10 | `reserve__upDown__group.reservation` | `('float','1d_map','3d_map')` | `Constant or time.` | `('float','1d_map','2d_map','3d_map')` | `Constant, time, period+time or stochastic time.` | `_param_shapes.py:246` declares 4 shapes; schema missing 2d_map; suffix missing "stochastic". |

Bonus (confident wrongs not in top-10):
- `solve.stochastic_branches`: type-list `('4d_map',)`, description has no
  accepted-types suffix at all — should append `"Stochastic-branch map (period, branch, time, realised/weight)."`.
- `profile.profile`: type-list `('1d_map','3d_map')`, no suffix —
  should append `"Time or stochastic time."`.
- `commodity.price_ladder_annual`: type-list `('2d_map','3d_map')`, no
  suffix — should append `"Stepped supply curve; per-tier (constant) or per-period (stochastic time)."` or similar.
- `node.storage_state_reference_value`: type-list `('float','1d_map')`,
  suffix `Constant.` — should be `Constant, period, time, or period+time.` once `2d_map` is added per `_param_shapes.py:196`.
- `commodity.price`: type-list `('float','1d_map')`, suffix `Constant or period.` is wrong — `_param_shapes.py:215` declares `{SCALAR, MAP_PERIOD, MAP_TIME}`, so suffix should be `Constant, period or time.`.
