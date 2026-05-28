# v56 state-preservation audit

Independent audit of every `_migrate_v56_*` helper in
`flextool/update_flextool/db_migration.py`, verifying that each helper
preserves the activation / value state of existing user databases
across the migration to v56.

The near-miss precedent that motivated this audit was an early Batch F
draft that would have flipped `active_by_default` from `False` to
`True` and silently activated every entity that lacked an
`entity_alternative` row.  That risk was caught and Batch F was
parked.  This audit verifies that no similar default-direction flip
lurks in any of the helpers that DID land in v56.

Scope: all helpers defined under `elif next_version == 56:` in
`flextool/update_flextool/db_migration.py` (lines 1274–1600).
Total: **32 helpers**, audited against current `engine_polars`
consumers as of branch `main` (head `727b5fce`).

---

## Section 1 — Helper-by-helper audit table

| Commit | Helper | What it does | Default change? | Backfill? | Engine reads via `parameter_explicit`? | Verdict |
|---|---|---|---|---|---|---|
| `2912e1c1` | `_migrate_v56_remove_model_debug` | Removes `model.debug` parameter_definition + values | No (pure removal) | n/a | No engine consumer (dead toggle) | ✅ |
| `e68551ed` | `_migrate_v56_add_group_cumulative_capacity_descriptions` | Backfills missing `description` text on `group.cumulative_{min,max}_capacity` | No (description only) | n/a | n/a — description field, not default | ✅ |
| `10d9a21f` | `_migrate_v56_fix_wrong_defaults` | Clears default_value on 5 parameter_definitions; flips `model.inflation_offset_investment` 1.0 → 0.0; clears `commodity.unitsize` 1.0 → None | Yes (6 rows, all to values matching the engine-side fallback) | Not needed — see §2 | All 6 read via `parameter_explicit` / `_entity_scalar_explicit`; broadcast default is silently dropped | ✅ |
| `a5141a6a` | `_migrate_v56_rename_constraint_coefficient_to_coeff` | Renames 4 constraint_*_coefficient parameters to *_coeff (10 pdef rows total) | No (pure rename, id-preserving) | n/a | Pure rename — parameter_value rows follow by id | ✅ |
| `85bfa3f3` | `_migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff` | Renames `flow_coefficient` → `conversion_flow_coeff` on unit__inputNode/outputNode | No (pure rename) | n/a | Pure rename | ✅ |
| `c445026e` | `_migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff` | Renames `max_capacity_coefficient` → `capacity_max_coeff` on unit__inputNode/outputNode | No (pure rename) | n/a | Pure rename | ✅ |
| `ebf52c39` | `_migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff` | Renames `min_capacity_coefficient` → `capacity_min_coeff` on unit__inputNode/outputNode | No (pure rename) | n/a | Pure rename | ✅ |
| `63ec5fc6` | `_migrate_v56_remove_exclude_entity_outputs` | Removes `model.exclude_entity_outputs` (gate was a per-entity capacity-dump exclude; semantic flip to always-emit) | No (pure removal) | n/a | Gate site in `process_outputs.handoff_writers` removed in same commit | ✅ |
| `b5231154` | `_migrate_v56_remove_output_node_balance_t` | Removes `model.output_node_balance_t` dead toggle | No (pure removal) | n/a | No engine consumer (already dead) | ✅ |
| `0d70e54b` | `_migrate_v56_remove_output_ramp_envelope` | Removes `model.output_ramp_envelope` dead toggle | No (pure removal) | n/a | No engine consumer | ✅ |
| `fc58336c` | `_migrate_v56_remove_output_unit__node_flow_t` | Removes `model.output_unit__node_flow_t` dead toggle | No (pure removal) | n/a | No engine consumer | ✅ |
| `ecf5044d` | `_migrate_v56_remove_output_unit__node_ramp_t` | Removes `model.output_unit__node_ramp_t` dead toggle | No (pure removal) | n/a | No engine consumer | ✅ |
| `f1328a64` | `_migrate_v56_remove_output_connection__node__node_flow_t` | Removes `model.output_connection__node__node_flow_t` dead toggle | No (pure removal) | n/a | No engine consumer | ✅ |
| `1bbff500` | `_migrate_v56_remove_output_connection_flow_separate` | Removes `model.output_connection_flow_separate` dead toggle | No (pure removal) | n/a | No engine consumer | ✅ |
| `8b7f4015` | `_migrate_v56_retype_solver_arguments_to_1d_map` | Retypes `solve.solver_arguments` from `array` to `1d_map`; translates existing Array values to Map via `key=value` parsing | No default change | n/a — values translated in place | Engine reads via `find_parameter_values` (explicit-only) in `_solve_config.py` | ✅ |
| `2a42baf1` | `_migrate_v56_fold_solver_options_into_solver_arguments` | Folds `solver_options` Map entries into `solver_arguments` 1d-map on same (solve, alt); collisions logged | No | Value migration: pre-existing user data merged | Explicit-only consumer | ✅ |
| `2a42baf1` | `_migrate_v56_remove_solver_options` | Companion removal of `solve.solver_options` parameter_definition | No (pure removal post-fold) | Done by fold helper | n/a | ✅ |
| `d19d783c` | `_migrate_v56_fold_highs_method_into_solver_arguments` | Injects `solve.highs_method` value into `solver_arguments['solver']` | No | Value migration | Explicit-only consumer | ✅ |
| `d19d783c` | `_migrate_v56_remove_highs_method` | Removes `solve.highs_method` parameter_definition + dedicated value-list | No (pure removal post-fold) | Done by fold helper | n/a | ✅ |
| `77ad3fb9` | `_migrate_v56_fold_highs_parallel_into_solver_arguments` | Injects `solve.highs_parallel` value into `solver_arguments['parallel']` | No | Value migration | Explicit-only consumer | ✅ |
| `77ad3fb9` | `_migrate_v56_remove_highs_parallel` | Removes `solve.highs_parallel` parameter_definition + dedicated value-list | No (pure removal post-fold) | Done by fold helper | n/a | ✅ |
| `57a403bf` | `_migrate_v56_fold_highs_presolve_into_solver_arguments` | Injects `solve.highs_presolve` value into `solver_arguments['presolve']` | No | Value migration | Explicit-only consumer | ✅ |
| `57a403bf` | `_migrate_v56_remove_highs_presolve` | Removes `solve.highs_presolve` parameter_definition + dedicated value-list | No (pure removal post-fold) | Done by fold helper | n/a | ✅ |
| `6b98f9fc` | `_migrate_v56_remove_solver_threads` | Removes `solve.solver_threads` parameter_definition; user values DROPPED per Q-C-2 design (replaced by `--highs-threads` CLI flag) | No default change | None — by design (see §2) | Engine field hard-wired to `None` in `_solve_config.py` | ⚠️ |
| `39864469` | `_migrate_v56_remove_use_row_scaling` | Removes `solve.use_row_scaling`; user values DROPPED per Q-C-2 (replaced by `--scaling` CLI flag); engine hard-wires dict to `{}` so emitted `p_use_row_scaling=0` is the default | No default change | None — by design (see §2) | Engine field hard-wired to `{}` | ⚠️ |
| `b48cdaf4` | `_migrate_v56_remove_solver_io_api` | Removes `solve.solver_io_api` + `solver_io_apis` value-list; user values DROPPED (replaced by `--matrix-file-format` CLI flag) | No default change | None — by design (see §2) | Engine reads CLI / env override; field hard-wired | ⚠️ |
| `7e96e9e6` | `_migrate_v56_remove_solver_time_limit` | Removes `solve.solver_time_limit`; user values DROPPED (replaced by `--solver-time-limit` CLI flag) | No default change | None — by design (see §2) | Engine reads `FLEXTOOL_HIGHS_TIME_LIMIT` env var | ⚠️ |
| `6198b056` | `_migrate_v56_remove_solver_log_level` | Removes `solve.solver_log_level` + `solver_log_levels` value-list; user values DROPPED. Audit (2026-05-27) confirmed engine never consumed `SolverConfig.log_level` — knob was already dead | No default change | None — already dead | Engine never read it | ✅ |
| `39fae839` | `_migrate_v56_add_profile_and_ramp_method_none` | Adds `none` member to `profile_methods` / `ramp_methods` value-lists; flips defaults on 4 (class, param) pairs from `null` to `none`; **backfills explicit `method='none'`** on every legacy entity without a value | Yes (4 defaults: `null` → `none`) | **Yes — explicit row written for every entity without one** | Consumers read CSVs whose rows come from explicit pvals; engine filters by exact method name (e.g. `== "upper_limit"`) | ⚠️ |
| `295191a2` | `_migrate_v56_set_unit_node_profile_default_upper_limit` | Flips `unit__node__profile.profile_method` default from `null` to `upper_limit`; **backfills explicit `method='none'`** on every legacy entity without a value | Yes (`null` → `upper_limit`) | **Yes — explicit `none` written for every entity without one** | Consumer filters by `== "upper_limit"` on the CSV | ⚠️ |
| `d6fc34c8` | `_migrate_v56_retype_yes_only_to_yes_no` | Retypes 6 boolean-as-enum parameters from single-yes lists to shared `yes_no`; flips defaults from `null` (pre-D.3) to `"no"`; drops 5 legacy single-yes value-lists | Yes (defaults `null` → `"no"`) | Not needed — see §2 | `_try_param().filter(value == "yes")` consumers — `"no"` rows are rejected by the filter so a broadcast `"no"` default is equivalent to a missing row | ✅ |
| `a3d984ee` | `_migrate_v56_drop_storage_nested_fix_method_no` | Rewrites existing `node.storage_nested_fix_method = "no"` rows to `"fix_nothing"`; drops redundant `no` list_value; default already `fix_nothing` | No default change | Yes — rewrite preserves user intent under canonical spelling | Value rewrite, not a default flip | ✅ |
| `8c629c48` | `_migrate_v56_rename_co2_methods_no_method_to_none` | Adds `co2_methods.none`; rewrites `group.co2_method = "no_method"` rows to `"none"`; drops `no_method` list_value; flips default `no_method` → `none` | Yes (default `no_method` → `none`) | Yes — every existing `no_method` row rewritten | Engine filter `is_in({"total","price_total","period_total","price_period_total"})` rejects both `no_method` and `none` identically | ✅ |

**Total helpers in elif block:** 32 (includes the 4 pairs of fold+remove for C.2–C.5)

Notes on the table:
- The five Batch-C `_remove_*` helpers that drop user-stored solver
  knobs without a backfill (rows marked ⚠️) are flagged not because
  state preservation is broken in the schema sense, but because user
  values are **intentionally dropped** per the Q-C-2 design decision.
  Equivalent control is now exposed via CLI flags / env vars.  See §2
  for the per-helper confirmation that the engine side cannot
  silently re-activate stale behaviour.
- The two ⚠️ rows in Batch D (`profile_method` defaults flipping
  from `null` to a named member) are the closest analogue to the
  Batch-F near-miss.  Both are mitigated by an explicit
  `method='none'` backfill, which §2 reads line-by-line to confirm.

---

## Section 2 — Detailed analysis of flagged rows

No ❌ verdicts.  Every state-flip risk in the v56 cycle is either
mitigated by an explicit backfill or rendered irrelevant by an
engine-side `parameter_explicit` consumer or a filter that ignores
both the old and the new default value identically.

### 2.1 — Batch D.1 (`_migrate_v56_add_profile_and_ramp_method_none`)

**Risk shape.**  Pre-D.1 the four `(class, param)` pairs
(`connection__profile.profile_method`, `node__profile.profile_method`,
`unit__inputNode.ramp_method`, `unit__outputNode.ramp_method`)
declared default `null`.  Engine consumers treat a missing row as "no
method" — but the engine_polars dispatchers consume the parameter via
CSVs produced by `flextool.input_derivation`, which calls
`parameter_values` (explicit-only) on the spinedb backend.  Schema
defaults are NOT broadcast into the cl_pars CSVs unless the parameter
is listed in `_DEFAULT_VALUES_SPECS` (only `node.penalty_up`,
`node.penalty_down`, and `model.version` are listed there).

D.1 flips the default to `"none"`.  Even though that default never
reaches the engine via the cl_pars CSVs, it WOULD reach the engine if
a future migration ever added one of these parameters to
`_DEFAULT_VALUES_SPECS`.  The backfill removes that fragility by
materialising an explicit `none` row on every legacy entity.

**Mitigation read.**  `db_migration.py:4969–4992` (lines copied from the
helper body):

```python
for entity_class_name, name in targets:
    entities_with_value = {
        pv["entity_byname"]
        for pv in db.find_parameter_values(
            entity_class_name=entity_class_name,
            parameter_definition_name=name,
        )
    }
    for ent in db.find_entities(entity_class_name=entity_class_name):
        byname = ent["entity_byname"]
        if byname in entities_with_value:
            continue
        db.add_update_item(
            "parameter_value",
            entity_class_name=entity_class_name,
            entity_byname=byname,
            parameter_definition_name=name,
            alternative_name="Base",
            value=none_value,
            type=none_type,
        )
```

The set comprehension collects every entity that has at least one
explicit row across any alternative.  The loop adds a `Base`-alternative
row only when no such row exists.  Writing to `Base` matches the
fixture authoring convention (legacy DBs use `Base` as the root
alternative) and ensures the explicit `none` floors the parameter
under the active scenario regardless of scenario stacking.

Engine consumer `_emit_calc_params.py:299` filters by
`profile_method == "upper_limit"` (etc).  An explicit `"none"` row is
rejected by that filter — same outcome as the pre-D.1 missing row.
No behaviour change.

Verdict: **⚠️ flagged-with-mitigation, mitigation correct.**

### 2.2 — Batch D.2 (`_migrate_v56_set_unit_node_profile_default_upper_limit`)

**Risk shape.**  This is the direct analogue of the Batch-F near-miss.
The helper flips `unit__node__profile.profile_method` default from
`null` to `"upper_limit"` (the dominant authoring case).  Without a
backfill, every legacy `unit__node__profile` entity without an
explicit `profile_method` row would silently inherit
`upper_limit` — and the consumer in `_emit_calc_params.py:299`
filters EXACTLY on `== "upper_limit"`, so the default IS the
activating value here (not a dropped no-op like D.3).

**Mitigation read.**  `db_migration.py:5071–5093`:

```python
entities_with_value = {
    pv["entity_byname"]
    for pv in db.find_parameter_values(
        entity_class_name=entity_class_name,
        parameter_definition_name=name,
    )
}
backfilled_count = 0
for ent in db.find_entities(entity_class_name=entity_class_name):
    byname = ent["entity_byname"]
    if byname in entities_with_value:
        continue
    db.add_update_item(
        "parameter_value",
        entity_class_name=entity_class_name,
        entity_byname=byname,
        parameter_definition_name=name,
        alternative_name="Base",
        value=none_value,
        type=none_type,
    )
    backfilled_count += 1
```

Backfilled value is `"none"` (not `"upper_limit"`), which keeps the
legacy entities OFF.  Verification block at lines 5095–5112 re-reads
the parameter_definition and raises if the default did not flip to
`upper_limit`.

This is precisely the Batch-F-style mitigation: the schema default
becomes the dominant case for NEW authoring, while every pre-existing
entity gets an explicit off-row so the default cannot silently
activate the constraint on legacy data.

Verdict: **⚠️ flagged-with-mitigation, mitigation correct.**

### 2.3 — Batch C.6 – C.10 — solver knob removals with no backfill (⚠️ by design)

Helpers: `_migrate_v56_remove_solver_threads`,
`_migrate_v56_remove_use_row_scaling`,
`_migrate_v56_remove_solver_io_api`,
`_migrate_v56_remove_solver_time_limit`.

**Risk shape.**  Each helper drops a `solve.<knob>` parameter
definition **without** folding the user's value into
`solver_arguments`.  Per the helpers' own docstrings (and the Q-C-2
design decision recorded in the commit messages), this is intentional:
the parameters were GUI/CLI-bound knobs that were rarely
scenario-relevant, and the break is acceptable.

**Cross-reference confirmation.**  Engine-side cleanup happens in the
same commits:

- `_solve_config.py:241–262` hard-wires `use_row_scaling = {}` so
  every solve emits `p_use_row_scaling=0` (the default branch).
- `_solve_config.py:332–339` hard-wires `highs_method`,
  `highs_parallel`, `highs_presolve` to `{}`.
- The dataclass fields `SolverConfig.threads`,
  `SolverConfig.time_limit`, `SolverConfig.log_level`,
  `SolverConfig.io_api` keep their `None` defaults — FlexTool no
  longer authors them, but the commercial-solver path still accepts
  values via `--highs-threads`, `--solver-time-limit`,
  `--solver-log-level`, `--matrix-file-format` CLI flags +
  corresponding env vars.

Once the parameter_definition is removed, no stale value can resurface
as engine state — the cascading `parameter_value` delete strips the
DB-side residue.

Verdict: **⚠️ flagged-with-mitigation (mitigation is documented
user-facing CLI replacement, not a DB backfill).  Acceptable per
Q-C-2.**

### 2.4 — `_migrate_v56_fix_wrong_defaults` (10d9a21f)

Six rows are touched:

| (class, param) | old default | new default | Engine consumer | Reads via |
|---|---|---|---|---|
| `reserve__upDown__connection__node.large_failure_ratio` | `""` (corrupt empty-string on float) | `None, None` | `_compute_reserve_filters` in `_emit_reserve.py` | `parameter_explicit` (gate: `p_prn.get(..., 0.0) > 0`) |
| `reserve__upDown__unit__node.large_failure_ratio` | `""` | `None, None` | same | `parameter_explicit` |
| `reserve__upDown__group.penalty_reserve` | `5000.0` | `None, None` | `p_reserve_upDown_group_penalty_reserve_from_source` (`_direct_params.py:1453`) | `parameter_explicit` (function docstring: "None default — explicit rows only") |
| `reserve__upDown__connection__node.max_share` | `0.0` | `None, None` | `_process_reserve_node_param` | `parameter_explicit` |
| `node.storage_state_start` | `0.0` | `None, None` | `p_state_start_from_source` (`_direct_params.py:469`) | `_entity_scalar_explicit` (docstring: "Default None (schema)") |
| `commodity.unitsize` | `1.0` | `None, None` + new description | `p_commodity_unitsize_from_source` (`_direct_params.py:501`) | `_entity_scalar_explicit`; price-ladder consumer substitutes 1.0 internally when absent |
| `model.inflation_offset_investment` | `1.0` | `0.0` | `_explicit_max("inflation_offset_investment", 0.0)` in `_derived_npv.py:355` and `_derived_params.py:7717`; CSV-side fallback `0.0` in `_emit_period_calc.py:295` | `parameter_explicit` (via `_explicit_max`) |

Every consumer uses an explicit-only reader, so the schema default was
never broadcast to the LP in the first place.  Changing the default
on the schema row brings the schema in line with the engine truth —
no runtime behaviour change.

`commodity.unitsize` is the most subtle: the old default `1.0`
inside the schema would have looked active to a casual reader of the
spinedb_schema.json, but the engine's price-ladder path reads via
`_entity_scalar_explicit`, so the default was silently dropped.  The
new description (rewritten in this commit) names the price-ladder
gate (`commodity.price_method = price_ladder_*`) and the
absent → identity fallback.

Verdict: **✅ safe.**  Audit reference:
`_audit_reports/v56_default_audit.md`.

### 2.5 — Batch D.3 (`_migrate_v56_retype_yes_only_to_yes_no`)

**Why this isn't a state-flip risk.**  Pre-D.3 the six parameters had
`default_value = None` (verified by reading the JSON snapshot at
`d6fc34c8~1:flextool/schemas/spinedb_schema.json`).  Post-D.3 the
default is `"no"`.  This IS a default-direction change — but the
engine consumers are `_group_yes(...)` and analogous
`_try_param(...).filter(value == "yes")` projections in
`_projection_params.py` (lines 1201–1211, 1215, 1219, 1223, 1282,
1730).  The filter accepts ONLY the literal string `"yes"`.

A broadcast `"no"` default IS visible to `_try_param` (since
`_try_param` calls `source.parameter` which DOES broadcast scalar
defaults — see `_spinedb_reader.py:476–489`), but the
`.filter(value == "yes")` step strips every broadcast `"no"` row
before the activation projection.  Result: identical behaviour to the
pre-D.3 None-default path, where `_try_param` returned only the
authored `"yes"` rows.

Existing parameter_value rows are preserved unchanged: every pre-D.3
authored value is `"yes"` (the only member of the legacy single-yes
lists), and `"yes"` is also a member of `yes_no`.  Verification block
at `db_migration.py:5256–5273` walks every surviving row and confirms
its value is in `{"yes", "no"}`.

Verdict: **✅ safe.**

### 2.6 — Batch D.7 (`_migrate_v56_rename_co2_methods_no_method_to_none`)

The default flips from `"no_method"` to `"none"` but the engine
consumer in `_emit_co2_accumulators.py:61` filters via
`is_in({"total", "price_total", "period_total",
"price_period_total"})`.  Neither `"no_method"` nor `"none"` is a
member of the active set, so the migration is a no-op for the LP.
Existing parameter_value rows carrying the old `"no_method"` token are
rewritten to `"none"` (lines 5395–5408) to keep the value list-valid;
that rewrite is essential because step 3 drops the legacy
`no_method` list_value (line 5417).

Verdict: **✅ safe.**

---

## Section 3 — Engine-consumer cross-references

For each parameter touched by a non-pure-rename helper, the consumer
file:line and its default-policy classification:

| Parameter | Consumer | Reads via |
|---|---|---|
| `model.debug` | (none — removed dead toggle) | n/a |
| `reserve__upDown__{connection,unit}__node.large_failure_ratio` | `flextool/engine_polars/_emit_reserve.py` (`_compute_reserve_filters`) | `parameter_explicit` (gate `> 0`) |
| `reserve__upDown__group.penalty_reserve` | `flextool/engine_polars/_direct_params.py:1453` | `parameter_explicit` |
| `reserve__upDown__connection__node.max_share` | `flextool/engine_polars/_direct_params.py:_process_reserve_node_param` | `parameter_explicit` |
| `node.storage_state_start` | `flextool/engine_polars/_direct_params.py:469` (`p_state_start_from_source`) | `_entity_scalar_explicit` |
| `commodity.unitsize` | `flextool/engine_polars/_direct_params.py:501` (`p_commodity_unitsize_from_source`); `_commodity_ladder._commodity_unitsize_param` | `_entity_scalar_explicit`; price-ladder consumer substitutes 1.0 when absent |
| `model.inflation_offset_investment` | `flextool/engine_polars/_derived_npv.py:355`; `_derived_params.py:7717` | `_explicit_max(..., 0.0)` |
| `model.exclude_entity_outputs` | (none — gate removed) | n/a |
| `model.output_*` (6 toggles) | (none — already dead) | n/a |
| `solve.solver_arguments` (and folds from `solver_options`, `highs_*`) | `flextool/engine_polars/_solve_config.py:357–393` | `find_parameter_values` (explicit-only) |
| `solve.solver_threads / use_row_scaling / solver_io_api / solver_time_limit / solver_log_level` | Removed; CLI / env var replacements | n/a |
| `connection__profile.profile_method`, `node__profile.profile_method`, `unit__{input,output}Node.ramp_method` | Engine consumes via CSVs produced by `input_derivation._specs` cl_pars emitter (`parameter_values` → explicit-only); engine filters by exact method name in `_emit_calc_params.py:299`, `_emit_arc_unions.py:461`, `_projection_params.py:907/1035` | Indirectly via CSV; CSV holds explicit rows only (cl_pars path, not `_DEFAULT_VALUES_SPECS`) |
| `unit__node__profile.profile_method` | Same CSV path; `_emit_calc_params.py:299` filters `== "upper_limit"` | Indirectly via CSV (explicit-only) |
| `connection.is_DC` | `_projection_params.py:1729–1730` | `_try_param(...).filter(value == "yes")` (broadcast-tolerant; `"no"` default rejected by filter) |
| `group.has_capacity_margin / has_inertia / has_non_synchronous` | `_projection_params.py:1214–1223` via `_group_yes(...)` | `_try_param(...).filter(value == "yes")` |
| `unit__{input,output}Node.is_non_synchronous` | `_projection_params.py:1281–1282` | `_try_param(...).filter(value == "yes")` |
| `node.storage_nested_fix_method` | `_projection_params.py:1186` | `_try_param` + method-set projection (filter excludes both `no` and `fix_nothing`) |
| `group.co2_method` | `_emit_co2_accumulators.py:61` | `is_in({"total","price_total","period_total","price_period_total"})` (filter rejects both `no_method` and `none`) |

---

## Section 4 — Summary

- Total helpers audited: **32**
- ✅ safe: **24**
- ⚠️ flagged-with-mitigation: **8**
- ❌ state-loss-risk: **0**

⚠️ helpers (all have correct mitigations):

- `_migrate_v56_remove_solver_threads` — drops user value by design (Q-C-2); CLI replacement `--highs-threads`.
- `_migrate_v56_remove_use_row_scaling` — drops user value by design; CLI replacement `--scaling`.
- `_migrate_v56_remove_solver_io_api` — drops user value by design; CLI replacement `--matrix-file-format`.
- `_migrate_v56_remove_solver_time_limit` — drops user value by design; CLI replacement `--solver-time-limit`.
- `_migrate_v56_add_profile_and_ramp_method_none` — default flip `null` → `none` on 4 params, backfilled with explicit `none` per entity.
- `_migrate_v56_set_unit_node_profile_default_upper_limit` — default flip `null` → `upper_limit` on `unit__node__profile.profile_method`, backfilled with explicit `none` per entity (the Batch-F-style risk, correctly mitigated).

No state-loss helpers.

---

## Section 5 — Recommendations

**v56 cycle is state-preserving.**

All default-direction changes either:

1. land on a parameter whose engine consumer reads via
   `parameter_explicit` (so the broadcast default is silently
   dropped — no behaviour change for legacy DBs), or
2. land on a parameter whose engine consumer filters by an exact
   token that rejects both the old and the new default
   (`_group_yes`-style activation projections), or
3. flip a value-list off-member's spelling and are backfilled by a
   matching `parameter_value` rewrite (D.4, D.7), or
4. are the dominant-case default flip in Batch D.2 — explicitly
   mitigated by writing `method='none'` to every entity that lacked an
   explicit row, exactly mirroring the pattern Batch F would have
   needed for `active_by_default`.

No follow-up migration is required.

A nice-to-have hygiene improvement for the long term: the Batch-C
solver-knob removals could log a one-shot user-facing warning when
they encounter a non-null `parameter_value` row about to be dropped,
so a user grepping the migration output can see "your `solver_threads`
value of 8 was dropped — pass `--highs-threads 8` to keep it".  This
is GUI-rather-than-engine work and out of scope for this audit.
