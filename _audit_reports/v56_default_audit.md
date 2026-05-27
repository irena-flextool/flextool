# v56 audit — wrong `default_value` rows in `spinedb_schema.json`

Batch A.2 of the v56 schema cleanup.  Scope: every
`parameter_definition` row whose schema-declared default disagrees with
how the engine actually consumes the parameter.

## Methodology

1. Enumerated the 232 rows in
   `flextool/schemas/spinedb_schema.json :: parameter_definitions`.
2. For every row with a non-null default, traced the consumer chain
   in `flextool/engine_polars/` (mainly `_direct_params.py`,
   `_derived_arithmetic.py`, `_emit_reserve.py`, `_param_shapes.py`)
   and `flextool/process_outputs/`.
3. Classified the row as **high** (engine clearly treats the default
   as "feature disabled" / discards the schema default), **medium**
   (engine handles it via a collapse trick but the contract is muddy),
   **surface** (ambiguous), or **fine** (schema default matches engine
   semantics).
4. Method-enum defaults (`transfer_method = "regular"`,
   `storage_binding_method = "bind_forward_only"`, …) are deliberately
   out of scope — they are batch D.

Column key
- *Current default*: positions 3-4 from the schema array
  (`default_value, parameter_value_list_name`) — for the schema rows
  the engine-affecting field is column 3.
- *Engine path*: which `_direct_params.py`/`_derived_arithmetic.py` /
  `_emit_*.py` helper consumes the value and whether it uses
  `parameter_explicit` (explicit rows only — schema default silently
  ignored) or `parameter` (broadcast — schema default IS used).

## High-confidence rows — patched by `_migrate_v56_fix_wrong_defaults`

| Class | Parameter | Current default | Proposed default | Rationale | Confidence |
|---|---|---|---|---|---|
| `reserve__upDown__connection__node` | `large_failure_ratio` | `""` (empty string) | `None` | Engine consumer in `_emit_reserve.py:546` gates on `p_prn.get(..., 0.0) > 0` — any non-zero enables the N-1 reserve constraint.  The `""` default is a corrupt artefact (engine cannot interpret a string when `parameter_type_list=("float",)`); user-named in the spec.  Sister row `increase_reserve_ratio` on the same class already uses `null, null`. | **high** |
| `reserve__upDown__unit__node` | `large_failure_ratio` | `""` (empty string) | `None` | Same engine consumer at `_emit_reserve.py:546`.  Same `""` artefact.  Sister row `increase_reserve_ratio` on the class already uses `null, null`. | **high** |
| `reserve__upDown__group` | `penalty_reserve` | `5000.0` | `None` | Engine reads via `parameter_explicit` in `p_reserve_upDown_group_penalty_reserve_from_source` (`_direct_params.py:1455`) — schema default 5000 is silently dropped.  Docstring even claims "None default — explicit rows only" (line 1452).  The penalty enters the objective via `vq_reserve * res * pen * op_factor` (`_reserve.py:608`); leaving the default at 5000 misleads users into thinking soft-reserve violations are enabled out-of-the-box when in fact no row → no penalty term → no soft slack. | **high** |
| `reserve__upDown__connection__node` | `max_share` | `0.0` | `None` | Engine reads via `parameter_explicit` in `_process_reserve_node_param` (`_direct_params.py:1487`) — schema default is dropped.  Sister row on `reserve__upDown__unit__node` already uses `null, null` (schema inconsistency).  No LP impact today because the default of 0 happens to be the no-op value, but the schema contract is wrong. | **high** |
| `node` | `storage_state_start` | `0.0` | `None` | Engine reads via `_entity_scalar_explicit` (`_direct_params.py:469`) — explicit rows only.  Docstring already declares the schema contract as "Default ``None`` (schema)." (line 467), but the actual schema row carries `0.0`.  If the schema default were actually honoured (via `parameter`), every storage node would be forced to start at state 0 under the `fix_start` method (which is itself the schema default for `storage_start_end_method`) — a major silent LP perturbation.  Aligning the schema to None matches what the engine already does and what the engine doc already promises. | **high** |

## High-confidence (resolved) — promoted from medium with user approval

These two rows were originally classified ``medium`` and have since been
promoted to ``high`` with explicit user approval; both are patched by
``_migrate_v56_fix_wrong_defaults`` in the same v56 commit as the five
rows above.

| Class | Parameter | Current default | Applied fix | Rationale | Confidence |
|---|---|---|---|---|---|
| `model` | `inflation_offset_investment` | `1.0` | `0.0` (set via `to_database(0.0)`) | Engine fallback in `_derived_npv.py:355` and `_emit_period_calc.py:295` uses `0.0`, not `1.0`.  Engine has a "collapse trick" (`_derived_npv.py:337-347`) that detects "all rows equal the spine default" and substitutes the engine default — so the schema default of 1.0 was silently overridden whenever the user accepted it.  Patched to `0.0` to make the schema default match the engine fallback verbatim. | **high (resolved)** |
| `commodity` | `unitsize` | `1.0` | `None` + rewritten description | Engine reads via `_entity_scalar_explicit` (`_direct_params.py:501`) — explicit rows only; schema default 1.0 ignored.  Only consulted when the commodity uses the price-ladder feature (gated by `commodity.price_method = price_ladder_annual` or `price_ladder_cumulative`).  Inside the gate, `_commodity_unitsize_param` in `_commodity_ladder.py` substitutes `1.0` per commodity when the explicit Param is absent — so absent → identity, explicit-and-non-1.0 → real LP scaling on `v_trade` (balance LHS, per-tier cap LHS, objective term).  Description rewritten to name the feature, the price_method gate, and the absent → identity semantics. | **high (resolved)** |

## Surface / out-of-scope

| Class | Parameter | Current default | Note |
|---|---|---|---|
| (all `*_method` enums) | — | enum literal (e.g. `regular`, `bind_forward_only`, `not_allowed`) | Method-default review is batch D; do not auto-fix per task instructions. |
| `model.output_*`, `model.exclude_entity_outputs`, `model.output_horizon`, `group.flow_aggregator`, etc. | — | `"yes"` / `"no"` | Yes/no toggles; not a "feature disabled" sentinel pattern.  Batch B covers the broader output-options cleanup. |
| `group.penalty_capacity_margin`, `group.penalty_inertia`, `group.penalty_non_synchronous` | — | `5000.0` | Engine reads via `_param_shapes.resolve_param_shape`, which DOES honour the schema default (broadcast path at `_param_shapes.py:632-640`).  Default is a real LP coefficient gated by the corresponding `has_*` method.  Not a wrong default. |
| `node.penalty_up`, `node.penalty_down` | — | `10000.0` | `_derived_arithmetic.py:155-223` uses `_try_param` which honours broadcast.  Real LP coefficient.  Not a wrong default. |
| `connection.availability`, `node.availability`, `unit.availability`, `connection.efficiency`, `unit.efficiency` | — | `1.0` | Real "identity" defaults, broadcast as LP coefficients. |
| `connection.discount_rate`, `node.discount_rate`, `unit.discount_rate` | — | `0.05` | Real default with engine fallback at `_derived_npv.py:630` (`r ≤ 0 → 0.05`).  Self-consistent. |
| `constraint.constant` | — | `0.0` | Broadcast as RHS for user constraints.  Self-consistent. |
| `reserve__upDown__connection__node.reliability`, `reserve__upDown__unit__node.reliability` | — | `1.0` | Broadcast default; engine also applies `if v == 0.0: v = 1.0` collapse in `_emit_reserve.py:540-542`, making default 1.0 self-consistent. |
| `model.max_flow_for_unconstrained_variables` | — | `1000000.0` | Engine fallback in `_emit_arc_unions.py:2469` uses the same value.  Self-consistent. |
| `model.version` | — | `55.0` | Schema version sentinel; do not touch. |
| `solve.timeline_hole_multiplier` | — | `1.0` | Real default consumed by `_solve_config.py:422`. |

## Patching plan

In `db_migration.py`, add `_migrate_v56_fix_wrong_defaults(db)`, wire it
into the `elif next_version == 56:` block after the existing two v56
helpers, and have it call `db.update_item("parameter_definition", ...,
default_value=None, default_type=None)` for each of the five
**high**-confidence rows.

In `spinedb_schema.json`, change column 3 from the current value to
`null` for each of the five rows so a fresh v55 init lands on the same
contract.

No version bump (still preparing v56).  No method-enum changes.  No
canonical/fixture regeneration.
