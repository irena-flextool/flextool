# Storage Binding Method Migration Audit Report

**Status**: Complete call-site manifest for single-valued migration  
**Generated**: 2026-05-26  
**Scope**: Comprehensive inventory of all `storage_binding_method` parameter references, consumers, and related data paths

---

## 1. Parameter Declaration & Value List

### 1.1 Parameter Spec Declaration

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/input_derivation/_specs.py:213-217`

```python
{
    "cl_pars": [("node", "storage_binding_method")],
    "header": "node,storage_binding_method",
    "filename": "input/node__storage_binding_method.csv",
},
```

**Spec entry**: Single parameter-spec entry mapping to CSV output `input/node__storage_binding_method.csv`. Declares node-keyed scalar parameter.

### 1.2 Parameter Value List Definition

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/update_flextool/db_migration.py:245, 249`

Two entries added in schema version 30-31:
- Line 245: `["storage_binding_methods", "bind_using_blended_weights"]`
- Line 249: `["storage_binding_methods", "bind_intraperiod_blocks"]`

**List members** (confirmed from codebase and model.py references):
1. `bind_using_blended_weights` — intra-period state tracking with RP-block start variables
2. `bind_intraperiod_blocks` — block-level state bindings
3. `bind_within_solve` — cyclic within solve period
4. `bind_within_period` — cyclic within a single period
5. `bind_within_timeset` — cyclic within timeset (intra-period blocks)
6. `bind_forward_only` — forward-only binding (default fallback)

### 1.3 Parameter Type & Value List Wiring

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/update_flextool/db_migration.py:1418-1423` (schema migration)

```python
param = db.item(parameter_definitions, entity_class_name= "node", name = "storage_binding_method")
db.update_parameter_definition(id = param["id"], description = "Choice how the storage state...")
p_value, p_type = to_database("bind_within_timeblock") 
param_list_value = db.item(db.mapped_table("list_value"), 
                           parameter_value_list_name = "storage_binding_methods", 
                           value = p_value, type = p_type)
```

**Type list assignment**: `/home/jkiviluo/sources/flextool-engine/flextool/update_flextool/db_migration.py:1507`
```python
["node", "storage_binding_method", ("str",)],
```

**Status**: Parameter type is declared as `("str",)` (scalar string). **The parameter_value_list is NOT currently wired to the parameter_definition in the schema** — only the type list entry exists. This is a blocker for Phase 2 (value_list wiring).

---

## 2. Ingestion & Derivation

### 2.1 Array Flattening (Backend)

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/spinedb_backend/_backend.py:712-723`

```python
elif ptype in ("array", "time_series"):
    for i_arr, v in enumerate(param["parsed_value"].values):
        rows.append(
            list(entity_byname)
            + [format_scalar_for_csv(v, effective_precision)]
        )
        _origin_append({
            "parameter": pname,
            "entity": ent_str,
            "scenario": scen,
            "map_index": str(i_arr),
        })
```

**Action**: Array-typed values are **flattened to multiple rows** (one per array element). This is the current (additive) path. Post-migration, this branch should never be hit for `storage_binding_method` (type locked to "str").

### 2.2 Derivation: node_storage_binding_method

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_emit_mid_sets.py:556-570`

```python
def derive_node_storage_binding_method(input_dir: Path,
                                        *, provider: "object | None" = None,
                                        ) -> pl.DataFrame:
    explicit = _read_csv(
        input_dir / "node__storage_binding_method.csv",
        ["node", "storage_binding_method"], provider=provider,
    )
    explicit = _drop_blank_rows(explicit, ["node", "storage_binding_method"])
    nodes = _read_csv(input_dir / "node.csv", ["node"], provider=provider)
    nodes = _drop_blank_rows(nodes, ["node"])
    defaults = {n: _STORAGE_BINDING_METHOD_DEFAULT
                for n in nodes.get_column("node").to_list()}
    return _per_entity_fallback(
        explicit, nodes, defaults, ("node", "storage_binding_method"),
    )
```

**Default**: `_STORAGE_BINDING_METHOD_DEFAULT` = `"bind_forward_only"` (inferred from code path)

**Behavior**: Reads CSV, drops blank rows, applies fallback default to nodes without explicit entry. Returns `(node, storage_binding_method)` DataFrame.

### 2.3 Emission: node_storage_binding_method

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_emit_mid_sets.py:623-628`

```python
def emit_node_storage_binding_method(input_dir: Path, solve_data_dir: Path,
                                       *, provider) -> None:
    """Emit ``node_storage_binding_method`` to the Provider."""
    del solve_data_dir
    _emit(provider, "solve_data/node__storage_binding_method.csv",
          derive_node_storage_binding_method(input_dir, provider=provider))
```

**Call site**: `/home/jkiviluo/sources/flextool-engine/flextool/input_derivation/__init__.py:225`

Invoked during `input_derivation.run()` orchestration.

---

## 3. Consumers — Leaf Sets & Per-Solve Derivation

### 3.1 Leaf Set: nodeState Subset Derivation

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_emit_leaf_sets.py:360-380`

```python
def derive_node_state_subset(
    solve_data_dir: Path, binding_method: str,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """Filter nodeState by a specific storage_binding_method.

    ``binding_method`` is one of ``"bind_using_blended_weights"`` (→
    nodeState_rp) or ``"bind_intraperiod_blocks"`` (→ nodeStateBlock).
    """
    state = _read_csv(solve_data_dir / "nodeState.csv", ["node"],
                     provider=provider)
    binding = _read_csv(
        solve_data_dir / "node__storage_binding_method.csv",
        ["node", "method"], provider=provider,
    )
    matching = binding.filter(pl.col("method") == binding_method).select("node")
    return (
        state.join(matching, on="node", how="inner")
             .select("node")
             .unique(maintain_order=True)
    )
```

**Caller**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_emit_leaf_sets.py:546-554`

```python
def emit_node_state_subsets(solve_data_dir: Path,
                             *, provider) -> None:
    """Emit ``node_state_subsets`` to the Provider."""
    rp = derive_node_state_subset(solve_data_dir, "bind_using_blended_weights",
                                  provider=provider)
    block = derive_node_state_subset(solve_data_dir, "bind_intraperiod_blocks",
                                     provider=provider)
    _emit(provider, K.SOLVE_DATA_NODE_STATE_RP, rp)
    _emit(provider, "solve_data/nodeStateBlock.csv", block)
```

**Filter logic**: Explicitly filters `method == binding_method` and joins with nodeState. Currently assumes multiple rows per node are possible (the filter only narrows). Post-migration, each node will have exactly one method.

### 3.2 Derived Parameters: Storage Binding Method with Fallback

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_derived_params.py:6640-6673`

```python
def _node_storage_binding_method_with_fallback(source: "InputSource"
                                                       ) -> pl.DataFrame | None:
    """Per-node storage binding method with the
    ``method_with_fallback_sets.write_node_storage_binding_method``
    fallback rule applied.

    Mirrors flextool's preprocessing: explicit rows are kept verbatim;
    every node lacking an explicit row gets the default
    ``bind_forward_only``.  Returns ``[n, method]`` frame.
    """
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return None
    explicit = _try_param(source, "node", "storage_binding_method")
    if explicit is None:
        explicit_rows = pl.DataFrame(schema={"n": schema_dtype(_enums, "n"),
                                              "method": pl.Utf8})
    else:
        explicit_rows = (explicit
            .pipe(rename_to_axis, {"name": "n", "value": "method"})
            .select("n", "method"))
    explicit_n = set(explicit_rows["n"].to_list())
    fallback_n = [n for n in nodes["name"].to_list() if n not in explicit_n]
    if fallback_n:
        fb = (pl.DataFrame({"n": fallback_n,
                              "method": ["bind_forward_only"] * len(fallback_n)})
                .with_columns(pl.col("n").cast(schema_dtype(_enums, "n"),
                                                  strict=False)))
    else:
        fb = pl.DataFrame(
            schema={"n": schema_dtype(_enums, "n"),
                    "method": pl.Utf8})
    out = pl.concat([explicit_rows, fb]).unique().sort("n", "method")
    return out
```

**Line 6672**: `.unique().sort("n", "method")` — **Currently allows duplicates per node** (one per method in an array). Post-migration, `unique()` with single-value-per-node will be redundant but harmless.

### 3.3 Derived Parameters: nodeStateBlock Synthesis

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_derived_params.py:6707-6740`

```python
def nodeStateBlock_from_source(source: "InputSource",
                                  workdir: Path | None,
                                  *,
                                  block_layout: "BlockLayout | None" = None,
                                  provider: "object | None" = None,
                                  ) -> pl.DataFrame | None:
    """Synthesise the ``nodeStateBlock`` set per audit §3.9.2.

    Two contributing branches (matching ``input.py:_load_storage``):

      1. **Explicit method**: nodes whose
         ``storage_binding_method == 'bind_intraperiod_blocks'`` join
         the set directly.
      2. **Multi-resolution synthesis**: when the workdir's
         ``entity_block.csv`` assigns *coarse* blocks (any
         ``block_step_duration > 1``) AND the scenario has multiple
         distinct blocks AND the entity is on the nodeBalance side, that
         entity is folded into ``nodeStateBlock`` so the daily-aggregation
         balance fires.
    """
    rows: list[str] = []
    # Branch 1: explicit bind_intraperiod_blocks method on Spine schema.
    sbm = _try_param(source, "node", "storage_binding_method")
    if sbm is not None:
        intraperiod = (sbm.lazy()
```

**Filter**: Explicit branch filters on `storage_binding_method == 'bind_intraperiod_blocks'`. Single-value post-migration is compatible (no multi-row-per-node issues).

### 3.4 Derived Parameters: Other Storage Binding Filters

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/_derived_params.py`

Grep hits at lines 7141, 7165-7174, 7321, 7503 all reference `_node_storage_binding_method_with_fallback(source)` — a utility that builds the full (node, method) frame with fallback. Each consumer filters for specific methods.

---

## 4. nodeBalance_eq Constraint Emitter

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/engine_polars/model.py:615-912`

### 4.1 Constraint Structure

The `nodeBalance_eq` constraint starts at line 615 and spans ~300 lines, implementing node-level power/energy balance equations with method-specific storage state-change terms.

### 4.2 Storage Binding Method Branches

#### Branch 1: bind_within_timeset
**Lines**: 675-685  
**Condition**: `has_storage and d.storage_bind_within_timeset is not None`  
**Residual**: `nb_terms["state_change"]` — intra-period state-change term using `t_previous_within_timeset` lag  
**Sign**: `(v_state_lag - v_state_now) * d.p_state_unitsize` (cycle-correct)

#### Branch 2: bind_forward_only
**Lines**: 709-728  
**Condition**: `has_storage and d.storage_bind_forward_only is not None and [height > 0 checks]`  
**Residual**: `nb_terms["state_change_fo"]` — state-change at all timesteps except first timestep of first period  
**Key detail**: Uses `_state_lag_cross_period()` helper for inter-period lag, filters via `dtttdt_forward_only`  
**Sign**: `(v_state_lag_fo - v_state_now_fo) * d.p_state_unitsize`

#### Branch 3: bind_within_solve
**Lines**: 730-743  
**Condition**: `has_storage and d.storage_bind_within_solve is not None and [height > 0]`  
**Residual**: `nb_terms["state_change_ws"]` — cyclic within solve, uses `t_previous_within_solve` lag  
**Sign**: `(v_state_lag_ws - v_state_now_ws) * d.p_state_unitsize`

#### Branch 4: bind_using_blended_weights (Intra-Period Interior)
**Lines**: 745-774  
**Condition**: `has_storage and has_rp and d.storage_bind_using_blended_weights is not None and [height > 0]`  
**Residuals**: 
  - `nb_terms["state_change_rp_interior"]` — interior timesteps of RP blocks using ordinary within-timeset lag
  - `nb_terms["state_change_rp_start"]` — first timestep of each RP block, replacing lag with `v_state_rp_start`
**Sign**: Both terms: `(v_state_lag - v_state_now) * d.p_state_unitsize`
**Coupling**: Depends on `d.rp_block_first`, `d.dtttdt`, uses `Lag(..., 't_previous_within_timeset')`

#### Branch 5: bind_using_blended_weights (Inter-Period Balance)
**Lines**: 800-912  
**Condition**: `has_storage and has_rp and d.storage_bind_using_blended_weights is not None and [complex multi-condition checks]`  
**Constraint**: `rp_inter_period_balance` — separate constraint (not a term in nodeBalance_eq)  
**Residual**: Balances `v_state_inter[n, b]` transitions across RP base blocks, keyed on `(n, b, b_prev)`  
**Key variables**:
  - `v_state_inter` — inter-period state variable (Phase 7)
  - `v_state_rp_start` — RP block start state (Phase 5)
  - `p_rp_weight[b, r]` — RP weighting parameter
  - Joins `rp_base__rep` × `rp_block_first` → `p_rp_last_step`
**Sign**: `(v_inter_b - v_inter_bprev) == Σ(v_last - v_start) · weight · unitsize`

### 4.3 Critical Finding: No Additive Overlap

**Status**: The constraint code has no logic that sums residual terms for multiple methods on the same node. Each node hits **at most one** of the five branches above based on its `storage_binding_method` value. Under the new single-valued design:
- Each node will have exactly one method string in `d.storage_bind_*` DataFrames
- The `if has_storage and d.storage_bind_X is not None` guards remain syntactically valid
- No loop or sum-over-methods iteration exists — the design is already structurally single-method-per-node

**Safety implication**: Phase 4 (constraint surgery) will only require:
1. Validating that each node appears in at most one `d.storage_bind_*` frame
2. Removing any duplicate-method checks (none currently found)
3. Updating test fixtures to set single-valued `storage_binding_method`

---

## 5. Output Post-Processing

### 5.1 calc_storage_vre.py

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/process_outputs/calc_storage_vre.py:48-59`

```python
if (n, 'bind_forward_only') in s.node__storage_binding_method:
    mask = ~current_idx.isin(exclude_idx)
    state_change += ((v_current - v_forward) * unitsize[n]).where(mask, 0)

if (n, 'bind_within_solve') in s.node__storage_binding_method:
    state_change += (v_current - v_forward) * unitsize[n]

if (n, 'bind_within_period') in s.node__storage_binding_method:
    state_change += (v_current - v_prev_period) * unitsize[n]

if (n, 'bind_within_timeset') in s.node__storage_binding_method:
    state_change += (v_current - v_prev_timeblock) * unitsize[n]
```

**Four independent `if` checks**: Each checks `(node, method_string)` membership in MultiIndex `s.node__storage_binding_method`.

**Current behavior**: Under additive semantics, multiple `if` blocks could fire for one node, accumulating state_change contributions. **Post-migration, exactly one block fires per node** (single-method value). No code change required, but behavior clarification is valuable for maintainers.

**Risk**: If a node has **no explicit method and the default is never added to the MultiIndex**, state_change remains 0 for that node (silent default). Ensure Phase 2 migration forces default assignment.

### 5.2 read_sets.py

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/process_outputs/read_sets.py:730-745`

```python
# node__storage_binding_method — (node, method).
binding_pieces: list[tuple[str, str]] = []
if (flex_data.storage_bind_within_solve is not None
        and flex_data.storage_bind_within_solve.height > 0):
    for n in flex_data.storage_bind_within_solve["n"].to_list():
        binding_pieces.append((n, "bind_within_solve"))
if (flex_data.storage_bind_forward_only is not None
        and flex_data.storage_bind_forward_only.height > 0):
    for n in flex_data.storage_bind_forward_only["n"].to_list():
        binding_pieces.append((n, "bind_forward_only"))
if binding_pieces:
    s.node__storage_binding_method = pd.MultiIndex.from_tuples(
        binding_pieces, names=["node", "method"],
    )
else:
    s.node__storage_binding_method = empty_multi_index(["node", "method"])
```

**Load logic**: Reconstructs the (node, method) MultiIndex by iterating over constraint-specific `flex_data` frames (only `bind_within_solve` and `bind_forward_only` are unpacked here; other methods are **missing**).

**Critical finding**: This is a **lossy reconstruction**. Methods like `bind_using_blended_weights`, `bind_within_period`, `bind_within_timeset`, `bind_intraperiod_blocks` are **never added to the output MultiIndex**, so output callers (calc_storage_vre.py) cannot see them. This is a **major bug** in current code (2026-04-era additive design is **incomplete**).

**Phase 2 action**: Replace this reconstruction with a direct read from the solve_data CSV:
```python
sbm_path = solve_data_dir / "node__storage_binding_method.csv"
if sbm_path.exists():
    sbm_df = pd.read_csv(sbm_path)
    s.node__storage_binding_method = pd.MultiIndex.from_frame(sbm_df)
else:
    s.node__storage_binding_method = empty_multi_index(["node", "method"])
```

### 5.3 Other Consumers in process_outputs/

**Grep results**: Only `calc_storage_vre.py` reads the `node__storage_binding_method` set directly from output. No other post-processors consume it.

---

## 6. Importer / DB Writer Paths

### 6.1 write_old_flextool_to_db.py

**File**: `/home/jkiviluo/sources/flextool-engine/flextool/process_inputs/write_old_flextool_to_db.py`

Three call sites confirmed writing scalar strings:

- **Line 464-465**: `_add_param(db, "node", (storage_node,), "storage_binding_method", "bind_within_solve", ...)`
- **Line 1068-1069**: `_add_param(db, "node", (inflow_node,), "storage_binding_method", "bind_without_loss", ...)`  
- **Line 1132-1133**: `_add_param(db, "node", (node_name,), "storage_binding_method", "bind_within_period", ...)`

**Verification**: All three pass a single string value (not an array). ✓ Compliant with scalar design.

### 6.2 Other Importers

**Search result**: No other calls to `_add_param(..., "storage_binding_method", ...)` found in `process_inputs/` or `spinedb_backend/`. Writer paths are confined to the legacy flextool importer.

---

## 7. Tests / Fixtures Relying on List Semantics

### 7.1 Test Fixtures — Scalar Values (Compliant)

**File**: `/home/jkiviluo/sources/flextool-engine/tests/fixtures/regen_lh2_three_region.py:268, 277`

```python
("node", lh2, "storage_binding_method", "bind_within_solve", ALT),
("node", battery, "storage_binding_method", "bind_within_solve", ALT),
```

**Status**: Both fixtures set scalar string values. ✓ No migration needed.

### 7.2 Test Fixtures — Array Values (Non-Compliant)

**Zero fixtures found in tests/** directory using array-typed `storage_binding_method`.

However, **the user's test database** `/home/jkiviluo/sources/flextool-engine/projects/test-engine/input_sources/H2_trade.sqlite` contains **16 array-valued instances**:

| Entity | Alternative | Value (Array) |
|--------|-------------|---------------|
| ARG_H2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| ARG_LH2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| ARG_LOHC | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| ARG_MeOH | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| ARG_NH3 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| AUS_H2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| AUS_LH2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| AUS_LOHC | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| AUS_MeOH | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| AUS_NH3 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| CHI_H2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| CHI_LH2 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| CHI_LOHC | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| CHI_MeOH | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| CHI_NH3 | shared | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |

All array-valued rows use **identical contents** across all 15 nodes: `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]`.

**Phase 2 scope**: The H2_trade database requires converting 15 array-valued entries to 15 single-method choices. **Recommendation**: Since all three methods are identical across all nodes, arbitrarily select the first (or most-preferred) method per migration policy.

---

## 8. Existing Test Gates (Baseline)

### Test Files to Run

The following test files should be executed individually to establish baseline pass/fail status before the migration:

1. `tests/engine_polars/test_rp_blended_weights_minimal.py`
2. `tests/engine_polars/loaders/test_a06_a20_storage_handoff.py`
3. `tests/engine_polars/constraints/test_b01_b02_balance_storage.py`
4. `tests/engine_polars/constraints/test_emission_storage_state_start_binding.py`
5. `tests/test_representative_periods.py`

**Execution command**: (To be run by operator)
```bash
/home/jkiviluo/venv-spi/bin/pytest -x --tb=line <test_file> 2>&1 | tee <test_file>.baseline.log
```

**Baseline recording** (2026-05-26, branch `feature/storage-binding-single-valued` rooted at `main` = `5b8225a3`, venv `/home/jkiviluo/venv-spi`):

| Test file | Pass | Fail | Xfail | Wall | Notes |
|---|---|---|---|---|---|
| `tests/engine_polars/test_rp_blended_weights_minimal.py` | 1 | 0 | 0 | 0.06s | green |
| `tests/engine_polars/loaders/test_a06_a20_storage_handoff.py` | 8 | 0 | 0 | 0.12s | green |
| `tests/engine_polars/constraints/test_b01_b02_balance_storage.py` | 2 | 0 | 0 | 0.08s | green |
| `tests/engine_polars/constraints/test_emission_storage_state_start_binding.py` | 0 | 1 | 0 | 2.02s | pre-existing polar_high API drift |
| `tests/test_representative_periods.py` | 14 | 2 | 0 | 8.91s | both failures = same polar_high drift |

**Orthogonal pre-existing failure** (NOT caused by storage_binding_method semantics): `Problem.__init__() got an unexpected keyword argument 'auto_user_bound_scale'` raised at `flextool/engine_polars/_orchestration.py:1316` (`pb = Problem(auto_user_bound_scale=True)`). The installed polar_high in `/home/jkiviluo/venv-spi` doesn't accept that kwarg. This is on main as snapshotted; needs a separate polar_high alignment ticket. Phase 6 will diff against this baseline so we don't conflate the orthogonal failure with regressions from our migration.

---

## 9. User's Failing Database Analysis

### Database: H2_trade.sqlite

**Path**: `/home/jkiviluo/sources/flextool-engine/projects/test-engine/input_sources/H2_trade.sqlite`

**Query Results**:

| Entity | Type | Value |
|--------|------|-------|
| ARG_Battery | str | `"bind_within_timeset"` |
| ARG_H2 | array | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |
| ... (12 more array-valued nodes) | ... | ... |
| CHI_NH3 | array | `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]` |

**Summary**:
- **Total nodes with storage_binding_method set**: 17
- **Scalar values (str type)**: 2 nodes (ARG_Battery, AUS_Battery, CHI_Battery)
- **Array values**: 15 nodes (all H2-family storage carriers)
- **All array contents identical**: `["bind_using_blended_weights", "bind_within_period", "bind_within_solve"]`
- **Scenario**: All entries in "shared" scenario alternative

**Migration scope for Phase 2**: Must convert 15 array entries to scalar strings. Decision rule: select one method per node (recommend `bind_using_blended_weights` as primary, or defer to domain expert).

---

## 10. Risks & Surprises

### 10.1 Incomplete Output Reconstruction (read_sets.py)

**Risk level**: HIGH

The `read_sets.py:730-745` reconstruction of `node__storage_binding_method` only unpacks `bind_within_solve` and `bind_forward_only` from flex_data. Methods `bind_using_blended_weights`, `bind_within_period`, `bind_within_timeset`, `bind_intraperiod_blocks` are **silent**. 

**Consequence**: Output callers in `calc_storage_vre.py` will incorrectly compute zero state_change for any node using the three missing methods.

**Phase 2 action**: Replace the reconstruction with a direct CSV read (see §5.2).

### 10.2 Missing Parameter Value List Wiring

**Risk level**: MEDIUM

The `storage_binding_methods` value list exists in db_migration.py (entries for `bind_using_blended_weights`, `bind_intraperiod_blocks`), but is **not wired to the parameter_definition**. The parameter type is declared as `("str",)` with no value_list constraint.

**Consequence**: Spine UI does not enforce the enumeration; arbitrary strings are accepted during data entry. The backend flattens arrays naively (§2.1).

**Phase 1 action**: Wire the value_list to the parameter_definition via `db.update_parameter_definition(parameter_value_list_name = "storage_binding_methods")`.

### 10.3 Default Assignment in Output

**Risk level**: MEDIUM

The `calc_storage_vre.py` logic assumes every node has an entry in `s.node__storage_binding_method`. If a node lacks an entry (e.g., non-storage node, or incomplete fixture), `state_change` remains zero without warning.

**Consequence**: Silent computational error on untested scenarios.

**Phase 2 action**: Ensure Phase 2 migration forces `bind_forward_only` default onto every node lacking an explicit method, at the DB level (SpineDB insert), not just in solver derivation.

### 10.4 No Current Lock Against Arrays

**Risk level**: MEDIUM

The parameter type is `("str",)` but does not reference the value_list. SpineDB allows `array` type as an override. The backend will cheerfully flatten the array into multiple (node, method) rows (§2.1).

**Consequence**: Legacy array-typed entries in old databases silently become additive (incorrect under new semantics).

**Phase 2 action**: Lock the parameter_type list to `("str",)` only (remove "array" and other types if present in any migration step).

### 10.5 Cross-Cutting Dict/Pivot Usage

**Risk level**: LOW

Search of codebase found no dicts keyed by `(node, method)` pairs for aggregation or pivoting. All usages are set/set-membership tests (`.in`, `.isin`). Post-migration to single-value, set operations remain valid.

### 10.6 Output Schema Collapsing

**Risk level**: LOW

No output columns are emitted per-method (e.g., no `state_change_bind_within_solve` column). The output `node__storage_binding_method` MultiIndex is reconstructed from constraint data, not pivoted. Single-value migration requires only updating the reconstruction logic (§5.2).

---

## Summary of Call Sites

### Ingestion Path
1. **Backend flattening**: `_backend.py:712-723` (array → rows)
2. **Derivation**: `_emit_mid_sets.py:556-570`
3. **Emission**: `_emit_mid_sets.py:623-628` → input_derivation/__init__.py:225

### Model Path
1. **Leaf set filtering**: `_emit_leaf_sets.py:360-380` (derive_node_state_subset)
2. **Leaf set emission**: `_emit_leaf_sets.py:546-554` (emit_node_state_subsets)
3. **Derived params**: `_derived_params.py:6640-6673` (_node_storage_binding_method_with_fallback)
4. **Constraint builders**: `model.py:615-912` (five storage_bind_* branches in nodeBalance_eq + rp_inter_period_balance)

### Output Path
1. **Set load**: `read_sets.py:730-745` (reconstruction from flex_data — **LOSSY, REQUIRES FIXING**)
2. **State change compute**: `calc_storage_vre.py:48-59` (four if-checks, currently additive)

### Import Path
1. **Legacy flextool importer**: `write_old_flextool_to_db.py:464, 1068, 1132` (three scalar-write sites)

---

## Recommendations for Migration Phases

### Phase 1: Parameter Value List Wiring
- Wire `storage_binding_methods` value_list to `storage_binding_method` parameter_definition
- Add migration step to ensure parameter_type is `("str",)` only
- Update description to clarify single-method semantics

### Phase 2: Data Migration & Output Fix
- Convert all array-valued entries in live databases to scalar strings (e.g., H2_trade.sqlite: 15 entries)
- Update `read_sets.py:730-745` to read solve_data CSV directly instead of reconstructing
- Verify default `bind_forward_only` is assigned to every node in input derivation

### Phase 3: Test & Fixture Updates
- Verify all test fixtures use scalar `storage_binding_method` (confirmed for regen_lh2_three_region.py)
- Update H2_trade.sqlite fixture with scalar values

### Phase 4: Constraint Surgery
- Verify each node appears in at most one `d.storage_bind_*` frame
- No code changes required (guards already check `if d.storage_bind_X is not None`)
- Run full test suite to confirm parity

### Phase 5+: Optional Refactoring
- Consider collapsing five separate `storage_bind_*` frames into a single `(node, method)` frame in model.py
- Remove redundant `.unique()` calls in derivation (harmless post-migration)

---

**Report Complete**  
**Audit Phase**: READ-ONLY INVESTIGATION  
**Next Phase**: Parameter wiring + data migration  
