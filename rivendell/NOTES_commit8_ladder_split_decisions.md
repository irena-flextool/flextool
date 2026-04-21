# Commit 8 — Split `price_ladder` into per-method params (v40 fix)

## Context

Commit #1 (`a492c40`) introduced a single `commodity.price_ladder` param
(nested Map `tier → {price, quantity}`, no period dimension) that served
both `price_ladder_cumulative` and `price_ladder_annual`.  Users could
not express per-period annual caps (e.g. "2020 coal cap = 100, 2025 cap
= 50"), and the DB UI had no way to distinguish the two methods' input
shapes.

Schema v40 has not shipped, so this commit modifies v40 in-place rather
than stacking a v41 migration.

## Design

Two new params on `commodity`:

- `price_ladder_cumulative` — always 1d_map `Map(tier → {price, quantity})`.
  Only consulted when `price_method = 'price_ladder_cumulative'`.
- `price_ladder_annual` — accepts either 1d_map `Map(tier → {price,
  quantity})` (same limit every period) or 2d_map `Map(tier → Map(period
  → {price, quantity}))` (per-period).  Writer auto-detects and expands
  or keeps rows accordingly.

Preflight error if `price_method` is a ladder variant but the
corresponding param is unset — raised by `_validate_ladder_methods` in
`input_writer.py` before CSV emission, naming the commodity and
expected parameter.

## Mod side

- Sets split: `commodity__tier_cum` (direct from cumulative CSV),
  `commodity__tier__period_ann` (raw triples from annual CSV) with
  `commodity__tier_ann := setof {(c, i, d) in
  commodity__tier__period_ann} (c, i)`.  The annual CSV has one row per
  period so the membership set must be projected via `setof` — reading
  `commodity__tier_ann <- [commodity, tier]` directly triggers a
  duplicate-tuple error under GMPL.
- `p_ladder_cum_{price,quantity}`: indexed by `(c, i) in
  commodity__tier_cum`.
- `p_ladder_ann_{price,quantity}`: indexed by `(c, i) in
  commodity__tier_ann, d in periodAll`.
- `cndi_ladder_cum` / `cndi_ladder_ann` / `cndi_ladder` (union) for
  v_trade and ladder constraints.
- `ladder_tier_cap_infinite` split into `_cum` and `_ann` variants.
- Objective term split into two sums, one per method.

## Writer side

- Period discovery for 1d→2d expansion: reads
  `input/periods_available.csv` first (written by the model-parameter
  spec for `model.periods_available`); falls back to
  `model.periods_available` in the DB, then to `solve.period_timeset`
  map indexes.  Typical test fixtures use `period_timeset` only, so the
  solve-param fallback is essential for test coverage.
- 1d detection: row length after `convert_map_to_table` is 3
  (tier, facet, value).  2d: length ≥ 4 (tier, period, facet, value).

## Tests

- `test_commodity_ladder_smoke.py`: fixtures migrated to
  `price_ladder_annual`; two new tests (per-period 2d ladder, preflight
  error).
- `test_commodity_ladder_rolling.py`: fixtures migrated to the method-
  specific param name.
- `test_cumulative_handoffs.py`: CSV-writer helper splits rows across
  the two new filenames based on `price_method`.
- 50 tests pass in the four relevant suites; `test_scenarios.py -k
  coal` passes 25/25.

## Decisions of note

1. **Annual CSV reader uses raw triples + `setof` projection** — GMPL
   silently tolerates duplicate parameter rows but explicitly rejects
   duplicate tuples in set membership reads.  The alternative (use the
   same CSV twice, once for the 2-col set and once for the 3-key
   params) failed with "duplicate tuple (coal,1) detected".  The
   projection approach is idiomatic to how the mod already handles
   other "(key1, key2) derived from three-key CSV" patterns.

2. **Period discovery falls back to `solve.period_timeset`** — not all
   models populate `model.periods_available`.  Every rolling/test
   fixture in the repo uses `solve.period_timeset` alone, so the
   writer must walk that too for 1d expansion to work.

3. **Template regeneration required pre-downgrading the master** —
   `sync_master_json_template` re-runs migrations starting from the
   current master's version; since the master was already at v40,
   modifying v40 in-place needed the master's version field reset to
   39 and the old `price_ladder` entries stripped before sync could
   re-run the (modified) v40 block.
