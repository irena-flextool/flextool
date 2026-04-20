# Noted decisions — commit #2 (commodity ladder LP mechanism)

Non-obvious choices made while implementing the v_trade variable and
tier caps in `flextool/flextool.mod`.  One line per decision,
prefixed with the file:line it relates to.

## Mod (flextool/flextool.mod)

- `flextool.mod:287-291` — made `tier` a derived set (`setof (c, i) in commodity__tier (i)`) instead of the v1 domain set `set tier dimen 1;`.  Reason: nothing writes `input/tier.csv`; the commodity_ladder.csv read populates the tier indices via `commodity__tier <- [commodity, tier]` and we need `tier` to reflect that.  Dropped the `within {commodity, tier}` on `commodity__tier` for the same reason.

- `flextool.mod:370` — `p_commodity_ladder_quantity` default changed from `+Infinity` to `1e30` (sentinel).  Reason: GMPL's CSV reader rejects the literal `inf` / `Infinity` strings, so the Python writer emits `1e30` for user-provided `+Infinity` and the default must match the sentinel.  Constraint predicates use `>= 1e29` / `< 1e29` instead of `== +Infinity` / `< +Infinity`.

- `flextool.mod:1722-1732` — avoid reusing a bound name in an inner tuple.  `setof {(c, n) in commodity_node, (c, i) in commodity__tier}` (rebinding `c`) produced a "no value for tier" error in dependent expressions.  Switched to `{(c, n) in commodity_node, ..., i in tier : (c, i) in commodity__tier}`.  Same pattern in `ladder_tier_cap_annual`.

- `flextool.mod:~2206` — ladder objective term omits `p_years_represented_d[d]`.  Reason: the existing `pdtCommodity` price term at line 2174 does NOT carry `years_represented` directly; it relies on `p_inflation_factor_operations_yearly[d]` (which already sums `years_represented * discount`) to scale operational costs across the represented window.  Multiplying v_trade by an extra `years_represented` would double-count for the "single-tier `+Infinity` ladder at the same price as legacy" equivalence test.  Verified: `v_flow`'s matrix coefficient in the MPS is identical in coal baseline (via pdtCommodity) and coal_ladder (via balance eq → v_trade → objective) — e.g. both produce `4.5625` on `v_flow[coal_plant,coal_market,west,p2020,t0001]` in the example smoke test, yielding bit-identical total objective `1.144037e9`.

- `flextool.mod:~3548` balance eq — mirrors the full pdtCommodity buy/sell aggregation structure (noEff + eff with slope + min_load_efficiency online section).  Reason: the brief's simpler form using only `process_source_sink` would NOT match pdtCommodity exactly for efficient processes (coal plants, etc), breaking the "bit-identical when price_method='price' on single-tier ∞ ladder" invariant that the objective relies on.

- `flextool.mod:~3588` annual tier cap — RHS uses `tier_quantity × complete_period_share_of_year[d]`, NO `years_represented` factor.  Reason: v_trade is in realized-timeline MWh; annualizing by `÷ share` yields per-year MWh, the per-year cap applies directly.  The `years_represented` factor is carried by `p_inflation_factor_operations_yearly[d]` on the COST side, not on the physical decision.

- `flextool.mod:~3604` cumulative tier cap — LHS multiplies by `years_represented / share` to convert realized-timeline MWh into full-horizon MWh.  RHS is the raw `ladder_quantity` (no scaling).  Intended for a single solve covering the whole horizon; rolling solves reset this each roll (step 4 of the project will add the running-balance handoff).

- `flextool.mod:~3618` infinite-tier cap — uses `p_unconstrained_flow_cap × 8760 × complete_period_share_of_year[d]`, i.e. max_flow × realized-timeline-hours.  TODO decision #1: tighter bound via sum of connected-process `flow_max × entity_unitsize` requires a commodity__node→process connectivity set that doesn't exist.  Postponed to a follow-up commit; documented in the mod with a `TODO decision #1` comment.

- Preflight check (brief item §7) — SKIPPED.  Building the "is any connected process's `p_flow_max` also infinite" predicate needs the same connectivity set as decision #1.  Deferred together.

## Input writer (flextool/flextoolrunner/input_writer.py)

- `input_writer.py:~1440` — convert Python `inf` → `1e30` before CSV emission.  Reason: GMPL can't parse `inf`/`Infinity` from CSV.  Matches the mod's `1e30` default.

## Smoke tests (tests/test_commodity_ladder_smoke.py)

- Scenario is built by migrating `tests/fixtures/tests.json` (v38) to v40 and then adding a `coal_ladder` alternative that sets `price_method='price_ladder_annual'` with a two-tier ladder.  Integer tier indices ("1", "2") because `_write_commodity_ladder` skips non-integer tiers.

- Non-binding test (`tier1_quantity=1e12`, `tier1_price=20`) gives bit-identical objective to the legacy `coal` scenario within `rel=1e-6`.  Binding test (`tier1_quantity=1 MWh`) forces the LP to fall back on the ∞ tail tier; asserts feasibility only.

- No v_trade value asserted numerically — that's a commit #3 concern (parquet extraction).  The objective-match assertion is the strongest regression check available for commit #2.
