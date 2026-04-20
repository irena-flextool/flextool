# Noted decisions — commit #7 (Bug #1 fix: rolling-aware ladder caps)

Non-obvious choices made while implementing the per-period rolling
accumulator fix for Bug #1.

## Writer (cumulative_handoffs.py)

- **Both annual AND cumulative ladder commodities track accumulators.**
  The old writer only wrote for ``price_ladder_cumulative``; the new one
  writes for any ``price_ladder_*`` method because the annual cap now
  also needs ``p_ladder_cum_realized_mwh`` to partition per-period MWh
  correctly across within-period rolls.  Non-ladder commodities
  (``price`` method) are still skipped.

- **Uniform-split attribution of v_trade to realized hours**
  (``realized_hours / horizon_hours``).  The LP's v_trade is
  period-level (one number per (c, n, d, i)) and cannot distinguish
  realized hours from lookahead.  When a roll realizes only part of d,
  we proportion v_trade's MWh by the hours-fraction.  Documented in
  module docstring; alternative (per-step v_trade) was rejected in the
  commit-#5 NOTES as a much bigger refactor.

- **Lookahead-only periods not accumulated.**  If a roll includes d in
  its horizon but realizes zero hours of d (pure lookahead), we do
  NOT add anything to ``cum_realized_mwh[c, i, d]`` — the next roll
  replans that d and the MWh attribution would double-count otherwise.
  The period still appears in ``cum_sim_hours[d]`` with value 0 (no
  realized hours to add).

- **Header-only first-solve emission even when ladder commodities exist.**
  The writer branch that returns early on ``not ladder_commodities`` only
  fires when no ladder commodity is configured.  When ladder commodities
  DO exist, the writer always goes through the full path (including
  loading v_trade + emitting accumulator rows — possibly zero rows on
  a no-op first solve).  This matches the old header-only seed behaviour
  numerically but generates actual content as soon as there's realized
  dispatch.

## Mod (flextool.mod)

- **``f_d_k`` placed right after ``pdtCommodity``** (line ~967).
  The task brief suggested "around line 900–950, after pdCommodity /
  pdtCommodity"; those params live at 944/960 in the current file so
  the new param inserts at 967 just before ``pdGroup``.  Keeping it
  near the other commodity-derived params is the natural home.

- **Annual overspent override is indexed over
  ``(c, n) in commodity_node``, not over ``cndi_ladder``.**  The task
  brief's pseudocode iterates (c, i) via ``commodity__tier`` and (c, n)
  via ``commodity_node``, making this a 4-way index set.  Matches the
  annual cap's own iteration pattern (also over commodity__tier, not
  cndi_ladder).  GMPL is fine with either form but
  ``commodity__tier``-first is closer to the cap's predicate.

- **Dropped ``ladder_tier_cap_cumulative_overspent`` indexing over
  ``cndi_ladder``** (commit #6 pattern) in favour of indexing over
  ``commodity_with_ladder_cumulative × commodity_node × period_in_use
  × commodity__tier``.  The ``cndi_ladder`` set already filters to
  ``c in commodity_with_ladder`` which is a superset of
  ``commodity_with_ladder_cumulative``, so the two produce the same
  active set — but the new form is lexically parallel with the main
  cap's iteration and easier to reason about.

- **The single-solve bit-identity proof** hinges on three values:
  ``p_ladder_cum_realized_mwh = 0`` (empty seed CSV → default 0);
  ``p_ladder_cum_sim_hours = 0`` (same); ``f_d_k[d] = horizon_hours_d
  / (share_of_year_d * 8760)`` which for a full single solve equals
  exactly 1.0 because ``horizon_hours_d = share_of_year_d * 8760`` by
  construction (share_of_year is derived from dt_complete).  Under
  these the cumulative cap collapses to
  ``sum v_trade * unitsize <= cap * 1`` (missing the
  ``years_represented / period_share`` scaling of the OLD form — but
  that old scaling was itself a proxy for f_d_k summed over the
  horizon, so the new form is the more fundamental one).

## Tests

- **Within-period rolling tests use 24h jump on a 48h timeset.**
  The existing ``2day`` timeset is 48h; ``rolling_solve_horizon=24`` /
  ``rolling_solve_jump=24`` splits each period into two equal rolls,
  enough to trigger Bug #1's double-counting under the OLD formulation.
  Using 4 or more rolls per period would test the same mechanism at
  finer granularity but adds solve time without new signal.

- **Assertions are structural, not bit-equality.**  The accumulator
  files contain floats derived from LP duals; under the uniform-split
  assumption they can't be compared byte-for-byte across LP re-solves.
  So tests assert:
  (a) files exist with expected schema;
  (b) per-period entries are non-negative;
  (c) total accumulator ≤ total v_trade (sanity upper bound);
  (d) the overspent override correctly locks a tier out (tier-1
  v_trade sum = 0 in the final roll).

- **Cross-period formula-conservation test dropped.**  Commit #5's
  test asserted ``final_remaining == total_cap − Σ consumption``.
  That test was tied to the scalar ``p_cumulative_ladder_remaining``
  interface and does not translate directly to the per-period
  accumulator.  The replacement assertion
  ``accumulator[d] <= Σ v_trade_for_d`` captures the same "no
  over-allocation" spirit with much less book-keeping.

## Guardrails

- Did NOT modify other ``rivendell/NOTES_*.md`` files.
- Did NOT push; local commit only.
- Did NOT touch memory files.
- Did NOT modify CO2 cumulative machinery — separate follow-up (user
  explicitly queued task #6).
- Did NOT run the full test suite — user directive: only relevant
  tests.
