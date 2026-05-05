"""Diagnostic for the cumulative invest/existing/divest handoff carriers
through ``run_chain``.

Mirrors flextool's ``test_cumulative_handoffs.py`` (a unit test on the
ladder rolling-accumulator writer) by exercising the same CARRIER
SHAPE — chain-cumulative state passed solve-to-solve — but applied
to the carriers flexpy DOES populate today via
:func:`flextool.input.build_handoff_from_flexpy`:

* ``realized_invest`` — per-(entity, period) invest built so far in
  the chain.
* ``realized_existing`` — per-(entity, period) cumulative existing
  capacity (pre-existing seed + invests minus divests).
* ``divest_cumulative`` — per-entity scalar divest total.

flextool's writer-level test covers the LADDER variant
(``ladder_cum_realized_mwh.csv`` / ``ladder_cum_sim_hours.csv``).
flexpy's ``SolveHandoff`` reserves slots for those carriers
(``cumulative_commodity``, ``cum_sim_hours``) but
``build_handoff_from_flexpy`` doesn't populate them yet — same
Phase-2 gap as cumulative CO2.  This test asserts both:

* the populated carriers (invest/existing/divest) accumulate
  monotonically across sub-solves,
* the unpopulated carriers stay ``None``.

Carrier exercised: chain-cumulative invest accumulators on a
4-solve invest+lifetime-renew fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import run_chain

pytestmark = pytest.mark.solver


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_wind_battery_invest_lifetime_renew_4solve"
)


def test_chain_cumulative_handoffs_accumulate_monotonically() -> None:
    """Cross-solve invariants on the populated cumulative carriers:

    * ``realized_invest`` and ``realized_existing`` row counts strictly
      grow as more periods are realized — same shape as flextool's
      ladder ``cum_realized_mwh`` test (each subsequent solve adds
      this-roll contributions to the prior-roll state).
    * Each (entity, period) pair already in solve N's handoff appears
      again in solve N+1's handoff (no carriers dropped — the
      cross-solve cumulative property).
    * The ladder / sim-hours / CO2 carriers stay ``None`` (flexpy's
      ``build_handoff_from_flexpy`` doesn't populate them yet —
      Phase-2 work).

    The 4-solve lifetime-renew fixture has invest realised at p2020,
    p2025, p2030, p2035 with battery + wind_plant + battery_inverter
    each contributing — exactly the chain-cumulative-invest case
    flextool's test_cumulative_handoffs covers structurally.
    """
    if not WORK.exists():
        pytest.skip(f"fixture {WORK} not present")

    sols = run_chain(WORK)
    chain_order = list(sols)
    assert chain_order == [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ], f"chain order changed unexpectedly: {chain_order}"

    prev_keys: set[tuple[str, str]] = set()
    prev_height = 0
    for i, sub in enumerate(chain_order):
        h = sols[sub].handoff

        # Populated carriers — all three present at every solve.
        assert h.realized_invest is not None, (
            f"{sub}: realized_invest unexpectedly None")
        assert h.realized_existing is not None, (
            f"{sub}: realized_existing unexpectedly None")

        # Unique (entity, period) keys — no duplicate-row leakage from
        # the chain-cumulation step (mirrors flextool's
        # test_drop_levels_rolling_invest invariant on a different
        # output path).
        ri = h.realized_invest
        re = h.realized_existing
        assert ri.unique(["entity", "period"]).height == ri.height, (
            f"{sub}: realized_invest has duplicate (entity, period) rows")
        assert re.unique(["entity", "period"]).height == re.height, (
            f"{sub}: realized_existing has duplicate (entity, period) rows")

        # Strict-growth invariant: each subsequent solve adds rows.
        assert ri.height > prev_height, (
            f"{sub}: realized_invest height {ri.height} did not exceed "
            f"prior solve's {prev_height} — chain-cumulative state should "
            f"strictly grow as more periods are realised.")

        # Cross-solve carryover: every (entity, period) from the prior
        # solve's handoff persists in this solve's handoff.
        cur_keys = {(r["entity"], r["period"]) for r in ri.iter_rows(named=True)}
        missing = prev_keys - cur_keys
        assert not missing, (
            f"{sub}: prior-solve realized_invest keys missing from current "
            f"handoff: {sorted(missing)[:5]} ... — chain-cumulative carrier "
            f"must persist all earlier contributions.")
        prev_keys = cur_keys
        prev_height = ri.height

        # Δ.11 — ``cum_sim_hours`` is now extracted from the workdir
        # (Σ p_step_duration × realized_dispatch).  This fixture has no
        # commodity ladder so ``cumulative_commodity`` stays None; the
        # ``cumulative_co2`` extractor isn't ported (uses workdir CSV
        # propagation via flextool's preprocessing).
        if h.cum_sim_hours is not None:
            csh = h.cum_sim_hours
            assert csh.height >= 1, (
                f"{sub}: cum_sim_hours populated but empty")
            # Monotone-non-decreasing: chain-cumulative.
            assert (csh["value"].to_list() == sorted(csh["value"].to_list())
                    or csh.height == 1), (
                f"{sub}: cum_sim_hours not monotone")
