"""Δ.11 — hand-checked parity for the ``cum_sim_hours`` and
``cumulative_commodity`` SolveHandoff carriers.

These two carriers were "left unfilled" in the original Δ.1 split
(``build_handoff_from_solution`` only propagated them through the
workdir CSV / prior-handoff path).  The Δ.11 task-spec required us
either to find a fixture exercising them OR construct a hand-checked
fixture and verify the extraction.

The ``work_commodity_ladder_cumulative`` fixture is the natural test:

* 2-day single-period dispatch (48 hourly timesteps × 1 h step duration).
* Coal commodity with a 2-tier price ladder (cumulative variant):
    - tier 1: 20 €/MWh capped at 1 MWh cumulative,
    - tier 2: 30 €/MWh, +∞ cap (absorbs overflow).
* Single solve → no prior_handoff; the carriers are this-roll-only.

Hand-derivation
---------------

Per the legacy ladder rolling accumulator algorithm:

    horizon_hrs[p2020]   = Σ_t step_duration[p2020, t]   = 48
    realized_hrs[p2020]  = Σ_t step_duration[p2020, t]   = 48   (all realized)
    fraction[p2020]      = realized_hrs / horizon_hrs    = 1.0

    cum_sim_hours[p2020]                = 48             (this-roll = horizon)
    cumulative_commodity[coal, 1, p2020] = Σ_n v_trade[coal, n, p2020, 1]
                                              × unitsize[coal] × fraction
                                          ≈ 1.0          (tier 1 binds at cap)

Tier 2 (quantity = 1e30 infinity sentinel) is intentionally excluded
from the carrier — B1 parity with the legacy
``_load_finite_ladder_tiers`` filter.  The
mod's ``p_ladder_cum_realized_mwh`` defaults to 0 for any
unspecified ``(c, i, d)`` triple so an infinite-cap tier's absent
row collapses to the same 0-default the on-disk format produces.

The test asserts the extracted carrier matches the hand-derived
total within machine precision — which is the parity contract for
the Δ.11 in-memory carrier path.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars.input import build_handoff_from_solution

pytestmark = pytest.mark.solver


# Phase 3d: scenario added to ``tests.json`` via _augment_phase3d.py.
SCENARIO = "coal_ladder_cumulative"


@pytest.fixture(scope="module")
def _fixture(scenario_workdir):
    """Solve ``coal_ladder_cumulative`` once and return
    ``(sol, handoff, work, data)``.

    The same scenario is used by ``test_flex_commodity_ladder_cumulative``
    for objective-level parity; here we focus on the carrier extraction.

    Pass ``flex_data=`` to ``build_handoff_from_solution`` so the
    cumulative-commodity / cum_sim_hours carrier extraction picks up
    ``p_step_duration`` from the in-memory frame.  The cascade does not
    emit ``solve_data/p_step_duration.csv`` (it's a derived value, not
    a writer output), so the disk-only fallback would return None on
    these tmp workdirs.
    """
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, "LP must be optimal for the carrier extraction"
    handoff = build_handoff_from_solution(
        sol, work, "y2020_2day_dispatch", flex_data=data,
    )
    return sol, handoff, work, data


def test_cum_sim_hours_extracted_from_workdir(_fixture):
    """``cum_sim_hours[p2020] == 48`` (Σ step_duration over realized
    timesteps)."""
    _sol, handoff, _work, _data = _fixture
    carrier = handoff.cum_sim_hours
    assert carrier is not None, (
        "cum_sim_hours must be populated for a fixture with "
        "realized_dispatch + p_step_duration")
    assert carrier.height == 1, (
        f"single period → single carrier row, got {carrier.height}")
    row = carrier.row(0, named=True)
    assert row["period"] == "p2020"
    # 48 timesteps × 1 hour each = 48 hours realized.
    assert abs(row["p_ladder_cum_sim_hours"] - 48.0) < 1e-9, (
        f"hand-derived cum_sim_hours[p2020] = 48; got {row['p_ladder_cum_sim_hours']}")


def test_cumulative_commodity_extracted_from_v_trade(_fixture):
    """``cumulative_commodity[coal, i, p2020]`` matches the
    uniform-split formula: Σ_n v_trade × unitsize × (realized/horizon).

    B1 — infinite-cap tiers (quantity >= 1e29 sentinel) are filtered
    out of the carrier, mirroring legacy
    ``write_ladder_rolling_accumulators`` which only persists rows for
    ``_load_finite_ladder_tiers``.  This fixture has tier-1 (quantity =
    1.0) finite and tier-2 (quantity = 1e30) infinite, so only the
    tier-1 row materialises.  The mod's ``p_ladder_cum_realized_mwh``
    defaults to 0 for unspecified ``(c, i, d)`` triples so infinite
    tiers are silently 0 anyway — the on-disk byte parity stays intact.
    """
    sol, handoff, _work, _data = _fixture
    carrier = handoff.cumulative_commodity
    assert carrier is not None, (
        "cumulative_commodity must be populated for a fixture with "
        "v_trade + ci_ladder_cumulative")
    # Fixture: tier-1 finite (cap = 1 MWh), tier-2 infinite (1e30 sentinel).
    # Only finite tiers materialise in the carrier (B1 parity with legacy
    # ``_load_finite_ladder_tiers``).
    assert carrier.height == 1, (
        f"one finite tier → one carrier row, got {carrier.height}")
    by_tier = {(r["commodity"], int(r["tier"])): float(r["p_ladder_cum_realized_mwh"])
                  for r in carrier.iter_rows(named=True)}

    # Hand check: tier 1 binds at the 1 MWh cumulative cap.
    assert abs(by_tier[("coal", 1)] - 1.0) < 1e-6, (
        f"hand-derived tier-1 mwh = 1.0; got {by_tier[('coal', 1)]}")
    # Tier 2 (infinite cap) is intentionally absent — the mod defaults
    # ``p_ladder_cum_realized_mwh`` to 0 for unspecified rows.
    assert ("coal", 2) not in by_tier, (
        "infinite-cap tier 2 must NOT appear in carrier; got "
        f"{by_tier.get(('coal', 2))}")

    # Independent re-derivation from v_trade for the parity
    # cross-check (mirror of the extractor's algorithm, computed
    # without going through the SolveHandoff carrier path).  Restricted
    # to finite tiers (tier-1 only) to match the carrier's filter.
    v_trade = sol.value("v_trade")
    # All timesteps realized → fraction = 1.0; unitsize defaults to 1.0
    # since p_commodity_unitsize.csv is absent for this fixture.
    expected: dict[tuple[str, int], float] = {}
    for r in v_trade.iter_rows(named=True):
        c = str(r["c"])
        i = int(r["i"])
        v = float(r["value"])
        if (c, i) != ("coal", 1):
            continue
        expected[(c, i)] = expected.get((c, i), 0.0) + v
    for key, exp_v in expected.items():
        got = by_tier.get(key)
        assert got is not None, f"missing carrier row for {key}"
        # fraction = 1.0 → carrier mwh == Σ v_trade.
        assert abs(got - exp_v) < 1e-6, (
            f"carrier {key} mismatch vs hand-derived: "
            f"carrier={got}, expected={exp_v}")


def test_cumulative_carriers_propagate_with_prior_handoff(_fixture):
    """Δ.11 — verify the chain accumulator: ``carrier_n =
    prior_n−1 + this_roll_n``.

    Pass an artificial prior_handoff with non-zero
    ``cum_sim_hours`` and ``cumulative_commodity`` carriers; the
    extractor must add this-roll's contribution on top.
    """
    sol, _handoff, work, data = _fixture
    # Prior carries 100 simulated hours in p2019 and 12 mwh on (coal, 1, p2019).
    from flextool.engine_polars._solve_handoff import SolveHandoff
    prior = SolveHandoff(
        cum_sim_hours=pl.DataFrame(
            [("p2019", 100.0)], schema=["period", "p_ladder_cum_sim_hours"], orient="row"),
        cumulative_commodity=pl.DataFrame(
            [("coal", 1, "p2019", 12.0)],
            schema=["commodity", "tier", "period", "p_ladder_cum_realized_mwh"], orient="row"),
    )
    handoff2 = build_handoff_from_solution(
        sol, work, "y2020_2day_dispatch",
        prior_handoff=prior, flex_data=data)
    # cum_sim_hours: prior p2019=100 carried; this-roll p2020=48 added.
    csh = {r["period"]: float(r["p_ladder_cum_sim_hours"])
              for r in handoff2.cum_sim_hours.iter_rows(named=True)}
    assert csh.get("p2019") == 100.0, (
        f"prior cum_sim_hours[p2019]=100 must carry forward; got {csh.get('p2019')}")
    assert abs(csh.get("p2020", 0.0) - 48.0) < 1e-9, (
        f"this-roll cum_sim_hours[p2020]=48; got {csh.get('p2020')}")
    # cumulative_commodity: prior (coal, 1, p2019)=12 carried;
    # this-roll (coal, 1, p2020)≈1.0 added independently.
    cc = {(r["commodity"], int(r["tier"]), r["period"]): float(r["p_ladder_cum_realized_mwh"])
              for r in handoff2.cumulative_commodity.iter_rows(named=True)}
    assert cc.get(("coal", 1, "p2019")) == 12.0, (
        f"prior commodity p2019 row must carry; got {cc.get(('coal', 1, 'p2019'))}")
    assert abs(cc.get(("coal", 1, "p2020"), 0.0) - 1.0) < 1e-6, (
        f"this-roll (coal, 1, p2020)≈1.0; got {cc.get(('coal', 1, 'p2020'))}")
