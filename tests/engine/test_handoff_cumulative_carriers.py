"""Δ.11 — hand-checked parity for the ``cum_sim_hours`` and
``cumulative_commodity`` SolveHandoff carriers.

These two carriers were "left unfilled" in the original Δ.1 split
(``build_handoff_from_solution`` only propagated them through the
workdir CSV / prior-handoff path).  The Δ.11 task-spec required us
either to find a fixture exercising them OR construct a hand-checked
fixture and verify the extraction.

The ``work_commodity_ladder_cumulative`` fixture is the natural test:

* 2-day single-period dispatch (48 hourly timesteps × 1 h step duration).
* Coal commodity with a 3-tier price ladder (cumulative variant):
    - tier 1: 20 €/MWh capped at 1,000,000 MWh cumulative,
    - tier 2: 25 €/MWh capped at 1,000,000 MWh cumulative,
    - tier 3: 30 €/MWh, +∞ cap (absorbs overflow).
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

The 2-day coal demand is far below the 1,000,000 MWh tier-1 cap, so
tier 1 absorbs it all without binding; tier 2 (finite, also unused)
materialises at 0.  The exact MWh total is cross-checked against
``v_trade`` directly in the test below rather than hard-coded here.

Tier 3 (quantity = 1e30 infinity sentinel) is intentionally excluded
from the carrier — B1 parity with the legacy
``_load_finite_ladder_tiers`` filter, which only keeps finite tiers.  The
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
from flextool.engine import build_flextool
from flextool.engine.preprocess.flex_data import build_handoff_from_solution
from tests._support.load import clone_provider, load_scenario

pytestmark = pytest.mark.solver


# Phase 3d: scenario added to ``tests.json`` via _augment_phase3d.py.
SCENARIO = "coal_ladder_cumulative"


@pytest.fixture(scope="module")
def _fixture(scenario_run):
    """Solve ``coal_ladder_cumulative`` once and return
    ``(sol, handoff, run, data, provider)``.

    The same scenario is used by ``test_flex_commodity_ladder_cumulative``
    for objective-level parity; here we focus on the carrier extraction.

    The production call pair: the FlexData is rebuilt with the step's
    provider (a clone, so the session-cached provider is never touched) and
    the handoff is built from the same provider, with ``flex_data=`` (the
    carrier extraction reads ``p_step_duration`` from the in-memory frame),
    as the cascade passes them.  ``work_folder`` is the run's empty load
    folder: the builder reads its inputs from the provider.
    """
    run = scenario_run(SCENARIO)
    provider = clone_provider(run.last.provider)
    data = load_scenario(run, provider=provider)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, "LP must be optimal for the carrier extraction"
    handoff = build_handoff_from_solution(
        sol, run.load_folder, "y2020_2day_dispatch",
        prior_handoff=run.last.prior_handoff, flex_data=data, provider=provider,
    )
    return sol, handoff, run, data, provider


def test_cum_sim_hours_extracted_from_workdir(_fixture):
    """``cum_sim_hours[p2020] == 48`` (Σ step_duration over realized
    timesteps)."""
    _sol, handoff, _run, _data, _provider = _fixture
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
    ``_load_finite_ladder_tiers``.  This fixture has tier-1 and tier-2
    (quantity = 1,000,000 each) finite and tier-3 (quantity = 1e30)
    infinite, so a row materialises for every finite tier regardless of
    whether it is actually used — only the infinite tier-3 is excluded.
    The 2-day coal demand is well under the tier-1 cap, so all of it
    lands in tier 1 and tier 2's row is present but 0.
    """
    sol, handoff, _run, _data, _provider = _fixture
    carrier = handoff.cumulative_commodity
    assert carrier is not None, (
        "cumulative_commodity must be populated for a fixture with "
        "v_trade + ci_ladder_cumulative")
    # Fixture: tier-1/2 finite (cap = 1,000,000 MWh each) both
    # materialise; tier-3 is infinite (1e30 sentinel) and excluded.
    assert carrier.height == 2, (
        f"two finite tiers → two carrier rows, got {carrier.height}")
    by_tier = {(r["commodity"], int(r["tier"])): float(r["p_ladder_cum_realized_mwh"])
                  for r in carrier.iter_rows(named=True)}

    # Tier 3 (infinite cap) is intentionally absent — the mod defaults
    # ``p_ladder_cum_realized_mwh`` to 0 for unspecified rows.
    assert ("coal", 3) not in by_tier, (
        "infinite-cap tier 3 must NOT appear in carrier; got "
        f"{by_tier.get(('coal', 3))}")
    # Tier 2 is finite but unused for this fixture's demand — its row
    # is present with an explicit 0, not omitted.
    assert by_tier.get(("coal", 2)) == 0.0, (
        f"unused finite tier 2 should be present at 0; got {by_tier.get(('coal', 2))}")

    # Independent re-derivation from v_trade for the parity
    # cross-check (mirror of the extractor's algorithm, computed
    # without going through the SolveHandoff carrier path).  Restricted
    # to finite tiers (tier-1 and tier-2) to match the carrier's filter.
    v_trade = sol.value("v_trade")
    # All timesteps realized → fraction = 1.0; unitsize defaults to 1.0
    # since p_commodity_unitsize.csv is absent for this fixture.
    expected: dict[tuple[str, int], float] = {}
    for r in v_trade.iter_rows(named=True):
        c = str(r["c"])
        i = int(r["i"])
        v = float(r["value"])
        if c != "coal" or i not in (1, 2):
            continue
        expected[(c, i)] = expected.get((c, i), 0.0) + v
    assert expected[("coal", 1)] < 1_000_000.0, (
        "tier 1 should not bind at its cap for this 2-day fixture; "
        f"got {expected[('coal', 1)]}")
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
    sol, _handoff, run, data, provider = _fixture
    # Prior carries 100 simulated hours in p2019 and 12 mwh on (coal, 1, p2019).
    from flextool.engine.solve._handoff import SolveHandoff
    prior = SolveHandoff(
        cum_sim_hours=pl.DataFrame(
            [("p2019", 100.0)], schema=["period", "p_ladder_cum_sim_hours"], orient="row"),
        cumulative_commodity=pl.DataFrame(
            [("coal", 1, "p2019", 12.0)],
            schema=["commodity", "tier", "period", "p_ladder_cum_realized_mwh"], orient="row"),
    )
    handoff2 = build_handoff_from_solution(
        sol, run.load_folder, "y2020_2day_dispatch",
        prior_handoff=prior, flex_data=data, provider=provider)
    # cum_sim_hours: prior p2019=100 carried; this-roll p2020=48 added.
    csh = {r["period"]: float(r["p_ladder_cum_sim_hours"])
              for r in handoff2.cum_sim_hours.iter_rows(named=True)}
    assert csh.get("p2019") == 100.0, (
        f"prior cum_sim_hours[p2019]=100 must carry forward; got {csh.get('p2019')}")
    assert abs(csh.get("p2020", 0.0) - 48.0) < 1e-9, (
        f"this-roll cum_sim_hours[p2020]=48; got {csh.get('p2020')}")
    # cumulative_commodity: prior (coal, 1, p2019)=12 carried;
    # this-roll (coal, 1, p2020) is added independently of the prior
    # value — cross-check against the same roll's own v_trade total
    # (hand-derived the same way as test_cumulative_commodity_extracted_from_v_trade)
    # rather than a hard-coded MWh literal.
    cc = {(r["commodity"], int(r["tier"]), r["period"]): float(r["p_ladder_cum_realized_mwh"])
              for r in handoff2.cumulative_commodity.iter_rows(named=True)}
    assert cc.get(("coal", 1, "p2019")) == 12.0, (
        f"prior commodity p2019 row must carry; got {cc.get(('coal', 1, 'p2019'))}")
    this_roll_tier1 = (
        sol.value("v_trade")
        .filter((pl.col("c") == "coal") & (pl.col("i") == "1"))["value"].sum()
    )
    assert abs(cc.get(("coal", 1, "p2020"), 0.0) - this_roll_tier1) < 1e-6, (
        f"this-roll (coal, 1, p2020) should equal Σ v_trade[coal,1]="
        f"{this_roll_tier1}; got {cc.get(('coal', 1, 'p2020'))}")
