"""Parity test for the commodity price-ladder cumulative variant.

Fixture: ``tests/data/work_commodity_ladder_cumulative`` — coal scenario
with ``price_method = price_ladder_cumulative`` and a 2-tier ladder
(tier 1: 20 €/MWh capped at 1 MWh cumulative; tier 2: 30 €/MWh, +∞).

On a single-period non-rolling solve the cumulative cap reduces to
the pre-refactor form (Σ f_d_k = 1.0, no prior accumulator).
"""
from pathlib import Path

import polars as pl

from polar_high_opt import Problem
from flextool.engine_polars import build_flextool, load_flextool


WORK = Path(__file__).resolve().parent / "data" / "work_commodity_ladder_cumulative"


def test_commodity_ladder_cumulative_parity():
    """flexpy obj == flextool's HiGHS objective for the cumulative ladder.

    Also confirms the per-tier v_trade decomposition: tier 1 binds at
    its 1-MWh cumulative cap and tier 2 (+∞) absorbs the rest.
    """
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    rel = abs(sol.obj - flextool_obj) / max(1.0, flextool_obj)
    assert rel < 1e-6, (
        f"cumulative ladder parity failed: flexpy={sol.obj}, "
        f"flextool={flextool_obj}, rel={rel:.3e}"
    )

    v_trade = sol.value("v_trade").sort("c", "n", "d", "i")
    tier1_total = v_trade.filter(pl.col("i") == "1")["value"].sum()
    tier2_total = v_trade.filter(pl.col("i") == "2")["value"].sum()
    assert 0.5 < tier1_total < 1.5, (
        f"tier 1 should bind at ~1 MWh cap, got {tier1_total}")
    assert tier2_total > 100.0, (
        f"tier 2 (+∞) should absorb overflow, got {tier2_total}")

    # r_cost_ladder proxy: Σ price[c,i] × v_trade[c,n,d,i] must be > 0.
    # Cumulative ladder has no period dimension — the same per-tier price
    # applies across all periods. Catches regressions where v_trade routes
    # correctly but prices fail to flow into the objective.
    prices = pl.read_csv(WORK / "input" / "commodity_ladder_cumulative.csv").select(
        pl.col("commodity").alias("c"),
        pl.col("tier").cast(pl.Utf8).alias("i"),
        pl.col("price"),
    )
    ladder_cost = (
        v_trade.join(prices, on=["c", "i"], how="inner")
        .select((pl.col("price") * pl.col("value")).sum())
        .item()
    )
    assert ladder_cost > 0, (
        f"expected positive ladder cost (Σ price × v_trade), got {ladder_cost}")
