"""Parity test for the commodity price-ladder annual variant.

Fixture: ``tests/data/work_commodity_ladder_annual`` — coal scenario
with ``price_method = price_ladder_annual`` and a 2-tier ladder
(tier 1: 20 €/MWh capped at 1 MWh; tier 2: 30 €/MWh, +∞).

The cap binds tier 1, forcing the LP to spill into tier 2 at the
higher per-MWh price.  polar_high's per-tier ``v_trade`` decomposition
plus the new objective term must match flextool's HiGHS objective
to relative tolerance 1e-6.
"""
import polars as pl

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
import pytest

pytestmark = pytest.mark.solver


# Phase 3d: scenario ``coal_ladder_annual`` added to ``tests.json`` via
# ``tests/fixtures/_augment_phase3d.py`` (coal + alt ladder_ann_on).
SCENARIO = "coal_ladder_annual"


def test_commodity_ladder_annual_parity(scenario_workdir):
    """polar_high obj == flextool's HiGHS objective for the annual ladder.

    Also confirms the per-tier v_trade decomposition: the cheap finite
    tier 1 binds at its 1-MWh cap, and the +∞ tier 2 absorbs the
    overflow.  This pins the entire algebra (variables × balance × cap
    × per-tier price) end-to-end.
    """
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    rel = abs(sol.obj - flextool_obj) / max(1.0, flextool_obj)
    assert rel < 1e-6, (
        f"annual ladder parity failed: polar_high={sol.obj}, "
        f"flextool={flextool_obj}, rel={rel:.3e}"
    )

    # v_trade splits across tiers — tier 1 binds at the 1 MWh cap, tier
    # 2 (+∞) absorbs overflow at the 30 €/MWh price.
    v_trade = sol.value("v_trade").sort("c", "n", "d", "i")
    # Pre-Tier-4 backfill: nail down v_trade column structure so the smoke
    # test's parquet column assertion (commodity, node, tier) can retire.
    assert {"c", "n", "i"}.issubset(set(v_trade.columns)), (
        f"v_trade missing expected (commodity=c, node=n, tier=i) columns: "
        f"got {v_trade.columns}"
    )
    tier1_total = v_trade.filter(pl.col("i") == "1")["value"].sum()
    tier2_total = v_trade.filter(pl.col("i") == "2")["value"].sum()
    assert 0.5 < tier1_total < 1.5, (
        f"tier 1 should bind at ~1 MWh cap, got {tier1_total}")
    assert tier2_total > 100.0, (
        f"tier 2 (+∞) should absorb overflow, got {tier2_total}")

    # r_cost_ladder proxy: Σ price[c,d,i] × v_trade[c,n,d,i] must be > 0.
    # Catches regressions where v_trade routes correctly (so the per-tier
    # asserts above still pass) but ladder prices fail to flow into the
    # objective — e.g. a tier-index mismatch between writer and reader.
    from flextool.engine_polars._axis_enums import cast_dim
    # Pre-Tier-4 backfill: pin the CSV header + row-count so the smoke
    # test's TestLadderPerPeriodAnnual assertions can retire.
    _ladder_csv = work / "input" / "commodity_ladder_annual.csv"
    _ladder_lines = _ladder_csv.read_text().splitlines()
    assert _ladder_lines[0] == "commodity,period,tier,price,quantity", (
        f"unexpected commodity_ladder_annual.csv header: {_ladder_lines[0]}"
    )
    assert len(_ladder_lines) - 1 >= 2, (
        f"expected >=2 ladder rows (one per tier), got {_ladder_lines[1:]}"
    )
    prices = pl.read_csv(_ladder_csv).select(
        cast_dim(pl.col("commodity"), None, "c").alias("c"),
        cast_dim(pl.col("period"), None, "d").alias("d"),
        cast_dim(pl.col("tier").cast(pl.Utf8), None, "i").alias("i"),
        pl.col("price"),
    )
    ladder_cost = (
        v_trade.join(prices, on=["c", "d", "i"], how="inner")
        .select((pl.col("price") * pl.col("value")).sum())
        .item()
    )
    assert ladder_cost > 0, (
        f"expected positive ladder cost (Σ price × v_trade), got {ladder_cost}")
