"""Commodity price ladder: a commodity bought in tiers, each with its own price and quantity.

For commodities with price_method price_ladder_annual / price_ladder_cumulative, the purchase at each node
and period is split into tiers (v_trade).  A balance row ties the tiers to the commodity flow, cap rows
limit each tier per period (annual) or over the horizon (cumulative), and the objective prices each tier,
weighted by the branch weight of the period in stochastic solves.  The forecast branch periods of a
stochastic solve use the annual ladder of their anchor period.  The single commodity price term in
model/core.py skips these commodities.  Earlier solves of a chain enter
through f_d_k (share of the period simulated so far) and the tier MWh they already bought.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Expr

from flextool.engine.model._common import _has_rows
from flextool.engine.preprocess.params._branch import branch_anchor_pairs, inherit_anchor_period

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


# Tier quantities at or above this are unlimited (+Inf is written as 1e30) and get no cap row.
_INF_SENTINEL = 1e29


# ---------------------------------------------------------------------------
# Feature detection


def has_feature(d) -> bool:
    """True when at least one commodity uses a ladder price method."""
    cwl = getattr(d, "commodity_with_ladder", None)
    return cwl is not None and cwl.height > 0


# ---------------------------------------------------------------------------
# Loader


def _provider_get(provider, key: str) -> pl.DataFrame | None:
    """The provider's frame for ``key``; None when the provider is missing, lacks it, or it is empty."""
    if provider is None or not provider.has(key):
        return None
    df = provider.get(key)
    return df if df is not None and df.height > 0 else None


def _read_single_col(provider, key: str, col_in: str, col_out: str) -> pl.DataFrame | None:
    """The one-column frame ``key`` with ``col_in`` renamed to ``col_out``; None when missing or empty."""
    df = _provider_get(provider, key)
    if df is None:
        return None
    if col_in in df.columns and col_in != col_out:
        df = df.rename({col_in: col_out})
    return df.select(col_out)


def _read_long(provider, key: str, rename: dict[str, str]) -> pl.DataFrame | None:
    """The frame ``key`` with the present columns of ``rename`` renamed; None when missing or empty."""
    df = _provider_get(provider, key)
    if df is None:
        return None
    cols_present = {c: r for c, r in rename.items() if c in df.columns}
    if cols_present:
        df = df.rename(cols_present)
    return df


def load_data(inp_dir: str | Path, sd_dir: str | Path, *, provider=None) -> dict:
    """The ladder sets and Params, as ``field_name -> DataFrame | Param | None``.

    Every value is None when no commodity has a ladder.  Frames are read through ``provider``; ``inp_dir``
    / ``sd_dir`` are accepted for the loaders' uniform call.  Tier labels ``i`` are Utf8 everywhere.
    The forecast branch periods of a stochastic solve get their annual ladder later, in
    inherit_branch_periods (called by load_flextool once every producer of the ladder Params has run).
    p_commodity_unitsize is filled by the direct-parameter pass (default 1.0, _commodity_unitsize_param).
    """
    blank = dict(
        commodity_with_ladder=None,
        commodity_with_ladder_annual=None,
        commodity_with_ladder_cumulative=None,
        cnd_ladder=None,
        cndi_ladder=None,
        cndi_ladder_ann=None,
        cndi_ladder_cum=None,
        ci_ladder_cumulative=None,
        commodity__tier_ann=None,
        commodity__tier_cum=None,
        p_ladder_ann_price=None,
        p_ladder_ann_quantity=None,
        p_ladder_cum_price=None,
        p_ladder_cum_quantity=None,
        p_commodity_unitsize=None,
        p_f_d_k=None,
        p_ladder_cum_realized_mwh=None,
    )

    cwl = _read_single_col(provider, "solve_data/commodity_with_ladder", "commodity", "c")
    if cwl is None:
        return blank
    cwla = _read_single_col(provider, "solve_data/commodity_with_ladder_annual", "commodity", "c")
    cwlc = _read_single_col(provider, "solve_data/commodity_with_ladder_cumulative", "commodity", "c")

    cnd = _read_long(provider, "solve_data/cnd_ladder_set.csv", {"commodity": "c", "node": "n", "period": "d"})
    if cnd is not None:
        cnd = cnd.select("c", "n", "d")

    def _read_cndi(key: str) -> pl.DataFrame | None:
        """(c, n, d, i) rows of ``key``; the integer tier read as Utf8 to match the Param tables."""
        df = _read_long(provider, key, {"commodity": "c", "node": "n", "period": "d", "tier": "i"})
        if df is None:
            return None
        return df.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "n", "d", "i")

    cndi = _read_cndi("solve_data/cndi_ladder_set.csv")
    cndi_ann = _read_cndi("solve_data/cndi_ladder_ann_set.csv")
    cndi_cum = _read_cndi("solve_data/cndi_ladder_cum_set.csv")

    ci_cum = _read_long(provider, "solve_data/ci_ladder_cumulative.csv", {"commodity": "c", "tier": "i"})
    if ci_cum is not None:
        ci_cum = ci_cum.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "i")

    ct_ann = _read_long(provider, "solve_data/commodity__tier_ann.csv", {"commodity": "c", "tier": "i"})
    if ct_ann is not None:
        ct_ann = ct_ann.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "i")
    # commodity__tier_cum: the (commodity, tier) pairs of the cumulative ladder input.
    ct_cum = None
    cum_inp = _provider_get(provider, "input/commodity_ladder_cumulative")
    if cum_inp is not None:
        ct_cum = (cum_inp.rename({"commodity": "c", "tier": "i"})
                  .with_columns(pl.col("i").cast(pl.Utf8))
                  .select("c", "i").unique())

    # ── Annual price / quantity (c, i, d) ──
    p_ann_price = None
    p_ann_quantity = None
    ann = _provider_get(provider, "input/commodity_ladder_annual")
    if ann is not None:
        ann = (ann.rename({"commodity": "c", "tier": "i", "period": "d"})
               .with_columns(pl.col("i").cast(pl.Utf8))
               .select("c", "i", "d", "price", "quantity"))
        p_ann_price = Param(("c", "i", "d"), ann.select("c", "i", "d", value=pl.col("price").cast(pl.Float64)))
        p_ann_quantity = Param(("c", "i", "d"),
                               ann.select("c", "i", "d", value=pl.col("quantity").cast(pl.Float64)))

    # ── Cumulative price / quantity (c, i) ──
    p_cum_price = None
    p_cum_quantity = None
    cum = _provider_get(provider, "input/commodity_ladder_cumulative")
    if cum is not None:
        cum = (cum.rename({"commodity": "c", "tier": "i"})
               .with_columns(pl.col("i").cast(pl.Utf8))
               .select("c", "i", "price", "quantity"))
        p_cum_price = Param(("c", "i"), cum.select("c", "i", value=pl.col("price").cast(pl.Float64)))
        p_cum_quantity = Param(("c", "i"), cum.select("c", "i", value=pl.col("quantity").cast(pl.Float64)))

    # f_d_k[d]: share of period d simulated by the end of this solve (1 for a single solve).  The derived
    # pass also produces it; fixtures that skip that pass rely on this read.  The value column is Utf8.
    p_f_d_k = None
    fdk = _provider_get(provider, "solve_data/f_d_k.csv")
    if fdk is not None and fdk.height > 0:
        fdk = (fdk.rename({"period": "d"})
               .with_columns(pl.col("value").cast(pl.Float64))
               .select("d", "value"))
        p_f_d_k = Param(("d",), fdk)

    # Tier MWh bought by the earlier solves of a chain, from the solve handoff; None on the first solve
    # (counted as 0).
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    p_realized = None
    rel = read_handoff_frame(provider, K.HANDOFF_CUMULATIVE_COMMODITY)
    if rel is not None and rel.height > 0:
        value_col = "p_ladder_cum_realized_mwh" if "p_ladder_cum_realized_mwh" in rel.columns else "value"
        rel = (rel.rename({"commodity": "c", "tier": "i", "period": "d", value_col: "value"})
               .with_columns(pl.col("i").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
               .select("c", "i", "d", "value"))
        p_realized = Param(("c", "i", "d"), rel)

    return dict(
        commodity_with_ladder=cwl,
        commodity_with_ladder_annual=cwla,
        commodity_with_ladder_cumulative=cwlc,
        cnd_ladder=cnd,
        cndi_ladder=cndi,
        cndi_ladder_ann=cndi_ann,
        cndi_ladder_cum=cndi_cum,
        ci_ladder_cumulative=ci_cum,
        commodity__tier_ann=ct_ann,
        commodity__tier_cum=ct_cum,
        p_ladder_ann_price=p_ann_price,
        p_ladder_ann_quantity=p_ann_quantity,
        p_ladder_cum_price=p_cum_price,
        p_ladder_cum_quantity=p_cum_quantity,
        p_commodity_unitsize=None,
        p_f_d_k=p_f_d_k,
        p_ladder_cum_realized_mwh=p_realized,
    )


def inherit_branch_periods(flex_data, provider) -> None:
    """Give the forecast branch periods of a stochastic solve their anchor period's annual ladder.

    Branch periods (period1_upper, …) are minted per solve, so no ladder row can be authored for them;
    they are the anchor period's possible futures, so the anchor's tier curve applies.  Price and quantity
    are inherited together, per commodity: a branch period that has a curve of its own for a commodity
    keeps it whole (inherit_anchor_period, group key c).  Only branch periods with ladder rows
    (cnd_ladder) are filled.  The cumulative ladder is keyed (c, i) and already holds in every period.
    Runs on the final p_ladder_ann_price / p_ladder_ann_quantity: load_data and the direct-parameter pass
    (_direct.apply_direct_params_b, which overwrites them for the period-indexed form) both produce them.
    No-op outside stochastic solves (no non-self period__branch pair).
    """
    cnd = getattr(flex_data, "cnd_ladder", None)
    price = getattr(flex_data, "p_ladder_ann_price", None)
    quantity = getattr(flex_data, "p_ladder_ann_quantity", None)
    if not _has_rows(cnd) or (price is None and quantity is None):
        return
    pairs = branch_anchor_pairs(provider, cnd["d"].cast(pl.Utf8).unique())
    if pairs is None:
        return
    parts = [p.frame.select("c", "i", "d", "value").with_columns(field=pl.lit(name))
             for name, p in (("price", price), ("quantity", quantity)) if p is not None]
    both = inherit_anchor_period(pl.concat(parts, how="vertical"), pairs, group_keys=("c",))
    if price is not None:
        flex_data.p_ladder_ann_price = Param(
            ("c", "i", "d"), both.filter(pl.col("field") == "price").select("c", "i", "d", "value"))
    if quantity is not None:
        flex_data.p_ladder_ann_quantity = Param(
            ("c", "i", "d"), both.filter(pl.col("field") == "quantity").select("c", "i", "d", "value"))


# ---------------------------------------------------------------------------
# Variables


def add_variables(b: _Build) -> None:
    """Declare v_trade when a commodity has a ladder."""
    if not b.has_ladder or not _has_rows(getattr(b.d, "cndi_ladder", None)):
        return

    # ─── VAR. ─── v_trade ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_trade[c,n,d,i] ≥ 0   ∀ (c,n,d,i) ∈ cndi_ladder = cndi_ladder_ann ∪ cndi_ladder_cum
    # Purchase of commodity c at node n in period d in price tier i, in units of
    # the commodity unitsize: MWh = v_trade · commodity_unitsize[c].
    b.v_trade = b.m.add_var("v_trade", ("c", "n", "d", "i"), b.d.cndi_ladder, lower=0.0)


# ---------------------------------------------------------------------------
# Constraints


def _commodity_unitsize_param(d) -> Param:
    """commodity_unitsize[c]; 1.0 for every ladder commodity when the parameter is not set."""
    p = getattr(d, "p_commodity_unitsize", None)
    if p is not None:
        return p
    return Param(("c",), d.commodity_with_ladder.select("c").unique().with_columns(value=pl.lit(1.0)))


def add_constraints(b: _Build) -> None:
    """commodity_ladder_balance and the tier caps, when a commodity has a ladder."""
    if b.v_trade is None:
        return
    _eq_ladder_balance(b)
    _eq_tier_cap_annual(b)
    _eq_tier_cap_cumulative(b)


def _eq_ladder_balance(b: _Build) -> None:
    """commodity_ladder_balance: the tiers add up to the commodity bought in the period."""
    d, v_flow = b.d, b.v_flow

    # ─── EQ. ─── commodity_ladder_balance ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_i v_trade[c,n,d,i] · commodity_unitsize[c]
    #   = Σ_{(p,n,k,c) ∈ flow_from_commodity_eff, t}
    #       (v_flow[p,n,k,d,t] · unitsize[p] · slope[p,d,t] + v_online[p,d,t] · section[p,d,t] · unitsize[p]) · w[d,t]
    #   + Σ_{(p,n,k,c) ∈ flow_from_commodity_noEff, t} v_flow[p,n,k,d,t] · unitsize[p] · w[d,t]
    #   − Σ_{(p,s,n,c) ∈ flow_to_commodity, t}         v_flow[p,s,n,d,t] · unitsize[p] · w[d,t]
    #   ∀ (c,n,d) ∈ cnd_ladder;   w = step_duration · timestep_weight
    # The period's net purchase of commodity c at node n (buy − sell, MWh) is split into price tiers.
    # The efficiency-arc term uses the shared fuel measure _eff_fuel_input (core.py), so min_load_efficiency
    # units' part-load section fuel counts toward the tiers too, same as commodity_cost_eff / co2_price_eff
    # / the CO2 caps (#56) — before this fix a ladder commodity bought by such a unit was under-counted by
    # the part-load fuel.  A term is absent when its flow set is empty or there are no processes.
    from flextool.engine.model.core import _eff_fuel_input

    rhs: dict = {}
    weight = d.p_step_duration * d.p_timestep_weight
    if v_flow is not None and _has_rows(d.flow_from_commodity_eff):
        idx = d.flow_from_commodity_eff.with_columns(n=pl.col("source")).select("p", "source", "sink", "c", "n")
        fuel = None
        for _key, term in _eff_fuel_input(b, idx):
            fuel = term if fuel is None else fuel + term
        rhs["buy_eff"] = Sum(fuel * weight, over=("p", "source", "sink", "t"))
    if v_flow is not None and _has_rows(d.flow_from_commodity_noEff):
        idx = d.flow_from_commodity_noEff.with_columns(n=pl.col("source")).select("p", "source", "sink", "c", "n")
        rhs["buy_noEff"] = Sum(Where(v_flow * d.p_unitsize, idx) * weight, over=("p", "source", "sink", "t"))
    if v_flow is not None and _has_rows(d.flow_to_commodity):
        idx = d.flow_to_commodity.with_columns(n=pl.col("sink")).select("p", "source", "sink", "c", "n")
        rhs["sell"] = -Sum(Where(v_flow * d.p_unitsize, idx) * weight, over=("p", "source", "sink", "t"))
    b.m.add_cstr(
        "commodity_ladder_balance",
        over      = d.cnd_ladder,
        sense     = "==",
        lhs_terms = {"trade": Sum(b.v_trade * _commodity_unitsize_param(d), over=("i",))},
        rhs_terms = rhs,
    )


def _eq_tier_cap_annual(b: _Build) -> None:
    """ladder_tier_cap_annual_roll: purchase per tier and period up to the tier's annual quantity."""
    d = b.d
    if not _has_rows(d.cndi_ladder_ann) or d.p_ladder_ann_quantity is None:
        return
    finite = d.p_ladder_ann_quantity.frame.filter(pl.col("value") < _INF_SENTINEL).select("c", "i", "d")
    cndi_ann_finite = d.cndi_ladder_ann.join(finite, on=["c", "i", "d"], how="inner")
    if cndi_ann_finite.height == 0:
        return

    # ─── EQ. ─── ladder_tier_cap_annual_roll ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_n v_trade[c,n,d,i] · commodity_unitsize[c]  ≤  quantity_ann[c,i,d] · period_share[d] · window_fraction[d]
    #   ∀ (c,i,d) with (c,n,d,i) ∈ cndi_ladder_ann and quantity_ann[c,i,d] < 1e29
    # quantity_ann is a per-YEAR amount (GMPL "per-year MWh"): period_share[d] scales it down to this
    # period's own share of the modelled year, matching the price term's inflation_op/period_share
    # annualisation (#89) — a 48h solve of a 48h period gets 48h/8760 of the annual tier, not the whole
    # year's worth.  window_fraction[d] (1 outside rolling) further scales it to the share of the
    # period's hours this roll's own window covers (same window-share idea as co2_max_period,
    # core.py _eq_co2_max_period).  Deliberately no cross-roll carry/realized subtraction: each roll's
    # share is a self-contained, always-nonnegative budget, so it can never drive the row infeasible
    # (#143) — a period resolved over several overlapping look-ahead rolls can draw a little more than
    # its strict annual share in aggregate, the same approximate-compliance trade-off co2_max_period
    # already accepts (see core.py's F4 note there).  Unlimited tiers (quantity ≥ 1e29) get no row.
    # Branch periods of a stochastic solve use their anchor period's quantity (and price) unless they
    # have their own (inherit_branch_periods).
    window_fraction = getattr(d, "p_window_fraction", None)
    cap = d.p_ladder_ann_quantity * d.p_period_share
    if window_fraction is not None:
        cap = cap * window_fraction
    b.m.add_cstr(
        "ladder_tier_cap_annual_roll",
        over      = cndi_ann_finite.select("c", "i", "d").unique(),
        sense     = "<=",
        lhs_terms = {"trade": Sum(Where(b.v_trade * _commodity_unitsize_param(d), cndi_ann_finite), over=("n",))},
        rhs_terms = {"cap": cap},
    )


def _eq_tier_cap_cumulative(b: _Build) -> None:
    """ladder_tier_cap_cumulative_roll: purchase per tier over the horizon up to the tier's quantity."""
    d = b.d
    if (not _has_rows(d.ci_ladder_cumulative) or d.p_ladder_cum_quantity is None
            or not _has_rows(d.cndi_ladder_cum)):
        return
    finite = d.p_ladder_cum_quantity.frame.filter(pl.col("value") < _INF_SENTINEL).select("c", "i")
    ci_cum_finite = d.ci_ladder_cumulative.join(finite, on=["c", "i"], how="inner")
    if ci_cum_finite.height == 0:
        return

    # ─── EQ. ─── ladder_tier_cap_cumulative_roll ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{n,d} v_trade[c,n,d,i] · commodity_unitsize[c]
    #   ≤  quantity_cum[c,i] · Σ_d f_d_k[d] − Σ_d realized[c,i,d]
    #   ∀ (c,i) ∈ ci_ladder_cumulative with quantity_cum[c,i] < 1e29;  (c,n,d,i) ∈ cndi_ladder_cum
    # Caps what tier i supplies over all periods.  Σ_d f_d_k is a scalar (1 when f_d_k is absent);
    # realized = tier MWh bought by earlier solves (0 when absent).  Unlimited tiers get no row.
    cndi_cum_finite = d.cndi_ladder_cum.join(ci_cum_finite, on=["c", "i"], how="inner")
    sum_f_d_k = float(d.p_f_d_k.frame["value"].sum()) if d.p_f_d_k is not None else 1.0

    # ─── EQ. ─── ladder_tier_cap_cumulative_roll ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{n,d} v_trade[c,n,d,i] · commodity_unitsize[c]  ≤  remaining[c,i]
    #   remaining[c,i] = max(quantity_cum[c,i] · Σ_d f_d_k[d] − Σ_d realized[c,i,d], 0)
    #   ∀ (c,i) ∈ ci_ladder_cumulative with quantity_cum[c,i] < 1e29;  (c,n,d,i) ∈ cndi_ladder_cum
    # Caps what tier i supplies over all periods.  Σ_d f_d_k is a scalar (1 when f_d_k is absent);
    # realized = tier MWh bought by earlier rolls/solves (0 when absent).  f_d_k measures progress to
    # the end of a roll's own (look-ahead-extended) horizon, so an early roll's cap can be more generous
    # than a later roll's conservative, realized-only accounting of the same ground — carried "realized"
    # MWh can then land above what that later roll's own quantity·Σf_d_k computes, which used to leave a
    # negative right-hand side and crash the solve (#143).  Clamping the remaining budget at 0 keeps the
    # row feasible (a tier simply closes once it is spent, instead of going negative); the deeper fix —
    # keeping f_d_k from overshooting within a roll's own look-ahead — is left as a follow-up (see the
    # design note). Unlimited tiers get no row.
    cap_df = (ci_cum_finite.join(d.p_ladder_cum_quantity.frame, on=["c", "i"], how="left")
              .select("c", "i", cap=pl.col("value") * sum_f_d_k))
    if d.p_ladder_cum_realized_mwh is not None:
        realized_ci = (d.p_ladder_cum_realized_mwh.frame
                       .group_by(["c", "i"]).agg(pl.col("value").sum().alias("realized")))
        cap_df = cap_df.join(realized_ci, on=["c", "i"], how="left").with_columns(
            pl.col("realized").fill_null(0.0))
    else:
        cap_df = cap_df.with_columns(realized=pl.lit(0.0))
    cap_df = cap_df.with_columns(
        value=(pl.col("cap") - pl.col("realized")).clip(lower_bound=0.0)).select("c", "i", "value")
    b.m.add_cstr(
        "ladder_tier_cap_cumulative_roll",
        over      = ci_cum_finite,
        sense     = "<=",
        lhs_terms = {"trade": Sum(Where(b.v_trade * _commodity_unitsize_param(d), cndi_cum_finite),
                                  over=("n", "d"))},
        rhs_terms = {"cap": Param(("c", "i"), cap_df)},
    )


# ---------------------------------------------------------------------------
# Objective


def _branch_weight(d) -> Param | None:
    """bw[d]: the stochastic branch weight of each ladder period; None when there are no branch weights.

    pdt_branch_weight (the weight of every other operational cost, in b.op_factor) reduced to one value
    per period.  The tiers are bought per period, so the weight must be the same for every t of a period.
    It is: reveals happen only at the first step of a period (create_stochastic_periods rejects anything
    else), and every branch period copies its anchor's full active time, so all members of a sibling
    cohort share the same t set and the weight denominator is the same for every t.  A change that breaks
    this makes a period's weight vary over t, and the build raises instead of guessing a weight.  Ladder
    periods without dt rows (v_trade is 0 there) get 1.0.
    """
    pdt_bw = getattr(d, "pdt_branch_weight", None)
    if pdt_bw is None:
        return None
    periods = d.cnd_ladder.select("d").unique()
    per_d = (pdt_bw.frame.group_by("d")
             .agg(pl.col("value").min().alias("lo"), pl.col("value").max().alias("hi"))
             .with_columns(pl.col("d").cast(periods.schema["d"], strict=False)))
    if per_d["d"].null_count() > 0:
        raise ValueError("_commodity_ladder: a period of pdt_branch_weight is not in the ladder's period "
                         "vocabulary.")
    varying = per_d.filter(pl.col("lo") != pl.col("hi")).sort("d")["d"].to_list()
    if varying:
        raise ValueError(f"_commodity_ladder: the stochastic branch weight varies within period(s) {varying}; "
                         "a ladder purchase is per period and needs one weight per period.")
    frame = (periods.join(per_d.select("d", value=pl.col("lo")), on="d", how="left")
             .with_columns(pl.col("value").fill_null(1.0))
             .sort("d"))
    return Param(("d",), frame)


def objective_terms(b: _Build) -> Expr | None:
    """Price of the ladder purchases."""
    if b.v_trade is None:
        return None
    d = b.d

    # ─── OBJ. ─── commodity ladder ───────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # + Σ_{(c,n,d,i) ∈ cndi_ladder_cum} v_trade[c,n,d,i] · cu[c] · price_cum[c,i]
    #                                   · inflation_op[d] / period_share[d] · bw[d]
    # + Σ_{(c,n,d,i) ∈ cndi_ladder_ann} v_trade[c,n,d,i] · cu[c] · price_ann[c,i,d]
    #                                   · inflation_op[d] / period_share[d] · bw[d]
    #   cu = commodity_unitsize;  bw[d] = branch_weight[d,t], the same for every t of d (factor 1 outside
    #   stochastic solves)
    # Each tier's purchase at its price, annualised per period and weighted by the probability of the
    # period's branch, as w_op weights the other operational costs.  v_trade is already a period sum, so
    # no step duration or timestep weight enters; a period whose branch weight varies over t raises.
    traded = b.v_trade * _commodity_unitsize_param(d)
    annualization = d.p_inflation_op / d.p_period_share
    bw = _branch_weight(d)
    if bw is not None:
        annualization = annualization * bw
    obj = None
    if _has_rows(d.cndi_ladder_cum) and d.p_ladder_cum_price is not None:
        obj = Sum(Where(traded, d.cndi_ladder_cum) * d.p_ladder_cum_price * annualization)
    if _has_rows(d.cndi_ladder_ann) and d.p_ladder_ann_price is not None:
        term = Sum(Where(traded, d.cndi_ladder_ann) * d.p_ladder_ann_price * annualization)
        obj = term if obj is None else obj + term
    return obj
