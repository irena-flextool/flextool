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

from flextool.engine.model._common import _has_rows, _period_branch_pairs, branch_periods_inherit
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
        p_ladder_cum_weight=None,
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

    # p_ladder_cum_weight[d]: years_represented[d] * realized_hours_through_this_roll[d] / 8760 -- the
    # real-calendar-years-equivalent of a period's dispatch simulated so far, used by the cumulative tier
    # cap to annualise + replicate-across-years its quantity (see _eq_tier_cap_cumulative's header).
    # ``p_period_share`` / ``p_years_represented_d`` (the model-wide, dt-scoped Params the annual cap
    # uses) are both pruned to the current roll's own dt once a period leaves period_in_use, which is
    # exactly wrong for a cap that must keep every period the chain has ever touched (the same reason
    # f_d_k itself is read from a provider CSV that folds finished periods back in, #143). So this reads
    # two sources that are NOT dt-pruned: ``solve_data/p_years_represented.csv`` (the solve's static
    # period->year map, identical on every roll) for years_represented, and the realized-hours handoff
    # (``cum_sim_hours``, already retained across rolls by #143's own fix) + this roll's own
    # ``hours_in_period.csv`` for realized_hours_through_this_roll. realized_hours[d] is 0 for an
    # untouched period, so including the whole years_represented domain (even periods not yet reached)
    # is harmless. None when the inputs are absent (e.g. a synthetic fixture that skips the derived-param
    # pass) -- the cap row then falls back to the plain Sigma f_d_k of the pre-fix formula.
    # A period label of a stochastic fan has no dispatch (hours_in_period 0): its weight is the realized hours
    # of earlier solves only, and every branch path (d_leaf) holds the label with its own branch periods, so a
    # path counts the realized history once and this solve's hours once.
    p_cum_weight = None
    pyr_raw = _provider_get(provider, "solve_data/p_years_represented.csv")
    if pyr_raw is not None and {"period", "p_years_represented"}.issubset(pyr_raw.columns):
        years_by_d = (pyr_raw.select("period", "p_years_represented")
                      .rename({"period": "d"})
                      .with_columns(pl.col("p_years_represented").cast(pl.Float64))
                      .group_by("d").agg(pl.col("p_years_represented").sum().alias("years")))
        prior_hours: dict[str, float] = {}
        prior = read_handoff_frame(provider, K.HANDOFF_CUM_SIM_HOURS)
        if prior is not None and "period" in prior.columns:
            col = "p_ladder_cum_sim_hours" if "p_ladder_cum_sim_hours" in prior.columns else "value"
            prior_hours = dict(zip(prior["period"].to_list(),
                                   prior[col].cast(pl.Float64).to_list()))
        this_roll_hours: dict[str, float] = {}
        hip = _provider_get(provider, "solve_data/hours_in_period.csv")
        if hip is not None and {"period", "value"}.issubset(hip.columns):
            this_roll_hours = dict(zip(hip["period"].to_list(),
                                       hip["value"].cast(pl.Float64).to_list()))
        all_d = set(years_by_d["d"].to_list()) | set(prior_hours) | set(this_roll_hours)
        if all_d:
            from flextool.engine.input._constants import HOURS_PER_YEAR
            years_lookup = dict(zip(years_by_d["d"].to_list(), years_by_d["years"].to_list()))
            rows = [
                {"d": dd, "value": years_lookup.get(dd, 1.0)
                          * (prior_hours.get(dd, 0.0) + this_roll_hours.get(dd, 0.0)) / HOURS_PER_YEAR}
                for dd in all_d
            ]
            p_cum_weight = Param(("d",), pl.DataFrame(rows))

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
        p_ladder_cum_weight=p_cum_weight,
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
    #   + Σ_{(p,n,c) ∈ startup_from_commodity, t} v_startup[p,d,t] · startup_energy[p,n,d] · unitsize[p]
    #       · timestep_weight[d,t]   (start-up energy, MWh per event: no step_duration; core._startup_draw)
    #   ∀ (c,n,d) ∈ cnd_ladder;   w = step_duration · timestep_weight
    # The period's net purchase of commodity c at node n (buy − sell, MWh) is split into price tiers.
    # The efficiency-arc term uses the shared fuel measure _eff_fuel_input (core.py), so min_load_efficiency
    # units' part-load section fuel counts toward the tiers too, same as commodity_cost_eff / co2_price_eff
    # / the CO2 caps (#56) — before this fix a ladder commodity bought by such a unit was under-counted by
    # the part-load fuel.  A term is absent when its flow set is empty or there are no processes.
    from flextool.engine.model.core import _eff_fuel_input, _startup_draw

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
    if b.has_startup_draw and _has_rows(getattr(d, "startup_from_commodity", None)):
        ladder_su = d.startup_from_commodity.filter(pl.col("c").is_in(d.commodity_with_ladder["c"].implode()))
        for key, term in _startup_draw(b, ladder_su):
            rhs[f"buy_startup_{key}"] = Sum(term * d.p_timestep_weight, over=("p", "t"))
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
    # Σ_n v_trade[c,n,d,i] · commodity_unitsize[c]
    #   ≤  quantity_ann[c,i,d] · period_share[d] · f_d_k[d] − realized_ann[c,i,α(d)]
    #   ∀ (c,i,d) with (c,n,d,i) ∈ cndi_ladder_ann and quantity_ann[c,i,d] < 1e29
    #   f_d_k[d] = (realized hours of α(d) from earlier solves + this solve's hours of d) / complete hours of α(d)
    #   α(d) = the period label of a forecast branch period d, d itself otherwise; period_share, f_d_k and the
    #   realized carry of a forecast come from α(d) unless it has its own row: every forecast continues from
    #   its period's realized history (the realization solve's carry)
    # quantity_ann is a per-YEAR amount (GMPL "per-year MWh"): period_share[d] scales it down to this
    # period's own share of the modelled year, matching the price term's inflation_op/period_share
    # annualisation (#89) — a 48h solve of a 48h period gets 48h/8760 of the annual tier, not the whole
    # year's worth.  f_d_k[d] (1 outside rolling) further scales it to the share of the period's own
    # hours reached by the end of this roll's horizon (f_d_k already measures exactly that — see
    # p_f_d_k_from_source) — the quantity allocated so far, as of this roll.  realized_ann[c,i,d] (the
    # same per-(c,i,d) carried-purchases Param the cumulative ladder uses, 0 when absent) is what
    # earlier rolls already bought against that allocation, so the remaining budget can never go
    # negative: an earlier roll could buy at most up to its own (earlier) horizon end and realized
    # only part of that (#143's fix applies here too — no clamp needed, the row is feasible by
    # construction instead of approximately-compliant). Unlimited tiers (quantity ≥ 1e29) get no row.
    # Branch periods of a stochastic solve use their anchor period's quantity (and price) unless they
    # have their own (inherit_branch_periods).  share, f_d_k and the realized carry of a forecast come from
    # α(d) unless it has its own row (branch_periods_inherit): every forecast continues from its period's
    # realized history (the realization solve's carry); without it a null share/fdk would clamp the
    # tier to 0, and a realized 0 would re-open the period's already-spent budget (#59).
    share_frame = branch_periods_inherit(d, d.p_period_share.frame, ())
    fdk_frame = branch_periods_inherit(d, d.p_f_d_k.frame, ()) if d.p_f_d_k is not None else None
    realized_frame = (branch_periods_inherit(d, d.p_ladder_cum_realized_mwh.frame, ("c", "i"))
                      if d.p_ladder_cum_realized_mwh is not None else None)
    keys = cndi_ann_finite.select("c", "i", "d").unique()
    cap_df = (keys.join(d.p_ladder_ann_quantity.frame.rename({"value": "quantity"}), on=["c", "i", "d"], how="left")
              .join(share_frame.rename({"value": "share"}), on="d", how="left"))
    if fdk_frame is not None:
        cap_df = (cap_df.join(fdk_frame.rename({"value": "fdk"}), on="d", how="left")
                  .with_columns(pl.col("fdk").fill_null(1.0)))
    else:
        cap_df = cap_df.with_columns(fdk=pl.lit(1.0))
    cap_df = cap_df.with_columns(cap=pl.col("quantity") * pl.col("share") * pl.col("fdk"))
    if realized_frame is not None:
        cap_df = (cap_df.join(realized_frame.rename({"value": "realized"}),
                              on=["c", "i", "d"], how="left")
                  .with_columns(pl.col("realized").fill_null(0.0)))
    else:
        cap_df = cap_df.with_columns(realized=pl.lit(0.0))
    cap_df = cap_df.with_columns(value=pl.col("cap") - pl.col("realized")).select("c", "i", "d", "value")
    b.m.add_cstr(
        "ladder_tier_cap_annual_roll",
        over      = keys,
        sense     = "<=",
        lhs_terms = {"trade": Sum(Where(b.v_trade * _commodity_unitsize_param(d), cndi_ann_finite), over=("n",))},
        rhs_terms = {"cap": Param(("c", "i", "d"), cap_df)},
    )


def _eq_tier_cap_cumulative(b: _Build) -> None:
    """ladder_tier_cap_cumulative_roll[_path]: purchase per tier over the horizon up to the
    tier's quantity, one row per scenario path under recourse."""
    d = b.d
    if (not _has_rows(d.ci_ladder_cumulative) or d.p_ladder_cum_quantity is None
            or not _has_rows(d.cndi_ladder_cum)):
        return
    finite = d.p_ladder_cum_quantity.frame.filter(pl.col("value") < _INF_SENTINEL).select("c", "i")
    ci_cum_finite = d.ci_ladder_cumulative.join(finite, on=["c", "i"], how="inner")
    if ci_cum_finite.height == 0:
        return
    cndi_cum_finite = d.cndi_ladder_cum.join(ci_cum_finite, on=["c", "i"], how="inner")

    # Q16 (#58): with several scenario leaves (recourse), ``quantity_cum · Σ_d weight[d]`` summed
    # the weight of every forecast branch into ONE shared row, letting the branches buy N× the tier
    # between them.  one_leaf gates on whether v_trade itself spans >1 leaf (see below); a
    # deterministic / single-leaf solve keeps the legacy scalar shape, byte-identical.
    d_leaf = getattr(d, "d_leaf", None)
    leaf_pairs = None
    if d_leaf is not None and d_leaf.height > 0:
        d_dtype = b.v_trade.frame.schema["d"]
        leaves_all = (d_leaf if d_leaf.schema["d"] == d_dtype
                      else d_leaf.with_columns(pl.col("d").cast(d_dtype, strict=False)))
        leaves = leaves_all.join(b.v_trade.frame.select("d").unique(), on="d", how="semi")
        if leaves.height > 0 and leaves["leaf"].n_unique() > 1:
            leaf_pairs = leaves_all
    one_leaf = leaf_pairs is None

    # weight_frame: the per-d years-equivalent weight the legacy scalar summed whole (see the EQ.
    # header); None only in the degenerate fixture that skips the derived-param pass (flat 1.0 then).
    # The f_d_k fallback leaves the fan labels of this solve out: a label has no v_trade, and each forecast's
    # f_d_k already holds its period's realized hours, so a path counts them once.  A label outside this
    # solve's period_in_use (a fan an earlier solve finished) keeps its f_d_k: it is realized history.
    in_use = (d.period_in_use_set["d"].cast(pl.Utf8).unique()
              if _has_rows(getattr(d, "period_in_use_set", None)) else None)
    if d.p_ladder_cum_weight is not None and d.p_ladder_cum_weight.frame.height > 0:
        weight_frame = d.p_ladder_cum_weight.frame
    elif d.p_f_d_k is not None:
        weight_frame = d.p_f_d_k.frame
        pairs = _period_branch_pairs(d)
        if pairs is not None:
            labels = pairs["anchor"].unique()
            if in_use is not None:
                labels = labels.filter(labels.is_in(in_use.implode()))
            weight_frame = weight_frame.filter(~pl.col("d").cast(pl.Utf8).is_in(labels.implode()))
    else:
        weight_frame = None

    if d.p_ladder_cum_realized_mwh is not None:
        realized_ci = (d.p_ladder_cum_realized_mwh.frame
                       .group_by(["c", "i"]).agg(pl.col("value").sum().alias("realized")))
    else:
        realized_ci = None

    if one_leaf:
        # ─── EQ. ─── ladder_tier_cap_cumulative_roll ─────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_{n,d} v_trade[c,n,d,i] · commodity_unitsize[c]
        #   ≤  quantity_cum[c,i] · Σ_d (years_represented[d] · realized_hours[d] / 8760) − Σ_d realized[c,i,d]
        #   ∀ (c,i) ∈ ci_ladder_cumulative with quantity_cum[c,i] < 1e29;  (c,n,d,i) ∈ cndi_ladder_cum
        # Caps what tier i supplies over the whole modelled horizon.  quantity_cum, like quantity_ann, is a
        # per-modelled-YEAR amount: Σ_d (years_represented[d] · realized_hours[d] / 8760) is the real-
        # calendar-years-equivalent of representative dispatch simulated so far — realized_hours[d]/8760
        # annualises each period's own realized representative-sample hours up to a full calendar year
        # (the hours-form of period_share[d]·f_d_k[d], #89's annualisation), and years_represented[d] then
        # multiplies that by the number of real calendar years period d stands for (co2_max_total scales its
        # emissions cap the same way, docs: "each period's annualised emissions times the years it
        # represents"); without it a period block representing several years was charged for only one
        # (p_ladder_cum_weight, load_data).  Sums over every period the chain has ever touched — not just
        # the ones still active this roll — the same reason f_d_k itself is read from a provider CSV that
        # folds finished periods back in (#143): without that, this sum would reset to the new period's own
        # local progress each time the chain moved on, while realized kept climbing, eventually driving this
        # row negative.  "quantity allocated up to the end of this roll's horizon" is monotone non-
        # decreasing across rolls — the same requirement #89's fix gives the annual cap — and realized (what
        # earlier rolls actually bought, always ≤ what their own, earlier horizon end allocated) can never
        # exceed it, so the row is feasible by construction; no clamp.  Without p_ladder_cum_weight (load_data
        # couldn't build the years_represented/realized_hours Param — e.g. a synthetic fixture that skips the
        # derived-param pass) the weight is Σ f_d_k over the path's periods, fan labels left out (a member's
        # f_d_k holds its period's realized hours), the pre-years_represented-fix formula; realized = tier
        # MWh bought by earlier solves (0 when absent).  Unlimited tiers get no row.
        sum_weight = float(weight_frame["value"].sum()) if weight_frame is not None else 1.0
        cap_df = (ci_cum_finite.join(d.p_ladder_cum_quantity.frame, on=["c", "i"], how="left")
                  .select("c", "i", cap=pl.col("value") * sum_weight))
        if realized_ci is not None:
            cap_df = cap_df.join(realized_ci, on=["c", "i"], how="left").with_columns(
                pl.col("realized").fill_null(0.0))
        else:
            cap_df = cap_df.with_columns(realized=pl.lit(0.0))
        cap_df = cap_df.with_columns(value=pl.col("cap") - pl.col("realized")).select("c", "i", "value")
        b.m.add_cstr(
            "ladder_tier_cap_cumulative_roll",
            over      = ci_cum_finite,
            sense     = "<=",
            lhs_terms = {"trade": Sum(Where(b.v_trade * _commodity_unitsize_param(d), cndi_cum_finite),
                                      over=("n", "d"))},
            rhs_terms = {"cap": Param(("c", "i"), cap_df)},
        )
        return

    # ─── EQ. ─── ladder_tier_cap_cumulative_roll_path ─────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{n, d∈D(leaf)} v_trade[c,n,d,i] · commodity_unitsize[c]
    #   ≤ quantity_cum[c,i] · ( Σ_{d∈D(leaf)} w[d] + Σ_{d∉period_in_use} w[d] ) − Σ_d realized[c,i,d]
    #   ∀ (c,i) ∈ ci_ladder_cumulative with quantity_cum[c,i] < 1e29, leaf ∈ d_leaf, when v_trade spans >1 leaf
    # One row per scenario path (#58): the pooled row's weight summed every forecast into one budget,
    # letting the forecasts buy N× the tier between them.  w[d] = years_represented[d]·realized_hours[d]/8760
    # (p_ladder_cum_weight; fallback f_d_k[d] with this solve's fan labels left out).  D(leaf) = the leaf's
    # branch periods, its fan labels and the shared pre-reveal periods (the (d, leaf) join of d_leaf, as in the
    # invest/divest _path families' leaf_rows); a shared period's weight counts once on each path.  The second
    # sum is the periods earlier solves and rolls finished: one realized past shared by every path, matching
    # realized, which is the chain-wide carry of what that past bought.  Every period of period_in_use with a
    # weight lies on some leaf (raises otherwise), so the second sum holds only the finished history; it is added
    # only when non-zero.  A chained earlier solve's period missing from this solve's p_years_represented.csv
    # counts with years_represented 1 (as in the pooled row).  No branch weight on the row: constraints are per
    # path, the branch weights belong to the objective.
    cndi_leaf = cndi_cum_finite.join(leaf_pairs, on="d", how="inner")
    leaves_list = leaf_pairs.select("leaf").unique()
    over = ci_cum_finite.join(leaves_list, how="cross")
    if weight_frame is not None:
        per_leaf_weight = (weight_frame.join(leaf_pairs, on="d", how="inner")
                           .group_by("leaf").agg(pl.col("value").sum().alias("weight")))
        if in_use is not None:
            weights_utf8 = weight_frame.select(pl.col("d").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
            on_leaf = d_leaf["d"].cast(pl.Utf8).unique()
            off_path = (weights_utf8.filter(pl.col("d").is_in(in_use.implode())
                                            & ~pl.col("d").is_in(on_leaf.implode())
                                            & (pl.col("value") != 0.0))["d"].unique().sort().to_list())
            if off_path:
                raise ValueError(
                    f"_commodity_ladder: period(s) {off_path} of period_in_use carry a cumulative ladder "
                    "weight but lie on no scenario path (d_leaf); the lineage frames are incomplete.")
            w_hist = float(weights_utf8.filter(~pl.col("d").is_in(in_use.implode()))["value"].sum())
            if w_hist != 0.0:
                per_leaf_weight = per_leaf_weight.with_columns(pl.col("weight") + w_hist)
    else:
        per_leaf_weight = leaves_list.with_columns(weight=pl.lit(1.0))
    cap_df = (ci_cum_finite.join(d.p_ladder_cum_quantity.frame, on=["c", "i"], how="left")
              .join(per_leaf_weight, how="cross")
              .with_columns(cap=pl.col("value") * pl.col("weight")))
    if realized_ci is not None:
        cap_df = cap_df.join(realized_ci, on=["c", "i"], how="left").with_columns(
            pl.col("realized").fill_null(0.0))
    else:
        cap_df = cap_df.with_columns(realized=pl.lit(0.0))
    cap_df = cap_df.with_columns(value=pl.col("cap") - pl.col("realized")).select("c", "i", "leaf", "value")
    b.m.add_cstr(
        "ladder_tier_cap_cumulative_roll_path",
        over      = over,
        sense     = "<=",
        lhs_terms = {"trade": Sum(Where(b.v_trade * _commodity_unitsize_param(d), cndi_leaf),
                                  over=("n", "d"))},
        rhs_terms = {"cap": Param(("c", "i", "leaf"), cap_df)},
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
    # names: commodity_ladder_cost
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
