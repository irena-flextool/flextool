"""Commodity price ladder — tier-based stepwise pricing.

Mirrors the .mod's ``commodity_with_ladder`` family of constraints
(flextool.mod:3641-3745) and the per-tier objective term
(flextool.mod:2017-2028).

Concept
-------
For commodities listed in ``commodity_with_ladder`` (price_method =
``price_ladder_annual`` / ``price_ladder_cumulative``), the legacy
single-price commodity term is replaced by:

  * ``v_trade[c, n, d, i] >= 0``: per-tier MWh purchased.
  * **Balance**: Σ_i v_trade[c,n,d,i] · unitsize == period-aggregate
    commodity flow through (c, n) (the same buy − sell expression that
    the legacy ``pdtCommodity`` price term applies).
  * **Per-tier cap** (annual):
        Σ_n v_trade[c,n,d,i] · unitsize <= p_ladder_ann_quantity[c,i,d]
        · f_d_k[d] − p_ladder_cum_realized_mwh[c,i,d]
    (only when quantity < 1e29 sentinel).
  * **Per-tier cap** (cumulative):
        Σ_{n,d} v_trade[c,n,d,i] · unitsize
        <= p_ladder_cum_quantity[c,i] · Σ_d f_d_k[d]
           − Σ_d p_ladder_cum_realized_mwh[c,i,d]
    (one constraint per (c, i) ∈ ci_ladder_cumulative).
  * **Objective term**:
        + Σ price[c,i] · v_trade[c,n,d,i] · unitsize
              · inflation_op[d] / period_share[d]

For single-solve fixtures, ``f_d_k[d] = 1.0`` for every realized
period and ``p_ladder_cum_realized_mwh = 0``, so the caps reduce to
their pre-refactor form:

  annual_single:      Σ_n v_trade · unitsize <= quantity[c,i,d]
  cumulative_single:  Σ_{n,d} v_trade · unitsize <= quantity[c,i] · |period_in_use|

The infinite-tier sentinel (1e30, written by flextool's input writer
for ``+Inf`` quantity) is filtered out — those tiers carry no cap,
matching the .mod's ``< 1e29`` predicate.

Inputs
------
* ``input/commodity_ladder_annual.csv``     (commodity, period, tier, price, quantity)
* ``input/commodity_ladder_cumulative.csv`` (commodity, tier, price, quantity)
* ``input/p_commodity_unitsize.csv``        (commodity, p_commodity_unitsize)
* ``solve_data/commodity_with_ladder.csv``  (commodity)
* ``solve_data/commodity_with_ladder_annual.csv``     (commodity)
* ``solve_data/commodity_with_ladder_cumulative.csv`` (commodity)
* ``solve_data/cnd_ladder_set.csv``      (commodity, node, period)
* ``solve_data/cndi_ladder_set.csv``     (commodity, node, period, tier)
* ``solve_data/cndi_ladder_ann_set.csv`` (commodity, node, period, tier)
* ``solve_data/cndi_ladder_cum_set.csv`` (commodity, node, period, tier)
* ``solve_data/ci_ladder_cumulative.csv`` (commodity, tier)
* ``solve_data/commodity__tier_ann.csv``  (commodity, tier)
* ``solve_data/f_d_k.csv``                (period, value)
* ``solve_data/ladder_cum_realized_mwh.csv`` (commodity, tier, period, p_ladder_cum_realized_mwh)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flexpy import Param, Sum, Where

if TYPE_CHECKING:
    from flexpy.engine import Var


# .mod uses 1e30 as +Infinity sentinel; the constraint filter is
# ``quantity < 1e29``.  Re-using the same threshold keeps fixtures and
# flextool source-of-truth aligned bit-for-bit.
_INF_SENTINEL = 1e29


# ---------------------------------------------------------------------------
# Feature detection

def has_feature(d) -> bool:
    """True iff at least one commodity uses a ladder price method."""
    cwl = getattr(d, "commodity_with_ladder", None)
    return cwl is not None and cwl.height > 0


# ---------------------------------------------------------------------------
# Data loading

def _read_single_col(path: Path, col_in: str, col_out: str) -> pl.DataFrame | None:
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.height == 0:
        return None
    if col_in in df.columns and col_in != col_out:
        df = df.rename({col_in: col_out})
    return df.select(col_out)


def _read_long_csv(path: Path, rename: dict[str, str]) -> pl.DataFrame | None:
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.height == 0:
        return None
    cols_present = {c: r for c, r in rename.items() if c in df.columns}
    if cols_present:
        df = df.rename(cols_present)
    return df


def load_data(inp_dir: str | Path, sd_dir: str | Path) -> dict:
    """Load ladder CSVs from ``input/`` and ``solve_data/``.

    Returns a dict with keys matching ``FlexData`` field names.  Values
    are ``None`` (or empty frames) when the feature is inactive (every
    fixture without ``price_method = price_ladder_*`` has header-only
    CSVs, which return None here).
    """
    inp = Path(inp_dir)
    sd = Path(sd_dir)

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

    cwl = _read_single_col(sd / "commodity_with_ladder.csv", "commodity", "c")
    if cwl is None:
        return blank

    cwla = _read_single_col(
        sd / "commodity_with_ladder_annual.csv", "commodity", "c")
    cwlc = _read_single_col(
        sd / "commodity_with_ladder_cumulative.csv", "commodity", "c")

    cnd = _read_long_csv(sd / "cnd_ladder_set.csv",
                         {"commodity": "c", "node": "n", "period": "d"})
    if cnd is not None:
        cnd = cnd.select("c", "n", "d")

    def _read_cndi(path: Path) -> pl.DataFrame | None:
        df = _read_long_csv(path, {"commodity": "c", "node": "n",
                                    "period": "d", "tier": "i"})
        if df is None:
            return None
        # Tier is integer-as-string in the CSV; keep it as string for
        # consistent join keys (Param tables also keep it as Utf8).
        return df.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "n", "d", "i")

    cndi = _read_cndi(sd / "cndi_ladder_set.csv")
    cndi_ann = _read_cndi(sd / "cndi_ladder_ann_set.csv")
    cndi_cum = _read_cndi(sd / "cndi_ladder_cum_set.csv")

    ci_cum = _read_long_csv(sd / "ci_ladder_cumulative.csv",
                             {"commodity": "c", "tier": "i"})
    if ci_cum is not None:
        ci_cum = ci_cum.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "i")

    ct_ann = _read_long_csv(sd / "commodity__tier_ann.csv",
                             {"commodity": "c", "tier": "i"})
    if ct_ann is not None:
        ct_ann = ct_ann.with_columns(pl.col("i").cast(pl.Utf8)).select("c", "i")
    # commodity__tier_cum mirrors the input CSV's (commodity, tier) projection.
    ct_cum = None
    cum_inp_path = inp / "commodity_ladder_cumulative.csv"
    if cum_inp_path.exists():
        cum_inp = pl.read_csv(cum_inp_path)
        if cum_inp.height > 0:
            ct_cum = (cum_inp.rename({"commodity": "c", "tier": "i"})
                      .with_columns(pl.col("i").cast(pl.Utf8))
                      .select("c", "i").unique())

    # ── Annual price/quantity Params (c, i, d) ─────────────────────────
    p_ann_price = None
    p_ann_quantity = None
    ann_path = inp / "commodity_ladder_annual.csv"
    if ann_path.exists():
        ann = pl.read_csv(ann_path)
        if ann.height > 0:
            ann = (ann.rename({"commodity": "c", "tier": "i", "period": "d"})
                   .with_columns(pl.col("i").cast(pl.Utf8))
                   .select("c", "i", "d", "price", "quantity"))
            # The CSV's "1e30" sentinel is written as a Float64 by polars.
            p_ann_price = Param(
                ("c", "i", "d"),
                ann.select("c", "i", "d",
                           value=pl.col("price").cast(pl.Float64)),
            )
            p_ann_quantity = Param(
                ("c", "i", "d"),
                ann.select("c", "i", "d",
                           value=pl.col("quantity").cast(pl.Float64)),
            )

    # ── Cumulative price/quantity Params (c, i) ─────────────────────────
    p_cum_price = None
    p_cum_quantity = None
    cum_path = inp / "commodity_ladder_cumulative.csv"
    if cum_path.exists():
        cum = pl.read_csv(cum_path)
        if cum.height > 0:
            cum = (cum.rename({"commodity": "c", "tier": "i"})
                   .with_columns(pl.col("i").cast(pl.Utf8))
                   .select("c", "i", "price", "quantity"))
            p_cum_price = Param(
                ("c", "i"),
                cum.select("c", "i", value=pl.col("price").cast(pl.Float64)),
            )
            p_cum_quantity = Param(
                ("c", "i"),
                cum.select("c", "i",
                           value=pl.col("quantity").cast(pl.Float64)),
            )

    # ── Commodity unitsize ───────────────────────────────────────────────
    # File header is per-fixture either ``commodity,p_commodity_unitsize``
    # or empty.  Default to 1.0 when missing/empty.
    p_unitsize = None
    cu_path = inp / "p_commodity_unitsize.csv"
    if cu_path.exists():
        cu = pl.read_csv(cu_path)
        if cu.height > 0:
            value_col = "p_commodity_unitsize" if "p_commodity_unitsize" in cu.columns else "value"
            cu = (cu.rename({"commodity": "c", value_col: "value"})
                  .select("c", "value"))
            p_unitsize = Param(("c",), cu)

    # ── f_d_k (period → fraction realized this roll) ─────────────────────
    p_f_d_k = None
    fdk_path = sd / "f_d_k.csv"
    if fdk_path.exists():
        fdk = pl.read_csv(fdk_path)
        if fdk.height > 0:
            fdk = fdk.rename({"period": "d"}).select("d", "value")
            p_f_d_k = Param(("d",), fdk)

    # ── Cumulative realized accumulator (c, i, d) ────────────────────────
    p_realized = None
    rel_path = sd / "ladder_cum_realized_mwh.csv"
    if rel_path.exists():
        rel = pl.read_csv(rel_path)
        if rel.height > 0:
            value_col = "p_ladder_cum_realized_mwh" if "p_ladder_cum_realized_mwh" in rel.columns else "value"
            rel = (rel.rename({"commodity": "c", "tier": "i", "period": "d",
                                value_col: "value"})
                   .with_columns(pl.col("i").cast(pl.Utf8))
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
        p_commodity_unitsize=p_unitsize,
        p_f_d_k=p_f_d_k,
        p_ladder_cum_realized_mwh=p_realized,
    )


# ---------------------------------------------------------------------------
# Variable + constraint emission

def add_variables(m, d) -> dict:
    """Declare ``v_trade[c, n, d, i]`` over ``cndi_ladder``.

    Index domain is the union of the cumulative-tier and annual-tier
    sets (``cndi_ladder_cum_set.csv`` ∪ ``cndi_ladder_ann_set.csv``).
    """
    if not has_feature(d):
        return {}
    cndi = getattr(d, "cndi_ladder", None)
    if cndi is None or cndi.height == 0:
        return {}
    v_trade = m.add_var("v_trade", ("c", "n", "d", "i"), cndi, lower=0.0)
    return {"v_trade": v_trade}


def _commodity_unitsize_param(d) -> "Param":
    """Return the ``p_commodity_unitsize`` Param, defaulting to 1.0 for
    every commodity in ``commodity_with_ladder``.

    The .mod declares ``p_commodity_unitsize default 1.0`` so even an
    empty CSV must yield a unit-mass coefficient.
    """
    p = getattr(d, "p_commodity_unitsize", None)
    if p is not None:
        return p
    cwl = d.commodity_with_ladder
    return Param(
        ("c",),
        cwl.select("c").unique().with_columns(value=pl.lit(1.0)),
    )


def add_constraints(
    m, d, vars: dict, *,
    v_flow=None,
    p_unitsize=None,
    p_slope=None,
    p_step_duration=None,
    p_rp_cost_weight=None,
    flow_from_commodity_eff=None,
    flow_from_commodity_noEff=None,
    flow_to_commodity=None,
) -> None:
    """Emit balance constraint + per-tier cap constraints.

    Parameters
    ----------
    v_flow : Var
        The model's process flow variable, ``v_flow[p, source, sink, d, t]``.
    p_unitsize : Param
        Process unitsize ``p_entity_unitsize[p]``.
    p_slope : Param
        Process efficiency slope ``pdtProcess_slope[p, d, t]``.
    p_step_duration, p_rp_cost_weight : Param
        Time-weight params.
    flow_from_commodity_eff / flow_from_commodity_noEff : pl.DataFrame
        Buy-side index frames ``(p, source, sink, c)``.  ``source``
        is the commodity node (n).
    flow_to_commodity : pl.DataFrame
        Sell-side index frame ``(p, source, sink, c)``.  ``sink`` is
        the commodity node (n).
    """
    if not has_feature(d):
        return
    v_trade = vars.get("v_trade")
    if v_trade is None:
        return

    p_unit_c = _commodity_unitsize_param(d)
    cnd = d.cnd_ladder
    cndi = d.cndi_ladder

    # ── 1. commodity_ladder_balance: Σ_i v_trade · unitsize == flow ────
    # LHS: Σ_i v_trade[c, n, d, i] * p_commodity_unitsize[c]
    lhs_sum = Sum(v_trade * p_unit_c, over=("i",))

    # RHS: period-aggregate commodity flow through (c, n) — sums over
    # processes and timesteps.  Mirrors the .mod's
    # commodity_ladder_balance RHS (flextool.mod:3651-3670).
    #
    # The RHS is built by taking the legacy commodity-price flow
    # expression and dropping the price coefficient.  In flexpy we have
    # three index frames:
    #   * flow_from_commodity_eff   (p, source=n, sink, c)  — buy via efficient process
    #   * flow_from_commodity_noEff (p, source=n, sink, c)  — buy via no-eff process
    #   * flow_to_commodity         (p, source, sink=n, c)  — sell into priced node
    #
    # Each contributes weighted v_flow.  Result is summed over
    # (p, source, sink, t) to give a (c, n, d) aggregate.
    rhs_terms: list = []
    weight = (p_step_duration * p_rp_cost_weight) if p_step_duration is not None else None

    if (flow_from_commodity_eff is not None
            and flow_from_commodity_eff.height > 0
            and v_flow is not None and p_unitsize is not None
            and p_slope is not None):
        # source-as-n rename so the sum-over (p, source, sink) collapses
        # into the (c, n, d) index; n equals source for buy flows.
        idx_eff = (flow_from_commodity_eff
                   .with_columns(n=pl.col("source"))
                   .select("p", "source", "sink", "c", "n"))
        term = Where(v_flow * p_unitsize * p_slope, idx_eff)
        if weight is not None:
            term = term * weight
        rhs_terms.append(("buy_eff",
                          Sum(term, over=("p", "source", "sink", "t"))))

    if (flow_from_commodity_noEff is not None
            and flow_from_commodity_noEff.height > 0
            and v_flow is not None and p_unitsize is not None):
        idx_noEff = (flow_from_commodity_noEff
                     .with_columns(n=pl.col("source"))
                     .select("p", "source", "sink", "c", "n"))
        term = Where(v_flow * p_unitsize, idx_noEff)
        if weight is not None:
            term = term * weight
        rhs_terms.append(("buy_noEff",
                          Sum(term, over=("p", "source", "sink", "t"))))

    if (flow_to_commodity is not None
            and flow_to_commodity.height > 0
            and v_flow is not None and p_unitsize is not None):
        idx_sell = (flow_to_commodity
                    .with_columns(n=pl.col("sink"))
                    .select("p", "source", "sink", "c", "n"))
        term = Where(v_flow * p_unitsize, idx_sell)
        if weight is not None:
            term = term * weight
        rhs_terms.append(("sell",
                          -Sum(term, over=("p", "source", "sink", "t"))))

    rhs_dict = dict(rhs_terms) if rhs_terms else {}

    m.add_cstr(
        "commodity_ladder_balance",
        over      = cnd,
        sense     = "==",
        lhs_terms = {"trade": lhs_sum},
        rhs_terms = rhs_dict,
    )

    # ── 2. ladder_tier_cap_annual_roll ─────────────────────────────────
    # Σ_n v_trade[c, n, d, i] · unitsize <= quantity[c, i, d] · f_d_k[d]
    #                                         − realized[c, i, d]
    # Active only when quantity < 1e29 (drop infinite-tier rows).
    cndi_ann = d.cndi_ladder_ann
    if (cndi_ann is not None and cndi_ann.height > 0
            and d.p_ladder_ann_quantity is not None):
        # Restrict cndi_ann to (c, i, d) tuples where quantity < 1e29.
        ann_q = d.p_ladder_ann_quantity.frame
        ann_q_finite = ann_q.filter(pl.col("value") < _INF_SENTINEL).select("c", "i", "d")
        cndi_ann_finite = cndi_ann.join(
            ann_q_finite, on=["c", "i", "d"], how="inner")
        if cndi_ann_finite.height > 0:
            cap_idx = cndi_ann_finite.select("c", "i", "d").unique()
            # LHS: Σ_n v_trade · unitsize over the finite-cap rows.
            lhs_cap_ann = Sum(
                Where(v_trade * p_unit_c, cndi_ann_finite),
                over=("n",),
            )
            # RHS: quantity · f_d_k − realized.  When f_d_k is missing
            # (single-solve fixtures may omit it for non-realized periods)
            # we treat it as 1.0.  The realized accumulator is
            # likewise 0 by default.
            rhs_cap_ann_terms: dict = {}
            if d.p_f_d_k is not None:
                rhs_cap_ann_terms["cap"] = (
                    d.p_ladder_ann_quantity * d.p_f_d_k
                )
            else:
                rhs_cap_ann_terms["cap"] = d.p_ladder_ann_quantity
            if d.p_ladder_cum_realized_mwh is not None:
                rhs_cap_ann_terms["realized"] = -d.p_ladder_cum_realized_mwh
            m.add_cstr(
                "ladder_tier_cap_annual_roll",
                over      = cap_idx,
                sense     = "<=",
                lhs_terms = {"trade": lhs_cap_ann},
                rhs_terms = rhs_cap_ann_terms,
            )

    # ── 3. ladder_tier_cap_cumulative_roll ─────────────────────────────
    # Σ_{n, d} v_trade[c, n, d, i] · unitsize
    #     <= quantity[c, i] · Σ_d f_d_k[d] − Σ_d realized[c, i, d]
    # One constraint per (c, i) ∈ ci_ladder_cumulative with quantity < 1e29.
    ci_cum = d.ci_ladder_cumulative
    if (ci_cum is not None and ci_cum.height > 0
            and d.p_ladder_cum_quantity is not None
            and d.cndi_ladder_cum is not None
            and d.cndi_ladder_cum.height > 0):
        cum_q = d.p_ladder_cum_quantity.frame
        cum_q_finite = cum_q.filter(pl.col("value") < _INF_SENTINEL).select("c", "i")
        ci_cum_finite = ci_cum.join(cum_q_finite, on=["c", "i"], how="inner")
        if ci_cum_finite.height > 0:
            # LHS: Σ_{n, d} v_trade · unitsize over the cumulative-tier
            # rows.  Index by (c, i).
            cndi_cum_finite = (d.cndi_ladder_cum
                .join(ci_cum_finite, on=["c", "i"], how="inner"))
            lhs_cap_cum = Sum(
                Where(v_trade * p_unit_c, cndi_cum_finite),
                over=("n", "d"),
            )
            # RHS: quantity · Σ_d f_d_k − Σ_d realized.
            # Σ_d f_d_k is a scalar per the .mod's "sum {d in
            # period_in_use} f_d_k[d]" convention.  Compute it eagerly
            # from the f_d_k Param.
            sum_f_d_k = 1.0
            if d.p_f_d_k is not None:
                sum_f_d_k = float(d.p_f_d_k.frame["value"].sum())
            rhs_cap_cum_terms: dict = {
                "cap": d.p_ladder_cum_quantity * sum_f_d_k,
            }
            if d.p_ladder_cum_realized_mwh is not None:
                # Σ_d realized — collapse the (c, i, d) Param to (c, i).
                realized_ci = (d.p_ladder_cum_realized_mwh.frame
                    .group_by(["c", "i"]).agg(pl.col("value").sum())
                    .select("c", "i", "value"))
                rhs_cap_cum_terms["realized"] = -Param(
                    ("c", "i"), realized_ci)
            m.add_cstr(
                "ladder_tier_cap_cumulative_roll",
                over      = ci_cum_finite,
                sense     = "<=",
                lhs_terms = {"trade": lhs_cap_cum},
                rhs_terms = rhs_cap_cum_terms,
            )


def add_objective_terms(m, d, vars: dict, *,
                          p_unitsize_c=None,
                          p_inflation_op=None,
                          p_period_share=None):
    """Return the ladder per-tier price contribution to the objective.

    Mirrors .mod:2017-2028:

        + Σ_{(c,n,d,i) ∈ cndi_ladder_cum} cum_price[c,i] · v_trade
              · unitsize · inflation_op[d] / period_share[d]
        + Σ_{(c,n,d,i) ∈ cndi_ladder_ann} ann_price[c,i,d] · v_trade
              · unitsize · inflation_op[d] / period_share[d]
    """
    if not has_feature(d):
        return None
    v_trade = vars.get("v_trade")
    if v_trade is None:
        return None
    if p_unitsize_c is None:
        p_unitsize_c = _commodity_unitsize_param(d)
    annualization = None
    if p_inflation_op is not None and p_period_share is not None:
        annualization = p_inflation_op / p_period_share

    obj = None

    if (d.cndi_ladder_cum is not None and d.cndi_ladder_cum.height > 0
            and d.p_ladder_cum_price is not None):
        term = Where(v_trade * p_unitsize_c, d.cndi_ladder_cum) \
               * d.p_ladder_cum_price
        if annualization is not None:
            term = term * annualization
        obj = Sum(term)

    if (d.cndi_ladder_ann is not None and d.cndi_ladder_ann.height > 0
            and d.p_ladder_ann_price is not None):
        term = Where(v_trade * p_unitsize_c, d.cndi_ladder_ann) \
               * d.p_ladder_ann_price
        if annualization is not None:
            term = term * annualization
        if obj is None:
            obj = Sum(term)
        else:
            obj = obj + Sum(term)

    return obj
