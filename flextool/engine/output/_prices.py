"""Prices from balance-row duals: currency per physical unit.

One definition for every dual-based per-(d,t) price: the node prices (``price_t``, written by
``process_outputs.read_highs_solution.write_v_dual_node_balance``), the reserve prices (``reserve_price_t``,
``write_v_dual_reserve_balance``) and the nested ``fix_price`` handoff (``preprocess.flex_data.
build_handoff_from_solution``).

Every operational cost term of the objective carries ``w_op[d,t] = Δt · tw · inflation_op / period_share · bw``
(``model/core.py``, header above ``_obj_balance_slack``) and the whole objective is multiplied by
``scale_the_objective`` s.  The LP rows are written as built: Layer-2 row factors are removed by
``autoscale.unscale_solution`` before the duals are read, and the Layer-3 HiGHS options are un-scaled by HiGHS.

  # ─── PRICE. ─── price_t (v_dual_node_balance) ─────────────────────────────
  # price[n,d,t] = σ_row · dual[row(n,d,t)] / s / w[d,t]
  #   w[d,t] = timestep_weight[d,t] · inflation_op[d] / period_share[d] · branch_weight[d,t]
  #   row(n,d,t) = nodeBalance_eq[n,d,t]                      σ = +1
  #              | nodeBalanceCoarse_eq[n,d,b], t ∈ b         σ = +1   (node_coarse_step)
  #              | nodeBalanceBlock_eq[n,d,b],  t ∈ b         σ = +1   (period_block_time)
  #   s = scale_the_objective (live, per solve);   price = null where w = 0
  # The cost of one more MWh of demand at (n,d,t), in CUR/MWh in the price base of the cost inputs (inflation and
  # the years the period represents removed), not weighted by the timestep weight or by the branch probability.
  # A node balance row is in MWh per step, so its dual is already per MWh (no Δt).

  # ─── PRICE. ─── reserve_price_t ───────────────────────────────────────────
  # reserve_price[r,ud,g,d,t] = dual[reserveBalance_timeseries_eq | reserveBalance_dynamic_eq (dynamic_only)]
  #                             / s / (Δt[d,t] · w[d,t])
  # CUR per MW held for one hour (= CUR/MWh): the row is in MW per step and its shortfall costs penalty · w_op.

Absent weight components (no timestep weights, no branches, …) count as 1.  Polars only; ``FlexData`` is a type
hint only (``preprocess`` imports this module).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from flextool.engine.preprocess.flex_data import FlexData

# Orientation of each node balance family relative to "one more MWh of demand raises the objective".
ROW_SIGN: dict[str, float] = {
    "nodeBalance_eq": 1.0,
    "nodeBalanceCoarse_eq": 1.0,
    "nodeBalanceBlock_eq": 1.0,
}


def _frame(param) -> pl.DataFrame | None:
    """The ``(…, value)`` frame of a Param (or a frame), axis columns as Utf8; None when absent or empty."""
    if param is None:
        return None
    frame = getattr(param, "frame", param)
    if not isinstance(frame, pl.DataFrame) or frame.height == 0:
        return None
    keys = [c for c in ("d", "t") if c in frame.columns]
    return frame.select([pl.col(c).cast(pl.Utf8) for c in keys] + [pl.col("value").cast(pl.Float64)])


def price_weight_from_components(*, timestep_weight=None, inflation_op=None, period_share=None,
                                 branch_weight=None, step_duration=None,
                                 domain: pl.DataFrame | None = None) -> pl.DataFrame:
    """``(d, t, w)``: w = tw[d,t] · inflation_op[d] / period_share[d] · bw[d,t] (· Δt[d,t] when ``step_duration``
    is given).  Per-step components are ``(d, t, value)``, per-period ones ``(d, value)`` (Params or frames).  The
    (d, t) domain is ``domain`` (the solve's dt) united with the per-step components; a component missing at a
    cell counts as 1.  Without any (d, t) source the result is ``(d, w)``."""
    per_step = [(name, _frame(f)) for name, f in (("tw", timestep_weight), ("bw", branch_weight),
                                                  ("dur", step_duration))]
    per_step = [(name, f) for name, f in per_step if f is not None]
    per_period = [(name, _frame(f)) for name, f in (("infl", inflation_op), ("share", period_share))]
    per_period = [(name, f) for name, f in per_period if f is not None]
    dt_frames = [f.select("d", "t") for _n, f in per_step]
    if domain is not None and domain.height > 0:
        dt_frames.append(domain.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)))
    if dt_frames:
        domain = pl.concat(dt_frames, how="vertical").unique()
        keys = ["d", "t"]
    else:
        frames = [f.select("d") for _n, f in per_period]
        domain = (pl.concat(frames, how="vertical").unique() if frames
                  else pl.DataFrame(schema={"d": pl.Utf8}))
        keys = ["d"]
    out = domain
    for name, f in per_step:
        out = out.join(f.rename({"value": name}), on=["d", "t"], how="left")
    for name, f in per_period:
        out = out.join(f.select("d", pl.col("value").alias(name)), on="d", how="left")
    w = pl.lit(1.0)
    for name, _f in per_step + per_period:
        col = pl.col(name).fill_null(1.0)
        w = w / col if name == "share" else w * col
    return out.select(*keys, w.alias("w"))


def price_weight_dt(fd: "FlexData", *, per_hour: bool = False) -> pl.DataFrame:
    """The objective weight of one unit of a per-(d,t) balance quantity: ``b.op_factor`` without Δt
    (``per_hour=False``, rows in MWh per step) or with it (``per_hour=True``, rows in MW, e.g. reserve)."""
    return price_weight_from_components(
        timestep_weight=getattr(fd, "p_timestep_weight", None),
        inflation_op=getattr(fd, "p_inflation_op", None),
        period_share=getattr(fd, "p_period_share", None),
        branch_weight=getattr(fd, "pdt_branch_weight", None),
        step_duration=getattr(fd, "p_step_duration", None) if per_hour else None,
        domain=getattr(fd, "dt", None),
    )


def to_price(duals: pl.DataFrame, weight: pl.DataFrame, *, sign: float, inv_scale: float) -> pl.DataFrame:
    """``duals`` (…, d, t, dual) → (…, d, t, price): price = sign · dual · inv_scale / w (``inv_scale`` = 1/s).
    A cell the weight frame does not cover counts as w = 1; w = 0 (a step the objective does not weigh, e.g. a
    zero-weight stochastic trunk period) gives a null price."""
    keys = [c for c in ("d", "t") if c in weight.columns]
    tmp = duals.with_columns(pl.col(c).cast(pl.Utf8).alias(f"__{c}") for c in keys)
    tmp = tmp.join(weight.rename({c: f"__{c}" for c in keys}), on=[f"__{c}" for c in keys], how="left")
    w = pl.col("w").fill_null(1.0)
    price = pl.when(w == 0.0).then(None).otherwise(pl.lit(sign) * pl.col("dual") * pl.lit(inv_scale) / w)
    return tmp.with_columns(price.alias("price")).drop([f"__{c}" for c in keys] + ["w", "dual"])


def node_prices(duals: dict[str, pl.DataFrame | None], fd: "FlexData", *, inv_scale: float,
                weight: pl.DataFrame | None = None) -> pl.DataFrame:
    """Node prices ``(n, d, t, price)`` from the raw duals of the node balance families.

    ``duals["nodeBalance_eq"]`` is ``(n, d, t, dual)``; ``duals["nodeBalanceCoarse_eq"]`` and
    ``duals["nodeBalanceBlock_eq"]`` are ``(n, d, b_first, dual)``, broadcast to their fine steps through
    ``fd.node_coarse_step`` / ``fd.period_block_time`` before the division (tw and bw are per fine step).  The
    price of a coarse step / block is then the same at each of its steps whenever its weight is."""
    if weight is None:
        weight = price_weight_dt(fd)
    parts: list[pl.DataFrame] = []
    for family, raw in duals.items():
        if raw is None or raw.height == 0:
            continue
        if family == "nodeBalance_eq":
            long = raw.select(*(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")), pl.col("dual").cast(pl.Float64))
        else:
            if family == "nodeBalanceCoarse_eq":
                steps, on = getattr(fd, "node_coarse_step", None), ["n", "d", "b_first"]
            else:
                steps, on = getattr(fd, "period_block_time", None), ["d", "b_first"]
            if steps is None or steps.height == 0:
                continue
            steps = steps.select(pl.col(c).cast(pl.Utf8) for c in on + ["t"])
            long = (raw.select(*(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "b_first")),
                               pl.col("dual").cast(pl.Float64))
                    .join(steps, on=on, how="inner")
                    .select("n", "d", "t", "dual"))
        parts.append(to_price(long, weight, sign=ROW_SIGN[family], inv_scale=inv_scale))
    if not parts:
        return pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "price": pl.Float64})
    return pl.concat(parts, how="vertical")
