"""Closed-form helpers — one per major obj component.

Each function takes ``(data, sol)`` (where ``sol`` is a solved
``Solution``) and returns a float — that component's contribution to
the obj.  Components return ``0.0`` cleanly when their feature isn't
active in ``data``.

Mirrors ``flextool/model.py:944-1212`` (the obj-construction block)
section by section.  See ``audit/objective_audit.md`` for the
mod-side reference.

NOTE: ``pdt_branch_weight`` is treated as 1.0 universally — both the
audit and the current polar_high model.py emit it that way.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from flextool.engine_polars._axis_enums import align_join_dtypes
from flextool.engine_polars._param_shapes import promote_param_to_dt


# ---------------------------------------------------------------------------
# Shared helpers


def _op_factor_frame(d) -> pl.DataFrame:
    """Per-(d, t) operational multiplier:

        op_factor = step_duration * timestep_weight * inflation_op
                    / period_share

    Matches ``op_factor`` in ``flextool/model.py:945-946``.
    """
    return (
        d.p_step_duration.frame.rename({"value": "step_duration"})
        .join(d.p_timestep_weight.frame.rename({"value": "timestep_weight"}),
              on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "inflation_op"}),
              on="d")
        .join(d.p_period_share.frame.rename({"value": "period_share"}),
              on="d")
        .with_columns(op_factor=(pl.col("step_duration")
                                  * pl.col("timestep_weight")
                                  * pl.col("inflation_op")
                                  / pl.col("period_share")))
        .select("d", "t", "op_factor")
    )


def _startup_factor_frame(d) -> pl.DataFrame:
    """Startup-cost multiplier (no step_duration — startups are point
    events, see audit §6.1).
    """
    return (
        d.p_timestep_weight.frame.rename({"value": "timestep_weight"})
        .join(d.p_inflation_op.frame.rename({"value": "inflation_op"}),
              on="d")
        .join(d.p_period_share.frame.rename({"value": "period_share"}),
              on="d")
        .with_columns(startup_factor=(pl.col("timestep_weight")
                                       * pl.col("inflation_op")
                                       / pl.col("period_share")))
        .select("d", "t", "startup_factor")
    )


def _sol_value_or_empty(sol, var_name: str) -> Optional[pl.DataFrame]:
    """Return ``sol.value(var_name)`` if the variable exists, else None.

    polar_high's Solution stores variables in a private ``_vars`` dict
    keyed on var name; missing keys mean the feature wasn't present.
    """
    if var_name not in getattr(sol, "_vars", {}):
        return None
    return sol.value(var_name)


def _is_empty(df: Optional[pl.DataFrame]) -> bool:
    return df is None or df.height == 0


def _is_param_empty(p) -> bool:
    return p is None or p.frame.height == 0


# ---------------------------------------------------------------------------
# §1 Slack penalties


def slack_obj(data, sol) -> float:
    """§1.1 + §1.2: slack penalties on nodeBalance.

        Σ vq_state_up * penalty_up + vq_state_down * penalty_down,
        weighted by step_duration * timestep_weight * inflation_op /
        period_share, and (if loaded) by node_capacity_for_scaling[n,d].

    Mirrors ``flextool/model.py:953-960``.
    """
    vq_up = _sol_value_or_empty(sol, "vq_state_up")
    vq_dn = _sol_value_or_empty(sol, "vq_state_down")
    if _is_empty(vq_up) and _is_empty(vq_dn):
        return 0.0
    op = _op_factor_frame(data)
    pen_up = data.p_penalty_up.frame.rename({"value": "pen_up"})
    pen_dn = data.p_penalty_down.frame.rename({"value": "pen_dn"})
    df = (vq_up.rename({"value": "vq_up"})
          .join(vq_dn.rename({"value": "vq_dn"}), on=["n", "d", "t"])
          .join(pen_up, on=["n", "d", "t"])
          .join(pen_dn, on=["n", "d", "t"])
          .join(op, on=["d", "t"]))
    if data.p_node_capacity_for_scaling is not None:
        ncs = data.p_node_capacity_for_scaling.frame.rename(
            {"value": "ncs"})
        df = df.join(ncs, on=["n", "d"])
        return float(df.select(
            ((pl.col("vq_up") * pl.col("pen_up")
              + pl.col("vq_dn") * pl.col("pen_dn"))
             * pl.col("ncs") * pl.col("op_factor")).sum()).item())
    return float(df.select(
        ((pl.col("vq_up") * pl.col("pen_up")
          + pl.col("vq_dn") * pl.col("pen_dn"))
         * pl.col("op_factor")).sum()).item())


# ---------------------------------------------------------------------------
# §2 Commodity buy / sell


def commodity_buy_eff_obj(data, sol) -> float:
    """§2.2: v_flow * unitsize * slope * commodity_price * op_factor,
    indexed on flow_from_commodity_eff.

    Mirrors ``flextool/model.py:962-965``.

    NOTE: missing flow-coefficient ratio (sink/source) — currently
    treated as 1.0 (matches polar_high).  See audit §2.2.
    """
    fce = data.flow_from_commodity_eff
    if _is_empty(fce):
        return 0.0
    if (data.p_unitsize is None or data.p_slope is None
            or data.p_commodity_price is None):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    df = (fce.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink"])
            .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
            .join(data.p_slope.frame.rename({"value": "slope"}),
                  on=["p", "d", "t"])
            .join(promote_param_to_dt(data.p_commodity_price, data.dt).rename({"value": "cp"}).collect(),
                  on=["c", "d", "t"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("slope")
         * pl.col("cp") * pl.col("op_factor")).sum()).item())


def commodity_buy_noEff_obj(data, sol) -> float:
    """§2.1: v_flow * unitsize * commodity_price * op_factor,
    indexed on flow_from_commodity_noEff.

    Mirrors ``flextool/model.py:982-985``.
    """
    fcn = data.flow_from_commodity_noEff
    if _is_empty(fcn):
        return 0.0
    if data.p_unitsize is None or data.p_commodity_price is None:
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    df = (fcn.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink"])
            .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
            .join(promote_param_to_dt(data.p_commodity_price, data.dt).rename({"value": "cp"}).collect(),
                  on=["c", "d", "t"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("cp")
         * pl.col("op_factor")).sum()).item())


def commodity_sell_obj(data, sol) -> float:
    """§2.4: -Σ v_flow * unitsize * commodity_price * op_factor,
    indexed on flow_to_commodity (sink-side of priced commodity node).

    Mirrors ``flextool/model.py:989-994``.  Sign is **negative**.
    """
    ftc = data.flow_to_commodity
    if _is_empty(ftc):
        return 0.0
    if data.p_unitsize is None or data.p_commodity_price is None:
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    df = (ftc.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink"])
            .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
            .join(promote_param_to_dt(data.p_commodity_price, data.dt).rename({"value": "cp"}).collect(),
                  on=["c", "d", "t"])
            .join(op, on=["d", "t"]))
    return -float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("cp")
         * pl.col("op_factor")).sum()).item())


def commodity_section_obj(data, sol) -> float:
    """§2.3: Online section term in commodity-price envelope.

        + Σ (v_online_lin + v_online_int) * section * unitsize *
              commodity_price * op_factor,
        indexed on flow_from_commodity_eff filtered to
        process_min_load_eff.

    Mirrors ``flextool/model.py:966-981``.
    """
    fce = data.flow_from_commodity_eff
    if _is_empty(fce) or data.process_min_load_eff is None:
        return 0.0
    if (data.p_section is None or data.p_unitsize is None
            or data.p_commodity_price is None):
        return 0.0
    op = _op_factor_frame(data)
    fce_min = fce.join(data.process_min_load_eff, on="p", how="inner")
    if fce_min.height == 0:
        return 0.0
    section = data.p_section.frame.rename({"value": "section"})
    us = data.p_unitsize.frame.rename({"value": "us"})
    cp = promote_param_to_dt(data.p_commodity_price, data.dt).rename({"value": "cp"}).collect()
    total = 0.0
    for var_name in ("v_online_linear", "v_online_integer"):
        vo = _sol_value_or_empty(sol, var_name)
        if _is_empty(vo):
            continue
        df = (fce_min.join(vo.rename({"value": "online"}), on="p")
                     .join(section, on=["p", "d", "t"])
                     .join(us, on="p")
                     .join(cp, on=["c", "d", "t"])
                     .join(op, on=["d", "t"]))
        total += float(df.select(
            (pl.col("online") * pl.col("section") * pl.col("us")
             * pl.col("cp") * pl.col("op_factor")).sum()).item())
    return total


# ---------------------------------------------------------------------------
# §4 CO2 price


def co2_price_eff_obj(data, sol) -> float:
    """§4.2: source-side flow into CO2-priced node, with slope.

    Mirrors ``flextool/model.py:996-999``.
    """
    fc2 = data.flow_from_co2_priced
    if _is_empty(fc2):
        return 0.0
    if (data.p_unitsize is None or data.p_slope is None
            or data.p_co2_content is None or data.p_co2_price is None):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    df = (fc2.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink"])
            .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
            .join(data.p_slope.frame.rename({"value": "slope"}),
                  on=["p", "d", "t"])
            .join(data.p_co2_content.frame.rename({"value": "co2c"}),
                  on="c")
            .join(promote_param_to_dt(data.p_co2_price, data.dt).rename({"value": "co2p"}).collect(),
                  on=["g", "d", "t"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("slope")
         * pl.col("co2c") * pl.col("co2p") * pl.col("op_factor")
        ).sum()).item())


def co2_price_noEff_obj(data, sol) -> float:
    """§4.1: source-side flow into CO2-priced node (noEff).

    Mirrors ``flextool/model.py:1015-1020``.
    """
    fc2n = data.flow_from_co2_priced_noEff
    if _is_empty(fc2n):
        return 0.0
    if (data.p_unitsize is None or data.p_co2_content is None
            or data.p_co2_price is None):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    df = (fc2n.join(vf.rename({"value": "flow"}),
                    on=["p", "source", "sink"])
              .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
              .join(data.p_co2_content.frame.rename({"value": "co2c"}),
                    on="c")
              .join(promote_param_to_dt(data.p_co2_price, data.dt).rename({"value": "co2p"}).collect(),
                    on=["g", "d", "t"])
              .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("co2c") * pl.col("co2p")
         * pl.col("op_factor")).sum()).item())


def co2_section_obj(data, sol) -> float:
    """§4.3: Online section term in CO2-price envelope.

    Mirrors ``flextool/model.py:1000-1012``.
    """
    fc2 = data.flow_from_co2_priced
    if _is_empty(fc2) or data.process_min_load_eff is None:
        return 0.0
    if (data.p_section is None or data.p_unitsize is None
            or data.p_co2_content is None or data.p_co2_price is None):
        return 0.0
    op = _op_factor_frame(data)
    fc2_min = fc2.join(data.process_min_load_eff, on="p", how="inner")
    if fc2_min.height == 0:
        return 0.0
    section = data.p_section.frame.rename({"value": "section"})
    us = data.p_unitsize.frame.rename({"value": "us"})
    co2c = data.p_co2_content.frame.rename({"value": "co2c"})
    co2p = promote_param_to_dt(data.p_co2_price, data.dt).rename({"value": "co2p"}).collect()
    total = 0.0
    for var_name in ("v_online_linear", "v_online_integer"):
        vo = _sol_value_or_empty(sol, var_name)
        if _is_empty(vo):
            continue
        df = (fc2_min.join(vo.rename({"value": "online"}), on="p")
                     .join(section, on=["p", "d", "t"])
                     .join(us, on="p")
                     .join(co2c, on="c")
                     .join(co2p, on=["g", "d", "t"])
                     .join(op, on=["d", "t"]))
        total += float(df.select(
            (pl.col("online") * pl.col("section") * pl.col("us")
             * pl.col("co2c") * pl.col("co2p")
             * pl.col("op_factor")).sum()).item())
    return total


# ---------------------------------------------------------------------------
# §5 Process variable cost (other_operational_cost)


def varcost_noEff_obj(data, sol) -> float:
    """§5.1: pdtProcess__source__sink__dt_varCost * v_flow * unitsize *
    op_factor.

    Mirrors ``flextool/model.py:1025-1030``.
    """
    idx = data.pssdt_varCost_noEff
    if _is_empty(idx):
        return 0.0
    if data.p_unitsize is None or _is_param_empty(data.p_pssdt_varCost):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    vc = data.p_pssdt_varCost.frame.rename({"value": "vc"})
    df = (idx.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink", "d", "t"])
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(vc, on=["p", "source", "sink", "d", "t"])
             .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("vc")
         * pl.col("op_factor")).sum()).item())


def varcost_eff_source_obj(data, sol) -> float:
    """§5.2: -Σ pdtProcess_source[...,'other_operational_cost']
    * (v_flow * unitsize * slope + section term) * op_factor.

    Mirrors ``flextool/model.py:1040-1065``.  Sign is **negative**.
    """
    idx = data.pssdt_varCost_eff_unit_source
    if _is_empty(idx):
        return 0.0
    if (data.p_unitsize is None or data.p_slope is None
            or _is_param_empty(data.p_pdt_varCost_source)):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    vc = data.p_pdt_varCost_source.frame.rename({"value": "vc"})
    # flow piece (-Σ flow * us * slope * vc * op)
    df = (idx.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink", "d", "t"])
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(data.p_slope.frame.rename({"value": "slope"}),
                   on=["p", "d", "t"])
             .join(vc, on=["p", "source", "d", "t"])
             .join(op, on=["d", "t"]))
    total = -float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("slope") * pl.col("vc")
         * pl.col("op_factor")).sum()).item())
    # section piece
    if (data.process_min_load_eff is not None and data.p_section is not None):
        section_idx = idx.join(data.process_min_load_eff, on="p",
                               how="inner")
        if section_idx.height > 0:
            section = data.p_section.frame.rename({"value": "section"})
            us = data.p_unitsize.frame.rename({"value": "us"})
            for var_name in ("v_online_linear", "v_online_integer"):
                vo = _sol_value_or_empty(sol, var_name)
                if _is_empty(vo):
                    continue
                sdf = (section_idx
                       .join(vo.rename({"value": "online"}), on="p")
                       .join(section, on=["p", "d", "t"])
                       .join(us, on="p")
                       .join(vc, on=["p", "source", "d", "t"])
                       .join(op, on=["d", "t"]))
                total -= float(sdf.select(
                    (pl.col("online") * pl.col("section") * pl.col("us")
                     * pl.col("vc") * pl.col("op_factor")
                    ).sum()).item())
    return total


def varcost_eff_sink_obj(data, sol) -> float:
    """§5.3: + pdtProcess_sink[...,'other_operational_cost']
    * v_flow * unitsize * op_factor.

    Mirrors ``flextool/model.py:1070-1075``.
    """
    idx = data.pssdt_varCost_eff_unit_sink
    if _is_empty(idx):
        return 0.0
    if data.p_unitsize is None or _is_param_empty(data.p_pdt_varCost_sink):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    vc = data.p_pdt_varCost_sink.frame.rename({"value": "vc"})
    df = (idx.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink", "d", "t"])
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(vc, on=["p", "sink", "d", "t"])
             .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("vc")
         * pl.col("op_factor")).sum()).item())


def varcost_eff_connection_obj(data, sol) -> float:
    """§5.4: + pdtProcess[...,'other_operational_cost']
    * v_flow * unitsize * op_factor.

    Mirrors ``flextool/model.py:1080-1085``.
    """
    idx = data.pssdt_varCost_eff_connection
    if _is_empty(idx):
        return 0.0
    if (data.p_unitsize is None
            or _is_param_empty(data.p_pdt_varCost_process)):
        return 0.0
    vf = _sol_value_or_empty(sol, "v_flow")
    if _is_empty(vf):
        return 0.0
    op = _op_factor_frame(data)
    vc = data.p_pdt_varCost_process.frame.rename({"value": "vc"})
    df = (idx.join(vf.rename({"value": "flow"}),
                   on=["p", "source", "sink", "d", "t"])
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(vc, on=["p", "d", "t"])
             .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("flow") * pl.col("us") * pl.col("vc")
         * pl.col("op_factor")).sum()).item())


# ---------------------------------------------------------------------------
# §6 Startup costs


def startup_cost_obj(data, sol) -> float:
    """§6.1 + §6.2: v_startup * startup_cost * unitsize * startup_factor.

    Mirrors ``flextool/model.py:1090-1099``.  Note: no step_duration —
    startups are point events.
    """
    if data.p_startup_cost is None or data.p_unitsize is None:
        return 0.0
    sf = _startup_factor_frame(data)
    sc = data.p_startup_cost.frame.rename({"value": "sc"})
    us = data.p_unitsize.frame.rename({"value": "us"})
    total = 0.0
    for var_name, idx_attr in (("v_startup_linear", "pdt_online_linear"),
                                ("v_startup_integer",
                                 "pdt_online_integer")):
        vs = _sol_value_or_empty(sol, var_name)
        idx = getattr(data, idx_attr, None)
        if _is_empty(vs) or _is_empty(idx):
            continue
        df = (idx.join(vs.rename({"value": "startup"}),
                       on=["p", "d", "t"])
                 .join(sc, on=["p", "d"])
                 .join(us, on="p")
                 .join(sf, on=["d", "t"]))
        total += float(df.select(
            (pl.col("startup") * pl.col("sc") * pl.col("us")
             * pl.col("startup_factor")).sum()).item())
    return total


# Convenience alias — the proposal mentions ``section_obj`` as a single
# entry combining commodity- and CO2-side section terms.  Keep both
# split components above (they map 1:1 to obj lines in model.py); this
# alias just lets callers refer to "section" as one summand if they want.
def section_obj(data, sol) -> float:
    """Aggregate of §2.3 + §4.3 section terms (commodity + CO2 side)."""
    return commodity_section_obj(data, sol) + co2_section_obj(data, sol)


# ---------------------------------------------------------------------------
# §7 Investment / divestment


def _ed_param_for(data, attr: str, e_alias: str, ref=None):
    """Return ``data.<attr>`` re-aliased so its ``e`` column becomes
    ``e_alias`` (one of ``p`` or ``n``).

    When ``ref`` is supplied, the returned frame's ``e_alias`` and
    ``d`` columns are dtype-aligned against ``ref`` via
    :func:`align_join_dtypes`.  Bridges the cross-vocab Enum boundary
    between the entity-wide ``e`` axis on ``data.ed_*`` parameters and
    the narrower process/node-only vocab on ``sol.value("v_invest_*")``.
    """
    p = getattr(data, attr, None)
    if p is None:
        return None
    out = p.frame.rename({"e": e_alias, "value": attr})
    if ref is not None:
        _, out = align_join_dtypes(ref, out, (e_alias, "d"))
    return out


def invest_p_annuity_obj(data, sol) -> float:
    """§7.1: + Σ v_invest_p * unitsize * ed_entity_annual_discounted.

    Mirrors ``flextool/model.py:1109-1110``.
    """
    vip = _sol_value_or_empty(sol, "v_invest_p")
    if _is_empty(vip) or data.p_unitsize is None:
        return 0.0
    annu = _ed_param_for(data, "ed_entity_annual_discounted", "p", ref=vip)
    if annu is None:
        return 0.0
    df = (vip.rename({"value": "iv"})
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(annu, on=["p", "d"]))
    return float(df.select(
        (pl.col("iv") * pl.col("us")
         * pl.col("ed_entity_annual_discounted")).sum()).item())


def invest_n_annuity_obj(data, sol) -> float:
    """§7.2: + Σ v_invest_n * state_unitsize * ed_entity_annual_discounted.

    Mirrors ``flextool/model.py:1132-1133``.
    """
    vin = _sol_value_or_empty(sol, "v_invest_n")
    if _is_empty(vin) or data.p_state_unitsize is None:
        return 0.0
    annu = _ed_param_for(data, "ed_entity_annual_discounted", "n", ref=vin)
    if annu is None:
        return 0.0
    df = (vin.rename({"value": "iv"})
             .join(data.p_state_unitsize.frame.rename({"value": "us"}),
                   on="n")
             .join(annu, on=["n", "d"]))
    return float(df.select(
        (pl.col("iv") * pl.col("us")
         * pl.col("ed_entity_annual_discounted")).sum()).item())


def lifetime_fixed_cost_p_obj(data, sol) -> float:
    """§7.3: + Σ v_invest_p * unitsize * ed_lifetime_fixed_cost.

    Mirrors ``flextool/model.py:1111-1112``.
    """
    vip = _sol_value_or_empty(sol, "v_invest_p")
    if _is_empty(vip) or data.p_unitsize is None:
        return 0.0
    lf = _ed_param_for(data, "ed_lifetime_fixed_cost", "p", ref=vip)
    if lf is None:
        return 0.0
    df = (vip.rename({"value": "iv"})
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(lf, on=["p", "d"]))
    return float(df.select(
        (pl.col("iv") * pl.col("us")
         * pl.col("ed_lifetime_fixed_cost")).sum()).item())


def lifetime_fixed_cost_n_obj(data, sol) -> float:
    """§7.4: + Σ v_invest_n * state_unitsize * ed_lifetime_fixed_cost.

    Mirrors ``flextool/model.py:1134-1135``.
    """
    vin = _sol_value_or_empty(sol, "v_invest_n")
    if _is_empty(vin) or data.p_state_unitsize is None:
        return 0.0
    lf = _ed_param_for(data, "ed_lifetime_fixed_cost", "n", ref=vin)
    if lf is None:
        return 0.0
    df = (vin.rename({"value": "iv"})
             .join(data.p_state_unitsize.frame.rename({"value": "us"}),
                   on="n")
             .join(lf, on=["n", "d"]))
    return float(df.select(
        (pl.col("iv") * pl.col("us")
         * pl.col("ed_lifetime_fixed_cost")).sum()).item())


def divest_p_savings_obj(data, sol) -> float:
    """§7.5: -Σ v_divest_p * unitsize * ed_lifetime_fixed_cost_divest.

    Mirrors ``flextool/model.py:1120-1121``.  Sign is **negative**.
    """
    vdp = _sol_value_or_empty(sol, "v_divest_p")
    if _is_empty(vdp) or data.p_unitsize is None:
        return 0.0
    lfd = _ed_param_for(data, "ed_lifetime_fixed_cost_divest", "p", ref=vdp)
    if lfd is None:
        return 0.0
    df = (vdp.rename({"value": "dv"})
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(lfd, on=["p", "d"]))
    return -float(df.select(
        (pl.col("dv") * pl.col("us")
         * pl.col("ed_lifetime_fixed_cost_divest")).sum()).item())


def divest_n_savings_obj(data, sol) -> float:
    """§7.6: -Σ v_divest_n * state_unitsize * ed_lifetime_fixed_cost_divest.

    Mirrors ``flextool/model.py:1144-1145``.  Sign is **negative**.
    """
    vdn = _sol_value_or_empty(sol, "v_divest_n")
    if _is_empty(vdn) or data.p_state_unitsize is None:
        return 0.0
    lfd = _ed_param_for(data, "ed_lifetime_fixed_cost_divest", "n", ref=vdn)
    if lfd is None:
        return 0.0
    df = (vdn.rename({"value": "dv"})
             .join(data.p_state_unitsize.frame.rename({"value": "us"}),
                   on="n")
             .join(lfd, on=["n", "d"]))
    return -float(df.select(
        (pl.col("dv") * pl.col("us")
         * pl.col("ed_lifetime_fixed_cost_divest")).sum()).item())


def divest_p_salvage_obj(data, sol) -> float:
    """§7.7: -Σ v_divest_p * unitsize * ed_entity_annual_divest_discounted.

    Mirrors ``flextool/model.py:1122-1123``.  Sign is **negative**.
    """
    vdp = _sol_value_or_empty(sol, "v_divest_p")
    if _is_empty(vdp) or data.p_unitsize is None:
        return 0.0
    annd = _ed_param_for(data, "ed_entity_annual_divest_discounted", "p", ref=vdp)
    if annd is None:
        return 0.0
    df = (vdp.rename({"value": "dv"})
             .join(data.p_unitsize.frame.rename({"value": "us"}), on="p")
             .join(annd, on=["p", "d"]))
    return -float(df.select(
        (pl.col("dv") * pl.col("us")
         * pl.col("ed_entity_annual_divest_discounted")).sum()).item())


def divest_n_salvage_obj(data, sol) -> float:
    """§7.8: -Σ v_divest_n * state_unitsize *
    ed_entity_annual_divest_discounted.

    Mirrors ``flextool/model.py:1146-1147``.  Sign is **negative**.
    """
    vdn = _sol_value_or_empty(sol, "v_divest_n")
    if _is_empty(vdn) or data.p_state_unitsize is None:
        return 0.0
    annd = _ed_param_for(data, "ed_entity_annual_divest_discounted", "n", ref=vdn)
    if annd is None:
        return 0.0
    df = (vdn.rename({"value": "dv"})
             .join(data.p_state_unitsize.frame.rename({"value": "us"}),
                   on="n")
             .join(annd, on=["n", "d"]))
    return -float(df.select(
        (pl.col("dv") * pl.col("us")
         * pl.col("ed_entity_annual_divest_discounted")).sum()).item())


# ---------------------------------------------------------------------------
# §8 Existing-entity fixed cost (constant) — DEFERRED in polar_high


def existing_entity_fixed_cost_obj(data, sol) -> float:
    """§8.1: + Σ p_entity_all_existing * ed_fixed_cost * inflation_op.

    Currently NOT wired in polar_high (model.py:1149-1161 documents the
    deferral).  Computed here for completeness; tests that need to flag
    this term should compare against it explicitly.
    """
    if (data.p_ed_fixed_cost is None
            or data.p_entity_all_existing is None):
        return 0.0
    fc = data.p_ed_fixed_cost.frame.rename({"value": "fc"})
    ae = data.p_entity_all_existing.frame.rename({"value": "ae"})
    infl = data.p_inflation_op.frame.rename({"value": "infl"})
    df = fc.join(ae, on=["e", "d"]).join(infl, on="d")
    return float(df.select(
        (pl.col("fc") * pl.col("ae") * pl.col("infl")).sum()).item())


# ---------------------------------------------------------------------------
# §9 Group-level slack (capacity_margin / inertia / non_synchronous)


def vq_capacity_margin_obj(data, sol) -> float:
    """§9.3: + Σ vq_capacity_margin * group_capacity_for_scaling
    * penalty_capacity_margin * inflation_op.  Period-only (no t).
    """
    vq = _sol_value_or_empty(sol, "vq_capacity_margin")
    if _is_empty(vq):
        return 0.0
    if (data.p_group_capacity_for_scaling is None
            or data.pdGroup_penalty_capacity_margin is None):
        return 0.0
    gcs = data.p_group_capacity_for_scaling.frame.rename({"value": "gcs"})
    pen = data.pdGroup_penalty_capacity_margin.frame.rename(
        {"value": "pen"})
    infl = data.p_inflation_op.frame.rename({"value": "infl"})
    df = (vq.rename({"value": "vq"})
            .join(gcs, on=["g", "d"])
            .join(pen, on=["g", "d"])
            .join(infl, on="d"))
    return float(df.select(
        (pl.col("vq") * pl.col("gcs") * pl.col("pen") * pl.col("infl")
        ).sum()).item())


def vq_inertia_obj(data, sol) -> float:
    """§9.1: + Σ vq_inertia * inertia_limit * penalty_inertia
    * step_duration * timestep_weight * inflation_op / period_share.
    """
    vq = _sol_value_or_empty(sol, "vq_inertia")
    if _is_empty(vq):
        return 0.0
    if (data.pdGroup_inertia_limit is None
            or data.pdGroup_penalty_inertia is None):
        return 0.0
    op = _op_factor_frame(data)
    lim = data.pdGroup_inertia_limit.frame.rename({"value": "lim"})
    pen = data.pdGroup_penalty_inertia.frame.rename({"value": "pen"})
    df = (vq.rename({"value": "vq"})
            .join(lim, on=["g", "d"])
            .join(pen, on=["g", "d"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("vq") * pl.col("lim") * pl.col("pen") * pl.col("op_factor")
        ).sum()).item())


def vq_non_synchronous_obj(data, sol) -> float:
    """§9.2: + Σ vq_non_synchronous * group_capacity_for_scaling
    * penalty_non_synchronous * step_duration * timestep_weight
    * inflation_op / period_share.
    """
    vq = _sol_value_or_empty(sol, "vq_non_synchronous")
    if _is_empty(vq):
        return 0.0
    if (data.p_group_capacity_for_scaling is None
            or data.pdGroup_penalty_non_synchronous is None):
        return 0.0
    op = _op_factor_frame(data)
    gcs = data.p_group_capacity_for_scaling.frame.rename({"value": "gcs"})
    pen = data.pdGroup_penalty_non_synchronous.frame.rename(
        {"value": "pen"})
    df = (vq.rename({"value": "vq"})
            .join(gcs, on=["g", "d"])
            .join(pen, on=["g", "d"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("vq") * pl.col("gcs") * pl.col("pen") * pl.col("op_factor")
        ).sum()).item())


def vq_reserve_obj(data, sol) -> float:
    """§9.4: + Σ vq_reserve * pdtReserve_reservation
    * penalty_reserve * step_duration * timestep_weight
    * inflation_op / period_share.
    """
    vq = _sol_value_or_empty(sol, "vq_reserve")
    if _is_empty(vq):
        return 0.0
    if (data.pdtReserve_upDown_group_reservation is None
            or data.p_reserve_upDown_group_penalty_reserve is None):
        return 0.0
    op = _op_factor_frame(data)
    res = data.pdtReserve_upDown_group_reservation.frame.rename(
        {"value": "res"})
    pen = data.p_reserve_upDown_group_penalty_reserve.frame.rename(
        {"value": "pen"})
    df = (vq.rename({"value": "vq"})
            .join(res, on=["r", "ud", "g", "d", "t"])
            .join(pen, on=["r", "ud", "g"])
            .join(op, on=["d", "t"]))
    return float(df.select(
        (pl.col("vq") * pl.col("res") * pl.col("pen") * pl.col("op_factor")
        ).sum()).item())


# ---------------------------------------------------------------------------
# §10 Storage state reference price


def storage_state_reference_price_obj(data, sol) -> float:
    """§10.1: -Σ v_state(d_last, t_last) * unitsize *
    storage_state_reference_price * timestep_weight * inflation_op
    / period_share.  Sign is **negative** — terminal state is rewarded.

    Mirrors ``flextool/engine_polars/model.py`` §10.1 (B1b).  The term
    fires when ``p_storage_state_reference_price`` is populated for at
    least one (n, d), and only at the last (d, t) of every period_last.
    No ``step_duration`` (terminal valuation is a point event — mirrors
    startup_factor).
    """
    p_ref = getattr(data, "p_storage_state_reference_price", None)
    if p_ref is None:
        return 0.0
    v_state = _sol_value_or_empty(sol, "v_state")
    if (_is_empty(v_state) or data.p_state_unitsize is None
            or data.nodeState_last_dt is None
            or data.nodeState_last_dt.height == 0
            or data.period_last is None
            or data.period_last.height == 0):
        return 0.0
    # Domain: (n, d, t) ∈ nodeState_last_dt with d ∈ period_last and
    # (n, d) populated in p_storage_state_reference_price.
    over = (data.nodeState_last_dt
            .join(data.period_last, on="d", how="inner")
            .join(p_ref.frame.select("n", "d"), on=["n", "d"], how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return 0.0
    ref = p_ref.frame.rename({"value": "ref"})
    us = data.p_state_unitsize.frame.rename({"value": "us"})
    rp = data.p_timestep_weight.frame.rename({"value": "rp"})
    infl = data.p_inflation_op.frame.rename({"value": "infl"})
    ps = data.p_period_share.frame.rename({"value": "ps"})
    df = (over
          .join(v_state.rename({"value": "vs"}), on=["n", "d", "t"])
          .join(us, on="n")
          .join(ref, on=["n", "d"])
          .join(rp, on=["d", "t"])
          .join(infl, on="d")
          .join(ps, on="d"))
    return -float(df.select(
        (pl.col("vs") * pl.col("us") * pl.col("ref")
         * pl.col("rp") * pl.col("infl") / pl.col("ps")
        ).sum()).item())


# ---------------------------------------------------------------------------
# Total — sum every component, return per-component dict for diagnostics


def total_decomposed_obj(data, sol) -> tuple[float, dict[str, float]]:
    """Sum every closed-form component, returning ``(total, dict)``.

    The dict is keyed by component name (matching the function names
    above, minus the ``_obj`` suffix) and provides per-summand floats
    for diagnostic printing on test failure.
    """
    components: dict[str, float] = {}
    components["slack"]                       = slack_obj(data, sol)
    components["commodity_buy_eff"]           = commodity_buy_eff_obj(data, sol)
    components["commodity_buy_noEff"]         = commodity_buy_noEff_obj(data, sol)
    components["commodity_sell"]              = commodity_sell_obj(data, sol)
    components["commodity_section"]           = commodity_section_obj(data, sol)
    components["co2_price_eff"]               = co2_price_eff_obj(data, sol)
    components["co2_price_noEff"]             = co2_price_noEff_obj(data, sol)
    components["co2_section"]                 = co2_section_obj(data, sol)
    components["varcost_noEff"]               = varcost_noEff_obj(data, sol)
    components["varcost_eff_source"]          = varcost_eff_source_obj(data, sol)
    components["varcost_eff_sink"]            = varcost_eff_sink_obj(data, sol)
    components["varcost_eff_connection"]      = varcost_eff_connection_obj(data, sol)
    components["startup_cost"]                = startup_cost_obj(data, sol)
    components["invest_p_annuity"]            = invest_p_annuity_obj(data, sol)
    components["invest_n_annuity"]            = invest_n_annuity_obj(data, sol)
    components["lifetime_fixed_cost_p"]       = lifetime_fixed_cost_p_obj(data, sol)
    components["lifetime_fixed_cost_n"]       = lifetime_fixed_cost_n_obj(data, sol)
    components["divest_p_savings"]            = divest_p_savings_obj(data, sol)
    components["divest_n_savings"]            = divest_n_savings_obj(data, sol)
    components["divest_p_salvage"]            = divest_p_salvage_obj(data, sol)
    components["divest_n_salvage"]            = divest_n_salvage_obj(data, sol)
    components["vq_capacity_margin"]          = vq_capacity_margin_obj(data, sol)
    components["vq_inertia"]                  = vq_inertia_obj(data, sol)
    components["vq_non_synchronous"]          = vq_non_synchronous_obj(data, sol)
    components["vq_reserve"]                  = vq_reserve_obj(data, sol)
    components["storage_state_reference_price"] = storage_state_reference_price_obj(data, sol)
    total = sum(components.values())
    return total, components
