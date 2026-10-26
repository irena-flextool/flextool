"""Group soft floors: capacity margin, inertia and the non-synchronous share limit.

Owns the variables vq_capacity_margin, vq_inertia and vq_non_synchronous, the rows capacityMargin,
inertia_constraint and non_sync_constraint, and their penalties in the objective.  Each row asks a node
group to meet a floor (available capacity, rotating inertia) or to stay under a ceiling (share of
non-synchronous infeed); the slack variable absorbs the violation and is priced in the objective.
Every slack is a fraction of the group's scaling quantity (group_capacity_for_scaling or the inertia
limit), which keeps the rows well scaled.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Expr

from flextool.engine.input._axis_enums import align_join_dtypes
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.model._common import _has_rows, _var_view

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def _provider_get(provider, path: Path) -> pl.DataFrame | None:
    """The provider's frame for ``path``, or None when the provider is missing or lacks it."""
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)


# ---------------------------------------------------------------------------
# Loader

# Every FlexData field this module owns.  load_data returns all of them (None when not loaded here); the
# Params among them are filled by the direct- and derived-parameter passes.
_FIELDS: tuple[str, ...] = (
    "groupCapacityMargin", "groupInertia", "groupNonSync",
    "group_node",
    "process_unit",
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
    "p_inv_group_cap", "p_group_capacity_for_scaling",
    "pdGroup_capacity_margin", "pdGroup_penalty_capacity_margin",
    "pdGroup_inertia_limit", "pdGroup_penalty_inertia",
    "pdGroup_non_synchronous_limit", "pdGroup_penalty_non_synchronous",
    "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
    "p_positive_inflow", "p_negative_inflow",
    "pdtNodeInflow_per_step",
)


def _read_csv_or_none(p: Path, *, provider=None) -> pl.DataFrame | None:
    """The provider's frame for ``p``; None when missing or empty."""
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return None
    return df


def _read_inertia_constants(inp: Path, *, provider=None) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """``(sink_df, source_df)``: the non-zero ``inertia_constant`` rows of ``p_process_sink.csv`` /
    ``p_process_source.csv`` as ``(p, sink|source, value)``, each None when absent.

    Both files are long format: ``process, sink|source, sourceSinkParam, <value>``.
    """
    def _read(path: Path, side: str) -> pl.DataFrame | None:
        df = _provider_get(provider, path)
        if df is None or df.height == 0:
            return None
        if not {"process", "sourceSinkParam"}.issubset(df.columns):
            return None
        value_col = df.columns[-1]
        out = (df.filter(pl.col("sourceSinkParam") == "inertia_constant")
               .rename({"process": "p", value_col: "value"})
               .select("p", side, "value")
               .with_columns(pl.col("value").cast(pl.Float64, strict=False))
               .filter(pl.col("value") != 0.0))
        return out if out.height > 0 else None
    return _read(inp / "p_process_sink.csv", "sink"), _read(inp / "p_process_source.csv", "source")


def load_data(inp: Path, sd: Path, dt: pl.DataFrame,
              nb: pl.DataFrame | None,
              pss_eff: pl.DataFrame | None,
              pss_noEff: pl.DataFrame | None,
              p_unitsize: Param | None,
              *, provider=None,
              **_unused) -> dict:
    """Group sets and scaling factors of the three features, as ``field_name -> DataFrame | Param | None``.

    The caller merges the dict into FlexData.  Every field of ``_FIELDS`` is present; missing files give
    None.  The group limits, penalties, inertia constants and signed inflows are filled by the direct- and
    derived-parameter passes, not here.  ``inp`` / ``sd`` are the input and solve_data directories (files
    are read through ``provider``); the other positional arguments are accepted for the loader's uniform
    call and unused.
    """
    out = {f: None for f in _FIELDS}

    # Group sets from the method / flag files, which the input derivation fills with the groups whose
    # capacity_margin_method is manual / has_inertia is yes / has_non_synchronous is yes.  A limit
    # authored without its method or flag does not make a group.
    for field in ("groupCapacityMargin", "groupInertia", "groupNonSync"):
        df = _provider_get(provider, inp / f"{field}.csv")
        if df is not None and df.height > 0:
            out[field] = df.rename({df.columns[0]: "g"}).select("g").unique()

    # group_node: solve_data/group_node.csv, else the raw input/group__node.csv; the first with rows wins.
    gn = None
    for path in (sd / "group_node.csv", inp / "group__node.csv"):
        df = _provider_get(provider, path)
        if df is None:
            continue
        if df.height > 0:
            gn = df.rename({"group": "g", "node": "n"}).select("g", "n").unique()
            break
    out["group_node"] = gn

    # process_unit: processes that are units.  The capacity margin counts units only, so connections do
    # not count transmission capacity as group capacity.  Header "process_unit" or "process".  A present but
    # empty file gives an empty frame (no units: every process is a connection, which the connection reserve
    # rows need to know); a missing file leaves the field None.
    df = _provider_get(provider, sd / "process_unit.csv")
    if df is not None and df.width > 0:
        col = df.columns[0]
        out["process_unit"] = df.rename({col: "p"}).select("p").unique()

    # group_capacity_for_scaling and its inverse inv_group_cap, each derived from the other when only
    # one file is present.
    df = _provider_get(provider, sd / "group_capacity_for_scaling.csv")
    if df is not None and df.height > 0:
        df = (df.rename({"group": "g", "period": "d"})
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .select("g", "d", "value")
              .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0)))
        if df.height > 0:
            out["p_group_capacity_for_scaling"] = Param(("g", "d"), df)
    df = _provider_get(provider, sd / "inv_group_cap.csv")
    if df is not None and df.height > 0:
        df = (df.rename({"group": "g", "period": "d"})
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .select("g", "d", "value")
              .filter(pl.col("value").is_not_null()))
        if df.height > 0:
            out["p_inv_group_cap"] = Param(("g", "d"), df)
    if out["p_inv_group_cap"] is None and out["p_group_capacity_for_scaling"] is not None:
        gcs = out["p_group_capacity_for_scaling"].frame
        out["p_inv_group_cap"] = Param(("g", "d"), gcs.with_columns(value=1.0 / pl.col("value"))
                                       .select("g", "d", "value"))
    if out["p_group_capacity_for_scaling"] is None and out["p_inv_group_cap"] is not None:
        ig = out["p_inv_group_cap"].frame
        out["p_group_capacity_for_scaling"] = Param(("g", "d"), ig.with_columns(value=1.0 / pl.col("value"))
                                                    .select("g", "d", "value"))

    # Arcs that carry an inertia constant (the constants themselves come from the direct-parameter pass).
    sink_df, src_df = _read_inertia_constants(inp, provider=provider)
    if sink_df is not None:
        out["process_sink_inertia"] = sink_df.select("p", "sink").unique()
    if src_df is not None:
        out["process_source_inertia"] = src_df.select("p", "source").unique()

    # Non-synchronous infeed arcs, and the (process, group) pairs whose flows stay inside the group.
    p_sink_ns = _read_csv_or_none(sd / "process__sink_nonSync.csv", provider=provider)
    if p_sink_ns is not None:
        out["process_sink_nonSync"] = p_sink_ns.rename({"process": "p"}).select("p", "sink").unique()
    p_grp_inside = _read_csv_or_none(sd / "process__group_inside_group_nonSync.csv", provider=provider)
    if p_grp_inside is not None:
        rn = {}
        if "process" in p_grp_inside.columns:
            rn["process"] = "p"
        if "group" in p_grp_inside.columns:
            rn["group"] = "g"
        out["process_group_inside_nonSync"] = p_grp_inside.rename(rn).select("p", "g").unique()

    return out


# ---------------------------------------------------------------------------
# Feature flags (set on the build context by model._prepare)


def _capacity_margin_periods(d) -> pl.DataFrame | None:
    """``(d,)`` periods of dt that are in ``period_invest`` (the capacity-margin domain); None when there are none.

    ``period_invest`` = the solve's investment periods and their stochastic branch periods."""
    pinv = getattr(d, "period_invest", None)
    if pinv is None or pinv.height == 0 or d.dt is None:
        return None
    keep = pinv.get_column("d").cast(pl.Utf8).implode()
    out = d.dt.select("d").unique(maintain_order=True).filter(pl.col("d").cast(pl.Utf8).is_in(keep))
    return out if out.height > 0 else None


def has_capacity_margin(d) -> bool:
    """True when some group has capacity_margin_method = manual, the group scaling is populated and the solve has
    an investment period in dt.

    A group without a capacity_margin value has margin 0 (its capacity must cover the peak net load).  A solve
    without investment periods (dispatch, rolling dispatch) has no capacity margin."""
    return (_has_rows(getattr(d, "groupCapacityMargin", None)) and d.p_inv_group_cap is not None
            and _capacity_margin_periods(d) is not None)


def has_inertia(d) -> bool:
    """True when some group has has_inertia = yes, some group has an inertia limit, and the data to build
    the row is populated.  A flagged group without a limit has no requirement (its row is Σ … ≥ 0)."""
    return (_has_rows(getattr(d, "groupInertia", None))
            and d.pdGroup_inertia_limit is not None and d.group_node is not None
            and d.process_source_sink is not None and d.p_unitsize is not None)


def has_non_sync(d) -> bool:
    """True when some group has has_non_synchronous = yes and the data to build its row is populated.

    A flagged group without a non_synchronous_limit has limit 0."""
    return (_has_rows(getattr(d, "groupNonSync", None))
            and d.group_node is not None
            and d.process_source_sink is not None and d.p_unitsize is not None
            and d.p_step_duration is not None and d.p_inv_group_cap is not None)


# ---------------------------------------------------------------------------
# Variables


def add_variables(b: _Build) -> None:
    """Declare the slack variables of the active group floors."""
    if not (b.has_capacity_margin or b.has_inertia or b.has_non_sync):
        return
    m, d = b.m, b.d

    # ─── VAR. ─── vq_capacity_margin ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_capacity_margin[g,d] ≥ 0   ∀ g ∈ groupCapacityMargin, d ∈ D_inv
    #   D_inv = period_invest ∩ periods of dt: the solve's investment periods and the branch periods of those
    #   anchors (branch periods kept as before, pending the stochastics re-grounding)
    # Capacity margin shortfall of group g in period d, as a fraction of
    # group_capacity_for_scaling: MW = vq_capacity_margin · group_capacity_for_scaling[g,d].
    if b.has_capacity_margin:
        b.vq_capacity_margin = m.add_var("vq_capacity_margin", ("g", "d"),
                                         d.groupCapacityMargin.join(_capacity_margin_periods(d), how="cross"),
                                         lower=0.0)

    # ─── VAR. ─── vq_inertia ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_inertia[g,d,t] ≥ 0   ∀ g ∈ groupInertia, (d,t) ∈ dt
    # Inertia shortfall of group g, as a fraction of its inertia limit:
    # MWs = vq_inertia · inertia_limit[g,d].
    if b.has_inertia:
        b.vq_inertia = m.add_var("vq_inertia", ("g", "d", "t"), d.groupInertia.join(d.dt, how="cross"), lower=0.0)

    # ─── VAR. ─── vq_non_synchronous ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_non_synchronous[g,d,t] ≥ 0   ∀ g ∈ groupNonSync, (d,t) ∈ dt
    # Non-synchronous infeed of group g above its allowed share, as a
    # fraction of group_capacity_for_scaling: MW = vq_non_synchronous · group_capacity_for_scaling[g,d].
    if b.has_non_sync:
        b.vq_non_synchronous = m.add_var("vq_non_synchronous", ("g", "d", "t"),
                                         d.groupNonSync.join(d.dt, how="cross"), lower=0.0)


# ---------------------------------------------------------------------------
# Constraints


def add_constraints(b: _Build) -> None:
    """capacityMargin, inertia_constraint and non_sync_constraint, each when its group set is active."""
    _eq_capacity_margin(b)
    _eq_inertia(b)
    _eq_non_sync(b)


def _arcs_at_group(arcs: pl.DataFrame, side: str, group_nodes: pl.DataFrame) -> pl.DataFrame:
    """``arcs`` (p, source, sink, …) whose ``side`` node is in ``group_nodes`` (g, n), with g added.

    ``side`` carries the entity-union Enum and ``n`` the node Enum; the join widens both to one dtype.
    """
    arcs_n, gn = align_join_dtypes(arcs.rename({side: "n"}), group_nodes, ("n",))
    return arcs_n.join(gn, on="n", how="inner").rename({"n": side})


def _invest_change(b: _Build, procs: pl.DataFrame, tag: str) -> tuple[Expr | None, Expr | None]:
    """``(Σ_{d′} v_invest_p[p,d′], Σ_{d′} v_divest_p[p,d′])`` over (p,d′,d) ∈ edd_invest_set / edd_divest_active,
    restricted to the processes in ``procs``; each None when absent.  Indexed (p, d)."""
    d = b.d
    procs_p = procs.select("p").unique()
    invest = divest = None
    edd_inv = getattr(d, "edd_invest_set", None)
    if b.has_invest_p and b.v_invest_p is not None and _has_rows(edd_inv):
        e = edd_inv.rename({"e": "p"}) if "e" in edd_inv.columns else edd_inv
        e, pp = align_join_dtypes(e, procs_p, ("p",))
        e = e.join(pp, on="p", how="inner")
        if e.height > 0:
            invest = Sum(Where(_var_view(b.v_invest_p, f"__{tag}_inv", {"d": "d_invest"}), e), over=("d_invest",))
    edd_div = getattr(d, "edd_divest_active", None)
    if b.has_divest_p and b.v_divest_p is not None and _has_rows(edd_div):
        e, pp = align_join_dtypes(edd_div, procs_p, ("p",))
        e = e.join(pp, on="p", how="inner")
        if e.height > 0:
            divest = Sum(Where(_var_view(b.v_divest_p, f"__{tag}_div", {"d": "d_divest"}), e), over=("d_divest",))
    return invest, divest


def _eq_capacity_margin(b: _Build) -> None:
    """capacityMargin: available unit capacity of a group covers its net demand plus the margin."""
    if not b.has_capacity_margin:
        return
    m, d = b.m, b.d
    gcm = d.groupCapacityMargin
    invc = d.p_inv_group_cap
    p_us = d.p_unitsize
    exist_cnt = d.p_process_existing_count
    prof_val = d.p_profile_value

    # ─── EQ. ─── capacityMargin ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,f) ∈ prof(g)} (Σ_{d′} v_invest_p[p,d′] − Σ_{d′} v_divest_p[p,d′]) · unitsize[p] · profile[f,d,t]
    #                                                                                         · icap[g,d]
    #   + Σ_{(p,s,k) ∈ cap(g)} (Σ_{d′} v_invest_p[p,d′] − Σ_{d′} v_divest_p[p,d′]) · unitsize[p] · icap[g,d]
    #   − Σ_{(p,s,k) ∈ use_eff(g)} v_flow[p,s,k,d,t] · unitsize[p] · slope[p,d,t] · icap[g,d]
    #   − Σ_{(p,s,k) ∈ use_eff(g), p ∈ process_min_load_eff} v_online[p,d,t] · section[p] · unitsize[p] · icap[g,d]
    #   − Σ_{(p,s,k) ∈ use_noEff(g)} v_flow[p,s,k,d,t] · unitsize[p] · icap[g,d]
    #   + vq_capacity_margin[g,d]
    #   ≥ capacity_margin[g,d] · icap[g,d] − Σ_{n ∈ N(g)} inflow_real[n,d,t] / Δt[d,t] · icap[g,d]
    #     − Σ_{prof(g)} existing_count[p,d] · unitsize[p] · profile[f,d,t] · icap[g,d]
    #     − Σ_{cap(g)} existing_count[p,d] · unitsize[p] · icap[g,d]
    #   ∀ g ∈ groupCapacityMargin, (d,t) ∈ dt : d ∈ D_inv
    #   groupCapacityMargin = groups with capacity_margin_method = manual;  capacity_margin[g,d] = 0 where not authored
    #   D_inv = period_invest ∩ periods of dt (investment periods of the solve and the branch periods of those
    #   anchors, the latter kept pending the stochastics re-grounding); a solve without investment periods has
    #   no capacity margin
    #   N(g) = nodes of g without a storage state;  icap = inv_group_cap = 1 / group_capacity_for_scaling
    #   prof(g) = unit arcs into N(g) with an upper_limit or fixed profile f;  cap(g) = the other unit arcs into N(g)
    #   use_*(g) = unit arcs out of N(g);  Σ_{d′} over edd_invest_set / edd_divest_active
    #   inflow_real = the node's inflow before the invest-solve energy margin (energy_margin_multiplier / _adder),
    #   which is an adequacy device and not load; the margin's capacity is the capacity_margin parameter
    #   (pdtNodeInflow_per_step = inflow_real / Δt)
    # The capacity a group's units could deliver — existing plus invested
    # minus divested, profile-limited where a profile caps the output — less
    # what its units consume covers the group's net demand plus the margin;
    # the shortfall is penalised.  Existing capacity is a constant and moves
    # to the right-hand side.  Only units count (process_unit; all processes
    # when the set is empty), so connections do not count as capacity.  The
    # row is scaled by icap, which makes the slack a fraction of
    # group_capacity_for_scaling.  Without group_node only the margin and the
    # slack remain.
    lhs: dict = {"slack": b.vq_capacity_margin}
    rhs: dict = {}

    proc_unit = getattr(d, "process_unit", None)
    gn_no_state = None
    if d.group_node is not None:
        gn_no_state = d.group_node.join(gcm, on="g", how="inner")
        if d.nodeState is not None and d.nodeState.height > 0:
            gn_no_state = gn_no_state.join(d.nodeState, on="n", how="anti")

    def _units(df: pl.DataFrame) -> pl.DataFrame:
        if proc_unit is not None and proc_unit.height > 0:
            return df.join(proc_unit.select("p"), on="p", how="inner")
        return df

    # Unit arcs into the group (p, source, sink, g), split into profile-limited and capacity-limited.
    pss_to_grp = None
    if d.process_source_sink is not None and gn_no_state is not None:
        pss_to_grp = _arcs_at_group(_units(d.process_source_sink), "sink", gn_no_state)
    profile_idx = None
    prof_sets = [s for s in (d.process_profile_upper, d.process_profile_fixed) if s is not None and s.height > 0]
    if prof_sets and pss_to_grp is not None:
        profile_idx = (pl.concat([s.select("p", "source", "sink", "f") for s in prof_sets], how="vertical").unique()
                       .join(pss_to_grp, on=["p", "source", "sink"], how="inner"))
    capacity_idx = pss_to_grp
    if pss_to_grp is not None and profile_idx is not None:
        capacity_idx = pss_to_grp.join(profile_idx.select("p", "source", "sink").unique(),
                                       on=["p", "source", "sink"], how="anti")
    has_process_data = pss_to_grp is not None and exist_cnt is not None and p_us is not None

    existing_frames: list[pl.DataFrame] = []     # existing capacity · icap, (g, d, t, value)
    if has_process_data and profile_idx is not None and prof_val is not None:
        invest, divest = _invest_change(b, profile_idx, "cap_margin")
        if invest is not None:
            lhs["prof_invest"] = Sum(Where(invest, profile_idx) * p_us * prof_val * invc,
                                     over=("p", "source", "sink", "f"))
        if divest is not None:
            lhs["prof_divest"] = Sum(-(Where(divest, profile_idx) * p_us * prof_val * invc),
                                     over=("p", "source", "sink", "f"))
        existing_frames.append(
            profile_idx.join(exist_cnt.frame.rename({"value": "exist"}), on="p", how="inner")
            .join(p_us.frame.rename({"value": "us"}), on="p", how="inner")
            .join(prof_val.frame.rename({"value": "prof"}), on=["f", "d"], how="inner")
            .join(invc.frame.rename({"value": "inv"}), on=["g", "d"], how="inner")
            .with_columns(value=pl.col("exist") * pl.col("us") * pl.col("prof") * pl.col("inv"))
            .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
            .select("g", "d", "t", "value"))
    if has_process_data and capacity_idx is not None and capacity_idx.height > 0:
        invest, divest = _invest_change(b, capacity_idx, "cap_margin")
        if invest is not None:
            lhs["cap_invest"] = Sum(Where(invest, capacity_idx) * p_us * invc, over=("p", "source", "sink"))
        if divest is not None:
            lhs["cap_divest"] = Sum(-(Where(divest, capacity_idx) * p_us * invc), over=("p", "source", "sink"))
        existing_frames.append(
            capacity_idx.join(exist_cnt.frame.rename({"value": "exist"}), on="p", how="inner")
            .join(p_us.frame.rename({"value": "us"}), on="p", how="inner")
            .join(invc.frame.rename({"value": "inv"}), on=["g", "d"], how="inner")
            .with_columns(value=pl.col("exist") * pl.col("us") * pl.col("inv"))
            .group_by(["g", "d"]).agg(pl.col("value").sum())
            .join(d.dt, on="d", how="inner").select("g", "d", "t", "value"))

    # Consumption of the group's units.  The eff term counts input at the efficiency slope and needs
    # p_slope; the min-load section input of online units is part of it.
    if has_process_data and b.v_flow is not None:
        pss_eff = d.process_source_sink_eff
        if pss_eff is not None and pss_eff.height > 0:
            use_eff = _arcs_at_group(_units(pss_eff), "source", gn_no_state)
            if use_eff.height > 0 and d.p_slope is not None:
                lhs["consume_eff"] = Sum(-(Where(b.v_flow, use_eff) * p_us * d.p_slope * invc),
                                         over=("p", "source", "sink"))
                if (d.process_min_load_eff is not None and d.process_min_load_eff.height > 0
                        and d.p_section is not None):
                    use_mle = use_eff.join(d.process_min_load_eff, on="p", how="inner")
                    if use_mle.height > 0:
                        if b.v_online_linear is not None:
                            lhs["consume_section_lin"] = Sum(
                                -(Where(b.v_online_linear, use_mle) * d.p_section * p_us * invc),
                                over=("p", "source", "sink"))
                        if b.v_online_integer is not None:
                            lhs["consume_section_int"] = Sum(
                                -(Where(b.v_online_integer, use_mle) * d.p_section * p_us * invc),
                                over=("p", "source", "sink"))
        pss_noEff = d.process_source_sink_noEff
        if pss_noEff is not None and pss_noEff.height > 0:
            use_noEff = _arcs_at_group(_units(pss_noEff), "source", gn_no_state)
            if use_noEff.height > 0:
                lhs["consume_noEff"] = Sum(-(Where(b.v_flow, use_noEff) * p_us * invc), over=("p", "source", "sink"))

    if existing_frames:
        ex = (pl.concat(existing_frames, how="vertical")
              .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
              .with_columns(value=-pl.col("value")))
        rhs["existing_neg"] = Param(("g", "d", "t"), ex.select("g", "d", "t", "value"))
    if d.pdGroup_capacity_margin is not None:
        rhs["floor"] = d.pdGroup_capacity_margin * invc
    inflow_per = getattr(d, "pdtNodeInflow_per_step", None)
    if inflow_per is not None and gn_no_state is not None and gn_no_state.height > 0:
        rhs_inflow = (gn_no_state.join(inflow_per.frame.rename({"value": "inflow"}), on="n", how="inner")
                      .join(invc.frame.rename({"value": "inv"}), on=["g", "d"], how="inner")
                      .with_columns(value=-pl.col("inflow") * pl.col("inv"))
                      .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                      .select("g", "d", "t", "value"))
        if rhs_inflow.height > 0:
            rhs["inflow_neg"] = Param(("g", "d", "t"), rhs_inflow)

    dt_inv = d.dt.join(_capacity_margin_periods(d), on="d", how="semi")
    m.add_cstr("capacityMargin", over=gcm.join(dt_inv, how="cross"), sense=">=", lhs_terms=lhs, rhs_terms=rhs)


def _eq_inertia(b: _Build) -> None:
    """inertia_constraint: rotating inertia of a group meets its inertia limit."""
    if not b.has_inertia:
        return
    m, d = b.m, b.d
    pss = d.process_source_sink
    gn_in_gi = d.group_node.join(d.groupInertia, on="g", how="inner")

    # ─── EQ. ─── inertia_constraint ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{side ∈ {source, sink}} Σ_{(p,s,k) ∈ arcs_side(g)} x[p,·,d,t] · inertia_constant[p,side] · unitsize[p]
    #   + vq_inertia[g,d,t] · inertia_limit[g,d]
    #   ≥ inertia_limit[g,d]
    #   ∀ g ∈ groupInertia, (d,t) ∈ dt
    #   groupInertia = groups with has_inertia = yes;  inertia_limit[g,d] = 0 where not authored (no requirement:
    #   the row reads Σ … ≥ 0 and its slack is in no row)
    #   arcs_side(g) = arcs whose side node is in g and that carry an inertia constant on that side
    #   x = v_online[p,d,t] for p ∈ process_online (linear or integer), else v_flow[p,s,k,d,t]
    # Synchronous machines connected to the group's nodes supply inertia (MWs)
    # in proportion to the units online, or to their flow when they have no
    # online variable; the shortfall is penalised.  The slack is a fraction of
    # the limit.
    lhs: dict = {"slack": b.vq_inertia * d.pdGroup_inertia_limit}
    for side in ("source", "sink"):
        side_idx = getattr(d, f"process_{side}_inertia", None)
        const_param = getattr(d, f"p_process_{side}_inertia_constant", None)
        if side_idx is None or side_idx.height == 0 or const_param is None:
            continue
        with_g = _arcs_at_group(pss.join(side_idx, on=["p", side], how="inner"), side, gn_in_gi)
        if with_g.height == 0:
            continue
        coef = const_param * d.p_unitsize
        for online_set, v_online, key in ((d.process_online_linear, b.v_online_linear, f"online_lin_{side}"),
                                          (d.process_online_integer, b.v_online_integer, f"online_int_{side}")):
            if online_set is not None and online_set.height > 0 and v_online is not None:
                with_g_on = with_g.join(online_set, on="p", how="inner")
                if with_g_on.height > 0:
                    lhs[key] = Sum(Where(v_online, with_g_on) * coef, over=("p", "source", "sink"))
        if b.v_flow is not None:
            with_g_off = with_g
            if d.process_online is not None and d.process_online.height > 0:
                with_g_off = with_g.join(d.process_online, on="p", how="anti")
            if with_g_off.height > 0:
                lhs[f"flow_{side}"] = Sum(Where(b.v_flow, with_g_off) * coef, over=("p", "source", "sink"))

    m.add_cstr("inertia_constraint", over=d.groupInertia.join(d.dt, how="cross"), sense=">=",
               lhs_terms=lhs, rhs_terms={"floor": d.pdGroup_inertia_limit})


def _eq_non_sync(b: _Build) -> None:
    """non_sync_constraint: non-synchronous infeed of a group stays under its share of the group's supply."""
    if not b.has_non_sync:
        return
    m, d = b.m, b.d
    pss = d.process_source_sink
    p_us = d.p_unitsize
    invc = d.p_inv_group_cap
    pgns = d.pdGroup_non_synchronous_limit
    gn_in_gns = d.group_node.join(d.groupNonSync, on="g", how="inner")
    proc_inside = d.process_group_inside_nonSync          # (p, g) | None
    sink_ns = d.process_sink_nonSync                      # (p, sink) | None

    def _not_inside(idx_g: pl.DataFrame) -> pl.DataFrame:
        if proc_inside is None or proc_inside.height == 0:
            return idx_g
        return idx_g.join(proc_inside, on=["p", "g"], how="anti")

    # ─── EQ. ─── non_sync_constraint ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k) ∈ in_ns(g)} v_flow[p,s,k,d,t] · unitsize[p] · Δt[d,t] · icap[g,d]
    #   + Σ_{n ∈ N(g)} positive_inflow[n,d,t] · icap[g,d]
    #   − vq_non_synchronous[g,d,t] · Δt[d,t]
    #   ≤ ns_limit[g,d] · icap[g,d] · ( Σ_{(p,s,k) ∈ out_noEff(g)} v_flow[p,s,k,d,t] · unitsize[p] · Δt[d,t]
    #                                 + Σ_{(p,s,k) ∈ out_eff(g)} v_flow[p,s,k,d,t] · slope[p,d,t] · unitsize[p]
    #                                                            · Δt[d,t]
    #                                 + Σ_{(p,s,k) ∈ out_eff(g), p ∈ process_min_load_eff}
    #                                       v_online[p,d,t] · section[p] · unitsize[p] · Δt[d,t]
    #                                 − Σ_{n ∈ N(g)} negative_inflow[n,d,t] )
    #   ∀ g ∈ groupNonSync, (d,t) ∈ dt
    #   groupNonSync = groups with has_non_synchronous = yes
    #   N(g) = nodes of g;  icap = inv_group_cap;  ns_limit = non_synchronous_limit, 0 where not authored (the
    #   right-hand side then vanishes: no non-synchronous infeed without the penalised slack)
    #   in_ns(g) = arcs into N(g) with (p,k) ∈ process_sink_nonSync
    #   out_*(g) = arcs out of N(g);  arcs of processes inside the group, (p,g) ∈ process_group_inside_nonSync,
    #   are left out of in_ns and out_*
    #   positive_inflow / negative_inflow = max(inflow_real, 0) / min(inflow_real, 0);  inflow_real = the node's
    #   inflow before the invest-solve energy margin
    # Non-synchronous infeed (flagged arcs and positive exogenous inflow,
    # which counts as non-synchronous) is at most ns_limit times the group's
    # supply, measured as the energy drawn by the processes and the demand
    # leaving its nodes (the real demand: an energy margin does not count); the excess is penalised.  The slack has no icap,
    # which makes it a fraction of group_capacity_for_scaling.  slope already
    # holds the sink/source coefficient ratio of multi-arc processes.
    lhs: dict = {"slack_neg": -(b.vq_non_synchronous * d.p_step_duration)}
    if sink_ns is not None and sink_ns.height > 0 and b.v_flow is not None:
        pss_in_g = (pss.join(sink_ns, on=["p", "sink"], how="inner")
                    .rename({"sink": "n"})
                    .join(gn_in_gns, on="n", how="inner")
                    .rename({"n": "sink"}))
        pss_in_g = _not_inside(pss_in_g)
        if pss_in_g.height > 0:
            lhs["incoming_nonSync"] = Sum(Where(b.v_flow, pss_in_g) * p_us * d.p_step_duration * invc,
                                          over=("p", "source", "sink"))
    p_pos = d.p_positive_inflow
    if p_pos is not None and gn_in_gns.height > 0:
        exo_in = (gn_in_gns.join(p_pos.frame.rename({"value": "v"}), on="n", how="inner")
                  .join(invc.frame.rename({"value": "iv"}), on=["g", "d"], how="inner")
                  .with_columns(value=pl.col("v") * pl.col("iv"))
                  .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                  .select("g", "d", "t", "value"))
        if exo_in.height > 0:
            lhs["exo_inflow"] = Param(("g", "d", "t"), exo_in)

    rhs: dict = {}
    # Every right-hand-side term is × ns_limit, so without any limit the right-hand side is 0.
    out_factor = p_us * d.p_step_duration * pgns * invc if pgns is not None else None
    pss_noEff = d.process_source_sink_noEff
    if out_factor is not None and pss_noEff is not None and pss_noEff.height > 0 and b.v_flow is not None:
        out_noEff = _not_inside(pss_noEff.rename({"source": "n"}).join(gn_in_gns, on="n", how="inner")
                                .rename({"n": "source"}))
        if out_noEff.height > 0:
            rhs["out_noEff"] = Sum(Where(b.v_flow, out_noEff) * out_factor, over=("p", "source", "sink"))
    pss_eff = d.process_source_sink_eff
    if out_factor is not None and pss_eff is not None and pss_eff.height > 0 and b.v_flow is not None:
        out_eff = _not_inside(pss_eff.rename({"source": "n"}).join(gn_in_gns, on="n", how="inner")
                              .rename({"n": "source"}))
        if out_eff.height > 0:
            if d.p_slope is not None:
                term = Where(b.v_flow, out_eff) * d.p_slope * out_factor
            else:
                term = Where(b.v_flow, out_eff) * out_factor
            rhs["out_eff"] = Sum(term, over=("p", "source", "sink"))
            if (d.process_min_load_eff is not None and d.process_min_load_eff.height > 0
                    and d.p_section is not None):
                out_mle = out_eff.join(d.process_min_load_eff, on="p", how="inner")
                if out_mle.height > 0:
                    if b.v_online_linear is not None:
                        rhs["out_section_lin"] = Sum(Where(b.v_online_linear, out_mle) * d.p_section * out_factor,
                                                     over=("p", "source", "sink"))
                    if b.v_online_integer is not None:
                        rhs["out_section_int"] = Sum(Where(b.v_online_integer, out_mle) * d.p_section * out_factor,
                                                     over=("p", "source", "sink"))
    p_neg = d.p_negative_inflow
    if p_neg is not None and pgns is not None and gn_in_gns.height > 0:
        exo_out = (gn_in_gns.join(p_neg.frame.rename({"value": "v"}), on="n", how="inner")
                   .join(invc.frame.rename({"value": "iv"}), on=["g", "d"], how="inner")
                   .join(pgns.frame.rename({"value": "lim"}), on=["g", "d"], how="inner")
                   .with_columns(value=-pl.col("v") * pl.col("iv") * pl.col("lim"))
                   .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                   .select("g", "d", "t", "value"))
        if exo_out.height > 0:
            rhs["exo_demand"] = Param(("g", "d", "t"), exo_out)

    m.add_cstr("non_sync_constraint", over=d.groupNonSync.join(d.dt, how="cross"), sense="<=",
               lhs_terms=lhs, rhs_terms=rhs)


# ---------------------------------------------------------------------------
# Objective


def objective_terms(b: _Build) -> Expr | None:
    """Penalties of the inertia, non-synchronous and capacity margin slacks."""
    if not (b.has_capacity_margin or b.has_inertia or b.has_non_sync):
        return None
    d = b.d

    # ─── OBJ. ─── inertia / non-synchronous / capacity margin slack ──────────
    # ─────────────────────────────────────────────────────────────────────────
    # + Σ_{(g,d,t)} vq_inertia[g,d,t] · inertia_limit[g,d] · penalty_inertia[g,d] · w_op[d,t]
    # + Σ_{(g,d,t)} vq_non_synchronous[g,d,t] · gcs[g,d] · penalty_non_synchronous[g,d] · w_op[d,t]
    # + Σ_{g ∈ groupCapacityMargin, d ∈ D_inv} vq_capacity_margin[g,d] · gcs[g,d] · penalty_capacity_margin[g,d]
    #                                                                     · inflation_op[d] · 1000
    #   gcs = group_capacity_for_scaling;  w_op = b.op_factor
    # Each slack is scaled back to physical units (MWs, MW) and priced per hour
    # of operation.  The capacity margin is a planning quantity: it is priced
    # once per period, without step weights or branch weights, and its penalty
    # is authored in CUR/kW (× 1000 to CUR/MW).  The output layer
    # (calc_slacks) reports the same costs.
    pieces: list[Expr] = []
    if (b.vq_inertia is not None and d.pdGroup_inertia_limit is not None
            and d.pdGroup_penalty_inertia is not None):
        pieces.append(Sum(b.vq_inertia * d.pdGroup_inertia_limit * d.pdGroup_penalty_inertia * b.op_factor))
    if (b.vq_non_synchronous is not None and d.p_group_capacity_for_scaling is not None
            and d.pdGroup_penalty_non_synchronous is not None):
        pieces.append(Sum(b.vq_non_synchronous * d.p_group_capacity_for_scaling
                          * d.pdGroup_penalty_non_synchronous * b.op_factor))
    if (b.vq_capacity_margin is not None and d.p_group_capacity_for_scaling is not None
            and d.pdGroup_penalty_capacity_margin is not None and d.p_inflation_op is not None):
        pieces.append(Sum(b.vq_capacity_margin * d.p_group_capacity_for_scaling
                          * d.pdGroup_penalty_capacity_margin * d.p_inflation_op * 1000.0))
    if not pieces:
        return None
    total = pieces[0]
    for piece in pieces[1:]:
        total = total + piece
    return total
