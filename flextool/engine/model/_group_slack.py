"""Group soft floors: capacity margin, inertia and the non-synchronous share limit.

Owns the variables vq_capacity_margin, vq_inertia and vq_non_synchronous, the rows capacityMargin,
inertia_constraint and non_sync_constraint, and their penalties in the objective.  Each row asks a node
group to meet a floor (available capacity, rotating inertia) or to stay under a ceiling (share of
non-synchronous infeed); the slack variable absorbs the violation and is priced in the objective.
Every slack is a fraction of the group's scaling quantity (group_capacity_for_scaling or the inertia
limit), which keeps the rows well scaled.

Coarse steps (``new_stepduration`` groups), the row rule: a group whose row has only coarse members (its flows and
units online are those of coarse processes, ``flex_data.coarse_row_members``) gets one row per coarse step, the
row's Δt-weighted mean over the step (``<family>_coarse``, ``_coarse_flows.add_cstr``), and its inertia /
non-synchronous slack is one value per coarse step; a group with a fine member keeps its rows per solve step and
reads the coarse terms at their constant value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Expr

from flextool.engine.input._axis_enums import align_join_dtypes, join_aligned
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.model import _coarse_flows
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
    "pdtGroup_inertia_limit", "pdGroup_penalty_inertia",
    "pdtGroup_non_synchronous_limit", "pdGroup_penalty_non_synchronous",
    "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
    "p_positive_inflow", "p_negative_inflow",
    "pdtNodeInflow_per_step",
)


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
    derived-parameter passes, not here.  The three group sets and ``process_sink_nonSync`` come from the
    projection pass (``apply_projection_params``) and ``process_group_inside_nonSync`` from
    ``_derived.process_group_inside_nonSync_from_flex``; they stay None here.  ``inp`` / ``sd`` are the
    input and solve_data directories (files are read through ``provider``); the other positional arguments
    are accepted for the loader's uniform call and unused.
    """
    out = {f: None for f in _FIELDS}

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


def _capacity_margin_branch_weight(d) -> Param | None:
    """``(d,)`` branch weight of the capacity-margin periods; None when the solve has no ``pd_branch_weight``.

    Left join onto the capacity-margin domain: a period without a weight row has weight 1."""
    periods = _capacity_margin_periods(d)
    pd_bw = getattr(d, "pd_branch_weight", None)
    if periods is None or pd_bw is None:
        return None
    left, right = align_join_dtypes(periods, pd_bw.frame.select("d", "value"), ("d",))
    frame = left.join(right, on="d", how="left").with_columns(pl.col("value").fill_null(1.0))
    return Param(("d",), frame)


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
            and d.pdtGroup_inertia_limit is not None and d.group_node is not None
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
    #   D_inv = period_invest ∩ periods of dt: the solve's investment periods and, in a stochastic fan, the
    #   forecast branch periods of those anchors (a period label has no dt); one slack per forecast
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
    # MWs = vq_inertia · inertia_limit[g,d,t].
    # ─── VAR. ─── vq_inertia, vq_non_synchronous (coarse rows) ───────────────
    # vq[g,d,b] ≥ 0   ∀ g whose inertia / non-synchronous row is averaged over coarse steps (the row rule), (d,b)
    #   the coarse steps of its row (b = the step's first fine step)
    # One shortfall per coarse step, held over its fine steps (a level, _coarse_flows.alias), as the row it relaxes.
    if b.has_inertia:
        idx = d.groupInertia.join(d.dt, how="cross")
        steps = _row_steps(b, "inertia", idx)
        b.vq_inertia = _coarse_flows.alias(
            b, m.add_var("vq_inertia", ("g", "d", "t"), _coarse_flows.domain(b, idx, steps), lower=0.0), idx, steps)

    # ─── VAR. ─── vq_non_synchronous ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_non_synchronous[g,d,t] ≥ 0   ∀ g ∈ groupNonSync, (d,t) ∈ dt
    # Non-synchronous infeed of group g above its allowed share, as a
    # fraction of group_capacity_for_scaling: MW = vq_non_synchronous · group_capacity_for_scaling[g,d].
    if b.has_non_sync:
        idx = d.groupNonSync.join(d.dt, how="cross")
        steps = _row_steps(b, "non_sync", idx)
        b.vq_non_synchronous = _coarse_flows.alias(
            b, m.add_var("vq_non_synchronous", ("g", "d", "t"), _coarse_flows.domain(b, idx, steps), lower=0.0),
            idx, steps)


def _row_partition(b: _Build, family: str, like: pl.DataFrame) -> pl.DataFrame | None:
    """The coarse steps ``(g, d, b_first, t, w)`` of the groups of ``family`` whose row has only coarse members
    (``flex_data.coarse_row_members``, :func:`_coarse_flows.row_partition`), in the dtypes of ``like``; None without
    any.  Kept on ``b.group_row_steps`` so the slack and the row read one partition."""
    if family not in b.group_row_steps:
        part = None
        if b.has_coarse_proc:
            from flextool.engine.preprocess.flex_data import coarse_row_members

            part = _coarse_flows.row_partition(b, coarse_row_members(b.d, family), like)
        b.group_row_steps[family] = part
    return b.group_row_steps[family]


def _row_steps(b: _Build, family: str, like: pl.DataFrame) -> pl.DataFrame:
    """``(g, d, b_first, t)`` of :func:`_row_partition` (the steps of the slack of ``family``); empty without."""
    part = _row_partition(b, family, like)
    if part is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in ("g", "d", "b_first", "t")})
    return part.select("g", "d", "b_first", "t")


# ---------------------------------------------------------------------------
# Constraints


def add_constraints(b: _Build) -> None:
    """capacityMargin, inertia_constraint and non_sync_constraint, each when its group set is active."""
    _eq_capacity_margin(b)
    _eq_inertia(b)
    _eq_non_sync(b)


def _arcs_at_group(arcs: pl.DataFrame, side: str, group_nodes: pl.DataFrame) -> pl.DataFrame:
    """``arcs`` (p, source, sink, …) whose ``side`` node is in ``group_nodes`` (g, n), with g added.

    ``side`` carries the entity-union Enum and ``n`` the node Enum; the node key is cast to the arcs' entity-union
    Enum, so ``side`` keeps its dtype.
    """
    return join_aligned(arcs.rename({side: "n"}), group_nodes, on="n", how="inner").rename({"n": side})


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
    d = b.d
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
    #   D_inv = period_invest ∩ periods of dt (investment periods of the solve; a period label has no dt, so a
    #   fan contributes its forecast branch periods, one row per forecast); a solve without investment periods
    #   has no capacity margin
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
    # A retirement v_divest[e,d′] counts in d iff (e,d′,d) ∈ edd_divest_active (ndd_divest_active):
    #   d′ ≤ d < EOL[e], the end of life of e's pre-existing capacity (none for reinvest_automatic).
    # Row rule (coarse steps): a group whose consuming units are all coarse processes gets one row per coarse step,
    # capacityMargin_coarse = the row's Δt-weighted mean over the step (_coarse_flows.add_cstr): the profile, the
    # inflow and the slope act through their step means, so the margin covers the step-mean net demand, not its
    # peak (named in the coarse-flow behaviour notice).  Otherwise the rows stay per solve step.
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
    over = gcm.join(dt_inv, how="cross")
    _coarse_flows.add_row_rule_cstr(b, "capacityMargin", over=over, sense=">=", lhs_terms=lhs, rhs_terms=rhs,
                                    partition=_row_partition(b, "capacity_margin", over))


def _eq_inertia(b: _Build) -> None:
    """inertia_constraint: rotating inertia of a group meets its inertia limit."""
    if not b.has_inertia:
        return
    d = b.d
    pss = d.process_source_sink
    gn_in_gi = d.group_node.join(d.groupInertia, on="g", how="inner")

    # ─── EQ. ─── inertia_constraint ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{side ∈ {source, sink}} Σ_{(p,s,k) ∈ arcs_side(g)} x[p,·,d,t] · inertia_constant[p,side] · unitsize[p]
    #   + vq_inertia[g,d,t] · inertia_limit[g,d,t]
    #   ≥ inertia_limit[g,d,t]
    #   ∀ g ∈ groupInertia, (d,t) ∈ dt
    #   groupInertia = groups with has_inertia = yes;  inertia_limit[g,d,t] = the authored constant, period, time
    #   or period + time value, 0 where not authored or authored 0 (no requirement: the row reads Σ … ≥ 0 and its
    #   slack is in no row)
    #   arcs_side(g) = arcs whose side node is in g and that carry an inertia constant on that side
    #   x = v_online[p,d,t] for p ∈ process_online (linear or integer), else v_flow[p,s,k,d,t]
    # Synchronous machines connected to the group's nodes supply inertia (MWs)
    # in proportion to the units online, or to their flow when they have no
    # online variable; the shortfall is penalised.  The slack is a fraction of
    # the limit.
    # Row rule (coarse steps): a group whose inertia providers are all coarse processes gets one row per coarse
    # step, inertia_constraint_coarse = the row's mean over the step (_coarse_flows.add_cstr), on one slack per
    # step; the limit enters as its Δt-weighted step mean (a time step the limit leaves out counts as 0), and a
    # limit that varies within a coarse step is named in the coarse-flow behaviour notice.  Otherwise per solve
    # step.
    lhs: dict = {"slack": b.vq_inertia * d.pdtGroup_inertia_limit}
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

    over = d.groupInertia.join(d.dt, how="cross")
    _coarse_flows.add_row_rule_cstr(b, "inertia_constraint", over=over, sense=">=", lhs_terms=lhs,
                                    rhs_terms={"floor": d.pdtGroup_inertia_limit},
                                    partition=_row_partition(b, "inertia", over))


def _eq_non_sync(b: _Build) -> None:
    """non_sync_constraint: non-synchronous infeed of a group stays under its share of the group's supply."""
    if not b.has_non_sync:
        return
    d = b.d
    pss = d.process_source_sink
    p_us = d.p_unitsize
    invc = d.p_inv_group_cap
    pgns_all = d.pdtGroup_non_synchronous_limit           # authored values, zeros included
    # The coefficients read the nonzero values only (a 0 limit adds no right-hand-side term).
    pgns = (None if pgns_all is None
            else Param(pgns_all.dims, pgns_all.lazy.filter(pl.col("value") != 0.0)))
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
    #   + Σ_{(p,s,k) ∈ back_in_ns(g)} v_flow_back[p,s,k,d,t] · unitsize[p] · Δt[d,t] · icap[g,d]
    #   + Σ_{n ∈ N(g)} positive_inflow[n,d,t] · icap[g,d]
    #   − vq_non_synchronous[g,d,t] · Δt[d,t]
    #   ≤ ns_limit[g,d,t] · icap[g,d] · ( Σ_{(p,s,k) ∈ out_noEff(g)} v_flow[p,s,k,d,t] · unitsize[p] · Δt[d,t]
    #                                 + Σ_{(p,s,k) ∈ out_eff(g)} v_flow[p,s,k,d,t] · slope[p,d,t] · unitsize[p]
    #                                                            · Δt[d,t]
    #                                 + Σ_{(p,s,k) ∈ out_eff(g), p ∈ process_min_load_eff}
    #                                       v_online[p,d,t] · section[p] · unitsize[p] · Δt[d,t]
    #                                 + Σ_{(p,s,k) ∈ back_out(g)} v_flow_back[p,s,k,d,t] · unitsize[p] · Δt[d,t]
    #                                 − Σ_{n ∈ N(g)} negative_inflow[n,d,t] )
    #   ∀ g ∈ groupNonSync, (d,t) ∈ dt, unless g has a time or period + time map without a value at (d,t)
    #   groupNonSync = groups with has_non_synchronous = yes
    #   N(g) = nodes of g;  icap = inv_group_cap;  ns_limit = non_synchronous_limit (constant, period, time or
    #   period + time); a group without any value, and an authored 0, read as 0 (the right-hand side then
    #   vanishes: no non-synchronous infeed without the penalised slack); a time step that a group's time map
    #   leaves out has no limit (no row; its slack is in no row and costs, so it stays 0)
    #   in_ns(g)      = arcs (p,s,k), k ∈ N(g), (p,k) ∈ process_sink_nonSync, (p,g) ∉ inside
    #   back_in_ns(g) = twoway1var_arcs (p,s,k), s ∈ N(g), (p,s) ∈ process_sink_nonSync, (p,g) ∉ inside
    #   out_*(g)      = arcs (p,s,k), s ∈ N(g), (p,g) ∉ inside
    #   back_out(g)   = twoway1var_arcs (p,s,k), k ∈ N(g), (p,g) ∉ inside
    #   process_sink_nonSync = unit arcs with is_non_synchronous ∪ (c,n) for is_DC connections c and their nodes n
    #   inside = process_group_inside_nonSync: processes with a source and a different sink node in g
    #   positive_inflow / negative_inflow = max(inflow_real, 0) / min(inflow_real, 0);  inflow_real = the node's
    #   inflow before the invest-solve energy margin
    # Non-synchronous infeed into the group (flagged unit arcs, DC connections,
    # positive exogenous inflow) is at most ns_limit times the group's supply,
    # measured as the energy its nodes deliver to processes, exports and the
    # real demand (an energy margin does not count); the excess is penalised.
    # Flows between nodes of the group are neither (inside).  The reverse flow
    # of a one-variable / DC arc arrives at the arc's source and leaves its
    # sink, lossless, as in nodeBalance_eq.  The slack has no icap, which makes
    # it a fraction of group_capacity_for_scaling.  slope already holds the
    # sink/source coefficient ratio of multi-arc processes.
    # Row rule (coarse steps): a group whose processes are all coarse gets one row per coarse step,
    # non_sync_constraint_coarse = the row's Δt-weighted mean over the step (_coarse_flows.add_cstr), on one slack
    # per step: the non-synchronous share then holds for the step's energy, with the inflow, the slope and the
    # limit through their step means (named in the coarse-flow behaviour notice).  Otherwise per solve step.
    lhs: dict = {"slack_neg": -(b.vq_non_synchronous * d.p_step_duration)}
    if sink_ns is not None and sink_ns.height > 0 and b.v_flow is not None:
        pss_in_g = _not_inside(_arcs_at_group(pss.join(sink_ns, on=["p", "sink"], how="inner"), "sink", gn_in_gns))
        if pss_in_g.height > 0:
            lhs["incoming_nonSync"] = Sum(Where(b.v_flow, pss_in_g) * p_us * d.p_step_duration * invc,
                                          over=("p", "source", "sink"))
    back = b.twoway1var_arcs if (b.v_flow_back is not None and _has_rows(b.twoway1var_arcs)) else None
    if back is not None and sink_ns is not None and sink_ns.height > 0:
        back_in = _not_inside(_arcs_at_group(back.join(sink_ns.rename({"sink": "source"}), on=["p", "source"],
                                                       how="inner"), "source", gn_in_gns))
        if back_in.height > 0:
            lhs["incoming_nonSync_back"] = Sum(Where(b.v_flow_back, back_in) * p_us * d.p_step_duration * invc,
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
    has_out_factor = pgns is not None

    def out_factor(arcs: pl.DataFrame) -> Param:
        # unitsize · Δt · ns_limit · icap, with unitsize first narrowed to the arcs' processes: the term's inner
        # join with the variable keeps only those, and the unnarrowed product is a (p, d, t, g) cross join over
        # every process.  Same cells, same multiplication order, so the coefficients are unchanged.  The keys
        # take unitsize's dtype (never the reverse), so the product's join dtypes stay as they were.
        p_dtype = p_us.frame.schema["p"]
        keys = arcs.select("p").unique()
        if keys.schema["p"] != p_dtype:
            keys = keys.select(pl.col("p").cast(p_dtype, strict=False)).drop_nulls()
        us = Param(p_us.dims, p_us.frame.join(keys, on="p", how="semi"), name=p_us.name)
        return us * d.p_step_duration * pgns * invc
    pss_noEff = d.process_source_sink_noEff
    if has_out_factor and pss_noEff is not None and pss_noEff.height > 0 and b.v_flow is not None:
        out_noEff = _not_inside(_arcs_at_group(pss_noEff, "source", gn_in_gns))
        if out_noEff.height > 0:
            rhs["out_noEff"] = Sum(Where(b.v_flow, out_noEff) * out_factor(out_noEff), over=("p", "source", "sink"))
    pss_eff = d.process_source_sink_eff
    if has_out_factor and pss_eff is not None and pss_eff.height > 0 and b.v_flow is not None:
        out_eff = _not_inside(_arcs_at_group(pss_eff, "source", gn_in_gns))
        if out_eff.height > 0:
            if d.p_slope is not None:
                term = Where(b.v_flow, out_eff) * d.p_slope * out_factor(out_eff)
            else:
                term = Where(b.v_flow, out_eff) * out_factor(out_eff)
            rhs["out_eff"] = Sum(term, over=("p", "source", "sink"))
            if (d.process_min_load_eff is not None and d.process_min_load_eff.height > 0
                    and d.p_section is not None):
                out_mle = out_eff.join(d.process_min_load_eff, on="p", how="inner")
                if out_mle.height > 0:
                    if b.v_online_linear is not None:
                        rhs["out_section_lin"] = Sum(
                            Where(b.v_online_linear, out_mle) * d.p_section * out_factor(out_mle),
                            over=("p", "source", "sink"))
                    if b.v_online_integer is not None:
                        rhs["out_section_int"] = Sum(
                            Where(b.v_online_integer, out_mle) * d.p_section * out_factor(out_mle),
                            over=("p", "source", "sink"))
    if has_out_factor and back is not None:
        back_out = _not_inside(_arcs_at_group(back, "sink", gn_in_gns))
        if back_out.height > 0:
            rhs["out_back"] = Sum(Where(b.v_flow_back, back_out) * out_factor(back_out), over=("p", "source", "sink"))
    p_neg = d.p_negative_inflow
    if p_neg is not None and pgns is not None and gn_in_gns.height > 0:
        # The limit keeps its authored dims ⊆ (g, d, t): promoted onto the solve's (d, t) for the eager join.
        lim = promote_param_to_dt(pgns, d.dt).rename({"value": "lim"}).collect()
        exo_out = (gn_in_gns.join(p_neg.frame.rename({"value": "v"}), on="n", how="inner")
                   .join(invc.frame.rename({"value": "iv"}), on=["g", "d"], how="inner")
                   .join(lim.select("g", "d", "t", "lim"), on=["g", "d", "t"], how="inner")
                   .with_columns(value=-pl.col("v") * pl.col("iv") * pl.col("lim"))
                   .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                   .select("g", "d", "t", "value"))
        if exo_out.height > 0:
            rhs["exo_demand"] = Param(("g", "d", "t"), exo_out)

    over = _non_sync_rows(d, pgns_all)
    _coarse_flows.add_row_rule_cstr(b, "non_sync_constraint", over=over, sense="<=", lhs_terms=lhs, rhs_terms=rhs,
                                    partition=_row_partition(b, "non_sync", over))


def _non_sync_rows(d, limit: "Param | None") -> pl.DataFrame:
    """``(g, d, t)`` of the non_sync_constraint rows: every (d, t) of a flagged group, less the time steps that its
    time or period + time map leaves out (no limit there).  A group without any value keeps every step (limit 0).

        rows = { (g,d,t) ∈ groupNonSync × dt : limit[g,d,t] has a value  or  g has no value at all }
    """
    over = d.groupNonSync.join(d.dt, how="cross")
    if limit is None or "t" not in limit.dims:
        return over
    cells = promote_param_to_dt(limit, d.dt).select("g", "d", "t").collect()
    cells, over_k = align_join_dtypes(cells, over, ("g", "d", "t"))
    valued = over_k.join(cells.unique(), on=["g", "d", "t"], how="semi")
    without = over_k.join(cells.select("g").unique(), on="g", how="anti")
    rows = pl.concat([valued, without])
    return rows.select(*(pl.col(c).cast(over.schema[c]) for c in over.columns))


# ---------------------------------------------------------------------------
# Objective


def objective_terms(b: _Build) -> Expr | None:
    """Penalties of the inertia, non-synchronous and capacity margin slacks."""
    if not (b.has_capacity_margin or b.has_inertia or b.has_non_sync):
        return None
    d = b.d

    # ─── OBJ. ─── inertia / non-synchronous / capacity margin slack ──────────
    # names: inertia_penalty, non_synchronous_penalty, capacity_margin_penalty
    # ─────────────────────────────────────────────────────────────────────────
    # + Σ_{(g,d,t)} vq_inertia[g,d,t] · inertia_limit[g,d,t] · penalty_inertia[g,d] · w_op[d,t]
    # + Σ_{(g,d,t)} vq_non_synchronous[g,d,t] · gcs[g,d] · penalty_non_synchronous[g,d] · w_op[d,t]
    # + Σ_{g ∈ groupCapacityMargin, d ∈ D_inv} vq_capacity_margin[g,d] · gcs[g,d] · penalty_capacity_margin[g,d]
    #                                                         · inflation_op[d] · bw[d] · 1000
    #   gcs = group_capacity_for_scaling;  w_op = b.op_factor
    #   bw = pd_branch_weight[d], 1 where it has no row (deterministic and pre-reveal periods)
    # Each slack is scaled back to physical units (MWs, MW) and priced per hour
    # of operation.  The capacity margin is a planning quantity: it is priced
    # once per period, weighted by pd_branch_weight, without step weights, and
    # its penalty is authored in CUR/kW (× 1000 to CUR/MW).  In a stochastic
    # fan each forecast branch period keeps its own slack and pays it with its
    # probability.  The output layer (calc_slacks) reports the same costs for
    # the realized periods (weight 1).
    pieces: list[Expr] = []
    if (b.vq_inertia is not None and d.pdtGroup_inertia_limit is not None
            and d.pdGroup_penalty_inertia is not None):
        pieces.append(Sum(b.vq_inertia * d.pdtGroup_inertia_limit * d.pdGroup_penalty_inertia * b.op_factor))
    if (b.vq_non_synchronous is not None and d.p_group_capacity_for_scaling is not None
            and d.pdGroup_penalty_non_synchronous is not None):
        pieces.append(Sum(b.vq_non_synchronous * d.p_group_capacity_for_scaling
                          * d.pdGroup_penalty_non_synchronous * b.op_factor))
    if (b.vq_capacity_margin is not None and d.p_group_capacity_for_scaling is not None
            and d.pdGroup_penalty_capacity_margin is not None and d.p_inflation_op is not None):
        cm_cost = (b.vq_capacity_margin * d.p_group_capacity_for_scaling
                   * d.pdGroup_penalty_capacity_margin * d.p_inflation_op * 1000.0)
        cm_bw = _capacity_margin_branch_weight(d)
        if cm_bw is not None:
            cm_cost = cm_cost * cm_bw
        pieces.append(Sum(cm_cost))
    if not pieces:
        return None
    total = pieces[0]
    for piece in pieces[1:]:
        total = total + piece
    return total
