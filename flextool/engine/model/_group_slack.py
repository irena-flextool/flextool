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
from flextool.engine.model import _balance_terms, _coarse_flows
from flextool.engine.model._common import _availability_factor, _has_rows, _indirect_sum_arcs, _var_view

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
    "pdGroup_capacity_margin_share", "pdGroup_peak_demand",
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

    # The capacity-margin groups' peak demand over each period's original timeline (share_of_peak;
    # preprocess.emit._capacity_margin_peak), MW.
    df = _provider_get(provider, sd / "pdGroup_peak_demand.csv")
    if df is not None and df.height > 0:
        out["pdGroup_peak_demand"] = Param(("g", "d"), df.rename({"group": "g", "period": "d"})
                                           .select("g", "d", pl.col("value").cast(pl.Float64)))

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
    """True when some group has capacity_margin_method manual or share_of_peak, the group scaling is populated and
    the solve has an investment period in dt.

    A group without a margin value has margin 0 (its capacity must cover its demand).  A solve without investment
    periods (dispatch, rolling dispatch) has no capacity margin."""
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

    # ─── VAR. ─── v_capacity_margin_slack_energy ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_capacity_margin_slack_energy[s,d] ≥ 0   ∀ s ∈ ∪_g S(g), d ∈ D_inv
    #   S(g) = the storage nodes with an upward balance slack that feed N(g) (capacity_margin_storage_slack_nodes)
    # The energy the upward balance slack of storage node s makes in period d, MWh (capacityMargin_slack_energy).
    # The capacity margin takes it back, so slack energy cannot pose as storage delivery.
    if b.has_capacity_margin and getattr(b, "vq_state_up", None) is not None:
        idx = _slack_energy_index(b)
        if idx is not None:
            b.v_capacity_margin_slack_energy = m.add_var("v_capacity_margin_slack_energy", ("n", "d"), idx,
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


def _utf8(frame: "pl.DataFrame | None", cols: tuple[str, ...]) -> "pl.DataFrame | None":
    """``frame`` with ``cols`` cast to Utf8; None when absent, empty or missing a column."""
    if frame is None or frame.height == 0 or any(c not in frame.columns for c in cols):
        return None
    return frame.select(*(pl.col(c).cast(pl.Utf8) for c in cols))


def _names(frame: "pl.DataFrame | None", col: str) -> list[str]:
    if frame is None or frame.height == 0 or col not in frame.columns:
        return []
    return sorted({str(v) for v in frame.get_column(col).to_list() if v is not None})


def capacity_margin_nodes(d) -> pl.DataFrame | None:
    """``(g, n)`` in ``group_node`` dtypes: N(g), the nodes of each capacity-margin group that are in
    ``nodeBalance`` and have no storage state (the nodes whose demand the margin covers); None without any."""
    gn = getattr(d, "group_node", None)
    gcm = getattr(d, "groupCapacityMargin", None)
    if gn is None or gcm is None or gn.height == 0 or gcm.height == 0:
        return None
    out = gn.filter(pl.col("g").cast(pl.Utf8).is_in(_names(gcm, "g")))
    balance = _names(getattr(d, "nodeBalance", None), "n")
    out = out.filter(pl.col("n").cast(pl.Utf8).is_in(balance))
    state = _names(getattr(d, "nodeState", None), "n")
    if state:
        out = out.filter(~pl.col("n").cast(pl.Utf8).is_in(state))
    out = out.select("g", "n").unique().sort("g", "n")
    return out if out.height > 0 else None


def capacity_margin_counted_arcs(d) -> pl.DataFrame | None:
    """``(g, p, source, sink, f)`` Utf8: the unit output arcs into N(g) whose capacity counts in g's capacity
    margin (``f`` the arc's upper_limit or fixed profile, null for a firm arc); None without any.

    A unit's output arc into N(g) (:func:`capacity_margin_nodes`) counts at capacity unless

    * the unit is storage-backed: it has an input from a node with a storage state and its other inputs cannot
      run it at full output alone (Σ input_share_max over them < 1, the default 1 each); or
    * it is a converter inside g: it has an input from a node of N(g) (a heat pump in an elec + heat group, an
      inverter fed from a DC bus of the group; a lossy one with every end in N(g) is left out of the row,
      :func:`capacity_margin_left_out_arcs`); or
    * the arc has conversion_flow_coeff 0 (uncapped: no capacity bounds it).

    Inputs with conversion_flow_coeff 0 carry no energy and are left out of both rules.  input_share_max is a
    constant, so a unit has one class in every period.  A counted arc with a profile is VRE, the others firm; the
    capacity they count (:func:`_counted_coefficient`) is the dispatch bound of the unit's flow into N(g): per arc
    for a unit with one output, once for the sum of its outputs into N(g) for a unit of maxOutputSum (a CHP with
    both outputs in g counts its capacity once).  The arcs that do not count enter the row by their flow.
    """
    arcs = capacity_margin_unit_arcs(d)
    if arcs is None:
        return None
    out = arcs.filter(pl.col("cls").is_in(["firm", "vre"])).select("g", "p", "source", "sink", "f")
    return out if out.height > 0 else None


def capacity_margin_unit_arcs(d) -> pl.DataFrame | None:
    """``(g, p, source, sink, f, cls)`` Utf8: every unit output arc into N(g) with its class in g's capacity
    margin (:func:`capacity_margin_counted_arcs`): ``firm`` / ``vre`` (counted at capacity; ``f`` the profile of a
    VRE arc), by flow: ``storage`` (a storage-backed unit), ``internal`` (a converter with an input from N(g))
    or ``uncapped`` (an arc with conversion_flow_coeff 0), or ``left_out`` (a lossy converter inside g,
    :func:`capacity_margin_left_out_arcs`); ``f`` is null for the arcs not counted at capacity.  The row
    (:func:`_eq_capacity_margin`) and the output breakdown (:func:`capacity_margin_report`) read these classes.
    None without any unit output arc into N(g)."""
    nodes = _utf8(capacity_margin_nodes(d), ("g", "n"))
    pss = _utf8(getattr(d, "process_source_sink", None), ("p", "source", "sink"))
    units = _names(getattr(d, "process_unit", None), "p")
    if nodes is None or pss is None or not units:
        return None
    pss = pss.filter(pl.col("p").is_in(units))
    out = pss.join(nodes.rename({"n": "sink"}), on="sink", how="inner")
    if out.height == 0:
        return None
    uncapped = _utf8(getattr(d, "process_source_sink_uncapped", None), ("p", "source", "sink"))
    inputs = pss.filter(pl.col("source") != pl.col("p")).select("p", "source").unique()
    uncapped_arcs = None
    if uncapped is not None:
        zero_in = uncapped.filter(pl.col("sink") == pl.col("p")).select("p", "source").unique()
        inputs = inputs.join(zero_in, on=["p", "source"], how="anti")
        uncapped_arcs = out.join(uncapped, on=["p", "source", "sink"], how="semi").select("g", "p", "source", "sink")
    state = _names(getattr(d, "nodeState", None), "n")
    share = _utf8(_param_frame(getattr(d, "p_process_source_input_share_max", None)), ("p", "source"))
    shares = inputs.with_columns(share=pl.lit(1.0))
    if share is not None:
        values = getattr(d, "p_process_source_input_share_max").frame
        values = values.collect() if isinstance(values, pl.LazyFrame) else values
        shares = (inputs.join(values.select(pl.col("p").cast(pl.Utf8), pl.col("source").cast(pl.Utf8),
                                            pl.col("value").cast(pl.Float64).alias("share")),
                              on=["p", "source"], how="left")
                  .with_columns(pl.col("share").fill_null(1.0)))
    is_state = pl.col("source").is_in(state)
    storage_backed = (shares.group_by("p")
                      .agg(is_state.any().alias("has_state"),
                           pl.col("share").filter(~is_state).sum().alias("other"))
                      .filter(pl.col("has_state") & (pl.col("other") < 1.0))
                      .select("p"))
    internal = inputs.join(nodes.rename({"n": "source"}), on="source", how="inner").select("g", "p").unique()
    arcs = out.select("g", "p", "source", "sink")
    # The by-flow classes: a storage-backed unit first, then a converter inside g, then an uncapped arc.  A lossy
    # converter inside g (capacity_margin_left_out_arcs) is left out of the row.
    lossy = _lossy_internal_units(d, nodes, pss, inputs)
    by_flow = [arcs.join(storage_backed, on="p", how="semi").with_columns(cls=pl.lit("storage"))]
    rest = arcs.join(storage_backed, on="p", how="anti")
    if lossy is not None:
        by_flow.append(rest.join(lossy, on=["g", "p"], how="semi").with_columns(cls=pl.lit("left_out")))
        rest = rest.join(lossy, on=["g", "p"], how="anti")
    by_flow.append(rest.join(internal, on=["g", "p"], how="semi").with_columns(cls=pl.lit("internal")))
    rest = rest.join(internal, on=["g", "p"], how="anti")
    if uncapped_arcs is not None:
        by_flow.append(rest.join(uncapped_arcs, on=["g", "p", "source", "sink"], how="semi")
                       .with_columns(cls=pl.lit("uncapped")))
        rest = rest.join(uncapped_arcs, on=["g", "p", "source", "sink"], how="anti")
    profiles = [f for f in (_utf8(getattr(d, "process_profile_upper", None), ("p", "source", "sink", "f")),
                            _utf8(getattr(d, "process_profile_fixed", None), ("p", "source", "sink", "f")))
                if f is not None]
    if profiles:
        counted = rest.join(pl.concat(profiles).unique(), on=["p", "source", "sink"], how="left")
    else:
        counted = rest.with_columns(f=pl.lit(None, dtype=pl.Utf8))
    counted = counted.with_columns(cls=pl.when(pl.col("f").is_null()).then(pl.lit("firm")).otherwise(pl.lit("vre")))
    flow = pl.concat(by_flow).with_columns(f=pl.lit(None, dtype=pl.Utf8)).select("g", "p", "source", "sink", "f",
                                                                                   "cls")
    return (pl.concat([counted.select("g", "p", "source", "sink", "f", "cls"), flow]).unique()
            .sort("g", "p", "source", "sink", "f", nulls_last=True))


def _unit_inputs(d, pss: pl.DataFrame) -> pl.DataFrame:
    """``(p, source)`` Utf8: the input arcs of the units in ``pss`` (Utf8 p, source, sink) that carry energy (an
    input with conversion_flow_coeff 0 is left out)."""
    inputs = pss.filter(pl.col("source") != pl.col("p")).select("p", "source").unique()
    uncapped = _utf8(getattr(d, "process_source_sink_uncapped", None), ("p", "source", "sink"))
    if uncapped is not None:
        zero_in = uncapped.filter(pl.col("sink") == pl.col("p")).select("p", "source").unique()
        inputs = inputs.join(zero_in, on=["p", "source"], how="anti")
    return inputs


def _lossy_internal_units(d, nodes: pl.DataFrame, pss: pl.DataFrame,
                          inputs: pl.DataFrame) -> "pl.DataFrame | None":
    """``(g, p)`` Utf8: the lossy converters inside g, left out of g's capacity margin.  A unit is one when

    * every input that carries energy (``inputs``) comes from a node of N(g) and every output goes into N(g), with
      at least one of each (an electrolyser into a hydrogen node of g, an electric boiler into a heat node of g);
    * its efficiency at full load is below 1 at some step: slope + section > 1 (``p_slope``, ``p_section``; the
      input per unit of output at full load, 1 / efficiency).

    Such a unit enters the row with output − input = −its losses (and its min-load section), which shedding the
    demand it serves would lower; left out, the demand it serves counts one to one.  A unit with efficiency ≥ 1 at
    every step (a heat pump) stays in the row by its flow: its input is what the demand it serves needs.
    ``nodes`` (g, n), ``pss`` (p, source, sink) of the units and ``inputs`` (p, source) are Utf8."""
    outputs = pss.filter(pl.col("sink") != pl.col("p")).select("p", pl.col("sink").alias("n")).unique()
    ins = inputs.select("p", pl.col("source").alias("n"))
    ends = pl.concat([ins, outputs]).unique()
    if ends.height == 0:
        return None
    cand = (nodes.join(ins.select("p", "n").unique(), on="n", how="inner").select("g", "p")
            .join(nodes.join(outputs, on="n", how="inner").select("g", "p"), on=["g", "p"], how="semi").unique())
    if cand.height == 0:
        return None
    # Every end of the unit in N(g): no end outside g.
    outside = (cand.join(ends, on="p", how="inner")
               .join(nodes.with_columns(_in=pl.lit(True)), on=["g", "n"], how="left")
               .filter(pl.col("_in").is_null()).select("g", "p").unique())
    cand = cand.join(outside, on=["g", "p"], how="anti")
    if cand.height == 0:
        return None
    slope = _param_frame(getattr(d, "p_slope", None))
    if slope is None or slope.height == 0:
        return None
    full = slope.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                        pl.col("value").cast(pl.Float64).alias("slope"))
    section = _param_frame(getattr(d, "p_section", None))
    if section is not None and section.height > 0:
        full = (full.join(section.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                         pl.col("t").cast(pl.Utf8), pl.col("value").cast(pl.Float64).alias("sec")),
                          on=["p", "d", "t"], how="left")
                .with_columns(slope=pl.col("slope") + pl.col("sec").fill_null(0.0)))
    lossy_p = full.group_by("p").agg(pl.col("slope").max()).filter(pl.col("slope") > 1.0 + 1e-12).select("p")
    out = cand.join(lossy_p, on="p", how="semi").sort("g", "p")
    return out if out.height > 0 else None


def capacity_margin_left_out_arcs(d) -> pl.DataFrame | None:
    """``(g, p, source, sink)`` Utf8: the arcs left out of g's capacity margin, at both of their ends:

    * the connections inside g: both ends in N(g) (:func:`capacity_margin_nodes`), both directions;
    * the lossy converters inside g (:func:`_lossy_internal_units`): their inputs from and outputs into N(g).

    Their losses are not part of the requirement: the demand they serve inside g counts one to one.  A row that kept
    them would require the losses of the dispatch at each step, and shedding the demand behind a lossy path would
    lower that requirement by more than it raises the unserved energy (the LP sheds to hold the margin).  None
    without any."""
    nodes = _utf8(capacity_margin_nodes(d), ("g", "n"))
    pss = _utf8(getattr(d, "process_source_sink", None), ("p", "source", "sink"))
    if nodes is None or pss is None:
        return None
    units = _names(getattr(d, "process_unit", None), "p")
    conns = pss.filter(~pl.col("p").is_in(units) & (pl.col("source") != pl.col("p")) & (pl.col("sink") != pl.col("p")))
    parts = [conns.join(nodes.rename({"n": "source"}), on="source", how="inner")
             .join(nodes.rename({"n": "sink"}), on=["g", "sink"], how="inner")
             .select("g", "p", "source", "sink")]
    upss = pss.filter(pl.col("p").is_in(units))
    if upss.height > 0:
        lossy = _lossy_internal_units(d, nodes, upss, _unit_inputs(d, upss))
        if lossy is not None:
            arcs = upss.join(lossy, on="p", how="inner")
            at_in = arcs.join(nodes.rename({"n": "source"}), on=["g", "source"], how="semi")
            at_out = arcs.join(nodes.rename({"n": "sink"}), on=["g", "sink"], how="semi")
            parts.append(pl.concat([at_in, at_out]).select("g", "p", "source", "sink"))
    out = pl.concat(parts).unique().sort("g", "p", "source", "sink")
    return out if out.height > 0 else None


def capacity_margin_storage_slack_nodes(d) -> pl.DataFrame | None:
    """``(g, n)`` Utf8: the storage nodes whose balance slack could pose as storage delivery in g's capacity
    margin: a node with a storage state and an upward balance slack (``penalty_method`` not off) that is an input
    of a storage-backed unit into N(g) (:func:`capacity_margin_unit_arcs` class ``storage``) or the other end of a
    connection at N(g) that is not left out.  The row takes back the slack energy made at such a node in the
    period (``v_capacity_margin_slack_energy``).  None without any."""
    nodes = _utf8(capacity_margin_nodes(d), ("g", "n"))
    pss = _utf8(getattr(d, "process_source_sink", None), ("p", "source", "sink"))
    state = _names(getattr(d, "nodeState", None), "n")
    if nodes is None or pss is None or not state:
        return None
    parts: list[pl.DataFrame] = []
    unit_arcs = capacity_margin_unit_arcs(d)
    if unit_arcs is not None:
        backed = unit_arcs.filter(pl.col("cls") == "storage").select("g", "p").unique()
        ins = pss.filter((pl.col("source") != pl.col("p")) & pl.col("source").is_in(state))
        parts.append(backed.join(ins, on="p", how="inner").select("g", pl.col("source").alias("n")))
    units = _names(getattr(d, "process_unit", None), "p")
    conns = pss.filter(~pl.col("p").is_in(units))
    for here, there in (("source", "sink"), ("sink", "source")):
        at = conns.join(nodes.rename({"n": here}), on=here, how="inner").filter(pl.col(there).is_in(state))
        parts.append(at.select("g", pl.col(there).alias("n")))
    if not parts:
        return None
    balance = _names(getattr(d, "nodeBalance", None), "n")
    off = _names(getattr(d, "nodeBalance_penalty_off", None), "n")
    out = (pl.concat(parts).unique()
           .filter(pl.col("n").is_in(balance) & ~pl.col("n").is_in(off))
           .sort("g", "n"))
    return out if out.height > 0 else None


def _param_frame(param) -> "pl.DataFrame | None":
    if param is None:
        return None
    frame = param.frame
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def resolve_capacity_margin(d) -> None:
    """The margin of the ``share_of_peak`` groups into ``pdGroup_capacity_margin`` [MW]:

        M[g,d] = capacity_margin[g,d]                              g with capacity_margin_method manual
               = capacity_margin_share[g,d] · P[g,d]                g with capacity_margin_method share_of_peak
        P[g,d] = pdGroup_peak_demand (preprocess.emit._capacity_margin_peak): the peak demand of N(g) over the
                 period's original timeline

    The row, the group scaling, the Benders split and the outputs then read one MW value whatever the method.  A
    share_of_peak group whose share is positive but whose peak is 0 (no demand) gets a warning: its margin is 0.
    Rows of 0 are dropped (margin 0).  Idempotent: a group's rows are replaced, not added to.
    """
    share = _param_frame(getattr(d, "pdGroup_capacity_margin_share", None))
    if share is None or share.height == 0:
        return
    peak = _param_frame(getattr(d, "pdGroup_peak_demand", None))
    share_u = share.select(pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                           pl.col("value").cast(pl.Float64).alias("share"))
    if peak is None:
        peak_u = pl.DataFrame(schema={"g": pl.Utf8, "d": pl.Utf8, "peak": pl.Float64})
    else:
        peak_u = peak.select(pl.col("g").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                             pl.col("value").cast(pl.Float64).alias("peak"))
    joined = share_u.join(peak_u, on=["g", "d"], how="left").with_columns(pl.col("peak").fill_null(0.0))
    no_peak = sorted(set(joined.filter((pl.col("share") != 0.0) & (pl.col("peak") <= 0.0))["g"].to_list()))
    if no_peak:
        from flextool.engine import _behaviour_changes

        _behaviour_changes.warn_once(
            ("capacity_margin_share_no_peak", tuple(no_peak)),
            f"Warning: capacity margin group(s) {', '.join(no_peak)} have capacity_margin_method share_of_peak but "
            "no demand in some investment period: the margin there is 0.")
    margin = (joined.with_columns(value=pl.col("share") * pl.col("peak"))
              .filter(pl.col("value") != 0.0).select("g", "d", "value"))
    groups = sorted(set(share_u["g"].to_list()))
    current = _param_frame(getattr(d, "pdGroup_capacity_margin", None))
    if current is not None:
        g_dtype, d_dtype = current.schema["g"], current.schema["d"]
        keep = current.filter(~pl.col("g").cast(pl.Utf8).is_in(groups))
        margin = margin.select(pl.col("g").cast(g_dtype), pl.col("d").cast(d_dtype),
                               pl.col("value").cast(current.schema["value"]))
        frame = pl.concat([keep.select("g", "d", "value"), margin]).sort("g", "d")
    else:
        g_dtype, d_dtype = share.schema["g"], share.schema["d"]
        frame = margin.select(pl.col("g").cast(g_dtype), pl.col("d").cast(d_dtype), "value").sort("g", "d")
    d.pdGroup_capacity_margin = Param(("g", "d"), frame) if frame.height > 0 else None


def _eq_capacity_margin(b: _Build) -> None:
    """capacityMargin: the firm and VRE capacity of a group covers its demand plus the margin, with the group's
    other flows by their value."""
    if not b.has_capacity_margin:
        return
    d = b.d
    gcm = d.groupCapacityMargin
    # Earlier releases counted the margin differently (behaviour-change notice; printed only while the input
    # database carries the marker).
    from flextool.engine import _behaviour_changes

    _behaviour_changes.notice("capacity_margin_counts_by_flow", "groups", _names(gcm, "g"))
    invc = d.p_inv_group_cap
    p_us = d.p_unitsize
    exist_cnt = d.p_process_existing_count

    # ─── EQ. ─── capacityMargin ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k) ∈ K(g)} (Σ_{d′} v_invest_p[p,d′] − Σ_{d′} v_divest_p[p,d′]) · unitsize[p] · κ[g,p,s,k,d,t] · icap[g,d]
    #   + Σ_{n ∈ N(g)} B¬FV[g,n,d,t] · icap[g,d]
    #   − Σ_{n ∈ N(g)} vq_state_down[n,d,t] · icap[g,d]
    #   − Σ_{s ∈ S(g)} v_capacity_margin_slack_energy[s,d] / Δt[d,t] · icap[g,d]
    #   + vq_capacity_margin[g,d]
    #   ≥ (M[g,d] + Σ_{n ∈ N(g)} demand[n,d,t]) · icap[g,d]
    #     − Σ_{(p,s,k) ∈ K(g)} existing_count[p,d] · unitsize[p] · κ[g,p,s,k,d,t] · icap[g,d]
    #   ∀ g ∈ groupCapacityMargin, (d,t) ∈ dt : d ∈ D_inv
    #   groupCapacityMargin = groups with capacity_margin_method manual or share_of_peak
    #   D_inv = period_invest ∩ periods of dt (investment periods of the solve; a period label has no dt, so a
    #   fan contributes its forecast branch periods, one row per forecast); a solve without investment periods
    #   has no capacity margin
    #   N(g) = nodes of g in nodeBalance without a storage state;  icap = inv_group_cap = 1 / group_capacity_for_scaling
    #   F(g), V(g) = the unit output arcs into N(g) that count at capacity (capacity_margin_counted_arcs): not of a
    #   storage-backed unit (an input from a storage-state node, Σ input_share_max of its other inputs < 1), not of
    #   a unit with an input from N(g), not conversion_flow_coeff 0; V(g) those with an upper_limit or fixed profile
    #   K(g) = F(g) ∪ V(g), except that a unit of maxOutputSum (process_indirect) keeps only its first arc into N(g)
    #   κ = c · av (firm arc);  min(c, profile) · av (VRE arc);  min(av, Σ_{its arcs into N(g)} κ[arc]) (a unit of
    #   maxOutputSum: its outputs share one capacity);  c = capacity_max_coeff (p_arc_max_cap_coef, default 1),
    #   av = availability (default 1), profile = the smallest value of the arc's upper_limit and fixed profiles
    #   B¬FV[g,n,d,t] = the flow terms of nodeBalance_eq at n (balance_flow_terms, power: no Δt) without the arcs
    #   of F(g) ∪ V(g) and without the arcs left out (capacity_margin_left_out_arcs: the connections with both ends
    #   in N(g), the lossy converters inside g); the other units' output and input (slope, section), connections at
    #   N(g) (v_flow, v_flow_back), Benders half-flows; only at the (g, n) pairs of N(g)
    #   S(g) = the storage nodes with an upward balance slack that feed N(g) through a storage-backed unit or a
    #   connection (capacity_margin_storage_slack_nodes);  v_capacity_margin_slack_energy = Σ_t vq_state_up · Δt
    #   (capacityMargin_slack_energy)
    #   demand = −inflow_real / Δt (pdtNodeInflow_per_step: the inflow before any energy margin)
    #   M[g,d] = capacity_margin[g,d] (manual)  |  capacity_margin_share[g,d] · P[g,d] (share_of_peak), 0 where not
    #   authored;  P[g,d] = max(0, max_t Σ_{n ∈ N(g)} demand[n,d,t]) over the period's original time series
    #   (before representative periods and new_stepduration; _capacity_margin_peak, resolve_capacity_margin)
    #   Σ_{d′} over edd_invest_set / edd_divest_active
    # Substituting the balance of each n ∈ N(g) (flow_FV + B¬FV + q_up − q_down = demand) gives
    #   Σ_{K} (counted capacity − flow into N(g)) + vq ≥ M + Σ_n q_up − (slack energy of S(g)) / Δt:
    # only firm and VRE headroom holds the margin.  Each counted capacity bounds the unit's dispatch into N(g)
    # (maxFlow, profile_flow_*, maxOutputSum), so the headroom is never negative.  Storage-backed units, imports,
    # exports and the group's own consumption count by their flow: imports and storage help by displacing firm
    # output, exports add to the requirement.  Energy dumped with vq_state_down does not count, and neither does
    # energy that a storage node's balance slack makes (it could be stored and sent at the binding step).  Lines
    # and lossy converters inside g are left out, so their losses are not part of the requirement and shedding the
    # demand behind them cannot lower it: unserved energy (q_up) leaves the row unchanged where counted capacity
    # served it, and is priced once, by penalty_up.  Units fed from fuel or commodity nodes count at capacity: the
    # fuel is assumed to last through the peak.  Existing capacity is a constant and moves to the right-hand side.
    # The row is scaled by icap, which makes the slack a fraction of group_capacity_for_scaling.
    # A retirement v_divest[e,d′] counts in d iff (e,d′,d) ∈ edd_divest_active (ndd_divest_active):
    #   d′ ≤ d < EOL[e], the end of life of e's pre-existing capacity (none for reinvest_automatic).
    # Row rule (coarse steps): a group whose flow members (the processes of B¬FV) are all coarse processes gets one
    # row per coarse step, capacityMargin_coarse = the row's Δt-weighted mean over the step
    # (_coarse_flows.add_cstr): availability, the profile, the inflow and the slope act through their step means,
    # so the margin covers the step-mean demand, not its peak (named in the coarse-flow behaviour notice).
    # Otherwise the rows stay per solve step.
    lhs: dict = {"slack": b.vq_capacity_margin}
    rhs: dict = {}
    dt_inv = d.dt.join(_capacity_margin_periods(d), on="d", how="semi")
    gn = capacity_margin_nodes(d)

    # arcs: (g, p, source, sink) in process_source_sink dtypes, the counted arcs (F ∪ V, left out of B¬FV);
    # kappa: (g, p, source, sink, d, t) on the arcs of K(g), κ (_counted_coefficient)
    arcs, kappa = _counted_terms(d, gn, dt_inv)

    if arcs is not None:
        invest, divest = _invest_change(b, arcs, "cap_margin")
        if invest is not None:
            lhs["cap_invest"] = Sum(Where(invest, arcs) * p_us * kappa * invc, over=("p", "source", "sink"))
        if divest is not None:
            lhs["cap_divest"] = Sum(-(Where(divest, arcs) * p_us * kappa * invc), over=("p", "source", "sink"))
        ex = pl.DataFrame() if exist_cnt is None else (
              arcs.join(exist_cnt.frame.rename({"value": "exist"}), on="p", how="inner")
              .join(p_us.frame.rename({"value": "us"}), on="p", how="inner")
              .join(kappa.frame.rename({"value": "kappa"}), on=["g", "p", "source", "sink", "d"], how="inner")
              .join(invc.frame.rename({"value": "inv"}), on=["g", "d"], how="inner")
              .with_columns(value=-(pl.col("exist") * pl.col("us") * pl.col("kappa") * pl.col("inv")))
              .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
              .select("g", "d", "t", "value").sort("g", "d", "t"))
        if ex.height > 0:
            rhs["existing_neg"] = Param(("g", "d", "t"), ex)

    if gn is not None and b.v_flow is not None:
        exclude = _excluded_arcs(d, gn, arcs)
        flows = _balance_terms.balance_flow_terms(b, nodes=gn, group_nodes=gn, exclude_arcs=exclude, per_step=True)
        for key, term in flows.items():
            lhs[f"flow_{key}"] = Sum(term, over=("n",)) * invc
        if getattr(b, "vq_state_down", None) is not None:
            lhs["dump_neg"] = -(Sum(Where(b.vq_state_down, gn), over=("n",)) * invc)
    slack_energy = _slack_energy_term(b, dt_inv)
    if slack_energy is not None:
        lhs["storage_slack_neg"] = -(slack_energy * invc)

    if d.pdGroup_capacity_margin is not None:
        rhs["floor"] = d.pdGroup_capacity_margin * invc
    inflow_per = getattr(d, "pdtNodeInflow_per_step", None)
    if inflow_per is not None and gn is not None:
        rhs_inflow = (join_aligned(gn, inflow_per.frame.rename({"value": "inflow"}), on="n", how="inner")
                      .join(invc.frame.rename({"value": "inv"}), on=["g", "d"], how="inner")
                      .with_columns(value=-pl.col("inflow") * pl.col("inv"))
                      .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                      .select("g", "d", "t", "value").sort("g", "d", "t"))
        if rhs_inflow.height > 0:
            rhs["inflow_neg"] = Param(("g", "d", "t"), rhs_inflow)

    _eq_slack_energy(b)
    over = gcm.join(dt_inv, how="cross")
    _coarse_flows.add_row_rule_cstr(b, "capacityMargin", over=over, sense=">=", lhs_terms=lhs, rhs_terms=rhs,
                                    partition=_row_partition(b, "capacity_margin", over))


def _excluded_arcs(d, gn: pl.DataFrame, counted: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``(g, p, source, sink)`` in ``process_source_sink`` / ``group_node`` dtypes: the arcs left out of g's B¬FV,
    the counted arcs (``counted``, :func:`_counted_terms`) and :func:`capacity_margin_left_out_arcs`."""
    parts = [] if counted is None else [counted.select("g", "p", "source", "sink")]
    left_out = capacity_margin_left_out_arcs(d)
    if left_out is not None:
        pss = d.process_source_sink.select("p", "source", "sink")
        at = pl.concat([_arcs_at_group(pss, side, gn).select("g", "p", "source", "sink") for side in ("source", "sink")])
        at = (at.with_columns(_g=pl.col("g").cast(pl.Utf8), _p=pl.col("p").cast(pl.Utf8),
                              _s=pl.col("source").cast(pl.Utf8), _k=pl.col("sink").cast(pl.Utf8))
              .join(left_out.rename({"g": "_g", "p": "_p", "source": "_s", "sink": "_k"}),
                    on=["_g", "_p", "_s", "_k"], how="semi")
              .select("g", "p", "source", "sink"))
        if at.height > 0:
            parts.append(at)
    if not parts:
        return None
    return pl.concat(parts).unique().sort("g", "p", "source", "sink")


def _slack_energy_term(b: _Build, dt_inv: pl.DataFrame) -> "Expr | None":
    """``Σ_{s ∈ S(g)} v_capacity_margin_slack_energy[s,d] / Δt[d,t]``, open dims (g, d, t), for the rows at
    ``dt_inv`` (:func:`capacity_margin_storage_slack_nodes`); None without any."""
    e = getattr(b, "v_capacity_margin_slack_energy", None)
    gs = _capacity_margin_slack_pairs(b.d)
    if e is None or gs is None:
        return None
    sd = _param_frame(b.d.p_step_duration)
    inv_dt = (sd.join(dt_inv.select("d", "t"), on=["d", "t"], how="semi")
              .with_columns(value=1.0 / pl.col("value")).select("d", "t", "value"))
    return Sum(Where(e, gs) * Param(("d", "t"), inv_dt), over=("n",))


def _capacity_margin_slack_pairs(d) -> "pl.DataFrame | None":
    """:func:`capacity_margin_storage_slack_nodes` as ``(g, n)`` in ``group_node`` dtypes; None without any."""
    pairs = capacity_margin_storage_slack_nodes(d)
    gn = getattr(d, "group_node", None)
    if pairs is None or gn is None:
        return None
    native = gn.select("g").unique().join(_node_axis(d), how="cross")
    out = (native.with_columns(_g=pl.col("g").cast(pl.Utf8), _n=pl.col("n").cast(pl.Utf8))
           .join(pairs.rename({"g": "_g", "n": "_n"}), on=["_g", "_n"], how="semi").select("g", "n")
           .sort("g", "n"))
    return out if out.height > 0 else None


def _slack_energy_index(b: _Build) -> "pl.DataFrame | None":
    """``(n, d)`` in ``vq_state_up`` dtypes: the storage nodes of :func:`capacity_margin_storage_slack_nodes` that
    have an upward balance slack, × the capacity margin's periods; None without any."""
    pairs = capacity_margin_storage_slack_nodes(b.d)
    if pairs is None:
        return None
    periods = _names(_capacity_margin_periods(b.d), "d")
    idx = (b.vq_state_up.frame.select("n", "d").unique()
           .filter(pl.col("n").cast(pl.Utf8).is_in(_names(pairs, "n")) & pl.col("d").cast(pl.Utf8).is_in(periods))
           .sort("n", "d"))
    return idx if idx.height > 0 else None


def _eq_slack_energy(b: _Build) -> None:
    """capacityMargin_slack_energy: the slack energy of the storage nodes that feed a capacity-margin group."""
    e = getattr(b, "v_capacity_margin_slack_energy", None)
    if e is None:
        return
    d = b.d
    idx = e.frame.select("n", "d")

    # ─── EQ. ─── capacityMargin_slack_energy ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_capacity_margin_slack_energy[s,d] = Σ_{t : (d,t) ∈ dt} vq_state_up[s,d,t] · Δt[d,t]
    #   ∀ (s,d) of v_capacity_margin_slack_energy
    # The energy made by storage node s's upward balance slack over the period's steps.  Energy made at any step
    # can be stored and sent into the group at the binding step, so the capacityMargin row takes back the whole
    # period's slack energy, in MW at its own step (÷ Δt): one variable per (s,d) keeps the rows sparse.
    m_vq = Sum(Where(b.vq_state_up, idx) * d.p_step_duration, over=("t",))
    b.m.add_cstr("capacityMargin_slack_energy", over=idx, sense="==", lhs_terms={"energy": e},
                 rhs_terms={"slack": m_vq})


def _node_axis(d) -> pl.DataFrame:
    """``(n)``: the nodes of ``nodeBalance`` in its dtype."""
    return d.nodeBalance.select("n").unique()


def _counted_terms(d, gn: "pl.DataFrame | None", dt_inv: pl.DataFrame) -> "tuple[pl.DataFrame | None, Param | None]":
    """``(arcs, kappa)`` of the capacity margin's counted arcs (:func:`capacity_margin_counted_arcs`) at the steps
    ``dt_inv``: ``arcs`` (g, p, source, sink) in ``process_source_sink`` dtypes, ``kappa`` their capacity factor
    (:func:`_counted_coefficient`); ``(None, None)`` without any (or without the existing count / unitsize).
    One construction for the row (:func:`_eq_capacity_margin`) and its output breakdown
    (:func:`capacity_margin_report`)."""
    counted = capacity_margin_counted_arcs(d)
    if counted is None or gn is None or d.p_unitsize is None:
        return None, None
    pss = d.process_source_sink.select("p", "source", "sink")
    arcs = (_arcs_at_group(pss, "sink", gn)
            .with_columns(_g=pl.col("g").cast(pl.Utf8), _p=pl.col("p").cast(pl.Utf8),
                          _s=pl.col("source").cast(pl.Utf8), _k=pl.col("sink").cast(pl.Utf8))
            .join(counted.rename({"g": "_g", "p": "_p", "source": "_s", "sink": "_k", "f": "_f"}),
                  on=["_g", "_p", "_s", "_k"], how="inner"))
    kappa = _counted_coefficient(d, arcs, dt_inv, d.p_profile_value)
    arcs = arcs.select("g", "p", "source", "sink").unique().sort("g", "p", "source", "sink")
    if kappa is None or arcs.height == 0:
        return None, None
    return arcs, kappa


def _counted_coefficient(d, arcs: pl.DataFrame, dt_inv: pl.DataFrame, prof_val: "Param | None") -> "Param | None":
    """``Param(("g", "p", "source", "sink", "d", "t"))``: the capacity factor of each counted unit of ``arcs`` (with
    the Utf8 key columns ``_p``, ``_s``, ``_k`` and the profile ``_f``) at the steps ``dt_inv``, the bound the
    dispatch puts on its flow into N(g):

        κ[arc] = c[p,s,k] · av[p,d,t]                                firm arc (no profile)
               = min(c[p,s,k], min_f profile[f,d,t]) · av[p,d,t]      VRE arc (its upper_limit and fixed profiles f)
        κ[p]   = min(av[p,d,t], Σ_{arcs of p into N(g)} κ[arc])        a unit of maxOutputSum (process_indirect)

    c = p_arc_max_cap_coef (default 1), av = availability (default 1).  maxFlow (and profile_flow_*) bound each
    arc by κ[arc]; maxOutputSum bounds the sum of an indirect unit's outputs by av, so such a unit counts once, at
    κ[p], on the first of its arcs into N(g) (sorted by sink); its other arcs into N(g) get no row (they are still
    left out of B¬FV).  Every unit's counted capacity is thus at least the flow it can send into N(g)."""
    base = (arcs.select("g", "p", "source", "sink", "_p", "_s", "_k", "_f")
            .join(dt_inv.select("d", "t"), how="cross")
            .with_columns(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8)))
    if d.p_process_availability is not None:
        av = _availability_factor(d, arcs.select("p").unique()).frame
        base = (base.join(av.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                    pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").alias("av")),
                          on=["_p", "_d", "_t"], how="left")
                .with_columns(pl.col("av").fill_null(1.0)))
    else:
        base = base.with_columns(av=pl.lit(1.0))
    coef = getattr(d, "p_arc_max_cap_coef", None)
    if coef is not None:
        cf = _param_frame(coef)
        base = (base.join(cf.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("source").cast(pl.Utf8).alias("_s"),
                                    pl.col("sink").cast(pl.Utf8).alias("_k"), pl.col("value").alias("c")),
                          on=["_p", "_s", "_k"], how="left")
                .with_columns(pl.col("c").fill_null(1.0)))
    else:
        base = base.with_columns(c=pl.lit(1.0))
    keys = ["g", "p", "source", "sink", "d", "t"]
    firm = base.filter(pl.col("_f").is_null()).with_columns(value=pl.col("c") * pl.col("av"))
    parts = [firm.select(*keys, "_p", "_k", "av", "value")]
    vre = base.filter(pl.col("_f").is_not_null())
    if vre.height > 0:
        if prof_val is None:
            raise ValueError("capacityMargin: profile-limited arcs without p_profile_value")
        pf = _param_frame(prof_val)
        vre = (vre.join(pf.select(pl.col("f").cast(pl.Utf8).alias("_f"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                  pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").alias("prof")),
                        on=["_f", "_d", "_t"], how="inner")
               .group_by(*keys)
               .agg(pl.col("_p").first(), pl.col("_k").first(), pl.col("av").first(),
                    (pl.min_horizontal(pl.col("prof").min(), pl.col("c").first()) * pl.col("av").first())
                    .alias("value")))
        parts.append(vre.select(*keys, "_p", "_k", "av", "value"))
    frame = pl.concat(parts)
    sum_arcs = _indirect_sum_arcs(d)
    indirect = _names(sum_arcs, "p") if sum_arcs is not None else []
    if indirect:
        is_ind = pl.col("_p").is_in(indirect)
        ind = (frame.filter(is_ind).sort("g", "_p", "d", "t", "_k")
               .group_by("g", "_p", "d", "t", maintain_order=True)
               .agg(pl.col("p").first(), pl.col("source").first(), pl.col("sink").first(),
                    pl.min_horizontal(pl.col("av").first(), pl.col("value").sum()).alias("value")))
        frame = pl.concat([frame.filter(~is_ind).select(*keys, "value"), ind.select(*keys, "value")])
    else:
        frame = frame.select(*keys, "value")
    frame = frame.sort(*keys)
    return Param(tuple(keys), frame) if frame.height > 0 else None


# ---------------------------------------------------------------------------
# The capacity margin's breakdown of a solution (outputs)

#: The classes of :func:`capacity_margin_report` (the ``class`` column of its per-entity frame) -> the summary
#: column they add to.  Each value is the class's signed contribution to the row's left-hand side [MW].
CAPACITY_MARGIN_CLASSES: dict[str, str] = {
    "firm": "firm", "vre": "vre", "storage": "storage_flow", "storage_slack": "storage_flow", "unit_by_flow": "unit_flow",
    "connection": "connections_net", "consumption": "consumption", "dumped": "dumped",
}


def _as_utf8(frame: "pl.DataFrame | pl.LazyFrame | None") -> "pl.DataFrame | None":
    """``frame`` collected, its key columns (all but ``value``) cast to Utf8; None when absent."""
    if frame is None:
        return None
    frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return frame.with_columns(*(pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value"))


def _param_utf8(param, cols: tuple[str, ...], alias: str) -> "pl.DataFrame | None":
    """``param``'s frame as ``(*cols, alias)``, keys Utf8 and the value Float64; None when absent."""
    frame = _as_utf8(None if param is None else param.frame)
    if frame is None or any(c not in frame.columns for c in cols):
        return None
    return frame.select(*cols, pl.col("value").cast(pl.Float64).alias(alias))


def _invest_units(d, sol, var: str, edd: "pl.DataFrame | None", d_col: str) -> "pl.DataFrame | None":
    """``(p, d, value)``: Σ_{d′} ``var``[p,d′] over ``edd`` (p | e, d′ = ``d_col``, d) at the solution (the
    Σ_{d′} v_invest_p / v_divest_p of the row, :func:`_invest_change`); None without either."""
    if edd is None or edd.height == 0 or var not in getattr(sol, "_vars", {}):
        return None
    edd = _as_utf8(edd.rename({"e": "p"}) if "e" in edd.columns else edd)
    val = _as_utf8(sol.value(var)).rename({"d": d_col})
    out = (edd.join(val, on=["p", d_col], how="inner")
           .group_by("p", "d").agg(pl.col("value").cast(pl.Float64).sum()))
    return out if out.height > 0 else None


def _row_duals(sol, family: str, step: str) -> "pl.DataFrame | None":
    """``(g, d, <step>, dual)`` (Utf8 keys) of the rows ``family[g,d,step]``; None without the family.  The key
    is split from the right, so a group name may hold a comma; quotes (a timestamp step) are stripped."""
    try:
        raw = sol.constraint_dual(family)
    except KeyError:
        return None
    if raw is None or raw.height == 0 or "key" not in raw.columns:
        return None
    parts = raw.select(pl.col("key").cast(pl.Utf8).str.extract_groups(r"^(.*),([^,]*),([^,]*)$").alias("k"),
                       pl.col("dual").cast(pl.Float64)).unnest("k")
    names = parts.columns[:3]
    return (parts.drop_nulls(names)
            .select(*(pl.col(c).str.strip_chars("'").alias(a) for c, a in zip(names, ("g", "d", step))), "dual"))


def capacity_margin_report(d, sol, *, scale_the_objective: "float | None" = None,
                           duals_valid: bool = True) -> "tuple[pl.DataFrame, pl.DataFrame] | None":
    """The capacity margin of a solved solve, per group and investment period, at the row's binding step:
    ``(summary, entity)``; None when the solve has no capacity margin.

    One evaluation of the row's terms at the solution, from the class frames the row is built from
    (:func:`capacity_margin_unit_arcs`, :func:`_counted_terms`, ``_balance_terms.balance_flow_values``):

        firm, vre   = Σ_{arcs of F(g) / V(g)} (existing_count + Σ invest − Σ divest) · unitsize · κ   (per unit)
        storage_flow, unit_flow, connections_net, consumption
                    = the B¬FV terms of the storage-backed units' output, the other by-flow unit outputs
                      (converters inside g, conversion_flow_coeff 0), the connections (both ends, reverse
                      flows) and the unit inputs from N(g), each signed as in the node balance
        dumped      = − Σ_{n ∈ N(g)} vq_state_down
        counted     = the sum of these (the row's left-hand side without the slack)
        requirement = M + demand,  demand = Σ_{n ∈ N(g)} −inflow_real / Δt
        deficit[t]  = requirement − counted;  binding step = argmax_t deficit (the first of equal ones)
        shortfall   = vq_capacity_margin · group_capacity_for_scaling   (= max(0, max_t deficit) at the optimum)
        penalty_cost = shortfall · penalty_capacity_margin · 1000 · inflation_op / 1e6                 [M CUR]
        capacity_price = Σ_t dual(capacityMargin[g,d,t]) · icap / s / (1000 · inflation_op · bw)      [CUR/kW/a]

    A group whose row is one row per coarse step (the row rule) is evaluated at the Δt-weighted mean of each coarse
    step (weights ``w`` of ``coarse_row_partition``), as its LP row is; its binding step is the coarse step's first
    fine step.  ``penalty_cost`` follows the output cost tables (``calc_slacks``: the run's operations inflation, no
    branch weight).  ``capacity_price`` is the shadow price of the margin: it equals penalty_capacity_margin while
    there is a shortfall, and is null when the duals are not valid (``duals_valid`` False, a MIP).

    ``summary``: ``(group, period, binding_time, margin, peak_demand, demand, requirement, firm, vre, storage_flow,
    unit_flow, connections_net, consumption, dumped, counted, shortfall, penalty_cost, capacity_price)`` [MW
    unless named above], ``binding_time`` the binding step.
    ``entity``: ``(group, period, binding_time, class, entity, value)``: each entity's contribution at the binding step
    (``class`` of :data:`CAPACITY_MARGIN_CLASSES`; ``entity`` a unit, connection or, for ``dumped``, a node); its
    values sum to ``counted``.
    """
    if not has_capacity_margin(d) or "vq_capacity_margin" not in getattr(sol, "_vars", {}):
        return None
    resolve_capacity_margin(d)
    dt_inv = d.dt.join(_capacity_margin_periods(d), on="d", how="semi")
    steps = _as_utf8(dt_inv.select("d", "t")).with_row_index("__order")
    groups = pl.DataFrame({"g": _names(d.groupCapacityMargin, "g")}, schema={"g": pl.Utf8})
    gn = capacity_margin_nodes(d)
    gn_u = _as_utf8(gn) if gn is not None else pl.DataFrame(schema={"g": pl.Utf8, "n": pl.Utf8})
    contrib: list[pl.DataFrame] = []
    cols = ["g", "d", "t", "cls", "entity", "value"]

    # F and V: the counted arcs at capacity.
    arcs, kappa = _counted_terms(d, gn, dt_inv)
    if arcs is not None:
        units = _param_utf8(d.p_process_existing_count, ("p", "d"), "exist")
        units = units if units is not None else pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8, "exist": pl.Float64})
        inv = _invest_units(d, sol, "v_invest_p", getattr(d, "edd_invest_set", None), "d_invest")
        div = _invest_units(d, sol, "v_divest_p", getattr(d, "edd_divest_active", None), "d_divest")
        kap = _as_utf8(kappa.frame).rename({"value": "kappa"})
        cap = kap.join(units, on=["p", "d"], how="left")
        for frame, alias in ((inv, "inv"), (div, "div")):
            frame = frame if frame is not None else pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8,
                                                                       "value": pl.Float64})
            cap = cap.join(frame.rename({"value": alias}), on=["p", "d"], how="left")
        cls = (capacity_margin_unit_arcs(d).filter(pl.col("cls").is_in(["firm", "vre"]))
               .select("g", "p", "source", "sink", "cls").unique(["g", "p", "source", "sink"]))
        us = _param_utf8(d.p_unitsize, ("p",), "us")
        cap = (cap.join(us, on="p", how="inner").join(cls, on=["g", "p", "source", "sink"], how="inner")
               .with_columns(value=(pl.col("exist").fill_null(0.0) + pl.col("inv").fill_null(0.0)
                                    - pl.col("div").fill_null(0.0)) * pl.col("us") * pl.col("kappa")))
        contrib.append(cap.select("g", "d", "t", "cls", pl.col("p").alias("entity"), "value"))

    # B¬FV: the other flows at N(g), by their value.
    if gn is not None:
        flows = _balance_terms.balance_flow_values(d, sol, group_nodes=gn, exclude_arcs=_excluded_arcs(d, gn, arcs))
        flows = flows.join(steps.select("d", "t"), on=["d", "t"], how="semi")
        unit_names = _names(getattr(d, "process_unit", None), "p")
        by_flow = capacity_margin_unit_arcs(d)
        by_flow = (by_flow.filter(~pl.col("cls").is_in(["firm", "vre"]))
                   .select("g", "p", "source", "sink", pl.col("cls").alias("__ucls"))
                   if by_flow is not None else pl.DataFrame(schema={c: pl.Utf8 for c in
                                                                    ("g", "p", "source", "sink", "__ucls")}))
        flows = flows.join(by_flow.unique(["g", "p", "source", "sink"]), on=["g", "p", "source", "sink"],
                           how="left")
        is_unit = pl.col("p").is_in(unit_names)
        flows = flows.with_columns(cls=pl.when(~is_unit).then(pl.lit("connection"))
                                   .when(pl.col("term") != "sink_flow").then(pl.lit("consumption"))
                                   .when(pl.col("__ucls") == "storage").then(pl.lit("storage"))
                                   .otherwise(pl.lit("unit_by_flow")))
        contrib.append(flows.select("g", "d", "t", "cls", pl.col("p").alias("entity"), "value"))
        if "vq_state_down" in getattr(sol, "_vars", {}):
            dump = _as_utf8(sol.value("vq_state_down")).join(gn_u, on="n", how="inner")
            dump = dump.join(steps.select("d", "t"), on=["d", "t"], how="semi")
            contrib.append(dump.select("g", "d", "t", pl.lit("dumped").alias("cls"), pl.col("n").alias("entity"),
                                       (0.0 - pl.col("value").cast(pl.Float64)).alias("value")))
    # The slack energy of the storage nodes feeding g, taken back at every step (÷ Δt).
    pairs = capacity_margin_storage_slack_nodes(d)
    if pairs is not None and "v_capacity_margin_slack_energy" in getattr(sol, "_vars", {}):
        energy = _as_utf8(sol.value("v_capacity_margin_slack_energy")).select(
            "n", "d", pl.col("value").cast(pl.Float64).alias("energy"))
        sd = _param_utf8(d.p_step_duration, ("d", "t"), "dt")
        se = (pairs.join(energy, on="n", how="inner").join(steps.select("d", "t"), on="d", how="inner")
              .join(sd, on=["d", "t"], how="inner"))
        contrib.append(se.select("g", "d", "t", pl.lit("storage_slack").alias("cls"), pl.col("n").alias("entity"),
                                 (0.0 - pl.col("energy") / pl.col("dt")).alias("value")))
    contrib_df = (pl.concat([c.select(cols) for c in contrib]) if contrib
                  else pl.DataFrame(schema={**{c: pl.Utf8 for c in cols[:-1]}, "value": pl.Float64}))
    contrib_df = (contrib_df.join(groups, on="g", how="semi")
                  .group_by("g", "d", "t", "cls", "entity").agg(pl.col("value").sum()))

    # The requirement per step: M + the real demand of N(g).
    base = groups.join(steps, how="cross")
    margin = _param_utf8(d.pdGroup_capacity_margin, ("g", "d"), "margin")
    base = (base.join(margin, on=["g", "d"], how="left") if margin is not None
            else base.with_columns(margin=pl.lit(None, dtype=pl.Float64)))
    base = base.with_columns(pl.col("margin").fill_null(0.0))
    inflow = _param_utf8(getattr(d, "pdtNodeInflow_per_step", None), ("n", "d", "t"), "inflow")
    if inflow is not None and gn_u.height > 0:
        dem = (gn_u.join(inflow, on="n", how="inner").group_by("g", "d", "t")
               .agg((-pl.col("inflow")).sum().alias("demand")))
        base = base.join(dem, on=["g", "d", "t"], how="left")
    else:
        base = base.with_columns(demand=pl.lit(None, dtype=pl.Float64))
    base = base.with_columns(pl.col("demand").fill_null(0.0))

    # The row rule: a coarse group's rows are the Δt-weighted means of its coarse steps.
    part = None
    if _has_rows(getattr(d, "process_coarse_step", None)):
        from flextool.engine.preprocess.flex_data import coarse_row_partition

        part = coarse_row_partition(d, "capacity_margin")
        part = None if part is None or part.height == 0 else part.select(
            *(pl.col(c).cast(pl.Utf8) for c in ("g", "d", "b_first", "t")), pl.col("w").cast(pl.Float64))
    if part is not None:
        def coarse(frame: pl.DataFrame, keys: list[str], values: list[str]) -> pl.DataFrame:
            j = frame.join(part, on=["g", "d", "t"], how="left")
            fine = j.filter(pl.col("b_first").is_null()).select(frame.columns)
            agg = (j.filter(pl.col("b_first").is_not_null())
                   .group_by(*[k for k in keys if k != "t"], "b_first")
                   .agg(*((pl.col(v) * pl.col("w")).sum().alias(v) for v in values))
                   .rename({"b_first": "t"}))
            return pl.concat([fine, agg.select(frame.columns)])
        contrib_df = coarse(contrib_df, ["g", "d", "t", "cls", "entity"], ["value"])
        base = base.drop("__order")
        base = coarse(base, ["g", "d", "t"], ["margin", "demand"]).join(steps, on=["d", "t"], how="left")

    sums = (contrib_df.with_columns(col=pl.col("cls").replace_strict(CAPACITY_MARGIN_CLASSES, default=None))
            .group_by("g", "d", "t", "col").agg(pl.col("value").sum()))
    wide = base.with_columns(requirement=pl.col("margin") + pl.col("demand"))
    class_cols = list(dict.fromkeys(CAPACITY_MARGIN_CLASSES.values()))
    for col in class_cols:
        wide = wide.join(sums.filter(pl.col("col") == col).select("g", "d", "t", pl.col("value").alias(col)),
                         on=["g", "d", "t"], how="left").with_columns(pl.col(col).fill_null(0.0))
    wide = wide.with_columns(counted=pl.sum_horizontal(class_cols))
    wide = wide.with_columns(__deficit=pl.col("requirement") - pl.col("counted"))
    binding = (wide.sort("g", "d", "__deficit", "__order", descending=[False, False, True, False])
               .unique(["g", "d"], keep="first", maintain_order=True))

    # The slack, its penalty and the shadow price.
    vq = _as_utf8(sol.value("vq_capacity_margin")).select("g", "d", pl.col("value").cast(pl.Float64).alias("vq"))
    gcs = _param_utf8(d.p_group_capacity_for_scaling, ("g", "d"), "gcs")
    pen = _param_utf8(d.pdGroup_penalty_capacity_margin, ("g", "d"), "penalty")
    infl_run_param = getattr(d, "p_inflation_op_run", None)
    infl_run = _param_utf8(infl_run_param if infl_run_param is not None else d.p_inflation_op, ("d",), "infl_run")
    infl = _param_utf8(d.p_inflation_op, ("d",), "infl")
    peak = _param_utf8(getattr(d, "pdGroup_peak_demand", None), ("g", "d"), "peak_demand")
    out = binding.join(vq, on=["g", "d"], how="left").join(gcs, on=["g", "d"], how="left")
    out = out.with_columns(shortfall=pl.col("vq").fill_null(0.0) * pl.col("gcs").fill_null(1.0))
    for frame, on in ((pen, ["g", "d"]), (infl_run, ["d"]), (infl, ["d"]), (peak, ["g", "d"])):
        if frame is not None:
            out = out.join(frame, on=on, how="left")
    for c in ("penalty", "infl_run", "infl", "peak_demand"):
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    out = out.with_columns(penalty_cost=pl.col("shortfall") * pl.col("penalty").fill_null(0.0) * 1000.0
                           * pl.col("infl_run").fill_null(1.0) / 1e6)
    price = None
    if duals_valid:
        duals = [f for f in (_row_duals(sol, "capacityMargin", "t"), _row_duals(sol, "capacityMargin_coarse", "t"))
                 if f is not None]
        if duals:
            bw = _capacity_margin_branch_weight(d)
            bw_u = (_param_utf8(bw, ("d",), "bw") if bw is not None
                    else pl.DataFrame(schema={"d": pl.Utf8, "bw": pl.Float64}))
            icap = _param_utf8(d.p_inv_group_cap, ("g", "d"), "icap")
            s_obj = 1.0 if scale_the_objective is None or scale_the_objective <= 0 else float(scale_the_objective)
            price = (pl.concat(duals).group_by("g", "d").agg(pl.col("dual").sum())
                     .join(icap, on=["g", "d"], how="left").join(infl, on="d", how="left")
                     .join(bw_u, on="d", how="left")
                     .with_columns(capacity_price=pl.col("dual") * pl.col("icap") / s_obj
                                   / (1000.0 * pl.col("infl").fill_null(1.0) * pl.col("bw").fill_null(1.0)))
                     .select("g", "d", "capacity_price"))
    out = (out.join(price, on=["g", "d"], how="left") if price is not None
           else out.with_columns(capacity_price=pl.lit(None, dtype=pl.Float64)))
    out = out.sort("g", "__order")
    # + 0.0 turns a −0.0 (a negated zero term) into 0.0.
    measures = ["margin", "peak_demand", "demand", "requirement", *class_cols, "counted", "shortfall",
                "penalty_cost", "capacity_price"]
    summary = out.select(pl.col("g").alias("group"), pl.col("d").alias("period"), pl.col("t").alias("binding_time"),
                         *((pl.col(c) + 0.0).alias(c) for c in measures))
    # The nodes that dumped nothing and the storage nodes whose slack made nothing are left out of the entity table.
    quiet = pl.col("cls").is_in(["dumped", "storage_slack"]) & (pl.col("value") == 0.0)
    entity = (contrib_df.filter(~quiet).with_columns(pl.col("value") + 0.0)
              .join(out.select("g", "d", "t", "__order"), on=["g", "d", "t"], how="inner")
              .sort("g", "__order", "cls", "entity")
              .select(pl.col("g").alias("group"), pl.col("d").alias("period"), pl.col("t").alias("binding_time"),
                      pl.col("cls").alias("class"), "entity", "value"))
    return summary, entity


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
    #   or period + time value (a time map covers every time step of the solve: a gap is an input error,
    #   _direct.check_group_limit_time_coverage), 0 for a group without any value or an authored 0 (no requirement:
    #   the row reads Σ … ≥ 0 and its slack is in no row)
    #   arcs_side(g) = arcs whose side node is in g and that carry an inertia constant on that side
    #   x = v_online[p,d,t] for p ∈ process_online (linear or integer), else v_flow[p,s,k,d,t]
    # Synchronous machines connected to the group's nodes supply inertia (MWs)
    # in proportion to the units online, or to their flow when they have no
    # online variable; the shortfall is penalised.  The slack is a fraction of
    # the limit.
    # Row rule (coarse steps): a group whose inertia providers are all coarse processes gets one row per coarse
    # step, inertia_constraint_coarse = the row's mean over the step (_coarse_flows.add_cstr), on one slack per
    # step; the limit enters as its Δt-weighted step mean, and a limit that varies within a coarse step is named
    # in the coarse-flow behaviour notice.  Otherwise per solve step.
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
    #   ∀ g ∈ groupNonSync, (d,t) ∈ dt
    #   groupNonSync = groups with has_non_synchronous = yes
    #   N(g) = nodes of g;  icap = inv_group_cap;  ns_limit = non_synchronous_limit (constant, period, time or
    #   period + time); a group without any value, and an authored 0, read as 0 (the right-hand side then
    #   vanishes: no non-synchronous infeed without the penalised slack); a time map covers every time step of
    #   the solve (a gap is an input error)
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

    # Rows on every (d, t): a time map covers every time step of the solve (a gap is an input error,
    # _direct.check_group_limit_time_coverage).
    over = d.groupNonSync.join(d.dt, how="cross")
    _coarse_flows.add_row_rule_cstr(b, "non_sync_constraint", over=over, sense="<=", lhs_terms=lhs, rhs_terms=rhs,
                                    partition=_row_partition(b, "non_sync", over))


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
