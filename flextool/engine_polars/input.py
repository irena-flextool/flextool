"""Read flextool ``input/`` + ``solve_data/`` into a single
``FlexData`` bag.

All fields are optional — None / empty when the scenario doesn't
exercise that feature.  ``build_flextool(p, d)`` switches on field
presence to decide which constraints / variables / objective terms
to add.

Pipeline shape today:

    Spine DB → flextool preprocess → input/ + solve_data/ CSVs → load_flextool

Once flextool's preprocessing migration to Python is complete, this
module gains a parallel entry point that consumes the in-memory
preprocessing state directly, skipping the CSV roundtrip.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl
from polar_high_opt import Param

from . import _group_slack
from . import _reserve
from . import _cumulative_invest
from . import _delay
from . import _dc_power_flow
from . import _commodity_ladder


# ---------------------------------------------------------------------------
# CSV-shape helpers (same three shapes as before)

def _read_long(path: Path, *, drop=("solve",), rename=None) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.drop([c for c in drop if c in df.columns])
    if rename: df = df.rename(rename)
    return df


def _read_wide_per_entity(path: Path, value_col: str = "value",
                          rename=None) -> pl.DataFrame:
    """Reads a wide-per-entity CSV (header: solve, period, time, e1, e2,…)
    OR a long CSV (header: <entity_col>, period, time, value) — the new
    Python-preprocessed format.  In the long case the entity column is
    whatever flextool's preprocessor wrote (node / process / commodity
    /…); the caller's ``rename={'entity': X}`` is applied either way."""
    df = pl.read_csv(path)
    if "value" in df.columns and "solve" not in df.columns:
        # Long format from new Python preprocessing.  Entity column is
        # the first; rename to "entity" to keep the downstream contract.
        entity_col = df.columns[0]
        out = (df.rename({entity_col: "entity",
                           "period": "d", "time": "t"})
                 .with_columns(value=pl.col(value_col).cast(pl.Float64,
                                                            strict=False))
                 .select("entity", "d", "t", "value"))
        out = out.with_columns(value=pl.col("value").fill_null(0.0))
        if rename: out = out.rename(rename)
        return out
    # Legacy wide-per-entity format.
    df = df.drop("solve")
    id_cols = ["period", "time"]
    value_cols = [c for c in df.columns if c not in id_cols]
    out = (df.unpivot(on=value_cols, index=id_cols, variable_name="entity",
                      value_name=value_col)
             .rename({"period": "d", "time": "t"}))
    if rename: out = out.rename(rename)
    return out


def _read_unitsize(path: Path) -> pl.DataFrame:
    """Read ``p_entity_unitsize.csv``.  The canonical Python-preprocessing
    output is long-format ``(entity, value)`` in ``solve_data/``.  The
    ``.mod`` also printf's a wide-format twin to ``input/`` (one row,
    columns are entity names) — supported as a fallback for legacy
    fixtures."""
    df = pl.read_csv(path)
    if {"entity", "value"}.issubset(df.columns):
        return (df.rename({"entity": "e"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False))
                  .select("e", "value"))
    # legacy wide format: drop the first column (label "entity"/"value"),
    # then transpose so column names become rows.
    df = df.drop(df.columns[0])
    return (df.transpose(include_header=True, header_name="e",
                         column_names=["value"])
              .with_columns(value=pl.col("value").cast(pl.Float64)))


def _read_capacity(path: Path,
                    previously_invested_path: Path | None = None,
                    all_existing_path: Path | None = None) -> pl.DataFrame:
    # Prefer ``p_entity_all_existing.csv`` if available — it's the
    # cumulative existing capacity per period (reflecting lifetime,
    # carried over across periods within a solve), which is what the
    # .mod's ``p_entity_dispatch_capacity_max`` formula uses.
    if all_existing_path is not None and all_existing_path.exists():
        df = pl.read_csv(all_existing_path)
        if "solve" in df.columns: df = df.drop("solve")
        # Long-format variant: columns are (entity, period, value).
        if {"entity", "period", "value"}.issubset(df.columns):
            return (df.rename({"entity": "e", "period": "d"})
                      .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(0.0))
                      .select("e", "d", "value"))
        # Wide-format variant: columns are (period, entity1, entity2, …).
        val_cols = [c for c in df.columns if c != "period"]
        if df.height == 0 or not val_cols:
            return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
        out = (df.unpivot(on=val_cols, index=["period"], variable_name="e",
                          value_name="value")
                 .rename({"period": "d"})
                 .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
                 .select("e", "d", "value"))
        return out

    # Legacy fallback: derive from per-period existing/invested fields.
    # ``p_entity_period_existing_capacity`` (post-solve snapshot) =
    #   base + prior-solve invest + current-solve invest realized.
    # ``p_entity_period_invested_capacity`` = sum of all realized invest.
    # ``p_entity_previously_invested_capacity`` = prior-solve invest only.
    #
    #   effective_existing = existing − invested + previously_invested
    #                      = (base + prior + current) − (prior + current) + prior
    #                      = base + prior
    df = pl.read_csv(path).with_columns(
        pl.col("p_entity_period_existing_capacity").cast(pl.Float64, strict=False),
        pl.col("p_entity_period_invested_capacity").cast(pl.Float64, strict=False),
    )
    base_plus_prior = (
        pl.col("p_entity_period_existing_capacity").fill_null(0.0)
        - pl.col("p_entity_period_invested_capacity").fill_null(0.0)
    )
    df = df.with_columns(value=base_plus_prior)
    if previously_invested_path is not None and previously_invested_path.exists():
        prior_df = pl.read_csv(previously_invested_path)
        rename = ({"value": "prior"} if "value" in prior_df.columns
                  else {"p_entity_previously_invested_capacity": "prior"})
        prior = prior_df.rename(rename)
        df = (df.join(prior, on=["entity", "period"], how="left")
                .with_columns(prior=pl.col("prior").fill_null(0.0))
                .with_columns(value=pl.col("value") + pl.col("prior")))
    return (df.rename({"entity": "e", "period": "d"})
              .select("e", "d", "value"))


def _read_p_flow_max(path: Path) -> pl.DataFrame | None:
    """Read flextool's canonical ``solve_data/p_flow_max.csv`` long-format
    file ``[process, source, sink, period, time, value]`` (the same file
    flextool.mod consumes via ``table data IN``)."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.height == 0:
        return None
    return df.rename({"process": "p", "period": "d", "time": "t"}) \
             .select("p", "source", "sink", "d", "t", "value")


def _slice_param(path: Path, entity_col: str, param_value: str,
                 has_time: bool = True,
                 rename_entity_to: str | None = None) -> pl.DataFrame | None:
    """Slice a generic param-bearing canonical input
    (``pdtNode.csv``/``pdtCommodity.csv``/``pdtProcess.csv``/``pdProcess.csv``/``pdtGroup.csv``)
    by a literal ``param`` string — the same operation .mod does inline
    via e.g. ``pdtNode[n, 'penalty_up', d, t]``.

    Returns ``(entity, d, t, value)`` (or ``(entity, d, value)`` when
    ``has_time=False``) or ``None`` if the file is missing or the slice
    is empty.  ``rename_entity_to`` renames the entity column for
    downstream consumers (e.g. ``"node" -> "n"``)."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.height == 0:
        return None
    sliced = df.filter(pl.col("param") == param_value).drop("param")
    if sliced.height == 0:
        return None
    rename = {"period": "d"}
    if has_time:
        rename["time"] = "t"
    if rename_entity_to is not None:
        rename[entity_col] = rename_entity_to
    out = sliced.rename(rename)
    cols = [rename.get(entity_col, entity_col), "d"] + (["t"] if has_time else []) + ["value"]
    return out.select(cols)


def _read_step_previous(path: Path) -> pl.DataFrame | None:
    """Read flextool's canonical ``solve_data/step_previous.csv`` (the
    same file .mod reads as the ``dtttdt`` set, see flextool.mod:786).
    Renames columns to the names flexpy's downstream ``Lag`` call sites
    expect (``t_previous``, ``t_previous_within_timeset``, ``d_previous``,
    ``t_previous_within_solve``)."""
    if not path.exists():
        return None
    df = pl.read_csv(path)
    rename = {
        "period": "d", "time": "t",
        "previous": "t_previous",
        "previous_within_timeset": "t_previous_within_timeset",
        "previous_period": "d_previous",
        "previous_within_solve": "t_previous_within_solve",
    }
    out = df.rename({k: v for k, v in rename.items() if k in df.columns})
    keep = [v for v in rename.values() if v in out.columns]
    return out.select(keep)


# ---------------------------------------------------------------------------
# Single FlexData container

@dataclass
class FlexData:
    """Naming convention:

    * **Sets** (index frames, ``pl.DataFrame``) — no prefix.
      Examples: ``nodeBalance``, ``process_source_sink``, ``flow_to_n``,
      ``cdt_eq``, ``nodeState``.
    * **Parameters** (numeric ``Param``) — ``p_`` prefix.
      Examples: ``p_inflow``, ``p_unitsize``, ``p_commodity_price``.

    Variables created by ``build_flextool`` use ``v_`` for primal,
    ``vq_`` for slack — same convention as flextool.mod.
    """

    # ─── Time / weighting (always present) ────────────────────────────────
    dt: pl.DataFrame                         # set: (d, t)
    p_step_duration: Param                   # (d, t)
    p_rp_cost_weight: Param                  # (d, t)
    p_inflation_op: Param                    # (d,)
    p_period_share: Param                    # (d,)

    # ─── Nodes (always present in tested scenarios) ───────────────────────
    nodeBalance: pl.DataFrame                # set: (n,)
    nodeBalance_dt: pl.DataFrame             # set: nodeBalance × dt
    p_inflow: Param                          # (n, d, t)
    p_penalty_up: Param                      # (n, d, t)
    p_penalty_down: Param                    # (n, d, t)

    # ─── Process topology  ───────────────────────────────────────────────
    process_source_sink: pl.DataFrame | None = None
    process_source_sink_eff: pl.DataFrame | None = None
    process_source_sink_noEff: pl.DataFrame | None = None
    pss_dt: pl.DataFrame | None = None
    flow_to_n: pl.DataFrame | None = None
    flow_from_n: pl.DataFrame | None = None
    flow_from_commodity_eff: pl.DataFrame | None = None
    flow_from_commodity_noEff: pl.DataFrame | None = None
    flow_to_commodity: pl.DataFrame | None = None  # §2.4 sell into priced commodity node
    p_unitsize: Param | None = None              # (p,)
    p_flow_upper: Param | None = None            # (p, source, sink, d, t) — preprocessed structural max (existing + max_invest_cum)
    p_flow_upper_existing: Param | None = None   # (p, source, sink, d) — existing/unitsize only; used by maxToSink
    p_slope: Param | None = None                 # (p, d, t)
    p_commodity_price: Param | None = None       # (c, d, t)
    pd_neg_cap: pl.DataFrame | None = None       # set: (p, d) where existing<0 AND unitsize<0
                                                  # (anti-energy semantics: forces v_flow ≥ |existing|/|unitsize|)

    # ─── CO2 price ────────────────────────────────────────────────────────
    flow_from_co2_priced: pl.DataFrame | None = None
    flow_from_co2_priced_noEff: pl.DataFrame | None = None
    p_co2_content: Param | None = None           # (c,)
    p_co2_price: Param | None = None             # (g, d, t)

    # ─── CO2 cap (period) ─────────────────────────────────────────────────
    group_co2_max_period: pl.DataFrame | None = None
    flow_from_co2_capped: pl.DataFrame | None = None        # eff partition (slope)
    flow_from_co2_capped_noEff: pl.DataFrame | None = None  # noEff partition (no slope)
    p_co2_max_period: Param | None = None        # (g, d)
    group_d_co2_capped: pl.DataFrame | None = None

    # ─── Indirect-conversion (CHP) ────────────────────────────────────────
    process_indirect: pl.DataFrame | None = None
    process_input_flows: pl.DataFrame | None = None
    process_output_flows: pl.DataFrame | None = None
    process_indirect_dt: pl.DataFrame | None = None
    # Per-arc multipliers on the source / sink side of the
    # ``conversion_indirect`` equation (.mod:2557-2580).  Both default to
    # 1.0 (the absent-Param convention) and are only populated when at
    # least one row in the corresponding ``input/p_process_*_flow_coefficient.csv``
    # has a non-default, non-zero value.  When populated, the Param covers
    # *all* relevant (p, source) / (p, sink) rows of the indirect inputs /
    # outputs (filled to 1.0 where the CSV is silent), so multiplying
    # ``v_flow * unitsize * Param`` won't drop any flows.
    p_process_source_flow_coef: Param | None = None  # (p, source)
    p_process_sink_flow_coef: Param | None = None    # (p, sink)

    # ─── User-defined flow constraints ────────────────────────────────────
    flow_constraint_idx: pl.DataFrame | None = None
    p_flow_constraint_coef: Param | None = None  # (p, source, sink, c)
    p_constraint_constant: Param | None = None   # (c,)
    cdt_eq: pl.DataFrame | None = None
    cdt_le: pl.DataFrame | None = None
    cdt_ge: pl.DataFrame | None = None
    p_node_constraint_invested_capacity_coefficient: Param | None = None  # (n, c)
    p_process_constraint_invested_capacity_coefficient: Param | None = None  # (p, c)
    p_node_constraint_state_coefficient: Param | None = None  # (n, c) — user-cstr v_state coefficient
    p_node_constraint_prebuilt_capacity_coefficient: Param | None = None  # (n, c)
    p_process_constraint_prebuilt_capacity_coefficient: Param | None = None  # (p, c)

    # ─── Profiles ─────────────────────────────────────────────────────────
    process_profile_upper: pl.DataFrame | None = None    # (p, source, sink, f)
    process_profile_lower: pl.DataFrame | None = None
    process_profile_fixed: pl.DataFrame | None = None
    p_profile_value: Param | None = None         # (f, d, t)
    p_process_existing_count: Param | None = None  # (p, d) = cap / unitsize
    p_process_availability: Param | None = None  # (p, d, t)

    # ─── Invest / divest ──────────────────────────────────────────────────
    ed_invest_set: pl.DataFrame | None = None        # (e, d) — invest var index
    ed_divest_set: pl.DataFrame | None = None        # (e, d) — divest var index
    pd_invest_set: pl.DataFrame | None = None        # (p, d) — process-side
    pd_divest_set: pl.DataFrame | None = None        # (p, d)
    nd_invest_set: pl.DataFrame | None = None        # (n, d) — node-side
    nd_divest_set: pl.DataFrame | None = None        # (n, d)
    edd_invest_set: pl.DataFrame | None = None       # (e, d_invest, d)
    edd_invest_lookback_set: pl.DataFrame | None = None  # (e, d_invest, d) strict d_invest<d
    edd_divest_active: pl.DataFrame | None = None    # (p, d_divest, d) where d_divest ≤ d
    p_entity_max_units: Param | None = None          # (e, d)
    ed_lifetime_fixed_cost: Param | None = None      # (e, d)
    ed_lifetime_fixed_cost_divest: Param | None = None
    ed_entity_annual_discounted: Param | None = None
    ed_entity_annual_divest_discounted: Param | None = None
    e_invest_total: pl.DataFrame | None = None       # (e,)
    e_divest_total: pl.DataFrame | None = None
    e_invest_max_total: Param | None = None          # (e,)
    e_divest_max_total: Param | None = None
    ed_invest_period_set: pl.DataFrame | None = None  # (e, d) — entities with per-period invest cap
    ed_divest_period_set: pl.DataFrame | None = None  # (e, d)
    ed_invest_max_period: Param | None = None        # (e, d)
    ed_divest_max_period: Param | None = None        # (e, d)

    # Multi-solve handoff state — populated when running a sub-solve of
    # a chain, tells the LP what investment/divestment was realized in
    # prior sub-solves so that cumulative caps stay tight.  See
    # flextool.mod:3597-3623 (maxInvest_entity_total / maxDivest_entity_total
    # / minInvest_entity_total / minDivest_entity_total).
    p_entity_previously_invested_capacity: Param | None = None  # (e, d)
    p_entity_invested: Param | None = None      # (e,)  — cumulative prior-solve invest, used by min/max divest variants when not solveFirst
    p_entity_divested: Param | None = None      # (e,)  — cumulative prior-solve divest, used by max/min divest variants when not solveFirst

    # ─── Ramp limits ──────────────────────────────────────────────────────
    process_source_sink_ramp_limit_sink_up:   pl.DataFrame | None = None
    process_source_sink_ramp_limit_sink_down: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_up: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_down: pl.DataFrame | None = None
    p_ramp_speed_up_sink:   Param | None = None    # (p, sink)
    p_ramp_speed_down_sink: Param | None = None
    p_ramp_speed_up_source:   Param | None = None  # (p, source)
    p_ramp_speed_down_source: Param | None = None

    # ─── Online / min_load (unit commitment) ──────────────────────────────
    process_online: pl.DataFrame | None = None              # set: (p,)
    process_online_linear: pl.DataFrame | None = None        # set: (p,)
    process_online_integer: pl.DataFrame | None = None       # set: (p,)
    process_minload: pl.DataFrame | None = None              # set: (p,)
    process_min_load_eff: pl.DataFrame | None = None  # (p,) where ct_method=min_load_efficiency
    p_online_dt: pl.DataFrame | None = None                  # set: (p, d, t) — UC var domain
    pdt_online_linear: pl.DataFrame | None = None  # (p, d, t) — startup-cost obj index, linear
    pdt_online_integer: pl.DataFrame | None = None # (p, d, t) — startup-cost obj index, integer
    p_min_load: Param | None = None                          # (p,)
    p_startup_cost: Param | None = None                      # (p, d)
    p_section: Param | None = None                           # (p, d, t)
    pdt_uptime_set: pl.DataFrame | None = None               # (p, d, t) — minimum_uptime constraint domain
    pdt_downtime_set: pl.DataFrame | None = None             # (p, d, t) — minimum_downtime constraint domain
    uptime_lookback: pl.DataFrame | None = None              # (p, d, t, d_back, t_back) — startup lookback window
    downtime_lookback: pl.DataFrame | None = None            # (p, d, t, d_back, t_back) — shutdown lookback window

    # ─── Storage ─────────────────────────────────────────────────────────
    nodeState: pl.DataFrame | None = None
    nodeState_dt: pl.DataFrame | None = None
    nodeState_first_dt: pl.DataFrame | None = None
    storage_bind_within_timeset: pl.DataFrame | None = None
    storage_bind_forward_only: pl.DataFrame | None = None    # set: (n,)
    storage_bind_within_solve: pl.DataFrame | None = None    # set: (n,)
    storage_fix_start: pl.DataFrame | None = None
    dtttdt: pl.DataFrame | None = None           # (d, t, t_previous_*, ...)
    dtttdt_forward_only: pl.DataFrame | None = None  # dtttdt with first (d,t) per solve dropped
    # Rolling-horizon (nested-solve) framework — flextool.mod:2196 + 2760.
    # ``p_nested_solve_first``: tri-state.  None → no p_nested_model.csv,
    # treat as single-solve (== solveFirst).  True / False — read from
    # ``solve_data/p_nested_model.csv``'s ``solveFirst`` row.
    # When False, the nodeBalance ``fwd_fix_*`` block is *replaced* with a
    # ``roll_continue`` term that pins
    # ``v_state[n, d_first, t_first] * unitsize == p_roll_continue_state[n]``.
    p_nested_solve_first: bool | None = None
    p_roll_continue_state: Param | None = None        # (n,)
    n_fix_storage_quantity: pl.DataFrame | None = None  # (n,)
    ndt_fix_storage_quantity: pl.DataFrame | None = None  # (n, d_upper, t_upper)
    p_fix_storage_quantity: Param | None = None       # (n, d_upper, t_upper)
    dtt_timeline_matching: pl.DataFrame | None = None  # (d, t, t_upper) — lower→upper step map
    period_branch: pl.DataFrame | None = None         # (d_upper, d) — period→branch map
    period_last: pl.DataFrame | None = None           # (d,)
    nodeState_last_dt: pl.DataFrame | None = None     # (n, d, t) — block_period_time_last × node__block × nodeState
    # ─── Intraperiod-block storage (bind_intraperiod_blocks) ─────────────
    nodeStateBlock: pl.DataFrame | None = None             # set: (n,)
    period_block: pl.DataFrame | None = None               # set: (d, b_first)
    period_block_succ: pl.DataFrame | None = None          # set: (d, b_first, b_next)
    period_block_time: pl.DataFrame | None = None          # set: (d, b_first, t)
    dtttdt_block_interior: pl.DataFrame | None = None      # dtttdt rows where t_previous_within_timeset == t_previous (interior-of-block jump=1)
    # ─── Per-arc effective block step durations (M-matrix collapsed) ──
    # Indexed (p, source, sink, d, t) with value = block_step_duration of
    # the arc's relevant side block at fine step (d, t).  Drives the daily
    # flow-aggregation in nodeBalanceBlock_eq when coarse blocks are
    # active.  None for fixtures without process_side_block.csv.
    p_arc_step_duration_sink: Param | None = None
    p_arc_step_duration_source: Param | None = None
    # ─── Per-arc-side block aggregation index for nodeBalanceBlock_eq ──
    # (p, source, sink, d, b_first, t, weight): for each (n=sink, d, b_first)
    # in nodeStateBlock, the fine timesteps t (and weights) at which v_flow
    # contributes to the daily nodeBalance via the .mod's overlap × block_
    # step_duration aggregation.  weight = block_step_duration[b_f, d, t]
    # where b_f is the arc's sink-side block.  For coarse-side arcs only
    # the coarse step (t=b_first) appears, with weight=24 (or whatever sd).
    # For fine-side arcs (e.g., electrolyser source on hourly_group when
    # h2 is sink on daily_group), all 24 fine steps appear with weight=1.
    arc_sink_block_dt: pl.DataFrame | None = None    # (p, source, sink, d, b_first, t, weight)
    arc_source_block_dt: pl.DataFrame | None = None  # (p, source, sink, d, b_first, t, weight)
    p_arc_sink_weight: Param | None = None     # (p, source, sink, d, t) → weight
    p_arc_source_weight: Param | None = None   # (p, source, sink, d, t) → weight
    flow_from_nodeBalance_eff: pl.DataFrame | None = None
    flow_from_nodeBalance_noEff: pl.DataFrame | None = None
    p_state_upper: Param | None = None           # (n, d) — capacity / unitsize
    p_state_unitsize: Param | None = None        # (n,)
    p_state_self_discharge: Param | None = None  # (n,)
    p_state_start: Param | None = None           # (n,)
    p_state_existing_capacity: Param | None = None  # (n, d)
    # ─── Storage end-state binding (use_reference_value) ─────────────────
    # mod:2802-2822 — pins v_state at the last timestep of period_last to
    # ``reference_value × existing/unitsize`` for nodes with
    # ``storage_solve_horizon_method=use_reference_value`` and no
    # competing fix_end / fix_start_end / bind_within_solve method.
    storage_use_reference_value: pl.DataFrame | None = None  # (n,)
    p_storage_state_reference_value: Param | None = None     # (n, d, t)
    # ─── State-profile bounds (node__profile__profile_method) ────────────
    # (n, f) tuples for nodes with a profile-method state bound.  Mirrors
    # ``process_profile_*`` (process side) but for ``v_state``.
    node_profile_upper: pl.DataFrame | None = None  # (n, f)
    node_profile_lower: pl.DataFrame | None = None  # (n, f)
    node_profile_fixed: pl.DataFrame | None = None  # (n, f)
    p_node_availability: Param | None = None     # (n, d, t) — slice of pdtNode availability

    # ─── Process variable cost (other_operational_cost) ──────────────────
    pssdt_varCost_noEff: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_source: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_sink: pl.DataFrame | None = None
    pssdt_varCost_eff_connection: pl.DataFrame | None = None
    p_pssdt_varCost: Param | None = None     # (p, source, sink, d, t)
    p_pdt_varCost_source: Param | None = None  # (p, source, d, t) — eff source O&M
    p_pdt_varCost_sink: Param | None = None    # (p, sink, d, t) — eff sink O&M
    p_pdt_varCost_process: Param | None = None # (p, d, t) — connection O&M

    # ─── Existing-entity fixed cost (constant; reported in objective) ─────
    p_ed_fixed_cost: Param | None = None         # (e, d)
    p_entity_all_existing: Param | None = None   # (e, d)

    # ─── Slack penalty scaling ────────────────────────────────────────────
    p_node_capacity_for_scaling: Param | None = None  # (n, d)

    # ─── Group-level slack (capacity_margin / inertia / non_sync) ─────────
    groupCapacityMargin: pl.DataFrame | None = None      # (g,)
    groupInertia: pl.DataFrame | None = None             # (g,)
    groupNonSync: pl.DataFrame | None = None             # (g,)
    group_node: pl.DataFrame | None = None               # (g, n)
    process_unit: pl.DataFrame | None = None             # (p,)  set of unit-typed processes (mod's process_unit set)
    process_sink_inertia: pl.DataFrame | None = None     # (p, sink)
    process_source_inertia: pl.DataFrame | None = None   # (p, source)
    process_sink_nonSync: pl.DataFrame | None = None     # (p, sink)
    process_group_inside_nonSync: pl.DataFrame | None = None  # (p, g)
    p_inv_group_cap: Param | None = None                 # (g, d)
    p_group_capacity_for_scaling: Param | None = None    # (g, d)
    pdGroup_capacity_margin: Param | None = None         # (g, d)
    pdGroup_penalty_capacity_margin: Param | None = None # (g, d)
    pdGroup_inertia_limit: Param | None = None           # (g, d)
    pdGroup_penalty_inertia: Param | None = None         # (g, d)
    pdGroup_non_synchronous_limit: Param | None = None   # (g, d)
    pdGroup_penalty_non_synchronous: Param | None = None # (g, d)
    p_process_sink_inertia_constant: Param | None = None    # (p, sink)
    p_process_source_inertia_constant: Param | None = None  # (p, source)
    p_positive_inflow: Param | None = None               # (n, d, t)
    p_negative_inflow: Param | None = None               # (n, d, t)
    pdtNodeInflow_per_step: Param | None = None          # (n, d, t)

    # ─── Reserves (timeseries / dynamic / n-1, plus per-process upper) ────
    reserve_upDown_group: pl.DataFrame | None = None                  # (r, ud, g) — gate
    reserve_upDown_group_method_timeseries: pl.DataFrame | None = None  # (r, ud, g, method)
    reserve_upDown_group_method_dynamic: pl.DataFrame | None = None     # (r, ud, g, method)
    reserve_upDown_group_method_n_1: pl.DataFrame | None = None         # (r, ud, g, method)
    prundt: pl.DataFrame | None = None                                  # (p, r, ud, n, d, t) — v_reserve domain
    process_reserve_upDown_node_active: pl.DataFrame | None = None      # (p, r, ud, n)
    process_reserve_upDown_node_increase_reserve_ratio: pl.DataFrame | None = None  # (p, r, ud, n)
    process_reserve_upDown_node_large_failure_ratio: pl.DataFrame | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_reliability: Param | None = None      # (p, r, ud, n)
    pdtReserve_upDown_group_reservation: Param | None = None            # (r, ud, g, d, t)
    p_reserve_upDown_group_penalty_reserve: Param | None = None         # (r, ud, g)
    p_process_reserve_upDown_node_max_share: Param | None = None        # (p, r, ud, n)
    p_process_reserve_upDown_node_large_failure_ratio_value: Param | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_increase_reserve_ratio_value: Param | None = None  # (p, r, ud, n)

    # ─── Cumulative / group-invest / min-invest (read by _cumulative_invest) ─
    # Sets
    ed_invest_forbidden_no_investment: pl.DataFrame | None = None  # (e, d) — pin v_invest = 0
    ed_invest_cumulative: pl.DataFrame | None = None               # (e, d) — cumulative-cap rows
    group_entity: pl.DataFrame | None = None                       # (g, e)
    g_invest_total: pl.DataFrame | None = None                     # (g,)
    g_divest_total: pl.DataFrame | None = None                     # (g,)
    g_invest_cumulative: pl.DataFrame | None = None                # (g,)
    gd_invest_period: pl.DataFrame | None = None                   # (g, d)
    gd_divest_period: pl.DataFrame | None = None                   # (g, d)
    gdt_maxInstantFlow: pl.DataFrame | None = None                 # (g, d, t)
    gdt_minInstantFlow: pl.DataFrame | None = None                 # (g, d, t)
    group_process_node: pl.DataFrame | None = None                 # (g, p, n)
    # Parameters
    ed_invest_min_period: Param | None = None             # (e, d)
    ed_divest_min_period: Param | None = None             # (e, d)
    e_invest_min_total: Param | None = None               # (e,)
    e_divest_min_total: Param | None = None               # (e,)
    ed_cumulative_max_capacity: Param | None = None       # (e, d)
    ed_cumulative_min_capacity: Param | None = None       # (e, d)
    p_group_invest_max_period: Param | None = None        # (g, d)
    p_group_invest_min_period: Param | None = None        # (g, d)
    p_group_retire_max_period: Param | None = None        # (g, d)
    p_group_retire_min_period: Param | None = None        # (g, d)
    p_group_invest_max_total: Param | None = None         # (g,)
    p_group_invest_min_total: Param | None = None         # (g,)
    p_group_retire_max_total: Param | None = None         # (g,)
    p_group_retire_min_total: Param | None = None         # (g,)
    p_group_invest_max_cumulative: Param | None = None    # (g,)
    p_group_invest_min_cumulative: Param | None = None    # (g,)
    p_group_max_cumulative_flow: Param | None = None      # (g,)
    p_group_min_cumulative_flow: Param | None = None      # (g,)
    pd_max_cumulative_flow: Param | None = None           # (g, d)
    pd_min_cumulative_flow: Param | None = None           # (g, d)
    pdt_max_instant_flow: Param | None = None             # (g, d, t)
    pdt_min_instant_flow: Param | None = None             # (g, d, t)

    # ─── Delayed processes (read by _delay) ───────────────────────────
    process_delayed: pl.DataFrame | None = None                  # (p,)
    process_delayed__duration: pl.DataFrame | None = None        # (p, td)
    process_source_delayed: pl.DataFrame | None = None           # (p, source)
    process_source_undelayed: pl.DataFrame | None = None         # (p, source)
    process_source_sink_delayed: pl.DataFrame | None = None      # (p, source, sink)
    process_source_sink_undelayed: pl.DataFrame | None = None    # (p, source, sink)
    dtt__delay_duration: pl.DataFrame | None = None              # (d, t_source, t_sink, td)
    p_process_delay_weight: Param | None = None                  # (p, td)

    # ─── DC power flow (read by _dc_power_flow) ──────────────────────────
    # Populated only when ``input/node_dc_power_flow.csv`` and
    # ``connection_dc_power_flow.csv`` carry rows.  See
    # :mod:`flextool._dc_power_flow` for the constraint emission.
    node_dc_power_flow: pl.DataFrame | None = None               # (n,)
    connection_dc_power_flow: pl.DataFrame | None = None         # (p,)
    node_reference_angle: pl.DataFrame | None = None             # (n,)
    p_connection_susceptance: Param | None = None                # (p,)

    # ─── Commodity price ladder (read by _commodity_ladder) ─────────────
    # Populated only when at least one commodity has
    # ``price_method = price_ladder_*``.  See
    # :mod:`flextool._commodity_ladder` for the constraint emission.
    commodity_with_ladder: pl.DataFrame | None = None            # (c,)
    commodity_with_ladder_annual: pl.DataFrame | None = None     # (c,)
    commodity_with_ladder_cumulative: pl.DataFrame | None = None # (c,)
    cnd_ladder: pl.DataFrame | None = None                       # (c, n, d)
    cndi_ladder: pl.DataFrame | None = None                      # (c, n, d, i)
    cndi_ladder_ann: pl.DataFrame | None = None                  # (c, n, d, i)
    cndi_ladder_cum: pl.DataFrame | None = None                  # (c, n, d, i)
    ci_ladder_cumulative: pl.DataFrame | None = None             # (c, i)
    commodity__tier_ann: pl.DataFrame | None = None              # (c, i)
    commodity__tier_cum: pl.DataFrame | None = None              # (c, i)
    p_ladder_ann_price: Param | None = None                      # (c, i, d)
    p_ladder_ann_quantity: Param | None = None                   # (c, i, d)
    p_ladder_cum_price: Param | None = None                      # (c, i)
    p_ladder_cum_quantity: Param | None = None                   # (c, i)
    p_commodity_unitsize: Param | None = None                    # (c,)
    p_f_d_k: Param | None = None                                 # (d,)
    p_ladder_cum_realized_mwh: Param | None = None               # (c, i, d)

    # ─── Stochastic / multi-branch operational data (A6) ─────────────────
    # All fields populated only when the active solve actually runs a
    # multi-branch stochastic dispatch (signalled by ``solve_data/
    # pdt_branch_weight.csv`` containing rows where the cohort
    # (anchor period d) has multiple sibling periods b).  When stochastics
    # is inactive every (d, t) carries weight 1.0 and these fields stay
    # ``None`` (the model layer falls back to the deterministic path).
    #
    # ``period_branch_full`` is the unfiltered ``period__branch.csv``
    # (anchor d → sibling b).  Distinct from the existing
    # ``period_branch`` field which is the rolling-handoff helper
    # (renamed columns).  Both share the same source CSV; we keep them
    # separate to avoid disturbing the rolling-handoff consumer.
    pdt_branch_weight: Param | None = None        # (d, t) — operational weight (defaults 1.0)
    pd_branch_weight: Param | None = None         # (d,) — period-level weight (defaults 1.0)
    period_branch_full: pl.DataFrame | None = None  # (d, b) — full anchor→sibling map
    dt_non_anticipativity: pl.DataFrame | None = None  # (d, t) — realised dispatch + fix-storage timesteps
    groupStochastic: pl.DataFrame | None = None   # (g,) — groups enabling storage non-anticipativity
    period_in_use_set: pl.DataFrame | None = None  # (d,) — periods active this solve (filters branches)

    # ─── HiGHS solver options (read from input/solve_mode.csv) ───────────
    # Maps HiGHS option name → value (str / int / float / bool).  flextool
    # writes ``highs_method``, ``highs_parallel``, ``highs_presolve`` rows
    # keyed on ``solve``; load_flextool picks the row for the active solve
    # (solve_data/solve_current.csv) and renames keys to HiGHS canonical
    # option names (``solver``, ``parallel``, ``presolve``).  Applied in
    # ``Problem.solve()`` via ``Highs.setOptionValue``.  ``None`` means no
    # CSV / no rows for the active solve → HiGHS defaults.
    solver_options: dict | None = None

    def dump_csvs(self,
                   workdir: "Path | str",
                   *,
                   copy_meta_from: "Path | str | None" = None,
                   ) -> "Path":
        """Materialise this FlexData to flextool's CSV layout under ``workdir``.

        See :mod:`flextool._dump_csvs` for the full mapping.  Round-trip
        contract: ``load_flextool(dump_csvs(out))`` reproduces every
        populated FlexData field frame-for-frame (modulo row order).

        ``copy_meta_from`` is the original workdir whose per-solve
        metadata (``solve_current.csv``, timeline reference files,
        period-first markers, …) we copy through verbatim — these are
        runner state, not FlexData fields, but the CSV reader needs
        them.  When the round-trip caller has access to the original
        workdir, pass it here.
        """
        # Local import — avoids a circular import at module-load.
        from flextool.engine_polars._dump_csvs import dump_csvs as _impl
        return _impl(self, workdir, copy_meta_from=copy_meta_from)


# ---------------------------------------------------------------------------
# Time + node helpers (always loaded)

def _load_time(sd: Path):
    # ``steps_in_use.csv`` is the canonical source for both the dt set
    # and step_duration (.mod reads them together at flextool.mod:781).
    # ``dt.csv`` and ``p_step_duration.csv`` are .mod printf debug-exports
    # that only cover dispatch periods — using them silently drops the
    # invest-period (d, t) rows in multi-period scenarios.
    siu = pl.read_csv(sd / "steps_in_use.csv").rename(
        {"period": "d", "step": "t", "step_duration": "value"})
    dt = siu.select("d", "t")
    step_dur = Param(("d","t"), siu.select("d", "t", "value"))
    # rp_cost_weight: canonical ``rp_cost_weight.csv``
    # (.mod's ``p_rp_cost_weight.csv`` is a printf debug-export).
    # Defaults to 1.0 per (d, t) when the canonical file is empty
    # (matches .mod's ``param p_rp_cost_weight ... default 1`` clause).
    rp_default = dt.with_columns(value=pl.lit(1.0))
    rp_cw_path = sd / "rp_cost_weight.csv"
    if rp_cw_path.exists():
        rp_df = pl.read_csv(rp_cw_path)
        if rp_df.height > 0:
            # canonical column is named ``weight`` per .mod's ``table data IN``.
            value_col = "weight" if "weight" in rp_df.columns else "value"
            rp_df = (rp_df.rename({"period": "d", "time": "t",
                                    value_col: "value"})
                          .with_columns(value=pl.col("value")
                                                 .cast(pl.Float64, strict=False))
                          .select("d", "t", "value"))
            # Left-join the default with explicit overrides.
            rp_default = (rp_default.join(rp_df, on=["d","t"], how="left",
                                            suffix="__r")
                                     .with_columns(value=pl.coalesce(
                                          pl.col("value__r"), pl.col("value")))
                                     .select("d","t","value"))
    rp_cw = Param(("d","t"), rp_default)
    infl = Param(("d",),
        _read_long(sd / "p_inflation_factor_operations_yearly.csv", rename={"period": "d"}))
    # complete_period_share_of_year: canonical
    # ``complete_period_share_of_year_calc.csv``.
    psh = Param(("d",),
        _read_long(sd / "complete_period_share_of_year_calc.csv",
                    rename={"period": "d"}))
    return dt, step_dur, rp_cw, infl, psh


def _load_node(sd: Path, dt: pl.DataFrame):
    nb = pl.read_csv(sd / "nodeBalance.csv").rename({"node": "n"})
    # pdtNodeInflow.csv is canonical (.mod reads it via `table data IN`).
    # penalty_up/penalty_down are sliced from the canonical pdtNode.csv —
    # same operation .mod does inline via `pdtNode[n, 'penalty_up', d, t]`.
    inflow_long = _read_wide_per_entity(sd / "pdtNodeInflow.csv", rename={"entity":"n"})
    pen_up_long = _slice_param(sd / "pdtNode.csv", "node", "penalty_up",   rename_entity_to="n")
    pen_dn_long = _slice_param(sd / "pdtNode.csv", "node", "penalty_down", rename_entity_to="n")
    return (nb, nb.join(dt, how="cross"),
            Param(("n","d","t"), inflow_long.select("n","d","t","value")),
            Param(("n","d","t"), pen_up_long),
            Param(("n","d","t"), pen_dn_long))


# ---------------------------------------------------------------------------
# Process-topology helpers (skipped if no processes)

def _load_process_topology(inp: Path, sd: Path, dt: pl.DataFrame):
    pss_path = sd / "process_source_sink.csv"
    if not pss_path.exists():
        return {k: None for k in ("pss","pss_eff","pss_noEff","pss_dt",
                                   "flow_to_n","flow_from_n",
                                   "flow_from_commodity_eff",
                                   "flow_from_commodity_noEff",
                                   "unitsize","flow_upper","slope","commodity_price")}
    pss     = pl.read_csv(pss_path).rename({"process": "p"})
    if pss.height == 0:
        return {k: None for k in ("pss","pss_eff","pss_noEff","pss_dt",
                                   "flow_to_n","flow_from_n",
                                   "flow_from_commodity_eff",
                                   "flow_from_commodity_noEff",
                                   "unitsize","flow_upper","slope","commodity_price")}
    pss_eff = pl.read_csv(sd / "process_source_sink_eff.csv").rename({"process": "p"})
    pss_noEff = pl.read_csv(sd / "process_source_sink_noEff.csv").rename({"process": "p"})

    flow_to_n   = pss.with_columns(n=pl.col("sink"))
    flow_from_n = pss.with_columns(n=pl.col("source"))

    # ─── Filter arcs by block compatibility (mod's process_side_block) ──
    # In the .mod, an arc contributes to a node's nodeBalance_eq iff the
    # overlap set has a row connecting (b_n, t) ↔ (b_f, t_f) where
    # (p, side, b_f) ∈ process_side_block.  In particular, a daily-side
    # arc (e.g. electrolyser_A's sink on daily_group) does NOT contribute
    # to a fine-grid (hourly/default) node's hourly nodeBalance because
    # the overlap (hourly_group, t, daily_group, t_f) doesn't exist.
    # We replicate this restriction by filtering ``flow_to_n``/
    # ``flow_from_n`` to drop (p, source, sink) rows whose relevant
    # side-block doesn't connect via overlap to the node's own block.
    psb_path_arc = sd / "process_side_block.csv"
    eb_path_arc = sd / "entity_block.csv"
    ov_path_arc = sd / "overlap_set.csv"
    if (psb_path_arc.exists() and eb_path_arc.exists()
            and ov_path_arc.exists()):
        psb_local = pl.read_csv(psb_path_arc).rename(
            {"process": "p", "block": "b_f"})
        eb_local = pl.read_csv(eb_path_arc).rename(
            {"entity": "n", "block": "b"})
        ov_local = pl.read_csv(ov_path_arc)
        if (psb_local.height > 0 and eb_local.height > 0
                and ov_local.height > 0):
            # Build the set of (b_n, b_f) pairs that have at least one
            # overlap row.
            block_compat = (ov_local
                .rename({"block_coarse": "b", "block_fine": "b_f"})
                .select("b", "b_f").unique())
            # For each arc-side, look up b_f.  Then for each node (n, b),
            # an arc contributes iff (b, b_f) ∈ block_compat.
            psb_sink = psb_local.filter(pl.col("side") == "sink").select("p", "b_f")
            psb_source = psb_local.filter(pl.col("side") == "source").select("p", "b_f")
            # flow_to_n is keyed by sink-as-n; the relevant side is 'sink'.
            ftn_with_blocks = (flow_to_n
                .join(psb_sink, on="p", how="left")
                .join(eb_local, on="n", how="left"))
            # If b_f or b is null, treat as 'default' (compatibility default).
            ftn_with_blocks = ftn_with_blocks.with_columns(
                b_f=pl.col("b_f").fill_null("default"),
                b=pl.col("b").fill_null("default"),
            )
            # Inner-join with block_compat to keep compatible rows.
            ftn_filtered = (ftn_with_blocks
                .join(block_compat, on=["b", "b_f"], how="inner")
                .select("p", "source", "sink", "n").unique())
            # Replace flow_to_n if filter actually drops rows.
            if ftn_filtered.height > 0 and ftn_filtered.height < flow_to_n.height:
                flow_to_n = ftn_filtered
            # flow_from_n: source-as-n, side='source'.
            ffn_with_blocks = (flow_from_n
                .join(psb_source, on="p", how="left")
                .join(eb_local, on="n", how="left"))
            ffn_with_blocks = ffn_with_blocks.with_columns(
                b_f=pl.col("b_f").fill_null("default"),
                b=pl.col("b").fill_null("default"),
            )
            ffn_filtered = (ffn_with_blocks
                .join(block_compat, on=["b", "b_f"], how="inner")
                .select("p", "source", "sink", "n").unique())
            if ffn_filtered.height > 0 and ffn_filtered.height < flow_from_n.height:
                flow_from_n = ffn_filtered

    cn = pl.read_csv(inp / "commodity__node.csv")
    flow_from_commodity_eff = (pss_eff
        .join(cn, left_on="source", right_on="node", how="inner")
        .rename({"commodity": "c"})
        .select("p","source","sink","c"))
    flow_from_commodity_noEff = (pss_noEff
        .join(cn, left_on="source", right_on="node", how="inner")
        .rename({"commodity": "c"})
        .select("p","source","sink","c"))
    # §2.4 commodity sell: sink-side flow into a commodity-priced node.
    # No slope correction — straight v_flow * unitsize * commodity_price.
    flow_to_commodity = (pss
        .join(cn, left_on="sink", right_on="node", how="inner")
        .rename({"commodity": "c"})
        .select("p","source","sink","c"))

    unitsize_long = _read_unitsize((sd / "p_entity_unitsize.csv") if (sd / "p_entity_unitsize.csv").exists() else (inp / "p_entity_unitsize.csv"))
    unitsize_p = (unitsize_long.rename({"e": "p"})
                       .filter(pl.col("p").is_in(pss["p"].unique())))

    slope_long = _read_wide_per_entity(sd / "pdtProcess_slope.csv", rename={"entity":"p"})
    # commodity price sliced from canonical pdtCommodity.csv —
    # `pdtCommodity[c, 'price', d, t]` in .mod.
    cp_long = _slice_param(sd / "pdtCommodity.csv", "commodity", "price",
                            rename_entity_to="c")

    # flow_upper is the canonical ``p_flow_max.csv`` long-format file
    # the .mod reads via ``table data IN`` (`[process, source, sink,
    # period, time], p_flow_max~value`).
    flow_upper_psskdt = _read_p_flow_max(sd / "p_flow_max.csv")

    return dict(
        pss = pss,
        pss_eff = pss_eff,
        pss_noEff = pss_noEff,
        pss_dt = pss.join(dt, how="cross"),
        flow_to_n = flow_to_n,
        flow_from_n = flow_from_n,
        flow_from_commodity_eff = flow_from_commodity_eff,
        flow_from_commodity_noEff = flow_from_commodity_noEff,
        flow_to_commodity = flow_to_commodity,
        unitsize = Param(("p",), unitsize_p.select("p","value")),
        flow_upper = Param(("p","source","sink","d","t"), flow_upper_psskdt),
        slope = Param(("p","d","t"), slope_long.select("p","d","t","value")),
        commodity_price = Param(("c","d","t"), cp_long),
    )


# ---------------------------------------------------------------------------
# Optional features (CO2 price, CO2 cap, indirect, user-defined, profiles)

def _load_co2_price(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                     pss_noEff: pl.DataFrame | None = None):
    if pss_eff is None: return (None, None, None, None)
    files = ["group_co2_price.csv", "commodity_node_co2.csv", "pdtGroup.csv"]
    if not all((sd / f).exists() for f in files): return (None, None, None, None)
    g_price = pl.read_csv(sd / "group_co2_price.csv").rename({"group": "g"})
    if g_price.height == 0: return (None, None, None, None)
    cn_co2 = pl.read_csv(sd / "commodity_node_co2.csv").rename({"commodity":"c","node":"n"})
    g_node = pl.read_csv(inp / "group__node.csv").rename({"group":"g","node":"n"})
    gcn = (g_price.join(g_node, on="g", how="inner")
                  .join(cn_co2, on="n", how="inner")
                  .select("g","c","n"))
    flow_from_co2_priced = (pss_eff
        .join(gcn, left_on="source", right_on="n", how="inner")
        .select("p","source","sink","c","g"))
    # noEff variant: source flow into a CO2-priced commodity node where the
    # process is on the noEff side.  Rare but used for "cheap simplified"
    # gas/coal models that don't model efficiency curves.
    flow_from_co2_priced_noEff = None
    if pss_noEff is not None:
        flow_from_co2_priced_noEff = (pss_noEff
            .join(gcn, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if flow_from_co2_priced_noEff.height == 0:
            flow_from_co2_priced_noEff = None
    if flow_from_co2_priced.height == 0 and flow_from_co2_priced_noEff is None:
        return (None, None, None, None)
    p_comm = pl.read_csv(inp / "p_commodity.csv")
    co2_content = Param(("c",),
        p_comm.filter(pl.col("commodityParam")=="co2_content")
              .rename({"commodity":"c","p_commodity":"value"})
              .select("c","value"))
    # group co2_price sliced from canonical pdtGroup.csv —
    # `pdtGroup[g, 'co2_price', d, t]` in .mod.
    cp = _slice_param(sd / "pdtGroup.csv", "group", "co2_price",
                       rename_entity_to="g")
    co2_price = Param(("g","d","t"), cp) if cp is not None else None
    return (flow_from_co2_priced, flow_from_co2_priced_noEff,
            co2_content, co2_price)


def _load_co2_cap(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                   dt: pl.DataFrame,
                   pss_noEff: pl.DataFrame | None = None):
    if pss_eff is None and pss_noEff is None:
        return (None, None, None, None, None)
    p = sd / "group_co2_max_period.csv"
    if not p.exists(): return (None, None, None, None, None)
    g_max = pl.read_csv(p).rename({"group":"g"})
    if g_max.height == 0: return (None, None, None, None, None)
    cn_co2 = pl.read_csv(sd / "commodity_node_co2.csv").rename({"commodity":"c","node":"n"})
    g_node = pl.read_csv(inp / "group__node.csv").rename({"group":"g","node":"n"})
    gcn = (g_max.join(g_node, on="g", how="inner")
                .join(cn_co2, on="n", how="inner")
                .select("g","c","n"))
    if gcn.height == 0: return (None, None, None, None, None)
    # The .mod's co2_max_period sums emissions over (p, source, sink)
    # for processes whose source is a CO2-priced node — but with
    # different formulae for eff vs noEff.  eff is multiplied by
    # ``pdtProcess_slope[p, d, t]`` (the conversion-efficiency factor);
    # noEff is just ``v_flow * unitsize`` with no slope.  flexpy must
    # therefore split the set into two and handle each leg separately
    # — using a single combined set with the eff-style slope multiplier
    # would over-count noEff processes' emissions (e.g. coal_chp's
    # slope=1.111 inflates its CO2 by ~11%, breaking co2_max_period
    # parity on multi-period fixtures with non-trivial CHP shares).
    flow_from_co2_capped_eff = None
    flow_from_co2_capped_noEff = None
    if pss_eff is not None and pss_eff.height > 0:
        eff = (pss_eff.select("p","source","sink")
            .join(gcn, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if eff.height > 0:
            flow_from_co2_capped_eff = eff
    if pss_noEff is not None and pss_noEff.height > 0:
        noeff = (pss_noEff.select("p","source","sink")
            .join(gcn, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if noeff.height > 0:
            flow_from_co2_capped_noEff = noeff
    if flow_from_co2_capped_eff is None and flow_from_co2_capped_noEff is None:
        return (None, None, None, None, None)
    pd_group = pl.read_csv(inp / "pd_group.csv")
    cap_long = (pd_group.filter(pl.col("groupParam")=="co2_max_period")
                        .rename({"group":"g","period":"d","pd_group":"value"})
                        .select("g","d","value"))
    co2_max_period = Param(("g","d"), cap_long)
    period = dt.select("d").unique()
    return (g_max, flow_from_co2_capped_eff, flow_from_co2_capped_noEff,
            co2_max_period, g_max.join(period, how="cross"))


def _load_indirect(sd: Path, pss: pl.DataFrame | None, dt: pl.DataFrame,
                    inp: Path | None = None):
    if pss is None: return (None, None, None, None, None, None)
    p = sd / "process__method_indirect.csv"
    if not p.exists(): return (None, None, None, None, None, None)
    raw = pl.read_csv(p).rename({"process":"p"})
    if raw.height == 0: return (None, None, None, None, None, None)
    indirect = raw.select("p").unique()
    inputs  = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("sink")==pl.col("p")))
    outputs = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("source")==pl.col("p")))

    # The .mod's conversion_indirect LHS multiplies each source-side
    # v_flow by ``p_process_source_flow_coefficient[p, source]`` and the
    # RHS sum by ``p_process_sink_flow_coefficient[p, sink]`` (.mod:2557-2580).
    # Most scenarios have all coefs = 1 (the default); a zero coefficient
    # effectively drops that flow from the conversion equation; the
    # ``coal_chp_extraction`` scenario uses non-default sink coefficients
    # ({heat: 0.2, west: 2.0}) to encode the iso-fuel relationship via the
    # source-side capacity bound.  Build optional Params restricted to the
    # indirect inputs / outputs sets — only when non-default coefficients
    # are present — and let model.py multiply them into the conversion
    # equation.  Zero-coefficient rows are still anti-joined out (so they
    # don't survive into ``inputs`` / ``outputs``).
    p_source_flow_coef = None
    p_sink_flow_coef = None
    if inp is not None:
        src_path = inp / "p_process_source_flow_coefficient.csv"
        if src_path.exists():
            srcdf = pl.read_csv(src_path)
            if srcdf.height > 0 and "p_process_source_flow_coefficient" in srcdf.columns:
                src_long = (srcdf
                    .rename({"process": "p",
                             "p_process_source_flow_coefficient": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "source", "coef"))
                zero_src = src_long.filter(pl.col("coef") == 0.0).select("p", "source")
                if zero_src.height > 0:
                    inputs = inputs.join(zero_src, on=["p", "source"], how="anti")
                # If any non-default, non-zero coefficient applies to a
                # surviving (p, source) row, build a Param covering ALL
                # surviving (p, source) pairs (defaulted to 1.0 where not
                # listed) so the inner-join in v_flow * Param doesn't drop
                # rows.  Zero-coef rows have already been removed.
                nonzero_nondefault = src_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    in_pair = inputs.select("p", "source").unique()
                    if in_pair.height > 0:
                        merged = (in_pair.join(src_long, on=["p", "source"], how="left")
                                          .with_columns(pl.col("coef").fill_null(1.0))
                                          .rename({"coef": "value"}))
                        p_source_flow_coef = Param(("p", "source"), merged)
        sink_path = inp / "p_process_sink_flow_coefficient.csv"
        if sink_path.exists():
            sinkdf = pl.read_csv(sink_path)
            if sinkdf.height > 0 and "p_process_sink_flow_coefficient" in sinkdf.columns:
                sink_long = (sinkdf
                    .rename({"process": "p",
                             "p_process_sink_flow_coefficient": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "sink", "coef"))
                zero_sink = sink_long.filter(pl.col("coef") == 0.0).select("p", "sink")
                if zero_sink.height > 0:
                    outputs = outputs.join(zero_sink, on=["p", "sink"], how="anti")
                nonzero_nondefault = sink_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    out_pair = outputs.select("p", "sink").unique()
                    if out_pair.height > 0:
                        merged = (out_pair.join(sink_long, on=["p", "sink"], how="left")
                                           .with_columns(pl.col("coef").fill_null(1.0))
                                           .rename({"coef": "value"}))
                        p_sink_flow_coef = Param(("p", "sink"), merged)

    return (indirect, inputs, outputs, indirect.join(dt, how="cross"),
            p_source_flow_coef, p_sink_flow_coef)


def _load_user_constraints(inp: Path, pss: pl.DataFrame | None, dt: pl.DataFrame):
    """Returns 12 items:
    flow_cstr_idx, flow_cstr_coef, constraint_constant, cdt_eq, cdt_le, cdt_ge,
    n_inv_cstr_coef, p_inv_cstr_coef, n_state_cstr_coef,
    n_prebuilt_cstr_coef, p_prebuilt_cstr_coef, has_user_cstr.

    The ``*_inv_cstr_coef`` Params carry
    ``p_<entity>_constraint_invested_capacity_coefficient`` data;
    ``n_state_cstr_coef`` carries ``p_node_constraint_state_coefficient``
    (user-cstr v_state contribution); the ``*_prebuilt_cstr_coef`` Params
    carry ``p_<entity>_constraint_prebuilt_capacity_coefficient``
    (existing + prior-period invest)."""
    if pss is None: return [None]*12
    cs_path = inp / "constraint__sense.csv"
    if not cs_path.exists(): return [None]*12
    cs = pl.read_csv(cs_path).rename({"constraint":"c"})
    if cs.height == 0: return [None]*12
    coef_path = inp / "p_process_node_constraint_flow_coefficient.csv"
    flow_cstr_idx = flow_cstr_coef = None
    if coef_path.exists():
        coef_long = (pl.read_csv(coef_path)
            .rename({"process":"p","node":"n","constraint":"c",
                     "p_process_node_constraint_flow_coefficient":"coef"})
            .select("p","n","c","coef"))
        src_match = (pss.join(coef_long, left_on=["p","source"], right_on=["p","n"],
                              how="inner").select("p","source","sink","c","coef"))
        sink_match = (pss.join(coef_long, left_on=["p","sink"], right_on=["p","n"],
                               how="inner").select("p","source","sink","c","coef"))
        if src_match.height + sink_match.height > 0:
            joined = (pl.concat([src_match, sink_match], how="vertical")
                        .group_by(["p","source","sink","c"])
                        .agg(pl.col("coef").sum()))
            flow_cstr_idx  = joined.select("p","source","sink","c")
            flow_cstr_coef = Param(("p","source","sink","c"),
                joined.select("p","source","sink","c","coef").rename({"coef":"value"}))
    # Invest-capacity coefficient files — used to add v_invest_n and
    # v_invest_p terms to user constraints.  Both files are optional;
    # node-side and process-side are read independently.
    n_inv_path = inp / "p_node_constraint_invested_capacity_coefficient.csv"
    n_inv_cstr_coef = None
    if n_inv_path.exists():
        ndf = pl.read_csv(n_inv_path)
        if ndf.height > 0:
            ndf = (ndf.rename({"node":"n", "constraint":"c",
                                "p_node_constraint_invested_capacity_coefficient":"value"})
                      .select("n", "c", "value"))
            n_inv_cstr_coef = Param(("n", "c"), ndf)
    p_inv_path = inp / "p_process_constraint_invested_capacity_coefficient.csv"
    p_inv_cstr_coef = None
    if p_inv_path.exists():
        pdf = pl.read_csv(p_inv_path)
        if pdf.height > 0:
            pdf = (pdf.rename({"process":"p", "constraint":"c",
                                "p_process_constraint_invested_capacity_coefficient":"value"})
                      .select("p", "c", "value"))
            p_inv_cstr_coef = Param(("p", "c"), pdf)
    # ─── State coefficient (user-cstr × v_state) ──────────────────────────
    n_state_path = inp / "p_node_constraint_state_coefficient.csv"
    n_state_cstr_coef = None
    if n_state_path.exists():
        sdf = pl.read_csv(n_state_path)
        if sdf.height > 0:
            sdf = (sdf.rename({"node":"n", "constraint":"c",
                                "p_node_constraint_state_coefficient":"value"})
                      .select("n", "c", "value"))
            n_state_cstr_coef = Param(("n", "c"), sdf)
    # ─── Prebuilt-capacity coefficient (user-cstr × existing+prior_invest) ─
    n_pre_path = inp / "p_node_constraint_cumulative_pre_built_capacity_coefficient.csv"
    n_prebuilt_cstr_coef = None
    if n_pre_path.exists():
        ndf = pl.read_csv(n_pre_path)
        if ndf.height > 0:
            ndf = (ndf.rename({"node":"n", "constraint":"c",
                                "p_node_constraint_prebuilt_capacity_coefficient":"value"})
                      .select("n", "c", "value"))
            n_prebuilt_cstr_coef = Param(("n", "c"), ndf)
    p_pre_path = inp / "p_process_constraint_cumulative_pre_built_capacity_coefficient.csv"
    p_prebuilt_cstr_coef = None
    if p_pre_path.exists():
        pdf = pl.read_csv(p_pre_path)
        if pdf.height > 0:
            pdf = (pdf.rename({"process":"p", "constraint":"c",
                                "p_process_constraint_prebuilt_capacity_coefficient":"value"})
                      .select("p", "c", "value"))
            p_prebuilt_cstr_coef = Param(("p", "c"), pdf)
    const_path = inp / "p_constraint_constant.csv"
    constraint_constant = Param(("c",),
        (pl.read_csv(const_path).rename({"constraint":"c","p_constraint_constant":"value"})
         if const_path.exists()
         else cs.select("c").with_columns(value=pl.lit(0.0))))
    cdt_eq = cdt_le = cdt_ge = None
    for s, slot in [("equal","eq"), ("less_than","le"), ("greater_than","ge")]:
        cs_s = cs.filter(pl.col("sense")==s).select("c")
        if cs_s.height > 0:
            axes = cs_s.join(dt, how="cross")
            if   slot=="eq": cdt_eq = axes
            elif slot=="le": cdt_le = axes
            else:            cdt_ge = axes
    return (flow_cstr_idx, flow_cstr_coef, constraint_constant,
            cdt_eq, cdt_le, cdt_ge, n_inv_cstr_coef, p_inv_cstr_coef,
            n_state_cstr_coef, n_prebuilt_cstr_coef, p_prebuilt_cstr_coef,
            True)


def _read_wide_e_d(path: Path) -> pl.DataFrame:
    """Read a CSV in either long-format (``entity, period, value``) or
    wide-format (``solve, period, e1, e2, …``) and return long form
    ``(e, d, value)``."""
    if not path.exists():
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    df = pl.read_csv(path)
    if "solve" in df.columns:
        df = df.drop("solve")
    if df.height == 0 or "period" not in df.columns:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    # Long-format detection: explicit (entity, period, value) columns.
    if {"entity", "period", "value"}.issubset(df.columns):
        return (df.rename({"entity": "e", "period": "d"})
                  .with_columns(value=pl.col("value")
                                        .cast(pl.Float64, strict=False)
                                        .fill_null(0.0))
                  .select("e", "d", "value"))
    val_cols = [c for c in df.columns if c != "period"]
    return (df.unpivot(on=val_cols, index=["period"], variable_name="e",
                       value_name="value")
              .rename({"period": "d"})
              .with_columns(value=pl.col("value")
                                  .cast(pl.Float64, strict=False)
                                  .fill_null(0.0)))


def _load_invest(sd: Path, dt: pl.DataFrame, inp: Path,
                  pss: pl.DataFrame | None) -> dict:
    """Load invest/divest sets and per-(e, d) cost params.  Empty when
    neither ed_invest nor ed_divest has any row."""
    blank = dict(
        ed_invest_set=None, ed_divest_set=None,
        pd_invest_set=None, pd_divest_set=None,
        nd_invest_set=None, nd_divest_set=None,
        edd_invest_set=None, edd_invest_lookback_set=None,
        edd_divest_active=None,
        p_entity_max_units=None,
        ed_lifetime_fixed_cost=None,
        ed_lifetime_fixed_cost_divest=None,
        ed_entity_annual_discounted=None,
        ed_entity_annual_divest_discounted=None,
        e_invest_total=None, e_divest_total=None,
        e_invest_max_total=None, e_divest_max_total=None,
        ed_invest_period_set=None, ed_divest_period_set=None,
        ed_invest_max_period=None, ed_divest_max_period=None,
    )
    # ``ed_invest.csv`` etc. are the canonical Python-preprocessing
    # outputs that .mod reads via ``table data IN`` (flextool.mod:1428).
    # The ``solve__`` prefixed twins are .mod printf debug-exports of
    # the *current solve's* subset and must NOT be used as inputs —
    # using them silently drops invest variables for non-realized
    # periods (e.g. p2025 in a 2-period invest scenario), making the
    # LP smaller than .mod's by exactly the missing periods.
    def _read_invest_set(name: str, kind_col: str) -> pl.DataFrame:
        path = sd / f"{name}.csv"
        if not path.exists():
            return pl.DataFrame(schema={kind_col: pl.Utf8, "d": pl.Utf8})
        df = pl.read_csv(path)
        if df.height == 0:
            return pl.DataFrame(schema={kind_col: pl.Utf8, "d": pl.Utf8})
        rename_src = ("entity" if "entity" in df.columns
                      else "node" if "node" in df.columns
                      else "process")
        return df.rename({rename_src: kind_col, "period": "d"}).select(kind_col, "d")

    ed_inv = _read_invest_set("ed_invest", "e")
    ed_div = _read_invest_set("ed_divest", "e")
    if ed_inv.height == 0 and ed_div.height == 0:
        return blank

    # ed_invest_forbidden_no_investment: entities that may NOT invest in
    # specified periods (lifetime_method=no_investment combined with
    # invest_method=invest_no_limit at periods where the lifetime window
    # disallows new build).  flextool encodes this as
    # ``fix_v_invest_no_investment_eq`` pinning the variable to 0 — we
    # achieve the same effect by removing the (entity, period) tuple
    # from every invest set so the variable is never created.
    forbid_path = sd / "ed_invest_forbidden_no_investment.csv"
    if forbid_path.exists():
        forbid = pl.read_csv(forbid_path)
        if forbid.height > 0:
            forbid = forbid.rename({"entity": "e", "period": "d"}).select("e", "d")
            ed_inv = ed_inv.join(forbid, on=["e", "d"], how="anti")

    pd_inv = _read_invest_set("pd_invest", "p")
    pd_div = _read_invest_set("pd_divest", "p")
    nd_inv = _read_invest_set("nd_invest", "n")
    nd_div = _read_invest_set("nd_divest", "n")
    # Only keep nd/pd entries whose entity also appears in ed_inv/ed_div
    # — the ed_* set is the "current-solve" set in the post-solve fixtures
    # we read; pd_/nd_ are structural unions and may be wider.
    if ed_inv.height > 0:
        pd_inv = pd_inv.join(ed_inv.rename({"e": "p"}), on=["p", "d"], how="inner")
        nd_inv = nd_inv.join(ed_inv.rename({"e": "n"}), on=["n", "d"], how="inner")
    else:
        pd_inv = pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8})
        nd_inv = pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8})
    if ed_div.height > 0:
        pd_div = pd_div.join(ed_div.rename({"e": "p"}), on=["p", "d"], how="inner")
        nd_div = nd_div.join(ed_div.rename({"e": "n"}), on=["n", "d"], how="inner")
    else:
        pd_div = pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8})
        nd_div = pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8})

    edd_inv_path = sd / "edd_invest.csv"
    if edd_inv_path.exists():
        edd_inv_df = pl.read_csv(edd_inv_path)
        cols = set(edd_inv_df.columns)
        if {"d_invest", "d", "entity"}.issubset(cols):
            edd_inv = edd_inv_df.rename({"entity": "e"}).select("e", "d_invest", "d")
        elif {"period_history", "period", "entity"}.issubset(cols):
            # newer flextool preprocessing: ``period_history`` is the
            # invest-period, ``period`` is the dispatch-period.
            edd_inv = (edd_inv_df.rename({"entity": "e",
                                           "period_history": "d_invest",
                                           "period": "d"})
                                   .select("e", "d_invest", "d"))
        else:
            edd_inv = pl.DataFrame(schema={"e": pl.Utf8,
                                            "d_invest": pl.Utf8, "d": pl.Utf8})
    else:
        edd_inv = pl.DataFrame(schema={"e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8})

    # Drop edd_inv rows whose invest period (d_invest) was filtered out
    # of ed_inv by ed_invest_forbidden_no_investment — those reference
    # non-existent variables.
    if edd_inv.height > 0 and ed_inv.height > 0:
        edd_inv = edd_inv.join(
            ed_inv.rename({"d": "d_invest"}), on=["e", "d_invest"], how="inner")

    # edd_divest_active = (p, d_divest, d) where (p, d_divest) ∈ pd_divest
    # and d_divest ≤ d.  Use p_years_d to order.
    pyd_path = sd / "p_years_d.csv"
    pyd = None
    if pyd_path.exists():
        pyd_raw = pl.read_csv(pyd_path)
        # Column for years may be 'value' (long format) or 'p_years_d' (legacy).
        yr_col = "value" if "value" in pyd_raw.columns else "p_years_d"
        pyd = pyd_raw.rename({"period": "d", yr_col: "yr"}).select("d", "yr")
    if pyd is not None and pd_div.height > 0:
        # cross-join pd_divest with all (d) in dt × periods, filter by year ordering
        dt_period = dt.select("d").unique()
        edd_div = (pd_div.rename({"d": "d_divest"})
                          .join(dt_period, how="cross")
                          .join(pyd.rename({"d": "d_divest", "yr": "yr_divest"}), on="d_divest")
                          .join(pyd.rename({"yr": "yr"}), on="d")
                          .filter(pl.col("yr_divest") <= pl.col("yr"))
                          .select("p", "d_divest", "d"))
    else:
        edd_div = pl.DataFrame(schema={"p": pl.Utf8, "d_divest": pl.Utf8, "d": pl.Utf8})

    # edd_invest_lookback: edd_invest filtered to year[d_invest] < year[d].
    # Used by the user-constraint prebuilt-capacity LHS (mod:2885-2898) to
    # add Σ_{d_invest < d} v_invest[p, d_invest] · coef[p, c] · unitsize[p].
    # Without this, the prebuilt LHS only carries the static existing
    # term and any constraint that depends on cumulative prior invests
    # (e.g. ``wind_growth_cap`` on multi_year_wind_growth_cap) is too
    # tight by exactly the missing variable contribution.
    if pyd is not None and edd_inv.height > 0:
        edd_inv_lookback = (edd_inv
            .join(pyd.rename({"d": "d_invest", "yr": "yr_invest"}), on="d_invest")
            .join(pyd.rename({"yr": "yr"}), on="d")
            .filter(pl.col("yr_invest") < pl.col("yr"))
            .select("e", "d_invest", "d"))
    else:
        edd_inv_lookback = pl.DataFrame(
            schema={"e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8})

    p_max_units = Param(("e", "d"),
        _read_wide_e_d(sd / "p_entity_max_units.csv")
            .filter(pl.col("value") > 0)
            .select("e", "d", "value")) if (sd / "p_entity_max_units.csv").exists() else None

    def _cost_param(name: str, dims=("e", "d"), per_e: bool = True) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists():
            return None
        if per_e:
            df = _read_wide_e_d(f).filter(pl.col("value") != 0)
            if df.height == 0:
                return None
            return Param(dims, df.select(*dims, "value"))
        else:
            df = pl.read_csv(f).drop("solve").rename({"period": "d"})
            if df.height == 0:
                return None
            return Param(dims, df.select(*dims, "value"))

    # Per-entity total caps (across all periods)
    def _e_total_set(name: str) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        return df.rename({"entity": "e"}).select("e")
    def _e_total_param(name: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        return Param(("e",), df.rename({"entity": "e"}).select("e", "value"))

    # Per-period invest/divest caps.  ``ed_invest_period`` is the set of
    # (e, d) pairs for which a per-period upper bound applies;
    # ``ed_invest_max_period`` is the cap (in absolute units, post-unitsize).
    def _read_period_set(name: str) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        return (df.rename({"entity": "e", "period": "d"}).select("e", "d"))
    def _read_period_cap(name: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        return Param(("e", "d"),
            df.rename({"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                   .fill_null(0.0))
              .select("e", "d", "value"))

    # Multi-solve handoff state.  These files are written between
    # sub-solves; the .mod uses them as constants on the
    # max/min Invest/Divest_entity_total + cumulative-group
    # constraints.  Empty / missing → no prior-solve activity.
    def _read_handoff_e_d(name: str) -> Param | None:
        # Long-format (entity, period, value); cleaned to (e, d, value).
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        df = (df.rename({"entity": "e", "period": "d"})
                .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                     .fill_null(0.0))
                .filter(pl.col("value") != 0.0)
                .select("e", "d", "value"))
        if df.height == 0: return None
        return Param(("e", "d"), df)

    def _read_handoff_e(name: str, value_col: str | None = None) -> Param | None:
        # Wide-format with single value column (entity, p_entity_invested)
        # → (e, value).
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        # Canonical column is "p_entity_invested" / "p_entity_divested"
        if value_col is None:
            non_entity = [c for c in df.columns if c != "entity"]
            if not non_entity: return None
            value_col = non_entity[0]
        df = (df.rename({"entity": "e", value_col: "value"})
                .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                     .fill_null(0.0))
                .filter(pl.col("value") != 0.0)
                .select("e", "value"))
        if df.height == 0: return None
        return Param(("e",), df)

    return dict(
        ed_invest_set=ed_inv if ed_inv.height > 0 else None,
        ed_divest_set=ed_div if ed_div.height > 0 else None,
        pd_invest_set=pd_inv if pd_inv.height > 0 else None,
        pd_divest_set=pd_div if pd_div.height > 0 else None,
        nd_invest_set=nd_inv if nd_inv.height > 0 else None,
        nd_divest_set=nd_div if nd_div.height > 0 else None,
        edd_invest_set=edd_inv if edd_inv.height > 0 else None,
        edd_invest_lookback_set=edd_inv_lookback if edd_inv_lookback.height > 0 else None,
        edd_divest_active=edd_div if edd_div.height > 0 else None,
        p_entity_max_units=p_max_units,
        ed_lifetime_fixed_cost=_cost_param("ed_lifetime_fixed_cost"),
        ed_lifetime_fixed_cost_divest=_cost_param("ed_lifetime_fixed_cost_divest"),
        ed_entity_annual_discounted=_cost_param("ed_entity_annual_discounted"),
        ed_entity_annual_divest_discounted=_cost_param("ed_entity_annual_divest_discounted"),
        e_invest_total=_e_total_set("e_invest_total"),
        e_divest_total=_e_total_set("e_divest_total"),
        e_invest_max_total=_e_total_param("e_invest_max_total"),
        e_divest_max_total=_e_total_param("e_divest_max_total"),
        ed_invest_period_set=_read_period_set("ed_invest_period"),
        ed_divest_period_set=_read_period_set("ed_divest_period"),
        ed_invest_max_period=_read_period_cap("ed_invest_max_period"),
        ed_divest_max_period=_read_period_cap("ed_divest_max_period"),
        p_entity_previously_invested_capacity=_read_handoff_e_d(
            "p_entity_previously_invested_capacity"),
        p_entity_invested=_read_handoff_e("p_entity_invested"),
        p_entity_divested=_read_handoff_e("p_entity_divested"),
    )


def _read_p_process_side(path: Path, side_col: str) -> dict[str, pl.DataFrame]:
    """Parse ``input/p_process_sink.csv`` or ``input/p_process_source.csv``.

    Canonical (Python-preprocessing-input) format: long, columns
    ``[process, sink_or_source, sourceSinkParam, p_process_sink_or_source]``.
    The .mod also printf's a wide debug-export to ``solve_data/`` with
    a 2-row hierarchical header (process row, side row, then
    param/value rows) — supported as a fallback.  Returns
    ``{param_name: DataFrame(p, side, value)}``."""
    out: dict[str, pl.DataFrame] = {}
    if not path.exists():
        return out
    df = pl.read_csv(path)
    # canonical long: ``process, <side>, sourceSinkParam, p_process_<side>``
    if {"process", "sourceSinkParam"}.issubset(df.columns):
        if df.height == 0:
            return out
        # value column is last; rename for uniformity
        value_col = df.columns[-1]
        out_df = (df.filter(pl.col(value_col) != 0)
                    .rename({"process": "p",
                             "sourceSinkParam": "param",
                             value_col: "value"}))
        for param, sub in out_df.group_by("param", maintain_order=True):
            param_str = param[0] if isinstance(param, tuple) else param
            out[param_str] = sub.select("p", side_col, "value")
        return out
    # legacy 2-row-header printf-export format
    import csv
    with path.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return out
    procs = rows[0][1:]
    sides = rows[1][1:]
    n = min(len(procs), len(sides))
    for r in rows[2:]:
        if len(r) < 2:
            continue
        param = r[0]
        if not param:
            continue
        ps, ss, vs = [], [], []
        for i in range(n):
            cell = r[i + 1] if i + 1 < len(r) else None
            if cell is None or cell == "":
                continue
            try:
                v = float(cell)
            except (ValueError, TypeError):
                continue
            if v == 0:
                continue
            ps.append(procs[i]); ss.append(sides[i]); vs.append(v)
        if ps:
            out[param] = pl.DataFrame({"p": ps, side_col: ss, "value": vs})
    return out


def _load_ramp(inp: Path, sd: Path, pss: pl.DataFrame | None) -> dict:
    """Load ramp-limit sets and ramp_speed params.  Empty when no
    process_source_sink_ramp_limit_* row is populated."""
    blank = dict(
        process_source_sink_ramp_limit_sink_up=None,
        process_source_sink_ramp_limit_sink_down=None,
        process_source_sink_ramp_limit_source_up=None,
        process_source_sink_ramp_limit_source_down=None,
        p_ramp_speed_up_sink=None,
        p_ramp_speed_down_sink=None,
        p_ramp_speed_up_source=None,
        p_ramp_speed_down_source=None,
    )
    if pss is None:
        return blank

    def _read_set(name: str) -> pl.DataFrame | None:
        p = sd / f"process_source_sink_ramp_limit_{name}.csv"
        if not p.exists(): return None
        df = pl.read_csv(p)
        if df.height == 0: return None
        return df.rename({"process": "p"}).select("p", "source", "sink")

    sets = {f"process_source_sink_ramp_limit_{name}": _read_set(name)
            for name in ("sink_up", "sink_down", "source_up", "source_down")}
    if not any(s is not None for s in sets.values()):
        return blank

    # Canonical input lives in input/p_process_{sink,source}.csv (long
    # format; .mod reads it via ``table data IN`` at flextool.mod:735).
    # Fall back to the .mod-printf debug-export in solve_data/ for
    # legacy fixtures.
    def _pick(name: str) -> Path:
        a = inp / name; b = sd / name
        return a if a.exists() else b
    sink_params = _read_p_process_side(_pick("p_process_sink.csv"), "sink")
    src_params  = _read_p_process_side(_pick("p_process_source.csv"), "source")
    def _param(d: dict, key: str, side: str) -> Param | None:
        df = d.get(key)
        if df is None or df.height == 0: return None
        return Param(("p", side), df)

    return dict(
        **sets,
        p_ramp_speed_up_sink   = _param(sink_params, "ramp_speed_up",   "sink"),
        p_ramp_speed_down_sink = _param(sink_params, "ramp_speed_down", "sink"),
        p_ramp_speed_up_source   = _param(src_params, "ramp_speed_up",   "source"),
        p_ramp_speed_down_source = _param(src_params, "ramp_speed_down", "source"),
    )


def _load_online(inp: Path, sd: Path, dt: pl.DataFrame,
                  pss: pl.DataFrame | None) -> dict:
    """Load online / min_load / startup data.  Empty dict-of-Nones when
    no process is online."""
    blank = dict(
        process_online=None, process_online_linear=None,
        process_online_integer=None, process_minload=None,
        process_min_load_eff=None,
        p_online_dt=None, pdt_online_linear=None, pdt_online_integer=None,
        p_min_load=None, p_startup_cost=None, p_section=None,
        pdt_uptime_set=None, pdt_downtime_set=None,
        uptime_lookback=None, downtime_lookback=None,
    )
    if pss is None:
        return blank
    online_path = sd / "process_online.csv"
    if not online_path.exists():
        return blank
    p_online = pl.read_csv(online_path).rename({"process": "p"})
    if p_online.height == 0:
        return blank

    p_online_lin = pl.read_csv(sd / "process_online_linear.csv").rename({"process": "p"})
    p_online_int_path = sd / "process_online_integer.csv"
    p_online_int = (pl.read_csv(p_online_int_path).rename({"process": "p"})
                    if p_online_int_path.exists() else
                    pl.DataFrame(schema={"p": pl.Utf8}))
    p_minload_path = sd / "process_minload.csv"
    p_minload = (pl.read_csv(p_minload_path).rename({"process": "p"})
                 if p_minload_path.exists() else
                 pl.DataFrame(schema={"p": pl.Utf8}))

    # ct_method: min_load_efficiency rows.  Canonical input file is
    # ``input/process__ct_method.csv`` with columns (process, ct_method);
    # the .mod also printf's a debug-export to ``solve_data/`` with
    # column ``method`` — tolerate either schema/location.
    ctm_path = inp / "process__ct_method.csv"
    if not ctm_path.exists():
        ctm_path = sd / "process__ct_method.csv"
    p_min_load_eff = pl.DataFrame(schema={"p": pl.Utf8})
    if ctm_path.exists():
        ctm = pl.read_csv(ctm_path).rename({"process": "p"})
        method_col = "ct_method" if "ct_method" in ctm.columns else "method"
        p_min_load_eff = (ctm.filter(pl.col(method_col) == "min_load_efficiency")
                          .select("p").unique())

    # p_online_dt — block-aware variable indexing (process, period, step)
    p_odt = pl.read_csv(sd / "p_online_dt_set.csv").rename({"process": "p", "step": "t"})
    p_odt = p_odt.select("p", "period", "t").rename({"period": "d"})

    # pdProcess — extract min_load and startup_cost
    p_proc = pl.read_csv(inp / "p_process.csv")
    min_load_rows = (p_proc.filter(pl.col("processParam") == "min_load")
                     .rename({"process": "p", "p_process": "value"})
                     .select("p", "value"))
    p_min_load = (Param(("p",), min_load_rows)
                  if min_load_rows.height > 0 else None)

    # startup_cost is per (p, d) — sliced from canonical pdProcess.csv
    # (`pdProcess[p, 'startup_cost', d]` in .mod).
    sc_long = _slice_param(sd / "pdProcess.csv", "process", "startup_cost",
                            has_time=False, rename_entity_to="p")
    p_startup_cost = None
    pdt_online_lin = pdt_online_int = None
    if sc_long is not None:
        sc_long = sc_long.filter(pl.col("value") != 0)
        if sc_long.height > 0:
            p_startup_cost = Param(("p", "d"), sc_long.select("p", "d", "value"))
            # pdt_online_linear: (p, d, t) for online_linear processes with non-zero startup_cost
            sc_p = sc_long.select("p", "d").unique()
            pdt_online_lin = (p_odt.join(p_online_lin, on="p", how="inner")
                                    .join(sc_p, on=["p", "d"], how="inner"))
            if p_online_int.height > 0:
                pdt_online_int = (p_odt.join(p_online_int, on="p", how="inner")
                                        .join(sc_p, on=["p", "d"], how="inner"))

    # pdtProcess_section — wide-per-process file
    sec_path = sd / "pdtProcess_section.csv"
    p_section = None
    if sec_path.exists():
        sec_long = _read_wide_per_entity(sec_path, rename={"entity": "p"})
        if sec_long.height > 0:
            p_section = Param(("p", "d", "t"), sec_long.select("p", "d", "t", "value"))

    # Minimum uptime / downtime: optional 3-col domain set + 5-col lookback
    # frame.  Both files are emitted by .mod's preprocessing; absence means
    # no min-up/down-time on any process in this scenario.
    def _read_pdt_set(name: str) -> pl.DataFrame | None:
        path = sd / f"{name}.csv"
        if not path.exists():
            return None
        df = pl.read_csv(path)
        if df.height == 0:
            return None
        return df.rename({"process": "p", "period": "d", "time": "t"})

    def _read_lookback(name: str) -> pl.DataFrame | None:
        path = sd / f"{name}.csv"
        if not path.exists():
            return None
        df = pl.read_csv(path)
        if df.height == 0:
            return None
        return df.rename({"process": "p", "period": "d", "time": "t",
                           "period_back": "d_back", "time_back": "t_back"})

    return dict(
        process_online=p_online,
        process_online_linear=p_online_lin,
        process_online_integer=p_online_int,
        process_minload=p_minload,
        process_min_load_eff=p_min_load_eff,
        p_online_dt=p_odt,
        pdt_online_linear=pdt_online_lin,
        pdt_online_integer=pdt_online_int,
        p_min_load=p_min_load,
        p_startup_cost=p_startup_cost,
        p_section=p_section,
        pdt_uptime_set=_read_pdt_set("pdt_uptime_set"),
        pdt_downtime_set=_read_pdt_set("pdt_downtime_set"),
        uptime_lookback=_read_lookback("uptime_lookback"),
        downtime_lookback=_read_lookback("downtime_lookback"),
    )


def _load_storage(inp: Path, sd: Path, dt: pl.DataFrame,
                   nb: pl.DataFrame,
                   pss_eff: pl.DataFrame | None,
                   pss_noEff: pl.DataFrame | None,
                   cap_pd: pl.DataFrame | None,
                   unitsize: Param | None) -> dict:
    """Load storage feature: nodeState set, capacity bounds, binding
    methods, dtttdt, and source-side nodeBalance topology.

    Returns dict with all storage-related fields.  Empty if no
    nodeState entries."""
    # Source-side nodeBalance flow mappings.  These describe processes
    # whose source is a balance node — needed for both transmission
    # (network scenarios with no storage) and storage discharge.  Compute
    # unconditionally so a network-without-storage fixture still has the
    # source flow contributions in nodeBalance.
    flow_from_nb_eff = flow_from_nb_noEff = None
    if pss_eff is not None:
        flow_from_nb_eff = (pss_eff
            .filter(pl.col("source").is_in(nb["n"]))
            .with_columns(n=pl.col("source"))
            .select("p","source","sink","n"))
    if pss_noEff is not None:
        flow_from_nb_noEff = (pss_noEff
            .filter(pl.col("source").is_in(nb["n"]))
            .with_columns(n=pl.col("source"))
            .select("p","source","sink","n"))

    # Apply the same block-compatibility filter as in flow_from_n /
    # flow_to_n: arc contributes to node's nodeBalance only if (b_n, b_f)
    # has an overlap row.  Uses process_side_block.csv + entity_block.csv
    # + overlap_set.csv from the solve_data dir.
    psb_p = sd / "process_side_block.csv"
    eb_p = sd / "entity_block.csv"
    ov_p = sd / "overlap_set.csv"
    if (psb_p.exists() and eb_p.exists() and ov_p.exists()
            and (flow_from_nb_eff is not None or flow_from_nb_noEff is not None)):
        psb_l = pl.read_csv(psb_p).rename({"process": "p", "block": "b_f"})
        eb_l = pl.read_csv(eb_p).rename({"entity": "n", "block": "b"})
        ov_l = pl.read_csv(ov_p)
        if psb_l.height > 0 and eb_l.height > 0 and ov_l.height > 0:
            block_compat_l = (ov_l
                .rename({"block_coarse": "b", "block_fine": "b_f"})
                .select("b", "b_f").unique())
            psb_src_l = psb_l.filter(pl.col("side") == "source").select("p", "b_f")
            def _filter_by_compat(df: pl.DataFrame) -> pl.DataFrame:
                if df is None or df.height == 0:
                    return df
                with_blocks = (df
                    .join(psb_src_l, on="p", how="left")
                    .join(eb_l, on="n", how="left"))
                with_blocks = with_blocks.with_columns(
                    b_f=pl.col("b_f").fill_null("default"),
                    b=pl.col("b").fill_null("default"),
                )
                f = (with_blocks
                    .join(block_compat_l, on=["b", "b_f"], how="inner")
                    .select("p", "source", "sink", "n").unique())
                if f.height < df.height and f.height > 0:
                    return f
                return df
            flow_from_nb_eff = _filter_by_compat(flow_from_nb_eff)
            flow_from_nb_noEff = _filter_by_compat(flow_from_nb_noEff)

    # dtttdt — needed for ramps and online dynamics regardless of storage.
    dtttdt = _read_step_previous(sd / "step_previous.csv")

    blank = dict(
        nodeState = None, nodeState_dt = None, nodeState_first_dt = None,
        p_state_upper = None, p_state_unitsize = None,
        p_state_self_discharge = None, p_state_start = None,
        p_state_existing_capacity = None,
        storage_bind_within_timeset = None,
        storage_bind_forward_only = None,
        storage_fix_start = None,
        storage_use_reference_value = None,
        p_storage_state_reference_value = None,
        dtttdt = dtttdt,
        dtttdt_forward_only = None,
        nodeStateBlock = None,
        period_block = None,
        period_block_succ = None,
        period_block_time = None,
        dtttdt_block_interior = None,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = None,
        p_roll_continue_state = None,
        n_fix_storage_quantity = None,
        ndt_fix_storage_quantity = None,
        p_fix_storage_quantity = None,
        dtt_timeline_matching = None,
        period_branch = None,
        period_last = None,
        nodeState_last_dt = None,
        node_profile_upper = None,
        node_profile_lower = None,
        node_profile_fixed = None,
        p_node_availability = None,
    )
    ns_path = sd / "nodeState.csv"
    if not ns_path.exists():
        return blank
    nodeState = pl.read_csv(ns_path).rename({"node": "n"})
    if nodeState.height == 0:
        return blank

    nodeState_dt = nodeState.join(dt, how="cross")

    # First (d, t) per period — used for storage_state_start_binding.
    # The .mod uses ``period_first_of_solve`` for the boundary tests in
    # both the fwd_fix start binding (mod:2197) and the roll_continue
    # term (mod:2196).  ``period_first.csv`` is the legacy single-solve
    # source (often empty in nested / rolling-horizon fixtures), so we
    # prefer ``period_first_of_solve.csv`` when it has rows; otherwise
    # fall back to ``period_first.csv``; otherwise the first dt period.
    fpos_path = sd / "period_first_of_solve.csv"
    fp_path = sd / "period_first.csv"
    first_period = None
    if fpos_path.exists():
        df = pl.read_csv(fpos_path)
        if df.height > 0:
            first_period = df.rename({"period": "d"}).select("d").unique()
    if first_period is None and fp_path.exists():
        df = pl.read_csv(fp_path)
        if df.height > 0:
            first_period = df.rename({"period": "d"}).select("d").unique()
    if first_period is None:
        # Fallback: take the lexicographically smallest period.
        first_period = (dt.select("d").unique()
                          .sort("d").head(1))
    first_dt = (nodeState_dt
        .join(first_period, on="d", how="inner")
        .group_by("n", "d")
        .agg(pl.col("t").min().alias("t"))
        .select("n", "d", "t"))

    # state_upper = capacity / unitsize per (n, d)  (assume node unitsize)
    if unitsize is not None and cap_pd is not None:
        # cap_pd from process side; for nodes we need a node-side capacity.
        cap_long = _read_capacity(sd / "p_entity_period_existing_capacity.csv",
                                   sd / "p_entity_previously_invested_capacity.csv",
                                   sd / "p_entity_all_existing.csv")
        unitsize_long = _read_unitsize((sd / "p_entity_unitsize.csv") if (sd / "p_entity_unitsize.csv").exists() else (inp / "p_entity_unitsize.csv"))
        state_existing = (cap_long.rename({"e":"n","value":"cap"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n","d","cap"))
        state_us_long = (unitsize_long.rename({"e":"n"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n","value"))
        state_existing_capacity = Param(("n","d"),
            state_existing.rename({"cap":"value"}))
        state_unitsize = Param(("n",), state_us_long)
        state_upper_long = (state_existing
            .join(state_us_long.rename({"value":"us"}), on="n", how="inner")
            .with_columns(value=pl.col("cap")/pl.col("us"))
            .select("n","d","value"))
        state_upper = Param(("n","d"), state_upper_long)
    else:
        state_unitsize = state_existing_capacity = state_upper = None

    # p_node parameters (storage_state_start, self_discharge_loss).
    p_node = pl.read_csv(inp / "p_node.csv")
    def _node_param(name: str) -> Param | None:
        rows = (p_node.filter(pl.col("nodeParam") == name)
                       .rename({"node":"n","p_node":"value"})
                       .select("n","value"))
        if rows.height == 0: return None
        return Param(("n",), rows)
    state_self_discharge = _node_param("self_discharge_loss")
    state_start = _node_param("storage_state_start")

    # Binding methods (sd-level, per node).
    # NOTE: the .mod attaches a (v_state[t] - v_state[t-1]) term in
    # nodeBalance for several binding methods, with subtle differences:
    #   * ``bind_within_timeset``  — fully cyclic (wraps via
    #     ``t_previous_within_timeset``).
    #   * ``bind_within_period``   — cyclic within period
    #     (``t_previous`` column).
    #   * ``bind_within_solve``    — cyclic within solve
    #     (``t_previous_within_solve``); equivalent to within_timeset for
    #     a single-block dispatch.
    #   * ``bind_forward_only``    — also uses
    #     ``t_previous_within_solve``, BUT the .mod *omits* the
    #     state-change term at the first timestep of the first period
    #     (line 2188 condition).  This makes the storage non-cyclic at
    #     the boundary.
    # In the current flexpy parity tests, every fixture that *does*
    # exercise a state node uses ``bind_within_timeset`` (which is what
    # this loader picks up).  ``work_water_pump`` is the only fixture
    # that uses ``bind_forward_only``, and faithful parity there
    # requires modelling the first-timestep exemption — see
    # questions_for_user.md#water_pump.
    sbm_path = sd / "node__storage_binding_method.csv"
    binding_within_timeset = None
    binding_forward_only = None
    binding_within_solve = None
    if sbm_path.exists():
        sbm = pl.read_csv(sbm_path)
        # Column names in this file have varied — handle both schemas
        if "storage_binding_method" in sbm.columns:
            sbm = sbm.rename({"node":"n","storage_binding_method":"method"})
        elif "method" in sbm.columns:
            sbm = sbm.rename({"node":"n"})
        binding_within_timeset = (sbm.filter(pl.col("method")=="bind_within_timeset")
                                     .select("n").unique())
        fo = (sbm.filter(pl.col("method")=="bind_forward_only")
                 .select("n").unique())
        if fo.height > 0:
            binding_forward_only = fo
        ws = (sbm.filter(pl.col("method")=="bind_within_solve")
                 .select("n").unique())
        if ws.height > 0:
            binding_within_solve = ws

    # ``bind_forward_only`` mirrors ``bind_within_solve`` (uses the
    # ``t_previous_within_solve`` lag column) BUT the .mod omits the
    # state-change term at the very first timestep of the first period
    # (flextool.mod:2188).  We model that exemption by dropping the
    # corresponding row from the lag frame — the wrap row whose
    # ``t_previous_within_solve`` jumps backwards.  Sorting by (d, t)
    # and dropping the first row is equivalent for single-solve fixtures
    # (flexpy is single-solve per build).
    dtttdt_forward_only_df = None
    if binding_forward_only is not None and dtttdt is not None and dtttdt.height > 0:
        dtttdt_forward_only_df = dtttdt.sort("d", "t").slice(1)
        if dtttdt_forward_only_df.height == 0:
            dtttdt_forward_only_df = None

    # node__storage_start_end_method is read by .mod from input/
    # (flextool.mod:662) — that's the canonical user-input source.
    # solve_data/ may have a .mod-printf debug-export with renamed
    # column "method"; tolerate either schema.
    sse_path = inp / "node__storage_start_end_method.csv"
    if not sse_path.exists():
        sse_path = sd / "node__storage_start_end_method.csv"
    fix_start = None
    fix_end = None
    fix_start_end = None
    if sse_path.exists():
        sse = pl.read_csv(sse_path)
        if "storage_start_end_method" in sse.columns:
            sse = sse.rename({"node":"n","storage_start_end_method":"method"})
        elif "method" in sse.columns:
            sse = sse.rename({"node":"n"})
        fix_start = (sse.filter(pl.col("method")=="fix_start").select("n").unique())
        fix_end = (sse.filter(pl.col("method")=="fix_end").select("n").unique())
        fix_start_end = (sse.filter(pl.col("method")=="fix_start_end").select("n").unique())

    # node__storage_solve_horizon_method (.mod:663): nodes with method
    # ``use_reference_value`` get a v_state pin at the last timestep of
    # the last period, equal to ``reference_value × existing/unitsize``.
    sshm_path = inp / "node__storage_solve_horizon_method.csv"
    if not sshm_path.exists():
        sshm_path = sd / "node__storage_solve_horizon_method.csv"
    use_reference_value = None
    if sshm_path.exists():
        sshm = pl.read_csv(sshm_path)
        col = ("storage_solve_horizon_method"
               if "storage_solve_horizon_method" in sshm.columns
               else "method")
        sshm = sshm.rename({"node": "n", col: "method"})
        use_reference_value = (sshm
            .filter(pl.col("method") == "use_reference_value")
            .select("n").unique())
        # Filter out nodes with a competing storage method (mod:2806-2811):
        # fix_end / fix_start_end / bind_within_solve / bind_within_period /
        # bind_within_timeset / bind_intraperiod_blocks.
        # ``nodeStateBlock`` is the set carrying bind_intraperiod_blocks
        # (loaded below; we look it up via the on-disk CSV here to keep
        # ordering simple).  bind_within_period not exercised yet.
        nsb_for_excl = None
        nsb_path_local = sd / "nodeStateBlock.csv"
        if nsb_path_local.exists():
            nsb_df_local = pl.read_csv(nsb_path_local)
            if nsb_df_local.height > 0:
                nsb_for_excl = nsb_df_local.rename({"node": "n"}).select("n")
        for excl in (fix_end, fix_start_end,
                     binding_within_solve, binding_within_timeset,
                     nsb_for_excl):
            if excl is not None and excl.height > 0:
                use_reference_value = use_reference_value.join(
                    excl, on="n", how="anti")
        if use_reference_value.height == 0:
            use_reference_value = None

    # storage_state_reference_value: sliced from canonical pdtNode.csv
    # (parameter = ``storage_state_reference_value``).
    p_ssrv = None
    if use_reference_value is not None:
        ssrv_long = _slice_param(sd / "pdtNode.csv", "node",
                                  "storage_state_reference_value",
                                  rename_entity_to="n")
        if ssrv_long is not None and ssrv_long.height > 0:
            ssrv_long = (ssrv_long
                .filter(pl.col("n").is_in(use_reference_value["n"])))
            if ssrv_long.height > 0:
                p_ssrv = Param(("n", "d", "t"), ssrv_long)

    # ─── Intraperiod-block (bind_intraperiod_blocks) sets ────────────────
    # Only loaded if the corresponding solve_data CSVs exist.  Used by
    # ``stateConstantWithinBlock_eq`` and ``nodeBalanceBlock_eq`` in
    # model.py for nodes whose binding method is ``bind_intraperiod_blocks``.
    nodeStateBlock = None
    nsb_path = sd / "nodeStateBlock.csv"
    if nsb_path.exists():
        df = pl.read_csv(nsb_path)
        if df.height > 0:
            nodeStateBlock = df.rename({"node": "n"}).select("n").unique()

    period_block = None
    pb_path = sd / "period_block_set.csv"
    if pb_path.exists():
        df = pl.read_csv(pb_path)
        if df.height > 0:
            period_block = (df
                .rename({"period": "d", "block_first": "b_first"})
                .select("d", "b_first")
                .unique())

    period_block_succ = None
    pbs_path = sd / "period_block_succ.csv"
    if pbs_path.exists():
        df = pl.read_csv(pbs_path)
        if df.height > 0:
            period_block_succ = (df
                .rename({"period": "d", "block_first": "b_first",
                         "block_first_next": "b_next"})
                .select("d", "b_first", "b_next"))

    period_block_time = None
    pbt_path = sd / "period_block_time.csv"
    if pbt_path.exists():
        df = pl.read_csv(pbt_path)
        if df.height > 0:
            period_block_time = (df
                .rename({"period": "d", "block_first": "b_first",
                         "step": "t"})
                .select("d", "b_first", "t"))

    # Interior-of-block dtttdt rows: rows where the within-timeset previous
    # equals the plain (within-period) previous — i.e. NOT the block wrap row.
    # These pin v_state[n,d,t] = v_state[n,d,t_previous] for nodeStateBlock
    # nodes (mod:2322-2326, jump=1 interior rows).
    dtttdt_block_interior = None
    if (dtttdt is not None and dtttdt.height > 0
            and "t_previous_within_timeset" in dtttdt.columns
            and "t_previous" in dtttdt.columns):
        dtttdt_block_interior = (dtttdt
            .filter(pl.col("t_previous_within_timeset")
                    == pl.col("t_previous"))
            .select("d", "t", "t_previous"))
        if dtttdt_block_interior.height == 0:
            dtttdt_block_interior = None

    # ─── Multi-resolution block synthesis ───────────────────────────────
    # If the fixture defines temporal-resolution blocks coarser than 'default'
    # via entity_block.csv + block_step_duration.csv (e.g. lh2_three_region's
    # daily_group), repurpose the existing nodeStateBlock / period_block_*
    # infrastructure to emit one nodeBalance per (n, d, b_first) for nodes
    # whose block != 'default'.  This is exactly what flextool's .mod does at
    # nodeBalance_eq line 2185+ via the overlap M-matrix, in the degenerate
    # case where every overlap row has fraction 1.0.
    #
    # The rewrite is *opt-in*: only fires when entity_block.csv assigns a
    # non-'default' block to one or more nodeBalance nodes.  Fixtures without
    # entity_block.csv (or with everything set to 'default') keep their
    # existing pre-v51 hourly nodeBalance.
    bsd_path = sd / "block_step_duration.csv"
    eb_path2 = sd / "entity_block.csv"
    if (eb_path2.exists() and bsd_path.exists()
            and nb is not None and nb.height > 0):
        eb2 = pl.read_csv(eb_path2)
        if eb2.height > 0:
            eb2 = eb2.rename({"entity": "n", "block": "b"}).select("n", "b")
            # Identify *coarse* blocks: those whose ``block_step_duration``
            # contains any entry > 1.  ``hourly_group`` (24-hourly grid with
            # sd=1.0 per row) reduces to the default fine grid and shouldn't
            # trigger the rewrite.  Only blocks with at least one row of
            # step_duration > 1 (e.g. ``daily_group`` with sd=24.0) need the
            # M-matrix aggregation.
            #
            # Additional gate: the rewrite only makes semantic sense when
            # MULTIPLE distinct blocks are in use (e.g. lh2 has both
            # ``hourly_group`` and ``daily_group``; some entities live on
            # the fine grid while others aggregate to coarse).  A fixture
            # with a single block (typically ``default``) is just a single
            # natural-resolution LP — the per-step duration may exceed 1
            # (e.g. 6h for storage_fullYear_6h) but there's no overlap /
            # M-matrix structure to synthesise.  Without this gate the
            # rewrite blanket-promotes every nodeBalance entity into
            # nodeStateBlock, killing the per-step nodeBalance_eq AND
            # forcing v_state to be constant across the period via the
            # stateConstantWithinBlock_eq cyclic chain (battery can't
            # charge/discharge → ~3-5% obj-value gap).
            bsd_full = pl.read_csv(bsd_path)
            distinct_blocks = bsd_full["block"].unique().to_list()
            if len(distinct_blocks) < 2:
                coarse_blocks = []
            else:
                coarse_blocks = (bsd_full
                    .filter(pl.col("step_duration") > 1.0)["block"]
                    .unique().to_list())
            # Restrict to nodeBalance nodes with a *coarse* block.
            non_default_nodes = (eb2
                .filter(pl.col("b").is_in(coarse_blocks))
                .join(nb, on="n", how="inner"))
            if non_default_nodes.height > 0:
                bsd = bsd_full.rename(
                    {"period": "d", "step": "b_first"})
                # Keep only coarse blocks (b in non_default_nodes' set).
                bsd = bsd.filter(
                    pl.col("block").is_in(non_default_nodes["b"].unique()))
                if bsd.height > 0:
                    # period_block: union of (d, b_first) per non-default
                    # block — each block's coarse timesteps are the b_first
                    # rows of block_step_duration.
                    new_pb = bsd.select("d", "b_first").unique()
                    # period_block_succ: cyclic chain of consecutive
                    # block_first values within each (block, period) — order
                    # is lexicographic on b_first (t0001 < t0025 < … < t0145).
                    succ_rows = []
                    for (blk, dval), grp in (bsd
                            .sort("block", "d", "b_first")
                            .group_by(["block", "d"], maintain_order=True)):
                        b_firsts = grp["b_first"].to_list()
                        n_blk = len(b_firsts)
                        for i in range(n_blk):
                            cur = b_firsts[i]
                            nxt = b_firsts[(i + 1) % n_blk]
                            succ_rows.append((dval, cur, nxt))
                    new_pbs = (pl.DataFrame(
                        succ_rows, schema=["d", "b_first", "b_next"],
                        orient="row")
                        if succ_rows else None)
                    # period_block_time: derived from overlap_set
                    # (b_coarse=non-default, b_fine=default) — every fine
                    # 'default' step that overlaps a coarse step.
                    new_pbt = None
                    ov_path = sd / "overlap_set.csv"
                    if ov_path.exists():
                        ov = pl.read_csv(ov_path)
                        if ov.height > 0:
                            ov = ov.rename({
                                "period": "d",
                                "block_coarse": "b",
                                "step_coarse": "b_first",
                                "block_fine": "b_fine",
                                "step_fine": "t",
                            })
                            ov_keep = ov.filter(
                                pl.col("b").is_in(non_default_nodes["b"].unique())
                                & (pl.col("b_fine") == "default"))
                            if ov_keep.height > 0:
                                new_pbt = ov_keep.select(
                                    "d", "b_first", "t").unique()
                    if new_pbt is None:
                        # Fallback: synthesise period_block_time from bsd —
                        # each daily block (b_first, step_duration=N) covers
                        # the next N consecutive 'default' fine steps.
                        new_pbt = None
                    # Synthesised dtttdt_block_interior: hours within the
                    # same daily block need t→t_prev (previous hour within
                    # day).  Built from new_pbt: for each (d, b_first),
                    # consecutive sorted t's give (d, t, t_previous=t_prev).
                    new_dbi = None
                    if new_pbt is not None and new_pbt.height > 0:
                        ints = []
                        for (dval, bf), grp in (new_pbt
                                .sort("d", "b_first", "t")
                                .group_by(["d", "b_first"], maintain_order=True)):
                            ts = grp["t"].to_list()
                            for i in range(1, len(ts)):
                                ints.append((dval, ts[i], ts[i - 1]))
                        if ints:
                            new_dbi = pl.DataFrame(
                                ints, schema=["d", "t", "t_previous"],
                                orient="row").unique()
                    # *Replace* the existing block sets with the
                    # synthesised coarse-block versions.  flextool's runner
                    # writes degenerate identity rows (single-block
                    # period_block_time spanning all 168 hours, self-loop
                    # period_block_succ) when there's no
                    # bind_intraperiod_blocks node, but those rows are wrong
                    # for daily-aggregation (they'd put every fine hour into
                    # one giant block).  When coarse blocks are present we
                    # use the synthesised daily structure exclusively.  The
                    # 5weeks_battery_intraperiod_blocks fixture has all
                    # entities at block='default' so this branch is a no-op
                    # there and the existing CSVs are kept verbatim.
                    if new_pb is not None and new_pb.height > 0:
                        period_block = new_pb
                    if new_pbs is not None and new_pbs.height > 0:
                        period_block_succ = new_pbs
                    if new_pbt is not None and new_pbt.height > 0:
                        period_block_time = new_pbt
                    if new_dbi is not None and new_dbi.height > 0:
                        dtttdt_block_interior = new_dbi
                    # Add ALL non-default-block nodeBalance nodes (state +
                    # non-state) to nodeStateBlock so the existing
                    # nodeBalanceBlock_eq path handles them.  For non-state
                    # nodes (h2 here), v_state isn't declared — the state-
                    # change term becomes 0 by virtue of empty v_state frame,
                    # and the constraint reduces to a daily-aggregated
                    # instantaneous balance.
                    new_nsb = non_default_nodes.select("n").unique()
                    if nodeStateBlock is None:
                        nodeStateBlock = new_nsb
                    else:
                        nodeStateBlock = pl.concat(
                            [nodeStateBlock, new_nsb],
                            how="vertical").unique()


    # ─── Rolling-horizon nested-solve framework (flextool.mod:2196 + 2760) ─
    # p_nested_model.csv: { modelParam, p_nested_model } with rows
    # solveFirst / solveLast.  Tri-state: missing → None (single-solve);
    # 0 → False; non-zero → True.
    p_nested_solve_first: bool | None = None
    nm_path = sd / "p_nested_model.csv"
    if nm_path.exists():
        nm = pl.read_csv(nm_path)
        if nm.height > 0:
            # Column may be ``p_nested_model`` (canonical) or ``value``.
            value_col = "p_nested_model" if "p_nested_model" in nm.columns else "value"
            row = nm.filter(pl.col("modelParam") == "solveFirst")
            if row.height > 0:
                p_nested_solve_first = bool(int(row[value_col][0]))

    # p_roll_continue_state per node — values handed off from the previous
    # sub-solve (.mod writes this at end-of-solve via fn_p_roll_continue_state).
    p_roll_continue_state = None
    rcs_path = sd / "p_roll_continue_state.csv"
    if rcs_path.exists():
        df = pl.read_csv(rcs_path)
        # Tolerate a leading-space column header (".mod writes 'node, p_roll_…'").
        df.columns = [c.strip() for c in df.columns]
        if df.height > 0:
            df = (df.rename({"node": "n", "p_roll_continue_state": "value"})
                    .with_columns(value=pl.col("value").cast(pl.Float64))
                    .select("n", "value"))
            p_roll_continue_state = Param(("n",), df)

    # n_fix_storage_quantity (set), ndt_fix_storage_quantity (n, d_upper, t_upper),
    # p_fix_storage_quantity (Param), and dtt_timeline_matching (d, t, t_upper).
    n_fix_storage_quantity = None
    nfsq_path = sd / "n_fix_storage_quantity_set.csv"
    if nfsq_path.exists():
        df = pl.read_csv(nfsq_path)
        if df.height > 0:
            n_fix_storage_quantity = df.rename({"node": "n"}).select("n").unique()

    ndt_fix_storage_quantity = None
    p_fix_storage_quantity = None
    fsq_path = sd / "fix_storage_quantity.csv"
    if fsq_path.exists():
        df = pl.read_csv(fsq_path)
        if df.height > 0:
            df = (df.rename({"period": "d", "step": "t", "node": "n",
                              "p_fix_storage_quantity": "value"})
                    .with_columns(value=pl.col("value").cast(pl.Float64))
                    .select("n", "d", "t", "value"))
            ndt_fix_storage_quantity = df.select("n", "d", "t").unique()
            p_fix_storage_quantity = Param(("n", "d", "t"), df)

    dtt_timeline_matching = None
    tm_path = sd / "timeline_matching_map.csv"
    if tm_path.exists():
        df = pl.read_csv(tm_path)
        if df.height > 0:
            # Schema: period, step, upper_step → (d, t, t_upper).  We rename
            # ``upper_step`` to ``t_upper`` so the model.py constraint can
            # join on (d, t) and emit RHS via t_upper → t2 in p_fix_storage_quantity.
            dtt_timeline_matching = (df
                .rename({"period": "d", "step": "t", "upper_step": "t_upper"})
                .select("d", "t", "t_upper")
                .unique())

    # period__branch: (d_upper, d).  Used to map sub-solve period d to
    # upper-level branch d_upper (the "anchor" period of fix_storage_quantity).
    period_branch = None
    pb_path = sd / "period__branch.csv"
    if pb_path.exists():
        df = pl.read_csv(pb_path)
        if df.height > 0:
            period_branch = (df
                .rename({"period": "d", "branch": "d_upper"})
                .select("d_upper", "d")
                .unique())

    # period_last: (d,).
    period_last_df = None
    pl_path = sd / "period_last.csv"
    if pl_path.exists():
        df = pl.read_csv(pl_path)
        if df.height > 0:
            period_last_df = df.rename({"period": "d"}).select("d").unique()

    # nodeState_last_dt: (n, d, t) — last (d, t) per node, used as the index
    # for ``node_balance_fix_quantity_eq_lower``.  Built from
    # block_period_time_last (b, d, t) × entity_block (e=n, b) × nodeState.
    nodeState_last_dt = None
    bptl_path = sd / "block_period_time_last.csv"
    eb_path = sd / "entity_block.csv"
    if (nodeState is not None and nodeState.height > 0
            and bptl_path.exists() and eb_path.exists()):
        bptl = pl.read_csv(bptl_path)
        if bptl.height > 0:
            bptl = bptl.rename({"block": "b", "period": "d", "step": "t"}).select("b", "d", "t")
            eb = pl.read_csv(eb_path)
            if eb.height > 0:
                eb = eb.rename({"entity": "n", "block": "b"}).select("n", "b")
                nodeState_last_dt = (nodeState.select("n")
                    .join(eb, on="n", how="inner")
                    .join(bptl, on="b", how="inner")
                    .select("n", "d", "t").unique())
                if nodeState_last_dt.height == 0:
                    nodeState_last_dt = None

    # ─── State-profile bounds — node__profile__profile_method ────────────
    # Maps (n, f, method) where method ∈ {upper_limit, lower_limit, fixed}.
    # Drives ``profile_state_*_limit`` constraints on v_state.  Located in
    # input/ (canonical user data), with a fallback to solve_data/ for
    # debug exports.
    node_profile_upper_df = node_profile_lower_df = node_profile_fixed_df = None
    npp_path = inp / "node__profile__profile_method.csv"
    if not npp_path.exists():
        npp_path = sd / "node__profile__profile_method.csv"
    if npp_path.exists():
        npp = pl.read_csv(npp_path)
        if npp.height > 0:
            npp = npp.rename({"node": "n", "profile": "f"})
            method_col = ("profile_method" if "profile_method" in npp.columns
                          else "method")
            up = (npp.filter(pl.col(method_col) == "upper_limit")
                     .select("n", "f"))
            lo = (npp.filter(pl.col(method_col) == "lower_limit")
                     .select("n", "f"))
            fx = (npp.filter(pl.col(method_col) == "fixed")
                     .select("n", "f"))
            node_profile_upper_df = up if up.height > 0 else None
            node_profile_lower_df = lo if lo.height > 0 else None
            node_profile_fixed_df = fx if fx.height > 0 else None

    # Node availability (n, d, t) — sliced from pdtNode[n, 'availability', d, t].
    # Used as an RHS multiplier on profile_state_* constraints (mod:2645).
    p_node_avail = None
    avail_long = _slice_param(sd / "pdtNode.csv", "node", "availability",
                                rename_entity_to="n")
    if avail_long is not None and avail_long.height > 0:
        # Restrict to nodeState entries.
        avail_long = (avail_long
            .filter(pl.col("n").is_in(nodeState["n"])))
        if avail_long.height > 0:
            p_node_avail = Param(("n", "d", "t"), avail_long)

    return dict(
        nodeState = nodeState,
        nodeState_dt = nodeState_dt,
        nodeState_first_dt = first_dt,
        p_state_upper = state_upper,
        p_state_unitsize = state_unitsize,
        p_state_self_discharge = state_self_discharge,
        p_state_start = state_start,
        p_state_existing_capacity = state_existing_capacity,
        storage_bind_within_timeset = binding_within_timeset,
        storage_bind_forward_only = binding_forward_only,
        storage_bind_within_solve = binding_within_solve,
        storage_fix_start = fix_start,
        storage_use_reference_value = use_reference_value,
        p_storage_state_reference_value = p_ssrv,
        dtttdt = dtttdt,
        dtttdt_forward_only = dtttdt_forward_only_df,
        nodeStateBlock = nodeStateBlock,
        period_block = period_block,
        period_block_succ = period_block_succ,
        period_block_time = period_block_time,
        dtttdt_block_interior = dtttdt_block_interior,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = p_nested_solve_first,
        p_roll_continue_state = p_roll_continue_state,
        n_fix_storage_quantity = n_fix_storage_quantity,
        ndt_fix_storage_quantity = ndt_fix_storage_quantity,
        p_fix_storage_quantity = p_fix_storage_quantity,
        dtt_timeline_matching = dtt_timeline_matching,
        period_branch = period_branch,
        period_last = period_last_df,
        nodeState_last_dt = nodeState_last_dt,
        node_profile_upper = node_profile_upper_df,
        node_profile_lower = node_profile_lower_df,
        node_profile_fixed = node_profile_fixed_df,
        p_node_availability = p_node_avail,
    )


def _load_profiles(inp: Path, sd: Path, pss: pl.DataFrame | None,
                    unitsize: Param | None,
                    cap_pd: pl.DataFrame | None):
    """Load profile_flow_upper/lower/fixed mappings.  ``cap_pd`` is the
    (p, d, base_cap) frame; combined with unitsize we get the
    ``existing_count`` term used on the RHS."""
    if pss is None or unitsize is None or cap_pd is None:
        return [None]*6
    pp_path = sd / "process__source__sink__profile__profile_method.csv"
    if not pp_path.exists():
        return [None]*6
    pp = pl.read_csv(pp_path).rename({"process":"p"})
    if pp.height == 0:
        return [None]*6
    method_col = "method" if "method" in pp.columns else "profile_method"
    upper = pp.filter(pl.col(method_col)=="upper_limit").select("p","source","sink","profile")
    lower = pp.filter(pl.col(method_col)=="lower_limit").select("p","source","sink","profile")
    fixed = pp.filter(pl.col(method_col)=="fixed").select("p","source","sink","profile")

    # profile values - file is solve, period, time, p1, p2... — wide per profile.
    pdt_profile = sd / "pdtProfile.csv"
    profile_value = None
    if pdt_profile.exists():
        prof_long = _read_wide_per_entity(pdt_profile, rename={"entity":"f"})
        if "profile" in upper.columns:
            upper = upper.rename({"profile": "f"})
            lower = lower.rename({"profile": "f"})
            fixed = fixed.rename({"profile": "f"})
        profile_value = Param(("f","d","t"), prof_long.select("f","d","t","value"))

    # existing_count = capacity / unitsize per (p, d).  For our scenarios
    # (no investment yet) this equals base_cap_pd.
    existing_count = Param(("p","d"), cap_pd.rename({"base":"value"}))

    # availability sliced from canonical pdtProcess.csv —
    # `pdtProcess[p, 'availability', d, t]` in .mod.
    avail_long = _slice_param(sd / "pdtProcess.csv", "process", "availability",
                               rename_entity_to="p")
    availability = (Param(("p","d","t"), avail_long)
                    if avail_long is not None else None)

    return upper, lower, fixed, profile_value, existing_count, availability


# ---------------------------------------------------------------------------
# The single loader

def _load_varcost(sd: Path, pss: pl.DataFrame | None) -> dict:
    """Load process variable-cost (other_operational_cost) sets and Params.

    The .mod has 4 disjoint sets:
      pssdt_varCost_noEff       — uses pdtProcess__source__sink__dt_varCost
      pssdt_varCost_eff_unit_source — uses pdtProcess_source[…,'other_operational_cost']
      pssdt_varCost_eff_unit_sink   — uses pdtProcess_sink[…,'other_operational_cost']
      pssdt_varCost_eff_connection  — uses pdtProcess[…,'other_operational_cost']
    """
    blank = dict(
        pssdt_varCost_noEff=None,
        pssdt_varCost_eff_unit_source=None,
        pssdt_varCost_eff_unit_sink=None,
        pssdt_varCost_eff_connection=None,
        p_pssdt_varCost=None,
        p_pdt_varCost_source=None,
        p_pdt_varCost_sink=None,
        p_pdt_varCost_process=None,
    )
    if pss is None:
        return blank

    def _read_pssdt_set(name: str) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not f.exists():
            return None
        df = pl.read_csv(f)
        if df.height == 0:
            return None
        return df.rename({"process": "p", "period": "d", "time": "t"}) \
                 .select("p", "source", "sink", "d", "t")

    pssdt_noEff = _read_pssdt_set("pssdt_varCost_noEff")
    pssdt_es = _read_pssdt_set("pssdt_varCost_eff_unit_source")
    pssdt_ek = _read_pssdt_set("pssdt_varCost_eff_unit_sink")
    pssdt_ec = _read_pssdt_set("pssdt_varCost_eff_connection")

    # pdtProcess__source__sink__dt_varCost.csv: long-format (p,source,sink,d,t,value)
    p_pssdt_var = None
    var_path = sd / "pdtProcess__source__sink__dt_varCost.csv"
    if var_path.exists():
        df = pl.read_csv(var_path)
        if df.height > 0:
            df = df.rename({"process": "p", "period": "d", "time": "t"}) \
                   .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                          .fill_null(0.0)) \
                   .filter(pl.col("value") != 0)
            if df.height > 0:
                p_pssdt_var = Param(("p","source","sink","d","t"),
                                    df.select("p","source","sink","d","t","value"))

    # pdtProcess_source[p,source,'other_operational_cost',d,t] — wide param file
    def _slice_pds(name: str, side_col: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists():
            return None
        df = pl.read_csv(f)
        if df.height == 0:
            return None
        sliced = df.filter(pl.col("param") == "other_operational_cost") \
                   .drop("param")
        if sliced.height == 0:
            return None
        sliced = (sliced.rename({"process": "p", "period": "d", "time": "t"})
                          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                              .fill_null(0.0))
                          .filter(pl.col("value") != 0))
        if sliced.height == 0:
            return None
        return Param(("p", side_col, "d", "t"),
                     sliced.select("p", side_col, "d", "t", "value"))

    p_var_src  = _slice_pds("pdtProcess_source", "source")
    p_var_sink = _slice_pds("pdtProcess_sink",   "sink")

    # pdtProcess[p,'other_operational_cost',d,t] — process-level (no source/sink dim)
    p_var_proc = None
    pp_path = sd / "pdtProcess.csv"
    if pp_path.exists():
        df = pl.read_csv(pp_path)
        if df.height > 0:
            sliced = df.filter(pl.col("param") == "other_operational_cost") \
                       .drop("param")
            if sliced.height > 0:
                sliced = (sliced.rename({"process": "p", "period": "d", "time": "t"})
                                  .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                                      .fill_null(0.0))
                                  .filter(pl.col("value") != 0))
                if sliced.height > 0:
                    p_var_proc = Param(("p", "d", "t"),
                                       sliced.select("p", "d", "t", "value"))

    return dict(
        pssdt_varCost_noEff=pssdt_noEff,
        pssdt_varCost_eff_unit_source=pssdt_es,
        pssdt_varCost_eff_unit_sink=pssdt_ek,
        pssdt_varCost_eff_connection=pssdt_ec,
        p_pssdt_varCost=p_pssdt_var,
        p_pdt_varCost_source=p_var_src,
        p_pdt_varCost_sink=p_var_sink,
        p_pdt_varCost_process=p_var_proc,
    )


def _load_fixed_cost(sd: Path) -> dict:
    """Load (e, d) ed_fixed_cost and (e, d) p_entity_all_existing — both
    needed for the constant existing-entity-fixed-cost objective term
    (mod §8.1)."""
    blank = dict(p_ed_fixed_cost=None, p_entity_all_existing=None)
    fc_path = sd / "ed_fixed_cost.csv"
    fc = None
    if fc_path.exists():
        df = _read_wide_e_d(fc_path).filter(pl.col("value") != 0)
        if df.height > 0:
            fc = Param(("e", "d"), df.select("e", "d", "value"))
    # p_entity_all_existing per (e, d) — sum of existing+previously_invested
    ae_path = sd / "p_entity_all_existing.csv"
    pe = None
    if ae_path.exists():
        df = pl.read_csv(ae_path)
        if df.height > 0:
            df = df.rename({"entity": "e", "period": "d"}) \
                   .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
            pe = Param(("e", "d"), df.select("e", "d", "value"))
    if fc is None and pe is None:
        return blank
    return dict(p_ed_fixed_cost=fc, p_entity_all_existing=pe)


def _load_node_capacity_for_scaling(sd: Path,
                                     nb: pl.DataFrame) -> dict:
    """Load node_capacity_for_scaling[n, d] for slack-penalty scaling."""
    blank = dict(p_node_capacity_for_scaling=None)
    f = sd / "node_capacity_for_scaling.csv"
    if not f.exists():
        return blank
    df = pl.read_csv(f)
    if df.height == 0:
        return blank
    df = df.rename({"node": "n", "period": "d"}) \
           .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
    # Restrict to nodes in nodeBalance to avoid spurious rows
    if nb is not None and nb.height > 0:
        df = df.join(nb, on="n", how="inner")
    if df.height == 0:
        return blank
    return dict(p_node_capacity_for_scaling=Param(("n", "d"), df.select("n", "d", "value")))


def _load_cumulative_invest(inp: Path, sd: Path, dt: pl.DataFrame) -> dict:
    """Load the new ``FlexData`` fields consumed by ``_cumulative_invest``.

    All fields are independently optional — missing CSV / empty file ⇒ None.
    Sets are filtered to keep only non-empty rows; per-period parameters
    drop all-zero rows so ``has_feature(d)`` won't fire on placeholder
    fixtures whose CSVs exist with all-zero placeholders.
    """
    out: dict = {}

    def _read_set(name: str, src_to_dst: dict[str, str]) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        rename = {s: d for s, d in src_to_dst.items() if s in df.columns}
        out_df = df.rename(rename).select(*src_to_dst.values()).unique()
        return out_df if out_df.height > 0 else None

    def _read_set_drop_zeros(name: str, key_renames: dict[str, str]) -> pl.DataFrame | None:
        """Read a (set, value) CSV and drop value==0 rows; return only the keys."""
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        if "value" in df.columns:
            df = df.with_columns(
                pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0)
            ).filter(pl.col("value") != 0.0)
        rename = {s: d for s, d in key_renames.items() if s in df.columns}
        out_df = df.rename(rename).select(*key_renames.values()).unique()
        return out_df if out_df.height > 0 else None

    def _read_e_d_param(name: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        df = (df.rename({"entity": "e", "period": "d"})
                .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                .filter(pl.col("value") != 0.0)
                .select("e", "d", "value"))
        if df.height == 0: return None
        return Param(("e", "d"), df)

    def _read_e_param(name: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        df = (df.rename({"entity": "e"})
                .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                .filter(pl.col("value") != 0.0)
                .select("e", "value"))
        if df.height == 0: return None
        return Param(("e",), df)

    def _slice_pdgroup(param_name: str) -> pl.DataFrame | None:
        """Long pdGroup.csv slice → (g, d, value), zero rows dropped."""
        f = sd / "pdGroup.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        sliced = (df.filter(pl.col("param") == param_name)
                    .rename({"group": "g", "period": "d"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
                    .select("g", "d", "value"))
        return sliced if sliced.height > 0 else None

    def _slice_pgroup(param_name: str) -> pl.DataFrame | None:
        """input/p_group.csv slice → (g, value), zero rows dropped."""
        f = inp / "p_group.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        # Columns: group, groupParam, p_group
        if "groupParam" not in df.columns:
            return None
        val_col = [c for c in df.columns if c not in ("group", "groupParam")][0]
        sliced = (df.filter(pl.col("groupParam") == param_name)
                    .rename({"group": "g", val_col: "value"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
                    .select("g", "value"))
        return sliced if sliced.height > 0 else None

    def _slice_pdtgroup(param_name: str) -> pl.DataFrame | None:
        """solve_data/pdtGroup.csv slice → (g, d, t, value), zero dropped."""
        f = sd / "pdtGroup.csv"
        if not f.exists(): return None
        df = pl.read_csv(f)
        if df.height == 0: return None
        sliced = (df.filter(pl.col("param") == param_name)
                    .rename({"group": "g", "period": "d", "time": "t"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
                    .select("g", "d", "t", "value"))
        return sliced if sliced.height > 0 else None

    # ── Sets (key-only frames) ────────────────────────────────────────────
    out["ed_invest_forbidden_no_investment"] = _read_set(
        "ed_invest_forbidden_no_investment",
        {"entity": "e", "period": "d"})
    out["ed_invest_cumulative"] = _read_set(
        "ed_invest_cumulative", {"entity": "e", "period": "d"})

    # group_entity: prefer solve_data, fallback input/group__entity.csv
    ge = None
    for cand, mapping in [
        (sd / "group_entity.csv",   {"group": "g", "entity": "e"}),
        (inp / "group__entity.csv", {"group": "g", "entity": "e"}),
    ]:
        if cand.exists():
            df = pl.read_csv(cand)
            if df.height > 0:
                ge = df.rename(mapping).select("g", "e").unique()
                break
    out["group_entity"] = ge

    # group_process_node: solve_data/group_process_node.csv (preprocessed long)
    # or input/group__process__node.csv (raw long)
    gpn = None
    for cand, mapping in [
        (sd / "group_process_node.csv",  {"group": "g", "process": "p", "node": "n"}),
        (inp / "group__process__node.csv", {"group": "g", "process": "p", "node": "n"}),
    ]:
        if cand.exists():
            df = pl.read_csv(cand)
            if df.height > 0:
                gpn = df.rename(mapping).select("g", "p", "n").unique()
                break
    out["group_process_node"] = gpn

    out["g_invest_total"]      = _read_set("g_invest_total", {"group": "g"})
    out["g_divest_total"]      = _read_set("g_divest_total", {"group": "g"})
    out["g_invest_cumulative"] = _read_set("g_invest_cumulative", {"group": "g"})
    out["gd_invest_period"]    = _read_set("gd_invest_period",
                                            {"group": "g", "period": "d"})
    out["gd_divest_period"]    = _read_set("gd_divest_period",
                                            {"group": "g", "period": "d"})

    # ── Parameters (e, d) ────────────────────────────────────────────────
    out["ed_invest_min_period"]       = _read_e_d_param("ed_invest_min_period")
    out["ed_divest_min_period"]       = _read_e_d_param("ed_divest_min_period")
    out["ed_cumulative_max_capacity"] = _read_e_d_param("ed_cumulative_max_capacity")
    out["ed_cumulative_min_capacity"] = _read_e_d_param("ed_cumulative_min_capacity")

    # ── Parameters (e,) ──────────────────────────────────────────────────
    out["e_invest_min_total"] = _read_e_param("e_invest_min_total")
    out["e_divest_min_total"] = _read_e_param("e_divest_min_total")

    # ── Group params from p_group / pdGroup / pdtGroup slices ────────────
    def _g_param(slice_name: str) -> Param | None:
        df = _slice_pgroup(slice_name)
        return Param(("g",), df) if df is not None else None
    def _gd_param(slice_name: str) -> Param | None:
        df = _slice_pdgroup(slice_name)
        return Param(("g", "d"), df) if df is not None else None
    def _gdt_param(slice_name: str) -> Param | None:
        df = _slice_pdtgroup(slice_name)
        return Param(("g", "d", "t"), df) if df is not None else None

    out["p_group_invest_max_period"]      = _gd_param("invest_max_period")
    out["p_group_invest_min_period"]      = _gd_param("invest_min_period")
    out["p_group_retire_max_period"]      = _gd_param("retire_max_period")
    out["p_group_retire_min_period"]      = _gd_param("retire_min_period")
    out["p_group_invest_max_total"]       = _g_param("invest_max_total")
    out["p_group_invest_min_total"]       = _g_param("invest_min_total")
    out["p_group_retire_max_total"]       = _g_param("retire_max_total")
    out["p_group_retire_min_total"]       = _g_param("retire_min_total")
    out["p_group_invest_max_cumulative"]  = _g_param("invest_max_cumulative")
    out["p_group_invest_min_cumulative"]  = _g_param("invest_min_cumulative")
    out["p_group_max_cumulative_flow"]    = _g_param("max_cumulative_flow")
    out["p_group_min_cumulative_flow"]    = _g_param("min_cumulative_flow")
    out["pd_max_cumulative_flow"]         = _gd_param("max_cumulative_flow")
    out["pd_min_cumulative_flow"]         = _gd_param("min_cumulative_flow")
    pdt_max = _gdt_param("max_instant_flow")
    pdt_min = _gdt_param("min_instant_flow")
    out["pdt_max_instant_flow"] = pdt_max
    out["pdt_min_instant_flow"] = pdt_min
    # Support of pdt_*_instant_flow (rows where param is non-null/non-zero)
    out["gdt_maxInstantFlow"] = (pdt_max.frame.select("g", "d", "t")
                                  if pdt_max is not None else None)
    out["gdt_minInstantFlow"] = (pdt_min.frame.select("g", "d", "t")
                                  if pdt_min is not None else None)

    return out


# ---------------------------------------------------------------------------
# HiGHS solver options (input/solve_mode.csv)
#
# flextool's ``solve_mode.csv`` is keyed (param, solve, value).  Three
# ``param`` rows feed HiGHS directly:
#   * ``highs_method``   → HiGHS option ``solver``    (str: simplex/ipm/choose)
#   * ``highs_parallel`` → HiGHS option ``parallel``  (str: on/off)
#   * ``highs_presolve`` → HiGHS option ``presolve``  (str: on/off/choose)
# Other ``param`` rows (notably ``solve_mode``) describe the flextool
# solve framework, not HiGHS, and are ignored here.
#
# When the file lists multiple solves, we pick the row whose ``solve``
# matches ``solve_data/solve_current.csv``.  If ``solve_current`` is
# absent or the row is missing for that solve, we silently fall back to
# HiGHS defaults (``solver_options=None``) — current behavior.

# flextool param → HiGHS canonical option name + coercion
_HIGHS_PARAM_MAP: dict[str, tuple[str, type]] = {
    "highs_method":   ("solver",   str),
    "highs_parallel": ("parallel", str),
    "highs_presolve": ("presolve", str),
    # Numeric / boolean HiGHS options that flextool *may* emit in the
    # future — wire them up so they Just Work when they appear.
    "highs_time_limit":                 ("time_limit",                 float),
    "highs_mip_rel_gap":                ("mip_rel_gap",                float),
    "highs_mip_abs_gap":                ("mip_abs_gap",                float),
    "highs_threads":                    ("threads",                    int),
    "highs_random_seed":                ("random_seed",                int),
    "highs_output_flag":                ("output_flag",                bool),
    "highs_primal_feasibility_tolerance":
        ("primal_feasibility_tolerance", float),
    "highs_dual_feasibility_tolerance":
        ("dual_feasibility_tolerance",   float),
}


def _coerce_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "yes", "on", "1"): return True
    if s in ("false", "no", "off", "0"): return False
    raise ValueError(f"cannot coerce {v!r} to bool")


def _load_solver_options(sd: Path) -> dict | None:
    p = sd.parent / "input" / "solve_mode.csv"
    if not p.exists():
        # Fixtures sometimes drop the CSV under solve_data/ instead of input/.
        p = sd / "solve_mode.csv"
        if not p.exists():
            return None
    df = pl.read_csv(p)
    if df.height == 0 or "param" not in df.columns or "value" not in df.columns:
        return None

    # Pick the active solve.  ``solve_current.csv`` has a single ``solve``
    # column with one row.  If multiple HiGHS rows exist for a single
    # ``param`` across solves and we can't disambiguate, prefer the
    # solve_current match; otherwise (single-solve fixtures) take whatever's
    # there.
    cur_path = sd / "solve_current.csv"
    cur_solve: str | None = None
    if cur_path.exists():
        cur_df = pl.read_csv(cur_path)
        if cur_df.height > 0 and "solve" in cur_df.columns:
            cur_solve = str(cur_df["solve"][0])

    if cur_solve is not None and "solve" in df.columns:
        df_active = df.filter(pl.col("solve") == cur_solve)
        if df_active.height == 0:
            df_active = df  # fall back: no rows for current solve
    else:
        df_active = df

    out: dict = {}
    for row in df_active.iter_rows(named=True):
        param = str(row["param"]).strip()
        if param not in _HIGHS_PARAM_MAP:
            continue
        opt_name, opt_type = _HIGHS_PARAM_MAP[param]
        raw = row["value"]
        try:
            if opt_type is bool:
                val = _coerce_bool(raw)
            elif opt_type is int:
                val = int(float(raw))   # tolerate "1.0"
            elif opt_type is float:
                val = float(raw)
            else:
                val = str(raw).strip()
        except (TypeError, ValueError):
            # Don't crash on a malformed cell — let HiGHS defaults stand.
            continue
        out[opt_name] = val
    return out or None


def _load_stochastics(inp: Path, sd: Path, dt: pl.DataFrame) -> dict:
    """Load multi-branch stochastic operational data.

    Mirrors flextool.mod's stochastic feature (mod:38-41, :562-588,
    :873-895, :988, :1978-2142, :4173-4233).  Reads four CSVs:

    * ``solve_data/pdt_branch_weight.csv`` (period, time, value) →
      :class:`Param` keyed (d, t).  Per-branch operational probability
      that multiplies every dispatch-class objective term.  Defaults to
      1.0 per (d, t) when CSV is empty / missing.
    * ``solve_data/pd_branch_weight.csv`` (period, value) →
      :class:`Param` keyed (d,).  Per-branch period-level probability
      for investment-fixed-cost terms.  Defaults to 1.0.
    * ``solve_data/dt_non_anticipativity_set.csv`` (period, time) →
      ``pl.DataFrame``.  Realised-dispatch + fix-storage timesteps where
      the four ``non_anticipativity_*`` constraints fire.  Empty when
      stochastics inactive.
    * ``input/groupIncludeStochastics.csv`` (group,) → ``pl.DataFrame``.
      Groups whose ``group_node`` membership unlocks the storage
      non-anticipativity coupling (``non_anticipativity_storage_use``).

    Also loads the unfiltered ``period__branch.csv`` (anchor → sibling)
    distinct from the existing ``period_branch`` rolling-handoff field
    (which renames columns to ``d_upper``/``d``).  And the active
    ``period_in_use_set`` from ``solve_data/period_in_use_set.csv`` —
    used by the model layer to filter branch periods that exist in the
    metadata-only ``period_branch`` map but aren't part of the actual
    LP (e.g. ``period1_realized`` in the 2_day_stochastic_dispatch
    fixture).
    """
    # pdt_branch_weight: (d, t) → value, defaults 1.0
    pdt_bw_path = sd / "pdt_branch_weight.csv"
    if pdt_bw_path.exists():
        df = pl.read_csv(pdt_bw_path)
        if df.height > 0:
            df = (df.rename({"period": "d", "time": "t"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "t", "value"))
            # Build a (d,t)-dense Param: dt × value with value defaulting
            # to 1.0 where pdt_branch_weight is silent.  Mirrors .mod's
            # ``param pdt_branch_weight {(d,t) in dt}`` declaration —
            # dense over dt.
            base = dt.with_columns(value=pl.lit(1.0)).select("d", "t", "value")
            base = (base
                    .join(df, on=["d", "t"], how="left", suffix="__r")
                    .with_columns(value=pl.coalesce(
                        pl.col("value__r"), pl.col("value")))
                    .select("d", "t", "value"))
            pdt_branch_weight = Param(("d", "t"), base)
        else:
            pdt_branch_weight = None
    else:
        pdt_branch_weight = None

    # pd_branch_weight: (d,) → value
    pd_bw_path = sd / "pd_branch_weight.csv"
    pd_branch_weight = None
    if pd_bw_path.exists():
        df = pl.read_csv(pd_bw_path)
        if df.height > 0:
            df = (df.rename({"period": "d"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "value"))
            pd_branch_weight = Param(("d",), df)

    # dt_non_anticipativity_set: (d, t)
    dt_na_path = sd / "dt_non_anticipativity_set.csv"
    dt_non_anticipativity = None
    if dt_na_path.exists():
        df = pl.read_csv(dt_na_path)
        if df.height > 0:
            dt_non_anticipativity = (df
                .rename({"period": "d", "time": "t"})
                .select("d", "t").unique())

    # groupIncludeStochastics: (g,)
    gis_path = inp / "groupIncludeStochastics.csv"
    groupStochastic = None
    if gis_path.exists():
        df = pl.read_csv(gis_path)
        if df.height > 0:
            # CSV column is named ``group``; rename to canonical ``g``.
            df = df.rename({df.columns[0]: "g"})
            groupStochastic = df.select("g").unique()

    # period__branch.csv (anchor d → sibling b, dimen 2).  This is the
    # FULL map.  flexpy already loads a *filtered* renamed version into
    # ``period_branch`` for rolling-handoff.  We load the unfiltered raw
    # form here for the non-anticipativity constraints.
    period_branch_full = None
    pb_path = sd / "period__branch.csv"
    if pb_path.exists():
        df = pl.read_csv(pb_path)
        if df.height > 0:
            period_branch_full = (df
                .rename({"period": "d", "branch": "b"})
                .select("d", "b").unique())

    # period_in_use_set: (d,) — periods active in the active solve.
    period_in_use_set = None
    piu_path = sd / "period_in_use_set.csv"
    if piu_path.exists():
        df = pl.read_csv(piu_path)
        if df.height > 0:
            df = df.rename({df.columns[0]: "d"})
            period_in_use_set = df.select("d").unique()

    return dict(
        pdt_branch_weight=pdt_branch_weight,
        pd_branch_weight=pd_branch_weight,
        dt_non_anticipativity=dt_non_anticipativity,
        groupStochastic=groupStochastic,
        period_branch_full=period_branch_full,
        period_in_use_set=period_in_use_set,
    )


def _assign_param_names(data: "FlexData") -> "FlexData":
    """Stamp the FlexData attribute name onto every :class:`Param` field.

    Enables :class:`polar_high_opt.WarmProblem`'s Param-tracked auto-update by
    giving each Param a stable logical name (``"p_inflow"`` etc.) that
    flows through the algebra primitives' source-Param metadata.
    Anonymous (``name is None``) Params are not tracked.
    """
    from dataclasses import fields as _dc_fields
    for f in _dc_fields(data):
        v = getattr(data, f.name, None)
        if isinstance(v, Param) and v.name is None:
            v.name = f.name
    return data


def load_flextool(source: "Path | str | FlexInputSource",
                   *,
                   db_reader: "object | None" = None) -> FlexData:
    """Load a :class:`FlexData` from either a workdir on disk or a
    :class:`flextool._input_source.FlexInputSource`.

    Backward-compatible: passing a ``Path`` (today's call style) wraps
    it as a :class:`flextool._input_source.CsvSource` internally and
    behaves identically.  Passing a
    :class:`flextool._spinedb_source.SpineDbSource` triggers
    DB-driven materialisation on first directory access — the rest of
    this loader walks the resulting CSVs unchanged.

    Γ.1 of the deeper DB-direct migration adds an optional ``db_reader``
    keyword: when supplied, the per-(entity_class, parameter_name)
    :class:`flextool._input_source.InputSource` (typically
    :class:`SpineDbReader`) overrides the chosen first-wave Direct
    Params with frames built directly from the DB.  Every other
    ``FlexData`` field is still loaded via the CSV path; the full
    sweep into ``input.py`` happens in Γ.2/Γ.3.

    See ``audit/db_direct_param_map.md §7.1`` for the migration plan.
    """
    # Late-import the Protocol + adapters to avoid a circular import
    # against the tests' fixture-loaders (which sometimes import this
    # module before flextool.__init__ finishes).
    from flextool.engine_polars._input_source import CsvSource, FlexInputSource, InputSource

    if isinstance(source, (str, Path)):
        source = CsvSource(source)
    elif not isinstance(source, FlexInputSource):
        raise TypeError(
            f"load_flextool expects Path | str | FlexInputSource, "
            f"got {type(source).__name__}"
        )
    if db_reader is not None and not isinstance(db_reader, InputSource):
        raise TypeError(
            f"load_flextool db_reader must implement InputSource, "
            f"got {type(db_reader).__name__}"
        )

    inp = source.input_dir
    sd  = source.solve_data_dir

    dt, step_dur, rp_cw, infl, psh = _load_time(sd)
    nb, nb_dt, inflow, pen_up, pen_dn = _load_node(sd, dt)

    proc = _load_process_topology(inp, sd, dt)

    # base_cap_pd = (p, d, base) for profile RHS — recompute here; small.
    base_cap_pd = None
    p_flow_upper_existing = None
    pd_neg_cap = None
    if proc["pss"] is not None:
        cap_long = _read_capacity(sd / "p_entity_period_existing_capacity.csv",
                                   sd / "p_entity_previously_invested_capacity.csv",
                                   sd / "p_entity_all_existing.csv")
        unitsize_long = _read_unitsize((sd / "p_entity_unitsize.csv") if (sd / "p_entity_unitsize.csv").exists() else (inp / "p_entity_unitsize.csv"))
        cap_us_pd = (cap_long.rename({"e":"p","value":"cap"})
            .filter(pl.col("p").is_in(proc["pss"]["p"].unique()))
            .join(unitsize_long.rename({"e":"p","value":"us"}), on="p", how="inner"))
        base_cap_pd = (cap_us_pd
            .with_columns(base=pl.col("cap")/pl.col("us"))
            .select("p","d","base"))
        # pd_neg_cap = (p, d) where both existing and unitsize are negative.
        # In the .mod, maxToSink is ``v_flow * unitsize ≤ existing × ...``.
        # When both are negative (e.g. anti_energy_plant: us=-50, existing=-50)
        # dividing by unitsize FLIPS the inequality direction, yielding
        # ``v_flow ≥ existing/unitsize`` (a forced *minimum* output).
        # We therefore route these (p, d) rows out of the standard ``≤``
        # maxToSink and into a sign-flipped ``≥`` companion constraint.
        neg_pd = cap_us_pd.filter(
            (pl.col("cap") < 0.0) & (pl.col("us") < 0.0)
        ).select("p", "d")
        if neg_pd.height > 0:
            pd_neg_cap = neg_pd
        # p_flow_upper_existing = (existing/unitsize) per (p, source, sink, d).
        # This is the *true* structural existing-capacity upper bound on
        # v_flow.  It corresponds to the .mod's RHS without invest/divest
        # (assuming cap_coef=1).  flextool's preprocessed p_flow_max may
        # bake in max_invest_cum (for invest-method = invest_no_limit) and
        # is therefore looser; using p_flow_upper_existing + the explicit
        # invest tightening on the LHS gives the tight constraint.
        p_flow_upper_existing = Param(("p", "source", "sink", "d"),
            base_cap_pd.rename({"base": "value"})
                       .join(proc["pss"], on="p", how="inner")
                       .select("p", "source", "sink", "d", "value"))

    flow_co2_p, flow_co2_p_noEff, co2c, co2pr = _load_co2_price(
        inp, sd, proc["pss_eff"], proc.get("pss_noEff"))
    g_co2_max, flow_co2_cap, flow_co2_cap_noEff, co2_max_p, g_d_capped = _load_co2_cap(
        inp, sd, proc["pss_eff"], dt, pss_noEff=proc.get("pss_noEff"))
    if co2_max_p is not None and co2c is None:
        p_comm = pl.read_csv(inp / "p_commodity.csv")
        co2c = Param(("c",),
            p_comm.filter(pl.col("commodityParam")=="co2_content")
                  .rename({"commodity":"c","p_commodity":"value"})
                  .select("c","value"))

    (indir_set, indir_in, indir_out, indir_dt,
     p_source_flow_coef, p_sink_flow_coef) = _load_indirect(sd, proc["pss"], dt, inp)
    (fc_idx, fc_coef, c_const, cdt_eq, cdt_le, cdt_ge,
     n_inv_coef, p_inv_coef,
     n_state_coef, n_prebuilt_coef, p_prebuilt_coef,
     _) = _load_user_constraints(inp, proc["pss"], dt)

    p_up, p_lo, p_fx, prof_v, exist_cnt, avail = _load_profiles(
        inp, sd, proc["pss"], proc["unitsize"], base_cap_pd)
    # existing_count is also needed by the online/UC feature even when
    # no profile features are active; fall back to base_cap_pd directly.
    if exist_cnt is None and base_cap_pd is not None:
        exist_cnt = Param(("p", "d"), base_cap_pd.rename({"base": "value"}))
    # availability: default to 1.0 from preprocessing — also used by UC
    # capacity bounds; if loader didn't populate (no profile data), try
    # to read pdtProcess_availability.csv standalone.
    if avail is None and proc["pss"] is not None:
        avail_long = _slice_param(sd / "pdtProcess.csv", "process", "availability",
                                   rename_entity_to="p")
        if avail_long is not None:
            avail = Param(("p","d","t"), avail_long)

    # dtttdt is needed by both storage and online features — always load
    # it when present (preprocessing always emits it for non-trivial
    # solves).  p_process_existing_count (= existing/unitsize per (p, d))
    # is needed by online + profile features — always load when processes
    # exist.
    dtttdt = _read_step_previous(sd / "step_previous.csv")

    online = _load_online(inp, sd, dt, proc["pss"])
    ramp = _load_ramp(inp, sd, proc["pss"])
    invest = _load_invest(sd, dt, inp, proc["pss"])
    varcost = _load_varcost(sd, proc["pss"])
    fixed_cost = _load_fixed_cost(sd)
    capacity_for_scaling = _load_node_capacity_for_scaling(sd, nb)

    # ─── Storage (nodeState + binding methods + dtttdt + node-balance source-side flows)
    storage = _load_storage(inp, sd, dt, nb,
                             proc["pss_eff"], proc["pss_noEff"],
                             base_cap_pd, proc["unitsize"])
    # _load_storage emits its own dtttdt; if storage is inactive it'll be
    # None there but we want the top-level read.
    if storage["dtttdt"] is None:
        storage["dtttdt"] = dtttdt

    # ─── Per-arc block step durations (reserved for future use) ──────────
    p_arc_step_duration_sink = None
    p_arc_step_duration_source = None

    # ─── Per-arc-side daily-block aggregation index ──────────────────────
    # For each (n, d, b_first) in nodeStateBlock × period_block, build the
    # set of (p, source, sink, t, weight) that contribute to that daily
    # nodeBalance via the .mod's overlap × block_step_duration aggregation.
    # The arc's relevant side block (process_side_block.csv) determines
    # the timesteps & weights:
    #   * b_f=daily_group → t=b_first only, weight=block_step_duration
    #     (=24 for daily_group's coarse step).
    #   * b_f=hourly_group/default → all fine t in period_block_time[d,
    #     b_first], weight=1.
    # This matches the .mod's nodeBalance_eq treatment of mixed-block arcs
    # (e.g., electrolyser_A's sink-side daily, source-side hourly) where
    # the daily-side balance only references v_flow at coarse steps × 24,
    # while the hourly-side balance integrates all 24 hourly v_flow values.
    # See flextool.mod:2208-2246.
    arc_sink_block_dt = None
    arc_source_block_dt = None
    p_arc_sink_weight = None
    p_arc_source_weight = None
    psb_path = sd / "process_side_block.csv"
    bsd_for_arc_path = sd / "block_step_duration.csv"
    if (proc["pss"] is not None and proc["pss"].height > 0
            and storage.get("nodeStateBlock") is not None
            and storage["nodeStateBlock"].height > 0
            and storage.get("period_block_time") is not None
            and storage["period_block_time"].height > 0
            and psb_path.exists() and bsd_for_arc_path.exists()):
        psb = pl.read_csv(psb_path).rename(
            {"process": "p", "block": "b_f"})
        bsd_arc = pl.read_csv(bsd_for_arc_path).rename(
            {"block": "b_f", "period": "d", "step": "t",
             "step_duration": "weight"})
        nsb_set = storage["nodeStateBlock"]["n"].unique()
        pss_local = proc["pss"]
        pbt = storage["period_block_time"]   # (d, b_first, t)

        # Sink-side: arcs where sink ∈ nodeStateBlock.
        # process_side_block (p, side='sink', b_f) → arc-side block
        psb_sink = psb.filter(pl.col("side") == "sink").select("p", "b_f")
        sink_arcs = (pss_local
            .filter(pl.col("sink").is_in(nsb_set))
            .join(psb_sink, on="p", how="inner"))
        if sink_arcs.height > 0:
            # For each arc-side, restrict (d, t) to those where
            # block_step_duration[b_f, d, t] is defined (i.e., t is a
            # coarse step of b_f).  Then join to period_block_time to
            # group by (d, b_first).
            arc_sink_block_dt = (sink_arcs
                .join(bsd_arc, on="b_f", how="inner")  # adds (d, t, weight)
                .join(pbt.rename({}), on=["d", "t"], how="inner")
                .select("p", "source", "sink", "d", "b_first", "t", "weight")
                .unique())
            if arc_sink_block_dt.height == 0:
                arc_sink_block_dt = None
            else:
                # Keyed (p, source, sink, d, t) so it joins naturally with
                # v_flow on (p, source, sink, d, t).  The sink→n rename
                # happens in nbb_eq's index frame instead.
                weight_frame = (arc_sink_block_dt
                    .select("p", "source", "sink", "d", "t", "weight")
                    .unique()
                    .rename({"weight": "value"}))
                p_arc_sink_weight = Param(
                    ("p", "source", "sink", "d", "t"), weight_frame)
        # Source-side: arcs where source ∈ nodeStateBlock.
        psb_src = psb.filter(pl.col("side") == "source").select("p", "b_f")
        src_arcs = (pss_local
            .filter(pl.col("source").is_in(nsb_set))
            .join(psb_src, on="p", how="inner"))
        if src_arcs.height > 0:
            arc_source_block_dt = (src_arcs
                .join(bsd_arc, on="b_f", how="inner")
                .join(pbt.rename({}), on=["d", "t"], how="inner")
                .select("p", "source", "sink", "d", "b_first", "t", "weight")
                .unique())
            if arc_source_block_dt.height == 0:
                arc_source_block_dt = None
            else:
                # Keep ``source`` column unchanged so this Param joins with
                # v_flow on (p, source, sink, d, t).  The arc-source-side
                # rename happens in nbb_eq's index frame instead.
                weight_frame_src = (arc_source_block_dt
                    .select("p", "source", "sink", "d", "t", "weight")
                    .unique()
                    .rename({"weight": "value"}))
                p_arc_source_weight = Param(
                    ("p", "source", "sink", "d", "t"), weight_frame_src)

    # ─── Group-level slack (capacity_margin / inertia / non_sync) ────────
    group_slack = _group_slack.load_data(
        inp=inp, sd=sd, dt=dt,
        nb=nb,
        pss_eff=proc["pss_eff"],
        pss_noEff=proc["pss_noEff"],
        p_unitsize=proc["unitsize"],
    )

    # ─── Reserves (timeseries / dynamic / n_1 / per-process upper) ────────
    reserve_data = _reserve.load_data(inp=inp, sd=sd, dt=dt)
    # ``group_node`` is shared between _group_slack and _reserve (both
    # populate it from the canonical solve_data/group_node.csv).  Drop the
    # reserve copy to avoid duplicate-kwargs at the FlexData(...) call when
    # group_slack already provided it; reserve will read it back off d in
    # add_constraints.  If group_slack didn't populate it, hand the
    # reserve copy through.
    if "group_node" in reserve_data and group_slack.get("group_node") is not None:
        reserve_data = {k: v for k, v in reserve_data.items() if k != "group_node"}

    # ─── Cumulative / group-invest / min-invest data ─────────────────────
    # The module's ``load_data`` is a no-op stub; ``flextool/input.py`` is
    # the canonical loader.  Call it for symmetry, then populate the new
    # ``FlexData`` fields from the canonical helper below.
    _cumulative_invest.load_data(inp=inp, sd=sd, dt=dt)
    ci_data = _load_cumulative_invest(inp=inp, sd=sd, dt=dt)

    # ─── Delayed processes / DR data ─────────────────────────────────────
    delay_data = _delay.load_data(inp_dir=inp, sd_dir=sd)

    # ─── DC power flow data ──────────────────────────────────────────────
    dc_pf_data = _dc_power_flow.load_data(inp_dir=inp)

    # ─── Commodity price ladder data ─────────────────────────────────────
    ladder_data = _commodity_ladder.load_data(inp_dir=inp, sd_dir=sd)

    # ─── Multi-branch stochastic data (A6) ───────────────────────────────
    stoch_data = _load_stochastics(inp=inp, sd=sd, dt=dt)

    flex_data = FlexData(
        dt = dt,
        p_step_duration = step_dur,
        p_rp_cost_weight = rp_cw,
        p_inflation_op = infl,
        p_period_share = psh,

        nodeBalance = nb,
        nodeBalance_dt = nb_dt,
        p_inflow = inflow,
        p_penalty_up = pen_up,
        p_penalty_down = pen_dn,

        process_source_sink       = proc["pss"],
        process_source_sink_eff   = proc["pss_eff"],
        process_source_sink_noEff = proc["pss_noEff"],
        pss_dt                    = proc["pss_dt"],
        flow_to_n                 = proc["flow_to_n"],
        flow_from_n               = proc["flow_from_n"],
        flow_from_commodity_eff   = proc["flow_from_commodity_eff"],
        flow_from_commodity_noEff = proc["flow_from_commodity_noEff"],
        flow_to_commodity         = proc.get("flow_to_commodity"),
        p_unitsize                = proc["unitsize"],
        p_flow_upper              = proc["flow_upper"],
        p_flow_upper_existing     = p_flow_upper_existing,
        p_slope                   = proc["slope"],
        p_commodity_price         = proc["commodity_price"],
        pd_neg_cap                = pd_neg_cap,

        flow_from_co2_priced = flow_co2_p,
        flow_from_co2_priced_noEff = flow_co2_p_noEff,
        p_co2_content = co2c,
        p_co2_price = co2pr,

        group_co2_max_period = g_co2_max,
        flow_from_co2_capped = flow_co2_cap,
        flow_from_co2_capped_noEff = flow_co2_cap_noEff,
        p_co2_max_period = co2_max_p,
        group_d_co2_capped = g_d_capped,

        process_indirect = indir_set,
        process_input_flows = indir_in,
        process_output_flows = indir_out,
        process_indirect_dt = indir_dt,
        p_process_source_flow_coef = p_source_flow_coef,
        p_process_sink_flow_coef = p_sink_flow_coef,

        flow_constraint_idx = fc_idx,
        p_flow_constraint_coef = fc_coef,
        p_constraint_constant = c_const,
        cdt_eq = cdt_eq,
        cdt_le = cdt_le,
        cdt_ge = cdt_ge,
        p_node_constraint_invested_capacity_coefficient = n_inv_coef,
        p_process_constraint_invested_capacity_coefficient = p_inv_coef,
        p_node_constraint_state_coefficient = n_state_coef,
        p_node_constraint_prebuilt_capacity_coefficient = n_prebuilt_coef,
        p_process_constraint_prebuilt_capacity_coefficient = p_prebuilt_coef,

        process_profile_upper = p_up,
        process_profile_lower = p_lo,
        process_profile_fixed = p_fx,
        p_profile_value = prof_v,
        p_process_existing_count = exist_cnt,
        p_process_availability = avail,

        **online,
        **ramp,
        **invest,
        **storage,
        **varcost,
        **fixed_cost,
        **capacity_for_scaling,
        **group_slack,
        **reserve_data,
        **ci_data,
        **delay_data,
        **dc_pf_data,
        **ladder_data,
        **stoch_data,
        p_arc_step_duration_sink = p_arc_step_duration_sink,
        p_arc_step_duration_source = p_arc_step_duration_source,
        arc_sink_block_dt = arc_sink_block_dt,
        arc_source_block_dt = arc_source_block_dt,
        p_arc_sink_weight = p_arc_sink_weight,
        p_arc_source_weight = p_arc_source_weight,
        solver_options = _load_solver_options(sd),
    )

    # Γ.1 — DB-direct override for the first-wave Direct Params.  The
    # CSV path above produced every FlexData field; here we replace the
    # subset that have a Direct equivalent in the source plugin so that
    # downstream behaviour is unchanged but the data flows through the
    # DB-direct port.  The full sweep happens in Γ.2 / Γ.3.
    if db_reader is not None:
        from flextool.engine_polars import _direct_params as _dp
        overrides = _dp.first_wave_overrides(db_reader, flex_data)
        for field, value in overrides.items():
            setattr(flex_data, field, value)
        # Γ.2 — Projection Params over the same source.  Same overlay
        # pattern: empty DB-side overlays don't blank out CSV data; only
        # populated DB-side frames replace the CSV equivalent.
        from flextool.engine_polars import _projection_params as _pp
        proj = _pp.projection_overrides(db_reader, flex_data)
        for field, value in proj.items():
            setattr(flex_data, field, value)
        # Γ.3.A — foundational Derived overrides (time / weighting,
        # inflow, profiles, stochastic).  Applied after Direct +
        # Projection so dependent helpers can read the (already-DB-
        # overridden) FlexData.  Each overlay is gated on a frame-equal
        # pre-check against the CSV-loaded value so we never overlay
        # rows the simple algorithm can't reproduce (multi-block
        # timelines, multi-year inflation, stochastic branches —
        # those land in Batches B/C/D).
        try:
            workdir = source.workdir if hasattr(source, "workdir") \
                       else source.input_dir.parent
        except Exception:  # pragma: no cover — defensive
            workdir = None
        if workdir is not None:
            from flextool.engine_polars import _derived_params as _drA
            der = _drA.derived_overrides_a(flex_data, db_reader,
                                            Path(workdir))
            for field, value in der.items():
                setattr(flex_data, field, value)
            # Γ.3.B — process topology + reclassified method-derived
            # overlays (§3.3 / §3.5 / §3.10).  Same overlay pattern: each
            # field is gated on a frame-equal precheck against the CSV
            # value so the simple algorithm never corrupts a multi-block
            # / lifetime-cumulative / multi-method fixture.
            der_b = _drA.derived_overrides_b(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_b.items():
                setattr(flex_data, field, value)
            # Γ.3.C — invest/divest + online/UC + group slack + multi-year
            # inflation cascade overlays (§3.7 / §3.8 / §3.11 / §3.12).
            # Each field is gated on a frame-equal precheck against the
            # CSV value so multi-year invest cascades / scaling-active
            # fixtures keep the CSV-loaded value when the simple path
            # can't reproduce them.
            der_c = _drA.derived_overrides_c(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_c.items():
                setattr(flex_data, field, value)
            # Γ.3.D — final batch of conservative narrow-scope overlays:
            # §3.11 ``p_entity_all_existing``, §3.16 ``node_reference_angle``,
            # §3.13 ``process_reserve_upDown_node_active``.  Storage block
            # algebra, lifetime cascade, ladder, delay and multi-branch
            # Params are deferred to Γ.3.E (see progress.md).  Same gate-
            # on-equality discipline as earlier batches.
            der_d = _drA.derived_overrides_d(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_d.items():
                setattr(flex_data, field, value)
            # Γ.3.E — storage block algebra (§3.9): dtttdt + period_block
            # family + nodeStateBlock multi-resolution synthesis +
            # arc-block weights + state caps + reference-value exclusion +
            # rolling-handoff handoff carriers.  No defensive gating per
            # the §Γ.3.E architectural shift — helpers either produce
            # the canonical frame or the parity test fails loudly.
            der_e = _drA.derived_overrides_e(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_e.items():
                setattr(flex_data, field, value)
            # Γ.3.F — lifetime cascade + handoff state + full multi-year
            # inflation cascade (§3.1.3 / §3.7.5/6 / §3.7.7/8).  Helpers
            # depend on the (already-overridden) ed_invest_set /
            # ed_divest_set frames from Γ.3.C, hence run last.  Same
            # no-defensive-gating discipline as Γ.3.E.
            der_f = _drA.derived_overrides_f(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_f.items():
                setattr(flex_data, field, value)
            # Γ.3.G — residual Derived Params (commodity ladder §3.17,
            # reserves §3.13 prundt, delay §3.15, full multi-branch
            # normalisation §3.18).  No defensive gating per the §Γ.3.E
            # architectural shift; helpers either produce the canonical
            # frame or the parity test fails loudly.
            der_g = _drA.derived_overrides_g(flex_data, db_reader,
                                               Path(workdir))
            for field, value in der_g.items():
                setattr(flex_data, field, value)

    return _assign_param_names(flex_data)


def _read_period_set(path: Path) -> set[str]:
    """Read a single-column period CSV (header row, then one period per row)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as fh:
        reader = __import__("csv").reader(fh)
        next(reader, None)
        for r in reader:
            if r and r[0]:
                out.add(r[0])
    return out


def _read_realize_invest_periods(path: Path) -> set[str]:
    """Read ``realized_invest_periods_of_current_solve.csv`` (single
    ``period`` column written by ``solve_writers.write_periods``).

    Empty file or missing → empty set (treat as: nothing realized this solve).
    """
    return _read_period_set(path)


def _read_realized_dispatch_periods(path: Path) -> set[str]:
    """Read distinct periods from ``realized_dispatch.csv`` (cols include ``period``)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    csv = __import__("csv")
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        try:
            i = header.index("period")
        except ValueError:
            return set()
        for r in reader:
            if len(r) > i and r[i]:
                out.add(r[i])
    return out


def _read_solve_first(work_folder: Path) -> bool:
    """Read ``p_model.csv``'s ``solveFirst`` flag.

    flextool's per-solve preprocessing writes ``solve_data/p_model.csv``
    with the chain-position flag (``solveFirst=1`` only for the first
    sub-solve in the multi-solve cascade, ``0`` for the rest).  The
    static ``input/p_model.csv`` does not exist in DB-driven fixtures —
    the file is purely a preprocessing-derived artifact.

    Resolution order:
    1. ``solve_data/p_model.csv`` — preferred (chain-aware).
    2. ``input/p_model.csv`` — legacy fallback when a fixture predates
       the preprocessing rewrite.
    3. Default ``True`` when neither exists.

    Bug-fix anchor: prior to Γ.8.E this only consulted ``input/`` which
    in the native cascade path produced ``solveFirst=True`` for every
    sub-solve, causing ``build_handoff_from_flexpy`` to add
    ``pre_existing`` to ``realized_existing`` on every iteration —
    inflating the chain's cumulative ``p_entity_period_existing_capacity``
    by ``Σ pre_existing`` per extra sub-solve and zeroing out demand on
    sub-solves 3+ of fixtures like ``wind_battery_invest_lifetime_renew_4solve``.
    """
    csv = __import__("csv")
    for cand in ("solve_data/p_model.csv", "input/p_model.csv"):
        path = work_folder / cand
        if not path.exists():
            continue
        with path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader, None) or []
            try:
                param_idx = header.index("modelParam")
                value_idx = header.index("p_model")
            except ValueError:
                return True
            for r in reader:
                if len(r) > max(param_idx, value_idx) and r[param_idx] == "solveFirst":
                    try:
                        return bool(int(r[value_idx]))
                    except (ValueError, TypeError):
                        return True
        # File existed but didn't contain the flag — treat as default.
        return True
    return True


def _read_unitsize_long(work_folder: Path) -> dict[str, float]:
    """Read ``solve_data/p_entity_unitsize.csv`` (long format: entity, value)."""
    path = work_folder / "solve_data" / "p_entity_unitsize.csv"
    if not path.exists():
        path = work_folder / "input" / "p_entity_unitsize.csv"
    if not path.exists():
        return {}
    csv = __import__("csv")
    out: dict[str, float] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # Long format: entity, value.  If wide-format detected (header
        # row 1 is ``value``), fall through — flexpy currently uses
        # long elsewhere.
        for r in reader:
            if len(r) >= 2 and r[0]:
                try:
                    out[r[0]] = float(r[1])
                except ValueError:
                    continue
    return out


def _read_pre_existing_long(work_folder: Path) -> dict[tuple[str, str], float]:
    """Read ``solve_data/p_entity_pre_existing.csv`` (long: entity, period, value).

    Returns ``{(period, entity): value}`` to match
    flextool's ``_load_pre_existing`` key order (``[d, e]`` lookup).
    """
    path = work_folder / "solve_data" / "p_entity_pre_existing.csv"
    if not path.exists():
        return {}
    csv = __import__("csv")
    out: dict[tuple[str, str], float] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[0] and r[1]:
                try:
                    out[(str(r[1]), str(r[0]))] = float(r[2])
                except ValueError:
                    continue
    return out


def _read_singles_csv(path: Path) -> list[str]:
    """Read a single-column CSV (header row, then one value per row)."""
    if not path.exists():
        return []
    csv = __import__("csv")
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def build_handoff_from_flexpy(
    sol, work_folder: Path, solve_name: str,
    *, prior_handoff=None,
):
    """Build a ``SolveHandoff`` from a flexpy ``Solution`` + the work
    folder's per-solve metadata, mirroring flextool's post-solve
    ``write_p_entity_period_existing_capacity`` + ``write_p_entity_divested``
    logic but in-memory.

    Covers all 9 carriers (Γ.8.D extension — was 3 of 9 before the
    Γ.8.D port):

    * ``realized_invest`` — per-(entity, period) chain-cumulative invest.
    * ``realized_existing`` — per-(entity, period) chain-cumulative existing.
    * ``divest_cumulative`` — per-entity chain-cumulative divest.
    * ``roll_end_state`` — last-step v_state per nodeState node.
    * ``fix_storage`` — wide [node, period, time, quantity, price, usage]
      with NULL columns for inactive metrics.  ``quantity`` is populated
      from v_state at fix_storage_timesteps for fix_quantity nodes; the
      price (dual-based) and usage (flow-based) variants stay NULL until
      a fixture exercises them and the dual / flow extraction lands.
    * ``cumulative_co2`` — per-(group, period), summed from
      ``solve_data/co2_cum_realized_tonnes.csv`` if present.
    * ``cumulative_commodity`` — per-(commodity, tier, period),
      summed from ``solve_data/commodity_ladder_cumulative.csv`` if
      present.
    * ``cum_sim_hours`` — per-period running sim-hour total, sourced
      from ``solve_data/ladder_cum_sim_hours.csv`` if present.
    * ``periods_already_emitted`` — per-period bare-set, from
      ``solve_data/period_capacity.csv`` if present.

    The work folder must already have completed flextool's per-solve
    preprocessing for ``solve_name`` (so ``solve_data/`` carries
    ``period_first.csv``, ``solve__ed_invest.csv``,
    ``realized_invest_periods_of_current_solve.csv``, etc.).
    """
    import polars as pl  # local — keep this helper's import surface narrow
    # Native import — Γ.8.D moved SolveHandoff into engine_polars; the
    # legacy ``flextool.flextoolrunner.solve_handoff`` path re-exports
    # the same class for source compatibility (see R-O2 mitigation in
    # ``audit/solve_orchestration_plan.md`` and the shim header in
    # ``flextool/flextoolrunner/solve_handoff.py``).
    from flextool.engine_polars._solve_handoff import SolveHandoff

    sd = work_folder / "solve_data"
    first_solve = _read_solve_first(work_folder)
    unitsize = _read_unitsize_long(work_folder)
    pre_existing = _read_pre_existing_long(work_folder) if first_solve else {}

    # Prior solve's accumulators — sourced from the in-memory handoff
    # carriers when supplied, else empty (multi-solve cascade always
    # passes the parent handoff).
    prior_existing: dict[tuple[str, str], float] = {}
    prior_invested: dict[tuple[str, str], float] = {}
    prior_divested: dict[str, float] = {}
    if prior_handoff is not None:
        if prior_handoff.realized_existing is not None:
            for r in prior_handoff.realized_existing.iter_rows(named=True):
                prior_existing[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.realized_invest is not None:
            for r in prior_handoff.realized_invest.iter_rows(named=True):
                prior_invested[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.divest_cumulative is not None:
            for r in prior_handoff.divest_cumulative.iter_rows(named=True):
                prior_divested[str(r["entity"])] = float(r["value"])

    # ---- v_invest / v_divest from polar_high_opt ----
    invest_by_ed: dict[tuple[str, str], float] = {}
    divest_by_e: dict[str, float] = {}
    for var_name, entity_col in (("v_invest_p", "p"), ("v_invest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                invest_by_ed[(str(r[entity_col]), str(r["d"]))] = v
    for var_name, entity_col in (("v_divest_p", "p"), ("v_divest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                e = str(r[entity_col])
                divest_by_e[e] = divest_by_e.get(e, 0.0) + v

    # ---- iteration set: prior keys ∪ entity × iteration_periods ----
    realize_invest = _read_realize_invest_periods(
        sd / "realized_invest_periods_of_current_solve.csv"
    )
    period_first = _read_period_set(sd / "period_first.csv")
    if first_solve:
        realized_periods = _read_realized_dispatch_periods(sd / "realized_dispatch.csv")
        fix_storage_periods = _read_realized_dispatch_periods(sd / "fix_storage_timesteps.csv")
        iter_periods = realize_invest | realized_periods | fix_storage_periods
    else:
        iter_periods = set(realize_invest)

    iter_keys: set[tuple[str, str]] = set(prior_existing.keys())
    entities = _read_singles_csv(sd / "entity.csv")
    if not entities:
        entities = _read_singles_csv(work_folder / "input" / "entity.csv")
    for e in entities:
        for d in iter_periods:
            iter_keys.add((e, d))

    # ---- compute realized_invest + realized_existing per (e, d) ----
    # The handoff is a CHAIN-CUMULATIVE record: every solve carries
    # forward prior solves' (e, d) contributions and ADDS its own.
    # ``first_solve`` (the .mod's solveFirst flag) means the SOLVE
    # treats itself as fresh on the LP side (no roll-state subtraction)
    # — but the OUTPUT handoff still has to cumulate, otherwise downstream
    # solves lose history.  So prior_existing / prior_invested are added
    # for every key, regardless of first_solve.
    inv_rows: list[tuple[str, str, float]] = []
    exist_rows: list[tuple[str, str, float]] = []
    for e, d in sorted(iter_keys):
        existing = 0.0
        invested = 0.0
        # Carry prior solves' contributions forward unconditionally.
        existing += prior_existing.get((e, d), 0.0)
        invested += prior_invested.get((e, d), 0.0)
        # First-solve seed: the user-defined pre-existing capacity becomes
        # part of ``realized_existing`` only on the first solve in the chain,
        # at periods belonging to that solve's period_first set.
        if first_solve and d in period_first:
            existing += pre_existing.get((d, e), 0.0)
        # This solve's invest contribution at (e, d).
        if (e, d) in invest_by_ed and d in realize_invest:
            v = invest_by_ed[(e, d)]
            us = unitsize.get(e, 1.0)
            existing += v * us
            invested += v * us
        inv_rows.append((e, d, invested))
        exist_rows.append((e, d, existing))

    # ---- divest_cumulative: prior + sum_d v_divest * unitsize ----
    entity_divest = set(_read_singles_csv(sd / "entityDivest.csv"))
    div_rows: list[tuple[str, float]] = []
    for e in sorted(entity_divest):
        cum = prior_divested.get(e, 0.0) + divest_by_e.get(e, 0.0) * unitsize.get(e, 1.0)
        div_rows.append((e, cum))

    # ---- roll_end_state: v_state[n, last_t] * unitsize per nodeState node ----
    # Mirrors flextool's ``write_p_roll_continue_state``: takes v_state at
    # the LAST realized (period, time) pair (from ``period__time_last.csv``),
    # multiplies by p_entity_unitsize, and emits a (node, value) row per
    # nodeState node.  Skipped when the solve has no nodeState or no
    # period_last.  See flextool/process_outputs/handoff_writers.py:425-468.
    roll_end_state_df = None
    nodes_state = _read_singles_csv(sd / "nodeState.csv")
    pt_last_path = sd / "period__time_last.csv"
    if not pt_last_path.exists():
        pt_last_path = sd / "block_period_time_last.csv"
    if nodes_state and pt_last_path.exists() and "v_state" in sol._vars:
        # Schema: ``period, step`` for period__time_last, or
        # ``block, period, step`` for block_period_time_last.  We want the
        # LAST (period, step) — flextool's writer iterates over the file
        # and overwrites, so the LAST row of the file's per-period entries
        # wins.  For block_period_time_last each block writes one row per
        # period; we just take the unique (period, step) at the maximum.
        last_pairs_df = pl.read_csv(pt_last_path)
        if last_pairs_df.height > 0:
            cols = last_pairs_df.columns
            d_col = "period" if "period" in cols else "d"
            t_col = "step" if "step" in cols else "t"
            # Pick the lexically-last (d, t) pair — matches flextool's
            # "iterates and overwrites; final entry wins" semantics.
            last_pairs_df = (last_pairs_df
                .select(pl.col(d_col).alias("d"),
                         pl.col(t_col).alias("t"))
                .unique()
                .sort(["d", "t"]))
            if last_pairs_df.height > 0:
                last_d = last_pairs_df["d"][-1]
                last_t = last_pairs_df["t"][-1]
                v_state = sol.value("v_state")
                rcs_rows: list[tuple[str, float]] = []
                if v_state is not None and v_state.height > 0:
                    last_state = v_state.filter(
                        (pl.col("d") == last_d) & (pl.col("t") == last_t))
                    nodes_state_set = set(nodes_state)
                    for r in last_state.iter_rows(named=True):
                        n = str(r["n"])
                        if n not in nodes_state_set:
                            continue
                        v = float(r["value"]) * unitsize.get(n, 1.0)
                        rcs_rows.append((n, v))
                if rcs_rows:
                    roll_end_state_df = pl.DataFrame(
                        rcs_rows, schema=["node", "value"], orient="row")

    # ---- fix_storage: v_state at fix_quantity timesteps × unitsize ----
    # Mirrors flextool's ``write_fix_storage_quantity`` (handoff_writers.py
    # :380).  Restricted to nodes whose storage_nested_fix_method is
    # ``fix_quantity`` and (period, step) in fix_storage_timesteps.csv.
    # The fix_price (dual-based) and fix_usage (flow-based) variants are
    # left unfilled for now — they require nodeBalance_eq dual extraction
    # / per-arc flow summation which is significantly more involved than
    # the quantity case and isn't exercised by the multi_invest fixture.
    fix_storage_df = None
    fq_nodes: set[str] = set()
    nsfm_path = sd / "node__storage_nested_fix_method.csv"
    if nsfm_path.exists():
        nsfm_df = pl.read_csv(nsfm_path)
        if nsfm_df.height > 0 and "method" in nsfm_df.columns:
            fq_nodes = set(
                nsfm_df.filter(pl.col("method") == "fix_quantity")["node"]
                .cast(pl.Utf8).to_list()
            )
    fix_steps_path = sd / "fix_storage_timesteps.csv"
    if (fq_nodes
            and fix_steps_path.exists()
            and "v_state" in sol._vars):
        fs_steps_df = pl.read_csv(fix_steps_path)
        if fs_steps_df.height > 0 and {"period", "step"}.issubset(
                fs_steps_df.columns):
            fs_steps = (fs_steps_df
                .select(pl.col("period").alias("d"),
                         pl.col("step").alias("t"))
                .unique())
            v_state = sol.value("v_state")
            if v_state is not None and v_state.height > 0:
                fq_rows = (v_state
                    .filter(pl.col("n").is_in(list(fq_nodes)))
                    .join(fs_steps, on=["d", "t"], how="inner"))
                if fq_rows.height > 0:
                    # Multiply by unitsize (per-node) and emit wide schema
                    # [node, period, time, quantity, price, usage].
                    us_rows = [(n, unitsize.get(n, 1.0))
                                 for n in sorted(fq_nodes)]
                    us_df = pl.DataFrame(
                        us_rows, schema=["n", "us"], orient="row")
                    fq_rows = (fq_rows
                        .join(us_df, on="n", how="inner")
                        .with_columns(quantity=pl.col("value") * pl.col("us"))
                        .select(
                            pl.col("n").alias("node"),
                            pl.col("d").alias("period"),
                            pl.col("t").alias("time"),
                            pl.col("quantity"),
                            pl.lit(None).cast(pl.Float64).alias("price"),
                            pl.lit(None).cast(pl.Float64).alias("usage"),
                        ))
                    if fq_rows.height > 0:
                        fix_storage_df = fq_rows

    # ---- cumulative_co2: per-(group, period) running total ----
    # Producer: written by flextool's preprocessing into
    # ``solve_data/co2_cum_realized_tonnes.csv`` between solves.  When the
    # snapshot already carries the file, propagate it; when prior_handoff
    # has the carrier, prefer that (in-memory beats disk).  When neither
    # is present, leave None so unexercised fixtures don't pay the cost.
    cumulative_co2_df = None
    if prior_handoff is not None and prior_handoff.cumulative_co2 is not None:
        cumulative_co2_df = prior_handoff.cumulative_co2
    co2_path = sd / "co2_cum_realized_tonnes.csv"
    if co2_path.exists():
        try:
            co2_df = pl.read_csv(co2_path)
        except pl.exceptions.NoDataError:
            co2_df = None
        if co2_df is not None and co2_df.height > 0 and \
                "p_co2_cum_realized_tonnes" in co2_df.columns:
            cumulative_co2_df = (
                co2_df.with_columns(
                    value=pl.col("p_co2_cum_realized_tonnes")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0))
                  .select("group", "period", "value"))

    # ---- cumulative_commodity: per-(commodity, tier, period) running mwh ----
    # Same propagation pattern as cumulative_co2.
    cumulative_commodity_df = None
    if prior_handoff is not None and prior_handoff.cumulative_commodity is not None:
        cumulative_commodity_df = prior_handoff.cumulative_commodity
    cc_path = sd / "commodity_ladder_cumulative.csv"
    if cc_path.exists():
        try:
            cc_df = pl.read_csv(cc_path)
        except pl.exceptions.NoDataError:
            cc_df = None
        if cc_df is not None and cc_df.height > 0:
            # Tolerate either ``mwh`` or ``p_ladder_cum_realized_mwh`` as
            # the value column name — the file's writer may use either.
            if "mwh" in cc_df.columns:
                value_col = "mwh"
            elif "p_ladder_cum_realized_mwh" in cc_df.columns:
                value_col = "p_ladder_cum_realized_mwh"
            else:
                value_col = None
            if value_col is not None and {"commodity", "tier", "period"}.issubset(
                    cc_df.columns):
                cumulative_commodity_df = (
                    cc_df.with_columns(
                        mwh=pl.col(value_col).cast(pl.Float64, strict=False)
                                .fill_null(0.0))
                      .select("commodity", "tier", "period", "mwh"))

    # ---- cum_sim_hours: per-period running sim-hour total ----
    cum_sim_hours_df = None
    if prior_handoff is not None and prior_handoff.cum_sim_hours is not None:
        cum_sim_hours_df = prior_handoff.cum_sim_hours
    csh_path = sd / "ladder_cum_sim_hours.csv"
    if csh_path.exists():
        try:
            csh_df = pl.read_csv(csh_path)
        except pl.exceptions.NoDataError:
            csh_df = None
        if csh_df is not None and csh_df.height > 0 and \
                "p_ladder_cum_sim_hours" in csh_df.columns:
            cum_sim_hours_df = (
                csh_df.with_columns(
                    value=pl.col("p_ladder_cum_sim_hours")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0))
                   .select("period", "value"))

    # ---- periods_already_emitted: bare set of period strings ----
    # Each solve adds the periods it just emitted output rows for; the
    # carrier accumulates across the chain so a downstream solve can
    # gate re-emission.  Source: ``solve_data/period_capacity.csv``.
    periods_already_emitted_df = None
    prior_periods: set[str] = set()
    if prior_handoff is not None and prior_handoff.periods_already_emitted is not None:
        prior_periods = set(
            str(p) for p in prior_handoff.periods_already_emitted["period"].to_list()
        )
    pae_path = sd / "period_capacity.csv"
    new_periods: set[str] = set()
    if pae_path.exists():
        try:
            pae_df = pl.read_csv(pae_path)
        except pl.exceptions.NoDataError:
            pae_df = None
        if pae_df is not None and pae_df.height > 0 and "period" in pae_df.columns:
            new_periods = set(str(p) for p in pae_df["period"].to_list())
    all_periods = prior_periods | new_periods
    if all_periods:
        periods_already_emitted_df = pl.DataFrame(
            {"period": sorted(all_periods)}
        )

    # ---- fix_storage_price / fix_storage_usage extraction ----
    # The .csv reads above (sd / fix_storage_price.csv etc.) may already
    # carry parent-deposited values when this is a child of a
    # storage-fixing parent.  When fix_storage_df is already non-None
    # from the v_state-based quantity extraction, fold the price / usage
    # rows in via outer-join.  When the quantity extraction yielded
    # nothing, build the wide frame purely from the on-disk metric files.
    def _read_fix_csv(name: str, value_col: str) -> "pl.DataFrame | None":
        p = sd / name
        if not p.exists():
            return None
        try:
            df = pl.read_csv(p)
        except pl.exceptions.NoDataError:
            return None
        if df.height == 0 or value_col not in df.columns:
            return None
        # On-disk schema is (period, step, node, value_col).  Rename to
        # the carrier convention (node, period, time, metric).
        return (df
            .rename({"step": "time"})
            .select("node", "period", "time", value_col))

    fp = _read_fix_csv("fix_storage_price.csv", "p_fix_storage_price")
    fu = _read_fix_csv("fix_storage_usage.csv", "p_fix_storage_usage")
    if fp is not None or fu is not None:
        merged = fix_storage_df  # may be None if no v_state quantity rows
        for src, value_col, out_col in (
            (fp, "p_fix_storage_price", "price"),
            (fu, "p_fix_storage_usage", "usage"),
        ):
            if src is None:
                continue
            renamed = src.rename({value_col: out_col})
            if merged is None:
                merged = renamed
            else:
                merged = merged.join(
                    renamed, on=["node", "period", "time"],
                    how="full", coalesce=True,
                )
        # Backfill NULL columns for any of the three metrics still missing.
        if merged is not None:
            for c in ("quantity", "price", "usage"):
                if c not in merged.columns:
                    merged = merged.with_columns(
                        pl.lit(None).cast(pl.Float64).alias(c)
                    )
            fix_storage_df = merged.select(
                "node", "period", "time", "quantity", "price", "usage"
            )

    return SolveHandoff(
        realized_invest=pl.DataFrame(
            inv_rows, schema=["entity", "period", "value"], orient="row",
        ) if inv_rows else None,
        realized_existing=pl.DataFrame(
            exist_rows, schema=["entity", "period", "value"], orient="row",
        ) if exist_rows else None,
        divest_cumulative=pl.DataFrame(
            div_rows, schema=["entity", "value"], orient="row",
        ) if div_rows else None,
        roll_end_state=roll_end_state_df,
        fix_storage=fix_storage_df,
        cumulative_co2=cumulative_co2_df,
        cumulative_commodity=cumulative_commodity_df,
        cum_sim_hours=cum_sim_hours_df,
        periods_already_emitted=periods_already_emitted_df,
    )


def apply_handoff(flex_data: "FlexData", handoff,
                   solve_data_dir: Path | None = None) -> "FlexData":
    """Overlay an in-memory ``SolveHandoff`` onto an already-loaded
    :class:`FlexData`, returning a NEW FlexData.

    The original is unchanged (we use :func:`dataclasses.replace`).  This
    lets the chain runner load each sub-solve's snapshot for STRUCTURE
    (entity sets, methods, profiles, time, …) and then swap the
    sub-solve's pre-written handoff CSV state for the in-memory one
    extracted from the prior flexpy solve via
    :func:`build_handoff_from_flexpy` — a true standalone chain run.

    Carriers overlaid (target FlexData fields):

    * ``p_entity_previously_invested_capacity (e, d)``  ← derived from
      ``realized_invest`` summed over historical periods using
      ``solve_data/edd_history.csv``.  This mirrors flextool's
      ``write_p_entity_previously_invested_capacity`` (see
      ``preprocessing/entity_period_calc_params.py:1584``):
      ``v[e, d] = Σ_{(e, d_h, d) ∈ edd_history ∧ (e, d_h) realized}  realized_invest[(e, d_h)]``.
    * ``p_entity_invested (e,)``  ← ``realized_invest`` summed over period.
    * ``p_entity_divested (e,)``  ← ``divest_cumulative``.
    * ``p_roll_continue_state (n,)``  ← ``roll_end_state``.
    * ``p_fix_storage_quantity (n, d, t)``  ← ``fix_storage.quantity``.

    For each carrier, ``None`` on the handoff side leaves the FlexData
    field untouched (snapshot wins).  Non-None replaces the entire
    field — the handoff is the source of truth.  Rows with value=0.0
    are filtered to match the canonical loader's behaviour (see
    ``_read_handoff_e_d`` at L1215).

    Parameters
    ----------
    flex_data : FlexData
        Base FlexData (typically from ``load_flextool``) carrying the
        sub-solve's structure (sets, profiles, methods).
    handoff : SolveHandoff
        Carrier set built by :func:`build_handoff_from_flexpy` from the
        prior sub-solve's flexpy solution.
    solve_data_dir : Path, optional
        Path to the current sub-solve's ``solve_data/`` directory.
        Required for the ``p_entity_previously_invested_capacity``
        overlay (it reads ``edd_history.csv`` to know which prior
        invest periods feed each current period).  ``None`` skips that
        carrier; the snapshot's pre-written value is then used as-is.
    """
    from dataclasses import replace
    overrides: dict = {}

    # --- p_entity_previously_invested_capacity (e, d): realized_invest
    # summed over the historical d_h that feed each current d, per
    # solve_data/edd_history.csv ∩ ed_history_realized.
    # Mirrors flextool/preprocessing/entity_period_calc_params.py:1525-1543.
    if (handoff.realized_invest is not None
            and solve_data_dir is not None
            and (solve_data_dir / "edd_history.csv").exists()):
        # Build the (e, d_h) → realized_invest dict.
        ppic: dict[tuple[str, str], float] = {}
        for r in handoff.realized_invest.iter_rows(named=True):
            ppic[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        # ed_history_realized = keys(ppic) ∪ ed_history_realized_first.csv.
        ed_realized: set[tuple[str, str]] = set(ppic.keys())
        ehrf_path = solve_data_dir / "ed_history_realized_first.csv"
        if ehrf_path.exists():
            ehrf = pl.read_csv(ehrf_path)
            if ehrf.height > 0:
                for r in ehrf.iter_rows(named=True):
                    ed_realized.add((str(r["entity"]), str(r["period"])))
        # Sum realized_invest over historical d_h per (e, d).
        edd_hist = pl.read_csv(solve_data_dir / "edd_history.csv")
        prev_inv: dict[tuple[str, str], float] = {}
        if edd_hist.height > 0:
            for r in edd_hist.iter_rows(named=True):
                e = str(r["entity"]); d_h = str(r["period_history"])
                d = str(r["period"])
                if (e, d_h) in ed_realized:
                    prev_inv[(e, d)] = prev_inv.get((e, d), 0.0) \
                                       + ppic.get((e, d_h), 0.0)
        if prev_inv:
            rows = [(e, d, v) for (e, d), v in prev_inv.items() if v != 0.0]
            if rows:
                df = pl.DataFrame(rows,
                                    schema=["e", "d", "value"], orient="row")
                overrides["p_entity_previously_invested_capacity"] = \
                    Param(("e", "d"), df)
            else:
                overrides["p_entity_previously_invested_capacity"] = None
        else:
            overrides["p_entity_previously_invested_capacity"] = None

    # --- realized_invest → p_entity_invested (e,)  (sum over period) ---
    # ``p_entity_invested`` is a per-entity scalar (cumulative prior-solve
    # invest), the sum of ``realized_invest`` rows for that entity.
    if handoff.realized_invest is not None:
        df = (handoff.realized_invest
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                 .fill_null(0.0))
            .group_by("entity").agg(pl.col("value").sum())
            .rename({"entity": "e"})
            .filter(pl.col("value") != 0.0)
            .select("e", "value"))
        overrides["p_entity_invested"] = (
            Param(("e",), df) if df.height > 0 else None)

    # --- divest_cumulative → p_entity_divested (e,) ---
    if handoff.divest_cumulative is not None:
        df = (handoff.divest_cumulative
            .rename({"entity": "e"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                 .fill_null(0.0))
            .filter(pl.col("value") != 0.0)
            .select("e", "value"))
        overrides["p_entity_divested"] = (
            Param(("e",), df) if df.height > 0 else None)

    # --- roll_end_state → p_roll_continue_state (n,) ---
    if handoff.roll_end_state is not None:
        df = (handoff.roll_end_state
            .rename({"node": "n"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
            .select("n", "value"))
        # The loader does NOT filter zero rows for this carrier (see
        # input.py:1996-1999) — it keeps them.  Match that behaviour.
        overrides["p_roll_continue_state"] = (
            Param(("n",), df) if df.height > 0 else None)

    # --- fix_storage → p_fix_storage_quantity (n, d, t) ---
    # Only the ``quantity`` metric is consumed today; price/usage extractors
    # are out of scope for this session (see SolveHandoff docstring).
    if handoff.fix_storage is not None and "quantity" in handoff.fix_storage.columns:
        df = (handoff.fix_storage
            .filter(pl.col("quantity").is_not_null())
            .rename({"node": "n", "period": "d", "time": "t",
                     "quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
            .select("n", "d", "t", "value"))
        overrides["p_fix_storage_quantity"] = (
            Param(("n", "d", "t"), df) if df.height > 0 else None)

    if not overrides:
        return flex_data
    return _assign_param_names(replace(flex_data, **overrides))


def load_flextool_from_db(input_db_url: str | Path,
                           scenario_name: str | None = None,
                           *,
                           flextool_dir: Path | str | None = None,
                           bin_dir: Path | str | None = None,
                           work_folder: Path | str | None = None,
                           ) -> "FlexData":
    """Load FlexData by running flextool's preprocessing pipeline directly
    from a Spine input database, bypassing flextool's GMPL solver.

    Internally this:

    1. Constructs a ``FlexToolRunner`` (reads DB into ``RunnerState``).
    2. Calls ``write_input()`` which writes ``input/`` and the L0-L9
       batch ``solve_data/*.csv`` via Python preprocessing.
    3. Runs ``orchestration.run_model()`` with a no-op solver, so the
       per-solve preprocessing (``preprocessing_solve_time``,
       ``solve_writers``) writes all the additional ``solve_data/*.csv``
       flexpy needs — without invoking glpsol/HiGHS on flextool's side.
    4. Loads from the resulting work folder via :func:`load_flextool`.

    The CSV roundtrip still happens to disk (in ``work_folder``, which
    can be a tempdir).  Eliminating the roundtrip requires refactoring
    each preprocessing module to return frames in addition to / instead
    of writing CSVs — that's a separate, larger effort.

    Parameters
    ----------
    input_db_url : str | Path
        Spine SQLite URL or path.  A bare path is upgraded to ``sqlite:///``.
    scenario_name : str, optional
        Scenario filter to apply.  ``None`` picks the first scenario
        in the database.
    flextool_dir, bin_dir : Path, optional
        Override the default flextool install location.  Default: assume
        ``~/sources/flextool/{flextool,bin}``.
    work_folder : Path | str, optional
        Where to stage the CSVs.  ``None`` (default) uses a tempdir
        that is **not** auto-cleaned (so failures can be inspected).
    """
    import logging
    import sys
    import tempfile
    REPO = Path("/home/jkiviluo/sources/flextool")
    # Append (not insert) so flexpy's local ``flextool/`` package takes
    # precedence as the importable name; flextool's runner submodule
    # is reachable via ``flextool.flextoolrunner`` because flextool's
    # __init__ exports it.
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
    from flextool.flextoolrunner import orchestration
    from flextool.flextoolrunner.solver_runner import SolverRunner

    if work_folder is None:
        work_folder = Path(tempfile.mkdtemp(prefix="flexpy_db_"))
    else:
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)

    db_url = str(input_db_url)
    if not db_url.startswith("sqlite:"):
        db_url = f"sqlite:///{db_url}"

    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario_name,
        flextool_dir=Path(flextool_dir) if flextool_dir else REPO / "flextool",
        bin_dir=Path(bin_dir) if bin_dir else REPO / "bin",
        work_folder=work_folder,
    )
    runner.write_input(db_url, scenario_name)

    # Quiet the flextool logger's stdout chatter.
    runner.state.logger.setLevel(logging.ERROR)

    # Detect single- vs multi-solve from the (already-built) solve config.
    # Top-level solves are the values of the first ``model`` entry; nested
    # rolling expands these into more iterations inside ``run_model``,
    # but for the supported (non-rolling) cascade fixtures the top-level
    # count matches the iteration count.
    solves = next(iter(runner.state.solve.model_solve.values()))
    total_solves = len(solves)

    if total_solves <= 1:
        # Single-solve: orchestration writes per-solve preprocessing
        # CSVs (timesets, scaling, period_first, etc.) and the no-op
        # solver suppresses the actual GMPL/HiGHS run.
        class _NoOpSolver(SolverRunner):
            def run(self, complete_solve_name: str) -> int:  # noqa: ARG002
                return 0
        orchestration.run_model(runner.state, _NoOpSolver(runner.state))
        return load_flextool(work_folder)

    # Multi-solve cascade: drive flextool's loop with a custom solver
    # that runs flexpy on every solve except the last, builds a
    # ``SolveHandoff`` from each solution, and deposits it into
    # ``state.handoffs`` so the next iteration's preprocessing picks
    # it up via the consume side wired in flextool.  The final solve's
    # preprocessing runs but the solve itself is skipped — the caller
    # builds + solves it externally on the returned ``FlexData`` and
    # compares to the multi-solve reference obj.
    runner.state.handoffs = {}  # opt-in: enable capture + consume

    class _FlexpyCascadeSolver(SolverRunner):
        def __init__(self, runner_state, total_solves: int):
            super().__init__(runner_state)
            self._total = total_solves
            self._count = 0

        def run(self, complete_solve_name: str) -> int:
            self._count += 1
            if self._count == self._total:
                return 0  # caller solves the last one
            data = load_flextool(self.state.paths.work_folder)
            from polar_high_opt import Problem
            from flextool.engine_polars.model import build_flextool as _build
            pb = Problem()
            _build(pb, data)
            sol = pb.solve()
            if not sol.optimal:
                self.state.logger.error(
                    f"flexpy non-optimal for {complete_solve_name}"
                )
                return 1
            prior = (
                self.state.handoffs.get(self.state.last_captured_solve)
                if self.state.last_captured_solve is not None else None
            )
            handoff = build_handoff_from_flexpy(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
            )
            self.state.handoffs[complete_solve_name] = handoff
            return 0

    orchestration.run_model(
        runner.state, _FlexpyCascadeSolver(runner.state, total_solves),
    )
    return load_flextool(work_folder)


