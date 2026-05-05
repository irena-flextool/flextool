"""Group-level slack constraints: ``capacity_margin``, ``inertia``, ``non_sync``.

Each is a "soft floor" constraint: a sum over processes (or unit existence)
must meet a per-group floor; a slack variable absorbs any shortfall and is
priced at a high penalty in the objective.

Module API
----------

``has_feature(d) -> bool``
    True if any of the three group-slack features is active in this scenario.

``load_data(inp, sd, dt, nb, pss_eff, pss_noEff, p_unitsize, **_) -> dict``
    Read the solve_data / input CSVs the features need and return a dict
    of new ``FlexData`` fields.  Fields that are unused in the current
    scenario are returned as ``None`` (mirroring the rest of the loader's
    convention so the merge agent can `**` -merge into the FlexData ctor).

``add_constraints(m, d, vars) -> None``
    Emit ``capacityMargin``, ``inertia_constraint`` and ``non_sync_constraint``
    where their data is non-empty.  Declares its own slack variables
    ``vq_capacity_margin``, ``vq_inertia``, ``vq_non_synchronous`` and
    publishes them back into ``vars`` (key names match var names) so
    ``add_objective_terms`` can consume them.

``add_objective_terms(m, d, vars, op_factor) -> Expr | None``
    Build the slack-penalty contribution of these three features.  Returns
    ``None`` if the feature is not active (caller must guard).

New FlexData fields (set by ``load_data``)
------------------------------------------

Sets
~~~~

* ``groupCapacityMargin``  — ``(g,)``  rows are groups whose
  ``pdGroup_capacity_margin`` is non-empty.
* ``groupInertia``         — ``(g,)``  rows are groups whose
  ``pdGroup_inertia_limit``  is non-empty.
* ``groupNonSync``         — ``(g,)``  rows are groups in
  ``input/groupNonSync.csv`` (mirrors the .mod's hand-curated set).
* ``group_node``           — ``(g, n)`` flextool's group__node (sd copy).
* ``process_sink_inertia`` — ``(p, sink)`` rows for which
  ``p_process_sink[p, sink, 'inertia_constant']`` exists.
* ``process_source_inertia`` — ``(p, source)`` analogous.
* ``process_sink_nonSync`` — ``(p, sink)`` flextool's process__sink_nonSync.
* ``process_group_inside_nonSync`` — ``(p, g)`` set of process-inside-group
  pairs that are excluded from non-sync flow accounting.

Parameters
~~~~~~~~~~

* ``p_inv_group_cap``                     — ``(g, d)``  inverse of
  group_capacity_for_scaling, used as a row-scaler in capacity_margin and
  non_sync.  Equals ``1 / group_capacity_for_scaling`` per (g, d).
* ``p_group_capacity_for_scaling``        — ``(g, d)``  the raw scaling
  factor — used by the slack term in the objective for capacity_margin.
* ``pdGroup_capacity_margin``             — ``(g, d)``  RHS of capacity
  margin constraint.
* ``pdGroup_penalty_capacity_margin``     — ``(g, d)``  unit penalty.
* ``pdGroup_inertia_limit``               — ``(g, d)``  RHS of inertia.
* ``pdGroup_penalty_inertia``             — ``(g, d)``  unit penalty.
* ``pdGroup_non_synchronous_limit``       — ``(g, d)``  per-group share.
* ``pdGroup_penalty_non_synchronous``     — ``(g, d)``  unit penalty.
* ``p_process_sink_inertia_constant``     — ``(p, sink)``.
* ``p_process_source_inertia_constant``   — ``(p, source)``.
* ``p_positive_inflow``                   — ``(n, d, t)`` exogenous +
  inflow part used by non_sync.  Note flextool also has a no-overlap
  ``p_negative_inflow`` (consumed in the same constraint).
* ``p_negative_inflow``                   — ``(n, d, t)``.
* ``pdtNodeInflow_per_step``              — ``(n, d, t)`` =
  ``pdtNodeInflow / p_step_duration``.  Used in capacity_margin RHS.

Variables created in ``add_constraints``
----------------------------------------

* ``vq_capacity_margin[g, d]``          (period-only).
* ``vq_inertia[g, d, t]``               .
* ``vq_non_synchronous[g, d, t]``       .

Required existing data fields (must be present before ``add_constraints``)
--------------------------------------------------------------------------

* ``v_flow``              (in ``vars``).
* ``v_online_lin`` / ``v_online_int``   (only if the model has UC; used by
  inertia LHS for ``process_online`` processes and by non-sync's
  min-load section term).
* ``v_invest_p`` / ``v_divest_p``       (only if the scenario has invest;
  used by capacity_margin LHS for available-capacity-with-invest).
* ``p_unitsize``, ``p_slope``, ``p_section`` (already on ``d``).
* ``p_step_duration``, ``p_rp_cost_weight``, ``p_inflation_op``,
  ``p_period_share`` (already on ``d``).
* ``p_process_existing_count``, ``edd_invest_set``, ``pd_divest_set``,
  ``edd_divest_active`` (only when invest is active).
* ``p_inflow``  -OR- direct ``pdtNodeInflow`` (used to derive
  ``pdtNodeInflow_per_step``); we recompute from ``p_inflow *
  p_step_duration`` if the canonical pdtNodeInflow.csv is unavailable.

Hooks
-----

* In ``build_flextool``: invoke ``_group_slack.add_constraints(m, d, vars)``
  AFTER existing constraint emission (storage, ramps, profiles), BEFORE
  ``m.set_objective``.  Then ``obj = obj + _group_slack.add_objective_terms(m,
  d, vars, op_factor) or 0``.
* In ``load_flextool``: ``data = data._replace(**_group_slack.load_data(
  inp, sd, dt, nb, pss_eff, pss_noEff, p_unitsize))`` before returning, or
  set the fields directly as part of the ctor call.

Sign conventions follow ``audit/constraints_audit.md`` and
``audit/objective_audit.md`` term-by-term.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

from polar_high import Sum, Where, Param
from polar_high.engine import Var, Expr

from ._input_source import _read_csv_file


# ---------------------------------------------------------------------------
# Sentinel keys (must match the FlexData fields the merge agent will add).

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


def _blank() -> dict:
    """Return a dict-of-Nones for every field this module owns."""
    return {f: None for f in _FIELDS}


# ---------------------------------------------------------------------------
# Public predicate

def has_feature(d) -> bool:
    """True if *any* of the three group-level slack features is active.

    A feature is active when its corresponding group-set field has at
    least one row.  ``getattr`` with a default of ``None`` keeps this
    callable even if the merge agent is staged before the FlexData
    extension lands."""
    for f in ("groupCapacityMargin", "groupInertia", "groupNonSync"):
        s = getattr(d, f, None)
        if s is not None and s.height > 0:
            return True
    return False


def _has_capacity_margin(d) -> bool:
    s = getattr(d, "groupCapacityMargin", None)
    return s is not None and s.height > 0


def _has_inertia(d) -> bool:
    s = getattr(d, "groupInertia", None)
    return s is not None and s.height > 0


def _has_non_sync(d) -> bool:
    s = getattr(d, "groupNonSync", None)
    return s is not None and s.height > 0


# ---------------------------------------------------------------------------
# Data loading helpers

def _read_csv_or_none(p: Path) -> pl.DataFrame | None:
    if not p.exists(): return None
    df = _read_csv_file(p)
    if df.height == 0:    return None
    return df


def _slice_pdgroup(sd: Path, param: str) -> pl.DataFrame | None:
    """Slice ``solve_data/pdGroup.csv`` (canonical long format
    ``group, param, period, value``) by literal ``param`` string —
    same operation .mod does inline via ``pdGroup[g, '<param>', d]``.

    Returns ``(g, d, value)`` or ``None`` if the file is missing /
    the slice is empty / the slice contains only zeros."""
    p = sd / "pdGroup.csv"
    if not p.exists(): return None
    df = _read_csv_file(p)
    if df.height == 0: return None
    sliced = (df.filter(pl.col("param") == param)
                .rename({"group": "g", "period": "d"})
                .select("g", "d", "value")
                .with_columns(pl.col("value").cast(pl.Float64, strict=False)))
    # Drop zero rows so the constraint set is naturally empty when the
    # feature isn't exercised.
    sliced = sliced.filter(pl.col("value") != 0.0)
    if sliced.height == 0: return None
    return sliced


def _slice_pdgroup_topfile(sd: Path, fname: str, value_col: str) -> pl.DataFrame | None:
    """Some pdGroup_*.csv files are written by flextool preprocessing as
    standalone wide-by-group files keyed on ``solve, period, <group1, …>``
    (e.g. ``pdGroup_inertia_limit.csv``).  Read either format and return
    a long ``(g, d, value)`` frame with zero rows dropped."""
    p = sd / fname
    if not p.exists(): return None
    df = _read_csv_file(p)
    if df.height == 0: return None
    if "solve" in df.columns:
        df = df.drop("solve")
    if {"period", value_col}.issubset(df.columns) and df.columns[0] == "group":
        # long form with explicit value column name
        out = (df.rename({"group": "g", "period": "d", value_col: "value"})
                 .select("g", "d", "value")
                 .with_columns(pl.col("value").cast(pl.Float64, strict=False)))
    elif "period" in df.columns and "group" in df.columns and "value" in df.columns:
        out = (df.rename({"group": "g", "period": "d"})
                 .select("g", "d", "value")
                 .with_columns(pl.col("value").cast(pl.Float64, strict=False)))
    elif "period" in df.columns:
        # wide-per-group: cols are (period, g1, g2, ...).
        val_cols = [c for c in df.columns if c != "period"]
        if not val_cols: return None
        out = (df.unpivot(on=val_cols, index=["period"],
                           variable_name="g", value_name="value")
                 .rename({"period": "d"})
                 .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                 .select("g", "d", "value"))
    else:
        return None
    out = out.filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
    if out.height == 0: return None
    return out


def _read_inertia_constants(inp: Path) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Parse ``p_process_sink.csv`` and ``p_process_source.csv`` for the
    ``inertia_constant`` parameter.

    Returns ``(sink_df, source_df)``, each ``(p, side, value)`` or None.

    Both files are long-format with columns
    ``[process, sink|source, sourceSinkParam, p_process_sink|source]``.
    See ``_read_p_process_side`` in input.py for the canonical shape."""
    def _read(path: Path, side: str) -> pl.DataFrame | None:
        if not path.exists(): return None
        df = _read_csv_file(path)
        if df.height == 0: return None
        if not {"process", "sourceSinkParam"}.issubset(df.columns):
            return None
        value_col = df.columns[-1]
        out = (df.filter(pl.col("sourceSinkParam") == "inertia_constant")
                 .rename({"process": "p", value_col: "value"})
                 .select("p", side, "value")
                 .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                 .filter(pl.col("value") != 0.0))
        return out if out.height > 0 else None
    sink_df  = _read(inp / "p_process_sink.csv",   "sink")
    src_df   = _read(inp / "p_process_source.csv", "source")
    return sink_df, src_df


def _read_inflow_signed(sd: Path, sign: str) -> pl.DataFrame | None:
    """Read flextool's ``p_positive_inflow.csv`` /
    ``p_negative_inflow.csv``.  Long format (node, period, time, value).
    Returns the long ``(n, d, t, value)`` frame; zero rows kept (the
    constraint emitter Where-joins them to group_node so empties drop)."""
    fname = "p_positive_inflow.csv" if sign == "pos" else "p_negative_inflow.csv"
    p = sd / fname
    if not p.exists(): return None
    df = _read_csv_file(p)
    if df.height == 0: return None
    out = (df.rename({"node": "n", "period": "d", "time": "t"})
             .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
             .select("n", "d", "t", "value"))
    return out


# ---------------------------------------------------------------------------
# load_data

def load_data(inp: Path, sd: Path, dt: pl.DataFrame,
              nb: pl.DataFrame | None,
              pss_eff: pl.DataFrame | None,
              pss_noEff: pl.DataFrame | None,
              p_unitsize: Param | None,
              **_unused) -> dict:
    """Read group-level slack data from ``input/`` + ``solve_data/``.

    The merge agent calls this from ``load_flextool`` and merges the
    result into the ``FlexData`` constructor.

    All returned fields are independently optional.  Missing CSVs ⇒ None.
    """
    out = _blank()

    # ── Group sets ────────────────────────────────────────────────────────
    # The .mod derives groupCapacityMargin / groupInertia from non-empty rows
    # in pdGroup_capacity_margin.csv / pdGroup_inertia_limit.csv (param-bearing
    # rows of pdGroup.csv).  groupNonSync is hand-curated in
    # input/groupNonSync.csv.
    cap_pd = _slice_pdgroup_topfile(sd, "pdGroup_capacity_margin.csv",
                                     "capacity_margin")
    if cap_pd is None:
        cap_pd = _slice_pdgroup(sd, "capacity_margin")
    iner_pd = _slice_pdgroup_topfile(sd, "pdGroup_inertia_limit.csv",
                                      "electricity")  # value col is the group name
    if iner_pd is None:
        iner_pd = _slice_pdgroup(sd, "inertia_limit")
    nsync_pd = _slice_pdgroup(sd, "non_synchronous_limit")  # only via pdGroup.csv

    # Set frames keyed by group name.  Δ.12-drop: the corresponding
    # ``pdGroup_capacity_margin`` / ``pdGroup_inertia_limit`` /
    # ``pdGroup_non_synchronous_limit`` Params are produced
    # authoritatively by ``apply_direct_params`` (Δ.4b).
    if cap_pd is not None:
        out["groupCapacityMargin"] = cap_pd.select("g").unique()
    if iner_pd is not None:
        out["groupInertia"] = iner_pd.select("g").unique()

    # groupNonSync: prefer the explicit input file (canonical), fallback to
    # pd_group's non_synchronous_limit slice.
    g_ns_path = inp / "groupNonSync.csv"
    if g_ns_path.exists():
        df = _read_csv_file(g_ns_path)
        if df.height > 0:
            # Header column might be "groupNonSync" (legacy) or "group".
            col = df.columns[0]
            out["groupNonSync"] = (df.rename({col: "g"})
                                     .select("g").unique())
    if out["groupNonSync"] is None and nsync_pd is not None:
        out["groupNonSync"] = nsync_pd.select("g").unique()

    # Δ.12-drop: ``pdGroup_penalty_capacity_margin`` /
    # ``pdGroup_penalty_inertia`` / ``pdGroup_penalty_non_synchronous``
    # produced authoritatively by ``apply_direct_params`` (Δ.4b).  The
    # legacy CSV reads of pdGroup_penalty_*.csv (and the pdGroup.csv
    # ``penalty_*`` slice fallback) are dropped.

    # ── group_node ────────────────────────────────────────────────────────
    # Canonical preprocessing target: solve_data/group_node.csv.
    # Fallback: input/group__node.csv (raw user input).
    gn = None
    for path in (sd / "group_node.csv", inp / "group__node.csv"):
        if path.exists():
            df = _read_csv_file(path)
            if df.height > 0:
                # both shapes are (group, node)
                gn = df.rename({"group": "g", "node": "n"}).select("g", "n").unique()
                break
    out["group_node"] = gn

    # ── process_unit ──────────────────────────────────────────────────────
    # The .mod's `process_unit` set narrows producer/consumer LHS terms in
    # capacityMargin (and friends) to "unit"-typed processes — i.e. it
    # excludes connection processes which would otherwise double-count
    # transmission capacity.
    # Canonical solve_data file has header `process_unit`; some legacy
    # files use `process`.  Either works here.
    pu_path = sd / "process_unit.csv"
    if pu_path.exists():
        df = _read_csv_file(pu_path)
        if df.height > 0:
            col = df.columns[0]
            out["process_unit"] = (df.rename({col: "p"})
                                     .select("p").unique())

    # ── group_capacity_for_scaling + inv_group_cap ───────────────────────
    p_gcs = sd / "group_capacity_for_scaling.csv"
    p_inv = sd / "inv_group_cap.csv"
    if p_gcs.exists():
        df = _read_csv_file(p_gcs)
        if df.height > 0:
            df = (df.rename({"group": "g", "period": "d"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .select("g", "d", "value")
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0)))
            if df.height > 0:
                out["p_group_capacity_for_scaling"] = Param(("g", "d"), df)
    if p_inv.exists():
        df = _read_csv_file(p_inv)
        if df.height > 0:
            df = (df.rename({"group": "g", "period": "d"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .select("g", "d", "value")
                    .filter(pl.col("value").is_not_null()))
            if df.height > 0:
                out["p_inv_group_cap"] = Param(("g", "d"), df)
    # Derive inv_group_cap from group_capacity_for_scaling if missing.
    if (out["p_inv_group_cap"] is None
            and out["p_group_capacity_for_scaling"] is not None):
        gcs = out["p_group_capacity_for_scaling"].frame
        gcs_inv = (gcs.with_columns(value=1.0 / pl.col("value"))
                      .select("g", "d", "value"))
        out["p_inv_group_cap"] = Param(("g", "d"), gcs_inv)
    # And vice versa (so each is non-None when the other is provided).
    if (out["p_group_capacity_for_scaling"] is None
            and out["p_inv_group_cap"] is not None):
        ig = out["p_inv_group_cap"].frame
        gcs = (ig.with_columns(value=1.0 / pl.col("value"))
                 .select("g", "d", "value"))
        out["p_group_capacity_for_scaling"] = Param(("g", "d"), gcs)

    # ── inertia_constant index sets ─────────────────────────────
    # Δ.12-drop: ``p_process_sink_inertia_constant`` /
    # ``p_process_source_inertia_constant`` Params are produced
    # authoritatively by ``apply_direct_params`` (Δ.4b).  The set frames
    # ``process_sink_inertia`` / ``process_source_inertia`` are kept on
    # the seed path for SIMPLE_PROJECTIONS' fall-through semantics.
    sink_df, src_df = _read_inertia_constants(inp)
    if sink_df is not None:
        out["process_sink_inertia"]                 = sink_df.select("p", "sink").unique()
    if src_df is not None:
        out["process_source_inertia"]               = src_df.select("p", "source").unique()

    # ── non-sync supporting sets ─────────────────────────────────────────
    p_sink_ns = _read_csv_or_none(sd / "process__sink_nonSync.csv")
    if p_sink_ns is not None:
        out["process_sink_nonSync"] = (p_sink_ns
            .rename({"process": "p"})
            .select("p", "sink").unique())
    p_grp_inside = _read_csv_or_none(sd / "process__group_inside_group_nonSync.csv")
    if p_grp_inside is not None:
        cols = p_grp_inside.columns
        rn = {}
        if "process" in cols: rn["process"] = "p"
        if "group"   in cols: rn["group"]   = "g"
        out["process_group_inside_nonSync"] = (
            p_grp_inside.rename(rn).select("p", "g").unique())

    # Δ.12-drop: ``p_positive_inflow`` / ``p_negative_inflow`` /
    # ``pdtNodeInflow_per_step`` produced authoritatively by
    # ``apply_derived_c`` (helpers ``p_positive_inflow_from_inflow`` /
    # ``p_negative_inflow_from_inflow`` /
    # ``pdtNodeInflow_per_step_from_inflow``).  Seeds dropped.

    return out


# ---------------------------------------------------------------------------
# Constraint emitters

def add_constraints(m, d, vars: dict) -> None:
    """Emit ``capacityMargin``, ``inertia_constraint``, ``non_sync_constraint``.

    Each block is independently active depending on whether the
    corresponding group set has any rows.  Constraints are no-op when
    inactive, mirroring flextool.mod's data-driven shape.

    ``vars`` is the dict of decision vars created earlier in
    ``build_flextool``.  Slack variables created here are written back
    into ``vars`` under their var names for the objective stage.
    """
    if _has_capacity_margin(d):
        _add_capacity_margin(m, d, vars)
    if _has_inertia(d):
        _add_inertia(m, d, vars)
    if _has_non_sync(d):
        _add_non_sync(m, d, vars)


def _gd_index(d) -> pl.DataFrame:
    """Return distinct (d,) periods present in dt (used as the period axis
    for capacity_margin which is keyed on (g, d), not (g, d, t))."""
    return d.dt.select("d").unique()


# ── capacityMargin ───────────────────────────────────────────────────────

def _add_capacity_margin(m, d, vars: dict) -> None:
    """capacityMargin (mod:4093-4154).

    Indexed on ``g in groupCapacityMargin × (d, t) in dt`` where d is in
    ``period_invest`` (we treat d ∈ dt as the operational set, matching
    flexpy's existing pattern for other dispatch constraints).

    Sense: ``>=``.

    LHS terms (each scaled by inv_group_cap[g, d]):
      + profile-limited producing units: profile · capacity ·
        (existing + invest − divest), p ∈ process_unit, sink ∈ group_node \\
        nodeState, profile_method ∈ {upper_limit, fixed};
      + capacity-limited producing units (same predicates without profile):
        (existing + invest − divest);
      − consuming units: v_flow * (slope * sink_coef/source_coef) for eff
        + section term for min_load_efficiency online + v_flow for noEff;
      + vq_capacity_margin[g, d] · group_capacity_for_scaling · inv_group_cap
        = vq_capacity_margin[g, d]   (the slack scaling cancels).

    RHS terms (×inv_group_cap):
      + Σ_{(g,n) in group_node, n ∉ nodeState} (− pdtNodeInflow / step_duration)
      + pdGroup[g, 'capacity_margin', d]

    Notes on simplifications vs the .mod
      * We skip the consuming-units LHS and section-term refinements when
        ``v_flow``, ``p_slope`` or ``p_section`` are not on ``d`` — flexpy
        only emits the term in the LP when its data is materialised.
      * The ``inv_group_cap`` row-scaler is applied uniformly to every
        term except the slack (where * group_cap_for_scaling cancels).
    """
    # required pieces
    gcm  = d.groupCapacityMargin                      # (g,)
    pgcm = d.pdGroup_capacity_margin                  # Param (g, d)
    invc = d.p_inv_group_cap                          # Param (g, d)
    if gcm is None or pgcm is None or invc is None:
        return

    # Slack variable vq_capacity_margin[g, d] — exists once per (g, d) where
    # the constraint is emitted.  Domain: groupCapacityMargin × period.
    vq_dom = gcm.join(_gd_index(d), how="cross")
    vq_cap = m.add_var("vq_capacity_margin", ("g", "d"), vq_dom, lower=0.0)
    vars["vq_capacity_margin"] = vq_cap

    # Constraint domain: groupCapacityMargin × dt.
    over = gcm.join(d.dt, how="cross")

    # Scale everything by inv_group_cap[g, d].
    # The slack term's column-coefficient already multiplies by
    # group_capacity_for_scaling (so * group_cap_for_scaling * inv_group_cap = 1).
    lhs: dict = {}

    # Slack — coef +1 (after the cancellation noted above).
    lhs["slack"] = vq_cap

    # Producing-unit available capacity: existing + Σ_d_inv v_invest − Σ_d_div v_divest.
    # The .mod RHS in p_entity_all_existing already includes prior-solve
    # invest carryover; flexpy's preprocessing exposes existing/unitsize per
    # (p, d) in p_process_existing_count, in *unit count* — so to get
    # capacity we must multiply by p_entity_unitsize.
    pss     = d.process_source_sink
    pss_eff = d.process_source_sink_eff
    pss_noEff = d.process_source_sink_noEff
    proc_unit = getattr(d, "process_unit", None)   # mod's process_unit set
    # group_node is required to know which (g, n) pairs are in scope; if
    # absent we still emit the constraint with floor / inflow / slack so
    # the slack is properly priced (no producer/consumer terms then).
    gn_no_state = None
    pss_to_grp = None
    if d.group_node is not None:
        # group_node restricted to (g, n) ∈ groupCapacityMargin × n.
        gn = d.group_node.join(gcm, on="g", how="inner")  # (g, n)
        # Producer predicate: sink in group_node and sink not in nodeState.
        gn_no_state = gn
        if d.nodeState is not None and d.nodeState.height > 0:
            gn_no_state = gn.join(d.nodeState, on="n", how="anti")
    if pss is not None and gn_no_state is not None:
        # Restrict producer/consumer LHS to process_unit per the .mod
        # (filters out connection processes which double-count transmission
        # capacity).  If process_unit is missing, fall back to all of pss
        # — no worse than the prior behaviour, but log via a soft fall-back.
        pss_unit = pss
        if proc_unit is not None and proc_unit.height > 0:
            pss_unit = pss.join(proc_unit.select("p"), on="p", how="inner")
        # Map sink → n for join.
        pss_to_grp = (pss_unit.rename({"sink": "n"})
                              .join(gn_no_state, on="n", how="inner")
                              .rename({"n": "sink"}))   # (p, source, sink, g)

    # Profile-limited subset (upper_limit or fixed):
    # pss_to_grp ∩ (process_profile_upper ∪ process_profile_fixed)
    has_profile = (
        (d.process_profile_upper is not None and d.process_profile_upper.height > 0) or
        (d.process_profile_fixed is not None and d.process_profile_fixed.height > 0))
    profile_idx = None
    if has_profile and pss_to_grp is not None:
        parts = []
        if d.process_profile_upper is not None and d.process_profile_upper.height > 0:
            parts.append(d.process_profile_upper.select("p", "source", "sink", "f"))
        if d.process_profile_fixed is not None and d.process_profile_fixed.height > 0:
            parts.append(d.process_profile_fixed.select("p", "source", "sink", "f"))
        profile_idx = pl.concat(parts, how="vertical").unique()
        # Restrict to (p, source, sink) in pss_to_grp.
        profile_idx = profile_idx.join(pss_to_grp,
                                        on=["p", "source", "sink"], how="inner")

    # Capacity-limited subset = pss_to_grp \ profile_idx (same (p, source, sink)).
    capacity_idx = pss_to_grp
    if pss_to_grp is not None and profile_idx is not None:
        capacity_idx = pss_to_grp.join(
            profile_idx.select("p", "source", "sink").unique(),
            on=["p", "source", "sink"], how="anti")

    # ── Producer LHS: profile_limited (profile · capacity · inv_group_cap)
    # We express "capacity" = existing_count[p, d] · unitsize[p]  +
    # Σ_d_inv v_invest_p[d_inv] · unitsize − Σ_d_div v_divest_p[d_div] · unitsize.
    # But flexpy's existing_count is already in unit count (=cap/unitsize),
    # so existing_count · unitsize · profile is a constant Param contribution
    # (RHS-side); the *variable* contributions are the invest / divest
    # times unitsize times profile.
    prof_val = d.p_profile_value                     # Param (f, d, t)
    exist_cnt = d.p_process_existing_count           # Param (p, d)
    p_us      = d.p_unitsize                         # Param (p,)
    # Note: if there are no processes (slack-only fixture) we skip all
    # producer/consumer blocks but still emit the constraint with
    # floor + inflow + slack (the .mod's degenerate case).
    has_process_data = (pss_to_grp is not None
                        and exist_cnt is not None and p_us is not None)

    # Constant (existing_count · unitsize · profile · inv_group_cap) is
    # subtracted from the RHS so the LHS has only variable terms.
    # We handle the profile & capacity legs separately, accumulate constants
    # into a Param `const_subtract` and the per-(g, d, t) RHS.

    # Helper: invest tightening summed over d_invest, restricted to p in pset.
    has_invest_p = (getattr(d, "pd_invest_set", None) is not None
                    and d.pd_invest_set.height > 0)
    has_divest_p = (getattr(d, "pd_divest_set", None) is not None
                    and d.pd_divest_set.height > 0)
    v_invest_p   = vars.get("v_invest_p")
    v_divest_p   = vars.get("v_divest_p")
    edd_inv_set  = getattr(d, "edd_invest_set", None)
    edd_div_act  = getattr(d, "edd_divest_active", None)

    def _invest_sum(v_inv: Var, pset_pl: pl.DataFrame, edd: pl.DataFrame,
                    rename_e: bool) -> Expr | None:
        """Return Σ_{d_invest} v_invest_p[d_invest] over edd ∩ pset, with
        d_invest contracted, leaving (p, d).  ``edd`` is keyed (e or p,
        d_invest, d).  rename_e=True ⇒ "e"→"p"."""
        if v_inv is None or edd is None or edd.height == 0:
            return None
        e = edd
        if rename_e and "e" in e.columns:
            e = e.rename({"e": "p"})
        e = e.join(pset_pl.select("p").unique(), on="p", how="inner")
        if e.height == 0:
            return None
        v_at = Var(name=v_inv.name + "__cap_margin_inv",
                   dims=("p", "d_invest"),
                   frame=v_inv.frame.rename({"d": "d_invest"}),
                   lower=v_inv.lower, upper=v_inv.upper)
        return Sum(Where(v_at, e), over=("d_invest",))

    def _divest_sum(v_div: Var, pset_pl: pl.DataFrame, edd: pl.DataFrame) -> Expr | None:
        if v_div is None or edd is None or edd.height == 0:
            return None
        e = edd.join(pset_pl.select("p").unique(), on="p", how="inner")
        if e.height == 0:
            return None
        v_at = Var(name=v_div.name + "__cap_margin_div",
                   dims=("p", "d_divest"),
                   frame=v_div.frame.rename({"d": "d_divest"}),
                   lower=v_div.lower, upper=v_div.upper)
        return Sum(Where(v_at, e), over=("d_divest",))

    # Constant existing-capacity contribution (subtracted from RHS).
    # Term-by-term so we don't lose nullity: each Param multiplication
    # is well-defined only if all factors are non-None.
    const_terms_add: list = []   # things added on RHS (existing capacity contributions)

    # — Profile-limited producers (LHS):
    if has_process_data and profile_idx is not None and prof_val is not None:
        # invest variables × profile × unitsize × inv_group_cap
        if has_invest_p:
            inv_sum = _invest_sum(v_invest_p, profile_idx, edd_inv_set, rename_e=True)
            if inv_sum is not None:
                # inv_sum is over (p, d).  Where on profile_idx adds
                # (source, sink, f, g) so it can join the dt index.
                term = (Where(inv_sum, profile_idx) * p_us * prof_val * invc)
                lhs["prof_invest"] = Sum(term, over=("p", "source", "sink", "f"))
        if has_divest_p:
            div_sum = _divest_sum(v_divest_p, profile_idx, edd_div_act)
            if div_sum is not None:
                term = -(Where(div_sum, profile_idx) * p_us * prof_val * invc)
                lhs["prof_divest"] = Sum(term, over=("p", "source", "sink", "f"))
        # Constant: existing_count · unitsize · profile · inv_group_cap
        # (= existing[p,d] · profile[f,d,t] · inv_group_cap[g,d]).
        # This becomes an *additive* RHS contribution (we move it from LHS
        # to RHS by sign flip).
        const_terms_add.append(
            ("prof_existing",
             # existing in unit count, * unitsize * profile * inv_group_cap.
             # Sum over (p, source, sink, f) leaves (g, d, t).
             # The Param-only product is realised via Sum-of-1*Param: we use
             # a dummy unit Var? No — instead, build a polars DF of the
             # constant aggregate at constraint emission time, see below.
             profile_idx, prof_val, exist_cnt, p_us, invc))

    # — Capacity-limited producers (LHS):  (no profile factor)
    if has_process_data and capacity_idx is not None and capacity_idx.height > 0:
        if has_invest_p:
            inv_sum = _invest_sum(v_invest_p, capacity_idx, edd_inv_set, rename_e=True)
            if inv_sum is not None:
                term = Where(inv_sum, capacity_idx) * p_us * invc
                lhs["cap_invest"] = Sum(term, over=("p", "source", "sink"))
        if has_divest_p:
            div_sum = _divest_sum(v_divest_p, capacity_idx, edd_div_act)
            if div_sum is not None:
                term = -(Where(div_sum, capacity_idx) * p_us * invc)
                lhs["cap_divest"] = Sum(term, over=("p", "source", "sink"))
        const_terms_add.append(
            ("cap_existing",
             capacity_idx, None, exist_cnt, p_us, invc))

    # — Consuming units (LHS, negative): for each (p, source, sink) in pss
    #   with source ∈ group_node, source ∉ nodeState, p ∈ process_unit.
    # eff: − v_flow * unitsize * slope * sink_coef/source_coef * inv_group_cap.
    # noEff: − v_flow * unitsize * inv_group_cap.
    # min_load_efficiency online section term: − v_online * section * unitsize * inv_group_cap.
    v_flow = vars.get("v_flow")
    v_online_lin = vars.get("v_online_lin")
    v_online_int = vars.get("v_online_int")
    if has_process_data and v_flow is not None:
        gn_src = gn_no_state
        # process_unit filter on the consumer side, mirroring the .mod
        # (the consumer LHS branch also has `p in process_unit`).
        def _filter_unit(df: pl.DataFrame) -> pl.DataFrame:
            if proc_unit is not None and proc_unit.height > 0:
                return df.join(proc_unit.select("p"), on="p", how="inner")
            return df
        # eff consumers
        if pss_eff is not None and pss_eff.height > 0:
            consuming_eff = (_filter_unit(pss_eff)
                              .rename({"source": "n"})
                              .join(gn_src, on="n", how="inner")
                              .rename({"n": "source"}))   # (p,source,sink,g)
            if consuming_eff.height > 0 and d.p_slope is not None:
                term = -(Where(v_flow, consuming_eff) * p_us * d.p_slope * invc)
                lhs["consume_eff"] = Sum(term, over=("p", "source", "sink"))
                # min_load section term (only if process_min_load_eff has any
                # of these processes).
                if (d.process_min_load_eff is not None
                        and d.process_min_load_eff.height > 0
                        and d.p_section is not None):
                    mle_in_eff = (consuming_eff.join(d.process_min_load_eff,
                                                      on="p", how="inner"))
                    if mle_in_eff.height > 0:
                        if v_online_lin is not None:
                            t_sec = -(Where(v_online_lin, mle_in_eff)
                                       * d.p_section * p_us * invc)
                            lhs["consume_section_lin"] = Sum(
                                t_sec, over=("p", "source", "sink"))
                        if v_online_int is not None:
                            t_sec = -(Where(v_online_int, mle_in_eff)
                                       * d.p_section * p_us * invc)
                            lhs["consume_section_int"] = Sum(
                                t_sec, over=("p", "source", "sink"))
        # noEff consumers
        if pss_noEff is not None and pss_noEff.height > 0:
            consuming_noEff = (_filter_unit(pss_noEff)
                                .rename({"source": "n"})
                                .join(gn_src, on="n", how="inner")
                                .rename({"n": "source"}))
            if consuming_noEff.height > 0:
                term = -(Where(v_flow, consuming_noEff) * p_us * invc)
                lhs["consume_noEff"] = Sum(term, over=("p", "source", "sink"))

    # ── RHS terms: existing-capacity LHS contributions (moved from LHS),
    #   plus pdGroup_capacity_margin · inv_group_cap, plus
    #   −Σ pdtNodeInflow/step_duration · inv_group_cap.

    # Build the existing-capacity Param aggregates as polars frames so we
    # can put them on the LHS as a Param (constant), but Param needs a
    # ``value`` column.  Param goes on the RHS (or LHS) as scalars only;
    # since we want to keep these on the LHS *side* of the inequality
    # alongside the variable terms, we instead subtract them from the RHS.
    # We build (g, d, t) → existing-capacity contribution.
    # However: capacity-limited LHS does *not* depend on t.  We broadcast
    # it onto (g, d, t) via a join with d.dt at the end.
    rhs: dict = {}

    # Compute static existing-capacity contributions per (g, d) or (g, d, t):
    # We'll merge into a single polars DataFrame and add as a Param on RHS
    # with a sign flip (since these are LHS positive contributions):
    #   LHS_existing = + cap_existing + prof_existing
    #   constraint:   LHS_var + LHS_existing >= RHS_terms
    #   ⇒  LHS_var  >=  RHS_terms − LHS_existing
    # On the RHS we add (− LHS_existing).
    existing_dt_frames: list[pl.DataFrame] = []   # (g, d, t, value)
    for entry in const_terms_add:
        kind = entry[0]
        if kind == "prof_existing":
            (_, idx, prof, exist, us, inv) = entry
            # idx = (p, source, sink, f, g)
            base = (idx.join(exist.frame.rename({"value": "exist"}),
                              on="p", how="inner")
                       .join(us.frame.rename({"value": "us"}),
                              on="p", how="inner")
                       .join(prof.frame.rename({"value": "prof"}),
                              on=["f", "d"], how="inner")
                       .join(inv.frame.rename({"value": "inv"}),
                              on=["g", "d"], how="inner"))
            # multiply
            agg = (base.with_columns(value=pl.col("exist") * pl.col("us")
                                            * pl.col("prof") * pl.col("inv"))
                       .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                       .select("g", "d", "t", "value"))
            existing_dt_frames.append(agg)
        elif kind == "cap_existing":
            (_, idx, _none, exist, us, inv) = entry
            # idx = (p, source, sink, g) — no t dependence
            base = (idx.join(exist.frame.rename({"value": "exist"}),
                              on="p", how="inner")
                       .join(us.frame.rename({"value": "us"}),
                              on="p", how="inner")
                       .join(inv.frame.rename({"value": "inv"}),
                              on=["g", "d"], how="inner"))
            agg_gd = (base.with_columns(value=pl.col("exist") * pl.col("us")
                                              * pl.col("inv"))
                          .group_by(["g", "d"]).agg(pl.col("value").sum()))
            agg = agg_gd.join(d.dt, on="d", how="inner").select("g", "d", "t", "value")
            existing_dt_frames.append(agg)
    if existing_dt_frames:
        ex = pl.concat(existing_dt_frames, how="vertical")
        ex = ex.group_by(["g", "d", "t"]).agg(pl.col("value").sum())
        # Negate so it goes to RHS.
        ex = ex.with_columns(value=-pl.col("value"))
        rhs["existing_neg"] = Param(("g", "d", "t"),
                                     ex.select("g", "d", "t", "value"))

    # pdGroup[g, 'capacity_margin', d] · inv_group_cap.
    rhs["floor"] = pgcm * invc

    # − pdtNodeInflow / step_duration · inv_group_cap, summed over n in
    # group_node : n ∉ nodeState.
    inflow_per = getattr(d, "pdtNodeInflow_per_step", None)
    if (inflow_per is not None and gn_no_state is not None
            and gn_no_state.height > 0):
        # Param → polars by joining group_node and inv_group_cap.
        flow_join = (gn_no_state.join(inflow_per.frame.rename({"value": "inflow"}),
                                       on="n", how="inner")
                                .join(invc.frame.rename({"value": "inv"}),
                                       on=["g", "d"], how="inner"))
        # − inflow * inv per (g, d, t), summed over n.
        rhs_inflow = (flow_join.with_columns(value=-pl.col("inflow") * pl.col("inv"))
                              .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
                              .select("g", "d", "t", "value"))
        if rhs_inflow.height > 0:
            rhs["inflow_neg"] = Param(("g", "d", "t"), rhs_inflow)

    m.add_cstr(
        "capacityMargin",
        over      = over,
        sense     = ">=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


# ── inertia_constraint ───────────────────────────────────────────────────

def _add_inertia(m, d, vars: dict) -> None:
    """inertia_constraint (mod:3943-3957).

    Indexed on ``g in groupInertia × (d, t) in dt``.  Sense: ``>=``.

    LHS:
      + Σ_{(p, source, sink) : (p, source) in process_source_inertia,
                               (g, source) in group_node}
          · ((v_online if p in process_online) | v_flow else)
          · inertia_constant[p, source]
          · unitsize[p]
      + same for sink-side
      + vq_inertia[g, d, t] * pdGroup[g, 'inertia_limit', d]

    RHS: pdGroup[g, 'inertia_limit', d].
    """
    if d.groupInertia is None or d.pdGroup_inertia_limit is None:
        return
    if d.group_node is None:
        return
    pss = d.process_source_sink
    if pss is None:
        return
    p_us = d.p_unitsize
    if p_us is None:
        return

    gi = d.groupInertia
    gn_in_gi = d.group_node.join(gi, on="g", how="inner")    # (g, n)

    # Slack variable.
    vq_dom = gi.join(d.dt, how="cross")     # (g, d, t)
    vq_in = m.add_var("vq_inertia", ("g", "d", "t"), vq_dom, lower=0.0)
    vars["vq_inertia"] = vq_in

    over = vq_dom

    lhs: dict = {}
    # Slack contribution: vq_inertia[g, d, t] * pdGroup[g, 'inertia_limit', d].
    lhs["slack"] = vq_in * d.pdGroup_inertia_limit

    v_flow = vars.get("v_flow")
    v_online_lin = vars.get("v_online_lin")
    v_online_int = vars.get("v_online_int")

    # process_online set used to switch v_online vs v_flow.
    proc_on = d.process_online
    proc_on_lin = d.process_online_linear
    proc_on_int = d.process_online_integer

    def _inertia_side(side: str, side_idx: pl.DataFrame | None,
                       const_param: Param | None) -> None:
        """side ∈ {"source", "sink"}.

        side_idx = (p, side) where inertia_constant exists.
        const_param = Param(("p", side), value).
        """
        if side_idx is None or side_idx.height == 0 or const_param is None:
            return
        # process__source__sink restricted to the inertia-bearing side.
        # Then filter by (g, side_value) ∈ group_node ∩ groupInertia.
        # Bring g in via join on side_value=node.
        pss_side = pss.join(side_idx, on=["p", side], how="inner")
        # add g: rename side → n for the join
        with_g = (pss_side.rename({side: "n"})
                          .join(gn_in_gi, on="n", how="inner")
                          .rename({"n": side}))   # (p, source, sink, g)
        if with_g.height == 0:
            return
        # Inertia-constant Param has dims (p, side). Multiply by unitsize.
        # Then attach to v_online or v_flow per process.
        coef_psd = const_param * p_us           # Param(p, side) -> (p, side)

        # Online cases: (a) linear UC, (b) integer UC.
        if proc_on_lin is not None and proc_on_lin.height > 0 and v_online_lin is not None:
            with_g_on = with_g.join(proc_on_lin, on="p", how="inner")
            if with_g_on.height > 0:
                # v_online_lin's natural dims are (p, d, t); Where on with_g_on
                # adds (source, sink, g).  Then * coef_psd (joined on (p, side)).
                term = Where(v_online_lin, with_g_on) * coef_psd
                lhs[f"online_lin_{side}"] = Sum(term, over=("p", "source", "sink"))
        if proc_on_int is not None and proc_on_int.height > 0 and v_online_int is not None:
            with_g_on = with_g.join(proc_on_int, on="p", how="inner")
            if with_g_on.height > 0:
                term = Where(v_online_int, with_g_on) * coef_psd
                lhs[f"online_int_{side}"] = Sum(term, over=("p", "source", "sink"))
        # Non-online case: v_flow.
        if v_flow is not None:
            with_g_off = with_g
            if proc_on is not None and proc_on.height > 0:
                with_g_off = with_g.join(proc_on, on="p", how="anti")
            if with_g_off.height > 0:
                term = Where(v_flow, with_g_off) * coef_psd
                lhs[f"flow_{side}"] = Sum(term, over=("p", "source", "sink"))

    _inertia_side("source",
                   getattr(d, "process_source_inertia", None),
                   getattr(d, "p_process_source_inertia_constant", None))
    _inertia_side("sink",
                   getattr(d, "process_sink_inertia", None),
                   getattr(d, "p_process_sink_inertia_constant", None))

    rhs = {"floor": d.pdGroup_inertia_limit}

    m.add_cstr(
        "inertia_constraint",
        over      = over,
        sense     = ">=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


# ── non_sync_constraint ───────────────────────────────────────────────────

def _add_non_sync(m, d, vars: dict) -> None:
    """non_sync_constraint (mod:4054-4091).

    Indexed on ``g in groupNonSync × (d, t) in dt``.  Sense: ``<=``.

    LHS (every term × inv_group_cap[g, d]):
      + Σ_{(p,source,sink) : (g, sink) ∈ group_node ∧ (p, sink) ∈
                              process__sink_nonSync ∧ (p, g) ∉
                              process__group_inside_group_nonSync}
          v_flow[p,source,sink,d,t] * unitsize * step_duration[d,t]
      + Σ_{(g, n) ∈ group_node} p_positive_inflow[n, d, t]
      − vq_non_synchronous[g, d, t] * step_duration[d, t]

    RHS:
      + Σ_outgoing-flow-from-group  × pdGroup[g, 'non_synchronous_limit'] × inv_group_cap.

    Outgoing flow has three parts (mirroring the .mod):
      noEff:   v_flow * unitsize * step_duration
      eff:     v_flow * unitsize * slope * sink_coef/source_coef * step_duration
      eff/min_load_eff section:  v_online * unitsize * step_duration * section
      − exogenous outflow: − Σ_n p_negative_inflow.

    NOTE: We omit the eff sink_coef/source_coef ratio (reflecting the
    .mod's process_unit-only branch) because flexpy's existing constraints
    already incorporate it via `p_slope` (slope encodes the ratio for
    indirect/multi-flow).  The non_sync constraint in the .mod only
    treats `process_unit`; for non-unit (connection) processes the ratio
    is 1 by definition.
    """
    if d.groupNonSync is None or d.pdGroup_non_synchronous_limit is None:
        return
    if d.group_node is None:
        return
    pss = d.process_source_sink
    if pss is None:
        return
    p_us = d.p_unitsize
    if p_us is None or d.p_step_duration is None:
        return

    gns      = d.groupNonSync
    invc     = d.p_inv_group_cap
    if invc is None:
        return
    gn_in_gns = d.group_node.join(gns, on="g", how="inner")     # (g, n)
    proc_inside = d.process_group_inside_nonSync                # (p, g) | None
    sink_ns     = d.process_sink_nonSync                        # (p, sink) | None

    # Slack variable.
    vq_dom = gns.join(d.dt, how="cross")
    vq_ns  = m.add_var("vq_non_synchronous", ("g", "d", "t"), vq_dom, lower=0.0)
    vars["vq_non_synchronous"] = vq_ns

    over = vq_dom

    # ── LHS ──────────────────────────────────────────────────────────────
    lhs: dict = {}
    v_flow = vars.get("v_flow")

    # − vq_non_synchronous * step_duration  (via inv_group_cap-cancellation,
    # this becomes − vq * step_duration).
    lhs["slack_neg"] = -(vq_ns * d.p_step_duration)

    # Incoming non-sync flows.
    if (sink_ns is not None and sink_ns.height > 0 and v_flow is not None):
        # (p, source, sink) in pss with (p, sink) ∈ process__sink_nonSync.
        pss_in = pss.join(sink_ns, on=["p", "sink"], how="inner")
        # (g, sink) in group_node ∩ groupNonSync (sink → n for the join).
        pss_in_g = (pss_in.rename({"sink": "n"})
                          .join(gn_in_gns, on="n", how="inner")
                          .rename({"n": "sink"}))
        # Drop (p, g) ∈ process__group_inside_group_nonSync.
        if proc_inside is not None and proc_inside.height > 0:
            pss_in_g = pss_in_g.join(proc_inside, on=["p", "g"], how="anti")
        if pss_in_g.height > 0:
            term = (Where(v_flow, pss_in_g) * p_us * d.p_step_duration * invc)
            lhs["incoming_nonSync"] = Sum(term, over=("p", "source", "sink"))

    # Exogenous positive inflow (assumed non-synchronous in the .mod).
    p_pos = d.p_positive_inflow
    if p_pos is not None and gn_in_gns.height > 0:
        joined = (gn_in_gns.join(p_pos.frame.rename({"value": "v"}),
                                  on="n", how="inner")
                            .join(invc.frame.rename({"value": "iv"}),
                                  on=["g", "d"], how="inner")
                            .with_columns(value=pl.col("v") * pl.col("iv"))
                            .group_by(["g", "d", "t"])
                            .agg(pl.col("value").sum())
                            .select("g", "d", "t", "value"))
        if joined.height > 0:
            lhs["exo_inflow"] = Param(("g", "d", "t"), joined)

    # ── RHS ──────────────────────────────────────────────────────────────
    pgns = d.pdGroup_non_synchronous_limit              # Param (g, d)
    rhs: dict = {}

    pss_eff   = d.process_source_sink_eff
    pss_noEff = d.process_source_sink_noEff
    p_slope   = d.p_slope

    # Outgoing flow (source ∈ group_node) with the source-not-inside guard.
    def _filter_out_inside(idx_g: pl.DataFrame) -> pl.DataFrame:
        if proc_inside is None or proc_inside.height == 0:
            return idx_g
        return idx_g.join(proc_inside, on=["p", "g"], how="anti")

    # Per‐tuple coef:  unitsize · step_duration · pdGroup_non_synchronous_limit · inv_group_cap.
    # We keep these chained as Params and let the engine broadcast at term time.
    out_factor = p_us * d.p_step_duration * pgns * invc

    if pss_noEff is not None and pss_noEff.height > 0 and v_flow is not None:
        out_idx_noEff = (pss_noEff.rename({"source": "n"})
                                  .join(gn_in_gns, on="n", how="inner")
                                  .rename({"n": "source"}))
        out_idx_noEff = _filter_out_inside(out_idx_noEff)
        if out_idx_noEff.height > 0:
            term = Where(v_flow, out_idx_noEff) * out_factor
            rhs["out_noEff"] = Sum(term, over=("p", "source", "sink"))
    if pss_eff is not None and pss_eff.height > 0 and v_flow is not None:
        out_idx_eff = (pss_eff.rename({"source": "n"})
                              .join(gn_in_gns, on="n", how="inner")
                              .rename({"n": "source"}))
        out_idx_eff = _filter_out_inside(out_idx_eff)
        if out_idx_eff.height > 0:
            if p_slope is not None:
                term = Where(v_flow, out_idx_eff) * p_slope * out_factor
            else:
                term = Where(v_flow, out_idx_eff) * out_factor
            rhs["out_eff"] = Sum(term, over=("p", "source", "sink"))
            # Section term for min_load_eff online.
            if (d.process_min_load_eff is not None
                    and d.process_min_load_eff.height > 0
                    and d.p_section is not None):
                mle_idx = out_idx_eff.join(d.process_min_load_eff,
                                            on="p", how="inner")
                if mle_idx.height > 0:
                    v_online_lin = vars.get("v_online_lin")
                    v_online_int = vars.get("v_online_int")
                    if v_online_lin is not None:
                        t_sec = Where(v_online_lin, mle_idx) * d.p_section * out_factor
                        rhs["out_section_lin"] = Sum(t_sec, over=("p", "source", "sink"))
                    if v_online_int is not None:
                        t_sec = Where(v_online_int, mle_idx) * d.p_section * out_factor
                        rhs["out_section_int"] = Sum(t_sec, over=("p", "source", "sink"))

    # Exogenous demand (-p_negative_inflow): pdGroup * inv_group_cap * Σ_n -p_neg.
    p_neg = d.p_negative_inflow
    if p_neg is not None and gn_in_gns.height > 0:
        joined = (gn_in_gns.join(p_neg.frame.rename({"value": "v"}),
                                  on="n", how="inner")
                            .join(invc.frame.rename({"value": "iv"}),
                                  on=["g", "d"], how="inner")
                            .join(pgns.frame.rename({"value": "lim"}),
                                  on=["g", "d"], how="inner")
                            .with_columns(value=-pl.col("v") * pl.col("iv")
                                                  * pl.col("lim"))
                            .group_by(["g", "d", "t"])
                            .agg(pl.col("value").sum())
                            .select("g", "d", "t", "value"))
        if joined.height > 0:
            rhs["exo_demand"] = Param(("g", "d", "t"), joined)

    m.add_cstr(
        "non_sync_constraint",
        over      = over,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


# ---------------------------------------------------------------------------
# Objective contribution

def add_objective_terms(m, d, vars: dict, op_factor) -> "Expr | None":
    """Build the slack-penalty contribution.  Returns ``None`` if no
    feature is active (caller must guard with ``or 0``).

    Terms (audit/objective_audit.md §9):

    9.1 Inertia slack:
      + Σ_{g,d,t} pdt_branch_weight · vq_inertia · pdGroup_inertia_limit
                  · pdGroup_penalty_inertia · step_duration · rp_cost_weight
                  · inflation_op / period_share
      The ``op_factor`` encodes ``step_duration · rp_cost_weight ·
      inflation_op / period_share``; the model.py caller folds
      ``pdt_branch_weight`` into ``op_factor`` when stochastics is active
      (A6 close), so this term inherits the per-(d,t) probability
      automatically.

    9.2 Non-sync slack:
      + Σ_{g,d,t} pdt_branch_weight · vq_non_synchronous · group_capacity_for_scaling
                  · pdGroup_penalty_non_synchronous · step_duration · rp_cost_weight
                  · inflation_op / period_share

    9.3 Capacity-margin slack (period-only, NO step_duration / rp_cost_weight /
    pdt_branch_weight):
      + Σ_{g,d} vq_capacity_margin · group_capacity_for_scaling
                · pdGroup_penalty_capacity_margin · inflation_op
      Asymmetry preserved per .mod — capacity_margin is a planning-level
      slack and intentionally un-weighted by pdt_branch_weight.  See
      audit/a6_stochastic_audit.md §5.
    """
    pieces: list[Expr] = []

    if "vq_inertia" in vars and d.pdGroup_inertia_limit is not None \
            and d.pdGroup_penalty_inertia is not None:
        pieces.append(Sum(
            vars["vq_inertia"]
              * d.pdGroup_inertia_limit
              * d.pdGroup_penalty_inertia
              * op_factor))

    if "vq_non_synchronous" in vars and d.p_group_capacity_for_scaling is not None \
            and d.pdGroup_penalty_non_synchronous is not None:
        pieces.append(Sum(
            vars["vq_non_synchronous"]
              * d.p_group_capacity_for_scaling
              * d.pdGroup_penalty_non_synchronous
              * op_factor))

    if "vq_capacity_margin" in vars and d.p_group_capacity_for_scaling is not None \
            and d.pdGroup_penalty_capacity_margin is not None \
            and d.p_inflation_op is not None:
        # Period-only term: no step_duration / rp_cost_weight / period_share.
        pieces.append(Sum(
            vars["vq_capacity_margin"]
              * d.p_group_capacity_for_scaling
              * d.pdGroup_penalty_capacity_margin
              * d.p_inflation_op))

    if not pieces:
        return None
    out = pieces[0]
    for p in pieces[1:]:
        out = out + p
    return out
