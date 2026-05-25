"""Cumulative / group-invest / min-max invest constraint pack.

This module is one of the ``_*.py`` feature packs that the merge agent
plugs into ``flextool.model.build_flextool``.  It is **self-contained**:
it imports only from ``polar_high`` and reads the ``FlexData`` (``d``)
+ the ``vars`` dict that the caller supplies — it never touches
``flextool/model.py`` or ``flextool/input.py``.

Constraints emitted (mod-name → polar_high row name):

  Per-entity invest / divest, period-scope (mirror of the existing
  ``maxInvest_entity_period`` / ``maxDivest_entity_period`` already in
  model.py — we add the **min** counterparts and the
  ``no_investment``-pin):

    * ``minInvest_entity_period``      — per (e, d) lower bound
    * ``minDivest_entity_period``      — per (e, d) lower bound
    * ``fix_v_invest_no_investment_eq``— pin v_invest = 0 on
                                         ``ed_invest_forbidden_no_investment``

  Per-entity invest / divest, total-scope (sister of the existing
  ``maxInvest_entity_total`` / ``maxDivest_entity_total``):

    * ``minInvest_entity_total``       — per (e, d) lower bound on
                                         Σ_d_invest v_invest
    * ``minDivest_entity_total``       — per e lower bound

  Cumulative (existing + invest − divest) capacity, per-(e, d):

    * ``maxCumulative_capacity``
    * ``minCumulative_capacity``

  Group-level invest / divest:

    * ``maxInvestGroup_entity_period`` / ``minInvestGroup_entity_period``
    * ``maxDivestGroup_entity_period`` / ``minDivestGroup_entity_period``
    * ``maxInvestGroup_entity_total``  / ``minInvestGroup_entity_total``
    * ``maxDivestGroup_entity_total``  / ``minDivestGroup_entity_total``
    * ``maxInvestGroup_entity_cumulative``
    * ``minInvestGroup_entity_cumulative``

  Group-level cumulative / instant flow bounds:

    * ``maxCumulative_flow_solve``  / ``minCumulative_flow_solve``
    * ``maxCumulative_flow_period`` / ``minCumulative_flow_period``
    * ``maxInstant_flow``           / ``minInstant_flow``

Variables are **not** declared here — the caller's ``v_invest_p``,
``v_invest_n``, ``v_divest_p``, ``v_divest_n``, ``v_flow``,
``v_online_linear``, ``v_online_integer`` are supplied in ``vars``.

Deferred (require feature stack outside this module's scope):

  * ``p_years_represented_d`` weighting on the cumulative-flow-solve
    LHS (mod uses ``p_rp_cost_weight * p_years_represented_d /
    complete_period_share_of_year`` per (d, t); we use the simpler
    ``step_duration`` integration since current scenarios run with
    ``p_years_represented_d = 1`` and a single representative year).
  * ``p_process_sink_flow_coefficient / p_process_source_flow_coefficient``
    factor on eff source-flow contributions to the cumulative/instant
    flow LHS — same gap as the existing ``nodeBalance_eq``.
  * ``p_entity_previously_invested_capacity`` and ``p_entity_divested``
    history terms on the per-entity-total and group-total constraints
    — only relevant for multi-solve handoffs.
  * ``inv_group_cap`` row scaling on the group-flow constraints.
  * The ``multi_year_wind_growth_cap`` per-period cap (no single mod
    constraint corresponds; appears to be an externally-preprocessed
    per-(e, d) ``ed_invest_max_period`` override that already lives
    on the existing ``maxInvest_entity_period`` constraint — no new
    constraint emitted here).
"""

from __future__ import annotations

import polars as pl
from polar_high import Sum, Where, Param
from polar_high.engine import Var


# ---------------------------------------------------------------------------
# Field requirements

# Lightweight feature gate: any of these fields, populated and non-empty,
# turns this pack on.  We don't insist on any single one — the pack
# emits whichever subset of constraints have data backing them.
_GATE_FIELDS: tuple[str, ...] = (
    # per-entity totals (max already in model.py — min added here)
    "e_invest_min_total", "e_divest_min_total",
    # per-entity period (max already in model.py — min added here)
    "ed_invest_min_period", "ed_divest_min_period",
    # no_investment pin
    "ed_invest_forbidden_no_investment",
    # cumulative capacity
    "ed_invest_cumulative", "ed_cumulative_max_capacity",
    "ed_cumulative_min_capacity",
    # group-invest
    "gd_invest_period", "gd_divest_period",
    "g_invest_total",   "g_divest_total",  "g_invest_cumulative",
    "group_entity",
    # group-flow
    "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow",      "pd_min_cumulative_flow",
    "gdt_maxInstantFlow",          "gdt_minInstantFlow",
    "group_process_node",
)


def has_feature(d) -> bool:
    """True iff the data carries any of the cumulative / group-invest /
    min-max-invest fields populated and non-empty."""
    for f in _GATE_FIELDS:
        v = getattr(d, f, None)
        if v is not None and getattr(v, "height", 0) > 0:
            return True
    return False


def load_data(*args, **kwargs):
    """No-op.  ``flextool/input.py`` is the canonical loader; this pack
    only consumes whatever fields the loader has populated.  Kept on the
    module API to satisfy the merge-agent contract."""
    return None


# ---------------------------------------------------------------------------
# Helpers

def _is_node(d, e_set: pl.DataFrame) -> pl.DataFrame:
    """Return the subset of ``e_set`` whose ``e`` column is a node."""
    if d.nodeState is None or d.nodeState.height == 0:
        return e_set.head(0)
    return e_set.filter(pl.col("e").is_in(d.nodeState["n"].unique()))


def _is_process(d, e_set: pl.DataFrame) -> pl.DataFrame:
    if d.process_source_sink is None or d.process_source_sink.height == 0:
        return e_set.head(0)
    return e_set.filter(pl.col("e").is_in(d.process_source_sink["p"].unique()))


def _us_p(d) -> Param | None:
    """Process-side unitsize (renamed from p → e for symmetric joins)."""
    if d.p_unitsize is None:
        return None
    return Param(("e",), d.p_unitsize.frame.rename({"p": "e"}))


def _us_n(d) -> Param | None:
    """Node-side unitsize (renamed from n → e)."""
    if d.p_state_unitsize is None:
        return None
    return Param(("e",), d.p_state_unitsize.frame.rename({"n": "e"}))


def _v_invest_at(v_invest, side: str):
    """Rename a v_invest variable's ``d`` dim to ``d_invest`` so it can
    participate in Σ_{d_invest} aggregations bound to a constraint
    indexed by an outer ``d``."""
    if v_invest is None:
        return None
    pri = "p" if side == "p" else "n"
    return Var(
        name=v_invest.name + "__cuminv_at",
        dims=(pri, "d_invest"),
        frame=v_invest.frame.rename({"d": "d_invest"}),
        lower=v_invest.lower, upper=v_invest.upper,
    )


def _eff_lhs_term(d, vars):
    """Build the eff-source-side LHS contribution that is shared by the
    cumulative- and instant-flow constraints.

    Returns an Expr over (g, d, t) (when called inside a flow_period /
    flow_solve constraint we further multiply by step_duration before
    binding).  When ``flow_from_n`` is missing or empty the term is
    None.  Mirrors the .mod's

        - sum_{(p,n,sink) in pss_eff} v_flow * unitsize * slope
          - section term (deferred — only relevant when min_load_eff
            entities are inside a group; we conservatively skip).
    """
    v_flow = vars.get("v_flow")
    if v_flow is None or d.process_source_sink_eff is None:
        return None
    if d.flow_from_n is None or d.flow_from_n.height == 0:
        return None
    if d.group_process_node is None or d.group_process_node.height == 0:
        return None

    # Restrict flow_from_n to eff partition + the (p, n) pairs the group
    # sees.  flow_from_n has columns (p, source, sink, n) where n is the
    # source node; group_process_node has (g, p, n).
    pss_eff = d.process_source_sink_eff
    flow_from_n_eff = (d.flow_from_n
                       .join(pss_eff, on=["p", "source", "sink"], how="inner")
                       .join(d.group_process_node, on=["p", "n"], how="inner"))
    if flow_from_n_eff.height == 0:
        return None
    # eff source-flow:  v_flow * unitsize * slope  (introduces g via the join)
    return -Sum(
        Where(v_flow * d.p_unitsize * d.p_slope, flow_from_n_eff),
        over=("p", "source", "sink", "n"),
    )


def _noEff_lhs_term(d, vars):
    """Source-noEff contribution: ``- Σ v_flow * unitsize`` over (p,
    source, sink, n) where (p, n, sink) ∈ pss_noEff and (g, p, n) ∈
    group_process_node."""
    v_flow = vars.get("v_flow")
    if v_flow is None or d.process_source_sink_noEff is None:
        return None
    if d.flow_from_n is None or d.flow_from_n.height == 0:
        return None
    if d.group_process_node is None or d.group_process_node.height == 0:
        return None
    flow_from_n_noeff = (d.flow_from_n
                         .join(d.process_source_sink_noEff,
                               on=["p", "source", "sink"], how="inner")
                         .join(d.group_process_node, on=["p", "n"], how="inner"))
    if flow_from_n_noeff.height == 0:
        return None
    return -Sum(
        Where(v_flow * d.p_unitsize, flow_from_n_noeff),
        over=("p", "source", "sink", "n"),
    )


def _sink_lhs_term(d, vars):
    """Sink-flow contribution: ``+ Σ v_flow * unitsize`` over (p,
    source, sink) where (p, source, n=sink) ∈ pss and (g, p, n) ∈
    group_process_node."""
    v_flow = vars.get("v_flow")
    if v_flow is None or d.process_source_sink is None:
        return None
    if d.flow_to_n is None or d.flow_to_n.height == 0:
        return None
    if d.group_process_node is None or d.group_process_node.height == 0:
        return None
    flow_to_n_grp = d.flow_to_n.join(
        d.group_process_node, on=["p", "n"], how="inner")
    if flow_to_n_grp.height == 0:
        return None
    return Sum(
        Where(v_flow * d.p_unitsize, flow_to_n_grp),
        over=("p", "source", "sink", "n"),
    )


# ---------------------------------------------------------------------------
# Constraint emission

def add_constraints(m, d, vars: dict) -> None:
    """Emit every constraint family this pack handles, gated row-by-row
    on the underlying ``FlexData`` field being populated."""

    v_invest_p = vars.get("v_invest_p")
    v_invest_n = vars.get("v_invest_n")
    v_divest_p = vars.get("v_divest_p")
    v_divest_n = vars.get("v_divest_n")

    has_inv_p = v_invest_p is not None
    has_inv_n = v_invest_n is not None
    has_div_p = v_divest_p is not None
    has_div_n = v_divest_n is not None

    # ─── fix_v_invest_no_investment_eq ──────────────────────────────
    # Pin v_invest[e, d] == 0 on (e, d) ∈ ed_invest_forbidden_no_investment.
    # Mod: flextool.mod ``fix_v_invest_no_investment_eq``.
    forbid = getattr(d, "ed_invest_forbidden_no_investment", None)
    if forbid is not None and forbid.height > 0:
        if has_inv_p:
            # ``forbid["e"]`` is the entity-union Enum; after renaming to
            # ``p`` we must align it with ``v_invest_p``'s narrower
            # process-only ``p`` Enum before joining (Phase 4 Enum
            # cross-vocab guard — never cast back to Utf8).
            from ._axis_enums import align_join_dtypes
            f_p = forbid.rename({"e": "p"})
            inv_p_idx = v_invest_p.frame.select("p", "d")
            f_p, inv_p_idx = align_join_dtypes(f_p, inv_p_idx, ("p", "d"))
            f_p = f_p.join(inv_p_idx, on=["p", "d"], how="inner")
            if f_p.height > 0:
                m.add_cstr(
                    "fix_v_invest_no_investment_eq_p",
                    over      = f_p,
                    sense     = "==",
                    lhs_terms = {"invest": v_invest_p},
                    rhs_terms = {"zero":   0.0},
                )
        if has_inv_n:
            from ._axis_enums import align_join_dtypes
            f_n = forbid.rename({"e": "n"})
            inv_n_idx = v_invest_n.frame.select("n", "d")
            f_n, inv_n_idx = align_join_dtypes(f_n, inv_n_idx, ("n", "d"))
            f_n = f_n.join(inv_n_idx, on=["n", "d"], how="inner")
            if f_n.height > 0:
                m.add_cstr(
                    "fix_v_invest_no_investment_eq_n",
                    over      = f_n,
                    sense     = "==",
                    lhs_terms = {"invest": v_invest_n},
                    rhs_terms = {"zero":   0.0},
                )

    # ─── minInvest_entity_period / minDivest_entity_period ───────────
    # Mirror of the existing maxInvest/maxDivest_entity_period in
    # model.py:402-455 but with sense `>=` and a different RHS param.
    _emit_invest_period_minmax(m, d, vars, kind="invest", sense=">=")
    _emit_invest_period_minmax(m, d, vars, kind="divest", sense=">=")

    # ─── minInvest_entity_total / minDivest_entity_total ─────────────
    # Sister of model.py:459-531 maxInvest/maxDivest_entity_total.
    _emit_invest_total_minmax(m, d, vars, kind="invest", sense=">=")
    _emit_invest_total_minmax(m, d, vars, kind="divest", sense=">=")

    # ─── maxCumulative_capacity / minCumulative_capacity ─────────────
    # LHS:  + p_entity_all_existing[e,d]
    #       + Σ_{d_invest} v_invest[e, d_invest] * unitsize
    #       - v_divest[e, d] * unitsize  (if (e, d) ∈ ed_divest)
    # RHS:  ed_cumulative_{max,min}_capacity[e, d]
    _emit_cumulative_capacity(m, d, vars, sense="<=")
    _emit_cumulative_capacity(m, d, vars, sense=">=")

    # ─── Group invest / divest, period scope ─────────────────────────
    _emit_group_invest_period(m, d, vars, kind="invest", sense="<=")
    _emit_group_invest_period(m, d, vars, kind="invest", sense=">=")
    _emit_group_invest_period(m, d, vars, kind="divest", sense="<=")
    _emit_group_invest_period(m, d, vars, kind="divest", sense=">=")

    # ─── Group invest / divest, total scope ──────────────────────────
    _emit_group_invest_total(m, d, vars, kind="invest", sense="<=")
    _emit_group_invest_total(m, d, vars, kind="invest", sense=">=")
    _emit_group_invest_total(m, d, vars, kind="divest", sense="<=")
    _emit_group_invest_total(m, d, vars, kind="divest", sense=">=")

    # ─── Group invest, cumulative (max + min) ────────────────────────
    _emit_group_invest_cumulative(m, d, vars, sense="<=")
    _emit_group_invest_cumulative(m, d, vars, sense=">=")

    # ─── Group cumulative-flow / instant-flow ────────────────────────
    _emit_cumulative_flow_solve(m, d, vars, sense="<=")
    _emit_cumulative_flow_solve(m, d, vars, sense=">=")
    _emit_cumulative_flow_period(m, d, vars, sense="<=")
    _emit_cumulative_flow_period(m, d, vars, sense=">=")
    _emit_instant_flow(m, d, vars, sense="<=")
    _emit_instant_flow(m, d, vars, sense=">=")


def add_objective_terms(m, d, vars: dict, op_factor) -> None:
    """No objective contribution — every constraint here is a structural
    (in)equality, not a cost term."""
    return None


# ---------------------------------------------------------------------------
# Per-entity period min/max

def _emit_invest_period_minmax(m, d, vars: dict,
                                kind: str, sense: str) -> None:
    """Emit the min variant of {invest|divest}_entity_period."""
    # Pull the right (set, max-param, min-param) trio.
    if kind == "invest":
        idx_set   = getattr(d, "ed_invest_period_set", None)
        cap_param = getattr(d, "ed_invest_min_period", None)
        v_p = vars.get("v_invest_p"); v_n = vars.get("v_invest_n")
    else:  # divest
        idx_set   = getattr(d, "ed_divest_period_set", None)
        cap_param = getattr(d, "ed_divest_min_period", None)
        v_p = vars.get("v_divest_p"); v_n = vars.get("v_divest_n")
    if idx_set is None or idx_set.height == 0 or cap_param is None:
        return
    suffix = f"min{kind.capitalize()}_entity_period"

    # Process side
    if v_p is not None and d.p_unitsize is not None:
        sub = idx_set.rename({"e": "p"}).join(
            v_p.frame.select("p", "d"), on=["p", "d"], how="inner")
        if sub.height > 0:
            cap = Param(("p", "d"), cap_param.frame.rename({"e": "p"}))
            m.add_cstr(
                f"{suffix}_p",
                over      = sub,
                sense     = sense,
                lhs_terms = {kind: v_p * d.p_unitsize},
                rhs_terms = {"cap": cap},
            )
    # Node side
    if v_n is not None and d.p_state_unitsize is not None:
        sub = idx_set.rename({"e": "n"}).join(
            v_n.frame.select("n", "d"), on=["n", "d"], how="inner")
        if sub.height > 0:
            cap = Param(("n", "d"), cap_param.frame.rename({"e": "n"}))
            us  = Param(("n",),     d.p_state_unitsize.frame)
            m.add_cstr(
                f"{suffix}_n",
                over      = sub,
                sense     = sense,
                lhs_terms = {kind: v_n * us},
                rhs_terms = {"cap": cap},
            )


# ---------------------------------------------------------------------------
# Per-entity total min/max

def _emit_invest_total_minmax(m, d, vars: dict,
                               kind: str, sense: str) -> None:
    """Emit ``minInvest_entity_total`` / ``minDivest_entity_total``.

    For invest: indexed by (e, d) — Σ_{d_invest in edd_invest} v_invest
    summed; vs ``e_invest_min_total[e]``.
    For divest: indexed by e only — Σ_d v_divest summed; vs
    ``e_divest_min_total[e]``.
    """
    if kind == "invest":
        e_set     = getattr(d, "e_invest_total", None)
        cap_param = getattr(d, "e_invest_min_total", None)
        edd_set   = getattr(d, "edd_invest_set", None)
        v_p = vars.get("v_invest_p"); v_n = vars.get("v_invest_n")
    else:
        e_set     = getattr(d, "e_divest_total", None)
        cap_param = getattr(d, "e_divest_min_total", None)
        edd_set   = None
        v_p = vars.get("v_divest_p"); v_n = vars.get("v_divest_n")
    if e_set is None or e_set.height == 0 or cap_param is None:
        return
    cstr_name = f"min{kind.capitalize()}_entity_total"

    # Process side
    if v_p is not None and d.p_unitsize is not None:
        e_p = _is_process(d, e_set).rename({"e": "p"})
        if e_p.height > 0:
            cap_p = Param(("p",), cap_param.frame.rename({"e": "p"}))
            if kind == "invest":
                # over = (e_p × period_invest), as in maxInvest_entity_total.
                # We don't have period_invest as a separate frame, but
                # the canonical way is to use edd_invest's outer d column.
                if edd_set is None:
                    return
                edd_p = (edd_set.rename({"e": "p"}).join(
                    e_p, on="p", how="inner")
                    .filter(pl.col("p").is_in(v_p.frame["p"].unique())))
                if edd_p.height == 0:
                    return
                # outer index = unique (p, d) from edd_p
                outer = edd_p.select("p", "d").unique()
                v_inv_at = Var(
                    name=v_p.name + "__cuminv_total_at",
                    dims=("p", "d_invest"),
                    frame=v_p.frame.rename({"d": "d_invest"}),
                    lower=v_p.lower, upper=v_p.upper,
                )
                inv_sum = Sum(Where(v_inv_at * d.p_unitsize, edd_p),
                              over=("d_invest",))
                m.add_cstr(
                    f"{cstr_name}_p",
                    over      = outer,
                    sense     = sense,
                    lhs_terms = {"invest_total": inv_sum},
                    rhs_terms = {"cap":          cap_p},
                )
            else:  # divest — sum over d
                m.add_cstr(
                    f"{cstr_name}_p",
                    over      = e_p,
                    sense     = sense,
                    lhs_terms = {"divest_total":
                        Sum(Where(v_p * d.p_unitsize, e_p), over=("d",))},
                    rhs_terms = {"cap": cap_p},
                )

    # Node side
    if v_n is not None and d.p_state_unitsize is not None:
        e_n = _is_node(d, e_set).rename({"e": "n"})
        if e_n.height > 0:
            cap_n = Param(("n",), cap_param.frame.rename({"e": "n"}))
            us_n  = Param(("n",), d.p_state_unitsize.frame)
            if kind == "invest":
                if edd_set is None:
                    return
                edd_n = (edd_set.rename({"e": "n"}).join(
                    e_n, on="n", how="inner")
                    .filter(pl.col("n").is_in(v_n.frame["n"].unique())))
                if edd_n.height == 0:
                    return
                outer = edd_n.select("n", "d").unique()
                v_inv_at = Var(
                    name=v_n.name + "__cuminv_total_at",
                    dims=("n", "d_invest"),
                    frame=v_n.frame.rename({"d": "d_invest"}),
                    lower=v_n.lower, upper=v_n.upper,
                )
                inv_sum = Sum(Where(v_inv_at * us_n, edd_n),
                              over=("d_invest",))
                m.add_cstr(
                    f"{cstr_name}_n",
                    over      = outer,
                    sense     = sense,
                    lhs_terms = {"invest_total": inv_sum},
                    rhs_terms = {"cap":          cap_n},
                )
            else:
                m.add_cstr(
                    f"{cstr_name}_n",
                    over      = e_n,
                    sense     = sense,
                    lhs_terms = {"divest_total":
                        Sum(Where(v_n * us_n, e_n), over=("d",))},
                    rhs_terms = {"cap": cap_n},
                )


# ---------------------------------------------------------------------------
# Cumulative capacity

def _emit_cumulative_capacity(m, d, vars: dict, sense: str) -> None:
    """``maxCumulative_capacity`` / ``minCumulative_capacity``.

        + p_entity_all_existing[e, d]
        + Σ_{d_invest : (e,d_invest,d) in edd_invest} v_invest * unitsize
        - (if (e, d) in ed_divest then v_divest * unitsize)
        <=  ed_cumulative_max_capacity[e, d]   (sense='<=')
        >=  ed_cumulative_min_capacity[e, d]   (sense='>=')
    """
    idx = getattr(d, "ed_invest_cumulative", None)
    if idx is None or idx.height == 0:
        return
    cap_field = ("ed_cumulative_max_capacity" if sense == "<="
                 else "ed_cumulative_min_capacity")
    cap_param = getattr(d, cap_field, None)
    if cap_param is None:
        return
    name = "maxCumulative_capacity" if sense == "<=" else "minCumulative_capacity"
    existing_param = getattr(d, "p_entity_all_existing", None)
    edd_set = getattr(d, "edd_invest_set", None)
    if edd_set is None:
        return

    v_inv_p = vars.get("v_invest_p"); v_inv_n = vars.get("v_invest_n")
    v_div_p = vars.get("v_divest_p"); v_div_n = vars.get("v_divest_n")

    # Process side
    if v_inv_p is not None and d.p_unitsize is not None:
        # ``idx``'s ``e`` carries the entity-union Enum (entities cover
        # processes + nodes); ``v_inv_p.frame["p"]`` carries the
        # process-only Enum.  Cast the renamed ``p`` column to the
        # narrower process Enum so the ``is_in`` membership test
        # composes against a matching dtype.  Per the entity-union axis
        # convention: cast at the boundary, never let polars sort it
        # out via List(Enum).
        idx_p = idx.rename({"e": "p"})
        _p_dtype = v_inv_p.frame.schema["p"]
        if idx_p.schema["p"] != _p_dtype:
            idx_p = idx_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
        idx_p = idx_p.filter(
            pl.col("p").is_in(v_inv_p.frame["p"].unique()))
        if idx_p.height > 0:
            edd_p = edd_set.rename({"e": "p"})
            # Reconcile the renamed entity-union ``p`` (from ``edd_set``)
            # against ``idx_p``'s narrower process-only ``p`` Enum.
            from ._axis_enums import align_join_dtypes
            edd_p, idx_p_aligned = align_join_dtypes(
                edd_p, idx_p, ("p",),
            )
            edd_p = edd_p.join(idx_p_aligned, on=["p", "d"], how="inner")
            v_inv_at = Var(
                name=v_inv_p.name + "__cumcap_at",
                dims=("p", "d_invest"),
                frame=v_inv_p.frame.rename({"d": "d_invest"}),
                lower=v_inv_p.lower, upper=v_inv_p.upper,
            )
            invest_term = Sum(Where(v_inv_at * d.p_unitsize, edd_p),
                              over=("d_invest",))
            lhs: dict = {"invest_cum": invest_term}
            if v_div_p is not None:
                # Only subtract divest where (p, d) ∈ ed_divest_set.
                # Filter to v_divest_p's own frame which is exactly
                # pd_divest_set.
                lhs["divest"] = -Where(v_div_p * d.p_unitsize, idx_p)
            cap_p = Param(("p", "d"), cap_param.frame.rename({"e": "p"}))
            rhs: dict = {"cap": cap_p}
            if existing_param is not None:
                exist_p = Param(("p", "d"),
                    existing_param.frame.rename({"e": "p"}))
                # Move existing to the RHS as -existing.
                rhs["minus_existing"] = -exist_p
            m.add_cstr(
                f"{name}_p",
                over      = idx_p,
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = rhs,
            )

    # Node side
    if v_inv_n is not None and d.p_state_unitsize is not None:
        # Mirror of the process-side cast.  ``idx``'s ``e`` is the
        # entity-union Enum; ``v_inv_n.frame["n"]`` is the node-only
        # Enum.  Cast at the boundary before ``is_in``.
        idx_n = idx.rename({"e": "n"})
        _n_dtype = v_inv_n.frame.schema["n"]
        if idx_n.schema["n"] != _n_dtype:
            idx_n = idx_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
        idx_n = idx_n.filter(
            pl.col("n").is_in(v_inv_n.frame["n"].unique()))
        if idx_n.height > 0:
            us_n = Param(("n",), d.p_state_unitsize.frame)
            edd_n = edd_set.rename({"e": "n"})
            from ._axis_enums import align_join_dtypes
            edd_n, idx_n_aligned = align_join_dtypes(
                edd_n, idx_n, ("n",),
            )
            edd_n = edd_n.join(idx_n_aligned, on=["n", "d"], how="inner")
            v_inv_at = Var(
                name=v_inv_n.name + "__cumcap_at",
                dims=("n", "d_invest"),
                frame=v_inv_n.frame.rename({"d": "d_invest"}),
                lower=v_inv_n.lower, upper=v_inv_n.upper,
            )
            invest_term = Sum(Where(v_inv_at * us_n, edd_n),
                              over=("d_invest",))
            lhs: dict = {"invest_cum": invest_term}
            if v_div_n is not None:
                lhs["divest"] = -Where(v_div_n * us_n, idx_n)
            cap_n = Param(("n", "d"), cap_param.frame.rename({"e": "n"}))
            rhs: dict = {"cap": cap_n}
            if existing_param is not None:
                exist_n = Param(("n", "d"),
                    existing_param.frame.rename({"e": "n"}))
                rhs["minus_existing"] = -exist_n
            m.add_cstr(
                f"{name}_n",
                over      = idx_n,
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = rhs,
            )


# ---------------------------------------------------------------------------
# Group invest / divest — period scope

def _emit_group_invest_period(m, d, vars: dict,
                               kind: str, sense: str) -> None:
    """Σ_{(g,e) in group_entity : (e,d) in ed_{kind}}
         v_{kind}[e, d] * unitsize  <sense>  pdGroup[g, ?, d]"""
    if kind == "invest":
        gd_idx = getattr(d, "gd_invest_period", None)
        cap_field = ("p_group_invest_max_period" if sense == "<="
                     else "p_group_invest_min_period")
        v_p = vars.get("v_invest_p"); v_n = vars.get("v_invest_n")
    else:
        gd_idx = getattr(d, "gd_divest_period", None)
        cap_field = ("p_group_retire_max_period" if sense == "<="
                     else "p_group_retire_min_period")
        v_p = vars.get("v_divest_p"); v_n = vars.get("v_divest_n")
    cap_param = getattr(d, cap_field, None)
    group_entity = getattr(d, "group_entity", None)
    if (gd_idx is None or gd_idx.height == 0
            or cap_param is None
            or group_entity is None or group_entity.height == 0):
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}{kind.capitalize()}Group_entity_period"

    # Process branch
    if v_p is not None and d.p_unitsize is not None:
        ge_p = group_entity.rename({"e": "p"}).filter(
            pl.col("p").is_in(v_p.frame["p"].unique()))
        if ge_p.height > 0:
            # join group_entity (g, p) with v_p frame (p, d), filter to
            # gd_idx (g, d)
            joined = (ge_p.join(v_p.frame.select("p", "d"), on="p", how="inner")
                          .join(gd_idx, on=["g", "d"], how="inner"))
            if joined.height > 0:
                lhs = Sum(
                    Where(v_p * d.p_unitsize, joined.select("g", "p", "d")),
                    over=("p",),
                )
                m.add_cstr(
                    f"{name}_p",
                    over      = gd_idx,
                    sense     = sense,
                    lhs_terms = {kind: lhs},
                    rhs_terms = {"cap": cap_param},
                )
    # Node branch
    if v_n is not None and d.p_state_unitsize is not None:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        ge_n = group_entity.rename({"e": "n"}).filter(
            pl.col("n").is_in(v_n.frame["n"].unique()))
        if ge_n.height > 0:
            joined = (ge_n.join(v_n.frame.select("n", "d"), on="n", how="inner")
                          .join(gd_idx, on=["g", "d"], how="inner"))
            if joined.height > 0:
                lhs = Sum(
                    Where(v_n * us_n, joined.select("g", "n", "d")),
                    over=("n",),
                )
                m.add_cstr(
                    f"{name}_n",
                    over      = gd_idx,
                    sense     = sense,
                    lhs_terms = {kind: lhs},
                    rhs_terms = {"cap": cap_param},
                )


# ---------------------------------------------------------------------------
# Group invest / divest — total scope

def _emit_group_invest_total(m, d, vars: dict,
                              kind: str, sense: str) -> None:
    """Mod:

      maxInvestGroup_entity_total {g in g_invest_total, d in period_invest} :
        + Σ_{(g,e), d_inv : (e,d_inv,d) in edd_invest}
              v_invest[e, d_inv] * unitsize
        + Σ_{(g,e)} p_entity_previously_invested_capacity[e, d]   (deferred)
        <=  p_group[g, 'invest_max_total']                       (scalar over g)

    For divest the index is just g (no per-d outer) and LHS sums over d.
    """
    group_entity = getattr(d, "group_entity", None)
    if group_entity is None or group_entity.height == 0:
        return
    if kind == "invest":
        g_set     = getattr(d, "g_invest_total", None)
        cap_field = ("p_group_invest_max_total" if sense == "<="
                     else "p_group_invest_min_total")
        edd_set   = getattr(d, "edd_invest_set", None)
        v_p = vars.get("v_invest_p"); v_n = vars.get("v_invest_n")
    else:
        g_set     = getattr(d, "g_divest_total", None)
        cap_field = ("p_group_retire_max_total" if sense == "<="
                     else "p_group_retire_min_total")
        edd_set   = None
        v_p = vars.get("v_divest_p"); v_n = vars.get("v_divest_n")
    cap_param = getattr(d, cap_field, None)
    if g_set is None or g_set.height == 0 or cap_param is None:
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}{kind.capitalize()}Group_entity_total"

    if kind == "invest":
        # outer = (g × d) — use periods where ANY (e, d_inv, d) ∈ edd_invest
        if edd_set is None:
            return
        # outer (g, d): for each g, every d that appears in edd for some
        # (g, e) in group_entity.
        ge = group_entity.rename({"e": "_e"}).filter(
            pl.col("g").is_in(g_set["g"].unique()))
        # Build outer (g, d) frame from edd_set ∩ group_entity
        edd_for_g = (ge.join(edd_set.rename({"e": "_e"}),
                              on="_e", how="inner")
                       .select("g", "d").unique())
        if edd_for_g.height == 0:
            return

        # Process branch
        if v_p is not None and d.p_unitsize is not None:
            # Phase 4.8h: cross-Enum is_in (e-axis vs p-axis vocab); cast
            # at the boundary, never let polars sort it out via List(Enum).
            _p_dtype = v_p.frame.schema["p"]
            ge_p = group_entity.rename({"e": "p"})
            if ge_p.schema["p"] != _p_dtype:
                ge_p = ge_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
            ge_p = ge_p.filter(pl.col("p").is_in(v_p.frame["p"].unique()))
            edd_p = edd_set.rename({"e": "p"})
            if edd_p.schema["p"] != _p_dtype:
                edd_p = edd_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
            edd_p = edd_p.filter(pl.col("p").is_in(v_p.frame["p"].unique()))
            joined = ge_p.join(edd_p, on="p", how="inner")
            if joined.height > 0:
                v_inv_at = Var(
                    name=v_p.name + "__grp_total_at",
                    dims=("p", "d_invest"),
                    frame=v_p.frame.rename({"d": "d_invest"}),
                    lower=v_p.lower, upper=v_p.upper,
                )
                inv_sum = Sum(
                    Where(v_inv_at * d.p_unitsize, joined),
                    over=("p", "d_invest"),
                )
                m.add_cstr(
                    f"{name}_p",
                    over      = edd_for_g,
                    sense     = sense,
                    lhs_terms = {"invest_grp": inv_sum},
                    rhs_terms = {"cap": cap_param},
                )
        # Node branch
        if v_n is not None and d.p_state_unitsize is not None:
            us_n = Param(("n",), d.p_state_unitsize.frame)
            # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); cast
            # at the boundary, never let polars sort it out via List(Enum).
            _n_dtype = v_n.frame.schema["n"]
            ge_n = group_entity.rename({"e": "n"})
            if ge_n.schema["n"] != _n_dtype:
                ge_n = ge_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
            ge_n = ge_n.filter(pl.col("n").is_in(v_n.frame["n"].unique()))
            edd_n = edd_set.rename({"e": "n"})
            if edd_n.schema["n"] != _n_dtype:
                edd_n = edd_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
            edd_n = edd_n.filter(pl.col("n").is_in(v_n.frame["n"].unique()))
            joined = ge_n.join(edd_n, on="n", how="inner")
            if joined.height > 0:
                v_inv_at = Var(
                    name=v_n.name + "__grp_total_at",
                    dims=("n", "d_invest"),
                    frame=v_n.frame.rename({"d": "d_invest"}),
                    lower=v_n.lower, upper=v_n.upper,
                )
                inv_sum = Sum(
                    Where(v_inv_at * us_n, joined),
                    over=("n", "d_invest"),
                )
                m.add_cstr(
                    f"{name}_n",
                    over      = edd_for_g,
                    sense     = sense,
                    lhs_terms = {"invest_grp": inv_sum},
                    rhs_terms = {"cap": cap_param},
                )
    else:
        # divest: outer = g, sum over (e, d) ∈ ed_divest where (g, e) ∈ group_entity
        outer = g_set.select("g")
        # Process branch
        if v_p is not None and d.p_unitsize is not None:
            # Phase 4.8h: cross-Enum is_in (e-axis vs p-axis vocab); cast
            # at the boundary, never let polars sort it out via List(Enum).
            _p_dtype = v_p.frame.schema["p"]
            ge_p = group_entity.rename({"e": "p"})
            if ge_p.schema["p"] != _p_dtype:
                ge_p = ge_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
            ge_p = ge_p.filter(
                pl.col("p").is_in(v_p.frame["p"].unique())) \
                .filter(pl.col("g").is_in(g_set["g"].unique()))
            if ge_p.height > 0:
                lhs = Sum(
                    Where(v_p * d.p_unitsize, ge_p),
                    over=("p", "d"),
                )
                m.add_cstr(
                    f"{name}_p",
                    over      = outer,
                    sense     = sense,
                    lhs_terms = {"divest_grp": lhs},
                    rhs_terms = {"cap": cap_param},
                )
        # Node branch
        if v_n is not None and d.p_state_unitsize is not None:
            us_n = Param(("n",), d.p_state_unitsize.frame)
            # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); cast
            # at the boundary, never let polars sort it out via List(Enum).
            _n_dtype = v_n.frame.schema["n"]
            ge_n = group_entity.rename({"e": "n"})
            if ge_n.schema["n"] != _n_dtype:
                ge_n = ge_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
            ge_n = ge_n.filter(
                pl.col("n").is_in(v_n.frame["n"].unique())) \
                .filter(pl.col("g").is_in(g_set["g"].unique()))
            if ge_n.height > 0:
                lhs = Sum(
                    Where(v_n * us_n, ge_n),
                    over=("n", "d"),
                )
                m.add_cstr(
                    f"{name}_n",
                    over      = outer,
                    sense     = sense,
                    lhs_terms = {"divest_grp": lhs},
                    rhs_terms = {"cap": cap_param},
                )


# ---------------------------------------------------------------------------
# Group invest cumulative

def _emit_group_invest_cumulative(m, d, vars: dict, sense: str) -> None:
    """maxInvestGroup_entity_cumulative / min variant.

    LHS:
      + Σ_{(g,e), d_inv : (e,d_inv,d) in edd_invest} v_invest * unitsize
      + Σ_{(g,e)} p_entity_previously_invested_capacity[e, d]   (deferred)
      + Σ_{(g,e)} p_entity_all_existing[e, d]
      [- divest terms for the min variant — deferred per audit note]
    RHS:
      + p_group[g, 'invest_max_cumulative']  (or min_cumulative)
    """
    group_entity = getattr(d, "group_entity", None)
    g_set        = getattr(d, "g_invest_cumulative", None)
    edd_set      = getattr(d, "edd_invest_set", None)
    if (group_entity is None or group_entity.height == 0
            or g_set is None or g_set.height == 0
            or edd_set is None):
        return
    cap_field = ("p_group_invest_max_cumulative" if sense == "<="
                 else "p_group_invest_min_cumulative")
    cap_param = getattr(d, cap_field, None)
    if cap_param is None:
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}InvestGroup_entity_cumulative"

    # Build outer (g, d): every (g, d) where the group has a member with
    # an edd_invest row at that d.
    ge = group_entity.rename({"e": "_e"}).filter(
        pl.col("g").is_in(g_set["g"].unique()))
    outer = (ge.join(edd_set.rename({"e": "_e"}), on="_e", how="inner")
               .select("g", "d").unique())
    if outer.height == 0:
        return

    # LHS pieces: invest sum (process + node) and existing sum.
    v_inv_p = vars.get("v_invest_p"); v_inv_n = vars.get("v_invest_n")
    existing = getattr(d, "p_entity_all_existing", None)

    lhs_terms: dict = {}
    if v_inv_p is not None and d.p_unitsize is not None:
        # Phase 4.8h: cross-Enum is_in (e-axis vs p-axis vocab); cast
        # at the boundary, never let polars sort it out via List(Enum).
        _p_dtype = v_inv_p.frame.schema["p"]
        ge_p = group_entity.rename({"e": "p"})
        if ge_p.schema["p"] != _p_dtype:
            ge_p = ge_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
        ge_p = ge_p.filter(
            pl.col("p").is_in(v_inv_p.frame["p"].unique())).filter(
            pl.col("g").is_in(g_set["g"].unique()))
        edd_p = edd_set.rename({"e": "p"})
        if edd_p.schema["p"] != _p_dtype:
            edd_p = edd_p.with_columns(pl.col("p").cast(_p_dtype, strict=False))
        edd_p = edd_p.filter(
            pl.col("p").is_in(v_inv_p.frame["p"].unique()))
        joined = ge_p.join(edd_p, on="p", how="inner")
        if joined.height > 0:
            v_inv_at = Var(
                name=v_inv_p.name + "__grp_cum_at",
                dims=("p", "d_invest"),
                frame=v_inv_p.frame.rename({"d": "d_invest"}),
                lower=v_inv_p.lower, upper=v_inv_p.upper,
            )
            lhs_terms["invest_p"] = Sum(
                Where(v_inv_at * d.p_unitsize, joined),
                over=("p", "d_invest"),
            )
    if v_inv_n is not None and d.p_state_unitsize is not None:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); cast
        # at the boundary, never let polars sort it out via List(Enum).
        _n_dtype = v_inv_n.frame.schema["n"]
        ge_n = group_entity.rename({"e": "n"})
        if ge_n.schema["n"] != _n_dtype:
            ge_n = ge_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
        ge_n = ge_n.filter(
            pl.col("n").is_in(v_inv_n.frame["n"].unique())).filter(
            pl.col("g").is_in(g_set["g"].unique()))
        edd_n = edd_set.rename({"e": "n"})
        if edd_n.schema["n"] != _n_dtype:
            edd_n = edd_n.with_columns(pl.col("n").cast(_n_dtype, strict=False))
        edd_n = edd_n.filter(
            pl.col("n").is_in(v_inv_n.frame["n"].unique()))
        joined = ge_n.join(edd_n, on="n", how="inner")
        if joined.height > 0:
            v_inv_at = Var(
                name=v_inv_n.name + "__grp_cum_at",
                dims=("n", "d_invest"),
                frame=v_inv_n.frame.rename({"d": "d_invest"}),
                lower=v_inv_n.lower, upper=v_inv_n.upper,
            )
            lhs_terms["invest_n"] = Sum(
                Where(v_inv_at * us_n, joined),
                over=("n", "d_invest"),
            )

    rhs_terms: dict = {"cap": cap_param}
    # existing(g, d) is a known constant — push to RHS with sign flip.
    if existing is not None:
        # existing has dims (e, d); join to group_entity, sum over e
        ex_frame = (existing.frame
                    .join(group_entity, on="e", how="inner")
                    .group_by(["g", "d"]).agg(pl.col("value").sum())
                    .filter(pl.col("g").is_in(g_set["g"].unique())))
        if ex_frame.height > 0:
            ex_param = Param(("g", "d"), ex_frame.select("g", "d", "value"))
            rhs_terms["minus_existing"] = -ex_param

    if lhs_terms:
        m.add_cstr(
            name,
            over      = outer,
            sense     = sense,
            lhs_terms = lhs_terms,
            rhs_terms = rhs_terms,
        )


# ---------------------------------------------------------------------------
# Group cumulative-flow (whole solve)

def _flow_lhs(d, vars):
    """Build the (sink + eff_source + noEff_source) LHS for cumulative
    / instant flow constraints, returning a single Expr or None.

    Open dims: (g, d, t) when the v_flow Var has dims (p, source, sink,
    d, t).  Caller multiplies by step_duration & sums over (d, t) for
    cumulative variants, or just binds to the (g, d, t) frame for
    instant.
    """
    parts = []
    for fn in (_sink_lhs_term, _eff_lhs_term, _noEff_lhs_term):
        term = fn(d, vars)
        if term is not None:
            parts.append(term)
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def _emit_cumulative_flow_solve(m, d, vars: dict, sense: str) -> None:
    """Single-row constraint per group g in g_max_cumulative_flow_solve
    (resp. g_min_cumulative_flow_solve).  LHS is integrated over the
    whole horizon (Σ_{(d,t)} step_duration · flow_term)."""
    cap_field = ("p_group_max_cumulative_flow" if sense == "<="
                 else "p_group_min_cumulative_flow")
    cap_param = getattr(d, cap_field, None)
    if cap_param is None or cap_param.frame.height == 0:
        return
    flow_lhs = _flow_lhs(d, vars)
    if flow_lhs is None:
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}Cumulative_flow_solve"

    # Outer index: g only.
    outer = cap_param.frame.select("g").unique()
    # LHS:  Σ_{(d,t)} flow_lhs * step_duration  → collapses (d, t)
    # RHS:  p_group[g, ?] * hours_in_solve.
    # hours_in_solve = Σ_{(d,t)} step_duration[d, t]  — a scalar but
    # easier to compute from the frame here.
    hours_in_solve = float(d.p_step_duration.frame["value"].sum())
    # Cap_param has dims (g,); multiply by hours_in_solve scalar.
    rhs_param = Param(("g",), cap_param.frame.with_columns(
        value=pl.col("value") * hours_in_solve))
    lhs_int = Sum(flow_lhs * d.p_step_duration, over=("d", "t"))
    m.add_cstr(
        name,
        over      = outer,
        sense     = sense,
        lhs_terms = {"flow_int": lhs_int},
        rhs_terms = {"cap":      rhs_param},
    )


def _emit_cumulative_flow_period(m, d, vars: dict, sense: str) -> None:
    cap_field = ("pd_max_cumulative_flow" if sense == "<="
                 else "pd_min_cumulative_flow")
    cap_param = getattr(d, cap_field, None)
    if cap_param is None or cap_param.frame.height == 0:
        return
    flow_lhs = _flow_lhs(d, vars)
    if flow_lhs is None:
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}Cumulative_flow_period"

    # Outer index: (g, d) where pd_*_cumulative_flow is set.
    outer = cap_param.frame.select("g", "d").unique()
    # hours_in_period[d] = Σ_t step_duration[d, t]
    hip = (d.p_step_duration.frame
           .group_by("d").agg(pl.col("value").sum())
           .rename({"value": "hours_in_period"}))
    rhs_frame = (cap_param.frame.join(hip, on="d", how="inner")
                  .with_columns(value=pl.col("value")
                                       * pl.col("hours_in_period"))
                  .select("g", "d", "value"))
    rhs_param = Param(("g", "d"), rhs_frame)
    # LHS: integrate over t only — Σ_t flow_lhs * step_duration.
    # Sum collapses 't' but keeps 'd'.
    lhs_int = Sum(flow_lhs * d.p_step_duration, over=("t",))
    m.add_cstr(
        name,
        over      = outer,
        sense     = sense,
        lhs_terms = {"flow_int": lhs_int},
        rhs_terms = {"cap":      rhs_param},
    )


def _emit_instant_flow(m, d, vars: dict, sense: str) -> None:
    if sense == "<=":
        gdt_idx   = getattr(d, "gdt_maxInstantFlow", None)
        cap_field = "pdt_max_instant_flow"
    else:
        gdt_idx   = getattr(d, "gdt_minInstantFlow", None)
        cap_field = "pdt_min_instant_flow"
    if gdt_idx is None or gdt_idx.height == 0:
        return
    cap_param = getattr(d, cap_field, None)
    if cap_param is None:
        return
    flow_lhs = _flow_lhs(d, vars)
    if flow_lhs is None:
        return
    pre = "max" if sense == "<=" else "min"
    name = f"{pre}Instant_flow"
    m.add_cstr(
        name,
        over      = gdt_idx,
        sense     = sense,
        lhs_terms = {"flow": flow_lhs},
        rhs_terms = {"cap":  cap_param},
    )
