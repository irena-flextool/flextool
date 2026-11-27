"""Representative-period storage: the base-period state chain and the ends of the chain."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.model import _capacity
from flextool.engine.model import _self_discharge
from flextool.engine.model._common import _has_rows, dispatch_periods, _var_view
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._storage_links import (
    _invest_at_start_nodes,
    _nested_nodes,
    _notice_nodes,
    _roll_end_rows,
    _warn_reference_below_start,
)


# ---------------------------------------------------------------------------
# Equations


def _eq_rp_storage_chain(b: _Build) -> None:
    """Representative-period storage: the base-period state chain of every period and its bounds."""
    if not (b.has_storage and b.has_rp):
        return
    m, d = b.m, b.d
    v_inter = b.v_state_inter
    b_dtype = v_inter.frame.schema["b"]
    d_dtype = v_inter.frame.schema["d"]
    has_rep_blocks = d.rp_base__rep is not None and _has_rows(d.rp_block_first) and _has_rows(d.p_rp_last_step)
    nodes = d.nodeState_rp

    def typed(frame: pl.DataFrame) -> pl.DataFrame:
        """Base-period labels (b, b_prev) onto v_state_inter's b dtype, periods (d, d_prev) onto its d dtype."""
        return frame.with_columns(
            [pl.col(c).cast(b_dtype, strict=False) for c in ("b", "b_prev") if c in frame.columns]
            + [pl.col(c).cast(pl.Utf8).cast(d_dtype, strict=False) for c in ("d", "d_prev") if c in frame.columns])

    first = typed(d.rp_base_first) if _has_rows(d.rp_base_first) else None    # (d, b)
    last = typed(d.rp_base_last) if _has_rows(d.rp_base_last) else None       # (d, b)
    period_previous = typed(d.rp_period_previous) if _has_rows(d.rp_period_previous) else None
    inter_prev = _var_view(v_inter, "__prev", {"d": "d_prev", "b": "b_prev"})
    drift = _rp_weighted_drift(b, nodes) if has_rep_blocks else None          # (n, d, b)
    keys = ("n", "d", "b", "d_prev", "b_prev")

    # ─── EQ. ─── rp_inter_period_balance ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − v_state_inter[n,d′,b′] = drift[n,d,b]
    #   drift[n,d,b] = Σ_{(d,b,r) ∈ rp_base__rep, (d,r) ∈ rp_block_first, d ∈ periods of dt}
    #                    rp_weight[d,b,r] · (v_state[n,d,last_step[d,r]] − v_state_rp_start[n,d,r])
    #   ∀ n ∈ nodeState_rp, (d,b,b′) ∈ rp_base_chain, d′ = d                         (within a period)
    #   ∀ n ∈ bind_within_solve_blended_weights ∪ bind_forward_only_blended_weights,
    #     (d,d′) ∈ rp_period_previous, b = first base period of d, b′ = last of d′    (period to period)
    # Chains the storage state across the base periods of each period: a base
    # period is represented by weighted representative blocks r, and the
    # boundary state moves by the weighted net change of those blocks.  The
    # across-solve variants continue from the last base period of the previous
    # period in solve order, as bind_within_solve / bind_forward_only continue
    # v_state across periods.  All terms are in state units.
    edges = []
    if _has_rows(d.rp_base_chain):
        chain = typed(d.rp_base_chain).with_columns(d_prev=pl.col("d"))
        edges.append(nodes.join(chain.select("d", "b", "d_prev", "b_prev"), how="cross"))
    across = [f for f in (d.storage_bind_within_solve_blended_weights, d.storage_bind_forward_only_blended_weights)
              if _has_rows(f)]
    if across and period_previous is not None and first is not None and last is not None:
        cross = (period_previous
                 .join(first, on="d", how="inner")
                 .join(last.rename({"d": "d_prev", "b": "b_prev"}), on="d_prev", how="inner")
                 .select("d", "b", "d_prev", "b_prev"))
        edges.append(pl.concat(across, how="vertical").unique().join(cross, how="cross"))
    if drift is not None and edges:
        over = pl.concat([e.select(*keys) for e in edges], how="vertical").unique(maintain_order=True)
        m.add_cstr(
            "rp_inter_period_balance",
            over      = over,
            sense     = "==",
            lhs_terms = {"d_inter": Where(v_inter, over) - Where(inter_prev, over),
                         "neg_rp_drift": -Where(drift, over)},
            rhs_terms = {},
        )

    # ─── EQ. ─── rp_inter_period_cyclic ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b_first(d)] − v_state_inter[n,d′,b_last(d′)] = drift[n,d,b_first(d)]
    #   n ∈ bind_within_period_blended_weights: every rp period d, d′ = d
    #   n ∈ bind_within_solve_blended_weights: (d,d′) ∈ d_first × d_last, d_first = the rp periods without a
    #     predecessor in rp_period_previous, d_last = the rp periods that are no period's predecessor
    # Closes the chain: the last boundary state connects back to the first.
    # bind_within_period_blended_weights closes every period on itself,
    # bind_within_solve_blended_weights closes once per path of the solve:
    # deterministic, one pair (first period, last period).  In a stochastic
    # solve the rp periods form a tree (rp_period_previous): a fan after the
    # trunk period (the shared period before the split) has one d_first and
    # one d_last per forecast branch, so every branch's last period ends at
    # the level the trunk period starts from; a fan in the first period has
    # N roots and N leaves, and the N × N pairs are the loop of each path plus
    # an equal chain start of all the roots.
    # bind_forward_only_blended_weights has no closure.
    closures = []
    if first is not None and last is not None:
        within_period = d.storage_bind_within_period_blended_weights
        if _has_rows(within_period):
            pairs = first.join(last.rename({"b": "b_prev"}), on="d", how="inner").with_columns(d_prev=pl.col("d"))
            closures.append(within_period.join(pairs.select("d", "b", "d_prev", "b_prev"), how="cross"))
        within_solve = d.storage_bind_within_solve_blended_weights
        if _has_rows(within_solve):
            periods = first.select("d").unique()
            if period_previous is not None:
                d_first = periods.join(period_previous.select("d"), on="d", how="anti")
                d_last = periods.join(period_previous.select(pl.col("d_prev").alias("d")), on="d", how="anti")
            else:
                d_first = d_last = periods
            pair = (first.join(d_first, on="d", how="inner")
                    .join(last.join(d_last, on="d", how="inner").rename({"d": "d_prev", "b": "b_prev"}),
                          how="cross"))
            closures.append(within_solve.join(pair.select("d", "b", "d_prev", "b_prev"), how="cross"))
    if drift is not None and closures:
        over = pl.concat([c.select(*keys) for c in closures], how="vertical").unique(maintain_order=True)
        m.add_cstr(
            "rp_inter_period_cyclic",
            over      = over,
            sense     = "==",
            lhs_terms = {"d_cyclic": Where(v_inter, over) - Where(inter_prev, over),
                         "neg_rp_drift_first": -Where(drift, over)},
            rhs_terms = {},
        )

    # ─── EQ. ─── rp_inter_period_fan_start ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b_first(d)] − drift[n,d,b_first(d)]
    #   − v_state_inter[n,d″,b_first(d″)] + drift[n,d″,b_first(d″)] = 0
    #   ∀ n ∈ bind_forward_only_blended_weights, n ∉ the nodes of rp_inter_period_start_fix,
    #     d = the first root (sorted), d″ ∈ the other roots
    #   roots = the rp periods of period_first_of_solve (the forecast branch periods of a fan in the solve's
    #     first period; one root without a fan: no rows), read as the periods of nodeState_first_dt, which is
    #     built from period_first_of_solve
    # Every forecast of a fan in the solve's first period starts its base-period
    # chain from the same level: the chain start v_state_inter − drift of the
    # first base period, which no other row binds in a forward-only chain.  A
    # fix_start node is not included: rp_inter_period_start_fix pins each
    # root's chain start at the share of that forecast's own capacity.  The
    # bind_within_solve_blended_weights roots are
    # already tied by rp_inter_period_cyclic; the base-period levels inside a
    # tie window need no tie of their own (they follow from the tied net flows,
    # the tied block starts and this shared chain start).
    if (drift is not None and first is not None and _has_rows(d.storage_bind_forward_only_blended_weights)
            and _has_rows(d.nodeState_first_dt)):
        root_names = sorted(set(str(x) for x in d.nodeState_first_dt["d"].to_list())
                            & set(str(x) for x in first["d"].to_list()))
        if len(root_names) >= 2:
            roots = first.filter(pl.col("d").cast(pl.Utf8).is_in(root_names))
            ref = roots.filter(pl.col("d").cast(pl.Utf8) == root_names[0])
            other = (roots.filter(pl.col("d").cast(pl.Utf8) != root_names[0])
                     .rename({"d": "d_prev", "b": "b_prev"}))
            fan_nodes = d.storage_bind_forward_only_blended_weights.select("n").unique()
            pinned = _rp_start_pin_nodes(b)
            if pinned is not None:
                fan_nodes = fan_nodes.filter(~pl.col("n").cast(pl.Utf8).is_in(pinned["n"].cast(pl.Utf8).implode()))
            over = (fan_nodes
                    .join(ref.select("d", "b"), how="cross")
                    .join(other.select("d_prev", "b_prev"), how="cross")
                    .select(*keys).sort(pl.col("d_prev").cast(pl.Utf8)))
            drift_prev = _rp_weighted_drift(b, nodes, at_prev=True)
            if over.height > 0 and drift_prev is not None:
                m.add_cstr(
                    "rp_inter_period_fan_start",
                    over      = over,
                    sense     = "==",
                    lhs_terms = {"d_fan": Where(v_inter, over) - Where(inter_prev, over),
                                 "neg_rp_drift": -Where(drift, over),
                                 "rp_drift_prev": Where(drift_prev, over)},
                    rhs_terms = {},
                )

    # ─── EQ. ─── rp_inter_period_max_state / maxState_rp_start ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] + retired[n,d]
    #   − invested[n,d]  ≤  state_upper[n,d]
    #   ∀ n ∈ nodeState_rp, (d,b) ∈ rp_base_period, d ∈ periods of dt
    # v_state_rp_start[n,d,t] + (same invest/divest terms)  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_rp × rp_block_first
    # Capacity bounds of the boundary and block-start states, as maxState
    # bounds v_state: a base period of d is bounded by d's capacity.
    # state_upper = capacity / state_unitsize, so every term is in state units.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if _has_rows(d.rp_base_period_set):
        base_periods = typed(d.rp_base_period_set)
        if _has_rows(d.period_in_use_set):
            base_periods = base_periods.join(
                dispatch_periods(d).select(pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False)).unique(),
                on="d", how="inner")
        over = nodes.join(base_periods.select("d", "b"), how="cross").select("n", "d", "b")
        if over.height > 0:
            lhs: dict = {"state_inter": Where(v_inter, over)}
            lhs.update(_capacity.node_capacity_change_terms(b, nodes, "_rp"))
            m.add_cstr(
                "rp_inter_period_max_state",
                over      = over,
                sense     = "<=",
                lhs_terms = lhs,
                rhs_terms = {"upper": d.p_state_upper},
            )
    if _has_rows(b.rp_start_idx):
        lhs = {"state_rp_start": b.v_state_rp_start}
        lhs.update(_capacity.node_capacity_change_terms(b, nodes, "_rp"))
        m.add_cstr(
            "maxState_rp_start",
            over      = b.rp_start_idx,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": d.p_state_upper},
        )


def _rp_typed(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """Base-period labels (b, b_prev) onto v_state_inter's b dtype, periods (d, d_prev) onto its d dtype."""
    schema = b.v_state_inter.frame.schema
    return frame.with_columns(
        [pl.col(c).cast(schema["b"], strict=False) for c in ("b", "b_prev") if c in frame.columns]
        + [pl.col(c).cast(pl.Utf8).cast(schema["d"], strict=False) for c in ("d", "d_prev") if c in frame.columns])


def _rp_chain_paths(b: _Build) -> "tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None":
    """``(roots, leaves, paths)`` of the base-period chain of a representative-period solve, typed onto
    v_state_inter.

    roots (d, b): the first base period of each rp period without a predecessor in rp_period_previous (the solve's
    first period; each forecast branch of a fan in the first period); leaves (d, b): the last base period of each rp
    period that is no period's predecessor (the solve's last period; each terminal branch); paths (d, b, d_prev,
    b_prev): each leaf (d_prev, b_prev) with the root (d, b) its periods descend from.  None without a chain.
    """
    d = b.d
    if not (_has_rows(d.rp_base_first) and _has_rows(d.rp_base_last)):
        return None
    first = _rp_typed(b, d.rp_base_first).select("d", "b")
    last = _rp_typed(b, d.rp_base_last).select("d", "b")
    if _has_rows(d.period_in_use_set):
        in_use = dispatch_periods(d).select(pl.col("d").cast(pl.Utf8))
        first = first.filter(pl.col("d").cast(pl.Utf8).is_in(in_use["d"].implode()))
        last = last.filter(pl.col("d").cast(pl.Utf8).is_in(in_use["d"].implode()))
    prev_of: dict[str, str] = {}
    if _has_rows(d.rp_period_previous):
        prev_of = {str(a): str(p) for a, p in d.rp_period_previous.select(
            pl.col("d").cast(pl.Utf8), pl.col("d_prev").cast(pl.Utf8)).iter_rows()}
    periods = [str(x) for x in first["d"].cast(pl.Utf8).to_list()]
    is_prev = set(prev_of.values())
    root_names = [p for p in periods if p not in prev_of]
    leaf_names = [str(x) for x in last["d"].cast(pl.Utf8).to_list() if str(x) not in is_prev]
    if not root_names or not leaf_names:
        return None

    def root_of(period: str) -> str:
        seen = set()
        while period in prev_of and period not in seen:
            seen.add(period)
            period = prev_of[period]
        return period

    roots = first.filter(pl.col("d").cast(pl.Utf8).is_in(root_names))
    leaves = last.filter(pl.col("d").cast(pl.Utf8).is_in(leaf_names))
    pairs = pl.DataFrame({"_root": [root_of(p) for p in leaf_names], "_leaf": leaf_names},
                         schema={"_root": pl.Utf8, "_leaf": pl.Utf8})
    paths = (pairs
             .join(roots.with_columns(_root=pl.col("d").cast(pl.Utf8)), on="_root", how="inner")
             .join(leaves.rename({"d": "d_prev", "b": "b_prev"}).with_columns(_leaf=pl.col("d_prev").cast(pl.Utf8)),
                   on="_leaf", how="inner")
             .select("d", "b", "d_prev", "b_prev"))
    return roots, leaves, paths


def _rp_node_set(b: _Build, frame) -> "pl.DataFrame | None":
    """``frame`` (n) ∩ nodeState_rp, typed onto v_state_inter's n; None when empty."""
    d = b.d
    if not (_has_rows(frame) and _has_rows(d.nodeState_rp)):
        return None
    n_dtype = b.v_state_inter.frame.schema["n"]
    out = (frame.select(pl.col("n").cast(pl.Utf8)).unique()
           .join(d.nodeState_rp.select(pl.col("n").cast(pl.Utf8)).unique(), on="n", how="inner")
           .select(pl.col("n").cast(n_dtype, strict=False)).sort("n"))
    return out if out.height > 0 else None


def _rp_minus(frame: "pl.DataFrame | None", other) -> "pl.DataFrame | None":
    """``frame`` (n) without the nodes of ``other`` (any frame with n); None when empty."""
    if frame is None:
        return None
    if _has_rows(other):
        frame = frame.filter(~pl.col("n").cast(pl.Utf8).is_in(other.select(pl.col("n").cast(pl.Utf8))["n"].implode()))
    return frame if frame.height > 0 else None


def _rp_start_pin_nodes(b: _Build) -> "pl.DataFrame | None":
    """``(n,)``: the representative-period nodes whose chain start fix_start pins (storage_fix_start ∩
    nodeState_rp, not carried); None when none or the data to pin is missing."""
    d = b.d
    if not (b.has_storage and b.has_rp and d.p_state_start is not None and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return None
    nodes = _rp_node_set(b, d.storage_fix_start)
    return _rp_minus(nodes, _self_discharge.carried_nodes(d))


def _rp_end_pin_nodes(b: _Build) -> "pl.DataFrame | None":
    """``(n,)``: the representative-period nodes whose chain end fix_end pins (storage_fix_end ∩ nodeState_rp, in a
    solve whose chain ends at the end of the model: it holds the model's last period, storage_rp_chain_at_model_end,
    or its last step; no upper-level fix_quantity / fix_usage on the node)."""
    d = b.d
    at_end = d.storage_solve_at_model_end or getattr(d, "storage_rp_chain_at_model_end", False)
    if not (b.has_storage and b.has_rp and at_end and d.p_state_end is not None
            and b.cap.existing_state_n is not None and d.p_state_unitsize is not None):
        return None
    return _rp_minus(_rp_node_set(b, d.storage_fix_end), _nested_nodes(b, with_price=False))


def _eq_rp_chain_ends(b: _Build) -> None:
    """Representative-period storage: the absolute level at the start and the end of the base-period chain
    (fix_start, fix_end, use_reference_value, the start's capacity bounds and the matching end)."""
    if not (b.has_storage and b.has_rp and b.v_state_inter is not None):
        return
    m, d = b.m, b.d
    has_rep_blocks = d.rp_base__rep is not None and _has_rows(d.rp_block_first) and _has_rows(d.p_rp_last_step)
    shape = _rp_chain_paths(b)
    if not has_rep_blocks or shape is None:
        return
    roots, leaves, paths = shape
    drift = _rp_weighted_drift(b, d.nodeState_rp)
    if drift is None:
        return
    v_inter = b.v_state_inter
    inter_prev = _var_view(v_inter, "__chain_end", {"d": "d_prev", "b": "b_prev"})
    forward = _rp_node_set(b, d.storage_bind_forward_only_blended_weights)
    start_pin = _rp_start_pin_nodes(b)
    end_pin = _rp_end_pin_nodes(b)
    su = d.p_state_unitsize

    # ─── EQ. ─── rp_inter_period_start_fix ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − drift[n,d,b] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]
    #   = state_start[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_start ∩ nodeState_rp, n ∉ carried, (d,b) ∈ chain roots (the first base period of the
    #     solve's first period; of each forecast branch of a fan in the first period)
    # The level at the start of the chain (the level the first base period starts from: its end level less the
    # weighted net change of its blocks) is the share state_start of the storage capacity in service, in state
    # units.  The block-internal states start from free offsets (v_state_rp_start), so the start is pinned on the
    # chain, the absolute level, not on a state inside the first block.  With bind_within_solve_blended_weights /
    # bind_within_period_blended_weights the closure (rp_inter_period_cyclic) then holds the cycle at that level, as
    # storage_state_start_binding_cyclic_period does without representative periods.  Only where no state of the
    # node was realised before this solve (the first solve of the model and of each nested level).
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if start_pin is not None:
        over = start_pin.join(roots, how="cross").select("n", "d", "b")
        lhs: dict = {"state_inter": Where(v_inter, over), "neg_rp_drift": -Where(drift, over)}
        invested, divested = _capacity.node_in_service(b, start_pin, "_rp_start", coef=d.p_state_start)
        if invested is not None:
            lhs["invest_neg"] = -invested
        if divested is not None:
            lhs["divest"] = divested
        _notice_nodes("rp_storage_chain_pinned", start_pin, b)
        m.add_cstr(
            "rp_inter_period_start_fix",
            over      = over,
            sense     = "==",
            lhs_terms = lhs,
            rhs_terms = {"target": (d.p_state_start * b.cap.existing_state_n) / su},
        )

    # ─── EQ. ─── rp_inter_period_end_fix ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − state_end[n]·invested[n,d] + state_end[n]·retired[n,d]
    #   = state_end[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_end ∩ nodeState_rp, (d,b) ∈ chain leaves (the last base period of the solve's last
    #     period; of each terminal branch), the chain ends at the end of the model (storage_rp_chain_at_model_end: the
    #     solve holds the model's last period, whose last base period ends with it), no upper-level fix_quantity /
    #     fix_usage on n
    # The level at the end of the chain (the end of the last base period) is the share state_end of the storage
    # capacity in service, in every solve whose horizon holds the last step of the model, as
    # storage_state_end_binding pins v_state without representative periods.
    if end_pin is not None:
        over = end_pin.join(leaves, how="cross").select("n", "d", "b")
        lhs = {"state_inter": Where(v_inter, over)}
        invested, divested = _capacity.node_in_service(b, end_pin, "_rp_end", coef=d.p_state_end)
        if invested is not None:
            lhs["invest_neg"] = -invested
        if divested is not None:
            lhs["divest"] = divested
        _notice_nodes("rp_storage_chain_pinned", end_pin, b)
        m.add_cstr(
            "rp_inter_period_end_fix",
            over      = over,
            sense     = "==",
            lhs_terms = lhs,
            rhs_terms = {"target": (d.p_state_end * b.cap.existing_state_n) / su},
        )

    if forward is None:
        return
    # ─── EQ. ─── rp_inter_period_start_carry ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # (v_state_inter[n,d,b] − drift[n,d,b]) · state_unitsize[n] = roll_continue_state[n]
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp ∩ carried, (d,b) ∈ chain roots
    # A representative-period solve that follows another solve (a sequence, or a roll) starts the base-period chain
    # of a forward node from the level realised before it (the carry, MWh, routed by the cascade), as a
    # bind_forward_only node without representative periods does at its first step (owner decision).  Before, the
    # chain started free.  The block-internal states keep their free offsets: only the chain is absolute.
    carried = _rp_node_set(b, _self_discharge.carried_nodes(d))
    carry_start = None if carried is None else _rp_minus(
        forward.join(carried, on="n", how="semi"), start_pin)
    if carry_start is not None and d.p_roll_continue_state is not None:
        over = carry_start.join(roots, how="cross").select("n", "d", "b")
        level = (d.p_roll_continue_state.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
                 .join(over.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                   pl.col("b").cast(pl.Utf8)), on="n", how="inner")
                 .select("n", "d", "b", "value"))
        _notice_nodes("rp_storage_chain_pinned", carry_start, b)
        m.add_cstr(
            "rp_inter_period_start_carry",
            over      = over,
            sense     = "==",
            lhs_terms = {"state_inter": Where(v_inter, over) * su, "neg_rp_drift": -Where(drift, over) * su},
            rhs_terms = {"carry": Param(("n", "d", "b"), _rp_typed(b, level))},
        )
    # The forward chains with a free start: neither the start pin nor the carry above applies.
    free_start = _rp_minus(_rp_minus(forward, start_pin), carry_start)

    # ─── EQ. ─── rp_inter_period_start_max_state / rp_inter_period_start_min_state ─
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − drift[n,d,b] + retired[n,d] − invested[n,d]  ≤  state_upper[n,d]
    # v_state_inter[n,d,b] − drift[n,d,b]  ≥  0
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp, n ∉ the nodes of rp_inter_period_start_fix,
    #     (d,b) ∈ chain roots
    # The level a forward chain starts from is a physical level: within the storage capacity in service, as every
    # base-period level (rp_inter_period_max_state).  Without the bounds the start of a forward chain had no bound.
    if free_start is not None:
        over = free_start.join(roots, how="cross").select("n", "d", "b")
        lhs = {"state_inter": Where(v_inter, over), "neg_rp_drift": -Where(drift, over)}
        upper = dict(lhs)
        upper.update(_capacity.node_capacity_change_terms(b, free_start, "_rp_start_max"))
        _notice_nodes("rp_storage_chain_pinned", free_start, b)
        m.add_cstr(
            "rp_inter_period_start_max_state",
            over      = over,
            sense     = "<=",
            lhs_terms = upper,
            rhs_terms = {"upper": d.p_state_upper},
        )
        m.add_cstr(
            "rp_inter_period_start_min_state",
            over      = over,
            sense     = ">=",
            lhs_terms = lhs,
            rhs_terms = {},
        )

    # The forward chains whose start is a decision of this solve (storage_semantics: the matching end applies):
    # a free start, or a fix_start share of capacity the solve can invest in during the start period; not where an
    # upper-level solve fixes the node, nor where fix_start_end pins the end at least at the start share.
    root_periods = [str(x) for x in roots["d"].cast(pl.Utf8).to_list()]
    investable = _invest_at_start_nodes(b, _rp_minus(forward, free_start), root_periods)
    closed_by_end = set()
    if end_pin is not None and d.p_state_start is not None and d.p_state_end is not None:
        shares = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                  .join(d.p_state_end.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("e")), on="n"))
        pinned = set(str(n) for n in end_pin["n"].to_list())
        closed_by_end = {n for n, s, e in shares.iter_rows() if n in pinned and e >= s - 1e-12}
    match_names = (set() if free_start is None else set(str(n) for n in free_start["n"].to_list())) | \
        (investable - closed_by_end)
    nested = _nested_nodes(b, with_price=True)
    if nested is not None:
        match_names -= set(nested["n"].to_list())
    ref_nodes = _rp_minus(_rp_node_set(b, d.storage_use_reference_value), nested)
    ref_nodes = _rp_minus(ref_nodes, end_pin)
    if ref_nodes is not None:
        ref_nodes = ref_nodes.filter(pl.col("n").cast(pl.Utf8).is_in(forward["n"].cast(pl.Utf8).implode()))
        ref_nodes = ref_nodes if ref_nodes.height > 0 else None
    match = forward.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(match_names)))
    endowment = None
    if match.height > 0:
        # Endowment (state units): the start a fix_start node has on the capacity that exists before the solve
        # decides; 0 for a free start.  Only the start on the capacity the solve adds must come back.
        endowment = (match.select(pl.col("n").cast(pl.Utf8)).join(
            pl.DataFrame({"d": root_periods}, schema={"d": pl.Utf8}), how="cross"))
        if investable and d.p_state_start is not None and b.cap.existing_state_n is not None:
            exist = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                     .filter(pl.col("n").is_in(sorted(investable)))
                     .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                              pl.col("value").alias("cap")), on="n")
                     .join(su.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("us")), on="n")
                     .select("n", "d", value=-pl.col("s") * pl.col("cap") / pl.col("us")))
            endowment = endowment.join(exist, on=["n", "d"], how="left")
        else:
            endowment = endowment.with_columns(value=pl.lit(None, dtype=pl.Float64))
        endowment = endowment.with_columns(pl.col("value").fill_null(0.0))

    # ─── EQ. ─── rp_inter_period_end_match ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d′,b′] − (v_state_inter[n,d,b] − drift[n,d,b])  ≥  − state_start[n]·existing[n,d] / su[n]
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp whose start is a decision of the solve: a free start
    #     (no fix_start, or carried), or fix_start while the solve can invest in n in the start period d (not with
    #     fix_start_end at state_end ≥ state_start); n without an upper-level fix;
    #     (d,b,d′,b′) ∈ chain paths (root (d,b), the leaf (d′,b′) of each path); the right-hand side is 0 for a
    #     free start
    # The matching end: a forward chain whose start the solve chooses ends at least at that start, so the solve
    # cannot take energy it never produced (a free start full, the end empty; or storage built as a fuel tank whose
    # fix_start share is spent once).  With fix_start only the start on the capacity the solve adds must come back:
    # the start on existing capacity is the user's explicit endowment (state units).  An inequality: a start of 0,
    # or no investment, always meets it, so it never forces slack.  fix_end and use_reference_value still set the
    # end; with a free start this bounds the start by that end.
    if match.height > 0:
        over = match.join(paths, how="cross").select("n", "d", "b", "d_prev", "b_prev")
        rhs = Param(("n", "d"), _rp_typed(b, endowment.with_columns(
            pl.col("n").cast(v_inter.frame.schema["n"], strict=False))).select("n", "d", "value"))
        _notice_nodes("storage_end_matches_start", match, b)
        m.add_cstr(
            "rp_inter_period_end_match",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state_end": Where(inter_prev, over), "neg_state_start": -Where(v_inter, over),
                         "rp_drift_start": Where(drift, over)},
            rhs_terms = {"endowment": rhs},
        )

    # ─── EQ. ─── rp_inter_period_start_match ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] · state_unitsize[n]  ≥  min(start[n], capacity[n,d])
    #   ∀ n ∈ storage_roll_end(match) ∩ bind_forward_only_blended_weights ∩ nodeState_rp, (d,b) ∈ chain leaves,
    #     not pinned by fix_end, no upper-level fix
    #   start = the level (MWh) the node started the model from (a free start, recorded by the cascade)
    # The matching end of a carried chain: a later representative-period solve that reaches the end of the model
    # ends the chain at least at the level the model started the node from, so the solves together return it.
    starts = _roll_end_rows(d, ("match",))
    if starts is not None and b.cap.existing_state_n is not None:
        names = set(starts["n"].cast(pl.Utf8).to_list()) & set(forward["n"].cast(pl.Utf8).to_list())
        if end_pin is not None:
            names -= set(end_pin["n"].cast(pl.Utf8).to_list())
        if nested is not None:
            names -= set(nested["n"].to_list())
        nodes = forward.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(names)))
        if nodes.height > 0:
            over = nodes.join(leaves, how="cross").select("n", "d", "b")
            level = (over.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
                     .join(starts.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("_s")), on="n")
                     .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                              pl.col("value").alias("_c")), on=["n", "d"], how="left")
                     .select("n", "d", "b", value=pl.min_horizontal("_s", pl.col("_c").fill_null(pl.col("_s")))))
            _notice_nodes("storage_end_matches_start", nodes, b)
            m.add_cstr(
                "rp_inter_period_start_match",
                over      = over,
                sense     = ">=",
                lhs_terms = {"state_inter": Where(v_inter, over) * su},
                rhs_terms = {"start": Param(("n", "d", "b"), _rp_typed(b, level))},
            )

    # ─── EQ. ─── rp_inter_period_reference_value ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − reference_value[n,d,t]·invested[n,d] + reference_value[n,d,t]·retired[n,d]
    #   = reference_value[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_use_reference_value ∩ bind_forward_only_blended_weights ∩ nodeState_rp, (d,b) ∈ chain leaves,
    #     t = the node's last step in d (nodeState_last_dt), no upper-level fix on n
    # use_reference_value sets the level at the end of the chain (the absolute level at the end of the horizon), not
    # a state inside the last block.  A reference value below the start share of a node the solve can invest in:
    # the matching end above stays (the solve cannot build storage to spend its start share) and caps the
    # investment; a warning (_warn_reference_below_start).
    if ref_nodes is not None and d.p_storage_state_reference_value is not None and _has_rows(d.nodeState_last_dt):
        last_step = (d.nodeState_last_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                     .join(leaves.select(pl.col("d").cast(pl.Utf8)), on="d", how="inner")
                     .filter(pl.col("n").is_in(ref_nodes["n"].cast(pl.Utf8).implode())))
        rv = (promote_param_to_dt(d.p_storage_state_reference_value, d.dt)
              .select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), "value")
              .join(last_step.lazy(), on=["n", "d", "t"], how="inner")
              .select("n", "d", "value").unique(subset=["n", "d"], keep="first").collect())
        if investable and d.p_state_start is not None:
            share = {str(n): v for n, v in d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), "value").iter_rows()}
            _warn_reference_below_start(b, {n for n, _d, v in rv.iter_rows()
                                            if n in investable - closed_by_end and v < share.get(n, 0.0) - 1e-12})
        target = (rv.rename({"value": "rv"})
                  .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                            pl.col("value").alias("exist")), on=["n", "d"], how="left")
                  .join(su.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("us")), on="n", how="inner")
                  .with_columns(value=pl.col("rv") * pl.col("exist").fill_null(0.0) / pl.col("us")))
        part = target
        nodes = ref_nodes.filter(pl.col("n").cast(pl.Utf8).is_in(part["n"].implode()))
        over = _rp_typed(b, nodes.join(leaves, how="cross").select("n", "d", "b"))
        over = over.join(_rp_typed(b, part.select("n", "d").with_columns(
            pl.col("n").cast(v_inter.frame.schema["n"], strict=False))), on=["n", "d"], how="semi")
        if over.height > 0:
            coef = Param(("n", "d"), _rp_typed(b, part.select("n", "d", pl.col("rv").alias("value")).with_columns(
                pl.col("n").cast(v_inter.frame.schema["n"], strict=False))))
            lhs = {"state_inter": Where(v_inter, over)}
            invested, divested = _capacity.node_in_service(b, nodes, "_rp_ref", coef=coef)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            _notice_nodes("rp_storage_chain_pinned", nodes, b)
            m.add_cstr(
                "rp_inter_period_reference_value",
                over      = over,
                sense     = "==",
                lhs_terms = lhs,
                rhs_terms = {"target": Param(("n", "d"), _rp_typed(b, part.select("n", "d", "value").with_columns(
                    pl.col("n").cast(v_inter.frame.schema["n"], strict=False))))},
            )


def _rp_weighted_drift(b: _Build, nodes: pl.DataFrame, *, at_prev: bool = False):
    """drift[n,d,b] = Σ_r rp_weight[d,b,r] · (v_state[n,d,last_step[d,r]] − v_state_rp_start[n,d,r]).

    Block r of period d starts at (d, t = r) ∈ rp_block_first and ends at last_step[d,r].  Dims (n, d, b);
    with ``at_prev`` dims (n, d_prev, b_prev), the drift of the other side of a period-to-period row.
    None when no representative block is in use.
    """
    d = b.d
    t_dtype = d.rp_block_first.schema["t"]
    d_dtype = d.rp_block_first.schema["d"]
    b_dtype = b.v_state_inter.frame.schema["b"]
    state_t_dtype = b.v_state.frame.schema["t"]
    block_first = (d.rp_block_first.rename({"t": "r"})
                   .with_columns(pl.col("r").cast(t_dtype, strict=False)))
    rep = d.rp_base__rep.frame.with_columns(
        pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False),
        pl.col("b").cast(b_dtype, strict=False),
        pl.col("r").cast(t_dtype, strict=False))
    last_step = d.p_rp_last_step.with_columns(
        pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False),
        pl.col("r").cast(t_dtype, strict=False),
        pl.col("last_step").cast(state_t_dtype, strict=False))
    cohort = (rep.join(block_first, on=["d", "r"], how="inner")
              .join(last_step, on=["d", "r"], how="inner"))
    if d.period_in_use_set is not None:
        cohort = cohort.join(dispatch_periods(d).with_columns(pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False)),
                             on="d", how="inner")
    if cohort.height == 0:
        return None
    keys = ("d", "b", "r", "last_step")
    if at_prev:
        cohort = cohort.rename({"d": "d_prev", "b": "b_prev"})
        keys = ("d_prev", "b_prev", "r", "last_step")
        state_at_last = _var_view(b.v_state, "__rp_last_prev", {"d": "d_prev", "t": "last_step"})
        start_at_r = _var_view(b.v_state_rp_start, "__at_r_prev", {"d": "d_prev", "t": "r"})
    else:
        state_at_last = _var_view(b.v_state, "__rp_last", {"t": "last_step"})
        start_at_r = _var_view(b.v_state_rp_start, "__at_r", {"t": "r"})
    rp_weight = Param(keys, cohort.select(*keys, "value"))
    idx = nodes.select("n").unique().join(cohort.select(*keys), how="cross")
    return Sum((Where(state_at_last, idx) - Where(start_at_r, idx)) * rp_weight,
               over=("r", "last_step"))
