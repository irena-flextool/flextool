"""Storage state limits and the start and end conditions of storage chains."""

import polars as pl
from polar_high import Where, Param
from flextool.engine.model import _capacity
from flextool.engine.model import _non_anticipativity
from flextool.engine.model import _self_discharge
from flextool.engine.preprocess._step_relations import labels_with_steps, scope_end_pairs
from flextool.engine.model._common import _availability_factor, _has_rows, _var_view
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model.core._context import _Build, _LOG
from flextool.engine.model.core._storage_links import (
    _fix_quantity_target,
    _invest_at_start_nodes,
    _nested_fixed_end,
    _nested_nodes,
    _notice_nodes,
    _roll_end_rows,
    _warn_reference_below_start,
    _without_steps,
)


def _eq_max_state(b: _Build) -> None:
    """Capacity limit of the storage state."""
    if not b.has_storage:
        return
    m, d = b.m, b.d

    # ─── EQ. ─── maxState ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] + retired[n,d]
    #   − invested[n,d]  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_dt
    # Caps the storage state by the built storage capacity.  state_upper = capacity / state_unitsize, so
    # every term is in state units.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    lhs: dict = {"state": b.v_state}
    lhs.update(_capacity.node_capacity_change_terms(b, d.nodeState, ""))
    m.add_cstr(
        "maxState",
        over      = b.nodeState_dt,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"upper": d.p_state_upper},
    )


def _eq_state_profiles(b: _Build) -> None:
    """Profile limits of the storage state."""
    d = b.d
    profiles = [(idx, name, sense) for idx, name, sense in (
        (d.node_profile_upper, "profile_state_upper_limit", "<="),
        (d.node_profile_lower, "profile_state_lower_limit", ">="),
        (d.node_profile_fixed, "profile_state_fixed",       "=="),
    ) if _has_rows(idx)]
    if not (b.has_storage and profiles and d.p_profile_value is not None):
        return
    m = b.m

    # ─── EQ. ─── profile_state_{upper_limit,lower_limit,fixed} ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − profile[f,d,t]·av[n,d,t]·invested[n,d]
    #   + profile[f,d,t]·av[n,d,t]·retired[n,d]
    #   ≤ / ≥ / =  profile[f,d,t] · av[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ (n,f) ∈ node_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds the storage state by a profile times the built storage capacity, in state units.
    # av = node availability, 1 for nodes without availability data.  The invest/divest terms are
    # restricted to the (n,f) pairs of the method's set.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    state_unitsize = d.p_state_unitsize.frame.rename({"value": "us"})
    existing_count = Param(("n", "d"), b.cap.existing_state_n.frame.rename({"value": "cap"})
                           .join(state_unitsize, on="n", how="inner")
                           .with_columns(value=pl.col("cap") / pl.col("us"))
                           .select("n", "d", "value"))
    for idx, name, sense in profiles:
        over = idx.join(d.dt, how="cross")
        rhs = d.p_profile_value * existing_count
        avail = (_availability_factor(d, idx, axis="n", avail=d.p_node_availability)
                 if d.p_node_availability is not None else None)
        if avail is not None:
            rhs = rhs * avail
        lhs: dict = {"state": b.v_state}
        node_profile_pairs = idx.select("n", "f").unique()
        invested, divested = _capacity.node_in_service(b, d.nodeState, "_node_profile")
        if invested is not None:
            term = invested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["invest_neg"] = -Where(term, node_profile_pairs)
        if divested is not None:
            term = divested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["divest"] = Where(term, node_profile_pairs)
        m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"limit": rhs},
        )


def _cyclic_pin_coefficient(b: _Build) -> "Param | None":
    """``(n)``: the share of the capacity a cyclic node's first-roll wrap is pinned at: storage_state_start of the
    fix_start nodes, storage_state_end of the ``wrap_end`` nodes of storage_roll_end (a fix_end node of a rolling
    instance that reaches the model end, without fix_start).  None when neither."""
    d = b.d
    parts = []
    if _has_rows(d.storage_fix_start) and d.p_state_start is not None:
        parts.append(d.p_state_start.frame.join(d.storage_fix_start.select("n"), on="n", how="semi")
                     .select("n", "value"))
    wrap_end = _roll_end_rows(d, ("wrap_end",))
    if wrap_end is not None and d.p_state_end is not None:
        end = d.p_state_end.frame.join(wrap_end.select("n"), on="n", how="semi").select("n", "value")
        if parts:
            end = end.join(parts[0].select("n"), on="n", how="anti")
        parts.append(end)
    parts = [f for f in parts if f.height > 0]
    if not parts:
        return None
    return Param(("n",), pl.concat(parts, how="vertical"))


def _cyclic_start_pin_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the steps storage_state_start_binding_cyclic_period pins (the wrap targets of the cyclic
    fix_start nodes' first steps, and of the ``wrap_end`` nodes of storage_roll_end, less the steps an upper-level
    solve fixes, which are logged); None when there are none."""
    d = b.d
    coef = _cyclic_pin_coefficient(b)
    if not (b.has_storage
            and coef is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None
            and _has_rows(d.nodeState_first_dt)):
        return None
    cyclic = _self_discharge.node_union((d.storage_bind_within_timeblock, d.storage_bind_within_solve,
                                          d.storage_bind_within_period, _self_discharge.block_cyclic_nodes(d)))
    if cyclic is None:
        return None
    nodes = cyclic.join(coef.frame.select(pl.col("n").cast(cyclic.schema["n"], strict=False)), on="n", how="inner")
    nodes = _self_discharge.without_nodes(nodes, _self_discharge.carried_nodes(d))
    if nodes is None:
        return None
    # The step each node's first step starts from: its wrap under the node's own lag (the lags of the state
    # change, _self_discharge.state_links).
    first = d.nodeState_first_dt.join(nodes, on="n", how="inner").select(
        pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    links = _self_discharge.previous_steps(_self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb))
    wrap = (first.join(links, on=["n", "d", "t"], how="inner")
            .select("n", pl.col("d_prev").alias("d"), pl.col("t_prev").alias("t")).unique())
    fixed = _fix_quantity_target(b)
    if fixed is not None:
        # The wrap target is the step an upper-level solve fixes (a nested roll whose horizon is one timeblock, or
        # the solve for bind_within_solve): the two equalities would sit on one variable, and the upper value wins.
        fixed = fixed.select(pl.col("n").cast(pl.Utf8).alias("_n"), pl.col("d").cast(pl.Utf8).alias("_d"),
                             pl.col("t").cast(pl.Utf8).alias("_t"))
        clash = wrap.rename({"n": "_n", "d": "_d", "t": "_t"}).join(fixed, on=["_n", "_d", "_t"], how="semi")
        if clash.height > 0:
            _LOG.info("storage fix_start not pinned where the upper-level solve fixes the state at the same step: %s",
                      ", ".join(f"{n} {dd} {tt}" for n, dd, tt in clash.sort("_n", "_d", "_t").iter_rows()))
            wrap = wrap.join(clash.rename({"_n": "n", "_d": "d", "_t": "t"}), on=["n", "d", "t"], how="anti")
    over = (b.nodeState_dt
            .with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(wrap.rename({"n": "_n", "d": "_d", "t": "_t"}), on=["_n", "_d", "_t"], how="semi")
            .select("n", "d", "t").unique(maintain_order=True))
    return over if over.height > 0 else None


def _eq_storage_start_binding_cyclic(b: _Build) -> None:
    """Start state of fix_start storage nodes whose first step wraps to a later step of the solve."""
    d = b.d
    over = _cyclic_start_pin_steps(b)
    if over is None:
        return
    m = b.m

    # ─── EQ. ─── storage_state_start_binding_cyclic_period ───────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d′,t′] − state_start[n]·invested[n,d′]
    #   + state_start[n]·retired[n,d′]
    #   = state_start[n] · capacity[n,d′] / state_unitsize[n]
    #   ∀ n ∈ (storage_fix_start ∪ storage_roll_end(wrap_end)) ∩ (bind_within_timeblock ∪ bind_within_solve
    #     ∪ bind_within_period ∪ looping nodeStateBlock), n ∉ carried, (n,d,t) ∈ nodeState_first_dt,
    #     (d′,t′) = the step (d,t) starts from (the node's wrap), (n,d′,t′) ∉ the steps
    #     node_balance_fix_quantity_eq_lower fixes
    # The first step of a cyclic node starts from its wrap target, so the start state is pinned there, to the
    # share state_start of the built storage capacity (state units): the last step of the first timeblock
    # (bind_within_timeblock), of the solve (bind_within_solve) or of the first period (bind_within_period, also
    # bind_within_timeblock in a rolling solve, and bind_intraperiod_blocks; in a roll, of its part in the roll).  Only where no state of the node was
    # realised before this solve (the first solve of the model and of each nested level); a later solve neither
    # pins nor carries a cyclic node (it cycles within the solve; a later roll of a rolling bind_within_solve node
    # chains forward from the carry instead, _run_model._anchor_cyclic_in_rolling).  Where an
    # upper-level solve fixes the state at the wrap target the upper value wins (the two equalities would sit on
    # one variable).  A bind_intraperiod_blocks node that chains forward (fix_start_end, or a roll that holds its
    # first period only in part) has its start in nodeBalanceBlock_eq (_fix_start_terms); fix_start_end nodes of the
    # other cyclic methods are forward-only in the solve (_run_model._fix_start_end_chains_forward).
    # In a rolling instance that reaches the model end, a fix_end node without fix_start is pinned the same way at
    # storage_state_end (state_start below stands for that share; storage_roll_end kind wrap_end): its cycle level
    # is the end value, as in a solve that is not rolling, where the end pin sits on the wrap target.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    coef = _cyclic_pin_coefficient(b)
    lhs: dict = {"state": b.v_state}
    invested, divested = _capacity.node_in_service(b, d.nodeState, "_cyclic", coef=coef)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_start_binding_cyclic_period",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": (coef * b.cap.existing_state_n) / d.p_state_unitsize},
    )


def _utf8_cells(frame: "pl.DataFrame | None", cols=("n", "d", "t")) -> pl.DataFrame:
    """``frame``'s ``cols`` as Utf8 (an empty frame for None)."""
    if frame is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
    return frame.select([pl.col(c).cast(pl.Utf8) for c in cols])


def _state_cells_typed(b: _Build, frame: pl.DataFrame, rename: dict) -> pl.DataFrame:
    """Utf8 ``frame`` cast onto v_state's dtypes; the columns of ``rename`` ({column: v_state dim}) take that dim's
    dtype."""
    schema = b.v_state.frame.schema
    return frame.with_columns([pl.col(c).cast(schema[rename.get(c, c)], strict=False) for c in frame.columns])


def _state_keys_typed(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` (n, d, t, value) with its keys cast onto v_state's dtypes."""
    schema = b.v_state.frame.schema
    return frame.with_columns([pl.col(c).cast(schema[c], strict=False) for c in ("n", "d", "t")])


def _notice_stochastic_storage_paths(names) -> None:
    from flextool.engine import _behaviour_changes
    _behaviour_changes.notice("stochastic_storage_paths_tied", "nodes", sorted(set(str(n) for n in names)))


def _eq_storage_within_solve_path_end(b: _Build) -> None:
    """bind_within_solve in a stochastic solve: the last step of every path ends at the same level."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_bind_within_solve)
            and _has_rows(d.period_last)
            and _has_rows(d.nodeState_first_dt)
            and _has_rows(d.nodeState_last_dt)):
        return
    last_periods = sorted(set(str(x) for x in d.period_last["d"].to_list()))
    if len(last_periods) < 2:
        return

    # ─── EQ. ─── storage_within_solve_path_end ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t_last(d)] − v_state[n,d*,t_last(d*)] = 0
    #   ∀ n ∈ bind_within_solve, d ∈ period_last, d ≠ d*, (n,d,t_last(d)) and (n,d*,t_last(d*)) ∈ nodeState_last_dt,
    #     unless both ends are already set (each is a step storage_state_start_binding_cyclic_period pins or
    #     node_balance_fix_quantity_eq_lower fixes)
    #   d* = the last period the solve's first steps wrap to (the state-change lag of bind_within_solve,
    #     _self_discharge.state_links): the last period key of the solve, the trunk's wrap target
    # bind_within_solve closes one loop per path of a stochastic solve.  The first step of the solve continues from
    # the last step of d* only, so this row ends every other path (period_last: the solve's last period and its
    # forecast branch siblings) at the same level: every branch's end is the level the trunk starts from.  With a
    # fan in the solve's first period each root forecast's first step continues from the last step of its own path
    # (dtttdt_within_solve), so equal ends also give all the forecasts the same start.  A deterministic solve has
    # one last period and no rows.  A fix_start node keeps the rows where only one end is pinned (a fan after the
    # trunk period: the pin sits at d*'s end); with both ends pinned or fixed the row would only repeat (or
    # contradict) those values.
    ws_links = [link for link in _self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb)
                if link.key == "state_change_ws"]
    if not ws_links:
        return
    wraps = (_utf8_cells(d.nodeState_first_dt)
             .join(_self_discharge.previous_steps(ws_links), on=["n", "d", "t"], how="inner"))
    targets = sorted(set(wraps["d_prev"].to_list()) & set(last_periods))
    if not targets:
        return
    # Convention: d* is the last wrap target in sorted name order (the trunk wraps to the last period key; a root
    # of a first-period fan to the leaf its dtttdt_within_solve picks, also by sorted name, _derived.leaf_of).
    d_star = targets[-1]
    ends = (_utf8_cells(d.nodeState_last_dt)
            .join(_utf8_cells(d.storage_bind_within_solve, ("n",)), on="n", how="semi")
            .filter(pl.col("d").is_in(last_periods)))
    star_ends = ends.filter(pl.col("d") == d_star).select("n", d_end="d", t_end="t")
    over = (ends.filter(pl.col("d") != d_star)
            .join(star_ends, on="n", how="inner")
            .select("n", "d", "t", "d_end", "t_end"))
    over = _path_end_rows(b, over)
    if over is None:
        return
    b.m.add_cstr(
        "storage_within_solve_path_end",
        over      = over,
        sense     = "==",
        lhs_terms = _path_end_terms(b),
        rhs_terms = {},
    )
    _notice_stochastic_storage_paths(over["n"].to_list())


def _path_end_rows(b: _Build, over: pl.DataFrame) -> "pl.DataFrame | None":
    """The rows (n, d, t, d_end, t_end; Utf8) of a path-end tie, less those whose both ends are already set (a step
    storage_state_start_binding_cyclic_period pins or node_balance_fix_quantity_eq_lower fixes), on v_state's
    dtypes; None when none are left."""
    set_parts = [_utf8_cells(f) for f in (_cyclic_start_pin_steps(b), _fix_quantity_target(b)) if f is not None]
    if set_parts and over.height > 0:
        already = pl.concat(set_parts, how="vertical").unique()
        both = (over.join(already, on=["n", "d", "t"], how="semi")
                .join(already.rename({"d": "d_end", "t": "t_end"}), on=["n", "d_end", "t_end"], how="semi"))
        over = over.join(both, on=list(over.columns), how="anti")
    if over.height == 0:
        return None
    return _state_cells_typed(b, over.sort("n", "d"), {"d_end": "d", "t_end": "t"})


def _path_end_terms(b: _Build) -> dict:
    """v_state[n,d,t] − v_state[n,d_end,t_end]: the LHS of a path-end tie."""
    return {"state": b.v_state, "state_end": -_var_view(b.v_state, "__path_end", {"d": "d_end", "t": "t_end"})}


def _eq_storage_split_path_end(b: _Build) -> None:
    """bind_within_period and looping bind_intraperiod_blocks in a split realization roll: the period's part in the
    roll closes once per path."""
    d = b.d
    if not (b.has_storage and _has_rows(d.nodeState_last_dt)):
        return
    lab = labels_with_steps(d)
    if not lab:
        return

    # ─── EQ. ─── storage_split_path_end ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,p_f,t_last(p_f)] − v_state[n,d*,t_last(d*)] = 0
    #   ∀ n ∈ bind_within_period ∪ looping nodeStateBlock, p ∈ L* (a period label with steps: a split realization
    #     roll), p_f ∈ B(p), p_f ≠ d*, whose first step of p wraps to (d*, t_last(d*)) (the seam lag,
    #     _self_discharge.state_links), (n,p_f,t_last(p_f)) ∈ nodeState_last_dt; unless both ends are already set
    #     (pinned by storage_state_start_binding_cyclic_period or fixed by node_balance_fix_quantity_eq_lower)
    #   d* = the last branch period of p in name order
    # A split realization roll holds the period p as one path per forecast: p's jump steps, then the steps of one
    # branch period p_f.  The state cycles within the period's part in the roll, so each path closes on itself: the
    # first step of p continues from the end of every path.  The lag is single-valued (each p_f's first step
    # continues from p's last jump step, and p's first step from the end of d*), so this row ends every other path
    # at the level d* ends at: every path's end is the level p starts from.  A roll that continues the period
    # from the carry has no wrap at p's first step (scope_carry) and closes or anchors each path's end instead
    # (storage_state_roll_close / storage_state_roll_anchor at each p_f's last step).
    links = [link for link in _self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb)
             if link.key in ("state_change_wp_seam", "state_change_block")]
    if not links:
        return
    branch = pl.DataFrame([(p, f) for p, fs in lab.items() for f in fs], schema={"d": pl.Utf8, "d_prev": pl.Utf8},
                          orient="row")
    wraps = _self_discharge.previous_steps(links).join(branch, on=["d", "d_prev"], how="semi")
    ends = _utf8_cells(d.nodeState_last_dt)
    over = (wraps.select("n", "d", pl.col("d_prev").alias("d_end"), pl.col("t_prev").alias("t_end"))
            .join(branch.rename({"d_prev": "f"}), on="d", how="inner")
            .filter(pl.col("f") != pl.col("d_end"))
            .join(ends.rename({"d": "f"}), on=["n", "f"], how="inner")
            .select("n", pl.col("f").alias("d"), "t", "d_end", "t_end").unique())
    over = _path_end_rows(b, over)
    if over is None:
        return
    b.m.add_cstr(
        "storage_split_path_end",
        over      = over,
        sense     = "==",
        lhs_terms = _path_end_terms(b),
        rhs_terms = {},
    )


def _eq_storage_state_fan_start(b: _Build) -> None:
    """bind_forward_only with a free start: every forecast of a fan in the solve's first period starts at the
    same level."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_bind_forward_only)
            and _has_rows(d.nodeState_first_dt)):
        return
    roots = sorted(set(str(x) for x in d.nodeState_first_dt["d"].to_list()))
    if len(roots) < 2:
        return
    m = b.m

    # ─── EQ. ─── storage_state_fan_start ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t_first(d)] − v_state[n,d″,t_first(d″)] = 0
    #   ∀ n ∈ bind_forward_only, (n,d,t_first(d)) and (n,d″,t_first(d″)) ∈ nodeState_first_dt with no start term
    #     (n ∉ storage_fix_start ∪ carried), d = the first root (sorted), d″ ∈ the other roots, one row
    #     per group of roots that non_anticipativity_storage_state_first does not already tie to d's group
    #     (role uc_only: that row is absent, one row per other root)
    #   roots = the periods of nodeState_first_dt (built from period_first_of_solve): the forecast branch periods
    #     of a fan in the solve's first period; one root without a fan: no rows
    # The first step of a bind_forward_only node has no state change (dtttdt_forward_only): without fix_start or a
    # carried state it is free, so each forecast of a fan in the first period could pick its own start level from
    # what it later sees.  The fan has only forecasts (no realized member), so this row is what makes their start
    # the same.  A fix_start or carried node starts every root from the same value already (_fix_start_terms,
    # the roll carry).  For a node of an include_stochastics group, non_anticipativity_storage_state_first may tie
    # the first step of a root to its reference forecast already (a tie window shorter than the root period):
    # those pairs are left out, read from that row's domain.  Solve (1) of separate_dispatch (the uc_only role,
    # _non_anticipativity.na_role) has no such rows: every root pair of a free start is tied here.
    first = (_utf8_cells(d.nodeState_first_dt)
             .join(_utf8_cells(d.storage_bind_forward_only, ("n",)), on="n", how="semi"))
    for started in (_self_discharge.fix_start_first(d), _self_discharge.carried_nodes(d)):
        if started is not None:
            first = first.join(_utf8_cells(started, ("n",)).unique(), on="n", how="anti")
    if first.height == 0:
        return
    ref = first.filter(pl.col("d") == roots[0])
    other = first.filter(pl.col("d") != roots[0]).select("n", b="d", t_b="t")
    over = ref.join(other, on="n", how="inner").select("n", "d", "t", "b", "t_b")
    tied = _non_anticipativity.storage_state_first_cells(b)
    if tied is not None and over.height > 0:
        edges = (_utf8_cells(tied, ("n", "d", "b", "t"))
                 .join(first, on=["n", "d", "t"], how="semi")
                 .join(first.rename({"d": "b"}), on=["n", "b", "t"], how="semi"))
        if edges.height > 0:
            keep = []
            for node, group in over.group_by("n", maintain_order=True):
                parent: dict[str, str] = {}

                def find(x: str) -> str:
                    while parent.get(x, x) != x:
                        x = parent[x]
                    return x

                for x, y in edges.filter(pl.col("n") == node[0]).select("d", "b").iter_rows():
                    parent[find(x)] = find(y)
                for row in group.sort("b").iter_rows(named=True):
                    if find(row["b"]) != find(roots[0]):
                        parent[find(row["b"])] = find(roots[0])
                        keep.append(row)
            over = pl.DataFrame(keep, schema=over.schema, orient="row")
    if over.height == 0:
        return
    over = _state_cells_typed(b, over.sort("n", "b"), {"b": "d", "t_b": "t"})
    m.add_cstr(
        "storage_state_fan_start",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": b.v_state,
                     "state_root": -_var_view(b.v_state, "__fan_start", {"d": "b", "t": "t_b"})},
        rhs_terms = {},
    )
    _notice_stochastic_storage_paths(over["n"].to_list())


def _end_binding_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the steps storage_state_end_binding pins (the last step of the entire model, of the fix_end /
    fix_start_end nodes, less the steps an upper-level fix_quantity / fix_usage fixes); None when there are none."""
    d = b.d
    if not (b.has_storage
            and d.storage_solve_at_model_end
            and _has_rows(d.storage_fix_end)
            and d.p_state_end is not None
            and _has_rows(d.nodeState_last_dt)
            and d.period_last is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return None
    over = (d.nodeState_last_dt
            .join(d.period_last, on="d", how="inner")
            .join(d.storage_fix_end, on="n", how="inner")
            .select("n", "d", "t").unique())
    if _has_rows(d.nodeState_rp):
        # A representative-period node's end is the end of its base-period chain (rp_inter_period_end_fix), not a
        # state inside the last representative block.
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
    # An upper-level storage fix at the step wins (fix_quantity: a second equality on the same state; fix_usage: with
    # the end pinned the usage cap can be infeasible).  A nested fix_price does not conflict and keeps the pin.
    over = _without_steps(over, _nested_fixed_end(b, with_price=False))
    over = over.sort("n", "d", "t")
    return over if over.height > 0 else None


def _eq_storage_end_binding(b: _Build) -> None:
    """End state of fix_end / fix_start_end storage nodes at the last step of the entire model, in every sub-solve
    whose horizon holds that step."""
    d = b.d
    over = _end_binding_steps(b)
    if over is None:
        return

    # ─── EQ. ─── storage_state_end_binding ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_end[n]·invested[n,d]
    #   + state_end[n]·retired[n,d]
    #   = state_end[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_end, (n,d,t) ∈ nodeState_last_dt, d ∈ period_last, storage_solve_at_model_end,
    #     (n,d,t) ∉ the steps node_balance_fix_quantity_eq_lower / node_storage_usage_fix_le fix
    # Pins the state at the end of the last step of the entire model to the share state_end of the built storage
    # capacity (state units), in every sub-solve whose horizon holds that step (storage_solve_at_model_end,
    # _carry_routing.reaches_model_end: e.g. both an investment solve and the last dispatch roll; each nested level's
    # solve that holds it; a roll whose look-ahead runs to it); each last period (each terminal branch of a
    # stochastic solve) ends there.  storage_fix_end = fix_end ∪ fix_start_end.  Where an
    # upper-level solve fixes the quantity or caps the usage at that step the upper value wins; a nested fix_price
    # steers the end through the objective and keeps the pin.  A fix_start_end node chains forward whatever its
    # storage_binding_method (_run_model._fix_start_end_chains_forward, _self_discharge.block_forward_nodes), so its
    # start is in the first step's balance and never on this state; a cyclic node is pinned here by fix_end only.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    m = b.m
    lhs: dict = {"state": Where(b.v_state, over)}
    invested, divested = _capacity.node_in_service(b, d.storage_fix_end, "_end", coef=d.p_state_end)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_end_binding",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": (d.p_state_end * b.cap.existing_state_n) / d.p_state_unitsize},
    )


def _end_match_nodes(b: _Build) -> "tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]":
    """``(free_first, share_first, ends)`` of the matching end of forward chains without representative periods.

    free_first (n, d, t): the first steps of the bind_forward_only nodes whose chain starts free in this solve
    (no fix_start, no carry); share_first (n, d, t): those of the fix_start nodes the solve can invest in during the
    first period (not with fix_start_end at state_end ≥ state_start); ends (n, d, t): the last step of each last
    period of those nodes.  Only in a solve that holds both the start of the chain (not carried: the first solve of
    the model and of each nested level) and the last step of the model (storage_solve_at_model_end), and not for a
    node an upper-level solve fixes.  Each None when empty.
    """
    d = b.d
    none = (None, None, None)
    # The first roll of a rolling solve that does not reach the end of the model: its free-start nodes get the row
    # at its horizon end (storage_roll_end kind match_first, _run_model._anchor_cyclic_in_rolling).
    first_roll = _roll_end_rows(d, ("match_first",))
    if not (b.has_storage and (d.storage_solve_at_model_end or first_roll is not None)
            and _has_rows(d.storage_bind_forward_only)
            and _has_rows(d.nodeState_last_dt) and _has_rows(d.period_last) and _has_rows(d.nodeState_first_dt)):
        return none
    forward = d.storage_bind_forward_only.select(pl.col("n").cast(pl.Utf8)).unique()
    if not d.storage_solve_at_model_end:
        forward = forward.join(first_roll.select(pl.col("n").cast(pl.Utf8)), on="n", how="semi")
    nested = _nested_nodes(b, with_price=True)
    if nested is not None:
        forward = forward.join(nested, on="n", how="anti")
    if _has_rows(d.nodeState_rp):
        forward = forward.join(d.nodeState_rp.select(pl.col("n").cast(pl.Utf8)), on="n", how="anti")
    if forward.height == 0:
        return none

    def restrict(first) -> "pl.DataFrame | None":
        if first is None:
            return None
        out = first.filter(pl.col("n").cast(pl.Utf8).is_in(forward["n"].implode()))
        return out if out.height > 0 else None

    free_first = restrict(_self_discharge.open_first(d))
    share_first = restrict(_self_discharge.fix_start_first(d)) if d.storage_solve_at_model_end else None
    if share_first is not None:
        periods = sorted(set(str(x) for x in share_first["d"].to_list()))
        investable = _invest_at_start_nodes(b, share_first.select("n").unique(), periods)
        if _has_rows(d.storage_fix_end) and d.p_state_end is not None and d.p_state_start is not None:
            shares = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                      .join(d.p_state_end.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("e")),
                            on="n"))
            fix_end = set(str(n) for n in d.storage_fix_end["n"].to_list())
            investable -= {n for n, s, e in shares.iter_rows() if n in fix_end and e >= s - 1e-12}
        share_first = share_first.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(investable)))
        share_first = share_first if share_first.height > 0 else None
    names = set()
    for f in (free_first, share_first):
        if f is not None:
            names |= set(str(n) for n in f["n"].to_list())
    if not names:
        return none
    ends = (d.nodeState_last_dt.join(d.period_last, on="d", how="inner")
            .filter(pl.col("n").cast(pl.Utf8).is_in(sorted(names)))
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return free_first, share_first, (ends if ends.height > 0 else None)


def _eq_storage_end_match(b: _Build) -> None:
    """The matching end of forward chains (no representative periods) whose start is a decision of the solve."""
    d = b.d
    free_first, share_first, ends = _end_match_nodes(b)
    if ends is None:
        return
    m = b.m
    end_view = _var_view(b.v_state, "__end_match", {"d": "d_end", "t": "t_end"})
    ends = ends.rename({"d": "d_end", "t": "t_end"})

    # ─── EQ. ─── storage_state_end_match ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d_end,t_end] − v_state[n,d,t]  ≥  0
    #   ∀ n ∈ bind_forward_only, (n,d,t) ∈ nodeState_first_dt with a free start (no fix_start, n ∉ carried),
    #     (n,d_end,t_end) ∈ nodeState_last_dt, d_end ∈ period_last, storage_solve_at_model_end
    #     (or n ∈ storage_roll_end(match_first): the first roll of a rolling solve), no upper-level fix on n
    # The matching end: a forward chain that starts free (the solve chooses the level of the first step, which has
    # no state change) ends the model at least at that level, so the solve cannot start full and end empty with
    # energy it never produced.  An inequality: a start of 0 always meets it, so it does not force slack by
    # itself.  fix_end and use_reference_value still set the end; with a free start this bounds the start by that
    # end.  With a stochastic fan in the first period every root start is tied (storage_state_fan_start), so each
    # last step is held to it.  In a rolling solve the first roll gets the row at its horizon end, so it does not
    # choose its start for free while the rolls reaching the end of the model return it (storage_state_roll_match).
    if free_first is not None:
        over = (free_first.select("n", "d", "t")
                .join(ends.with_columns(pl.col("n").cast(free_first.schema["n"], strict=False)), on="n", how="inner")
                .sort("n", "d", "t", "d_end", "t_end"))
        if over.height > 0:
            _notice_nodes("storage_end_matches_start", over.select("n").unique(), b)
            m.add_cstr(
                "storage_state_end_match",
                over      = over,
                sense     = ">=",
                lhs_terms = {"state_end": Where(end_view, over), "neg_state_start": -Where(b.v_state, over)},
                rhs_terms = {},
            )

    # ─── EQ. ─── storage_state_end_match_share ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d_end,t_end] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]  ≥  0
    #   ∀ n ∈ bind_forward_only ∩ storage_fix_start, n ∉ carried, (n,d,t) ∈ nodeState_first_dt, the solve can
    #     invest in n in d (nd_invest_set), not fix_start_end with state_end ≥ state_start; (n,d_end,t_end) as above
    # The matching end of a fix_start start on capacity the solve can invest in: the start on the capacity the solve
    # adds in the first period (state units) comes back by the end of the model, so storage is not built as a fuel
    # tank whose fix_start share is spent once.  The start on existing capacity is the user's endowment and may be
    # used.  No investment always meets it.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if share_first is not None and d.p_state_start is not None:
        over = (share_first.select("n", "d", "t")
                .join(ends.with_columns(pl.col("n").cast(share_first.schema["n"], strict=False)), on="n", how="inner")
                .sort("n", "d", "t", "d_end", "t_end"))
        if over.height > 0:
            nodes = over.select("n").unique()
            lhs: dict = {"state_end": Where(end_view, over)}
            invested, divested = _capacity.node_in_service(b, nodes, "_end_match", coef=d.p_state_start)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            _notice_nodes("storage_end_matches_start", nodes, b)
            m.add_cstr(
                "storage_state_end_match_share",
                over      = over,
                sense     = ">=",
                lhs_terms = lhs,
                rhs_terms = {},
            )


def _roll_horizon_ends(b: _Build, nodes: pl.DataFrame) -> "pl.DataFrame | None":
    """``(n, d, t)`` of ``nodes`` (n): the last step of each last period (the roll's horizon end; each terminal
    branch of a stochastic roll); None when there are none."""
    d = b.d
    if not (_has_rows(d.nodeState_last_dt) and _has_rows(d.period_last)):
        return None
    over = (d.nodeState_last_dt.join(d.period_last, on="d", how="inner")
            .join(nodes.select(pl.col("n").cast(d.nodeState_last_dt.schema["n"], strict=False)), on="n", how="semi")
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return over if over.height > 0 else None


def _first_scope_ends(b: _Build, nodes: pl.DataFrame) -> "pl.DataFrame | None":
    """``(n, d, t)`` of ``nodes`` (n): where the first scope of a rolling roll ends within the roll — the step the
    roll's first step would wrap to: the last step of the first period in the roll (bind_within_period, also
    bind_within_timeblock, which a rolling solve uses as bind_within_period, and bind_intraperiod_blocks); in a
    split realization roll the last step of each path (each branch period of the label).  None when there are
    none."""
    pairs = scope_end_pairs(b.d)   # the pairs G3 checks (preprocess._step_relations)
    if pairs is None:
        return None
    names = nodes.select(pl.col("n").cast(pl.Utf8)).unique()
    utf = pairs.join(names, on="n", how="semi").select("n", d="d_end", t="t_end").unique()
    if utf.height == 0:
        return None
    over = (b.nodeState_dt
            .with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(utf.rename({"n": "_n", "d": "_d", "t": "_t"}), on=["_n", "_d", "_t"], how="semi")
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return over if over.height > 0 else None


def _do_nothing_level(b: _Build, start: pl.DataFrame, ends: pl.DataFrame) -> pl.DataFrame:
    """``(n, d, t, value)`` (Utf8 keys, MWh): the anchor level of each horizon end of ``ends`` — the lower of the
    carried level ``start`` (n, value) and the level the storage would reach from it by doing nothing.

    Doing nothing: no flow in or out, only the node's own ``inflow`` (a negative one is a forced outflow) and its
    self-discharge, step by step along the forward chain from the roll's first step (dtttdt_forward_only):
    S_t = (S⁻·(1 − β_t) + inflow_t) / (1 + α_t), clipped to [0, existing capacity] (the balance rows'
    self-discharge weights, _self_discharge).  That level is reachable by construction, so the anchor never forces
    slack; a lossless node without inflow keeps the carried level.
    """
    d = b.d
    start = start.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
    ends = ends.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    out = ends.join(start, on="n", how="inner")
    names = set(start["n"].to_list())
    inflow = None
    if d.p_inflow is not None and _has_rows(d.p_inflow.frame) and "t" in d.p_inflow.frame.columns:
        inflow = (d.p_inflow.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                          pl.col("t").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
                  .filter(pl.col("n").is_in(names) & (pl.col("value") != 0.0)))
    loss = None
    if b.self_discharge is not None:
        loss = (b.self_discharge.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                              pl.col("t").cast(pl.Utf8), "alpha", "beta")
                .filter(pl.col("n").is_in(names)))
    moving = set()
    for frame in (inflow, loss.filter((pl.col("alpha") > 0) | (pl.col("beta") > 0)) if loss is not None else None):
        if frame is not None:
            moving |= set(frame["n"].to_list())
    lag = d.dtttdt_forward_only
    if not moving or not _has_rows(lag) or not _has_rows(d.nodeState_first_dt):
        return out
    nxt: dict = {}
    for dd, tt, dp, tp in lag.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                                     pl.col("d_previous").cast(pl.Utf8),
                                     pl.col("t_previous_within_solve").cast(pl.Utf8)).iter_rows():
        nxt.setdefault((dp, tp), []).append((dd, tt))
    e_of = {} if inflow is None else {(n, dd, tt): v for n, dd, tt, v in inflow.iter_rows()}
    ab_of = {} if loss is None else {(n, dd, tt): (a, bb) for n, dd, tt, a, bb in loss.iter_rows()}
    cap_of: dict = {}
    if b.cap.existing_state_n is not None:
        cap_of = {(n, dd): v for n, dd, v in b.cap.existing_state_n.frame.select(
            pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("value").cast(pl.Float64)).iter_rows()}
    first = d.nodeState_first_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    carry = dict(start.iter_rows())
    level: dict = {}
    for node in sorted(moving & names):
        roots = [(dd, tt) for dd, tt in first.filter(pl.col("n") == node).select("d", "t").iter_rows()]
        queue = [(step, carry[node]) for step in roots]
        while queue:
            (dd, tt), prev = queue.pop()
            a, bb = ab_of.get((node, dd, tt), (0.0, 0.0))
            now = (prev * (1.0 - bb) + e_of.get((node, dd, tt), 0.0)) / (1.0 + a)
            now = min(max(now, 0.0), cap_of.get((node, dd), float("inf")))
            level[(node, dd, tt)] = now
            queue.extend((step, now) for step in nxt.get((dd, tt), ()))
    if not level:
        return out
    held = pl.DataFrame([(n, dd, tt, v) for (n, dd, tt), v in level.items()],
                        schema={"n": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "_hold": pl.Float64}, orient="row")
    return (out.join(held, on=["n", "d", "t"], how="left")
            .with_columns(value=pl.min_horizontal("value", pl.col("_hold").fill_null(pl.col("value"))))
            .drop("_hold"))


def _capped_at_capacity(b: _Build, level: pl.DataFrame) -> pl.DataFrame:
    """``level`` (n, d, t, value MWh, Utf8 keys) lowered to the storage capacity in service in the row's period
    (existing, MWh; owner decision: a closure keeps the start energy in MWh but never asks for more than the period
    can hold)."""
    cap = b.cap.existing_state_n
    if cap is None:
        return level
    return (level.join(cap.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                        pl.col("value").cast(pl.Float64).alias("_cap")), on=["n", "d"], how="left")
            .with_columns(value=pl.min_horizontal("value", pl.col("_cap").fill_null(pl.col("value"))))
            .drop("_cap"))


def _eq_storage_roll_end(b: _Build) -> None:
    """The horizon end of the bind_within_solve nodes of a rolling roll that continues from a carried state."""
    d = b.d
    if not (b.has_storage and _has_rows(getattr(d, "storage_roll_end", None)) and d.p_state_unitsize is not None):
        return
    m = b.m
    pinned_end = _end_binding_steps(b)

    def rows(kinds, *, with_price: bool):
        nodes = _roll_end_rows(d, kinds)
        if nodes is None:
            return None, None
        over = (_first_scope_ends(b, nodes) if all(k.startswith("scope_") for k in kinds)
                else _roll_horizon_ends(b, nodes))
        if over is None:
            return None, None
        over = _without_steps(_without_steps(over, _nested_fixed_end(b, with_price=with_price)),
                              None if pinned_end is None else _utf8_cells(pinned_end))
        return (over, nodes) if over.height > 0 else (None, None)

    # ─── EQ. ─── storage_state_roll_anchor ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] ≥ min(carry[n], hold[n,d,t])
    #   ∀ n ∈ storage_roll_end(anchor), (n,d,t) ∈ nodeState_last_dt, d ∈ period_last;
    #     n ∈ storage_roll_end(scope_anchor), (n,d,t) = the end of the roll's first scope (_first_scope_ends: the
    #     last step of the first period in the roll; in a split realization roll the last step of each path, i.e. of
    #     each branch period of the label);
    #     (n,d,t) ∉ the steps an upper-level solve fixes (fix_quantity, fix_usage, fix_price),
    #     (n,d,t) ∉ the steps storage_state_end_binding pins
    #   carry = the level the roll starts from (MWh, roll_continue_state); hold = the level reached from it by doing
    #   nothing (_do_nothing_level: inflow and self-discharge only, clipped to [0, existing capacity])
    # A bind_within_solve node of a rolling solve continues each roll from the level realised by the previous roll
    # (the cascade rewrites it to bind_forward_only for the roll, _run_model._anchor_cyclic_in_rolling), and a roll
    # whose horizon does not reach the end of its solve ends at least at that level: each roll's plan is cyclic on
    # its own start, as the first roll's wrap is.  An inequality, so that it can always be reached: storing more has
    # a cost, so a lossless node ends at the carried level.  Where an upper-level solve fixes, caps or prices the
    # state at the step, that replaces the anchor; fix_end (at the last step of the model) wins.  The same per
    # period (bind_within_period, bind_intraperiod_blocks; bind_within_timeblock in a rolling solve is
    # bind_within_period, _run_model._timeblock_binds_within_period_in_roll): a roll that
    # starts inside a period an earlier roll began continues it from the carry (scope_carry: its first step has no
    # wrap) and anchors the period's part in the roll; for bind_intraperiod_blocks also the roll that begins a period
    # at its first step and holds it in part (its blocks chain forward from the carry).
    over, nodes = rows(("anchor",), with_price=True)
    scope_over, scope_nodes = rows(("scope_anchor",), with_price=True)
    if scope_over is not None:
        # A bind_within_period / bind_intraperiod_blocks node whose first period continues from the carry: the
        # anchor sits where that period's part in the roll ends.
        over = scope_over if over is None else pl.concat([over, scope_over.cast(over.schema)]).unique()
        nodes = scope_nodes if nodes is None else pl.concat([nodes, scope_nodes.cast(nodes.schema)])
    if over is not None:
        level = _do_nothing_level(b, nodes, over)
        m.add_cstr(
            "storage_state_roll_anchor",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"level": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_close ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] = start[n]
    #   ∀ n ∈ storage_roll_end(close), (n,d,t) ∈ nodeState_last_dt, d ∈ period_last;
    #     n ∈ storage_roll_end(scope_close), (n,d,t) = the end of the roll's first scope (_first_scope_ends);
    #     (n,d,t) ∉ the steps an upper-level fix_quantity / fix_usage fixes,
    #     (n,d,t) ∉ the steps storage_state_end_binding pins
    #   start = the level (MWh) the first roll of the instance (scope_close: the roll that began the period)
    #   started from (the cascade records it), at most the storage capacity in service in d (owner
    #   decision: the start energy is kept in MWh, so capacity built later adds none, but a closure never asks
    #   for more than the closing period can hold)
    # A roll whose horizon holds the last step of its solve instance closes the cycle of bind_within_solve: it ends at
    # the level the instance started from, so the realised rolls together return what they took (every roll whose
    # horizon reaches that step, e.g. two when the horizon is not a multiple of the jump).  An upper-level
    # fix_quantity / fix_usage at the step replaces it (that target carries the upper solve's closure); a nested
    # fix_price only values the state and keeps it; fix_end wins at the last step of the model.  Per period
    # (bind_within_timeblock in a rolling solve is bind_within_period): the roll that holds the end of a period it
    # continues (or reaches the end of the instance) ends
    # the scope at its start level, so a realised roll boundary inside the scope creates no energy (logic doc
    # Lemma 4).
    over, nodes = rows(("close",), with_price=False)
    scope_over, scope_nodes = rows(("scope_close",), with_price=False)
    if scope_over is not None:
        # The end of a continued period that the roll holds: it ends at the level the period started from in the
        # roll that began it.
        over = scope_over if over is None else pl.concat([over, scope_over.cast(over.schema)]).unique()
        nodes = scope_nodes if nodes is None else pl.concat([nodes, scope_nodes.cast(nodes.schema)])
    if over is not None:
        level = _capped_at_capacity(b, over.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                                    .join(nodes.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64)),
                                          on="n"))
        m.add_cstr(
            "storage_state_roll_close",
            over      = over,
            sense     = "==",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"start": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_match ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] ≥ min(start[n], capacity[n,d])
    #   ∀ n ∈ storage_roll_end(match) ∖ nodeState_rp, (n,d,t) ∈ nodeState_last_dt, d ∈ period_last,
    #     (n,d,t) ∉ the steps an upper-level solve fixes, (n,d,t) ∉ the steps storage_state_end_binding pins
    #   start = the level (MWh) the first roll started the node from (free: fix_nothing; the cascade records it)
    # The matching end of a forward chain in a rolling solve or a sequence: a roll (or a later solve) that reaches
    # the end of the model ends a bind_forward_only node whose start the first roll (the first solve) chose at least
    # at that start (storage_state_end_match in a solve that holds the whole model).  An inequality: the first roll's own horizon end is held to its start
    # (storage_state_end_match), so it starts no higher than it can refill.
    over, nodes = rows(("match",), with_price=True)
    if over is not None and _has_rows(d.nodeState_rp):
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
        over = over if over.height > 0 else None
    if over is not None:
        level = _capped_at_capacity(b, over.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                                    .join(nodes.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64)),
                                          on="n"))
        m.add_cstr(
            "storage_state_roll_match",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"start": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_anchor_share / storage_state_roll_close_share ──
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]
    #   (≥ | =) state_start[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_roll_end(anchor_share | close_share), (n,d,t) as above (the anchor's or the closure's
    #     exclusions)
    # The first roll of a nested lower level under an upper-level fix_quantity / fix_usage starts from fix_start
    # (forward, _fix_start_terms, not from its wrap: the wrap target would be the upper target), so where no upper
    # target lies at its horizon end the anchor (≥, a roll that does not reach the end of its instance) or the
    # closure (=) is the fix_start share of the capacity in service.
    if b.cap.existing_state_n is not None and d.p_state_start is not None:
        for kind, name, sense, with_price in (("anchor_share", "storage_state_roll_anchor_share", ">=", True),
                                              ("close_share", "storage_state_roll_close_share", "==", False)):
            over, nodes = rows((kind,), with_price=with_price)
            if over is None:
                continue
            lhs: dict = {"state": Where(b.v_state, over)}
            invested, divested = _capacity.node_in_service(b, nodes.select("n"), f"_{kind}", coef=d.p_state_start)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            m.add_cstr(
                name,
                over      = over,
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = {"target": (d.p_state_start * b.cap.existing_state_n) / d.p_state_unitsize},
            )


def _horizon_end_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the storage steps where the solve horizon ends for the storage_solve_horizon_method — the last
    step of each last period (the solve's last period and its stochastic branch siblings, period_last) and, in a
    rolling solve, the end of a chained bind_intraperiod_blocks tail that a looping period follows
    (_self_discharge.block_tail_ends).  None when there are none."""
    d = b.d
    if not _has_rows(d.nodeState_last_dt):
        return None
    parts = []
    if _has_rows(d.period_last):
        parts.append(d.nodeState_last_dt.join(d.period_last, on="d", how="inner").select("n", "d", "t"))
    tails = _self_discharge.block_tail_ends(d)
    if tails is not None:
        parts.append(tails.select("n", "d", "t"))
    if not parts:
        return None
    out = pl.concat(parts, how="vertical").unique()
    return out if out.height > 0 else None


def _eq_storage_reference_value(b: _Build) -> None:
    """End state of storage nodes with a reference value."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_use_reference_value)
            and d.p_storage_state_reference_value is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return
    ends = _horizon_end_steps(b)
    if ends is None:
        return
    m = b.m
    over = (ends
            .join(d.storage_use_reference_value, on="n", how="inner")
            .select("n", "d", "t").unique())
    if _has_rows(d.nodeState_rp):
        # A representative-period node takes its reference value on the end of its base-period chain
        # (rp_inter_period_reference_value), not on a state inside the last representative block; a cyclic one
        # none (its closure wins).
        rp_ref = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="semi")
        _notice_nodes("rp_storage_chain_pinned", rp_ref.select("n").unique(), b)
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
    if _has_rows(d.nodeStateBlock):
        # A bind_intraperiod_blocks node (in storage_use_reference_value in rolling solves only) is pinned where the
        # roll holds the period in part (period_block_chain): at the roll end and at the end of a chained tail that a
        # looping period follows; where its blocks loop the end is tied to the start.
        chain = [str(x) for x in d.period_block_chain["d"].to_list()] if _has_rows(d.period_block_chain) else []
        is_block = pl.col("n").cast(pl.Utf8).is_in([str(x) for x in d.nodeStateBlock["n"].to_list()])
        over = over.filter(~is_block | pl.col("d").cast(pl.Utf8).is_in(chain))
    # An upper-level storage fix at the step (fix_quantity, fix_usage or fix_price) overrides the horizon-end
    # reference value (how_to/solve_sequences.md, nested solves); the reference stays at every other roll end.
    over = _without_steps(over, _nested_fixed_end(b, with_price=True))
    # A rolling roll that closes a bind_intraperiod_blocks period it continues (storage_roll_end kind scope_close)
    # ends the period at its start level there; the reference stays at the roll's other ends.
    closes = _roll_end_rows(d, ("scope_close",))
    if closes is not None:
        over = _without_steps(over, _utf8_cells(_first_scope_ends(b, closes)))
    if over.height == 0:
        return

    # ─── EQ. ─── storage_state_solve_horizon_reference_value ─────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − reference_value[n,d,t]·invested[n,d]
    #   + reference_value[n,d,t]·retired[n,d]
    #   = reference_value[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_use_reference_value, (n,d,t) ∈ horizon_end,
    #     (n ∈ nodeStateBlock ⇒ d ∈ period_block_chain), (n,d,t) ∉ the steps an upper-level solve fixes
    #   horizon_end = nodeState_last_dt ∩ period_last  ∪  block_tail_end
    # Pins the state at the end of the solve horizon to the reference share of the built storage capacity
    # (state units), as maxState and the start pins count it.  period_last is the solve's last period and its
    # stochastic branch siblings only: an earlier period of a multi-period solve is not a horizon end.  Nodes with a
    # competing end condition are not in storage_use_reference_value; a bind_intraperiod_blocks node is, in a
    # rolling solve, and is pinned where the roll holds the period only in part (there its blocks chain forward
    # instead of looping): at the roll end and at block_tail_end, the last step of a chained part-period tail that
    # a period held whole follows (nothing continues from it, _self_discharge.block_tail_ends).  Where an
    # upper-level solve fixes the node at the step (nested fix_quantity, fix_usage, fix_price) the upper value
    # overrides the reference value: two equalities on one state (fix_quantity), a usage cap the pin could make
    # infeasible (fix_usage), or a price meant to steer that end (fix_price).
    # reference_value on the pinned rows only: as a coefficient of the invest / divest sums it would otherwise span
    # every (n, d_invest, d, t) of the solve.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    rv = (promote_param_to_dt(d.p_storage_state_reference_value, d.dt)
          .join(over.lazy(), on=["n", "d", "t"], how="inner")
          .select("n", "d", "t", "value")
          .collect())
    target = (rv.lazy().rename({"value": "rv"})
              .join(b.cap.existing_state_n.frame.lazy().rename({"value": "exist"}),
                    on=["n", "d"], how="inner")
              .join(d.p_state_unitsize.frame.lazy().rename({"value": "us"}), on="n", how="inner")
              .with_columns(value=pl.col("rv") * pl.col("exist") / pl.col("us"))
              .select("n", "d", "t", "value")
              .collect())
    # A reference value below the start share (fix_start, or the reference at the first step) of a node whose start
    # the solve decides by investing: the matching end (storage_state_end_match_share) stays, as for fix_start_end,
    # and caps the investment; a warning (_warn_reference_below_start).
    _free, share_first, _ends = _end_match_nodes(b)
    if share_first is not None and d.p_state_start is not None:
        share = {str(n): v for n, v in d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), "value").iter_rows()}
        matched = set(str(n) for n in share_first["n"].to_list())
        _warn_reference_below_start(b, {str(n) for n, v in rv.select("n", "value").iter_rows()
                                        if str(n) in matched and v < share[str(n)] - 1e-12})
    lhs: dict = {"state": Where(b.v_state, over)}
    invested, divested = _capacity.node_in_service(b, d.storage_use_reference_value, "_ref_value",
                                                   coef=Param(("n", "d", "t"), rv))
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_solve_horizon_reference_value",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )
