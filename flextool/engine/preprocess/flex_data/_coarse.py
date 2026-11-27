"""Coarse steps of the loaded FlexData: the node / process coarse frames, the coarse-row rule and its
behaviour notice."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from polar_high import Param
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read
from flextool.engine.preprocess.flex_data._container import FlexData


def _realized_steps(sd: Path, flex_data: "FlexData", *, provider: "object | None" = None) -> "pl.DataFrame | None":
    """``(d, t)`` the solve realizes: ``solve_data/realized_dispatch`` — the same source as
    ``_last_realized_step`` (the roll carriers).  Without that key (unit-test synthesis) the output row set
    ``flex_data.realized_dispatch``, which holds the same de-duplicated realized steps."""
    frame = None
    rd_path = sd / "realized_dispatch.csv"
    if _provider_has(provider, "solve_data/realized_dispatch", rd_path):
        try:
            frame = _provider_read(provider, "solve_data/realized_dispatch", rd_path)
        except pl.exceptions.NoDataError:
            frame = None
    if frame is None:
        frame = flex_data.realized_dispatch
    if frame is None or frame.height == 0:
        return None
    d_col = "period" if "period" in frame.columns else "d"
    t_col = "step" if "step" in frame.columns else "t"
    return frame.select(pl.col(d_col).cast(pl.Utf8).alias("d"), pl.col(t_col).cast(pl.Utf8).alias("t"))


def _apply_coarse_frames(flex_data: "FlexData", sd: Path, *, provider: "object | None" = None) -> None:
    """Set ``nodeCoarse`` / ``node_coarse_step`` / ``node_coarse_interior`` from the solve's block layout
    (node -> group, group steps) and its steps (``params._coarse.coarse_node_frames``)."""
    from flextool.engine.preprocess.params._coarse import coarse_node_frames

    layout = flex_data.block_layout
    if layout is None or "default" not in layout.block_step_duration or len(layout.block_step_duration) < 2:
        return
    realized = _realized_steps(sd, flex_data, provider=provider)
    psd = flex_data.p_step_duration
    frames = coarse_node_frames(
        node_group=layout.node_block, group_step=layout.block_step_duration,
        solve_step=float(layout.block_step_duration["default"]), dt=flex_data.dt,
        step_duration=None if psd is None else psd.frame, dtttdt=flex_data.dtttdt,
        rp_block_first=flex_data.rp_block_first, realized=realized, balance_nodes=flex_data.nodeBalance,
        state_nodes=flex_data.nodeState, block_nodes=flex_data.nodeStateBlock)
    if frames is None:
        return
    flex_data.nodeCoarse = frames.nodeCoarse
    flex_data.node_coarse_step = frames.node_coarse_step
    flex_data.node_coarse_interior = frames.node_coarse_interior


def _frame_values(frame: "pl.DataFrame | None", col: str) -> set[str]:
    if frame is None or frame.height == 0 or col not in frame.columns:
        return set()
    return set(frame[col].cast(pl.Utf8).to_list())


def coarse_process_exclusions(flex_data: "FlexData") -> dict[str, str]:
    """Process -> the feature that keeps it at the solve's step (``params._coarse`` module docstring): rows of
    these features are per fine step and couple the flow with other per-step terms."""
    d = flex_data
    out: dict[str, str] = {}

    def add(procs: set[str], why: str) -> None:
        for p in sorted(procs):
            out.setdefault(p, why)

    add(_frame_values(getattr(d, "process_delayed", None), "p"), "it has a delay")
    return out


def _param_frame(param: "Param | None") -> "pl.DataFrame | None":
    return None if param is None else param.frame


def coarse_process_reasons(flex_data: "FlexData", steps: pl.DataFrame) -> dict[str, list[str]]:
    """Coarse process -> why its results can differ from flows at the solve's step (the behaviour notice).

    Exact otherwise: with every per-step parameter constant within each coarse step (or one time-varying limit
    per arc), averaging a fine-feasible flow over each coarse step keeps every row and the cost.  Reasons: a
    variable cost or a price that varies within a coarse step (the fine flows could pick the cheapest steps),
    an efficiency that varies (the most efficient steps), two or more limits that vary (the averaged rows are
    their relaxation), ramp limits (ramps between coarse steps), online variables (units online, start-ups and
    shut-downs once per coarse step), a minimum up / down time (counted in whole coarse steps) and a minimum load
    that varies within a coarse step (the averaged floor is the step-mean floor); with ramp limits too, the start-up /
    shut-down capacity reduction of the coarse step (the mean share over the whole step); reserves (one per coarse
    step while the requirement is checked at every solve step), and with an availability or profile that varies
    within a step, the reserve headroom on its step mean.  Rows under the row rule (:func:`_row_rule_reasons`): a
    capacity-margin / inertia / non-synchronous / flow-group instant-flow / user-constraint / DC power-flow row that
    mixes coarse and fine members stays at the solve's step and holds the coarse flow constant over its step; an
    averaged capacity-margin / non-synchronous / instant-flow row whose demand, inflow, profile or limit varies
    within a coarse step uses its step mean (a peak within the step is not seen)."""
    from flextool.engine.preprocess.params._coarse import varies_within_coarse_steps as varies
    from flextool.engine.preprocess.params._coarse import varying_within_coarse_steps as varying

    d = flex_data
    out: dict[str, list[str]] = {}

    def add(procs: set[str], why: str) -> None:
        for p in sorted(procs):
            out.setdefault(p, []).append(why)

    cost: set[str] = set()
    for param, keys in ((d.p_pssdt_varCost, ("p", "source", "sink")), (d.p_pdt_varCost_source, ("p", "source")),
                        (d.p_pdt_varCost_sink, ("p", "sink")), (d.p_pdt_varCost_process, ("p",))):
        cost |= varies(_param_frame(param), steps, keys)
    add(cost, "its variable cost varies within a coarse step")
    price: set[str] = set()
    for pairs, param, key in ((d.flow_from_commodity_eff, d.p_commodity_price, "c"),
                              (d.flow_from_commodity_noEff, d.p_commodity_price, "c"),
                              (d.flow_to_commodity, d.p_commodity_price, "c"),
                              (d.flow_from_co2_priced, d.p_co2_price, "g"),
                              (getattr(d, "flow_to_co2_priced", None), d.p_co2_price, "g")):
        frame = _param_frame(param)
        if pairs is None or pairs.height == 0 or frame is None or key not in pairs.columns:
            continue
        by_p = (pairs.select(pl.col("p").cast(pl.Utf8), pl.col(key).cast(pl.Utf8)).unique()
                .join(frame.with_columns(pl.col(key).cast(pl.Utf8)), on=key, how="inner"))
        price |= varies(by_p, steps, ("p", key))
    add(price, "the price of a commodity or CO2 it buys or sells varies within a coarse step")
    add(varies(_param_frame(d.p_slope), steps, ("p",)), "its efficiency varies within a coarse step")
    limits: dict[str, int] = {p: 1 for p in varies(_param_frame(d.p_process_availability), steps, ("p",))}
    profile = _param_frame(d.p_profile_value)
    for idx in (d.process_profile_upper, d.process_profile_lower, d.process_profile_fixed):
        if idx is None or idx.height == 0 or profile is None:
            continue
        f_col = "f" if "f" in idx.columns else "profile"
        arcs = idx.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink")),
                          pl.col(f_col).cast(pl.Utf8).alias("f")).unique()
        by_arc = arcs.join(profile.with_columns(pl.col("f").cast(pl.Utf8)), on="f", how="inner")
        for p in varying(by_arc, steps, ("p", "source", "sink", "f"))["p"].to_list():
            limits[p] = limits.get(p, 0) + 1
    add({p for p, n in limits.items() if n >= 2}, "two or more of its limits (availability, profiles) vary "
        "within a coarse step")
    ramp: set[str] = set()
    for k in ("sink_up", "sink_down", "source_up", "source_down"):
        ramp |= _frame_values(getattr(d, f"process_source_sink_ramp_limit_{k}", None), "p")
    add(ramp & _frame_values(steps, "p"), "it has ramp limits (they apply between coarse steps)")
    coarse = _frame_values(steps, "p")
    add(_frame_values(d.process_online, "p") & coarse,
        "its units online, start-ups and shut-downs are decided once per coarse step")
    add((_frame_values(getattr(d, "pdt_uptime_set", None), "p") | _frame_values(getattr(d, "pdt_downtime_set", None),
                                                                               "p")) & coarse,
        "its minimum up / down time counts whole coarse steps")
    add(varies(_param_frame(getattr(d, "p_min_load", None)), steps, ("p",))
        & _frame_values(getattr(d, "process_minload", None), "p") & _frame_values(d.process_online, "p"),
        "its minimum load varies within a coarse step")
    add(ramp & _frame_values(d.process_online, "p") & coarse,
        "its start-up / shut-down capacity reduction uses the coarse step")
    reserve = _frame_values(getattr(d, "prundt", None), "p") & coarse
    add(reserve, "it holds one reserve per coarse step (the requirement is checked at every time step)")
    headroom = varies(_param_frame(d.p_process_availability), steps, ("p",))
    for idx in (d.process_profile_upper, d.process_profile_lower, d.process_profile_fixed):
        if idx is None or idx.height == 0 or profile is None:
            continue
        f_col = "f" if "f" in idx.columns else "profile"
        arcs = idx.select(pl.col("p").cast(pl.Utf8), pl.col(f_col).cast(pl.Utf8).alias("f")).unique()
        headroom |= varies(arcs.join(profile.with_columns(pl.col("f").cast(pl.Utf8)), on="f", how="inner"),
                           steps, ("p", "f"))
    add(reserve & headroom, "its reserve headroom uses the step-mean availability / profile")
    for p, why in _row_rule_reasons(d, coarse):
        out.setdefault(p, []).append(why)
    return out


def _spread_keys(part: pl.DataFrame, frame: "pl.DataFrame | None", key: str) -> set[str]:
    """The keys of ``part`` ``(key, d, b_first, t)`` (Utf8) where ``frame`` ``(key, …, d, t, value)`` (Utf8 keys) takes
    more than one value within a coarse step (per its other key columns)."""
    if frame is None or frame.height == 0 or part.height == 0 or "t" not in frame.columns:
        return set()
    other = [c for c in frame.columns if c not in (key, "d", "t", "value")]
    on = [c for c in (key, "d", "t") if c in frame.columns]
    spread = (part.select(key, "d", "b_first", "t").join(frame, on=on, how="inner")
              .group_by(key, *other, "d", "b_first")
              .agg((pl.col("value").max() - pl.col("value").min()).alias("__s"),
                   pl.col("value").abs().max().alias("__m"))
              .filter(pl.col("__s") > 1e-9 * pl.max_horizontal(pl.lit(1.0), pl.col("__m"))))
    return set(spread[key].to_list())


def _node_param_by_group(gn: "pl.DataFrame | None", param: "Param | None") -> "pl.DataFrame | None":
    """``(g, n, d, t, value)`` (Utf8 keys) of a node parameter ``(n, d, t)`` at the nodes of each group."""
    frame = _param_frame(param)
    if gn is None or frame is None or not {"n", "d", "t"}.issubset(frame.columns):
        return None
    f = frame.select(*(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")), pl.col("value").cast(pl.Float64))
    return gn.join(f, on="n", how="inner").select("g", "n", "d", "t", "value")


def _time_varying_groups(part: pl.DataFrame, param: "Param | None") -> set[str]:
    """The groups of ``part`` ``(g, d, b_first, t)`` whose group limit ``param`` (authored dims ⊆ ``(g, d, t)``)
    takes more than one value within a coarse step; a limit without a time axis never does.  A time step the
    limit leaves out (a dropped zero) is not counted."""
    cap = _param_frame(param)
    if cap is None or "t" not in cap.columns:
        return set()
    keys = [c for c in ("g", "d", "t") if c in cap.columns]
    return _spread_keys(part, cap.select(*(pl.col(c).cast(pl.Utf8) for c in keys),
                                         pl.col("value").cast(pl.Float64)), "g")


def _row_rule_reasons(flex_data: "FlexData", coarse: set[str]) -> list[tuple[str, str]]:
    """``(p, reason)`` of the coarse processes ``coarse`` in rows under the row rule (:func:`coarse_process_reasons`):
    a row mixing coarse and fine members (per solve step, the coarse flow constant over its step), and an averaged
    row whose demand / inflow / profile / limit varies within a coarse step (the step mean)."""
    d = flex_data
    out: list[tuple[str, str]] = []
    if not coarse:
        return out
    from flextool.engine.model import _group_slack

    gn = _utf8_frame(getattr(d, "group_node", None), ("g", "n"))
    active = {"capacity_margin": _group_slack.has_capacity_margin(d), "inertia": _group_slack.has_inertia(d),
              "non_sync": _group_slack.has_non_sync(d), "instant_flow": True, "user_constraint": True,
              "storage_use": False}
    for family, what in COARSE_ROW_FAMILIES.items():
        if not active[family]:
            continue
        members = coarse_row_members(d, family)
        if members.height == 0:
            continue
        key = members.columns[0]
        part = coarse_row_partition(d, family, members)
        averaged = set(part[key].to_list())
        procs = members.filter(pl.col("kind") == "p")
        fine_rows = procs.filter(~pl.col(key).is_in(sorted(averaged))).select(key, "e").unique().sort(key, "e")
        for k, p in fine_rows.iter_rows():
            if p in coarse:
                out.append((p, f"the {what} row of {k!r} stays at the solve's step (not all its members take the "
                               "coarse step) and holds its flow constant over the coarse step"))
        varying: set[str] = set()
        if family == "capacity_margin":
            nodes = None if gn is None else gn.filter(
                ~pl.col("n").is_in(sorted(_frame_values(getattr(d, "nodeState", None), "n"))))
            varying |= _spread_keys(part, _node_param_by_group(nodes, getattr(d, "pdtNodeInflow_per_step", None)),
                                    "g")
            profile = _param_frame(d.p_profile_value)
            pss = _utf8_frame(d.process_source_sink, ("p", "source", "sink"))
            if nodes is not None and profile is not None and pss is not None:
                for idx in (d.process_profile_upper, d.process_profile_fixed):
                    arcs = _utf8_frame(idx, ("p", "sink", "f"))
                    if arcs is None:
                        continue
                    by_g = (arcs.join(nodes.rename({"n": "sink"}), on="sink", how="inner")
                            .join(profile.select(*(pl.col(c).cast(pl.Utf8) for c in ("f", "d", "t")),
                                                 pl.col("value").cast(pl.Float64)), on="f", how="inner")
                            .select("g", "f", "d", "t", "value"))
                    varying |= _spread_keys(part, by_g, "g")
        elif family == "non_sync":
            for name in ("p_positive_inflow", "p_negative_inflow"):
                varying |= _spread_keys(part, _node_param_by_group(gn, getattr(d, name, None)), "g")
            varying |= _time_varying_groups(part, d.pdtGroup_non_synchronous_limit)
        elif family == "inertia":
            varying |= _time_varying_groups(part, d.pdtGroup_inertia_limit)
        elif family == "instant_flow":
            for name in ("pdt_max_instant_flow", "pdt_min_instant_flow"):
                varying |= _time_varying_groups(part, getattr(d, name, None))
        varying_rows = procs.filter(pl.col(key).is_in(sorted(varying))).select(key, "e").unique().sort(key, "e")
        for k, p in varying_rows.iter_rows():
            out.append((p, f"the {what} row of {k!r} uses the step mean of a demand, inflow, profile or limit that "
                           "varies within a coarse step"))
    out.extend(_dc_mixed_reasons(d, coarse))
    return out


def _dc_mixed_reasons(flex_data: "FlexData", coarse: set[str]) -> list[tuple[str, str]]:
    """``(p, reason)`` of the coarse DC lines with an end node whose angle stays at the solve's step (its DC network
    is not all coarse, :func:`coarse_angle_steps`): their DC power-flow row is per solve step with the flow
    constant."""
    d = flex_data
    a = _dc_arcs_utf8(d)
    if a is None:
        return []
    angle = set(coarse_angle_steps(d)["n"].to_list())
    a = a.unique()
    mixed = a.filter(pl.col("p").is_in(sorted(coarse))
                     & ~(pl.col("source").is_in(sorted(angle)) & pl.col("sink").is_in(sorted(angle))))
    return [(p, "its DC power-flow row stays at the solve's step (its DC network has a line at the solve's step "
                "or nodes with different coarse steps) and holds the angle difference constant over the coarse step")
            for p in sorted(set(mixed["p"].to_list()))]


def coarse_tie_straddles(flex_data: "FlexData", steps: pl.DataFrame) -> set[str]:
    """The coarse processes of ``steps`` ``(p, d, b_first, t)`` with a coarse step that a forecast tie covers only
    in part, or whose coarse steps differ between a forecast and its reference forecast over the tied steps.

    The ties hold over the tied steps ``dt_non_anticipativity`` (the committed steps) only, between the forecasts
    of an anchor period: the net flow into each balance node of a ``groupStochastic`` group
    (``non_anticipativity_storage_use``), the units online of every online process
    (``non_anticipativity_online_*``) and the reserve of every reserve provider (``non_anticipativity_reserve``).
    A coarse process has one flow, one units-online value and one reserve per coarse step,
    so a tie on one fine step of a coarse step holds the whole step: where a coarse step reaches past the end of
    the tied steps the forecasts could no longer differ after it, although nothing is committed there.  A tie row
    pairs the column of a forecast's coarse step with the reference forecast's column at the same step, which is
    one value per step only where both forecasts cut the tied steps into the same coarse steps.  Such a process
    keeps the solve's step.  Under the ``uc_only`` role (solve (1) of separate_dispatch,
    ``_non_anticipativity.na_role``) only the units online of the ``_fix_in_uc`` processes are tied."""
    from flextool.engine.model._non_anticipativity import na_role, reference_forecasts

    d = flex_data
    pss, tie = d.process_source_sink, d.dt_non_anticipativity
    if any(f is None or f.height == 0 for f in (pss, tie, steps)):
        return set()
    forecasts = [(a, f) for a, f in reference_forecasts(d) if len(f) >= 2]
    if not forecasts:
        return set()
    if na_role(d) == "uc_only":
        procs = _frame_values(d.process_online, "p") & _frame_values(d.process_online_fix_in_uc, "p")
    else:
        procs = _frame_values(d.process_online, "p") | _frame_values(
            getattr(d, "process_reserve_upDown_node_active", None), "p")
    gn, gs = d.group_node, d.groupStochastic
    if na_role(d) != "uc_only" and all(f is not None and f.height > 0 for f in (gn, gs, d.nodeBalance)):
        nodes = (_frame_values(gn.join(gs.select("g").unique(), on="g", how="semi"), "n")
                 & _frame_values(d.nodeBalance, "n"))
        arcs = pss.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink")))
        procs |= set(arcs.filter(pl.col("source").is_in(sorted(nodes)) | pl.col("sink").is_in(sorted(nodes)))
                     ["p"].to_list())
    if not procs:
        return set()
    tied_by_anchor = tie.select(pl.col("d").cast(pl.Utf8).alias("anchor"), pl.col("t").cast(pl.Utf8)).unique()
    fan = pl.DataFrame([(a, f, fs[0]) for a, fs in forecasts for f in fs],
                       schema={"anchor": pl.Utf8, "d": pl.Utf8, "d_ref": pl.Utf8}, orient="row")
    tied = (fan.join(tied_by_anchor, on="anchor", how="inner")
            .select("d", "d_ref", "t", __tied=pl.lit(True)))
    st = (steps.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "b_first", "t")))
          .filter(pl.col("p").is_in(sorted(procs)) & pl.col("d").is_in(fan["d"].to_list()))
          .join(tied, on=["d", "t"], how="left"))
    part = (st.group_by("p", "d", "b_first")
            .agg(pl.col("__tied").is_not_null().any().alias("__any"), pl.col("__tied").is_not_null().all().alias("__all"))
            .filter(pl.col("__any") & ~pl.col("__all")))
    ref = steps.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_ref"), pl.col("t").cast(pl.Utf8),
                       pl.col("b_first").cast(pl.Utf8).alias("__b_ref"))
    other = (st.filter(pl.col("__tied").is_not_null())
             .join(ref, on=["p", "d_ref", "t"], how="left")
             .filter(pl.col("__b_ref").is_null() | (pl.col("__b_ref") != pl.col("b_first"))))
    return set(part["p"].to_list()) | set(other["p"].to_list())


def coarse_pin_varies(flex_data: "FlexData", steps: pl.DataFrame) -> set[str]:
    """The coarse processes of ``steps`` ``(p, d, b_first, t)`` with a coarse step whose units online are pinned
    (``p_fix_online``: the realization solve takes the committed units online of its parent) on only part of the
    step, or to values that change within the step.  The fine model pins each step to its own value; one value per
    coarse step could only take a mean (fractional for integer units) or pin the free steps too.  Such a process
    keeps the solve's step."""
    from flextool.engine.preprocess.params._coarse import varies_within_coarse_steps

    if steps is None or steps.height == 0:
        return set()
    st = steps.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "b_first", "t")))
    out: set[str] = set()
    # One entry per pinned variable family, keyed by its non-(d, t) axes; a pin of another family is added here.
    for name, keys in (("p_fix_online", ("p",)),):
        pin = _param_frame(getattr(flex_data, name, None))
        if pin is None or pin.height == 0:
            continue
        pins = pin.select(*(pl.col(c).cast(pl.Utf8) for c in (*keys, "d", "t")), pl.col("value").cast(pl.Float64))
        # A pinned (p, keys…) whose pin covers only part of a coarse step of p.
        part = (st.join(pins.select(*keys).unique(), on="p", how="inner")
                .join(pins.select(*keys, "d", "t", __pin=pl.lit(True)), on=[*keys, "d", "t"], how="left")
                .group_by(*keys, "d", "b_first")
                .agg(pl.col("__pin").is_not_null().any().alias("__any"),
                     pl.col("__pin").is_not_null().all().alias("__all"))
                .filter(pl.col("__any") & ~pl.col("__all")))
        out |= set(part["p"].to_list()) | varies_within_coarse_steps(pins, st, keys)
    return out


def _utf8_frame(frame: "pl.DataFrame | None", cols: tuple[str, ...]) -> "pl.DataFrame | None":
    if frame is None or frame.height == 0 or any(c not in frame.columns for c in cols):
        return None
    return frame.select(*(pl.col(c).cast(pl.Utf8) for c in cols))


def _dc_arcs_utf8(flex_data: "FlexData") -> "pl.DataFrame | None":
    """``(p, source, sink)`` (Utf8) of the DC arcs, as ``_dc_power_flow.dc_arcs_frame`` picks them (the arcs of
    ``connection_dc_power_flow`` lines between ``node_dc_power_flow`` nodes, one orientation), read with Utf8 keys:
    the frames can still differ in dtype while FlexData is being built."""
    d = flex_data
    lines = _frame_values(getattr(d, "connection_dc_power_flow", None), "p")
    nodes = _frame_values(getattr(d, "node_dc_power_flow", None), "n")
    src = getattr(d, "process_source_toSink_dc", None)
    arcs = _utf8_frame(src if src is not None else d.process_source_sink, ("p", "source", "sink"))
    if arcs is None or not lines or not nodes:
        return None
    out = arcs.filter(pl.col("p").is_in(sorted(lines)) & pl.col("source").is_in(sorted(nodes))
                      & pl.col("sink").is_in(sorted(nodes)))
    return out if out.height > 0 else None


def coarse_angle_steps(flex_data: "FlexData") -> pl.DataFrame:
    """``(n, d, b_first, t)`` (Utf8): the coarse steps of the *coarse-angle* nodes — the nodes of a DC network
    component (the nodes of ``node_dc_power_flow`` joined by DC lines, ``_dc_power_flow.dc_arcs_frame``) whose
    every node is coarse (``node_coarse_step``) with the same coarse steps and whose every DC line is a coarse
    process.  Their voltage angle is one value per coarse step.  Every other DC node keeps an angle per solve step
    (owner decision 2026-11-04, Q3), so a DC line at the solve's step keeps its per-step physics: a coarse angle
    next to a node with a per-step line would make the coarse line's row (flow and one end angle constant over the
    step) hold that node's angle constant, and through it the angles and fine flows of the rest of the component.
    In a component of coarse-angle nodes the per-step rows of the coarse lines already hold every angle difference
    constant over a coarse step, and the reference angle is 0, so one angle per coarse step is exact.  Empty
    without such a component."""
    empty = pl.DataFrame(schema={c: pl.Utf8 for c in ("n", "d", "b_first", "t")})
    d = flex_data
    ncs = _utf8_frame(getattr(d, "node_coarse_step", None), ("n", "d", "b_first", "t"))
    dc_nodes = _frame_values(getattr(d, "node_dc_power_flow", None), "n")
    coarse_p = _frame_values(getattr(d, "processCoarse", None), "p")
    if ncs is None or not dc_nodes or not coarse_p or d.process_source_sink is None:
        return empty
    a = _dc_arcs_utf8(d)
    arcs = [] if a is None else a.select("p", "source", "sink").unique().sort("p", "source", "sink").rows()
    # Connected components of the DC network (union-find over the nodes of node_dc_power_flow).
    parent = {n: n for n in dc_nodes}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for _p, s, k in arcs:
        rs, rk = root(s), root(k)
        if rs != rk:
            parent[max(rs, rk)] = min(rs, rk)
    steps = {n: frozenset(g.select("d", "b_first", "t").rows())
             for (n,), g in ncs.filter(pl.col("n").is_in(sorted(dc_nodes))).group_by("n")}
    bad: set[str] = {root(n) for n in dc_nodes if n not in steps}
    bad |= {root(s) for p, s, _k in arcs if p not in coarse_p}
    first: dict[str, frozenset] = {}
    for n in sorted(steps):
        r = root(n)
        if first.setdefault(r, steps[n]) != steps[n]:
            bad.add(r)
    keep = sorted(n for n in dc_nodes if root(n) not in bad)
    return ncs.filter(pl.col("n").is_in(keep))


#: The row families under the row rule (owner decision 2026-11-04) -> the row's name in the behaviour notice.
COARSE_ROW_FAMILIES: dict[str, str] = {
    "capacity_margin": "capacity-margin", "inertia": "inertia", "non_sync": "non-synchronous",
    "instant_flow": "flow-group instant-flow", "user_constraint": "user constraint",
    "storage_use": "forecast storage-use tie",
}


def _arcs_at_nodes(arcs: pl.DataFrame, nodes: pl.DataFrame, side: str) -> pl.DataFrame:
    """``arcs`` ``(p, source, sink)`` (Utf8) whose ``side`` node is a node of ``nodes`` ``(key, n)``, with key."""
    return arcs.join(nodes.rename({"n": side}), on=side, how="inner")


def coarse_row_members(flex_data: "FlexData", family: str) -> pl.DataFrame:
    """``(key, kind, e)`` (Utf8) of the rows of ``family`` (:data:`COARSE_ROW_FAMILIES`): the entities whose
    time-varying variables can appear in the row of each key (``params._coarse.coarse_row_steps``: kind ``p`` a
    process — its flows, units online —, kind ``n`` a node — the state of a coarse storage node).  The model and
    the output readers both read the coarse steps of a row from these (:func:`coarse_row_partition`), so a
    variable keyed like the row (the inertia and non-synchronous slacks) has the steps of its row.  A superset of
    the row's entities is safe (one more member can only keep the row at the solve's step).

    * ``capacity_margin`` (key ``g``): the processes whose flows enter the row (storage-backed units, connections,
      consumption: the arcs at N(g) that neither count at capacity, ``_group_slack.capacity_margin_counted_arcs``,
      nor are left out, ``_group_slack.capacity_margin_left_out_arcs``);
    * ``inertia`` (``g``): the processes with an inertia constant on a side at a node of g;
    * ``non_sync`` (``g``): the processes with an arc at a node of g, without those inside g;
    * ``instant_flow`` (``g``): the processes of the flow group (``group_process_node``);
    * ``user_constraint`` (``cn``): the processes with a flow coefficient, the processes of the solve with an
      online coefficient (``constraint_online_coeff``) and the nodes with a state coefficient;
    * ``storage_use`` (``n``): the processes with an arc at the stochastic node n."""
    d = flex_data
    key = {"user_constraint": "cn", "storage_use": "n"}.get(family, "g")
    if family not in COARSE_ROW_FAMILIES:
        raise ValueError(f"coarse_row_members: unknown family {family!r}")
    empty = pl.DataFrame(schema={key: pl.Utf8, "kind": pl.Utf8, "e": pl.Utf8})
    pss = _utf8_frame(d.process_source_sink, ("p", "source", "sink"))
    parts: list[pl.DataFrame] = []

    def procs(frame: "pl.DataFrame | None") -> None:
        if frame is not None and frame.height > 0:
            parts.append(frame.select(pl.col(key), kind=pl.lit("p"), e=pl.col("p")))

    gn = _utf8_frame(getattr(d, "group_node", None), ("g", "n"))
    if family in ("capacity_margin", "inertia", "non_sync"):
        groups = {"capacity_margin": "groupCapacityMargin", "inertia": "groupInertia",
                  "non_sync": "groupNonSync"}[family]
        gs = _frame_values(getattr(d, groups, None), "g")
        if gn is None or pss is None or not gs:
            return empty
        nodes = gn.filter(pl.col("g").is_in(sorted(gs)))
        if family == "capacity_margin":
            from flextool.engine.model import _group_slack

            nodes = _utf8_frame(_group_slack.capacity_margin_nodes(d), ("g", "n"))
            if nodes is None:
                return empty
            at = pl.concat([_arcs_at_nodes(pss, nodes, "source"), _arcs_at_nodes(pss, nodes, "sink")],
                           how="diagonal_relaxed").select("g", "p", "source", "sink")
            for left in (_group_slack.capacity_margin_counted_arcs(d), _group_slack.capacity_margin_left_out_arcs(d)):
                if left is not None:
                    at = at.join(left.select("g", "p", "source", "sink"), on=["g", "p", "source", "sink"],
                                 how="anti")
            procs(at.select("g", "p").unique())
        elif family == "inertia":
            for side in ("source", "sink"):
                idx = _utf8_frame(getattr(d, f"process_{side}_inertia", None), ("p", side))
                if idx is not None:
                    procs(_arcs_at_nodes(pss.join(idx, on=["p", side], how="inner"), nodes, side))
        else:
            at = pl.concat([_arcs_at_nodes(pss, nodes, "source"), _arcs_at_nodes(pss, nodes, "sink")],
                           how="diagonal_relaxed").select("g", "p")
            inside = _utf8_frame(getattr(d, "process_group_inside_nonSync", None), ("p", "g"))
            if inside is not None:
                at = at.join(inside, on=["p", "g"], how="anti")
            procs(at)
    elif family == "instant_flow":
        gs = (_frame_values(_param_frame(getattr(d, "pdt_max_instant_flow", None)), "g")
              | _frame_values(_param_frame(getattr(d, "pdt_min_instant_flow", None)), "g"))
        gpn = _utf8_frame(getattr(d, "group_process_node", None), ("g", "p"))
        if gpn is None or not gs:
            return empty
        procs(gpn.filter(pl.col("g").is_in(sorted(gs))))
    elif family == "user_constraint":
        procs(_utf8_frame(getattr(d, "flow_constraint_idx", None), ("cn", "p")))
        online = _utf8_frame(_param_frame(getattr(d, "p_process_constraint_online_coeff", None)), ("cn", "p"))
        if online is not None and pss is not None:
            procs(online.join(pss.select("p").unique(), on="p", how="semi"))
        state = _utf8_frame(_param_frame(getattr(d, "p_node_constraint_state_coeff", None)), ("cn", "n"))
        if state is not None:
            parts.append(state.select("cn", kind=pl.lit("n"), e=pl.col("n")))
    else:
        if pss is None:
            return empty
        for side in ("source", "sink"):
            procs(pss.select("p", pl.col(side).alias("n")))
    if not parts:
        return empty
    return pl.concat([f.select(key, "kind", "e") for f in parts]).unique().sort(key, "kind", "e")


def coarse_row_partition(flex_data: "FlexData", family: str,
                         members: "pl.DataFrame | None" = None) -> pl.DataFrame:
    """``(key, d, b_first, t, w)`` (Utf8 keys): the coarse steps of the keys of ``family`` whose every member
    (:func:`coarse_row_members`, or ``members``) is coarse — the row rule: one averaged row per coarse step.  Empty
    without such a key (every row stays per solve step)."""
    from flextool.engine.preprocess.params._coarse import coarse_row_steps

    mem = coarse_row_members(flex_data, family) if members is None else members
    psd = flex_data.p_step_duration
    return coarse_row_steps(mem, getattr(flex_data, "process_coarse_step", None),
                            getattr(flex_data, "node_coarse_step", None), None if psd is None else psd.frame)


def _apply_coarse_process_frames(flex_data: "FlexData") -> None:
    """Set ``processCoarse`` / ``process_coarse_step`` (``params._coarse.coarse_process_frames``) from the
    coarse node frames, and print the info lines of the coarse processes."""
    from flextool.engine.preprocess.params._coarse import (
        coarse_groups, coarse_process_frames, coarse_process_messages)

    layout = flex_data.block_layout
    ncs = flex_data.node_coarse_step
    pss = flex_data.process_source_sink
    if layout is None or ncs is None or ncs.height == 0 or pss is None or pss.height == 0:
        return
    solve_step = float(layout.block_step_duration["default"])
    groups = coarse_groups(layout.node_block, layout.block_step_duration, solve_step)
    coarse_nodes = _frame_values(flex_data.nodeCoarse, "n")
    node_step = {n: groups[g] for n, g in layout.node_block.items() if n in coarse_nodes and g in groups}
    arcs = pss.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink"))).unique()
    process_nodes: dict[str, set[str]] = {}
    for p, s, k in arcs.iter_rows():
        process_nodes.setdefault(p, set()).update(n for n in (s, k) if n != p)
    psd = flex_data.p_step_duration
    excluded = coarse_process_exclusions(flex_data)

    def frames_for(excluded: dict[str, str]):
        return coarse_process_frames(
            process_nodes=process_nodes, balance_nodes=_frame_values(flex_data.nodeBalance, "n"),
            node_step=node_step, node_coarse_step=ncs, step_duration=None if psd is None else psd.frame,
            excluded=excluded, solve_step=solve_step)

    frames = frames_for(excluded)
    if frames is not None and frames.processCoarse.height > 0:
        kept: dict[str, str] = {}
        why = ("the forecasts are tied (non-anticipativity) over only part of one of its coarse steps, and one "
               "value per coarse step would tie the whole step")
        kept.update({p: why for p in sorted(coarse_tie_straddles(flex_data, frames.process_coarse_step))})
        why = ("its units online are pinned to a commitment that covers only part of one of its "
               "coarse steps or changes within it")
        for p in sorted(coarse_pin_varies(flex_data, frames.process_coarse_step)):
            kept.setdefault(p, why)
        if kept:
            frames = frames_for({**excluded, **kept})
    coarse_process_messages(frames)
    if frames is None or frames.processCoarse.height == 0:
        return
    flex_data.processCoarse = frames.processCoarse
    flex_data.process_coarse_step = frames.process_coarse_step
    reasons = coarse_process_reasons(flex_data, frames.process_coarse_step)
    if reasons:
        from flextool.engine import _behaviour_changes

        _behaviour_changes.notice(
            "coarse_process_flows", "units and connections",
            [f"{p} ({'; '.join(why)})" for p, why in sorted(reasons.items())])
