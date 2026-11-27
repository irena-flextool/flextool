"""The storage usage, cycle start and level a solved solve hands on."""

from __future__ import annotations

import polars as pl
from flextool.engine.input._axis_enums import alias_to_axis


def _storage_usage_mwh(sol, flex_data, nodes: set, steps_df: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``[node, period, step, p_fix_storage_usage]``: the net energy (MWh) drawn from each storage node of ``nodes``
    in each step of ``steps_df`` (``period``, ``step``) of a solved solve, by the terms of
    ``model.core._node_balance._eq_storage_usage_fix`` (see the fix_storage_usage producer in
    :func:`build_handoff_from_solution`); None when there is none."""
    if not (nodes
            and steps_df is not None
            and steps_df.height > 0
            and {"period", "step"}.issubset(steps_df.columns)
            and sol is not None
            and flex_data is not None
            and "v_flow" in getattr(sol, "_vars", {})):
        return None
    from flextool.engine.output._coarse_flows import var_values

    v_flow_long = var_values(sol, flex_data, "v_flow")
    if v_flow_long is not None and v_flow_long.height > 0:
        fs_steps_fu = (steps_df
            .select(alias_to_axis("period", "d"),
                     alias_to_axis("step", "t"))
            .unique()
            .with_columns(pl.col("d").cast(pl.Utf8),
                          pl.col("t").cast(pl.Utf8)))

        us_param = (getattr(flex_data, "p_all_entity_unitsize", None)
                    or getattr(flex_data, "p_unitsize", None))
        if us_param is not None:
            # ``p_all_entity_unitsize`` keys the per-process unitsize on
            # the entity-union axis ``e``; ``p_unitsize`` keys it on the
            # process axis ``p``.  Read whichever entity column the chosen
            # carrier carries, then rename to ``p`` (cast to Utf8 to match
            # the Utf8 join below).
            _us_col = "e" if "e" in us_param.frame.columns else "p"
            unitsize_frame = (us_param.frame
                .select(pl.col(_us_col).cast(pl.Utf8).alias("p"),
                        pl.col("value").alias("us")))
        else:
            unitsize_frame = pl.DataFrame(
                schema={"p": pl.Utf8, "us": pl.Float64})

        step_dur_param = getattr(flex_data, "p_step_duration", None)
        if step_dur_param is not None:
            step_dur_frame = (step_dur_param.frame
                .select("d", "t", pl.col("value").alias("dur"))
                .with_columns(pl.col("d").cast(pl.Utf8),
                              pl.col("t").cast(pl.Utf8)))
        else:
            step_dur_frame = pl.DataFrame(
                schema={"d": pl.Utf8, "t": pl.Utf8, "dur": pl.Float64})

        fu_node_df = pl.DataFrame(
            {"n": sorted(nodes)}, schema={"n": pl.Utf8},
        )

        flows = (v_flow_long
            .with_columns(
                pl.col("p").cast(pl.Utf8),
                pl.col("source").cast(pl.Utf8),
                pl.col("sink").cast(pl.Utf8),
                pl.col("d").cast(pl.Utf8),
                pl.col("t").cast(pl.Utf8),
            )
            .join(fs_steps_fu, on=["d", "t"], how="inner")
            .join(unitsize_frame, on="p", how="left")
            .with_columns(pl.col("us").fill_null(1.0)))

        # n-as-sink: inflow contribution (negative, raw v_flow * unitsize).
        # Mirrors constraint LHS sink_flow term (model._eq_storage_usage_fix).
        # Applied to all sink arcs touching nodes regardless of
        # eff vs noEff partition.
        sink_side = (flows
            .join(fu_node_df, left_on="sink", right_on="n", how="inner")
            .with_columns(
                contrib=-pl.col("value") * pl.col("us"),
                n=pl.col("sink"),
            )
            .select("n", "d", "t", "contrib"))

        # n-as-source contributions partitioned to match constraint:
        #   - eff partition: v_flow * unitsize * slope (+ section term
        #     for min_load_efficiency processes, see below).
        #   - noEff partition: v_flow * unitsize (raw).
        contrib_sides: list[pl.DataFrame] = [sink_side]

        pss_eff_fu = None
        if (flex_data.process_source_sink_eff is not None
                and flex_data.process_source_sink_eff.height > 0):
            pss_eff_fu = (flex_data.process_source_sink_eff
                .with_columns(
                    pl.col("p").cast(pl.Utf8),
                    pl.col("source").cast(pl.Utf8),
                    pl.col("sink").cast(pl.Utf8),
                )
                .join(fu_node_df, left_on="source", right_on="n",
                      how="inner"))
            if pss_eff_fu.height > 0:
                # source_eff with slope multiplier (model._eq_storage_usage_fix
                # gates on p_slope is not None — mirror that gate).
                if flex_data.p_slope is not None:
                    slope_frame = (flex_data.p_slope.frame
                        .select("p", "d", "t",
                                pl.col("value").alias("slope"))
                        .with_columns(
                            pl.col("p").cast(pl.Utf8),
                            pl.col("d").cast(pl.Utf8),
                            pl.col("t").cast(pl.Utf8),
                        ))
                    source_eff_side = (flows
                        .join(pss_eff_fu.select("p", "source", "sink"),
                              on=["p", "source", "sink"], how="inner")
                        .join(slope_frame, on=["p", "d", "t"], how="left")
                        .with_columns(pl.col("slope").fill_null(1.0))
                        .with_columns(
                            contrib=pl.col("value") * pl.col("us")
                                    * pl.col("slope"),
                            n=pl.col("source"),
                        )
                        .select("n", "d", "t", "contrib"))
                    contrib_sides.append(source_eff_side)

                # source_eff min_load_efficiency section term
                # (model._eq_storage_usage_fix).  Uses v_online_linear /
                # v_online_integer * section * unitsize.  No slope
                # multiplier (per the constraint).
                if (flex_data.process_min_load_eff is not None
                        and flex_data.process_min_load_eff.height > 0
                        and flex_data.p_section is not None):
                    mle_p = (flex_data.process_min_load_eff
                        .select("p")
                        .with_columns(pl.col("p").cast(pl.Utf8))
                        .unique())
                    section_arcs = (pss_eff_fu
                        .select("p", "source", "sink")
                        .join(mle_p, on="p", how="inner"))
                    if section_arcs.height > 0:
                        section_frame = (flex_data.p_section.frame
                            .select("p", "d", "t",
                                    pl.col("value").alias("section"))
                            .with_columns(
                                pl.col("p").cast(pl.Utf8),
                                pl.col("d").cast(pl.Utf8),
                                pl.col("t").cast(pl.Utf8),
                            ))
                        # Combine v_online_linear + v_online_integer
                        # (both contribute to the section term per
                        # model._eq_storage_usage_fix; only one is typically
                        # present per process, but the sum is safe).
                        online_long = None
                        sol_vars = getattr(sol, "_vars", {}) or {}
                        for var_name in (
                            "v_online_linear", "v_online_integer",
                        ):
                            if var_name in sol_vars:
                                vd = var_values(sol, flex_data, var_name)
                                if vd is not None and vd.height > 0:
                                    vd = vd.with_columns(
                                        pl.col("p").cast(pl.Utf8),
                                        pl.col("d").cast(pl.Utf8),
                                        pl.col("t").cast(pl.Utf8),
                                    ).select("p", "d", "t", "value")
                                    online_long = (vd if online_long is None
                                                   else pl.concat(
                                                       [online_long, vd],
                                                       how="vertical"))
                        if (online_long is not None
                                and online_long.height > 0):
                            section_side = (online_long
                                .join(fs_steps_fu, on=["d", "t"],
                                      how="inner")
                                .join(section_arcs, on="p", how="inner")
                                .join(section_frame,
                                      on=["p", "d", "t"], how="left")
                                .with_columns(
                                    pl.col("section").fill_null(0.0))
                                .join(unitsize_frame, on="p", how="left")
                                .with_columns(
                                    pl.col("us").fill_null(1.0))
                                .with_columns(
                                    contrib=pl.col("value")
                                            * pl.col("section")
                                            * pl.col("us"),
                                    n=pl.col("source"),
                                )
                                .select("n", "d", "t", "contrib"))
                            contrib_sides.append(section_side)

        if (flex_data.process_source_sink_noEff is not None
                and flex_data.process_source_sink_noEff.height > 0):
            pss_noEff_fu = (flex_data.process_source_sink_noEff
                .with_columns(
                    pl.col("p").cast(pl.Utf8),
                    pl.col("source").cast(pl.Utf8),
                    pl.col("sink").cast(pl.Utf8),
                )
                .join(fu_node_df, left_on="source", right_on="n",
                      how="inner"))
            if pss_noEff_fu.height > 0:
                source_noEff_side = (flows
                    .join(pss_noEff_fu.select("p", "source", "sink"),
                          on=["p", "source", "sink"], how="inner")
                    .with_columns(
                        contrib=pl.col("value") * pl.col("us"),
                        n=pl.col("source"),
                    )
                    .select("n", "d", "t", "contrib"))
                contrib_sides.append(source_noEff_side)

        # Fallback when neither partition is populated: legacy
        # simplified formula on the full arc set (covers fixtures
        # whose eff/noEff partition is missing, e.g. the no-process
        # toy fixtures used by wiring tests; the new formula must
        # not regress those).
        if len(contrib_sides) == 1:  # sink_side only — no source partitions
            source_side = (flows
                .join(fu_node_df, left_on="source", right_on="n",
                      how="inner")
                .with_columns(
                    contrib=pl.col("value") * pl.col("us"),
                    n=pl.col("source"),
                )
                .select("n", "d", "t", "contrib"))
            contrib_sides.append(source_side)

        net = (pl.concat(contrib_sides, how="vertical")
            .group_by(["n", "d", "t"])
            .agg(pl.col("contrib").sum().alias("net")))

        if net.height > 0:
            weighted = (net
                .join(step_dur_frame, on=["d", "t"], how="left")
                .with_columns(pl.col("dur").fill_null(1.0))
                .with_columns(
                    p_fix_storage_usage=pl.col("net") * pl.col("dur"),
                )
                # A window with zero usage stays: it caps the lower solve's usage at 0 (item 150).
                .select(
                    alias_to_axis("n", "node"),
                    alias_to_axis("d", "period"),
                    alias_to_axis("t", "step"),
                    pl.col("p_fix_storage_usage"),
                ))
            if weighted.height > 0:
                return weighted
    return None


def _storage_start_and_level(sol, flex_data, nsfm_df, unitsize: dict):
    """``(cycle_start_state, storage_level, scope_start_state)`` of a solved solve (:class:`SolveHandoff`), each None
    when empty.

    ``cycle_start_state`` ``[node, value]`` (MWh): the level each storage node's first step starts from
    (``model._self_discharge.start_levels``).  ``storage_level`` ``[node, period, step, value]`` (MWh, the end of
    each step): v_state × state_unitsize at every step of the solve for the nodes of
    ``node__storage_nested_fix_method`` (``nsfm_df``, any method).  ``scope_start_state`` ``[node, period, step,
    value]`` (MWh): the level each cyclic period / timeblock scope that starts in the solve starts from
    (``model._self_discharge.scope_start_levels``).  All need ``flex_data`` and a solved v_state.
    """
    if flex_data is None or "v_state" not in getattr(sol, "_vars", {}):
        return None, None, None
    v_state = sol.value("v_state")
    if v_state is None or v_state.height == 0:
        return None, None, None
    state = v_state.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                           pl.col("value").cast(pl.Float64))

    def chosen(var: str, dims: tuple) -> "pl.DataFrame | None":
        if var not in sol._vars:
            return None
        frame = sol.value(var)
        if frame is None or frame.height == 0:
            return None
        keys = [c for c in frame.columns if c != "value"]
        return frame.select([pl.col(k).cast(pl.Utf8).alias(n) for k, n in zip(keys, dims)]
                            + [pl.col("value").cast(pl.Float64)])

    from flextool.engine.model import _self_discharge
    start = _self_discharge.start_levels(flex_data, state, chosen)
    level = None
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        nodes = sorted(set(nsfm_df.filter(pl.col("method").is_in(["fix_quantity", "fix_price", "fix_usage"]))
                           ["node"].cast(pl.Utf8).to_list()))
        if nodes:
            us = pl.DataFrame({"n": nodes, "us": [float(unitsize.get(n, 1.0)) for n in nodes]})
            level = (state.join(us, on="n", how="inner")
                     .select(pl.col("n").alias("node"), pl.col("d").alias("period"), pl.col("t").alias("step"),
                             value=pl.col("value") * pl.col("us"))
                     .sort("node", "period", "step"))
            if level.height == 0:
                level = None
    scopes = _self_discharge.scope_start_levels(flex_data, state, start)

    def values(var: str) -> "pl.DataFrame | None":
        if var not in sol._vars:
            return None
        frame = sol.value(var)
        if frame is None or frame.height == 0:
            return None
        return frame.with_columns([pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value"])

    # A representative-period node's start is the start of its base-period chain (start_levels leaves it out).
    chain = _self_discharge.rp_chain_start_levels(flex_data, values)
    if chain is not None:
        start = chain.select("node", pl.col("value").cast(pl.Float64)) if start is None else pl.concat(
            [start.select("node", pl.col("value").cast(pl.Float64)),
             chain.select("node", pl.col("value").cast(pl.Float64))], how="vertical")
    return start, level, scopes
