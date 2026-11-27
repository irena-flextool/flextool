"""Loader of the storage feature."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from polar_high import Param
from flextool.engine.input._axis_enums import cast_dim, get_global_axis_enums, rename_to_axis
from flextool.engine.preprocess.params._block_layout import BlockLayout
from flextool.engine.preprocess.flex_data._readers import (
    _provider_has,
    _provider_read,
    _read_capacity,
    _read_step_previous,
    _read_unitsize,
    _roll_end_tied_nodes,
    _split_period_branch,
    _within_period_lag,
)
from flextool.engine.preprocess.flex_data._load_features import _load_roll_period_whole


def _load_storage(inp: Path, sd: Path, dt: pl.DataFrame,
                   nb: pl.DataFrame,
                   nbp: pl.DataFrame | None,
                   pss_eff: pl.DataFrame | None,
                   pss_noEff: pl.DataFrame | None,
                   cap_pd: pl.DataFrame | None,
                   unitsize: Param | None,
                   block_layout: "BlockLayout | None" = None,
                   *,
                   provider: "object | None" = None) -> dict:
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
    # ``source`` is the entity-union (``e``) enum; ``nb["n"]`` is the
    # narrower node (``n``) enum.  Use a semi-join keyed on the node
    # rename (cast against ``e`` via rename_to_axis) so the membership
    # filter survives Phase 4 activation without a cross-vocab is_in.
    # ``n`` is then aliased from the surviving e-typed source column
    # (all surviving values are nodes per the semi-join, but the dtype
    # stays e so the downstream block-compat join with eb_l.n (also
    # cast to e below) composes natively in Enum).
    _enums_local = get_global_axis_enums()
    # Membership set for the source-side semi-join is the UNION of the
    # per-(d,t) balance nodes (``nb``) and the per-period balance nodes
    # (``nbp``, legacy ``balance_within_period``): a process whose source
    # is *either* kind of balance node must still contribute its flow to
    # that node's balance.  ``nbp`` carries the identical ``n``-column
    # dtype as ``nb`` (both produced by the same ``rename_to_axis
    # {"node": "n"}``), so the concat is dtype-safe with no extra cast.
    #
    # Byte-identity guarantee: when ``nbp`` is None or empty (every
    # fixture except the new period-node ones) the union collapses to
    # ``nb`` verbatim — we take the ``nb`` branch unchanged, with no
    # concat and no ``.unique()`` reorder.  The semi-join only consumes
    # the distinct ``source`` membership and emits rows in left-frame
    # (``pss_*``) order, so even when the union path runs it cannot
    # reorder or duplicate the existing ``nb``-only matches.
    if nbp is not None and nbp.height > 0:
        nb_source_set = (pl.concat([nb.select("n"), nbp.select("n")])
                         .unique())
    else:
        nb_source_set = nb
    nb_as_source = nb_source_set.lazy().pipe(rename_to_axis, {"n": "source"})
    if pss_eff is not None:
        flow_from_nb_eff = (pss_eff.lazy()
            .join(nb_as_source.select("source"), on="source", how="semi")
            .with_columns(n=cast_dim(pl.col("source"), _enums_local, "e"))
            .select("p","source","sink","n")
            .collect())
    if pss_noEff is not None:
        flow_from_nb_noEff = (pss_noEff.lazy()
            .join(nb_as_source.select("source"), on="source", how="semi")
            .with_columns(n=cast_dim(pl.col("source"), _enums_local, "e"))
            .select("p","source","sink","n")
            .collect())

    # dtttdt — needed for ramps and online dynamics regardless of storage.
    dtttdt = _read_step_previous(sd / "step_previous.csv", provider=provider)

    blank = dict(
        nodeState = None, nodeState_dt = None, nodeState_first_dt = None,
        p_state_upper = None, p_state_unitsize = None,
        p_state_self_discharge = None, p_state_start = None,
        p_state_existing_capacity = None,
        storage_bind_within_timeblock = None,
        storage_bind_forward_only = None,
        storage_fix_start = None,
        storage_fix_end = None,
        storage_solve_at_model_end = True,
        storage_rp_chain_at_model_end = True,
        storage_roll_end = None,
        storage_use_reference_value = None,
        p_storage_state_reference_value = None,
        p_storage_state_reference_price = None,
        dtttdt = dtttdt,
        dtttdt_forward_only = None,
        nodeStateBlock = None,
        period_block = None,
        period_block_succ = None,
        period_block_time = None,
        dtttdt_block_interior = None,
        period_block_chain = None,
        nodeState_rp = None,
        rp_base_period_set = None,
        rp_base_chain = None,
        rp_base_first = None,
        rp_base_last = None,
        rp_block_first = None,
        p_rp_last_step = None,
        rp_base__rep = None,
        rp_period_previous = None,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = None,
        p_roll_continue_state = None,
        n_fix_storage_quantity = None,
        ndt_fix_storage_quantity = None,
        p_fix_storage_quantity = None,
        n_fix_storage_usage = None,
        ndt_fix_storage_usage = None,
        p_fix_storage_usage = None,
        ndt_fix_storage_price = None,
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
    if not _provider_has(provider, "solve_data/nodeState", ns_path):
        return blank
    nodeState = _provider_read(provider, "solve_data/nodeState", ns_path).pipe(rename_to_axis, {"node": "n"})
    if nodeState.height == 0:
        return blank

    # Phase E.3: ``nodeState_dt`` is no longer materialised here.
    # Consumers call ``_pdt_join.compute_nodeState_dt`` on demand.
    # ``nodeState_first_dt`` below stays materialised — it's a small
    # one-row-per-node slice and the CSV-fallback resolution for
    # ``first_period`` needs the slow-path provider context.

    # First (d, t) per period — used for the start of the storage chains.
    # The .mod uses ``period_first_of_solve`` for the boundary tests in
    # both the fwd_fix start binding (mod:2197) and the roll_continue
    # term (mod:2196).  ``period_first.csv`` is the legacy single-solve
    # source (often empty in nested / rolling-horizon fixtures), so we
    # prefer ``period_first_of_solve.csv`` when it has rows; otherwise
    # fall back to ``period_first.csv``; otherwise the first dt period.
    # TODO(Δ.18+): no canonical helper yet for ``nodeState_first_dt`` —
    # the override-chain produces ``period_branch`` / ``dtt_timeline_matching``
    # but not the simple per-(n, d) first timestep.  ``_read_period_first``
    # in ``_derived`` reads similar data but is workdir-only and
    # used by the existing-chain helper.
    fpos_path = sd / "period_first_of_solve.csv"
    fp_path = sd / "period_first.csv"
    first_period = None
    if _provider_has(provider, "solve_data/period_first_of_solve", fpos_path):
        df = _provider_read(provider, "solve_data/period_first_of_solve", fpos_path)
        if df.height > 0:
            first_period = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()
    if first_period is None and _provider_has(provider, "solve_data/period_first", fp_path):
        df = _provider_read(provider, "solve_data/period_first", fp_path)
        if df.height > 0:
            first_period = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()
    if first_period is None:
        # Fallback: take the lexicographically smallest period.
        first_period = (dt.select("d").unique()
                          .sort("d").head(1))
    # Phase E.3: build ``first_dt`` lazily from ``nodeState`` × ``dt``
    # without persisting the full cross-product on ``flex_data``.
    first_dt = (
        nodeState.lazy()
        .join(dt.lazy(), how="cross")
        .join(first_period.lazy(), on="d", how="inner")
        .group_by("n", "d")
        .agg(pl.col("t").min().alias("t"))
        .select("n", "d", "t")
        .sort("n", "d")  # group_by order varies run to run
        .collect()
    )

    # Δ.18 — restore CSV-fallback seeds for ``p_state_existing_capacity``
    # / ``p_state_unitsize`` / ``p_state_upper``.  They were dropped in
    # Δ.17 batch 1 because ``apply_derived_e`` was authoritative — but
    # for synthetic per-sub-solve fixtures the per-solve override chain
    # is skipped (workdir's ``solve_current.csv`` names a solve not in
    # Spine, see ``_apply_db_overrides`` Δ.18 gate) and the snapshot CSV
    # is the only source.  When the override does run, it overlays
    # these via ``setattr`` so the seed becomes inert.
    if unitsize is not None and cap_pd is not None and nodeState is not None:
        cap_long = _read_capacity(sd / "p_entity_all_existing.csv", provider=provider)
        unitsize_long = _read_unitsize(sd / "p_entity_unitsize.csv", provider=provider)
        state_existing = (cap_long.pipe(rename_to_axis, {"e": "n", "value": "cap"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n", "d", "cap"))
        state_us_long = (unitsize_long.pipe(rename_to_axis, {"e": "n"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n", "value"))
        if state_existing.height > 0 and state_us_long.height > 0:
            state_existing_capacity = Param(("n", "d"),
                state_existing.rename({"cap": "value"}))
            state_unitsize = Param(("n",), state_us_long)
            state_upper_long = (state_existing
                .join(state_us_long.rename({"value": "us"}), on="n", how="inner")
                .with_columns(value=pl.col("cap") / pl.col("us"))
                .select("n", "d", "value"))
            state_upper = Param(("n", "d"), state_upper_long)
        else:
            state_unitsize = state_existing_capacity = state_upper = None
    else:
        state_unitsize = state_existing_capacity = state_upper = None

    # Δ.12-drop: ``state_self_discharge`` (``p_state_self_discharge``) and
    # ``state_start`` (``p_state_start``) seeds dropped — both are now
    # produced authoritatively by ``apply_direct_params`` via
    # ``p_state_self_discharge_from_source`` / ``p_state_start_from_source``.
    state_self_discharge = None
    state_start = None

    # Binding methods (sd-level, per node).
    # NOTE: the .mod attaches a (v_state[t] - v_state[t-1]) term in
    # nodeBalance for several binding methods, with subtle differences:
    #   * ``bind_within_timeblock``  — fully cyclic (wraps via
    #     ``t_previous_within_timeset``).
    #   * ``bind_within_period``   — cyclic within period: the first step of
    #     each period wraps to the period's own last step
    #     (``dtttdt_within_period``, built below).
    #   * ``bind_within_solve``    — cyclic within solve
    #     (``t_previous_within_solve``); equivalent to within_timeset for
    #     a single-block dispatch.
    #   * ``bind_forward_only``    — also uses
    #     ``t_previous_within_solve``, BUT the .mod *omits* the
    #     state-change term at the first timestep of the first period
    #     (line 2188 condition).  This makes the storage non-cyclic at
    #     the boundary.
    # In the current parity tests, every fixture that *does*
    # exercise a state node uses ``bind_within_timeblock`` (which is what
    # this loader picks up).  ``work_water_pump`` is the only fixture
    # that uses ``bind_forward_only``, and faithful parity there
    # requires modelling the first-timestep exemption — see
    # questions_for_user.md#water_pump.
    sbm_path = sd / "node__storage_binding_method.csv"
    binding_within_timeblock = None
    binding_forward_only = None
    binding_within_solve = None
    binding_within_period = None
    # Phase C — ``binding_within_solve_blended_weights`` is also derived
    # here from the per-solve provider so the partition reflects any
    # cascade-applied silent downgrade (see
    # :func:`._run_model._rp_methods._downgrade_rp_methods_for_non_rp_solve`).
    # apply_projection_params no longer touches this field — the
    # per-solve CSV is the single source of truth.
    binding_within_solve_blended_weights = None
    # Phase C — not-yet-implemented methods.  Loaded so the model/core/
    # guard can surface a precise error instead of silently emitting
    # zero state-change residuals.  Constraint wiring lands in Phases
    # D / E of the storage-binding-methods restructure.
    binding_within_period_blended_weights = None
    binding_forward_only_blended_weights = None
    if _provider_has(provider, "solve_data/node__storage_binding_method", sbm_path):
        sbm = _provider_read(provider, "solve_data/node__storage_binding_method", sbm_path)
        # Column names in this file have varied — handle both schemas
        if "storage_binding_method" in sbm.columns:
            sbm = sbm.pipe(rename_to_axis, {"node":"n","storage_binding_method":"method"})
        elif "method" in sbm.columns:
            sbm = sbm.pipe(rename_to_axis, {"node":"n"})
        binding_within_timeblock = (sbm.filter(pl.col("method")=="bind_within_timeblock")
                                     .select("n").unique(maintain_order=True))
        fo = (sbm.filter(pl.col("method")=="bind_forward_only")
                 .select("n").unique(maintain_order=True))
        if fo.height > 0:
            binding_forward_only = fo
        ws = (sbm.filter(pl.col("method")=="bind_within_solve")
                 .select("n").unique(maintain_order=True))
        if ws.height > 0:
            binding_within_solve = ws
        wp = (sbm.filter(pl.col("method")=="bind_within_period")
                 .select("n").unique(maintain_order=True))
        if wp.height > 0:
            binding_within_period = wp
        wsbw = (sbm.filter(pl.col("method")=="bind_within_solve_blended_weights")
                   .select("n").unique(maintain_order=True))
        if wsbw.height > 0:
            binding_within_solve_blended_weights = wsbw
        wpbw = (sbm.filter(pl.col("method")=="bind_within_period_blended_weights")
                   .select("n").unique(maintain_order=True))
        if wpbw.height > 0:
            binding_within_period_blended_weights = wpbw
        fobw = (sbm.filter(pl.col("method")=="bind_forward_only_blended_weights")
                   .select("n").unique(maintain_order=True))
        if fobw.height > 0:
            binding_forward_only_blended_weights = fobw

    # ``bind_forward_only`` mirrors ``bind_within_solve`` (uses the
    # ``t_previous_within_solve`` lag column) BUT the .mod omits the
    # state-change term at the very first timestep of the first period
    # (flextool.mod:2188).  We model that exemption by dropping the
    # corresponding row from the lag frame — the wrap row whose
    # ``t_previous_within_solve`` jumps backwards.  Sorting by (d, t)
    # and dropping the first row is equivalent for single-solve fixtures
    # (the native engine is single-solve per build).
    dtttdt_forward_only_df = None
    if binding_forward_only is not None and dtttdt is not None and dtttdt.height > 0:
        from flextool.engine.preprocess.params._derived import (
            dtttdt_forward_only_from_dtttdt, read_period_first_of_solve)
        dtttdt_forward_only_df = dtttdt_forward_only_from_dtttdt(
            dtttdt, read_period_first_of_solve(sd, provider=provider))

    # The seed value of the bind_within_solve lag.  _derived.apply_derived_e recomputes it from the source dtttdt
    # with the same dtttdt_within_solve_from_dtttdt and overwrites this one: that value is authoritative, and the
    # two must agree (same function, same period_first_of_solve).
    dtttdt_within_solve_df = None
    if binding_within_solve is not None and dtttdt is not None and dtttdt.height > 0:
        from flextool.engine.preprocess.params._derived import (
            dtttdt_within_solve_from_dtttdt, read_period_first_of_solve)
        dtttdt_within_solve_df = dtttdt_within_solve_from_dtttdt(
            dtttdt, read_period_first_of_solve(sd, provider=provider))

    dtttdt_within_period_df = None
    if binding_within_period is not None and dtttdt is not None and dtttdt.height > 0:
        dtttdt_within_period_df = _within_period_lag(dtttdt, _split_period_branch(sd, provider))

    # node__storage_start_end_method is read by .mod from input/
    # (flextool.mod:662) — that's the canonical user-input source.
    # solve_data/ may have a .mod-printf debug-export with renamed
    # column "method"; tolerate either schema.
    sse_path = inp / "node__storage_start_end_method.csv"
    sse_key = "input/node__storage_start_end_method"
    if not _provider_has(provider, sse_key, sse_path):
        sse_path = sd / "node__storage_start_end_method.csv"
        sse_key = "solve_data/node__storage_start_end_method"
    fix_start = None
    fix_end = None
    if _provider_has(provider, sse_key, sse_path):
        sse = _provider_read(provider, sse_key, sse_path)
        if "storage_start_end_method" in sse.columns:
            sse = sse.pipe(rename_to_axis, {"node":"n","storage_start_end_method":"method"})
        elif "method" in sse.columns:
            sse = sse.pipe(rename_to_axis, {"node":"n"})
        # fix_start_end fixes both the start and the end.
        fix_start = (sse.filter(pl.col("method").is_in(["fix_start", "fix_start_end"])).select("n").unique())
        fix_end = (sse.filter(pl.col("method").is_in(["fix_end", "fix_start_end"])).select("n").unique())
        if fix_end.height == 0:
            fix_end = None

    # Whether this sub-solve's horizon holds the last step of the entire model (_run_model): the end pin of fix_end /
    # fix_start_end applies there only.  Default True (a single solve, a FlexData built without the cascade).
    from flextool.engine.preprocess import _provider_keys as K
    at_model_end = True
    if _provider_has(provider, K.SOLVE_DATA_STORAGE_SOLVE_AT_MODEL_END, sd / "storage_solve_at_model_end.csv"):
        ssl = _provider_read(provider, K.SOLVE_DATA_STORAGE_SOLVE_AT_MODEL_END, sd / "storage_solve_at_model_end.csv")
        if ssl.height > 0 and "value" in ssl.columns:
            at_model_end = bool(ssl["value"].cast(pl.Boolean)[0])
    rp_chain_at_model_end = at_model_end
    if _provider_has(provider, K.SOLVE_DATA_STORAGE_RP_CHAIN_AT_MODEL_END, sd / "storage_rp_chain_at_model_end.csv"):
        rce = _provider_read(provider, K.SOLVE_DATA_STORAGE_RP_CHAIN_AT_MODEL_END,
                             sd / "storage_rp_chain_at_model_end.csv")
        if rce.height > 0 and "value" in rce.columns:
            rp_chain_at_model_end = bool(rce["value"].cast(pl.Boolean)[0])

    # The horizon-end condition of the rolling bind_within_solve nodes continued from a carried state (_run_model).
    roll_end = None
    if _provider_has(provider, K.SOLVE_DATA_STORAGE_ROLL_END, sd / "storage_roll_end.csv"):
        sre = _provider_read(provider, K.SOLVE_DATA_STORAGE_ROLL_END, sd / "storage_roll_end.csv")
        if sre.height > 0:
            roll_end = (sre.pipe(rename_to_axis, {"node": "n"})
                        .select("n", pl.col("kind").cast(pl.Utf8),
                                pl.col("value").cast(pl.Float64, strict=False),
                                (pl.col("period").cast(pl.Utf8) if "period" in sre.columns
                                 else pl.lit(None, dtype=pl.Utf8)).alias("period")))

    # Rolling sub-solve: the bind_intraperiod_blocks blocks of a period the roll holds only in part chain forward.
    solve_is_rolling = False
    period_block_chain = None
    rpw = _load_roll_period_whole(sd, provider=provider)
    if rpw is not None:
        solve_is_rolling = True
        chained = rpw.filter(~pl.col("whole").cast(pl.Boolean)).pipe(rename_to_axis, {"period": "d"})
        if chained.height > 0:
            period_block_chain = chained.select("d").unique(maintain_order=True)

    # node__storage_solve_horizon_method (.mod:663): nodes with method
    # ``use_reference_value`` get a v_state pin at the last timestep of
    # the last period, equal to ``reference_value × existing capacity / unitsize``.
    sshm_path = inp / "node__storage_solve_horizon_method.csv"
    sshm_key = "input/node__storage_solve_horizon_method"
    if not _provider_has(provider, sshm_key, sshm_path):
        sshm_path = sd / "node__storage_solve_horizon_method.csv"
        sshm_key = "solve_data/node__storage_solve_horizon_method"
    use_reference_value = None
    if _provider_has(provider, sshm_key, sshm_path):
        sshm = _provider_read(provider, sshm_key, sshm_path)
        col = ("storage_solve_horizon_method"
               if "storage_solve_horizon_method" in sshm.columns
               else "method")
        sshm = sshm.pipe(rename_to_axis, {"node": "n", col: "method"})
        use_reference_value = (sshm
            .filter(pl.col("method") == "use_reference_value")
            .select("n").unique())
        # Filter out nodes with a competing storage method (mod:2806-2811):
        # fix_end / fix_start_end / bind_within_solve / bind_within_period /
        # bind_within_timeblock / bind_intraperiod_blocks.
        # ``nodeStateBlock`` is the set carrying bind_intraperiod_blocks
        # (loaded below; we look it up via the on-disk CSV here to keep
        # ordering simple).  In a rolling solve a bind_intraperiod_blocks node keeps the pin: it applies where the
        # roll's last period chains forward (model.core._storage_state._eq_storage_reference_value), not where its blocks loop.
        nsb_for_excl = None
        nsb_path_local = sd / "nodeStateBlock.csv"
        if not solve_is_rolling and _provider_has(provider, "solve_data/nodeStateBlock", nsb_path_local):
            nsb_df_local = _provider_read(provider, "solve_data/nodeStateBlock", nsb_path_local)
            if nsb_df_local.height > 0:
                nsb_for_excl = nsb_df_local.pipe(rename_to_axis, {"node": "n"}).select("n")
        # A rolling bind_within_solve node continued from a carried state chains forward in the roll and its end is
        # tied to its start by storage_roll_end (_run_model._rolling_storage._anchor_cyclic_in_rolling): no reference
        # pin.
        for excl in (fix_end,
                     binding_within_solve, binding_within_period, binding_within_timeblock,
                     nsb_for_excl, _roll_end_tied_nodes(roll_end)):
            if excl is not None and excl.height > 0:
                use_reference_value = use_reference_value.join(
                    excl, on="n", how="anti")
        if use_reference_value.height == 0:
            use_reference_value = None

    # Δ.17c Gap C: ``p_storage_state_reference_value`` produced
    # authoritatively by ``apply_direct_params`` via
    # ``p_storage_state_reference_value_from_source`` (uses the
    # ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade).  The
    # ``use_reference_value`` consumer-side filter runs in apply_direct_params
    # — kept here as None so the override path takes ownership.
    p_ssrv = None

    # B1a — load ``p_storage_state_reference_price`` (n, d) from the canonical
    # ``solve_data/p_storage_state_reference_price.csv`` emitted by
    # ``_arc_unions.emit_p_storage_state_reference_price``.  Schema is
    # ``(node, period, value)``; per-(node, period) parameter consumed by
    # B1b's ``use_reference_price`` objective term.  Field is loaded but
    # unused at this commit; B1b adds the consumer.
    p_storage_state_reference_price = None
    psrp_path = sd / "p_storage_state_reference_price.csv"
    if _provider_has(provider, "solve_data/p_storage_state_reference_price",
                     psrp_path):
        df_psrp = _provider_read(
            provider, "solve_data/p_storage_state_reference_price", psrp_path,
        )
        if df_psrp.height > 0:
            value_col = ("p_storage_state_reference_price"
                         if "p_storage_state_reference_price" in df_psrp.columns
                         else "value")
            df_psrp = (df_psrp
                       .pipe(rename_to_axis,
                             {"node": "n", "period": "d",
                              value_col: "value"})
                       .with_columns(value=pl.col("value")
                                              .cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                       .select("n", "d", "value"))
            if df_psrp.height > 0:
                p_storage_state_reference_price = Param(("n", "d"), df_psrp)

    # ─── Intraperiod-block (bind_intraperiod_blocks) sets ────────────────
    # Used by ``stateConstantWithinBlock_eq`` and ``nodeBalanceBlock_eq``
    # in model/core/ for nodes whose binding method is ``bind_intraperiod_blocks``.
    # Δ.18 — CSV-fallback seed for ``nodeStateBlock`` (the bind_intraperiod_blocks
    # nodes).  ``apply_derived_e`` (``nodeStateBlock_from_source``) overlays the
    # same set when active; for dump_csvs roundtrip workdirs (no DB) the
    # canonical CSV is the only source.
    nodeStateBlock = None
    nsb_path_seed = sd / "nodeStateBlock.csv"
    if _provider_has(provider, "solve_data/nodeStateBlock", nsb_path_seed):
        nsb_df_seed = _provider_read(provider, "solve_data/nodeStateBlock", nsb_path_seed)
        if nsb_df_seed.height > 0 and "node" in nsb_df_seed.columns:
            nodeStateBlock = nsb_df_seed.pipe(rename_to_axis, {"node": "n"}).select("n").unique()
    period_block = None
    period_block_succ = None
    period_block_time = None
    dtttdt_block_interior = None

    # ─── RP-blended-weights (bind_within_solve_blended_weights) sets / params ───
    # Eight per-solve frames driving the .mod's intra-period state-change
    # branch for ``nodeState_rp`` plus the three ``rp_inter_period_*``
    # constraints (.mod:2197-2200, .mod:2965-2997).  Emitted by
    # ``_leaf_sets.emit_node_state_subsets`` (nodeState_rp),
    # ``_per_solve`` (rp_base_period_set), and
    # ``_solve_writers._compute_rp_frames`` (chain / first / last /
    # block_first / block_start_last / weights).  Loaded here; model/core/
    # wiring lands in Phase 5+.
    from flextool.engine.preprocess import _provider_keys as _K_rp
    nodeState_rp = None
    nsrp_path = sd / "nodeState_rp.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_NODE_STATE_RP, nsrp_path):
        df_nsrp = _provider_read(
            provider, _K_rp.SOLVE_DATA_NODE_STATE_RP, nsrp_path,
        )
        if df_nsrp.height > 0 and "node" in df_nsrp.columns:
            nodeState_rp = (df_nsrp
                            .pipe(rename_to_axis, {"node": "n"})
                            .select("n").unique(maintain_order=True))
            if nodeState_rp.height == 0:
                nodeState_rp = None

    # Every frame carries the owning period ``d``: two periods of a solve may share a timeset or a timeline
    # (the same base-period names), and each period has its own base-period chain, blocks and weights.
    def _rp_frame(key: str, basename: str, axis: dict, plain: dict, keep: tuple) -> "pl.DataFrame | None":
        """The frame at ``key``: ``axis`` columns renamed onto their axis (typed), ``plain`` ones renamed as
        text (base-period / rep-start labels, cast at their use site in model/core/)."""
        path = sd / basename
        if not _provider_has(provider, key, path):
            return None
        df = _provider_read(provider, key, path)
        if df.height == 0:
            return None
        df = df.rename(plain).pipe(rename_to_axis, axis).select(*keep).unique(maintain_order=True)
        return df if df.height > 0 else None

    rp_base_period_set = _rp_frame(_K_rp.SOLVE_DATA_RP_BASE_PERIOD_SET, "rp_base_period_set.csv",
                                   {"period": "d"}, {"base_start": "b"}, ("d", "b"))
    rp_base_chain = _rp_frame(_K_rp.SOLVE_DATA_RP_BASE_CHAIN, "rp_base_chain.csv",
                              {"period": "d"}, {"base_start": "b", "prev_base_start": "b_prev"}, ("d", "b", "b_prev"))
    rp_base_first = _rp_frame(_K_rp.SOLVE_DATA_RP_BASE_FIRST, "rp_base_first.csv",
                              {"period": "d"}, {"base_start": "b"}, ("d", "b"))
    rp_base_last = _rp_frame(_K_rp.SOLVE_DATA_RP_BASE_LAST, "rp_base_last.csv",
                             {"period": "d"}, {"base_start": "b"}, ("d", "b"))
    rp_block_first = _rp_frame(_K_rp.SOLVE_DATA_RP_BLOCK_FIRST, "rp_block_first.csv",
                               {"period": "d", "step": "t"}, {}, ("d", "t"))
    # ``p_rp_last_step`` — a relation (d, r, last_step), NOT a numeric Param: block r of period d ends at
    # last_step (model/core/ joins on (d, r) → last_step).
    p_rp_last_step = _rp_frame(_K_rp.SOLVE_DATA_RP_BLOCK_START_LAST, "rp_block_start_last.csv",
                               {"period": "d"}, {"rep_start": "r"}, ("d", "r", "last_step"))
    # ``rp_period_previous`` (d, d_prev): the across-solve chain of the rp periods in solve order.
    rp_period_previous = _rp_frame(_K_rp.SOLVE_DATA_RP_PERIOD_PREVIOUS, "rp_period_previous.csv",
                                   {"period": "d"}, {"prev_period": "d_prev"}, ("d", "d_prev"))

    # ``rp_base__rep`` — Param keyed by (d, b, r) with the weight of rep block r in base period b of period d
    # (``rp_weights.csv``: period, base_start, rep_start, weight as a stringified float).
    rp_base__rep = None
    rpw_path = sd / "rp_weights.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_WEIGHTS, rpw_path):
        df_rpw = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_WEIGHTS, rpw_path,
        )
        if df_rpw.height > 0:
            df_rpw = (df_rpw
                      .rename({"base_start": "b",
                               "rep_start": "r",
                               "weight": "value"})
                      .pipe(rename_to_axis, {"period": "d"})
                      .with_columns(value=pl.col("value")
                                              .cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                      .select("d", "b", "r", "value"))
            if df_rpw.height > 0:
                rp_base__rep = Param(("d", "b", "r"), df_rpw)

    # ─── Invariant: when nodeState_rp is non-empty, the four tightly
    # coupled fields below must also be non-empty.  Silent absence has no
    # graceful degradation — the model just emits no inter-period state
    # constraint and the LP solves the wrong problem.  Mirrors the
    # ``_fast_load.py:686-690`` block invariant pattern.
    if nodeState_rp is not None and nodeState_rp.height > 0:
        _rp_required = (
            ("rp_base_period_set", rp_base_period_set),
            ("rp_base__rep", rp_base__rep),
            ("rp_block_first", rp_block_first),
            ("p_rp_last_step", p_rp_last_step),
        )
        for _name, _field in _rp_required:
            _frame = (_field.frame if hasattr(_field, "frame") else _field)
            if _frame is None or _frame.height == 0:
                raise ValueError(
                    f"FlexData loader (backstop check): nodeState_rp is "
                    f"non-empty ({nodeState_rp.height} node(s)) but the "
                    f"tightly-coupled field `{_name}` is missing or "
                    f"empty.  Under the storage-binding-methods-restructure "
                    f"(Phase C onwards), the per-solve downgrade in "
                    f"_run_model._rp_methods._downgrade_rp_methods_for_non_rp_solve "
                    f"normally strips bind_within_solve_blended_weights "
                    f"to bind_within_solve when a solve's active "
                    f"timeset has no representative_period_weights.  "
                    f"If you are seeing THIS error, the downgrade was "
                    f"bypassed — either load_flextool was called "
                    f"directly (not via _run_model) or there "
                    f"is an upstream emitter bug.  Please report this "
                    f"with the calling code.  Required RP set family: "
                    f"nodeState_rp, rp_base_period_set, rp_base__rep, "
                    f"rp_block_first, p_rp_last_step."
                )

    # ─── Timeset blocks of bind_intraperiod_blocks nodes ─────────────────
    # Built by the override chain (``period_block_family_from_source`` +
    # ``nodeStateBlock_from_source`` + ``dtttdt_block_interior_from_dtttdt``).
    # The coarse new_stepduration group frames are set by
    # ``_apply_coarse_frames`` at the end of ``load_flextool``.

    # ─── Rolling-horizon nested-solve framework (flextool.mod:2196 + 2760) ─
    # p_nested_model.csv: { modelParam, p_nested_model } with rows
    # solveFirst / solveLast.  Tri-state: missing → None (single-solve);
    # 0 → False; non-zero → True.
    p_nested_solve_first: bool | None = None
    nm_path = sd / "p_nested_model.csv"
    if _provider_has(provider, "solve_data/p_nested_model", nm_path):
        nm = _provider_read(provider, "solve_data/p_nested_model", nm_path)
        if nm.height > 0:
            # Column may be ``p_nested_model`` (canonical) or ``value``.
            value_col = "p_nested_model" if "p_nested_model" in nm.columns else "value"
            row = nm.filter(pl.col("modelParam") == "solveFirst")
            if row.height > 0:
                p_nested_solve_first = bool(int(row[value_col][0]))

    # Phase 4.2-0 — read the rolling end-state carrier from the canonical
    # handoff Provider key (``handoff/roll_end_state``, schema
    # ``[node, value]``).  The translator populates the key from
    # ``SolveHandoff.roll_end_state`` at iteration start; the legacy
    # ``solve_data/p_roll_continue_state`` CSV-fallback path is gone.
    #
    # The cascade routes both keys (``solve._carry_routing``): they hold the same frame, the latest state realised
    # before this sub-solve at its nesting level or deeper (which gives the upward feedback of
    # specs/feature_fixes.md §1: a storage roll continues from the dispatch realised below it).  An external
    # override of either key wins (read_handoff_frame); the upward key is preferred when both are populated.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    p_roll_continue_state = None
    df_rcs = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STATE)
    df_upward = read_handoff_frame(provider, K.HANDOFF_UPWARD_ROLL_END_STATE)
    if df_upward is not None and df_upward.height > 0:
        df_rcs = df_upward
    if df_rcs is not None and df_rcs.height > 0:
        df_rcs = (df_rcs
                  .pipe(rename_to_axis, {"node": "n"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "value"))
        if df_rcs.height > 0:
            p_roll_continue_state = Param(("n",), df_rcs)

    # Phase 4.1d — read the rolling-handoff fix_storage_quantity carrier
    # from the canonical handoff Provider key
    # (``handoff/fix_storage_quantity``, schema
    # ``[node, period, step, p_fix_storage_quantity]``).  The translator
    # populates the key from ``SolveHandoff.fix_storage_quantity`` at
    # iteration start; the legacy ``solve_data/fix_storage_quantity``
    # read path is gone.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    p_fix_storage_quantity = None
    df_fsq = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_QUANTITY)
    if df_fsq is not None and df_fsq.height > 0:
        df_fsq = (df_fsq
                  .pipe(rename_to_axis, {"node": "n", "period": "d",
                                         "step": "t",
                                         "p_fix_storage_quantity": "value"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "d", "t", "value"))
        if df_fsq.height > 0:
            p_fix_storage_quantity = Param(("n", "d", "t"), df_fsq)

    # ``n_fix_storage_quantity`` / ``ndt_fix_storage_quantity`` are
    # derived from ``p_fix_storage_quantity`` after the helper assigns it
    # (see ``_finalise_fix_storage_index_sets`` below).
    n_fix_storage_quantity = None
    ndt_fix_storage_quantity = None
    if p_fix_storage_quantity is not None:
        fsq_frame = p_fix_storage_quantity.frame
        n_fix_storage_quantity = fsq_frame.select("n").unique(maintain_order=True)
        ndt_fix_storage_quantity = fsq_frame.select("n", "d", "t").unique(maintain_order=True)

    # Phase B4-pre — read the rolling-handoff fix_storage_usage carrier
    # from the canonical handoff Provider key
    # (``handoff/fix_storage_usage``, schema
    # ``[node, period, step, p_fix_storage_usage]``).  Populated by B3's
    # extractor; consumed by the B4 constraint (added in the next commit).
    # Loader-only at this commit — fields exist on FlexData but are not
    # yet referenced by any LP block, so zero LP effect.
    p_fix_storage_usage = None
    df_fsu = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_USAGE)
    if df_fsu is not None and df_fsu.height > 0:
        df_fsu = (df_fsu
                  .pipe(rename_to_axis, {"node": "n", "period": "d",
                                         "step": "t",
                                         "p_fix_storage_usage": "value"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "d", "t", "value"))
        if df_fsu.height > 0:
            p_fix_storage_usage = Param(("n", "d", "t"), df_fsu)

    n_fix_storage_usage = None
    ndt_fix_storage_usage = None
    if p_fix_storage_usage is not None:
        fsu_frame = p_fix_storage_usage.frame
        n_fix_storage_usage = fsu_frame.select("n").unique()
        ndt_fix_storage_usage = fsu_frame.select("n", "d", "t").unique()

    # The steps an upper-level solve prices for a nested fix_price (handoff/fix_storage_price, upper-level (d, t)):
    # the horizon-end reference value yields to the upper solve there (storage_state_solve_horizon_reference_value).
    ndt_fix_storage_price = None
    df_fsp = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_PRICE)
    if df_fsp is not None and df_fsp.height > 0:
        df_fsp = (df_fsp.pipe(rename_to_axis, {"node": "n", "period": "d", "step": "t"})
                  .filter(pl.col("p_fix_storage_price").is_not_null())
                  .select("n", "d", "t").unique())
        if df_fsp.height > 0:
            ndt_fix_storage_price = df_fsp

    # ``dtt_timeline_matching`` (d, t, t_upper) and ``period_branch``
    # (d_upper, d) — seed from the snapshot CSVs.  ``apply_derived_e``
    # (the override-chain producer) is bypassed entirely for
    # *synthetic* per-sub-solve workdirs (their ``solve_current`` names a
    # solve that doesn't exist in Spine — the multi_fullYear_battery_nested
    # rolling-handoff snapshots are the canonical case), so without this
    # seed both fields stay ``None`` and ``node_balance_fix_quantity_eq_lower``
    # never fires for the upper-level anchor pin at the last timestep —
    # leaving the LP severely under-constrained at the rolling-handoff
    # boundary (vq_state_up slack lights up, objective explodes).
    dtt_timeline_matching = None
    tlm_path = sd / "timeline_matching_map.csv"
    if _provider_has(provider, "solve_data/timeline_matching_map", tlm_path):
        df_tlm = _provider_read(provider, "solve_data/timeline_matching_map", tlm_path)
        if df_tlm.height > 0:
            dtt_timeline_matching = (df_tlm
                .pipe(rename_to_axis,
                      {"period": "d", "step": "t", "upper_step": "t_upper"})
                .select("d", "t", "t_upper").unique(maintain_order=True))
            if dtt_timeline_matching.height == 0:
                dtt_timeline_matching = None

    period_branch = None
    pb_path = sd / "period__branch.csv"
    if _provider_has(provider, "solve_data/period__branch", pb_path):
        df_pb = _provider_read(provider, "solve_data/period__branch", pb_path)
        if df_pb.height > 0:
            period_branch = (df_pb
                .pipe(rename_to_axis, {"period": "d", "branch": "d_upper"})
                .select("d_upper", "d").unique(maintain_order=True))
            if period_branch.height == 0:
                period_branch = None

    # period_last: (d,).
    # TODO(Δ.18+): no canonical helper yet for ``period_last`` — this is
    # preprocessing-only data (flextool's per-solve last-period anchor for
    # storage/handoff binding).  The override-chain produces
    # ``dtt_timeline_matching`` / ``period_branch`` / handoff carriers but
    # not the simple ``period_last`` set frame.
    period_last_df = None
    pl_path = sd / "period_last.csv"
    if _provider_has(provider, "solve_data/period_last", pl_path):
        df = _provider_read(provider, "solve_data/period_last", pl_path)
        if df.height > 0:
            period_last_df = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique(maintain_order=True)

    # nodeState_last_dt: (n, d, t) — the last step of each period for every storage node, used as the index
    # for ``node_balance_fix_quantity_eq_lower`` and the storage usage / cyclic start rows.  Built from the
    # default block's block_period_time_last (bk, d, t) × nodeState: every node lives on the solve's steps (a
    # coarse storage node's state is constant within its last coarse step).
    nodeState_last_dt = None
    if (nodeState is not None and nodeState.height > 0
            and block_layout is not None
            and block_layout.block_period_time_last_frame.height > 0):
        from flextool.engine.preprocess.params._block import BlockBundle, nodeState_last_dt_lf
        nodeState_last_dt = nodeState_last_dt_lf(nodeState, BlockBundle(layout=block_layout)).collect()
        if nodeState_last_dt.height == 0:
            nodeState_last_dt = None

    # Δ.17 — ``node_profile_upper`` / ``node_profile_lower`` /
    # ``node_profile_fixed`` produced authoritatively by
    # ``apply_projection_params`` (Γ.2 SIMPLE_PROJECTIONS).  Verified
    # row-by-row parity across all 72 work_* fixtures with
    # node__profile__profile_method.csv.  Seed dropped (1
    # ``_read_csv_file`` call retired).
    node_profile_upper_df = node_profile_lower_df = node_profile_fixed_df = None

    # Δ.17c Gap C: ``p_node_availability`` produced authoritatively by
    # ``apply_direct_params`` via ``p_node_availability_from_source``
    # (uses the ``_param_shapes`` resolver — full broadcast cascade with
    # explicit allow-list).  Local pdtNode.csv slice dropped.
    p_node_avail = None

    return dict(
        nodeState = nodeState,
        # Phase E.3: ``nodeState_dt`` no longer materialised; consumers
        # call ``_pdt_join.compute_nodeState_dt`` on demand.
        nodeState_dt = None,
        nodeState_first_dt = first_dt,
        p_state_upper = state_upper,
        p_state_unitsize = state_unitsize,
        p_state_self_discharge = state_self_discharge,
        p_state_start = state_start,
        p_state_existing_capacity = state_existing_capacity,
        storage_bind_within_timeblock = binding_within_timeblock,
        storage_bind_forward_only = binding_forward_only,
        storage_bind_within_solve = binding_within_solve,
        storage_bind_within_period = binding_within_period,
        # Phase C — single source of truth for the RP-flavoured partition
        # is the per-solve CSV; apply_projection_params no longer
        # contributes here so the cascade's silent-downgrade rewrite
        # propagates correctly.
        storage_bind_within_solve_blended_weights = binding_within_solve_blended_weights,
        # Phase C — not-yet-implemented method partitions; consumed by
        # the model/core/ guard which raises FlexToolConfigError when
        # non-empty.
        storage_bind_within_period_blended_weights = binding_within_period_blended_weights,
        storage_bind_forward_only_blended_weights = binding_forward_only_blended_weights,
        storage_fix_start = fix_start,
        storage_fix_end = fix_end,
        storage_solve_at_model_end = at_model_end,
        storage_rp_chain_at_model_end = rp_chain_at_model_end,
        storage_roll_end = roll_end,
        storage_use_reference_value = use_reference_value,
        p_storage_state_reference_value = p_ssrv,
        p_storage_state_reference_price = p_storage_state_reference_price,
        dtttdt = dtttdt,
        dtttdt_forward_only = dtttdt_forward_only_df,
        dtttdt_within_solve = dtttdt_within_solve_df,
        dtttdt_within_period = dtttdt_within_period_df,
        nodeStateBlock = nodeStateBlock,
        period_block = period_block,
        period_block_succ = period_block_succ,
        period_block_time = period_block_time,
        dtttdt_block_interior = dtttdt_block_interior,
        period_block_chain = period_block_chain,
        nodeState_rp = nodeState_rp,
        rp_base_period_set = rp_base_period_set,
        rp_base_chain = rp_base_chain,
        rp_base_first = rp_base_first,
        rp_base_last = rp_base_last,
        rp_block_first = rp_block_first,
        p_rp_last_step = p_rp_last_step,
        rp_base__rep = rp_base__rep,
        rp_period_previous = rp_period_previous,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = p_nested_solve_first,
        p_roll_continue_state = p_roll_continue_state,
        n_fix_storage_quantity = n_fix_storage_quantity,
        ndt_fix_storage_quantity = ndt_fix_storage_quantity,
        p_fix_storage_quantity = p_fix_storage_quantity,
        n_fix_storage_usage = n_fix_storage_usage,
        ndt_fix_storage_usage = ndt_fix_storage_usage,
        p_fix_storage_usage = p_fix_storage_usage,
        ndt_fix_storage_price = ndt_fix_storage_price,
        dtt_timeline_matching = dtt_timeline_matching,
        period_branch = period_branch,
        period_last = period_last_df,
        nodeState_last_dt = nodeState_last_dt,
        node_profile_upper = node_profile_upper_df,
        node_profile_lower = node_profile_lower_df,
        node_profile_fixed = node_profile_fixed_df,
        p_node_availability = p_node_avail,
    )
