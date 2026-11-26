""":func:`load_flextool`, the single loader, and the steps it runs after the loaders (storage reference
start, integer unit-count warning, handoff overlay)."""

from __future__ import annotations

import logging
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl
from polar_high import Param

if TYPE_CHECKING:
    from flextool.engine.input._input_source import FlexInputSource, InputSource

from flextool.engine.model import _group_slack
from flextool.engine.model import _reserve
from flextool.engine.model import _delay
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _commodity_ladder
# substrate retained for Path B — see handoff
from flextool.engine.input._axis_enums import cast_flexdata_axes, rename_to_axis, set_global_axis_enums
from flextool.engine.input._param_shapes import (
    activate_stochastic_time_source,
    deactivate_stochastic_time_source,
)
from flextool.engine.input._solve_step_aggregation import (
    activate_solve_step_aggregation,
    deactivate_solve_step_aggregation,
)
from flextool.engine.preprocess.params._block_layout import BlockLayout
from flextool.engine.preprocess.params._branch import (
    activate_branch_anchor_pairs,
    deactivate_branch_anchor_pairs,
    solve_branch_anchor_pairs,
)
from flextool.engine.preprocess._co2_carry import load_co2_rolling_carry
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.emit._base_steps import NEW_STEP_BASE_STEPS
from flextool.engine.preprocess.emit import _flowgroup_accumulators as _fg
from flextool.engine.preprocess.flex_data._readers import (
    _loadflex_prof,
    _provider_has,
    _provider_read,
    _read_capacity,
    _read_step_previous,
    _read_unitsize,
    _slice_param,
)
from flextool.engine.preprocess.flex_data._container import FlexData
from flextool.engine.preprocess.flex_data._coarse import _apply_coarse_frames, _apply_coarse_process_frames
from flextool.engine.preprocess.flex_data._load_topology import (
    _load_inflow_unmargined,
    _load_node,
    _load_process_topology,
    _load_time,
)
from flextool.engine.preprocess.flex_data._load_co2 import (
    _load_co2_cap,
    _load_co2_cap_total,
    _load_co2_price,
    _load_startup_inputs,
    _startup_co2_sets,
)
from flextool.engine.preprocess.flex_data._load_features import (
    _load_cumulative_invest,
    _load_fixed_cost,
    _load_indirect,
    _load_invest,
    _load_node_capacity_for_scaling,
    _load_period_invest,
    _load_profiles,
    _load_roll_period_whole,
    _load_stochastics,
    _load_user_constraints,
    _load_varcost,
)
from flextool.engine.preprocess.flex_data._load_ramp_online import _load_online, _load_ramp, _read_process_exact
from flextool.engine.preprocess.flex_data._load_storage import _load_storage
from flextool.engine.preprocess.flex_data._db_overrides import _apply_db_overrides
from flextool.engine.preprocess.flex_data._handoff_inputs import _load_handoff_aux_pair


def _assign_param_names(data: "FlexData") -> "FlexData":
    """Stamp the FlexData attribute name onto every :class:`Param` field.

    Enables :class:`polar_high.WarmProblem`'s Param-tracked auto-update by
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


def _years_represented_boundary(dt: "pl.DataFrame | None", db_reader, provider,
                                workdir: "Path | str | None") -> "Param | None":
    """``p_years_represented_d`` over the periods of ``dt`` and the period labels for a sub-solve that skipped
    ``apply_derived_a``.

    Built only from the provider's canonical ``solve_data/p_years_represented.csv`` (the parent solve's
    years, also for stochastic branch periods).  Without a provider or without that CSV the source arm
    would give every period one year silently, so the Param stays None: a ``co2_max_total`` build then
    raises and the outputs use their own fallback.  A header-only CSV (``years_represented`` not
    authored) deliberately keeps the default of one year per period.
    """
    from flextool.engine.preprocess.params import _derived as _drv_yr

    if (dt is None or dt.height == 0
            or not _drv_yr._provider_has_key(provider, Path("solve_data") / "p_years_represented.csv")):
        return None
    active = (_drv_yr._read_active_solve(Path(workdir), provider=provider)
              if workdir is not None else None)
    return _drv_yr.p_years_represented_d_from_source(
        db_reader, active, _drv_yr.bookkeeping_periods(dt, workdir, provider=provider), provider=provider)


def _validate_has_balance(flex_data: "FlexData") -> None:
    """Raise when no node of the solve has an energy balance (a user's model, at load; a reduced FlexData that
    the engine derives from a loaded one, such as a Benders master holding only flows between regions, may have
    none and still builds)."""
    nb = flex_data.nodeBalance
    nbp = flex_data.nodeBalancePeriod
    if (nb is None or nb.height == 0) and (nbp is None or nbp.height == 0):
        raise ValueError(
            "No node has an energy balance: every node resolved to "
            "node_type = 'commodity' (or 'none'), so there is no "
            "per-time-step balance ('balance'/'storage') and no period "
            "balance ('balance_within_period') to solve.  Check the "
            "node_type of your nodes and the alternative ranking of the "
            "active scenario — a high-ranked alternative (e.g. 'Base') "
            "that sets node_type = 'commodity' will override the "
            "'balance'/'storage' assignments of lower-ranked "
            "alternatives."
        )


def load_flextool(source: "Path | str | FlexInputSource",
                   *,
                   db_reader: "object | None" = None,
                   handoff: "object | None" = None,
                   provider: "object | None" = None,
                   _dump_hooks: "object | None" = None) -> FlexData:
    """Load a :class:`FlexData` from either a workdir on disk or a
    :class:`flextool._input_source.FlexInputSource`.

    A ``Path`` is wrapped as a :class:`flextool._input_source.CsvSource`; it names the sub-solve's work folder
    (the path keys of the provider frames), not files to read.  The DB-direct parameters come from ``db_reader``
    (the per-(entity_class, parameter_name) :class:`flextool._input_source.InputSource`, a
    :class:`SpineDbReader` in a run); the per-solve frames come from ``provider``.

    ``provider`` (the sub-solve's :class:`FlexDataProvider`, which the cascade passes from ``_drive_cascade``) and
    ``db_reader`` (a :class:`SpineDbReader` for the run's scenario) are required; without either the loader raises
    :class:`FlexToolConfigError`.  Nothing is discovered from the work folder: no DB reader, scenario or axis-Enum
    vocabulary is built from a ``tests.sqlite`` (or any other file) found there.  Axis Enums are active only when
    ``provider.axis_enums`` is set.

    Δ.11 — ``handoff`` (in-memory :class:`SolveHandoff`) overlay
    ------------------------------------------------------------

    When ``handoff`` is supplied, the loader populates the five
    handoff-derived FlexData fields directly during the build (replacing
    the previous post-load :func:`apply_handoff` call):

      * ``p_entity_previously_invested_capacity`` / ``p_entity_invested`` ← the frames the solve emitted from
        ``realized_cohort`` (``_chain_params.emit_existing_capacity``).
      * ``p_entity_divested`` ← ``divest_cumulative``.
      * ``p_roll_continue_state`` ← ``roll_end_state``.
      * ``p_fix_storage_quantity`` ← ``fix_storage_quantity``.

    Construct-with-handoff replaces the old overlay-after-load pattern
    so the in-memory carriers flow into the build as inputs, no separate
    "apply" step.  Snapshot CSV state for these fields is overwritten
    by the in-memory handoff (the in-memory handoff is the source of
    truth, even when the workdir's ``solve_data/*.csv`` already carries
    a value).

    Private: ``_dump_hooks`` is passed only by ``flextool.engine._csv_reload.load_from_dump`` (the unsupported
    CSV reload), and it is the only way to load without a reader.  It carries two callables: ``topology(sd,
    provider)`` builds the process topology when there is no reader, and ``field_seeds(flex_data, sd, inp,
    provider)`` runs where the DB overlay (``_apply_db_overrides``) would run on a reader-less load; it seeds the
    fields that only that overlay writes in production.
    """
    _loadflex_prof("enter")
    # Late-import the Protocol + adapters to avoid a circular import
    # against the tests' fixture-loaders (which sometimes import this
    # module before flextool.__init__ finishes).
    from flextool.engine.input._input_source import CsvSource, FlexInputSource, InputSource

    workdir_for_db: Path | None = None
    if isinstance(source, (str, Path)):
        workdir_for_db = Path(source)
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
    from flextool.engine.input._solve_state import FlexToolConfigError
    if provider is None:
        raise FlexToolConfigError(
            "load_flextool needs the sub-solve's FlexDataProvider (the cascade passes it from _drive_cascade). "
            "Loading a model from a CSV dump is unsupported; see flextool.engine._csv_reload.load_from_dump.")
    if db_reader is None and _dump_hooks is None:
        raise FlexToolConfigError(
            "load_flextool needs db_reader (a SpineDbReader for the scenario); only flextool.engine._csv_reload "
            "may load without one.")

    inp = source.input_dir
    sd  = source.solve_data_dir

    from flextool.engine.solve._orchestration import get_phase_recorder
    _entry_rec = get_phase_recorder()
    _entry_logger = logging.getLogger("flextool.engine.preprocess.flex_data")
    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_provider_seeded",
            _entry_logger,
            user_label="load_flextool: provider seeded",
        )

    # Δ.12a — build the per-solve typed state (``SolveContext``, read
    # from the provider) up front so the apply_derived_* passes consume
    # its fields.  ``None`` when no workdir is known (handoff-only callers).
    ctx = None
    ctx_workdir = workdir_for_db
    if ctx_workdir is None:
        ctx_workdir = (Path(source.work_folder)
                          if hasattr(source, "work_folder")
                          else None)
    if ctx_workdir is None and hasattr(source, "input_dir"):
        try:
            ctx_workdir = Path(source.input_dir).parent
        except Exception:  # pragma: no cover — defensive
            ctx_workdir = None
    if ctx_workdir is not None:
        from flextool.engine.preprocess._solve_context import SolveContext
        ctx = SolveContext.from_workdir(ctx_workdir, provider=provider)

    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_solve_context_ready",
            _entry_logger,
            user_label="load_flextool: SolveContext ready",
        )

    # ── Phase 4 — axis enum vocabulary activation ────────────────────
    # Enum activation follows the provider only (``provider.axis_enums``; production leaves it None).
    # See ``specs/enum_dtype_refactor_plan.md §Phase 4``.
    axis_enums: "dict[str, pl.Enum] | None" = getattr(
        provider, "axis_enums", None,
    )
    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_axis_enums_done",
            _entry_logger,
            user_label="load_flextool: axis_enums built",
        )

    # Phase 4.6 — flip the cascade-wide global on.  Every cascade module
    # that uses ``rename_to_axis`` / ``alias_to_axis`` / ``lit_axis`` /
    # ``schema_dtype`` / ``cast_dim`` picks up the Enum dtypes for the
    # duration of this load.  The ``finally`` block at the end of the
    # try below resets to ``None`` so concurrent / nested loads see a
    # clean slate.
    #
    # The cross-axis alias sweep landed in clusters 4.0-4.5b + the
    # 4.6 fix-up commit (canonical self-references + ``time``/``d_h``
    # synonyms); the cross-Enum value comparisons that the activation
    # would otherwise expose were fixed in this dispatch's Step 2.
    # Set unconditionally: a Utf8 load (``axis_enums is None``) must not run
    # under the vocabulary an earlier axis-Enum load in this process left live.
    set_global_axis_enums(axis_enums)

    _branch_tok = None
    _stoch_tok = None
    _steps_tok = None
    try:
        # Stochastic branch periods inherit the period-keyed input values of their anchor period:
        # arm the solve's (anchor, branch) pairs for the DB-direct readers before any of them runs
        # (_load_online already reads startup_cost), reset in the finally below.  Only DB reads use
        # them; a reader-less load (the ``_dump_hooks`` CSV reload, no Spine DB) has no reader to
        # arm.  Likewise the emit frames from which the readers take the values of entities
        # authoring a stochastic-time map (the per-solve pbt fold).
        if db_reader is not None:
            _branch_tok = activate_branch_anchor_pairs(solve_branch_anchor_pairs(provider))
            # Warning dedupe kept on the run's reader: one reader serves every roll of a run.
            _memo = getattr(db_reader, "run_memo", None)
            _stoch_tok = activate_stochastic_time_source(
                provider, _memo(("stochastic_time_warnings",), set) if _memo is not None else None)
            # The solve's new_stepduration steps (empty without one): the DB-direct readers combine a time
            # series to them by its TIME_AGGREGATION kind.
            _steps_key = _provider_key(Path("solve_data") / NEW_STEP_BASE_STEPS)
            # A varying min_load is approximate only on the min_load_efficiency units (their curve).  Not on
            # piecewise_linear units: their curve (piecewise_linear_efficiency) does not read min_load, which only
            # bounds the online flow there, where the step mean is exact.
            from flextool.engine.preprocess.params._direct import _names_with_method

            def _mle_units() -> "dict[tuple[str, str], frozenset[str]]":
                return {("unit", "min_load"): frozenset(_names_with_method(
                    db_reader, "unit", "conversion_method", frozenset({"min_load_efficiency"})))}

            _steps = provider.get(_steps_key) if provider.has(_steps_key) else None
            _approx = None
            if _steps is not None and _steps.height > 0:
                _approx = _memo(("solve_step_approx_entities",), _mle_units) if _memo is not None else _mle_units()
            _steps_tok = activate_solve_step_aggregation(
                _steps, _memo(("solve_step_aggregation_notes",), set) if _memo is not None else None, _approx)

        # Reuse the recorder bound at the function entry above so the
        # sub-checkpoints (axis_enums / SpineDbReader / load_flextool
        # start) share a single timeline.
        _rec = _entry_rec
        _load_logger = _entry_logger

        def _load_mem(label: str, user_label: str) -> None:
            if _rec is not None:
                _rec.checkpoint(label, _load_logger, user_label=user_label)

        _load_mem("load_flextool_start", "load_flextool start")

        # Δ.2: build the per-solve BlockLayout once from flextool's
        # solve_data/ block CSVs (still produced by flextool's
        # ``write_block_data_for_solve``).  Downstream block-aware helpers
        # consume the in-memory frames instead of re-reading the same CSVs
        # at each call site.  When the orchestrator transitions to building
        # ``BlockLayout`` natively (Δ.3+), this load_from_solve_data call
        # becomes a no-op or is replaced by passing the live layout in.
        block_layout = BlockLayout.load_from_solve_data(sd, provider=provider)

        dt, step_dur, tsw, infl, psh = _load_time(sd, provider=provider)
        nb, nbp, nb_dt, inflow, pen_up, pen_dn, penalty_off = _load_node(sd, dt, provider=provider)
        _load_mem("load_node_end", "load_flextool: time + node loaded")

        proc = _load_process_topology(inp, sd, dt,
                                       source=db_reader,
                                       provider=provider,
                                       dump_topology=(_dump_hooks.topology
                                                      if _dump_hooks is not None else None))
        _load_mem("load_process_topology_end",
                  "load_flextool: process topology loaded")

        # base_cap_pd = (p, d, base) for profile RHS — recompute here; small.
        base_cap_pd = None
        p_flow_upper_existing = None
        pd_neg_cap = None
        all_entity_unitsize_param = None
        if proc["pss"] is not None:
            cap_long = _read_capacity(sd / "p_entity_all_existing.csv", provider=provider)
            unitsize_long = _read_unitsize(sd / "p_entity_unitsize.csv", provider=provider)
            # p_all_entity_unitsize: unfiltered — covers processes, connections AND nodes.
            # Used by the scaling analyzer to compute the full entity-unitsize spread.
            if unitsize_long.height > 0:
                all_entity_unitsize_param = Param(
                    ("e",),
                    unitsize_long.pipe(rename_to_axis, {"e": "e"}).select("e", "value"),
                )
            cap_us_pd = (cap_long.pipe(rename_to_axis, {"e":"p","value":"cap"})
                .filter(pl.col("p").is_in(proc["pss"]["p"].unique()))
                .join(unitsize_long.pipe(rename_to_axis, {"e":"p","value":"us"}), on="p", how="inner"))
            base_cap_pd = (cap_us_pd
                .with_columns(base=pl.col("cap")/pl.col("us"))
                .select("p","d","base"))
            # pd_neg_cap = (p, d) where both existing and unitsize are negative.
            # In the .mod, maxFlow is ``v_flow * unitsize ≤ existing × ...``.
            # When both are negative (e.g. anti_energy_plant: us=-50, existing=-50)
            # dividing by unitsize FLIPS the inequality direction, yielding
            # ``v_flow ≥ capacity/unitsize`` (a forced *minimum* output).
            # We therefore route these (p, d) rows out of the standard ``≤``
            # maxFlow and into a sign-flipped ``≥`` companion constraint.
            neg_pd = cap_us_pd.filter(
                (pl.col("cap") < 0.0) & (pl.col("us") < 0.0)
            ).select("p", "d")
            if neg_pd.height > 0:
                pd_neg_cap = neg_pd
            # p_flow_upper_existing = existing unit count (capacity/unitsize) per (p, source, sink, d).
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

        # Start-up input nodes and their CO2 groups: before the CO2 loaders, whose arc gates they widen.
        startup = _load_startup_inputs(inp, sd, dt, proc["pss"], nb, nbp, source=db_reader, provider=provider)
        startup.update(_startup_co2_sets(inp, sd, startup["process_startup_node"], provider=provider))
        flow_co2_p, flow_co2_p_noEff, co2c, co2pr, flow_co2_p_rem = _load_co2_price(
            inp, sd, proc["pss_eff"], proc.get("pss_noEff"),
            pss=proc.get("pss"), provider=provider,
            startup_emitters=startup["startup_from_co2_priced"] is not None)
        (g_co2_max, flow_co2_cap, flow_co2_cap_noEff, co2_max_p, g_d_capped,
         flow_co2_cap_rem) = _load_co2_cap(
            inp, sd, proc["pss_eff"], dt, pss_noEff=proc.get("pss_noEff"),
            pss=proc.get("pss"), provider=provider,
            startup_emitters=startup["startup_from_co2_capped"] is not None)
        (g_co2_max_total, flow_co2_cap_total, flow_co2_cap_total_noEff,
         co2_max_total_p, flow_co2_cap_total_rem) = _load_co2_cap_total(
            inp, sd, proc["pss_eff"], pss_noEff=proc.get("pss_noEff"),
            pss=proc.get("pss"), provider=provider,
            startup_emitters=startup["startup_from_co2_capped_total"] is not None)
        # The total cap counts only the groups with a cap value (the loader's gate).
        if startup["startup_from_co2_capped_total"] is not None:
            kept = (None if g_co2_max_total is None else startup["startup_from_co2_capped_total"].join(
                g_co2_max_total.select(pl.col("g").cast(pl.Utf8)).unique(), on="g", how="semi"))
            startup["startup_from_co2_capped_total"] = kept if kept is not None and kept.height > 0 else None
        (co2_f_d_k, co2_cum_realized, co2_cum_realized_rep, co2_horizon_fraction,
         co2_hours_before_solve) = load_co2_rolling_carry(sd, provider=provider)

        (indir_set, indir_in, indir_out, indir_dt,
         p_source_flow_coef, p_sink_flow_coef) = _load_indirect(
             sd, proc["pss"], dt, inp, provider=provider)
        (fc_idx, fc_coef, c_const, cdt_eq, cdt_le, cdt_ge,
         n_inv_coef, p_inv_coef,
         n_state_coef, n_prebuilt_coef, p_prebuilt_coef,
         _) = _load_user_constraints(inp, proc["pss"], dt, provider=provider)

        p_up, p_lo, p_fx, prof_v, exist_cnt, avail = _load_profiles(
            inp, sd, proc["pss"], proc["unitsize"], base_cap_pd,
            provider=provider)
        # existing_count is also needed by the online/UC feature even when
        # no profile features are active; fall back to base_cap_pd directly.
        if exist_cnt is None and base_cap_pd is not None:
            exist_cnt = Param(("p", "d"), base_cap_pd.rename({"base": "value"}))
        # availability: default to 1.0 from preprocessing — also used by UC
        # capacity bounds; if loader didn't populate (no profile data), try
        # to read pdtProcess_availability.csv standalone.
        if avail is None and proc["pss"] is not None:
            avail_long = _slice_param(sd / "pdtProcess.csv", "process", "availability",
                                       rename_entity_to="p",
                                       provider=provider)
            if avail_long is not None:
                avail = Param(("p","d","t"), avail_long)

        # dtttdt is needed by both storage and online features — always load
        # it when present (preprocessing always emits it for non-trivial
        # solves).  p_process_existing_count (= capacity/unitsize per (p, d))
        # is needed by online + profile features — always load when processes
        # exist.
        dtttdt = _read_step_previous(sd / "step_previous.csv", provider=provider)

        online = _load_online(inp, sd, dt, proc["pss"], source=db_reader,
                              provider=provider)
        online["process_exact"] = _read_process_exact(inp, provider=provider)
        ramp = _load_ramp(inp, sd, proc["pss"], dt=dt, provider=provider)
        # The online carry lands on the ramp's seam roots: one adjacency for ramp, min up / down and online.
        if online.get("p_roll_continue_online") is not None and ramp.get("dt_ramp_carry") is not None:
            online["dt_online_carry"] = ramp["dt_ramp_carry"].select("d", "t").unique(maintain_order=True)
        # The online carry and the ramp carry / ramp link must agree.  Checked by the caller on the solve's
        # reduced data (params._ramp_links.check_online_ramp_carry, after preprocess._reduce_detail), since a
        # reduce_unit_detail_method that leaves out online variables also removes the online carry.
        invest = _load_invest(sd, dt, inp, proc["pss"], db_reader=db_reader,
                              provider=provider)
        varcost = _load_varcost(sd, proc["pss"], provider=provider)
        _load_mem("load_varcost_end", "load_flextool: varcost loaded")
        fixed_cost = _load_fixed_cost(sd, provider=provider)
        capacity_for_scaling = _load_node_capacity_for_scaling(sd, nb, nbp,
                                                                provider=provider)

        # ─── Storage (nodeState + binding methods + dtttdt + node-balance source-side flows)
        storage = _load_storage(inp, sd, dt, nb, nbp,
                                 proc["pss_eff"], proc["pss_noEff"],
                                 base_cap_pd, proc["unitsize"],
                                 block_layout=block_layout,
                                 provider=provider)
        # _load_storage emits its own dtttdt; if storage is inactive it'll be
        # None there but we want the top-level read.
        if storage["dtttdt"] is None:
            storage["dtttdt"] = dtttdt

        # ─── Group-level slack (capacity_margin / inertia / non_sync) ────────
        group_slack = _group_slack.load_data(
            inp=inp, sd=sd, dt=dt,
            nb=nb,
            pss_eff=proc["pss_eff"],
            pss_noEff=proc["pss_noEff"],
            p_unitsize=proc["unitsize"],
            provider=provider,
        )

        # ─── Reserves (timeseries / dynamic / n_1 / per-process upper) ────────
        reserve_data = _reserve.load_data(inp=inp, sd=sd, dt=dt,
                                            provider=provider)
        # ``group_node`` comes from group_slack only (None when no group has
        # a node); the reserve groups without nodes are left out at the end
        # of the load (``_reserve.drop_groups_without_nodes``).

        # ─── Cumulative / group-invest / min-invest data ─────────────────────
        ci_data = _load_cumulative_invest(inp=inp, sd=sd, dt=dt, provider=provider)

        # ─── Delayed processes / DR data ─────────────────────────────────────
        delay_data = _delay.load_data(inp_dir=inp, sd_dir=sd,
                                         provider=provider)

        # ─── DC power flow data ──────────────────────────────────────────────
        # Step 2.5-F Phase B: Provider holds the four DC PF frames
        # under input/<key>; disk arm is reserved for off-cascade
        # fixture loaders without a Provider.
        dc_pf_data = _dc_power_flow.load_data(inp_dir=inp, provider=provider)

        # ─── Commodity price ladder data ─────────────────────────────────────
        # Step 2.5-F Phases D + E: Provider carries the two ladder
        # ``input/commodity_ladder_*`` frames; disk arm reserved for
        # off-cascade fixture loaders.
        ladder_data = _commodity_ladder.load_data(
            inp_dir=inp, sd_dir=sd, provider=provider,
        )

        # ─── Multi-branch stochastic data (A6) ───────────────────────────────
        stoch_data = _load_stochastics(inp=inp, sd=sd, dt=dt, provider=provider)
        _load_mem("load_stochastics_end",
                  "load_flextool: stochastics + remaining CSV loaders done")

        # Model-wide small-coefficient cutoff (model.small_number_threshold).
        # Exported as a one-row-per-model CSV by the input_derivation spec
        # ``p_small_number_threshold``.  Take the max over models (mirrors
        # ``p_max_flow_for_unconstrained_variables``); fall back to the
        # schema default 0.0001 when the file is absent / empty.
        small_number_threshold = 0.0001
        if _provider_has(
            provider, "input/p_small_number_threshold",
            inp / "p_small_number_threshold.csv",
        ):
            _snt_df = _provider_read(
                provider, "input/p_small_number_threshold",
                inp / "p_small_number_threshold.csv",
            )
            _snt_vals = [
                float(r[1]) for r in _snt_df.iter_rows()
                if len(r) >= 2 and r[0] not in (None, "")
                and str(r[1]).strip() not in ("", "None")
            ]
            if _snt_vals:
                small_number_threshold = max(_snt_vals)

        _loadflex_prof("before_flexdata_ctor")
        flex_data = FlexData(
            small_number_threshold = small_number_threshold,
            dt = dt,
            p_step_duration = step_dur,
            p_timestep_weight = tsw,
            p_inflation_op = infl,
            p_period_share = psh,

            nodeBalance = nb,
            nodeBalancePeriod = nbp,
            nodeBalance_penalty_off = penalty_off,
            nodeBalance_dt = nb_dt,
            p_inflow = inflow,
            p_inflow_unmargined = _load_inflow_unmargined(sd, provider=provider),
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
            process_source_canonical  = proc.get("pss_source_canonical"),
            process_sink_canonical    = proc.get("pss_sink_canonical"),
            p_unitsize                = proc["unitsize"],
            p_all_entity_unitsize     = all_entity_unitsize_param,
            p_flow_upper              = proc["flow_upper"],
            p_flow_upper_existing     = p_flow_upper_existing,
            p_slope                   = proc["slope"],
            p_commodity_price         = proc["commodity_price"],
            pd_neg_cap                = pd_neg_cap,

            flow_from_co2_priced = flow_co2_p,
            flow_from_co2_priced_noEff = flow_co2_p_noEff,
            flow_to_co2_priced = flow_co2_p_rem,
            p_co2_content = co2c,
            p_co2_price = co2pr,

            group_co2_max_period = g_co2_max,
            flow_from_co2_capped = flow_co2_cap,
            flow_from_co2_capped_noEff = flow_co2_cap_noEff,
            flow_to_co2_capped = flow_co2_cap_rem,
            p_co2_max_period = co2_max_p,
            group_d_co2_capped = g_d_capped,

            group_co2_max_total = g_co2_max_total,
            flow_from_co2_capped_total = flow_co2_cap_total,
            flow_from_co2_capped_total_noEff = flow_co2_cap_total_noEff,
            flow_to_co2_capped_total = flow_co2_cap_total_rem,
            p_co2_max_total = co2_max_total_p,
            p_co2_f_d_k = co2_f_d_k,
            p_co2_cum_realized_tonnes = co2_cum_realized,
            p_co2_cum_realized_represented = co2_cum_realized_rep,
            co2_horizon_fraction_reached = co2_horizon_fraction,
            p_co2_hours_before_solve = co2_hours_before_solve,
            flowgroup_realized = _fg.load_flowgroup_rolling_carry(provider),

            process_indirect = indir_set,
            process_input_flows = indir_in,
            process_output_flows = indir_out,
            process_indirect_dt = indir_dt,
            p_process_source_conversion_flow_coeff = p_source_flow_coef,
            p_process_sink_conversion_flow_coeff = p_sink_flow_coef,

            flow_constraint_idx = fc_idx,
            p_flow_constraint_coef = fc_coef,
            p_constraint_constant = c_const,
            cdt_eq = cdt_eq,
            cdt_le = cdt_le,
            cdt_ge = cdt_ge,
            p_node_constraint_invested_capacity_coeff = n_inv_coef,
            p_process_constraint_invested_capacity_coeff = p_inv_coef,
            p_node_constraint_state_coeff = n_state_coef,
            p_node_constraint_prebuilt_capacity_coeff = n_prebuilt_coef,
            p_process_constraint_prebuilt_capacity_coeff = p_prebuilt_coef,

            process_profile_upper = p_up,
            process_profile_lower = p_lo,
            process_profile_fixed = p_fx,
            p_profile_value = prof_v,
            p_process_existing_count = exist_cnt,
            p_process_availability = avail,

            **startup,
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
            period_invest=_load_period_invest(sd, provider=provider),
            # Every roll, with or without storage nodes (the capacity margin, the recourse guard and the realized
            # steps of a roll read it).
            solve_is_rolling=_load_roll_period_whole(sd, provider=provider) is not None,
            # The per-solve BlockLayout loaded above (node -> group, group steps); the coarse-step frames are
            # derived from it at the end of the load (``_apply_coarse_frames``).
            block_layout = block_layout,
        )
        _loadflex_prof("after_flexdata_ctor")

        # Gap F final — handoff-path auxiliaries: surface the three CSVs
        # that ``build_handoff_from_solution`` would otherwise re-read.
        # Each is lenient: file missing or empty → field stays ``None``
        # and the handoff extractor's disk fallback kicks in.
        #
        # The output row set: ``dt_realize_dispatch_set.csv`` (built by
        # ``_per_solve.emit_per_solve_sets``) is the de-duplicated
        # ``realized_dispatch`` — realized steps only, never forecast-branch
        # or look-ahead steps (``model.output_horizon`` writes those to its
        # separate raw dump).  Every dt-indexed output and the parquet
        # extractor's canonical row order follow it.
        flex_data.realized_dispatch = _load_handoff_aux_pair(
            sd / "dt_realize_dispatch_set.csv", ("period", "time"),
            provider=provider)
        if flex_data.realized_dispatch is not None:
            flex_data.realized_dispatch = flex_data.realized_dispatch.pipe(
                rename_to_axis, {"time": "step"})
        if flex_data.realized_dispatch is None:
            flex_data.realized_dispatch = _load_handoff_aux_pair(
                sd / "realized_dispatch.csv", ("period", "step"),
                provider=provider)
        flex_data.fix_storage_timesteps = _load_handoff_aux_pair(
            sd / "fix_storage_timesteps.csv", ("period", "step"), provider=provider)
        rip_path = sd / "realized_invest_periods_of_current_solve.csv"
        if _provider_has(provider, "solve_data/realized_invest_periods_of_current_solve", rip_path):
            try:
                rip = _provider_read(provider, "solve_data/realized_invest_periods_of_current_solve", rip_path)
                flex_data.realized_invest_periods = (
                    [str(v) for v in rip["period"].to_list() if v is not None] if "period" in rip.columns else [])
            except pl.exceptions.NoDataError:
                flex_data.realized_invest_periods = []
        flex_data.na_online_fix_in_uc = _na_online_fix_in_uc(sd, provider=provider)
        # ``node__storage_nested_fix_method`` lives in solve_data/ for
        # cascade solves; fall back to input/ if not yet copied.  Explicit
        # ``is None`` chain (DataFrame is non-truthy in polars).
        nsfm = _load_handoff_aux_pair(
            sd / "node__storage_nested_fix_method.csv", ("node", "method"),
            provider=provider)
        if nsfm is None:
            nsfm = _load_handoff_aux_pair(
                inp / "node__storage_nested_fix_method.csv", ("node", "method"),
                provider=provider)
        flex_data.node__storage_nested_fix_method = nsfm

        # Δ.3/Δ.4 — DB-direct construction.  Replaces the previous 3-pass
        # override layering (CSV → first_wave_overrides → projection_overrides
        # → derived_overrides_a..g).  Δ.3 collapsed the dict-overlay
        # round-trip into linear ``apply_*`` mutations; Δ.4 deleted the nine
        # deprecated wrapper aliases.  Each FlexData field that has a
        # DB-direct helper is now built by exactly one helper that mutates
        # ``flex_data`` directly — no dict-overlay round-trip, no "override"
        # semantics.  See progress.md (Δ.3 / Δ.4 close stanzas).
        #
        # NOTE: the CSV path above still populates every field as the
        # initial seed.  In Δ.5+, when every FlexData field has a DB-direct
        # helper, the CSV path will retire and these apply_* calls become
        # the primary loader.
        # Δ.12a — ctx (the provider-backed ``SolveContext``) was built at
        # the top of ``load_flextool`` and is threaded into apply_derived_*.
        # We deliberately DEFER the FlexData → Enum sweep until AFTER
        # ``_apply_db_overrides`` (the derived-cascade).  The cascade
        # builds many scratch frames with hard-coded ``pl.Utf8`` dim
        # column schemas in ``_derived.py`` /
        # ``_block.py`` / etc., and joining Enum-cast FlexData
        # fields against String-dtype scratch frames raises
        # ``SchemaError``.  Casting after the cascade keeps the entire
        # CSV-read + cascade pipeline in String land and converts to
        # Enum once at the end — Var construction in ``model/core/`` and
        # the model-build cross-joins then operate on Enum-typed
        # frames.
        if db_reader is not None:
            _loadflex_prof("before_apply_db_overrides")
            _apply_db_overrides(flex_data, db_reader, source, ctx=ctx,
                                 provider=provider)
            _loadflex_prof("after_apply_db_overrides")

            # Load-boundary invariant for derived Params that ``model/core/``
            # dereferences UNCONDITIONALLY (no ``is not None`` guard) once
            # their owning structural set is present.  ``_apply_db_overrides``
            # produces these in ``apply_derived_b`` / ``apply_derived_e``, but
            # the synthetic-solve early-return skips passes 3-10 entirely (the
            # same gap class as RESERVE-1 / SECTION-1 documented in that
            # function).  On a synthetic / rolling sub-solve they are then left
            # ``None`` while the ctor-built index sets (``flow_constraint_idx``,
            # ``process_source_sink``, ``nodeState``) are populated — and the
            # build dies with ``TypeError: ... 'Expr' and 'NoneType'``.
            #
            # ``p_flow_constraint_coef`` is the one that actually bites: it has
            # NO ctor seed (the ctor builds only the ``flow_constraint_idx``
            # index — see ``_load_user_constraints``), so the user-defined
            # flow-constraint block (``model/core/`` §user_defined_flow_constraints)
            # does ``Where(v_flow * p_unitsize, flow_constraint_idx) *
            # p_flow_constraint_coef`` with a ``None`` coefficient.  ``p_unitsize``
            # / ``p_state_unitsize`` DO have ctor seeds so they are rarely
            # ``None`` here, but we guard them too for symmetry (and because the
            # producers are now complete-by-construction).
            #
            # Enforce the invariant for EVERY override path by rebuilding from
            # the live ``db_reader`` source whenever a Param is missing but its
            # set exists.  No-op on the non-synthetic path (the passes already
            # set them), so it is parity-clean — not a guard that masks a
            # divergence by silently dropping the term.
            from flextool.engine.preprocess.params._arithmetic import (
                p_state_unitsize_from_source,
                p_unitsize_from_source,
            )
            from flextool.engine.preprocess.params._derived import (
                p_flow_constraint_coef_from_source,
            )
            _pss_inv = getattr(flex_data, "process_source_sink", None)
            _pss_ok = _pss_inv is not None and _pss_inv.height > 0
            if flex_data.p_unitsize is None and _pss_ok:
                flex_data.p_unitsize = p_unitsize_from_source(
                    db_reader, _pss_inv)
            if flex_data.p_state_unitsize is None:
                _ns_inv = getattr(flex_data, "nodeState", None)
                if _ns_inv is not None and _ns_inv.height > 0:
                    flex_data.p_state_unitsize = p_state_unitsize_from_source(
                        db_reader, _ns_inv)
            _fci = getattr(flex_data, "flow_constraint_idx", None)
            if (flex_data.p_flow_constraint_coef is None
                    and _fci is not None and _fci.height > 0
                    and _pss_ok):
                flex_data.p_flow_constraint_coef = (
                    p_flow_constraint_coef_from_source(db_reader, _pss_inv))
            # ``p_years_represented_d`` (step 2b of ``apply_derived_a``) is
            # skipped by the same early return, so a rolling or synthetic
            # sub-solve would report every period as 1 year.  Build it over
            # the sub-solve's dt periods; the provider's canonical
            # ``p_years_represented.csv`` carries the parent solve's years.
            if flex_data.p_years_represented_d is None:
                try:  # the workdir as resolved in ``_apply_db_overrides``
                    _wd_yr = (source.workdir if hasattr(source, "workdir")
                              else source.input_dir.parent)
                except Exception:  # pragma: no cover — defensive
                    _wd_yr = None
                flex_data.p_years_represented_d = _years_represented_boundary(
                    getattr(flex_data, "dt", None), db_reader, provider, _wd_yr)
            # piecewise_linear units: the load-boundary step (one authority on every path).
            from flextool.engine.preprocess.params import _derived as _drv_k1
            from flextool.engine.preprocess.params._piecewise import apply_piecewise_curves
            apply_piecewise_curves(flex_data, db_reader, provider)
            # One-variable connections: an efficiency that is not a unit conversion is ignored (W-0591).
            _drv_k1.apply_one_variable_efficiency(flex_data, db_reader, provider)
        elif _dump_hooks is not None:
            # Reader-less CSV reload: the reload's own seeds of the fields the overlay above writes.
            _dump_hooks.field_seeds(flex_data, sd, inp, provider)

        # Δ.11 — overlay in-memory handoff carriers onto the FlexData built so far (the handoff is an input to
        # the build, not a separate overlay step), then the chain-carried constants of :func:`apply_existing_chain`
        # (db_reader is required for that path).
        if handoff is not None:
            sd_dir = workdir_for_db / "solve_data" if workdir_for_db is not None else None
            flex_data = _overlay_handoff(flex_data, handoff, sd_dir, ctx=ctx,
                                            provider=provider)
            _loadflex_prof("after_overlay_handoff")
            if db_reader is not None and workdir_for_db is not None:
                from flextool.engine.preprocess.params import _existing as _ex
                _ex.apply_existing_chain(flex_data, db_reader, workdir_for_db,
                                              handoff=handoff, ctx=ctx,
                                              provider=provider)

        # ``p_window_fraction``: the share of each period's complete-solve hours that this sub-solve's
        # dt covers.  ``apply_derived_a`` sets it to 1.0 next to ``p_period_share``; the sub-solves that
        # skip it (rolls, synthetic nested sub-solves) and the reader-less loads (a dumped workdir) get
        # it here from their own step durations and the (parent) period share.
        if (flex_data.p_window_fraction is None and getattr(flex_data, "dt", None) is not None
                and flex_data.p_step_duration is not None and flex_data.p_period_share is not None):
            from flextool.engine.preprocess.params._derived import p_window_fraction_from_shares
            flex_data.p_window_fraction = p_window_fraction_from_shares(
                flex_data.dt, flex_data.p_step_duration, flex_data.p_period_share)

        # constraint_period_method period_total: off in the rolls of a rolling solve that does not realize whole
        # periods in one roll (the run's pre-solve check warned).
        from flextool.engine.preprocess import _provider_keys as _pk
        if provider is not None and _provider_has(provider, _pk.SOLVE_DATA_USER_CONSTRAINT_PERIOD_TOTAL_OFF,
                                                  sd / "user_constraint_period_total_off.csv"):
            pto = _provider_read(provider, _pk.SOLVE_DATA_USER_CONSTRAINT_PERIOD_TOTAL_OFF,
                                 sd / "user_constraint_period_total_off.csv")
            if pto is not None and pto.height > 0 and "value" in pto.columns:
                flex_data.user_constraint_period_total_off = bool(pto["value"].cast(pl.Boolean)[0])

        # Coarse new_stepduration groups: the coarse steps of their nodes, from this solve's own steps (every
        # solve, roll, nested and stochastic sub-solve).
        _apply_coarse_frames(flex_data, sd, provider=provider)

        # Annual commodity ladder: the forecast branch periods of a stochastic solve get their anchor
        # period's tier curve.  Runs after every producer of the ladder Params (load_data and the
        # direct-parameter pass).
        _commodity_ladder.inherit_branch_periods(flex_data, provider)

        # Reserves of groups without nodes in the solve are left out (with a warning).  After every producer
        # of the reserve sets and parameters, and of group_node.
        _reserve.drop_groups_without_nodes(flex_data)

        # The margin of a relative_of_peak_demand capacity-margin group: capacity_margin_relative times its peak demand, in
        # pdGroup_capacity_margin with the absolute groups' MW.  After the direct-parameter pass (the share) and
        # load_data (the peak).
        if db_reader is not None:
            _group_slack.resolve_capacity_margin(flex_data)

        # The minFlow_minload floor's units: min_load > 0 on some (d, t) of this solve, read from the
        # solve-resolved p_min_load (every DB path, synthetic sub-solves included, has produced it by now).
        # A reader-less load keeps the dumped set.
        if db_reader is not None:
            from flextool.engine.preprocess.params._derived import process_minload_from_p_min_load
            flex_data.process_minload = process_minload_from_p_min_load(flex_data.p_min_load)

        # Units / connections whose every balance node is coarse: their coarse steps.  After every set the
        # rule reads (process_minload above included).
        _apply_coarse_process_frames(flex_data)

        _warn_fractional_integer_unit_counts(flex_data, db_reader)

        # use_reference_value without fix_start: the start comes from the reference value (owner decision, Phase 5).
        _reference_start_as_fix_start(flex_data)

        # Phase 4.6 — end-of-load FlexData → Enum sweep.  With the global
        # flipped on above, the cascade emits Enum-typed dim columns
        # throughout, but ad-hoc CSV-fallback paths inside the cascade
        # can still produce Utf8 dim columns.  Sweep the FlexData
        # container once before return so every Param / DataFrame /
        # LazyFrame field has its dim columns cast against the canonical
        # axis enums.  Idempotent: columns already in the correct Enum
        # dtype are skipped by ``cast_frame_axes``.
        if axis_enums is not None:
            cast_flexdata_axes(flex_data, axis_enums)
        _loadflex_prof("after_cast")

        # Stash the resolved axis_enums on the returned FlexData so
        # downstream consumers (``build_flextool`` and the dumpers that
        # already read ``getattr(data, "_axis_enums", None)``) can rebind
        # the global ContextVar — substrate helpers ``cast_dim`` /
        # ``schema_dtype`` / ``lit_axis`` / ``alias_to_axis`` reach for
        # the live ContextVar when called with ``enums=None``, and
        # without this stash the ContextVar is reset before
        # ``build_flextool`` runs (see _delay.delayed_input_expr
        # which calls ``cast_dim(..., None, ...)``).
        flex_data._axis_enums = axis_enums

        _validate_has_balance(flex_data)

        result = _assign_param_names(flex_data)
        load_ok = True  # noqa: F841 — consumed by finally clause below
        _loadflex_prof("return")
        return result
    finally:
        if _steps_tok is not None:
            deactivate_solve_step_aggregation(_steps_tok)
        if _stoch_tok is not None:
            deactivate_stochastic_time_source(_stoch_tok)
        if _branch_tok is not None:
            deactivate_branch_anchor_pairs(_branch_tok)
        # On the success path, leave the global ContextVar set to the
        # cascade's axis_enums — ``build_flextool`` runs AFTER
        # ``load_flextool`` returns and still needs the live vocabulary
        # for ``cast_dim(..., None, "e")`` call sites in _delay /
        # downstream model builders.  On cascade error reset so a
        # half-built state doesn't leak into the next cascade.  When
        # activation is off (``axis_enums is None``) the reset is a no-op
        # because ``set_global_axis_enums(None)`` matches the default.
        if not locals().get("load_ok", False):
            set_global_axis_enums(None)
        else:
            set_global_axis_enums(axis_enums)


def reference_start_shares(flex_data: "FlexData") -> dict:
    """``{node: share}``: the storage_state_reference_value at the first step each use_reference_value storage node
    starts from in this solve (not carried from an earlier solve; the model's first step in the first solve).  With
    several first steps (a stochastic fan) the first in sorted order.  Empty when there are none.

    The reference value of a step is the level at its end, so the start level (before the first step) would be the
    reference of the step before; at the model start no such step exists (and a solve's data holds only its own
    steps), so the first step's value stands for it (review L2, a decision: the difference is one step's change of the
    reference series, negligible for a smooth one)."""
    from flextool.engine.input._param_shapes import promote_param_to_dt
    ref, rv, first = (flex_data.storage_use_reference_value, flex_data.p_storage_state_reference_value,
                      flex_data.nodeState_first_dt)
    if ref is None or ref.height == 0 or rv is None or first is None or first.height == 0 or flex_data.dt is None:
        return {}
    names = set(str(n) for n in ref["n"].to_list())
    carry = flex_data.p_roll_continue_state
    if carry is not None and carry.frame.height > 0:
        names -= set(str(n) for n in carry.frame["n"].to_list())
    if not names:
        return {}
    steps = first.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")).filter(pl.col("n").is_in(sorted(names)))
    values = promote_param_to_dt(rv, flex_data.dt)
    values = values.collect() if isinstance(values, pl.LazyFrame) else values
    values = values.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")).with_columns(
        value=values["value"].cast(pl.Float64))
    hit = steps.join(values, on=["n", "d", "t"], how="inner").sort("n", "d", "t").unique("n", keep="first")
    return {str(n): float(v) for n, v in hit.select("n", "value").iter_rows() if v is not None}


def _reference_start_as_fix_start(flex_data: "FlexData") -> None:
    """A use_reference_value storage node without fix_start starts from the reference value at its first step (owner
    decision, Phase 5; before, its start was free and only the matching end bounded it): it joins
    ``storage_fix_start`` with that share as ``p_state_start``, so every fix_start rule applies to it (the start pin,
    the matching end of storage the solve can invest in).  A node with its own fix_start keeps it
    (``model.core._prepare._check_storage_accounting`` warns where the two differ)."""
    shares = reference_start_shares(flex_data)
    if not shares:
        return
    fix_start = flex_data.storage_fix_start
    fixed = set() if fix_start is None else set(str(n) for n in fix_start["n"].to_list())
    new = {n: v for n, v in shares.items() if n not in fixed}
    if not new:
        return
    names = sorted(new)
    dtype = flex_data.nodeState.schema["n"] if flex_data.nodeState is not None else pl.Utf8
    added = pl.DataFrame({"n": names}).with_columns(pl.col("n").cast(dtype, strict=False))
    if fix_start is None or fix_start.height == 0:
        flex_data.storage_fix_start = added
    else:
        flex_data.storage_fix_start = pl.concat([fix_start, added.cast(fix_start.schema)], how="vertical")
    rows = pl.DataFrame({"n": names, "value": [float(new[n]) for n in names]})
    old = flex_data.p_state_start
    if old is None or old.frame.height == 0:
        frame = rows.with_columns(pl.col("n").cast(dtype, strict=False))
    else:
        keep = old.frame.filter(~pl.col("n").cast(pl.Utf8).is_in(names))
        frame = pl.concat([keep, rows.with_columns(pl.col("n").cast(keep.schema["n"], strict=False),
                                                   pl.col("value").cast(keep.schema["value"]))],
                          how="vertical")
    flex_data.p_state_start = Param(("n",), frame)
    from flextool.engine import _behaviour_changes
    _behaviour_changes.notice("storage_reference_start", "nodes", names, solve="")


# Units already reported by :func:`_warn_fractional_integer_unit_counts`, per run: a rolling run loads one FlexData
# per roll through the run's one ``db_reader``, and each unit is reported once per run.
_FRACTIONAL_COUNT_WARNED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _warn_fractional_integer_unit_counts(flex_data: "FlexData", db_reader: "InputSource | None") -> None:
    """Warn (W-0526, one line) about the units / connections with integer online variables whose ``p_process_existing_count`` (existing
    capacity / unit size, the number of existing units) is not a whole number in some period of this solve.

    ``v_online_integer`` counts whole units up to that count (``maxOnline_integer`` / ``maxShutdown_integer``), so
    a count such as 0.5 cannot all be online: below 1 the unit can never be online, which can make a floor such as
    ``min_instant_flow`` infeasible.  Without ``virtual_unitsize`` the unit size is the largest ``existing`` over
    the periods, so a period-Map ``existing`` that drops (17 → 8.5 MW) gives a fraction.  Read from the solve's own
    Param, so single, rolling, nested and stochastic sub-solves are all checked, and a count derived from a wrong
    unit size shows too.  Entities with a positive ``virtual_unitsize`` are left to the input check
    (``input_derivation._validators.warn_existing_not_whole_units``, where ``existing`` is the count itself).
    Results are unchanged.
    """
    ec = flex_data.p_process_existing_count
    integer = flex_data.process_online_integer
    if ec is None or integer is None or integer.height == 0 or ec.frame.height == 0:
        return
    names = {str(x) for x in integer.get_column(integer.columns[0]).to_list()}
    warned: set[str] = set()
    if db_reader is not None:
        from flextool.engine.preprocess.params._derived import _try_param_explicit
        try:
            warned = _FRACTIONAL_COUNT_WARNED.setdefault(db_reader, set())
        except TypeError:  # pragma: no cover — a reader that cannot be weak-referenced: no per-run memory
            pass
        for cls in ("unit", "connection"):
            vu = _try_param_explicit(db_reader, cls, "virtual_unitsize")
            if vu is not None and vu.height > 0:
                names -= {str(n) for n, v in zip(vu["name"].to_list(), vu["value"].to_list())
                          if v is not None and float(v) > 0.0}
    names -= warned
    if not names:
        return
    frame = (ec.frame.with_columns(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
             .filter(pl.col("p").is_in(sorted(names))
                     & ((pl.col("value") - pl.col("value").round(0)).abs()
                        > 1e-6 * pl.max_horizontal(pl.lit(1.0), pl.col("value").abs()))))
    if frame.height == 0:
        return
    units: set[str] = set()
    if flex_data.process_unit is not None and flex_data.process_unit.height > 0:
        units = {str(x) for x in flex_data.process_unit.get_column(flex_data.process_unit.columns[0]).to_list()}
    unitsize: dict[str, float] = {}
    if flex_data.p_unitsize is not None:
        us = flex_data.p_unitsize.frame
        unitsize = dict(zip(us["p"].cast(pl.Utf8).to_list(), us["value"].to_list()))
    found: dict[str, str] = {}
    for (p,), rows in frame.sort(["p", "d"]).group_by(["p"], maintain_order=True):
        counts = ", ".join(f"{v:g} in {d}" for d, v in zip(rows["d"].to_list(), rows["value"].to_list()))
        size = unitsize.get(p)
        size_txt = f" of {size:g} MW" if size is not None else ""
        cls = "unit" if p in units else "connection"
        warned.add(p)
        found[p] = f"{cls}: existing capacity makes {counts} units{size_txt}"
    from flextool import run_warnings
    run_warnings.warn_grouped("W-0526", list(found), details=found)


def _na_online_fix_in_uc(sd: Path, *, provider: "object | None" = None) -> bool:
    """True when ``solve_data/na_online_fix_in_uc`` holds a ``yes`` row (``emit_na_online_fix_in_uc``: solve (1) of
    separate_dispatch); False when it is absent or header-only."""
    path = sd / "na_online_fix_in_uc.csv"
    if not _provider_has(provider, "solve_data/na_online_fix_in_uc", path):
        return False
    try:
        frame = _provider_read(provider, "solve_data/na_online_fix_in_uc", path)
    except pl.exceptions.NoDataError:
        return False
    return (frame is not None and "value" in frame.columns
            and "yes" in frame["value"].cast(pl.Utf8).to_list())


def _overlay_handoff(flex_data: "FlexData", handoff,
                       solve_data_dir: Path | None = None,
                       *,
                       ctx: "object | None" = None,
                       provider: "object | None" = None) -> "FlexData":
    """Δ.11 — internal helper used by :func:`load_flextool` to overlay an
    in-memory :class:`SolveHandoff` onto the FlexData built from disk.

    Returns a NEW FlexData with the 4 carrier-derived fields replaced
    (uses :func:`dataclasses.replace`, original untouched).  Called from
    inside :func:`load_flextool` when ``handoff`` is supplied — there is
    no longer a public ``apply_handoff`` entry point; the construct-with-
    handoff path is the only supported way to pipe an in-memory
    :class:`SolveHandoff` into a fresh :class:`FlexData`.

    Carriers overlaid (target FlexData fields):

    * ``p_entity_previously_invested_capacity (e, d)`` / ``p_entity_invested (e,)`` ← the frames the solve emitted
      from ``realized_cohort`` (``solve_data/p_entity_previously_invested_capacity`` /
      ``solve_data/p_entity_invested``).
    * ``p_entity_divested (e,)``  ← ``divest_cumulative``.
    * ``p_fix_storage_quantity (n, d, t)``  ← ``fix_storage_quantity``.

    ``p_roll_continue_state`` is not overlaid: the cascade routes the storage carry (the latest realised state at
    the sub-solve's nesting level or deeper, ``solve._carry_routing``) into ``handoff/roll_end_state``, which the
    loader reads; the sequential predecessor's ``roll_end_state`` is not always that state.

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
        Carrier set built by :func:`build_handoff_from_solution` from the
        prior sub-solve's polar_high solution.
    solve_data_dir : Path, optional
        Path to the current sub-solve's ``solve_data/`` directory (the keys of the emitted frames).  ``None`` skips
        ``p_entity_previously_invested_capacity`` / ``p_entity_invested``; the snapshot's value is used as-is.
    """
    from dataclasses import replace
    overrides: dict = {}

    # --- p_entity_previously_invested_capacity (e, d) / p_entity_invested (e,): the carried cohorts in service and
    # all of them, as the solve emitted them from the handoff ``realized_cohort``
    # (``_chain_params.emit_existing_capacity``): the values every other consumer reads.
    if solve_data_dir is not None:
        from flextool.engine.preprocess.params._derived import (
            p_entity_invested_from_workdir,
            p_entity_previously_invested_capacity_from_workdir,
        )
        overrides["p_entity_previously_invested_capacity"] = p_entity_previously_invested_capacity_from_workdir(
            solve_data_dir.parent, provider=provider)
        overrides["p_entity_invested"] = p_entity_invested_from_workdir(solve_data_dir.parent, provider=provider)

    # --- divest_cumulative → p_entity_divested (e,) ---
    if handoff.divest_cumulative is not None:
        df = (handoff.divest_cumulative
            .pipe(rename_to_axis, {"entity": "e"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                 .fill_null(0.0))
            .filter(pl.col("value") != 0.0)
            .select("e", "value"))
        overrides["p_entity_divested"] = (
            Param(("e",), df) if df.height > 0 else None)

    # ``p_roll_continue_state`` is not overlaid: the storage carry is routed by the cascade
    # (``solve._carry_routing``) into ``handoff/roll_end_state`` and read from there (the single source).

    # --- fix_storage_quantity → p_fix_storage_quantity (n, d, t) ---
    # Phase 4.1k — read the canonical narrow field
    # ``SolveHandoff.fix_storage_quantity`` (schema
    # ``[node, period, step, p_fix_storage_quantity]``) populated by
    # ``build_handoff_from_solution`` (4.1c).  Mirrors the Provider-key
    # consumer at L2620-2641.
    if handoff.fix_storage_quantity is not None:
        df = (handoff.fix_storage_quantity
            .pipe(rename_to_axis, {"node": "n", "period": "d", "step": "t",
                     "p_fix_storage_quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
            .select("n", "d", "t", "value"))
        overrides["p_fix_storage_quantity"] = (
            Param(("n", "d", "t"), df) if df.height > 0 else None)

    if not overrides:
        return flex_data
    return _assign_param_names(replace(flex_data, **overrides))
