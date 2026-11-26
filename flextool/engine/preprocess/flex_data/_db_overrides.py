"""The DB-direct construction passes :func:`load_flextool` applies to a loaded FlexData."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.flex_data._readers import _loadflex_prof
from flextool.engine.preprocess.flex_data._container import FlexData


def _apply_db_overrides(flex_data: "FlexData", db_reader: "InputSource",
                          source: "object",
                          ctx: "object | None" = None,
                          *,
                          provider: "object | None" = None) -> None:
    """Apply the DB-direct construction passes to ``flex_data`` in
    place.  Δ.3 consolidates the previous 9-wrapper override chain into
    a single linear sequence; each ``apply_*`` callee mutates
    ``flex_data`` directly via setattr (no dict round-trip).

    Pass order (preserved from the legacy chain — each pass may depend
    on fields written by earlier passes):
      1. Direct Params — scalar + relationship 1d_map.
      2. Projection Params — entity-instance + reserve-method partitions.
      3. Derived A — dt / step duration / weighting / inflow / profiles.
      4. Derived B — process topology + reclassified method-derived.
      5. Derived C — invest/divest + online/UC + group slack.
      6. Derived D — p_entity_all_existing, node_reference_angle, etc.
      7. Derived E — storage block algebra.
      8. Derived F — lifetime cascade + handoff state + full inflation.
      9. Derived G — commodity ladder, reserves, delay, multi-branch.

    Δ.12a — when *ctx* is supplied, helpers may consume the typed
    in-memory state (``ctx.solve_name``, ``ctx.realized_periods``, …)
    instead of re-reading the provider frames.  The ctx is built by
    :func:`load_flextool` from the provider before this function runs.
    """
    # The run work folder comes first: passes 3-10 read it, and without it only a subset of the DB-direct
    # passes could run, which would leave CSV-seeded fields in place of DB values.  Resolve it before any
    # pass so that a source without one fails loudly instead of producing a partial overlay.
    # CsvSource exposes ``workdir``; any other FlexInputSource exposes ``input_dir`` (= ``<workdir>/input``).
    try:
        workdir = source.workdir if hasattr(source, "workdir") \
                   else source.input_dir.parent
    except Exception:
        workdir = None
    if workdir is None:
        raise RuntimeError(
            "Cannot apply the database values to the model input: the input source "
            f"({type(source).__name__}) has no workdir (no 'workdir' and no usable 'input_dir'). "
            "The DB overrides need the run work folder (the folder that holds input/ and solve_data/). "
            "Pass load_flextool a work-folder path or a source that names its work folder."
        )
    workdir_path = Path(workdir)

    from flextool.engine.preprocess.params import _direct as _dp
    from flextool.engine.preprocess.params import _projection as _pp
    from flextool.engine.preprocess.params import _derived as _drv

    # F1 — defensive per-solve reset of the recourse process-globals BEFORE
    # any cascade pass runs, so no producer can inherit a stale anchor-pairs
    # holder / walker Provider from a prior solve in a chained multi-solve.
    # (The flag-aware Layer-4 boundaries still overwrite them; this just
    # guarantees the invariant for any future earlier-step per-period producer.)
    _drv._reset_recourse_scope()

    from flextool.engine.solve._orchestration import get_phase_recorder
    _rec = get_phase_recorder()
    _logger = logging.getLogger("flextool.engine.preprocess.flex_data")

    def _timed(label, fn, *args, **kwargs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if _rec is not None:
            # Emit via the phase recorder so the line carries RSS +
            # Δrss + Δpeak (when full diagnostics is on) in the same
            # format as the cascade/build checkpoints.
            _rec.checkpoint(
                f"input_pass_{label.split()[0]}",
                _logger,
                user_label=f"input pass {label}",
            )
        else:
            # Fallback: legacy plain timer line (used by unit tests
            # that don't go through run_orchestration).
            print(f"  input pass {label}: {elapsed:.3f}s")

    # Pass 1a-2: source-only Params (they do not read the workdir).
    # Δ.28 — pass 1 splits into 1a (dt-independent) and 1b (dt-dependent).
    # Pass 1b runs after ``apply_derived_a`` populates ``flex_data.dt``
    # so scalar / 1d_map[period] / 1d_map[time] values authored on the
    # source can broadcast across the active solve's (d, t) axis.  See
    # ``_direct.apply_direct_params_a/b`` docstrings for the
    # full Δ.28 rationale.
    _loadflex_prof("apply_db_overrides:pass_direct_params_a")
    _timed("1a direct_params_a", _dp.apply_direct_params_a, db_reader, flex_data)
    _loadflex_prof("apply_db_overrides:pass_projection_params")
    _timed("2  projection_params", _pp.apply_projection_params, db_reader, flex_data)

    # Pass 3-9: workdir-aware Params (the workdir was resolved at entry).

    # Δ.18 — synthetic per-sub-solve detection.  When the workdir's
    # ``solve_data/solve_current.csv`` names a solve that doesn't exist
    # in Spine (e.g. nested-multi-invest fixtures whose orchestrator emits
    # per-period sub-solve names like ``invest_5weeks_p2020`` that are
    # synthesized at runtime, not declared in the data DB), the per-solve
    # override chain (``apply_derived_a..g`` + ``apply_existing_chain``)
    # would return None for every helper that takes ``active_solve`` as
    # a key — wiping out the snapshot CSV seeds those helpers would have
    # otherwise overlaid.  For these synthetic-solve workdirs, the
    # snapshot CSV is the canonical and the per-solve overrides must be
    # skipped entirely.  Direct + Projection Params (passes 1-2) are
    # solve-agnostic and remain authoritative, and the solve-agnostic
    # producers re-wired below (RESERVE-1, SECTION-1, MINLOAD-1, PROFILE-1 —
    # profiles are read DB-direct —, PENALTY-1, DELAY-1, CAPCOEF-1, INFLOW-1,
    # NA-1 with the scenario-lineage frames, MINTIME-1) replace their CSV seeds.
    #
    # Δ.19 — for the synthetic ``<base>_<anchor>`` shape we additionally
    # produce the 8 invest-set frames from ``<base>``'s Spine entries
    # filtered to the anchor's period subset (see
    # :func:`_derived._resolve_synthetic_solve` and
    # :func:`_derived.apply_synthetic_invest_sets`).  This cuts the
    # 8 invest-set disk reads in ``_invest_seeds.py``.  The investment
    # annuity and lifetime fixed cost come from ``_npv.apply_invest_lumps``
    # (in ``apply_run_fields_of_synthetic_solve`` below); the divest lumps
    # and the per-period caps stay on the CSV-seed path.
    active_solve = _drv._read_active_solve(workdir_path, provider=provider)
    if active_solve is not None and not _drv._solve_in_spine(db_reader,
                                                                  active_solve):
        synth = _drv._resolve_synthetic_solve(db_reader, active_solve)
        if synth is not None:
            _loadflex_prof("apply_db_overrides:pass_synthetic_invest_sets")
            _timed("c.synth invest_sets",
                   _drv.apply_synthetic_invest_sets,
                   flex_data, db_reader, active_solve, synth, workdir_path,
                   provider=provider)
        # A nested sub-solve (``<base>_<period>``) never gets the recourse wiring: ``recourse_invest`` and
        # the recourse anchor scope are set only in ``apply_derived_c``, which this early return skips, so
        # its forecast investments would lose their non-anticipativity ties without a word.  Refuse it while
        # the sub-solve can invest.  The rolls of a rolling solve are synthetic too but make no investment
        # (a rolling recourse solve is warned about and checked on its own), so they are not affected.
        _pi = getattr(flex_data, "period_invest", None)
        if (_pi is not None and _pi.height > 0 and not getattr(flex_data, "solve_is_rolling", False)
                and _drv._recourse_invest_active(workdir_path, ctx=ctx, provider=provider)):
            from flextool.engine.input._solve_state import FlexToolConfigError
            raise FlexToolConfigError(
                f"Nested sub-solve '{active_solve}' has an invest_solve_branching_method whose investments branch "
                "(investments_after_reveal_period or investments_at_reveal_period), which a nested solve does not "
                "support: make it a top-level solve, or use operations_only.")
        # Δ.28 — synthetic solve: apply_derived_a is skipped, but in the
        # slow path ``flex_data.dt`` is already populated from CSV (see
        # ``_load_time``).  Run pass 1b so broadcast-needing Direct
        # Params still apply against the CSV-loaded dt.
        _loadflex_prof("apply_db_overrides:pass_direct_params_b")
        _timed("1b direct_params_b", _dp.apply_direct_params_b, db_reader, flex_data,
               active_solve=active_solve)

        # RESERVE-1 — the synthetic-solve early-return skips passes 3-10
        # entirely, but two reserve-feature fields are produced by those
        # passes and are required by ``_reserve.add_variables`` /
        # ``add_constraints`` when ``reserve_upDown_group`` is non-empty
        # (the CSV-seeded ``reserve_upDown_group`` activates the reserve
        # feature).  Both producers are solve-agnostic
        # (no ``active_solve`` filter), so calling them on the synthetic
        # path is safe and mirrors what ``apply_derived_d`` / ``apply_derived_g``
        # do on the non-synthetic path.  See
        # ``specs/model_bugs.md::RESERVE-1`` for the full diagnosis.
        def _wire_reserve_for_synthetic_solve():
            flex_data.process_reserve_upDown_node_active = (
                _drv.process_reserve_upDown_node_active_from_source(db_reader))
            flex_data.prundt = _drv.prundt_from_source(
                db_reader, active_solve, getattr(flex_data, "dt", None))
        _loadflex_prof("apply_db_overrides:pass_synthetic_reserve")
        _timed("c.synth reserve", _wire_reserve_for_synthetic_solve)

        # SECTION-1 — same class of gap as RESERVE-1.  The synthetic-solve
        # early-return skips passes 3-10, so ``apply_derived_c``'s
        # ``p_section`` producer never runs.  After the Δ.12-drop deleted
        # the ``pdtProcess_section.csv`` loader seed (see ``_load_online``),
        # ``apply_derived_c`` is the SOLE producer of ``flex_data.p_section``
        # — so on a rolling sub-solve the CSV-seeded ``process_min_load_eff``
        # (Projection Params, pass 2) stays populated while ``p_section`` is
        # left ``None``.  ``build_flextool``'s ``MINLOAD_EFF`` ``_check`` then
        # raises, and (when it doesn't) the section energy-balance term
        # (``model/core/`` online cost / flow) and the section cost output are
        # silently dropped because both gate on ``p_section is not None``.
        # ``p_section_from_source`` is solve-agnostic (no ``active_solve``
        # filter), so calling it here mirrors ``apply_derived_c`` exactly
        # (``_derived`` §3.8.1 wiring).  See
        # ``specs/model_bugs.md::SECTION-1``.
        def _wire_section_for_synthetic_solve():
            dt = getattr(flex_data, "dt", None)
            if dt is None or dt.height == 0:
                return
            classified = _drv._classify_process_method(db_reader)
            if classified is None or classified.height == 0:
                return
            flex_data.p_section = _drv.p_section_from_source(
                db_reader, dt, classified)
        _loadflex_prof("apply_db_overrides:pass_synthetic_section")
        _timed("c.synth section", _wire_section_for_synthetic_solve)

        # MINLOAD-1 — same class of gap as SECTION-1 / RESERVE-1.  The
        # synthetic-solve early-return skips passes 3-10, so
        # ``apply_derived_b``'s ``p_min_load`` producer never runs.  After
        # this fix moved ``p_min_load`` out of pass 1a (``apply_direct_params``)
        # into ``apply_derived_b`` (so a Map min_load becomes a shape-correct
        # (p, d, t) Param instead of a summed (p,) scalar — see
        # ``specs/fix_map_reading.md`` §5), a rolling / synthetic sub-solve
        # would leave ``p_min_load`` ``None``, and the minFlow_minload floor of
        # every unit with min_load > 0 would fail ``model/core/``'s
        # ``_check(MIN_LOAD)`` on the synthetic path.
        # ``p_min_load_pdt_from_source`` is solve-agnostic (no ``active_solve``
        # filter) and reads the CSV-loaded ``flex_data.dt``, so calling it here
        # mirrors ``apply_derived_b`` exactly.
        def _wire_min_load_for_synthetic_solve():
            dt = getattr(flex_data, "dt", None)
            if dt is None or dt.height == 0:
                return
            flex_data.p_min_load = _drv.p_min_load_pdt_from_source(
                db_reader, dt)
        _loadflex_prof("apply_db_overrides:pass_synthetic_min_load")
        _timed("c.synth min_load", _wire_min_load_for_synthetic_solve)

        # PROFILE-1 — same class of gap as SECTION-1 / MINLOAD-1.  The
        # synthetic-solve early-return skips ``apply_derived_a``, whose
        # ``apply_profile_cascade`` is the DB-direct producer of
        # ``p_profile_value``; the CSV seed (emit ``pdtProfile``) has no period
        # tier, so a period-keyed profile (1-D period or 2-D period x time map)
        # was read as 0 in every roll / nested sub-solve.  The cascade is
        # solve-agnostic (``source`` + the CSV-loaded ``dt`` + the per-solve
        # workdir CSVs for the stochastic and time-averaged tiers), so this
        # mirrors ``apply_derived_a`` exactly.  Per roll it costs only the
        # ``dt`` joins: the time tier reads the per-roll ``pt_profile.csv``,
        # and the scalar / period / period x time tiers unroll just their
        # own profiles once per run (memoised on the run's ``db_reader``).
        from flextool.engine.preprocess.params._profiles import apply_profile_cascade
        _loadflex_prof("apply_db_overrides:pass_synthetic_profile")
        _timed("c.synth profile", apply_profile_cascade, flex_data, db_reader,
               workdir_path, provider=provider)

        # PENALTY-1 — same class of gap as PROFILE-1.  The synthetic-solve
        # early-return skips ``apply_derived_a``, whose step 9 is the
        # DB-direct producer of ``p_penalty_up`` / ``p_penalty_down``; the
        # seed (emit ``pdtNode``) has no period x time tier.  The producer is
        # solve-agnostic (``source`` + the CSV-loaded ``nodeBalance`` / ``dt``)
        # and memoises the authored values on the run's ``db_reader``, so a
        # roll pays only the ``dt`` joins.  Only a value the resolver cannot classify (or an
        # empty node set / ``dt``) keeps the seed; a stochastic-time map is resolved by the
        # producer's own per-solve fold, not left to the seed.
        from flextool.engine.preprocess.params._arithmetic import apply_penalty_params
        _loadflex_prof("apply_db_overrides:pass_synthetic_penalty")
        _timed("c.synth penalty", apply_penalty_params, flex_data, db_reader,
               getattr(flex_data, "dt", None))

        # DELAY-1 — same class of gap: ``apply_derived_g`` produces
        # ``dtt__delay_duration``, ``p_process_delay_weight`` and
        # ``process_delayed__duration`` together (one source read, Float64
        # ``td``).  The loader does not seed them (the former CSV seeds' labels
        # did not have to match the duration set and made a rolling run with a
        # delayed unit crash on the ``td`` join).  The producer is
        # solve-agnostic and builds the shift table over this sub-solve's
        # ``dt`` exactly like the per-solve CSV writer.
        _loadflex_prof("apply_db_overrides:pass_synthetic_delay")
        _timed("c.synth delay", _drv.apply_delay_params, flex_data, db_reader)

        # CAPCOEF-1 — same class of gap: ``p_arc_max_cap_coef`` /
        # ``process_source_sink_uncapped`` (apply_derived_b) and
        # ``p_indirect_input_cap`` (apply_derived_d) are skipped on the
        # synthetic path, so roll sub-solves lost the output
        # ``capacity_max_coeff``, the ``conversion_flow_coeff = 0``
        # uncapped edges and the indirect input-arc limits.  All three
        # producers are solve-agnostic; runs after the section wiring
        # because the input-arc limit reads ``p_section``.
        _loadflex_prof("apply_db_overrides:pass_synthetic_arc_capacity")
        _timed("c.synth arc capacity", _drv.wire_arc_capacity_params,
               flex_data, db_reader)

        # INFLOW-1 — synthetic solves skip ``apply_derived_c``; re-wire the
        # inflow signed-split so the non_sync / capacity_margin demand budget
        # isn't silently lost.  ``apply_derived_c`` (``_derived`` §3.12)
        # builds ``p_positive_inflow`` / ``p_negative_inflow`` /
        # ``pdtNodeInflow_per_step`` from the real inflow (``flex_data.p_inflow_unmargined``
        # when an energy margin left one on the Provider, else ``flex_data.p_inflow``)
        # and ``flex_data.p_step_duration``, all of which are already loaded
        # at ``FlexData`` construction (``_load_node`` / step_dur) BEFORE this
        # early-return and are correctly sized for the synthetic sub-solve
        # (each solve emits its own ``pdtNodeInflow.csv``).  All three helpers
        # are solve-agnostic (no ``active_solve`` filter) and the inputs match
        # ``apply_derived_c``'s exactly, so this is byte-identical to the
        # non-synthetic path.  Without it ``_group_slack._eq_non_sync`` (RHS ``exo_demand``
        # / LHS ``p_positive_inflow`` terms gate on these being non-None) and
        # ``_eq_capacity_margin`` (``pdtNodeInflow_per_step``) silently drop the
        # demand budget — VRE pins to 0 and coal backfills.  See
        # ``specs/model_bugs.md::INFLOW-1``.
        def _wire_inflow_split_for_synthetic_solve():
            inflow_real = _drv.real_inflow(getattr(flex_data, "p_inflow", None),
                                           getattr(flex_data, "p_inflow_unmargined", None))
            sd_csv = getattr(flex_data, "p_step_duration", None)
            flex_data.p_positive_inflow = _drv.p_positive_inflow_from_inflow(inflow_real)
            flex_data.p_negative_inflow = _drv.p_negative_inflow_from_inflow(inflow_real)
            flex_data.pdtNodeInflow_per_step = _drv.pdtNodeInflow_per_step_from_inflow(inflow_real, sd_csv)
        _loadflex_prof("apply_db_overrides:pass_synthetic_inflow")
        _timed("c.synth inflow", _wire_inflow_split_for_synthetic_solve)

        # The non-synchronous groups' internal transfers: the same rule as ``apply_derived_c`` (the
        # sub-solve's process_source_sink / group_node / groupNonSync are loaded), not the CSV seed.
        flex_data.process_group_inside_nonSync = _drv.process_group_inside_nonSync_from_flex(flex_data)

        # NA-1: the synthetic-solve early-return skips passes 3-10, so
        # ``apply_derived_g``'s ``apply_branch_cluster`` — the sole producer of
        # ``period_branch_full`` / ``dt_non_anticipativity`` and of the
        # scenario-lineage frames (``dd_same_scenario``,
        # ``pd_non_anticipativity``, ``d_leaf``) — never runs on a
        # rolling sub-solve (each ``<base>_rolling_roll_N`` names a solve absent
        # from Spine).  Without this wire both frames keep their None seeds
        # (above), the non-anticipativity gate never fires, and stochastic
        # branches stay uncoupled across the rolling solve.  The two ``*_df``
        # readers are solve-agnostic — they unroll the per-roll workdir CSVs
        # (``period__branch.csv`` / ``dt_non_anticipativity_set.csv``) already
        # written correctly by the per-roll emit — returning None for a
        # non-stochastic roll.  We replicate ONLY ``apply_branch_cluster``'s two
        # frame assignments (seed-preserving overlay) and deliberately do NOT
        # call the full helper, which would overwrite the already-seeded
        # ``period_in_use_set`` via an unsorted ``.unique()``.  Safe for
        # non-stochastic synthetic solves: no d≠b fan rows → ``db_pairs`` empty
        # → the gate's d≠b guard skips the constraint.
        #
        # The lineage frames come from the same ``apply_lineage_frames`` call
        # single solves make (it does not touch ``period_in_use_set``): without
        # ``d_leaf`` the per-path families (cumulative ladder tier cap,
        # co2_max_total, cumulative flow limits, invest totals) pooled every
        # forecast of a roll into one row.  A roll without a fan gets a single
        # ``__anchor`` leaf, so its rows stay the legacy ones.
        #
        # ``period_branch_full`` also feeds ``synthetic_branch_tokens`` (invest
        # handoff guards) and ``is_genuinely_stochastic`` (benders); on a
        # rolling-stochastic solve WITH investment or benders those now
        # activate (parity with the single-solve path), not yet fixture-covered.
        def _wire_non_anticipativity_for_synthetic_solve():
            from flextool.engine.preprocess.params._branch import (
                apply_lineage_frames,
                dt_non_anticipativity_df,
                period_branch_full_df,
            )
            _new_pbf = period_branch_full_df(workdir_path, provider=provider)
            if _new_pbf is not None or getattr(
                    flex_data, "period_branch_full", None) is None:
                flex_data.period_branch_full = _new_pbf
            _new_dtn = dt_non_anticipativity_df(
                workdir_path, provider=provider)
            if _new_dtn is not None or getattr(
                    flex_data, "dt_non_anticipativity", None) is None:
                flex_data.dt_non_anticipativity = _new_dtn
            apply_lineage_frames(flex_data, workdir_path, db_reader,
                                 active_solve, ctx=ctx, provider=provider)
        _loadflex_prof(
            "apply_db_overrides:pass_synthetic_non_anticipativity")
        _timed("c.synth non_anticipativity",
               _wire_non_anticipativity_for_synthetic_solve)

        # MINTIME-1 — the synthetic-solve early return skips apply_derived_e, which builds the min_uptime /
        # min_downtime look-backs (``uptime_lookback`` / ``pdt_uptime_set`` and the downtime twins, the sole inputs
        # of the ``minimum_uptime`` / ``minimum_downtime`` constraints).  Without them a rolling sub-solve would
        # drop the constraints even within a roll.  ``apply_mintime_lookbacks`` reads only ``flex_data`` frames
        # populated before this return and does nothing when no process authors a minimum time.
        def _wire_mintime_for_synthetic_solve():
            # The look-back follows flex_data.dt_ramp_link (seeded from solve_data/step_previous before this
            # early return): the same predecessor links the ramp rows of this roll use.  The roll carry
            # (``uptime_carry`` / ``downtime_carry``) starts at the same roll's ``dt_ramp_carry`` steps.
            _drv.apply_mintime_lookbacks(flex_data, db_reader, _drv._roll_first_step_link(workdir_path, provider),
                                         active_solve)
        _loadflex_prof("apply_db_overrides:pass_synthetic_mintime")
        _timed("c.synth mintime", _wire_mintime_for_synthetic_solve)

        # BLOCK-1 — same class of gap.  ``nodeStateBlock`` (the bind_intraperiod_blocks nodes) is CSV-seeded, but
        # its timeset blocks (``period_block`` / ``period_block_succ`` / ``period_block_time``,
        # ``dtttdt_block_interior``) are produced only by ``apply_derived_e``, so a nested anchor sub-solve
        # (``<solve>_<period>``) had a block node without blocks — no energy balance at all.  The producer reads
        # this sub-solve's own steps (``_dt_period_active_steps`` falls back to ``steps_in_use``), so it gives
        # what the same solve gets as a Spine solve.  The rolls of a rolling solve are synthetic too: their blocks
        # are also cut at the roll's realized end, and the periods a roll holds only in part chain forward
        # (``period_block_chain``, see ``_derived.dtttdt_block_lag_from_dtttdt``).
        def _wire_blocks_for_synthetic_solve():
            nsb = getattr(flex_data, "nodeStateBlock", None)
            if nsb is None or nsb.height == 0:
                return
            family = _drv.period_block_family_from_source(
                db_reader, active_solve, workdir_path, ctx=ctx, provider=provider,
                realized=_drv._rolling_realized_steps(flex_data, workdir_path, provider))
            if family is None:
                return
            from flextool.engine.preprocess._step_relations import null_split_block_wraps
            pbf = getattr(flex_data, "period_branch_full", None)
            flex_data.period_block = family["period_block"]
            flex_data.period_block_succ = null_split_block_wraps(family["period_block_succ"], pbf, flex_data.dt)
            flex_data.period_block_time = family["period_block_time"]
            flex_data.dtttdt_block_interior = _drv.dtttdt_block_interior_from_dtttdt(
                getattr(flex_data, "dtttdt", None), family["period_block_time"])
            flex_data.dtttdt_block_lag = _drv.dtttdt_block_lag_from_dtttdt(
                getattr(flex_data, "dtttdt", None), family["period_block_time"],
                getattr(flex_data, "period_block_chain", None), pbf)
        _loadflex_prof("apply_db_overrides:pass_synthetic_blocks")
        _timed("c.synth blocks", _wire_blocks_for_synthetic_solve)

        # ``apply_derived_f`` (``apply_npv``) is skipped: the investment annuity and lifetime fixed cost of a
        # sub-solve that invests are derived here by the same ``_npv`` rule over its own periods, and the run-level
        # fields of the reported costs (outputs only) discount from the start of the run over the whole run.
        from flextool.engine.preprocess.params._lifetime_costs import apply_run_fields_of_synthetic_solve
        _timed("c.synth run costs", apply_run_fields_of_synthetic_solve, flex_data, db_reader, workdir_path,
               provider=provider)
        return

    _loadflex_prof("apply_db_overrides:pass_derived_a")
    _timed("3  derived_a", _drv.apply_derived_a, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)

    # Δ.28 — pass 1b: broadcast-needing Direct Params now have
    # ``flex_data.dt`` populated by ``apply_derived_a`` step 1.  The
    # broadcast helpers (``broadcast_to_period_time`` /
    # ``broadcast_to_period`` / ``_entity_period_scalar`` /
    # ``_entity_period_time_param``) require non-empty ``period_filter``
    # to fan scalar/1d_map values across the (d, t) axis; running this
    # pass before ``apply_derived_a`` (the legacy ordering) left
    # ``p_commodity_price`` / ``p_process_availability`` / similar
    # fields empty on the fast path even when Spine carried the data.
    _loadflex_prof("apply_db_overrides:pass_direct_params_b")
    _timed("1b direct_params_b", _dp.apply_direct_params_b, db_reader, flex_data,
               active_solve=active_solve)

    _loadflex_prof("apply_db_overrides:pass_derived_b")
    _timed("4  derived_b", _drv.apply_derived_b, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _loadflex_prof("apply_db_overrides:pass_derived_c")
    _timed("5  derived_c", _drv.apply_derived_c, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _loadflex_prof("apply_db_overrides:pass_derived_d")
    _timed("6  derived_d", _drv.apply_derived_d, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _loadflex_prof("apply_db_overrides:pass_derived_e")
    _timed("7  derived_e", _drv.apply_derived_e, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _loadflex_prof("apply_db_overrides:pass_derived_f")
    _timed("8  derived_f", _drv.apply_derived_f, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _loadflex_prof("apply_db_overrides:pass_derived_g")
    _timed("9  derived_g", _drv.apply_derived_g, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)

    # The chain-carried constants (the carried retirements, the lead-time capacity) last.
    from flextool.engine.preprocess.params import _existing as _ex
    _loadflex_prof("apply_db_overrides:pass_existing_chain")
    _timed("10 existing_chain", _ex.apply_existing_chain, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
