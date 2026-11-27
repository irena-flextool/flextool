"""The master loop: :func:`_drive_cascade` drives ``native_run_model`` with the polar_high cascade solver
(load, build or warm reuse, solve, outputs, handoff) for every sub-solve."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Collection

from flextool import run_record, run_warnings
from flextool.engine.solve._acceptance import (
    classify_acceptance,
    highs_objective_bound_warning,
)
from flextool.engine.solve._solver_outcome import note_solver_outcome
from flextool.engine.solve import _objective_options
from flextool.engine.solve._objective_options import (
    check_benders_options as _check_benders_objective_options,
)
from flextool.engine.preprocess._stochastic_detect import (
    is_genuinely_stochastic as _is_genuinely_stochastic,
)
from flextool.engine.input._solve_state import FlexToolConfigError, RunnerState
from flextool.engine.autoscale import (
    Layer2Plan as _AutoscaleLayer2Plan,
    RangeReport as _AutoscaleRangeReport,
    RollRecheck as _AutoscaleRollRecheck,
    enforce_scaling_guard as _autoscale_enforce_guard,
    resolve_guard_mode as _autoscale_resolve_guard_mode,
    resolve_scaling_config as _autoscale_resolve_config,
    resolve_user_bound_scale_override as _resolve_user_bound_scale_override,
)

if TYPE_CHECKING:
    import polars as pl
    from polar_high.autoscale import ScalingConfig

from flextool.engine.solve._orchestration._highs_options import (
    _baseline_highs_options,
    _basis_cache_active,
    _basis_cache_dir,
    _format_solver_args_line,
    _highs_echo_translations,
    _highs_option_layers,
    _log_cold_rebuild,
    _print_solver_args,
    _resolve_effective_obj_scale,
    _run_options_or_env,
    _solve_highs_options,
)
from flextool.engine.solve._orchestration._autoscale_hooks import (
    _AutoscaleShapeCacheEntry,
    _autoscale_apply_layer2_from_cache,
    _autoscale_apply_layer2_pre_solve,
    _autoscale_apply_layer3_from_cache,
    _autoscale_apply_layer3_pre_solve,
    _autoscale_disable_cache,
    _autoscale_emit_console_summary,
    _autoscale_emit_layer1,
    _autoscale_guard_check,
    _autoscale_lp_shape_signature,
    _autoscale_report_recheck_stop,
    _autoscale_unscale_post_solve,
    _autoscale_write_presolve_report,
    _roll_label,
)
from flextool.engine.solve._orchestration._memory import (
    _NoopMemoryRecorder,
    _audit_cold_rebuild_release,
    _audit_prior_level_release,
    _phase_prof,
    _try_malloc_trim,
    get_phase_recorder,
)
from flextool.engine.solve._orchestration._steps import (
    OrchestrationStep,
    SnapshotSolution,
    _emit_provider_obj_scale,
    _write_scale_csv,
)
from flextool.engine.solve._orchestration._guards import (
    _benders_consume_guard_message,
    _dispatch_realization_raw_guard,
    _invest_branching_raw_guard,
    _spawn_realized_dispatch_siblings,
)


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------


def _drive_cascade(
    state: RunnerState,
    work_folder: Path,
    solves: list[str],
    runner_factory,
    *,
    db_url: str | None = None,
    scenario_name: str | None = None,
    keep_solutions: bool = False,
    warm: bool = True,
) -> dict[str, OrchestrationStep]:
    """Drive the flextool master loop with a polar_high cascade solver.

    For every per-solve iteration:

    1. Read the snapshot via ``load_flextool``.
    2. Build the LP via ``build_flextool`` (cold rebuild) OR reuse the
       prior iteration's :class:`polar_high.WarmProblem` when the
       sub-solve is identical.
    3. Solve via HiGHS.
    4. Build the handoff via ``build_handoff_from_solution``.
    5. Deposit it into ``state.handoffs`` so the next iteration's
       preprocessing picks it up.

    Parameter ``warm`` toggles the in-process WarmProblem path: when True,
    the cascade reuses the built ``WarmProblem`` for the next iteration
    only when its FlexData and solver configuration are identical (the
    reuse contract in :mod:`flextool.engine.solve._warm`); the warm path
    pushes no updates.  ``warm=False`` builds every sub-solve cold and
    routes HiGHS through the ``cmd_solve_mps`` subprocess.

    Emits an :class:`OrchestrationStep` per solve and runs the LAST
    solve too (the polar_high-side bookkeeping is the deliverable,
    not just an intermediate).
    """
    # Late imports — keep the orchestration module's import surface narrow
    # for callers that only need the dataclass.
    from flextool.engine.solve._solver_base import SolverRunner
    from flextool.engine.solve._run_model import native_run_model

    from polar_high import NamedBasis, Problem, WarmProblem
    from flextool.engine.preprocess.flex_data import (
        build_handoff_from_solution,
        load_flextool,
    )
    from flextool.engine.model.core import build_flextool
    from flextool.engine.output._horizon_dump import (
        output_horizon_enabled,
        snapshot_cstr_keys,
        warn_benders_no_dump,
    )
    from flextool.engine.output._writer import (
        OutputWriterState,
        write_outputs_for_solve,
    )
    from flextool.engine.solve._warm import (
        _IncompatibleUpdate,
        _apply_warm_updates,
        _build_warm_problem,
        _fingerprint,
    )

    results: dict[str, OrchestrationStep] = {}
    # Δ.1: adapter that reuses flextool's process_outputs writers.  The
    # state carrier collects ``periods_already_emitted`` across the
    # cascade so we don't have to round-trip through SolveHandoff.
    writer_state = OutputWriterState()

    # The runner_factory hook lets tests inject a mock; the default uses
    # FlexToolRunner constructed against the same DB the state was
    # loaded from.  Since RunnerState doesn't carry a DB URL
    # by default, callers must supply this via runner_factory or use
    # ``run_chain_from_db`` which constructs the runner explicitly.
    if runner_factory is None:
        raise FlexToolConfigError(
            "run_orchestration requires a runner_factory to construct "
            "the underlying FlexToolRunner.  Use run_chain_from_db for "
            "the canonical end-to-end path that wires this for you."
        )
    if db_url is None or scenario_name is None:
        raise FlexToolConfigError(
            f"_drive_cascade needs db_url and scenario_name (got db_url={db_url!r}, "
            f"scenario_name={scenario_name!r}): every sub-solve loads its parameters through a "
            "SpineDbReader for that scenario, and the cascade has no reader-less mode. Use "
            "run_chain_from_db(input_db_url, scenario_name)."
        )
    _drive_rec = get_phase_recorder()
    _drive_logger = state.logger
    runner = runner_factory()
    if _drive_rec is not None:
        _drive_rec.checkpoint(
            "flextool_runner_constructed", _drive_logger,
            user_label="FlexToolRunner constructed",
        )
    # Push our state's handoff slot onto the runner's state so the
    # cascade and any consume hooks share the same dict.
    runner.state.handoffs = state.handoffs
    # The solves run on the runner's state: give it the option files the run
    # read at its start (``run_chain_from_db``), so no solve reads them again.
    if state.paths is not None and state.paths.solver_option_files is not None:
        runner.state.paths.solver_option_files = state.paths.solver_option_files
    # Per-level Provider cache (Design A).  ``native_run_model`` lazily
    # initialises this on first iter, but seeding it here makes the
    # invariant ``state._level_providers is dict`` explicit at every
    # entry point (cascade + fast_load) instead of relying on hasattr
    # probes downstream.
    runner.state._level_providers = {}
    # Step 2.5 — forward the cascade-input Provider seeded in
    # ``run_chain_from_db`` onto runner.state so the per-sub-solve hook
    # at :mod:`flextool.engine.solve._run_model` (line 365-370)
    # picks it up.  ``None`` is allowed for entry points that bypass
    # ``run_chain_from_db`` — the hook then builds an empty Provider.
    _cip = getattr(state, "cascade_input_provider", None)
    if _cip is not None:
        runner.state.cascade_input_provider = _cip
    # Phase 5c — forward the engine-side ``override_provider``
    # callable onto ``runner.state`` so the per-sub-solve hook in
    # :mod:`flextool.engine.solve._run_model` (Phase 5b) picks
    # it up.  ``None`` keeps the no-override default.
    _op = getattr(state, "override_provider", None)
    if _op is not None:
        runner.state.override_provider = _op
    # ``--csv-dump``: the per-sub-solve gates (``dump_csvs`` and the handoff writers' dump branch) read
    # ``runner.state``, so the flag must be on the runner's state, not only on the outer one.
    runner.state.csv_dump = bool(getattr(state, "csv_dump", False))  # type: ignore[attr-defined]
    # The run's operator options (flags / environment variables), resolved once: a RunnerState built without
    # them (a direct ``run_orchestration`` call, a test) reads the environment here.  Both states carry the same
    # object so every reader of the cascade sees one resolution.
    run_options = _run_options_or_env(getattr(state, "run_options", None))
    state.run_options = run_options
    runner.state.run_options = run_options
    runner.state.logger.setLevel(logging.ERROR)
    # Forward the opt-in memory recorder (no-op when env var unset) so
    # ``_PolarHighCascadeSolver.run`` can fire the first-iter checkpoints.
    runner.state._memory_recorder = getattr(  # type: ignore[attr-defined]
        state, "_memory_recorder", _NoopMemoryRecorder()
    )

    # Δ.12c — build a SpineDbReader once and reuse it across the cascade.
    # The override chain fires for every per-solve load through it.  A
    # reader that cannot be built stops the run: without it every load
    # would silently run without the DB overrides and build a wrong LP.
    from flextool.engine.input._spinedb_reader import SpineDbReader
    # Phase 4.6 — thread axis_enums + contract from the cascade
    # provider if available so the reader casts on emit.
    _cip_for_reader = getattr(state, "cascade_input_provider", None)
    _cascade_axis_enums = getattr(_cip_for_reader, "axis_enums", None) \
        if _cip_for_reader is not None else None
    _cascade_contract = getattr(_cip_for_reader, "contract", None) \
        if _cip_for_reader is not None else None
    try:
        cascade_db_reader = SpineDbReader(
            db_url, scenario=scenario_name,
            axis_enums=_cascade_axis_enums,
            contract=_cascade_contract,
        )
    except Exception as exc:  # noqa: BLE001 — any failure is re-raised with context
        raise FlexToolConfigError(
            f"Could not open the input database {db_url} for scenario {scenario_name!r}: "
            f"{type(exc).__name__}: {exc}. Check that the database file exists, is not locked "
            "by another program, is migrated to the current FlexTool version, and contains "
            "this scenario."
        ) from exc
    if _drive_rec is not None:
        _drive_rec.checkpoint(
            "cascade_spinedb_reader_constructed", _drive_logger,
            user_label="Inputs prepared",
        )
    # The run's existing cohorts are built from this reader once the run calendar is known (``_run_model``).
    runner.state.input_source = cascade_db_reader
    # The solves of a 2-D period map are solves of the reader too (before any reader-side lookup).
    from flextool.engine.input._solve_config import register_solve_copies
    register_solve_copies(runner.state.solve, cascade_db_reader)
    # One dedupe set for the solve-step aggregation warnings of the run: the CSV frames
    # (TimelineConfig.create_averaged_timeseries, once per complete solve) and the DB-direct readers
    # (load_flextool arms the reader's memo) warn once per run between them.
    _timeline_cfg = getattr(runner.state, "timeline", None)
    if _timeline_cfg is not None:
        _timeline_cfg.solve_step_notes = cascade_db_reader.run_memo(("solve_step_aggregation_notes",), set)

    # ``input/p_all_entity_unitsize`` — solve-invariant entity-unitsize
    # cascade (virtual_unitsize OR existing OR 1000.0) over ALL entities
    # (unit ∪ node ∪ connection).  Computed ONCE here against the
    # whole-model (unfiltered-scenario) ``cascade_db_reader`` and seeded
    # into the cascade-input Provider, which every per-sub-solve Provider
    # copies frame-by-frame (see ``_run_model`` seed loop).  This
    # (a) avoids the per-roll recompute in ``apply_derived_b`` and
    # (b) gives the output reader a complete carrier covering every
    # entity — including invest candidates absent from the LAST solve's
    # pss/invest sets — so ``read_parameters`` no longer KeyErrors on
    # such entities.  ``_cip`` is the same Provider object forwarded onto
    # ``runner.state.cascade_input_provider`` above and read back by the
    # cascade as ``cascade_input_provider``.  In axis-Enum mode the
    # cascade vocabulary is made live for the call, so the per-class
    # entity frames land on the one entity-union ``e`` Enum (the per-solve
    # loads set it themselves, this call runs before any of them).
    if _cip is not None:
        from flextool.engine.input._axis_enums import (
            reset_global_axis_enums,
            set_global_axis_enums,
        )
        from flextool.engine.preprocess.params._derived._unitsize import _entity_unitsize_lf
        _us_enums_token = (set_global_axis_enums(_cascade_axis_enums)
                           if _cascade_axis_enums is not None else None)
        try:
            _all_us_df = (
                _entity_unitsize_lf(cascade_db_reader)
                .rename({"us": "value"})
                .collect()
            )
        finally:
            if _us_enums_token is not None:
                reset_global_axis_enums(_us_enums_token)
        if _all_us_df.height > 0:
            _cip.put("input/p_all_entity_unitsize", _all_us_df)

    # W4b — register the engine-automatic realized-dispatch sibling for
    # every eligible ``separate_dispatch`` top-level solve, BEFORE
    # ``native_run_model`` expands the solve tree.  Dual in-memory
    # registration onto the DRIVING config (``runner.state.solve``) and
    # the cascade reader so the sibling takes the full override path.
    # No-op (and byte-parity) when no solve authors separate_dispatch.
    _spawn_realized_dispatch_siblings(
        runner.state.solve, cascade_db_reader, _drive_logger,
        timeline_cfg=runner.state.timeline, cascade_solve_cfg=state.solve,
    )

    class _PolarHighCascadeSolver(SolverRunner):
        def __init__(self, runner_state):
            super().__init__(runner_state)
            self._all_steps: dict[str, OrchestrationStep] = results
            # Warm-LP carry-over state.  ``_warm_problem`` holds the live
            # :class:`polar_high.WarmProblem` of the last in-process build;
            # ``_prior_data`` / ``_prior_fp`` / ``_prior_solver_key`` hold
            # that sub-solve's FlexData, fingerprint and solver
            # configuration, against which the next sub-solve is compared
            # (reuse only when all are identical; :mod:`._warm`).  All stay
            # None when ``warm=False``; :meth:`_release_warm_problem` drops
            # them before every LP that is not a reuse (fresh warm build,
            # cold / subprocess build, Benders solve).
            self._warm_problem: "WarmProblem | None" = None
            # ``model.output_horizon`` with a Benders solve: the console
            # notice that it is not dumped is printed once per run.
            self._benders_dump_notice_printed = False
            self._prior_data = None
            self._prior_fp: "tuple | None" = None
            self._prior_solver_key: "tuple | None" = None
            # Per-iter slim of the PRIOR step's parked Solution — see the
            # block just before ``self._all_steps[step_key] = ...`` in
            # :meth:`run`.  Tracks the step_key parked on the previous
            # iter so we can null its heavy ``_vars`` + ``highs`` once the
            # per-iter writers and ``build_handoff_from_solution`` have
            # finished consuming it.  Bounds peak RSS during the cascade
            # — without this, every iter's full ``Var.frame`` dataframe
            # set stays parked until the post-loop slim at the bottom of
            # :func:`_run_model`, which on multi-roll runs is too
            # late (storage→dispatch OOMs).
            self._prev_step_key: "str | None" = None
            # Phase 2 — per-step level_key sidecar.  Populated when
            # parking each step so the warm-path "keep one
            # ``Solution.highs`` + one ``flex_data_provider`` per level"
            # slim can iterate prior steps and resolve their level.
            # Keyed by the same ``step_key`` used in ``self._all_steps``.
            self._step_level_keys: "dict[str, tuple]" = {}
            # Per-base-solve gating for the scaling CSV.  The CSV value
            # (effective_obj_scale) is invariant across rolls of the same
            # base solve, so we track which base solve names already have
            # ``scale_the_objective.csv`` written and skip subsequent
            # rolls.  Phase 2b dropped the legacy diagnostic TXT report
            # (``FLEXTOOL_SCALING_REPORT=1``); the autoscaler's per-solve
            # YAML report (``solve_data/autoscale_<solve>.yaml``) is the
            # replacement and is gated inside
            # :func:`_autoscale_emit_layer1`.
            self._scale_csv_written: set[str] = set()
            # Autoscale console summary dedup: one line per base solve.
            # Layer 1/2/3 decisions are identical across rolls of the
            # same base solve, so the operator-facing summary fires once.
            self._autoscale_summary_emitted: set[str] = set()
            # Warm-path autoscale plan cache: Layer 2 writes side
            # vectors on the Problem at first-build; the WarmProblem's
            # canonical matrix bakes them in.  A reuse solves exactly the
            # LP it was built for (identical sub-solve, nothing pushed), so
            # the plan stays exact.  It is needed on every warm solve so
            # :func:`_autoscale_unscale_post_solve` can restore the
            # solution to physical coordinates.  Cleared whenever
            # ``self._warm_problem`` is dropped.
            self._autoscale_warm_layer2_plan: "_AutoscaleLayer2Plan | None" = None
            # Cache of the pre-Layer-2 RangeReport from the warm first
            # build, surfaced as ``Solution.streamed_lp_ranges`` after
            # every warm solve so downstream telemetry (the LP-bound-
            # range smoke in ``test_invest_chain_regression``, the
            # autoscale Layer-1 YAML) sees the same four (min, max)
            # pairs the autoscaler decided on.  Mirrors the cold path's
            # post-``run_one_solve`` ``sol.streamed_lp_ranges = …``
            # assignment (see the longer comment around the cold-path
            # write below); dropped together with the warm problem.
            self._autoscale_warm_ranges_pre: (
                "_AutoscaleRangeReport | None"
            ) = None
            # Per-structural-shape autoscale DECISION cache.  Keyed by the
            # BUILT LP's structural signature
            # (:func:`_autoscale_lp_shape_signature` — matrix shape +
            # per-family layout, scoped by base solve name), which stays
            # invariant across rolls of a rolling solve where
            # ``_fingerprint(data)`` would slide; each
            # value is an :class:`_AutoscaleShapeCacheEntry` carrying the
            # Layer-2 exponents, the Layer-3 plan, and the pre/post
            # RangeReports computed on the FIRST solve of that shape.  On
            # every subsequent same-shape solve (notably the COLD-rebuild-
            # per-roll path where ladder Params force a cold rebuild yet
            # the matrix shape is invariant) the cached decision is
            # re-applied via :func:`_autoscale_apply_layer2_from_cache` /
            # :func:`_autoscale_apply_layer3_from_cache` WITHOUT any
            # ``detect_ranges`` / ``bucket_coefficients`` traversal — that
            # traversal was the source of the per-roll multi-GB transient
            # peaks (the autoscale memory pyramid).  Disable with
            # ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1`` (always recompute).
            self._autoscale_shape_cache: (
                "dict[tuple, _AutoscaleShapeCacheEntry]"
            ) = {}
            # Phase 4 Step 4b — in-process warm-start basis cache.  Keyed
            # by the built LP's name-set fingerprint
            # (``Problem.basis_name_fingerprint``); each value is a
            # :class:`polar_high.NamedBasis` captured off a fresh build's
            # optimal solve and re-injected into an identically-named fresh
            # build on a later cross-run solve (UC2 sweep / UC3 resume).
            # Populated only under ``FLEXTOOL_WARM_START=1`` on the
            # in-process HiGHS fresh-build path and mirrored on disk as
            # ``<cache_dir>/<fp>.nbasis`` (JSON) alongside the subprocess
            # arm's ``<fp>.bas``.  Stays empty + untouched when the opt-in
            # is off (byte-identical off-path).
            self._inproc_basis_cache: "dict[str, NamedBasis]" = {}
            # v60/v62 per-solve decomposition — complete-solve names that
            # ran under ``decomposition=benders``.  Used by the consume-side
            # guard in :meth:`run` to raise loudly if a downstream solve
            # would try to consume a Benders solve's (absent) handoff —
            # cross-scheme handoff is a deferred follow-up.
            self._benders_solve_names: set[str] = set()
            # TIER 1 — complete-solve names whose Benders solve deposited a
            # USABLE investment handoff (non-empty
            # ``result.invest_solution_vars`` → ``realized_cohort`` /
            # ``divest_cumulative`` carriers).  A downstream solve may
            # consume these; the consume-side guard fires only for Benders
            # predecessors NOT in this set.
            self._benders_invest_handoff_names: set[str] = set()
            # The BendersResult of the Benders solve just run (``_run_benders_solve``), read by its final solve.
            self._benders_result = None

        def _release_warm_problem(self) -> None:
            """Drop the previous in-process WarmProblem and everything tied
            to it (Layer-2 plan, cached ranges, the prior sub-solve's data,
            fingerprint and solver key).  Called before every LP that is not
            a reuse of it: a fresh warm build, a cold / subprocess build
            (non-HiGHS solver, save-memory) and a Benders solve.  Its HiGHS
            instance (``WarmProblem._h``) is not released by the parked
            ``Solution.highs`` loop, so keeping it would let two LPs coexist."""
            self._warm_problem = None
            self._autoscale_warm_layer2_plan = None
            self._autoscale_warm_ranges_pre = None
            self._prior_data = None
            self._prior_fp = None
            self._prior_solver_key = None

        def _dump_solve_inputs(self, data, complete_solve_name: str) -> None:
            """``--csv-dump``: write *data* (the sub-solve's FlexData) to ``input/`` + ``solve_data/`` in the work
            folder; a no-op without the flag.  Called by the monolithic and the Benders sub-solve alike.

            A failed debug dump does not stop the run (the LP and the outputs do not depend on it), but it is
            logged at ERROR: the runner's logger is pinned to ERROR in ``_drive_cascade``, so a warning would
            never be seen."""
            if not getattr(self.state, "csv_dump", False):
                return
            try:
                data.dump_csvs(self.state.paths.work_folder)
            except Exception as exc:  # noqa: BLE001
                self.state.logger.error(
                    f"dump_csvs failed for {complete_solve_name}: {exc!r}"
                )

        def _run_benders_solve(
            self,
            complete_solve_name: str,
            base_solve_name: str,
            data,
            *,
            scaling_config: "ScalingConfig",
            final_solve: bool = False,
        ) -> int:
            """Run *complete_solve_name* via the Benders region driver.

            Selected per solve from ``solve.decomposition = benders``.
            Decomposes over the groups whose
            ``group.decomposition_method`` is ``benders_regional`` (the
            ``decomp_<REG>`` groups the PLEXOS-to-FlexTool writer emits),
            using the per-solve knobs resolved by
            :meth:`SolveConfig.benders_config_for`.

            The solve runs and reports convergence/objective + the valid
            LB/UB sandwich, then (TIER 1) assembles its owner-selected
            whole-system invest/divest decisions
            (``result.invest_solution_vars``) into a :class:`SolveHandoff`
            and deposits it into ``state.handoffs`` so a DOWNSTREAM
            rolling-dispatch solve consumes the invested capacity.  When
            the model has no investment the dict is empty, no handoff is
            built, and the deposited step carries a
            :class:`SnapshotSolution` with empty ``_vars``.  Every Benders
            solve is recorded in ``self._benders_solve_names``; those that
            handed forward a usable invest handoff are also recorded in
            ``self._benders_invest_handoff_names`` so the (now narrowed)
            consume-side guard in :meth:`run` fires only for a downstream
            consumer of a Benders solve that produced no investment.

            *final_solve*: the final solve (item 287) follows in :meth:`_run_one`, which then dumps the inputs
            (``--csv-dump``) once.  The :class:`BendersResult` is kept on ``self._benders_result`` for it.
            """
            from flextool.engine.solve._benders import solve_benders
            from flextool.decomposition.region_filter import (
                discover_decomposition_regions_from_db,
            )

            # Slice C Guard 2 (primary): Benders x genuine stochastics is
            # not yet supported (the master's C_by_conn sum over all invest
            # periods would double-count exclusive branch futures — recourse
            # plan §4).  Flag-independent: it closes the today-untested
            # Benders x stochastics combination regardless of
            # invest_solve_branching_method.  The detector reads two FlexData
            # frames already in scope on ``data``.
            if _is_genuinely_stochastic(data):
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' requests decomposition=benders "
                    f"on a stochastic model (period__branch has non-realized "
                    f"branch periods with active time). Benders x stochastics "
                    f"is not yet supported (the master's C_by_conn sum over "
                    f"all invest periods would double-count exclusive branch "
                    f"futures — recourse plan §4). Run this solve "
                    f"monolithically (decomposition=none) or remove the "
                    f"stochastic branches."
                )

            # Regions are discovered from the DB (the same source the
            # group-level decomposition_method lives in).  The DB-driven
            # run path (run_chain_from_db) always supplies ``db_url``; a
            # caller that bypasses it cannot resolve the region groups, so
            # fail with an actionable message rather than guess.
            if db_url is None:
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' requests "
                    f"decomposition=benders but the run was started "
                    f"without a database URL (db_url is None). Benders "
                    f"region decomposition is only available on the "
                    f"DB-driven run path (run_chain_from_db)."
                )
            regions = discover_decomposition_regions_from_db(db_url)
            if len(regions) < 2:
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' requests "
                    f"decomposition=benders but the model declares "
                    f"{len(regions)} group(s) with "
                    f"decomposition_method='benders_regional' "
                    f"({regions or '(none)'}); at least two region groups "
                    f"are required."
                )

            max_iter, tol, in_out_weight_db = state.solve.benders_config_for(
                base_solve_name
            )
            # Resolve the objective scale the SAME way the monolithic path
            # does (the solve's ``scale_the_objective``; malformed/unset =>
            # 1e-6).  Both the Benders master and the region subproblems
            # build at this scale so the cut coefficients stay consistent.
            _user_obj_scale = state.solve.scale_the_objective.get(
                complete_solve_name
            ) or state.solve.scale_the_objective.get(base_solve_name)
            obj_scale = _resolve_effective_obj_scale(_user_obj_scale)
            _emit_provider_obj_scale(self.state, obj_scale)

            # Benders progress is surfaced via ``print`` (flushed), the
            # same channel as the cascade's "Solve start" markers — the
            # per-solve logger is pinned to ERROR in ``_drive_cascade``, so
            # logger.info/.warning would be swallowed.
            _tag = f"[benders {base_solve_name}]"

            # ``_emit`` is called from the main thread (iteration / summary
            # lines) AND from polar-high worker threads (the per-region
            # ``subsolve_callback`` fires from the parallel region pass), so
            # the lock keeps lines from interleaving mid-string.
            _emit_lock = threading.Lock()

            def _emit(line: str) -> None:
                with _emit_lock:
                    try:
                        print(line, flush=True)
                    except OSError:
                        pass

            # Put the config INSIDE the tag so it stays visible even though the
            # (long, non-wrapping) region list pushes the rest off-screen.
            _emit(
                f"[benders {base_solve_name}, max_iter={max_iter}, "
                f"tol={tol:g}, obj_scale={obj_scale:g}] "
                f"start: {len(regions)} regions {regions}"
            )

            # MASTER-HOSTED node announcement (plan C8 / risk R10 tripwire):
            # balance/state nodes in NO region group are hosted natively in
            # the Benders MASTER, and that re-partition must never happen
            # silently — the driver's _logger.info is swallowed here (the
            # cascade pins per-solve loggers to ERROR), so surface the list
            # on the same print channel as the other Benders lines.  With
            # every node grouped the set is empty and NOTHING extra is
            # emitted (and the solve itself takes today's exact path — this
            # announcement adds no solves either way, it is derived from
            # the already-loaded input frames).
            from flextool.engine.solve._region_filter import (
                compute_master_hosted_nodes,
                load_region_membership,
                master_arc_counts,
            )

            _membership = load_region_membership(data, regions)
            _master_hosted = sorted(compute_master_hosted_nodes(data, _membership))
            if _master_hosted:
                _emit(
                    f"{_tag} Benders: {len(_master_hosted)} master-hosted "
                    f"node(s): {_master_hosted}"
                )
            # What the master holds: the connections between regions (and at master-hosted nodes) and the units
            # with nodes in more than one region (or in a region and the master), by their node relationships.
            _n_conn, _n_in, _n_out = master_arc_counts(data, regions, _membership)

            def _count(n: int, what: str) -> str:
                return f"{n} {what}{'' if n == 1 else 's'}"

            _emit(f"{_tag} Benders master: {_count(_n_conn, 'connection')}, "
                  f"{_count(_n_in, 'unit__inputNode arc')}, {_count(_n_out, 'unit__outputNode arc')}")

            # Live per-iteration callback — one line as each outer Benders
            # iteration completes, reporting the valid LB/UB sandwich + gap
            # (all in REAL units, ÷s).
            def _on_iteration(entry: dict) -> None:
                _emit(
                    f"{_tag}   iter {entry['iter']:>3}/{max_iter}: "
                    f"LB={entry['lower_bound']:.6g}  "
                    f"UB={entry['best_upper_bound']:.6g}  "
                    f"gap={entry['gap']:.4g}"
                )
                run_record.active().benders_progress(entry["iter"], entry["gap"])

            # Per-region sub-solve FINISH line — replaces the raw per-sub-solve
            # HiGHS banner with a useful "which region finished" log.  Fires
            # from polar-high worker threads (parallel region pass); ``_emit``
            # is lock-guarded.  (``iter`` 0 is the autarkic bootstrap pass.)
            def _on_subsolve(entry: dict) -> None:
                _emit(
                    f"{_tag}   iter {entry['iter']:>3}: "
                    f"{entry['region']} done (obj={entry['obj']:.6g})"
                )

            # The user layers of the solve's HiGHS options (highs.opt <
            # solver_mip_gap < solver_arguments < CLI flags).  No engine
            # baseline: Benders keeps its own validated pins.  A whole-model
            # objective target / bound is refused here, before any build.
            _benders_options = _solve_highs_options(
                state, complete_solve_name, baseline=None,
            )
            _check_benders_objective_options(
                _benders_options,
                solve_name=base_solve_name,
                layers=_highs_option_layers(state, complete_solve_name, _benders_options),
            )
            # Benders master and region solves run on HiGHS (polar-high).
            run_record.active().note_solver("highs")
            result = solve_benders(
                data,
                regions,
                max_iters=max_iter,
                tol=tol,
                monolith_objective=None,
                scale_the_objective=obj_scale,
                in_out_weight=in_out_weight_db,
                progress_callback=_on_iteration,
                subsolve_callback=_on_subsolve,
                solver_options=_benders_options,
                scaling_config=scaling_config,
                # ``--benders-workers`` / ``FLEXTOOL_BENDERS_WORKERS`` and
                # ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT``, resolved once for the run
                # (also ``--scaling-guard`` for the region subproblems).
                run_options=run_options,
                solve_name=complete_solve_name,
            )
            # The run record: converged / not converged with a feasible
            # incumbent (the run goes on) / no incumbent (the run fails).
            run_record.active().solve_benders(
                converged=bool(result.converged),
                has_incumbent=(
                    result.total_objective is not None
                    and math.isfinite(result.total_objective)
                ),
                objective=result.total_objective,
                gap=result.gap,
                tol=result.tol,
                iterations=result.iterations,
            )

            # Final summary — convergence + the valid LB/UB sandwich that
            # is the headline benefit of Benders over the subgradient path
            # (a certified lower bound, not just a dual estimate).
            _status = "CONVERGED" if result.converged else "DID NOT CONVERGE"
            _lb = result.lower_bound
            _ub = result.upper_bound
            _emit(
                f"{_tag} {_status} after {result.iterations}/{max_iter} "
                f"iterations"
            )
            _emit(f"{_tag}   lower bound (valid) = {_lb:.6g}")
            _emit(f"{_tag}   upper bound (best)  = {_ub:.6g}")
            _emit(
                f"{_tag}   optimality gap      = {result.gap:.4g} "
                f"(relative, (UB-LB)/|UB|)"
            )
            if result.region_costs:
                _rc = ", ".join(
                    f"{r}={c:.6g}"
                    for r, c in result.region_costs.items()
                )
                _emit(f"{_tag}   region costs: {_rc}")
            if not result.converged:
                # Also log at ERROR so non-convergence still surfaces in
                # error-only contexts (CI, log scrapers); the chain is not
                # aborted — the decomposition produced a usable point.
                self.state.logger.error(
                    "Solve '%s': Benders decomposition did NOT converge "
                    "after %d/%d iterations (gap %.4g).",
                    base_solve_name, result.iterations, max_iter, result.gap,
                )

            # TIER 1 — turn the assembled, owner-selected whole-system
            # invest/divest frames into a handoff so a DOWNSTREAM rolling-
            # dispatch solve consumes this Benders solve's invested
            # capacity.  ``result.invest_solution_vars`` is a dict keyed by
            # ``v_invest_p`` / ``v_invest_n`` / ``v_divest_p`` /
            # ``v_divest_n``, the built retirements ``v_divest_built_{p,n}``
            # (per cohort) / ``v_divested_built_{p,n}`` and the life extensions
            # ``v_extend_{p,n}`` (per cohort) / ``v_extended_{p,n}`` to long-form frames
            # whose columns match ``polar_high.Solution.value(name)`` exactly;
            # it is ``{}`` when the model has no investment.
            # ``--csv-dump``: dump the solve's undecomposed FlexData (the master and the regions are derived from it);
            # the final solve, when it follows, dumps the same data.
            if not final_solve:
                self._dump_solve_inputs(data, complete_solve_name)
            self._benders_result = result

            from flextool.engine.preprocess.flex_data import (
                build_handoff_from_solution,
            )

            self._benders_solve_names.add(complete_solve_name)

            # Carrier: a SnapshotSolution duck-types the ``_vars``
            # membership + ``.value(name)`` contract the invest/divest
            # extraction in ``build_handoff_from_solution`` needs (it
            # reads ONLY ``name in sol._vars`` and ``sol.value(name)`` on
            # that path — no Var internals, no ``col_value``).
            snap = SnapshotSolution(_vars=dict(result.invest_solution_vars))

            handoff = None
            if result.invest_solution_vars:
                # Mirror the monolithic path's prior-handoff resolution
                # (the predecessor this solve loaded), so the cumulative
                # invest/existing carriers chain forward correctly.
                _prior = (
                    self.state.handoffs.get(self.state.last_captured_solve)
                    if self.state.last_captured_solve is not None else None
                )
                # CRITICAL: pass ``flex_data=None``.  With a real
                # ``flex_data`` the builder also runs the co2 /
                # cumulative-commodity / fix_storage branches, the last of
                # which is gated on storage nodes (not on ``v_flow in
                # _vars``) and calls ``sol.constraint_dual(...)`` —
                # absent on SnapshotSolution → AttributeError.  TIER 1
                # needs none of those derivations from the invest solve;
                # ``flex_data=None`` collapses them to prior-handoff
                # passthroughs and keeps the call on the pure invest/divest
                # path.  ``provider`` is safe (it only feeds the
                # ``solve_data/`` CSV reads that source the invest periods
                # / entity sets, independent of ``sol``) and is REQUIRED
                # for the carriers to populate.
                handoff = build_handoff_from_solution(
                    snap,
                    self.state.paths.work_folder,
                    complete_solve_name,
                    prior_handoff=_prior,
                    flex_data=None,
                    parent_handoff=None,
                    provider=getattr(
                        self.state, "current_provider", None,
                    ),
                    scale_the_objective=obj_scale,
                    # The undecomposed solve's cohort table: the realized
                    # investments are carried as its cohorts.
                    cohort=getattr(data, "cohort", None),
                )
                # Deposit so the next solve's load picks it up via the
                # ``last_captured_solve`` translation in _run_model
                # (which re-reads ``state.handoffs[last_captured_solve]``
                # after this method returns 0).
                self.state.handoffs[complete_solve_name] = handoff
                # Track that this Benders solve handed forward usable
                # investment so the narrowed consume-side guard does NOT
                # fire for a downstream consumer of it.
                self._benders_invest_handoff_names.add(complete_solve_name)

            self._all_steps[complete_solve_name] = OrchestrationStep(
                solve_name=complete_solve_name,
                solution=snap,
                handoff=handoff,
                obj=result.total_objective,
                optimal=result.converged,
                warm_used=False,
                is_benders=True,
                benders_gap=result.gap,
                benders_tol=result.tol,
                benders_iterations=result.iterations,
                # "invest_slice" once the investment-only results are written (final solve off or failed).
                benders_results="none",
                flex_data=data,
                flex_data_provider=getattr(
                    self.state, "current_provider", None,
                ),
            )
            self._prev_step_key = complete_solve_name
            return 0

        def _benders_final_not_accepted(self, complete_solve_name: str, *, problem: str, details: str | None,
                                        recorded: bool, bres, data) -> int:
            """The final solve of Benders solve *complete_solve_name* gave no accepted solution: W-0572, and the
            Benders step, its handoff and its run-record verdict stay (``benders.final_solve`` says it failed).
            The run goes on (owner answer 287-Q6): the Benders result itself is valid.  With a later solve the
            Benders solve writes its investment-only results (:meth:`_write_benders_invest_slice`); as the last
            solve it writes none (the result tables need a dispatch, which no solve of the run then has)."""
            if not recorded:
                run_record.active().solve_benders_final(accepted=False, model_status=None, objective=None)
            run_warnings.warn("W-0572", solve=complete_solve_name, problem=problem, details=details)
            # The Benders step stays the solve's step (the final solve's step would have replaced it): the next
            # solve's cold-path slim releases its input data.
            self._prev_step_key = complete_solve_name
            if self._later_solve_follows():
                self._write_benders_invest_slice(complete_solve_name, bres, data)
            return 0

        def _benders_final_out_of_memory(self, complete_solve_name: str, exc: BaseException, *, bres,
                                         data) -> int:
            """Building or writing the whole model of the final solve of Benders solve *complete_solve_name* ran
            out of memory: W-0572 (it names ``--benders-final-solve off``), the Benders result stands
            (:meth:`_benders_final_not_accepted`)."""
            import gc

            gc.collect()
            _try_malloc_trim()
            return self._benders_final_not_accepted(
                complete_solve_name, problem="failed (out of memory; see --benders-final-solve off)",
                details=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
                recorded=False, bres=bres, data=data,
            )

        def _later_solve_follows(self) -> bool:
            """Whether a solve of the run comes after the one running (``_run_model`` sets the flag per solve;
            unknown → False)."""
            _window = getattr(self.state, "current_committed_window_check", None)
            return bool(_window[2]) if _window is not None else False

        def _write_benders_invest_slice(self, complete_solve_name: str, bres, data) -> None:
            """The investment-only results of Benders solve *complete_solve_name* (no final solve result):
            its investment decisions, their costs and its objective (the upper bound) go to ``output_raw/``
            (:func:`~flextool.engine.output._benders_slice.write_benders_invest_slice`); its step gets
            ``benders_results = "invest_slice"``.  A failed write is logged and leaves the step at ``"none"``."""
            from flextool.engine.output._benders_slice import write_benders_invest_slice

            _user = (state.solve.scale_the_objective.get(complete_solve_name)
                     or state.solve.scale_the_objective.get(re.sub(r"_roll_\d+$", "", complete_solve_name)))
            try:
                write_benders_invest_slice(
                    bres.invest_solution_vars, bres.total_objective,
                    work_folder=self.state.paths.work_folder, solve_name=complete_solve_name,
                    flex_data=data, provider=getattr(self.state, "current_provider", None),
                    scale_the_objective=_resolve_effective_obj_scale(_user),
                )
            except Exception as exc:  # noqa: BLE001 - as the monolithic writer: the run goes on, loudly
                self.state.logger.error(
                    f"writing the investment results of Benders solve {complete_solve_name} failed: {exc!r}")
                return
            step = self._all_steps.get(complete_solve_name)
            if step is not None:
                step.benders_results = "invest_slice"

        def check_run_inputs(self, steps_of_run) -> None:
            """Before the first solve: a group limit's time map covers every time step of the run's solves
            (:func:`~flextool.engine.preprocess.params._direct.check_group_limit_time_coverage`, read through the
            run's reader, in the cascade's axis vocabulary like the per-solve loads).  A solve whose
            ``reduce_system_detail_method`` leaves a limit out is not checked for it."""
            from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums
            from flextool.engine.preprocess._reduce_detail import system_parts
            from flextool.engine.preprocess.params._direct import check_group_limit_time_coverage

            def _reduced(solve: str) -> frozenset:
                return system_parts(self.state.solve.reduce_system_detail_for(solve))

            token = (set_global_axis_enums(_cascade_axis_enums) if _cascade_axis_enums is not None else None)
            try:
                check_group_limit_time_coverage(cascade_db_reader, steps_of_run, _reduced)
            finally:
                if token is not None:
                    reset_global_axis_enums(token)

        def run(self, complete_solve_name: str) -> int:
            """One sub-solve (a roll, or a whole solve), with its coded warnings gathered: each code prints one
            line for the sub-solve, before the solver runs (:func:`_print_solver_args`) or when it ends; the
            solve-scoped codes (``solve=``) print once per complete solve."""
            with run_warnings.collecting():
                status = self._run_one(complete_solve_name)
            # The gathered solver-outcome note (W-0540: in how many rolls, the largest factor) of each complete
            # solve whose last sub-solve in the run was this one prints now, after its solver output.  Without the
            # run's order (a caller outside ``native_run_model``) the cascade's final flush prints it.
            last = getattr(self.state, "_last_sub_solve_index", None)
            index = getattr(self.state, "_current_iter_index", None)
            if last is not None and index is not None:
                from flextool.engine.solve._solver_outcome import still_running
                self.flush_solver_outcome_notes(keep=still_running(last, index + 1))
            return status

        def flush_solver_outcome_notes(self, keep: Collection[str] = frozenset()) -> None:
            """Print the W-0540 notes gathered over the rolls of each complete solve but *keep*
            (:func:`~flextool.engine.solve._solver_outcome.flush_solver_outcome_notes`)."""
            pending = self.state.__dict__.get("_solver_outcome_pending")
            if not pending:
                return
            from flextool.engine.solve._solver_outcome import flush_solver_outcome_notes
            try:
                flush_solver_outcome_notes(pending, self.state.__dict__.setdefault("_solver_outcome_noted", set()),
                                           keep=keep)
            except Exception:  # noqa: BLE001 - diagnostic only
                logging.getLogger(__name__).debug("the solver-outcome notes could not be printed", exc_info=True)

        def _run_one(self, complete_solve_name: str) -> int:
            _phase_prof("run_enter")
            # Authoritative: refuse an unrecognised invest_solve_branching_method for any solve that RUNS —
            # including each roll of a rolling solve and each contained/chained solve — before building its
            # FlexData/LP.  The roll-suffix strip mirrors the base-name derivation below.
            _guard_base = re.sub(r"_roll_\d+$", "", complete_solve_name)
            _invest_branching_raw_guard(state.solve, _guard_base)
            # W4 Slice 1 Guard (authoritative): reject an unrecognised
            # dispatch_realization_method, or an unsupported
            # 'separate_dispatch', for any solve that RUNS — including each
            # roll of a rolling solve and each contained/chained solve —
            # before building its FlexData/LP.  Omitting this site would let
            # a contained/chained solve authoring 'separate_dispatch' run
            # without its realization solve.
            _dispatch_realization_raw_guard(state.solve, _guard_base)
            # Cross-level eviction — release any EXHAUSTED prior solve-level's
            # live HiGHS instance + flex_data_provider BEFORE this solve builds
            # its FlexData/LP, so two level footprints never coexist (the DES
            # storage+dispatch ≈ 2x peak that drove the 7/9 near-OOM).  This
            # hoists the post-solve slim's exhausted-level branch
            # (``:2632-2647``) ahead of the allocation instead of running it
            # after — the prior level's per-iter writers + handoff already
            # consumed its solution on its own iter, so releasing here is safe.
            # A level is "exhausted" when no upcoming iter shares its level_key
            # and it is not the level THIS solve belongs to; same-level steps
            # are kept for warm reuse.  ``self._warm_problem`` for an exhausted
            # level was already nulled at the level boundary
            # (``_run_model._native_run_model.native_run_model``, the per-level boundary), so nulling the step's
            # ``solution.highs`` here drops the last reference and frees it.
            # malloc_trim reclaims the HiGHS (glibc) heap; the polars-side
            # provider is freed by dropping the Python ref.
            if not keep_solutions:
                _ilk = getattr(self.state, "_all_level_keys", ())
                _iidx = getattr(self.state, "_current_iter_index", None)
                _tlvl = getattr(self.state, "_current_level_key", None)
                _upcoming = (
                    set(_ilk[_iidx + 1:])
                    if (_iidx is not None and _ilk) else set()
                )
                _released = False
                for _k, _step in (getattr(self, "_all_steps", None) or {}).items():
                    _lvl = self._step_level_keys.get(_k)
                    if _lvl is None or _lvl == _tlvl or _lvl in _upcoming:
                        continue  # current level or still-upcoming: keep
                    _sol = getattr(_step, "solution", None)
                    if _sol is not None and getattr(_sol, "highs", None) is not None:
                        _sol.highs = None
                        _released = True
                    if getattr(_step, "flex_data_provider", None) is not None:
                        _step.flex_data_provider = None
                        _released = True
                    # Also evict the exhausted level's entry from the
                    # per-level FlexDataProvider cache — it is keyed by
                    # level_key and only reused by FUTURE same-level rolls,
                    # of which an exhausted level has none.  Without this
                    # the cache (``state._level_providers``) pins the
                    # level's polars FlexData for the whole cascade even
                    # after the step ref above is dropped.
                    _lp = getattr(self.state, "_level_providers", None)
                    if isinstance(_lp, dict) and _lp.pop(_lvl, None) is not None:
                        _released = True
                if _released:
                    _try_malloc_trim()
                # Cross-level retention audit — runs AFTER the eviction above so
                # a correct release records zero violators.  Gated by the same
                # ``not keep_solutions`` as the eviction: the release invariant
                # only applies when slimming is active (``keep_solutions=True``
                # deliberately retains every level's solution).
                _audit_prior_level_release(
                    steps=getattr(self, "_all_steps", None),
                    step_level_keys=getattr(self, "_step_level_keys", {}),
                    all_level_keys=getattr(self.state, "_all_level_keys", ()),
                    iter_idx=getattr(self.state, "_current_iter_index", None),
                    this_level=getattr(self.state, "_current_level_key", None),
                    complete_solve_name=complete_solve_name,
                )
            # Optional per-iter phase-timing (opt-in via env var).  Emits
            # `per_iter` rows to the workdir's timings.csv covering
            # lp_build / solve / handoff and a warm_used marker.  See
            # specs/warm_start_phase_breakdown_handoff.md.
            _phase_timing = (
                os.environ.get("FLEXTOOL_PHASE_TIMING") == "1"
                and getattr(self.state, "timing_recorder", None) is not None
            )
            _tr = self.state.timing_recorder if _phase_timing else None
            _roll_idx = getattr(self.state, "current_roll_index", "")
            if _roll_idx is None:
                _roll_idx = ""
            _t_build_start = time.perf_counter() if _phase_timing else 0.0
            # Δ.12 — wire ``handoff=`` through ``load_flextool`` so the
            # in-memory carriers from the prior solve flow into this
            # solve's FlexData directly.  Replaces the previous
            # implicit dependency on flextool's per-solve preprocessing
            # rewriting ``solve_data/p_entity_*.csv`` between solves.
            # After Δ.12 the cascade reads these four carrier-derived
            # fields from the in-memory ``SolveHandoff`` rather than the
            # workdir CSVs:
            #
            #   * ``p_entity_invested``
            #   * ``p_entity_divested``
            #   * ``p_entity_previously_invested_capacity``
            #   * ``p_fix_storage_quantity``
            #
            # (``p_roll_continue_state`` comes from the routed
            # ``handoff/roll_end_state`` key, ``_carry_routing``.)
            prior_for_load = (
                self.state.handoffs.get(self.state.last_captured_solve)
                if self.state.last_captured_solve is not None else None
            )
            _sub_solve_provider = getattr(
                self.state, "current_provider", None,
            )
            # The solve's reduce_system_detail_method parts: a limit the solve leaves out need not cover its steps.
            from flextool.engine.preprocess._reduce_detail import system_parts as _system_parts
            from flextool.engine.preprocess.params._direct import reduced_system_parts
            with reduced_system_parts(_system_parts(
                    self.state.solve.reduce_system_detail_for(complete_solve_name))):
                data = load_flextool(
                    self.state.paths.work_folder,
                    handoff=prior_for_load,
                    db_reader=cascade_db_reader,
                    provider=_sub_solve_provider,
                )
            # Release heap held by the broadcast cascade scratch frames.
            # On H2_trade y2050 this drops RSS ~1.6 GB / 41 %; expected
            # to scale with timeline size.  No-op on non-glibc.
            _try_malloc_trim()
            # Memory checkpoint — fires on level-boundary iters (the
            # last roll of a roll group) so the recorded delta aggregates
            # across all rolls in the group.  ``_run_model`` sets
            # the flag before each ``solver.run()`` call.
            _emit_phase = bool(getattr(
                self.state, "emit_phase_checkpoints_this_iter", False,
            ))
            _memrec_local = getattr(self.state, "_memory_recorder", None)
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "load_flextool_end", self.state.logger,
                    user_label="FlexData built",
                )

            # Solve-level detail reduction (solve.reduce_unit_detail_method /
            # reduce_system_detail_method): every consumer below — the
            # realization checks, Benders, the cold / warm build, the handoff
            # and the outputs — reads the reduced data.  A roll and a
            # realization solve take their base / parent solve's levels.
            from flextool.engine.preprocess._reduce_detail import (
                apply_reduce_detail,
                cold_start_online_flow_carry,
                online_gone,
                output_static_view,
            )
            from flextool.engine.preprocess.params._ramp_links import check_online_ramp_carry
            _unit_level = self.state.solve.reduce_unit_detail_for(complete_solve_name)
            _unreduced = data
            data, _reduce_report = apply_reduce_detail(
                data,
                _unit_level,
                self.state.solve.reduce_system_detail_for(complete_solve_name),
            )
            # Obligations carried in from an earlier solve (minimum up-time, realization pins) yield to an
            # outage that solve did not see (W-0592), judged on the reduced data and after the coarse frames.
            from flextool.engine.preprocess.flex_data import carried_obligations_yield_to_outages
            carried_obligations_yield_to_outages(
                data, _roll_label(complete_solve_name, getattr(self.state, "current_roll_index", None)),
                _unit_level)
            _reduce_base = re.sub(r"_roll_\d+$", "", complete_solve_name)
            if _reduce_report.warnings:
                _warned = self.__dict__.setdefault("_reduce_detail_warned", set())
                # One warning per chain: a realization solve repeats its parent's.
                _warn_key = (getattr(self.state.solve, "realized_dispatch_parent", {}) or {}).get(
                    _reduce_base, _reduce_base)
                if _warn_key not in _warned:
                    _warned.add(_warn_key)
                    for _line in _reduce_report.warnings:
                        self.state.logger.warning("Solve %s: %s", _reduce_base, _line)
                        print(f"Warning: solve {_reduce_base}: {_line}", flush=True)
            # One info line per solve (at its first roll) says what was left
            # out and how the levels interact; the run status of every roll
            # records both levels, and the first roll's entry the line too; the
            # outputs leave the cells of the left-out features empty for this
            # sub-solve (process_outputs.reduced_cells).
            if _reduce_report.unit != "none" or _reduce_report.system != "none":
                from flextool.engine.preprocess._reduce_detail import (
                    output_entities,
                    output_features,
                    summary_line,
                )
                _info_seen = self.__dict__.setdefault("_reduce_info_printed", set())
                _summary = None
                if _reduce_base not in _info_seen:
                    _info_seen.add(_reduce_base)
                    _summary = summary_line(_reduce_report, _unreduced, _reduce_base)
                    self.state.logger.info(_summary)
                    print(_summary, flush=True)
                run_record.active().solve_reduced(_reduce_report.unit, _reduce_report.system, _summary)
                if data is not _unreduced:
                    # Written with this sub-solve's outputs (output._writer).
                    data.reduced_output_features = output_features(_reduce_report, _unreduced)
                    data.reduced_output_entities = output_entities(_reduce_report)
            # The outputs read static sets from the last sub-solve: give
            # them the pre-transform data, except the fields every
            # sub-solve so far reduced, so a reduced solve never erases the
            # tables of a full one (_reduce_detail.output_static_view).
            _everywhere = self.__dict__.get("_reduce_everywhere")
            _everywhere = (_reduce_report.fields if _everywhere is None
                           else _everywhere & _reduce_report.fields)
            self._reduce_everywhere = _everywhere
            # The processes every sub-solve so far left without online variables (the online sets of the
            # outputs keep the others, _reduce_detail.output_static_view).
            _gone = online_gone(_unreduced, data)
            _gone_everywhere = self.__dict__.get("_online_gone_everywhere")
            _gone_everywhere = _gone if _gone_everywhere is None else _gone_everywhere & _gone
            self._online_gone_everywhere = _gone_everywhere
            if data is not _unreduced:
                data.output_static_data = output_static_view(data, _unreduced, _everywhere, _gone_everywhere)
                # Only the last sub-solve's view is read: release the views of the stored earlier steps.
                for _prev in (getattr(self, "_all_steps", None) or {}).values():
                    _prev_fd = getattr(_prev, "flex_data", None)
                    if _prev_fd is not None and getattr(_prev_fd, "output_static_data", None) is not None:
                        _prev_fd.output_static_data = None
            # A ramp-carry flow from a producer that left out online
            # variables (online*, or under integers* those of units
            # without a use) has no online carry: those online units start
            # their ramp and commitment cold (the carry routing is kept).
            # Only the processes the producer left out: any other missing
            # online carry is an inconsistency and the check below raises.
            _carry_left_out = getattr(self.state, "current_carry_online_left_out", frozenset()) or frozenset()
            if _carry_left_out:
                data, _cold = cold_start_online_flow_carry(data, _carry_left_out)
                if _cold:
                    _cold_msg = (
                        f"Solve {complete_solve_name}: ramp and commitment of {', '.join(_cold)} start as at "
                        "the start of a solve (the solve it continues from left out online variables, "
                        "reduce_unit_detail_method).")
                    _cold_seen = self.__dict__.setdefault("_reduce_cold_printed", set())
                    if _reduce_base not in _cold_seen:
                        _cold_seen.add(_reduce_base)
                        self.state.logger.info(_cold_msg)
                        print(_cold_msg, flush=True)
            # The online carry and the ramp carry / ramp link must agree on the
            # reduced sets (params._ramp_links): raise instead of silently
            # falling back to a predecessor of 0.
            check_online_ramp_carry(data)

            # A realization solve fixes the stochastic solve's investments
            # and pins the units online of its _fix_in_uc processes to the NA-tied values: name
            # the hard-balance node the realized data cannot balance before
            # the LP is built.
            if complete_solve_name in (
                getattr(self.state.solve, "realized_dispatch_parent", {}) or {}
            ):
                from flextool.engine.input._solve_config import is_investment_solve
                from flextool.engine.solve._realization import (
                    check_pinned_realization_feasible,
                    note_realization_inherits_nothing,
                )
                check_pinned_realization_feasible(data, complete_solve_name)
                _parent = self.state.solve.realized_dispatch_parent[complete_solve_name]
                _base = re.sub(r"_roll_\d+$", "", _parent)
                # The rolls of a rolling solve build no investment, whatever it authors (_recursive_solve warns).
                note_realization_inherits_nothing(
                    data, complete_solve_name, _parent, is_investment_solve(self.state.solve, _base))
            _window_check = getattr(self.state, "current_committed_window_check", None)
            if _window_check is not None:
                from flextool.engine.solve._realization import check_committed_window_realized
                check_committed_window_realized(data, *_window_check)

            # --- LP scaling -------------------------------------------------
            # Phase 2b — the legacy ``scaling.analyze_solve`` /
            # ``ScaleTable`` / ``resolve_effective_scaling`` pipeline has
            # been retired in favour of the autoscale package; the
            # cascade now resolves the per-solve effective objective
            # scale directly from the user's DB override (defaulting to
            # the legacy 1e-6 when absent) and lets autoscale Layer 3
            # handle residual cost / bound magnitudes inside HiGHS.
            base_solve_name = re.sub(r"_roll_\d+$", "", complete_solve_name)

            # v60/v62 per-solve decomposition routing -------------------
            # Consume-side guard: if the immediately-preceding captured
            # solve ran under decomposition=benders and THIS solve is a
            # different base solve, it would consume the (intentionally
            # absent) Benders handoff.  Cross-scheme handoff
            # (Benders → monolithic dispatch) is a deferred follow-up,
            # so we fail loudly rather than silently load handoff=None.
            _guard_msg = _benders_consume_guard_message(
                base_solve_name,
                self.state.last_captured_solve,
                self._benders_solve_names,
                self._benders_invest_handoff_names,
            )
            if _guard_msg is not None:
                raise FlexToolConfigError(_guard_msg)
            # When this solve resolves to decomposition=benders, run it
            # through the Benders region coordinator instead of building
            # and solving a monolithic LP.
            # ``model.output_horizon`` (raw horizon dump): the constraint
            # keys are snapshotted from the built LP before the solve
            # releases them; ``None`` while the flag is off (zero cost).
            _horizon_on = output_horizon_enabled(
                self.state.paths.work_folder if self.state.paths is not None else None,
                provider=_sub_solve_provider,
            )
            _horizon_keys = None
            # The autoscaler config of this sub-solve, resolved once: ``--scaling`` /
            # ``FLEXTOOL_SCALING`` > the solve's ``solve.scaling`` > ``full``.  Every
            # consumer gets it explicitly — the Benders regions, the baseline options, the
            # cold/warm LP build and the Layer 1/2/3 helpers — so a solve never sees another
            # solve's mode.
            _scaling_cfg = _autoscale_resolve_config(
                None, run_options=run_options,
                db_scaling=state.solve.scaling_for(base_solve_name),
            )
            # The Benders incumbent of this solve when its final solve (item 287) runs below: the whole model,
            # built as a monolithic solve, with every investment decision fixed to it.  ``None`` otherwise.
            _bfinal = None
            if state.solve.decomposition_for(base_solve_name) == "benders":
                from flextool.engine.solve._benders_final import final_solve_on
                _final_on = final_solve_on(run_options)
                # Without the final solve a Benders solve writes no outputs, so it has no raw dump either.
                if _horizon_on and not _final_on:
                    warn_benders_no_dump(state.logger, complete_solve_name,
                                         console=not self._benders_dump_notice_printed)
                    self._benders_dump_notice_printed = True
                # The Benders master / regions are their own WarmProblems:
                # release the previous monolithic LP for the whole loop.
                self._release_warm_problem()
                _prev_key_before = self._prev_step_key
                self._benders_result = None
                _rc = self._run_benders_solve(
                    complete_solve_name, base_solve_name, data,
                    scaling_config=_scaling_cfg, final_solve=_final_on,
                )
                _bres, self._benders_result = self._benders_result, None
                _has_incumbent = (
                    _bres is not None and _bres.total_objective is not None
                    and math.isfinite(_bres.total_objective)
                )
                if _rc != 0 or not _final_on or not _has_incumbent:
                    # Final solve off and a later solve gives the dispatch: the Benders solve writes its
                    # investment-only results (W-0571: no investment marginal values).  Last: nothing (W-0570).
                    if _rc == 0 and not _final_on and _has_incumbent and self._later_solve_follows():
                        run_warnings.warn("W-0571", solve=complete_solve_name)
                        self._write_benders_invest_slice(complete_solve_name, _bres, data)
                    return _rc
                _bfinal = _bres
                # The final solve's step replaces the Benders one: the cold-path slim keeps the step before it.
                self._prev_step_key = _prev_key_before
                # The master and region LPs are gone with ``solve_benders``: return their memory before the
                # whole model is built.
                import gc
                gc.collect()
                _try_malloc_trim()
                _btag = f"[benders {base_solve_name}]"
                print(f"{_btag} final solve: the whole model with these investments fixed", flush=True)
                print(f"{_btag}   (if it does not fit in memory: --benders-final-solve off)", flush=True)

            user_obj_scale = state.solve.scale_the_objective.get(complete_solve_name)
            effective_obj_scale = _resolve_effective_obj_scale(user_obj_scale)
            _emit_provider_obj_scale(self.state, effective_obj_scale)
            # ``user_bound_scale`` resolution priority:
            # ``--user-bound-scale`` / ``FLEXTOOL_USER_BOUND_SCALE`` (the
            # run's options) > autoscale Layer 3's automatic recommendation
            # > HiGHS' own internal scaling.  The database has no
            # ``user_bound_scale`` parameter.  HiGHS' "Consider setting the
            # user_bound_scale option to <N>" warning still prints a
            # value if any case slips through Layer 3; pass it via
            # ``--user-bound-scale``.
            user_bound_scale_override = _resolve_user_bound_scale_override(
                run_options.value("user_bound_scale")
            )
            _scaling_mode = _scaling_cfg.mode

            # Per-solve ``solver_arguments`` 1d-map (Batch C.1), listed first
            # in the echoed "Solver args" line.  The full option set of this
            # sub-solve comes from ``_solve_highs_options``.
            _solver_args_map = state.solve.solver_settings.arguments.get(
                complete_solve_name, {}
            )

            # --- LP build & solve ------------------------------------------
            # Warm-LP per-iteration decision.  When ``warm`` is True AND the
            # prior iteration left a live WarmProblem, it is reused only when
            # this sub-solve would build exactly the same LP: equal
            # fingerprint (cheap pre-filter), the same solver configuration
            # and every FlexData field identical, labels included
            # (``_apply_warm_updates`` raises ``_IncompatibleUpdate`` naming
            # the first differing field).  Nothing is pushed into the live
            # LP.  Every other iteration — the first, any rolling roll
            # (``dt`` shifts), any changed field — builds a new LP.
            #
            # Phase 3 — warm-LP is a HiGHS-only design (polar-high's
            # WarmProblem wraps a single live HiGHS instance).  When the
            # active solve picks a commercial solver we disable warm
            # reuse for this iteration, log a one-time warning, and
            # cold-rebuild + dispatch through ``run_one_solve``.
            _active_solver_cfg = state.solve.solver_config_for(complete_solve_name)
            run_record.active().note_solver(_active_solver_cfg.name)
            _warm_disabled_by_solver = (
                warm and _active_solver_cfg.name != "highs"
            )
            # ``FLEXTOOL_SAVE_MEMORY=1`` opts into polar-high's
            # ``save_memory=True`` solve path, which drops the polar-side
            # LP source and round-trips the HiGHS instance through MPS
            # mid-solve.  The Problem is then in a "released" state and
            # can no longer be warm-reused, so every iteration must
            # cold-rebuild.  Resolve once per sub-solve (cheap) so the
            # knob can be toggled between runs of the same cascade.
            _save_memory = bool(run_options.value("save_memory"))
            _warm_disabled_by_save_memory = warm and _save_memory
            if _warm_disabled_by_save_memory and not getattr(
                self, "_warm_disabled_by_save_memory_warned", False
            ):
                # Coded note (``state.logger`` is at ERROR in a cascade, so a
                # logger warning here never showed); once per run.
                run_warnings.note("W-0551")
                self._warm_disabled_by_save_memory_warned = True
            if _warm_disabled_by_solver and not getattr(
                self, "_warm_disabled_warned", False
            ):
                run_warnings.note("W-0552", solver=_active_solver_cfg.name)
                self._warm_disabled_warned = True
            # HiGHS soft-promote: warm=False on HiGHS without
            # FLEXTOOL_SAVE_MEMORY=1 used to fall through to an in-
            # process cold rebuild that built a fresh ``highspy.Highs``
            # inside this Python process — undoing the entire reason
            # warm reuse exists in the first place (peak RSS).  Retired
            # path: route every HiGHS cold solve through the same
            # ``cmd_solve_mps`` subprocess the save-memory branch uses,
            # bounding peak RSS to ``write_mps``'s footprint.  Mutate
            # only the local ``_save_memory`` — do NOT touch the env
            # var, which would leak to sibling solves.
            if (
                (not warm)
                and _active_solver_cfg.name == "highs"
                and not _save_memory
            ):
                if not getattr(
                    self, "_warm_disabled_softpromote_warned", False,
                ):
                    run_warnings.note("W-0553")
                    self._warm_disabled_softpromote_warned = True
                _save_memory = True
            if _bfinal is not None and _active_solver_cfg.name == "highs":
                # The final solve of a Benders solve runs in the HiGHS child process (``cmd_solve_mps``): a model
                # too big for the memory left after the Benders loop then fails that solve, which the run
                # survives, instead of killing the run.
                _save_memory = True
            warm_used = False
            warm_active = (
                warm
                and not _warm_disabled_by_solver
                and not _warm_disabled_by_save_memory
                # The final solve is built cold (its pinned bounds are no reuse of a WarmProblem).
                and _bfinal is None
            )
            if warm_active:
                fp = _fingerprint(data)
                # Solver configuration of this sub-solve, resolved BEFORE the
                # reuse decision: a reused LP keeps the options, autoscale
                # mode and objective scale of its build, so a sub-solve with
                # other settings must build a new LP.  ``highs_options`` is
                # also what the fresh-build branch below applies.
                highs_options = _solve_highs_options(
                    state, complete_solve_name,
                    baseline=_baseline_highs_options(
                        user_bound_scale_override=user_bound_scale_override,
                        scaling_mode=_scaling_mode,
                    ),
                )
                solver_key = (
                    repr(effective_obj_scale),
                    repr(_scaling_mode),
                    repr(user_bound_scale_override),
                    tuple(sorted((str(k), repr(v)) for k, v in highs_options.items())),
                )
                # Phase 4 Step 4b bookkeeping — ``did_fresh_build`` gates
                # the post-solve basis CAPTURE to fresh builds only (a warm
                # reuse re-solves the identical LP, whose basis was already
                # captured at its build); ``fresh_fp`` carries the name-set
                # cache key from the inject block to the capture block.
                # Both are inert unless the basis-cache gate is active.
                did_fresh_build = False
                fresh_fp = None
                tried_warm = (
                    self._warm_problem is not None
                    and self._prior_data is not None
                    and self._prior_fp == fp
                )
                if tried_warm and self._prior_solver_key != solver_key:
                    tried_warm = False
                    state.logger.info(
                        "warm reuse fell back to COLD REBUILD for %s: the "
                        "solver configuration differs from the previous "
                        "sub-solve (solver options / scaling / "
                        "user_bound_scale / objective scale)",
                        complete_solve_name,
                    )
                if tried_warm:
                    try:
                        _apply_warm_updates(self._warm_problem,
                                            self._prior_data, data)
                        warm_used = True
                    except _IncompatibleUpdate as _warm_exc:
                        # The normal outcome for every roll (``dt`` shifts),
                        # so INFO, not WARNING.  The message names the first
                        # differing FlexData field.
                        _log_cold_rebuild(
                            state.logger, self.__dict__.setdefault("_cold_rebuild_logged", set()),
                            complete_solve_name, str(_warm_exc),
                        )
                if not warm_used:
                    # Drop the previous WarmProblem BEFORE the new build on
                    # every non-reuse path (fingerprint mismatch, solver
                    # configuration change, differing field): its HiGHS
                    # instance (``WarmProblem._h``) is not released by the
                    # ``Solution.highs`` loop below, so keeping it would let
                    # two LPs coexist during the build.  The cached Layer 2
                    # plan and ranges die with it.
                    self._release_warm_problem()
                    # Cross-level eviction — same-level COLD-rebuild case
                    # (e.g. every roll of a rolling solve).  We are about to
                    # build a fresh LP and are NOT warm-reusing, so no parked
                    # step's HiGHS is a reuse source and ``self._warm_problem``
                    # was just nulled above.  Any prior step still holding a
                    # live ``solution.highs`` — typically the previous
                    # same-level roll; exhausted *other* levels were already
                    # freed at the top of run() — is now the ONLY reference to
                    # that HiGHS.  Release it BEFORE ``_build_warm_problem``
                    # allocates, else two same-level footprints coexist (the
                    # DES 7/9 dispatch-on-dispatch stack).  Per-iter writers +
                    # handoff already consumed each prior step on its own iter,
                    # so this is safe.  ``flex_data_provider`` is NOT dropped
                    # here — same-level rolls reuse the per-level provider
                    # cache (``state._level_providers``).
                    if not keep_solutions:
                        _cr_released = False
                        for _ck, _cstep in (getattr(self, "_all_steps", None) or {}).items():
                            _csol = getattr(_cstep, "solution", None)
                            if _csol is not None and getattr(_csol, "highs", None) is not None:
                                _csol.highs = None
                                _cr_released = True
                        if _cr_released:
                            _try_malloc_trim()
                    if not keep_solutions:
                        _audit_cold_rebuild_release(
                            steps=getattr(self, "_all_steps", None),
                            complete_solve_name=complete_solve_name,
                            warm_problem_live=self._warm_problem is not None,
                        )
                    # Build the warm problem first WITHOUT solver
                    # options so we can inspect LP ranges, then push the
                    # finalised HiGHS options through ``set_solver_options``
                    # on the underlying Problem.
                    _phase_prof("build_start")
                    self._warm_problem = _build_warm_problem(
                        data,
                        scale_the_objective=effective_obj_scale,
                        solver_options=None,
                        solve_name=complete_solve_name,
                    )
                    _phase_prof("build_done")
                    if _memrec_local is not None and _emit_phase:
                        _memrec_local.checkpoint(
                            "lp_build_end", self.state.logger,
                            user_label="Matrix built by polar-high",
                        )
                    inner_pb = self._warm_problem.problem
                    # ``highs_options`` was resolved above, before the reuse
                    # decision (it is part of ``solver_key``).
                    inner_pb.set_solver_options(highs_options)
                    # Autoscale Layer 2 + Layer 3 on the warm-active
                    # first-build branch.  Same call sequence as the
                    # cold path below — see the longer-form comment
                    # there for the rationale.  Skipping these on warm
                    # solves used to leave HiGHS staring at an unscaled
                    # LP, costing both numerical health and ~tens of GB
                    # of internal simplex working set on
                    # poorly-conditioned LPs.
                    #
                    # First-build only: Layer 2 writes side vectors on
                    # the Problem and ``WarmProblem._initial_build``
                    # bakes them into the canonical matrix.  A reuse
                    # solves the identical LP, so no Layer 2 re-apply is
                    # needed; ``self._autoscale_warm_layer2_plan`` caches
                    # the plan for use by
                    # :func:`_autoscale_unscale_post_solve` after every
                    # warm solve (first build AND reuses).
                    #
                    # Per-shape autoscale DECISION cache: keyed on the
                    # BUILT LP's structural signature (matrix shape +
                    # per-family layout), which is invariant across rolls
                    # of the same rolling solve even though
                    # ``_fingerprint(data)`` slides (a windowed period/dt
                    # field's height tracks the rolling horizon).  On a HIT
                    # we replay the cached Layer-2 exponents + Layer-3 plan
                    # WITHOUT any ``detect_ranges`` / ``bucket_coefficients``
                    # walk (the per-roll multi-GB spike).  A cold rebuild of
                    # an already-seen shape (every roll after the first)
                    # therefore skips the traversals entirely.
                    _shape_key = _autoscale_lp_shape_signature(
                        inner_pb, base_solve_name,
                    )
                    _cache_entry = (
                        None if _autoscale_disable_cache()
                        else self._autoscale_shape_cache.get(_shape_key)
                    )
                    _l3_readout: dict = {}
                    _autoscale_hook = None
                    _phase_prof("autoscale_l2_start")
                    if _cache_entry is not None:
                        # HIT — replay decision, no range walk.
                        self._autoscale_warm_layer2_plan = (
                            _autoscale_apply_layer2_from_cache(
                                inner_pb, _cache_entry,
                                solve_name=complete_solve_name,
                                logger=self.state.logger,
                            )
                        )
                        _autoscale_ranges_pre = _cache_entry.ranges_pre
                        self._autoscale_warm_ranges_pre = _autoscale_ranges_pre
                        _phase_prof("autoscale_l3_start")
                        _autoscale_layer3_plan = (
                            _autoscale_apply_layer3_from_cache(
                                inner_pb, _cache_entry,
                                solve_name=complete_solve_name,
                                logger=self.state.logger,
                            )
                        )
                        _autoscale_ranges_post = _cache_entry.ranges_post
                        # Replay the first roll's guard outcome; recheck this
                        # roll's built LP in WarmProblem's first build.
                        _autoscale_guard = _cache_entry.guard
                        if _autoscale_layer3_plan is not None:
                            _autoscale_hook = _AutoscaleRollRecheck(
                                inner_pb, plan=_autoscale_layer3_plan,
                                config=_scaling_cfg,
                                mode=_autoscale_resolve_guard_mode(run_options),
                                solve_name=_roll_label(complete_solve_name, _roll_idx),
                                base_solve=base_solve_name,
                                layer2_applied=self._autoscale_warm_layer2_plan is not None,
                            )
                    else:
                        # MISS — run the full traversals, then cache.
                        (
                            self._autoscale_warm_layer2_plan,
                            _autoscale_ranges_pre,
                        ) = _autoscale_apply_layer2_pre_solve(
                            inner_pb,
                            solve_name=complete_solve_name,
                            logger=self.state.logger,
                            scaling_config=_scaling_cfg,
                        )
                        # Cache the pre-Layer-2 RangeReport so subsequent
                        # warm reuses can still attach it as
                        # ``Solution.streamed_lp_ranges`` (the cascade only
                        # builds the LP once, so the four ranges are
                        # invariant across rolls of the same warm problem).
                        self._autoscale_warm_ranges_pre = _autoscale_ranges_pre
                        _phase_prof("autoscale_l3_start")
                        _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
                            inner_pb,
                            layer2_plan=self._autoscale_warm_layer2_plan,
                            solve_name=complete_solve_name,
                            logger=self.state.logger,
                            scaling_config=_scaling_cfg,
                            ranges_pre=_autoscale_ranges_pre,
                            readout=_l3_readout,
                        )
                        # The post-Layer-2 readout Layer 3 decided on, shown by
                        # the summary when the trigger fired (as the cold path).
                        _autoscale_ranges_post = (
                            _l3_readout.get("ranges")
                            if _autoscale_ranges_pre is not None
                            and _autoscale_ranges_pre.trigger
                            else None
                        )
                        _autoscale_guard = _autoscale_guard_check(
                            inner_pb, ranges=_l3_readout.get("ranges"),
                            ranges_pre=_autoscale_ranges_pre,
                            layer2_plan=self._autoscale_warm_layer2_plan,
                            layer3_plan=_autoscale_layer3_plan,
                            solve_name=complete_solve_name, run_options=run_options,
                            scaling_config=_scaling_cfg, logger=self.state.logger,
                        )
                        if not _autoscale_disable_cache():
                            _l2p = self._autoscale_warm_layer2_plan
                            self._autoscale_shape_cache[_shape_key] = (
                                _AutoscaleShapeCacheEntry(
                                    layer2_exponents=(
                                        dict(_l2p.type_exponents)
                                        if _l2p is not None else None
                                    ),
                                    layer2_buckets_before=(
                                        dict(_l2p.type_buckets_before)
                                        if _l2p is not None else {}
                                    ),
                                    layer2_buckets_after=(
                                        dict(_l2p.type_buckets_after)
                                        if _l2p is not None else {}
                                    ),
                                    layer3_plan=_autoscale_layer3_plan,
                                    ranges_pre=_autoscale_ranges_pre,
                                    ranges_post=_autoscale_ranges_post,
                                    guard=_autoscale_guard,
                                )
                            )
                    _phase_prof("autoscale_summary_start")
                    _autoscale_emit_console_summary(
                        ranges_pre=_autoscale_ranges_pre,
                        ranges_post=_autoscale_ranges_post,
                        layer2_plan=self._autoscale_warm_layer2_plan,
                        layer3_plan=_autoscale_layer3_plan,
                        solve_name=base_solve_name,
                        already_emitted=self._autoscale_summary_emitted,
                        memrec=_memrec_local if _emit_phase else None,
                        logger=self.state.logger,
                        scaling_config=_scaling_cfg,
                        guard=_autoscale_guard,
                    )
                    _autoscale_write_presolve_report(
                        solve_name=complete_solve_name, logger=self.state.logger,
                        work_folder=self.state.paths.work_folder,
                        ranges=_autoscale_ranges_post or _autoscale_ranges_pre,
                        layer2_plan=self._autoscale_warm_layer2_plan,
                        layer3_plan=_autoscale_layer3_plan,
                        guard=_autoscale_guard, scaling_config=_scaling_cfg,
                    )
                    # Outside every ``except``: an error stops the run here,
                    # before HiGHS runs.
                    _autoscale_enforce_guard(_autoscale_guard, base_solve=base_solve_name)
                    _phase_prof("autoscale_done")
                    # Phase 4 Step 4b — in-process warm-start basis
                    # INJECT (fresh build only).  Keyed on the LP name-set
                    # fingerprint, which is Layer-2-invariant, so it is
                    # computed HERE — after Layer 2 — to reuse the memoized
                    # ``canonicalise`` rather than assemble the names
                    # twice.  On a cache hit (in-process dict first, else
                    # the on-disk ``<fp>.nbasis``) we record the basis on
                    # the WarmProblem; the actual ``setBasis`` fires once
                    # in ``_initial_build`` before the first solve and
                    # safely falls back to a cold solve on any fingerprint
                    # mismatch.  Warm-start must NEVER break a solve, so
                    # every step here is defensive — any failure logs and
                    # continues cold.
                    did_fresh_build = True
                    if _basis_cache_active(
                        state.solve.decomposition_for(base_solve_name),
                        _active_solver_cfg.name,
                        run_options,
                    ):
                        try:
                            fresh_fp = inner_pb.basis_name_fingerprint()
                            if fresh_fp:
                                nb = self._inproc_basis_cache.get(fresh_fp)
                                if nb is None:
                                    _bc_dir = _basis_cache_dir(
                                        self.state.paths.work_folder,
                                        run_options,
                                    )
                                    _nb_path = (
                                        _bc_dir / f"{fresh_fp}.nbasis"
                                    )
                                    if _nb_path.exists():
                                        _raw = json.loads(
                                            _nb_path.read_text()
                                        )
                                        nb = NamedBasis(
                                            col_status=_raw["col_status"],
                                            row_status=_raw["row_status"],
                                            fingerprint=_raw["fingerprint"],
                                        )
                                if nb is not None:
                                    self._warm_problem.set_named_basis(
                                        nb, policy="exact",
                                    )
                                    self.state.logger.info(
                                        "in-process warm-basis cache hit "
                                        "%s", fresh_fp,
                                    )
                        except Exception as _basis_exc:  # noqa: BLE001
                            self.state.logger.debug(
                                "in-process warm-basis inject skipped "
                                "(%s); solving cold", _basis_exc,
                            )
                            # Coded: ``state.logger`` is at ERROR in a
                            # cascade.  One line per complete solve.
                            run_warnings.warn_grouped(
                                "W-0550", solve=complete_solve_name,
                                step="loaded", effect="solved from scratch",
                                details=f"in '{complete_solve_name}': "
                                        f"{_basis_exc}",
                            )
                # ``WarmProblem.solve`` always keeps the HiGHS instance
                # alive on ``Solution.highs`` — that's the whole point
                # of warm reuse — so the output writer adapter
                # (``write_all_variables`` / ``write_all_handoffs``)
                # sees the live solver as it does for cold rebuilds
                # under ``keep_solver=True``.  No extra kwarg required.
                # Blank line so HiGHS' "Running HiGHS …" banner (and the
                # grey solver-output block in the GUI) separates from the
                # scaling/LP-build rows above it.  The sub-solve's coded
                # warnings gathered so far print before it.
                run_warnings.flush()
                print("", flush=True)
                # Echo the effective solver options right before HiGHS'
                # own banner: the full merged, post-precedence set the
                # solver actually received (``highs.opt`` floor ∪
                # ``solver_mip_gap`` ∪ DB ``solver_arguments`` ∪ CLI overrides), so the run is
                # replicable from the log alone.  Resolved with
                # ``baseline=None`` so the engine-internal determinism /
                # scale keys are excluded — only what the operator
                # touched is shown.  Uses ``print`` (not the logger) to
                # sit in the same stdout stream as the blank separator
                # above and HiGHS' native banner below — the GUI's
                # ``execution_window`` parses these printed markers.
                _eff_touched = _solve_highs_options(state, complete_solve_name, baseline=None)
                _sa_line = _format_solver_args_line(
                    _eff_touched, prefer_first=list(_solver_args_map or {}),
                    translated=_highs_echo_translations(
                        self._warm_problem.problem, _eff_touched, complete_solve_name,
                    ),
                )
                _print_solver_args(_sa_line)
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("solve_start")
                # A fresh build replaying a cached plan is rechecked from its
                # own ranges (polar-high runs the hook on the first build only;
                # a reuse solves the LP already checked).
                _warm_hook = locals().get("_autoscale_hook") if did_fresh_build else None
                if _warm_hook is not None:
                    try:
                        sol = self._warm_problem.solve(pre_run_check=_warm_hook)
                    except BaseException:
                        # The recheck (or the build after it) stopped the
                        # solve: the YAML gets this roll's own verdict.
                        _autoscale_report_recheck_stop(
                            _warm_hook, solve_name=complete_solve_name,
                            logger=self.state.logger,
                            work_folder=self.state.paths.work_folder,
                            ranges=_autoscale_ranges_post or _autoscale_ranges_pre,
                            layer2_plan=self._autoscale_warm_layer2_plan,
                            scaling_config=_scaling_cfg,
                        )
                        raise
                    if _warm_hook.called:
                        _autoscale_layer3_plan = _warm_hook.plan
                        if _warm_hook.outcome is not None:
                            _autoscale_guard = _warm_hook.outcome
                else:
                    sol = self._warm_problem.solve()
                if _horizon_on:
                    # A WarmProblem keeps its constraint families alive.
                    _horizon_keys = snapshot_cstr_keys(self._warm_problem.problem)
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("after_solve")
                # Attach the cached pre-Layer-2 RangeReport as
                # ``streamed_lp_ranges`` on the warm Solution so the
                # Layer-1 emit hook and downstream telemetry
                # (e.g. ``test_invest_chain_lp_bound_range_smoke``) see
                # the four (min, max) pairs the autoscaler decided on.
                # ``WarmProblem.solve`` returns a bare Solution with no
                # streamed ranges of its own — the polar-high in-process
                # streaming-solve path that populates that attribute is
                # bypassed when HiGHS' ``Highs.run`` is invoked directly
                # on the live instance, so the warm path has to surface
                # the ranges itself.  Mirrors the cold-path assignment
                # below (search for ``sol.streamed_lp_ranges = {``).
                _warm_ranges = self._autoscale_warm_ranges_pre
                if (
                    _warm_ranges is not None
                    and getattr(sol, "streamed_lp_ranges", None) is None
                ):
                    try:
                        sol.streamed_lp_ranges = {
                            "matrix": _warm_ranges.matrix,
                            "cost": _warm_ranges.cost,
                            "col_bound": _warm_ranges.bound,
                            "row_bound": _warm_ranges.rhs,
                        }
                    except Exception:  # pragma: no cover — Solution may
                        # forbid the assignment in a future polar-high
                        pass
                # Eager unscale on every warm solve (first-build AND
                # reuses) so output writers see physical-coordinate
                # primal / duals / reduced costs.  No-op when the
                # cached plan is None (Layer 2 was off or didn't
                # trigger at first-build).
                _phase_prof("unscale_start")
                _autoscale_unscale_post_solve(
                    sol, self._autoscale_warm_layer2_plan,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                    problem=self._warm_problem.problem,
                )
                _phase_prof("unscale_done")
                # Phase 4 Step 4b — in-process warm-start basis CAPTURE
                # (fresh build only).  ``sol.highs`` is the live retained
                # handle (``WarmProblem.solve`` always keeps the solver),
                # so ``get_named_basis`` succeeds.  Persist atomically to
                # the shared cache dir so a later cross-run solve of the
                # same structural model can inject it (UC2 sweep / UC3
                # resume).  Never capture on the warm-reuse branch
                # (``did_fresh_build`` stays False there) — the LP is the
                # one captured at its build and is already optimally warm.
                # The cache is keyed on the exact name set, so a roll is
                # warm-started only from the same roll of an earlier run,
                # never from the previous roll (whose labels differ).
                # Capture failure is non-fatal (log + continue).  Only an
                # optimal solve's basis is a seed: a solve stopped by a limit
                # (kept or rejected by the acceptance check below) is not.
                if (
                    did_fresh_build
                    and sol.optimal
                    and fresh_fp
                    and _basis_cache_active(
                        state.solve.decomposition_for(base_solve_name),
                        _active_solver_cfg.name,
                        run_options,
                    )
                ):
                    try:
                        nb_out = sol.get_named_basis()
                        self._inproc_basis_cache[fresh_fp] = nb_out
                        _bc_dir = _basis_cache_dir(
                            self.state.paths.work_folder,
                            run_options,
                        )
                        _tmp = (
                            _bc_dir
                            / f"{fresh_fp}.nbasis.tmp.{os.getpid()}"
                        )
                        _tmp.write_text(json.dumps({
                            "col_status": nb_out.col_status,
                            "row_status": nb_out.row_status,
                            "fingerprint": nb_out.fingerprint,
                        }))
                        os.replace(
                            _tmp, _bc_dir / f"{fresh_fp}.nbasis",
                        )
                        self.state.logger.info(
                            "in-process warm-basis cache captured %s",
                            fresh_fp,
                        )
                    except Exception as _basis_exc:  # noqa: BLE001
                        self.state.logger.debug(
                            "in-process warm-basis capture skipped (%s)",
                            _basis_exc,
                        )
                        run_warnings.warn_grouped(
                            "W-0550", solve=complete_solve_name,
                            step="saved", effect="a later solve starts cold",
                            details=f"in '{complete_solve_name}': "
                                    f"{_basis_exc}",
                        )
                self._prior_data = data
                self._prior_fp = fp
                self._prior_solver_key = solver_key
            else:
                # Not warm-active (non-HiGHS solver, save-memory, warm=False):
                # this LP is built cold, so the previous WarmProblem is no
                # reuse source; release it before the build.
                self._release_warm_problem()
                pb = Problem()
                _phase_prof("build_start")
                try:
                    build_flextool(pb, data, scale_the_objective=effective_obj_scale,
                                   solve_name=complete_solve_name)
                    if _bfinal is not None:
                        # In the model's own units, before Layer 2 scales the columns.
                        from flextool.engine.solve._benders_final import pin_invest_decisions
                        pin_invest_decisions(pb, _bfinal.invest_solution_vars)
                except MemoryError as _mem_exc:
                    if _bfinal is None:
                        raise
                    # The whole model of the final solve does not fit in the memory left.
                    pb = None
                    return self._benders_final_out_of_memory(complete_solve_name, _mem_exc,
                                                             bres=_bfinal, data=data)
                _phase_prof("build_done")
                if _memrec_local is not None and _emit_phase:
                    _memrec_local.checkpoint(
                        "lp_build_end", self.state.logger,
                        user_label="Matrix built by polar-high",
                    )
                highs_options = _solve_highs_options(
                    state, complete_solve_name,
                    baseline=_baseline_highs_options(
                        user_bound_scale_override=user_bound_scale_override,
                        scaling_mode=_scaling_mode,
                    ),
                )
                pb.set_solver_options(highs_options)
                # ── DIAGNOSTIC: per-substep RSS in the pre-write_mps gap ──
                # OOM in this gap (post "Matrix built", pre write_mps) is
                # invisible to both ``_MemoryRecorder`` (single checkpoint
                # for "Matrix built") and ``POLAR_HIGH_WRITE_MPS_PROFILE``
                # (only fires inside write_mps).  This closure samples
                # ``psutil.Process().memory_info().rss`` at each substep
                # below and writes to stderr in the same format as the
                # polar-high profile.  Activate with
                # ``FLEXTOOL_AUTOSCALE_PROFILE=1``; zero overhead when off.
                _autoscale_profile = (
                    os.environ.get("FLEXTOOL_AUTOSCALE_PROFILE") == "1"
                )
                if _autoscale_profile:
                    try:
                        import psutil as _psutil
                        _ap_proc = _psutil.Process()
                        _ap_t0 = time.monotonic()
                        _ap_prev = _ap_proc.memory_info().rss / (1024 ** 3)
                        import sys as _sys
                        def _ap(phase: str, **extras) -> None:
                            nonlocal _ap_prev
                            rss = _ap_proc.memory_info().rss / (1024 ** 3)
                            delta = rss - _ap_prev
                            wall = time.monotonic() - _ap_t0
                            sign = "+" if delta >= 0 else ""
                            extras_str = "\t".join(
                                f"{k}={v}" for k, v in extras.items()
                            )
                            print(
                                f"[autoscale profile]\tphase={phase}\t"
                                f"rss_gb={rss:.2f}\tdelta_gb={sign}{delta:.2f}"
                                f"\twall_s={wall:.2f}"
                                + (f"\t{extras_str}" if extras_str else ""),
                                file=_sys.stderr, flush=True,
                            )
                            _ap_prev = rss
                        _ap("enter")
                    except ImportError:
                        _autoscale_profile = False
                        print(
                            "FLEXTOOL_AUTOSCALE_PROFILE=1 but psutil not "
                            "installed; profiling disabled.",
                            file=__import__("sys").stderr, flush=True,
                        )
                # autoscale Layer 2 (semantic per-type) pre-solve apply.
                # Trigger gate is the same Layer-1 four-range readout —
                # see ``_autoscale_apply_layer2_pre_solve``.  Plan is
                # consumed by ``_autoscale_unscale_post_solve`` once the
                # solve returns so downstream output writers see the
                # un-scaled solution.
                # Per-shape autoscale DECISION cache (cold ``warm=False``
                # / save_memory path).  Keyed on the BUILT LP's structural
                # signature (matrix shape + per-family layout), which is
                # invariant across rolls of the same rolling solve even
                # though ``_fingerprint(data)`` slides as the rolling
                # horizon moves.  Cheap to compute (no coefficient walk) so
                # repeated same-shape cold solves replay the cached decision
                # without the per-roll ``detect_ranges`` /
                # ``bucket_coefficients`` traversal.  Honours
                # ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1``.
                _cold_key = (
                    None if _autoscale_disable_cache()
                    else _autoscale_lp_shape_signature(pb, base_solve_name)
                )
                _cache_entry = (
                    self._autoscale_shape_cache.get(_cold_key)
                    if _cold_key is not None else None
                )
                _l3_readout: dict = {}
                _new_cache_entry: "_AutoscaleShapeCacheEntry | None" = None
                _phase_prof("autoscale_l2_start")
                if _cache_entry is not None:
                    # HIT — replay decision, NO range walk.
                    _autoscale_layer2_plan = _autoscale_apply_layer2_from_cache(
                        pb, _cache_entry,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    _autoscale_ranges_pre = _cache_entry.ranges_pre
                    if _autoscale_profile:
                        _ap("layer2_applied_cached",
                            n_cstrs=len(pb._cstrs),
                            n_vars=len(pb._vars))
                    _phase_prof("autoscale_l3_start")
                    _autoscale_layer3_plan = _autoscale_apply_layer3_from_cache(
                        pb, _cache_entry,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    if _autoscale_profile:
                        _ap("layer3_applied_cached")
                    _phase_prof("autoscale_summary_start")
                    _autoscale_ranges_post = _cache_entry.ranges_post
                    if _autoscale_profile:
                        _ap("ranges_post_cached",
                            ranges_post_ran=str(_autoscale_ranges_post is not None))
                else:
                    # MISS — run the full traversals, then cache.
                    (
                        _autoscale_layer2_plan,
                        _autoscale_ranges_pre,
                    ) = _autoscale_apply_layer2_pre_solve(
                        pb,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                        scaling_config=_scaling_cfg,
                    )
                    if _autoscale_profile:
                        _ap("layer2_applied",
                            n_cstrs=len(pb._cstrs),
                            n_vars=len(pb._vars))
                    # Layer 3 (HiGHS-native top-up): set user_objective_scale,
                    # user_bound_scale, and simplex_scale_strategy from the
                    # post-Layer-2 ranges so HiGHS sees a final LP that is
                    # already inside its comfort zone.  Layer 3 is HiGHS-
                    # internal (no inverse transform on the solution); the
                    # writeModel MPS export remains unscaled.
                    _phase_prof("autoscale_l3_start")
                    _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
                        pb,
                        layer2_plan=_autoscale_layer2_plan,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                        scaling_config=_scaling_cfg,
                        ranges_pre=_autoscale_ranges_pre,
                        readout=_l3_readout,
                    )
                    if _autoscale_profile:
                        _ap("layer3_applied")
                    # Console summary: one user-visible line per base solve
                    # describing the autoscaler's pre/post ranges and the
                    # Layer 2 / Layer 3 decisions.  Read post-Layer-2 ranges
                    # from the (mutated) Problem so the "after" view reflects
                    # what HiGHS will see; Layer 3 acts inside HiGHS so its
                    # values are surfaced separately in the same line.
                    _phase_prof("autoscale_summary_start")
                    # The post-Layer-2 readout Layer 3 decided on (no second
                    # walk); the summary shows it when the trigger fired.
                    _autoscale_ranges_post: "_AutoscaleRangeReport | None" = (
                        _l3_readout.get("ranges")
                        if _autoscale_ranges_pre is not None and _autoscale_ranges_pre.trigger
                        else None
                    )
                    if _autoscale_profile:
                        _ap("ranges_post_computed",
                            ranges_post_ran=str(_autoscale_ranges_post is not None))
                    if _cold_key is not None:
                        _l2p = _autoscale_layer2_plan
                        _new_cache_entry = self._autoscale_shape_cache[_cold_key] = (
                            _AutoscaleShapeCacheEntry(
                                layer2_exponents=(
                                    dict(_l2p.type_exponents)
                                    if _l2p is not None else None
                                ),
                                layer2_buckets_before=(
                                    dict(_l2p.type_buckets_before)
                                    if _l2p is not None else {}
                                ),
                                layer2_buckets_after=(
                                    dict(_l2p.type_buckets_after)
                                    if _l2p is not None else {}
                                ),
                                layer3_plan=_autoscale_layer3_plan,
                                ranges_pre=_autoscale_ranges_pre,
                                ranges_post=_autoscale_ranges_post,
                            )
                        )
                # The commercial parameters are resolved before the scaling
                # guard, which judges a commercial solve at the solver's own
                # feasibility tolerance (and prints the ``Solver args`` echo
                # computed here further down, right before the solver).
                _commercial_opts = None
                if _active_solver_cfg.name != "highs":
                    from flextool.engine.solve._solver_dispatch import (
                        _PARAM_MAP as _NATIVE_NAMES,
                        commercial_opt_file_layer,
                        problem_is_mip,
                        resolve_commercial_options,
                    )
                    if _active_solver_cfg.name in _NATIVE_NAMES:
                        _opt_text, _opt_label, _opt_untrusted = commercial_opt_file_layer(
                            _active_solver_cfg.name,
                            solver_option_files=state.paths.solver_option_files,
                            solver_config_dir=state.paths.solver_config_dir,
                            run_options=run_options,
                        )
                        _commercial_opts = resolve_commercial_options(
                            _active_solver_cfg.name,
                            opt_file=_opt_label,
                            opt_text=_opt_text,
                            opt_untrusted=_opt_untrusted,
                            solver_config=_active_solver_cfg,
                            run_options=run_options,
                            solve_name=complete_solve_name,
                            logger=state.logger,
                            is_mip=problem_is_mip(getattr(pb, "_vars", None)),
                        )
                        # The values in unscaled currency become the solver's
                        # values when the option file is written
                        # (``solve_via_subprocess``); the echo shows both.
                        _, _commercial_translated = _objective_options.translate_for_solver(
                            pb, _active_solver_cfg.name, "mps", _commercial_opts,
                            solve_name=complete_solve_name,
                        )
                        _sa_line = _format_solver_args_line(
                            {o.name: o.value for o in _commercial_opts},
                            solver=_active_solver_cfg.name,
                            translated=_objective_options.echo_values(_commercial_translated),
                        )
                    else:
                        _sa_line = None
                else:
                    _eff_touched = _solve_highs_options(state, complete_solve_name, baseline=None)
                    _sa_line = _format_solver_args_line(
                        _eff_touched, prefer_first=list(_solver_args_map or {}),
                        translated=_highs_echo_translations(pb, _eff_touched, complete_solve_name),
                    )
                # Scaling guard (before any solver runs; the raise is outside
                # every ``except``).  First solve of a shape: judged from the
                # Layer 3 readout.  A cache HIT replays that outcome (no walk,
                # no ``locate_extremes``) and rechecks this roll's built LP in
                # polar-high's ``pre_run_check`` (HiGHS only: in-process
                # ``Problem.solve`` and the --save-memory ``write_mps``).
                _autoscale_hook = None
                if _cache_entry is not None:
                    _autoscale_guard = _cache_entry.guard
                    if (
                        _autoscale_layer3_plan is not None
                        and _active_solver_cfg.name == "highs"
                    ):
                        _autoscale_hook = _AutoscaleRollRecheck(
                            pb, plan=_autoscale_layer3_plan, config=_scaling_cfg,
                            mode=_autoscale_resolve_guard_mode(run_options),
                            solve_name=_roll_label(complete_solve_name, _roll_idx),
                            base_solve=base_solve_name,
                            layer2_applied=_autoscale_layer2_plan is not None,
                        )
                else:
                    _autoscale_guard = _autoscale_guard_check(
                        pb, ranges=_l3_readout.get("ranges"),
                        ranges_pre=_autoscale_ranges_pre,
                        layer2_plan=_autoscale_layer2_plan,
                        layer3_plan=_autoscale_layer3_plan,
                        solve_name=complete_solve_name, run_options=run_options,
                        scaling_config=_scaling_cfg, logger=self.state.logger,
                        solver=_active_solver_cfg.name,
                        commercial_options=_commercial_opts,
                    )
                    if _new_cache_entry is not None:
                        _new_cache_entry.guard = _autoscale_guard
                _autoscale_emit_console_summary(
                    ranges_pre=_autoscale_ranges_pre,
                    ranges_post=_autoscale_ranges_post,
                    layer2_plan=_autoscale_layer2_plan,
                    layer3_plan=_autoscale_layer3_plan,
                    solve_name=base_solve_name,
                    already_emitted=self._autoscale_summary_emitted,
                    memrec=_memrec_local if _emit_phase else None,
                    logger=self.state.logger,
                    scaling_config=_scaling_cfg,
                    guard=_autoscale_guard,
                )
                _autoscale_write_presolve_report(
                    solve_name=complete_solve_name, logger=self.state.logger,
                    work_folder=self.state.paths.work_folder,
                    ranges=_autoscale_ranges_post or _autoscale_ranges_pre,
                    layer2_plan=_autoscale_layer2_plan,
                    layer3_plan=_autoscale_layer3_plan,
                    guard=_autoscale_guard, scaling_config=_scaling_cfg,
                )
                _autoscale_enforce_guard(_autoscale_guard, base_solve=base_solve_name)
                _phase_prof("autoscale_done")
                if _autoscale_profile:
                    _ap("console_summary_done")
                # Phase 3 — multi-solver dispatch.  ``run_one_solve`` calls
                # ``pb.solve(keep_solver=True)`` for the default HiGHS path
                # (byte-identical to the pre-Phase-3 behaviour); routes to
                # ``solve_via_subprocess`` (HiGHS CLI / commercial CLI) on
                # every cold path, which always returns a real
                # ``polar_high.Solution`` with the HiGHS instance read back
                # from the MPS.  The cascade-level SolverConfig lookup uses
                # the active solve name with the standard
                # default-when-absent fallback.
                from flextool.engine.solve._solver_dispatch import (
                    run_one_solve,
                )
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                # Pre-solve range capture: every cold path now goes
                # subprocess, and the subprocess Solution carries no
                # ``streamed_lp_ranges`` (polar-high populates that
                # during its in-process streaming solve, which the
                # subprocess child doesn't share with us).  Re-use the
                # post-Layer-2 RangeReport already computed above to
                # synthesize the dict :func:`_autoscale_emit_layer1`
                # consumes, so the per-solve ``autoscale_<solve>.yaml``
                # still lands under ``solve_data/``.
                _ranges_for_l1 = _autoscale_ranges_post or _autoscale_ranges_pre
                # Blank line so HiGHS' "Running HiGHS …" banner (and the
                # grey solver-output block in the GUI) separates from the
                # scaling/LP-build rows above it.  The sub-solve's coded
                # warnings gathered so far print before it.
                run_warnings.flush()
                print("", flush=True)
                # Echo the effective solver options right before HiGHS'
                # own banner: the full merged, post-precedence set the
                # solver actually received (``highs.opt`` floor ∪
                # ``solver_mip_gap`` ∪ DB ``solver_arguments`` ∪ CLI overrides), so the run is
                # replicable from the log alone.  Resolved with
                # ``baseline=None`` so the engine-internal determinism /
                # scale keys are excluded — only what the operator
                # touched is shown.  Uses ``print`` (not the logger) to
                # sit in the same stdout stream as the blank separator
                # above and HiGHS' native banner below — the GUI's
                # ``execution_window`` parses these printed markers.
                # A commercial solve echoes the parameters it passes, under
                # their native names (``Solver args (gurobi): ...``), from the
                # same resolved list ``run_one_solve`` hands to the CLI
                # (``_sa_line`` is computed above, before the scaling guard).
                _print_solver_args(_sa_line)
                _phase_prof("solve_start")
                if _horizon_on:
                    # Before the solve: it releases ``pb._cstrs``.
                    _horizon_keys = snapshot_cstr_keys(pb)
                try:
                    sol = run_one_solve(
                        pb, _active_solver_cfg, logger=state.logger,
                        save_memory=_save_memory,
                        solve_name=complete_solve_name,
                        work_folder=state.paths.work_folder,
                        run_options=run_options,
                        solver_config_dir=state.paths.solver_config_dir,
                        commercial_options=_commercial_opts,
                        solver_option_files=state.paths.solver_option_files,
                        pre_run_check=_autoscale_hook,
                    )
                except BaseException as _solve_exc:
                    # The roll recheck (in ``Problem.solve`` or the
                    # --save-memory ``write_mps``) or the solve after it
                    # stopped: the YAML gets this roll's own verdict.
                    _autoscale_report_recheck_stop(
                        _autoscale_hook, solve_name=complete_solve_name,
                        logger=self.state.logger,
                        work_folder=self.state.paths.work_folder,
                        ranges=_ranges_for_l1, layer2_plan=_autoscale_layer2_plan,
                        scaling_config=_scaling_cfg,
                    )
                    if _bfinal is None:
                        raise
                    from flextool.engine.solve._solver_dispatch import FlexToolSolveOutcomeError
                    from flextool.engine.solve._subprocess_solve import SolveOutcomeError
                    if isinstance(_solve_exc, MemoryError):
                        # Writing the MPS of the whole model ran out of memory.
                        pb = None
                        return self._benders_final_out_of_memory(complete_solve_name, _solve_exc,
                                                                 bres=_bfinal, data=data)
                    if not isinstance(_solve_exc, (SolveOutcomeError, FlexToolSolveOutcomeError)):
                        # A configuration, licence or internal error (an option the solver refuses, a missing
                        # binary or licence, a solution file that does not line up): the run fails, as it
                        # would without Benders.  Only a genuine solve outcome is a failed final solve.
                        raise
                    # The solver ended without a solution (infeasible, unbounded, a limit, its process stopped
                    # or killed): the Benders result stands.
                    return self._benders_final_not_accepted(
                        complete_solve_name, problem="failed (the solver gave no solution)",
                        details=str(_solve_exc), recorded=False, bres=_bfinal, data=data,
                    )
                if _autoscale_hook is not None and _autoscale_hook.called:
                    # The roll's own recheck: report what it used.
                    _autoscale_layer3_plan = _autoscale_hook.plan
                    if _autoscale_hook.outcome is not None:
                        _autoscale_guard = _autoscale_hook.outcome
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("after_solve")
                # Attach the pre-solve ranges as
                # ``streamed_lp_ranges`` on the Solution so the L1
                # emit hook (which expects a dict per polar-high's
                # in-process contract) sees the four (min, max) pairs
                # the solver actually saw — matrix / cost / col_bound /
                # row_bound, matching ``ranges_from_streamed``'s key
                # contract.
                if (
                    _ranges_for_l1 is not None
                    and getattr(sol, "streamed_lp_ranges", None) is None
                ):
                    try:
                        sol.streamed_lp_ranges = {
                            "matrix": _ranges_for_l1.matrix,
                            "cost": _ranges_for_l1.cost,
                            "col_bound": _ranges_for_l1.bound,
                            "row_bound": _ranges_for_l1.rhs,
                        }
                    except Exception:  # pragma: no cover — Solution may
                        # forbid the assignment in a future polar-high
                        pass
                # Eager unscale — restore primal / duals / reduced costs
                # to the un-scaled coordinate so output writers and
                # subsequent rolling iterations see physical values.
                _phase_prof("unscale_start")
                _autoscale_unscale_post_solve(
                    sol, _autoscale_layer2_plan,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                    problem=pb,
                )
                _phase_prof("unscale_done")
            # autoscale Layer 1 (detect) — log the four LP coefficient
            # ranges + trigger flag now that ``streamed_lp_ranges`` is
            # populated.  Detection-only in Phase 1b; Layer 2 / Layer 3
            # actions land in later phases.
            _phase_prof("layer1emit_start")
            _autoscale_emit_layer1(
                sol,
                solve_name=complete_solve_name,
                logger=self.state.logger,
                work_folder=self.state.paths.work_folder
                if self.state.paths is not None else None,
                # The warm path keeps its Layer 2 plan on the runner (it
                # outlives the first build), the cold path in a local.
                layer2_plan=(
                    self._autoscale_warm_layer2_plan if warm_active
                    else locals().get("_autoscale_layer2_plan")
                ),
                layer3_plan=locals().get("_autoscale_layer3_plan"),
                scaling_config=_scaling_cfg,
                guard=locals().get("_autoscale_guard"),
            )
            _phase_prof("layer1emit_done")
            # Memory checkpoint after the solve completes.  Fires on
            # level-boundary iters only; on within-group rolling iters
            # the suppressed deltas accumulate into the next emitted
            # group's lines.
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "solve_end", self.state.logger,
                    user_label="Solver",
                )
            # Emit per-iter lp_build / solve / warm_used rows now that
            # the solve is done and we have valid timestamps from
            # whichever branch ran.  handoff is recorded at end-of-run.
            if _phase_timing:
                _tr.record(
                    "per_iter",
                    subphase="lp_build",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=_t_solve_start - _t_build_start,
                    t_start=_t_build_start,
                )
                _tr.record(
                    "per_iter",
                    subphase="solve",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=_t_solve_end - _t_solve_start,
                    t_start=_t_solve_start,
                )
                _tr.record(
                    "per_iter",
                    subphase="warm_used",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=1.0 if warm_used else 0.0,
                )
            _t_handoff_start = (
                time.perf_counter() if _phase_timing else 0.0
            )
            # Write the ``scale_the_objective.csv`` (consumed by the
            # output writers' un-scaling path) BEFORE the optimality
            # check, so a non-optimal / time-limited solve still leaves
            # behind a coherent solve_data/ directory.
            #
            # Per-base-solve gating: the CSV value is invariant across
            # rolls of the same base solve, so emit it only on the first
            # roll.  The legacy ``scaling_report.txt`` diagnostic (and
            # its ``FLEXTOOL_SCALING_REPORT=1`` gate) was retired in
            # Phase 2b — the autoscaler's per-solve YAML report (written
            # from ``_autoscale_emit_layer1``) is the replacement.
            _t_scale_start = time.perf_counter() if _phase_timing else 0.0
            _write_csv = base_solve_name not in self._scale_csv_written
            if _write_csv:
                _write_scale_csv(
                    solve_data_dir=self.state.paths.work_folder / "solve_data",
                    solve_name=complete_solve_name,
                    effective_obj_scale=effective_obj_scale,
                    logger=self.state.logger,
                )
                self._scale_csv_written.add(base_solve_name)
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="scale_csv_report",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_scale_start,
                    t_start=_t_scale_start,
                )
            # Accept/reject the solve on its actual solver diagnostics, not
            # a bare ``kOptimal`` check.  An interior-point solve run without
            # crossover (the model generator's ``run_crossover`` choice) can
            # return a feasible, in-practice-optimal primal that HiGHS refuses
            # to *certify* (status Unknown) because postsolve left the dual
            # objective slightly inconsistent.  ``classify_acceptance`` accepts
            # that case (using the feasible primal) while still rejecting
            # genuine failures — and reports only what the diagnostics
            # actually show, never a raw-range scaling guess as the cause.
            _acc = classify_acceptance(
                sol,
                ranges_post=locals().get("_autoscale_ranges_post"),
                solve_name=complete_solve_name,
            )
            if _bfinal is not None:
                # The final solve of a Benders solve: recorded beside the Benders verdict, which stays the
                # solve's acceptance (a non-converged Benders solve is not made optimal by its final solve).
                run_record.active().solve_benders_final(
                    accepted=_acc.accepted, model_status=_acc.model_status,
                    objective=(sol.obj / effective_obj_scale if sol.obj is not None else None),
                )
                if not _acc.accepted:
                    return self._benders_final_not_accepted(
                        complete_solve_name, problem=f"failed ({_acc.model_status or 'no solution'})",
                        details=_acc.message, recorded=True, bres=_bfinal, data=data,
                    )
                if sol.obj is not None:
                    from flextool.engine.solve._benders_final import check_final_objective
                    _fobj = sol.obj / effective_obj_scale
                    if not check_final_objective(_fobj, _bfinal.lower_bound, _bfinal.upper_bound):
                        run_warnings.warn(
                            "W-0572", solve=complete_solve_name,
                            problem=(f"objective {_fobj:.4g} outside "
                                     f"[{_bfinal.lower_bound:.4g}, {_bfinal.upper_bound:.4g}]"),
                            details=(f"objective {_fobj!r}, benders lower bound {_bfinal.lower_bound!r}, "
                                     f"upper bound {_bfinal.upper_bound!r}"),
                        )
            else:
                run_record.active().solve_classified(
                    model_status=_acc.model_status,
                    acceptance=(
                        "rejected" if not _acc.accepted
                        else "near_optimal" if _acc.near_optimal
                        else "optimal"
                    ),
                    message=_acc.message,
                    objective=(
                        sol.obj / effective_obj_scale
                        if sol.obj is not None else None
                    ),
                    warm_used=warm_used,
                )
            if not _acc.accepted:
                self.state.logger.error(_acc.message)
                if _acc.scaling_hint:
                    print(_acc.scaling_hint, flush=True)
                return 1
            # The two warnings below go to this module's logger: the runner's
            # own (``self.state.logger``) is held at ERROR during the cascade,
            # so a warning there never reaches the run's output.  The limit
            # warning is already the solve's message in the run record
            # (``solve_classified`` above), so it is not added to the
            # record's warnings again.
            if _acc.limit_stopped:
                logging.getLogger(__name__).warning(
                    _acc.message, extra={run_record.IN_SOLVE_RECORD: True},
                )
            elif _acc.near_optimal:
                self.state.logger.info(_acc.message)
            # One ``Note:`` on what a not-clean HiGHS outcome means for the
            # results (W-0540..W-0542), once per complete solve.  Diagnostic
            # only: a failure in it must not fail a run that has solved.
            try:
                note_solver_outcome(
                    sol, _acc, solve_name=complete_solve_name, solver=_active_solver_cfg.name,
                    noted=self.state.__dict__.setdefault("_solver_outcome_noted", set()),
                    pending=self.state.__dict__.setdefault("_solver_outcome_pending", {}),
                    sub_solve=getattr(self.state, "current_scale_solve_name", None),
                )
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "the solver-outcome note of %s failed", complete_solve_name, exc_info=True)
            if _active_solver_cfg.name == "highs" and _acc.model_status == "kOptimal":
                # A HiGHS MIP treats ``objective_bound`` as a cutoff and calls
                # the best solution within it optimal.  Every roll of a solve
                # has the same options, so the warning names the solve once.
                _bound_solve = re.sub(r"_roll_\d+$", "", complete_solve_name)
                _bound_warned = self.state.__dict__.setdefault(
                    "_objective_bound_warned", set(),
                )
                if _bound_solve not in _bound_warned:
                    _bound_warning = highs_objective_bound_warning(
                        sol, _bound_solve, highs_options,
                    )
                    if _bound_warning is not None:
                        _bound_warned.add(_bound_solve)
                        logging.getLogger(__name__).warning(_bound_warning)
            if _acc.above_fixed_ceiling:
                from flextool.engine import _behaviour_changes
                _behaviour_changes.notice(
                    "acceptance_ceiling_follows_tolerance", "solves", [complete_solve_name],
                )

            prior = prior_for_load
            # ``--csv-dump``: gate ``data.dump_csvs`` behind the
            # orchestrator-level ``csv_dump`` flag.  In default mode the
            # cascade stays in-memory; ``--csv-dump`` materialises the
            # full FlexData → CSV snapshot for debug.
            _t_dump_start = time.perf_counter() if _phase_timing else 0.0
            self._dump_solve_inputs(data, complete_solve_name)
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="dump_csvs",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_dump_start,
                    t_start=_t_dump_start,
                )
            # Emit TIER A ``output_raw`` artefacts BEFORE the in-memory
            # handoff is built.  ``write_all_handoffs`` (called by the
            # adapter) refreshes ``solve_data/period_capacity.csv`` and
            # other handoff CSVs; ``build_handoff_from_solution`` then
            # reads those refreshed files for the in-memory handoff.
            _t_wofs_start = time.perf_counter() if _phase_timing else 0.0
            _phase_prof("write_outputs_start")
            try:
                # Phase G — pass in-memory FlexData and the cascade-known
                # ``is_first_solve`` boolean so handoff writers + extractors
                # can short-circuit ~12 + ~10 per-iter CSV reads (audit:
                # specs/in_memory_carriers_audit.md).  CSV fallback paths
                # remain in place for callers that synthesize a Solution
                # without a FlexData (e.g. unit tests).
                _is_first = (
                    self.state.last_captured_solve is None
                    or len(self.state.handoffs or {}) == 0
                )
                write_outputs_for_solve(
                    sol,
                    work_folder=self.state.paths.work_folder,
                    solve_name=complete_solve_name,
                    prior_handoff=prior,
                    writer_state=writer_state,
                    flex_data=data,
                    is_first_solve=_is_first,
                    scale_the_objective=effective_obj_scale,
                    provider=getattr(
                        self.state, "current_provider", None,
                    ),
                    csv_dump=getattr(self.state, "csv_dump", False),
                    horizon_cstr_keys=_horizon_keys,
                )
            except Exception as exc:  # noqa: BLE001
                # item 163 §3.5 / owner Q3: ERROR, not WARNING — the runner
                # logger is pinned at ERROR (see RunnerState construction),
                # so a WARNING here was invisible and a partial/missing
                # output set went unnoticed. The run still continues.
                self.state.logger.error(
                    f"write_outputs_for_solve failed for "
                    f"{complete_solve_name}: {exc}"
                )
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="write_outputs_for_solve",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_wofs_start,
                    t_start=_t_wofs_start,
                )
            _phase_prof("write_outputs_done")

            # Phase 4 (Gap F) — thread the in-memory FlexData + the
            # upper-level parent handoff so the extractors / fix_storage
            # parent overlay skip their workdir CSV reads where the same
            # data is already in scope.  ``parent_handoff`` is the upper
            # nesting parent (used for fix_storage propagation, deposited
            # by ``_run_model``); ``prior_handoff`` is the
            # sequence predecessor (used for cumulative accumulators).
            parent_complete = getattr(
                self.state, "current_parent_complete", None
            )
            parent_handoff = (
                self.state.handoffs.get(parent_complete)
                if parent_complete is not None
                and self.state.handoffs is not None else None
            )
            _t_bhf_start = time.perf_counter() if _phase_timing else 0.0
            _phase_prof("build_handoff_start")
            handoff = build_handoff_from_solution(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
                flex_data=data,
                parent_handoff=parent_handoff,
                provider=getattr(self.state, "current_provider", None),
                realized_dispatch_parents=set(
                    getattr(self.state.solve, "realized_dispatch_parent", {})
                    .values()
                ),
                scale_the_objective=effective_obj_scale,
            )
            _phase_prof("build_handoff_done")
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="build_handoff_from_solution",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_bhf_start,
                    t_start=_t_bhf_start,
                )
            # The producer's unit level travels with the routed carriers
            # (a later solve that keeps online variables starts cold).
            handoff.unit_detail = _unit_level
            handoff.online_left_out = _gone
            # Deposit so the next iteration's translator picks it up
            # AND we have it for the result dict.
            self.state.handoffs[complete_solve_name] = handoff
            if _bfinal is not None:
                # The final solve's full handoff replaces the Benders one: a later solve may consume it (also
                # when the model has no investment).
                self._benders_invest_handoff_names.add(complete_solve_name)
            # Un-scale the objective value back to user-facing units.
            # ``build_flextool`` multiplied the objective coefficients by
            # ``effective_obj_scale``, so HiGHS reports a scaled value.
            # Overwrite ``sol.obj`` in place so the public
            # ``step.solution.obj`` matches the unscaled ``step.obj`` /
            # the ``v_obj__{solve}.parquet`` value that the legacy
            # writer un-scales via ``_resolve_inv_scale_the_objective``.
            # Without this, callers reading ``step.solution.obj`` see
            # the LP-internal (scaled-by-1e-6) magnitude — a parity
            # break with the legacy flextool objective.
            unscaled_obj = (
                sol.obj / effective_obj_scale if sol.obj is not None else None
            )
            if sol is not None and unscaled_obj is not None:
                sol.obj = unscaled_obj
            # In rolling solves, every iteration's ``complete_solve_name``
            # is the parent solve name (see ``recursive_solves.py:259``:
            # ``complete_solves[roll_name] = complete_solve_name``).  Use
            # the actual per-roll name from ``solve_data/solve_current.csv``
            # — the file flextool rewrites between rolls — so
            # ``_all_steps`` has one entry per roll instead of every roll
            # overwriting the parent key.  ``write_outputs`` keys its
            # union over sub-solves on this dict, and the parquet
            # writers use the same per-roll name (see
            # ``read_highs_solution._actual_solve_name``).
            from flextool.process_outputs.read_highs_solution import (
                _actual_solve_name,
            )
            step_key = _actual_solve_name(
                complete_solve_name,
                provider=getattr(self.state, "current_provider", None),
            )
            # NOTE: the previous "slim PRIOR iter's _vars + highs at
            # the start of iter N" block has been retired.  Both paths
            # now do their slim AFTER ``Outputs written`` below:
            #
            # * Warm path: per-level retention slim (Phase 2) — keeps
            #   one ``Solution.highs`` + one ``flex_data_provider`` per
            #   live level, drops everything else.
            # * Cold path (``_save_memory``): eager prior-iter slim —
            #   nulls everything heavy on the prior step.
            #
            # Both blocks live at the bottom of this method, just after
            # the ``outputs_written_end`` memory checkpoint.
            # Capture per-sub-solve decision-variable frames before the
            # step is deposited.  Polar-high may release ``sol._vars``
            # internally between sub-solves (a memory optimisation on
            # the ``polar-high-opt`` branch), so the LAST iter is the
            # only one whose live ``solution._vars`` survives to
            # end-of-cascade.  End-of-cascade writers
            # (``_entity_all_capacity`` and friends) need EACH sub-
            # solve's own ``v_invest_p`` / ``v_invest_n`` /
            # ``v_divest_p`` / ``v_divest_n`` and the built capacity
            # retired so far ``v_divested_built_p`` / ``v_divested_built_n`` and
            # the extended capacity ``v_extended_p`` / ``v_extended_n``;
            # snapshot those long-form polars frames here so they survive
            # the between-solve release.  Cost: 6 small DataFrames per
            # sub-solve (typically a few hundred rows each).  See
            # :class:`SnapshotSolution` for the wrapper consumers read.
            captured_vars: "dict[str, pl.DataFrame]" = {}
            _phase_prof("captured_vars_start")
            if sol is not None and getattr(sol, "_vars", None):
                for _vname in (
                    "v_invest_p", "v_invest_n",
                    "v_divest_p", "v_divest_n",
                    "v_divested_built_p", "v_divested_built_n",
                    "v_extended_p", "v_extended_n",
                ):
                    if _vname in sol._vars:
                        try:
                            captured_vars[_vname] = sol.value(_vname)
                        except Exception:  # noqa: BLE001
                            # Best-effort: if a variable can't be
                            # materialised here, the end-of-cascade
                            # writer's empty-fallback branch handles
                            # it (see ``_entity_all_capacity._try_value``).
                            pass
            _phase_prof("captured_vars_done")
            # The final solve of a Benders solve overwrites the Benders step of the same name; it keeps the
            # Benders verdict (``optimal`` = converged, no near-optimal) for the CLI exit-code scan.
            _benders_fields = {} if _bfinal is None else {
                "is_benders": True,
                "benders_gap": _bfinal.gap,
                "benders_tol": _bfinal.tol,
                "benders_iterations": _bfinal.iterations,
                "benders_results": "full",
            }
            self._all_steps[step_key] = OrchestrationStep(
                solve_name=step_key,
                solution=sol,
                handoff=handoff,
                obj=unscaled_obj,
                optimal=(
                    bool(_bfinal.converged) if _bfinal is not None
                    else bool(getattr(sol, "optimal", False)) if sol is not None else None
                ),
                # ``_acc`` (from ``classify_acceptance`` above) is authoritative:
                # reaching here means the solve was accepted (a reject returned 1
                # and raised before this point).  Record the near-optimal verdict
                # so the CLI exit-code scan does not re-reject a usable, accepted
                # crossover-off solve off the strict ``optimal`` mirror alone.
                near_optimal=_bfinal is None and bool(getattr(_acc, "near_optimal", False)),
                warm_used=warm_used,
                **_benders_fields,
                flex_data=data,
                flex_data_provider=getattr(
                    self.state, "current_provider", None,
                ),
                captured_vars=captured_vars,
            )
            # Track the just-parked step_key so the next iter can slim
            # THIS iter's Solution (see block above).
            self._prev_step_key = step_key
            # Phase 2 — record this step's level_key for the warm-path
            # per-level slim below.  ``state._current_level_key`` was
            # set by ``_run_model`` immediately before this call.
            _this_level_key = getattr(
                self.state, "_current_level_key", None,
            )
            if _this_level_key is not None:
                self._step_level_keys[step_key] = _this_level_key
            if _phase_timing:
                _tr.record(
                    "per_iter",
                    subphase="handoff",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_handoff_start,
                    t_start=_t_handoff_start,
                )
            # End-of-iter heap trim — release per-roll scratch frames so
            # the next iter's load_flextool doesn't compound on stale heap.
            # No-op on non-glibc.  Cost: ~10-50ms per iter.
            _try_malloc_trim()
            # Memory checkpoint after the per-iter writers + handoff
            # capture finish.  Same gating as the other phase checkpoints:
            # level-boundary iters only.
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "outputs_written_end", self.state.logger,
                    user_label="Outputs written",
                )
            # Phase 2 — warm-path per-level retention slim.  Per the
            # user's design:
            #
            # * Keep ONE ``Solution.highs`` (the live ``polar_high.Solution``
            #   wrapping the WarmProblem's HiGHS instance) per level —
            #   the MOST RECENT parked one of each level.
            # * Keep ONE ``flex_data_provider`` per level — same gating.
            # * Drop both as soon as the pipeline has no more upcoming
            #   solves of that level (``state._all_level_keys[i+1:]``).
            # * Drop ``Solution._vars`` after ``Outputs written`` on
            #   warm too (warm-start uses HiGHS' basis, not these
            #   polars frames).
            # * Drop ``flex_data`` after the solve consumes it.
            #
            # ``flex_data`` and ``solution`` (the Python object, minus
            # the heavy ``.highs`` + ``._vars`` slots) on the LAST step
            # overall survive the per-iter slim because cmd_run_flextool
            # passes them to ``write_outputs``.  We never null the
            # ``step.solution`` object itself here — only its
            # ``_vars`` dict and its ``.highs`` reference.
            #
            # Memory-pressure-yielding for kept Highs instances is a
            # future concern, explicitly out of scope per the user.
            #
            # ``keep_solutions=True`` opts out of per-iter slimming
            # entirely — callers like ``tests/test_scenarios.py`` and
            # any other ``solve_steps``-style end-of-cascade walker
            # union per-step ``flex_data`` + ``solution`` after the
            # cascade returns and would crash on the nulled fields
            # otherwise.
            if not _save_memory and not keep_solutions:
                _all_level_keys = getattr(
                    self.state, "_all_level_keys", ()
                )
                _iter_idx = getattr(
                    self.state, "_current_iter_index", None,
                )
                _upcoming_levels: "set" = set()
                if _iter_idx is not None and _all_level_keys:
                    _upcoming_levels = set(
                        _all_level_keys[_iter_idx + 1:]
                    )
                _this_level = self._step_level_keys.get(step_key)
                # Walk every parked step.  For each, decide whether to
                # keep its ``solution.highs`` + ``flex_data_provider``.
                for _k, _step in self._all_steps.items():
                    _step_lvl = self._step_level_keys.get(_k)
                    _is_just_parked = (_k == step_key)
                    # ``solution._vars`` and ``flex_data`` are dropped on
                    # every PRIOR step regardless of level (the per-iter
                    # writers + handoff carrier already consumed them).
                    if not _is_just_parked and _step.solution is not None:
                        try:
                            _step.solution._vars = {}
                        except Exception:  # noqa: BLE001
                            pass
                    if not _is_just_parked:
                        _step.flex_data = None
                    # ``solution.highs`` + ``flex_data_provider`` are
                    # kept only on the MOST RECENT parked step of each
                    # level whose pipeline still has upcoming iters.
                    # Just-parked step's level always has at least one
                    # member (itself) so the "drop entire level" rule
                    # fires only on PRIOR steps whose level is exhausted.
                    if _is_just_parked:
                        # Even the just-parked step drops its highs +
                        # flex_data_provider when its level has no
                        # more upcoming iters.  Saves the level's last
                        # parked Highs for the duration of subsequent
                        # other-level work that would otherwise pin it.
                        if (
                            _step_lvl is not None
                            and _step_lvl not in _upcoming_levels
                            and _this_level != _step_lvl
                        ):
                            # Can't happen: just-parked step's level
                            # IS ``_this_level``.  Defensive no-op.
                            pass
                        continue
                    # Prior step.  Drop its highs / provider when EITHER:
                    #   (a) the level it belongs to is exhausted
                    #       (``_step_lvl not in _upcoming_levels`` and
                    #       ``_step_lvl != _this_level``), OR
                    #   (b) the level it belongs to is the same as the
                    #       just-parked step's level — in which case
                    #       the just-parked step is the new "most recent
                    #       of this level" and this older sibling is
                    #       superseded.
                    _level_exhausted = (
                        _step_lvl is not None
                        and _step_lvl != _this_level
                        and _step_lvl not in _upcoming_levels
                    )
                    _same_level_older = (
                        _step_lvl is not None
                        and _step_lvl == _this_level
                    )
                    if _level_exhausted or _same_level_older:
                        if _step.solution is not None:
                            try:
                                # Null the WHOLE solution, not just
                                # ``.highs``.  A superseded prior step's
                                # ``polar_high.Solution`` retains the
                                # ``col_names``/``row_names`` lists and the
                                # ``col_value``/``row_dual``/``col_dual``
                                # numpy arrays — all O(LP size) (~1.26 GB
                                # per roll on the real DES).  Nulling only
                                # ``.highs`` released the HiGHS handle but
                                # left those arrays parked for the whole
                                # cascade; dropping the whole object is the
                                # per-roll floor-ratchet release.  Warm
                                # reuse is unaffected — it runs off
                                # ``self._warm_problem``, never off parked
                                # step solutions; and ``handoff`` /
                                # ``captured_vars`` (the only per-step state
                                # later consumers need) live on the step,
                                # not inside ``solution``.
                                _step.solution = None
                            except Exception:  # noqa: BLE001
                                pass
                        _step.flex_data_provider = None
                # Trim the libc heap after potentially dropping multiple
                # large Highs instances + polars frames.
                _try_malloc_trim()
            # Cold-path (save-memory) eager slim of the PRIOR iter's
            # parked OrchestrationStep.  We can't slim the JUST-parked
            # step here — the orchestration cli (``cmd_run_flextool``)
            # passes the LAST step's ``flex_data`` + ``solution`` to
            # :func:`write_outputs`, and from inside the per-iter callback
            # we don't yet know which iter is last.  Slimming the PRIOR
            # iter instead leaves one step's heavy state live at any
            # given time (the just-parked one) and guarantees the LAST
            # step survives the cascade intact.
            #
            # By the time we reach here the PRIOR iter's per-iter
            # consumers all ran on its own iter:
            #
            # * ``write_outputs_for_solve`` (the writers that needed
            #   ``sol.highs.allVariableNames()`` / ``getSolution()``
            #   / ``getLp().row_names_``) — done.
            # * ``build_handoff_from_solution`` — done; carrier stored
            #   in ``step.handoff`` survives this slim.
            # * ``captured_vars`` snapshot — done; lives on
            #   ``step.captured_vars`` independently of ``sol._vars``.
            #
            # On cold the cascade rebuilds the LP from scratch every
            # sub-solve (warm reuse is disabled via the
            # ``_warm_disabled_by_save_memory`` branch at the top of
            # this method), so the prior iter has no further consumer.
            # Drop everything heavy on it — that is the root-cause fix
            # for the cross-solve RSS climb on ``--save-memory`` runs.
            #
            # ``flex_data_provider`` is dropped here by default; set
            # ``FLEXTOOL_COLD_KEEP_PROVIDER=1`` to retain it across the
            # cold-path cascade (trades higher RSS for skipping the
            # per-iter Spine DB re-read).  Real-model measurement at
            # the time of writing did not produce a default-changing
            # signal — the knob exists for workloads where the DB
            # re-read dominates wall time.
            # ``keep_solutions=True`` opts out (see the warm-slim
            # block above for the same rationale — callers that union
            # per-step state after the cascade returns crash on nulled
            # fields).
            if _save_memory and not keep_solutions and self._prev_step_key is not None:
                _prev_step = self._all_steps.get(self._prev_step_key)
                if _prev_step is not None and _prev_step is not self._all_steps.get(step_key):
                    _prev_step.flex_data = None
                    if os.environ.get("FLEXTOOL_COLD_KEEP_PROVIDER") != "1":
                        _prev_step.flex_data_provider = None
                    _psol = _prev_step.solution
                    if _psol is not None:
                        try:
                            _psol._vars = {}
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            _psol.highs = None
                        except Exception:  # noqa: BLE001
                            pass
            return 0

    # Drive the cascade via the native ``native_run_model``.  Native
    # emitters thread ``sub_solve_provider`` through every emit_* call.
    _cascade_solver = _PolarHighCascadeSolver(runner.state)
    try:
        native_run_model(runner.state, _cascade_solver)
    finally:
        # The last rolling solve's gathered solver-outcome note (W-0540).
        _cascade_solver.flush_solver_outcome_notes()
    # Mirror the in-memory handoff dict back onto our state in case
    # callers want to inspect it.
    state.handoffs = runner.state.handoffs

    # Phase C.5 — slim every step except the LAST, releasing the heaviest
    # per-step state (Solution + FlexData + FlexDataProvider) once
    # downstream consumers (handoff extraction, raw-output write) have
    # run for that sub-solve.  ``keep_solutions=True`` opts out — used
    # by tests that need per-step ``solution`` / ``flex_data`` access.
    if not keep_solutions and results:
        last_key = next(reversed(results))
        for k, step in results.items():
            if k == last_key:
                continue
            step.solution = None
            step.flex_data = None
            step.flex_data_provider = None
        # Free the HiGHS heap once the per-step references are gone.
        # Cheap and a no-op on non-glibc.
        _try_malloc_trim()
    return results
