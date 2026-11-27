"""The HiGHS options of a sub-solve: the ``Solver args`` echo, the objective scale, the baseline, operator and
per-solve option sets with where each was set, and the warm-start basis cache gate and directory."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from flextool import run_warnings
from flextool.run_options import RunOptions
from flextool._thread_budget import highs_thread_options
from flextool.engine.solve import _objective_options
from flextool.engine.input._solve_state import RunnerState
from flextool.engine.solve._determinism import (
    DETERMINISM_OPTIONS,
    SIMPLEX_SCALE_STRATEGY_ADVANCED,
)
from flextool.engine.autoscale import (
    ScalingMode as _AutoscaleScalingMode,
    USER_BOUND_SCALE_MAX as _USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN as _USER_BOUND_SCALE_MIN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from polar_high import Problem


def _format_solver_args_line(
    effective_options: Mapping[str, Any] | None,
    *,
    prefer_first: "Sequence[str]" = (),
    solver: str | None = None,
    translated: "Mapping[str, str] | None" = None,
) -> str | None:
    """Render the effective solver options as a one-liner.

    Emitted just before HiGHS' own "Running HiGHS …" banner so the run
    log shows exactly the options the solver received — the full merged,
    post-precedence set of everything the operator touched:
    ``solver_config/highs.opt`` (project floor) ∪ the solve's
    ``solver_mip_gap`` ∪ the DB-authored ``solver_arguments`` ∪ the
    CLI-flag overrides.  Engine-pinned
    *baseline* options (determinism keys + the Curtis-Reid scale) are
    deliberately excluded — they are internal and always present, so
    listing them would bury the operator's intent.  Pass
    ``effective_options`` already resolved with ``baseline=None`` so this
    exclusion holds (see :func:`_resolve_effective_highs_options`).

    *prefer_first* names keys (typically the ``solver_arguments`` keys)
    to surface at the front of the line; the remaining keys follow in
    ``effective_options`` order.  Each key's value is the winning one
    after precedence resolution.

    Returns ``None`` when *effective_options* is empty (nothing the
    operator set → no line).

    A commercial solve passes *solver*: the line then reads
    ``Solver args (<solver>): ...`` and lists the native parameters the
    CLI receives (option file included, since it has no engine baseline).

    *translated* maps an objective-valued option (written in unscaled
    currency) to its text from :func:`_objective_options.echo_values`: the
    value the solver gets, then the value the user wrote in parentheses.
    """
    if not effective_options:
        return None
    translated = translated or {}
    order = [k for k in prefer_first if k in effective_options]
    order += [k for k in effective_options if k not in order]
    body = ", ".join(f"{k}={translated.get(k, effective_options[k])}" for k in order)
    label = f"Solver args ({solver})" if solver else "Solver args"
    return f"{label}: {body}"


#: The last ``Solver args`` line printed in this process's current run (:func:`_print_solver_args`).
_SOLVER_ARGS_PRINTED: dict[str, str | None] = {"line": None}


def _reset_solver_args_echo() -> None:
    """A new run prints its first ``Solver args`` line again."""
    _SOLVER_ARGS_PRINTED["line"] = None


def _print_solver_args(line: str | None) -> None:
    """Right before a solver runs: print the coded warnings gathered so far for the sub-solve
    (:func:`flextool.run_warnings.flush`, so they precede the solver's output), then the ``Solver args`` line —
    once per run, and again only when it changes (the rolls and solves of a run mostly share their options).
    The line stays whole (it may pass 100 characters: the GUI wraps it), so it can be copied as is."""
    run_warnings.flush()
    if not line or line == _SOLVER_ARGS_PRINTED["line"]:
        return
    _SOLVER_ARGS_PRINTED["line"] = line
    print(line, flush=True)


def _log_cold_rebuild(logger: logging.Logger, logged: set, solve: str, reason: str) -> None:
    """The warm LP reuse of a roll fell back to a cold rebuild (*reason*: the first FlexData field that differs).
    The normal outcome for every roll of a rolling solve (``dt`` shifts), so one INFO line per complete solve and
    reason; the other rolls go to the debug log."""
    base = run_warnings.complete_solve(solve)
    key = (base, reason)
    if key in logged:
        logger.debug("warm reuse fell back to a cold rebuild for %s: %s", solve, reason)
        return
    logged.add(key)
    logger.info("No warm LP reuse in solve '%s': %s", base, reason)


def _highs_echo_translations(
    problem: "Problem", echoed: "Mapping[str, Any] | None", solve_name: str,
) -> dict[str, str]:
    """The ``Solver args`` text of the objective-valued HiGHS options in *echoed* (the user's layers): the value
    HiGHS compares (``scale_the_objective`` × ``2**user_objective_scale`` of the options applied to *problem*,
    autoscale Layer 3 included) and the value written.  polar-high itself translates them for the solve."""
    _, records = _objective_options.translate_for_solver(
        problem, "highs", "direct", echoed, solve_name=solve_name, factor_from_problem=True,
    )
    return _objective_options.echo_values(records)


# Legacy ``scale_the_objective`` default — historically the
# ``flextool_base.dat``'s ``param scale_the_objective default 1e-6`` and the
# pre-autoscale analyser's fallback when the rough-objective heuristic
# returned a non-finite value.  Phase 2b drops the legacy data-driven
# analyser entirely; we keep the build-time cost multiplier so that:
#
#   * The MPS export (post-Layer-2, pre-solve) still carries pre-scaled
#     cost coefficients in the same magnitude as before — handoff to
#     external solvers continues to see a recognisable objective.
#   * The output writers' un-scaling path (``_resolve_inv_scale_the_objective``)
#     continues to find ``solve_data/scale_the_objective.csv`` with a non-1
#     value; missing-file fallback in that helper is also ``1.0 / 1e-6``,
#     so the un-scaling round-trip is byte-stable.
#   * The user can still override per-solve via DB
#     ``solve.scale_the_objective`` — autoscale Layer 3's
#     ``user_objective_scale`` is HiGHS-internal and stacks on top
#     (HiGHS un-scales internally on output, so layering the build-time
#     scale and the HiGHS-side scale is well-defined).
_LEGACY_DEFAULT_OBJECTIVE_SCALE = 1e-6


def _resolve_effective_obj_scale(user_value: object | None) -> float:
    """Coerce a raw DB ``solve.scale_the_objective`` value to a float.

    Returns the user value when finite and strictly positive; otherwise
    falls back to :data:`_LEGACY_DEFAULT_OBJECTIVE_SCALE` (1e-6).  This
    mirrors the defensive contract the retired
    ``scaling.resolve_effective_scaling`` had: any malformed user value
    (None, non-numeric string, 0, NaN, negative) falls back rather than
    crashing the cascade — the failure mode would be HiGHS's own
    division by zero on output un-scaling.
    """
    if user_value is None:
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    try:
        candidate = float(user_value)
    except (TypeError, ValueError):
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    if not math.isfinite(candidate) or candidate <= 0.0:
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    return candidate


def _baseline_highs_options(
    *,
    user_bound_scale_override: int | None = None,
    scaling_mode: "_AutoscaleScalingMode | None" = None,
) -> dict[str, object]:
    """Build the base HiGHS solver-option dict (determinism + matrix scale).

    Replaces the retired ``scaling.recommended_highs_options`` helper.
    Sets:

    * ``simplex_scale_strategy`` — defaults to
      :data:`SIMPLEX_SCALE_STRATEGY_ADVANCED` (Curtis-Reid matrix
      equilibration).  When ``scaling_mode == ScalingMode.OFF`` we force
      ``simplex_scale_strategy=0`` so HiGHS' own equilibration is also
      disabled — that's the only mode in which FlexTool touches the
      HiGHS-internal scaling knob.  Layer 3's :func:`apply_scaling` may
      re-assert this value on the cold path; on warm rebuilds the value
      set here is the authoritative one.
    * :data:`DETERMINISM_OPTIONS` — ``random_seed`` / ``parallel`` /
      ``solver`` / ``presolve`` pins for byte-deterministic LP solutions.
    * ``user_bound_scale`` — only when ``user_bound_scale_override`` is
      a non-zero integer (``--user-bound-scale N`` /
      ``FLEXTOOL_USER_BOUND_SCALE``).  Clamped to the HiGHS-safe range
      ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.  When unset, the
      autoscaler's Layer 3 may still emit its own value based on the
      coefficient ranges.

    ``user_cost_scale`` is intentionally NOT set — costs are already
    multiplied by ``scale_the_objective`` inside ``build_flextool``, and
    Layer 3 may add ``user_objective_scale`` on top; layering a third
    cost-side knob would compound confusingly.
    """
    if scaling_mode is _AutoscaleScalingMode.OFF:
        # OFF mode: disable HiGHS' internal equilibration too.  ``parallel``
        # / ``random_seed`` / ``solver`` / ``presolve`` pins from
        # DETERMINISM_OPTIONS still apply — OFF is about scaling, not
        # determinism.
        simplex_scale = 0
    else:
        simplex_scale = SIMPLEX_SCALE_STRATEGY_ADVANCED
    options: dict[str, object] = {
        "simplex_scale_strategy": simplex_scale,
        **DETERMINISM_OPTIONS,
    }
    if user_bound_scale_override is not None and user_bound_scale_override != 0:
        n = int(user_bound_scale_override)
        if n > _USER_BOUND_SCALE_MAX:
            n = _USER_BOUND_SCALE_MAX
        if n < _USER_BOUND_SCALE_MIN:
            n = _USER_BOUND_SCALE_MIN
        options["user_bound_scale"] = n
    return options


def _run_options_or_env(run_options: "RunOptions | None") -> RunOptions:
    """*run_options*, or the environment's options for a caller that passes none (library call, test)."""
    return run_options if run_options is not None else RunOptions.from_env()


# A --highs-threads above --threads is reported once per process, not once per solve.
_THREAD_WARNINGS_SHOWN: set[str] = set()


def _operator_highs_overrides(run_options: "RunOptions | None" = None) -> dict[str, object]:
    """HiGHS options the operator set for this run, by flag or environment variable.

    The top layer of :func:`_solve_highs_options`.  Only options whose source is a flag or an environment variable
    contribute; a registry default adds nothing, so an unset option leaves the lower layers in charge.  ``None``
    reads the environment (:meth:`RunOptions.from_env`).

    * ``--solver-time-limit S`` (``FLEXTOOL_SOLVER_TIME_LIMIT``, else the HiGHS-only
      ``FLEXTOOL_HIGHS_TIME_LIMIT``) → ``time_limit``
    * ``--solver-mip-gap GAP`` (``FLEXTOOL_SOLVER_MIP_GAP``, else the HiGHS-only
      ``FLEXTOOL_HIGHS_MIP_GAP``) → ``mip_rel_gap`` (only bites MIP solves)
    * ``--presolve {on,off,choose}`` (``FLEXTOOL_HIGHS_PRESOLVE``) → ``presolve``
    * ``--threads B`` (``FLEXTOOL_THREADS``) and ``--highs-threads N``
      (``FLEXTOOL_HIGHS_THREADS``), by
      :func:`flextool._thread_budget.highs_thread_options`: N > 1 →
      ``threads=N`` and ``parallel=on`` (trades determinism for wall-clock
      speed), N at most B; B alone → ``threads=B`` with ``parallel`` left to
      the DETERMINISM_OPTIONS pin (off); neither → nothing.  Every Highs
      instance of a sub-solve sees the same value, which sidesteps HiGHS'
      "global scheduler already initialised" rejection.
    * ``--solver-log-level {silent,normal,verbose}``
      (``FLEXTOOL_SOLVER_LOG_LEVEL``): ``silent`` → ``output_flag=False``,
      ``verbose`` → ``output_flag=True`` + ``log_dev_level=2``, ``normal`` →
      ``output_flag=True``.

    A malformed environment value is ignored with one warning (:func:`flextool.run_options.resolve`).
    """
    opts = _run_options_or_env(run_options)

    def given(key: str):
        resolved = opts.get(key)
        return resolved.value if resolved.source in ("cli", "env") else None

    cli: dict[str, object] = {}
    tlim = given("solver_time_limit")
    if tlim is None:
        tlim = given("highs_time_limit")
    if tlim is not None:
        cli["time_limit"] = float(tlim)
    mip_gap = given("solver_mip_gap")
    if mip_gap is None:
        mip_gap = given("highs_mip_gap")
    if mip_gap is not None:
        cli["mip_rel_gap"] = float(mip_gap)
    presolve = given("presolve")
    if presolve is not None:
        cli["presolve"] = presolve
    thread_options, thread_warning = highs_thread_options(opts)
    cli.update(thread_options)
    if thread_warning is not None and thread_warning not in _THREAD_WARNINGS_SHOWN:
        _THREAD_WARNINGS_SHOWN.add(thread_warning)
        logging.getLogger(__name__).warning(thread_warning)
    log_level = given("solver_log_level")
    if log_level == "silent":
        cli["output_flag"] = False
    elif log_level == "verbose":
        cli["output_flag"] = True
        cli["log_dev_level"] = 2
    elif log_level == "normal":
        cli["output_flag"] = True
    return cli


def _solve_highs_options(
    state: "RunnerState",
    complete_solve_name: str,
    *,
    baseline: "Mapping[str, Any] | None",
) -> dict[str, object]:
    """The HiGHS option set of one sub-solve, the same on every solve path.

    Layers, lowest → highest (:func:`_resolve_effective_highs_options`):
    ``baseline`` (engine determinism pins + scale; ``None`` for the echoed
    "Solver args" line and for Benders) < ``solver_config/highs.opt`` <
    the solve's convenience knobs (``solve.solver_mip_gap``) < the solve's
    ``solver_arguments`` < the operator's flags / environment variables
    (:func:`_operator_highs_overrides`, from ``state.run_options``).  The two
    database layers take no paths or server options
    (:func:`~flextool.engine.solve._solver_dispatch.check_database_solver_option`).
    ``run_chain_from_db`` reads the run's option files once at its start
    (``state.paths.solver_option_files``); a bare ``PathConfig`` reads
    ``<solver_config_dir>/highs.opt``, or has no highs.opt layer when that is
    None.
    """
    from flextool.engine.solve._solver_dispatch import (
        _resolve_effective_highs_options,
        solver_workspace,
    )

    solver_config = state.solve.solver_config_for(complete_solve_name)
    # A solve on a commercial solver has its solver_arguments in that solver's names, checked against that
    # solver's list (resolve_commercial_options); they are not HiGHS options (item 240 B1).
    solver_arguments = (
        state.solve.solver_settings.arguments.get(complete_solve_name, {}) if solver_config.name == "highs" else {}
    )
    files = state.paths.solver_option_files
    highs_opt_text = files.text("highs") if files is not None else None
    if files is not None:
        highs_opt_path = files.origin("highs") if highs_opt_text is not None else None
    elif state.paths.solver_config_dir is not None:
        highs_opt_path = state.paths.solver_config_dir / "highs.opt"
    else:
        highs_opt_path = None
    options = _resolve_effective_highs_options(
        highs_opt_text=highs_opt_text,
        solver_arguments_map=solver_arguments,
        highs_opt_path=highs_opt_path,
        cli_overrides=_operator_highs_overrides(state.run_options),
        baseline=baseline,
        solver_config=solver_config,
        solve_name=complete_solve_name,
        # A plain file name the database gives a HiGHS file option lands in the solve's work folder.
        workspace=solver_workspace(state.paths.work_folder),
        # A highs.opt found outside a workspace this computer set up gets the database's rules (item 240 B3).
        highs_opt_untrusted=files.untrusted_note("highs") if files is not None and highs_opt_text is not None else None,
    )
    # Every HiGHS path (in-process cold and warm, --save-memory, Benders) takes its options from here: refuse an
    # objective-valued option the installed polar-high cannot scale here, not in a later echo or option-file write.
    _objective_options.require_supported("highs", options, solve_name=complete_solve_name)
    return options


def _highs_option_layers(
    state: "RunnerState", complete_solve_name: str, options: "Mapping[str, Any] | None",
) -> dict[str, str]:
    """Where each of the solve's user HiGHS *options* was set, for a message: the solve's
    ``solver_arguments``, a flag / environment variable, else ``highs.opt`` (the lowest layer; also the
    layer reported for ``mip_rel_gap`` set by ``solve.solver_mip_gap``)."""
    solver_arguments = state.solve.solver_settings.arguments.get(complete_solve_name, {}) or {}
    operator = _operator_highs_overrides(state.run_options)
    files = state.paths.solver_option_files
    if files is not None:
        opt_label = str(files.origin("highs"))
    elif state.paths.solver_config_dir is not None:
        opt_label = str(state.paths.solver_config_dir / "highs.opt")
    else:
        opt_label = "highs.opt"
    layers: dict[str, str] = {}
    for name in options or {}:
        if name in operator:
            layers[name] = "command-line flag or environment variable"
        elif name in solver_arguments:
            layers[name] = f"solver_arguments of solve {complete_solve_name}"
        else:
            layers[name] = opt_label
    return layers


def _basis_cache_active(
    decomposition: "str | None", solver_name: str, run_options: "RunOptions | None" = None,
) -> bool:
    """Gate for the Phase-4 in-process warm-start basis cache.

    True only when (a) the operator opted in with ``--warm-start`` /
    ``FLEXTOOL_WARM_START=1`` (``run_options``; ``None`` reads the environment),
    (b) the in-process solver is HiGHS — the only solver whose live handle
    supports ``setBasis`` / ``getBasis`` for basis transfer, and (c) the
    active solve is NOT Benders-decomposed.  A Benders solve never builds a
    monolithic ``WarmProblem`` (it returns early through
    ``_run_benders_solve`` before the warm branch), and its region
    subproblems are not keyed by this cache; the ``decomposition`` check is
    a belt-and-suspenders guard on top of that early return.

    When this returns False the caller does nothing new, so the off-path
    (and every non-HiGHS / Benders path) stays byte-identical.
    """
    return (
        bool(_run_options_or_env(run_options).value("warm_start"))
        and solver_name == "highs"
        and decomposition != "benders"
    )


def _basis_cache_dir(
    work_folder: "Path | str | None", run_options: "RunOptions | None" = None,
) -> Path:
    """Resolve the shared warm-start basis cache directory (Phase-2 pattern).

    The same resolution as the subprocess ``.bas`` arm
    (:func:`flextool.engine.solve._subprocess_solve.resolve_basis_cache_dir`), so the in-process ``.nbasis`` files
    land alongside their ``.bas`` counterparts.
    """
    from flextool.engine.solve._subprocess_solve import resolve_basis_cache_dir

    return resolve_basis_cache_dir(work_folder, _run_options_or_env(run_options))
