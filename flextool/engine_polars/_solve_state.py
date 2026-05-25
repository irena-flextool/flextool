"""Solve-orchestration state types.

Foundation module: every downstream orchestration module (timeline,
recursive solve, stochastic, orchestration loop) needs the exception
classes, ``ActiveTimeEntry`` namedtuple, ``SolveResult`` dataclass and
a slim ``RunnerState`` carrier.  Runs natively on HiGHS via
``polar_high``; per-CLI flags live on the CLI wrapper rather than in
shared state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from flextool.cli._timing import TimingRecorder
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._solve_handoff import SolveHandoff


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class FlexToolError(Exception):
    """Base exception for FlexTool runner errors."""


class FlexToolConfigError(FlexToolError):
    """Raised for configuration / input data errors."""


class FlexToolSolveError(FlexToolError):
    """Raised for solver execution errors."""


# ---------------------------------------------------------------------------
# Lightweight value types
# ---------------------------------------------------------------------------


class ActiveTimeEntry(NamedTuple):
    """A single timestep in an active time list.

    Backwards-compatible with the previous ``(timestep, index, duration)``
    tuples — ``entry[0]`` still works, but ``entry.timestep`` is preferred.
    """

    timestep: str
    index: int
    duration: str


@dataclass
class SolveResult:
    """Result container for the recursive solve structure builder.

    Populated by the recursive solve builder (Γ.8.C) and consumed by the
    orchestration loop (Γ.8.D).
    """

    solves: list = field(default_factory=list)
    complete_solves: dict = field(default_factory=dict)
    active_time_lists: dict = field(default_factory=dict)
    fix_storage_time_lists: dict = field(default_factory=dict)
    realized_time_lists: dict = field(default_factory=dict)
    parent_roll_lists: dict = field(default_factory=dict)


@dataclass
class PathConfig:
    """Directory layout for a FlexTool run.

    Carries the work folder plus the optional ancillary directories
    :class:`FlexToolRunner` resolves (a package data dir, a
    CLI-overrideable ``solver_config/`` for ``highs.opt`` and a project
    root).  Native engine_polars callers only populate ``work_folder``;
    the rest are ``None``.
    """

    work_folder: Path
    output_path: Path | None = None
    flextool_dir: Path | None = None
    solver_config_dir: Path | None = None
    root_dir: Path | None = None


@dataclass
class RunnerState:
    """Cross-cutting state for a native polar_high solve run.

    Only the fields needed by Γ.8.A are populated.  Timeline + handoff
    fields will be filled in Γ.8.B / Γ.8.D as those modules land.
    """

    paths: PathConfig
    solve: "SolveConfig"
    logger: logging.Logger
    # Filled by Γ.8.B (timeline module).  Typed as ``object`` so that
    # importing :class:`RunnerState` doesn't pull a non-existent module
    # into the import graph.
    timeline: object | None = None
    # Agent 8 (LP-scaling): opt-in flag — when True the Python
    # ScaleAnalyzer's recommendations are auto-applied (currently only
    # to ``solve.use_row_scaling`` when the user hasn't set it).
    # Always-False in the default path preserves pre-Agent-8 behaviour.
    auto_scale: bool = False
    # Roll-loop scratch — set by the orchestration loop just before
    # ``solver.run`` so per-roll diagnostics / handoff capture can find
    # the correct row.  ``None`` outside an active solve iteration.
    current_roll_index: int | None = None
    # Agent 18c (LP-scaling): the orchestration loop sets this to the
    # ``ScaleTable`` for the currently-active solve just before calling
    # ``solver.run``.  ``_run_highs`` uses it to update bound-scaling
    # diagnostics in the right cache entry even when the roll name
    # differs from the parent (complete) solve name passed to the
    # solver.  ``None`` outside an active solve iteration.
    current_scale_solve_name: str | None = None
    # Name of the most-recent solve whose post-solve hook deposited a
    # ``SolveHandoff`` into ``handoffs``.  Set by ``orchestration.run_model``
    # after each capture; consulted by post-solve writers (e.g. the
    # cumulative-handoff writers in ``solver_runner._run_highs``) to
    # source prior-roll state from the in-memory dict instead of disk.
    # ``None`` outside an active solve loop and on the first solve of
    # any loop.
    last_captured_solve: str | None = None
    # In-memory solve-to-solve handoff.  ``None`` keeps file-based
    # behaviour; opt-in by setting ``state.handoffs = {}``.  See
    # ``audit/handoff_csv_retirement.md`` for the migration plan.
    handoffs: "dict[str, SolveHandoff] | None" = None
    # Phase 5b — external override provider.  When set, the runner
    # invokes this callable at iteration start (after the sequential
    # + parent handoff translators) and fans the returned dict into
    # the ``override/*`` Provider namespace via
    # :func:`flextool.engine_polars._provider_translators.translate_overrides_to_provider`.
    # The callable is owned by external code wrapping the runner
    # (e.g. file-watch, ZeroMQ bridge); ``None`` means no overrides.
    override_provider: Callable[[], "dict[str, pl.DataFrame]"] | None = None
    # Per-CLI-invocation phase timing recorder.  The CLI constructs one
    # very early in ``cmd_run_flextool.main`` and assigns it onto
    # ``state.timing_recorder``; callers using :class:`FlexToolRunner`
    # directly (without going through the CLI) bootstrap their own in
    # ``FlexToolRunner.__init__``.  Always non-None inside an active run.
    timing_recorder: "TimingRecorder | None" = None
    # HiGHS thread count (CLI override; solver_runner defaults to 4 when None).
    highs_threads: int | None = None
    # Gates ``data.dump_csvs`` from inside the cascade — set by
    # ``run_orchestration`` from its ``csv_dump`` argument.
    csv_dump: bool = False
    # Seeded cascade-input Provider — set by ``write_input`` / by the
    # orchestration entry point so per-sub-solve Providers can clone
    # the ``input/<class>`` frames.  Typed as ``object`` to avoid
    # pulling :class:`FlexDataProvider` into the import graph.
    cascade_input_provider: object | None = None
    # Per-sub-solve Provider currently driving the cascade.  Set by
    # ``_native_run_model`` immediately before each ``solver.run``
    # invocation; consumed by post-solve writers that need a Provider
    # handle but were called without one.  ``None`` outside an active
    # solve iteration.
    current_provider: object | None = None
    # Per-level Provider cache — keyed by :func:`compute_level_key`.
    # The orchestration loop populates this so sub-solves at the same
    # "level" (matching LP matrix shape) share a :class:`FlexDataProvider`.
    # Typed as ``dict | None``; ``None`` means "not yet initialised".
    _level_providers: dict | None = None


# ---------------------------------------------------------------------------
# Per-level Provider — level-key helper (Design A, step A1).
# ---------------------------------------------------------------------------


def compute_level_key(
    *,
    solve_name: str,
    complete_solve_name: str,
    solve_config,
    timeline_config,
) -> tuple:
    """Compute a cheap level identifier for a sub-solve.

    Two sub-solves with the same level_key share LP matrix shape and
    can share a :class:`FlexDataProvider` (per the user's per-level
    Provider intent).  Two sub-solves with different keys must not
    share.

    Composition (in order):

    1. Tuple of timesets used by ``complete_solve_name``
       (``solve_config.timesets_used_by_solves[complete_solve_name]``,
       sorted for determinism).
    2. ``timeline_config.new_step_durations.get(complete_solve_name)``
       — the explicit step-size override (1h, 3h, …).  ``None`` when
       absent.
    3. ``solve_config.rolling_times.get(complete_solve_name)`` — the
       rolling-window triple ``[jump, horizon, duration]``, coerced to
       a tuple.  ``None`` when not rolling.
    4. ``solve_config.solve_modes.get(complete_solve_name)``.

    Returns a hashable tuple.  Same key on consecutive iterations
    means "same level — reuse Provider"; different key means "level
    transition — fresh Provider".
    """
    timesets = solve_config.timesets_used_by_solves.get(
        complete_solve_name, ()
    )
    timesets = tuple(sorted(timesets)) if timesets else ()
    step_dur = timeline_config.new_step_durations.get(complete_solve_name)
    rolling = solve_config.rolling_times.get(complete_solve_name)
    if rolling is not None:
        rolling = tuple(rolling)
    mode = solve_config.solve_modes.get(complete_solve_name)
    return (timesets, step_dur, rolling, mode)


__all__ = [
    "FlexToolError",
    "FlexToolConfigError",
    "FlexToolSolveError",
    "ActiveTimeEntry",
    "SolveResult",
    "PathConfig",
    "RunnerState",
    "compute_level_key",
]
