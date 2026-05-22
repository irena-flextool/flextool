"""Solve-orchestration state types ported from flextool's runner_state.py.

This is the foundation module for Γ.8.A (see ``audit/solve_orchestration_plan.md``
§3.1, §4): every downstream orchestration module (timeline, recursive solve,
stochastic, orchestration loop) needs the exception classes,
``ActiveTimeEntry`` namedtuple, ``SolveResult`` dataclass and a slim
``RunnerState`` carrier.

Direct 1:1 port of the relevant types from
``flextool/flextoolrunner/runner_state.py`` (lines 22-133).  Fields that are
specific to the legacy orchestrator (file-source-only flags, mod-side
phase capture, glpsol pathways) are intentionally absent here — flexpy
runs natively on HiGHS via ``polar_high`` and the per-CLI flags live
on the CLI wrapper rather than in shared state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from flextool.engine_polars._solve_config import SolveConfig


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

    Mirrors :class:`flextool.flextoolrunner.runner_state.SolveResult`.
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
    """Directory layout for a flexpy native run.

    Slimmer than flextool's :class:`PathConfig`:
    flexpy doesn't use a separate ``flextool_dir``/``bin_dir`` because the
    AMPL/GLPK pathway isn't reachable from the engine_polars stack.
    """

    work_folder: Path
    output_path: Path | None = None


@dataclass
class RunnerState:
    """Cross-cutting state for a flexpy native solve run.

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
    # Roll-loop scratch — set by the orchestration loop just before
    # ``solver.run`` so per-roll diagnostics / handoff capture can find
    # the correct row.  ``None`` outside an active solve iteration.
    current_roll_index: int | None = None
    current_scale_solve_name: str | None = None
    last_captured_solve: str | None = None
    # In-memory solve-to-solve handoff.  ``None`` keeps file-based
    # behaviour; opt-in by setting ``state.handoffs = {}``.  See
    # ``audit/handoff_csv_retirement.md`` for the migration plan.
    handoffs: dict | None = None
    # Phase 5b — external override provider.  When set, the runner
    # invokes this callable at iteration start (after the sequential
    # + parent handoff translators) and fans the returned dict into
    # the ``override/*`` Provider namespace via
    # :func:`flextool.engine_polars._provider_translators.translate_overrides_to_provider`.
    # The callable is owned by external code wrapping the runner
    # (e.g. file-watch, ZeroMQ bridge); ``None`` means no overrides.
    override_provider: Callable[[], "dict[str, pl.DataFrame]"] | None = None


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
