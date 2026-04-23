"""
RunnerState — lightweight container types for cross-cutting state.

PathConfig holds directory paths.  RunnerState bundles PathConfig together
with SolveConfig, TimelineConfig and the logger so that downstream modules
can receive a single object instead of many individual parameters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.flextoolrunner.solve_config import SolveConfig
    from flextool.flextoolrunner.timeline_config import TimelineConfig


class FlexToolError(Exception):
    """Base exception for FlexTool runner errors."""
    pass


class FlexToolConfigError(FlexToolError):
    """Raised for configuration/input data errors."""
    pass


class FlexToolSolveError(FlexToolError):
    """Raised for solver execution errors."""
    pass


class ActiveTimeEntry(NamedTuple):
    """A single timestep in an active time list.

    Backwards-compatible with the previous (timestep, index, duration) tuples:
    entry[0] still works, but entry.timestep is preferred for clarity.
    """
    timestep: str
    index: int
    duration: str


@dataclass
class SolveResult:
    """Result container for the recursive solve structure builder."""
    solves: list
    complete_solves: dict
    active_time_lists: dict
    fix_storage_time_lists: dict
    realized_time_lists: dict
    parent_roll_lists: dict


@dataclass
class PathConfig:
    """Directory layout for a FlexTool run."""
    flextool_dir: Path
    bin_dir: Path
    root_dir: Path
    output_path: Path
    work_folder: Path


@dataclass
class RunnerState:
    """Cross-cutting state shared across all FlexTool runner modules."""
    paths: PathConfig
    solve: SolveConfig
    timeline: TimelineConfig
    logger: logging.Logger
    # When True, keep only the legacy glpsol-phase-3 CSV output pathway
    # (``output_raw/*.csv``) and skip the HiGHS → parquet extractor.  When
    # False (default) the parquet extractor runs alongside glpsol phase 3
    # during the PoC migration; glpsol phase 3 will itself be made optional
    # once variable coverage is complete.
    use_old_raw_csv: bool = False
    # Agent 8 (LP-scaling): opt-in flag — when True the Python
    # ScaleAnalyzer's recommendations are auto-applied (currently only
    # to ``solve.use_row_scaling`` when the user hasn't set it).
    # Always-False in the default path preserves pre-Agent-8 behaviour.
    auto_scale: bool = False
    # Agent 18c (LP-scaling): the orchestration loop sets this to the
    # ``ScaleTable`` for the currently-active solve just before calling
    # ``solver.run``.  ``_run_highs`` uses it to update bound-scaling
    # diagnostics in the right cache entry even when the roll name
    # differs from the parent (complete) solve name passed to the
    # solver.  ``None`` outside an active solve iteration.
    current_scale_solve_name: str | None = None
    # Agent 18d (LP-scaling): user-facing solver-option knobs.
    # ``relax_feasibility`` is the explicit tolerance (e.g. ``1e-5``)
    # applied to HiGHS' primal + dual feasibility tolerance when the
    # user passed ``--relax-feasibility``; ``None`` means "keep HiGHS
    # defaults".  ``use_ipm`` switches HiGHS to interior-point when
    # True.  Both are set by the CLI layer; neither has DB precedence.
    relax_feasibility: float | None = None
    use_ipm: bool = False
