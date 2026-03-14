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
