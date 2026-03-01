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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.flextoolrunner.solve_config import SolveConfig
    from flextool.flextoolrunner.timeline_config import TimelineConfig


@dataclass
class PathConfig:
    """Directory layout for a FlexTool run."""
    flextool_dir: Path
    bin_dir: Path
    root_dir: Path
    output_path: Path


@dataclass
class RunnerState:
    """Cross-cutting state shared across all FlexTool runner modules."""
    paths: PathConfig
    solve: SolveConfig
    timeline: TimelineConfig
    logger: logging.Logger
