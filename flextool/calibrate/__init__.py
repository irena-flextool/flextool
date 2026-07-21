"""FlexTool energy-margin calibrator package.

The calibrator drives an investment+dispatch solve each iteration by
shelling out to :mod:`flextool.cli.cmd_run_flextool`, then reads the
per-node unserved-energy slack from the produced outputs to decide how to
adjust the energy margin.  This package hosts that loop (the ``C1`` CLI,
``python -m flextool.calibrate``, lands here) and its building blocks.

The first building block is the resilient solve-success detector in
:mod:`flextool.calibrate._solve_status` — it decides "did this solve
actually succeed?" from the solve's *output directory* plus the (weak)
subprocess exit code, tolerating a known model-specific post-solve writer
crash that can bubble to a nonzero exit even when every output is present.
"""

from __future__ import annotations

from flextool.calibrate._solve_status import (
    OutputCheck,
    SolveOutcome,
    assess_solve,
    default_required_outputs,
)

__all__ = [
    "OutputCheck",
    "SolveOutcome",
    "assess_solve",
    "default_required_outputs",
]
