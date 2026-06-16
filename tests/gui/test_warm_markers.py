"""Canary for the solver "warmed" stdout markers.

``_is_solver_warm_marker`` flips a job to ``footprint_solid`` (so admission
trusts its measured RSS) the moment the solver has loaded the matrix and begun
reserving memory. The marker set is a *fast path* only — the watchdog's
RSS-plateau backstop covers any line we miss — but this test pins the strings
against a captured HiGHS banner so a HiGHS upgrade that drifts the wording, or
an accidental edit to the marker list, turns red and we notice rather than
silently falling back to the ~10s-later plateau.
"""

from __future__ import annotations

from flextool.gui.execution_manager import (
    _SOLVER_WARM_MARKERS,
    _is_solver_warm_marker,
)

# A representative slice of real HiGHS console output (v1.x). Each line here
# MUST keep matching, or the fast-path warmed flip silently regresses.
HIGHS_BANNER_LINES = [
    "Running HiGHS 1.7.2 (git hash: nnnnnnn): Copyright (c) 2024 HiGHS",
    "Coefficient ranges:",
    "  Matrix [1e+00, 4e+02]",
    "Presolving model",
    "Solving MIP model with:",
    "Solving LP without presolve, or with basis, or unconstrained",
]

# Our own engine phase checkpoint (emitted even when HiGHS' own log is quiet,
# except on suppressed within-group rolling iters — there the plateau backstop
# takes over).
ENGINE_MARKER_LINE = "lp_build_end  |  Matrix built by polar-high  |  3.2 GB"


def test_each_marker_string_matches_itself() -> None:
    for marker in _SOLVER_WARM_MARKERS:
        assert _is_solver_warm_marker(marker)


def test_matching_is_case_insensitive() -> None:
    assert _is_solver_warm_marker("RUNNING HIGHS 1.7.2")
    assert _is_solver_warm_marker("  COEFFICIENT RANGES:")


def test_real_highs_banner_lines_are_recognised() -> None:
    matched = [ln for ln in HIGHS_BANNER_LINES if _is_solver_warm_marker(ln)]
    # At least the banner, the coefficient-ranges block, presolve, and both
    # solve phases — i.e. multiple independent lines, so one wording change
    # can't blind the detector.
    assert len(matched) >= 4


def test_engine_phase_checkpoint_is_recognised() -> None:
    assert _is_solver_warm_marker(ENGINE_MARKER_LINE)


def test_ordinary_lines_do_not_match() -> None:
    for line in (
        "Solve start: base_solve, 1/3",   # too early — model not built yet
        "Reading database",
        "Writing outputs",
        "total_cost.val = 1234.5",
        "",
    ):
        assert not _is_solver_warm_marker(line)
