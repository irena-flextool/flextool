"""Solver-free unit tests for ``create_stochastic_periods`` continuation
fan-out (design ``specs/step05_continuation_fanout_design.md`` §5.4).

Drives the method directly with a stub state (only ``logger`` and
``timeline.stochastic_timesteps`` are touched) and hand-built
``ActiveTimeEntry`` lists.

DEFECT PIN: on unfixed main the continuation ``else`` branch wrote
nothing into ``new_active_time_list`` / ``new_realized_time_list`` /
``new_fix_storage_time_list`` — the two-period case's
``active_time_lists`` came back with only the p2035 cohort and
``realized_time_lists`` lost p2040 entirely.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from flextool.engine_polars._solve_state import ActiveTimeEntry, FlexToolConfigError
from flextool.engine_polars._stochastic import StochasticSolver


def _ats(names_and_indices):
    return [ActiveTimeEntry(timestep=t, index=i, duration=1.0)
            for t, i in names_and_indices]


P1 = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])
P2 = _ats([("t0004", 3), ("t0005", 4), ("t0006", 5)])

# stochastic_branches rows: (period, branch, start_step, realized, weight)
BRANCH_INFO = [
    ("p2035", "rlz", "t0001", "yes", "1"),
    ("p2035", "low", "t0001", "no", "3"),
]


def _make_solver() -> StochasticSolver:
    state = SimpleNamespace(
        logger=logging.getLogger("test_create_stochastic_periods"),
        timeline=SimpleNamespace(stochastic_timesteps=defaultdict(list)),
    )
    return StochasticSolver(state)


def _run(solver, info, active, realized, fix_storage):
    return solver.create_stochastic_periods(
        {"s": info},
        ["s"],
        {"s": "s"},
        {"s": active},
        {"s": fix_storage},
        {"s": realized},
    )


def test_two_period_fan_all_structures() -> None:
    """§5.4 case 1 — the §4 shape, all seven returned/mutated
    structures verbatim, including ``new_active_time_list`` key order."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    fix_storage = {"p2040": P2[0:]}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run(
        solver, BRANCH_INFO, active, realized, fix_storage,
    )

    assert pb["s"] == [
        ("p2035", "p2035"),
        ("p2035", "p2035_rlz"),
        ("p2035", "p2035_low"),
        ("p2040", "p2040"),
        ("p2040", "p2040_rlz"),
        ("p2040", "p2040_low"),
    ]
    # Fan rows for active branches first (branching then continuation),
    # then the realized-branch resolution rows per period.
    assert sbtb["s"] == [
        ("p2035_low", "low"),
        ("p2040_low", "low"),
        ("p2035", "rlz"),
        ("p2040", "rlz"),
    ]
    # Key ORDER is part of the contract (drives steps_in_use emission
    # order and the step-jump walk).
    assert list(atl["s"].keys()) == [
        "p2035", "p2035_low", "p2040", "p2040_low",
    ]
    assert atl["s"]["p2035"] == P1
    assert atl["s"]["p2035_low"] == P1
    assert atl["s"]["p2040"] == P2
    assert atl["s"]["p2040_low"] == P2
    # Realized branch keeps REAL period names with its windows.
    assert list(rtl["s"].keys()) == ["p2035", "p2040"]
    assert rtl["s"]["p2040"] == P2
    # fix-storage window carried through the continuation period.
    assert list(fstl["s"].keys()) == ["p2040"]
    assert fstl["s"]["p2040"] == P2
    assert bstl["s"] == ("p2035", "t0001")
    # Step-jump rows (design §4 step_previous shape).
    assert jumps["s"] == [
        ("p2035", "t0001", "t0003", "t0003", "p2040_low", "t0006", -5),
        ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001", 1),
        ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002", 1),
        ("p2035_low", "t0001", "t0003", "t0003", "p2035_low", "t0003", -2),
        ("p2035_low", "t0002", "t0001", "t0001", "p2035_low", "t0001", 1),
        ("p2035_low", "t0003", "t0002", "t0002", "p2035_low", "t0002", 1),
        ("p2040", "t0004", "t0006", "t0006", "p2035", "t0003", 1),
        ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004", 1),
        ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005", 1),
        ("p2040_low", "t0004", "t0006", "t0006", "p2035_low", "t0003", 1),
        ("p2040_low", "t0005", "t0004", "t0004", "p2040_low", "t0004", 1),
        ("p2040_low", "t0006", "t0005", "t0005", "p2040_low", "t0005", 1),
    ]
    # stochastic_timesteps: full-fan mirror (metadata rlz rows included,
    # matching the branching-period convention).
    ts = solver.state.timeline.stochastic_timesteps["s"]
    assert ts == (
        [("p2035_rlz", t) for t in ("t0001", "t0002", "t0003")]
        + [("p2035_low", t) for t in ("t0001", "t0002", "t0003")]
        + [("p2040_rlz", t) for t in ("t0004", "t0005", "t0006")]
        + [("p2040_low", t) for t in ("t0004", "t0005", "t0006")]
    )


def test_rolling_window_partial_realized() -> None:
    """§5.4 case 2 — a rolling roll whose realized (jump) window covers
    only part of the horizon: the ``if period in realized_time_list``
    guards keep the partial windows and the fan-out is unchanged."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[:1]}       # only the jump slice realized
    fix_storage: dict = {}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run(
        solver, BRANCH_INFO, active, realized, fix_storage,
    )
    assert list(atl["s"].keys()) == [
        "p2035", "p2035_low", "p2040", "p2040_low",
    ]
    # Realized: only the declared window; p2040 has none (post-jump).
    assert list(rtl["s"].keys()) == ["p2035"]
    assert rtl["s"]["p2035"] == P1[:1]
    assert fstl["s"] == {}
    assert pb["s"] == [
        ("p2035", "p2035"),
        ("p2035", "p2035_rlz"),
        ("p2035", "p2035_low"),
        ("p2040", "p2040"),
        ("p2040", "p2040_rlz"),
        ("p2040", "p2040_low"),
    ]


def test_zero_weight_branch_metadata_only() -> None:
    """§5.4 case 3 — a zero-weight branch gets metadata
    ``period__branch`` + ``stochastic_timesteps`` rows at the
    continuation period but NO active time and NO
    ``solve_branch__time_branch`` row (mirrors the branching period)."""
    solver = _make_solver()
    info = BRANCH_INFO + [("p2035", "zero", "t0001", "no", "0")]
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run(
        solver, info, active, realized, {},
    )
    assert pb["s"] == [
        ("p2035", "p2035"),
        ("p2035", "p2035_rlz"),
        ("p2035", "p2035_low"),
        ("p2035", "p2035_zero"),
        ("p2040", "p2040"),
        ("p2040", "p2040_rlz"),
        ("p2040", "p2040_low"),
        ("p2040", "p2040_zero"),
    ]
    # No active time for the zero-weight branch in either period.
    assert list(atl["s"].keys()) == [
        "p2035", "p2035_low", "p2040", "p2040_low",
    ]
    # No sbtb rows for the zero-weight branch.
    assert sbtb["s"] == [
        ("p2035_low", "low"),
        ("p2040_low", "low"),
        ("p2035", "rlz"),
        ("p2040", "rlz"),
    ]
    # Metadata timesteps present for the zero-weight branch.
    ts = solver.state.timeline.stochastic_timesteps["s"]
    assert [("p2040_zero", t) for t in ("t0004", "t0005", "t0006")] == [
        r for r in ts if r[0] == "p2040_zero"
    ]


# ---------------------------------------------------------------------------
# W1 — triggering-cohort validation (spec §A.1/§A.2/§F-step2/§H.3).
#
# The shipped ``1_week_rolling_wind`` fixture fans at EVERY roll, so it never
# exercises the pre-reveal-roll branch of Fix 1.  These solver-free cases cover
# it (plus the reject paths) directly.  Rolling vs single is detected
# structurally via ``state.solve.solve_modes[complete_solve] == "rolling_window"``
# (not the ``_roll_N`` name suffix); so the state stub here grows a ``solve``
# namespace carrying that map.
# ---------------------------------------------------------------------------

# A single-period roll window of three steps (period p1).
ROLL_WIN = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])


def _make_solver_with_modes(solve_modes: dict) -> StochasticSolver:
    state = SimpleNamespace(
        logger=logging.getLogger("test_create_stochastic_periods"),
        timeline=SimpleNamespace(stochastic_timesteps=defaultdict(list)),
        solve=SimpleNamespace(solve_modes=dict(solve_modes)),
    )
    return StochasticSolver(state)


def _run_one(
    solver, info, active, *, solve="s", complete=None, realized=None,
    fix_storage=None,
):
    complete = solve if complete is None else complete
    return solver.create_stochastic_periods(
        {complete: info},
        [solve],
        {solve: complete},
        {solve: active},
        {solve: fix_storage or {}},
        {solve: realized or {}},
    )


def test_w1_rolling_pre_reveal_roll_does_not_raise() -> None:
    """Fix 1: a rolling roll whose sliced window legitimately PRECEDES the
    first reveal (reveal declared at a later roll's first step, unreachable
    here) stays a deterministic trunk — no raise, no branching."""
    solver = _make_solver_with_modes({"roll_c": "rolling_window"})
    # Reveal declared at t0010 (a later roll's first step); this roll's window
    # only reaches t0001..t0003, so no reveal is reachable.
    info = [
        ("p1", "rlz", "t0010", "yes", "1"),
        ("p1", "low", "t0010", "no", "3"),
    ]
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run_one(
        solver, info, {"p1": ROLL_WIN},
        solve="roll_c_roll_0", complete="roll_c",
    )
    # Deterministic trunk: self-row only, no fan, no branch start.
    assert pb["roll_c_roll_0"] == [("p1", "p1")]
    assert sbtb["roll_c_roll_0"] == []
    assert bstl["roll_c_roll_0"] is None
    assert list(atl["roll_c_roll_0"].keys()) == ["p1"]


def test_w1_rolling_fan_at_first_passes() -> None:
    """Fix 1: a rolling roll whose reveal triggers at the roll's first step
    (fan-at-first per roll, Model 2) passes and fans out."""
    solver = _make_solver_with_modes({"roll_c": "rolling_window"})
    info = [
        ("p1", "rlz", "t0001", "yes", "1"),
        ("p1", "low", "t0001", "no", "3"),
    ]
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run_one(
        solver, info, {"p1": ROLL_WIN},
        solve="roll_c_roll_0", complete="roll_c",
    )
    assert bstl["roll_c_roll_0"] == ("p1", "t0001")
    assert pb["roll_c_roll_0"] == [
        ("p1", "p1"),
        ("p1", "p1_rlz"),
        ("p1", "p1_low"),
    ]
    assert sbtb["roll_c_roll_0"] == [("p1_low", "low"), ("p1", "rlz")]


def test_w1_trigger_not_at_period_first_step_raises() -> None:
    """Reject: the triggering reveal's start_step is not the period's first
    step in this solve (not a period-boundary reveal)."""
    solver = _make_solver_with_modes({})  # single solve
    # Reveal declared at t0002, but the window's first step is t0001.
    info = [
        ("p1", "rlz", "t0002", "yes", "1"),
        ("p1", "low", "t0002", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="FIRST step"):
        _run_one(solver, info, {"p1": ROLL_WIN})


def test_w1_trigger_cohort_two_realized_raises() -> None:
    """Reject: the triggering cohort carries two realized:yes rows."""
    solver = _make_solver_with_modes({})
    info = [
        ("p1", "rlz1", "t0001", "yes", "1"),
        ("p1", "rlz2", "t0001", "yes", "1"),
        ("p1", "low", "t0001", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="exactly one realized"):
        _run_one(solver, info, {"p1": ROLL_WIN})


def test_w1_trigger_cohort_zero_realized_raises() -> None:
    """Reject: the triggering cohort carries no realized:yes row."""
    solver = _make_solver_with_modes({})
    info = [
        ("p1", "low", "t0001", "no", "3"),
        ("p1", "high", "t0001", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="exactly one realized"):
        _run_one(solver, info, {"p1": ROLL_WIN})


def test_w1_single_solve_no_reachable_reveal_raises() -> None:
    """Safety net preserved: a SINGLE (non-rolling) solve with non-empty
    info but no reachable reveal raises (silently-ignored branches)."""
    solver = _make_solver_with_modes({})  # single solve
    info = [
        ("p1", "rlz", "t0099", "yes", "1"),
        ("p1", "low", "t0099", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="silently ignored"):
        _run_one(solver, info, {"p1": ROLL_WIN})


def test_deterministic_solve_passthrough() -> None:
    """No stochastic rows → every period keeps its time lists and gets
    a (period, period) self-row; no branches anywhere (byte-parity
    guard for the deterministic path)."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    fix_storage = {"p2035": P1[0:]}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run(
        solver, [], active, realized, fix_storage,
    )
    assert pb["s"] == [("p2035", "p2035"), ("p2040", "p2040")]
    assert sbtb["s"] == []
    assert list(atl["s"].keys()) == ["p2035", "p2040"]
    assert list(rtl["s"].keys()) == ["p2035", "p2040"]
    assert list(fstl["s"].keys()) == ["p2035"]
    assert bstl["s"] is None
    assert solver.state.timeline.stochastic_timesteps["s"] == []
