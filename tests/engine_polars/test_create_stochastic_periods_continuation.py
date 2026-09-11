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

from flextool.engine_polars._solve_state import ActiveTimeEntry
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
