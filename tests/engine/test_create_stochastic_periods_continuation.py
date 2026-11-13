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

import numpy as np
import pytest

from flextool.engine.input._solve_state import ActiveTimeEntry, FlexToolConfigError
from flextool.engine.solve._stochastic import StochasticSolver


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
    """§5.4 case 1 — the §4 shape, all eight returned/mutated
    structures verbatim, including ``new_active_time_list`` key order."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    fix_storage = {"p2040": P2[0:]}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run(
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
    # order and the step-jump walk).  The period labels p2035 / p2040 have
    # no dispatch: they are in none of the time lists.
    assert list(atl["s"].keys()) == ["p2035_low", "p2040_low"]
    assert atl["s"]["p2035_low"] == P1
    assert atl["s"]["p2040_low"] == P2
    # Nothing is realized or storage-fixed in the fan's own time lists ...
    assert rtl["s"] == {}
    assert fstl["s"] == {}
    # ... the labels keep their pre-stochastic lists (the committed cells
    # inside the fan and the labels' steps).
    assert labels["s"].labels == frozenset({"p2035", "p2040"})
    assert labels["s"].active == {"p2035": P1, "p2040": P2}
    assert labels["s"].realized == {"p2035": P1, "p2040": P2}
    assert labels["s"].fix_storage == {"p2040": P2}
    assert bstl["s"] == ("p2035", "t0001")
    # Step-jump rows (design §4 step_previous shape), forecast periods only.
    assert jumps["s"] == [
        ("p2035_low", "t0001", "t0003", "t0003", "p2035_low", "t0003", -2),
        ("p2035_low", "t0002", "t0001", "t0001", "p2035_low", "t0001", 1),
        ("p2035_low", "t0003", "t0002", "t0002", "p2035_low", "t0002", 1),
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
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run(
        solver, BRANCH_INFO, active, realized, fix_storage,
    )
    assert list(atl["s"].keys()) == ["p2035_low", "p2040_low"]
    # Realized: only the declared window, kept on the period label; p2040 has
    # none (post-jump).
    assert rtl["s"] == {}
    assert labels["s"].realized == {"p2035": P1[:1]}
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
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run(
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
    # No active time for the zero-weight branch in either period (nor for
    # the period labels).
    assert list(atl["s"].keys()) == ["p2035_low", "p2040_low"]
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
# exercises a roll without a reveal (an error since 2026-11-13).  These solver-free cases cover
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


@pytest.mark.parametrize("name", [str, np.str_], ids=["str", "np.str_"])
def test_w1_rolling_roll_without_reveal_raises(name) -> None:
    """A roll of a stochastic rolling solve whose window reaches no reveal (reveal declared at t0010, a later
    roll's first step) would be solved without forecasts: a configuration error (owner 2026-11-13: there must be
    a forecast at each roll's start, or where it branches; before: a silent deterministic trunk).  The names
    print as plain strings also when they arrive as ``np.str_`` (numpy 2 prints ``np.str_('p1')``)."""
    solver = _make_solver_with_modes({"roll_c": "rolling_window"})
    info = [
        (name("p1"), name("rlz"), name("t0010"), "yes", "1"),
        (name("p1"), name("low"), name("t0010"), "no", "3"),
    ]
    win = [ActiveTimeEntry(timestep=name(e.timestep), index=e.index, duration=e.duration) for e in ROLL_WIN]
    with pytest.raises(FlexToolConfigError) as err:
        _run_one(solver, info, {name("p1"): win}, solve=name("roll_c_roll_0"), complete=name("roll_c"))
    assert str(err.value) == (
        "Roll 'roll_c_roll_0' of rolling solve 'roll_c' starts at period 'p1', time step 't0001', but no forecast "
        "of stochastic_branches starts in its horizon (period 'p1' 't0001' to period 'p1' 't0003'), so it would "
        "be solved without forecasts. Every roll of a stochastic rolling solve needs forecasts made at its start "
        "(or where it branches later in its horizon): add stochastic_branches rows for period 'p1' at "
        "analysis_time 't0001', or align the rolls with the forecast times (rolling_solve_jump).")


def test_w1_rolling_fan_at_first_passes() -> None:
    """Fix 1: a rolling roll whose reveal triggers at the roll's first step
    (fan-at-first per roll, Model 2) passes and fans out."""
    solver = _make_solver_with_modes({"roll_c": "rolling_window"})
    info = [
        ("p1", "rlz", "t0001", "yes", "1"),
        ("p1", "low", "t0001", "no", "3"),
    ]
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run_one(
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
    with pytest.raises(FlexToolConfigError, match="at most one"):
        _run_one(solver, info, {"p1": ROLL_WIN})


def test_w1_trigger_cohort_zero_realized_fans_out() -> None:
    """Accept: the fan is forecasts only — a cohort with zero realized:yes
    rows (two forecasts) is VALID and fans out into both forecasts (no
    realized self-row appended)."""
    solver = _make_solver_with_modes({})
    info = [
        ("p1", "low", "t0001", "no", "3"),
        ("p1", "high", "t0001", "no", "3"),
    ]
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run_one(
        solver, info, {"p1": ROLL_WIN}
    )
    assert bstl["s"] == ("p1", "t0001")
    assert pb["s"] == [
        ("p1", "p1"),
        ("p1", "p1_low"),
        ("p1", "p1_high"),
    ]
    # Both forecasts receive active time; no realized self-row (n_real=0).
    assert sbtb["s"] == [("p1_low", "low"), ("p1_high", "high")]
    assert list(atl["s"].keys()) == ["p1_low", "p1_high"]
    assert labels["s"].labels == frozenset({"p1"})


def test_w1_trigger_cohort_zero_forecast_raises() -> None:
    """Reject: a cohort with only a realized member (no forecast branch —
    realized:no with nonzero weight) violates the forecasts-only rule."""
    solver = _make_solver_with_modes({})
    info = [
        ("p1", "rlz", "t0001", "yes", "1"),
        ("p1", "dead", "t0001", "no", "0"),
    ]
    with pytest.raises(FlexToolConfigError, match="at least one forecast"):
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
    (pb, sbtb, atl, jumps, fstl, rtl, bstl, labels) = _run(
        solver, [], active, realized, fix_storage,
    )
    assert pb["s"] == [("p2035", "p2035"), ("p2040", "p2040")]
    assert sbtb["s"] == []
    assert list(atl["s"].keys()) == ["p2035", "p2040"]
    assert list(rtl["s"].keys()) == ["p2035", "p2040"]
    assert list(fstl["s"].keys()) == ["p2035"]
    assert bstl["s"] is None
    assert labels["s"].labels == frozenset()
    assert solver.state.timeline.stochastic_timesteps["s"] == []


# ── one realization series per roll sequence (item 248b) ──────────────
_NESTED_INFO = [
    ("p2020", "high", "t0001", "no", "1"),
    ("p2020", "w2030", "t0001", "yes", "0"),
    ("p2030", "high", "t0004", "no", "1"),
    ("p2030", "w2040", "t0004", "yes", "0"),
]


def _run_two(solver, names):
    """Two sub-solves of complete solve ``inner``: ``names[0]`` over p2020, ``names[1]`` over p2030, each
    fanning at its own first step with its own realized branch."""
    a, b = names
    return solver.create_stochastic_periods(
        {"inner": _NESTED_INFO},
        [a, b],
        {a: "inner", b: "inner"},
        {a: {"p2020": P1}, b: {"p2030": P2}},
        {a: {}, b: {}},
        {a: {"p2020": P1}, b: {"p2030": P2}},
    )


def test_renamed_inner_solves_have_their_own_realization_series() -> None:
    """A nested single inner solve is copied once per outer period (``inner_p2020``, ``inner_p2030``), all of
    complete solve ``inner``.  Each copy is its own solve with one fan point, so realized branches w2030 and
    w2040 at the two copies' fan points are valid (they are not rolls of one sequence)."""
    out = _run_two(_make_solver(), ("inner_p2020", "inner_p2030"))
    assert out[6] == {"inner_p2020": ("p2020", "t0001"), "inner_p2030": ("p2030", "t0004")}


def test_rolls_of_one_sequence_share_the_realization_series() -> None:
    """The same rows as two rolls of rolling solve ``inner``: the realized branch changes from w2030 to w2040
    between the rolls' fan points, an error."""
    solver = _make_solver()
    solver.state.solve = SimpleNamespace(solve_modes={"inner": "rolling_window"})
    with pytest.raises(FlexToolConfigError) as err:
        _run_two(solver, ("inner_roll_0", "inner_roll_1"))
    assert str(err.value) == (
        "The stochastic_branches of solve 'inner' mark different realized branches where its rolls fan out: "
        "'w2030' at period 'p2020' (analysis_time 't0001'); 'w2040' at period 'p2030' (analysis_time 't0004'). "
        "A solve has one realization series: name the same branch in every realized: yes row, and give that "
        "branch the data of each roll at the roll's analysis_time. Check stochastic_branches.")


def test_renamed_rolling_inner_solves_have_their_own_realization_series() -> None:
    """A nested rolling inner solve renamed per outer period: ``inner_p2020_roll_0`` and
    ``inner_p2030_roll_0`` are two roll sequences (``inner_p2020``, ``inner_p2030``), so different realized
    branches at their fan points are valid."""
    solver = _make_solver()
    solver.state.solve = SimpleNamespace(solve_modes={"inner": "rolling_window"})
    out = _run_two(solver, ("inner_p2020_roll_0", "inner_p2030_roll_0"))
    assert out[6] == {"inner_p2020_roll_0": ("p2020", "t0001"), "inner_p2030_roll_0": ("p2030", "t0004")}
