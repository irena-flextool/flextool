"""Solver-free unit tests for ``create_stochastic_periods`` under the
Option B ("realized-not-past-reveal") fan-out.

Drives the method directly with a stub state (only ``logger`` and
``timeline.stochastic_timesteps`` are touched) and hand-built
``ActiveTimeEntry`` lists.

Option B semantics (see ``_stochastic.create_stochastic_periods``):

* A realized branch requires a COMMITTED (pre-reveal) region — at least
  one period before the branching period.  Fan-at-first-with-realized
  (the reveal declared at the solve's FIRST period) is DEPRECATED and
  raises ``FlexToolConfigError`` (Stage A guard).
* The realized trunk STOPS at the reveal: committed periods and the
  reveal/branching period keep the REAL period name (and their
  realized / fix-storage windows); post-reveal CONTINUATION periods get
  ONLY synthetic ``<period>_<branch>`` forecast copies — the real-named
  trunk is NOT re-emitted, so ``realized_time_lists`` /
  ``fix_storage_time_lists`` cover only the committed region.
* Storage non-anticipativity still links every branch copy: the reveal
  copy inherits the committed trunk's end-of-period state (``p2035_low``
  -> ``p2030``) and each continuation copy continues from the same
  branch's previous period (``p2040_low`` -> ``p2035_low``).  No
  continuation step links to a realized period that no longer exists.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from flextool.engine_polars._solve_state import ActiveTimeEntry
from flextool.engine_polars._stochastic import StochasticSolver
from flextool.engine_polars._solve_state import FlexToolConfigError


def _ats(names_and_indices):
    return [ActiveTimeEntry(timestep=t, index=i, duration=1.0)
            for t, i in names_and_indices]


# Three periods: p2030 COMMITTED, p2035 REVEAL (branching), p2040
# CONTINUATION (post-reveal).
P0 = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])
P1 = _ats([("t0004", 3), ("t0005", 4), ("t0006", 5)])
P2 = _ats([("t0007", 6), ("t0008", 7), ("t0009", 8)])

# stochastic_branches rows: (period, branch, start_step, realized, weight)
# Reveal at p2035 (mid-horizon — p2030 precedes it as committed).
BRANCH_INFO = [
    ("p2035", "rlz", "t0004", "yes", "1"),
    ("p2035", "low", "t0004", "no", "3"),
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


def test_mid_horizon_realized_stops_at_reveal() -> None:
    """Realized trunk spans COMMITTED + REVEAL only; the continuation
    period yields forecast copies and NO real-named trunk."""
    solver = _make_solver()
    active = {"p2030": P0, "p2035": P1, "p2040": P2}
    realized = {"p2030": P0[0:], "p2035": P1[0:], "p2040": P2[0:]}
    # fix-storage declared on BOTH reveal and continuation; only the
    # reveal window survives (continuation is post-reveal).
    fix_storage = {"p2035": P1[0:], "p2040": P2[0:]}
    (pb, sbtb, atl, jumps, fstl, rtl, bstl) = _run(
        solver, BRANCH_INFO, active, realized, fix_storage,
    )

    # period__branch: committed self-row, reveal self + fan, continuation
    # FAN ONLY (no (p2040, p2040) self-row — the trunk stopped at reveal).
    assert pb["s"] == [
        ("p2030", "p2030"),
        ("p2035", "p2035"),
        ("p2035", "p2035_rlz"),
        ("p2035", "p2035_low"),
        ("p2040", "p2040_rlz"),
        ("p2040", "p2040_low"),
    ]
    # Active time: committed + reveal real-named, plus the active forecast
    # copies at reveal and continuation.  No real-named p2040.
    assert list(atl["s"].keys()) == [
        "p2030", "p2035", "p2035_low", "p2040_low",
    ]
    # Realized / fix-storage cover the COMMITTED region only (committed +
    # reveal); the continuation period is absent.
    assert list(rtl["s"].keys()) == ["p2030", "p2035"]
    assert rtl["s"]["p2030"] == P0
    assert rtl["s"]["p2035"] == P1
    assert list(fstl["s"].keys()) == ["p2035"]
    assert fstl["s"]["p2035"] == P1
    # solve_branch__time_branch: active-branch fan rows (reveal then
    # continuation), then the realized-branch resolution rows for the
    # committed + reveal periods ONLY (no continuation realized row).
    assert sbtb["s"] == [
        ("p2035_low", "low"),
        ("p2040_low", "low"),
        ("p2030", "rlz"),
        ("p2035", "rlz"),
    ]
    assert bstl["s"] == ("p2035", "t0004")


def test_mid_horizon_storage_linkage_no_missing_realized() -> None:
    """Step-jump linkage: the reveal branch copy inherits the committed
    trunk's end-state and the continuation copy continues from its own
    branch — never from a (now absent) realized continuation period."""
    solver = _make_solver()
    active = {"p2030": P0, "p2035": P1, "p2040": P2}
    realized = {"p2030": P0[0:], "p2035": P1[0:], "p2040": P2[0:]}
    (_pb, _sbtb, _atl, jumps, _fstl, _rtl, _bstl) = _run(
        solver, BRANCH_INFO, active, realized, {},
    )
    jump_map = {(r[0], r[1]): (r[4], r[5]) for r in jumps["s"]}
    # Reveal copy's first step links to the committed trunk's last step.
    assert jump_map[("p2035_low", "t0004")] == ("p2030", "t0003")
    # Continuation copy's first step links to the same branch's previous
    # (reveal) copy — NOT to any realized period.
    assert jump_map[("p2040_low", "t0007")] == ("p2035_low", "t0006")
    # The reveal realized trunk links to the committed realized trunk.
    assert jump_map[("p2035", "t0004")] == ("p2030", "t0003")
    # No jump row references a real-named continuation period as a source.
    assert all(r[4] != "p2040" for r in jumps["s"])


def test_mid_horizon_zero_weight_branch_metadata_only() -> None:
    """A zero-weight forecast gets period__branch + stochastic_timesteps
    metadata at reveal and continuation but no active time / no
    solve_branch__time_branch row."""
    solver = _make_solver()
    info = BRANCH_INFO + [("p2035", "zero", "t0004", "no", "0")]
    active = {"p2030": P0, "p2035": P1, "p2040": P2}
    realized = {"p2030": P0[0:], "p2035": P1[0:], "p2040": P2[0:]}
    (pb, sbtb, atl, _jumps, _fstl, _rtl, _bstl) = _run(
        solver, info, active, realized, {},
    )
    # zero-weight branch present as metadata at reveal + continuation.
    assert ("p2035", "p2035_zero") in pb["s"]
    assert ("p2040", "p2040_zero") in pb["s"]
    # No active time and no sbtb row for the zero-weight branch.
    assert list(atl["s"].keys()) == [
        "p2030", "p2035", "p2035_low", "p2040_low",
    ]
    assert all(r[0] != "p2035_zero" and r[0] != "p2040_zero"
               for r in sbtb["s"])
    ts = solver.state.timeline.stochastic_timesteps["s"]
    assert [("p2040_zero", t) for t in ("t0007", "t0008", "t0009")] == [
        r for r in ts if r[0] == "p2040_zero"
    ]


def test_fan_at_first_with_realized_raises() -> None:
    """Stage A guard: a realized branch declared at the solve's FIRST
    period (no committed region) is deprecated and must raise."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    info = [
        ("p2035", "rlz", "t0004", "yes", "1"),
        ("p2035", "low", "t0004", "no", "3"),
    ]
    with pytest.raises(FlexToolConfigError, match="committed region"):
        _run(solver, info, active, realized, {})


def test_deterministic_solve_passthrough() -> None:
    """No stochastic rows → every period keeps its time lists and gets
    a (period, period) self-row; no branches anywhere; the Stage A guard
    does NOT fire (byte-parity guard for the deterministic path)."""
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
