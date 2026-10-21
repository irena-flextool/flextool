"""Unit tests for cross-solve realization overlap (issue #321).

Pure tests of :mod:`flextool.engine_polars._realized_overlap` — no DB, no
solve.  Covers the LAST-WINS dispatch resolution (including the partial-
overlap cases that used to leave timesteps realized by no solve), the
aggregated dispatch-overlap warning, and the investment-period overlap
warning.
"""
from __future__ import annotations

import logging

from flextool.engine_polars._realized_overlap import (
    DispatchOverlap,
    InvestOverlap,
    effective_realized_invest_periods,
    find_realized_invest_overlaps,
    resolve_realized_dispatch_overlap,
    warn_realized_dispatch_overlaps,
    warn_realized_invest_overlaps,
)
from flextool.engine_polars._solve_state import ActiveTimeEntry

LOGGER_NAME = "flextool.engine_polars._realized_overlap"


def _ts(i: int) -> str:
    return f"t{i:04d}"


def _window(first: int, last: int) -> list[ActiveTimeEntry]:
    """Inclusive ``first..last`` run of hourly entries."""
    return [ActiveTimeEntry(_ts(i), i - 1, "1") for i in range(first, last + 1)]


def _steps(time_list: dict, period: str) -> list[str]:
    return [e.timestep for e in time_list.get(period, [])]


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == LOGGER_NAME and r.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------
# Dispatch overlap — resolution
# ---------------------------------------------------------------------------


def test_full_overlap_drops_earlier_and_warns(caplog):
    rtl = {
        "invest": {"p2020": _window(1, 100)},
        "dispatch": {"p2020": _window(1, 100)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)

    assert "p2020" not in rtl["invest"]
    assert _steps(rtl["dispatch"], "p2020") == [_ts(i) for i in range(1, 101)]
    assert overlaps == [DispatchOverlap("invest", "dispatch", "p2020", 100)]

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_realized_dispatch_overlaps(overlaps) == 1
    (msg,) = _warnings(caplog)
    assert "'invest'" in msg and "'dispatch'" in msg
    assert "'p2020'" in msg
    assert "100 overlapping timesteps" in msg
    assert "later solve's realization was kept" in msg


def test_later_solve_covers_start_earlier_keeps_tail():
    rtl = {
        "earlier": {"p2020": _window(1, 100)},
        "later": {"p2020": _window(1, 10)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)

    # No hole: t0011..t0100 stay with the earlier solve.
    assert _steps(rtl["earlier"], "p2020") == [_ts(i) for i in range(11, 101)]
    assert _steps(rtl["later"], "p2020") == [_ts(i) for i in range(1, 11)]
    assert overlaps == [DispatchOverlap("earlier", "later", "p2020", 10)]


def test_later_solve_covers_middle_earlier_keeps_both_sides():
    rtl = {
        "earlier": {"p2020": _window(1, 100)},
        "later": {"p2020": _window(40, 60)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)

    expected = [_ts(i) for i in list(range(1, 40)) + list(range(61, 101))]
    assert _steps(rtl["earlier"], "p2020") == expected  # order preserved
    assert _steps(rtl["later"], "p2020") == [_ts(i) for i in range(40, 61)]
    assert overlaps == [DispatchOverlap("earlier", "later", "p2020", 21)]


def test_kept_entries_are_the_original_objects_in_order():
    original = _window(1, 10)
    rtl = {"earlier": {"p": list(original)}, "later": {"p": _window(4, 6)}}
    resolve_realized_dispatch_overlap(rtl)
    assert rtl["earlier"]["p"] == original[:3] + original[6:]


def test_disjoint_lists_no_overlap_no_warning(caplog):
    rtl = {
        "a": {"p2020": _window(1, 50)},
        "b": {"p2020": _window(51, 100), "p2025": _window(1, 10)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)

    assert overlaps == []
    assert _steps(rtl["a"], "p2020") == [_ts(i) for i in range(1, 51)]
    assert _steps(rtl["b"], "p2020") == [_ts(i) for i in range(51, 101)]
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_realized_dispatch_overlaps(overlaps) == 0
    assert _warnings(caplog) == []


def test_rolling_rolls_with_consecutive_jump_windows_do_not_overlap():
    """Rolls of one rolling solve realize consecutive, disjoint jumps."""
    rtl = {
        f"disp_roll_{k}": {"p2020": _window(24 * k + 1, 24 * (k + 1))}
        for k in range(5)
    }
    assert resolve_realized_dispatch_overlap(rtl) == []
    for k in range(5):
        assert len(rtl[f"disp_roll_{k}"]["p2020"]) == 24


def test_same_period_name_in_different_periods_is_not_an_overlap():
    rtl = {"a": {"p2020": _window(1, 5)}, "b": {"p2025": _window(1, 5)}}
    assert resolve_realized_dispatch_overlap(rtl) == []
    assert len(rtl["a"]["p2020"]) == 5 and len(rtl["b"]["p2025"]) == 5


def test_last_solve_wins_among_three_and_counts_per_pair():
    rtl = {
        "first": {"p": _window(1, 30)},
        "second": {"p": _window(11, 30)},
        "third": {"p": _window(21, 30)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)

    assert _steps(rtl["first"], "p") == [_ts(i) for i in range(1, 11)]
    assert _steps(rtl["second"], "p") == [_ts(i) for i in range(11, 21)]
    assert _steps(rtl["third"], "p") == [_ts(i) for i in range(21, 31)]
    # "first" lost t0011-20 to "second" and t0021-30 to "third"; ordered
    # by earlier solve, then later solve, in cascade order.
    assert overlaps == [
        DispatchOverlap("first", "second", "p", 10),
        DispatchOverlap("first", "third", "p", 10),
        DispatchOverlap("second", "third", "p", 10),
    ]


def test_one_warning_line_per_pair_and_period(caplog):
    rtl = {
        "invest": {"p2020": _window(1, 10), "p2025": _window(1, 10)},
        "dispatch": {"p2020": _window(1, 10), "p2025": _window(1, 4)},
    }
    overlaps = resolve_realized_dispatch_overlap(rtl)
    assert overlaps == [
        DispatchOverlap("invest", "dispatch", "p2020", 10),
        DispatchOverlap("invest", "dispatch", "p2025", 4),
    ]
    assert _steps(rtl["invest"], "p2025") == [_ts(i) for i in range(5, 11)]
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_realized_dispatch_overlaps(overlaps) == 2
    msgs = _warnings(caplog)
    assert "'p2020'" in msgs[0] and "10 overlapping timesteps" in msgs[0]
    assert "'p2025'" in msgs[1] and "4 overlapping timesteps" in msgs[1]


def test_singular_timestep_wording(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        warn_realized_dispatch_overlaps([DispatchOverlap("a", "b", "p", 1)])
    (msg,) = _warnings(caplog)
    assert "(1 overlapping timestep)" in msg


def test_later_solve_that_does_not_realize_the_period_claims_nothing():
    """A solve whose realized_periods omit a period realizes nothing there,
    so it must not take that period's cells away from an earlier solve."""
    rtl = {
        "earlier": {"p2020": _window(1, 10)},
        "later": {"p2020": _window(1, 10), "p2025": _window(1, 10)},
    }
    overlaps = resolve_realized_dispatch_overlap(
        rtl, {"earlier": {"p2020"}, "later": {"p2025"}},
    )
    assert overlaps == []
    assert len(rtl["earlier"]["p2020"]) == 10
    # Entries outside the realized periods are left untouched.
    assert len(rtl["later"]["p2020"]) == 10


def test_empty_period_lists_are_removed():
    rtl = {"a": {"p": [], "q": _window(1, 2)}, "b": {"q": _window(3, 4)}}
    resolve_realized_dispatch_overlap(rtl)
    assert "p" not in rtl["a"]
    assert len(rtl["a"]["q"]) == 2


def test_silent_pair_suppresses_only_that_pair(caplog):
    overlaps = [
        DispatchOverlap("uc", "uc__realized_dispatch", "p", 24),
        DispatchOverlap("invest", "uc__realized_dispatch", "p", 24),
    ]
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        n = warn_realized_dispatch_overlaps(
            overlaps, silent_pairs={("uc", "uc__realized_dispatch")},
        )
    assert n == 1
    (msg,) = _warnings(caplog)
    assert "'invest'" in msg


# ---------------------------------------------------------------------------
# Investment overlap
# ---------------------------------------------------------------------------


def test_invest_overlap_warns_with_both_solves_and_period(caplog):
    overlaps = find_realized_invest_overlaps([
        ("invest_a", ["p2020", "p2025"]),
        ("dispatch", []),
        ("invest_b", ["p2025", "p2030"]),
    ])
    assert overlaps == [InvestOverlap("invest_a", "invest_b", "p2025")]

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_realized_invest_overlaps(overlaps) == 1
    (msg,) = _warnings(caplog)
    assert "'invest_a'" in msg and "'invest_b'" in msg
    assert "'p2025'" in msg
    assert "realized by only one solve" in msg


def test_invest_disjoint_periods_no_warning(caplog):
    overlaps = find_realized_invest_overlaps([
        ("invest_p2020", ["p2020"]),
        ("invest_p2025", ["p2025"]),
    ])
    assert overlaps == []
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_realized_invest_overlaps(overlaps) == 0
    assert _warnings(caplog) == []


def test_invest_overlap_chain_reports_previous_claimant():
    overlaps = find_realized_invest_overlaps([
        ("a", ["p"]), ("b", ["p"]), ("c", ["p"]),
    ])
    assert overlaps == [InvestOverlap("a", "b", "p"), InvestOverlap("b", "c", "p")]


def test_effective_realized_invest_periods_reads_p_in_and_dedups():
    assert effective_realized_invest_periods(
        "s",
        realized_invest_periods={"s": [("p2020", "p2020"), ("p2020", "p2025"),
                                       ("p2025", "p2025")]},
        invest_periods={},
        realized_periods={},
    ) == ["p2020", "p2025"]


def test_effective_realized_invest_periods_falls_back_to_realized_periods():
    kwargs = dict(
        realized_invest_periods={},
        invest_periods={"s": [("p2020", "p2020")]},
        realized_periods={"s": [("p2020", "p2020")]},
    )
    assert effective_realized_invest_periods("s", **kwargs) == ["p2020"]
    # A pure dispatch solve (no invest_periods) realizes no investment.
    assert effective_realized_invest_periods(
        "s",
        realized_invest_periods={},
        invest_periods={},
        realized_periods={"s": [("p2020", "p2020")]},
    ) == []
