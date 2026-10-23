"""Cross-solve realization overlap: last-wins resolution + user warnings.

Two kinds of overlap can occur between the solves of one model run
(GitHub issue irena-flextool/flextool#321):

* **Dispatch overlap.**  Two solves both realize the same
  ``(period, timestep)``.  This is legitimate (a nested invest+dispatch
  setup does it routinely), and the maintainer-decided resolution is
  LAST-WINS: the solve that runs LATER in cascade order keeps the cell and
  the earlier solve's claim is dropped.  That keeps each solve's persisted
  realized slice disjoint, so the multi-solve output union stays a clean
  concat.  :func:`resolve_realized_dispatch_overlap` does the resolution
  and reports what it dropped; :func:`warn_realized_dispatch_overlaps`
  turns that report into one aggregated warning line per
  ``(earlier solve, later solve, period)``.

* **Investment overlap.**  Two solves both list the same period in
  ``realized_invest_periods``.  Investments for a period should be
  realized by one solve only, so this is a configuration mistake.
  :func:`find_realized_invest_overlaps` detects it and
  :func:`warn_realized_invest_overlaps` reports it.  Warning only — the
  run continues unchanged.

Everything here is pure (no solve, no I/O) so it is unit-testable in
isolation; ``_native_run_model`` calls it at step 3.5.

Warnings go through a MODULE-level logger, not the run's ``state.logger``:
the orchestrator raises ``state.logger`` to ERROR for the whole cascade
(``_orchestration._drive_cascade``) to silence per-solve chatter, so a
WARNING sent there would never be seen.  Same pattern as
``_recursive_solve._warn_if_rolling_carries_invest``.
"""
from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MODULE_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch overlap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchOverlap:
    """Timesteps one solve lost to a later solve in one period."""

    earlier_solve: str
    later_solve: str
    period: str
    timesteps_dropped: int


def resolve_realized_dispatch_overlap(
    realized_time_lists: dict[str, dict[str, list[Any]]],
    realized_periods_by_solve: Mapping[str, Collection[str]] | None = None,
) -> list[DispatchOverlap]:
    """Make the per-solve realized dispatch windows disjoint (LAST-WINS).

    ``realized_time_lists`` maps ``solve -> {period: [entry, ...]}`` where
    each entry has a ``.timestep`` attribute (``ActiveTimeEntry``), and the
    dict's insertion order is the cascade order.  When several solves
    realize the same ``(period, timestep)`` the LAST of them keeps it and
    the cell is removed from every earlier solve.  Only the timesteps a
    later solve actually claims are removed; everything else an earlier
    solve realizes is kept, in its original order (so an earlier solve may
    end up with a non-contiguous window, e.g. both sides of a middle chunk
    claimed by a later solve).  Periods whose list becomes empty are
    deleted.

    Args:
        realized_time_lists: mutated in place.
        realized_periods_by_solve: optional ``solve -> periods`` that each
            solve REALLY realizes (``derive_realized_dispatch`` emits only
            those periods).  When given, entries in a solve's other periods
            neither claim cells nor lose them — such a solve realizes
            nothing there, so it must not take cells away from an earlier
            solve.  ``None`` treats every listed period as realized.

    Returns:
        One record per ``(earlier solve, later solve, period)`` that lost
        at least one timestep, ordered by earlier solve (cascade order),
        then period (listing order), then later solve (cascade order).
    """
    solve_order = list(realized_time_lists)
    solve_pos = {s: i for i, s in enumerate(solve_order)}
    # period -> timestep -> the latest solve that realizes it.
    owner: dict[str, dict[str, str]] = {}
    # Collected per earlier solve while walking backwards; re-ordered below.
    per_solve: dict[str, list[DispatchOverlap]] = {}

    for solve in reversed(solve_order):
        time_list = realized_time_lists[solve]
        realized = (
            None if realized_periods_by_solve is None
            else set(realized_periods_by_solve.get(solve, ()))
        )
        records: list[DispatchOverlap] = []
        for period in list(time_list):
            entries = time_list[period]
            if realized is not None and period not in realized:
                if not entries:
                    del time_list[period]
                continue
            period_owner = owner.setdefault(period, {})
            kept: list[Any] = []
            dropped: dict[str, int] = {}
            seen_here: set[str] = set()
            for entry in entries:
                ts = entry.timestep
                if ts in seen_here:
                    # Repeated timestep within one solve's own list — keep
                    # the first occurrence only (not a cross-solve overlap).
                    continue
                seen_here.add(ts)
                later = period_owner.get(ts)
                if later is None:
                    kept.append(entry)
                else:
                    dropped[later] = dropped.get(later, 0) + 1
            for entry in kept:
                period_owner[entry.timestep] = solve
            if kept:
                time_list[period] = kept
            else:
                del time_list[period]
            for later in sorted(dropped, key=solve_pos.__getitem__):
                records.append(DispatchOverlap(
                    earlier_solve=solve,
                    later_solve=later,
                    period=period,
                    timesteps_dropped=dropped[later],
                ))
        if records:
            per_solve[solve] = records

    out: list[DispatchOverlap] = []
    for solve in solve_order:
        out.extend(per_solve.get(solve, ()))
    return out


def warn_realized_dispatch_overlaps(
    overlaps: Iterable[DispatchOverlap],
    *,
    silent_pairs: Collection[tuple[str, str]] = (),
    logger: logging.Logger | None = None,
) -> int:
    """Log one warning line per ``(earlier, later, period)`` overlap.

    Args:
        overlaps: output of :func:`resolve_realized_dispatch_overlap`.
        silent_pairs: ``(earlier, later)`` pairs whose overlap the engine
            itself created on purpose and that must not warn — the
            auto-spawned ``<solve>__realized_dispatch`` sibling (and its
            rolls) re-realizes its parent's window by design.
        logger: defaults to this module's logger (see module docstring for
            why not ``state.logger``).

    Returns:
        The number of warning lines emitted.
    """
    log = logger if logger is not None else _MODULE_LOGGER
    silent = set(silent_pairs)
    n = 0
    for ov in overlaps:
        if (ov.earlier_solve, ov.later_solve) in silent:
            continue
        log.warning(
            "Solves '%s' and '%s' both realize dispatch in period '%s' "
            "(%d overlapping timestep%s). The later solve's realization was "
            "kept for those timesteps and the earlier solve's results for "
            "them are not reported. This is expected when, for example, a "
            "dispatch solve is nested inside an investment solve; otherwise "
            "check the realized_periods of these solves.",
            ov.earlier_solve, ov.later_solve, ov.period,
            ov.timesteps_dropped, "" if ov.timesteps_dropped == 1 else "s",
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Investment overlap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvestOverlap:
    """A period whose investments are realized by more than one solve."""

    earlier_solve: str
    later_solve: str
    period: str


def effective_realized_invest_periods(
    solve: str,
    *,
    realized_invest_periods: Mapping[str, Sequence[tuple[str, str]]],
    invest_periods: Mapping[str, Sequence[tuple[str, str]]],
    realized_periods: Mapping[str, Sequence[tuple[str, str]]],
) -> list[str]:
    """Periods whose investments *solve* realizes, as the cascade sees it.

    Mirrors the ``realized_invest_periods_of_current_solve`` emission in
    ``_native_run_model``: ``realized_invest_periods`` (the ``p_in`` member
    of each ``(p_from, p_in)`` tuple), falling back to ``realized_periods``
    when ``realized_invest_periods`` is empty but both ``invest_periods``
    and ``realized_periods`` are set.  Order preserved, duplicates removed.
    """
    tuples = list(realized_invest_periods.get(solve, []))
    if (
        not tuples
        and invest_periods.get(solve)
        and realized_periods.get(solve)
    ):
        tuples = list(realized_periods.get(solve, []))
    return list(dict.fromkeys(t[1] for t in tuples))


def find_realized_invest_overlaps(
    realized_invest_by_solve: Sequence[tuple[str, Collection[str]]],
) -> list[InvestOverlap]:
    """Find periods realized for investment by more than one solve.

    Args:
        realized_invest_by_solve: ``(solve, periods)`` pairs in cascade
            order, one per distinct solve.

    Returns:
        One record per repeated claim: the period, the solve that claimed
        it most recently before, and the solve claiming it again.  Ordered
        by the re-claiming solve (cascade order), then period.
    """
    last_claim: dict[str, str] = {}
    out: list[InvestOverlap] = []
    for solve, periods in realized_invest_by_solve:
        for period in dict.fromkeys(periods):
            earlier = last_claim.get(period)
            if earlier is not None and earlier != solve:
                out.append(InvestOverlap(
                    earlier_solve=earlier, later_solve=solve, period=period,
                ))
            last_claim[period] = solve
    return out


def warn_realized_invest_overlaps(
    overlaps: Iterable[InvestOverlap],
    *,
    logger: logging.Logger | None = None,
) -> int:
    """Log one warning line per investment-overlap record.

    Returns:
        The number of warning lines emitted.
    """
    log = logger if logger is not None else _MODULE_LOGGER
    n = 0
    for ov in overlaps:
        log.warning(
            "Solves '%s' and '%s' both list period '%s' in "
            "realized_invest_periods. Investments for a period should be "
            "realized by only one solve. If the later solve can also invest "
            "in that period, it treats the earlier solve's investment as "
            "existing capacity and may invest more on top of it; the "
            "reported invested capacity and investment cost for the period "
            "then come from the later solve only, although total capacity "
            "includes both. Remove '%s' from the realized_invest_periods of "
            "one of these solves.",
            ov.earlier_solve, ov.later_solve, ov.period, ov.period,
        )
        n += 1
    return n
