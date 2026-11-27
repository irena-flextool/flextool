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
  turns that report into one coded warning line (W-0500) per pair of
  complete solves (the periods and time steps counted).

* **Investment overlap.**  Two solves both list the same period in
  ``realized_invest_periods``.  Investments for a period should be
  realized by one solve only, so this is a configuration mistake.
  :func:`find_realized_invest_overlaps` detects it and
  :func:`warn_realized_invest_overlaps` reports it.  Warning only — the
  run continues unchanged.

* **Dispatch before the investments.**  A solve realizes the dispatch of a
  period that another solve invests in and no solve up to it has realized
  the investments of.
  :func:`check_dispatch_after_investments` raises before anything solves;
  a solve dispatching its own unrealized investment periods gets W-0641.

Everything here is pure (no solve, no I/O) so it is unit-testable in
isolation; ``_run_model`` calls it at step 3.5.

Warnings go through a MODULE-level logger, not the run's ``state.logger``:
the orchestrator raises ``state.logger`` to ERROR for the whole cascade
(``_orchestration._cascade._drive_cascade``) to silence per-solve chatter, so a
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


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _pairs(records, key) -> "dict[tuple[str, str], list]":
    """``{(earlier, later): [record, ...]}`` with the rolls ``<solve>_roll_<k>`` of a rolling solve counted as
    ``<solve>`` (one line per pair of complete solves, whatever levels and rolls the overlaps came from).  Two
    rolls of one solve give the pair ``(solve, solve)``."""
    out: dict[tuple[str, str], list] = {}
    for record in records:
        out.setdefault(key(record), []).append(record)
    return out


def warn_realized_dispatch_overlaps(
    overlaps: Iterable[DispatchOverlap],
    *,
    silent_pairs: Collection[tuple[str, str]] = (),
    logger: logging.Logger | None = None,
) -> int:
    """One coded warning line (W-0500) per pair of complete solves whose realized dispatch overlaps, with the
    number of periods and time steps; each ``(earlier, later, period)`` overlap goes to the debug log.

    Args:
        overlaps: output of :func:`resolve_realized_dispatch_overlap`.
        silent_pairs: ``(earlier, later)`` pairs whose overlap the engine
            itself created on purpose and that must not warn — the
            auto-spawned ``<solve>__realized_dispatch`` sibling (and its
            rolls) re-realizes its parent's window by design.
        logger: the logger of the per-overlap debug lines (defaults to this module's logger; see the module
            docstring for why not ``state.logger``).

    Returns:
        The number of warning lines emitted.
    """
    from flextool import run_warnings
    log = logger if logger is not None else _MODULE_LOGGER
    silent = set(silent_pairs)
    kept = [ov for ov in overlaps if (ov.earlier_solve, ov.later_solve) not in silent]
    pairs = _pairs(kept, lambda ov: (_complete(ov.earlier_solve), _complete(ov.later_solve)))
    n = 0
    for (earlier, later), records in pairs.items():
        for ov in records:
            log.debug("Solves '%s' and '%s' both realize dispatch in period '%s' (%d time steps); the later "
                      "solve's realization was kept.", ov.earlier_solve, ov.later_solve, ov.period,
                      ov.timesteps_dropped)
        periods = len(dict.fromkeys(ov.period for ov in records))
        steps = sum(ov.timesteps_dropped for ov in records)
        if earlier == later:   # two rolls of one rolling solve (a rolling_jump / horizon mix-up)
            run_warnings.warn_grouped("W-0506", steps=steps, periods=_plural(periods, "period"), solve=earlier,
                                      details=", ".join(f"{ov.earlier_solve} / {ov.later_solve} {ov.period}: "
                                                        f"{ov.timesteps_dropped}" for ov in records))
            n += 1
            continue
        run_warnings.warn_grouped("W-0500", [(earlier, later)], steps=steps, periods=_plural(periods, "period"),
                                  details={(earlier, later): ", ".join(f"{ov.period}: {ov.timesteps_dropped}"
                                                                       for ov in records)})
        n += 1
    return n


def _complete(solve: str) -> str:
    from flextool.run_warnings import complete_solve
    return complete_solve(solve) or ""


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
    ``_run_model``: ``realized_invest_periods`` (the ``p_in`` member
    of each ``(p_from, p_in)`` tuple), falling back to the
    ``realized_periods`` that are also ``invest_periods`` when
    ``realized_invest_periods`` is empty but both ``invest_periods`` and
    ``realized_periods`` are set (#319).  Order preserved (authored order
    of the source list), duplicates removed.  ``_run_model`` emits this
    result as ``realized_invest_periods_of_current_solve``, so the overlap
    warning and the emission share one source.
    """
    tuples = list(realized_invest_periods.get(solve, []))
    if tuples:
        return list(dict.fromkeys(t[1] for t in tuples))
    invest = {t[1] for t in invest_periods.get(solve, [])}
    realized = realized_periods.get(solve, [])
    if not invest or not realized:
        return []
    return list(dict.fromkeys(t[1] for t in realized if t[1] in invest))


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
    """One coded warning line (W-0501) per pair of solves that both realize the investments of some periods.

    Returns:
        The number of warning lines emitted.
    """
    from flextool import run_warnings
    log = logger if logger is not None else _MODULE_LOGGER
    n = 0
    for (earlier, later), records in _pairs(overlaps, lambda ov: (ov.earlier_solve, ov.later_solve)).items():
        if earlier == later:
            continue
        periods = list(dict.fromkeys(ov.period for ov in records))
        log.debug("Solves '%s' and '%s' both list %s in realized_invest_periods.", earlier, later,
                  ", ".join(periods))
        shown = f"period '{periods[0]}'" if len(periods) == 1 else _plural(len(periods), "period")
        run_warnings.warn_grouped("W-0501", [(earlier, later)], periods=shown,
                                  details={(earlier, later): ", ".join(periods)})
        n += 1
    return n


# ---------------------------------------------------------------------------
# Dispatch before the investments are realized
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UndecidedDispatch:
    """A solve that realizes the dispatch of periods whose investments no solve up to it has realized."""

    solve: str
    periods: tuple[str, ...]
    investors: tuple[str, ...]


def find_dispatch_before_investments(
    run: Sequence[tuple[str, Collection[str], Collection[str], Collection[str]]],
    invested: Mapping[str, Sequence[str]],
) -> tuple[list[UndecidedDispatch], list[UndecidedDispatch]]:
    """The solves that realize the dispatch of a period before its investments are realized.

    Args:
        run: ``(solve, realized dispatch periods, realized investment periods, investment periods)`` per sub-solve,
            in run order (the rolls of a solve under its name).
        invested: ``{period: solves that list it in invest_periods}`` over the whole run.

    Returns:
        ``(refused, own)``, one record per solve in each (in the order of its first period), naming the periods (in
        run order) and the solves that invest in them.  *refused*: periods that other solves invest in and no solve
        up to this one (this one included) has realized the investments of; the solve does not invest in them
        itself.  *own*: periods whose investments the solve decides itself (the same LP decides the dispatch) but
        that no solve up to it realizes.
    """
    realized: set[str] = set()
    late: dict[str, list[str]] = {}
    own: dict[str, list[str]] = {}
    for solve, dispatch, realized_invest, invest in run:
        realized.update(realized_invest)
        for p in dispatch:
            if p not in invested or p in realized:
                continue
            into = own if p in invest else late
            if p not in into.get(solve, []):
                into.setdefault(solve, []).append(p)

    def records(found: dict[str, list[str]]) -> list[UndecidedDispatch]:
        return [UndecidedDispatch(solve=solve, periods=tuple(periods), investors=tuple(dict.fromkeys(
                    s for p in periods for s in invested[p] if s != solve or found is own)))
                for solve, periods in found.items()]

    return records(late), records(own)


def dispatch_before_investments_message(
    found: Sequence[UndecidedDispatch],
    *,
    names: Mapping[str, str],
    parents: Mapping[str, str],
    per_forecast: Mapping[str, tuple[Collection[str], str]],
) -> str:
    """The error of :func:`find_dispatch_before_investments`, one sentence group per solve.

    Args:
        found: The records, in run order.
        names: ``{solve: how the message names it}`` (a 2-D map copy by its database solve and row); a solve not
            in it is named as is.
        parents: ``{realization solve: the solve it realizes}`` for the realization solves the engine adds: their
            ``realized_periods`` come from the parent, so the fix names the parent.
        per_forecast: ``{investment solve: (its per-forecast investment periods, the method sentence)}`` for the
            solves whose ``invest_solve_branching_method`` makes some investments per forecast branch; the sentence
            is added only where a refused period is one of them.
    """
    def name(solve: str) -> str:
        return names.get(solve, f"solve '{solve}'")

    parts: list[str] = []
    for rec in found:
        parent = parents.get(rec.solve)
        solve = (f"the realization solve '{rec.solve}' of {name(parent)}" if parent is not None
                 else name(rec.solve))
        periods = ", ".join(f"'{p}'" for p in rec.periods)
        them = "it" if len(rec.periods) == 1 else "them"
        word = "period" if len(rec.periods) == 1 else "periods"
        investors = ", ".join(name(s) for s in rec.investors)
        listed = "lists" if len(rec.investors) == 1 else "list"
        its = "its" if len(rec.periods) == 1 else "their"
        if parent is None:
            fix = (f"add {periods} to the realized_invest_periods of an investment solve that runs no later, or "
                   f"leave {them} out of the realized_periods of this solve.")
        else:
            fix = (f"add {periods} to the realized_invest_periods of {name(parent)} (or of an investment solve "
                   f"before it), or leave {them} out of the realized_periods of {name(parent)}: its realization "
                   "solve dispatches those realized_periods on the investments it realizes.")
        text = (
            f"{solve[0].upper()}{solve[1:]} realizes the dispatch of {word} {periods} (realized_periods) before {its} "
            f"investments are realized: {investors} {listed} {them} in invest_periods, and no investment solve up "
            f"to and including this one realizes {them} (realized_invest_periods). A period is dispatched only "
            f"after its investments are decided: {fix}")
        for s in rec.investors:
            branched, why = per_forecast.get(s, ((), ""))
            if any(p in branched for p in rec.periods):
                investor = name(s)
                text += f" {investor[0].upper()}{investor[1:]} {why}"
        parts.append(text)
    return " ".join(parts)


def check_dispatch_after_investments(solve_cfg: Any, run_order: Sequence[tuple[str, Collection[str]]]) -> None:
    """Raise :class:`FlexToolConfigError` when a solve of the run realizes the dispatch of a period before the
    period's investments are realized.

    *run_order*: ``(complete solve, periods)`` per sub-solve in run order, *periods* being those the sub-solve has
    (its active time; a nested solve renamed for one period has that period, a roll its window).  A sub-solve is
    checked and named as its complete solve (a roll and a nested solve renamed for one period by the solve the user
    wrote).  Of each, the ``realized_periods`` in *periods* are its realized dispatch; only an investment solve
    (:func:`is_investment_solve`) invests (``invest_periods``) and realizes investments
    (:func:`effective_realized_invest_periods`), in *periods*.  A realization solve (``realized_dispatch_parent``)
    invests in nothing and dispatches a part of its parent's realized periods on the investments its parent
    realized: it is refused when its parent dispatches a period on investments it does not realize (only the
    realized ones are handed on), named as the realization solve of its parent, with the fix on the parent.  A
    solve's own ``invest_periods`` count as decided for its own dispatch (the same LP decides both); dispatching
    such a period without realizing its investments gives warning W-0641 per solve (the solve of a 2-D period map
    by its database solve), when nothing is refused.  The advice to realize the investments in a later investment
    solve is added only for a refused period whose investments are per forecast branch
    (``SolveConfig.per_forecast_invest_periods``)."""
    from flextool.engine.input._solve_config import is_investment_solve
    from flextool.engine.input._solve_state import FlexToolConfigError

    invested: dict[str, list[str]] = {}
    run: list[tuple[str, list[str], list[str], list[str]]] = []
    for cs, periods in run_order:
        has = {str(p) for p in periods}
        investing = is_investment_solve(solve_cfg, cs)
        if investing:
            for p in dict.fromkeys(str(t[1]) for t in solve_cfg.invest_periods.get(cs, [])):
                if p in has and cs not in invested.setdefault(p, []):
                    invested[p].append(cs)
        realized_invest = [str(p) for p in effective_realized_invest_periods(
            cs, realized_invest_periods=solve_cfg.realized_invest_periods, invest_periods=solve_cfg.invest_periods,
            realized_periods=solve_cfg.realized_periods) if str(p) in has] if investing else []
        dispatch = [str(t[1]) for t in solve_cfg.realized_periods.get(cs, []) if str(t[1]) in has]
        own_invest = [str(t[1]) for t in solve_cfg.invest_periods.get(cs, []) if str(t[1]) in has] if investing else []
        run.append((cs, dispatch, realized_invest, own_invest))
    found, own = find_dispatch_before_investments(run, invested)
    copies = getattr(solve_cfg, "map2d_duplicate_of", None) or {}
    if not found:
        from flextool import run_warnings
        for rec in own:
            run_warnings.warn("W-0641", solve=copies.get(rec.solve, (rec.solve,))[0],
                              periods=", ".join(rec.periods))
        return
    names = {s: f"solve '{parent}' (2D-map entry '{outer}')" for s, (parent, outer) in copies.items()}
    per_forecast: dict[str, tuple[list[str], str]] = {}
    for s in dict.fromkeys(s for rec in found for s in rec.investors):
        branched = solve_cfg.per_forecast_invest_periods(s)
        if not branched:
            continue
        method = solve_cfg.invest_branching_method_for(s)
        reveal = solve_cfg.reveal_period(s)
        which = "after" if method == "investments_after_reveal_period" else "from"
        per_forecast[s] = (branched, (
            f"sets invest_solve_branching_method {method}: its investments {which} the reveal period '{reveal}' "
            f"{'on ' if which == 'from' else ''}are per forecast branch and cannot be realized there; realize them "
            "in a later investment solve, or dispatch only the periods whose investments it realizes."))
    raise FlexToolConfigError(dispatch_before_investments_message(
        found, names=names, parents=getattr(solve_cfg, "realized_dispatch_parent", None) or {},
        per_forecast=per_forecast))
