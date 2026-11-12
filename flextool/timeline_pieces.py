"""The time steps a solve with ``new_stepduration`` really uses: the timeline-chunking core, stdlib only.

The engine (:meth:`flextool.engine.input._timeline.TimelineConfig.create_timeline_from_timestep_duration`)
aggregates a solve's timelines with the functions here, so a caller outside the engine (the AI assistant's model
facts) gets the same steps without importing :mod:`flextool.engine`:

* :func:`aggregate_timeline_pieces` cuts one timeline at every block start and end of the given timesets (and
  at extra cuts such as representative base-period starts) and aggregates each piece to the new step length;
* :func:`group_step_solves` groups the solves with ``new_stepduration`` that share a timeset (they share its
  aggregated timeline; the first solve names it);
* :func:`timeline_pieces_of_group` splits a group's timesets by timeline, with the representative base-period
  cuts of each timeline;
* :func:`solve_step_chunks` puts them together for one solve, as the engine does.

:func:`quiet_logger` gives a logger that prints nothing, for callers that must not write to the terminal.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple

__all__ = [
    "SolveTimeline",
    "aggregate_timeline_pieces",
    "group_step_solves",
    "quiet_logger",
    "solve_step_chunks",
    "timeline_pieces_of_group",
]

QUIET_LOGGER_NAME = "flextool.timeline_pieces.quiet"


def quiet_logger(name: str = QUIET_LOGGER_NAME) -> logging.Logger:
    """A logger that prints nothing: it does not pass records to its parents (``propagate`` off) and its only
    handler is a :class:`logging.NullHandler` (a logger without any handler would print warnings through
    :data:`logging.lastResort`)."""
    log = logging.getLogger(name)
    log.propagate = False
    if len(log.handlers) != 1 or not isinstance(log.handlers[0], logging.NullHandler):
        for handler in list(log.handlers):
            log.removeHandler(handler)
        log.addHandler(logging.NullHandler())
    return log


def aggregate_timeline_pieces(
    old_steps: list,
    timeset_blocks: dict[str, list],
    step_duration: float,
    logger: logging.Logger | None,
    extra_cuts: dict[str, list[str]] | None = None,
    *,
    solve_name: str,
    timeline_name: str,
    check_fit: bool = True,
    error_cls: type[Exception] = ValueError,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, int]]]]:
    """Aggregate a whole timeline to ``step_duration`` hours, piece by piece.

    ``old_steps`` is the original ``[(step, duration), ...]``; the pieces are
    the runs between the block starts and block ends of ``timeset_blocks``
    (``{timeset: [(start, count), ...]}``) and the steps in ``extra_cuts``
    (``{owner: [step, ...]}``, e.g. representative base-period starts).  Each piece is aggregated from its
    first step: a chunk closes as soon as its steps add up to ``step_duration`` hours (within a relative
    tolerance of 1e-4, so steps entered with rounded fractions still fit), and a chunk always holds at least
    one step.  The steps may have any length (0.5 h, 2 h, mixed).  The
    tolerance is capped at half the shortest step, so a long new step still
    closes at the right step.
    Returns the chunks ``[(first step, str(hours)), ...]`` in timeline order and
    per timeset ``[(block start, number of chunks in the block), ...]`` (a block
    whose start is not on the timeline is kept as given; ``get_active_time``
    skips it).  Logs a warning when a block ends with a shorter chunk, naming
    the other timeset when its boundary is what cut the chunk short.
    ``logger`` ``None`` = :func:`quiet_logger`.

    Raises ``error_cls`` (the engine passes ``FlexToolConfigError``; only with ``check_fit``, i.e. for a
    solve the model runs) when a chunk that starts inside a block
    would be longer than ``step_duration``: the new step is not a whole multiple
    of the timeline's steps, or (steps of different lengths) it would end inside
    a step.  A gap chunk (outside every block, never active) may be longer, e.g.
    one long filler step between two blocks.
    """
    if logger is None:
        logger = quiet_logger()
    index_of = {step[0]: i for i, step in enumerate(old_steps)}
    n_steps = len(old_steps)
    # Block index ranges [first, end) per timeset, and who owns each cut.
    ranges: dict[str, list[tuple[str, int, int] | None]] = {}
    cut_owners: dict[int, set[str]] = {0: set()}
    for ts, blocks in timeset_blocks.items():
        ranges[ts] = []
        for start, count in blocks:
            first = index_of.get(start)
            if first is None:
                ranges[ts].append(None)
                continue
            end = min(first + int(float(count)), n_steps)
            ranges[ts].append((start, first, end))
            cut_owners.setdefault(first, set()).add(ts)
            cut_owners.setdefault(end, set()).add(ts)
    for owner, labels in (extra_cuts or {}).items():
        for label in labels:
            if label in index_of:
                cut_owners.setdefault(index_of[label], set()).add(owner)
    cuts = sorted(c for c in cut_owners if c < n_steps) + [n_steps]
    block_ranges = [(rng[1], rng[2]) for rngs in ranges.values() for rng in rngs if rng is not None]
    # Relative to the new step (rounded fractions like 0.333333 h still add
    # up), but below half a base step, so a long new step never closes one
    # base step early.
    positive = [float(step[1]) for step in old_steps if float(step[1]) > 0]
    tol = min(1e-4 * step_duration, 0.5 * min(positive)) if positive else 0.0

    chunks: list[tuple[str, str]] = []
    chunk_first: list[int] = []
    short_chunks: list[tuple[int, int, float]] = []   # (first, end, hours)
    for piece_first, piece_end in zip(cuts, cuts[1:]):
        i = piece_first
        while i < piece_end:
            first = i
            hours = 0.0
            while i < piece_end and (hours < step_duration - tol or i == first):
                hours += float(old_steps[i][1])
                i += 1
            label = old_steps[first][0]
            if (check_fit and hours > step_duration + tol
                    and any(b_first <= first < b_end for b_first, b_end in block_ranges)):
                ds = [float(step[1]) for step in old_steps[first:i]]
                if all(d == ds[0] for d in ds):
                    message = (
                        f"Solve '{solve_name}': new_stepduration {step_duration:g} h is not a whole multiple of "
                        f"the {ds[0]:g} h time steps of timeline '{timeline_name}' (the new time step starting "
                        f"at '{label}' would be {hours:g} h long). Give a new_stepduration that is a whole "
                        f"multiple of the timeline's time steps."
                    )
                else:
                    message = (
                        f"Solve '{solve_name}': new_stepduration {step_duration:g} h does not fit the time steps "
                        f"of timeline '{timeline_name}': the new time step starting at '{label}' would end inside "
                        f"base time step '{old_steps[i - 1][0]}' ({hours:g} h in total). On a timeline with time "
                        f"steps of different lengths, every new time step must end where a time step of the "
                        f"timeline ends."
                    )
                logger.error(message)
                raise error_cls(message)
            chunks.append((label, str(hours)))
            chunk_first.append(first)
            if hours < step_duration - tol:
                short_chunks.append((first, i, hours))

    block_chunks: dict[str, list[tuple[str, int]]] = {}
    for ts, blocks in timeset_blocks.items():
        out: list[tuple[str, int]] = []
        for (start, count), rng in zip(blocks, ranges[ts]):
            if rng is None:
                out.append((start, count))
                continue
            _start, first, end = rng
            out.append((start, sum(1 for c in chunk_first if first <= c < end)))
        block_chunks[ts] = out

    warned: set[tuple[str, str]] = set()
    for first, end, hours in short_chunks:
        for ts, rngs in ranges.items():
            for rng in rngs:
                if rng is None or not (rng[1] <= first < rng[2]) or (ts, rng[0]) in warned:
                    continue
                warned.add((ts, rng[0]))
                label = old_steps[first][0]
                others = sorted(cut_owners.get(end, set()) - {ts}) if end < rng[2] else []
                if others:
                    logger.warning(
                        f"new_stepduration {step_duration} h: step {label!r} of timeset {ts!r} "
                        f"(block starting {rng[0]!r}) is only {hours} h long, because a block "
                        f"or base period of {', '.join(repr(o) for o in others)} on the same "
                        f"timeline starts or ends inside it."
                    )
                else:
                    logger.warning(
                        f"new_stepduration {step_duration} h: the block of timeset {ts!r} "
                        f"starting {rng[0]!r} is not a multiple of the step; its last step "
                        f"{label!r} is {hours} h long."
                    )
    return chunks, block_chunks


def group_step_solves(
    step_solves: Mapping[str, float],
    timesets_of: Callable[[str], list[str]],
    has_blocks: Callable[[str], bool],
) -> list[tuple[str, float, list[str]]]:
    """Group the solves with ``new_stepduration`` that share a timeset: they share its aggregated timeline.

    ``step_solves`` is ``{solve: new_stepduration}`` in the order of the solves' ``new_stepduration`` values;
    ``timesets_of(solve)`` the solve's timesets in order and ``has_blocks(timeset)`` whether the timeset has
    ``timeset_duration`` blocks (a timeset without blocks is left out).  Returns
    ``[(first solve, its new_stepduration, [timeset, ...]), ...]``: the first solve of a group names its
    timeline.  Two groups that a later solve joins are merged into the earlier one; the later group stays in
    the list with no timesets.
    """
    groups: list[tuple[str, float, list[str]]] = []
    group_of_timeset: dict[str, int] = {}
    for solve_name, sd in step_solves.items():
        timesets = [ts for ts in timesets_of(solve_name) if has_blocks(ts)]
        hit = sorted({group_of_timeset[ts] for ts in timesets if ts in group_of_timeset})
        if not hit:
            groups.append((solve_name, sd, []))
            gi = len(groups) - 1
        else:
            gi = hit[0]
            for other in hit[1:]:
                groups[gi][2].extend(groups[other][2])
                for ts in groups[other][2]:
                    group_of_timeset[ts] = gi
                groups[other] = (groups[other][0], groups[other][1], [])
        for ts in timesets:
            if ts not in groups[gi][2]:
                groups[gi][2].append(ts)
            group_of_timeset[ts] = gi
    return groups


def timeline_pieces_of_group(
    timesets: Iterable[str],
    timeline_of: Mapping[str, str],
    rp_weights: Mapping[str, Iterable] | None = None,
) -> dict[str, tuple[list[str], dict[str, list[str]]]]:
    """``{timeline: ([timeset, ...], base cuts)}`` of a group's timesets, by the timeline each is on.

    The base cuts are ``{"<timeset> (representative_period_weights)": [base period, ...]}`` of the timesets with
    representative periods (``rp_weights[timeset]``: the base periods are its keys, or any iterable of the
    base-period labels): every base period then starts at a step of the aggregated timeline.
    """
    rp_weights = rp_weights or {}
    by_timeline: dict[str, list[str]] = {}
    for ts in timesets:
        by_timeline.setdefault(timeline_of[ts], []).append(ts)
    out: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
    for timeline_name, tl_timesets in by_timeline.items():
        base_cuts = {
            f"{ts} (representative_period_weights)": [str(b) for b in rp_weights[ts]]
            for ts in tl_timesets if rp_weights.get(ts)
        }
        out[timeline_name] = (tl_timesets, base_cuts)
    return out


class SolveTimeline(NamedTuple):
    """One aggregated timeline of a solve."""

    #: The engine's name of the aggregated timeline: ``<original timeline>_<first solve of the group>``.
    name: str
    #: ``[(first original step, str(hours)), ...]`` in timeline order (gap chunks included).
    chunks: list[tuple[str, str]]
    #: ``{timeset: [(block start, number of chunks in the block), ...]}`` of the group's timesets on the timeline.
    block_chunks: dict[str, list[tuple[str, int]]]


def solve_step_chunks(
    solve_name: str,
    *,
    new_step_durations: Mapping[str, float | None],
    timesets_of_solve: Mapping[str, Iterable[str]],
    timeset_blocks: Mapping[str, list],
    timeline_of_timeset: Mapping[str, str],
    timelines: Mapping[str, list],
    rp_weights: Mapping[str, Iterable] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, SolveTimeline]:
    """The aggregated timelines the engine builds for ``solve_name``: ``{original timeline: SolveTimeline}``.

    Mirrors the engine's grouping: the solves with ``new_stepduration`` (``new_step_durations``, ``None`` = not
    set) that share a timeset share its aggregated timeline, which is cut at the blocks of every timeset of the
    group on that timeline and at their representative base-period starts.  ``timesets_of_solve`` is
    ``{solve: [timeset, ...]}`` (the timesets of the solve's periods, repeats allowed), ``timeset_blocks``
    ``{timeset: [(block start, count), ...]}`` (the original ``timeset_duration`` blocks), ``timeline_of_timeset``
    the original timeline of each timeset and ``timelines`` ``{timeline: [(step, duration), ...]}``.

    The fit of the new step is not checked (``check_fit`` off) and the shorter-step warnings go to ``logger``
    (``None`` = :func:`quiet_logger`).  An empty dict when the solve has no ``new_stepduration`` (it runs on the
    original steps) or none of its timesets has blocks.
    """
    step_solves: dict[str, float] = {s: float(raw) for s, raw in new_step_durations.items() if raw is not None}
    if solve_name not in step_solves:
        return {}

    def _timesets_of(name: str) -> list[str]:
        return list(dict.fromkeys(timesets_of_solve.get(name, ())))

    groups = group_step_solves(step_solves, _timesets_of, lambda ts: bool(timeset_blocks.get(ts)))
    own = {ts for ts in _timesets_of(solve_name) if timeset_blocks.get(ts)}
    out: dict[str, SolveTimeline] = {}
    for first_solve, step_duration, timesets in groups:
        if not own & set(timesets):
            continue
        pieces = timeline_pieces_of_group(timesets, timeline_of_timeset, rp_weights)
        for timeline_name, (tl_timesets, base_cuts) in pieces.items():
            chunks, block_chunks = aggregate_timeline_pieces(
                timelines[timeline_name],
                {ts: timeset_blocks[ts] for ts in tl_timesets},
                step_duration,
                logger,
                extra_cuts=base_cuts,
                solve_name=first_solve,
                timeline_name=timeline_name,
                check_fit=False,
            )
            out[timeline_name] = SolveTimeline(timeline_name + "_" + first_solve, chunks, block_chunks)
    return out
