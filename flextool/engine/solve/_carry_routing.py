"""Where a sub-solve's roll-forward carries come from: the latest realised state of an earlier sub-solve.

A storage node continues from the state realised before the sub-solve starts, at the sub-solve's own nesting level
or a deeper one (owner decision Q-c15e-B, with the nested clarification of 2026-10-25):

* a later roll of a rolling solve continues from the previous roll;
* a later solve of a sequence continues from the previous solve (``storage_start_end_method`` affects only the start
  of the whole model, reference/nodes.md);
* an upper-level solve continues from the dispatch solved below it (the upward feedback of how_to/solve_sequences.md: "the dispatch
  solves can then send the end storage values to the investment and long-term storage solves as start values");
* a lower nested level starts from ``fix_start`` at its first appearance (no realised state exists at its level or
  below), and later instances of the level continue from the latest realised state at that level or below; the
  upper solve supplies only the END of the lower horizon (the nested fix methods).

A node with no realised state yet starts from ``fix_start`` (or free).  ``nesting_level`` counts the
``contains_solve`` hops above a sub-solve.  An auto-spawned realised-dispatch sibling is at its parent's level and
starts where its parent starts: it never sees the state its parent realised in the same window.

The same rule routes the other carriers tied to the realised end step (:data:`STEP_FIELDS`: the committed online
state, the ramp-limited flows and the step itself, the min up / down time history), taken whole from the latest
entry: the next roll's unit commitment and ramps continue from the dispatch that was realised last, also across a
storage or investment solve in between.  The routed ``roll_end_step`` also says where that step ends on the base
timeline (:func:`with_end_position`), so a solve with longer steps continues the finer dispatch realized before it
when its own first step's predecessor ends at the same time.  The running totals (``cumulative_co2``, ``cumulative_commodity``,
``cum_sim_hours``) are not routed: every sub-solve passes them on (its own realised part added), so the sequential
predecessor already holds the total realised before the sub-solve, at any level.

The registry is run-scoped (one :class:`CarryRouter` per cascade run), entries are appended in run order (which is
deterministic), and "latest" means the last appended entry.  The realised end step of each entry must lie before the
consuming sub-solve's first step in time; an entry that does not is not used (a warning names it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import polars as pl

_LOG = logging.getLogger(__name__)

Position = tuple[int, int]
# (period rank, base timeline, base-timeline index after the step, branch period or None)
EndPosition = tuple[int, str, int, "str | None"]

# The SolveHandoff carriers tied to the realised end step, routed with the storage state.
STEP_FIELDS: tuple[str, ...] = ("roll_end_online", "roll_end_flow", "roll_end_step", "online_history")


@dataclass(frozen=True)
class RealisedEntry:
    """The realised end of one sub-solve: its handoff and where that end lies in time."""
    index: int
    solve: str
    level: int
    end_step: "tuple[str, str] | None"      # (period, step) of the last realised step
    end: "Position | None"                  # its position in time (period rank, timeline index)
    roll_end_state: "pl.DataFrame | None"   # [node, value], MWh
    handoff: object = None                  # the SolveHandoff (the STEP_FIELDS carriers)


class CarryRouter:
    """Run-scoped registry of realised states and the routing rule of the storage carry.

    ``parent_roll``: sub-solve → the parent sub-solve it is nested in (None at the top level).
    ``siblings``: realised-dispatch sibling sub-solve → its parent sub-solve.
    ``position``: ``(period, step) → (period rank, timeline index)`` or None when unknown.
    ``end_of``: ``(period, step, duration) → EndPosition`` or None (:meth:`_TimeIndex.end_of`); the routed
    ``roll_end_step`` carries it (:func:`with_end_position`).
    """

    def __init__(self, parent_roll: dict, siblings: "dict | None" = None,
                 position: "Callable[[str, str], Position | None] | None" = None,
                 end_of: "Callable[[str, str, float], EndPosition | None] | None" = None) -> None:
        self._parent_roll = dict(parent_roll)
        self._siblings = dict(siblings or {})
        self._position = position or (lambda _d, _t: None)
        self._end_of = end_of
        self._entries: list[RealisedEntry] = []
        self._cutoff: dict[str, int] = {}
        self._warned: set = set()

    # -- structure ---------------------------------------------------------------------------------------------

    def nesting_level(self, solve: str) -> int:
        """0 for a top-level sub-solve, +1 per contains_solve above it; a sibling takes its parent's level."""
        parent = self._siblings.get(solve)
        if parent is not None and parent != solve:
            return self.nesting_level(parent)
        level, seen = 0, {solve}
        node = self._parent_roll.get(solve)
        while node is not None and node not in seen:
            seen.add(node)
            level += 1
            node = self._parent_roll.get(node)
        return level

    # -- run-time --------------------------------------------------------------------------------------------

    def start(self, solve: str) -> None:
        """Mark the start of ``solve``: the entries it may draw from are those recorded so far (a sibling: those
        recorded when its parent started)."""
        parent = self._siblings.get(solve)
        if parent is not None and parent in self._cutoff:
            self._cutoff[solve] = self._cutoff[parent]
        else:
            self._cutoff[solve] = len(self._entries)

    def record(self, solve: str, handoff) -> None:
        """Append the realised end of ``solve`` (its handoff), when it realised a step or a storage state."""
        if handoff is None:
            return
        state = getattr(handoff, "roll_end_state", None)
        if state is not None and state.height == 0:
            state = None
        end_step = _end_step(handoff)
        if state is None and end_step is None:
            return
        end = self._position(*end_step) if end_step is not None else None
        self._entries.append(RealisedEntry(len(self._entries), solve, self.nesting_level(solve), end_step, end,
                                           state, handoff))

    def sources(self, solve: str, first_step: "tuple[str, str] | None") -> list[RealisedEntry]:
        """The entries ``solve`` may continue from, oldest first: recorded before it started, at its nesting level
        or deeper, and realised before its first step in time (when both positions are known)."""
        level = self.nesting_level(solve)
        cutoff = self._cutoff.get(solve, len(self._entries))
        first = self._position(*first_step) if first_step is not None else None
        out = []
        for entry in self._entries[:cutoff]:
            if entry.level < level:
                continue
            if first is not None and entry.end is not None and not entry.end < first:
                key = (solve, entry.solve)
                if key not in self._warned:
                    self._warned.add(key)
                    _LOG.warning(
                        "The state realised by solve '%s' (ending at %s) is not before the start of solve '%s' "
                        "(%s); it is not carried into it.", entry.solve, entry.end_step, solve, first_step)
                continue
            out.append(entry)
        if not out:
            self._warn_shallower(solve, level, cutoff, first, first_step)
        return out

    def roll_end_state(self, solve: str, first_step: "tuple[str, str] | None") -> "pl.DataFrame | None":
        """``[node, value]`` (MWh, sorted by node): per node the state of the latest entry holding it; None when no
        entry applies (the sub-solve is the first of its level for every node)."""
        latest: dict[str, float] = {}
        for entry in self.sources(solve, first_step):
            frame = entry.roll_end_state
            if frame is None:
                continue
            for node, value in zip(frame["node"].cast(pl.Utf8).to_list(),
                                   frame["value"].cast(pl.Float64, strict=False).to_list()):
                latest[node] = 0.0 if value is None else float(value)
        if not latest:
            return None
        nodes = sorted(latest)
        return pl.DataFrame({"node": nodes, "value": [latest[n] for n in nodes]},
                            schema={"node": pl.Utf8, "value": pl.Float64})

    def step_carriers(self, solve: str, first_step: "tuple[str, str] | None") -> dict:
        """``{field: frame | None}`` of :data:`STEP_FIELDS` from the latest entry with a realised end step (all None
        when there is none: the sub-solve is the first of its level)."""
        entries = [e for e in self.sources(solve, first_step) if e.end_step is not None]
        handoff = entries[-1].handoff if entries else None
        out = {f: (getattr(handoff, f, None) if handoff is not None else None) for f in STEP_FIELDS}
        if self._end_of is not None:
            out["roll_end_step"] = with_end_position(out["roll_end_step"], self._end_of)
        return out

    def _warn_shallower(self, solve: str, level: int, cutoff: int, first, first_step) -> None:
        """The first sub-solve of a level starts from fix_start although a shallower level realised a state before
        it (a level that first appears later in the model): visible, not changed (owner rule)."""
        shallow = [e for e in self._entries[:cutoff] if e.level < level
                   and (first is None or e.end is None or e.end < first)]
        if not shallow:
            return
        stateful = [e for e in shallow if e.roll_end_state is not None]
        if not stateful:
            return
        entry = stateful[-1]
        key = ("shallower", solve)
        if key in self._warned:
            return
        self._warned.add(key)
        nodes = ", ".join(sorted(str(n) for n in entry.roll_end_state["node"].to_list()))
        _LOG.warning(
            "Solve '%s' is the first solve of its nesting level: its storage node(s) %s start from "
            "storage_start_end_method (fix_start) or free, not from the state solve '%s' realised at a higher level "
            "before it.", solve, nodes, entry.solve)


class RoutedCarriers:
    """The routed :data:`STEP_FIELDS` frames as attributes (the shape the handoff translator reads)."""

    def __init__(self, frames: dict) -> None:
        for name in STEP_FIELDS:
            setattr(self, name, frames.get(name))


END_POSITION_COLUMNS: tuple[str, ...] = ("end_rank", "end_root", "end_index", "end_branch")


def with_end_position(step: "pl.DataFrame | None",
                      end_of: "Callable[[str, str, float], EndPosition | None]") -> "pl.DataFrame | None":
    """``roll_end_step`` ``[period, step, duration]`` with :data:`END_POSITION_COLUMNS` (Utf8, empty when the step
    cannot be placed): where the carried step ends on the base timeline.  A solve whose first step continues a step
    of another length (a coarser solve after a finer dispatch) matches the carried step by this position
    (``params._ramp_links.ramp_roll_carry``).  ``None`` / empty frames pass through."""
    if step is None or step.height == 0:
        return step
    base = step.drop([c for c in END_POSITION_COLUMNS if c in step.columns])
    cols: dict[str, list[str]] = {c: [] for c in END_POSITION_COLUMNS}
    for period, name, duration in zip(base["period"].cast(pl.Utf8).to_list(), base["step"].cast(pl.Utf8).to_list(),
                                      base["duration"].cast(pl.Utf8).to_list()):
        pos = end_of(period, name, duration) if period is not None and name is not None else None
        for c, v in zip(END_POSITION_COLUMNS, pos if pos is not None else (None,) * 4):
            cols[c].append("" if v is None else str(v))
    return base.with_columns(pl.Series(c, v, dtype=pl.Utf8) for c, v in cols.items())


def _end_step(handoff) -> "tuple[str, str] | None":
    step = getattr(handoff, "roll_end_step", None)
    if step is None or step.height == 0:
        return None
    return str(step["period"][0]), str(step["step"][0])


class _TimeIndex:
    """Places the steps of the run's sub-solves in time: periods ranked by first appearance in the run order (a
    stochastic branch period by the period it branches from), steps by their index in the period's base timeline
    (the timeline before any ``new_stepduration`` aggregation, so an aggregated solve's steps and an hourly solve's
    steps compare)."""

    def __init__(self, solves: list, active_time_lists: dict, period_branch_lists: dict, *, complete_solve: dict,
                 timesets_used_by_solves: dict, timesets__timeline: dict, timelines: dict,
                 original_timeline: dict) -> None:
        self._rank: dict[str, int] = {}
        self._index: dict[tuple[str, str], int] = {}
        self._anchor: dict[str, str] = {}
        self._ends: dict[str, "Position | None"] = {}
        # (base period, step) → (base timeline, index) of every placement; a step placed on two different spots
        # (two solves putting the same period on different base timelines) has no end position (end_of).
        self._placed: dict[tuple[str, str], "tuple[str, int] | None"] = {}
        self._hours: dict[str, list[float]] = {}

        def base_of(timeline: str) -> str:
            seen: set = set()
            while timeline in original_timeline and timeline not in seen:
                seen.add(timeline)
                timeline = original_timeline[timeline]
            return timeline

        base_index: dict[str, dict[str, int]] = {}
        base_hours: dict[str, list[float]] = {}
        for solve in solves:
            for period, branch in period_branch_lists.get(solve, []) or []:
                if branch != period:
                    self._anchor.setdefault(str(branch), str(period))
        for solve in solves:
            timeset_of = dict(timesets_used_by_solves.get(complete_solve.get(solve, solve), []) or [])
            ends: list[Position] = []
            for period, steps in (active_time_lists.get(solve) or {}).items():
                base = self._anchor.get(str(period), str(period))
                self._rank.setdefault(base, len(self._rank))
                timeset = timeset_of.get(period, timeset_of.get(base))
                timeline = timesets__timeline.get(timeset) if timeset is not None else None
                if timeline is None:
                    continue
                root = base_of(timeline)
                if root not in base_index:
                    base_index[root] = {str(name): i for i, (name, _dur) in enumerate(timelines.get(root, []))}
                    base_hours[root] = [float(dur) for _name, dur in timelines.get(root, [])]
                names = base_index[root]
                for entry in steps:
                    pos = names.get(str(entry.timestep))
                    if pos is not None:
                        key = (base, str(entry.timestep))
                        self._index.setdefault(key, pos)
                        if key not in self._placed:
                            self._placed[key] = (root, pos)
                        elif self._placed[key] != (root, pos):
                            self._placed[key] = None
                if steps:
                    last = steps[-1]
                    pos = names.get(str(last.timestep))
                    if pos is not None:
                        ends.append((self._rank[base], _covered_to(base_hours[root], pos, float(last.duration))))
            self._ends[solve] = max(ends) if ends else None
        self._hours = base_hours

    def position(self, period: str, step: str) -> "Position | None":
        """``(period rank, base-timeline index)`` of the start of the step; None when it cannot be placed."""
        base = self._anchor.get(str(period), str(period))
        if base not in self._rank or (base, str(step)) not in self._index:
            return None
        return self._rank[base], self._index[(base, str(step))]

    def end_of(self, period: str, step: str, duration: float) -> "EndPosition | None":
        """Where a step ends on the base timeline: ``(period rank, base timeline, base index after the step, branch)``.

        ``branch`` is the period itself for a stochastic branch period and None otherwise (a branch period ranks with
        the period it branches from, so two sibling branches would otherwise share every position).  None when the
        step cannot be placed, when two solves place it on different base timelines or indices, or when its
        ``duration`` hours do not cover whole base steps exactly (a partial cover cannot be told apart from the
        next whole one)."""
        base = self._anchor.get(str(period), str(period))
        placed = self._placed.get((base, str(step)))
        if base not in self._rank or placed is None:
            return None
        root, start = placed
        hours = self._hours.get(root)
        try:
            dur = float(duration)
        except (TypeError, ValueError):
            return None
        if hours is None or not dur > 0:
            return None
        end, acc = start, 0.0
        while end < len(hours) and acc < dur - 1e-9:
            acc += hours[end]
            end += 1
        if abs(acc - dur) > 1e-6:
            return None
        branch = str(period) if str(period) in self._anchor else None
        return self._rank[base], root, end, branch

    def end(self, solve: str) -> "Position | None":
        """``(period rank, base-timeline index after the solve's last step)``: where the sub-solve's horizon ends (the
        latest end over its periods, so a stochastic solve ends with its terminal branches); None when unknown."""
        return self._ends.get(solve)


def _covered_to(hours: list[float], start: int, duration: float) -> int:
    """The base-timeline index after a step that starts at base index ``start`` and lasts ``duration`` hours (an
    aggregated step covers several base steps)."""
    end, acc = start, 0.0
    while end < len(hours) and acc < duration - 1e-9:
        acc += hours[end]
        end += 1
    return max(end, start + 1)


def make_time_index(solves: list, active_time_lists: dict, period_branch_lists: dict, *, complete_solve: dict,
                    timesets_used_by_solves: dict, timesets__timeline: dict, timelines: dict,
                    original_timeline: dict) -> _TimeIndex:
    """The run's :class:`_TimeIndex` (``position`` for the carry routing, ``end_of`` for the seam match of a
    continuing solve)."""
    return _TimeIndex(solves, active_time_lists, period_branch_lists, complete_solve=complete_solve,
                      timesets_used_by_solves=timesets_used_by_solves, timesets__timeline=timesets__timeline,
                      timelines=timelines, original_timeline=original_timeline)


def make_position(solves: list, active_time_lists: dict, period_branch_lists: dict, *, complete_solve: dict,
                  timesets_used_by_solves: dict, timesets__timeline: dict, timelines: dict,
                  original_timeline: dict) -> Callable:
    """``(period, step) → (period rank, step index)`` (:class:`_TimeIndex`); None for a step that cannot be placed."""
    return _TimeIndex(solves, active_time_lists, period_branch_lists, complete_solve=complete_solve,
                      timesets_used_by_solves=timesets_used_by_solves, timesets__timeline=timesets__timeline,
                      timelines=timelines, original_timeline=original_timeline).position


def reaches_model_end(solves: list, active_time_lists: dict, period_branch_lists: dict, *, complete_solve: dict,
                      timesets_used_by_solves: dict, timesets__timeline: dict, timelines: dict,
                      original_timeline: dict, siblings: "dict | None" = None) -> list[bool]:
    """Per sub-solve (by position in ``solves``): whether its horizon holds the last step of the entire model.

    The model's last step is the latest end over every sub-solve of the run (:meth:`_TimeIndex.end`: base-timeline
    position, so an aggregated solve whose last coarse step covers the last hour reaches it too).  fix_end /
    fix_start_end pin the end state in every sub-solve that reaches it (owner decision Q-c15f-D): an investment solve
    and the last dispatch roll of a sequence, each nested level's solve that holds it, a rolling solve's rolls whose
    horizon (look-ahead included) runs to it.  A sub-solve that cannot be placed in time does not reach it.  When no
    sub-solve can be placed, the last sub-solve of the run reaches it (with its realised-dispatch sibling,
    ``siblings``: sibling → parent).
    """
    index = _TimeIndex(solves, active_time_lists, period_branch_lists, complete_solve=complete_solve,
                       timesets_used_by_solves=timesets_used_by_solves, timesets__timeline=timesets__timeline,
                       timelines=timelines, original_timeline=original_timeline)
    ends = [index.end(solve) for solve in solves]
    known = [e for e in ends if e is not None]
    if known:
        last = max(known)
        return [e == last for e in ends]
    siblings = siblings or {}
    tail = next((s for s in reversed(solves) if siblings.get(s, s) == s), None)
    return [s == tail or (siblings.get(s) == tail and tail is not None) for s in solves]


def first_step(active_time_list: dict) -> "tuple[str, str] | None":
    """``(period, step)`` of the first step of a sub-solve's active time list."""
    for period, steps in (active_time_list or {}).items():
        if steps:
            return str(period), str(steps[0].timestep)
    return None


__all__ = ["STEP_FIELDS", "CarryRouter", "RealisedEntry", "RoutedCarriers", "first_step", "make_position",
           "make_time_index", "reaches_model_end", "with_end_position", "END_POSITION_COLUMNS"]
