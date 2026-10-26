"""Where a sub-solve's storage state carry comes from: the latest realised state of an earlier sub-solve.

A storage node continues from the state realised before the sub-solve starts, at the sub-solve's own nesting level
or a deeper one (owner decision Q-c15e-B, with the nested clarification of 2026-10-25):

* a later roll of a rolling solve continues from the previous roll;
* a later solve of a sequence continues from the previous solve (``storage_start_end_method`` affects only the start
  of the whole model, reference.md);
* an upper-level solve continues from the dispatch solved below it (the upward feedback of how_to.md: "the dispatch
  solves can then send the end storage values to the investment and long-term storage solves as start values");
* a lower nested level starts from ``fix_start`` at its first appearance (no realised state exists at its level or
  below), and later instances of the level continue from the latest realised state at that level or below; the
  upper solve supplies only the END of the lower horizon (the nested fix methods).

A node with no realised state yet starts from ``fix_start`` (or free).  ``nesting_level`` counts the
``contains_solve`` hops above a sub-solve.  An auto-spawned realised-dispatch sibling is at its parent's level and
starts where its parent starts: it never sees the state its parent realised in the same window.

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


@dataclass(frozen=True)
class RealisedEntry:
    """The realised end of one sub-solve: its handoff and where that end lies in time."""
    index: int
    solve: str
    level: int
    end_step: "tuple[str, str] | None"      # (period, step) of the last realised step
    end: "Position | None"                  # its position in time (period rank, timeline index)
    roll_end_state: "pl.DataFrame | None"   # [node, value], MWh


class CarryRouter:
    """Run-scoped registry of realised states and the routing rule of the storage carry.

    ``parent_roll``: sub-solve → the parent sub-solve it is nested in (None at the top level).
    ``siblings``: realised-dispatch sibling sub-solve → its parent sub-solve.
    ``position``: ``(period, step) → (period rank, timeline index)`` or None when unknown.
    """

    def __init__(self, parent_roll: dict, siblings: "dict | None" = None,
                 position: "Callable[[str, str], Position | None] | None" = None) -> None:
        self._parent_roll = dict(parent_roll)
        self._siblings = dict(siblings or {})
        self._position = position or (lambda _d, _t: None)
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
        """Append the realised end of ``solve`` (its handoff), when it realised a storage state."""
        if handoff is None:
            return
        state = getattr(handoff, "roll_end_state", None)
        if state is None or state.height == 0:
            return
        end_step = _end_step(handoff)
        end = self._position(*end_step) if end_step is not None else None
        self._entries.append(RealisedEntry(len(self._entries), solve, self.nesting_level(solve), end_step, end,
                                           state))

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
                        "Storage state realised by solve '%s' (ending at %s) is not before the start of solve '%s' "
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
            for node, value in zip(frame["node"].cast(pl.Utf8).to_list(),
                                   frame["value"].cast(pl.Float64, strict=False).to_list()):
                latest[node] = 0.0 if value is None else float(value)
        if not latest:
            return None
        nodes = sorted(latest)
        return pl.DataFrame({"node": nodes, "value": [latest[n] for n in nodes]},
                            schema={"node": pl.Utf8, "value": pl.Float64})

    def _warn_shallower(self, solve: str, level: int, cutoff: int, first, first_step) -> None:
        """The first sub-solve of a level starts from fix_start although a shallower level realised a state before
        it (a level that first appears later in the model): visible, not changed (owner rule)."""
        shallow = [e for e in self._entries[:cutoff] if e.level < level
                   and (first is None or e.end is None or e.end < first)]
        if not shallow:
            return
        entry = shallow[-1]
        key = ("shallower", solve)
        if key in self._warned:
            return
        self._warned.add(key)
        nodes = ", ".join(sorted(str(n) for n in entry.roll_end_state["node"].to_list()))
        _LOG.warning(
            "Solve '%s' is the first solve of its nesting level: its storage node(s) %s start from "
            "storage_start_end_method (fix_start) or free, not from the state solve '%s' realised at a higher level "
            "before it.", solve, nodes, entry.solve)


def _end_step(handoff) -> "tuple[str, str] | None":
    step = getattr(handoff, "roll_end_step", None)
    if step is None or step.height == 0:
        return None
    return str(step["period"][0]), str(step["step"][0])


def make_position(solves: list, active_time_lists: dict, period_branch_lists: dict, *, complete_solve: dict,
                  timesets_used_by_solves: dict, timesets__timeline: dict, timelines: dict,
                  original_timeline: dict) -> Callable:
    """``(period, step) → (period rank, step index)``: periods ranked by first appearance in the run order (a
    stochastic branch period by the period it branches from), steps by their index in the period's base timeline
    (the timeline before any ``new_stepduration`` aggregation, so an aggregated solve's steps and an hourly solve's
    steps compare); None for a step that cannot be placed."""
    rank: dict[str, int] = {}
    index: dict[tuple[str, str], int] = {}
    anchor: dict[str, str] = {}
    base_index: dict[str, dict[str, int]] = {}

    def base_of(timeline: str) -> str:
        seen: set = set()
        while timeline in original_timeline and timeline not in seen:
            seen.add(timeline)
            timeline = original_timeline[timeline]
        return timeline

    for solve in solves:
        for period, branch in period_branch_lists.get(solve, []) or []:
            if branch != period:
                anchor.setdefault(str(branch), str(period))
    for solve in solves:
        timeset_of = dict(timesets_used_by_solves.get(complete_solve.get(solve, solve), []) or [])
        for period, steps in (active_time_lists.get(solve) or {}).items():
            base = anchor.get(str(period), str(period))
            rank.setdefault(base, len(rank))
            timeset = timeset_of.get(period, timeset_of.get(base))
            timeline = timesets__timeline.get(timeset) if timeset is not None else None
            if timeline is None:
                continue
            root = base_of(timeline)
            if root not in base_index:
                base_index[root] = {str(name): i for i, (name, _dur) in enumerate(timelines.get(root, []))}
            names = base_index[root]
            for entry in steps:
                pos = names.get(str(entry.timestep))
                if pos is not None:
                    index.setdefault((base, str(entry.timestep)), pos)

    def position(period: str, step: str) -> "Position | None":
        base = anchor.get(str(period), str(period))
        if base not in rank or (base, str(step)) not in index:
            return None
        return rank[base], index[(base, str(step))]
    return position


def first_step(active_time_list: dict) -> "tuple[str, str] | None":
    """``(period, step)`` of the first step of a sub-solve's active time list."""
    for period, steps in (active_time_list or {}).items():
        if steps:
            return str(period), str(steps[0].timestep)
    return None


__all__ = ["CarryRouter", "RealisedEntry", "first_step", "make_position"]
