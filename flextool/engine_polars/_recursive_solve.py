"""Recursive solve structure builder — rolling/nested solve expansion.

Γ.8.C foundation module.  Direct 1:1 port of
``flextool/flextoolrunner/recursive_solves.py`` (505 LOC; the read-only
reference at ``flextool-engine/flextool/flextoolrunner/recursive_solves.py``).

Architecture notes
------------------

* This is the algorithmically most-complex orchestration module.  It
  walks the solve tree, expanding rolling-window solves into per-roll
  sub-solves and threading parent scope through nested ``contains_solves``
  chains.  Every orchestration phase downstream (Γ.8.D) reads the flat
  ``all_solves`` list and per-solve time lists this module produces; any
  divergence from the flextool reference manifests as wrong solve
  ordering, missing rolls, or mis-scoped active times.

* :class:`RecursiveSolveBuilder` mutates several fields of
  ``state.solve``:

  - ``roll_counter[solve] += 1`` per roll (R-O5 in the audit).  Reset
    at the start of every top-level ``define_solve_recursive`` call so
    repeated invocations on the same SolveConfig don't desync.
  - ``real_solves`` accumulates every solve name encountered (including
    renamed solves).
  - ``first_of_complete_solve`` / ``last_of_solve`` track per-complete-solve
    boundary names for state initial / terminal constraints downstream.
  - ``duplicate_solve(...)`` is invoked for the single-matching-period
    rename path which fans an input solve into ``solve + "_" + period``.

* Lazy import of :class:`StochasticSolver`: ``recursive_solve`` only
  needs ``find_next_timestep`` (called once when a child is constrained
  by a parent's roll start).  ``_stochastic`` does not currently import
  back, but the lazy import is defensive against future cross-module
  growth.

* :func:`get_active_time` is imported from :mod:`_timeline` and called
  at every node of the tree walk.

Behavioural quirks (preserved verbatim from flextool)
-----------------------------------------------------

* **Rolling expansion** produces names ``solve + "_roll_" + str(counter)``.
  The counter is sourced from ``state.solve.roll_counter[solve]`` and
  incremented per roll.

* **Single-matching-period rename**: when a child solve's period set
  intersects its parent's by exactly one period, the child is renamed
  to ``solve + "_" + period`` and ``duplicate_solve`` carbon-copies the
  parent's lockstep dicts under the new name.  ``duplicate_solve`` does
  NOT carbon-copy the period attributes (``invest_periods``,
  ``realized_periods``, …); the empty-scope fallback at the parent_scope
  computation site catches this.

* **Multi-period child** (``len(matching_periods) > 1``) is the
  single-solve-with-multi-period parent case: keep the original solve
  name and let the active-time filter constrain it later.

* **"More than one solve in a rolling solve, not managed"** raise is
  preserved because the fix-storage handoff between sibling rolling
  children of a rolling parent isn't implemented downstream.

Reference: ``flextool/flextoolrunner/recursive_solves.py``.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from typing import TYPE_CHECKING

from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    SolveResult,
)
from flextool.engine_polars._timeline import get_active_time

if TYPE_CHECKING:
    from flextool.engine_polars._solve_state import RunnerState


# ---------------------------------------------------------------------------
# ParentSolveInfo — walked through the tree
# ---------------------------------------------------------------------------


ParentSolveInfo = namedtuple("ParentSolveInfo", ["solve", "roll"])
"""Parent context threaded into recursive ``define_solve_recursive`` calls.

* ``solve``: parent solve name, or ``None`` for top-level solves.
* ``roll``: parent's per-roll name, or ``None`` if the parent is single-
  solve mode (or the call is top-level).
"""


# ---------------------------------------------------------------------------
# RecursiveSolveBuilder
# ---------------------------------------------------------------------------


class RecursiveSolveBuilder:
    """Walks the solve tree and produces the flat ``all_solves`` ordering.

    The entry point :meth:`define_solve_recursive` is called once per
    top-level solve in ``state.solve.model_solve``.  It dispatches to
    :meth:`_process_rolling_solve` or :meth:`_process_single_solve`
    based on the current solve's ``solve_mode`` and threads parent
    scope through any ``contains_solves`` chains.
    """

    def __init__(self, state: "RunnerState") -> None:
        self.state = state
        self.logger: logging.Logger = state.logger

    # ------------------------------------------------------------------
    # Rolling expansion
    # ------------------------------------------------------------------

    def create_rolling_solves(
        self,
        solve: str,
        full_active_time_list: dict,
        jump: float,
        horizon: float,
        start: list | None = None,
        duration: float = -1,
    ) -> tuple[list[str], dict, dict]:
        """Expand *solve* into an overlapping sequence of "rolls".

        Each roll is a sub-solve with name ``solve + "_roll_" + str(roll_counter)``
        whose active-time list is a slice of *full_active_time_list*
        bounded by the roll's start and horizon, and whose realized-time
        list is a slice from the start to the next jump boundary.  The
        ``state.solve.roll_counter[solve]`` counter is incremented per
        roll produced.

        Args:
            solve: Parent solve name (without roll suffix).
            full_active_time_list: ``{period: [ActiveTimeEntry, ...]}``
                covering the entire solve's domain.
            jump: Per-roll commitment horizon (hours).
            horizon: Per-roll optimisation horizon (hours).
            start: Optional ``[period, timestep]`` to start from.  When
                ``None``, the first entry of *full_active_time_list*
                wins.
            duration: Total duration (hours) the rolling sequence
                should cover.  ``-1`` means "until the active time
                list is exhausted".

        Returns:
            ``(roll_solve_names, active_time_lists, realized_time_lists)``.

        Raises:
            FlexToolConfigError: Start point not found in active time.
        """
        active_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        solves: list[str] = []
        starts: list[list] = []
        jumps: list[list] = []
        horizons: list[list] = []
        duration_counter: float = 0
        horizon_counter: float = 0
        jump_counter: float = 0
        started: bool = False
        ended: bool = False
        last_index: list = [None, None]

        # Single-pass walk: record ``[period, index]`` pairs at each
        # jump / horizon / duration boundary.
        for period, active_time in list(full_active_time_list.items()):
            for i, step in enumerate(active_time):
                if not ended:
                    if started:
                        if duration_counter >= float(duration) and duration != -1:
                            jumps.append(last_index)
                            horizons.append(last_index)
                            ended = True
                            break
                        if jump_counter >= float(jump):
                            jumps.append(last_index)
                            starts.append([period, i])
                            jump_counter -= float(jump)
                        if horizon_counter >= float(horizon):
                            horizons.append(last_index)
                            horizon_counter -= float(jump)
                        horizon_counter += float(step.duration)
                        jump_counter += float(step.duration)
                        duration_counter += float(step.duration)
                        last_index = [period, i]
                    else:
                        if start is None or (start == [period, step.timestep]):
                            starts.append([period, i])
                            started = True
                            horizon_counter += float(step.duration)
                            jump_counter += float(step.duration)
                            duration_counter += float(step.duration)
                            last_index = [period, i]
        if not started:
            message = "Start point not found"
            self.logger.error(message)
            raise FlexToolConfigError(message)

        # If a roll opened but didn't end (no jump/horizon boundary
        # crossed before the active time exhausted), close it at the
        # last index.
        diff = len(starts) - len(horizons)
        for _ in range(0, diff):
            horizons.append(last_index)
        diff = len(starts) - len(jumps)
        for _ in range(0, diff):
            jumps.append(last_index)

        # Pre-compute ordered period list for index-based range extraction.
        period_order = list(full_active_time_list.keys())
        period_pos = {p: i for i, p in enumerate(period_order)}

        # Slice the full active-time list per-roll using the recorded
        # boundary indexes.
        for index, roll_start in enumerate(starts):
            solve_name = solve + "_roll_" + str(self.state.solve.roll_counter[solve])
            self.state.solve.roll_counter[solve] += 1
            solves.append(solve_name)
            active = self._extract_time_range(
                full_active_time_list,
                period_order,
                period_pos,
                roll_start,
                horizons[index],
            )
            realized = self._extract_time_range(
                full_active_time_list,
                period_order,
                period_pos,
                roll_start,
                jumps[index],
            )
            active_time_lists[solve_name] = active
            realized_time_lists[solve_name] = realized
        return solves, active_time_lists, realized_time_lists

    @staticmethod
    def _extract_time_range(
        full_active_time_list: dict,
        period_order: list[str],
        period_pos: dict[str, int],
        range_start: list,
        range_end: list,
    ) -> dict:
        """Slice *full_active_time_list* between *range_start* and *range_end*.

        Both endpoints are inclusive ``[period, index_within_period]``
        pairs.  The four boundary cases are handled explicitly:

        1. **Single-period range** (``range_start[0] == range_end[0]``):
           one slice ``[start[1] : end[1] + 1]``.
        2. **Start period** of a multi-period range: from ``start[1]``
           to end-of-period.
        3. **End period** of a multi-period range: from beginning to
           ``end[1] + 1``.
        4. **Middle period(s)**: full period included.

        Args:
            full_active_time_list: ``{period: [ActiveTimeEntry, ...]}``.
            period_order: ordered list of period names from
                *full_active_time_list*.
            period_pos: ``period_name -> position_in_period_order``.
            range_start: ``[period, index]`` start (inclusive).
            range_end: ``[period, index]`` end (inclusive).

        Returns:
            New dict with the sliced sub-time-list.
        """
        result: dict = {}
        start_pos = period_pos[range_start[0]]
        end_pos = period_pos[range_end[0]]
        for pos in range(start_pos, end_pos + 1):
            period = period_order[pos]
            if pos == start_pos and pos == end_pos:
                result[period] = full_active_time_list[period][
                    range_start[1] : range_end[1] + 1
                ]
            elif pos == start_pos:
                result[period] = full_active_time_list[period][range_start[1] :]
            elif pos == end_pos:
                result[period] = full_active_time_list[period][0 : range_end[1] + 1]
            else:
                result[period] = full_active_time_list[period]
        return result

    @staticmethod
    def _filter_time_list_by_periods(
        full_time_list: dict,
        period_dict: dict,
        solve_name: str,
    ) -> dict:
        """Filter *full_time_list* to periods listed in *period_dict[solve_name]*.

        Args:
            full_time_list: ``{period: [ActiveTimeEntry, ...]}``.
            period_dict: ``{solve: [(p_from, p_in), ...]}``.
            solve_name: Lookup key in *period_dict*.

        Returns:
            Sub-dict of *full_time_list* containing only periods present
            in ``period_dict[solve_name]``.
        """
        filtered: dict = dict()
        for period_tuple in period_dict.get(solve_name, []):
            period = period_tuple[1]  # period_included
            if period in full_time_list:
                filtered[period] = full_time_list[period]
        return filtered

    @staticmethod
    def _get_periods_from_parent_time_list(parent_time_list: dict) -> set[str]:
        """Return the set of periods in *parent_time_list*."""
        return set(parent_time_list.keys())

    @staticmethod
    def _filter_time_list_by_parent_scope(
        child_time_list: dict,
        parent_periods: set[str],
    ) -> dict:
        """Restrict *child_time_list* to *parent_periods*."""
        filtered: dict = dict()
        for period, timesteps in child_time_list.items():
            if period in parent_periods:
                filtered[period] = timesteps
        return filtered

    # ------------------------------------------------------------------
    # Rolling-solve dispatcher
    # ------------------------------------------------------------------

    def _process_rolling_solve(
        self,
        solve: str,
        complete_solve_name: str,
        full_active_time_list: dict,
        parent_info: ParentSolveInfo,
        start: list | None,
        duration: float,
    ) -> SolveResult:
        """Expand a rolling-window solve into per-roll sub-solves.

        Recurses into ``contains_solves`` once per roll, threading the
        roll's first timestep + the parent's jump as the child's
        ``start`` / ``duration``.
        """
        # Lazy import: keep the cross-module dependency one-way at
        # import time; both modules are needed only when a parent's roll
        # constrains a child.
        from flextool.engine_polars._stochastic import StochasticSolver

        solves: list[str] = []
        complete_solves: dict = dict()
        active_time_lists: dict = dict()
        fix_storage_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        parent_roll_lists: dict = dict()

        rolling_times = self.state.solve.rolling_times[solve]  # [jump, horizon, duration]
        if duration == -1:
            duration = float(rolling_times[2])

        # Find start timestep if constrained by parent.
        period_start_timestep = start
        if start is not None:
            start_timestep = StochasticSolver(self.state).find_next_timestep(
                full_active_time_list, start, parent_info.solve, solve
            )
            period_start_timestep = [start[0], start_timestep]

        # Create rolling solves.
        roll_solves, roll_active_time_lists, roll_realized_time_lists = (
            self.create_rolling_solves(
                solve,
                full_active_time_list,
                float(rolling_times[0]),
                float(rolling_times[1]),
                period_start_timestep,
                duration,
            )
        )

        # Track metadata for each roll.
        for roll_name in roll_solves:
            complete_solves[roll_name] = complete_solve_name
            parent_roll_lists[roll_name] = parent_info.roll

        active_time_lists.update(roll_active_time_lists)

        # For rolling solves, fix_storage and realized are the same
        # (the "jump" portion).
        fix_storage_time_lists.update(roll_realized_time_lists)
        realized_time_lists.update(roll_realized_time_lists)

        # Mark first solve of this complete solve (for state start
        # constraints).  When this rolling solve is itself nested under
        # another (parent_info.roll != None), only mark it as a
        # first-of-complete if the parent roll was already in the list.
        if parent_info.roll is not None:
            if parent_info.roll in self.state.solve.first_of_complete_solve:
                self.state.solve.first_of_complete_solve.append(roll_solves[0])
        else:
            self.state.solve.first_of_complete_solve.append(roll_solves[0])

        self.state.solve.last_of_solve.append(roll_solves[-1])

        # Process contained solves.
        if solve in self.state.solve.contains_solves:
            contain_solves = self.state.solve.contains_solves[solve]
            if len(contain_solves) > 1:
                message = "More than one solve in a rolling solve, not managed"
                logging.error(message)
                raise FlexToolConfigError(message)

            contains_solve = contain_solves[0]

            for index, roll_name in enumerate(roll_solves):
                solves.append(roll_name)

                # The first roll passes start=None (use first active-time
                # entry); subsequent rolls explicitly pass the roll's
                # first timestep as the child's start point.
                if index != 0:
                    first_period = list(roll_active_time_lists[roll_name].keys())[0]
                    first_timestep = roll_active_time_lists[roll_name][first_period][0].timestep
                    child_start = [first_period, first_timestep]
                else:
                    child_start = None

                # Child duration equals parent's jump (the realized portion).
                child_duration = float(rolling_times[0])

                # Get parent's realized periods for child to use as scope.
                parent_realized_periods = set(roll_realized_time_lists[roll_name].keys())

                # Recursively process child solve.
                child_parent_info = ParentSolveInfo(solve=solve, roll=roll_name)
                child = self.define_solve_recursive(
                    contains_solve,
                    child_parent_info,
                    parent_realized_periods,
                    child_start,
                    child_duration,
                )

                solves += child.solves
                complete_solves.update(child.complete_solves)
                parent_roll_lists.update(child.parent_roll_lists)
                active_time_lists.update(child.active_time_lists)
                fix_storage_time_lists.update(child.fix_storage_time_lists)
                realized_time_lists.update(child.realized_time_lists)
        else:
            solves += roll_solves

        return SolveResult(
            solves=solves,
            complete_solves=complete_solves,
            active_time_lists=active_time_lists,
            fix_storage_time_lists=fix_storage_time_lists,
            realized_time_lists=realized_time_lists,
            parent_roll_lists=parent_roll_lists,
        )

    # ------------------------------------------------------------------
    # Single-solve dispatcher
    # ------------------------------------------------------------------

    def _process_single_solve(
        self,
        solve: str,
        full_active_time_list: dict,
        parent_info: ParentSolveInfo,
        complete_solve_name: str | None = None,
    ) -> SolveResult:
        """Process a single-solve (non-rolling) solve.

        When *solve* has children (``contains_solves[solve]``) recurse
        into each child with parent scope = union of fix_storage,
        realized, and realized_invest periods of *solve*.  When the
        scope is empty (because *solve* was renamed via the
        single-matching-period path and ``duplicate_solve`` did not
        carbon-copy the period attributes), fall back to *solve*'s own
        active periods.

        Args:
            solve: Active solve name (may be a renamed copy).
            full_active_time_list: ``{period: [ActiveTimeEntry, ...]}``.
            parent_info: ParentSolveInfo from the caller.
            complete_solve_name: Original (un-renamed) solve name.  When
                ``None`` defaults to *solve*.  See module docstring for
                why renamed solves keep their original complete-solve
                pointer.
        """
        if complete_solve_name is None:
            complete_solve_name = solve

        solves: list[str] = [solve]
        complete_solves: dict = dict()
        active_time_lists: dict = dict()
        invest_time_lists: dict = dict()
        fix_storage_time_lists: dict = dict()
        realized_time_lists: dict = dict()
        parent_roll_lists: dict = dict()

        complete_solves[solve] = complete_solve_name
        parent_roll_lists[solve] = parent_info.roll
        active_time_lists[solve] = full_active_time_list

        # Period-shaped subsets of the full active time list.
        invest_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.invest_periods, solve
        )
        fix_storage_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.fix_storage_periods, solve
        )
        realized_time_lists[solve] = self._filter_time_list_by_periods(
            full_active_time_list, self.state.solve.realized_periods, solve
        )

        self.state.solve.first_of_complete_solve.append(solve)
        self.state.solve.last_of_solve.append(solve)

        # Process contained solves.
        if solve in self.state.solve.contains_solves:
            contain_solves = self.state.solve.contains_solves[solve]

            # Parent scope: union of fix_storage, realized, and
            # realized_invest periods.  Invest solves only have
            # invest/realized_invest periods (not fix_storage/realized),
            # so realized_invest is needed to provide scope for their
            # children.
            realized_invest_time_list = self._filter_time_list_by_periods(
                full_active_time_list, self.state.solve.realized_invest_periods, solve
            )
            parent_scope_periods = (
                set(fix_storage_time_lists[solve].keys())
                | set(realized_time_lists[solve].keys())
                | set(realized_invest_time_list.keys())
            )

            # Fallback: when a solve was renamed (e.g.
            # ``storage_fullYear_6h_p2020``), its period attributes
            # aren't copied by ``duplicate_solve``, leaving scope empty.
            # Use the solve's own active periods as scope in that case.
            if not parent_scope_periods:
                parent_scope_periods = set(full_active_time_list.keys())

            for contain_solve in contain_solves:
                child_parent_info = ParentSolveInfo(solve=solve, roll=solve)
                child = self.define_solve_recursive(
                    contain_solve, child_parent_info, parent_scope_periods, None, -1
                )

                solves += child.solves
                complete_solves.update(child.complete_solves)
                parent_roll_lists.update(child.parent_roll_lists)
                active_time_lists.update(child.active_time_lists)
                fix_storage_time_lists.update(child.fix_storage_time_lists)
                realized_time_lists.update(child.realized_time_lists)

        return SolveResult(
            solves=solves,
            complete_solves=complete_solves,
            active_time_lists=active_time_lists,
            fix_storage_time_lists=fix_storage_time_lists,
            realized_time_lists=realized_time_lists,
            parent_roll_lists=parent_roll_lists,
        )

    # ------------------------------------------------------------------
    # Recursive entry
    # ------------------------------------------------------------------

    def define_solve_recursive(
        self,
        solve: str,
        parent_info: ParentSolveInfo,
        parent_scope_periods: set[str] | None = None,
        start: list | None = None,
        duration: float = -1,
    ) -> SolveResult:
        """Walk the solve tree rooted at *solve*.

        For each invocation:

        1. Append *solve* to ``state.solve.real_solves`` (deduplicated).
        2. If *solve* has a parent and exactly one period overlaps,
           rename it to ``solve + "_" + period`` via
           :meth:`SolveConfig.duplicate_solve` and re-shape
           ``timesets_used_by_solves`` to keep only entries for that
           period.
        3. Compute the full active-time list via :func:`get_active_time`.
        4. If *parent_scope_periods* is non-empty, restrict the active-
           time list to those periods.
        5. Dispatch to :meth:`_process_rolling_solve` or
           :meth:`_process_single_solve`.

        Args:
            solve: Solve name to process.
            parent_info: ParentSolveInfo (top-level call uses
                ``ParentSolveInfo(None, None)``).
            parent_scope_periods: Periods the parent committed to.
                ``None`` for top-level solves.
            start: ``[period, timestep]`` start (rolling-window child
                only).
            duration: Hours of duration to cover (rolling-window child
                only).
        """
        new_name = solve
        if new_name not in self.state.solve.real_solves:
            self.state.solve.real_solves.append(new_name)

        # Single-matching-period rename path.  When the child's period
        # set intersects the parent's by exactly one period, rename
        # ``solve`` → ``solve + "_" + period`` and carbon-copy the
        # parent dicts.  Multiple-period overlap keeps the original
        # name (single-solve-with-multi-period parent case).
        if parent_info.solve:
            current_solve_periods = (
                {t[0] for t in self.state.solve.invest_periods[solve]}
                | {t[0] for t in self.state.solve.fix_storage_periods[solve]}
                | {t[0] for t in self.state.solve.realized_periods[solve]}
            )
            parent_period = (
                {t[0] for t in self.state.solve.invest_periods[parent_info.solve]}
                | {t[0] for t in self.state.solve.fix_storage_periods[parent_info.solve]}
                | {t[0] for t in self.state.solve.realized_periods[parent_info.solve]}
            )
            matching_periods = list(current_solve_periods & parent_period)
            if len(matching_periods) == 1:
                current_solve_period = matching_periods[0]
                new_name = solve + "_" + str(current_solve_period)
                self.state.solve.duplicate_solve(
                    solve, new_name, update_model_solves=False
                )
                self.state.solve.solve_period_years_represented[new_name] = (
                    self.state.solve.solve_period_years_represented[solve]
                )

                new_period_timeset_list = [
                    pt
                    for pt in self.state.solve.timesets_used_by_solves.get(solve, [])
                    if pt[0] == current_solve_period
                ]
                if new_name not in self.state.solve.timesets_used_by_solves:
                    self.state.solve.timesets_used_by_solves[new_name] = (
                        new_period_timeset_list
                    )
                else:
                    existing = self.state.solve.timesets_used_by_solves[new_name]
                    for item in new_period_timeset_list:
                        if item not in existing:
                            existing.append(item)
            # When multiple periods match (single-solve parent with
            # multiple periods), keep the original solve name and all
            # its timesets — the active time will be filtered by
            # parent scope later.

        # Full active time list for this solve (every timestep it
        # could potentially use).
        full_active_time_list_own = get_active_time(
            new_name,
            self.state.solve.timesets_used_by_solves,
            self.state.timeline.timeset_durations,
            self.state.timeline.timelines,
            self.state.timeline.timesets__timeline,
        )

        # Restrict to parent scope when applicable.
        if not parent_scope_periods:
            full_active_time_list = full_active_time_list_own
        else:
            full_active_time_list = self._filter_time_list_by_parent_scope(
                full_active_time_list_own, parent_scope_periods
            )

        # Determine solve mode.
        solve_mode = self.state.solve.solve_modes.get(new_name, "single_solve")
        if solve_mode == "rolling_window":
            complete_solve_name = solve
            return self._process_rolling_solve(
                new_name,
                complete_solve_name,
                full_active_time_list,
                parent_info,
                start,
                duration,
            )
        else:
            # Single-solve: pass the *original* name as
            # complete_solve_name so renamed solves (e.g.
            # ``storage_fullYear_6h_p2020``) keep mapping back to the
            # complete-solve name in ``state.solve.real_solves`` /
            # ``solve_period_history``.
            return self._process_single_solve(
                new_name,
                full_active_time_list,
                parent_info,
                complete_solve_name=solve,
            )


__all__ = [
    "ParentSolveInfo",
    "RecursiveSolveBuilder",
]
