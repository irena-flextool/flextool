"""Stochastic branching of time periods.

Architecture notes
------------------

* :class:`StochasticSolver` adds branched versions of periods to the
  per-solve active-time lists when ``state.solve.stochastic_branches``
  declares branches for a solve.  Branches receive LP variables
  (active time) but are NOT committed (``realized_time_lists`` /
  ``fix_storage_time_lists`` are filtered to the realized branch only).

* :func:`write_timeline_matching_map` returns an in-memory dict
  instead of writing CSV; downstream consumers read the dict directly.

* **R-O6 invariant** (``audit/a6_b_dim_alternative.md §1-3``): branches
  do NOT enter ``invest_periods``.  ``v_invest`` stays realized-only.
  Recourse investment is a future capability requiring a separate
  b-dim refactor; this module must not introduce per-branch invest
  variables.

* Mutates ``state.timeline.stochastic_timesteps`` (a slight code smell
  — it would be cleaner to return a fresh dict, but downstream readers
  consume the mutated state).

Behavioural quirks
------------------

* **Zero-weight branch**: excluded from ``new_active_time_list`` but
  only excluded from ``solve_branch__time_branch_lists`` if both
  ``branch != period`` AND the realized flag is not ``"yes"``.  Subtle
  three-way condition.

* **Continuation-after-branching**: when a roll's jump exceeds a
  period length, branches continue into subsequent periods using the
  same ``period + "_" + branch`` naming.  The naming is "sticky"
  across periods.

* **Single realized branch per period**: each period must have
  exactly one ``realized: yes`` row (or zero before any branching has
  occurred).  Found 0 or > 1 → raise ``FlexToolConfigError``.
"""
from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from typing import Any, TYPE_CHECKING

from flextool.engine_polars._solve_state import FlexToolConfigError
from flextool.engine_polars._timeline import make_step_jump

if TYPE_CHECKING:
    from flextool.engine_polars._solve_state import RunnerState


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------


def _get_timeset_for_period(
    period_timesets: list[tuple[str, str]],
    real_period: str,
) -> str | None:
    """Find the timeset corresponding to *real_period* in *period_timesets*."""
    for period_timeset in period_timesets:
        if period_timeset[0] == real_period:
            return period_timeset[1]
    return None


def connect_two_timelines(
    state: "RunnerState",
    period: str,
    first_solve: str,
    second_solve: str,
    period__branch: list[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute cumulative durations from start for two solves' timelines.

    Returns two ``{timestep: cumulative_hours}`` dicts so callers can
    align timesteps between the two solves.

    Args:
        state: Runner state (reads ``solve.timesets_used_by_solves`` and
            ``timeline.{timesets__timeline, timelines}``).
        period: Period (or branch) name to align on.
        first_solve / second_solve: Solve names.
        period__branch: List of ``(real_period, branched_period)``
            tuples.  Used to map *period* back to its real period for
            timeset lookup.

    Raises:
        ValueError: When the timeset for *real_period* can't be located
            in either solve's ``timesets_used_by_solves``.
    """
    first_period_timesets = state.solve.timesets_used_by_solves[first_solve]
    second_period_timesets = state.solve.timesets_used_by_solves[second_solve]

    real_period: str | None = None
    for row in period__branch:
        if row[1] == period:
            real_period = row[0]

    first_timeset = _get_timeset_for_period(first_period_timesets, real_period)
    second_timeset = _get_timeset_for_period(second_period_timesets, real_period)

    if first_timeset is None:
        raise ValueError(
            f"Could not find first_timeset for real_period={real_period} "
            f"in first_period_timesets={first_period_timesets}"
        )
    if second_timeset is None:
        raise ValueError(
            f"Could not find second_timeset for real_period={real_period} "
            f"in second_period_timesets={second_period_timesets}"
        )

    first_timeline = state.timeline.timesets__timeline[first_timeset]
    second_timeline = state.timeline.timesets__timeline[second_timeset]

    first_timeline_duration_from_start: dict[str, float] = dict()
    second_timeline_duration_from_start: dict[str, float] = dict()
    counter: float = 0
    for timestep in state.timeline.timelines[first_timeline]:
        first_timeline_duration_from_start[timestep[0]] = counter
        counter += float(timestep[1])
    counter = 0
    for timestep in state.timeline.timelines[second_timeline]:
        second_timeline_duration_from_start[timestep[0]] = counter
        counter += float(timestep[1])

    return first_timeline_duration_from_start, second_timeline_duration_from_start


def find_previous_timestep(
    state: "RunnerState",
    from_active_time_list: dict[str, list[tuple[str, ...]]],
    period_timestamp: tuple[str, str],
    this_solve: str,
    from_solve: str,
    period__branch: list[tuple[str, str]],
) -> str:
    """Find the previous timestep in *from_solve* matching *period_timestamp* in *this_solve*."""
    this_timeline_duration_from_start, from_timeline_duration_from_start = (
        connect_two_timelines(
            state, period_timestamp[0], this_solve, from_solve, period__branch
        )
    )

    real_period: str | None = None
    for row in period__branch:
        if row[1] == period_timestamp[0]:
            real_period = row[0]
    from_start = this_timeline_duration_from_start[period_timestamp[1]]
    last_timestep = from_active_time_list[real_period][0].timestep
    previous_timestep = from_active_time_list[real_period][-1].timestep
    for timestep in from_active_time_list[real_period]:
        if from_timeline_duration_from_start[timestep.timestep] > from_start:
            previous_timestep = last_timestep
            break
        last_timestep = timestep.timestep
    return previous_timestep


def find_next_timestep(
    state: "RunnerState",
    from_active_time_list: dict[str, list[tuple[str, ...]]],
    period_timestamp: tuple[str, str],
    this_solve: str,
    from_solve: str,
) -> str:
    """Find the next timestep in *from_solve* matching *period_timestamp* in *this_solve*.

    The ``period__branch`` arg is implicitly the identity mapping
    ``[(period, period)]`` here — used when the lookup happens before
    any branches have been generated.
    """
    this_timeline_duration_from_start, from_timeline_duration_from_start = (
        connect_two_timelines(
            state,
            period_timestamp[0],
            this_solve,
            from_solve,
            [(period_timestamp[0], period_timestamp[0])],
        )
    )

    from_start = this_timeline_duration_from_start[period_timestamp[1]]
    next_timestep = from_active_time_list[period_timestamp[0]][-1].timestep
    for timestep in from_active_time_list[period_timestamp[0]]:
        if from_timeline_duration_from_start[timestep.timestep] >= from_start:
            next_timestep = timestep.timestep
            break
    return next_timestep


def write_timeline_matching_map(
    state: "RunnerState",
    upper_active_time_list: dict[str, list[tuple[str, ...]]],
    lower_active_time_list: dict[str, list[tuple[str, ...]]],
    upper_solve: str,
    lower_solve: str,
    period__branch: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Build the timeline matching map for nested storage fixing.

    Maps each lower-solve ``(period, timestep)`` pair to the
    corresponding upper-solve timestep using binary search on
    cumulative timeline durations.

    Returns:
        ``{(period, timestep): upper_timestep, ...}`` dict.

    Note: the AMPL/.mod reference path also writes
    ``solve_data/timeline_matching_map.csv`` to disk.  The native
    engine keeps the result in-memory; the CSV write is a separate
    concern that lives in the orchestrator (Γ.8.D) when file output
    is requested for compatibility with downstream tools.
    """
    # Pre-compute period -> real_period mapping.
    period_to_real: dict[str, str] = {row[1]: row[0] for row in period__branch}

    upper_period_timesets = state.solve.timesets_used_by_solves[upper_solve]
    upper_periods: set[str] = {pt[0] for pt in upper_period_timesets}

    matching_map: dict[tuple[str, str], str] = dict()
    for period, lower_active_time in lower_active_time_list.items():
        real_period = period_to_real.get(period)

        # Skip periods that don't exist in the upper solve's timesets.
        if real_period not in upper_periods:
            continue

        # Call connect_two_timelines once per period, not per timestep.
        this_timeline_duration, from_timeline_duration = connect_two_timelines(
            state, period, lower_solve, upper_solve, period__branch
        )

        upper_timesteps = upper_active_time_list[real_period]

        from_durations: list[float] = []
        from_timestep_names: list[str] = []
        for ts in upper_timesteps:
            from_durations.append(from_timeline_duration[ts.timestep])
            from_timestep_names.append(ts.timestep)

        default_timestep = upper_timesteps[-1].timestep

        for timestep in lower_active_time:
            period_timestep = (period, timestep.timestep)
            from_start = this_timeline_duration[timestep.timestep]

            # Binary search: find first index where duration > from_start.
            idx = bisect.bisect_right(from_durations, from_start)

            if idx == 0:
                previous_timestep = upper_timesteps[0].timestep
            elif idx >= len(from_durations):
                previous_timestep = default_timestep
            else:
                previous_timestep = from_timestep_names[idx - 1]

            matching_map[period_timestep] = previous_timestep

    return matching_map


# ---------------------------------------------------------------------------
# StochasticSolver
# ---------------------------------------------------------------------------


class StochasticSolver:
    """Apply stochastic branching to per-solve active-time lists.

    See :meth:`create_stochastic_periods` for the entry point.  The
    standalone helpers above (``connect_two_timelines``,
    ``find_previous_timestep``, ``find_next_timestep``,
    ``write_timeline_matching_map``) are kept as module-level functions
    so callers don't need a ``StochasticSolver`` instance for read-only
    queries.
    """

    def __init__(self, state: "RunnerState") -> None:
        self.state = state
        self.logger: logging.Logger = state.logger

    # ------------------------------------------------------------------
    # Convenience instance wrappers (kept for backwards compatibility).
    # The underlying logic lives in module-level helpers above.
    # ------------------------------------------------------------------

    def connect_two_timelines(
        self,
        period: str,
        first_solve: str,
        second_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        return connect_two_timelines(
            self.state, period, first_solve, second_solve, period__branch
        )

    def find_previous_timestep(
        self,
        from_active_time_list: dict[str, list[tuple[str, ...]]],
        period_timestamp: tuple[str, str],
        this_solve: str,
        from_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> str:
        return find_previous_timestep(
            self.state,
            from_active_time_list,
            period_timestamp,
            this_solve,
            from_solve,
            period__branch,
        )

    def find_next_timestep(
        self,
        from_active_time_list: dict[str, list[tuple[str, ...]]],
        period_timestamp: tuple[str, str],
        this_solve: str,
        from_solve: str,
    ) -> str:
        return find_next_timestep(
            self.state,
            from_active_time_list,
            period_timestamp,
            this_solve,
            from_solve,
        )

    def write_timeline_matching_map(
        self,
        upper_active_time_list: dict[str, list[tuple[str, ...]]],
        lower_active_time_list: dict[str, list[tuple[str, ...]]],
        upper_solve: str,
        lower_solve: str,
        period__branch: list[tuple[str, str]],
    ) -> dict[tuple[str, str], str]:
        return write_timeline_matching_map(
            self.state,
            upper_active_time_list,
            lower_active_time_list,
            upper_solve,
            lower_solve,
            period__branch,
        )

    # ------------------------------------------------------------------
    # create_stochastic_periods — main entry point
    # ------------------------------------------------------------------

    def create_stochastic_periods(
        self,
        stochastic_branches: dict[str, Any],
        solves: list[str],
        complete_solves: dict[str, str],
        active_time_lists: dict[str, dict],
        fix_storage_time_lists: dict[str, dict],
        realized_time_lists: dict[str, dict],
    ) -> tuple[
        defaultdict[str, list],
        defaultdict[str, list],
        dict[str, dict],
        dict[str, Any],
        dict[str, dict],
        dict[str, dict],
        defaultdict[str, Any],
    ]:
        """Apply stochastic branching to per-solve active-time lists.

        Branches are added to ``active_time_lists`` for optimization
        but NOT to ``realized_time_lists`` / ``fix_storage_time_lists``;
        branches are future scenarios that aren't committed.

        IMPORTANT: branches do NOT enter ``invest_periods`` either.
        Per ``audit/a6_b_dim_alternative.md`` (R-O6), ``v_invest`` is
        realized-only — recourse investment is a future capability.

        Args:
            stochastic_branches: ``{solve: [(period, branch, start_step,
                realized_yes_no, weight), ...]}``.  Read from the DB.
            solves: Flat list of all solve names (output of
                :class:`RecursiveSolveBuilder`).
            complete_solves: ``{solve: complete_solve_name}``.
            active_time_lists / fix_storage_time_lists / realized_time_lists:
                Per-solve time lists from the recursive builder.

        Returns:
            ``(period__branch_lists, solve_branch__time_branch_lists,
            active_time_lists, jump_lists, fix_storage_time_lists,
            realized_time_lists, branch_start_time_lists)``.

        Raises:
            FlexToolConfigError: When a solve's start time isn't in any
                ``stochastic_branches`` row, or when more than one
                ``realized: yes`` row matches a period.
        """
        period__branch_lists: defaultdict[str, list] = defaultdict(list)
        solve_branch__time_branch_lists: defaultdict[str, list] = defaultdict(list)
        jump_lists: dict[str, Any] = dict()
        branch_start_time_lists: defaultdict[str, Any] = defaultdict()

        for solve in solves:
            new_realized_time_list: dict = dict()
            new_fix_storage_time_list: dict = dict()
            new_active_time_list: dict = dict()

            info = stochastic_branches[complete_solves[solve]]
            active_time_list = active_time_lists[solve]
            realized_time_list = realized_time_lists[solve]
            fix_storage_time_list = fix_storage_time_lists[solve]

            branched = False
            branches: list[str] = []
            branch_start_time_lists[solve] = None

            # First step of the solve (used for validation below).
            first_step: tuple[str, str] | None = None
            for period, active_time in active_time_list.items():
                first_step = (period, active_time[0].timestep)
                break

            # Validate: when stochastic_branches has rows for this
            # complete-solve, at least one row's start_step must match
            # the solve's first step AND be marked realized.
            found_start = False
            for row in info:
                if first_step is not None and first_step[1] == row[2] and "yes" == row[3]:
                    found_start = True
            if found_start is False and len(info) != 0:
                message = (
                    "A realized start time of the solve cannot be found from "
                    "the stochastic_branches parameter. "
                    "Check that stochastic_branches has a realized : yes, "
                    "branch for the start of the solve "
                    "and that the possible rolling_jump matches with the "
                    "branch starts"
                )
                self.logger.error(message)
                raise FlexToolConfigError(message)

            # Walk periods — at the first branch trigger, fan out into
            # branches; subsequent periods continue the same branches
            # (continuation-after-branching at lines 329-340 of the
            # reference).
            for period, active_time in active_time_list.items():
                if not branched:
                    period__branch_lists[solve].append((period, period))
                    # Get all start times: row[0]=period, row[1]=branch,
                    # row[2]=start_step, row[3]=realized_yn, row[4]=weight.
                    start_times: defaultdict[str, list] = defaultdict(list)
                    for row in info:
                        if row[0] == period:
                            start_times[row[2]].append((row[1], row[4], row[3]))

                    # Trigger branching at the first step that appears
                    # in start_times.
                    for step in active_time:
                        if step.timestep in start_times.keys():
                            branched = True
                            branch_start_time_lists[solve] = (period, step.timestep)

                            # Add active time for base period.
                            new_active_time_list[period] = active_time

                            # Copy realized + fix_storage for base period only.
                            if period in realized_time_list:
                                new_realized_time_list[period] = realized_time_list[period]
                            if period in fix_storage_time_list:
                                new_fix_storage_time_list[period] = fix_storage_time_list[period]

                            # Create branches: each gets active time but
                            # NOT realized/fix_storage.
                            for branch__weight__real in start_times[step.timestep]:
                                branch = branch__weight__real[0]
                                branches.append(branch)
                                solve_branch = period + "_" + branch

                                # Zero-weight branch is excluded from
                                # both active time AND
                                # solve_branch__time_branch when the
                                # three-way condition holds.
                                if (
                                    float(branch__weight__real[1]) != 0.0
                                    and branch != period
                                    and branch__weight__real[2] != "yes"
                                ):
                                    new_active_time_list[solve_branch] = active_time[0:]
                                    solve_branch__time_branch_lists[solve].append(
                                        (solve_branch, branch)
                                    )

                                period__branch_lists[solve].append((period, solve_branch))

                                # Track branched timesteps for downstream
                                # disambiguation.  Mutates state.timeline.
                                for i in active_time[0:]:
                                    self.state.timeline.stochastic_timesteps[solve].append(
                                        (solve_branch, i.timestep)
                                    )
                            break
                else:
                    # Continuation-after-branching: extend each branch
                    # into the next period using the same branch-name
                    # prefix.
                    for branch in branches:
                        solve_branch = period + "_" + branch
                        period__branch_lists[solve].append((period, solve_branch))
                        solve_branch__time_branch_lists[solve].append(
                            (solve_branch, branch)
                        )
                        for i in active_time_list[period]:
                            self.state.timeline.stochastic_timesteps[solve].append(
                                (solve_branch, i.timestep)
                            )

                # Periods before any branching: copy as-is.
                if not branched:
                    new_active_time_list[period] = active_time_list[period]
                    if period in realized_time_list:
                        new_realized_time_list[period] = realized_time_list[period]
                    if period in fix_storage_time_list:
                        new_fix_storage_time_list[period] = fix_storage_time_list[period]

            # Find the realized branch for each period.  Each period
            # must have exactly one realized: yes row (or zero before
            # any branching has occurred).
            for period, active_time in active_time_list.items():
                found = 0
                # Before branching: row[0]==period, row[2]==first step,
                # row[3]=='yes'.
                for row in info:
                    if (
                        row[0] == period
                        and row[2] == active_time[0].timestep
                        and row[3] == "yes"
                    ):
                        found += 1
                        solve_branch__time_branch_lists[solve].append((period, row[1]))
                # After branching: lookup against the branch-start row.
                if found == 0 and branch_start_time_lists[solve] is not None:
                    for row in info:
                        if (
                            row[0] == branch_start_time_lists[solve][0]
                            and row[2] == branch_start_time_lists[solve][1]
                            and row[3] == "yes"
                        ):
                            found += 1
                            solve_branch__time_branch_lists[solve].append(
                                (period, row[1])
                            )
                if (branch_start_time_lists[solve] is not None and found == 0) or found > 1:
                    message = (
                        "Each period should have one and only one realized branch. "
                        "Found: " + str(found) + "\n"
                    )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)

            # Update the time lists for this solve.
            realized_time_lists[solve] = new_realized_time_list
            fix_storage_time_lists[solve] = new_fix_storage_time_list
            active_time_lists[solve] = new_active_time_list
            jump_lists[solve] = make_step_jump(
                new_active_time_list,
                period__branch_lists[solve],
                solve_branch__time_branch_lists[solve],
            )

        return (
            period__branch_lists,
            solve_branch__time_branch_lists,
            active_time_lists,
            jump_lists,
            fix_storage_time_lists,
            realized_time_lists,
            branch_start_time_lists,
        )


__all__ = [
    "StochasticSolver",
    "connect_two_timelines",
    "find_previous_timestep",
    "find_next_timestep",
    "write_timeline_matching_map",
]
