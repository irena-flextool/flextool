"""Stochastic branching of time periods.

Architecture notes
------------------

* :class:`StochasticSolver` adds branched versions of periods to the
  per-solve active-time lists when ``state.solve.stochastic_branches``
  declares branches for a solve.  Branches receive LP variables
  (active time) but are NOT committed; a fan period's original name (its
  period label) has no active time at all.

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

* **Continuation-after-branching**: every period after the branching
  period continues the fan.  Each non-realized nonzero-weight branch gets
  a synthetic ``period + "_" + branch`` copy of the period's active time
  (LP variables, never committed).  The original period name is a period
  label: no active time, no realized or fix-storage window (kept aside
  for the bookkeeping emits).  The naming is "sticky" across periods.
  ``period__branch`` mirrors the branching period's metadata rows
  (``(period, period)`` self-row plus one row per fan member,
  realized and zero-weight included).

* **Forecasts-only fan with optional realized designation**: a
  stochastic fan is FORECASTS ONLY (probability-weighted, Σ=1, no
  realized member).  "Realized" is an OPTIONAL 0-or-1 designation per
  revealing cohort, keyed off the ``realized_yn == "yes"`` column — it
  feeds only the realization region / separate realized-dispatch solve
  and is NEVER a weighted fan member.  A revealing cohort must carry
  ≥1 forecast branch (``realized: no`` with nonzero weight); 0 or 1
  ``realized: yes`` rows are both valid; 2+ realized rows → raise
  ``FlexToolConfigError``.
"""
from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._timeline import make_step_jump
from flextool.engine.preprocess.params._branch import fan_period_labels

if TYPE_CHECKING:
    from flextool.engine.input._solve_state import RunnerState


@dataclass(frozen=True)
class FanLabelTimes:
    """The period labels of one sub-solve and their pre-stochastic time lists.

    A period label is the original name of a period that opens a fan (see
    :func:`~flextool.engine.preprocess.params._branch.fan_period_labels`).  It has no dispatch, so
    :meth:`StochasticSolver.create_stochastic_periods` drops it from the active, realized and fix-storage lists; the
    lists it had before are kept here: ``active`` for the label's steps, ``realized`` and ``fix_storage`` for the
    cells the sub-solve commits inside the fan.
    """

    labels: frozenset
    active: dict
    realized: dict
    fix_storage: dict


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

    def _validate_cohort(self, info, period, step, descr, realized_suffix=""):
        """Enforce the fan invariants on one reveal cohort: at most one
        ``realized:yes`` branch and at least one nonzero-weight forecast.

        ``step`` restricts the count to that start-step (the triggering
        cohort); ``step=None`` counts every step of ``period`` (the
        single-solve per-reveal sweep).  ``descr`` / ``realized_suffix``
        carry the differing message fragments so the raised text stays
        byte-identical to the two original call sites.  Raises
        FlexToolConfigError on violation.
        """
        n_real = sum(
            1
            for r in info
            if r[0] == period
            and (step is None or r[2] == step)
            and r[3] == "yes"
        )
        n_forecast = sum(
            1
            for r in info
            if r[0] == period
            and (step is None or r[2] == step)
            and r[3] != "yes"
            and float(r[4]) != 0.0
        )
        if n_real > 1:
            message = (
                f"{descr} must have at most one realized:yes branch per "
                f"revealing cohort; found {n_real}. Check "
                f"stochastic_branches{realized_suffix}."
            )
            self.logger.error(message)
            raise FlexToolConfigError(message)
        if n_forecast < 1:
            message = (
                f"{descr} must contain at least one forecast (non-realized, "
                f"nonzero-weight) branch; found {n_forecast}. A stochastic "
                "fan is forecasts only — realized is an optional "
                "designation. Check stochastic_branches."
            )
            self.logger.error(message)
            raise FlexToolConfigError(message)

    @staticmethod
    def _fan_points(
        solves: list[str],
        branch_start_time_lists: dict[str, Any],
        complete_solves: dict[str, str],
    ) -> defaultdict[str, set]:
        """``{complete solve: {(period, step), ...}}`` -- the fan point of
        every sub-solve (solve or roll) that fans out, per complete solve."""
        triggers: defaultdict[str, set] = defaultdict(set)
        for x in solves:
            if branch_start_time_lists.get(x) is not None:
                triggers[complete_solves[x]].add(tuple(branch_start_time_lists[x]))
        return triggers

    def _check_unfanned_cohorts(
        self,
        unfanned: dict[str, tuple[str, str, bool, list]],
        solves: list[str],
        branch_start_time_lists: dict[str, Any],
        complete_solves: dict[str, str],
    ) -> None:
        """Raise when a fanned sub-solve declares a later forecast cohort
        that no sub-solve of its complete solve fans out at.

        A solve or roll fans out only once, at its first reveal; a later
        cohort ``(period, start_step)`` with forecast branches would be
        dropped silently.  A later cohort is legitimate when it is the fan
        point (``branch_start_time_lists``) of any sub-solve ``x`` with
        ``complete_solves[x] == complete_solves[s]`` — e.g. the roll that
        starts there reads it through its own fan.

        Args:
            unfanned: ``{solve: (trig_period, trig_step,
                is_rolling_sub_solve, [(period, step, forecasts), ...])}``
                — the later forecast cohorts reachable in each fanned
                sub-solve, in timeline order.
            solves: All sub-solves, in order (the first offender raises).
            branch_start_time_lists: ``{solve: (period, step) | None}``.
            complete_solves: ``{solve: complete_solve_name}``.

        Raises:
            FlexToolConfigError: For the first sub-solve with such a cohort,
                naming the solve (roll), its fan point and every cohort.
        """
        triggers = self._fan_points(solves, branch_start_time_lists, complete_solves)
        for s in solves:
            if s not in unfanned:
                continue
            trig_period, trig_step, is_rolling, candidates = unfanned[s]
            c = complete_solves[s]
            remaining = [
                (p, t, f) for (p, t, f) in candidates if (p, t) not in triggers[c]
            ]
            if not remaining:
                continue
            if is_rolling:
                who = f"Roll {s!r} of rolling solve {c!r}"
                hint = (
                    "Each roll fans out at the first reveal in its horizon, "
                    "and no roll fans out at these. Remove these rows, or make "
                    "a roll start there (rolling_solve_jump)."
                )
            else:
                who = f"Solve {s!r}"
                hint = (
                    f"The branches opened at {trig_period!r} continue through "
                    "every later period, and a forecast series given at a "
                    "later period's first time step is read without these "
                    "rows. Remove these rows, or put the later fan in a "
                    "separate solve."
                )
            items = "; ".join(
                f"period {p!r} at analysis_time {t!r} (forecast branches "
                + ", ".join(repr(b) for b in f)
                + ")"
                for (p, t, f) in remaining
            )
            message = (
                f"{who} opens its stochastic fan at period {trig_period!r} "
                f"(analysis_time {trig_step!r}). A solve or roll fans out only "
                f"once, but the stochastic_branches of solve {c!r} also "
                f"declare forecast branches later in its horizon: {items}. "
                f"FlexTool would ignore them. {hint} Check stochastic_branches."
            )
            self.logger.error(message)
            raise FlexToolConfigError(message)

    def _check_realized_switch(
        self,
        unfanned_realized: dict[str, tuple[str, str, bool, list]],
        solves: list[str],
        branch_start_time_lists: dict[str, Any],
        complete_solves: dict[str, str],
        cohort_rows: dict[str, dict[tuple[str, str], list]],
    ) -> None:
        """Raise when the ``realized: yes`` rows of a solve would switch its
        realization series (one realization series per solve).

        R1: a fanned sub-solve reaches a later *realized-only* cohort (a
        ``realized: yes`` row and no nonzero-weight forecast) that is not
        the fan point of any sub-solve of its complete solve -- whatever
        branch it names, such a row either switches the series or repeats
        it.  Checked for every sub-solve in ``solves`` order, the first
        offender raises.

        R2: the fan points of one roll sequence (the sub-solve name without
        its trailing ``_roll_<N>``) carry ``realized: yes`` rows naming
        different branches.  A non-rolling sub-solve is a sequence of one
        fan point and never raises; the per-outer-period copies of a nested
        inner solve are separate sequences.  Sequences are checked in order
        of first appearance in ``solves``.  A fan point without a
        ``realized: yes`` row is not checked.

        Args:
            unfanned_realized: ``{solve: (trig_period, trig_step,
                is_rolling_sub_solve, [(period, step, realized), ...])}``
                -- the later realized-only cohorts reachable in each fanned
                sub-solve, in timeline order.
            solves: All sub-solves, in order.
            branch_start_time_lists: ``{solve: (period, step) | None}``.
            complete_solves: ``{solve: complete_solve_name}``.
            cohort_rows: ``{complete solve: {(period, step): rows}}``.

        Raises:
            FlexToolConfigError: R1 before R2, naming the solve (roll), the
                period, the analysis time and the branch.
        """
        triggers = self._fan_points(solves, branch_start_time_lists, complete_solves)
        for s in solves:
            if s not in unfanned_realized:
                continue
            trig_period, trig_step, is_rolling, realized_later = unfanned_realized[s]
            c = complete_solves[s]
            remaining = [
                (p, t, r) for (p, t, r) in realized_later if (p, t) not in triggers[c]
            ]
            if not remaining:
                continue
            who = (
                f"Roll {s!r} of rolling solve {c!r}" if is_rolling else f"Solve {s!r}"
            )
            items = "; ".join(
                f"period {p!r} at analysis_time {t!r} ("
                + ("realized branch " if len(r) == 1 else "realized branches ")
                + ", ".join(repr(b) for b in r)
                + ")"
                for (p, t, r) in remaining
            )
            n_rows = sum(len(r) for (_p, _t, r) in remaining)
            if n_rows == 1:
                marks, row, rows, switch = (
                    "a realized branch", "This row", "this row",
                    "switches the realization dispatch to its branch from that period "
                    "on, and it is not allowed even when it names",
                )
            else:
                marks, row, rows, switch = (
                    "realized branches", "These rows", "these rows",
                    "switch the realization dispatch to their branch from that period "
                    "on, and they are not allowed even when they name",
                )
            message = (
                f"{who} opens its stochastic fan at period {trig_period!r} "
                f"(analysis_time {trig_step!r}). The stochastic_branches of solve "
                f"{c!r} also mark {marks} later in its horizon, where no solve or "
                f"roll fans out: {items}. {row} {switch} the branch already "
                "realized. A solve has one realization series: give the realized: "
                "yes row only where a solve or roll fans out, always with the same "
                "branch, and change that branch's series to give the realization "
                f"new data. Remove {rows}. Check stochastic_branches."
            )
            self.logger.error(message)
            raise FlexToolConfigError(message)

        # Roll sequence -> (complete solve, [(branch, period, step), ...]).
        sequences: dict[str, tuple[str, list[tuple[str, str, str]]]] = dict()
        seen: dict[str, set] = dict()
        for x in solves:
            fan = branch_start_time_lists.get(x)
            if fan is None:
                continue
            key = re.sub(r"_roll_\d+$", "", x)
            point = tuple(fan)
            if point in seen.setdefault(key, set()):
                continue
            seen[key].add(point)
            c = complete_solves[x]
            marks = sequences.setdefault(key, (c, []))[1]
            rows = cohort_rows.get(c, {}).get(point, [])
            for b in sorted({r[1] for r in rows if r[3] == "yes"}):
                marks.append((b, point[0], point[1]))
        for c, marks in sequences.values():
            if len({b for b, _p, _t in marks}) < 2:
                continue
            items = "; ".join(
                f"{b!r} at period {p!r} (analysis_time {t!r})" for b, p, t in marks
            )
            message = (
                f"The stochastic_branches of solve {c!r} mark different realized "
                f"branches where its rolls fan out: {items}. A solve has one "
                "realization series: name the same branch in every realized: yes "
                "row, and give that branch the data of each roll at the roll's "
                "analysis_time. Check stochastic_branches."
            )
            self.logger.error(message)
            raise FlexToolConfigError(message)

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
        dict[str, "FanLabelTimes"],
    ]:
        """Apply stochastic branching to per-solve active-time lists.

        At the first reveal reachable in a solve's active time the solve
        fans out: every nonzero-weight forecast branch ``b`` (``realized:
        no``) gets one synthetic ``period + "_" + b`` period per period of
        the fan (the reveal period and every later period of the solve),
        carrying a full copy of that period's active time.  The original
        name of each fan period is its *period label*
        (:func:`~flextool.engine.preprocess.params._branch.fan_period_labels`):
        it has no dispatch, so it is dropped from the active, realized and
        fix-storage lists; its pre-stochastic lists are returned in
        :class:`FanLabelTimes` (the committed cells inside the fan and the
        label's steps, for the bookkeeping emits).  Forecast branches are
        never realized or storage-fixed.  Periods before the reveal keep
        their lists unchanged.

        ``period__branch`` holds, per fan period, the label's self-row and
        one row per fan member (zero-weight and ``realized: yes`` members
        included, as metadata); ``solve_branch__time_branch`` gains a row
        per forecast branch period, plus the designated realized branch's
        row on each label when a ``realized: yes`` row exists (projected
        onto a realization solve), and, in a forecasts-only fan, the
        reference forecast (lexically smallest forecast token) on each
        pre-reveal period so its branch-keyed data binds (R-NAME).

        IMPORTANT: branches do NOT enter ``invest_periods``; the recourse
        investment domain is resolved downstream (a period label has no
        investment column under recourse).

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
            realized_time_lists, branch_start_time_lists,
            fan_label_time_lists)``.

        Raises:
            FlexToolConfigError: When the reveal that triggers a solve does
                not start at its branching period's first step (not a
                period-boundary reveal), when that triggering cohort carries
                more than one ``realized: yes`` row (0 or 1 is valid — the
                fan is forecasts only, realized is an optional designation),
                when that cohort carries no forecast branch (``realized: no``
                with nonzero weight), or when a single (non-rolling) solve
                declares ``stochastic_branches`` none of which reveal at a
                reachable step of its active time (silently-ignored
                branches); and, after every sub-solve is processed, when a
                fanned sub-solve declares forecast branches at a later
                reachable cohort (a later period, or a later step of the fan
                period) that is not the fan point of any sub-solve of the
                same complete solve (a solve or roll fans out only once);
                then when a fanned sub-solve marks a ``realized: yes`` row
                at a later reachable cohort without forecasts where no
                sub-solve fans out, or when the fan points of one roll
                sequence mark different realized branches (one realization
                series per solve; ``_check_realized_switch``).
        """
        period__branch_lists: defaultdict[str, list] = defaultdict(list)
        solve_branch__time_branch_lists: defaultdict[str, list] = defaultdict(list)
        jump_lists: dict[str, Any] = dict()
        branch_start_time_lists: defaultdict[str, Any] = defaultdict()
        fan_label_time_lists: dict[str, FanLabelTimes] = dict()
        # Later forecast cohorts reachable in each fanned sub-solve; checked
        # after the loop, once every sub-solve's fan point is known.
        unfanned: dict[str, tuple[str, str, bool, list]] = dict()
        # Later realized-only cohorts (realized: yes, no forecast) reachable
        # in each fanned sub-solve; checked after the loop.
        unfanned_realized: dict[str, tuple[str, str, bool, list]] = dict()
        # Per complete solve: the stochastic_branches rows grouped by cohort
        # (period, start_step), and the sorted forecast and realized
        # branches of each cohort reached so far.  Built once per complete solve, so a roll
        # walks only its own horizon, not every roll's rows.
        cohort_rows: dict[str, dict[tuple[str, str], list]] = dict()
        cohort_forecasts: dict[str, dict[tuple[str, str], list[str]]] = dict()
        cohort_realized: dict[str, dict[tuple[str, str], list[str]]] = dict()

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
            # Branches that received active time at the branching period
            # (nonzero weight, != period, not realized).  Only these are
            # extended with active time into continuation periods.
            active_branches: list[str] = []
            branch_start_time_lists[solve] = None

            # ----------------------------------------------------------
            # Triggering-cohort validation (W1).  A solve is triggered by
            # exactly ONE reveal: the FIRST declared ``start_step`` reachable
            # while walking THIS solve's (already period/roll-sliced)
            # ``active_time_list`` in order — the same walk the trigger loop
            # below does.  ``info`` is read per COMPLETE solve, so a rolling
            # parent's every roll sees ALL rolls' reveal rows; only the row
            # that triggers THIS roll is validatable here, so we validate the
            # triggering cohort only and tolerate the rest (each is validated
            # in the roll where it triggers — a clean cross-roll pre-pass is
            # infeasible, since a mid-horizon reveal is a PERIOD boundary, not
            # a roll boundary, so it can't be told from a garbage step here).
            # The later forecast cohorts in this solve's horizon are checked
            # after the loop (``_check_unfanned_cohorts``): one that no
            # sub-solve of the same complete solve fans out at is an error.
            # Checks: (i) the trigger step must be its period's FIRST step
            # (reveals are period-granular boundaries); (ii) the cohort
            # {rows sharing that (period, start_step)} must carry ≤1
            # realized:yes (the fan is forecasts-only; realized is optional)
            # and ≥1 nonzero-weight forecast.
            #
            # No reachable reveal → deterministic trunk (``branched`` stays
            # False): correct for a rolling PRE-REVEAL roll, but for a SINGLE
            # (non-rolling) solve a non-empty ``info`` means the branches
            # would be silently ignored → raise.  Rolling vs single is
            # detected structurally via ``solve_mode == "rolling_window"``
            # (set in RecursiveSolveBuilder), not the ``_roll_N`` name.  For a
            # single solve the whole reveal picture is visible in one window,
            # so EVERY declared reveal is validated strictly (same checks).
            # ----------------------------------------------------------
            is_rolling_sub_solve = (
                getattr(
                    getattr(self.state, "solve", None), "solve_modes", {}
                ).get(complete_solves[solve], "single_solve")
                == "rolling_window"
            )

            declared_start_steps: defaultdict[str, set] = defaultdict(set)
            for row in info:
                declared_start_steps[row[0]].add(row[2])

            trig_period: str | None = None
            trig_step: str | None = None
            for period_w, active_time_w in active_time_list.items():
                if period_w in declared_start_steps:
                    for step_w in active_time_w:
                        if step_w.timestep in declared_start_steps[period_w]:
                            trig_period = period_w
                            trig_step = step_w.timestep
                            break
                if trig_period is not None:
                    break

            if trig_period is None:
                if info and not is_rolling_sub_solve:
                    message = (
                        "A single (non-rolling) stochastic solve declares "
                        "stochastic_branches but none reveal at a reachable "
                        "step of its active time — the branches would be "
                        "silently ignored. Check stochastic_branches "
                        f"start_step values against solve {solve!r}'s "
                        "timeline."
                    )
                    split_of = getattr(getattr(self.state, "solve", None), "map2d_duplicate_of", {}).get(
                        complete_solves[solve])
                    if split_of is not None:
                        split = complete_solves[solve]
                        own = list(active_time_list)
                        firsts = ", ".join(
                            f"{p}/{active_time_list[p][0].timestep}" for p in own if active_time_list[p])
                        message += (
                            f" Solve {split!r} is the 2-D period map row {split_of[1]!r} of solve "
                            f"{split_of[0]!r}, and it has the stochastic_branches of {split_of[0]!r}. Those "
                            f"start in the period(s) {', '.join(sorted(declared_start_steps))}, but {split!r} has "
                            f"only the period(s) {', '.join(own)}. Either give stochastic_branches of "
                            f"{split_of[0]!r} forecasts that also start in a period of {split!r} (at its first "
                            f"time step: {firsts}), or write the split solves as separate solves (one solve per "
                            f"row of the 2-D map, with 1-D period lists) and leave stochastic_branches out of "
                            f"the solve for {', '.join(own)}."
                        )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)
            else:
                if trig_step != active_time_list[trig_period][0].timestep:
                    message = (
                        "A stochastic branch must start at the FIRST step of "
                        "its branching period (period-boundary reveal). The "
                        f"reveal triggering solve {solve!r} is period "
                        f"{trig_period!r} start_step {trig_step!r}, which is "
                        "not that period's first step in this solve. Check "
                        "stochastic_branches."
                    )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)
                self._validate_cohort(
                    info,
                    trig_period,
                    trig_step,
                    f"The triggering stochastic branching cohort (period "
                    f"{trig_period!r}, start_step {trig_step!r})",
                    realized_suffix=(
                        " (the fan is forecasts only; realized is an optional "
                        "0-or-1 designation; start_step at the period's first "
                        "step)"
                    ),
                )

                if not is_rolling_sub_solve:
                    # Single solve: validate EVERY declared reveal, not just
                    # the triggering cohort — the whole picture is visible, so
                    # a mid-period or mis-realized reveal here is a real
                    # authoring error (pre-W1 per-row strictness).
                    for period_k in declared_start_steps:
                        if period_k not in active_time_list:
                            continue
                        period_first_k = active_time_list[period_k][0].timestep
                        for step_k in sorted(declared_start_steps[period_k]):
                            if step_k != period_first_k:
                                message = (
                                    "A stochastic branch must start at the "
                                    "FIRST step of its branching period "
                                    "(period-boundary reveal). Solve "
                                    f"{solve!r} declares period {period_k!r} "
                                    f"start_step {step_k!r}, which is not that "
                                    "period's first step. Check "
                                    "stochastic_branches."
                                )
                                self.logger.error(message)
                                raise FlexToolConfigError(message)
                        self._validate_cohort(
                            info,
                            period_k,
                            None,
                            f"Stochastic branching period {period_k!r}",
                        )

                # Later forecast cohorts in this solve's horizon (collected
                # from the pre-stochastic list; checked after the loop).
                order = list(active_time_list)
                trig_index = order.index(trig_period)
                trig_steps = [e.timestep for e in active_time_list[trig_period]]
                trig_step_index = trig_steps.index(trig_step)
                complete = complete_solves[solve]
                if complete not in cohort_rows:
                    rows_by_cohort: defaultdict[tuple[str, str], list] = (
                        defaultdict(list)
                    )
                    for row in info:
                        rows_by_cohort[(row[0], row[2])].append(row)
                    cohort_rows[complete] = dict(rows_by_cohort)
                    cohort_forecasts[complete] = dict()
                    cohort_realized[complete] = dict()
                rows_by_cohort_c = cohort_rows[complete]
                forecasts_by_cohort = cohort_forecasts[complete]
                realized_by_cohort = cohort_realized[complete]
                candidates = []
                realized_later = []
                for period_c in order[trig_index:]:
                    steps_c = [e.timestep for e in active_time_list[period_c]]
                    for idx_c, step_c in enumerate(steps_c):
                        cohort = (period_c, step_c)
                        if cohort not in rows_by_cohort_c:
                            continue
                        if period_c == trig_period and idx_c <= trig_step_index:
                            continue
                        forecasts = forecasts_by_cohort.get(cohort)
                        if forecasts is None:
                            forecasts = sorted(
                                {
                                    r[1]
                                    for r in rows_by_cohort_c[cohort]
                                    if r[3] != "yes"
                                    and float(r[4]) != 0.0
                                    and r[1] != period_c
                                }
                            )
                            forecasts_by_cohort[cohort] = forecasts
                        if forecasts:
                            candidates.append((period_c, step_c, forecasts))
                            continue
                        realized = realized_by_cohort.get(cohort)
                        if realized is None:
                            realized = sorted(
                                {
                                    r[1]
                                    for r in rows_by_cohort_c[cohort]
                                    if r[3] == "yes"
                                }
                            )
                            realized_by_cohort[cohort] = realized
                        if realized:
                            realized_later.append((period_c, step_c, realized))
                if candidates:
                    unfanned[solve] = (
                        trig_period,
                        trig_step,
                        is_rolling_sub_solve,
                        candidates,
                    )
                if realized_later:
                    unfanned_realized[solve] = (
                        trig_period,
                        trig_step,
                        is_rolling_sub_solve,
                        realized_later,
                    )

            # Walk periods — at the first branch trigger, fan out into
            # branches; subsequent periods continue the same branches (a
            # solve fans out only once: a later forecast cohort that no
            # sub-solve fans out at is rejected by _check_unfanned_cohorts,
            # and a realized: yes row that would switch the realization
            # series by _check_realized_switch).
            # NB: the reference implementation's continuation block
            # (flextoolrunner.py:1745-1752) was dead code — it never
            # wrote active/realized/fix-storage time for continuation
            # periods, silently truncating the horizon.  This module
            # intentionally diverges by actually emitting the
            # continuation timelines (realized branch keeps real period
            # names; active branches get synthetic copies).
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
                                    active_branches.append(branch)
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
                    # Continuation-after-branching: the realized branch
                    # keeps the REAL period name; each active branch is
                    # extended into this period as a synthetic
                    # '<period>_<branch>' copy.  Append order mirrors
                    # the branching period: the (period, period)
                    # self-row first, then fan rows in DB order.  The
                    # self-row is load-bearing for the branch-weight
                    # normalization (the realized member of each cohort
                    # is detected via its (d, d) row) — see
                    # derive_solve_branch_weight.
                    period__branch_lists[solve].append((period, period))
                    new_active_time_list[period] = active_time_list[period]
                    if period in realized_time_list:
                        new_realized_time_list[period] = realized_time_list[period]
                    if period in fix_storage_time_list:
                        new_fix_storage_time_list[period] = (
                            fix_storage_time_list[period]
                        )
                    for branch in branches:
                        solve_branch = period + "_" + branch
                        period__branch_lists[solve].append((period, solve_branch))
                        if branch in active_branches:
                            new_active_time_list[solve_branch] = (
                                active_time_list[period][0:]
                            )
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

            # Find the realized branch for each period, and bind the
            # reference forecast where none is designated.  Realized is an
            # optional 0-or-1 designation: a period may have no realized
            # row (forecasts-only cohort) or exactly one; 2+ → raise.  When
            # NO branch is realized (n_real=0), the real (un-suffixed)
            # period names are owned by no realized branch, so a PRE-REVEAL
            # real-named period (the shared trunk, which has no suffixed
            # forecast copy of its own) would get no solve_branch__time_branch
            # owner and its branch-keyed data (profiles, pdt params) would
            # silently fall through to the zero fallback.  In that case the
            # REFERENCE forecast (the lexically-smallest forecast raw token —
            # the same d_ref R-NA's db_pairs star uses) owns the real names
            # for pre-reveal periods so their data binds.  Reveal and
            # continuation periods carry their data on the suffixed forecast
            # copies, so their real-named self-row stays a harmless weight-0
            # phantom we leave alone (binding it would collide with the
            # recourse lineage chain grouping — see R-NAME design §2).
            #
            # ``order`` and ``d_ref_tok`` are loop-invariant across the period
            # loop (active_time_list and active_branches are fixed here) —
            # compute once.  For a hypothetical multi-reveal solve d_ref scopes
            # to the FIRST reveal cohort's forecasts; with a single reveal
            # (every current fixture) active_branches IS exactly that cohort,
            # so its lexical-min token is R-NA's reference forecast.
            order = list(active_time_list.keys())
            d_ref_tok = (
                sorted(set(active_branches))[0] if active_branches else None
            )
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
                if found > 1:
                    message = (
                        "Each period should have at most one realized branch. "
                        "Found: " + str(found) + "\n"
                    )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)

                # R-NAME: when no realized branch owns the real names
                # (n_real=0), a PRE-REVEAL real-named period (the shared
                # trunk) binds the reference forecast's data — the
                # lexically-smallest forecast raw token, the same d_ref
                # R-NA's db_pairs star uses.  Scoped to pre-reveal ONLY: a
                # blanket binding would also point the POST-reveal weight-0
                # phantom self-row at d_ref, making it join the recourse
                # lineage chain and trip the leaf-weight-constancy
                # precondition (_branch.py).  The lineage precondition
                # already excludes pre-reveal anchors, so the pre-reveal trunk
                # is safe.  A pre-reveal period has no suffixed copy of its
                # own, so this is the only owner it can get.
                if found == 0 and active_branches:
                    bs = branch_start_time_lists[solve]
                    reveal_period = bs[0] if bs is not None else None
                    if (
                        reveal_period is not None
                        and period in order
                        and reveal_period in order
                        and order.index(period) < order.index(reveal_period)
                    ):
                        solve_branch__time_branch_lists[solve].append(
                            (period, d_ref_tok)
                        )

            # A period label (the original name of a period that opens a fan) has no dispatch: drop it from
            # the active, realized and fix-storage lists.  Its pre-stochastic steps and committed cells are kept
            # aside for the bookkeeping emits (period set, non-anticipativity window).
            labels = fan_period_labels(set(period__branch_lists[solve]))
            fan_label_time_lists[solve] = FanLabelTimes(
                labels=frozenset(labels),
                active={p: v for p, v in new_active_time_list.items() if p in labels},
                realized={p: v for p, v in new_realized_time_list.items() if p in labels},
                fix_storage={p: v for p, v in new_fix_storage_time_list.items() if p in labels},
            )
            for label in labels:
                new_active_time_list.pop(label, None)
                new_realized_time_list.pop(label, None)
                new_fix_storage_time_list.pop(label, None)

            # Update the time lists for this solve.
            realized_time_lists[solve] = new_realized_time_list
            fix_storage_time_lists[solve] = new_fix_storage_time_list
            active_time_lists[solve] = new_active_time_list
            jump_lists[solve] = make_step_jump(
                new_active_time_list,
                period__branch_lists[solve],
                solve_branch__time_branch_lists[solve],
            )

        self._check_unfanned_cohorts(
            unfanned, solves, branch_start_time_lists, complete_solves
        )
        self._check_realized_switch(
            unfanned_realized,
            solves,
            branch_start_time_lists,
            complete_solves,
            cohort_rows,
        )

        return (
            period__branch_lists,
            solve_branch__time_branch_lists,
            active_time_lists,
            jump_lists,
            fix_storage_time_lists,
            realized_time_lists,
            branch_start_time_lists,
            fan_label_time_lists,
        )


__all__ = [
    "StochasticSolver",
    "connect_two_timelines",
    "find_previous_timestep",
    "find_next_timestep",
    "write_timeline_matching_map",
]
