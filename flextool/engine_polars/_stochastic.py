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

* **Continuation-after-branching**: every period after the branching
  period continues the fan.  The realized branch keeps the REAL period
  name (with its active / realized / fix-storage windows); each
  non-realized nonzero-weight branch gets a synthetic
  ``period + "_" + branch`` copy of the period's active time (LP
  variables, never committed).  The naming is "sticky" across periods.
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

        After the branching period, the fan extends through every
        remaining period of the solve: the realized branch keeps the
        REAL period names (and their realized / fix-storage windows),
        while each non-realized nonzero-weight branch ``b`` gets one
        synthetic ``period + "_" + b`` period per continuation period
        carrying a full copy of that period's active time.
        ``period__branch`` mirrors the branching period's metadata rows
        (self-row plus one row per fan member including the realized
        and zero-weight branches); ``solve_branch__time_branch`` gains
        rows only for branches that received active time.

        Real-name data ownership has a 0-or-1 realized designation: when
        a period has a realized branch it owns the real names' data; when
        the whole fan is forecasts-only (n_real=0) no realized branch
        owns them, so a PRE-REVEAL real-named period (the shared trunk)
        instead binds the REFERENCE forecast (the lexically-smallest
        forecast raw token) so its branch-keyed data does not fall through
        to the zero fallback.  Reveal / continuation periods carry their
        data on the suffixed forecast copies; their real-named self-row is
        a harmless weight-0 phantom left untouched (R-NAME).

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
            FlexToolConfigError: When the reveal that triggers a solve does
                not start at its branching period's first step (not a
                period-boundary reveal), when that triggering cohort carries
                more than one ``realized: yes`` row (0 or 1 is valid — the
                fan is forecasts only, realized is an optional designation),
                when that cohort carries no forecast branch (``realized: no``
                with nonzero weight), or when a single (non-rolling) solve
                declares ``stochastic_branches`` none of which reveal at a
                reachable step of its active time (silently-ignored
                branches).
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
            # Branches that received active time at the branching period
            # (nonzero weight, != period, not realized).  Only these are
            # extended with active time into continuation periods.
            active_branches: list[str] = []
            branch_start_time_lists[solve] = None

            # ----------------------------------------------------------
            # Triggering-cohort validation (W1 — spec §A.1/§A.2/§F-step2/
            # §H.3; replaces the old "every row at its period's first step"
            # + "one realized per period NAME" pair, which crashed rolling).
            #
            # A solve is triggered by exactly ONE reveal: the FIRST declared
            # ``start_step`` reachable while walking THIS solve's (already
            # period/roll-sliced) ``active_time_list`` in order (period,
            # then step) — the same walk the trigger loop below performs.
            # ``info`` is read per COMPLETE solve, so for a rolling parent
            # every roll sees ALL rolls' reveal rows; only the row that
            # triggers THIS roll is validatable here (the rest are other
            # rolls' triggers, each validated when it is its own roll's
            # trigger).  We therefore validate the triggering cohort only
            # and tolerate the non-triggering rows:
            #   (i)  the triggering ``start_step`` must be the FIRST step of
            #        its period in this solve's window — investment is
            #        period-granular, so a reveal is a PERIOD BOUNDARY; this
            #        admits BOTH Model-1 mid-horizon reveals and Model-2
            #        fan-at-first (and does NOT re-introduce Option-B's
            #        "first period can't branch" guard — see Appendix).
            #   (ii) the triggering cohort {rows sharing that (period,
            #        start_step)} must carry AT MOST ONE realized:yes row
            #        (0 or 1 — the fan is forecasts only; realized is an
            #        optional designation) AND AT LEAST ONE forecast branch
            #        (realized:no with nonzero weight); 2+ realized or 0
            #        forecasts → raise.
            # When no reveal is reachable in this solve's window the solve
            # stays a deterministic trunk (``branched`` remains False, so
            # ``branch_start_time_lists[solve]`` stays None and the walk
            # below copies every period as-is).  That is correct for a
            # rolling PRE-REVEAL roll; but for a SINGLE (non-rolling) solve
            # a non-empty ``info`` with no reachable reveal means the
            # branches would be silently ignored → misconfiguration, raise.
            # Rolling vs single is detected structurally: a rolling roll's
            # complete-solve carries ``solve_mode == "rolling_window"`` (set
            # in RecursiveSolveBuilder), not by sniffing the ``_roll_N``
            # name suffix.
            #
            # Fix 2 (non-triggering malformed reveals): a clean cross-roll
            # strict check is NOT feasible at this per-roll call site.  Each
            # roll sees only its own sliced window, and a declared step that
            # is some OTHER roll's legitimate trigger is indistinguishable
            # here from a jump-misaligned garbage step.  A cross-roll
            # pre-pass keyed on roll first-steps is no help either — it
            # would false-reject a legitimate Model-1 mid-horizon reveal
            # (which is a PERIOD boundary, not a roll boundary).  So each
            # reveal is validated in the roll where it triggers; this is the
            # documented limitation for the rolling case.  For a SINGLE
            # (non-rolling) solve the WHOLE reveal picture is visible in the
            # one window, so every declared reveal is validated strictly —
            # each must sit at its period's first step, each branching
            # period must carry at most one realized:yes (0 or 1) and at
            # least one forecast branch (realized:no, nonzero weight) —
            # restoring the pre-W1 per-row strictness (a mid-period or extra
            # reveal in a single solve is a real authoring error, not another
            # roll's reveal).  This raises, matching the pre-W1 behaviour.
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
                n_real = sum(
                    1
                    for r in info
                    if r[0] == trig_period and r[2] == trig_step and r[3] == "yes"
                )
                n_forecast = sum(
                    1
                    for r in info
                    if r[0] == trig_period
                    and r[2] == trig_step
                    and r[3] != "yes"
                    and float(r[4]) != 0.0
                )
                if n_real > 1:
                    message = (
                        "The triggering stochastic branching cohort (period "
                        f"{trig_period!r}, start_step {trig_step!r}) must have "
                        f"at most one realized:yes branch per revealing cohort; "
                        f"found {n_real}. Check stochastic_branches (the fan is "
                        "forecasts only; realized is an optional 0-or-1 "
                        "designation; start_step at the period's first step)."
                    )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)
                if n_forecast < 1:
                    message = (
                        "The triggering stochastic branching cohort (period "
                        f"{trig_period!r}, start_step {trig_step!r}) must "
                        "contain at least one forecast (non-realized, "
                        f"nonzero-weight) branch; found {n_forecast}. A "
                        "stochastic fan is forecasts only — realized is an "
                        "optional designation. Check stochastic_branches."
                    )
                    self.logger.error(message)
                    raise FlexToolConfigError(message)

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
                        n_real_k = sum(
                            1 for r in info
                            if r[0] == period_k and r[3] == "yes"
                        )
                        n_forecast_k = sum(
                            1 for r in info
                            if r[0] == period_k
                            and r[3] != "yes"
                            and float(r[4]) != 0.0
                        )
                        if n_real_k > 1:
                            message = (
                                f"Stochastic branching period {period_k!r} "
                                "must have at most one realized:yes branch per "
                                f"revealing cohort; found {n_real_k}. Check "
                                "stochastic_branches."
                            )
                            self.logger.error(message)
                            raise FlexToolConfigError(message)
                        if n_forecast_k < 1:
                            message = (
                                f"Stochastic branching period {period_k!r} "
                                "must contain at least one forecast "
                                "(non-realized, nonzero-weight) branch; found "
                                f"{n_forecast_k}. A stochastic fan is forecasts "
                                "only — realized is an optional designation. "
                                "Check stochastic_branches."
                            )
                            self.logger.error(message)
                            raise FlexToolConfigError(message)

            # Walk periods — at the first branch trigger, fan out into
            # branches; subsequent periods continue the same branches.
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
                # precondition (_derived_branch.py).  The lineage precondition
                # already excludes pre-reveal anchors, so the pre-reveal trunk
                # is safe.  A pre-reveal period has no suffixed copy of its
                # own, so this is the only owner it can get.
                if found == 0 and active_branches:
                    bs = branch_start_time_lists[solve]
                    reveal_period = bs[0] if bs is not None else None
                    order = list(active_time_list.keys())
                    if (
                        reveal_period is not None
                        and period in order
                        and reveal_period in order
                        and order.index(period) < order.index(reveal_period)
                    ):
                        # For a hypothetical multi-reveal solve, scope d_ref
                        # to the FIRST reveal cohort's forecasts; with a
                        # single reveal (every current fixture) active_branches
                        # IS exactly that cohort, so its lexical-min token is
                        # R-NA's reference forecast.
                        d_ref_tok = sorted(set(active_branches))[0]
                        solve_branch__time_branch_lists[solve].append(
                            (period, d_ref_tok)
                        )

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
