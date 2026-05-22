"""
TimelineConfig — timeline-level state loaded from the database.

All timeline definitions, timeset mappings, and methods that build or
transform timeline data live here.  Free functions at the bottom of the
module (get_active_time, make_step_jump, make_steps, make_timeset_timeline,
separate_period_and_timeseries_data) are pure — they have no side effects
beyond their return values (or the files they write).
"""
from __future__ import annotations

import csv
import logging
import math
import shutil
from collections import defaultdict
from typing import TYPE_CHECKING

from flextool.flextoolrunner.db_reader import DictMode, get_single_entities, params_to_dict
from flextool.flextoolrunner.runner_state import ActiveTimeEntry, FlexToolConfigError

if TYPE_CHECKING:
    from pathlib import Path

    from spinedb_api import DatabaseMapping
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    from flextool.flextoolrunner.solve_config import SolveConfig


class TimelineConfig:
    """All timeline definitions and timeset mappings for a FlexTool run.

    Attributes
    ----------
    timelines : defaultdict[str, list[tuple]]
        timeline_name -> [(timestep, duration), ...]
    timesets : list[str]
        timeset entity name list.
    timesets__timeline : defaultdict[str, str]
        timeset_name -> timeline_name.
    timeset_durations : defaultdict[str, list[tuple]]
        timeset_name -> [(start, count), ...].
    new_step_durations : dict[str, str]
        solve_name -> new_duration (hours).  Solve-scoped as of DB v50
        (prior to v50 this was keyed by timeset_name; the runtime
        already required all timesets within a solve to carry the same
        value, so the effective scope is unchanged).
    rp_weights : dict[str, dict]
        timeset_name -> {base_start: {rep_start: weight}}.
    stochastic_timesteps : defaultdict[str, list[tuple]]
        Mutable — populated during the solve loop.
    original_timeline : defaultdict[str, str]
        new_timeline_name -> original_timeline_name.
    """

    def __init__(
        self,
        timelines: defaultdict,
        timesets: list,
        timesets__timeline: defaultdict,
        timeset_durations: defaultdict,
        new_step_durations: dict,
        rp_weights: dict | None = None,
        timeset_weights: dict | None = None,
    ) -> None:
        self.timelines = timelines
        self.timesets = timesets
        self.timesets__timeline = timesets__timeline
        self.timeset_durations = timeset_durations
        self.new_step_durations = new_step_durations
        self.rp_weights: dict = rp_weights or {}
        # timeset_name -> {timestep: weight}. Raw user input; runner normalizes.
        self.timeset_weights: dict[str, dict[str, float]] = timeset_weights or {}

        # Mutable state — populated later
        self.stochastic_timesteps: defaultdict[str, list] = defaultdict(list)
        self.original_timeline: defaultdict[str, str] = defaultdict()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(cls, db: DatabaseMapping, logger: logging.Logger) -> TimelineConfig:
        """Read all timeline-level parameters from *db* and return a TimelineConfig."""
        timelines: defaultdict = params_to_dict(
            db=db, cl="timeline", par="timestep_duration", mode=DictMode.DEFAULTDICT
        )
        timesets: list = get_single_entities(
            db=db, entity_class_name="timeset"
        )
        timeset_durations: defaultdict = params_to_dict(
            db=db, cl="timeset", par="timeset_duration", mode=DictMode.DEFAULTDICT
        )
        timesets__timeline: defaultdict = params_to_dict(
            db=db, cl="timeset", par="timeline", mode=DictMode.DEFAULTDICT
        )
        # Solve-scoped as of DB v50 (moved from timeset.new_stepduration
        # by db_migration v50).  Keys are solve entity names.
        new_step_durations: dict = params_to_dict(
            db=db, cl="solve", par="new_stepduration", mode=DictMode.DICT
        )
        # Read representative period weights (nested Map: base_start -> {rep_start -> weight})
        rp_weights_raw = params_to_dict(
            db=db, cl="timeset", par="representative_period_weights", mode=DictMode.DICT
        )
        rp_weights: dict = {}
        import spinedb_api as api
        for timeset_name, nested_map in rp_weights_raw.items():
            if isinstance(nested_map, api.Map):
                weight_dict: dict[str, dict[str, float]] = {}
                for i, base_key in enumerate(nested_map.indexes):
                    inner = nested_map.values[i]
                    if isinstance(inner, api.Map):
                        weight_dict[str(base_key)] = {
                            str(k): float(v) for k, v in zip(inner.indexes, inner.values)
                        }
                rp_weights[timeset_name] = weight_dict
            elif isinstance(nested_map, list):
                # ``params_to_dict`` returns one of two list shapes when the
                # nested Map is converted to a table:
                #   (a) Flat triples [base, rep, weight] — when the full
                #       Map-of-Map is flattened.
                #   (b) [(base, inner_map_or_list), ...] — when only the
                #       outer Map is flattened.
                # Both decode to {base: {rep: weight}}.
                weight_dict: dict[str, dict[str, float]] = {}
                for entry in nested_map:
                    if len(entry) >= 3 and not isinstance(entry[1], (list, api.Map)):
                        base_key = str(entry[0])
                        rep_key = str(entry[1])
                        weight = float(entry[2])
                        weight_dict.setdefault(base_key, {})[rep_key] = weight
                    elif len(entry) >= 2:
                        base_key = str(entry[0])
                        inner_data = entry[1]
                        if isinstance(inner_data, list):
                            weight_dict[base_key] = {str(k): float(v) for k, v in inner_data}
                        elif isinstance(inner_data, api.Map):
                            weight_dict[base_key] = {
                                str(k): float(v) for k, v in zip(inner_data.indexes, inner_data.values)
                            }
                if weight_dict:
                    rp_weights[timeset_name] = weight_dict

        # Read timeset_weights (flat Map: timestep -> weight). Used for non-RP
        # cost/slack weighting when timesteps represent unequal year fractions.
        timeset_weights_raw = params_to_dict(
            db=db, cl="timeset", par="timeset_weights", mode=DictMode.DICT
        )
        timeset_weights: dict[str, dict[str, float]] = {}
        for timeset_name, flat_map in timeset_weights_raw.items():
            if isinstance(flat_map, api.Map):
                timeset_weights[timeset_name] = {
                    str(k): float(v) for k, v in zip(flat_map.indexes, flat_map.values)
                }
            elif isinstance(flat_map, list):
                timeset_weights[timeset_name] = {
                    str(entry[0]): float(entry[1]) for entry in flat_map if len(entry) >= 2
                }
        return cls(
            timelines=timelines,
            timesets=timesets,
            timesets__timeline=timesets__timeline,
            timeset_durations=timeset_durations,
            new_step_durations=new_step_durations,
            rp_weights=rp_weights,
            timeset_weights=timeset_weights,
        )

    # ------------------------------------------------------------------
    # Methods (moved from FlexToolRunner)
    # ------------------------------------------------------------------

    def create_timeline_from_timestep_duration(
        self, solve_config: SolveConfig
    ) -> None:
        """Synthesize new timelines for every solve that sets new_stepduration.

        As of DB v50 ``new_stepduration`` is a ``solve`` parameter
        (previously it was on ``timeset``; the runtime has always
        required the value to be identical across every timeset used
        by a given solve, so the scope change is a no-op semantically).

        Each solve with a ``new_stepduration`` value gets:

        * a freshly aggregated timeline named ``"{timeline}_{solve}"``
          in :attr:`timelines`,
        * every timeset it uses re-pointed at that new timeline in
          :attr:`timesets__timeline`,
        * re-computed :attr:`timeset_durations` entries for those
          timesets,
        * a :attr:`original_timeline` entry mapping the new timeline
          back to the original.

        If a timeset is shared between two solves with different
        ``new_stepduration`` values the runtime still needs one
        timeline per timeset — any such config should already have
        been rejected at migration time.
        """
        logger = logging.getLogger(__name__)
        for solve_name, step_duration_raw in self.new_step_durations.items():
            if step_duration_raw is None:
                continue
            step_duration = float(step_duration_raw)
            period_timesets = solve_config.timesets_used_by_solves.get(solve_name, [])
            # De-duplicate timeset names while keeping their order.
            seen_timesets: set[str] = set()
            timesets_in_solve: list[str] = []
            for _period, timeset_name in period_timesets:
                if timeset_name in seen_timesets:
                    continue
                seen_timesets.add(timeset_name)
                timesets_in_solve.append(timeset_name)

            for timeset_name in timesets_in_solve:
                timeset = self.timeset_durations.get(timeset_name)
                if not timeset:
                    continue
                timeline_name = self.timesets__timeline[timeset_name]
                old_steps = self.timelines[timeline_name]
                new_steps: list[tuple[str, str]] = []
                new_timesets: list[tuple[str, int]] = []
                for ts in timeset:
                    first_step = ts[0]
                    first_index = [step[0] for step in old_steps].index(ts[0])
                    step_counter = 0
                    last_index = first_index + int(float(ts[1]))
                    added_steps = 0
                    for step in old_steps[first_index:last_index]:
                        if step_counter >= step_duration:
                            new_steps.append((first_step, str(step_counter)))
                            first_step = step[0]
                            step_counter = 0
                            added_steps += 1
                            if step_counter > step_duration:
                                logger.warning(
                                    "Warning: All new steps are not the size of the given "
                                    "step duration. The new step duration has to be multiple "
                                    "of old step durations for this to happen."
                                )
                        step_counter += float(step[1])
                    new_steps.append((first_step, str(step_counter)))
                    added_steps += 1
                    new_timesets.append((ts[0], added_steps))
                self.timeset_durations[timeset_name] = new_timesets
                new_timeline_name = timeline_name + "_" + solve_name
                self.timelines[new_timeline_name] = new_steps
                self.timesets__timeline[timeset_name] = new_timeline_name
                self.original_timeline[new_timeline_name] = timeline_name

    def create_assumptive_parts(self, solve_config: SolveConfig) -> None:
        """Create missing timestructure parts from assumptions when data is incomplete.

        This covers multiple fallback scenarios:
        - No timeset defined → create from single timeline
        - No timeset_durations → create full-timeline timeset
        - No timeset:timeline mapping → use single timeline
        - No model:solves → use single solve
        - No solve entity → create from periods_available
        - No period_timeset → create from single timeset
        - No realized/invest periods → assume all periods realized
        - No years_represented → assume 1 year per period
        """
        # If no timeset defined and only one timeline exists, create a timeset from timeline
        if not self.timesets and len(self.timelines) == 1:
            timeset_name = "full_timeline"
            self.timesets = list(timeset_name)

        # If no timeset_durations defined for a timeset and only one timeline exists
        for timeset_name in self.timesets:
            if not self.timeset_durations and len(self.timelines) == 1:
                self.timesets__timeline[timeset_name] = list(self.timelines.keys())[0]
                self.timeset_durations[timeset_name] = [
                    (list(self.timelines.values())[0][0][0], len(list(self.timelines.values())[0]))
                ]

        # If timeset:timeline not defined and only one timeline exists, use that timeline
        for timeset_name, block in self.timeset_durations.items():
            if timeset_name not in self.timesets__timeline:
                if len(self.timelines) == 1:
                    self.timesets__timeline[timeset_name] = list(self.timelines.keys())[0]
                elif len(self.timelines) > 1:
                    message = (
                        "More than one timeline available and FlexTool does not know which "
                        "ones to use. Please use 'timeline' parameter of 'timeset' class to "
                        "define which timelines are part of the timeset(s) in the model instance"
                    )
                    logging.getLogger(__name__).error(message)
                    raise FlexToolConfigError(message)

        # If model:solves does not exist and only one solve exists, use that
        if not solve_config.model_solve:
            if len(solve_config.model) == 1:
                solve = None
                if len(solve_config.timesets_used_by_solves) == 1:
                    solve = list(solve_config.timesets_used_by_solves.keys())[0]
                elif len(solve_config.timesets_used_by_solves) > 1:
                    message = (
                        "Data contains multiple solve entities and FlexTool does not know "
                        "which to use. Please use 'solves' parameter of 'model' class to "
                        "inform which solves are to be included in the model instance."
                    )
                    logging.getLogger(__name__).error(message)
                    raise FlexToolConfigError(message)
                all_solves_in_periods = (
                    set(solve_config.realized_periods.keys())
                    | set(solve_config.invest_periods.keys())
                )
                for solve_name in all_solves_in_periods:
                    if solve:
                        if solve_name != solve:
                            message = (
                                "Data contains multiple solve entities and FlexTool does not "
                                "know which to use. Please use 'solves' parameter of 'model' "
                                "class to inform which solves are to be included in the model "
                                "instance."
                            )
                            logging.getLogger(__name__).error(message)
                            raise FlexToolConfigError(message)
                    else:
                        solve = solve_name
                solve_config.model_solve[solve_config.model[0]] = [solve]
            else:
                message = (
                    "More than one model entity found in the database and FlexTool does "
                    "not know which to use."
                )
                logging.getLogger(__name__).error(message)
                raise FlexToolConfigError(message)

        # If no solve entity for model:solves, create a solve entity with all available periods
        for model, solves in solve_config.model_solve.items():
            for solve in solves:
                if (
                    solve not in solve_config.realized_periods
                    and solve not in solve_config.invest_periods
                    and solve not in solve_config.timesets_used_by_solves
                ):
                    if model in solve_config.periods_available:
                        for period in solve_config.periods_available[model]:
                            solve_config.realized_periods[solve].append((period, period))
                    else:
                        message = (
                            f"The solve {solve} in the model: solves array does not have any "
                            f"periods defined: (period_timeset, realized_periods, invest_periods)\n"
                            f"Alternatively add periods_available to the model to create simple "
                            f"full timelines for those periods"
                        )
                        logging.getLogger(__name__).error(message)
                        raise FlexToolConfigError(message)

        # If solve:period_timeset does not exist and only one timeset exists
        for solve in list(solve_config.model_solve.values())[0]:
            if solve not in list(solve_config.timesets_used_by_solves.keys()):
                if len(self.timeset_durations) == 1:
                    period__timeset_list: list[tuple[str, str]] = []
                    timeset_name = list(self.timeset_durations.keys())[0]
                    for period_tuple in (
                        solve_config.invest_periods.get(solve, [])
                        + solve_config.realized_periods.get(solve, [])
                    ):
                        period__timeset_list.append((period_tuple[1], timeset_name))
                    solve_config.timesets_used_by_solves[solve] = period__timeset_list
                else:
                    message = (
                        "More than one timeset available and FlexTool does not know which "
                        "ones to use. Please use 'period_timeset' parameter of 'solve' class "
                        "to define which periods and which timesets are part of the solve(s) "
                        "in the model instance"
                    )
                    logging.getLogger(__name__).error(message)
                    raise FlexToolConfigError(message)

        # If realized/invest/nested solves do not exist but period__timeset exists
        for solve in list(solve_config.model_solve.values())[0]:
            if (
                solve not in solve_config.realized_periods
                and solve not in solve_config.invest_periods
                and not solve_config.contains_solves[solve]
                and solve in solve_config.timesets_used_by_solves
            ):
                for period_timeset in solve_config.timesets_used_by_solves[solve]:
                    solve_config.realized_periods[solve].append(
                        (period_timeset[0], period_timeset[0])
                    )

        # If solve_period_years_represented not set, assume 1 year per period
        for solve in list(solve_config.model_solve.values())[0]:
            all_periods_tuples = (
                solve_config.realized_periods[solve] + solve_config.invest_periods[solve]
            )
            all_periods = {item for tup in all_periods_tuples for item in tup}
            for period in all_periods:
                if solve not in solve_config.solve_period_years_represented:
                    solve_config.solve_period_years_represented[solve].append([period, 1.0])

    def create_averaged_timeseries(
        self, solve: str, solve_config: SolveConfig, logger: logging.Logger,
        *,
        provider: "object",
        work_folder: "Path | None" = None,
    ) -> None:
        """Average or sum timeseries data when step durations have been changed.

        Delegates to :func:`flextool.engine_polars._timeline.TimelineConfig.create_averaged_timeseries`
        — Step 2.5-E Phase C routes all reads / writes through the
        active :class:`FlexDataProvider`.  *provider* is required; the
        cascade's per-sub-solve Provider supplies ``input/pt_*`` and
        receives the derived ``solve_data/pt_*`` frames.
        """
        from flextool.engine_polars._timeline import (
            TimelineConfig as _NativeTimelineConfig,
        )
        # The native ``create_averaged_timeseries`` is a method on the
        # native TimelineConfig but only consults the same attributes
        # this legacy class carries (``timelines``, ``new_step_durations``,
        # ``timesets__timeline``, ``original_timeline``).  Bind it to
        # ``self`` directly to avoid duplicating state.
        _NativeTimelineConfig.create_averaged_timeseries(
            self, solve, solve_config, logger,
            provider=provider,
            work_folder=work_folder,
        )


# ------------------------------------------------------------------
# Free functions (pure — no side effects beyond return values / file I/O)
# ------------------------------------------------------------------


def get_active_time(
    current_solve: str,
    timesets_used_by_solves: dict,
    timesets: dict,
    timelines: dict,
    timesets__timelines: dict,
) -> defaultdict:
    """Map periods to their corresponding timeline entries for a given solve.

    Returns a defaultdict mapping period IDs to lists of
    (timestep, index, duration) tuples.
    """
    active_time: defaultdict = defaultdict(list)

    if current_solve not in timesets_used_by_solves:
        raise ValueError(
            f"{current_solve}: this solve does not have period_timeset defined. "
            "Check that it has period_timeset parameter defined and the names of "
            "period and timeset are spelled correctly (case sensitive). Check that "
            "the alternative is included in the scenario."
        )

    # Pre-build index for O(1) timestep-to-position lookup per timeline
    timeline_index_cache: dict[str, dict[str, int]] = {}

    for period, timeset_id in timesets_used_by_solves[current_solve]:
        timeline_id = timesets__timelines.get(timeset_id)
        if not timeline_id:
            continue
        timeline_data = timelines.get(timeline_id, [])
        if not timeline_data:
            continue
        # Build index for this timeline if not yet cached
        if timeline_id not in timeline_index_cache:
            timeline_index_cache[timeline_id] = {
                time_val: idx for idx, (time_val, _) in enumerate(timeline_data)
            }
        time_val_to_idx = timeline_index_cache[timeline_id]
        for start_time, duration in timesets[timeset_id]:
            idx = time_val_to_idx.get(start_time)
            if idx is not None:
                for step in range(int(float(duration))):
                    if idx + step < len(timeline_data):
                        entry = timeline_data[idx + step]
                        active_time[period].append(ActiveTimeEntry(timestep=entry[0], index=idx + step, duration=entry[1]))

    if not active_time:
        raise ValueError(
            f"{current_solve}: Failed to map timeset to timeline. Check that all "
            "timeset entities have timeline parameter defined and the name of the "
            "timeline is spelled correctly (case sensitive). Check that the "
            "alternative of the timeline parameter is included in the modelled scenario."
        )

    return active_time


def make_step_jump(
    active_time_list: dict,
    period__branch: list[tuple],
    solve_branch__time_branch_list: list[tuple],
) -> list[tuple]:
    """Build a list of step-jump entries for the solver.

    Each entry describes the jump from one simulation step to the next,
    including cross-period jumps.
    """
    step_lengths: list[tuple] = []
    period_start_pos = 0
    period_counter = -1
    first_period_name = list(active_time_list)[0]
    last_period_name = list(active_time_list)[-1]
    for period, active_time in reversed(active_time_list.items()):
        period_counter -= 1
        period_last = len(active_time)
        block_last = len(active_time) - 1
        if period == first_period_name:
            previous_period_name = last_period_name
        else:
            previous_period_name = list(active_time_list)[period_counter]
        for i, step in enumerate(reversed(active_time)):
            j = period_last - i - 1
            if j > 0:
                jump = active_time[j].index - active_time[j - 1].index
                if jump > 1:
                    step_lengths.insert(
                        period_start_pos,
                        (period, step.timestep, active_time[j - 1].timestep, active_time[block_last].timestep,
                         period, active_time[j - 1].timestep, jump),
                    )
                    block_last = j - 1
                else:
                    step_lengths.insert(
                        period_start_pos,
                        (period, step.timestep, active_time[j - 1].timestep, active_time[j - 1].timestep,
                         period, active_time[j - 1].timestep, jump),
                    )
            else:
                if (period, period) not in period__branch:
                    for pb in period__branch:
                        if pb[1] == period:
                            original_period = pb[0]
                    if (
                        (original_period, original_period) in period__branch
                        and original_period in active_time_list
                    ):
                        jump = active_time[j].index - active_time[-1].index
                        step_lengths.insert(
                            period_start_pos,
                            (period, step.timestep, active_time[j - 1].timestep, active_time[block_last].timestep,
                             period, active_time_list[period][-1].timestep, jump),
                        )
                    elif (original_period, original_period) in period__branch:
                        jump = active_time[j].index - active_time[-1].index
                        step_lengths.insert(
                            period_start_pos,
                            (period, step.timestep, active_time[j - 1].timestep, active_time[block_last].timestep,
                             period, active_time_list[period][-1].timestep, jump),
                        )
                    else:
                        for sb_tb in solve_branch__time_branch_list:
                            if sb_tb[0] == period:
                                time_branch = sb_tb[1]
                        past = False
                        found = False
                        previous_period_with_branch = None
                        for solve_period, a_t in reversed(active_time_list.items()):
                            if past:
                                for sb_tb in solve_branch__time_branch_list:
                                    if sb_tb[0] == solve_period and sb_tb[1] == time_branch:
                                        previous_period_with_branch = solve_period
                                        found = True
                                if found:
                                    break
                            else:
                                if solve_period == period:
                                    past = True
                        jump = active_time[j].index - active_time_list[previous_period_with_branch][-1].index
                        step_lengths.insert(
                            period_start_pos,
                            (period, step.timestep, active_time[j - 1].timestep, active_time[block_last].timestep,
                             previous_period_with_branch,
                             active_time_list[previous_period_with_branch][-1].timestep, jump),
                        )
                else:
                    jump = active_time[j].index - active_time_list[previous_period_name][-1].index
                    step_lengths.insert(
                        period_start_pos,
                        (period, step.timestep, active_time[j - 1].timestep, active_time[block_last].timestep,
                         previous_period_name, active_time_list[previous_period_name][-1].timestep,
                         jump),
                    )
    return step_lengths


def make_period_block(active_time_list: dict) -> tuple[list[tuple], list[tuple]]:
    """Build period_block_time and period_block_succ data.

    Blocks are maximal contiguous runs of active steps in the timeline —
    same detection rule as make_step_jump (jump > 1 starts a new block).
    Used by the bind_intraperiod_blocks storage binding method.

    Returns:
        period_block_time: [(period, block_first, step)], one row per active step
        period_block_succ: [(period, block_first, block_first_next)], cyclic within period
    """
    period_block_time: list[tuple] = []
    period_block_succ: list[tuple] = []

    for period, active_time in active_time_list.items():
        if not active_time:
            continue
        block_firsts_in_order: list[str] = [active_time[0].timestep]
        cur_block_first = active_time[0].timestep
        for j, step in enumerate(active_time):
            if j > 0 and active_time[j].index - active_time[j - 1].index > 1:
                cur_block_first = step.timestep
                block_firsts_in_order.append(cur_block_first)
            period_block_time.append((period, cur_block_first, step.timestep))

        n_blocks = len(block_firsts_in_order)
        for i, b_first in enumerate(block_firsts_in_order):
            b_next = block_firsts_in_order[(i + 1) % n_blocks]
            period_block_succ.append((period, b_first, b_next))

    return period_block_time, period_block_succ


def make_steps(steplist: list, start: int, stop: int) -> list:
    """Return a slice of *steplist* from *start* to *stop* (inclusive)."""
    active_step = start
    steps: list = []
    while active_step <= stop:
        steps.append(steplist[active_step])
        active_step += 1
    return steps


def make_timeset_timeline(steplist: list, start: str, length: float) -> list:
    """Build a timeset timeline from a steplist, starting at *start* for *length* steps."""
    result: list = []
    startnum = steplist.index(start)
    for i in range(startnum, math.ceil(startnum + float(length))):
        result.append(steplist[i])
    return result


def separate_period_and_timeseries_data(
    timelines: dict,
    solve__period__timeset: dict,
    *,
    provider: "FlexDataProvider",
    work_folder: "Path | None" = None,
) -> None:
    """Separate period data from timeseries data in pdt_*.csv input files.

    Reads the source ``pdt_<class>.csv`` frame from the cascade-input
    Provider (Step 2.5-G Phase A) — falling back to disk for off-cascade
    callers via :func:`_provider_open` — and writes the resulting
    ``pd_<class>.csv`` (period rows) and ``pt_<class>.csv`` (timeseries
    rows) into ``input/`` for the legacy downstream consumers that still
    read those CSVs from disk (e.g.
    :mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`).

    The frame splits land on *provider* too, mirroring the engine_polars
    twin at :func:`flextool.engine_polars._timeline.separate_period_and_timeseries_data`,
    so engine_polars-side readers can pick them up via the Provider.

    Args:
        timelines: Dict of timeline data.
        solve__period__timeset: Dict mapping solves to period/timeset pairs.
        provider: Required cascade-input Provider — supplies the
            ``input/pdt_<class>`` source frame and receives the derived
            ``input/pd_<class>`` / ``input/pt_<class>`` shards.
        work_folder: Optional working directory.  When provided, all
            ``input/`` paths are resolved relative to this folder instead of
            the current working directory.
    """
    from pathlib import Path

    from flextool.engine_polars._emit_provider_io import (
        _provider_key,
        _provider_open,
    )

    if provider is None:  # pragma: no cover — explicit guard
        raise TypeError(
            "separate_period_and_timeseries_data requires a FlexDataProvider; "
            "the cascade-input Provider is the canonical source for the "
            "input/pdt_<class> frames (Step 2.5-G Phase A)."
        )

    wf = Path(work_folder) if work_folder is not None else Path.cwd()

    inputfiles = ['pdt_commodity.csv', 'pdt_group.csv']
    for inputfile in inputfiles:
        output_period = str(wf / f'input/pd_{inputfile[4:]}')
        output_timeseries = str(wf / f'input/pt_{inputfile[4:]}')
        in_path = wf / 'input' / inputfile
        in_key = _provider_key(in_path)
        timesteps: list[str] = []
        for timeline in list(timelines.values()):
            for step in timeline:
                timesteps.append(step[0])
        periods: list[str] = []
        for period__timesets in solve__period__timeset.values():
            for period__timeset in period__timesets:
                periods.append(period__timeset[0])

        handle = _provider_open(provider, in_key, in_path)
        if handle is None:
            # No source — neither Provider nor disk carries the frame.
            # Skip the split; downstream consumers tolerate missing pd_/pt_
            # CSVs.  Keep the empty CSVs absent to mirror the historical
            # behaviour where this function would have raised; the explicit
            # ``continue`` makes the new memory-only contract observable.
            continue
        with open(output_period, 'w', newline='') as blk_p:
            period_writer = csv.writer(blk_p, delimiter=',')
            with open(output_timeseries, 'w', newline='') as blk_t:
                timeseries_writer = csv.writer(blk_t, delimiter=',')
                with handle as blk:
                    filereader = csv.reader(blk, delimiter=',')
                    headers = next(filereader)
                    timeseries_writer.writerow(headers)
                    period_writer.writerow(
                        headers[:-2] + ['period', f'pd_{headers[-1][3:]}']
                    )
                    while True:
                        try:
                            datain = next(filereader)
                            if datain[2] in periods:
                                period_writer.writerow(datain)
                            elif datain[2] in timesteps:
                                timeseries_writer.writerow(datain)
                        except StopIteration:
                            break
