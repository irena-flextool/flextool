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
    from spinedb_api import DatabaseMapping
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
        timeset_name -> new_duration.
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
        new_step_durations: dict = params_to_dict(
            db=db, cl="timeset", par="new_stepduration", mode=DictMode.DICT
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
                # params_to_dict may return list of tuples for nested maps
                weight_dict = {}
                for entry in nested_map:
                    if len(entry) >= 2:
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

    def create_timeline_from_timestep_duration(self) -> None:
        """Synthesize new timelines when a timeset has a new_stepduration parameter."""
        for timeset_name, timeset in list(self.timeset_durations.items()):
            if timeset_name in self.new_step_durations:
                step_duration = float(self.new_step_durations[timeset_name])
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
                                logging.getLogger(__name__).warning(
                                    "Warning: All new steps are not the size of the given "
                                    "step duration. The new step duration has to be multiple "
                                    "of old step durations for this to happen."
                                )
                        step_counter += float(step[1])
                    new_steps.append((first_step, str(step_counter)))
                    added_steps += 1
                    new_timesets.append((ts[0], added_steps))
                self.timeset_durations[timeset_name] = new_timesets
                new_timeline_name = timeline_name + "_" + timeset_name
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
        work_folder: "Path | None" = None,
    ) -> None:
        """Average or sum timeseries data when step durations have been changed.

        If no timeset in the solve uses new_step_durations, simply copies files.
        Otherwise re-aggregates each timeseries file to match the new step size.

        Args:
            solve: Name of the current solve.
            solve_config: Solve configuration.
            logger: Logger instance.
            work_folder: Optional working directory.  When provided, all
                ``input/`` and ``solve_data/`` paths are resolved relative to
                this folder instead of the current working directory.
        """
        from pathlib import Path

        wf = Path(work_folder) if work_folder is not None else Path.cwd()
        timeseries_map: dict[str, str] = {
            'pt_node_inflow.csv': "sum",
            'pt_commodity.csv': "average",
            'pt_group.csv': "average",
            'pt_node.csv': "average",
            'pt_process.csv': "average",
            'pt_profile.csv': "average",
            'pt_process_source.csv': "average",
            'pt_process_sink.csv': "average",
            'pt_reserve__upDown__group.csv': "average",
            'pbt_node_inflow.csv': "sum",
            'pbt_node.csv': "average",
            'pbt_process.csv': "average",
            'pbt_profile.csv': "average",
            'pbt_process_source.csv': "average",
            'pbt_process_sink.csv': "average",
            'pbt_reserve__upDown__group.csv': "average",
        }
        create = False
        for period_timeset in solve_config.timesets_used_by_solves[solve]:
            if period_timeset[1] in self.new_step_durations:
                create = True
        if not create:
            for timeseries in timeseries_map:
                shutil.copy(str(wf / 'input' / timeseries), str(wf / 'solve_data' / timeseries))
        else:
            timelines_list: list[str] = []
            for period, timeset in solve_config.timesets_used_by_solves[solve]:
                timeline = self.timesets__timeline[timeset]
                if timeline not in timelines_list:
                    if len(timelines_list) != 0:
                        message = (
                            "Error: More than one timeline in the solve or the same timeline "
                            "with different step durations in different timesets"
                        )
                        logger.error(message)
                        raise FlexToolConfigError(message)
                    timelines_list.append(timeline)
            # Pre-build timeline duration lookup for O(1) access per row
            timeline_duration_lookup: dict[str, int] = {}
            for timeline in timelines_list:
                new_timeline = self.timelines[timeline]
                for timeline_row in new_timeline:
                    timeline_duration_lookup[timeline_row[0]] = int(float(timeline_row[1]))
            for timeseries in timeseries_map:
                with open(str(wf / 'input' / timeseries), 'r', encoding='utf-8') as blk:
                    filereader = csv.reader(blk, delimiter=',')
                    with open(str(wf / 'solve_data' / timeseries), 'w', newline='') as solve_file:
                        filewriter = csv.writer(solve_file, delimiter=',')
                        headers = next(filereader)
                        filewriter.writerow(headers)
                        time_index = headers.index('time')
                        while True:
                            try:
                                datain = next(filereader)
                                timeline_step_duration = timeline_duration_lookup.get(datain[time_index])
                                if timeline_step_duration is not None:
                                    values: list[float] = []
                                    params = datain[0:time_index]
                                    row = datain[0:time_index + 1]
                                    values.append(float(datain[time_index + 1]))
                                    if datain[1] != 'storage_state_reference_value':
                                        for i in range(timeline_step_duration - 1):
                                            datain = next(filereader)
                                            if datain[0:time_index] != params:
                                                message = (
                                                    "Cannot find the same timesteps in input "
                                                    "data as in timeline for file  "
                                                    + timeseries + " after " + row[-1]
                                                )
                                                logger.error(message)
                                                raise FlexToolConfigError(message)
                                            values.append(float(datain[time_index + 1]))

                                    if timeseries_map[timeseries] == "average":
                                        out_value = round(sum(values) / len(values), 6)
                                    else:
                                        out_value = sum(values)
                                    row.append(out_value)
                                    filewriter.writerow(row)
                                else:
                                    if datain[1] == 'storage_state_reference_value':
                                        counter = 0
                                        for timestep in self.timelines[
                                            self.original_timeline[timeline]
                                        ]:
                                            if datain[2] == timestep[0]:
                                                current_index = counter
                                            counter += 1
                                        found = False
                                        for timestep in reversed(
                                            self.timelines[self.original_timeline[timeline]][
                                                0 : current_index + 1
                                            ]
                                        ):
                                            for timeline_row in new_timeline:
                                                if timeline_row[0] == timestep[0]:
                                                    new_index = timeline_row[0]
                                                    found = True
                                                    break
                                            if found:
                                                break
                                        if found:
                                            row = datain[0:time_index]
                                            row.append(new_index)
                                            row.append(datain[time_index + 1])
                                            filewriter.writerow(row)
                            except StopIteration:
                                break
            # Constrain inflow to a longer step size
            node__inflow: list[list[str]] = []
            with open(str(wf / 'input/p_node.csv'), 'r', encoding='utf-8') as blk:
                filereader = csv.reader(blk, delimiter=',')
                _read_header = next(filereader)
                while True:
                    try:
                        datain = next(filereader)
                        if datain[1] == 'inflow':
                            node__inflow.append([datain[0], datain[2]])
                    except StopIteration:
                        break
            with open(str(wf / 'solve_data/pt_node_inflow.csv'), 'a', newline='') as blk:
                filewriter = csv.writer(blk, delimiter=',')
                for timeline in timelines_list:
                    new_timeline = self.timelines[timeline]
                    for node__value in node__inflow:
                        for timeline_row in new_timeline:
                            timeline_step_duration = int(float(timeline_row[1]))
                            value = float(node__value[1]) * timeline_step_duration
                            row_out = [node__value[0], timeline_row[0], value]
                            filewriter.writerow(row_out)


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
    timelines: dict, solve__period__timeset: dict, work_folder: "Path | None" = None
) -> None:
    """Separate period data from timeseries data in pdt_*.csv input files.

    Writes pd_*.csv (period rows) and pt_*.csv (timeseries rows) into input/.

    Args:
        timelines: Dict of timeline data.
        solve__period__timeset: Dict mapping solves to period/timeset pairs.
        work_folder: Optional working directory.  When provided, all
            ``input/`` paths are resolved relative to this folder instead of
            the current working directory.
    """
    from pathlib import Path

    wf = Path(work_folder) if work_folder is not None else Path.cwd()

    inputfiles = ['pdt_commodity.csv', 'pdt_group.csv']
    for inputfile in inputfiles:
        output_period = str(wf / f'input/pd_{inputfile[4:]}')
        output_timeseries = str(wf / f'input/pt_{inputfile[4:]}')
        timesteps: list[str] = []
        for timeline in list(timelines.values()):
            for step in timeline:
                timesteps.append(step[0])
        periods: list[str] = []
        for period__timesets in solve__period__timeset.values():
            for period__timeset in period__timesets:
                periods.append(period__timeset[0])

        with open(output_period, 'w', newline='') as blk_p:
            period_writer = csv.writer(blk_p, delimiter=',')
            with open(output_timeseries, 'w', newline='') as blk_t:
                timeseries_writer = csv.writer(blk_t, delimiter=',')
                with open(str(wf / f'input/{inputfile}'), 'r', encoding='utf-8') as blk:
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
