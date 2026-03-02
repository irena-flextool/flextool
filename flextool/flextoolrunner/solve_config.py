"""
SolveConfig — solve-level state loaded from the database.

All solve-level parameters (solver settings, period/timeset mappings,
rolling-window parameters, stochastic branches, etc.) live here.
Mutable tracking lists (real_solves, first_of_complete_solve, last_of_solve)
are also initialised here and mutated during the solve loop.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import spinedb_api as api

from flextool.flextoolrunner.db_reader import DictMode, get_single_entities, params_to_dict

if TYPE_CHECKING:
    from spinedb_api import DatabaseMapping


@dataclass
class HiGHSConfig:
    """HiGHS solver option overrides."""
    presolve: dict[str, str]
    method: dict[str, str]
    parallel: dict[str, str]


@dataclass
class SolverSettings:
    """Solver selection and invocation settings."""
    solvers: dict[str, str]
    precommand: dict[str, str]
    arguments: defaultdict[str, list]


class SolveConfig:
    """All solve-level parameters and mutable tracking state for a FlexTool run.

    Attributes
    ----------
    model : list[str]
        Model entity names.
    model_solve : defaultdict[str, list[str]]
        model → list of solve names.
    solve_modes : dict[str, str]
        solve → solve_mode string.
    roll_counter : dict[str, int]
        solve → current rolling-window counter.
    rolling_times : defaultdict[str, list]
        solve → [jump, horizon, duration].
    highs : HiGHSConfig
        HiGHS solver option overrides (presolve, method, parallel).
    solver_settings : SolverSettings
        Solver selection and invocation settings (solvers, precommand, arguments).
    solve_period_years_represented : defaultdict[str, list]
        solve → [[period, years], …].
    hole_multipliers : defaultdict[str, str]
        solve → timeline_hole_multiplier value.
    timesets_used_by_solves : defaultdict[str, list[tuple]]
        solve → [(period, timeset), …].
    contains_solves : defaultdict[str, list[str]]
        solve → list of nested solve names.
    stochastic_branches : defaultdict[str, list]
        solve → stochastic branch definitions.
    invest_periods : defaultdict[str, list[tuple]]
        solve → [(period_from, period_included), …].
    realized_periods : defaultdict[str, list[tuple]]
    realized_invest_periods : defaultdict[str, list[tuple]]
    fix_storage_periods : defaultdict[str, list[tuple]]
    periods_available : dict[str, list]
        model → list of available period names.
    delay_durations : dict[str, float]
        unit → delay value.
    real_solves : list[str]
        Ordered list of solve names as executed (populated during solve loop).
    first_of_complete_solve : list[str]
        Populated during solve loop.
    last_of_solve : list[str]
        Populated during solve loop.
    """

    def __init__(
        self,
        model: list,
        model_solve: defaultdict,
        solve_modes: dict,
        rolling_times: defaultdict,
        highs: HiGHSConfig,
        solver_settings: SolverSettings,
        solve_period_years_represented: defaultdict,
        hole_multipliers: defaultdict,
        contains_solves: defaultdict,
        stochastic_branches: defaultdict,
        periods_available: dict,
        delay_durations: dict,
        logger: logging.Logger,
    ) -> None:
        # Base fields (read directly from DB in load_from_db)
        self.model = model
        self.model_solve = model_solve
        self.solve_modes = solve_modes
        self.rolling_times = rolling_times
        self.highs = highs
        self.solver_settings = solver_settings
        self.solve_period_years_represented = solve_period_years_represented
        self.hole_multipliers = hole_multipliers
        self.contains_solves = contains_solves
        self.stochastic_branches = stochastic_branches
        self.periods_available = periods_available
        self.delay_durations = delay_durations
        self.logger = logger

        # Computed fields — populated by load_from_db after construction
        self.roll_counter: dict[str, int] = {}
        self.timesets_used_by_solves: defaultdict = defaultdict(list)
        self.invest_periods: defaultdict = defaultdict(list)
        self.realized_periods: defaultdict = defaultdict(list)
        self.realized_invest_periods: defaultdict = defaultdict(list)
        self.fix_storage_periods: defaultdict = defaultdict(list)

        # Mutable tracking — populated during the solve loop
        self.real_solves: list[str] = []
        self.first_of_complete_solve: list[str] = []
        self.last_of_solve: list[str] = []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(cls, db: DatabaseMapping, logger: logging.Logger) -> SolveConfig:
        """Read all solve-level parameters from *db* and return a SolveConfig.

        Loading order is preserved exactly:
        1. Basic params (model, solvers, rolling_times, …)
        2. make_roll_counter (needs solve_modes)
        3. get_period_timesets (needs model_solve, contains_solves; may call duplicate_solve)
        4. periods_to_tuples x4 (needs timesets_used_by_solves; may call duplicate_solve)
        """
        model = get_single_entities(db=db, entity_class_name="model")
        model_solve: defaultdict = params_to_dict(
            db=db, cl="model", par="solves", mode=DictMode.DEFAULTDICT
        )
        # If no model:solves defined and only one solve exists, wire it up automatically
        solves_temp = get_single_entities(db=db, entity_class_name="solve")
        if len(model_solve) == 0 and len(solves_temp) == 1:
            model_solve["flextool"] = [solves_temp[0]]

        solve_modes: dict = params_to_dict(
            db=db, cl="solve", par="solve_mode", mode=DictMode.DICT
        )
        highs_presolve: dict = params_to_dict(
            db=db, cl="solve", par="highs_presolve", mode=DictMode.DICT
        )
        highs_method: dict = params_to_dict(
            db=db, cl="solve", par="highs_method", mode=DictMode.DICT
        )
        highs_parallel: dict = params_to_dict(
            db=db, cl="solve", par="highs_parallel", mode=DictMode.DICT
        )
        solve_period_years_represented: defaultdict = params_to_dict(
            db=db, cl="solve", par="years_represented", mode=DictMode.DEFAULTDICT
        )
        solvers: dict = params_to_dict(db=db, cl="solve", par="solver", mode=DictMode.DICT)
        solver_precommand: dict = params_to_dict(
            db=db, cl="solve", par="solver_precommand", mode=DictMode.DICT
        )
        solver_arguments: defaultdict = params_to_dict(
            db=db, cl="solve", par="solver_arguments", mode=DictMode.DEFAULTDICT
        )
        stochastic_branches: defaultdict = params_to_dict(
            db=db, cl="solve", par="stochastic_branches", mode=DictMode.DEFAULTDICT
        )
        contains_solves: defaultdict = params_to_dict(
            db=db, cl="solve", par="contains_solves", mode=DictMode.DEFAULTDICT, str_to_list=True
        )
        hole_multipliers: defaultdict = params_to_dict(
            db=db, cl="solve", par="timeline_hole_multiplier", mode=DictMode.DEFAULTDICT
        )
        delay_durations: dict = params_to_dict(
            db=db, cl="unit", par="delay", mode=DictMode.DICT
        )
        periods_available: dict = params_to_dict(
            db=db, cl="model", par="periods_available", mode=DictMode.DICT
        )

        # S02: Simplified rolling_times assembly (replaced the convoluted enumerate loop)
        rolling_duration: dict = params_to_dict(
            db=db, cl="solve", par="rolling_duration", mode=DictMode.DICT
        )
        rolling_solve_horizon: dict = params_to_dict(
            db=db, cl="solve", par="rolling_solve_horizon", mode=DictMode.DICT
        )
        rolling_solve_jump: dict = params_to_dict(
            db=db, cl="solve", par="rolling_solve_jump", mode=DictMode.DICT
        )
        all_keys = set(rolling_duration) | set(rolling_solve_horizon) | set(rolling_solve_jump)
        rolling_times: defaultdict = defaultdict(
            list,
            {
                key: [
                    rolling_solve_jump.get(key, 0),
                    rolling_solve_horizon.get(key, 0),
                    rolling_duration.get(key, -1),
                ]
                for key in all_keys
            },
        )

        highs = HiGHSConfig(
            presolve=highs_presolve,
            method=highs_method,
            parallel=highs_parallel,
        )
        solver_settings = SolverSettings(
            solvers=solvers,
            precommand=solver_precommand,
            arguments=solver_arguments,
        )

        obj = cls(
            model=model,
            model_solve=model_solve,
            solve_modes=solve_modes,
            rolling_times=rolling_times,
            highs=highs,
            solver_settings=solver_settings,
            solve_period_years_represented=solve_period_years_represented,
            hole_multipliers=hole_multipliers,
            contains_solves=contains_solves,
            stochastic_branches=stochastic_branches,
            periods_available=periods_available,
            delay_durations=delay_durations,
            logger=logger,
        )

        # Computed fields (loading order must be preserved exactly)
        obj.roll_counter = obj.make_roll_counter()
        obj.timesets_used_by_solves = obj.get_period_timesets(db=db)
        obj.invest_periods = obj.periods_to_tuples(db=db, cl="solve", par="invest_periods")
        obj.realized_periods = obj.periods_to_tuples(db=db, cl="solve", par="realized_periods")
        obj.realized_invest_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="realized_invest_periods"
        )
        obj.fix_storage_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="fix_storage_periods"
        )

        return obj

    # ------------------------------------------------------------------
    # Methods (moved from FlexToolRunner)
    # ------------------------------------------------------------------

    def make_roll_counter(self) -> dict[str, int]:
        """Return a roll counter initialised to 0 for every rolling-window solve."""
        roll_counter_map: dict[str, int] = {}
        for key, mode in list(self.solve_modes.items()):
            if mode == "rolling_window":
                roll_counter_map[key] = 0
        return roll_counter_map

    def get_period_timesets(self, db: DatabaseMapping) -> defaultdict:
        """Read period_timeset relationships from the database.

        May call duplicate_solve when a solve has a Map-valued period_timeset
        parameter (one solve becomes multiple sub-solves, one per map key).
        """
        entities = db.find_entities(entity_class_name="solve")
        params = db.find_parameter_values(
            entity_class_name="solve",
            parameter_definition_name="period_timeset",
        )
        timesets_used_by_solves: defaultdict = defaultdict(list)
        solves_in_model = [
            item
            for sublist in list(self.model_solve.values()) + list(self.contains_solves.values())
            for item in sublist
        ]
        for entity in entities:
            if entity["name"] in solves_in_model:
                for param in params:
                    if param["entity_name"] == entity["name"]:
                        param_value = api.from_database(param["value"], param["type"])
                        for i, _row in enumerate(param_value.indexes):
                            if isinstance(param_value.values[i], api.Map):
                                new_name = param["entity_name"] + "_" + param_value.indexes[i]
                                self.duplicate_solve(param["entity_name"], new_name)
                                timesets_used_by_solves[new_name].append(
                                    (
                                        param_value.values[i].indexes[i],
                                        param_value.values[i].values[i],
                                    )
                                )
                            else:
                                timesets_used_by_solves[param["entity_name"]].append(
                                    (param_value.indexes[i], param_value.values[i])
                                )
        return timesets_used_by_solves

    def duplicate_solve(
        self,
        old_solve: str,
        new_name: str,
        update_model_solves: bool = True,  # S11: renamed from first_level_flag
    ) -> None:
        """Duplicate all solve parameters from *old_solve* under *new_name*.

        When *update_model_solves* is True (the default), the original solve is
        replaced by the new name in model_solve, keeping the execution list
        consistent.  Pass False when called from the rolling-solver (where
        sub-solves should not replace their parent in model_solve).
        """
        if new_name not in self.model_solve.values() and new_name not in self.contains_solves.values():
            dup_map_list = [
                self.solve_modes,
                self.roll_counter,
                self.highs.presolve,
                self.highs.method,
                self.highs.parallel,
                self.solve_period_years_represented,
                self.solver_settings.solvers,
                self.solver_settings.precommand,
                self.solver_settings.arguments,
                self.contains_solves,
                self.rolling_times,
            ]
            for dup_map in dup_map_list:
                if old_solve in dup_map.keys():
                    dup_map[new_name] = dup_map[old_solve]
            if update_model_solves:
                for model, solves in list(self.model_solve.items()):
                    if old_solve in solves:
                        solves.remove(old_solve)
                    if new_name not in solves:
                        solves.append(new_name)
                    self.model_solve[model] = solves

    def periods_to_tuples(
        self, db: DatabaseMapping, cl: str, par: str
    ) -> defaultdict:
        """Read period parameters from the database and return as a dict of tuples.

        Returns
        -------
        defaultdict[str, list[tuple]]
            solve → [(period_from, period_included), …]
        """
        entities = db.find_entities(entity_class_name=cl)
        params = db.find_parameter_values(
            entity_class_name=cl,
            parameter_definition_name=par,
        )
        result_dict: defaultdict = defaultdict(list)
        for entity in entities:
            for param in params:
                if param["entity_name"] == entity["name"]:
                    param_value = api.from_database(param["value"], param["type"])
                    for i, row in enumerate(param_value.values):
                        if isinstance(param_value.values[i], api.Map):
                            for j, _row2 in enumerate(row.values):
                                if isinstance(param_value.values[j], api.Map):
                                    new_name = (
                                        param["entity_name"] + "_" + param_value.indexes[i]
                                    )
                                    self.duplicate_solve(param["entity_name"], new_name)
                                    result_dict[new_name].append(
                                        (
                                            param_value.indexes[i],
                                            param_value.values[i].indexes[j],
                                        )
                                    )
                                    new_period_timeset_list = []
                                    for solve, period__timeset_list in list(
                                        self.timesets_used_by_solves.items()
                                    ):
                                        if solve == param["entity_name"]:
                                            for period__timeset in period__timeset_list:
                                                if (
                                                    period__timeset[0]
                                                    == param_value.values[i].indexes[j]
                                                ):
                                                    new_period_timeset_list.append(period__timeset)
                                    if new_name not in self.timesets_used_by_solves.keys():
                                        self.timesets_used_by_solves[new_name] = (
                                            new_period_timeset_list
                                        )
                                    else:
                                        for item in new_period_timeset_list:
                                            if item not in self.timesets_used_by_solves[new_name]:
                                                self.timesets_used_by_solves[new_name].append(item)
                                else:
                                    raise ValueError(
                                        f"periods_to_tuple function handles only arrays or 2d maps:"
                                        f" {entity}, {param}"
                                    )
                        else:
                            result_dict[param["entity_name"]].append((row, row))
        return result_dict
