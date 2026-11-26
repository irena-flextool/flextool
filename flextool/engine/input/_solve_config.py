"""Solve-level configuration loaded from a SpineDB scenario.

Architecture notes
------------------

* Every downstream module reads ``state.solve.<dict>`` keys produced
  here; the keys must match the canonical names (including the
  ``np.str_`` leak through Spine ``Map.indexes``, the stringified-
  float ``rolling_times`` entries).
* The factory uses :class:`spinedb_api.DatabaseMapping` directly.
* Reads ``solve``, ``model``, ``unit`` parameter classes only.
  Every per-solve parameter is read under the database solve names; a
  2-D period map is split into solves of their own afterwards
  (:meth:`SolveConfig._split_2d_period_maps`).  A solve made from another
  one (a 2-D period map, a nested solve renamed for its period, the
  realization solve) gets every per-solve field of its parent through one
  step, :meth:`SolveConfig.derive_solve` (registry
  :data:`PER_SOLVE_FIELDS`), overriding or dropping only what its path
  lists.
* The DB schema is assumed to be v50+.  v50 moved
  ``new_stepduration`` from ``timeset`` to ``solve``;
  ``update_flextool/db_migration/`` handles upgrades for older DBs
  before this loader runs.
"""
from __future__ import annotations

import copy
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.sqlite_url import sqlite_url_string

#: The values of ``solve.reduce_unit_detail_method`` / ``solve.reduce_system_detail_method`` the engine accepts:
#: the v79 value lists ``reduce_unit_detail_methods`` / ``reduce_system_detail_methods`` (pinned against the schema
#: by tests/engine/test_reduce_detail.py).
REDUCE_UNIT_DETAIL_METHODS: tuple[str, ...] = (
    "none", "integers", "integers_ramps", "integers_min_limits", "integers_ramps_min_limits", "online",
    "online_ramps")
REDUCE_SYSTEM_DETAIL_METHODS: tuple[str, ...] = (
    "none", "reserves", "inertia", "non_synchronous", "inertia_non_synchronous",
    "reserves_inertia_non_synchronous", "dc_power_flow", "reserves_inertia_non_synchronous_dc_power_flow")

#: The members of ``solve.invest_solve_branching_method`` (value list ``invest_solve_branching_methods``, data version
#: 84), the default first: which decisions of an investment solve are shared across its forecasts.
INVEST_SOLVE_BRANCHING_METHODS: tuple[str, ...] = (
    "operations_only", "investments_after_reveal_period", "investments_at_reveal_period")

if TYPE_CHECKING:
    from spinedb_api import DatabaseMapping


# ---------------------------------------------------------------------------
# DB-reader helpers used by the solve-config factory.
# ---------------------------------------------------------------------------


class DictMode(Enum):
    """Output container shape for :func:`params_to_dict`."""

    DICT = "dict"
    DEFAULTDICT = "defaultdict"
    LIST = "list"


def get_single_entities(db: "DatabaseMapping", entity_class_name: str) -> list[str]:
    """Return entity names for a single-dimension entity class."""
    return [
        entity["entity_byname"][0]
        for entity in db.find_entities(entity_class_name=entity_class_name)
    ]


def params_to_dict(
    db: "DatabaseMapping",
    cl: str,
    par: str,
    mode: DictMode,
    str_to_list: bool = False,
) -> dict | defaultdict | list:
    """Read parameter values of *par* on entity class *cl*.

    Value-type dispatch:

    * Map of float → list of (index, float) tuples.
    * Map of str → list of (index, str) tuples.
    * Map of Map → :func:`spinedb_api.convert_map_to_table` flattening.
    * Array → ``param_value.values`` (the raw numpy-string-typed list).
    * Float scalar → :class:`str` of the float (preserved verbatim).
    * String scalar → either the bare string or ``[string]`` when
      *str_to_list* is set.
    """
    all_params = db.find_parameter_values(
        entity_class_name=cl, parameter_definition_name=par
    )
    result: dict | defaultdict | list
    if mode == DictMode.DEFAULTDICT:
        result = defaultdict(list)
    elif mode == DictMode.DICT:
        result = dict()
    elif mode == DictMode.LIST:
        result = []
    else:  # pragma: no cover — exhaustive enum
        raise ValueError(f"Unknown DictMode: {mode!r}")
    for param in all_params:
        param_value = api.from_database(param["value"], param["type"])
        if mode in (DictMode.DEFAULTDICT, DictMode.DICT):
            if isinstance(param_value, api.Map):
                if isinstance(param_value.values[0], float):
                    result[param["entity_name"]] = list(
                        zip(
                            list(param_value.indexes),
                            list(map(float, param_value.values)),
                        )
                    )
                elif isinstance(param_value.values[0], str):
                    result[param["entity_name"]] = list(
                        zip(list(param_value.indexes), param_value.values)
                    )
                elif isinstance(param_value.values[0], api.Map):
                    result[param["entity_name"]] = api.convert_map_to_table(
                        param_value
                    )
                else:
                    raise TypeError(
                        "params_to_dict function does not handle other "
                        "values than floats and strings"
                    )
            elif isinstance(param_value, api.Array):
                result[param["entity_name"]] = param_value.values
            elif isinstance(param_value, float):
                result[param["entity_name"]] = str(param_value)
            elif isinstance(param_value, str):
                if str_to_list:
                    result[param["entity_name"]] = [param_value]
                else:
                    result[param["entity_name"]] = param_value
        elif mode == DictMode.LIST:
            if isinstance(param_value, (float, str)):
                result.append([param["entity_name"], param_value])  # type: ignore[union-attr]
    return result


# ---------------------------------------------------------------------------
# Solver-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HiGHSConfig:
    """HiGHS solver option overrides — solve-level dispatch.

    Each field is keyed by solve name; values are the raw strings stored
    in Spine (e.g. ``"on"`` / ``"off"`` / ``"choose"``).  ``HiGHSProblem``
    converts them at solve time.
    """

    presolve: dict[str, str]
    method: dict[str, str]
    parallel: dict[str, str]


@dataclass
class SolverSettings:
    """Solver selection + invocation settings, keyed by solve name.

    *arguments* is the per-solve HiGHS option overrides Map authored
    on the ``solver_arguments`` parameter (1d-map of HiGHS option name
    → value).  Read into the effective-options resolver via
    :func:`flextool.engine.solve._solver_dispatch._resolve_effective_highs_options`
    where it is overlaid on top of ``solver_config/highs.opt`` and
    below the CLI overrides.  Empty when no solve authored an entry.

    *precommand* is ``solver_precommand`` per database solve, kept only to
    warn that it is not run (item 236 A3: FlexTool never runs a command stored
    in a database).
    """

    solvers: dict[str, str]
    precommand: dict[str, str]
    arguments: dict[str, dict[str, str]]


@dataclass
class SolverConfig:
    """Per-solve multi-solver dispatch configuration (v52 schema).

    Holds the solver-selection parameters introduced by the
    v52 migration (see ``specs/flextool-multi-solver-handoff.md``
    Step 1).  Defaults match the v52 parameter definition defaults so
    a solve that does not author any ``solver_*`` parameter still
    constructs a meaningful :class:`SolverConfig` (HiGHS via the
    direct in-process API, no convenience knobs set, "normal" log).

    Fields
    ------
    name
        Solver name; one of polar-high's ``available_solvers``
        ("highs", "gurobi", "cplex", "xpress", "copt").  Default
        ``"highs"``.
    options
        Free-form key→value dict forwarded raw to the solver.  Empty
        by default; user populates via the ``solver_arguments``
        1d-map parameter (Batch C.2: the legacy ``solver_options``
        Map was folded into ``solver_arguments`` and removed).
    mip_gap
        Relative MIP gap (``solve.solver_mip_gap``); ``None`` means
        solver default.  Translated to each solver's native parameter
        name by
        :func:`flextool.engine.solve._solver_dispatch.resolve_commercial_options`
        and ``build_solver_options``.  A time limit or thread count comes
        from ``solver_arguments`` (the solver's own name) or the run's
        flags (``--solver-time-limit``, ``--threads``); the database's
        ``solver_time_limit`` / ``solver_threads`` were removed in v56.
        The solver's log level comes from the run's ``--solver-log-level``
        flag; the database's ``solver_log_level`` was removed in v56.
    """

    name: str = "highs"
    options: dict[str, Any] = field(default_factory=dict)
    mip_gap: float | None = None


# ---------------------------------------------------------------------------
# Solves made from another solve: the per-solve field registry
# ---------------------------------------------------------------------------


class _NoEntry:
    """The type of :data:`NO_ENTRY`."""

    def __repr__(self) -> str:
        return "NO_ENTRY"


#: In the ``overrides`` of :meth:`SolveConfig.derive_solve`: the new solve gets no entry in that field.
NO_ENTRY = _NoEntry()

#: Every :class:`SolveConfig` attribute keyed by solve name.  :meth:`SolveConfig.derive_solve` gives a solve made
#: from another one (a solve of a 2-D period map, a nested solve renamed for its period, the realization solve) a
#: deep copy of the parent's entry in each of them, unless the path overrides or drops the field.  A field added
#: later is copied by default; an attribute keyed by solve name that is in none of the three lists below stops
#: the run (``_check_field_registry``).  Dotted names reach into the solver dataclasses.
PER_SOLVE_FIELDS: tuple[str, ...] = (
    "solve_modes",
    "roll_counter",
    "rolling_times",
    "contains_solves",
    "timesets_used_by_solves",
    "invest_periods",
    "realized_periods",
    "realized_invest_periods",
    "fix_storage_periods",
    "solve_period_years_represented",
    "stochastic_branches",
    "invest_solve_branching_method",
    "dispatch_realization_method",
    "hole_multipliers",
    "decomposition",
    "benders_max_iter",
    "benders_tolerance",
    "benders_in_out_weight",
    "scaling",
    "reduce_unit_detail_method",
    "reduce_system_detail_method",
    "use_row_scaling",
    "scale_the_objective",
    "solver_configs",
    "solver_settings.solvers",
    "solver_settings.arguments",
    "highs.presolve",
    "highs.method",
    "highs.parallel",
)

#: Attributes keyed by solve name that a solve made from another one does not get, with the reason.
NOT_COPIED_FIELDS: dict[str, str] = {
    "solver_settings.precommand": "never run (item 236); FlexTool names it once per database solve in a warning",
    "deterministic_branches": "set after the solves are made (apply_deterministic_realization); a solve made from "
                              "a deterministic solve is never stochastic",
}

#: Dict attributes that are not keyed by solve name, with what they are keyed by.
OTHER_DICT_FIELDS: dict[str, str] = {
    "model_solve": "model",
    "periods_available": "model",
    "delay_durations": "unit",
    "map2d_period_parents": "period parameter (the solves split by a 2-D map of it)",
    "map2d_duplicate_of": "solve of a 2-D period map (set per copy: its database solve and outer key)",
    "realized_dispatch_parent": "realization solve (set per copy: its parent)",
    "roll_tail_plan": "sub-solve (a realization roll; set per run by native_run_model)",
    "roll_branch_rows": "sub-solve (a realization roll; set per run by native_run_model)",
    "derived_solves": "solve made from another solve (its parent and the fields it does not copy)",
    "_split_records": "database solve with a 2-D period map (read, not yet split)",
}

#: The period parameters a 2-D map splits into solves of their own, with their :class:`SolveConfig` field.
SPLIT_PERIOD_PARAMS: dict[str, str] = {
    "invest_periods": "invest_periods",
    "realized_periods": "realized_periods",
    "realized_invest_periods": "realized_invest_periods",
    "fix_storage_periods": "fix_storage_periods",
}

#: The fields the engine-automatic realization solve ``<solve>__realized_dispatch`` does not take from its parent:
#: it is a deterministic, fixed-capacity, monolithic dispatch (no investment, no forecasts, no decomposition, no
#: contained solves, no realization solve of its own).  The reader half is
#: ``SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS`` (in schema names).
REALIZATION_SOLVE_DROPS: frozenset[str] = frozenset((
    "invest_periods",
    "realized_invest_periods",
    "fix_storage_periods",
    "stochastic_branches",
    "invest_solve_branching_method",
    "decomposition",
    "benders_max_iter",
    "benders_tolerance",
    "benders_in_out_weight",
    "contains_solves",
    "dispatch_realization_method",
))


def _per_solve_field(cfg: object, name: str) -> "dict | None":
    """The dict of field *name* (dotted: through a dataclass) of *cfg*; ``None`` when *cfg* lacks it (a test's
    partly built config)."""
    obj = cfg
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, dict) else None


_ABSENT = object()


def _entry_of(values: dict, key: str) -> Any:
    """The entry of *key* in *values*; a missing key of a ``defaultdict`` is its empty default (a read through
    ``[]`` creates it), of a plain dict a marker of its own (an absent entry stays absent)."""
    if key in values:
        return values[key]
    factory = getattr(values, "default_factory", None)
    return factory() if factory is not None else _ABSENT


def _check_field_registry(cfg: object) -> None:
    """Raise when *cfg* has a dict attribute (also inside its solver dataclasses) that is in none of
    :data:`PER_SOLVE_FIELDS`, :data:`NOT_COPIED_FIELDS` and :data:`OTHER_DICT_FIELDS`: a per-solve field that a
    solve made from another one would silently lose."""
    known = set(PER_SOLVE_FIELDS) | set(NOT_COPIED_FIELDS) | set(OTHER_DICT_FIELDS)
    found: list[str] = []
    for attr, value in vars(cfg).items():
        if isinstance(value, dict):
            found.append(attr)
        elif attr in ("highs", "solver_settings") and value is not None:
            found += [f"{attr}.{sub}" for sub, inner in vars(value).items() if isinstance(inner, dict)]
    unknown = sorted(set(found) - known)
    if unknown:
        raise RuntimeError(
            f"SolveConfig attribute(s) {unknown} are not classified in _solve_config.PER_SOLVE_FIELDS / "
            "NOT_COPIED_FIELDS / OTHER_DICT_FIELDS, so a solve made from another one could lose them. This is a "
            "FlexTool bug; please report it.")


# ---------------------------------------------------------------------------
# SolveConfig — main container
# ---------------------------------------------------------------------------


def solve_decides_investments(solve_cfg, solve: str) -> bool:
    """Whether *solve* authors investment decisions: ``invest_periods`` or ``realized_invest_periods`` non-empty
    (the test of the rolling-invest warning in ``_recursive_solve``)."""
    if (getattr(solve_cfg, "invest_periods", None) or {}).get(solve):
        return True
    return bool((getattr(solve_cfg, "realized_invest_periods", None) or {}).get(solve))


def is_investment_solve(solve_cfg, solve: str) -> bool:
    """Whether *solve* is an investment solve: it decides investments (:func:`solve_decides_investments`) and is not
    rolling (the rolls of a rolling solve build no investment).  A solve of a 2-D period map is one when its own
    rows are.  Every other solve is a dispatch solve."""
    return (solve_decides_investments(solve_cfg, solve)
            and (getattr(solve_cfg, "solve_modes", None) or {}).get(solve) != "rolling_window")


class SolveConfig:
    """All solve-level parameters and mutable tracking state.

    See ``audit/solve_orchestration_plan.md §1.3`` for the per-field
    contract; downstream modules read these dicts directly so the
    container shape (``defaultdict(list)`` vs plain ``dict``), the
    keys, and the value types must match flextool exactly.
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
        use_row_scaling: dict | None = None,
        scale_the_objective: dict | None = None,
        solver_configs: dict[str, "SolverConfig"] | None = None,
        decomposition: dict | None = None,
        benders_max_iter: dict | None = None,
        benders_tolerance: dict | None = None,
        benders_in_out_weight: dict | None = None,
        scaling: dict | None = None,
        invest_solve_branching_method: dict | None = None,
        dispatch_realization_method: dict | None = None,
        reduce_unit_detail_method: dict | None = None,
        reduce_system_detail_method: dict | None = None,
    ) -> None:
        # Base fields (read directly from DB in load_from_db).
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
        # solve-name → "yes"/"no" string from the DB (default off everywhere).
        self.use_row_scaling: dict = (
            use_row_scaling if use_row_scaling is not None else {}
        )
        self.scale_the_objective: dict = (
            scale_the_objective if scale_the_objective is not None else {}
        )
        # v52 multi-solver dispatch — solve-name → :class:`SolverConfig`.
        # Empty when no ``solver_*`` parameters are authored on any solve;
        # callers fall back to ``SolverConfig()`` defaults (HiGHS/direct).
        self.solver_configs: dict[str, SolverConfig] = (
            solver_configs if solver_configs is not None else {}
        )

        # v60/v62 per-solve decomposition.  ``decomposition`` maps
        # solve-name → "none"/"benders"; absent means the schema
        # default "none" (monolithic).  The two ``benders_*`` dicts carry
        # the per-solve Benders knobs (str(float) values, only present
        # when authored); absence falls back to the schema defaults via
        # :meth:`benders_config_for`.
        self.decomposition: dict = (
            decomposition if decomposition is not None else {}
        )
        self.benders_max_iter: dict = (
            benders_max_iter if benders_max_iter is not None else {}
        )
        self.benders_tolerance: dict = (
            benders_tolerance if benders_tolerance is not None else {}
        )
        self.benders_in_out_weight: dict = (
            benders_in_out_weight if benders_in_out_weight is not None else {}
        )
        # v64 per-solve autoscaler mode (solve-name -> "off"/"solver_only"/
        # "basic"/"full"); only authored solves appear.  Resolved at access
        # time via :meth:`scaling_for` (absent -> None -> CLI/env/default).
        self.scaling: dict = (
            scaling if scaling is not None else {}
        )

        # v84 per-solve branching of an investment solve (solve-name -> a member of
        # :data:`INVEST_SOLVE_BRANCHING_METHODS`); only authored solves appear.  Resolved at access time via
        # :meth:`invest_branching_method_for` (absent -> "operations_only").
        self.invest_solve_branching_method: dict = (
            invest_solve_branching_method if invest_solve_branching_method is not None else {}
        )

        # v75 per-solve dispatch-realization method (solve-name ->
        # "within_solve"/"separate_dispatch"/"deterministic"); only authored
        # solves appear.  Resolved at access time via
        # :meth:`dispatch_realization_method_for` (absent -> "deterministic").
        self.dispatch_realization_method: dict = (
            dispatch_realization_method
            if dispatch_realization_method is not None else {}
        )
        # The ``stochastic_branches`` rows of the solves that run
        # ``deterministic``, moved aside by :meth:`apply_deterministic_realization`:
        # such a solve reads its ``realized: yes`` branch only.
        self.deterministic_branches: dict[str, list] = {}

        # v79 per-solve detail reductions (solve-name -> value-list string); only authored solves appear.
        # Resolved at access time via :meth:`reduce_unit_detail_for` / :meth:`reduce_system_detail_for`.
        self.reduce_unit_detail_method: dict = (
            reduce_unit_detail_method if reduce_unit_detail_method is not None else {}
        )
        self.reduce_system_detail_method: dict = (
            reduce_system_detail_method if reduce_system_detail_method is not None else {}
        )

        # Computed fields — populated by load_from_db after construction.
        self.roll_counter: dict[str, int] = {}
        self.timesets_used_by_solves: defaultdict = defaultdict(list)
        self.invest_periods: defaultdict = defaultdict(list)
        self.realized_periods: defaultdict = defaultdict(list)
        self.realized_invest_periods: defaultdict = defaultdict(list)
        self.fix_storage_periods: defaultdict = defaultdict(list)
        # Solve entities whose value of a period parameter is a 2D map
        # (``periods_to_tuples`` fans them out into ``<solve>_<outer>``
        # duplicates), keyed by parameter name.  Read by
        # ``_check_realized_invest_periods``.
        self.map2d_period_parents: defaultdict = defaultdict(set)
        # ``{<solve>_<outer>: (solve, outer)}`` of those duplicates, so an
        # error can name the database solve and its 2D-map entry.
        self.map2d_duplicate_of: dict[str, tuple[str, str]] = {}
        # ``{solve: (parent, fields)}`` of every solve made from another one (:meth:`derive_solve`): its parent
        # and the fields it does not copy from it (overridden or dropped); read by :meth:`check_derived_solves`.
        self.derived_solves: dict[str, tuple[str, frozenset[str]]] = {}
        # ``{database solve: {parameter: {outer key: [period, ...]}}}`` of the 2-D period maps read by
        # :meth:`periods_to_tuples`, split by :meth:`_split_2d_period_maps` once every parameter is read.
        self._split_records: dict[str, dict[str, dict[str, list]]] = {}
        # The ``solve`` entities of the database (load_from_db): a split solve's name must not be one of them.
        self.database_solves: list[str] = []

        # Realization registry: maps a realization solve to the stochastic
        # solve whose committed window it realizes.  Populated by
        # :meth:`spawn_realized_dispatch_sibling` (engine-spawned
        # ``<solve>__realized_dispatch``) and :meth:`register_realization_solve`
        # (an author-declared deterministic solve right after its parent, see
        # ``solve._realization.configured_realization_solve``).  Consumers:
        # roll interleaving, the PIN translation, the CarryRouter siblings and
        # the realized-branch projection in ``_run_model``, the fix_online
        # producer and the silent last-wins pair.  The realization solve is
        # deterministic (no ``stochastic_branches``), so
        # ``create_stochastic_periods`` leaves its
        # ``solve_branch__time_branch`` scaffolding empty; the projection post-pass
        # reuses THIS explicit map (never name-parsing the suffix) to
        # re-populate it from the parent's realized-branch rows.  Empty when
        # no stochastic solve has a realization solve (the consumers then
        # no-op, preserving byte-parity on deterministic fixtures).
        self.realized_dispatch_parent: dict[str, str] = {}

        # The realization roll R_k of each roll of a rolling within_solve / separate_dispatch solve
        # (``solve._realization_tail``), keyed by sub-solve: its plan (``RollTail``) and the ``stochastic_branches``
        # rows it reads (a list in database order).  Set by ``native_run_model`` before the stochastic expansion;
        # empty without realization rolls (the readers then take the complete solve's rows).
        self.roll_tail_plan: dict[str, Any] = {}
        self.roll_branch_rows: dict[str, list] = {}

        # User constraints with constraint_period_method period_total (load_from_db); the rolls that leave their
        # rows out and the solves that write them (RecursiveSolveBuilder._check_period_total_rolls).
        self.period_total_constraints: list[str] = []
        self.period_total_off_solves: set[str] = set()
        self.period_total_enforcing_solves: set[str] = set()

        # Mutable tracking — populated during the recursive solve loop.
        self.real_solves: list[str] = []
        self.first_of_complete_solve: list[str] = []
        self.last_of_solve: list[str] = []

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(
        cls, db: "DatabaseMapping", logger: logging.Logger,
    ) -> "SolveConfig":
        """Read all solve-level parameters from *db* into a SolveConfig.

        1. Basic params (model, solvers, rolling_times, …)
        2. ``make_roll_counter`` (needs ``solve_modes``).
        3. ``get_period_timesets`` (needs ``model_solve`` +
           ``contains_solves``).
        4. The period lists (``periods_to_tuples``; a 2-D map is only
           recorded), the shared periods.
        5. ``_split_2d_period_maps``: the solves of every 2-D period map,
           each with every per-solve parameter of its database solve.
        """
        model = get_single_entities(db=db, entity_class_name="model")
        model_solve: defaultdict = params_to_dict(
            db=db, cl="model", par="solves", mode=DictMode.DEFAULTDICT
        )
        # Auto-wire when no model:solves defined and only one solve exists.
        solves_temp = get_single_entities(db=db, entity_class_name="solve")
        if len(model_solve) == 0 and len(solves_temp) == 1:
            model_solve["flextool"] = [solves_temp[0]]

        solve_modes: dict = params_to_dict(
            db=db, cl="solve", par="solve_mode", mode=DictMode.DICT
        )
        # Batch C.5 — ``highs_presolve`` shortcut removed; override
        # is now keyed as ``presolve`` inside ``solver_arguments``.
        highs_presolve: dict = {}
        # Batch C.3 — ``highs_method`` shortcut removed; the equivalent
        # override is now keyed as ``solver`` inside
        # ``solver_arguments``.
        highs_method: dict = {}
        # Batch C.4 — ``highs_parallel`` shortcut removed; override
        # is now keyed as ``parallel`` inside ``solver_arguments``.
        highs_parallel: dict = {}
        solve_period_years_represented: defaultdict = params_to_dict(
            db=db,
            cl="solve",
            par="years_represented",
            mode=DictMode.DEFAULTDICT,
        )
        solvers: dict = params_to_dict(
            db=db, cl="solve", par="solver", mode=DictMode.DICT
        )
        solver_precommand: dict = params_to_dict(
            db=db, cl="solve", par="solver_precommand", mode=DictMode.DICT
        )
        # Batch C.1 — ``solver_arguments`` retyped from array to 1d-map
        # at v56 schema; read directly via the Spine API the same way
        # ``solver_options`` is read below so we get a dict-of-dicts
        # (solve name → option-key → value) ready for the
        # effective-options resolver in ``_solver_dispatch``.
        solver_arguments: dict[str, dict[str, str]] = {}
        for param in db.find_parameter_values(
            entity_class_name="solve", parameter_definition_name="solver_arguments"
        ):
            pv = api.from_database(param["value"], param["type"])
            if isinstance(pv, api.Map):
                solver_arguments[param["entity_name"]] = {
                    str(k): str(v)
                    for k, v in zip(list(pv.indexes), list(pv.values))
                }
            elif pv is None:
                continue
            else:
                logger.warning(
                    "solve.%s.solver_arguments is not a 1d-map (%r) — "
                    "ignoring; expected a Map of HiGHS option name -> value",
                    param["entity_name"],
                    type(pv).__name__,
                )
        # v52 multi-solver dispatch params.  Each per-solve value is
        # rolled up into one :class:`SolverConfig` keyed by solve name
        # (see ``specs/flextool-multi-solver-handoff.md`` Steps 1-3).
        # The params are read individually here and aggregated
        # into ``solver_configs`` below.  ``solver_options`` is a Map of
        # str→Any forwarded raw to the chosen solver; everything else is
        # a scalar default-None convenience knob (None means "no override").
        #
        # NOTE: ``solver`` is also read via the older ``solvers``
        # variable above for the legacy :class:`SolverSettings` dataclass
        # (which downstream callers still consume); we reuse that dict
        # here rather than re-querying.
        # Batch C.7 (v56) dropped the ``solver_log_level`` DB parameter:
        # the --solver-log-level CLI flag sets it (silent / normal /
        # verbose).
        # solver_mip_gap comes back as the stringified float that
        # ``params_to_dict`` produces for scalar floats (see line
        # ~141).  Batches C.6 and C.8 (v56) dropped the ``solver_threads``
        # and ``solver_time_limit`` DB parameters: a time limit or thread
        # count goes in ``solver_arguments`` (the solver's own name) or the
        # --solver-time-limit / --threads flags.
        solver_mip_gap_raw: dict = params_to_dict(
            db=db, cl="solve", par="solver_mip_gap", mode=DictMode.DICT
        )
        # Batch C.2 — ``solver_options`` was folded into
        # ``solver_arguments`` (now the canonical 1d-map) and the
        # parameter_definition was removed.  ``SolverConfig.options``
        # is sourced from ``solver_arguments`` below.
        stochastic_branches: defaultdict = params_to_dict(
            db=db,
            cl="solve",
            par="stochastic_branches",
            mode=DictMode.DEFAULTDICT,
        )
        contains_solves: defaultdict = params_to_dict(
            db=db,
            cl="solve",
            par="contains_solves",
            mode=DictMode.DEFAULTDICT,
            str_to_list=True,
        )
        hole_multipliers: defaultdict = params_to_dict(
            db=db,
            cl="solve",
            par="timeline_hole_multiplier",
            mode=DictMode.DEFAULTDICT,
        )
        delay_durations: dict = params_to_dict(
            db=db, cl="unit", par="delay", mode=DictMode.DICT
        )
        periods_available: dict = params_to_dict(
            db=db, cl="model", par="periods_available", mode=DictMode.DICT
        )
        # Per-solve opt-in for automatic LP-row scaling.  Default
        # absent / "no" leaves AMPL behaviour as pre-Agent-5; the
        # native engine consumes this flag during preprocessing the
        # same way.
        # Batch C.10 — DB-level ``use_row_scaling`` removed; use the
        # --scaling CLI flag (autoscale; off/solver_only/basic/full).
        # The per-solve dict is hard-wired to {} so every solve
        # emits p_use_row_scaling=0 (the
        # ``use_row_scaling.get(solve, "no")`` default branch in
        # _solve_writers.derive_p_use_row_scaling), preserving
        # the Mode A pre-scaling behaviour for the row-scaling
        # capacity-proxy emitter.  The autoscaler's Layer 2 + Layer
        # 3 (driven by --scaling) are unaffected.
        use_row_scaling: dict = {}
        # ``scale_the_objective`` is not a schema parameter (the objective scale is fixed); the per-solve dict
        # stays empty, like ``use_row_scaling``.
        scale_the_objective: dict = {}

        # v60/v62 per-solve decomposition scheme + Benders knobs.  Only
        # solves that explicitly author the parameter appear in each
        # dict; absent solves fall back to the schema defaults
        # (decomposition "none"; max_iter 50 / tol 1e-3) at access time
        # via ``decomposition_for`` / ``benders_config_for``.
        decomposition: dict = params_to_dict(
            db=db, cl="solve", par="decomposition", mode=DictMode.DICT
        )
        benders_max_iter: dict = params_to_dict(
            db=db, cl="solve", par="benders_max_iter", mode=DictMode.DICT
        )
        benders_tolerance: dict = params_to_dict(
            db=db, cl="solve", par="benders_tolerance", mode=DictMode.DICT
        )
        benders_in_out_weight: dict = params_to_dict(
            db=db, cl="solve", par="benders_in_out_weight", mode=DictMode.DICT
        )

        # v64 per-solve autoscaler mode.  Only solves that explicitly
        # author solve.scaling appear; absent solves resolve to None via
        # ``scaling_for`` (caller falls back to CLI/env/default "full").
        scaling: dict = params_to_dict(
            db=db, cl="solve", par="scaling", mode=DictMode.DICT
        )

        # v84 per-solve branching of an investment solve.  Only solves that author
        # solve.invest_solve_branching_method appear; absent solves resolve to "operations_only" via
        # ``invest_branching_method_for``.
        invest_solve_branching_method: dict = params_to_dict(
            db=db, cl="solve", par="invest_solve_branching_method", mode=DictMode.DICT,
        )

        # v75 per-solve dispatch-realization method.  Only solves that
        # explicitly author solve.dispatch_realization_method appear; absent
        # solves resolve to "within_solve" via
        # ``dispatch_realization_method_for``.
        dispatch_realization_method: dict = params_to_dict(
            db=db, cl="solve", par="dispatch_realization_method",
            mode=DictMode.DICT,
        )

        # v79 per-solve detail reductions.  Only solves that author them appear; absent solves resolve to
        # "none" via ``reduce_unit_detail_for`` / ``reduce_system_detail_for``.
        reduce_unit_detail_method: dict = params_to_dict(
            db=db, cl="solve", par="reduce_unit_detail_method", mode=DictMode.DICT,
        )
        reduce_system_detail_method: dict = params_to_dict(
            db=db, cl="solve", par="reduce_system_detail_method", mode=DictMode.DICT,
        )

        # rolling_times: assemble per-solve [jump, horizon, duration].
        rolling_duration: dict = params_to_dict(
            db=db, cl="solve", par="rolling_duration", mode=DictMode.DICT
        )
        rolling_solve_horizon: dict = params_to_dict(
            db=db, cl="solve", par="rolling_solve_horizon", mode=DictMode.DICT
        )
        rolling_solve_jump: dict = params_to_dict(
            db=db, cl="solve", par="rolling_solve_jump", mode=DictMode.DICT
        )
        all_keys = (
            set(rolling_duration)
            | set(rolling_solve_horizon)
            | set(rolling_solve_jump)
        )
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

        # Roll the v52 per-solve param dicts into a single
        # ``solver_configs[solve_name] -> SolverConfig`` mapping.  Keys
        # are the union of solve names appearing across any of the
        # dicts — a solve that authors *any* solver_* param gets
        # an explicit entry; solves with no override remain absent and
        # callers fall back to ``SolverConfig()`` defaults.
        # Batch C.2 — ``solver_options`` removed; the per-solve free-form
        # option dict for the commercial-solver path is sourced from
        # ``solver_arguments`` (the same 1d-map the HiGHS-side resolver
        # consumes).
        solver_config_keys = (
            set(solvers)
            | set(solver_arguments)
            | set(solver_mip_gap_raw)
        )

        def _opt_float(raw: dict, key: str) -> float | None:
            v = raw.get(key)
            return float(v) if v is not None else None

        solver_configs: dict[str, SolverConfig] = {}
        for key in solver_config_keys:
            solver_configs[key] = SolverConfig(
                name=solvers.get(key, "highs"),
                options=dict(solver_arguments.get(key, {})),
                mip_gap=_opt_float(solver_mip_gap_raw, key),
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
            use_row_scaling=use_row_scaling,
            scale_the_objective=scale_the_objective,
            solver_configs=solver_configs,
            decomposition=decomposition,
            benders_max_iter=benders_max_iter,
            benders_tolerance=benders_tolerance,
            benders_in_out_weight=benders_in_out_weight,
            scaling=scaling,
            invest_solve_branching_method=invest_solve_branching_method,
            dispatch_realization_method=dispatch_realization_method,
            reduce_unit_detail_method=reduce_unit_detail_method,
            reduce_system_detail_method=reduce_system_detail_method,
        )

        # Computed fields.  Every per-solve parameter is read under the database solve names first; a 2-D
        # period map is only recorded (``_split_records``) and split by ``_split_2d_period_maps`` after all of
        # them, so each solve it stands for gets every parameter of its database solve, whatever the order of
        # the reads.
        obj.database_solves = list(solves_temp)
        obj.roll_counter = obj.make_roll_counter()
        obj.timesets_used_by_solves = obj.get_period_timesets(db=db)
        for par, field_name in SPLIT_PERIOD_PARAMS.items():
            setattr(obj, field_name, obj.periods_to_tuples(db=db, cl="solve", par=par, split=True))
        period_method: dict = params_to_dict(
            db=db, cl="constraint", par="constraint_period_method", mode=DictMode.DICT
        )
        disabled = {name for name, value in params_to_dict(
            db=db, cl="constraint", par="is_enabled", mode=DictMode.DICT
        ).items() if value == "no"}
        # Only a constraint with a sense has rows (flex_data builds cdt_* from the sense); one without would
        # trigger a spurious rolling warning or the "no solve enforces" error.
        with_sense = {name for name, value in params_to_dict(
            db=db, cl="constraint", par="sense", mode=DictMode.DICT
        ).items() if value in ("equal", "less_than", "greater_than")}
        obj.period_total_constraints = sorted(
            name for name, value in period_method.items()
            if value == "period_total" and name not in disabled and name in with_sense
        )
        # Before the 2-D map split, so the message names the solve of the database.
        obj._check_branches_named_like_their_fan_period()
        obj._split_2d_period_maps()
        # After every period dict, so the 2D-map duplicate solves exist.
        obj._check_realized_invest_periods()
        obj._check_split_realized_periods()
        obj.apply_deterministic_realization()
        obj.check_invest_branching()
        obj.check_derived_solves()

        return obj

    @classmethod
    def load_from_db_url(
        cls,
        db_url: str,
        scenario: str,
        logger: logging.Logger | None = None,
    ) -> "SolveConfig":
        """Convenience factory: open *db_url*, apply *scenario*, load.

        Builds a short-lived :class:`spinedb_api.DatabaseMapping` with the
        scenario filter applied, calls :meth:`load_from_db`, and closes
        the DB.  Useful for tests / CLI entry points that don't already
        own a session.
        """
        from flextool.update_flextool.db_handle import open_database_mapping
        from spinedb_api.filters.scenario_filter import (
            apply_scenario_filter_to_subqueries,
        )

        if logger is None:
            logger = logging.getLogger(f"flextool.engine.solve_config[{scenario}]")
        url = str(db_url)
        if "://" not in url:
            url = sqlite_url_string(url)
        with open_database_mapping(url) as db:
            apply_scenario_filter_to_subqueries(db, scenario)
            # Pre-warm the entity + parameter_value caches so the
            # ``find_entities`` / ``find_parameter_values`` calls inside
            # ``load_from_db`` (get_period_timesets, get_single_param, …)
            # hit memory rather than running a SQL round-trip each.
            # Pre-fetch at construction and share the DB context across
            # the whole pipeline.  Measured 5.5-6.4× speedup on large
            # customer DBs (H2_trade ≈ 13 MB, 2.5 s → 0.35 s).
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            return cls.load_from_db(db, logger)

    @classmethod
    def load_from_source(
        cls,
        source: object,
        logger: logging.Logger | None = None,
    ) -> "SolveConfig":
        """Load via the :class:`InputSource` Protocol.

        Currently only :class:`flextool.engine.input._spinedb_reader.SpineDbReader`
        sources are supported — they expose the underlying ``db_url`` and
        ``scenario`` so the canonical :meth:`load_from_db_url` path can
        be reused.  In-memory and CSV-backed sources for solve-class
        parameters are out of Γ.8.A scope; Γ.8.D wires those once the
        chain.run_chain integration needs them.
        """
        # Late import: SpineDbReader brings spinedb_api into the import
        # graph, but we only need it for the isinstance check.
        from flextool.engine.input._spinedb_reader import SpineDbReader

        if isinstance(source, SpineDbReader):
            return cls.load_from_db_url(
                source.db_url, source.scenario, logger=logger
            )
        raise NotImplementedError(
            f"SolveConfig.load_from_source does not yet support "
            f"{type(source).__name__!r} sources.  Use load_from_db / "
            f"load_from_db_url with a Spine DB for now; the in-memory and "
            f"CSV adapters land in Gamma.8.D when chain.run_chain is rewired."
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def _check_branches_named_like_their_fan_period(self) -> None:
        """Raise when a ``stochastic_branches`` branch has the name of the period where it fans out.

        Owner decision 2026-11-15: such a branch would not be a member of the fan (it gets no periods) while the
        fan check still counts it, so its weight would silently disappear.  Every row is checked (forecast,
        zero-weight or realized), over the solves the model runs (:meth:`solves_in_run`).  A branch named like
        some other period of the run only warns (``StochasticSolver._warn_branches_named_like_periods``).
        """
        problems: list[str] = []
        for s in self.solves_in_run():
            bad = list(dict.fromkeys(
                (str(row[1]), str(row[0])) for row in (self.stochastic_branches.get(s) or [])
                if str(row[1]) == str(row[0])
            ))
            for branch, period in bad:
                problems.append(
                    f"solve '{s}': stochastic_branches has a branch '{branch}' in period '{period}', the "
                    "period where the branch fans out. A branch cannot have the name of the period where it fans "
                    f"out: rename the branch '{branch}' (for example to a name that is not a period name)"
                )
        if problems:
            raise FlexToolConfigError("; ".join(problems))

    def _check_realized_invest_periods(self) -> None:
        """Raise when a solve realizes investments in a period it does not invest in (#319).

        A realized investment period must be one of the solve's
        ``invest_periods`` (the inner period ``p_in`` of each
        ``(p_from, p_in)`` tuple is compared).  Checked over the solves
        that run (``model_solve`` and ``contains_solves`` children, with
        the 2D-map duplicates already made); rolls are made later from a
        checked complete solve.  A 1D ``realized_invest_periods`` left on
        a solve whose ``invest_periods`` is a 2D map never runs (the
        duplicates replace it), so it gets its own message.
        """
        inv_2d = self.map2d_period_parents.get("invest_periods", set())
        rinv_2d = self.map2d_period_parents.get("realized_invest_periods", set())
        for parent in sorted(inv_2d - rinv_2d):
            if self.realized_invest_periods.get(parent):
                raise FlexToolConfigError(
                    f"solve '{parent}': invest_periods is a 2D map, so "
                    "realized_invest_periods must be a 2D map with the same "
                    "solve names (outer keys)"
                )
        running: list[str] = []
        for solves in (
            list(self.model_solve.values())
            + list(self.contains_solves.values())
        ):
            for s in solves:
                if s not in running:
                    running.append(s)
        problems: list[str] = []
        for s in running:
            realized = [t[1] for t in self.realized_invest_periods.get(s, [])]
            if not realized:
                continue
            invest = [t[1] for t in self.invest_periods.get(s, [])]
            # Plain ``str`` for the messages: numpy 2 shows a ``np.str_`` as ``np.str_('p1')`` in a list.
            bad = [str(p) for p in dict.fromkeys(realized) if p not in set(invest)]
            if not bad:
                continue
            if s in self.map2d_duplicate_of:
                parent, outer = self.map2d_duplicate_of[s]
                where = f"solve '{parent}' (2D-map entry '{outer}')"
            else:
                where = f"solve '{s}'"
            if not invest:
                problems.append(
                    f"{where} has realized_invest_periods {bad} but no "
                    "invest_periods: remove realized_invest_periods (or set "
                    "invest_periods if this solve should invest)"
                )
            else:
                problems.append(
                    f"{where}: realized_invest_periods lists period(s) "
                    f"{bad} that are not in its invest_periods "
                    f"{[str(p) for p in dict.fromkeys(invest)]}. A realized investment "
                    "period must also be an investment period: add it to "
                    "invest_periods, or remove it from realized_invest_periods"
                )
        if problems:
            raise FlexToolConfigError("; ".join(problems))

    def make_roll_counter(self) -> dict[str, int]:
        """Return a roll counter initialised to 0 for every rolling-window
        solve.  Single-solve mode entries are intentionally absent from
        the returned dict (NOT zero) so callers can distinguish ``not in``
        from ``== 0`` semantics.
        """
        roll_counter_map: dict[str, int] = {}
        for key, mode in list(self.solve_modes.items()):
            if mode == "rolling_window":
                roll_counter_map[key] = 0
        return roll_counter_map

    def get_period_timesets(self, db: "DatabaseMapping") -> defaultdict:
        """Read ``period_timeset`` (period -> timeset) for every solve of the model.

        A 2-D ``period_timeset`` is refused: a solve is split into solves of their own by a 2-D map of the
        period parameters (:data:`SPLIT_PERIOD_PARAMS`), and each of those keeps the timesets of its periods.
        """
        entities = db.find_entities(entity_class_name="solve")
        params = db.find_parameter_values(
            entity_class_name="solve",
            parameter_definition_name="period_timeset",
        )
        timesets_used_by_solves: defaultdict = defaultdict(list)
        solves_in_model = [
            item
            for sublist in (
                list(self.model_solve.values())
                + list(self.contains_solves.values())
            )
            for item in sublist
        ]
        for entity in entities:
            if entity["name"] not in solves_in_model:
                continue
            for param in params:
                if param["entity_name"] != entity["name"]:
                    continue
                param_value = api.from_database(param["value"], param["type"])
                for i, _row in enumerate(param_value.indexes):
                    if isinstance(param_value.values[i], api.Map):
                        raise FlexToolConfigError(
                            f"solve '{param['entity_name']}': period_timeset is a 2-D map. It takes one timeset "
                            "per period (a 1-D map period -> timeset). To split the solve into one solve per "
                            "period group, give period_timeset all of its periods and make invest_periods, "
                            "realized_invest_periods, realized_periods or fix_storage_periods a 2-D map."
                        )
                    timesets_used_by_solves[param["entity_name"]].append(
                        (
                            param_value.indexes[i],
                            param_value.values[i],
                        )
                    )
        return timesets_used_by_solves

    def derive_solve(
        self,
        parent: str,
        new: str,
        *,
        overrides: "dict[str, Any] | None" = None,
        drops: "frozenset[str] | set[str]" = frozenset(),
    ) -> None:
        """Make the solve *new* from the solve *parent*: a deep copy of the parent's entry in every
        :data:`PER_SOLVE_FIELDS` field, except the fields in *overrides* (``{field: value}``;
        :data:`NO_ENTRY` = no entry) and in *drops* (no entry).  A field the parent has no entry in gives the new
        solve none either.

        The one copy step of every path that makes a solve from another one: a solve of a 2-D period map
        (:meth:`_split_2d_period_maps`), a nested solve renamed for its period
        (``RecursiveSolveBuilder.define_solve_recursive``) and the realization solve
        (:meth:`derive_realization_solve`).  ``model_solve`` is the caller's.  Raises ``RuntimeError`` (a FlexTool
        bug) when the config has a dict attribute that is not classified (``_check_field_registry``), and
        ``ValueError`` when *overrides* / *drops* name a field that is not per-solve.  Works on a test's partly
        built config (fields it lacks are skipped).
        """
        overrides = dict(overrides or {})
        drops = frozenset(drops)
        unknown = sorted((set(overrides) | drops) - set(PER_SOLVE_FIELDS))
        if unknown:
            raise ValueError(f"derive_solve: {unknown} are not per-solve fields")
        _check_field_registry(self)
        for name in PER_SOLVE_FIELDS:
            values = _per_solve_field(self, name)
            if values is None:
                continue
            if name in overrides:
                if overrides[name] is NO_ENTRY:
                    values.pop(new, None)
                else:
                    values[new] = overrides[name]
            elif name in drops or parent not in values:
                values.pop(new, None)
            else:
                values[new] = copy.deepcopy(values[parent])
        derived = getattr(self, "derived_solves", None)
        if derived is not None:
            derived[new] = (parent, frozenset(overrides) | drops)

    def check_derived_solves(self, names: "Iterable[str] | None" = None) -> None:
        """Raise when a solve made from another one (:attr:`derived_solves`; only *names* when given) differs from
        its parent in a :data:`PER_SOLVE_FIELDS` field other than the fields its path overrides or drops: a lost
        parameter is a FlexTool bug, never a silent default.  Run right after the solves are made (later passes,
        such as the defaults of ``TimelineConfig.create_assumptive_parts``, may change a solve on its own).  A
        missing key of a ``defaultdict`` field counts as its empty default."""
        _check_field_registry(self)
        derived = getattr(self, "derived_solves", {})
        problems: list[str] = []
        for new in (list(derived) if names is None else [n for n in names if n in derived]):
            parent, own = derived[new]
            for name in PER_SOLVE_FIELDS:
                if name in own:
                    continue
                values = _per_solve_field(self, name)
                if values is None:
                    continue
                if _entry_of(values, new) != _entry_of(values, parent):
                    problems.append(f"'{new}' (from '{parent}'): {name}")
        if problems:
            raise RuntimeError(
                "A solve made from another one has lost its parent's settings: " + "; ".join(problems)
                + ". This is a FlexTool bug; please report it.")

    def solves_in_run(self) -> list[str]:
        """The solves the model runs, in order: ``model.solves`` and the solves they contain (transitively; a
        ``contains_solves`` entry may be one name or a list).  After :meth:`_split_2d_period_maps` a split solve
        appears as its ``<solve>_<outer>`` solves.  For a model without ``model.solves``, every database solve
        (``create_assumptive_parts`` picks the sole one later).  The realization solves FlexTool adds later are not
        listed: they take their parent's solver settings (:meth:`derive_realization_solve`)."""
        if not self.model_solve:
            return list(getattr(self, "database_solves", []))
        out: list[str] = []
        stack = [s for solves in self.model_solve.values() for s in solves]
        while stack:
            s = stack.pop(0)
            if s in out:
                continue
            out.append(s)
            children = self.contains_solves.get(s) or []
            stack.extend([children] if isinstance(children, str) else children)
        return out

    def _split_2d_period_maps(self) -> None:
        """Make the solves that the 2-D period maps read by :meth:`periods_to_tuples` stand for.

        A database solve ``s`` whose ``invest_periods`` / ``realized_periods`` / ``realized_invest_periods`` /
        ``fix_storage_periods`` is a 2-D map ``{outer: [period, ...]}`` runs as one solve ``s_<outer>`` per
        outer key (the docs: "the model will generate these solves, with the periods given").  Each is
        :meth:`derive_solve` of ``s`` (every per-solve parameter of ``s``) with:

        * each 2-D parameter: the solve's inner list, as ``(outer, period)`` tuples (no entry when the map has no
          such outer key);
        * the solve's periods: the inner lists of all its 2-D parameters;
        * ``timesets_used_by_solves``: the database solve's entries of those periods;
        * each 1-D period list of ``s``: its periods that the solve has (no entry when none);
        * ``years_represented``: the whole map, as before (``_run_model`` keeps a solve's own periods);
        * ``stochastic_branches``: the rows of the periods it realizes (:meth:`_split_fan_rows`), as the solve written
          by hand has them.  A row that no split solve keeps raises :class:`FlexToolConfigError`: it would be
          silently ignored;
        * ``map2d_duplicate_of`` / ``map2d_period_parents`` record the split, and the solves replace ``s`` in
          ``model_solve``, in its place.

        Only the solves the model runs are split (:meth:`solves_in_run`; a 2-D map on another solve adds no
        solves).  Raises :class:`FlexToolConfigError` when the split solve is not in ``model.solves`` (only a
        contained one) and when a split solve's name is already a ``solve`` of the database
        (:meth:`_check_split_realized_periods` checks the realized periods afterwards).
        """
        in_run = set(self.solves_in_run())
        top_level = {s for solves in self.model_solve.values() for s in solves}
        database_solves = set(getattr(self, "database_solves", []))
        for parent, by_par in self._split_records.items():
            if parent not in in_run:
                continue
            maps = ", ".join(sorted(by_par))
            if self.model_solve and parent not in top_level:
                raise FlexToolConfigError(
                    f"solve '{parent}': {maps} is a 2-D map, which splits a solve of model.solves into one solve "
                    f"per row, but '{parent}' is only contained in another solve (contains_solves). Give the "
                    "contained solve 1-D period lists, or split the solve that contains it."
                )
            outers = list(dict.fromkeys(o for entries in by_par.values() for o in entries))
            taken = sorted(parent + "_" + o for o in outers if parent + "_" + o in database_solves)
            if taken:
                raise FlexToolConfigError(
                    f"solve '{parent}': its 2-D map ({maps}) makes the solve(s) {taken}, but the database has a "
                    "solve of that name already. Rename that solve, or change the first column of the 2-D map."
                )
            new_names: list[str] = []
            parent_timesets = list(self.timesets_used_by_solves.get(parent, []))
            # The forecast fans (``stochastic_branches`` rows) of the database solve, the rows some split solve
            # keeps, and the periods of the split solves (for the message on the rows none keeps).
            fan_rows = list(self.stochastic_branches.get(parent) or [])
            kept_rows: set[int] = set()
            has_period: set[str] = set()
            timeline_order = list(dict.fromkeys(str(entry[0]) for entry in parent_timesets))
            for outer in outers:
                new = parent + "_" + outer
                periods = list(dict.fromkeys(p for entries in by_par.values() for p in entries.get(outer, [])))
                own = set(periods)
                overrides: dict[str, Any] = {}
                for par, field_name in SPLIT_PERIOD_PARAMS.items():
                    if par in by_par:
                        inner = by_par[par].get(outer, [])
                        overrides[field_name] = [(outer, p) for p in inner] if inner else NO_ENTRY
                    elif parent in getattr(self, field_name):
                        kept = [t for t in getattr(self, field_name)[parent] if t[1] in own]
                        overrides[field_name] = kept if kept else NO_ENTRY
                timesets: list = []
                for p in periods:
                    for entry in parent_timesets:
                        if entry[0] == p and entry not in timesets:
                            timesets.append(entry)
                overrides["timesets_used_by_solves"] = timesets
                if fan_rows:
                    kept = self._split_fan_rows(fan_rows, overrides, periods, timeline_order,
                                                stand_for_fix_storage="fix_storage_periods" in by_par)
                    overrides["stochastic_branches"] = [copy.deepcopy(fan_rows[i]) for i in kept]
                    kept_rows.update(kept)
                    has_period.update(periods)
                self.derive_solve(parent, new, overrides=overrides)
                self.map2d_duplicate_of[new] = (parent, str(outer))
                for par in by_par:
                    self.map2d_period_parents[par].add(parent)
                new_names.append(new)
            if fan_rows and len(kept_rows) < len(fan_rows):
                raise FlexToolConfigError(self._unused_fan_rows_message(
                    parent, [fan_rows[i] for i in range(len(fan_rows)) if i not in kept_rows], has_period,
                    new_names))
            # The split solves replace their solve in its place in model.solves (the later solves run after them).
            for model, solves in list(self.model_solve.items()):
                if parent in solves:
                    at = solves.index(parent)
                    self.model_solve[model] = solves[:at] + new_names + solves[at + 1:]

    @staticmethod
    def _split_fan_rows(fan_rows: list, overrides: dict[str, Any], periods: list[str], timeline_order: list[str],
                        *, stand_for_fix_storage: bool) -> list[int]:
        """The indexes of the ``stochastic_branches`` rows ``(period, branch, analysis_time, realized, weight)`` of
        a database solve that one of its split solves keeps, as the solve written by hand has them (one fan per
        solve; a second fan in its horizon stops the run, item 229).  The split solve's periods are *periods*; its
        period lists are in *overrides* (``(row, period)`` tuples, or ``NO_ENTRY``).

        The split solve stands for the periods it realizes: ``realized_periods`` and ``realized_invest_periods``,
        and ``fix_storage_periods`` only when that is a 2-D map (*stand_for_fix_storage*; a 1-D list is kept over
        all periods of every split solve, so it would add the look-ahead periods).  With none of them, all of its
        periods.  It keeps

        1. the rows of the periods it stands for, when there are any;
        2. else the rows of its first period (in the database solve's ``period_timeset`` order) that has rows: a fan
           in its look-ahead, as in a solve written by hand whose first period is deterministic;
        3. else every row: then none of them reveals in the solve, and the run stops with the message that says so
           (``_stochastic.py``, with the advice for a 2-D map row).
        """
        fields = ["realized_periods", "realized_invest_periods"] + (["fix_storage_periods"] if stand_for_fix_storage
                                                                     else [])
        stand_for = {str(t[1]) for name in fields
                     for t in (overrides.get(name) if overrides.get(name) not in (None, NO_ENTRY) else [])}
        own = {str(p) for p in periods}
        stand_for = stand_for or own
        by_period: dict[str, list[int]] = {}
        for i, row in enumerate(fan_rows):
            by_period.setdefault(str(row[0]), []).append(i)
        kept = [i for p in sorted(stand_for & set(by_period)) for i in by_period[p]]
        if kept:
            return sorted(kept)
        order = [p for p in timeline_order if p in own] + [str(p) for p in periods if str(p) not in timeline_order]
        for p in order:
            if p in by_period:
                return by_period[p]
        return list(range(len(fan_rows)))

    @staticmethod
    def _unused_fan_rows_message(parent: str, rows: list, has_period: set[str], new_names: list[str]) -> str:
        """The error on ``stochastic_branches`` rows of a split database solve that none of its split solves
        keeps (:meth:`_split_fan_rows`): they would be silently ignored."""
        by_period: dict[str, list[str]] = {}
        for row in rows:
            by_period.setdefault(str(row[0]), []).append(f"{row[1]} at {row[2]}")
        parts = []
        for period, branches in by_period.items():
            if period in has_period:
                why = ("is a period of these solves, but each of them takes the forecasts of the periods it realizes "
                       "(realized_periods, realized_invest_periods) or, when it has none there, of its first period "
                       f"with forecasts: add '{period}' to realized_periods (or realized_invest_periods) in the row of "
                       f"the 2-D maps whose solve should realize it")
            else:
                why = (f"is a period of none of these solves: add '{period}' to the 2-D maps, in the row whose solve "
                       "should realize it (realized_periods or realized_invest_periods)")
            parts.append(f"period '{period}' ({', '.join(branches)}) {why}")
        periods = ", ".join(by_period)
        return (f"solve '{parent}': the stochastic_branches rows of period(s) {periods} are used by none of the "
                f"solves its 2-D period maps make ({', '.join(new_names)}): " + "; ".join(parts) + ". Or remove "
                "these rows.")

    def _check_split_realized_periods(self) -> None:
        """Raise when two split solves of one solve realize the same period (``realized_periods`` /
        ``realized_invest_periods``): the later one's results would overwrite the earlier ones (decision D3).  Run
        after :meth:`_check_realized_invest_periods`, whose messages are more specific."""
        by_parent: dict[str, list[str]] = {}
        for new, (parent, _outer) in self.map2d_duplicate_of.items():
            by_parent.setdefault(parent, []).append(new)
        for parent, new_names in by_parent.items():
            for field_name in ("realized_periods", "realized_invest_periods"):
                realized = {new: list(dict.fromkeys(str(t[1]) for t in getattr(self, field_name).get(new, [])))
                            for new in new_names}
                seen: dict[str, str] = {}
                for new in new_names:
                    for period in realized[new]:
                        if period in seen:
                            raise FlexToolConfigError(
                                self._split_overlap_message(parent, field_name, seen[period], new, period, realized))
                        seen[period] = new

    def _split_overlap_message(self, parent: str, field_name: str, first: str, second: str, period: str,
                               realized: dict[str, list[str]]) -> str:
        """The D3 error: what overlaps, where it comes from, and the values that would run."""
        row = {new: self.map2d_duplicate_of[new][1] for new in realized}
        head = (f"solve '{parent}': the split solves '{first}' and '{second}' both realize period '{period}' "
                f"({field_name}), and each period can be realized by one solve only.")
        if parent in self.map2d_period_parents.get(field_name, set()):
            return (f"{head} {field_name} is a 2-D map whose rows '{row[first]}' and '{row[second]}' both list "
                    f"'{period}': remove '{period}' from one of the two rows.")
        # A 1-D list: each split solve realizes the periods of the list that it has.  Suggest the 2-D map that gives
        # each period to the last split solve that has it (the one that looks least far ahead).
        last: dict[str, str] = {}
        for new, periods in realized.items():
            for p in periods:
                last[p] = new
        rows = {row[new]: [p for p, owner in last.items() if owner == new] for new in realized}
        listed = "; ".join(f"'{new}' realizes {', '.join(periods)}" for new, periods in realized.items())
        suggestion = "{" + ", ".join(f"{r}: [{', '.join(ps)}]" for r, ps in rows.items() if ps) + "}"
        return (f"{head} {field_name} of '{parent}' is a 1-D list, and each split solve realizes the periods of the "
                f"list that its row of the 2-D maps has ({listed}). Make {field_name} a 2-D map with the same "
                f"first column and each period in one row only, for example {suggestion}.")

    def copy_reader_values(self, copy_name: str) -> dict[str, Any]:
        """The ``solve`` parameter values a solve of a 2-D period map (*copy_name*) has instead of its database
        solve's, as parsed Spine values for ``SpineDbReader.register_derived_solve(replace=...)`` (``None`` = no
        value): the period lists as arrays, ``period_timeset`` and ``years_represented`` (its own periods only;
        like every solve it discounts from the start of the run, the years before it coming from the run's
        period history) as maps with the index name
        ``period``, and ``stochastic_branches`` with its own rows (:meth:`_split_fan_rows`).  Every other parameter is
        its database solve's."""
        def as_array(tuples: "list | None") -> "api.Array | None":
            return api.Array([str(t[1]) for t in tuples], value_type=str) if tuples else None

        out: dict[str, Any] = {par: as_array(getattr(self, field_name).get(copy_name))
                               for par, field_name in SPLIT_PERIOD_PARAMS.items()}
        timesets = self.timesets_used_by_solves.get(copy_name, [])
        out["period_timeset"] = (api.Map([str(p) for p, _ in timesets], [str(ts) for _, ts in timesets],
                                         index_name="period") if timesets else None)
        own = {str(p) for p, _ in timesets}
        years = [(str(p), float(y)) for p, y in self.solve_period_years_represented.get(copy_name, [])
                 if str(p) in own]
        out["years_represented"] = (api.Map([p for p, _ in years], [y for _, y in years], index_name="period")
                                    if years else None)
        if copy_name in self.stochastic_branches:
            out["stochastic_branches"] = _fan_rows_to_map(self.stochastic_branches[copy_name])
        return out

    def spawn_realized_dispatch_sibling(
        self, parent: str, sibling: str,
    ) -> None:
        """W4b — positionally insert *sibling* right after *parent* in
        ``model_solve`` and make it the deterministic, fixed-capacity,
        monolithic dispatch over the parent's REALIZED window
        (:meth:`derive_realization_solve`).

        The sibling is the engine-automatic counterpart of W8.1's
        author-declared ``*_realized_dispatch`` solve: it consumes the
        parent's committed capacity through the normal invest→dispatch
        handoff (hence the insert IMMEDIATELY after the parent, not an
        append) and last-wins hands it the realized window it then owns.

        Note on multi-period storage: dropping ``fix_storage_periods`` is
        correct even when the realized window spans MULTIPLE periods.
        ``fix_storage_periods`` pins storage at period boundaries for
        ROLLING / multi-solve handoff continuity; a single monolithic
        ``single_solve`` instead links storage across its periods through
        the ordinary storage-balance constraints and respects the
        node-level ``storage_binding_method`` (an entity parameter, read
        unchanged via the full override path).  So the sibling's storage
        trajectory over its realized periods is a correct free
        deterministic optimisation, not a re-floated gap.
        """
        # ►► The reader half of the sibling config is ``SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS``
        # (feature map: ``_orchestration._spawn_realized_dispatch_siblings``; pinned by
        # ``tests/engine/test_realized_dispatch_coherence.py``).
        # Positional insert right after the parent (NOT append + remove).
        for _model, solves in self.model_solve.items():
            if parent in solves and sibling not in solves:
                solves.insert(solves.index(parent) + 1, sibling)
        SolveConfig.derive_realization_solve(self, parent, sibling)
        # W4c — record the explicit sibling -> parent mapping so the
        # native-run post-pass can project each stochastic parameter onto
        # the parent's realized branch (see ``_run_model`` step 3.6).
        # Using this map (not a ``__realized_dispatch`` name-parse) keeps
        # the sibling identification robust.
        self.realized_dispatch_parent[sibling] = parent

    def database_solve_names(self) -> dict[str, str]:
        """Solve -> the database solve it was made from by a 2-D period map (``<solve>_<period>`` -> ``<solve>``),
        for messages that should name what the user wrote."""
        return {solve: parent for solve, (parent, _outer) in self.map2d_duplicate_of.items()}

    def derive_realization_solve(self, parent: str, sibling: str) -> None:
        """Give the realization solve *sibling* of *parent* its per-solve settings (:meth:`derive_solve`): every
        field of the parent except :data:`REALIZATION_SOLVE_DROPS`, with ``solve_modes`` set to the parent's mode
        (``rolling_window`` or ``single_solve``) and ``realized_periods`` to the parent's realized window.  The
        realization solve of a single solve has the parent's periods from its first to its last realized period
        (:meth:`realization_window`): it dispatches no look-ahead period with the realized data (item 255).

        Applied to the driving ``runner.state.solve`` (:meth:`spawn_realized_dispatch_sibling`) and to the outer
        cascade ``state.solve``, where the ``run()`` closure reads each solve's solver and scaling
        (``_orchestration._spawn_realized_dispatch_siblings``): the sibling runs on its parent's solver,
        ``solver_arguments``, ``solver_mip_gap`` and ``scaling``."""
        parent_mode = self.solve_modes.get(parent, "single_solve")
        mode = "rolling_window" if parent_mode == "rolling_window" else "single_solve"
        realized = copy.deepcopy(self.realized_periods.get(parent, []))
        if mode == "single_solve":
            # The realization of an investment solve starts at its reveal: the periods before it are the parent's.
            window = {str(p) for p, _ts in SolveConfig.realization_window(self, parent)[0]}
            realized = [t for t in realized if str(t[1]) in window]
        overrides: dict[str, Any] = {
            "solve_modes": mode,
            "realized_periods": realized,
        }
        if mode == "single_solve":
            # Only the committed periods: a look-ahead period dispatched with the realized data would let the
            # committed periods' operation (storage) use information the stochastic solve did not have (item
            # 255).  A rolling sibling's rolls take the windows of the rolls they realize instead (``_run_model``
            # step 1b): the whole horizon for a realization roll (``_realization_tail``), else the jump.
            timesets, years = SolveConfig.realization_window(self, parent)
            overrides["timesets_used_by_solves"] = timesets
            overrides["solve_period_years_represented"] = years
        # A rolling sibling needs a roll_counter entry (make_roll_counter ran at config load).
        roll_counter = getattr(self, "roll_counter", None)
        if mode == "rolling_window" and roll_counter is not None:
            overrides["roll_counter"] = roll_counter.get(parent, 0)
        SolveConfig.derive_solve(self, parent, sibling, overrides=overrides, drops=REALIZATION_SOLVE_DROPS)

    def realization_window(self, parent: str) -> "tuple[list, list]":
        """The ``timesets_used_by_solves`` and ``solve_period_years_represented`` entries of the realization solve
        of the single solve *parent*: the parent's entries from its first to its last realized period, in the
        parent's order (item 255: the realization dispatches the committed periods, not the look-ahead).  For an
        investment solve (:func:`is_investment_solve`) the window starts at the later of its first realized period
        and its reveal period (:meth:`reveal_period`): the investment solve realizes the operations before the
        reveal itself, so its realization does not dispatch them again, and starts from the storage they end with
        (``_carry_routing``).

        A period the parent does not realize between two it realizes is dispatched too, so storage runs through
        it as in the parent, but not reported: the realization solve's ``realized_periods`` are the parent's in the
        window.  It lies before the last committed period, so dispatching it with the realized data uses no later
        actuals.
        """
        realized = {str(t[1]) for t in (getattr(self, "realized_periods", None) or {}).get(parent, [])}
        timesets = (getattr(self, "timesets_used_by_solves", None) or {}).get(parent, [])
        order = list(dict.fromkeys(str(p) for p, _ts in timesets))
        positions = [i for i, p in enumerate(order) if p in realized]
        if positions and is_investment_solve(self, parent):
            reveal = SolveConfig.reveal_period(self, parent)
            if reveal in order:
                positions = [i for i in positions if i >= order.index(reveal)]
        span = set(order[positions[0]:positions[-1] + 1]) if positions else set()
        return ([copy.deepcopy(row) for row in timesets if str(row[0]) in span],
                [copy.deepcopy(row) for row in
                 (getattr(self, "solve_period_years_represented", None) or {}).get(parent, [])
                 if str(row[0]) in span])

    def realization_reader_values(self, parent: str) -> dict[str, Any]:
        """The ``solve`` parameter values the reader gives the realization solve of *parent* instead of the
        parent's, for ``SpineDbReader.register_derived_solve(replace=...)``: for a single solve, ``period_timeset``
        and ``years_represented`` of its first to last realized period (:meth:`realization_window`), as maps with the
        index name ``period`` (as :meth:`copy_reader_values`); empty for a rolling solve (each roll of its
        realization solve takes the window of the roll it realizes, ``_run_model`` step 1b)."""
        if (getattr(self, "solve_modes", None) or {}).get(parent, "single_solve") == "rolling_window":
            return {}
        timesets, years = SolveConfig.realization_window(self, parent)
        return {
            "period_timeset": (api.Map([str(p) for p, _ in timesets], [str(ts) for _, ts in timesets],
                                       index_name="period") if timesets else None),
            "years_represented": (api.Map([str(p) for p, _ in years], [float(y) for _, y in years],
                                          index_name="period") if years else None),
        }

    def solver_config_for(self, solve_name: str) -> SolverConfig:
        """The solver configuration *solve_name* runs with (HiGHS defaults when the solve sets none).

        Raises when ``solver_configs`` and ``solver_settings.solvers`` name different solvers: a solve made from
        another one (a 2-D period map, a realization solve) that lost its solver configuration would otherwise
        run on HiGHS without a word, as before item 240."""
        config = self.solver_configs.get(solve_name, SolverConfig())
        named = self.solver_settings.solvers.get(solve_name, "highs")
        if named != config.name:
            raise RuntimeError(
                f"Solve {solve_name!r}: its solver is {named!r}, but its solver configuration says "
                f"{config.name!r}. This is a FlexTool bug (a solve made from another one lost its solver "
                "settings); please report it."
            )
        return config

    def register_realization_solve(self, realization_solve: str, parent: str) -> None:
        """Register the author-declared *realization_solve* as the realization solve of *parent*.

        The solve keeps its authored configuration; registering it gives it the same treatment as an
        engine-spawned sibling: it takes its parent's pins (investment, NA-tied units online), the
        stochastic parameters projected onto the parent's realized branch, and the parent's realized
        window through a silent last-wins.
        """
        self.realized_dispatch_parent[realization_solve] = parent

    # ------------------------------------------------------------------
    # v60/v62 per-solve decomposition accessors
    # ------------------------------------------------------------------

    def decomposition_for(self, solve_name: str) -> str:
        """Resolve the decomposition scheme for *solve_name*.

        Returns ``"benders"`` only when the solve explicitly authors
        ``solve.decomposition = benders``; everything else (unset, the
        schema default ``"none"``, blank, or any unrecognised value)
        resolves to ``"none"`` (monolithic).  Recognised values are
        normalised to lower-case so authoring case does not matter.
        """
        raw = self.decomposition.get(solve_name)
        if raw is None:
            return "none"
        value = str(raw).strip().lower()
        return "benders" if value == "benders" else "none"

    def scaling_for(self, solve_name: str) -> str | None:
        """Resolve the autoscaler scaling mode authored on *solve_name*.

        Returns the normalised lower-case mode string (one of
        ``off`` / ``solver_only`` / ``basic`` / ``full``) when the solve
        explicitly authors ``solve.scaling``; returns ``None`` when the
        parameter is absent, blank, or unrecognised so the caller falls
        back to the run-time ``--scaling`` / ``FLEXTOOL_SCALING``
        override or the engine default (``full``).  Authoring case does
        not matter.
        """
        raw = self.scaling.get(solve_name)
        if raw is None:
            return None
        value = str(raw).strip().lower()
        return value if value in (
            "off", "solver_only", "basic", "full"
        ) else None

    def invest_branching_method_for(self, solve_name: str) -> str:
        """Resolve ``solve.invest_solve_branching_method`` for *solve_name*: a member of
        :data:`INVEST_SOLVE_BRANCHING_METHODS`.  Absent, blank or unknown -> ``"operations_only"``, the shipped
        default (like every ``solve`` parameter of this class, the definition default is not read), and so is every
        value of a solve that runs ``deterministic`` (:attr:`deterministic_branches`: it has no forecasts left).
        Authoring case is ignored.  ``_invest_branching_raw_guard`` refuses an unknown value before the solve runs, and
        :meth:`check_invest_branching` a value on a dispatch solve or on a solve without ``stochastic_branches``."""
        raw = None if solve_name in self.deterministic_branches else self.invest_solve_branching_method.get(solve_name)
        value = "" if raw is None else str(raw).strip().lower()
        return value if value in INVEST_SOLVE_BRANCHING_METHODS else INVEST_SOLVE_BRANCHING_METHODS[0]

    def recourse_invest(self, solve_name: str) -> bool:
        """Whether the investments of *solve_name* branch after its reveal (``investments_after_reveal_period`` or
        ``investments_at_reveal_period``): the model's recourse switch, emitted as
        ``solve_data/stochastic_invest_method.csv`` (``recourse`` / ``none``) and read back by
        ``_derived._recourse_invest_active``."""
        return self.invest_branching_method_for(solve_name) != INVEST_SOLVE_BRANCHING_METHODS[0]

    def reveal_period(self, solve_name: str) -> str | None:
        """The period where the fan of *solve_name* opens: the first period, in the solve's period order
        (``timesets_used_by_solves``), of a forecast row (``realized`` other than ``yes``) of its
        ``stochastic_branches``.  ``None`` without a forecast row.  A row period the solve does not have comes after
        the solve's own periods, in row order."""
        rows = self.stochastic_branches.get(solve_name) or []
        periods = list(dict.fromkeys(str(row[0]) for row in rows
                                     if not (len(row) > 3 and str(row[3]).strip().lower() == "yes")))
        if not periods:
            return None
        order = [str(p) for p, _ts in self.timesets_used_by_solves.get(solve_name, [])]
        return min(periods, key=lambda p: (order.index(p), 0) if p in order else (len(order), periods.index(p)))

    def invest_tie_periods_for(self, solve_name: str) -> list[tuple[str, str]]:
        """The periods whose investment decisions *solve_name* shares across its forecasts as one decision of every
        branch copy, as ``(p, p)`` tuples: the reveal period under ``investments_after_reveal_period``, none
        otherwise.  Under ``operations_only`` the investments are on the real period names (one decision by
        construction); under ``investments_at_reveal_period`` every forecast invests on its own from the reveal
        period on.  The periods before the reveal have one copy each.  Emitted as
        ``solve_data/pd_non_anticipativity_invest_periods.csv`` (``_per_solve.emit_per_solve_sets``)."""
        if self.invest_branching_method_for(solve_name) != "investments_after_reveal_period":
            return []
        reveal = self.reveal_period(solve_name)
        return [] if reveal is None else [(reveal, reveal)]

    def per_forecast_invest_periods(self, solve_name: str) -> list[str]:
        """The periods of *solve_name* (in its period order, ``timesets_used_by_solves``) whose investments are per
        forecast branch, so cannot be realized: those after the reveal period under
        ``investments_after_reveal_period``, the reveal period and later ones under ``investments_at_reveal_period``,
        none under ``operations_only`` or without a reveal period the solve has."""
        method = self.invest_branching_method_for(solve_name)
        if method == INVEST_SOLVE_BRANCHING_METHODS[0]:
            return []
        reveal = self.reveal_period(solve_name)
        order = [str(p) for p, _ts in self.timesets_used_by_solves.get(solve_name, [])]
        if reveal is None or reveal not in order:
            return []
        return order[order.index(reveal) + (1 if method == "investments_after_reveal_period" else 0):]

    def operational_window_for(self, solve_name: str) -> list[tuple[str, str]] | None:
        """The operational non-anticipativity window of *solve_name* (``_per_solve.emit_per_solve_sets``):
        ``[]`` for an investment solve (:func:`is_investment_solve`) with ``stochastic_branches``, whose forecasts
        operate each on their own from the reveal (nothing tied); ``None`` for every other solve, which ties the
        storage, units online and reserve of its forecasts over its committed window (the realized-dispatch,
        committed-in-fan and fix-storage steps)."""
        if self.stochastic_branches.get(solve_name) and is_investment_solve(self, solve_name):
            return []
        return None

    def check_invest_branching(self) -> None:
        """Raise :class:`FlexToolConfigError` naming the solve (the solve the user wrote for a solve of a 2-D period
        map) when ``invest_solve_branching_method`` is set on a solve that is not an investment solve
        (:func:`is_investment_solve`: a rolling solve, or one without ``invest_periods`` /
        ``realized_invest_periods``), when it is set on a solve without ``stochastic_branches``, or when a solve whose
        investments branch realizes a period whose investments are per branch: under
        ``investments_after_reveal_period`` a period after the reveal period, under
        ``investments_at_reveal_period`` the reveal period or a later one.  Over the solves the model runs
        (:meth:`solves_in_run`), after the 2-D map split."""
        problems: list[str] = []
        for s in self.solves_in_run():
            raw = self.invest_solve_branching_method.get(s)
            if raw is None or not str(raw).strip():
                continue
            name = self.map2d_duplicate_of.get(s, (s,))[0]
            method = self.invest_branching_method_for(s)
            if not is_investment_solve(self, s):
                kind = ("it is a rolling solve" if self.solve_modes.get(s) == "rolling_window"
                        else "it has no invest_periods or realized_invest_periods")
                problems.append(
                    f"Solve '{name}' sets invest_solve_branching_method {method}, which applies to investment solves "
                    f"only, but {kind}. Remove invest_solve_branching_method from '{name}': a dispatch solve with "
                    "stochastic_branches is set up with dispatch_realization_method.")
                continue
            if not self.stochastic_branches.get(s):
                problems.append(
                    f"Solve '{name}' sets invest_solve_branching_method {method} but has no stochastic_branches, "
                    "so there is nothing to branch. Add stochastic_branches to the solve, or remove "
                    "invest_solve_branching_method.")
                continue
            branched = set(self.per_forecast_invest_periods(s))
            if not branched:
                continue
            reveal = self.reveal_period(s)
            for par, field_name in (("realized_invest_periods", "realized_invest_periods"),
                                    ("realized_periods", "realized_periods")):
                bad = list(dict.fromkeys(str(t[1]) for t in getattr(self, field_name).get(s, [])
                                         if str(t[1]) in branched))
                if bad:
                    which = ("after the reveal period" if method == "investments_after_reveal_period"
                             else "from the reveal period on")
                    problems.append(
                        f"Solve '{name}' sets invest_solve_branching_method {method}, so its investments "
                        f"{which} ('{reveal}') are per forecast branch, but its {par} include "
                        f"{', '.join(repr(p) for p in bad)}. A per-branch decision cannot be realized: leave "
                        f"{'that period' if len(bad) == 1 else 'those periods'} out of {par}, and realize "
                        f"{'it' if len(bad) == 1 else 'them'} in a later solve.")
        if problems:
            raise FlexToolConfigError(" ".join(problems))

    def dispatch_realization_method_for(self, solve_name: str) -> str:
        """Resolve ``solve.dispatch_realization_method`` for *solve_name*.

        Returns the normalised lower-case member: ``"deterministic"``,
        ``"within_solve"`` or ``"separate_dispatch"``.  Absent / blank ->
        ``"deterministic"``, the shipped default: like every ``solve``
        parameter of this class, the definition default is not read, so the
        fall-back is hard-coded.  An unrecognised value also reads
        ``"deterministic"`` here; ``_dispatch_realization_raw_guard`` refuses
        it before the solve runs.  An investment solve (:func:`is_investment_solve`) does not read the
        method: it reads ``"within_solve"`` (it solves its forecasts; a ``separate_dispatch`` there is ignored,
        warning W-0640 from :meth:`apply_deterministic_realization`).  Authoring case is ignored.  Without
        ``stochastic_branches``, ``deterministic`` and ``within_solve`` give
        the same model (``separate_dispatch`` needs a ``realized: yes``
        branch).
        """
        if is_investment_solve(self, solve_name):
            return "within_solve"
        raw = self.dispatch_realization_method.get(solve_name)
        value = "" if raw is None else str(raw).strip().lower()
        return value if value in ("within_solve", "separate_dispatch") else "deterministic"

    def apply_deterministic_realization(self) -> None:
        """Make each solve with ``stochastic_branches`` that runs ``deterministic`` a deterministic solve.

        The forecasts, ``invest_solve_branching_method`` and the realization solve are ignored: the solve's
        ``stochastic_branches`` rows move to :attr:`deterministic_branches` (the solve then reads only its
        ``realized: yes`` branch, flattened into one timeline, ``native_run_model`` step 3.7), and
        :meth:`invest_branching_method_for` reads it as ``operations_only``.  A dispatch solve runs ``deterministic`` when its method is blank,
        absent or ``deterministic``, with one warning W-0593 per database solve (a solve of a 2-D period map is named
        by the solve the user wrote).  An investment solve (:func:`is_investment_solve`) never does: it does not read
        the method and solves its forecasts; a ``separate_dispatch`` on it is ignored with one warning W-0640 per
        database solve.  Raises :class:`FlexToolConfigError` naming the solve when no
        row says ``realized: yes`` (the same exact match as the stochastic pass), or when the ``realized: yes``
        rows name more than one branch (one realized series per solve, as for ``within_solve``).  Only a blank,
        absent or ``deterministic`` value counts; an unknown value is left for ``_dispatch_realization_raw_guard``.
        Over the solves the model runs (:meth:`solves_in_run`, after the 2-D map split)."""
        from flextool import run_warnings

        warned: set[str] = set()
        for s in self.solves_in_run():
            rows = self.stochastic_branches.get(s) or []
            raw = self.dispatch_realization_method.get(s)
            # A realization solve has no rows here (engine-spawned ones drop stochastic_branches, an author-declared
            # one is deterministic; a realization roll's forecasts come from its plan), so it is never made
            # deterministic.
            method = str("" if raw is None else raw).strip().lower()
            name = self.map2d_duplicate_of.get(s, (s,))[0]
            if is_investment_solve(self, s):
                if method == "separate_dispatch" and name not in warned:
                    warned.add(name)
                    run_warnings.warn("W-0640", solve=name)
                continue
            if not rows or method not in ("", "deterministic"):
                continue
            realized = list(dict.fromkeys(row[1] for row in rows if len(row) > 3 and row[3] == "yes"))
            if not realized:
                raise FlexToolConfigError(
                    f"Solve '{name}' has stochastic_branches and dispatch_realization_method deterministic (the "
                    "default of a dispatch solve since FlexTool 4.7.0), "
                    "but no branch with realized: yes. Mark the branch that holds "
                    "the realized data with realized: yes, or set dispatch_realization_method within_solve or "
                    "separate_dispatch to solve the forecasts.")
            if len(realized) > 1:
                raise FlexToolConfigError(
                    f"Solve '{name}' has stochastic_branches and dispatch_realization_method deterministic, and its "
                    f"realized: yes rows name different branches: {', '.join(repr(str(b)) for b in realized)}. "
                    "A deterministic solve reads one realized series: name the same branch in every realized: yes "
                    "row. Check stochastic_branches.")
            self.deterministic_branches[s] = self.stochastic_branches.pop(s)
            if name not in warned:
                warned.add(name)
                run_warnings.warn("W-0593", solve=name)

    def _reduce_detail_for(self, solve_name: str, values: dict, param: str, valid: tuple[str, ...]) -> str:
        """The authored ``solve.<param>`` of *solve_name*: its roll suffix stripped, a realization solve mapped to
        its parent (:attr:`realized_dispatch_parent`), absent or blank -> ``"none"``.  Case and surrounding blanks
        are ignored.  An unknown value raises :class:`FlexToolConfigError` naming the solve and the valid values (a
        typo must not silently run the full model)."""
        base = re.sub(r"_roll_\d+$", "", solve_name)
        owner = self.realized_dispatch_parent.get(base, base)
        owner = re.sub(r"_roll_\d+$", "", owner)
        raw = values.get(owner)
        if raw is None or str(raw).strip() == "":
            return "none"
        value = str(raw).strip().lower()
        if value not in valid:
            raise FlexToolConfigError(
                f"solve '{owner}': {param} = '{raw}' is not one of {', '.join(valid)}.")
        return value

    def reduce_detail_overridden(self, realization_solve: str) -> list[str]:
        """The levels an author set on the realization solve *realization_solve* that its parent's levels override
        (one line per parameter whose authored value differs from the parent's): a realization solve uses the
        parent's levels, so that its online variables and reserve rows match those of the solve it realizes.  Empty when it sets none, or the same."""
        parent = self.realized_dispatch_parent.get(realization_solve)
        if parent is None:
            return []
        out = []
        for param, values, resolve in (
                ("reduce_unit_detail_method", self.reduce_unit_detail_method, self.reduce_unit_detail_for),
                ("reduce_system_detail_method", self.reduce_system_detail_method, self.reduce_system_detail_for)):
            raw = values.get(realization_solve)
            if raw is None or str(raw).strip() == "":
                continue
            own, used = str(raw).strip().lower(), resolve(realization_solve)
            if own != used:
                out.append(f"realization solve '{realization_solve}' sets {param} = '{raw}', but it uses "
                           f"'{used}', the value of its parent solve '{parent}' (so that its online variables "
                           "and reserve rows match those of the solve it realizes).")
        return out

    def reduce_unit_detail_for(self, solve_name: str) -> str:
        """Resolve ``solve.reduce_unit_detail_method`` for *solve_name* (a roll ``<solve>_roll_N`` and a
        realization solve take the base solve's / the parent's value).  ``"none"`` when not authored."""
        return self._reduce_detail_for(solve_name, self.reduce_unit_detail_method, "reduce_unit_detail_method",
                                       REDUCE_UNIT_DETAIL_METHODS)

    def reduce_system_detail_for(self, solve_name: str) -> str:
        """Resolve ``solve.reduce_system_detail_method`` for *solve_name*, as :meth:`reduce_unit_detail_for`."""
        return self._reduce_detail_for(solve_name, self.reduce_system_detail_method,
                                       "reduce_system_detail_method", REDUCE_SYSTEM_DETAIL_METHODS)

    def benders_config_for(self, solve_name: str) -> tuple[int, float, float]:
        """Resolve ``(max_iter, tol, in_out_weight)`` for *solve_name*.

        Falls back to the schema defaults (50 / 1e-3 / 0.0) for any knob
        the solve does not author.  ``params_to_dict`` stores scalar
        floats as ``str(float)``, so values are coerced through ``float``
        here; ``max_iter`` is additionally rounded to ``int``.  The
        ``in_out_weight`` (default 0.0 = OFF = exact Benders) is the DB
        value; the machine-local ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` env
        overrides it downstream in
        :func:`flextool.engine.solve._benders._resolve_benders_in_out_weight`.
        """
        max_iter_raw = self.benders_max_iter.get(solve_name)
        tol_raw = self.benders_tolerance.get(solve_name)
        weight_raw = self.benders_in_out_weight.get(solve_name)
        max_iter = int(float(max_iter_raw)) if max_iter_raw is not None else 50
        tol = float(tol_raw) if tol_raw is not None else 1e-3
        in_out_weight = float(weight_raw) if weight_raw is not None else 0.0
        return max_iter, tol, in_out_weight

    def periods_to_tuples(
        self,
        db: "DatabaseMapping",
        cl: str,
        par: str,
        *,
        split: bool = False,
    ) -> defaultdict:
        """Read a period-list solve parameter as ``{solve: [(p, p), ...]}`` (an Array of periods, in order).

        With *split* (the :data:`SPLIT_PERIOD_PARAMS`), a 2-D map ``{outer: {period: ...}}`` is recorded in
        ``_split_records`` (``{outer: [period, ...]}``; the inner values do not matter) for
        :meth:`_split_2d_period_maps`, which runs after every parameter is read; the database solve gets no
        entry of its own.  Without it a map is refused.
        """
        entities = db.find_entities(entity_class_name=cl)
        params = db.find_parameter_values(
            entity_class_name=cl,
            parameter_definition_name=par,
        )
        result_dict: defaultdict = defaultdict(list)
        for entity in entities:
            for param in params:
                if param["entity_name"] != entity["name"]:
                    continue
                param_value = api.from_database(param["value"], param["type"])
                if not isinstance(param_value, api.Map):
                    for row in param_value.values:
                        result_dict[param["entity_name"]].append((row, row))
                    continue
                if not split:
                    raise FlexToolConfigError(
                        f"solve '{param['entity_name']}': {par} is a map; it takes an array of periods."
                    )
                entries: dict = {}
                for outer, inner in zip(param_value.indexes, param_value.values):
                    if not isinstance(inner, api.Map):
                        raise FlexToolConfigError(
                            f"solve '{param['entity_name']}': {par} mixes a 2-D map with plain values (entry "
                            f"'{outer}'). It takes an array of periods, or a 2-D map whose first column names "
                            "the solve and whose second column lists its periods."
                        )
                    entries[outer] = list(inner.indexes)
                self._split_records.setdefault(param["entity_name"], {})[par] = entries
        return result_dict


def _fan_rows_to_map(rows: list) -> "api.Map | None":
    """The ``stochastic_branches`` value of the rows ``(period, branch, analysis_time, realized, weight)``: a map
    period -> branch -> analysis_time -> realized -> weight, each level in the order of its first row."""
    if not rows:
        return None
    tree: dict = {}
    for period, branch, analysis_time, realized, weight in rows:
        tree.setdefault(str(period), {}).setdefault(str(branch), {}).setdefault(str(analysis_time), {})[
            str(realized)] = weight
    names = ("period", "branch", "analysis_time", "realized")

    def build(level: dict, depth: int) -> "api.Map":
        keys = list(level)
        values = [level[k] if depth == len(names) - 1 else build(level[k], depth + 1) for k in keys]
        return api.Map(keys, values, index_name=names[depth])

    return build(tree, 0)


def register_solve_copies(solve_cfg, db_reader) -> None:
    """Register the solves of every 2-D period map (``SolveConfig.map2d_duplicate_of``) as ``solve`` entities of
    *db_reader*, so the reader-side lookups by the active solve's name find them like a solve written by hand.

    Each gets every ``solve`` parameter value of its database solve, with its own period lists, timesets and
    ``years_represented`` instead (``SolveConfig.copy_reader_values``: arrays of periods, maps with the index
    name ``period``); the 2-D rows of every solve are removed from the reader (a split solve does not run, nor does
    a solve outside the run, and the maps would sit next to the period arrays of other solves in one frame).  A no-op without 2-D period maps."""
    copies = getattr(solve_cfg, "map2d_duplicate_of", None) or {}
    split_params: dict[str, set] = {}
    for copy_name, (parent, _outer) in copies.items():
        db_reader.register_derived_solve(parent, copy_name, replace=solve_cfg.copy_reader_values(copy_name))
        split_params.setdefault(parent, set())
    for par, parents in getattr(solve_cfg, "map2d_period_parents", {}).items():
        for parent in parents:
            split_params.setdefault(parent, set()).add(par)
    # A 2-D map on a solve the model does not run is not split; its rows go too (no solve of the run reads them).
    for parent, by_par in getattr(solve_cfg, "_split_records", {}).items():
        split_params.setdefault(parent, set()).update(by_par)
    for parent, params in split_params.items():
        db_reader.remove_solve_values(parent, sorted(params))


__all__ = [
    "DictMode",
    "HiGHSConfig",
    "SolverConfig",
    "SolverSettings",
    "SolveConfig",
    "get_single_entities",
    "params_to_dict",
    "register_solve_copies",
]
