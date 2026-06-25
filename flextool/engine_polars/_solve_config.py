"""Solve-level configuration loaded from a SpineDB scenario.

Architecture notes
------------------

* Every downstream module reads ``state.solve.<dict>`` keys produced
  here; the keys must match the canonical names (including the
  ``np.str_`` leak through Spine ``Map.indexes``, the stringified-
  float ``rolling_times`` entries, and the lockstep mutation in
  :meth:`SolveConfig.duplicate_solve`).
* The factory uses :class:`spinedb_api.DatabaseMapping` directly.
* Reads ``solve``, ``model``, ``unit`` parameter classes only.
  Loading order matters (``make_roll_counter`` →
  ``get_period_timesets`` → 4× ``periods_to_tuples``) because each
  may call :meth:`duplicate_solve`, which mutates 19 sibling
  defaultdicts in lockstep.
* The DB schema is assumed to be v50+.  v50 moved
  ``new_stepduration`` from ``timeset`` to ``solve``;
  ``update_flextool/db_migration.py`` handles upgrades for older DBs
  before this loader runs.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import spinedb_api as api

from flextool.engine_polars._solve_state import FlexToolConfigError

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
    :func:`flextool.engine_polars._solver_dispatch._resolve_effective_highs_options`
    where it is overlaid on top of ``solver_config/highs.opt`` and
    below the CLI overrides.  Empty when no solve authored an entry.
    """

    solvers: dict[str, str]
    precommand: dict[str, str]
    arguments: dict[str, dict[str, str]]


@dataclass
class SolverConfig:
    """Per-solve multi-solver dispatch configuration (v52 schema).

    Holds the seven solver-selection parameters introduced by the
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
    io_api
        "direct" (in-process binding, fastest), "mps" or "lp" (file
        fallback).  Default ``"direct"``.
    options
        Free-form key→value dict forwarded raw to the solver.  Empty
        by default; user populates via the ``solver_arguments``
        1d-map parameter (Batch C.2: the legacy ``solver_options``
        Map was folded into ``solver_arguments`` and removed).
    time_limit
        Wall-clock seconds; ``None`` means no limit.  Translated to
        each solver's native parameter name by
        :func:`flextool.engine_polars._solver_dispatch.build_solver_options`.
    mip_gap
        Relative MIP gap; ``None`` means solver default.
    threads
        Worker thread cap; ``None`` means solver default.
    log_level
        ``"silent"`` / ``"normal"`` / ``"verbose"``.  Default
        ``"normal"``.
    """

    name: str = "highs"
    io_api: str = "direct"
    options: dict[str, Any] = field(default_factory=dict)
    time_limit: float | None = None
    mip_gap: float | None = None
    threads: int | None = None
    log_level: str = "normal"


# ---------------------------------------------------------------------------
# SolveConfig — main container
# ---------------------------------------------------------------------------


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
        user_bound_scale: dict | None = None,
        solver_configs: dict[str, "SolverConfig"] | None = None,
        decomposition: dict | None = None,
        benders_max_iter: dict | None = None,
        benders_tolerance: dict | None = None,
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
        # solve-name → integer user_bound_scale override.  When set, overrides
        # polar-high's stream-time auto-pick
        # (``polar_high.engine._recommend_user_bound_scale``).  Pass the
        # value HiGHS recommends in its "user-scaled problem has some
        # excessively large row bounds — Consider setting the user_bound_scale
        # option to <N>" warning for the most reliable result.
        self.user_bound_scale: dict = (
            user_bound_scale if user_bound_scale is not None else {}
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

        # Computed fields — populated by load_from_db after construction.
        self.roll_counter: dict[str, int] = {}
        self.timesets_used_by_solves: defaultdict = defaultdict(list)
        self.invest_periods: defaultdict = defaultdict(list)
        self.realized_periods: defaultdict = defaultdict(list)
        self.realized_invest_periods: defaultdict = defaultdict(list)
        self.fix_storage_periods: defaultdict = defaultdict(list)

        # Mutable tracking — populated during the recursive solve loop.
        self.real_solves: list[str] = []
        self.first_of_complete_solve: list[str] = []
        self.last_of_solve: list[str] = []

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(
        cls, db: "DatabaseMapping", logger: logging.Logger
    ) -> "SolveConfig":
        """Read all solve-level parameters from *db* into a SolveConfig.

        Loading order is preserved exactly — see the four-step block at
        the bottom of this method.

        1. Basic params (model, solvers, rolling_times, …)
        2. ``make_roll_counter`` (needs ``solve_modes``).
        3. ``get_period_timesets`` (needs ``model_solve`` +
           ``contains_solves``; may call ``duplicate_solve``).
        4. Four ``periods_to_tuples`` calls
           (``invest_periods``, ``realized_periods``,
           ``realized_invest_periods``, ``fix_storage_periods``;
           may call ``duplicate_solve`` for 2D-Map values).
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
        # The seven params are read individually here and aggregated
        # into ``solver_configs`` below.  ``solver_options`` is a Map of
        # str→Any forwarded raw to the chosen solver; everything else is
        # a scalar default-None convenience knob (None means "no override").
        #
        # NOTE: ``solver`` is also read via the older ``solvers``
        # variable above for the legacy :class:`SolverSettings` dataclass
        # (which downstream callers still consume); we reuse that dict
        # here rather than re-querying.
        # Batch C.9 — ``solver_io_api`` DB axis removed.  Replaced by
        # the ``--matrix-file-format`` CLI flag (mps | lp),
        # env-var-plumbed via ``FLEXTOOL_MATRIX_FILE_FORMAT``.  The
        # in-process vs. file dispatch is implicit:
        # * HiGHS + no --save-memory: ``SolverConfig.io_api`` defaults
        #   to ``"direct"`` (in-process binding).
        # * HiGHS + --save-memory: ``Problem.solve(save_memory=True)``
        #   round-trips through MPS internally; ``io_api`` is ignored
        #   on that path.
        # * Commercial solver: ``polar_high.solvers.solve`` consults
        #   ``io_api``.  When the CLI flag is set its value
        #   (``"mps"`` or ``"lp"``) applies uniformly to every solve.
        # ``matrix_file_format`` stays an empty dict (no per-solve
        # author override since the DB axis is gone); the default is
        # resolved below using the env var when present, else
        # ``"direct"``.
        import os as _os_c9
        _cli_io_api = _os_c9.environ.get("FLEXTOOL_MATRIX_FILE_FORMAT")
        matrix_file_format: dict = {}
        # Batch C.7 — ``solver_log_level`` shortcut removed; use the
        # --solver-log-level CLI flag (silent / normal / verbose →
        # HiGHS output_flag + log_dev_level).
        solver_log_level: dict = {}
        # solver_mip_gap comes back as the stringified float that
        # ``params_to_dict`` produces for scalar floats (see line
        # ~141).  Batches C.6 and C.8 dropped the ``solver_threads``
        # and ``solver_time_limit`` DB axes (use --highs-threads and
        # --solver-time-limit CLI flags instead).
        solver_time_limit_raw: dict = {}
        solver_mip_gap_raw: dict = params_to_dict(
            db=db, cl="solve", par="solver_mip_gap", mode=DictMode.DICT
        )
        solver_threads_raw: dict = {}
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
        # _emit_solve_writers.derive_p_use_row_scaling), preserving
        # the Mode A pre-scaling behaviour for the row-scaling
        # capacity-proxy emitter.  The autoscaler's Layer 2 + Layer
        # 3 (driven by --scaling) are unaffected.
        use_row_scaling: dict = {}
        scale_the_objective: dict = params_to_dict(
            db=db, cl="solve", par="scale_the_objective", mode=DictMode.DICT
        )
        user_bound_scale: dict = params_to_dict(
            db=db, cl="solve", par="user_bound_scale", mode=DictMode.DICT
        )

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

        # Roll the seven v52 per-solve param dicts into a single
        # ``solver_configs[solve_name] -> SolverConfig`` mapping.  Keys
        # are the union of solve names appearing across any of the
        # seven dicts — a solve that authors *any* solver_* param gets
        # an explicit entry; solves with no override remain absent and
        # callers fall back to ``SolverConfig()`` defaults.
        # Batch C.2 — ``solver_options`` removed; the per-solve free-form
        # option dict for the commercial-solver path is sourced from
        # ``solver_arguments`` (the same 1d-map the HiGHS-side resolver
        # consumes).
        solver_config_keys = (
            set(solvers)
            | set(matrix_file_format)
            | set(solver_arguments)
            | set(solver_time_limit_raw)
            | set(solver_mip_gap_raw)
            | set(solver_threads_raw)
            | set(solver_log_level)
        )

        def _opt_float(raw: dict, key: str) -> float | None:
            v = raw.get(key)
            return float(v) if v is not None else None

        def _opt_int(raw: dict, key: str) -> int | None:
            v = raw.get(key)
            return int(float(v)) if v is not None else None

        # Batch C.9 — the CLI ``--matrix-file-format`` env var override
        # (when set, ``"mps"`` or ``"lp"``) becomes the default
        # ``SolverConfig.io_api`` for every solve.  Otherwise the
        # default is ``"direct"`` (HiGHS in-process binding; commercial
        # solvers' in-process Python API).
        _io_api_default = _cli_io_api if _cli_io_api in ("mps", "lp") else "direct"
        solver_configs: dict[str, SolverConfig] = {}
        for key in solver_config_keys:
            solver_configs[key] = SolverConfig(
                name=solvers.get(key, "highs"),
                io_api=matrix_file_format.get(key, _io_api_default),
                options=dict(solver_arguments.get(key, {})),
                time_limit=_opt_float(solver_time_limit_raw, key),
                mip_gap=_opt_float(solver_mip_gap_raw, key),
                threads=_opt_int(solver_threads_raw, key),
                log_level=solver_log_level.get(key, "normal"),
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
            user_bound_scale=user_bound_scale,
            solver_configs=solver_configs,
            decomposition=decomposition,
            benders_max_iter=benders_max_iter,
            benders_tolerance=benders_tolerance,
        )

        # Computed fields — loading order MUST be preserved exactly.
        # ``duplicate_solve`` mutates 19 sibling dicts in lockstep, so
        # any reordering desyncs them and downstream reads silently
        # produce empty/zero results.
        obj.roll_counter = obj.make_roll_counter()
        obj.timesets_used_by_solves = obj.get_period_timesets(db=db)
        obj.invest_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="invest_periods"
        )
        obj.realized_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="realized_periods"
        )
        obj.realized_invest_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="realized_invest_periods"
        )
        obj.fix_storage_periods = obj.periods_to_tuples(
            db=db, cl="solve", par="fix_storage_periods"
        )

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
        from spinedb_api import DatabaseMapping
        from spinedb_api.filters.scenario_filter import (
            apply_scenario_filter_to_subqueries,
        )

        if logger is None:
            logger = logging.getLogger(f"flextool.engine_polars.solve_config[{scenario}]")
        url = str(db_url)
        if "://" not in url:
            url = f"sqlite:///{url}"
        with DatabaseMapping(url) as db:
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

        Currently only :class:`flextool.engine_polars._spinedb_reader.SpineDbReader`
        sources are supported — they expose the underlying ``db_url`` and
        ``scenario`` so the canonical :meth:`load_from_db_url` path can
        be reused.  In-memory and CSV-backed sources for solve-class
        parameters are out of Γ.8.A scope; Γ.8.D wires those once the
        chain.run_chain integration needs them.
        """
        # Late import: SpineDbReader brings spinedb_api into the import
        # graph, but we only need it for the isinstance check.
        from flextool.engine_polars._spinedb_reader import SpineDbReader

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
        """Read ``period_timeset`` parameters for every active solve.

        May call :meth:`duplicate_solve` when a solve carries a
        2D-Map-shaped ``period_timeset`` parameter (one input solve fans
        out into one solve per outer-Map key).
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
                        new_name = (
                            param["entity_name"] + "_" + param_value.indexes[i]
                        )
                        self.duplicate_solve(param["entity_name"], new_name)
                        timesets_used_by_solves[new_name].append(
                            (
                                param_value.values[i].indexes[i],
                                param_value.values[i].values[i],
                            )
                        )
                    else:
                        timesets_used_by_solves[param["entity_name"]].append(
                            (
                                param_value.indexes[i],
                                param_value.values[i],
                            )
                        )
        return timesets_used_by_solves

    def duplicate_solve(
        self,
        old_solve: str,
        new_name: str,
        update_model_solves: bool = True,
    ) -> None:
        """Duplicate every solve-level dict entry from *old_solve* under
        *new_name*.

        Mutates 19 sibling defaultdicts (and ``model_solve`` when
        *update_model_solves* is set) so downstream readers can address
        the duplicated solve transparently.

        ``update_model_solves=False`` is used by the rolling builder
        (Γ.8.C) where roll-named sub-solves should NOT replace their
        parent in ``model_solve``.
        """
        if (
            new_name not in self.model_solve.values()
            and new_name not in self.contains_solves.values()
        ):
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
                self.realized_periods,
                self.realized_invest_periods,
                self.invest_periods,
                self.fix_storage_periods,
                self.decomposition,
                self.benders_max_iter,
                self.benders_tolerance,
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

    def benders_config_for(self, solve_name: str) -> tuple[int, float]:
        """Resolve ``(max_iter, tol)`` for *solve_name*.

        Falls back to the schema defaults (50 / 1e-3) for any knob the
        solve does not author.  ``params_to_dict`` stores scalar floats
        as ``str(float)``, so values are coerced through ``float`` here;
        ``max_iter`` is additionally rounded to ``int``.
        """
        max_iter_raw = self.benders_max_iter.get(solve_name)
        tol_raw = self.benders_tolerance.get(solve_name)
        max_iter = int(float(max_iter_raw)) if max_iter_raw is not None else 50
        tol = float(tol_raw) if tol_raw is not None else 1e-3
        return max_iter, tol

    def periods_to_tuples(
        self,
        db: "DatabaseMapping",
        cl: str,
        par: str,
    ) -> defaultdict:
        """Read period-shaped solve parameters as ``[(p_from, p_in), …]``.

        For 1D Array values (e.g. ``realized_periods=["p2020","p2025"]``)
        each element ``p`` becomes ``(p, p)``.

        For 2D Map values (e.g. ``invest_periods`` under the
        ``invest_twoYears4Times_5weeks`` scenario where one solve ladders
        an "invest in p2020 covers p2020+p2025" pattern) the outer Map
        triggers :meth:`duplicate_solve` and per-(outer, inner) tuples
        flow into the new solve's entry.  Inner shape is required to be
        Map-of-scalars; mixed 1D/2D with the same name raises.
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
                for i, row in enumerate(param_value.values):
                    if isinstance(param_value.values[i], api.Map):
                        # 2D Map: outer index → inner Map of (p, "yes")-ish.
                        for j, _row2 in enumerate(row.values):
                            if isinstance(param_value.values[j], api.Map):
                                new_name = (
                                    param["entity_name"]
                                    + "_"
                                    + param_value.indexes[i]
                                )
                                self.duplicate_solve(
                                    param["entity_name"], new_name
                                )
                                result_dict[new_name].append(
                                    (
                                        param_value.indexes[i],
                                        param_value.values[i].indexes[j],
                                    )
                                )
                                # Re-shape ``timesets_used_by_solves`` for
                                # the duplicated solve: keep only entries
                                # whose period matches the inner index.
                                new_period_timeset_list = []
                                for solve, period__timeset_list in list(
                                    self.timesets_used_by_solves.items()
                                ):
                                    if solve != param["entity_name"]:
                                        continue
                                    for period__timeset in period__timeset_list:
                                        if (
                                            period__timeset[0]
                                            == param_value.values[i].indexes[j]
                                        ):
                                            new_period_timeset_list.append(
                                                period__timeset
                                            )
                                if (
                                    new_name
                                    not in self.timesets_used_by_solves.keys()
                                ):
                                    self.timesets_used_by_solves[new_name] = (
                                        new_period_timeset_list
                                    )
                                else:
                                    for item in new_period_timeset_list:
                                        if (
                                            item
                                            not in self.timesets_used_by_solves[
                                                new_name
                                            ]
                                        ):
                                            self.timesets_used_by_solves[
                                                new_name
                                            ].append(item)
                            else:
                                raise FlexToolConfigError(
                                    "periods_to_tuple function handles only "
                                    f"arrays or 2d maps: {entity}, {param}"
                                )
                    else:
                        result_dict[param["entity_name"]].append((row, row))
        return result_dict


__all__ = [
    "DictMode",
    "HiGHSConfig",
    "SolverConfig",
    "SolverSettings",
    "SolveConfig",
    "get_single_entities",
    "params_to_dict",
]
