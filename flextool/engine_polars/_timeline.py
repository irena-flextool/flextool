"""Timeline-level configuration loaded from a SpineDB scenario.

Γ.8.B foundation module.  Direct 1:1 port of
``flextool/flextoolrunner/timeline_config.py`` (808 LOC, the read-only
reference at ``flextool-engine/flextool/flextoolrunner/timeline_config.py``).

Architecture notes
------------------

* Algorithm parity with flextool is the contract — every downstream
  module (recursive solve builder, stochastic, orchestration loop)
  reads ``state.timeline.<dict>`` keys produced here and the keys must
  match exactly.  Quirks preserved:

  - ``timeset_weights`` and ``representative_period_weights`` decoders
    accept BOTH :class:`spinedb_api.Map` and ``list[tuple]`` shapes
    (Spine API quirk; see ``audit/solve_orchestration_plan.md §3.2``).
  - ``new_step_durations`` is solve-scoped as of DB v50 (was
    timeset-scoped pre-v50; ``update_flextool/db_migration.py`` upgrades
    older DBs before this loader runs).
  - The six-rule fallback chain in :meth:`TimelineConfig.create_assumptive_parts`
    mutates *both* ``self`` AND the supplied ``solve_config`` in three
    of its rules (lines 309, 322-336, 348-358 of the flextool source).
    Order matters; idempotency is achieved through the membership
    checks each rule applies before mutating.

* :func:`get_active_time` is the most-called free function — every
  recursive solve invocation walks it.  It reads :attr:`timesets`
  (``list[str]``) but the parameter is sometimes a *dict* of timeset
  durations passed by the orchestration layer (see
  ``recursive_solves.py:479``); the function tolerates either by
  taking timeset durations as the second argument.

* Files: ``create_averaged_timeseries`` aggregates per-(``solve``)
  timeline data by averaging or summing matched timesteps.  The list
  of files lives at lines 405-422 of the flextool source; the
  ``pt_node_inflow.csv`` / ``pbt_node_inflow.csv`` files are summed,
  every other ``pt_*.csv`` / ``pbt_*.csv`` is averaged.  The
  ``storage_state_reference_value`` row in any of those files bypasses
  aggregation entirely (line 474 of the source) — preserved verbatim.

* Refactor of ``_derived_params.py::dt_and_step_duration_from_source``
  to delegate to this module: that helper is polars-lazy and consumes
  the per-(entity_class, parameter_name) ``InputSource`` Protocol;
  this module ports the legacy ``flextoolrunner/timeline_config.py``
  load_from_db path which uses ``params_to_dict`` directly.  The two
  paths produce identical timeline / timeset state (parity sweep
  validates this on every fixture).  Γ.8.D's review (see
  ``audit/solve_orchestration_plan.md §Γ.8.D``) confirms convergence
  is unnecessary: ``dt_and_step_duration_from_source`` already covers
  every shape this module handles, including stochastic-branch
  periods broadcasting (lines 203-219).  The two paths coexist
  because they serve different consumers — the InputSource one feeds
  Param-frame helpers (Γ.3 era), this one feeds the orchestration
  loop (Γ.8 era).

Reference: ``flextool/flextoolrunner/timeline_config.py`` (read-only
mirror at ``flextool-engine/flextool/flextoolrunner/``).
"""
from __future__ import annotations

import csv
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import spinedb_api as api

from flextool.engine_polars._solve_config import (
    DictMode,
    get_single_entities,
    params_to_dict,
)
from flextool.engine_polars._solve_state import (
    ActiveTimeEntry,
    FlexToolConfigError,
)

if TYPE_CHECKING:
    from spinedb_api import DatabaseMapping

    from flextool.engine_polars._solve_config import SolveConfig


# ---------------------------------------------------------------------------
# TimelineConfig — main container
# ---------------------------------------------------------------------------


class TimelineConfig:
    """All timeline definitions and timeset mappings for a flexpy run.

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
    timeset_weights : dict[str, dict[str, float]]
        timeset_name -> {timestep: weight}.  Raw user input; runner
        normalizes during ``write_timeset_cost_weight``.
    stochastic_timesteps : defaultdict[str, list[tuple]]
        Mutable — populated during the solve loop (Γ.8.C).
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
        self.timeset_weights: dict[str, dict[str, float]] = (
            timeset_weights or {}
        )

        # Mutable state — populated later by orchestration / stochastic.
        self.stochastic_timesteps: defaultdict[str, list] = defaultdict(list)
        self.original_timeline: defaultdict[str, str] = defaultdict()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(
        cls, db: "DatabaseMapping", logger: logging.Logger
    ) -> "TimelineConfig":
        """Read all timeline-level parameters from *db* into a TimelineConfig.

        Mirrors :meth:`flextool.flextoolrunner.timeline_config.TimelineConfig.load_from_db`
        (lines 80-155 of the source).  Reads:

        * ``timeline.timestep_duration`` — base timeline definitions.
        * ``timeset`` entities — flat list.
        * ``timeset.timeset_duration`` — (start, count) blocks per timeset.
        * ``timeset.timeline`` — timeset → timeline mapping.
        * ``solve.new_stepduration`` — per-solve aggregation override
          (DB v50+; older DBs are upgraded by db_migration first).
        * ``timeset.representative_period_weights`` — nested Map decoder
          (handles BOTH ``api.Map`` and ``list[tuple]`` shapes).
        * ``timeset.timeset_weights`` — flat Map decoder (same dual-
          shape handling).
        """
        timelines: defaultdict = params_to_dict(
            db=db,
            cl="timeline",
            par="timestep_duration",
            mode=DictMode.DEFAULTDICT,
        )
        timesets: list = get_single_entities(
            db=db, entity_class_name="timeset"
        )
        timeset_durations: defaultdict = params_to_dict(
            db=db,
            cl="timeset",
            par="timeset_duration",
            mode=DictMode.DEFAULTDICT,
        )
        timesets__timeline: defaultdict = params_to_dict(
            db=db, cl="timeset", par="timeline", mode=DictMode.DEFAULTDICT
        )
        new_step_durations: dict = params_to_dict(
            db=db,
            cl="solve",
            par="new_stepduration",
            mode=DictMode.DICT,
        )
        # Representative period weights: nested Map
        # ``base_start -> {rep_start -> weight}``.  ``params_to_dict``
        # returns the raw spinedb value because the inner is a Map
        # (the DICT mode doesn't auto-flatten nested Map-of-Map).
        rp_weights_raw = params_to_dict(
            db=db,
            cl="timeset",
            par="representative_period_weights",
            mode=DictMode.DICT,
        )
        rp_weights: dict = _decode_rp_weights(rp_weights_raw)

        # Flat Map: ``timestep -> weight``.  Used for non-RP per-step
        # cost/slack weighting.  Decodes BOTH ``api.Map`` and
        # ``list[tuple]`` shapes (Spine API quirk).
        timeset_weights_raw = params_to_dict(
            db=db,
            cl="timeset",
            par="timeset_weights",
            mode=DictMode.DICT,
        )
        timeset_weights: dict[str, dict[str, float]] = (
            _decode_timeset_weights(timeset_weights_raw)
        )
        return cls(
            timelines=timelines,
            timesets=timesets,
            timesets__timeline=timesets__timeline,
            timeset_durations=timeset_durations,
            new_step_durations=new_step_durations,
            rp_weights=rp_weights,
            timeset_weights=timeset_weights,
        )

    @classmethod
    def load_from_db_url(
        cls,
        db_url: str,
        scenario: str,
        logger: logging.Logger | None = None,
    ) -> "TimelineConfig":
        """Convenience factory: open *db_url*, apply *scenario*, load.

        Mirrors :meth:`flextool.engine_polars._solve_config.SolveConfig.load_from_db_url`
        — same pattern.
        """
        from spinedb_api import DatabaseMapping
        from spinedb_api.filters.scenario_filter import (
            apply_scenario_filter_to_subqueries,
        )

        if logger is None:
            logger = logging.getLogger(f"flexpy.timeline[{scenario}]")
        url = str(db_url)
        if not url.startswith("sqlite:") and not url.startswith("postgresql"):
            url = f"sqlite:///{url}"
        with DatabaseMapping(url) as db:
            apply_scenario_filter_to_subqueries(db, scenario)
            # Pre-warm caches — see SolveConfig.load_from_db_url for
            # the rationale.  Measured 1.76× speedup on large customer
            # DBs (H2_trade), insignificant noise on small test DBs.
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            return cls.load_from_db(db, logger)

    @classmethod
    def load_from_source(
        cls,
        source: object,
        logger: logging.Logger | None = None,
    ) -> "TimelineConfig":
        """Load via the :class:`InputSource` Protocol (Γ.8.D wiring).

        Currently only :class:`flextool.engine_polars._spinedb_reader.SpineDbReader`
        sources are supported — they expose ``db_url`` and ``scenario`` so
        the canonical :meth:`load_from_db_url` path can be reused.
        In-memory and CSV-backed sources are deferred to Γ.8.D.
        """
        from flextool.engine_polars._spinedb_reader import SpineDbReader

        if isinstance(source, SpineDbReader):
            return cls.load_from_db_url(
                source.db_url, source.scenario, logger=logger
            )
        raise NotImplementedError(
            f"TimelineConfig.load_from_source does not yet support "
            f"{type(source).__name__!r} sources.  Use load_from_db / "
            f"load_from_db_url with a Spine DB for now; the in-memory and "
            f"CSV adapters land in Γ.8.D when chain.run_chain is rewired."
        )

    # ------------------------------------------------------------------
    # Methods (1:1 port from ``flextoolrunner/timeline_config.py``)
    # ------------------------------------------------------------------

    def create_timeline_from_timestep_duration(
        self, solve_config: "SolveConfig"
    ) -> None:
        """Synthesise per-solve aggregated timelines.

        For each solve that sets ``new_stepduration``, build a freshly
        aggregated timeline ``"{timeline}_{solve}"`` and re-point every
        timeset the solve uses at that new timeline.  Preserves the
        ``original_timeline`` map so :meth:`create_averaged_timeseries`
        can find the unaggregated source.

        Mirrors lines 161-236 of the flextool source.

        A defensive guard (``"fail loudly"`` per
        ``audit/handoff_post_split_todo.md``): if the same timeset is
        shared between two solves with *different* ``new_stepduration``
        values, raise rather than silently overwrite.  Flextool's source
        comment claims this is "rejected at migration time" but doesn't
        check; flexpy enforces the invariant.
        """
        logger = logging.getLogger(__name__)

        # Defensive check for "shared timeset, conflicting new_stepduration".
        # Walk all (solve, timeset) pairs and assert at most one
        # ``new_stepduration`` value per timeset.
        timeset_to_step_duration: dict[str, tuple[str, float]] = {}
        for solve_name, step_duration_raw in self.new_step_durations.items():
            if step_duration_raw is None:
                continue
            sd = float(step_duration_raw)
            for _period, ts in solve_config.timesets_used_by_solves.get(
                solve_name, []
            ):
                prev = timeset_to_step_duration.get(ts)
                if prev is None:
                    timeset_to_step_duration[ts] = (solve_name, sd)
                elif prev[1] != sd:
                    message = (
                        f"Timeset '{ts}' is shared between solves "
                        f"'{prev[0]}' (new_stepduration={prev[1]}) and "
                        f"'{solve_name}' (new_stepduration={sd}); "
                        f"flexpy needs one timeline per timeset."
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)

        for solve_name, step_duration_raw in self.new_step_durations.items():
            if step_duration_raw is None:
                continue
            step_duration = float(step_duration_raw)
            period_timesets = solve_config.timesets_used_by_solves.get(
                solve_name, []
            )
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
                    step_counter = 0.0
                    last_index = first_index + int(float(ts[1]))
                    added_steps = 0
                    for step in old_steps[first_index:last_index]:
                        if step_counter >= step_duration:
                            new_steps.append((first_step, str(step_counter)))
                            first_step = step[0]
                            step_counter = 0.0
                            added_steps += 1
                            if step_counter > step_duration:
                                logger.warning(
                                    "Warning: All new steps are not the "
                                    "size of the given step duration. The "
                                    "new step duration has to be multiple "
                                    "of old step durations for this to "
                                    "happen."
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

    def create_assumptive_parts(self, solve_config: "SolveConfig") -> None:
        """Fill in missing time-structure parts from assumptions when
        the input data is incomplete.

        Six fallback rules (lines 238-382 of the flextool source):

        1. No timeset → synthesise from a single timeline.
        2. No ``timeset_durations`` → full-timeline timeset.
        3. No ``timesets__timeline`` mapping → use the sole timeline.
        4. No ``model:solves`` → use the sole solve (mutates
           ``solve_config.model_solve``).
        5. No ``period_timeset`` → use the sole timeset (mutates
           ``solve_config.timesets_used_by_solves`` and
           ``solve_config.realized_periods``).
        6. No ``solve_period_years_represented`` → 1 year per period
           (mutates ``solve_config.solve_period_years_represented``).

        The rules mutate ``self`` AND ``solve_config`` in-place.  Each
        rule's guard checks whether its target is already populated;
        running :meth:`create_assumptive_parts` twice on the same pair
        is a no-op (idempotent).

        Mirrors :meth:`flextool.flextoolrunner.timeline_config.TimelineConfig.create_assumptive_parts`.
        """
        logger = logging.getLogger(__name__)

        # Rule 1: No timeset → synthesise "full_timeline" from sole timeline.
        if not self.timesets and len(self.timelines) == 1:
            timeset_name = "full_timeline"
            self.timesets = list(timeset_name)

        # Rule 2: No timeset_durations + sole timeline → block the whole
        # timeline as one timeset_duration entry.
        for timeset_name in self.timesets:
            if not self.timeset_durations and len(self.timelines) == 1:
                self.timesets__timeline[timeset_name] = list(
                    self.timelines.keys()
                )[0]
                self.timeset_durations[timeset_name] = [
                    (
                        list(self.timelines.values())[0][0][0],
                        len(list(self.timelines.values())[0]),
                    )
                ]

        # Rule 3: timeset:timeline missing + sole timeline → use it.
        # If multiple timelines exist, the user must declare which one
        # this timeset uses — fail loudly.
        for timeset_name, _block in self.timeset_durations.items():
            if timeset_name not in self.timesets__timeline:
                if len(self.timelines) == 1:
                    self.timesets__timeline[timeset_name] = list(
                        self.timelines.keys()
                    )[0]
                elif len(self.timelines) > 1:
                    message = (
                        "More than one timeline available and FlexTool "
                        "does not know which ones to use. Please use "
                        "'timeline' parameter of 'timeset' class to "
                        "define which timelines are part of the "
                        "timeset(s) in the model instance"
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)

        # Rule 4: No model:solves → pick the sole solve (or fail).
        # MUTATES solve_config.model_solve.
        if not solve_config.model_solve:
            if len(solve_config.model) == 1:
                solve = None
                if len(solve_config.timesets_used_by_solves) == 1:
                    solve = list(
                        solve_config.timesets_used_by_solves.keys()
                    )[0]
                elif len(solve_config.timesets_used_by_solves) > 1:
                    message = (
                        "Data contains multiple solve entities and "
                        "FlexTool does not know which to use. Please use "
                        "'solves' parameter of 'model' class to inform "
                        "which solves are to be included in the model "
                        "instance."
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)
                all_solves_in_periods = (
                    set(solve_config.realized_periods.keys())
                    | set(solve_config.invest_periods.keys())
                )
                for solve_name in all_solves_in_periods:
                    if solve:
                        if solve_name != solve:
                            message = (
                                "Data contains multiple solve entities "
                                "and FlexTool does not know which to "
                                "use. Please use 'solves' parameter of "
                                "'model' class to inform which solves "
                                "are to be included in the model "
                                "instance."
                            )
                            logger.error(message)
                            raise FlexToolConfigError(message)
                    else:
                        solve = solve_name
                solve_config.model_solve[solve_config.model[0]] = [solve]
            else:
                message = (
                    "More than one model entity found in the database "
                    "and FlexTool does not know which to use."
                )
                logger.error(message)
                raise FlexToolConfigError(message)

        # Rule 5a: Solve listed but no realized_periods/invest_periods/
        # timesets_used_by_solves → fall back to model.periods_available.
        # MUTATES solve_config.realized_periods.
        for model, solves in solve_config.model_solve.items():
            for solve in solves:
                if (
                    solve not in solve_config.realized_periods
                    and solve not in solve_config.invest_periods
                    and solve not in solve_config.timesets_used_by_solves
                ):
                    if model in solve_config.periods_available:
                        for period in solve_config.periods_available[model]:
                            solve_config.realized_periods[solve].append(
                                (period, period)
                            )
                    else:
                        message = (
                            f"The solve {solve} in the model: solves "
                            f"array does not have any periods defined: "
                            f"(period_timeset, realized_periods, "
                            f"invest_periods)\nAlternatively add "
                            f"periods_available to the model to create "
                            f"simple full timelines for those periods"
                        )
                        logger.error(message)
                        raise FlexToolConfigError(message)

        # Rule 5b: solve has no period_timeset but a single timeset
        # exists → use it.  MUTATES timesets_used_by_solves.
        for solve in list(solve_config.model_solve.values())[0]:
            if solve not in list(
                solve_config.timesets_used_by_solves.keys()
            ):
                if len(self.timeset_durations) == 1:
                    period__timeset_list: list[tuple[str, str]] = []
                    timeset_name = list(self.timeset_durations.keys())[0]
                    for period_tuple in (
                        solve_config.invest_periods.get(solve, [])
                        + solve_config.realized_periods.get(solve, [])
                    ):
                        period__timeset_list.append(
                            (period_tuple[1], timeset_name)
                        )
                    solve_config.timesets_used_by_solves[solve] = (
                        period__timeset_list
                    )
                else:
                    message = (
                        "More than one timeset available and FlexTool "
                        "does not know which ones to use. Please use "
                        "'period_timeset' parameter of 'solve' class to "
                        "define which periods and which timesets are "
                        "part of the solve(s) in the model instance"
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)

        # Rule 5c: solve has period__timeset but no realized/invest/
        # contains_solves → assume all in period_timeset are realized.
        # MUTATES solve_config.realized_periods.
        for solve in list(solve_config.model_solve.values())[0]:
            if (
                solve not in solve_config.realized_periods
                and solve not in solve_config.invest_periods
                and not solve_config.contains_solves[solve]
                and solve in solve_config.timesets_used_by_solves
            ):
                for period_timeset in solve_config.timesets_used_by_solves[
                    solve
                ]:
                    solve_config.realized_periods[solve].append(
                        (period_timeset[0], period_timeset[0])
                    )

        # Rule 6: solve_period_years_represented missing → 1 year/period.
        # MUTATES solve_config.solve_period_years_represented.
        for solve in list(solve_config.model_solve.values())[0]:
            all_periods_tuples = (
                solve_config.realized_periods[solve]
                + solve_config.invest_periods[solve]
            )
            all_periods = {item for tup in all_periods_tuples for item in tup}
            for period in all_periods:
                if solve not in solve_config.solve_period_years_represented:
                    solve_config.solve_period_years_represented[solve].append(
                        [period, 1.0]
                    )

    def create_averaged_timeseries(
        self,
        solve: str,
        solve_config: "SolveConfig",
        logger: logging.Logger,
        work_folder: "Path | None" = None,
    ) -> None:
        """Average or sum input timeseries to match a coarsened timeline.

        If ``solve`` does not set ``new_stepduration`` (solve-scoped as
        of DB v50), copy ``input/pt_*.csv`` files unchanged into
        ``solve_data/``.  Otherwise re-aggregate each timeseries file
        to match the new step size.

        Aggregation rules (lines 405-422 of the flextool source):

        * ``pt_node_inflow.csv`` and ``pbt_node_inflow.csv`` →
          ``"sum"`` (inflow energies sum across coarsened steps).
        * Every other ``pt_*.csv`` / ``pbt_*.csv`` → ``"average"``.
        * Rows with parameter ``"storage_state_reference_value"`` are
          ALWAYS bypassed (line 474) — they're per-step state targets
          rather than time-integrated quantities.

        After file aggregation, ``input/p_node.csv``'s ``inflow`` rows
        are appended to ``solve_data/pt_node_inflow.csv`` as one entry
        per new timestep, scaled by the new step's duration (lines
        522-543 of the source).

        Mirrors :meth:`flextool.flextoolrunner.timeline_config.TimelineConfig.create_averaged_timeseries`.

        Args:
            solve: Active solve name.
            solve_config: SolveConfig containing the
                ``timesets_used_by_solves`` map.
            logger: Logger for error reporting.
            work_folder: Working directory.  When provided, ``input/``
                and ``solve_data/`` are resolved under it; defaults to
                the current working directory.
        """
        wf = Path(work_folder) if work_folder is not None else Path.cwd()
        timeseries_map: dict[str, str] = {
            "pt_node_inflow.csv": "sum",
            "pt_commodity.csv": "average",
            "pt_group.csv": "average",
            "pt_node.csv": "average",
            "pt_process.csv": "average",
            "pt_profile.csv": "average",
            "pt_process_source.csv": "average",
            "pt_process_sink.csv": "average",
            "pt_reserve__upDown__group.csv": "average",
            "pbt_node_inflow.csv": "sum",
            "pbt_node.csv": "average",
            "pbt_process.csv": "average",
            "pbt_profile.csv": "average",
            "pbt_process_source.csv": "average",
            "pbt_process_sink.csv": "average",
            "pbt_reserve__upDown__group.csv": "average",
        }
        # As of DB v50 new_stepduration is keyed by solve.  ``None`` is
        # the parameter_definition default in the master template; treat
        # it as "not set" and skip aggregation.
        create = (
            solve in self.new_step_durations
            and self.new_step_durations[solve] is not None
        )
        if not create:
            for timeseries in timeseries_map:
                shutil.copy(
                    str(wf / "input" / timeseries),
                    str(wf / "solve_data" / timeseries),
                )
            return

        # All timesets used by *solve* must resolve to the same
        # timeline — new_stepduration rewrites that timeline once per
        # solve and we re-aggregate every timeseries row by it.
        timelines_list: list[str] = []
        for _period, timeset in solve_config.timesets_used_by_solves[solve]:
            timeline = self.timesets__timeline[timeset]
            if timeline not in timelines_list:
                if len(timelines_list) != 0:
                    message = (
                        f"solve '{solve}' sets new_stepduration but its "
                        f"timesets resolve to more than one timeline; "
                        f"new_stepduration requires a single shared "
                        f"timeline per solve."
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)
                timelines_list.append(timeline)
        # Pre-build timeline duration lookup for O(1) access per row.
        timeline_duration_lookup: dict[str, int] = {}
        for timeline in timelines_list:
            new_timeline = self.timelines[timeline]
            for timeline_row in new_timeline:
                timeline_duration_lookup[timeline_row[0]] = int(
                    float(timeline_row[1])
                )
        for timeseries in timeseries_map:
            input_path = wf / "input" / timeseries
            output_path = wf / "solve_data" / timeseries
            with open(input_path, "r", encoding="utf-8") as blk:
                filereader = csv.reader(blk, delimiter=",")
                with open(output_path, "w", newline="") as solve_file:
                    filewriter = csv.writer(solve_file, delimiter=",")
                    headers = next(filereader)
                    filewriter.writerow(headers)
                    time_index = headers.index("time")
                    while True:
                        try:
                            datain = next(filereader)
                        except StopIteration:
                            break
                        timeline_step_duration = (
                            timeline_duration_lookup.get(datain[time_index])
                        )
                        if timeline_step_duration is not None:
                            values: list[float] = []
                            params = datain[0:time_index]
                            row = datain[0 : time_index + 1]
                            values.append(float(datain[time_index + 1]))
                            if datain[1] != "storage_state_reference_value":
                                for _i in range(timeline_step_duration - 1):
                                    try:
                                        datain = next(filereader)
                                    except StopIteration:
                                        break
                                    if datain[0:time_index] != params:
                                        message = (
                                            "Cannot find the same "
                                            "timesteps in input data as "
                                            "in timeline for file  "
                                            + timeseries
                                            + " after "
                                            + row[-1]
                                        )
                                        logger.error(message)
                                        raise FlexToolConfigError(message)
                                    values.append(
                                        float(datain[time_index + 1])
                                    )

                            if timeseries_map[timeseries] == "average":
                                out_value = round(
                                    sum(values) / len(values), 6
                                )
                            else:
                                out_value = sum(values)
                            row.append(out_value)
                            filewriter.writerow(row)
                        else:
                            # Bypass aggregation for storage_state_reference_value
                            # rows — pin to the new timeline's nearest step
                            # at-or-before the original target step.
                            if datain[1] == "storage_state_reference_value":
                                counter = 0
                                current_index = 0
                                for timestep in self.timelines[
                                    self.original_timeline[timeline]
                                ]:
                                    if datain[2] == timestep[0]:
                                        current_index = counter
                                    counter += 1
                                found = False
                                new_index = None
                                for timestep in reversed(
                                    self.timelines[
                                        self.original_timeline[timeline]
                                    ][0 : current_index + 1]
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

        # Append per-step inflow contributions from p_node (the
        # constant-inflow Param) to pt_node_inflow.csv, scaled by each
        # new step's duration.
        node__inflow: list[list[str]] = []
        with open(wf / "input/p_node.csv", "r", encoding="utf-8") as blk:
            filereader = csv.reader(blk, delimiter=",")
            _read_header = next(filereader)
            while True:
                try:
                    datain = next(filereader)
                except StopIteration:
                    break
                if datain[1] == "inflow":
                    node__inflow.append([datain[0], datain[2]])
        with open(
            wf / "solve_data/pt_node_inflow.csv", "a", newline=""
        ) as blk:
            filewriter = csv.writer(blk, delimiter=",")
            for timeline in timelines_list:
                new_timeline = self.timelines[timeline]
                for node__value in node__inflow:
                    for timeline_row in new_timeline:
                        timeline_step_duration = int(float(timeline_row[1]))
                        value = (
                            float(node__value[1]) * timeline_step_duration
                        )
                        row_out = [node__value[0], timeline_row[0], value]
                        filewriter.writerow(row_out)


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _decode_rp_weights(rp_weights_raw: dict) -> dict:
    """Decode :attr:`TimelineConfig.rp_weights`.

    Spine API quirk: a ``Map`` of ``Map`` may come back from
    ``params_to_dict`` either as :class:`spinedb_api.Map` (when the
    outer Map is read directly) or as ``list[tuple]`` (when the outer
    is flattened via ``convert_map_to_table``).  Both shapes must
    decode to the same nested ``{base: {rep: weight}}`` dict.

    Mirrors lines 100-130 of the flextool source.
    """
    rp_weights: dict = {}
    for timeset_name, nested_map in rp_weights_raw.items():
        if isinstance(nested_map, api.Map):
            weight_dict: dict[str, dict[str, float]] = {}
            for i, base_key in enumerate(nested_map.indexes):
                inner = nested_map.values[i]
                if isinstance(inner, api.Map):
                    weight_dict[str(base_key)] = {
                        str(k): float(v)
                        for k, v in zip(inner.indexes, inner.values)
                    }
            rp_weights[timeset_name] = weight_dict
        elif isinstance(nested_map, list):
            # ``convert_map_to_table`` shape: [(base, inner_data), ...]
            weight_dict = {}
            for entry in nested_map:
                if len(entry) >= 2:
                    base_key = str(entry[0])
                    inner_data = entry[1]
                    if isinstance(inner_data, list):
                        weight_dict[base_key] = {
                            str(k): float(v) for k, v in inner_data
                        }
                    elif isinstance(inner_data, api.Map):
                        weight_dict[base_key] = {
                            str(k): float(v)
                            for k, v in zip(
                                inner_data.indexes, inner_data.values
                            )
                        }
            if weight_dict:
                rp_weights[timeset_name] = weight_dict
    return rp_weights


def _decode_timeset_weights(
    timeset_weights_raw: dict,
) -> dict[str, dict[str, float]]:
    """Decode :attr:`TimelineConfig.timeset_weights`.

    Like :func:`_decode_rp_weights` but flat (one level of Map).
    Handles both ``api.Map`` and ``list[tuple]`` shapes.

    Mirrors lines 132-146 of the flextool source.
    """
    timeset_weights: dict[str, dict[str, float]] = {}
    for timeset_name, flat_map in timeset_weights_raw.items():
        if isinstance(flat_map, api.Map):
            timeset_weights[timeset_name] = {
                str(k): float(v)
                for k, v in zip(flat_map.indexes, flat_map.values)
            }
        elif isinstance(flat_map, list):
            timeset_weights[timeset_name] = {
                str(entry[0]): float(entry[1])
                for entry in flat_map
                if len(entry) >= 2
            }
    return timeset_weights


# ---------------------------------------------------------------------------
# Free functions (pure — no side effects beyond return values / file I/O)
# ---------------------------------------------------------------------------


def get_active_time(
    current_solve: str,
    timesets_used_by_solves: dict,
    timesets: dict,
    timelines: dict,
    timesets__timelines: dict,
) -> defaultdict:
    """Map periods to their corresponding timeline entries for a solve.

    Returns a defaultdict mapping period IDs to lists of
    :class:`ActiveTimeEntry` records.  The recursive solve builder
    (Γ.8.C) calls this at every node of its tree walk; the shape is
    load-bearing for downstream :func:`make_step_jump` and
    :func:`make_period_block` callers.

    Args:
        current_solve: Active solve name.
        timesets_used_by_solves: ``solve -> [(period, timeset), ...]``.
        timesets: ``timeset -> [(start, count), ...]`` (this is the
            ``timeset_durations`` field of :class:`TimelineConfig`,
            despite the parameter name).
        timelines: ``timeline -> [(timestep, duration), ...]``.
        timesets__timelines: ``timeset -> timeline``.

    Mirrors :func:`flextool.flextoolrunner.timeline_config.get_active_time`.
    """
    active_time: defaultdict = defaultdict(list)

    if current_solve not in timesets_used_by_solves:
        raise ValueError(
            f"{current_solve}: this solve does not have period_timeset "
            "defined. Check that it has period_timeset parameter "
            "defined and the names of period and timeset are spelled "
            "correctly (case sensitive). Check that the alternative is "
            "included in the scenario."
        )

    # Pre-build index for O(1) timestep-to-position lookup per timeline.
    timeline_index_cache: dict[str, dict[str, int]] = {}

    for period, timeset_id in timesets_used_by_solves[current_solve]:
        timeline_id = timesets__timelines.get(timeset_id)
        if not timeline_id:
            continue
        timeline_data = timelines.get(timeline_id, [])
        if not timeline_data:
            continue
        if timeline_id not in timeline_index_cache:
            timeline_index_cache[timeline_id] = {
                time_val: idx
                for idx, (time_val, _) in enumerate(timeline_data)
            }
        time_val_to_idx = timeline_index_cache[timeline_id]
        for start_time, duration in timesets[timeset_id]:
            idx = time_val_to_idx.get(start_time)
            if idx is not None:
                for step in range(int(float(duration))):
                    if idx + step < len(timeline_data):
                        entry = timeline_data[idx + step]
                        active_time[period].append(
                            ActiveTimeEntry(
                                timestep=entry[0],
                                index=idx + step,
                                duration=entry[1],
                            )
                        )

    if not active_time:
        raise ValueError(
            f"{current_solve}: Failed to map timeset to timeline. "
            "Check that all timeset entities have timeline parameter "
            "defined and the name of the timeline is spelled correctly "
            "(case sensitive). Check that the alternative of the "
            "timeline parameter is included in the modelled scenario."
        )

    return active_time


def make_step_jump(
    active_time_list: dict,
    period__branch: list[tuple],
    solve_branch__time_branch_list: list[tuple],
) -> list[tuple]:
    """Build a list of step-jump entries for the solver.

    Each entry describes the jump from one simulation step to the
    next, including cross-period jumps.  Stochastic branches use the
    last-realized-step on a sibling branch as the predecessor.

    Mirrors :func:`flextool.flextoolrunner.timeline_config.make_step_jump`
    (lines 608-703 of the source).
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
                        (
                            period,
                            step.timestep,
                            active_time[j - 1].timestep,
                            active_time[block_last].timestep,
                            period,
                            active_time[j - 1].timestep,
                            jump,
                        ),
                    )
                    block_last = j - 1
                else:
                    step_lengths.insert(
                        period_start_pos,
                        (
                            period,
                            step.timestep,
                            active_time[j - 1].timestep,
                            active_time[j - 1].timestep,
                            period,
                            active_time[j - 1].timestep,
                            jump,
                        ),
                    )
            else:
                if (period, period) not in period__branch:
                    original_period = None
                    for pb in period__branch:
                        if pb[1] == period:
                            original_period = pb[0]
                    if (
                        original_period is not None
                        and (original_period, original_period)
                        in period__branch
                        and original_period in active_time_list
                    ):
                        jump = (
                            active_time[j].index - active_time[-1].index
                        )
                        step_lengths.insert(
                            period_start_pos,
                            (
                                period,
                                step.timestep,
                                active_time[j - 1].timestep,
                                active_time[block_last].timestep,
                                period,
                                active_time_list[period][-1].timestep,
                                jump,
                            ),
                        )
                    elif (
                        original_period is not None
                        and (original_period, original_period)
                        in period__branch
                    ):
                        jump = (
                            active_time[j].index - active_time[-1].index
                        )
                        step_lengths.insert(
                            period_start_pos,
                            (
                                period,
                                step.timestep,
                                active_time[j - 1].timestep,
                                active_time[block_last].timestep,
                                period,
                                active_time_list[period][-1].timestep,
                                jump,
                            ),
                        )
                    else:
                        time_branch = None
                        for sb_tb in solve_branch__time_branch_list:
                            if sb_tb[0] == period:
                                time_branch = sb_tb[1]
                        past = False
                        found = False
                        previous_period_with_branch = None
                        for solve_period, _a_t in reversed(
                            active_time_list.items()
                        ):
                            if past:
                                for sb_tb in solve_branch__time_branch_list:
                                    if (
                                        sb_tb[0] == solve_period
                                        and sb_tb[1] == time_branch
                                    ):
                                        previous_period_with_branch = (
                                            solve_period
                                        )
                                        found = True
                                if found:
                                    break
                            else:
                                if solve_period == period:
                                    past = True
                        jump = (
                            active_time[j].index
                            - active_time_list[
                                previous_period_with_branch
                            ][-1].index
                        )
                        step_lengths.insert(
                            period_start_pos,
                            (
                                period,
                                step.timestep,
                                active_time[j - 1].timestep,
                                active_time[block_last].timestep,
                                previous_period_with_branch,
                                active_time_list[
                                    previous_period_with_branch
                                ][-1].timestep,
                                jump,
                            ),
                        )
                else:
                    jump = (
                        active_time[j].index
                        - active_time_list[previous_period_name][-1].index
                    )
                    step_lengths.insert(
                        period_start_pos,
                        (
                            period,
                            step.timestep,
                            active_time[j - 1].timestep,
                            active_time[block_last].timestep,
                            previous_period_name,
                            active_time_list[previous_period_name][
                                -1
                            ].timestep,
                            jump,
                        ),
                    )
    return step_lengths


def make_period_block(
    active_time_list: dict,
) -> tuple[list[tuple], list[tuple]]:
    """Build period_block_time and period_block_succ data.

    Blocks are maximal contiguous runs of active steps in the timeline
    — same detection rule as :func:`make_step_jump` (jump > 1 starts a
    new block).  Used by the ``bind_intraperiod_blocks`` storage
    binding method.

    Returns:
        period_block_time: ``[(period, block_first, step), ...]``,
            one row per active step.
        period_block_succ: ``[(period, block_first, block_first_next),
            ...]``, cyclic within period.

    Mirrors :func:`flextool.flextoolrunner.timeline_config.make_period_block`.
    """
    period_block_time: list[tuple] = []
    period_block_succ: list[tuple] = []

    for period, active_time in active_time_list.items():
        if not active_time:
            continue
        block_firsts_in_order: list[str] = [active_time[0].timestep]
        cur_block_first = active_time[0].timestep
        for j, step in enumerate(active_time):
            if (
                j > 0
                and active_time[j].index - active_time[j - 1].index > 1
            ):
                cur_block_first = step.timestep
                block_firsts_in_order.append(cur_block_first)
            period_block_time.append(
                (period, cur_block_first, step.timestep)
            )

        n_blocks = len(block_firsts_in_order)
        for i, b_first in enumerate(block_firsts_in_order):
            b_next = block_firsts_in_order[(i + 1) % n_blocks]
            period_block_succ.append((period, b_first, b_next))

    return period_block_time, period_block_succ


def make_steps(steplist: list, start: int, stop: int) -> list:
    """Return a slice of *steplist* from *start* to *stop* (inclusive).

    Mirrors :func:`flextool.flextoolrunner.timeline_config.make_steps`.
    """
    active_step = start
    steps: list = []
    while active_step <= stop:
        steps.append(steplist[active_step])
        active_step += 1
    return steps


def make_timeset_timeline(
    steplist: list, start: str, length: float
) -> list:
    """Build a timeset timeline from a steplist.

    Returns a slice of *steplist* starting at *start* and continuing
    for ``ceil(length)`` steps.

    Mirrors :func:`flextool.flextoolrunner.timeline_config.make_timeset_timeline`.
    """
    result: list = []
    startnum = steplist.index(start)
    for i in range(startnum, math.ceil(startnum + float(length))):
        result.append(steplist[i])
    return result


def separate_period_and_timeseries_data(
    timelines: dict,
    solve__period__timeset: dict,
    work_folder: "Path | None" = None,
) -> None:
    """Split ``input/pdt_*.csv`` into per-period and per-timeseries shards.

    For each ``pdt_<X>.csv`` (currently ``pdt_commodity.csv`` and
    ``pdt_group.csv``), write:

    * ``pd_<X>.csv`` — rows whose third column is a period identifier.
    * ``pt_<X>.csv`` — rows whose third column is a timestep identifier.

    The discriminator is column-2 membership in the union of all
    timesteps (across all timelines) vs. all periods (from the
    ``solve__period__timeset`` map).

    Mirrors :func:`flextool.flextoolrunner.timeline_config.separate_period_and_timeseries_data`.
    """
    wf = Path(work_folder) if work_folder is not None else Path.cwd()

    inputfiles = ["pdt_commodity.csv", "pdt_group.csv"]
    for inputfile in inputfiles:
        output_period = wf / f"input/pd_{inputfile[4:]}"
        output_timeseries = wf / f"input/pt_{inputfile[4:]}"
        timesteps: list[str] = []
        for timeline in list(timelines.values()):
            for step in timeline:
                timesteps.append(step[0])
        periods: list[str] = []
        for period__timesets in solve__period__timeset.values():
            for period__timeset in period__timesets:
                periods.append(period__timeset[0])

        with open(output_period, "w", newline="") as blk_p:
            period_writer = csv.writer(blk_p, delimiter=",")
            with open(output_timeseries, "w", newline="") as blk_t:
                timeseries_writer = csv.writer(blk_t, delimiter=",")
                with open(
                    wf / f"input/{inputfile}", "r", encoding="utf-8"
                ) as blk:
                    filereader = csv.reader(blk, delimiter=",")
                    headers = next(filereader)
                    timeseries_writer.writerow(headers)
                    period_writer.writerow(
                        headers[:-2]
                        + ["period", f"pd_{headers[-1][3:]}"]
                    )
                    while True:
                        try:
                            datain = next(filereader)
                        except StopIteration:
                            break
                        if datain[2] in periods:
                            period_writer.writerow(datain)
                        elif datain[2] in timesteps:
                            timeseries_writer.writerow(datain)


__all__ = [
    "TimelineConfig",
    "get_active_time",
    "make_step_jump",
    "make_period_block",
    "make_steps",
    "make_timeset_timeline",
    "separate_period_and_timeseries_data",
]
