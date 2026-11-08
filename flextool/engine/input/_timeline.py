"""Timeline-level configuration loaded from a SpineDB scenario.

Architecture notes
------------------

* Every downstream module (recursive solve builder, stochastic,
  orchestration loop) reads ``state.timeline.<dict>`` keys produced
  here.  Quirks:

  - the ``representative_period_weights`` decoder accepts BOTH
    :class:`spinedb_api.Map` and ``list[tuple]`` shapes (Spine API quirk;
    see ``audit/solve_orchestration_plan.md §3.2``); ``timeset_weights``
    is read from the raw Map and its index shape-checked.
  - ``new_step_durations`` is solve-scoped as of DB v50 (was
    timeset-scoped pre-v50; ``update_flextool/db_migration.py`` upgrades
    older DBs before this loader runs).
  - The six-rule fallback chain in :meth:`TimelineConfig.create_assumptive_parts`
    mutates *both* ``self`` AND the supplied ``solve_config`` in three
    of its rules.  Order matters; idempotency is achieved through the
    membership checks each rule applies before mutating.

* :func:`get_active_time` is the most-called free function — every
  recursive solve invocation walks it.  It reads :attr:`timesets`
  (``list[str]``) but the parameter is sometimes a *dict* of timeset
  durations passed by the orchestration layer; the function tolerates
  either by taking timeset durations as the second argument.

* ``create_averaged_timeseries`` aggregates per-(``solve``) timeline
  data by averaging or summing matched timesteps.  The
  ``pt_node_inflow.csv`` / ``pbt_node_inflow.csv`` files are summed;
  every other ``pt_*.csv`` / ``pbt_*.csv`` is averaged.  The
  ``storage_state_reference_value`` row in any of those files bypasses
  aggregation entirely.

* ``_derived.py::dt_and_step_duration_from_source`` is a
  polars-lazy alternative that consumes the per-(entity_class,
  parameter_name) ``InputSource`` Protocol; this module uses
  ``params_to_dict`` directly.  The two paths produce identical
  timeline / timeset state and coexist because they serve different
  consumers — the InputSource one feeds Param-frame helpers, this one
  feeds the orchestration loop.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import spinedb_api as api

from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.input._timeline_rules import SYNTHETIC_TIMESET
from flextool.engine.input._solve_config import (
    DictMode,
    get_single_entities,
    params_to_dict,
)
from flextool.engine.input._solve_state import (
    ActiveTimeEntry,
    FlexToolConfigError,
)
from flextool.engine.preprocess.emit._base_steps import NEW_STEP_BASE_STEPS, NEW_STEP_BASE_STEPS_SCHEMA
from flextool.engine.preprocess._provider_io import (
    _provider_key,
)

if TYPE_CHECKING:
    from spinedb_api import DatabaseMapping

    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.engine.input._solve_config import SolveConfig


# ---------------------------------------------------------------------------
# TimelineConfig — main container
# ---------------------------------------------------------------------------


class TimelineConfig:
    """All timeline definitions and timeset mappings for a native run.

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
        # Original timeset durations / timelines, kept by the first
        # create_timeline_from_timestep_duration call.
        self._raw_timeset_durations: dict[str, list] | None = None
        self._raw_timesets__timeline: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load_from_db(
        cls, db: "DatabaseMapping", logger: logging.Logger
    ) -> "TimelineConfig":
        """Read all timeline-level parameters from *db* into a TimelineConfig.

        Reads:

        * ``timeline.timestep_duration`` — base timeline definitions.
        * ``timeset`` entities — flat list.
        * ``timeset.timeset_duration`` — (start, count) blocks per timeset.
        * ``timeset.timeline`` — timeset → timeline mapping.
        * ``solve.new_stepduration`` — per-solve aggregation override
          (DB v50+; older DBs are upgraded by db_migration first).
        * ``timeset.representative_period_weights`` — nested Map decoder
          (handles BOTH ``api.Map`` and ``list[tuple]`` shapes).
        * ``timeset.timeset_weights`` — flat ``time`` Map, shape-checked
          (:func:`_read_timeset_weights`).
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
        # cost/slack weighting.  Read from the raw values so the Map's
        # ``index_name`` is validated (a period map or a scalar raises).
        timeset_weights = _read_timeset_weights(db)
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

        Mirrors :meth:`flextool.engine.input._solve_config.SolveConfig.load_from_db_url`
        — same pattern.
        """
        from flextool.update_flextool.db_handle import open_database_mapping
        from spinedb_api.filters.scenario_filter import (
            apply_scenario_filter_to_subqueries,
        )

        if logger is None:
            logger = logging.getLogger(f"flextool.engine.timeline[{scenario}]")
        url = str(db_url)
        if "://" not in url:
            url = f"sqlite:///{url}"
        with open_database_mapping(url) as db:
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

        Currently only :class:`flextool.engine.input._spinedb_reader.SpineDbReader`
        sources are supported — they expose ``db_url`` and ``scenario`` so
        the canonical :meth:`load_from_db_url` path can be reused.
        In-memory and CSV-backed sources are deferred to Γ.8.D.
        """
        from flextool.engine.input._spinedb_reader import SpineDbReader

        if isinstance(source, SpineDbReader):
            return cls.load_from_db_url(
                source.db_url, source.scenario, logger=logger
            )
        raise NotImplementedError(
            f"TimelineConfig.load_from_source does not yet support "
            f"{type(source).__name__!r} sources.  Use load_from_db / "
            f"load_from_db_url with a Spine DB for now; the in-memory and "
            f"CSV adapters land in Gamma.8.D when chain.run_chain is rewired."
        )

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def create_timeline_from_timestep_duration(
        self, solve_config: "SolveConfig"
    ) -> None:
        """Synthesise per-solve aggregated timelines.

        For each solve that sets ``new_stepduration`` (Δ), build one
        aggregated timeline ``"{timeline}_{solve}"`` per original timeline
        its timesets use and re-point those timesets at it.  The whole
        original timeline is cut at every block start and block end of the
        timesets on it, and each piece — a block or a gap between blocks —
        is aggregated from its first step in chunks of Δ hours (label = the
        first original step, duration = the summed original durations; a
        piece whose length is not a multiple of Δ ends with a shorter
        chunk).  Gap chunks are never active (no timeset covers them) but
        keep the gaps in the step numbering, so steps are adjacent in the
        aggregated timeline exactly when they are adjacent on the original
        timeline, and every timeset of the solve keeps its own steps.  Each
        timeset's ``timeset_durations`` becomes ``[(block start, number of
        chunks in the block)]``.

        Always aggregates from the original timelines and timeset durations
        (kept on the first call), so a timeset shared by several solves with
        the same Δ gets one timeline, built from the timesets of all those
        solves.  ``original_timeline`` maps the new name to the original.

        Raises :class:`FlexToolConfigError` when a timeset is shared between
        two solves with different ``new_stepduration`` values, or between a
        solve of the model with ``new_stepduration`` and one without:
        flextool needs one timeline per timeset.
        """
        logger = logging.getLogger(__name__)
        if self._raw_timeset_durations is None:
            self._raw_timeset_durations = {
                ts: list(blocks) for ts, blocks in self.timeset_durations.items()
            }
            self._raw_timesets__timeline = dict(self.timesets__timeline)
        raw_durations = self._raw_timeset_durations
        raw_timeline_of = self._raw_timesets__timeline

        def _timesets_of(solve_name: str) -> list[str]:
            return list(dict.fromkeys(
                ts for _period, ts in solve_config.timesets_used_by_solves.get(solve_name, [])
            ))

        step_solves: dict[str, float] = {
            solve_name: float(raw)
            for solve_name, raw in self.new_step_durations.items()
            if raw is not None
        }

        # One new_stepduration per timeset.
        timeset_to_step_duration: dict[str, tuple[str, float]] = {}
        for solve_name, sd in step_solves.items():
            for ts in _timesets_of(solve_name):
                prev = timeset_to_step_duration.get(ts)
                if prev is None:
                    timeset_to_step_duration[ts] = (solve_name, sd)
                elif prev[1] != sd:
                    message = (
                        f"Timeset '{ts}' is shared between solves "
                        f"'{prev[0]}' (new_stepduration={prev[1]}) and "
                        f"'{solve_name}' (new_stepduration={sd}); "
                        f"flextool needs one timeline per timeset."
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)
        # ... also between a solve of the model with new_stepduration and one without.
        in_model = _solves_in_model(solve_config)
        step_solve_in_model_of: dict[str, str] = {}
        for solve_name in step_solves:
            if solve_name in in_model:
                for ts in _timesets_of(solve_name):
                    step_solve_in_model_of.setdefault(ts, solve_name)
        for solve_name in sorted(in_model - set(step_solves)):
            for ts in _timesets_of(solve_name):
                other = step_solve_in_model_of.get(ts)
                if other is not None:
                    message = (
                        f"Timeset '{ts}' is shared between solve '{other}' "
                        f"(new_stepduration={step_solves[other]}) and solve '{solve_name}' "
                        f"(no new_stepduration); flextool needs one timeline "
                        f"per timeset — give the solves separate timesets."
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)

        # Solves sharing a timeset (same Δ by the check above) share its
        # aggregated timeline: group them, first solve names the timeline.
        groups: list[tuple[str, float, list[str]]] = []
        group_of_timeset: dict[str, int] = {}
        for solve_name, sd in step_solves.items():
            timesets = [ts for ts in _timesets_of(solve_name) if raw_durations.get(ts)]
            hit = sorted({group_of_timeset[ts] for ts in timesets if ts in group_of_timeset})
            if not hit:
                groups.append((solve_name, sd, []))
                gi = len(groups) - 1
            else:
                gi = hit[0]
                for other in hit[1:]:
                    groups[gi][2].extend(groups[other][2])
                    for ts in groups[other][2]:
                        group_of_timeset[ts] = gi
                    groups[other] = (groups[other][0], groups[other][1], [])
            for ts in timesets:
                if ts not in groups[gi][2]:
                    groups[gi][2].append(ts)
                group_of_timeset[ts] = gi

        for first_solve, step_duration, timesets in groups:
            by_timeline: dict[str, list[str]] = {}
            for ts in timesets:
                by_timeline.setdefault(raw_timeline_of[ts], []).append(ts)
            for timeline_name, tl_timesets in by_timeline.items():
                old_steps = self.timelines[timeline_name]
                # Representative-period base starts are cut too, so every base
                # period starts at a step of the aggregated timeline (the rp
                # base chain is ordered by timeline position).
                base_cuts = {
                    f"{ts} (representative_period_weights)": [str(b) for b in self.rp_weights[ts]]
                    for ts in tl_timesets if self.rp_weights.get(ts)
                }
                # The fit of the new step is checked only for solves the model
                # runs: an unused solve must not stop the run.
                used_by = [step_solve_in_model_of[ts] for ts in tl_timesets if ts in step_solve_in_model_of]
                chunks, block_chunks = _aggregate_timeline_pieces(
                    old_steps,
                    {ts: raw_durations[ts] for ts in tl_timesets},
                    step_duration,
                    logger,
                    extra_cuts=base_cuts,
                    solve_name=used_by[0] if used_by else first_solve,
                    timeline_name=timeline_name,
                    check_fit=bool(used_by),
                )
                # Guard: every base start on the original timeline must start a step
                # of the aggregated one, else the rp base chain would sort it last.
                chunk_labels = {label for label, _hours in chunks}
                original_labels = {step[0] for step in old_steps}
                for owner, labels in base_cuts.items():
                    off = [b for b in labels if b in original_labels and b not in chunk_labels]
                    if off:
                        message = (
                            f"new_stepduration {step_duration} h: base period start(s) {off[:5]} of "
                            f"{owner} are not time steps of the aggregated timeline of solve "
                            f"'{first_solve}'."
                        )
                        logger.error(message)
                        raise FlexToolConfigError(message)
                new_timeline_name = timeline_name + "_" + first_solve
                self.timelines[new_timeline_name] = chunks
                self.original_timeline[new_timeline_name] = timeline_name
                for ts in tl_timesets:
                    self.timeset_durations[ts] = block_chunks[ts]
                    self.timesets__timeline[ts] = new_timeline_name

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
        """
        logger = logging.getLogger(__name__)

        # Rule 1: No timeset → synthesise one timeset, SYNTHETIC_TIMESET, from the sole timeline.
        if not self.timesets and len(self.timelines) == 1:
            self.timesets = [SYNTHETIC_TIMESET]

        # Rule 2: No timeset_durations + sole timeline → block the whole
        # timeline as one timeset_duration entry.  The guard is evaluated
        # ONCE, before the loop (Q-c25-A's default, item #171): it must
        # stay scenario-wide ("no timeset_durations anywhere"), not
        # per-timeset, or the first timeset's assignment flips the guard
        # false and every later timeset of a multi-timeset, single-
        # timeline model with no authored durations gets no steps at all
        # — which contradicts this rule's own docstring ("No
        # timeset_durations → full-timeline timeset" for every timeset).
        # This matches the DB-side default
        # (``_per_solve_sets.timeset_duration_lf``), which gives every
        # timeset of the solve the whole timeline unconditionally.
        apply_rule_2 = not self.timeset_durations and len(self.timelines) == 1
        for timeset_name in self.timesets:
            if apply_rule_2:
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
                elif not self.timeset_durations:
                    message = (
                        f"Solve '{solve}' has no 'period_timeset' and no "
                        f"timeset is available: FlexTool creates one from "
                        f"the timeline only when the model has exactly one "
                        f"timeline (it has {len(self.timelines)}). Please "
                        f"define a timeset with 'timeline' and "
                        f"'timeset_duration' and use 'period_timeset' "
                        f"parameter of 'solve' class to define which "
                        f"periods and which timesets are part of the solve"
                    )
                    logger.error(message)
                    raise FlexToolConfigError(message)
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
            # Ordered de-dup (first occurrence) — a set here would make
            # the seeded row order vary with PYTHONHASHSEED.
            all_periods = list(
                dict.fromkeys(
                    item for tup in all_periods_tuples for item in tup
                )
            )
            # Membership check OUTSIDE the loop: the first append below
            # inserts the solve key into the defaultdict, so an in-loop
            # check would stop after seeding a single period.
            if solve not in solve_config.solve_period_years_represented:
                for period in all_periods:
                    solve_config.solve_period_years_represented[solve].append(
                        [period, 1.0]
                    )

    def create_averaged_timeseries(
        self,
        solve: str,
        solve_config: "SolveConfig",
        logger: logging.Logger,
        *,
        provider: "FlexDataProvider",
        work_folder: "Path | None" = None,
    ) -> None:
        """Average or sum input timeseries to match a coarsened timeline.

        If ``solve`` does not set ``new_stepduration`` (solve-scoped as
        of DB v50), pass ``input/pt_*`` frames through verbatim under
        the ``solve_data/pt_*`` Provider key.  Otherwise re-aggregate
        each timeseries row-by-row to match the new step size and
        ``provider.put`` the aggregated frame under the canonical
        ``solve_data/<basename>`` key.

        Aggregation rules (lines 405-422 of the flextool source):

        Each new step combines the time steps of the original timeline
        that it covers (any step length; see ``_base_steps_of_new_steps``):
        the rows of those steps are read and combined into one row.

        * ``pt_node_inflow.csv`` and ``pbt_node_inflow.csv`` →
          ``"sum"`` (inflow energies per step sum across coarsened steps).
        * Every other ``pt_*.csv`` / ``pbt_*.csv``, plus ``pdt_process.csv`` / ``pdt_node.csv`` (the
          genuine period × time maps — the group key includes their leading ``period`` column, found by
          name like every other column before ``time``) → ``"average"``, weighted by the durations of
          the combined steps (the plain mean when they are equally long), rounded to 6 decimals.
        * Rows with parameter ``"storage_state_reference_value"`` are
          ALWAYS bypassed (line 474) — they're per-step state targets
          rather than time-integrated quantities.

        After file aggregation, ``input/p_node``'s ``inflow`` rows
        are appended to ``solve_data/pt_node_inflow`` as one entry
        per new timestep, times the number of time steps the new step
        combines (a constant inflow is energy per time step).

        Args:
            solve: Active solve name.
            solve_config: SolveConfig containing the
                ``timesets_used_by_solves`` map.
            logger: Logger for error reporting.
            provider: Active cascade-input :class:`FlexDataProvider`.
                Required — supplies the ``input/pt_*`` frames and
                receives the derived ``solve_data/pt_*`` frames.
            work_folder: Working directory.  When provided,
                ``solve_data/`` is resolved under it (used only to
                build canonical Provider keys); defaults to the
                current working directory.
        """
        wf = Path(work_folder) if work_folder is not None else Path.cwd()
        timeseries_map: dict[str, str] = {
            "pt_node_inflow.csv": "sum",
            "pt_commodity.csv": "average",
            "pt_group.csv": "average",
            "pt_node.csv": "average",
            "pt_process.csv": "average",
            "pdt_process.csv": "average",
            "pdt_node.csv": "average",
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
        base_steps_key = _provider_key(wf / "solve_data" / NEW_STEP_BASE_STEPS)
        if not create:
            for timeseries in timeseries_map:
                src_path = wf / "input" / timeseries
                dst_path = wf / "solve_data" / timeseries
                src_key = _provider_key(src_path)
                dst_key = _provider_key(dst_path)
                frame = _provider_get_or_read(provider, src_key, src_path)
                if frame is None:
                    continue
                provider.put(dst_key, frame)
            # Every solve step is one original step (an earlier coarsened
            # solve's rows must not reach this solve).
            provider.put(base_steps_key, pl.DataFrame(schema=NEW_STEP_BASE_STEPS_SCHEMA))
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
        # Durations of the original time steps each new step combines
        # (O(1) access per row); a timeline that is not an aggregated
        # one keeps one step per step.
        base_durations: dict[str, list[float]] = {}
        # (original step, the new step that combines it), in timeline order.
        base_step_rows: list[tuple[str, str]] = []
        for timeline in timelines_list:
            new_timeline = self.timelines[timeline]
            original = self.original_timeline.get(timeline)
            if original is not None and original in self.timelines:
                base_durations.update(_base_steps_of_new_steps(
                    self.timelines[original], new_timeline, logger,
                    solve_name=solve, timeline_name=timeline,
                ))
                base_step_rows.extend(_new_step_of_base_steps(self.timelines[original], new_timeline))
            else:
                for timeline_row in new_timeline:
                    base_durations[timeline_row[0]] = [float(timeline_row[1])]
                    base_step_rows.append((timeline_row[0], timeline_row[0]))
        # The scalings read how many original steps each solve step combines
        # (inflow and peak_inflow are energy per original step).
        provider.put(base_steps_key, pl.DataFrame(
            [(t, s, float(len(base_durations[s]))) for t, s in base_step_rows],
            schema=NEW_STEP_BASE_STEPS_SCHEMA, orient="row"))

        # node-name → constant-inflow value (used to append per-step
        # contributions to pt_node_inflow below).
        node__inflow: list[list[str]] = []
        p_node_path = wf / "input" / "p_node.csv"
        p_node_key = _provider_key(p_node_path)
        p_node_frame = _provider_get_or_read(
            provider, p_node_key, p_node_path,
        )
        if p_node_frame is not None and p_node_frame.height > 0:
            # Expected (node, nodeParam, p_node); the 'inflow' rows
            # have value in column index 2.
            for row in p_node_frame.iter_rows():
                if len(row) >= 3 and row[1] == "inflow":
                    node__inflow.append([row[0], row[2]])

        for timeseries in timeseries_map:
            src_path = wf / "input" / timeseries
            dst_path = wf / "solve_data" / timeseries
            src_key = _provider_key(src_path)
            dst_key = _provider_key(dst_path)
            src_frame = _provider_get_or_read(provider, src_key, src_path)
            if src_frame is None:
                continue
            headers = src_frame.columns
            try:
                time_index = headers.index("time")
            except ValueError:
                # No time column — pass through verbatim.
                provider.put(dst_key, src_frame)
                continue
            # Row-by-row state machine port of the legacy CSV loop.
            rows_iter = iter(src_frame.iter_rows())
            out_rows: list[list[Any]] = []
            timeline_for_bypass = (
                timelines_list[0] if timelines_list else None
            )
            try:
                while True:
                    datain = next(rows_iter)
                    step_base_durations = base_durations.get(datain[time_index])
                    if step_base_durations is not None:
                        values: list[float] = []
                        params = list(datain[0:time_index])
                        row = list(datain[0 : time_index + 1])
                        values.append(float(datain[time_index + 1]))
                        if (
                            time_index >= 2
                            and datain[1] != "storage_state_reference_value"
                        ) or time_index < 2:
                            for _i in range(len(step_base_durations) - 1):
                                try:
                                    datain = next(rows_iter)
                                except StopIteration:
                                    break
                                if list(datain[0:time_index]) != params:
                                    message = (
                                        "Cannot find the same "
                                        "timesteps in input data as "
                                        "in timeline for file  "
                                        + timeseries
                                        + " after "
                                        + str(row[-1])
                                    )
                                    logger.error(message)
                                    raise FlexToolConfigError(message)
                                values.append(
                                    float(datain[time_index + 1])
                                )

                        if timeseries_map[timeseries] == "average":
                            ds = step_base_durations[: len(values)]
                            if all(d == ds[0] for d in ds):
                                out_value = round(
                                    sum(values) / len(values), 6
                                )
                            else:
                                # Rates per hour (profile, limits, prices):
                                # the new step keeps their time integral.
                                out_value = round(
                                    sum(v * d for v, d in zip(values, ds)) / sum(ds), 6
                                )
                        else:
                            out_value = sum(values)
                        row.append(out_value)
                        out_rows.append(row)
                    else:
                        # Bypass aggregation for
                        # storage_state_reference_value rows — pin to
                        # the new timeline's nearest step at-or-before
                        # the original target step.
                        if (
                            time_index >= 2
                            and datain[1] == "storage_state_reference_value"
                            and timeline_for_bypass is not None
                        ):
                            original_tl = self.original_timeline[
                                timeline_for_bypass
                            ]
                            new_timeline = self.timelines[
                                timeline_for_bypass
                            ]
                            counter = 0
                            current_index = 0
                            for timestep in self.timelines[original_tl]:
                                if datain[2] == timestep[0]:
                                    current_index = counter
                                counter += 1
                            found = False
                            new_index = None
                            for timestep in reversed(
                                self.timelines[original_tl][
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
                                row = list(datain[0:time_index])
                                row.append(new_index)
                                row.append(datain[time_index + 1])
                                out_rows.append(row)
            except StopIteration:
                pass

            # Append per-step inflow contributions from p_node to
            # pt_node_inflow only; times the number of time steps each new
            # step combines (a constant inflow is energy per time step).
            if timeseries == "pt_node_inflow.csv" and node__inflow:
                for timeline in timelines_list:
                    new_timeline = self.timelines[timeline]
                    for node__value in node__inflow:
                        for timeline_row in new_timeline:
                            value = (
                                float(node__value[1])
                                * len(base_durations[timeline_row[0]])
                            )
                            out_rows.append(
                                [node__value[0], timeline_row[0], value]
                            )

            provider.put(
                dst_key,
                _build_frame_with_headers(headers, out_rows),
            )

        # ── Full-timeline inflow energy (timeset-independent scaling) ──
        # ``scale_to_annual_flow`` normalises each node's inflow to its
        # ``annual_flow`` using the ORIGINAL unmodified timeline
        # (pre-``new_stepduration``).  A shared timeset makes
        # ``create_timeline_from_timestep_duration`` accumulate every
        # solve's periods into a single coarsened timeline; summing over
        # that leaks later solves' inflow into this solve's scaling.
        # Instead sum the raw inflow over the WHOLE original timeline and
        # annualise it with its ``timeline_duration_in_years`` (= total
        # hours / HOURS_PER_YEAR), exactly as the sibling
        # ``scale_in_proportion`` method annualises.  No "find one year"
        # windowing is needed: a sub-year timeline scales up (duration <
        # 1), a multi-year timeline averages over its years (duration >
        # 1), and a one-year timeline is unchanged (duration == 1).
        new_tl = timelines_list[0] if timelines_list else None
        # Walk back to the truly unmodified (pre-coarsening) timeline.
        base_tl = new_tl
        seen_tl: set[str] = set()
        while base_tl in self.original_timeline and base_tl not in seen_tl:
            seen_tl.add(base_tl)
            base_tl = self.original_timeline[base_tl]
        base_steps = self.timelines.get(base_tl, [])
        all_step_names: set[str] = {step[0] for step in base_steps}
        total_hours = sum(float(step[1]) for step in base_steps)
        duration_in_years = total_hours / HOURS_PER_YEAR
        if duration_in_years:
            inflow_path = wf / "input" / "pt_node_inflow.csv"
            inflow_frame = _provider_get_or_read(
                provider, _provider_key(inflow_path), inflow_path,
            )
            ts_sum: dict[str, float] = {}
            if inflow_frame is not None and inflow_frame.height > 0:
                cols = inflow_frame.columns
                grouped = (
                    inflow_frame
                    .filter(pl.col(cols[1]).is_in(all_step_names))
                    .select(
                        pl.col(cols[0]).alias("node"),
                        pl.col(cols[-1]).cast(pl.Float64, strict=False)
                        .alias("v"),
                    )
                    .group_by("node")
                    .agg(pl.col("v").sum().alias("v"))
                )
                for n_name, v in grouped.iter_rows():
                    if n_name:
                        ts_sum[str(n_name)] = (
                            float(v) if v is not None else 0.0
                        )
            const_for_node: dict[str, float] = {}
            for node__value in node__inflow:
                try:
                    const_for_node[node__value[0]] = float(node__value[1])
                except (ValueError, TypeError):
                    continue
            flow_rows: list[list] = []
            for n_name in dict.fromkeys(
                list(ts_sum) + list(const_for_node)
            ):
                # A constant inflow is energy per time step: once per step.
                energy = (
                    ts_sum.get(n_name, 0.0)
                    + const_for_node.get(n_name, 0.0) * len(base_steps)
                )
                flow_rows.append([n_name, energy / duration_in_years])
            flow_path = wf / "solve_data" / "node_full_timeline_flow.csv"
            provider.put(
                _provider_key(flow_path),
                _build_frame_with_headers(["node", "value"], flow_rows),
            )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _solves_in_model(solve_config: "SolveConfig") -> set[str]:
    """Solves the model runs: ``model.solves`` and, recursively, their ``contains_solves``."""
    out: set[str] = set()
    todo = [s for solves in solve_config.model_solve.values() for s in solves]
    while todo:
        s = todo.pop()
        if s in out:
            continue
        out.add(s)
        todo.extend(solve_config.contains_solves.get(s, []))
    return out


def _base_steps_of_new_steps(
    old_steps: list, new_steps: list, logger: logging.Logger, *, solve_name: str, timeline_name: str,
) -> dict[str, list[float]]:
    """``{new step: [durations of the original steps it combines]}``.

    The new steps (chunks of :func:`_aggregate_timeline_pieces`) partition
    the whole original timeline from its first step, each labelled by its
    first original step, so every original step belongs to the latest new
    step label seen on the walk.
    """
    labels = {step[0] for step in new_steps}
    out: dict[str, list[float]] = {}
    current: list[float] | None = None
    for name, duration in old_steps:
        if name in labels:
            current = out.setdefault(name, [])
        if current is not None:
            current.append(float(duration))
    missing = [step[0] for step in new_steps if step[0] not in out]
    if missing:
        message = (
            f"Solve '{solve_name}': time steps {missing[:5]} of the aggregated timeline '{timeline_name}' are not "
            f"time steps of its original timeline."
        )
        logger.error(message)
        raise FlexToolConfigError(message)
    return out


def _new_step_of_base_steps(old_steps: list, new_steps: list) -> list[tuple[str, str]]:
    """``[(original step, the new step that combines it)]`` in timeline order: the walk of
    :func:`_base_steps_of_new_steps` (every original step belongs to the latest new step label seen)."""
    labels = {step[0] for step in new_steps}
    out: list[tuple[str, str]] = []
    current: str | None = None
    for name, _duration in old_steps:
        if name in labels:
            current = name
        if current is not None:
            out.append((name, current))
    return out


def _aggregate_timeline_pieces(
    old_steps: list,
    timeset_blocks: dict[str, list],
    step_duration: float,
    logger: logging.Logger,
    extra_cuts: dict[str, list[str]] | None = None,
    *,
    solve_name: str,
    timeline_name: str,
    check_fit: bool = True,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, int]]]]:
    """Aggregate a whole timeline to ``step_duration`` hours, piece by piece.

    ``old_steps`` is the original ``[(step, duration), ...]``; the pieces are
    the runs between the block starts and block ends of ``timeset_blocks``
    (``{timeset: [(start, count), ...]}``) and the steps in ``extra_cuts``
    (``{owner: [step, ...]}``, e.g. representative base-period starts).  Each piece is aggregated from its
    first step: a chunk closes as soon as its steps add up to ``step_duration`` hours (within a relative
    tolerance of 1e-4, so steps entered with rounded fractions still fit), and a chunk always holds at least
    one step.  The steps may have any length (0.5 h, 2 h, mixed).  The
    tolerance is capped at half the shortest step, so a long new step still
    closes at the right step.
    Returns the chunks ``[(first step, str(hours)), ...]`` in timeline order and
    per timeset ``[(block start, number of chunks in the block), ...]`` (a block
    whose start is not on the timeline is kept as given; ``get_active_time``
    skips it).  Logs a warning when a block ends with a shorter chunk, naming
    the other timeset when its boundary is what cut the chunk short.

    Raises :class:`FlexToolConfigError` (only with ``check_fit``, i.e. for a
    solve the model runs) when a chunk that starts inside a block
    would be longer than ``step_duration``: the new step is not a whole multiple
    of the timeline's steps, or (steps of different lengths) it would end inside
    a step.  A gap chunk (outside every block, never active) may be longer, e.g.
    one long filler step between two blocks.
    """
    index_of = {step[0]: i for i, step in enumerate(old_steps)}
    n_steps = len(old_steps)
    # Block index ranges [first, end) per timeset, and who owns each cut.
    ranges: dict[str, list[tuple[str, int, int] | None]] = {}
    cut_owners: dict[int, set[str]] = {0: set()}
    for ts, blocks in timeset_blocks.items():
        ranges[ts] = []
        for start, count in blocks:
            first = index_of.get(start)
            if first is None:
                ranges[ts].append(None)
                continue
            end = min(first + int(float(count)), n_steps)
            ranges[ts].append((start, first, end))
            cut_owners.setdefault(first, set()).add(ts)
            cut_owners.setdefault(end, set()).add(ts)
    for owner, labels in (extra_cuts or {}).items():
        for label in labels:
            if label in index_of:
                cut_owners.setdefault(index_of[label], set()).add(owner)
    cuts = sorted(c for c in cut_owners if c < n_steps) + [n_steps]
    block_ranges = [(rng[1], rng[2]) for rngs in ranges.values() for rng in rngs if rng is not None]
    # Relative to the new step (rounded fractions like 0.333333 h still add
    # up), but below half a base step, so a long new step never closes one
    # base step early.
    positive = [float(step[1]) for step in old_steps if float(step[1]) > 0]
    tol = min(1e-4 * step_duration, 0.5 * min(positive)) if positive else 0.0

    chunks: list[tuple[str, str]] = []
    chunk_first: list[int] = []
    short_chunks: list[tuple[int, int, float]] = []   # (first, end, hours)
    for piece_first, piece_end in zip(cuts, cuts[1:]):
        i = piece_first
        while i < piece_end:
            first = i
            hours = 0.0
            while i < piece_end and (hours < step_duration - tol or i == first):
                hours += float(old_steps[i][1])
                i += 1
            label = old_steps[first][0]
            if (check_fit and hours > step_duration + tol
                    and any(b_first <= first < b_end for b_first, b_end in block_ranges)):
                ds = [float(step[1]) for step in old_steps[first:i]]
                if all(d == ds[0] for d in ds):
                    message = (
                        f"Solve '{solve_name}': new_stepduration {step_duration:g} h is not a whole multiple of "
                        f"the {ds[0]:g} h time steps of timeline '{timeline_name}' (the new time step starting "
                        f"at '{label}' would be {hours:g} h long). Give a new_stepduration that is a whole "
                        f"multiple of the timeline's time steps."
                    )
                else:
                    message = (
                        f"Solve '{solve_name}': new_stepduration {step_duration:g} h does not fit the time steps "
                        f"of timeline '{timeline_name}': the new time step starting at '{label}' would end inside "
                        f"base time step '{old_steps[i - 1][0]}' ({hours:g} h in total). On a timeline with time "
                        f"steps of different lengths, every new time step must end where a time step of the "
                        f"timeline ends."
                    )
                logger.error(message)
                raise FlexToolConfigError(message)
            chunks.append((label, str(hours)))
            chunk_first.append(first)
            if hours < step_duration - tol:
                short_chunks.append((first, i, hours))

    block_chunks: dict[str, list[tuple[str, int]]] = {}
    for ts, blocks in timeset_blocks.items():
        out: list[tuple[str, int]] = []
        for (start, count), rng in zip(blocks, ranges[ts]):
            if rng is None:
                out.append((start, count))
                continue
            _start, first, end = rng
            out.append((start, sum(1 for c in chunk_first if first <= c < end)))
        block_chunks[ts] = out

    warned: set[tuple[str, str]] = set()
    for first, end, hours in short_chunks:
        for ts, rngs in ranges.items():
            for rng in rngs:
                if rng is None or not (rng[1] <= first < rng[2]) or (ts, rng[0]) in warned:
                    continue
                warned.add((ts, rng[0]))
                label = old_steps[first][0]
                others = sorted(cut_owners.get(end, set()) - {ts}) if end < rng[2] else []
                if others:
                    logger.warning(
                        f"new_stepduration {step_duration} h: step {label!r} of timeset {ts!r} "
                        f"(block starting {rng[0]!r}) is only {hours} h long, because a block "
                        f"or base period of {', '.join(repr(o) for o in others)} on the same "
                        f"timeline starts or ends inside it."
                    )
                else:
                    logger.warning(
                        f"new_stepduration {step_duration} h: the block of timeset {ts!r} "
                        f"starting {rng[0]!r} is not a multiple of the step; its last step "
                        f"{label!r} is {hours} h long."
                    )
    return chunks, block_chunks


def _provider_get_or_read(
    provider: "FlexDataProvider | None",
    key: str,
    path: "Path",  # noqa: ARG001 — kept for symmetry with call sites
) -> "pl.DataFrame | None":
    """Return the frame for *key* from the Provider.

    Provider-only — the disk fallback was deleted in Step 2.5.
    Returns ``None`` when the Provider is missing or doesn't carry
    *key*.  Callers that previously seeded inputs to disk without
    populating the Provider must now seed the Provider directly.
    """
    if provider is None:
        return None
    if not provider.has(key):
        return None
    return provider.get(key)


def _build_frame_with_headers(
    headers: list[str], rows: list[list],
) -> "pl.DataFrame":
    """Build an all-Utf8 polars frame matching *headers* with *rows*.

    Each row's values are stringified verbatim (``str(v)``) so the
    frame round-trips through :meth:`polars.DataFrame.write_csv`
    byte-identical to the legacy ``csv.writer.writerow`` output.
    Rows shorter than ``headers`` are padded with empty strings;
    rows longer are truncated.
    """
    n = len(headers)
    cols: list[list[str]] = [[] for _ in range(n)]
    for row in rows:
        for i in range(n):
            if i < len(row):
                v = row[i]
                cols[i].append("" if v is None else str(v))
            else:
                cols[i].append("")
    schema = {h: pl.Utf8 for h in headers}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(
        {headers[i]: cols[i] for i in range(n)},
        schema=schema,
    )


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
            # Two list-of-entry shapes coming from ``params_to_dict``:
            #   (a) Flat triples [base, rep, weight] — what
            #       ``convert_map_to_table`` produces when a 2-level Map
            #       is flattened.
            #   (b) Nested [(base, inner_map_or_list), ...] — emitted when
            #       only the outer Map is flattened.
            # Both decode to {base: {rep: weight}}.
            weight_dict: dict[str, dict[str, float]] = {}
            for entry in nested_map:
                if len(entry) >= 3 and not isinstance(entry[1], (list, api.Map)):
                    # (a) flat triple
                    base_key = str(entry[0])
                    rep_key = str(entry[1])
                    weight = float(entry[2])
                    weight_dict.setdefault(base_key, {})[rep_key] = weight
                elif len(entry) >= 2:
                    # (b) [base, inner]
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


def _read_timeset_weights(db: "DatabaseMapping") -> dict[str, dict[str, float]]:
    """``{timeset: {timestep: weight}}`` of ``timeset.timeset_weights``.

    Each value must be a 1-D Map indexed by ``time`` (or Spine's silent
    ``x``): the shape is checked against the parameter's allow-list by
    :func:`flextool.engine.input._param_shapes.check_index_name_path`, the
    same rule the DB-direct reader (``_derived.p_timestep_weight_from_source``)
    applies, so a period map, a scalar or a deeper map raises
    :class:`FlexToolConfigError` instead of being read as weights of no
    timestep.
    """
    from flextool.engine.input._param_shapes import check_index_name_path
    from flextool.engine.input._solve_state import FlexToolConfigError

    timeset_weights: dict[str, dict[str, float]] = {}
    for param in db.find_parameter_values(
        entity_class_name="timeset", parameter_definition_name="timeset_weights"
    ):
        value = api.from_database(param["value"], param["type"])
        if value is None:
            continue
        name = param["entity_name"]
        if not isinstance(value, api.Map):
            if isinstance(value, (float, int, str)):
                check_index_name_path("timeset", "timeset_weights", name, [])
            raise FlexToolConfigError(
                f"Parameter ('timeset', 'timeset_weights') of {name!r} must be a map "
                f"(timestep -> weight); got {type(value).__name__}.")
        labels: list[str | None] = []
        level: Any = value
        while isinstance(level, api.Map):
            labels.append(level.index_name)
            level = level.values[0] if len(level.values) else None
        check_index_name_path("timeset", "timeset_weights", name, labels)
        timeset_weights[name] = {
            str(k): float(v) for k, v in zip(value.indexes, value.values)
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

    # Track which (timeset, block-start) first claimed each (period,
    # timestep) so an overlap can name the two colliding blocks.
    seen_by_period: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    # (period, timestep, incumbent (timeset, block_start), colliding (…)).
    overlaps: list[tuple[str, str, tuple[str, str], tuple[str, str]]] = []

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
        seen = seen_by_period[period]
        for start_time, duration in timesets[timeset_id]:
            idx = time_val_to_idx.get(start_time)
            if idx is not None:
                for step in range(int(float(duration))):
                    if idx + step < len(timeline_data):
                        entry = timeline_data[idx + step]
                        # An overlap means a single calendar timestep would
                        # belong to two representative-period blocks at once.
                        # We cannot silently collapse it: the timeline
                        # (``steps_in_use``) needs each timestep exactly once,
                        # but its representation weight is emitted per block
                        # (:func:`_compute_rp_frames` → ``timestep_weight``),
                        # so a collapse would keep one block's weight and drop
                        # the other — a silently wrong annualisation rather
                        # than the sum.  Refuse and point at the source data.
                        if entry[0] in seen:
                            overlaps.append(
                                (period, entry[0], seen[entry[0]],
                                 (timeset_id, start_time)))
                            continue
                        seen[entry[0]] = (timeset_id, start_time)
                        active_time[period].append(
                            ActiveTimeEntry(
                                timestep=entry[0],
                                index=idx + step,
                                duration=entry[1],
                            )
                        )

    if overlaps:
        n = len(overlaps)

        def _block(b: tuple[str, str]) -> str:
            ts, start = b
            return f"block starting {start!r}" + (
                f" (timeset {ts!r})" if ts != overlaps[0][3][0] else "")

        sample = "; ".join(
            f"period {p!r} timestep {t!r} is in both {_block(a)} and "
            f"{_block(c)}"
            for p, t, a, c in overlaps[:5]
        )
        more = "" if n <= 5 else f" (+{n - 5} more)"
        bad_timesets = sorted(
            {o[2][0] for o in overlaps} | {o[3][0] for o in overlaps})
        raise ValueError(
            f"{current_solve}: overlapping timeset_duration blocks put the "
            f"same timestep into more than one representative period. "
            f"{n} overlapping timestep(s) in timeset(s) "
            f"{', '.join(repr(ts) for ts in bad_timesets)}: {sample}{more}. "
            "Each timestep may belong to exactly one representative period — "
            "adjust the timeset_duration start/length values so the blocks "
            "do not overlap (a shorter block wholly inside a longer one, or "
            "two blocks whose windows intersect, both trigger this)."
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
    next, including cross-period jumps.

    Cross-period linkage for the first step of a period:

    * **anchor periods** (``(period, period) ∈ period__branch``): the
      nearest previous ANCHOR key in ``active_time_list`` order (skips
      branch periods, so a realized continuation period links to the
      previous realized period).  The solve's FIRST period keeps the
      cyclic wrap to the last dict key (a branch period in stochastic
      solves — retained quirk, consumed only by cyclic storage-binding
      paths).
    * **branch periods**: the nearest earlier period sharing the same
      time-branch in ``solve_branch__time_branch_list`` (a continuation
      branch period continues from the same branch's previous period);
      when no such period exists (the branching period's own fan) the
      period self-cycles on its own last step.
    """
    step_lengths: list[tuple] = []
    period_start_pos = 0
    period_counter = -1
    period_names = list(active_time_list)
    first_period_name = period_names[0]
    last_period_name = period_names[-1]
    for period, active_time in reversed(active_time_list.items()):
        period_counter -= 1
        period_last = len(active_time)
        block_last = len(active_time) - 1
        if period == first_period_name:
            previous_period_name = last_period_name
        else:
            previous_period_name = period_names[period_counter]
            if (period, period) in period__branch:
                # Anchor period: skip branch periods — link to the
                # nearest previous key that is itself an anchor.
                # Deterministic solves (every key an anchor) reduce to
                # the positional previous; when no earlier anchor
                # exists the positional previous is kept as fallback.
                pos = period_names.index(period)
                for k in range(pos - 1, -1, -1):
                    if (period_names[k], period_names[k]) in period__branch:
                        previous_period_name = period_names[k]
                        break
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
                    # Branch period: find the nearest EARLIER period
                    # sharing this period's time-branch — the
                    # continuation linkage (a continuation branch
                    # period continues from the same branch's previous
                    # period).  When none exists (the branching
                    # period's own fan: earlier periods carry only the
                    # realized time-branch), fall back to the
                    # self-cycle row on the period's own last step.
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
                    if found:
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
                        # No earlier same-time-branch period exists.  Two
                        # sub-cases (Slice E design §5.2):
                        #   * MID-HORIZON reveal — this branch copy's own
                        #     anchor period is NOT the first period, so a
                        #     shared pre-reveal trunk period precedes it.
                        #     The copy must inherit the shared end-of-trunk
                        #     storage state, so link its first step to the
                        #     nearest anchor STRICTLY BEFORE the copy's own
                        #     anchor period (two-stage storage-state
                        #     non-anticipativity).
                        #   * FIRST-PERIOD fan — the copy's anchor IS the
                        #     first period; no earlier anchor exists, so it
                        #     self-cycles on its own last step (byte-parity
                        #     with the pre-Slice-E behaviour).
                        # The search is anchored on the copy's OWN anchor
                        # period (via the (a, b) period__branch rows), NOT
                        # the copy's list index — an index-based search
                        # would find the copy's own realized sibling in the
                        # same calendar period (mid-horizon) or the first
                        # anchor at index 0 (first-period fan, breaking
                        # byte-parity).
                        prev_anchor = None
                        p_anchor = None
                        for a, b in period__branch:
                            if b == period and a != period:
                                p_anchor = a
                                break
                        if p_anchor is not None and p_anchor in period_names:
                            pos_anchor = period_names.index(p_anchor)
                            for kk in range(pos_anchor - 1, -1, -1):
                                if (
                                    period_names[kk],
                                    period_names[kk],
                                ) in period__branch:
                                    prev_anchor = period_names[kk]
                                    break
                        elif p_anchor is not None:
                            # The anchor is a period label (no steps, not
                            # a key): walk the anchors in period__branch
                            # order back from it to the nearest one with
                            # steps.
                            pb_anchors = [a for a, b in period__branch if a == b]
                            if p_anchor in pb_anchors:
                                for a in reversed(pb_anchors[:pb_anchors.index(p_anchor)]):
                                    if a in period_names:
                                        prev_anchor = a
                                        break
                        if prev_anchor is not None:
                            jump = (
                                active_time[j].index
                                - active_time_list[prev_anchor][-1].index
                            )
                            step_lengths.insert(
                                period_start_pos,
                                (
                                    period,
                                    step.timestep,
                                    active_time[j - 1].timestep,
                                    active_time[block_last].timestep,
                                    prev_anchor,
                                    active_time_list[prev_anchor][
                                        -1
                                    ].timestep,
                                    jump,
                                ),
                            )
                        else:
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
    """Return a slice of *steplist* from *start* to *stop* (inclusive)."""
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
    """
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
    """Split ``input/pdt_*.csv`` into per-period and per-timeseries shards.

    For each ``pdt_<X>.csv`` (currently ``pdt_commodity.csv`` and
    ``pdt_group.csv``), put two derived frames back onto the Provider:

    * ``input/pd_<X>`` — rows whose third column is a period identifier.
    * ``input/pt_<X>`` — rows whose third column is a timestep identifier.

    The discriminator is column-2 membership in the union of all
    timesteps (across all timelines) vs. all periods (from the
    ``solve__period__timeset`` map).

    *provider* is required: the cascade's Provider supplies the
    ``input/pdt_<X>`` frames and receives the two derived shards
    keyed under the same canonical ``<parent>/<stem>`` convention
    other writer-port modules use.  The source frame is read only from
    the Provider; the work folder is not read.
    """
    wf = Path(work_folder) if work_folder is not None else Path.cwd()

    inputfiles = ["pdt_commodity.csv", "pdt_group.csv"]
    timesteps: set[str] = set()
    for timeline in timelines.values():
        for step in timeline:
            timesteps.add(step[0])
    periods: set[str] = set()
    for period__timesets in solve__period__timeset.values():
        for period__timeset in period__timesets:
            periods.add(period__timeset[0])

    for inputfile in inputfiles:
        suffix = inputfile[4:-4]  # e.g. "commodity"
        in_path = wf / "input" / inputfile
        in_key = _provider_key(in_path)
        pd_key = _provider_key(wf / "input" / f"pd_{suffix}.csv")
        pt_key = _provider_key(wf / "input" / f"pt_{suffix}.csv")

        if not provider.has(in_key):
            # No source — skip the split.  Downstream consumers tolerate
            # missing keys.  Step 2.5: Provider-only — no disk fallback.
            continue
        df = provider.get(in_key)
        key_col = df.columns[2]
        pt_frame = df.filter(pl.col(key_col).is_in(timesteps))
        pd_frame = _rename_pdt_to_pd(
            df.filter(pl.col(key_col).is_in(periods))
        )
        provider.put(pt_key, pt_frame)
        provider.put(pd_key, pd_frame)


def _rename_pdt_to_pd(df: "pl.DataFrame") -> "pl.DataFrame":
    """Rename a ``pdt_<X>`` frame's last-two columns to the ``pd_<X>``
    convention: ``[..., 'period', 'pd_<X>_value']``.

    Legacy ``write_csv`` emitted ``headers[:-2] + ['period',
    f'pd_{headers[-1][3:]}']``; reproduce the same column-name
    transform on the frame so downstream readers (which key by
    position) see byte-identical column orderings.
    """
    cols = df.columns
    if len(cols) < 2:
        return df
    new_names = list(cols)
    new_names[-2] = "period"
    last = cols[-1]
    new_names[-1] = f"pd_{last[3:]}" if len(last) > 3 else last
    return df.rename(dict(zip(cols, new_names)))


__all__ = [
    "TimelineConfig",
    "get_active_time",
    "make_step_jump",
    "make_period_block",
    "make_steps",
    "make_timeset_timeline",
    "separate_period_and_timeseries_data",
]
