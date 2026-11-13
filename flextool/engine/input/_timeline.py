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

* ``create_averaged_timeseries`` combines the provider's time-series
  frames (``pt_*`` / ``pbt_*`` / ``pdt_process`` / ``pdt_node`` /
  ``pdt_energy_margin_adder``) to the steps of a solve with
  ``new_stepduration`` and writes them under ``solve_data/``; without
  ``new_stepduration`` they pass through verbatim.  The per-solve emit
  folds (``PdtLookup``, ``pdtNodeInflow``, ``pdtProfile``, ``pdtGroup``,
  ``pdtCommodity``, ``pdtReserve``, the energy-margin adder) read those
  ``solve_data/`` frames, never ``input/``.  The kind of each row
  (mean, harmonic mean, sum, last) is the parameter's
  ``TIME_AGGREGATION`` entry and the arithmetic is
  ``aggregate_to_solve_steps``, both in ``_solve_step_aggregation``: the
  same function the DB-direct readers use, so both paths give the same
  values.

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
from flextool.engine.input._solve_step_aggregation import (
    TIME_AGGREGATION,
    Agg,
    aggregate_to_solve_steps,
    kind_of,
)
from flextool.engine.preprocess.emit._base_steps import NEW_STEP_BASE_STEPS, NEW_STEP_BASE_STEPS_SCHEMA
from flextool.engine.preprocess._provider_io import (
    _provider_key,
)
from flextool.timeline_pieces import aggregate_timeline_pieces, group_step_solves, timeline_pieces_of_group
from flextool.sqlite_url import sqlite_url_string

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
        # The run's dedupe set of the solve-step aggregation warnings: the orchestration shares it with the
        # DB-direct readers (the reader's ``solve_step_aggregation_notes`` memo), so a warning is printed once per
        # run, not once per path and per complete solve.
        self.solve_step_notes: set = set()

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
            url = sqlite_url_string(url)
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
        # The grouping and the chunking live in flextool.timeline_pieces (also
        # used outside the engine, by the assistant's model facts).
        groups = group_step_solves(step_solves, _timesets_of, lambda ts: bool(raw_durations.get(ts)))

        for first_solve, step_duration, timesets in groups:
            # Representative-period base starts are cut too, so every base
            # period starts at a step of the aggregated timeline (the rp
            # base chain is ordered by timeline position).
            pieces = timeline_pieces_of_group(timesets, raw_timeline_of, self.rp_weights)
            for timeline_name, (tl_timesets, base_cuts) in pieces.items():
                old_steps = self.timelines[timeline_name]
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

    def validate_timeset_blocks(self, solve_config: "SolveConfig", warned: set[tuple] | None = None) -> None:
        """Check the ``timeset_duration`` blocks of the timesets the model's solves use, against their timelines.

        Runs on the original blocks, after :meth:`create_assumptive_parts` and before
        :meth:`create_timeline_from_timestep_duration` (the ``new_stepduration`` aggregation). Only the timesets
        in ``solve_config.timesets_used_by_solves`` (the solves of the model) are checked: a timeset no solve
        uses does not stop the run. The steps are taken in the order of the timeline's list, the order
        :func:`get_active_time` expands the blocks in.

        - A block whose start is not a step of the timeline → :class:`FlexToolConfigError` (with a
          case-insensitive suggestion). Before this check, such a block was skipped without a word.
        - A block that runs past the end of the timeline → a warning; the expansion cuts it at the end of the
          timeline, as before. ``warned`` is the set of the warnings already printed in this run: a run passes
          the same set to each of its construction sites (``run_chain_from_db`` → ``FlexToolRunner``), so a
          warning is printed once per run, and a new run (any entry point) starts with a new set. Without it,
          every call prints its warnings (no state is kept between calls or runs).
          On a representative-period timeset (one with ``representative_period_weights``) the block cannot be
          cut (the base-period mapping needs whole blocks) → error.
        - A block length that is not a whole number of time steps → a warning (the expansion uses the whole
          part, as before).
        - A used timeset with blocks whose timeline has no steps (not a timeline of the scenario) → error.

        All the errors go into one :class:`FlexToolConfigError` (up to 5 shown).
        """
        solve_of: dict[str, str] = {}
        for solve_name, pairs in solve_config.timesets_used_by_solves.items():
            for _period, ts in pairs:
                solve_of.setdefault(ts, solve_name)
        errors: list[str] = []
        warnings: list[tuple[tuple, str]] = []
        for ts, solve_name in solve_of.items():
            blocks = self.timeset_durations.get(ts)
            if not blocks:
                continue
            where = f"Timeset {ts!r} (solve {solve_name!r})"
            timeline_id = self.timesets__timeline.get(ts)
            steps = [s for s, _d in self.timelines.get(timeline_id, [])] if timeline_id else []
            if not steps:
                if timeline_id:
                    errors.append(
                        f"{where}: its timeline {timeline_id!r} is not a timeline of the scenario (it has no "
                        "timestep_duration). Check the name of the timeline (case sensitive) and that the "
                        "alternative of its timestep_duration is in the scenario.")
                else:
                    errors.append(
                        f"{where}: it has no timeline. Set the timeset's 'timeline' parameter.")
                continue
            index_of = {s: i for i, s in enumerate(steps)}
            lower = {}
            for s in steps:
                lower.setdefault(str(s).lower(), s)
            for start, count in blocks:
                n = int(float(count))
                if float(count) != n:
                    warnings.append((
                        (ts, "fraction", start, float(count)),
                        f"Warning: timeset {ts!r} (solve {solve_name!r}): the block starting {str(start)!r} has "
                        f"length {float(count):g}, which is not a whole number of time steps; {n} time steps "
                        "are used."))
                idx = index_of.get(start)
                if idx is None:
                    hint = lower.get(str(start).lower())
                    hint_txt = (f" Did you mean {str(hint)!r}? Time step names are case sensitive."
                                if hint is not None else "")
                    errors.append(
                        f"{where}: the block starting {str(start)!r} does not start at a time step of timeline "
                        f"{timeline_id!r} ({str(steps[0])!r} .. {str(steps[-1])!r}).{hint_txt}")
                    continue
                left = len(steps) - idx
                if n <= left:
                    continue
                if ts in self.rp_weights:
                    errors.append(
                        f"{where}: the block starting {str(start)!r} has {n} time steps, but timeline "
                        f"{timeline_id!r} has only {left} time steps from {str(start)!r} (to {str(steps[-1])!r}). A "
                        "representative-period block cannot be cut: shorten the block or start it earlier.")
                    continue
                warnings.append((
                    (ts, timeline_id, start, n),
                    f"Warning: timeset {ts!r} (solve {solve_name!r}): the block starting {str(start)!r} has {n} time "
                    f"steps, but timeline {timeline_id!r} has only {left} from {str(start)!r} (to {str(steps[-1])!r}); "
                    f"the last {n - left} are not used."))
        if errors:
            shown = errors[:5]
            more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
            raise FlexToolConfigError("timeset_duration: " + " ".join(shown) + more)
        for key, text in warnings:
            if warned is not None:
                if key in warned:
                    continue
                warned.add(key)
            print(text, flush=True)

    def inherit_solve_copies(self, solve_config: "SolveConfig") -> None:
        """Give each solve of a 2-D period map (``solve_config.map2d_duplicate_of``) the ``new_stepduration`` of
        its database solve: it runs on that solve's timesets, which ``create_timeline_from_timestep_duration``
        re-points to the aggregated timeline, so it needs the aggregated data too (the timeline half of
        ``SolveConfig.derive_solve``).  Called first by :meth:`create_assumptive_parts`; idempotent."""
        for copy_name, (parent, _outer) in getattr(solve_config, "map2d_duplicate_of", {}).items():
            if parent in self.new_step_durations:
                self.new_step_durations[copy_name] = self.new_step_durations[parent]

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
        self.inherit_solve_copies(solve_config)

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
            # The period of each ``(key, period)`` tuple (the key of a split solve is its 2-D map row name).
            all_periods = list(
                dict.fromkeys(tup[1] for tup in all_periods_tuples)
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
    ) -> list[str]:
        """Average or sum input timeseries to match a coarsened timeline.

        If ``solve`` does not set ``new_stepduration`` (solve-scoped as
        of DB v50), pass ``input/pt_*`` frames through verbatim under
        the ``solve_data/pt_*`` Provider key.  Otherwise re-aggregate
        each timeseries row-by-row to match the new step size and
        ``provider.put`` the aggregated frame under the canonical
        ``solve_data/<basename>`` key.

        Aggregation rules: each new step combines the time steps of the
        original timeline that it covers (any step length; see
        ``_base_steps_of_new_steps``).  The rows of each file (grouped by
        the columns before ``time``, which include the leading ``period``
        of ``pdt_process`` / ``pdt_node`` and the ``branch`` /
        ``time_start`` of ``pbt_*``) are combined by the kind of their
        parameter in ``TIME_AGGREGATION`` (:data:`_CSV_SERIES` names the
        parameter of each row; :func:`aggregate_to_solve_steps` does the
        arithmetic): ``inflow`` is summed, efficiencies take the harmonic
        mean, ``storage_state_reference_value`` the value of the step's
        last original step, every other series the duration-weighted mean
        (only the profile rounded to 6 decimals; a step whose values are
        all equal keeps the value itself).  ``solve_data/new_step_base_steps`` lists the
        original steps (``time``), their new step, its number of original
        steps, their hours and timeline positions.

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

        Returns:
            The provider keys written (every ``solve_data/`` frame of this
            solve's steps), so the caller can mirror exactly these frames
            onto the sub-solve provider.
        """
        wf = Path(work_folder) if work_folder is not None else Path.cwd()
        written: list[str] = []

        def _put(key: str, frame: pl.DataFrame) -> None:
            provider.put(key, frame)
            written.append(key)
        # As of DB v50 new_stepduration is keyed by solve.  ``None`` is
        # the parameter_definition default in the master template; treat
        # it as "not set" and skip aggregation.
        create = (
            solve in self.new_step_durations
            and self.new_step_durations[solve] is not None
        )
        base_steps_key = _provider_key(wf / "solve_data" / NEW_STEP_BASE_STEPS)
        if not create:
            for timeseries in _CSV_SERIES:
                src_path = wf / "input" / timeseries
                dst_path = wf / "solve_data" / timeseries
                src_key = _provider_key(src_path)
                dst_key = _provider_key(dst_path)
                frame = _provider_get_or_read(provider, src_key, src_path)
                if frame is None:
                    continue
                _put(dst_key, frame)
            # Every solve step is one original step (an earlier coarsened
            # solve's rows must not reach this solve).
            _put(base_steps_key, pl.DataFrame(schema=NEW_STEP_BASE_STEPS_SCHEMA))
            # No full-timeline inflow energy: scale_to_annual_flow uses the
            # solve's own period share (not an earlier coarsened solve's).
            _put(_provider_key(wf / "solve_data" / "node_full_timeline_flow.csv"),
                         pl.DataFrame(schema={"node": pl.Utf8, "value": pl.Float64}))
            return written

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
        # (original step, the new step that combines it, the original step's hours), in timeline order.
        base_step_rows: list[tuple[str, str, float]] = []
        for timeline in timelines_list:
            new_timeline = self.timelines[timeline]
            original = self.original_timeline.get(timeline)
            if original is not None and original in self.timelines:
                base_durations.update(_base_steps_of_new_steps(
                    self.timelines[original], new_timeline, logger,
                    solve_name=solve, timeline_name=timeline,
                ))
                hours = {name: float(d) for name, d in self.timelines[original]}
                base_step_rows.extend((t, s, hours[t])
                                      for t, s in _new_step_of_base_steps(self.timelines[original], new_timeline))
            else:
                for timeline_row in new_timeline:
                    base_durations[timeline_row[0]] = [float(timeline_row[1])]
                    base_step_rows.append((timeline_row[0], timeline_row[0], float(timeline_row[1])))
        # The scalings read how many original steps each solve step combines
        # (inflow and peak_inflow are energy per original step); the aggregation
        # reads the original steps' hours and timeline positions.
        base_steps = pl.DataFrame(
            [(t, s, float(len(base_durations[s])), d, i) for i, (t, s, d) in enumerate(base_step_rows)],
            schema=NEW_STEP_BASE_STEPS_SCHEMA, orient="row")
        _put(base_steps_key, base_steps)

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

        notes = self.solve_step_notes
        approx = _approx_entities_from_provider(provider, wf)
        for timeseries in _CSV_SERIES:
            src_path = wf / "input" / timeseries
            dst_path = wf / "solve_data" / timeseries
            src_key = _provider_key(src_path)
            dst_key = _provider_key(dst_path)
            src_frame = _provider_get_or_read(provider, src_key, src_path)
            if src_frame is None:
                continue
            headers = src_frame.columns
            if "time" not in headers:
                # No time column — pass through verbatim.
                _put(dst_key, src_frame)
                continue
            out_rows = _aggregate_csv_series(timeseries, src_frame, base_steps, notes, approx)

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

            _put(
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
            _put(
                _provider_key(flow_path),
                _build_frame_with_headers(["node", "value"], flow_rows),
            )
        return written


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


# The provider's time-series frames that a solve's ``new_stepduration`` aggregates: file -> (the entity classes of
# its rows, the parameter when the file holds one parameter only, else None: the parameter is in the file's
# ``…Param`` column).  The kind of each row comes from TIME_AGGREGATION (one kind per (file, parameter): every
# class of a file agrees, tests/engine/test_time_aggregation_registry.py).
_CSV_SERIES: dict[str, tuple[tuple[str, ...], "str | None"]] = {
    "pt_node_inflow.csv": (("node",), "inflow"),
    "pt_commodity.csv": (("commodity",), None),
    "pt_group.csv": (("group", "flowGroup"), None),
    "pt_node.csv": (("node",), None),
    "pt_process.csv": (("unit", "connection"), None),
    "pdt_process.csv": (("unit", "connection"), None),
    "pdt_node.csv": (("node",), None),
    "pt_profile.csv": (("profile",), "profile"),
    "pt_process_source.csv": (("unit__inputNode",), None),
    "pt_process_sink.csv": (("unit__outputNode",), None),
    "pt_reserve__upDown__group.csv": (("reserve__upDown__group",), None),
    "pbt_node_inflow.csv": (("node",), "inflow"),
    "pbt_node.csv": (("node",), None),
    "pbt_process.csv": (("unit", "connection"), None),
    "pbt_profile.csv": (("profile",), "profile"),
    "pbt_process_source.csv": (("unit__inputNode",), None),
    "pbt_process_sink.csv": (("unit__outputNode",), None),
    "pbt_reserve__upDown__group.csv": (("reserve__upDown__group",), None),
    # The period Map has no time column and passes through; the emit multiplies it by the step's base-step count.
    "pd_energy_margin_adder.csv": (("node",), "energy_margin_adder"),
    "pdt_energy_margin_adder.csv": (("node",), "energy_margin_adder"),
}


# Frames whose values earlier releases already combined over a new_stepduration step (the time-map profile and
# inflow), or that have a notice of their own (the energy-margin adder): no solve_step_series_combined notice.
_CSV_COMBINED_BEFORE = frozenset({"pt_profile.csv", "pt_node_inflow.csv", "pdt_energy_margin_adder.csv"})


def csv_series_kind(file: str, parameter: str) -> "tuple[tuple[str, str], Agg]":
    """``((class, parameter), kind)`` of the rows of ``parameter`` in the provider frame ``file``
    (:data:`_CSV_SERIES`): the first class of the file with a TIME_AGGREGATION entry; every class of the file that
    has one must agree.  A parameter without an entry raises :class:`FlexToolConfigError`."""
    classes, _fixed = _CSV_SERIES[file]
    found = [((c, parameter), TIME_AGGREGATION[(c, parameter)]) for c in classes
             if (c, parameter) in TIME_AGGREGATION]
    if not found:
        return (classes[0], parameter), kind_of((classes[0], parameter))
    if len({k for _key, k in found}) > 1:
        raise FlexToolConfigError(
            f"{file}: the classes {[key[0] for key, _k in found]} combine '{parameter}' over a new_stepduration "
            f"step differently ({[k.value for _key, k in found]}); TIME_AGGREGATION must agree within one file.")
    return found[0]


def _approx_entities_from_provider(provider: "FlexDataProvider", wf: Path) -> "dict[tuple[str, str], frozenset[str]]":
    """The units whose varying ``min_load`` makes a solve step approximate (``_APPROXIMATE_WHEN_VARYING``): the
    ``min_load_efficiency`` units of ``input/process__ct_method.csv`` (none when the frame is absent)."""
    frame = _provider_get_or_read(provider, _provider_key(wf / "input" / "process__ct_method.csv"),
                                  wf / "input" / "process__ct_method.csv")
    units: frozenset[str] = frozenset()
    if frame is not None and frame.height > 0 and frame.width >= 2:
        units = frozenset(frame.filter(pl.col(frame.columns[1]).cast(pl.Utf8) == "min_load_efficiency")
                          .get_column(frame.columns[0]).cast(pl.Utf8).to_list())
    return {("unit", "min_load"): units}


def _aggregate_csv_series(file: str, frame: pl.DataFrame, base_steps: pl.DataFrame, notes: set,
                          approx_entities: "dict[tuple[str, str], frozenset[str]] | None" = None) -> list[list]:
    """The rows of the provider frame ``file`` combined to the solve's steps (:func:`aggregate_to_solve_steps`):
    one row per (columns before ``time``, solve step), the value last; every value as a float.
    ``approx_entities``: see :func:`aggregate_to_solve_steps`."""
    headers = frame.columns
    ti = headers.index("time")
    group_cols = headers[:ti]
    value_col = headers[ti + 1]
    classes, fixed = _CSV_SERIES[file]
    report = file not in _CSV_COMBINED_BEFORE
    if fixed is not None:
        key, kind = csv_series_kind(file, fixed)
        out = aggregate_to_solve_steps(frame, kind, group_cols, base_steps, time_col="time",
                                       value_col=value_col, keys=key, notes=notes, report_changes=report,
                                       approx_entities=approx_entities)
    else:
        param_col = next((c for c in group_cols if c.endswith("Param")), None)
        if param_col is None:
            raise FlexToolConfigError(f"{file}: no parameter column (…Param) before 'time' in {headers}.")
        params = frame.get_column(param_col).cast(pl.Utf8).unique(maintain_order=True).to_list()
        resolved = {p: csv_series_kind(file, p) for p in params if p is not None}
        out = aggregate_to_solve_steps(
            frame, param_col, group_cols, base_steps, time_col="time", value_col=value_col,
            keys={p: key for p, (key, _k) in resolved.items()},
            kind_values={p: k for p, (_key, k) in resolved.items()}, notes=notes, report_changes=report,
            approx_entities=approx_entities)
    return [list(r) for r in out.select(*group_cols, "time", value_col).iter_rows()]


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
    """Aggregate a whole timeline to ``step_duration`` hours, piece by piece: :func:`flextool.timeline_pieces.
    aggregate_timeline_pieces` (see there), raising :class:`FlexToolConfigError` when a new step does not fit
    (only with ``check_fit``, i.e. for a solve the model runs)."""
    return aggregate_timeline_pieces(
        old_steps, timeset_blocks, step_duration, logger, extra_cuts,
        solve_name=solve_name, timeline_name=timeline_name, check_fit=check_fit,
        error_cls=FlexToolConfigError,
    )


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
            return f"block starting {str(start)!r}" + (
                f" (timeset {str(ts)!r})" if ts != overlaps[0][3][0] else "")

        sample = "; ".join(
            f"period {str(p)!r} timestep {str(t)!r} is in both {_block(a)} and "
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
