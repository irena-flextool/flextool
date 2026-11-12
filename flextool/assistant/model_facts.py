"""The model facts of the scenario in focus: what the context block and ``get_gui_context`` tell the model about
the scenario's solves, time steps, periods and latest run, so that it never guesses them.

The facts come from two places:

* the input data (:func:`input_facts`): the solve sequence (``model.solves`` and the solves nested with
  ``contains_solves``), and per solve its mode (a single solve, a rolling window with its horizon and jump,
  stochastic forecast branches with their weights, representative periods), its periods with the years each
  represents and which of them it does not realise (look-ahead periods, or periods whose investments only it
  realises), and the time steps it really uses: ``timeline.timestep_duration`` over the blocks of its timesets,
  or with ``solve.new_stepduration`` the steps the engine aggregates them to (:mod:`flextool.timeline_pieces`, the
  engine's own chunking, so a block that is not a multiple of the new step ends with a shorter step);
* the scenario's latest run (:func:`run_facts`): the newest run folder of the scenario and input source, by the
  entry fields its listing gives (state, verdict, whether the input changed after it, whether its results are
  used, its error), and the share of a year of each realised period from its result table
  ``period_share_of_year__d`` (only when the run's results are used and the table is there: a run saved before
  that table has no scaling line).

:func:`facts_for` gives the :class:`ScenarioFacts` of a snapshot, or ``None`` when the snapshot has no scenario or
anything goes wrong (never cached, never raised: a question is never stopped by its facts).  The input part is
cached by ``(database key, scenario, FACTS_VERSION)``; the run part is read again for each question.  The facts
hold the names as stored: :func:`render` redacts each name and cuts it (redacted first, then cut), and keeps the
facts within :data:`FACTS_TOKENS_MAX` tokens by the cut rule of :func:`render`.  Nothing here imports
:mod:`flextool.engine`.  :class:`ScenarioFacts` and its parts are plain dataclasses (standard library only).
"""
from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flextool.assistant.context import AssistantContext, ContextSnapshot

#: The version of the facts: bumped whenever what the facts say or how they are worded changes (the evaluation
#: runner records it with each cell, ``tools/assistant_eval.py`` ``setting_key``).
FACTS_VERSION = 3
#: The first line of the facts in the context block (the scenario is named on the block's Scenario line).
HEADER = "Model facts (from the scenario's input data and latest run):"
#: The most tokens (``budget.approx_tokens``, text) the facts may take, header included.
FACTS_TOKENS_MAX = 200
#: The longest name, and the longest run error, the facts show (both redacted before they are cut).
NAME_MAX = 40
ERROR_MAX = 120
#: Solves described on their own line: the first and those that differ from it in mode or time steps.
DETAILED_SOLVES = 3
#: Names of one list shown before "and N more": the solve sequence, periods, branches, solves of a step length.
SEQUENCE_NAMES = 6
LIST_NAMES = 4
GROUP_NAMES = 3
STEP_GROUPS = 4
SHARE_GROUPS = 3
#: The result table of the realised periods' shares of a year.
SHARE_TABLE = "period_share_of_year__d"
SHARE_COLUMN = "period_share_of_year"
#: The verdict values a succeeded run may carry (``run_record.VERDICT_ORDER``) are shown as they are.
FACT_CLASSES = ("model", "solve", "timeset", "timeline")


# ---------------------------------------------------------------------------------------------------------------
# The facts (plain data)


@dataclass(frozen=True)
class BranchFan:
    """The forecast branches of one stochastic solve at one period and start time."""

    period: str
    start: str
    forecasts: tuple[tuple[str, float], ...]     # (branch, weight normalised over the forecasts of this start);
    #                                              a zero-weight forecast is left out, as the engine leaves it out
    realized: str | None                         # the branch flagged realized: yes (not a weighted forecast)


@dataclass(frozen=True)
class SolveFacts:
    """One solve of the scenario's solve sequence."""

    name: str
    parent: str | None                           # the solve whose contains_solves holds it
    mode: str                                    # solve_mode: single_solve or rolling_window
    horizon: float | None                        # rolling_solve_horizon (hours)
    jump: float | None                           # rolling_solve_jump (hours)
    duration: float | None                       # rolling_duration (hours)
    periods: tuple[tuple[str, float], ...]       # (period, years represented), in the solve's order
    realized: tuple[str, ...]                    # realized_periods: the periods whose operations it realises
    realized_invest: tuple[str, ...]             # realized_invest_periods (else realized_periods): investments
    invest: tuple[str, ...]                      # invest_periods: the periods it may invest in
    steps: tuple[tuple[float, int], ...]         # (hours, number of steps) of the steps it uses, most first
    steps_per_period: tuple[int, int] | None     # the fewest and the most steps of one period
    branches: tuple[BranchFan, ...]
    representative: tuple[tuple[str, int, int], ...]   # (timeset, representative periods, base periods)


@dataclass(frozen=True)
class RunFacts:
    """The scenario's latest run, from its listing entry."""

    folder: str
    state: str
    verdict: str | None
    verified: bool                               # it has a run record
    results_usable: bool
    stale: bool | None                           # the input data changed after the run
    error: str | None                            # as recorded (redacted and cut when rendered)
    shares: tuple[tuple[str, float], ...]        # (realised period, share of a year), from SHARE_TABLE


@dataclass(frozen=True)
class ScenarioFacts:
    """The facts of one scenario: its solves, its latest run (``None``: none), and whether they are worded for
    the basic level."""

    solves: tuple[SolveFacts, ...]
    run: RunFacts | None
    basic: bool = False


# ---------------------------------------------------------------------------------------------------------------
# Getting them


_cache: "OrderedDict[tuple, tuple[SolveFacts, ...]]" = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 8


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def facts_for(ctx: AssistantContext, snapshot: ContextSnapshot | None, tier: str | None) -> ScenarioFacts | None:
    """The facts of the snapshot's scenario in its input source (the connection's default source, else the only
    one, when it names none), worded for *tier* (internal or user-facing level name; ``None``: not basic).  ``None``
    without a scenario, and on any error (nothing is cached then)."""
    if snapshot is None or not snapshot.scenario:
        return None
    try:
        from flextool.assistant import project

        basic = tier is not None and _level(tier) == "small"
        source = project.resolve_source(ctx, snapshot.input_source)
        db, _notes = project.open_source(ctx, source)
        if snapshot.scenario not in db.scenarios:
            return None
        key = (db.key, snapshot.scenario, FACTS_VERSION)
        with _cache_lock:
            solves = _cache.get(key)
            if solves is not None:
                _cache.move_to_end(key)
        if solves is None:
            solves = input_facts(db, snapshot.scenario)
            with _cache_lock:
                _cache[key] = solves
                while len(_cache) > CACHE_SIZE:
                    _cache.popitem(last=False)
        run = run_facts(ctx, source.name, snapshot.scenario)      # the resolved source (the snapshot may name none)
        return ScenarioFacts(solves, run, basic)
    except Exception:  # noqa: BLE001 — the facts are an aid: any failure leaves them out
        return None


def with_facts(ctx: AssistantContext, snapshot: ContextSnapshot, tier: str | None) -> ContextSnapshot:
    """*snapshot* with its :attr:`~flextool.assistant.context.ContextSnapshot.scenario_facts` filled in (the same
    snapshot when there are none)."""
    facts = facts_for(ctx, snapshot, tier)
    return snapshot if facts is None else replace(snapshot, scenario_facts=facts)


def _level(tier: str) -> str:
    from flextool.assistant.prompt import level_name

    return level_name(tier)


# --- the input part ---------------------------------------------------------------------------------------------


def _items(value: Any) -> list[tuple[Any, Any]]:
    """``[(index, value), ...]`` of a map value."""
    if isinstance(value, dict):
        return list(value.items())
    return list(zip(value.indexes, value.values))


def _names(value: Any) -> list[str]:
    """The names of an array value (a single name for a string)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value.values]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class _Data:
    """The scenario's values of the classes the facts read: winners of the active entities, else the
    parameter's default."""

    def __init__(self, db: Any, scenario: str) -> None:
        self.db = db
        self.view = db.view(scenario)
        self.active: dict[str, list[str]] = {}
        for cls in FACT_CLASSES:
            entities = db.entities.get(cls, {})
            self.active[cls] = [e.name for e in entities.values() if self.view.active(cls, e.byname)[0]]

    def value(self, cls: str, name: str, parameter: str) -> Any:
        item = self.view.value(cls, (name,), parameter)
        if item is not None:
            return item.parsed()
        definition = self.db.definition(cls, parameter)
        if definition is not None and definition.default_value is not None:
            return definition.default()
        return None

    def stored(self, cls: str, parameter: str) -> dict[str, Any]:
        """``{entity: value}`` of the active entities of *cls* that have a stored *parameter*."""
        out: dict[str, Any] = {}
        for name in self.active[cls]:
            item = self.view.value(cls, (name,), parameter)
            if item is not None:
                out[name] = item.parsed()
        return out


def _sequence(data: _Data) -> list[tuple[str, str | None]]:
    """``[(solve, parent), ...]``: ``model.solves`` in order, each followed by the solves nested in it."""
    top: list[str] = []
    for model in data.active["model"]:
        top = _names(data.value("model", model, "solves"))
        if top:
            break
    if not top and len(data.active["solve"]) == 1:
        top = list(data.active["solve"])        # the engine's rule: the sole solve
    known = set(data.active["solve"])
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def walk(solve: str, parent: str | None) -> None:
        if solve in seen or solve not in known:
            return
        seen.add(solve)
        out.append((solve, parent))
        for inner in _names(data.value("solve", solve, "contains_solves")):
            walk(inner, solve)

    for solve in top:
        walk(solve, None)
    return out


def _fans(raw: Any) -> tuple[BranchFan, ...]:
    """The forecast fans of a ``stochastic_branches`` map (period → branch → analysis time → realized → weight)."""
    fans: dict[tuple[str, str], tuple[list[tuple[str, float]], list[str]]] = {}
    for period, by_branch in _items(raw):
        for branch, by_start in _items(by_branch):
            for start, by_flag in _items(by_start):
                flags = _items(by_flag) if not isinstance(by_flag, (int, float)) else [("no", by_flag)]
                for flag, weight in flags:
                    forecasts, realized = fans.setdefault((str(period), str(start)), ([], []))
                    if flag is True or str(flag).strip().lower() in ("yes", "true"):
                        realized.append(str(branch))
                    elif (_number(weight) or 0.0) != 0.0:     # the engine leaves a zero-weight forecast out
                        forecasts.append((str(branch), _number(weight) or 0.0))
    out = []
    for (period, start), (forecasts, realized) in fans.items():
        total = sum(w for _b, w in forecasts)
        normalised = tuple((b, w / total if total > 0 else 0.0) for b, w in forecasts)
        out.append(BranchFan(period, start, normalised, realized[0] if realized else None))
    return tuple(out)


def input_facts(db: Any, scenario: str) -> tuple[SolveFacts, ...]:
    """The :class:`SolveFacts` of the scenario's solve sequence, from the input database *db* (an
    :class:`~flextool.assistant.inputs._db.InputDB`)."""
    from flextool.timeline_pieces import solve_step_chunks

    data = _Data(db, scenario)
    timelines: dict[str, list[tuple[str, float]]] = {}
    for name, raw in data.stored("timeline", "timestep_duration").items():
        timelines[name] = [(str(step), float(hours)) for step, hours in _items(raw)]
    timeline_of: dict[str, str] = {}
    blocks: dict[str, list[tuple[str, float]]] = {}
    for name in data.active["timeset"]:
        tl = data.value("timeset", name, "timeline")
        if tl is None and len(timelines) == 1:
            tl = next(iter(timelines))          # the engine's rule: the sole timeline
        if tl is not None:
            timeline_of[name] = str(tl)
        raw = data.value("timeset", name, "timeset_duration")
        if raw is not None:
            blocks[name] = [(str(start), float(count)) for start, count in _items(raw)]
    if not blocks and len(timelines) == 1:      # no blocks anywhere: each timeset is the whole timeline
        (only,) = timelines.values()
        for name in data.active["timeset"]:
            if only:
                blocks[name] = [(only[0][0], float(len(only)))]
    rp_raw = data.stored("timeset", "representative_period_weights")
    rp_bases = {ts: [str(base) for base, _inner in _items(raw)] for ts, raw in rp_raw.items()}

    period_timesets: dict[str, list[tuple[str, str]]] = {}
    step_durations: dict[str, float | None] = {}
    for solve in data.active["solve"]:
        raw = data.value("solve", solve, "period_timeset")
        pairs = [(str(p), str(ts)) for p, ts in _items(raw)] if raw is not None else []
        if not pairs:
            listed = _names(data.value("solve", solve, "invest_periods")) + \
                _names(data.value("solve", solve, "realized_periods"))
            with_blocks = [ts for ts in data.active["timeset"] if blocks.get(ts)]
            if len(with_blocks) == 1:             # the engine's rule: the sole timeset
                pairs = [(p, with_blocks[0]) for p in dict.fromkeys(listed)]
        period_timesets[solve] = pairs
        step_durations[solve] = _number(data.value("solve", solve, "new_stepduration"))
    timesets_of = {solve: [ts for _p, ts in pairs] for solve, pairs in period_timesets.items()}

    out = []
    for solve, parent in _sequence(data):
        pairs = period_timesets.get(solve, [])
        periods = list(dict.fromkeys(
            [p for p, _ts in pairs] + _names(data.value("solve", solve, "realized_periods"))
            + _names(data.value("solve", solve, "invest_periods"))))
        years_raw = data.value("solve", solve, "years_represented")
        years = {str(p): _number(y) for p, y in _items(years_raw)} if years_raw is not None else {}
        period_years = tuple((p, years.get(p) if years.get(p) is not None else 1.0) for p in periods)
        realized = tuple(dict.fromkeys(_names(data.value("solve", solve, "realized_periods"))))
        invest = tuple(dict.fromkeys(_names(data.value("solve", solve, "invest_periods"))))
        realized_invest = tuple(dict.fromkeys(_names(data.value("solve", solve, "realized_invest_periods")))) \
            or realized                         # the engine's rule: without it the realized_periods realise invests

        chunks = solve_step_chunks(
            solve, new_step_durations=step_durations, timesets_of_solve=timesets_of, timeset_blocks=blocks,
            timeline_of_timeset=timeline_of, timelines=timelines, rp_weights=rp_bases)
        lengths_of: dict[str, list[float]] = {}
        for ts in dict.fromkeys(timesets_of.get(solve, [])):
            lengths_of[ts] = _timeset_steps(ts, blocks, timeline_of, timelines, chunks)
        counts: Counter[float] = Counter()
        for lengths in lengths_of.values():
            counts.update(round(h, 6) for h in lengths)
        steps = tuple(sorted(counts.items(), key=lambda item: (-item[1], -item[0])))
        per_period = [len(lengths_of[ts]) for _p, ts in pairs if ts in lengths_of]
        mode = str(data.value("solve", solve, "solve_mode") or "single_solve")
        stochastic = data.value("solve", solve, "stochastic_branches")
        representative = tuple(
            (ts, len({str(rep) for _b, inner in _items(rp_raw[ts]) for rep, _w in _items(inner)}),
             len(rp_bases[ts]))
            for ts in dict.fromkeys(timesets_of.get(solve, [])) if ts in rp_raw)
        out.append(SolveFacts(
            name=solve, parent=parent, mode=mode,
            horizon=_number(data.value("solve", solve, "rolling_solve_horizon")),
            jump=_number(data.value("solve", solve, "rolling_solve_jump")),
            duration=_number(data.value("solve", solve, "rolling_duration")),
            periods=period_years, realized=realized, realized_invest=realized_invest, invest=invest,
            steps=steps,
            steps_per_period=(min(per_period), max(per_period)) if per_period else None,
            branches=_fans(stochastic) if stochastic is not None else (),
            representative=representative))
    return tuple(out)


def _timeset_steps(ts: str, blocks: dict, timeline_of: dict, timelines: dict, chunks: dict) -> list[float]:
    """The lengths (hours) of the steps of timeset *ts*'s blocks, on the solve's aggregated timeline when it
    has one (*chunks*, :func:`flextool.timeline_pieces.solve_step_chunks`)."""
    tl = timeline_of.get(ts)
    if tl is None or tl not in timelines:
        return []
    aggregated = chunks.get(tl)
    if aggregated is not None and ts in aggregated.block_chunks:
        steps = [(label, float(hours)) for label, hours in aggregated.chunks]
        ts_blocks = aggregated.block_chunks[ts]
    else:
        steps = timelines[tl]
        ts_blocks = blocks.get(ts, [])
    index_of = {label: i for i, (label, _h) in enumerate(steps)}
    out: list[float] = []
    for start, count in ts_blocks:
        first = index_of.get(start)
        if first is None:
            continue
        out.extend(h for _label, h in steps[first:first + int(float(count))])
    return out


# --- the run part -----------------------------------------------------------------------------------------------


def run_facts(ctx: AssistantContext, source: str | None, scenario: str) -> RunFacts | None:
    """The newest run of *scenario* from *source* (any source when ``None``), by its listing entry; ``None``
    when there is none."""
    from flextool import run_record
    from flextool.assistant.results import _runs

    found = _runs.select_runs(ctx, source=source, scenario=scenario)
    if not found:
        return None
    run = found[0]
    entry = run.entry
    status = run.status or {}
    error = None
    if not entry["results_usable"]:
        recorded = status.get("error") if isinstance(status, dict) else None
        message = recorded.get("message") if isinstance(recorded, dict) else None
        error = str(message) if message else (run_record.unusable_reason(run.status) if run.status else None)
    shares: tuple[tuple[str, float], ...] = ()
    if entry["results_usable"]:
        shares = _shares(run.folder)
    return RunFacts(folder=run.folder.name, state=str(entry["state"]), verdict=entry.get("verdict"),
                    verified=run.verified, results_usable=bool(entry["results_usable"]),
                    stale=entry.get("stale_vs_input"), error=error, shares=shares)


def _shares(folder: Path) -> tuple[tuple[str, float], ...]:
    """``((period, share of a year), ...)`` of the run's ``period_share_of_year__d`` (empty without it, or when
    it cannot be read now)."""
    path = folder / f"{SHARE_TABLE}.parquet"
    if not path.is_file():
        return ()
    try:
        from flextool.lean_parquet import read_lean_parquet

        frame = read_lean_parquet(path)
        if "period" in frame.columns:
            frame = frame.set_index("period")
        columns = [c for c in frame.columns
                   if (c[-1] if isinstance(c, tuple) else str(c).split(",")[-1].strip(" ()'\"")) == SHARE_COLUMN]
        column = columns[0] if columns else frame.columns[0]
        out = []
        for period, value in frame[column].items():
            number = _number(value)
            if number is not None and number > 0:
                out.append((str(period[-1] if isinstance(period, tuple) else period), number))
        return tuple(out)
    except Exception:  # noqa: BLE001 — a file being written, or of another layout: no scaling line
        return ()


# ---------------------------------------------------------------------------------------------------------------
# Rendering


@dataclass(frozen=True)
class _Cut:
    """How much of the facts one rendering shows (:func:`render` steps down until they fit)."""

    weights: bool = True
    periods: bool = True
    detailed: int = DETAILED_SOLVES
    scaling: bool = True
    realised: bool = True
    error_max: int = ERROR_MAX
    name_max: int = NAME_MAX
    sequence: int = SEQUENCE_NAMES
    step_groups: int = STEP_GROUPS
    group_names: int = GROUP_NAMES


#: The cut rule: weights, then periods, then the solves after the first, then the scaling line and the line of the
#: realised periods go; then the run
#: error is shortened and left out, the names shortened, and the step line keeps fewer step lengths and names.
#: The step and run lines always stay.
CUTS: tuple[_Cut, ...] = (
    _Cut(),
    _Cut(weights=False),
    _Cut(weights=False, periods=False),
    _Cut(weights=False, periods=False, detailed=1),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=60),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0, name_max=24, sequence=3),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0, name_max=12, sequence=2),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0, name_max=6, sequence=1),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0, name_max=6, sequence=1,
         step_groups=2, group_names=1),
    _Cut(weights=False, periods=False, detailed=1, scaling=False, realised=False, error_max=0, name_max=6, sequence=1,
         step_groups=1, group_names=0),
)


def _num(x: float) -> str:
    return f"{x:g}"


def _hours(x: float) -> str:
    return f"{x:g} h"


def _more(shown: list[str], total: int) -> str:
    text = ", ".join(shown)
    return f"{text} and {total - len(shown)} more" if total > len(shown) else text


class _Renderer:
    def __init__(self, facts: ScenarioFacts, redact: Callable[[str], str] | None, cut: _Cut) -> None:
        self.f = facts
        self.redact = redact
        self.cut = cut

    def name(self, value: str, limit: int | None = None) -> str:
        from flextool.assistant.prompt import _name

        return _name(value, self.redact, self.cut.name_max if limit is None else limit)

    def names(self, values: Iterable[str], most: int) -> str:
        values = list(values)
        if most <= 0:
            return f"{len(values)} solve{'' if len(values) == 1 else 's'}"
        return _more([self.name(v) for v in values[:most]], len(values))

    # -- solves --
    def mode(self, s: SolveFacts) -> str:
        if s.mode == "rolling_window":
            parts = [f"horizon {_hours(s.horizon)}" if s.horizon is not None else None,
                     f"jump {_hours(s.jump)}" if s.jump is not None else None,
                     f"rolling over {_hours(s.duration)}" if s.duration is not None else None]
            shown = [p for p in parts if p]
            text = "rolling window" + (f" ({', '.join(shown)})" if shown else "")
        elif s.mode == "single_solve":
            text = "single solve"
        else:
            text = f"solve_mode {self.name(s.mode)}"
        if s.branches:
            text += "; " + self.fans(s.branches)
        if s.representative:
            ts, reps, bases = s.representative[0]
            text += (f"; representative periods: {reps} for {bases} base period{'' if bases == 1 else 's'} "
                     f"(timeset {self.name(ts)}"
                     + (f" and {len(s.representative) - 1} more" if len(s.representative) > 1 else "") + ")")
        return text

    def fans(self, fans: tuple[BranchFan, ...]) -> str:
        first = fans[0]
        n = len(first.forecasts)
        names = [b for b, _w in first.forecasts]
        weights = [w for _b, w in first.forecasts]
        if self.cut.weights and n and all(abs(w - weights[0]) < 1e-9 for w in weights):
            listed = self.names(names, LIST_NAMES) + ", equal weights"
        elif self.cut.weights:
            shown = [f"{self.name(b)} {w:.3g}" for b, w in first.forecasts[:LIST_NAMES]]
            listed = _more(shown, n)
        else:
            listed = self.names(names, LIST_NAMES)
        text = f"stochastic: {n} forecast branch{'' if n == 1 else 'es'}" + (f" ({listed})" if n else "")
        if len(fans) > 1:
            text += f" at {len(fans)} start times"
        else:
            text += f" from {self.name(first.period)} {self.name(first.start)}"
        realized = next((fan.realized for fan in fans if fan.realized), None)
        if realized:
            text += f"; realized branch {self.name(realized)}"
        else:
            text += "; no realized branch, so the fan's operational results are empty"
        return text

    def unrealised(self, s: SolveFacts) -> str:
        """The periods of *s* whose operations it does not realise: look-ahead periods (neither operations nor
        investments realised; an investment look-ahead when it may invest in them) and periods whose investments
        only it realises; empty when it realises them all."""
        names = [p for p, _y in s.periods]
        operations, investments, invest = set(s.realized), set(s.realized_invest), set(s.invest)
        ahead = [p for p in names if p not in operations and p not in investments]
        groups = (([p for p in ahead if p in invest], "investment look-ahead, not realised"),
                  ([p for p in ahead if p not in invest], "look-ahead, not realised"),
                  ([p for p in names if p not in operations and p in investments],
                   "investments realised, operations not"))
        parts = []
        for group, words in groups:
            if not group:
                continue
            if len(group) == len(names) > 1:
                who = "all"
            elif len(group) <= LIST_NAMES:
                who = ", ".join(self.name(p) for p in group)
            else:
                who = f"{len(group)} periods"
            parts.append(f"{who} {words}")
        return "; ".join(parts)

    def periods(self, s: SolveFacts) -> str:
        periods = s.periods
        if not periods:
            return "no periods"
        names = [p for p, _y in periods]
        if len(names) == 1:
            shown = f"period {self.name(names[0])}"
        elif len(names) <= LIST_NAMES:
            shown = "periods " + ", ".join(self.name(p) for p in names)
        else:
            shown = f"{len(names)} periods {self.name(names[0])} to {self.name(names[-1])}"
        years = [y for _p, y in periods]
        low, high = min(years), max(years)
        if low == high:
            word = "year" if low == 1 else "years"
            text = f"{_num(low)} {word} represented{' each' if len(years) > 1 else ''}"
        else:
            text = f"{_num(low)} to {_num(high)} years represented"
        marks = self.unrealised(s)
        return f"{shown} ({text}{'; ' + marks if marks else ''})"

    def realised_line(self, solves: tuple[SolveFacts, ...]) -> str:
        """The periods whose operations some solve realises, in the order of the solves."""
        names = list(dict.fromkeys(p for s in solves for p in s.realized))
        if not names:
            shown = "none"
        elif len(names) <= LIST_NAMES:
            shown = ", ".join(self.name(p) for p in names)
        else:
            shown = f"{len(names)} periods {self.name(names[0])} to {self.name(names[-1])}"
        return f"Realised periods across all solves: {shown}."

    def solve_line(self, s: SolveFacts, *, only: bool, periods: bool = True, nested: bool = False) -> str:
        text = self.mode(s)
        if self.cut.periods and periods:
            text += "; " + self.periods(s)
        if only:
            return f"Solve {self.name(s.name)}: {text}."
        inside = f" (inside {self.name(s.parent)})" if s.parent and nested else ""
        return f"- {self.name(s.name)}{inside}: {text}."

    def sequence_line(self, solves: tuple[SolveFacts, ...]) -> str:
        shown = []
        for s in solves[:self.cut.sequence]:
            shown.append(self.name(s.name) + (f" (inside {self.name(s.parent)})" if s.parent else ""))
        return f"Solves: {_more(shown, len(solves))}."

    # -- steps --
    @staticmethod
    def step_text(s: SolveFacts) -> str:
        if not s.steps:
            return "not known"
        hours = [h for h, _n in s.steps]
        text = _hours(hours[0]) if len(hours) == 1 else f"{_num(min(hours))} to {_hours(max(hours))}"
        if s.steps_per_period is not None:
            low, high = s.steps_per_period
            if low == high:
                text += f", {low} step{'' if low == 1 else 's'} a period"
            else:
                text += f", {low} to {high} steps a period"
        return text

    def steps_line(self, solves: tuple[SolveFacts, ...]) -> str:
        if not solves:
            return "Time steps: not known (the input data names no solve to run)."
        groups: dict[str, list[str]] = {}
        for s in solves:
            groups.setdefault(self.step_text(s), []).append(s.name)
        if len(groups) == 1:
            (text,) = groups
            return f"Time steps: {text}{' in every solve' if len(solves) > 1 else ''}."
        most = self.cut.step_groups
        parts = [f"{text} ({self.names(names, self.cut.group_names)})" for text, names in list(groups.items())[:most]]
        if len(groups) > most:
            rest = sum(len(names) for names in list(groups.values())[most:])
            parts.append(f"other lengths in {rest} more solves")
        return "Time steps: " + "; ".join(parts) + "."

    def varying_line(self, solves: tuple[SolveFacts, ...]) -> str | None:
        varying = [s for s in solves if len(s.steps) > 1]
        if not varying:
            return None
        s = varying[0]
        counts = ", ".join(f"{n} of {_hours(h)}" for h, n in s.steps[:3])
        if len(s.steps) > 3:
            counts += f" and {len(s.steps) - 3} more lengths"
        where = counts if len(solves) == 1 else f"{self.name(s.name)}: {counts}"
        if len(varying) > 1:
            where += f"; and {len(varying) - 1} more solves"
        text = (f"FlexTool note: the time steps differ in length ({where}); weight per-step (_dt_) results by "
                "the step length when you add or average them.")
        if self.f.basic:
            text += " Use the per-period (_d_) results instead of per-step ones."
        return text

    # -- the run --
    def scaling_line(self, run: RunFacts | None) -> str | None:
        if run is None or not run.shares or not self.cut.scaling:
            return None
        groups: dict[str, list[tuple[str, float]]] = {}
        for period, share in run.shares:
            groups.setdefault(f"{share:.3g}", []).append((period, share))
        parts = []
        for text, members in list(groups.items())[:SHARE_GROUPS]:
            names = [p for p, _s in members]
            if len(names) <= LIST_NAMES:
                shown = ", ".join(self.name(p) for p in names)
            else:
                shown = f"{len(names)} periods {self.name(names[0])} to {self.name(names[-1])}"
            each = " each" if len(names) > 1 else ""
            parts.append(f"{shown}: {text}{each}, so × {1 / members[0][1]:.4g}")
        if len(groups) > SHARE_GROUPS:
            rest = sum(len(m) for m in list(groups.values())[SHARE_GROUPS:])
            parts.append(f"and {rest} more periods")
        # The categories are those of the annualised transforms of flextool.process_outputs._output_meta
        # (Semantics.ANNUALIZED: energy, annualised costs, emissions, inertia slack; COUNT_ANNUAL: start-ups); the
        # LEVEL, PRICE, AVERAGE, RATIO and DISCOUNTED ones are not scaled.
        return ("Annualised per-period (_d_) results (flows, energy, annualised costs, emissions, start-ups): per "
                "realised period, its weighted sum over its time steps divided by its share of a year "
                f"({'; '.join(parts)}). Capacities, prices, averages, ratios and discounted costs are not scaled.")

    def run_line(self, run: RunFacts | None) -> str:
        if run is None:
            return "Latest run: none of this scenario from this input source."
        folder = self.name(run.folder)
        if not run.verified:
            return f"Latest run: {folder} has result files but no run record; its success is not known."
        if run.state == "succeeded":
            text = f"Latest run: {folder} succeeded" + (f" ({self.name(run.verdict, NAME_MAX)})" if run.verdict else "")
        elif run.state == "running":
            text = f"Latest run: {folder} is still running"
        elif run.state == "failed":
            text = f"Latest run: {folder} failed"
        else:
            text = f"Latest run: {folder} ended {self.name(run.state, NAME_MAX)}"
        if not run.results_usable and run.state != "running":
            if run.error and self.cut.error_max > 0:
                text += f": {self.name(run.error, self.cut.error_max)}"
            text += "; its results are not used, see the run log"
        if run.stale:
            text += "; the input data changed after it"
        return text + "."

    def lines(self) -> list[str]:
        solves = self.f.solves
        out = [HEADER]
        if len(solves) == 1:
            out.append(self.solve_line(solves[0], only=True))
        elif solves:
            detailed = _detailed(solves, self.cut.detailed, self.step_text)
            every = len(detailed) == len(solves)                # each on its own line: no sequence line
            out.append("Solves:" if every else self.sequence_line(solves))
            # The same periods (and the same of them realised) in every solve: said once; else each detailed
            # solve says its own and one line gives the periods realised across all solves.
            shared = len({self.periods(s) for s in solves}) == 1
            out += [self.solve_line(s, only=False, periods=not shared, nested=every) for s in detailed]
            if shared and self.cut.periods:
                text = self.periods(solves[0])
                out.append(f"{text[0].upper()}{text[1:]} in every solve.")
            elif not shared and self.cut.realised:
                out.append(self.realised_line(solves))
        else:
            out.append("Solves: none found (model.solves names no solve of the scenario).")
        out.append(self.steps_line(solves))
        varying = self.varying_line(solves)
        if varying:
            out.append(varying)
        scaling = self.scaling_line(self.f.run)
        if scaling:
            out.append(scaling)
        out.append(self.run_line(self.f.run))
        return out


def _detailed(solves: tuple[SolveFacts, ...], most: int, step_text: Callable[[SolveFacts], str]
              ) -> list[SolveFacts]:
    """The solves described on their own line: the first, then those whose mode or time steps differ from every
    one described already, at most *most*."""
    def signature(s: SolveFacts) -> tuple:
        return (s.mode, s.horizon, s.jump, s.duration, bool(s.branches), bool(s.representative), step_text(s))

    picked: list[SolveFacts] = []
    for s in solves:
        if len(picked) >= most:
            break
        if not picked or all(signature(s) != signature(p) for p in picked):
            picked.append(s)
    return picked


def _tokens(text: str) -> int:
    from flextool.assistant.budget import approx_tokens

    return approx_tokens(text, "text")


def _cut_to(text: str, tokens: int) -> str:
    """*text* cut at its end (with an ellipsis) to at most *tokens* tokens; empty when not even a start fits."""
    from flextool.assistant.caps import cap_text

    while text and _tokens(text) > tokens:
        text = cap_text(text, len(text) - max(4, len(text) // 8)) if len(text) > 8 else ""
    return text


def render(facts: ScenarioFacts, redact: Callable[[str], str] | None = None, *, header: bool = True) -> str:
    """The facts as the context block shows them (*header*: with :data:`HEADER`): every name redacted and then
    cut, within :data:`FACTS_TOKENS_MAX` tokens (header included) by the steps of :data:`CUTS`; when the last step
    is still too long, the solve lines and then the step lines are cut at their end (the header and the run line
    stay whole)."""
    text = ""
    lines: list[str] = []
    for cut in CUTS:
        lines = _Renderer(facts, redact, cut).lines()
        text = "\n".join(lines)
        if _tokens(text) <= FACTS_TOKENS_MAX:
            break
    else:
        # The last step: the solve lines are cut at their end, then the step lines; the run line stays whole.
        steps_at = next(i for i, line in enumerate(lines) if line.startswith("Time steps: "))
        solves, steps, run = lines[1:steps_at], lines[steps_at:-1], lines[-1]
        solves_text = _cut_to("\n".join(solves), FACTS_TOKENS_MAX - _tokens("\n".join([HEADER, *steps, run])))
        head = "\n".join([HEADER, solves_text] if solves_text else [HEADER])
        steps_text = _cut_to("\n".join(steps), FACTS_TOKENS_MAX - _tokens(f"{head}\n{run}"))
        text = "\n".join(part for part in (head, steps_text, run) if part)
    if not header:
        text = text.removeprefix(HEADER).lstrip("\n")
    return text
