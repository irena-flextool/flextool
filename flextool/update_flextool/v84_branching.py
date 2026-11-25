"""Data version 84 (inside FlexTool 4.7.0): one branching setting of investment solves (item 1024).

``solve.invest_solve_branching_method`` replaces ``stochastic_invest_method``, ``shared_invest_periods`` and
``shared_operation_periods``; ``dispatch_realization_method`` ``separate_dispatch`` leaves investment solves.  The
step :func:`_migrate_v84_schema_batch` is run by ``db_migration``, which imports this module where it uses it (this
module uses the helpers of ``db_migration``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from spinedb_api import Array, Map, to_database

from flextool.update_flextool.db_migration import (
    BEHAVIOUR_CHANGE_MARKER_PREFIX,
    V83_SEPARATE_DISPATCH_MARKER,
    _V83_WITHIN_SOLVE,
    _already_applied,
    _append_list_members,
    _commit_step,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
    _replace_text_pairs,
    _scenario_alternatives,
    _v77_set_marker,
    _v83_dispatch_realization_texts,
    _v83_read,
    _v83_scenarios_text,
    _v83_write_values,
)


#: The value list and the members of ``solve.invest_solve_branching_method``, default first (the engine's
#: ``_solve_config.INVEST_SOLVE_BRANCHING_METHODS``; pinned by ``tests/update_flextool/test_v84_branching.py``).
_V84_LIST = "invest_solve_branching_methods"
_V84_METHODS: tuple[str, ...] = (
    "operations_only", "investments_after_reveal_period", "investments_at_reveal_period")
#: The parameters (and the value list) v84 replaces.
_V84_OLD_PARAMETERS: tuple[str, ...] = (
    "stochastic_invest_method", "shared_invest_periods", "shared_operation_periods")
_V84_OLD_LIST = "stochastic_invest_methods"
#: The behaviour-change marker of v84 (W-0637).
V84_MARKER = "stochastic_invest_controls_rewritten"
_V84_DESCRIPTION = (
    "[choice] Investment solves only (not rolling, with invest_periods or realized_invest_periods, and "
    "stochastic_branches): which decisions all forecast branches share. The reveal period is the period where the "
    "forecasts start. operations_only (default): the investments of every period are one plan for all branches; "
    "each branch operates on its own from the reveal period on. investments_after_reveal_period: the investments "
    "up to and including the reveal period are shared; each branch invests on its own in the later periods, which "
    "cannot be realized. investments_at_reveal_period: the investments before the reveal period are shared; each "
    "branch invests on its own from the reveal period on, and those periods cannot be realized.")


def _v84_definitions() -> dict[tuple[str, str], dict]:
    """The definition v84 adds, with its fields (the default is the list member ``operations_only``)."""
    return {("solve", "invest_solve_branching_method"): {
        "parameter_type_list": ("str",), "parameter_value_list_name": _V84_LIST,
        "parameter_group_name": "solve_advanced", "description": _V84_DESCRIPTION}}


def _v84_dispatch_realization_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the ``dispatch_realization_method`` description as v83 wrote it and its v84
    text (what an investment solve reads)."""
    (key, (_old, v83)), = _v83_dispatch_realization_texts().items()
    return {key: (v83, v83 + (
        " On an investment solve (not rolling, with invest_periods or realized_invest_periods) only deterministic "
        "is read, as a perfect-foresight benchmark with a warning: without it the forecasts are solved "
        "(invest_solve_branching_method); separate_dispatch is refused there."))}


def _v84_definitions_part(db, *, update: bool = True) -> None:
    """Part 1 of v84: the value list ``invest_solve_branching_methods`` (members appended when missing) and the
    definition ``solve.invest_solve_branching_method`` (default ``operations_only``, group ``solve_advanced``),
    or (``update``) its fields.  ``update=False`` only adds what is missing.  No value is written.  Idempotent."""
    if not db.get_item("entity_class", name="solve"):
        return
    if not db.get_item("parameter_value_list", name=_V84_LIST):
        db.add_parameter_value_list(name=_V84_LIST)
    _append_list_members(db, _V84_LIST, _V84_METHODS)
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    for (cls, name), fields in _v84_definitions().items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            val, typ = to_database(_V84_METHODS[0])
            db.add_parameter_definition(entity_class_name=cls, name=name, default_value=val, default_type=typ,
                                        **fields)
        elif update and any((tuple(pdef[k] or ()) if k == "parameter_type_list" else (pdef[k] or "")) != v
                            for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


def _v84_nonempty(value) -> bool:
    """Whether a period parameter value names a period (an Array or a map with an entry)."""
    if value is None:
        return False
    values = getattr(value, "values", None)
    return bool(values) if values is not None else True


def _v84_fan_periods(value) -> list[str]:
    """The periods of a ``stochastic_branches`` map (period, branch, analysis_time, realized -> weight) that have
    a forecast row (``realized`` other than ``yes``), in map order."""
    def forecast(inner, depth: int) -> bool:
        if depth == 3:
            return any(str(k).strip().lower() != "yes" for k in getattr(inner, "indexes", ()))
        return any(forecast(v, depth + 1) for v in getattr(inner, "values", ()) if isinstance(v, Map))

    if not isinstance(value, Map):
        return []
    return [str(p) for p, inner in zip(value.indexes, value.values) if isinstance(inner, Map) and forecast(inner, 1)]


def _v84_period_order(value) -> list[str]:
    """The periods of a ``period_timeset`` value in order (a 2-D map: its inner keys), else ``[]``."""
    if not isinstance(value, Map):
        return []
    out: list[str] = []
    for key, inner in zip(value.indexes, value.values):
        out.extend(str(k) for k in inner.indexes) if isinstance(inner, Map) else out.append(str(key))
    return list(dict.fromkeys(out))


def _v84_reveal(branches, order) -> str | None:
    """The reveal period: the first forecast period of *branches* in the period *order* (else in map order)."""
    periods = _v84_fan_periods(branches)
    if not periods:
        return None
    return min(periods, key=lambda p: (order.index(p), 0) if p in order else (len(order), periods.index(p)))


def _v84_shared_text(value) -> str:
    if value is None:
        return "unset"
    values = getattr(value, "values", None)
    return "[" + ", ".join(str(v) for v in values) + "]" if values is not None else repr(value)


@dataclass(frozen=True)
class BranchingPlan:
    """What v84 writes.  ``writes``: ``(solve, alternative, method)`` of ``invest_solve_branching_method``;
    ``converted``: ``(solve, alternative, text)`` of each written ``recourse`` row, the text naming the
    ``shared_invest_periods`` of each scenario where ``recourse`` wins;
    ``removed``: ``(solve, alternative)`` rows of ``dispatch_realization_method`` ``separate_dispatch`` that are
    removed (the solve is an investment solve in every scenario where the row wins); ``within_solve``: such rows
    that become ``within_solve`` instead, because removing them would let a lower-ranked value win;
    ``undecided``: ``(solve, reason)``, nothing written for the solve (each is listed too); ``listed``: ``{solve:
    reason}`` for the marker; ``drop_from_v83_marker``: the solves to drop from the v83 marker
    ``separate_dispatch_realization_looks_ahead`` (no ``separate_dispatch`` wins for them any more)."""
    writes: tuple = ()
    converted: tuple = ()
    removed: tuple = ()
    within_solve: tuple = ()
    undecided: tuple = ()
    listed: dict = field(default_factory=dict)
    drop_from_v83_marker: tuple = ()


def _v84_has_rows(value) -> bool:
    """Whether a ``stochastic_branches`` value has a row at all (the engine refuses a branching setting without)."""
    return isinstance(value, Map) and bool(value.indexes)


def _v84_has_realized_row(value) -> bool:
    """Whether a ``stochastic_branches`` map has a ``realized: yes`` row (a realization dispatch may follow)."""
    def realized(inner, depth: int) -> bool:
        if depth == 3:
            return any(str(k).strip().lower() == "yes" for k in getattr(inner, "indexes", ()))
        return any(realized(v, depth + 1) for v in getattr(inner, "values", ()) if isinstance(v, Map))

    return isinstance(value, Map) and any(realized(v, 1) for v in value.values if isinstance(v, Map))


def _v84_periods(value) -> list[str]:
    """The periods a period-list value names: an Array's members, a 2-D map's inner members (every row), in order."""
    if isinstance(value, Map):
        return list(dict.fromkeys(p for v in value.values for p in _v84_periods(v)))
    values = getattr(value, "values", None)
    if values is not None:
        return [str(v) for v in values]
    return [] if value is None else [str(value)]


@dataclass(frozen=True)
class _Reading:
    """How a solve reads in one scenario: ``investment`` (an investment solve), ``reveal`` (``None`` without a
    forecast row), ``has_rows`` (``stochastic_branches`` has a row), ``flat`` (it runs ``deterministic``: an explicit
    one on an investment solve, a blank, absent or explicit one on a dispatch solve; its forecasts are ignored),
    ``order`` (its periods in order), ``realized`` (its ``realized_periods`` and ``realized_invest_periods``) and
    ``realized_row`` (a ``realized: yes`` row)."""
    investment: bool
    reveal: str | None
    has_rows: bool
    flat: bool
    order: tuple
    realized: tuple
    realized_row: bool

    @property
    def branches(self) -> bool:
        """The solve's investments can branch here: an investment solve with forecasts that are solved."""
        return self.investment and self.reveal is not None and not self.flat

    def after_reveal(self, periods) -> list[str] | None:
        """*periods* at or after the reveal in the solve's period order; ``None`` when one is not in the order."""
        if self.reveal not in self.order or any(p not in self.order for p in periods):
            return None
        first = self.order.index(self.reveal)
        return [p for p in periods if self.order.index(p) >= first]

    def refusal(self, method: str) -> str | None:
        """Why the engine refuses *method* of ``invest_solve_branching_method`` here (``check_invest_branching``),
        else ``None``."""
        if not self.investment:
            return "it is a dispatch solve there (rolling, or without investments), where the setting is refused"
        if not self.has_rows:
            return "it has no stochastic_branches there, where the setting is refused"
        if self.flat or method != _V84_METHODS[1] or self.reveal not in self.order:
            return None
        later = self.order[self.order.index(self.reveal) + 1:]
        bad = [p for p in self.realized if p in later]
        if bad:
            return (f"it realizes {', '.join(bad)} after the reveal period {self.reveal} there, which "
                    f"{_V84_METHODS[1]} refuses (a per-branch investment cannot be realized): leave "
                    f"{'that period' if len(bad) == 1 else 'those periods'} out of realized_periods and "
                    f"realized_invest_periods (realize {'it' if len(bad) == 1 else 'them'} in a later solve) before "
                    f"setting {_V84_METHODS[1]}")
        return None


def plan_branching_methods(values, scenarios) -> BranchingPlan:
    """Where v84 writes ``invest_solve_branching_method`` and what it lists.  Pure: no database.

    *values* ``{parameter: {(solve, alternative): parsed value, None for a null row}}`` of ``solve_mode``,
    ``invest_periods``, ``realized_invest_periods``, ``realized_periods``, ``stochastic_branches``,
    ``period_timeset``, ``dispatch_realization_method`` and the three old parameters; *scenarios* ``{scenario:
    [alternative, ...]}``, lowest rank first.  In a scenario a solve is an *investment solve* when its winning
    ``solve_mode`` is not ``rolling_window`` and its winning ``invest_periods`` or ``realized_invest_periods`` names
    a period (:func:`_v83_read`: a winning null reads as unset); its reveal is the first forecast period of its
    winning ``stochastic_branches`` (:func:`_v84_reveal`); it is *flat* where it runs ``deterministic``
    (:class:`_Reading`).  The rule: an upgraded database never stops at run time on a value v84 wrote, and every
    solve whose results may change is listed.

    * ``stochastic_invest_method`` ``recourse`` had an effect only where the solve is an investment solve with
      forecasts that are solved.  A ``recourse`` row that wins in such a scenario becomes
      ``investments_after_reveal_period`` in its own alternative; any other row (``none``, null, or a ``recourse``
      that had no effect) becomes ``operations_only`` only where it would otherwise let such a write win in a
      scenario where it won, so every scenario reads what it read.  ``recourse`` without such a scenario writes
      nothing (it had no effect, or it was on a dispatch solve).  The solve is undecided (nothing written) when,
      under ``recourse``, ``shared_invest_periods`` ties a period after the reveal, or when a written value would
      be refused in a scenario (:meth:`_Reading.refusal`: on a dispatch solve, without ``stochastic_branches``, or
      ``investments_after_reveal_period`` realizing a period after the reveal).  Listed: ``recourse`` on a dispatch
      solve that is not flat (its realization changes); a converted solve whose ``shared_invest_periods`` tied no
      period from the reveal on (unset, ``[]`` or only earlier periods: each branch invested on its own from the
      reveal period); a converted solve with a ``realized: yes`` row (a realization dispatch may now follow it).
    * ``shared_operation_periods``: listed for a stochastic investment solve, not flat, where it is unset or a list
      (its forecasts were tied, now they are free) and for a stochastic dispatch solve, not flat, where it is set
      (no longer read).
    * ``dispatch_realization_method`` ``separate_dispatch``: a row that wins only in scenarios where the solve is an
      investment solve (the engine refuses ``separate_dispatch`` there, and an investment solve reads no other
      value but an explicit ``deterministic``) is removed and the solve listed.  Where removing the rows would
      let a lower-ranked value other than ``within_solve`` win in such a scenario (an explicit ``deterministic``
      would flatten the solve), the row becomes ``within_solve`` instead, which reads as no value.  A row that also
      wins where the solve is a dispatch solve is kept (undecided).  ``drop_from_v83_marker``: the solves where no
      ``separate_dispatch`` wins any more.

    Every undecided solve is listed too, with what to set.
    """
    def table(name):
        return values.get(name, {})

    def read(name, solve, alts):
        return _v83_read(alts, table(name), solve)

    def text(value) -> str:
        return str(value or "").strip().lower()

    solves = sorted({key[0] for name in values for key in values[name]})
    ordered = sorted(scenarios.items())
    writes, converted, removed, within, undecided, listed, drop = [], [], [], [], [], {}, []

    def add_reason(solve, reason):
        listed[solve] = f"{listed[solve]}; {reason}" if solve in listed else reason

    def undecide(solve, reason, effect):
        undecided.append((solve, reason))
        add_reason(solve, f"nothing written by the upgrade ({reason}); {effect}")

    for solve in solves:
        states: dict[str, _Reading] = {}
        for s, alts in ordered:
            investment = bool(read("solve_mode", solve, alts)[0] != "rolling_window" and (
                _v84_nonempty(read("invest_periods", solve, alts)[0])
                or _v84_nonempty(read("realized_invest_periods", solve, alts)[0])))
            branches = read("stochastic_branches", solve, alts)[0]
            order = _v84_period_order(read("period_timeset", solve, alts)[0])
            method = text(read("dispatch_realization_method", solve, alts)[0])
            states[s] = _Reading(
                investment=investment, reveal=_v84_reveal(branches, order), has_rows=_v84_has_rows(branches),
                flat=method == "deterministic" or (not investment and method == ""), order=tuple(order),
                realized=tuple(dict.fromkeys(_v84_periods(read("realized_periods", solve, alts)[0])
                                             + _v84_periods(read("realized_invest_periods", solve, alts)[0]))),
                realized_row=_v84_has_realized_row(branches))
        rows = {a: v for (e, a), v in table("stochastic_invest_method").items() if e == solve}
        winner = {s: read("stochastic_invest_method", solve, alts)[1] for s, alts in ordered}
        recourse_wins = [s for s, _alts in ordered if text(rows.get(winner[s])) == "recourse"]
        dispatch_wins = [s for s in recourse_wins if not states[s].investment and not states[s].flat]
        branch_wins = [s for s in recourse_wins if states[s].branches]
        if dispatch_wins:
            add_reason(solve, f"stochastic_invest_method recourse dropped on a dispatch solve "
                              f"({_v83_scenarios_text(dispatch_wins)}): it has no investments to branch, and its "
                              "forecasts are now realized as on any dispatch solve")
        if branch_wins:
            problems, wait_and_see, readings = [], [], []
            for s in branch_wins:
                reading = states[s]
                shared = read("shared_invest_periods", solve, scenarios[s])[0]
                periods = [str(v) for v in getattr(shared, "values", None) or []]
                readings.append(f"{s}: {_v84_shared_text(shared)}")
                if shared is None:
                    tied = []
                elif not isinstance(shared, Array):
                    tied = None
                else:
                    tied = reading.after_reveal(periods)
                    if tied is None and periods == [reading.reveal]:
                        tied = periods
                if tied == []:
                    wait_and_see.append(s)
                elif tied != [reading.reveal]:
                    problems.append(f"in scenario {s} shared_invest_periods is {_v84_shared_text(shared)}, which "
                                    f"ties investments after the reveal period {reading.reveal!r}")
            plan = {winner[s]: _V84_METHODS[1] for s in branch_wins}
            changed = True
            while changed:
                changed = False
                for s, alts in ordered:
                    if winner[s] is None or winner[s] in plan:
                        continue
                    top = next((a for a in reversed(alts) if a in plan), None)
                    if top is not None and plan[top] == _V84_METHODS[1]:
                        plan[winner[s]] = _V84_METHODS[0]
                        changed = True
            for s, alts in ordered:
                top = next((a for a in reversed(alts) if a in plan), None)
                why = None if top is None else states[s].refusal(plan[top])
                if why:
                    problems.append(f"{plan[top]} would win in scenario {s}, but {why}")
            if problems:
                undecide(solve, "; ".join(problems),
                         "it runs operations_only (the default) until invest_solve_branching_method is set")
            else:
                for alt, method in sorted(plan.items()):
                    writes.append((solve, alt, method))
                    if method == _V84_METHODS[1]:
                        converted.append((solve, alt, "; ".join(readings)))
                if wait_and_see:
                    add_reason(solve, "investments_after_reveal_period written for recourse with "
                                      "shared_invest_periods tying no period from the reveal on "
                                      f"({_v83_scenarios_text(wait_and_see)}): the reveal period's investment is "
                                      "now shared by the branches")
                realization = [s for s in branch_wins if states[s].realized_row]
                if realization:
                    add_reason(solve, f"recourse solve with a realized: yes branch ({_v83_scenarios_text(realization)})"
                                      ": investments_after_reveal_period gets a realization dispatch "
                                      f"({solve}__realized_dispatch) where the realized branch reaches the realized "
                                      "periods; recourse got none")
        tied, untied_dispatch = [], []
        for s, alts in ordered:
            reading = states[s]
            if reading.reveal is None or reading.flat:
                continue
            ops, where = read("shared_operation_periods", solve, alts)
            if reading.investment and (ops is None or _v84_nonempty(ops)):
                tied.append(s)
            elif not reading.investment and where is not None and ops is not None:
                untied_dispatch.append(s)
        if tied:
            add_reason(solve, f"investment solve whose forecasts were tied ({_v83_scenarios_text(tied)}): each "
                              "branch now operates on its own from the reveal period")
        if untied_dispatch:
            add_reason(solve, f"dispatch solve with shared_operation_periods ({_v83_scenarios_text(untied_dispatch)}):"
                              " no longer read, its forecasts are tied over its realized_periods")
        methods = table("dispatch_realization_method")
        candidates, win_scenarios = [], {}
        for alt, v in sorted((a, v) for (e, a), v in methods.items() if e == solve):
            if text(v) != "separate_dispatch":
                continue
            wins = [s for s, alts in ordered if read("dispatch_realization_method", solve, alts)[1] == alt]
            if wins and all(states[s].investment for s in wins):
                candidates.append(alt)
                win_scenarios[alt] = wins
            elif any(states[s].investment for s in wins):
                undecide(solve, f"dispatch_realization_method separate_dispatch in alternative {alt} wins where the "
                                "solve is an investment solve and where it is a dispatch solve",
                         "the run stops where it is an investment solve until the value is moved to an alternative "
                         "where it is a dispatch solve")
        without = {k: v for k, v in methods.items() if not (k[0] == solve and k[1] in candidates)}
        for alt in candidates:
            surfaced = {text(_v83_read(scenarios[s], without, solve)[0]) for s in win_scenarios[alt]}
            if surfaced <= {"", _V83_WITHIN_SOLVE}:
                removed.append((solve, alt))
                add_reason(solve, f"dispatch_realization_method separate_dispatch in alternative {alt} removed "
                                  "(refused on an investment solve)")
            else:
                within.append((solve, alt))
                add_reason(solve, f"dispatch_realization_method separate_dispatch in alternative {alt} became "
                                  "within_solve (refused on an investment solve; removing it would expose a "
                                  "lower-ranked value)")
        after = {**without, **{(solve, a): _V83_WITHIN_SOLVE for e, a in within if e == solve}}
        if candidates and not any(text(_v83_read(alts, after, solve)[0]) == "separate_dispatch"
                                  for _s, alts in ordered):
            drop.append(solve)
    return BranchingPlan(tuple(writes), tuple(converted), tuple(removed), tuple(within), tuple(undecided), listed,
                         tuple(drop))


def _v84_inputs(db) -> dict:
    """:func:`plan_branching_methods` *values* read from *db* (parsed values; a null row is kept as ``None``)."""
    names = ("solve_mode", "invest_periods", "realized_invest_periods", "realized_periods", "stochastic_branches",
             "period_timeset", "dispatch_realization_method", *_V84_OLD_PARAMETERS)
    values: dict = {}
    for name in names:
        values[name] = {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
                        for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name=name)}
    return values


def _v84_branching(db) -> None:
    """Part 2 of v84: ``invest_solve_branching_method`` written from ``stochastic_invest_method``
    (:func:`plan_branching_methods`), the ``separate_dispatch`` of investment solves removed (or made
    ``within_solve`` where a lower-ranked value would win; and
    those solves dropped from the v83 marker ``separate_dispatch_realization_looks_ahead``), the solves whose
    results change listed in the marker ``stochastic_invest_controls_rewritten`` (``{"since": 84, "solves": [...],
    "reasons": {solve: text}}``, merged with a stored one), and the three old parameters and the value list
    ``stochastic_invest_methods`` removed with their values.  Without the old definitions (a rerun) nothing is
    planned.  Idempotent; no commit."""
    if not db.get_item("entity_class", name="solve"):
        return
    old = [pdef for name in _V84_OLD_PARAMETERS if (pdef := _pdef_or_none(db, "solve", name)) is not None]
    if not old:
        _already_applied(84, "invest_solve_branching_method: already converted")
    else:
        plan = plan_branching_methods(_v84_inputs(db), _scenario_alternatives(db))
        _v83_write_values(db, "solve", "invest_solve_branching_method", plan.writes)
        _v83_write_values(db, "solve", "dispatch_realization_method",
                          [(solve, alt, _V83_WITHIN_SOLVE) for solve, alt in plan.within_solve])
        if plan.removed:
            gone = set(plan.removed)
            db.remove_items("parameter_value", *[
                pv["id"] for pv in db.find_parameter_values(entity_class_name="solve",
                                                            parameter_definition_name="dispatch_realization_method")
                if (pv["entity_byname"][0], pv["alternative_name"]) in gone])
        if plan.converted:
            print("v84: stochastic_invest_method recourse became invest_solve_branching_method "
                  "investments_after_reveal_period (the reveal period's investment is shared by the branches; the "
                  "later periods invest per branch; investments_at_reveal_period lets every branch invest on its "
                  "own from the reveal period):")
            for solve, alt, shared in plan.converted:
                print(f"  solve {solve} [{alt}]; shared_invest_periods in {shared}")
        for solve, reason in plan.undecided:
            print(f"v84: solve {solve}: nothing written ({reason}). Set invest_solve_branching_method on it "
                  "yourself: investments_after_reveal_period shares the reveal period's investment, "
                  "investments_at_reveal_period shares only the periods before it, operations_only (the default) "
                  "shares every period; dispatch_realization_method separate_dispatch is refused on an investment "
                  "solve.")
        for solve, alt in plan.removed:
            print(f"v84: solve {solve} [{alt}]: dispatch_realization_method separate_dispatch removed (refused on an "
                  "investment solve; with a realized: yes branch the solve gets a realization dispatch of its own, "
                  "results change; for a stochastic dispatch after the investments, add a dispatch solve after it)")
        for solve, alt in plan.within_solve:
            print(f"v84: solve {solve} [{alt}]: dispatch_realization_method separate_dispatch became within_solve "
                  "(refused on an investment solve; removing it would let a lower-ranked value win; with a "
                  "realized: yes branch the solve gets a realization dispatch of its own, results change)")
        for solve, reason in plan.listed.items():
            if "dispatch solve with shared_operation_periods" in reason:
                print(f"v84: solve {solve}: shared_operation_periods is no longer read; its forecasts are tied over "
                      "its committed window, the realized_periods (results change). To leave a period untied, "
                      "leave it out of realized_periods.")
        if plan.drop_from_v83_marker:
            _v84_drop_from_marker(db, V83_SEPARATE_DISPATCH_MARKER, set(plan.drop_from_v83_marker))
        if plan.listed:
            name = BEHAVIOUR_CHANGE_MARKER_PREFIX + V84_MARKER
            stored: dict = {}
            for md in db.get_items("metadata"):
                if md["name"] == name:
                    try:
                        stored = dict(json.loads(md["value"]))
                    except (TypeError, ValueError):
                        stored = {}
            solves = list(stored.get("solves") or [])
            reasons = dict(stored.get("reasons") or {})
            for solve, reason in plan.listed.items():
                if solve not in solves:
                    solves.append(solve)
                reasons.setdefault(solve, reason)
            _v77_set_marker(db, V84_MARKER, json.dumps({"since": 84, "solves": solves, "reasons": reasons}))
        _remove_items_bounded(db, "parameter_definition", [pdef["id"] for pdef in old],
                              "v84: removed solve.stochastic_invest_method, shared_invest_periods and "
                              "shared_operation_periods")
    old_list = db.get_item("parameter_value_list", name=_V84_OLD_LIST)
    if old_list and not any(pd["parameter_value_list_name"] == _V84_OLD_LIST
                            for pd in db.get_items("parameter_definition")):
        db.remove_item("parameter_value_list", old_list["id"])


def _v84_drop_from_marker(db, change_id: str, solves: set) -> None:
    """Remove *solves* from the ``solves`` list of the marker *change_id*; the marker goes when the list empties."""
    name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
    for md in list(db.get_items("metadata")):
        if md["name"] != name:
            continue
        try:
            value = dict(json.loads(md["value"]))
        except (TypeError, ValueError):
            continue
        kept = [s for s in value.get("solves") or [] if s not in solves]
        if kept:
            value["solves"] = kept
            db.update_item("metadata", id=md["id"], value=json.dumps(value))
        else:
            db.remove_item("metadata", md["id"])


def _migrate_v84_schema_batch(db) -> None:
    """All schema changes of database version 84 (v83 -> v84), in one step; each part is idempotent, and the step
    commits once at the end.

    1. The value list ``invest_solve_branching_methods`` and the definition ``solve.invest_solve_branching_method``
       (:func:`_v84_definitions_part`).
    2. ``stochastic_invest_method``, ``shared_invest_periods`` and ``shared_operation_periods`` replaced by it
       (:func:`_v84_branching`).
    3. The ``dispatch_realization_method`` description says what an investment solve reads (by text alone).
    """
    _v84_definitions_part(db)
    _v84_branching(db)
    _replace_text_pairs(db, _v84_dispatch_realization_texts())
    _commit_step(db, "v84: invest_solve_branching_method replaces stochastic_invest_method, shared_invest_periods "
                     "and shared_operation_periods")
