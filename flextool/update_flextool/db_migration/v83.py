"""Data version 83: minimum_time_method, startup methods, realization notices, descriptions."""
import json
from dataclasses import dataclass
from typing import NamedTuple

from spinedb_api import from_database, Map, to_database

from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _append_list_members,
    _attach_list_defaults,
    _commit_step,
    _describe_pv,
    _detach_list_defaults,
    _list_members,
    _pdef_or_none,
    _replace_text_pairs,
    BEHAVIOUR_CHANGE_MARKER_PREFIX,
    FlexToolMigrationError,
)
from flextool.update_flextool.db_migration.v77 import (
    _inactive_in_scenarios,
    _merge_marker_lists,
    _scenario_alternatives,
    _unit_count_vu_text,
    _V77_NO_SCENARIO,
    _v77_set_marker,
    _v77_winner,
)
from flextool.update_flextool.db_migration.v82 import _v82_definitions_part, _v82_superseded_texts


# --- Version 83 part 1: minimum_time_method (C rev 3 §6) --------------------------------------------------------

#: ``startup_method`` values that give a unit online variables, in both vocabularies (part 2 renames ``binary``; a
#: rerun of part 1 after a kill sees the new names).
_V83_ONLINE = frozenset(("linear", "binary", "linear_fix_in_uc", "integer", "integer_fix_in_uc"))
#: The methods that enable each time.
_V83_ENABLES = {"min_uptime": frozenset(("min_uptime", "both")), "min_downtime": frozenset(("min_downtime", "both"))}
#: The members of ``minimum_time_methods``, in the order part 1 appends the missing ones.
_V83_MIN_TIME_METHODS = ("min_uptime", "min_downtime", "both", "none")
#: The unit parameters whose parsed values decide a unit's online variables and min up/down rows.
_V83_READ = ("startup_method", "conversion_method", "piecewise_linear_efficiency", "minimum_time_method",
             "min_uptime", "min_downtime")
#: The behaviour-change marker of part 1 (W-0634).
V83_MIN_TIME_MARKER = "minimum_time_method_written"
_V83_MEMBER_MESSAGE = ("A version 83 upgrade step changed an existing member of the value list {list} ({members}); "
                       "the upgrade stops so that no default or value changes with it. The file was restored from "
                       "the backup. Please report this.")


@dataclass(frozen=True)
class MinTimePlan:
    """What part 1 of v83 writes.  ``writes``: ``(unit, alternative, method)``; ``undecided``: ``(unit, reason,
    candidate scenarios, ((candidate scenario, the kinds it needs), ...))`` (nothing is written for such a unit)."""
    writes: tuple = ()
    undecided: tuple = ()


def _v83_write_values(db, entity_class: str, name: str, writes) -> None:
    """Write each ``(entity, alternative, value)`` of *writes* into *name* of *entity_class*, by value: a row
    already there (a null one: :func:`_v83_read` reads it as unset) is updated, else one is added."""
    rows = {(pv["entity_byname"][0], pv["alternative_name"]): pv["id"]
            for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=name)}
    for entity, alternative, value in writes:
        val, typ = to_database(value)
        if (entity, alternative) in rows:
            db.update_parameter_value(id=rows[(entity, alternative)], value=val, type=typ)
        else:
            db.add_parameter_value(entity_class_name=entity_class, entity_byname=(entity,),
                                   parameter_definition_name=name, alternative_name=alternative, value=val, type=typ)


def _v83_scenarios_text(scenarios) -> str:
    """``scenario S``, ``scenarios S1, S2`` or, without scenarios (:data:`_V77_NO_SCENARIO`), ``the database``
    (the v83 prints)."""
    if tuple(scenarios) == (_V77_NO_SCENARIO,):
        return "the database"
    return f"scenario{'s' if len(scenarios) > 1 else ''} {', '.join(scenarios)}"


def _v83_check_members(db, name: str, before: dict, removed=frozenset()) -> None:
    """The v83 member guard: every member of the value list *name* in *before* (:func:`_list_members`, taken before
    the part) that the part does not remove on purpose (*removed*: database ids) is still there with the same
    ``(list, value, type)``, and every id of *removed* is gone.  Raises :class:`FlexToolMigrationError` (before the
    step's commit) otherwise."""
    after = _list_members(db, [name])
    changed = sorted(f"{from_database(v, t)!r}{' (its planned removal did not happen)' if i in removed else ''}"
                     for i, (_lst, v, t) in before.items()
                     if (i in after if i in removed else after.get(i) != (_lst, v, t)))
    if changed:
        raise FlexToolMigrationError(_V83_MEMBER_MESSAGE.format(list=name, members=", ".join(changed)))


def _v83_is_time(value) -> bool:
    """The engine's test for a min up/down time: ``float(v) > 0``, of any leaf of a non-scalar value."""
    leaves = getattr(value, "values", None)
    if leaves is not None:
        return any(_v83_is_time(v) for v in leaves)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _v83_read(alternatives, table, key, default=None) -> tuple:
    """``(value, alternative)`` of *key* in a scenario of *alternatives* (lowest rank first), as the engine reads
    it: the value of the winning row of *table* ``{(key, alternative): parsed value}`` (the highest-ranked
    alternative holding a row, a null one included), else *default*; ``alternative`` is where that row is (None
    without one).  A winning null reads as unset, so *default* applies, not a lower alternative's value: the engine's
    scenario filter keeps the top row, ``SpineDbReader._unroll_rows`` drops a whole-null value and
    ``_parameter_frame`` fills in the definition default (``engine/input/_spinedb_reader.py``); the solve reader
    ``params_to_dict`` (``engine/input/_solve_config.py``) skips it, and
    ``SolveConfig.dispatch_realization_method_for`` reads the absent value as within_solve."""
    where = _v77_winner(alternatives, table, key)
    value = table[(key, where)] if where is not None else None
    return (default if value is None else value), where


def _v83_method(unit: str, alternatives, table, default) -> tuple:
    """``(resolves, method, alternative)`` of *unit*'s ``minimum_time_method`` in a scenario of *alternatives*
    (:func:`_v83_read`: a winning null reads the definition *default*).  It resolves when a non-null value wins, or
    when the default that reads is a member other than ``none``; ``alternative``: where the winning row is (a null
    row there still hides a write below it)."""
    method, where = _v83_read(alternatives, table, unit, default)
    explicit = where is not None and table[(unit, where)] is not None
    return explicit or method not in (None, "none"), method, where


class _V83State(NamedTuple):
    """A unit in one scenario as the engine reads it before data version 83 (C rev 3 §6.2)."""
    online: bool  #: it has online variables
    need: frozenset  #: the kinds with a time when online (their rows are built)
    resolves: bool  #: a minimum_time_method value wins, or the default is a member other than none
    method: object  #: the method that reads
    where: str | None  #: where the winning method row is (None for none)
    off_by: tuple  #: when not online: the alternatives whose winning values keep it off (startup_method, then
    #: conversion_method, then the curve)


def _v83_unit_state(unit: str, alternatives, values, defaults) -> _V83State:
    """*unit* in a scenario of *alternatives* (lowest rank first) as the engine reads it before data version 83.
    ``online``: ``startup_method`` in :data:`_V83_ONLINE`, else ``no_startup`` with ``minimum_time_method``
    promoting it, else with a ``piecewise_linear`` curve with ``auto_startup``.  Every value through
    :func:`_v83_read` (a winning null reads the default)."""
    def read(name):
        return _v83_read(alternatives, values.get(name, {}), unit, defaults.get(name))

    startup, st_alt = read("startup_method")
    startup = startup or "no_startup"
    resolves, method, where = _v83_method(unit, alternatives, values.get("minimum_time_method", {}),
                                          defaults.get("minimum_time_method"))
    conversion, ct_alt = read("conversion_method")
    curve, curve_alt = read("piecewise_linear_efficiency")
    piecewise = (conversion or "constant_efficiency") == "piecewise_linear"
    online = startup in _V83_ONLINE or startup == "no_startup" and (
        method in ("min_uptime", "min_downtime", "both") or piecewise and bool(curve))
    need = frozenset(k for k in _V83_ENABLES if online and _v83_is_time(read(k)[0]))
    off_by = () if online else tuple(a for a, keeps_off in (
        (st_alt, True), (ct_alt, startup == "no_startup" and not piecewise),
        (curve_alt, startup == "no_startup" and piecewise and not curve)) if a is not None and keeps_off)
    return _V83State(online, need, resolves, method, where, off_by)


def _v83_unit_writes(unit, method, candidates, before, values, held, scenarios, defaults) -> tuple[dict, str | None]:
    """C rev 3 §6.3 steps 3-5 for one unit: ``({alternative: method}, None)``, or ``({}, reason)`` when it stays
    undecided.  *held*: the ``(unit, alternative)`` holding any value of the unit."""
    planned: dict[str, str] = {}
    for s in candidates:
        target = _v77_winner(scenarios[s], values.get("startup_method", {}), unit)
        if target is None:
            target = _v77_winner(scenarios[s], held, unit)
        null_row = before[s].where  # a winning null method row (a candidate resolves no method), None without one
        if null_row is not None and (target is None or scenarios[s].index(null_row) > scenarios[s].index(target)):
            target = null_row  # it reads as not set and would hide a write below it: the write goes into it
        if target is None:
            return {}, f"no alternative of scenario {s} holds a value of the unit"
        planned[target] = method
    own = values.get("minimum_time_method", {})
    for s, state in before.items():
        alts = scenarios[s]
        winner = _v77_winner(alts, {**own, **{(unit, a): m for a, m in planned.items()}}, unit)
        if not state.online and winner in planned and planned[winner] != "none":
            # The switch-off: none into the alternative that keeps u off in s, else (it ranks too low to win, or
            # there is none) into the highest alternative of s.
            off = next((a for a in state.off_by if alts.index(a) > alts.index(winner)), alts[-1])
            if off in planned:
                return {}, f"a value in alternative {off} would give the unit online variables in scenario {s}"
            planned[off] = "none"
    table = {**own, **{(unit, a): m for a, m in planned.items()}}
    for s, state in before.items():
        new_resolves, new, winner = _v83_method(unit, scenarios[s], table, defaults.get("minimum_time_method"))
        new = new if new_resolves else None
        if state.resolves and new != state.method:
            return {}, (f"a value in alternative {winner} would replace your minimum_time_method in alternative "
                        f"{state.where} in scenario {s}")
        if s in candidates and not all(new in _V83_ENABLES[k] for k in state.need):
            return {}, f"the values written would not enable its {' and '.join(sorted(state.need))} in scenario {s}"
        if not state.online and new in ("min_uptime", "min_downtime", "both"):
            return {}, f"a value in alternative {winner} would give the unit online variables in scenario {s}"
    return planned, None


def plan_minimum_time_method(units, values, scenarios, defaults, inactive=frozenset()) -> MinTimePlan:
    """Where part 1 of v83 writes ``minimum_time_method`` so that every unit keeps its min up/down rows once the
    engine reads the method (C rev 3 §6.3).  Pure: no database.

    *units*: the unit names; *values* ``{parameter: {(unit, alternative): parsed value}}`` with every stored unit
    parameter (those outside :data:`_V83_READ` only mark the alternatives that hold a value of the unit; a null
    value is kept: it reads the default), the value of ``piecewise_linear_efficiency`` being ``auto_startup`` of its
    curves; *scenarios* ``{scenario: [alternative, ...]}`` lowest rank first; *defaults* ``{parameter: parsed
    definition default}``; *inactive* ``{(scenario, unit)}`` (left out of that scenario).

    A unit's candidate scenarios are those where it has rows today and no method resolves.  It gets ``both`` when
    they need both times, else the one time, written into each candidate scenario's ``startup_method`` winner (else
    its highest alternative holding any value of the unit), or into the winning null ``minimum_time_method`` row
    when that one ranks higher (it reads as not set; its row is updated), plus ``none`` where the written method would win in a
    scenario where the unit has no online variables today: into the alternative whose value keeps it off there
    (``startup_method``, else ``conversion_method``, else the curve), else, when that one ranks too low to win, into
    the scenario's highest alternative.  Then every scenario is checked: a method that resolved is unchanged, every
    needed time is enabled, a unit without online variables gets no enabling method.  A unit that fails any of these
    gets no write (undecided, the reason of the first failing scenario by name)."""
    writes, undecided = [], []
    held: dict[str, set] = {}
    for table in values.values():
        for key in table:
            held.setdefault(key[0], set()).add(key)
    for unit in sorted(units):
        before = {s: _v83_unit_state(unit, scenarios[s], values, defaults)
                  for s in sorted(scenarios) if (s, unit) not in inactive}
        candidates = [s for s, state in before.items() if state.need and not state.resolves]
        if not candidates:
            continue
        kinds = frozenset().union(*(before[s].need for s in candidates))
        method = "both" if len(kinds) == 2 else next(iter(kinds))
        planned, reason = _v83_unit_writes(unit, method, candidates, before, values, held.get(unit, set()),
                                           scenarios, defaults)
        if reason is None:
            writes.extend((unit, alternative, m) for alternative, m in sorted(planned.items()))
        else:
            undecided.append((unit, reason, tuple(candidates),
                              tuple((s, tuple(k for k in _V83_ENABLES if k in before[s].need)) for s in candidates)))
    return MinTimePlan(writes=tuple(writes), undecided=tuple(undecided))


def _v83_unit_inputs(db) -> tuple:
    """:func:`plan_minimum_time_method` arguments read from *db* (parsed values only, C rev 3 §6.1)."""
    from flextool.engine.input._piecewise_curve import piecewise_unit_flags
    from flextool.input_derivation._piecewise import curves_of_value

    units = sorted(e["name"] for e in db.get_items("entity", entity_class_name="unit"))
    values: dict[str, dict] = {}
    for pv in db.find_parameter_values(entity_class_name="unit"):
        name = pv["parameter_definition_name"]
        value = pv["parsed_value"] if name in _V83_READ else True
        if name == "piecewise_linear_efficiency" and value is not None:
            curves, problems = curves_of_value(value)
            value = not problems and piecewise_unit_flags(points for _p, points in curves).auto_startup
        values.setdefault(name, {})[(pv["entity_byname"][0], pv["alternative_name"])] = value
    defaults = {}
    for name in _V83_READ:
        pdef = _pdef_or_none(db, "unit", name)
        if name != "piecewise_linear_efficiency" and pdef is not None and pdef["default_value"] is not None:
            defaults[name] = from_database(pdef["default_value"], pdef["default_type"])
    scenarios = _scenario_alternatives(db)
    return units, values, scenarios, defaults, _inactive_in_scenarios(db, "unit", units, scenarios)


def _v83_minimum_time_method(db) -> None:
    """Part 1 of v83: the engine now builds a unit's min up/down rows only when ``minimum_time_method`` enables the
    time.  Appends the missing members of ``minimum_time_methods``, then writes the method where a unit relied on a
    time alone (:func:`plan_minimum_time_method`, results unchanged), and records the writes and the undecided units
    in the marker ``minimum_time_method_written`` (``{"since": 83, "writes": [[unit, alternative, method], ...],
    "undecided": [[unit, reason, [scenario, ...]], ...]}``; writes merged, undecided replaced).  Skips a database
    without the class, the definition or the list.  Idempotent; no commit."""
    if (not db.get_item("entity_class", name="unit") or _pdef_or_none(db, "unit", "minimum_time_method") is None
            or not db.get_item("parameter_value_list", name="minimum_time_methods")):
        return
    before = _list_members(db, ["minimum_time_methods"])
    _append_list_members(db, "minimum_time_methods", _V83_MIN_TIME_METHODS)
    _v83_check_members(db, "minimum_time_methods", before)  # the writes below touch no member
    marked = any(md["name"] == BEHAVIOUR_CHANGE_MARKER_PREFIX + V83_MIN_TIME_MARKER for md in db.get_items("metadata"))
    plan = plan_minimum_time_method(*_v83_unit_inputs(db))
    _v83_write_values(db, "unit", "minimum_time_method", plan.writes)
    if marked and not plan.writes:
        _already_applied(83, "minimum_time_method: already written")
    if plan.writes:
        print("v83: minimum_time_method written so that min_uptime / min_downtime keep applying (results unchanged):")
        for unit, alternative, method in plan.writes:
            print(f"  unit {unit} [{alternative}]: {method}")
    for unit, reason, _candidates, needs in plan.undecided:
        groups: dict[tuple, list] = {}
        for s, kinds in needs:
            groups.setdefault(kinds, []).append(s)
        lost = "; ".join(f"in {_v83_scenarios_text(ss)} its {' and '.join(kinds)} no longer "
                         f"{'applies' if len(kinds) == 1 else 'apply'}" for kinds, ss in groups.items())
        print(f"v83: minimum_time_method not written for unit {unit}: {reason}. {lost[0].upper()}{lost[1:]} (results "
              "change). Set minimum_time_method yourself (both, min_uptime or min_downtime) where it should apply.")
    _merge_marker_lists(db, V83_MIN_TIME_MARKER, 83, "writes", plan.writes,
                        [[unit, reason, list(candidates)] for unit, reason, candidates, _n in plan.undecided])


#: The members part 2 of v83 appends to ``startup_methods`` (when missing), in this order, and the one it replaces.
_V83_STARTUP_METHODS = ("linear_fix_in_uc", "integer", "integer_fix_in_uc")
_V83_STARTUP_RENAMES = {"binary": "integer_fix_in_uc"}
_V83_ONLINE_NEED = ("Needs online variables (startup_method linear or binary",
                    "Needs online variables (a startup_method other than no_startup")
_V83_CONTINUOUS = ("With linear startup the number is continuous.",
                   "With startup_method linear or linear_fix_in_uc the number is continuous.")


def _v83_startup_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions that name a ``startup_method`` member, as every database at 82
    has them unless the user edited them, and their v83 texts (part 2 of v83, :func:`_replace_superseded_texts`)."""
    online_coeff = ("[unit of the constraint per unit online] A map of coefficients (Index: constraint name, value: "
                    "coefficient). Puts the number of units {what}online in each time step on the left side of the "
                    "user constraint. One unit is virtual_unitsize; without it, the largest existing capacity, else "
                    "1000 MW (set virtual_unitsize when the capacity is invested). Needs online variables "
                    "(startup_method linear or binary{mt}); a coefficient without them is an error. With linear "
                    "startup the number is continuous. For MW online, use the unit size (MW per unit) as the "
                    "coefficient.")
    startup_input = (
        "[choice] What the input arc supplies. just_input (default): the normal input flow. startup_and_input: the "
        "normal input flow, and each start-up of the unit also takes startup_energy x MW started from this node. "
        "startup_only: only the start-up energy; the arc has no flow and takes no part in the conversion, the "
        "capacity, input shares or ramps. The start-up input can be fuel (oil to start a coal unit), heat (from the "
        "tank of a CSP plant) or electricity. Needs online variables (startup_method linear or binary, or a "
        "minimum_time_method). The start-up input is priced as the node's commodity, counts in CO2 prices and caps "
        "and is dropped with the start-ups by reduce_unit_detail_method online. It is not part of "
        "constraint_flow_coeff, flow groups or the capacity margin.")
    reduce_detail = (
        "[choice] Detail of units and connections that this solve leaves out, to make it faster (typically an "
        "investment solve). none (default): all detail. integers: binary startup_method is solved as continuous and "
        "exact connections behave as regular. integers_ramps: integers, and no ramp limits or ramp costs. "
        "integers_min_limits: integers, and no min_load, capacity_min_coeff floors or minimum up / down times; a "
        "min_load_efficiency unit that is more efficient at min load than at full load (efficiency_at_min_load "
        "above efficiency) then runs at its full-load efficiency. integers_ramps_min_limits: integers, no ramps and "
        "no min limits. Under the integers levels, a unit or connection whose online variables nothing in the solve "
        "still uses (no start-up cost, min_load, capacity_min_coeff floor, minimum up / down time, "
        "min_load_efficiency curve, piecewise_linear curve whose efficiency rises from its first point, "
        "constraint_online_coeff in a user constraint, reserve, inertia or ramp limit left) has none, as with "
        "startup_method no_startup; the run log names them. online: everything of integers_min_limits, and no "
        "online variables: no start-up costs, user constraints that count units online are left out whole, "
        "min_load_efficiency and piecewise_linear units run at their full-load efficiency (the same fuel at full "
        "output; at part load less or more fuel than with their curve), and ramp limits apply to the capacity. "
        "Reserves are then limited by built capacity, not online capacity, and inertia is counted from flows. "
        "online_ramps: online, and no ramp limits or ramp costs. Never adds detail.")
    out = {
        ("unit", "startup_method"): (
            "[choice] Whether the unit has online variables. linear: any share of the capacity can be online. "
            "binary: whole units only (slower to solve). Online status links to the previous time step; that link "
            "stops at a gap in the timeline.",
            "[choice] Whether the unit has online variables. linear: any share of the capacity can be online. "
            "integer: whole units only (a MIP, slower to solve); several units of one entity are counted as whole "
            "units. linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve (a "
            "rolling separate_dispatch solve, or a single solve with a realized: yes branch) the units online over "
            "the jump (the realized periods) are kept at what the stochastic (unit commitment) solve chose; without "
            "_fix_in_uc the realization decides them again with the realized data. Elsewhere, including a rolling "
            "within_solve solve, a member and its _fix_in_uc variant are the same. Online status links to the "
            "previous time step; that link stops at a gap in the timeline."),
        ("connection", "startup_method"): (
            "[choice] Gives the connection online capacity in both directions. Flows and reserves are limited by the "
            "online capacity, and startup_cost is paid per MW brought online. No minimum load or minimum up/down "
            "time. Not allowed for dc_power_flow_with_angles lines.",
            "[choice] Gives the connection online capacity in both directions (linear: any share; integer: whole "
            "units, a MIP; linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve "
            "the online capacity over the jump (the realized periods) is kept at what the stochastic solve chose). "
            "Flows and reserves are limited by the online capacity, and startup_cost is paid per MW brought online. "
            "No minimum load or minimum up/down time. Not allowed for dc_power_flow_with_angles lines."),
        ("unit__inputNode", "startup_input_method"): (startup_input, startup_input.replace(*_V83_ONLINE_NEED)),
        ("solve", "reduce_unit_detail_method"): (reduce_detail, reduce_detail.replace(
            "integers: binary startup_method is solved as continuous",
            "integers: startup_method integer and integer_fix_in_uc are solved as continuous")),
    }
    for cls, what, mt in (("unit", "", ", or a minimum_time_method"), ("connection", "of the connection ", "")):
        old = online_coeff.format(what=what, mt=mt)
        out[(cls, "constraint_online_coeff")] = (old, old.replace(*_V83_ONLINE_NEED).replace(*_V83_CONTINUOUS))
    return out


def _v83_check_no_references(db, list_name: str, removing: dict) -> None:
    """Before members of the value list *list_name* are removed (*removing*: ``{mapped id: parsed value}``), check that
    nothing still refers to them: no default of any definition and no value of a definition on the list holds a
    reference to one, or (a plain string) its value.  At the commit a removed member's id is reused by a member added
    in the same session, so a reference left to it would silently read as that member.  Raises
    :class:`FlexToolMigrationError` (before the step's commit) otherwise."""
    names = set(removing.values())
    left = []
    for pdef in db.get_items("parameter_definition"):
        on_list = pdef["parameter_value_list_name"] == list_name
        if pdef["list_value_id"] in removing or (on_list and pdef["parsed_value"] in names):
            left.append(f"{pdef['entity_class_name']}.{pdef['name']} default")
        if not on_list:
            continue
        for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                           parameter_definition_name=pdef["name"]):
            if pv["list_value_id"] in removing or (isinstance(pv["parsed_value"], str) and pv["parsed_value"] in names):
                left.append(f"{_describe_pv(pv)} ({pdef['name']})")
    if left:
        raise FlexToolMigrationError(
            f"A version 83 upgrade step would remove the member(s) {', '.join(sorted(map(repr, names)))} of the value "
            f"list {list_name} while still in use ({'; '.join(left)}); the upgrade stops so that no default or value "
            "changes with it. The file was restored from the backup. Please report this.")


def _v83_startup_method(db) -> None:
    """Part 2 of v83: ``startup_methods`` becomes no_startup, linear, linear_fix_in_uc, integer, integer_fix_in_uc.

    Appends the missing new members, rewrites every value ``binary`` of every definition on the list (every class
    and alternative; a list reference or a plain string) and a ``binary`` default to ``integer_fix_in_uc`` (the
    same model), then removes the ``binary`` member (after the rewrites, checked by
    :func:`_v83_check_no_references`: at the commit a removed member's id is reused by a member added in the same
    session, so a reference left to it would read as that member), and replaces the six descriptions that name the
    members (:func:`_v83_startup_texts`, by text alone).  ``linear`` is untouched.  Without the list, only the
    descriptions are replaced.  Idempotent; no commit."""
    if db.get_item("parameter_value_list", name="startup_methods"):
        before = _list_members(db, ["startup_methods"])
        planned = {i for i, (_lst, v, t) in before.items() if from_database(v, t) in _V83_STARTUP_RENAMES}
        _append_list_members(db, "startup_methods", _V83_STARTUP_METHODS)
        _v83_check_members(db, "startup_methods", before)  # a changed member would fail the by-value writes first
        rewritten = []
        for pdef in db.get_items("parameter_definition"):
            if pdef["parameter_value_list_name"] != "startup_methods":
                continue
            for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                               parameter_definition_name=pdef["name"]):
                new = _V83_STARTUP_RENAMES.get(pv["parsed_value"]) if isinstance(pv["parsed_value"], str) else None
                if new is not None:
                    rewritten.append(f"{_describe_pv(pv)} ({pdef['name']})")
                    val, typ = to_database(new)
                    db.update_parameter_value(id=pv["id"], value=val, type=typ)
        if rewritten:
            print("v83: the startup_methods member binary is now integer_fix_in_uc (the same model) for:")
            for line in rewritten:
                print("  " + line)
        pending = _detach_list_defaults(db, "startup_methods", _V83_STARTUP_RENAMES)
        removing = {lv["id"]: lv["parsed_value"]
                    for lv in db.get_items("list_value", parameter_value_list_name="startup_methods")
                    if getattr(lv["id"], "db_id", lv["id"]) in planned}
        _v83_check_no_references(db, "startup_methods", removing)
        for lv_id in removing:
            db.remove_item("list_value", lv_id)
        _attach_list_defaults(db, pending)
        _v83_check_members(db, "startup_methods", before, removed=planned)
    _replace_text_pairs(db, _v83_startup_texts())


#: Part 5 of v83: the member appended to ``dispatch_realization_methods`` (the new shipped default) and the method
#: written on the stochastic solves that relied on the default.
_V83_DETERMINISTIC = "deterministic"
_V83_WITHIN_SOLVE = "within_solve"


def _v83_dispatch_realization_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the ``dispatch_realization_method`` description as every database at 82 has
    it unless the user edited it, and its v83 text (part 5 of v83, :func:`_replace_superseded_texts`)."""
    return {("solve", "dispatch_realization_method"): (
        "[choice] How the committed part of a stochastic solve is realized. within_solve (default): a realized: yes "
        "branch gets a realization solve automatically. separate_dispatch: always runs a separate realized dispatch "
        "solve; needs a realized: yes branch (error otherwise).",
        "[choice] How a solve with stochastic_branches is solved. deterministic (default): the forecasts are ignored "
        "and the solve uses the data of the realized: yes branch, with a warning (an error without that branch). "
        "within_solve: the forecasts are solved together; a rolling solve realizes each jump with the realized: yes "
        "branch and looks ahead with the forecasts made at the end of the jump; a single solve with a realized: yes "
        "branch gets a separate realization solve. separate_dispatch: a first, stochastic solve decides the units "
        "online of _fix_in_uc units; a realization solve keeps them and dispatches the realized: yes branch over "
        "the jump and, in a rolling solve, looks ahead with the forecasts made at the end of the jump (a single "
        "solve realizes its realized periods only); needs a realized: yes branch. Without stochastic_branches, deterministic and within_solve give the same model.")}


def plan_dispatch_realization_method(branches, methods, scenarios) -> tuple[tuple, tuple]:
    """Where part 5 of v83 writes ``dispatch_realization_method`` ``within_solve`` so that every stochastic solve
    keeps running within_solve once an absent method means ``deterministic`` (plan §2a step 2).  Pure: no database.

    *branches* ``{(solve, alternative): the stochastic_branches value is non-empty}``; *methods* ``{(solve,
    alternative): method, None for a null row}``; *scenarios* ``{scenario: [alternative, ...]}`` lowest rank first.
    A scenario is a candidate for a solve when its winning ``stochastic_branches`` is non-empty and no method
    resolves (a winning null row reads as no method, :func:`_v83_read`); the target is the alternative of that
    winning value, or the winning null method row when that one ranks higher (it is overwritten).  With the writes
    in place every scenario where a method resolved must read the same method and every candidate must read
    within_solve, else the solve gets no write at all (the reason names every scenario that fails).  Returns
    ``(writes, undecided)``: ``((solve, alternative), ...)`` and ``((solve, reason, candidate scenarios), ...)``."""
    writes, undecided = [], []
    for solve in sorted({key[0] for key in branches}):
        before = {s: _v83_read(alts, methods, solve) for s, alts in sorted(scenarios.items())}
        candidates, targets = [], set()
        for s, alts in sorted(scenarios.items()):
            where = _v77_winner(alts, branches, solve)
            if where is not None and branches[(solve, where)] and before[s][0] is None:
                candidates.append(s)
                null_row = before[s][1]  # a winning null method row (None without one)
                targets.add(where if null_row is None or alts.index(null_row) < alts.index(where) else null_row)
        if not candidates:
            continue
        table = {**methods, **{(solve, a): _V83_WITHIN_SOLVE for a in targets}}
        reasons = []
        for s, (own, where) in before.items():
            new, winner = _v83_read(scenarios[s], table, solve)
            if own is not None and new != own:
                reasons.append(f"a value in alternative {winner} would replace your dispatch_realization_method in "
                               f"alternative {where} in scenario {s}")
            elif s in candidates and new != _V83_WITHIN_SOLVE:
                reasons.append(f"the within_solve written for scenario {s} would not win there")
        if not reasons:
            writes.extend((solve, a) for a in sorted(targets))
        else:
            undecided.append((solve, "; ".join(reasons), tuple(candidates)))
    return tuple(writes), tuple(undecided)


def _v83_solve_inputs(db) -> tuple:
    """:func:`plan_dispatch_realization_method` arguments read from *db* (parsed values only; a null method row
    is kept: it reads as no method where it wins, :func:`_v83_read`; an empty map has no branch)."""
    branches, methods = {}, {}
    for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="stochastic_branches"):
        value = pv["parsed_value"]
        branches[(pv["entity_byname"][0], pv["alternative_name"])] = value is not None and not (
            isinstance(value, Map) and len(value) == 0)
    for pv in db.find_parameter_values(entity_class_name="solve",
                                       parameter_definition_name="dispatch_realization_method"):
        methods[(pv["entity_byname"][0], pv["alternative_name"])] = pv["parsed_value"]
    return branches, methods, _scenario_alternatives(db)


def _v83_dispatch_realization_method(db) -> None:
    """Part 5 of v83: ``dispatch_realization_methods`` gets ``deterministic``, the new shipped default.

    Appends the member, writes ``within_solve`` where a stochastic solve relied on the default (the engine reads no
    solve definition default: an absent method ran within_solve; :func:`plan_dispatch_realization_method`, results
    unchanged), sets a ``within_solve`` definition default to ``deterministic`` (a ``NULL`` or another member stays)
    and replaces the description by text alone.  Skips the member, writes and default without the class, the
    definition or the list.  Idempotent; no commit, no marker."""
    name = "dispatch_realization_methods"
    if (db.get_item("entity_class", name="solve") and db.get_item("parameter_value_list", name=name)
            and _pdef_or_none(db, "solve", "dispatch_realization_method") is not None):
        before = _list_members(db, [name])
        # Not from *before*: _list_members leaves out the members not yet committed (a rerun in the session).
        rerun = any(x["parsed_value"] == _V83_DETERMINISTIC
                    for x in db.get_items("list_value", parameter_value_list_name=name))
        _append_list_members(db, name, (_V83_DETERMINISTIC,))
        _v83_check_members(db, name, before)  # a changed member would fail the by-value writes first
        writes, undecided = plan_dispatch_realization_method(*_v83_solve_inputs(db))
        _v83_write_values(db, "solve", "dispatch_realization_method",
                          [(solve, alternative, _V83_WITHIN_SOLVE) for solve, alternative in writes])
        if rerun:
            _already_applied(83, "dispatch_realization_method: already written")
        if writes:
            print("v83: dispatch_realization_method within_solve written on the stochastic solves that relied on "
                  "the default (results unchanged):")
            for solve, alternative in writes:
                print(f"  solve {solve} [{alternative}]")
        for solve, reason, candidates in undecided:
            print(f"v83: dispatch_realization_method not written for solve {solve}: {reason}. In "
                  f"{_v83_scenarios_text(candidates)} it now runs deterministic: the forecasts are ignored (results "
                  "change), or the run stops when there is no realized: yes branch. Set dispatch_realization_method "
                  "within_solve yourself where it should apply.")
        pdef = _pdef_or_none(db, "solve", "dispatch_realization_method")
        if pdef["parsed_value"] == _V83_WITHIN_SOLVE:
            val, typ = to_database(_V83_DETERMINISTIC)
            db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
    _replace_text_pairs(db, _v83_dispatch_realization_texts())


#: Part 3 of v83: the behaviour-change markers of the realization changes (``{"since": 83, "solves": [...]}``).
V83_WITHIN_SOLVE_MARKER = "rolling_within_solve_one_solve"
V83_SEPARATE_DISPATCH_MARKER = "separate_dispatch_realization_looks_ahead"


def plan_realization_notices(branches, modes, methods, scenarios) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The solves part 3 of v83 lists in the markers :data:`V83_WITHIN_SOLVE_MARKER` and
    :data:`V83_SEPARATE_DISPATCH_MARKER` (plan §2a "Notices").  Pure: no database.

    *branches* ``{(solve, alternative): the stochastic_branches value is non-empty}``, *modes* and *methods*
    ``{(solve, alternative): parsed solve_mode / dispatch_realization_method, None for a null row}``, *scenarios*
    ``{scenario: [alternative, ...]}`` lowest rank first.  In each scenario a solve is *stochastic* when its winning
    ``stochastic_branches`` is non-empty (one forecast counts), *rolling* when its winning ``solve_mode`` is
    ``rolling_window``, and its method is the winning explicit ``dispatch_realization_method`` (part 5's writes
    included; no definition default is read: the engine reads none).  Listed for within_solve: rolling, stochastic
    and ``within_solve`` in some scenario; for separate_dispatch: stochastic and ``separate_dispatch`` in some
    scenario, rolling or single.  A solve that is not stochastic in the scenario where its method resolves is not
    listed for it.  Returns ``(within_solve solves, separate_dispatch solves)``, sorted."""
    within, separate = set(), set()
    for solve in sorted({key[0] for key in branches}):
        for alts in scenarios.values():
            where = _v77_winner(alts, branches, solve)
            if where is None or not branches[(solve, where)]:
                continue
            method = _v83_read(alts, methods, solve)[0]
            if method == "separate_dispatch":
                separate.add(solve)
            elif method == _V83_WITHIN_SOLVE and _v83_read(alts, modes, solve)[0] == "rolling_window":
                within.add(solve)
    return tuple(sorted(within)), tuple(sorted(separate))


def _v83_realization_notices(db) -> None:
    """Part 3 of v83: rolling ``within_solve`` solves now realize each jump in one solve, and ``separate_dispatch``
    realizations look ahead after the jump (their first solve shares only the units online of ``_fix_in_uc``
    units).  Lists the solves whose results change (:func:`plan_realization_notices`) in the markers
    ``rolling_within_solve_one_solve`` and ``separate_dispatch_realization_looks_ahead`` (``{"since": 83,
    "solves": [...]}``, merged with a stored list).  Runs after part 5, whose writes count.  Metadata only;
    idempotent; no commit."""
    if not db.get_item("entity_class", name="solve"):
        return
    branches, methods, scenarios = _v83_solve_inputs(db)
    modes = {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
             for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="solve_mode")}
    for change_id, solves in zip((V83_WITHIN_SOLVE_MARKER, V83_SEPARATE_DISPATCH_MARKER),
                                 plan_realization_notices(branches, modes, methods, scenarios)):
        name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
        stored: list = []
        for md in db.get_items("metadata"):
            if md["name"] == name:
                try:
                    stored = list(json.loads(md["value"]).get("solves") or [])
                except (TypeError, ValueError, AttributeError):
                    stored = []
        merged = stored + [s for s in solves if s not in stored]
        if merged:
            _v77_set_marker(db, change_id, json.dumps({"since": 83, "solves": merged}))


# --- Version 83: the batch ---------------------------------------------------------------------------------------

#: The members v83 adds to each value list (parts 1, 2 and 5): what a database at 83 has on the lists it has.
_V83_NEW_MEMBERS: dict[str, tuple[str, ...]] = {"minimum_time_methods": _V83_MIN_TIME_METHODS,
                                                "startup_methods": _V83_STARTUP_METHODS,
                                                "dispatch_realization_methods": (_V83_DETERMINISTIC,)}


#: Part 4 of v83 (item 278): the descriptions whose shipped text names an old member or an old set of value shapes,
#: ``{(class, name): (old, new)}``.  The old texts are what the v25 template has at 82; the new ones name the current
#: members and the shapes each parameter takes.
_V83_SHAPE_TEXTS: dict[tuple[str, str], tuple[str, str]] = {
    ("connection", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for connection flows. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for connection flows. Constant, period, time or "
        "period+time."),
    ("connection", "other_operational_cost"): (
        "[CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, Period or time.",
        "[CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, period, time or "
        "period+time."),
    ("group", "co2_method"): (
        "[choice] Choice of the CO2 method: no_method, price, period, total, price_period, price_total, period_total, "
        "price_period_total",
        "[choice] Choice of the CO2 method: none, price, period, total, price_period, price_total, period_total, "
        "price_period_total"),
    ("group", "co2_price"): (
        "[CUR/tCO2] CO2 price for a group of nodes. Constant, period or time.",
        "[CUR/tCO2] CO2 price for a group of nodes. Constant, period, time or period+time."),
    ("node", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for storage. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for storage. Constant, period, time, period+time or "
        "stochastic time."),
    ("node", "penalty_up"): (
        "[CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, Period or Time.",
        "[CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, period, time, "
        "period+time, or stochastic time."),
    ("node", "penalty_down"): (
        "[CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, Period or Time.",
        "[CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, period, time, "
        "period+time, or stochastic time."),
    ("node", "storage_state_reference_value"): (
        "[-] 0-1, share of the storage capacity. Relative state of storage at then end of each solve (can be "
        "overwritten in the next solve).  Requires 'use_reference_value' in 'storage_solve_horizon_method'. Constant.",
        "[-] 0-1, share of the storage capacity. Relative state of storage at then end of each solve (can be "
        "overwritten in the next solve).  Requires 'use_reference_value' in 'storage_solve_horizon_method'. Constant, "
        "period, time or period+time."),
    ("reserve__upDown__group", "reservation"): (
        "[MW] Amount of reserve to be reserved. Constant or time.",
        "[MW] Amount of reserve to be reserved. Constant, period, time, period+time or stochastic time."),
    ("unit", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for flows from/to the unit. For online units, the "
        "online variable is multiplied by the availability. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for flows from/to the unit. For online units, the "
        "online variable is multiplied by the availability. Constant, period, time, period+time or stochastic time."),
    **{(cls, "other_operational_cost"): (
        "[CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.",
        "[CUR/MWh] Other operational variable cost for energy flows. Constant, period, time, period+time or "
        "stochastic time.") for cls in ("unit__inputNode", "unit__outputNode")},
}


#: Part 4 of v83 (item 278, D26): the descriptions that a database made from a release's shipped database or template
#: (FlexTool 3.22.0-3.43.0) still holds at 82, because the step that rewrote the text ran before the release's data
#: version: ``{(class, name): ((old texts...), new)}``.  The old texts are read from the shipped databases of those tags
#: migrated to 82 (``""`` is a description that was never written); the new ones are the shipped schema's.  Kept apart
#: from :data:`_V83_SHAPE_TEXTS`, whose single old text per definition is what the v25 template holds at 82.
_V83_RELEASE_TEXTS: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {
    ("unit", "minimum_time_method"): (
        ("[choice] Not functional yet. Choice between minimum up- and downtimes (<empty>, min_downtime, min_uptime, "
         "both).",),
        "[choice] Choice between minimum up- and downtimes (none, min_downtime, min_uptime, both). Setting this to "
        "anything other than 'none' will activate online variables (at least linear) for the unit. Default: none (no "
        "minimum time constraints)."),
    ("node", "penalty_up"): (
        ("[CUR/MW] Penalty cost for decreasing consumption in the node. Constant, Period or Time.",),
        _V83_SHAPE_TEXTS[("node", "penalty_up")][1]),
    ("node", "penalty_down"): (
        ("[CUR/MW] Penalty cost for increasing consumption in the node. Constant, Period or Time.",),
        _V83_SHAPE_TEXTS[("node", "penalty_down")][1]),
    ("group", "penalty_inertia"): (
        ("[CUR/MWs per hour] Penalty for violating the inertia constraint. Constant or period.",),
        "[CUR/MWs per hour] Penalty for violating the inertia constraint. Cost scales with the duration of the "
        "violation. Constant or period."),
    ("connection", "reactance"): (
        ("",),
        "[p.u.] Per-unit reactance of the transmission line. Used for DC power flow when the connection is within a "
        "nodeGroup that has transfer_method set to dc_power_flow_with_angles. The susceptance used in the flow "
        "equation is base_MVA / reactance."),
    ("connection", "transfer_method"): (
        ("[choice] Choice of transfer method",),
        "[choice] Choice of transfer method. Options: regular (default), no_losses_no_variable_cost, "
        "variable_cost_only, exact, unidirectional. 'unidirectional' restricts flow to source → sink only (single "
        "non-negative variable); 'regular'/'exact'/'variable_cost_only' are bidirectional two-variable variants; "
        "'no_losses_no_variable_cost' is a single signed variable with no losses and no VOM."),
    ("group", "transfer_method"): (
        ("[choice] Override transfer_method for all connections within this nodeGroup. Options: "
         "use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, "
         "variable_cost_only, dc_power_flow_with_angles. When set to dc_power_flow_with_angles, connections between "
         "member nodes use B-theta DC power flow (requires reactance parameter on connections).",),
        "[choice] Override transfer_method for all connections within this nodeGroup. Options: "
        "use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, "
        "variable_cost_only, unidirectional, dc_power_flow_with_angles. Setting 'unidirectional' gates every member "
        "connection to one-way flow (source → sink only). Setting 'dc_power_flow_with_angles' uses B-theta DC power "
        "flow (requires reactance on connections)."),
    ("commodity", "price_ladder_cumulative"): (
        ("[CUR/MWh; MWh]",
         "[CUR/MWh; MWh] Stepped supply curve for price_method='price_ladder_cumulative'.  Structure: Map(tier -> "
         "{price, quantity}).  1-based integer tier index.  quantity=inf marks an unbounded tail tier.  "
         "Period-agnostic — the cap is a single total across the full model horizon."),
        "[CUR/MWh; MWh] Stepped supply curve for price_method='price_ladder_cumulative'.  2d map with rows "
        "'tier,price,quantity' — one row per tier, giving the tier's unit price and its cumulative quantity cap.  "
        "1-based integer tier index.  quantity=inf marks an unbounded tail tier.  Period-agnostic — the cap is a "
        "single total across the full model horizon."),
}


def _replace_release_texts(db) -> None:
    """The current description in place of each old shipped one of :data:`_V83_RELEASE_TEXTS` (exactly equal; a
    description the user edited is kept).  Idempotent."""
    for key, (olds, new) in _V83_RELEASE_TEXTS.items():
        pdef = _pdef_or_none(db, *key)
        if pdef is not None and (pdef["description"] or "") in olds:
            db.update_parameter_definition(id=pdef["id"], description=new)


def _v83_description_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions part 4 of v83 replaces by text alone
    (:func:`_replace_superseded_texts`): the twelve of :data:`_V83_SHAPE_TEXTS` and ``virtual_unitsize`` of units
    and connections, whose v77 text (:func:`_unit_count_vu_text`) names the removed ``binary`` startups."""
    out = dict(_V83_SHAPE_TEXTS)
    for cls in ("unit", "connection"):
        old = _unit_count_vu_text(cls)
        out[(cls, "virtual_unitsize")] = (old, old.replace(
            "Binary startups commit", "Integer startups (startup_method integer or integer_fix_in_uc) commit"))
    return out


def _v83_descriptions(db) -> None:
    """Part 4 of v83 (item 278): the current description in place of each old shipped one of
    :func:`_v83_description_texts` and of :data:`_V83_RELEASE_TEXTS` (by text alone, whatever the default; a
    description the user edited is kept), and the value list ``solver`` deleted when no definition is on it (``solve.solver`` is on ``solvers`` since v52; an
    older template can still hold the unused list).  Descriptions and an unused list only.  Idempotent."""
    _replace_text_pairs(db, _v83_description_texts())
    _replace_release_texts(db)
    solver = db.get_item("parameter_value_list", name="solver")
    if solver and not any(pd["parameter_value_list_name"] == "solver" for pd in db.get_items("parameter_definition")):
        db.remove_item("parameter_value_list", solver["id"])


def _v83_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions the parts of v83 replace by text alone."""
    return {**_v83_startup_texts(), **_v83_dispatch_realization_texts(), **_v83_description_texts()}


def _v83_carry_over(db) -> None:
    """Part 0 of v83: a database migrated to 82 by a development build of v82 gets the v82 definitions it lacks
    (an existing definition keeps its fields) and the current description in place of a superseded shipped one
    (:func:`_v82_superseded_texts`, by text alone).  After a full v82 it changes nothing.  Idempotent."""
    _v82_definitions_part(db, update=False)
    _replace_text_pairs(db, _v82_superseded_texts())


def _migrate_v83_schema_batch(db) -> None:
    """All schema changes of database version 83 (v82 -> v83), in one step.  The parts are numbered by name and
    run in the order below; each is idempotent, and the step commits once at the end.

    0. The v82 carry-over (:func:`_v83_carry_over`).
    1. ``minimum_time_method`` now decides whether a unit's min up/down times apply: the method is written where a
       unit relied on a time alone (results unchanged), with the marker ``minimum_time_method_written``
       (:func:`_v83_minimum_time_method`).
    2. ``startup_methods`` gets ``linear_fix_in_uc``, ``integer`` and ``integer_fix_in_uc``; every ``binary`` becomes
       ``integer_fix_in_uc`` (the same model) and the ``binary`` member is removed (:func:`_v83_startup_method`).
    5. ``dispatch_realization_methods`` gets ``deterministic``, the new shipped default; ``within_solve`` is written
       on the stochastic solves that relied on the default (results unchanged;
       :func:`_v83_dispatch_realization_method`).
    3. Rolling ``within_solve`` solves realize each jump in one solve and ``separate_dispatch`` realizations look
       ahead: the solves whose results change are listed in two behaviour-change markers (after part 5, whose
       writes count; :func:`_v83_realization_notices`).
    4. Item 278: the current description in place of fourteen old shipped ones (value shapes, ``no_method``,
       ``binary`` startups) and of the older texts that databases made from a release still hold
       (:data:`_V83_RELEASE_TEXTS`), and the unused value list ``solver`` deleted (:func:`_v83_descriptions`).

    On a non-local URL the list defaults are not repaired before the step (item 279): a default still stored as its
    member's text reads as nothing, and a description written on its definition stores it as ``NULL``; it read as
    nothing before too.
    """
    _v83_carry_over(db)
    _v83_minimum_time_method(db)
    _v83_startup_method(db)
    _v83_dispatch_realization_method(db)
    _v83_realization_notices(db)
    _v83_descriptions(db)
    _commit_step(db, "v83: minimum_time_method decides the min up/down times, startup_method binary is "
                     "integer_fix_in_uc, dispatch_realization_method deterministic is the default, the "
                     "realization of stochastic solves looks ahead, descriptions updated")
