"""Notes on input parameters that are easy to misread (P0 §4.3 "Notes").

:func:`traps` maps ``(entity class, parameter)`` to its notes.  Most are written here; two families are generated
so they cannot drift: every parameter entered per kW or kWh (its description bracket starts ``[CUR/kW``) gets the
×1000 note, and every run option that a database parameter sets (``run_options.OPTIONS`` with ``db``) gets the note
that a run flag or an environment variable overrides it.  A note that depends on another parameter of the same
entity (``invest_method``, ``penalty_method``, ``inflow_method``, ``virtual_unitsize``) states its conclusion for
that entity's value (or the default): "Here invest_method is invest_total, so invest_max_period is not used"
(:func:`notes_for`).  Small models read a general rule plus "Here invest_method is invest_total" backwards
(failure analysis 2026-11-12 §3.3), so the general rule is given only when the value cannot be resolved.

Which invest_method values use which limit is the engine's own sets (:func:`methods`), not a copy.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flextool.assistant.inputs._schema import bracket, shipped_schema


@dataclass(frozen=True)
class Trap:
    """A note on a parameter.  *control*: the entity's parameter the note depends on; *conclude* gives the note
    for the entity's value of it (``(value, is_default)``), or ``None`` when that value has no conclusion here (a
    map, an unknown value): then the general *text* is given with the value."""

    text: str
    control: str | None = None
    conclude: Callable[[Any, bool], str | None] | None = None


_INVEST_CLASSES = ("unit", "connection", "node", "group")


@dataclass(frozen=True)
class InvestMethods:
    """The invest_method values that allow each decision and use each limit, as the engine builds them
    (engine/preprocess/params/_existing.py).  ``known``: every value of a unit, connection, node or group;
    ``group_no_invest``: the group values under which the group limits no investment (``not_active`` and the
    retire-only values)."""

    known: frozenset[str]
    no_invest: frozenset[str]
    retire_only: frozenset[str]
    invest_total: frozenset[str]
    invest_period: frozenset[str]
    retire_total: frozenset[str]
    retire_period: frozenset[str]
    cumulative: frozenset[str]
    group_no_invest: frozenset[str]


@functools.cache
def methods() -> InvestMethods:
    """The engine's invest_method sets (imported when first asked: the engine is not loaded with the tools)."""
    from flextool.engine.preprocess.emit._mid_sets import _CUMULATIVE_METHODS
    from flextool.engine.preprocess.params import _existing as e

    return InvestMethods(
        known=e.INVEST_METHODS_ALL | e.GROUP_INVEST_NOT_ACTIVE,
        no_invest=e.INVEST_NOT_ALLOWED,
        retire_only=e.INVEST_NOT_ALLOWED - {"not_allowed"},
        invest_total=e._INVEST_TOTAL_METHODS,
        invest_period=e._INVEST_PERIOD_METHODS,
        retire_total=e._DIVEST_TOTAL_METHODS,
        retire_period=e._DIVEST_PERIOD_METHODS,
        cumulative=_CUMULATIVE_METHODS,
        group_no_invest=e.GROUP_INVEST_NOT_ACTIVE,
    )


#: The limits each invest_method family uses, by parameter: the field of :class:`InvestMethods` whose values use it.
LIMIT_USERS: dict[str, str] = {
    "invest_max_period": "invest_period", "invest_min_period": "invest_period",
    "invest_max_total": "invest_total", "invest_min_total": "invest_total",
    "retire_max_period": "retire_period", "retire_min_period": "retire_period",
    "retire_max_total": "retire_total", "retire_min_total": "retire_total",
    "cumulative_max_capacity": "cumulative", "cumulative_min_capacity": "cumulative",
}


def users_of(parameter: str) -> frozenset[str] | None:
    """The invest_method values that use the limit *parameter* (``None``: not a limit)."""
    field = LIMIT_USERS.get(parameter)
    return getattr(methods(), field) if field else None


def listed(values: frozenset[str]) -> str:
    """*values* in the order of the schema's invest_method list, comma-separated."""
    order = shipped_schema().value_lists.get("group_invest_methods", ())
    return ", ".join(sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v)))


def _here(control: str, value: Any, is_default: bool) -> str:
    return f"Here {control} is {value}" + (" (default)" if is_default else "")


def _invest_conclusion(cls: str, param: str) -> Callable[[Any, bool], str | None]:
    """The conclusion of an investment parameter of *cls* for the entity's invest_method."""
    def conclude(method: Any, is_default: bool) -> str | None:
        m = methods()
        allowed = m.group_no_invest if cls == "group" else m.no_invest
        if not isinstance(method, str) or method not in m.known:
            return None
        here = _here("invest_method", method, is_default)
        if method in allowed:
            why = ("the group sets no investment limits" if method == "not_active"
                   else f"the {cls} cannot invest" if method == "not_allowed"
                   else "the group limits retirement only" if cls == "group"
                   else "it allows retirement only")
            return f"{here}, so {why} and {param} is not used."
        users = users_of(param)
        if users is None:                          # invest_cost, invest_forced: every method that invests
            return f"{here}, so the {cls} can invest and {param} is used."
        if method in users:
            return f"{here}, so {param} is used."
        if method.endswith("_no_limit"):
            return f"{here}, so investment has no limits and {param} is not used."
        if method in m.cumulative:
            return (f"{here}, so the cumulative capacity limits apply instead and {param} is not used (only "
                    f"{listed(users)} use it).")
        return f"{here}, so {param} is not used (only {listed(users)} use it)."
    return conclude


def _invest_rule(cls: str, param: str, default: Any) -> str:
    default_text = f" (default {default}" + (": the group sets no limits)" if default == "not_active" else ")") \
        if default else ""
    users = users_of(param)
    if users is None:
        return f"Used only when the {cls}'s invest_method allows investment{default_text}."
    return f"Used only when the {cls}'s invest_method is one of {listed(users)}{default_text}."


def _existing_conclusion(capacity: str) -> Callable[[Any, bool], str | None]:
    def conclude(size: Any, is_default: bool) -> str | None:
        if size is None:
            return f"Here virtual_unitsize is not set, so existing is the capacity itself ({capacity})."
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            return None
        if size > 0:
            return (f"Here virtual_unitsize is {size:g}, so existing is a number of units: the capacity is "
                    f"existing × {size:g} {capacity}.")
        return f"Here virtual_unitsize is {size:g} (0 or below counts as not set), so existing is the capacity " \
               f"itself ({capacity})."
    return conclude


#: What FlexTool does with a node's stored inflow under each inflow_method.
INFLOW_METHODS: dict[str, str] = {
    "use_original": "FlexTool uses the stored inflow as it is",
    "no_inflow": "FlexTool ignores the stored inflow: the node has no inflow",
    "scale_to_annual_flow": ("FlexTool multiplies the stored inflow by one factor so that its annual energy equals "
                             "annual_flow: the stored values give the shape"),
    "scale_in_proportion": ("FlexTool multiplies the stored inflow by annual_flow divided by the stored inflow's "
                            "annualized sum: the stored values give the shape"),
    "scale_to_annual_and_peak_flow": ("FlexTool transforms the stored inflow so that its annual energy equals "
                                      "annual_flow and its highest step reaches peak_inflow: the stored values "
                                      "give the shape"),
}


def _inflow_conclusion(method: Any, is_default: bool) -> str | None:
    if not isinstance(method, str) or method not in INFLOW_METHODS:
        return None
    return f"{_here('inflow_method', method, is_default)}, so {INFLOW_METHODS[method]}."


#: What penalty_up and penalty_down price (the node's slack under penalty_method regular).
PENALTY_PRICES = {"penalty_up": "energy not served", "penalty_down": "excess energy"}


def _penalty_conclusion(param: str) -> Callable[[Any, bool], str | None]:
    def conclude(method: Any, is_default: bool) -> str | None:
        here = _here("penalty_method", method, is_default)
        if method == "regular":
            return f"{here}, so {param} is used: it is the cost per MWh of the node's {PENALTY_PRICES[param]}."
        if method == "off":
            return (f"{here}, so {param} is not used: the node balance has no slack, and an imbalance makes the "
                    f"solve infeasible.")
        return None
    return conclude


def option_trap_text(flag: str | None, env: str | None) -> str:
    """The note of a run option that a database parameter also sets."""
    if flag and env:
        return f"A run flag ({flag}) or the environment variable {env} overrides this value."
    if env:
        return f"The environment variable {env} overrides this value."
    return f"A run flag ({flag}) overrides this value."


@functools.cache
def traps() -> dict[tuple[str, str], tuple[Trap, ...]]:
    from flextool.run_options import OPTIONS

    schema = shipped_schema()
    out: dict[tuple[str, str], list[Trap]] = {}

    def add(key: tuple[str, str], trap: Trap) -> None:
        if key in schema.parameters:
            out.setdefault(key, []).append(trap)

    # (a) existing is a count only with virtual_unitsize above 0
    for cls, capacity in (("unit", "MW"), ("connection", "MW"), ("node", "MWh")):
        add((cls, "existing"), Trap(
            f"existing is a number of units only when virtual_unitsize is above 0; the capacity is then existing "
            f"× virtual_unitsize. With virtual_unitsize unset, 0 or below, existing is the capacity itself "
            f"({capacity}).", control="virtual_unitsize", conclude=_existing_conclusion(capacity)))
        add((cls, "virtual_unitsize"), Trap(
            "A value of 0 or below counts as not set: existing is then the capacity itself, not a number of units."))
    # (b) a connection's capacity limits the delivered flow
    for param in ("existing", "efficiency"):
        add(("connection", param), Trap(
            "A connection's capacity limits the flow delivered into the receiving node. With losses (efficiency "
            "below 1) the sending node gives more than that: delivered flow / efficiency."))
    # (c) run options that a database parameter sets
    for spec in OPTIONS.values():
        if spec.db:
            cls, _, param = spec.db.partition(".")
            add((cls, param), Trap(option_trap_text(spec.flag, spec.env)))
    # (d) values entered per kW or kWh
    for (cls, param), definition in schema.parameters.items():
        mark = bracket(definition.description) or ""
        if mark.startswith("[CUR/kW"):
            add((cls, param), Trap(
                "Entered per kW (or kWh): FlexTool multiplies the value by 1000 inside the model, to a value per MW "
                "(or MWh). With other units it is per thousandth of the flow unit."))
    # (e) inflow and inflow_method
    add(("node", "inflow"), Trap(
        "How FlexTool uses inflow depends on the node's inflow_method (default use_original: as stored). The "
        "scaling methods rescale the stored values when the model runs, and no_inflow ignores them.",
        control="inflow_method", conclude=_inflow_conclusion))
    # (f) investment parameters need an invest_method that allows investment
    for (cls, param), definition in schema.parameters.items():
        if cls in _INVEST_CLASSES and param.startswith("invest_") and param != "invest_method":
            default = schema.parameters.get((cls, "invest_method"))
            add((cls, param), Trap(_invest_rule(cls, param, default.default if default is not None else None),
                                   control="invest_method", conclude=_invest_conclusion(cls, param)))
    # (g) penalties: used under penalty_method regular, the slack removed under off
    for param in ("penalty_up", "penalty_down"):
        add(("node", param), Trap(
            f"The cost per MWh of the node's {PENALTY_PRICES[param]} under the node's penalty_method regular "
            "(the default). penalty_method off removes that slack: the node balance is then a hard equality.",
            control="penalty_method", conclude=_penalty_conclusion(param)))
    # (h) storage_state_start is a share
    add(("node", "storage_state_start"), Trap(
        "A share of the storage capacity (0 to 1), not an amount of energy."))
    return {k: tuple(v) for k, v in out.items()}


def notes_for(entity_class: str, parameter: str, *, value_of: Callable[[str], Any] | None = None) -> list[str]:
    """The notes of a parameter; with *value_of* (the entity's parsed values) each note that depends on another
    parameter adds the entity's value of it."""
    out: list[str] = []
    for trap in traps().get((entity_class, parameter), ()):
        text = trap.text
        if trap.control and value_of is not None:
            current = value_of(trap.control)
            is_default = current is None
            if is_default:
                definition = shipped_schema().parameters.get((entity_class, trap.control))
                current = definition.default if definition is not None else None
            concluded = trap.conclude(current, is_default) if trap.conclude is not None else None
            if concluded is not None:
                text = concluded
            elif current is None:
                text += f" Here {trap.control} is not set."
            elif isinstance(current, (str, int, float)):
                text += f" {_here(trap.control, current, is_default)}."
        out.append(text)
    return out
