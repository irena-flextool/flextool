"""Notes on input parameters that are easy to misread (P0 §4.3 "Notes").

:func:`traps` maps ``(entity class, parameter)`` to its notes.  Most are written here; two families are generated
so they cannot drift: every parameter entered per kW or kWh (its description bracket starts ``[CUR/kW``) gets the
×1000 note, and every run option that a database parameter sets (``run_options.OPTIONS`` with ``db``) gets the note
that a run flag or an environment variable overrides it.  A note that depends on another parameter of the same
entity (``invest_method``, ``penalty_method``, ``inflow_method``, ``virtual_unitsize``) adds that entity's value
(:func:`notes_for`).
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flextool.assistant.inputs._schema import bracket, shipped_schema


@dataclass(frozen=True)
class Trap:
    text: str
    control: str | None = None       # the entity's parameter whose value the note adds


_INVEST_CLASSES = ("unit", "connection", "node", "group")


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
            f"({capacity}).", control="virtual_unitsize"))
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
        control="inflow_method"))
    # (f) investment parameters need an invest_method that allows investment
    for (cls, param), definition in schema.parameters.items():
        if cls in _INVEST_CLASSES and param.startswith("invest_") and param != "invest_method":
            default = schema.parameters.get((cls, "invest_method"))
            default_text = f" (default {default.default})" if default is not None and default.default else ""
            add((cls, param), Trap(
                f"Used only when the {cls}'s invest_method allows investment{default_text}.",
                control="invest_method"))
    # (g) penalties unused with penalty_method off
    for param in ("penalty_up", "penalty_down"):
        add(("node", param), Trap(
            "Not used when the node's penalty_method is off (default regular): the node balance then has no "
            "slack, and an imbalance makes the solve infeasible.", control="penalty_method"))
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
            if current is None:
                definition = shipped_schema().parameters.get((entity_class, trap.control))
                default = definition.default if definition is not None else None
                text += (f" Here {trap.control} is not set" + (f" (default {default})." if default is not None
                                                                else "."))
            elif isinstance(current, (str, int, float)):
                text += f" Here {trap.control} is {current}."
        out.append(text)
    return out
