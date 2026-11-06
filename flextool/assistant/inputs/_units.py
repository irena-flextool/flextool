"""The unit of an input parameter, from the shipped schema's description bracket (P0 §4.3).

``[-]`` is a ratio or share (shown as dimensionless); ``[MW or MWh]`` of a group depends on its members (units and
connections: MW, storage nodes: MWh, ``resolve_group_capacity_type``); ``[MW or count]`` / ``[MWh or count]`` of
``existing`` depends on the entity's ``virtual_unitsize`` (above 0: a number of units).  Markers that are not units
(``[choice]``, ``[yes/no]``, ``[name]``, ``[text]``, ``[version number]``) give no unit.  The tools word the unit
with :func:`flextool.assistant.unit_wording.shown_unit`.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from flextool.assistant.inputs._schema import bracket, shipped_schema
from flextool.assistant.unit_wording import DIMENSIONLESS

NOT_UNITS = frozenset({"[choice]", "[yes/no]", "[name]", "[text]", "[version number]"})
_COUNT_FORMS = {"[MW or count]": "MW", "[MWh or count]": "MWh"}
_GROUP_FORM = "[MW or MWh]"
_PER_RE = re.compile(r"^\s*(per [^.;,]+)")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parameter_unit(entity_class: str, parameter: str, *, value_of: Callable[[str], Any] | None = None,
                   members: Iterable[str] | None = None) -> tuple[str | None, str | None]:
    """``(unit, unit_reason)`` of a parameter as the shipped schema describes it.

    *value_of* gives the entity's parsed value of another parameter (``None`` when unset), for ``[MW or count]``;
    *members* the entity classes of a group's members, for ``[MW or MWh]``.  Without them the context form and a
    reason in words are returned.
    """
    definition = shipped_schema().parameters.get((entity_class, parameter))
    if definition is None:
        return None, "This FlexTool does not define the parameter."
    mark = bracket(definition.description)
    if mark is None:
        return None, "The parameter description gives no unit."
    if mark in NOT_UNITS:
        return None, None
    if mark == "[-]":
        return DIMENSIONLESS, "a ratio or a share"
    if mark in _COUNT_FORMS:
        capacity = _COUNT_FORMS[mark]
        if value_of is None:
            return f"{capacity} or count", (f"a number of units when virtual_unitsize is above 0, else the capacity "
                                            f"in {capacity}")
        size = _number(value_of("virtual_unitsize"))
        if size is not None and size > 0:
            return "count", (f"virtual_unitsize is {value_of('virtual_unitsize')}, so this is a number of units "
                             f"(capacity = {parameter} × virtual_unitsize)")
        return capacity, "virtual_unitsize is not set (or not above 0), so this is the capacity itself"
    if mark == _GROUP_FORM:
        from flextool.engine.autoscale._quantity_types import QuantityType, resolve_group_capacity_type

        word = {QuantityType.POWER: "MW", QuantityType.ENERGY: "MWh"}
        kinds = sorted({word[resolve_group_capacity_type(parameter, m)] for m in (members or ())
                        if m in ("unit", "connection", "node")})
        if len(kinds) == 1:
            return kinds[0], ("the group's members are units or connections" if kinds == ["MW"]
                              else "the group's members are storage nodes")
        if len(kinds) > 1:
            return "MW or MWh", "the group has units or connections (MW) and storage nodes (MWh) as members"
        return "MW or MWh", "MW for a group of units and connections, MWh for a group of storage nodes"
    unit = mark[1:-1]
    rest = definition.description[len(mark):]
    per = _PER_RE.match(rest)
    if per:
        unit = f"{unit} {per.group(1).strip()}"
    return unit, None
