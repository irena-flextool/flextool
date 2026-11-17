"""The lifetime and salvage parameters of units, nodes and connections, and the payback-time rule.

An entity gives three numbers:

* ``lifetime_technical`` — how long the capacity operates: the end of life of existing capacity, the investment
  and retirement windows;
* ``payback_time`` — the years over which the investment is annualised (the annuity factor); an entity without
  it pays back over its ``lifetime_technical``;
* ``net_salvage_value`` — the value of retired capacity.

The payback time is read, per entity, from the first name of :data:`ROLE_CHAIN` under which the entity has a value
(:func:`role_sources`): ``payback_time``, else ``lifetime_technical``, for all its periods.  The database readers,
the CSV-generation readers and the input checks each pass :func:`role_sources` their own way of listing the
entities that have a value of a parameter.  The ``lifetime_method`` value :data:`NO_REINVESTMENT` retires the
capacity at the end of its technical lifetime and allows investment only in the first technical lifetime.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

LIFETIME_TECHNICAL = "lifetime_technical"
PAYBACK_TIME = "payback_time"
NET_SALVAGE_VALUE = "net_salvage_value"

#: Role → the parameter names it is read from, in order: an entity takes the first one it has a value of.
#: ``payback_time`` falls back to the technical lifetime of the same entity.
ROLE_CHAIN: Mapping[str, tuple[str, ...]] = {
    PAYBACK_TIME: (PAYBACK_TIME, LIFETIME_TECHNICAL),
}

NO_REINVESTMENT = "no_reinvestment"

END_OF_LIFE_METHOD = "end_of_life_method"
#: The ``end_of_life_method`` values: how an end of life is rounded to a period start (``_run_calendar``).
END_OF_LIFE_METHODS: tuple[str, ...] = ("closest_seam", "previous_seam", "next_seam")
#: The ``end_of_life_method`` of an entity without one (the schema default; a database without the definition).
END_OF_LIFE_METHOD_DEFAULT = "next_seam"

ENTITY_CLASSES = ("unit", "node", "connection")


def role_sources(role: str, entities_with: Callable[[str], Iterable[str]]) -> dict[str, str]:
    """``{entity: parameter name}``: for each entity with a value of ``role``, the name it is read from.

    ``entities_with(name)``: the entities that have a value of parameter ``name`` (none when the parameter is not
    defined), within one entity class (or the classes read together).  Each entity takes the first name of
    ``ROLE_CHAIN[role]`` it has a value of, for all its periods: a ``payback_time`` map that lacks a period leaves
    that period without a payback time (it does not fall back to the technical lifetime for that period).  An
    entity without any is not in the result.
    """
    out: dict[str, str] = {}
    for name in ROLE_CHAIN[role]:
        for e in sorted({str(e) for e in entities_with(name)}):
            out.setdefault(e, name)
    return out


def source_entities_with(source: "InputSource", entity_class: str) -> Callable[[str], set[str]]:
    """``entities_with`` of :func:`role_sources` for ``entity_class`` of the database ``source``: the entities
    with an explicit, non-null value (a schema default is not a value of the entity: an entity without an explicit
    ``payback_time`` falls back to its technical lifetime whatever the definition's default)."""
    def entities_with(name: str) -> set[str]:
        try:
            frame = source.parameter_explicit(entity_class, name)
        except (KeyError, AttributeError):
            return set()
        if frame.height == 0 or "name" not in frame.columns:
            return set()
        return {str(e) for e in frame.filter(frame["value"].is_not_null())["name"].to_list()}

    return entities_with


def source_role_sources(source: "InputSource", entity_class: str, role: str) -> dict[str, str]:
    """:func:`role_sources` of ``entity_class`` in the database ``source``."""
    return role_sources(role, source_entities_with(source, entity_class))


def lookup_roles(lookup) -> dict[str, dict[str, str]]:
    """``{role: role_sources(role)}`` of every role for the CSV-generation lookup ``lookup`` (a ``PdLookup``: its
    ``explicit_rows(name)`` lists the explicit values of a parameter)."""
    def entities_with(name: str) -> set[str]:
        return {str(e) for e, _value in lookup.explicit_rows(name)}

    return {role: role_sources(role, entities_with) for role in ROLE_CHAIN}


def role_param(roles: Mapping[str, Mapping[str, str]], e: str, role: str) -> str:
    """The parameter name entity ``e`` gives ``role`` in ``roles`` (:func:`lookup_roles`); the last name of the
    chain when it gives none (a lookup of it then finds no value)."""
    return roles[role].get(e, ROLE_CHAIN[role][-1])
