"""The roles of the lifetime and salvage parameters, and the names they are read under.

The database gives a unit, node or connection three numbers where it used to give two:

* ``lifetime_technical`` — how long the capacity operates: the end of life of existing capacity, the investment
  and retirement windows;
* ``payback_time`` — the years over which the investment is annualised (the annuity factor); an entity without
  it pays back over its ``lifetime_technical``;
* ``net_salvage_value`` — the value of retired capacity (``salvage_value`` before).

Older databases carry ``lifetime`` for both lifetimes and ``salvage_value``.  Each role is read, per entity, from
the first name of its chain (:data:`ROLE_CHAIN`) under which the entity has a value.  A class that carries values
under both a new name and the old name it replaces is an error.  The ``lifetime_method`` value ``no_investment``
is now ``no_reinvestment`` (:func:`canonical_lifetime_method`).

:func:`role_sources` is the one place of that rule; the database readers, the CSV-generation readers and the
input checks each pass it their own way of listing the entities that have a value of a parameter.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

LIFETIME_TECHNICAL = "lifetime_technical"
PAYBACK_TIME = "payback_time"
NET_SALVAGE_VALUE = "net_salvage_value"

#: New parameter name → the old name it replaces.
ROLE_LEGACY: Mapping[str, str] = {
    LIFETIME_TECHNICAL: "lifetime",
    PAYBACK_TIME: "lifetime",
    NET_SALVAGE_VALUE: "salvage_value",
}

#: Role → the parameter names it is read from, in order: an entity takes the first one it has a value of.
#: ``payback_time`` falls back to the technical lifetime of the same entity.
ROLE_CHAIN: Mapping[str, tuple[str, ...]] = {
    LIFETIME_TECHNICAL: (LIFETIME_TECHNICAL, "lifetime"),
    PAYBACK_TIME: (PAYBACK_TIME, LIFETIME_TECHNICAL, "lifetime"),
    NET_SALVAGE_VALUE: (NET_SALVAGE_VALUE, "salvage_value"),
}

NO_REINVESTMENT = "no_reinvestment"
#: Old ``lifetime_method`` value → the value it is now.
LIFETIME_METHOD_LEGACY: Mapping[str, str] = {"no_investment": NO_REINVESTMENT}

ENTITY_CLASSES = ("unit", "node", "connection")


def canonical_lifetime_method(value: str) -> str:
    """``lifetime_method`` value ``value`` under its current name (``no_investment`` → ``no_reinvestment``)."""
    return LIFETIME_METHOD_LEGACY.get(value, value)


def canonical_method_expr(values: pl.Expr, parameter_name: str) -> pl.Expr:
    """The method strings ``values`` of the method parameter ``parameter_name``: a ``lifetime_method`` under its
    current value names (:func:`canonical_lifetime_method`), any other method as it is."""
    if parameter_name != "lifetime_method":
        return values
    return values.replace(dict(LIFETIME_METHOD_LEGACY))


def role_sources(role: str, entities_with: Callable[[str], Iterable[str]], *,
                 where: str) -> dict[str, str]:
    """``{entity: parameter name}``: for each entity with a value of ``role``, the name it is read from.

    ``entities_with(name)``: the entities that have a value of parameter ``name`` (none when the parameter is not
    defined), within one entity class (or the classes read together); ``where`` names them in the error.  Each
    entity takes the first name of ``ROLE_CHAIN[role]`` it has a value of, for all its periods: a ``payback_time``
    map that lacks a period leaves that period without a payback time (it does not fall back to the technical
    lifetime for that period).  An entity without any is not in the result.

    Raises :class:`FlexToolConfigError` when entities carry values under both a new name of the chain and the old
    name it replaces (a database is migrated to the new names as a whole).
    """
    chain = ROLE_CHAIN[role]
    have = {name: sorted({str(e) for e in entities_with(name)}) for name in chain}
    for new in chain:
        old = ROLE_LEGACY.get(new)
        if old in have and have[new] and have[old]:
            raise FlexToolConfigError(
                f"{where}: '{old}' is given ({', '.join(have[old])}) next to '{new}' ({', '.join(have[new])}). "
                f"'{old}' is replaced by {_replacements(old)}: give those, not '{old}'.")
    out: dict[str, str] = {}
    for name in chain:
        for e in have[name]:
            out.setdefault(e, name)
    return out


def _replacements(old: str) -> str:
    names = [f"'{new}'" for new, legacy in ROLE_LEGACY.items() if legacy == old]
    return " and ".join(names)


def physical_name(role: str, defined: Callable[[str], bool]) -> str:
    """The name ``role`` is given under: the role's own (new) name when ``defined(name)`` (the database defines
    it), else the old name it replaces.  Used to name the parameter in messages."""
    return role if defined(role) else ROLE_LEGACY[role]


def source_entities_with(source: "InputSource", entity_class: str) -> Callable[[str], set[str]]:
    """``entities_with`` of :func:`role_sources` for ``entity_class`` of the database ``source``: the entities
    with an explicit, non-null value."""
    from flextool.engine.input._param_shapes import _try_parameter_frame

    def entities_with(name: str) -> set[str]:
        frame = _try_parameter_frame(source, entity_class, name)
        if frame is None or frame.height == 0 or "name" not in frame.columns:
            return set()
        return {str(e) for e in frame.filter(frame["value"].is_not_null())["name"].to_list()}

    return entities_with


def source_defines(source: "InputSource", classes: Iterable[str] = ENTITY_CLASSES) -> Callable[[str], bool]:
    """``defined`` of :func:`physical_name` for the database ``source``: some class of ``classes`` defines the
    parameter."""
    def defined(name: str) -> bool:
        for cls in classes:
            try:
                source.parameter(cls, name)
            except KeyError:
                continue
            return True
        return False

    return defined


def source_role_sources(source: "InputSource", entity_class: str, role: str) -> dict[str, str]:
    """:func:`role_sources` of ``entity_class`` in the database ``source``."""
    return role_sources(role, source_entities_with(source, entity_class), where=entity_class)


def source_physical_name(source: "InputSource", role: str) -> str:
    """:func:`physical_name` of ``role`` in the database ``source``."""
    return physical_name(role, source_defines(source))


def lookup_roles(lookup, where: str) -> dict[str, dict[str, str]]:
    """``{role: role_sources(role)}`` of every role for the CSV-generation lookup ``lookup`` (a ``PdLookup``: its
    ``explicit_rows(name)`` lists the explicit values of a parameter); ``where`` names its entity classes."""
    def entities_with(name: str) -> set[str]:
        return {str(e) for e, _value in lookup.explicit_rows(name)}

    return {role: role_sources(role, entities_with, where=where) for role in ROLE_CHAIN}


def role_param(roles: Mapping[str, Mapping[str, str]], e: str, role: str) -> str:
    """The parameter name entity ``e`` gives ``role`` in ``roles`` (:func:`lookup_roles`); the old name when it
    gives none (a lookup of it then finds no value)."""
    return roles[role].get(e, ROLE_CHAIN[role][-1])
