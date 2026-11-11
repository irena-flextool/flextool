"""The unit of an input parameter, from the shipped schema's description bracket (P0 §4.3).

``[-]`` is a ratio or share (shown as dimensionless); ``[MW or MWh]`` of a group depends on its members (units and
connections: MW, storage nodes: MWh, ``resolve_group_capacity_type``); ``[MW or count]`` / ``[MWh or count]`` of
``existing`` depends on the entity's ``virtual_unitsize`` (above 0: a number of units).  Markers that are not units
(``[choice]``, ``[yes/no]``, ``[name]``, ``[text]``, ``[version number]``) give no unit.  The tools word the unit
with :func:`flextool.assistant.unit_wording.shown_unit`.

A node can set ``unit_of_measure`` (m3, t, MWh_th; default MWh).  With *nodes* (a :class:`NodeTopology` of the
scenario) the MWh / MW / kWh / kW of a bracket are rewritten for the nodes the parameter is about
(:func:`node_sources`): the amount tokens (MWh, kWh) to the node's unit A, the rate tokens (MW, kW) to its rate R
(m3 -> m3/h) or the kilo forms.  Each token has its own source (critique M8): ``unit__inputNode.startup_energy``
``[MWh per MW started]`` takes MWh from the input node and MW from the unit's capacity side.  Nodes of different
units give ``Mixed units of measure!`` and the reason names them.  The power features (reserves, inertia,
non-synchronous share, capacity margin, ``MWs``, ``MVA``) are never rewritten.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from flextool import units_of_measure as uom
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


@dataclass(frozen=True)
class UnitInfo:
    """The unit of a parameter: its text, the reason in words (or ``None``), and whether a node's
    ``unit_of_measure`` set it (then the tools show it plain, not as a convention unit)."""

    unit: str | None
    reason: str | None
    node_set: bool = False


def parameter_unit(entity_class: str, parameter: str, *, value_of: Callable[[str], Any] | None = None,
                   members: Iterable[str] | None = None, nodes: "NodeTopology | None" = None,
                   byname: tuple[str, ...] | None = None) -> tuple[str | None, str | None]:
    """``(unit, unit_reason)`` of a parameter as the shipped schema describes it (:func:`parameter_unit_info`)."""
    info = parameter_unit_info(entity_class, parameter, value_of=value_of, members=members, nodes=nodes,
                               byname=byname)
    return info.unit, info.reason


def parameter_unit_info(entity_class: str, parameter: str, *, value_of: Callable[[str], Any] | None = None,
                        members: Iterable[str] | None = None, nodes: "NodeTopology | None" = None,
                        byname: tuple[str, ...] | None = None) -> UnitInfo:
    """The :class:`UnitInfo` of a parameter as the shipped schema describes it.

    *value_of* gives the entity's parsed value of another parameter (``None`` when unset), for ``[MW or count]``;
    *members* the entity classes of a group's members, for ``[MW or MWh]``.  Without them the context form and a
    reason in words are returned.  *nodes* and *byname* (the entity) rewrite the units for the nodes'
    ``unit_of_measure`` (:func:`follow_nodes`); without them, or when those nodes are MWh, the default text.
    """
    unit, reason = _schema_unit(entity_class, parameter, value_of=value_of, members=members)
    if unit is None or nodes is None or byname is None or not nodes.node_units:
        return UnitInfo(unit, reason)
    return follow_nodes(UnitInfo(unit, reason), entity_class, parameter, tuple(byname), nodes)


def _schema_unit(entity_class: str, parameter: str, *, value_of: Callable[[str], Any] | None = None,
                 members: Iterable[str] | None = None) -> tuple[str | None, str | None]:
    """``(unit, unit_reason)`` of a parameter from the shipped schema's bracket (see :func:`parameter_unit_info`)."""
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


# ---------------------------------------------------------------------------------------------------------
# Node units of measure

#: An MWh / kWh / MW / kW of a unit text (not MWs, MVA, MW_th or a name that contains it).
_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(kWh|MWh|kW|MW)(?![A-Za-z0-9_])")
#: Parameters that are power features: they stay in MW whatever the nodes (the validators warn about non-MWh
#: members of such groups).
_POWER_FEATURES = frozenset({
    ("unit__inputNode", "inertia_constant"), ("unit__outputNode", "inertia_constant"),
    ("group", "capacity_margin"), ("group", "penalty_capacity_margin"), ("group", "penalty_non_synchronous"),
    ("group", "candidate_precapacity_to_avoid_big_m"), ("model", "max_flow_for_unconstrained_variables"),
})
_POWER_CLASSES = frozenset({"reserve__upDown__group", "reserve__upDown__unit__node",
                            "reserve__upDown__connection__node", "model"})


@dataclass(frozen=True)
class NodeTopology:
    """What the input tools need to rewrite units for the nodes' ``unit_of_measure``: the nodes whose unit is not
    MWh (``node_units``; empty = every node MWh, nothing is rewritten) and the active topology of the scenario."""

    node_units: dict[str, str]
    outputs: dict[str, tuple[str, ...]]         # unit -> its output nodes (unit__outputNode)
    inputs: dict[str, tuple[str, ...]]          # unit -> its input nodes (unit__inputNode)
    connection_ends: dict[str, tuple[str, ...]]  # connection -> its nodes (connection__node__node)
    commodity_nodes: dict[str, tuple[str, ...]]  # commodity -> its nodes (commodity__node)
    group_members: dict[str, tuple[tuple[str, str], ...]]  # group -> (class, member) of group__node/unit/connection
    flowgroup_nodes: dict[str, tuple[str, ...]]  # flowGroup -> the nodes of its arcs

    def capacity_side(self, unit: str) -> tuple[str, ...]:
        """A unit's capacity-side nodes: its outputs, else its inputs."""
        return self.outputs.get(unit) or self.inputs.get(unit, ())


def node_topology(view) -> NodeTopology:
    """The :class:`NodeTopology` of a scenario view of the input database (``_db.ScenarioView``): the active
    entities only.  Empty node units (nothing to rewrite) when no node sets a ``unit_of_measure`` other than MWh."""
    db = view.db

    def active(cls: str):
        for byname, item in db.entities.get(cls, {}).items():
            if view.active(cls, byname)[0]:
                yield item.elements or byname

    node_units = uom.node_units_from_rows(
        (byname[0], item.parsed()) for byname in db.entities.get("node", {})
        if view.active("node", byname)[0] and (item := view.value("node", byname, "unit_of_measure")) is not None)
    if not node_units:
        return NodeTopology({}, {}, {}, {}, {}, {}, {})

    def pairs(cls: str, first: int = 0, second: int = 1) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for el in active(cls):
            if len(el) > max(first, second) and el[second] not in out.setdefault(el[first], []):
                out[el[first]].append(el[second])
        return {k: tuple(v) for k, v in out.items()}

    ends: dict[str, list[str]] = {}
    for el in active("connection__node__node"):
        for n in el[1:3]:
            if n not in ends.setdefault(el[0], []):
                ends[el[0]].append(n)
    members: dict[str, list[tuple[str, str]]] = {}
    for cls, kind in (("group__node", "node"), ("group__unit", "unit"), ("group__connection", "connection")):
        for el in active(cls):
            members.setdefault(el[0], []).append((kind, el[1]))
    flow: dict[str, list[str]] = {}
    for cls in ("flowGroup__unit__node", "flowGroup__connection__node"):
        for el in active(cls):
            if len(el) == 3 and el[2] not in flow.setdefault(el[0], []):
                flow[el[0]].append(el[2])
    return NodeTopology(node_units, pairs("unit__outputNode"), pairs("unit__inputNode"),
                        {k: tuple(v) for k, v in ends.items()}, pairs("commodity__node"),
                        {k: tuple(v) for k, v in members.items()}, {k: tuple(v) for k, v in flow.items()})


@dataclass(frozen=True)
class _Source:
    """The nodes one kind of token follows: *nodes*, read as one amount (``units_of``) or as a unit's capacity
    side (*capacity*: :func:`~flextool.units_of_measure.capacity_units`), named *what* in the reason."""

    nodes: tuple[str, ...]
    what: str
    capacity: bool = False


def node_sources(entity_class: str, parameter: str, byname: tuple[str, ...],
                 nodes: NodeTopology) -> "tuple[_Source, _Source] | None":
    """``(amount source, rate source)`` of a parameter: which nodes its MWh / kWh and its MW / kW are of, or
    ``None`` when the parameter does not follow the nodes (a power feature, a class without nodes)."""
    if entity_class in _POWER_CLASSES or (entity_class, parameter) in _POWER_FEATURES:
        return None
    name = byname[0] if byname else ""
    if entity_class == "node":
        src = _Source((name,), f"node {name}")
        return src, src
    if entity_class in ("unit__inputNode", "unit__outputNode", "connection__node") and len(byname) >= 2:
        arc = _Source((byname[1],), f"node {byname[1]}")
        if entity_class == "unit__inputNode" and parameter == "startup_energy":
            return arc, _Source(nodes.capacity_side(byname[0]), f"the capacity side of unit {byname[0]}",
                                capacity=True)
        return arc, arc
    if entity_class == "unit":
        side = _Source(nodes.capacity_side(name), f"the capacity side of unit {name}", capacity=True)
        return side, side
    if entity_class == "connection":
        ends = _Source(nodes.connection_ends.get(name, ()), f"the nodes of connection {name}")
        return ends, ends
    if entity_class == "commodity":
        src = _Source(nodes.commodity_nodes.get(name, ()), f"the nodes of commodity {name}")
        return src, src
    if entity_class == "flowGroup":
        src = _Source(nodes.flowgroup_nodes.get(name, ()), f"the nodes of flowGroup {name}")
        return src, src
    if entity_class == "group":
        found: list[str] = []
        for kind, member in nodes.group_members.get(name, ()):
            side = ((member,) if kind == "node" else nodes.capacity_side(member) if kind == "unit"
                    else nodes.connection_ends.get(member, ()))
            found.extend(n for n in side if n not in found)
        src = _Source(tuple(found), f"the members of group {name}")
        return src, src
    return None


def _grouped(nodes: Iterable[str], node_units: dict[str, str]) -> str:
    """The nodes grouped by unit, ``MWh: elec; m3: water``."""
    by_unit: dict[str, list[str]] = {}
    for n in nodes:
        by_unit.setdefault(uom.node_unit(n, node_units), []).append(n)
    return "; ".join(f"{u}: {', '.join(ns)}" for u, ns in by_unit.items())


def follow_nodes(info: UnitInfo, entity_class: str, parameter: str, byname: tuple[str, ...],
                 nodes: NodeTopology) -> UnitInfo:
    """*info* with its MWh / kWh / MW / kW rewritten for the nodes of :func:`node_sources` (unchanged when they
    are MWh, the parameter does not follow its nodes, or the text has no such token)."""
    unit = info.unit
    if unit is None or not _TOKEN.search(unit):
        return info
    sources = node_sources(entity_class, parameter, byname, nodes)
    if sources is None:
        return info
    amount, rate = sources
    units = nodes.node_units
    if all(uom.is_default(uom.node_unit(n, units)) for n in amount.nodes + rate.nodes):
        return info
    a_unit = uom.units_of(amount.nodes, units)
    r_unit = uom.units_of(rate.nodes, units)
    used = {m.group(1) for m in _TOKEN.finditer(unit)}
    # One quantity over nodes of different units has no unit.  A unit's capacity over outputs in different units
    # is not mixed: it is shared or capped per output (owner decision S1, capacity_units).
    for src, u, tokens in ((amount, a_unit, {"MWh", "kWh"}), (rate, r_unit, {"MW", "kW"})):
        if u == uom.MIXED_UOM and used & tokens and not (src.capacity and tokens == {"MW", "kW"}):
            return UnitInfo(uom.MIXED_UOM, f"{src.what} have different units of measure "
                                           f"({_grouped(src.nodes, units)})", node_set=True)

    def rate_text(template: str) -> str:
        if rate.capacity:
            return uom.capacity_units(template, rate.nodes, units, shared=True)
        return uom.fill(template, r_unit)

    values = {"MWh": uom.fill("{A}", a_unit), "kWh": uom.fill("{kA}", a_unit),
              "MW": rate_text("{R}"), "kW": rate_text("{kR}")}

    def substitute(match: re.Match) -> str:
        value = values[match.group(1)]
        before = unit[match.start() - 1] if match.start() > 0 else ""
        compound = not (value.startswith("(") and value.endswith(")")) and ("/" in value or " " in value)
        # A shared capacity (MW + MW_th) is one rate: in parentheses wherever it stands.
        return f"({value})" if compound and (before == "/" or " + " in value) else value

    text = _TOKEN.sub(substitute, unit)
    if text.endswith(" or other unit"):
        text = text[:-len(" or other unit")]
    shown = sorted({n for n in amount.nodes + rate.nodes if not uom.is_default(uom.node_unit(n, units))})
    reason = "set by the unit_of_measure of " + ", ".join(
        f"node {n} ({uom.node_unit(n, units)})" for n in shown)
    if info.reason:
        reason = f"{info.reason}; {reason}"
    return UnitInfo(text, reason, node_set=True)
