"""Units of measure of nodes: the one derivation rule from a node's ``unit_of_measure`` to the units shown.

A node's ``unit_of_measure`` is the unit of its *amounts* (``A``: MWh, m3, t, MWh_th; default MWh).  FlexTool
converts nothing: the value is a label.  Time steps are hours, so the node's *rate* (``R``) is ``A`` per hour,
derived by :func:`rate_of` (MWh -> MW, MWh_th -> MW_th, m3 -> m3/h).  Unit templates use the tokens

* ``{A}``  the amount unit (``node_state``, inflow per step; ``{A}/a`` for annual results, ``CUR/{A}`` prices),
* ``{R}``  the rate unit (flows, capacities),
* ``{kR}`` / ``{kA}`` the kilo forms of the per-kW / per-kWh costs (:func:`kilo_rate` / :func:`kilo_amount`),

filled by :func:`fill`.  One quantity summed over nodes whose units differ has no unit: :data:`MIXED_UOM`
(:func:`units_of`).  A unit's capacity over outputs in different units is not a sum of mixed units but a
capacity used in a known way (:func:`capacity_units`).

The output resolver, the plot labels, the input validators and the assistant use this module, so the rule is
the same everywhere.  No dependencies: the module imports nothing from FlexTool or third-party packages.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import NamedTuple

#: The amount unit of a node without ``unit_of_measure`` (the schema default).
DEFAULT_A = "MWh"
#: The label of one quantity summed (or stacked, averaged) over nodes whose units of measure differ.  It replaces
#: the whole unit text (never ``Mixed units of measure!/a``).
MIXED_UOM = "Mixed units of measure!"
#: The provider frame (``input/node__unit_of_measure.csv``) that holds the explicit values: ``node``,
#: ``unit_of_measure``.
NODE_UNITS_FRAME = "input/node__unit_of_measure"
#: The template tokens, longest first (``{kR}`` before ``{R}``).
TOKENS = ("{kR}", "{kA}", "{R}", "{A}")

_WATT_HOUR = re.compile(r"^(.*W)h(.*)$")
_MEGAWATT_HOUR = re.compile(r"^MWh(.*)$")
#: A value that names a rate, not an amount: ends in ``/h`` or ``/s``, or a ``W`` not followed by ``h``.
_RATE_LIKE = re.compile(r"(/h|/s)$|W(?!h)")


def normalise(value) -> str:
    """The amount unit of a ``unit_of_measure`` value: the text stripped; empty or ``None`` -> :data:`DEFAULT_A`."""
    if value is None:
        return DEFAULT_A
    text = str(value).strip()
    return text or DEFAULT_A


def is_default(unit: str) -> bool:
    """Whether *unit* (an amount unit) is the default :data:`DEFAULT_A`."""
    return normalise(unit) == DEFAULT_A


def rate_of(amount: str) -> str:
    """The rate unit of the amount unit *amount* (amount per hour).

    ``<x>Wh<y>`` -> ``<x>W<y>`` (MWh -> MW, MWh_th -> MW_th, kWh -> kW, GWh -> GW), else ``amount + "/h"``
    (m3 -> m3/h, t -> t/h).  Case-sensitive: ``mwh`` -> ``mwh/h``.  :data:`MIXED_UOM` stays mixed."""
    if amount == MIXED_UOM:
        return MIXED_UOM
    match = _WATT_HOUR.match(amount)
    if match:
        return match.group(1) + match.group(2)
    return f"{amount}/h"


def kilo_rate(amount: str) -> str:
    """The rate a per-kW cost is per: ``kW<suffix>`` for ``MWh<suffix>`` (MWh -> kW, MWh_th -> kW_th), else
    ``(0.001 <rate>)`` (m3 -> ``(0.001 m3/h)``)."""
    if amount == MIXED_UOM:
        return MIXED_UOM
    match = _MEGAWATT_HOUR.match(amount)
    if match:
        return f"kW{match.group(1)}"
    return f"(0.001 {rate_of(amount)})"


def kilo_amount(amount: str) -> str:
    """The amount a per-kWh cost is per: ``kWh<suffix>`` for ``MWh<suffix>``, else ``(0.001 <amount>)``."""
    if amount == MIXED_UOM:
        return MIXED_UOM
    match = _MEGAWATT_HOUR.match(amount)
    if match:
        return f"kWh{match.group(1)}"
    return f"(0.001 {amount})"


def has_tokens(template: str) -> bool:
    """Whether *template* contains a unit token (a literal label has none)."""
    return any(token in template for token in TOKENS)


def fill(template: str, amount: str) -> str:
    """*template* with its tokens filled for the amount unit *amount*.

    A template without tokens is returned as written.  A template with tokens and ``amount ==``
    :data:`MIXED_UOM` gives :data:`MIXED_UOM` (the whole text: a mixed sum has no unit, per year or not)."""
    if not has_tokens(template):
        return template
    if amount == MIXED_UOM:
        return MIXED_UOM
    return (template.replace("{kR}", kilo_rate(amount)).replace("{kA}", kilo_amount(amount))
            .replace("{R}", rate_of(amount)).replace("{A}", amount))


def node_unit(node: str, node_units: Mapping[str, str]) -> str:
    """The amount unit of *node* (:data:`DEFAULT_A` when *node_units* has no entry)."""
    return normalise(node_units.get(node))


def distinct_units(nodes: Iterable[str], node_units: Mapping[str, str]) -> list[str]:
    """The distinct amount units of *nodes*, in the order of first appearance."""
    out: list[str] = []
    for node in nodes:
        unit = node_unit(node, node_units)
        if unit not in out:
            out.append(unit)
    return out


def units_of(nodes: Iterable[str], node_units: Mapping[str, str]) -> str:
    """The one amount unit of a set of nodes, :data:`MIXED_UOM` when they differ, :data:`DEFAULT_A` for none."""
    units = distinct_units(nodes, node_units)
    if not units:
        return DEFAULT_A
    return units[0] if len(units) == 1 else MIXED_UOM


def shares_capacity(method: str) -> bool:
    """Whether a process of *method* (``input/process_method``) bounds the sum of its outputs by its capacity.

    An indirect unit (``METHOD_INDIRECT``, the ``nvar`` methods: more than one input or output arc, or a delay)
    gets ``maxOutputSum`` (with ``maxOutputSum_online``): the sum of its capacity-side outputs is at most its
    capacity, and ``maxFlow`` caps each arc as well.  A direct unit has one flow, capped by ``maxFlow``.  The
    indirect methods are exactly the ``_nvar_`` ones (a test pins this against ``METHOD_INDIRECT``)."""
    return "_nvar_" in method


def capacity_units(template: str, nodes: Iterable[str], node_units: Mapping[str, str], *,
                   shared: bool) -> str:
    """The unit of a unit's capacity quantity (*template*: ``{R}``, ``CUR/{R}``, ``CUR/{kR}``) over its
    capacity-side nodes *nodes* (its output nodes with a non-zero ``conversion_flow_coeff``, in output-arc
    order; its input nodes when it has no output).

    The label shows how the capacity is used (owner decision S1); no warning:

    * all nodes in one unit: that unit (``MW``), so default outputs are unchanged;
    * *shared* (:func:`shares_capacity`: the outputs share the capacity, ``maxOutputSum``; when per-arc caps apply
      as well, the sum is the tighter description): the rates joined with ``" + "``, in parentheses when the
      template has more text (``MW + MW_th``, ``CUR/(MW + MW_th)``);
    * only per-arc caps (capacity x ``capacity_max_coeff``): each arc in its own unit, the filled templates joined
      with ``"; "`` (``MW; MW_th``)."""
    units = distinct_units(nodes, node_units) or [DEFAULT_A]
    if len(units) == 1 or not has_tokens(template):
        return fill(template, units[0])
    if not shared:
        return "; ".join(fill(template, unit) for unit in units)
    out = template
    for token, form in (("{kR}", kilo_rate), ("{kA}", kilo_amount), ("{R}", rate_of), ("{A}", lambda a: a)):
        if token not in out:
            continue
        joined = " + ".join(form(unit) for unit in units)
        out = out.replace(token, joined if out == token else f"({joined})")
    return out


def looks_like_rate(value: str) -> bool:
    """Whether a ``unit_of_measure`` value names a rate (``m3/h``, ``kg/s``, ``MW``), not an amount."""
    return bool(_RATE_LIKE.search(normalise(value)))


def node_units_from_rows(rows: Iterable[tuple[str, object]]) -> dict[str, str]:
    """``{node: amount unit}`` of ``(node, unit_of_measure)`` rows, the non-default units only (an empty map is
    the default case)."""
    out: dict[str, str] = {}
    for node, value in rows:
        unit = normalise(value)
        if unit != DEFAULT_A:
            out[str(node)] = unit
    return out


def read_node_units(provider) -> dict[str, str]:
    """``{node: amount unit}`` of the nodes with a non-default ``unit_of_measure``, from the provider frame
    :data:`NODE_UNITS_FRAME` (columns ``node``, ``unit_of_measure``).

    Empty when there is no provider, no frame (a database without the definition) or no explicit value: every
    node is then :data:`DEFAULT_A`."""
    if provider is None or not provider.has(NODE_UNITS_FRAME):
        return {}
    frame = provider.get(NODE_UNITS_FRAME)
    if frame is None or frame.height == 0 or "node" not in frame.columns or "unit_of_measure" not in frame.columns:
        return {}
    return node_units_from_rows(frame.select("node", "unit_of_measure").iter_rows())


class UnitSourceKind(Enum):
    """How a level of a result table carries the node(s) whose unit a column (or row) takes."""

    #: The level value is a node (``node``, ``sink``, ``source``; the ``node`` level of an arc table).
    NODE = "node"
    #: The level value is a connection: its two end nodes (one quantity over both, so mixed when they differ).
    CONNECTION_ENDS = "connection_ends"
    #: The level value is a unit: its capacity side (:func:`capacity_units`).
    CAPACITY_SIDE = "capacity_side"
    #: The level value is an entity of the table's class (node: itself; unit: capacity side; connection: ends).
    ENTITY_BY_CLASS = "entity_by_class"
    #: The level value is a nodeGroup: its member nodes (``group__node``).
    GROUP_MEMBERS = "group_members"
    #: The level value is a flowGroup: the nodes of its arcs.
    FLOWGROUP_ARCS = "flowgroup_arcs"


class UnitSource(NamedTuple):
    """Where a result table's unit comes from: the *level* that carries the node(s), how (*kind*), and whether
    the level is a column level (``row=False``) or a row index level (``row=True``, a row-keyed table whose
    unit varies by row)."""

    level: str
    kind: UnitSourceKind
    row: bool = False
