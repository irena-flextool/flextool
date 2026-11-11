"""Build the :class:`~flextool.process_outputs._output_meta.UnitContext` of one run from its sets.

The node units come from the provider frame ``input/node__unit_of_measure`` (the explicit, non-MWh values of
``node.unit_of_measure``).  When every node is MWh the context is empty and nothing else is read, so the default
run does exactly what it did before.  Otherwise the maps that say which nodes a result column (or row) carries
are built once per run from the post-processing sets ``s`` and the conversion coefficients in ``par``:

* a unit's capacity side: its output nodes (``s.process_sink``, in arc order) whose ``conversion_flow_coeff``
  is not 0 (a 0 output is not in the capacity constraints), else its input nodes (``s.process_source``);
* the units whose outputs share the capacity (the provider frame ``input/process_method``, the method the
  model is built with, :func:`~flextool.units_of_measure.shares_capacity`; ``s.process_method`` is only a
  1var / 2way seed); a unit without a method does not share;
* a connection's two ends (its source and sink arcs);
* a nodeGroup's member nodes (``s.group_node``) and a flowGroup's arc nodes (``s.group_process_node``, after
  ``_backfill_group_indicator_sets``).
"""
from __future__ import annotations

import pandas as pd

from flextool import units_of_measure as uom
from flextool.process_outputs._output_meta import UnitContext


def _pairs(index) -> list[tuple[str, str]]:
    """``(first, second)`` level values of a two-level set, as text, in their order (``[]`` when absent)."""
    if index is None or len(index) == 0 or getattr(index, "nlevels", 1) < 2:
        return []
    return list(zip((str(v) for v in index.get_level_values(0)), (str(v) for v in index.get_level_values(1))))


def _zero_outputs(par) -> set[tuple[str, str]]:
    """The ``(unit, sink)`` arcs whose ``conversion_flow_coeff`` is 0 (not in the capacity constraints)."""
    coeff = getattr(par, "process_sink_conversion_flow_coeff", None) if par is not None else None
    if coeff is None or not isinstance(coeff, pd.Series) or coeff.empty or coeff.index.nlevels < 2:
        return set()
    frame = pd.DataFrame({"p": coeff.index.get_level_values(0).astype(str),
                          "n": coeff.index.get_level_values(1).astype(str),
                          "nonzero": pd.to_numeric(coeff.to_numpy(), errors="coerce") != 0.0})
    # An arc is left out only when its coefficient is 0 everywhere (a time- or period-varying value counts).
    any_nonzero = frame.groupby(["p", "n"], sort=False)["nonzero"].any()
    return {key for key, nonzero in any_nonzero.items() if not nonzero}


def _members(index, *, node_level: int) -> dict[str, tuple[str, ...]]:
    """``{group: (node, ...)}`` of a set whose first level is the group and level *node_level* the node."""
    if index is None or len(index) == 0 or getattr(index, "nlevels", 1) <= node_level:
        return {}
    out: dict[str, list[str]] = {}
    for g, n in zip(index.get_level_values(0), index.get_level_values(node_level)):
        nodes = out.setdefault(str(g), [])
        if str(n) not in nodes:
            nodes.append(str(n))
    return {g: tuple(nodes) for g, nodes in out.items()}


def _methods(provider) -> list[tuple[str, str]]:
    """``(process, method)`` of the provider frame ``input/process_method`` (``[]`` without it)."""
    if provider is None or not provider.has("input/process_method"):
        return []
    frame = provider.get("input/process_method")
    if frame is None or frame.height == 0 or not {"process", "method"} <= set(frame.columns):
        return []
    return [(str(p), str(m)) for p, m in frame.select("process", "method").iter_rows()]


def unit_context_from(s, par=None, provider=None) -> UnitContext:
    """The :class:`UnitContext` of a run: its node units (from *provider*) and, when some node is not MWh, the
    capacity sides, shared-capacity units, connection ends and group members from the sets *s* (see the module
    docstring).  An empty context when every node is MWh, when there is no provider or when the database has no
    ``unit_of_measure`` definition."""
    node_units = uom.read_node_units(provider)
    if not node_units:
        return UnitContext()
    units = {str(u) for u in getattr(s, "process_unit", [])}
    connections = {str(c) for c in getattr(s, "process_connection", [])}
    sinks = _pairs(getattr(s, "process_sink", None))
    sources = _pairs(getattr(s, "process_source", None))
    zero = _zero_outputs(par)
    outputs: dict[str, list[str]] = {}
    inputs: dict[str, list[str]] = {}
    for p, n in sinks:
        if p in units and (p, n) not in zero and n not in outputs.setdefault(p, []):
            outputs[p].append(n)
    for p, n in sources:
        if p in units and n not in inputs.setdefault(p, []):
            inputs[p].append(n)
    all_outputs = {p for p, _n in sinks if p in units}
    capacity_nodes = {}
    for u in sorted(units):
        side = outputs.get(u) or ([n for p, n in sinks if p == u] if u in all_outputs else []) or inputs.get(u, [])
        capacity_nodes[u] = tuple(side)
    connection_nodes: dict[str, list[str]] = {}
    for p, n in sources + sinks:
        if p in connections and n not in connection_nodes.setdefault(p, []):
            connection_nodes[p].append(n)
    shared = frozenset(p for p, m in _methods(provider) if p in units and uom.shares_capacity(m))
    return UnitContext(
        node_units=node_units,
        capacity_nodes=capacity_nodes,
        shared_capacity=shared,
        group_nodes=_members(getattr(s, "group_node", None), node_level=1),
        connection_nodes={c: tuple(nodes) for c, nodes in connection_nodes.items()},
        flowgroup_nodes=_members(getattr(s, "group_process_node", None), node_level=2),
    )
