"""Axis labels whose unit follows the nodes' ``unit_of_measure``.

The labels of ``default_plots.yaml`` (and of users' ``plot_settings.yaml``) may carry the unit tokens of
:mod:`flextool.units_of_measure` (``{A}``, ``{R}``, ``{kR}``, ``{kA}``): ``{A}/h`` for a flow, ``{A} (annualized)``
for an annual amount, ``Currency/{A}`` for a price.  A label without tokens is shown as written.

* Every node MWh (no :class:`~flextool.process_outputs._output_meta.UnitContext`, or a default one): the tokens
  are filled for MWh (:func:`fill_default`), so the labels read exactly as the literal labels did.
* Some node in another unit: an :class:`AxisLabeler` fills the label per subplot from the nodes the subplot's
  data carries (the table's level in :data:`~flextool.process_outputs._output_meta.UNIT_SOURCE`).  One unit:
  that unit.  Several, and the plot sums, stacks or averages over that level (rules ``m``, ``a``, ``y``, ``z``,
  ``s``): :data:`~flextool.units_of_measure.MIXED_UOM`.  Several otherwise (lines, separate bars): each filled
  label, joined with ``"; "``.  Subplots that resolve differently get a per-subplot label list.

The power-only tables (:data:`~flextool.process_outputs._output_meta.DEFAULT_UNITS_ONLY`: reserves, inertia,
non-synchronous share, capacity margin) never follow their nodes; their default labels have no tokens.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pandas as pd

from flextool import units_of_measure as uom

#: Plot rules that collapse a level into one value (sum, average, period-weighted sum / average) or have no
#: role; a level under one of them is summed over in the drawn value.
_COLLAPSING_RULES = frozenset("mayzx")
#: The rule that stacks a level's members on top of each other (a visual sum).
_STACKING_RULE = "s"


def fill_default(label):
    """*label* (a string, a per-subplot list or ``None``) with its unit tokens filled for MWh: the label of a
    run whose nodes are all MWh, identical to the literal label the tokens replaced."""
    if isinstance(label, str):
        return uom.fill(label, uom.DEFAULT_A)
    if isinstance(label, list):
        return [fill_default(x) for x in label]
    return label


def merged_context(ctx):
    """One :class:`~flextool.process_outputs._output_meta.UnitContext` for *ctx* (one context, one per compared
    scenario, or ``None``); ``None`` when every node of every scenario is MWh.

    A comparison plot draws several scenarios on one axis, so a node whose unit differs between the scenarios
    reads as :data:`~flextool.units_of_measure.MIXED_UOM`; the other maps are united."""
    from flextool.process_outputs._output_meta import UnitContext

    if ctx is None:
        return None
    if isinstance(ctx, UnitContext):
        return None if ctx.is_default else ctx
    contexts = [c for c in ctx.values() if c is not None]
    if not any(not c.is_default for c in contexts):
        return None
    nodes = {n for c in contexts for n in c.node_units}
    node_units: dict[str, str] = {}
    for n in sorted(nodes):
        units = {uom.node_unit(n, c.node_units) for c in contexts}
        node_units[n] = units.pop() if len(units) == 1 else uom.MIXED_UOM

    def united(name: str) -> dict:
        out: dict = {}
        for c in contexts:
            for k, v in getattr(c, name).items():
                out.setdefault(k, v)
        return out

    return UnitContext(
        node_units=node_units,
        capacity_nodes=united("capacity_nodes"),
        shared_capacity=frozenset().union(*(c.shared_capacity for c in contexts)),
        group_nodes=united("group_nodes"),
        connection_nodes=united("connection_nodes"),
        flowgroup_nodes=united("flowgroup_nodes"),
    )


def _level_values(index, level: str) -> list[str] | None:
    """The distinct values of the level *level* of *index* (first-seen order), ``None`` without that level."""
    names = list(getattr(index, "names", None) or [])
    if level not in names:
        return None
    values = index.get_level_values(level) if isinstance(index, pd.MultiIndex) else index
    return list(dict.fromkeys(str(v) for v in values))


def _sub_dict(sub, sub_level_names) -> dict[str, str]:
    """``{level name: value}`` of a subplot (or file member) value *sub* over the levels *sub_level_names*."""
    if sub is None or not sub_level_names:
        return {}
    values = sub if isinstance(sub, tuple) else (sub,)
    return {str(n): str(v) for n, v in zip(sub_level_names, values)}


@dataclass(frozen=True)
class AxisLabeler:
    """Fills the unit tokens of one plot's axis labels per subplot (see the module docstring).

    Built by :meth:`build` for one result table and plot configuration; :meth:`for_file_member` fixes the
    level values a file split takes away; :meth:`batch_label` gives the label of one figure (file batch)."""

    result_key: str
    ctx: object  # UnitContext (merged, with a node that is not MWh)
    level: str
    kind: object  # UnitSourceKind
    #: The data's column levels as a frame (``None`` when the level is a row level), to find the level's
    #: values that a collapsed subplot sums over.
    columns: pd.DataFrame | None
    #: The level's values in the data before the plot rules.
    values: tuple[str, ...]
    #: The level is summed, averaged or stacked in the drawn value.
    summed: bool
    fixed: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, result_key: str | None, ctx, df: pd.DataFrame, cfg) -> "AxisLabeler | None":
        """The labeler of *result_key*'s data *df* plotted with the configuration *cfg*, or ``None`` when the
        labels are the default ones (no node that is not MWh, a table that does not follow its nodes, or a
        level the rules cannot place): the caller then uses :func:`fill_default`."""
        from flextool.process_outputs._output_meta import DEFAULT_UNITS_ONLY, UNIT_SOURCE

        if not result_key or result_key in DEFAULT_UNITS_ONLY or result_key not in UNIT_SOURCE:
            return None
        merged = merged_context(ctx)
        if merged is None:
            return None
        source = UNIT_SOURCE[result_key]
        map_dims = getattr(cfg, "map_dimensions_for_plots", None)
        if not map_dims or len(map_dims) < 2 or not map_dims[1]:
            return None
        rules = str(map_dims[1]).replace("_", "")
        names = [str(n) for n in list(df.index.names) + list(df.columns.names)]
        if len(rules) != len(names) or source.level not in names:
            return None
        role = rules[names.index(source.level)]
        col_values = _level_values(df.columns, source.level)
        values = col_values if col_values is not None else _level_values(df.index, source.level)
        columns = None
        if col_values is not None and isinstance(df.columns, pd.MultiIndex):
            columns = df.columns.to_frame(index=False).astype(str)
            columns.columns = [str(c) for c in columns.columns]
        return cls(result_key, merged, source.level, source.kind, columns, tuple(values or ()),
                   summed=role in _COLLAPSING_RULES or role == _STACKING_RULE)

    def for_file_member(self, file_member, file_level_names) -> "AxisLabeler":
        """This labeler for the file of *file_member* (its values on the levels *file_level_names*)."""
        fixed = _sub_dict(file_member, [str(n) for n in file_level_names])
        return replace(self, fixed=fixed) if fixed else self

    def _label_of(self, template: str, value: str) -> str:
        from flextool.process_outputs._output_meta import entity_label

        node_template = template.replace("{kR}", "{kA}").replace("{R}", "{A}")
        return entity_label(template, self.kind, value, self.ctx, node_template=node_template)

    def _collapsed_values(self, known: Mapping[str, str]) -> tuple[str, ...]:
        """The level's values that a subplot with the level values *known* sums over (all of them when the
        level is a row level or nothing narrows them)."""
        if self.columns is None:
            return self.values
        frame = self.columns
        mask = pd.Series(True, index=frame.index)
        for name, value in known.items():
            if name in frame.columns and name != self.level:
                mask &= frame[name] == value
        return tuple(dict.fromkeys(frame.loc[mask, self.level])) if self.level in frame.columns else self.values

    def subplot_label(self, template, df_sub: pd.DataFrame, sub: Mapping[str, str]):
        """The label *template* filled for one subplot: its data *df_sub* and the values *sub* of the levels the
        subplot (and the file) took away."""
        if not isinstance(template, str) or not uom.has_tokens(template):
            return template
        known = {**self.fixed, **sub}
        if self.level in known:
            values: tuple[str, ...] | list[str] = (known[self.level],)
        else:
            found = _level_values(df_sub.columns, self.level)
            if found is None:
                found = _level_values(df_sub.index, self.level)
            values = found if found is not None else self._collapsed_values(known)
        labels = list(dict.fromkeys(self._label_of(template, v) for v in values))
        if not labels:
            return uom.fill(template, uom.DEFAULT_A)
        if len(labels) == 1:
            return labels[0]
        if self.summed or any(uom.MIXED_UOM in label for label in labels):
            return uom.MIXED_UOM
        from flextool.process_outputs._output_meta import summarise_units
        return summarise_units(labels)

    def batch_label(self, label, plots):
        """The label of one figure from the configured *label* (a string, a per-subplot list or ``None``) and the
        figure's subplots *plots* (``(df_sub, {level: value})`` each, in drawing order).  A string label stays a
        string when every subplot resolves to the same text, else it becomes one label per subplot."""
        if label is None:
            return None
        if isinstance(label, str):
            filled = [self.subplot_label(label, df_sub, sub) for df_sub, sub in plots]
            if not filled or all(x == filled[0] for x in filled):
                return filled[0] if filled else fill_default(label)
            return filled
        if isinstance(label, list):
            out = fill_default(label)
            for k, (df_sub, sub) in enumerate(plots[:len(label)]):
                out[k] = self.subplot_label(label[k], df_sub, sub)
            return out
        return label


def batch_labels(labeler: "AxisLabeler | None", label, batches) -> list | None:
    """The label of every figure of a plot (one per batch of ``(df_sub, sub)`` subplots), or ``None`` when
    *labeler* is ``None`` (the caller uses :func:`fill_default`)."""
    if labeler is None:
        return None
    return [labeler.batch_label(label, plots) for plots in batches]


#: The y-axis label template of the dispatch plots (flows in, out and stored at a node or nodeGroup).
DISPATCH_YLABEL = "{A}/h"


def dispatch_ylabel(ctx, *, group: str | None = None, node: str | None = None) -> str:
    """The y-axis label of a dispatch plot of the nodeGroup *group* or the node *node* in a scenario of the
    :class:`~flextool.process_outputs._output_meta.UnitContext` *ctx*: ``MWh/h`` when every node is MWh (the
    label of 4.2), else ``<unit>/h`` of the node or the group's members (``m3/h``), or
    :data:`~flextool.units_of_measure.MIXED_UOM` for a group over nodes of different units (the stack sums
    them)."""
    if ctx is None or ctx.is_default:
        return uom.fill(DISPATCH_YLABEL, uom.DEFAULT_A)
    if node is not None:
        nodes: tuple[str, ...] = (node,)
    else:
        nodes = tuple(ctx.group_nodes.get(str(group), ()))
    return uom.fill(DISPATCH_YLABEL, uom.units_of(nodes, ctx.node_units))
