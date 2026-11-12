"""explain_entity's readers: one unit, connection or node of one run, its inputs beside its results, and the
hints the two give together (item 209).

**Inputs** are those of the run's scenario in the run's own input source
(:func:`~flextool.assistant.results._runs.run_inputs`): each parameter the scenario sets as get_entity lists it
(``[parameter, value, unit, from_alternative]``, a short map as its table, a longer one summarised), with
``invest_method`` and ``existing`` always shown (``"default"`` / ``"not set"`` when no alternative sets them), the
parameters of its flow and profile links (:data:`LINK_CLASSES`) and the names of its other memberships (groups,
reserves, flowGroups).  A node lists only its own links (profiles, groups), not the units and connections that
feed it.

**Results** are read per period, each block naming its table and unit: capacity (existing, invested, divested,
in construction, total, never summed over periods), the flows per connected node (never summed over nodes; the
``_d_`` flow tables are annualized), capacity factor, curtailment (VRE units), storage state and prices at the
entity's nodes (mean, minimum and maximum over the period's time steps, unweighted), the non-zero discounted cost
categories (with their totals over the periods) and the investment value.  A table the run lacks, or that has no
values of the entity, is named in ``not_in_run``.

**Hints** are deterministic and worded "consistent with": they compare a result with the input that may explain
it (:func:`hints`), and name both.  The result is cut by :data:`CUT_STEPS` until it fits the call's characters;
the last step always fits: the header, notes, capacity and hints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flextool.assistant.caps import dumps, sent_payload
from flextool.assistant.errors import ToolError
from flextool.assistant.results import _query
from flextool.assistant.unit_wording import shown_unit
from flextool.lean_parquet import to_long

CLASSES = ("unit", "connection", "node")
#: The link classes whose parameters are inputs of the entity (the others are listed by name only).
LINK_CLASSES = ("unit__inputNode", "unit__outputNode", "unit__node__profile", "connection__node__node",
                "node__profile", "connection__profile")
#: The links of a node that are its own (a node's flow links are the inputs of its units and connections).
NODE_LINK_PREFIXES = ("node__", "group__node")
#: Parameters shown also when no alternative sets them: (parameter, what the row says without a default).
ALWAYS_SHOWN = ("invest_method", "existing")
NOT_SET = "not set"
DEFAULT = "default"
#: The parameters kept by the cut step "investment-relevant inputs only".
INVESTMENT_PREFIXES = ("invest", "retire", "existing", "lifetime", "cumulative", "salvage", "fixed_cost",
                       "discount_rate", "unitsize", "virtual_unitsize", "construction")
#: |value| below this is zero (slack and investment-value readings).
ZERO = 1e-6
#: Relative tolerance of a hint's equality (a result equal to its limit).
REL_TOL = 1e-6

#: invest_method values: which allow no investment, and which use each limit (reference/unit.md, invest_method).
NO_INVEST = frozenset({"not_allowed", "retire_no_limit", "retire_period", "retire_total", "retire_period_total"})
RETIRE_ONLY = frozenset({"retire_no_limit", "retire_period", "retire_total", "retire_period_total"})
INVEST_TOTAL = frozenset({"invest_total", "invest_period_total", "invest_retire_total", "invest_retire_period_total"})
INVEST_PERIOD = frozenset({"invest_period", "invest_period_total", "invest_retire_period",
                           "invest_retire_period_total"})
RETIRE_TOTAL = frozenset({"retire_total", "retire_period_total", "invest_retire_total", "invest_retire_period_total"})
RETIRE_PERIOD = frozenset({"retire_period", "retire_period_total", "invest_retire_period",
                           "invest_retire_period_total"})
CUMULATIVE = frozenset({"cumulative_limits"})

#: The texts explain_entity adds (hints and notes), checked by the plain-English test.
TEXTS: dict[str, str] = {
    "hint_not_allowed": "Nothing was invested: invest_method is {method}, which allows no investment.",
    "hint_retire_only": "Nothing was invested: invest_method is {method}, which allows retirement only.",
    "hint_total": ("Invested {invested} {unit} over the periods, equal to invest_max_total ({limit}): consistent "
                   "with the total investment limit binding."),
    "hint_period": ("In {period} invested {invested} {unit}, equal to invest_max_period ({limit}): consistent "
                    "with the period investment limit binding."),
    "hint_period_zero": "In {period} invest_max_period is 0: investment is not allowed in that period.",
    "hint_cumulative": ("In {period} the total capacity is {total} {unit}, equal to cumulative_max_capacity "
                        "({limit}): consistent with the cumulative capacity limit binding."),
    "hint_min_total": ("Invested {invested} {unit} over the periods, equal to invest_min_total ({limit}): "
                       "consistent with the minimum investment forcing it."),
    "hint_min_period": ("In {period} invested {invested} {unit}, equal to invest_min_period ({limit}): consistent "
                        "with the minimum investment forcing it."),
    "hint_retire_total": ("Retired {divested} {unit} over the periods, equal to retire_max_total ({limit}): "
                          "consistent with the total retirement limit binding."),
    "hint_retire_period": ("In {period} retired {divested} {unit}, equal to retire_max_period ({limit}): "
                           "consistent with the period retirement limit binding."),
    "hint_value_positive": ("The investment value is positive in {periods} ({table}): consistent with an upper "
                            "limit holding the investment back (the entity's own, a group's{groups} or a "
                            "cumulative one)."),
    "hint_value_negative": ("The investment value is negative in {periods} ({table}): consistent with more "
                            "capacity not paying off (or a minimum investment forcing more than pays off)."),
    "costs_note": ("Discounted costs, total over the periods and per period; the annualized costs per period are "
                   "in {table}."),
    "prices_note": ("Mean, minimum and maximum of the price over each period's time steps (every solve), "
                    "unweighted over time steps."),
    "state_note": "Mean, minimum and maximum of the storage state over each period's time steps, unweighted.",
    "annualized_note": "The flows and curtailment per period are annualized (scaled to a full year).",
    "periods_cut": "Only the first and last periods are shown; query_result on each table gives the others.",
    "rows_cut": "{shown} of {total} rows; query_result on {table} gives them all.",
    "inputs_cut": ("Only the investment-relevant inputs are shown; get_entity or get_parameter gives the "
                   "others."),
    "final_cut": ("The result was shortened to fit: {blocks} left out. Ask with a larger max_chars; get_entity "
                  "gives the inputs and query_result each table."),
    "no_values": "not_in_run: tables of this run without values of {entity}.",
    "several_runs": ("The scenario also has the result folders {folders}; this is the newest. Name one in run to "
                     "explain the entity there."),
}


# ---------------------------------------------------------------------------------------------------------
# Result tables


@dataclass(frozen=True)
class Block:
    """One result block: *key* in the payload, *table*, the *level* that names the entity, the level kept beside
    the period (*by*: node, sink), and how it is read: ``"series"`` (a value per period), ``"capacity"``,
    ``"costs"`` or ``"steps"`` (mean, minimum, maximum of the time steps per period).  *optional*: not named in
    ``not_in_run`` when the run has no values of the entity (curtailment of a unit that is not VRE)."""

    key: str
    table: str
    level: str
    kind: str = "series"
    by: str | None = None
    optional: bool = False


BLOCKS: dict[str, tuple[Block, ...]] = {
    "unit": (
        Block("capacity", "unit_capacity_ed_p", "unit", "capacity"),
        Block("output_flows", "unit_outputNode_d_ee", "unit", by="node"),
        Block("input_flows", "unit_inputNode_d_ee", "unit", by="node", optional=True),
        Block("capacity_factor", "unit_outputs_cf_d_ee", "unit", by="sink"),
        Block("curtailment", "unit_curtailment_outputNode_d_ee", "unit", by="sink", optional=True),
        Block("costs", "cost_unit_discounted_d_ec", "unit", "costs"),
        Block("investment_value", "dual_invest_effective_unit_d_e", "entity", optional=True),
    ),
    "connection": (
        Block("capacity", "connection_capacity_ed_p", "connection", "capacity"),
        Block("net_flow", "connection_d_eee", "process"),
        Block("flow_to_first_node", "connection_leftward_d_eee", "process", by="node"),
        Block("flow_to_second_node", "connection_rightward_d_eee", "process", by="node"),
        Block("losses", "connection_losses_d_eee", "process", optional=True),
        Block("capacity_factor", "connection_cf_d_e", "connection"),
        Block("costs", "cost_connection_discounted_d_ec", "connection", "costs"),
        Block("investment_value", "dual_invest_effective_connection_d_e", "entity", optional=True),
    ),
    "node": (
        Block("slack_up", "node_slack_up_d_e", "node"),
        Block("slack_down", "node_slack_down_d_e", "node"),
        Block("costs", "cost_node_discounted_d_ec", "node", "costs"),
        Block("capacity", "node_capacity_ed_p", "node", "capacity", optional=True),
        Block("storage_state", "node_state_dt_e", "node", "steps", optional=True),
        Block("investment_value", "dual_invest_effective_node_d_e", "entity", optional=True),
    ),
}
PRICES_TABLE = "node_prices_dt_e"
ANNUALIZED_COSTS = {"unit": "cost_unit_annualized_d_ec", "connection": "cost_connection_annualized_d_ec",
                    "node": "cost_node_annualized_d_ec"}
#: The tables with time steps a further question about the entity reads (listed when the run has them).
RELATED_TABLES = {
    "unit": ("unit_outputNode_dt_ee", "unit_inputNode_dt_ee", "unit_online_dt_e",
             "unit_curtailment_outputNode_dt_ee", "cost_unit_annualized_d_ec"),
    "connection": ("connection_dt_eee", "connection_leftward_dt_eee", "connection_rightward_dt_eee",
                   "connection_losses_dt_eee", "cost_connection_annualized_d_ec"),
    "node": ("node_d_ep", "node_dt_ep", "node_prices_dt_e", "node_slack_up_dt_e", "node_slack_down_dt_e",
             "cost_node_annualized_d_ec"),
}
CAPACITY_PARTS = ("existing", "invested", "divested", "in_construction", "total")
#: The blocks whose values add up to "annualized" (the annualized_note).
ANNUALIZED_BLOCKS = frozenset({"output_flows", "input_flows", "curtailment", "net_flow", "flow_to_first_node",
                               "flow_to_second_node", "losses", "slack_up", "slack_down"})


def _finite(value: Any) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _open(folder: Path, stem: str) -> _query.Table | None:
    if not (folder / f"{stem}.parquet").is_file():
        return None
    try:
        return _query.open_table(folder, stem, where=folder.name)
    except ToolError:
        return None


def _entity_columns(table: _query.Table, level: str, name: str) -> list[tuple[str, dict[str, str]]]:
    """``(physical column, {column level: value})`` of the columns whose *level* is *name*."""
    names = list(table.layout.col)
    if level not in names:
        return []
    at = names.index(level)
    out = []
    for physical, key in zip(table.layout.physical, table.layout.columns):
        if str(key[at]) == name:
            out.append((physical, {n: str(v) for n, v in zip(names, key)}))
    return out


def _period_rows(table: _query.Table, columns: list[str]) -> tuple[list[str], dict[str, list[float | None]]]:
    """``(periods in order, {physical column: value per period})`` of a table with ``period`` as its row."""
    wide = _query.read_wide(table, columns)
    periods = [str(p) for p in wide.get_column("period").to_list()]
    return periods, {c: [_finite(v) for v in wide.get_column(c).to_list()] for c in columns}


def _unit_of(table: _query.Table, physical: str) -> str | None:
    unit = table.measure_of()[physical].unit
    return shown_unit(unit) if unit else None


def _with_units(rows: list[list], units: list[str | None]) -> tuple[Any, list[list]]:
    """The block's ``unit`` and its rows: one unit for the block, or each row's unit appended (a block of several
    units: nodes with their own unit_of_measure)."""
    distinct = set(units)
    if len(distinct) <= 1:
        return (distinct.pop() if distinct else None), rows
    return None, [[*row, unit] for row, unit in zip(rows, units)]


def read_series(folder: Path, block: Block, name: str) -> dict | None:
    """A value per period (and per *block.by*) of the entity, or ``None`` when the run has none."""
    table = _open(folder, block.table)
    if table is None or "period" not in table.row_levels:
        return None
    found = _entity_columns(table, block.level, name)
    if not found:
        return None
    periods, values = _period_rows(table, [p for p, _ in found])
    rows, units = [], []
    for i, period in enumerate(periods):
        for physical, labels in found:
            value = values[physical][i]
            if value is None:
                continue
            rows.append([period, labels[block.by], _query._number(value)] if block.by else
                        [period, _query._number(value)])
            units.append(_unit_of(table, physical))
    if not rows:
        return None
    columns = ["period", "node" if block.by else None, "value"]
    columns = [c for c in columns if c]
    unit, rows = _with_units(rows, units)
    if unit is None and any(u for u in units):
        columns.append("unit")
    return {"table": block.table, "unit": unit, "columns": columns, "rows": rows}


def _capacity_unit(table: _query.Table, name: str, physical: list[str]) -> str | None:
    for p in physical:
        unit = table.row_unit(name, p)
        if unit:
            return shown_unit(unit)
    return shown_unit(table.measures[0].unit) if table.measures and table.measures[0].unit else None


def read_capacity(folder: Path, block: Block, name: str) -> dict | None:
    """The entity's capacity per period: ``[period, existing, invested, divested, in_construction, total]`` (the
    parts the table has; an empty cell is a period with no such decision)."""
    import polars as pl

    table = _open(folder, block.table)
    if table is None or block.level not in table.row_levels:
        return None
    labels = {p: str(key[-1]) for p, key in zip(table.layout.physical, table.layout.columns)}
    parts = [part for part in CAPACITY_PARTS if part in labels.values()]
    physical = {labels[p]: p for p in table.layout.physical if labels[p] in parts}
    wide = _query.read_wide(table, list(physical.values())).filter(pl.col(block.level) == name)
    if wide.height == 0:
        return None
    rows = []
    for row in wide.iter_rows(named=True):
        rows.append([row["period"], *(_query._number(_finite(row[physical[part]])) for part in parts)])
    return {"table": block.table, "unit": _capacity_unit(table, name, list(physical.values())),
            "columns": ["period", *parts], "rows": rows}


def read_costs(folder: Path, block: Block, name: str) -> dict | None:
    """The entity's non-zero discounted cost categories: ``[category, total, <one column per period>]``, the total
    over the periods first (a discounted cost adds up over periods), largest total first."""
    import polars as pl

    table = _open(folder, block.table)
    if table is None or block.level not in table.row_levels:
        return None
    wide = _query.read_wide(table, list(table.layout.physical)).filter(pl.col(block.level) == name)
    if wide.height == 0:
        return None
    long = to_long(wide, table.layout)
    if _query.SCENARIO_LEVEL in long.columns:
        long = long.drop(_query.SCENARIO_LEVEL)
    category = next((c for c in long.columns if c not in ("period", block.level, "value")), None)
    if category is None:
        return None
    periods = list(dict.fromkeys(str(p) for p in long.get_column("period").to_list()))
    cells: dict[str, dict[str, float]] = {}
    for cat, period, value in long.select(category, "period", "value").iter_rows():
        if value is not None and math.isfinite(value):
            cells.setdefault(str(cat), {})[str(period)] = float(value)
    rows = []
    for cat, by_period in cells.items():
        if not any(abs(v) > 0 for v in by_period.values()):
            continue
        rows.append([cat, sum(by_period.values()), *(by_period.get(p) for p in periods)])
    rows.sort(key=lambda r: (-r[1], r[0]))
    rows = [[r[0], *(_query._number(v) for v in r[1:])] for r in rows]
    unit = shown_unit(table.measures[0].unit) if table.measures and table.measures[0].unit else None
    return {"table": block.table, "unit": unit, "columns": ["category", "total", *periods], "rows": rows}


def _step_stats(table: _query.Table, columns: list[str]) -> dict[str, list[list]]:
    """``{physical column: [[period, mean, min, max], …]}`` over each period's time steps (every solve), empty
    cells left out."""
    import polars as pl

    wide = _query.read_wide(table, columns)
    out: dict[str, list[list]] = {}
    for c in columns:
        stats = (wide.filter(pl.col(c).is_not_null())
                 .group_by("period", maintain_order=True)
                 .agg(pl.col(c).mean().alias("mean"), pl.col(c).min().alias("min"), pl.col(c).max().alias("max")))
        out[c] = [[str(p), _query._number(a), _query._number(b), _query._number(m)] for p, a, b, m in stats.iter_rows()]
    return out


def read_steps(folder: Path, block: Block, name: str) -> dict | None:
    """The entity's mean, minimum and maximum over each period's time steps (storage state)."""
    table = _open(folder, block.table)
    if table is None or "period" not in table.row_levels:
        return None
    found = _entity_columns(table, block.level, name)
    if not found:
        return None
    physical = found[0][0]
    rows = _step_stats(table, [physical])[physical]
    if not rows:
        return None
    return {"table": block.table, "unit": _unit_of(table, physical), "columns": ["period", "mean", "min", "max"],
            "rows": rows, "note": TEXTS["state_note"]}


def read_prices(folder: Path, nodes: list[str]) -> dict | None:
    """The prices at *nodes*, per node and period: ``[node, period, mean, min, max]``."""
    table = _open(folder, PRICES_TABLE)
    if table is None or "period" not in table.row_levels or not nodes:
        return None
    found = [(node, p) for node in nodes for p, _ in _entity_columns(table, "node", node)[:1]]
    if not found:
        return None
    stats = _step_stats(table, [p for _, p in found])
    rows, units = [], []
    for node, physical in found:
        for period, mean, low, high in stats[physical]:
            rows.append([node, period, mean, low, high])
            units.append(_unit_of(table, physical))
    if not rows:
        return None
    columns = ["node", "period", "mean", "min", "max"]
    unit, rows = _with_units(rows, units)
    if unit is None and any(units):
        columns.append("unit")
    return {"table": PRICES_TABLE, "unit": unit, "columns": columns, "rows": rows, "note": TEXTS["prices_note"]}


READERS = {"series": read_series, "capacity": read_capacity, "costs": read_costs, "steps": read_steps}


def classes_in_results(folder: Path, name: str) -> list[str]:
    """The classes (unit, connection, node) whose result tables of the run have *name*: the entity's class when
    the input data cannot be read."""
    import polars as pl

    out = []
    for cls in CLASSES:
        for block in BLOCKS[cls][:2]:
            table = _open(folder, block.table)
            if table is None:
                continue
            if block.level in table.row_levels:
                if _query.read_wide(table, []).filter(pl.col(block.level) == name).height:
                    out.append(cls)
                    break
            elif _entity_columns(table, block.level, name):
                out.append(cls)
                break
    return out


def read_results(folder: Path, entity_class: str, name: str) -> tuple[dict[str, dict], list[str]]:
    """``({block key: block}, tables without values of the entity)`` of the entity's class."""
    blocks: dict[str, dict] = {}
    missing: list[str] = []
    for block in BLOCKS[entity_class]:
        read = READERS[block.kind](folder, block, name)
        if read is None:
            if not block.optional:
                missing.append(block.table)
            continue
        blocks[block.key] = read
    return blocks, missing


def related_tables(folder: Path, entity_class: str) -> list[str]:
    return [t for t in RELATED_TABLES[entity_class] if (folder / f"{t}.parquet").is_file()]


# ---------------------------------------------------------------------------------------------------------
# Hints


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1.0)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _finite(value)


def _shown_number(value: float) -> str:
    return f"{_query._number(value):g}"


def _by_period(value: Any, periods: list[str]) -> dict[str, float]:
    """A period limit as ``{period: value}``: a constant for every period, a one-level map by its rows."""
    from flextool.assistant.inputs import _values

    constant = _num(value)
    if constant is not None:
        return {p: constant for p in periods}
    if not _values.is_indexed(value):
        return {}
    out: dict[str, float] = {}
    for row in _values.rows_of(value):
        if len(row) == 2 and _num(row[1]) is not None:
            out[str(row[0])] = float(row[1])
    return out


def _capacity_cells(capacity: dict | None) -> tuple[list[str], dict[str, dict[str, float | None]]]:
    if capacity is None:
        return [], {}
    columns = capacity["columns"]
    cells = {row[0]: dict(zip(columns[1:], row[1:])) for row in capacity["rows"]}
    return list(cells), cells


def _summed(cells: dict[str, dict[str, float | None]], part: str) -> float | None:
    values = [c.get(part) for c in cells.values() if c.get(part) is not None]
    return sum(values) if values else None


def hints(inputs: dict[str, Any], blocks: dict[str, dict], groups: list[str]) -> list[str]:
    """The hints of the entity's inputs (*inputs*: the value each hint parameter has in the scenario, its default
    for invest_method) and its results (*blocks*); *groups*: the groups it is a member of."""
    out: list[str] = []
    capacity = blocks.get("capacity")
    periods, cells = _capacity_cells(capacity)
    unit = (capacity or {}).get("unit") or ""
    method = inputs.get("invest_method")
    method = method if isinstance(method, str) else None
    invested = _summed(cells, "invested")
    divested = _summed(cells, "divested")
    constructing = any((c.get("in_construction") or 0) > ZERO for c in cells.values())
    if method in NO_INVEST and capacity is not None and not (invested and invested > ZERO):
        key = "hint_retire_only" if method in RETIRE_ONLY else "hint_not_allowed"
        out.append(TEXTS[key].format(method=method))

    def total_hint(key: str, part_sum: float | None, limit_name: str, applies: bool, word: str) -> None:
        limit = _num(inputs.get(limit_name))
        if (applies and limit is not None and part_sum is not None and part_sum > ZERO
                and _close(part_sum, limit) and not constructing):
            out.append(TEXTS[key].format(**{word: _shown_number(part_sum)}, unit=unit,
                                         limit=_shown_number(limit)))

    def period_hints(key: str, part: str, limit_name: str, applies: bool, word: str, zero: bool) -> None:
        if not applies:
            return
        limits = _by_period(inputs.get(limit_name), periods)
        for period in periods:
            limit = limits.get(period)
            value = cells[period].get(part)
            if limit is None:
                continue
            if zero and limit == 0:
                out.append(TEXTS["hint_period_zero"].format(period=period))
            elif value is not None and value > ZERO and _close(value, limit):
                out.append(TEXTS[key].format(period=period, **{word: _shown_number(value)}, unit=unit,
                                             limit=_shown_number(limit)))

    total_hint("hint_total", invested, "invest_max_total", method in INVEST_TOTAL, "invested")
    period_hints("hint_period", "invested", "invest_max_period", method in INVEST_PERIOD, "invested", zero=True)
    if method in CUMULATIVE:
        limits = _by_period(inputs.get("cumulative_max_capacity"), periods)
        for period in periods:
            total, limit = cells[period].get("total"), limits.get(period)
            if total is not None and limit is not None and limit > ZERO and _close(total, limit):
                out.append(TEXTS["hint_cumulative"].format(period=period, total=_shown_number(total), unit=unit,
                                                           limit=_shown_number(limit)))
    # A minimum applies wherever the method allows investing (critique 9: forced builds).
    investing = method is not None and method not in NO_INVEST
    total_hint("hint_min_total", invested, "invest_min_total", investing, "invested")
    period_hints("hint_min_period", "invested", "invest_min_period", investing, "invested", zero=False)
    total_hint("hint_retire_total", divested, "retire_max_total", method in RETIRE_TOTAL, "divested")
    period_hints("hint_retire_period", "divested", "retire_max_period", method in RETIRE_PERIOD, "divested",
                 zero=False)

    value = blocks.get("investment_value")
    if value is not None:
        positive = [r[0] for r in value["rows"] if r[1] is not None and r[1] > ZERO]
        negative = [r[0] for r in value["rows"] if r[1] is not None and r[1] < -ZERO]
        if positive:
            named = f" ({', '.join(groups)})" if groups else ""
            out.append(TEXTS["hint_value_positive"].format(periods=", ".join(positive), table=value["table"],
                                                           groups=named))
        if negative:
            out.append(TEXTS["hint_value_negative"].format(periods=", ".join(negative), table=value["table"]))
    return out


# ---------------------------------------------------------------------------------------------------------
# Cutting

#: The cut, in this order, until the result fits (each step keeps the earlier ones' cuts): drop related_tables;
#: costs as totals only (no columns per period); cost categories under 1 % of the entity's costs dropped; maps
#: summarised; investment-relevant inputs only; the first and last periods only; then the final step that always
#: fits (header, notes, capacity, hints).
CUT_STEPS = ("full", "related", "cost_totals", "cost_1%", "maps", "inputs", "periods", "final")


def _cost_totals(costs: dict) -> dict:
    out = dict(costs)
    out["columns"] = costs["columns"][:2]
    out["rows"] = [row[:2] for row in costs["rows"]]
    return out


def _cost_share(costs: dict) -> dict:
    scale = sum(abs(r[1] or 0) for r in costs["rows"])
    kept = [r for r in costs["rows"] if abs(r[1] or 0) >= 0.01 * scale]
    out = dict(costs)
    if len(kept) < len(costs["rows"]):
        out["rows"] = kept
        out["shown"] = f"{len(kept)} of {len(costs['rows'])} categories (those of at least 1 % of the costs)"
    return out


def _first_last(block: dict) -> dict:
    """*block* with the rows (a cost block: the columns) of its first and last periods only."""
    columns = block.get("columns") or []
    if "period" not in columns:
        if columns[:2] == ["category", "total"] and len(columns) > 4:
            return {**block, "columns": [*columns[:3], columns[-1]],
                    "rows": [[*r[:3], r[-1]] for r in block["rows"]]}
        return block
    at = columns.index("period")
    periods = list(dict.fromkeys(r[at] for r in block["rows"]))
    if len(periods) <= 2:
        return block
    keep = {periods[0], periods[-1]}
    return {**block, "rows": [r for r in block["rows"] if r[at] in keep]}


def _cap_rows(block: dict, max_rows: int) -> dict:
    rows = block.get("rows") or []
    if len(rows) <= max_rows:
        return block
    return {**block, "rows": rows[:max_rows],
            "shown": TEXTS["rows_cut"].format(shown=max_rows, total=len(rows), table=block["table"])}


def fitted(build, max_chars: int, redact) -> dict:
    """The payload of the first :data:`CUT_STEPS` step whose sent text fits *max_chars* (``build(step)``); the
    final step's when none does."""
    last: dict = {}
    for step in CUT_STEPS:
        last = sent_payload(build(step), redact)
        if len(dumps(last)) <= max_chars:
            return last
    return last


def steps_from(step: str) -> set[str]:
    """The cuts in force at *step* (it and every earlier one)."""
    return set(CUT_STEPS[1: CUT_STEPS.index(step) + 1])


def cut_blocks(blocks: dict[str, dict], cuts: set[str], max_rows: int) -> dict[str, dict]:
    out = {}
    for key, block in blocks.items():
        if key == "costs":
            if "cost_totals" in cuts:
                block = _cost_totals(block)
            if "cost_1%" in cuts:
                block = _cost_share(block)
        if "periods" in cuts:
            block = _first_last(block)
        out[key] = _cap_rows(block, max_rows)
    return out
