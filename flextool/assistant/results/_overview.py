"""The numbers list_runs gives about a run (Cycle 2, agent feedback 2026-11-09 items 1 and 2): its cost totals for a
scan across runs, and a one-call overview of one run (costs by category, capacity per entity, slacks).

**Costs** come from ``costs_discounted_p_``: every cost category the model minimises, discounted and summed over
all periods (the total ``calibrate._readers.read_total_system_cost`` uses; the run record's per-solve objective
is not used, since rolling solves overlap and the record keeps a limited number of solves).  ``total_cost`` adds
every category (``commodity_sales`` is negative), ``penalty_cost`` the categories whose name holds "penalty" (the
slack penalties), and ``largest_cost`` is the category with the largest positive value; the overview's cost split
lists the non-zero categories by signed value, largest first (critique m3).

**Capacity** comes from ``unit_ / connection_ / node_capacity_ed_p``, whose ``existing`` and ``total`` are amounts
per period (a stock): they are never added up over periods (critique M4).  A row is ``[entity, existing in the
first period, invested summed over the periods (filled cells only: an empty cell is a period with no investment
decision), total in the last period]``, with ``divested`` (summed) and ``in_construction`` (last period) only when
a listed entity has one.  No total over entities: units and storages of different carriers do not add up (C1-R1).

**Slacks**: for each penalised slack table (:data:`~flextool.assistant.results._slacks.SLACKS`) with a value of at
least :data:`SLACK_MIN`, its number of entries and its largest one ``[[entity levels], period, value]``; no sums
over nodes, groups or periods.

No number is read from a run whose results may not be used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flextool.assistant.caps import dumps, sent_payload
from flextool.assistant.errors import ToolError
from flextool.assistant.results import _query, _slacks
from flextool.assistant.unit_wording import shown_unit
from flextool.lean_parquet import to_long

COSTS_TABLE = "costs_discounted_p_"
#: The capacity tables of the overview: (key, table, the level that names the entity).
CAPACITY_TABLES = (("unit", "unit_capacity_ed_p", "unit"), ("connection", "connection_capacity_ed_p", "connection"),
                   ("storage", "node_capacity_ed_p", "node"))
SLACK_MIN = 1e-6
#: The columns of a capacity class's rows; a class with divested or in-construction capacity gives its own.
CAPACITY_COLUMNS = ["entity", "existing", "invested", "total"]
COST_UNIT_TEXT = "discounted, all periods ({table}); compare runs over the same periods; penalty_cost: slack penalties"
SCAN_COST_TEXT = "; largest_cost: [category, value] of the largest cost"
SLACK_NOTE = "Each slack's entries: query_result on its table."
NOT_USABLE = "not shown: the results of this run may not be used"
#: The overview's cut, in this order, until it fits the result (critique M5): (cost categories: "all", "1%" (those
#: of at least 1 % of the costs) or a number (the largest of those of at least 1 %, so a step never shows a
#: category an earlier one dropped); entities per capacity class; slacks).  ``None``: the level's cap.
CUT_STEPS: tuple[tuple[Any, int | None, int | None], ...] = (
    ("all", None, None), ("1%", None, None), ("1%", 3, None), ("1%", 3, 3), (3, 3, 3), (3, 2, 3), (3, 1, 3),
    (3, 1, 1), (1, 1, 1), (0, 0, 0),
)


@dataclass(frozen=True)
class Costs:
    unit: str | None
    categories: tuple[tuple[str, float], ...]     # in the table's order

    @property
    def total(self) -> float:
        return sum(v for _, v in self.categories)

    @property
    def penalty(self) -> float:
        return sum(v for c, v in self.categories if "penalty" in c.lower())

    @property
    def largest(self) -> list | None:
        positive = [(c, v) for c, v in self.categories if v > 0]
        if not positive:
            return None
        category, value = max(positive, key=lambda cv: cv[1])
        return [category, _query._number(value)]

    def split(self) -> list[list]:
        """The non-zero categories, largest signed value first (equal values by name)."""
        rows = sorted(((c, v) for c, v in self.categories if v != 0), key=lambda cv: (-cv[1], cv[0]))
        return [[c, _query._number(v)] for c, v in rows]


def read_costs(folder: Path) -> Costs | None:
    """The discounted costs of the run folder *folder*, or ``None`` when it has no readable cost table."""
    if not (folder / f"{COSTS_TABLE}.parquet").is_file():
        return None
    try:
        table = _query.open_table(folder, COSTS_TABLE, where=folder.name)
        long = _long(table)
    except ToolError:
        return None
    category = next((c for c in long.columns if c not in ("period", "value")), None)
    if category is None:
        return None
    sums: dict[str, float] = {}
    for name, value in long.select(category, "value").iter_rows():
        if value is not None:
            sums[str(name)] = sums.get(str(name), 0.0) + float(value)
    unit = shown_unit(table.measures[0].unit) if table.measures else None
    return Costs(unit, tuple(sums.items()))


def _long(table: _query.Table):
    """Every value of *table* as rows ``<row levels>, <column levels>, value`` without the scenario level."""
    wide = _query.read_wide(table, list(table.layout.physical))
    long = to_long(wide, table.layout)
    return long.drop(_query.SCENARIO_LEVEL) if _query.SCENARIO_LEVEL in long.columns else long


def cost_unit_text(unit: str | None, *, scan: bool) -> str:
    text = f"{unit or 'unit not recorded'}, " + COST_UNIT_TEXT.format(table=COSTS_TABLE)
    return text + (SCAN_COST_TEXT if scan else "")


# ---------------------------------------------------------------------------------------------------------
# Capacity


@dataclass(frozen=True)
class CapacityRow:
    entity: str
    existing: float | None
    invested: float | None
    divested: float | None
    in_construction: float | None
    total: float | None

    @property
    def changed(self) -> bool:
        return any(v is not None and v > 0 for v in (self.invested, self.divested, self.in_construction))

    @property
    def empty(self) -> bool:
        return not self.changed and not any(v for v in (self.existing, self.total))


def read_capacity(folder: Path, table_name: str, level: str) -> tuple[dict, list[CapacityRow]] | None:
    """``({table, unit, first, last}, rows)`` of a capacity table, or ``None`` without a readable one."""
    import math

    if not (folder / f"{table_name}.parquet").is_file():
        return None
    try:
        table = _query.open_table(folder, table_name, where=folder.name)
        long = _long(table)
    except ToolError:
        return None
    if not {level, "period", "parameter"} <= set(long.columns):
        return None
    periods = list(dict.fromkeys(long.get_column("period").to_list()))
    order = {p: i for i, p in enumerate(periods)}
    cells: dict[str, dict[str, list[tuple[int, float | None]]]] = {}
    for entity, period, parameter, value in long.select(level, "period", "parameter", "value").iter_rows():
        finite = value if value is not None and math.isfinite(value) else None
        cells.setdefault(str(entity), {}).setdefault(str(parameter), []).append((order[period], finite))
    rows = []
    for entity, by_parameter in cells.items():
        def at(parameter: str, last: bool) -> float | None:
            values = sorted(by_parameter.get(parameter, []))
            return (values[-1][1] if last else values[0][1]) if values else None

        def summed(parameter: str) -> float | None:
            values = [v for _, v in by_parameter.get(parameter, []) if v is not None]
            return sum(values) if values else None

        rows.append(CapacityRow(entity, at("existing", False), summed("invested"), summed("divested"),
                                at("in_construction", True), at("total", True)))
    unit = shown_unit(table.measures[0].unit) if table.measures else None
    return {"table": table_name, "unit": unit, "first": periods[0] if periods else None,
            "last": periods[-1] if periods else None}, rows


def _capacity_part(meta: dict, rows: list[CapacityRow], level: str, cap: int) -> dict:
    """The listed rows of one class: the entities with new, divested or in-construction capacity first, then the
    largest totals, at most *cap*, shown largest total first."""
    def by_total(row: CapacityRow) -> tuple[float, str]:
        return (-(row.total if row.total is not None else float("-inf")), row.entity)

    candidates = [r for r in rows if not r.empty]
    picked = sorted(candidates, key=lambda r: (not r.changed, *by_total(r)))[:cap]
    picked.sort(key=by_total)
    extra = [name for name in ("divested", "in_construction")
             if any(getattr(r, name) for r in picked)]
    columns = [*CAPACITY_COLUMNS, *extra]
    out: dict[str, Any] = {"table": meta["table"], "unit": meta["unit"]}
    if extra:
        out["columns"] = columns
    out["rows"] = [[r.entity, *(_query._number(getattr(r, c)) for c in columns[1:])] for r in picked]
    if len(picked) < len(candidates):
        changed = sum(1 for r in candidates if r.changed)
        out["shown"] = f"{len(picked)} of {len(candidates)}; {changed} changed"
    return out


# ---------------------------------------------------------------------------------------------------------
# Slacks


def read_slacks(folder: Path) -> list[list]:
    """``[slack, table, unit, entries, largest]`` of every slack table with a value of at least
    :data:`SLACK_MIN`; largest is ``[[entity levels], period, value]``."""
    import polars as pl

    out = []
    for spec in _slacks.SLACKS:
        if not (folder / f"{spec.table}.parquet").is_file():
            continue
        try:
            table = _query.open_table(folder, spec.table, where=folder.name)
            if not table.layout.physical:
                continue
            long = _long(table)
        except ToolError:
            continue
        long = long.filter(pl.col("value").abs() >= SLACK_MIN)
        if long.height == 0:
            continue
        levels = [c for c in long.columns if c not in ("period", "value")]
        top = long.sort("value", descending=True, maintain_order=True).row(0, named=True)
        unit = shown_unit(table.measures[0].unit) if table.measures else None
        out.append([spec.slack, spec.table, unit, long.height,
                    [[top[c] for c in levels], top.get("period"), _query._number(top["value"])]])
    return out


# ---------------------------------------------------------------------------------------------------------
# The overview


def overview_numbers(folder: Path, max_entities: int, max_chars: int, head: dict, redact) -> dict:
    """*head* (the run's own fields) with the overview's numbers, cut by :data:`CUT_STEPS` until its sent text
    fits *max_chars*."""
    costs = read_costs(folder)
    capacity = []
    for key, table_name, level in CAPACITY_TABLES:
        read = read_capacity(folder, table_name, level)
        if read is not None:
            capacity.append((key, level, *read))
    slacks = read_slacks(folder)

    def build(cost_mode: Any, entities: int | None, n_slacks: int | None) -> dict:
        out = dict(head)
        notes = list(out.pop("notes", []))
        if costs is None:
            out["total_cost"] = out["penalty_cost"] = None
            notes.append(f"This run has no cost table ({COSTS_TABLE}), so its costs are not shown.")
        else:
            out["total_cost"] = _query._number(costs.total)
            out["penalty_cost"] = _query._number(costs.penalty)
            out["cost_unit"] = cost_unit_text(costs.unit, scan=False)
            rows = costs.split()
            n = len(rows)
            if cost_mode != "all":
                # A number of categories cuts the 1 % list, so no later step brings back a category under 1 %
                # (Cycle 2 review m2).
                scale = sum(abs(v) for _, v in costs.categories)
                rows = [r for r in rows if abs(r[1]) >= 0.01 * scale]
                if cost_mode != "1%":
                    rows = rows[:cost_mode]
            split: dict[str, Any] = {"columns": ["category", "value"], "rows": rows}
            if len(rows) < n:
                split["shown"] = f"{len(rows)} of {n}; every one: query_result on {COSTS_TABLE}"
            out["cost_split"] = split
        cap_entities = max_entities if entities is None else min(entities, max_entities)
        parts: dict[str, Any] = {}
        spans = {(meta["first"], meta["last"]) for _, _, meta, _ in capacity}
        for key, level, meta, rows in capacity:
            part = _capacity_part(meta, rows, level, cap_entities)
            if len(spans) > 1:
                part["periods"] = [meta["first"], meta["last"]]
            parts[key] = part
        if parts:
            out["capacity"] = {"columns": CAPACITY_COLUMNS, **parts}
            if len(spans) == 1:
                first, last = next(iter(spans))
                when = f"existing: {first}; total, in_construction: {last}; "
            else:
                when = "existing: the first period, total, in_construction: the last (periods: each class); "
            out["capacity_periods"] = (when + "invested, divested: summed over periods; changed: invested, "
                                       "divested or in construction above 0; never summed over entities")
        shown_slacks = slacks if n_slacks is None else slacks[:n_slacks]
        block: dict[str, Any] = {"columns": ["slack", "table", "unit", "entries", "largest"], "rows": shown_slacks}
        if len(shown_slacks) < len(slacks):
            block["shown"] = f"{len(shown_slacks)} of {len(slacks)} slacks"
        out["slacks"] = block
        if slacks:
            notes.append(SLACK_NOTE)
        if notes:
            out["notes"] = notes
        return out

    last: dict = {}
    for cost_mode, entities, n_slacks in CUT_STEPS:
        last = sent_payload(build(cost_mode, entities, n_slacks), redact)
        if len(dumps(last)) <= max_chars:
            return last
    return last
