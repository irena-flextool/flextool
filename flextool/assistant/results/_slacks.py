"""Where a run used its slacks: the penalised slack variables (``vq_*``, ``docs/dev/slack_convention.md``) as
the result tables write them, with the inputs that decide them and, for a node, the units and connections that
feed it.

Each slack has its period table (written in absolute units, the row scaling undone): the node balance slacks
``vq_state_up`` / ``vq_state_down`` (``node_slack_up_d_e`` / ``node_slack_down_d_e``), the group slacks of the
non-synchronous limit, the inertia limit, the reserve requirement and the capacity margin.  Their costs are the
penalty categories of ``annualized_costs_d_p``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flextool.assistant import project
from flextool.assistant.caps import Rendered
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, quoted
from flextool.assistant.results import _query, _runs
from flextool.assistant.unit_wording import shown_unit
from flextool.lean_parquet import to_long

COSTS_TABLE = "annualized_costs_d_p"
DOCS = [{"doc": "results.md", "section": "Slack and penalty values"}, {"doc": "dev/slack_convention.md"}]
_FEEDING_CLASSES = ("unit", "connection")


@dataclass(frozen=True)
class Slack:
    slack: str
    table: str
    meaning: str
    related_inputs: tuple[str, ...]
    node_level: str | None = None          # the column level that names a node (for feeding_entities)


SLACKS: tuple[Slack, ...] = (
    Slack("vq_state_up", "node_slack_up_d_e",
          "Upward slack of the node's energy balance: demand the node could not meet (loss of load), annualized.",
          ("node.penalty_up", "node.inflow", "the capacities of the units and connections feeding the node "
           "(unit.existing, connection.existing and what investments allow)"), "node"),
    Slack("vq_state_down", "node_slack_down_d_e",
          "Downward slack of the node's energy balance: energy the node had to get rid of (excess load), "
          "annualized.",
          ("node.penalty_down", "node.inflow", "the capacities of the units and connections feeding the node "
           "(unit.existing, connection.existing and what investments allow)"), "node"),
    Slack("vq_non_synchronous", "nodeGroup_slack_nonsync_d_g",
          "Non-synchronous generation above the group's limit, annualized.",
          ("group.non_synchronous_limit", "group.penalty_non_synchronous")),
    Slack("vq_inertia", "nodeGroup_slack_inertia_d_g",
          "Inertia the group lacked against its requirement, annualized.",
          ("group.inertia_limit", "group.penalty_inertia")),
    Slack("vq_reserve", "nodeGroup_slack_reserve_d_eeg",
          "Reserve requirement the group did not meet, annualized.",
          ("reserve__upDown__group.reservation", "reserve__upDown__group.penalty_reserve")),
    Slack("vq_capacity_margin", "nodeGroup_slack_capacity_margin_d_g",
          "Capacity the group lacked against its capacity margin in the period (not annualized).",
          ("group.capacity_margin", "group.penalty_capacity_margin")),
)


def _long(table: _query.Table):
    """Every value of a period table as rows ``period, <column levels>, value`` (these tables are small: one
    row per period)."""
    wide = _query.read_wide(table, list(table.layout.physical))
    long = to_long(wide, table.layout)
    return long.drop(_query.SCENARIO_LEVEL) if _query.SCENARIO_LEVEL in long.columns else long


def _feeding(ctx: AssistantContext, rr: _runs.ResolvedRun, nodes: list[str]) -> tuple[dict | None, str | None]:
    """``{node: [{entity_class, entity}]}``: the units and connections that have the node as a dimension and are
    active in the run's scenario of its input source."""
    run = rr.run
    scenario = rr.entry.get("db_scenario")
    if run is None or run.source is None or not scenario:
        return None, "The input source of this run is not known, so its units and connections are not listed."
    try:
        db, _ = project.open_source(ctx, run.source)
        if scenario not in db.scenarios:
            return None, Rendered(f"The input source no longer has the scenario {quoted(scenario, ctx.redact)} "
                                  "of this run.")
        view = db.view(scenario)
    except ToolError as exc:
        return None, Rendered(f"The input data of this run could not be read: {exc.render(ctx.redact)}")
    out: dict[str, list[dict]] = {}
    for node in nodes:
        found: set[tuple[str, tuple[str, ...]]] = set()
        for e in db.relationships_of("node", node):
            dims = db.classes[e.entity_class].dimensions
            if dims and dims[0] in _FEEDING_CLASSES and view.active(e.entity_class, e.byname)[0]:
                found.add((e.entity_class, tuple(e.byname)))
        out[node] = [{"entity_class": cls, "entity": list(byname)} for cls, byname in sorted(found)]
    note = None
    if rr.entry.get("stale_vs_input"):
        note = "The input data changed after this run; the units and connections are those of the current data."
    return out, note


def slack_summary(ctx: AssistantContext, run: str, *, min_value: float, max_rows: int) -> dict:
    import polars as pl

    rr = _runs.resolve_run(ctx, run)
    slacks: list[dict] = []
    notes: list[str] = []
    for spec in SLACKS:
        if not (rr.folder / f"{spec.table}.parquet").is_file():
            continue
        table = _query.open_table(rr.folder, spec.table, where=rr.folder.name)
        if not table.layout.physical:
            continue
        long = _long(table).filter(pl.col("value").abs() >= min_value)
        if long.height == 0:
            continue
        entity_levels = [c for c in long.columns if c not in ("period", "value")]
        long = long.sort("value", descending=True, maintain_order=True)
        entries = [{"entity": [row[c] for c in entity_levels], "period": row.get("period"),
                    "value": _query._number(row["value"])} for row in long.head(max_rows).iter_rows(named=True)]
        totals = (long.group_by("period", maintain_order=True).agg(pl.col("value").sum()).sort("period")
                  if "period" in long.columns else None)
        record: dict[str, Any] = {
            "slack": spec.slack,
            "table": spec.table,
            "meaning": spec.meaning,
            "unit": shown_unit(table.measures[0].unit) if table.measures else None,
            "total": ({str(p): _query._number(v) for p, v in totals.iter_rows()} if totals is not None else None),
            "entries": entries,
            "entries_shown": f"{len(entries)} of {long.height}",
            "related_inputs": list(spec.related_inputs),
            "feeding_entities": None,
            "docs": DOCS,
        }
        if spec.node_level is not None and spec.node_level in entity_levels:
            nodes = list(dict.fromkeys(e["entity"][entity_levels.index(spec.node_level)] for e in entries))
            feeding, note = _feeding(ctx, rr, nodes)
            record["feeding_entities"] = feeding
            if note and note not in notes:
                notes.append(note)
        slacks.append(record)

    penalty_costs = None
    if (rr.folder / f"{COSTS_TABLE}.parquet").is_file():
        costs = _query.open_table(rr.folder, COSTS_TABLE, where=rr.folder.name)
        long = _long(costs)
        category = next((c for c in long.columns if c not in ("period", "value")), None)
        if category is not None:
            long = long.filter(pl.col(category).str.contains("penalty") & (pl.col("value").abs() >= min_value))
            penalty_costs = {
                "table": COSTS_TABLE,
                "unit": shown_unit(costs.measures[0].unit) if costs.measures else None,
                "entries": [{"category": row[category], "period": row.get("period"),
                             "value": _query._number(row["value"])} for row in long.iter_rows(named=True)],
            }
    none_found = not slacks and not (penalty_costs and penalty_costs["entries"])
    notes.append(f"Values below {min_value:g} are left out. A slack is used when meeting the constraint costs "
                 "more than its penalty, or is impossible with the capacities the model has.")
    return {"run": rr.folder.name, "slacks": slacks, "penalty_costs": penalty_costs, "none_found": none_found,
            "notes": notes, "run_notes": rr.run_notes}
