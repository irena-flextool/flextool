"""Discovery tools: the project's input sources, and its scenario runs with whether their results may be read."""
from __future__ import annotations

from flextool.assistant.context import AssistantContext
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.results import _runs


def list_sources(ctx: AssistantContext, args: dict) -> Fit:
    rows = _runs.list_sources(ctx)
    return Fit({"sources": rows, "total": len(rows)}, rows_key="sources", what="sources",
               hint="name one input source in the other tools")


def list_runs(ctx: AssistantContext, args: dict) -> Fit:
    payload = _runs.list_runs(ctx, source=args.get("source"), scenario=args.get("scenario"))
    return Fit(payload, rows_key="runs", what="runs", hint="name a source or a scenario")


TOOLS = [
    make_tool(
        name="list_sources", title="Input sources of the project",
        description=("The input sources of the project (spreadsheets and databases) with their number (the "
                     "suffix of their result folders), scenarios, database version and whether it is current, the "
                     "state of a spreadsheet's conversion, and notices of FlexTool behaviour changes."),
        properties={}, handler=list_sources, area="discovery", small_tier=True),
    make_tool(
        name="list_runs", title="Scenario runs of the project",
        description=("The scenario result folders of the project (for example S2_Dry_1: the scenario name and the "
                     "input source number) with the state of their latest run, its verdict and times, whether the "
                     "input changed since, and whether its results may be read (results_usable; unusable_code "
                     "says why not). Results are read only from a run that succeeded. source and scenario narrow "
                     "the list."),
        properties={"source": {"type": "string"}, "scenario": {"type": "string"}},
        handler=list_runs, area="discovery", small_tier=True),
]
