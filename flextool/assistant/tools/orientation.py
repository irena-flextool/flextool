"""Orientation tools: what the connection is set to, and the FlexTool documentation (search, read, equations)."""
from __future__ import annotations

import re

from flextool.assistant.caps import CapArg, Rendered, clean
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import NameList, ToolError, close_names, docs_error, name_list, similar_text
from flextool.assistant.registry import Fit, make_tool

QUERY_MAX = 200
RELATED_MAX = 10


def _stem(name: str) -> str:
    """The family part of a header name: ``maxOnline_{linear,integer}`` → ``maxOnline``."""
    return re.split(r"[_{\[( ]", name, maxsplit=1)[0]


def _store():
    from flextool.assistant import docs_store

    try:
        return docs_store.load()
    except docs_store.DocsUnavailable as exc:
        raise docs_error(exc) from None


NOT_SHARED = ("The input source and scenario are the defaults this connection was started with; the tools use "
              "them when a request names none. What is selected in the FlexTool window is not shared.")
WINDOW_CLOSED = "The FlexTool window is not open for this project; showing the startup options."
WINDOW_BUSY = ("The FlexTool window is open for this project but has not answered for over two minutes (it may be "
               "busy with a long operation); showing the startup options.")
FROM_WINDOW = ("What is selected in the FlexTool window for this project and shown in its result viewer, as of "
               "written_at. The tools still use the input source and scenario this connection was started with "
               "when a request names none.")


def _startup_options(ctx: AssistantContext, note: str, *, gui: str = "not running") -> dict:
    out = {
        "front_end": ctx.front_end,
        "project": ctx.project_name,
        "input_source": ctx.default_source,
        "scenario": ctx.default_scenario,
        "gui": gui,
        "note": note,
    }
    if gui == "running":
        out["responding"] = False
    return out


def _model_facts(ctx: AssistantContext, snapshot) -> str | None:
    """The model facts of *snapshot*'s scenario as text (already redacted), ``None`` without them: the
    snapshot's own (the chat filled them in when the question was sent), else computed here, in this process."""
    from flextool.assistant import model_facts

    facts = snapshot.scenario_facts if snapshot is not None else None
    if facts is None:
        facts = model_facts.facts_for(ctx, snapshot, None)
    return None if facts is None else Rendered(model_facts.render(facts, ctx.redact, header=False))


def get_gui_context(ctx: AssistantContext, args: dict) -> dict:
    """In the FlexTool window: the snapshot taken when the question was sent.  For an agent (the connection
    server, ``--call``): what the window publishes in its runtime context file, else the startup options.  In the
    terminal: the startup options.  Each answer carries ``model_facts``: the model facts of the scenario it names
    (:mod:`flextool.assistant.model_facts`; ``None`` without a scenario), computed in this process."""
    from flextool.assistant.context import ContextSnapshot, snapshot_payload

    if ctx.front_end == "gui":
        snapshot = ctx.snapshot()
        return {**snapshot_payload(snapshot, gui_running="running", redact=ctx.redact),
                "model_facts": _model_facts(ctx, snapshot)}
    defaults = ContextSnapshot(project=ctx.project_name, input_source=ctx.default_source,
                               scenario=ctx.default_scenario, front_end=ctx.front_end)
    if ctx.front_end == "agent":
        from flextool.assistant.runtime_context import read_runtime_context

        context, running = read_runtime_context(ctx.project_path)
        if context is None and running == "running":
            # A live window whose file went stale: busy (or hung), not closed (review G18 critique m4).
            return {**_startup_options(ctx, WINDOW_BUSY, gui="running"),
                    "model_facts": _model_facts(ctx, defaults)}
        if context is None or running != "running":
            return {**_startup_options(ctx, WINDOW_CLOSED), "model_facts": _model_facts(ctx, defaults)}
        source, scenario = context.get("input_source"), context.get("scenario")
        shown = ContextSnapshot(project=ctx.project_name, input_source=source if isinstance(source, str) else None,
                                scenario=scenario if isinstance(scenario, str) else None, front_end="agent")
        return {**context, "front_end": ctx.front_end, "note": FROM_WINDOW,
                "model_facts": _model_facts(ctx, shown)}
    return {**_startup_options(ctx, NOT_SHARED), "model_facts": _model_facts(ctx, defaults)}


def _shipped(value):
    """The texts of FlexTool's own documentation, cleaned but not redacted (they hold no data of the user's;
    redaction would cut their relative links and ``sqlite:///`` examples to a last part)."""
    if isinstance(value, str):
        return value if isinstance(value, Rendered) else Rendered(clean(value))
    if isinstance(value, dict):
        return {k: _shipped(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shipped(v) for v in value]
    return value


def search_docs(ctx: AssistantContext, args: dict) -> Fit:
    query = str(args["query"])[:QUERY_MAX]
    kind = args.get("kind", "all")
    hits = _store().search(query, kind=kind, max_results=args["max_results"])
    return Fit({"query": query, "kind": kind, "results": _shipped(hits)}, rows_key="results", what="results",
               hint="ask for fewer results")


def _doc_name(store, doc: str) -> str:
    names = store.doc_names()
    text = doc.strip().lstrip("/")
    for candidate in (text, f"{text}.md"):
        if candidate in names:
            return candidate
    by_base = [n for n in names if n.rsplit("/", 1)[-1] in (text, f"{text}.md")]
    if len(by_base) == 1:
        return by_base[0]
    raise ToolError("unknown_doc", doc=doc, docs=list(names))


def read_doc(ctx: AssistantContext, args: dict) -> Fit | dict:
    from flextool.docs_build.index import headings

    store = _store()
    doc = _doc_name(store, str(args["doc"]))
    markdown = store.documents[doc]
    hs = headings(markdown)
    section = args.get("section")
    if not section:
        outline = [{"level": h.level, "title": h.title, "anchor": h.anchor} for h in hs]
        return Fit({"doc": doc, "outline": _shipped(outline), "note": "Ask for a section by its title or anchor."},
                   rows_key="outline", what="headings", hint="ask for a section to read it")
    wanted = str(section).strip().lstrip("#").lower()
    hit = next((k for k, h in enumerate(hs) if wanted in (h.title.lower(), h.anchor.lower())), None)
    if hit is None:
        titles = [h.title for h in hs]
        shown = close_names(str(section), titles) or titles
        raise ToolError("unknown_section", doc=doc, section=section, sections=name_list(shown, 30))
    lines = markdown.splitlines()
    head = hs[hit]
    end = next((h.line for h in hs[hit + 1:] if h.level <= head.level), len(lines))
    text = "\n".join(lines[head.line:end]).strip("\n")
    limit = args["max_chars"]
    budget = max(limit - 400, 200)       # room for the other fields of the result
    payload: dict = {"doc": doc, "section": head.title, "anchor": head.anchor}
    if len(text) > budget:
        cut = text[:budget]
        if "\n" in cut[budget // 2:]:
            cut = cut[: cut.rindex("\n")]
        payload["text"] = cut
        subsections = [h.title for h in hs[hit + 1:] if h.line < end]
        payload["truncated"] = (f"showing the first {len(cut)} of {len(text)} characters — ask for a subsection"
                                + (f" ({name_list(subsections, 10)})" if subsections else "")
                                + " or a larger max_chars")
    else:
        payload["text"] = text
    return _shipped(payload)


def get_formulation(ctx: AssistantContext, args: dict) -> dict:
    store = _store()
    name = str(args["name"]).strip()
    record = store.formulation(name)
    if record is None:
        hits = store.search(name, kind="formulations", max_results=3)
        names = [h["name"] for h in hits]
        raise ToolError("unknown_formulation", name=name,
                        similar=NameList(tuple(names), form="similar") if names else
                        similar_text(name, [r["name"] for r in store.formulations]))
    stem = _stem(record["name"])
    related = [r["name"] for r in store.formulations
               if r is not record and r["module"] == record["module"] and _stem(r["name"]) == stem][:RELATED_MAX]
    return _shipped({
        "name": record["name"],
        "names": record["names"],
        "kind": record["kind"],
        "module": record["module"],
        "formula": record["formula"],
        "notes": record["notes"],
        "source": f"{record['file']}:{record['line']}",
        "function": record["function"],
        "related": related,
    })


_KINDS = ["docs", "formulations", "all"]

TOOLS = [
    make_tool(
        name="get_gui_context", title="Connection context",
        description=("The project this connection reads, the input source and scenario it uses by default, and what "
                     "is selected in the FlexTool window and shown in its result viewer when the window is open "
                     "for the project (input sources, scenarios, result folders with whether their results may "
                     "be used, the plot and time range shown)."),
        properties={}, handler=get_gui_context, area="orientation", small_tier=False),
    make_tool(
        name="search_docs", title="Search the FlexTool documentation",
        description=("Keyword search over the FlexTool user documentation, the developer pages that ship with it, "
                     "and the model's equations (kind: docs, formulations or all; default all). Returns the best "
                     "sections with a snippet, or equation names. query is at most 200 characters; max_results is 1 "
                     "to 10 (default 5)."),
        properties={"query": {"type": "string"}, "kind": {"type": "string", "enum": _KINDS},
                    "max_results": {"type": "integer"}},
        required=("query",), handler=search_docs, area="orientation", small_tier=True,
        caps={"max_results": CapArg("max_results", maximum=10)}),
    make_tool(
        name="read_doc", title="Read a FlexTool document",
        description=("Read a document of the FlexTool documentation, for example reference/units.md. Without "
                     "section it returns the outline (headings and anchors); with section (a heading or an anchor) "
                     "that section with its subsections. max_chars limits the text."),
        properties={"doc": {"type": "string"}, "section": {"type": "string"}, "max_chars": {"type": "integer"}},
        required=("doc",), handler=read_doc, area="orientation", small_tier=True,
        caps={"max_chars": CapArg("max_chars")}),
    make_tool(
        name="get_formulation", title="An equation of the model",
        description=("The formula, notes and source line of one equation, variable or objective term of the "
                     "FlexTool model, by its name (for example nodeBalance_eq or maxOnline_linear)."),
        properties={"name": {"type": "string"}}, required=("name",), handler=get_formulation,
        area="orientation", small_tier=False),
]
