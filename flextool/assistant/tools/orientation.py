"""Orientation tools: what the connection is set to, and the FlexTool documentation (search, read, equations)."""
from __future__ import annotations

import re

from flextool.assistant.caps import CapArg
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, close_names, name_list, similar_text
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
        raise ToolError("docs_unavailable", text=str(exc)) from None


def get_gui_context(ctx: AssistantContext, args: dict) -> dict:
    return {
        "front_end": ctx.front_end,
        "project": ctx.project_name,
        "input_source": ctx.default_source,
        "scenario": ctx.default_scenario,
        "gui": "not running",
        "note": ("The input source and scenario are the defaults this connection was started with; the tools use "
                 "them when a request names none. What is selected in the FlexTool window is not shared."),
    }


def search_docs(ctx: AssistantContext, args: dict) -> Fit:
    query = str(args["query"])[:QUERY_MAX]
    kind = args.get("kind", "all")
    hits = _store().search(query, kind=kind, max_results=args["max_results"])
    return Fit({"query": query, "kind": kind, "results": hits}, rows_key="results", what="results",
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
    raise ToolError("unknown_doc", doc=doc, docs=", ".join(names))


def read_doc(ctx: AssistantContext, args: dict) -> Fit | dict:
    from flextool.docs_build.index import headings

    store = _store()
    doc = _doc_name(store, str(args["doc"]))
    markdown = store.documents[doc]
    hs = headings(markdown)
    section = args.get("section")
    if not section:
        outline = [{"level": h.level, "title": h.title, "anchor": h.anchor} for h in hs]
        return Fit({"doc": doc, "outline": outline, "note": "Ask for a section by its title or anchor."},
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
    return payload


def get_formulation(ctx: AssistantContext, args: dict) -> dict:
    store = _store()
    name = str(args["name"]).strip()
    record = store.formulation(name)
    if record is None:
        hits = store.search(name, kind="formulations", max_results=3)
        names = [h["name"] for h in hits]
        raise ToolError("unknown_formulation", name=name,
                        similar=f" Similar names: {', '.join(names)}." if names else
                        similar_text(name, [r["name"] for r in store.formulations]))
    stem = _stem(record["name"])
    related = [r["name"] for r in store.formulations
               if r is not record and r["module"] == record["module"] and _stem(r["name"]) == stem][:RELATED_MAX]
    return {
        "name": record["name"],
        "names": record["names"],
        "kind": record["kind"],
        "module": record["module"],
        "formula": record["formula"],
        "notes": record["notes"],
        "source": f"{record['file']}:{record['line']}",
        "function": record["function"],
        "related": related,
    }


_KINDS = ["docs", "formulations", "all"]

TOOLS = [
    make_tool(
        name="get_gui_context", title="Connection context",
        description="The project this connection reads, and the input source and scenario it uses by default.",
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
        description=("Read a document of the FlexTool documentation, for example reference.md. Without section it "
                     "returns the outline (headings and anchors); with section (a heading or an anchor) that section "
                     "with its subsections. max_chars limits the text."),
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
