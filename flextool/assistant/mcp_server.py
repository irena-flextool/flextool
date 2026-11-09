"""The connection server for AI agents: the assistant tools, the documentation and the level prompts over stdio.

``flextool-assistant --mcp`` runs :func:`run_stdio` on the protocol streams that ``flextool_assistant_entry`` kept
before anything could write to stdout (stdout itself points at stderr while the server runs).  The server is
mcp 2.x's low-level ``Server`` (P0-D1): the registry owns the tool schemas, and
:func:`flextool.assistant.registry.call_tool` validates the arguments (mcp does not on stdio), runs the tool and
fits its result into compact JSON text.  Every tool is read-only.  Tool calls run in a worker thread that takes the
context's lock, so ``ping`` and ``tools/list`` answer while a slow call runs.

* Tools: every tool, or with ``--tier basic`` only the basic set (any other tool is refused).
* Resources: the Markdown pages of the documentation index as ``flextool-docs:///<page>``.
* Prompts: the level texts (``basic``, ``standard``, ``advanced``; only ``basic`` with ``--tier basic``), each
  with an optional ``question``.

:func:`run_degraded` serves one tool, ``flextool_setup_problem``, when ``import flextool`` stopped (this
processor needs the compatible polars build): the entry module loads this file by path, so it must import only
the standard library at module level and ``mcp`` inside its functions.
"""
from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import quote, unquote

DOCS_SCHEME = "flextool-docs"
SETUP_TOOL = "flextool_setup_problem"
SERVER_NAME = "flextool"
#: The longest text of a protocol error (an unknown document or prompt; the documentation not available).
PROTOCOL_ERROR_MAX = 2_000

logger = logging.getLogger("flextool.assistant.mcp_server")


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flextool")
    except PackageNotFoundError:
        return ""


def _docs_uri(doc: str) -> str:
    return f"{DOCS_SCHEME}:///{quote(doc, safe='/')}"


def _doc_from_uri(uri: str) -> str | None:
    prefix = f"{DOCS_SCHEME}:///"
    return unquote(uri[len(prefix):]) if uri.startswith(prefix) else None


def _new_server(instructions: str, **handlers: Any):
    """A low-level server with no telemetry middleware (no exporter is installed; nothing leaves the machine)."""
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME, version=_version(), instructions=instructions, **handlers)
    server.middleware = []
    return server


def build_server(ctx, *, tier: str | None = None):
    """The server for the context *ctx*; *tier* is the internal level name of ``--tier`` (``None``: no level)."""
    import anyio
    import mcp.types as types
    from mcp.shared.exceptions import MCPError

    from flextool.assistant import prompt, registry
    from flextool.assistant.caps import cap_text
    from flextool.assistant.errors import docs_error, render

    tools = registry.tools_for_tier(tier)
    allowed = {t.name for t in tools}
    levels = ["small"] if tier == "small" else list(prompt.ADDENDA)
    by_prompt_name = {name: level for name, level in prompt.LEVELS.items() if level in levels}

    def _locked_call(name: str, arguments: dict) -> registry.ToolOutcome:
        with ctx.lock:
            try:
                refused = name not in allowed and name in registry.all_tools()
            except Exception as exc:  # noqa: BLE001 — a fault is reported, never a traceback to the agent
                import traceback

                traceback.print_exc(file=sys.stderr)
                return registry.error_outcome(ctx, "fault", type=type(exc).__name__)
            if refused:
                return registry.error_outcome(ctx, "not_in_level", name=name)
            return registry.call_tool(ctx, name, arguments)

    async def _list_tools(_ctx, _params) -> types.ListToolsResult:
        annotations = types.ToolAnnotations(read_only_hint=True, open_world_hint=False)
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, title=t.title, description=t.description, input_schema=t.input_schema,
                       annotations=annotations)
            for t in tools
        ])

    async def _call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        out = await anyio.to_thread.run_sync(_locked_call, params.name, dict(params.arguments or {}))
        return types.CallToolResult(content=[types.TextContent(type="text", text=out.text)], is_error=not out.ok)

    def _protocol_text(text: str) -> str:
        """A protocol error's text (not a tool result): rendered as the tool errors are, cut to
        :data:`PROTOCOL_ERROR_MAX`."""
        return cap_text(text, PROTOCOL_ERROR_MAX)

    def _store():
        from flextool.assistant import docs_store

        try:
            return docs_store.load()
        except docs_store.DocsUnavailable as exc:
            raise MCPError(types.INTERNAL_ERROR, _protocol_text(docs_error(exc).render(ctx.redact))) from None

    async def _list_resources(_ctx, _params) -> types.ListResourcesResult:
        store = await anyio.to_thread.run_sync(_store)
        return types.ListResourcesResult(resources=[
            types.Resource(name=doc, uri=_docs_uri(doc), mime_type="text/markdown") for doc in store.doc_names()
        ])

    async def _read_resource(_ctx, params: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = str(params.uri)
        doc = _doc_from_uri(uri)
        store = await anyio.to_thread.run_sync(_store)
        if doc is None or doc not in store.documents:
            raise MCPError(types.INVALID_PARAMS, _protocol_text(render(
                "unknown_doc", {"doc": doc or uri, "docs": list(store.doc_names())}, ctx.redact)))
        return types.ReadResourceResult(contents=[
            types.TextResourceContents(uri=uri, mime_type="text/markdown", text=store.documents[doc])
        ])

    async def _list_prompts(_ctx, _params) -> types.ListPromptsResult:
        question = types.PromptArgument(name="question", description="The question to answer at this level.",
                                        required=False)
        return types.ListPromptsResult(prompts=[
            types.Prompt(name=name, description=prompt.PROMPT_DESCRIPTIONS[level], arguments=[question])
            for name, level in by_prompt_name.items()
        ])

    async def _get_prompt(_ctx, params: types.GetPromptRequestParams) -> types.GetPromptResult:
        level = by_prompt_name.get(params.name)
        if level is None:
            raise MCPError(types.INVALID_PARAMS, _protocol_text(render(
                "unknown_prompt", {"name": params.name, "names": list(by_prompt_name)}, ctx.redact)))
        question = (params.arguments or {}).get("question")
        text = prompt.level_prompt(level, ctx.redact(str(question)) if question else None)
        return types.GetPromptResult(description=prompt.PROMPT_DESCRIPTIONS[level], messages=[
            types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))
        ])

    return _new_server(
        prompt.instructions(ctx.project_name, ctx.redact),
        on_list_tools=_list_tools, on_call_tool=_call_tool,
        on_list_resources=_list_resources, on_read_resource=_read_resource,
        on_list_prompts=_list_prompts, on_get_prompt=_get_prompt,
    )


def _protocol_streams(proto_in: int, proto_out: int):
    """The protocol's text streams over the descriptors the entry module kept (UTF-8, ``\\n`` line ends)."""
    import io
    import os

    import anyio

    stdin = anyio.wrap_file(io.TextIOWrapper(os.fdopen(proto_in, "rb"), encoding="utf-8", errors="replace",
                                             newline="\n"))
    stdout = anyio.wrap_file(io.TextIOWrapper(os.fdopen(proto_out, "wb"), encoding="utf-8", errors="strict",
                                              newline="\n"))
    return stdin, stdout


def _serve(server, proto_in: int, proto_out: int) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    async def main() -> None:
        stdin, stdout = _protocol_streams(proto_in, proto_out)
        # Explicit streams: mcp leaves fd 0 and fd 1 alone (the entry module already pointed them away).
        async with stdio_server(stdin=stdin, stdout=stdout) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)


def run_stdio(ctx, proto_in: int, proto_out: int, *, tier: str | None = None) -> None:
    """Serve the tools of *ctx* over the protocol descriptors until the agent closes the connection."""
    _serve(build_server(ctx, tier=tier), proto_in, proto_out)


#: Instructions and the one tool of a server whose FlexTool could not load.
DEGRADED_INSTRUCTIONS = (
    "FlexTool could not start on this computer, so its tools are not available. Tell the user what to do; the "
    "fix follows.\n\n{problem}"
)
DEGRADED_DESCRIPTION = (
    "Why FlexTool could not start on this computer, and what the user can do about it. Call it and pass the "
    "answer on to the user."
)


def _home_as_tilde(text: str) -> str:
    """*text* with the home folder written ``~``: the fix keeps a command the user can paste, without the user
    name in it (the full redaction would cut the command's paths to their last part)."""
    from pathlib import Path

    home = str(Path.home())
    return text.replace(home, "~") if len(home) > 1 else text


def run_degraded(problem: str, proto_in: int, proto_out: int) -> None:
    """Serve only :data:`SETUP_TOOL`, which returns *problem* (the reason FlexTool could not load and the fix)."""
    import mcp.types as types

    print(f"FlexTool could not start; the connection serves only {SETUP_TOOL}.\n{problem}", file=sys.stderr)
    problem = _home_as_tilde(problem)

    async def _list_tools(_ctx, _params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[types.Tool(
            name=SETUP_TOOL, title="Why FlexTool could not start", description=DEGRADED_DESCRIPTION,
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            annotations=types.ToolAnnotations(read_only_hint=True, open_world_hint=False))])

    async def _call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name != SETUP_TOOL:
            return types.CallToolResult(content=[types.TextContent(
                type="text", text=f"FlexTool could not start, so only {SETUP_TOOL} is available.")], is_error=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=problem)], is_error=False)

    server = _new_server(DEGRADED_INSTRUCTIONS.format(problem=problem), on_list_tools=_list_tools,
                         on_call_tool=_call_tool)
    _serve(server, proto_in, proto_out)
