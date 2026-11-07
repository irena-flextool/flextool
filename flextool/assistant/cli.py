"""``flextool-assistant``: the read-only FlexTool tools for AI assistants, from the command line.

* ``--mcp`` serves the tools to an AI agent (Claude Code and others) over stdin and stdout
  (:mod:`flextool.assistant.mcp_server`); the agent starts this command itself.
* ``--call TOOL [JSON | @file.json]`` runs one tool and prints its result (for support and testing).

Both first delete the saved chat conversations older than the expiry in the assistant settings
(:func:`flextool.assistant.store.purge_at_start`).  Both need the project: ``--project-path`` (an absolute
folder) or ``--project NAME`` (``projects/NAME`` under the current folder, as the FlexTool window keeps them).
``--threads N`` sets FlexTool's own threads in this process; the default is 1, so the tools never compete with a
running model (``flextool/_thread_budget.py`` reads the flag before polars is imported; this parser only checks
it).

The command starts from ``flextool_assistant_entry``, which keeps stdout clean for the protocol before this module
is imported.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

from flextool.assistant.caps import DEFAULT_PRESET, PRESETS
from flextool.assistant.prompt import LEVELS

#: Said in ``--help`` (plain English, no policy links; D10, D11, Q5).
DATA_NOTICE = (
    "Data use: everything the tools return (input data, results, run logs, documentation) is sent to the AI "
    "service that your agent uses. FlexTool cannot say how that service stores or uses it; check its policies "
    "yourself. The tools only read; they cannot change the project."
)

NO_FRONT_END = ("The chat in the terminal comes in a later FlexTool release. To use these tools from an AI agent, "
                "start with --mcp (see --help).")
MCP_MISSING = ("The connection for AI assistants could not load the package mcp: {error}. Reinstall FlexTool with "
               "Update FlexTool.")


def _threads(text: str) -> int:
    from flextool.run_options import OPTIONS

    try:
        return OPTIONS["threads"].parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flextool-assistant",
        description="Read-only FlexTool tools for AI assistants: input data, results, run logs and the "
                    "documentation of one project.",
        epilog=DATA_NOTICE + " Example, for Claude Code: claude mcp add flextool -- flextool-assistant --mcp "
                             "--project-path /absolute/path/to/project",
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mcp", action="store_true",
                      help="serve the tools to an AI agent over stdin and stdout (the agent starts this command)")
    mode.add_argument("--call", nargs="+", metavar=("TOOL", "ARGUMENTS"),
                      help="run one tool and print its result; ARGUMENTS is a JSON object or @file.json")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--project-path", metavar="FOLDER", help="the project folder (an absolute path)")
    where.add_argument("--project", metavar="NAME", help="the project projects/NAME under the current folder")
    parser.add_argument("--input-source", metavar="SOURCE",
                        help="the input source the tools use when a request names none (file name or number)")
    parser.add_argument("--scenario", help="the scenario the tools use when a request names none")
    parser.add_argument("--result-cap", choices=list(PRESETS), default=None,
                        help=f"how much one tool result may hold (default {DEFAULT_PRESET}; small with --tier "
                             "basic, large with --tier advanced)")
    parser.add_argument("--tier", choices=list(LEVELS), default=None,
                        help="the assistant level: basic offers only the basic tools and smaller results")
    parser.add_argument("--threads", type=_threads, default=1, metavar="N",
                        help="threads for FlexTool's own processing in this process (default 1)")
    parser.add_argument("--verbose", action="store_true", help="more messages on stderr")
    return parser


def _project_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.project_path is None and args.project is None:
        parser.error("give the project with --project-path FOLDER or --project NAME")
    if args.project_path is not None:
        path = Path(args.project_path)
        if not path.is_absolute():
            parser.error(f"--project-path must be an absolute path (the agent may start this command in any "
                         f"folder): {args.project_path}")
    else:
        from flextool.gui.project_utils import get_projects_dir

        path = (get_projects_dir() / args.project).absolute()
        print(f"Project folder: {path}", file=sys.stderr)
    if not path.is_dir():
        parser.error(f"the project folder does not exist: {path}")
    return path


def _call_arguments(raw: str | None, parser: argparse.ArgumentParser) -> dict:
    if raw is None:
        return {}
    text = raw
    if raw.startswith("@"):
        try:
            text = Path(raw[1:]).read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"cannot read {raw[1:]}: {exc.strerror}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        parser.error(f"the tool arguments are not valid JSON: {exc}")
    if not isinstance(value, dict):
        parser.error("the tool arguments must be a JSON object")
    return value


def main(argv: list[str] | None = None, *, proto_in: int | None = None, proto_out: int | None = None) -> int:
    """Run ``flextool-assistant``; *proto_in* / *proto_out* are the protocol descriptors of ``--mcp``."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.mcp and not args.call:
        print(NO_FRONT_END, file=sys.stderr)
        return 2
    if args.call and len(args.call) > 2:
        parser.error("--call takes a tool name and at most one JSON object")
    project = _project_path(args, parser)
    tier = LEVELS[args.tier] if args.tier else None
    result_cap = args.result_cap or tier or DEFAULT_PRESET

    from flextool.assistant.context import make_context
    from flextool.assistant.store import purge_at_start

    if args.mcp:
        try:
            from flextool.assistant import mcp_server
            import mcp.server.lowlevel  # noqa: F401  (a missing or broken mcp is reported here, not mid-session)
        except ImportError as exc:
            print(MCP_MISSING.format(error=exc), file=sys.stderr)
            return 2
        if proto_in is None or proto_out is None:
            parser.error("--mcp starts from the flextool-assistant command")
        ctx = make_context(project, input_source=args.input_source, scenario=args.scenario,
                           result_cap=result_cap, front_end="agent")
        # Old conversations are deleted while the connection starts (never in its way; stdlib only).
        threading.Thread(target=purge_at_start, name="flextool-assistant-purge", daemon=True).start()
        mcp_server.run_stdio(ctx, proto_in, proto_out, tier=tier)
        return 0

    from flextool.assistant import registry
    from flextool.assistant.errors import message

    name, raw = args.call[0], (args.call[1] if len(args.call) > 1 else None)
    arguments = _call_arguments(raw, parser)
    purge_at_start()
    ctx = make_context(project, input_source=args.input_source, scenario=args.scenario, result_cap=result_cap,
                       front_end="terminal")
    allowed = {t.name for t in registry.tools_for_tier(tier)}
    if name in registry.all_tools() and name not in allowed:
        outcome = registry.ToolOutcome(False, None, message("not_in_level", name=name))
    else:
        outcome = registry.call_tool(ctx, name, arguments)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(outcome.text)
    return 0 if outcome.ok else 1
