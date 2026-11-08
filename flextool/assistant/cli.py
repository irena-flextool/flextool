"""``flextool-assistant``: FlexTool's AI assistant in the terminal, and the read-only FlexTool tools for AI agents.

* Without ``--mcp`` or ``--call``: the chat with an AI model in the terminal (an interactive chat, or one question
  with ``-q``), and the small commands ``--list-models``, ``--set-key``, ``--forget-key`` and ``--usage``
  (:mod:`flextool.assistant.chat_cli`, imported only for them: ``--mcp`` and ``--call`` load no AI service
  library).
* ``--mcp`` serves the tools to an AI agent (Claude Code and others) over stdin and stdout
  (:mod:`flextool.assistant.mcp_server`); the agent starts this command itself.
* ``--call TOOL [JSON | @file.json]`` runs one tool and prints its result (for support and testing).

The chat, ``--mcp`` and ``--call`` first delete the saved chat conversations older than the expiry in the assistant
settings (:func:`flextool.assistant.store.purge_at_start`).  They need the project: ``--project-path`` (an absolute
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
from flextool.assistant.providers import PRESETS as PROVIDERS

#: Said in ``--help`` (plain English, no policy links; D10, D11, Q5).
DATA_NOTICE = (
    "Data use: your questions and everything the tools return (input data, results, run logs, documentation) are "
    "sent to the AI service that the chat or your agent uses. FlexTool cannot say how that service stores or uses "
    "it; check its policies yourself. The tools only read; they cannot change the project."
)
EXAMPLES = (
    "Chat in the terminal: flextool-assistant --project-path /absolute/path/to/project. One question: add -q "
    "\"your question\". For Claude Code: claude mcp add flextool -- flextool-assistant --mcp --project-path "
    "/absolute/path/to/project"
)
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
        description="FlexTool's AI assistant in the terminal, and read-only FlexTool tools for AI agents: input "
                    "data, results, run logs and the documentation of one project.",
        epilog=DATA_NOTICE + " " + EXAMPLES,
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mcp", action="store_true",
                      help="serve the tools to an AI agent over stdin and stdout (the agent starts this command)")
    mode.add_argument("--call", nargs="+", metavar=("TOOL", "ARGUMENTS"),
                      help="run one tool and print its result; ARGUMENTS is a JSON object or @file.json")
    mode.add_argument("--list-models", action="store_true", help="list the models of the AI service")
    mode.add_argument("--set-key", action="store_true",
                      help="store the key of the AI service in the system keyring (asked in the terminal)")
    mode.add_argument("--forget-key", action="store_true", help="remove the stored key of the AI service")
    mode.add_argument("--usage", action="store_true",
                      help="the tokens used, by month, AI service and model, as each service reported them")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--project-path", metavar="FOLDER", help="the project folder (an absolute path)")
    where.add_argument("--project", metavar="NAME", help="the project projects/NAME under the current folder")
    parser.add_argument("--input-source", metavar="SOURCE",
                        help="the input source the tools use when a request names none (file name or number)")
    parser.add_argument("--scenario", help="the scenario the tools use when a request names none")
    parser.add_argument("--provider", choices=list(PROVIDERS), metavar="ID",
                        help="the AI service of the chat, for this session: " + ", ".join(PROVIDERS))
    parser.add_argument("--model", metavar="NAME", help="the model of the chat, for this session")
    parser.add_argument("--base-url", metavar="URL",
                        help="the address of the AI service, for a custom endpoint or a model server on another "
                             "port or computer")
    parser.add_argument("--result-cap", choices=list(PRESETS), default=None,
                        help=f"how much one tool result may hold (default {DEFAULT_PRESET}; small with --tier "
                             "basic, large with --tier advanced)")
    parser.add_argument("--tier", choices=list(LEVELS), default=None,
                        help="the assistant level. With --mcp: basic offers only the basic tools and smaller "
                             "results; standard and advanced only set the result size. In the chat: a level the "
                             "model allows (its automatic level or lower; FlexTool picks the automatic one from "
                             "the model name and its context size when this is not given)")
    parser.add_argument("--free-tier", choices=("on", "off", "auto"), default=None,
                        help="the free plan of the chat: fewer steps for each question, to stay within a free "
                             "plan's request limits (auto: on only for an OpenRouter model whose name ends :free; "
                             "default: the saved setting, else auto)")
    parser.add_argument("-q", "--question", metavar="QUESTION",
                        help="ask one question, print the answer and stop; nothing is asked in the terminal")
    parser.add_argument("--accept-data-notice", action="store_true",
                        help="accept the data notice of this AI service for good: it is not asked again until the "
                             "service, its address, the proxy or the notice changes (the notice is printed on "
                             "stderr)")
    parser.add_argument("--continue", dest="resume", nargs="?", const="", default=None, metavar="ID",
                        help="go on with the newest saved conversation of the project, or the one with this id")
    parser.add_argument("--all", dest="all_models", action="store_true",
                        help="with --list-models: every model of the list, also the older versions (by default "
                             "only the latest version of each model of a remote service)")
    parser.add_argument("--key-file", action="store_true",
                        help="with --set-key: store the key unencrypted in a file of your user folder instead of "
                             "the system keyring")
    parser.add_argument("--no-color", action="store_true", help="no colours in the terminal")
    parser.add_argument("--threads", type=_threads, default=1, metavar="N",
                        help="threads for FlexTool's own processing in this process (default 1)")
    parser.add_argument("--verbose", action="store_true", help="more messages on stderr")
    return parser


def project_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
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
        from flextool.assistant import chat_cli

        return chat_cli.run(args, parser)
    if args.call and len(args.call) > 2:
        parser.error("--call takes a tool name and at most one JSON object")
    project = project_path(args, parser)
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
                       front_end="agent")
    allowed = {t.name for t in registry.tools_for_tier(tier)}
    if name in registry.all_tools() and name not in allowed:
        outcome = registry.ToolOutcome(False, None, message("not_in_level", name=name))
    else:
        outcome = registry.call_tool(ctx, name, arguments)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(outcome.text)
    return 0 if outcome.ok else 1
