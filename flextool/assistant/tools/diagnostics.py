"""Diagnostic tools: a run's record and log (also for a run that failed, to help get it to run), and where a run
that succeeded used its slacks."""
from __future__ import annotations

from typing import Any

from flextool import run_record
from flextool.assistant.caps import LOG_LINE_MAX, CapArg, Rendered, cap_text, dumps, sent_text
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, name_list
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.results import _runs, _slacks
from flextool.assistant.unit_wording import UNITS_CONVENTION, add_session_notes

SOLVES_MAX = 50
WARNINGS_MAX = 20
GREP_MATCHES_MAX = 50
GREP_CONTEXT = 2
MESSAGE_MAX = 500
_ACCEPTED = ("accepted",)


def _option_source(key: str, entry: dict) -> str:
    from flextool.run_options import OPTIONS

    spec = OPTIONS.get(key)
    source = entry.get("source")
    if source == "env":
        return f"set by the environment variable {entry.get('env') or (spec.env if spec else key)}"
    if source == "db":
        return f"from the database parameter {spec.db}" if spec is not None and spec.db else "from the database"
    if source == "cli":
        return f"set by the run flag {spec.flag}" if spec is not None and spec.flag else "set by a run flag"
    return "default"


def _options(status: dict) -> Any:
    resolved = status.get("options_resolved")
    if isinstance(resolved, dict) and resolved:
        return [{"option": key, "value": entry.get("value"), "set_by": _option_source(key, entry)}
                for key, entry in resolved.items() if isinstance(entry, dict)]
    return list(status.get("options") or [])


def _solve(entry: dict) -> dict:
    keys = ("name", "parent", "kind", "state", "model_status", "acceptance", "message", "objective", "wall_s",
            "warm_used", "benders", "log_anchor")
    out = {k: entry.get(k) for k in keys}
    out["message"] = cap_text(out["message"], MESSAGE_MAX)
    return out


def _status_block(status: dict) -> dict:
    solves = list(status.get("solves") or [])
    ordered = sorted(solves, key=lambda s: 0 if s.get("state") not in _ACCEPTED else 1)     # stable: failed first
    shown = [_solve(s) for s in ordered[:SOLVES_MAX]]
    omitted = int(status.get("solves_omitted") or 0) + max(len(solves) - SOLVES_MAX, 0)
    warnings = list(status.get("warnings") or [])
    error = status.get("error")
    if isinstance(error, dict):
        error = {"type": error.get("type"), "message": cap_text(error.get("message"), MESSAGE_MAX)}
    out: dict[str, Any] = {
        "state": run_record.effective_state(status),
        "verdict": status.get("verdict"),
        "error": error,
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "wall_time_s": status.get("wall_time_s"),
        "solver": status.get("solver"),
        "launcher": status.get("launcher"),
        "log_capture": status.get("log_capture"),
        "threads": status.get("threads"),
        "solves": shown,
        "warnings": [{"level": w.get("level"), "logger": w.get("logger"),
                      "message": cap_text(w.get("message"), MESSAGE_MAX)} for w in warnings[:WARNINGS_MAX]],
        "options": _options(status),
    }
    if omitted:
        out["solves_omitted"] = omitted
    warnings_omitted = int(status.get("warnings_omitted") or 0) + max(len(warnings) - WARNINGS_MAX, 0)
    if warnings_omitted:
        out["warnings_omitted"] = warnings_omitted
    return out


def _log_lines(ctx: AssistantContext, folder) -> list[tuple[int, str]]:
    """``(line number, text)`` of the run log, without the run record's marker lines; each line redacted, then cut
    to :data:`LOG_LINE_MAX` (a cut first could leave part of a folder name)."""
    path = folder / run_record.RUN_LOG_FILE
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [(i, Rendered(cap_text(sent_text(line, ctx.redact), LOG_LINE_MAX)))
            for i, line in enumerate(text.splitlines(), start=1) if not line.startswith(run_record.MARKER)]


def _fit_lines(ctx: AssistantContext, lines: list[str], budget: int, *, from_end: bool) -> list[str]:
    """The lines that fit into *budget* characters of JSON, taken from the start (or the end)."""
    out: list[str] = []
    used = 0
    for line in (reversed(lines) if from_end else lines):
        size = len(dumps(sent_text(line, ctx.redact))) + 1
        if used + size > budget:
            break
        out.append(line)
        used += size
    return list(reversed(out)) if from_end else out


def get_run_log(ctx: AssistantContext, args: dict) -> dict:
    run = _runs.find_run(ctx, str(args["run"]))
    status = run.status
    if status is None:
        raise ToolError("no_run_record")
    payload: dict[str, Any] = {"run": run.folder.name, "status": _status_block(status)}
    lines = _log_lines(ctx, run.folder)
    total = len(lines)
    tail = int(args["tail_lines"])
    grep, solve = args.get("grep"), args.get("solve")
    notes: list[str] = []
    if solve is not None:
        entry = next((s for s in status.get("solves") or [] if s.get("name") == solve), None)
        if entry is None:
            names = [str(s.get("name")) for s in status.get("solves") or []]
            raise ToolError("unknown_solve", run=run.folder.name, solve=solve, solves=name_list(names))
        anchor = entry.get("log_anchor")
        start = next((k for k, (_, text) in enumerate(lines) if anchor and anchor in text), None)
        if start is None:
            notes.append("The start of this solve is no longer in the log (a long log keeps its beginning and its "
                         "end); showing the end of the log.")
            picked, mode = lines[-tail:], "tail"
        else:
            picked, mode = lines[start:start + tail], "solve"
    elif grep:
        needle = str(grep).lower()
        hits = [k for k, (_, text) in enumerate(lines) if needle in text.lower()]
        if len(hits) > GREP_MATCHES_MAX:
            notes.append(f"showing the first {GREP_MATCHES_MAX} of {len(hits)} matching lines")
            hits = hits[:GREP_MATCHES_MAX]
        keep: list[int] = sorted({j for k in hits for j in range(max(k - GREP_CONTEXT, 0),
                                                                min(k + GREP_CONTEXT + 1, total))})
        picked, mode = [lines[j] for j in keep], "grep"
        if not hits:
            notes.append(f"No log line contains '{grep}'.")
    else:
        picked, mode = lines[-tail:], "tail"
    texts = [Rendered(f"{n}: {text}") for n, text in picked]       # redacted already (_log_lines)
    log: dict[str, Any] = {"lines": [], "from_line": picked[0][0] if picked else None, "total_lines": total,
                           "omitted_note": None}
    payload["log"] = log
    payload["notes"] = notes
    budget = int(args["max_chars"]) - len(dumps(_redacted(ctx, payload))) - 200
    fitted = _fit_lines(ctx, texts, max(budget, 0), from_end=(mode == "tail"))
    log["lines"] = fitted
    if fitted:
        log["from_line"] = int(fitted[0].split(":", 1)[0])
    if len(fitted) < len(texts):
        where = "last" if mode == "tail" else "first"
        log["omitted_note"] = (f"showing the {where} {len(fitted)} of {len(texts)} selected lines — use grep, solve "
                               "or a smaller tail_lines")
    if not lines:
        log["omitted_note"] = "The run log is empty or missing."
    return payload


def _redacted(ctx: AssistantContext, value: Any) -> Any:
    if isinstance(value, str):
        return sent_text(value, ctx.redact)
    if isinstance(value, dict):
        return {k: _redacted(ctx, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(ctx, v) for v in value]
    return value


def get_slack_summary(ctx: AssistantContext, args: dict) -> Fit:
    min_value = float(args.get("min_value", 1e-6))
    payload = _slacks.slack_summary(ctx, str(args["run"]), min_value=min_value, max_rows=int(args["max_rows"]))
    return Fit(add_session_notes(payload, [UNITS_CONVENTION], ctx.notes_sent, ctx.notes_in_flight), rows_key="slacks", what="slacks",
               hint="lower max_rows or raise min_value")


_RUN_TEXT = "run is a result folder name from list_runs (for example S2_Dry_1)."

TOOLS = [
    make_tool(
        name="get_run_log", title="Run record and log of a run",
        description=("The record of a run's latest run (also one that failed): state, verdict, error, times, "
                     "solver, the threads it used, its solves (failed first) with their status and messages, "
                     "warnings, and the run options with what set each (run flag, environment variable, "
                     "database or default); and lines of its log. solve jumps to that solve's start in the log; "
                     "grep keeps the lines that contain a text (any case, with two lines around each); otherwise "
                     "the last tail_lines lines (default 200, up to 2000). " + _RUN_TEXT),
        properties={"run": {"type": "string"}, "grep": {"type": "string"}, "solve": {"type": "string"},
                    "tail_lines": {"type": "integer"}, "max_chars": {"type": "integer"}},
        required=("run",), handler=get_run_log, area="diagnostics", small_tier=False,
        caps={"tail_lines": CapArg("max_rows", maximum=2000, default=200), "max_chars": CapArg("max_chars")}),
    make_tool(
        name="get_slack_summary", title="Where a run used its slacks",
        description=("The penalised slacks a run that succeeded used: node balance (loss of load and excess "
                     "load), non-synchronous limit, inertia, reserve and capacity margin, each with what it "
                     "means, its unit, the total per period, the largest entries, the inputs that decide it and, "
                     "for a node, the units and connections that feed it in the run's scenario; and the penalty "
                     "costs. min_value (default 0.000001) leaves out smaller values; max_rows (default 20) entries "
                     "per slack. " + _RUN_TEXT),
        properties={"run": {"type": "string"}, "min_value": {"type": "number"}, "max_rows": {"type": "integer"},
                    "max_chars": {"type": "integer"}},
        required=("run",), handler=get_slack_summary, area="diagnostics", small_tier=False,
        caps={"max_rows": CapArg("max_rows", default=20), "max_chars": CapArg("max_chars")}),
]
