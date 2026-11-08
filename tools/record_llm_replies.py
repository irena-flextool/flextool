"""Record the replies of a real AI service for FlexTool's tests (P1 §15a).  A developer tool: not packaged, never
run in CI.

    python tools/record_llm_replies.py --preset gemini --model gemini-2.5-flash
    python tools/record_llm_replies.py --preset ollama --model qwen3:14b
    python tools/record_llm_replies.py --preset openai_compatible --base-url http://localhost:8000/v1 --model m

The tool builds the synthetic test project (``tests/fixtures/tests.json`` with the runs of the assistant tests) in
a temporary folder — never a user's project — and asks it two fixed questions (a parameter question and a result
question) through the real session, backend and probe.  Every request body, status, allow-listed response header
and raw server-sent event line is written to ``tests/assistant/llm/recorded/<preset>/<case>.json``, and the model
server's own answers (Ollama, LM Studio, the OpenRouter model list) to ``probes``.

The key comes from where the chat finds it (the preset's environment variable, the system keyring or the key file),
never from the command line.  Before writing, the recording is scrubbed (ids, the key, run ids, paths, user and
host names; ``tests/assistant/llm/recordings.py``), checked (no key, no long letter-and-digit run outside a
signature, no absolute path, no e-mail address, no request header, the fixture's project name only) and replayed
through the backend to the same final answers.  Nothing is written when a question fails, when the model made
fewer than two rounds of tool calls, or when the check or the replay finds a problem.

Exit codes: 0 written · 1 not written (the reason is printed) · 2 usage.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx2  # noqa: E402  (after the repository root is on the path)

from flextool.assistant import endpoint, keys, net, probes, prompt  # noqa: E402
from flextool.assistant.llm_adapter import Connection, make_backend  # noqa: E402
from flextool.assistant.providers import PRESETS, tier_from_model  # noqa: E402
from flextool.assistant.settings import DEFAULT_KNOBS  # noqa: E402
from tests.assistant.llm import recordings  # noqa: E402

#: The two questions: one about input data, one about a result (P1 §15a).
QUESTIONS = (
    "What is the existing capacity of the unit coal_plant in the scenario coal? Give the value with its unit of "
    "measure and say where it comes from.",
    "Which runs does this project have, and what does the result table node_d_ee of the run coal_1 show for the "
    "node west?",
)
CASE = "two_questions"
MIN_TOOL_ROUNDS = 2


class RecordingTransport(httpx2.BaseTransport):
    """The network as it is, with each exchange written down: the request body, the status, the allow-listed
    response headers and the whole answer (decoded; a stream as its lines).  The answer is handed on at once,
    complete."""

    def __init__(self, inner: httpx2.BaseTransport, exchanges: list[dict[str, Any]], phase: dict[str, str]) -> None:
        self.inner = inner
        self.exchanges = exchanges
        self.phase = phase

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        raw_request = request.read()
        response = self.inner.handle_request(request)
        try:
            raw = b"".join(response.stream)
        finally:
            response.close()
        decoded = httpx2.Response(response.status_code, headers=response.headers, stream=httpx2.ByteStream(raw))
        text = decoded.read().decode("utf-8", errors="replace")
        entry: dict[str, Any] = {
            "phase": self.phase["now"],
            "request": {"method": request.method, "path": request.url.path,
                        "body": json.loads(raw_request) if raw_request else None},
            "status": response.status_code,
            "headers": {k.lower(): v for k, v in response.headers.items() if recordings.keep_header(k)},
        }
        if "text/event-stream" in response.headers.get("content-type", ""):
            entry["sse_lines"] = text.split("\n")
        else:
            try:
                entry["body"] = json.loads(text) if text.strip() else None
            except ValueError:
                entry["body"] = text
        self.exchanges.append(entry)
        return httpx2.Response(response.status_code, headers=response.headers, stream=httpx2.ByteStream(raw),
                               request=request, extensions=response.extensions)

    def close(self) -> None:
        self.inner.close()


def _inner_transport(base_url: str) -> httpx2.BaseTransport:
    """The transport FlexTool's own client would use: no proxy for a local endpoint, else the one it names."""
    local = net.is_local(base_url)
    return httpx2.HTTPTransport(trust_env=not local, proxy=None if local else net.proxy_for(base_url))


def _probe_recorder(log: list[dict[str, Any]], model: str):
    """A replacement of the probe's request that writes down each answer of the model server (a model list only
    with the model asked for)."""
    original = probes._Asker._fetch

    def fetch(self, method: str, path: str, payload: dict | None, timeout: float):
        answer = original(self, method, path, payload, timeout)
        body = answer.body
        if isinstance(body, dict) and isinstance(body.get("data"), list) and path.endswith("/models"):
            body = {**body, "data": [m for m in body["data"] if isinstance(m, dict) and m.get("id") == model]}
        log.append({"method": method, "path": path, "request": payload, "status": answer.status, "body": body})
        return answer

    return original, fetch


def record(preset_id: str, model: str, *, base_url: str | None, tier: str | None, project: Path,
           work: Path, say=print) -> dict[str, Any]:
    """Ask :data:`QUESTIONS` on *project* through the real service; the recording before the scrub.  Raises
    :class:`SystemExit` with the reason when nothing may be written."""
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.chat_cli import Renderer, Terminal
    from flextool.assistant.context import make_context
    from flextool.assistant.llm_errors import LLMError, message_for, render
    from flextool.assistant.session import Session, TurnCaps, result_caps_for, tool_specs
    from flextool.assistant.store import open_store

    preset = PRESETS[preset_id]
    address = net.check_base_url(base_url or preset.base_url or "")
    key, source, note = keys.resolve_key(preset, address)
    route = endpoint.route_for(preset, address, model=model)
    conn = Connection(base_url=address, model=model, api_key=key or "", key_source=source.describe(preset),
                      knobs=dict(DEFAULT_KNOBS), route=route)
    if preset.key_required and not key:
        raise SystemExit(render("no_key", error_context(preset, conn)) + (f" ({note})" if note else ""))
    level = prompt.level_name(tier) if tier else tier_from_model(preset.id, model)
    exchanges: list[dict[str, Any]] = []
    probe_log: list[dict[str, Any]] = []
    phase = {"now": "probe"}
    transport = RecordingTransport(_inner_transport(address), exchanges, phase)
    terminal = Terminal(sys.stdin, sys.stderr, sys.stderr)
    renderer = Renderer(terminal, quiet=False)

    original, fetch = _probe_recorder(probe_log, model)
    probes._Asker._fetch = fetch
    backend = make_backend(preset, conn, transport=transport)
    try:
        facts = probes.probe(preset, conn, threading.Event(),
                             needed=probes.needed_tokens(prompt.system_layers("terminal", level), tool_specs(level)),
                             on_event=renderer.event, backend=backend if preset.context_probe == "anthropic" else None)
    except LLMError as exc:
        backend.close()
        raise SystemExit(message_for(exc, "terminal")) from None
    finally:
        probes._Asker._fetch = original
    final_route = endpoint.route_for(preset, address, facts=facts)
    if final_route != route:
        backend.close()
        conn = Connection(base_url=address, model=model, api_key=conn.api_key, key_source=conn.key_source,
                          knobs=conn.knobs, route=final_route)
        transport = RecordingTransport(_inner_transport(address), exchanges, phase)
        backend = make_backend(preset, conn, transport=transport)
    phase["now"] = "chat"

    ctx = make_context(project, front_end="terminal")
    ctx.caps = result_caps_for(level, facts)
    store = open_store(work / "config")
    conversation = store.new_conversation(ctx, backend.origin, level)
    caps = TurnCaps.for_tier(level)
    session = Session(ctx, backend, preset, conversation=conversation, caps=caps, tier=level,
                      store=store, ledger=None, facts=facts, route=final_route)
    questions: list[dict[str, Any]] = []
    try:
        for question in QUESTIONS:
            say(f"you> {question}")
            tools: list[str] = []

            def on_event(event, tools=tools) -> None:
                from flextool.assistant.llm_adapter import ToolDone

                if isinstance(event, ToolDone):
                    tools.append(event.name)
                renderer.event(event)

            result = session.run_turn(question, on_event, threading.Event())
            renderer.finish()
            if not result.ok:
                raise SystemExit("Not written: the question failed: "
                                 + (message_for(result.error, "terminal") if result.error else "no answer"))
            questions.append({"question": question, "final_text": result.text, "ok": result.ok,
                              "rounds": result.rounds, "tools": tools})
    finally:
        store.close(conversation)
        backend.close()
    tool_rounds = sum(q["rounds"] - 1 for q in questions)
    if tool_rounds < MIN_TOOL_ROUNDS:
        raise SystemExit(f"Not written: the model made {tool_rounds} round(s) of tool calls, at least "
                         f"{MIN_TOOL_ROUNDS} are needed. Try again, or another model.")
    return {
        "format": recordings.FORMAT,
        "preset": preset.id,
        "model": model,
        "base_url": recordings.CUSTOM_BASE_URL if preset.id == "openai_compatible" else address,
        "tier": level,
        "max_continuations": caps.max_continuations,
        "knobs": dict(conn.knobs),
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "flextool_version": importlib.metadata.version("flextool"),
        "facts": {"context_window": facts.context_window, "loaded_context": facts.loaded_context,
                  "tools": facts.tools, "remote_host": facts.remote_host},
        "questions": questions,
        "probes": probe_log,
        "exchanges": exchanges,
        "_secrets": (key or "",),
    }


def finish(raw: dict[str, Any], project: Path, work: Path) -> dict[str, Any]:
    """Scrub, check and replay *raw*; the recording to write.  Raises :class:`SystemExit` with the problems."""
    from flextool.assistant.redact import Redactor
    from tests.assistant.conftest import PROJECT_DIR_NAME

    secrets = tuple(raw.pop("_secrets", ()))
    clean = recordings.scrub(raw, secrets=secrets, redact=Redactor(project).redact)
    found = recordings.problems(clean, project_name=PROJECT_DIR_NAME)
    if found:
        raise SystemExit("Not written: the scrubbed recording still has\n  " + "\n  ".join(found))
    texts, transport = recordings.replay(clean, project, work / "replay_config")
    expected = [q["final_text"] for q in clean["questions"]]
    if texts != expected or transport.served != len(transport.exchanges):
        raise SystemExit("Not written: the replay did not give the recorded answers "
                         f"({transport.served} of {len(transport.exchanges)} exchanges used).")
    return clean


def write(clean: dict[str, Any], out: Path, case: str) -> Path:
    path = out / clean["preset"] / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="record_llm_replies.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, choices=list(PRESETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="the address, for a custom endpoint or a model server elsewhere")
    parser.add_argument("--tier", choices=list(prompt.LEVELS), help="the level (default: from the model name)")
    parser.add_argument("--case", default=CASE, help=f"the file name of the recording (default {CASE})")
    parser.add_argument("--out", type=Path, default=recordings.RECORDED, help="the recordings folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    from tests.assistant.conftest import build_assistant_project, build_tests_db

    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="flextool-recording-") as folder:
        work = Path(folder)
        (work / "db").mkdir()
        project = build_assistant_project(work / "projects", build_tests_db(work / "db" / "tests.sqlite"))
        try:
            raw = record(args.preset, args.model, base_url=args.base_url, tier=args.tier, project=project,
                         work=work, say=lambda text: print(text, file=sys.stderr))
            clean = finish(raw, project, work)
        except SystemExit as exc:
            print(exc.code if isinstance(exc.code, str) else "Not written.", file=sys.stderr)
            return 1
        except net.BaseUrlRefused as exc:
            print(exc, file=sys.stderr)
            return 2
    path = write(clean, args.out, args.case)
    print(f"Written: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
