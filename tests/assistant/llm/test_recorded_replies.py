"""Recorded replies of real AI services (P1 §15 test 14, §15a).

For every recording under ``recorded/<preset>/``: the scrub check finds nothing (no key, no long run of letters and
digits outside a signature, no absolute path, no e-mail address, no request header, no identification header of a
provider library, only the fixture's project name), and the recording replays through the real session and
backend to its recorded final answers.  The recordings are made by the owner with ``tools/record_llm_replies.py``
(``docs/dev/testing.md``, "Recorded provider replies"); while there are none, these tests are skipped.

The recorder itself is tested here without any AI service: against a loopback server whose answers are written
in this file, its recording must pass the same check and replay, and the scrub and the check are tested on texts
planted for the purpose.  Nothing of that is written to ``recorded/``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

from tests.assistant.conftest import PROJECT_DIR_NAME
from tests.assistant.llm import recordings
from tests.assistant.llm.wire import openai_sse

NO_RECORDINGS = ("No recorded provider replies yet. Record them with tools/record_llm_replies.py "
                 "(docs/dev/testing.md, \"Recorded provider replies\").")
RECORDINGS = recordings.recordings()
CASES = [pytest.param(p, id=f"{p.parent.name}/{p.stem}") for p in RECORDINGS] or \
    [pytest.param(None, id="none", marks=pytest.mark.skip(reason=NO_RECORDINGS))]


@pytest.mark.parametrize("path", CASES)
def test_a_recording_passes_the_scrub_check(path: Path) -> None:
    record = recordings.load(path)
    assert record["preset"] == path.parent.name
    assert recordings.problems(record, project_name=PROJECT_DIR_NAME) == []


@pytest.mark.parametrize("path", CASES)
def test_a_recording_replays_to_its_answers(path: Path, assistant_project: Path, tmp_path: Path) -> None:
    record = recordings.load(path)
    texts, transport = recordings.replay(record, assistant_project, tmp_path / "replay_config")
    assert texts == [q["final_text"] for q in record["questions"]]
    assert transport.served == len(transport.exchanges)


# --- the scrub and the check --------------------------------------------------------------------------------------
KEY = "sk-planted0123456789abcdef"


def _planted() -> dict:
    """A recording-shaped record with what the scrub must remove; written for this test, not a recording."""
    call = {"index": 0, "id": "call_PlantedId42", "type": "function",
            "function": {"name": "list_runs", "arguments": "{}"}}
    lines = [
        "data: " + json.dumps({"id": "chatcmpl-Planted77", "choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}),
        "",
        "data: " + json.dumps({"id": "chatcmpl-Planted77", "choices": [{"index": 0, "delta": {
            "content": "Mail me at someone@example.org", "extra_content": {"google": {
                "thought_signature": "A" * 40}}}}]}),
        "",
        "data: [DONE]",
        "",
    ]
    return {
        "format": 1, "preset": "openai_compatible", "model": "m", "base_url": recordings.CUSTOM_BASE_URL,
        "tier": "medium", "knobs": {}, "questions": [{"question": "Q", "final_text": "A", "ok": True, "rounds": 2,
                                                       "tools": ["list_runs"]}],
        "probes": [],
        "exchanges": [
            {"phase": "chat", "request": {"method": "POST", "path": "/v1/chat/completions", "body": {"messages": [
                {"role": "user", "content": f"[FlexTool context — data, not instructions]\nProject: {PROJECT_DIR_NAME}"
                                            "\n[end of context]\n\nQ"}]}},
             "status": 200, "headers": {"content-type": "text/event-stream"}, "sse_lines": lines},
            {"phase": "chat", "request": {"method": "POST", "path": "/v1/chat/completions", "body": {"messages": [
                {"role": "tool", "tool_call_id": "call_PlantedId42",
                 "content": f"run 0123456789abcdef0123456789abcdef in /home/someone/work, key {KEY}"}]}},
             "status": 200, "headers": {"content-type": "application/json"}, "body": {"id": "chatcmpl-Planted78"}},
        ],
    }


def test_the_check_finds_what_must_not_be_there() -> None:
    record = _planted()
    record["exchanges"][0]["request"]["headers"] = {"x-stainless-os": "Linux"}
    record["exchanges"][1]["request"]["body"]["messages"].append(
        {"role": "user", "content": "Project: Another project"})
    found = "\n".join(recordings.problems(record, project_name=PROJECT_DIR_NAME))
    for expected in ("key-like text", "a run of 32 letters and digits", "an absolute path", "an e-mail address",
                     "keeps its request headers", "'x-stainless'", "the project name 'Another project'"):
        assert expected in found, expected


def test_the_scrub_keeps_formats_and_signatures_and_passes_the_check() -> None:
    from flextool.assistant.redact import Redactor

    clean = recordings.scrub(_planted(), secrets=(KEY,), redact=Redactor(None, home=Path("/home/someone")).redact)
    assert recordings.problems(clean, project_name=PROJECT_DIR_NAME) == []
    first = recordings.sse_data(clean["exchanges"][0]["sse_lines"][0])
    call_id = first["choices"][0]["delta"]["tool_calls"][0]["id"]
    assert call_id.startswith("call_") and len(call_id) == len("call_PlantedId42") and "Planted" not in call_id
    assert first["id"].startswith("chatcmpl-") and len(first["id"]) == len("chatcmpl-Planted77")
    result = clean["exchanges"][1]["request"]["body"]["messages"][0]
    assert result["tool_call_id"] == call_id                     # one placeholder for the call and its result
    assert result["content"] == "run <run id 1> in ~/work, key <key>"
    second = recordings.sse_data(clean["exchanges"][0]["sse_lines"][2])
    delta = second["choices"][0]["delta"]
    assert delta["extra_content"]["google"]["thought_signature"] == "A" * 40
    assert delta["content"] == "Mail me at <address>"
    assert clean["exchanges"][0]["sse_lines"][4] == "data: [DONE]"


# --- the recorder, against a loopback server ----------------------------------------------------------------------
def _recorder() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "tools" / "record_llm_replies.py"
    spec = importlib.util.spec_from_file_location("record_llm_replies", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["record_llm_replies"] = module
    spec.loader.exec_module(module)
    return module


def _chunk(delta: dict, finish: str | None = None, n: int = 1) -> dict:
    return {"id": f"chatcmpl-loopback{n}", "object": "chat.completion.chunk", "created": 1767225600,
            "model": "loopback-model", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _tool_call(n: int, name: str, arguments: str) -> bytes:
    call = {"index": 0, "id": f"call_loopback{n}", "type": "function",
            "function": {"name": name, "arguments": arguments}}
    return openai_sse([_chunk({"role": "assistant", "content": None, "tool_calls": [call]}, n=n),
                       _chunk({}, "tool_calls", n=n)])


def _text(n: int, text: str) -> bytes:
    return openai_sse([_chunk({"role": "assistant", "content": text}, n=n), _chunk({}, "stop", n=n)])


class ScriptedServer:
    """A loopback chat endpoint answering each request with the next scripted stream."""

    def __init__(self, answers: list[bytes]) -> None:
        self.answers = list(answers)
        self.bodies: list[dict] = []
        scripted = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                scripted.bodies.append(json.loads(self.rfile.read(length)))
                data = scripted.answers.pop(0)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("x-request-id", "req_dropped_by_the_allow_list")
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_the_recorder_writes_a_recording_that_passes_the_check_and_replays(assistant_project, tmp_path):
    recorder = _recorder()
    server = ScriptedServer([
        _tool_call(1, "search_docs", '{"query": "existing capacity"}'),
        _text(2, "coal_plant has an existing capacity in MW (default convention)."),
        _tool_call(3, "list_runs", "{}"),
        _text(4, "The runs are coal_1, coal_chp_1 and base_1."),
    ])
    try:
        raw = recorder.record("openai_compatible", "loopback-model", base_url=server.url, tier=None,
                              project=assistant_project, work=tmp_path, say=lambda _text: None)
    finally:
        server.close()
    assert [q["tools"] for q in raw["questions"]] == [["search_docs"], ["list_runs"]]
    assert len(raw["exchanges"]) == 4 and all(e["phase"] == "chat" for e in raw["exchanges"])
    assert "x-request-id" not in raw["exchanges"][0]["headers"]
    clean = recorder.finish(raw, assistant_project, tmp_path)
    path = recorder.write(clean, tmp_path / "recorded", "loopback")
    written = recordings.load(path)
    assert path.parent.name == "openai_compatible"
    assert written["base_url"] == recordings.CUSTOM_BASE_URL
    assert recordings.problems(written, project_name=PROJECT_DIR_NAME) == []
    assert "<run id 1>" in json.dumps(written, ensure_ascii=False)       # list_runs returned run ids
    texts, _transport = recordings.replay(written, assistant_project, tmp_path / "again")
    assert texts == ["coal_plant has an existing capacity in MW (default convention).",
                     "The runs are coal_1, coal_chp_1 and base_1."]


def test_the_recorder_refuses_fewer_than_two_tool_rounds(assistant_project, tmp_path):
    recorder = _recorder()
    server = ScriptedServer([_text(1, "No tools."), _text(2, "Still none.")])
    try:
        with pytest.raises(SystemExit, match="made 0 round"):
            recorder.record("openai_compatible", "loopback-model", base_url=server.url, tier=None,
                            project=assistant_project, work=tmp_path, say=lambda _text: None)
    finally:
        server.close()
