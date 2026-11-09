"""Recorded replies of real AI services (P1 §15 test 14, §15a).

For every recording under ``recorded/<preset>/``: the scrub check finds nothing (no key, no long run of letters and
digits outside a signature, no absolute path, no e-mail address, no request header, no identification header of a
provider library, only the fixture's project name, no private address or host), and the recording replays through the real session and
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


# --- private hosts and addresses (C6-M3) --------------------------------------------------------------------------
def _local_planted(base_url: str = "http://gpu-box.lan:11434/v1", host: str = "gpu-box.lan") -> dict:
    """An Ollama recording made against a model server on the local network; written for this test."""
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hi from 192.168.1.20 and fe80::1, "}}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "…at 192.168.1.20. (10.1.2.3), "
                                                                              "the slice x[1::2] stays."}}]}),
        "data: [DONE]",
    ]
    return {
        "format": 1, "preset": "ollama", "model": "m", "base_url": base_url, "tier": "small", "knobs": {},
        "facts": {"context_window": 32768, "loaded_context": 16384, "tools": True, "remote_host": host},
        "questions": [{"question": "Q", "final_text": "A", "ok": True, "rounds": 2, "tools": []}],
        "probes": [{"method": "POST", "path": "/api/show", "request": {"model": "m"}, "status": 200,
                    "body": {"remote_host": f"http://{host}:11434", "note": f"served by http://{host}:11434"}}],
        "exchanges": [{"phase": "chat", "request": {"method": "POST", "path": "/v1/chat/completions",
                                                    "body": {"messages": [{"role": "user", "content": "Q"}]}},
                       "status": 200, "headers": {"content-type": "text/event-stream"}, "sse_lines": lines}],
    }


def _scrub(record: dict) -> dict:
    from flextool.assistant.redact import Redactor

    return recordings.scrub(record, secrets=(), redact=Redactor(None, home=Path("/home/someone"), host="builder",
                                                                 user="someone").redact)


def test_the_scrub_replaces_local_hosts_everywhere() -> None:
    clean = _scrub(_local_planted())
    whole = json.dumps(clean, ensure_ascii=False)
    assert clean["base_url"] == "http://endpoint.example.test:11434/v1"         # a name reads as remote
    assert clean["facts"]["remote_host"] == "forwarded.example.test"
    assert clean["probes"][0]["body"]["remote_host"] == "http://forwarded.example.test:11434"
    for gone in ("gpu-box", "192.168.", "10.1.2.3", "fe80"):
        assert gone not in whole, gone
    text = "".join(recordings.sse_data(line)["choices"][0]["delta"]["content"]
                   for line in clean["exchanges"][0]["sse_lines"][:2])
    assert text == ("hi from 192.0.2.1 and 2001:db8::1, …at 192.0.2.1. (192.0.2.2), the slice x[1::2] stays.")
    assert recordings.problems(clean, project_name=PROJECT_DIR_NAME, private_hosts=["gpu-box.lan"]) == []


def test_the_check_flags_local_hosts() -> None:
    found = "\n".join(recordings.problems(_local_planted(), project_name=PROJECT_DIR_NAME))
    for expected in ("a host under 'base_url'", "a host under 'remote_host'", "an IP address under",
                     "an IPv6 address under", "a host name of the local network under"):
        assert expected in found, expected
    # a single-label host after :// and an address split over two streamed pieces
    record = _local_planted("http://gpu:11434/v1", "ollama.com")
    record["exchanges"][0]["sse_lines"] = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "at 192.16"}}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "8.1.20 now"}}]})]
    record["probes"] = []
    found = "\n".join(recordings.problems(record, project_name=PROJECT_DIR_NAME))
    assert "a host under 'base_url'" in found and "a one-label host after :// under 'base_url'" in found
    assert "an IP address under 'content'" in found


def test_a_one_label_host_is_replaced_only_as_a_host() -> None:
    record = _local_planted("http://gpu:11434/v1", "ollama.com")
    record["exchanges"][0]["sse_lines"] = ["data: " + json.dumps({"choices": [{"index": 0, "delta": {
        "content": "the gpu maintains http://gpu:11434/v1 and gpu:11434"}}]})]
    record["probes"] = []
    clean = _scrub(record)
    content = recordings.sse_data(clean["exchanges"][0]["sse_lines"][0])["choices"][0]["delta"]["content"]
    assert content == ("the gpu maintains http://endpoint.example.test:11434/v1 and endpoint.example.test:11434")
    assert clean["base_url"] == "http://endpoint.example.test:11434/v1"
    assert clean["facts"]["remote_host"] == "ollama.com"
    # the bare word is left (it is also an ordinary word) and the check, told the host, refuses it
    found = recordings.problems(clean, project_name=PROJECT_DIR_NAME, private_hosts=["gpu"])
    assert found == ["the private host name 'gpu' under 'content'"]


def test_the_placeholder_keeps_the_route_kind() -> None:
    from flextool.assistant import net

    lan = recordings.placeholder_base_url("ollama", "http://192.168.1.20:11434/v1")
    assert lan == "http://192.0.2.1:11434/v1"
    assert net.endpoint_kind(lan) == net.endpoint_kind("http://192.168.1.20:11434/v1") == "local_network"
    assert recordings.placeholder_base_url("ollama", "http://[fe80::1]:11434/v1") == "http://[2001:db8::1]:11434/v1"
    assert recordings.placeholder_base_url("ollama", "http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert recordings.placeholder_base_url("anthropic", "https://api.anthropic.com") == "https://api.anthropic.com"
    gateway = recordings.placeholder_base_url("openai", "https://gw.example.com/v1/account-7/gateway")
    assert gateway == "https://endpoint.example.test/v1" and net.endpoint_kind(gateway) == "remote"
    assert recordings.placeholder_base_url("openai_compatible", "http://10.0.0.5:8000/v1") == \
        recordings.CUSTOM_BASE_URL


def test_the_numbered_placeholders_never_repeat_the_endpoints() -> None:
    record = _local_planted("http://192.168.1.20:11434/v1", "ollama.com")
    record["probes"] = []
    record["exchanges"][0]["request"]["path"] = "/v1/chat/completions"
    clean = _scrub(record)
    assert clean["base_url"] == "http://192.0.2.1:11434/v1"
    text = "".join(recordings.sse_data(line)["choices"][0]["delta"]["content"]
                   for line in clean["exchanges"][0]["sse_lines"][:2])
    assert "192.0.2.1 and" in text and "(192.0.2.2)" in text                 # the endpoint is 1, the other 2


def test_the_scrub_replaces_a_gateway_path() -> None:
    record = _local_planted("https://gw.example.com/v1/account-7/gateway", "ollama.com")
    record["preset"] = "openai"
    record["probes"] = []
    record["exchanges"][0]["request"]["path"] = "/v1/account-7/gateway/chat/completions"
    clean = _scrub(record)
    assert clean["base_url"] == "https://endpoint.example.test/v1"
    assert clean["exchanges"][0]["request"]["path"] == "/v1/chat/completions"


# --- the probe answers (C6-M4) -----------------------------------------------------------------------------------
def test_the_probe_recorder_keeps_only_read_fields() -> None:
    recorder = _recorder()
    show = {"modelfile": "FROM /root/.ollama/models/blobs/sha256-" + "a" * 64, "license": "LICENSE TEXT",
            "template": "{{ .Prompt }}", "capabilities": ["completion", "tools"], "details": {
                "parameter_size": "14.8B", "format": "gguf"},
            "model_info": {"general.architecture": "qwen3", "general.parameter_count": 14_768_307_200,
                           "qwen3.context_length": 40_960, "general.base_model.0.repo_url": "https://hf.example"}}
    ps = {"models": [{"name": "other:7b", "model": "other:7b", "digest": "b" * 64},
                     {"name": "m:14b", "model": "m:14b", "digest": "c" * 64, "size": 10, "size_vram": 10,
                      "context_length": 16_384, "expires_at": "2026-11-09T10:00:00Z", "details": {"family": "q"}}]}
    v1 = {"models": [{"key": "other"}, {"key": "m:14b", "max_context_length": 32_768,
                                         "loaded_instances": [{"id": "m:14b", "config": {"x": 1}}]}]}
    v0 = {"id": "m:14b", "path": "/home/someone/models/m.gguf", "max_context_length": 32_768,
          "loaded_context_length": 16_384, "state": "loaded", "capabilities": ["tool_use"], "publisher": "x"}
    answers = {"/api/show": show, "/api/ps": ps, "/api/v1/models": v1, "/api/v0/models/m:14b": v0}

    class Answer:
        def __init__(self, body):
            self.body, self.status = body, 200

    log: list[dict] = []
    from flextool.assistant import probes

    real = probes._Asker._fetch
    probes._Asker._fetch = lambda self, method, path, payload, timeout: Answer(answers[path])
    try:
        _original, fetch = recorder._probe_recorder(log, "m:14b")
        for path in answers:
            fetch(None, "GET", path, None, 1.0)
    finally:
        probes._Asker._fetch = real
    bodies = {entry["path"]: entry["body"] for entry in log}
    assert bodies["/api/show"] == {"capabilities": ["completion", "tools"], "details": {"parameter_size": "14.8B"},
                                   "model_info": {"general.architecture": "qwen3",
                                                  "general.parameter_count": 14_768_307_200,
                                                  "qwen3.context_length": 40_960}}
    assert bodies["/api/ps"] == {"models": [{"name": "m:14b", "model": "m:14b", "context_length": 16_384,
                                             "size": 10, "size_vram": 10}]}
    assert bodies["/api/v1/models"] == {"models": [{"key": "m:14b", "max_context_length": 32_768,
                                                    "loaded_instances": [{}]}]}
    assert bodies["/api/v0/models/m:14b"] == {"id": "m:14b", "max_context_length": 32_768,
                                              "loaded_context_length": 16_384, "state": "loaded",
                                              "capabilities": ["tool_use"]}
    # the reduced answers pass the check, and a 64-hex run is flagged under any key
    record = {"probes": log, "exchanges": []}
    assert recordings.problems(record, project_name=PROJECT_DIR_NAME) == []
    assert recordings.problems({"probes": [{"body": {"digest": "a" * 64}}]}, project_name=PROJECT_DIR_NAME) == \
        ["a run of 64 letters and digits under 'digest'"]
