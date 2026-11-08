"""The chat in the terminal (P1 §15 test 10).

``chat_cli.main`` runs in this process with scripted input (a stdin that says it is a terminal, or one that does
not) and :class:`FakeBackend` from the backend factory, so each test sees exactly what reached the model.  The data
notice: ``-q`` without an acceptance exits 3 and sends nothing; ``--accept-data-notice`` is lasting; another
endpoint, a proxy and a model that Ollama forwards to its cloud ask again; an endpoint on this computer asks
nothing.  The welcome is shown once and never in ``-q``; in ``-q`` stdout holds only the answer; a chat whose
input is not a terminal exits 2; Ctrl-C while a real backend waits for the first byte of a loopback server returns
within a second.  One test runs the installed ``flextool-assistant`` command against a loopback server.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flextool.assistant import chat_cli, endpoint, registry
from flextool.assistant.chat_cli import WELCOME, Terminal
from flextool.assistant.llm_adapter import Message, ModelEntry, Reply, ToolCall, Usage
from flextool.assistant.registry import make_tool
from flextool.assistant.settings import load_settings
from tests.assistant.llm.fake_backend import FakeBackend
from tests.assistant.llm.local_server import FakeServer
from tests.assistant.llm.wire import openai_sse

REMOTE = "https://llm.example.test/v1"
USAGE = Usage(120, 0, 0, 15)
NOT_ACCEPTED = ("The data notice for Custom endpoint at llm.example.test has not been accepted. Run once without "
                "-q, or add --accept-data-notice.")
NOTICE_START = ("Your questions, the case data the assistant reads (input parameters, results, logs) and the names "
                "of your project and scenarios are sent to Custom endpoint at llm.example.test.")
WELCOME_HEAD = "How the AI assistant works"
PROXY_VARIABLES = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY",
                   "no_proxy")


# --- the terminal and the model -----------------------------------------------------------------------------------
class TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def terminal(lines: list[str] = (), *, tty: bool = True) -> Terminal:
    """A terminal whose input is *lines*; a key typed without echo is read from the same lines."""
    text = "".join(line + "\n" for line in lines)
    term = Terminal(TtyInput(text) if tty else io.StringIO(text), io.StringIO(), io.StringIO())

    def secret(prompt: str) -> str | None:
        term.stdout.write(prompt)
        line = term.stdin.readline()
        return None if line == "" else line.rstrip("\n")

    term.secret_reader = secret
    return term


def say(text: str) -> Reply:
    return Reply(Message("assistant", text), "end", None, USAGE)


def echo_call(text: str, *, before: str = "") -> Reply:
    call = ToolCall(f"call{text}", "echo", {"text": text}, json.dumps({"text": text}))
    return Reply(Message("assistant", before, tool_calls=[call]), "tools", None, USAGE)


class Models:
    """The backend factory: every backend the chat builds is a :class:`FakeBackend` taking its replies from one
    script, and is kept (with its connection) for the checks."""

    def __init__(self, script: list[object], fake_backends: list[FakeBackend], models: list[ModelEntry] = ()) -> None:
        self.script = script
        self.made: list[FakeBackend] = []
        self.connections: list = []
        self.models = list(models)
        self._registry = fake_backends

    def __call__(self, preset, conn):
        backend = FakeBackend(self.script, origin=(preset.id, conn.base_url, conn.model), models=self.models)
        backend.script = self.script            # one script shared by every backend of the run
        self.made.append(backend)
        self.connections.append(conn)
        self._registry.append(backend)
        return backend

    @property
    def requests(self) -> list:
        return [r for b in self.made for r in b.requests]


@pytest.fixture
def echo_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's only tool is ``echo``."""
    props = {"text": {"type": "string", "description": "a text"}}
    tool = make_tool(name="echo", title="echo", description="The echo tool.", properties=props,
                     handler=lambda _ctx, args: {"items": [args.get("text", "")]}, area="orientation",
                     small_tier=True)
    monkeypatch.setattr(registry, "_tools", {"echo": tool})


@pytest.fixture(autouse=True)
def no_proxy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "Chat project"
    folder.mkdir()
    return folder


def remote(project: Path, *extra: str, url: str = REMOTE) -> list[str]:
    return ["--project-path", str(project), "--provider", "openai_compatible", "--base-url", url,
            "--model", "example-model", *extra]


def acks(config: Path) -> list[dict]:
    path = config / "acknowledgements.json"
    return json.loads(path.read_text(encoding="utf-8")).get("acks", []) if path.exists() else []


# --- the data notice ----------------------------------------------------------------------------------------------
def test_q_without_an_accepted_notice_exits_3_and_sends_nothing(project, fake_backends, echo_tool):
    models = Models([say("never")], fake_backends)
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "What is it?"), backend_factory=models, terminal=term) == 3
    assert models.made == []
    assert term.stdout.getvalue() == ""
    assert NOT_ACCEPTED in term.stderr.getvalue()


def test_accept_data_notice_is_lasting(project, fake_backends, echo_tool, user_config_in_tmp):
    models = Models([say("Answer one."), say("Answer two.")], fake_backends)
    first = terminal()
    assert chat_cli.main(remote(project, "-q", "One?", "--accept-data-notice"), backend_factory=models,
                         terminal=first) == 0
    assert first.stdout.getvalue() == "Answer one.\n"
    err = first.stderr.getvalue()
    assert NOTICE_START in err and "Data goes to: Custom endpoint (llm.example.test)." in err
    assert "it is asked again only when the provider, the address, the proxy or the notice changes" in err
    [ack] = acks(user_config_in_tmp)
    assert (ack["preset"], ack["endpoint"], ack["proxy"], ack["forwarded_to"]) == (
        "openai_compatible", "https://llm.example.test/v1", None, None)
    assert ack["notice"] == endpoint.NOTICE_HASH and ack["at"].endswith("Z")
    second = terminal()
    assert chat_cli.main(remote(project, "-q", "Two?"), backend_factory=models, terminal=second) == 0
    assert second.stdout.getvalue() == "Answer two.\n"
    assert NOTICE_START not in second.stderr.getvalue()


def test_an_acceptance_in_the_flextool_window_covers_the_terminal(project, fake_backends, echo_tool,
                                                                   user_config_in_tmp):
    """The window accepts the notice under the same key from its saved settings (P2 §8.3)."""
    from flextool.assistant.settings import ProviderSettings, Settings

    saved = Settings(provider=ProviderSettings(preset="openai_compatible", base_url=REMOTE, model="example-model"))
    route = endpoint.route_from_settings(saved)
    key = endpoint.consent_key("openai_compatible", REMOTE, route)
    assert endpoint.needs_consent(key)
    endpoint.record_consent(key)
    models = Models([say("Answer.")], fake_backends)
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "Q?"), backend_factory=models, terminal=term) == 0
    assert term.stdout.getvalue() == "Answer.\n"
    assert NOTICE_START not in term.stderr.getvalue()
    assert len(acks(user_config_in_tmp)) == 1


def test_another_endpoint_or_a_proxy_asks_again(project, fake_backends, echo_tool, monkeypatch):
    models = Models([say("ok")], fake_backends)
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=terminal()) == 0
    other = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", url="https://other.example.test/v1"), backend_factory=models,
                         terminal=other) == 3
    assert "The data notice for Custom endpoint at other.example.test has not been accepted." in other.stderr.getvalue()
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    proxied = terminal()
    assert chat_cli.main(remote(project, "-q", "Q"), backend_factory=models, terminal=proxied) == 3
    assert NOT_ACCEPTED in proxied.stderr.getvalue()
    assert len(models.made) == 1                            # nothing was sent on the new routes


def test_the_notice_names_the_proxy(project, fake_backends, echo_tool, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    models = Models([say("ok")], fake_backends)
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=term) == 0
    err = term.stderr.getvalue()
    assert ("Your network settings send this data through the proxy proxy.example.test, which can also see it."
            in err)
    assert "Data goes to: Custom endpoint (llm.example.test). It is sent through the proxy proxy.example.test." in err


@pytest.fixture
def ollama() -> Iterator[FakeServer]:
    server = FakeServer()
    yield server
    server.close()


def _ollama_routes(server: FakeServer, model: str, show_extra: dict | None = None) -> list[str]:
    server.routes[("POST", "/api/show")] = (200, {"capabilities": ["completion", "tools"],
                                                  "model_info": {"general.architecture": "qwen3",
                                                                 "qwen3.context_length": 65536},
                                                  **(show_extra or {})})
    server.routes[("POST", "/api/generate")] = (200, {"model": model, "done": True})
    server.routes[("GET", "/api/ps")] = (200, {"models": [{"name": model, "model": model, "size": 10,
                                                           "size_vram": 10, "context_length": 65536}]})
    return ["--provider", "ollama", "--base-url", f"http://127.0.0.1:{server.port}/v1", "--model", model]


def test_a_model_ollama_forwards_to_its_cloud_asks_for_the_notice(project, fake_backends, echo_tool, ollama):
    flags = _ollama_routes(ollama, "gpt-oss:120b", {"remote_host": "https://ollama.com:443",
                                                    "remote_model": "gpt-oss:120b"})
    models = Models([say("from the cloud")], fake_backends)
    term = terminal()
    assert chat_cli.main(["--project-path", str(project), *flags, "-q", "Q"], backend_factory=models,
                         terminal=term) == 3
    err = term.stderr.getvalue()
    assert "Data goes to: Ollama cloud (ollama.com), forwarded by Ollama on this computer." in err
    assert ("The data notice for Ollama cloud at ollama.com has not been accepted. Run once without -q, or add "
            "--accept-data-notice.") in err
    assert models.made == [] and "/api/generate" not in ollama.paths       # nothing loaded, nothing sent


def test_a_model_on_this_computer_needs_no_notice(project, fake_backends, echo_tool, ollama):
    flags = _ollama_routes(ollama, "qwen3:14b")
    models = Models([say("from here")], fake_backends)
    term = terminal()
    assert chat_cli.main(["--project-path", str(project), *flags, "-q", "Q"], backend_factory=models,
                         terminal=term) == 0
    assert term.stdout.getvalue() == "from here\n"
    assert models.connections[0].route.kind == "loopback"


# --- the welcome, the output streams, the terminal ----------------------------------------------------------------
def test_welcome_once_and_never_in_q(project, fake_backends, echo_tool):
    models = Models([say("ok")], fake_backends)
    quiet = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=quiet) == 0
    assert WELCOME_HEAD not in quiet.stdout.getvalue() + quiet.stderr.getvalue()
    assert not endpoint.welcome_seen()
    first = terminal(["/quit"])
    assert chat_cli.main(remote(project), backend_factory=models, terminal=first) == 0
    assert first.stdout.getvalue().startswith(WELCOME + "\n")
    assert endpoint.welcome_seen()
    again = terminal(["/welcome", "/quit"])
    assert chat_cli.main(remote(project), backend_factory=models, terminal=again) == 0
    assert again.stdout.getvalue().count(WELCOME_HEAD) == 1           # only the one /welcome asked for
    assert len(models.made) == 1                                       # starting the chat sent nothing


def test_q_stdout_holds_only_the_answer(project, fake_backends, echo_tool):
    models = Models([echo_call("a", before="Let me look."), say("The answer.")], fake_backends)
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=term) == 0
    assert term.stdout.getvalue() == "The answer.\n"
    err = term.stderr.getvalue()
    assert '· echo text="a" → 1 items (' in err
    assert "tokens 240 in · 30 out · conversation 240 in · 30 out" in err
    assert len(models.requests) == 2


def test_q_with_a_provider_error_exits_1(project, fake_backends, echo_tool):
    from flextool.assistant.llm_errors import ErrorContext, QuotaExhausted, make_error

    err = make_error(QuotaExhausted, "quota", ErrorContext(provider="Custom endpoint"))
    models = Models([err], fake_backends)
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=term) == 1
    assert term.stdout.getvalue() == ""
    assert "Your Custom endpoint quota is used up for now." in term.stderr.getvalue()


def test_the_interactive_chat_needs_a_terminal(project, fake_backends):
    models = Models([], fake_backends)
    term = terminal(["yes", "Q"], tty=False)
    assert chat_cli.main(remote(project), backend_factory=models, terminal=term) == 2
    assert term.stderr.getvalue().strip() == "The interactive chat needs a terminal; use -q for a single question."
    assert models.made == [] and not endpoint.welcome_seen()


def test_q_without_a_provider_exits_2(project, fake_backends):
    term = terminal()
    assert chat_cli.main(["--project-path", str(project), "-q", "Q"], terminal=term) == 2
    assert term.stderr.getvalue().strip() == (
        f"No AI provider is set up. Run flextool-assistant --project-path {project} once without -q to choose one.")


# --- the interactive chat -----------------------------------------------------------------------------------------
def test_repl_asks_for_the_notice_before_the_first_question(project, fake_backends, echo_tool, user_config_in_tmp):
    models = Models([say("Streamed answer.")], fake_backends)
    term = terminal(["What is it?", "yes", "/quit"])
    assert chat_cli.main(remote(project), backend_factory=models, terminal=term) == 0
    out = term.stdout.getvalue()
    assert out.index(NOTICE_START) < out.index("Type yes to send data to Custom endpoint, anything else to stop: ")
    assert "Streamed answer.\n" in out and "tokens 120 in · 15 out" in out
    assert len(acks(user_config_in_tmp)) == 1
    assert [m.role for m in models.requests[0].messages] == ["user"]


def test_repl_anything_but_yes_sends_nothing(project, fake_backends, echo_tool, user_config_in_tmp):
    models = Models([], fake_backends)
    term = terminal(["What is it?", "no", "/quit"])
    assert chat_cli.main(remote(project), backend_factory=models, terminal=term) == 0
    assert "Nothing was sent to Custom endpoint." in term.stdout.getvalue()
    assert models.made == [] and acks(user_config_in_tmp) == []


def test_first_run_setup_lists_the_models_and_saves(project, fake_backends, echo_tool):
    from flextool.assistant.providers import PRESETS

    number = list(PRESETS).index("openai_compatible") + 1
    models = Models([], fake_backends, models=[ModelEntry("model-a"), ModelEntry("model-b", note="a note")])
    term = terminal([str(number), REMOTE, "", "2", "yes", "/quit"])
    assert chat_cli.main(["--project-path", str(project)], backend_factory=models, terminal=term) == 0
    out = term.stdout.getvalue()
    assert "  2. model-b  (a note)" in out
    assert "Saved: the chat starts with Custom endpoint and model-b next time." in out
    saved = load_settings().provider
    assert (saved.preset, saved.base_url, saved.model) == ("openai_compatible", REMOTE, "model-b")
    assert models.requests == []                    # the model list only; no question was sent


def test_continue_goes_on_with_the_newest_conversation(project, fake_backends, echo_tool):
    models = Models([say("First answer."), say("Second answer.")], fake_backends)
    assert chat_cli.main(remote(project, "-q", "First question", "--accept-data-notice"), backend_factory=models,
                         terminal=terminal()) == 0
    term = terminal()
    assert chat_cli.main(remote(project, "-q", "Second question", "--continue"), backend_factory=models,
                         terminal=term) == 0
    texts = [m.text for m in models.made[-1].requests[0].messages]
    assert any(t.endswith("First question") for t in texts) and "First answer." in texts
    assert texts[-1] == "Second question"           # the context block is not sent again unchanged


def test_ctrl_c_at_the_prompt_leaves(project, fake_backends):
    class Interrupted(TtyInput):
        def readline(self, *args: object) -> str:
            raise KeyboardInterrupt

    term = Terminal(Interrupted(""), io.StringIO(), io.StringIO())
    endpoint.mark_welcome_seen()
    assert chat_cli.main(remote(project), backend_factory=Models([], fake_backends), terminal=term) == 0


def test_set_key_needs_a_terminal_and_stores_in_the_keyring(fake_keyring):
    piped = terminal(["sk-piped"], tty=False)
    assert chat_cli.main(["--set-key", "--provider", "mistral"], terminal=piped) == 2
    assert "never read from piped text" in piped.stderr.getvalue()
    typed = terminal(["sk-typed-key"])
    assert chat_cli.main(["--set-key", "--provider", "mistral"], terminal=typed) == 0
    assert "The key is stored in the system keyring." in typed.stdout.getvalue()
    assert "sk-typed-key" not in typed.stdout.getvalue()
    from flextool.assistant import keys
    from flextool.assistant.providers import PRESETS

    assert keys.resolve_key(PRESETS["mistral"])[0] == "sk-typed-key"


# --- Ctrl-C while the model has not answered yet --------------------------------------------------------------------
class SlowServer:
    """A loopback chat endpoint that holds every request until released (as a loading model would)."""

    def __init__(self, answer: bytes = b"") -> None:
        self.got = threading.Event()
        self.release = threading.Event()
        self.answer = answer
        slow = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                slow.got.set()
                if not slow.answer:
                    slow.release.wait(30)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(slow.answer)))
                self.end_headers()
                self.wfile.write(slow.answer)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def test_ctrl_c_before_the_first_byte_stops_within_a_second(project, echo_tool):
    server = SlowServer()
    term = terminal()
    box: dict[str, int] = {}
    argv = ["--project-path", str(project), "--provider", "openai_compatible", "--base-url", server.url,
            "--model", "slow-model", "-q", "Q"]
    worker = threading.Thread(target=lambda: box.update(code=chat_cli.main(argv, terminal=term)), daemon=True)
    try:
        worker.start()
        assert server.got.wait(30), term.stderr.getvalue()
        started = time.monotonic()
        term.interrupt.set()
        worker.join(5)
        elapsed = time.monotonic() - started
    finally:
        server.close()
    assert not worker.is_alive()
    assert elapsed <= 1.0, elapsed
    assert box["code"] == 130
    assert term.stderr.getvalue().strip().endswith("Stopped.")


# --- the installed command ----------------------------------------------------------------------------------------
def _script() -> Path:
    name = "flextool-assistant.exe" if sys.platform == "win32" else "flextool-assistant"
    path = Path(sysconfig.get_path("scripts")) / name
    if not path.is_file():
        pytest.fail(f"{path} is missing: reinstall FlexTool (pip install -e .) to get the flextool-assistant command")
    return path


def test_the_installed_command_answers_one_question(project):
    chunks = [{"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1767225600, "model": "m",
               "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Forty-two."},
                            "finish_reason": None}]},
              {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1767225600, "model": "m",
               "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
    server = SlowServer(openai_sse(chunks))
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARIABLES}
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"       # never the real keyring
    try:
        result = subprocess.run([str(_script()), "--project-path", str(project), "--provider", "openai_compatible",
                                 "--base-url", server.url, "--model", "m", "-q", "What is the answer?"],
                                capture_output=True, text=True, encoding="utf-8", timeout=180, env=env,
                                stdin=subprocess.DEVNULL)
    finally:
        server.close()
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Forty-two.\n"
    assert "tokens not reported by Custom endpoint" in result.stderr


# --- levels and the free plan (P3 §5) -----------------------------------------------------------------------------
NAME_ONLY = "The level comes from the model name; it may be lowered when FlexTool sees the model's context size."
LOWERED = "The level chosen before (advanced) is above this model's level, so the automatic level is used."


def _model(project: Path, model: str, *extra: str) -> list[str]:
    argv = remote(project, *extra)
    argv[argv.index("example-model")] = model
    return argv


def test_the_banner_gives_the_level_and_why(project, fake_backends, user_config_in_tmp):
    term = terminal(["/quit"])
    endpoint.mark_welcome_seen()
    assert chat_cli.main(_model(project, "qwen2.5-14b"), backend_factory=Models([], fake_backends),
                         terminal=term) == 0
    lines = term.stdout.getvalue().splitlines()
    at = lines.index("Level: basic (a model of about 14 billion parameters)")
    assert lines[at + 1:at + 3] == ["Higher levels are not available for it.", NAME_ONLY]
    assert not any(line.startswith("Free plan") for line in lines)


def test_q_with_a_level_above_the_model_exits_2_and_sends_nothing(project, fake_backends, echo_tool,
                                                                  user_config_in_tmp):
    models = Models([say("never")], fake_backends)
    term = terminal()
    argv = _model(project, "gpt-5-mini", "-q", "Q", "--tier", "advanced", "--accept-data-notice")
    assert chat_cli.main(argv, backend_factory=models, terminal=term) == 2
    assert term.stdout.getvalue() == ""
    assert term.stderr.getvalue().strip() == (
        "The level advanced is not available for this model, so nothing was sent. This model is placed at the basic "
        "level from its name (a small model of its family (mini)). Higher levels are not available for it. Levels "
        "it allows: basic.")
    assert models.made == [] and acks(user_config_in_tmp) == []


def test_q_with_an_allowed_lower_level_is_answered(project, fake_backends, echo_tool):
    models = Models([say("Answer.")], fake_backends)
    term = terminal()
    argv = remote(project, "-q", "Q", "--tier", "basic", "--accept-data-notice")
    assert chat_cli.main(argv, backend_factory=models, terminal=term) == 0
    assert term.stdout.getvalue() == "Answer.\n"
    assert any(layer.startswith("Your level: basic.") for layer in models.requests[0].system)


def test_the_interactive_chat_lowers_a_level_above_the_model_with_a_notice(project, fake_backends,
                                                                           user_config_in_tmp):
    term = terminal(["/quit"])
    endpoint.mark_welcome_seen()
    assert chat_cli.main(_model(project, "gpt-5-mini", "--tier", "advanced"),
                         backend_factory=Models([], fake_backends), terminal=term) == 0
    lines = term.stdout.getvalue().splitlines()
    at = lines.index("Level: basic (a small model of its family (mini))")
    assert lines[at + 1:at + 4] == ["Higher levels are not available for it.", NAME_ONLY, LOWERED]


def test_free_tier_flag_saved_and_automatic(project, fake_backends, echo_tool, user_config_in_tmp):
    free_notice = ("Free plan: the assistant uses at most 4 steps for each question to stay within the provider's "
                   "request limits.")
    models = Models([say("One."), say("Two."), say("Three.")], fake_backends)
    term = terminal(["/save", "/quit"])
    endpoint.mark_welcome_seen()
    assert chat_cli.main(remote(project, "--free-tier", "on"), backend_factory=models, terminal=term) == 0
    assert "Free plan: on, so each question uses fewer steps." in term.stdout.getvalue()
    assert load_settings().free_tier is True
    saved = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--accept-data-notice"), backend_factory=models,
                         terminal=saved) == 0
    assert free_notice in saved.stderr.getvalue()
    auto = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--free-tier", "auto"), backend_factory=models,
                         terminal=auto) == 0
    assert "Free plan" not in auto.stderr.getvalue()
    off = terminal()
    assert chat_cli.main(remote(project, "-q", "Q", "--free-tier", "off"), backend_factory=models,
                         terminal=off) == 0
    assert "Free plan" not in off.stderr.getvalue()
    assert load_settings().free_tier is True          # only /save keeps the flag


def test_context_command_gives_the_window_caps_and_rhythm(project, fake_backends, echo_tool, user_config_in_tmp):
    models = Models([say("Answer.")], fake_backends)
    term = terminal(["/context", "Q", "yes", "/context", "/quit"])
    endpoint.mark_welcome_seen()
    assert chat_cli.main(remote(project), backend_factory=models, terminal=term) == 0
    out = term.stdout.getvalue()
    assert "Context: not known yet (FlexTool checks it before the first question)." in out
    assert ("Context: 128 000 tokens (assumed: this service does not report it); tool results up to 20 000 "
            "characters; earlier results shortened every 6 questions.") in out


def test_a_continuation_is_one_line_in_the_terminal(project, fake_backends, echo_tool):
    """P3 §4.7: at basic the fourth step goes on with a fresh context; -q prints one line on stderr and only the
    final answer on stdout (the note is not shown)."""
    models = Models([echo_call("a"), echo_call("b"), echo_call("c"), say("A note.\nKeep: 1"), say("Answer.")],
                    fake_backends)
    term = terminal()
    argv = remote(project, "-q", "Q", "--tier", "basic", "--accept-data-notice")
    assert chat_cli.main(argv, backend_factory=models, terminal=term) == 0
    assert term.stdout.getvalue() == "Answer.\n"
    assert term.stderr.getvalue().count("Continuing with a fresh context (part 2 of 3)…") == 1
    assert "A note." not in term.stderr.getvalue()
    assert len(models.requests) == 5


# --- the model list -----------------------------------------------------------------------------------------------
def test_list_models_shows_the_newest_stable_versions_and_all_with_all(fake_backends, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    listed = [ModelEntry("gemini-2.5-flash"), ModelEntry("gemini-3-flash", "a note"),
              ModelEntry("gemini-3.1-flash-preview"), ModelEntry("gemini-3.1-pro-preview"),
              ModelEntry("gemini-flash-latest")]
    models = Models([], fake_backends, models=listed)
    term = terminal(tty=False)
    assert chat_cli.main(["--provider", "gemini", "--list-models"], backend_factory=models, terminal=term) == 0
    assert term.stdout.getvalue() == "gemini-3-flash  a note\n"
    assert term.stderr.getvalue() == ("Not shown: 2 pre-releases, 1 -latest alias, 1 older version; add --all to "
                                      "list them.\n")
    every = terminal(tty=False)
    assert chat_cli.main(["--provider", "gemini", "--list-models", "--all"], backend_factory=models,
                         terminal=every) == 0
    assert every.stdout.getvalue().splitlines() == [
        "gemini-2.5-flash", "gemini-3-flash  a note", "gemini-3.1-flash-preview", "gemini-3.1-pro-preview",
        "gemini-flash-latest"]
    assert every.stderr.getvalue() == ""
    one = terminal(tty=False)
    models = Models([], fake_backends, models=[ModelEntry("gemini-3-flash"), ModelEntry("gemini-3.1-flash-preview")])
    assert chat_cli.main(["--provider", "gemini", "--list-models"], backend_factory=models, terminal=one) == 0
    assert one.stderr.getvalue() == "Not shown: 1 pre-release; add --all to list it.\n"
    chat = terminal(tty=False)
    models = Models([], fake_backends, models=[ModelEntry("gemini-3-flash"), ModelEntry("gemini-3.8-flash-tts"),
                                               ModelEntry("gemini-embedding-2")])
    assert chat_cli.main(["--provider", "gemini", "--list-models"], backend_factory=models, terminal=chat) == 0
    assert chat.stdout.getvalue() == "gemini-3-flash\n"
    assert chat.stderr.getvalue() == "Not shown: 2 models that are not chat models; add --all to list them.\n"
