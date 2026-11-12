"""The assistant's chat in the terminal (P1 §11): ``flextool-assistant --project-path P`` and ``-q QUESTION``.

``flextool-assistant`` without ``--mcp`` or ``--call`` comes here (:func:`run`, from :func:`flextool.assistant.cli.main`;
:func:`main` parses the command line itself, for tests).  Besides the chat it runs the small commands
``--list-models``, ``--set-key`` / ``--forget-key`` and ``--usage``.

* **The REPL** (a terminal only: a chat whose input is not a terminal exits 2, so neither the data notice nor a key
  is ever read from piped text).  The first start prints the condensed welcome; without a provider it asks for one
  (preset, address, key, model from the live list) and offers to save it.  Then ``you>`` questions and ``/``
  commands.  The answer streams; tool calls, waits and retries are one dim line each; a token line follows each
  answer.  Ctrl-C during an answer stops it at once ("Stopped."); Ctrl-C or Ctrl-D at the prompt leaves.
  ``/retry`` asks the last question again and ``/edit`` changes it first (item 220): its answer goes, also in a
  chat continued with ``--continue``.  On a terminal with GNU readline the question is put on the input line to
  change; elsewhere (Windows without readline, libedit, a question of several lines) it is printed to copy.
* **One question** (``-q``): never asks anything.  Only the answer goes to stdout; tool lines, notices, the token
  line and errors go to stderr.  Exit codes: 0 answered · 1 the provider gave an error · 2 setup or usage problem ·
  3 the data notice is not accepted · 130 stopped.

Nothing is sent to the model before the first question.  Before it, on a remote route (a service on the internet,
or a model a local server forwards, such as an Ollama cloud model), the data notice must have been accepted
(:mod:`flextool.assistant.endpoint`): in the REPL the user types yes; ``--accept-data-notice`` records a lasting
acceptance.  Then the model is probed (:func:`flextool.assistant.probes.probe`), which may show that a local model
is forwarded elsewhere, so the notice is checked again for that route; then the session (:mod:`.session`) answers.
The flags fill the context snapshot (input source, scenario); none has an environment twin (P1-D8).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TextIO

from flextool.assistant import endpoint, prompt
from flextool.assistant.llm_adapter import (
    Backend,
    Connection,
    Continuing,
    Event,
    Notice,
    RetryWait,
    Route,
    TextDelta,
    Thinking,
    ToolDone,
    ToolRunning,
    ToolWaiting,
    Waiting,
)
from flextool.assistant.providers import PRESETS, Preset, knob_reports, model_id, relaxed_knobs

if TYPE_CHECKING:
    from flextool.assistant.context import AssistantContext
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.session import Session, TurnResult, TurnWorker
    from flextool.assistant.settings import Settings
    from flextool.assistant.store import Conversation, Store
    from flextool.assistant.tiers import TierChoice
    from flextool.assistant.usage import Ledger

BackendFactory = Callable[[Preset, Connection], Backend]

# --- what the chat says (plain English, no abbreviations, no links) -------------------------------------------------
#: The condensed welcome (the owner-approved window text, P2 §8.2, shortened for the terminal).
WELCOME = (
    "How the AI assistant works\n"
    "Starting this chat sends none of your data to an AI service: your data leaves this computer only when you "
    "send a question here. The assistant reads what it needs from your input data, results and run logs with "
    "read-only tools and gives advice; it cannot change your model.\n"
    "You need an account with an AI provider and its key, or an AI model running on this computer. Free plans "
    "work, but they limit how many questions you can send and often let the provider use your data for training; "
    "paid plans have higher limits and may more often let you limit how your data is used. Check the provider's "
    "current policies yourself. A model running on this computer keeps your data here, but it needs a powerful "
    "computer. Some models offered through programs on this computer still run elsewhere, such as Ollama's cloud "
    "models; FlexTool tells you when that happens.\n"
    "FlexTool picks the assistant level (basic, standard or advanced) from the model name and its context size; "
    "you can choose a lower level with --tier. Type /help for the commands."
)
HELP = (
    "Commands:\n"
    "  /help          this list\n"
    "  /new           start a new conversation\n"
    "  /new carry     start a new conversation with a short summary of this one\n"
    "  /retry         ask your last question again; its answer is replaced\n"
    "  /edit          change your last question and ask it again; its answer is replaced\n"
    "  /model NAME    use another model of this provider (/model alone lists them)\n"
    "  /provider      choose another provider\n"
    "  /privacy       where the data goes and the privacy settings sent with each request\n"
    "  /usage         the tokens used today and by month, provider and model\n"
    "  /context       what the assistant is told about your choices, and the model's context size\n"
    "  /welcome       how the assistant works\n"
    "  /save          keep this provider, model and level (and --free-tier) for the next start\n"
    "  /quit          leave (Ctrl-D or Ctrl-C at the prompt also leaves)\n"
    "End a line with \\ to continue the question on the next line. Ctrl-C stops an answer."
)
TEXTS: dict[str, str] = {
    "needs_terminal": "The interactive chat needs a terminal; use -q for a single question.",
    "no_provider": ("No AI provider is set up. Run flextool-assistant {project} once without -q to choose one."),
    "unknown_saved_provider": "The saved provider {preset} is not known to this FlexTool; choose one again.",
    "no_model": ("No model is chosen for {provider}. Add --model NAME; flextool-assistant --list-models --provider "
                 "{preset} lists the models."),
    "no_address": ("{provider} needs the address of the endpoint: add --base-url, for example "
                   "--base-url http://localhost:8000/v1."),
    "not_accepted": ("The data notice for {provider} at {endpoint} has not been accepted. Run once without -q, or "
                     "add --accept-data-notice."),
    "consent_question": "Type yes to send data to {provider}, anything else to stop: ",
    "nothing_sent": "Nothing was sent to {provider}.",
    "accepted_lasting": ("The data notice is accepted for {provider} at {endpoint}; it is asked again only when the "
                         "provider, the address, the proxy or the notice changes."),
    "accept_not_kept": "The acceptance could not be kept ({reason}); it holds for this session only.",
    "stopped": "Stopped.",
    "partial_not_sent": "The unfinished question was not sent.",
    "unknown_command": "There is no command {command}. Type /help for the commands.",
    "choose_provider": "Choose the AI service:",
    "choose_number": "Number (Enter to stop): ",
    "not_a_choice": "That is not one of the numbers.",
    "address": "Address of the endpoint [{default}]: ",
    "address_needed": "Address of the endpoint, for example http://localhost:8000/v1: ",
    "key_found": "Using the key from {source}.",
    "key_prompt": "Paste the key for {provider} (it is not shown): ",
    "key_prompt_optional": "Paste the key for {provider} (it is not shown; Enter for none): ",
    "key_needed": "{provider} needs a key; nothing was set up.",
    "key_stored": "The key is stored in {source}.",
    "key_file_question": "Type yes to store it there, anything else to use it for this session only: ",
    "key_session": "The key is used for this session only.",
    "key_removed": "The stored key for {provider} is removed.",
    "forget_question": "Remove the stored {provider} key? [y/N] ",
    "forget_kept": "The stored key for {provider} is kept.",
    "forget_terminal": ("Removing the stored key asks first, and there is no terminal to ask in; add --yes to "
                        "remove it without asking."),
    "yes_only_with_forget": "--yes goes with --forget-key.",
    "key_terminal": "Storing a key needs a terminal, so the key is never read from piped text.",
    "key_file_only_with_set": "--key-file goes with --set-key.",
    "key_not_needed": "{provider} needs no key.",
    "models_head": "Models of {provider}:",
    "hidden_models": "Not shown: {kinds}; add --all to list them.",
    "hidden_model": "Not shown: {kinds}; add --all to list it.",
    "hidden_pre-release": "{n} pre-releases",
    "hidden_pre-release_one": "1 pre-release",
    "hidden_alias": "{n} -latest aliases",
    "hidden_alias_one": "1 -latest alias",
    "hidden_older": "{n} older versions",
    "hidden_older_one": "1 older version",
    "hidden_not-chat": "{n} models that are not chat models",
    "hidden_not-chat_one": "1 model that is not a chat model",
    "hidden_unavailable": "{n} models a test found not available to your account",
    "hidden_unavailable_one": "1 model a test found not available to your account",
    "all_only_with_list": "--all goes with --list-models.",
    "testing": "Testing {model} of {provider}…",
    "model_choice": "Model (number or name; Enter to stop): ",
    "model_name": "Model name (Enter to stop): ",
    "no_models": "{provider} lists no models.",
    "save_question": "Type yes to use this provider and model also next time, anything else for this time only: ",
    "saved": "Saved: the chat starts with {provider} and {model} next time.",
    "save_failed": "The settings could not be saved: {reason}",
    "conversation_new": "A new conversation.",
    "conversation_carry": "A new conversation; your next question starts with a short summary of the last one.",
    "conversation_continued": "Continuing the conversation of {created} ({n} questions so far).",
    "conversation_missing": "There is no saved conversation {which}for this project; starting a new one.",
    "no_retry": "No question to ask again.",
    "no_edit": "No question to edit.",
    "asking_again": "Asking again: {question}",
    "edit_line": "Change the question and press Enter; an empty line keeps the question and its answer.",
    "edit_copy": ("Your last question is below. Type or paste it with your changes; an empty line keeps the "
                  "question and its answer."),
    "edit_kept": "Nothing changed: the question and its answer are kept.",
    "model_switched": "The next question goes to {model}.",
    "provider_switched": "The next question goes to {provider}, model {model}.",
    "give_provider": "Give the provider with --provider, one of: {ids}.",
    "tier_not_allowed": ("The level {level} is not available for this model, so nothing was sent. {why} Levels it "
                         "allows: {allowed}."),
}
#: The parts of the banner.
BANNER = "FlexTool assistant · project {project}"
BANNER_MODEL = "Model {model} · {key}"
BANNER_FREE = "Free plan: on, so each question uses fewer steps."
#: The level line of the banner; *why* is the reason the model is at its level, or :data:`LEVEL_CHOSEN`.
LEVEL_LINE = "Level: {level} ({why})"
LEVEL_CHOSEN = "chosen; the automatic level for this model is {automatic}"
BANNER_SENT = "Sent with every request: {reports}."
BANNER_END = "Type /help for the commands and /quit to leave."
WAITING = "Waiting for the model ({seconds} s) — Ctrl-C stops"
THINKING = "thinking…"
TOOL_WAITING = "· {name} waits for another tool call to finish"
PRIVACY_NONE = "Sent with every request: no privacy setting (this service has none per request)."
PRIVACY_ACCOUNT = "Set in your provider account (FlexTool cannot set these):"
#: What /privacy says of the context block's model facts (sent with each question that has a scenario).
PRIVACY_FACTS = ("With each question about a scenario (--scenario) the assistant is also sent its model facts: the "
                 "names of its solves, periods, forecast branches and timesets, its time steps, and the run folder "
                 "name and state of its latest run, with that run's error message.")
CONTEXT_LINE = ("Context: {window} tokens ({source}); tool results up to {chars} characters; earlier results "
                "shortened every {every}.")
#: The longest /context waits for a tool call to finish before it shows the block without the model facts.
SHOW_CONTEXT_LOCK_WAIT = 5.0
CONTEXT_NOT_YET = "Context: not known yet (FlexTool checks it before the first question)."
#: Where the context size of :data:`CONTEXT_LINE` came from (``Budget.window_source``).
WINDOW_SOURCES: dict[str, str] = {
    "loaded": "as the model is loaded on its server",
    "model": "as the provider reports it for this model",
    "model_list": "from the provider's model list",
    "assumed": "assumed: this service does not report it",
}
#: ``--free-tier`` -> the setting (``None``: automatic).
FREE_TIER_FLAG: dict[str, bool | None] = {"on": True, "off": False, "auto": None}

LEVEL_NAMES: dict[str, str] = {internal: name for name, internal in prompt.LEVELS.items()}
ARGS_MAX = 80


# --- the terminal -----------------------------------------------------------------------------------------------
def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def enable_windows_colour() -> bool:
    """Turn on the escape sequences of a Windows console (virtual-terminal mode); ``False`` when it cannot be."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.GetStdHandle(-11)                 # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))     # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except (AttributeError, OSError):
        return False


@dataclass
class Terminal:
    """The streams of the chat.  *interrupt* is a second way to press Ctrl-C (tests set it); *secret_reader*
    replaces :func:`getpass.getpass`."""

    stdin: TextIO
    stdout: TextIO
    stderr: TextIO
    color: bool = False
    interrupt: threading.Event = field(default_factory=threading.Event)
    secret_reader: Callable[[str], str | None] | None = None

    @classmethod
    def real(cls, *, no_color: bool) -> Terminal:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")
        color = not no_color and _isatty(sys.stdout) and enable_windows_colour()
        return cls(sys.stdin, sys.stdout, sys.stderr, color=color)

    @property
    def interactive(self) -> bool:
        return _isatty(self.stdin)

    def _is_console(self) -> bool:
        return self.stdin is sys.stdin and self.stdout is sys.stdout and self.interactive

    def read_line(self, text: str) -> str | None:
        """One line typed after *text*; ``None`` at the end of the input.  Ctrl-C raises KeyboardInterrupt."""
        if self._is_console():
            try:
                return input(text)
            except EOFError:
                return None
        self.stdout.write(text)
        self.stdout.flush()
        line = self.stdin.readline()
        if line == "":
            return None
        return line.rstrip("\r\n")

    def secret(self, text: str) -> str | None:
        """A line typed without echo (a key)."""
        if self.secret_reader is not None:
            return self.secret_reader(text)
        import getpass

        try:
            return getpass.getpass(text)
        except EOFError:
            return None

    def say(self, text: str, *, err: bool = False) -> None:
        stream = self.stderr if err else self.stdout
        stream.write(text + "\n")
        stream.flush()


def line_editor() -> Any | None:
    """GNU readline, imported only when asked (importing it changes how :func:`input` reads); ``None`` without it
    or with libedit (macOS), whose startup hook does not put text on the line the same way."""
    try:
        import readline
    except ImportError:
        return None
    if getattr(readline, "backend", "readline") == "editline" or "libedit" in (readline.__doc__ or ""):
        return None
    if not (hasattr(readline, "set_startup_hook") and hasattr(readline, "insert_text")):
        return None
    return readline


def _dim(text: str, color: bool) -> str:
    return f"\033[2m{text}\033[0m" if color else text


def args_text(arguments: str) -> str:
    """A tool call's arguments as ``key="value"`` pairs, at most :data:`ARGS_MAX` characters."""
    from flextool.assistant.caps import cap_text

    try:
        value = json.loads(arguments) if arguments else {}
    except ValueError:
        value = None
    if isinstance(value, dict):
        text = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in value.items())
    else:
        text = " ".join(arguments.split())
    return cap_text(text, ARGS_MAX)


def _seconds(value: float) -> str:
    return f"{value:.1f} s"


class Renderer:
    """Shows the events of a turn.  The answer streams to stdout (in ``-q`` only the final answer, at the end);
    tool lines, waits and notices are dim lines on stdout in the REPL and plain lines on stderr in ``-q``.  On a
    terminal, waits and running tools are one line that is rewritten; elsewhere each kind is written once."""

    def __init__(self, term: Terminal, *, quiet: bool) -> None:
        self.term = term
        self.quiet = quiet
        self.info = term.stderr if quiet else term.stdout
        self.color = term.color and not quiet
        self.rewrite = _isatty(self.info)
        self._lock = threading.Lock()
        self._transient = False
        self._last_kind: object = None
        self._open_line = False

    def _clear(self) -> None:
        if self._transient:
            self.info.write("\r\033[K")
            self._transient = False

    def _end_answer_line(self) -> None:
        if self._open_line:
            self.term.stdout.write("\n")
            self._open_line = False

    def _line(self, text: str) -> None:
        self._clear()
        if not self.quiet:
            self._end_answer_line()
        self.info.write(_dim(text, self.color) + "\n")
        self.info.flush()
        self._last_kind = None

    def _passing(self, kind: object, text: str) -> None:
        if self.rewrite:
            if not self.quiet:
                self._end_answer_line()
            self.info.write("\r\033[K" + _dim(text, self.color))
            self.info.flush()
            self._transient = True
        elif kind != self._last_kind:
            self._line(text)
        self._last_kind = kind

    def event(self, ev: Event) -> None:
        with self._lock:
            if isinstance(ev, TextDelta):
                if self.quiet or not ev.text:
                    return
                self._clear()
                self.term.stdout.write(ev.text)
                self.term.stdout.flush()
                self._open_line = not ev.text.endswith("\n")
            elif isinstance(ev, Thinking):
                self._passing("thinking", THINKING)
            elif isinstance(ev, Waiting):
                self._passing("waiting", WAITING.format(seconds=ev.seconds))
            elif isinstance(ev, RetryWait):
                self._passing(("retry", ev.attempt), ev.reason)
            elif isinstance(ev, ToolRunning):
                if self.rewrite:            # elsewhere the line of the finished call is enough
                    self._passing(("tool", ev.name), f"· {ev.name} {args_text(ev.args_preview)}…")
            elif isinstance(ev, ToolWaiting):
                self._passing(("tool wait", ev.name), TOOL_WAITING.format(name=ev.name))
            elif isinstance(ev, ToolDone):
                outcome = ev.summary if ev.ok else f"failed: {ev.summary}"
                self._line(f"· {ev.name} {args_text(ev.arguments)} → {outcome} ({_seconds(ev.seconds)})")
            elif isinstance(ev, Notice):
                self._line(ev.text)
            elif isinstance(ev, Continuing):
                from flextool.assistant.session import continuing_text

                self._line(continuing_text(ev.part, ev.of))

    def note(self, text: str) -> None:
        """A line of the chat itself (not of the model)."""
        with self._lock:
            self._line(text)

    def message(self, text: str) -> None:
        """An error or a stop: on stderr, after the answer so far."""
        with self._lock:
            self._clear()
            self._end_answer_line()
            self.term.stdout.flush()
            self.term.stderr.write(text + "\n")
            self.term.stderr.flush()
            self._last_kind = None

    def finish(self) -> None:
        """The end of an answer: the streamed line closed."""
        with self._lock:
            self._clear()
            self._end_answer_line()
            self.term.stdout.flush()


# --- the provider -----------------------------------------------------------------------------------------------
class SetupProblem(Exception):
    """The chat cannot start or go on; the message says why.  *code* is the exit code."""

    def __init__(self, text: str, code: int = 2) -> None:
        super().__init__(text)
        self.code = code


@dataclass
class Choice:
    """The provider, address, model and level of this session (flags over the saved settings)."""

    preset: Preset
    base_url: str
    model: str | None
    tier: str | None                    # the user's level (basic / standard / advanced); None: from the model
    knobs: dict[str, object]
    session_key: str | None = field(default=None, repr=False)      # typed in, kept for this session only

    def tier_choice(self, facts: ModelFacts | None = None) -> TierChoice:
        """The level of this model (P3 §3): with the probe's *facts*, else with the facts FlexTool last found for
        it (the name alone when there are none).  Raises :class:`~flextool.assistant.tiers.WindowTooSmall` only
        for the probe's facts."""
        from flextool.assistant import probes, tiers

        model = self.model or ""
        if facts is not None:
            route = endpoint.route_for(self.preset, self.base_url, facts=facts)
            return tiers.resolve_tier(self.preset, model, facts, self.tier, route)
        route = endpoint.route_for(self.preset, self.base_url, model=model)
        cached = probes.cached_facts(self.preset.id, self.base_url, model) if model else None
        try:
            return tiers.resolve_tier(self.preset, model, cached, self.tier, route)
        except tiers.WindowTooSmall:
            return tiers.resolve_tier(self.preset, model, None, self.tier, route)

    @property
    def level(self) -> str:
        """The internal level name before the probe (:meth:`tier_choice`)."""
        return self.tier_choice().chosen



def level_lines(tier: TierChoice, *, chosen: bool) -> list[str]:
    """The banner's level line and the sentences about the level (P3 §5): the reason in the line, then why higher
    levels are left out, the context's ceiling, the name-only and lowered notices.  *chosen*: the user gave the
    level (``--tier`` or the saved one) and it is in use."""
    from flextool.assistant import tiers

    if chosen:
        why = LEVEL_CHOSEN.format(automatic=tier.automatic_name)
    else:
        why = tier.reason
    lines = tier.lines()
    if tier.source != "fallback":
        placement = tier.placement()
        lines = [line for line in lines if line != placement]
        if placement.endswith(tiers.NO_HIGHER):
            lines.insert(0, tiers.NO_HIGHER)
    return [LEVEL_LINE.format(level=tier.level_name, why=why), *lines]


def free_tier_setting(args: argparse.Namespace, settings: Settings) -> bool | None:
    """The free-plan setting of this session: ``--free-tier``, else the saved one (``None``: automatic)."""
    flag = getattr(args, "free_tier", None)
    return settings.free_tier if flag is None else FREE_TIER_FLAG[flag]


def _check_address(url: str) -> str:
    from flextool.assistant.net import BaseUrlRefused, check_base_url

    try:
        return check_base_url(url)
    except BaseUrlRefused as exc:
        raise SetupProblem(str(exc)) from None


def choice_from(args: argparse.Namespace, settings: Settings, term: Terminal | None = None) -> Choice | None:
    """The provider of this session: the flags, then the saved settings of the same provider; ``None`` when no
    provider is set."""
    saved = settings.provider
    preset_id = args.provider or saved.preset
    if not preset_id:
        return None
    preset = PRESETS.get(preset_id)
    if preset is None:
        if term is not None:
            term.say(TEXTS["unknown_saved_provider"].format(preset=preset_id), err=True)
        return None
    same = preset_id == saved.preset
    base_url = args.base_url or (saved.base_url if same else None) or preset.base_url
    if not base_url:
        raise SetupProblem(TEXTS["no_address"].format(provider=preset.label))
    return Choice(preset=preset, base_url=_check_address(base_url), model=args.model or (saved.model if same else None),
                  tier=args.tier or (saved.tier if same else None), knobs=dict(saved.knobs))


def resolve_key(choice: Choice) -> tuple[str, str, str | None]:
    """``(key or "", where it came from, a note on the keyring)`` of *choice*."""
    from flextool.assistant import keys

    if choice.session_key:
        return choice.session_key, "what you typed in this session", None
    key, source, note = keys.resolve_key(choice.preset, choice.base_url)
    if key:
        return key, source.describe(choice.preset), None
    return "", "", note


def connection_for(choice: Choice, route: Route | None = None) -> Connection:
    """The connection of *choice*; its route from the model name until the probe settles it."""
    key, source, _note = resolve_key(choice)
    if route is None:
        route = endpoint.route_for(choice.preset, choice.base_url, model=choice.model)
    return Connection(base_url=choice.base_url, model=model_id(choice.preset.id, choice.model or ""), api_key=key,
                      key_source=source,
                      knobs=dict(choice.knobs), route=route)


def default_backend_factory(verbose: bool) -> BackendFactory:
    def make(preset: Preset, conn: Connection) -> Backend:
        from flextool.assistant.llm_adapter import make_backend

        return make_backend(preset, conn, verbose=verbose)
    return make


def _error_text(err: Exception) -> str:
    from flextool.assistant.llm_errors import LLMError, message_for

    return message_for(err, "terminal") if isinstance(err, LLMError) else str(err)


def list_models(choice: Choice, factory: BackendFactory, conn: Connection | None = None) -> list:
    """The live model list of *choice*'s provider (sends the key, no case data); *conn*: its connection, when the
    caller already has it."""
    backend = factory(choice.preset, conn or connection_for(choice))
    try:
        return backend.list_models()
    finally:
        backend.close()


# --- the interactive setup ----------------------------------------------------------------------------------------
def _ask_number(term: Terminal, count: int) -> int | None:
    while True:
        answer = term.read_line(TEXTS["choose_number"])
        if answer is None or not answer.strip():
            return None
        if answer.strip().isdigit() and 1 <= int(answer.strip()) <= count:
            return int(answer.strip()) - 1
        term.say(TEXTS["not_a_choice"])


def _setup_key(term: Terminal, choice: Choice) -> bool:
    """Find or ask for the key of *choice*; ``False`` when a needed key was not given."""
    from flextool.assistant import keys

    preset = choice.preset
    if not (preset.key_required or preset.key_optional):
        return True
    key, source, note = keys.resolve_key(preset, choice.base_url)
    if key:
        term.say(TEXTS["key_found"].format(source=source.describe(preset)))
        return True
    if note:
        term.say(note)
    asked = TEXTS["key_prompt" if preset.key_required else "key_prompt_optional"].format(provider=preset.label)
    typed = (term.secret(asked) or "").strip()
    if not typed:
        if preset.key_required:
            term.say(TEXTS["key_needed"].format(provider=preset.label))
            return False
        return True
    _store_key(term, preset, choice.base_url, typed, choice)
    return True


def _store_key(term: Terminal, preset: Preset, base_url: str | None, key: str, choice: Choice | None,
               *, to_file: bool = False) -> bool:
    """Store *key* in the keyring (or the key file after its warning); a key that cannot be stored is used for
    this session only (with *choice*)."""
    from flextool.assistant import keys

    if not to_file:
        try:
            source = keys.store_key(preset, base_url, key)
            term.say(TEXTS["key_stored"].format(source=source.describe(preset)))
            return True
        except keys.KeyStoreError as exc:
            term.say(str(exc))
    term.say(keys.key_file_warning())
    answer = term.read_line(TEXTS["key_file_question"])
    if answer is not None and answer.strip().lower() == "yes":
        try:
            source = keys.store_key(preset, base_url, key, to_file=True)
        except keys.KeyStoreError as exc:
            term.say(str(exc))
        else:
            term.say(TEXTS["key_stored"].format(source=source.describe(preset)))
            return True
    if choice is not None:
        choice.session_key = key
        term.say(TEXTS["key_session"])
    return False


def _choose_model(term: Terminal, choice: Choice, factory: BackendFactory) -> str | None:
    from flextool.assistant.llm_errors import LLMError, render

    try:
        models = list_models(choice, factory)
    except LLMError as exc:
        term.say(render("model_list", reason=_error_text(exc).rstrip(".")))
        models = None
    if models:
        from flextool.assistant.providers import shown_models

        shown = set(shown_models(choice.preset, [e.id for e in models]))
        models = [e for e in models if e.id in shown]
        term.say(TEXTS["models_head"].format(provider=choice.preset.label))
        for i, entry in enumerate(models, 1):
            term.say(f"  {i}. {entry.id}" + (f"  ({entry.note})" if entry.note else ""))
        answer = term.read_line(TEXTS["model_choice"])
        if answer is None or not answer.strip():
            return None
        text = answer.strip()
        if text.isdigit() and 1 <= int(text) <= len(models):
            return models[int(text) - 1].id
        return text
    if models is not None:
        term.say(TEXTS["no_models"].format(provider=choice.preset.label))
    answer = term.read_line(TEXTS["model_name"])
    return answer.strip() if answer and answer.strip() else None


def interactive_setup(term: Terminal, settings: Settings, factory: BackendFactory,
                      tier: str | None = None) -> Choice | None:
    """Choose the provider, its address, key and model in the terminal; offer to save them.  ``None``: stopped."""
    presets = list(PRESETS.values())
    term.say(TEXTS["choose_provider"])
    for i, preset in enumerate(presets, 1):
        where = " (a model on this computer)" if preset.local else ""
        term.say(f"  {i}. {preset.label}{where}")
    index = _ask_number(term, len(presets))
    if index is None:
        return None
    preset = presets[index]
    base_url = preset.base_url
    if preset.base_url is None or preset.local:
        while True:
            asked = TEXTS["address"].format(default=preset.base_url) if preset.base_url else TEXTS["address_needed"]
            answer = term.read_line(asked)
            if answer is None:
                return None
            text = answer.strip() or (preset.base_url or "")
            if not text:
                continue
            try:
                base_url = _check_address(text)
                break
            except SetupProblem as exc:
                term.say(str(exc))
    assert base_url is not None
    choice = Choice(preset=preset, base_url=base_url, model=None, tier=tier, knobs=dict(settings.provider.knobs))
    if not _setup_key(term, choice):
        return None
    choice.model = _choose_model(term, choice, factory)
    if not choice.model:
        return None
    answer = term.read_line(TEXTS["save_question"])
    if answer is not None and answer.strip().lower() == "yes":
        save_choice(term, settings, choice)
    return choice


def save_choice(term: Terminal, settings: Settings, choice: Choice) -> bool:
    from flextool.assistant import _files
    from flextool.assistant.settings import save_settings

    provider = settings.provider
    provider.preset = choice.preset.id
    provider.base_url = None if choice.base_url == choice.preset.base_url else choice.base_url
    provider.model = choice.model
    provider.tier = choice.tier
    provider.knobs = dict(choice.knobs)
    try:
        save_settings(settings)
    except (_files.NewerFormat, _files.LockTimeout, OSError) as exc:
        term.say(TEXTS["save_failed"].format(reason=exc))
        return False
    term.say(TEXTS["saved"].format(provider=choice.preset.label, model=choice.model))
    return True


# --- the chat -----------------------------------------------------------------------------------------------------
@dataclass
class Link:
    """The prepared connection: the probe's facts, the final route and the backend."""

    choice: Choice
    conn: Connection
    facts: ModelFacts
    backend: Backend
    tier: TierChoice

    @property
    def level(self) -> str:
        """The internal level in use: the probe's facts decided it."""
        return self.tier.chosen


class Chat:
    def __init__(self, args: argparse.Namespace, term: Terminal, ctx: AssistantContext, settings: Settings,
                 choice: Choice, factory: BackendFactory, store: Store, ledger: Ledger, *, quiet: bool) -> None:
        self.args = args
        self.term = term
        self.ctx = ctx
        self.settings = settings
        self.choice = choice
        self.factory = factory
        self.store = store
        self.ledger = ledger
        self.quiet = quiet
        self.renderer = Renderer(term, quiet=quiet)
        self.link: Link | None = None
        self.session: Session | None = None
        self.worker: TurnWorker | None = None
        self.seed: str | None = None
        #: The summary the first question of this conversation started with (a /new carry): asked again with it.
        self.sent_seed: str | None = None
        self.accepted: set[endpoint.ConsentKey] = set()
        self.free_tier = free_tier_setting(args, settings)
        self.budget = None
        self.conversation = self._open_conversation()

    # --- conversations --------------------------------------------------------------------------------------------
    def _origin(self) -> tuple[str, str, str]:
        return (self.choice.preset.id, self.choice.base_url, self.choice.model or "")

    def _open_conversation(self) -> Conversation:
        from flextool.assistant.store import ConversationDamaged, ConversationNotFound

        resume = getattr(self.args, "resume", None)
        if resume is not None:
            try:
                conv = self.store.open(self.ctx.project_path, resume or None)
            except ConversationNotFound:
                which = f"{resume} " if resume else ""
                self.renderer.note(TEXTS["conversation_missing"].format(which=which))
            except ConversationDamaged as exc:
                self.renderer.note(str(exc))
            else:
                if conv.notice:
                    self.renderer.note(conv.notice)
                else:
                    questions = len({m.turn for m in conv.messages if m.role == "user"})
                    self.renderer.note(TEXTS["conversation_continued"].format(created=conv.created, n=questions))
                return conv
        return self.store.new_conversation(self.ctx, self._origin(), self.choice.tier)

    def new_conversation(self, *, carry: bool) -> None:
        from flextool.assistant.session import carry_summary

        self._finish_worker()
        old = self.conversation
        self.seed = carry_summary(old.messages) if carry and old.messages else None
        self.sent_seed = None
        self.store.close(old)
        self.conversation = self.store.new_conversation(self.ctx, self._origin(), self.choice.tier)
        self.session = None
        self.worker = None
        if self.link is not None:
            self._make_session(self.link)
        self.renderer.note(TEXTS["conversation_carry" if self.seed else "conversation_new"])

    # --- the data notice ------------------------------------------------------------------------------------------
    def consent(self, route: Route) -> int | None:
        """``None`` when data may go on *route*; else the exit code (3), after saying why."""
        return ask_consent(self.term, self.args, self.choice, route, self.accepted, quiet=self.quiet)

    # --- the connection -------------------------------------------------------------------------------------------
    def _wait(self, done: threading.Event, stop: Callable[[], None]) -> bool:
        """Wait for *done*; on Ctrl-C call *stop*, say "Stopped." and return ``True``."""
        try:
            while not done.wait(0.05):
                if self.term.interrupt.is_set():
                    self.term.interrupt.clear()
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            stop()
            self.renderer.message(TEXTS["stopped"])
            return True
        return False

    def _run(self, fn: Callable[[], Any], cancel: threading.Event) -> tuple[bool, Any, BaseException | None]:
        """Run *fn* in a daemon thread; ``(stopped, result, error)``."""
        box: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 — handed to the waiting thread
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=run, name="flextool-assistant-prepare", daemon=True).start()
        if self._wait(done, cancel.set):
            return True, None, None
        return False, box.get("result"), box.get("error")

    def prepare(self) -> int | None:
        """Make the connection ready for a question: the data notice, the probe, the notice for the probed route,
        the backend and the session.  ``None`` when ready, else the exit code."""
        from flextool.assistant import probes, tiers
        from flextool.assistant.llm_errors import LLMError, NoKey
        from flextool.assistant.session import tool_specs

        if self.link is not None:
            return None
        choice = self.choice
        if not choice.model:
            self.renderer.message(TEXTS["no_model"].format(provider=choice.preset.label, preset=choice.preset.id))
            return 2
        code = self._refuse_tier(choice.tier_choice())
        if code is not None:
            return code
        conn = connection_for(choice)
        if choice.preset.key_required and not conn.api_key:
            from flextool.assistant.backends._common import error_context
            from flextool.assistant.llm_errors import render

            self.renderer.message(render("no_key", error_context(choice.preset, conn)))
            return 2
        code = self.consent(conn.route)
        if code is not None:
            return code
        needed = probes.needed_tokens(prompt.system_layers("terminal", "small"), tool_specs("small"))
        cancel = threading.Event()
        stopped, facts, error = self._run(
            lambda: probes.probe(choice.preset, conn, cancel, needed=needed, on_event=self.renderer.event), cancel)
        if stopped:
            return 130
        if error is not None:
            if not isinstance(error, LLMError):
                raise error
            if error.code == "cancelled":
                self.renderer.message(TEXTS["stopped"])
                return 130
            self.renderer.message(_error_text(error))
            return 2 if isinstance(error, NoKey) else 1
        for note in facts.notes:
            self.renderer.note(note)
        try:
            tier = choice.tier_choice(facts)
        except tiers.WindowTooSmall as exc:
            self.renderer.message(_error_text(probes.window_error(choice.preset, conn, exc)))
            return 1
        code = self._refuse_tier(tier)
        if code is not None:
            return code
        for line in (tier.notice, tier.window_line()):
            if line:
                self.renderer.note(line)
        route = endpoint.route_for(choice.preset, choice.base_url, facts=facts)
        if route != conn.route:
            conn = dataclasses.replace(conn, route=route)
            self.renderer.note(endpoint.indicator_text(route))
            code = self.consent(route)
            if code is not None:
                return code
        try:
            backend = self.factory(choice.preset, conn)
        except LLMError as exc:
            self.renderer.message(_error_text(exc))
            return 2 if isinstance(exc, NoKey) else 1
        link = Link(choice=choice, conn=conn, facts=facts, backend=backend, tier=tier)
        if self.session is None:
            self._make_session(link)
        else:
            self._switch_session(link)
        self.link = link
        return None

    def _refuse_tier(self, tier: TierChoice) -> int | None:
        """2 when ``-q`` asks with ``--tier`` for a level above the ones the model allows (P3 §5), after saying
        why; ``None`` otherwise (the interactive chat lowers the level with a notice instead)."""
        if not (self.quiet and self.args.tier and tier.notice):
            return None
        why = tier.window_line() or tier.placement()
        allowed = ", ".join(LEVEL_NAMES[level] for level in tier.allowed)
        level = LEVEL_NAMES[prompt.level_name(self.args.tier)]
        self.renderer.message(TEXTS["tier_not_allowed"].format(level=level, why=why, allowed=allowed))
        return 2

    def _budget(self, link: Link):
        """The session's budget (P3 §4): the level fitted to the model's window, the free plan, the settings'
        compaction overrides and ``--result-cap``; the result caps go into the context."""
        from flextool.assistant import budget

        free = budget.free_tier_on(link.choice.preset.id, link.conn.model, self.free_tier)
        b = budget.budget_for(link.tier, free_tier=free, compaction_every=self.settings.compaction_every,
                              keep_recent_turns=self.settings.keep_recent_turns, result_cap=self.args.result_cap)
        self.ctx.caps = b.result_caps
        self.budget = b
        return b

    def _make_session(self, link: Link) -> None:
        from flextool.assistant.session import Session, TurnCaps, TurnWorker
        from flextool.assistant.settings import mark_free_tier_suggested

        b = self._budget(link)
        self.conversation.level = link.choice.tier          # the user's level, never the resolved one
        self.session = Session(self.ctx, link.backend, link.choice.preset, conversation=self.conversation,
                               caps=TurnCaps.from_budget(b), tier=link.level, store=self.store, ledger=self.ledger,
                               facts=link.facts, route=link.conn.route, budget=b,
                               free_tier_hint=mark_free_tier_suggested)
        self.worker = TurnWorker(self.session)

    def _switch_session(self, link: Link) -> None:
        from flextool.assistant.session import TurnCaps

        assert self.session is not None
        self._finish_worker()
        b = self._budget(link)
        self.conversation.level = link.choice.tier
        self.session.switch(link.backend, link.choice.preset, tier=link.level, facts=link.facts,
                            route=link.conn.route, caps=TurnCaps.from_budget(b), budget=b)

    def drop_link(self) -> None:
        """Forget the prepared connection (another model or provider): the next question prepares again."""
        if self.link is not None:
            self._finish_worker()
            self.link.backend.close()
            self.link = None

    def _finish_worker(self, timeout: float = 5.0) -> None:
        if self.worker is not None:
            self.worker.join(timeout)

    # --- a question -----------------------------------------------------------------------------------------------
    def ask(self, text: str, *, rewind: int | None = None) -> int:
        """Answer *text*; the exit code of ``-q``.  *rewind*: *text* replaces question *rewind* and everything
        after it (:meth:`Session.run_turn`), once the connection is ready."""
        from flextool.assistant.llm_errors import message_for
        from flextool.assistant.usage import format_turn_line

        code = self.prepare()
        if code is not None:
            return code
        assert self.worker is not None and self.link is not None
        if rewind is not None:
            seed = self.sent_seed
        else:
            seed, self.seed = self.seed, None
            if seed:
                self.sent_seed = seed
        done = threading.Event()
        box: dict[str, TurnResult] = {}

        def finished(result: TurnResult) -> None:
            box["result"] = result
            done.set()

        self.worker.start(text, self.renderer.event, finished, rewind=rewind, seed=seed)
        if self._wait(done, self.worker.stop):
            return 130
        result = box["result"]
        self.renderer.finish()
        if self.quiet and result.text.strip():
            self.term.stdout.write(result.text.rstrip("\n") + "\n")
            self.term.stdout.flush()
        if result.error is None and not result.ok:
            return 1                        # nothing was asked (TurnWorker: a question that cannot be asked again)
        line = format_turn_line(result.usage, self.conversation.usage, self.link.conn.route.label)
        if result.error is not None:
            if result.error.code == "cancelled":
                self.renderer.message(TEXTS["stopped"])
                return 130
            self.renderer.message(message_for(result.error, "terminal"))
        self.renderer.note(line)
        return 0 if result.ok else 1

    # --- the REPL -------------------------------------------------------------------------------------------------
    def banner(self) -> None:
        choice = self.choice
        key, source, note = resolve_key(choice)
        if key:
            key_text = f"key from {source}"
        else:
            key_text = "no key" if choice.preset.key_required else "no key needed"
        from flextool.assistant.budget import free_tier_on

        route = endpoint.route_for(choice.preset, choice.base_url, model=choice.model)
        lines = [BANNER.format(project=self.ctx.project_name), endpoint.indicator_text(route),
                 BANNER_MODEL.format(model=choice.model or "not chosen", key=key_text)]
        if note:
            lines.append(note)
        if choice.model:
            tier = choice.tier_choice()
            chosen = bool(choice.tier) and tier.notice is None
            lines += level_lines(tier, chosen=chosen)
            if free_tier_on(choice.preset.id, choice.model, self.free_tier):
                lines.append(BANNER_FREE)
        reports = knob_reports(choice.preset, choice.knobs)
        if reports:
            lines.append(BANNER_SENT.format(reports="; ".join(reports)))
        lines += relaxed_knobs(choice.preset, choice.knobs)
        lines.append(BANNER_END)
        for line in lines:
            self.term.say(line)

    def read_question(self) -> str | None:
        lines: list[str] = []
        while True:
            line = self.term.read_line("you> " if not lines else "...> ")
            if line is None:
                if lines:                   # Ctrl-D on a continuation line: leave without sending the part
                    self.term.say("")
                    self.term.say(TEXTS["partial_not_sent"])
                return None
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            return "\n".join(lines)

    def repl(self) -> int:
        self.banner()
        while True:
            try:
                text = self.read_question()
            except KeyboardInterrupt:
                self.term.say("")
                return 0
            if text is None:
                return 0
            if not text.strip():
                continue
            try:
                if text.strip().startswith("/"):
                    if not self.command(text.strip()):
                        return 0
                    continue
                self.ask(text)
            except KeyboardInterrupt:
                # Ctrl-C at a prompt inside a question or a command (the data notice, a chooser, a key), or one
                # that came before the wait took over: any running turn is stopped, and the chat goes on.
                self._stop_turn()
                self.term.say("")
                self.term.say(TEXTS["stopped"])

    def _stop_turn(self) -> None:
        if self.worker is not None and not self.worker.join(0):
            self.worker.stop()
            self._finish_worker()

    def command(self, line: str) -> bool:
        """Run a ``/`` command; ``False`` to leave."""
        name, _, rest = line.partition(" ")
        rest = rest.strip()
        if name == "/quit":
            return False
        if name == "/help":
            self.term.say(HELP)
        elif name == "/new":
            self.new_conversation(carry=rest == "carry")
        elif name == "/retry":
            self.retry()
        elif name == "/edit":
            self.edit()
        elif name == "/model":
            self.change_model(rest)
        elif name == "/provider":
            self.change_provider()
        elif name == "/privacy":
            self.privacy()
        elif name == "/usage":
            from flextool.assistant.usage import format_totals

            self.term.say(format_totals(self.ledger.read()))
        elif name == "/context":
            self.show_context()
        elif name == "/welcome":
            self.term.say(WELCOME)
        elif name == "/save":
            if self.args.free_tier is not None:
                self.settings.free_tier = self.free_tier
            save_choice(self.term, self.settings, self.choice)
        else:
            self.term.say(TEXTS["unknown_command"].format(command=name))
        return True

    # --- asking again ---------------------------------------------------------------------------------------------
    def _last_question(self) -> int | None:
        """The number of the conversation's last question (answered or not), ``None`` when it has none."""
        from flextool.assistant.session import CONTROL_KINDS

        return max((m.turn for m in self.conversation.messages
                    if m.role == "user" and m.kind not in CONTROL_KINDS and m.turn >= 1), default=None)

    def retry(self) -> None:
        """``/retry``: ask the last question again; its answer and anything after it go."""
        from flextool.assistant.session import question_text

        self._finish_worker()
        turn = self._last_question()
        if turn is None:
            self.term.say(TEXTS["no_retry"])
            return
        text = question_text(self.conversation.messages, turn, self.sent_seed)
        self.renderer.note(TEXTS["asking_again"].format(question=text))
        self.ask(text, rewind=turn)

    def edit(self) -> None:
        """``/edit``: change the last question, then ask it instead; an empty line changes nothing."""
        from flextool.assistant.session import question_text

        self._finish_worker()
        turn = self._last_question()
        if turn is None:
            self.term.say(TEXTS["no_edit"])
            return
        old = question_text(self.conversation.messages, turn, self.sent_seed)
        text = self._edited(old)
        if text is None or not text.strip():
            self.term.say(TEXTS["edit_kept"])
            return
        self.ask(text, rewind=turn)

    def _edited(self, old: str) -> str | None:
        """The question the user types to replace *old*: on the console with GNU readline *old* is on the input line
        to change; else *old* is printed to copy, with `` \\`` ending each line but the last (a question of several
        lines is typed that way).  ``None`` at the end of the input; Ctrl-C raises KeyboardInterrupt."""
        editor = line_editor() if self.term._is_console() and "\n" not in old else None
        if editor is None:
            self.term.say(TEXTS["edit_copy"])
            self.term.say(" \\\n".join(old.split("\n")))
            return self.read_question()
        self.term.say(TEXTS["edit_line"])
        inserted = False

        def prefill() -> None:
            nonlocal inserted
            if not inserted:                # the first line only, not a continuation line
                inserted = True
                editor.insert_text(old)

        editor.set_startup_hook(prefill)
        try:
            return self.read_question()
        finally:
            editor.set_startup_hook()

    def change_model(self, model: str) -> None:
        if not model:
            model = _choose_model(self.term, self.choice, self.factory) or ""
            if not model:
                return
        self.drop_link()
        self.choice = dataclasses.replace(self.choice, model=model)
        self.term.say(TEXTS["model_switched"].format(model=model))

    def change_provider(self) -> None:
        choice = interactive_setup(self.term, self.settings, self.factory, tier=self.args.tier)
        if choice is None:
            return
        self.drop_link()
        self.choice = choice
        self.term.say(TEXTS["provider_switched"].format(provider=choice.preset.label, model=choice.model))

    def privacy(self) -> None:
        choice = self.choice
        route = self.link.conn.route if self.link is not None else \
            endpoint.route_for(choice.preset, choice.base_url, model=choice.model)
        self.term.say(endpoint.indicator_text(route))
        reports = knob_reports(choice.preset, choice.knobs)
        self.term.say(BANNER_SENT.format(reports="; ".join(reports)) if reports else PRIVACY_NONE)
        for text in relaxed_knobs(choice.preset, choice.knobs):
            self.term.say(text)
        if choice.preset.account_knobs:
            self.term.say(PRIVACY_ACCOUNT)
            for text in choice.preset.account_knobs:
                self.term.say(f"  - {text}")
        self.term.say(PRIVACY_FACTS)

    def show_context(self) -> None:
        from flextool.assistant.model_facts import with_facts

        snapshot = self.ctx.snapshot()
        if self.ctx.lock.acquire(timeout=SHOW_CONTEXT_LOCK_WAIT):     # a tool call still running: shown without facts
            try:
                snapshot = with_facts(self.ctx, snapshot, self.link.level if self.link is not None else None)
            finally:
                self.ctx.lock.release()
        block = prompt.context_block(snapshot, self.ctx.redact)
        self.term.say(block)
        b = self.budget
        if self.link is None or b is None:
            self.term.say(CONTEXT_NOT_YET)
            return
        from flextool.assistant.usage import number

        every = "question" if b.compaction_every == 1 else f"{b.compaction_every} questions"
        self.term.say(CONTEXT_LINE.format(window=number(b.window), source=WINDOW_SOURCES[b.window_source],
                                          chars=number(b.result_caps.max_chars), every=every))

    def close(self) -> None:
        self._finish_worker(2.0)
        self.store.close(self.conversation)
        if self.link is not None:
            self.link.backend.close()


# --- the small commands -------------------------------------------------------------------------------------------
def ask_consent(term: Terminal, args: argparse.Namespace, choice: Choice, route: Route,
                accepted: set[endpoint.ConsentKey], *, quiet: bool) -> int | None:
    """The data notice of *route*: ``None`` when data may go there (accepted before, in *accepted*, with
    ``--accept-data-notice`` or by typing yes); else the exit code (3), after saying why.  *quiet*: no question is
    asked (``-q``, or no terminal)."""
    key = endpoint.consent_key(choice.preset.id, choice.base_url, route)
    if key in accepted or not endpoint.needs_consent(key):
        return None
    provider = route.label
    where = route.forwarded_to or route.host
    notice = endpoint.consent_notice(route, required_headers=choice.preset.required_headers)
    if args.accept_data_notice:
        term.say(notice, err=True)
        term.say(endpoint.indicator_text(route), err=True)
        _record_consent(term, args, accepted, key, provider, where, err=True)
        return None
    if quiet:
        term.say(TEXTS["not_accepted"].format(provider=provider, endpoint=where), err=True)
        return 3
    term.say(notice)
    term.say(endpoint.indicator_text(route))
    try:
        answer = term.read_line(TEXTS["consent_question"].format(provider=provider))
    except KeyboardInterrupt:                       # Ctrl-C is a no
        term.say("")
        answer = None
    if answer is not None and answer.strip().lower() == "yes":
        _record_consent(term, args, accepted, key, provider, where, err=False)
        return None
    term.say(TEXTS["nothing_sent"].format(provider=provider))
    return 3


def _record_consent(term: Terminal, args: argparse.Namespace, accepted: set[endpoint.ConsentKey],
                    key: endpoint.ConsentKey, provider: str, where: str, *, err: bool) -> None:
    from flextool.assistant import _files

    accepted.add(key)
    try:
        endpoint.record_consent(key)
    except (_files.NewerFormat, _files.LockTimeout, OSError) as exc:
        term.say(TEXTS["accept_not_kept"].format(reason=exc), err=err)
        return
    if args.accept_data_notice:
        term.say(TEXTS["accepted_lasting"].format(provider=provider, endpoint=where), err=err)


def _need_choice(args: argparse.Namespace, settings: Settings, term: Terminal) -> Choice:
    choice = choice_from(args, settings, term)
    if choice is None:
        raise SetupProblem(TEXTS["give_provider"].format(ids=", ".join(PRESETS)))
    return choice


def cmd_list_models(args: argparse.Namespace, term: Terminal, factory: BackendFactory) -> int:
    from flextool.assistant.llm_errors import LLMError
    from flextool.assistant.settings import load_settings

    choice = _need_choice(args, load_settings(), term)
    conn = connection_for(choice)
    try:
        models = list_models(choice, factory, conn)
    except LLMError as exc:
        term.say(_error_text(exc), err=True)
        return 1
    if not models:
        term.say(TEXTS["no_models"].format(provider=choice.preset.label), err=True)
        return 0
    from flextool.assistant import model_list_cache
    from flextool.assistant.keys import key_fingerprint
    from flextool.assistant.providers import hidden_reason, shown_models

    closed = model_list_cache.unavailable(model_list_cache.load_marks(), choice.preset.id, choice.base_url,
                                          key_fingerprint(conn.api_key))
    shown = set(shown_models(choice.preset, [e.id for e in models], show_all=args.all_models, unavailable=closed))
    hidden = [hidden_reason(e.id, closed) for e in models if e.id not in shown]
    models = [e for e in models if e.id in shown]
    for entry in models:
        parts = [entry.id]
        if entry.id in closed:
            parts.append("not available to your account")
        if entry.context_window:
            from flextool.assistant.usage import number

            parts.append(f"context {number(entry.context_window)} tokens")
        if entry.note:
            parts.append(entry.note)
        term.say("  ".join(parts))
    if hidden:
        kinds = [TEXTS[f"hidden_{kind}_one"] if hidden.count(kind) == 1 else
                 TEXTS[f"hidden_{kind}"].format(n=hidden.count(kind))
                 for kind in ("pre-release", "alias", "older", "not-chat", "unavailable") if kind in hidden]
        term.say(TEXTS["hidden_model" if len(hidden) == 1 else "hidden_models"].format(kinds=", ".join(kinds)),
                 err=True)
    return 0


def cmd_test_model(args: argparse.Namespace, term: Terminal, factory: BackendFactory) -> int:
    """``--test-model``: one tiny request to the chosen model (:func:`flextool.assistant.model_check.check_model`)
    after the data notice; 0 when it answered, 1 when not, 130 when stopped."""
    from flextool.assistant import model_check, model_list_cache
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.budget import free_tier_on
    from flextool.assistant.llm_adapter import RetryWait, Waiting
    from flextool.assistant.llm_errors import LLMError, render
    from flextool.assistant.settings import load_settings
    from flextool.assistant.usage import Ledger, ledger_key

    settings = load_settings()
    choice = _need_choice(args, settings, term)
    preset = choice.preset
    if not choice.model:
        term.say(TEXTS["no_model"].format(provider=preset.label, preset=preset.id), err=True)
        return 2
    conn = connection_for(choice)
    if preset.key_required and not conn.api_key:
        term.say(render("no_key", error_context(preset, conn)), err=True)
        return 2
    route = conn.route
    code = ask_consent(term, args, choice, route, set(), quiet=not term.interactive)
    if code is not None:
        return code
    try:
        backend = factory(preset, conn)
    except LLMError as exc:
        term.say(_error_text(exc), err=True)
        return 1
    renderer = Renderer(term, quiet=True)            # the waits on stderr, the outcome alone on stdout
    renderer.note(TEXTS["testing"].format(model=conn.model, provider=route.label))
    cancel = threading.Event()
    box: dict[str, Any] = {}
    done = threading.Event()

    def waits(event: object) -> None:
        if isinstance(event, (RetryWait, Waiting)):
            renderer.event(event)  # type: ignore[arg-type]

    def run() -> None:
        try:
            box["result"] = model_check.check_model(
                preset, backend, cancel, label=route.label, ledger=Ledger(),
                ledger_key=ledger_key(preset.id, choice.base_url, forwarded_to=route.forwarded_to),
                free_plan=free_tier_on(preset.id, conn.model, free_tier_setting(args, settings)),
                front_end="terminal", on_event=waits)
        except BaseException as exc:  # noqa: BLE001 — handed to the waiting thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name="flextool-assistant-model-test", daemon=True).start()
    try:
        try:
            while not done.wait(0.05):
                if term.interrupt.is_set():
                    term.interrupt.clear()
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            cancel.set()
            backend.abort()
            done.wait(5.0)
            renderer.finish()
            term.say(TEXTS["stopped"])
            return 130
        renderer.finish()
        if "error" in box:
            raise box["error"]
        result: model_check.CheckResult = box["result"]
    finally:
        backend.close()
    if result.unavailable:
        from flextool.assistant.keys import key_fingerprint

        model_list_cache.mark_unavailable(preset.id, choice.base_url, conn.model, key_fingerprint(conn.api_key))
    elif result.outcome == "available":
        model_list_cache.clear_marks(preset.id, choice.base_url, conn.model)
    term.say(result.message)
    return {"available": 0, "stopped": 130}.get(result.outcome, 1)


def cmd_set_key(args: argparse.Namespace, term: Terminal) -> int:
    from flextool.assistant.settings import load_settings

    choice = _need_choice(args, load_settings(), term)
    preset = choice.preset
    if not (preset.key_required or preset.key_optional):
        term.say(TEXTS["key_not_needed"].format(provider=preset.label))
        return 0
    if not term.interactive:
        term.say(TEXTS["key_terminal"], err=True)
        return 2
    typed = (term.secret(TEXTS["key_prompt"].format(provider=preset.label)) or "").strip()
    if not typed:
        term.say(TEXTS["key_needed"].format(provider=preset.label), err=True)
        return 2
    stored = _store_key(term, preset, choice.base_url, typed, None, to_file=args.key_file)
    if stored:
        from flextool.assistant import model_list_cache

        model_list_cache.clear_marks(preset.id, choice.base_url)       # a new key: the marks are tested again
    return 0 if stored else 2


def cmd_forget_key(args: argparse.Namespace, term: Terminal) -> int:
    from flextool.assistant import keys
    from flextool.assistant.settings import load_settings

    choice = _need_choice(args, load_settings(), term)
    provider = choice.preset.label
    if not args.yes:
        if not term.interactive:
            term.say(TEXTS["forget_terminal"], err=True)
            return 2
        answer = term.read_line(TEXTS["forget_question"].format(provider=provider))
        if answer is None:                      # the end of the input: the line the question left open
            term.say("")
        if answer is None or answer.strip().lower() not in ("y", "yes"):
            term.say(TEXTS["forget_kept"].format(provider=provider))
            return 1
    from flextool.assistant import model_list_cache

    note = keys.forget_key(choice.preset, choice.base_url)
    model_list_cache.clear_marks(choice.preset.id, choice.base_url)
    term.say(TEXTS["key_removed"].format(provider=provider))
    if note:
        term.say(note, err=True)
    return 0


def cmd_usage(term: Terminal) -> int:
    from flextool.assistant.usage import Ledger, format_totals

    term.say(format_totals(Ledger().read()))
    return 0


# --- the entry points ---------------------------------------------------------------------------------------------
def _project_text(args: argparse.Namespace) -> str:
    if args.project_path:
        return f"--project-path {args.project_path}"
    return f"--project {args.project}"


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, *, backend_factory: BackendFactory | None = None,
        terminal: Terminal | None = None) -> int:
    """The chat or a small command, for the parsed *args* (not ``--mcp`` / ``--call``)."""
    term = terminal or Terminal.real(no_color=args.no_color)
    factory = backend_factory or default_backend_factory(args.verbose)
    if args.key_file and not args.set_key:
        parser.error(TEXTS["key_file_only_with_set"])
    if args.yes and not args.forget_key:
        parser.error(TEXTS["yes_only_with_forget"])
    if args.all_models and not args.list_models:
        parser.error(TEXTS["all_only_with_list"])
    try:
        if args.usage:
            return cmd_usage(term)
        if args.list_models:
            return cmd_list_models(args, term, factory)
        if args.test_model:
            return cmd_test_model(args, term, factory)
        if args.set_key:
            return cmd_set_key(args, term)
        if args.forget_key:
            return cmd_forget_key(args, term)
        return _chat(args, parser, term, factory)
    except SetupProblem as exc:
        term.say(str(exc), err=True)
        return exc.code
    except KeyboardInterrupt:                       # Ctrl-C at a prompt of a small command or the setup
        term.say("", err=True)
        return 130


def _chat(args: argparse.Namespace, parser: argparse.ArgumentParser, term: Terminal, factory: BackendFactory) -> int:
    from flextool.assistant.cli import project_path
    from flextool.assistant.context import make_context
    from flextool.assistant.settings import load_settings
    from flextool.assistant.store import open_store, purge_at_start
    from flextool.assistant.usage import Ledger

    quiet = args.question is not None
    project = project_path(args, parser)
    if not quiet and not term.interactive:
        term.say(TEXTS["needs_terminal"], err=True)
        return 2
    purge_at_start()
    settings = load_settings()
    for note in settings.notes:
        term.say(note, err=True)
    if not quiet and not endpoint.welcome_seen():
        term.say(WELCOME)
        term.say("")
        try:
            endpoint.mark_welcome_seen()
        except Exception as exc:  # noqa: BLE001 — the welcome is shown again next time, nothing else is lost
            term.say(str(exc), err=True)
    choice = choice_from(args, settings, term)
    if choice is None:
        if quiet:
            term.say(TEXTS["no_provider"].format(project=_project_text(args)), err=True)
            return 2
        try:
            choice = interactive_setup(term, settings, factory, tier=args.tier)
        except KeyboardInterrupt:
            term.say("")
            return 2
        if choice is None:
            return 2
    elif not choice.model and quiet:
        term.say(TEXTS["no_model"].format(provider=choice.preset.label, preset=choice.preset.id), err=True)
        return 2
    elif not choice.model:
        try:
            choice.model = _choose_model(term, choice, factory)
        except KeyboardInterrupt:
            term.say("")
            return 2
        if not choice.model:
            return 2
    # The level's result caps until the probe has seen the model (Chat._caps sets them again then).
    ctx = make_context(project, input_source=args.input_source, scenario=args.scenario,
                       result_cap=args.result_cap or choice.level, front_end="terminal")
    chat = Chat(args, term, ctx, settings, choice, factory, open_store(), Ledger(), quiet=quiet)
    try:
        if quiet:
            return chat.ask(args.question)
        return chat.repl()
    finally:
        chat.close()


def main(argv: list[str] | None = None, *, backend_factory: BackendFactory | None = None,
         terminal: Terminal | None = None) -> int:
    """Parse *argv* (the command line without the program name) and run the chat or a small command."""
    from flextool.assistant.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.mcp or args.call:
        parser.error("--mcp and --call start from flextool-assistant itself")
    return run(args, parser, backend_factory=backend_factory, terminal=terminal)

