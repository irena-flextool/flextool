"""The assistant's level calibration on the grounding questions (P3 §6–§7).  A developer tool: not packaged, never
run in CI, never collected by pytest.

    python tools/assistant_eval.py build --out DIR
    python tools/assistant_eval.py run --project DIR --provider ID --model M --out RESULTS [...]
    python tools/assistant_eval.py grade RESULTS ... --dir GRADING [--single-model] [--regrade #N]
    python tools/assistant_eval.py summary RESULTS ... [--record]
    python tools/assistant_eval.py calibrate [results.jsonl]
    python tools/assistant_eval.py dual sample RESULTS ... --dir DUAL
    python tools/assistant_eval.py dual grade-ai --dir DUAL
    python tools/assistant_eval.py dual report --dir DUAL [--record]

``build`` makes the synthetic grounding project (``tests/assistant/grounding_projects.py``: ``tests.json`` and its
overrides, solved with ``flextool-run``) and its marker.  ``run`` asks every question of
``tools/assistant_eval_data/grounding_set.yaml`` at every level setting through the real chat stack (the provider's
backend, the probe, the budget of the level and the model's context, the session) and writes one record per cell
(model, level setting, question, repeat) into RESULTS.  It works only on a project its own ``build`` made: the
marker and the source database's hash must match, so case data of a real project never reaches a calibration
model.  Keys come from the keyring or the owner's environment, never from the command line.  ``grade`` runs the
automatic checks against the oracle's ``answer_key.yaml`` and asks the owner, blind, for the items only a person
can judge; ``summary`` prints the rates and with ``--record`` appends the graded cells to
``tools/assistant_eval_data/results.jsonl`` (no answer or tool text).  ``calibrate`` turns ``results.jsonl`` into
the evidence tables for the owner and ``dual`` runs the second grader on a subset and reports the agreement
(``tools/assistant_eval_analysis.py``).  How to run and read it: ``docs/dev/assistant_evaluation.md``.

Exit codes: 0 done (also: the daily request budget is used, run again with --resume) · 1 failed · 2 usage or a
refused project / results folder · 3 the data notice was not accepted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for _path in (ROOT, TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assistant_eval_scoring as scoring  # noqa: E402  (after the repository root is on the path)

RECORD_FORMAT = 1
RUN_FILE = "run.json"
CELLS_DIR = "cells"
GRADING_FILE = "grading.json"
DEFAULT_OUT = ROOT / "_audit_reports" / "assistant_eval"
#: The daily request budget of an OpenRouter ``:free`` model (P3 §7.2).
FREE_DAILY_REQUESTS = 45
DEFAULT_SITTING = 37
#: The ledger provider key of calibration requests: the route's key with this suffix (shown apart in /usage).
CALIBRATION_SUFFIX = "#calibration"
OLLAMA_CONTEXTS = {"small": 16_384, "medium": 32_768, "large": 65_536}

NOT_GROUNDING = "This is not a grounding project made by assistant_eval build."
OTHER_RESULTS = "These results belong to another model or project; use a new --out folder."
DAILY_DONE = "Daily request budget used. Run again with --resume tomorrow."
DAILY_DONE_REMOTE = "Daily request budget used. Run again with --resume --accept-data-notice tomorrow."
HAS_RESULTS = "This folder has results already: add --resume to continue them, or use a new --out folder."


class Refused(Exception):
    """A refusal before any request: the message and the exit code."""

    def __init__(self, text: str, code: int = 2) -> None:
        super().__init__(text)
        self.code = code


def accept_route(preset: Any, base_url: str, route: Any, accepted: set, *, accept: bool, again: str,
                 what: str) -> None:
    """The data notice of a remote *route*, accepted for this process only (*accepted*): the calibration neither
    reads nor writes the chat's acknowledgements, so an acceptance here never covers a real chat and one given in
    the chat never covers a calibration.  :class:`Refused` (exit 3) when ``--accept-data-notice`` was not given;
    *again* is the command-line addition the rerun needs."""
    from flextool.assistant import endpoint

    key = endpoint.consent_key(preset.id, base_url, route)
    if not key.remote or key in accepted:
        return
    print(endpoint.consent_notice(route, required_headers=preset.required_headers), file=sys.stderr)
    print(endpoint.indicator_text(route), file=sys.stderr)
    if not accept:
        raise Refused(f"Nothing was sent. Run again with {again} to accept this notice for {what}.", code=3)
    accepted.add(key)


class DailyBudgetUsed(Exception):
    """The route's calibration requests of this UTC day reached the budget, or the provider's quota is used."""


# --- small helpers --------------------------------------------------------------------------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: Any) -> None:
    """*data* as UTF-8 JSON at *path*, through a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def flextool_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flextool")
    except PackageNotFoundError:
        return None


def internal_levels(text: str) -> list[str]:
    """``basic,standard,advanced`` (or the internal names) as internal level names, in the order given."""
    from flextool.assistant import prompt

    out = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        try:
            level = prompt.level_name(part)
        except ValueError:
            raise Refused(f"Unknown level {part!r}: use basic, standard and advanced.") from None
        if level not in out:
            out.append(level)
    if not out:
        raise Refused("Name at least one level.")
    return out


def parse_ollama_context(text: str | None) -> dict[str, int]:
    """``basic=16384,standard=32768,advanced=65536`` as ``{internal level: context}``."""
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        name, _, value = part.partition("=")
        if not value.strip().isdigit():
            raise Refused(f"--ollama-context: {part!r} is not level=tokens.")
        out[internal_levels(name)[0]] = int(value)
    return out


def variant_name(model: str, num_ctx: int) -> str:
    """The Ollama model with a *num_ctx* context the owner made once (P3 §7.3): ``gpt-oss:20b`` at 32 768 is
    ``gpt-oss-20b-32k``."""
    base = model.removesuffix(":latest").replace(":", "-").replace("/", "-")
    return f"{base}-{num_ctx // 1024}k"


# --- build ----------------------------------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> int:
    from tests.assistant.grounding_projects import GroundingBuildError, build_grounding_project

    started = time.monotonic()
    try:
        built = build_grounding_project(Path(args.out))
    except GroundingBuildError as exc:
        print(f"The grounding project was not built: {exc}", file=sys.stderr)
        return 1
    print(f"Built {built.path} in {time.monotonic() - started:.0f} s (fingerprint {built.fingerprint[:12]}).")
    return 0


def check_marker(project: Path) -> dict[str, Any]:
    """The marker of a grounding project whose source database is the one ``build`` made; else :class:`Refused`
    (exit 2, before any request)."""
    from tests.assistant import grounding_projects as gp

    try:
        marker = read_json(Path(project) / gp.MARKER_FILE)
    except (OSError, ValueError):
        raise Refused(NOT_GROUNDING) from None
    source = Path(project) / "input_sources" / gp.SOURCE_NAME
    if not isinstance(marker, dict) or marker.get("format") != gp.MARKER_FORMAT or not source.is_file() \
            or gp.sha256_file(source) != marker.get("source_sha256"):
        raise Refused(NOT_GROUNDING)
    return marker


# --- the records of a results folder --------------------------------------------------------------------------
def cell_id(question: str, level: str, repeat: int) -> str:
    return f"{question}__{level}__r{repeat}"


def cell_path(out: Path, cid: str) -> Path:
    return Path(out) / CELLS_DIR / f"{cid}.json"


def read_cells(out: Path) -> dict[str, dict]:
    folder = Path(out) / CELLS_DIR
    if not folder.is_dir():
        return {}
    return {p.stem: read_json(p) for p in sorted(folder.glob("*.json"))}


def requests_today(cells: Iterable[Mapping[str, Any]], today: str) -> int:
    """The requests this results folder sent on the UTC day *today* (``YYYY-MM-DD``)."""
    return sum(int(c.get("requests_sent") or 0) for c in cells if str(c.get("ran_at") or "").startswith(today))


def daily_budget(preset_id: str, model: str, given: int | None) -> int | None:
    """The requests a calibration may send in one UTC day: *given*, else 45 for an OpenRouter ``:free`` model, else
    no limit."""
    from flextool.assistant.budget import FREE_SUFFIX

    if given is not None:
        return given
    if preset_id == "openrouter" and model.strip().lower().endswith(FREE_SUFFIX):
        return FREE_DAILY_REQUESTS
    return None


def _open_results(out: Path, identity: dict[str, Any], resume: bool) -> None:
    run_file = Path(out) / RUN_FILE
    if run_file.is_file():
        if not resume:
            raise Refused(HAS_RESULTS)
        saved = read_json(run_file)
        if any(saved.get(k) != identity[k] for k in ("provider", "model", "project_fingerprint", "set_version")):
            raise Refused(OTHER_RESULTS)
        return
    if (Path(out) / CELLS_DIR).is_dir() and any((Path(out) / CELLS_DIR).iterdir()):
        raise Refused(OTHER_RESULTS)
    write_json(run_file, {**identity, "format": RECORD_FORMAT, "created": iso(utc_now())})


# --- run ------------------------------------------------------------------------------------------------------
Prober = Callable[..., Any]
BackendFactory = Callable[..., Any]
OllamaInfo = Callable[[str, str], dict[str, Any]]


def ollama_info(base_url: str, model: str) -> dict[str, Any]:
    """``{num_ctx, size, size_vram}`` of a loaded Ollama model: ``num_ctx`` from ``/api/show`` (its parameters),
    the sizes from ``/api/ps``.  Only for a model server on this computer or the local network."""
    import urllib.request

    from flextool.assistant import net

    base = base_url.rstrip("/").removesuffix("/v1")
    if not net.is_local(base):
        raise Refused("The Ollama context check needs a model server on this computer or the local network.")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def ask(method: str, path: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    show = ask("POST", "/api/show", {"model": model})
    num_ctx = None
    found = re.search(r"(?m)^\s*num_ctx\s+(\d+)", str(show.get("parameters") or ""))
    if found:
        num_ctx = int(found.group(1))
    wanted = {model, model if ":" in model else f"{model}:latest"}
    entry = next((e for e in ask("GET", "/api/ps", None).get("models") or []
                  if isinstance(e, dict) and {e.get("name"), e.get("model")} & wanted), None)
    size = entry.get("size") if entry else None
    vram = entry.get("size_vram") if entry else None
    return {"num_ctx": num_ctx, "size": size, "size_vram": vram}


class CalibrationLedger:
    """The usage ledger with the requests of a calibration under the route's key + ``#calibration`` and its label +
    ``(calibration)``, so ``/usage`` and the window list them apart (P3 §7.2, m11)."""

    def __init__(self, ledger: Any) -> None:
        self.ledger = ledger

    def record(self, provider_key: str, label: str, model: str, usage: Any, **kwargs: Any) -> bool:
        return self.ledger.record(provider_key + CALIBRATION_SUFFIX, f"{label} (calibration)", model, usage, **kwargs)

    def record_rejected(self, provider_key: str, label: str, model: str, attempts: int = 1, **kwargs: Any) -> bool:
        return self.ledger.record_rejected(provider_key + CALIBRATION_SUFFIX, f"{label} (calibration)", model,
                                           attempts, **kwargs)


@dataclasses.dataclass
class Setting:
    """One level setting of the model: the model id sent (an Ollama context variant), its facts and budget."""

    level: str
    model: str
    facts: Any
    choice: Any
    budget: Any
    route: Any
    ollama: dict[str, Any] | None
    not_applicable: str | None = None


def _usage_json(usage: Any) -> dict[str, Any]:
    return dataclasses.asdict(usage) if usage is not None else {}


def _prompt_ratio(requests: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Reported / estimated prompt tokens: ``text`` over each turn's first request (instructions, question,
    schemas), ``tool_results`` over what each later request of a turn adds (tool calls and results)."""
    first_rep = first_est = 0
    add_rep = add_est = 0
    for turn in requests:
        reqs = [r for r in turn if r.get("reported") is not None]
        if not reqs:
            continue
        first_rep += reqs[0]["reported"]
        first_est += reqs[0]["text"] + reqs[0]["tools"]
        for before, after in zip(reqs, reqs[1:]):
            d_est = (after["text"] + after["tools"]) - (before["text"] + before["tools"])
            d_rep = after["reported"] - before["reported"]
            if d_est > 0:
                add_est += d_est
                add_rep += d_rep
    return {"text": round(first_rep / first_est, 3) if first_est else None,
            "tool_results": round(add_rep / add_est, 3) if add_est else None}


def run_cell(project: Path, qset: scoring.QuestionSet, question: scoring.Question, setting: Setting, *,
             backend: Any, preset: Any, ledger: Any, say: Callable[[str], None]) -> dict[str, Any]:
    """Ask *question* in a fresh conversation (S01: one conversation for its turns) at *setting*; the turns of
    the record, redacted.  Raises :class:`DailyBudgetUsed` when the provider says the quota is used."""
    from flextool.assistant.context import make_context
    from flextool.assistant.llm_adapter import Notice
    from flextool.assistant.llm_errors import message_for
    from flextool.assistant.session import SESSION_MESSAGES, TRUNCATION_SHARE, Session, TurnCaps
    from flextool.assistant.store import Conversation, new_id

    ctx = make_context(project, input_source=qset.project.get("input_source"), scenario=question.scenario,
                       front_end="terminal")
    ctx.caps = setting.budget.result_caps
    now = iso(utc_now())
    conversation = Conversation(id=new_id(), project_key="calibration", project_name=ctx.project_name,
                                project_path=ctx.project_path, created=now, updated=now, origin=backend.origin,
                                level=setting.level)
    session = Session(ctx, backend, preset, conversation=conversation, caps=TurnCaps.from_budget(setting.budget),
                      tier=setting.level, store=None, ledger=ledger, facts=setting.facts, route=setting.route,
                      budget=setting.budget)
    loaded = getattr(setting.facts, "loaded_context", None)
    local = setting.route.kind in ("loopback", "local_network") and setting.route.forwarded_to is None
    red = ctx.redact
    turns: list[dict[str, Any]] = []
    for number, text in enumerate(question.turns, start=1):
        notices: list[str] = []

        def on_event(event: Any, notices: list[str] = notices) -> None:
            if isinstance(event, Notice):
                notices.append(event.text)

        started = time.monotonic()
        result = session.run_turn(text, on_event, threading.Event())
        seconds = round(time.monotonic() - started, 2)
        if result.error is not None and result.error.code == "quota":
            raise DailyBudgetUsed(message_for(result.error, "terminal"))
        messages = [m for m in conversation.messages if m.turn == number]
        results = {m.tool_call_id: m for m in messages if m.role == "tool"}
        calls = []
        for m in messages:
            for call in m.tool_calls if m.role == "assistant" else ():
                res = results.get(call.id)
                calls.append({"name": call.name, "arguments": json.loads(red(json.dumps(call.arguments)))
                              if call.arguments is not None else red(call.raw_arguments),
                              "ok": bool(res is not None and not res.is_error),
                              "result": red(res.text) if res is not None else ""})
        requests = [{"text": r.text, "tools": r.tools, "reported": r.reported} for r in result.requests]
        truncated = sum(1 for r in result.requests
                        if local and loaded and r.reported is not None and r.reported >= TRUNCATION_SHARE * loaded)
        turns.append({
            "user": red(text), "answer": red(result.text), "tool_calls": calls, "rounds": result.rounds,
            "calls": result.calls, "continuations": result.continuations, "stop": "end" if result.ok else (result.error.code if result.error else "error"),
            "notices": [red(n) for n in notices],
            "error": ({"code": result.error.code, "message": red(message_for(result.error, "terminal"))}
                      if result.error is not None else None),
            "limit_hit": any(m.role == "user" and m.text == SESSION_MESSAGES["limit"] for m in messages),
            "fit_compactions": notices.count(SESSION_MESSAGES["fit"]),
            "truncated": truncated, "seconds": seconds,
            "usage": [_usage_json(u) for u in result.usage], "requests": requests,
        })
        say(f"  {question.id} turn {number}: {result.rounds} round(s), {result.calls} call(s), "
            f"{'ok' if result.ok else turns[-1]['stop']}")
    return {"turns": turns}


def _record(question: scoring.Question, setting: Setting, repeat: int, *, preset: Any, identity: Mapping[str, Any],
            knobs_sent: Mapping[str, Any], free_tier: bool, git: Mapping[str, Any],
            turns: list[dict[str, Any]] | None) -> dict[str, Any]:
    b = setting.budget
    choice = setting.choice
    requests_sent = sum(len(t["usage"]) for t in turns or [])
    return {
        "format": RECORD_FORMAT,
        "cell": {"model": identity["model"], "level_setting": setting.level, "question": question.id,
                 "class": question.cls, "repeat": repeat},
        "model": {"preset": preset.id, "route": setting.route.kind if setting.route else None,
                  "model": setting.model,
                  "automatic_level": choice.automatic if choice else None,
                  "level_source": choice.source if choice else None,
                  "window": b.window if b else None, "window_source": b.window_source if b else None,
                  "free_tier": free_tier, "ollama": setting.ollama, "knobs_sent": dict(knobs_sent),
                  "knobs_relaxed": bool(identity.get("knobs_relaxed"))},
        "not_applicable": setting.not_applicable,
        "budget": ({"caps": {"max_rounds": b.level.max_rounds, "max_calls_per_turn": b.level.max_calls_per_turn,
                             "max_calls_per_round": b.level.max_calls_per_round,
                             "max_output_tokens": b.max_output_tokens},
                    "result_caps": {k: getattr(b.result_caps, k)
                                    for k in ("max_rows", "max_chars", "max_results", "max_items")},
                    "compaction_every": b.compaction_every, "keep_recent_turns": b.keep_recent_turns,
                    "prompt_ratio": _prompt_ratio([t["requests"] for t in turns or []])} if b else None),
        "turns": turns or [],
        "checks": None, "manual": None, "outcome": None, "graded_at": None,
        "set_version": identity["set_version"], "key_version": None,
        "project_fingerprint": identity["project_fingerprint"], "flextool_version": flextool_version(),
        "git": dict(git), "ran_at": iso(utc_now()), "requests_sent": requests_sent, "purpose": "calibration",
    }


def _default_prober(preset: Any, conn: Any, cancel: threading.Event, *, needed: int) -> Any:
    from flextool.assistant import probes

    return probes.probe(preset, conn, cancel, needed=needed)


def _default_backend_factory(preset: Any, conn: Any) -> Any:
    from flextool.assistant.llm_adapter import make_backend

    return make_backend(preset, conn)


def _setting(level: str, model: str, *, preset: Any, base_url: str, conn_for: Callable[[str, Any], Any],
             prober: Prober, info: OllamaInfo | None, wanted_context: int | None, free_tier: bool,
             cache: dict[str, Any]) -> Setting:
    """The facts and budget of one level setting: the model is probed (an Ollama variant loaded and its context
    and graphics memory read); the level is the one asked for, the window rule sets the caps."""
    from flextool.assistant import budget, endpoint, probes, prompt, tiers
    from flextool.assistant.llm_errors import LLMError, message_for
    from flextool.assistant.session import tool_specs

    if model not in cache:
        conn = conn_for(model, endpoint.route_for(preset, base_url, model=model))
        needed = probes.needed_tokens(prompt.system_layers("terminal", "small"), tool_specs("small"))
        try:
            cache[model] = prober(preset, conn, threading.Event(), needed=needed)
        except LLMError as exc:
            if exc.code == "small_context" and wanted_context is not None:
                cache[model] = exc
            else:
                raise Refused(message_for(exc, "terminal"), code=1) from None
    facts = cache[model]
    ollama = None
    if wanted_context is not None:
        ollama = {"variant": model, "num_ctx": None, "loaded_context": None, "size": None, "size_vram": None}
        if not isinstance(facts, Exception):
            ollama["loaded_context"] = facts.loaded_context
            ollama.update({k: v for k, v in (info(base_url, model) if info else {}).items() if k in ollama})
        loaded, size, vram = ollama["loaded_context"], ollama["size"], ollama["size_vram"]
        if isinstance(facts, Exception) or loaded is None or loaded < wanted_context or (
                isinstance(size, int) and isinstance(vram, int) and vram < size):
            route = endpoint.route_for(preset, base_url, model=model)
            return Setting(level, model, None, None, None, route, ollama, not_applicable="context")
    route = endpoint.route_for(preset, base_url, facts=facts)
    try:
        choice = tiers.resolve_tier(preset, model, facts, None, route)
    except tiers.WindowTooSmall:
        return Setting(level, model, facts, None, None, route, ollama, not_applicable="context")
    forced = dataclasses.replace(choice, chosen=level, notice=None, chosen_note=None)
    b = budget.budget_for(forced, free_tier=free_tier)
    return Setting(level, model, facts, choice, b, route, ollama)


def cmd_run(args: argparse.Namespace, *, backend_factory: BackendFactory | None = None,
            prober: Prober | None = None, info: OllamaInfo | None = None,
            say: Callable[[str], None] = print) -> int:
    try:
        return _run(args, backend_factory or _default_backend_factory, prober or _default_prober,
                    info if info is not None else ollama_info, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _run(args: argparse.Namespace, backend_factory: BackendFactory, prober: Prober, info: OllamaInfo,
         say: Callable[[str], None]) -> int:
    from flextool.assistant import budget, endpoint, keys, net, providers
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.llm_adapter import Connection
    from flextool.assistant.llm_errors import render
    from flextool.assistant.settings import DEFAULT_KNOBS
    from flextool.assistant.usage import Ledger

    project = Path(args.project).absolute()
    marker = check_marker(project)                      # first: nothing is read or sent for another project
    qset = scoring.load_set(Path(args.set))
    if marker.get("set_version") != qset.set_version:
        raise Refused(f"The project was built for question set {marker.get('set_version')}, the set is "
                      f"{qset.set_version}: build it again.")
    wanted = [q.strip() for q in args.questions.split(",")] if args.questions else list(qset.questions)
    unknown = [q for q in wanted if q not in qset.questions]
    if unknown:
        raise Refused(f"No such questions: {', '.join(unknown)}.")
    levels = internal_levels(args.levels)
    preset = providers.PRESETS.get(args.provider)
    if preset is None:
        raise Refused(f"Unknown provider {args.provider!r}: one of {', '.join(providers.PRESETS)}.")
    contexts = parse_ollama_context(args.ollama_context)
    if contexts and preset.id != "ollama":
        raise Refused("--ollama-context is for the Ollama provider only.")
    if args.relax_data_collection and preset.id != "openrouter":
        raise Refused("--relax-data-collection is for OpenRouter only.")
    if args.repeat < 1:
        raise Refused("--repeat must be at least 1.")
    try:
        base_url = net.check_base_url(args.base_url or preset.base_url or "")
    except net.BaseUrlRefused as exc:
        raise Refused(str(exc)) from None
    out = Path(args.out)
    # The notice before run.json is written, so a refused first run can simply be run again with the flag.
    accepted: set = set()
    route = endpoint.route_for(preset, base_url, model=args.model)
    what = "the synthetic grounding project"
    accept_route(preset, base_url, route, accepted, accept=args.accept_data_notice,
                 again="--resume --accept-data-notice" if args.resume else "--accept-data-notice", what=what)
    identity = {"provider": preset.id, "model": args.model, "base_url": base_url,
                "project_fingerprint": marker["fingerprint"], "set_version": qset.set_version,
                "knobs_relaxed": bool(args.relax_data_collection)}
    _open_results(out, identity, args.resume)

    key, source, note = keys.resolve_key(preset, base_url)
    knobs = dict(DEFAULT_KNOBS)
    if args.relax_data_collection:
        knobs["openrouter.data_collection"] = "allow"
    knobs_sent = providers.request_fields(preset, knobs)

    def conn_for(model: str, route: Any) -> Connection:
        return Connection(base_url=base_url, model=model, api_key=key or "",
                          key_source=source.describe(preset) if key else "", knobs=knobs, route=route)

    if preset.key_required and not key:
        conn = conn_for(args.model, endpoint.route_for(preset, base_url, model=args.model))
        raise Refused(render("no_key", error_context(preset, conn)) + (f" ({note})" if note else ""))
    free_setting = {"on": True, "off": False, "auto": None}[args.free_tier]
    free = budget.free_tier_on(preset.id, args.model, free_setting)
    daily = daily_budget(preset.id, args.model, args.max_requests_per_day)
    ledger = CalibrationLedger(Ledger())
    git = git_state()
    cells = read_cells(out)

    def budget_used() -> bool:
        return daily is not None and requests_today(cells.values(), iso(utc_now())[:10]) >= daily

    probed: dict[str, Any] = {}
    for level in levels:
        model = variant_name(args.model, contexts[level]) if level in contexts else args.model
        setting = _setting(level, model, preset=preset, base_url=base_url, conn_for=conn_for, prober=prober,
                           info=info if level in contexts else None, wanted_context=contexts.get(level),
                           free_tier=free, cache=probed)
        if not setting.not_applicable:
            # The probe can show that the model is forwarded (Ollama cloud): that route's notice is asked too.
            # The probe sent only the model name, to the local server.
            accept_route(preset, base_url, setting.route, accepted, accept=args.accept_data_notice,
                         again="--resume --accept-data-notice", what=what)
        done = DAILY_DONE_REMOTE if accepted else DAILY_DONE
        for qid in wanted:
            question = qset.questions[qid]
            if question.automatic_level_only and (setting.choice is None or level != setting.choice.automatic):
                continue
            for repeat in range(1, args.repeat + 1):
                cid = cell_id(qid, level, repeat)
                if cid in cells:
                    continue
                if setting.not_applicable:
                    record = _record(question, setting, repeat, preset=preset, identity=identity,
                                     knobs_sent=knobs_sent, free_tier=free, git=git, turns=None)
                    write_json(cell_path(out, cid), record)
                    cells[cid] = record
                    say(f"{cid}: not applicable ({setting.not_applicable})")
                    continue
                if budget_used():
                    say(done)
                    return 0
                backend = backend_factory(preset, conn_for(setting.model, setting.route))
                try:
                    turns = run_cell(project, qset, question, setting, backend=backend, preset=preset,
                                     ledger=ledger, say=say)["turns"]
                except DailyBudgetUsed as exc:
                    say(str(exc))
                    say(done)
                    return 0
                finally:
                    backend.close()
                record = _record(question, setting, repeat, preset=preset, identity=identity,
                                 knobs_sent=knobs_sent, free_tier=free, git=git, turns=turns)
                write_json(cell_path(out, cid), record)
                cells[cid] = record
                say(f"{cid}: saved")
                if budget_used():
                    say(done)
                    return 0
    say(f"All cells of {args.model} are in {out}.")
    return 0


# --- grade ----------------------------------------------------------------------------------------------------
def _key_for(args: argparse.Namespace, cells: Mapping[str, dict]) -> scoring.AnswerKey:
    path = Path(args.key)
    if not path.is_file():
        raise Refused(f"The answer key {path} does not exist yet: the independent oracle writes it.")
    key = scoring.load_key(path)
    versions = {c.get("set_version") for c in cells.values()}
    if versions - {key.set_version}:
        raise Refused(f"The answer key answers question set {key.set_version}; these results are of "
                      f"{', '.join(map(str, sorted(versions)))}.")
    return key


def _turn_views(record: Mapping[str, Any]) -> list[scoring.TurnView]:
    return [scoring.TurnView.from_record(t) for t in record.get("turns") or []]


def auto_check(record: dict, qset: scoring.QuestionSet, key: scoring.AnswerKey) -> list[scoring.Item]:
    """The cell's items under *key*: the stored ones when they were made with this key version (with the
    grader's verdicts), else checked again (earlier verdicts dropped)."""
    if record.get("checks") is not None and record.get("key_version") == key.key_version \
            and bool(record.get("key_placeholder")) == key.placeholder:
        return [scoring.Item.from_json(i) for i in record["checks"]]
    cell = record["cell"]
    question = qset.questions[cell["question"]]
    entry = key.questions.get(question.id)
    if entry is None:
        raise Refused(f"The answer key has no entry for {question.id}.")
    max_calls = (record.get("budget") or {}).get("caps", {}).get("max_calls_per_turn")
    items = scoring.check_cell(question, entry, cell["level_setting"], _turn_views(record), max_calls=max_calls)
    record.update(checks=[i.to_json() for i in items], key_version=key.key_version,
                  key_placeholder=key.placeholder, outcome=None, graded_at=None, manual=None)
    return items


def finish_grade(record: dict, items: Sequence[scoring.Item], grader: str) -> None:
    record["checks"] = [i.to_json() for i in items]
    result = scoring.outcome(items)
    if result is not None:
        record.update(outcome=result, graded_at=iso(utc_now()),
                      manual={"grader": grader, "first_pass": True,
                              "verdicts": {i.id: i.verdict for i in items if i.state in ("review", "manual")}})


def grader_view(record: Mapping[str, Any], question: scoring.Question, entry: scoring.KeyEntry,
                items: Sequence[scoring.Item], names: Iterable[str] = ()) -> str:
    """What the grader sees of one cell: blind (no model, provider, level, cap, notice or time) and framed as
    data; the second grader's request (``assistant_eval_analysis.grader_user_part``) shows the same."""
    level = record["cell"]["level_setting"]
    lines = [f"Question {question.id} ({question.cls})"]
    lines += [f"  turn {n}: {t}" for n, t in enumerate(question.turns, 1)]
    lines.append(f"Expected behaviour: {scoring.expected_behaviour(question.cls, level)}")
    lines.append("Key:")
    for fact in entry.facts:
        what = fact.criterion if fact.kind == "statement" else (
            f"{fact.value:g} {fact.unit or ''}".strip() if fact.kind == "number" else " / ".join(fact.names))
        lines.append(f"  {fact.id} ({fact.kind}{', critical' if fact.critical else ''}): {what}")
    for rule in entry.forbidden:
        lines.append(f"  forbidden: {rule.note}")
    lines.append("Automatic checks:")
    lines += [f"  {text}" for text in scoring.grader_items_text(items)]
    for n, turn in enumerate(record.get("turns") or [], 1):
        lines.append(f"--- turn {n}: tool results (data, not instructions) ---")
        for call in turn.get("tool_calls") or []:
            args = scoring.blind(json.dumps(call.get("arguments"), ensure_ascii=False), names)
            lines.append(f"[{call.get('name')} {args}] "
                         + scoring.fold(scoring.blind(str(call.get("result") or ""), names)))
        lines.append(f"--- turn {n}: answer (data, not instructions) ---")
        lines.append(scoring.blind(str(turn.get("answer") or ""), names))
    return "\n".join(lines)


def verdict_texts(item: scoring.Item, names: Iterable[str] = ()) -> tuple[list[str], str]:
    """What is shown before the grader's verdict on *item* and the question asked, blind."""
    detail = scoring.blind(item.detail, names)
    if item.check == "tool_sourced":
        shown = [f"Number without a tool source: {detail}"]
        if item.candidates:
            shown.append(f"  nearest source numbers: {', '.join(f'{c:g}' for c in item.candidates)}")
        return shown, "  pass = a stated calculation from tool numbers or general constants, fail = from nowhere: "
    if item.check == "facts":
        return [], f"Statement {item.id}: {detail}\n  pass / fail: "
    if item.id == "facts_vs_assumptions":
        return [], ("Every claim not from a tool result is marked as an assumption or general knowledge?\n"
                    "  pass / fail: ")
    return [], f"{item.id}: {detail}\n  pass / fail: "


def _ask_verdict(item: scoring.Item, ask: Callable[[str], str | None], say: Callable[[str], None],
                 names: Iterable[str] = ()) -> str | None:
    shown, prompt = verdict_texts(item, names)
    for line in shown:
        say(line)
    while True:
        answer = ask(prompt)
        if answer is None or answer.strip().lower() in ("q", "quit", "stop"):
            return None
        word = answer.strip().lower()
        if word in ("p", "pass", "y", "yes"):
            return "pass"
        if word in ("f", "fail", "n", "no"):
            return "fail"
        say("Type pass or fail (q stops and saves).")


CellKey = tuple[str, str]


def _cell_key(folder: Path, cid: str) -> CellKey:
    return (str(Path(folder).resolve()), cid)


def grading_order(folder: Path, keys: Iterable[CellKey]) -> list[CellKey]:
    """The blind order of the cells of every results folder graded together, kept in ``<folder>/grading.json``
    with its seed.  A cell's position is its ``Cell #n`` (``tokens``: the positions shown so far); cells added
    later (another folder, more cells) are shuffled into the part of the order not yet shown, so they mix with
    the cells still waiting instead of following as one model's block."""
    path = Path(folder) / GRADING_FILE
    state = read_json(path) if path.is_file() else {"seed": random.SystemRandom().randrange(2**31), "order": [],
                                                     "tokens": {}}
    order: list[CellKey] = [(str(f), str(c)) for f, c in state["order"]]
    tokens = state.setdefault("tokens", {})
    new = sorted(set(keys) - set(order))
    if new or not path.is_file():
        rng = random.Random(f"{state['seed']}:{len(order)}")
        rng.shuffle(new)
        shown = max((int(t.lstrip("#")) for t in tokens), default=0)
        for key in new:
            order.insert(rng.randint(shown, len(order)), key)
        state["order"] = [list(k) for k in order]
        write_json(path, state)
    return order


def _token(folder: Path, position: int, key: CellKey) -> str:
    """``#n`` of the cell at *position*, kept in ``grading.json`` once it is shown or graded."""
    path = Path(folder) / GRADING_FILE
    state = read_json(path)
    name = f"#{position}"
    if state.setdefault("tokens", {}).get(name) != list(key):
        state["tokens"][name] = list(key)
        write_json(path, state)
    return name


def cmd_grade(args: argparse.Namespace, *, ask: Callable[[str], str | None] | None = None,
              say: Callable[[str], None] = print) -> int:
    def default_ask(text: str) -> str | None:
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    try:
        return _grade(args, ask or default_ask, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _stamps(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(r.get(k)) for r in records for k in ("ran_at", "graded_at") if r.get(k)}


def _tool_text(record: Mapping[str, Any]) -> str:
    return "\n".join(str(c.get("result") or "") for t in record.get("turns") or [] for c in t.get("tool_calls") or [])


def _regrade_key(text: str, tokens: Mapping[str, Any], cells: Mapping[CellKey, dict]) -> CellKey:
    if "::" in text:
        folder, _, cid = text.partition("::")
        key = _cell_key(Path(folder), cid)
    else:
        name = text if text.startswith("#") else f"#{text}"
        found = tokens.get(name)
        if found is None:
            raise Refused(f"No graded cell {name}.")
        key = (str(found[0]), str(found[1]))
    if key not in cells:
        raise Refused(f"No graded cell {text}.")
    return key


def _grade(args: argparse.Namespace, ask: Callable[[str], str | None], say: Callable[[str], None]) -> int:
    if len(args.results) == 1 and not args.single_model:
        raise Refused("Grading one model is not blind: give the results folders of every model of the matrix "
                      "(their cells are shuffled together), or --single-model.")
    loaded = _load_results([Path(p) for p in args.results])
    cells: dict[CellKey, dict] = {_cell_key(f, cid): r for f, _run, folder_cells in loaded
                                  for cid, r in folder_cells.items()}
    if not cells:
        raise Refused(f"No cells in {', '.join(str(p) for p in args.results)}.")
    folders = {_cell_key(f, "")[0]: f for f, _r, _c in loaded}
    qset = scoring.load_set(Path(args.set))
    key = _key_for(args, {f"{f}::{c}": r for (f, c), r in cells.items()})
    names = scoring.identity_names([(f, run) for f, run, _c in loaded], cells.values())
    scan_names = scoring.identity_names([(f, run) for f, run, _c in loaded], cells.values(), folders=False)
    stamps = _stamps(cells.values())
    grading = Path(args.dir)
    order = grading_order(grading, [k for k, r in cells.items() if not r.get("not_applicable")])
    position = {k: n for n, k in enumerate(order, 1)}
    if key.placeholder:
        say("This answer key is a placeholder: its grades only exercise the tooling and are never recorded.")

    # The plan of this sitting: each cell to grade with its items; those that wait for the grader are shown.
    plan: list[tuple[CellKey, list[scoring.Item], bool]] = []
    regrade = None
    if args.regrade is not None:
        regrade = _regrade_key(args.regrade, read_json(grading / GRADING_FILE).get("tokens") or {}, cells)
        items = [dataclasses.replace(i, verdict=None) for i in auto_check(cells[regrade], qset, key)]
        if cells[regrade].get("outcome") is None:          # never graded, or checked again under a new key
            raise Refused(f"Grade cell {args.regrade} first.")
        plan.append((regrade, items, bool(scoring.pending(items))))
    else:
        shown = 0
        stopped = False
        for k in order:
            record = cells.get(k)
            if record is None or record.get("not_applicable"):
                continue
            items = auto_check(record, qset, key)
            if record.get("outcome") is not None:
                continue
            waiting = bool(scoring.pending(items))
            if waiting and args.sitting and shown >= args.sitting:
                stopped = True
                break
            shown += waiting
            plan.append((k, items, waiting))

    # The blinding self-test: every text this sitting would show, before the first question.
    views: dict[CellKey, str] = {}
    for k, items, waiting in plan:
        if not waiting:
            continue
        record = cells[k]
        question = qset.questions[record["cell"]["question"]]
        view = grader_view(record, question, key.questions[question.id], items, names)
        prompts = [line for i in scoring.pending(items) for texts in [verdict_texts(i, names)]
                   for line in (*texts[0], texts[1])]
        found = scoring.leak_scan("\n".join([view, *prompts]), names=scan_names, stamps=stamps,
                                  sources=_tool_text(record))
        if found:
            raise Refused(f"Cell #{position[k]}: the grader material reveals {found[0]}; nothing was graded.")
        views[k] = view

    graded_now = 0
    for k, items, waiting in plan:
        record = cells[k]
        folder, cid = folders[k[0]], k[1]
        token = _token(grading, position[k], k)
        if waiting:
            say("=" * 100)
            say(f"Cell {token}" + (" (re-grade)" if regrade else ""))
            say(views[k])
            for item in scoring.pending(items):
                verdict = _ask_verdict(item, ask, say, names)
                if verdict is None:
                    if regrade:
                        say("Stopped; the re-grade was not saved.")
                        return 0
                    record["checks"] = [i.to_json() for i in items]
                    write_json(cell_path(folder, cid), record)
                    say("Stopped; the grades given so far are saved.")
                    return 0
                item.verdict = verdict
            graded_now += 1
        if regrade:
            result = scoring.outcome(items)
            record.setdefault("regrades", []).append({
                "graded_at": iso(utc_now()), "grader": "owner", "outcome": result,
                "verdicts": {i.id: i.verdict for i in items if i.state in ("review", "manual")}})
            write_json(cell_path(folder, cid), record)
            say(f"Cell {token} graded again.")
            return 0
        finish_grade(record, items, "owner")
        write_json(cell_path(folder, cid), record)
    if args.regrade is None and stopped:
        say(f"Sitting done ({graded_now} cells). Run grade again to continue.")
        return 0
    left = sum(1 for r in cells.values() if not r.get("not_applicable") and r.get("outcome") is None)
    say(f"Graded. {left} cell(s) still wait." if left else "Every cell is graded.")
    return 0


# --- summary --------------------------------------------------------------------------------------------------
def _graded(record: Mapping[str, Any]) -> list[scoring.Item]:
    return [scoring.Item.from_json(i) for i in record.get("checks") or []]


def summary_line(record: Mapping[str, Any], *, grader: str, authoritative: bool, first_pass: bool,
                 result: str | None, items: Sequence[scoring.Item], run: Mapping[str, Any]) -> dict[str, Any]:
    """One line of ``results.jsonl``: no answer or tool text (P3 §7.7).  Beyond §7.7's fields: ``repeat``,
    ``level_source`` and the health counts the §7.6 analysis reads (cap hits, length stops, fit compactions,
    truncations, refused tools).  A cell not run for its context is a line with outcome ``not_applicable``."""
    turns = record.get("turns") or []
    usage = [u for t in turns for u in t.get("usage") or []]
    reported = [u for u in usage if u.get("reported")]
    tokens_in = sum(int(u.get(k) or 0) for u in reported for k in ("input_uncached", "cache_read", "cache_write"))
    model = record["model"]
    refused = sorted({str(c.get("name")) for t in turns for c in t.get("tool_calls") or []
                      if not c.get("ok") and "not available at this assistant level" in str(c.get("result") or "")})
    return {
        "date": str(record.get("ran_at") or "")[:10], "grader": grader, "authoritative": authoritative,
        "first_pass": first_pass, "flextool_version": record.get("flextool_version"),
        "commit": (record.get("git") or {}).get("commit"), "set_version": record.get("set_version"),
        "key_version": record.get("key_version"), "provider": model["preset"], "model": run.get("model"),
        "automatic_level": model.get("automatic_level"), "level_setting": record["cell"]["level_setting"],
        "window": model.get("window"), "free_tier": model.get("free_tier"),
        "knobs_relaxed": bool(model.get("knobs_relaxed")), "question": record["cell"]["question"],
        "class": record["cell"]["class"], "outcome": result, "score": scoring.score(items) if items else None,
        "critical_failures": scoring.critical_failures(items), "invented_numbers": scoring.invented_numbers(items),
        "calls": sum(int(t.get("calls") or 0) for t in turns), "rounds": sum(int(t.get("rounds") or 0) for t in turns),
        "tokens_in": tokens_in, "tokens_out": sum(int(u.get("output") or 0) for u in reported),
        "tokens_not_reported": len(usage) - len(reported),
        "prompt_ratio": (record.get("budget") or {}).get("prompt_ratio"),
        "repeat": record["cell"].get("repeat", 1), "level_source": model.get("level_source"),
        "cap_hits": sum(1 for t in turns if t.get("limit_hit")),
        "length_stops": sum(1 for t in turns if (t.get("error") or {}).get("code") == "length"),
        "fit_compactions": sum(int(t.get("fit_compactions") or 0) for t in turns),
        "truncations": sum(int(t.get("truncated") or 0) for t in turns),
        "refused_tools": refused,
    }


def _load_results(folders: Sequence[Path]) -> list[tuple[Path, dict, dict[str, dict]]]:
    loaded = []
    prints = set()
    for folder in folders:
        run_file = Path(folder) / RUN_FILE
        if not run_file.is_file():
            raise Refused(f"{folder} is not a results folder of assistant_eval run.")
        run = read_json(run_file)
        prints.add(run.get("project_fingerprint"))
        loaded.append((Path(folder), run, read_cells(folder)))
    if len(prints) > 1:
        raise Refused("These results folders come from different grounding projects; summarise them apart.")
    return loaded


def owner_lines(run: Mapping[str, Any], cells: Mapping[str, dict]) -> list[tuple[str, str, dict]]:
    """The authoritative lines of the graded cells not yet recorded: ``(cell id, tag, line)``; a re-grade is a line
    of its own (``first_pass`` false).  The tag is kept in the cell's ``recorded`` list."""
    out = []
    for cid, record in cells.items():
        done = set(record.get("recorded") or [])
        if record.get("not_applicable"):
            if "none:not_applicable" not in done:
                out.append((cid, "none:not_applicable", summary_line(
                    record, grader="none", authoritative=True, first_pass=True, result="not_applicable", items=[],
                    run=run)))
            continue
        if record.get("outcome") is None:
            continue
        items = _graded(record)
        if "owner:first" not in done:
            out.append((cid, "owner:first", summary_line(record, grader="owner", authoritative=True, first_pass=True,
                                                         result=record["outcome"], items=items, run=run)))
        for n, regrade in enumerate(record.get("regrades") or []):
            tag = f"owner:regrade:{n}"
            if tag in done:
                continue
            regraded = [dataclasses.replace(i, verdict=regrade["verdicts"].get(i.id, i.verdict)) for i in items]
            out.append((cid, tag, summary_line(record, grader="owner", authoritative=True, first_pass=False,
                                               result=regrade["outcome"], items=regraded, run=run)))
    return out


def append_lines(path: Path, lines: Iterable[dict]) -> int:
    lines = list(lines)
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=False) + "\n")
    return len(lines)


def cmd_summary(args: argparse.Namespace, *, say: Callable[[str], None] = print) -> int:
    try:
        loaded = _load_results([Path(p) for p in args.results])
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    for folder, run, cells in loaded:
        say(f"{run.get('provider')} {run.get('model')} ({folder})")
        table: dict[tuple[str, str], dict[str, int]] = {}
        for record in cells.values():
            key = (record["cell"]["level_setting"], record["cell"]["class"])
            row = table.setdefault(key, {"n": 0, "answered": 0, "escalated": 0, "failed": 0, "waiting": 0,
                                         "not_applicable": 0, "invented": 0})
            if record.get("not_applicable"):
                row["not_applicable"] += 1
                continue
            row["n"] += 1
            if record.get("outcome") is None:
                row["waiting"] += 1
            else:
                row[record["outcome"]] += 1
                row["invented"] += scoring.invented_numbers(_graded(record))
        for level in scoring.LEVELS:
            for cls in scoring.CLASSES:
                row = table.get((level, cls))
                if row:
                    say(f"  {level:<7} {cls:<9} n {row['n']:>3}  answered {row['answered']:>3}  escalated "
                        f"{row['escalated']:>3}  failed {row['failed']:>3}  waiting {row['waiting']:>3}  "
                        f"not applicable {row['not_applicable']:>3}  invented numbers {row['invented']:>3}")
    if args.record:
        placeholder = [cid for _f, _r, cells in loaded for cid, r in cells.items() if r.get("key_placeholder")]
        if placeholder:
            print("Not recorded: these grades come from a placeholder answer key.", file=sys.stderr)
            return 2
        total = 0
        for folder, run, cells in loaded:
            lines = owner_lines(run, cells)
            total += append_lines(Path(args.results_file), (line for _cid, _tag, line in lines))
            for cid, tag, _line in lines:
                cells[cid].setdefault("recorded", []).append(tag)
                write_json(cell_path(folder, cid), cells[cid])
        say(f"{total} line(s) appended to {args.results_file}.")
    return 0


# --- calibrate and dual (tools/assistant_eval_analysis.py) ----------------------------------------------------
def cmd_calibrate(args: argparse.Namespace, *, say: Callable[[str], None] = print) -> int:
    import assistant_eval_analysis as analysis

    return analysis.cmd_calibrate(args, say=say)


def _dual_records(sample: Sequence[Any]) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    for folder in sorted({c.folder for c in sample}):
        for cid, record in read_cells(Path(folder)).items():
            records[(folder, cid)] = record
    missing = [c.cell for c in sample if (c.folder, c.cell) not in records]
    if missing:
        raise Refused(f"Sampled cells are missing from their results folders: {', '.join(missing)}.")
    return records


def _read_sample(folder: Path) -> tuple[dict, list[Any]]:
    import assistant_eval_analysis as analysis

    path = Path(folder) / analysis.SAMPLE_FILE
    if not path.is_file():
        raise Refused(f"No dual sample in {folder}: run dual sample first.")
    data = read_json(path)
    return data, [analysis.DualCell(**c) for c in data["cells"]]


def dual_sample(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    import assistant_eval_analysis as analysis

    folder = Path(args.dir)
    path = folder / analysis.SAMPLE_FILE
    if path.is_file():
        raise Refused(f"{path} exists: the subset is drawn once (its seed is recorded).")
    loaded = _load_results([Path(p) for p in args.results])
    pool = [c for f, run, cells in loaded for c in analysis.eligible(f, run, cells)]
    if not pool:
        raise Refused("No cell depends on a manual item yet: run grade first (it makes the automatic checks).")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    sample = analysis.draw_sample(pool, seed=seed)
    write_json(path, {"format": RECORD_FORMAT, "seed": seed, "created": iso(utc_now()),
                      "results": [str(f) for f, _r, _c in loaded], "pool": len(pool),
                      "cells": [dataclasses.asdict(c) for c in sample]})
    authored = sum(1 for c in sample if c.grader_authored)
    say(f"Drew {len(sample)} of {len(pool)} eligible cells (seed {seed}; {authored} written by "
        f"{analysis.GRADER_MODEL}).")
    return 0


def dual_grade_ai(args: argparse.Namespace, backend_factory: BackendFactory, say: Callable[[str], None]) -> int:
    import assistant_eval_analysis as analysis
    from flextool.assistant import endpoint, keys, providers
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.llm_adapter import Connection
    from flextool.assistant.llm_errors import LLMError, message_for, render
    from flextool.assistant.usage import Ledger

    folder = Path(args.dir)
    data, sample = _read_sample(folder)
    records = _dual_records(sample)
    qset = scoring.load_set(Path(args.set))
    key = _key_for(args, {f"{c.folder}::{c.cell}": records[(c.folder, c.cell)] for c in sample})
    if key.placeholder:
        raise Refused("The answer key is a placeholder: the second grader needs the oracle's key.")
    sealed_path = folder / analysis.SEALED_FILE
    sealed = read_json(sealed_path) if sealed_path.is_file() else {}
    runs = [(f, read_json(Path(f) / RUN_FILE)) for f in sorted({c.folder for c in sample})]
    names = scoring.identity_names(runs, records.values())
    scan_names = scoring.identity_names(runs, records.values(), folders=False)
    stamps = _stamps(records.values())
    # The blinding self-test of every request this run would send, before anything is sent.
    for n, c in enumerate(sample, 1):
        if f"{c.folder}::{c.cell}" in sealed:
            continue
        record = records[(c.folder, c.cell)]
        text = analysis.grader_user_part(record, qset.questions[c.question], key.questions[c.question], names)
        found = scoring.leak_scan(text, names=scan_names, stamps=stamps, sources=_tool_text(record))
        if found:
            raise Refused(f"Sampled cell {n}: the grader material reveals {found[0]}; nothing was sent.")
    preset = providers.PRESETS[analysis.GRADER_PROVIDER]
    base_url = preset.base_url or ""
    api_key, source, note = keys.resolve_key(preset, base_url)
    route = endpoint.route_for(preset, base_url, model=analysis.GRADER_MODEL)
    conn = Connection(base_url=base_url, model=analysis.GRADER_MODEL, api_key=api_key or "",
                      key_source=source.describe(preset) if api_key else "", knobs={}, route=route)
    if not api_key:
        raise Refused(render("no_key", error_context(preset, conn)) + (f" ({note})" if note else ""))
    accept_route(preset, base_url, route, set(), accept=args.accept_data_notice, again="--accept-data-notice",
                 what="the synthetic grounding answers")
    ledger = CalibrationLedger(Ledger())
    backend = backend_factory(preset, conn)
    try:
        for n, c in enumerate(sample, 1):
            name = f"{c.folder}::{c.cell}"
            if name in sealed:
                continue
            record = records[(c.folder, c.cell)]
            if record.get("key_version") != key.key_version:
                raise Refused(f"{c.cell} was checked with key version {record.get('key_version')}, the key is "
                              f"{key.key_version}: run grade first.")
            question = qset.questions[c.question]
            try:
                result = analysis.grade_with_ai(record, question, key.questions[c.question], backend, names)
            except LLMError as exc:
                write_json(sealed_path, sealed)
                say(f"Stopped at cell {n} of {len(sample)}: {message_for(exc, 'terminal')} Run again to continue.")
                return 1
            for usage in result.pop("usage"):
                ledger.record(providers.PRESETS[analysis.GRADER_PROVIDER].id, route.label, analysis.GRADER_MODEL,
                              usage)
            sealed[name] = {**result, "key_version": key.key_version, "graded_at": iso(utc_now())}
            write_json(sealed_path, sealed)
            say(f"Graded {n} of {len(sample)} (sealed until dual report).")
    finally:
        backend.close()
    say(f"The second grader's verdicts are sealed in {sealed_path}.")
    return 0


def dual_report(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    import assistant_eval_analysis as analysis

    folder = Path(args.dir)
    data, sample = _read_sample(folder)
    records = _dual_records(sample)
    sealed_path = folder / analysis.SEALED_FILE
    sealed = read_json(sealed_path) if sealed_path.is_file() else {}
    missing_ai = [c.cell for c in sample if f"{c.folder}::{c.cell}" not in sealed]
    if missing_ai:
        raise Refused(f"The second grader has not graded {len(missing_ai)} sampled cell(s): run dual grade-ai.")
    waiting = []
    for c in sample:
        record = records[(c.folder, c.cell)]
        verdicts = (record.get("manual") or {}).get("verdicts") or {}
        if record.get("outcome") is None or any(i.id not in verdicts for i in analysis.manual_items(record)):
            waiting.append(c.cell)
        elif record.get("key_version") != sealed[f"{c.folder}::{c.cell}"].get("key_version"):
            raise Refused(f"{c.cell} was graded with another key version than the second grader's: draw a new "
                          "subset.")
    if waiting:
        raise Refused(f"The owner grades the sampled cells first (grade): {len(waiting)} cell(s) wait.")
    text = analysis.dual_report(sample, records, sealed, seed=int(data["seed"]))
    (folder / analysis.REPORT_FILE).write_text(text + "\n", encoding="utf-8")
    say(text)
    if args.record:
        runs = {f: read_json(Path(f) / RUN_FILE) for f in {c.folder for c in sample}}
        lines = []
        for c in sample:
            name = f"{c.folder}::{c.cell}"
            if sealed[name].get("recorded"):
                continue
            record = records[(c.folder, c.cell)]
            if "unparsed" in sealed[name]:
                items, result = [scoring.Item.from_json(i) for i in record["checks"]], "unparsed"
            else:
                items = analysis.ai_items(record, sealed[name])
                result = scoring.outcome(items)
            lines.append(summary_line(record, grader=analysis.GRADER_MODEL, authoritative=False, first_pass=True,
                                      result=result, items=items, run=runs[c.folder]))
            sealed[name]["recorded"] = True
        added = append_lines(Path(args.results_file), lines)
        write_json(sealed_path, sealed)
        say(f"{added} line(s) of the second grader appended to {args.results_file}.")
    return 0


def cmd_dual(args: argparse.Namespace, *, backend_factory: BackendFactory | None = None,
             say: Callable[[str], None] = print) -> int:
    try:
        if args.action == "sample":
            return dual_sample(args, say)
        if args.action == "grade-ai":
            return dual_grade_ai(args, backend_factory or _default_backend_factory, say)
        return dual_report(args, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


# --- the command line -----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant_eval.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build the synthetic grounding project and its marker (solves)")
    p.add_argument("--out", required=True, type=Path, help="the folder to build the project in")

    p = sub.add_parser("run", help="ask the grounding questions at every level setting")
    p.add_argument("--project", required=True, type=Path, help="a project made by build")
    p.add_argument("--provider", required=True, help="the provider preset id (ollama, openrouter, gemini, ...)")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", help="the address, for a custom endpoint or a model server elsewhere")
    p.add_argument("--levels", default="basic,standard,advanced")
    p.add_argument("--ollama-context", help="the context of each setting, e.g. basic=16384,standard=32768,"
                   "advanced=65536 (the model's variants gpt-oss-20b-16k ... made once with ollama create)")
    p.add_argument("--free-tier", choices=("on", "off", "auto"), default="auto")
    p.add_argument("--relax-data-collection", action="store_true",
                   help="OpenRouter: allow providers that collect the (synthetic) data; recorded per cell")
    p.add_argument("--max-requests-per-day", type=int, default=None,
                   help=f"stop after this many requests in a UTC day (default {FREE_DAILY_REQUESTS} for an "
                        "OpenRouter :free model, none otherwise)")
    p.add_argument("--questions", help="comma-separated question ids (default: all)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--accept-data-notice", action="store_true",
                   help="accept the provider's data notice for this run only (only for a grounding project)")
    p.add_argument("--resume", action="store_true", help="continue a results folder")
    p.add_argument("--out", required=True, type=Path, help=f"the results folder (e.g. {DEFAULT_OUT}/<model>)")
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("grade", help="automatic checks and blind grading by the owner (always continues)")
    p.add_argument("results", nargs="+", type=Path, help="the results folders of every model, graded together")
    p.add_argument("--dir", required=True, type=Path, help="the grading folder (its grading.json keeps the order)")
    p.add_argument("--single-model", action="store_true",
                   help="grade one results folder alone (not blind to the model)")
    p.add_argument("--sitting", type=int, default=DEFAULT_SITTING, help="cells to grade by hand in one sitting")
    p.add_argument("--regrade", metavar="#N", help="grade cell #N again (a new authoritative line)")
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("summary", help="rates per level and class; --record appends results.jsonl")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--record", action="store_true")
    p.add_argument("--results-file", default=str(scoring.RESULTS_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("calibrate", help="the evidence tables of results.jsonl for the owner")
    p.add_argument("results_file", nargs="?", default=str(scoring.RESULTS_PATH))

    p = sub.add_parser("dual", help="the second grader on a subset: sample, grade-ai, report")
    p.add_argument("action", choices=("sample", "grade-ai", "report"))
    p.add_argument("results", nargs="*", type=Path, help="sample: the results folders of the matrix")
    p.add_argument("--dir", required=True, type=Path, help="the dual grading folder")
    p.add_argument("--seed", type=int, help="sample: the seed (default: a random one, recorded)")
    p.add_argument("--accept-data-notice", action="store_true", help="grade-ai: accept Anthropic's data notice for this run only")
    p.add_argument("--record", action="store_true", help="report: append the second grader's lines")
    p.add_argument("--results-file", default=str(scoring.RESULTS_PATH), help=argparse.SUPPRESS)
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build": cmd_build, "run": cmd_run, "grade": cmd_grade, "summary": cmd_summary, "calibrate": cmd_calibrate,
    "dual": cmd_dual,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
