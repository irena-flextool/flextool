"""The assistant's level calibration on the grounding questions (P3 §6–§7).  A developer tool: not packaged, never
run in CI, never collected by pytest.

    python tools/assistant_eval.py build --out DIR
    python tools/assistant_eval.py run --project DIR --provider ID --model M --out RESULTS [...]
    python tools/assistant_eval.py grade-ai RESULTS ... --dir GRADING [--accept-data-notice] [--changed]
    python tools/assistant_eval.py review --dir GRADING [--sitting N] [--regrade #N] [--all]
    python tools/assistant_eval.py summary --dir GRADING [--record]
    python tools/assistant_eval.py calibrate [results.jsonl]

``build`` makes the synthetic grounding project (``tests/assistant/grounding_projects.py``: ``tests.json`` and its
overrides, solved with ``flextool-run``) and its marker.  ``run`` asks every question of
``tools/assistant_eval_data/grounding_set.yaml`` at every level setting through the real chat stack (the provider's
backend, the probe, the budget of the level and the model's context, the session) and writes one record per cell
(model, level setting, question, repeat) into RESULTS.  It works only on a project its own ``build`` made: the
marker and the source database's hash must match, so case data of a real project never reaches a calibration
model.  Keys come from the keyring or the owner's environment, never from the command line.  ``grade-ai`` runs
the automatic checks against the oracle's ``answer_key.yaml`` and has Claude Opus 5.5 grade every cell, blind (one
request per cell; a cell without a final answer is failed in code); ``review`` shows the owner the queued cells
(low confidence, a number without a tool source, an injection hint, the cells Opus wrote, a random sample) and
takes the owner's verdicts before Opus's are shown; ``summary`` prints the outcomes and the owner–Opus agreement and
with ``--record`` appends the final grades to ``tools/assistant_eval_data/results.jsonl`` (no answer or tool
text).  All grading state lives in the grading folder; results folders are only read.  ``calibrate`` turns
``results.jsonl`` into the evidence tables for the owner (``tools/assistant_eval_analysis.py``).  How to run and
read it: ``docs/dev/assistant_evaluation.md``.

Exit codes: 0 done (also: the daily request budget is used, run again with --resume) · 1 failed (also: an error
that is not the model's behaviour stopped a cell, which is not saved: run again with --resume) · 2 usage or a
refused project / results folder · 3 the data notice was not accepted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for _path in (ROOT, TOOLS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import assistant_eval_scoring as scoring  # noqa: E402  (after the repository root is on the path)

from flextool.assistant._files import iso, utc_now  # noqa: E402  (the store's own; tests patch utc_now here)

RECORD_FORMAT = 1
RUN_FILE = "run.json"
CELLS_DIR = "cells"
GRADING_FILE = "grading.json"
#: The requests of the cells that stopped without a record (``[{"ran_at", "requests", "cell", "code"}]``): they
#: count against the daily budget like the requests of the saved cells.
SPENT_FILE = "spent.json"
#: The turn stops that are the model's or the budget's behaviour and so a graded outcome: cut off, still too long
#: for the context (the fit refusal), the provider's refusal.  Any other error stops the cell unsaved (C6-M5).
GRADED_STOPS = frozenset({"length", "context", "refused"})
#: What a results folder's ``run.json`` must match to be continued; the levels, questions and repeats only pick
#: the cells.
IDENTITY_KEYS = ("provider", "model", "base_url", "free_tier", "ollama_context", "knobs_relaxed",
                 "project_fingerprint", "set_version", "cpu_offload_allowed")
#: The value of an identity key a ``run.json`` from before that key was added stands for (it was always so).
IDENTITY_DEFAULTS = {"cpu_offload_allowed": False}
DEFAULT_OUT = ROOT / "_audit_reports" / "assistant_eval"
#: The daily request budget of an OpenRouter ``:free`` model (P3 §7.2).
FREE_DAILY_REQUESTS = 45
DEFAULT_SITTING = 37
#: The ledger provider key of calibration requests: the route's key with this suffix (shown apart in /usage).
CALIBRATION_SUFFIX = "#calibration"
OLLAMA_CONTEXTS = {"small": 16_384, "medium": 32_768, "large": 65_536}

NOT_GROUNDING = "This is not a grounding project made by assistant_eval build."
CHANGED_GROUNDING = "This is not a grounding project made by assistant_eval build (it has changed since: {what})."
UNSOLVED_GROUNDING = "This grounding project was built without its runs: build it again with assistant_eval build."
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


class CellStopped(Exception):
    """A turn of the cell ended in an error that is not the model's behaviour (connection, timeout, busy, key,
    model or routing, a fault): the cell is not saved and ``--resume`` asks it again.  *requests_sent*: the
    requests that reached the provider before the stop (they count against the daily budget)."""

    def __init__(self, text: str, requests_sent: int = 0, code: str = "") -> None:
        super().__init__(text)
        self.requests_sent = requests_sent
        self.code = code


class DailyBudgetUsed(CellStopped):
    """The provider's quota is used: the run stops as when the daily budget is used (exit 0, D-Q4)."""


# --- small helpers --------------------------------------------------------------------------------------------
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


def _key_or_refuse(preset: Any, base_url: str, model: str, route: Any) -> tuple[str, str]:
    """The key of *preset* (keyring or environment, never the command line) and where it came from, for ``run``
    and ``grade-ai``; :class:`Refused` with the chat's "no key" message when the preset needs one."""
    from flextool.assistant import keys
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.llm_adapter import Connection
    from flextool.assistant.llm_errors import render

    key, source, note = keys.resolve_key(preset, base_url)
    if preset.key_required and not key:
        conn = Connection(base_url=base_url, model=model, api_key="", key_source="", knobs={}, route=route)
        raise Refused(render("no_key", error_context(preset, conn)) + (f" ({note})" if note else ""))
    return key or "", source.describe(preset) if key else ""


def check_marker(project: Path, *, allow_unsolved: bool = False) -> dict[str, Any]:
    """The marker of a grounding project as ``build`` left it; else :class:`Refused` (exit 2, before any request).
    Checked: the marker's format, that the project was solved (*allow_unsolved*: the tests' inputs-only project),
    the source database's hash, no other input source, no result folder of another run and every run's id."""
    from flextool import run_record
    from tests.assistant import grounding_projects as gp

    project = Path(project)
    try:
        marker = read_json(project / gp.MARKER_FILE)
    except (OSError, ValueError):
        raise Refused(NOT_GROUNDING) from None
    if not isinstance(marker, dict) or marker.get("format") != gp.MARKER_FORMAT \
            or not isinstance(marker.get("runs"), dict):
        raise Refused(NOT_GROUNDING)
    if not marker.get("solved") and not allow_unsolved:
        raise Refused(UNSOLVED_GROUNDING)

    def changed(what: str) -> Refused:
        return Refused(CHANGED_GROUNDING.format(what=what))

    source = project / "input_sources" / gp.SOURCE_NAME
    if not source.is_file() or gp.sha256_file(source) != marker.get("source_sha256"):
        raise changed("its source database")
    allowed = {gp.SOURCE_NAME, *(gp.SOURCE_NAME + tail for tail in ("-journal", "-wal", "-shm"))}
    for entry in sorted((project / "input_sources").iterdir()):
        if entry.name not in allowed:
            raise changed(f"another input source {entry.name}")
    runs: dict[str, Any] = marker["runs"]
    for output in sorted(p for p in project.glob("output_*") if p.is_dir()):
        for sub in sorted(p for p in output.iterdir() if p.is_dir()):
            if sub.name not in runs and any(sub.iterdir()):
                raise changed(f"results {output.name}/{sub.name} of another run")
    for folder, run_id in sorted(runs.items()):
        status = run_record.read_status_file(project / "output_parquet" / folder) or {}
        if status.get("run_id") != run_id:
            raise changed(f"the run in output_parquet/{folder}")
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


def requests_today(cells: Iterable[Mapping[str, Any]], today: str, spent: Iterable[Mapping[str, Any]] = ()) -> int:
    """The requests this results folder sent on the UTC day *today* (``YYYY-MM-DD``): those of its saved cells
    and those of the cells that stopped unsaved or were replaced (*spent*)."""
    return sum(int(c.get("requests_sent") or 0) for c in cells if str(c.get("ran_at") or "").startswith(today)) \
        + sum(int(e.get("requests") or 0) for e in spent if str(e.get("ran_at") or "").startswith(today))


def read_spent(out: Path) -> list[dict]:
    path = Path(out) / SPENT_FILE
    return list(read_json(path)) if path.is_file() else []


def ended_before_answer(turn: Mapping[str, Any]) -> bool:
    """Whether a saved *turn* is a provider failure saved as a cut-off answer (calibration 2026-11-10): it stopped
    with "the connection ended before the answer was complete", without answer text, and its last request reported
    no output tokens (0 or none: a model's reasoning streamed before the break is not counted as output, a tool
    call or text is).  Saved by a runner from before the adapter retried such answers."""
    from flextool.assistant.llm_errors import MESSAGES

    error = turn.get("error") or {}
    if error.get("code") != "length" or error.get("message") != MESSAGES["length_no_finish"]:
        return False
    if str(turn.get("answer") or "").strip():
        return False
    usage = turn.get("usage") or []
    return not usage or not (usage[-1] or {}).get("output")


def misclassified(record: Mapping[str, Any]) -> bool:
    """Whether a saved cell has a turn that :func:`ended_before_answer`."""
    return any(ended_before_answer(t) for t in record.get("turns") or [])


def finished(record: Mapping[str, Any]) -> bool:
    """Whether a saved cell is done: run (not ``not_applicable``), no turn stopped by an error other than the
    graded stops (an older folder can hold a cell saved with a connection error: ``--resume`` asks it again), and
    no turn a provider failure saved as cut off (:func:`misclassified`; asked again too)."""
    if record.get("not_applicable"):
        return False
    return (all((t.get("error") or {}).get("code") in (None, *GRADED_STOPS) for t in record.get("turns") or [])
            and not misclassified(record))


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
        missing = object()
        if any(saved.get(k, IDENTITY_DEFAULTS.get(k, missing)) != identity[k] for k in IDENTITY_KEYS):
            raise Refused(OTHER_RESULTS)
        return
    if (Path(out) / CELLS_DIR).is_dir() and any((Path(out) / CELLS_DIR).iterdir()):
        raise Refused(OTHER_RESULTS)
    write_json(run_file, {**identity, "format": RECORD_FORMAT, "created": iso(utc_now()),
                          MATRIX_KEY: {"questions": [], "levels": [], "repeat": 0, "automatic_at": {}}})


#: run.json's record of the cells its runs planned (every run adds its questions, levels and repeats; the
#: grading refuses a folder that misses one of them).  ``automatic_at``: level → whether the question asked only
#: at the automatic level is asked there (known once that level's setting is made).  A folder from before has
#: none; its planned cells are taken from the cells of every folder graded with it (:func:`expected_cells`).
MATRIX_KEY = "matrix"


def _plan_matrix(out: Path, questions: Sequence[str], levels: Sequence[str], repeat: int) -> None:
    """Add this run's questions, levels and repeats to run.json's matrix, before any cell is asked (a folder
    without a matrix, from before, is left as it is)."""
    run_file = Path(out) / RUN_FILE
    run = read_json(run_file)
    matrix = run.get(MATRIX_KEY)
    if matrix is None:
        return
    matrix["questions"] = [*matrix["questions"], *(q for q in questions if q not in matrix["questions"])]
    matrix["levels"] = [*matrix["levels"], *(lv for lv in levels if lv not in matrix["levels"])]
    matrix["repeat"] = max(int(matrix["repeat"]), repeat)
    write_json(run_file, run)


def _plan_automatic(out: Path, level: str, asked: bool) -> None:
    run_file = Path(out) / RUN_FILE
    run = read_json(run_file)
    matrix = run.get(MATRIX_KEY)
    if matrix is not None and matrix["automatic_at"].get(level) != asked:
        matrix["automatic_at"][level] = asked
        write_json(run_file, run)


def expected_cells(run: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]], qset: scoring.QuestionSet,
                   fallback: Mapping[str, Any]) -> tuple[set[str], list[str]]:
    """``(expected cell ids, unknown)`` of a results folder: its run.json matrix (or *fallback*, for a folder from
    before the matrix: the questions, levels and repeats of every folder graded with it).  A question asked only
    at the automatic level is expected at a level whose setting asks it; where that is not known yet (the level
    was never reached) it is in *unknown* as ``"S01__<level>__r1 (if <level> is the automatic level)"``."""
    matrix = run.get(MATRIX_KEY) or fallback
    automatic_at = dict(matrix.get("automatic_at") or {})
    if MATRIX_KEY not in run:
        # From the folder's own records: a level asks the automatic-only question where its applicable cells
        # name that level as the automatic one.
        for record in cells.values():
            level = record["cell"]["level_setting"]
            if record.get("not_applicable"):
                automatic_at.setdefault(level, False)
            else:
                automatic_at[level] = automatic_at.get(level, False) or (
                    (record.get("model") or {}).get("automatic_level") == level)
    expected: set[str] = set()
    unknown: list[str] = []
    for qid in matrix["questions"]:
        question = qset.questions.get(qid)
        if question is None:
            continue
        for level in matrix["levels"]:
            for repeat in range(1, int(matrix["repeat"]) + 1):
                cid = cell_id(qid, level, repeat)
                if not question.automatic_level_only or automatic_at.get(level):
                    expected.add(cid)
                elif level not in automatic_at and cid not in cells:
                    unknown.append(f"{cid} (if {level} is the automatic level)")
    return expected, unknown


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
    the record, redacted.  Raises :class:`CellStopped` when a turn ends in an error that is not a graded stop
    (:data:`GRADED_STOPS`), :class:`DailyBudgetUsed` when the provider says the quota is used."""
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
        if result.error is not None and result.error.code not in GRADED_STOPS:
            sent = sum(len(t["usage"]) for t in turns) + len(result.usage)
            stop = DailyBudgetUsed if result.error.code == "quota" else CellStopped
            raise stop(message_for(result.error, "terminal"), sent, result.error.code)
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
    offload = None if setting.not_applicable else cpu_offload(setting.ollama)
    return {
        "format": RECORD_FORMAT,
        "cell": {"model": identity["model"], "level_setting": setting.level, "question": question.id,
                 "class": question.cls, "repeat": repeat},
        "model": {"preset": preset.id, "route": setting.route.kind if setting.route else None,
                  "model": setting.model,
                  "automatic_level": choice.automatic if choice else None,
                  "level_source": choice.source if choice else None,
                  "window": b.window if b else None, "window_source": b.window_source if b else None,
                  "free_tier": free_tier, "ollama": setting.ollama, "cpu_offload": offload,
                  "knobs_sent": dict(knobs_sent),
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
        "set_version": identity["set_version"],
        "project_fingerprint": identity["project_fingerprint"], "flextool_version": flextool_version(),
        "git": dict(git), "ran_at": iso(utc_now()), "requests_sent": requests_sent, "purpose": "calibration",
    }


def cpu_offload(ollama: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``{size, size_vram, cpu_share}`` of an Ollama variant that is partly on the CPU (``size_vram`` below
    ``size``, in bytes; ``cpu_share`` the part of ``size`` not in graphics memory), else None."""
    size, vram = (ollama or {}).get("size"), (ollama or {}).get("size_vram")
    if not (isinstance(size, int) and isinstance(vram, int) and 0 <= vram < size):
        return None
    return {"size": size, "size_vram": vram, "cpu_share": round((size - vram) / size, 4)}


def _default_prober(preset: Any, conn: Any, cancel: threading.Event, *, needed: int) -> Any:
    from flextool.assistant import probes

    return probes.probe(preset, conn, cancel, needed=needed)


def _default_backend_factory(preset: Any, conn: Any) -> Any:
    from flextool.assistant.llm_adapter import make_backend

    return make_backend(preset, conn)


def _setting(level: str, model: str, *, preset: Any, base_url: str, conn_for: Callable[[str, Any], Any],
             prober: Prober, info: OllamaInfo | None, wanted_context: int | None, free_tier: bool,
             cache: dict[str, Any], allow_cpu_offload: bool = False) -> Setting:
    """The facts and budget of one level setting: the model is probed (an Ollama variant loaded and its context
    and graphics memory read); the level is the one asked for, the window rule sets the caps.  A variant partly
    on the CPU is not applicable unless *allow_cpu_offload*; a smaller loaded context than asked never is."""
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
                not allow_cpu_offload and isinstance(size, int) and isinstance(vram, int) and vram < size):
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
            say: Callable[[str], None] = print, allow_unsolved: bool = False) -> int:
    """``run``; *allow_unsolved* (no command-line flag) lets the tests use an inputs-only grounding project."""
    try:
        return _run(args, backend_factory or _default_backend_factory, prober or _default_prober,
                    info if info is not None else ollama_info, say, allow_unsolved=allow_unsolved)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _run(args: argparse.Namespace, backend_factory: BackendFactory, prober: Prober, info: OllamaInfo,
         say: Callable[[str], None], *, allow_unsolved: bool = False) -> int:
    from flextool.assistant import budget, endpoint, net, providers
    from flextool.assistant.chat_cli import FREE_TIER_FLAG
    from flextool.assistant.llm_adapter import Connection
    from flextool.assistant.settings import DEFAULT_KNOBS
    from flextool.assistant.usage import Ledger

    project = Path(args.project).absolute()
    marker = check_marker(project, allow_unsolved=allow_unsolved)   # first: nothing is read or sent elsewhere
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
    if args.allow_cpu_offload and not contexts:
        raise Refused("--allow-cpu-offload is for an Ollama run with --ollama-context only.")
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
    free = budget.free_tier_on(preset.id, args.model, FREE_TIER_FLAG[args.free_tier])
    identity = {"provider": preset.id, "model": args.model, "base_url": base_url, "free_tier": free,
                "ollama_context": contexts, "knobs_relaxed": bool(args.relax_data_collection),
                "project_fingerprint": marker["fingerprint"], "set_version": qset.set_version,
                "cpu_offload_allowed": bool(args.allow_cpu_offload)}
    _open_results(out, identity, args.resume)
    _plan_matrix(out, wanted, levels, args.repeat)

    key, source = _key_or_refuse(preset, base_url, args.model, endpoint.route_for(preset, base_url, model=args.model))
    knobs = dict(DEFAULT_KNOBS)
    if args.relax_data_collection:
        knobs["openrouter.data_collection"] = "allow"
    knobs_sent = providers.request_fields(preset, knobs)

    def conn_for(model: str, route: Any) -> Connection:
        return Connection(base_url=base_url, model=model, api_key=key, key_source=source, knobs=knobs, route=route)

    daily = daily_budget(preset.id, args.model, args.max_requests_per_day)
    ledger = CalibrationLedger(Ledger())
    git = git_state()
    cells = read_cells(out)
    spent = read_spent(out)
    failed = sorted(cid for cid, record in cells.items() if misclassified(record))
    if failed:
        say(f"{len(failed)} saved cell(s) are a provider failure saved as cut off (the answer broke off before "
            f"any of it came); they are asked again: {', '.join(failed)}.")

    def budget_used() -> bool:
        return daily is not None and requests_today(cells.values(), iso(utc_now())[:10], spent) >= daily

    def spend(entry: dict[str, Any]) -> None:
        spent.append(entry)
        write_json(out / SPENT_FILE, spent)

    probed: dict[str, Any] = {}
    for level in levels:
        model = variant_name(args.model, contexts[level]) if level in contexts else args.model
        setting = _setting(level, model, preset=preset, base_url=base_url, conn_for=conn_for, prober=prober,
                           info=info if level in contexts else None, wanted_context=contexts.get(level),
                           free_tier=free, cache=probed, allow_cpu_offload=args.allow_cpu_offload)
        if not setting.not_applicable:
            # The probe can show that the model is forwarded (Ollama cloud): that route's notice is asked too.
            # The probe sent only the model name, to the local server.
            accept_route(preset, base_url, setting.route, accepted, accept=args.accept_data_notice,
                         again="--resume --accept-data-notice", what=what)
        _plan_automatic(out, level, bool(not setting.not_applicable and setting.choice is not None
                                         and setting.choice.automatic == level))
        done = DAILY_DONE_REMOTE if accepted else DAILY_DONE
        for qid in wanted:
            question = qset.questions[qid]
            if question.automatic_level_only and (setting.choice is None or level != setting.choice.automatic):
                continue
            for repeat in range(1, args.repeat + 1):
                cid = cell_id(qid, level, repeat)
                saved = cells.get(cid)
                if saved is not None and (finished(saved) or (saved.get("not_applicable")
                                                              and setting.not_applicable)):
                    continue                       # done, or still not applicable (kept with its recorded tags)
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
                if saved is not None and misclassified(saved):
                    say(f"{cid}: asked again (a provider failure saved as cut off)")
                # A saved cell asked again (an error stop of an older folder, a provider failure saved as cut
                # off, or now applicable) stays on disk until its new record replaces it: a stop on the way
                # leaves the folder with the old record, never without the cell.
                backend = backend_factory(preset, conn_for(setting.model, setting.route))
                try:
                    turns = run_cell(project, qset, question, setting, backend=backend, preset=preset,
                                     ledger=ledger, say=say)["turns"]
                except CellStopped as exc:
                    spend({"ran_at": iso(utc_now()), "requests": exc.requests_sent, "cell": cid, "code": exc.code})
                    say(str(exc))
                    if isinstance(exc, DailyBudgetUsed):
                        say(done)
                        return 0
                    say(f"{cid}: nothing was saved for this cell. Run again with --resume"
                        + (" --accept-data-notice." if accepted else "."))
                    return 1
                finally:
                    backend.close()
                record = _record(question, setting, repeat, preset=preset, identity=identity,
                                 knobs_sent=knobs_sent, free_tier=free, git=git, turns=turns)
                if saved is not None and int(saved.get("requests_sent") or 0):
                    # The replaced record's requests stay counted for their day (written first: a crash
                    # between the two counts them twice, never not at all).
                    spend({"ran_at": saved.get("ran_at"), "requests": int(saved["requests_sent"]), "cell": cid,
                           "code": "replaced"})
                write_json(cell_path(out, cid), record)              # temp + os.replace: the old one goes
                cells[cid] = record
                say(f"{cid}: saved")
                if budget_used():
                    say(done)
                    return 0
    say(f"All cells of {args.model} are in {out}.")
    return 0


# --- grading v2: the grading folder ---------------------------------------------------------------------------
GRADING_FORMAT = 2
REVIEW_FILE = "review.json"
AGREEMENT_FILE = "agreement.txt"
LOCK_FILE = ".grade-ai.lock"
#: The items' verdict letters the owner types.
LETTERS = {"r": "right", "m": "missing", "w": "wrong"}
OLD_GRADING = ("This grading folder was made by the earlier grading, which had no 'missing' verdict and no "
               "'incomplete' outcome. Its grades are not used. Grade into a new folder: --dir NEW_FOLDER.")
OTHER_FOLDERS = "This grading covers other results folders; use a new --dir."
CHANGED_MATERIAL = ("Cell {token} was graded from other material (its answer, the answer key or the rubric changed). "
                    "Run grade-ai --changed to grade the changed cells again, or use a new --dir.")
SUMMARY_DIR = "summary now reads a grading folder: summary --dir GRADING."
REASON_WITHHELD = "[reason withheld]"

CellKey = tuple[str, str]


def _cell_key(folder: Path, cid: str) -> CellKey:
    return (str(Path(folder).resolve()), cid)


def _stamps(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(r.get(k)) for r in records for k in ("ran_at", "graded_at") if r.get(k)}


def _tool_text(record: Mapping[str, Any]) -> str:
    """The cell's tool calls as the leak scan's sources of dates and times: the results and the arguments the
    model wrote (a time range it asked for is no time of the run; ``ran_at`` is checked exactly all the same)."""
    return "\n".join(text for t in record.get("turns") or [] for c in t.get("tool_calls") or []
                     for text in (str(c.get("result") or ""), json.dumps(c.get("arguments"), ensure_ascii=False)))


def _turn_views(record: Mapping[str, Any]) -> list[scoring.TurnView]:
    return [scoring.TurnView.from_record(t) for t in record.get("turns") or []]


def _key_for(args: argparse.Namespace, records: Iterable[Mapping[str, Any]]) -> scoring.AnswerKey:
    path = Path(args.key)
    if not path.is_file():
        raise Refused(f"The answer key {path} does not exist yet: the independent oracle writes it.")
    key = scoring.load_key(path)
    if key.placeholder:
        raise Refused("The answer key is a placeholder: grading needs the oracle's key.")
    versions = {r.get("set_version") for r in records}
    if versions - {key.set_version}:
        raise Refused(f"The answer key answers question set {key.set_version}; these results are of "
                      f"{', '.join(map(str, sorted(versions)))}.")
    return key


def _read_grading(folder: Path) -> dict | None:
    """``grading.json`` of a v2 grading folder (``None`` when there is none); :class:`Refused` for the earlier
    format, before anything is written."""
    path = Path(folder) / GRADING_FILE
    if not path.is_file():
        return None
    state = read_json(path)
    if not isinstance(state, dict) or state.get("format") != GRADING_FORMAT:
        raise Refused(OLD_GRADING)
    return state


def own_model(run: Mapping[str, Any]) -> bool:
    """Whether a results folder's model is the AI grader's own (its cells test self-preference)."""
    import assistant_eval_analysis as analysis

    return run.get("provider") == analysis.GRADER_PROVIDER and str(run.get("model") or "").startswith(
        analysis.GRADER_MODEL)


@dataclasses.dataclass
class GradingCell:
    """One cell of a grading: its record (read only), the graded items, the material every grader sees and the
    hash of the request it makes; *state* is ``GRADING/cells/<n>.json`` (``None`` before Opus graded it)."""

    token: str
    key: CellKey
    record: dict
    run: dict
    question: scoring.Question
    entry: scoring.KeyEntry
    items: list[scoring.GradeItem]
    diagnostics: dict[str, str]
    empty: bool
    material: str
    request_sha256: str
    path: Path
    state: dict | None

    @property
    def judged(self) -> list[scoring.GradeItem]:
        return [i for i in self.items if not i.auto]

    @property
    def allowed(self) -> dict[str, tuple[str, ...]]:
        return {i.id: i.allowed for i in self.judged}

    @property
    def stale(self) -> bool:
        return self.state is not None and self.state.get("request_sha256") != self.request_sha256

    @property
    def ai(self) -> dict | None:
        """Opus's parsed grading (``None`` for none, an unparsed reply or an empty answer)."""
        ai = (self.state or {}).get("ai")
        return ai if ai and "unparsed" not in ai else None

    @property
    def unparsed(self) -> bool:
        return "unparsed" in ((self.state or {}).get("ai") or {})

    @property
    def graded(self) -> bool:
        """Opus graded it (an unparsed reply counts), or it has no answer (failed in code)."""
        return self.state is not None and (self.state.get("no_answer") or self.state.get("ai") is not None)

    @property
    def number(self) -> int:
        return int(self.token.lstrip("#"))


def request_hash(material: str, key: scoring.AnswerKey) -> str:
    """The hash of what a grading request is made from: the grader, the rubric, the system and user parts and the
    key's versions (an unchanged hash: the cell's grading still stands)."""
    import hashlib

    import assistant_eval_analysis as analysis

    data = json.dumps([analysis.GRADER_MODEL, scoring.RUBRIC_VERSION, key.set_version, key.key_version,
                       scoring.GRADER_PROMPT_V2, material], ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class Grading:
    folder: Path
    state: dict
    cells: list[GradingCell]
    not_applicable: list[tuple[Path, dict, dict, str]]     # (folder, run, record, cell id)
    names: list[str]
    scan_names: list[str]
    stamps: set[str]
    key: scoring.AnswerKey
    old_grades: int


def load_grading(args: argparse.Namespace, *, create: bool) -> Grading:
    """The grading of ``args.dir`` with the cells of its results folders (``args.results`` when *create*), checked:
    the earlier format refused, the folder set fixed by the first ``grade-ai``, every planned cell there and
    finished (:func:`_refuse_incomplete`).  Nothing is
    written (a new grading's ``grading.json`` is written by :func:`_start` once every check passed)."""
    folder = Path(args.dir)
    state = _read_grading(folder)
    if state is None:
        if not create:
            raise Refused(f"No grading in {folder}: run grade-ai first.")
        if not args.results:
            raise Refused("Give the results folders of every model to grade.")
    results = [Path(p) for p in (args.results or state["results"])]
    wanted = sorted(str(Path(p).resolve()) for p in results)
    if state is not None and args.results and wanted != sorted(state["results"]):
        raise Refused(OTHER_FOLDERS)
    loaded = _load_results([Path(p) for p in wanted])
    records = {_cell_key(f, cid): r for f, _run, cells in loaded for cid, r in cells.items()}
    runs = {_cell_key(f, "")[0]: run for f, run, _cells in loaded}
    if not records:
        raise Refused(f"No cells in {', '.join(wanted)}.")
    qset = scoring.load_set(Path(args.set))
    _refuse_incomplete(loaded, qset)
    key = _key_for(args, records.values())
    applicable = sorted(k for k, r in records.items() if not r.get("not_applicable"))
    if state is None:
        seed = random.SystemRandom().randrange(2**31)
        order = list(applicable)
        random.Random(seed).shuffle(order)
        state = {"format": GRADING_FORMAT, "rubric_version": scoring.RUBRIC_VERSION, "seed": seed,
                 "results": wanted, "order": [list(k) for k in order],
                 "tokens": {f"#{n}": list(k) for n, k in enumerate(order, 1)}, "set_version": key.set_version,
                 "key_version": key.key_version, "grader_model": _grader_model(), "created": iso(utc_now()),
                 "recorded_not_applicable": []}
    order = [(str(f), str(c)) for f, c in state["order"]]
    if set(order) != set(applicable):
        raise Refused("The results folders hold other cells than this grading covers (cells added, removed or "
                      "now not applicable); use a new --dir.")
    run_pairs = [(f, run) for f, run, _c in loaded]
    names = scoring.identity_names(run_pairs, records.values())
    scan_names = scoring.identity_names(run_pairs, records.values(), folders=False)
    cells = []
    for n, k in enumerate(order, 1):
        token = f"#{n}"
        record = records[k]
        question = qset.questions[record["cell"]["question"]]
        entry = key.questions.get(question.id)
        if entry is None:
            raise Refused(f"The answer key has no entry for {question.id}.")
        views = _turn_views(record)
        max_calls = ((record.get("budget") or {}).get("caps") or {}).get("max_calls_per_turn")
        items, diagnostics = scoring.grading_items(question, entry, record["cell"]["level_setting"], views,
                                                   max_calls=max_calls)
        try:
            material = scoring.grader_material(record, question, entry, items, names,
                                               scoring.data_tag(state["seed"], token))
        except scoring.MaterialError:
            raise Refused(f"Cell {token}: its material contains its data tag; nothing was sent.") from None
        path = folder / CELLS_DIR / f"{n}.json"
        cells.append(GradingCell(token, k, record, runs[k[0]], question, entry, items, diagnostics,
                                 scoring.no_answer(views), material, request_hash(material, key), path,
                                 read_json(path) if path.is_file() else None))
    not_applicable = [(Path(f), runs[f], r, cid) for (f, cid), r in sorted(records.items()) if r.get("not_applicable")]
    old = sum(1 for r in records.values() if r.get("outcome") is not None or r.get("checks"))
    return Grading(folder, state, cells, not_applicable, names, scan_names, _stamps(records.values()), key, old)


def _refuse_incomplete(loaded: Sequence[tuple[Path, dict, dict[str, dict]]], qset: scoring.QuestionSet) -> None:
    """:class:`Refused` when a results folder misses a cell of its run.json matrix, or holds one that is not
    finished: a cell that came back later would change the grading's cells, so nothing is graded before every
    folder is complete.  The refusal names the cells and the ``run --resume`` command of each folder."""
    seen = [r["cell"] for _f, _run, cells in loaded for r in cells.values()]
    fallback = {"questions": sorted({c["question"] for c in seen}),
                "levels": [lv for lv in scoring.LEVELS if any(c["level_setting"] == lv for c in seen)],
                "repeat": max((int(c.get("repeat") or 1) for c in seen), default=1)}
    lines = []
    total = 0
    for folder, run, cells in loaded:
        expected, unknown = expected_cells(run, cells, qset, fallback)
        missing = sorted(expected - set(cells))
        unfinished = sorted(cid for cid, r in cells.items() if not r.get("not_applicable") and not finished(r))
        named = [f"{cid} (missing)" for cid in missing] + unknown + [f"{cid} (not finished)" for cid in unfinished]
        if not named:
            continue
        total += len(named)
        command = (f"python tools/assistant_eval.py run --resume --project PROJECT --provider {run.get('provider')} "
                   f"--model {run.get('model')} --out {folder}")
        lines.append(f"{Path(folder).name}: {', '.join(named)}\n  complete them with: {command} (with the "
                     "levels, repeats and other options of its first run)")
    if lines:
        raise Refused(f"{total} cell(s) are missing or not finished; nothing is graded before every results "
                      "folder is complete:\n" + "\n".join(lines))


def _grader_model() -> str:
    import assistant_eval_analysis as analysis

    return analysis.GRADER_MODEL


def _refuse_stale(grading: Grading) -> None:
    stale = [c for c in grading.cells if c.stale]
    if stale:
        raise Refused(CHANGED_MATERIAL.format(token=stale[0].token) + f" ({len(stale)} cell(s) changed.)")


class _Lock:
    """``GRADING/.grade-ai.lock``, made exclusively by ``grade-ai`` and ``review``: two of them on one folder never
    both write (a review's verdicts are never overwritten by a ``grade-ai --changed``).  A grading folder the lock
    created is removed again when nothing else was written into it (a refused first ``grade-ai``)."""

    def __init__(self, folder: Path, command: str = "grade-ai") -> None:
        self.path = Path(folder) / LOCK_FILE
        self.command = command
        self.created = False

    def __enter__(self) -> _Lock:
        self.created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Refused(f"Another grade-ai or review works on this grading folder ({self.path} exists). If none "
                          "does, delete that file.") from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} {self.command}\n")
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)
        if self.created:
            try:
                self.path.parent.rmdir()            # only when empty
            except OSError:
                pass


# --- grade-ai: Opus grades every cell -------------------------------------------------------------------------
def _scan(grading: Grading, cell: GradingCell, text: str) -> list[str]:
    return scoring.leak_scan(text, names=grading.scan_names, stamps=grading.stamps, sources=_tool_text(cell.record))


def _request_text(material: str, allowed: Iterable[str] = ()) -> str:
    """Every text a grading request can send, as sent: the system part, the user part and the retry (the retry
    names only FlexTool's own item ids and field names)."""
    return "\n".join([scoring.GRADER_PROMPT_V2, material,
                      scoring.retry_text([f"Missing: {', '.join(allowed)} (give every item to grade a verdict)."])])


class RetryStopped(Exception):
    """A provider error during the "JSON only" retry: *pending* keeps the first reply (already paid), so the next
    ``grade-ai`` sends only the retry."""

    def __init__(self, error: Exception, pending: dict[str, Any]) -> None:
        super().__init__(str(error))
        self.error = error
        self.pending = pending


def ask_grader(backend: Any, material: str, allowed: Mapping[str, Sequence[str]], *, pacing: Any,
               on_usage: Callable[[Any], None], on_rejected: Callable[[Any], None], passed: Iterable[str] = (),
               pending: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One grading request through FlexTool's retries and pacing, and at most one "JSON only" retry that names
    what was wrong: the parsed reply with ``attempts`` and ``usage``, or ``{"unparsed": text, ...}``.  *passed*
    are the items the automatic checks found right (a verdict for one is dropped, see
    :func:`scoring.check_reply`).  *on_usage* gets the usage of every request that reached the model,
    *on_rejected* every rate-limit or overload answer.

    A provider error in the retry raises :class:`RetryStopped` with the first reply; given back as *pending*, the
    next call sends only the retry (the first request is not paid twice)."""
    import assistant_eval_analysis as analysis
    from flextool.assistant.llm_adapter import Message, Request
    from flextool.assistant.llm_errors import LLMError
    from flextool.assistant.llm_retry import RetryPolicy, call_with_retries
    from flextool.assistant.thoughts import strip_thoughts

    cancel = threading.Event()
    messages = [Message("user", material)]
    usage: list[dict[str, Any]] = []
    last = ""
    problems: list[str] = []
    start = 0
    if pending:
        last, usage, problems, start = str(pending["reply"]), list(pending["usage"]), list(pending["problems"]), 1
        messages += [Message("assistant", last or "(no text)"), Message("user", scoring.retry_text(pending["problems"]))]
    for attempt in range(start, 2):
        # No prompt caching: every cell's prompt is unique, so a cache write would only cost more.
        req = Request(system=(scoring.GRADER_PROMPT_V2,), tools=(), messages=tuple(messages),
                      max_output_tokens=analysis.GRADER_MAX_OUTPUT, tool_choice="none", cache=False)
        try:
            reply = call_with_retries(lambda req=req: backend.stream(req, lambda _e: None, cancel), RetryPolicy(),
                                      lambda _e: None, cancel, on_rejected=on_rejected, pacing=pacing)
        except LLMError as exc:
            if attempt == 0:
                raise
            raise RetryStopped(exc, {"reply": last, "usage": usage, "problems": problems,
                                     "retry_error": exc.code}) from exc
        on_usage(reply.usage)
        usage.append(_usage_json(reply.usage))
        last = strip_thoughts(reply.message.text) if reply.message is not None else ""
        parsed, problems = scoring.check_reply(last, allowed, passed)
        if parsed is not None:
            return {**parsed, "attempts": attempt + 1, "usage": usage}
        messages += [Message("assistant", last or "(no text)"), Message("user", scoring.retry_text(problems))]
    return {"unparsed": last[:2_000], "attempts": 2, "usage": usage, "problems": problems}


def _new_state(cell: GradingCell, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    history = list((previous or {}).get("history") or [])
    if previous:
        history.append({k: previous.get(k) for k in ("request_sha256", "graded_at", "no_answer", "ai", "owner_first",
                                                     "owner", "changes", "regrades")})
    return {"token": cell.token, "cell": list(cell.key), "request_sha256": cell.request_sha256,
            "rubric_version": scoring.RUBRIC_VERSION, "items": [i.to_json() for i in cell.items],
            "no_answer": cell.empty, "ai": None, "owner_first": None, "owner": None, "changes": [], "regrades": [],
            "history": history, "recorded": list((previous or {}).get("recorded") or []), "graded_at": None}


def _fresh_state(cell: GradingCell) -> dict | None:
    """``cells/<n>.json`` as it is now on disk: what a ``review`` wrote since the grading was loaded is kept."""
    return read_json(cell.path) if cell.path.is_file() else None


def _current_state(cell: GradingCell) -> dict[str, Any]:
    """The cell's state re-read from disk (a pending retry continued: no new history entry)."""
    return dict(_fresh_state(cell) or cell.state or {})


def cmd_grade_ai(args: argparse.Namespace, *, backend_factory: BackendFactory | None = None,
                 say: Callable[[str], None] = print) -> int:
    try:
        if _read_grading(Path(args.dir)) is None and len(args.results) == 1 and not args.single_model:
            raise Refused("Grading one model is not blind: give the results folders of every model of the matrix "
                          "(their cells are shuffled together), or --single-model.")
        with _Lock(Path(args.dir)):
            return _grade_ai(args, backend_factory or _default_backend_factory, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _grade_ai(args: argparse.Namespace, backend_factory: BackendFactory, say: Callable[[str], None]) -> int:
    import assistant_eval_analysis as analysis
    from flextool.assistant import endpoint, providers
    from flextool.assistant.llm_adapter import Connection
    from flextool.assistant.llm_errors import LLMError, message_for
    from flextool.assistant.llm_retry import Pacing
    from flextool.assistant.pacing import PROCESS_PACER, pace_key
    from flextool.assistant.usage import Ledger

    grading = load_grading(args, create=True)
    if grading.old_grades:
        say(f"{grading.old_grades} cells carry grades from the earlier grading; they are ignored.")
    stale = [c for c in grading.cells if c.stale]
    if stale and not args.changed:
        raise Refused(CHANGED_MATERIAL.format(token=stale[0].token) + f" ({len(stale)} cell(s) changed.)")
    todo = [c for c in grading.cells if c.state is None or c.stale or not c.graded]
    send = [c for c in todo if not c.empty]
    # The blinding self-test of every request this run would send, all of it as sent, before anything is sent.
    hits = [(c, found) for c in send for found in [_scan(grading, c, _request_text(c.material, c.allowed))] if found]
    if hits:
        raise Refused("\n".join(f"Cell {c.token}: the grader material reveals {', '.join(found)}; nothing was sent."
                                for c, found in hits))
    total = len(grading.cells)
    backend = ledger = route = preset = None
    if send:
        preset = providers.PRESETS[analysis.GRADER_PROVIDER]
        base_url = preset.base_url or ""
        route = endpoint.route_for(preset, base_url, model=analysis.GRADER_MODEL)
        accept_route(preset, base_url, route, set(), accept=args.accept_data_notice, again="--accept-data-notice",
                     what="the synthetic grounding answers")
        api_key, source = _key_or_refuse(preset, base_url, analysis.GRADER_MODEL, route)
        conn = Connection(base_url=base_url, model=analysis.GRADER_MODEL, api_key=api_key, key_source=source,
                          knobs={}, route=route)
    if not (grading.folder / GRADING_FILE).is_file():
        write_json(grading.folder / GRADING_FILE, grading.state)
    for c in todo:
        if c.empty:
            c.state = {**_new_state(c, _fresh_state(c)), "graded_at": iso(utc_now())}
            write_json(c.path, c.state)
    if not send:
        say(f"Every cell is graded ({total}; {sum(c.empty for c in grading.cells)} without an answer are failed "
            "in code).")
        return 0
    ledger = CalibrationLedger(Ledger())
    backend = backend_factory(preset, conn)
    try:
        paced = bool(getattr(type(backend), "paces", False))
        pacer = backend.pacer if paced else PROCESS_PACER
        pacer.prepare(pace_key(backend.origin), providers.pacing_seed(preset, False))
        pacing = Pacing(pace_key(backend.origin), pacer, here=not paced, label=route.label)

        def on_usage(usage: Any) -> None:
            ledger.record(preset.id, route.label, analysis.GRADER_MODEL, usage)

        def on_rejected(_err: Any) -> None:
            ledger.record_rejected(preset.id, route.label, analysis.GRADER_MODEL)

        for done, c in enumerate(send, 1):
            # A first reply kept from a retry a provider error stopped: only the retry is sent again.
            pending = (c.state or {}).get("pending") if not c.stale else None
            try:
                result = ask_grader(backend, c.material, c.allowed, pacing=pacing, on_usage=on_usage,
                                    on_rejected=on_rejected, passed=[i.id for i in c.items if i.auto],
                                    pending=pending)
            except (LLMError, RetryStopped) as raised:
                exc = raised.error if isinstance(raised, RetryStopped) else raised
                if exc.partial_usage is not None or exc.streamed:
                    ledger.record(preset.id, route.label, analysis.GRADER_MODEL, exc.partial_usage)
                if isinstance(raised, RetryStopped):
                    base = _current_state(c) if pending else _new_state(c, _fresh_state(c))
                    c.state = {**base, "pending": raised.pending}
                    write_json(c.path, c.state)
                say(f"Stopped at cell {c.token} of {total}: {message_for(exc, 'terminal')} Run the same command "
                    "again to continue.")
                return 1
            base = _current_state(c) if pending else _new_state(c, _fresh_state(c))
            base.pop("pending", None)
            c.state = {**base, "ai": result, "graded_at": iso(utc_now())}
            write_json(c.path, c.state)
            say(f"Cell {c.token}: graded ({done} of {len(send)} to send"
                + ("; the reply could not be read" if "unparsed" in result else "") + ").")
    finally:
        backend.close()
    say(f"Every cell is graded ({total}; {sum(c.empty for c in grading.cells)} without an answer are failed in "
        "code). Next: review --dir.")
    return 0


# --- review: the owner judges first, then sees Opus's verdicts --------------------------------------------------
def targeted_reasons(cell: GradingCell) -> list[str]:
    """Why a graded cell goes to the owner whatever the random sample: Opus's reply unread or flagged, a low
    confidence, an injection hint, a number without a tool source (whatever Opus said), a cell Opus wrote."""
    if cell.empty:
        return []
    reasons = []
    ai = cell.ai
    if cell.unparsed:
        reasons.append("unparsed")
    elif ai is not None and (ai["confidence"] == "low" or any(v["confidence"] == "low"
                                                              for v in ai["items"].values())):
        reasons.append("low_confidence")
    if (ai is not None and ai.get("injection")) or scoring.injection_hint(cell.record):
        reasons.append("injection")
    if any(i.check == "tool_sourced" for i in cell.judged):
        reasons.append("unsourced_number")
    if own_model(cell.run):
        reasons.append("own_model")
    return reasons


def review_state(grading: Grading) -> dict[str, Any]:
    """``review.json``: drawn once every cell is graded, never redrawn.  The pool is the graded cells no other
    reason queues, in token order; the sample is ``random.Random(seed).sample(pool, k)`` with k the larger of 15 %
    and 35 cells (all of the pool when smaller).  The queue (token → reasons) follows the cells' reasons; a cell
    ``grade-ai --changed`` graded again that is outside the pool and has no other reason is queued as
    ``changed``."""
    import assistant_eval_analysis as analysis

    path = grading.folder / REVIEW_FILE
    graded = sum(c.graded for c in grading.cells)
    if graded < len(grading.cells):
        raise Refused(f"Opus has graded {graded} of {len(grading.cells)} cells; run grade-ai first.")
    if path.is_file():
        review = read_json(path)
    else:
        pool = [c.token for c in grading.cells if not c.empty and not targeted_reasons(c)]
        k = min(len(pool), max(math.ceil(analysis.REVIEW_SHARE * len(pool)), analysis.REVIEW_FLOOR))
        sample = sorted(random.Random(grading.state["seed"]).sample(pool, k), key=lambda t: int(t.lstrip("#")))
        review = {"seed": grading.state["seed"], "share": analysis.REVIEW_SHARE, "floor": analysis.REVIEW_FLOOR,
                  "pool": pool, "sample": sample, "all": False, "queue": {}}
    queue = {}
    for c in grading.cells:
        reasons = targeted_reasons(c)
        if c.token in review["sample"]:
            reasons.append("sample")
        if review["all"] and not c.empty:
            reasons.append("all")
        if not reasons and not c.empty and c.token not in review["pool"] and (c.state or {}).get("history"):
            # Graded again by --changed and no longer targeted: outside the pool the sample estimates, so the
            # owner reviews it.
            reasons.append("changed")
        if reasons:
            queue[c.token] = reasons
    if queue != review["queue"] or not path.is_file():
        review["queue"] = queue
        write_json(path, review)
    return review


def _owner_done(state: Mapping[str, Any] | None) -> bool:
    return bool(state and state.get("owner"))


def _blinded_reason(grading: Grading, cell: GradingCell, text: str) -> str:
    """One of Opus's reasons as the owner sees it: blind, or withheld when the blinding cannot clean it."""
    shown = scoring.blind(text, grading.names)
    return REASON_WITHHELD if _scan(grading, cell, shown) else shown


def _ask_letter(ask: Callable[[str], str | None], say: Callable[[str], None], prompt: str,
                choices: Mapping[str, Any], *, keep: Any = None) -> tuple[bool, Any]:
    """``(stopped, choice)``: one of *choices* by its letter; Enter gives *keep* when one is given (else asks
    again); q stops."""
    while True:
        answer = ask(prompt)
        if answer is None or answer.strip().lower() in ("q", "quit", "stop"):
            return True, None
        word = answer.strip().lower()
        if not word and keep is not None:
            return False, keep
        if word in choices:
            return False, choices[word]
        say(f"Type {' / '.join(choices)}" + (", or Enter to keep it" if keep is not None else "")
            + " (q stops and saves).")


def _letters(allowed: Sequence[str]) -> dict[str, str]:
    return {k: v for k, v in LETTERS.items() if v in allowed}


def _judge_first(cell: GradingCell, given: dict[str, Any], ask: Callable[[str], str | None],
                 say: Callable[[str], None]) -> bool:
    """The owner's own verdicts, nothing of Opus's shown: each judged item, then understandability.  *given* is
    filled in place (the verdicts already given are kept); False when the owner stopped."""
    items = given.setdefault("items", {})
    for i in cell.judged:
        if i.id in items:
            continue
        choices = _letters(i.allowed)
        stopped, verdict = _ask_letter(ask, say, f"{i.id}{' (critical)' if i.critical else ''}: {i.what} — "
                                       f"{' / '.join(choices)}: ", choices)
        if stopped:
            return False
        items[i.id] = verdict
    if given.get("understandability") is None:
        say(f"Understandability: {scoring.UNDERSTANDABILITY_TEXT}")
        stopped, u = _ask_letter(ask, say, "1 / 2 / 3: ", {"1": 1, "2": 2, "3": 3})
        if stopped:
            return False
        given["understandability"] = u
    return True


def _reveal(grading: Grading, cell: GradingCell, first: Mapping[str, Any], ask: Callable[[str], str | None],
            say: Callable[[str], None]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Opus's verdicts after the owner's: each shown with the owner's, and each disagreement asked again (Enter
    keeps the owner's).  ``(final, changes)``; ``None`` when the owner stopped.  Without a readable reply of Opus's
    the owner's first verdicts are final."""
    ai = cell.ai
    final = {"items": dict(first["items"]), "understandability": first["understandability"]}
    if ai is None:
        return final, []
    say("Opus's grading (shown after yours):")
    say(f"  the cell ({ai['confidence']}): {_blinded_reason(grading, cell, ai['reason'])}")
    for i in cell.judged:
        mine, theirs = first["items"][i.id], ai["items"][i.id]
        say(f"  {i.id}: you {mine}; Opus {theirs['verdict']} ({theirs['confidence']}): "
            f"{_blinded_reason(grading, cell, theirs['reason'])}" + ("   <- differs" if mine != theirs["verdict"]
                                                                       else ""))
    say(f"  understandability: you {first['understandability']}; Opus {ai['understandability']}"
        + ("   <- differs" if first["understandability"] != ai["understandability"] else ""))
    changes = []
    for i in cell.judged:
        mine, theirs = first["items"][i.id], ai["items"][i.id]["verdict"]
        if mine == theirs:
            continue
        choices = _letters(i.allowed)
        stopped, verdict = _ask_letter(ask, say, f"{i.id}: Enter keeps yours ({mine}), {' / '.join(choices)} "
                                       "changes it: ", choices, keep=mine)
        if stopped:
            return None
        final["items"][i.id] = verdict
        if verdict != mine:
            changes.append({"id": i.id, "opus": theirs, "first": mine, "owner": verdict})
    if first["understandability"] != ai["understandability"]:
        stopped, u = _ask_letter(ask, say, f"understandability: Enter keeps yours ({first['understandability']}), "
                                 "1 / 2 / 3 changes it: ", {"1": 1, "2": 2, "3": 3}, keep=first["understandability"])
        if stopped:
            return None
        final["understandability"] = u
        if u != first["understandability"]:
            changes.append({"id": "understandability", "opus": ai["understandability"],
                            "first": first["understandability"], "owner": u})
    return final, changes


def cmd_review(args: argparse.Namespace, *, ask: Callable[[str], str | None] | None = None,
               say: Callable[[str], None] = print) -> int:
    def default_ask(text: str) -> str | None:
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    try:
        with _Lock(Path(args.dir), "review"):
            return _review(args, ask or default_ask, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _review(args: argparse.Namespace, ask: Callable[[str], str | None], say: Callable[[str], None]) -> int:
    args.results = []
    grading = load_grading(args, create=False)
    _refuse_stale(grading)
    review = review_state(grading)
    if args.all and not review["all"]:
        review["all"] = True
        write_json(grading.folder / REVIEW_FILE, review)
        review = review_state(grading)
    by_token = {c.token: c for c in grading.cells}
    if args.regrade is not None:
        token = args.regrade if args.regrade.startswith("#") else f"#{args.regrade}"
        cell = by_token.get(token)
        if cell is None or not _owner_done(cell.state):
            raise Refused(f"No reviewed cell {token}.")
        plan = [cell]
    else:
        plan = [by_token[t] for t in sorted(review["queue"], key=lambda t: int(t.lstrip("#")))
                if not _owner_done(by_token[t].state)][:args.sitting or None]
    # The blinding self-test of what this sitting shows, before the first question.
    for cell in plan:
        found = _scan(grading, cell, cell.material)
        if found:
            raise Refused(f"Cell {cell.token}: the grader material reveals {', '.join(found)}; nothing is shown.")
    reviewed = 0
    for cell in plan:
        regrade = args.regrade is not None
        say("=" * 100)
        say(f"Cell {cell.token}" + (" (re-grade)" if regrade else ""))
        say(cell.material)
        given: dict[str, Any] = {} if regrade else dict(cell.state.get("owner_first") or {})
        if not _judge_first(cell, given, ask, say):
            if regrade:
                say("Stopped; the re-grade was not saved.")
            else:
                cell.state["owner_first"] = given
                write_json(cell.path, cell.state)
                say("Stopped; the verdicts given so far are saved.")
            return 0
        if not regrade:
            cell.state["owner_first"] = given
            write_json(cell.path, cell.state)
        revealed = _reveal(grading, cell, given, ask, say)
        if revealed is None:
            say("Stopped; the re-grade was not saved." if regrade else
                "Stopped; your own verdicts are saved, Opus's are shown again next time.")
            return 0
        final, changes = revealed
        if regrade:
            cell.state.setdefault("regrades", []).append({"first": given, "owner": final, "changes": changes,
                                                          "at": iso(utc_now())})
            write_json(cell.path, cell.state)
            say(f"Cell {cell.token} graded again.")
            return 0
        cell.state.update(owner={**final, "at": iso(utc_now())}, changes=changes)
        write_json(cell.path, cell.state)
        reviewed += 1
    left = sum(1 for t in review["queue"] if not _owner_done(by_token[t].state))
    say(f"Sitting done ({reviewed} cells). {left} cell(s) still wait: run review again to continue." if left
        else f"Every queued cell is reviewed ({len(review['queue'])}).")
    return 0


# --- summary: the final grades, the agreement and results.jsonl -------------------------------------------------
def final_grade(cell: GradingCell, review: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The cell's grade: the owner's (the latest re-grade wins) when reviewed, else Opus's when the cell is not
    queued; ``None`` while it waits (not graded, or queued and not reviewed)."""
    import assistant_eval_analysis as analysis

    state = cell.state or {}
    if cell.empty and state:
        return {"grader": "none", "outcome": "failed", "verdicts": {}, "understandability": None}
    if _owner_done(state):
        owner = (state.get("regrades") or [{}])[-1].get("owner") or state["owner"]
        return {"grader": "owner", "outcome": scoring.outcome(cell.items, owner["items"]),
                "verdicts": owner["items"], "understandability": owner["understandability"]}
    if review is None or cell.token in review["queue"] or cell.ai is None:
        return None
    ai = cell.ai
    verdicts = {k: v["verdict"] for k, v in ai["items"].items()}
    return {"grader": analysis.GRADER_MODEL, "outcome": scoring.outcome(cell.items, verdicts), "verdicts": verdicts,
            "understandability": ai["understandability"]}


def summary_line(record: Mapping[str, Any], run: Mapping[str, Any], *, key_version: int | None,
                 cell: GradingCell | None = None, grade: Mapping[str, Any] | None = None,
                 review: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One line of ``results.jsonl`` (``line_format`` 2): no answer or tool text (P3 §7.7).  The grade fields of
    the final grade (*grade*: the owner's when reviewed, else Opus's) and Opus's own outcome next to them; the
    run's counts the §7.6 analysis reads.  A cell not run for its context (no *cell*) is a line with outcome
    ``not_applicable``."""
    turns = record.get("turns") or []
    usage = [u for t in turns for u in t.get("usage") or []]
    reported = [u for u in usage if u.get("reported")]
    tokens_in = sum(int(u.get(k) or 0) for u in reported for k in ("input_uncached", "cache_read", "cache_write"))
    model = record["model"]
    refused = sorted({str(c.get("name")) for t in turns for c in t.get("tool_calls") or []
                      if not c.get("ok") and "not available at this assistant level" in str(c.get("result") or "")})
    graded: dict[str, Any] = {"grader": "none", "reviewed": False, "review_reasons": [], "outcome": "not_applicable",
                              "missing": [], "wrong": [], "critical_missing": [], "form_missing": [],
                              "invented_numbers": 0, "no_answer": False, "cut_off": False, "understandability": None,
                              "ai_outcome": None, "ai_understandability": None, "ai_confidence": None,
                              "items_overridden": [], "diagnostics": {}}
    if cell is not None and grade is not None:
        ai = cell.ai
        ai_verdicts = {k: v["verdict"] for k, v in ai["items"].items()} if ai else {}
        graded.update(grader=grade["grader"], reviewed=grade["grader"] == "owner",
                      review_reasons=list((review or {}).get("queue", {}).get(cell.token, [])),
                      outcome=grade["outcome"], **scoring.verdict_counts(cell.items, grade["verdicts"]),
                      no_answer=cell.empty, cut_off=scoring.cut_off(_turn_views(record)),
                      understandability=grade["understandability"],
                      ai_outcome=scoring.outcome(cell.items, ai_verdicts) if ai else None,
                      ai_understandability=ai["understandability"] if ai else None,
                      ai_confidence=ai["confidence"] if ai else None,
                      items_overridden=sorted(k for k, v in grade["verdicts"].items()
                                              if ai and grade["grader"] == "owner" and ai_verdicts.get(k) != v),
                      diagnostics=dict(cell.diagnostics))
    return {
        "line_format": 2, "rubric_version": scoring.RUBRIC_VERSION, "date": str(record.get("ran_at") or "")[:10],
        "grader": graded.pop("grader"), "authoritative": True,
        "reviewed": graded.pop("reviewed"), "review_reasons": graded.pop("review_reasons"),
        "flextool_version": record.get("flextool_version"),
        "commit": (record.get("git") or {}).get("commit"), "set_version": record.get("set_version"),
        "key_version": key_version, "provider": model["preset"], "model": run.get("model"),
        "automatic_level": model.get("automatic_level"), "level_setting": record["cell"]["level_setting"],
        "window": model.get("window"), "free_tier": model.get("free_tier"),
        "knobs_relaxed": bool(model.get("knobs_relaxed")), "question": record["cell"]["question"],
        "class": record["cell"]["class"], **graded,
        "calls": sum(int(t.get("calls") or 0) for t in turns), "rounds": sum(int(t.get("rounds") or 0) for t in turns),
        "tokens_in": tokens_in, "tokens_out": sum(int(u.get("output") or 0) for u in reported),
        "tokens_not_reported": len(usage) - len(reported),
        "prompt_ratio": (record.get("budget") or {}).get("prompt_ratio"),
        "repeat": record["cell"].get("repeat", 1), "level_source": model.get("level_source"),
        "cap_hits": sum(1 for t in turns if t.get("limit_hit")),
        "length_stops": sum(1 for t in turns if (t.get("error") or {}).get("code") == "length"),
        "fit_compactions": sum(int(t.get("fit_compactions") or 0) for t in turns),
        "truncations": sum(int(t.get("truncated") or 0) for t in turns),
        "refused_tools": refused, "cpu_offload": model.get("cpu_offload"),
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
        raise Refused("These results folders come from different grounding projects; grade them apart.")
    return loaded


def append_lines(path: Path, lines: Iterable[dict]) -> int:
    lines = list(lines)
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=False) + "\n")
    return len(lines)


def reviewed_cells(grading: Grading, review: Mapping[str, Any]) -> list[Any]:
    """The reviewed cells Opus's reply was read for, as agreement pairs: the owner's first verdicts (given before
    Opus's were shown) against Opus's."""
    import assistant_eval_analysis as analysis

    out = []
    for c in grading.cells:
        first = (c.state or {}).get("owner_first") or {}
        if not _owner_done(c.state) or c.ai is None or c.empty:
            continue
        ai = {k: v["verdict"] for k, v in c.ai["items"].items()}
        reasons = review["queue"].get(c.token, [])
        out.append(analysis.ReviewedCell(
            token=c.token, question=c.question.id, items=tuple((first["items"][i.id], ai[i.id]) for i in c.judged),
            outcome=(scoring.outcome(c.items, first["items"]), scoring.outcome(c.items, ai)),
            understandability=(first["understandability"], c.ai["understandability"]),
            sample=c.token in review["sample"],
            own_matched=[r for r in reasons if r not in ("sample", "all")] == ["own_model"]))
    return out


def cmd_summary(args: argparse.Namespace, *, say: Callable[[str], None] = print) -> int:
    try:
        if args.record and args.dir and not args.results:
            with _Lock(Path(args.dir), "summary --record"):       # it writes the cells' states
                return _summary(args, say)
        return _summary(args, say)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


def _summary(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    import assistant_eval_analysis as analysis

    if args.results or not args.dir:
        raise Refused(SUMMARY_DIR)
    args.results = []
    grading = load_grading(args, create=False)
    _refuse_stale(grading)
    graded = sum(c.graded for c in grading.cells)
    review = review_state(grading) if graded == len(grading.cells) else None
    grades = {c.token: final_grade(c, review) for c in grading.cells}
    for folder in grading.state["results"]:
        cells = [c for c in grading.cells if c.key[0] == folder]
        run = read_json(Path(folder) / RUN_FILE)
        say(f"{run.get('provider')} {run.get('model')} ({folder})")
        table: dict[tuple[str, str], dict[str, int]] = {}
        for c in cells:
            row = table.setdefault((c.record["cell"]["level_setting"], c.question.cls), {
                "n": 0, **{o: 0 for o in scoring.OUTCOMES}, "no answer": 0, "waiting": 0, "invented": 0,
                **{f"u{u}": 0 for u in scoring.UNDERSTANDABILITY}})
            row["n"] += 1
            grade = grades[c.token]
            if grade is None:
                row["waiting"] += 1
                continue
            row[grade["outcome"]] += 1
            row["no answer"] += c.empty
            row["invented"] += scoring.verdict_counts(c.items, grade["verdicts"])["invented_numbers"]
            if grade["understandability"] is not None:
                row[f"u{grade['understandability']}"] += 1
        for level in scoring.LEVELS:
            for cls in scoring.CLASSES:
                row = table.get((level, cls))
                if row:
                    say(f"  {level:<7} {cls:<9} n {row['n']:>3}  answered {row['answered']:>3}  incomplete "
                        f"{row['incomplete']:>3}  escalated {row['escalated']:>3}  failed {row['failed']:>3} "
                        f"(no answer {row['no answer']})  waiting {row['waiting']:>3}  invented {row['invented']:>3}"
                        f"  understandability 1/2/3 {row['u1']}/{row['u2']}/{row['u3']}")
        na = sum(1 for f, _r, _rec, _c in grading.not_applicable if str(f) == folder)
        if na:
            say(f"  not applicable (context): {na}")
    if review is None:
        say(f"Opus has graded {graded} of {len(grading.cells)} cells; run grade-ai first.")
    else:
        pairs = reviewed_cells(grading, review)
        unreviewed = sum(1 for c in grading.cells if not c.empty and not _owner_done(c.state))
        text = analysis.agreement_report(pairs, unreviewed=unreviewed,
                                         whole_rest=len(review["sample"]) == len(review["pool"]),
                                         unparsed=sum(1 for c in grading.cells if c.unparsed),
                                         seed=int(grading.state["seed"]))
        (grading.folder / AGREEMENT_FILE).write_text(text + "\n", encoding="utf-8")
        say(text)
    if not args.record:
        return 0
    if review is None:
        raise Refused(f"Opus has graded {graded} of {len(grading.cells)} cells; run grade-ai first.")
    waiting = sum(1 for g in grades.values() if g is None)
    if waiting:
        raise Refused(f"{waiting} cells still wait for review.")
    lines = []
    for c in grading.cells:
        tag = f"{len(c.state.get('history') or [])}:{len(c.state.get('regrades') or [])}:{grades[c.token]['grader']}"
        if tag in (c.state.get("recorded") or []):
            continue
        lines.append((c, tag, summary_line(c.record, c.run, key_version=grading.key.key_version, cell=c,
                                           grade=grades[c.token], review=review)))
    done_na = {tuple(x) for x in grading.state.get("recorded_not_applicable") or []}
    na_lines = [((str(f), cid), summary_line(r, run, key_version=grading.key.key_version))
                for f, run, r, cid in grading.not_applicable if (str(f), cid) not in done_na]
    total = append_lines(Path(args.results_file), [line for _c, _t, line in lines] + [x for _k, x in na_lines])
    for c, tag, _line in lines:
        c.state.setdefault("recorded", []).append(tag)
        write_json(c.path, c.state)
    if na_lines:
        grading.state["recorded_not_applicable"] = sorted([*map(list, done_na), *(list(k) for k, _x in na_lines)])
        write_json(grading.folder / GRADING_FILE, grading.state)
    say(f"{total} line(s) appended to {args.results_file}.")
    return 0


# --- calibrate (tools/assistant_eval_analysis.py) ---------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace, *, say: Callable[[str], None] = print) -> int:
    import assistant_eval_analysis as analysis

    return analysis.cmd_calibrate(args, say=say)


# --- the command line -----------------------------------------------------------------------------------------
def _abs_path(text: str) -> Path:
    """A command-line path made absolute at once: the runs of ``build`` work in other folders, so a relative path
    would point elsewhere there."""
    return Path(text).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant_eval.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build the synthetic grounding project and its marker (solves)")
    p.add_argument("--out", required=True, type=_abs_path, help="the folder to build the project in")

    p = sub.add_parser("run", help="ask the grounding questions at every level setting")
    p.add_argument("--project", required=True, type=_abs_path, help="a project made by build")
    p.add_argument("--provider", required=True, help="the provider preset id (ollama, openrouter, gemini, ...)")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", help="the address, for a custom endpoint or a model server elsewhere")
    p.add_argument("--levels", default="basic,standard,advanced")
    p.add_argument("--ollama-context", help="the context of each setting, e.g. basic=16384,standard=32768,"
                   "advanced=65536 (the model's variants gpt-oss-20b-16k ... made once with ollama create)")
    p.add_argument("--allow-cpu-offload", action="store_true",
                   help="run a variant that loaded with the asked context but is partly on the CPU (the share "
                        "on the CPU is recorded per cell); without it such a setting is not applicable")
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
    p.add_argument("--out", required=True, type=_abs_path, help=f"the results folder (e.g. {DEFAULT_OUT}/<model>)")
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("grade-ai", help="Opus grades every cell, blind (one request per cell; continues)")
    p.add_argument("results", nargs="+", type=_abs_path, help="the results folders of every model, graded together")
    p.add_argument("--dir", required=True, type=_abs_path, help="the grading folder (all grading state lives here)")
    p.add_argument("--single-model", action="store_true", help="grade one results folder alone (not blind)")
    p.add_argument("--changed", action="store_true",
                   help="grade again the cells whose answer, key or rubric changed (their review starts over)")
    p.add_argument("--accept-data-notice", action="store_true",
                   help="accept Anthropic's data notice for this run only")
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("review", help="the owner reviews the queued cells: own verdict first, then Opus's")
    p.add_argument("--dir", required=True, type=_abs_path, help="the grading folder")
    p.add_argument("--sitting", type=int, default=DEFAULT_SITTING, help="cells to review in one sitting")
    p.add_argument("--regrade", metavar="#N", help="review cell #N again (the latest review wins)")
    p.add_argument("--all", action="store_true", help="queue every cell (full manual grading)")
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("summary", help="outcomes and agreement of a grading; --record appends results.jsonl")
    p.add_argument("results", nargs="*", help=argparse.SUPPRESS)
    p.add_argument("--dir", type=_abs_path, help="the grading folder")
    p.add_argument("--record", action="store_true")
    p.add_argument("--results-file", default=str(scoring.RESULTS_PATH), help=argparse.SUPPRESS)
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("calibrate", help="the evidence tables of results.jsonl for the owner")
    p.add_argument("results_file", nargs="?", default=str(scoring.RESULTS_PATH))
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "build": cmd_build, "run": cmd_run, "grade-ai": cmd_grade_ai, "review": cmd_review, "summary": cmd_summary,
    "calibrate": cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
