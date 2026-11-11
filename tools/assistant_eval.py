"""The assistant's level calibration on the grounding questions (P3 §6–§7).  A developer tool: not packaged, never
run in CI, never collected by pytest.

    python tools/assistant_eval.py build --out DIR
    python tools/assistant_eval.py run --project DIR --provider ID --model M --out RESULTS [...]
    python tools/assistant_eval.py run ... --out RESULTS --skip CELL --skip-reason TEXT  (or --unskip CELL)
    python tools/assistant_eval.py grade-ai RESULTS ... --dir GRADING [--accept-data-notice] [--changed]
    python tools/assistant_eval.py grade-ai RESULTS ... --dir GRADING --export-requests OUT  (then --import-replies OUT)
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
that is not the model's behaviour stopped a cell: after a temporary one its checkpoint is kept and --resume
continues it from the request that failed; after any other nothing of it is saved; and a run that went on past
cells kept after a tool call the service could not read, also when it then ended for the day's budget) · 2 usage
or a refused project / results folder · 3 the data notice was not accepted.
"""
from __future__ import annotations

import argparse
import contextlib
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
    requests that reached the provider before the stop (they count against the daily budget).  *provider*: what
    the provider said in the stopping answer (:func:`failure_json`'s fields, ``None`` when no answer said
    anything); *attempts*: every failed attempt of the cell (:func:`failure_json`), the stopping one last;
    *said*: the console lines of what the provider said."""

    def __init__(self, text: str, requests_sent: int = 0, code: str = "", *, provider: dict | None = None,
                 attempts: Sequence[Mapping[str, Any]] = (), said: Sequence[str] = ()) -> None:
        super().__init__(text)
        self.requests_sent = requests_sent
        self.code = code
        self.provider = provider
        self.attempts = [dict(a) for a in attempts]
        self.said = list(said)
        #: The stopped turn's failed attempts (an earlier run's first) and seconds so far, for its checkpoint.
        self.current: dict[str, Any] = {}
        #: The usage of the requests after the last checkpoint (a streamed break-off): not in any turn's usage.
        self.lost_usage: list[dict[str, Any]] = []
        self.kept_text = ""


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


def requests_today(cells: Iterable[Mapping[str, Any]], today: str, spent: Iterable[Mapping[str, Any]] = (),
                   partials: Iterable[Mapping[str, int]] = ()) -> int:
    """The requests this results folder sent on the UTC day *today* (``YYYY-MM-DD``): those of its saved cells
    (by their ``requests_by_day``, else ``ran_at`` and ``requests_sent``), those of the cells that stopped, were
    replaced or whose checkpoint was thrown away (*spent*: by ``by_day``, ``day`` or ``ran_at``) and those the
    checkpoints of unfinished cells hold (*partials*, ``{day: requests}``).  Each request is counted once."""
    def of_record(c: Mapping[str, Any]) -> int:
        days = c.get("requests_by_day")
        if isinstance(days, dict):
            return int(days.get(today) or 0)
        return int(c.get("requests_sent") or 0) if str(c.get("ran_at") or "").startswith(today) else 0

    def of_spent(e: Mapping[str, Any]) -> int:
        days = e.get("by_day")
        if isinstance(days, dict):
            return int(days.get(today) or 0)
        day = e.get("day") or str(e.get("ran_at") or "")[:10]
        return int(e.get("requests") or 0) if day == today else 0

    return (sum(of_record(c) for c in cells) + sum(of_spent(e) for e in spent)
            + sum(int(d.get(today) or 0) for d in partials))


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


#: run.json's log of how each run sent its requests: ``[{"started", "pace", "stream"}]`` (``run --pace``,
#: ``--no-stream``).  Part of the folder's record, not of its identity: a ``--resume`` may change them.  A folder
#: from before has none (it was paced by the pacing alone, and streamed).
RUNS_KEY = "runs"


def _log_run(out: Path, sending: Mapping[str, Any]) -> None:
    run_file = Path(out) / RUN_FILE
    run = read_json(run_file)
    run[RUNS_KEY] = [*(run.get(RUNS_KEY) or []), {"started": iso(utc_now()), **sending}]
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


#: run.json's record of the cells the owner skipped (``run --skip``): cell id → ``{status, reason, skipped_at,
#: attempts, requests}``.  Part of the folder's record, not of its identity (:data:`IDENTITY_KEYS`): a later
#: ``--resume`` is not refused, it leaves the cell out.  ``run --unskip`` moves an entry to :data:`UNSKIPPED_KEY`
#: with its ``unskipped_at``, and the next ``--resume`` asks the cell.
SKIPPED_KEY = "skipped"
UNSKIPPED_KEY = "unskipped"
SKIPPED_STATUS = "not applicable (skipped by owner)"


def skipped_cells(run: Mapping[str, Any]) -> dict[str, dict]:
    """The cells of a results folder the owner skipped (its run.json), cell id → entry."""
    return dict(run.get(SKIPPED_KEY) or {})


def parse_cell_id(cid: str) -> tuple[str, str, int]:
    """``(question, level, repeat)`` of a cell id ``G01__small__r1``."""
    question, level, repeat = cid.split("__")
    return question, level, int(repeat.removeprefix("r"))


def known_cells(run: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Every cell id a results folder can have: each question × level × repeat of its run.json matrix (the
    automatic-level question at every level), or of its saved cells for a folder from before the matrix, and the
    saved and skipped cells."""
    matrix = run.get(MATRIX_KEY)
    if matrix is None:
        seen = [r["cell"] for r in cells.values()]
        matrix = {"questions": {c["question"] for c in seen}, "levels": {c["level_setting"] for c in seen},
                  "repeat": max((int(c.get("repeat") or 1) for c in seen), default=1)}
    planned = {cell_id(q, lv, r) for q in matrix["questions"] for lv in matrix["levels"]
               for r in range(1, int(matrix["repeat"]) + 1)}
    return planned | set(cells) | set(skipped_cells(run))


def _attempts(cid: str, spent: Sequence[Mapping[str, Any]], spent_known: bool, saved: Mapping[str, Any] | None,
              partial: Any = None) -> tuple[int | None, int | None]:
    """``(attempts, requests)`` a cell took: the attempts are every stop and replaced record of spent.json (a
    checkpoint's own list of stops repeats them, so it is not counted) and the saved record; the requests are
    those of spent.json (also a thrown-away checkpoint's), of the saved record and those its checkpoint holds.
    ``(None, None)`` when nothing tells (no spent.json, a folder from before it, no record, no checkpoint)."""
    entries = [e for e in spent if e.get("cell") == cid]
    if not spent_known and saved is None and partial is None:
        return None, None
    requests = (sum(int(e.get("requests") or 0) for e in entries) + int((saved or {}).get("requests_sent") or 0)
                + sum(held_requests(partial).values()))
    return sum(1 for e in entries if e.get("code") != "discarded") + (saved is not None), requests


def _closest(cid: str, known: Iterable[str]) -> str:
    import difflib

    close = difflib.get_close_matches(cid, sorted(known), n=3, cutoff=0.6)
    return f" (closest: {', '.join(close)})" if close else " (a cell id is like G01__small__r1)"


def mark_skipped(out: Path, identity: Mapping[str, Any], skip: Sequence[str], unskip: Sequence[str],
                 reason: str | None, say: Callable[[str], None]) -> None:
    """``run --skip`` / ``--unskip``: record in run.json that the owner leaves the cells *skip* out (with *reason*,
    the time and the attempts spent.json and the saved records show) and that *unskip* are asked again by the next
    ``--resume``.  Nothing is asked.  :class:`Refused` before anything is written for a folder of another model
    or project, an unknown cell id (the closest named), a finished or not-applicable cell to skip, a cell
    skipped already, or one to unskip that is not skipped."""
    run_file = Path(out) / RUN_FILE
    if not run_file.is_file():
        raise Refused(f"No results in {out}: --skip and --unskip are for a results folder of assistant_eval run.")
    run = read_json(run_file)
    missing = object()
    if any(run.get(k, IDENTITY_DEFAULTS.get(k, missing)) != identity[k] for k in IDENTITY_KEYS):
        raise Refused(OTHER_RESULTS)
    if skip and not (reason or "").strip():
        raise Refused("Give the reason with --skip-reason TEXT; it is kept with the skip.")
    if reason and not skip:
        raise Refused("--skip-reason goes with --skip.")
    both = sorted(set(skip) & set(unskip))
    if both:
        raise Refused(f"Both --skip and --unskip: {', '.join(both)}.")
    cells = read_cells(out)
    known = known_cells(run, cells)
    unknown = [cid for cid in [*skip, *unskip] if cid not in known]
    if unknown:
        raise Refused("\n".join(f"No cell {cid} in {Path(out).name}{_closest(cid, known)}." for cid in unknown))
    skipped = skipped_cells(run)
    problems = []
    for cid in skip:
        saved = cells.get(cid)
        if cid in skipped:
            problems.append(f"{cid} is skipped already ({skipped[cid].get('reason')}); --unskip it first.")
        elif saved is not None and saved.get("not_applicable"):
            problems.append(f"{cid} is not applicable already ({saved['not_applicable']}).")
        elif saved is not None and finished(saved):
            problems.append(f"{cid} is finished: a finished cell is graded, not skipped.")
    quota_na = {cid for cid in unskip if cid not in skipped and (cells.get(cid) or {}).get(NA_SCOPE_KEY) == "cell"}
    problems += [f"{cid} is not skipped." for cid in unskip if cid not in skipped and cid not in quota_na]
    if problems:
        raise Refused("\n".join(problems))
    spent_known = (Path(out) / SPENT_FILE).is_file()
    spent = read_spent(out)
    partials = read_partials(out)
    now = iso(utc_now())
    for cid in dict.fromkeys(skip):
        attempts, requests = _attempts(cid, spent, spent_known, cells.get(cid), partials.get(cid))
        skipped[cid] = {"status": SKIPPED_STATUS, "reason": reason.strip(), "skipped_at": now, "attempts": attempts,
                        "requests": requests}
        say(f"{cid}: {SKIPPED_STATUS}: {reason.strip()} (attempts: "
            f"{'not known' if attempts is None else attempts}).")
    for cid in dict.fromkeys(unskip):
        if cid in quota_na:
            # Not applicable for the provider's quota per request: the record goes (its requests stay counted,
            # written first) and the next --resume asks the cell from the start.
            record = cells[cid]
            if int(record.get("requests_sent") or 0):
                spent.append(_replaced(record, cid))
                write_json(Path(out) / SPENT_FILE, spent)
            run[UNSKIPPED_KEY] = [*(run.get(UNSKIPPED_KEY) or []),
                                  {"cell": cid, "status": f"not applicable ({record.get('not_applicable')})",
                                   "quota": record.get("quota"), "unskipped_at": now}]
            cell_path(out, cid).unlink()
            say(f"{cid}: no longer not applicable; run --resume asks it from the start.")
            continue
        run[UNSKIPPED_KEY] = [*(run.get(UNSKIPPED_KEY) or []), {"cell": cid, **skipped.pop(cid), "unskipped_at": now}]
        say(f"{cid}: no longer skipped; run --resume asks it.")
    run[SKIPPED_KEY] = skipped
    write_json(run_file, run)
    say("Nothing was asked.")


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


def failure_json(err: Any, red: Callable[[str], str]) -> dict[str, Any]:
    """One failed request attempt for spent.json and the records: FlexTool's code, and the HTTP status, the
    provider's error code and its message (redacted, at most ``MAX_PROVIDER_SAID`` characters) when the provider
    answered with an error; ``None`` for each the provider did not say.  For a Google quota answer also the quotas
    it names (metric, quota id, limit, model) and the wait it suggests in seconds (``quota``, ``retry_delay``)."""
    said = err.provider_said
    return {"code": err.code, "status": said.status if said else None, "provider_code": said.code if said else None,
            "message": red(said.message) if said else None,
            "quota": [q.as_json() for q in said.quota] if said else [],
            "retry_delay": said.retry_delay if said else None}


def said_lines(final: Any, failed: Sequence[Any], red: Callable[[str], str],
               earlier_json: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """The console lines after FlexTool's message: what the provider said in the stopping answer, then what it
    said in the earlier failed attempts of the cell (each different text once, with how often): *earlier_json*
    first (the stopped turn's attempts an earlier run kept in its checkpoint, :func:`failure_json`), then
    *failed*."""
    from flextool.assistant.llm_errors import provider_said_line

    lines = []
    line = provider_said_line(final) if final is not None else None
    if line:
        lines.append(red(line))
    earlier: dict[str, int] = {}
    for attempt in earlier_json:
        text = _said_text(attempt)
        if text is not None:
            earlier[text] = earlier.get(text, 0) + 1
    for err in failed:
        if err is not final and err.provider_said is not None:
            text = red(err.provider_said.text())
            earlier[text] = earlier.get(text, 0) + 1
    lines += [f"  earlier attempt{'s' if n > 1 else ''} ({n}): {text}" for text, n in earlier.items()]
    return lines


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


# --- checkpoints: a cell continued from the request that failed (item 216) ---------------------------------------
#: ``<out>/partial/<cell id>.json``: the checkpoint of a cell that has not finished.  Not in ``cells/``, whose
#: ``*.json`` are the records.  It holds what was sent, unredacted (0600): never share a results folder with it.
PARTIAL_DIR = "partial"
PARTIAL_FORMAT = 1
#: The keys of a checkpoint file, all of them always; a file with another set is of another format.
PARTIAL_KEYS = ("format", "cell", "identity", "setting_key", "flextool_version", "git", "created", "updated",
                "turns", "current", "session", "requests_by_day", "stopped", "stops", "resumes")
CURRENT_KEYS = ("turn", "notices", "failed_attempts", "seconds")
#: The keys of one finished turn of a record (and of a checkpoint's ``turns``), as :func:`run_cell` writes them.
TURN_KEYS = ("user", "answer", "tool_calls", "rounds", "calls", "continuations", "stop", "notices", "error",
             "failed_attempts", "limit_hit", "fit_compactions", "truncated", "seconds", "usage", "requests")
#: The keys of one stop of a checkpoint (``stops``), all of them always.
STOP_KEYS = ("code", "turn", "round", "request", "request_sha256", "est_input", "provider", "lost_usage",
             "stopped_at", "pace", "stream")
#: The stops after which the cell is kept and ``--resume`` continues it from its checkpoint: the provider busy or
#: limiting (also its per-minute input-token quota and its daily quota), no connection, no answer in time, and the
#: service unable to read a tool call in the model's answer (``tool_call_unreadable``, Ollama's "error parsing tool
#: call": the service's format problem with one answer, never a graded stop).  Any other stop (key, permission,
#: credit, model, routing, tools, history, a fault) throws the checkpoint away.
TEMPORARY_STOPS = frozenset({"busy", "rpm", "tpm_input", "connection", "timeout", "quota", "tool_call_unreadable"})
#: A cell whose same request the provider refused for its input tokens per minute this many times in a row is not
#: applicable (the request alone is over the quota; CAL-FINDING 2026-11-10): recorded, never asked again.
QUOTA_PER_REQUEST = "provider quota per request"
QUOTA_REPEATS = 2
#: The same request stopped this many times in a row (any other temporary code): the runner says so.
SAME_REQUEST_REPEATS = 3
#: A record's ``not_applicable_scope``: ``"cell"`` = this cell only, whatever its level setting (never asked
#: again by ``--resume``; ``--unskip`` takes it back).  Without it a not-applicable record is the setting's.
NA_SCOPE_KEY = "not_applicable_scope"


def partial_path(out: Path, cid: str) -> Path:
    return Path(out) / PARTIAL_DIR / f"{cid}.json"


def read_partials(out: Path) -> dict[str, Any]:
    """The checkpoint files of a results folder, cell id → content (``None`` for a file that does not read as
    JSON).  Temp files (``….json.<writer>.tmp``) are not matched."""
    folder = Path(out) / PARTIAL_DIR
    if not folder.is_dir():
        return {}
    found: dict[str, Any] = {}
    for p in sorted(folder.glob("*.json")):
        try:
            found[p.stem] = read_json(p)
        except (OSError, ValueError):
            found[p.stem] = None
    return found


def held_requests(partial: Any) -> dict[str, int]:
    """``{UTC day: requests}`` a checkpoint file holds (none for one of another format)."""
    days = partial.get("requests_by_day") if isinstance(partial, dict) else None
    if not isinstance(days, dict):
        return {}
    return {str(d): int(n) for d, n in days.items() if isinstance(n, int) and not isinstance(n, bool)}


def unfinished_text(partial: Any) -> str:
    """``unfinished: checkpoint at turn 2, request 7`` (grade-ai's refusal and the console): only the position is
    read from a checkpoint, never its history (which is unredacted)."""
    current = partial.get("current") if isinstance(partial, dict) else None
    turn = current.get("turn") if isinstance(current, dict) else "?"
    return f"unfinished: checkpoint at turn {turn}, request {sum(held_requests(partial).values()) + 1}"


def setting_key(setting: Setting) -> dict[str, Any]:
    """What a cell's requests are made from beyond the folder's identity: the level, the model id sent, the window
    and its source, the caps and result caps, the compaction rhythm, the history policy, the free plan, the route
    kind and the hashes of the instructions and tools.  A checkpoint of another key starts its cell again."""
    from flextool.assistant import budget, prompt
    from flextool.assistant.session import text_hash, tool_specs, tools_hash

    b = setting.budget
    history, history_turns = budget.history_policy(setting.level)
    caps = {k: getattr(b.result_caps, k) for k in ("max_rows", "max_chars", "max_results", "max_items")}
    key = {"level": setting.level, "model": setting.model, "window": b.window, "window_source": b.window_source,
           "caps": {"max_rounds": b.level.max_rounds, "max_calls_per_turn": b.level.max_calls_per_turn,
                    "max_calls_per_round": b.level.max_calls_per_round,
                    "max_continuations": b.level.max_continuations, "max_output_tokens": b.max_output_tokens},
           "result_caps": caps, "compaction_every": b.compaction_every, "keep_recent_turns": b.keep_recent_turns,
           "history": [history, history_turns], "free_tier": b.free_tier,
           "route": setting.route.kind if setting.route else None,
           "system_hash": text_hash(list(prompt.system_layers("terminal", setting.level))),
           "tools_hash": tools_hash(tool_specs(setting.level))}
    return json.loads(json.dumps(key))


def partial_problem(partial: Any, identity: Mapping[str, Any], key: Mapping[str, Any]) -> str | None:
    """Why a checkpoint file cannot continue its cell here (``None``: it can): another format (a key missing or
    unknown: never a default), another model or project, another level setting, other instructions or tools."""
    if not isinstance(partial, dict) or set(partial) != set(PARTIAL_KEYS) or partial.get("format") != PARTIAL_FORMAT:
        return "a checkpoint of another format"
    current = partial.get("current")
    if not isinstance(current, dict) or set(current) != set(CURRENT_KEYS) or not isinstance(partial["stops"], list) \
            or not isinstance(partial["resumes"], list) or not isinstance(partial["turns"], list):
        return "a checkpoint of another format"
    session = partial["session"]
    if not isinstance(session, dict) or not isinstance(session.get("pending"), dict) \
            or not isinstance(session.get("state"), dict) \
            or any(not isinstance(e, dict) or set(e) != set(STOP_KEYS) for e in partial["stops"]) \
            or any(not isinstance(t, dict) or set(t) != set(TURN_KEYS) for t in partial["turns"]):
        return "a checkpoint of another format"
    if partial["identity"] != {k: identity[k] for k in IDENTITY_KEYS}:
        return "the checkpoint belongs to another model or project"
    saved = partial["setting_key"]
    if not isinstance(saved, dict) or set(saved) != set(key):
        return "a checkpoint of another format"
    if saved.get("system_hash") != key["system_hash"] or saved.get("tools_hash") != key["tools_hash"]:
        return "the instructions or tools changed"
    changed = [k for k in key if saved.get(k) != key[k]]
    if changed:
        return f"the level setting changed: {', '.join(changed)}"
    return None


class CheckpointRefused(Exception):
    """The session refused the checkpoint (:class:`~flextool.assistant.session.CheckpointMismatch`): nothing was
    sent; the cell starts again from the start."""


class CellCheckpoint:
    """The checkpoint file of one cell (:data:`PARTIAL_DIR`).  :meth:`save` is the session's ``on_checkpoint``
    (through :func:`run_cell`): just before each request it writes the cell's finished turns, the current turn's
    notices and failed attempts, the session's snapshot and ``requests_by_day`` (the answered requests whose replies
    it holds, by UTC day); then, when the day's budget is used, it sets the cell's cancel event so that nothing is
    sent (``budget_stop``).  A write that fails is said once; the cell goes on without a checkpoint."""

    def __init__(self, out: Path, cid: str, *, identity: Mapping[str, Any], key: Mapping[str, Any],
                 git: Mapping[str, Any], say: Callable[[str], None], previous: dict | None = None,
                 over_budget: Callable[[dict[str, int]], bool] = lambda _days: False) -> None:
        self.path = partial_path(out, cid)
        self.cid = cid
        self.say = say
        self.over_budget = over_budget
        self.cancel = threading.Event()
        self.budget_stop = False
        self.write_failed = False
        now = iso(utc_now())
        self.data: dict[str, Any] = dict(previous) if previous is not None else {
            "format": PARTIAL_FORMAT, "cell": cid, "identity": {k: identity[k] for k in IDENTITY_KEYS},
            "setting_key": dict(key), "flextool_version": flextool_version(), "git": dict(git), "created": now,
            "updated": now, "turns": [], "current": None, "session": None, "requests_by_day": {}, "stopped": False,
            "stops": [], "resumes": []}
        #: The commit of the run that wrote the checkpoint before (a resume may run another one: recorded).
        self.previous_git = (previous or {}).get("git")
        self.data["git"] = dict(git)
        self.data["flextool_version"] = flextool_version()
        #: The requests held by the file as last written (each request counted once: those after it are the
        #: stop's, the file's are its own until it is removed).
        self.on_disk: dict[str, int] = held_requests(previous) if previous is not None else {}
        self.exists = previous is not None
        self.turn_usage = 0              # the current turn's usage entries the last checkpoint holds

    @property
    def held(self) -> int:
        return sum(held_requests(self.data).values())

    def save(self, turns: Sequence[Mapping[str, Any]], current: dict[str, Any], snapshot: dict) -> None:
        total = sum(len(t["usage"]) for t in turns) + len(snapshot["turn_usage"])
        days = dict(held_requests(self.data))
        new = total - sum(days.values())
        if new > 0:
            today = iso(utc_now())[:10]
            days[today] = days.get(today, 0) + new
        self.turn_usage = len(snapshot["turn_usage"])
        self.data.update(turns=[dict(t) for t in turns], current=current, session=snapshot, requests_by_day=days,
                         updated=iso(utc_now()), stopped=False)
        self._write()
        if self.over_budget(days):
            self.budget_stop = True
            self.cancel.set()                # nothing is sent: call_with_retries checks it before the send

    def _write(self) -> bool:
        if self.write_failed:
            return False
        from flextool.assistant import _files

        try:
            _files.write_text_atomic(self.path, _files.dumps(self.data), private=True)
        except OSError as exc:
            self.write_failed = True
            self.say(f"{self.cid}: the checkpoint could not be written ({exc}); the cell goes on without one.")
            return False
        self.exists = True
        self.on_disk = held_requests(self.data)
        return True

    def stop(self, entry: dict[str, Any], current: dict[str, Any] | None) -> bool:
        """Write the stop into the file (before its spent.json entry: a crash between the two counts the requests
        twice, never not at all).  ``False`` when there is no file to continue from."""
        if not self.exists or self.write_failed or self.data.get("session") is None:
            return False
        if current is not None:
            self.data["current"] = {**self.data["current"], **current}
        self.data["stops"] = [*self.data["stops"], entry]
        self.data["stopped"] = True
        return self._write()

    def resumed(self, point: dict[str, Any]) -> None:
        self.data["resumes"] = [*self.data["resumes"], point]

    def discard(self, spend: Callable[[dict[str, Any]], None]) -> None:
        """Remove the file; the requests it held are written to spent.json first (``discarded``, per day)."""
        for day, n in sorted(self.on_disk.items()):
            if n:
                spend({"ran_at": iso(utc_now()), "day": day, "requests": n, "cell": self.cid, "code": "discarded"})
        self.remove()

    def remove(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink()
        self.exists = False
        self.on_disk = {}


def _said_text(attempt: Mapping[str, Any]) -> str | None:
    """What the provider said in a failed attempt kept as JSON (:func:`failure_json`), as the console shows it."""
    from flextool.assistant.llm_errors import ProviderSaid

    if attempt.get("status") is None and not attempt.get("provider_code") and not attempt.get("message"):
        return None
    return ProviderSaid(attempt.get("status"), attempt.get("provider_code") or "", attempt.get("message") or "").text()


def run_cell(project: Path, qset: scoring.QuestionSet, question: scoring.Question, setting: Setting, *,
             backend: Any, preset: Any, ledger: Any, say: Callable[[str], None],
             checkpoint: CellCheckpoint | None = None) -> dict[str, Any]:
    """Ask *question* in a fresh conversation (S01: one conversation for its turns) at *setting*; the turns of
    the record, redacted, each with its failed request attempts (``failed_attempts``, :func:`failure_json`).
    Raises :class:`CellStopped` when a turn ends in an error that is not a graded stop (:data:`GRADED_STOPS`),
    :class:`DailyBudgetUsed` when the provider says the quota is used; either carries every failed attempt of
    this run of the cell and what the provider said.

    *checkpoint*: just before each request the cell is checkpointed there; a checkpoint it holds already (a cell
    that stopped, or a process that died) is continued: its finished turns are kept, the session restored and its
    pending request sent again (:class:`CheckpointRefused`, before anything is sent, when the session refuses
    it)."""
    from flextool.assistant.context import make_context
    from flextool.assistant.llm_adapter import Notice
    from flextool.assistant.llm_errors import message_for
    from flextool.assistant.session import SESSION_MESSAGES, TRUNCATION_SHARE, CheckpointMismatch, Session, TurnCaps
    from flextool.assistant.store import Conversation, new_id

    ctx = make_context(project, input_source=qset.project.get("input_source"), scenario=question.scenario,
                       front_end="terminal")
    ctx.caps = setting.budget.result_caps
    now = iso(utc_now())
    conversation = Conversation(id=new_id(), project_key="calibration", project_name=ctx.project_name,
                                project_path=ctx.project_path, created=now, updated=now, origin=backend.origin,
                                level=setting.level)
    failed: list[Any] = []              # every failed request attempt of this run of the cell, in order
    red = ctx.redact
    turns: list[dict[str, Any]] = []
    # The turn being asked: its number, notices, first failed attempt of this run, start, and what an earlier run
    # of it left (notices, failed attempts and seconds up to its checkpoint).
    live: dict[str, Any] = {"number": 0, "notices": [], "before": 0, "started": 0.0, "prior": None}

    def current() -> dict[str, Any]:
        prior = live["prior"] or {"notices": [], "failed_attempts": [], "seconds": 0.0}
        return {"turn": live["number"], "notices": [*prior["notices"], *(red(n) for n in live["notices"])],
                "failed_attempts": [*prior["failed_attempts"], *(failure_json(e, red) for e in failed[live["before"]:])],
                "seconds": round(prior["seconds"] + time.monotonic() - live["started"], 2)}

    def on_checkpoint(snapshot: dict) -> None:
        if checkpoint is not None:
            checkpoint.save(turns, current(), snapshot)

    session = Session(ctx, backend, preset, conversation=conversation, caps=TurnCaps.from_budget(setting.budget),
                      tier=setting.level, store=None, ledger=ledger, facts=setting.facts, route=setting.route,
                      budget=setting.budget, on_attempt_error=failed.append,
                      on_checkpoint=on_checkpoint if checkpoint is not None else None)
    resume = checkpoint.data if checkpoint is not None and checkpoint.data.get("session") is not None else None
    start = 1
    if resume is not None:
        try:
            session.restore(resume["session"])
        except CheckpointMismatch as exc:
            raise CheckpointRefused(str(exc)) from None
        turns[:] = [dict(t) for t in resume["turns"]]
        start = int(resume["current"]["turn"])
    cancel = checkpoint.cancel if checkpoint is not None else threading.Event()
    loaded = getattr(setting.facts, "loaded_context", None)
    local = setting.route.kind in ("loopback", "local_network") and setting.route.forwarded_to is None
    for number, text in enumerate(question.turns, start=1):
        if number < start:
            continue
        resuming = resume is not None and number == start
        notices: list[str] = []

        def on_event(event: Any, notices: list[str] = notices) -> None:
            if isinstance(event, Notice):
                notices.append(event.text)

        live.update(number=number, notices=notices, before=len(failed), started=time.monotonic(),
                    prior=dict(resume["current"]) if resuming else None)
        if resuming:
            result = session.resume_turn(on_event, cancel)
        else:
            result = session.run_turn(text, on_event, cancel)
        state = current()
        if result.error is not None and result.error.code not in GRADED_STOPS:
            sent = sum(len(t["usage"]) for t in turns) + len(result.usage)
            stop = DailyBudgetUsed if result.error.code == "quota" else CellStopped
            attempts = [failure_json(e, red) for e in failed]
            if not any(e is result.error for e in failed):
                attempts.append(failure_json(result.error, red))     # an error of no request attempt
            earlier = [a for a in (live["prior"] or {}).get("failed_attempts", [])]
            exc = stop(message_for(result.error, "terminal"), sent, result.error.code,
                       provider=failure_json(result.error, red) if result.error.provider_said else None,
                       attempts=attempts, said=said_lines(result.error, failed, red, earlier))
            exc.current = {"failed_attempts": state["failed_attempts"], "seconds": state["seconds"]}
            exc.lost_usage = [_usage_json(u) for u in result.usage[checkpoint.turn_usage:]] if checkpoint else []
            raise exc
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
            "calls": result.calls, "continuations": result.continuations,
            "stop": "end" if result.ok else (result.error.code if result.error else "error"),
            "notices": state["notices"],
            "error": ({"code": result.error.code, "message": red(message_for(result.error, "terminal")),
                       "provider": (failure_json(result.error, red) if result.error.provider_said else None)}
                      if result.error is not None else None),
            "failed_attempts": state["failed_attempts"],
            "limit_hit": any(m.role == "user" and m.text == SESSION_MESSAGES["limit"] for m in messages),
            "fit_compactions": [*(live["prior"] or {}).get("notices", []), *notices].count(SESSION_MESSAGES["fit"]),
            "truncated": truncated, "seconds": state["seconds"],
            "usage": [_usage_json(u) for u in result.usage], "requests": requests,
        })
        say(f"  {question.id} turn {number}: {result.rounds} round(s), {result.calls} call(s), "
            f"{'ok' if result.ok else turns[-1]['stop']}")
    return {"turns": turns}


def _record(question: scoring.Question, setting: Setting, repeat: int, *, preset: Any, identity: Mapping[str, Any],
            knobs_sent: Mapping[str, Any], free_tier: bool, git: Mapping[str, Any],
            turns: list[dict[str, Any]] | None, sending: Mapping[str, Any]) -> dict[str, Any]:
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
                  "knobs_relaxed": bool(identity.get("knobs_relaxed")), **sending},
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
    from flextool.assistant.pacing import PROCESS_PACER, pace_key
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
    if not (math.isfinite(args.pace) and args.pace >= 0):
        raise Refused("--pace is the least number of seconds between two requests: 0 or more.")
    if args.no_stream and preset.backend != "openai":
        raise Refused(f"--no-stream is for the OpenAI-compatible providers; {preset.label} answers streamed.")
    try:
        base_url = net.check_base_url(args.base_url or preset.base_url or "")
    except net.BaseUrlRefused as exc:
        raise Refused(str(exc)) from None
    out = Path(args.out)
    free = budget.free_tier_on(preset.id, args.model, FREE_TIER_FLAG[args.free_tier])
    identity = {"provider": preset.id, "model": args.model, "base_url": base_url, "free_tier": free,
                "ollama_context": contexts, "knobs_relaxed": bool(args.relax_data_collection),
                "project_fingerprint": marker["fingerprint"], "set_version": qset.set_version,
                "cpu_offload_allowed": bool(args.allow_cpu_offload)}
    if args.skip or args.unskip or args.skip_reason:
        mark_skipped(out, identity, args.skip or [], args.unskip or [], args.skip_reason, say)   # sends nothing
        return 0
    # The notice before run.json is written, so a refused first run can simply be run again with the flag.
    accepted: set = set()
    route = endpoint.route_for(preset, base_url, model=args.model)
    what = "the synthetic grounding project"
    accept_route(preset, base_url, route, accepted, accept=args.accept_data_notice,
                 again="--resume --accept-data-notice" if args.resume else "--accept-data-notice", what=what)
    _open_results(out, identity, args.resume)
    _plan_matrix(out, wanted, levels, args.repeat)
    # How the requests are sent: part of the folder's record (each run in run.json, each cell and stop with its
    # own), not of its identity, so a --resume may change it.
    sending = {"pace": args.pace, "stream": not args.no_stream}
    _log_run(out, sending)

    api_key, key_source = _key_or_refuse(preset, base_url, args.model,
                                         endpoint.route_for(preset, base_url, model=args.model))
    knobs = dict(DEFAULT_KNOBS)
    if args.relax_data_collection:
        knobs["openrouter.data_collection"] = "allow"
    knobs_sent = providers.request_fields(preset, knobs)

    # The resolved key is bound as defaults, once: a name rebound later in this function cannot reach a connection.
    def conn_for(model: str, route: Any, *, api_key: str = api_key, key_source: str = key_source) -> Connection:
        return Connection(base_url=base_url, model=model, api_key=api_key, key_source=key_source, knobs=knobs,
                          route=route, stream=not args.no_stream)

    daily = daily_budget(preset.id, args.model, args.max_requests_per_day)
    ledger = CalibrationLedger(Ledger())
    git = git_state()
    cells = read_cells(out)
    spent = read_spent(out)
    skipped = skipped_cells(read_json(out / RUN_FILE))
    failed = sorted(cid for cid, record in cells.items() if misclassified(record))
    if failed:
        say(f"{len(failed)} saved cell(s) are a provider failure saved as cut off (the answer broke off before "
            f"any of it came); they are asked again: {', '.join(failed)}.")

    def spend(entry: dict[str, Any]) -> None:
        spent.append(entry)
        write_json(out / SPENT_FILE, spent)

    partials = read_partials(out)
    for cid in sorted(partials):
        saved = cells.get(cid)
        if saved is not None and (finished(saved) or saved.get(NA_SCOPE_KEY) == "cell"):
            # A checkpoint left beside the record that finished its cell (a stop between the two writes): the
            # record holds its requests.
            with contextlib.suppress(OSError):
                partial_path(out, cid).unlink()
            partials.pop(cid)
    held = {cid: held_requests(p) for cid, p in partials.items()}

    def today_count(extra: Mapping[str, int] | None = None, without: str | None = None) -> int:
        day = iso(utc_now())[:10]
        days = [d for c, d in held.items() if c != without] + ([extra] if extra is not None else [])
        return requests_today(cells.values(), day, spent, days)

    def budget_used() -> bool:
        return daily is not None and today_count() >= daily

    # The cells kept after a tool call the service could not read (one answer's format problem, not the
    # service's state): the run goes on with the next cell and ends with exit code 1, naming them.
    unreadable: list[str] = []
    again = "--resume" + (" --accept-data-notice" if accepted else "")

    def ended(code: int) -> int:
        if not unreadable:
            return code
        say(f"{len(unreadable)} cell(s) kept after a tool call the service could not read: "
            f"{', '.join(unreadable)}; run again with {again} to continue them.")
        return 1

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
        again = "--resume" + (" --accept-data-notice" if accepted else "")
        for qid in wanted:
            question = qset.questions[qid]
            if question.automatic_level_only and (setting.choice is None or level != setting.choice.automatic):
                continue
            for repeat in range(1, args.repeat + 1):
                cid = cell_id(qid, level, repeat)
                if cid in skipped:
                    say(f"{cid}: {SKIPPED_STATUS} ({skipped[cid].get('reason')}); run --unskip {cid} to ask it")
                    continue
                saved = cells.get(cid)
                if saved is not None and saved.get(NA_SCOPE_KEY) == "cell":
                    say(f"{cid}: not applicable ({saved.get('not_applicable')}); run --unskip {cid} to ask it again")
                    continue
                if saved is not None and (finished(saved) or (saved.get("not_applicable")
                                                              and setting.not_applicable)):
                    continue                       # done, or still not applicable (kept with its recorded tags)
                has_partial = cid in partials
                partial = partials.pop(cid, None)
                if setting.not_applicable:
                    if has_partial:
                        CellCheckpoint(out, cid, identity=identity, key={}, git=git, say=say,
                                       previous=partial if isinstance(partial, dict) else {}).discard(spend)
                        held.pop(cid, None)
                    record = _record(question, setting, repeat, preset=preset, identity=identity,
                                     knobs_sent=knobs_sent, free_tier=free, git=git, turns=None, sending=sending)
                    write_json(cell_path(out, cid), record)
                    cells[cid] = record
                    say(f"{cid}: not applicable ({setting.not_applicable})")
                    continue
                if budget_used():
                    say(done)
                    return ended(0)
                setting_id = setting_key(setting)

                def checkpoint_for(previous: dict | None, cid: str = cid, setting_id: dict = setting_id) \
                        -> CellCheckpoint:
                    return CellCheckpoint(out, cid, identity=identity, key=setting_id, git=git, say=say,
                                          previous=previous,
                                          over_budget=lambda days, cid=cid: daily is not None
                                          and today_count(days, without=cid) >= daily)

                cp = checkpoint_for(None)
                if has_partial:
                    problem = partial_problem(partial, identity, setting_id)
                    if problem is None and partial.get("session") is None:
                        problem = "a checkpoint of another format"
                    if problem is not None:
                        say(f"{cid}: started again from the start ({problem})")
                        checkpoint_for(partial if isinstance(partial, dict) else {}).discard(spend)
                        held.pop(cid, None)
                    else:
                        cp = checkpoint_for(partial)
                        _continue_partial(cp, spend, sending, say)
                if saved is not None and misclassified(saved):
                    say(f"{cid}: asked again (a provider failure saved as cut off)")
                # A saved cell asked again (an error stop of an older folder, a provider failure saved as cut
                # off, or now applicable) stays on disk until its new record replaces it: a stop on the way
                # leaves the folder with the old record, never without the cell.
                backend = backend_factory(preset, conn_for(setting.model, setting.route))
                # --pace: the least gap between request starts, on top of the pacing's own (the pacer the session
                # uses: the backend's when it paces itself, else the process's).
                pacer = backend.pacer if getattr(type(backend), "paces", False) else PROCESS_PACER
                pacer.set_floor(pace_key(backend.origin), args.pace)
                try:
                    try:
                        turns = run_cell(project, qset, question, setting, backend=backend, preset=preset,
                                         ledger=ledger, say=say, checkpoint=cp)["turns"]
                    except CheckpointRefused as refused:
                        say(f"{cid}: started again from the start ({refused})")
                        cp.discard(spend)
                        held.pop(cid, None)
                        cp = checkpoint_for(None)
                        turns = run_cell(project, qset, question, setting, backend=backend, preset=preset,
                                         ledger=ledger, say=say, checkpoint=cp)["turns"]
                except CellStopped as exc:
                    outcome = _stopped(exc, cp, cid, spend=spend, sending=sending, say=say)
                    held[cid] = cp.on_disk if cp.exists else {}
                    if not cp.exists:
                        held.pop(cid, None)
                    if outcome == "not_applicable":
                        record = _record(question, setting, repeat, preset=preset, identity=identity,
                                         knobs_sent=knobs_sent, free_tier=free, git=git, turns=cp.data["turns"],
                                         sending=sending)
                        record.update(_na_fields(cp, exc))
                        if saved is not None and int(saved.get("requests_sent") or 0):
                            spend(_replaced(saved, cid))
                        write_json(cell_path(out, cid), record)
                        cells[cid] = record
                        cp.remove()
                        held.pop(cid, None)
                        say(f"{cid}: not applicable ({QUOTA_PER_REQUEST}): the provider refused the same request "
                            f"{QUOTA_REPEATS} times for its input tokens per minute; run --unskip {cid} to ask it "
                            "again")
                        continue
                    if outcome in ("budget", "quota") or cp.budget_stop or isinstance(exc, DailyBudgetUsed):
                        say(done)
                        return ended(0)
                    if outcome == "kept":
                        say(f"{cid}: {exc.kept_text}; run again with {again} to continue it")
                        if exc.code == "tool_call_unreadable":
                            unreadable.append(cid)       # the service's state is fine: the next cell is asked
                            continue
                    else:
                        say(f"{cid}: nothing was saved for this cell. Run again with {again}.")
                    return ended(1)
                finally:
                    backend.close()
                record = _record(question, setting, repeat, preset=preset, identity=identity,
                                 knobs_sent=knobs_sent, free_tier=free, git=git, turns=turns, sending=sending)
                record.update(_resume_fields(cp, turns))
                record["setting_key"] = setting_id              # the summary warns of a level's mixed settings
                if saved is not None and int(saved.get("requests_sent") or 0):
                    # The replaced record's requests stay counted for their day (written first: a crash
                    # between the two counts them twice, never not at all).
                    spend(_replaced(saved, cid))
                write_json(cell_path(out, cid), record)              # temp + os.replace: the old one goes
                cells[cid] = record
                cp.remove()                                          # record first: a crash between leaves a
                held.pop(cid, None)                                  # stale checkpoint, removed by the next run
                say(f"{cid}: saved")
                if budget_used():
                    say(done)
                    return ended(0)
    if unreadable:
        say(f"The other cells of {args.model} are in {out}"
            + (f" ({len(skipped)} skipped by the owner)." if skipped else "."))
        return ended(0)
    say(f"All cells of {args.model} are in {out}"
        + (f" ({len(skipped)} skipped by the owner)." if skipped else "."))
    return 0


def _replaced(saved: Mapping[str, Any], cid: str) -> dict[str, Any]:
    """The spent.json entry of a record replaced by a new one: its requests, per day when it has them."""
    entry: dict[str, Any] = {"ran_at": saved.get("ran_at"), "requests": int(saved["requests_sent"]), "cell": cid,
                             "code": "replaced"}
    days = saved.get("requests_by_day")
    if isinstance(days, dict) and days:
        entry["by_day"] = dict(days)
    return entry


def _continue_partial(cp: CellCheckpoint, spend: Callable[[dict[str, Any]], None], sending: Mapping[str, Any],
                      say: Callable[[str], None]) -> None:
    """Continue a checkpoint: a process that died with it (no stop written: Ctrl-C, a crash) may have sent its
    pending request, which is counted once on that day (conservative) and its stop written as ``crashed``; then
    the resume point is recorded."""
    data = cp.data
    pending = data["session"]["pending"]
    state = data["session"]["state"]
    if not data["stopped"]:
        spend({"ran_at": data["updated"], "requests": 1, "cell": cp.cid, "code": "crashed",
               "request_sha256": pending.get("request_sha256"), "est_input": pending.get("est_input")})
        data["stops"] = [*data["stops"], {
            "code": "crashed", "turn": data["current"]["turn"], "round": state.get("rounds"), "request": cp.held + 1,
            "request_sha256": pending.get("request_sha256"), "est_input": pending.get("est_input"), "provider": None,
            "lost_usage": [], "stopped_at": data["updated"], "pace": None, "stream": None}]
        data["stopped"] = True
    last = data["stops"][-1]
    cp.resumed({"turn": last["turn"], "round": last["round"], "request": last["request"], "code": last["code"],
                "provider": last.get("provider"), "lost_usage": last.get("lost_usage") or [],
                "stopped_at": last["stopped_at"], "resumed_at": iso(utc_now()), **dict(sending), "git": dict(cp.data["git"]),
                "stopped_git": cp.previous_git, "request_sha256": pending.get("request_sha256"),
                "body_sha256": pending.get("body_sha256")})
    say(f"{cp.cid}: continued from turn {last['turn']}, request {last['request']} ({last['code']})")


def _stopped(exc: CellStopped, cp: CellCheckpoint, cid: str, *, spend: Callable[[dict[str, Any]], None],
             sending: Mapping[str, Any], say: Callable[[str], None]) -> str:
    """A cell that stopped: for a temporary stop (:data:`TEMPORARY_STOPS`, the day's budget) with a checkpoint, the
    stop written into it first; then its spent.json entry: the requests the checkpoint file does not hold (those
    after the last checkpoint; all but the file's own when the checkpoint is thrown away, whose own are written as
    ``discarded``).  Returns ``"kept"``, ``"budget"`` / ``"quota"`` (kept, the run ends with exit 0),
    ``"not_applicable"`` (the same request refused :data:`QUOTA_REPEATS` times for its input tokens per minute: the
    caller writes the record) or ``"discarded"`` (nothing kept)."""
    code = "budget" if cp.budget_stop else exc.code
    snapshot = cp.data.get("session") or {}
    pending = snapshot.get("pending") or {}
    kept = False
    if code in TEMPORARY_STOPS or code == "budget":
        current = cp.data.get("current") or {}
        stop = {"code": code, "turn": current.get("turn"), "round": (snapshot.get("state") or {}).get("rounds"),
                "request": cp.held + 1, "request_sha256": pending.get("request_sha256"),
                "est_input": pending.get("est_input"), "provider": exc.provider, "lost_usage": list(exc.lost_usage),
                "stopped_at": iso(utc_now()), **dict(sending)}
        kept = cp.stop(stop, None if code == "budget" else exc.current)
    after = max(0, exc.requests_sent - (cp.held if kept else sum(cp.on_disk.values())))
    entry = {"ran_at": iso(utc_now()), "requests": after, "cell": cid, "code": code, "provider": exc.provider,
             "attempts": exc.attempts, **dict(sending)}
    if pending:
        entry.update(request_sha256=pending.get("request_sha256"), est_input=pending.get("est_input"))
    if code != "budget" or after:
        spend(entry)
    if code != "budget":
        say(str(exc))
        for line in exc.said:
            say(line)
    if not kept:
        if cp.exists:
            cp.discard(spend)
        return "discarded"
    stops = cp.data["stops"]
    last = stops[-QUOTA_REPEATS:]
    if len(last) == QUOTA_REPEATS and all(s["code"] == "tpm_input" for s in last) \
            and len({s["request_sha256"] for s in last}) == 1:
        return "not_applicable"
    same = stops[-SAME_REQUEST_REPEATS:]
    if len(same) == SAME_REQUEST_REPEATS and len({s["request_sha256"] for s in same}) == 1:
        say(f"{cid}: the same request failed {SAME_REQUEST_REPEATS} times; --skip it or wait")
    exc.kept_text = f"kept up to turn {stop['turn']}, request {stop['request']}"
    return "budget" if code == "budget" else "quota" if isinstance(exc, DailyBudgetUsed) else "kept"


def _na_fields(cp: CellCheckpoint, exc: CellStopped) -> dict[str, Any]:
    """The fields of a record not applicable for the provider's quota per request: the quota the provider named,
    the stops, and every request the cell's checkpoint held (the record holds them now)."""
    days = held_requests(cp.data)
    return {"not_applicable": QUOTA_PER_REQUEST, NA_SCOPE_KEY: "cell",
            "quota": list((exc.provider or {}).get("quota") or []), "provider": exc.provider,
            "stops": list(cp.data["stops"]), "requests_by_day": days, "requests_sent": sum(days.values()),
            "resume": {"count": len(cp.data["resumes"]), "points": list(cp.data["resumes"])}}


def _resume_fields(cp: CellCheckpoint, turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A finished record's ``resume`` (how often and where the cell was continued) and ``requests_by_day`` (its
    checkpoint's, and the requests after the last checkpoint on this day)."""
    days = dict(held_requests(cp.data))
    total = sum(len(t["usage"]) for t in turns)
    new = total - sum(days.values())
    if new > 0:
        today = iso(utc_now())[:10]
        days[today] = days.get(today, 0) + new
    return {"resume": {"count": len(cp.data["resumes"]), "points": list(cp.data["resumes"])},
            "requests_by_day": days}


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
#: The file route (``grade-ai --export-requests`` / ``--import-replies``): the same model graded through Claude
#: Code subagents, which read the exported requests and write the replies (no API request, no usage).
FILE_GRADER = "claude-opus-5-5 via Claude Code (file route)"
FILE_ROUTE = "file"
API_ROUTE = "api"
EXPORT_INDEX = "index.json"
EXPORT_FORMAT = 1
EXPORT_INSTRUCTIONS = (
    "Each <n>.request.txt is one grading request: act as the grader its SYSTEM section describes, grade the "
    "material of its USER section and write your reply, the JSON object alone, to <n>.reply.json in this folder. "
    "A <n>.retry.request.txt is the whole conversation of a retry (your first reply was not the asked JSON): "
    "write the reply to its last USER section to <n>.retry.reply.json.")
NO_GRADING_FILE_ROUTE = ("No grading in {folder}: the file route continues a grading that grade-ai started (its "
                         "grading.json fixes the blind order).")

CellKey = tuple[str, str]


def _cell_key(folder: Path, cid: str) -> CellKey:
    return (str(Path(folder).resolve()), cid)


def _stamps(records: Iterable[Mapping[str, Any]]) -> set[str]:
    """The times of the records the leak scan looks for: when each ran or was graded, and when it stopped and was
    continued (``resume.points``)."""
    out: set[str] = set()
    for r in records:
        out |= {str(r.get(k)) for k in ("ran_at", "graded_at") if r.get(k)}
        out |= {str(p.get(k)) for p in (r.get("resume") or {}).get("points") or [] for k in ("stopped_at", "resumed_at")
                if p.get(k)}
    return out


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
    def route(self) -> str | None:
        """How the AI grade came: ``api`` (an Anthropic request; grades without a route are such) or ``file``
        (a Claude Code subagent through ``--export-requests`` / ``--import-replies``); ``None`` without one."""
        ai = (self.state or {}).get("ai")
        return str(ai.get("route") or API_ROUTE) if ai else None

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
class SkippedCell:
    """A cell the owner skipped: its results folder and run.json, its id, its run.json entry and its question."""

    folder: Path
    run: dict
    cid: str
    entry: dict
    question: scoring.Question


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
    #: The cells the owner skipped (``run --skip``): left out of every rate and agreement figure.
    skipped: list[SkippedCell] = dataclasses.field(default_factory=list)


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
    raw = _load_results([Path(p) for p in wanted])
    # A cell the owner skipped is no gap and no cell of the grading (a record it has, unfinished, is not read).
    loaded = [(f, run, {cid: r for cid, r in cells.items() if cid not in skipped_cells(run)}) for f, run, cells in raw]
    records = {_cell_key(f, cid): r for f, _run, cells in loaded for cid, r in cells.items()}
    runs = {_cell_key(f, "")[0]: run for f, run, _cells in loaded}
    if not records:
        raise Refused(f"No cells in {', '.join(wanted)}.")
    qset = scoring.load_set(Path(args.set))
    _refuse_incomplete(loaded, qset)
    skipped = [SkippedCell(Path(f), run, cid, dict(entry), qset.questions[parse_cell_id(cid)[0]])
               for f, run, _cells in loaded for cid, entry in sorted(skipped_cells(run).items())]
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
    return Grading(folder, state, cells, not_applicable, names, scan_names, _stamps(records.values()), key, old,
                   skipped)


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
        skipped = skipped_cells(run)                  # the owner's skips are no gap
        unknown = [u for u in unknown if u.split(" ", 1)[0] not in skipped]
        missing = sorted(expected - set(cells) - set(skipped))
        unfinished = sorted(cid for cid, r in cells.items()
                            if cid not in skipped and not r.get("not_applicable") and not finished(r))
        # A cell with only a checkpoint is named by it (a checkpoint beside a finished record does not count).
        partials = {cid: p for cid, p in read_partials(folder).items()
                    if cid not in skipped and not (cid in cells and (finished(cells[cid])
                                                                     or cells[cid].get("not_applicable")))}
        named = ([f"{cid} ({unfinished_text(partials[cid])})" if cid in partials else f"{cid} (missing)"
                  for cid in missing] + unknown
                 + [f"{cid} ({unfinished_text(partials[cid])})" if cid in partials else f"{cid} (not finished)"
                    for cid in unfinished])
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
        last, parsed, problems = read_reply(reply.message.text if reply.message is not None else "", allowed, passed)
        if parsed is not None:
            return graded_result(parsed, attempt + 1, usage)
        messages += [Message("assistant", last or "(no text)"), Message("user", scoring.retry_text(problems))]
    return unparsed_result(last, usage, problems)


def read_reply(text: str, allowed: Mapping[str, Sequence[str]], passed: Iterable[str] = ()) \
        -> tuple[str, dict[str, Any] | None, list[str]]:
    """``(text, parsed, problems)`` of a grader reply, whichever route it came by: its reasoning blocks removed,
    then :func:`scoring.check_reply` (the strict check)."""
    from flextool.assistant.thoughts import strip_thoughts

    last = strip_thoughts(text)
    parsed, problems = scoring.check_reply(last, allowed, passed)
    return last, parsed, problems


def graded_result(parsed: Mapping[str, Any], attempts: int, usage: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {**parsed, "attempts": attempts, "usage": list(usage)}


def unparsed_result(last: str, usage: Sequence[Mapping[str, Any]], problems: Sequence[str]) -> dict[str, Any]:
    """A reply invalid after the retry: kept (cut) for the owner, who grades the cell from scratch."""
    return {"unparsed": last[:2_000], "attempts": 2, "usage": list(usage), "problems": list(problems)}


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


def _record_ai(cell: GradingCell, result: Mapping[str, Any], *, pending: bool) -> None:
    """Saves the AI grade *result* of *cell* (``cells/<n>.json``, atomically): a review written meanwhile is kept,
    an earlier grade moves into the history (``--changed``), a continued pending retry adds no history entry."""
    base = _current_state(cell) if pending else _new_state(cell, _fresh_state(cell))
    base.pop("pending", None)
    cell.state = {**base, "ai": dict(result), "graded_at": iso(utc_now())}
    write_json(cell.path, cell.state)


def _to_grade(grading: Grading, changed: bool) -> tuple[list[GradingCell], list[GradingCell]]:
    """``(todo, send)``: the cells grade-ai grades now (not graded, or changed with *changed*) and of them those
    with an answer (one request each).  Refused: changed cells without *changed*, and any leak the blinding
    self-test finds in a request as it would be sent (every hit named, nothing sent)."""
    stale = [c for c in grading.cells if c.stale]
    if stale and not changed:
        raise Refused(CHANGED_MATERIAL.format(token=stale[0].token) + f" ({len(stale)} cell(s) changed.)")
    todo = [c for c in grading.cells if c.state is None or c.stale or not c.graded]
    send = [c for c in todo if not c.empty]
    # The blinding self-test of every request this run would send, all of it as sent, before anything is sent.
    hits = [(c, found) for c in send for found in [_scan(grading, c, _request_text(c.material, c.allowed))] if found]
    if hits:
        raise Refused("\n".join(f"Cell {c.token}: the grader material reveals {', '.join(found)}; nothing was sent."
                                for c, found in hits))
    return todo, send


def cmd_grade_ai(args: argparse.Namespace, *, backend_factory: BackendFactory | None = None,
                 say: Callable[[str], None] = print) -> int:
    try:
        if args.export_requests and args.import_replies:
            raise Refused("Give --export-requests or --import-replies, not both.")
        if args.export_requests or args.import_replies:
            if _read_grading(Path(args.dir)) is None:
                raise Refused(NO_GRADING_FILE_ROUTE.format(folder=args.dir))
            if args.export_requests:                # it only reads the grading folder: no lock
                return _export_requests(args, Path(args.export_requests), say)
            with _Lock(Path(args.dir)):
                return _import_replies(args, Path(args.import_replies), say)
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
    for line in analysis.mixed_settings(c.record for c in grading.cells):
        say(line)
    if grading.old_grades:
        say(f"{grading.old_grades} cells carry grades from the earlier grading; they are ignored.")
    todo, send = _to_grade(grading, args.changed)
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
            _record_ai(c, result, pending=bool(pending))
            say(f"Cell {c.token}: graded ({done} of {len(send)} to send"
                + ("; the reply could not be read" if "unparsed" in result else "") + ").")
    finally:
        backend.close()
    say(f"Every cell is graded ({total}; {sum(c.empty for c in grading.cells)} without an answer are failed in "
        "code). Next: review --dir.")
    return 0


# --- grade-ai, the file route: Claude Code subagents grade instead of the API ---------------------------------
def request_file_text(material: str, retry: tuple[str, Sequence[str]] | None = None) -> str:
    """A grading request as a plain text an LLM agent reads: ``SYSTEM:`` then ``USER:``, exactly the system part
    and the user part the API request sends.  *retry* ``(first reply, problems)`` adds the "JSON only" retry as
    the API sends it: the first reply (``ASSISTANT:``) and the retry text (``USER:``)."""
    parts = ["SYSTEM:", scoring.GRADER_PROMPT_V2, "", "USER:", material]
    if retry is not None:
        first, problems = retry
        parts += ["", "ASSISTANT:", first or "(no text)", "", "USER:", scoring.retry_text(problems)]
    return "\n".join(parts) + "\n"


def _write_text(path: Path, text: str) -> None:
    """*text* at *path*, through a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _set_aside(path: Path) -> bool:
    """A reply that answered other material is kept as ``<name>.stale``, never read again."""
    if not path.is_file():
        return False
    os.replace(path, path.with_name(path.name + ".stale"))
    return True


def _export_paths(out: Path, cell: GradingCell) -> dict[str, Path]:
    n = cell.number
    return {"request": out / f"{n}.request.txt", "reply": out / f"{n}.reply.json",
            "retry_request": out / f"{n}.retry.request.txt", "retry_reply": out / f"{n}.retry.reply.json"}


def _token_range(tokens: Sequence[str]) -> str:
    numbers = sorted(int(t.lstrip("#")) for t in tokens)
    return f"#{numbers[0]}–#{numbers[-1]}" if numbers else "none"


def _export_requests(args: argparse.Namespace, out: Path, say: Callable[[str], None]) -> int:
    """``grade-ai --export-requests OUT``: the request of every cell grade-ai would send now, as
    ``OUT/<n>.request.txt``, and ``OUT/index.json`` (the tokens and their request hashes; no model, level or
    folder name).  The same selection, blinding and leak scan as the API route; nothing is written to the grading
    folder.  A reply in OUT to a cell whose request changed since the last export is set aside."""
    grading = load_grading(args, create=False)
    todo, send = _to_grade(grading, args.changed)
    old = read_json(out / EXPORT_INDEX) if (out / EXPORT_INDEX).is_file() else {}
    before = {e["token"]: e["request_sha256"] for e in old.get("cells") or []}
    entries, aside = [], 0
    for c in send:
        paths = _export_paths(out, c)
        if before.get(c.token) not in (None, c.request_sha256):
            aside += sum(_set_aside(paths[k]) for k in ("reply", "retry_request", "retry_reply"))
        _write_text(paths["request"], request_file_text(c.material))
        entries.append({"token": c.token, "request": paths["request"].name, "reply": paths["reply"].name,
                        "request_sha256": c.request_sha256})
    write_json(out / EXPORT_INDEX, {"format": EXPORT_FORMAT, "instructions": EXPORT_INSTRUCTIONS,
                                    "cells": entries})
    empty = sum(1 for c in todo if c.empty)
    say(f"Exported {len(entries)} request(s) to {out} ({_token_range([e['token'] for e in entries])})"
        + (f"; {empty} cell(s) without an answer need none (import fails them in code)" if empty else "")
        + (f"; {aside} reply file(s) to an earlier request set aside as .stale" if aside else "")
        + ". When the replies are written: grade-ai ... --import-replies " + str(out))
    return 0


def _import_replies(args: argparse.Namespace, out: Path, say: Callable[[str], None]) -> int:
    """``grade-ai --import-replies OUT``: each exported cell's ``OUT/<n>.reply.json`` through the same strict
    check as an API reply, recorded as the API route records it (grader :data:`FILE_GRADER`, route ``file``, no
    usage).  A cell whose request changed since the export is refused (export again).  A reply that fails the
    check is not recorded: ``OUT/<n>.retry.request.txt`` gets the API's retry, and a later import reads
    ``OUT/<n>.retry.reply.json`` (invalid again: recorded unparsed, as the API route does).  A graded cell is never
    overwritten unless it changed and ``--changed`` is given; running it again continues."""
    if not (out / EXPORT_INDEX).is_file():
        raise Refused(f"No {EXPORT_INDEX} in {out}: run grade-ai --export-requests {out} first.")
    index = read_json(out / EXPORT_INDEX)
    if not isinstance(index, dict) or index.get("format") != EXPORT_FORMAT:
        raise Refused(f"{out / EXPORT_INDEX} is not an export of grade-ai --export-requests; export again.")
    grading = load_grading(args, create=False)
    todo, _send = _to_grade(grading, args.changed)
    waiting = {c.token for c in todo}
    for c in todo:
        if c.empty:                        # failed in code, as the API route records it
            c.state = {**_new_state(c, _fresh_state(c)), "graded_at": iso(utc_now())}
            write_json(c.path, c.state)
    by_token = {c.token: c for c in grading.cells}
    recorded, retry, missing, refused, done = [], [], [], [], []
    for entry in index.get("cells") or []:
        token = str(entry.get("token"))
        c = by_token.get(token)
        if c is None:
            refused.append(token)
            say(f"Cell {token}: not a cell of this grading; export again.")
            continue
        if token not in waiting:
            done.append(token)             # graded already (not overwritten)
            continue
        if entry.get("request_sha256") != c.request_sha256:
            refused.append(token)
            say(f"Cell {token}: its request changed since the export (its answer, the answer key or the rubric); "
                f"not recorded. Export again: grade-ai ... --export-requests {out}")
            continue
        paths = _export_paths(out, c)
        if not paths["reply"].is_file():
            missing.append(token)
            continue
        passed = [i.id for i in c.items if i.auto]
        first, parsed, problems = read_reply(paths["reply"].read_text(encoding="utf-8"), c.allowed, passed)
        result: dict[str, Any] | None = None
        if parsed is not None:
            result = graded_result(parsed, 1, [])
        else:
            text = request_file_text(c.material, (first, problems))
            found = _scan(grading, c, text)
            if found:
                refused.append(token)
                say(f"Cell {token}: the retry request reveals {', '.join(found)}; not written.")
                continue
            current = paths["retry_request"].read_text(encoding="utf-8") if paths["retry_request"].is_file() else None
            if current == text and paths["retry_reply"].is_file():
                last, parsed, problems = read_reply(paths["retry_reply"].read_text(encoding="utf-8"), c.allowed,
                                                    passed)
                result = graded_result(parsed, 2, []) if parsed is not None else unparsed_result(last, [], problems)
            else:
                if current != text:        # a retry reply to an earlier first reply answers another retry
                    _set_aside(paths["retry_reply"])
                    _write_text(paths["retry_request"], text)
                retry.append(token)
                continue
        pending = (c.state or {}).get("pending") if not c.stale else None
        _record_ai(c, {**result, "grader": FILE_GRADER, "route": FILE_ROUTE}, pending=bool(pending))
        recorded.append(token)
    left = sum(1 for c in grading.cells if not c.graded)
    say(f"Recorded {len(recorded)}; needing a retry {len(retry)}"
        + (f" (write <n>.retry.reply.json for {', '.join(retry)})" if retry else "")
        + f"; missing {len(missing)}" + (f" ({_token_range(missing)})" if missing else "")
        + (f"; refused {len(refused)}" if refused else "") + (f"; graded already {len(done)}" if done else "")
        + f". {left} of {len(grading.cells)} cells still wait for a grade.")
    return 2 if refused else 0


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


# --- review through markdown files: the owner fills a form in an editor -----------------------------------------
MD_MARK = re.compile(r"^<!-- flextool-review (first|final|helpful) (#\d+) ([0-9a-f]{16}) -->$", re.MULTILINE)
MD_LINE = re.compile(r"^- `([^`]+)`.*\[([^\]\n]*)\]:[ \t]*(.*?)[ \t]*$")
#: The form line of the first file format (before the item texts moved next to the letters).
MD_LINE_V1 = re.compile(r"^- `([^`]+)`([^:\n]*):[ \t]*(.*?)[ \t]*$")
#: The material sections the owner's files leave out: the tool calls and the number excerpts (the numbers are
#: Opus's to check) and the items (they stand next to the letters).
MD_DROP = ("Tool calls and results", "Excerpts", "Items to grade")
MD_FIRST_HEAD = """<!-- flextool-review file: your own verdicts. Opus's are not in this file. -->
# Review: your own verdicts

Each cell ends with **Your verdicts**: each item to grade, then the letters it allows in brackets; write one
after the colon: `r` right, `m` missing, `w` wrong, and the understandability `1`, `2` or `3`. What the letters
mean for each kind of item is in `{key}`. The numbers (number facts and numbers without a tool source) are
Opus's verdicts and are not asked here.

Leave a line empty to answer it later: import the file whenever you like
(`review --dir GRADING --import-md FILE`); the verdicts given are kept and an empty line is asked again by the next
`--export-md`. A verdict once imported stays your first verdict (agreement uses it); edit only the letters.
After the import, the cells where Opus judged otherwise go to a second file, `{reveal}`, where you see Opus's
verdicts and can change yours.
"""
MD_FINAL_HEAD = """<!-- flextool-review file: Opus's verdicts after yours. -->
# Review: Opus's verdicts after yours

Your own verdicts are recorded. For each cell, Opus's verdicts are shown next to yours; under **Your final
verdicts** each line where you differ already holds your letter. Leave it to keep yours, or write another letter
to change it, then import this file (`review --dir GRADING --import-md FILE`).
"""


def _md_letter(verdict: Any) -> str:
    return next((k for k, v in LETTERS.items() if v == verdict), "") if isinstance(verdict, str) else \
        str(verdict) if verdict is not None else ""


def _deferred(cell: GradingCell) -> set[str]:
    """The judged items the owner leaves to Opus in the review files: the numbers (a number fact the check did not
    find, a number without a tool source).  None when Opus has no readable reply."""
    if cell.ai is None:
        return set()
    numbers = {f"fact:{f.id}" for f in cell.entry.facts if f.kind == "number"}
    return {i.id for i in cell.judged if i.check == "tool_sourced" or (i.check == "facts" and i.id in numbers)}


def _owner_items(cell: GradingCell) -> list[scoring.GradeItem]:
    deferred = _deferred(cell)
    return [i for i in cell.judged if i.id not in deferred]


def _with_deferred(cell: GradingCell, items: Mapping[str, Any]) -> dict[str, Any]:
    """The owner's verdicts with Opus's for the items left to Opus: a whole set of the cell's judged items."""
    return {**{i: cell.ai["items"][i]["verdict"] for i in _deferred(cell)}, **items}


def _split_what(what: str) -> tuple[str, str]:
    """``(label, meanings)`` of an item's text: what is checked, and what its verdicts mean (empty for a
    statement, whose verdicts mean what they always do)."""
    m = re.search(r"[:.] (?=(right|missing|wrong) = )", what)
    return (what[:m.start()], what[m.end():]) if m else (what, "")


MD_KEY_HEAD = """# Review key

What the letters mean. An item without its own entry below (a statement of the answer key, a name) is judged
as follows:

- `r` right: the answer gets this item right;
- `m` missing: the answer does not give it, and says nothing false about it;
- `w` wrong: the answer says something false about it, or gives another value.

A missing unit next to a right value, or a true claim not marked as an assumption, is only a form slip: it
does not make the cell incomplete. A wrong unit or a false unmarked claim fails the cell.

**Understandability** — {understandability}
"""


def _md_key(grading: Grading) -> str:
    """The key file: the general meaning of the letters, then each kind of item the owner judges that has its
    own (from the items' texts, the same as the graders read)."""
    seen: dict[str, str] = {}
    for cell in grading.cells:
        for i in _owner_items(cell):
            label, meanings = _split_what(i.what)
            if meanings:
                title = "matched a forbidden claim" if i.check == "forbidden" else label
                seen.setdefault(title, meanings)
    parts = [MD_KEY_HEAD.format(understandability=scoring.UNDERSTANDABILITY_TEXT)]
    for title, meanings in seen.items():
        lines = re.split(r"; (?=(?:right|missing|wrong) = )", meanings.rstrip("."))
        parts.append(f"## {title}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n")
    return "\n".join(parts)


def _md_key_path(path: Path) -> Path:
    return path.with_name(f"{path.name.removesuffix('.md')}.key.md")


def _md_item_line(i: scoring.GradeItem, tail: str, letter: Any) -> str:
    label = _split_what(i.what)[0]
    return (f"- `{i.id}`{' (critical)' if i.critical else ''}: {label} [{tail}]: "
            f"{_md_letter(letter)}").rstrip()


def _md_cell(grading: Grading, cell: GradingCell, kind: str) -> str:
    """The markdown of one cell: its heading, its mark (kind, token, request hash) and either the owner's form
    (``first``) or Opus's verdicts and the final form (``final``), with the material."""
    tag = scoring.data_tag(grading.state["seed"], cell.token)
    parts = [f"# Cell {cell.token}", f"<!-- flextool-review {kind} {cell.token} {cell.request_sha256[:16]} -->", ""]
    material = scoring.owner_markdown(cell.material, tag, MD_DROP)
    if kind == "first":
        given = cell.state.get("owner_first") or {}
        parts += [material, "", "### Your verdicts", ""]
        for i in _owner_items(cell):
            parts.append(_md_item_line(i, " / ".join(_letters(i.allowed)), (given.get("items") or {}).get(i.id)))
        parts.append(f"- `understandability` [1 / 2 / 3]: {_md_letter(given.get('understandability'))}".rstrip())
        return "\n".join(parts)
    first, ai = cell.state["owner_first"], cell.ai
    parts += ["### Opus's grading (shown after yours)", "",
              f"The cell ({ai['confidence']}): {_blinded_reason(grading, cell, ai['reason'])}", "",
              "| item | yours | Opus | Opus's reason |", "|---|---|---|---|"]
    form = []
    for i in _owner_items(cell):
        mine, theirs = first["items"][i.id], ai["items"][i.id]
        differs = mine != theirs["verdict"]
        reason = _blinded_reason(grading, cell, theirs["reason"]).replace("|", "\\|").replace("\n", " ")
        parts.append(f"| `{i.id}` | {mine} | {'**' if differs else ''}{theirs['verdict']}{'**' if differs else ''} "
                     f"({theirs['confidence']}) | {reason} |")
        if differs:
            form.append(_md_item_line(i, f"{' / '.join(_letters(i.allowed))}; yours {mine}, Opus "
                                         f"{theirs['verdict']}", mine))
    u_differs = first["understandability"] != ai["understandability"]
    parts.append(f"| understandability | {first['understandability']} | {'**' if u_differs else ''}"
                 f"{ai['understandability']}{'**' if u_differs else ''} | |")
    if u_differs:
        form.append(f"- `understandability` [1 / 2 / 3; yours {first['understandability']}, Opus "
                    f"{ai['understandability']}]: {first['understandability']}")
    parts += ["", "### Your final verdicts", "", *form, "", "### The cell", "", material]
    return "\n".join(parts)


def _md_disagrees(cell: GradingCell, first: Mapping[str, Any]) -> bool:
    ai = cell.ai
    return ai is not None and (any(first["items"][i.id] != ai["items"][i.id]["verdict"] for i in _owner_items(cell))
                               or first["understandability"] != ai["understandability"])


def _md_first_done(cell: GradingCell) -> bool:
    first = cell.state.get("owner_first") or {}
    return all(i.id in (first.get("items") or {}) for i in _owner_items(cell)) \
        and first.get("understandability") is not None


def _md_reveal_path(path: Path) -> Path:
    return path.with_name(f"{path.name.removesuffix('.md')}.final.md")


def _md_recorded(cell: GradingCell, kind: str, iid: str) -> Any:
    """What a review file's line *iid* of *kind* stands for in the cell's state now: the first verdict, or the
    final one (the first while the cell is not finished)."""
    state = cell.state or {}
    source = state.get("owner") if kind == "final" and _owner_done(state) else state.get("owner_first") or {}
    return source.get("understandability") if iid == "understandability" else (source.get("items") or {}).get(iid)


def _md_unimported(grading: Grading, text: str) -> list[str]:
    """The cells of a review file with a line whose letter is not what the grading records (a letter written and
    not imported yet, or one that is no letter)."""
    by_token = {c.token: c for c in grading.cells}
    out = []
    for kind, token, _sha, lines in _md_blocks(text):
        cell = by_token.get(token)
        if cell is None:
            continue
        for iid, word in lines:
            allowed = {"1": 1, "2": 2, "3": 3} if iid == "understandability" else \
                next((_letters(i.allowed) for i in cell.judged if i.id == iid), LETTERS)
            try:
                given, value = _md_answer(word, allowed)
            except ValueError:
                out.append(token)
                break
            if given and value != _md_recorded(cell, kind, iid):
                out.append(token)
                break
    return out


def _md_settle(cell: GradingCell) -> bool:
    """Finish a cell whose first verdicts are all given and that Opus has nothing else on (agrees, or no
    readable reply): the first verdicts are final.  True when it did."""
    if _owner_done(cell.state) or not _md_first_done(cell) or _md_disagrees(cell, cell.state["owner_first"]):
        return False
    first = cell.state["owner_first"]
    cell.state.update(owner={"items": _with_deferred(cell, first["items"]),
                             "understandability": first["understandability"], "at": iso(utc_now())}, changes=[])
    write_json(cell.path, cell.state)
    return True


def _md_write(grading: Grading, path: Path, kind: str, cells: Sequence[GradingCell]) -> None:
    """*cells* as one file at *path*.  A file there is overwritten only when it holds no letter the owner wrote
    and that is not imported yet (it equals the file written from the cells it names); else :class:`Refused`."""
    head = (MD_FIRST_HEAD.format(key=_md_key_path(path).name, reveal=_md_reveal_path(path).name)
            if kind == "first" else MD_FINAL_HEAD)
    blocks = []
    for cell in cells:
        # The material as shown (the forms hold only item ids and letters; Opus's reasons are scanned one by one
        # in _blinded_reason, as in the terminal, and the file's own words name Opus).
        found = _scan(grading, cell, scoring.owner_markdown(
            cell.material, scoring.data_tag(grading.state["seed"], cell.token), MD_DROP))
        if found:
            raise Refused(f"Cell {cell.token}: the review file would reveal {', '.join(found)}; nothing is written.")
        blocks.append(_md_cell(grading, cell, kind))
    content = head + "".join(f"\n\n{b}" for b in blocks) + "\n"
    unimported = _md_unimported(grading, path.read_text(encoding="utf-8")) if path.is_file() else []
    if unimported:
        raise Refused(f"{path} has letters that are not imported yet ({', '.join(unimported[:5])}"
                      f"{' …' if len(unimported) > 5 else ''}): import it first (review --dir {grading.folder} "
                      f"--import-md {path}), or move it away.")
    path.write_text(content, encoding="utf-8")


def _export_md(grading: Grading, review: Mapping[str, Any], path: Path, say: Callable[[str], None]) -> int:
    by_token = {c.token: c for c in grading.cells}
    queued = [by_token[t] for t in sorted(review["queue"], key=lambda t: int(t.lstrip("#")))
              if not _owner_done(by_token[t].state)]
    queued = [c for c in queued if not _md_settle(c)]
    first = [c for c in queued if not _md_first_done(c)]
    final = [c for c in queued if _md_first_done(c)]
    if first:
        _md_write(grading, path, "first", first)
        _md_key_path(path).write_text(_md_key(grading), encoding="utf-8")
        say(f"{len(first)} cell(s) for your own verdicts: {path} (what the letters mean: {_md_key_path(path).name})")
    if final:
        _md_write(grading, _md_reveal_path(path), "final", final)
        say(f"{len(final)} cell(s) with Opus's verdicts after yours: {_md_reveal_path(path)}")
    if not queued:
        say(f"Every queued cell is reviewed ({len(review['queue'])}).")
    return 0


def _md_answer(word: str, allowed: Mapping[str, Any]) -> tuple[bool, Any]:
    """``(given, value)`` of what the owner wrote on a line: empty is not given; a letter (or its word) of
    *allowed*; anything else :class:`ValueError`."""
    word = word.strip().strip("*_`").strip().lower()
    if not word:
        return False, None
    if word in allowed:
        return True, allowed[word]
    for value in allowed.values():
        if str(value) == word:
            return True, value
    raise ValueError(word)


def _md_blocks(text: str) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
    """``(kind, token, request hash prefix, [(id, written)])`` of each cell of a review file: the form lines that
    follow its mark (until the next mark)."""
    marks = [m for m in MD_MARK.finditer(text) if m.group(1) != "helpful"]
    out = []
    for n, m in enumerate(marks):
        body = text[m.end():marks[n + 1].start() if n + 1 < len(marks) else len(text)]
        form = body.split("\n### Your final verdicts\n" if m.group(1) == "final" else "\n### Your verdicts\n", 1)
        lines = form[1].split("\n### ", 1)[0] if len(form) == 2 else ""
        out.append((m.group(1), m.group(2), m.group(3),
                    [(lm.group(1), lm.group(3)) for lm in (MD_LINE.match(x) or MD_LINE_V1.match(x)
                                                           for x in lines.split("\n")) if lm]))
    return out


def _import_md(grading: Grading, path: Path, say: Callable[[str], None]) -> int:
    if not path.is_file():
        raise Refused(f"No review file {path}.")
    blocks = _md_blocks(path.read_text(encoding="utf-8"))
    if not blocks:
        raise Refused(f"{path} holds no cell of a review file (written by review --export-md).")
    by_token = {c.token: c for c in grading.cells}
    done = partial = kept = 0
    problems: list[str] = []
    for kind, token, sha, lines in blocks:
        cell = by_token.get(token)
        if cell is None or cell.state is None:
            problems.append(f"{token}: no such graded cell")
            continue
        if cell.request_sha256[:16] != sha:
            problems.append(f"{token}: the cell's material changed since the file was written; export again")
            continue
        if _owner_done(cell.state):
            kept += 1
            continue
        written = dict(lines)
        if kind == "first":
            first = cell.state["owner_first"] = cell.state.get("owner_first") or {}
            items = first.setdefault("items", {})
            asked = [(i.id, _letters(i.allowed)) for i in _owner_items(cell)] + \
                    [("understandability", {"1": 1, "2": 2, "3": 3})]
            for iid, allowed in asked:
                try:
                    given, value = _md_answer(written.get(iid, ""), allowed)
                except ValueError as exc:
                    problems.append(f"{token} {iid}: '{exc}' is not one of {' / '.join(allowed)}")
                    continue
                before = first.get(iid) if iid == "understandability" else items.get(iid)
                if not given:
                    continue
                if before is not None and before != value:
                    problems.append(f"{token} {iid}: your first verdict {before} was imported already and is kept "
                                    f"(review --regrade {token} gives a later one)")
                elif iid == "understandability":
                    first["understandability"] = value
                else:
                    items[iid] = value
            write_json(cell.path, cell.state)
            if not _md_first_done(cell):
                partial += 1
            elif _md_settle(cell):
                done += 1
            continue
        if not _md_first_done(cell) or cell.ai is None:
            problems.append(f"{token}: your own verdicts are not all imported; import the first file first")
            continue
        first, ai = cell.state["owner_first"], cell.ai
        final = {"items": _with_deferred(cell, first["items"]), "understandability": first["understandability"]}
        changes, bad = [], False
        for i in _owner_items(cell):
            if first["items"][i.id] == ai["items"][i.id]["verdict"]:
                continue
            try:
                given, value = _md_answer(written.get(i.id, ""), _letters(i.allowed))
            except ValueError as exc:
                problems.append(f"{token} {i.id}: '{exc}' is not one of {' / '.join(_letters(i.allowed))}")
                bad = True
                continue
            if given and value != first["items"][i.id]:
                final["items"][i.id] = value
                changes.append({"id": i.id, "opus": ai["items"][i.id]["verdict"], "first": first["items"][i.id],
                                "owner": value})
        if first["understandability"] != ai["understandability"]:
            try:
                given, value = _md_answer(written.get("understandability", ""), {"1": 1, "2": 2, "3": 3})
            except ValueError as exc:
                problems.append(f"{token} understandability: '{exc}' is not one of 1 / 2 / 3")
                bad = True
                given = False
            if given and value != first["understandability"]:
                final["understandability"] = value
                changes.append({"id": "understandability", "opus": ai["understandability"],
                                "first": first["understandability"], "owner": value})
        if bad:
            continue
        cell.state.update(owner={**final, "at": iso(utc_now())}, changes=changes)
        write_json(cell.path, cell.state)
        done += 1
    for p in problems:
        say(f"  {p}")
    review = review_state(grading)
    left = [by_token[t] for t in review["queue"] if not _owner_done(by_token[t].state)]
    reveal = [c for c in left if _md_first_done(c)]
    say(f"Imported {path.name}: {done} cell(s) reviewed, {partial} with verdicts still to give, {len(problems)} "
        f"problem(s){f', {kept} already reviewed (left as they are)' if kept else ''}. {len(left)} cell(s) still "
        "wait.")
    if reveal and blocks[0][0] == "first":
        target = _md_reveal_path(path)
        _md_write(grading, target, "final", reveal)
        say(f"{len(reveal)} cell(s) where Opus judged otherwise: {target}")
    elif reveal:
        say("Export again (review --export-md) for the cells still to finish.")
    return 1 if problems else 0


# --- the owner's helpfulness review: Opus's factual grades are final ---------------------------------------------
#: The cells rated for helpfulness per level setting (owner 2026-11-11: a small sample of each level).
HELPFUL_PER_LEVEL = 8
HELPFUL_SCALE = {1: 1, 2: 2, 3: 3}
#: The order a level's sample takes its cells in: the classes where the shape of the answer matters most first.
HELPFUL_CLASSES = ("diagnose", "interpret", "lookup")
HELPFUL_DROP = (*MD_DROP, "Found right by the automatic checks")
HELPFUL_TEXT = ("How helpful is the answer for the FlexTool user who asked: does it answer the question, at the "
                "right length, and could the user act on it? 1 = helpful: the user gets what they need; 2 = "
                "partly: the answer is there but too long, unfocused or missing something useful (such as what "
                "to change); 3 = not helpful.")
MD_HELPFUL_HEAD = """<!-- flextool-review file: helpfulness. -->
# Review: how helpful the answers are

Opus's factual grading is final; here you rate how helpful each answer is. {text}

Each cell ends with **Your rating**: write `1`, `2` or `3` after the colon, and anything you want to say about
the answer below **Your note** (any length; it stays in the grading folder, never in results.jsonl). The cells are
blind: the model and the level are not shown; `summary --dir GRADING` reports the ratings and notes by level
and model. Import the file whenever you like (`review --dir GRADING --helpfulness --import-md FILE`); a rating
imported again replaces the earlier one.
"""
HELPFUL_NOTE = "#### Your note"


def helpful_sample(grading: Grading, per_level: int) -> list[str]:
    """The cells rated for helpfulness: *per_level* of each level setting, drawn with the grading's seed from the
    cells with an answer and a readable grading of Opus's.  A level's cells are taken by class in turn
    (:data:`HELPFUL_CLASSES`); each pick is of the class's cells of the model picked least often at that level so
    far (ties in a drawn order), so the models share the sample as evenly as their answered cells allow."""
    out = []
    for level in scoring.LEVELS:
        rng = random.Random(f"{grading.state['seed']}:helpfulness:{level}")
        pool = {cls: [c for c in grading.cells if not c.empty and c.ai is not None and c.question.cls == cls
                      and c.record["cell"]["level_setting"] == level] for cls in HELPFUL_CLASSES}
        for cells in pool.values():
            rng.shuffle(cells)
        used: dict[str, int] = {}
        picked: list[str] = []
        while len(picked) < per_level and any(pool.values()):
            for cls in HELPFUL_CLASSES:
                if not pool[cls] or len(picked) >= per_level:
                    continue
                cell = min(pool[cls], key=lambda c: used.get(str(c.key[0]), 0))   # the first of the fewest
                pool[cls].remove(cell)
                used[str(cell.key[0])] = used.get(str(cell.key[0]), 0) + 1
                picked.append(cell.token)
        out += picked
    return sorted(out, key=lambda t: int(t.lstrip("#")))


def helpful_state(grading: Grading, review: dict[str, Any], per_level: int | None) -> dict[str, Any]:
    """``review.json``'s ``helpfulness`` (its sample, drawn once) and ``opus_final`` (set with it: Opus's factual
    grades are final, the factual queue no longer waits)."""
    if review.get("helpfulness") is None:
        n = per_level or HELPFUL_PER_LEVEL
        review["helpfulness"] = {"per_level": n, "sample": helpful_sample(grading, n), "drawn": iso(utc_now())}
        review["opus_final"] = True
        write_json(grading.folder / REVIEW_FILE, review)
    elif per_level is not None and per_level != review["helpfulness"]["per_level"]:
        raise Refused(f"The helpfulness sample was drawn with {review['helpfulness']['per_level']} cells per level; "
                      "it is never drawn again.")
    return review["helpfulness"]


def _opus_line(cell: GradingCell) -> str:
    """Opus's factual grading of a cell in one line: its outcome and the items it found missing or wrong."""
    verdicts = {k: v["verdict"] for k, v in cell.ai["items"].items()}
    counts = scoring.verdict_counts(cell.items, verdicts)
    detail = [f"{name}: {', '.join(f'`{i}`' for i in counts[name])}" for name in ("missing", "wrong") if counts[name]]
    return (f"Opus's factual grading: **{scoring.outcome(cell.items, verdicts)}**"
            + (f" ({'; '.join(detail)})" if detail else "") + ".")


def _helpful_cell(grading: Grading, cell: GradingCell) -> str:
    tag = scoring.data_tag(grading.state["seed"], cell.token)
    given = cell.state.get("helpfulness") or {}
    return "\n".join([
        f"# Cell {cell.token}", f"<!-- flextool-review helpful {cell.token} {cell.request_sha256[:16]} -->", "",
        scoring.owner_markdown(cell.material, tag, HELPFUL_DROP), "", "### Opus's grading", "", _opus_line(cell),
        "", "### Your rating", "",
        f"- `helpfulness` [1 / 2 / 3]: {given.get('rating') or ''}".rstrip(), "", HELPFUL_NOTE, "",
        *([given["note"], ""] if given.get("note") else [])]).rstrip("\n")


def _export_helpful(grading: Grading, sample: Sequence[str], path: Path, say: Callable[[str], None]) -> int:
    by_token = {c.token: c for c in grading.cells}
    cells = [by_token[t] for t in sample]
    blocks = []
    for cell in cells:
        found = _scan(grading, cell, scoring.owner_markdown(
            cell.material, scoring.data_tag(grading.state["seed"], cell.token), HELPFUL_DROP))
        if found:
            raise Refused(f"Cell {cell.token}: the review file would reveal {', '.join(found)}; nothing is written.")
        blocks.append(_helpful_cell(grading, cell))
    if path.is_file():
        changed = []
        for t, rating, note in _helpful_blocks(path.read_text(encoding="utf-8")):
            h = ((by_token[t].state or {}).get("helpfulness") or {}) if t in by_token else None
            if h is not None and (rating == -1 or (rating is not None and rating != h.get("rating"))
                                  or note != (h.get("note") or "")):
                changed.append(t)
        if changed:
            raise Refused(f"{path} has ratings or notes that are not imported yet ({', '.join(changed[:5])}): import "
                          f"it first (review --dir {grading.folder} --helpfulness --import-md {path}), or move it "
                          "away.")
    path.write_text(MD_HELPFUL_HEAD.format(text=HELPFUL_TEXT) + "".join(f"\n\n{b}" for b in blocks) + "\n",
                    encoding="utf-8")
    rated = sum(1 for c in cells if (c.state.get("helpfulness") or {}).get("rating"))
    say(f"{len(cells)} cell(s) to rate for helpfulness ({rated} rated already): {path}")
    return 0


def _helpful_blocks(text: str) -> list[tuple[str, int | None, str]]:
    """``(token, rating, note)`` of each cell of a helpfulness file; a rating that is no 1–3 is ``-1``."""
    marks = [m for m in MD_MARK.finditer(text) if m.group(1) == "helpful"]
    out = []
    for n, m in enumerate(marks):
        body = text[m.end():marks[n + 1].start() if n + 1 < len(marks) else len(text)]
        body = re.sub(r"\n# Cell #\d+\s*$", "\n", body)
        line = re.search(r"^- `helpfulness` \[1 / 2 / 3\]:[ \t]*(.*?)[ \t]*$", body, re.MULTILINE)
        word = line.group(1).strip().strip("*_`") if line else ""
        rating = None if not word else int(word) if word in ("1", "2", "3") else -1
        note = body.split(f"\n{HELPFUL_NOTE}\n", 1)[1].strip() if f"\n{HELPFUL_NOTE}\n" in body else ""
        out.append((m.group(2), rating, note))
    return out


def _import_helpful(grading: Grading, sample: Sequence[str], path: Path, say: Callable[[str], None]) -> int:
    if not path.is_file():
        raise Refused(f"No review file {path}.")
    blocks = _helpful_blocks(path.read_text(encoding="utf-8"))
    if not blocks:
        raise Refused(f"{path} holds no cell of a helpfulness file (written by review --helpfulness --export-md).")
    by_token = {c.token: c for c in grading.cells}
    problems, saved = [], 0
    for token, rating, note in blocks:
        cell = by_token.get(token)
        if cell is None or token not in sample:
            problems.append(f"{token}: not in the helpfulness sample")
            continue
        if rating == -1:
            problems.append(f"{token}: the rating is not 1, 2 or 3")
            continue
        if rating is None and not note:
            continue
        old = cell.state.get("helpfulness") or {}
        new = {"rating": rating, "note": note}
        if {"rating": old.get("rating"), "note": old.get("note") or ""} != new:
            cell.state["helpfulness"] = {**new, "at": iso(utc_now())}
            write_json(cell.path, cell.state)
            saved += 1
    for p in problems:
        say(f"  {p}")
    rated = sum(1 for t in sample if ((by_token[t].state or {}).get("helpfulness") or {}).get("rating"))
    say(f"Imported {path.name}: {saved} cell(s) saved, {len(problems)} problem(s). {rated} of {len(sample)} cells "
        "rated.")
    return 1 if problems else 0


def helpful_report(grading: Grading, sample: Sequence[str]) -> list[str]:
    """The owner's helpfulness ratings by level setting (and by model within it), then the notes by level."""
    by_token = {c.token: c for c in grading.cells}
    out = ["== Helpfulness (the owner's ratings of the sample; 1 helpful, 2 partly, 3 not) =="]
    notes = []
    for level in scoring.LEVELS:
        cells = [by_token[t] for t in sample if by_token[t].record["cell"]["level_setting"] == level]
        rated = [(c, c.state["helpfulness"]) for c in cells if (c.state.get("helpfulness") or {}).get("rating")]
        dist = [sum(1 for _c, h in rated if h["rating"] == r) for r in (1, 2, 3)]
        out.append(f"  {level:<7} rated {len(rated)} of {len(cells)}  1/2/3 {dist[0]}/{dist[1]}/{dist[2]}")
        models: dict[str, list[int]] = {}
        for c, h in rated:
            models.setdefault(f"{c.run.get('provider')} {c.run.get('model')}", []).append(h["rating"])
        for model, ratings in sorted(models.items()):
            out.append(f"    {model}: {' '.join(map(str, ratings))}")
        for c in cells:
            note = (c.state.get("helpfulness") or {}).get("note")
            if note:
                notes.append(f"  {c.token} ({level}, {c.question.id} {c.question.cls}, {c.run.get('model')}): "
                             + " ".join(note.split()))
    if notes:
        out += ["Notes:", *notes]
    return out


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
    if args.helpfulness:
        if not (args.export_md or args.import_md):
            raise Refused("--helpfulness goes with --export-md FILE or --import-md FILE.")
        sample = helpful_state(grading, review, args.per_level)["sample"]
        if args.export_md:
            return _export_helpful(grading, sample, args.export_md, say)
        return _import_helpful(grading, sample, args.import_md, say)
    if args.per_level is not None:
        raise Refused("--per-level goes with --helpfulness.")
    if args.export_md or args.import_md:
        if args.regrade is not None:
            raise Refused("--regrade works in the terminal only.")
        if args.export_md:
            return _export_md(grading, review, args.export_md, say)
        return _import_md(grading, args.import_md, say)
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
    queued or Opus's grades are final (``opus_final``, the helpfulness review); ``None`` while it waits (not graded,
    or queued and not reviewed)."""
    import assistant_eval_analysis as analysis

    state = cell.state or {}
    if cell.empty and state:
        return {"grader": "none", "outcome": "failed", "verdicts": {}, "understandability": None}
    if _owner_done(state):
        owner = (state.get("regrades") or [{}])[-1].get("owner") or state["owner"]
        return {"grader": "owner", "outcome": scoring.outcome(cell.items, owner["items"]),
                "verdicts": owner["items"], "understandability": owner["understandability"]}
    if review is None or (cell.token in review["queue"] and not review.get("opus_final")) or cell.ai is None:
        return None
    ai = cell.ai
    verdicts = {k: v["verdict"] for k, v in ai["items"].items()}
    return {"grader": ai.get("grader") or analysis.GRADER_MODEL, "outcome": scoring.outcome(cell.items, verdicts),
            "verdicts": verdicts, "understandability": ai["understandability"]}


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
                              "helpfulness": None,
                              "ai_outcome": None, "ai_understandability": None, "ai_confidence": None,
                              "ai_route": None, "items_overridden": [], "diagnostics": {}}
    if cell is not None and grade is not None:
        ai = cell.ai
        ai_verdicts = {k: v["verdict"] for k, v in ai["items"].items()} if ai else {}
        graded.update(grader=grade["grader"], reviewed=grade["grader"] == "owner",
                      review_reasons=list((review or {}).get("queue", {}).get(cell.token, [])),
                      outcome=grade["outcome"], **scoring.verdict_counts(cell.items, grade["verdicts"]),
                      no_answer=cell.empty, cut_off=scoring.cut_off(_turn_views(record)),
                      understandability=grade["understandability"],
                      helpfulness=((cell.state or {}).get("helpfulness") or {}).get("rating"),
                      ai_outcome=scoring.outcome(cell.items, ai_verdicts) if ai else None,
                      ai_understandability=ai["understandability"] if ai else None,
                      ai_confidence=ai["confidence"] if ai else None, ai_route=cell.route if ai else None,
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
        # How often the cell was continued from a checkpoint (run --resume), where and after which stop.
        "resumes": int((record.get("resume") or {}).get("count") or 0),
        "resume_points": [[p.get("turn"), p.get("round")] for p in (record.get("resume") or {}).get("points") or []],
        "resume_codes": [p.get("code") for p in (record.get("resume") or {}).get("points") or []],
    }


def skipped_line(cell: SkippedCell, *, key_version: int | None) -> dict[str, Any]:
    """The ``results.jsonl`` line of a cell the owner skipped: outcome ``skipped`` with the reason and attempts,
    left out of every rate (``calibrate`` lists it apart)."""
    question, level, repeat = parse_cell_id(cell.cid)
    record = {"cell": {"model": cell.run.get("model"), "level_setting": level, "question": question,
                       "class": cell.question.cls, "repeat": repeat},
              "model": {"preset": cell.run.get("provider"), "free_tier": cell.run.get("free_tier")},
              "set_version": cell.run.get("set_version"), "ran_at": cell.entry.get("skipped_at"), "turns": []}
    line = summary_line(record, cell.run, key_version=key_version)
    line.update(outcome="skipped", skip_reason=cell.entry.get("reason"), skip_attempts=cell.entry.get("attempts"))
    return line


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
        # Items the owner left to Opus (the review files' numbers) count in the owner's outcome with Opus's
        # verdict, and in no item pair.
        out.append(analysis.ReviewedCell(
            token=c.token, question=c.question.id,
            items=tuple((first["items"][i.id], ai[i.id]) for i in c.judged if i.id in first["items"]),
            outcome=(scoring.outcome(c.items, {**ai, **first["items"]}), scoring.outcome(c.items, ai)),
            understandability=(first["understandability"], c.ai["understandability"]),
            sample=c.token in review["sample"],
            own_matched=[r for r in reasons if r not in ("sample", "all")] == ["own_model"],
            route=c.route or API_ROUTE))
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
    for line in analysis.mixed_settings(c.record for c in grading.cells):
        say(line)
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
        attempts = analysis.failed_attempts_line(analysis.failed_attempt_codes(c.record for c in cells))
        if attempts:
            say(attempts)
        na = sum(1 for f, _r, _rec, _c in grading.not_applicable if str(f) == folder)
        if na:
            say(f"  not applicable (context): {na}")
        skips = [s for s in grading.skipped if str(s.folder) == folder]
        if skips:
            say(f"  skipped: {len(skips)} (by the owner, in no rate: "
                + "; ".join(f"{s.cid}: {s.entry.get('reason')}" for s in skips) + ")")
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
    if review is not None and review.get("helpfulness"):
        for line in helpful_report(grading, review["helpfulness"]["sample"]):
            say(line)
    routes: dict[str, int] = {}
    for c in grading.cells:
        if c.route is not None:
            routes[c.route] = routes.get(c.route, 0) + 1
    if routes:
        say("AI grades by route: " + ", ".join(f"{r} {routes[r]}" for r in sorted(routes))
            + f" ({API_ROUTE}: {analysis.GRADER_MODEL} through the API; {FILE_ROUTE}: {FILE_GRADER}).")
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
        if (c.state.get("helpfulness") or {}).get("rating"):
            tag += f":h{c.state['helpfulness']['rating']}"    # a rating given or changed later is a new line
        if tag in (c.state.get("recorded") or []):
            continue
        lines.append((c, tag, summary_line(c.record, c.run, key_version=grading.key.key_version, cell=c,
                                           grade=grades[c.token], review=review)))
    done_na = {tuple(x) for x in grading.state.get("recorded_not_applicable") or []}
    na_lines = [((str(f), cid), summary_line(r, run, key_version=grading.key.key_version))
                for f, run, r, cid in grading.not_applicable if (str(f), cid) not in done_na]
    done_skip = {tuple(x) for x in grading.state.get("recorded_skipped") or []}
    skip_lines = [((str(s.folder), s.cid), skipped_line(s, key_version=grading.key.key_version))
                  for s in grading.skipped if (str(s.folder), s.cid) not in done_skip]
    total = append_lines(Path(args.results_file), [line for _c, _t, line in lines] + [x for _k, x in na_lines]
                         + [x for _k, x in skip_lines])
    for c, tag, _line in lines:
        c.state.setdefault("recorded", []).append(tag)
        write_json(c.path, c.state)
    if na_lines:
        grading.state["recorded_not_applicable"] = sorted([*map(list, done_na), *(list(k) for k, _x in na_lines)])
    if skip_lines:
        grading.state["recorded_skipped"] = sorted([*map(list, done_skip), *(list(k) for k, _x in skip_lines)])
    if na_lines or skip_lines:
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
    p.add_argument("--pace", type=float, default=0.0, metavar="SECONDS",
                   help="at least this many seconds between two request starts in this run, on top of FlexTool's "
                        "own pacing (e.g. 20 for Gemma via Google); recorded, may change between resumes")
    p.add_argument("--no-stream", action="store_true",
                   help="ask for whole answers instead of streamed ones (OpenAI-compatible providers); recorded, "
                        "may change between resumes")
    p.add_argument("--accept-data-notice", action="store_true",
                   help="accept the provider's data notice for this run only (only for a grounding project)")
    p.add_argument("--resume", action="store_true", help="continue a results folder")
    p.add_argument("--skip", action="append", metavar="CELL",
                   help="record the cell (e.g. S01__small__r1) as not applicable, skipped by the owner, with "
                        "--skip-reason; nothing is asked (repeat for more cells)")
    p.add_argument("--skip-reason", metavar="TEXT", help="why the --skip cells are left out (kept with them)")
    p.add_argument("--unskip", action="append", metavar="CELL",
                   help="take a skipped cell back: the next --resume asks it (nothing is asked now)")
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
    p.add_argument("--export-requests", type=_abs_path, metavar="OUT",
                   help="the file route: write the requests grade-ai would send now to OUT (nothing is sent, "
                        "nothing written to --dir) for Claude Code subagents to grade")
    p.add_argument("--import-replies", type=_abs_path, metavar="OUT",
                   help="the file route: record the replies the subagents wrote to OUT (OUT/<n>.reply.json)")
    p.add_argument("--key", default=str(scoring.KEY_PATH), help=argparse.SUPPRESS)
    p.add_argument("--set", default=str(scoring.SET_PATH), help=argparse.SUPPRESS)

    p = sub.add_parser("review", help="the owner reviews the queued cells: own verdict first, then Opus's")
    p.add_argument("--dir", required=True, type=_abs_path, help="the grading folder")
    p.add_argument("--sitting", type=int, default=DEFAULT_SITTING, help="cells to review in one sitting")
    p.add_argument("--regrade", metavar="#N", help="review cell #N again (the latest review wins)")
    p.add_argument("--all", action="store_true", help="queue every cell (full manual grading)")
    md = p.add_mutually_exclusive_group()
    md.add_argument("--export-md", metavar="FILE", type=_abs_path,
                    help="write the queued cells to a markdown file to fill in an editor (every cell, no sitting)")
    md.add_argument("--import-md", metavar="FILE", type=_abs_path,
                    help="record the verdicts written in a review file; disagreements go to FILE.final.md")
    p.add_argument("--helpfulness", action="store_true",
                   help="rate how helpful the answers of a small sample per level are (with --export-md or "
                        "--import-md); Opus's factual grades become final")
    p.add_argument("--per-level", type=int, metavar="N",
                   help=f"cells per level setting in the helpfulness sample (default {HELPFUL_PER_LEVEL}; drawn once)")
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
