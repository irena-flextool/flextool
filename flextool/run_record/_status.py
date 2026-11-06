"""The engine side of the run record: ``_run_status.json`` of one engine run.

:class:`RunRecorder` is created by ``cmd_run_flextool.main`` once the scenario and the output folder are known
(:meth:`RunRecorder.start`), made the process' active recorder (:func:`set_active`) and fed by hooks in the solve
loop, the acceptance check, the Benders driver, the database reader and ``write_outputs``.  Those hooks call
:func:`active`, which returns a no-op recorder when no run is recording (a library call, a test, a replay).
The active recorder is a module global rather than a ``RunnerState`` field because ``write_outputs`` and the
database version check run outside the state the cascade builds, and the console copy and the warning handler
are process-wide anyway.

Writes are throttled to at most one per second (a heartbeat thread flushes a pending change on its 1 s tick and
stamps ``heartbeat_at`` every 30 s); a solve start and the final write go out at once.  An ``OSError`` while
writing logs one warning and turns the recorder into a no-op: the run record never changes a run's exit code.
When another run takes the folder over (another ``run_id`` on disk) this recorder stops writing.

``psutil`` is imported lazily (process start time, :func:`effective_state`).
"""
from __future__ import annotations

import glob
import logging
import math
import os
import platform as _platform
import socket
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from flextool.run_record._owner import (
    RUN_LOG_FILE,
    RUN_STATUS_FILE,
    STATUS_SCHEMA,
    STATUS_SCHEMA_VERSION,
    STATUS_TMP_PREFIX,
    STATUS_TMP_SUFFIX,
    now_iso,
    parse_iso,
    read_status_file,
    worst_verdict,
    write_log_header,
    write_status_atomic,
)

if TYPE_CHECKING:
    from flextool.run_options import RunOptions

_logger = logging.getLogger(__name__)

#: The command-line text that identifies an engine process (see :func:`effective_state`).
CMDLINE_TAG = "cmd_run_flextool"
HEARTBEAT_SECONDS = 30.0
_TICK_SECONDS = 1.0
_MIN_WRITE_INTERVAL = 1.0
#: A ``running`` status whose heartbeat is older than this, and whose process is gone, is ``abandoned``.
ABANDONED_AFTER_SECONDS = 600.0
_CREATE_TIME_SLACK = 5.0

MAX_SOLVES = 1000
_KEEP_FIRST_SOLVES = 100
MAX_WARNINGS = 50
_MESSAGE_LIMIT = 2000
_WARNING_MESSAGE_LIMIT = 500


def _finite(value: Any) -> float | None:
    """*value* as a float when it is a finite number (JSON has no NaN / infinity), else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pct(value: float | None) -> str:
    return f"{value * 100:.4g}%" if value is not None else "unknown"


def _jsonable(value: Any) -> Any:
    """A JSON-safe copy of an option value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _mtime_iso(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def db_path_from_url(db_url: str | None) -> Path | None:
    """The file of a ``sqlite:///`` URL or bare path, without a ``?query``; ``None`` for other databases."""
    if not db_url:
        return None
    if "://" in db_url and not db_url.startswith("sqlite:"):
        return None
    text = db_url.replace("sqlite:///", "", 1).split("?", 1)[0]
    return Path(text) if text else None


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for key, dist in (("flextool", "flextool"), ("polar_high", "polar-high"), ("highspy", "highspy")):
        try:
            out[key] = version(dist)
        except PackageNotFoundError:
            out[key] = None
    return out


def _host_key() -> tuple[str, str]:
    return socket.gethostname(), sys.platform


def _cmdline_tag(cmdline: Sequence[str]) -> str:
    """The text of this process' command line that :func:`effective_state` looks for later.

    ``cmd_run_flextool`` for the engine module (``-m flextool.cli.cmd_run_flextool``, the GUI, the supervisor);
    ``run_flextool`` for the root ``run_flextool.py`` script; else the program's name (a Spine Toolbox console
    runs the engine inside ``python -i``), where the start time alone tells a reused pid apart.
    """
    joined = " ".join(cmdline)
    for tag in (CMDLINE_TAG, "run_flextool"):
        if tag in joined:
            return tag
    if cmdline:
        return os.path.basename(cmdline[0]) or CMDLINE_TAG
    return CMDLINE_TAG


def _process_info() -> dict[str, Any]:
    create_time: float | None = None
    cmdline: list[str] = list(sys.orig_argv) if hasattr(sys, "orig_argv") else list(sys.argv)
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        create_time = proc.create_time()
        cmdline = proc.cmdline()
    except Exception:  # noqa: BLE001 - psutil missing or the process table unreadable
        pass
    host, plat = _host_key()
    return {
        "pid": os.getpid(),
        "pid_create_time": create_time,
        "host": host,
        "platform": plat,
        "python": _platform.python_version(),
        "cmdline_tag": _cmdline_tag(cmdline),
    }


def threads_record(run_options: "RunOptions") -> dict[str, Any]:
    """The run's thread budget: the facts of the run header's ``Threads:`` line.

    ``budget`` / ``source`` from ``_thread_budget.run_budget``; ``polars`` / ``polars_source`` from the decision
    made at ``import flextool`` (``already_loaded``, with the pool polars has, when polars was imported earlier or
    by an earlier run in this interpreter); ``highs`` is the ``threads`` HiGHS gets (``None`` = its default).
    """
    from flextool import _thread_budget

    budget, source = _thread_budget.run_budget(run_options)
    polars, polars_source = _thread_budget.polars_pool()
    highs = _thread_budget.highs_thread_options(run_options)[0].get("threads")
    return {
        "budget": budget,
        "source": source,
        "polars": polars,
        "polars_source": polars_source,
        "highs": highs,
    }


def options_resolved_record(run_options: "RunOptions") -> dict[str, dict[str, Any]]:
    """``{key: {value, source, env}}`` of every registered run option."""
    from flextool.run_options import OPTIONS

    out: dict[str, dict[str, Any]] = {}
    for key, resolved in run_options.resolved.items():
        spec = OPTIONS.get(key)
        out[key] = {
            "value": _jsonable(resolved.value),
            "source": resolved.source,
            "env": spec.env if spec is not None else None,
        }
    return out


def _previous_outputs(folder: Path, old: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``outputs`` a new run starts with, carried from the folder's previous status (D2).

    ``written`` -> ``previous_run`` pointing at that run; ``previous_run`` -> the same ``previous`` (so a series of
    failures points at the run that wrote the files); ``partial`` stays ``partial``.  Without a usable old record
    the folder decides: parquet files present -> ``previous_run`` of an unknown run (``finished_at`` = the
    newest parquet's time), none -> ``none``.
    """
    outputs: dict[str, Any] = {"parquet_state": "none", "written_at": None, "by": "run", "previous": None,
                               "reason": None}
    old_outputs = (old or {}).get("outputs") or {}
    old_state = old_outputs.get("parquet_state")
    if old is not None and old_state == "written":
        outputs["parquet_state"] = "previous_run"
        outputs["previous"] = {"run_id": old.get("run_id"), "finished_at": old.get("finished_at")}
        return outputs
    if old is not None and old_state == "previous_run":
        outputs["parquet_state"] = "previous_run"
        outputs["previous"] = old_outputs.get("previous")
        return outputs
    if old is not None and old_state == "partial":
        outputs["parquet_state"] = "partial"
        outputs["previous"] = old_outputs.get("previous")
        return outputs
    parquet = [p for p in folder.glob("*.parquet") if p.is_file()]
    if parquet:
        newest = max(p.stat().st_mtime for p in parquet)
        outputs["parquet_state"] = "previous_run"
        outputs["previous"] = {
            "run_id": None,
            "finished_at": datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds"),
        }
    return outputs


class _WarningHandler(logging.Handler):
    """Collects WARNING and worse records into the active recorder's ``warnings`` list."""

    def __init__(self, recorder: "RunRecorder") -> None:
        super().__init__(level=logging.WARNING)
        self._recorder = recorder
        self._busy = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._busy, "on", False):
            return
        self._busy.on = True
        try:
            try:
                message = record.getMessage()
            except Exception:  # noqa: BLE001 - a malformed log call must not break the run
                message = str(record.msg)
            self._recorder._add_warning(record.levelname, record.name, message)
        finally:
            self._busy.on = False


class RunRecorder:
    """Writes ``_run_status.json`` of one engine run.  Create it with :meth:`start`."""

    #: ``False`` for the no-op recorder returned by :func:`active` when no run records.
    is_recording = True

    def __init__(self) -> None:
        self.run_id: str = uuid.uuid4().hex
        self.folder: Path | None = None
        self._status: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dirty = False
        self._last_write = 0.0
        self._last_heartbeat = 0.0
        self._broken = False
        self._finished = False
        self._current: dict[str, Any] | None = None
        self._current_t0: float | None = None
        self._t0 = time.monotonic()
        self._handler: _WarningHandler | None = None

    # -- construction ---------------------------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        *,
        parquet_dir: Path | str,
        input: Mapping[str, Any],
        output: Mapping[str, Any],
        options: Sequence[str],
        argv: Sequence[str],
        log_capture: str,
        launcher: str,
        run_options: "RunOptions",
    ) -> "RunRecorder":
        """Create the folder, carry the previous run's outputs, truncate ``_run.log`` and write ``running``.

        *input* gives ``db_url``, ``scenario`` and optionally ``source_file``; the database file, its time and the
        source file's time are filled in here.  *output* gives ``location``, ``subdir``, ``work_folder`` and
        ``write_methods``.  Raises ``OSError`` when the folder cannot be written.
        """
        rec = cls()
        folder = Path(parquet_dir)
        folder.mkdir(parents=True, exist_ok=True)
        rec.folder = folder
        old = read_status_file(folder)
        outputs = _previous_outputs(folder, old)
        for stale in glob.glob(str(folder / (glob.escape(STATUS_TMP_PREFIX) + "*" + STATUS_TMP_SUFFIX))):
            try:
                os.remove(stale)
            except OSError:
                pass
        started_at = now_iso()
        write_log_header(folder / RUN_LOG_FILE, rec.run_id, started_at)

        db_url = input.get("db_url")
        db_path = db_path_from_url(db_url)
        source_file = input.get("source_file")
        source_path = Path(source_file) if source_file else None
        rec._status = {
            "schema": STATUS_SCHEMA,
            "schema_version": STATUS_SCHEMA_VERSION,
            "run_id": rec.run_id,
            "state": "running",
            "verdict": None,
            "exit_code": None,
            "error": None,
            "started_at": started_at,
            "updated_at": started_at,
            "heartbeat_at": started_at,
            "finished_at": None,
            "wall_time_s": None,
            "process": _process_info(),
            "launcher": launcher,
            "log_capture": log_capture,
            "argv": list(argv),
            "versions": _versions(),
            "solver": None,
            "input": {
                "db_url": db_url,
                "db_path": str(db_path.resolve()) if db_path is not None else None,
                "db_mtime": _mtime_iso(db_path),
                "db_version": input.get("db_version"),
                "scenario": input.get("scenario"),
                "alternatives": list(input.get("alternatives") or []),
                "source_file": str(source_path.resolve()) if source_path is not None else None,
                "source_file_mtime": _mtime_iso(source_path),
            },
            "output": {
                "location": output.get("location"),
                "subdir": output.get("subdir"),
                "parquet_dir": str(folder.resolve()),
                "work_folder": output.get("work_folder"),
                "write_methods": list(output.get("write_methods") or []),
            },
            "options": list(options),
            "options_resolved": options_resolved_record(run_options),
            "threads": threads_record(run_options),
            "outputs": outputs,
            "progress": None,
            "solves": [],
            "solves_omitted": 0,
            "warnings": [],
            "warnings_omitted": 0,
        }
        write_status_atomic(folder, rec._status)
        rec._last_write = rec._last_heartbeat = time.monotonic()
        rec._thread = threading.Thread(target=rec._heartbeat_loop, name="flextool-run-record", daemon=True)
        rec._thread.start()
        return rec

    # -- writing --------------------------------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(_TICK_SECONDS):
            with self._lock:
                if self._broken or self._finished:
                    continue
                now = time.monotonic()
                if now - self._last_heartbeat >= HEARTBEAT_SECONDS:
                    self._last_heartbeat = now
                    self._status["heartbeat_at"] = now_iso()
                    self._write(force=True)
                elif self._dirty:
                    self._write()

    def _write(self, *, force: bool = False, final: bool = False) -> None:
        """Write the status (caller holds the lock); throttled unless *force*."""
        if self._broken or self.folder is None:
            return
        now = time.monotonic()
        if not force and now - self._last_write < _MIN_WRITE_INTERVAL:
            self._dirty = True
            return
        on_disk = read_status_file(self.folder)
        if on_disk is not None and on_disk.get("run_id") != self.run_id:
            self._broken = True
            _logger.warning(
                "Another run took over %s (run %s); this run no longer updates its run record.",
                self.folder, on_disk.get("run_id"),
            )
            return
        self._status["updated_at"] = now_iso()
        try:
            write_status_atomic(self.folder, self._status, final=final)
        except (OSError, ValueError) as exc:
            self._broken = True
            _logger.warning("Cannot write the run record %s: %s; the run continues without it.",
                            self.folder / RUN_STATUS_FILE, exc)
            return
        self._dirty = False
        self._last_write = now

    def _update(self, *, force: bool = False) -> None:
        with self._lock:
            self._write(force=force)

    # -- facts ----------------------------------------------------------------------------------------------

    def note_input(self, **fields: Any) -> None:
        """Add facts about the input: ``db_version``, ``alternatives`` (ranked), ..."""
        with self._lock:
            target = self._status.setdefault("input", {})
            for key, value in fields.items():
                if key == "alternatives":
                    value = [str(v) for v in value]
                target[key] = value
            self._write()

    def note_solver(self, name: str | None) -> None:
        """The solver a solve was dispatched to (the last one wins)."""
        with self._lock:
            if name is not None and self._status.get("solver") != name:
                self._status["solver"] = str(name)
                self._write()

    def solve_started(self, name: str, parent: str | None, index: int, log_anchor: str | None) -> None:
        """A solve of the loop starts (``name`` = the per-roll solve, ``parent`` its complete solve)."""
        with self._lock:
            entry = {
                "name": name,
                "parent": parent if parent and parent != name else None,
                "index": index,
                "kind": "lp",
                "state": "running",
                "model_status": None,
                "acceptance": None,
                "message": None,
                "objective": None,
                "started_at": now_iso(),
                "wall_s": None,
                "warm_used": None,
                "benders": None,
                "log_anchor": log_anchor,
            }
            solves = self._status["solves"]
            solves.append(entry)
            if len(solves) > MAX_SOLVES:
                del solves[_KEEP_FIRST_SOLVES]
                self._status["solves_omitted"] += 1
            self._current = entry
            self._current_t0 = time.monotonic()
            self._status["progress"] = None
            self._write(force=True)

    def solve_classified(
        self,
        *,
        model_status: str | None,
        acceptance: str,
        message: str | None,
        objective: float | None,
        warm_used: bool | None,
    ) -> None:
        """The acceptance check's decision on the running LP solve (``optimal`` / ``near_optimal`` /
        ``rejected``)."""
        with self._lock:
            entry = self._current
            if entry is None:
                return
            entry["state"] = "rejected" if acceptance == "rejected" else "accepted"
            entry["model_status"] = model_status
            entry["acceptance"] = acceptance
            entry["message"] = (message or None) and message[:_MESSAGE_LIMIT]
            entry["objective"] = _finite(objective)
            entry["warm_used"] = None if warm_used is None else bool(warm_used)
            self._status["progress"] = None
            self._status["verdict"] = worst_verdict(self._status["solves"])
            self._write()

    def solve_benders(
        self,
        *,
        converged: bool,
        has_incumbent: bool,
        objective: float | None,
        gap: float | None,
        tol: float | None,
        iterations: int | None,
    ) -> None:
        """The running solve was a Benders decomposition: ``converged``, ``not_converged`` (a feasible incumbent,
        the run goes on) or ``no_incumbent`` (the run fails)."""
        with self._lock:
            entry = self._current
            if entry is None:
                return
            entry["kind"] = "benders"
            gap_v, tol_v = _finite(gap), _finite(tol)
            if converged:
                entry["acceptance"], entry["state"] = "converged", "accepted"
                entry["message"] = None
            elif has_incumbent:
                entry["acceptance"], entry["state"] = "not_converged", "accepted"
                entry["message"] = (
                    f"decomposed solve {entry['name']!r} did not meet its convergence tolerance after "
                    f"{iterations} iterations (gap {_pct(gap_v)}, tolerance {_pct(tol_v)}); its results are "
                    "feasible but not certified optimal"
                )
            else:
                entry["acceptance"], entry["state"] = "no_incumbent", "rejected"
                entry["message"] = (
                    f"decomposed solve {entry['name']!r} found no feasible solution in {iterations} iterations"
                )
            entry["objective"] = _finite(objective)
            entry["warm_used"] = False
            entry["benders"] = {"gap": gap_v, "tol": tol_v,
                                "iterations": None if iterations is None else int(iterations)}
            self._status["progress"] = None
            self._status["verdict"] = worst_verdict(self._status["solves"])
            self._write()

    def benders_progress(self, iteration: int, gap: float | None) -> None:
        """A Benders iteration finished (shown as ``progress`` until the solve ends)."""
        with self._lock:
            entry = self._current
            self._status["progress"] = {
                "solve": entry["name"] if entry is not None else None,
                "benders_iteration": int(iteration),
                "benders_gap": _finite(gap),
            }
            self._write()

    def solve_finished(self, exit_status: int) -> None:
        """``solver.run`` returned *exit_status* for the running solve."""
        with self._lock:
            entry = self._current
            if entry is None:
                return
            if self._current_t0 is not None:
                entry["wall_s"] = round(time.monotonic() - self._current_t0, 3)
            if entry["state"] == "running":
                entry["state"] = "accepted" if exit_status == 0 else "error"
            self._current = None
            self._current_t0 = None
            self._status["verdict"] = worst_verdict(self._status["solves"])
            self._write()

    def _set_outputs(self, **fields: Any) -> None:
        with self._lock:
            self._status["outputs"].update(fields)
            self._write()

    def outputs_cleared(self) -> None:
        """``write_outputs`` emptied the parquet folder: until :meth:`outputs_written` it holds a partial set."""
        self._set_outputs(parquet_state="partial", previous=None, reason=None)

    def outputs_written(self, by: str = "run") -> None:
        """The parquet files of this run are complete."""
        self._set_outputs(parquet_state="written", written_at=now_iso(), by=by, reason=None)

    def outputs_skipped(self, reason: str) -> None:
        """The run wrote no outputs although it succeeded (the folder keeps what it had)."""
        with self._lock:
            outputs = self._status["outputs"]
            if outputs.get("parquet_state") != "partial":
                outputs["parquet_state"] = "skipped"
            outputs["reason"] = str(reason)
            self._write()

    def outputs_not_requested(self) -> None:
        """The run's write methods have no parquet."""
        with self._lock:
            outputs = self._status["outputs"]
            if outputs.get("parquet_state") != "written":
                outputs["parquet_state"] = "not_requested"
                outputs["reason"] = None
            self._write()

    @property
    def parquet_state(self) -> str | None:
        with self._lock:
            return (self._status.get("outputs") or {}).get("parquet_state")

    def _add_warning(self, level: str, logger_name: str, message: str) -> None:
        with self._lock:
            if self._finished:
                return
            warnings = self._status.setdefault("warnings", [])
            if len(warnings) >= MAX_WARNINGS:
                self._status["warnings_omitted"] = self._status.get("warnings_omitted", 0) + 1
            else:
                warnings.append({"level": level, "logger": logger_name, "message": message[:_WARNING_MESSAGE_LIMIT]})
            self._write()

    def warning_handler(self) -> logging.Handler:
        """A logging handler (WARNING and worse) that adds each record to ``warnings`` (at most 50 kept)."""
        if self._handler is None:
            self._handler = _WarningHandler(self)
        return self._handler

    # -- end ------------------------------------------------------------------------------------------------

    def finish(self, state: str, *, exit_code: int | None, error: Mapping[str, Any] | None) -> None:
        """Write the run's final state (idempotent: only the first call counts).

        For a ``failed`` run with a rejected solve, the error message is that solve's acceptance message (the
        exception the solve loop raises only says ``Error: 1``).  Solves still running become ``error``.
        """
        with self._lock:
            if self._finished:
                return
            status = self._status
            err = dict(error) if error else None
            if state != "succeeded":
                for entry in status["solves"]:
                    if entry.get("state") == "running":
                        entry["state"] = "error"
            if err is not None and state == "failed":
                rejected = [e for e in status["solves"] if e.get("state") == "rejected" and e.get("message")]
                if rejected:
                    err["message"] = rejected[-1]["message"]
            if err is not None:
                err["message"] = str(err.get("message") or "")[:_MESSAGE_LIMIT]
            status["state"] = state
            status["exit_code"] = exit_code
            status["error"] = err
            status["verdict"] = worst_verdict(status["solves"])
            status["progress"] = None
            status["finished_at"] = now_iso()
            status["wall_time_s"] = round(time.monotonic() - self._t0, 3)
            self._write(force=True, final=True)
            self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def close(self) -> None:
        """Stop the heartbeat thread and write a pending change."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            if self._dirty and not self._finished:
                self._write(force=True)

    def snapshot(self) -> dict[str, Any]:
        """A copy of the status as it would be written now (for tests and diagnostics)."""
        import copy

        with self._lock:
            return copy.deepcopy(self._status)


class _NullRecorder:
    """The recorder :func:`active` returns when no run records: every call does nothing."""

    is_recording = False
    run_id = None
    folder = None
    finished = False
    parquet_state = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _noop

    def warning_handler(self) -> logging.Handler:
        return logging.NullHandler()


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


NULL_RECORDER = _NullRecorder()
_active: RunRecorder | None = None


def active() -> "RunRecorder | _NullRecorder":
    """The recorder of the run in progress in this process, or a no-op recorder."""
    rec = _active
    if rec is None or rec._broken:
        return NULL_RECORDER
    return rec


def set_active(rec: RunRecorder | None) -> None:
    """Make *rec* the process' active recorder (``None`` ends it)."""
    global _active
    _active = rec


def effective_state(status: Mapping[str, Any] | None, *, now: datetime | None = None) -> str | None:
    """The state to show for *status*: its ``state``, except that a ``running`` record is checked.

    ``running`` stays when the recording process is alive on this host (host name and platform match, which
    also tells WSL from Windows), is not a zombie, started within 5 s of the recorded start time and has the
    recorded ``cmdline_tag`` (``cmd_run_flextool`` for an engine module run) in its command line.  Otherwise a record whose heartbeat is more than 10 minutes old is
    ``abandoned``, a newer one ``unknown`` (the process may be on another host, or just ended).
    """
    if status is None:
        return None
    state = status.get("state")
    if state != "running":
        return state
    process = status.get("process") or {}
    if (process.get("host"), process.get("platform")) == _host_key() and _process_alive(process):
        return "running"
    beat = parse_iso(status.get("heartbeat_at")) or parse_iso(status.get("started_at"))
    current = now or datetime.now().astimezone()
    if beat is not None and (current - beat).total_seconds() > ABANDONED_AFTER_SECONDS:
        return "abandoned"
    return "unknown"


def _process_alive(process: Mapping[str, Any]) -> bool:
    try:
        import psutil
    except ImportError:
        return False
    pid = process.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        recorded = _finite(process.get("pid_create_time"))
        if recorded is None or abs(proc.create_time() - recorded) > _CREATE_TIME_SLACK:
            return False
        tag = process.get("cmdline_tag") or CMDLINE_TAG
        return tag in " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False


__all__ = [
    "NULL_RECORDER",
    "RunRecorder",
    "active",
    "db_path_from_url",
    "effective_state",
    "options_resolved_record",
    "set_active",
    "threads_record",
]
