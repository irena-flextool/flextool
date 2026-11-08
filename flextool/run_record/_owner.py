"""The run record's files, and the side of it that the owner of a run's output pipe needs.

Every engine run writes ``_run_status.json`` into its ``output_parquet/<subdir>/`` folder (see
:mod:`flextool.run_record._status`).  ``_run.log`` holds the run's console output.  It is written by the process
that holds the engine's stdout pipe (the GUI job thread, the ``flextool-run`` supervisor): only that process sees
the native solver's lines and a crash dump.  Without such an owner the engine copies its own Python streams into
the log (:mod:`flextool.run_record._tee`).

This module uses the standard library only and no relative or ``flextool`` imports: the supervisor loads it from
its file without importing ``flextool`` (which imports polars).  It holds:

* the file names, the marker the engine prints for its pipe owner and the environment variable names;
* :class:`RunLogWriter`, the tail-capped ``_run.log`` writer, and :class:`RunLogOwner`, the pipe owner's relay;
* :func:`read_status_file` / :func:`write_status_atomic` and the owner's :func:`finalize_unfinished`; a strict
  read tells a file that is absent (``None``) from one that cannot be read now (``{"state": "unreadable"}``) and
  one that is damaged or has the wrong shape (``{"state": "malformed"}``);
* :func:`results_usable` / :func:`unusable_reason` / :func:`unusable_code`, the one rule the GUI and the
  assistant share for whether a folder's results may be offered.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

_logger = logging.getLogger(__name__)

RUN_STATUS_FILE = "_run_status.json"
RUN_LOG_FILE = "_run.log"
#: The files of the run record; ``write_outputs`` keeps them when it empties a parquet folder.
RUN_RECORD_FILES = frozenset({RUN_STATUS_FILE, RUN_LOG_FILE})
#: A status temp file (``_run_status.json.<pid>.tmp``) starts with this and ends with :data:`STATUS_TMP_SUFFIX`.
STATUS_TMP_PREFIX = RUN_STATUS_FILE + "."
STATUS_TMP_SUFFIX = ".tmp"

STATUS_SCHEMA = "flextool.run_status"
STATUS_SCHEMA_VERSION = 1

#: The line the engine prints once for its pipe owner, followed by ``{"folder": ..., "run_id": ...}`` (ASCII).
MARKER = "##flextool-run-record## "

#: Set by a pipe owner in its direct child's environment; the engine reads and removes it.
RUN_LOG_OWNER_ENV = "FLEXTOOL_RUN_LOG_OWNER"
#: The number of bytes ``_run.log`` keeps from the end of a long log.
RUN_LOG_MAX_BYTES_ENV = "FLEXTOOL_RUN_LOG_MAX_BYTES"
#: Set by the GUI for an auxiliary job's process (a calibration, a representative-period iteration) and inherited
#: by every ``flextool-run`` supervisor that process starts: the path of a file whose appearance asks the
#: supervisor to stop its engine and record the run as ``killed`` (the file holds the reason).  The supervisor
#: drops it from its engine's environment.
RUN_STOP_FILE_ENV = "FLEXTOOL_RUN_STOP_FILE"
#: The longest stop reason read from the stop file.
STOP_REASON_MAX_CHARS = 500
#: The reason recorded when the stop file cannot be read.
STOP_REASON_UNREADABLE = "stopped by its launcher"

#: ``_run.log`` keeps its first ``LOG_HEAD_BYTES`` and its last ``FLEXTOOL_RUN_LOG_MAX_BYTES`` bytes.
LOG_HEAD_BYTES = 64 * 1024
LOG_TAIL_BYTES_DEFAULT = 4 * 1024 * 1024
#: The pipe owner keeps at most this many bytes of output that arrive before the engine's marker.
PRE_MARKER_BUFFER_BYTES = 2 * 1024 * 1024
#: The pipe owner re-reads the status ``run_id`` every this many lines.
OWNER_RECHECK_LINES = 256

#: Solve verdicts, worst first: the run's ``verdict`` is the worst of its solves'.
VERDICT_ORDER = (
    "error", "infeasible", "unbounded", "limit", "not_certified", "no_incumbent", "not_converged",
    "near_optimal", "optimal",
)

_LIMIT_STATUSES = frozenset({
    "kTimeLimit", "kIterationLimit", "kMemoryLimit", "kSolutionLimit", "kObjectiveBound", "kObjectiveTarget",
    "kInterrupt", "kHighsInterrupt",
})
_ERROR_STATUSES = frozenset({
    "kSolveError", "kPresolveError", "kPostsolveError", "kLoadError", "kModelError", "kModelEmpty",
})
_INFEASIBLE_STATUSES = frozenset({"kInfeasible", "kUnboundedOrInfeasible"})

_WINDOWS_CTRL_C_EXIT = 0xC000013A

#: The ``state`` of the status a strict :func:`read_status_file` returns for a file that exists but cannot be
#: read now (locked, or cut short while it is being written): ask again in a moment.
UNREADABLE_STATE = "unreadable"
#: The ``state`` of the status a strict read returns for a file that is damaged or has the wrong shape.
MALFORMED_STATE = "malformed"
#: A status that does not parse is taken as being written while it is younger than this, else as damaged.
_WRITE_IN_PROGRESS_SECONDS = 10.0
#: How long :func:`read_status_settled` keeps trying to read a status that cannot be read now.
_FINALIZE_READ_SECONDS = 2.0


def now_iso() -> str:
    """The local time with its UTC offset, to the second (the status file's time format)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(text: Any) -> datetime | None:
    """The :class:`~datetime.datetime` of an ISO-8601 text, or ``None`` for anything else."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def log_tail_bytes(environ: Mapping[str, str] | None = None) -> int:
    """The ``_run.log`` tail size: ``FLEXTOOL_RUN_LOG_MAX_BYTES`` when it is a whole number of at least 1."""
    env = os.environ if environ is None else environ
    raw = env.get(RUN_LOG_MAX_BYTES_ENV)
    if raw is None or not raw.strip():
        return LOG_TAIL_BYTES_DEFAULT
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value < 1:
        _logger.warning("Ignoring %s=%r (must be a whole number of at least 1).", RUN_LOG_MAX_BYTES_ENV, raw)
        return LOG_TAIL_BYTES_DEFAULT
    return value


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------


def format_marker(folder: Path | str, run_id: str) -> str:
    """The marker line (without a newline) that tells the pipe owner where the run record is."""
    return MARKER + json.dumps({"folder": str(folder), "run_id": run_id}, ensure_ascii=True)


def parse_marker(line: str) -> tuple[Path, str] | None:
    """``(folder, run_id)`` of a marker line, or ``None`` when *line* is not one."""
    text = line.rstrip("\r\n")
    if not text.startswith(MARKER):
        return None
    try:
        data = json.loads(text[len(MARKER):])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    folder, run_id = data.get("folder"), data.get("run_id")
    if not isinstance(folder, str) or not folder or not isinstance(run_id, str) or not run_id:
        return None
    return Path(folder), run_id


# ---------------------------------------------------------------------------
# The stop file of an auxiliary job
# ---------------------------------------------------------------------------


def write_stop_request(path: Path | str, reason: str) -> None:
    """Write *reason* into the stop file at *path* at once (a temp file beside it, then ``os.replace``), so a
    supervisor never reads half of it.  The folder is the launcher's own private one."""
    path = Path(path)
    tmp = path.with_name(path.name + STATUS_TMP_SUFFIX)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(reason)
    os.replace(tmp, path)


def read_stop_reason(path: Path | str) -> str | None:
    """The reason in the stop file at *path* (at most :data:`STOP_REASON_MAX_CHARS` characters), ``None`` when
    there is no such file; :data:`STOP_REASON_UNREADABLE` when it is there but cannot be read."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(STOP_REASON_MAX_CHARS)
    except FileNotFoundError:
        return None
    except OSError:
        return STOP_REASON_UNREADABLE
    return text.strip() or STOP_REASON_UNREADABLE


# ---------------------------------------------------------------------------
# _run.log
# ---------------------------------------------------------------------------


def log_header(run_id: str, started_at: str) -> str:
    """The first line of ``_run.log`` (without a newline)."""
    return f"# FlexTool run {run_id} started {started_at}"


def write_log_header(path: Path, run_id: str, started_at: str) -> None:
    """Truncate ``_run.log`` at *path* to its header line."""
    with open(path, "wb") as fh:
        fh.write((log_header(run_id, started_at) + "\n").encode("utf-8"))


def _omission_line(n: int) -> bytes:
    return f"[... {n} bytes of log omitted ...]\n".encode("utf-8")


class RunLogWriter:
    """Appends lines to ``_run.log`` and keeps it to its head and its tail.

    The file is opened in binary mode and every line is encoded as UTF-8 with ``\\n`` by hand, and flushed.
    When the file grows past ``head + 2 * tail`` bytes it is compacted in place: the first *head* bytes (cut at a
    newline; fixed at the first compaction) stay, then one ``[... N bytes of log omitted ...]`` line with the
    total number of log bytes dropped so far, then the last *tail* bytes (cut at a newline).  No rename, no
    second file: a reader on Windows can always open it.

    *guard*, when given, is called before each compaction; when it returns ``False`` the writer stops writing
    (the folder was taken over by another run).  The writer is its run's only writer of the file, so before each
    write it checks the file's size against its own: any other size means another run truncated (and maybe
    extended) the log, and the writer stops (:attr:`taken_over`) instead of writing at its stale offset.  An
    ``OSError`` while writing closes the writer and is raised.  Thread-safe.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        head: int = LOG_HEAD_BYTES,
        tail: int | None = None,
        guard: Callable[[], bool] | None = None,
    ) -> None:
        self.path = Path(path)
        self.head = int(head)
        self.tail = log_tail_bytes() if tail is None else int(tail)
        self._guard = guard
        self._lock = threading.Lock()
        mode = "r+b" if self.path.exists() else "w+b"
        self._fh = open(self.path, mode)
        self._fh.seek(0, os.SEEK_END)
        self._size = self._fh.tell()
        self._head_end: int | None = None
        self._body_start = 0
        self._omitted = 0
        self._closed = False
        self._taken_over = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def taken_over(self) -> bool:
        """Whether the writer stopped because another writer changed the file (a takeover of the folder)."""
        return self._taken_over

    def _other_writer_locked(self) -> bool:
        """Whether the file's size is not the writer's own; closes the writer when so (caller holds the lock)."""
        try:
            size = os.fstat(self._fh.fileno()).st_size
        except OSError:
            return False
        if size == self._size:
            return False
        self._taken_over = True
        self._close_locked()
        return True

    @property
    def omitted_bytes(self) -> int:
        """Log bytes dropped so far (compactions and lines the caller reported with :meth:`write_omission`)."""
        return self._omitted

    def write_line(self, line: str) -> None:
        """Append *line* (a newline is added when it has none)."""
        data = line.encode("utf-8", errors="replace")
        if not data.endswith(b"\n"):
            data += b"\n"
        with self._lock:
            if self._closed or self._other_writer_locked():
                return
            try:
                self._fh.write(data)
                self._fh.flush()
                self._size += len(data)
                if self._size > self.head + 2 * self.tail:
                    self._compact()
            except OSError:
                # The file's size is no longer known: stop (not a takeover) and let the caller report it.
                self._close_locked()
                raise

    def write_omission(self, n: int) -> None:
        """Record *n* log bytes the caller dropped before writing them (counted into later omission lines)."""
        if n <= 0:
            return
        with self._lock:
            if self._closed or self._other_writer_locked():
                return
            self._omitted += n
            line = _omission_line(self._omitted)
            try:
                self._fh.write(line)
                self._fh.flush()
            except OSError:
                self._close_locked()
                raise
            self._size += len(line)

    def _compact(self) -> None:
        if self._guard is not None and not self._guard():
            self._close_locked()
            return
        fh = self._fh
        if self._head_end is None:
            fh.seek(0)
            head = fh.read(min(self.head, self._size))
            cut = head.rfind(b"\n")
            self._head_end = cut + 1 if cut >= 0 else len(head)
            body_start = self._head_end
        else:
            body_start = self._body_start
        tail_from = max(body_start, self._size - self.tail)
        fh.seek(tail_from)
        kept = fh.read()
        if tail_from > body_start:
            # Start the kept tail at a line start.
            nl = kept.find(b"\n")
            if nl >= 0:
                kept = kept[nl + 1:]
        dropped = self._size - len(kept) - body_start
        self._omitted += dropped
        omission = _omission_line(self._omitted)
        fh.seek(self._head_end)
        fh.write(omission)
        fh.write(kept)
        fh.truncate()
        fh.flush()
        self._body_start = self._head_end + len(omission)
        self._size = self._body_start + len(kept)

    def _close_locked(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._close_locked()


class RunLogOwner:
    """The relay of the process that holds an engine run's output pipe (the GUI job thread, the supervisor).

    Create it before the engine starts and pass every output line to :meth:`feed`.  Lines that arrive before the
    engine's marker are buffered (at most :data:`PRE_MARKER_BUFFER_BYTES`; the oldest are dropped first and
    counted).  The first marker whose ``run_id`` matches the ``_run_status.json`` on disk opens that folder's
    ``_run.log`` at its end (after the engine's header), writes the buffered lines and then every later line.
    The marker line itself is hidden (:meth:`feed` returns ``False``); a later marker is an ordinary line.
    Every :data:`OWNER_RECHECK_LINES` lines and before each compaction the status ``run_id`` is read again; when
    another run has taken the folder over, the owner stops writing.

    :meth:`close` finalises a status that the engine left ``running`` (crash, kill, signal) from the exit code.
    *launcher* names the owner (``gui``, ``flextool-run``) in the lines it adds to the log.
    """

    def __init__(
        self,
        *,
        launcher: str,
        preamble: Iterable[str] = (),
        head: int = LOG_HEAD_BYTES,
        tail: int | None = None,
        buffer_bytes: int = PRE_MARKER_BUFFER_BYTES,
    ) -> None:
        self.launcher = launcher
        self._head = head
        self._tail = tail
        self._buffer_limit = int(buffer_bytes)
        self._buffer: collections.deque[str] = collections.deque()
        self._buffer_size = 0
        self._dropped_bytes = 0
        self._marker_seen = False
        self._writer: RunLogWriter | None = None
        self.folder: Path | None = None
        self.run_id: str | None = None
        self._lines_since_check = 0
        self._lock = threading.Lock()
        self._closed = False
        #: Whether :meth:`close` changed the run's ``_run_status.json`` (it finished a record the engine left).
        self.record_changed = False
        for line in preamble:
            self._buffer_line(line)

    @property
    def attached(self) -> bool:
        """Whether the owner writes into a run's ``_run.log``."""
        return self._writer is not None and not self._writer.closed

    def _buffer_line(self, line: str) -> None:
        size = len(line.encode("utf-8", errors="replace")) + (0 if line.endswith("\n") else 1)
        self._buffer.append(line)
        self._buffer_size += size
        while self._buffer_size > self._buffer_limit and self._buffer:
            old = self._buffer.popleft()
            old_size = len(old.encode("utf-8", errors="replace")) + (0 if old.endswith("\n") else 1)
            self._buffer_size -= old_size
            self._dropped_bytes += old_size

    def _still_ours(self) -> bool:
        """Whether the folder's status is still this run's; a status that cannot be read now counts as ours (a
        reader holding the file on Windows must not stop the log for good)."""
        if self.folder is None:
            return False
        status = read_status_file(self.folder, strict=True)
        if status is None:
            return False
        if status.get("state") in (UNREADABLE_STATE, MALFORMED_STATE):
            return True
        return status.get("run_id") == self.run_id

    def _attach(self, folder: Path, run_id: str) -> None:
        status = read_status_file(folder, strict=True)
        # Unreadable right now (a reader holds the file the engine has just written): attach on the marker's word;
        # ``_still_ours`` checks the run id again later.
        unreadable = status is not None and status.get("state") == UNREADABLE_STATE
        if not unreadable and (status is None or status.get("run_id") != run_id):
            _logger.warning("Run record marker for %s does not match its status file; not writing its log.", folder)
            return
        try:
            writer = RunLogWriter(folder / RUN_LOG_FILE, head=self._head, tail=self._tail, guard=self._still_ours)
        except OSError as exc:
            _logger.warning("Cannot open the run log in %s: %s", folder, exc)
            return
        self.folder, self.run_id, self._writer = folder, run_id, writer
        try:
            writer.write_omission(self._dropped_bytes)
            for line in self._buffer:
                writer.write_line(line)
        except OSError as exc:
            _logger.warning("Cannot write the run log in %s: %s", folder, exc)
            writer.close()
        self._buffer.clear()
        self._buffer_size = 0

    def feed(self, line: str) -> bool:
        """Take one output line; return whether it is shown (``False`` for the engine's marker)."""
        with self._lock:
            if self._closed:
                return True
            if not self._marker_seen:
                parsed = parse_marker(line)
                if parsed is not None:
                    self._marker_seen = True
                    self._attach(*parsed)
                    return False
                self._buffer_line(line)
                return True
            writer = self._writer
            if writer is None or writer.closed:
                return True
            self._lines_since_check += 1
            if self._lines_since_check >= OWNER_RECHECK_LINES:
                self._lines_since_check = 0
                if not self._still_ours():
                    writer.close()
                    return True
            try:
                writer.write_line(line)
            except OSError as exc:
                _logger.warning("Cannot write the run log in %s: %s", self.folder, exc)
                writer.close()
            return True

    def close(
        self,
        returncode: int | None,
        *,
        killed: bool = False,
        kill_reason: str | None = None,
        outcome: tuple[str, str] | None = None,
    ) -> Path | None:
        """Finish the relay when the engine has ended; return the run folder (``None`` when no marker came).

        Idempotent.  A status the engine left ``running`` is finalised from *returncode* (see
        :func:`finalize_unfinished`); *killed* marks a run the owner stopped on purpose, *outcome* a
        ``(state, reason)`` the owner knows better than the exit code (an interrupted run it had to kill).
        """
        with self._lock:
            if self._closed:
                return self.folder
            self._closed = True
            writer = self._writer
            if self.folder is not None and self.run_id is not None:
                try:
                    finalized = finalize_unfinished(
                        self.folder, self.run_id, returncode, killed=killed, kill_reason=kill_reason,
                        outcome=outcome,
                    )
                except OSError as exc:
                    _logger.warning("Cannot finish the run record in %s: %s", self.folder, exc)
                    finalized = False
                self.record_changed = finalized
                if finalized and writer is not None and not writer.closed:
                    status = read_status_file(self.folder) or {}
                    error = status.get("error") or {}
                    try:
                        writer.write_line(
                            f"# {self.launcher}: the run ended without finishing its record "
                            f"(exit code {returncode}): {status.get('state')}"
                            + (f", {error.get('message')}" if error.get("message") else "")
                        )
                    except OSError:
                        pass
            if writer is not None:
                writer.close()
            return self.folder


# ---------------------------------------------------------------------------
# _run_status.json
# ---------------------------------------------------------------------------


def _is_dict_or_none(value: Any) -> bool:
    return value is None or isinstance(value, dict)


def _is_list_of_dicts(value: Any) -> bool:
    return value is None or (isinstance(value, list) and all(isinstance(item, dict) for item in value))


def well_formed(status: Any) -> bool:
    """Whether *status* has the shape the readers of a status rely on (and is not a strict read's
    ``unreadable`` / ``malformed`` answer).

    ``state`` a text; ``input`` / ``output`` / ``versions`` / ``process`` / ``outputs`` / ``options_resolved`` /
    ``error`` objects (or absent / null); ``solves`` / ``warnings`` lists of objects (or absent).
    """
    if not isinstance(status, Mapping):
        return False
    state = status.get("state")
    if not isinstance(state, str) or state in (UNREADABLE_STATE, MALFORMED_STATE):
        return False
    if not all(_is_dict_or_none(status.get(key))
               for key in ("input", "output", "versions", "process", "outputs", "options_resolved", "error")):
        return False
    return _is_list_of_dicts(status.get("solves")) and _is_list_of_dicts(status.get("warnings"))


def _undecodable(path: Path) -> dict:
    """The strict answer for a status that does not parse: being written when it is new, else damaged."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return {"state": UNREADABLE_STATE}
    return {"state": MALFORMED_STATE if age > _WRITE_IN_PROGRESS_SECONDS else UNREADABLE_STATE}


def read_status_file(folder: Path | str, *, strict: bool = False) -> dict | None:
    """The status in *folder*; ``None`` when it is absent.

    A file being replaced (``JSONDecodeError`` / ``PermissionError`` on Windows) is read again up to 3 times.
    The lenient read (the default) also gives ``None`` for a file it cannot read or that is not a JSON object.
    With *strict*, such a file gives ``{"state": "unreadable"}`` (locked, or not parsing and younger than 10 s:
    probably being written; ask again) or ``{"state": "malformed"}`` (not parsing for longer, not an object, or
    an object of the wrong shape, see :func:`well_formed`).  Callers that decide whether results are usable, or
    which run a folder belongs to, read strictly.
    """
    path = Path(folder) / RUN_STATUS_FILE
    for attempt in range(4):
        last = attempt == 3
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            data = json.loads(text)
        except FileNotFoundError:
            return None
        except PermissionError:
            if last:
                return {"state": UNREADABLE_STATE} if strict else None
            time.sleep(0.05)
            continue
        except ValueError:  # JSONDecodeError, or bytes that are not UTF-8: cut short while it is written
            if last:
                return _undecodable(path) if strict else None
            time.sleep(0.05)
            continue
        except OSError:
            return {"state": UNREADABLE_STATE} if strict else None
        if not isinstance(data, dict):
            return {"state": MALFORMED_STATE} if strict else None
        if strict and not well_formed(data):
            return {"state": MALFORMED_STATE}
        return data
    return None


def read_status_settled(folder: Path | str, seconds: float | None = None) -> dict | None:
    """:func:`read_status_file` read strictly, and read again every 0.1 s for up to *seconds* (2 s by default)
    while it answers ``unreadable`` (a reader on Windows holding the file, or a write in progress); the last
    answer then stands.  For callers that must decide now and keep the decision: the pipe owner finishing a
    record, the comparison recording which run each member holds."""
    deadline = time.monotonic() + (_FINALIZE_READ_SECONDS if seconds is None else seconds)
    while True:
        status = read_status_file(folder, strict=True)
        if status is None or status.get("state") != UNREADABLE_STATE or time.monotonic() >= deadline:
            return status
        time.sleep(0.1)


def _replace(tmp: Path, path: Path, *, final: bool) -> None:
    """``os.replace`` with retries on ``PermissionError`` (a reader holds the target open on Windows)."""
    deadline = time.monotonic() + 2.0
    attempts = 0
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            attempts += 1
            if not final and attempts >= 5:
                raise
            if final and time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def write_status_atomic(folder: Path | str, data: Mapping[str, Any], *, final: bool = False) -> None:
    """Write *data* as ``_run_status.json`` in *folder* through a sibling temp file, so readers never see half.

    UTF-8, ``\\n`` line ends, non-ASCII kept as is.  A ``PermissionError`` on the replace is retried 5 times
    50 ms apart; for the *final* write of a run it is retried for 2 s and then the file is written in place.
    """
    path = Path(folder) / RUN_STATUS_FILE
    text = json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False)
    tmp = path.with_name(f"{RUN_STATUS_FILE}.{os.getpid()}{STATUS_TMP_SUFFIX}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    try:
        _replace(tmp, path, final=final)
    except PermissionError:
        if not final:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def solve_verdict(entry: Mapping[str, Any]) -> str | None:
    """The verdict of one solve entry of a status (``None`` while it runs or when nothing is known)."""
    state = entry.get("state")
    acceptance = entry.get("acceptance")
    if state == "running":
        return None
    if state == "error":
        return "error"
    if acceptance in ("optimal", "converged"):
        return "optimal"
    if acceptance in ("near_optimal", "not_converged", "no_incumbent"):
        return acceptance
    if acceptance == "rejected" or state == "rejected":
        status = entry.get("model_status")
        if status in _INFEASIBLE_STATUSES:
            return "infeasible"
        if status == "kUnbounded":
            return "unbounded"
        if status in _LIMIT_STATUSES:
            return "limit"
        if status in _ERROR_STATUSES:
            return "error"
        return "not_certified"
    return None


def worst_verdict(solves: Iterable[Mapping[str, Any]]) -> str | None:
    """The worst :func:`solve_verdict` of *solves* in :data:`VERDICT_ORDER` (``None`` when none finished)."""
    rank = {v: i for i, v in enumerate(VERDICT_ORDER)}
    worst: str | None = None
    for entry in solves:
        v = solve_verdict(entry)
        if v is not None and (worst is None or rank[v] < rank[worst]):
            worst = v
    return worst


# ---------------------------------------------------------------------------
# Exit codes and owner finalisation
# ---------------------------------------------------------------------------

_env_check: ModuleType | None = None


def _load_env_check() -> ModuleType:
    """``flextool/env_check.py`` loaded from its file (stdlib only), without importing ``flextool``.

    The same pattern as ``flextool/__init__.py``'s ``_ensure_polars_runtime``, under a private name of its own;
    the module is removed from :data:`sys.modules` after it runs (the module object is kept here).
    """
    global _env_check
    if _env_check is None:
        name = "flextool_run_supervisor._env_check"
        path = Path(__file__).resolve().parent.parent / "env_check.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look their module up while it executes
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        _env_check = module
    return _env_check


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def classify_exit(returncode: int | None, plat: str | None = None) -> tuple[str, str]:
    """``(state, reason)`` for an engine that ended with *returncode* without finishing its status.

    POSIX: SIGKILL is ``killed`` (often the out-of-memory killer), SIGTERM / SIGHUP / SIGINT ``interrupted``, a
    native fault (``env_check.is_native_fault``) ``crashed``, any other signal ``killed``.  Windows: Ctrl-C
    (``0xC000013A``) is ``interrupted``, an NTSTATUS fault ``crashed``.  Exit code 0 is ``succeeded`` (the record
    was not finished), any other code ``failed``.
    """
    plat = sys.platform if plat is None else plat
    if returncode is None:
        return "crashed", "no exit code"
    windows = plat.startswith("win")
    if windows and (returncode & 0xFFFFFFFF) == _WINDOWS_CTRL_C_EXIT:
        return "interrupted", "interrupted by Ctrl-C"
    if not windows and returncode == -9:
        return "killed", "terminated by SIGKILL (often the operating system's out-of-memory killer)"
    if not windows and returncode in (-15, -1, -2):
        return "interrupted", f"stopped by {_signal_name(-returncode)}"
    env_check = _load_env_check()
    if env_check.is_native_fault(returncode, plat):
        return "crashed", env_check.describe_fault(returncode, plat)
    if not windows and returncode < 0:
        return "killed", f"terminated by {_signal_name(-returncode)}"
    if returncode == 0:
        return "succeeded", "the engine exited 0 without finishing its run record"
    return "failed", f"exit code {returncode}"


_ERROR_TYPES = {
    "killed": "Killed",
    "interrupted": "Interrupted",
    "crashed": "Crashed",
    "succeeded": "RecordIncomplete",
    "failed": "ProcessEnded",
}


def finalize_unfinished(
    folder: Path | str,
    run_id: str,
    returncode: int | None,
    *,
    killed: bool = False,
    kill_reason: str | None = None,
    outcome: tuple[str, str] | None = None,
) -> bool:
    """Finish a status the engine left ``running``; return whether it was changed.

    Only a status that still says ``running`` with *run_id* is touched; one that cannot be read is read again for
    up to 2 s.  *killed* (the owner stopped the run) gives
    ``killed`` with *kill_reason*; *outcome* gives that ``(state, reason)`` (the supervisor killing a run that a
    signal had already asked to stop: ``interrupted``); otherwise :func:`classify_exit` decides.  A solve still
    running becomes ``error``.
    """
    folder = Path(folder)
    status = read_status_settled(folder)
    if status is not None and status.get("state") == UNREADABLE_STATE:
        _logger.warning("Cannot read the run record in %s to finish it; it keeps the state the engine left.",
                        folder)
        return False
    if status is None or status.get("state") != "running" or status.get("run_id") != run_id:
        return False
    if killed:
        state, reason = "killed", kill_reason or "stopped by the user"
    elif outcome is not None:
        state, reason = outcome
    else:
        state, reason = classify_exit(returncode)
    now = now_iso()
    status["state"] = state
    status["exit_code"] = returncode
    status["error"] = {"type": _ERROR_TYPES[state], "message": reason[:2000]}
    for entry in status.get("solves") or ():
        if isinstance(entry, dict) and entry.get("state") == "running":
            entry["state"] = "error"
    status["verdict"] = worst_verdict(e for e in status.get("solves") or () if isinstance(e, dict))
    status["progress"] = None
    status["finished_at"] = now
    status["updated_at"] = now
    started = parse_iso(status.get("started_at"))
    finished = parse_iso(now)
    if started is not None and finished is not None:
        status["wall_time_s"] = round((finished - started).total_seconds(), 3)
    write_status_atomic(folder, status, final=True)
    return True


# ---------------------------------------------------------------------------
# Whether a folder's results may be offered
# ---------------------------------------------------------------------------


def results_usable(status: Mapping[str, Any] | None) -> bool:
    """Whether the results in a run folder whose status is *status* may be offered (plots, comparison, assistant).

    ``None`` (a folder from before run records) counts as usable; the caller still checks for ``*.parquet``.
    Otherwise only a ``succeeded`` run that wrote its parquet (``outputs.parquet_state == "written"``) is usable.
    Any other state -- running, failed, interrupted, killed, crashed -- is not, whatever parquet of an earlier run
    the folder still holds; nor is a succeeded run that wrote no parquet, nor a record that could not be read or
    has the wrong shape (:func:`well_formed`).  Pure: no I/O.
    """
    if status is None:
        return True
    if not well_formed(status):
        return False
    outputs = status.get("outputs") or {}
    return status.get("state") == "succeeded" and outputs.get("parquet_state") == "written"


def unusable_reason(status: Mapping[str, Any] | None) -> str | None:
    """Why :func:`results_usable` says no, in plain words (``None`` when the results are usable).

    Reads the raw ``state``; a caller that shows a stale ``running`` as abandoned replaces it for display only.
    """
    if results_usable(status):
        return None
    assert status is not None
    if not well_formed(status):
        if isinstance(status, Mapping) and status.get("state") == UNREADABLE_STATE:
            return "the run record could not be read (it may be being written); try again in a moment"
        return "the run record is damaged; see the run log"
    state = status.get("state")
    error = status.get("error") or {}
    message = error.get("message") if isinstance(error, dict) else None
    outputs = status.get("outputs") or {}
    parquet = outputs.get("parquet_state")
    if state == "running":
        return "latest run is still running"
    if state == "failed":
        return f"latest run failed: {message}" if message else "latest run failed"
    if state in ("interrupted", "killed", "crashed"):
        verb = {"interrupted": "was interrupted", "killed": "was killed", "crashed": "crashed"}[state]
        return f"latest run {verb}: {message}" if message else f"latest run {verb}"
    if state == "succeeded":
        if parquet == "not_requested":
            return "the run wrote no parquet (parquet was not among its write methods)"
        if parquet == "skipped":
            reason = outputs.get("reason")
            return f"the run wrote no parquet ({reason})" if reason else "the run wrote no parquet"
        if parquet == "partial":
            return "the run's parquet output is incomplete"
        if parquet == "previous_run":
            return "the parquet files belong to an earlier run"
        return "the run wrote no parquet"
    return f"latest run state is {state!r}"


#: The values of :func:`unusable_code`.
UNUSABLE_CODES = ("failed", "running", "abandoned", "no_parquet", "previous_run", "partial", "unreadable",
                  "malformed")


def unusable_code(status: Mapping[str, Any] | None, effective: str | None = None) -> str | None:
    """Why :func:`results_usable` says no, as one of :data:`UNUSABLE_CODES` (``None`` when the results are usable).

    ``running`` -- the latest run has not finished; ``abandoned`` -- the same, when *effective* (the caller's
    ``effective_state`` of the status) says the recording process is gone; ``failed`` -- the latest run failed,
    was interrupted, killed or crashed; for a run that succeeded: ``previous_run`` -- the parquet files are an
    earlier run's, ``partial`` -- its parquet output is incomplete, ``no_parquet`` -- it wrote none;
    ``unreadable`` -- the record could not be read now (ask again), ``malformed`` -- it is damaged or has the wrong
    shape.  Pure: no I/O.
    """
    if results_usable(status):
        return None
    assert status is not None
    if not well_formed(status):
        if isinstance(status, Mapping) and status.get("state") == UNREADABLE_STATE:
            return "unreadable"
        return "malformed"
    state = status.get("state")
    if state == "running":
        return "abandoned" if effective == "abandoned" else "running"
    if state != "succeeded":
        return "failed"
    parquet = (status.get("outputs") or {}).get("parquet_state")
    if parquet in ("previous_run", "partial"):
        return parquet
    return "no_parquet"


__all__ = [
    "LOG_HEAD_BYTES",
    "LOG_TAIL_BYTES_DEFAULT",
    "MALFORMED_STATE",
    "MARKER",
    "PRE_MARKER_BUFFER_BYTES",
    "RUN_LOG_FILE",
    "RUN_LOG_MAX_BYTES_ENV",
    "RUN_LOG_OWNER_ENV",
    "RUN_RECORD_FILES",
    "RUN_STATUS_FILE",
    "RUN_STOP_FILE_ENV",
    "STATUS_SCHEMA",
    "STATUS_SCHEMA_VERSION",
    "UNREADABLE_STATE",
    "VERDICT_ORDER",
    "RunLogOwner",
    "RunLogWriter",
    "classify_exit",
    "finalize_unfinished",
    "format_marker",
    "log_header",
    "log_tail_bytes",
    "now_iso",
    "parse_iso",
    "parse_marker",
    "read_status_file",
    "read_status_settled",
    "read_stop_reason",
    "results_usable",
    "solve_verdict",
    "UNUSABLE_CODES",
    "unusable_code",
    "unusable_reason",
    "well_formed",
    "worst_verdict",
    "write_log_header",
    "write_status_atomic",
    "write_stop_request",
]
