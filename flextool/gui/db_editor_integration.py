"""Manages Spine DB Editor instances and detects potential uncommitted changes."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class DbEditorManager:
    """Manages communication with Spine DB Editor instances.

    Tracks launched editor processes so the GUI can:
    - Warn when a user tries to execute scenarios whose source database
      is still open in an editor.
    - Open additional databases by launching new editor instances.

    Spine DB Editor (``spine-db-editor``) accepts multiple database URLs
    on the command line and opens them as tabs (use ``-s`` for separate
    tabs).  However, there is no IPC mechanism to add a tab to an
    already-running instance, so each call to :meth:`open_database`
    launches a new process.
    """

    def __init__(self) -> None:
        # source_name -> list of Popen objects (a source can be opened multiple times)
        self._processes: dict[str, list[subprocess.Popen]] = {}
        # Captured stdout/stderr per launched process, so an immediate crash
        # (e.g. an incomplete Spine Toolbox install) can be shown in the GUI
        # instead of only printing a traceback to the launching terminal.
        self._launch_logs: dict[subprocess.Popen, Path] = {}

    # ── Opening databases ─────────────────────────────────────────

    def open_database(self, db_url: str, source_name: str) -> subprocess.Popen | None:
        """Launch ``spine-db-editor`` for *db_url* and track the process.

        The editor's own stdout/stderr is redirected to a temporary log file
        so that, if it exits immediately, the caller can read the failure via
        :meth:`read_launch_output` and surface it to the user.

        Args:
            db_url: SQLAlchemy-style database URL (e.g. ``sqlite:///path``).
            source_name: Logical name used to track which input source is
                being edited.

        Returns:
            The :class:`subprocess.Popen` object, or ``None`` if the
            editor executable is not found on ``PATH``.
        """
        exe = shutil.which("spine-db-editor")
        if exe is None:
            logger.warning("spine-db-editor not found on PATH")
            return None

        log_path: Path | None = None
        log_handle = None
        try:
            fd, name = tempfile.mkstemp(prefix="flextool_db_editor_", suffix=".log")
            log_path = Path(name)
            log_handle = os.fdopen(fd, "wb")
        except OSError:
            logger.debug("Could not create db-editor launch log", exc_info=True)
            log_path = None

        try:
            if log_handle is not None:
                proc = subprocess.Popen(
                    [exe, db_url], stdout=log_handle, stderr=subprocess.STDOUT
                )
            else:
                proc = subprocess.Popen([exe, db_url])
        except OSError:
            logger.warning("Failed to launch spine-db-editor", exc_info=True)
            if log_path is not None:
                try:
                    log_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None
        finally:
            # The child inherits its own dup of the fd; close the parent copy.
            if log_handle is not None:
                log_handle.close()

        if log_path is not None:
            self._launch_logs[proc] = log_path
        self._processes.setdefault(source_name, []).append(proc)
        self._reap_dead(source_name)
        return proc

    def read_launch_output(self, proc: subprocess.Popen) -> str:
        """Return (and consume) the captured launch output for *proc*.

        Used by the GUI when an editor process died right after launch, to
        show the user why.  The temporary log is removed once read.
        """
        log_path = self._launch_logs.pop(proc, None)
        if log_path is None:
            return ""
        text = ""
        try:
            if log_path.exists():
                text = log_path.read_text(errors="replace")
        except OSError:
            logger.debug("Could not read db-editor launch log", exc_info=True)
        finally:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass
        return text

    # ── Status queries ────────────────────────────────────────────

    def is_editor_running(self, source_name: str) -> bool:
        """Return ``True`` if at least one editor process is alive for *source_name*."""
        self._reap_dead(source_name)
        procs = self._processes.get(source_name, [])
        return any(p.poll() is None for p in procs)

    def has_uncommitted_changes(self, db_path: Path) -> bool:
        """Heuristic check for potential uncommitted changes.

        The Spine DB Editor keeps uncommitted edits in an in-memory
        SQLAlchemy session, so there is **no reliable** way to detect
        them from outside the process.  This method uses two heuristics:

        1. **Process tracking** -- if we launched an editor for the
           database and it is still running, we conservatively assume
           there *may* be uncommitted changes.
        2. **SQLite artefact files** -- the presence of a non-empty
           ``-journal`` or ``-wal`` file alongside the database can
           indicate an open write transaction (though these can also
           linger after a crash).

        Both checks are combined: if either signals a potential issue,
        ``True`` is returned.
        """
        source_name = db_path.name

        # Check 1: Is an editor process we launched still running?
        if self.is_editor_running(source_name):
            return True

        # Check 2: SQLite journal / WAL artefacts
        journal = db_path.parent / (db_path.name + "-journal")
        wal = db_path.parent / (db_path.name + "-wal")
        for artefact in (journal, wal):
            try:
                if artefact.exists() and artefact.stat().st_size > 0:
                    return True
            except OSError:
                pass

        return False

    # ── Cleanup ───────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop tracking all processes (without killing them)."""
        self._processes.clear()
        for log_path in self._launch_logs.values():
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._launch_logs.clear()

    # ── Internals ─────────────────────────────────────────────────

    def _reap_dead(self, source_name: str) -> None:
        """Remove finished processes from the tracking list.

        A process that exited cleanly has its launch log discarded here; a
        non-zero exit keeps its log so :meth:`read_launch_output` can still
        surface the failure to the user.
        """
        procs = self._processes.get(source_name)
        if procs is None:
            return
        alive: list[subprocess.Popen] = []
        for p in procs:
            if p.poll() is None:
                alive.append(p)
            elif p.returncode == 0:
                self._drop_log(p)
        self._processes[source_name] = alive

    def _drop_log(self, proc: subprocess.Popen) -> None:
        """Discard the captured launch log for *proc*, if any."""
        log_path = self._launch_logs.pop(proc, None)
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass
