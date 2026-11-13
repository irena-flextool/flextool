"""Stopping a process together with the processes it started (stdlib + psutil; no GUI imports).

Used by the FlexTool window (a job's engine run, a calibrator's runs) and by the commercial solver spawn
(``engine/solve/_subprocess_solve.py``: the solver, and whatever its own program started, on a watchdog stop, an
interrupt or an orphan that keeps the log pipe open).  The solver is not started in a process group or session of
its own: it stays in FlexTool's, so whatever ends FlexTool's group (a closed terminal) ends the solver too.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Iterable

import psutil


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill *proc* and every process it started (a ``--save-memory`` solve child, a calibrator's engine run).

    Killing only *proc* would leave such a child running and holding the output pipe, so the job's reader would
    wait for it.  The children are listed before *proc* dies (afterwards they are no longer its children).
    """
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    try:
        proc.kill()
    except OSError:
        pass
    kill_processes(children)


def kill_processes(processes: Iterable[psutil.Process]) -> None:
    """Kill each of *processes* that is still running (a gone or foreign one is skipped)."""
    for process in processes:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def pipe_inode(fileobj: object) -> int | None:
    """The inode of the pipe behind *fileobj* (Linux), to find the processes that hold its other end open later
    (:func:`processes_holding_pipes`); ``None`` elsewhere or when it is not a pipe."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        st = os.fstat(fileobj.fileno())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return None
    return st.st_ino if stat.S_ISFIFO(st.st_mode) else None


def processes_holding_pipes(inodes: Iterable[int | None]) -> list[psutil.Process]:
    """The processes other than this one that have one of the pipes *inodes* open (Linux ``/proc``; empty
    elsewhere).

    It finds a process the solver started that still holds the solver's output open after the solver exited:
    re-parented, it is no longer the solver's child, so only its open pipe ties it to the solve.  Processes of
    other users cannot be read and are skipped.
    """
    targets = {f"pipe:[{ino}]" for ino in inodes if ino is not None}
    if not targets or not sys.platform.startswith("linux"):
        return []
    me = os.getpid()
    found: list[psutil.Process] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit() or int(entry) == me:
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if link in targets:
                try:
                    found.append(psutil.Process(int(entry)))
                except psutil.NoSuchProcess:
                    pass
                break
    return found


def kill_solver_tree(proc: subprocess.Popen, known: Iterable[psutil.Process] = ()) -> None:
    """Kill *proc*, the processes it started and every process in *known* (a snapshot of its children, and the
    processes that hold its output open).

    *proc* stays in FlexTool's own process group, so a signal to that group (a closed terminal, a dropped SSH
    session, ``kill -- -<pgid>``) also reaches it.  Its children are listed while it runs (POSIX: psutil;
    Windows: ``taskkill /F /T`` first), and *known* covers those re-parented after their parent exited.
    """
    if os.name != "posix" and proc.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    current: list[psutil.Process] = []
    if proc.poll() is None:
        try:
            current = psutil.Process(proc.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            current = []
    try:
        proc.kill()
    except OSError:
        pass
    kill_processes(current)
    kill_processes(known)


__all__ = ["kill_processes", "kill_process_tree", "kill_solver_tree", "pipe_inode", "processes_holding_pipes"]
