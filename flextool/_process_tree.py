"""Stopping a process together with the processes it started (stdlib + psutil; no GUI imports).

Used by the FlexTool window (a job's engine run, a calibrator's runs) and by the commercial solver spawn
(``engine/solve/_subprocess_solve.py``: the solver, and whatever its own program started, on a watchdog stop, an
interrupt or an orphan that keeps the log pipe open).
"""
from __future__ import annotations

import os
import signal
import subprocess
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


def own_group_popen_kwargs() -> dict[str, object]:
    """``Popen`` keywords that start the child as the leader of its own process group on POSIX.

    :func:`kill_process_group` then reaches every process of that group, also one whose parent has exited and
    which was re-parented (psutil can no longer list it as a child).  The child no longer gets the terminal's
    Ctrl-C itself; the caller stops it when it is interrupted.  On Windows nothing changes: its tree is stopped
    with ``taskkill /T`` while the parent lives, and from a snapshot of its children afterwards.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def kill_process_group(proc: subprocess.Popen, known: Iterable[psutil.Process] = ()) -> None:
    """Kill *proc*, the processes it started and every process in *known* (a snapshot of its children).

    POSIX: the process group *proc* leads (:func:`own_group_popen_kwargs`), which also holds re-parented
    descendants.  Windows: ``taskkill /F /T`` on *proc*'s tree while *proc* runs.  Then, on both, the processes of
    *known* still alive and *proc* itself.
    """
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    elif proc.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        try:
            current = psutil.Process(proc.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            current = []
        kill_processes(current)
    kill_processes(known)
    try:
        proc.kill()
    except OSError:
        pass


__all__ = ["kill_process_group", "kill_process_tree", "kill_processes", "own_group_popen_kwargs"]
