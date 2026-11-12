"""``flextool-run``: run one FlexTool engine run as a child process and write its ``_run.log``.

The engine (``python -m flextool.cli.cmd_run_flextool``) runs in a child process whose stdout and stderr come to
this process through one pipe.  This process is the run's *pipe owner*: it echoes the child's output to the
console and writes it into the run's ``_run.log`` (``output_parquet/<subdir>/``), including what only the pipe
sees -- the native solver's lines, a ``--save-memory`` solve child's output, a crash dump.  When the engine dies
without finishing its ``_run_status.json`` (a crash, a kill, a signal), this process finishes it from the exit
code.  See ``flextool/run_record/`` and ``docs/dev/architecture.md`` "Run record".

This module uses the standard library only and never imports ``flextool``: ``import flextool`` imports polars,
pandas and spinedb_api (about a second and 180 MB), which this process would hold for the whole run.  It loads
two stdlib-only files of the installed package by path instead (:func:`_load_file`, the one by-path loader):
``flextool/run_record/_owner.py`` (the log writer, the status helpers, the signal names and exit codes) and,
when it is first needed, ``flextool/env_check.py`` (the compatible-polars check, asked here, in the process that
has the terminal, before the child starts; ``_owner`` classifies exits with the same module object).

The engine's command line passes through unchanged (``--threads`` too: the child reads it from its own command
line), plus ``--run-launcher flextool-run`` when it names no launcher.  The child's environment is this
process's, plus three variables that describe the pipe (``FLEXTOOL_RUN_LOG_OWNER=1``, ``PYTHONIOENCODING=utf-8``,
``PYTHONUNBUFFERED=1``); ``os.environ`` itself is not changed for them.  ``FLEXTOOL_RUN_STOP_FILE`` is left out
of it.

Stop file.  When ``FLEXTOOL_RUN_STOP_FILE`` names a file (the GUI sets it for an auxiliary job's process, and this
process inherits it), its appearance asks for a stop: before the child starts, nothing starts (exit 1, no run
record); while it runs, the child is killed (its process group on POSIX) and the run is recorded as ``killed`` with
the file's text as the reason (exit 1).  The file is looked for every 0.25 s.

Signals.  POSIX: the child runs in its own session, so a Ctrl-C at the terminal reaches only this process, which
forwards SIGINT / SIGTERM / SIGHUP to the child once.  A second SIGINT, or 10 s after SIGTERM / SIGHUP without
the child ending, kills the child's process group (SIGKILL) and the run is recorded as ``interrupted``.  Windows:
the child shares the console and gets the Ctrl-C itself; this process keeps reading its output, and a second
Ctrl-C kills the child.  This process exits like the child: by the same signal on POSIX, with the child's exit
code otherwise (a Windows NTSTATUS code as the signed 32-bit number).
"""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import IO, Callable, Mapping, Sequence

#: The engine module the supervisor runs.
ENGINE_MODULE = "flextool.cli.cmd_run_flextool"
#: The interpreter flag of the engine child: the current folder stays off ``sys.path``, so a ``flextool/`` folder in
#: the folder ``flextool-run`` is started from is never imported instead of FlexTool (``flextool/_child_python.py``,
#: which this stdlib-only module cannot import).
SAFE_PATH_FLAG = "-P"
#: The launcher label this supervisor records (``--run-launcher``) and writes in the log.
LAUNCHER = "flextool-run"
#: After SIGTERM / SIGHUP, the child gets this many seconds to end before it is killed.
TERMINATE_GRACE_S = 10.0
#: After the child has ended, its output pipe is read for at most this long (a process it started may hold it).
DRAIN_TIMEOUT_S = 5.0
#: The stop file is looked for every this many 0.05 s ticks of the main loop.
STOP_FILE_CHECK_TICKS = 5

_PRIVATE_NAME = "flextool_run_supervisor"
_owner_module: ModuleType | None = None


# ---------------------------------------------------------------------------
# The package files loaded by path
# ---------------------------------------------------------------------------


def _flextool_dir() -> Path:
    """The installed ``flextool`` package folder, found without importing the package."""
    spec = importlib.util.find_spec("flextool")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        raise ImportError("FlexTool is not installed in this Python environment (no 'flextool' package).")
    return Path(locations[0])


def _load_file(name: str, path: Path) -> ModuleType:
    """Execute the stdlib-only module at *path* under the private *name*, without importing its package.

    The pattern of ``flextool/__init__.py``'s ``_ensure_polars_runtime``: the module is in :data:`sys.modules`
    only while it executes (dataclasses look their module up then), so nothing else can import it by that name.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def load_owner() -> ModuleType:
    """``flextool/run_record/_owner.py`` (the run record's stdlib-only side), loaded once by path."""
    global _owner_module
    if _owner_module is None:
        flextool_dir = _flextool_dir()
        owner = _load_file(f"{_PRIVATE_NAME}._owner", flextool_dir / "run_record" / "_owner.py")
        owner.set_env_check_loader(
            lambda: _load_file(f"{_PRIVATE_NAME}._env_check", flextool_dir / "env_check.py")
        )
        _owner_module = owner
    return _owner_module


def load_env_check() -> ModuleType:
    """``flextool/env_check.py``, loaded by path (the same module object ``_owner`` classifies exits with)."""
    return load_owner()._load_env_check()


def _signal_name(signum: int | None) -> str:
    return load_owner().signal_name(signum)


# ---------------------------------------------------------------------------
# The child's command line and environment
# ---------------------------------------------------------------------------


def with_launcher(argv: Sequence[str], launcher: str = LAUNCHER) -> list[str]:
    """*argv* plus ``--run-launcher <launcher>`` unless it already names one (the outermost launcher wins)."""
    args = list(argv)
    if any(a == "--run-launcher" or a.startswith("--run-launcher=") for a in args):
        return args
    return [*args, "--run-launcher", launcher]


def engine_command() -> list[str]:
    """The engine child's interpreter and module: ``[sys.executable, "-P", "-m", ENGINE_MODULE]``."""
    return [sys.executable, SAFE_PATH_FLAG, "-m", ENGINE_MODULE]


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of *environ* (default ``os.environ``) plus the pipe owner's three variables and ``PYTHONSAFEPATH=1``,
    without the stop file's variable (it is this process's, not the engine's)."""
    owner = load_owner()
    env = dict(os.environ if environ is None else environ)
    env.pop(owner.RUN_STOP_FILE_ENV, None)
    env[owner.RUN_LOG_OWNER_ENV] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    return env


def _command_text(cmd: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(cmd))
    import shlex

    return shlex.join(cmd)


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


class _Echo:
    """Writes the child's visible lines to this process's stdout in the console's encoding.

    A failed write (closed or broken stdout) stops the echo for good; the log is written on.  Without a
    ``sys.stdout`` (``pythonw``) nothing is echoed.
    """

    def __init__(self, stream: IO[str] | None) -> None:
        self._stream = stream
        self.enabled = stream is not None

    def write(self, text: str) -> None:
        if not self.enabled:
            return
        stream = self._stream
        try:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                buffer.write(text.encode(getattr(stream, "encoding", None) or "utf-8", errors="replace"))
                buffer.flush()
            else:
                stream.write(text)
                stream.flush()
        except (OSError, ValueError):
            self.enabled = False
            if sys.stdout is stream:
                # Nothing more goes there; without this, the flush at exit would report the broken pipe.
                sys.stdout = None


def _relay(pipe: IO[bytes], owner, echo: _Echo) -> None:
    """Read the child's output to its end: every line to the owner, the visible ones to the console."""
    for raw in iter(pipe.readline, b""):
        text = raw.decode("utf-8", errors="replace")
        body = text.rstrip("\r\n")
        if owner.feed(body):
            echo.write(text)
    pipe.close()


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class _Stopper:
    """What a stop request (Ctrl-C, SIGTERM, SIGHUP) does to the child; see the module docstring.

    Installed before the child starts: a request that comes before :meth:`attach` is passed on as soon as the
    child exists.
    """

    def __init__(self, *, windows: bool) -> None:
        self._proc: subprocess.Popen | None = None
        self._windows = windows
        self.first: int | None = None
        self._deadline: float | None = None
        #: ``(state, reason)`` for the record when this process had to kill the child.
        self.outcome: tuple[str, str] | None = None
        #: Whether the child was stopped through the stop file (this process then exits 1).
        self.stopped_by_launcher = False
        #: A kill asked for before the child existed: ``(reason, state)``, done by :meth:`attach`.
        self._pending_kill: tuple[str, str] | None = None

    def attach(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        if self._pending_kill is not None:
            self._kill(*self._pending_kill)
        elif self.first is not None and not self._windows:
            self._send(self.first)

    def handle(self, signum: int, frame=None) -> None:
        if self.first is None:
            self.first = signum
            if not self._windows:
                # The child is in its own session: a terminal signal reached only this process.
                self._send(signum)
                if signum != signal.SIGINT:
                    self._deadline = time.monotonic() + TERMINATE_GRACE_S
            return
        if signum == signal.SIGINT and self.outcome is None:
            self._kill(f"second {_signal_name(signum)}: the run was killed")

    def tick(self) -> None:
        """Kill the child once the grace time after SIGTERM / SIGHUP is over."""
        if self._deadline is not None and self.outcome is None and time.monotonic() >= self._deadline:
            self._kill(
                f"{_signal_name(self.first)}: the run did not end within {TERMINATE_GRACE_S:g} s and was killed"
            )

    def _send(self, signum: int) -> None:
        if self._proc is None:
            return
        try:
            self._proc.send_signal(signum)
        except OSError:
            pass

    def stop(self, reason: str) -> None:
        """The launcher's stop file appeared: kill the child and record the run as ``killed`` with *reason*."""
        if self.outcome is not None:
            return
        if self._kill(reason, state="killed"):
            self.stopped_by_launcher = True

    def _kill(self, reason: str, state: str = "interrupted") -> bool:
        """Kill a child that is still running and set :attr:`outcome`; return whether it was running.

        A child that has already ended keeps its own exit code: no outcome is set, so the record and this
        process's exit status come from that code.  The outcome is set before the kill, so the main loop never
        sees the killed child's exit without it.
        """
        proc = self._proc
        if proc is None:
            self._pending_kill = (reason, state)
            return False
        if proc.poll() is not None:
            return False
        self.outcome = (state, reason)
        if not self._windows:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return True
            except OSError:
                pass
        try:
            proc.kill()
        except OSError:
            pass
        return True

    def exit_status(self, returncode: int | None) -> int:
        """The status this process ends with: the child's, or the stop signal when this process killed the child."""
        if self.outcome is None:
            return 1 if returncode is None else returncode
        if self.stopped_by_launcher:
            return 1
        if self._windows:
            return load_owner().WINDOWS_CTRL_C_EXIT_SIGNED
        return -(self.first or signal.SIGINT)


def _install_handlers(handler: Callable[[int, object], None], *, windows: bool) -> Callable[[], None]:
    """Install *handler* for the stop signals (main thread only); return the function that puts the old ones back."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    signums = [signal.SIGINT] if windows else [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
    previous = {}
    for signum in signums:
        try:
            previous[signum] = signal.signal(signum, handler)
        except (ValueError, OSError):
            continue

    def restore() -> None:
        for signum, old in previous.items():
            try:
                signal.signal(signum, old if old is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass

    return restore


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_supervised(
    argv: Sequence[str],
    *,
    child_cmd: Sequence[str] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Run the engine with *argv* as this process's child and own its output; return the status to exit with.

    *child_cmd* replaces ``[sys.executable, "-P", "-m", "flextool.cli.cmd_run_flextool"]`` (tests run a stub engine);
    *stdout* replaces ``sys.stdout`` as the echo target.  The result is the child's exit code (negative: the
    signal that ended it, POSIX), or the stop signal when this process had to kill the child after one.
    """
    owner_mod = load_owner()
    windows = os.name == "nt"
    stop_file = os.environ.get(owner_mod.RUN_STOP_FILE_ENV) or None
    if stop_file is not None and owner_mod.read_stop_reason(stop_file) is not None:
        print(f"{LAUNCHER}: stopped before the engine started", file=sys.stderr, flush=True)
        return 1
    cmd = [*(child_cmd if child_cmd is not None else engine_command()), *with_launcher(argv)]
    owner = owner_mod.RunLogOwner(launcher=LAUNCHER, preamble=[f"# {LAUNCHER}: {_command_text(cmd)}"])
    popen_kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "env": child_environment()}
    if not windows:
        popen_kwargs["start_new_session"] = True
    stopper = _Stopper(windows=windows)
    restore = _install_handlers(stopper.handle, windows=windows)
    proc: subprocess.Popen | None = None
    returncode: int | None = None
    try:
        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except OSError as exc:
            print(f"{LAUNCHER}: cannot start the FlexTool engine: {exc}", file=sys.stderr, flush=True)
            return 1
        stopper.attach(proc)
        echo = _Echo(sys.stdout if stdout is None else stdout)
        relay = threading.Thread(
            target=_relay, args=(proc.stdout, owner, echo), name="flextool-run relay", daemon=True,
        )
        relay.start()
        ticks = 0
        while proc.poll() is None:
            # A short sleep, not a blocking wait: the signal handlers run between sleeps (also on Windows).
            time.sleep(0.05)
            stopper.tick()
            ticks += 1
            if stop_file is not None and ticks % STOP_FILE_CHECK_TICKS == 0:
                reason = owner_mod.read_stop_reason(stop_file)
                if reason is not None:
                    stopper.stop(reason)
        returncode = proc.wait()
        if stop_file is not None and stopper.outcome is None:
            # The launcher's fallback may have killed the engine before this process saw the file.
            reason = owner_mod.read_stop_reason(stop_file)
            if reason is not None:
                stopper.stopped_by_launcher = True
                stopper.outcome = ("killed", reason)
        relay.join(DRAIN_TIMEOUT_S)
        if relay.is_alive() and not windows:
            # A process the engine started (a --save-memory solve) outlived it and holds the pipe.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            relay.join(DRAIN_TIMEOUT_S)
    finally:
        restore()
        if proc is not None and returncode is None:
            # Left by an exception: do not leave the engine running without its owner.
            stopper._kill("the flextool-run supervisor failed")
            returncode = proc.wait()
        owner.close(returncode, outcome=stopper.outcome)
    return stopper.exit_status(returncode)


def exit_like(status: int) -> None:
    """End this process with *status* as :func:`run_supervised` returns it."""
    if status < 0 and os.name != "nt":
        signum = -status
        try:
            sys.stdout and sys.stdout.flush()
        except (OSError, ValueError):
            pass
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError):
            pass  # SIGKILL and SIGSTOP have no handler to reset
        try:
            os.kill(os.getpid(), signum)
        except OSError:
            pass
        # A signal whose default action does not end the process: end with the shell's code for it.
        sys.exit(128 + signum)
    if os.name == "nt" and status >= 2**31:
        status -= 2**32
    sys.exit(status)


def main(argv: Sequence[str] | None = None, *, child_cmd: Sequence[str] | None = None) -> None:
    """The ``flextool-run`` console script: check the polars runtime, run the engine, exit like it."""
    args = list(sys.argv[1:] if argv is None else argv)
    env_check = load_env_check()
    try:
        # Asked here, with the terminal: behind the pipe the child could not ask.
        env_check.ensure_polars_runtime_before_import(entry=env_check.ENTRY_ASK)
    except env_check.CompatRuntimeRequired as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(1)
    exit_like(run_supervised(args, child_cmd=child_cmd))


if __name__ == "__main__":
    main()
