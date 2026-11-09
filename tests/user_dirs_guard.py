"""No test reads or writes the real user folders of FlexTool or the system keyring.

The root ``conftest.py`` calls :func:`redirect` when it is loaded, before any test module is imported: every
pytest process (an xdist worker too) gets its own ``FLEXTOOL_USER_HOME`` in a fresh temp folder, and the
keyring a started process finds is keyring's null backend.  Both are plain environment variables, so they hold
for the processes a test starts and for threads that outlive a test (a monkeypatch is undone at teardown, while a
worker thread may still write).  Tests that want a folder of their own set ``FLEXTOOL_USER_HOME`` with
``monkeypatch.setenv`` (``tests/assistant/conftest.py``).

The keyring (:func:`forbid_system_keyring`): in every pytest process the default keyring is a stand-in that finds
nothing and records any store or removal, and every call of a system keyring backend class is recorded and raises —
also one made by a thread that outlived its test, or after a test undid its monkeypatch.  A process a test starts
gets the null backend through the environment, also when the test built that environment itself
(:func:`patch_subprocess_env`).  A recorded call fails the session.

The guard: the process that runs the session (the xdist controller, or the only process) takes a
:func:`snapshot` of the real folders when it starts and :func:`changes` compares it at the end; a file added,
changed or removed fails the session.  A FlexTool window or terminal used during the run changes them too; the
one change such a window makes by itself, rewriting its runtime context file (``runtime/<key>.<pid>.json``), is
left out while that process still runs at the end of the session (a process a test started has ended by then).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

USER_HOME_ENV = "FLEXTOOL_USER_HOME"
KEYRING_ENV = "PYTHON_KEYRING_BACKEND"
NULL_KEYRING = "keyring.backends.null.Keyring"
#: The file every pytest process of a session appends a keyring violation to (set by the process that runs the
#: session, inherited by the xdist workers and the processes the tests start).
KEYRING_LOG_ENV = "FLEXTOOL_TEST_KEYRING_LOG"
#: The keyring backends of the computer (and of keyrings.alt): any call into one fails the session.
_SYSTEM_BACKEND_MODULES = ("keyring.backends.SecretService", "keyring.backends.libsecret", "keyring.backends.kwallet",
                           "keyring.backends.macOS", "keyring.backends.Windows", "keyrings.alt.file",
                           "keyrings.alt.Gnome", "keyrings.alt.Google", "keyrings.alt.pyfs", "keyrings.alt.Windows")
_BACKEND_METHODS = ("get_password", "set_password", "delete_password", "get_credential")

Snapshot = dict[str, tuple[int, int]]

_RUNTIME_FILE = re.compile(r"^[0-9a-f]+\.(\d+)\.json$")


def real_user_dirs() -> tuple[Path, ...]:
    """The platform's user config, cache, data and state folders of FlexTool (``platformdirs`` does not read
    ``FLEXTOOL_USER_HOME``)."""
    import platformdirs

    return tuple(dict.fromkeys(Path(p) for p in (
        platformdirs.user_config_dir("flextool", appauthor=False),
        platformdirs.user_config_dir("flextool"),
        platformdirs.user_cache_dir("flextool", appauthor=False),
        platformdirs.user_cache_dir("flextool"),
        platformdirs.user_data_dir("flextool"),
        platformdirs.user_state_dir("flextool"),
    )))


def redirect() -> Path:
    """Point ``FLEXTOOL_USER_HOME`` at a fresh temp folder of this process and the keyring at the null backend;
    returns the folder."""
    home = Path(tempfile.mkdtemp(prefix="flextool-test-user-"))
    os.environ[USER_HOME_ENV] = str(home)
    os.environ[KEYRING_ENV] = NULL_KEYRING
    return home


def remove(home: Path) -> None:
    shutil.rmtree(home, ignore_errors=True)


# --- the keyring ---------------------------------------------------------------------------------------------------
class RealKeyringTouched(Exception):
    """A test reached a keyring backend of the computer (or wrote to this process's stand-in for it)."""


#: The session's keyring log as this process found it (a test that clears ``os.environ`` does not lose it).
_LOG_PATH: str | None = None


def _record(line: str) -> None:
    path = os.environ.get(KEYRING_LOG_ENV) or _LOG_PATH
    test = os.environ.get("PYTEST_CURRENT_TEST", "outside a test")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as log:
                log.write(f"{line} (pid {os.getpid()}, {test})\n")
        except OSError:
            pass


def _keyring_backend_class() -> type:
    from keyring.backend import KeyringBackend

    class StandInKeyring(KeyringBackend):
        """This process's default keyring in a test run, in place of the system keyring: a read finds nothing (as
        the null backend), a store or a removal is recorded (the session fails) and raises.  A test that wants a
        keyring installs its own (``tests/assistant/conftest.py``); one reaching this backend reached the keyring
        every other process of the computer uses."""
        priority = 1

        def get_password(self, service: str, username: str) -> str | None:
            return None

        def set_password(self, service: str, username: str, password: str) -> None:
            _record(f"store into the default keyring: {service} / {username}")
            raise RealKeyringTouched(f"a test stored {service} / {username} in the default keyring")

        def delete_password(self, service: str, username: str) -> None:
            _record(f"removal from the default keyring: {service} / {username}")
            raise RealKeyringTouched(f"a test removed {service} / {username} from the default keyring")

    StandInKeyring.__module__ = __name__
    return StandInKeyring


def _refuse(cls: type, method: str) -> None:
    def refused(self: object, service: str, username: str | None = None, *args: object, **kwargs: object) -> None:
        _record(f"{method} on the system keyring {cls.__module__}.{cls.__name__}: {service} / {username}")
        raise RealKeyringTouched(f"a test called {method} on the system keyring ({cls.__module__}.{cls.__name__})")

    refused.__name__ = method
    setattr(cls, method, refused)


def keyring_log_dir() -> Path | None:
    """A fresh folder for the session's keyring log when this process is the first of the session (no log path
    inherited); ``None`` otherwise."""
    if KEYRING_LOG_ENV in os.environ:
        return None
    return Path(tempfile.mkdtemp(prefix="flextool-test-keyring-"))


def forbid_system_keyring(log_dir: Path | None) -> None:
    """Keep this process away from the computer's keyring: the default keyring becomes a stand-in
    (:func:`_keyring_backend_class`), and every call of a system backend class (Secret Service, KWallet, macOS
    Keychain, Windows Credential Locker, keyrings.alt) is recorded and raises, also one made by a thread that
    outlived its test or after a test undid its monkeypatch.  The record goes to the session's log
    (:data:`KEYRING_LOG_ENV`, in *log_dir* when this process is the first of the session); the process that
    runs the session fails it when the log is not empty (:func:`keyring_violations`).  Processes a test starts get
    keyring's null backend through the environment (:func:`redirect`, :func:`force_null_keyring`)."""
    import importlib

    import keyring.core

    global _LOG_PATH
    if log_dir is not None:
        os.environ[KEYRING_LOG_ENV] = str(log_dir / "keyring_violations.log")
    _LOG_PATH = os.environ.get(KEYRING_LOG_ENV)
    for name in _SYSTEM_BACKEND_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:                   # noqa: BLE001  (a backend of another platform, or not installed)
            continue
        for value in vars(module).values():
            if isinstance(value, type) and value.__module__ == name and \
                    any(hasattr(value, m) for m in _BACKEND_METHODS):
                for method in _BACKEND_METHODS:
                    if method in vars(value):
                        _refuse(value, method)
    keyring.core.set_keyring(_keyring_backend_class()())


def keyring_violations() -> list[str]:
    """What the processes of this session recorded about the system keyring (:func:`forbid_system_keyring`)."""
    try:
        return Path(_LOG_PATH).read_text(encoding="utf-8").splitlines() if _LOG_PATH else []
    except OSError:
        return []


def force_null_keyring(env: dict[str, str] | None) -> dict[str, str] | None:
    """*env* of a process a test starts, with keyring's null backend unless the test chose a backend of its own (a
    test module's): an environment built without ``os.environ`` would otherwise give the child the system
    keyring.  ``None`` (the child inherits this process's environment, where :func:`redirect` set it) is kept."""
    if env is None:
        return None
    chosen = env.get(KEYRING_ENV, "")
    own = bool(chosen) and not chosen.startswith(("keyring.", "keyrings."))
    if own or chosen in (NULL_KEYRING, "keyring.backends.fail.Keyring"):
        return env
    return {**env, KEYRING_ENV: NULL_KEYRING}


def patch_subprocess_env() -> None:
    """Every ``subprocess.Popen`` of this process (``subprocess.run`` and ``check_output`` use it) starts its child
    with :func:`force_null_keyring` applied to an explicit ``env``."""
    import subprocess

    if getattr(subprocess.Popen.__init__, "_keyring_guard", False):
        return
    original = subprocess.Popen.__init__

    def init(self: subprocess.Popen, *args: object, **kwargs: object) -> None:
        if len(args) > 10:                  # env given by position (the 11th: args, bufsize, ..., cwd, env)
            args = (*args[:10], force_null_keyring(args[10]), *args[11:])  # type: ignore[arg-type]
        elif "env" in kwargs:
            kwargs["env"] = force_null_keyring(kwargs["env"])  # type: ignore[arg-type]
        original(self, *args, **kwargs)

    init._keyring_guard = True  # type: ignore[attr-defined]
    subprocess.Popen.__init__ = init  # type: ignore[method-assign]


def snapshot(dirs: tuple[Path, ...]) -> Snapshot:
    """``{path: (mtime_ns, size)}`` of every file and folder under *dirs* (the folders themselves included)."""
    found: Snapshot = {}
    for top in dirs:
        if not top.exists():
            continue
        for folder, subdirs, files in os.walk(top):
            for name in [".", *subdirs, *files]:
                path = os.path.normpath(os.path.join(folder, name))
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                found[path] = (st.st_mtime_ns, st.st_size)
    return found


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return False


def _of_a_running_window(path: str) -> bool:
    """*path* is the runtime context file of a FlexTool process that still runs, or the runtime folder."""
    parent, name = os.path.split(path)
    if name == "runtime":
        return True
    found = _RUNTIME_FILE.match(name)
    return os.path.basename(parent) == "runtime" and found is not None and _alive(int(found.group(1)))


def changes(before: Snapshot, after: Snapshot) -> list[str]:
    """The paths added, changed or removed between two snapshots, each with what happened; the runtime context
    files of FlexTool processes that still run are left out (the module text)."""
    lines = [f"added   {p}" for p in sorted(after.keys() - before.keys())]
    lines += [f"changed {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]]
    lines += [f"removed {p}" for p in sorted(before.keys() - after.keys())]
    return [line for line in lines if not _of_a_running_window(line[8:])]
