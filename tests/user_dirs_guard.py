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

The user folders (:func:`redirect`, :func:`forbid_real_user_dirs`): besides ``FLEXTOOL_USER_HOME``, the process
that runs the session points ``HOME`` (``USERPROFILE`` on Windows), ``XDG_CONFIG_HOME``, ``XDG_CACHE_HOME``,
``XDG_DATA_HOME``, ``XDG_STATE_HOME``, ``APPDATA`` and ``LOCALAPPDATA`` into one temp folder of the session, which the
xdist workers and every process a test starts inherit: a fallback to the platform's folder (platformdirs,
``Path.home()``) lands there too.  The real home and the real FlexTool folders are kept in
:data:`REAL_HOME_ENV` / :data:`REAL_DIRS_ENV` for the workers.  Windows platformdirs asks the shell for the
AppData folders and ignores ``APPDATA``, so in every pytest process :mod:`flextool.user_dirs` is wrapped: a call
that would give a folder under the real home's config, cache or AppData folders, or under a real FlexTool folder,
is recorded (the session fails) and raises before anything is written.  The autouse fixture
:func:`tests.conftest._user_folders_stay_redirected` fails a test that left these variables changed.

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
import sys
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


#: The real home folder and the real FlexTool user folders, as the first process of the session found them (the
#: xdist workers and the processes a test starts inherit the redirected ``HOME`` and ``XDG_*``).
REAL_HOME_ENV = "FLEXTOOL_TEST_REAL_HOME"
REAL_DIRS_ENV = "FLEXTOOL_TEST_REAL_USER_DIRS"
#: The session's stand-in home folder (made by the first process of the session, inherited by the others).
SANDBOX_ENV = "FLEXTOOL_TEST_SANDBOX"
#: The variables :func:`redirect` points into the sandbox, each with its folder there.
_SANDBOX_VARIABLES = (("HOME", "home"), ("XDG_CONFIG_HOME", "home/.config"), ("XDG_CACHE_HOME", "home/.cache"),
                      ("XDG_DATA_HOME", "home/.local/share"), ("XDG_STATE_HOME", "home/.local/state"),
                      ("APPDATA", "home/AppData/Roaming"), ("LOCALAPPDATA", "home/AppData/Local"),
                      *((("USERPROFILE", "home"),) if sys.platform == "win32" else ()))
#: The folders of the real home that no FlexTool user folder of a test may lie under.
_REAL_HOME_CONFIG_DIRS = (".config", ".cache", ".local", "Library", "AppData")


def real_home() -> Path:
    """The user's real home folder (the first process of the session records it before :func:`redirect`)."""
    if REAL_HOME_ENV not in os.environ:
        os.environ[REAL_HOME_ENV] = str(Path.home())
    return Path(os.environ[REAL_HOME_ENV])


def real_user_dirs() -> tuple[Path, ...]:
    """The platform's user config, cache, data and state folders of FlexTool (``platformdirs`` does not read
    ``FLEXTOOL_USER_HOME``), as the first process of the session found them before :func:`redirect`."""
    if REAL_DIRS_ENV not in os.environ:
        import platformdirs

        found = dict.fromkeys(str(Path(p)) for p in (
            platformdirs.user_config_dir("flextool", appauthor=False),
            platformdirs.user_config_dir("flextool"),
            platformdirs.user_cache_dir("flextool", appauthor=False),
            platformdirs.user_cache_dir("flextool"),
            platformdirs.user_data_dir("flextool"),
            platformdirs.user_state_dir("flextool"),
        ))
        os.environ[REAL_DIRS_ENV] = os.pathsep.join(found)
    return tuple(Path(p) for p in os.environ[REAL_DIRS_ENV].split(os.pathsep) if p)


def forbidden_roots() -> tuple[Path, ...]:
    """The real FlexTool folders and the real home's config, cache, data and AppData folders."""
    home = real_home()
    return (*real_user_dirs(), *(home / name for name in _REAL_HOME_CONFIG_DIRS))


def _full(path: Path | str) -> Path:
    try:
        return Path(os.path.abspath(path)).resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(path))


def is_real_user_path(path: Path) -> bool:
    """*path* lies under one of :func:`forbidden_roots` (both compared as full, resolved paths), and not in the
    system temp folder.

    On Windows the temp folder is ``<home>\\AppData\\Local\\Temp``, under the real home's AppData folder: pytest's
    ``tmp_path`` and the session sandbox live there, so a path in the temp folder is never a real user folder
    (unless the temp folder itself lies in a real FlexTool folder)."""
    full = _full(path)
    real_dirs = tuple(_full(root) for root in real_user_dirs())
    temp = _full(tempfile.gettempdir())
    if full.is_relative_to(temp) and not any(temp.is_relative_to(root) for root in real_dirs):
        return False
    return any(full == root or full.is_relative_to(root) for root in (_full(r) for r in forbidden_roots()))


def redirect() -> tuple[Path, Path | None]:
    """Point ``FLEXTOOL_USER_HOME`` at a fresh temp folder of this process, the home and platform folders
    (:data:`_SANDBOX_VARIABLES`) into the session's sandbox, and the keyring at the null backend.  Records the real
    home and FlexTool folders first (:func:`real_home`, :func:`real_user_dirs`).  Returns this process's
    ``FLEXTOOL_USER_HOME`` and the sandbox when this process made it (``None`` when it inherited one)."""
    real_home()
    real_user_dirs()
    made = None
    sandbox = os.environ.get(SANDBOX_ENV, "")
    if not sandbox or not Path(sandbox).is_dir():
        made = Path(tempfile.mkdtemp(prefix="flextool-test-sandbox-"))
        sandbox = str(made)
        os.environ[SANDBOX_ENV] = sandbox
    for variable, folder in _SANDBOX_VARIABLES:
        path = Path(sandbox, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    home = Path(tempfile.mkdtemp(prefix="flextool-test-user-"))
    os.environ[USER_HOME_ENV] = str(home)
    os.environ[KEYRING_ENV] = NULL_KEYRING
    return home, made


def redirected_environment() -> dict[str, str]:
    """The variables :func:`redirect` set in this process, with their values (for the per-test check)."""
    names = (USER_HOME_ENV, *(variable for variable, _ in _SANDBOX_VARIABLES))
    return {name: os.environ.get(name, "") for name in names}


class RealUserDirTouched(Exception):
    """A test reached a real FlexTool user folder, or a config folder of the real home."""


def forbid_real_user_dirs() -> None:
    """Wrap :mod:`flextool.user_dirs` in this process: a call that gives a folder under :func:`forbidden_roots` is
    recorded (the session fails, :func:`keyring_violations`) and raises, so nothing is written there."""
    from flextool import user_dirs

    for name in ("user_config_dir", "user_cache_dir", "_platform_dir"):
        original = getattr(user_dirs, name)
        if getattr(original, "_user_dirs_guard", False):
            continue

        def guarded(*args: object, _original=original, _name=name, **kwargs: object) -> Path:
            path = _original(*args, **kwargs)
            if is_real_user_path(path):
                _record(f"flextool.user_dirs.{_name} gave the real user folder {path}")
                raise RealUserDirTouched(f"flextool.user_dirs.{_name} gave the real user folder {path}; a test "
                                         "must keep its user files under FLEXTOOL_USER_HOME")
            return path

        guarded.__name__ = name
        guarded.__doc__ = original.__doc__
        guarded._user_dirs_guard = True  # type: ignore[attr-defined]
        setattr(user_dirs, name, guarded)


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
