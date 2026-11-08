"""No test reads or writes the real user folders of FlexTool or the system keyring.

The root ``conftest.py`` calls :func:`redirect` when it is loaded, before any test module is imported: every
pytest process (an xdist worker too) gets its own ``FLEXTOOL_USER_HOME`` in a fresh temp folder, and the
keyring a started process finds is keyring's null backend.  Both are plain environment variables, so they hold
for the processes a test starts and for threads that outlive a test (a monkeypatch is undone at teardown, while a
worker thread may still write).  Tests that want a folder of their own set ``FLEXTOOL_USER_HOME`` with
``monkeypatch.setenv`` (``tests/assistant/conftest.py``).

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

Snapshot = dict[str, tuple[int, int]]

_RUNTIME_FILE = re.compile(r"^[0-9a-f]+\.(\d+)\.json$")


def real_user_dirs() -> tuple[Path, ...]:
    """The platform's user config, cache, data and state folders of FlexTool (``platformdirs`` does not read
    ``FLEXTOOL_USER_HOME``)."""
    import platformdirs

    return tuple(dict.fromkeys(Path(p) for p in (
        platformdirs.user_config_dir("flextool", appauthor=False),
        platformdirs.user_config_dir("flextool"),
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
