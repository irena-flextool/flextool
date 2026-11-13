"""No test can reach FlexTool's real user folders or the system keyring (``tests/user_dirs_guard.py``).

The session check in the root ``conftest.py`` fails a run that changed the real folders; these tests check the
redirect it relies on, in this process and in a process a test starts, and that the check sees a change.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import user_dirs_guard

from flextool.user_dirs import USER_HOME_ENV

_WHERE = """
import json
import keyring
from flextool.assistant import _files, docs_store, probes
print(json.dumps({"config": str(_files.config_root()), "cache": str(probes.cache_root()),
                  "docs": str(docs_store.cache_root()), "keyring": type(keyring.get_keyring()).__module__}))
"""


def _assert_under_test_home(paths: dict[str, str], home: Path) -> None:
    real = user_dirs_guard.real_user_dirs()
    for name in ("config", "cache", "docs"):
        path = Path(paths[name])
        assert path.is_relative_to(home), (name, path, home)
        assert not any(path.is_relative_to(r) for r in real), (name, path)


def test_this_process_keeps_user_files_under_a_temp_folder():
    home = Path(os.environ[USER_HOME_ENV])
    assert not any(home.is_relative_to(r) or r.is_relative_to(home) for r in user_dirs_guard.real_user_dirs())
    from flextool.assistant import _files, docs_store, probes

    _assert_under_test_home({"config": str(_files.config_root()), "cache": str(probes.cache_root()),
                             "docs": str(docs_store.cache_root())}, home)


def test_a_started_process_keeps_user_files_there_and_gets_no_keyring():
    out = subprocess.run([sys.executable, "-c", _WHERE], capture_output=True, text=True, encoding="utf-8",
                         timeout=120, check=True)
    found = json.loads(out.stdout.strip().splitlines()[-1])
    _assert_under_test_home(found, Path(os.environ[USER_HOME_ENV]))
    assert found["keyring"] == "keyring.backends.null"


_HOME_IN_CHILD = """
import json, os
from pathlib import Path
print(json.dumps({"home": str(Path.home()), **{v: os.environ.get(v, "") for v in (
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "APPDATA", "LOCALAPPDATA")}}))
"""


def test_home_and_platform_folders_point_into_the_session_sandbox_here_and_in_a_started_process():
    """A fallback to the platform's folder, or to ``Path.home()``, stays in the session's temp folder."""
    sandbox = Path(os.environ[user_dirs_guard.SANDBOX_ENV])
    assert not user_dirs_guard.is_real_user_path(sandbox)
    out = subprocess.run([sys.executable, "-c", _HOME_IN_CHILD], capture_output=True, text=True, encoding="utf-8",
                         timeout=120, check=True)
    for found in ({"home": str(Path.home()), **{v: os.environ[v] for v in (
            "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "APPDATA", "LOCALAPPDATA")}},
            json.loads(out.stdout.strip().splitlines()[-1])):
        for name, value in found.items():
            assert Path(value).is_relative_to(sandbox), (name, value)
    assert user_dirs_guard.real_home() != Path.home()


def test_a_user_dirs_call_that_gives_a_real_folder_raises_and_is_recorded(tmp_path: Path, monkeypatch):
    """Windows platformdirs ignores APPDATA: whatever still names a real folder raises before anything is
    written, and the session check sees it."""
    from flextool import user_dirs

    log = tmp_path / "violations.log"
    monkeypatch.setenv(user_dirs_guard.KEYRING_LOG_ENV, str(log))
    real = user_dirs_guard.real_home() / ".config" / "flextool"
    monkeypatch.setenv(USER_HOME_ENV, str(real))
    with pytest.raises(user_dirs_guard.RealUserDirTouched):
        user_dirs.user_config_dir()
    assert "gave the real user folder" in log.read_text(encoding="utf-8")
    monkeypatch.setenv(USER_HOME_ENV, str(tmp_path / "own"))
    assert user_dirs.user_config_dir() == tmp_path / "own" / "config"


def test_the_session_check_sees_added_changed_and_removed_files(tmp_path: Path):
    (tmp_path / "assistant").mkdir()
    kept, changed, removed = (tmp_path / "assistant" / n for n in ("kept.json", "changed.json", "removed.json"))
    for path in (kept, changed, removed):
        path.write_text("{}", encoding="utf-8")
    before = user_dirs_guard.snapshot((tmp_path,))
    assert user_dirs_guard.changes(before, user_dirs_guard.snapshot((tmp_path,))) == []
    changed.write_text('{"a": 1}', encoding="utf-8")
    removed.unlink()
    (tmp_path / "assistant" / "model_facts.json").write_text("{}", encoding="utf-8")
    found = user_dirs_guard.changes(before, user_dirs_guard.snapshot((tmp_path,)))
    assert f"added   {tmp_path / 'assistant' / 'model_facts.json'}" in found
    assert f"changed {changed}" in found
    assert f"removed {removed}" in found
    assert not any(str(kept) in line for line in found)


def test_a_running_windows_runtime_file_is_left_out_and_a_finished_ones_is_not(tmp_path: Path):
    runtime = tmp_path / "assistant" / "runtime"
    runtime.mkdir(parents=True)
    ended = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True,
                           check=True, timeout=60)
    before = user_dirs_guard.snapshot((tmp_path,))
    running = runtime / f"ef8cf030dfe1ad1e.{os.getppid()}.json"       # the parent of this process still runs
    finished = runtime / f"ef8cf030dfe1ad1e.{int(ended.stdout)}.json"
    running.write_text("{}", encoding="utf-8")
    finished.write_text("{}", encoding="utf-8")
    assert user_dirs_guard.changes(before, user_dirs_guard.snapshot((tmp_path,))) == [f"added   {finished}"]


def test_the_windows_folders_name_flextool_once(monkeypatch):
    """On Windows the cache folder is ``…\\Local\\flextool\\Cache``, not ``…\\flextool\\flextool\\Cache`` (review
    C3-n4: without ``appauthor=False`` platformdirs puts the app name in twice); ``FLEXTOOL_USER_HOME`` still
    stands in for both folders."""
    import platformdirs
    import platformdirs.windows

    from flextool import user_dirs

    home = os.environ[USER_HOME_ENV]
    monkeypatch.setattr(platformdirs.windows, "get_win_folder",
                        lambda csidl: {"CSIDL_LOCAL_APPDATA": "C:/Users/u/AppData/Local",
                                       "CSIDL_APPDATA": "C:/Users/u/AppData/Roaming"}[csidl])
    monkeypatch.setattr(platformdirs, "PlatformDirs", platformdirs.windows.Windows)
    monkeypatch.delenv(USER_HOME_ENV)
    assert user_dirs.user_cache_dir().as_posix() == "C:/Users/u/AppData/Local/flextool/Cache"
    assert user_dirs.user_config_dir().as_posix() == "C:/Users/u/AppData/Local/flextool"
    monkeypatch.setenv(USER_HOME_ENV, home)
    assert user_dirs.user_cache_dir() == Path(home) / "cache"
    assert user_dirs.user_config_dir() == Path(home) / "config"


# --- the keyring ---------------------------------------------------------------------------------------------------
_KEYRING_IN_CHILD = "import keyring; print(type(keyring.get_keyring()).__module__)"
_NO_SESSION_BUS = "unix:path=/nonexistent/flextool-test-bus"      # a child that tries D-Bus anyway finds nothing


def test_a_removal_from_the_default_keyring_raises_and_is_recorded(tmp_path: Path, monkeypatch):
    """A thread that outlived its test, or code after a monkeypatch was undone, reaches this process's default
    keyring: a removal there raises and is recorded for the session check (before: keyring's null backend let it
    pass in silence, and before that the system keyring removed the key)."""
    import keyring

    log = tmp_path / "keyring.log"
    monkeypatch.setenv(user_dirs_guard.KEYRING_LOG_ENV, str(log))
    assert keyring.get_password("flextool-assistant", "mistral") is None
    with pytest.raises(user_dirs_guard.RealKeyringTouched):
        keyring.delete_password("flextool-assistant", "mistral")
    with pytest.raises(user_dirs_guard.RealKeyringTouched):
        keyring.set_password("flextool-assistant", "mistral", "a-test-key")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [line.split(" (pid")[0] for line in lines] == [
        "removal from the default keyring: flextool-assistant / mistral",
        "store into the default keyring: flextool-assistant / mistral"]
    assert "test_a_removal_from_the_default_keyring_raises_and_is_recorded" in lines[0]
    assert "a-test-key" not in log.read_text(encoding="utf-8")


def test_the_system_keyring_backend_refuses_every_call(tmp_path: Path, monkeypatch):
    """A test that builds the Secret Service backend itself never reaches D-Bus: the call raises and is recorded."""
    secret_service = pytest.importorskip("keyring.backends.SecretService")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", _NO_SESSION_BUS)
    log = tmp_path / "keyring.log"
    monkeypatch.setenv(user_dirs_guard.KEYRING_LOG_ENV, str(log))
    backend = secret_service.Keyring()
    for call in (lambda: backend.get_password("flextool-assistant", "mistral"),
                 lambda: backend.delete_password("flextool-assistant", "mistral")):
        with pytest.raises(user_dirs_guard.RealKeyringTouched):
            call()
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_a_started_process_with_an_environment_of_its_own_gets_no_keyring():
    """A test that builds the child's environment without the keyring variable: the child still gets the null
    backend (before: the system keyring)."""
    env = {k: v for k, v in os.environ.items() if k != user_dirs_guard.KEYRING_ENV}
    env["DBUS_SESSION_BUS_ADDRESS"] = _NO_SESSION_BUS
    out = subprocess.run([sys.executable, "-c", _KEYRING_IN_CHILD], capture_output=True, text=True,
                         encoding="utf-8", timeout=120, check=True, env=env)
    assert out.stdout.strip().splitlines()[-1] == "keyring.backends.null"
    chosen = {**env, user_dirs_guard.KEYRING_ENV: "keyring.backends.SecretService.Keyring"}
    out = subprocess.run([sys.executable, "-c", _KEYRING_IN_CHILD], capture_output=True, text=True,
                         encoding="utf-8", timeout=120, check=True, env=chosen)
    assert out.stdout.strip().splitlines()[-1] == "keyring.backends.null"
