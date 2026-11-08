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
