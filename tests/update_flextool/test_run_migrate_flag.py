"""``--migrate`` on the model run: an older input database is migrated before the engine starts, once across runs.

Each run is a separate process (``python -m flextool.cli.cmd_run_flextool``), as ``flextool-run`` starts it, so the
lock is exercised across processes.  Seed: ``data/v75_seed_stoch_two_period.json`` migrated to
``FLEXTOOL_DB_VERSION - 1`` (a database built from the current schema JSON is already current); scenario ``det``.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from flextool._file_lock import exclusive_file_lock
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import frozen as fz
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.migrate_before_run import migrate_before_run, migration_lock, migration_lock_path
from tests.db_utils import json_to_db

REPO = Path(__file__).resolve().parents[2]
SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
OLD = FLEXTOOL_DB_VERSION - 1
MIGRATING = "Migrating the input database from version"
WAITING = "Waiting for another migration of"
BY_OTHER = "The input database was migrated by another run"


@pytest.fixture(scope="module")
def old_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("run_migrate_seed") / "seed.sqlite"
    migrate_database(json_to_db(SEED, path), up_to=OLD)
    assert fz.read_db_version_ro(path) == OLD
    return path


def _copy(template: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.sqlite"
    shutil.copy2(template, path)
    return path


def _run_cmd(db: Path, run_folder: Path, *extra: str) -> list[str]:
    return [
        sys.executable, "-m", "flextool.cli.cmd_run_flextool", str(db), "--scenario-name", "det",
        "--write-methods", "parquet", "--work-folder", str(run_folder / "work"),
        "--output-location", str(run_folder / "out"), *extra,
    ]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    env.setdefault("POLARS_MAX_THREADS", "2")
    return env


def _run(db: Path, run_folder: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(_run_cmd(db, run_folder, *extra), cwd=run_folder.parent, env=_env(),
                          capture_output=True, text=True, timeout=600)


def test_without_the_flag_an_older_database_is_refused(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path / "db")
    digest = fz.sha256_of(db)
    result = _run(db, tmp_path / "run")
    assert result.returncode != 0
    assert f"needs version {FLEXTOOL_DB_VERSION}" in result.stdout + result.stderr
    assert "add --migrate" in result.stdout + result.stderr
    assert fz.sha256_of(db) == digest


def test_with_the_flag_an_older_database_is_migrated_and_runs(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path / "db")
    result = _run(db, tmp_path / "run", "--migrate")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert result.stdout.count(MIGRATING) == 1
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION
    assert (tmp_path / "run" / "out" / "output_parquet" / "det").is_dir()


def test_a_frozen_database_is_not_migrated_with_the_flag(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path / "db")
    con = sqlite3.connect(db)
    with con:
        con.execute("INSERT INTO metadata (name, value) VALUES (?, ?)",
                    (fz.FROZEN_KEY, f"Reference case 1.0|{OLD}|2026-11-01"))
    con.close()
    digest = fz.sha256_of(db)
    result = _run(db, tmp_path / "run", "--migrate")
    assert result.returncode == 2
    assert "model.sqlite is frozen ('Reference case 1.0'" in result.stderr and "--copy-to" in result.stderr
    assert MIGRATING not in result.stdout
    assert fz.sha256_of(db) == digest


def test_parallel_runs_with_the_flag_migrate_once_and_both_run(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path / "db")
    # Hold the migration lock while both runs start, so that both reach it before either can migrate.
    with exclusive_file_lock(migration_lock_path(db.resolve())):
        procs = []
        for name in ("run_a", "run_b"):
            folder = tmp_path / name
            folder.mkdir()
            procs.append(subprocess.Popen(_run_cmd(db, folder, "--migrate"), cwd=tmp_path, env=_env(),
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        time.sleep(5)
        assert fz.read_db_version_ro(db) == OLD
    outputs = [proc.communicate(timeout=600) for proc in procs]
    for proc, (out, err) in zip(procs, outputs):
        assert proc.returncode == 0, out[-3000:] + err[-3000:]
    # Both runs waited for the lock (if one had arrived after the release, the double check would be untested).
    assert all(out.count(WAITING) == 1 for out, _err in outputs), [out[-3000:] for out, _err in outputs]
    assert sum(out.count(MIGRATING) for out, _err in outputs) == 1
    assert sum(out.count(BY_OTHER) for out, _err in outputs) == 1
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION
    for name in ("run_a", "run_b"):
        assert (tmp_path / name / "out" / "output_parquet" / "det").is_dir()


def _start(target, *args) -> tuple[threading.Thread, list]:
    result: list = []

    def _run_it() -> None:
        try:
            result.append(target(*args))
        except BaseException as exc:  # noqa: BLE001 - handed to the test thread
            result.append(exc)

    thread = threading.Thread(target=_run_it, daemon=True)
    thread.start()
    return thread, result


def test_a_database_unreadable_during_another_migration_waits_for_the_lock(
        tmp_path: Path, old_db_template: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    # Another migration holds the lock and SQLite's write lock: the version cannot be read, so the run must wait for
    # the lock and decide under it, not skip the migration.
    monkeypatch.setattr(fz, "_BUSY_TIMEOUT_S", 0.1)
    db = _copy(old_db_template, tmp_path / "db").resolve()
    writer = sqlite3.connect(db)
    with exclusive_file_lock(migration_lock_path(db)):
        writer.execute("BEGIN EXCLUSIVE")
        thread, result = _start(migrate_before_run, f"sqlite:///{db}")
        thread.join(2)
        assert thread.is_alive() and not result
        writer.rollback()
        writer.close()
    thread.join(600)
    assert result == [OLD]
    out = capsys.readouterr().out
    assert out.count(WAITING) == 1 and out.count(MIGRATING) == 1
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION


def test_migrate_database_takes_the_migration_lock(tmp_path: Path, old_db_template: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    # flextool-migrate-database, the GUI refresh and the Toolbox tool all go through migrate_database.
    db = _copy(old_db_template, tmp_path / "db").resolve()
    with exclusive_file_lock(migration_lock_path(db)):
        thread, result = _start(migrate_database, str(db))
        thread.join(2)
        assert thread.is_alive() and fz.read_db_version_ro(db) == OLD
    thread.join(600)
    assert result == [None]
    assert capsys.readouterr().out.count(WAITING) == 1
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION


def test_the_migration_lock_is_reentrant_within_a_thread(tmp_path: Path, old_db_template: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    db = _copy(old_db_template, tmp_path / "db").resolve()
    with migration_lock(db):
        assert migrate_before_run(str(db)) == OLD
    assert WAITING not in capsys.readouterr().out
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION


def test_a_file_system_without_locks_stops_with_a_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    import errno

    fcntl = pytest.importorskip("fcntl")  # the POSIX flock path; Windows takes msvcrt.locking

    from flextool._file_lock import FileLockError
    from flextool.cli.cmd_run_flextool import _migrate_input_or_exit

    def _no_locks(_fd: int, _op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", _no_locks)
    lock = tmp_path / "x.sqlite.migrate.lock"
    with pytest.raises(FileLockError, match="x.sqlite.migrate.lock: the file system does not support file locks"):
        with exclusive_file_lock(lock):
            pass
    db = tmp_path / "x.sqlite"
    db.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        _migrate_input_or_exit(f"sqlite:///{db}")
    assert excinfo.value.code == 1
    assert "The input database was not migrated: Cannot lock" in capsys.readouterr().err


@pytest.fixture(scope="module")
def current_db_template(tmp_path_factory: pytest.TempPathFactory, old_db_template: Path) -> Path:
    path = tmp_path_factory.mktemp("run_migrate_current") / "current.sqlite"
    shutil.copy2(old_db_template, path)
    migrate_database(str(path))
    migration_lock_path(path).unlink(missing_ok=True)
    assert fz.is_current_ro(path)
    return path


def test_a_current_database_creates_no_lock_file(tmp_path: Path, current_db_template: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    from flextool.gui.db_version_check import check_and_upgrade_database

    db = _copy(current_db_template, tmp_path / "db").resolve()
    digest = fz.sha256_of(db)
    migrate_database(str(db))
    assert "already up-to-date at version" in capsys.readouterr().out
    assert migrate_before_run(str(db)) is None
    check_and_upgrade_database(db)
    assert not migration_lock_path(db).exists()
    assert fz.sha256_of(db) == digest


def test_a_current_database_does_not_wait_for_a_held_lock(tmp_path: Path, current_db_template: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    db = _copy(current_db_template, tmp_path / "db").resolve()
    with exclusive_file_lock(migration_lock_path(db)):
        for target in (migrate_database, migrate_before_run):
            thread, result = _start(target, str(db))
            thread.join(30)
            assert not thread.is_alive() and result == [None]
    assert WAITING not in capsys.readouterr().out


def test_a_bad_lock_timeout_does_not_stop_a_posix_migration(tmp_path: Path, old_db_template: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fcntl")
    from flextool.update_flextool.migrate_before_run import TIMEOUT_ENV

    monkeypatch.setenv(TIMEOUT_ENV, "ten minutes")
    db = _copy(old_db_template, tmp_path / "db")
    assert migrate_before_run(str(db)) == OLD
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION


def test_the_lock_module_imports_where_errno_has_no_edeadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS's ``errno`` has ``EDEADLK`` but no ``EDEADLOCK``; importing flextool must not need it."""
    import errno
    import importlib.util

    import flextool._file_lock as file_lock

    monkeypatch.delattr(errno, "EDEADLOCK", raising=False)
    # A fresh copy under another name: the imported module (and its exception classes) stays untouched.
    spec = importlib.util.spec_from_file_location("_file_lock_without_edeadlock", file_lock.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy._WINDOWS_HELD == {errno.EACCES, errno.EDEADLK}
