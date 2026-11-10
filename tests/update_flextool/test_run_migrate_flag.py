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
import time
from pathlib import Path

import pytest

from flextool._file_lock import exclusive_file_lock
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import frozen as fz
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.migrate_before_run import migration_lock_path
from tests.db_utils import json_to_db

REPO = Path(__file__).resolve().parents[2]
SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
OLD = FLEXTOOL_DB_VERSION - 1
MIGRATING = "Migrating the input database from version"


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
    assert sum(out.count(MIGRATING) for out, _err in outputs) == 1
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION
    for name in ("run_a", "run_b"):
        assert (tmp_path / name / "out" / "output_parquet" / "det").is_dir()
