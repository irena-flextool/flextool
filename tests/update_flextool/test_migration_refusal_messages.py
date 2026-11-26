"""A database that is not migrated, or not restored, gets a message and an exit code, never a traceback.

The cases: a pre-migration copy (``<db>.premigration.bak``) left by an earlier migration, no room for the copy, and a
failed migration whose copy cannot be put back.  ``flextool-migrate-database`` and ``flextool-run --migrate`` print
the message and exit 1; ``flextool-update`` lists the database and goes on with the others.  A database truncated by
an interrupted restore (its original in the kept copy) gets the kept-copy message, not the frozen check's "cannot
read" one, and a current database is not held up by a leftover copy.

Seed: ``data/v75_seed_stoch_two_period.json`` migrated to ``FLEXTOOL_DB_VERSION - 1`` (never a checked-in
``.sqlite``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from tests.update_flextool._rerun import patch_migration
from flextool.update_flextool import migration_backup as mb
from flextool.update_flextool.frozen import read_db_version_ro
from tests.db_utils import json_to_db

REPO = Path(__file__).resolve().parents[2]
SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
OLD = FLEXTOOL_DB_VERSION - 1
KEPT = "still there"


@pytest.fixture(scope="module")
def old_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("refusal_seed") / "seed.sqlite"
    m.migrate_database(json_to_db(SEED, path), up_to=OLD)
    assert read_db_version_ro(path) == OLD
    return path


def _copy(template: Path, folder: Path, name: str = "db.sqlite") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    shutil.copy2(template, path)
    return path


def _keep_a_copy(path: Path) -> Path:
    """Leave a pre-migration copy beside *path*, as a killed migration does (the copy is the original)."""
    backup = mb.backup_path(path)
    shutil.copy2(path, backup)
    return backup


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    return env


def _migrate_cli(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "flextool.cli.cmd_migrate_database", str(path)],
                          capture_output=True, text=True, env=_env(), timeout=600)


def test_migrate_database_cli_refuses_a_kept_copy_with_a_message(tmp_path, old_template) -> None:
    path = _copy(old_template, tmp_path)
    before = path.read_bytes()
    backup = _keep_a_copy(path)
    done = _migrate_cli(path)
    assert done.returncode == 1, done.stderr
    assert KEPT in done.stderr and backup.name in done.stderr
    assert "Traceback" not in done.stderr + done.stdout
    assert path.read_bytes() == before and backup.read_bytes() == before


def test_the_kept_copy_message_says_when_the_copy_was_made_and_advises_once(tmp_path, old_template) -> None:
    path = _copy(old_template, tmp_path)
    backup = _keep_a_copy(path)
    with pytest.raises(mb.MigrationBackupError) as raised:
        m.migrate_database(str(path))
    text = str(raised.value)
    assert mb._made_on(backup) in text and mb._made_on(backup).startswith("on 20")
    assert text.count(f"rename {backup.name} to {path.name}") == 1
    assert text.count(f"delete {backup.name}") == 1


def test_migrate_database_cli_reports_no_room_with_a_message(tmp_path, old_template, monkeypatch, capsys) -> None:
    from flextool.cli import cmd_migrate_database

    path = _copy(old_template, tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(mb.shutil, "disk_usage", 
                        lambda _p: type("Usage", (), {"free": 1000})())
    assert cmd_migrate_database.main([str(path)]) == 1
    assert "no room for its pre-migration copy" in capsys.readouterr().err
    assert path.read_bytes() == before and not mb.backup_path(path).exists()


def test_migrate_database_cli_reports_a_failed_restore(tmp_path, old_template, monkeypatch, capsys) -> None:
    """The step fails and the copy cannot be put back: exit 1, the message names the kept copy (the original)."""
    from flextool.cli import cmd_migrate_database

    path = _copy(old_template, tmp_path)
    before = path.read_bytes()

    def failing_step(db) -> None:
        raise RuntimeError("step failed")

    patch_migration(monkeypatch, f"_migrate_v{FLEXTOOL_DB_VERSION}_schema_batch", failing_step)
    monkeypatch.setattr(mb, "restore_from_backup", lambda _db, _backup: False)
    assert cmd_migrate_database.main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "could not be restored" in err and mb.backup_path(path).name in err
    assert mb.backup_path(path).read_bytes() == before


def test_run_migrate_refuses_a_kept_copy_with_exit_1(tmp_path, old_template, capsys) -> None:
    from flextool.cli.cmd_run_flextool import _migrate_input_or_exit

    path = _copy(old_template, tmp_path)
    before = path.read_bytes()
    _keep_a_copy(path)
    with pytest.raises(SystemExit) as stopped:
        _migrate_input_or_exit(str(path))
    assert stopped.value.code == 1
    assert KEPT in capsys.readouterr().err
    assert path.read_bytes() == before


def test_self_update_goes_on_after_a_database_it_cannot_migrate(tmp_path, old_template, capsys) -> None:
    """The first database has a kept copy: it is listed and left as it is; the second is still migrated."""
    from flextool.update_flextool.self_update import _migrate_databases, _report_skipped_databases

    blocked = _copy(old_template, tmp_path, "blocked.sqlite")
    before = blocked.read_bytes()
    _keep_a_copy(blocked)
    other = _copy(old_template, tmp_path, "other.sqlite")
    frozen, not_migrated = _migrate_databases([str(blocked), str(other)])
    assert frozen == [] and len(not_migrated) == 1 and KEPT in not_migrated[0]
    assert blocked.read_bytes() == before
    assert read_db_version_ro(other) == FLEXTOOL_DB_VERSION and not mb.backup_path(other).exists()
    assert _report_skipped_databases(frozen, not_migrated) == 1
    assert KEPT in capsys.readouterr().err


def _truncate_after_interrupted_restore(path: Path) -> Path:
    """The file as an interrupted Windows restore leaves it: the copy (the original) kept, the file cut short."""
    backup = _keep_a_copy(path)
    path.write_bytes(backup.read_bytes()[:12288])
    return backup


def test_a_truncated_database_with_a_kept_copy_names_the_copy(tmp_path, old_template) -> None:
    """``migrate_database``, the run's pre-migration and the CLI name the kept copy instead of failing to read the
    frozen marker of the truncated file."""
    from flextool.update_flextool.migrate_before_run import migrate_before_run

    path = _copy(old_template, tmp_path)
    original = path.read_bytes()
    backup = _truncate_after_interrupted_restore(path)
    with pytest.raises(mb.MigrationBackupError, match=KEPT):
        m.migrate_database(str(path))
    with pytest.raises(mb.MigrationBackupError, match=KEPT):
        migrate_before_run(str(path))
    done = _migrate_cli(path)
    assert done.returncode == 1 and KEPT in done.stderr and "frozen" not in done.stderr
    assert backup.read_bytes() == original


def test_the_window_names_the_kept_copy_of_a_truncated_database(tmp_path, old_template) -> None:
    from flextool.gui import db_version_check

    path = _copy(old_template, tmp_path)
    backup = _truncate_after_interrupted_restore(path)
    was_upgraded, failed, messages = db_version_check.check_and_upgrade_database(path)
    assert failed and not was_upgraded
    assert len(messages) == 1 and KEPT in messages[0] and backup.name in messages[0]


def test_a_current_database_is_not_held_up_by_a_leftover_copy(tmp_path, old_template) -> None:
    """A copy left after a completed migration (its delete failed) does not stop a database that has nothing to
    migrate; the copy is left for the user."""
    path = _copy(old_template, tmp_path)
    m.migrate_database(str(path))
    backup = mb.backup_path(path)
    backup.write_bytes(b"left over")
    m.migrate_database(str(path))
    assert read_db_version_ro(path) == FLEXTOOL_DB_VERSION and backup.read_bytes() == b"left over"
