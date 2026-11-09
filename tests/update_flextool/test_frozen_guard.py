"""Frozen-database guard (C-8): a frozen database is never migrated in place without the user saying so.

Seed: ``data/v75_seed_stoch_two_period.json`` migrated to ``FLEXTOOL_DB_VERSION - 1``, a genuinely older database (a
database built from the current schema JSON is already current, so ``up_to`` could not make it older).  A database
frozen by an older FlexTool carries its marker at that older version; :func:`_mark_frozen` writes it the same way
``freeze`` does.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping

from flextool.cli import cmd_freeze_database, cmd_migrate_database
from flextool.cli.cmd_execute_flextool_workflow import migrate_input_if_needed
from flextool.engine.input._db_reader import check_version
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.gui.data_models import ProjectSettings
from flextool.gui.db_version_check import check_and_upgrade_database
from flextool.gui.input_sources import InputSourceManager
from flextool.update_flextool import FLEXTOOL_DB_VERSION, db_handle, db_migration
from flextool.update_flextool import frozen as fz
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
OLD = FLEXTOOL_DB_VERSION - 1
LABEL = "Reference case 1.0"


@pytest.fixture(scope="module")
def old_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("frozen_seed") / "seed.sqlite"
    migrate_database(json_to_db(SEED, path), up_to=OLD)
    assert fz.read_db_version_ro(path) == OLD
    return path


@pytest.fixture(scope="module")
def current_db_template(tmp_path_factory: pytest.TempPathFactory, old_db_template: Path) -> Path:
    path = tmp_path_factory.mktemp("frozen_current") / "current.sqlite"
    shutil.copy2(old_db_template, path)
    migrate_database(str(path))
    assert fz.read_db_version_ro(path) == FLEXTOOL_DB_VERSION
    return path


def _copy(template: Path, folder: Path, name: str = "model.sqlite") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    shutil.copy2(template, path)
    return path


def _mark_frozen(path: Path, version: int = OLD) -> None:
    """The marker an older FlexTool's ``freeze`` wrote (only a current database can be frozen)."""
    with DatabaseMapping(f"sqlite:///{path}", upgrade=False) as db:
        db.add_item("metadata", name=fz.FROZEN_KEY, value=f"{LABEL}|{version}|2026-11-01")
        db.commit_session("test: freeze")


@pytest.fixture()
def no_upgrade_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fail any ``upgrade=True`` open, so a probe or guard that would change the SpineDB schema is caught."""
    opened: list[str] = []
    real = db_handle.open_database_mapping

    def guarded(url, **kwargs):
        if kwargs.get("upgrade"):
            opened.append(url)
            raise AssertionError(f"upgrade=True open of {url}")
        return real(url, **kwargs)

    monkeypatch.setattr(db_handle, "open_database_mapping", guarded)
    monkeypatch.setattr(db_migration, "open_database_mapping", guarded)
    return opened


# ── 1. freeze / read / unfreeze ───────────────────────────────────────


def test_freeze_status_unfreeze_round_trip(tmp_path: Path, current_db_template: Path, capsys) -> None:
    db = _copy(current_db_template, tmp_path)
    assert fz.read_frozen(db) is None
    assert cmd_freeze_database.main([str(db), "--label", LABEL]) == 0
    info = fz.read_frozen(db)
    assert info == fz.FrozenInfo(label=LABEL, version=FLEXTOOL_DB_VERSION, date=info.date)
    assert len(info.date) == 10
    capsys.readouterr()
    assert cmd_freeze_database.main([str(db), "--status"]) == 0
    assert f"Frozen: '{LABEL}', frozen at version {FLEXTOOL_DB_VERSION}" in capsys.readouterr().out
    # Freezing again replaces the marker: one row only.
    fz.freeze(db, "Second label")
    assert len(fz._metadata_rows(db, fz.FROZEN_KEY)) == 1
    assert fz.read_frozen(db).label == "Second label"
    assert cmd_freeze_database.main([str(db), "--unfreeze"]) == 0
    assert fz.read_frozen(db) is None
    assert fz._metadata_rows(db, fz.FROZEN_KEY) == []


def test_freeze_refuses_an_older_database_and_bad_labels(tmp_path: Path, old_db_template: Path,
                                                         current_db_template: Path) -> None:
    old = _copy(old_db_template, tmp_path / "old")
    digest = fz.sha256_of(old)
    with pytest.raises(fz.FreezeError, match=f"only a database at the current version \\({FLEXTOOL_DB_VERSION}\\)"):
        fz.freeze(old, LABEL)
    assert cmd_freeze_database.main([str(old), "--label", LABEL]) == 2
    assert fz.sha256_of(old) == digest
    current = _copy(current_db_template, tmp_path / "current")
    for bad in ("", "a|b", "x" * (fz.MAX_LABEL_LENGTH + 1)):
        with pytest.raises(fz.FreezeError):
            fz.freeze(current, bad)
    assert fz.read_frozen(current) is None


# ── 2. flextool-migrate-database ──────────────────────────────────────


def test_cli_refuses_a_frozen_older_database(tmp_path: Path, old_db_template: Path, capsys) -> None:
    db = _copy(old_db_template, tmp_path)
    _mark_frozen(db)
    digest = fz.sha256_of(db)
    assert cmd_migrate_database.main([str(db)]) == cmd_migrate_database.EXIT_FROZEN == 2
    err = capsys.readouterr().err
    assert "is frozen" in err and LABEL in err and "--copy-to" in err and "--allow-migrate-frozen" in err
    assert fz.sha256_of(db) == digest


def test_cli_copy_to_migrates_a_copy_and_leaves_the_original(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path)
    _mark_frozen(db)
    digest = fz.sha256_of(db)
    copy = tmp_path / "copy.sqlite"
    assert cmd_migrate_database.main([str(db), "--copy-to", str(copy)]) == 0
    assert fz.sha256_of(db) == digest
    assert fz.read_db_version_ro(copy) == FLEXTOOL_DB_VERSION
    assert fz.read_frozen(copy) is None
    assert fz.read_migrated_from(copy) == fz.MigratedFrom(label=LABEL, version=OLD, sha256=digest)
    # An existing destination is refused.
    assert cmd_migrate_database.main([str(db), "--copy-to", str(copy)]) == 1


def test_cli_allow_migrate_frozen_migrates_in_place_and_keeps_the_label(tmp_path: Path,
                                                                       old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path)
    _mark_frozen(db)
    assert cmd_migrate_database.main([str(db), "--allow-migrate-frozen"]) == 0
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION
    assert fz.read_frozen(db) == fz.FrozenInfo(label=LABEL, version=FLEXTOOL_DB_VERSION, date="2026-11-01")


def test_a_frozen_current_database_is_left_unchanged(tmp_path: Path, current_db_template: Path) -> None:
    """``migrate_database`` on a current database commits a description refresh; a frozen one is not touched."""
    db = _copy(current_db_template, tmp_path)
    fz.freeze(db, LABEL)
    digest = fz.sha256_of(db)
    assert cmd_migrate_database.main([str(db)]) == 0
    assert fz.sha256_of(db) == digest


# ── 3. GUI planning and the in-place upgrade ──────────────────────────


def _project(tmp_path: Path) -> InputSourceManager:
    (tmp_path / "input_sources").mkdir(parents=True, exist_ok=True)
    return InputSourceManager(tmp_path, ProjectSettings())


def test_plan_routes_frozen_sources_to_the_choice_without_upgrading(tmp_path: Path, old_db_template: Path,
                                                                   no_upgrade_open: list[str]) -> None:
    mgr = _project(tmp_path / "project")
    internal = _copy(old_db_template, mgr.input_dir, "frozen_internal.sqlite")
    _mark_frozen(internal)
    plain = _copy(old_db_template, mgr.input_dir, "plain.sqlite")
    external = _copy(old_db_template, tmp_path / "elsewhere", "frozen_external.sqlite")
    _mark_frozen(external)
    mgr.add_external_ref(external)
    digests = {p: fz.sha256_of(p) for p in (internal, external)}

    internal_list, external_list, frozen_list, _messages = mgr.plan_db_migrations()

    assert internal_list == [("plain.sqlite", plain)]
    assert external_list == []
    assert {(e.source_name, e.external, e.label, e.current_version) for e in frozen_list} == {
        ("frozen_internal.sqlite", False, LABEL, OLD), ("frozen_external.sqlite", True, LABEL, OLD)}
    assert no_upgrade_open == []
    assert {p: fz.sha256_of(p) for p in digests} == digests

    # The guard of every migration path refuses before any upgrade=True open.
    with pytest.raises(fz.FrozenDatabaseError):
        migrate_database(str(internal))
    upgraded, failed, messages = check_and_upgrade_database(internal)
    assert (upgraded, failed) == (False, True) and "is frozen" in messages[0]
    assert no_upgrade_open == []
    assert {p: fz.sha256_of(p) for p in digests} == digests


def test_declined_frozen_source_is_not_asked_again_and_not_runnable(tmp_path: Path,
                                                                   old_db_template: Path) -> None:
    mgr = _project(tmp_path)
    db = _copy(old_db_template, mgr.input_dir, "frozen.sqlite")
    _mark_frozen(db)
    (entry,) = mgr.plan_db_migrations()[2]
    mgr.decline_frozen(entry.path)
    assert mgr.plan_db_migrations()[2] == []
    (source,) = mgr.refresh()
    assert source.status == "frozen_old" and source.frozen_label == LABEL
    assert mgr.get_all_scenarios() == []
    mgr.undecline_frozen(entry.path)
    assert len(mgr.plan_db_migrations()[2]) == 1


def test_a_migrated_copy_stops_the_prompt_and_runs(tmp_path: Path, old_db_template: Path) -> None:
    mgr = _project(tmp_path)
    db = _copy(old_db_template, mgr.input_dir, "frozen.sqlite")
    _mark_frozen(db)
    digest = fz.sha256_of(db)
    (entry,) = mgr.plan_db_migrations()[2]
    copy = mgr.frozen_copy_path(entry)
    assert copy.name == f"frozen_v{FLEXTOOL_DB_VERSION}.sqlite"
    fz.migrate_copy(db, copy)
    assert fz.sha256_of(db) == digest
    assert mgr.plan_db_migrations() == ([], [], [], [])
    sources = {s.name: s for s in mgr.refresh()}
    assert sources["frozen.sqlite"].status == "frozen_old"
    assert copy.name in sources["frozen.sqlite"].frozen_note
    assert sources[copy.name].status == "ok" and sources[copy.name].frozen_label is None
    assert {s.source_name for s in mgr.get_all_scenarios()} == {copy.name}


def test_allowed_in_place_upgrade_keeps_the_marker(tmp_path: Path, old_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path)
    _mark_frozen(db)
    upgraded, failed, _messages = check_and_upgrade_database(db, allow_frozen=True)
    assert (upgraded, failed) == (True, False)
    assert fz.read_frozen(db).version == FLEXTOOL_DB_VERSION
    assert not (tmp_path / "model.sqlite.premigration.bak").exists()


# ── 4. the workflow's migrate step ────────────────────────────────────


def test_workflow_migrates_an_old_database_but_not_a_frozen_one(tmp_path: Path, old_db_template: Path,
                                                               current_db_template: Path) -> None:
    db = _copy(old_db_template, tmp_path / "plain")
    assert migrate_input_if_needed(f"sqlite:///{db}") == OLD
    assert fz.read_db_version_ro(db) == FLEXTOOL_DB_VERSION
    current = _copy(current_db_template, tmp_path / "current")
    digest = fz.sha256_of(current)
    assert migrate_input_if_needed(str(current)) is None
    assert fz.sha256_of(current) == digest
    frozen = _copy(old_db_template, tmp_path / "frozen")
    _mark_frozen(frozen)
    digest = fz.sha256_of(frozen)
    with pytest.raises(fz.FrozenDatabaseError):
        migrate_input_if_needed(str(frozen))
    assert fz.sha256_of(frozen) == digest


# ── 5. the engine refuses a database that is not current ──────────────


def test_engine_refuses_an_unmigrated_database(tmp_path: Path, old_db_template: Path,
                                               current_db_template: Path) -> None:
    import logging

    db = _copy(old_db_template, tmp_path / "old")
    with DatabaseMapping(f"sqlite:///{db}") as mapping:
        with pytest.raises(FlexToolConfigError) as excinfo:
            check_version(mapping, logging.getLogger(__name__))
    message = str(excinfo.value)
    assert f"version {OLD}" in message and f"needs version {FLEXTOOL_DB_VERSION}" in message
    assert f"flextool-migrate-database {db}" in message and "--copy-to" in message
    current = _copy(current_db_template, tmp_path / "current")
    with DatabaseMapping(f"sqlite:///{current}") as mapping:
        check_version(mapping, logging.getLogger(__name__))
