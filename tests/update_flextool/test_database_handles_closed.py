"""The fixture build / migrate tools leave no open handle to the databases they create.

Windows cannot delete a file that is still open, so a temporary folder whose database a tool left open cannot be
removed (WinError 32).  ``with DatabaseMapping(...)`` closes only the session; the engine keeps the SQLite file
open until it is disposed.  Each test lists the process's open files under the folder right after the call.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool.update_flextool.db_migration import migrate_repository_database
from flextool.update_flextool.export_database import export_database
from flextool.update_flextool.extend_tests_fixture import _apply_delta_to_db
from flextool.update_flextool.generate_canonical import _filter_db
from flextool.update_flextool.sync_master_json_template import _create_db_from_template, _export_db_to_json
from tests.db_utils import db_to_json, json_to_db

pytestmark = pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="open files are listed from /proc")

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE = FIXTURES / "stoch_two_period.json"


def _open_files(match) -> list[str]:
    out = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if match(target):
            out.append(target)
    return out


def _open_files_under(folder: Path) -> list[str]:
    return _open_files(lambda target: target.startswith(str(folder)))


@pytest.fixture
def staged(tmp_path) -> tuple[Path, str]:
    """The fixture imported into a staging database, checked to leave it closed."""
    db_path = tmp_path / "staging.sqlite"
    url = json_to_db(FIXTURE, db_path)
    assert _open_files_under(tmp_path) == [], "json_to_db"
    return db_path, url


def test_migrate_leaves_no_open_handle(staged, tmp_path) -> None:
    _, url = staged
    migrate_repository_database(url)
    assert _open_files_under(tmp_path) == []


def test_db_to_json_leaves_no_open_handle(staged, tmp_path) -> None:
    db_path, _ = staged
    db_to_json(db_path, tmp_path / "out.json")
    assert _open_files_under(tmp_path) == []


def test_export_database_leaves_no_open_handle(staged, tmp_path) -> None:
    db_path, _ = staged
    export_database(str(db_path), str(tmp_path / "out.json"))
    assert _open_files_under(tmp_path) == []


def test_canonical_filter_leaves_no_open_handle(staged, tmp_path) -> None:
    _, url = staged
    _filter_db(url, [], ["horizon"], [])
    assert _open_files_under(tmp_path) == []


def test_fixture_delta_leaves_no_open_handle(staged, tmp_path) -> None:
    _, url = staged
    _apply_delta_to_db(url, {"new_alternatives": [{"name": "added"}]}, {})
    assert _open_files_under(tmp_path) == []


def test_master_template_sync_leaves_no_open_handle(tmp_path) -> None:
    db_path = tmp_path / "template.sqlite"
    _create_db_from_template(Path(package_data_path("schemas/spinedb_schema.json")), db_path)
    assert _open_files_under(tmp_path) == []
    _export_db_to_json(db_path)
    assert _open_files_under(tmp_path) == []


def test_migrate_fixture_leaves_no_open_handle(monkeypatch, tmp_path) -> None:
    # The failing path on Windows CI: each temporary folder is checked for open files just before its removal.
    import contextlib
    import tempfile

    from flextool.update_flextool import test_fixtures

    held: list[list[str]] = []
    folders = iter(range(100))

    @contextlib.contextmanager
    def checked_temporary_directory():
        folder = tmp_path / f"tmp{next(folders)}"
        folder.mkdir()
        yield str(folder)
        held.append(_open_files_under(folder))

    monkeypatch.setattr(tempfile, "TemporaryDirectory", checked_temporary_directory)
    test_fixtures.migrate_fixture(FIXTURE, tmp_path / "out.json")
    assert held and held == [[]] * len(held)
