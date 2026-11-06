"""Every assistant test runs with the user cache folder in its own tmp folder and a fresh docs store.

``assistant_project`` is a FlexTool project in a folder whose path has a space, ``#``, ``%25`` and ``ä`` (the
characters that break an unencoded SQLite URI): ``input_sources/tests.sqlite`` is built from
``tests/fixtures/tests.json`` (and migrated to this FlexTool), and ``settings.yaml`` numbers it 1.
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_DIR_NAME = "G #2 ä%25x project"


@pytest.fixture(autouse=True)
def user_cache_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``platformdirs.user_cache_dir`` points into ``tmp_path``: no test writes the real user cache."""
    import platformdirs

    from flextool.assistant import docs_store

    cache = tmp_path / "user_cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *args, **kwargs: str(cache))
    docs_store.reset()
    yield cache
    docs_store.reset()


@pytest.fixture(autouse=True)
def fresh_input_cache() -> Iterator[None]:
    """Each test loads its databases itself (a cached load would hide what the test checks)."""
    from flextool.assistant.inputs import _db

    _db.clear_cache()
    yield
    _db.clear_cache()


@pytest.fixture(scope="session")
def tests_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``tests/fixtures/tests.json`` as a database at this FlexTool's version, in a plain tmp folder."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    path = tmp_path_factory.mktemp("tests_db") / "tests.sqlite"
    migrate_database(json_to_db(FIXTURES / "tests.json", path))
    return path


def make_project(root: Path, db: Path, name: str = "tests.sqlite", number: int = 1) -> Path:
    """A project folder under *root* with *db* copied to ``input_sources/<name>`` and numbered *number*."""
    project = root / PROJECT_DIR_NAME
    (project / "input_sources").mkdir(parents=True)
    shutil.copy(db, project / "input_sources" / name)
    (project / "settings.yaml").write_text(f"input_source_numbers:\n  {name}: {number}\n", encoding="utf-8")
    return project


@pytest.fixture(scope="session")
def assistant_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    return make_project(tmp_path_factory.mktemp("assistant"), tests_db)


@pytest.fixture
def ctx(assistant_project: Path):
    from flextool.assistant.context import make_context

    return make_context(assistant_project)
