"""Tests for the settings-DB auto-seed helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flextool.update_flextool.ensure_settings_db import (
    SETTINGS_TEMPLATES,
    _sqlite_url_to_path,
    ensure_settings_db,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("basename", list(SETTINGS_TEMPLATES))
def test_template_files_exist(basename: str) -> None:
    """Each mapped JSON template must actually be present in the repo."""
    template = REPO_ROOT / SETTINGS_TEMPLATES[basename]
    assert template.is_file(), f"Template {template} is missing"


def test_seeds_when_missing(tmp_path: Path) -> None:
    """A known basename is created from its JSON template on first call."""
    target = tmp_path / "output_info.sqlite"
    assert not target.exists()
    result = ensure_settings_db(str(target), REPO_ROOT)
    assert result == target
    assert target.exists()
    assert target.stat().st_size > 0
    # Smoke check: file is a valid SQLite DB.
    con = sqlite3.connect(target)
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        con.close()
    assert tables, "Seeded DB should contain at least one Spine table"


def test_preserves_existing_file(tmp_path: Path) -> None:
    """If the file already exists, it must NOT be touched."""
    target = tmp_path / "output_settings.sqlite"
    target.write_bytes(b"user-edited")
    result = ensure_settings_db(str(target), REPO_ROOT)
    assert result is None
    assert target.read_bytes() == b"user-edited"


def test_unknown_basename_is_ignored(tmp_path: Path) -> None:
    """A custom name (not in the mapping) is never auto-seeded."""
    target = tmp_path / "my_custom_settings.sqlite"
    result = ensure_settings_db(str(target), REPO_ROOT)
    assert result is None
    assert not target.exists()


def test_none_and_empty_return_none() -> None:
    assert ensure_settings_db(None, REPO_ROOT) is None


def test_accepts_sqlite_url_three_slashes(tmp_path: Path) -> None:
    target = tmp_path / "comparison_settings.sqlite"
    url = f"sqlite:///{target}"
    result = ensure_settings_db(url, REPO_ROOT)
    assert result == target
    assert target.exists()


def test_accepts_sqlite_url_four_slashes(tmp_path: Path) -> None:
    target = tmp_path / "output_info.sqlite"
    url = f"sqlite:////{target.relative_to('/')}"
    result = ensure_settings_db(url, REPO_ROOT)
    assert result is not None
    assert result.exists()


def test_creates_parent_directory(tmp_path: Path) -> None:
    """If the parent dir doesn't exist yet, it should be created."""
    target = tmp_path / "nested" / "deeper" / "output_info.sqlite"
    assert not target.parent.exists()
    result = ensure_settings_db(str(target), REPO_ROOT)
    assert result == target
    assert target.exists()


def test_non_sqlite_url_returns_none() -> None:
    # mysql / remote URLs are out of scope for this helper
    assert ensure_settings_db("mysql://user:pw@host/db", REPO_ROOT) is None


def test_sqlite_url_to_path_variants(tmp_path: Path) -> None:
    # relative (sqlite:///relative/path.sqlite)
    p = _sqlite_url_to_path("sqlite:///foo/bar.sqlite")
    assert p == Path("foo/bar.sqlite")
    # absolute (sqlite:////abs/path.sqlite)
    p = _sqlite_url_to_path("sqlite:////abs/foo.sqlite")
    assert p == Path("/abs/foo.sqlite")
    # bare path
    p = _sqlite_url_to_path("bare_file.sqlite")
    assert p == Path("bare_file.sqlite")
    # None
    assert _sqlite_url_to_path(None) is None
