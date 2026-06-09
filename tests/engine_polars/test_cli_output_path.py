"""Unit tests for the 4-tier output-root resolver in cmd_run_flextool.

``resolve_output_path`` decides the TRUE output root for a CLI run — the
path that outputs land under and that is persisted to the Output-info DB
as ``scenario/output_location`` (Toolbox's comparison / re-create steps
read it back to find each scenario's parquet).  The four tiers, in
precedence order:

  1. explicit ``--output-location`` wins,
  2. GUI project layout ``<project>/input_sources/<db>.sqlite`` → ``<project>``,
  3. legacy ``--flextool-location``.parent.parent,
  4. CWD fallback.

These are pure path-logic tests; no DB is read and nothing is solved.
"""
from __future__ import annotations

from pathlib import Path

from flextool.cli.cmd_run_flextool import (
    _input_db_filesystem_path,
    resolve_output_path,
)


def _touch_db(path: Path) -> Path:
    """Create an empty file at *path* (parents made) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# Tier 1 — explicit --output-location wins.
# ---------------------------------------------------------------------------


def test_tier1_output_location_wins_over_input_sources_layout(tmp_path):
    """An explicit output_location beats every other tier, including an
    input_sources-layout DB (which would otherwise fire tier 2)."""
    project = tmp_path / "projects" / "Foo"
    db = _touch_db(project / "input_sources" / "in.sqlite")
    explicit = tmp_path / "elsewhere"

    out = resolve_output_path(
        input_db_url=f"sqlite:///{db}",
        flextool_location=str(tmp_path / "anchor" / "templates" / "x.txt"),
        output_location=str(explicit),
        cwd=tmp_path / "cwd",
    )
    assert out == Path(str(explicit))
    # Specifically NOT the project folder.
    assert out != project


# ---------------------------------------------------------------------------
# Tier 2 — GUI project layout: <project>/input_sources/<db>.sqlite.
# ---------------------------------------------------------------------------


def test_tier2_input_sources_layout_roots_at_project(tmp_path):
    """A DB at ``.../projects/Foo/input_sources/x.sqlite`` roots the
    output at ``.../projects/Foo`` (the project folder)."""
    project = tmp_path / "projects" / "Foo"
    db = _touch_db(project / "input_sources" / "x.sqlite")

    out = resolve_output_path(
        input_db_url=f"sqlite:///{db}",
        flextool_location=None,
        output_location=None,
        cwd=tmp_path / "cwd",
    )
    assert out == project.resolve()


def test_tier2_input_sources_with_filter_query(tmp_path):
    """A sqlite URL carrying an appended Spine filter query-config still
    resolves to the project folder (the query part is stripped)."""
    project = tmp_path / "projects" / "Bar"
    db = _touch_db(project / "input_sources" / "model.sqlite")

    url = (
        f"sqlite:///{db}"
        "?spinedbfilter=%7B%22type%22%3A%22scenario_filter%22%7D"
    )
    out = resolve_output_path(
        input_db_url=url,
        flextool_location=None,
        output_location=None,
        cwd=tmp_path / "cwd",
    )
    assert out == project.resolve()


def test_tier2_skipped_when_db_missing(tmp_path):
    """If the input_sources-layout path doesn't exist on disk, tier 2 is
    skipped (no crash) and resolution falls through to the next tier."""
    # Note: this path is NEVER created on disk.
    missing_db = tmp_path / "projects" / "Ghost" / "input_sources" / "x.sqlite"
    cwd = tmp_path / "cwd"

    out = resolve_output_path(
        input_db_url=f"sqlite:///{missing_db}",
        flextool_location=None,
        output_location=None,
        cwd=cwd,
    )
    # Falls through to tier 4 (CWD) — NOT rooted at the project folder.
    assert out == Path(cwd)
    assert out != missing_db.parent.parent


# ---------------------------------------------------------------------------
# Tier 3 / 4 — DB NOT in input_sources falls through; never roots at the
# DB's own parent.
# ---------------------------------------------------------------------------


def test_tier3_flextool_location_when_not_input_sources(tmp_path):
    """A DB that is NOT inside input_sources/ (e.g. a bare repo-root db)
    falls through to the flextool_location anchor walk
    (``.parent.parent``) — and is NEVER rooted at the DB's own parent."""
    # DB parent and anchor's .parent.parent are deliberately DIFFERENT
    # dirs so the "never roots at the DB's own parent" check is meaningful.
    db = _touch_db(tmp_path / "dbdir" / "input_data.sqlite")
    anchor = tmp_path / "install" / "templates" / "flextool_location.txt"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("# anchor\n")

    out = resolve_output_path(
        input_db_url=f"sqlite:///{db}",
        flextool_location=str(anchor),
        output_location=None,
        cwd=tmp_path / "cwd",
    )
    assert out == anchor.resolve().parent.parent  # == tmp_path/install
    # Critically NOT the DB's own parent.
    assert out != db.parent


def test_tier4_cwd_when_nothing_set(tmp_path):
    """A bare ``/tmp/x.sqlite``-style DB with no flextool_location and no
    output_location falls all the way through to CWD; NEVER the DB's own
    parent."""
    db = _touch_db(tmp_path / "loose" / "x.sqlite")
    cwd = tmp_path / "cwd"

    out = resolve_output_path(
        input_db_url=f"sqlite:///{db}",
        flextool_location=None,
        output_location=None,
        cwd=cwd,
    )
    assert out == Path(cwd)
    assert out != db.parent


def test_tier4_bare_path_input_db(tmp_path):
    """A bare filesystem path (no ``sqlite:///`` scheme) in input_sources
    layout still fires tier 2 — the URL idiom tolerates both forms."""
    project = tmp_path / "projects" / "Baz"
    db = _touch_db(project / "input_sources" / "y.sqlite")

    out = resolve_output_path(
        input_db_url=str(db),  # bare path, no scheme
        flextool_location=None,
        output_location=None,
        cwd=tmp_path / "cwd",
    )
    assert out == project.resolve()


# ---------------------------------------------------------------------------
# Helper: _input_db_filesystem_path edge cases.
# ---------------------------------------------------------------------------


def test_helper_non_sqlite_scheme_returns_none():
    """A non-sqlite URL (e.g. mysql) is not a local file → None."""
    assert _input_db_filesystem_path("mysql://user@host/db") is None


def test_helper_empty_returns_none():
    assert _input_db_filesystem_path("") is None
    assert _input_db_filesystem_path(None) is None


def test_helper_strips_query_config(tmp_path):
    db = tmp_path / "a" / "b.sqlite"
    got = _input_db_filesystem_path(f"sqlite:///{db}?spinedbfilter=xyz")
    assert got == Path(str(db))


def test_helper_non_sqlite_with_input_sources_does_not_fire_tier2(tmp_path):
    """Defensive: a non-sqlite scheme must not accidentally root at a
    project even if the URL text contains 'input_sources'."""
    out = resolve_output_path(
        input_db_url="mysql://host/projects/Foo/input_sources/db",
        flextool_location=None,
        output_location=None,
        cwd=tmp_path / "cwd",
    )
    assert out == Path(tmp_path / "cwd")
