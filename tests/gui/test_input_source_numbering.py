"""Tests for InputSourceManager numbering, the source registry, ghost rows.

Covers the persistent-identity behaviour that keeps a source's number
stable across settings loss and surfaces deleted-but-not-empty sources:

* a fresh source gets the lowest free number;
* a source whose ``input_source_numbers`` entry was lost reclaims its
  original number from ``source_registry`` (by path, then name) instead of
  drifting onto a new one and orphaning its own results — the renumbering
  bug;
* a deleted file whose result folders survive is emitted as a retired
  "ghost" row carrying its recorded name + result count;
* registry entries are garbage-collected once a number has neither a live
  file nor any results, so empty ghosts never accumulate.
"""

from __future__ import annotations

from pathlib import Path

from flextool.gui.data_models import ProjectSettings, SourceRecord
from flextool.gui.input_sources import InputSourceManager
from flextool.gui.settings_io import (
    load_project_settings,
    save_project_settings,
)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "input_sources").mkdir()
    (tmp_path / "output_parquet").mkdir()
    return tmp_path


def _add_file(project: Path, name: str) -> None:
    # An empty file is enough for numbering; scenario reading fails → status
    # "error", which numbering does not depend on.
    (project / "input_sources" / name).write_bytes(b"")


def _add_results(project: Path, scenario: str, number: int, count: int = 1) -> None:
    for i in range(count):
        (project / "output_parquet" / f"{scenario}{i}_{number}").mkdir()


def _mgr(project: Path) -> tuple[InputSourceManager, ProjectSettings]:
    settings = load_project_settings(project)
    return InputSourceManager(project, settings), settings


def test_fresh_source_gets_number_one(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_file(project, "examples.sqlite")
    mgr, settings = _mgr(project)

    sources = mgr.refresh()

    assert [s.number for s in sources] == [1]
    assert settings.input_source_numbers == {"examples.sqlite": 1}
    assert settings.source_registry["1"] == SourceRecord(
        name="examples.sqlite", path="input_sources/examples.sqlite"
    )


def test_reclaims_original_number_after_settings_loss(tmp_path: Path) -> None:
    """The renumbering bug: registry lets the file keep number 1."""
    project = _project(tmp_path)
    _add_file(project, "examples.sqlite")
    _add_results(project, "scen", number=1, count=3)
    # Registry remembers identity, but the name→number index was lost.
    settings = ProjectSettings()
    settings.source_registry = {
        "1": SourceRecord(name="examples.sqlite", path="input_sources/examples.sqlite")
    }
    mgr = InputSourceManager(project, settings)

    sources = mgr.refresh()

    live = [s for s in sources if not s.retired]
    assert [s.number for s in live] == [1]  # reclaimed, NOT bumped to 2
    assert settings.input_source_numbers == {"examples.sqlite": 1}
    assert not [s for s in sources if s.retired]  # no orphan: 1 is live again


def test_without_registry_orphan_results_bump_and_surface_ghost(tmp_path: Path) -> None:
    """Legacy orphan with no recorded identity: file bumps to 2, ghost for 1."""
    project = _project(tmp_path)
    _add_file(project, "examples.sqlite")
    _add_results(project, "scen", number=1, count=2)
    mgr, settings = _mgr(project)

    sources = mgr.refresh()

    live = [s for s in sources if not s.retired]
    ghosts = [s for s in sources if s.retired]
    assert [s.number for s in live] == [2]
    assert len(ghosts) == 1
    ghost = ghosts[0]
    assert ghost.number == 1
    assert ghost.status == "retired"
    assert ghost.result_count == 2
    assert ghost.name == "(source 1)"  # identity unknown


def test_deleted_file_with_results_becomes_named_ghost(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_results(project, "scen", number=1, count=1)
    settings = ProjectSettings()
    settings.source_registry = {
        "1": SourceRecord(name="examples.sqlite", path="input_sources/examples.sqlite")
    }
    mgr = InputSourceManager(project, settings)

    sources = mgr.refresh()

    assert len(sources) == 1
    ghost = sources[0]
    assert ghost.retired and ghost.number == 1
    assert ghost.name == "examples.sqlite"
    assert ghost.file_type == "sqlite"
    assert ghost.result_count == 1
    # Registry entry retained while results survive.
    assert "1" in settings.source_registry


def test_registry_gc_when_no_file_and_no_results(tmp_path: Path) -> None:
    project = _project(tmp_path)
    settings = ProjectSettings()
    settings.source_registry = {
        "1": SourceRecord(name="gone.sqlite", path="input_sources/gone.sqlite")
    }
    mgr = InputSourceManager(project, settings)

    sources = mgr.refresh()

    assert sources == []
    assert settings.source_registry == {}  # pruned: nothing backs number 1
    # And it is gone from the persisted file too.
    assert load_project_settings(project).source_registry == {}


def test_delete_then_readd_reclaims_number_and_relinks(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_file(project, "a.sqlite")
    _add_file(project, "b.sqlite")
    mgr, settings = _mgr(project)
    mgr.refresh()
    assert settings.input_source_numbers == {"a.sqlite": 1, "b.sqlite": 2}

    # b produced results, then its file is deleted (registry kept).
    _add_results(project, "scen", number=2, count=1)
    (project / "input_sources" / "b.sqlite").unlink()
    mgr.refresh()
    ghosts = [s for s in mgr.refresh() if s.retired]
    assert [g.number for g in ghosts] == [2]

    # Re-adding b reclaims number 2 (matching its results), not a fresh 3.
    _add_file(project, "b.sqlite")
    sources = mgr.refresh()
    live = {s.name: s.number for s in sources if not s.retired}
    assert live == {"a.sqlite": 1, "b.sqlite": 2}
    assert not [s for s in sources if s.retired]


def test_source_registry_round_trips_through_yaml(tmp_path: Path) -> None:
    project = _project(tmp_path)
    settings = ProjectSettings()
    settings.source_registry = {
        "1": SourceRecord(name="a.sqlite", path="input_sources/a.sqlite"),
        "2": SourceRecord(name="b.xlsx", path="../shared/b.xlsx"),
    }
    save_project_settings(project, settings)

    reloaded = load_project_settings(project)
    assert reloaded.source_registry == settings.source_registry


def test_malformed_registry_entries_are_dropped(tmp_path: Path) -> None:
    project = _project(tmp_path)
    (project / "settings.yaml").write_text(
        "source_registry:\n"
        "  '1': {name: ok.sqlite, path: input_sources/ok.sqlite}\n"
        "  notanumber: {name: bad.sqlite}\n"
        "  '2': brokenscalar\n"
    )
    reloaded = load_project_settings(project)
    assert set(reloaded.source_registry) == {"1"}
    assert reloaded.source_registry["1"].name == "ok.sqlite"


def test_delete_results_removes_folders_and_ghost_disappears(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_results(project, "scen", number=1, count=3)
    settings = ProjectSettings()
    settings.source_registry = {
        "1": SourceRecord(name="gone.sqlite", path="input_sources/gone.sqlite")
    }
    mgr = InputSourceManager(project, settings)
    assert [s.retired for s in mgr.refresh()] == [True]  # ghost present

    removed = mgr.delete_results(1)

    assert removed == 3
    assert list((project / "output_parquet").iterdir()) == []
    assert mgr.refresh() == []  # ghost gone
    assert settings.source_registry == {}  # entry GC'd


def test_relink_results_moves_folders_to_live_source(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_file(project, "examples.sqlite")
    # Orphaned results under number 1; examples will be live under 2.
    _add_results(project, "alpha", number=1, count=1)
    _add_results(project, "beta", number=1, count=1)
    mgr, settings = _mgr(project)
    sources = mgr.refresh()
    live_num = next(s.number for s in sources if not s.retired)
    assert live_num == 2  # bumped past orphaned 1

    moved, conflicts = mgr.relink_results(1, live_num)

    assert moved == 2 and conflicts == []
    parquet = project / "output_parquet"
    assert {p.name for p in parquet.iterdir()} == {"alpha0_2", "beta0_2"}
    # No more orphan: refresh shows only the live source.
    assert [s.retired for s in mgr.refresh()] == [False]


def test_relink_reports_conflicts_without_clobbering(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_file(project, "examples.sqlite")
    _add_results(project, "alpha", number=1, count=1)  # -> alpha0_1
    mgr, settings = _mgr(project)
    mgr.refresh()  # examples -> 2
    # Pre-existing target folder for the same scenario under source 2.
    (project / "output_parquet" / "alpha0_2").mkdir()

    moved, conflicts = mgr.relink_results(1, 2)

    assert moved == 0
    assert conflicts == ["alpha0"]
    # Source folder left in place — still an orphan after the conflict.
    assert (project / "output_parquet" / "alpha0_1").is_dir()


def test_reclaim_prefers_path_over_name(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _add_file(project, "data.sqlite")
    _add_results(project, "scen", number=5, count=1)
    settings = ProjectSettings()
    # Same name recorded at a different path under number 9, and the right
    # path under number 5. Path match must win.
    settings.source_registry = {
        "5": SourceRecord(name="data.sqlite", path="input_sources/data.sqlite"),
        "9": SourceRecord(name="data.sqlite", path="elsewhere/data.sqlite"),
    }
    _add_results(project, "other", number=9, count=1)
    mgr = InputSourceManager(project, settings)

    sources = mgr.refresh()

    live = [s for s in sources if not s.retired]
    assert [s.number for s in live] == [5]
