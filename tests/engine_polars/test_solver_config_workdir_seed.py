"""Tests for work-folder solver-config seeding.

``run_chain_from_db`` copies the active ``<solver>.opt`` files into
``<work_folder>/solver_config/`` so every run is self-contained and
repeatable, and so BOTH the in-process HiGHS path and the
subprocess/commercial path read the *same* files (the GUI's cwd is
frequently not the folder holding the project's ``highs.opt``).

The copy logic is factored into
:func:`flextool.engine_polars._orchestration._seed_work_folder_solver_config`,
tested here directly to avoid a heavy end-to-end solve.  Covered:

* the work folder gets ``solver_config/highs.opt`` and its contents
  match the chosen source;
* an already-present work-folder file is NOT clobbered;
* a solver with no operator file falls back to the bundled template;
* a solver with neither operator file nor template is skipped without
  error;
* the destination dir is returned and created.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flextool._resources import package_data_path
from flextool.engine_polars._orchestration import (
    _SOLVER_CONFIG_NAMES,
    _seed_work_folder_solver_config,
)

_LOGGER = logging.getLogger("test_solver_config_workdir_seed")


def test_seeds_from_source_dir(tmp_path: Path) -> None:
    """Operator ``<source>/highs.opt`` is copied into the work folder."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "highs.opt").write_text("presolve on\n# operator edit\n")

    work = tmp_path / "work"
    work.mkdir()

    dest_dir = _seed_work_folder_solver_config(work, source, _LOGGER)

    assert dest_dir == work / "solver_config"
    seeded = work / "solver_config" / "highs.opt"
    assert seeded.is_file()
    assert seeded.read_text() == "presolve on\n# operator edit\n"


def test_does_not_clobber_existing_work_folder_file(tmp_path: Path) -> None:
    """A ``<solver>.opt`` already in the work folder is left untouched."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "highs.opt").write_text("SOURCE VERSION\n")

    work = tmp_path / "work"
    dest_dir = work / "solver_config"
    dest_dir.mkdir(parents=True)
    (dest_dir / "highs.opt").write_text("HAND TWEAKED\n")

    _seed_work_folder_solver_config(work, source, _LOGGER)

    assert (dest_dir / "highs.opt").read_text() == "HAND TWEAKED\n"


def test_falls_back_to_bundled_template(tmp_path: Path) -> None:
    """With no operator file, the bundled ``.opt.template`` seeds the file."""
    source = tmp_path / "src"  # deliberately does not exist
    work = tmp_path / "work"
    work.mkdir()

    _seed_work_folder_solver_config(work, source, _LOGGER)

    seeded = work / "solver_config" / "highs.opt"
    assert seeded.is_file()
    template = Path(package_data_path("solver_config/highs.opt.template"))
    assert seeded.read_text() == template.read_text()


def test_all_bundled_solvers_seeded(tmp_path: Path) -> None:
    """Every solver with a bundled template lands in the work folder."""
    source = tmp_path / "src"  # does not exist -> template path
    work = tmp_path / "work"
    work.mkdir()

    _seed_work_folder_solver_config(work, source, _LOGGER)

    for solver in _SOLVER_CONFIG_NAMES:
        template = Path(package_data_path(f"solver_config/{solver}.opt.template"))
        seeded = work / "solver_config" / f"{solver}.opt"
        # All five ship a template, so all five must be seeded.
        assert template.is_file()
        assert seeded.is_file()
        assert seeded.read_text() == template.read_text()


def test_missing_source_and_template_skipped(tmp_path: Path, monkeypatch) -> None:
    """A solver with neither operator file nor template is skipped, no crash."""
    import flextool.engine_polars._orchestration as orch

    # Force the template lookup to fail for one synthetic solver name so we
    # exercise the "neither source nor template" branch.
    monkeypatch.setattr(orch, "_SOLVER_CONFIG_NAMES", ("nosuchsolver",))

    source = tmp_path / "src"  # does not exist
    work = tmp_path / "work"
    work.mkdir()

    dest_dir = _seed_work_folder_solver_config(work, source, _LOGGER)

    assert dest_dir == work / "solver_config"
    # Dir is created but the unknown solver produced no file.
    assert dest_dir.is_dir()
    assert not (dest_dir / "nosuchsolver.opt").exists()
