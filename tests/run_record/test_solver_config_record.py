"""The run record names the solver option files a run used (item 234).

Hashes are worked out in the test from the literal bytes with ``hashlib``, not by the module's helper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flextool._solver_config_source import (
    REASON_FLAG,
    SolverConfigSource,
    snapshot_solver_option_files,
)
from flextool.project_package import scrub_run_status
from flextool.run_record._owner import well_formed

_HIGHS = b"mip_rel_gap=0.01\n"


def _record(tmp_path: Path, highs_bytes: bytes, folder: Path | None = None) -> dict:
    cfg = folder or tmp_path / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "highs.opt").write_bytes(highs_bytes)
    files = snapshot_solver_option_files(SolverConfigSource(cfg, REASON_FLAG), tmp_path / "work")
    return files.record()


def test_record_names_the_folder_the_files_and_their_hashes(tmp_path: Path) -> None:
    rec = _record(tmp_path, _HIGHS)
    cfg = (tmp_path / "cfg").resolve()
    assert rec["dir"] == str(cfg) and rec["reason"] == "--solver-config-dir" and rec["skipped"] == []
    assert rec["snapshot_dir"] == str((tmp_path / "work" / "solve_data" / "solver_config").resolve())
    assert rec["files"]["highs.opt"] == {
        "source": str(cfg / "highs.opt"),
        "sha256": hashlib.sha256(_HIGHS).hexdigest(),
        "default": False,
    }
    # Not in the folder: the packaged template, which is FlexTool's unedited default.
    assert rec["files"]["gurobi.opt"]["source"] == "FlexTool default"
    assert rec["files"]["gurobi.opt"]["default"] is True
    assert set(rec["files"]) == {"highs.opt", "gurobi.opt", "cplex.opt", "xpress.opt", "copt.opt"}


def test_crlf_file_gives_the_same_hash(tmp_path: Path) -> None:
    rec = _record(tmp_path, b"mip_rel_gap=0.01\r\n")
    assert rec["files"]["highs.opt"]["sha256"] == hashlib.sha256(_HIGHS).hexdigest()


def test_relative_folder_is_stored_absolute(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "highs.opt").write_bytes(_HIGHS)
    monkeypatch.chdir(tmp_path)
    rec = snapshot_solver_option_files(SolverConfigSource(Path("cfg"), REASON_FLAG), Path("work")).record()
    assert rec["dir"] == str((tmp_path / "cfg").resolve())
    assert rec["files"]["highs.opt"]["source"] == str((tmp_path / "cfg" / "highs.opt").resolve())
    assert rec["snapshot_dir"] == str((tmp_path / "work" / "solve_data" / "solver_config").resolve())


def test_well_formed_with_and_without_the_block() -> None:
    base = {"state": "succeeded"}
    assert well_formed(base)
    assert well_formed({**base, "solver_config": None})
    assert well_formed({**base, "solver_config": {"dir": None, "files": {}}})
    assert not well_formed({**base, "solver_config": []})


def test_scrub_removes_the_workspace_path_and_keeps_the_names(tmp_path: Path) -> None:
    """The FlexTool window's folder is ``<workspace>/solver_config``, outside ``<workspace>/projects/<P>``."""
    workspace = tmp_path / "ws"
    project = workspace / "projects" / "P"
    project.mkdir(parents=True)
    rec = _record(tmp_path, _HIGHS, folder=workspace / "solver_config")
    scrubbed = scrub_run_status({"state": "succeeded", "solver_config": rec}, project)["solver_config"]
    assert str(tmp_path.resolve()) not in json.dumps(scrubbed)
    assert scrubbed["dir"].endswith("/solver_config")
    assert scrubbed["dir"].startswith(("<path>/", "~/"))
    assert scrubbed["files"]["highs.opt"]["source"].endswith("/highs.opt")
    assert scrubbed["files"]["highs.opt"]["sha256"] == hashlib.sha256(_HIGHS).hexdigest()

    # A folder inside the project becomes <project>/...
    rec_in = _record(tmp_path, _HIGHS, folder=project / "solver_config")
    scrubbed_in = scrub_run_status({"state": "succeeded", "solver_config": rec_in}, project.resolve())
    assert scrubbed_in["solver_config"]["dir"] == "<project>/solver_config"
