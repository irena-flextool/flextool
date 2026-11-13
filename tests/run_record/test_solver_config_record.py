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
        "reason": "--solver-config-dir",
        "trusted": True,
        "sha256": hashlib.sha256(_HIGHS).hexdigest(),
        "default": False,
    }
    assert rec["layers"] == [{"dir": str(cfg), "reason": "--solver-config-dir", "trusted": True}]
    # Not in the folder: the packaged template, which is FlexTool's unedited default.
    assert rec["files"]["gurobi.opt"]["source"] == "FlexTool default"
    assert rec["files"]["gurobi.opt"]["reason"] == "FlexTool default"
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
    # A scrubbed path keeps the platform's separator (``<path>\\solver_config`` on Windows).
    def posix(text: str) -> str:
        return text.replace("\\", "/")

    assert posix(scrubbed["dir"]).endswith("/solver_config")
    assert posix(scrubbed["dir"]).startswith(("<path>/", "~/"))
    assert posix(scrubbed["files"]["highs.opt"]["source"]).endswith("/highs.opt")
    assert scrubbed["files"]["highs.opt"]["sha256"] == hashlib.sha256(_HIGHS).hexdigest()

    # A folder inside the project becomes <project>/...
    rec_in = _record(tmp_path, _HIGHS, folder=project / "solver_config")
    scrubbed_in = scrub_run_status({"state": "succeeded", "solver_config": rec_in}, project.resolve())
    assert posix(scrubbed_in["solver_config"]["dir"]) == "<project>/solver_config"


def test_record_names_the_folder_of_each_file_across_the_ladder(tmp_path: Path) -> None:
    """Two folders, file by file: highs.opt from the upper folder, gurobi.opt from the lower one (the upper has
    none), cplex.opt FlexTool's default; each file records its own source and that folder's reason."""
    upper, lower = tmp_path / "ws" / "projects" / "P" / "solver_config", tmp_path / "ws" / "solver_config"
    upper.mkdir(parents=True)
    lower.mkdir(parents=True)
    (upper / "highs.opt").write_bytes(_HIGHS)
    (lower / "highs.opt").write_bytes(b"mip_rel_gap=0.5\n")
    (lower / "gurobi.opt").write_bytes(b"Threads 2\n")
    source = SolverConfigSource(upper, REASON_FLAG, lower=((lower, REASON_FLAG),))
    rec = snapshot_solver_option_files(source, tmp_path / "work").record()
    assert rec["dir"] == str(upper.resolve())
    assert rec["layers"] == [{"dir": str(upper.resolve()), "reason": REASON_FLAG, "trusted": True},
                             {"dir": str(lower.resolve()), "reason": REASON_FLAG, "trusted": True}]
    assert rec["files"]["highs.opt"]["source"] == str((upper / "highs.opt").resolve())
    assert rec["files"]["highs.opt"]["sha256"] == hashlib.sha256(_HIGHS).hexdigest()
    assert rec["files"]["gurobi.opt"]["source"] == str((lower / "gurobi.opt").resolve())
    assert rec["files"]["gurobi.opt"]["sha256"] == hashlib.sha256(b"Threads 2\n").hexdigest()
    assert (rec["files"]["cplex.opt"]["source"], rec["files"]["cplex.opt"]["reason"]) == (
        "FlexTool default", "FlexTool default")
    snapshot = tmp_path / "work" / "solve_data" / "solver_config"
    assert (snapshot / "highs.opt").read_bytes() == _HIGHS
    assert (snapshot / "gurobi.opt").read_bytes() == b"Threads 2\n"


def test_scrub_removes_every_folder_of_a_several_folder_value(tmp_path: Path) -> None:
    """A run record holds several folders as a list (the run's flags, each folder its own argument) and, from
    the environment, as one text joined with the path separator: no folder of either keeps its full path."""
    import os

    project = tmp_path / "ws" / "projects" / "P"
    project.mkdir(parents=True)
    joined = os.pathsep.join(["/srv/other/solver_config", "/opt/team/solver_config"])
    status = {
        "argv": ["--solver-config-dir", "/srv/other/solver_config", "--solver-config-dir", "/opt/team/solver_config"],
        "options_resolved": {"solver_config_dir": {"value": ["/srv/other/solver_config", "/opt/team/solver_config"],
                                                   "source": "cli"}},
        "env_text": joined,
    }
    text = json.dumps(scrub_run_status(status, project))
    assert "/srv/other" not in text and "/opt/team" not in text, text
    assert text.count("<path>/solver_config") == 6, text
