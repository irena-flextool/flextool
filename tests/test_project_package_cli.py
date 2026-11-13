"""Project packages (C-9): ``flextool.project_package`` and the ``flextool-project`` command line.

The fixture project is built in ``tmp_path``: an input database from the schema JSON, an external input database
(source number 2, frozen), a ``settings.yaml`` with machine-specific settings, a workspace ``solver_config`` and two
result runs with small parquet files.  ``B_2`` is stale: its external input changed after the run.  No solve.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from flextool import project_package as pp
from flextool.cli import cmd_project_package as cli
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import frozen as fz

SCHEMA_JSON = Path(__file__).resolve().parent.parent / "flextool" / "schemas" / "spinedb_schema.json"
MACHINE_SETTINGS = {
    "execution_limits": {"max_cores_per_job": 8, "memory_cap_per_job_gb": 12.0, "system_reserve_gb": 0.5,
                         "swap_allowance_gb": 0.0},
    "max_workers": 4,
    "scenario_resource_history": {"A_1": {"peak_rss_mb": 900.0, "runtime_s": 12.0, "last_run": "2026-11-01"}},
}
T_OLD = 1_700_000_000_123_456_789   # ns, kept exactly by the round trip
T_NEW = 1_750_000_000_987_654_321


@pytest.fixture(scope="module")
def db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from flextool.update_flextool.initialize_database import initialize_database

    path = tmp_path_factory.mktemp("package_db") / "model.sqlite"
    initialize_database(str(SCHEMA_JSON), str(path))
    return path


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9).astimezone().isoformat(timespec="seconds")


def _run(folder: Path, value: float, db_mtime_ns: int, state: str = "succeeded", **extra) -> None:
    folder.mkdir(parents=True)
    pl.DataFrame({"t": ["t0001", "t0002"], "v": [value, value * 2]}).write_parquet(folder / "unit_flow_dt.parquet")
    project = folder.parent.parent
    (folder / "_run_status.json").write_text(json.dumps({
        "state": state, "verdict": "optimal", "process": {"host": "builder-host", "pid": 4242},
        "argv": ["/opt/work/flextool3__toolbox/cmd_run_flextool.py", "http://127.0.0.1:48767"],
        "input": {"db_mtime": _iso(db_mtime_ns), "db_path": str(project / "input_sources" / "model.sqlite")},
        "output": {"location": str(project), "parquet_dir": str(folder)}, **extra}))
    (folder / "_run.log").write_text(f"solving in {project}\n")
    (folder / "plot_plans").mkdir()
    (folder / "plot_plans" / "plan.json").write_text("{}")


@pytest.fixture()
def workspace(tmp_path: Path, db_template: Path) -> dict:
    projects = tmp_path / "projects"
    project = projects / "Proj"
    for sub in ("input_sources", "output_parquet", "output_csv/A_1", "work", "converted"):
        (project / sub).mkdir(parents=True)
    model = project / "input_sources" / "model.sqlite"
    shutil.copy2(db_template, model)
    external = tmp_path / "data" / "ext.sqlite"
    external.parent.mkdir()
    shutil.copy2(db_template, external)
    fz.freeze(external, "Reference 1.0", today="2026-11-10")
    settings = {
        "auto_generate_scen_plots": True,
        "solver_log_level": "verbose",
        "input_source_numbers": {"model.sqlite": 1, "ext.sqlite": 2},
        "source_registry": {"1": {"name": "model.sqlite", "path": "input_sources/model.sqlite"},
                            "2": {"name": "ext.sqlite", "path": "../../data/ext.sqlite"}},
        "external_refs": {"ext.sqlite": "../../data/ext.sqlite"},
        "checked_input_sources": ["model.sqlite", "ext.sqlite"],
        "scenario_order": ["A", "B"],
        **MACHINE_SETTINGS,
    }
    (project / "settings.yaml").write_text(yaml.dump(settings, sort_keys=False))
    (project / "plot_settings.yaml").write_text("colors: {}\n")
    (project / "results.sqlite").write_bytes(b"x")
    (project / "output_csv" / "A_1" / "flow.csv").write_text("a,b\n")
    (project / "work" / "scratch.txt").write_text("scratch")
    _run(project / "output_parquet" / "A_1", 1.0, T_OLD)
    _run(project / "output_parquet" / "B_2", 3.0, T_OLD)
    (project / "output_parquet" / "_axis_bounds.json").write_text(json.dumps({
        "unit_flow_dt": {"chunks": {"<untitled>": {"A_1": [0.0, 2.0], "B_2": [0.0, 6.0]}}}}))
    config = tmp_path / "solver_config"
    config.mkdir()
    (config / "highs.opt").write_text("mip_rel_gap = 0.01\n")
    for path in project.rglob("*"):
        if path.is_file() and path.name != "settings.yaml":
            os.utime(path, ns=(T_OLD, T_OLD))
    os.utime(external, ns=(T_NEW, T_NEW))       # changed after the run B_2
    return {"root": tmp_path, "projects": projects, "project": project, "config": config, "external": external}


def _export(ws: dict, runs: list[str] | None, name: str = "Proj.flextool.zip", **kw) -> tuple[Path, pp.Manifest]:
    out = ws["root"] / "out" / name
    manifest = pp.export_project(ws["project"], out, runs, solver_config_dir=ws["config"], **kw)
    return out, manifest


def _rewrite(src: Path, dst: Path, change) -> None:
    """Copy the zip *src* to *dst*; ``change(name, data)`` returns the new data, or ``None`` to keep it."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info)
            new = change(info.filename, data)
            zout.writestr(info, data if new is None else new)


def test_round_trip_keeps_files_times_numbers_and_frozen(workspace: dict) -> None:
    project, projects = workspace["project"], workspace["projects"]
    out, manifest = _export(workspace, ["A_1", "B_2"])
    assert manifest.stale_runs == ["B_2"]
    assert manifest.bundled_external == {"ext.sqlite": "ext.sqlite"}
    assert manifest.db_versions == {"input_sources/model.sqlite": FLEXTOOL_DB_VERSION,
                                    "input_sources/ext.sqlite": FLEXTOOL_DB_VERSION}
    assert manifest.frozen == {"input_sources/ext.sqlite": f"Reference 1.0|{FLEXTOOL_DB_VERSION}|2026-11-10"}
    assert manifest.solver_config == ["highs.opt"]

    report = pp.import_project(out, projects, "Copy", solver_config_dir=workspace["config"])
    copy = report.project_path
    assert copy == projects / "Copy" and not list(projects.glob(".*partial"))

    pairs = {f"input_sources/{p.name}": p for p in (project / "input_sources").iterdir()}
    pairs["input_sources/ext.sqlite"] = workspace["external"]
    pairs["plot_settings.yaml"] = project / "plot_settings.yaml"
    pairs["solver_config/highs.opt"] = workspace["config"] / "highs.opt"
    for path in (project / "output_parquet").rglob("*"):
        if path.is_file() and path.name not in ("_run_status.json", "_run.log", "_axis_bounds.json"):
            pairs[path.relative_to(project).as_posix()] = path
    for rel in ("output_parquet/A_1/_run_status.json", "output_parquet/_axis_bounds.json"):
        assert (copy / rel).stat().st_mtime_ns == (project / rel).stat().st_mtime_ns, rel
    assert not (copy / "output_parquet" / "A_1" / "_run.log").exists()
    for rel, original in pairs.items():
        imported = copy / rel
        assert imported.read_bytes() == original.read_bytes(), rel
        assert imported.stat().st_mtime_ns == original.stat().st_mtime_ns, rel
    for arcname, facts in manifest.files.items():
        rel = arcname.removeprefix("Proj/")
        assert pp._hash_file(copy / rel) == facts["sha256"], rel
    assert fz.read_frozen(copy / "input_sources" / "ext.sqlite").label == "Reference 1.0"

    original = yaml.safe_load((project / "settings.yaml").read_text())
    shipped = yaml.safe_load((copy / "settings.yaml").read_text())
    for key in pp.DROPPED_SETTINGS:
        assert key not in shipped
        original.pop(key)
    original["source_registry"]["2"]["path"] = "input_sources/ext.sqlite"
    assert shipped == original

    # The bundled external database keeps its source number 2, so the shipped B_2 pairs with a re-run of it.
    from flextool.gui.input_sources import InputSourceManager
    from flextool.gui.settings_io import load_project_settings

    sources = InputSourceManager(copy, load_project_settings(copy)).refresh()
    assert {s.name: s.number for s in sources if not s.retired} == {"model.sqlite": 1, "ext.sqlite": 2}
    assert (workspace["config"] / "highs.opt").read_text() == "mip_rel_gap = 0.01\n"
    assert report.solver_config == [("highs.opt", "same")]
    assert not (copy / "imported_solver_config").exists()


def test_results_exclusion_and_regenerated_parts(workspace: dict) -> None:
    out, manifest = _export(workspace, ["A_1"])
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert zf.getinfo("Proj/output_parquet/A_1/unit_flow_dt.parquet").compress_type == zipfile.ZIP_STORED
    assert "Proj/output_parquet/A_1/unit_flow_dt.parquet" in names
    assert "Proj/output_parquet/_axis_bounds.json" in names
    with zipfile.ZipFile(out) as zf:     # the left-out run's name and ranges do not travel
        bounds = json.loads(zf.read("Proj/output_parquet/_axis_bounds.json"))
    assert bounds == {"unit_flow_dt": {"chunks": {"<untitled>": {"A_1": [0.0, 2.0]}}}}
    assert not any(n.startswith("Proj/output_parquet/B_2/") for n in names)
    assert not any(n.startswith(("Proj/results.sqlite", "Proj/output_csv", "Proj/work", "Proj/converted"))
                   for n in names)
    none_out, none_manifest = _export(workspace, None, "none.flextool.zip")
    with zipfile.ZipFile(none_out) as zf:
        assert not any(n.startswith("Proj/output_parquet") for n in zf.namelist())
    assert none_manifest.results == [] and manifest.results == ["A_1"]


def test_name_clash_exit_code_and_suggestions(workspace: dict, capsys: pytest.CaptureFixture) -> None:
    out, _ = _export(workspace, None)
    projects = workspace["projects"]
    code = cli.main(["import", str(out), "--projects-dir", str(projects)])
    assert code == cli.EXIT_NAME_CLASH
    assert "Proj_2" in capsys.readouterr().err
    assert pp.suggest_name(projects, "Proj") == "Proj_2"
    (projects / "proj_2").mkdir()               # a clash is case-insensitive
    assert pp.suggest_name(projects, "Proj") == "Proj_3"
    assert cli.main(["import", str(out), "--projects-dir", str(projects), "--name", "Proj_3"]) == cli.EXIT_OK
    assert (projects / "Proj_3" / "input_sources" / "model.sqlite").is_file()


def _bump_db_version(name: str, data: bytes) -> bytes | None:
    if name != pp.MANIFEST_NAME:
        return None
    manifest = json.loads(data)
    manifest["db_versions"]["input_sources/model.sqlite"] = FLEXTOOL_DB_VERSION + 1
    return json.dumps(manifest).encode()


def _newer_flextool(name: str, data: bytes) -> bytes | None:
    if name != pp.MANIFEST_NAME:
        return None
    manifest = json.loads(data)
    manifest["flextool"] = "999.0.0"
    return json.dumps(manifest).encode()


def _tamper(name: str, data: bytes) -> bytes | None:
    if not name.endswith("model.sqlite"):
        return None
    return data[:100] + bytes([data[100] ^ 1]) + data[101:]


def _zip_slip(src: Path, dst: Path) -> None:
    _rewrite(src, dst, lambda _n, _d: None)
    with zipfile.ZipFile(dst, "a") as zf:
        zf.writestr("Proj/../evil.txt", b"evil")
        manifest = json.loads(zf.read(pp.MANIFEST_NAME))
    manifest["files"]["Proj/../evil.txt"] = {"sha256": "0", "bytes": 4, "mtime_ns": 0}
    with zipfile.ZipFile(dst, "a") as zf:
        zf.writestr(pp.MANIFEST_NAME, json.dumps(manifest))   # the later entry wins on read


@pytest.mark.parametrize(("case", "refused", "message"), [
    ("tampered", True, "checksum does not match"),
    ("zip_slip", True, "outside the project folder"),
    ("db_newer", True, "upgrade FlexTool"),
    ("flextool_newer", False, "newer than this FlexTool"),
])
def test_integrity_and_version_gate(workspace: dict, case: str, refused: bool, message: str) -> None:
    out, _ = _export(workspace, ["A_1"])
    bad = out.with_name("bad.flextool.zip")
    if case == "zip_slip":
        with pytest.warns(UserWarning):   # zipfile warns about the duplicate manifest name
            _zip_slip(out, bad)
    else:
        _rewrite(out, bad, {"tampered": _tamper, "db_newer": _bump_db_version,
                            "flextool_newer": _newer_flextool}[case])
    projects = workspace["projects"]
    if refused:
        with pytest.raises(pp.PackageError, match=message):
            pp.import_project(bad, projects, "Imported")
        assert sorted(p.name for p in projects.iterdir()) == ["Proj"]   # no partial folder left
        assert not (workspace["root"] / "evil.txt").exists()
    else:
        report = pp.import_project(bad, projects, "Imported")
        assert any(message in w for w in report.warnings)
        assert (projects / "Imported" / "settings.yaml").is_file()


def test_verify_detects_a_modified_file(workspace: dict, capsys: pytest.CaptureFixture) -> None:
    out, _ = _export(workspace, ["A_1", "B_2"])
    assert cli.main(["verify", str(out)]) == cli.EXIT_OK
    text = capsys.readouterr().out
    assert "files match their checksums" in text and "not signed" in text
    assert "B_2" in text and "changed after the run" in text
    assert "pip install flextool==" in text and "polar-high==" in text and "highspy==" in text
    bad = out.with_name("bad.flextool.zip")
    _rewrite(out, bad, _tamper)
    assert cli.main(["verify", str(bad)]) == cli.EXIT_PROBLEM
    assert "input_sources/model.sqlite: the checksum does not match" in capsys.readouterr().err


def test_cli_export_results_and_solver_hint(workspace: dict, capsys: pytest.CaptureFixture) -> None:
    out = workspace["root"] / "cli.flextool.zip"
    code = cli.main(["export", str(workspace["project"]), "-o", str(out), "--results", "B_2", "--no-external",
                     "--solver-config-dir", str(workspace["config"])])
    assert code == cli.EXIT_OK
    assert "changed after the run" in capsys.readouterr().out
    manifest = pp.read_manifest(out)
    assert manifest.results == ["B_2"] and manifest.external_not_included == ["ext.sqlite"]
    (workspace["config"] / "highs.opt").write_text("mip_rel_gap = 0.05\n")   # the receiver's own options
    report = pp.import_project(out, workspace["projects"], "Other", solver_config_dir=workspace["config"])
    assert report.solver_config == [("highs.opt", "different")]
    assert "-mip_rel_gap = 0.05" in report.solver_config_diffs["highs.opt"]
    assert "+mip_rel_gap = 0.01" in "\n".join(pp.describe_import(report))
    assert report.solver_config_hint == (
        f"The shipped solver options are in {workspace['projects'] / 'Other' / 'solver_config'}. Runs of this project "
        f"use them ahead of the workspace's ({workspace['config']}), file by file; delete a file there to use the "
        "workspace's.")


def _status_in(zip_path: Path, run: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(f"Proj/output_parquet/{run}/_run_status.json"))


def test_run_records_scrubbed_by_default(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "builder-host")
    project = str(workspace["project"])
    out, manifest = _export(workspace, ["A_1"])
    status = _status_in(out, "A_1")
    text = json.dumps(status)
    assert project not in text and "builder-host" not in text and "/opt/work" not in text
    # The scrub keeps the platform's separator (``<project>\\output_parquet\\A_1`` on Windows).
    assert set(status["output"]) == {"location", "parquet_dir"} and status["output"]["location"] == "<project>"
    assert status["output"]["parquet_dir"].replace("\\", "/") == "<project>/output_parquet/A_1"
    assert status["process"] == {"host": None, "pid": 4242}
    assert status["argv"][0] == "<path>/cmd_run_flextool.py" and status["argv"][1] == "http://127.0.0.1:48767"
    assert status["state"] == "succeeded" and status["input"]["db_mtime"]   # what the readers use is kept
    with zipfile.ZipFile(out) as zf:
        assert "Proj/output_parquet/A_1/_run.log" not in zf.namelist()
    assert manifest.run_records_scrubbed

    kept, kept_manifest = _export(workspace, ["A_1"], "kept.flextool.zip", scrub_run_records=False)
    assert _status_in(kept, "A_1")["output"]["location"] == project
    with zipfile.ZipFile(kept) as zf:
        assert "Proj/output_parquet/A_1/_run.log" in zf.namelist()
    assert not pp.read_manifest(kept).run_records_scrubbed and not kept_manifest.run_records_scrubbed


def test_running_run_refused_and_left_out_of_all(workspace: dict, capsys: pytest.CaptureFixture) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    _run(workspace["project"] / "output_parquet" / "C_1", 5.0, T_OLD, state="running", heartbeat_at=now)
    runs = {r.name: r for r in pp.list_runs(workspace["project"])}
    assert runs["C_1"].active and not runs["A_1"].active
    with pytest.raises(pp.PackageError, match="still being written.*C_1"):
        _export(workspace, ["A_1", "C_1"])
    out = workspace["root"] / "all.flextool.zip"
    assert cli.main(["export", str(workspace["project"]), "-o", str(out), "--results", "all",
                     "--solver-config-dir", str(workspace["config"])]) == cli.EXIT_OK
    assert "Left out, still being written: C_1" in capsys.readouterr().out
    assert pp.read_manifest(out).results == ["A_1", "B_2"]


def test_wal_database_copied_consistently(workspace: dict) -> None:
    import sqlite3

    model = workspace["project"] / "input_sources" / "model.sqlite"
    writer = sqlite3.connect(model)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE probe (x INTEGER)")
        writer.execute("INSERT INTO probe VALUES (7)")
        writer.commit()                              # committed, but only in model.sqlite-wal
        assert (model.parent / "model.sqlite-wal").stat().st_size > 0
        out, manifest = _export(workspace, None)
    finally:
        writer.close()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        shipped = workspace["root"] / "shipped.sqlite"
        shipped.write_bytes(zf.read("Proj/input_sources/model.sqlite"))
    assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in names)
    with closing(sqlite3.connect(shipped)) as conn:
        assert conn.execute("SELECT x FROM probe").fetchall() == [(7,)]
    assert manifest.db_versions["input_sources/model.sqlite"] == FLEXTOOL_DB_VERSION


def _handmade(dst: Path, members: dict[str, bytes], **facts) -> None:
    import hashlib

    files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mtime_ns": T_OLD, **facts}
             for name, data in members.items()}
    with zipfile.ZipFile(dst, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr(pp.MANIFEST_NAME, json.dumps({"format": pp.FORMAT, "format_version": 1, "project": "Proj",
                                                  "files": files}))


@pytest.mark.parametrize(("members", "facts", "message"), [
    ({"Proj/input_sources/a.sqlite:ads": b"x"}, {}, "character Windows does not allow"),
    ({"Proj/input_sources/CON.txt": b"x"}, {}, "Windows reserves"),
    ({"Proj/input_sources/a. ": b"x"}, {}, "dot or a space"),
    ({"Proj/a.txt": b"x", "Proj/A.txt": b"y"}, {}, "differ only in case"),
    ({"Proj/a": b"x", "Proj/a/b": b"y"}, {}, "both a file and the folder"),
    ({"Proj/a.txt": b"x"}, {"mtime_ns": 10**30}, "not a time"),
])
def test_unportable_members_refused_before_writing(tmp_path: Path, capsys: pytest.CaptureFixture, members: dict,
                                                   facts: dict, message: str) -> None:
    bad = tmp_path / "bad.flextool.zip"
    _handmade(bad, members, **facts)
    projects = tmp_path / "projects"
    with pytest.raises(pp.PackageError, match=message):
        pp.import_project(bad, projects)
    assert cli.main(["import", str(bad), "--projects-dir", str(projects)]) == cli.EXIT_REFUSED
    assert message in capsys.readouterr().err
    assert not projects.exists() or not any(projects.iterdir())
    assert pp.invalid_name_reason("NUL") is not None and pp.invalid_name_reason("a:b") is not None


def test_export_refuses_unportable_name_and_existing_file(workspace: dict, capsys: pytest.CaptureFixture) -> None:
    out, _ = _export(workspace, None)
    with pytest.raises(pp.PackageExistsError):
        _export(workspace, None)
    args = ["export", str(workspace["project"]), "-o", str(out), "--solver-config-dir", str(workspace["config"])]
    assert cli.main(args) == cli.EXIT_PROBLEM
    assert "Use --force" in capsys.readouterr().err
    assert cli.main([*args, "--force"]) == cli.EXIT_OK
    blocker = workspace["root"] / "a_file"
    blocker.write_text("x")
    assert cli.main(["export", str(workspace["project"]), "-o", str(blocker / "p.flextool.zip")]) == cli.EXIT_PROBLEM
    assert "Unable to write the package" in capsys.readouterr().err      # an OSError, said without a traceback
    (workspace["project"] / "input_sources" / "aux.xlsx").write_bytes(b"x")
    with pytest.raises(pp.PackageError, match="Windows reserves"):
        _export(workspace, None, "aux.flextool.zip")
    partial = out.with_name(out.name + ".partial")
    partial.write_bytes(b"x")
    assert "unfinished package" in pp.verify_package(partial).problems[0]


def test_disk_space_external_left_out_and_partial_hidden(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    import collections

    out, manifest = _export(workspace, ["A_1"], include_external=False)
    assert manifest.external_not_included == ["ext.sqlite"]
    projects = workspace["projects"]
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(pp.shutil, "disk_usage", lambda _p: usage(10**9, 10**9, 1000))
    with pytest.raises(pp.PackageError, match="MB free"):
        pp.import_project(out, projects, "Small")
    monkeypatch.undo()
    report = pp.import_project(out, projects, "Recv")
    assert any("ext.sqlite was not in the package" in w for w in report.warnings)
    shipped = yaml.safe_load((report.project_path / "settings.yaml").read_text())
    assert shipped["source_registry"]["2"] == {"name": "ext.sqlite", "path": ""}   # the sender's layout is gone

    from flextool.gui import project_utils

    (projects / ".Stopped.partial").mkdir()
    monkeypatch.chdir(workspace["root"])
    assert project_utils.list_projects() == ["Proj", "Recv"]


@pytest.mark.parametrize(("member", "message"), [
    ("Proj/flextool/__init__.py", "not part of a FlexTool project"),
    ("Proj/input_sources/flextool/__init__.py", "program or script"),
    ("Proj/output_parquet/A_1/run.pth", "program or script"),
    ("Proj/input_sources/setup.BAT", "program or script"),
    ("Proj/solver_config/highs.sh", "program or script"),
    ("Proj/solver_config/sub/highs.opt", "holds only .opt files"),
    ("Proj/solver_config/cplex.opt.review", "holds only .opt files"),
    ("Proj/notes.txt", "not part of a FlexTool project"),
    # G3: solver option folders FlexTool would read by itself (next to an input database), in any case.
    ("Proj/input_sources/solver_config/gurobi.opt", "only in its top-level solver_config/"),
    ("Proj/input_sources/Solver_Config/gurobi.opt", "only in its top-level solver_config/"),
    ("Proj/input_sources/sub/SOLVER_CONFIG/highs.opt", "only in its top-level solver_config/"),
    ("Proj/input_sources/imported_solver_config/cplex.opt", "only in its top-level solver_config/"),
    ("Proj/output_parquet/A_1/solver_config/xpress.opt", "only in its top-level solver_config/"),
    ("Proj/Solver_Config/gurobi.opt", "only in its top-level solver_config/"),
    ("Proj/imported_solver_config/gurobi.opt", "only in its top-level solver_config/"),
])
def test_code_and_unknown_files_in_a_package_are_refused(tmp_path: Path, member: str, message: str) -> None:
    """A package holds only what export writes; code never unpacks into the project (it could replace FlexTool)."""
    bad = tmp_path / "bad.flextool.zip"
    _handmade(bad, {"Proj/settings.yaml": b"{}\n", member: b"open('marker', 'w')\n"})
    projects = tmp_path / "projects"
    with pytest.raises(pp.PackageError, match=message):
        pp.import_project(bad, projects)
    assert not projects.exists() or not any(projects.iterdir())
    assert not pp.verify_package(bad).ok


def test_export_leaves_out_programs_and_the_package_imports(workspace: dict) -> None:
    (workspace["project"] / "input_sources" / "helper.py").write_text("print('hi')\n")
    out, manifest = _export(workspace, ["A_1"])
    assert not any(name.endswith("helper.py") for name in manifest.files)
    assert any("helper.py is a program or script" in w for w in manifest.export_warnings)
    report = pp.import_project(out, workspace["projects"], "Recv")
    assert not (report.project_path / "input_sources" / "helper.py").exists()


def test_export_leaves_out_a_solver_option_folder_inside_the_project(workspace: dict) -> None:
    """A user's own solver_config/ next to a project database stays on their computer (import would refuse it);
    the package still carries the workspace's solver_config/ files and imports them into the project's
    solver_config/."""
    nested = workspace["project"] / "input_sources" / "solver_config"
    nested.mkdir()
    (nested / "gurobi.opt").write_text("Threads 2\n")
    out, manifest = _export(workspace, ["A_1"])
    assert not any("input_sources/solver_config" in name for name in manifest.files)
    assert any("input_sources/solver_config/gurobi.opt is in a solver option folder" in w
               for w in manifest.export_warnings)
    assert "solver_config/highs.opt" in {name.split("/", 1)[1] for name in manifest.files}
    report = pp.import_project(out, workspace["projects"], "Recv")
    assert not (report.project_path / "input_sources" / "solver_config").exists()
    assert (report.project_path / "solver_config" / "highs.opt").is_file()


def test_import_drops_input_files_outside_the_project(tmp_path: Path) -> None:
    """A hand-made package cannot point an input source at a file elsewhere on this computer."""
    settings = {
        "input_source_numbers": {"model.sqlite": 1, "far.sqlite": 2, "abs.sqlite": 3, "inside.sqlite": 4},
        "source_registry": {"1": {"name": "model.sqlite", "path": "input_sources/model.sqlite"},
                            "2": {"name": "far.sqlite", "path": "../../secret/far.sqlite"},
                            "3": {"name": "abs.sqlite", "path": "/etc/abs.sqlite"}},
        "external_refs": {"far.sqlite": "../../secret/far.sqlite", "abs.sqlite": "/etc/abs.sqlite",
                          "inside.sqlite": "input_sources/inside.sqlite"},
        "checked_input_sources": ["model.sqlite", "far.sqlite", "abs.sqlite"],
    }
    package = tmp_path / "refs.flextool.zip"
    _handmade(package, {"Proj/settings.yaml": yaml.dump(settings).encode(),
                        "Proj/input_sources/model.sqlite": b"x", "Proj/input_sources/inside.sqlite": b"y"})
    report = pp.import_project(package, tmp_path / "projects")
    shipped = yaml.safe_load((report.project_path / "settings.yaml").read_text())
    assert shipped["external_refs"] == {"inside.sqlite": "input_sources/inside.sqlite"}
    assert shipped["source_registry"]["1"]["path"] == "input_sources/model.sqlite"
    assert shipped["source_registry"]["2"] == {"name": "far.sqlite", "path": ""}
    assert shipped["source_registry"]["3"] == {"name": "abs.sqlite", "path": ""}
    assert shipped["checked_input_sources"] == ["model.sqlite"]
    assert shipped["input_source_numbers"] == settings["input_source_numbers"]
    (warning,) = [w for w in report.warnings if "outside the project folder" in w]
    assert "far.sqlite" in warning and "abs.sqlite" in warning and "inside.sqlite" not in warning
# --- Item 231: the project's own solver_config, file by file ------------------------------------------------------


def _template_text(solver: str) -> str:
    from flextool._resources import package_data_path

    return Path(package_data_path(f"solver_config/{solver}.opt.template")).read_text(encoding="utf-8")


def _add_member(zip_path: Path, arcname: str, data: bytes) -> None:
    """Add *arcname* to the package and its manifest (a package as a 4.4 development build made it: templates shipped too)."""
    import hashlib

    tmp = zip_path.with_name("tmp.zip")
    with zipfile.ZipFile(zip_path) as zin, zipfile.ZipFile(tmp, "w") as zout:
        manifest = json.loads(zin.read(pp.MANIFEST_NAME))
        for info in zin.infolist():
            if info.filename != pp.MANIFEST_NAME:
                zout.writestr(info, zin.read(info))
        manifest["files"][arcname] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                      "mtime_ns": T_OLD}
        zout.writestr(arcname, data)
        zout.writestr(pp.MANIFEST_NAME, json.dumps(manifest))
    tmp.replace(zip_path)


def test_export_ships_the_files_the_project_runs_use(workspace: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Project gurobi.opt over the workspace's; the workspace's highs.opt; the workspace's unedited cplex
    template is not shipped (it equals the default)."""
    monkeypatch.chdir(workspace["root"])
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    own = workspace["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text("Threads 2\n")
    (workspace["config"] / "gurobi.opt").write_text("Threads 4\n")
    (workspace["config"] / "cplex.opt").write_text(_template_text("cplex"))
    (workspace["config"] / "notes.opt").write_text("not a solver file\n")
    out = workspace["root"] / "out" / "eff.flextool.zip"
    manifest = pp.export_project(workspace["project"], out, [])
    assert sorted(manifest.solver_config) == ["gurobi.opt", "highs.opt"]
    with zipfile.ZipFile(out) as zf:
        assert zf.read("Proj/solver_config/gurobi.opt") == b"Threads 2\n"
        assert zf.read("Proj/solver_config/highs.opt") == b"mip_rel_gap = 0.01\n"
        assert not [n for n in zf.namelist() if n.endswith(("cplex.opt", "notes.opt"))]


def test_import_activates_clean_files_and_holds_back_path_options(workspace: dict) -> None:
    own = workspace["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text("Threads 2\n")
    out = workspace["root"] / "out" / "held.flextool.zip"
    pp.export_project(workspace["project"], out, [], solver_config_dir=[own, workspace["config"]])
    # A package made elsewhere (an export leaves such a file out): a path option.  A shipped .review file is
    # refused (the solver_config folder of a package holds only .opt files).
    _add_member(out, "Proj/solver_config/cplex.opt", b"logfile /tmp/x\n")
    _add_member(out, "Proj/solver_config/xpress.opt", _template_text("xpress").encode("utf-8"))
    report = pp.import_project(out, workspace["projects"], "Held", solver_config_dir=workspace["config"])
    target = workspace["projects"] / "Held" / "solver_config"
    assert (target / "gurobi.opt").read_text() == "Threads 2\n"
    assert not (target / "cplex.opt").exists()
    assert (target / "cplex.opt.review").read_text() == "logfile /tmp/x\n"
    assert not (target / "xpress.opt").exists()            # an unedited template is not unpacked
    assert not (workspace["projects"] / "Held" / "imported_solver_config").exists()
    assert list(report.solver_config_held) == ["cplex.opt"]
    assert report.solver_config_held["cplex.opt"][0].startswith("line 1: logfile /tmp/x")
    assert report.solver_config_defaults == ["xpress.opt"]
    text = "\n".join(pp.describe_import(report))
    assert "cplex.opt: kept as solver_config/cplex.opt.review and not used" in text
    assert "rename it to cplex.opt to use it" in text


def test_first_open_renames_a_development_build_import(tmp_path: Path) -> None:
    project = tmp_path / "projects" / "Old"
    imported = project / "imported_solver_config"
    imported.mkdir(parents=True)
    (imported / "highs.opt").write_text("mip_rel_gap = 0.03\n")
    (imported / "gurobi.opt").write_text("Threads 2\nComputeServer srv.example\n")
    lines = pp.adopt_imported_solver_config(project)
    own = project / "solver_config"
    assert not imported.exists() and (own / "highs.opt").read_text() == "mip_rel_gap = 0.03\n"
    assert not (own / "gurobi.opt").exists() and (own / "gurobi.opt.review").is_file()
    assert lines[0] == ("Project Old: renamed imported_solver_config/ to solver_config/. This project's runs now "
                        "use its solver option files (highs.opt) ahead of the workspace's.")
    assert len(lines) == 2 and "gurobi.opt.review" in lines[1] and "line 2: ComputeServer" in lines[1]
    assert pp.adopt_imported_solver_config(project) == []
    # Both folders: nothing is touched, one line says which one runs read.
    imported.mkdir()
    (imported / "copt.opt").write_text("Threads 1\n")
    assert pp.adopt_imported_solver_config(project) == [
        "Project Old: imported_solver_config/ is not read; this project's runs use solver_config/ (move any file "
        "you need from it and delete imported_solver_config/)."]
    assert (imported / "copt.opt").is_file() and not (own / "copt.opt").exists()


def _old_import(tmp_path: Path) -> tuple[Path, Path]:
    """A project imported by a 4.4 development build: two shipped files to hold back, one clean, one template."""
    project = tmp_path / "projects" / "Old"
    imported = project / "imported_solver_config"
    imported.mkdir(parents=True)
    (imported / "highs.opt").write_text("mip_rel_gap = 0.03\n")
    (imported / "gurobi.opt").write_text("ComputeServer evil:61000\n")
    (imported / "cplex.opt").write_text("logfile /home/u/.bashrc\n")
    (imported / "xpress.opt").write_text(_template_text("xpress"))
    return project, imported


def _unchanged(project: Path, imported: Path) -> None:
    assert not (project / "solver_config").exists()
    assert sorted(p.name for p in imported.iterdir()) == ["cplex.opt", "gurobi.opt", "highs.opt", "xpress.opt"]


def test_first_open_holds_before_renaming_and_names_only_used_files(tmp_path: Path) -> None:
    project, imported = _old_import(tmp_path)
    lines = pp.adopt_imported_solver_config(project)
    own = project / "solver_config"
    assert sorted(p.name for p in own.iterdir()) == ["cplex.opt.review", "gurobi.opt.review", "highs.opt",
                                                       "xpress.opt"]
    # xpress.opt is an unedited template, which runs skip: not named as used.
    assert lines[0] == ("Project Old: renamed imported_solver_config/ to solver_config/. This project's runs now "
                        "use its solver option files (highs.opt) ahead of the workspace's.")
    assert len(lines) == 3


def test_first_open_changes_nothing_when_a_file_cannot_be_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second hold fails (a read-only folder, an antivirus lock): the first is renamed back, the folder
    stays, and the line says the runs keep the workspace's options (never "now use")."""
    project, imported = _old_import(tmp_path)
    real = pp._rename_with_retry

    def fail_on_cplex(src: Path, dst: Path) -> None:
        if src.name == "cplex.opt":
            raise PermissionError(13, "Permission denied", str(dst))
        real(src, dst)

    monkeypatch.setattr(pp, "_rename_with_retry", fail_on_cplex)
    lines = pp.adopt_imported_solver_config(project)
    _unchanged(project, imported)
    assert len(lines) == 1 and "cannot check the solver option files" in lines[0]
    assert lines[0].endswith("imported_solver_config/ is left as it is and this project's runs keep using the "
                             "workspace's solver options.")


def test_first_open_changes_nothing_when_a_review_file_exists(tmp_path: Path) -> None:
    project, imported = _old_import(tmp_path)
    (imported / "gurobi.opt.review").write_text("Threads 1\n")
    lines = pp.adopt_imported_solver_config(project)
    assert not (project / "solver_config").exists()
    assert (imported / "gurobi.opt").read_text() == "ComputeServer evil:61000\n"
    assert (imported / "gurobi.opt.review").read_text() == "Threads 1\n"
    assert (imported / "cplex.opt").is_file() and not (imported / "cplex.opt.review").exists()
    assert len(lines) == 1 and "gurobi.opt.review exists already" in lines[0]


def test_first_open_rolls_back_when_the_folder_cannot_be_renamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, imported = _old_import(tmp_path)
    real = pp._rename_with_retry

    def fail_on_folder(src: Path, dst: Path) -> None:
        if src == imported:
            raise PermissionError(13, "Permission denied", str(dst))
        real(src, dst)

    monkeypatch.setattr(pp, "_rename_with_retry", fail_on_folder)
    lines = pp.adopt_imported_solver_config(project)
    _unchanged(project, imported)
    assert len(lines) == 1 and "cannot rename imported_solver_config/ to solver_config/" in lines[0]


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX folder permissions; root ignores them")
def test_first_open_of_a_read_only_folder_changes_nothing(tmp_path: Path) -> None:
    project, imported = _old_import(tmp_path)
    imported.chmod(0o555)
    try:
        lines = pp.adopt_imported_solver_config(project)
        _unchanged(project, imported)
    finally:
        imported.chmod(0o755)
    assert len(lines) == 1 and "keep using the workspace's solver options" in lines[0]


def test_export_leaves_out_a_file_with_server_or_path_lines(
    workspace: dict, capsys: pytest.CaptureFixture,
) -> None:
    """A credential or a folder of this computer does not leave unseen: the file is left out and named, with its
    lines, before (``always_included``) and after the export (CLI warning)."""
    own = workspace["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text("Threads 2\nWLSSecret s3cr3t\n")
    (own / "cplex.opt").write_text("threads 2\n")
    folders = [own, workspace["config"]]
    fixed = pp.always_included(workspace["project"], folders)
    assert sorted(fixed["solver_config_files"]) == ["cplex.opt", "highs.opt"]
    assert fixed["solver_config_withheld"] == {"gurobi.opt": (own / "gurobi.opt", [
        "line 2: WLSSecret <hidden> connects the solver to another computer (a server, cloud or licence service) or "
        "carries its credentials"])}
    out = workspace["root"] / "out" / "withheld.flextool.zip"
    assert cli.main(["export", str(workspace["project"]), "-o", str(out), "--solver-config-dir", str(own),
                     "--solver-config-dir", str(workspace["config"])]) == cli.EXIT_OK
    text = capsys.readouterr().out
    assert (f"Warning: Solver options gurobi.opt (from {own}) left out of the package, because line 2: WLSSecret "
            "<hidden> connects the solver") in text
    assert "s3cr3t" not in text
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "Proj/solver_config/cplex.opt" in names and not [n for n in names if "gurobi" in n]
        assert b"s3cr3t" not in zf.read(pp.MANIFEST_NAME)
    assert sorted(pp.read_manifest(out).solver_config) == ["cplex.opt", "highs.opt"]


def test_import_names_the_workspace_default(workspace: dict, tmp_path: Path) -> None:
    """A workspace that holds only the seeded template uses FlexTool's default: said so, not "has none"."""
    own = workspace["project"] / "solver_config"
    own.mkdir()
    (own / "gurobi.opt").write_text("Threads 2\n")
    out = workspace["root"] / "out" / "default.flextool.zip"
    pp.export_project(workspace["project"], out, [], solver_config_dir=[own])
    receiver = tmp_path / "receiver"
    receiver.mkdir()
    (receiver / "gurobi.opt").write_text(_template_text("gurobi"))
    report = pp.import_project(out, workspace["projects"], "Recv", solver_config_dir=receiver)
    assert "Solver options gurobi.opt: this workspace uses FlexTool's default." in pp.describe_import(report)


def test_scrub_keeps_no_folder_name_with_spaces(workspace: dict) -> None:
    status = {"a": "D:\\My Data\\ws\\solver_config", "b": "/srv/My Data/ws/x.opt", "c": "see /tmp/x and /var/y/z",
              "d": "https://example.org/a b/c"}
    assert pp.scrub_run_status(status, workspace["project"]) == {
        "a": "<path>\\solver_config", "b": "<path>/x.opt", "c": "see <path>/x and <path>/z",
        "d": "https://example.org/a b/c"}


def test_scrub_keeps_the_text_after_a_path(workspace: dict) -> None:
    """A folder name holds one space only when the next word ends at a separator: prose after a path is kept,
    and ``D:\\My Data\\`` is still removed whole."""
    status = {"a": "from /home/u/proj, wrote and/or copied",
              "b": "error in /usr/lib/python3.11 module foo/bar.py line 3",
              "c": "read D:\\My Data\\run.log, wrote and/or copied"}
    assert pp.scrub_run_status(status, workspace["project"]) == {
        "a": "from <path>/proj, wrote and/or copied",
        "b": "error in <path>/python3.11 module foo/bar.py line 3",
        "c": "read <path>\\run.log, wrote and/or copied"}
