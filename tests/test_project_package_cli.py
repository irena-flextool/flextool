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


def _run(folder: Path, value: float, db_mtime_ns: int) -> None:
    folder.mkdir(parents=True)
    pl.DataFrame({"t": ["t0001", "t0002"], "v": [value, value * 2]}).write_parquet(folder / "unit_flow_dt.parquet")
    (folder / "_run_status.json").write_text(json.dumps({
        "state": "finished", "verdict": "optimal", "host": "builder", "input": {"db_mtime": _iso(db_mtime_ns)}}))
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
    (project / "output_parquet" / "_axis_bounds.json").write_text("{}")
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
    pairs["imported_solver_config/highs.opt"] = workspace["config"] / "highs.opt"
    for path in (project / "output_parquet").rglob("*"):
        if path.is_file():
            pairs[path.relative_to(project).as_posix()] = path
    for rel, original in pairs.items():
        imported = copy / rel
        assert imported.read_bytes() == original.read_bytes(), rel
        assert imported.stat().st_mtime_ns == original.stat().st_mtime_ns, rel
    for arcname, facts in manifest.files.items():
        rel = arcname.removeprefix("Proj/").replace("solver_config/", "imported_solver_config/", 1)
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


def test_results_exclusion_and_regenerated_parts(workspace: dict) -> None:
    out, manifest = _export(workspace, ["A_1"])
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert zf.getinfo("Proj/output_parquet/A_1/unit_flow_dt.parquet").compress_type == zipfile.ZIP_STORED
    assert "Proj/output_parquet/A_1/unit_flow_dt.parquet" in names
    assert "Proj/output_parquet/_axis_bounds.json" in names
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
    assert "FLEXTOOL_SOLVER_CONFIG_DIR=" in report.solver_config_hint
    assert "imported_solver_config" in report.solver_config_hint
