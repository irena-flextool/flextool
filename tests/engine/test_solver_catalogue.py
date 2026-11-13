"""FlexTool's list of each solver's parameters (item 240 B1): the shipped files, their review, the generator.

``flextool/engine/solve/solver_catalogue/<solver>.json`` is written by ``tools/solver_catalogue.py`` from the solver
packages; every name has a reviewed class in ``tools/solver_catalogue_review.json``.  Checked here:

* every shipped name is reviewed (a new name without a review entry fails), keyed as the solver reads it, with the
  class its review gives; text-valued names and every class other than the name rules' carry a note;
* the shipped file equals a fresh listing of each installed package (names, types, classes; not the version text);
* the classes that decide security, set by hand from each solver's parameter reference;
* the generator stops on an unreviewed name and never removes one;
* the files are package data, and a missing file stops the run instead of letting every name pass.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

from flextool.engine.solve import _solver_catalogue
from flextool.engine.solve._solver_catalogue import SOLVERS, CatalogueError, catalogue_key, rule_class

REPO = Path(__file__).resolve().parents[2]


def _generator():
    name = "_solver_catalogue_tool"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "solver_catalogue.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _shipped(solver: str) -> dict:
    return json.loads((REPO / "flextool" / _solver_catalogue.CATALOGUE_FOLDER / f"{solver}.json").read_text())


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_shipped_name_is_reviewed_and_keyed_as_the_solver_reads_it(solver) -> None:
    gen = _generator()
    data = _shipped(solver)
    review = gen.load_review()
    params = {k: (v["name"], v["type"]) for k, v in data["parameters"].items()}
    assert gen.review_problems(review, solver, params) == []
    for key, item in data["parameters"].items():
        assert catalogue_key(solver, item["name"]) == key, (key, item["name"])
        assert (item["class"], item.get("reason")) == gen.reviewed_class(review, solver, key), key
    assert _solver_catalogue.parse_catalogue(solver, data).versions  # every file says what it was made from


@pytest.mark.parametrize("solver", SOLVERS)
def test_shipped_catalogue_matches_the_installed_package(solver) -> None:
    """``tools/solver_catalogue.py --check``: a package of a version with other parameters (a highspy pin bump)
    fails here until the catalogue is regenerated and the new names are reviewed."""
    gen = _generator()
    package = {"highs": "highspy", "gurobi": "gurobipy", "cplex": "cplex", "xpress": "xpress", "copt": "coptpy"}
    pytest.importorskip(package[solver])
    try:
        _versions, listed = gen.LISTERS[solver]()
    except (SystemExit, Exception) as exc:  # noqa: BLE001 - e.g. a package without its native library here
        pytest.skip(f"{solver}: cannot list the parameters here ({exc})")
    assert gen.check(solver, listed, gen.load_review()) == []


#: The classes that keep a database from reaching another computer or a file outside the work folder, each set
#: by hand from the solver's parameter reference.
_HAND_CLASSES = [
    *[("gurobi", n, "local_only", "remote") for n in (
        "ComputeServer", "CSManager", "CSRouter", "CloudHost", "TokenServer", "WorkerPool", "WLSProxy", "WLSAccessID",
        "CSPriority", "ServerTimeout", "TSPort", "LicenseID", "Username")],
    ("gurobi", "JobID", "local_only", "unreviewed"),
    ("gurobi", "LogFile", "file", None), ("gurobi", "NodefileDir", "file", None),
    ("gurobi", "Record", "file", None), ("gurobi", "ResultFile", "flextool", None),
    ("gurobi", "MIPGap", "plain", None),
    ("xpress", "COMPUTE", "local_only", "remote"), ("xpress", "COMPUTEEXECSERVICE", "local_only", "remote"),
    ("xpress", "TUNEROUTPUTPATH", "file", None), ("xpress", "MAXTIME", "plain", None),
    ("highs", "log_file", "file", None), ("highs", "mip_rel_gap", "plain", None),
    ("highs", "presolve", "plain", None),
    ("cplex", "workdir", "file", None), ("cplex", "logfile", "file", None),
    ("cplex", "mip tolerances mipgap", "plain", None), ("cplex", "read fileencoding", "plain", None),
    ("cplex", "defaults", "local_only", "command"), ("cplex", "distmip config", "local_only", "remote"),
    ("copt", "LogFile", "file", None), ("copt", "RelGap", "plain", None),
]


@pytest.mark.parametrize("solver, name, cls, reason", _HAND_CLASSES)
def test_security_classes_are_as_the_parameter_references_say(solver, name, cls, reason) -> None:
    entry = _solver_catalogue.load(solver).entry(name)
    assert entry is not None and (entry.cls, entry.reason) == (cls, reason)


def test_cplex_console_names_cite_ibms_documentation() -> None:
    data = _shipped("cplex")
    assert any("ibm.com/docs" in s for s in data["sources"])
    for key in ("logfile", "defaults", "distmip config"):
        assert "IBM Documentation" in data["parameters"][key]["source"], key


def test_flextools_own_names_and_the_shipped_templates_are_plain() -> None:
    """The names FlexTool itself writes (``solver_mip_gap`` / flags, :data:`_PARAM_MAP`) and every line of the
    shipped option templates are ordinary parameters of their solver."""
    import flextool
    from flextool.engine.solve._solver_dispatch import _PARAM_MAP, _highs_opt_lines
    from flextool.engine.solve._subprocess_solve import _native_opt_lines

    for solver, names in _PARAM_MAP.items():
        for name in names.values():
            assert _solver_catalogue.load(solver).entry(name).cls == "plain", (solver, name)
    for solver in SOLVERS:
        text = (Path(flextool.__file__).parent / "solver_config" / f"{solver}.opt.template").read_text()
        lines = _highs_opt_lines(text) if solver == "highs" else _native_opt_lines(text)
        assert lines, solver
        for _lineno, name, _value in lines:
            assert _solver_catalogue.load(solver).entry(name).cls == "plain", (solver, name)


def test_name_rules_apply_to_every_type() -> None:
    """An integer can switch remote solving on (Xpress ``COMPUTE``) or write files (Gurobi ``Record``), so the
    rules look at the name only."""
    assert rule_class("xpress", "compute") == ("local_only", "remote")
    assert rule_class("gurobi", "csnewnumber") == ("local_only", "remote")
    assert rule_class("highs", "write_new_thing") == ("file", None)
    assert rule_class("cplex", "read newlimit") == ("file", None)
    assert rule_class("copt", "relgap") is None


def test_generator_stops_on_an_unreviewed_name_and_on_an_unexplained_class() -> None:
    gen = _generator()
    review = gen.load_review()
    params = {k: (v["name"], v["type"]) for k, v in _shipped("gurobi")["parameters"].items()}
    new = {**params, "newtext": ("NewText", "string"), "csnewswitch": ("CSNewSwitch", "int"),
           "newnumber": ("NewNumber", "double")}
    problems = gen.review_problems(review, "gurobi", new)
    assert sorted(p for p in problems if "not reviewed" in p) == [
        "gurobi: 'csnewswitch' (int) is not reviewed", "gurobi: 'newnumber' (double) is not reviewed",
        "gurobi: 'newtext' (string) is not reviewed"]
    assert gen.proposal("gurobi", "csnewswitch", "int").endswith("local_only/remote")
    # A text name reviewed without a note, and a server-prefix name reviewed as plain without a note, are refused.
    edited = json.loads(json.dumps(review))
    edited["gurobi"]["plain"] += ["newtext", "csnewswitch", "newnumber"]
    problems = gen.review_problems(edited, "gurobi", new)
    assert any("'newtext' is a string parameter" in p for p in problems)
    assert any("'csnewswitch' matches the name rule" in p for p in problems)
    assert not any("newnumber" in p for p in problems)


def test_regeneration_keeps_every_name_and_records_the_versions() -> None:
    gen = _generator()
    shipped = _shipped("gurobi")
    newer = {"package": "gurobipy", "version": "99.0.0", "library": "99.0.0"}
    merged = gen.merge("gurobi", shipped, newer, {"mipgap": ("MIPGap", "double")}, gen.load_review())
    assert set(merged["parameters"]) == set(shipped["parameters"])
    assert merged["parameters"]["mipgap"]["seen_in"] == ["13.0.2", "99.0.0"]
    assert merged["parameters"]["logfile"]["seen_in"] == ["13.0.2"]
    assert merged["generated_from"][-1] == newer and len(merged["generated_from"]) == 2
    assert _solver_catalogue.parse_catalogue("gurobi", merged).versions_text() == "made from Gurobi 13.0.2, 99.0.0"


def test_catalogue_is_package_data_and_a_missing_file_stops(monkeypatch, tmp_path) -> None:
    globs = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["flextool"]
    assert f"{_solver_catalogue.CATALOGUE_FOLDER}/*.json" in globs
    import flextool._resources as resources

    monkeypatch.setattr(resources, "package_data_path", lambda rel: tmp_path / rel)
    _solver_catalogue.load.cache_clear()
    try:
        with pytest.raises(CatalogueError, match=r"list of Gurobi parameters .* reinstall FlexTool"):
            _solver_catalogue.load("gurobi")
    finally:
        _solver_catalogue.load.cache_clear()



def test_docs_list_the_file_parameters_and_versions_of_the_catalogue() -> None:
    """``docs/solvers/index.md`` "What the database may set": the table of file and folder parameters is the
    catalogue's text-valued file entries, and the versions table names each catalogue's versions."""
    text = (REPO / "docs" / "solvers" / "index.md").read_text()
    section = text.split("#### What the database may set", 1)[1].split("\n#### ", 1)[0]
    table = section.split("| Solver | File and folder parameters |", 1)[1].split("\n\n", 1)[0]
    labels = {v: k for k, v in _solver_catalogue.SOLVER_LABEL.items()}
    rows = {}
    for line in table.splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[labels[cells[0]]] = set(re.findall(r"`([^`]+)`", cells[1]))
    for solver in SOLVERS:
        catalogue = _solver_catalogue.load(solver)
        expected = {e.name for e in catalogue.entries.values() if e.cls in ("file", "flextool") and e.type == "string"}
        assert rows[solver] == expected, solver
        versions = section.split("| List made from |", 1)[1].split("\n\n", 1)[0]
        cells = {line.split("|")[1].strip(): line for line in versions.splitlines() if line.count("|") >= 3}
        row = cells[catalogue.label]
        assert all(v.rsplit(".", 1)[0] in row for v in catalogue.versions), (solver, row)


def test_solver_arguments_examples_in_the_docs_are_listed_names() -> None:
    """Every ``solver_arguments:`` example block of ``docs/solvers/`` names listed parameters (the solver of the
    page; ``index.md``'s example is Gurobi's)."""
    found = 0
    for page in sorted((REPO / "docs" / "solvers").glob("*.md")):
        solver = "gurobi" if page.stem == "index" else page.stem
        for block in page.read_text().split("solver_arguments:\n")[1:]:
            for line in block.split("```", 1)[0].splitlines():
                if "=" in line:
                    name = line.split("=", 1)[0].strip()
                    assert _solver_catalogue.load(solver).entry(name).cls == "plain", (page.name, name)
                    found += 1
    assert found >= 7
