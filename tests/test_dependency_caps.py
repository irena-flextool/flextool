"""The weekly canary's cap lifter (.github/scripts/dependency_caps.py).

The script is not part of the package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / ".github" / "scripts" / "dependency_caps.py"


@pytest.fixture(scope="module")
def caps_module():
    spec = importlib.util.spec_from_file_location("dependency_caps", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


_PYPROJECT = """\
[project]
name = "demo"
dependencies = [
    "highspy>=1.15.1,<1.16",
    # a comment mentioning polars<3 is left alone
    "polars>=1.40,<3",
    "numpy>=1.26",
]

[project.optional-dependencies]
viewer = ["networkx>=3.0"]
compat = ["polars[rtcompat]>=1.40,<3"]
dev = ["pytest>=8.0"]
"""


def test_lift_drops_caps_in_core_and_in_extras(tmp_path, caps_module):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_PYPROJECT, encoding="utf-8")
    caps_out = tmp_path / "caps.json"

    caps = caps_module.lift(pyproject, caps_out)

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["dependencies"] == ["highspy>=1.15.1", "polars>=1.40", "numpy>=1.26"]
    extras = data["project"]["optional-dependencies"]
    assert extras["compat"] == ["polars[rtcompat]>=1.40"]
    assert extras["viewer"] == ["networkx>=3.0"]
    assert extras["dev"] == ["pytest>=8.0"]
    assert "# a comment mentioning polars<3 is left alone" in pyproject.read_text(encoding="utf-8")
    # One entry per package: polars is recorded once, from the core list.
    assert caps == {"highspy": ">=1.15.1,<1.16", "polars": ">=1.40,<3"}
    assert json.loads(caps_out.read_text(encoding="utf-8")) == caps


def test_lift_records_an_extra_only_package_under_its_name(tmp_path, caps_module):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\ndependencies = ["numpy>=1.26"]\n\n'
        '[project.optional-dependencies]\ncompat = ["polars[rtcompat]==1.40.1"]\n',
        encoding="utf-8",
    )
    caps = caps_module.lift(pyproject, tmp_path / "caps.json")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["optional-dependencies"]["compat"] == ["polars[rtcompat]>=1.40.1"]
    assert caps == {"polars": "==1.40.1"}


def test_lift_refuses_a_requirement_it_cannot_locate(tmp_path, caps_module):
    # The same requirement string also appears quoted outside the lists: the
    # rewrite would touch it, so the lift stops instead.
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\ndependencies = ["polars>=1.40,<3"]\n\n'
        '[tool.other]\nnote = "polars>=1.40,<3"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="found dependency 'polars>=1.40,<3' 2 times"):
        caps_module.lift(pyproject, tmp_path / "caps.json")


def test_lift_of_the_repository_pyproject_keeps_compat_in_step_with_core(tmp_path, caps_module):
    # What the canary does to the real file: no upper cap survives anywhere,
    # and the compat extra still carries the core polars requirement.
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text((_REPO / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8")
    caps_module.lift(pyproject, tmp_path / "caps.json")

    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    raws = list(project["dependencies"])
    for group in project["optional-dependencies"].values():
        raws.extend(group)
    for raw in raws:
        assert not caps_module._is_capped(caps_module.Requirement(raw)), raw
    (core,) = [r for r in project["dependencies"] if r.startswith("polars")]
    (compat,) = project["optional-dependencies"]["compat"]
    assert compat == "polars[rtcompat]" + core[len("polars"):]
