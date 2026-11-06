"""The Toolbox bundle's editable FlexTool install stays relocatable.

A fake bundle is built in a tmp dir (``_internal/Python/Lib/site-packages`` next
to a ``flextool`` clone holding a stub ``flextool`` package) with the editable
files setuptools writes, in both styles: a path ``.pth`` and a finder ``.pth``
(generated with setuptools' own finder template).  ``import flextool`` is
checked in a subprocess that processes only the fake site-packages
(``-I -S`` + ``site.addsitedir``), started outside the bundle.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.update_flextool import bundle_install, self_update

VERSION = "9.9.9"
RELATIVE = "../../../../flextool"


def _make_bundle(root: Path, style: str) -> tuple[Path, Path]:
    """A fake bundle at *root* with an absolute editable install of *style*; returns (site-packages, clone)."""
    site_packages = root / "_internal" / "Python" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    clone = root / "flextool"
    package = clone / "flextool"
    (package / "schemas").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "schemas" / "marker.json").write_text("{}", encoding="utf-8")
    pth = site_packages / f"__editable__.flextool-{VERSION}.pth"
    if style == "path":
        pth.write_text(str(clone.resolve()), encoding="utf-8")
    else:
        # setuptools' own finder template, so the test sees the file setuptools writes; setuptools is not
        # installed in every test environment (Python ≥ 3.12 venvs do not ship it).
        editable_wheel = pytest.importorskip("setuptools.command.editable_wheel")
        finder = f"__editable___flextool_{VERSION.replace('.', '_')}_finder"
        mapping = {"flextool": str(package.resolve())}
        namespaces = {"flextool.schemas": [str((package / "schemas").resolve())]}
        (site_packages / f"{finder}.py").write_text(
            editable_wheel._finder_template(f"__editable__.flextool-{VERSION}.finder", mapping, namespaces), encoding="utf-8"
        )
        pth.write_text(f"import {finder}; {finder}.install()", encoding="utf-8")
        (site_packages / "__pycache__").mkdir()
        (site_packages / "__pycache__" / f"{finder}.cpython-311.pyc").write_bytes(b"")
    return site_packages, clone


def _import_flextool(site_packages: Path, cwd: Path) -> subprocess.CompletedProcess:
    code = "import site, sys; site.addsitedir(sys.argv[1]); import flextool; print(flextool.__file__)"
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(site_packages)],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def _move(root: Path, target: Path) -> tuple[Path, Path]:
    shutil.move(str(root), str(target))
    return target / "_internal" / "Python" / "Lib" / "site-packages", target / "flextool"


@pytest.mark.parametrize("style", ["path", "finder"])
def test_absolute_editable_install_breaks_when_the_bundle_moves(tmp_path, style):
    """What setuptools writes: imports from the clone in place, not after the bundle is moved."""
    site_packages, clone = _make_bundle(tmp_path / "bundle", style)
    in_place = _import_flextool(site_packages, tmp_path)
    assert in_place.returncode == 0, in_place.stderr
    assert Path(in_place.stdout.strip()).resolve().is_relative_to(clone.resolve())
    site_packages, _ = _move(tmp_path / "bundle", tmp_path / "moved")
    moved = _import_flextool(site_packages, tmp_path)
    assert moved.returncode != 0
    assert "ModuleNotFoundError" in moved.stderr


@pytest.mark.parametrize("style", ["path", "finder"])
def test_relative_editable_install_imports_after_the_bundle_moves(tmp_path, style):
    site_packages, clone = _make_bundle(tmp_path / "bundle", style)
    assert bundle_install.check_editable_install_relative(site_packages, clone)

    assert bundle_install.make_editable_install_relative(site_packages, clone) == RELATIVE
    assert (site_packages / f"__editable__.flextool-{VERSION}.pth").read_text(encoding="utf-8") == RELATIVE + "\n"
    assert not list(site_packages.glob("__editable___flextool_*"))
    if style == "finder":
        assert not list((site_packages / "__pycache__").glob("__editable___flextool_*"))
    assert bundle_install.check_editable_install_relative(site_packages, clone) == []

    site_packages, clone = _move(tmp_path / "bundle", tmp_path / "elsewhere" / "Spine Toolbox moved")
    moved = _import_flextool(site_packages, tmp_path)
    assert moved.returncode == 0, moved.stderr
    assert Path(moved.stdout.strip()).resolve() == (clone / "flextool" / "__init__.py").resolve()
    assert bundle_install.check_editable_install_relative(site_packages, clone) == []
    # Idempotent: a second run leaves the relative install as it is.
    assert bundle_install.make_editable_install_relative(site_packages, clone) == RELATIVE


def test_bundle_detected_only_for_the_bundled_python(tmp_path):
    site_packages, clone = _make_bundle(tmp_path / "bundle", "finder")
    python_home = site_packages.parent.parent
    assert bundle_install.bundle_site_packages(clone, prefix=python_home) == site_packages.resolve()
    assert bundle_install.bundle_site_packages(clone, prefix=tmp_path / "venv") is None
    # A clone next to an _internal/Python that is not the running interpreter is not the bundle either.
    assert bundle_install.bundle_site_packages(clone, prefix=sys.prefix) is None


def test_normal_environment_is_left_untouched(tmp_path):
    site_packages, clone = _make_bundle(tmp_path / "bundle", "finder")
    before = {p: p.read_bytes() for p in site_packages.rglob("*") if p.is_file()}
    assert bundle_install.relativise_bundle_install(clone, prefix=tmp_path / "venv") is None
    assert {p: p.read_bytes() for p in site_packages.rglob("*") if p.is_file()} == before
    # This test session's own environment is not the bundle.
    assert bundle_install.relativise_bundle_install() is None


def test_update_flextool_reinstall_makes_the_bundle_install_relative(tmp_path, monkeypatch, capsys):
    """After "Update FlexTool"'s editable reinstall (which writes absolute paths), the install is relative."""
    site_packages, clone = _make_bundle(tmp_path / "bundle", "finder")
    monkeypatch.setattr(bundle_install, "this_clone_root", lambda: clone)
    monkeypatch.setattr(sys, "prefix", str(site_packages.parent.parent))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        stdout = f"Name: flextool\nEditable project location: {clone}\n" if "show" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(self_update.subprocess, "run", fake_run)
    self_update._reinstall_if_needed()
    assert any("install" in argv and "-e" in argv for argv in calls)
    assert "FlexTool editable install made relative: ../../../../flextool" in capsys.readouterr().out
    assert bundle_install.check_editable_install_relative(site_packages, clone) == []


def test_script_runs_without_importing_flextool(tmp_path):
    """The bundle build runs the file as a script while the absolute path no longer leads to the clone."""
    site_packages, clone = _make_bundle(tmp_path / "bundle", "finder")
    script = clone / "flextool" / "update_flextool" / "bundle_install.py"
    script.parent.mkdir()
    shutil.copy(bundle_install.__file__, script)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def run(*args):
        return subprocess.run([sys.executable, "-I", "-S", str(script), "--site-packages", str(site_packages), *args],
                              cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)

    assert run("--check").returncode == 1
    done = run()
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith(RELATIVE)
    assert run("--check").returncode == 0
    # Without --site-packages, outside the bundled Python, the script refuses.
    refused = subprocess.run([sys.executable, "-I", "-S", str(script)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)
    assert refused.returncode == 2
    assert "Not the Toolbox bundle" in refused.stderr
