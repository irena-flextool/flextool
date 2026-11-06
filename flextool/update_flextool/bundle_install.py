"""Keep FlexTool's editable install in the Windows Toolbox bundle relocatable.

The bundle (built by ``.github/workflows/toolbox-flextool-bundle.yml``) has
this layout::

    <bundle>/_internal/Python/python.exe           the embedded Python
    <bundle>/_internal/Python/Lib/site-packages    its site-packages
    <bundle>/flextool/                              the FlexTool git clone

FlexTool is installed editable from the clone.  setuptools records the
clone by its absolute path, in one of two styles:

* a path ``.pth``: ``__editable__.flextool-<ver>.pth`` holds the path of the
  project root, which ``site`` adds to ``sys.path``;
* a finder ``.pth``: the same file holds ``import __editable___flextool_<ver>_finder;
  ...install()`` and the finder module maps ``flextool`` to the absolute path
  of the package (the style setuptools uses for FlexTool's flat layout).

Users unzip the bundle anywhere and may move it later, so an absolute path
breaks ``import flextool`` in processes that do not start in the clone.
:func:`make_editable_install_relative` rewrites the ``.pth`` to a path line
relative to site-packages (``site`` joins such a line to the site-packages
folder) and removes the finder.  The bundle build calls it, and so does
"Update FlexTool" after its editable reinstall, which writes absolute paths
again.  Outside the bundle (a venv, a conda environment) nothing is changed.

Standard library only, and runnable as a script (``python
flextool/update_flextool/bundle_install.py [--check]``), so it works even when
the absolute path no longer leads to the clone and ``import flextool`` fails.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PTH_GLOB = "__editable__.flextool-*.pth"
FINDER_GLOB = "__editable___flextool_*_finder.py"
FINDER_PYC_GLOB = "__editable___flextool_*_finder.*.pyc"


def bundle_site_packages(clone_root: Path, prefix: str | os.PathLike[str] | None = None) -> Path | None:
    """Site-packages of the bundled Python, or ``None`` outside the bundle.

    The bundle is recognised only when all hold: the running interpreter's
    prefix is ``<clone_root>/../_internal/Python`` (the embedded Python sits
    next to the clone), that Python has ``Lib/site-packages``, and the clone
    holds the ``flextool`` package.

    Args:
        clone_root: the FlexTool clone (the folder with ``flextool/__init__.py``).
        prefix: the interpreter prefix; defaults to ``sys.prefix``.
    """
    clone_root = Path(clone_root).resolve()
    python_home = (clone_root.parent / "_internal" / "Python").resolve()
    running = Path(sys.prefix if prefix is None else prefix).resolve()
    if running != python_home:
        return None
    site_packages = python_home / "Lib" / "site-packages"
    if not site_packages.is_dir() or not (clone_root / "flextool" / "__init__.py").is_file():
        return None
    return site_packages


def relative_clone_path(site_packages: Path, clone_root: Path) -> str:
    """The clone's path relative to site-packages, with forward slashes."""
    return Path(os.path.relpath(Path(clone_root).resolve(), Path(site_packages).resolve())).as_posix()


def _editable_pth(site_packages: Path) -> Path:
    found = sorted(site_packages.glob(PTH_GLOB))
    if len(found) != 1:
        raise RuntimeError(f"expected one FlexTool editable .pth in {site_packages}, found {len(found)}")
    return found[0]


def _finder_files(site_packages: Path) -> list[Path]:
    files = list(site_packages.glob(FINDER_GLOB))
    pycache = site_packages / "__pycache__"
    if pycache.is_dir():
        files += pycache.glob(FINDER_PYC_GLOB)
    return sorted(files)


def make_editable_install_relative(site_packages: Path, clone_root: Path) -> str:
    """Point FlexTool's editable install in *site_packages* to *clone_root* by a relative path.

    Works from either setuptools style (path ``.pth`` or finder ``.pth``) and
    from an already relative ``.pth`` (then nothing changes).  The ``.pth``
    gets one line, the clone's path relative to *site_packages*; the finder
    module and its compiled copies are deleted.  *clone_root* need not exist
    yet (the bundle build relativises before the clone is moved in).

    Returns:
        The relative path written.

    Raises:
        RuntimeError: there is not exactly one FlexTool editable ``.pth``, or
            site-packages holds a copy of the ``flextool`` package (a
            non-editable install).
    """
    site_packages = Path(site_packages)
    if (site_packages / "flextool").exists():
        raise RuntimeError(f"{site_packages} has a copy of flextool, not an editable install")
    pth = _editable_pth(site_packages)
    relative = relative_clone_path(site_packages, clone_root)
    if pth.read_text(encoding="utf-8").strip() != relative:
        pth.write_text(relative + "\n", encoding="utf-8")
    for finder in _finder_files(site_packages):
        finder.unlink()
    return relative


def check_editable_install_relative(site_packages: Path, clone_root: Path) -> list[str]:
    """Problems that keep the editable install from being relocatable (empty when it is)."""
    site_packages = Path(site_packages)
    problems = []
    try:
        pth = _editable_pth(site_packages)
    except RuntimeError as exc:
        return [str(exc)]
    relative = relative_clone_path(site_packages, clone_root)
    content = pth.read_text(encoding="utf-8").strip()
    if content != relative:
        problems.append(f"{pth.name} holds {content!r}, not {relative!r}")
    elif not (site_packages / relative / "flextool" / "__init__.py").is_file():
        problems.append(f"no FlexTool clone at {relative} from {site_packages}")
    problems += [f"setuptools finder left: {finder.name}" for finder in _finder_files(site_packages)]
    if (site_packages / "flextool").exists():
        problems.append(f"{site_packages} has a copy of flextool")
    return problems


def this_clone_root() -> Path:
    """The clone this file belongs to (``<clone>/flextool/update_flextool/bundle_install.py``)."""
    return Path(__file__).resolve().parents[2]


def relativise_bundle_install(clone_root: Path | None = None, prefix: str | None = None) -> str | None:
    """In the Toolbox bundle, make the editable install relative; elsewhere do nothing.

    Returns:
        The relative path written, or ``None`` outside the bundle.
    """
    clone_root = this_clone_root() if clone_root is None else Path(clone_root)
    site_packages = bundle_site_packages(clone_root, prefix)
    if site_packages is None:
        return None
    return make_editable_install_relative(site_packages, clone_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site-packages", type=Path,
                        help="site-packages to rewrite (default: the bundled Python's, which must be running)")
    parser.add_argument("--clone", type=Path, help="the FlexTool clone (default: the one holding this file)")
    parser.add_argument("--check", action="store_true",
                        help="only check that the install is relative and reaches the clone")
    args = parser.parse_args(argv)
    clone_root = this_clone_root() if args.clone is None else args.clone
    site_packages = args.site_packages or bundle_site_packages(clone_root)
    if site_packages is None:
        print(f"Not the Toolbox bundle: {sys.prefix} is not {clone_root.resolve().parent / '_internal' / 'Python'}",
              file=sys.stderr)
        return 2
    if args.check:
        problems = check_editable_install_relative(site_packages, clone_root)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"FlexTool editable install in {site_packages} is relative")
        return 1 if problems else 0
    relative = make_editable_install_relative(site_packages, clone_root)
    print(f"FlexTool editable install in {site_packages}: {relative}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
