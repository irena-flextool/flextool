"""Build step: copy the curated documentation Markdown into the package.

``[tool.setuptools.cmdclass] build_py = "_flextool_build.BuildPy"`` runs this before setuptools' own
``build_py``.  It copies the Markdown files listed in ``flextool/docs_build/scope.py`` (``CURATED_DOCS``) from
``docs/`` into ``<build_lib>/flextool/assistant/_docs/md/``, keeping their folders, so a wheel (also one built
from the sdist, which ``MANIFEST.in`` gives the docs and this module) carries the documentation the assistant
indexes on first use.  An editable install reads ``docs/`` of its clone instead and copies nothing.

Standard library and setuptools only: ``scope.py`` is read by path, because importing ``flextool`` runs its
``__init__`` (and its dependencies need not be installed in an isolated build environment).
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

from setuptools.command.build_py import build_py as _build_py

PROJECT_ROOT = Path(__file__).resolve().parent
SCOPE_FILE = Path("flextool") / "docs_build" / "scope.py"
DOCS_DIR = "docs"
# Where the copies go, relative to the build folder (``build_lib``).
PACKAGED_DOCS = Path("flextool") / "assistant" / "_docs" / "md"


def load_scope(project_root: Path = PROJECT_ROOT) -> ModuleType:
    """``flextool/docs_build/scope.py`` loaded by path, without importing ``flextool``."""
    path = project_root / SCOPE_FILE
    spec = importlib.util.spec_from_file_location("_flextool_docs_scope", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the documentation scope from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_docs(build_lib: Path, project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Copy every ``CURATED_DOCS`` file from ``<project_root>/docs`` into ``<build_lib>/flextool/assistant/
    _docs/md``; return the written paths.  A file of the scope missing from ``docs/`` fails the build."""
    curated: tuple[str, ...] = load_scope(project_root).CURATED_DOCS
    source = project_root / DOCS_DIR
    missing = [rel for rel in curated if not (source / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"documentation files of flextool/docs_build/scope.py missing from {source}: {', '.join(missing)}")
    target = build_lib / PACKAGED_DOCS
    if target.exists():
        shutil.rmtree(target)          # a file dropped from the scope must not linger from an earlier build
    written: list[Path] = []
    for rel in curated:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
        written.append(dest)
    return written


class BuildPy(_build_py):
    """setuptools ``build_py`` that first copies the curated documentation into the package."""

    def _copies_docs(self) -> bool:
        return not getattr(self, "editable_mode", False)

    def run(self) -> None:
        if self._copies_docs():
            copy_docs(Path(self.build_lib))
        super().run()

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode))
        if self._copies_docs():
            target = Path(self.build_lib) / PACKAGED_DOCS
            outputs += [str(target / rel) for rel in load_scope().CURATED_DOCS]
        return outputs
