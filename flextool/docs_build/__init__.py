"""Generated documentation pages: the model formulation, built from the installed code.

One generator for the web docs (the mkdocs hook ``tools/mkdocs_hooks.py``) and for the copies other tools
build.  ``scope`` lists what the documentation ships.
"""
from __future__ import annotations

from pathlib import Path

from flextool.docs_build import scope


def render_page(name: str, markdown_root: Path) -> str:
    """The Markdown of the generated page ``name`` (one of ``scope.GENERATED_PAGES``)."""
    if name == scope.FORMULATION_PAGE:
        from flextool.docs_build.formulation import render_formulation
        return render_formulation(markdown_root)
    raise ValueError(f"{name!r} is not a generated page ({', '.join(scope.GENERATED_PAGES)})")
