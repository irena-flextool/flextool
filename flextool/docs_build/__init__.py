"""Generated documentation pages: the model formulation and the results catalogue, built from the installed code.

One generator for the web docs (the mkdocs hook ``tools/mkdocs_hooks.py``), for ``python -m
flextool.docs_build`` and for the assistant's documentation cache (``index.generate``, called on first use by
``flextool.assistant.docs_store``; that path skips the private-names check).  ``scope`` lists what the
documentation ships.

Private names: when the local, git-ignored list ``.private_names`` (``scope.PRIVATE_NAMES_FILE``) exists at the
repository root, a generated page or a shipped Markdown file that matches one of its lines fails the build
(:class:`PrivateNameError`); without the file the check is skipped.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from flextool.docs_build import scope

REPO_ROOT = Path(__file__).resolve().parents[2]


class PrivateNameError(RuntimeError):
    """A page the documentation ships or generates contains a name from the private-names list."""


def render_page(name: str, markdown_root: Path) -> str:
    """The Markdown of the generated page ``name`` (one of ``scope.GENERATED_PAGES``)."""
    if name == scope.FORMULATION_PAGE:
        from flextool.docs_build.formulation import render_formulation
        return render_formulation(markdown_root)
    if name == scope.RESULTS_CATALOGUE_PAGE:
        from flextool.docs_build.catalogue import render_catalogue
        return render_catalogue(markdown_root)
    raise ValueError(f"{name!r} is not a generated page ({', '.join(scope.GENERATED_PAGES)})")


def render_pages(markdown_root: Path) -> dict[str, str]:
    """Every generated page, ``{name: markdown}``."""
    return {name: render_page(name, markdown_root) for name in scope.GENERATED_PAGES}


# ---------------------------------------------------------------------------
# Private names


def default_private_names_path(repo_root: Path = REPO_ROOT) -> Path:
    """``.private_names`` at the repository root; in a git worktree, at the root of its main checkout (the file
    is git-ignored, so a worktree has no copy of its own)."""
    own = repo_root / scope.PRIVATE_NAMES_FILE
    if own.exists():
        return own
    git = repo_root / ".git"
    if git.is_file():
        text = git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            if not gitdir.is_absolute():
                gitdir = (repo_root / gitdir).resolve()
            common = gitdir / "commondir"
            if common.exists():
                common_dir = Path(common.read_text(encoding="utf-8").strip())
                if not common_dir.is_absolute():
                    common_dir = (gitdir / common_dir).resolve()
                return common_dir.parent / scope.PRIVATE_NAMES_FILE
    return own


def load_private_names(path: Path | None) -> list[re.Pattern[str]] | None:
    """The patterns of the list (one case-insensitive regular expression per line, ``#`` comments), or ``None``
    when there is no list."""
    if path is None or not path.exists():
        return None
    patterns: list[re.Pattern[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def check_private_names(pages: dict[str, str], patterns: list[re.Pattern[str]] | None) -> None:
    """Raise :class:`PrivateNameError` listing ``page:line`` of every match.  The message names the pattern by
    its number in the list, never the name itself."""
    if not patterns:
        return
    hits: list[str] = []
    for page, text in pages.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            for i, pattern in enumerate(patterns, start=1):
                if pattern.search(line):
                    hits.append(f"{page}:{lineno} (private-names pattern {i})")
    if hits:
        raise PrivateNameError("Private names in the documentation:\n  " + "\n  ".join(hits))


def shipped_markdown(markdown_root: Path) -> dict[str, str]:
    """The curated Markdown files the documentation ships, ``{relative path: text}`` (missing files skipped)."""
    out: dict[str, str] = {}
    for rel in scope.CURATED_DOCS:
        path = markdown_root / rel
        if path.exists():
            out[rel] = path.read_text(encoding="utf-8")
    return out


def check_documentation(markdown_root: Path, pages: dict[str, str], private_names: Path | None) -> None:
    """The private-names check over the generated pages and the shipped Markdown files."""
    check_private_names({**shipped_markdown(markdown_root), **pages}, load_private_names(private_names))


# ---------------------------------------------------------------------------
# Writing


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 with LF line ends through a sibling temp file, so a reader never sees a half-written page."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_pages(out_dir: Path, markdown_root: Path, private_names: Path | None) -> list[Path]:
    """Render, check and write every generated page into ``out_dir``; return the written paths."""
    pages = render_pages(markdown_root)
    check_documentation(markdown_root, pages, private_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in pages.items():
        path = out_dir / name
        write_text_atomic(path, text)
        written.append(path)
    return written
