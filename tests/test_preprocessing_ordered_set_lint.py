"""Lint: no bare ``set`` constructors or set literals in preprocessing code.

The python-preprocessing migration requires every set written to CSV to
preserve insertion order — MathProg's iteration order over the loaded
set determines constraint generation order, which in turn determines
MPS row/column ordering. Python's ``set`` and ``frozenset`` are
*unordered*; iteration order is hash-dependent and unstable across
Python builds.

The discipline is:
- Use ``dict.fromkeys(iterable)`` (or ``dict[key] = None``) whenever
  you need an ordered, deduplicated container.
- Use ``list`` if you need positional access.
- ``frozenset`` is permitted *only* when constructed from an already-
  ordered source for hot-path membership testing — the linter trusts
  the developer here and does NOT flag it.

This test scans ``flextool/flextoolrunner/preprocessing/`` (the
designated home for new migration code) and fails if any module
contains a bare ``set(...)`` call or a set literal. If the directory
does not exist yet, the test is a no-op (forward-compatible — fires
the moment the first preprocessing module lands).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_TARGET_DIR = (
    Path(__file__).parent.parent / "flextool" / "flextoolrunner" / "preprocessing"
)


def _iter_py_files(root: Path):
    if not root.exists():
        return
    yield from sorted(root.rglob("*.py"))


class _BareSetVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Bare ``set(...)`` constructor.
        # Allow ``set`` accessed as a method or attribute (e.g. ``foo.set(...)``).
        if isinstance(node.func, ast.Name) and node.func.id == "set":
            self.violations.append((node.lineno, "bare set() constructor"))
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        # Set literal ``{1, 2, 3}`` — always unordered.
        self.violations.append((node.lineno, "set literal {...}"))
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        # Set comprehension ``{x for x in ...}`` — always unordered.
        self.violations.append((node.lineno, "set comprehension {x for ...}"))
        self.generic_visit(node)


def test_no_unordered_set_in_preprocessing_modules():
    """Fail on any bare set construction in preprocessing/."""
    if not _TARGET_DIR.exists():
        pytest.skip(f"{_TARGET_DIR} does not exist yet — lint is a no-op")

    failures: list[str] = []
    for path in _iter_py_files(_TARGET_DIR):
        tree = ast.parse(path.read_text(), filename=str(path))
        v = _BareSetVisitor()
        v.visit(tree)
        rel = path.relative_to(_TARGET_DIR.parent.parent.parent)
        for lineno, kind in v.violations:
            failures.append(f"  {rel}:{lineno} — {kind}")
    if failures:
        pytest.fail(
            "Bare set / set-literal / set-comp found in "
            f"flextool/flextoolrunner/preprocessing/:\n"
            + "\n".join(failures)
            + "\n\nUse dict.fromkeys(iterable) for an ordered deduplicated "
            "container, or list with explicit dedup. frozenset is permitted "
            "only when wrapping an already-ordered source for hot-path "
            "membership testing — see migration/README.md."
        )
