"""Line-range index of Python modules, so a reader can open just the part it needs.

Large engine modules (``preprocess/params/_derived.py``, ``preprocess/flex_data.py``,
``solve/_orchestration.py`` …) run to thousands of lines.  This tool lists every
top-level function and class (and class methods) with its line range and the first
line of its docstring, so an agent or a person can read a single range instead of
the whole file.

Usage::

    python tools/code_index.py flextool/engine/preprocess/flex_data.py
    python tools/code_index.py flextool/engine/preprocess/params/            # every .py below
    python tools/code_index.py --find apply_derived_a                       # search the package
    python tools/code_index.py --find _load_ --root flextool/engine          # substring match
    python tools/code_index.py --big 2000                                    # modules over N lines

Output lines are ``path:start-end  kind name  — docstring first line``.  Ranges come
from the AST (``lineno`` / ``end_lineno``, decorators included), so they stay exact
as the code changes; nothing is cached.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "flextool"


def _first_doc_line(node: ast.AST) -> str:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:100]


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def index_file(path: Path) -> list[tuple[int, int, str, str, str]]:
    """Return ``(start, end, kind, qualified name, doc)`` for the module's defs."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"{path}: cannot parse ({exc})", file=sys.stderr)
        return []
    rows: list[tuple[int, int, str, str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows.append((_start(node), node.end_lineno, "def", node.name, _first_doc_line(node)))
        elif isinstance(node, ast.ClassDef):
            rows.append((_start(node), node.end_lineno, "class", node.name, _first_doc_line(node)))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows.append((_start(sub), sub.end_lineno, "  def", f"{node.name}.{sub.name}",
                                 _first_doc_line(sub)))
    return rows


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _py_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
    return [target]


def _print_rows(path: Path, rows: list[tuple[int, int, str, str, str]]) -> None:
    for start, end, kind, name, doc in rows:
        tail = f"  — {doc}" if doc else ""
        print(f"{_rel(path)}:{start}-{end}  {kind} {name}{tail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to index")
    parser.add_argument("--find", help="print defs whose name contains this text (searches --root)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="search root for --find / --big")
    parser.add_argument("--big", type=int, metavar="N", help="list modules under --root longer than N lines")
    args = parser.parse_args(argv)

    if args.big is not None:
        sizes = []
        for path in _py_files(args.root):
            n = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
            if n > args.big:
                sizes.append((n, path))
        for n, path in sorted(sizes, reverse=True):
            print(f"{n:6d}  {_rel(path)}")
        return 0

    if args.find:
        needle = args.find
        for path in _py_files(args.root):
            rows = [r for r in index_file(path) if needle in r[3]]
            _print_rows(path, rows)
        return 0

    if not args.paths:
        parser.error("give a file/directory, or --find NAME, or --big N")
    for target in args.paths:
        for path in _py_files(target):
            _print_rows(path, index_file(path))
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.stdout.flush()
        sys.exit(rc)
    except BrokenPipeError:  # output piped into head/grep that closed early
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
