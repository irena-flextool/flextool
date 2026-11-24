"""Test modules that import the code you changed, so a change runs its own tests and not a sweep.

A test module is related to a changed source module when it imports it by name (``import a.b.c``,
``from a.b.c import x`` or ``from a.b import c``).  Importing only the package (``from flextool.engine import
model``) does not count: that would select half the suite.  Changed test modules are related to themselves.
End-to-end tests that reach the code only through a solve are not found this way; prove the LP effect of a
model change with ``tools/lp_snapshot.py``.

Usage::

    python tools/related_tests.py                       # files changed against HEAD (+ untracked)
    python tools/related_tests.py --base main           # files changed since the merge base with main
    python tools/related_tests.py flextool/engine/model/_online.py
    python tools/related_tests.py --run                 # run them: pytest -n 8 -q --tb=short

Each line is ``<seconds in CI> <test module>``, slowest last (seconds from ``tests/ci_durations.json``; ``?``
when the module has no entry yet).  ``--run`` leaves out modules over ``HEAVY_SECONDS`` (run single tests of
those with ``<module>::<test>``) unless ``--include-heavy``.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEAVY_SECONDS = 300.0


def _git_lines(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True)
    return [line for line in out.stdout.splitlines() if line]


def changed_files(base: str | None) -> list[str]:
    if base:
        return _git_lines("diff", "--name-only", f"{base}...HEAD") + _git_lines("diff", "--name-only", "HEAD")
    return (_git_lines("diff", "--name-only", "HEAD")
            + _git_lines("ls-files", "--others", "--exclude-standard"))


def module_name(rel: str) -> str | None:
    if not rel.endswith(".py") or not rel.startswith(("flextool/", "flextool_")):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def related(files: list[str]) -> list[str]:
    targets = {m for m in map(module_name, files) if m}
    tests = {f for f in files if f.startswith("tests/") and Path(f).name.startswith("test_")
             and (REPO / f).exists()}
    if targets:
        for path in (REPO / "tests").rglob("test_*.py"):
            if imported_modules(path) & targets:
                tests.add(path.relative_to(REPO).as_posix())
    return sorted(tests)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="changed files (default: git diff against HEAD + untracked)")
    ap.add_argument("--base", help="also include commits since the merge base with this ref")
    ap.add_argument("--run", action="store_true", help="run the related modules with pytest -n 8")
    ap.add_argument("--include-heavy", action="store_true", help=f"--run also modules over {HEAVY_SECONDS:.0f} s")
    args = ap.parse_args(argv)
    files = args.files or changed_files(args.base)
    tests = related(files)
    durations = json.loads((REPO / "tests" / "ci_durations.json").read_text(encoding="utf-8"))
    tests.sort(key=lambda t: durations.get(t, 0.0))
    heavy = {t for t in tests if durations.get(t, 0.0) > HEAVY_SECONDS}
    for t in tests:
        sec = durations.get(t)
        print(f"{'?' if sec is None else f'{sec:.0f}':>6} {t}{'  (heavy: run single tests)' if t in heavy else ''}")
    if not tests:
        print("no related test modules (end-to-end coverage: tools/lp_snapshot.py)", file=sys.stderr)
        return 0
    if not args.run:
        return 0
    to_run = tests if args.include_heavy else [t for t in tests if t not in heavy]
    if not to_run:
        return 0
    cmd = [sys.executable, "-m", "pytest", "-n", "8", "-q", "--tb=short"]
    if any(t.startswith("tests/csv_reload/") for t in to_run):
        # The opt-in csv_reload tests run only when -m names them (tests/csv_reload/conftest.py).
        cmd += ["-m", "csv_reload or not (csv_reload or axis_enums or commercial_shim)"]
    return subprocess.call([*cmd, *to_run], cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
