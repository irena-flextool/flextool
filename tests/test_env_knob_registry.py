"""Every environment variable FlexTool reads is registered and documented, and no page lists a dead one.

The registry is :data:`flextool.run_options.OPTIONS` (variables that are the twin of a flag) plus
:data:`flextool.run_options.ENV_KNOBS` (process-level, developer and test variables).  The checks:

* every variable name in ``flextool/`` and the top-level modules (``[tool.setuptools] py-modules``, such as the
  ``flextool-run`` supervisor and the ``flextool-assistant`` entry) -- a string literal ``FLEXTOOL_*`` / ``POLARS_*`` / ``POLAR_HIGH_*`` /
  ``MALLOC_*`` / ``_RJEM_*`` (a direct read or a ``*_ENV = "..."`` constant), or any literal passed to
  ``os.environ[...]`` / ``.get`` / ``.setdefault`` / ``.pop`` / ``os.getenv`` / ``in os.environ`` -- is
  registered, except a few system variables FlexTool reads but does not define;
* every ``FLEXTOOL_*`` name the test suite reads is registered (test-only knobs go to ``docs/dev/testing.md``);
* every registered variable that FlexTool reads is still read somewhere;
* every registered variable is on the page of its audience, and the variable tables of the three pages list
  only variables registered to that page;
* no documentation page mentions a ``FLEXTOOL_*`` name that is not registered.

Add a new variable to the registry and to its page, and this test passes again.
"""
from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from flextool.run_options import AUDIENCE_PAGES, ENV_KNOBS, OPTIONS

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "flextool"
TESTS = REPO / "tests"
#: Shipped modules outside the package (they may not import it, so they name their variables themselves).
TOP_LEVEL_MODULES = [
    REPO / f"{name}.py"
    for name in tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"].get(
        "py-modules", []
    )
]

_PREFIXED = re.compile(r"^(FLEXTOOL|POLARS|POLAR_HIGH|MALLOC|_RJEM)_[A-Z0-9_]+$")
# Read by FlexTool, defined by the operating system / desktop / Python, not by FlexTool.
_SYSTEM_VARIABLES = frozenset({"PATH", "VIRTUAL_ENV", "GDK_SCALE"})
# FLEXTOOL_-prefixed names that are not environment variables.
_NOT_ENV = frozenset({"FLEXTOOL_DB_VERSION"})
# The registry itself names every variable; it does not count as a reader.
_REGISTRY_MODULE = PACKAGE / "run_options.py"

_ENV_CALL_SUFFIXES = ("environ.get", "environ.setdefault", "environ.pop", "getenv")


def _option_env_pages() -> dict[str, str]:
    return {spec.env: AUDIENCE_PAGES[spec.audience] for spec in OPTIONS.values() if spec.env}


def _registered_pages() -> dict[str, str]:
    pages = _option_env_pages()
    pages.update({name: knob.page for name, knob in ENV_KNOBS.items()})
    return pages


def _python_files(root: Path) -> list[Path]:
    """The Python files under *root* that belong to the repository: tracked or new, not git-ignored.

    Git-ignored files (local-only tests, scratch scripts) are left out so the result is the same on every
    checkout.  Without git (an unpacked source archive) every file counts.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", str(root)],
            cwd=REPO, capture_output=True, text=True, check=True, timeout=60,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return sorted(root.rglob("*.py"))
    return sorted(REPO / name for name in listed if name.endswith(".py") and (REPO / name).is_file())


def _env_names_in(root: Path, *, prefixed_only: bool) -> dict[str, list[str]]:
    """Variable names used in the Python files under *root*, with where each is used."""
    found: dict[str, list[str]] = {}
    for path in _python_files(root):
        if path == _REGISTRY_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        where = f"{path.relative_to(REPO)}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _PREFIXED.match(node.value):
                found.setdefault(node.value, []).append(f"{where}:{node.lineno}")
                continue
            if prefixed_only:
                continue
            arg = None
            if isinstance(node, ast.Call) and node.args and ast.unparse(node.func).endswith(_ENV_CALL_SUFFIXES):
                arg = node.args[0]
            elif isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("environ"):
                arg = node.slice
            elif isinstance(node, ast.Compare) and any(
                ast.unparse(c).endswith("environ") for c in node.comparators
            ):
                arg = node.left
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.setdefault(arg.value, []).append(f"{where}:{node.lineno}")
    return found


@pytest.fixture(scope="module")
def package_names() -> dict[str, list[str]]:
    names = _env_names_in(PACKAGE, prefixed_only=False)
    for module in TOP_LEVEL_MODULES:
        for name, where in _env_names_in(module, prefixed_only=False).items():
            names.setdefault(name, []).extend(where)
    return names


def test_top_level_modules_are_scanned():
    assert REPO / "flextool_run_supervisor.py" in TOP_LEVEL_MODULES
    assert REPO / "flextool_assistant_entry.py" in TOP_LEVEL_MODULES
    assert all(_python_files(module) == [module] for module in TOP_LEVEL_MODULES)


@pytest.fixture(scope="module")
def test_suite_names() -> dict[str, list[str]]:
    names = _env_names_in(TESTS, prefixed_only=True)
    return {name: where for name, where in names.items() if name.startswith("FLEXTOOL_")}


def test_every_variable_flextool_reads_is_registered(package_names):
    registered = _registered_pages()
    missing = {
        name: where[:3] for name, where in package_names.items()
        if name not in registered and name not in _SYSTEM_VARIABLES and name not in _NOT_ENV
    }
    assert not missing, (
        "environment variables read in flextool/ or a top-level module but not registered in flextool/run_options.py "
        f"(OPTIONS or ENV_KNOBS): {missing}"
    )


def test_every_variable_the_test_suite_reads_is_registered(test_suite_names):
    registered = _registered_pages()
    missing = {
        name: where[:3] for name, where in test_suite_names.items()
        if name not in registered and name not in _NOT_ENV
    }
    assert not missing, f"FLEXTOOL_* variables used in tests/ but not registered: {missing}"


def test_every_registered_knob_is_still_read(package_names, test_suite_names):
    """A registered variable nobody reads any more is dead: remove it from the registry and its page."""
    dead = []
    for name, knob in ENV_KNOBS.items():
        if knob.read_by != "flextool":
            continue
        readers = test_suite_names if knob.audience == "test" else package_names
        if name not in readers:
            dead.append(name)
    assert not dead, f"registered but no longer read: {dead}"


def test_audiences_have_pages():
    for spec in OPTIONS.values():
        assert spec.audience in AUDIENCE_PAGES, spec.key
    for knob in ENV_KNOBS.values():
        assert knob.page == AUDIENCE_PAGES[knob.audience], knob.name
    assert not set(_option_env_pages()) & set(ENV_KNOBS), "a variable is both a flag twin and a knob"


@pytest.mark.parametrize("page", sorted(set(AUDIENCE_PAGES.values())))
def test_each_registered_variable_is_on_its_page(page):
    text = (REPO / page).read_text(encoding="utf-8")
    absent = sorted(name for name, name_page in _registered_pages().items()
                    if name_page == page and f"`{name}`" not in text)
    assert not absent, f"registered for {page} but not described there: {absent}"


_TABLE_NAME = re.compile(r"^\|\s*`([A-Z_][A-Z0-9_]+)`")


@pytest.mark.parametrize("page", sorted(set(AUDIENCE_PAGES.values())))
def test_page_tables_list_only_variables_registered_to_the_page(page):
    registered = _registered_pages()
    wrong = []
    for line in (REPO / page).read_text(encoding="utf-8").splitlines():
        match = _TABLE_NAME.match(line)
        if match is None or not _PREFIXED.match(match.group(1)):
            continue
        name = match.group(1)
        if registered.get(name) != page:
            wrong.append((name, registered.get(name, "not registered")))
    assert not wrong, f"{page} lists variables that belong elsewhere or are not read: {wrong}"


_DOC_NAME = re.compile(r"\bFLEXTOOL_[A-Z0-9_]*[A-Z0-9](?![A-Z0-9_]*\*)")


def test_documentation_mentions_only_registered_flextool_variables():
    registered = _registered_pages()
    docs = sorted((REPO / "docs").rglob("*.md")) + [REPO / "CONTRIBUTING.md", REPO / "README.md"]
    stale = {}
    for path in docs:
        if not path.is_file():
            continue
        for name in _DOC_NAME.findall(path.read_text(encoding="utf-8")):
            if name not in registered and name not in _NOT_ENV:
                stale.setdefault(name, str(path.relative_to(REPO)))
    assert not stale, f"documentation names FLEXTOOL_* variables that FlexTool no longer reads: {stale}"
