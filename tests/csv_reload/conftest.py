"""CSV reload contract tests: opt-in, run with ``pytest -m csv_reload tests/csv_reload/``.

These tests load a CSV work folder (a provider snapshot, a ``dump_csvs``
tree or hand-written CSVs) with the unsupported reload package
``flextool.engine._csv_reload`` (``load_from_dump``; a ``dump_csvs`` tree
first gets the run's metadata CSVs with ``copy_runner_meta``).
Production never does that: it loads every sub-solve from the in-memory
provider with a DB reader (``tests/_support/load.py::load_scenario``).
They pin the CSV reload path (``--csv-dump`` debugging, the reload-only
seeds of the loader) and are kept apart from the production-path suite
(c24 stage 1, S1-8).

Every test collected under this folder gets the ``csv_reload`` marker.
The default ``addopts`` (``-m "not csv_reload"``) deselects them.  A
command-line ``-m`` replaces the addopts one (CI's ``-m smoke`` /
``-m slow``), so they are also skipped unless the ``-m`` expression names
``csv_reload``.

Reload-only fixtures of the loader tests (moved from
``tests/engine/loaders/conftest.py``):

* ``tiny_workdir(tmp_path)`` copies the session-cached ``base`` work
  folder (provider snapshot CSVs + ``tests.sqlite``) into ``tmp_path``;
  ``load_from_dump(tiny_workdir)`` is a reader-less CSV load of it (the
  copy keeps the work folder's dump stamp).
* ``write_csv(workdir, name, rows)`` writes a CSV overlay under the
  matching sub-directory (bare names → ``solve_data/``).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import polars as pl
import pytest

_HERE = Path(__file__).resolve().parent
_NAMES_CSV_RELOAD = re.compile(r"\bcsv_reload\b")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark this folder's tests ``csv_reload``; skip them unless ``-m`` names ``csv_reload``.

    ``tryfirst`` so the marker is on the items before ``-m`` deselects.
    """
    named = bool(_NAMES_CSV_RELOAD.search(config.getoption("markexpr", "") or ""))
    skip = pytest.mark.skip(
        reason="CSV reload contract test: run with -m csv_reload (tests/csv_reload/)"
    )
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if _HERE not in path.resolve().parents:
            continue
        item.add_marker(pytest.mark.csv_reload)
        if not named:
            item.add_marker(skip)


@pytest.fixture(scope="function")
def tiny_workdir(tmp_path: Path, scenario_workdir) -> Path:
    """Copy the session-cached ``base`` work folder into ``tmp_path / 'work'``.

    Reload-only: ``load_from_dump(tiny_workdir)`` reads the snapshot CSVs
    without a DB reader (no ``db_url``; the copied ``tests.sqlite`` is unused).
    Production-path loader tests use ``tiny_run``
    (``tests/engine/loaders/conftest.py``).
    """
    source = scenario_workdir("base")
    target = tmp_path / "work"
    shutil.copytree(source, target)
    return target


def write_csv(workdir: Path, name: str,
              rows: "list[dict] | pl.DataFrame") -> Path:
    """Write ``rows`` as a CSV under ``workdir / <subdir> / <basename>`` (reload-only overlays).

    ``name`` may be ``"foo.csv"`` (→ ``solve_data/foo.csv``) or
    ``"input/foo.csv"`` / ``"solve_data/foo.csv"``.  ``rows`` is a list of
    dicts or a ``pl.DataFrame``.  Returns the path written; overwrites.
    """
    target = workdir / name if "/" in name else workdir / "solve_data" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    df.write_csv(target)
    return target
