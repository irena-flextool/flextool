"""Verify Step 2.5 item 20: ``--csv-dump`` round-trips through the
:class:`FlexDataProvider`.

The cascade in Step 2.5 runs purely in-memory by default; the
``--csv-dump`` flag toggles two snapshot paths:

* ``data.dump_csvs(work_folder)`` — per-solve snapshot of every
  ``solve_data/*.csv`` artefact (executed inside the per-iter handoff
  in :mod:`flextool.engine_polars._orchestration`).
* :func:`FlexDataProvider.snapshot_processed_inputs` — post-cascade
  snapshot of every ``input/*.csv`` derived frame the writer-port
  modules produced.

This test runs a small fixture through :func:`run_chain_from_db` with
``csv_dump=True`` and asserts the round-trip materialises both shards.

Hard rules from Step 2.5:

* ``--csv-dump`` must NOT change cascade arithmetic — the LP / outputs
  are byte-equivalent regardless of the flag.  We don't re-assert the
  full numeric parity here (that's covered by every other cascade
  test); we only assert the new CSV artefacts appear when the flag is
  on.
* The dump must include at least the per-solve ``solve_data/`` shard
  (``data.dump_csvs``); the ``input/`` snapshot via
  ``snapshot_processed_inputs`` is exercised by the CLI wrapper in
  :mod:`flextool.cli.cmd_run_flextool`, not by ``run_chain_from_db``
  itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


@pytest.fixture(scope="module")
def small_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Same fixture as :mod:`tests.test_workdir_empty_default`, copied
    here so the two files have independent module-scoped state.
    """
    db_path = tmp_path_factory.mktemp("db_csv_dump") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    return url


def _csv_files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.csv"))


class TestCsvDumpRoundTrip:
    """``csv_dump=True`` materialises the per-solve ``solve_data/``
    shard via :func:`FlexData.dump_csvs`.

    The post-cascade ``snapshot_processed_inputs`` is a separate
    one-way dump (driven by the CLI wrapper, not by the orchestrator);
    we exercise it explicitly below for completeness.
    """

    def test_csv_dump_writes_solve_data_csvs(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        workdir = tmp_path_factory.mktemp("csv_dump_solve_data")
        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            steps = run_chain_from_db(
                small_db_url, "coal",
                work_folder=workdir,
                csv_dump=True,
            )
        finally:
            os.chdir(cwd0)
        last_step = next(reversed(list(steps.values())))
        assert last_step.optimal

        solve_data_csvs = _csv_files_under(workdir / "solve_data")
        assert solve_data_csvs, (
            f"with csv_dump=True the cascade should snapshot to "
            f"{workdir / 'solve_data'} but no CSVs were found"
        )

    def test_snapshot_processed_inputs_materialises_input_csvs(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """The Provider's ``snapshot_processed_inputs`` writes every
        derived ``input/*.csv`` frame for off-cascade tooling.  This
        is the dump invoked by the CLI wrapper after a successful run.
        """
        workdir = tmp_path_factory.mktemp("csv_dump_snapshot")
        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            steps = run_chain_from_db(
                small_db_url, "coal",
                work_folder=workdir,
                csv_dump=True,
                keep_solutions=True,
            )
        finally:
            os.chdir(cwd0)
        last_step = next(reversed(list(steps.values())))
        provider = getattr(last_step, "flex_data_provider", None)
        assert provider is not None, (
            "keep_solutions=True should retain the Provider for "
            "post-cascade tooling"
        )
        provider.snapshot_processed_inputs(workdir)

        input_csvs = _csv_files_under(workdir / "input")
        assert input_csvs, (
            f"snapshot_processed_inputs should populate {workdir / 'input'}"
        )
