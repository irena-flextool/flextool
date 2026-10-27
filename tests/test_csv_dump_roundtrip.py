"""Verify Step 2.5 item 20: ``--csv-dump`` round-trips through the
:class:`FlexDataProvider`.

The cascade in Step 2.5 runs purely in-memory by default; the
``--csv-dump`` flag toggles two snapshot paths:

* ``data.dump_csvs(work_folder)`` — per-solve snapshot of every
  ``solve_data/*.csv`` artefact (executed inside the per-iter handoff
  in :mod:`flextool.engine.solve._orchestration`).
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
* The dump must include the per-solve ``input/`` + ``solve_data/`` shard
  (``data.dump_csvs``, once per sub-solve into the one work folder); the
  ``input/`` snapshot via ``snapshot_processed_inputs`` is exercised by the
  CLI wrapper in :mod:`flextool.cli.cmd_run_flextool`, not by
  ``run_chain_from_db`` itself.
* A provider snapshot carries the objective scale the solve was built with
  (one row), so an offline reload does not need it passed separately.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.engine.preprocess._dump_stamp import read_dump_stamp  # noqa: E402
from flextool.engine.preprocess.flex_data import FlexData  # noqa: E402
from flextool.engine.solve import _orchestration  # noqa: E402
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


def _run(db_url: str, scenario: str, workdir: Path, **kwargs):
    cwd0 = os.getcwd()
    os.chdir(workdir)
    try:
        return run_chain_from_db(db_url, scenario, work_folder=workdir, **kwargs)
    finally:
        os.chdir(cwd0)


@pytest.fixture
def dump_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every ``FlexData.dump_csvs`` call (the per-sub-solve dump) and pass it through."""
    calls: list[Path] = []
    real = FlexData.dump_csvs

    def spy(self, workdir, *args, **kwargs):
        calls.append(Path(workdir))
        return real(self, workdir, *args, **kwargs)

    monkeypatch.setattr(FlexData, "dump_csvs", spy)
    return calls


class TestCsvDumpRoundTrip:
    """``csv_dump=True`` materialises the per-solve ``input/`` +
    ``solve_data/`` shard via :func:`FlexData.dump_csvs`.

    The post-cascade ``snapshot_processed_inputs`` is a separate
    one-way dump (driven by the CLI wrapper, not by the orchestrator);
    we exercise it explicitly below for completeness.
    """

    def test_csv_dump_writes_solve_data_csvs(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
        dump_calls: list[Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``csv_dump=True`` runs the per-sub-solve ``dump_csvs`` into the work folder, without errors."""
        workdir = tmp_path_factory.mktemp("csv_dump_solve_data")
        with caplog.at_level(logging.ERROR):
            steps = _run(small_db_url, "coal", workdir, csv_dump=True)
        last_step = next(reversed(list(steps.values())))
        assert last_step.optimal

        assert dump_calls, "csv_dump=True never called dump_csvs (the flag did not reach the per-sub-solve gate)"
        assert set(dump_calls) == {workdir}
        # Written only by ``dump_csvs``.
        assert (workdir / "input" / "p_entity_unitsize.csv").is_file()
        stamp = read_dump_stamp(workdir)
        assert stamp is not None and "dump_csvs" in stamp["writers"], stamp
        solve_data_csvs = _csv_files_under(workdir / "solve_data")
        assert solve_data_csvs, (
            f"with csv_dump=True the cascade should snapshot to "
            f"{workdir / 'solve_data'} but no CSVs were found"
        )
        failed = [r.getMessage() for r in caplog.records if "dump_csvs failed" in r.getMessage()]
        assert not failed, failed

    def test_no_csv_dump_writes_no_dump(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
        dump_calls: list[Path],
    ) -> None:
        """Without the flag the run writes no per-sub-solve dump and no stamp."""
        workdir = tmp_path_factory.mktemp("no_csv_dump")
        steps = _run(small_db_url, "coal", workdir)
        assert next(reversed(list(steps.values()))).optimal
        assert dump_calls == []
        assert not (workdir / "input" / "p_entity_unitsize.csv").exists()
        assert read_dump_stamp(workdir) is None

    def test_csv_dump_keeps_the_lp(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
        dump_calls: list[Path],
    ) -> None:
        """A multi-solve run with and without the dump: same steps, objectives and variable names."""
        scenario = "5weeks_invest_fullYear_dispatch_coal_wind"
        plain = _run(small_db_url, scenario, tmp_path_factory.mktemp("lp_plain"), keep_solutions=True)
        assert dump_calls == []
        dumped = _run(small_db_url, scenario, tmp_path_factory.mktemp("lp_dump"), keep_solutions=True,
                      csv_dump=True)
        assert len(dump_calls) == len(dumped) > 1
        assert list(plain) == list(dumped)
        for key in plain:
            a, b = plain[key], dumped[key]
            assert a.optimal and b.optimal, key
            assert b.obj == pytest.approx(a.obj, rel=1e-12), key
            assert set(a.solution._vars) == set(b.solution._vars), key

    def test_snapshot_carries_the_objective_scale(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A provider snapshot of a solved run holds the solve's objective scale as one row.

        The scale is forced to a non-default value so the check cannot pass on the 1e-6 fallback; the solve's own
        ``solve_data/scale_the_objective.csv`` in the work folder must hold the same value.
        """
        scale = 0.25
        monkeypatch.setattr(_orchestration, "_resolve_effective_obj_scale", lambda _value: scale)
        workdir = tmp_path_factory.mktemp("scale_work")
        steps = _run(small_db_url, "coal", workdir, csv_dump=True, keep_solutions=True)
        last_step = next(reversed(list(steps.values())))
        dump = tmp_path_factory.mktemp("scale_dump")
        last_step.flex_data_provider.snapshot_processed_inputs(dump)
        for folder in (dump, workdir):
            frame = pd.read_csv(folder / "solve_data" / "scale_the_objective.csv")
            assert frame.shape[0] == 1, f"{folder}: {frame}"
            assert float(frame["value"].iloc[0]) == scale, f"{folder}: {frame}"
        # The CLI writes the snapshot into the work folder itself, over the file the solve wrote.
        last_step.flex_data_provider.snapshot_processed_inputs(workdir)
        frame = pd.read_csv(workdir / "solve_data" / "scale_the_objective.csv")
        assert frame.shape[0] == 1 and float(frame["value"].iloc[0]) == scale

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
        steps = _run(small_db_url, "coal", workdir, csv_dump=True, keep_solutions=True)
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
