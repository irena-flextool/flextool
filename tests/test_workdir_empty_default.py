"""Verify Step 2.5 item 19: a default cascade run does not leak the
**input-cascade** CSVs to the workdir.

Pre-Step-2.5 the cascade emitted ~150 ``input/*.csv`` and ``solve_data/
*.csv`` files unconditionally — the writer-port modules wrote frames
through ``path.open("w")``.  Step 2.5 routed every input-time and
preprocessing-time writer through the :class:`FlexDataProvider` and
gated disk emission behind the orchestrator-level ``csv_dump`` flag
(default ``False``).

What this test asserts
----------------------

1. ``workdir/input/`` is empty of CSVs.  ``input_derivation.run`` and
   the SpineDBBackend spec loops now populate the Provider; no
   ``input/*.csv`` should hit disk in default mode.
2. The preprocessing **writer-port cascade** does not leak any of its
   canonical ``solve_data/*.csv`` outputs to disk.  We assert this via
   a whitelist intersection: any basename present in
   :attr:`flextool.engine_polars._flex_data_accumulator._THIN_WRAPPER_BASENAMES`
   that ends up on disk indicates a bug in the ``capture_frames``
   plumbing.

What this test does NOT assert
------------------------------

The post-solve **handoff / output writers** (``period_capacity.csv``,
``fix_storage_*.csv``, block-layout CSVs, etc.) intentionally write
to ``solve_data/`` as cross-roll carriers; they execute outside the
``capture_frames`` context.  These are out of Step 2.5 scope (handoff
writers are scheduled for Step 2.6+) and so the test treats CSVs not
in the writer-port manifest as allowed.
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
from flextool.engine_polars._flex_data_accumulator import (  # noqa: E402
    _THIN_WRAPPER_BASENAMES,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


@pytest.fixture(scope="module")
def small_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Tiny scenario reused by the workdir-emptiness check.

    The base ``tests.json`` fixture's ``coal`` scenario solves in <1s
    and exercises the full preprocessing chain, which is exactly what
    we want to verify.
    """
    db_path = tmp_path_factory.mktemp("db_workdir_empty") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    return url


def _csv_files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.csv"))


class TestWorkdirEmptyByDefault:
    """Default cascade (no ``--csv-dump``) keeps preprocessing CSVs off disk."""

    def test_default_run_leaves_input_dir_csv_free(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """``workdir/input/`` has no CSVs in default mode.

        Step 2.5 contract: the SpineDBBackend spec loops feed every
        ``input/*.csv`` frame directly into the Provider via
        :func:`flextool.input_derivation.run`.  No disk emission
        happens unless ``--csv-dump`` is on.
        """
        workdir = tmp_path_factory.mktemp("workdir_empty_default")
        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            steps = run_chain_from_db(
                small_db_url, "coal",
                work_folder=workdir,
                # csv_dump intentionally NOT set — the default cascade
                # must keep input-cascade CSVs off disk.
            )
        finally:
            os.chdir(cwd0)

        last_step = next(reversed(list(steps.values())))
        assert last_step.optimal, "cascade must solve cleanly for the gate"

        input_csvs = _csv_files_under(workdir / "input")
        assert not input_csvs, (
            f"workdir/input/ should have no CSVs without --csv-dump; "
            f"found {[p.name for p in input_csvs]}.  This means a "
            f"SpineDBBackend spec loop or a writer-port _write call "
            f"still hits disk; route it through provider.put()."
        )

    # Cross-roll handoff carriers: these basenames ALSO appear in the
    # writer-port manifest because preprocessing populates an initial
    # version, but the post-solve handoff_writers in
    # ``flextool.process_outputs.handoff_writers`` overwrite them on
    # disk for the next roll to read.  Step 2.6+ will retire that disk
    # dependency; for now they are documented exceptions to the
    # writer-port-only invariant.
    _HANDOFF_OVERWRITES: frozenset[str] = frozenset({
        "fix_storage_price.csv",
        "fix_storage_quantity.csv",
        "fix_storage_usage.csv",
        "p_entity_divested.csv",
        "p_entity_invested.csv",
        "p_entity_pre_existing.csv",
        "p_entity_period_existing_capacity.csv",
        "period_capacity.csv",
        "scale_the_objective.csv",
    })

    def test_writer_port_manifest_csvs_stay_in_memory(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """No canonical writer-port basename appears on disk in default mode.

        ``_flex_data_accumulator._THIN_WRAPPER_BASENAMES`` lists every
        ``solve_data/*.csv`` produced by the preprocessing writer-port
        cascade.  Without ``--csv-dump`` none of them should hit
        ``workdir/solve_data/`` — :func:`capture_frames` redirects
        every ``_write(df, path)`` into ``provider.put(name, df)``
        and skips the disk emission.

        Exceptions: a small set of basenames listed in
        :attr:`_HANDOFF_OVERWRITES` are written by the post-solve
        handoff writers (cross-roll carriers, Step 2.6 scope).
        """
        workdir = tmp_path_factory.mktemp("workdir_empty_manifest")
        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            steps = run_chain_from_db(
                small_db_url, "coal", work_folder=workdir,
            )
        finally:
            os.chdir(cwd0)
        last_step = next(reversed(list(steps.values())))
        assert last_step.optimal

        manifest = set(_THIN_WRAPPER_BASENAMES) - self._HANDOFF_OVERWRITES
        leaked = [
            p for p in _csv_files_under(workdir / "solve_data")
            if p.name in manifest
        ]
        assert not leaked, (
            f"writer-port manifest CSVs leaked to disk without "
            f"--csv-dump: {[p.name for p in leaked]}.  "
            f"capture_frames(provider=...) should keep these in the "
            f"Provider; investigate the _write helper on the emitting "
            f"writer module."
        )

    def test_default_run_writes_parquet_outputs(
        self,
        small_db_url: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """The cascade still writes parquet outputs — that's the
        canonical post-solve artefact location, independent of the
        ``--csv-dump`` debug toggle.
        """
        workdir = tmp_path_factory.mktemp("workdir_empty_parquet")
        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            steps = run_chain_from_db(
                small_db_url, "coal", work_folder=workdir,
            )
        finally:
            os.chdir(cwd0)
        last_step = next(reversed(list(steps.values())))
        assert last_step.optimal

        output_raw = workdir / "output_raw"
        parquets = sorted(output_raw.rglob("*.parquet"))
        assert parquets, (
            f"cascade must emit parquet outputs into {output_raw}; "
            f"found none"
        )
