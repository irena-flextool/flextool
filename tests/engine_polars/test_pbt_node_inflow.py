"""Phase B parity test for stochastic ``pbt_node_inflow`` fold-in.

The augmented fixture
``tests/fixtures/stochastics_pbt_inflow.json`` carries a 3d_map
``node.inflow`` on ``hydro_reservoir`` with 4 branches
(``realized``, ``upper``, ``lower``, ``mid``) × 1 ``time_start``
(``t0001``) × 48 timesteps.  The hand-derived parity oracle lives at
``tests/expected/stochastics_pbt_inflow/golden_pdtNodeInflow.csv``;
the 192 ``hydro_reservoir`` rows are the Branch 1 + Branch 2 fold-in
output of the legacy preprocessing.

This test runs the FlexToolRunner to produce a workdir (so the
required ``solve_data/`` scaffolding CSVs exist), then directly
invokes :func:`apply_p_inflow_with_scaling` and asserts the resulting
``flex_data.p_inflow`` matches the golden for the ``hydro_reservoir``
rows.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import polars as pl
import pytest

from tests.db_utils import json_to_db

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
EXPECTED_DIR = Path(__file__).resolve().parents[1] / "expected" / "stochastics_pbt_inflow"


@pytest.fixture(scope="module")
def stochastic_pbt_inflow_db_url(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Stochastic + pbt_node_inflow augmented DB.

    Imports the augmented JSON fixture into a fresh SQLite DB once per
    module, then migrates to current ``FLEXTOOL_DB_VERSION``.
    """
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path_factory.mktemp("db_pbt") / "stochastics_pbt_inflow.sqlite"
    url = json_to_db(
        FIXTURES_DIR / "stochastics_pbt_inflow.json", db_path,
    )
    migrate_database(url)
    return url


@pytest.fixture(scope="module")
def stochastic_pbt_inflow_workdir(
    stochastic_pbt_inflow_db_url: str,
    test_bin_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the native cascade on the augmented fixture to materialise
    the workdir CSVs that the fold-in helper consumes.

    ``run_chain_from_db`` runs the legacy preprocessing (``write_input``
    + per-solve set / time slice builders) before each native LP build,
    so the produced workdir carries every ``solve_data/`` CSV the
    fold-in helper needs (``steps_in_use.csv``, ``first_timesteps.csv``,
    ``solve_branch__time_branch.csv``, ``period__branch.csv``,
    ``nodeBalance.csv``, ...).
    """
    from flextool.engine_polars import run_chain_from_db

    workdir = tmp_path_factory.mktemp("pbt_inflow_run")
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        run_chain_from_db(
            stochastic_pbt_inflow_db_url,
            "2_day_stochastic_dispatch",
            work_folder=workdir,
        )
    finally:
        os.chdir(cwd)
    return workdir


def _load_golden_hydro_reservoir() -> pl.DataFrame:
    """Load the 192 ``hydro_reservoir`` rows from the parity oracle."""
    df = pl.read_csv(EXPECTED_DIR / "golden_pdtNodeInflow.csv")
    return (df.filter(pl.col("node") == "hydro_reservoir")
              .rename({"node": "n", "period": "d", "time": "t"})
              .select("n", "d", "t", "value")
              .sort("n", "d", "t"))


def test_pbt_node_inflow_branch1_parity_hydro_reservoir(
    stochastic_pbt_inflow_db_url: str,
    stochastic_pbt_inflow_workdir: Path,
) -> None:
    """Branch 1 + Branch 2 fold-in matches the legacy golden on
    ``hydro_reservoir`` (192 rows = 4 periods × 48 timesteps).
    """
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._inflow_scaling import (
        apply_p_inflow_with_scaling,
    )

    source = SpineDbReader(
        stochastic_pbt_inflow_db_url,
        scenario="2_day_stochastic_dispatch",
    )

    workdir = stochastic_pbt_inflow_workdir

    # Build the active dt frame from steps_in_use.csv (the same source
    # the legacy ``write_pdtNodeInflow`` uses).
    su_path = workdir / "solve_data" / "steps_in_use.csv"
    assert su_path.exists(), f"missing {su_path}"
    dt_rows: list[tuple[str, str]] = []
    with su_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                dt_rows.append((row[0], row[1]))
    assert dt_rows, "steps_in_use.csv is empty"
    dt = pl.DataFrame(
        {"d": [r[0] for r in dt_rows], "t": [r[1] for r in dt_rows]}
    )

    # Build a minimal flex_data stand-in: only ``p_inflow`` is touched
    # (set / read) and ``nodeBalance`` is consulted for the balance gate.
    class _FlexData:
        p_inflow: object | None = None
        nodeBalance: object | None = None

    flex_data = _FlexData()

    # Read nodeBalance for the helper.
    nb_path = workdir / "solve_data" / "nodeBalance.csv"
    if nb_path.exists():
        try:
            flex_data.nodeBalance = pl.read_csv(nb_path)
        except Exception:
            flex_data.nodeBalance = None

    ok = apply_p_inflow_with_scaling(
        flex_data, source, workdir, dt, per_solve_aggs=None,
    )
    assert ok, "apply_p_inflow_with_scaling returned False on pbt fixture"
    p_inflow = flex_data.p_inflow
    assert p_inflow is not None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    assert fr.height > 0

    got = (fr.filter(pl.col("n") == "hydro_reservoir")
             .select("n", "d", "t", "value")
             .sort("n", "d", "t"))
    golden = _load_golden_hydro_reservoir()

    assert got.height == golden.height, (
        f"row count mismatch: got {got.height}, expected {golden.height}"
    )
    assert got.height == 192, (
        f"unexpected row count {got.height}; should be 4×48=192"
    )

    # Compare row-by-row.  LP coefficients are exact floats — tolerance
    # 1e-9 abs is more than enough but defends against double-rounding
    # in the polars cast chain.
    joined = got.join(
        golden.rename({"value": "expected"}),
        on=["n", "d", "t"], how="inner",
    )
    assert joined.height == 192
    diffs = joined.with_columns(
        diff=(pl.col("value") - pl.col("expected")).abs()
    )
    max_diff = diffs["diff"].max()
    assert max_diff is not None and max_diff < 1e-9, (
        f"max abs diff = {max_diff!r}; first mismatches:\n"
        f"{diffs.sort('diff', descending=True).head(5)}"
    )
