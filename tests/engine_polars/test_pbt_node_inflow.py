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


# ---------------------------------------------------------------------------
# Phase C — Fixture A: multi-``time_start`` Branch 1 fold on ``downriver``.
# ---------------------------------------------------------------------------


def _prepare_workdir(json_fixture: Path, tmp_path: Path) -> tuple[str, Path]:
    """Migrate ``json_fixture`` into a sqlite db, run the cascade to
    materialise the workdir CSVs, and return ``(db_url, workdir)``.

    Used by the Phase C parity tests (which then mutate
    ``first_timesteps.csv`` in the workdir before invoking the fold-in).

    The cascade is invoked the same way the Phase B parity test does it
    (``run_chain_from_db`` on the ``2_day_stochastic_dispatch`` scenario);
    we tolerate the model build potentially raising once preprocessing has
    completed, since we only need the ``solve_data/`` CSVs.
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "fixture.sqlite"
    url = json_to_db(json_fixture, db_path)
    migrate_database(url)

    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        try:
            run_chain_from_db(
                url, "2_day_stochastic_dispatch", work_folder=workdir,
            )
        except Exception:
            # Preprocessing has already written ``solve_data/`` by the
            # time the model build runs; downstream failures are fine
            # for our purposes (we only inspect the workdir CSVs).
            pass
    finally:
        os.chdir(cwd)
    return url, workdir


def _build_dt_from_workdir(workdir: Path) -> pl.DataFrame:
    """Build the active ``(d, t)`` frame from ``steps_in_use.csv``."""
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
    return pl.DataFrame(
        {"d": [r[0] for r in dt_rows], "t": [r[1] for r in dt_rows]}
    )


def _invoke_fold(db_url: str, workdir: Path, dt: pl.DataFrame) -> pl.DataFrame:
    """Run :func:`apply_p_inflow_with_scaling` and return the resulting
    ``p_inflow.frame``.
    """
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._inflow_scaling import (
        apply_p_inflow_with_scaling,
    )

    source = SpineDbReader(db_url, scenario="2_day_stochastic_dispatch")

    class _FlexData:
        p_inflow: object | None = None
        nodeBalance: object | None = None

    flex_data = _FlexData()

    nb_path = workdir / "solve_data" / "nodeBalance.csv"
    if nb_path.exists():
        try:
            flex_data.nodeBalance = pl.read_csv(nb_path)
        except Exception:
            flex_data.nodeBalance = None

    ok = apply_p_inflow_with_scaling(
        flex_data, source, workdir, dt, per_solve_aggs=None,
    )
    assert ok, "apply_p_inflow_with_scaling returned False"
    p_inflow = flex_data.p_inflow
    assert p_inflow is not None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    assert fr.height > 0
    return fr


def _mutate_first_timesteps(
    workdir: Path, extra_rows: list[tuple[str, str]]
) -> None:
    """Append ``extra_rows`` to ``solve_data/first_timesteps.csv``.

    Legacy preprocessing always writes exactly one ``time_start`` per
    period, so the multi-``ts`` fold-in branches are structurally
    unreachable from real scenarios.  This helper injects additional
    rows so the parity tests can lock in the multi-``ts`` algorithm.
    """
    fts_path = workdir / "solve_data" / "first_timesteps.csv"
    assert fts_path.exists(), f"missing {fts_path}"
    with fts_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        for d, t in extra_rows:
            writer.writerow([d, t])


def test_pbt_branch1_multi_time_start(tmp_path: Path) -> None:
    """Branch 1 stochastic fold-in correctly sums over multiple
    ``time_starts`` per period when ``ts_for_d[d]`` has cardinality > 1.

    Fixture A (``multi_ts_branch1.json``) adds ``downriver`` to
    ``add_stochastics`` and authors a 3d_map with branches
    ``(realized, upper)`` × time_starts ``(t0001, t0025)`` × all 48
    timesteps, using ``value(b_idx, ts_idx, t_idx) = b_idx*100 +
    ts_idx*10 + t_idx``.

    The test mutates ``first_timesteps.csv`` to add a 2nd ``time_start``
    (``t0025``) under ``period1`` BEFORE invoking the fold-in.  Because
    ``tb_for_d[period1] = [realized]`` only, Branch 1 for
    ``(downriver, period1, t)`` becomes a sum over ``ts ∈ {t0001, t0025}``
    of ``pbt[downriver, realized, ts, t]``.

    Hand-derived oracle (see the augmentation script docstring):

    * ``(downriver, period1, t0001)`` → 0 + 10 = 10
    * ``(downriver, period1, t0005)`` → 4 + 14 = 18
    * ``(downriver, period1, t0048)`` → 47 + 57 = 104
    * ``(downriver, period1_upper, t0001)`` (single ts) →
      pbt[upper, t0001, t0001] = 1*100 + 0*10 + 0 = 100
    * ``(downriver, period1_upper, t0010)`` →
      pbt[upper, t0001, t0010] = 1*100 + 0*10 + 9 = 109
    """
    db_url, workdir = _prepare_workdir(
        FIXTURES_DIR / "multi_ts_branch1.json", tmp_path
    )
    # Inject a 2nd time_start for period1.
    _mutate_first_timesteps(workdir, [("period1", "t0025")])

    dt = _build_dt_from_workdir(workdir)
    fr = _invoke_fold(db_url, workdir, dt)

    got = (fr.filter(pl.col("n") == "downriver")
             .select("d", "t", "value")
             .sort("d", "t"))
    assert got.height > 0, "no downriver rows produced"

    def _value(d: str, t: str) -> float:
        sub = got.filter((pl.col("d") == d) & (pl.col("t") == t))
        assert sub.height == 1, f"missing/duplicate ({d}, {t}): {sub}"
        return float(sub["value"][0])

    # Multi-ts Branch 1 cells (period1 has ts ∈ {t0001, t0025}).
    assert _value("period1", "t0001") == pytest.approx(10.0, abs=1e-9), (
        f"period1/t0001 = {_value('period1', 't0001')} (expected 10)"
    )
    assert _value("period1", "t0005") == pytest.approx(18.0, abs=1e-9)
    assert _value("period1", "t0048") == pytest.approx(104.0, abs=1e-9)

    # Single-ts Branch 1 cells (period1_upper has only ts=t0001 ;
    # tb_for_d[period1_upper] = [upper]).
    assert _value("period1_upper", "t0001") == pytest.approx(100.0, abs=1e-9)
    assert _value("period1_upper", "t0010") == pytest.approx(109.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Phase C — Fixture B: Branch 2 parent-period fold on ``downriver``.
# ---------------------------------------------------------------------------


def test_pbt_branch2_parent_period_fold(tmp_path: Path) -> None:
    """Branch 2 parent-period fold-in fires for child periods whose
    parent (via ``period__branch``) has pbt data, when the node itself
    isn't in ``stoch_node``.

    Fixture B (``branch2_parent_period.json``) authors ``downriver.inflow``
    as a 3d_map with branches ``(realized,)`` × time_starts
    ``(t0001, t0025)`` × all 48 timesteps, using
    ``value(ts_idx, t_idx) = ts_idx*10 + t_idx + 1``.
    ``downriver`` is NOT added to ``add_stochastics`` → Branch 1 does
    not fire → Branch 2 takes over.

    The test mutates ``first_timesteps.csv`` to add ``(period1_upper,
    t0025)`` so the multi-ts Branch 2 fold actually iterates.

    Hand-derived oracle (see the augmentation script docstring):

    * ``(downriver, period1, t0001)`` (Branch 2 from itself):
        pe_for_d[period1] = [period1]; tb_for_d[period1] = [realized];
        ts_for_d[period1] = [t0001]
        → pbt[realized, t0001, t0001] = 1

    * ``(downriver, period1_lower, t0001)``  → pbt[realized, t0001, t0001] = 1
    * ``(downriver, period1_mid, t0005)``    → pbt[realized, t0001, t0005] = 5
    * ``(downriver, period1_upper, t0001)``  (mutated, multi-ts):
        ts_for_d[period1_upper] = [t0001, t0025]
        → pbt[realized, t0001, t0001] + pbt[realized, t0025, t0001]
        = 1 + 11 = 12
    * ``(downriver, period1_upper, t0010)`` (multi-ts):
        → pbt[realized, t0001, t0010] + pbt[realized, t0025, t0010]
        = (0*10 + 9 + 1) + (1*10 + 9 + 1) = 10 + 20 = 30
    """
    db_url, workdir = _prepare_workdir(
        FIXTURES_DIR / "branch2_parent_period.json", tmp_path
    )
    # Inject a 2nd time_start for period1_upper.
    _mutate_first_timesteps(workdir, [("period1_upper", "t0025")])

    dt = _build_dt_from_workdir(workdir)
    fr = _invoke_fold(db_url, workdir, dt)

    got = (fr.filter(pl.col("n") == "downriver")
             .select("d", "t", "value")
             .sort("d", "t"))
    assert got.height > 0, "no downriver rows produced"

    def _value(d: str, t: str) -> float:
        sub = got.filter((pl.col("d") == d) & (pl.col("t") == t))
        assert sub.height == 1, f"missing/duplicate ({d}, {t}): {sub}"
        return float(sub["value"][0])

    # Single-ts Branch 2 cells.
    assert _value("period1", "t0001") == pytest.approx(1.0, abs=1e-9)
    assert _value("period1_lower", "t0001") == pytest.approx(1.0, abs=1e-9)
    assert _value("period1_mid", "t0005") == pytest.approx(5.0, abs=1e-9)

    # Multi-ts Branch 2 cells (period1_upper has ts ∈ {t0001, t0025}).
    assert _value("period1_upper", "t0001") == pytest.approx(12.0, abs=1e-9)
    assert _value("period1_upper", "t0010") == pytest.approx(30.0, abs=1e-9)
