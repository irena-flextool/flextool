"""CLI subprocess smoke tests covering the default run path and the
Lagrangian decomposition entry point.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import spinedb_api as api


FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
LH2_GOLDEN_OBJ = FLEXTOOL_ROOT / "tests" / "engine_polars" / "data" / \
    "work_lh2_three_region" / "golden_obj.json"


# ---------------------------------------------------------------------------
# End-to-end: the default CLI invocation drives the native cascade against
# ``work_base``.
# ---------------------------------------------------------------------------


@pytest.fixture
def work_base_db_with_scenario(tmp_path, scenario_workdir):
    """Provide the path to ``work_base/tests.sqlite`` plus the first
    scenario name discovered in it.  Skips when the fixture isn't on
    disk (e.g. in a sparse checkout).
    """
    sqlite = scenario_workdir("base") / "tests.sqlite"
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        scenarios = sorted(s.name for s in db.query(db.scenario_sq).all())
    return sqlite, scenarios[0]


def test_cli_subprocess_e2e(work_base_db_with_scenario, tmp_path) -> None:
    """``cmd_run_flextool`` runs end-to-end and produces the canonical
    user-facing ``output_parquet/<scenario>/`` tree.

    ``output_raw/`` is engine-internal scaffolding that
    ``write_outputs`` consumes; without ``--csv-dump`` the CLI deletes
    it on a successful run, so the user-visible contract is the
    canonical output tree under ``--output-location``.
    """
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{sqlite}",
            "--scenario-name", scenario,
            "--work-folder", str(work_folder),
            "--write-methods", "csv", "parquet",
            "--output-location", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
    )

    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output_parquet = tmp_path / "output_parquet" / scenario
    assert output_parquet.exists() and output_parquet.is_dir(), (
        f"canonical output_parquet/<scenario>/ missing: {output_parquet}"
    )
    parquets = list(output_parquet.glob("*.parquet"))
    assert parquets, (
        f"expected at least one .parquet under {output_parquet}"
    )
    # output_raw/ is engine-internal; without --csv-dump it must be
    # cleaned up on a successful run.
    output_raw = work_folder / "output_raw"
    assert not output_raw.exists(), (
        f"output_raw/ should be deleted without --csv-dump but still "
        f"exists at {output_raw}"
    )


def test_cli_csv_dump_keeps_output_raw(
    work_base_db_with_scenario, tmp_path,
) -> None:
    """With ``--csv-dump`` the engine-internal ``output_raw/`` survives
    the run alongside the canonical output tree, for debug inspection.
    """
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{sqlite}",
            "--scenario-name", scenario,
            "--work-folder", str(work_folder),
            "--write-methods", "csv", "parquet",
            "--output-location", str(tmp_path),
            "--csv-dump",
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
    )

    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output_parquet = tmp_path / "output_parquet" / scenario
    assert output_parquet.exists() and any(output_parquet.iterdir()), (
        f"canonical output_parquet/<scenario>/ missing under "
        f"--csv-dump: {output_parquet}"
    )
    output_raw = work_folder / "output_raw"
    assert output_raw.exists() and any(output_raw.iterdir()), (
        f"output_raw/ should survive --csv-dump but is missing/empty: "
        f"{output_raw}"
    )


# ---------------------------------------------------------------------------
# Group-flow output wiring — the production CLI must populate the group-flow
# column families in ``group_flows__dt.csv``.  Regression lock for the
# provider-wiring bug (cmd_run_flextool called write_outputs without
# flex_data_provider → single-solve backfill got no provider → group sets
# stayed empty → from_unitGroup / from_connectionGroup columns vanished).
# ---------------------------------------------------------------------------


TESTS_JSON_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json"
GROUP_SCENARIO = "aggregate_outputs_network_coal_wind_chp"


@pytest.fixture
def group_scenario_db(tmp_path):
    """Materialise ``tests.json`` (which carries the
    ``aggregate_outputs_network_coal_wind_chp`` group scenario) into a
    fresh SQLite and return ``(db_url, scenario_name)``.

    Per repo rule we build the DB from JSON/schema — never read a
    checked-in ``.sqlite``.
    """
    if not TESTS_JSON_FIXTURE.exists():
        pytest.skip(f"tests.json fixture not present: {TESTS_JSON_FIXTURE}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "group_scenario.sqlite"
    db_url = json_to_db(TESTS_JSON_FIXTURE, db_path)
    migrate_database(db_url)
    return db_url, GROUP_SCENARIO


def _group_flows_family_levels(csv_path: Path) -> set[str]:
    """Return the set of column-family names from ``group_flows__dt.csv``.

    The CSV has a 3-row MultiIndex header (group / family / detail).  The
    family names live on the *second* header row (``level_1``); read it
    raw so we don't depend on pandas' MultiIndex parsing of duplicate
    group names.
    """
    with open(csv_path, "r", encoding="utf-8") as fh:
        fh.readline()  # group level (row 0)
        family_row = fh.readline().rstrip("\n")  # family level (row 1)
    # Drop the three leading index columns (solve, period, time) which the
    # family row backfills with ``Unnamed: *_level_1`` placeholders.
    cells = family_row.split(",")[3:]
    return {c for c in cells if c and not c.startswith("Unnamed:")}


def test_cli_group_flows_columns_present(group_scenario_db, tmp_path) -> None:
    """``cmd_run_flextool`` on the group scenario must emit
    ``group_flows__dt.csv`` *with* the group-flow column families.

    The production CLI path (``write_outputs`` driven from
    ``cmd_run_flextool``) must forward the live ``flex_data_provider`` so
    the single-solve ``_backfill_group_indicator_sets`` can populate the
    ``nodeGroupDispatch*`` / ``*Indicators`` sets.  Without that, the
    ``from_unitGroup`` / ``from_connectionGroup`` / ``from_unit`` /
    ``to_connectionGroup`` families silently disappear from
    ``group_flows__dt.csv`` (only slack / inflow / internal_losses
    survive).  This asserts they are present — the regression lock.
    """
    db_url, scenario = group_scenario_db
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            db_url,
            "--scenario-name", scenario,
            "--work-folder", str(work_folder),
            "--write-methods", "csv",
            "--output-location", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=600,
    )

    assert result.returncode == 0, (
        f"CLI failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    csv_path = tmp_path / "output_csv" / scenario / "group_flows__dt.csv"
    assert csv_path.exists(), (
        f"group_flows__dt.csv missing under {csv_path.parent}:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    families = _group_flows_family_levels(csv_path)
    missing = {"from_unitGroup", "from_connectionGroup"} - families
    assert not missing, (
        f"group_flows__dt.csv is missing the group-flow column "
        f"families {sorted(missing)} (the provider-wiring bug).  "
        f"families present: {sorted(families)}\n"
        f"CLI stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_backfill_group_indicator_sets_requires_provider(tmp_path) -> None:
    """STRICT contract lock: on the in-memory output path a Provider is
    REQUIRED.  ``_backfill_group_indicator_sets`` must RAISE on
    ``provider=None`` (a wiring bug), while an explicit *empty* Provider
    (a group-less in-memory run) must be tolerated → no raise, empty sets.
    """
    from types import SimpleNamespace

    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import (
        _backfill_group_indicator_sets,
    )

    # Missing Provider → wiring bug → raise.
    s = SimpleNamespace()
    with pytest.raises(ValueError, match="Provider is REQUIRED"):
        _backfill_group_indicator_sets(s, str(tmp_path), provider=None)

    # Empty Provider (no group keys) → tolerated, sets stay untouched.
    s2 = SimpleNamespace()
    _backfill_group_indicator_sets(s2, str(tmp_path), provider=FlexDataProvider())
    assert not hasattr(s2, "nodeGroupDispatch")


# ---------------------------------------------------------------------------
# --decomposition lagrangian — CLI dispatch onto the native coordinator
# ---------------------------------------------------------------------------


@pytest.fixture
def lh2_three_region_db(tmp_path):
    """Materialise the committed LH2 three-region JSON fixture into a
    fresh SQLite and return ``(db_url, scenario_name)``.

    The native ``solve_lagrangian`` coordinator and its DB schema piece
    (``group.decomposition_method``) are exercised here through the CLI;
    the JSON fixture pins three groups (``region_A/B/C``) at
    ``lagrangian_region`` plus the cross-region pipes ``pipe_AB`` and
    ``pipe_BC``.
    """
    if not LH2_FIXTURE_JSON.exists():
        pytest.skip(f"LH2 JSON fixture not present: {LH2_FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    db_path = tmp_path / "lh2_three_region.sqlite"
    db_url = json_to_db(LH2_FIXTURE_JSON, db_path)
    return db_url, "lh2_three_region"


@pytest.mark.solver
def test_cli_decomposition_lagrangian(
    lh2_three_region_db, tmp_path,
) -> None:
    """``cmd_run_flextool --decomposition lagrangian`` drives the native
    coordinator (``engine_polars._lagrangian.solve_lagrangian``) end-to-end
    on the LH2 three-region fixture.

    Acceptance bar (cf. ``specs/lagrangian_port_handoff.md``):

    1. Exit code 0 (converged) or 1 (max-iters hit) — the dual
       subgradient on LH2 oscillates around a 0.1 % gap due to
       bang-bang LP response on the pipeline flows, so non-convergence
       to a tight ``tol`` is normal; the CLI maps that to exit 1 by
       design.  We assert the dispatch ran to completion either way.
    2. Stdout contains ``total_objective=`` with a value within 2 % of
       the LH2 monolithic optimum pinned in ``golden_obj.json``.
    3. λ output is present for both cross-region pipes (``pipe_AB`` and
       ``pipe_BC``).

    Coordinator tuning matches the algorithm-level parity test in
    ``tests/engine_polars/test_lagrangian.py`` (``alpha=10``,
    ``max_iters=100``, ``tol=0.5``) — the CLI defaults were chosen for
    smaller, less-coupled scenarios and don't converge on LH2 within
    the budget.
    """
    db_url, scenario = lh2_three_region_db
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            db_url,
            "--scenario-name", scenario,
            "--decomposition", "lagrangian",
            "--lagrangian-alpha", "10.0",
            "--lagrangian-max-iter", "100",
            "--lagrangian-tolerance", "0.5",
            "--work-folder", str(work_folder),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=600,
    )

    assert result.returncode in (0, 1), (
        f"CLI failed (rc={result.returncode}, expected 0 or 1):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr

    # Total objective line is present and parseable.
    match = re.search(r"total_objective=([0-9.eE+\-]+)", combined)
    assert match, (
        f"total_objective=... line missing from CLI stdout:\n{combined}"
    )
    reported = float(match.group(1))

    # λ output for both cross-region pipes (4 couplings total: each
    # pipe carries two directions, but it's enough to find each name).
    assert "pipe_AB" in combined, f"λ for pipe_AB missing:\n{combined}"
    assert "pipe_BC" in combined, f"λ for pipe_BC missing:\n{combined}"

    # Within 2 % of monolithic optimum (cf. handoff acceptance bar).
    golden = json.loads(LH2_GOLDEN_OBJ.read_text())["obj"]
    rel_gap = abs(reported - golden) / abs(golden)
    assert rel_gap <= 0.02, (
        f"reported total_objective={reported:.6e} vs monolithic "
        f"golden={golden:.6e} → {rel_gap*100:.3f}% gap exceeds 2%.\n"
        f"CLI output:\n{combined}"
    )


# ---------------------------------------------------------------------------
# --highs-threads — CLI flag plumbs through to HiGHS via env var
# ---------------------------------------------------------------------------


def _run_cli_with_threads(
    work_base_db_with_scenario, tmp_path, n_threads: int,
) -> subprocess.CompletedProcess:
    """Drive the default CLI path with ``--highs-threads N``.  Returns
    the completed process so individual tests can assert on rc/stdout.
    """
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    # Scrub any inherited env var so the CLI's own conversion is the
    # only thing setting ``FLEXTOOL_HIGHS_THREADS`` for the subprocess.
    env = dict(os.environ)
    env.pop("FLEXTOOL_HIGHS_THREADS", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{sqlite}",
            "--scenario-name", scenario,
            "--work-folder", str(work_folder),
            "--write-methods", "csv", "parquet",
            "--output-location", str(tmp_path),
            "--highs-threads", str(n_threads),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
        env=env,
    )


def test_cli_highs_threads_default(work_base_db_with_scenario, tmp_path) -> None:
    """``--highs-threads 1`` is the default-equivalent path: keeps the
    DETERMINISM_OPTIONS pin in place, leaves ``parallel="off"``, and the
    solve completes without errors.
    """
    _, scenario = work_base_db_with_scenario
    result = _run_cli_with_threads(work_base_db_with_scenario, tmp_path, 1)
    assert result.returncode == 0, (
        f"CLI failed with --highs-threads 1 (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output_parquet = tmp_path / "output_parquet" / scenario
    assert output_parquet.exists() and any(output_parquet.iterdir()), (
        f"output_parquet/<scenario>/ missing or empty under "
        f"--highs-threads 1: {output_parquet}"
    )


def test_cli_highs_threads_two(work_base_db_with_scenario, tmp_path) -> None:
    """``--highs-threads 2`` flips HiGHS into parallel mode and the
    solve still completes.  We deliberately do NOT assert numerical
    parity: multi-threaded HiGHS is non-deterministic by design, and
    the user-facing flag's contract is exactly that trade-off.
    """
    _, scenario = work_base_db_with_scenario
    result = _run_cli_with_threads(work_base_db_with_scenario, tmp_path, 2)
    assert result.returncode == 0, (
        f"CLI failed with --highs-threads 2 (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output_parquet = tmp_path / "output_parquet" / scenario
    assert output_parquet.exists() and any(output_parquet.iterdir()), (
        f"output_parquet/<scenario>/ missing or empty under "
        f"--highs-threads 2: {output_parquet}"
    )
