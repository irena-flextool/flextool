"""Δ.21 — CLI ``--engine`` flag retirement tests.

The Δ.14 dispatch landed ``--engine={gmpl,native}`` with ``gmpl`` as
the default.  Δ.21 retires the GMPL path entirely:

1. ``--engine=gmpl`` is rejected with a clear retirement message.
2. ``--engine`` is optional; the only supported value is ``native``
   (the default).
3. ``FLEXPY_USE_NATIVE_ORCHESTRATION`` is vestigial — accepted for
   backward compat but emits a deprecation warning when set truthy.
4. The legacy GMPL-pipeline-only flags (``--ipm``, ``--auto-scale``,
   ``--relax-feasibility``, ``--use-old-raw-csv``, ``--glpsol-timing``,
   ``--report-near-duplicates``, ``--highs-threads >1``) are still
   accepted on the argparse surface but warn-deprecated and are no-op.

The unit-level tests pin :func:`_resolve_engine` directly for fast
verification.  An end-to-end test drives a real ``--engine=native``
invocation against the smallest in-tree fixture (``work_base``).
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

from flextool.cli.cmd_run_flextool import (
    _ENGINE_RETIRED_GMPL_MESSAGE,
    _resolve_engine,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
LH2_GOLDEN_OBJ = FLEXTOOL_ROOT / "tests" / "engine_polars" / "data" / \
    "work_lh2_three_region" / "golden_obj.json"


# ---------------------------------------------------------------------------
# _resolve_engine — Δ.21 truth-table (gmpl rejected; native is the only path).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cli_value,env_value,expected",
    [
        # 1. Explicit ``--engine=native`` is accepted.
        ("native", None,    "native"),
        ("native", "0",     "native"),
        ("native", "yes",   "native"),
        # 2. Default fallback (no flag, no env) is now native.
        (None,     None,    "native"),
        # 3. Env var truthy values are accepted (with deprecation warning).
        (None,     "1",     "native"),
        (None,     "true",  "native"),
        (None,     "TRUE",  "native"),
        (None,     "yes",   "native"),
        (None,     "on",    "native"),
        (None,     " 1 ",   "native"),
        # 4. Env var falsy / unrecognised — silent, default applies.
        (None,     "0",     "native"),
        (None,     "false", "native"),
        (None,     "no",    "native"),
        (None,     "",      "native"),
        (None,     "garbage", "native"),
    ],
)
def test_resolve_engine_native_truth_table(cli_value, env_value, expected) -> None:
    """Every accepted (cli, env) combination resolves to ``'native'``."""
    assert _resolve_engine(cli_value, env_value) == expected


def test_resolve_engine_rejects_gmpl_with_clear_message(capsys) -> None:
    """``--engine=gmpl`` triggers ``SystemExit(2)`` with the retirement
    banner on stderr.

    The banner mentions the Δ.21 retirement and points at the new
    default (``--engine=native``).  Automation / GUI invocations that
    haven't migrated yet get a readable diagnostic instead of a silent
    behaviour change.
    """
    with pytest.raises(SystemExit) as exc_info:
        _resolve_engine("gmpl", None)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert _ENGINE_RETIRED_GMPL_MESSAGE in captured.err
    assert "GMPL path retired" in captured.err


def test_resolve_engine_rejects_unknown_value(capsys) -> None:
    """Defensive: an unrecognised explicit value (bypassing argparse's
    ``choices=``) also bails out cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        _resolve_engine("highs", None)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "not recognised" in captured.err


def test_resolve_engine_env_var_truthy_emits_deprecation_warning(caplog) -> None:
    """The ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var is now vestigial;
    truthy values emit a single deprecation warning."""
    with caplog.at_level("WARNING"):
        _resolve_engine(None, "1")
    assert any("FLEXPY_USE_NATIVE_ORCHESTRATION is deprecated"
               in rec.getMessage() for rec in caplog.records)


def test_resolve_engine_env_var_falsy_silent(caplog) -> None:
    """Env var unset / falsy values stay silent."""
    with caplog.at_level("WARNING"):
        _resolve_engine(None, None)
        _resolve_engine(None, "")
        _resolve_engine(None, "0")
    assert all("FLEXPY_USE_NATIVE_ORCHESTRATION"
               not in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# End-to-end: ``--engine=native`` (and the default no-flag path) drive the
# native cascade against ``work_base``.
# ---------------------------------------------------------------------------


@pytest.fixture
def work_base_db_with_scenario(tmp_path):
    """Provide the path to ``work_base/tests.sqlite`` plus the first
    scenario name discovered in it.  Skips when the fixture isn't on
    disk (e.g. in a sparse checkout).
    """
    sqlite = DATA / "work_base" / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("work_base fixture not present")
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        scenarios = sorted(s.name for s in db.query(db.scenario_sq).all())
    if not scenarios:
        pytest.skip("no scenarios in work_base")
    return sqlite, scenarios[0]


def test_cli_engine_native_subprocess_e2e(work_base_db_with_scenario, tmp_path) -> None:
    """``cmd_run_flextool --engine=native`` runs end-to-end and
    produces ``output_raw/`` per-solve artefacts.

    Δ.14 known gap (still open as of Δ.21): the native cascade does
    not yet emit the wide-format ``solve_data/p_<entity>.csv`` files
    that ``process_outputs.read_parameters`` consumes.  The CLI handles
    this by catching the ``FileNotFoundError`` from ``write_outputs``
    and exiting 0 with a warning so ``output_raw/`` lands cleanly.
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
            "--engine", "native",
            "--work-folder", str(work_folder),
            "--write-methods", "csv",
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
    output_raw = work_folder / "output_raw"
    assert output_raw.exists() and output_raw.is_dir(), (
        f"native cascade did not emit {output_raw}"
    )
    raw_files = list(output_raw.iterdir())
    assert raw_files, f"output_raw/ is empty: {output_raw}"
    v_obj = list(output_raw.glob("v_obj__*.parquet"))
    assert v_obj, f"expected v_obj__*.parquet in {output_raw}"


def test_cli_engine_default_runs_native(work_base_db_with_scenario, tmp_path) -> None:
    """Δ.21 default behaviour: no ``--engine`` flag and no env var
    runs the native cascade (was GMPL pre-Δ.21).

    Discriminator: under native the ``--- Init time`` / ``--- Write
    time`` legacy GMPL phase prints never appear, and the CLI still
    succeeds end-to-end with ``output_raw/`` populated.
    """
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    env = os.environ.copy()
    env.pop("FLEXPY_USE_NATIVE_ORCHESTRATION", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{sqlite}",
            "--scenario-name", scenario,
            # NO --engine flag.
            "--work-folder", str(work_folder),
            "--write-methods", "csv",
            "--output-location", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
        env=env,
    )

    assert result.returncode == 0, (
        f"default-engine CLI failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    # Legacy GMPL phase prints must NOT appear — those came from
    # ``runner.write_input`` + ``runner.run_model`` only.
    assert "--- Init time" not in combined, (
        f"GMPL legacy phase prints leaked into native default run:\n{combined}"
    )
    assert "--- Write time" not in combined, (
        f"GMPL legacy phase prints leaked into native default run:\n{combined}"
    )
    assert (work_folder / "output_raw").exists()


def test_cli_engine_gmpl_rejected_subprocess(work_base_db_with_scenario, tmp_path) -> None:
    """``--engine=gmpl`` exits 2 with the retirement banner.

    The CLI prints the banner to stderr before any setup runs, so the
    rejection is fast and produces no work-folder side-effects.
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
            "--engine", "gmpl",
            "--work-folder", str(work_folder),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=120,
    )
    assert result.returncode == 2, (
        f"expected exit code 2 for --engine=gmpl, got {result.returncode}:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert _ENGINE_RETIRED_GMPL_MESSAGE in combined, (
        f"retirement banner missing from --engine=gmpl rejection:\n{combined}"
    )


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
def test_cli_decomposition_lagrangian_runs_native(
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


def test_cli_env_var_emits_deprecation_warning_subprocess(
    work_base_db_with_scenario, tmp_path,
) -> None:
    """Setting ``FLEXPY_USE_NATIVE_ORCHESTRATION=1`` runs native (no
    behaviour change) but emits a deprecation warning surface."""
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    env = os.environ.copy()
    env["FLEXPY_USE_NATIVE_ORCHESTRATION"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{sqlite}",
            "--scenario-name", scenario,
            "--work-folder", str(work_folder),
            "--write-methods", "csv",
            "--output-location", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, (
        f"native run via env var failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "FLEXPY_USE_NATIVE_ORCHESTRATION is deprecated" in combined, (
        f"deprecation warning missing:\n{combined}"
    )
