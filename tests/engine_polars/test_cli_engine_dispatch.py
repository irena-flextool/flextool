"""Δ.14 — CLI ``--engine`` flag wiring tests.

Covers the dispatch glue that makes ``cmd_run_flextool`` route to either
the legacy GMPL pipeline or the native polar-high-opt cascade, per the
precedence rules documented on the ``--engine`` help text:

1. Explicit ``--engine=native|gmpl`` flag wins.
2. ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var (``1`` / ``true`` /
   ``yes`` / ``on`` → native).
3. Default ``gmpl`` for backward compat with GUI / Toolbox subprocess
   invocations.

The unit-level tests pin :func:`_resolve_engine` directly for fast
verification of every truth-table cell.  An end-to-end test drives a
real ``--engine=native`` invocation against the smallest in-tree
fixture (``work_base``) and asserts the canonical output tree shape
(``output_raw/`` + ``output_csv/``) appears.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import spinedb_api as api

from flextool.cli.cmd_run_flextool import (
    _resolve_engine,
    _warn_dropped_native_flags,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# _resolve_engine — pure function, exhaustive truth-table coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cli_value,env_value,expected",
    [
        # 1. Explicit CLI flag always wins (env var ignored when present).
        ("native",  None,    "native"),
        ("gmpl",    None,    "gmpl"),
        ("native",  "0",     "native"),  # flag overrides env=0.
        ("gmpl",    "1",     "gmpl"),    # flag overrides env=1.
        ("native",  "yes",   "native"),
        ("gmpl",    "yes",   "gmpl"),
        # 2. Env var (truthy) selects native.
        (None,      "1",     "native"),
        (None,      "true",  "native"),
        (None,      "TRUE",  "native"),  # case-insensitive.
        (None,      "yes",   "native"),
        (None,      "on",    "native"),
        (None,      " 1 ",   "native"),  # leading/trailing whitespace.
        # 3. Env var (falsy / unrecognised) keeps default.
        (None,      "0",     "gmpl"),
        (None,      "false", "gmpl"),
        (None,      "no",    "gmpl"),
        (None,      "",      "gmpl"),    # empty string (env var set blank).
        (None,      "garbage", "gmpl"),
        # 4. Both unset → default 'gmpl'.
        (None,      None,    "gmpl"),
    ],
)
def test_resolve_engine_truth_table(cli_value, env_value, expected) -> None:
    assert _resolve_engine(cli_value, env_value) == expected


def test_resolve_engine_default_override() -> None:
    """The default kwarg lets callers flip the fallback.

    Reserved as an escape hatch for downstream embedders (e.g. a future
    GUI dispatcher) that want native to be the default for their
    invocations without forcing every CLI user to pass the flag.
    Default for the CLI is ``'gmpl'`` per Δ.14 acceptance bar #4.
    """
    assert _resolve_engine(None, None, default="native") == "native"
    assert _resolve_engine(None, None, default="gmpl") == "gmpl"
    # Explicit flag still wins even with a non-default fallback.
    assert _resolve_engine("gmpl", None, default="native") == "gmpl"


# ---------------------------------------------------------------------------
# _warn_dropped_native_flags — list of GMPL-only flags surfaced to user.
# ---------------------------------------------------------------------------


class _ArgsStub:
    """Minimal stand-in for ``argparse.Namespace`` — we only need the
    attributes :func:`_warn_dropped_native_flags` reads."""

    def __init__(self, **kwargs):
        defaults = dict(
            use_old_raw_csv=False,
            ipm=False,
            auto_scale=False,
            relax_feasibility=None,
            glpsol_timing=False,
            highs_threads=1,
            precision_digits=None,
            report_near_duplicates=False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_warn_dropped_native_flags_no_drops_when_clean() -> None:
    args = _ArgsStub()
    assert _warn_dropped_native_flags(args) == []


def test_warn_dropped_native_flags_lists_active_gmpl_flags() -> None:
    args = _ArgsStub(
        use_old_raw_csv=True,
        ipm=True,
        auto_scale=True,
        relax_feasibility="default",
        glpsol_timing=True,
        highs_threads=4,
        precision_digits=10,
        report_near_duplicates=True,
    )
    dropped = _warn_dropped_native_flags(args)
    assert "--use-old-raw-csv" in dropped
    assert "--ipm" in dropped
    assert "--auto-scale" in dropped
    assert "--relax-feasibility" in dropped
    assert "--glpsol-timing" in dropped
    assert "--highs-threads (>1)" in dropped
    assert "--precision-digits" in dropped
    assert "--report-near-duplicates" in dropped


def test_warn_dropped_native_flags_highs_threads_one_is_silent() -> None:
    """``--highs-threads 1`` is the safe default and shouldn't trigger
    the warning even though the flag is set.

    The native path doesn't expose threading config to HiGHS today,
    but the GMPL pipeline's serial default IS the same as the native
    path's effective default — no behaviour difference, no warning.
    """
    args = _ArgsStub(highs_threads=1)
    assert _warn_dropped_native_flags(args) == []


# ---------------------------------------------------------------------------
# End-to-end: --engine=native subprocess invocation against work_base.
# ---------------------------------------------------------------------------
#
# Mirrors the existing ``test_xlsx_workflow`` subprocess pattern but
# stripped down to the minimum required for engine-dispatch coverage:
# we don't validate the LP solution itself (the chain tests already do
# that), only that the CLI plumbing works end-to-end with the new flag.


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

    Δ.14 known gap: the native cascade does NOT yet emit the
    wide-format ``solve_data/p_<entity>.csv`` files that
    ``process_outputs.read_parameters`` consumes — closing that gap is
    Δ.15 scope.  The CLI handles this by catching the
    ``FileNotFoundError`` from ``write_outputs`` under
    ``--engine=native`` and exiting 0 with a warning, so ``output_raw/``
    (the only artefact the cascade currently produces) lands cleanly.
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
    # output_raw/ is emitted per-solve by the cascade (write_outputs_for_solve).
    output_raw = work_folder / "output_raw"
    assert output_raw.exists() and output_raw.is_dir(), (
        f"native cascade did not emit {output_raw}"
    )
    raw_files = list(output_raw.iterdir())
    assert raw_files, f"output_raw/ is empty: {output_raw}"
    # Spot-check: the per-solve objective parquet (a signature of the
    # native cascade's ``write_outputs_for_solve`` path).
    v_obj = list(output_raw.glob("v_obj__*.parquet"))
    assert v_obj, f"expected v_obj__*.parquet in {output_raw}"


def test_cli_engine_env_var_subprocess_e2e(work_base_db_with_scenario, tmp_path) -> None:
    """``FLEXPY_USE_NATIVE_ORCHESTRATION=1`` selects native without an
    explicit ``--engine`` flag.

    Discriminator: when the native path runs, it emits the
    GMPL-only-flag-warning logged by :func:`_warn_dropped_native_flags`
    iff any of those flags are set.  We pass ``--ipm`` to deliberately
    activate that warning so we can assert on it, confirming the
    dispatch went native.  GMPL ignores ``--ipm`` silently.
    """
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
            # Pin the native-path discriminator: under native this
            # triggers the "ignoring GMPL-only flag(s)" warning; under
            # GMPL ``--ipm`` is silently consumed by the solver.
            "--ipm",
        ],
        capture_output=True,
        text=True,
        cwd=str(FLEXTOOL_ROOT),
        timeout=300,
        env=env,
    )

    assert result.returncode == 0, (
        f"CLI failed under env-var dispatch (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # output_raw/ + output_csv/ exist on both engines; presence proves
    # dispatch reached the post-solve write_outputs hop.
    assert (work_folder / "output_raw").exists()
    assert (tmp_path / "output_csv").exists()
    # Native-path discriminator: the dropped-flag warning fires from
    # inside ``_run_native_solve``.  GMPL never emits this string.
    combined = result.stdout + result.stderr
    assert "engine=native: ignoring GMPL-only flag" in combined, (
        f"expected env-var dispatch to route through the native path "
        f"(no 'engine=native' warning in output):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_cli_engine_default_is_gmpl_subprocess_e2e(work_base_db_with_scenario, tmp_path) -> None:
    """Backward-compat regression (acceptance bar #4): no ``--engine``
    flag and no env var keeps the legacy GMPL pipeline.

    Discriminators (any one is sufficient — we assert the conjunction
    for robustness):

    1. The ``engine=native: ignoring GMPL-only flag`` warning never
       appears (the native dispatcher is what emits it).
    2. The legacy ``--- Init time``, ``--- Write time`` phase prints
       appear (the native path doesn't print these — it doesn't have
       those phase boundaries).
    """
    sqlite, scenario = work_base_db_with_scenario
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    # Make absolutely sure the env var isn't leaking in from the dev
    # shell — the test harness forwards os.environ by default.
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
        f"GMPL default failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "engine=native: ignoring GMPL-only flag" not in combined, (
        f"default dispatch should NOT route to native:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Legacy phase prints are unique to the GMPL path.
    assert "--- Init time" in combined or "--- Write time" in combined, (
        f"expected GMPL phase prints in output but found none:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
