"""Benders silences the per-sub-solve HiGHS native log by default.

The Benders driver solves a master plus N region subproblems (cold build +
a warm re-solve per iteration, regions in parallel); leaving every one of
those HiGHS solves verbose floods the console with many interleaved native
logs.  By DEFAULT the engine now mutes the HiGHS native log on the master
WarmProblem and on every region WarmProblem (via
``WarmProblem.set_output_flag(False)``, gated by
``flextool.engine_polars._benders._benders_quiet``) and relies on the
orchestrator's concise per-iteration LB/UB/gap log for visible progress.
``FLEXTOOL_BENDERS_VERBOSE`` restores the full native HiGHS output.

This module drives the production orchestrator path
(``run_chain_from_db`` on ``lh2_three_region_trade_invest``, the same small
prototype the invest-handoff chain test uses) and asserts:

  (a) by DEFAULT the HiGHS native solve log is ABSENT while the clean
      per-iteration Benders log is PRESENT;
  (b) with ``FLEXTOOL_BENDERS_VERBOSE=1`` the HiGHS native log REAPPEARS;
  (c) the converged objective is IDENTICAL with and without silencing —
      muting the log changes no solve numerics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TI_FIXTURE_JSON = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region_trade_invest.json"
)
TI_SCENARIO = "lh2_three_region_trade_invest"

# Native HiGHS solve-log fingerprints — these only ever appear when the
# per-solve ``output_flag`` is left enabled.
HIGHS_NEEDLES = (
    "Presolving model",
    "Solving the presolved LP",
    "Simplex   iterations",
    "HiGHS run time",
    "Coefficient ranges",
)
# Clean orchestrator per-iteration progress lines (emitted via ``print``,
# independent of HiGHS' output_flag).
CLEAN_NEEDLES = ("[benders", "LB=", "UB=", "lower bound (valid)")


def _build_trade_invest_db(tmp_path: Path) -> str:
    """Materialise the committed trade-invest JSON fixture into a fresh
    SQLite and migrate to the current schema (build-from-JSON per CLAUDE.md
    invariant 3)."""
    if not TI_FIXTURE_JSON.exists():
        pytest.skip(f"trade-invest JSON fixture not present: {TI_FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "lh2_trade_invest.sqlite"
    url = json_to_db(TI_FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


def _run_chain(url: str, work_folder: Path):
    """Drive the Benders solve through the production orchestrator and
    return ``(stdout, converged_objective)``."""
    import contextlib
    import io

    from flextool.engine_polars import run_chain_from_db

    work_folder.mkdir(exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        steps = run_chain_from_db(
            url, TI_SCENARIO, work_folder=work_folder, keep_solutions=True,
        )
    key = next((k for k in steps if "lh2_trade_invest" in k), None)
    assert key is not None, f"no lh2_trade_invest step in {list(steps)}"
    return buf.getvalue(), steps[key].obj


def _highs_log_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if any(n in ln for n in HIGHS_NEEDLES)]


@pytest.mark.solver
def test_benders_silences_highs_log_by_default(tmp_path, monkeypatch) -> None:
    """(a) DEFAULT: no native HiGHS solve log, clean Benders log present."""
    monkeypatch.delenv("FLEXTOOL_BENDERS_VERBOSE", raising=False)
    url = _build_trade_invest_db(tmp_path)

    out, obj = _run_chain(url, tmp_path / "work_quiet")

    leaked = _highs_log_lines(out)
    assert not leaked, (
        "HiGHS native solve log leaked despite default silencing:\n"
        + "\n".join(leaked[:10])
    )
    for needle in CLEAN_NEEDLES:
        assert needle in out, (
            f"clean Benders progress marker {needle!r} missing from stdout:\n{out}"
        )
    assert obj is not None


@pytest.mark.solver
def test_benders_verbose_env_restores_highs_log(tmp_path, monkeypatch) -> None:
    """(b) With FLEXTOOL_BENDERS_VERBOSE=1 the native HiGHS log reappears,
    and (c) the converged objective is IDENTICAL to the silenced run."""
    # Silenced (default) baseline objective.
    monkeypatch.delenv("FLEXTOOL_BENDERS_VERBOSE", raising=False)
    url_quiet = _build_trade_invest_db(tmp_path / "quiet")
    out_quiet, obj_quiet = _run_chain(url_quiet, tmp_path / "work_quiet")
    assert not _highs_log_lines(out_quiet), "quiet baseline unexpectedly noisy"

    # Verbose run — same fixture, env set.
    monkeypatch.setenv("FLEXTOOL_BENDERS_VERBOSE", "1")
    url_verbose = _build_trade_invest_db(tmp_path / "verbose")
    out_verbose, obj_verbose = _run_chain(url_verbose, tmp_path / "work_verbose")

    # (b) native HiGHS log is back.
    assert _highs_log_lines(out_verbose), (
        "FLEXTOOL_BENDERS_VERBOSE=1 did not restore the native HiGHS log:\n"
        + out_verbose
    )
    # The clean Benders log is present in both modes.
    for needle in CLEAN_NEEDLES:
        assert needle in out_verbose, f"clean marker {needle!r} missing under verbose"

    # (c) silencing changed no numerics — converged objectives identical.
    assert obj_verbose == obj_quiet, (
        f"objective differs quiet={obj_quiet!r} vs verbose={obj_verbose!r}"
    )
