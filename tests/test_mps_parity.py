"""Layer 2 MPS-parity tests — see tests/README.md.

Each baseline runs the corresponding flextool scenario in a tmp workdir,
then asserts the produced ``flextool.mps`` hashes identically to the
committed baseline JSON in ``migration/baselines/``.

These run a real solver so they're slow (5-60s per baseline). Mark
with ``@pytest.mark.solver`` so they can be excluded from the smoke
gate.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
REPO_ROOT = TEST_DIR.parent
BASELINES_DIR = REPO_ROOT / "migration" / "baselines"

# db_utils is imported via tests/conftest.py too — keep the path-prep
# defensive so this module can also be invoked standalone (``pytest
# tests/test_mps_parity.py``).
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Parametrization table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParityCase:
    name: str               # pytest id + log label
    baseline: str           # filename under migration/baselines/
    scenario: str           # scenario name in the source DB
    db_source: str          # 'tests_sqlite' | 'h2_trade_parity'


_CASES: tuple[_ParityCase, ...] = (
    _ParityCase(
        name="test_a_lot",
        baseline="test_a_lot_baseline.json",
        scenario="test_a_lot",
        db_source="tests_sqlite",
    ),
    _ParityCase(
        name="fullYear_roll",
        baseline="fullYear_roll_baseline.json",
        scenario="fullYear_roll",
        db_source="tests_sqlite",
    ),
    _ParityCase(
        name="5weeks_invest_fullYear_dispatch_coal_wind",
        baseline="5weeks_invest_fullYear_dispatch_coal_wind_baseline.json",
        scenario="5weeks_invest_fullYear_dispatch_coal_wind",
        db_source="tests_sqlite",
    ),
    _ParityCase(
        name="multi_fullYear_battery_nested_24h_invest_one_solve",
        baseline="multi_fullYear_battery_nested_24h_invest_one_solve_baseline.json",
        scenario="multi_fullYear_battery_nested_24h_invest_one_solve",
        db_source="tests_sqlite",
    ),
    _ParityCase(
        name="h2_trade",
        baseline="h2_trade_baseline.json",
        scenario="scenario_test_6h_no_carrier_storage",
        db_source="h2_trade_parity",
    ),
)


# ---------------------------------------------------------------------------
# Session-scoped fixture: build h2_parity.sqlite once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def h2_parity_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a fresh sqlite from ``h2_trade_parity.json`` once per session.

    The JSON fixture is the parity slice of the workshop H2 trade DB
    (see ``tests/fixtures/build_h2_trade_parity.py``). One scenario:
    ``scenario_test_6h_no_carrier_storage``.
    """
    db_path = tmp_path_factory.mktemp("h2_parity_db") / "h2_parity.sqlite"
    return json_to_db(FIXTURES_DIR / "h2_trade_parity.json", db_path)


def _resolve_db_url(case: _ParityCase, test_db_url: str, h2_parity_db_url: str) -> str:
    if case.db_source == "tests_sqlite":
        return test_db_url
    if case.db_source == "h2_trade_parity":
        return h2_parity_db_url
    raise ValueError(f"Unknown db_source: {case.db_source!r}")


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "MPS parity is obsolete after the Δ.22 GMPL→native cascade "
        "switch: ``flextool.mps`` was a glpsol/HiGHS-LP-file artefact "
        "produced by the legacy MathProg pipeline; the native cascade "
        "builds the LP directly via polar_high and never emits an MPS "
        "file.  The baseline hashes (migration/baselines/*.json) "
        "anchor a deleted code path.  Keeping the test under a skip "
        "marker preserves the documented baselines + parity case "
        "table; structural correctness is now covered by the in-memory "
        "objective + per-solve parquet golden tests in test_scenarios.py."
    )
)
@pytest.mark.solver
@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_mps_parity(
    case: _ParityCase,
    test_db_url: str,
    h2_parity_db_url: str,
    test_bin_dir: Path,
    tmp_path: Path,
) -> None:
    """Run ``case.scenario`` and assert the produced MPS matches the baseline.

    Structural equivalence at 7-sig-fig precision is asserted via
    ``migration.mps_parity.parse_mps`` + ``canonical_hash``. On failure,
    we surface ``diff_canonical`` so the diagnostic is actionable
    (which rows / columns / coefficients diverged).
    """
    # Imported lazily so ``pytest --collect-only`` doesn't pull in
    # migration/* unnecessarily.
    from migration.mps_parity import canonical_hash, diff_canonical, parse_mps

    baseline_path = BASELINES_DIR / case.baseline
    assert baseline_path.exists(), f"baseline missing: {baseline_path}"
    baseline = json.loads(baseline_path.read_text())

    db_url = _resolve_db_url(case, test_db_url, h2_parity_db_url)

    # Each parity case gets its own work dir under tmp_path so the
    # session-scoped h2_parity_db_url survives across cases.
    workdir = tmp_path
    prev_cwd = os.getcwd()
    try:
        # FlexToolRunner writes some files relative to cwd (solve_data/,
        # output_raw/, HiGHS.log) — mirror the LH2 test's chdir pattern
        # so nothing leaks into the repo root.
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url=db_url,
            scenario_name=case.scenario,
            root_dir=workdir,
            bin_dir=test_bin_dir,
            work_folder=workdir,
            highs_threads=1,
        )
        runner.write_input(db_url, case.scenario)
        rc = runner.run_model()
    finally:
        os.chdir(prev_cwd)

    assert rc == 0, f"solve failed for {case.name} (rc={rc})"

    mps_path = workdir / "flextool.mps"
    assert mps_path.exists(), f"flextool.mps not produced for {case.name}"

    canon = parse_mps(mps_path)
    actual_hash = canonical_hash(canon)

    if actual_hash == baseline["hash"]:
        return

    # Hash mismatch: surface a structured diff. We don't have the
    # baseline canonical form (only its summary + hash), so the
    # diagnostic is necessarily one-sided — report shape deltas vs
    # the baseline summary, and offer the regeneration command.
    summary_delta = (
        f"  baseline rows={baseline['n_rows']} cols={baseline['n_cols']} "
        f"coefs={baseline['n_coefficients']}\n"
        f"  current  rows={len(canon.rows)} cols={len(canon.columns)} "
        f"coefs={sum(len(p) for _, p in canon.columns)}"
    )
    # Parse a reference MPS too if there's one alongside the baseline
    # summary (future enhancement — not currently committed). Until
    # then, the canonical-form diff is impossible without an actual
    # reference MPS file, so just include the shape delta.
    _ = diff_canonical  # imported for the docstring contract; not used here
    pytest.fail(
        f"MPS parity FAILED for {case.name}\n"
        f"  baseline: {baseline_path.relative_to(REPO_ROOT)}\n"
        f"  expected hash: {baseline['hash']}\n"
        f"  actual hash:   {actual_hash}\n"
        f"{summary_delta}\n"
        f"  regenerate: python -m migration.mps_parity baseline {mps_path} "
        f"--out {baseline_path}"
    )
