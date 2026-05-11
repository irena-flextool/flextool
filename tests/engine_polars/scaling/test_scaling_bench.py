"""Scaling test bench — sandbox for iterating on LP scaling fixes.

Two parametrised cases:

* ``baseline`` — uses the long-lived ``multi_year_one_solve_battery``
  scenario in ``tests/fixtures/tests.json`` (a model that already has
  reasonable HiGHS-reported coefficient ranges).  Acts as a regression
  guardrail: if a future scaling change accidentally ruins the
  well-scaled case, this fails.
* ``stressed`` — uses ``tests/engine_polars/data/scaling_stress/input.json``
  (built from ``tests.json`` via ``_build_input.py``), a small fixture
  authored to exercise poor coefficient compounding using *only*
  realistic FlexTool modeling levers: 0.01 vs 50 €/MWh VOM spread,
  realistic invest_cost spread (250 vs 8000 €/kW), 1e+5 slack penalties,
  mixed unit sizes (5000 / 50 MW), and — most importantly — a per-period
  ``years_represented`` spread of 1/10/1/10 yr inside one solve.

These are *not* parity tests — there are no golden objectives.  The
job is just to print scaling diagnostics under ``pytest -s`` so we can
iterate on the analyser / solver options::

    pytest -sv tests/engine_polars/scaling/test_scaling_bench.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure tests/ is on sys.path so we can import db_utils.
TESTS_DIR = Path(__file__).resolve().parents[2]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_single_solve_from_db  # noqa: E402

REPO_ROOT = TESTS_DIR.parent
FIXTURES = TESTS_DIR / "fixtures"
DATA = TESTS_DIR / "engine_polars" / "data"

STRESS_JSON = DATA / "scaling_stress" / "input.json"
BASELINE_JSON = FIXTURES / "tests.json"


CASES = [
    pytest.param(
        "baseline",
        BASELINE_JSON,
        "multi_year_one_solve_battery",
        id="baseline",
    ),
    pytest.param(
        "stressed",
        STRESS_JSON,
        "scaling_stress",
        id="stressed",
    ),
]


def _print_artifacts(label: str, work: Path, step) -> None:
    """Dump scaling artifacts to stdout under ``pytest -s``."""
    print(f"\n=== [{label}] obj = {step.obj!r}  optimal = {step.solution.optimal}")
    sd = work / "solve_data"
    analysis_path = sd / "scaling_analysis.json"
    if analysis_path.exists():
        with analysis_path.open() as fh:
            analysis = json.load(fh)
        print(f"\n--- [{label}] scaling_analysis.json ---")
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        analysis = None
        print(f"[{label}] WARN: {analysis_path} missing")

    report_path = sd / "scaling_report.txt"
    if report_path.exists():
        print(f"\n--- [{label}] scaling_report.txt ---")
        print(report_path.read_text())
    else:
        print(f"[{label}] WARN: {report_path} missing")

    return analysis


@pytest.mark.parametrize("label,json_path,scenario", CASES)
def test_scaling_bench(
    label: str,
    json_path: Path,
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build a fresh sqlite from JSON, solve, and dump scaling diagnostics."""
    if not json_path.exists():
        pytest.skip(f"fixture JSON missing: {json_path}")

    # Isolate CWD so HiGHS.log + temp artefacts go to tmp_path.
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / f"{label}.sqlite"
    db_url = json_to_db(json_path, db_path)

    work = tmp_path / "work"
    step = run_single_solve_from_db(
        db_url,
        scenario_name=scenario,
        work_folder=work,
    )

    analysis = _print_artifacts(label, work, step)

    assert step.solution is not None, f"[{label}] no solution returned"
    assert step.solution.optimal, (
        f"[{label}] HiGHS non-optimal "
        f"(status={getattr(step.solution, 'status', None)})"
    )

    if label == "stressed":
        # Guardrail: if a future model edit accidentally tames the
        # stress, this assertion should catch it.  Thresholds chosen
        # generously below the values the fixture currently produces.
        assert analysis is not None, (
            "stressed case: scaling_analysis.json was not written"
        )
        cost_spread = analysis.get("cost_spread_log10", 0.0)
        unitsize_spread = analysis.get("unitsize_spread_log10", 0.0)
        # With the realistic-lever fixture we get roughly:
        #   cost_spread ~ 7.5-8 decades (vom 0.01 vs invest 1e+6+)
        #   unitsize_spread ~ 3.7 decades (5000 vs 1)
        # Floors are well below the realistic-fixture levels but
        # high enough to catch a future edit that accidentally erases
        # the stress (e.g. dropping years_represented spread or vom
        # spread).
        assert cost_spread >= 5.0 or unitsize_spread >= 2.0, (
            f"stressed case lost its stress: "
            f"cost_spread={cost_spread:.2f}, "
            f"unitsize_spread={unitsize_spread:.2f}"
        )
