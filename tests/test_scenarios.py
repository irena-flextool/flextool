"""Parametrized end-to-end scenario tests for FlexTool.

Each test:
  1. Uses the session-scoped SQLite DB (imported from test/fixtures/tests.json)
  2. Runs FlexToolRunner for the given scenario in an isolated tmp workdir
  3. Calls write_outputs(write_methods=['csv'])
  4. Compares selected CSVs against golden files in test/expected/<scenario>/

Scenario definitions live in test/scenarios.yaml — edit that file to add
or modify scenarios. See test/README.md for the full workflow.
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

TEST_DIR = Path(__file__).parent
EXPECTED_DIR = TEST_DIR / "expected"
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import round_for_comparison  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.engine_polars import run_chain_from_db
from flextool.process_outputs.write_outputs import write_outputs


def _load_scenarios() -> list:
    """Load scenarios from YAML, applying ``smoke`` marker where requested.

    Each entry may set ``smoke: true`` to be included in the per-commit
    smoke gate (``pytest -m smoke``). Returns a list of ``pytest.param``
    objects so markers can be attached per-parametrize-case.

    Optional ``expected_objective`` and ``expected_objective_tolerance``
    fields enable hand-derived objective-value assertions: a regression
    that produces a consistently-wrong objective (but still writes
    self-consistent CSVs that the golden-comparison would lock in) is
    caught here.

    Optional ``time_budget_seconds`` field enables a per-scenario timing
    assertion: the test body (write_input → run_model → write_outputs)
    must finish within the given budget. Catches large performance
    regressions in the writer/runner/post-process layers.
    """
    with open(TEST_DIR / "scenarios.yaml") as f:
        entries = yaml.safe_load(f)
    params = []
    for e in entries:
        marks = []
        if e.get("smoke"):
            marks.append(pytest.mark.smoke)
        params.append(
            pytest.param(
                e["scenario"],
                e["csvs"],
                e.get("expected_objective"),
                e.get("expected_objective_tolerance", 1e-3),
                e.get("time_budget_seconds"),
                e.get("db_fixture", "main"),
                marks=marks,
                id=e["scenario"],
            )
        )
    return params


def _parse_summary_solve_objective(summary_path: Path) -> float:
    """Extract the full-horizon total cost from ``summary_solve.csv``.

    The file's format (see tests/expected/base/summary_solve.csv) is a
    free-form CSV with this row near the top:

        "Total cost (calculated) full horizon (M CUR)",4780.16775,...

    This row is the most stable single-number summary across single-
    and multi-solve scenarios — for rolling/multi-solve runs the
    ``Solve,Objective`` rows are per-roll while this row aggregates the
    full horizon.
    """
    label = "Total cost (calculated) full horizon"
    for raw in summary_path.read_text().splitlines():
        if label in raw:
            # Format: "<label> (M CUR)",<value>,...
            parts = raw.split(",")
            for part in parts[1:]:
                cleaned = part.strip().strip('"')
                if not cleaned:
                    continue
                try:
                    return float(cleaned)
                except ValueError:
                    continue
            raise ValueError(
                f"Could not parse objective value from row: {raw!r}"
            )
    raise ValueError(
        f"No '{label}' row found in {summary_path}"
    )


# CSVs with non-standard formatting that pd.read_csv cannot parse.
# These are compared as plain text instead.
_FREEFORM_CSVS = {"summary_solve.csv"}


def _is_freeform_csv(csv_name: str) -> bool:
    return csv_name in _FREEFORM_CSVS


def _header_row_count(path: Path) -> int:
    """Detect 1-, 2- or 3-row CSV header.

    FlexTool emits CSVs with 1-row headers (most outputs), 2-row headers
    (per-entity outputs like ``unit__outputNode__dt.csv``: outer = process,
    inner = source/sink node), or 3-row headers (group_flows__dt.csv:
    group / parameter / item).  Continuation header rows begin with empty
    fields aligned with the leading index columns (``,,,...``) or with
    pandas' ``Unnamed: N_level_N`` placeholders when round-tripped through
    ``pd.read_csv`` + ``to_csv``.  A genuine data row never begins with
    either signal.
    """
    with open(path) as f:
        first = f.readline()
        second = f.readline()
        third = f.readline()
    if not second or second.count(",") != first.count(","):
        return 1
    looks_header_two = second.startswith(",") or second.startswith("Unnamed:")
    if not looks_header_two:
        return 1
    if third and third.count(",") == first.count(","):
        if third.startswith(",") or third.startswith("Unnamed:"):
            return 3
    return 2


def _has_two_row_header(path: Path) -> bool:
    """Backwards-compatible helper — True for both 2- and 3-row headers.

    Older callers only distinguished single vs multi-row headers; keep
    that surface area while the new :func:`_header_row_count` drives the
    actual header parsing in :func:`_read_csv`.
    """
    return _header_row_count(path) >= 2


_DEDUP_SUFFIX = re.compile(r"\.\d+$")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a golden/actual CSV, auto-detecting multi-row headers.

    For multi-row-header CSVs we strip pandas-style ``.N`` dedup suffixes
    from every header level.  Some pre-existing goldens were last
    regenerated under a code path that dedup'd duplicate outer labels
    (``west`` → ``west``, ``west.1``…) before round-tripping through
    ``to_csv``; the current writer leaves outer duplicates intact and
    pandas instead dedups the inner level on read.  Stripping both sides
    keeps the actual/golden comparison meaningful without touching the
    goldens.
    """
    nh = _header_row_count(path)
    if nh >= 2:
        df = pd.read_csv(path, header=list(range(nh)))
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(_DEDUP_SUFFIX.sub("", lvl) for lvl in t) for t in df.columns]
        )
        return df
    return pd.read_csv(path)


def _strip_timestamps(text: str) -> str:
    """Remove run-specific timestamps so freeform CSVs can be compared across runs."""
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}", "<TIMESTAMP>", text)


SCENARIOS = _load_scenarios()


@pytest.mark.parametrize(
    "scenario,csvs,expected_objective,expected_objective_tolerance,time_budget_seconds,db_fixture",
    SCENARIOS,
)
def test_scenario(
    scenario: str,
    csvs: list[str],
    expected_objective: float | None,
    expected_objective_tolerance: float,
    time_budget_seconds: float | None,
    db_fixture: str,
    scenario_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
    request: pytest.FixtureRequest,
) -> None:
    regenerate = request.config.getoption("--regenerate")

    # Run the model via the engine_polars cascade (the legacy GMPL
    # ``FlexToolRunner.run_model`` path was removed in Δ.22 — see
    # ``flextool/flextoolrunner/solver_runner.py``).  ``run_chain_from_db``
    # writes ``input/`` + ``solve_data/`` and drives the per-solve
    # cascade end-to-end; the post-process CSVs are then written by
    # ``write_outputs`` below (which needs the last sub-solve's
    # ``flex_data`` + ``solution`` after Δ.31).
    t_start = time.perf_counter()
    steps = run_chain_from_db(
        input_db_url=scenario_db_url,
        scenario_name=scenario,
        work_folder=workdir,
        bin_dir=test_bin_dir,
        # Phase C.5 — the ``solve_steps`` block below consumes every
        # sub-solve's ``flex_data`` + ``solution`` to union par/s over
        # the full dt axis.  Opt out of the slim-step cascade.
        keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for scenario '{scenario}'"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal, (
        f"Last sub-solve for scenario '{scenario}' did not produce an "
        f"optimal solution"
    )

    # Write CSV outputs.  Post-Δ.31 the in-memory readers need each
    # sub-solve's flex_data; ``solve_steps`` bundles every roll's
    # ``(solve_name, flex_data)`` so ``par`` / ``s`` are unioned over
    # the FULL dt axis — matching the union ``v`` carries from the
    # per-sub-solve parquets read by ``write_outputs_for_solve``.
    # ``solution`` is the last step's; downstream consumers
    # (entity_all_capacity, etc.) only read post-solve cumulative
    # state which is invariant across rolls of a single cascade.
    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
        raw_output_dir=str(workdir / "output_raw"),
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.solution)
            for s in steps.values()
        ],
    )
    elapsed_seconds = time.perf_counter() - t_start

    # Compare (or regenerate) each expected CSV
    if regenerate == scenario:
        for csv_name in csvs:
            actual_path = workdir / "output_csv" / scenario / csv_name
            assert actual_path.exists(), (
                f"Expected output not found: {actual_path}\n"
                f"Check that '{csv_name}' is a valid filename from templates/default_plots.yaml"
            )
            expected_path = EXPECTED_DIR / scenario / csv_name
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            if _is_freeform_csv(csv_name):
                shutil.copy2(actual_path, expected_path)
            else:
                round_for_comparison(_read_csv(actual_path)).to_csv(expected_path, index=False)
        pytest.skip(f"Regenerated {len(csvs)} file(s) for scenario '{scenario}'")
    else:
        for csv_name in csvs:
            actual_path = workdir / "output_csv" / scenario / csv_name
            expected_path = EXPECTED_DIR / scenario / csv_name

            assert actual_path.exists(), (
                f"Expected output not found: {actual_path}\n"
                f"Check that '{csv_name}' is a valid filename from templates/default_plots.yaml"
            )
            assert expected_path.exists(), (
                f"No golden file at {expected_path.relative_to(REPO_ROOT)}\n"
                f"Generate it with: pytest test/ --regenerate {scenario}"
            )

            if _is_freeform_csv(csv_name):
                actual_text = _strip_timestamps(actual_path.read_text())
                expected_text = _strip_timestamps(expected_path.read_text())
                assert actual_text == expected_text, (
                    f"{scenario}/{csv_name} content differs from expected"
                )
            else:
                actual = round_for_comparison(_read_csv(actual_path))
                expected = round_for_comparison(_read_csv(expected_path))
                # rtol=1e-4 + atol=1e-4: rtol handles large values
                # (millions of EUR ± 100), atol absorbs the ±5e-5
                # rounding step introduced by round_for_comparison(4)
                # for small values close to 1 where a tight rtol-only
                # bound is mathematically tighter than the round step
                # (see TODO in tests/db_utils.py::round_for_comparison
                # for the longer-term tighten-and-regen plan).
                pd.testing.assert_frame_equal(
                    actual,
                    expected,
                    check_exact=False,
                    rtol=1e-4,
                    atol=1e-4,
                    obj=f"{scenario}/{csv_name}",
                )

        # Optional hand-derived objective check. Catches a class of bugs
        # where outputs are self-consistent (golden CSV diff passes) but
        # the underlying objective is wrong.
        if expected_objective is not None:
            summary_path = workdir / "output_csv" / scenario / "summary_solve.csv"
            assert summary_path.exists(), (
                f"summary_solve.csv missing for scenario '{scenario}' "
                f"— required for expected_objective check"
            )
            actual_objective = _parse_summary_solve_objective(summary_path)
            denom = max(abs(expected_objective), 1e-9)
            rel_err = abs(actual_objective - expected_objective) / denom
            assert rel_err <= expected_objective_tolerance, (
                f"objective drift: scenario={scenario} "
                f"expected={expected_objective} got={actual_objective} "
                f"rel_err={rel_err:.3e} tolerance={expected_objective_tolerance:.3e}"
            )

    # Optional timing budget. Budgets are set to ~1.5x the observed max
    # over a small sample of clean runs (see tests/README.md), so a
    # tripped assertion indicates a real performance regression rather
    # than CI noise. Placed last so a CSV/objective regression is
    # reported first; pytest stops at the first failed assertion.
    if time_budget_seconds is not None:
        assert elapsed_seconds <= time_budget_seconds, (
            f"timing regression: scenario={scenario} "
            f"observed={elapsed_seconds:.2f}s budget={time_budget_seconds:.2f}s "
            f"(set in tests/scenarios.yaml; bump if the increase is intended)"
        )
