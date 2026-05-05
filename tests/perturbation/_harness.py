"""Harness for Tier 6 perturbation tests.

Each perturbation test runs the same scenario twice in one workdir:

1. Baseline solve via :func:`run_baseline` — write_input, preprocessing,
   solve, write_outputs(csv) → parse ``summary_solve.csv`` for the
   baseline objective. Returns the still-live ``FlexToolRunner`` so the
   second invocation can reuse the workdir without re-running
   ``write_input`` (which would overwrite a mutated CSV).

2. Mutate one column of one CSV in ``solve_data/`` (or ``input/``).
   The mutator targets long-format ``solve,period,time,value`` files
   (``pdtNode.csv``, ``pdtGroup.csv``, …), period-keyed long files
   (``pdProcess.csv``, ``p_inflation_factor_operations_yearly.csv``),
   and short value-only files (``steps_in_use.csv``).

3. Re-run via :func:`rerun_and_get_obj`. The rerun cleans
   ``output_raw/``, ``HiGHS.log``, ``output_csv/<scenario>/``, and the
   per-solve scaling caches; without the cleanup the second solve
   reuses stale parquet/log/cache state.

4. Assert with :func:`assert_obj_changed_by`. The signature mirrors
   ``flexpy_spike``'s harness so cross-repo failure messages line up.

Critical detail
---------------
``preprocessing_solve_time.run(...)`` (called inside ``run_model``) and
``solve_writers.write_active_timelines`` (called from the orchestrator
solve loop) overwrite many files in ``solve_data/`` from the immutable
``input/`` CSVs and ``state.timeline``. A mutation that lives in those
preprocessor-managed files (e.g. ``pdtNode.csv``) is therefore
clobbered before the LP is generated — unless the writer is
monkey-patched. :func:`scale_input_csv_column` therefore patches the
relevant writer with a wrapper that runs the original and then
re-applies the scaling pass on top, persisting through the rerun.
The patch is captured on a context-manager-style ``unpatch`` callable
that the test removes after the rerun (or pytest cleanup) so the next
test's writer is back to normal.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

_TEST_DIR = Path(__file__).resolve().parent.parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Re-export the canonical objective parser so tests share one source of truth.
from test_scenarios import (  # noqa: E402
    OUTPUT_CONFIG,
    _parse_summary_solve_objective,
)

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402


# ---------------------------------------------------------------------------
# Per-target writer patch table.
#
# Every target file we want to perturb is rewritten on every solve by one
# of the orchestration / preprocessing functions. ``_PATCH_SPEC`` maps a
# rel_path (relative to workdir) to the (module, function_name) that
# emits that file last before the solver runs. The harness installs a
# wrapper around the function: original write → re-apply mutation.
#
# Two distinct "last writers" exist for the same target file in some
# cases (write_input emits an initial version and preprocessing rewrites
# it on every solve); only the per-solve writer needs to be patched
# because mutations in ``input/`` are only consumed by the per-solve
# pass.
# ---------------------------------------------------------------------------

_PATCH_SPEC: dict[str, tuple[str, str]] = {
    # Test 1 — operational inflation factor (per-period long file).
    "solve_data/p_inflation_factor_operations_yearly.csv": (
        "flextool.flextoolrunner.preprocessing.period_calculated_params",
        "write_period_calculated_params",
    ),
    # Test 2 — co2 price lives in pdtGroup (long-format with `value` col).
    "solve_data/pdtGroup.csv": (
        "flextool.flextoolrunner.preprocessing.entity_period_calc_params",
        "write_pdtGroup",
    ),
    # Test 3 — startup_cost lives in pdProcess (long-format with `value` col).
    "solve_data/pdProcess.csv": (
        "flextool.flextoolrunner.preprocessing.entity_period_calc_params",
        "write_entity_period_calc_params",
    ),
    # Test 4 — penalty_up lives in pdtNode (long-format with `value` col).
    "solve_data/pdtNode.csv": (
        "flextool.flextoolrunner.preprocessing.entity_period_calc_params",
        "write_pdtNode",
    ),
    # Test 5 — step duration lives in steps_in_use (period,step,step_duration).
    # Written by orchestration via solve_writers.write_active_timelines
    # BEFORE preprocessing. Patching it before run_model() suffices.
    "solve_data/steps_in_use.csv": (
        "flextool.flextoolrunner.solve_writers",
        "write_active_timelines",
    ),
}


def run_baseline(
    workdir: Path,
    scenario: str,
    test_db_url: str,
    test_bin_dir: Path,
) -> tuple[FlexToolRunner, float]:
    """Write input, run model, parse summary_solve.csv → baseline obj.

    Returns the runner so the caller can re-invoke ``run_model()`` after
    mutating a CSV. Calling ``write_input`` again would overwrite the
    mutation, so the caller MUST NOT do that — the harness's
    :func:`rerun_and_get_obj` is the supported re-run path.
    """
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(test_db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Baseline run failed for scenario {scenario!r}"
    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
    )
    summary_path = workdir / "output_csv" / scenario / "summary_solve.csv"
    return runner, _parse_summary_solve_objective(summary_path)


# ---------------------------------------------------------------------------
# CSV mutator
# ---------------------------------------------------------------------------


def _apply_scale_to_csv(
    csv_path: Path,
    column: str,
    factor: float,
    filters: dict[str, object],
) -> None:
    """Read ``csv_path``, multiply ``column`` by ``factor`` for rows
    matching ``filters``, write back. Both wide and long formats are
    supported — ``column`` simply names whichever column carries the
    numeric value in the file's actual layout.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"perturbation target does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise KeyError(
            f"column {column!r} not in {csv_path} — columns are {list(df.columns)}"
        )
    mask = pd.Series(True, index=df.index)
    for k, v in filters.items():
        if k not in df.columns:
            raise KeyError(
                f"filter column {k!r} not in {csv_path} — columns are {list(df.columns)}"
            )
        mask &= df[k] == v
    if not mask.any():
        raise ValueError(
            f"no rows in {csv_path} match filters {filters} on columns "
            f"{list(df.columns)}"
        )
    # Cast the target column to float so multiplication is well-defined
    # even when the source CSV stores integers.
    df.loc[mask, column] = df.loc[mask, column].astype(float) * factor
    df.to_csv(csv_path, index=False)


def scale_input_csv_column(
    workdir: Path,
    rel_path: str,
    column: str,
    factor: float,
    **filters: object,
) -> Callable[[], None]:
    """Schedule a multiply-by-``factor`` mutation on ``column`` in
    ``workdir/<rel_path>``.

    The mutation is "scheduled" because the file is rewritten by
    preprocessing on every ``run_model()``. We work around that by
    monkey-patching the responsible writer to re-apply the mutation
    after its normal output. The patch is installed immediately and
    persists until the returned ``unpatch`` callable is invoked
    (typically inside the test, after the rerun has been read).

    The scaling also runs once immediately (so callers can inspect the
    mutated file in-line without an extra solve) — the wrapper then
    re-applies it on the next preprocessing pass.
    """
    if rel_path not in _PATCH_SPEC:
        raise KeyError(
            f"no patch entry for {rel_path!r}; add an entry to _PATCH_SPEC "
            f"or pick a different target file"
        )
    module_name, func_name = _PATCH_SPEC[rel_path]

    target = workdir / rel_path
    # Apply once now so the file matches the mutation immediately.
    _apply_scale_to_csv(target, column, factor, filters)

    # And install the wrapper so subsequent rewrites get re-mutated.
    import importlib
    module = importlib.import_module(module_name)
    original = getattr(module, func_name)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        if target.exists():
            _apply_scale_to_csv(target, column, factor, filters)
        return result

    setattr(module, func_name, wrapper)

    def unpatch() -> None:
        setattr(module, func_name, original)

    return unpatch


# ---------------------------------------------------------------------------
# Re-run (no write_input — mutation must survive)
# ---------------------------------------------------------------------------


def rerun_and_get_obj(runner: FlexToolRunner, workdir: Path, scenario: str) -> float:
    """Re-run ``run_model()`` on the same workdir and return the new
    objective.

    Cleans the per-run outputs that would otherwise carry over
    baseline state (output_raw parquets, HiGHS.log, the previous
    write_outputs CSVs, the LP-scaling per-solve cache) so the second
    solve sees a clean slate. Notably does NOT call ``write_input``
    again — that would overwrite the mutated CSVs.
    """
    # Clear stale per-run artefacts.
    output_raw = workdir / "output_raw"
    if output_raw.exists():
        shutil.rmtree(output_raw)
    highs_log = workdir / "HiGHS.log"
    if highs_log.exists():
        highs_log.unlink()
    scenario_csv = workdir / "output_csv" / scenario
    if scenario_csv.exists():
        shutil.rmtree(scenario_csv)
    # LP-scaling per-solve cache — cleared inside one test as well as at
    # test boundaries (the autouse fixture handles boundaries; this
    # call clears the cache within a single test between baseline and
    # perturbed solves so the second solve re-analyses the mutated CSVs).
    from flextool.flextoolrunner import scaling as _scaling
    _scaling.clear_cache()
    # Per-solve scaling artefact — recomputed by the analyser on the
    # next run, but stale-file-deleted defensively.
    scale_json = workdir / "solve_data" / "scaling_analysis.json"
    if scale_json.exists():
        scale_json.unlink()

    return_code = runner.run_model()
    assert return_code == 0, f"Perturbed run failed for scenario {scenario!r}"
    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
    )
    summary_path = workdir / "output_csv" / scenario / "summary_solve.csv"
    return _parse_summary_solve_objective(summary_path)


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------


def assert_obj_changed_by(
    base_obj: float,
    perturbed_obj: float,
    expected_delta: float,
    rel_tol: float = 1e-6,
    abs_tol: float = 1.0,
) -> None:
    """Assert ``(perturbed_obj - base_obj) ≈ expected_delta``.

    The tolerance is ``max(rel_tol * |expected_delta|, abs_tol)`` so
    very small expected deltas don't trigger spurious failures from
    LP solver noise (HiGHS reports objectives at ~1e-7 absolute).
    """
    observed = perturbed_obj - base_obj
    tol = max(rel_tol * abs(expected_delta), abs_tol)
    assert abs(observed - expected_delta) <= tol, (
        f"objective delta mismatch: "
        f"expected_delta={expected_delta} observed_delta={observed} "
        f"base_obj={base_obj} perturbed_obj={perturbed_obj} "
        f"tolerance=max({rel_tol}*|{expected_delta}|, {abs_tol})={tol}"
    )


# ---------------------------------------------------------------------------
# Local helper for tests #2 and #3 — single-category baseline lookup.
#
# (Kept local because the decomposition tier may not yet be merged. If
# tests/decomposition/_helpers.py grows a category-aware helper later,
# tests can switch to that.)
# ---------------------------------------------------------------------------


def _parse_costs_discounted_by_category(path: Path) -> dict[str, float]:
    """Return ``{category: sum-of-numeric-row}`` for ``costs_discounted.csv``.

    Single-period scenarios carry one numeric column (named ``"0"``);
    multi-period scenarios have one column per period. The category
    sum is the row-wise sum across all numeric columns.
    """
    df = pd.read_csv(path)
    if df.columns[0] != "category":
        raise ValueError(
            f"Unexpected header in {path}: expected first column 'category', "
            f"got {df.columns[0]!r}"
        )
    out: dict[str, float] = {}
    value_cols = df.columns[1:]
    for _, row in df.iterrows():
        category = row["category"]
        total = 0.0
        for col in value_cols:
            try:
                total += float(row[col])
            except (TypeError, ValueError):
                continue
        out[category] = total
    return out
