"""Native-cascade harness for the Tier 6 perturbation tests.

Δ.22 retired the legacy ``FlexToolRunner.write_input`` + ``run_model``
path the original harness depended on (and the workdir-CSV-as-input
contract was retired in Δ.22-P).  The native cascade derives every
``solve_data/*.csv`` in memory from the SQLite DB on every solve, so
mutating workdir CSVs after a baseline solve has no effect on the next
solve's LP.

The replacement strategy
========================

Each perturbation scenario in this directory uses a **single-solve**
fixture (no rolling-horizon cascade; one HiGHS call yields the entire
objective).  That lets the harness compress the two-solve protocol into
two ``polar_high.Problem`` builds against the same ``FlexData``:

1. Run the native cascade ONCE (:func:`run_chain_from_db`) to:
   * materialise the cascade-derived :class:`FlexData` bundle, and
   * exercise the cost-decomposition output writer (so tests that need
     a baseline per-category obj contribution can read it from
     ``output_csv/<scenario>/costs_discounted.csv``).
2. Compute the manual baseline objective by re-building the LP from the
   cascade's ``FlexData`` and solving with ``Problem.solve()`` — this is
   the apples-to-apples reference for the perturbation comparison
   (cascade objectives can differ by floating noise; the manual rebuild
   yields the canonical reference).
3. Mutate one ``Param`` field of ``FlexData`` (e.g. multiply every value
   by ``factor``, optionally row-masked).
4. Re-solve the mutated ``FlexData`` and compare.

The mutator is structurally typed against the ``FlexData`` slot — not
against a workdir CSV.  Each test passes the slot name + scaling
predicate; the harness does the rest.

The two-solve protocol is purely in-memory; no second cascade run is
needed because the perturbation scenarios are single-solve.  A future
rolling-horizon perturbation suite would need to drive
:func:`run_chain_from_db` twice with a custom ``flex_data_override``
hook; the orchestration plumbing for that is documented separately.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable

import polars as pl
import pytest

from polar_high import Param, Problem

from flextool.engine_polars import build_flextool, run_chain_from_db
from flextool.engine_polars.input import FlexData
from flextool.process_outputs.write_outputs import write_outputs


_TEST_DIR = Path(__file__).resolve().parent.parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Re-export the canonical objective parser so tests share one source of truth.
from test_scenarios import (  # noqa: E402
    OUTPUT_CONFIG,
    _parse_summary_solve_objective,
)


# ---------------------------------------------------------------------------
# Solver options — deterministic, single-thread.
# ---------------------------------------------------------------------------


def _solver_options() -> dict[str, Any]:
    return {"random_seed": 42, "parallel": "off"}


def _solve_in_memory(fd: FlexData) -> tuple[Problem, "Any"]:
    """Build + solve ``fd`` via polar_high directly.  The cascade applies an
    LP-range-based ``scale_the_objective`` adjustment per sub-solve; we
    bypass it here so both the baseline and the perturbed solve use the
    same unscaled coefficients and the obj-delta comparison is exact.
    """
    pb = Problem()
    build_flextool(pb, fd)
    sol = pb.solve(options=_solver_options())
    assert sol.optimal, "solve did not converge to optimal"
    return pb, sol


# ---------------------------------------------------------------------------
# Cascade entry — one-shot baseline.
# ---------------------------------------------------------------------------


def cascade_baseline(
    workdir: Path,
    scenario: str,
    test_db_url: str,
) -> tuple[FlexData, float]:
    """Run the cascade once to materialise FlexData and the cost-decomposition
    CSV.  Returns ``(flex_data, manual_baseline_obj)``.

    The cascade's own objective is discarded — instead we rebuild the LP
    from ``flex_data`` and re-solve with ``Problem.solve()``.  The two
    differ at most by floating-point noise (sub-1e-6 relative); the
    perturbation comparison uses the manual rebuild because the
    perturbed solve also goes through the same manual path, eliminating
    any cascade-side scaling drift.
    """
    steps = run_chain_from_db(
        test_db_url, scenario,
        work_folder=workdir, csv_dump=True, keep_solutions=True,
    )
    last_step = next(reversed(steps.values()))
    assert last_step.flex_data is not None, (
        f"FlexData missing on last step for scenario {scenario!r}; "
        f"keep_solutions=True should preserve it")
    assert last_step.solution is not None and last_step.solution.optimal, (
        f"baseline cascade solve failed for scenario {scenario!r}")

    # Emit costs_discounted.csv etc. so tests can read the per-category
    # baseline contribution.
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

    # Manual baseline obj for the apples-to-apples comparison below.
    _, sol = _solve_in_memory(last_step.flex_data)
    return last_step.flex_data, sol.obj


# ---------------------------------------------------------------------------
# Perturbation primitive — scale a Param's "value" column by ``factor``.
# ---------------------------------------------------------------------------


def scale_param(
    param: Param,
    factor: float,
    *,
    filters: dict[str, object] | None = None,
) -> Param:
    """Return a new :class:`polar_high.Param` whose ``value`` column is
    multiplied by ``factor`` on every row matching ``filters``.

    ``filters`` is a column → value mapping.  Rows not matching are
    preserved unchanged.  Empty ``filters`` scales every row.
    """
    frame = param.frame
    if filters is None or not filters:
        new_frame = frame.with_columns(value=pl.col("value") * factor)
    else:
        mask = pl.lit(True)
        for col, val in filters.items():
            if col not in frame.columns:
                raise KeyError(
                    f"filter column {col!r} not in Param frame; "
                    f"columns are {frame.columns}")
            mask = mask & (pl.col(col) == val)
        new_frame = frame.with_columns(
            value=pl.when(mask).then(pl.col("value") * factor)
                                .otherwise(pl.col("value")))
    return Param(param.dims, new_frame)


def perturbed_obj(
    flex_data: FlexData,
    field_name: str,
    factor: float,
    *,
    filters: dict[str, object] | None = None,
) -> float:
    """Mutate ``flex_data.<field_name>`` by ``scale_param`` and re-solve.

    Returns the manual-rebuild obj from the perturbed LP.
    """
    original = getattr(flex_data, field_name)
    if original is None:
        raise ValueError(
            f"FlexData.{field_name} is None — cannot perturb a slot the "
            f"cascade did not populate.  Check the scenario data.")
    if not isinstance(original, Param):
        raise TypeError(
            f"FlexData.{field_name} is {type(original).__name__}, expected "
            f"polar_high.Param")
    new_param = scale_param(original, factor, filters=filters)
    perturbed = dataclasses.replace(flex_data, **{field_name: new_param})
    _, sol = _solve_in_memory(perturbed)
    return sol.obj


# ---------------------------------------------------------------------------
# Assertion — same signature as the legacy harness for cross-repo grep'ability.
# ---------------------------------------------------------------------------


def assert_obj_changed_by(
    base_obj: float,
    perturbed_obj: float,
    expected_delta: float,
    rel_tol: float = 1e-6,
    abs_tol: float = 1.0,
) -> None:
    """Assert ``(perturbed_obj - base_obj) ≈ expected_delta``.

    Tolerance is ``max(rel_tol * |expected_delta|, abs_tol)`` so very
    small expected deltas don't trigger spurious failures from LP solver
    noise (HiGHS reports objectives at ~1e-7 absolute).
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
# Cost-decomposition lookup for tests #1-3.  Reused unchanged from the
# legacy harness — the CSV is emitted by ``write_outputs`` in
# :func:`cascade_baseline` above.
# ---------------------------------------------------------------------------


# costs_discounted.csv reports values in **M CUR** (millions of
# currency) per the FlexTool default-output convention; the LP
# objective returned by ``polar_high.Problem.solve()`` is in CUR.  We
# scale the parsed category totals by 1e6 so the test compares apples
# to apples in CUR throughout.
_M_TO_BASE_UNIT = 1.0e6


def _parse_costs_discounted_by_category(path: Path) -> dict[str, float]:
    """Return ``{category: sum-of-numeric-row}`` for ``costs_discounted.csv``,
    converted from M CUR → CUR so the result is directly comparable to
    ``polar_high.Problem.solve()``'s objective.

    Single-period scenarios carry one numeric column (named ``"0"``);
    multi-period scenarios have one column per period.  The category
    sum is the row-wise sum across all numeric columns.
    """
    import pandas as pd
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
        out[category] = total * _M_TO_BASE_UNIT
    return out


# ---------------------------------------------------------------------------
# Back-compat shims — the legacy harness exported these names.  Tests
# updated in this directory call the new entry points directly; the
# shims raise to flag any straggler that still imports them.
# ---------------------------------------------------------------------------


def run_baseline(*args: object, **kwargs: object) -> tuple[None, float]:
    raise RuntimeError(
        "tests.perturbation._harness.run_baseline is retired (Δ.22).  "
        "Use cascade_baseline(workdir, scenario, test_db_url) instead.")


def rerun_and_get_obj(*args: object, **kwargs: object) -> float:
    raise RuntimeError(
        "tests.perturbation._harness.rerun_and_get_obj is retired (Δ.22).  "
        "Use perturbed_obj(flex_data, field_name, factor, ...) instead.")


def scale_input_csv_column(*args: object, **kwargs: object) -> Callable[[], None]:
    raise RuntimeError(
        "tests.perturbation._harness.scale_input_csv_column is retired (Δ.22).  "
        "Use perturbed_obj(flex_data, field_name, factor, ...) which mutates "
        "in-memory FlexData rather than workdir CSVs.")
