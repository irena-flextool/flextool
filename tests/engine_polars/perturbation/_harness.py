"""Shared helpers for Tier-6 perturbation tests.

Pattern:
  1. Load a fixture that exercises the objective term.
  2. Solve baseline; capture ``sol.obj`` and any solution variables
     needed for the closed-form delta.
  3. Build a perturbed ``FlexData`` by scaling exactly one parameter.
  4. Solve perturbed.
  5. Predict ``expected_delta`` analytically (see
     ``audit/objective_audit.md`` for the term-by-term factor lists).
  6. Assert actual delta == expected delta within ``1e-6`` rel.

A failing test names the exact missing/wrong factor.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import polars as pl

from polar_high_opt import Param, Problem
from flextool.engine_polars import build_flextool


# ---------------------------------------------------------------------------
# Param scaling

def scale_param(data, field: str, factor: float):
    """Return a new ``FlexData`` with ``data.<field>.frame[value]`` × factor.

    All rows are scaled.  Use :func:`scale_param_filtered` to scale only
    rows matching a column-equality filter.
    """
    p = getattr(data, field)
    if p is None:
        raise ValueError(f"FlexData.{field} is None — cannot scale")
    new = Param(p.dims, p.frame.with_columns(value=pl.col("value") * float(factor)))
    return replace(data, **{field: new})


def scale_param_filtered(data, field: str, factor: float, **filters: Any):
    """Scale ``data.<field>.frame[value]`` × factor only on rows where
    every column in ``filters`` equals the given literal.  Other rows are
    left untouched.

    Example::

        scale_param_filtered(data, "p_unitsize", 2.0, p="coal_plant")
    """
    p = getattr(data, field)
    if p is None:
        raise ValueError(f"FlexData.{field} is None — cannot scale")
    mask = pl.lit(True)
    for col, val in filters.items():
        if col not in p.frame.columns:
            raise ValueError(f"Param {field} has no column {col!r}; "
                             f"available: {p.frame.columns}")
        mask = mask & (pl.col(col) == val)
    new_frame = p.frame.with_columns(
        value=pl.when(mask)
                .then(pl.col("value") * float(factor))
                .otherwise(pl.col("value"))
    )
    return replace(data, **{field: Param(p.dims, new_frame)})


# ---------------------------------------------------------------------------
# Solve helpers

def solve_obj(data, *, include_existing_fixed_cost: bool = False) -> float:
    """Build, solve, return the objective value (raises if non-optimal)."""
    pb = Problem()
    build_flextool(pb, data,
                    include_existing_fixed_cost=include_existing_fixed_cost)
    sol = pb.solve()
    assert sol.optimal, "LP did not solve to optimality"
    return float(sol.obj)


def solve_full(data, *, include_existing_fixed_cost: bool = False):
    """Build & solve; return ``(Problem, Solution)``."""
    pb = Problem()
    build_flextool(pb, data,
                    include_existing_fixed_cost=include_existing_fixed_cost)
    sol = pb.solve()
    return pb, sol


# ---------------------------------------------------------------------------
# Assertion

def assert_obj_changed_by(base_obj: float, perturbed_obj: float,
                           expected_delta: float,
                           rel_tol: float = 1e-6, abs_tol: float = 1.0) -> None:
    """Assert ``perturbed_obj - base_obj`` matches ``expected_delta``.

    Tolerance is relative to ``max(abs_tol, |expected_delta|, |base_obj|)``
    so very small base values don't blow up the relative comparison.
    """
    actual_delta = perturbed_obj - base_obj
    denom = max(abs_tol, abs(expected_delta), abs(base_obj))
    rel = abs(actual_delta - expected_delta) / denom
    assert rel < rel_tol, (
        f"obj delta off: actual={actual_delta!r}, "
        f"expected={expected_delta!r}, base={base_obj!r}, "
        f"perturbed={perturbed_obj!r}, rel={rel!r}"
    )
