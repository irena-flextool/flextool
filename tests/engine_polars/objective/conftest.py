"""Shared fixtures for the engine_polars objective audit (Surface B
items B.16–B.20).

Reuses the constraint fixtures via ``pytest_plugins`` so each toy
(``toy_1n1p_1d2t``, ``toy_storage_2t`` …) is available here without
duplicating construction.  Adds:

* ``toy_costs_only_1d2t`` — no demand, no slack-active state.  Each
  cost term can be activated by perturbing exactly one parameter.
* ``objective_term_value(problem, term_name) -> float`` — best-effort
  per-term contribution extractor against ``problem._obj_terms``.
* ``solve_problem(data) -> (Problem, Solution)`` — convenience.
"""
from __future__ import annotations

import sys
from typing import Any  # noqa: F401  # used in string annotation on solve_problem

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


# The sibling ``constraints/conftest.py`` is loaded as a top-level pytest
# plugin (``_engine_polars_constraints_conftest``) by the root
# ``tests/conftest.py``; pytest 8.x forbids declaring ``pytest_plugins`` in
# non-top-level conftests.  We still need a direct module handle here to
# re-export ``solver_options``.
_constraints_module = sys.modules["_engine_polars_constraints_conftest"]

solver_options = _constraints_module.solver_options


# ---------------------------------------------------------------------------
# toy_costs_only_1d2t — same shape as toy_1n1p_1d2t but with zero
# inflow and explicit price levels on every cost-term param so each
# can be selectively perturbed.

@pytest.fixture(scope="function")
def toy_costs_only_1d2t() -> FlexData:
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step_duration = Param(("d", "t"),
        dt.with_columns(value=pl.lit(1.0)))
    p_timestep_weight = Param(("d", "t"),
        dt.with_columns(value=pl.lit(1.0)))
    p_inflation_op = Param(("d",),
        pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_period_share = Param(("d",),
        pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    nodeBalance = pl.DataFrame({"n": ["n"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    # Zero inflow: nodeBalance is trivially satisfied with zero v_flow,
    # so no slack penalty fires unless the test perturbs p_inflow.
    p_inflow = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(0.0))
                       .select("n", "d", "t", "value"))
    # Penalty levels are explicit: tests scale them to surface the
    # nodeBalance slack term in isolation.
    p_pen_up = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(1000.0))
                       .select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nodeBalance_dt.with_columns(value=pl.lit(1000.0))
                       .select("n", "d", "t", "value"))

    pss = pl.DataFrame({"p": ["p"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))

    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["FUEL_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(
        ("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    # Non-zero commodity price → perturbing it surfaces the commodity
    # buy term.  Setting inflow to a negative value is the test's lever.
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(5.0))
          .select("c", "d", "t", "value"))

    return FlexData(
        dt=dt, p_step_duration=p_step_duration, p_timestep_weight=p_timestep_weight,
        p_inflation_op=p_inflation_op, p_period_share=p_period_share,
        nodeBalance=nodeBalance, nodeBalance_dt=nodeBalance_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
    )


# ---------------------------------------------------------------------------
# Helpers

def solve_problem(data: FlexData) -> tuple[Problem, "Any"]:
    """Build + solve and return ``(Problem, Solution)``."""
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


def objective_term_value(problem: Problem, term_name: str) -> float:
    """Best-effort: contribution of a single objective term to ``sol.obj``.

    polar_high stores objective terms as a list of ``_Term`` instances on
    ``problem._obj_terms``.  Each term carries a polars LazyFrame with
    columns ``(*dims, col_id, coef)``.  Variables themselves carry a
    ``name``; we approximate per-term selection by matching the term's
    column-id range against vars whose name starts with ``term_name``.

    This is intentionally pragmatic: there is no first-class per-term API.
    The function exists so downstream tests have one place to evolve the
    extraction logic without duplicating it across files.

    Returns ``float('nan')`` when no matching variable exists, so callers
    can ``assert math.isnan(...)`` to detect the "term not present" case.
    """
    # Resolve variable col_id range for the matching var.
    matching_cols: set[int] = set()
    for vname, var in problem._vars.items():
        if vname == term_name or vname.startswith(term_name):
            # Var's domain frame carries the per-row col_id assignment.
            try:
                col_ids = var.col_ids
            except AttributeError:
                continue
            if isinstance(col_ids, (list, tuple)):
                matching_cols.update(int(c) for c in col_ids)
            else:
                # polars.Series / np.ndarray
                matching_cols.update(int(c) for c in col_ids)
    if not matching_cols:
        return float("nan")

    # Re-evaluate every obj term against the latest solution col_value.
    # We don't re-solve; the caller is expected to have already solved
    # ``problem``.  ``Solution`` isn't held on Problem, so this helper
    # takes only Problem and pulls the most-recent ``col_value`` via the
    # private cache (re-solving would defeat the purpose).
    raise NotImplementedError(
        "objective_term_value: per-term decomposition is not yet wired."
        "  polar_high lacks a public per-term API; agents adding obj-side"
        " tests should compute term values directly from the closed form"
        " (see tests/engine_polars/synthetic/test_flex_toy_*.py for the"
        " pattern), or extend this helper once the upstream API lands."
    )
