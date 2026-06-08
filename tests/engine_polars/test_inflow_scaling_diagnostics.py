"""Focused unit test for the inflow-scaling ``f`` annualisation diagnostic.

Part B of ``specs/timeslice_weight_alignment.md``.  The diagnostic reports,
per scaling (node, period), the factor by which an even-sample annualisation
and the representative-weight annualisation disagree::

    f[n, d] = ( Σ_t I[n, t] ) / ( Σ_t I[n, t]·w[d, t] )

It is a PURE diagnostic — it never enters the LP and is emitted as the
standalone ``inflow_scaling_diagnostics.csv`` frame (NOT one of the 6
parity-gated scaling frames).  This test asserts:

* a uniform / weight-free input → ``f == 1.0`` for every (n, d) and NO
  warning fires;
* a non-uniform-``timeset_weights`` input → ``f != 1.0`` for the affected
  node and matches the hand-computed ``Σ I / Σ I·w`` for at least one
  (n, d), and the WARNING fires.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_inflow_scaling import (
    _compute_inflow_scaling_diagnostics,
    emit_node_inflow_scaling_params,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider

INP = Path("input")
SDD = Path("solve_data")


def _base_provider(*, rp_cost_weight: dict[tuple[str, str], float] | None,
                   ) -> FlexDataProvider:
    """A minimal Provider with one scaling node (``annN``,
    scale_to_annual_flow) over a single period ``d1`` with four timesteps.

    When *rp_cost_weight* is given it is authored as ``rp_cost_weight.csv``
    (period, time, weight); when ``None`` the file is absent → ``w ≡ 1.0``.
    Inflow profile is deliberately non-flat (1, 2, 3, 4) so a non-uniform
    weight moves the weighted sum away from the unweighted one.
    """
    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    times = ["t1", "t2", "t3", "t4"]
    inflow = [1.0, 2.0, 3.0, 4.0]

    put("input", "node", pl.DataFrame({"node": ["annN"]}))
    put("solve_data", "period_in_use_set", pl.DataFrame({"period": ["d1"]}))
    put("solve_data", "time", pl.DataFrame({"time": times}))
    put("input", "p_node", pl.DataFrame(
        {"node": [], "param": [], "value": []},
        schema={"node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": ["annN"] * 4, "time": times, "value": inflow}))
    put("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["annN"], "method": ["scale_to_annual_flow"]}))
    put("solve_data", "pdNode", pl.DataFrame({
        "node": ["annN"], "param": ["annual_flow"],
        "period": ["d1"], "value": [100.0]}))
    put("solve_data", "complete_period_share_of_year_calc",
        pl.DataFrame({"period": ["d1"], "value": [0.5]}))
    put("solve_data", "p_timeline_duration_in_years", pl.DataFrame(
        {"timeline": [], "value": []},
        schema={"timeline": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "period__timeline_set", pl.DataFrame(
        {"period": [], "timeline": []},
        schema={"period": pl.Utf8, "timeline": pl.Utf8}))
    put("solve_data", "complete_time_in_use_set",
        pl.DataFrame({"time": times}))
    put("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d1"] * 4, "time": times}))

    if rp_cost_weight is not None:
        put("solve_data", "rp_cost_weight", pl.DataFrame({
            "period": [k[0] for k in rp_cost_weight],
            "time": [k[1] for k in rp_cost_weight],
            "weight": list(rp_cost_weight.values())}))

    return provider


def _diag_rows(frame: pl.DataFrame) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in frame.iter_rows(named=True):
        out[(r["node"], r["period"])] = {
            "inflow_method": r["inflow_method"],
            "annual_flow": float(r["annual_flow"]),
            "f": float(r["f"]),
            "level_shift_pct": float(r["level_shift_pct"]),
        }
    return out


def test_uniform_input_f_is_one_and_no_warning():
    """Weight-free input: f == 1.0 everywhere, no warning."""
    provider = _base_provider(rp_cost_weight=None)
    frame, warnings = _compute_inflow_scaling_diagnostics(
        INP, SDD, provider=provider)

    rows = _diag_rows(frame)
    assert ("annN", "d1") in rows
    row = rows[("annN", "d1")]
    assert row["inflow_method"] == "scale_to_annual_flow"
    assert row["annual_flow"] == 100.0
    assert row["f"] == 1.0, f"uniform input must give f==1.0; got {row['f']}"
    assert row["level_shift_pct"] == 0.0
    assert warnings == [], f"no warning expected on uniform input; {warnings}"


def test_uniform_explicit_weights_f_is_one():
    """Explicit but UNIFORM weights (all 1.0) also give f == 1.0 and no
    warning (the divergence is only meaningful for non-uniform weights)."""
    w = {("d1", t): 1.0 for t in ("t1", "t2", "t3", "t4")}
    provider = _base_provider(rp_cost_weight=w)
    frame, warnings = _compute_inflow_scaling_diagnostics(
        INP, SDD, provider=provider)
    rows = _diag_rows(frame)
    assert rows[("annN", "d1")]["f"] == 1.0
    assert warnings == []


def test_nonuniform_weights_f_matches_handcompute_and_warns():
    """Non-uniform timeset_weights (0.5/1.0/1.5/2.0) → f != 1.0, matches the
    hand-computed Σ I / Σ I·w, AND the warning fires."""
    weights = [0.5, 1.0, 1.5, 2.0]
    times = ["t1", "t2", "t3", "t4"]
    inflow = [1.0, 2.0, 3.0, 4.0]
    w = {("d1", t): wv for t, wv in zip(times, weights)}
    provider = _base_provider(rp_cost_weight=w)

    frame, warnings = _compute_inflow_scaling_diagnostics(
        INP, SDD, provider=provider)
    rows = _diag_rows(frame)

    # Hand-compute: Σ I = 1+2+3+4 = 10; Σ I·w = 0.5+2+4.5+8 = 15 → f = 10/15.
    num = sum(inflow)
    den = sum(i * wv for i, wv in zip(inflow, weights))
    expected_f = num / den
    assert ("annN", "d1") in rows
    f_val = rows[("annN", "d1")]["f"]
    assert f_val != 1.0, "non-uniform weights must move f off 1.0"
    assert abs(f_val - expected_f) <= 1e-12, (
        f"f={f_val} != hand-computed {expected_f}")
    assert abs(rows[("annN", "d1")]["level_shift_pct"]
               - 100.0 * (expected_f - 1.0)) <= 1e-9

    # |f - 1| = |10/15 - 1| = 1/3 > 0.01 → warning fires for this (n, d).
    assert ("annN", "d1") in [(n, d) for n, d, _ in warnings], (
        f"non-uniform |f-1|>1% must warn; warnings={warnings}")


def test_warning_logged_at_warning_level(caplog):
    """The public emitter logs the WARNING line (naming node/period/f) for a
    non-uniform divergence, and still emits the diagnostic frame to the
    Provider."""
    weights = [0.5, 1.0, 1.5, 2.0]
    times = ["t1", "t2", "t3", "t4"]
    w = {("d1", t): wv for t, wv in zip(times, weights)}
    provider = _base_provider(rp_cost_weight=w)

    with caplog.at_level(
        logging.WARNING,
        logger="flextool.engine_polars._emit_inflow_scaling",
    ):
        emit_node_inflow_scaling_params(INP, SDD, provider=provider)

    msgs = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING]
    assert any("inflow scaling" in m and "annN" in m and "d1" in m
               and "f=" in m for m in msgs), (
        f"expected a node/period/f warning; got {msgs}")

    # The standalone diagnostic frame reached the Provider (surfaced via
    # the solve_data key, NOT mixed into the scaling-frame parity set).
    diag = provider.get("solve_data/inflow_scaling_diagnostics")
    assert diag is not None
    assert ("annN", "d1") in {
        (r[0], r[1]) for r in diag.select(["node", "period"]).iter_rows()}


def test_no_warning_on_uniform_input_via_emitter(caplog):
    """Uniform input through the public emitter logs NO inflow-scaling
    warning."""
    provider = _base_provider(rp_cost_weight=None)
    with caplog.at_level(
        logging.WARNING,
        logger="flextool.engine_polars._emit_inflow_scaling",
    ):
        emit_node_inflow_scaling_params(INP, SDD, provider=provider)
    msgs = [r.getMessage() for r in caplog.records
            if "inflow scaling" in r.getMessage()]
    assert msgs == [], f"no inflow-scaling warning expected; got {msgs}"
