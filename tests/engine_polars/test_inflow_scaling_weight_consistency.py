"""Hand-calculated correctness guard for the timeslice-weight fix.

Final commit of ``specs/timeslice_weight_alignment.md``.  There was no
pre-existing test/golden for this path, so every expected number below is
worked out BY HAND on a 2-3 timestep example and asserted literally — not
merely round-tripped (a round-trip ``reported == annual_flow`` can pass
vacuously because the multiplier is *constructed* to invert the
aggregation, so it would not catch a wrong formula).

The load-bearing number is ``period_flow_annual_multiplier`` (``M``): A1
builds it from the WEIGHTED sum ``Σ_t I·w`` (``w = p_timestep_weight``).
Pinning ``M`` to its hand value catches a regression to the old uniform
annualisation (which would give a different ``M``).  We also replay the A4
output aggregation (``Σ scaled·w / cpsoy``) and check it lands on the hand
value — so demand-scaling (A1) and energy-reporting (A4) are shown to use
the *same* ``w`` (the three-way-consistency property), with literal numbers
at every step.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_inflow_scaling import (
    _compute_inflow_scaling_frames,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider

INP = Path("input")
SDD = Path("solve_data")


def _provider(
    *,
    weights: list[float] | None,
    inflow: list[float],
    annual_flow: float,
    cpsoy: float,
    method: str = "scale_to_annual_flow",
    peak_inflow: float | None = None,
) -> tuple[FlexDataProvider, list[str]]:
    """One node ``annN`` over one period ``d1`` with ``len(inflow)``
    timesteps.  ``weights=None`` → no ``timestep_weight.csv`` (``w ≡ 1``);
    a list → the per-(period, time) representative weight.
    """
    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    times = [f"t{i + 1}" for i in range(len(inflow))]

    put("input", "node", pl.DataFrame({"node": ["annN"]}))
    put("solve_data", "period_in_use_set", pl.DataFrame({"period": ["d1"]}))
    put("solve_data", "time", pl.DataFrame({"time": times}))
    put("input", "p_node", pl.DataFrame(
        {"node": [], "param": [], "value": []},
        schema={"node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "pt_node_inflow", pl.DataFrame({
        "node": ["annN"] * len(inflow), "time": times, "value": inflow}))
    put("solve_data", "node__inflow_method", pl.DataFrame({
        "node": ["annN"], "method": [method]}))

    pd_rows = [("annN", "annual_flow", annual_flow)]
    if peak_inflow is not None:
        pd_rows.append(("annN", "peak_inflow", peak_inflow))
    put("solve_data", "pdNode", pl.DataFrame({
        "node": [r[0] for r in pd_rows],
        "param": [r[1] for r in pd_rows],
        "period": ["d1"] * len(pd_rows),
        "value": [r[2] for r in pd_rows]}))
    put("solve_data", "complete_period_share_of_year_calc",
        pl.DataFrame({"period": ["d1"], "value": [cpsoy]}))
    put("solve_data", "p_timeline_duration_in_years", pl.DataFrame(
        {"timeline": [], "value": []},
        schema={"timeline": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "period__timeline_set", pl.DataFrame(
        {"period": [], "timeline": []},
        schema={"period": pl.Utf8, "timeline": pl.Utf8}))
    put("solve_data", "complete_time_in_use_set",
        pl.DataFrame({"time": times}))
    # dt_complete[d1] = every timestep — the A4 output stage sums over these.
    put("solve_data", "steps_complete_solve", pl.DataFrame({
        "period": ["d1"] * len(times), "time": times}))

    if weights is not None:
        put("solve_data", "timestep_weight", pl.DataFrame({
            "period": ["d1"] * len(weights),
            "time": times,
            "weight": weights}))

    return provider, times


def _cell(frame: pl.DataFrame, key: tuple[str, str]) -> float:
    """Value of a (key1, key2, repr(value)) scaling frame at ``key``."""
    return {(r[0], r[1]): float(r[2]) for r in frame.iter_rows()}[key]


def _pti(frame: pl.DataFrame) -> dict[str, float]:
    """``ptNode_inflow`` (node, time, repr(value)) → {time: value} for annN."""
    return {r[1]: float(r[2]) for r in frame.iter_rows() if r[0] == "annN"}


def test_scale_to_annual_flow_hand_calculated():
    """scale_to_annual_flow, NON-uniform weights — every number by hand.

        I = [10, 30],  w = [1, 3],  cpsoy = 0.5,  annual_flow = 80
        Σ I·w        = 10·1 + 30·3            = 100
        period_share = |Σ I·w| / annual_flow  = 100 / 80 = 1.25
        multiplier M = cpsoy / period_share   = 0.5 / 1.25 = 0.4   <- pinned
        scaled inflow = M·I                   = [4, 12]
        reported      = Σ (scaled·w) / cpsoy  = (4·1 + 12·3) / 0.5 = 80 = AF

    Dropping the weight (uniform Σ I = 40) would give M = 0.5·80/40 = 1.0,
    so pinning 0.4 directly catches a regression to uniform annualisation.
    """
    inflow, weights = [10.0, 30.0], [1.0, 3.0]
    annual_flow, cpsoy = 80.0, 0.5
    provider, times = _provider(
        weights=weights, inflow=inflow, annual_flow=annual_flow, cpsoy=cpsoy)
    frames = _compute_inflow_scaling_frames(INP, SDD, provider=provider)

    mult = _cell(frames["period_flow_annual_multiplier.csv"], ("annN", "d1"))
    assert abs(mult - 0.4) <= 1e-12, f"multiplier {mult} != hand-calc 0.4"

    pti = _pti(frames["ptNode_inflow.csv"])
    scaled = [mult * pti[t] for t in times]
    assert all(abs(s - e) <= 1e-12 for s, e in zip(scaled, [4.0, 12.0])), \
        f"scaled {scaled} != hand-calc [4, 12]"

    reported = sum(s * w for s, w in zip(scaled, weights)) / cpsoy
    assert abs(reported - 80.0) <= 1e-12, f"reported {reported} != 80"


def test_uniform_control_hand_calculated():
    """Control: w ≡ 1 (no timestep_weight) on the same inputs.

        Σ I = 40,  period_share = 40/80 = 0.5,  M = 0.5/0.5 = 1.0
        reported = Σ scaled / cpsoy = 40 / 0.5 = 80 = annual_flow

    M = 1.0 here vs 0.4 in the weighted case isolates the weighting effect;
    both still land on annual_flow, so the fix is correct in both regimes.
    """
    inflow = [10.0, 30.0]
    annual_flow, cpsoy = 80.0, 0.5
    provider, times = _provider(
        weights=None, inflow=inflow, annual_flow=annual_flow, cpsoy=cpsoy)
    frames = _compute_inflow_scaling_frames(INP, SDD, provider=provider)

    mult = _cell(frames["period_flow_annual_multiplier.csv"], ("annN", "d1"))
    assert abs(mult - 1.0) <= 1e-12, f"uniform multiplier {mult} != 1.0"

    pti = _pti(frames["ptNode_inflow.csv"])
    reported = sum(mult * pti[t] for t in times) / cpsoy
    assert abs(reported - 80.0) <= 1e-12, f"reported {reported} != 80"


def test_peak_method_annual_leg_uses_weighted_orig_flow_sum():
    """scale_to_annual_and_peak_flow's ANNUAL leg (A3) is built from the
    WEIGHTED orig_flow_sum (Σ I·w), not the uniform Σ I.  By hand, with
    I = [5, 6, 7, 8], w = [0.5, 1, 1.5, 2]:

        weighted orig_flow_sum = 5·0.5 + 6·1 + 7·1.5 + 8·2 = 35   <- pinned
        uniform  orig_flow_sum = 5 + 6 + 7 + 8              = 26

    The peak match itself stays UNWEIGHTED (spec A3), so we pin the
    load-bearing weighted ingredient directly rather than round-trip the
    whole affine map.
    """
    provider, _times = _provider(
        weights=[0.5, 1.0, 1.5, 2.0], inflow=[5.0, 6.0, 7.0, 8.0],
        annual_flow=80.0, cpsoy=0.5,
        method="scale_to_annual_and_peak_flow", peak_inflow=12.0)
    frames = _compute_inflow_scaling_frames(INP, SDD, provider=provider)

    ofs = _cell(frames["orig_flow_sum.csv"], ("annN", "d1"))
    assert abs(ofs - 35.0) <= 1e-12, \
        f"orig_flow_sum {ofs} != weighted hand-calc 35 (uniform would be 26)"
