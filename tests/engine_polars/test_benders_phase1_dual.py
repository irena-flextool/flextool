"""Benders (Option C) Phase 1 — dual-verification harness.

Proves the "pin boundary injection → extract dual" mechanism yields the
CORRECT Benders cut slope on the prototype fixture
``lh2_three_region_trade_invest`` (regions A/B/C, greenfield investable
``pipe_AB`` / ``pipe_BC``).  NO Benders loop, NO master problem — this is a
self-contained verification of the interface every Phase 2 cut depends on.

Mechanism (see ``specs/benders_option_c.md`` §A–E + the PROCEED-WITH-CHANGES
critique):

1. **Monolith reference.** ``build_flextool`` over the undecomposed fixture
   data, solve.  Read the optimal FORWARD trade flows ``f*`` per ``(d, t)``
   on ``pipe_AB(lh2_A→lh2_B)`` and ``pipe_BC(lh2_B→lh2_C)`` and the boundary
   BLOCK-node marginals (``nodeBalanceBlock_eq``; lh2_* are block nodes, so
   ``nodeBalance_eq`` anti-joins them out — do NOT read it).

2. **Region split** via ``_region_filter.split`` (A/B/C subproblems).

3. **Pin + capacity patch.**  Patch a TEST-LOCAL ``p_flow_upper_existing``
   for the forward half-flow's virtual arc to a cap STRICTLY ≫ ``max f*``
   (asserted), so the ``maxFlow`` row stays slack and cannot leak into the
   dual; pin the forward half-flow's ``v_flow`` columns PER-``(d, t)`` to
   ``f*`` via ``WarmProblem.fix_cols`` (lower=upper), and pin the reverse
   half-flow to 0.  ``v_flow`` is unitsize-normalised (pipes unitsize=1000)
   and we pin the SAME normalised value the monolith reports — no unit
   conversion on the pin (the stale ``_inject_half_flows`` "unitsize = 1.0"
   docstring is wrong; the half-flow inherits 1000).

4. **Dual extraction.**  After solving each pinned region subproblem, the
   reduced cost of the pinned ``v_flow`` column
   (``Solution.col_dual[col_id]``) is the exact cut slope
   ``∂(region operating cost)/∂f̄``.  We verify it against a one-sided
   finite-difference re-solve (the true derivative on the binding side), and
   we map it to the region's own boundary block-node dual via
   ``rc = node_dual_raw × unitsize × (slope|1) × block_step_duration``.

PHASE 1 FINDING — exporter boundary dual ≠ monolith boundary dual (by
design).  The brief's draft PRIMARY gate "region block dual == monolith block
dual at that node" holds for the IMPORTER (region C: exact match — C's local
marginal value of energy equals the monolith's because the import sets C's
margin) but NOT the EXPORTER (region A).  In the monolith the boundary prices
form a chain ``π_B ≈ π_A·slope``, ``π_C ≈ π_B·slope`` (downstream C-scarcity
back-propagated through the efficiency-losing pipe), so the monolith's lh2_A
price (≈92245) is the DELIVERED-to-C value, not A's local marginal supply
cost (≈1061).  Region A's boundary dual is its true marginal export cost (cheap
wind) — and THAT is the correct Benders cut slope (verified by finite
difference to ~1e-9 here).  The severed subproblem cannot see downstream
scarcity (the virtual far-node absorbs flow for free), which is exactly the
intended Benders separation.  Hence this module's gates:

* IMPORTER C — region lh2_C block dual == MONOLITH lh2_C block dual
  (rtol=1e-6), the design's PRIMARY gate.
* EXPORTER A — reduced cost == one-sided finite-difference cut slope
  (the mechanism-validating gate; the monolith-match is NOT asserted because
  it is mathematically inappropriate for the exporter — see above).
* BOTH — reduced cost == region-own block dual × unitsize × (slope|1) ×
  block_step_duration, with the documented sign (importer negative, exporter
  positive ×slope=1/0.95), locking the sign + efficiency-side convention.

Units: ``v_flow`` is normalised by ``p_unitsize`` (1000 for the pipes); the
daily LH2 boundary nodes are balanced on a 24h block, so a one-unit increment
of the normalised forward ``v_flow`` injects ``unitsize × block_step`` MWh.
``p_step_duration`` is uniformly 1.0 in this fixture; the block aggregation
weight (24) lives in the block-aware ``nodeBalanceBlock_eq`` and is recovered
empirically here as ``BLOCK_STEP``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high import Param, Problem, WarmProblem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars import _region_filter
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)

# Raw HiGHS duals are in objective-scaled units; the user-facing price is
# ``raw_dual × (−_INV_SCALE_THE_OBJECTIVE / inflation[d])``.  Here we only
# ever COMPARE region duals to monolith duals (both raw), so the 1e6 cancels;
# we keep the constant only for the reduced-cost ↔ node-dual mapping where the
# basis is identical and the factor cancels too.  inflation[y2030] = 1.0.
_PIPE_UNITSIZE = 1000.0
_SLOPE = 1.0 / 0.95               # p_slope = 1/efficiency for the pipes
_BLOCK_STEP = 24.0                # daily LH2 block aggregation weight
_CAP = 50.0                       # test-local half-flow capacity (≫ max f*)
_RTOL = 1e-6
_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Fixtures (mirror the Phase 0 module: build from JSON via the session DB
# fixture; never read a checked-in .sqlite).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ti_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )


@pytest.fixture(scope="module")
def ti_data(ti_workdir):
    return load_flextool(ti_workdir)


@pytest.fixture(scope="module")
def monolith(ti_data):
    """Solve the whole undecomposed fixture once; expose the solution and
    the parsed boundary block-node duals + forward flows."""
    pb = Problem()
    build_flextool(pb, ti_data)
    sol = pb.solve()
    assert sol.optimal, "monolith solve not optimal"
    return sol


# ---------------------------------------------------------------------------
# Small local helpers.
# ---------------------------------------------------------------------------


def _forward_flow(sol, p: str, source: str, sink: str) -> pl.DataFrame:
    """Forward ``v_flow`` for one arc, sorted by ``(d, t)`` — one row per
    ``(d, t)`` cell on the arc's grid."""
    return (
        sol.value("v_flow")
        .filter(
            (pl.col("p") == p)
            & (pl.col("source") == source)
            & (pl.col("sink") == sink)
        )
        .sort("d", "t")
    )


def _parse_block_duals(sol) -> pl.DataFrame:
    """``constraint_dual('nodeBalanceBlock_eq')`` returns ``(key, dual)``
    with ``key = 'n,d,b_first'`` (the constraint's ``over`` axis order).
    Parse it back into ``(n, d, b_first, dual)``."""
    bd = sol.constraint_dual("nodeBalanceBlock_eq")
    rows = []
    for r in bd.iter_rows(named=True):
        n, d, b_first = r["key"].split(",")
        rows.append({"n": n, "d": d, "b_first": b_first, "dual": r["dual"]})
    return pl.DataFrame(rows)


def _node_block_dual(block_duals: pl.DataFrame, node: str) -> float:
    """The (single, block-constant) non-zero raw block dual at ``node``.

    The block dual lives only on the block's ``b_first`` row (0 elsewhere) and
    is constant across the fixture's two daily blocks; assert that and return
    the shared value."""
    sub = block_duals.filter(
        (pl.col("n") == node) & (pl.col("dual").abs() > 1e-12)
    )
    assert sub.height > 0, f"no non-zero block dual at {node!r}"
    vals = sub["dual"].to_numpy()
    assert np.allclose(vals, vals[0], rtol=_RTOL, atol=_ATOL), (
        f"{node!r} block dual not block-constant: {vals.tolist()}"
    )
    return float(vals[0])


def _half_flow(split, side: str, original_p: str, original_source: str,
               original_sink: str) -> "_region_filter.HalfFlow":
    for hf in split.half_flows:
        if (hf.side == side and hf.original_p == original_p
                and hf.original_source == original_source
                and hf.original_sink == original_sink):
            return hf
    raise AssertionError(
        f"no {side} half-flow for "
        f"({original_p}, {original_source}, {original_sink}) in "
        f"region {split.region!r}"
    )


def _patch_halfflow_cap(split, hf, cap: float, periods: list[str]) -> None:
    """Give the virtual half-flow ``p_flow_upper_existing = cap`` per period.

    The splitter inherits the ORIGINAL arc's existing cap; for the greenfield
    pipes that is 0, which would bound a positive pin to 0 (infeasible).  We
    overwrite the virtual arc's rows with ``cap`` (strictly ≫ max f*, asserted
    by the caller) so the ``maxFlow`` row stays slack and cannot leak into the
    dual.  Test-local: production decomposition code is untouched."""
    param = split.data.p_flow_upper_existing
    assert param is not None, "p_flow_upper_existing missing on split data"
    new_rows = pl.DataFrame(
        [
            {
                "p": hf.virtual_p,
                "source": hf.virtual_arc_source,
                "sink": hf.virtual_arc_sink,
                "d": d,
                "value": float(cap),
            }
            for d in periods
        ]
    )
    kept = param.frame.filter(
        ~(
            (pl.col("p") == hf.virtual_p)
            & (pl.col("source") == hf.virtual_arc_source)
            & (pl.col("sink") == hf.virtual_arc_sink)
        )
    )
    new_rows = new_rows.with_columns(
        [pl.col(c).cast(kept.schema[c]) for c in ("p", "source", "sink", "d")]
    )
    split.data.p_flow_upper_existing = Param(
        param.dims, pl.concat([kept, new_rows])
    )


def _arc_rows(var, hf) -> pl.DataFrame:
    """The ``v_flow`` frame rows for the half-flow's virtual arc, sorted by
    ``(d, t)`` — gives the col_ids and dim-tuples in a stable order."""
    return (
        var.frame.filter(
            (pl.col("p") == hf.virtual_p)
            & (pl.col("source") == hf.virtual_arc_source)
            & (pl.col("sink") == hf.virtual_arc_sink)
        )
        .sort("d", "t")
    )


def _dim_tuples(var, rows: pl.DataFrame) -> list[tuple]:
    return [tuple(r) for r in rows.select(*var.dims).iter_rows()]


def _pinned_region(ti_data, region: str, fwd_arc, monolith_forward):
    """Build region ``region``, patch + pin the forward half-flow of
    ``fwd_arc`` to the monolith forward flow, pin the reverse to 0, solve.

    ``fwd_arc`` = ``(original_p, original_source, original_sink)``.

    Returns ``(sol, var, pin_col_ids, pin_dim_tuples, fwd_rows)`` where the
    pin col_ids index ``sol.col_dual`` directly.
    """
    _de = getattr(ti_data, "_axis_enums", None)
    token = None
    if _de is not None and _de != get_global_axis_enums():
        token = set_global_axis_enums(_de)
    try:
        splits = _region_filter.split(
            ti_data, regions=["region_A", "region_B", "region_C"]
        )
        split = next(s for s in splits if s.region == region)
        op, osrc, osnk = fwd_arc
        # Forward half-flow: export in the source region, import in the sink.
        # Region A owns lh2_A → export; region C owns lh2_C → import.
        fwd_side = "export" if region == "region_A" else "import"
        rev_side = "import" if fwd_side == "export" else "export"
        hf_fwd = _half_flow(split, fwd_side, op, osrc, osnk)
        hf_rev = _half_flow(split, rev_side, op, osnk, osrc)

        periods = ti_data.p_inflation_op.frame["d"].to_list()
        f_max = float(monolith_forward["value"].max())
        assert _CAP > 20.0 * f_max, (
            f"capacity patch {_CAP} not ≫ max f* {f_max}; the maxFlow row "
            f"could go binding and leak into the dual"
        )
        _patch_halfflow_cap(split, hf_fwd, _CAP, periods)

        pb = Problem()
        build_flextool(pb, split.data)
        wp = WarmProblem(pb)
        wp.solve()  # initial build (fix_cols requires a built model)

        var = pb._vars["v_flow"]
        fwd_rows = _arc_rows(var, hf_fwd)
        fwd_dt = _dim_tuples(var, fwd_rows)
        # Align the pin values to fwd_rows order by (d, t).
        fmap = {
            (r["d"], r["t"]): r["value"]
            for r in monolith_forward.iter_rows(named=True)
        }
        pin_vals = np.array(
            [fmap[(r["d"], r["t"])] for r in fwd_rows.iter_rows(named=True)],
            dtype=np.float64,
        )
        wp.fix_cols("v_flow", fwd_dt, pin_vals)

        rev_rows = _arc_rows(var, hf_rev)
        rev_dt = _dim_tuples(var, rev_rows)
        wp.fix_cols("v_flow", rev_dt, np.zeros(len(rev_dt), dtype=np.float64))

        sol = wp.solve()
        pin_col_ids = fwd_rows["col_id"].to_numpy()
        return sol, var, pin_col_ids, fwd_dt, fwd_rows, wp, pin_vals
    finally:
        if token is not None:
            reset_global_axis_enums(token)


def _pinned_reduced_cost(sol, pin_col_ids: np.ndarray) -> float:
    """The (single, block-constant) non-zero reduced cost of the pinned
    forward column.  Flow is concentrated on the two block-first cells, so
    the reduced cost is non-zero only there and identical across blocks."""
    rc = sol.col_dual[pin_col_ids]
    nz = rc[np.abs(rc) > 1e-6]
    assert nz.size > 0, "pinned column reduced cost is all ~0 (pin not binding)"
    assert np.allclose(nz, nz[0], rtol=_RTOL, atol=_ATOL), (
        f"pinned reduced cost not block-constant: {nz.tolist()}"
    )
    return float(nz[0])


# ---------------------------------------------------------------------------
# IMPORTER region C — the design's PRIMARY gate (region == monolith dual).
# ---------------------------------------------------------------------------


def test_importer_region_dual_matches_monolith(ti_data, monolith) -> None:
    """Region C's lh2_C boundary block dual equals the MONOLITH's lh2_C block
    dual (rtol=1e-6) when C's import is pinned to the monolith f*.  This is
    the design's PRIMARY acceptance gate: the importer's local marginal value
    of energy at its boundary node is exactly the monolith's."""
    mono_duals = _parse_block_duals(monolith)
    mono_price_c = _node_block_dual(mono_duals, "lh2_C")

    f_bc = _forward_flow(monolith, "pipe_BC", "lh2_B", "lh2_C")
    sol, var, pin_ids, _dt, fwd_rows, _wp, pin_vals = _pinned_region(
        ti_data, "region_C", ("pipe_BC", "lh2_B", "lh2_C"), f_bc
    )
    assert sol.optimal, "pinned region C not optimal"

    # Canary: the pinned primal equals f* exactly.
    primal = sol.value("v_flow")
    pinned = primal.filter(pl.col("p") == fwd_rows["p"][0]).sort("d", "t")
    assert np.allclose(
        pinned["value"].to_numpy(), pin_vals, rtol=_RTOL, atol=_ATOL
    ), "pinned region-C primal did not equal f*"

    region_duals = _parse_block_duals(sol)
    region_price_c = _node_block_dual(region_duals, "lh2_C")

    rc = _pinned_reduced_cost(sol, pin_ids)

    if not np.isclose(region_price_c, mono_price_c, rtol=_RTOL, atol=_ATOL):
        pytest.fail(
            "IMPORTER C dual mismatch:\n"
            f"  monolith lh2_C block dual = {mono_price_c}\n"
            f"  region   lh2_C block dual = {region_price_c}\n"
            f"  pinned reduced cost       = {rc}\n"
            f"  f* (sum/max)              = {float(f_bc['value'].sum())} / "
            f"{float(f_bc['value'].max())}\n"
            f"  cap used                  = {_CAP}\n"
            f"  rtol                      = {_RTOL}"
        )

    # Reduced cost ↔ region's own block dual (importer: sink side, no slope):
    #   rc = node_dual_raw × unitsize × block_step   (negative — cost falls).
    expected_rc = region_price_c * _PIPE_UNITSIZE * _BLOCK_STEP
    assert rc < 0.0, f"importer reduced cost should be negative, got {rc}"
    assert np.isclose(rc, expected_rc, rtol=_RTOL, atol=_ATOL), (
        f"importer reduced cost {rc} != node_dual×unitsize×block_step "
        f"{expected_rc}"
    )


def test_importer_reduced_cost_is_true_cut_slope(ti_data, monolith) -> None:
    """The pinned column's reduced cost equals the one-sided finite-difference
    ∂cost_C/∂f̄ on the binding side — i.e. it is the EXACT Benders cut slope."""
    f_bc = _forward_flow(monolith, "pipe_BC", "lh2_B", "lh2_C")
    sol, var, pin_ids, fwd_dt, fwd_rows, wp, pin_vals = _pinned_region(
        ti_data, "region_C", ("pipe_BC", "lh2_B", "lh2_C"), f_bc
    )
    assert sol.optimal
    rc = _pinned_reduced_cost(sol, pin_ids)
    obj0 = sol.obj

    # Perturb the first binding cell by +eps; the importer's binding side is
    # the +direction (importing more reduces cost) — but to read the slope AT
    # the optimum we use a small two-sided-safe step on the binding side.
    nz = np.where(np.abs(sol.col_dual[pin_ids]) > 1e-6)[0]
    i = int(nz[0])
    eps = 1e-4
    perturbed = pin_vals.copy()
    perturbed[i] += eps
    wp.fix_cols("v_flow", fwd_dt, perturbed)
    sol1 = wp.solve()
    assert sol1.optimal
    fd = (sol1.obj - obj0) / eps
    assert np.isclose(fd, rc, rtol=1e-5, atol=1.0), (
        f"importer finite-difference cut slope {fd} != reduced cost {rc}"
    )


# ---------------------------------------------------------------------------
# EXPORTER region A — sign + efficiency-side lock.
# ---------------------------------------------------------------------------


def test_exporter_reduced_cost_sign_and_efficiency(ti_data, monolith) -> None:
    """Region A's export pin reduced cost is POSITIVE (exporting costs A its
    marginal generation) and equals ``+π_{lh2_A,region} × unitsize × slope ×
    block_step`` — locking the sign + the SOURCE-side efficiency convention
    (slope = 1/0.95).

    NOTE (Phase 1 finding): the exporter's REGION boundary dual does NOT equal
    the MONOLITH's lh2_A dual — the monolith price is the delivered-to-C value
    back-propagated through the pipe efficiency chain, while the region dual is
    A's local marginal supply cost (the correct cut slope).  We therefore do
    NOT assert region == monolith here; the monolith-match is the importer's
    gate.  The exporter gate is the reduced-cost ↔ region-dual identity plus
    the finite-difference slope test below.
    """
    f_ab = _forward_flow(monolith, "pipe_AB", "lh2_A", "lh2_B")
    sol, var, pin_ids, _dt, fwd_rows, _wp, pin_vals = _pinned_region(
        ti_data, "region_A", ("pipe_AB", "lh2_A", "lh2_B"), f_ab
    )
    assert sol.optimal, "pinned region A not optimal"

    primal = sol.value("v_flow")
    pinned = primal.filter(pl.col("p") == fwd_rows["p"][0]).sort("d", "t")
    assert np.allclose(
        pinned["value"].to_numpy(), pin_vals, rtol=_RTOL, atol=_ATOL
    ), "pinned region-A primal did not equal f*"

    region_duals = _parse_block_duals(sol)
    region_price_a = _node_block_dual(region_duals, "lh2_A")
    rc = _pinned_reduced_cost(sol, pin_ids)

    # Exporter: source side loses v_flow × unitsize × slope (efficiency on the
    # SOURCE side), so rc = −node_dual_raw × unitsize × slope × block_step.
    # node_dual_raw is negative (demand-short sign convention) ⇒ rc positive.
    expected_rc = -region_price_a * _PIPE_UNITSIZE * _SLOPE * _BLOCK_STEP
    assert rc > 0.0, (
        f"exporter reduced cost should be POSITIVE (exporting costs A), "
        f"got {rc}"
    )
    assert np.isclose(rc, expected_rc, rtol=_RTOL, atol=_ATOL), (
        "exporter reduced cost vs node-dual×unitsize×slope×block_step "
        f"mismatch:\n  rc       = {rc}\n  expected = {expected_rc}\n"
        f"  region lh2_A raw dual = {region_price_a}\n"
        f"  slope (1/0.95)        = {_SLOPE}"
    )

    # Document the (intentional) divergence from the monolith for the record.
    mono_price_a = _node_block_dual(_parse_block_duals(monolith), "lh2_A")
    assert not np.isclose(region_price_a, mono_price_a, rtol=1e-3), (
        "UNEXPECTED: exporter region lh2_A dual matched the monolith — the "
        "Phase 1 finding (region = local supply cost ≠ monolith delivered "
        f"value) no longer holds (region={region_price_a}, "
        f"monolith={mono_price_a}); revisit the design before Phase 2"
    )


def test_exporter_reduced_cost_is_true_cut_slope(ti_data, monolith) -> None:
    """The export pin reduced cost equals the one-sided finite-difference
    ∂cost_A/∂f̄ on the binding side (−eps; +eps crosses a wind-exhaustion
    kink).  This is the EXACT exporter cut slope the master will use."""
    f_ab = _forward_flow(monolith, "pipe_AB", "lh2_A", "lh2_B")
    sol, var, pin_ids, fwd_dt, fwd_rows, wp, pin_vals = _pinned_region(
        ti_data, "region_A", ("pipe_AB", "lh2_A", "lh2_B"), f_ab
    )
    assert sol.optimal
    rc = _pinned_reduced_cost(sol, pin_ids)
    obj0 = sol.obj

    nz = np.where(np.abs(sol.col_dual[pin_ids]) > 1e-6)[0]
    i = int(nz[0])
    eps = 1e-4
    # Binding side for the exporter is the −direction (exporting LESS frees A's
    # constrained cheap wind; +eps would cross the kink to coal/slack).
    perturbed = pin_vals.copy()
    perturbed[i] -= eps
    wp.fix_cols("v_flow", fwd_dt, perturbed)
    sol1 = wp.solve()
    assert sol1.optimal
    fd = (sol1.obj - obj0) / (-eps)
    assert np.isclose(fd, rc, rtol=1e-5, atol=1.0), (
        f"exporter finite-difference cut slope {fd} != reduced cost {rc}"
    )
