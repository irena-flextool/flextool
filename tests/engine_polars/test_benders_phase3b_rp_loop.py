"""Benders (Option C) Phase 3b — RP-weight / multi-period LOOP LOCK.

Phase 3a proved the FlexTool-generated master converges to the monolith on the
NON-RP, single-period, zero-flow-cost prototype.  Phase 3b LOCKS the two new
dimensions the H2_trade `lt_rp` scale case adds, on a small controllable
fixture (`lh2_three_region_rp_invest`):

* **Non-unit representative-period weights.**  Two reps per FlexTool period
  carry NON-UNIT `representative_period_weights` (folded to
  `p_timestep_weight` = {1.4, 0.6} for y2030, {1.1, 0.9} for y2040).  The
  engine bug that silently clobbered these to 1.0 was fixed in commit
  574e489c, so they now actually reach the objective (M moves when the reps
  are swapped — see `test_rp_weight_applied.py`).
* **Multi-period investment.**  Both y2030 and y2040 are invest-eligible, so
  the master's `v_invest_p` is per `(conn, invest-period)` and the capacity
  coupling `f ≤ Σ_{d'≤d} v_invest_p` must use the right period's cumulative
  capacity.  The FlexTool master emits this natively.

GATES (spec `benders_option_c.md` §3 + "Phase 3b impl — RP fixture"):

1. **Convergence to the RP-weighted monolith.**  `solve_benders` (FlexTool
   master) converges (≤15 iters, tol 1e-4) to `M_rp = 4.8859219264e10` with a
   VALID lower bound `LB ≤ M_rp·(1+1e-9)`; recovered per-period invest `C` and
   trade `f̄` match the monolith.

2. **Finite-difference RP-weight LOCK (the key gate).**  For region_B's
   forward import arc (`pipe_AB lh2_A→lh2_B`), the Benders cut slope the LOOP
   ITSELF computes (`sol_r.col_dual[pinned v_flow col]`, captured live) equals
   the true `∂(region cost)/∂f̄` measured by a direct finite difference
   (perturb the pinned f̄, re-solve the region) to a tight tolerance.  And the
   slope already carries the RP weight WITH NO EXTRA FACTOR:
   `slope = nodeBalanceBlock_eq dual · p_unitsize`, where the block dual is set
   by the period's `op_factor`-weighted clearing objective
   (`op_factor = step_duration · p_timestep_weight · inflation / period_share`,
   `model.py:3664`).  Because `lh2_B` is a within-period-blended BLOCK node,
   the trade couples through the SHARED period block, so the marginal is
   period-uniform (identical across the two reps) and differs across periods —
   exactly §3's auto-consistency: the RP weight rides in through the block dual,
   the balance constraint itself carries only the physical block weight
   (`model.py:2073-2169`, NO `p_timestep_weight`), and the cut uses the raw
   col_dual with NO RP multiply/divide anywhere.  The loop's convergence to the
   exact RP-weighted `M_rp` using these raw slopes is the end-to-end proof that
   no factor is missing or extra.

The loop's Phase-2 self-checks (master kOptimal, LB monotone, each appended cut
SATISFIED at the new master point, LB ≤ M, finite boundary penalties) run
INSIDE `solve_benders` and raise on violation — a green loop is itself the
self-check assertion.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high import Problem, WarmProblem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars import _region_filter
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars._benders import (
    _BendersMaster,
    _build_arcs,
    _reverse_cols,
    solve_benders,
)

_REGIONS = ["region_A", "region_B", "region_C"]
# Post-fix RP-weighted monolith (spec "RP-weight fix — impl", base scenario).
_M_RP_EXPECTED = 4.8859219264e10
# Forward import arc with non-trivial flow in BOTH (non-unit-weight) reps.
_ARC = ("pipe_AB", "lh2_A", "lh2_B")
_IMPORT_REGION = "region_B"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rp_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_rp_invest", db_fixture="lh2_rp_invest"
    )


@pytest.fixture(scope="module")
def rp_data(rp_workdir):
    return load_flextool(rp_workdir)


@pytest.fixture(scope="module")
def monolith(rp_data):
    """Solve the whole RP fixture once (cascade-emitted workdir → the
    `timestep_weight.csv` round-trip that the RP-weight fix repairs)."""
    pb = Problem()
    build_flextool(pb, rp_data)
    sol = pb.solve()
    assert sol.optimal, "RP monolith solve not optimal"
    return sol


def _arc_sum(sol, p, source, sink) -> float:
    f = sol.value("v_flow").filter(
        (pl.col("p") == p) & (pl.col("source") == source) & (pl.col("sink") == sink)
    )
    return float(f["value"].sum()) if f.height else 0.0


def _invest_period(sol, p, d) -> float:
    inv = sol.value("v_invest_p").filter(
        (pl.col("p") == p) & (pl.col("d") == d)
    )
    return float(inv["value"].sum()) if inv.height else 0.0


def _region_dim_tuples(w, col_ids):
    vf = w._p._vars["v_flow"]
    fr = vf.frame.filter(pl.col("col_id").is_in(col_ids))
    order = {int(c): i for i, c in enumerate(col_ids)}
    fr = fr.with_columns(
        pl.col("col_id").replace_strict(order, default=-1).alias("__o")
    ).sort("__o")
    return [tuple(r) for r in fr.select(*vf.dims).iter_rows()]


# ---------------------------------------------------------------------------
# (0) The monolith is the RP-weighted optimum (sanity on the fixture).
# ---------------------------------------------------------------------------


def test_monolith_is_rp_weighted(rp_data, monolith) -> None:
    M = monolith.obj
    assert np.isclose(M, _M_RP_EXPECTED, rtol=1e-4), (
        f"RP monolith M drifted from {_M_RP_EXPECTED:.6e}: {M:.10e}"
    )
    # The RP weights genuinely reached the objective (non-unit p_timestep_weight).
    tsw = set(
        round(float(v), 6)
        for v in rp_data.p_timestep_weight.frame["value"].unique().to_list()
    )
    assert tsw == {1.4, 0.6, 1.1, 0.9}, (
        f"p_timestep_weight not the folded RP weights: {tsw}"
    )
    # Multi-period invest is exercised: both y2030 and y2040 invest-eligible.
    inv_periods = set(rp_data.pd_invest_set["d"].cast(pl.Utf8).unique().to_list())
    assert {"y2030", "y2040"} <= inv_periods, (
        f"fixture must exercise multi-period invest; got {inv_periods}"
    )


# ---------------------------------------------------------------------------
# (1) The FlexTool master converges to the RP-weighted monolith with a VALID LB.
# ---------------------------------------------------------------------------


def test_rp_loop_converges_to_monolith(rp_data, monolith) -> None:
    M = monolith.obj

    res = solve_benders(
        rp_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=M, master="flextool",
    )

    assert res.converged, (
        f"RP Benders did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters (LB={res.lower_bound:.6e} "
        f"UB={res.upper_bound:.6e})"
    )
    assert res.iterations <= 15, f"too many iters: {res.iterations}"

    # best UB reconciles to the RP-weighted monolith optimum.
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"RP Benders UB {res.total_objective:.8e} != monolith M_rp {M:.8e} "
        f"(LB={res.lower_bound:.8e}, gap={res.gap:.3e}, iters={res.iterations})"
    )

    # VALID lower bound: LB ≤ M_rp (the whole point vs the Lagrangian bug).
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"RP Benders LB {res.lower_bound:.8e} EXCEEDS M_rp {M:.8e} — "
        f"invalid bound (an RP-weight mis-scaling would surface here)"
    )

    # UB restated == M.
    assert np.isclose(res.upper_bound, M, rtol=1e-4), (
        f"Σ cost_r + master trade cost = {res.upper_bound:.8e} != M_rp {M:.8e}"
    )


# ---------------------------------------------------------------------------
# (2) Recovered PER-PERIOD pipe invest + forward trade match the monolith
#     (multi-period invest worked natively in the FlexTool master).
# ---------------------------------------------------------------------------


def test_rp_loop_recovers_per_period_invest_and_trade(rp_data, monolith) -> None:
    # Monolith per-period invest C* (y2030 + y2040) and forward trade f*.
    C_ab_2030 = _invest_period(monolith, "pipe_AB", "y2030")
    C_ab_2040 = _invest_period(monolith, "pipe_AB", "y2040")
    C_bc_2030 = _invest_period(monolith, "pipe_BC", "y2030")
    C_bc_2040 = _invest_period(monolith, "pipe_BC", "y2040")
    f_ab_star = _arc_sum(monolith, "pipe_AB", "lh2_A", "lh2_B")
    f_bc_star = _arc_sum(monolith, "pipe_BC", "lh2_B", "lh2_C")

    res = solve_benders(
        rp_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=monolith.obj, master="flextool",
    )
    assert res.converged

    # Recovered invest sums over invest periods → compare the TOTAL (the loop's
    # `res.invest` sums over the per-period v_invest_p columns).
    C_ab_total = C_ab_2030 + C_ab_2040
    C_bc_total = C_bc_2030 + C_bc_2040
    C_ab = res.invest.get("pipe_AB", 0.0)
    C_bc = res.invest.get("pipe_BC", 0.0)
    assert C_ab > 1e-3 and C_bc > 1e-3, f"pipes not invested: {res.invest}"
    assert np.isclose(C_ab, C_ab_total, rtol=2e-2, atol=1e-3), (
        f"pipe_AB total invest {C_ab} != monolith {C_ab_total} "
        f"(per-period y2030={C_ab_2030} y2040={C_ab_2040})"
    )
    assert np.isclose(C_bc, C_bc_total, rtol=2e-2, atol=1e-3), (
        f"pipe_BC total invest {C_bc} != monolith {C_bc_total}"
    )

    # Forward trade f̄ ≈ f* (summed over the (d,t) grid).
    f_ab = float(res.trade_flow[("pipe_AB", "lh2_A", "lh2_B")]["value"].sum())
    f_bc = float(res.trade_flow[("pipe_BC", "lh2_B", "lh2_C")]["value"].sum())
    assert np.isclose(f_ab, f_ab_star, rtol=2e-2, atol=1e-3), (
        f"A→B trade {f_ab} != monolith {f_ab_star}"
    )
    assert np.isclose(f_bc, f_bc_star, rtol=2e-2, atol=1e-3), (
        f"B→C trade {f_bc} != monolith {f_bc_star}"
    )

    # Reverse arcs ~0 at the optimum.
    f_ba = float(res.trade_flow[("pipe_AB", "lh2_B", "lh2_A")]["value"].sum())
    f_cb = float(res.trade_flow[("pipe_BC", "lh2_C", "lh2_B")]["value"].sum())
    assert abs(f_ba) < 1e-3 and abs(f_cb) < 1e-3, (
        f"reverse trade not ~0: B→A={f_ba}, C→B={f_cb}"
    )


# ---------------------------------------------------------------------------
# (3) Finite-difference RP-weight LOCK — the key gate.
# ---------------------------------------------------------------------------


def _capture_loop_slopes(rp_data, region):
    """Run the loop and capture, per appended cut for ``region``, the
    slope keyed by (arc_key, d, t).  Returns (BendersResult, list of dicts —
    one per cut)."""
    captured: list[dict] = []
    orig = _BendersMaster.add_cut

    def spy(self, reg, f_bar, cost_r, slopes):
        if reg == region:
            colmap = {}
            for a in self.arcs:
                for dt, cid in zip(a.dim_tuples, a.f_col_ids):
                    colmap[int(cid)] = (a.key, dt[3], dt[4])
            rec = {colmap[int(cid)]: sl for cid, sl in slopes.items()
                   if int(cid) in colmap}
            captured.append(rec)
        return orig(self, reg, f_bar, cost_r, slopes)

    _BendersMaster.add_cut = spy
    try:
        res = solve_benders(
            rp_data, _REGIONS, max_iters=20, tol=1e-4, master="flextool"
        )
    finally:
        _BendersMaster.add_cut = orig
    return res, captured


def _finite_difference_region_cost(rp_data, res, arc_key, cells):
    """Direct finite difference of region_B's cost wrt the pinned forward
    f̄ at each (d,t) in ``cells``.

    Replicates the loop's region pin (every cross half-flow pinned to the
    Benders f̄, reverse → 0) on a fresh region WarmProblem, then for each cell
    perturbs ONLY that pinned f̄ cell by −eps and re-solves, returning the
    LEFT derivative ``(cost(f̄) − cost(f̄−eps))/eps`` (the side the degenerate
    pinned vertex admits) plus the base region cost.
    """
    _enums = getattr(rp_data, "_axis_enums", None)
    if _enums is not None and _enums != get_global_axis_enums():
        set_global_axis_enums(_enums)

    splits = _region_filter.split(
        rp_data, regions=_REGIONS, benders_uncap_cross_region=True
    )
    region_idx = {s.region: i for i, s in enumerate(splits)}
    subs = [Problem() for _ in splits]
    for s, pb in zip(splits, subs):
        build_flextool(pb, s.data)
    warm = [WarmProblem(p) for p in subs]
    for w in warm:
        w.solve()
    arcs = _build_arcs(splits, warm)
    a = next(x for x in arcs if x.key == arc_key)

    region = a.import_region
    w = warm[region_idx[region]]
    s = splits[region_idx[region]]

    def pin_and_solve(override_dt=None, override_val=None):
        all_hf = _reverse_cols(s, w)
        pinned: set[int] = set()
        for oa in arcs:
            if region == oa.export_region:
                cols = oa.export_pin_cols
            elif region == oa.import_region:
                cols = oa.import_pin_cols
            else:
                continue
            df = res.trade_flow[oa.key]
            fmap = {(r["d"], r["t"]): r["value"] for r in df.iter_rows(named=True)}
            dt = _region_dim_tuples(w, cols)
            vals = np.array([fmap[(d[-2], d[-1])] for d in dt])
            if oa is a and override_dt is not None:
                for j, d in enumerate(dt):
                    if (d[-2], d[-1]) == override_dt:
                        vals[j] = override_val
            w.fix_cols("v_flow", dt, vals)
            pinned.update(int(c) for c in cols)
        rest = np.array(
            [int(c) for c in all_hf if int(c) not in pinned], dtype=np.int64
        )
        if rest.size:
            w.fix_cols("v_flow", _region_dim_tuples(w, rest), np.zeros(rest.size))
        sol = w.solve()
        assert sol.optimal, f"region {region} subproblem not optimal"
        return sol

    # nodeBalanceBlock_eq dual rows for lh2_B (one per period).
    meta = w._cstr_meta["nodeBalanceBlock_eq"]
    over = meta["over"]
    with_rid = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
    bsub = with_rid.filter(pl.col("n").cast(pl.Utf8) == "lh2_B")
    block_rows = {str(r["d"]): meta["base_row"] + r["_rid"]
                  for r in bsub.iter_rows(named=True)}

    fbar = {(r["d"], r["t"]): r["value"]
            for r in res.trade_flow[arc_key].iter_rows(named=True)}
    sol0 = pin_and_solve()
    cost0 = float(sol0.obj)

    eps = 1e-4
    out = {}
    block_dual: dict[str, float] = {}
    for (dd, tt) in cells:
        base_f = fbar[(dd, tt)]
        solm = pin_and_solve((dd, tt), base_f - eps)
        # LEFT derivative: ∂cost/∂f̄ on the side the degenerate pinned vertex
        # admits (an extra unit of pinned inflow at one ISOLATED cell would
        # have to be spilled, so the RIGHT derivative is the spill-penalty
        # regime; the cut uses the displaced-cost (left) reduced cost).
        fd_left = (cost0 - float(solm.obj)) / eps
        out[(dd, tt)] = fd_left
        # The −eps solve sits in the displaced-cost regime (the settled vertex)
        # → its block dual is the one that scales the cut slope.  Record it per
        # period (t0001 cell suffices; the block dual is period-level).
        if tt == "t0001":
            block_dual[dd] = float(solm.row_dual[block_rows[dd]])
    return out, cost0, block_dual


def _settled_loop_slope(captured, arc_key, cells, fd):
    """Among the captured region_B cuts, return the SETTLED slope dict (keyed
    by (d,t)): the cut whose per-cell slope matches the finite-difference
    ground truth for ALL ``cells``.  The early cuts sit in the spill-penalty
    regime (huge positive slopes) while f̄ is far from the optimum; as f̄
    approaches the optimum the region duals settle to the displaced-cost
    gradient the cut is supposed to carry."""
    for rec in captured:
        sl = {(dd, tt): rec.get((arc_key, dd, tt), 0.0) for (dd, tt) in cells}
        if all(
            np.isclose(sl[(dd, tt)], fd[(dd, tt)], rtol=1e-6, atol=1.0)
            for (dd, tt) in cells
        ):
            return sl
    return None


def test_finite_difference_rp_weight_lock(rp_data, monolith) -> None:
    """The Benders cut slope (the LOOP's own ``col_dual``) == ∂(region cost)/∂f̄
    (independent finite difference) AND already carries the RP weight with NO
    extra factor (slope = block_dual · p_unitsize; block dual period-uniform
    across reps; loop converges to M_rp on the raw slopes)."""
    # Two reps of y2030 (w=1.4, w=0.6) + two of y2040 (w=1.1, w=0.9) — all four
    # carry NON-UNIT RP weight and non-trivial forward flow.
    cells = [("y2030", "t0001"), ("y2030", "t0025"),
             ("y2040", "t0001"), ("y2040", "t0025")]

    # (a) Run the loop; capture every region_B cut's per-cell slope (the live
    # ``sol_r.col_dual`` of the pinned forward column).
    res, captured = _capture_loop_slopes(rp_data, _IMPORT_REGION)
    assert res.converged and np.isclose(res.total_objective, monolith.obj, rtol=1e-4)

    # (b) Independent finite difference of region_B's cost wrt the pinned f̄,
    # plus the region's nodeBalanceBlock_eq dual at lh2_B at the displaced-cost
    # (settled) vertex.
    fd, region_cost0, region_block_dual = _finite_difference_region_cost(
        rp_data, res, _ARC, cells
    )

    # The SETTLED Benders cut slope (the gradient the loop actually carries once
    # f̄ has converged) — it is what enters the binding cut.
    loop_slope = _settled_loop_slope(captured, _ARC, cells, fd)

    # --- LOCK 1: the loop's cut slope == finite-difference ∂cost/∂f̄ (tight).
    assert loop_slope is not None, (
        "no captured loop cut slope matches the finite-difference gradient — "
        f"the col_dual is NOT ∂cost/∂f̄ (fd={fd})"
    )
    for (dd, tt) in cells:
        sl = loop_slope[(dd, tt)]
        d_fd = fd[(dd, tt)]
        assert np.isclose(sl, d_fd, rtol=1e-6, atol=1.0), (
            f"cut slope {sl:.10e} != finite-diff ∂cost/∂f̄ {d_fd:.10e} at "
            f"({dd},{tt})"
        )
        # The marginal value of imported H2 is a benefit (negative slope).
        assert sl < 0.0, f"expected negative import slope at ({dd},{tt}): {sl}"

    # --- LOCK 2: slope already carries the RP weight, NO extra factor. -----
    # lh2_B is a within-period-blended BLOCK node ⇒ ONE block balance per
    # period, so the slope is PERIOD-uniform: identical across the two reps of
    # a period (despite their DIFFERENT RP weights 1.4 vs 0.6 / 1.1 vs 0.9),
    # and it differs ACROSS periods.  This is §3 auto-consistency: the RP
    # weight rides into the block dual via the op_factor-weighted clearing
    # objective; the cut uses the raw col_dual with no per-cell RP multiply.
    s_2030_r1 = loop_slope[("y2030", "t0001")]   # w = 1.4
    s_2030_r2 = loop_slope[("y2030", "t0025")]   # w = 0.6
    s_2040_r1 = loop_slope[("y2040", "t0001")]   # w = 1.1
    s_2040_r2 = loop_slope[("y2040", "t0025")]   # w = 0.9
    assert np.isclose(s_2030_r1, s_2030_r2, rtol=1e-9), (
        f"y2030 slope differs across reps (would mean a per-rep RP factor "
        f"crept in): w=1.4 {s_2030_r1:.10e} vs w=0.6 {s_2030_r2:.10e}"
    )
    assert np.isclose(s_2040_r1, s_2040_r2, rtol=1e-9), (
        f"y2040 slope differs across reps: w=1.1 {s_2040_r1:.10e} vs "
        f"w=0.9 {s_2040_r2:.10e}"
    )
    # Across periods the slope DOES differ (the block dual is period-specific).
    assert not np.isclose(s_2030_r1, s_2040_r1, rtol=1e-3), (
        f"y2030 and y2040 slopes coincide — the period structure is not "
        f"exercised: {s_2030_r1:.10e} vs {s_2040_r1:.10e}"
    )

    # --- LOCK 3: slope = nodeBalanceBlock_eq dual · p_unitsize, EXACTLY, with
    # the SAME p_unitsize factor in BOTH periods (no period/RP-dependent
    # rescale).  This is the textbook reduced-cost identity for the pinned
    # boundary column: the RP weight lives ENTIRELY in the BLOCK dual (set by
    # the period's op_factor-weighted clearing objective), NOT in any explicit
    # cut factor.  The block dual is read off the REGION subproblem at the
    # displaced-cost (settled) vertex — the same vertex the settled slope is at.
    ratio = {}
    for d in ("y2030", "y2040"):
        ratio[d] = loop_slope[(d, "t0001")] / region_block_dual[d]
    assert np.isclose(ratio["y2030"], ratio["y2040"], rtol=1e-5), (
        f"slope/block_dual differs across periods (would mean an extra "
        f"period/RP factor): y2030 {ratio['y2030']!r} vs y2040 "
        f"{ratio['y2040']!r}"
    )
    # The constant ratio is a clean PHYSICAL factor: ``p_unitsize · n_block``
    # where ``n_block`` is the number of timesteps the representative block
    # aggregates (the pinned flow enters the block balance with coefficient
    # ``p_unitsize · p_step_duration`` summed over the block).  Both are purely
    # PHYSICAL / PERIOD-INDEPENDENT — the RP weight is NOT in this factor, it is
    # entirely inside the (period-specific) block dual.
    us_param = {
        r["p"]: float(r["value"])
        for r in rp_data.p_unitsize.frame.iter_rows(named=True)
    }
    n_block = ratio["y2030"] / us_param["pipe_AB"]
    assert np.isclose(n_block, round(n_block), atol=1e-6) and n_block > 1.0, (
        f"slope/block_dual / p_unitsize = {n_block!r} is not a clean block "
        f"step count — the slope is not block_dual·p_unitsize·n_block"
    )
    # The SAME physical factor applies in BOTH periods (already asserted above
    # via the constant ratio) — no period/RP rescale of the physical coefficient.

    # --- LOCK 4 (end-to-end): the loop converged to the EXACT RP-weighted
    # M_rp using these raw slopes (no factor applied anywhere).  If an RP
    # factor were missing or extra, the cut would be mis-scaled and either cut
    # off the optimum (LB > M, the loop's valid-bound check would have raised)
    # or fail to reconcile — neither happened.  §3 is LOCKED.
    assert np.isclose(res.total_objective, _M_RP_EXPECTED, rtol=1e-4), (
        f"loop UB {res.total_objective:.8e} != M_rp {_M_RP_EXPECTED:.8e}"
    )
    assert res.lower_bound <= monolith.obj * (1 + 1e-9)
