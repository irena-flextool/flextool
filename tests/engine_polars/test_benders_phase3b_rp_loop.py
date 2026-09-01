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

The fixture was REDESIGNED to be non-degenerate: it runs on a SHARED 4-day
(96h) timeline where y2030 uses days 1-2 and y2040 uses days 3-4 with a
HIGHER (grown +50%) LH2 demand.  Giving each period its own day-pair with
different demand makes the two periods genuinely physically distinct WITHOUT
period-keyed inflow — which (i) makes the per-period invest optimum UNIQUE
(both pipes, both periods; monolith == Benders to ~1e-6, not the old 2% band)
and (ii) makes the finite-difference RP-weight marginal PERIOD-SPECIFIC (the
symmetric single-2-day fixture made it period-uniform).  See
`regen_lh2_three_region._build_rp_invest_overlay`.

GATES (spec `benders_option_c.md` §3 + "Phase 3b impl — RP fixture"):

1. **Convergence to the RP-weighted monolith.**  `solve_benders` (FlexTool
   master) converges (≤15 iters, tol 1e-4) to `M_rp = _M_RP_EXPECTED` with a
   VALID lower bound `LB ≤ M_rp·(1+1e-9)`; recovered per-period invest `C` and
   trade `f̄` match the monolith to a TIGHT tolerance (the UNIQUE optimum).

2. **Finite-difference RP-weight LOCK (the key gate).**  For region_B's
   forward import arc (`pipe_AB lh2_A→lh2_B`), the Benders cut slope the LOOP
   ITSELF computes (`sol_r.col_dual[pinned v_flow col]`, captured live) equals
   the true `∂(region cost)/∂f̄` measured by a direct (two-sided, economic-side)
   finite difference (perturb the pinned f̄, re-solve the region) to a tight
   tolerance.  The slope is genuinely PERIOD-DISTINCT (different demand tier per
   period) and REP-DISTINCT (different wind cost + RP weight per rep), and the
   RP weight rides in through the `op_factor`-weighted clearing objective
   (`op_factor = step_duration · p_timestep_weight · inflation / period_share`,
   `model.py:3664`) with the cut using the raw col_dual — NO RP multiply/divide
   anywhere.  The loop's convergence to the exact RP-weighted `M_rp` using these
   raw slopes is the end-to-end proof that no factor is missing or extra.

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
# RP-weighted monolith optimum of the (non-degenerate, 4-day) base scenario.
# Derived from a real cascade+monolith solve of the regenerated fixture
# (``tests/fixtures/lh2_three_region_rp_invest.json``); see
# ``regen_lh2_three_region._build_rp_invest_overlay``.
_M_RP_EXPECTED = 7.5779590452e9
# Forward import arc with non-trivial flow in BOTH (non-unit-weight) reps.
_ARC = ("pipe_AB", "lh2_A", "lh2_B")
_IMPORT_REGION = "region_B"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rp_workdir_and_provider(
    tmp_path_factory, lh2_rp_invest_db_url, test_solver_config_dir
):
    """Run the full cascade for the RP-invest fixture and return
    ``(workdir, provider)``.

    Production threads the live in-memory sub-solve Provider into every
    re-solve (``_orchestration.py`` ``load_flextool(work_folder, ...,
    provider=_sub_solve_provider)``); the Benders decomposition sub-solves
    do exactly that.  Building the workdir here (rather than via the
    shared session ``scenario_workdir`` factory, which discards the
    Provider after snapshotting to disk) lets the Benders tests exercise
    that SAME in-memory Provider path instead of a bare-Path CSV reload —
    ``keep_solutions=True`` keeps the Provider alive for the reload.
    """
    import shutil
    from urllib.parse import urlparse

    from flextool.engine_polars._orchestration import run_chain_from_db

    parent = tmp_path_factory.mktemp("_root_rp_invest_benders")
    wf = parent / "work_lh2_three_region_rp_invest"
    wf.mkdir()
    steps = run_chain_from_db(
        input_db_url=lh2_rp_invest_db_url,
        scenario_name="lh2_three_region_rp_invest",
        work_folder=wf,
        solver_config_dir=test_solver_config_dir,
        csv_dump=True,
        keep_solutions=True,
    )
    _src = urlparse(lh2_rp_invest_db_url).path
    if len(_src) >= 3 and _src[0] == "/" and _src[2] == ":":
        _src = _src[1:]  # Windows '/C:/...' -> 'C:/...'
    shutil.copy(_src, wf / "tests.sqlite")
    last_step = next(reversed(list(steps.values())))
    provider = getattr(last_step, "flex_data_provider", None)
    assert provider is not None, "cascade did not retain a sub-solve Provider"
    provider.snapshot_processed_inputs(wf)
    return wf, provider


@pytest.fixture(scope="module")
def rp_workdir(rp_workdir_and_provider):
    return rp_workdir_and_provider[0]


@pytest.fixture(scope="module")
def rp_data(rp_workdir_and_provider):
    # Thread the in-memory Provider (production path) rather than a
    # bare-Path CSV reload — see ``rp_workdir_and_provider``.
    wf, provider = rp_workdir_and_provider
    return load_flextool(wf, provider=provider)


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


def _closure_rows(pb) -> int:
    return sum(
        pb.cstr_row_count(n) for n in pb.cstr_names()
        if n.startswith("rp_inter_period_cyclic")
    )


def test_coarse_rp_storage_closure_fires(rp_data) -> None:
    """The coarse ``daily_group`` rp-storage seasonal closure
    (``rp_inter_period_cyclic``) fires with NON-EMPTY rows in BOTH the
    monolith AND every region subproblem — the structure these Benders RP
    tests exist to exercise (a nulled RP label would silently drop it and
    under-constrain the LP; see ``test_rp_label_roundtrip``)."""
    pbm = Problem()
    build_flextool(pbm, rp_data)
    assert _closure_rows(pbm) > 0, (
        "rp_inter_period_cyclic did not fire in the monolith — the coarse "
        "rp-storage seasonal closure is not exercised"
    )

    _enums = getattr(rp_data, "_axis_enums", None)
    if _enums is not None and _enums != get_global_axis_enums():
        set_global_axis_enums(_enums)
    splits = _region_filter.split(
        rp_data, regions=_REGIONS, benders_uncap_cross_region=True
    )
    for s in splits:
        pbr = Problem()
        build_flextool(pbr, s.data)
        assert _closure_rows(pbr) > 0, (
            f"rp_inter_period_cyclic did not fire in region {s.region} "
            f"subproblem — coarse rp storage not exercised there"
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
    # UNIQUE optimum: the 4-day fixture has a strictly-pinned per-period
    # invest (both periods invest a positive, distinct increment), so
    # Benders must recover the SAME total capacity as the monolith to a
    # TIGHT tolerance — the old 2% band existed only to paper over the
    # degenerate equal-cost invest face, which this fixture removed.
    assert np.isclose(C_ab, C_ab_total, rtol=1e-3, atol=1e-4), (
        f"pipe_AB total invest {C_ab} != monolith {C_ab_total} "
        f"(per-period y2030={C_ab_2030} y2040={C_ab_2040})"
    )
    assert np.isclose(C_bc, C_bc_total, rtol=1e-3, atol=1e-4), (
        f"pipe_BC total invest {C_bc} != monolith {C_bc_total} "
        f"(per-period y2030={C_bc_2030} y2040={C_bc_2040})"
    )
    # Multi-period invest is genuinely exercised: BOTH periods build a
    # positive increment of at least one pipe (y2040's grown demand needs
    # more capacity than y2030 alone provides).
    assert C_ab_2040 > 1e-3 or C_bc_2040 > 1e-3, (
        f"y2040 invests nothing — the periods are not distinct: "
        f"pipe_AB y2040={C_ab_2040}, pipe_BC y2040={C_bc_2040}"
    )
    assert C_ab_2030 > 1e-3 and C_bc_2030 > 1e-3, (
        f"y2030 must build the base capacity: pipe_AB y2030={C_ab_2030}, "
        f"pipe_BC y2030={C_bc_2030}"
    )

    # Forward trade f̄ ≈ f* (summed over the (d,t) grid).
    f_ab = float(res.trade_flow[("pipe_AB", "lh2_A", "lh2_B")]["value"].sum())
    f_bc = float(res.trade_flow[("pipe_BC", "lh2_B", "lh2_C")]["value"].sum())
    assert np.isclose(f_ab, f_ab_star, rtol=1e-3, atol=1e-4), (
        f"A→B trade {f_ab} != monolith {f_ab_star}"
    )
    assert np.isclose(f_bc, f_bc_star, rtol=1e-3, atol=1e-4), (
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
    perturbs ONLY that pinned f̄ cell by ±eps and re-solves, returning the
    ECONOMIC (displaced-cost) one-sided derivative — i.e. the reduced cost of
    the pinned column, which is exactly what the Benders cut carries.

    Two-sided is REQUIRED: at the pinned optimum f̄ is at a degenerate
    vertex, so only ONE side stays in the economic (re-dispatch) regime while
    the other forces an unserved-energy penalty (magnitude ≈ penalty ·
    op_factor · unitsize, orders larger).  WHICH side is economic differs by
    cell — a period whose region_B is import-tight (its grown demand consumes
    every imported unit) admits the RIGHT derivative; a period with import
    slack admits the LEFT.  Taking the smaller-|·| (non-penalty) side yields
    the displaced-cost gradient the loop's ``col_dual`` reproduces, in BOTH
    periods.  Also returns the per-period ``nodeBalanceBlock_eq`` dual at
    lh2_B read at the same (settled) vertex.
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
    # first rep step of each period → where we read the block dual once.
    first_rep = {}
    for (dd, tt) in cells:
        first_rep.setdefault(dd, tt)
    for (dd, tt) in cells:
        base_f = fbar[(dd, tt)]
        solm = pin_and_solve((dd, tt), base_f - eps)
        solp = pin_and_solve((dd, tt), base_f + eps)
        fd_left = (cost0 - float(solm.obj)) / eps
        fd_right = (float(solp.obj) - cost0) / eps
        # The ECONOMIC (displaced-cost) side is the smaller-magnitude one; the
        # other forces an unserved-energy penalty (orders larger).  That is the
        # reduced cost the Benders cut carries.
        econ_left = abs(fd_left) <= abs(fd_right)
        out[(dd, tt)] = fd_left if econ_left else fd_right
        # Read the region block dual at lh2_B at the settled (economic) vertex
        # for this period — once, at its first rep cell.
        if tt == first_rep[dd]:
            sol_econ = solm if econ_left else solp
            block_dual[dd] = float(sol_econ.row_dual[block_rows[dd]])
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
    (independent finite difference) — the reduced cost of the pinned boundary
    column — AND the slopes are genuinely PERIOD-DISTINCT and REP-DISTINCT,
    with the loop converging to the RP-weighted monolith on those raw slopes.

    On the (redesigned, non-degenerate) 4-day fixture the two FlexTool periods
    select DIFFERENT days of the shared timeline with DIFFERENT (grown) demand,
    so the marginal value of imported LH2 — the cut slope — differs by PERIOD;
    and within a period the two representative days carry different wind cost
    and different RP weight, so the slope differs by REP too.  This is the
    corrected behaviour: the RP weight rides into the op_factor-weighted
    clearing objective, and the raw ``col_dual`` the cut carries reproduces the
    true region gradient with NO extra per-cell RP factor — the loop's exact
    convergence to ``M_rp`` on these raw slopes is the end-to-end proof.
    """
    # y2030 reps: t0001 (w=1.4, full-wind day) / t0025 (w=0.6, low-wind day);
    # y2040 reps: t0049 (w=1.1, full-wind day) / t0073 (w=0.9, low-wind day).
    # All four carry NON-UNIT RP weight and non-trivial forward flow, and the
    # y2040 days carry the grown demand that makes the periods distinct.
    cells = [("y2030", "t0001"), ("y2030", "t0025"),
             ("y2040", "t0049"), ("y2040", "t0073")]

    # (a) Run the loop; capture every region_B cut's per-cell slope (the live
    # ``sol_r.col_dual`` of the pinned forward column).
    res, captured = _capture_loop_slopes(rp_data, _IMPORT_REGION)
    assert res.converged and np.isclose(res.total_objective, monolith.obj, rtol=1e-4)

    # (b) Independent finite difference of region_B's cost wrt the pinned f̄
    # (two-sided → the economic/displaced-cost side per cell), plus the
    # region's nodeBalanceBlock_eq dual at lh2_B at that settled vertex.
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

    s_2030_r1 = loop_slope[("y2030", "t0001")]   # w = 1.4, full-wind day
    s_2030_r2 = loop_slope[("y2030", "t0025")]   # w = 0.6, low-wind day
    s_2040_r1 = loop_slope[("y2040", "t0049")]   # w = 1.1, full-wind day
    s_2040_r2 = loop_slope[("y2040", "t0073")]   # w = 0.9, low-wind day

    # --- LOCK 2: the slope is REP-DISTINCT within a period.  The two
    # representative days of a period carry DIFFERENT wind cost AND different
    # RP weight, so the op_factor-weighted marginal genuinely differs — the RP
    # weight reaches the cut per rep (a silently-clobbered weight, or a blended
    # single-block, would make the two reps coincide).
    assert not np.isclose(s_2030_r1, s_2030_r2, rtol=1e-3), (
        f"y2030 reps coincide — the per-rep RP weight did not reach the cut: "
        f"w=1.4 {s_2030_r1:.10e} vs w=0.6 {s_2030_r2:.10e}"
    )
    assert not np.isclose(s_2040_r1, s_2040_r2, rtol=1e-3), (
        f"y2040 reps coincide: w=1.1 {s_2040_r1:.10e} vs w=0.9 "
        f"{s_2040_r2:.10e}"
    )

    # --- LOCK 3: the slope is PERIOD-DISTINCT.  y2040's grown demand pushes
    # region A into a higher supply tier, so the imported-LH2 marginal in
    # region_B is genuinely different between the two periods (the degenerate
    # symmetric fixture made this period-UNIFORM — the whole point of the
    # redesign).  Assert at BOTH matched-rep positions to be robust.
    assert not np.isclose(s_2030_r1, s_2040_r1, rtol=1e-2), (
        f"y2030 and y2040 (full-wind rep) slopes coincide — the period "
        f"structure is not exercised: {s_2030_r1:.10e} vs {s_2040_r1:.10e}"
    )
    assert not np.isclose(s_2030_r2, s_2040_r2, rtol=1e-2), (
        f"y2030 and y2040 (low-wind rep) slopes coincide: "
        f"{s_2030_r2:.10e} vs {s_2040_r2:.10e}"
    )
    # The period-distinctness is also visible in the region block dual at
    # lh2_B: read at the settled vertex it differs between the periods (it is
    # set by the period's op_factor-weighted clearing objective).
    assert not np.isclose(
        region_block_dual["y2030"], region_block_dual["y2040"], rtol=1e-2
    ), (
        f"region block dual at lh2_B is period-uniform: "
        f"{region_block_dual['y2030']!r} vs {region_block_dual['y2040']!r}"
    )

    # --- LOCK 4 (end-to-end): the loop converged to the EXACT RP-weighted
    # M_rp using these raw slopes (no factor applied anywhere).  If an RP
    # factor were missing or extra, the cut would be mis-scaled and either cut
    # off the optimum (LB > M, the loop's valid-bound check would have raised)
    # or fail to reconcile — neither happened.
    assert np.isclose(res.total_objective, _M_RP_EXPECTED, rtol=1e-4), (
        f"loop UB {res.total_objective:.8e} != M_rp {_M_RP_EXPECTED:.8e}"
    )
    assert res.lower_bound <= monolith.obj * (1 + 1e-9)
