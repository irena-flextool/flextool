"""Tests for the adequacy-calibrator adder SIZING (C1b).

Three layers:

1. **Pure math** (no DB, no solver) — ``scalar_adder`` / ``sized_increments``
   / ``w_from_grids``: the exact ``a = λ·X/W`` inverse, the ``weight≡1``
   full-year reduction (``W == Σ n_d`` when ``share ≡ 1``), non-shedding
   nodes skipped, and the ``W > 0`` guard.

2. **W from the DB** (builds a tmp DB, no solve) — ``invest_weight_W``
   resolves the invest solve and reduces its per-period grid to
   ``W = Σ_d n_d / share_d`` via the engine's own
   ``derive_per_solve_aggregates``.

3. **Empirical W** (``solver``/``slow``) — proves ``a·W`` equals the annual
   demand a constant adder actually injects, on a REAL single-period invest
   solve, and that ``run_calibration`` drives unserved slack DOWN across
   iterations on a nested under-build fixture.

The empirical W check uses the DEMAND-delta (not the slack-delta): a
constant per-timestep adder ``a`` is a deterministic RHS change that
deepens the node's annual demand by EXACTLY ``a·W`` regardless of how the
solve serves it — so it isolates the annualisation weight cleanly, whereas
a slack-delta is confounded by otherwise-curtailed free supply absorbing
the marginal demand.  A SINGLE-period scenario is used so the full ``W``
equals the one period's weight and the match is exact (no per-period
node-presence ambiguity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.calibrate._sizing import (  # noqa: E402
    invest_weight_W,
    scalar_adder,
    size_timed,
    sized_increments,
    w_from_grids,
)

FIXTURE_JSON = _TESTS_DIR / "fixtures" / "tests.json"

# Single invest period (p2020 only) → full W == that period's weight, so the
# empirical a·W demand-delta is exact (8760).
FLAT_SCENARIO = "y2020_2029_1x10y"
FLAT_W = 8760.0
# Nested invest→dispatch chain: the adder lives in the invest (rep) grid and
# the read slack is the realised/dispatch slack → raising the margin builds
# more capacity and drives the realised shed DOWN (the calibrator's point).
NESTED_SCENARIO = "multi_fullYear_battery_nested_24h_invest_one_solve"
NESTED_W = 35040.0

# Representative-period (representative_period_weights) invest scenario — the
# calibrator's PRIMARY target.  Two RP periods (y2030, y2040), each 48 steps
# with NON-uniform per-step weights (0.6/1.4, 0.9/1.1) that annualise to
# 8760/period → W = 2 × 8760.  Lives in its own fixture JSON.
RP_FIXTURE_JSON = _TESTS_DIR / "fixtures" / "lh2_three_region_rp_invest.json"
RP_SCENARIO = "lh2_three_region_rp_invest"
RP_W = 17520.0
RP_INVEST_SOLVE = "lh2_rp_invest__lh2_three_region_rp_invest"
RP_DEMAND_NODE = "elec_A"  # spans BOTH RP periods → adder deepens both


# ---------------------------------------------------------------------------
# 1. Pure math
# ---------------------------------------------------------------------------

def test_scalar_adder_is_exact_inverse():
    # a·W must recover λ·X exactly.
    W, lam, X = 35040.0, 0.5, 358.0
    a = scalar_adder(X, W, lam)
    assert a == pytest.approx(lam * X / W)
    assert a * W == pytest.approx(lam * X)


def test_scalar_adder_guards_nonpositive_W():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            scalar_adder(100.0, bad, 1.0)


def test_w_from_grids_full_year_reduction():
    # weight ≡ 1, share ≡ 1  ⇒  W == total invest step count.
    assert w_from_grids({"p": 8760}, {"p": 1.0}) == 8760.0
    assert w_from_grids({"a": 100, "b": 40}, {"a": 1.0, "b": 1.0}) == 140.0


def test_w_from_grids_representative_periods():
    # 20 hourly steps sampling a full year (share 20/8760) → 8760 per period.
    W = w_from_grids({"a": 20, "b": 20}, {"a": 20 / 8760, "b": 20 / 8760})
    assert W == pytest.approx(2 * 8760.0)


def test_w_from_grids_uses_weight_sum_not_step_count():
    # A general RP grid: 40 steps in the period but Σ_t timestep_weight = 30
    # (unequal rep-block lengths / un-normalised weights).  W must use the
    # WEIGHT SUM (30), NOT the step count (40).
    share = 40 / 8760
    w_by_weight = w_from_grids({"p": 30.0}, {"p": share})
    w_by_stepcount = w_from_grids({"p": 40.0}, {"p": share})
    assert w_by_weight == pytest.approx(30.0 / share)
    assert w_by_weight != pytest.approx(w_by_stepcount)


def test_w_from_grids_guards_zero_share():
    with pytest.raises(ValueError):
        w_from_grids({"p": 20}, {"p": 0.0})


def test_sized_increments_math_and_skip():
    W, lam, tol = 35040.0, 0.5, 1e-6
    residual = {"shed": 358.0, "dust": 1e-9, "zero": 0.0}
    inc = sized_increments(residual, W=W, lam=lam, tol=tol)
    # Non-shedding nodes (residual <= tol) are skipped entirely.
    assert set(inc) == {"shed"}
    assert inc["shed"] == pytest.approx(lam * 358.0 / W)


def test_sized_increments_guards_nonpositive_W():
    with pytest.raises(ValueError):
        sized_increments({"n": 1.0}, W=0.0, lam=1.0)


def test_sized_increments_overshoot_multiplies():
    # overshoot scales the per-node increment linearly; default (1.0) is off.
    W, lam = 35040.0, 0.5
    residual = {"shed": 358.0}
    base = sized_increments(residual, W=W, lam=lam)
    over = sized_increments(residual, W=W, lam=lam, overshoot=1.2)
    assert base["shed"] == pytest.approx(lam * 358.0 / W)
    assert over["shed"] == pytest.approx(1.2 * lam * 358.0 / W)
    # Byte-parity of the default path: overshoot=1.0 == omitting it.
    assert sized_increments(residual, W=W, lam=lam, overshoot=1.0) == base


# ---------------------------------------------------------------------------
# 1b. Pure timed sizer — the T2 fold + per-cell sizing (no DB, no solver)
# ---------------------------------------------------------------------------

# A synthetic two-base-block / two-rep-block hull fold over one period "p".
# Timeline offsets h=0,1,2 within each block.  Hull weights (sum to 1 per base
# block): b0(t0..t2) -> {r0:0.6, r1:0.4}; b1(t3..t5) -> {r0:0.2, r1:0.8}.
# Every base cell folds onto the rep cell that SHARES its within-block offset.
_TIMED_EDGES = [
    # base b0 -> r0 (weight 0.6), offset-preserving
    ("p", "t0", "t0", 0.6), ("p", "t1", "t1", 0.6), ("p", "t2", "t2", 0.6),
    # base b0 -> r1 (weight 0.4)
    ("p", "t0", "t3", 0.4), ("p", "t1", "t4", 0.4), ("p", "t2", "t5", 0.4),
    # base b1 -> r0 (weight 0.2)
    ("p", "t3", "t0", 0.2), ("p", "t4", "t1", 0.2), ("p", "t5", "t2", 0.2),
    # base b1 -> r1 (weight 0.8)
    ("p", "t3", "t3", 0.8), ("p", "t4", "t4", 0.8), ("p", "t5", "t5", 0.8),
]
# Engine per-cell timestep weight: w_r[rep] = Σ_base·n_rp/n_base (n_rp=n_base=2
# here → factor 1).  r0 cells: 0.6+0.2=0.8; r1 cells: 0.4+0.8=1.2.
_TIMED_TW = {
    ("p", "t0"): 0.8, ("p", "t1"): 0.8, ("p", "t2"): 0.8,
    ("p", "t3"): 1.2, ("p", "t4"): 1.2, ("p", "t5"): 1.2,
}


def test_size_timed_energy_conservation():
    # The load-bearing property: the fold gives only the per-cell SHAPE, and
    # ``size_timed`` NORMALISES it to the TRUE annual residual — so the injected
    # energy equals λ·annual_residual REGARDLESS of the raw dt-slack total.
    #
    # Here annual_residual (30) ≠ Σ dt_slack (15): the dt table is on the
    # representative grid (its total is a fraction of the annual residual), so a
    # sizer that merely conserved the raw dt total would inject 15 and
    # under-provision 2×.  The old tautological test set residual == Σ dt_slack
    # so this gap could never surface.
    dt_slack = {"n": {("p", "t1"): 10.0, ("p", "t4"): 5.0}}  # Σ = 15
    annual_residual = 30.0                                    # ≠ Σ dt_slack
    residual = {"n": annual_residual}
    lam = 1.0
    out = size_timed(dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=lam)

    adder = out["n"]
    injected = sum(a * _TIMED_TW[cell] for cell, a in adder.items())
    # Normalisation target: Σ adder·tw_rep == λ·annual_residual (== 30), NOT
    # λ·Σ(dt_slack) (== 15) that the un-normalised fold would give.
    assert injected == pytest.approx(lam * annual_residual, rel=1e-12)
    assert injected != pytest.approx(lam * 15.0, rel=1e-3)

    # Shape preserved: slack_rep(t1)=0.6·10+0.2·5=7; (t4)=0.4·10+0.8·5=8; the
    # shape is scaled by annual_residual/Σshape = 30/15 = 2 then /tw_rep.
    shape_total = 7.0 + 8.0
    scale = lam * annual_residual / shape_total
    assert adder[("p", "t1")] == pytest.approx(scale * 7.0 / 0.8, rel=1e-12)
    assert adder[("p", "t4")] == pytest.approx(scale * 8.0 / 1.2, rel=1e-12)
    # The per-cell ENERGY split still follows the stress shape (7:8).
    e_t1 = adder[("p", "t1")] * 0.8
    e_t4 = adder[("p", "t4")] * 1.2
    assert e_t1 / e_t4 == pytest.approx(7.0 / 8.0, rel=1e-12)


def test_size_timed_unfolded_dt_is_unchanged_by_normalisation():
    # When the dt table IS pre-unfolded (Σ dt_slack == annual_residual, the H2
    # model), normalisation is a no-op: scale == λ and the per-cell adder is
    # exactly λ·slack_rep/tw_rep — byte-identical to the naive fold.
    dt_slack = {"n": {("p", "t1"): 10.0, ("p", "t4"): 5.0}}
    residual = {"n": 15.0}  # == Σ dt_slack (unfolded onto the base timeline)
    lam = 0.5
    out = size_timed(dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=lam)
    adder = out["n"]
    assert adder[("p", "t1")] == pytest.approx(lam * 7.0 / 0.8, rel=1e-12)
    assert adder[("p", "t4")] == pytest.approx(lam * 8.0 / 1.2, rel=1e-12)
    injected = sum(a * _TIMED_TW[cell] for cell, a in adder.items())
    assert injected == pytest.approx(lam * 15.0, rel=1e-12)


def test_size_timed_overshoot_scales_injected_energy():
    # The overshoot safety multiplier scales the injected energy linearly:
    # Σ adder·tw_rep == overshoot·λ·annual_residual, shape unchanged.
    dt_slack = {"n": {("p", "t1"): 10.0, ("p", "t4"): 5.0}}
    residual = {"n": 30.0}
    lam = 1.0
    base = size_timed(dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=lam)["n"]
    over = size_timed(
        dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=lam, overshoot=1.2,
    )["n"]
    inj_over = sum(a * _TIMED_TW[c] for c, a in over.items())
    assert inj_over == pytest.approx(1.2 * lam * 30.0, rel=1e-12)
    # Every cell is scaled by exactly 1.2 relative to overshoot=1.0.
    assert set(over) == set(base)
    for cell in base:
        assert over[cell] == pytest.approx(1.2 * base[cell], rel=1e-12)


def test_size_timed_mass_lands_at_stressed_cells():
    # Stress only at base offset 1 → mass only at the rep cells sharing that
    # offset (t1, t4); the unstressed rep cells get NO adder.
    dt_slack = {"n": {("p", "t1"): 10.0, ("p", "t4"): 5.0}}
    residual = {"n": 15.0}
    out = size_timed(dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=1.0)
    assert set(out["n"]) == {("p", "t1"), ("p", "t4")}
    for cell in (("p", "t0"), ("p", "t2"), ("p", "t3"), ("p", "t5")):
        assert cell not in out["n"]


def test_size_timed_skips_nonshedding_and_absent():
    dt_slack = {
        "shed": {("p", "t1"): 10.0},
        "dust": {("p", "t1"): 10.0},   # below tol → skipped
        "noprofile": {},               # shedding but no dt rows → skipped
    }
    residual = {"shed": 20.0, "dust": 1e-9, "noprofile": 50.0}
    out = size_timed(dt_slack, residual, _TIMED_EDGES, _TIMED_TW, lam=1.0)
    assert set(out) == {"shed"}


def test_size_timed_degenerate_single_rep_is_identity():
    # Identity fold: each invest cell maps to itself (weight 1), tw≡1 → the
    # per-cell adder is just λ·slack at that cell (the sensible reduction).
    edges = [("p", "t0", "t0", 1.0), ("p", "t1", "t1", 1.0)]
    tw = {("p", "t0"): 1.0, ("p", "t1"): 1.0}
    dt_slack = {"n": {("p", "t0"): 3.0, ("p", "t1"): 7.0}}
    residual = {"n": 10.0}
    out = size_timed(dt_slack, residual, edges, tw, lam=0.5)
    assert out["n"][("p", "t0")] == pytest.approx(0.5 * 3.0)
    assert out["n"][("p", "t1")] == pytest.approx(0.5 * 7.0)
    injected = sum(a * tw[c] for c, a in out["n"].items())
    assert injected == pytest.approx(0.5 * 10.0, rel=1e-12)


def test_rp_fold_edges_conserve_and_match_tw(tmp_path):
    """``_rp_fold_edges_and_tw`` on the RP fixture: every base cell's out-edge
    weights sum to 1 (total-conserving fold) and ``tw_rep`` reproduces the
    engine writer's per-period ``timestep_weight`` sum."""
    from flextool.calibrate._sizing import (
        _rp_fold_edges_and_tw,
        _rp_weight_sum_by_period,
    )
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig

    url = json_to_db(RP_FIXTURE_JSON, tmp_path / "c.sqlite")
    sc = SolveConfig.load_from_db_url(url, RP_SCENARIO)
    tc = TimelineConfig.load_from_db_url(url, RP_SCENARIO)

    for period, ts in sc.timesets_used_by_solves[RP_INVEST_SOLVE]:
        if ts not in tc.rp_weights:
            continue
        edges, tw_rep = _rp_fold_edges_and_tw(tc, str(period), ts)
        # Per base cell (period, base_time), Σ out-edge weight == 1.
        by_base: dict[tuple[str, str], float] = {}
        for p, bt, _rt, w in edges:
            by_base[(p, bt)] = by_base.get((p, bt), 0.0) + w
        for key, total in by_base.items():
            assert total == pytest.approx(1.0, rel=1e-9), key
        # tw_rep total for the period matches the uniform-sizer weight sum.
        want = _rp_weight_sum_by_period(tc, str(period), ts)[str(period)]
        got = sum(v for (p, _t), v in tw_rep.items() if p == str(period))
        assert got == pytest.approx(want, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. W from the DB (no solve)
# ---------------------------------------------------------------------------

def test_invest_weight_W_single_period(tmp_path):
    url = json_to_db(FIXTURE_JSON, tmp_path / "c.sqlite")
    assert invest_weight_W(url, FLAT_SCENARIO) == pytest.approx(FLAT_W)


def test_invest_weight_W_nested_invest(tmp_path):
    # Four representative-72-step periods → 4 × 8760 (NOT 4 × 72: the naive
    # /n_invest_steps would be 121× too small here).
    url = json_to_db(FIXTURE_JSON, tmp_path / "c.sqlite")
    W = invest_weight_W(url, NESTED_SCENARIO)
    assert W == pytest.approx(NESTED_W)


def test_invest_weight_W_representative_periods(tmp_path):
    # The RP target: no exception (the earlier NotImplementedError is gone),
    # and W = 2 × 8760 from the folded representative_period_weights.
    url = json_to_db(RP_FIXTURE_JSON, tmp_path / "c.sqlite")
    assert invest_weight_W(url, RP_SCENARIO) == pytest.approx(RP_W)


def test_rp_weight_sum_uses_actual_nonuniform_weights(tmp_path):
    """The RP path sums the ACTUAL (non-uniform) folded weights, not ``n_d``.

    ``_weight_sum_by_period`` must reproduce the engine writer's
    ``_compute_rp_frames`` per-period weight sum, whose underlying per-step
    weights are genuinely non-uniform (0.6/1.4, 0.9/1.1 here) — proving the
    sizer reads the real ``timestep_weight`` the annualiser applies, not a
    step-count shortcut that would ignore the weighting.
    """
    from flextool.calibrate._sizing import (
        _rp_weight_sum_by_period,
        _weight_sum_by_period,
    )
    from flextool.engine_polars._per_solve_sets import (
        derive_per_solve_aggregates,
    )
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._timeline import TimelineConfig

    url = json_to_db(RP_FIXTURE_JSON, tmp_path / "c.sqlite")
    src = SpineDbReader(url, RP_SCENARIO)
    sc = SolveConfig.load_from_db_url(url, RP_SCENARIO)
    tc = TimelineConfig.load_from_db_url(url, RP_SCENARIO)
    agg = derive_per_solve_aggregates(src, RP_INVEST_SOLVE)

    wsum = _weight_sum_by_period(src, sc, tc, RP_INVEST_SOLVE, agg.dt_complete)
    # Two RP periods, each summing to 48 (2 rep blocks × 24 steps, weights
    # scaled so Σ = n_d here — but reached by SUMMING non-uniform weights).
    assert set(wsum) == {"y2030", "y2040"}
    assert wsum["y2030"] == pytest.approx(48.0)
    assert wsum["y2040"] == pytest.approx(48.0)

    # The underlying per-step RP weights are genuinely non-uniform — the code
    # summed them, it did not count steps.
    for period, ts in sc.timesets_used_by_solves[RP_INVEST_SOLVE]:
        if ts in tc.rp_weights:
            import polars as pl

            from flextool.engine_polars._emit_solve_writers import (
                _compute_rp_frames,
            )
            tl = tc.timesets__timeline[ts]
            steps = [s for s, _ in tc.timelines[tl]]
            tw = _compute_rp_frames(
                tc.rp_weights[ts], tc.timeset_durations[ts], period, steps,
            )["timestep_weight.csv"].with_columns(pl.col("weight").cast(pl.Float64))
            uniq = set(tw["weight"].round(6).unique().to_list())
            assert len(uniq) > 1, (
                f"RP timeset {ts} weights unexpectedly uniform {uniq} — the "
                f"non-uniform-weight assertion would be vacuous"
            )
            assert 1.0 not in uniq or len(uniq) > 1

    # Sanity: per-timeset helper agrees with the aggregate.
    single = _rp_weight_sum_by_period(
        tc, "y2030", "rp_y2030__lh2_three_region_rp_invest",
    )
    assert single["y2030"] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# 3. Empirical (solver / slow)
# ---------------------------------------------------------------------------

def _build_db(tmp_path_factory, tag: str):
    from flextool.update_flextool.db_migration import migrate_database

    root = tmp_path_factory.mktemp(tag)
    url = json_to_db(FIXTURE_JSON, root / "c.sqlite")
    migrate_database(url)
    return url, root


def _node_inflow_total(assess_dir: Path, node: str) -> float:
    """Annual ``Inflow`` (demand, negative MWh) summed over periods for *node*."""
    from flextool.lean_parquet import read_lean_parquet

    df = read_lean_parquet(Path(assess_dir) / "node_d_ep.parquet")
    is_node = df.columns.get_level_values("node") == node
    is_inflow = df.columns.get_level_values("category") == "Inflow"
    return float(df.loc[:, is_node & is_inflow].values.sum())


def _aw_delta_inflow(tmp_path_factory, fixture_json, scenario, node, a):
    """Return ``(W, |ΔInflow|)`` for a known adder *a* on *node* in *scenario*.

    Builds two fresh DBs (adder-off / adder-on), solves each, and reports the
    node's annual-demand change — a deterministic RHS delta that equals
    ``a·W`` for a correct annualisation weight, independent of how the solve
    serves the demand.
    """
    from flextool.calibrate._db_alt import write_calib_alt
    from flextool.calibrate._solve import run_solve
    from flextool.update_flextool.db_migration import migrate_database

    def _solve(tag, adder):
        root = tmp_path_factory.mktemp(tag)
        url = json_to_db(fixture_json, root / "c.sqlite")
        migrate_database(url)
        if adder:
            write_calib_alt(url, scenario, {node: a})
        run = run_solve(
            url, scenario,
            work_dir=root / "w", out_root=root / "o", cache_dir=root / "cache",
        )
        return _node_inflow_total(run.assess_dir, node), url

    off, url = _solve("aw_off", False)
    W = invest_weight_W(url, scenario)
    on, _ = _solve("aw_on", True)
    return W, abs(on - off)


@pytest.mark.solver
@pytest.mark.slow
def test_empirical_a_times_W_is_injected_annual_demand(tmp_path_factory):
    """PROVE ``a·W`` == the annual demand a constant adder injects.

    On the single-period invest scenario a known adder ``a`` on west must
    deepen west's annual ``Inflow`` output by EXACTLY ``a·W`` — the exact
    end-to-end validation of the annualisation weight through a real solve.
    """
    a = 100.0
    W, delta = _aw_delta_inflow(
        tmp_path_factory, FIXTURE_JSON, FLAT_SCENARIO, "west", a,
    )
    print(
        f"\n[empirical a·W flat] W={W}  a={a}  a·W={a * W}  "
        f"|ΔInflow|={delta}  rtol={abs(delta - a * W) / (a * W):.3e}"
    )
    assert W == pytest.approx(FLAT_W)
    # A constant adder is a deterministic RHS change → exact to solver noise.
    assert delta == pytest.approx(a * W, rel=1e-3)


@pytest.mark.solver
@pytest.mark.slow
def test_empirical_a_times_W_representative_periods(tmp_path_factory):
    """PROVE ``a·W`` on an RP scenario — the calibrator's PRIMARY target.

    ``elec_A`` carries demand in BOTH RP periods (each 48 steps with
    NON-uniform folded weights 0.6/1.4 and 0.9/1.1).  A known adder ``a`` must
    deepen its annual ``Inflow`` by EXACTLY ``a·W`` — the end-to-end proof
    that ``W`` reads the same representative-period weights the annualiser
    applies to ``node_slack_up_d_e``.
    """
    a = 100.0
    W, delta = _aw_delta_inflow(
        tmp_path_factory, RP_FIXTURE_JSON, RP_SCENARIO, RP_DEMAND_NODE, a,
    )
    print(
        f"\n[empirical a·W RP] W={W}  a={a}  a·W={a * W}  "
        f"|ΔInflow|={delta}  rtol={abs(delta - a * W) / (a * W):.3e}"
    )
    assert W == pytest.approx(RP_W)
    assert delta == pytest.approx(a * W, rel=1e-3)


@pytest.mark.solver
@pytest.mark.slow
def test_run_calibration_drives_slack_down(tmp_path_factory):
    """On the nested under-build fixture, real sizing (undamped) must raise
    the adder each iteration and drive total unserved slack DOWN, monotone,
    toward the threshold.

    The adder lives in the invest (rep-period) grid; the read slack is the
    realised/dispatch slack, so more margin → more built capacity → less
    realised shed.  On THIS fixture the response is small per step (a uniform
    rep-period margin barely builds realised-PEAK capacity — the known
    rep-period peak-mismatch weakness, orthogonal to sizing correctness,
    which the exact a·W test above pins), so the assertion is on the SIGN and
    monotonicity, not the rate.  Undamped (λ=1) so the direction is clean.
    """
    from flextool.calibrate._loop import CalibConfig, run_calibration

    url, root = _build_db(tmp_path_factory, "converge")
    cfg = CalibConfig(
        iterations=3,
        slack_threshold_mwh=1.0,
        damping_first=1.0,
        damping_remaining=1.0,
        over_build_tightness=0.0,
        # stall_fraction=0.0 makes the guard a strict no-op: freezing needs
        # ΔSlack < 0 (residual GREW), which never happens while slack is
        # monotone improving — so the guard cannot flag on this converging
        # fixture (the role tightness=0.0 played under the old η gate).
        stall_fraction=0.0,
        warm_start_cache_dir=root / "cache",
        work_dir=root / "work",
        out_root=root / "out",
        debug=False,
    )
    result = run_calibration(url, NESTED_SCENARIO, cfg)

    totals = [r.total_unserved for r in result.trajectory]
    west_adders = [r.adders.get("west", 0.0) for r in result.trajectory]
    print(f"\n[convergence] unserved per iteration: {totals}")
    print(f"[convergence] west adder per iteration: {west_adders}")
    print(f"[convergence] final adders: {result.final_adders}")

    assert len(totals) >= 3, "need baseline + at least two corrections"
    assert totals[0] > 0.0, "baseline must shed for the test to be meaningful"

    # Sizing raised the adder on the shedding node, strictly, every step
    # (deterministic: residual > threshold ⇒ positive increment accumulated).
    assert result.final_adders.get("west", 0.0) > 0.0
    for prev, cur in zip(west_adders[1:], west_adders[2:]):
        assert cur > prev, f"west adder did not keep rising: {west_adders}"

    # Unserved is monotone non-increasing (tiny per-step, so allow solver
    # noise) and strictly lower overall by more than that noise floor — the
    # correct direction, not a wobble.
    for prev, cur in zip(totals, totals[1:]):
        assert cur <= prev + 1e-3, f"unserved rose mid-run: {totals}"
    assert totals[-1] < totals[0] - 1e-2, (
        f"unserved did not move toward the threshold: {totals}"
    )

    # C1c over-build guard is a NO-OP here: with stall_fraction=0.0 the freeze
    # condition ``ΔSlack < 0.0`` can never hold while slack is IMPROVING
    # (ΔSlack>0), so no node is resource-capped and monotone progress is
    # untouched.  The guard must not have flagged anything on this fixture.
    assert result.guard_flagged_nodes == [], (
        f"guard wrongly flagged nodes on a converging fixture: "
        f"{result.guard_flagged_nodes}"
    )
