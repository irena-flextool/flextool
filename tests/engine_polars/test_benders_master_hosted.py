"""Benders master-hosted nodes — driver integration (C8), end-to-end.

Master-hosted nodes are balance/state nodes in NO region group: the
driver (:func:`flextool.engine_polars._benders.solve_benders`) derives
the set from the input's ``group_node`` membership, announces it loudly,
re-partitions around it (single-sided region↔master coupling arcs,
master-local arcs/procs native in the master — C6/C7), folds the
master's stand-alone (autarky) cost into the stall-guard reference
scale (plan D-c, via the coordinator's ``extra_reference_cost`` hook),
extends the penalty precondition to master-hosted balance nodes (F9),
rejects mixed-resolution and in-out-stabilized configurations it cannot
solve EXACTLY (loud errors, never silent distortion), and rides the
master's own invest/divest decisions on the Tier-1 handoff.

Fixture: the ``lh2_master_hourly`` sibling of the trade-invest LH2
fixture (JSON-fixture DB per CLAUDE.md invariant #3) — node invest on
``lh2_B``, unit invest on ``liquefier_B``, small distinct flow costs on
the connections, and the ``daily_group`` time aggregation REMOVED so
every region↔master coupling arc joins SAME-resolution nodes (the
driver hard-errors on a mixed-resolution boundary; the daily-block
sibling ``lh2_master_build`` is the vehicle for THAT test).

MAIN CONFIGURATION (the spec's 2-region exactness shape): pass
``regions=["region_A", "region_C"]`` — region B's whole subsystem
(``elec_B`` / ``h2_B`` / ``lh2_B`` storage / ``battery_B``) is then in
no listed region group and becomes MASTER-HOSTED, exactly as the
masterfuel repartition hosts the carrier subsystem.  All B units
(wind/coal/battery/electrolyser/liquefier) are master-local, and the
ONLY boundary arcs are the two pipes — every coupling arc single-sided
region↔master, ZERO region↔region arcs (the load-bearing masterfuel
configuration, plan audit 0.D).
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import polar_high.benders as ph_benders
from polar_high import Param, Problem, WarmProblem

import flextool.engine_polars._benders as fx_benders
from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._benders import (
    _assert_finite_boundary_penalties,
    _master_autarky_cost,
    solve_benders,
)
from flextool.engine_polars._region_filter import (
    compute_master_hosted_nodes,
    load_region_membership,
    split,
)

#: The 2-region driver configuration: region B's subsystem is hosted.
REGIONS_AC = ["region_A", "region_C"]
#: The full 3-region list (all grouped ⇒ empty master set).
REGIONS_ABC = ["region_A", "region_B", "region_C"]

#: What ``regions=REGIONS_AC`` master-hosts on this fixture: every
#: balance/state node of region B.
MASTER_B_SUBSYSTEM = frozenset({"battery_B", "elec_B", "h2_B", "lh2_B"})

#: The coupling connections under REGIONS_AC — every one single-sided.
COUPLING_CONNS = {"pipe_AB", "pipe_BC"}

#: Objective scale for every full solve in this module — the hosted
#: master is a REAL hourly LP (balances + penalties + storage + cuts),
#: and at s=1.0 its cost range trips HiGHS into kUnknown mid-loop;
#: s=1e-6 is the production ``scale_the_objective``.
OBJ_SCALE = 1e-6


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _workdirs(scenario_workdir):
    """Build every cascade workdir BEFORE any load_flextool call (the
    known same-process global-axis-enum interleave hazard, cf.
    test_region_filter_master_hosted)."""
    mh = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_master_hourly"
    )
    ti = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    mb = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_master_build"
    )
    return mh, ti, mb


@pytest.fixture(scope="module")
def mh_data(_workdirs):
    """All-hourly master-hosted driver fixture (see module docstring)."""
    return load_flextool(_workdirs[0])


def _copy_with(data, **field_overrides):
    """Shallow FlexData copy with field overrides; re-attaches the
    post-init ``_axis_enums`` attribute that ``dataclasses.replace``
    drops."""
    new = dataclasses.replace(data)
    for k, v in field_overrides.items():
        setattr(new, k, v)
    new._axis_enums = getattr(data, "_axis_enums", None)
    return new


def _host_nodes(data, nodes: frozenset[str], regions: list[str]):
    """FlexData copy whose ``group_node`` no longer lists *nodes* in any
    of *regions* — the DB shape (un-grouped balance/state nodes) that
    makes the driver's membership rule master-host them.  Used by the
    validation tests that need a PARTIAL re-partition (e.g. a straddling
    unit); the main exactness path hosts a whole subsystem via the
    2-region ``regions`` list instead."""
    gn = data.group_node.filter(
        ~(
            pl.col("g").cast(pl.Utf8).is_in(regions)
            & pl.col("n").cast(pl.Utf8).is_in(sorted(nodes))
        )
    )
    return _copy_with(data, group_node=gn)


@pytest.fixture(scope="module")
def monolith(mh_data):
    """The whole undecomposed fixture solved once (M)."""
    pb = Problem()
    build_flextool(pb, mh_data)
    sol = pb.solve()
    assert sol.optimal, "monolith solve not optimal"
    return sol


@pytest.fixture(scope="module")
def benders_result(mh_data, monolith):
    return solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4,
        monolith_objective=monolith.obj, scale_the_objective=OBJ_SCALE,
    )


# ---------------------------------------------------------------------------
# Membership rule sanity: the 2-region list hosts exactly the B subsystem.
# ---------------------------------------------------------------------------


def test_master_hosted_set_detection(mh_data) -> None:
    mem_abc = load_region_membership(mh_data, REGIONS_ABC)
    assert compute_master_hosted_nodes(mh_data, mem_abc) == set()
    mem_ac = load_region_membership(mh_data, REGIONS_AC)
    assert compute_master_hosted_nodes(mh_data, mem_ac) == set(
        MASTER_B_SUBSYSTEM
    )


# ---------------------------------------------------------------------------
# (ii) Coupling-arc-only structure: zero region↔region arcs.
# ---------------------------------------------------------------------------


def test_coupling_arc_only_configuration(mh_data) -> None:
    """Under REGIONS_AC EVERY coupling arc is single-sided (exactly one
    region carries a half-flow) — there are no region↔region arcs left.
    The exactness gate below runs on THIS configuration, so 'coupling-
    arc-only solves and converges' is proven there, not assumed."""
    mem = load_region_membership(mh_data, REGIONS_AC)
    splits = split(
        mh_data, regions=REGIONS_AC, region_membership=mem,
        benders_uncap_cross_region=True,
        master_hosted_nodes=MASTER_B_SUBSYSTEM,
    )
    sides: dict[tuple, list] = {}
    for s in splits:
        for hf in s.half_flows:
            key = (hf.original_p, hf.original_source, hf.original_sink)
            sides.setdefault(key, []).append((s.region, hf.side))
    assert sides, "no coupling arcs found — fixture/topology changed?"
    assert {k[0] for k in sides} == COUPLING_CONNS
    for key, entries in sides.items():
        assert len(entries) == 1, (
            f"arc {key} carries {len(entries)} half-flows — expected "
            f"SINGLE-SIDED (region↔master) only: {entries}"
        )


# ---------------------------------------------------------------------------
# (i) Exactness: Benders == monolith, with a valid LB.
# ---------------------------------------------------------------------------


def test_exactness_converges_to_monolith(benders_result, monolith) -> None:
    M = monolith.obj
    res = benders_result
    assert res.converged, (
        f"Benders did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters (LB={res.lower_bound:.6e} "
        f"UB={res.upper_bound:.6e})"
    )
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"Benders UB {res.total_objective:.8e} != monolith M {M:.8e} "
        f"(LB={res.lower_bound:.8e}, gap={res.gap:.3e})"
    )
    # VALID lower bound: LB <= M.
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"Benders LB {res.lower_bound:.8e} EXCEEDS M {M:.8e}"
    )


# ---------------------------------------------------------------------------
# (iii) + master-local unit: the Tier-1 handoff carries the MASTER's invest.
# ---------------------------------------------------------------------------


def test_invest_handoff_contains_master_invest(benders_result) -> None:
    """The assembled ``invest_solution_vars`` must union the master's own
    frames: the master-hosted storage node's ``v_invest_n`` (lh2_B) and
    the master-local UNIT's ``v_invest_p`` (liquefier_B) — neither is in
    any region's membership, so only the master can contribute them."""
    isv = benders_result.invest_solution_vars
    assert "v_invest_n" in isv, f"missing v_invest_n (keys={list(isv)})"
    n_frame = isv["v_invest_n"]
    n_ents = set(n_frame[n_frame.columns[0]].cast(pl.Utf8).to_list())
    assert "lh2_B" in n_ents, (
        f"master-hosted node lh2_B missing from v_invest_n: {n_ents}"
    )
    assert n_frame.filter(
        pl.col(n_frame.columns[0]).cast(pl.Utf8) == "lh2_B"
    )["value"].is_finite().all()

    assert "v_invest_p" in isv, f"missing v_invest_p (keys={list(isv)})"
    p_frame = isv["v_invest_p"]
    p_ents = set(p_frame[p_frame.columns[0]].cast(pl.Utf8).to_list())
    assert "liquefier_B" in p_ents, (
        f"master-local unit liquefier_B missing from v_invest_p: {p_ents}"
    )


# ---------------------------------------------------------------------------
# (iv) Determinism: workers=1 vs workers=2, EXACT.
# ---------------------------------------------------------------------------


def test_workers_determinism(mh_data) -> None:
    seq = solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4, workers=1,
        scale_the_objective=OBJ_SCALE,
    )
    par = solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4, workers=2,
        scale_the_objective=OBJ_SCALE,
    )
    assert par.iterations == seq.iterations
    assert par.converged == seq.converged
    assert par.total_objective == seq.total_objective
    assert par.lower_bound == seq.lower_bound
    assert par.upper_bound == seq.upper_bound
    assert par.gap == seq.gap
    assert par.region_costs == seq.region_costs
    assert par.invest == seq.invest


# ---------------------------------------------------------------------------
# (v) Stall reference scale includes the master autarky term (D-c).
# ---------------------------------------------------------------------------


def test_stall_reference_includes_master_autarky(
    mh_data, monkeypatch
) -> None:
    """reference_scale = Σ_r |autarky_r| + |master_autarky|: spy on the
    coordinator's StallMonitor construction and on the driver's
    ``_master_autarky_cost`` (called exactly ONCE, post-bootstrap), and
    reconcile against the per-region bootstrap costs streamed by the
    iter-0 subsolve callback."""
    captured_ref: list[float] = []
    real_sm = ph_benders.StallMonitor

    def _spy_sm(reference_scale, **kwargs):
        captured_ref.append(float(reference_scale))
        return real_sm(reference_scale, **kwargs)

    monkeypatch.setattr(ph_benders, "StallMonitor", _spy_sm)

    autarky_vals: list[float] = []
    real_autarky = fx_benders._master_autarky_cost

    def _spy_autarky(master):
        v = real_autarky(master)
        autarky_vals.append(v)
        return v

    monkeypatch.setattr(fx_benders, "_master_autarky_cost", _spy_autarky)

    boot_costs: list[float] = []

    def _on_subsolve(entry: dict) -> None:
        if entry["iter"] == 0:
            boot_costs.append(entry["obj"])

    res = solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4,
        subsolve_callback=_on_subsolve, scale_the_objective=OBJ_SCALE,
    )
    assert res.converged
    assert len(autarky_vals) == 1, (
        f"_master_autarky_cost called {len(autarky_vals)}x — must be ONCE"
    )
    # The master hosts region B's real demand: at zero coupling flow it
    # pays penalty/storage slack, so the autarky term is strictly
    # positive (in scaled space).
    assert autarky_vals[0] > 0.0
    assert len(captured_ref) == 1
    assert len(boot_costs) == len(REGIONS_AC)
    # The callback streams REAL-unit costs (÷s); the reference scale
    # lives in the loop's SCALED space.
    expected = (
        sum(abs(c) for c in boot_costs) * OBJ_SCALE + abs(autarky_vals[0])
    )
    assert captured_ref[0] == pytest.approx(expected, rel=1e-9), (
        f"stall reference {captured_ref[0]:.6e} != Σ|bootstrap| + "
        f"|master autarky| = {expected:.6e}"
    )


def test_all_grouped_never_solves_master_autarky(
    _workdirs, monkeypatch
) -> None:
    """Byte-parity gate: with EVERY node grouped the driver must pass
    ``extra_reference_cost=None`` — no autarky pin, no extra master
    solve.  (The all-grouped trajectory itself is pinned bit-for-bit by
    ``test_benders_trajectory_pin.py``.)  Runs on the plain trade-invest
    fixture — the phase-2-proven all-grouped configuration."""
    ti_data = load_flextool(_workdirs[1])

    def _boom(master):
        raise AssertionError(
            "_master_autarky_cost ran on an all-grouped model — the "
            "empty-set path must add NO extra master solve"
        )

    monkeypatch.setattr(fx_benders, "_master_autarky_cost", _boom)
    res = solve_benders(ti_data, REGIONS_ABC, max_iters=40, tol=1e-4)
    assert res.converged


# ---------------------------------------------------------------------------
# (vi) The autarky pin restores the master's bounds exactly.
# ---------------------------------------------------------------------------


def _tiny_master_lp() -> Problem:
    """min 1·x0 + 2·x1 + 100·(s0+s1)  s.t.  x_i + s_i >= d_i, d=[3,4].

    Unpinned optimum: x=d, obj=11.  With the "coupling" x cols pinned
    to 0 the slack serves the demand: s=d, obj=700 (the analog of the
    master's penalty-slack autarky)."""
    p = Problem()
    idx = pl.DataFrame({"i": [0, 1]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=np.inf)
    s = p.add_var("s", "i", idx, lower=0.0, upper=np.inf)
    demand = Param(
        ("i",), pl.DataFrame({"i": [0, 1], "value": [3.0, 4.0]})
    )
    p.add_cstr(
        "meet", over=idx, sense=">=",
        lhs_terms={"lhs": x + s}, rhs_terms={"d": demand},
    )
    cost_x = Param(("i",), pl.DataFrame({"i": [0, 1], "value": [1.0, 2.0]}))
    cost_s = Param(
        ("i",), pl.DataFrame({"i": [0, 1], "value": [100.0, 100.0]})
    )
    p.set_objective(cost_x * x + cost_s * s, sense="min")
    p.set_solver_options({"output_flag": False})
    return p


def test_autarky_pin_restores_master_bounds() -> None:
    """``_master_autarky_cost`` = pin coupling cols to 0, solve, obj−Ση,
    RESTORE bounds: afterwards the master's bounds equal the pre-pin
    bounds exactly and a re-solve equals a control problem that never
    saw the pin (risk R6)."""
    wp = WarmProblem(_tiny_master_lp())
    sol0 = wp.solve()
    assert sol0.optimal
    assert sol0.obj == pytest.approx(11.0, abs=1e-9)

    x_cols = np.asarray(
        [wp.col_id_of_var("x", (0,)), wp.col_id_of_var("x", (1,))],
        dtype=np.int64,
    )
    s_cols = np.asarray(
        [wp.col_id_of_var("s", (0,)), wp.col_id_of_var("s", (1,))],
        dtype=np.int64,
    )
    all_cols = np.concatenate([x_cols, s_cols])
    lo0, hi0 = wp.get_col_bounds(all_cols)

    master = SimpleNamespace(
        _wp=wp,
        arcs=[SimpleNamespace(f_col_ids=x_cols)],
        _eta_col={},
    )
    autarky = _master_autarky_cost(master)
    assert autarky == pytest.approx(700.0, rel=1e-9)

    # Bounds restored EXACTLY (including the +inf uppers).
    lo1, hi1 = wp.get_col_bounds(all_cols)
    np.testing.assert_array_equal(lo1, lo0)
    np.testing.assert_array_equal(hi1, hi0)

    # Master re-solve equals a control run without the pin.
    sol1 = wp.solve()
    assert sol1.optimal
    control = WarmProblem(_tiny_master_lp())
    csol = control.solve()
    assert csol.optimal
    assert sol1.obj == pytest.approx(csol.obj, rel=1e-12)
    assert sol1.obj == pytest.approx(11.0, abs=1e-9)


def test_coupling_box_intersects_and_restores() -> None:
    """``apply_coupling_box`` intersects ``[center ± radius]`` with the
    coupling columns' ORIGINAL bounds (never widens), and
    ``clear_coupling_box`` restores those bounds EXACTLY — including across
    repeated apply/clear cycles (the coordinator applies + clears around every
    boxed solve)."""
    wp = WarmProblem(_tiny_master_lp())
    assert wp.solve().optimal

    x_cols = np.asarray(
        [wp.col_id_of_var("x", (0,)), wp.col_id_of_var("x", (1,))],
        dtype=np.int64,
    )
    # Give the coupling cols a FINITE original box so the upper clamp is
    # exercised too (the tiny LP's default upper is +inf).
    wp.set_col_bounds(x_cols, np.array([1.0, 2.0]), np.array([10.0, 8.0]))
    lo0, hi0 = wp.get_col_bounds(x_cols)

    master = SimpleNamespace(
        _wp=wp, arcs=[SimpleNamespace(f_col_ids=x_cols)],
        _coupling_box_saved=None,
    )
    apply_box = fx_benders._BendersMaster.apply_coupling_box
    clear_box = fx_benders._BendersMaster.clear_coupling_box
    center = {int(x_cols[0]): 5.0, int(x_cols[1]): 5.0}

    # (a) INTERSECTION: box [3,7],[3,7] ∩ original [1,10],[2,8] = [3,7],[3,7].
    apply_box(master, center, 2.0)
    blo, bhi = wp.get_col_bounds(x_cols)
    np.testing.assert_array_equal(blo, [3.0, 3.0])
    np.testing.assert_array_equal(bhi, [7.0, 7.0])
    assert master._coupling_box_saved is not None

    # Clear restores EXACTLY (finite bounds).
    clear_box(master)
    lo1, hi1 = wp.get_col_bounds(x_cols)
    np.testing.assert_array_equal(lo1, lo0)
    np.testing.assert_array_equal(hi1, hi0)
    assert master._coupling_box_saved is None
    # Double clear is a safe no-op.
    clear_box(master)
    assert master._coupling_box_saved is None

    # (b) NEVER WIDEN: a huge radius must clamp back to the original bounds,
    # not open them up ([−95,105] ∩ [1,10],[2,8] = the original box).
    apply_box(master, center, 100.0)
    wlo, whi = wp.get_col_bounds(x_cols)
    np.testing.assert_array_equal(wlo, lo0)
    np.testing.assert_array_equal(whi, hi0)
    clear_box(master)
    np.testing.assert_array_equal(wp.get_col_bounds(x_cols)[0], lo0)
    np.testing.assert_array_equal(wp.get_col_bounds(x_cols)[1], hi0)

    # (c) SAVE/RESTORE-SAFE across a second apply/clear cycle: re-reads the
    # (restored) original bounds, so the box is re-derived from lo0/hi0.
    apply_box(master, center, 2.0)
    np.testing.assert_array_equal(wp.get_col_bounds(x_cols)[0], [3.0, 3.0])
    clear_box(master)
    np.testing.assert_array_equal(wp.get_col_bounds(x_cols)[0], lo0)
    np.testing.assert_array_equal(wp.get_col_bounds(x_cols)[1], hi0)


def test_trust_region_converges_master_hosted(mh_data, monolith, monkeypatch) -> None:
    """End-to-end: with the trust-region stabilizer ON (machine-local env,
    in-out OFF ⇒ λ=0), the master-hosted decomposition still converges to the
    monolith optimum with a VALID lower bound — proving the coupling-box
    primitives drive the coordinator's two-solve trust-region path correctly."""
    monkeypatch.setenv("FLEXTOOL_BENDERS_TRUST_REGION_RADIUS", "1.0")
    M = monolith.obj
    res = solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4,
        monolith_objective=M, scale_the_objective=OBJ_SCALE,
    )
    assert res.converged, (
        f"Benders (trust region, master-hosted) did not converge: "
        f"gap={res.gap:.3e} after {res.iterations} iters "
        f"(LB={res.lower_bound:.6e} UB={res.upper_bound:.6e})"
    )
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"Benders trust-region UB {res.total_objective:.8e} != monolith M "
        f"{M:.8e} (LB={res.lower_bound:.8e}, gap={res.gap:.3e})"
    )
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"Benders trust-region LB {res.lower_bound:.8e} EXCEEDS M {M:.8e}"
    )


def test_trust_region_and_in_out_mutually_exclusive(mh_data, monkeypatch) -> None:
    """Selecting BOTH the trust region (env) and in-out (λ>0) is rejected up
    front with a clear FlexTool-side error — the two stabilizers are mutually
    exclusive."""
    monkeypatch.setenv("FLEXTOOL_BENDERS_TRUST_REGION_RADIUS", "1.0")
    with pytest.raises(ValueError, match=r"MUTUALLY EXCLUSIVE"):
        solve_benders(
            mh_data, REGIONS_AC, max_iters=5, tol=1e-4,
            scale_the_objective=OBJ_SCALE, in_out_weight=0.5,
        )


# ---------------------------------------------------------------------------
# (vii) Authored-data validation reaches the user through the driver.
# ---------------------------------------------------------------------------


def test_straddling_unit_raises_through_driver(mh_data) -> None:
    """Hosting ONLY lh2_B (3-region run, membership surgery) makes
    liquefier_B straddle: its input node h2_B stays in region_B while
    its output node lh2_B is hosted.  The split-level hard error (D-a)
    must propagate out of ``solve_benders`` naming the unit and the
    handover pattern."""
    data = _host_nodes(mh_data, frozenset({"lh2_B"}), REGIONS_ABC)
    with pytest.raises(RuntimeError, match=r"unit 'liquefier_B' straddles"):
        solve_benders(data, REGIONS_ABC, max_iters=5, tol=1e-4)


def test_mixed_resolution_coupling_raises_through_driver(_workdirs) -> None:
    """On the DAILY-block sibling (``lh2_master_build``), hosting the
    A/B carrier chains makes every electrolyser coupling arc join an
    hourly node (elec) to a daily-block node (h2) — a mixed-resolution
    boundary the single-sided half-flow pin cannot represent exactly
    (it silently relaxes the model).  The driver must reject it up
    front with the plain-English diagnostic pointing at the handover /
    same-time-aggregation pattern."""
    mb_data = load_flextool(_workdirs[2])
    data = _host_nodes(
        mb_data, frozenset({"h2_A", "lh2_A", "h2_B", "lh2_B"}), REGIONS_ABC
    )
    with pytest.raises(RuntimeError) as excinfo:
        solve_benders(data, REGIONS_ABC, max_iters=5, tol=1e-4)
    msg = str(excinfo.value)
    assert "different time resolutions" in msg
    assert "electrolyser" in msg
    assert "What this means:" in msg
    assert "new_stepduration" in msg


def test_in_out_with_master_hosted_converges_to_monolith(mh_data, monolith) -> None:
    """λ>0 in-out separation IS supported with master-hosted nodes: the
    coordinator scores the upper bound at one consistent overlay point
    (``native_cost_at``) instead of the invalid mixed sum, so the run
    produces a VALID bound and still converges to the monolith optimum.

    ``_BendersMaster.native_cost_flow_dependent`` is True here (the master
    hosts the lh2_B/elec_B balances), so the coordinator takes the
    consistent-point UB path; the old up-front rejection is gone."""
    M = monolith.obj
    res = solve_benders(
        mh_data, REGIONS_AC, max_iters=20, tol=1e-4,
        monolith_objective=M, scale_the_objective=OBJ_SCALE,
        in_out_weight=0.5,
    )
    assert res.converged, (
        f"Benders (in-out λ=0.5, master-hosted) did not converge: "
        f"gap={res.gap:.3e} after {res.iterations} iters "
        f"(LB={res.lower_bound:.6e} UB={res.upper_bound:.6e})"
    )
    assert np.isclose(res.total_objective, M, rtol=1e-4), (
        f"Benders in-out UB {res.total_objective:.8e} != monolith M "
        f"{M:.8e} (LB={res.lower_bound:.8e}, gap={res.gap:.3e})"
    )
    # VALID lower bound: the consistent-point UB keeps LB <= M.
    assert res.lower_bound <= M * (1 + 1e-9), (
        f"Benders in-out LB {res.lower_bound:.8e} EXCEEDS M {M:.8e}"
    )


def test_missing_penalty_raises_plain_english_via_driver(mh_data) -> None:
    """A master-hosted balance node with NO penalty_up rows must fail on
    the extended F9 precondition with the three-section plain-English
    diagnostic — BEFORE any master solve can hit a raw solver error."""
    up = mh_data.p_penalty_up
    filtered = up.frame.filter(pl.col("n").cast(pl.Utf8) != "h2_B")
    data = _copy_with(mh_data, p_penalty_up=Param(up.dims, filtered))
    with pytest.raises(RuntimeError) as excinfo:
        solve_benders(
            data, REGIONS_AC, max_iters=5, tol=1e-4,
            scale_the_objective=OBJ_SCALE,
        )
    msg = str(excinfo.value)
    assert "'h2_B'" in msg
    assert "penalty_up" in msg
    assert "What this means:" in msg
    assert "How to avoid it:" in msg


# ---------------------------------------------------------------------------
# (viii) F9 precondition unit level: ABSENT rows and INFINITE values both
# fire, for BOTH penalty params; clean data passes.
# ---------------------------------------------------------------------------


def test_penalty_precondition_clean_data_passes(mh_data) -> None:
    _assert_finite_boundary_penalties(
        mh_data, [], master_hosted_nodes=MASTER_B_SUBSYSTEM
    )


def test_penalty_precondition_absent_rows_fire(mh_data) -> None:
    down = mh_data.p_penalty_down
    filtered = down.frame.filter(pl.col("n").cast(pl.Utf8) != "elec_B")
    data = _copy_with(
        mh_data, p_penalty_down=Param(down.dims, filtered)
    )
    with pytest.raises(RuntimeError, match=r"'elec_B'.*penalty_down") as ei:
        _assert_finite_boundary_penalties(
            data, [], master_hosted_nodes=MASTER_B_SUBSYSTEM
        )
    assert "no value at all" in str(ei.value)


def test_penalty_precondition_whole_param_absent_fires(mh_data) -> None:
    data = _copy_with(mh_data, p_penalty_up=None)
    with pytest.raises(RuntimeError, match=r"penalty_up"):
        _assert_finite_boundary_penalties(
            data, [], master_hosted_nodes=MASTER_B_SUBSYSTEM
        )


def test_penalty_precondition_infinite_value_fires(mh_data) -> None:
    up = mh_data.p_penalty_up
    inf_frame = up.frame.with_columns(
        pl.when(pl.col("n").cast(pl.Utf8) == "elec_B")
        .then(pl.lit(float("inf")))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    data = _copy_with(mh_data, p_penalty_up=Param(up.dims, inf_frame))
    with pytest.raises(RuntimeError, match=r"'elec_B'.*penalty_up") as ei:
        _assert_finite_boundary_penalties(
            data, [], master_hosted_nodes=MASTER_B_SUBSYSTEM
        )
    assert "not finite" in str(ei.value)


def test_penalty_precondition_empty_set_is_boundary_only(mh_data) -> None:
    """With the default empty master set the function is byte-identical
    to the historical boundary-only check: no penalty rows at all on a
    (fabricated) boundary node keeps today's SILENT skip."""
    data = _copy_with(mh_data, p_penalty_up=None, p_penalty_down=None)
    _assert_finite_boundary_penalties(data, [])


# ---------------------------------------------------------------------------
# Stall rendering: the master pseudo-entry is named "the master-hosted
# nodes", never a fake node-group name.
# ---------------------------------------------------------------------------


def test_stall_rendering_names_master_hosted_nodes() -> None:
    exc = ph_benders.BendersStalled(
        "stalled (synthetic)",
        iteration=9,
        gap=1.02,
        tol=1e-4,
        window=8,
        reference_scale=1.5e9,
        sub_costs={"decomp_X": 2.0e8},
        sub_reference_costs={"decomp_X": 1.0e8},
    )
    with pytest.raises(RuntimeError) as ei:
        fx_benders._raise_stalled(
            exc,
            master_autarky=5.0e9,       # dominates ⇒ ROOT
            master_native_cost=9.0e12,  # dominates ⇒ SYMPTOM too
        )
    msg = str(ei.value)
    assert "The master-hosted nodes are the likely cause" in msg
    assert "the master-hosted nodes" in msg
    assert "'master-hosted nodes'" not in msg  # never a fake group name

    # Without master values the rendering is the historical one.
    with pytest.raises(RuntimeError) as ei2:
        fx_benders._raise_stalled(exc)
    assert "Node group 'decomp_X' is the likely cause" in str(ei2.value)
