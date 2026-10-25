"""Phase-1 in-out separation stabilization for spatial Benders.

Covers the FlexTool-side wiring of :class:`polar_high.InOutStabilizer` behind
the ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` env knob (default ``0.0`` = OFF):

* the env resolver ``_resolve_benders_in_out_weight`` (default + reject paths,
  mirroring ``_resolve_benders_max_stall``);
* the pure per-region separation predicate ``_cut_separates`` (the load-bearing
  row-scale tolerance);
* BYTE-PARITY: with ``weight=0.0`` (the default) a Benders solve produces the
  IDENTICAL objective/bound/iteration-count as with the env explicitly ``"0.0"``
  and reconciles to the monolith optimum — i.e. the ``if weight > 0.0`` guard
  makes the in-out block a true no-op.

See ``specs/benders_in_out_stabilization_plan.md``.
"""

from __future__ import annotations

import logging

import numpy as np
import polar_high.benders
import polar_high.decomposition
import polars as pl
import pytest

from polar_high import Problem

from flextool.engine import build_flextool, load_flextool
from flextool.engine.solve._benders import (
    _BENDERS_IN_OUT_WEIGHT_ENV,
    _cut_separates,
    _resolve_benders_in_out_weight,
    solve_benders,
)

_REGIONS = ["region_A", "region_B", "region_C"]

# The in-out upper-bound fix (#114) lives in polar-high (one value per shared coupling column, UB scored at that
# point).  CI installs polar-high from PyPI; until the release carrying it is pinned in pyproject.toml these tests
# run only against a polar-high that has it.  Feature detection, not a version check: an editable install keeps
# the old version metadata until it is reinstalled.
_needs_in_out_ub_fix = pytest.mark.skipif(
    not hasattr(polar_high.decomposition, "consistent_separation_points"),
    reason="needs polar-high >= 3.10.0 (#114)",
)


# ---------------------------------------------------------------------------
# Env resolver — default + reject paths (mirrors _resolve_benders_max_stall).
# ---------------------------------------------------------------------------


def test_resolver_default_is_zero(monkeypatch):
    """Unset ⇒ 0.0 (OFF), exactly."""
    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    assert _resolve_benders_in_out_weight() == 0.0


def test_resolver_empty_is_zero(monkeypatch):
    """Empty string is falsy ⇒ default 0.0."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, "")
    assert _resolve_benders_in_out_weight() == 0.0


@pytest.mark.parametrize("val,expected", [("0.0", 0.0), ("0.5", 0.5), ("0.3", 0.3), ("0.999", 0.999)])
def test_resolver_accepts_valid_range(monkeypatch, val, expected):
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, val)
    assert _resolve_benders_in_out_weight() == pytest.approx(expected)


def test_resolver_rejects_non_float_with_warning(monkeypatch, caplog):
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_in_out_weight() == 0.0
    assert any("non-float" in r.message for r in caplog.records)


@pytest.mark.parametrize("val", ["1.0", "1.5", "2.0"])
def test_resolver_rejects_weight_ge_one_with_warning(monkeypatch, caplog, val):
    """λ ≥ 1 never queries the master ⇒ non-convergent: IGNORED with a warning,
    default 0.0 used (NOT silently clamped)."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, val)
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_in_out_weight() == 0.0
    assert any("out-of-range" in r.message for r in caplog.records)


@pytest.mark.parametrize("val", ["-0.1", "-1.0"])
def test_resolver_rejects_weight_lt_zero_with_warning(monkeypatch, caplog, val):
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, val)
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_in_out_weight() == 0.0
    assert any("out-of-range" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DB-value / env precedence (the v63 solve.benders_in_out_weight param).
# ---------------------------------------------------------------------------


def test_resolver_uses_db_value_when_env_unset(monkeypatch):
    """No env ⇒ the per-solve DB value flows through unchanged."""
    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    assert _resolve_benders_in_out_weight(0.4) == pytest.approx(0.4)


def test_resolver_env_overrides_db_value(monkeypatch):
    """A valid env value OVERRIDES the DB value (machine-local precedence)."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, "0.7")
    assert _resolve_benders_in_out_weight(0.2) == pytest.approx(0.7)


def test_resolver_invalid_env_falls_back_to_db_value(monkeypatch, caplog):
    """A malformed / out-of-range env is IGNORED (warned) ⇒ DB value used."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_in_out_weight(0.3) == pytest.approx(0.3)
    assert any("non-float" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Pure per-region separation predicate (the load-bearing row-scale tolerance).
# ---------------------------------------------------------------------------


def test_cut_separates_true_when_master_underestimates():
    """A cut whose value at f_out is well ABOVE the master η (by more than the
    row-scale tolerance) separates."""
    slopes = {0: 2.0}
    f_out = {0: 10.0}
    f_sep = {0: 5.0}
    # cut_val = 3.0 + 2.0*(10 - 5) = 13.0; η = 4.0 ⇒ separates.
    assert _cut_separates(3.0, slopes, f_out, f_sep, eta_r=4.0)


def test_cut_separates_false_when_master_meets_cut():
    """η at/above the cut value ⇒ the master already honours it ⇒ NO separation
    ⇒ forces an exact-Benders out-step next."""
    slopes = {0: 2.0}
    f_out = {0: 10.0}
    f_sep = {0: 5.0}
    cut_val = 3.0 + 2.0 * (10.0 - 5.0)  # 13.0
    assert not _cut_separates(3.0, slopes, f_out, f_sep, eta_r=cut_val)
    assert not _cut_separates(3.0, slopes, f_out, f_sep, eta_r=cut_val + 1.0)


def test_cut_separates_tolerance_absorbs_roundoff():
    """A round-off-scale excess over η does NOT count as separation — the
    load-bearing tolerance (a bare ``>`` would livelock the forced out-step)."""
    slopes = {0: 1.0e6}
    f_out = {0: 3.0000001}
    f_sep = {0: 3.0}
    cost_r = 2.661232e6
    cut_val = cost_r + 1.0e6 * (f_out[0] - f_sep[0])
    row_scale = abs(cost_r) + 1.0e6 * (abs(f_out[0]) + abs(f_sep[0]))
    tol_sep = 1e-6 * max(1.0, abs(cut_val), abs(cut_val), row_scale)
    # η just below the cut value by LESS than tol_sep ⇒ NOT separated.
    assert not _cut_separates(
        cost_r, slopes, f_out, f_sep, eta_r=cut_val - 0.5 * tol_sep
    )
    # η below by MORE than tol_sep ⇒ separated.
    assert _cut_separates(
        cost_r, slopes, f_out, f_sep, eta_r=cut_val - 2.0 * tol_sep
    )


def test_cut_separates_verbatim_out_step_never_separates_at_zero_gap():
    """On a forced out-step f_sep == f_out (λ=0), so cut_val == cost_r; if the
    master already matches it there is no separation, but a strict underestimate
    still separates (exact Benders)."""
    slopes = {0: 2.0}
    f = {0: 7.0}
    # f_sep == f_out ⇒ cut_val = cost_r = 5.0.
    assert _cut_separates(5.0, slopes, f, f, eta_r=1.0)     # η well below
    assert not _cut_separates(5.0, slopes, f, f, eta_r=5.0)  # η meets it


# ---------------------------------------------------------------------------
# End-to-end byte-parity: weight=0.0 (default) is a no-op.
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
def monolith_obj(ti_data):
    pb = Problem()
    build_flextool(pb, ti_data)
    sol = pb.solve()
    assert sol.optimal, "monolith solve not optimal"
    return float(sol.obj)


def _solve(ti_data, monolith_obj):
    return solve_benders(
        ti_data, _REGIONS, max_iters=20, tol=1e-4,
        monolith_objective=monolith_obj,
    )


def test_weight_zero_default_matches_explicit_zero_bit_for_bit(
    ti_data, monolith_obj, monkeypatch
):
    """The default (env unset) and an explicit ``"0.0"`` must produce the
    IDENTICAL objective, bound, and iteration count — proving the
    ``if weight > 0.0`` guard makes the in-out block a true no-op."""
    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    res_default = _solve(ti_data, monolith_obj)

    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, "0.0")
    res_explicit = _solve(ti_data, monolith_obj)

    # Bit-for-bit identical: the OFF path is byte-parity by construction.
    assert res_default.total_objective == res_explicit.total_objective
    assert res_default.lower_bound == res_explicit.lower_bound
    assert res_default.upper_bound == res_explicit.upper_bound
    assert res_default.iterations == res_explicit.iterations
    assert res_default.converged == res_explicit.converged


def test_weight_zero_converges_to_monolith(ti_data, monolith_obj, monkeypatch):
    """Sanity: the OFF path still converges to the monolith optimum with a
    valid lower bound (no regression from the in-out threading)."""
    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    res = _solve(ti_data, monolith_obj)
    assert res.converged, (
        f"OFF-path Benders did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters"
    )
    assert np.isclose(res.total_objective, monolith_obj, rtol=1e-4), (
        f"OFF-path UB {res.total_objective:.8e} != monolith "
        f"{monolith_obj:.8e}"
    )
    assert res.lower_bound <= monolith_obj * (1 + 1e-9), "OFF-path LB invalid"


# ---------------------------------------------------------------------------
# In-out ON on the re-tuned trade fixture (c15b review F3): region A's enlarged
# chain ships LH2 A→B→C over both pipes, so λ=0 needs several iterations and
# the per-region stabilizers genuinely diverge.  Each pipe's flow column is
# pinned in its exporter AND its importer region.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retuned_data(scenario_workdir):
    return load_flextool(scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest_retuned"))


@pytest.fixture(scope="module")
def retuned_monolith(retuned_data):
    pb = Problem()
    build_flextool(pb, retuned_data)
    sol = pb.solve()
    assert sol.optimal, "re-tuned monolith solve not optimal"
    return float(sol.obj)


@_needs_in_out_ub_fix
@pytest.mark.parametrize("weight", ["0.3", "0.5", "0.7"])
def test_in_out_on_converges_to_same_optimum(
    retuned_data, retuned_monolith, monkeypatch, weight
):
    """With in-out ON (λ>0) the cuts are still VALID supporting hyperplanes, so
    the loop converges to the SAME monolith optimum with a valid lower bound —
    the optimum is unchanged, only the cut-generation point moves (plan §5)."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, weight)
    res = _solve(retuned_data, retuned_monolith)
    assert res.converged, (
        f"in-out (λ={weight}) did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters"
    )
    assert np.isclose(res.total_objective, retuned_monolith, rtol=1e-4), (
        f"in-out (λ={weight}) UB {res.total_objective:.8e} != monolith "
        f"{retuned_monolith:.8e}"
    )
    # Valid lower bound preserved (the key correctness invariant).
    assert res.lower_bound <= retuned_monolith * (1 + 1e-9), (
        f"in-out (λ={weight}) LB {res.lower_bound:.8e} EXCEEDS monolith "
        f"{retuned_monolith:.8e} — invalid bound"
    )


@_needs_in_out_ub_fix
@pytest.mark.parametrize("weight", [0.3, 0.5, 0.7])
def test_in_out_upper_bound_is_valid(retuned_data, retuned_monolith, monkeypatch, weight):
    """The in-out upper bound is the cost of one feasible point, so it is never
    below the monolith optimum M (#114: before, λ 0.3 / 0.5 / 0.7 stopped on
    UBs 1.3e-4 / 3.1e-4 / 5.8e-5 below M, because the exporter and importer
    halves of a pipe were pinned at different flows), and at tol 1e-6 it is
    within 1e-6 of M."""
    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    m = retuned_monolith
    res = solve_benders(
        retuned_data, _REGIONS, max_iters=60, tol=1e-6, monolith_objective=m,
        scale_the_objective=1e-6, in_out_weight=weight,
    )
    assert res.converged, f"λ={weight}: gap {res.gap:.3e} after {res.iterations} iterations"
    assert res.upper_bound >= m * (1 - 1e-8), (
        f"λ={weight}: UB {res.upper_bound:.10e} below the monolith {m:.10e} (rel {(res.upper_bound - m) / m:+.2e})"
    )
    assert res.upper_bound - m <= 1e-6 * m


@_needs_in_out_ub_fix
@pytest.mark.skipif(not hasattr(polar_high.benders.BendersMaster, "pinned_solve_at"),
                    reason="needs polar-high >= 3.10.0 (#114)")
def test_in_out_handed_plan_costs_the_best_upper_bound(retuned_data, monkeypatch):
    """Every incumbent the dispatch hand-off would receive is the plan its upper bound priced: the master's own
    cost with the handed coupling investments fixed and the flows at the incumbent point, plus the region costs,
    is that iteration's upper bound.  Before, the plan came from the master's outer vertex (more capacity than
    the interior flows need), so it could cost more than the bound."""
    from flextool.engine.solve import _benders

    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    masters: list = []
    captured: list[tuple[dict, dict]] = []
    real_init, real_loop = _benders._BendersMaster.__init__, _benders.solve_benders_loop

    def init(self, *a, **k):
        real_init(self, *a, **k)
        masters.append(self)

    def loop(*a, on_incumbent=None, **k):
        def record(msol, results, info):
            payload = on_incumbent(msol, results, info)
            captured.append((payload, {"point": dict(info["point"]), "upper_bound": info["upper_bound"],
                                       "sub_costs": dict(info["sub_costs"])}))
            return payload
        return real_loop(*a, on_incumbent=record, **k)

    monkeypatch.setattr(_benders._BendersMaster, "__init__", init)
    monkeypatch.setattr(_benders, "solve_benders_loop", loop)
    res = solve_benders(retuned_data, _REGIONS, max_iters=60, tol=1e-6, scale_the_objective=1e-6,
                        in_out_weight=0.5)
    assert res.converged and captured
    master = masters[-1]
    wp = master._wp
    trade = sorted({a.conn for a in master.arcs})
    frame = wp._p._vars["v_invest_p"].frame.with_columns(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
    for payload, info in captured:
        plan = payload["master_invest_frames"]["v_invest_p"].with_columns(
            pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8))
        fixed = frame.join(plan.filter(pl.col("p").is_in(trade)), on=["p", "d"], how="inner")
        assert fixed.height > 0
        cols = fixed["col_id"].to_numpy().astype(np.int64)
        lo, hi = wp.get_col_bounds(cols)
        wp.set_col_bounds(cols, fixed["value"].to_numpy(), fixed["value"].to_numpy())
        try:
            plan_native = master.native_cost_at(info["point"])
        finally:
            wp.set_col_bounds(cols, lo, hi)
        plan_cost = plan_native + sum(info["sub_costs"].values())
        assert plan_cost == pytest.approx(info["upper_bound"], rel=1e-7)


def test_in_out_warns_when_polar_high_lacks_the_upper_bound_fix(ti_data, monkeypatch, capsys):
    """With a polar-high older than 3.10.0 (feature detection) an in-out run warns that its UB may be invalid;
    with the fix, or with λ = 0, it does not."""
    from flextool.engine.solve import _benders

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.delenv(_BENDERS_IN_OUT_WEIGHT_ENV, raising=False)
    monkeypatch.setattr(_benders, "solve_benders_loop", stop)
    for fixed, weight, expect in ((False, 0.5, True), (True, 0.5, False), (False, 0.0, False)):
        monkeypatch.setattr(_benders, "_IN_OUT_UB_FIXED", fixed)
        with pytest.raises(_Stop):
            solve_benders(ti_data, _REGIONS, in_out_weight=weight)
        out = capsys.readouterr().out
        assert ("upper bound below the optimum" in out) is expect, (fixed, weight, out[-500:])
