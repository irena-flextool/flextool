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
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._benders import (
    _BENDERS_IN_OUT_WEIGHT_ENV,
    _cut_separates,
    _resolve_benders_in_out_weight,
    solve_benders,
)

_REGIONS = ["region_A", "region_B", "region_C"]


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


@pytest.mark.parametrize("weight", ["0.3", "0.5", "0.7"])
def test_in_out_on_converges_to_same_optimum(
    ti_data, monolith_obj, monkeypatch, weight
):
    """With in-out ON (λ>0) the cuts are still VALID supporting hyperplanes, so
    the loop converges to the SAME monolith optimum with a valid lower bound —
    the optimum is unchanged, only the cut-generation point moves (plan §5)."""
    monkeypatch.setenv(_BENDERS_IN_OUT_WEIGHT_ENV, weight)
    res = _solve(ti_data, monolith_obj)
    assert res.converged, (
        f"in-out (λ={weight}) did not converge: gap={res.gap:.3e} after "
        f"{res.iterations} iters"
    )
    assert np.isclose(res.total_objective, monolith_obj, rtol=1e-4), (
        f"in-out (λ={weight}) UB {res.total_objective:.8e} != monolith "
        f"{monolith_obj:.8e}"
    )
    # Valid lower bound preserved (the key correctness invariant).
    assert res.lower_bound <= monolith_obj * (1 + 1e-9), (
        f"in-out (λ={weight}) LB {res.lower_bound:.8e} EXCEEDS monolith "
        f"{monolith_obj:.8e} — invalid bound"
    )
