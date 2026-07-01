"""Regression: the post-master cut self-check (``_check_cuts_satisfied``) must
key its tolerance off the cut ROW's coefficient magnitude, not the (possibly
heavily cancelled) rhs.

Real failure (7-region H2-trade ``lt_rp_only_lagrangian``, obj_scale=1e-6):
early-iteration recourse overshoot spiked one node group's cost to ~2.66e6
(scaled), giving huge reduced-cost slopes.  ``cost_r`` and ``Σ g·f̄`` then
NEARLY CANCEL — the cut rhs collapsed to O(1) (3.8548) while the row
coefficients stayed at O(1e6).  The cut is a literal row in the master LP, so
the master already enforces ``η ≥ rhs`` to solver feasibility tolerance; the
residual 7.09e-4 gap is pure round-off on that ill-conditioned row (~2.7e-10
relative to the row scale).  The old ``tol_abs = 1e-5·max(1,|rhs|,|er|)`` keyed
off the cancelled rhs (~3.85) and demanded ~3.85e-5, hard-failing on noise.

A GROSS violation (an un-appended cut leaves η near its large-negative floor)
must still hard-fail with a plain-English diagnostic.
"""
from __future__ import annotations

import logging

import pytest

from flextool.engine_polars._benders import _check_cuts_satisfied


# The exact failing cell from the N=7 run, reconstructed as a single-cell cut
# whose cost_r and slope·f̄ cancel down to an O(1) rhs.
_COST_R = 2.661232e6          # scaled recourse cost (2.66123e12 × obj_scale 1e-6)
_SLOPE = 1.0e6               # large reduced-cost slope from the overshoot solve
_F_BAR = {0: 3.0}
# Pick new_f̄ so rhs = cost_r + slope·(new − old) ≈ 3.8548 (heavy cancellation).
_RHS_TARGET = 3.8548299698
_NEW_F_BAR = {0: _F_BAR[0] + (_RHS_TARGET - _COST_R) / _SLOPE}


def _rhs(cost_r, slopes, f_bar, new_f_bar):
    return cost_r + sum(g * (new_f_bar[c] - f_bar[c]) for c, g in slopes.items())


def test_cancelled_cut_within_solver_slack_passes():
    """The 7.09e-4 gap on a 1e6-scale row is round-off — must NOT raise."""
    slopes = {0: _SLOPE}
    rhs = _rhs(_COST_R, slopes, _F_BAR, _NEW_F_BAR)
    # eta sits just below rhs by the observed real-run margin.
    eta = {"decomp_MID": rhs - 7.09e-4}
    # Sanity: this is exactly the case the OLD |rhs|-keyed tolerance rejected.
    assert (rhs - eta["decomp_MID"]) > 1e-5 * max(1.0, abs(rhs))
    _check_cuts_satisfied(
        [("decomp_MID", _COST_R, slopes)], _F_BAR, _NEW_F_BAR, eta,
        iterations=2, inv_s=1e6,
    )  # must not raise


def test_within_row_scale_tolerance_is_silent(caplog):
    """A gap well under 1e-6·row_scale passes without even a warning."""
    slopes = {0: _SLOPE}
    rhs = _rhs(_COST_R, slopes, _F_BAR, _NEW_F_BAR)
    eta = {"decomp_MID": rhs - 7.09e-4}
    with caplog.at_level(logging.WARNING):
        _check_cuts_satisfied(
            [("decomp_MID", _COST_R, slopes)], _F_BAR, _NEW_F_BAR, eta,
            iterations=2, inv_s=1e6,
        )
    assert not caplog.records


def test_moderate_violation_warns_and_continues(caplog):
    """A gap between the pass tolerance and the gross band is flagged but does
    not abort the solve (the LB/sandwich guards still bracket the optimum)."""
    slopes = {0: _SLOPE}
    rhs = _rhs(_COST_R, slopes, _F_BAR, _NEW_F_BAR)
    # row_scale ≈ 6e6 → tol_abs ≈ 6, gross_tol ≈ 6e4.  Pick a gap inside (6, 6e4).
    eta = {"decomp_MID": rhs - 100.0}
    with caplog.at_level(logging.WARNING):
        _check_cuts_satisfied(
            [("decomp_MID", _COST_R, slopes)], _F_BAR, _NEW_F_BAR, eta,
            iterations=2, inv_s=1e6,
        )  # must not raise
    assert any("under-satisfied" in r.message for r in caplog.records)


def test_gross_violation_hard_fails_with_diagnostic():
    """An un-appended cut leaves η near its large-negative floor: a violation
    orders beyond any solver slack must still abort with a 3-section message."""
    slopes = {0: _SLOPE}
    eta = {"decomp_MID": -3.0e6}  # η pinned near floor, not honouring the cut
    with pytest.raises(RuntimeError) as exc:
        _check_cuts_satisfied(
            [("decomp_MID", _COST_R, slopes)], _F_BAR, _NEW_F_BAR, eta,
            iterations=2, inv_s=1e6,
        )
    msg = str(exc.value)
    assert "What this means:" in msg
    assert "How to avoid it:" in msg
    assert "node group 'decomp_MID'" in msg


def test_non_finite_eta_hard_fails_with_diagnostic():
    slopes = {0: _SLOPE}
    eta = {"decomp_MID": float("inf")}
    with pytest.raises(RuntimeError) as exc:
        _check_cuts_satisfied(
            [("decomp_MID", _COST_R, slopes)], _F_BAR, _NEW_F_BAR, eta,
            iterations=2, inv_s=1e6,
        )
    msg = str(exc.value)
    assert "not a finite number" in msg
    assert "What this means:" in msg
