"""Numpy-only special functions behind the synthetic profile generator.

Reference values are closed forms or published constants, never output of
the code under test.  scipy is deliberately not imported.
"""

import math

import numpy as np
import pytest

from flextool.model_builder._special import beta_ppf, betainc, norm_cdf
from flextool.model_builder.profiles import generate_autocorrelated_series


def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


U = np.concatenate([[1e-9, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-6], np.linspace(0.001, 0.999, 101)])


def test_norm_cdf_known_values():
    got = norm_cdf(np.array([0.0, 1.96, -1.0, 1.0]))
    assert got[0] == 0.5
    assert got[1] == pytest.approx(0.9750021048517795, abs=1e-15)
    assert got[2] == pytest.approx(0.15865525393145707, abs=1e-15)
    assert got[3] == pytest.approx(1.0 - 0.15865525393145707, abs=1e-15)
    # Lower tail keeps relative accuracy: Phi(-10) = 7.619853024160527e-24.
    assert norm_cdf(-10.0) == pytest.approx(7.619853024160527e-24, rel=1e-12)


@pytest.mark.parametrize(
    ("a", "b", "closed_form"),
    [
        (1.0, 1.0, lambda u: u),
        (2.0, 1.0, np.sqrt),
        (1.0, 2.0, lambda u: 1.0 - np.sqrt(1.0 - u)),
    ],
)
def test_beta_ppf_closed_forms(a, b, closed_form):
    np.testing.assert_allclose(beta_ppf(U, a, b), closed_form(U), rtol=0, atol=1e-12)


@pytest.mark.parametrize("a", [0.1, 0.5, 1.0, 2.5, 5.0])
def test_beta_ppf_symmetric_median(a):
    assert beta_ppf(0.5, a, a) == pytest.approx(0.5, abs=1e-12)


def test_beta_ppf_a2_b3_against_polynomial_cdf():
    # I_x(2, 3) = 6x^2 - 8x^3 + 3x^4; invert by plain bisection here.
    def cdf(x):
        return 6 * x**2 - 8 * x**3 + 3 * x**4

    for u in [0.01, 0.2, 0.5, 0.8, 0.99]:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cdf(mid) < u else (lo, mid)
        assert beta_ppf(u, 2.0, 3.0) == pytest.approx(0.5 * (lo + hi), abs=1e-12)
        assert betainc(0.5 * (lo + hi), 2.0, 3.0) == pytest.approx(u, abs=1e-13)


@pytest.mark.parametrize(("a", "b"), [(0.1, 0.1), (0.1, 5.0), (5.0, 0.1), (1.3, 4.7), (3.0, 3.0), (4.9, 1.1)])
def test_beta_round_trip(a, b):
    x = beta_ppf(U, a, b)
    assert np.all((x >= 0.0) & (x <= 1.0))
    assert np.all(np.diff(beta_ppf(np.sort(U), a, b)) >= 0.0)
    # Round trip I(ppf(u)) = u to 1e-10, widened only by the conditioning
    # of the root: a few ulps of x times the density.  With b < 1 the
    # upper quantiles sit closer to 1 than a double resolves
    # (1 - x ~ ((1 - u) b B(a, b))^(1/b)); those must have reached 1 - 1e-12.
    resolvable = x < 1.0 - 1e-12
    assert resolvable.sum() > 0.8 * U.size
    xr, ur = x[resolvable], U[resolvable]
    pdf = np.exp((a - 1) * np.log(xr) + (b - 1) * np.log1p(-xr) - _lbeta(a, b))
    tol = 1e-10 + 4.0 * pdf * np.spacing(xr)
    assert np.all(np.abs(betainc(xr, a, b) - ur) <= tol)
    tail = ((1.0 - U[~resolvable]) * b * np.exp(_lbeta(a, b))) ** (1.0 / b)
    assert np.all(tail < 1e-11)


def test_beta_ppf_edges():
    np.testing.assert_array_equal(beta_ppf(np.array([0.0, 1.0]), 2.0, 3.0), [0.0, 1.0])


def test_generated_profile_shape():
    # Same seed -> identical series; values stay within the PERT range and
    # the median sits near the PERT median (mode 0.3 in [0, 1] -> ~0.35).
    a = generate_autocorrelated_series(np.random.default_rng(7), 8760, 24, 6, 0.0, 0.3, 1.0)
    b = generate_autocorrelated_series(np.random.default_rng(7), 8760, 24, 6, 0.0, 0.3, 1.0)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert 0.25 < np.median(a) < 0.45
    # Strong lag-1 autocorrelation for a 24 h pattern length.
    assert np.corrcoef(a[:-1], a[1:])[0, 1] > 0.8
