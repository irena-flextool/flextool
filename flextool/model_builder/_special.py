"""Numpy/stdlib special functions used by the profile generator.

FlexTool does not depend on scipy.  The synthetic-profile generator
(:mod:`flextool.model_builder.profiles`) needs only two special functions
to map an AR(1) standard-normal series to a PERT (beta) range:

* :func:`norm_cdf` -- the standard normal CDF Phi(x);
* :func:`beta_ppf` -- the inverse of the regularized incomplete beta
  function I_x(a, b), i.e. the beta-distribution quantile.

Both are vectorised over the probability / abscissa argument; the shape
parameters ``a`` and ``b`` are scalars.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_HALF = 1.0 / math.sqrt(2.0)
_erfc_ufunc = np.frompyfunc(math.erfc, 1, 1)

# Continued-fraction settings (Numerical Recipes ``betacf``).
_CF_MAX_ITER = 300
_CF_EPS = 1e-16
_CF_FPMIN = 1e-300

# Inversion settings.
_INV_MAX_ITER = 400
_INV_TOL_ABS = 1e-12
_INV_TOL_REL = 1e-15  # bracket collapse (relative)
_INV_TOL_RATIO = 1e-10  # relative step tolerance for tiny roots


def norm_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF, Phi(x) = 0.5 * erfc(-x / sqrt(2)).

    The complementary-error form keeps full relative accuracy in the
    lower tail (where ``0.5 * (1 + erf(.))`` would cancel).  ``math.erfc``
    is applied element-wise; for the series lengths used by the profile
    generator (a year of hourly steps) this costs about a millisecond.
    """
    arr = np.asarray(x, dtype=float)
    return 0.5 * np.asarray(_erfc_ufunc(-arr * _SQRT_HALF), dtype=float)


def _betacf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Continued fraction of I_x(a, b) by modified Lentz (vectorised).

    Converges fast for ``x < (a + 1) / (a + b + 2)``; callers use the
    symmetry I_x(a, b) = 1 - I_{1-x}(b, a) otherwise.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _CF_FPMIN, _CF_FPMIN, d)
    d = 1.0 / d
    h = d.copy()
    done = np.zeros(x.shape, dtype=bool)
    for m in range(1, _CF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _CF_FPMIN, _CF_FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _CF_FPMIN, _CF_FPMIN, c)
        d = 1.0 / d
        h = np.where(done, h, h * d * c)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _CF_FPMIN, _CF_FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _CF_FPMIN, _CF_FPMIN, c)
        d = 1.0 / d
        delta = d * c
        h = np.where(done, h, h * delta)
        done |= np.abs(delta - 1.0) < _CF_EPS
        if done.all():
            break
    return h


def betainc(x: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Regularized incomplete beta function I_x(a, b), ``a, b > 0``.

    I_x(a, b) = x^a (1-x)^b / (a B(a, b)) * CF(x; a, b), with the
    prefactor in log space via :func:`math.lgamma` and the continued
    fraction from :func:`_betacf`.  Values outside [0, 1] are clipped.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"betainc needs a, b > 0 (got a={a}, b={b})")
    xa = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = np.empty_like(xa)
    out[xa <= 0.0] = 0.0
    out[xa >= 1.0] = 1.0
    inner = (xa > 0.0) & (xa < 1.0)
    if inner.any():
        xi = xa[inner]
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        log_front = a * np.log(xi) + b * np.log1p(-xi) - lbeta
        front = np.exp(log_front)
        direct = xi < (a + 1.0) / (a + b + 2.0)
        res = np.empty_like(xi)
        if direct.any():
            xd = xi[direct]
            res[direct] = front[direct] * _betacf(xd, a, b) / a
        if (~direct).any():
            xs = xi[~direct]
            res[~direct] = 1.0 - front[~direct] * _betacf(1.0 - xs, b, a) / b
        out[inner] = res
    return out


def _beta_log_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x) - lbeta


def _initial_guess(u: np.ndarray, a: float, b: float) -> np.ndarray:
    """Starting point for :func:`beta_ppf` from the tail asymptotics.

    Near 0, I_x(a, b) ~ x^a / (a B(a, b)); near 1, 1 - I_x(a, b) ~
    (1-x)^b / (b B(a, b)).  The branch is picked by comparing ``u`` with
    I at the mean, and the guess is clipped into the open interval.
    """
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    mean = a / (a + b)
    split = float(betainc(np.array([mean]), a, b)[0])
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        x_lo = np.exp((np.log(u) + math.log(a) + lbeta) / a)
        x_hi = 1.0 - np.exp((np.log1p(-u) + math.log(b) + lbeta) / b)
    guess = np.where(u < split, np.minimum(x_lo, mean), np.maximum(x_hi, mean))
    tiny = np.finfo(float).tiny
    return np.clip(guess, tiny, np.nextafter(1.0, 0.0))


def beta_ppf(u: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Beta-distribution quantile: the x in [0, 1] with I_x(a, b) = u.

    Safeguarded Newton iteration on I_x(a, b) - u, vectorised over ``u``:
    a bracket [lo, hi] is shrunk every step using the sign of the
    residual, and any Newton step that leaves the bracket (or has a
    non-finite derivative) is replaced by bisection.  Iterates until
    ``|dx| < min(1e-12, 1e-10 * x)`` or the bracket collapses.
    ``u <= 0`` maps to 0 and ``u >= 1`` to 1.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"beta_ppf needs a, b > 0 (got a={a}, b={b})")
    ua = np.asarray(u, dtype=float)
    shape = ua.shape
    uf = ua.ravel()
    out = np.where(uf >= 1.0, 1.0, 0.0)
    active_idx = np.nonzero((uf > 0.0) & (uf < 1.0))[0]
    if active_idx.size == 0:
        return out.reshape(shape)

    target = uf[active_idx]
    lo = np.zeros_like(target)
    hi = np.ones_like(target)
    x = _initial_guess(target, a, b)
    idx = np.arange(target.size)
    for _ in range(_INV_MAX_ITER):
        xi = x[idx]
        resid = betainc(xi, a, b) - target[idx]
        below = resid < 0.0
        lo[idx] = np.where(below, xi, lo[idx])
        hi[idx] = np.where(below, hi[idx], xi)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pdf = np.exp(_beta_log_pdf(xi, a, b))
            x_new = xi - resid / pdf
        bad = ~np.isfinite(x_new) | (x_new < lo[idx]) | (x_new > hi[idx])
        # Bisect geometrically while the bracket spans decades above zero
        # (tiny roots for small ``a``), arithmetically otherwise.
        lo_i, hi_i = lo[idx], hi[idx]
        with np.errstate(invalid="ignore"):
            geo = np.sqrt(lo_i * hi_i)
        mid = np.where((lo_i > 0.0) & (hi_i > 4.0 * lo_i), geo, 0.5 * (lo_i + hi_i))
        x_new = np.where(bad, mid, x_new)
        dx = np.abs(x_new - xi)
        x[idx] = x_new
        # Absolute 1e-12, tightened to relative for roots near zero.
        tol = np.minimum(_INV_TOL_ABS, _INV_TOL_RATIO * x_new)
        converged = (dx < tol) | (resid == 0.0) | ((hi[idx] - lo[idx]) <= _INV_TOL_REL * hi[idx])
        idx = idx[~converged]
        if idx.size == 0:
            break
    out[active_idx] = x
    return out.reshape(shape)
