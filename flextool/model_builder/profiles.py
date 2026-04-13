"""Autocorrelated time series generation for profiles and demand."""

import numpy as np
from numpy.random import Generator


def pert_sample(rng: Generator, min_val: float, mode_val: float, max_val: float, size: int = 1) -> np.ndarray:
    """Sample from a PERT distribution using Beta distribution transformation.

    The PERT distribution is a Beta distribution rescaled to [min, max] with
    the mode controlling the shape.
    """
    if max_val == min_val:
        return np.full(size, min_val)
    # PERT shape parameter (lambda=4 is standard)
    lam = 4.0
    mu = (min_val + lam * mode_val + max_val) / (lam + 2)
    # Avoid degenerate cases
    if mu <= min_val or mu >= max_val:
        mu = np.clip(mu, min_val + 1e-10, max_val - 1e-10)
    alpha = (mu - min_val) / (max_val - min_val) * (
        (mu - min_val) * (max_val - mu) / ((max_val - min_val) ** 2 / (lam + 2 + 1)) - 1
    )
    # Simplified PERT alpha/beta calculation
    range_val = max_val - min_val
    alpha = 1 + lam * (mu - min_val) / range_val
    beta = 1 + lam * (max_val - mu) / range_val
    alpha = max(alpha, 0.1)
    beta = max(beta, 0.1)
    samples = rng.beta(alpha, beta, size=size)
    return min_val + samples * range_val


def pert_integer_sample(rng: Generator, min_val: int, mode_val: int, max_val: int, size: int = 1) -> np.ndarray:
    """Sample integers from a PERT distribution (round the continuous sample)."""
    continuous = pert_sample(rng, float(min_val), float(mode_val), float(max_val), size)
    return np.round(continuous).astype(int)


def generate_autocorrelated_series(
    rng: Generator,
    length: int,
    pattern_length_avg: float,
    pattern_length_std: float,
    dist_min: float,
    dist_mode: float,
    dist_max: float,
) -> np.ndarray:
    """Generate an autocorrelated time series using AR(1) with pattern-length control.

    The autocorrelation coefficient is derived from the desired pattern length:
    a higher pattern_length means values persist longer (stronger autocorrelation).

    Values are then scaled to approximately match the PERT distribution,
    with soft clipping (scale first, clip ~5% of values).

    Args:
        rng: numpy random generator
        length: number of timesteps
        pattern_length_avg: average duration of a pattern in timesteps (hours)
        pattern_length_std: standard deviation of pattern length
        dist_min: minimum value of the target distribution
        dist_mode: mode value of the target distribution
        dist_max: maximum value of the target distribution
    """
    if pattern_length_avg <= 0:
        pattern_length_avg = 1.0

    # AR(1) coefficient from pattern length: phi = exp(-1/pattern_length)
    # This gives autocorrelation that decays with the desired timescale
    if pattern_length_std > 0:
        # Vary the pattern length itself
        actual_pattern_length = max(
            1.0, rng.normal(pattern_length_avg, pattern_length_std)
        )
    else:
        actual_pattern_length = pattern_length_avg

    phi = np.exp(-1.0 / actual_pattern_length)
    phi = np.clip(phi, 0.0, 0.999)

    # Generate AR(1) process: x[t] = phi * x[t-1] + (1-phi^2)^0.5 * noise
    noise_scale = np.sqrt(1 - phi**2)
    series = np.empty(length)
    series[0] = rng.normal(0, 1)
    for t in range(1, length):
        # Optionally vary pattern length over time
        if pattern_length_std > 0 and t % int(actual_pattern_length) == 0:
            actual_pattern_length = max(
                1.0, rng.normal(pattern_length_avg, pattern_length_std)
            )
            phi = np.exp(-1.0 / actual_pattern_length)
            phi = np.clip(phi, 0.0, 0.999)
            noise_scale = np.sqrt(max(0, 1 - phi**2))
        series[t] = phi * series[t - 1] + noise_scale * rng.normal(0, 1)

    # Transform from standard normal to target PERT range
    # Use CDF-based approach: map percentiles to PERT distribution
    # First, convert to uniform [0,1] via normal CDF
    from scipy.stats import norm

    uniform = norm.cdf(series)

    # Map uniform to PERT range using inverse PERT CDF (via beta)
    lam = 4.0
    range_val = dist_max - dist_min
    if range_val == 0:
        return np.full(length, dist_min)

    mu = (dist_min + lam * dist_mode + dist_max) / (lam + 2)
    mu = np.clip(mu, dist_min + 1e-10, dist_max - 1e-10)
    alpha = 1 + lam * (mu - dist_min) / range_val
    beta_param = 1 + lam * (dist_max - mu) / range_val
    alpha = max(alpha, 0.1)
    beta_param = max(beta_param, 0.1)

    from scipy.stats import beta as beta_dist

    result = beta_dist.ppf(uniform, alpha, beta_param) * range_val + dist_min

    # Soft clipping: allow ~5% of values outside range, clip the rest
    clip_margin = 0.0  # Already mapped to range via CDF, minimal clipping needed
    result = np.clip(result, dist_min, dist_max)

    return result


def generate_constant_series(value: float, length: int) -> np.ndarray:
    """Generate a constant time series."""
    return np.full(length, value)
