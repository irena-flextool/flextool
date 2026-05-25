"""Convex weight fitting and simplex projection.

Implements the weight computation from the hull clustering paper:
for each base period, find the convex combination of representative periods
that best approximates it (minimum L2 error).
"""

import numpy as np


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w >= 0, sum(w) = 1}.

    Uses Condat's algorithm (O(n) average case).

    Args:
        v: Input vector of length n.

    Returns:
        Projected vector on the simplex.
    """
    n = len(v)
    if n == 0:
        return v
    if n == 1:
        return np.array([1.0])

    u = np.sort(v)[::-1]  # sort descending
    cumsum = np.cumsum(u)
    k_candidates = u + (1.0 - cumsum) / np.arange(1, n + 1)
    K = np.max(np.where(k_candidates > 0)[0]) + 1 if np.any(k_candidates > 0) else 1
    tau = (cumsum[K - 1] - 1.0) / K
    return np.maximum(v - tau, 0.0)


def fit_convex_weights(
    R: np.ndarray,
    c: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
    alpha: float | None = None,
) -> np.ndarray:
    """Find convex weights w that minimize ||R @ w - c||^2 s.t. w in simplex.

    Uses projected gradient descent with simplex projection.

    Args:
        R: Matrix of representative period feature vectors, shape (n_features, n_rp).
        c: Feature vector of the base period to approximate, shape (n_features,).
        max_iter: Maximum PGD iterations.
        tol: Convergence tolerance.
        alpha: Learning rate. If None, computed from Lipschitz constant.

    Returns:
        Weight vector w of length n_rp, with w >= 0 and sum(w) = 1.
    """
    n_rp = R.shape[1]
    if n_rp == 1:
        return np.array([1.0])

    # Precompute R^T R and R^T c for gradient computation
    RtR = R.T @ R
    Rtc = R.T @ c

    # Learning rate from Lipschitz constant of gradient
    if alpha is None:
        L = np.linalg.norm(RtR, ord=2)
        alpha = 1.0 / max(L, 1e-10)

    # Initial guess via pseudoinverse
    R_pinv = np.linalg.pinv(R)
    w = project_onto_simplex(R_pinv @ c)

    for _ in range(max_iter):
        # Gradient: R^T (R w - c) = RtR w - Rtc
        grad = RtR @ w - Rtc
        w_prev = w.copy()
        w = project_onto_simplex(w - alpha * grad)

        if np.max(np.abs(w - w_prev)) < tol:
            break

    return w


def distance_to_hull(R: np.ndarray, c: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """Compute the L2 distance from point c to the convex hull of columns of R.

    Args:
        R: Matrix of representative period feature vectors, shape (n_features, n_rp).
        c: Feature vector to measure distance from, shape (n_features,).

    Returns:
        (distance, weights): the L2 distance and the optimal weight vector.
    """
    w = fit_convex_weights(R, c, **kwargs)
    residual = R @ w - c
    dist = np.linalg.norm(residual)
    return dist, w


def compute_weight_matrix(
    C: np.ndarray, rep_indices: list[int]
) -> np.ndarray:
    """Compute the full weight matrix W for all base periods.

    Args:
        C: Clustering matrix, shape (n_features, n_base_periods).
        rep_indices: Indices of selected representative periods.

    Returns:
        Weight matrix W of shape (n_base_periods, n_rp),
        where W[d, r] is the weight of representative period r for base period d.
        Each row sums to 1 and all entries >= 0.
    """
    n_base = C.shape[1]
    n_rp = len(rep_indices)
    R = C[:, rep_indices]
    W = np.zeros((n_base, n_rp))

    for d in range(n_base):
        W[d, :] = fit_convex_weights(R, C[:, d])

    return W
