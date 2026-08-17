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


def compute_nearest_assignment(
    C: np.ndarray,
    rep_indices: list[int],
    k: int = 1,
    power: float = 1.0,
    freq_preserve: bool = False,
    adaptive_frac: float = 0.0,
) -> np.ndarray:
    """Compute a nearest-``k`` inverse-distance weight matrix W for base periods.

    Drop-in alternative to :func:`compute_weight_matrix` that ties each base
    period to its ``k`` nearest representative periods instead of a convex
    blend. With the default ``k == 1`` each base period is assigned
    (weight 1.0) to the representative period ``r`` minimizing the Euclidean
    distance ``||C[:, d] - R[:, r]||``, where ``R = C[:, rep_indices]``; all
    other entries in the row are 0.0 — a pure single-nearest 0/1 assignment.

    For ``k > 1`` each base period is split across its ``k`` nearest
    representative periods, weighted by inverse Euclidean distance
    ``w_i ∝ 1 / d_i`` and normalized so the row sums to exactly 1.0. This
    smooths the inter-period storage chain in hourly-RP experiments. A base
    period that coincides with a representative period (distance 0) is a
    special case: inverse distance is infinite, so all of the row's weight
    goes to the zero-distance rep(s) (split equally if several of the ``k``
    are at distance 0) and none is spread to the finite-distance reps.

    Args:
        C: Clustering matrix, shape (n_features, n_base_periods).
        rep_indices: Indices of selected representative periods.
        k: Number of nearest representative periods to blend per base period.
            ``1`` (default) reproduces the single-nearest 0/1 assignment
            byte-for-byte. Clamped down to ``n_rp`` if larger. Must be >= 1.
        power: Inverse-distance exponent ``p`` for the ``k > 1`` blend, so
            ``w_i ∝ 1 / d_i**p``. ``1.0`` (default) is plain inverse distance
            and is byte-identical to the pre-``power`` behaviour. ``p > 1``
            sharpens the kernel toward the single-nearest rep (an isolated,
            much-closer rep keeps almost all the weight — preserving extreme
            reps at amplitude while still interpolating between similar reps);
            ``p -> inf`` approaches ``k == 1``. No effect when ``k == 1`` (the
            assignment is 0/1 regardless of ``power``). Must be > 0.
        freq_preserve: When ``True`` (and ``k > 1``), Sinkhorn-rescale the
            k-nearest weights so each rep's total (column) weight matches its
            single-nearest (k=1) Voronoi frequency, keeping the blend pattern
            (chain interpolation) but restoring frequency-correct invest sizing.
            Rows still sum to exactly 1; column margins are matched approximately.
            ``False`` (default) leaves the plain k-nearest weights unchanged.
        adaptive_frac: Fraction in ``[0, 1)`` of the most EXTREME base periods
            (by distance from the feature-space centroid) to pin back to
            single-nearest (k=1), preserving isolated extreme reps at full
            amplitude while interior periods keep the k-nearest blend. ``0.0``
            (default) is a no-op. Only affects ``k > 1``.

    Returns:
        Weight matrix W of shape (n_base_periods, n_rp). With ``k == 1`` each
        row has exactly one entry equal to 1.0 and the rest 0.0. With
        ``k > 1`` each row has up to ``k`` nonzero entries summing to 1.0
        (exactly one nonzero for rows coinciding with a rep). Ties in distance
        are broken toward the lowest representative index ``r`` via a STABLE
        sort, which preserves rep-index order for equal distances.

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if power <= 0:
        raise ValueError(f"power must be > 0, got {power}")
    if not 0.0 <= adaptive_frac < 1.0:
        raise ValueError(f"adaptive_frac must be in [0, 1), got {adaptive_frac}")

    n_base = C.shape[1]
    n_rp = len(rep_indices)
    k = min(k, n_rp)
    R = C[:, rep_indices]

    # Squared Euclidean distances between every base period and every rep,
    # vectorized via broadcasting: (n_features, n_base, 1) - (n_features, 1, n_rp).
    diff = C[:, :, np.newaxis] - R[:, np.newaxis, :]
    dist_sq = np.einsum("fbr,fbr->br", diff, diff)  # shape (n_base, n_rp)

    W = np.zeros((n_base, n_rp))

    if k == 1:
        # np.argmin returns the FIRST minimum → deterministic lowest-r tie-break.
        # Kept as a dedicated branch to guarantee byte-identical 0/1 output.
        nearest = np.argmin(dist_sq, axis=1)
        W[np.arange(n_base), nearest] = 1.0
        return W

    # Stable argsort keeps rep-index order on distance ties → lowest-r wins;
    # take the k smallest-distance reps per base period.
    order = np.argsort(dist_sq, axis=1, kind="stable")[:, :k]  # (n_base, k)
    rows = np.arange(n_base)[:, np.newaxis]
    sel_dist_sq = dist_sq[rows, order]  # (n_base, k) squared distances
    sel_dist = np.sqrt(sel_dist_sq)  # actual Euclidean distance

    # Zero-distance rows: base period coincides with one (or more) of the k
    # reps → inverse distance is infinite. Give all weight to the zero-distance
    # rep(s), split equally among them; do NOT divide by zero or spread.
    zero_mask = sel_dist == 0.0  # (n_base, k)
    row_has_zero = zero_mask.any(axis=1)  # (n_base,)

    weights = np.zeros_like(sel_dist)

    # Finite (no coincident rep) rows: inverse-distance, normalized to sum 1.
    finite_rows = ~row_has_zero
    if finite_rows.any():
        # power == 1.0 kept as a dedicated path to guarantee byte-identical
        # output to the pre-``power`` behaviour; p > 1 sharpens toward nearest.
        if power == 1.0:
            inv = 1.0 / sel_dist[finite_rows]
        else:
            inv = 1.0 / sel_dist[finite_rows] ** power
        weights[finite_rows] = inv / inv.sum(axis=1, keepdims=True)

    # Coincident rows: uniform split across the zero-distance rep(s) only.
    if row_has_zero.any():
        zsub = zero_mask[row_has_zero]
        weights[row_has_zero] = zsub / zsub.sum(axis=1, keepdims=True)

    # Scatter the per-selection weights back into the full (n_base, n_rp) matrix.
    np.add.at(W, (rows, order), weights)

    # --- Adaptive-k (lever 1): force the most EXTREME base periods back to
    # single-nearest (k=1) so isolated extreme reps keep full amplitude, while
    # interior base periods retain the k-nearest interpolation. Extremity =
    # distance from the base-period centroid in the (normalised) feature space;
    # the top ``adaptive_frac`` fraction is pinned to its single nearest rep.
    if adaptive_frac > 0.0:
        centroid = C.mean(axis=1, keepdims=True)  # (n_features, 1)
        d_cent = np.sqrt(((C - centroid) ** 2).sum(axis=0))  # (n_base,)
        thresh = np.quantile(d_cent, 1.0 - adaptive_frac)
        extreme = d_cent >= thresh
        if extreme.any():
            nearest1 = np.argmin(dist_sq, axis=1)
            W[extreme, :] = 0.0
            W[extreme, nearest1[extreme]] = 1.0

    # --- Frequency-preserving renormalisation (lever 2, the cheap proxy for
    # decoupling invest-weight from chain-interpolation): keep the k-nearest
    # blend PATTERN (so the inter-period storage chain still interpolates) but
    # rescale so each rep's total (column) weight matches its k=1 Voronoi
    # frequency (so the invest LP sizes capacity on true frequencies). Sinkhorn:
    # alternate column-scale to the k=1 targets and row-renorm to 1; the final
    # op is the row-renorm so rows sum to exactly 1 (column margins approximate).
    if freq_preserve:
        nearest1 = np.argmin(dist_sq, axis=1)
        targets = np.bincount(nearest1, minlength=n_rp).astype(float)  # (n_rp,)
        for _ in range(200):
            col = W.sum(axis=0)
            col[col == 0.0] = 1.0
            W *= (targets / col)[np.newaxis, :]  # scale columns toward targets
            rowsum = W.sum(axis=1, keepdims=True)
            rowsum[rowsum == 0.0] = 1.0
            W_new = W / rowsum  # renormalise rows to sum 1 (hard invariant)
            if np.max(np.abs(W_new - W)) < 1e-12:
                W = W_new
                break
            W = W_new

    return W
