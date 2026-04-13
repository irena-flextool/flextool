"""Greedy Convex Hull Clustering for representative period selection.

Implements Algorithm 2 from:
    "Hull Clustering with Blended Representative Periods"
    Neustroev et al., 2025.

The algorithm greedily selects representative periods by iteratively
picking the base period whose feature vector is furthest from the
convex hull of already-selected representatives. A caching optimisation
avoids recomputing hull distances when the new representative cannot
have improved the approximation for a given candidate.
"""

import numpy as np

from .weights import distance_to_hull


def greedy_convex_hull_clustering(
    C: np.ndarray,
    n_rp: int,
) -> list[int]:
    """Select representative periods using greedy convex hull clustering.

    Algorithm:
      1. Compute the mean of all columns of C.
      2. Select the column furthest from the mean as the first RP.
      3. Greedily add the column furthest from the current convex hull.
      4. Use caching: if dist(c_d, c_{r_new}) >= cached_dist[d],
         the cached hull distance is still valid (the hull only gets
         closer or stays the same when a point is added, never farther).

    Args:
        C: Clustering matrix, shape (n_features, n_base_periods).
           Each column is one base period's feature vector.
        n_rp: Number of representative periods to select.

    Returns:
        List of selected column indices (representative period indices).

    Raises:
        ValueError: If C is not a 2-D array or n_rp < 1.
    """
    if C.ndim != 2:
        raise ValueError(f"C must be a 2-D array, got {C.ndim}-D")
    n_features, n_base = C.shape

    if n_rp <= 0:
        raise ValueError(f"n_rp must be >= 1, got {n_rp}")

    # Edge case: requesting as many or more RPs than base periods
    if n_rp >= n_base:
        return list(range(n_base))

    # ------------------------------------------------------------------
    # Step 1: compute mean column vector
    # ------------------------------------------------------------------
    mean_col = C.mean(axis=1)  # shape (n_features,)

    # ------------------------------------------------------------------
    # Step 2: first RP = column furthest from the mean
    # ------------------------------------------------------------------
    dists_to_mean = np.linalg.norm(C - mean_col[:, np.newaxis], axis=0)
    first_rp = int(np.argmax(dists_to_mean))
    rep_indices: list[int] = [first_rp]
    print(
        f"Selected RP 1/{n_rp}, max distance to mean: "
        f"{dists_to_mean[first_rp]:.6f}"
    )

    # ------------------------------------------------------------------
    # Initialise cached hull distances for step 3
    # ------------------------------------------------------------------
    # After picking the first RP, the "hull" is a single point.
    # dist(c_d, hull) = ||c_d - c_{first_rp}||
    cached_dist = np.linalg.norm(
        C - C[:, first_rp : first_rp + 1], axis=0
    )  # shape (n_base,)
    # Mark selected indices so they are never picked again
    cached_dist[first_rp] = -1.0

    # ------------------------------------------------------------------
    # Step 3: greedily add the column furthest from current hull
    # ------------------------------------------------------------------
    for k in range(2, n_rp + 1):
        # The previous iteration chose rep_indices[-1] as r_new.
        r_new = rep_indices[-1]
        c_new = C[:, r_new]

        # Build the representative sub-matrix for hull distance queries
        R = C[:, rep_indices]  # shape (n_features, len(rep_indices))

        for d in range(n_base):
            if cached_dist[d] < 0.0:
                # Already selected as an RP — skip
                continue

            # Caching check: if the new RP is at least as far from c_d
            # as the cached hull distance, adding it cannot help.
            dist_to_new = float(np.linalg.norm(C[:, d] - c_new))
            if dist_to_new >= cached_dist[d]:
                # Cached value is still a valid upper bound and the hull
                # distance cannot have decreased — keep cached value.
                continue

            # Need to recompute hull distance with the updated hull
            hull_dist, _ = distance_to_hull(R, C[:, d])
            cached_dist[d] = hull_dist

        # Pick the candidate with the largest hull distance
        best_idx = int(np.argmax(cached_dist))
        best_dist = cached_dist[best_idx]
        rep_indices.append(best_idx)
        cached_dist[best_idx] = -1.0  # mark as selected

        print(
            f"Selected RP {k}/{n_rp}, max hull distance: "
            f"{best_dist:.6f}"
        )

    return rep_indices
