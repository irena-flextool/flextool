"""Downsampling utilities for time series display in the result viewer."""

from __future__ import annotations

import numpy as np


def downsample_for_display(
    x: np.ndarray,
    y: np.ndarray,
    n_out: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample time series data to n_out points for display.

    Uses MinMaxLTTB from tsdownsample if available, falls back to
    a simple numpy-based LTTB approximation.

    Returns (x_downsampled, y_downsampled).
    """
    if len(x) <= n_out:
        return x, y

    try:
        from tsdownsample import MinMaxLTTBDownsampler
        downsampler = MinMaxLTTBDownsampler()
        # tsdownsample requires contiguous float arrays
        y_cont = np.ascontiguousarray(y, dtype=np.float64)
        indices = downsampler.downsample(y_cont, n_out=n_out)
        return x[indices], y[indices]
    except ImportError:
        return _lttb_fallback(x, y, n_out)


def _lttb_fallback(
    x: np.ndarray,
    y: np.ndarray,
    n_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Largest Triangle Three Buckets downsampling (pure numpy).

    This is a fallback when tsdownsample is not installed. It's slower
    but produces visually similar results.
    """
    n = len(x)
    if n <= n_out:
        return x, y

    # Always include first and last points
    indices = [0]

    # Bucket size for the middle points
    bucket_size = (n - 2) / (n_out - 2)

    a_idx = 0  # Previously selected point index

    for i in range(1, n_out - 1):
        # Calculate bucket boundaries
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)

        # Calculate next bucket average (the "triangle tip")
        next_start = int(i * bucket_size) + 1
        next_end = int((i + 1) * bucket_size) + 1
        next_end = min(next_end, n)

        avg_x = np.mean(x[next_start:next_end].astype(np.float64))
        avg_y = np.mean(y[next_start:next_end].astype(np.float64))

        # Find the point in the current bucket with largest triangle area
        best_idx = bucket_start
        max_area = -1.0

        for j in range(bucket_start, bucket_end):
            area = abs(
                (x[a_idx] - avg_x) * (y[j] - y[a_idx])
                - (x[a_idx] - x[j]) * (avg_y - y[a_idx])
            )
            if area > max_area:
                max_area = area
                best_idx = j

        indices.append(best_idx)
        a_idx = best_idx

    indices.append(n - 1)
    idx_arr = np.array(indices)
    return x[idx_arr], y[idx_arr]
