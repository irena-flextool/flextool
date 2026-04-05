"""Tests for flextool.gui.downsampling."""

from __future__ import annotations

import numpy as np
import pytest

from flextool.gui.downsampling import downsample_for_display, _lttb_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine_data(n: int):
    """Return (x, y) arrays of length *n* with a noisy sine wave."""
    rng = np.random.default_rng(42)
    x = np.arange(n, dtype=np.float64)
    y = np.sin(x * 0.01) + rng.normal(scale=0.1, size=n)
    return x, y


# ---------------------------------------------------------------------------
# Tests for downsample_for_display (uses tsdownsample or fallback)
# ---------------------------------------------------------------------------

class TestDownsampleForDisplay:
    def test_short_data_returned_unchanged(self):
        x = np.arange(50, dtype=np.float64)
        y = np.sin(x)
        x_ds, y_ds = downsample_for_display(x, y, n_out=3000)
        np.testing.assert_array_equal(x_ds, x)
        np.testing.assert_array_equal(y_ds, y)

    def test_downsamples_long_data(self):
        x, y = _make_sine_data(10_000)
        x_ds, y_ds = downsample_for_display(x, y, n_out=100)
        assert len(x_ds) == 100
        assert len(y_ds) == 100

    def test_first_and_last_preserved(self):
        x, y = _make_sine_data(10_000)
        x_ds, y_ds = downsample_for_display(x, y, n_out=100)
        assert x_ds[0] == x[0]
        assert x_ds[-1] == x[-1]

    def test_preserves_peaks(self):
        n = 10_000
        x = np.arange(n, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        # Insert a sharp spike
        y[5000] = 1000.0
        x_ds, y_ds = downsample_for_display(x, y, n_out=200)
        assert 1000.0 in y_ds

    def test_nan_handling(self):
        x, y = _make_sine_data(10_000)
        y[100] = np.nan
        y[500] = np.nan
        y[9999] = np.nan
        # Should not raise
        x_ds, y_ds = downsample_for_display(x, y, n_out=200)
        assert len(x_ds) == 200


# ---------------------------------------------------------------------------
# Tests for _lttb_fallback directly
# ---------------------------------------------------------------------------

class TestLttbFallback:
    def test_short_data_returned_unchanged(self):
        x = np.arange(50, dtype=np.float64)
        y = np.sin(x)
        x_ds, y_ds = _lttb_fallback(x, y, n_out=3000)
        np.testing.assert_array_equal(x_ds, x)
        np.testing.assert_array_equal(y_ds, y)

    def test_downsamples_long_data(self):
        x, y = _make_sine_data(10_000)
        x_ds, y_ds = _lttb_fallback(x, y, n_out=100)
        assert len(x_ds) == 100
        assert len(y_ds) == 100

    def test_first_and_last_preserved(self):
        x, y = _make_sine_data(10_000)
        x_ds, y_ds = _lttb_fallback(x, y, n_out=100)
        assert x_ds[0] == x[0]
        assert x_ds[-1] == x[-1]

    def test_preserves_peaks(self):
        n = 10_000
        x = np.arange(n, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        y[5000] = 1000.0
        x_ds, y_ds = _lttb_fallback(x, y, n_out=200)
        assert 1000.0 in y_ds

    def test_nan_handling(self):
        x, y = _make_sine_data(10_000)
        y[100] = np.nan
        y[500] = np.nan
        y[9999] = np.nan
        x_ds, y_ds = _lttb_fallback(x, y, n_out=200)
        assert len(x_ds) == 200
