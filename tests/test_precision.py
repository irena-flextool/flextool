"""Unit tests for flextool.common_utils.precision.

Covers the two Agent 7 deliverables:

1. ``round_to_sigfigs`` — zero, negative, very small, very large, nan,
   inf, and assorted round-boundary inputs.
2. ``find_near_duplicates`` — identical inputs, unique inputs, and a
   cluster with exactly ``rel_tol`` spread.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# Make flextool importable when running the tests in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flextool.common_utils.precision import (
    find_near_duplicates,
    format_scalar_for_csv,
    resolve_precision_digits,
    resolve_report_near_duplicates,
    round_to_sigfigs,
)


# ---------------------------------------------------------------------------
# round_to_sigfigs
# ---------------------------------------------------------------------------


def test_round_to_sigfigs_zero_passthrough():
    assert round_to_sigfigs(0.0, 5) == 0.0
    # Signed zero
    assert math.copysign(1.0, round_to_sigfigs(-0.0, 5)) == -1.0


def test_round_to_sigfigs_digits_zero_or_negative_passthrough():
    assert round_to_sigfigs(1.234567, 0) == 1.234567
    assert round_to_sigfigs(1.234567, -3) == 1.234567


def test_round_to_sigfigs_basic():
    assert round_to_sigfigs(0.3333333333333333, 4) == 0.3333
    assert round_to_sigfigs(1 / 3, 10) == 0.3333333333
    assert round_to_sigfigs(123456.789, 3) == 123000.0
    assert round_to_sigfigs(123456.789, 5) == 123460.0


def test_round_to_sigfigs_negative_symmetric():
    assert round_to_sigfigs(-0.3333333, 4) == -0.3333
    assert round_to_sigfigs(-123456.789, 3) == -123000.0


def test_round_to_sigfigs_very_small():
    # Normal small values round normally.
    assert round_to_sigfigs(1.234e-10, 3) == 1.23e-10
    # Sub-denormal collapses to 0 — acceptable for LP coefficients.
    assert round_to_sigfigs(1e-320, 4) == 0.0


def test_round_to_sigfigs_very_large():
    # Values near float max still round without overflow.
    r = round_to_sigfigs(1.2345678e100, 4)
    assert r == 1.235e100


def test_round_to_sigfigs_nan_inf_passthrough():
    assert math.isnan(round_to_sigfigs(float("nan"), 5))
    assert math.isinf(round_to_sigfigs(float("inf"), 5))
    assert round_to_sigfigs(float("inf"), 5) > 0
    assert round_to_sigfigs(float("-inf"), 5) < 0


def test_round_to_sigfigs_integer_values_are_idempotent():
    # An already-rounded integer comes back bit-identical.
    assert round_to_sigfigs(1000.0, 4) == 1000.0
    assert round_to_sigfigs(1.0, 10) == 1.0


# ---------------------------------------------------------------------------
# format_scalar_for_csv
# ---------------------------------------------------------------------------


def test_format_scalar_default_passthrough():
    # precision_digits=0 → byte-for-byte same as str(value).
    for value in (0.3333333333333333, 1234567.89, "foo", "0.3333333333333333",
                  True, False, 42, "yes"):
        assert format_scalar_for_csv(value, 0) == str(value)


def test_format_scalar_float_rounded():
    assert format_scalar_for_csv(0.3333333333333333, 4) == "0.3333"


def test_format_scalar_integer_untouched():
    # Bare int stays "42", never "42.0".
    assert format_scalar_for_csv(42, 5) == "42"


def test_format_scalar_string_numeric_rounded():
    # Map values flow through as strings.
    assert format_scalar_for_csv("0.3333333333333333", 4) == "0.3333"
    # Pure integer literal string stays as-is.
    assert format_scalar_for_csv("42", 5) == "42"


def test_format_scalar_non_numeric_string_unchanged():
    assert format_scalar_for_csv("coal_plant", 5) == "coal_plant"
    assert format_scalar_for_csv("yes", 5) == "yes"


def test_format_scalar_bool_unchanged():
    assert format_scalar_for_csv(True, 5) == "True"
    assert format_scalar_for_csv(False, 5) == "False"


# ---------------------------------------------------------------------------
# find_near_duplicates
# ---------------------------------------------------------------------------


def test_find_near_duplicates_identical():
    clusters = find_near_duplicates([1.0, 1.0, 1.0, 1.0])
    assert len(clusters) == 1
    assert len(clusters[0]) == 4


def test_find_near_duplicates_unique():
    clusters = find_near_duplicates([1.0, 10.0, 100.0, 1000.0])
    # Widely-spaced values cluster nowhere.
    assert clusters == []


def test_find_near_duplicates_exact_rel_tol_spread():
    # Within exactly rel_tol — must cluster.
    rel_tol = 1e-6
    a = 1.0
    b = a * (1 + rel_tol)
    clusters = find_near_duplicates([a, b], rel_tol=rel_tol)
    assert clusters and len(clusters[0]) == 2


def test_find_near_duplicates_just_beyond_rel_tol():
    rel_tol = 1e-6
    a = 1.0
    b = a * (1 + 2 * rel_tol)  # well beyond
    clusters = find_near_duplicates([a, b], rel_tol=rel_tol)
    assert clusters == []


def test_find_near_duplicates_sorted_by_cluster_size():
    # Build: one triple around 10, one pair around 100, one singleton.
    values = [10.0, 10.0, 10.0, 100.0, 100.0, 1e6]
    clusters = find_near_duplicates(values, rel_tol=1e-6)
    assert [len(c) for c in clusters] == [3, 2]


def test_find_near_duplicates_sign_separated():
    # A +200 and a -200 are NOT the same value for symmetry detection;
    # clustering must split them by sign.
    clusters = find_near_duplicates([200.0, -200.0, 200.0, -200.0])
    # Expect one cluster of two positives and one of two negatives.
    assert len(clusters) == 2
    for c in clusters:
        assert all((v > 0) == (c[0] > 0) for v in c)
        assert len(c) == 2


def test_find_near_duplicates_handles_nan_and_inf():
    clusters = find_near_duplicates(
        [1.0, 1.0, float("nan"), float("inf"), 1.0]
    )
    assert clusters and len(clusters[0]) == 3


def test_find_near_duplicates_empty():
    assert find_near_duplicates([]) == []


# ---------------------------------------------------------------------------
# resolve_precision_digits / resolve_report_near_duplicates
# ---------------------------------------------------------------------------


def test_resolve_precision_digits_none_env_unset(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_PRECISION_DIGITS", raising=False)
    assert resolve_precision_digits(None) == 0


def test_resolve_precision_digits_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_PRECISION_DIGITS", "6")
    assert resolve_precision_digits(10) == 10


def test_resolve_precision_digits_env_used_when_cli_none(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_PRECISION_DIGITS", "6")
    assert resolve_precision_digits(None) == 6


def test_resolve_precision_digits_clamped_high(monkeypatch, capsys):
    # Reset the one-shot warning flag so the test is deterministic.
    import flextool.common_utils.precision as precision
    precision._WARNED_HIGH_PRECISION = False
    assert resolve_precision_digits(20) == 0
    out = capsys.readouterr().out
    assert "higher than float64" in out


def test_resolve_report_near_duplicates_cli_true():
    assert resolve_report_near_duplicates(True) is True


def test_resolve_report_near_duplicates_env_truthy(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_REPORT_NEAR_DUPS", "1")
    assert resolve_report_near_duplicates(False) is True
    monkeypatch.setenv("FLEXTOOL_REPORT_NEAR_DUPS", "yes")
    assert resolve_report_near_duplicates(False) is True


def test_resolve_report_near_duplicates_env_empty(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_REPORT_NEAR_DUPS", raising=False)
    assert resolve_report_near_duplicates(False) is False
