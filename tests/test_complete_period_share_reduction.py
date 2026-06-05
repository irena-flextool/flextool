"""Order-independence of the multi-solve ``complete_period_share_of_year``
reduction (the intermittent 121× period over-count regression).

In a nested invest→dispatch run BOTH sub-solves persist a row for the
realised period: the invest sub-solve a *sampled-window* share
(e.g. 72 h / 8760 ≈ 0.0082) and the dispatch sub-solve the *full-year*
share (1.0).  The multi-solve union concatenates the per-sub-solve slices
in an order that is NOT guaranteed — when the ``_solve_order.txt`` creation
manifest is absent the union falls back to filesystem glob order.  A plain
``keep='first'/'last'`` dedup is therefore a coin-flip: on the unlucky
ordering it keeps the 0.0082 window share and every DISPATCH period total
(energy/cost = Σ power·dur ÷ share) is inflated by 8760/72 ≈ 121.7×.

``_reduce_share_per_period`` collapses by MAX per period, which picks the
full-coverage dispatch share deterministically regardless of order.  These
tests assert that property directly, in BOTH union orders, so the fix
cannot silently regress to the order-dependent behaviour.
"""
from __future__ import annotations

import pandas as pd
import pytest

from flextool.process_outputs.drop_levels import _reduce_share_per_period

# Hand numbers: 3×24 h invest window over a 8760 h year.
_INVEST_SHARE = 72.0 / 8760.0          # ≈ 0.00821917808219
_DISPATCH_SHARE = 1.0                  # full-year dispatch


def _share_series(rows: list[tuple[str, str, float]]) -> pd.Series:
    """Build a ``(solve, period)``-indexed share Series from rows."""
    idx = pd.MultiIndex.from_tuples(
        [(solve, period) for solve, period, _ in rows],
        names=["solve", "period"],
    )
    return pd.Series([v for *_, v in rows], index=idx, name="value")


@pytest.mark.parametrize(
    "rows",
    [
        # invest first, dispatch last (manifest-present / correct order)
        [("invest", "p2020", _INVEST_SHARE),
         ("dispatch_p2020", "p2020", _DISPATCH_SHARE)],
        # dispatch first, invest last (glob-order fallback — the flaky bug;
        # old keep='last' picked invest here and inflated by 121.7×)
        [("dispatch_p2020", "p2020", _DISPATCH_SHARE),
         ("invest", "p2020", _INVEST_SHARE)],
    ],
    ids=["invest_first", "dispatch_first"],
)
def test_reduction_is_order_independent(rows):
    out = _reduce_share_per_period(_share_series(rows))
    assert list(out.index) == ["p2020"]
    # Always the full-year dispatch share — never the 0.0082 window share.
    assert out.loc["p2020"] == pytest.approx(1.0, rel=1e-12)
    assert out.loc["p2020"] != pytest.approx(_INVEST_SHARE, rel=1e-6)


def test_single_solve_share_unchanged():
    # Pure dispatch (S2-style): one row → value preserved.
    out = _reduce_share_per_period(_share_series([("dispatch", "p2020", 1.0)]))
    assert out.loc["p2020"] == pytest.approx(1.0, rel=1e-12)


def test_invest_only_period_keeps_its_window_share():
    # A period realised ONLY by the invest sub-solve (no dispatch sub-solve
    # for it) keeps the sampled-window share — that IS its correct
    # annualisation denominator (only the window was modelled).
    out = _reduce_share_per_period(
        _share_series([("invest", "p2099", _INVEST_SHARE)])
    )
    assert out.loc["p2099"] == pytest.approx(_INVEST_SHARE, rel=1e-12)


def test_multi_period_mixed_order():
    # Two periods, rows interleaved/reversed across solves; each period must
    # independently resolve to its full-coverage dispatch share, and period
    # first-appearance order is preserved (sort=False).
    rows = [
        ("dispatch_p2025", "p2025", 1.0),
        ("invest", "p2020", _INVEST_SHARE),
        ("invest", "p2025", _INVEST_SHARE),
        ("dispatch_p2020", "p2020", 1.0),
    ]
    out = _reduce_share_per_period(_share_series(rows))
    assert list(out.index) == ["p2025", "p2020"]      # first-appearance order
    assert out.loc["p2020"] == pytest.approx(1.0, rel=1e-12)
    assert out.loc["p2025"] == pytest.approx(1.0, rel=1e-12)
