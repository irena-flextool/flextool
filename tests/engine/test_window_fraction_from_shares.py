"""``p_window_fraction_from_shares``: the per-period window fraction of a sub-solve and its load-boundary raise.

The function runs in production for every roll and every synthetic nested sub-solve
(``flex_data.py``, the ``p_window_fraction`` step).  Its definition (docstring of
:func:`flextool.engine.preprocess.params._derived.p_window_fraction_from_shares`)::

    p_window_fraction[d] = (Σ_{t in dt[d]} step_duration[d,t] / 8760) / p_period_share[d]

* the window sum runs over the steps of ``dt`` only (``step_duration`` rows outside ``dt`` are ignored);
* a value within 1e-9 (relative) of 1 is snapped to exactly 1.0;
* a dt period without a ``p_period_share`` row raises ``ValueError``.

Hand derivation of the expected values (fixture below):

* p2020: dt steps t1, t2, t3 with durations 0.1 h, 0.2 h, 0.3 h.  In floating point
  0.1 + 0.2 + 0.3 = 0.6000000000000001 h (not exactly 0.6), so the window is 0.6000000000000001 / 8760.
  ``p_period_share[p2020]`` = 0.6 / 8760 (the period is covered whole).  Unsnapped ratio = 1 + ~2e-16,
  within 1e-9 of 1 → snapped to exactly 1.0.
* p2030: dt steps t1, t2 with durations 0.5 h and 1.0 h → window = 1.5 / 8760.  The ``step_duration``
  row (p2030, t3) = 4.5 h is not in dt and is ignored.  ``p_period_share[p2030]`` = 6.0 / 8760 (the complete
  solve covers 6 h of p2030).  Ratio = 1.5 / 6.0 = 0.25.
* ``p_period_share`` also has a p2040 row that dt does not use: the result has one row per dt period only.
* Raise case: the same dt with ``p_period_share`` holding only p2020 → p2030 has no share row →
  ``ValueError`` naming ``['p2030']``.
"""
from __future__ import annotations

import polars as pl
import pytest
from polar_high import Param

from flextool.engine.preprocess.params._derived import p_window_fraction_from_shares

HOURS_PER_YEAR = 8760.0


def _dt() -> pl.DataFrame:
    return pl.DataFrame({"d": ["p2020", "p2020", "p2020", "p2030", "p2030"],
                         "t": ["t1", "t2", "t3", "t1", "t2"]})


def _step_duration() -> Param:
    return Param(("d", "t"), pl.DataFrame({
        "d": ["p2020", "p2020", "p2020", "p2030", "p2030", "p2030"],
        "t": ["t1", "t2", "t3", "t1", "t2", "t3"],
        "value": [0.1, 0.2, 0.3, 0.5, 1.0, 4.5],
    }))


def _share(periods_hours: dict[str, float]) -> Param:
    return Param(("d",), pl.DataFrame({
        "d": list(periods_hours),
        "value": [h / HOURS_PER_YEAR for h in periods_hours.values()],
    }))


def test_window_fraction_values():
    """p2020 covers its whole share (snapped to exactly 1.0); p2030 covers 1.5 h of its 6 h → 0.25."""
    # The p2020 case only tests the snap if the window sum is not exactly 0.6 h.
    assert 0.1 + 0.2 + 0.3 != 0.6
    out = p_window_fraction_from_shares(
        _dt(), _step_duration(), _share({"p2020": 0.6, "p2030": 6.0, "p2040": 2.0}))
    assert out.dims == ("d",)
    rows = dict(zip(out.frame["d"].to_list(), out.frame["value"].to_list()))
    assert list(rows) == ["p2020", "p2030"]
    assert rows["p2020"] == 1.0
    assert rows["p2030"] == pytest.approx(0.25, rel=1e-12)


def test_dt_period_without_a_share_row_raises():
    """A dt period missing from ``p_period_share`` is refused instead of losing its op_factor and co2 cap."""
    with pytest.raises(ValueError, match=r"p_period_share has no row for dt periods \['p2030'\]"):
        p_window_fraction_from_shares(_dt(), _step_duration(), _share({"p2020": 0.6}))
