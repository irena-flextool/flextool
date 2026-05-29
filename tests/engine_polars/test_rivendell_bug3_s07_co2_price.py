"""Rivendell bug 3 regression — S07_co2price_slice ``p_co2_price`` resolver.

The Rivendell ``S07_co2price_slice`` scenario regressed (HEAD ``fcfc0d11``)
with::

    ValueError: build_flextool: feature 'co2_price' is active but
    data fields are not populated (None): ['p_co2_price'].  Either
    fill them in the data or don't enable the feature.

Root cause: ``group.co2_price`` in the Rivendell DB is authored as a 1d
Map whose ``index_name`` is the spinedb_api silent default (``"x"``),
NOT an explicit ``"period"`` or ``"time"``.  Unlike ``co2_max_period``
(see ``test_phase_e_i_s08_co2cap_smoke.py``) whose allow-list
``{SCALAR, MAP_PERIOD}`` admits a unique shape at depth 1 (so the
silent default disambiguates structurally), ``co2_price`` admits
``{SCALAR, MAP_PERIOD, MAP_TIME, MAP_PERIOD_TIME}`` — depth 1 covers
both ``MAP_PERIOD`` and ``MAP_TIME``, so the silent default cannot be
resolved structurally.

The fix: when structural inference is ambiguous, the resolver probes
the Map's actual index values against the active solve's known periods
/ timesteps (mirroring the legacy CSV pipeline's
:func:`flextool.engine_polars._timeline.separate_period_and_timeseries_data`
discriminator).  A Map keyed by ``y2019, y2020, ...`` is unambiguously
:class:`Shape.MAP_PERIOD`; a Map keyed by ``t00001, t00002, ...`` is
:class:`Shape.MAP_TIME`.

This test exercises the offending resolver path directly — no full LP
solve required.  It fails on pre-fix HEAD (returns ``None``) and passes
after the fix (returns a populated ``Param``).
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._param_shapes import (
    Shape,
    _disambiguate_shape_by_value_domain,
)


# ---------------------------------------------------------------------------
# A. Helper-level — value-domain disambiguation (no DB required).
# ---------------------------------------------------------------------------


def _frame_with_index(values: list[str]) -> pl.DataFrame:
    """Build a (name, x, value) parameter frame with the given index
    values — mirrors what :meth:`SpineDbReader.parameter` returns for a
    1d Map whose index_name is the silent default ``"x"``.
    """
    return pl.DataFrame({
        "name":  ["g0"] * len(values),
        "x":     values,
        "value": [float(i) for i in range(len(values))],
    })


def test_value_domain_disambiguates_period_indexed_map() -> None:
    """A Map keyed by ``y2019, y2020, y2021`` against a period_filter
    whose ``d`` column lists those years resolves to ``MAP_PERIOD``.
    """
    df = _frame_with_index(["y2019", "y2020", "y2021"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020", "y2021"],
        "t": ["t00001", "t00001", "t00001"],
    })
    allowed = {Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
               Shape.MAP_PERIOD_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_PERIOD


def test_value_domain_disambiguates_time_indexed_map() -> None:
    """A Map keyed by ``t00001, t00002`` against the same filter
    resolves to ``MAP_TIME``.
    """
    df = _frame_with_index(["t00001", "t00002"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020"],
        "t": ["t00001", "t00002"],
    })
    allowed = {Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
               Shape.MAP_PERIOD_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_TIME


def test_value_domain_no_filter_returns_none() -> None:
    """Without a ``period_filter`` (off-cascade call sites), the
    fallback cannot probe and returns ``None`` — the caller must then
    accept ambiguity (resolver returns None, field drops from FlexData).
    """
    df = _frame_with_index(["y2019"])
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=None,
    )
    assert shape is None


def test_value_domain_mixed_values_returns_none() -> None:
    """When index values match NEITHER the period set NOR the timestep
    set exclusively, we don't guess: the resolver stays at ``None`` and
    the caller falls back.  Guards against silently misclassifying
    fixtures with novel indexing.
    """
    df = _frame_with_index(["foo", "bar"])
    period_filter = pl.DataFrame({
        "d": ["y2019"],
        "t": ["t00001"],
    })
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape is None


def test_value_domain_superset_of_periods_still_resolves() -> None:
    """Cyprus regression: a Map authored with MORE period keys than the
    active solve realises must still resolve to ``MAP_PERIOD``.

    The Cyprus ``Cyprus_Grid.sqlite`` authors ``group.co2_price`` with
    keys ``p2024, p2025, ..., p2050`` (annual), but scenarios like
    ``D 24h 24_30_35_40_45_50`` only realise the subset ``p2024, p2030,
    p2035, p2040, p2045, p2050``.  Pre-fix the probe required strict
    ``idx_set ⊆ periods_set`` — the extra keys p2025..p2029 failed the
    test, the probe returned None, ``p_co2_price`` was silently dropped,
    and ``build_flextool`` aborted on the CO2_PRICE invariant.  Post-fix
    we use the looser "intersects-periods, doesn't-touch-timesteps"
    rule: extra keys are tolerated (they're filtered out downstream by
    ``_filter_param_by_periods``).
    """
    # Map keys = superset of active periods (every other period extra).
    df = _frame_with_index([
        "p2024", "p2025", "p2026", "p2027", "p2028", "p2029",
        "p2030", "p2035", "p2040", "p2045", "p2050",
    ])
    period_filter = pl.DataFrame({
        "d": ["p2024", "p2030", "p2035", "p2040", "p2045", "p2050"],
        "t": ["t00001"] * 6,
    })
    allowed = {Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
               Shape.MAP_PERIOD_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_PERIOD


def test_value_domain_superset_of_timesteps_still_resolves() -> None:
    """Symmetric counterpart: a timestep Map with extra keys must
    still resolve to ``MAP_TIME``.  Guards against the rule
    accidentally being one-sided after the relaxation.
    """
    df = _frame_with_index(["t00001", "t00002", "t00003", "t99999"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020"],
        "t": ["t00001", "t00002"],
    })
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_TIME


def test_value_domain_overlap_with_both_universes_returns_none() -> None:
    """Hard-mode: index values overlap BOTH periods AND timesteps (so
    we can't tell which axis the author meant).  Must stay ``None`` so
    the caller falls back — relaxing subset to intersection must not
    let ambiguity through.
    """
    df = _frame_with_index(["y2019", "t00001"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020"],
        "t": ["t00001", "t00002"],
    })
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape is None


# ---------------------------------------------------------------------------
# B. End-to-end on the Rivendell DB — the S07 regression case.
# ---------------------------------------------------------------------------
#
# Removed: the Rivendell DB is private user data and not available to the
# test suite.  Section A above exercises the value-domain disambiguation
# helper directly with synthetic frames; the end-to-end coverage on the
# user's S07 scenario and the full cascade smoke (formerly Section C)
# both depended on rivendell.sqlite and have been dropped.
