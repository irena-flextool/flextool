"""Slice H — §2.5 [F3] silent-no-lag guard (round-2 corrected).

The guard on ``period_walk_iterator``'s ``commission_lf`` join must:

* RAISE when the PRE-CAST (string-normalized) ``[e, d]`` key-overlap
  between the walk anchors and ``commission_lf`` is non-empty but the
  POST-CAST join yields zero non-null ``yr_c`` survivors — the real
  dtype/calendar desync that would otherwise silently drop the lag and
  produce a byte-parity-looking (but wrong) unlagged model;
* NOT raise on a legitimate multi-cohort model where ``commission_lf`` was
  computed globally but has ZERO overlap with a cohort carrying no
  lag-active entity (the availability walks are per-cohort).

The desync is simulated with a dtype-skewed ``commission_lf``: its ``d``
column is ``Int64`` whose string form matches the walk's Enum labels (so
the pre-cast overlap is non-empty) but which casts to null against the
Enum target (so the post-cast join drops every lag row).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._derived_walks import (
    WindowMethod,
    period_walk_iterator,
)

# Numeric period labels so an Int64 commission key string-matches the
# Enum labels while casting to null against the Enum.
PERIODS = ["1", "2", "3", "4"]
YEARS = [0.0, 5.0, 8.0, 20.0]
D_ENUM = pl.Enum(PERIODS)


class _StubSource:
    def __init__(self, params):
        self._params = params

    def parameter(self, entity_class: str, name: str) -> pl.DataFrame:
        try:
            return self._params[(entity_class, name)]
        except KeyError:
            raise KeyError((entity_class, name)) from None

    def parameter_default(self, entity_class: str, name: str):
        raise KeyError((entity_class, name))


def _source() -> _StubSource:
    return _StubSource({
        ("solve", "years_from_start"): pl.DataFrame({
            "name": ["s"] * len(PERIODS),
            "period": PERIODS,
            "value": YEARS,
        }),
    })


def _ed_lf(entity: str = "nuke") -> pl.LazyFrame:
    return pl.DataFrame({"e": [entity]}).with_columns(
        pl.Series("d", ["1"], dtype=D_ENUM)).lazy()


def _walk(commission_lf, entity: str = "nuke"):
    return period_walk_iterator(
        _source(), "s", _ed_lf(entity), PERIODS, PERIODS,
        window_method=WindowMethod.UNBOUNDED_FORWARD,
        life_lf=None, factor_side=None, commission_lf=commission_lf)


def test_guard_raises_on_dtype_desync() -> None:
    """Int64 d string-matches the Enum anchor '1' (pre-cast overlap > 0)
    but casts to null against the Enum (post-cast survivors == 0) -> the
    guard makes the silent-no-lag failure loud."""
    bad = pl.DataFrame({
        "e": ["nuke"],
        "d": pl.Series("d", [1], dtype=pl.Int64),  # string '1' == anchor
        "yr_c": [5.0],
    }).lazy()
    with pytest.raises(ValueError, match="no surviving lag rows"):
        _walk(bad).collect()


def test_guard_does_not_raise_on_zero_overlap_cohort() -> None:
    """A globally-computed commission_lf whose only entity ('gas') is NOT
    in this cohort's walk ('nuke') has zero pre-cast overlap -> the guard
    must NOT raise (round-2 MANDATORY case); the walk returns the unlagged
    set for this cohort."""
    other = pl.DataFrame({
        "e": ["gas"],
        "d": pl.Series("d", ["1"], dtype=D_ENUM),
        "yr_c": [5.0],
    }).lazy()
    out = _walk(other).collect()  # must not raise
    # nuke ordered at '1' (yr 0), no lag for this cohort -> alive all.
    assert sorted(out["d_all"].cast(pl.Utf8).to_list()) == PERIODS


def test_matching_commission_applies_lag_without_raising() -> None:
    """A well-typed commission_lf that overlaps the cohort applies the lag
    (order '1' -> commission year 5 -> alive '2','3','4') and does not
    raise — the positive control for the guard."""
    good = pl.DataFrame({
        "e": ["nuke"],
        "d": pl.Series("d", ["1"], dtype=D_ENUM),
        "yr_c": [5.0],
    }).lazy()
    out = _walk(good).collect()
    assert sorted(out["d_all"].cast(pl.Utf8).to_list()) == ["2", "3", "4"]
