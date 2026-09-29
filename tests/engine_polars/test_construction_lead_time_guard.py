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

from flextool.engine_polars._derived_existing import (
    _construction_lag_configured,
    edd_invest_set_lf,
)
from flextool.engine_polars._derived_params import _build_commission_lf
from flextool.engine_polars._derived_walks import (
    WindowMethod,
    period_walk_iterator,
)
from flextool.engine_polars._solve_state import CommissioningLagError

# Numeric period labels so an Int64 commission key string-matches the
# Enum labels while casting to null against the Enum.
PERIODS = ["1", "2", "3", "4"]
YEARS = [0.0, 5.0, 8.0, 20.0]
D_ENUM = pl.Enum(PERIODS)


class _StubSource:
    def __init__(self, params):
        self._params = params

    def entities(self, entity_class: str) -> pl.DataFrame:
        if entity_class == "unit":
            return pl.DataFrame({"name": ["nuke", "gas"]})
        raise KeyError(entity_class)

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


def _desync_commission() -> pl.LazyFrame:
    """Int64 d whose string form matches the Enum anchor '1' (pre-cast
    overlap > 0) but casts to null against the Enum (post-cast survivors
    == 0) — the real dtype/calendar desync the guard exists to catch."""
    return pl.DataFrame({
        "e": ["nuke"],
        "d": pl.Series("d", [1], dtype=pl.Int64),  # string '1' == anchor
        "yr_c": [5.0],
    }).lazy()


def test_guard_raises_on_dtype_desync() -> None:
    """The guard makes the silent-no-lag failure loud, raising the DEDICATED
    ``CommissioningLagError`` (not a bare ValueError that the downstream
    ``except Exception`` swallows would silence)."""
    with pytest.raises(CommissioningLagError, match="no surviving lag rows"):
        _walk(_desync_commission()).collect()


def test_commissioning_lag_error_subclasses_value_error() -> None:
    """CommissioningLagError <: ValueError so it can be raised from the
    guards without churning any ``pytest.raises(ValueError)`` pins, while
    its dedicated type lets the ``except Exception`` swallows re-raise it."""
    assert issubclass(CommissioningLagError, ValueError)


def test_guard_raises_through_real_builder(monkeypatch) -> None:
    """When the §2.5 guard fires inside ``period_walk_iterator``, the
    dedicated ``CommissioningLagError`` must PROPAGATE through the real
    builder ``edd_invest_set_lf`` — the exact callable ``apply_derived_c``
    wraps in its ``try`` as ``_edd_invest_lf`` — so its narrow re-raise
    (``except (LineageFilterError, CommissioningLagError): raise``) ahead of
    the broad ``except Exception`` fires instead of silently reverting to
    the unlagged set (MAJOR-1).

    The guard firing is modelled by patching ``period_walk_iterator`` (the
    walker the builder calls, and where the real guard lives) to raise the
    dedicated type: the builder has no swallow of its own, so the escape is
    what lets ``apply_derived_c``'s narrow except catch and re-raise it."""
    def _raise_lag(*_a, **_k):
        raise CommissioningLagError(
            "commission_lf join produced no surviving lag rows")
    monkeypatch.setattr(
        "flextool.engine_polars._derived_existing.period_walk_iterator",
        _raise_lag)
    with pytest.raises(CommissioningLagError, match="no surviving lag rows"):
        edd_invest_set_lf(
            _source(), "s", _ed_lf("nuke"), PERIODS, PERIODS,
            commission_lf=_desync_commission()).collect()


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


# --- MINOR-3: commissioning-helper failures must be loud when lag is set --

def _source_with_lead(lead: float, entity: str = "nuke") -> _StubSource:
    """A stub source carrying a scalar ``construction_lead_time`` on the
    unit class (so ``_construction_lag_configured`` sees a configured lag)."""
    src = _source()
    src._params[("unit", "construction_lead_time")] = pl.DataFrame(
        {"name": [entity], "value": [float(lead)]})
    return src


def test_lag_configured_true_when_lead_set() -> None:
    assert _construction_lag_configured(_source_with_lead(6.0)) is True


def test_lag_configured_false_no_lead() -> None:
    assert _construction_lag_configured(_source()) is False


def test_lag_configured_false_zero_lead() -> None:
    """L == 0 is the byte-parity no-lag case — must NOT count as configured
    (else existing L=0 models would raise on a benign helper hiccup)."""
    assert _construction_lag_configured(_source_with_lead(0.0)) is False


def _raise_runtime(*_a, **_k):
    raise RuntimeError("simulated commissioning_year_lf bug")


def test_build_commission_lf_surfaces_helper_error_when_lag_configured(
        monkeypatch) -> None:
    """A helper bug on a genuinely lag-configured model (L>0) surfaces as a
    CommissioningLagError instead of silently solving the unlagged model."""
    monkeypatch.setattr(
        "flextool.engine_polars._derived_existing.commissioning_year_lf",
        _raise_runtime)
    with pytest.raises(CommissioningLagError, match="MINOR-3"):
        _build_commission_lf(
            _source_with_lead(6.0), "s", _ed_lf("nuke"), PERIODS, None)


def test_build_commission_lf_swallows_helper_error_when_no_lag(
        monkeypatch) -> None:
    """No lead configured -> the helper would legitimately return None, so a
    failure is swallowed to None (byte-parity for existing models)."""
    monkeypatch.setattr(
        "flextool.engine_polars._derived_existing.commissioning_year_lf",
        _raise_runtime)
    assert _build_commission_lf(
        _source(), "s", _ed_lf("nuke"), PERIODS, None) is None


def test_build_commission_lf_propagates_commissioning_lag_error(
        monkeypatch) -> None:
    """A CommissioningLagError from the helper is NEVER swallowed, even on a
    no-lag source (the dedicated re-raise sits ahead of the broad except)."""
    def _boom(*_a, **_k):
        raise CommissioningLagError("desync from helper")
    monkeypatch.setattr(
        "flextool.engine_polars._derived_existing.commissioning_year_lf",
        _boom)
    with pytest.raises(CommissioningLagError, match="desync from helper"):
        _build_commission_lf(
            _source(), "s", _ed_lf("nuke"), PERIODS, None)


def test_build_commission_lf_none_when_no_periods() -> None:
    """No in-use periods -> None (no build attempted), regardless of lag."""
    assert _build_commission_lf(
        _source_with_lead(6.0), "s", _ed_lf("nuke"), [], None) is None
