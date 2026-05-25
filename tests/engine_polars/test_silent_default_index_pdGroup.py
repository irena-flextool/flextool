"""Silent-default index regression — Tier-1 ``pdGroup_*`` helpers.

A user's Cyprus DB authored ``group.inertia_limit`` as a 1d Map whose
``index_name`` is the spinedb_api silent default (``"x"``) but whose
keys are explicit period names (``p2024``, ``p2025``, ...).  Before the
Tier-1 migration, the legacy ``_entity_period_scalar`` helper routed
between scalar-broadcast and explicit-period branches based on
``if "period" not in cols`` — and since the silent-default column is
named ``"x"`` (not ``"period"``), the helper fell into the scalar branch
and cross-joined the Map's rows against the active solve's period
universe.  Result: ``N_map_rows × N_periods`` duplicate ``(g, d)`` keys
per entity, which crashed downstream emission and destroyed the
``period → value`` association on top.

Following the template from 2ad168f0 (Rivendell bug 3 fix for
``group.co2_price``), this commit migrates the 13 ``_entity_period_scalar``
helpers (the 12 group ``pdGroup_* / p_group_*`` 1d_map(period) helpers
plus ``p_startup_cost``) onto :func:`resolve_param_shape` +
:func:`broadcast_to_period`.  ``resolve_param_shape`` carries the
silent-default disambiguation via
:func:`_disambiguate_shape_by_value_domain`: when the registry permits
both ``MAP_PERIOD`` and ``SCALAR`` at depth 1 and the index column
carries the silent default, the resolver inspects the Map's index
values — period names vs. timestep tokens — to recover the author's
intent.  For these 13 helpers the allow-list is
``{SCALAR, MAP_PERIOD}`` (no time variant), so structural
disambiguation via :func:`_infer_silent_default_labels` already
recovers ``MAP_PERIOD`` from the silent default — no value-domain
probing is needed here, but the migration still benefits because the
resolver path no longer cross-joins explicit-period rows against the
period filter.

This test exercises the resolver directly against ``/tmp/cyprus_test.sqlite``
— it fails pre-migration on the duplicate-rows assertion and passes
post-migration.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


_CYPRUS_DB = Path("/tmp/cyprus_test.sqlite")
_SCENARIO = "D 5W 24-30_35_40_45_50"
# The scenario covers periods 2024..2030 plus 2035, 2040, 2045, 2050.
_PERIOD_NAMES = ["p2024", "p2025", "p2026", "p2027", "p2028", "p2029",
                 "p2030", "p2035", "p2040", "p2045", "p2050"]


@pytest.fixture(scope="module")
def cyprus_source():
    """SpineDbReader against the Cyprus migrated DB.

    Skips when the DB isn't present on this host (CI / other devs).
    """
    if not _CYPRUS_DB.exists():
        pytest.skip(f"Cyprus DB not present at {_CYPRUS_DB}")
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    return SpineDbReader(f"sqlite:///{_CYPRUS_DB}", _SCENARIO)


@pytest.fixture(scope="module")
def period_filter() -> pl.DataFrame:
    """A ``[d, t]`` frame covering the scenario's periods (one timestep
    per period; the value-domain probe only consults the ``d`` column).
    """
    return pl.DataFrame({
        "d": _PERIOD_NAMES,
        "t": ["t00001"] * len(_PERIOD_NAMES),
    })


def _collect(param) -> pl.DataFrame:
    """Materialise a ``Param.frame`` whether it's eager or lazy."""
    frame = param.frame
    return frame.collect() if hasattr(frame, "collect") else frame


def test_cyprus_inertia_limit_silent_default_index_resolves(
    cyprus_source, period_filter,
) -> None:
    """Top-level smoke: the silent-default ``index_name`` (``"x"``) on
    Cyprus' ``group.inertia_limit`` must be confirmed by the source, so
    this test really exercises the regression path."""
    info = cyprus_source.parameter_shape_info("group", "inertia_limit")
    assert info == ["x"], (
        f"Cyprus DB Map index_name is {info!r}; the regression test "
        "assumes the silent default 'x'."
    )


def test_cyprus_inertia_limit_no_duplicate_rows(
    cyprus_source, period_filter,
) -> None:
    """Pre-migration: ``_entity_period_scalar`` cross-joined the
    21-row Map against the 11-period filter, producing duplicate
    ``(g, d)`` keys for every entity authored with explicit periods.

    Post-migration: ``resolve_param_shape`` + ``broadcast_to_period``
    keep the authored ``(g, d)`` keys distinct.
    """
    from flextool.engine_polars._direct_params import (
        pdGroup_inertia_limit_from_source,
    )
    result = pdGroup_inertia_limit_from_source(
        cyprus_source, period_filter=period_filter)
    assert result is not None, (
        "pdGroup_inertia_limit_from_source returned None; expected a "
        "populated Param for the Cyprus DB."
    )
    frame = _collect(result)
    # Cast away potential Enum dtype to plain Utf8 for ergonomic asserts.
    frame = frame.with_columns(
        pl.col("g").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    dupes = (
        frame.group_by(["g", "d"])
             .len()
             .filter(pl.col("len") > 1)
    )
    assert dupes.height == 0, (
        "Duplicate (g, d) keys after broadcast — pre-migration "
        f"_entity_period_scalar cross-join bug.  Sample dupes:\n{dupes.head(10)}"
    )


def test_cyprus_inertia_limit_period_to_value_association(
    cyprus_source, period_filter,
) -> None:
    """Pre-migration the cross-join destroyed the ``period → value``
    association — every (g, d) pair carried whichever value happened to
    be last in the un-keyed broadcast.  Post-migration the explicit
    ``p2024 → 2.7621255747990894`` and ``p2050 → 292.0`` rows reach
    the LP intact.
    """
    from flextool.engine_polars._direct_params import (
        pdGroup_inertia_limit_from_source,
    )
    result = pdGroup_inertia_limit_from_source(
        cyprus_source, period_filter=period_filter)
    assert result is not None
    frame = _collect(result).with_columns(
        pl.col("g").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    aen = frame.filter(pl.col("g") == "All Electricity Nodes")
    p2024 = aen.filter(pl.col("d") == "p2024").get_column("value").to_list()
    p2050 = aen.filter(pl.col("d") == "p2050").get_column("value").to_list()
    assert len(p2024) == 1 and len(p2050) == 1, (
        f"Expected exactly one row each for p2024 / p2050; got "
        f"{len(p2024)} / {len(p2050)}.  Whole AEN frame:\n{aen}"
    )
    assert p2024[0] == pytest.approx(2.7621255747990894, rel=1e-10), (
        f"p2024 value mismatch: got {p2024[0]}")
    assert p2050[0] == pytest.approx(292.0), (
        f"p2050 value mismatch: got {p2050[0]}")


def test_cyprus_inertia_limit_scalar_broadcasts_to_all_periods(
    cyprus_source, period_filter,
) -> None:
    """``Diesel Units`` is authored as a scalar (Map index ``null``,
    value 2.0).  Post-migration the resolver detects ``Shape.SCALAR``
    and ``broadcast_to_period`` widens it to every active period.
    """
    from flextool.engine_polars._direct_params import (
        pdGroup_inertia_limit_from_source,
    )
    result = pdGroup_inertia_limit_from_source(
        cyprus_source, period_filter=period_filter)
    assert result is not None
    frame = _collect(result).with_columns(
        pl.col("g").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    diesel = frame.filter(pl.col("g") == "Diesel Units").sort("d")
    assert diesel.height == len(_PERIOD_NAMES), (
        f"Scalar source for 'Diesel Units' should broadcast to "
        f"{len(_PERIOD_NAMES)} periods; got {diesel.height} rows: {diesel}"
    )
    values = diesel.get_column("value").to_list()
    assert all(v == 2.0 for v in values), (
        f"All scalar-broadcast values should be 2.0; got {values}"
    )
    got_periods = set(diesel.get_column("d").to_list())
    assert got_periods == set(_PERIOD_NAMES), (
        f"Scalar broadcast covered {got_periods}, expected "
        f"{set(_PERIOD_NAMES)}"
    )


def _assert_no_key_duplicates(param) -> None:
    """Assert the Param frame has no duplicate keys across its dims.

    Handles both Phase-E.1 shapes resolver-produced Params can take:
    ``(g,)`` for scalar sources and ``(g, d)`` for period maps.  The
    legacy ``_entity_period_scalar`` bug always materialised as
    duplicate ``(g, d)`` rows — those will surface here regardless of
    which shape the post-migration resolver lands on.
    """
    dims = list(param.dims)
    frame = _collect(param)
    frame = frame.with_columns([pl.col(d).cast(pl.Utf8) for d in dims])
    dupes = frame.group_by(dims).len().filter(pl.col("len") > 1)
    assert dupes.height == 0, (
        f"Duplicate keys {dims} after broadcast:\n{dupes.head(10)}"
    )


def test_cyprus_penalty_inertia_no_duplicates_no_crash(
    cyprus_source, period_filter,
) -> None:
    """Sibling smoke check: ``penalty_inertia`` must not crash and must
    have no duplicate keys across its dims, even if every Cyprus entity
    authored it as a scalar (Param dims ``(g,)``)."""
    from flextool.engine_polars._direct_params import (
        pdGroup_penalty_inertia_from_source,
    )
    result = pdGroup_penalty_inertia_from_source(
        cyprus_source, period_filter=period_filter)
    if result is None:
        # All rows zero / no source rows — acceptable; the bug we're
        # guarding against is duplicate-rows on the explicit branch.
        return
    _assert_no_key_duplicates(result)


def test_cyprus_non_synchronous_limit_no_duplicates(
    cyprus_source, period_filter,
) -> None:
    """Second sibling smoke check: same shape family, same regression
    surface."""
    from flextool.engine_polars._direct_params import (
        pdGroup_non_synchronous_limit_from_source,
    )
    result = pdGroup_non_synchronous_limit_from_source(
        cyprus_source, period_filter=period_filter)
    if result is None:
        return
    _assert_no_key_duplicates(result)
