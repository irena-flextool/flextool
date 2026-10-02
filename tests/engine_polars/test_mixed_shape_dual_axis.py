"""Mixed period/time Map shapes within one parameter (spec fix_map_reading.md).

Builds a real Spine DB in ``tmp_path`` per case and exercises
:class:`SpineDbReader` per-row period/time placement (§3.1),
``parameter_shape_info`` / ``parameter_shape_variants`` (§3.2) and the
resolver's per-variant allow-list check (§3.3).

Every case is parametrised over **row order × reader mode** — "without
enums" (unit-test setup, label-only classification) and "with enums"
(production path: ``axis_enums`` + ``contract`` + ``scenario`` so
silent-label value-domain classification + scenario trim are active).

Implements the 13 cases of spec §4.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import SpineDbReader
from flextool.engine_polars._axis_enums import (
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars._param_shapes import (
    Shape,
    broadcast_to_period_time,
    resolve_param_shape,
)
from flextool.engine_polars._solve_state import FlexToolConfigError
from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError

PERIODS = ["p2025", "p2030"]
TIMESTEPS = ["t01", "t02"]

ORDERS = ["as_is", "reversed"]
MODES = ["no_enums", "with_enums"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _m(keys, values, index_name=None):
    """Build a spinedb_api Map with an optional explicit index_name."""
    from spinedb_api.parameter_value import Map
    if index_name is None:
        return Map(list(keys), list(values))
    return Map(list(keys), list(values), index_name=index_name)


def _ts(stamps, values):
    """Build a spinedb_api TimeSeries (variable-resolution)."""
    from spinedb_api.parameter_value import TimeSeriesVariableResolution
    return TimeSeriesVariableResolution(list(stamps), list(values), False, False)


def _build_db(
    path: Path,
    class_name: str,
    param: str,
    rows: list[tuple[str, object]],
    *,
    order: str = "as_is",
) -> str:
    """Build a one-class / one-parameter Spine DB; return the sqlite URL."""
    from spinedb_api import DatabaseMapping
    from spinedb_api.parameter_value import to_database

    if order == "reversed":
        rows = list(reversed(rows))
    url = f"sqlite:///{path}"
    with DatabaseMapping(url, create=True) as db:
        db.add_entity_class(name=class_name)
        db.add_parameter_definition(entity_class_name=class_name, name=param)
        db.add_scenario(name="s")
        db.add_scenario_alternative(
            scenario_name="s", alternative_name="Base", rank=1,
        )
        for ent, _v in rows:
            db.add_entity(entity_class_name=class_name, name=ent)
        for ent, v in rows:
            val, typ = to_database(v)
            db.add_parameter_value(
                entity_class_name=class_name, entity_byname=(ent,),
                parameter_definition_name=param, alternative_name="Base",
                value=val, type=typ,
            )
        db.commit_session("x")
    return url


def _reader(url, mode, *, periods=PERIODS, timesteps=TIMESTEPS):
    """Return (reader, enums) for the requested mode."""
    if mode == "with_enums":
        enums = {"d": pl.Enum(periods), "t": pl.Enum(timesteps)}
        return SpineDbReader(url, "s", axis_enums=enums), enums
    return SpineDbReader(url, "s"), None


def _dt(enums, *, periods=PERIODS, timesteps=TIMESTEPS):
    """Build a dense (d, t) period_filter frame covering periods×timesteps."""
    d_col, t_col = [], []
    for p in periods:
        for t in timesteps:
            d_col.append(p)
            t_col.append(t)
    if enums is not None:
        return pl.DataFrame({
            "d": pl.Series(d_col, dtype=enums["d"]),
            "t": pl.Series(t_col, dtype=enums["t"]),
        })
    return pl.DataFrame({"d": d_col, "t": t_col})


@contextmanager
def _maybe_global(enums):
    """Activate the cascade-wide axis enums when running the enum mode."""
    if enums is None:
        yield
        return
    tok = set_global_axis_enums(enums)
    try:
        yield
    finally:
        reset_global_axis_enums(tok)


def _broadcast_rows(reader, enums, class_name, param, entity_alias="group",
                    *, periods=PERIODS, timesteps=TIMESTEPS):
    """Resolve + broadcast a parameter to a materialised (e, d, t) frame."""
    dt = _dt(enums, periods=periods, timesteps=timesteps)
    with _maybe_global(enums):
        res = resolve_param_shape(reader, class_name, param, period_filter=dt)
        param_obj = broadcast_to_period_time(res, entity_alias, dt)
        mat = (param_obj.lazy.collect()
               if param_obj is not None else None)
    return res, mat


# ---------------------------------------------------------------------------
# Case 1 — labelled period map + labelled time map → 8 rows, order-stable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case01_labelled_period_plus_time(tmp_path, order, mode):
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_period", _m(PERIODS, [50.0, 60.0], "period")),
         ("fg_time", _m(TIMESTEPS, [10.0, 20.0], "time"))],
        order=order,
    )
    reader, enums = _reader(url, mode)
    res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    assert res is not None and res.shape is Shape.MAP_PERIOD_TIME
    assert mat.height == 8
    # fg_period: constant per period across both timesteps.
    fp = mat.filter(pl.col("group") == "fg_period").sort("d", "t")
    assert fp.get_column("value").to_list() == [50.0, 50.0, 60.0, 60.0]
    # fg_time: constant per timestep across both periods.
    ft = mat.filter(pl.col("group") == "fg_time").sort("d", "t")
    assert ft.get_column("value").to_list() == [10.0, 20.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# Case 2 — scalar + period map + time map + 2-D map on four entities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case02_four_shapes(tmp_path, order, mode):
    two_d = _m(PERIODS, [_m(TIMESTEPS, [1.0, 2.0], "time"),
                         _m(TIMESTEPS, [3.0, 4.0], "time")], "period")
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_scalar", 7.0),
         ("fg_period", _m(PERIODS, [50.0, 60.0], "period")),
         ("fg_time", _m(TIMESTEPS, [10.0, 20.0], "time")),
         ("fg_2d", two_d)],
        order=order,
    )
    reader, enums = _reader(url, mode)
    res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    assert res is not None and res.shape is Shape.MAP_PERIOD_TIME
    # Each entity fully covers the 2×2 grid = 4 rows ⇒ 16 total.
    assert mat.height == 16
    assert set(mat.get_column("group").to_list()) == {
        "fg_scalar", "fg_period", "fg_time", "fg_2d"}
    sc = mat.filter(pl.col("group") == "fg_scalar")
    assert sc.get_column("value").to_list() == [7.0, 7.0, 7.0, 7.0]
    d2 = mat.filter(pl.col("group") == "fg_2d").sort("d", "t")
    assert d2.get_column("value").to_list() == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Case 3 — 2-D Map(period,time) + 1-D Map(time): the time row lands in t
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case03_2d_plus_1d_time(tmp_path, order, mode):
    two_d = _m(PERIODS, [_m(TIMESTEPS, [1.0, 2.0], "time"),
                         _m(TIMESTEPS, [3.0, 4.0], "time")], "period")
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_2d", two_d),
         ("fg_time", _m(TIMESTEPS, [10.0, 20.0], "time"))],
        order=order,
    )
    reader, enums = _reader(url, mode)
    # The explicit (placed) frame: the time row has null period.
    expl = reader.parameter_explicit("flowGroup", "min_instant_flow")
    ft = expl.filter(pl.col("name") == "fg_time")
    assert ft.get_column("period").to_list() == [None, None]
    assert [str(x) for x in ft.get_column("t").to_list()] == TIMESTEPS
    res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    assert res.shape is Shape.MAP_PERIOD_TIME
    ft_b = mat.filter(pl.col("group") == "fg_time").sort("d", "t")
    assert ft_b.get_column("value").to_list() == [10.0, 20.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# Case 4 — silent "x" period map + silent "x" time map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case04_silent_period_plus_time(tmp_path, order, mode):
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_period", _m(PERIODS, [50.0, 60.0])),   # silent index_name
         ("fg_time", _m(TIMESTEPS, [10.0, 20.0]))],
        order=order,
    )
    reader, enums = _reader(url, mode)
    if mode == "with_enums":
        res, mat = _broadcast_rows(
            reader, enums, "flowGroup", "min_instant_flow")
        assert res.shape is Shape.MAP_PERIOD_TIME
        assert mat.height == 8
        fp = mat.filter(pl.col("group") == "fg_period").sort("d", "t")
        assert fp.get_column("value").to_list() == [50.0, 50.0, 60.0, 60.0]
    else:
        # Without enums the behaviour is as today (documented): the
        # silent labels cannot be value-domain classified, so the reader
        # falls back to the legacy positional path.  We only assert it
        # does not crash.
        reader.parameter_explicit("flowGroup", "min_instant_flow")


# ---------------------------------------------------------------------------
# Case 5 — silent "x" time map next to a labelled period map (enums)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_case05_silent_time_plus_labelled_period(tmp_path, order):
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_period", _m(PERIODS, [50.0, 60.0], "period")),
         ("fg_time", _m(TIMESTEPS, [10.0, 20.0]))],  # silent, timestep keys
        order=order,
    )
    reader, enums = _reader(url, "with_enums")
    res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    assert res.shape is Shape.MAP_PERIOD_TIME
    assert mat.height == 8
    ft = mat.filter(pl.col("group") == "fg_time").sort("d", "t")
    assert ft.get_column("value").to_list() == [10.0, 20.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# Case 6 — silent 1-D row next to a labelled 2-D row (works today)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case06_silent_1d_plus_labelled_2d(tmp_path, order, mode):
    two_d = _m(PERIODS, [_m(TIMESTEPS, [1.0, 2.0], "time"),
                         _m(TIMESTEPS, [3.0, 4.0], "time")], "period")
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_2d", two_d),
         ("fg_period", _m(PERIODS, [50.0, 60.0], "period"))],
        order=order,
    )
    reader, enums = _reader(url, mode)
    res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    assert res.shape is Shape.MAP_PERIOD_TIME
    assert mat.height == 8
    fp = mat.filter(pl.col("group") == "fg_period").sort("d", "t")
    assert fp.get_column("value").to_list() == [50.0, 50.0, 60.0, 60.0]


# ---------------------------------------------------------------------------
# Case 7 — commodity.price mixed period/time accepted; real 2-D rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case07_commodity_price_mixed_accepted(tmp_path, order, mode):
    url = _build_db(
        tmp_path / "db.sqlite", "commodity", "price",
        [("c_period", _m(PERIODS, [50.0, 60.0], "period")),
         ("c_time", _m(TIMESTEPS, [10.0, 20.0], "time"))],
        order=order,
    )
    reader, enums = _reader(url, mode)
    res, mat = _broadcast_rows(reader, enums, "commodity", "price",
                               entity_alias="commodity")
    # Combined shape is MAP_PERIOD_TIME (not in commodity.price's allow-list)
    # yet accepted because every per-row variant is allowed.
    assert res.shape is Shape.MAP_PERIOD_TIME
    assert mat.height == 8


@pytest.mark.parametrize("mode", MODES)
def test_case07_commodity_price_real_2d_rejected(tmp_path, mode):
    two_d = _m(PERIODS, [_m(TIMESTEPS, [1.0, 2.0], "time"),
                         _m(TIMESTEPS, [3.0, 4.0], "time")], "period")
    url = _build_db(
        tmp_path / "db.sqlite", "commodity", "price",
        [("c_2d", two_d)],
    )
    reader, enums = _reader(url, mode)
    dt = _dt(enums)
    with _maybe_global(enums):
        with pytest.raises(FlexToolConfigError):
            resolve_param_shape(reader, "commodity", "price", period_filter=dt)


# ---------------------------------------------------------------------------
# Case 8 — keys hit BOTH vocabularies → FlexDataIntegrityError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_case08_keys_hit_both_vocabs(tmp_path, order):
    # "shared" appears in BOTH the period and timestep vocabulary.
    periods = ["shared", "p2030"]
    timesteps = ["shared", "t02"]
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_ambig", _m(["shared"], [5.0]))],  # silent label, ambiguous key
        order=order,
    )
    reader, _enums = _reader(url, "with_enums",
                             periods=periods, timesteps=timesteps)
    with pytest.raises(FlexDataIntegrityError):
        reader.parameter_explicit("flowGroup", "min_instant_flow")


# ---------------------------------------------------------------------------
# Case 9 — multi-solve: out-of-active-solve period keys classify stably
# ---------------------------------------------------------------------------


def test_case09_multisolve_stable_dims(tmp_path):
    # Global vocabulary spans three periods; entity authors p2040 only.
    periods = ["p2025", "p2030", "p2040"]
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_future", _m(["p2040"], [99.0], "period")),
         ("fg_now", _m(["p2025", "p2030"], [1.0, 2.0], "period"))],
    )
    reader, enums = _reader(url, "with_enums", periods=periods)
    # Sub-solve A: active periods {p2025, p2030} (fg_future outside).
    dtA = _dt(enums, periods=["p2025", "p2030"])
    # Sub-solve B: active periods {p2040}.
    dtB = _dt(enums, periods=["p2040"])
    with _maybe_global(enums):
        resA = resolve_param_shape(reader, "flowGroup", "min_instant_flow",
                                   period_filter=dtA)
        resB = resolve_param_shape(reader, "flowGroup", "min_instant_flow",
                                   period_filter=dtB)
        pA = broadcast_to_period_time(resA, "group", dtA)
        pB = broadcast_to_period_time(resB, "group", dtB)
    assert resA.shape is resB.shape is Shape.MAP_PERIOD
    # Param dims stable across sub-solves (does not raise on out-of-solve).
    assert pA.dims == pB.dims == ("group", "d")


# ---------------------------------------------------------------------------
# Case 10 — TimeSeries row mixed with a Map(period) row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("mode", MODES)
def test_case10_timeseries_plus_period(tmp_path, order, mode):
    ts = _ts(["2025-01-01T00:00:00", "2025-01-01T01:00:00"], [10.0, 20.0])
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_ts", ts),
         ("fg_period", _m(PERIODS, [50.0, 60.0], "period"))],
        order=order,
    )
    reader, _enums = _reader(url, mode)
    # The TimeSeries level classifies to the t axis regardless of enums
    # (classification of a TimeSeries level is label-free).
    assert reader.parameter_shape_info(
        "flowGroup", "min_instant_flow") == ["period", "time"]
    assert reader.parameter_shape_variants("flowGroup", "min_instant_flow") == {
        ("period",), ("time",)}
    if mode == "no_enums":
        # Without the scenario trim, the placed frame shows the TimeSeries
        # row in the t column with a null period.
        expl = reader.parameter_explicit("flowGroup", "min_instant_flow")
        fts = expl.filter(pl.col("name") == "fg_ts")
        assert fts.get_column("period").to_list() == [None, None]
        assert fts.get_column("t").null_count() == 0


# ---------------------------------------------------------------------------
# Case 11 — null-split fix: inactive tokens are not broadcast (no dup keys)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_case11_inactive_token_not_broadcast(tmp_path, mode):
    # fg_time authors an out-of-vocabulary / inactive timestep t99.
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_time", _m(["t01", "t99"], [10.0, 20.0], "time")),
         ("fg_period", _m(PERIODS, [50.0, 60.0], "period"))],
    )
    reader, enums = _reader(url, mode)
    _res, mat = _broadcast_rows(reader, enums, "flowGroup", "min_instant_flow")
    # No duplicate (group, d, t) keys produced by a broadcast of t99.
    assert mat.height == mat.select("group", "d", "t").unique().height
    ft = mat.filter(pl.col("group") == "fg_time")
    # t99 is inactive ⇒ only the t01 column survives (one per period).
    assert set(str(x) for x in ft.get_column("t").to_list()) == {"t01"}


# ---------------------------------------------------------------------------
# Case 12 — period-only parameter with a time map → FlexToolConfigError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_case12_period_only_explicit_time_map_rejected(tmp_path, mode):
    url = _build_db(
        tmp_path / "db.sqlite", "group", "co2_max_period",
        [("g_time", _m(TIMESTEPS, [10.0, 20.0], "time"))],
    )
    reader, enums = _reader(url, mode)
    dt = _dt(enums)
    with _maybe_global(enums):
        with pytest.raises(FlexToolConfigError):
            resolve_param_shape(reader, "group", "co2_max_period",
                                period_filter=dt)


def test_case12_period_only_silent_time_keys_rejected(tmp_path):
    # Silent index_name but keys are timesteps → classified t → rejected.
    url = _build_db(
        tmp_path / "db.sqlite", "group", "co2_max_period",
        [("g_silent_time", _m(TIMESTEPS, [10.0, 20.0]))],
    )
    reader, enums = _reader(url, "with_enums")
    dt = _dt(enums)
    with _maybe_global(enums):
        with pytest.raises(FlexToolConfigError):
            resolve_param_shape(reader, "group", "co2_max_period",
                                period_filter=dt)


# ---------------------------------------------------------------------------
# Case 13 — explicit non-axis label → FlexToolConfigError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_case13_explicit_non_axis_label_rejected(tmp_path, mode):
    url = _build_db(
        tmp_path / "db.sqlite", "flowGroup", "min_instant_flow",
        [("fg_branch", _m(["b1", "b2"], [1.0, 2.0], "branch"))],
    )
    reader, _enums = _reader(url, mode)
    with pytest.raises(FlexToolConfigError):
        reader.parameter_explicit("flowGroup", "min_instant_flow")
