"""Stochastic-time maps of resolver-routed parameters: the per-entity split in ``_param_shapes`` (#78).

For a ``STOCHASTIC_TIME_PARAMS`` parameter, an entity whose value is a three-level map (``branch / analysis_time /
time``, or Spine's silent ``x``) is taken out of the period / time classifier's view; its values come from the
solve's emit fold, every other entity is read as before:

    value[e, d, t] = folded stochastic value of e at (d, t)    e authors a stochastic-time map
                   = the DB-direct value of e at (d, t)        any other shape

Unit tests on ``InMemoryReader`` (frames already in the reader's column shape) and on a ``SpineDbReader`` of a
JSON-built database, with a dict-backed provider standing in for the solve's emit.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._inmemory_reader import InMemoryReader
from flextool.engine.input._param_shapes import (
    PERIOD_TIME_PARAMS,
    STOCHASTIC_TIME_PARAMS,
    activate_stochastic_time_source,
    broadcast_to_period,
    broadcast_to_period_time,
    deactivate_stochastic_time_source,
    parameter_entity_values,
    parameter_has_values,
    resolve_param_shape,
    stochastic_time_entities,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.params._branch import (
    activate_branch_anchor_pairs,
    deactivate_branch_anchor_pairs,
)

REPO = Path(__file__).resolve().parents[2]
DT = pl.DataFrame({"d": ["p1", "p1", "p1_hi", "p1_hi"], "t": ["t1", "t2", "t1", "t2"]})

# A solve genuinely has stochastic branches in this module's tests only when a test cares about the
# "no active branch-anchor pairs" path (#82 decision 4 -- that path now raises); every other test
# activates this dummy pair so :func:`stochastic_time_values` reaches the behaviour it actually tests.
_DUMMY_BRANCH_PAIRS = pl.DataFrame({"anchor": ["p1"], "br": ["p1_hi"]})


class _Provider:
    """Stand-in for the solve's FlexDataProvider: ``has`` / ``get`` by key."""

    def __init__(self, frames: dict):
        self._frames = frames

    def has(self, name: str) -> bool:
        return name in self._frames

    def get(self, name: str):
        return self._frames[name]


def _pdt_process(rows: list[tuple]) -> pl.DataFrame:
    """Emit ``pdtProcess`` rows ``(process, param, period, time, value)``; values rendered with repr like the
    emit."""
    return pl.DataFrame([(p, par, d, t, repr(v)) for p, par, d, t, v in rows],
                        schema=["process", "param", "period", "time", "value"], orient="row")


FOLD = _pdt_process([("u1", "availability", d, t, v) for (d, t), v in
                     zip(DT.iter_rows(), (1.0, 1.0, 0.8, 0.7))]
                    + [("u2", "availability", d, t, 1.0) for d, t in DT.iter_rows()]
                    + [("u1", "efficiency", d, t, 0.5) for d, t in DT.iter_rows()])


@pytest.fixture
def emit_scope():
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": FOLD}))
    branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
    yield
    deactivate_branch_anchor_pairs(branch_token)
    deactivate_stochastic_time_source(token)


def _reader(avail: pl.DataFrame) -> InMemoryReader:
    return InMemoryReader(entities={"unit": pl.DataFrame({"name": ["u1", "u2", "u3"]})},
                          parameters={("unit", "availability"): avail})


def _rank3(name: str = "u1", labels=("branch", "analysis_time", "t")) -> pl.DataFrame:
    """u1 authors a stochastic-time map (InMemoryReader shape: one index column per level)."""
    return pl.DataFrame({"name": [name] * 2, labels[0]: ["hi", "hi"], labels[1]: ["t1", "t1"],
                         labels[2]: ["t1", "t2"], "value": [0.8, 0.7]})


def _with_time_map(frame: pl.DataFrame) -> pl.DataFrame:
    """Add u3 with a time map next to the rank-3 rows (null index columns where a row has no such level)."""
    other = pl.DataFrame({"name": ["u3", "u3"], "t": ["t1", "t2"], "value": [0.5, 0.6]})
    return pl.concat([frame, other], how="diagonal_relaxed")


def _cells(param) -> dict:
    f = param.frame if isinstance(param.frame, pl.DataFrame) else param.frame.collect()
    keys = [c for c in f.columns if c != "value"]
    assert not f.select(keys).is_duplicated().any(), "duplicate cells"
    return {tuple(str(x) for x in r[:-1]): r[-1] for r in f.select(*keys, "value").iter_rows()}


# ---------------------------------------------------------------------------
# Detection


def test_stochastic_entities_found_by_depth() -> None:
    src = _reader(_with_time_map(_rank3()))
    assert stochastic_time_entities(src, "unit", "availability") == frozenset({("u1",)})


def test_depth_two_or_less_is_not_stochastic() -> None:
    period_time = pl.DataFrame({"name": ["u1"], "period": ["p1"], "t": ["t1"], "value": [0.5]})
    assert stochastic_time_entities(_reader(period_time), "unit", "availability") == frozenset()


def test_parameters_outside_the_registry_are_never_split() -> None:
    src = InMemoryReader(entities={"connection": pl.DataFrame({"name": ["c1"]})},
                         parameters={("connection", "availability"): _rank3("c1")})
    assert ("connection", "availability") not in STOCHASTIC_TIME_PARAMS
    assert stochastic_time_entities(src, "connection", "availability") == frozenset()
    assert stochastic_time_entities(src, "connection", "no_such_parameter") == frozenset()


@pytest.mark.parametrize("labels", [("period", "time", "x"), ("branch", "period", "t")])
def test_depth_three_with_wrong_labels_raises(labels) -> None:
    """Only the stochastic triple (or silent labels) is exempt from the classifier."""
    with pytest.raises(FlexToolConfigError, match="stochastic-time value"):
        stochastic_time_entities(_reader(_rank3(labels=labels)), "unit", "availability")


def test_depth_four_raises() -> None:
    frame = _rank3().with_columns(pl.lit("q").alias("extra"))
    with pytest.raises(FlexToolConfigError, match="4-level map"):
        stochastic_time_entities(_reader(frame), "unit", "availability")


def test_depth_two_branch_label_still_raises() -> None:
    """A two-level map whose first label is ``branch`` is not stochastic time: the classifier error stays."""
    frame = pl.DataFrame({"name": ["u1"], "branch": ["hi"], "t": ["t1"], "value": [0.5]})
    src = _reader(frame)
    assert stochastic_time_entities(src, "unit", "availability") == frozenset()
    with pytest.raises(FlexToolConfigError):
        resolve_param_shape(src, "unit", "availability", period_filter=DT)


# ---------------------------------------------------------------------------
# The view without the stochastic entities


def test_without_entities_removes_only_that_parameter() -> None:
    eff = pl.DataFrame({"name": ["u1"], "value": [0.4]})
    src = InMemoryReader(entities={"unit": pl.DataFrame({"name": ["u1", "u3"]})},
                         parameters={("unit", "availability"): _with_time_map(_rank3()),
                                     ("unit", "efficiency"): eff})
    view = src.without_entities("unit", "availability", [("u1",)])
    avail = view.parameter("unit", "availability")
    assert set(avail["name"]) == {"u3"}
    assert set(avail.columns) == {"name", "t", "value"}, "index columns only the removed rows used are dropped"
    assert view.parameter("unit", "efficiency").equals(eff)
    assert src.parameter("unit", "availability").height == 4, "the source is not modified"


# ---------------------------------------------------------------------------
# Broadcast


def test_broadcast_merges_fold_and_direct(emit_scope) -> None:
    """u1 takes the emit fold, u3 its own time map; dims (p, d, t), sorted, no duplicates."""
    src = _reader(_with_time_map(_rank3()))
    resolved = resolve_param_shape(src, "unit", "availability", period_filter=DT)
    assert resolved.stochastic_entities == frozenset({("u1",)})
    out = broadcast_to_period_time(resolved, "p", DT)
    assert out.dims == ("p", "d", "t")
    assert isinstance(out.frame, pl.DataFrame)
    assert out.frame.equals(out.frame.sort("p", "d", "t"))
    want = {("u1", "p1", "t1"): 1.0, ("u1", "p1", "t2"): 1.0, ("u1", "p1_hi", "t1"): 0.8,
            ("u1", "p1_hi", "t2"): 0.7}
    want.update({("u3", d, t): (0.5 if t == "t1" else 0.6) for d, t in DT.iter_rows()})
    assert _cells(out) == want


def test_broadcast_filter_zero_applies_to_both_parts() -> None:
    fold = _pdt_process([("u1", "availability", d, t, 0.0 if t == "t1" else 0.9) for d, t in DT.iter_rows()])
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": fold}))
    branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
    try:
        frame = pl.concat([_rank3(), pl.DataFrame({"name": ["u3", "u3"], "t": ["t1", "t2"],
                                                     "value": [0.0, 0.6]})], how="diagonal_relaxed")
        resolved = resolve_param_shape(_reader(frame), "unit", "availability", period_filter=DT)
        out = broadcast_to_period_time(resolved, "p", DT, filter_zero=True)
    finally:
        deactivate_branch_anchor_pairs(branch_token)
        deactivate_stochastic_time_source(token)
    assert all(v != 0.0 for v in _cells(out).values())
    assert {k[2] for k in _cells(out)} == {"t2"}


def test_only_stochastic_entity_empty_view(emit_scope) -> None:
    """The parameter's only value is rank 3 (no default): the empty view still carries the stochastic entity
    (the stochastic branch runs before the empty-frame early return)."""
    resolved = resolve_param_shape(_reader(_rank3()), "unit", "availability", period_filter=DT)
    assert resolved is not None and resolved.frame.height == 0
    out = broadcast_to_period_time(resolved, "p", DT)
    assert set(k[0] for k in _cells(out)) == {"u1"}
    assert len(_cells(out)) == 4


def test_without_scope_raises() -> None:
    resolved = resolve_param_shape(_reader(_rank3()), "unit", "availability", period_filter=DT)
    with pytest.raises(FlexToolConfigError, match="only be read during load_flextool"):
        broadcast_to_period_time(resolved, "p", DT)


def test_missing_emit_frame_raises() -> None:
    token = activate_stochastic_time_source(_Provider({}))
    branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
    try:
        resolved = resolve_param_shape(_reader(_rank3()), "unit", "availability", period_filter=DT)
        with pytest.raises(FlexToolConfigError, match="solve_data/pdtProcess"):
            broadcast_to_period_time(resolved, "p", DT)
    finally:
        deactivate_branch_anchor_pairs(branch_token)
        deactivate_stochastic_time_source(token)


def test_entity_without_fold_rows_warns(caplog) -> None:
    """A stochastic entity outside the emit's domain gets no value, with a warning that names it."""
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": FOLD}))
    branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
    try:
        resolved = resolve_param_shape(_reader(_with_time_map(_rank3("u2_absent"))), "unit", "availability",
                                       period_filter=DT)
        with caplog.at_level(logging.WARNING):
            out = broadcast_to_period_time(resolved, "p", DT)
    finally:
        deactivate_branch_anchor_pairs(branch_token)
        deactivate_stochastic_time_source(token)
    assert {k[0] for k in _cells(out)} == {"u3"}
    assert any("u2_absent" in r.getMessage() and "no folded value" in r.getMessage() for r in caplog.records)


def test_broadcast_to_period_rejects_stochastic_entities() -> None:
    resolved = resolve_param_shape(_reader(_rank3()), "unit", "availability", period_filter=DT)
    with pytest.raises(FlexToolConfigError, match="broadcast_to_period_time"):
        broadcast_to_period(resolved, "p", DT)


# ---------------------------------------------------------------------------
# Shape-agnostic raw reads


def test_parameter_has_values_and_entity_values(emit_scope) -> None:
    src = _reader(_with_time_map(_rank3()))
    assert parameter_has_values(src, "unit", "availability")
    vals = parameter_entity_values(src, "unit", "availability")
    assert set(vals.columns) == {"name", "value"}
    by = vals.group_by("name").agg(pl.col("value").sort())
    assert dict(by.iter_rows()) == {"u1": [0.7, 0.8, 1.0, 1.0], "u3": [0.5, 0.6]}
    empty = InMemoryReader(entities={"unit": pl.DataFrame({"name": ["u1"]})},
                           parameters={("unit", "availability"): pl.DataFrame(
                               schema={"name": pl.Utf8, "value": pl.Float64})})
    assert not parameter_has_values(empty, "unit", "availability")
    assert parameter_entity_values(empty, "unit", "availability") is None


# ---------------------------------------------------------------------------
# SpineDbReader: the default broadcast of the view does not cover the stochastic entity


def _labelled_rank3(values: dict, timesteps: list[str]) -> list:
    from tests.engine._unit_db import b64

    def level(name, data):
        return {"type": "map", "index_type": "str", "index_name": name, "data": data}
    top = level("branch", [[b, level("analysis_time", [["t0001", level("time", [[t, v] for t in timesteps])]])]
                           for b, v in values.items()])
    return b64(top, "map")


def test_spinedb_default_does_not_reach_stochastic_entity(tmp_path_factory) -> None:
    """``coal_plant.availability`` rank 3 is the parameter's only value; the schema default 1.0 is broadcast
    over the other units, never over ``coal_plant`` (no duplicate keys), which takes the fold."""
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from tests.engine._unit_db import TIMESTEPS, make_db

    url, scen = make_db(tmp_path_factory, alt="c10_spine", base_chain=["init", "west", "coal", "coal_chp"],
                        values=[["unit", "coal_plant", "availability",
                                 _labelled_rank3({"hi": 0.8, "lo": 0.6}, TIMESTEPS[:2])]])
    src = SpineDbReader(url, scen)
    assert stochastic_time_entities(src, "unit", "availability") == frozenset({("coal_plant",)})
    units = sorted(src.entities("unit")["name"].to_list())
    assert "coal_plant" in units and len(units) > 1
    dt = pl.DataFrame({"d": ["p2020"] * 2, "t": TIMESTEPS[:2]})
    fold = _pdt_process([("coal_plant", "availability", "p2020", t, 0.8) for t in TIMESTEPS[:2]])
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": fold}))
    branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
    try:
        resolved = resolve_param_shape(src, "unit", "availability", period_filter=dt)
        out = _cells(broadcast_to_period_time(resolved, "p", dt))
        vals = parameter_entity_values(src, "unit", "availability")
    finally:
        deactivate_branch_anchor_pairs(branch_token)
        deactivate_stochastic_time_source(token)
    want = {(u, "p2020", t): (0.8 if u == "coal_plant" else 1.0) for u in units for t in TIMESTEPS[:2]}
    assert out == want
    assert sorted(vals.filter(pl.col("name") == "coal_plant")["value"].to_list()) == [0.8, 0.8]


# ---------------------------------------------------------------------------
# Registry pins


def test_registry_matches_schema_rank_three() -> None:
    """The registry is exactly the schema's rank-3 parameters among PERIOD_TIME_PARAMS (the price ladder's rank 3
    is a tier facet, not stochastic time)."""
    schema = json.loads((REPO / "flextool" / "schemas" / "spinedb_schema.json").read_text())
    rank3 = {(c, p) for c, p, kind, rank in schema["parameter_types"] if kind == "map" and rank == 3}
    assert set(STOCHASTIC_TIME_PARAMS) == rank3 & PERIOD_TIME_PARAMS


def test_registry_keys_have_a_pbt_spec() -> None:
    """Every key is read by a ``3d_map`` input-derivation spec (the emit's fold source)."""
    from flextool.input_derivation._specs import _PARAMETER_SPECS

    pbt = {tuple(cp) for spec in _PARAMETER_SPECS if "3d_map" in spec.get("filter_in_type", ())
           for cp in spec["cl_pars"]}
    missing = sorted(set(STOCHASTIC_TIME_PARAMS) - pbt)
    assert not missing, missing


def test_no_active_branch_pairs_raises(tmp_path_factory) -> None:
    """Owner decision #82(4): without active branch-anchor pairs there is no realized branch anywhere to take
    a rank-3 value from, so the engine raises (it no longer warns and falls back to the class default)."""
    src = _reader(_rank3("u2_dedupe"))
    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": FOLD}))
    try:
        with pytest.raises(FlexToolConfigError, match="no realized branch"):
            broadcast_to_period_time(resolve_param_shape(src, "unit", "availability", period_filter=DT), "p", DT)
    finally:
        deactivate_stochastic_time_source(token)


def test_warnings_dedupe_per_run(caplog) -> None:
    """With active branch-anchor pairs (so #82's "no branches" raise does not apply here), the remaining
    "no folded value" warning is deduplicated with the run's set passed to the scope: a second run (a fresh
    set) warns again, a second load of the same run (the same set) does not."""
    src = _reader(_rank3("u2_dedupe"))
    run_set: set = set()
    counts = []
    for warned in (run_set, run_set, set()):
        token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": FOLD}), warned)
        branch_token = activate_branch_anchor_pairs(_DUMMY_BRANCH_PAIRS)
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                broadcast_to_period_time(resolve_param_shape(src, "unit", "availability", period_filter=DT), "p",
                                         DT)
        finally:
            deactivate_branch_anchor_pairs(branch_token)
            deactivate_stochastic_time_source(token)
        counts.append(sum("u2_dedupe" in r.getMessage() for r in caplog.records))
    # each fresh set: the "no folded value" warning (1); the same run again: 0
    assert counts == [1, 0, 1], counts
