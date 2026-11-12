"""Per-entity silent-label fallback in ``resolve_param_shape`` (#167, #168).

``resolve_param_shape`` classifies a whole ``(entity_class, parameter_name)`` frame at once: when the
registry admits both ``MAP_PERIOD`` and ``MAP_TIME`` at depth 1 (the 21 keys of ``PARAM_ALLOWED_SHAPES``
that do) and the DB carried no ``index_name`` label, the whole-frame value-domain probe
(``_disambiguate_shape_by_value_domain``) looks at the UNION of every entity's keys.  When that union
still cannot decide (no overlap at all, or overlap on both axes), the parameter resolved to ``None`` for
EVERY entity -- including a scalar entity that authored nothing ambiguous at all.

The fix is per entity (``_silent_entity_axes`` + ``_per_entity_resolved``, ``flextool/engine/input/
_param_shapes.py``): an entity whose silent-label map has no key that is a period or a timestep of the
active solve has no value in this solve (dropped, same as "not authored"); when the remaining entities
split across both axes, the frame is built directly from each entity's own classified axis.

Uses ``InMemoryReader`` frames with a literal ``"x"`` column (Spine's silent-default ``index_name``,
mirroring :meth:`SpineDbReader.parameter_entity_label_paths`'s raw-label contract) and a 1-period x
3-step ``period_filter``.  Several cases below note that the WHOLE-FRAME probe already resolves them
before reaching the per-entity fallback at all -- i.e. they are guards on existing behaviour, not a
fail-before for the new code; each such case says so explicitly.  The real regression (one silent
no-key entity dropped) needs the union of every entity's authored keys to miss both the solve's periods
and its timesteps, or to hit both (two entities on different axes) -- otherwise the existing whole-frame
probe already resolves the parameter directly and :func:`_silent_entity_axes` never runs.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._inmemory_reader import InMemoryReader
from flextool.engine.input._param_shapes import (
    Shape,
    activate_stochastic_time_source,
    broadcast_to_period_time,
    deactivate_stochastic_time_source,
    resolve_param_shape,
)
from flextool.engine.preprocess.params._branch import (
    activate_branch_anchor_pairs,
    deactivate_branch_anchor_pairs,
)
from flextool.spinedb_backend._axis_enums import FlexDataIntegrityError

# 1 period (p1), 3 timesteps (t1..t3) -- matches the design's "1-period x 3-step period_filter".
DT = pl.DataFrame({"d": ["p1", "p1", "p1"], "t": ["t1", "t2", "t3"]})


def _reader(frame: pl.DataFrame, entity_class: str = "node") -> InMemoryReader:
    ent_cols = [c for c in frame.columns if c not in ("x", "value")]
    ents = frame.select(ent_cols).unique(maintain_order=True)
    return InMemoryReader(entities={entity_class: ents}, parameters={(entity_class, "penalty_up"): frame})


def _values_by_entity(frame: pl.DataFrame, ent_cols: "tuple[str, ...]" = ("name",)) -> dict:
    """``{entity tuple: value}``, asserting at most one row per entity (every case here is depth <= 1 per
    entity: a scalar or a single period/time key, never several)."""
    cols = list(ent_cols)
    assert not frame.select(cols).is_duplicated().any(), "more than one row for some entity"
    return {tuple(r[c] for c in cols): r["value"] for r in frame.select(*cols, "value").iter_rows(named=True)}


# ---------------------------------------------------------------------------
# (a)-(f): single-dim ``node.penalty_up``
# ---------------------------------------------------------------------------


def test_a_scalar_plus_silent_no_key_resolves_scalar_only() -> None:
    """A is scalar (no map); B's silent key ("p_other") matches neither the solve's period nor its
    timesteps -- dropped.  Before the fix: ``None`` for the whole frame (the union probe sees only
    "p_other", which matches neither vocabulary either)."""
    frame = pl.DataFrame({"name": ["A", "B"], "x": [None, "p_other"], "value": [5.0, 7.0]})
    resolved = resolve_param_shape(_reader(frame), "node", "penalty_up", period_filter=DT)
    assert resolved is not None
    assert resolved.shape is Shape.SCALAR
    assert _values_by_entity(resolved.frame) == {("A",): 5.0}


def test_b_silent_period_plus_silent_no_key_is_a_guard_not_a_fail_before() -> None:
    """A's key is "p1" (a solve period); B's is "p_other".  The UNION {"p1", "p_other"} already overlaps
    the period vocabulary and not the timestep one, so the EXISTING whole-frame probe
    (``_disambiguate_shape_by_value_domain``) resolves this to ``MAP_PERIOD`` directly -- the new
    per-entity fallback never runs.  B's "p_other" row is still in the raw resolved frame (resolver-level
    filtering by period membership is not this function's job); it is dropped later by the period-filter
    join inside ``broadcast_to_period_time``, both before and after the fix.  So this case is a guard on
    existing behaviour, not a fail-before for #167/#168 -- recorded per the design's instruction."""
    frame = pl.DataFrame({"name": ["A", "B"], "x": ["p1", "p_other"], "value": [5.0, 7.0]})
    resolved = resolve_param_shape(_reader(frame), "node", "penalty_up", period_filter=DT)
    assert resolved is not None
    assert resolved.shape is Shape.MAP_PERIOD
    assert _values_by_entity(resolved.frame, ("name", "period")) == {("A", "p1"): 5.0, ("B", "p_other"): 7.0}, (
        "raw frame still carries B; unfiltered")
    out = broadcast_to_period_time(resolved, "n", DT)
    assert set(out.frame["n"].to_list()) == {"A"}, "B's key never matches DT's period in the join either"


def test_c_silent_period_and_silent_time_and_silent_no_key() -> None:
    """A's key is "p1" (period), B's keys are "t1"/"t2" (time), C's is "p_other" (neither).  The union now
    hits BOTH axes (via A and B), so the whole-frame probe stays ambiguous and the per-entity fallback
    runs: A -> period, B -> time (two rows), C dropped.  Before the fix: ``None`` for the whole frame."""
    frame = pl.DataFrame({"name": ["A", "B", "B", "C"], "x": ["p1", "t1", "t2", "p_other"],
                          "value": [5.0, 1.0, 2.0, 7.0]})
    resolved = resolve_param_shape(_reader(frame), "node", "penalty_up", period_filter=DT)
    assert resolved is not None
    assert resolved.shape is Shape.MAP_PERIOD_TIME
    rows = resolved.frame.sort("name", "t" if "t" in resolved.frame.columns else "name")
    assert set(rows["name"].to_list()) == {"A", "B"}, "C has no key in the solve: absent"
    a = rows.filter(pl.col("name") == "A")
    assert a["period"].to_list() == ["p1"] and a["t"].to_list() == [None]
    b = rows.filter(pl.col("name") == "B").sort("t")
    assert b["t"].to_list() == ["t1", "t2"] and b["period"].to_list() == [None, None]


def test_d_both_axes_on_one_entity_raises() -> None:
    """D's silent map carries one key matching the period vocabulary and another matching the timestep
    vocabulary: ambiguous for a single entity, same rule ``SpineDbReader._classify_pt_level`` already
    raises for one row.  Before the fix: ``None`` (silently wrong), not an error."""
    frame = pl.DataFrame({"name": ["D", "D"], "x": ["p1", "t1"], "value": [1.0, 2.0]})
    with pytest.raises(FlexDataIntegrityError, match="D.*BOTH"):
        resolve_param_shape(_reader(frame), "node", "penalty_up", period_filter=DT)


def test_e_stochastic_entity_plus_scalar_plus_silent_no_key(caplog) -> None:
    """A ``STOCHASTIC_TIME_PARAMS`` key (``unit.availability``): ``u1`` authors a stochastic-time (rank-3)
    map, ``u2`` is scalar, ``u3`` is a silent map with no key in the solve.  The stochastic entity stays in
    ``stochastic_entities`` (resolved on the view without it, per the existing stochastic branch); within
    that view, the fallback drops ``u3`` and keeps ``u2``."""

    class _Provider:
        def __init__(self, frames: dict) -> None:
            self._frames = frames

        def has(self, name: str) -> bool:
            return name in self._frames

        def get(self, name: str):
            return self._frames[name]

    fold = pl.DataFrame(
        [("u1", "availability", d, t, repr(v)) for (d, t), v in
         zip(DT.select("d", "t").iter_rows(), (0.8, 0.7, 0.6))],
        schema=["process", "param", "period", "time", "value"], orient="row",
    )
    ents = pl.DataFrame({"name": ["u1", "u2", "u3"]})
    rank3 = pl.DataFrame({"name": ["u1"] * 3, "branch": ["hi"] * 3, "analysis_time": ["t1"] * 3,
                         "t": ["t1", "t2", "t3"], "value": [0.8, 0.7, 0.6]})
    other = pl.DataFrame({"name": ["u2", "u3"], "x": [None, "p_other"], "value": [1.0, 0.5]})
    frame = pl.concat([rank3, other], how="diagonal_relaxed")
    src = InMemoryReader(entities={"unit": ents}, parameters={("unit", "availability"): frame})

    token = activate_stochastic_time_source(_Provider({"solve_data/pdtProcess": fold}))
    try:
        resolved = resolve_param_shape(src, "unit", "availability", period_filter=DT)
    finally:
        deactivate_stochastic_time_source(token)
    assert resolved is not None
    assert resolved.stochastic_entities == frozenset({("u1",)})
    assert resolved.shape is Shape.SCALAR
    assert _values_by_entity(resolved.frame) == {("u2",): 1.0}, "u3 (no key): absent; u1: stochastic"


def test_f_no_period_filter_keeps_todays_none() -> None:
    """Without an active ``period_filter`` the fallback cannot classify anything (no vocabulary to probe
    against); the documented contract is unchanged: ``None``."""
    frame = pl.DataFrame({"name": ["A", "B"], "x": [None, "p_other"], "value": [5.0, 6.0]})
    assert resolve_param_shape(_reader(frame), "node", "penalty_up", period_filter=None) is None


# ---------------------------------------------------------------------------
# (g)-(h): multi-dim classes and flowGroup (critique item 4)
# ---------------------------------------------------------------------------


def test_g_two_dim_class_drops_by_entity_tuple() -> None:
    """``unit__outputNode.other_operational_cost`` (2-dim): a scalar entity, a silent period entity and a
    silent no-key entity, so the fallback must match entity TUPLES (unit, node) by position, not by a
    single ``name`` column.  Mixing a period and a time entity (critique item 4's dims, extended so the
    whole-frame probe actually stays ambiguous -- see the module docstring) proves the join survives both
    an extra dim and, below, an Enum-cast frame."""
    ents = pl.DataFrame({"unit": ["u1", "u2", "u3", "u4"], "node": ["n1"] * 4})
    frame = pl.DataFrame({"unit": ["u1", "u2", "u3", "u4"], "node": ["n1"] * 4,
                          "x": [None, "p1", "t1", "p_other"], "value": [5.0, 6.0, 7.0, 8.0]})
    src = InMemoryReader(entities={"unit__outputNode": ents},
                         parameters={("unit__outputNode", "other_operational_cost"): frame})
    resolved = resolve_param_shape(src, "unit__outputNode", "other_operational_cost", period_filter=DT)
    assert resolved is not None and resolved.shape is Shape.MAP_PERIOD_TIME
    units = set(resolved.frame["unit"].to_list())
    assert units == {"u1", "u2", "u3"}, "u4 (no key in the solve) is absent"

    # Entity-dim columns Enum-cast (the axis contract's post-resolution cast) must not break the join.
    enum_u = pl.Enum(["u1", "u2", "u3", "u4"])
    frame_enum = frame.with_columns(pl.col("unit").cast(enum_u))
    ents_enum = ents.with_columns(pl.col("unit").cast(enum_u))
    src_enum = InMemoryReader(entities={"unit__outputNode": ents_enum},
                              parameters={("unit__outputNode", "other_operational_cost"): frame_enum})
    resolved_enum = resolve_param_shape(src_enum, "unit__outputNode", "other_operational_cost",
                                        period_filter=DT)
    assert resolved_enum is not None
    assert set(str(u) for u in resolved_enum.frame["unit"].to_list()) == units


def test_g_three_dim_class_drops_by_entity_tuple() -> None:
    """``reserve__upDown__group.reservation`` (3-dim): same pattern, one more entity-dim column."""
    ents = pl.DataFrame({"reserve": ["r1"] * 4, "upDown": ["up"] * 4, "group": ["g1", "g2", "g3", "g4"]})
    frame = pl.DataFrame({"reserve": ["r1"] * 4, "upDown": ["up"] * 4, "group": ["g1", "g2", "g3", "g4"],
                          "x": [None, "p1", "t1", "p_other"], "value": [5.0, 6.0, 7.0, 8.0]})
    src = InMemoryReader(entities={"reserve__upDown__group": ents},
                         parameters={("reserve__upDown__group", "reservation"): frame})
    resolved = resolve_param_shape(src, "reserve__upDown__group", "reservation", period_filter=DT)
    assert resolved is not None and resolved.shape is Shape.MAP_PERIOD_TIME
    assert set(resolved.frame["group"].to_list()) == {"g1", "g2", "g3"}, "g4 (no key in the solve) is absent"


def test_h_flow_group_drops_by_entity() -> None:
    """``flowGroup.max_instant_flow``: the carve-out gave this class its own ``name`` column; confirm the
    fallback resolves it the same as any other 0-dim class."""
    ents = pl.DataFrame({"name": ["f1", "f2", "f3", "f4"]})
    frame = pl.DataFrame({"name": ["f1", "f2", "f3", "f4"], "x": [None, "p1", "t1", "p_other"],
                          "value": [5.0, 6.0, 7.0, 8.0]})
    src = InMemoryReader(entities={"flowGroup": ents}, parameters={("flowGroup", "max_instant_flow"): frame})
    resolved = resolve_param_shape(src, "flowGroup", "max_instant_flow", period_filter=DT)
    assert resolved is not None and resolved.shape is Shape.MAP_PERIOD_TIME
    assert set(resolved.frame["name"].to_list()) == {"f1", "f2", "f3"}


# ---------------------------------------------------------------------------
# Critique item 3 — classify against the periods a labelled twin would keep (branch anchors)
# ---------------------------------------------------------------------------


def test_anchor_period_outside_d_is_still_kept() -> None:
    """A labelled ``{anchor: v}`` row keeps its value for the branch periods even when the anchor itself
    has left the active solve's ``d`` (``inherit_active_branch_periods`` / ``_with_branch_periods``).  The
    silent-label fallback must classify against the SAME wider universe (solve's ``d`` plus the active
    branch-anchor pairs' anchors), or a silent entity would be dropped where its labelled twin is kept.
    Here the solve's only period is the branch "br1"; "p1" is its anchor and is NOT in ``d`` on its own."""
    frame = pl.DataFrame({"name": ["A", "B"], "x": [None, "p1"], "value": [5.0, 6.0]})
    src = _reader(frame)
    dt = pl.DataFrame({"d": ["br1", "br1"], "t": ["t1", "t2"]})
    pairs = pl.DataFrame({"anchor": ["p1"], "br": ["br1"]})

    # Without the active pairs, "p1" matches neither br1 nor the timesteps: dropped.
    resolved_inactive = resolve_param_shape(src, "node", "penalty_up", period_filter=dt)
    assert resolved_inactive is not None
    assert _values_by_entity(resolved_inactive.frame) == {("A",): 5.0}

    token = activate_branch_anchor_pairs(pairs)
    try:
        resolved_active = resolve_param_shape(src, "node", "penalty_up", period_filter=dt)
    finally:
        deactivate_branch_anchor_pairs(token)
    assert resolved_active is not None
    assert resolved_active.shape is Shape.MAP_PERIOD
    assert _values_by_entity(resolved_active.frame, ("name", "period")) == {
        ("A", None): 5.0, ("B", "p1"): 6.0}
