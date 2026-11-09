"""The solve-step aggregation of ``new_stepduration`` (C-13): the registry and the one aggregation function.

A solve with ``new_stepduration`` combines the time steps of its timeline into longer steps (labelled by their first
original step).  Every time series takes one value per new step by its parameter's kind
(``_solve_step_aggregation.TIME_AGGREGATION``):

    MEAN      v_s = Σ v·Δ / Σ Δ (rounded to 6 decimals)        prices, limits, availability, ...
    HARMONIC  v_s = 1 / (Σ Δ/v / Σ Δ)  (the mean of 1/v rounded)  efficiencies
    SUM       v_s = Σ v                                        inflow, energy_margin_adder
    LAST      v_s = the value of the last base step with one   storage_state_reference_value
    and MEAN / HARMONIC / LAST: all values equal → the value itself

The registry is complete (every schema map parameter is a time series with a kind, or listed as not one), the CSV
frames of the provider and the DB-direct broadcast use the same function (the same bits), and the broadcast refuses
a parameter without a kind while steps are active.  Values worked out by hand in each docstring.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    STOCHASTIC_TIME_PARAMS,
    ResolvedShape,
    Shape,
    broadcast_to_period_time,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._solve_step_aggregation import (
    NOT_TIME_SERIES,
    TIME_AGGREGATION,
    Agg,
    activate_solve_step_aggregation,
    aggregate_to_solve_steps,
    combine,
    deactivate_solve_step_aggregation,
)
from flextool.engine.input._timeline import _CSV_SERIES, _aggregate_csv_series, csv_series_kind

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "flextool" / "schemas" / "spinedb_schema.json"
_TIME_SHAPES = {Shape.MAP_TIME, Shape.MAP_PERIOD_TIME, Shape.MAP_TIME_PERIOD}


def _schema_maps() -> set[tuple[str, str]]:
    data = json.loads(SCHEMA.read_text())
    return {(c, p) for c, p, t, _rank in data["parameter_types"] if t == "map"}


def _schema_params() -> set[tuple[str, str]]:
    data = json.loads(SCHEMA.read_text())
    return {(r[0], r[1]) for r in data["parameter_definitions"]}


# ---------------------------------------------------------------------------
# 6 (a)-(c): the registry is complete


def test_every_schema_map_is_classified_once() -> None:
    """(a) Every map parameter of the schema is in exactly one of TIME_AGGREGATION and NOT_TIME_SERIES."""
    maps = _schema_maps()
    both = set(TIME_AGGREGATION) & NOT_TIME_SERIES
    missing = maps - set(TIME_AGGREGATION) - NOT_TIME_SERIES
    assert not both, f"in both: {sorted(both)}"
    assert not missing, f"schema map parameters without a TIME_AGGREGATION or NOT_TIME_SERIES entry: {sorted(missing)}"


def test_every_time_shaped_parameter_has_a_kind() -> None:
    """(b) Every PARAM_ALLOWED_SHAPES key that takes a time or period + time map, and every STOCHASTIC_TIME_PARAMS
    key, has a kind."""
    timed = {k for k, shapes in PARAM_ALLOWED_SHAPES.items() if _TIME_SHAPES & set(shapes)}
    assert timed - set(TIME_AGGREGATION) == set()
    assert set(STOCHASTIC_TIME_PARAMS) - set(TIME_AGGREGATION) == set()


def test_every_registry_key_is_a_schema_map() -> None:
    """(c) Every key of both registries is a map parameter of the schema (no stale or misspelt entry)."""
    assert set(TIME_AGGREGATION) - _schema_maps() == set()
    assert NOT_TIME_SERIES - _schema_maps() == set()
    assert (set(TIME_AGGREGATION) | NOT_TIME_SERIES) <= _schema_params()


def test_the_decided_kinds() -> None:
    """The orchestrator decisions of C-13: LAST for the reference state, SUM for the two energies per step,
    HARMONIC for both efficiencies, MEAN for the instant-flow limits."""
    assert TIME_AGGREGATION[("node", "storage_state_reference_value")] is Agg.LAST
    assert {k for k, v in TIME_AGGREGATION.items() if v is Agg.SUM} == {
        ("node", "inflow"), ("node", "energy_margin_adder")}
    assert {k for k, v in TIME_AGGREGATION.items() if v is Agg.HARMONIC} == {
        ("unit", "efficiency"), ("connection", "efficiency"), ("unit", "efficiency_at_min_load")}
    assert TIME_AGGREGATION[("flowGroup", "max_instant_flow")] is Agg.MEAN
    assert TIME_AGGREGATION[("flowGroup", "min_instant_flow")] is Agg.MEAN


def test_csv_files_agree_on_each_parameter() -> None:
    """(S11) Every (file, parameter) of the provider frames has one kind: the classes that share a file (unit and
    connection rows of ``pt_process``, group and flowGroup rows of ``pt_group``) agree."""
    for file, (classes, fixed) in _CSV_SERIES.items():
        params = {fixed} if fixed is not None else {p for c, p in TIME_AGGREGATION if c in classes}
        for param in params:
            kinds = {TIME_AGGREGATION[(c, param)] for c in classes if (c, param) in TIME_AGGREGATION}
            assert len(kinds) == 1, (file, param, kinds)
            assert csv_series_kind(file, param)[1] in kinds


# ---------------------------------------------------------------------------
# combine: hand-derived values


def test_combine_by_kind() -> None:
    """2 h steps (equal durations): MEAN 100/300 → 200; HARMONIC 0.4/0.5 → the mean of 1/η is (2.5 + 2.0)/2 = 2.25,
    so η = 1/2.25 and 1/η = 2.25 exactly (critique M1: not the mean η 0.45, whose 1/η is 2.2̅); SUM −100/−300 →
    −400; LAST 0.5/0.7 → 0.7."""
    assert combine(Agg.MEAN, [100.0, 300.0], [1.0, 1.0]) == 200.0
    eta = combine(Agg.HARMONIC, [0.4, 0.5], [1.0, 1.0])
    assert 1.0 / eta == 2.25
    assert combine(Agg.SUM, [-100.0, -300.0], [1.0, 1.0]) == -400.0
    assert combine(Agg.LAST, [0.5, 0.7], [1.0, 1.0]) == 0.7


def test_duration_weighting() -> None:
    """(test 2) A 1 h and a 3 h base step in one 4 h step: (1·100 + 3·300)/4 = 250 (the plain mean would be 200);
    HARMONIC 0.4 (1 h) / 0.5 (3 h): (1·2.5 + 3·2.0)/4 = 2.125 = 1/η."""
    assert combine(Agg.MEAN, [100.0, 300.0], [1.0, 3.0]) == 250.0
    assert 1.0 / combine(Agg.HARMONIC, [0.4, 0.5], [1.0, 3.0]) == 2.125


@pytest.mark.parametrize("value", [0.1, 0.7, 0.37, 1.0 / 3.0, 0.1234567])
@pytest.mark.parametrize("kind", [Agg.MEAN, Agg.HARMONIC, Agg.LAST])
def test_equal_values_stay_exact(kind, value) -> None:
    """(M3) Three equal base values give the value itself (no arithmetic, no rounding): Σ[a]·3/3 ≠ a for a = 0.1,
    0.7, 0.37, and round(1/3, 6) ≠ 1/3, so the arithmetic would change a constant authored as a map."""
    assert combine(kind, [value] * 3, [1.0] * 3) == value
    assert combine(kind, [value] * 3, [1.0, 2.0, 1.0]) == value


def test_sum_of_equal_values_is_summed() -> None:
    """SUM is an energy per base step: two equal hours −120 / −120 give −240, never the value itself (the
    equal-values rule is for the rates)."""
    assert combine(Agg.SUM, [-120.0, -120.0], [1.0, 1.0]) == -240.0


def test_harmonic_refuses_a_non_positive_efficiency() -> None:
    with pytest.raises(ValueError, match="positive"):
        combine(Agg.HARMONIC, [0.4, 0.0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# aggregate_to_solve_steps: grouping, LAST by timeline position, partial steps


def _base(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """``new_step_base_steps`` of ``[(time, new_step, duration)]`` in timeline order."""
    count = {}
    for _t, s, _d in rows:
        count[s] = count.get(s, 0) + 1
    return pl.DataFrame([(t, s, float(count[s]), d, i) for i, (t, s, d) in enumerate(rows)],
                        schema={"time": pl.Utf8, "new_step": pl.Utf8, "value": pl.Float64, "duration": pl.Float64,
                                "pos": pl.Int64}, orient="row")


def test_last_follows_the_timeline_not_the_rows() -> None:
    """LAST takes the last base step in timeline order (``pos``), not the frame's row order nor the label order:
    steps h9, h10 form new step h9; the frame lists h10 first.  h10 is later in the timeline → 0.7 (a string sort
    would put h10 before h9 and return 0.5)."""
    base = _base([("h8", "h8", 1.0), ("h9", "h9", 1.0), ("h10", "h9", 1.0)])
    frame = pl.DataFrame({"name": ["n", "n"], "t": ["h10", "h9"], "value": [0.7, 0.5]})
    out = aggregate_to_solve_steps(frame, Agg.LAST, ["name"], base, keys=("node", "storage_state_reference_value"))
    assert out.rows() == [("n", "h9", 0.7)]


def test_partial_step_takes_the_steps_it_has(caplog) -> None:
    """(S14) Availability 0.5 at h1 only of the 2 h step h1 (h2 has no value): the step takes 0.5, not
    (0.5 + default)/2, and one warning names the parameter (once per run, also for a second entity)."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0), ("h3", "h3", 1.0), ("h4", "h3", 1.0)])
    frame = pl.DataFrame({"name": ["u", "u", "u", "v"], "t": ["h1", "h3", "h4", "h1"],
                          "value": [0.5, 0.2, 0.4, 0.9]})
    notes: set = set()
    with caplog.at_level(logging.WARNING):
        out = aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=("unit", "availability"), notes=notes)
        aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=("unit", "availability"), notes=notes)
    assert out.rows() == [("u", "h1", 0.5), ("u", "h3", 0.3), ("v", "h1", 0.9)]
    msgs = [r.message for r in caplog.records if "unit.availability" in r.message]
    assert len(msgs) == 1 and "only some of its time steps" in msgs[0]


def test_varying_min_load_warns_once(caplog) -> None:
    """(M2) A min_load that varies within a solve step makes the min_load_efficiency curve approximate: one
    warning per run naming the unit; a constant min_load is silent."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    notes: set = set()
    with caplog.at_level(logging.WARNING):
        aggregate_to_solve_steps(pl.DataFrame({"name": ["u"] * 2, "t": ["h1", "h2"], "value": [0.3, 0.3]}),
                                 Agg.MEAN, ["name"], base, keys=("unit", "min_load"), notes=notes)
        assert not caplog.records
        out = aggregate_to_solve_steps(pl.DataFrame({"name": ["u"] * 2, "t": ["h1", "h2"], "value": [0.3, 0.5]}),
                                       Agg.MEAN, ["name"], base, keys=("unit", "min_load"), notes=notes)
    assert out.rows() == [("u", "h1", 0.4)]
    assert [r.message for r in caplog.records if "min_load varies" in r.message and "u" in r.message]


# ---------------------------------------------------------------------------
# The DB-direct broadcast (hook 1) and the CSV frames: one function, the same bits


def _armed(base: pl.DataFrame):
    return activate_solve_step_aggregation(base, set())


def _resolved(frame: pl.DataFrame, key: tuple[str, str], shape: Shape = Shape.MAP_TIME) -> ResolvedShape:
    return ResolvedShape(shape=shape, frame=frame, entity_dim_columns=("name",),
                         period_index_column="period" if "period" in frame.columns else None,
                         time_index_column="t", key=key)


def _collect(param) -> pl.DataFrame:
    f = param.frame
    f = f.collect() if isinstance(f, pl.LazyFrame) else f
    return f.select(*(pl.col(c).cast(pl.Utf8) if c != "value" else pl.col(c) for c in f.columns))


def test_broadcast_takes_the_weighted_step_mean() -> None:
    """(test 2, DB path) A time map 100 (1 h) / 300 (3 h) on one 4 h step: 250, the value the CSV path gives for the
    same rows (one implementation)."""
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 3.0)])
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    tok = _armed(base)
    try:
        out = _collect(broadcast_to_period_time(
            _resolved(pl.DataFrame({"name": ["g", "g"], "t": ["t1", "t2"], "value": [100.0, 300.0]}),
                      ("group", "co2_price")), "g", dt))
    finally:
        deactivate_solve_step_aggregation(tok)
    assert out.rows() == [("g", "t1", 250.0)]
    csv = _aggregate_csv_series("pt_profile.csv", pl.DataFrame(
        {"profile": ["f", "f"], "time": ["t1", "t2"], "pt_profile": ["100", "300"]}), base, set())
    assert csv == [["f", "t1", 250.0]]


def test_broadcast_mean_includes_a_zero_hour() -> None:
    """(M4) co2_price 0 / 30 with ``filter_zero``: the step mean is (0 + 30)/2 = 15, not 30 (the zero filter runs
    after the aggregation)."""
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0)])
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    tok = _armed(base)
    try:
        out = _collect(broadcast_to_period_time(
            _resolved(pl.DataFrame({"name": ["g", "g"], "t": ["t1", "t2"], "value": [0.0, 30.0]}),
                      ("group", "co2_price")), "g", dt, filter_zero=True))
    finally:
        deactivate_solve_step_aggregation(tok)
    assert out.rows() == [("g", "t1", 15.0)]


def test_broadcast_keeps_period_rows_and_aggregates_per_period() -> None:
    """A mixed period + time frame: the (p1, t1..t2) rows give 20 per period ((10 + 30)/2, (50 + 70)/2 for p2),
    the period-only row of entity h (no t) passes through and is broadcast to the step."""
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0)])
    dt = pl.DataFrame({"d": ["p1", "p2"], "t": ["t1", "t1"]})
    frame = pl.DataFrame({"name": ["g", "g", "g", "g", "h"], "period": ["p1", "p1", "p2", "p2", "p1"],
                          "t": ["t1", "t2", "t1", "t2", None], "value": [10.0, 30.0, 50.0, 70.0, 5.0]})
    tok = _armed(base)
    try:
        out = _collect(broadcast_to_period_time(_resolved(frame, ("group", "co2_price"), Shape.MAP_PERIOD_TIME),
                                                "g", dt))
    finally:
        deactivate_solve_step_aggregation(tok)
    assert sorted(out.rows()) == [("g", "p1", "t1", 20.0), ("g", "p2", "t1", 60.0), ("h", "p1", "t1", 5.0)]


def test_broadcast_without_a_kind_raises() -> None:
    """(d) While a solve's steps are armed, a time series whose key has no TIME_AGGREGATION entry stops the run;
    a SUM key (its constants need the base-step count) is refused by the generic broadcast."""
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0)])
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    frame = pl.DataFrame({"name": ["g", "g"], "t": ["t1", "t2"], "value": [1.0, 2.0]})
    tok = _armed(base)
    try:
        with pytest.raises(FlexToolConfigError, match="not in TIME_AGGREGATION"):
            broadcast_to_period_time(_resolved(frame, ("group", "no_such_series")), "g", dt)
        with pytest.raises(FlexToolConfigError, match="SUM"):
            broadcast_to_period_time(_resolved(frame, ("node", "inflow")), "g", dt)
    finally:
        deactivate_solve_step_aggregation(tok)


def test_nothing_armed_is_a_pass_through() -> None:
    """(test 5, unit level) Without armed steps (no new_stepduration, or an empty base-step frame) the broadcast
    keeps every authored hour."""
    dt = pl.DataFrame({"d": ["p1", "p1"], "t": ["t1", "t2"]})
    frame = pl.DataFrame({"name": ["g", "g"], "t": ["t1", "t2"], "value": [1.0, 2.0]})
    for base in (None, _base([])):
        tok = activate_solve_step_aggregation(base)
        try:
            out = _collect(broadcast_to_period_time(_resolved(frame, ("group", "co2_price")), "g", dt))
        finally:
            deactivate_solve_step_aggregation(tok)
        assert sorted(out.rows()) == [("g", "t1", 1.0), ("g", "t2", 2.0)]


def test_behaviour_change_notice_names_the_changed_entities(capsys) -> None:
    """With the marker ``solve_step_series_combined`` (an upgraded database), the notice names each parameter and
    entity whose step value is not its first base step's value: availability u 0.3 / 0.5 → 0.4 (changed), v 0.9 /
    0.9 → 0.9 (unchanged, not named).  Without the marker, or for a series earlier releases already combined
    (``report_changes=False``), nothing is printed."""
    from flextool.engine import _behaviour_changes as bc

    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    frame = pl.DataFrame({"name": ["u", "u", "v", "v"], "t": ["h1", "h2", "h1", "h2"], "value": [0.3, 0.5, 0.9, 0.9]})
    key = ("unit", "availability")
    try:
        bc.start_run({})
        aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=key)
        assert capsys.readouterr().out == ""
        bc.start_run({"solve_step_series_combined": "83"})
        aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=key, report_changes=False)
        assert capsys.readouterr().out == ""
        aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=key)
        out = capsys.readouterr().out
        assert "Affected parameters: unit.availability of u." in out
        assert "of v" not in out
    finally:
        bc.start_run(None)
