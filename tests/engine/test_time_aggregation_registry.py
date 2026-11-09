"""The solve-step aggregation of ``new_stepduration`` (C-13): the registry and the one aggregation function.

A solve with ``new_stepduration`` combines the time steps of its timeline into longer steps (labelled by their first
original step).  Every time series takes one value per new step by its parameter's kind
(``_solve_step_aggregation.TIME_AGGREGATION``):

    MEAN      v_s = Σ v·Δ / Σ Δ (the profile rounded to 6 dec.) prices, limits, availability, ...
    HARMONIC  v_s = 1 / (Σ Δ/v / Σ Δ)  (exact, not rounded)       efficiencies
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
    assert out.rows() == [("u", "h1", 0.5), ("u", "h3", (0.2 + 0.4) / 2), ("v", "h1", 0.9)]  # unrounded
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


# ---------------------------------------------------------------------------
# C-13 WP-A fix pass: exact means, the vectorised kernel, the roll's steps, timeline edge cases


def test_harmonic_is_exact_for_large_efficiencies() -> None:
    """(M2) HARMONIC is the reciprocal of the exact mean of 1/η, without rounding: η 250 / 300 → 2/(1/250 + 1/300)
    = 2/(0.004 + 0.0033̅) = 2/0.0073̅ = 272.7272̅ (rounding 1/η to 6 decimals gave 272.7025); η 2e6 / 3e6 →
    2/(5e-7 + 3.3̅e-7) = 2/8.3̅e-7 = 2.4e6 (rounding made the mean 0: ZeroDivisionError)."""
    assert combine(Agg.HARMONIC, [250.0, 300.0], [1.0, 1.0]) == pytest.approx(3000.0 / 11.0, rel=1e-12)
    assert combine(Agg.HARMONIC, [2e6, 3e6], [1.0, 1.0]) == pytest.approx(2.4e6, rel=1e-12)
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    out = aggregate_to_solve_steps(pl.DataFrame({"name": ["u", "u"], "t": ["h1", "h2"], "value": [2e6, 3e6]}),
                                   Agg.HARMONIC, ["name"], base, keys=("unit", "efficiency"))
    assert out.get_column("value")[0] == pytest.approx(2.4e6, rel=1e-12)


def test_harmonic_of_equal_non_positive_values_raises() -> None:
    """(N9) An all-0 HARMONIC step fails like 0 / 0.4 does (the positivity check runs before the equal-values
    shortcut)."""
    with pytest.raises(ValueError, match="positive"):
        combine(Agg.HARMONIC, [0.0, 0.0], [1.0, 1.0])
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    with pytest.raises(FlexToolConfigError, match="unit.efficiency: a harmonic mean needs positive values"):
        aggregate_to_solve_steps(pl.DataFrame({"name": ["u", "u"], "t": ["h1", "h2"], "value": [0.0, 0.0]}),
                                 Agg.HARMONIC, ["name"], base, keys=("unit", "efficiency"))


def test_mean_is_rounded_only_for_the_profile() -> None:
    """(S4) self_discharge_loss 1.2e-7 / 1.4e-7 per hour → 1.3e-7 (rounding to 6 decimals made it 0); the profile
    keeps the rounding of its earlier CSV path: 0.1234561 / 0.1234564 → round(0.12345625, 6) = 0.123456."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    loss = aggregate_to_solve_steps(pl.DataFrame({"name": ["n", "n"], "t": ["h1", "h2"], "value": [1.2e-7, 1.4e-7]}),
                                    Agg.MEAN, ["name"], base, keys=("node", "self_discharge_loss"))
    assert loss.get_column("value")[0] == pytest.approx(1.3e-7, rel=1e-12)
    prof = aggregate_to_solve_steps(pl.DataFrame({"name": ["f", "f"], "t": ["h1", "h2"],
                                                  "value": [0.1234561, 0.1234564]}),
                                    Agg.MEAN, ["name"], base, keys=("profile", "profile"))
    assert prof.get_column("value")[0] == 0.123456


def test_sum_of_equal_values_is_the_constant_times_n() -> None:
    """(WP-B S2) SUM is the correctly rounded sum (math.fsum), so n equal hours give exactly n·v: the value a constant
    gets (v times the number of base steps), on every Python version."""
    for v in (0.1, 0.7, -589.3, 1.0 / 3.0, 123.456789):
        for n in (3, 6, 24):
            assert combine(Agg.SUM, [v] * n, [1.0] * n) == v * n
            base = _base([(f"h{i}", "h0", 1.0) for i in range(n)])
            out = aggregate_to_solve_steps(pl.DataFrame({"name": ["n"] * n, "t": [f"h{i}" for i in range(n)],
                                                         "value": [v] * n}),
                                           Agg.SUM, ["name"], base, keys=("node", "inflow"))
            assert out.get_column("value")[0] == v * n


def _reference(frame: pl.DataFrame, kinds: dict, keys: dict, base: pl.DataFrame) -> pl.DataFrame:
    """The per-step Python reference: combine over each (name, param, step) in timeline order."""
    pos = {t: (s, d, i) for t, s, d, i in base.select("time", "new_step", "duration", "pos").iter_rows()}
    groups: dict = {}
    for row, (name, param, t, v) in enumerate(frame.select("name", "param", "t", "value").iter_rows()):
        if v is None or t not in pos:
            continue
        s, d, i = pos[t]
        g = groups.setdefault((name, param, s), {"row": row, "items": []})
        g["items"].append((i, v, d))
    out = []
    for (name, param, s), g in sorted(groups.items(), key=lambda kv: kv[1]["row"]):
        items = sorted(g["items"])
        value = combine(kinds[param], [v for _i, v, _d in items], [d for _i, _v, d in items],
                        round6=keys[param] in {("profile", "profile")})
        out.append((name, param, s, value))
    return pl.DataFrame(out, schema={"name": pl.Utf8, "param": pl.Utf8, "t": pl.Utf8, "value": pl.Float64},
                        orient="row")


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_kernel_equals_combine(seed) -> None:
    """(M3) The vectorised aggregation gives the bits of :func:`combine` applied per step (``.equals``): random
    frames of all four kinds and the profile's rounding, steps of 1-5 base steps with mixed durations, partial
    steps, equal-valued steps, rows in random order."""
    import random

    rng = random.Random(seed)
    rows, t = [], 0
    while t < 120:
        n = rng.randint(1, 5)
        rows.extend((f"h{t + j}", f"h{t}", float(rng.choice([1, 1, 2, 3]))) for j in range(n))
        t += n
    base = _base(rows)
    kinds = {"availability": Agg.MEAN, "efficiency": Agg.HARMONIC, "inflow": Agg.SUM,
             "storage_state_reference_value": Agg.LAST, "profile": Agg.MEAN}
    keys = {"availability": ("unit", "availability"), "efficiency": ("unit", "efficiency"),
            "inflow": ("node", "inflow"), "storage_state_reference_value": ("node", "storage_state_reference_value"),
            "profile": ("profile", "profile")}
    data = []
    for name in ("a", "b", "c"):
        for param in kinds:
            for tt, _s, _d in rows:
                if rng.random() < 0.15:
                    continue                                   # partial steps
                if param == "efficiency":
                    v = rng.choice([0.4, 0.4, rng.uniform(0.2, 0.6), rng.uniform(100, 3000)])
                else:
                    v = rng.choice([1.0, 1.0, rng.uniform(-500, 500), rng.uniform(0, 1) * 1e-6, 0.1, 0.2])
                data.append((name, param, tt, v))
    rng.shuffle(data)
    frame = pl.DataFrame(data, schema={"name": pl.Utf8, "param": pl.Utf8, "t": pl.Utf8, "value": pl.Float64},
                         orient="row")
    got = aggregate_to_solve_steps(frame, "param", ["name", "param"], base, keys=keys, kind_values=kinds,
                                   report_changes=False)
    assert got.equals(_reference(frame, kinds, keys, base))


def test_roll_aggregates_only_its_steps() -> None:
    """(M3a) ``aggregate_if_active(steps=...)`` aggregates only the base steps of the roll's solve steps: steps t1
    (t1, t2) and t3 (t3, t4), the roll holds t3 → one row (30 + 50)/2 = 40."""
    from flextool.engine.input._solve_step_aggregation import aggregate_if_active

    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0), ("t3", "t3", 1.0), ("t4", "t3", 1.0)])
    frame = pl.DataFrame({"name": ["g"] * 4, "t": ["t1", "t2", "t3", "t4"], "value": [10.0, 20.0, 30.0, 50.0]})
    tok = _armed(base)
    try:
        out = aggregate_if_active(frame, ("group", "co2_price"), steps=pl.Series(["t3"]))
    finally:
        deactivate_solve_step_aggregation(tok)
    assert out.rows() == [("g", "t3", 40.0)]


def test_aggregation_runs_before_the_enum_cast() -> None:
    """(S7) Under a global t vocabulary that holds only the solve-step label t1, a time map t1 = 10, t2 = 30 (one
    2 h step t1) gives (10 + 30)/2 = 20; aggregating after the axis cast would drop t2 (not in the vocabulary)
    and give 10."""
    from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums

    t_enum, d_enum = pl.Enum(["t1"]), pl.Enum(["p1"])
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0)])
    dt = pl.DataFrame({"d": pl.Series(["p1"], dtype=d_enum), "t": pl.Series(["t1"], dtype=t_enum)})
    enum_tok = set_global_axis_enums({"d": d_enum, "t": t_enum})
    tok = _armed(base)
    try:
        out = _collect(broadcast_to_period_time(
            _resolved(pl.DataFrame({"name": ["g", "g"], "t": ["t1", "t2"], "value": [10.0, 30.0]}),
                      ("group", "co2_price")), "g", dt))
    finally:
        deactivate_solve_step_aggregation(tok)
        reset_global_axis_enums(enum_tok)
    assert out.rows() == [("g", "t1", 20.0)]


def test_short_last_block_is_not_partial(caplog) -> None:
    """(S8) 5 base steps in 2 h steps: h1 (h1, h2), h3 (h3, h4), h5 (h5 alone, n = 1).  Availability 0.2 / 0.4 /
    0.6 / 0.8 / 0.9 → 0.3, 0.7 and 0.9; the one-step last block has every value it can have: no partial warning."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0), ("h3", "h3", 1.0), ("h4", "h3", 1.0), ("h5", "h5", 1.0)])
    frame = pl.DataFrame({"name": ["u"] * 5, "t": ["h1", "h2", "h3", "h4", "h5"],
                          "value": [0.2, 0.4, 0.6, 0.8, 0.9]})
    with caplog.at_level(logging.WARNING):
        out = aggregate_to_solve_steps(frame, Agg.MEAN, ["name"], base, keys=("unit", "availability"), notes=set())
    assert out.rows() == [("u", "h1", (0.2 + 0.4) / 2), ("u", "h3", (0.6 + 0.8) / 2), ("u", "h5", 0.9)]
    assert not [r for r in caplog.records if "only some of its time steps" in r.message]


def test_timeline_gap_does_not_merge_blocks() -> None:
    """(S8) Two timeset blocks with a gap (h1-h2, then h7-h8; h3-h6 are not in the solve): the new steps h1 and h7
    each take their own two hours, inflow −10 −20 = −30 and −70 −80 = −150; the hours of the gap (authored, but
    in no new step) are dropped, never merged across the gap."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0), ("h7", "h7", 1.0), ("h8", "h7", 1.0)])
    frame = pl.DataFrame({"name": ["n"] * 6, "t": ["h1", "h2", "h3", "h6", "h7", "h8"],
                          "value": [-10.0, -20.0, -30.0, -60.0, -70.0, -80.0]})
    out = aggregate_to_solve_steps(frame, Agg.SUM, ["name"], base, keys=("node", "inflow"))
    assert out.rows() == [("n", "h1", -30.0), ("n", "h7", -150.0)]


def test_min_load_warning_names_only_min_load_efficiency_units(caplog) -> None:
    """(S5) A varying min_load is approximate only on min_load_efficiency units: with ``approx_entities`` {u}, a
    varying min_load of u warns (naming u), one of v (not in the set) alone is silent."""
    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    key = ("unit", "min_load")
    approx = {key: frozenset({"u"})}
    with caplog.at_level(logging.WARNING):
        aggregate_to_solve_steps(pl.DataFrame({"name": ["v", "v"], "t": ["h1", "h2"], "value": [0.3, 0.5]}),
                                 Agg.MEAN, ["name"], base, keys=key, notes=set(), approx_entities=approx)
        assert not [r for r in caplog.records if "min_load varies" in r.message]
        aggregate_to_solve_steps(pl.DataFrame({"name": ["u", "u", "v", "v"], "t": ["h1", "h2", "h1", "h2"],
                                               "value": [0.3, 0.5, 0.3, 0.5]}),
                                 Agg.MEAN, ["name"], base, keys=key, notes=set(), approx_entities=approx)
    msgs = [r.message for r in caplog.records if "min_load varies" in r.message]
    assert len(msgs) == 1 and "min_load_efficiency: u)" in msgs[0]


def test_one_dedupe_set_serves_both_paths(caplog) -> None:
    """(S6) The CSV frames and the DB-direct broadcast share the run's dedupe set: a partial availability step
    warns once, not once per path."""
    base = _base([("t1", "t1", 1.0), ("t2", "t1", 1.0)])
    notes: set = set()
    with caplog.at_level(logging.WARNING):
        _aggregate_csv_series("pt_process.csv", pl.DataFrame(
            {"process": ["u"], "processParam": ["availability"], "time": ["t1"], "pt_process": ["0.5"]}), base, notes)
        tok = activate_solve_step_aggregation(base, notes)
        try:
            broadcast_to_period_time(
                _resolved(pl.DataFrame({"name": ["u"], "t": ["t1"], "value": [0.5]}), ("unit", "availability")),
                "p", pl.DataFrame({"d": ["p1"], "t": ["t1"]}))
        finally:
            deactivate_solve_step_aggregation(tok)
    assert len([r for r in caplog.records if "only some of its time steps" in r.message]) == 1


def test_timeline_config_keeps_one_dedupe_set() -> None:
    """(S6) TimelineConfig carries the run's dedupe set (the orchestration replaces it by the reader's memo)."""
    from collections import defaultdict

    from flextool.engine.input._timeline import TimelineConfig

    cfg = TimelineConfig(defaultdict(list), [], defaultdict(str), defaultdict(list), {})
    assert cfg.solve_step_notes == set()


def test_notice_names_a_step_whose_first_hour_is_missing(capsys) -> None:
    """(WP-B N5) Earlier releases took the value of the step's first base step (its default where that hour has no
    value): availability authored only at h2 = 0.6 of step h1 (h1, h2) keeps 0.6, but differs from what earlier
    releases used at h1, so the notice names it."""
    from flextool.engine import _behaviour_changes as bc

    base = _base([("h1", "h1", 1.0), ("h2", "h1", 1.0)])
    try:
        bc.start_run({"solve_step_series_combined": "83"})
        aggregate_to_solve_steps(pl.DataFrame({"name": ["u"], "t": ["h2"], "value": [0.6]}), Agg.MEAN, ["name"],
                                 base, keys=("unit", "availability"), notes=set())
        assert "Affected parameters: unit.availability of u." in capsys.readouterr().out
    finally:
        bc.start_run(None)
