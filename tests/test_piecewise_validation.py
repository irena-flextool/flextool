"""Checks of ``piecewise_linear`` units in the input derivation (``specs/piecewise_efficiency_final.md`` §5, T3).

Errors E1-E10, E12 (``validate_piecewise_linear_inputs``), warnings W1, W2, W6, W7, the start-up warning W4
(``validate_startup_inputs``) and the classifier's automatic start-ups W3 with its methods (forced fork, linear
start-ups; ``derive_process_method``).  Each test builds a small database from ``schemas/spinedb_schema.json`` (the
session ``schema_db_url``, copied) with one alternative.  The example curves and numbers are the design's / the
oracle's (``specs/piecewise_efficiency_oracle.md`` §0.8: W1 η(0.2) ≈ 0.19, β₁ ≈ 0.78; E8 hull η(0.7) ≈ 0.377).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, Map, TimeSeriesVariableResolution, to_database

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.input_derivation._process_method import derive_process_method
from flextool.input_derivation._validators import validate_piecewise_linear_inputs, validate_startup_inputs

O1 = Map([0.4, 0.7, 1.0], [0.30, 0.40, 0.42])
E8 = Map([0.4, 0.7, 1.0], [0.30, 0.36, 0.42])
ELECTROLYSER = Map([0.2, 0.6, 1.0], [0.70, 0.66, 0.60])
RISING = Map([0.4, 1.0], [0.30, 0.42])
FLAT = Map([0.2, 1.0], [0.66, 0.66])
PERIOD = {"index_name": "period"}


def _unit(name: str, *, inputs=("fuel",), outputs=("elec",), ct: str | None = "piecewise_linear",
          curve=O1) -> tuple[list, list]:
    ents = [("unit", (name,))] + [("unit__inputNode", (name, n)) for n in inputs] + [
        ("unit__outputNode", (name, n)) for n in outputs]
    vals = []
    if ct is not None:
        vals.append(("unit", (name,), "conversion_method", ct))
    if curve is not None:
        vals.append(("unit", (name,), "piecewise_linear_efficiency", curve))
    return ents, vals


NODES = [("node", (n,)) for n in ("fuel", "gas", "elec", "heat")]
SOLVE = [("solve", ("s1",)), ("timeline", ("tl",))]
SOLVE_VALUES = [("solve", ("s1",), "years_represented", Map(["p2030", "p2035"], [1.0, 1.0], **PERIOD)),
                ("timeline", ("tl",), "timestep_duration", Map(["t01", "t02"], [1.0, 1.0], index_name="time"))]


def _db(tmp_path: Path, schema_db_url: str, entities: list, values: list) -> str:
    path = tmp_path / "piecewise.sqlite"
    shutil.copy(schema_db_url.removeprefix("sqlite:///"), path)
    url = f"sqlite:///{path}"
    with DatabaseMapping(url) as db:
        for cls, byname in NODES + SOLVE + entities:
            db.add_entity(entity_class_name=cls, entity_byname=byname)
        for cls, byname, param, value in SOLVE_VALUES + values:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=param,
                                   alternative_name="base", value=val, type=typ)
        db.commit_session("piecewise test")
    return url


def _check(tmp_path, schema_db_url, caplog, units: list[tuple[list, list]], extra_values=(), extra_entities=()):
    """``(error text or None, warning messages)`` of ``validate_piecewise_linear_inputs``."""
    ents = [e for u in units for e in u[0]] + list(extra_entities)
    vals = [v for u in units for v in u[1]] + list(extra_values)
    url = _db(tmp_path, schema_db_url, ents, vals)
    logger = logging.getLogger("test_piecewise_validation")
    caplog.set_level(logging.INFO, logger=logger.name)
    error = None
    with DatabaseMapping(url) as db:
        try:
            validate_piecewise_linear_inputs(db, logger)
        except FlexToolConfigError as err:
            error = str(err)
    return error, [r.getMessage() for r in caplog.records if r.name == logger.name]


# ─── errors ──────────────────────────────────────────────────────────────


def test_valid_curve_passes(tmp_path, schema_db_url, caplog) -> None:
    error, warnings = _check(tmp_path, schema_db_url, caplog, [_unit("G")],
                             extra_values=[("unit", ("G",), "min_load", 0.4)])
    assert error is None and warnings == []


def test_E1_no_curve(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=None)])
    assert ("unit 'G': conversion_method piecewise_linear needs piecewise_linear_efficiency (a map from load share "
            "to efficiency)") in error


def test_E2_constant_and_three_levels(tmp_path, schema_db_url, caplog) -> None:
    deep = Map(["p2030"], [Map(["a"], [Map([1.0], [0.4])])], **PERIOD)
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("A", curve=0.4), _unit("B", curve=deep)])
    assert "unit 'A', parameter 'piecewise_linear_efficiency': the value is a constant" in error
    assert "unit 'B', parameter 'piecewise_linear_efficiency': the value has 3 index levels" in error
    assert ("This parameter takes a curve (map over load shares), or a curve per period (map over periods of maps "
            "over load shares)") in error


@pytest.mark.parametrize("key", [0.0, 1.2, "abc"])
def test_E3_load_key(tmp_path, schema_db_url, caplog, key) -> None:
    curve = Map([str(key), "1.0"], [0.3, 0.42]) if isinstance(key, str) else Map([key, 1.0], [0.3, 0.42])
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=curve)])
    assert (f"unit 'G': load share '{key}' must be a number above 0 and at most 1 (the efficiency at zero output "
            "is not defined; the fuel at zero output follows from the first point)") in error


def test_E3_load_just_above_one_is_one(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=Map([0.4, 1.0 + 1e-10], [0.3, 0.42]))])
    assert error is None


def test_E4_E5_E6(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [
        _unit("A", curve=Map(["0.5", "0.50", "1"], [0.3, 0.3, 0.4])),
        _unit("B", curve=Map([0.4, 0.9], [0.3, 0.4])),
        _unit("C", curve=Map([0.4, 1.0], [0.0, 0.4]))])
    assert "unit 'A': load share 0.5 is given twice" in error
    assert "unit 'B': the curve must end at full load (load share 1); it ends at 0.9" in error
    assert "unit 'C': efficiency at load 0.4 is 0; it must be above 0" in error


def test_E7_fuel_falls(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=Map([0.4, 0.6, 1.0], [0.3, 0.5, 0.55]))])
    assert ("unit 'G': the fuel use falls from load 0.4 to 0.6 (efficiency 0.3 → 0.5): more output for less "
            "fuel") in error
    assert "Nearest accepted curve" not in error


def test_E8_falling_incremental_heat_rate_with_hull(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("U", curve=E8)])
    assert ("unit 'U': the extra fuel per extra output falls from 2.04 (load 0.4–0.7) to 1.46 (load 0.7–1) for the "
            "curve {0.4: 0.3, 0.7: 0.36, 1.0: 0.42}; a linear model cannot use such a curve. Nearest accepted "
            "curve: {0.4: 0.3, 1.0: 0.42} (it raises the efficiency at load 0.7 from 0.36 to 0.377)") in error


def test_E8_suggested_curve_pasted_back_passes(tmp_path, schema_db_url, caplog) -> None:
    """The points of the given and the suggested curve are printed at full precision, so the suggestion pasted
    back into the database passes the convexity check (a 3-digit rounding of 0.3123456 would not be the hull)."""
    import ast
    import re

    curve = Map([0.3, 0.5, 0.8, 1.0], [0.3123456789, 0.33, 0.4087654321, 0.41])
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("U", curve=curve)])
    assert "for the curve {0.3: 0.3123456789, 0.5: 0.33, 0.8: 0.4087654321, 1.0: 0.41}" in error
    suggested = ast.literal_eval(re.search(r"Nearest accepted curve: (\{[^}]*\})", error).group(1))
    assert suggested == {0.3: 0.3123456789, 0.8: 0.4087654321, 1.0: 0.41}
    (tmp_path / "back").mkdir()
    error, _ = _check(tmp_path / "back", schema_db_url, caplog,
                      [_unit("U", curve=Map(list(suggested), list(suggested.values())))])
    assert error is None


def test_empty_curve(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [
        _unit("A", curve=Map([], [], index_type=float)),
        _unit("B", curve=Map(["p2030", "p2035"], [O1, Map([], [], index_type=float)], **PERIOD))])
    assert "unit 'A': the curve has no points (it must end at full load, load share 1)" in error
    assert "unit 'B' (period p2035): the curve has no points (it must end at full load, load share 1)" in error
    assert "p2030" not in error


def test_E3_key_above_one_gives_no_E5(tmp_path, schema_db_url, caplog) -> None:
    """A key just above the tolerance is refused (E3); E5 does not then name the first point as the curve's end."""
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=Map([0.4, 1.000000002], [0.3, 0.42]))])
    assert "unit 'G': load share '1.000000002' must be a number above 0 and at most 1" in error
    assert "the curve must end at full load" not in error


def test_E8_names_the_period(tmp_path, schema_db_url, caplog) -> None:
    curve = Map(["p2030", "p2035"], [O1, E8], **PERIOD)
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("U", curve=curve)])
    assert "unit 'U' (period p2035): the extra fuel per extra output falls" in error
    assert "p2030" not in error


def test_E9_time_varying_efficiency(tmp_path, schema_db_url, caplog) -> None:
    eff = Map(["t01", "t02"], [0.4, 0.4], index_name="time")
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("G")],
                      extra_values=[("unit", ("G",), "efficiency", eff)])
    assert ("unit 'G': a piecewise_linear unit cannot have a time-varying efficiency; its curve is "
            "piecewise_linear_efficiency (efficiency is not used)") in error
    ts = TimeSeriesVariableResolution(["2030-01-01T00:00", "2030-01-01T01:00"], [0.4, 0.4], False, False)
    (tmp_path / "ts").mkdir()
    error, _ = _check(tmp_path / "ts", schema_db_url, caplog, [_unit("G")],
                      extra_values=[("unit", ("G",), "efficiency", ts)])
    assert "cannot have a time-varying efficiency" in error


def test_E10_level_one_is_a_period(tmp_path, schema_db_url, caplog) -> None:
    by_time = Map(["t01"], [O1], index_name="time")
    no_period = Map(["p2099"], [O1])
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("A", curve=by_time), _unit("B", curve=no_period)])
    assert ("unit 'A', parameter 'piecewise_linear_efficiency': level 1 of the value is named 'time'; level 1 of a "
            "2-level curve is the period") in error
    assert ("unit 'B', parameter 'piecewise_linear_efficiency': level 1 of the value is a period, but 'p2099' is "
            "not a period of the database") in error


def test_E10_load_level_named_period(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("A", curve=Map([0.4, 1.0], [0.3, 0.42], **PERIOD))])
    assert "level 1 of the value is named 'period', but that is the load share level" in error


def test_E12_no_input_or_no_output(tmp_path, schema_db_url, caplog) -> None:
    error, _ = _check(tmp_path, schema_db_url, caplog, [_unit("A", inputs=()), _unit("B", outputs=())])
    assert "unit 'A': a piecewise_linear unit needs at least one input and one output" in error
    assert "unit 'B': a piecewise_linear unit needs at least one input and one output" in error


# ─── warnings ────────────────────────────────────────────────────────────


def test_W1_min_load_below_first_point(tmp_path, schema_db_url, caplog) -> None:
    _, warnings = _check(tmp_path, schema_db_url, caplog, [_unit("U")],
                         extra_values=[("unit", ("U",), "min_load", 0.2)])
    assert warnings == [
        "Unit 'U': min_load 0.2 is below the first point of its curve (load 0.4): between them the first segment is "
        "extended (efficiency at min_load ≈ 0.19); without a higher min_load the unit can be online at zero "
        "output, burning 0.78 MW of fuel per MW online"]


def test_W1_compares_each_period_with_its_min_load(tmp_path, schema_db_url, caplog) -> None:
    """min_load per period {p2030: 0.2, p2035: 0.5}: a curve per period meets its own period's min_load (p2030:
    0.2 below the first point 0.4 warns; p2035: 0.5 does not), and one curve for every period meets each
    period's min_load (only p2030 warns, naming it).  The smallest min_load over all periods (0.2) would warn for
    both periods."""
    min_load = Map(["p2030", "p2035"], [0.2, 0.5], **PERIOD)
    _, warnings = _check(tmp_path, schema_db_url, caplog,
                         [_unit("U", curve=Map(["p2030", "p2035"], [O1, O1], **PERIOD)), _unit("C")],
                         extra_values=[("unit", ("U",), "min_load", min_load), ("unit", ("C",), "min_load", min_load)])
    tail = ("(load 0.4): between them the first segment is extended (efficiency at min_load ≈ 0.19); without a "
            "higher min_load the unit can be online at zero output, burning 0.78 MW of fuel per MW online")
    assert sorted(warnings) == [f"Unit 'C': min_load 0.2 is below the first point of its curve in period p2030 {tail}",
                                f"Unit 'U': min_load 0.2 is below the first point of its curve in period p2030 {tail}"]


def test_W1_not_for_a_falling_curve_and_per_period(tmp_path, schema_db_url, caplog) -> None:
    curve = Map(["p2030", "p2035"], [ELECTROLYSER, O1], **PERIOD)
    _, warnings = _check(tmp_path, schema_db_url, caplog, [_unit("U", curve=curve)])
    assert len(warnings) == 1
    assert warnings[0].startswith("Unit 'U': min_load 0 is below the first point of its curve in period p2035 "
                                  "(load 0.4)")


def test_W2_efficiency_not_used(tmp_path, schema_db_url, caplog) -> None:
    _, warnings = _check(tmp_path, schema_db_url, caplog, [_unit("G", curve=FLAT)],
                         extra_values=[("unit", ("G",), "efficiency", 0.4),
                                       ("unit", ("G",), "efficiency_at_min_load", 0.3)])
    assert warnings == ["Unit 'G': efficiency and efficiency_at_min_load are not used by a piecewise_linear unit "
                        "(its curve gives the efficiency)"]


def test_W6_curve_on_another_method(tmp_path, schema_db_url, caplog) -> None:
    _, warnings = _check(tmp_path, schema_db_url, caplog, [_unit("G", ct=None), _unit("M", ct="min_load_efficiency")])
    assert warnings == ["Unit 'G': piecewise_linear_efficiency is not used (conversion_method is constant_efficiency)",
                        "Unit 'M': piecewise_linear_efficiency is not used (conversion_method is min_load_efficiency)"]


def test_W7_input_items_of_multi_line_units(tmp_path, schema_db_url, caplog) -> None:
    """A two-input unit: the upward reserve, the lower_limit profile, the flowGroup minimum, the greater_than
    constraint and the input ramp limit all stay on its input.  A one-input, one-output unit: only the reserve
    (its other input items act on its output)."""
    ents = [("reserve", ("r",)), ("profile", ("pr",)), ("flowGroup", ("fg",)), ("constraint", ("c",)),
            ("reserve__upDown__unit__node", ("r", "up", "E", "fuel")),
            ("reserve__upDown__unit__node", ("r", "down", "E", "gas")),
            ("unit__node__profile", ("E", "fuel", "pr")), ("flowGroup__unit__node", ("fg", "E", "gas")),
            ("reserve__upDown__unit__node", ("r", "up", "S", "fuel")), ("unit__node__profile", ("S", "fuel", "pr"))]
    vals = [("unit__node__profile", ("E", "fuel", "pr"), "profile_method", "lower_limit"),
            ("unit__node__profile", ("S", "fuel", "pr"), "profile_method", "fixed"),
            ("flowGroup", ("fg",), "min_instant_flow", 10.0), ("constraint", ("c",), "sense", "greater_than"),
            ("unit__inputNode", ("E", "gas"), "constraint_flow_coeff", Map(["c"], [1.0], index_name="constraint")),
            ("unit__inputNode", ("E", "gas"), "ramp_method", "ramp_limit")]
    _, warnings = _check(tmp_path, schema_db_url, caplog,
                         [_unit("E", inputs=("fuel", "gas"), curve=ELECTROLYSER), _unit("S", curve=ELECTROLYSER),
                          _unit("R", curve=RISING)], extra_values=vals, extra_entities=ents)
    tail = "can be met by burning input above the part-load curve (piecewise_linear rows are 'at least'); " \
           "check the post-solve warning"
    assert sorted(warnings) == sorted(
        [f"Unit 'E': its {item} on input '{n}' {tail}"
         for item, n in [("upward reserve 'r'", "fuel"), ("profile 'pr' (lower_limit)", "fuel"),
                         ("flowGroup 'fg' minimum", "gas"), ("constraint 'c'", "gas"), ("ramp limit", "gas")]]
        + [f"Unit 'S': its upward reserve 'r' on input 'fuel' {tail}",
           "Unit 'R': min_load 0 is below the first point of its curve (load 0.4): between them the first segment "
           "is extended (efficiency at min_load ≈ 0); without a higher min_load the unit can be online at zero "
           "output, burning 0.63 MW of fuel per MW online"])  # β = 0.4·(1/0.3 − 1/0.42)/0.6 = 0.635


# ─── classifier: methods, forced fork, automatic start-ups (W3), W4 ──────


class _Provider:
    def __init__(self) -> None:
        self.frames: dict[str, pl.DataFrame] = {}

    def has(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str) -> pl.DataFrame:
        return self.frames[key]

    def put(self, key: str, frame: pl.DataFrame) -> None:
        self.frames[key] = frame


def _classify(url: str, caplog) -> tuple[dict[str, str], list[str], _Provider]:
    logger = logging.getLogger("test_piecewise_classifier")
    caplog.set_level(logging.INFO, logger=logger.name)
    provider = _Provider()
    provider.put("input/process__startup_method",
                 pl.DataFrame({"process": ["B"], "startup_method": ["binary"]}))
    with DatabaseMapping(url) as db:
        derive_process_method(db, provider, logger, ct_method_overrides={})
    pm = provider.get("input/process_method")
    return (dict(zip(pm["process"].to_list(), pm["method"].to_list())),
            [r.getMessage() for r in caplog.records if r.name == logger.name], provider)


def test_classifier_methods_and_automatic_startups(tmp_path, schema_db_url, caplog) -> None:
    """Thermal O1 (two rising lines): fork_yes and linear start-ups -> method_1way_nvar_LP.  Rising two-point: one
    line, linear start-ups -> method_1way_1var_LP.  Flat: constant efficiency, no start-ups.  Electrolyser (falling,
    origin line): fork_yes, no start-ups.  A binary unit keeps binary start-ups.  O4 by period: W3 names p2030."""
    o4 = Map(["p2030", "p2035"], [Map([0.5, 1.0], [0.40, 0.50]), Map([0.5, 1.0], [0.55, 0.50])], **PERIOD)
    units = [_unit("G"), _unit("R", curve=RISING), _unit("F", curve=FLAT), _unit("E", curve=ELECTROLYSER),
             _unit("B"), _unit("P", curve=o4)]
    url = _db(tmp_path, schema_db_url, [e for u in units for e in u[0]],
              [v for u in units for v in u[1]] + [("unit", ("B",), "startup_method", "binary")])
    methods, infos, provider = _classify(url, caplog)
    assert {u: methods[u] for u in "GRFEBP"} == {
        "G": "method_1way_nvar_LP", "R": "method_1way_1var_LP", "F": "method_1way_1var_off",
        "E": "method_1way_nvar_off", "B": "method_1way_nvar_MIP", "P": "method_1way_nvar_LP"}
    assert ("Unit 'G': startup_method set to linear: its piecewise_linear_efficiency curve burns fuel at zero output "
            "(efficiency rises from load 0.4 to 0.7), which needs online variables") in infos
    assert ("Unit 'P': startup_method set to linear: its piecewise_linear_efficiency curve burns fuel at zero output "
            "(efficiency rises from load 0.5 to 1 in period p2030), which needs online variables") in infos
    assert not any(m.startswith(("Unit 'F'", "Unit 'E'", "Unit 'B'")) for m in infos)
    sm = provider.get("input/process__startup_method")
    assert dict(zip(sm["process"].to_list(), sm["startup_method"].to_list())) == {
        "B": "binary", "G": "linear", "R": "linear", "P": "linear"}


def test_W4_only_startup_only_inputs(tmp_path, schema_db_url, caplog) -> None:
    """A rising piecewise unit whose only input is startup_only: online by the automatic start-ups (no E2 of the
    start-up check), W4 from the start-up check, E12 from the piecewise check (startup_only inputs do not count)."""
    ents, vals = _unit("U", curve=RISING)
    vals += [("unit__inputNode", ("U", "fuel"), "startup_input_method", "startup_only"),
             ("unit__inputNode", ("U", "fuel"), "startup_energy", 1.0)]
    url = _db(tmp_path, schema_db_url, ents, vals)
    _methods, _infos, provider = _classify(url, caplog)
    logger = logging.getLogger("test_piecewise_w4")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with DatabaseMapping(url) as db:
        validate_startup_inputs(db, provider, logger)
        with pytest.raises(FlexToolConfigError, match="unit 'U': a piecewise_linear unit needs at least one input"):
            validate_piecewise_linear_inputs(db, logger)
    assert ("unit 'U': piecewise_linear with only startup_only inputs: its part-load curve has no input to draw "
            "from.") in [r.getMessage() for r in caplog.records if r.name == logger.name]
