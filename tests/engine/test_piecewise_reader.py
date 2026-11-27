"""The engine reader of ``unit.piecewise_linear_efficiency`` and the engine classifier (T3 / T7 reader side).

* :func:`_param_shapes.piecewise_curve_frame` reads a curve (1 level: load share -> efficiency, float or string
  keys, any index name but period / time) and a curve per period (2 levels) into ``(p, d, load, eff)``, the load
  keys bit-equal to the authored floats, with the axis enums on (a key is never cast or trimmed against the
  period vocabulary); a constant or a deeper value is the shape resolver's error.
* A constant curve and the equivalent period map classify the same and give the same lines in every period.
* The engine classifier (``params/_derived._process_method._classify_process_method``) and the input derivation's
  (``derive_process_method``) give every unit the same flags and the same method.
* The ``seg`` axis vocabulary is ``"0"`` .. the most points of a curve.

The database is built from ``schemas/spinedb_schema.json`` (the session ``schema_db_url``, copied).
"""
from __future__ import annotations

import logging

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, Map, to_database

from flextool.engine.input._piecewise_curve import curve_lines, unit_flags, unit_flags_frame
from flextool.engine.input._solve_state import FlexToolConfigError

O1 = [(0.4, 0.30), (0.7, 0.40), (1.0, 0.42)]
ELECTROLYSER = [(0.2, 0.70), (0.6, 0.66), (1.0, 0.60)]
PERIODS = ["p2030", "p2035"]


def _curve(points, **kw) -> Map:
    return Map([k for k, _ in points], [e for _, e in points], **kw)


CURVES = {
    "A": _curve(O1),                                                   # constant, float keys, unnamed
    "B": Map(["0.4", "0.7", "1"], [0.30, 0.40, 0.42]),                 # constant, string keys, unnamed
    "C": Map(PERIODS, [_curve(O1), _curve(O1)], index_name="period"),  # the equivalent period map of A
    "D": _curve(ELECTROLYSER, index_name="load_share"),
    "E": Map(PERIODS, [_curve([(0.5, 0.40), (1.0, 0.50)]), _curve([(0.5, 0.55), (1.0, 0.50)])]),  # O4, unnamed
    "F": _curve([(0.4, 0.30), (1.0, 0.42)]),                           # one rising line
}


def _db(tmp_path_factory, curves: dict) -> str:
    from flextool.update_flextool.initialize_database import initialize_database
    from flextool._resources import package_data_path

    path = tmp_path_factory.mktemp("piecewise_reader") / "db.sqlite"
    initialize_database(str(package_data_path("schemas/spinedb_schema.json")), str(path))
    url = f"sqlite:///{path}"
    with DatabaseMapping(url) as db:
        db.add_scenario(name="s")
        db.add_scenario_alternative(scenario_name="s", alternative_name="base", rank=1)
        ents = [("node", ("fuel",)), ("node", ("elec",)), ("solve", ("s1",))]
        for u in curves:
            ents += [("unit", (u,)), ("unit__inputNode", (u, "fuel")), ("unit__outputNode", (u, "elec"))]
        for cls, byname in ents:
            db.add_entity(entity_class_name=cls, entity_byname=byname)
            if cls in ("node", "unit"):
                db.add_entity_alternative(entity_class_name=cls, entity_byname=byname, alternative_name="base",
                                          active=True)
        vals = [("solve", ("s1",), "years_represented", Map(PERIODS, [1.0, 1.0], index_name="period"))]
        for u, curve in curves.items():
            vals += [("unit", (u,), "conversion_method", "piecewise_linear"),
                     ("unit", (u,), "piecewise_linear_efficiency", curve)]
        for cls, byname, param, value in vals:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=param,
                                   alternative_name="base", value=val, type=typ)
        db.commit_session("piecewise reader test")
    return url


def _reader(url: str):
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.spinedb_backend._axis_enums import build_scenario_axis_enums

    enums, contract = build_scenario_axis_enums(url, "s")
    return SpineDbReader(url, "s", axis_enums=enums, contract=contract), enums


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    url = _db(tmp_path_factory, CURVES)
    reader, enums = _reader(url)
    return url, reader, enums


def test_reader_shapes(built) -> None:
    from flextool.engine.input._param_shapes import piecewise_curve_frame

    _url, reader, _enums = built
    frame = piecewise_curve_frame(reader)
    assert frame.schema == pl.Schema({"p": pl.Utf8, "d": pl.Utf8, "load": pl.Float64, "eff": pl.Float64})
    got = {(p, d): [] for p, d in frame.select("p", "d").unique().rows()}
    for p, d, load, eff in frame.rows():
        got[(p, d)].append((load, eff))
    expected = {("A", None): O1, ("B", None): O1, ("C", "p2030"): O1, ("C", "p2035"): O1,
                ("D", None): ELECTROLYSER, ("E", "p2030"): [(0.5, 0.40), (1.0, 0.50)],
                ("E", "p2035"): [(0.5, 0.55), (1.0, 0.50)], ("F", None): [(0.4, 0.30), (1.0, 0.42)]}
    assert got == expected  # exact floats: 0.4 / 0.7 read bit-equal from float and string keys


def test_constant_curve_equals_its_period_map(built) -> None:
    from flextool.engine.input._param_shapes import piecewise_curve_frame

    frame = piecewise_curve_frame(built[1])
    flags = unit_flags(frame)
    assert flags["A"] == flags["B"] == flags["C"]
    for online in ({"A", "C"}, set()):
        lines = curve_lines(frame.filter(pl.col("p").is_in(["A", "C"])), online)
        by_unit = {p: lines.filter(pl.col("p") == p).drop("p", "d") for p in ("A", "C")}
        c = lines.filter(pl.col("p") == "C")
        assert set(c["d"].to_list()) == set(PERIODS)
        for d in PERIODS:
            assert c.filter(pl.col("d") == d).drop("p", "d").equals(by_unit["A"])


def test_both_classifiers_agree(built) -> None:
    """Flags and methods: engine (SpineDbReader) = input derivation (DatabaseMapping)."""
    from flextool.engine.preprocess.params._derived._process_method import _classify_process_method, _piecewise_flags
    from flextool.input_derivation._piecewise import parse_piecewise_curves
    from flextool.input_derivation._process_method import derive_process_method

    url, reader, _enums = built
    with DatabaseMapping(url) as db:
        parsed = parse_piecewise_curves(db)
        frames: dict[str, pl.DataFrame] = {}

        class _Provider:
            def has(self, key):
                return key in frames

            def get(self, key):
                return frames[key]

            def put(self, key, frame):
                frames[key] = frame

        derive_process_method(db, _Provider(), logging.getLogger("test_piecewise_reader"), ct_method_overrides={})
    assert parsed.errors == {}
    assert _piecewise_flags(reader).equals(unit_flags_frame(parsed.frame))
    engine = _classify_process_method(reader)
    engine_methods = dict(zip(engine["p"].cast(pl.Utf8).to_list(), engine["method"].to_list()))
    pm = frames["input/process_method"]
    assert engine_methods == dict(zip(pm["process"].to_list(), pm["method"].to_list()))
    assert {u: engine_methods[u] for u in CURVES} == {
        "A": "method_1way_nvar_LP", "B": "method_1way_nvar_LP", "C": "method_1way_nvar_LP",
        "D": "method_1way_nvar_off", "E": "method_1way_nvar_LP", "F": "method_1way_1var_LP"}


def test_seg_axis_vocabulary(built) -> None:
    assert built[2]["seg"].categories.to_list() == ["0", "1", "2", "3"]


def test_reader_refuses_a_constant(tmp_path_factory) -> None:
    from flextool.engine.input._param_shapes import piecewise_curve_frame

    reader, _ = _reader(_db(tmp_path_factory, {"A": 0.4}))
    with pytest.raises(FlexToolConfigError, match="unit 'A', parameter 'piecewise_linear_efficiency': the value is "
                                                  "a constant. This parameter takes a curve"):
        piecewise_curve_frame(reader)
