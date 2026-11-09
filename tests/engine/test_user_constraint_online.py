"""User constraints with a units-online term (``constraint_online_coeff``, database version 79).

The term is ``coefficient · v_online[p,d,t]``: the number of units online (of unitsize) in each step, integer with
binary startup, continuous with linear.  An entity with the coefficient but without online variables is a
configuration error.

Hand-built model (on ``tests/fixtures/tests.json``, alternative ``init`` only: one period p2020, timeset ``2day``
cut to two 1-h steps; never a checked-in ``.sqlite``): node ``elec`` with a demand of 100 MW, commodity node
``fuel`` at 20 €/MWh, units ``g1``, ``g2``, ``g3`` fuel -> elec, existing 100 MW each, ``min_load_efficiency`` with
efficiency 0.5, efficiency_at_min_load 0.4 and min_load 0.5, no startup cost.

Hand values (design c2 §5 and its critique, derived independently of the code): per unit online the fuel line is
section 0.5 · (1/0.4 − 1/0.5) / (1 − 0.5) = 0.5 MW per MW online and slope 1/0.5 − 0.5 = 1.5, so for 100 MW of
output and N units of 100 MW online the fuel is 1.5 · 100 + 0.5 · 100 · N = 150 + 50 · N MW (check: 100 MW from one
unit → 200 MW of fuel, η 0.5; 50 MW → 125, η 0.4).  The cost is fuel only, so the objective ratio to the
unconstrained model is free of annualisation.  The units are direct (one input, one output): their v_flow is the
output (100 MW in total per step) and the fuel enters the cost only, so the fuel is read as 200 MW times the
objective ratio:

    case                    startup  Σ online ≥  N    fuel   ratio
    control (no constraint) binary   —          1    200    1
    A                       binary   2          2    250    1.25
    B                       linear   1.5        1.5  225    1.125
    C                       binary   1.5        2    250    1.25
    mixed g1 linear, g2/g3  mixed    1.5        1.5  225    1.125   (g1 0.5 + one whole unit)
    binary

A′: one entity ``g`` with existing 3 units of virtual_unitsize 100 (existing is the unit count with a
virtual_unitsize) — the same numbers as A (count semantics).  D: A with a 2-h ``new_stepduration`` group on
``elec``: the units are coarse (``fuel`` is a commodity node, so their every balance node is coarse), the constraint
is one ``process_constraint_greater_than_coarse`` row for the 2-h step, the same objective.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import RunCache, b64, build, make_db, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
GE = "process_constraint_greater_than"
UNITS = ("g1", "g2", "g3")

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _map(data: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "constraint", "data": [[k, v] for k, v in data.items()]}, "map")


def _entities(units=UNITS, *, constraints=("c_online",), coarse: bool = False) -> list:
    out = [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
           ["group__node", ["electricity", "elec"], None],
           ["commodity__node", ["fuelc", "fuel"], None]]
    for u in units:
        out += [["unit", u, None], ["unit__inputNode", [u, "fuel"], None], ["unit__outputNode", [u, "elec"], None]]
    out += [["constraint", c, None] for c in constraints]
    if coarse:
        out += [["group", "two_h", None], ["group__node", ["two_h", "elec"], None]]
    return out


def _values(startup: dict[str, str | None], *, existing: float = 100.0, virtual_unitsize: float | None = None,
            constant: float | None = None, coarse: bool = False) -> list:
    """The hand-built model; ``startup`` maps each unit to its startup_method (None: unset); ``constant`` gives
    ``c_online`` (Σ online ≥ constant, coefficient 1 on each unit), None leaves the constraint out."""
    vals = [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
            ["solve", SOLVE, "solver_mip_gap", _f(0.0)],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-100.0)],
            ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)]]
    for u, method in startup.items():
        vals += [["unit", u, "conversion_method", _s("min_load_efficiency")], ["unit", u, "efficiency", _f(0.5)],
                 ["unit", u, "efficiency_at_min_load", _f(0.4)], ["unit", u, "min_load", _f(0.5)],
                 ["unit", u, "existing", _f(existing)]]
        if virtual_unitsize is not None:
            vals.append(["unit", u, "virtual_unitsize", _f(virtual_unitsize)])
        if method is not None:
            vals.append(["unit", u, "startup_method", _s(method)])
        if constant is not None:
            vals.append(["unit", u, "constraint_online_coeff", _map({"c_online": 1.0})])
    if constant is not None:
        vals += [["constraint", "c_online", "sense", _s("greater_than")],
                 ["constraint", "c_online", "constant", _f(constant)]]
    if coarse:
        vals.append(["group", "two_h", "new_stepduration", _f(2.0)])
    return vals


def _solve(tmp_path_factory, alt: str, values: list, entities: list):
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=["init"], entities=entities)
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None, "solve produced no solution"
    return step


def _per_step(step, var: str, *, source: str | None = None, scale: bool) -> list[float]:
    """Σ over the units of ``var`` per step (t0001, t0002); ``scale`` multiplies by the unitsize (MW)."""
    vals = utf8(step.solution.value(var))
    if source is not None:
        vals = vals.filter(pl.col("source") == source)
    if scale:
        us = utf8(step.flex_data.p_unitsize.frame).rename({"value": "us"})
        vals = vals.join(us, on="p").with_columns(value=pl.col("value") * pl.col("us"))
    out = vals.group_by("t").agg(pl.col("value").sum()).sort("t")
    assert out["t"].to_list() == ["t0001", "t0002"], out
    return out["value"].to_list()


def _online(step) -> list[float]:
    """Σ of the units online (count) per step, over both UC classes."""
    total = [0.0, 0.0]
    for kind in ("linear", "integer"):
        try:
            vals = _per_step(step, f"v_online_{kind}", scale=False)
        except KeyError:
            continue
        total = [a + b for a, b in zip(total, vals)]
    return total


def _control(tmp_path_factory):
    return _solve(tmp_path_factory, "uco_control", _values(dict.fromkeys(UNITS, "binary")), _entities(constraints=()))


@pytest.mark.parametrize("case, startup, constant, n_online, fuel, ratio", [
    ("A", "binary", 2.0, 2.0, 250.0, 1.25),
    ("B", "linear", 1.5, 1.5, 225.0, 1.125),
    ("C", "binary", 1.5, 2.0, 250.0, 1.25),
])
def test_units_online_floor(tmp_path_factory, case, startup, constant, n_online, fuel, ratio) -> None:
    """Σ online ≥ constant: fuel 150 + 50·N MW per step, N the units online (integer with binary startup, so
    1.5 needs 2; continuous with linear)."""
    control = _control(tmp_path_factory)
    assert _per_step(control, "v_flow", source="fuel", scale=True) == pytest.approx([100.0, 100.0], rel=1e-7)
    assert _online(control) == pytest.approx([1.0, 1.0], abs=1e-7)
    step = _solve(tmp_path_factory, f"uco_{case}", _values(dict.fromkeys(UNITS, startup), constant=constant),
                  _entities())
    assert _online(step) == pytest.approx([n_online, n_online], abs=1e-7)
    assert _per_step(step, "v_flow", source="fuel", scale=True) == pytest.approx([100.0, 100.0], rel=1e-7)
    assert float(step.obj) / float(control.obj) == pytest.approx(ratio, rel=1e-7)
    assert 200.0 * float(step.obj) / float(control.obj) == pytest.approx(fuel, rel=1e-7)
    rows = row_terms(build(step.flex_data), GE, "c_online,")
    assert sorted(rows) == [f"{GE}[c_online,p2020,t0001]", f"{GE}[c_online,p2020,t0002]"]
    kind = "integer" if startup == "binary" else "linear"
    assert rows[f"{GE}[c_online,p2020,t0001]"] == pytest.approx(
        {f"v_online_{kind}[{u},p2020,t0001]": 1.0 for u in UNITS})


def test_count_of_virtual_units(tmp_path_factory) -> None:
    """A′: one entity of 3 units of 100 MW (existing 3, virtual_unitsize 100) gives the numbers of A."""
    control = _control(tmp_path_factory)
    step = _solve(tmp_path_factory, "uco_A_virtual",
                  _values({"g": "binary"}, existing=3.0, virtual_unitsize=100.0, constant=2.0),
                  _entities(("g",)))
    assert _online(step) == pytest.approx([2.0, 2.0], abs=1e-7)
    assert _per_step(step, "v_flow", source="fuel", scale=True) == pytest.approx([100.0, 100.0], rel=1e-7)
    assert float(step.obj) / float(control.obj) == pytest.approx(1.25, rel=1e-7)


def test_linear_and_binary_in_one_row(tmp_path_factory) -> None:
    """g1 linear, g2 / g3 binary, Σ online ≥ 1.5: one row with a v_online_linear and two v_online_integer terms;
    the cheapest count is g1 0.5 plus one whole unit (1.5), fuel 225 MW."""
    control = _control(tmp_path_factory)
    step = _solve(tmp_path_factory, "uco_mixed",
                  _values({"g1": "linear", "g2": "binary", "g3": "binary"}, constant=1.5), _entities())
    assert _online(step) == pytest.approx([1.5, 1.5], abs=1e-7)
    assert _per_step(step, "v_flow", source="fuel", scale=True) == pytest.approx([100.0, 100.0], rel=1e-7)
    assert float(step.obj) / float(control.obj) == pytest.approx(1.125, rel=1e-7)
    rows = row_terms(build(step.flex_data), GE, "c_online,")
    assert rows[f"{GE}[c_online,p2020,t0002]"] == pytest.approx(
        {"v_online_linear[g1,p2020,t0002]": 1.0, "v_online_integer[g2,p2020,t0002]": 1.0,
         "v_online_integer[g3,p2020,t0002]": 1.0})


def test_coarse_units_one_row_per_coarse_step(tmp_path_factory) -> None:
    """D: A with a 2-h new_stepduration group on elec.  The units are coarse (fuel is a commodity node), so the
    constraint is one averaged row for the 2-h step; the objective is A's."""
    a = _solve(tmp_path_factory, "uco_A", _values(dict.fromkeys(UNITS, "binary"), constant=2.0), _entities())
    step = _solve(tmp_path_factory, "uco_D", _values(dict.fromkeys(UNITS, "binary"), constant=2.0, coarse=True),
                  _entities(coarse=True))
    pb = build(step.flex_data)
    assert list(row_terms(pb, GE + "_coarse", "c_online,")) == [f"{GE}_coarse[c_online,p2020,t0001]"]
    assert not row_terms(pb, GE, "c_online,")
    assert float(step.obj) == pytest.approx(float(a.obj), rel=1e-7)


def test_coefficient_without_online_variables_is_an_error(tmp_path_factory) -> None:
    """E: A with startup_method unset: a configuration error naming the unit and the constraint."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.input._solve_state import FlexToolConfigError

    values = [v for v in _values(dict.fromkeys(UNITS, None), constant=2.0)
              if v[2] not in ("min_load", "efficiency_at_min_load")]
    values = [["unit", u, "conversion_method", _s("constant_efficiency")] if v[2] == "conversion_method" else v
              for v in values for u in [v[1]]]
    url, scen = make_db(tmp_path_factory, alt="uco_E", values=values, base_chain=["init"], entities=_entities())
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("uco_E_w"))
    message = str(err.value)
    for name in ("unit 'g1' in constraint 'c_online'", "unit 'g2'", "unit 'g3'", "startup_method"):
        assert name in message, message


def test_online_term_makes_a_capacity_constraint_per_step(tmp_path_factory) -> None:
    """F: c_cap with only an invested-capacity term (g3 investable) is one row per period; an online coefficient
    on g1 makes it a row per step that holds both terms."""
    invest = [["solve", SOLVE, "invest_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
              ["unit", "g3", "invest_method", _s("invest_no_limit")], ["unit", "g3", "invest_cost", _f(1000.0)],
              ["unit", "g3", "lifetime", _f(20.0)], ["unit", "g3", "discount_rate", _f(0.05)],
              ["unit", "g3", "constraint_invested_capacity_coeff", _map({"c_cap": 1.0})],
              ["constraint", "c_cap", "sense", _s("greater_than")], ["constraint", "c_cap", "constant", _f(0.0)]]
    base = _values(dict.fromkeys(UNITS, "binary")) + invest
    entities = _entities(constraints=("c_cap",))
    capacity_only = _solve(tmp_path_factory, "uco_F_cap", base, entities)
    pb = build(capacity_only.flex_data)
    assert list(row_terms(pb, GE + "_period", "c_cap,")) == [f"{GE}_period[c_cap,p2020]"]
    assert not row_terms(pb, GE, "c_cap,")
    with_online = _solve(tmp_path_factory, "uco_F_online",
                         base + [["unit", "g1", "constraint_online_coeff", _map({"c_cap": 1.0})]], entities)
    pb = build(with_online.flex_data)
    assert not row_terms(pb, GE + "_period", "c_cap,")
    rows = row_terms(pb, GE, "c_cap,")
    assert sorted(rows) == [f"{GE}[c_cap,p2020,t0001]", f"{GE}[c_cap,p2020,t0002]"]
    for t in ("t0001", "t0002"):
        terms = rows[f"{GE}[c_cap,p2020,{t}]"]
        assert terms[f"v_online_integer[g1,p2020,{t}]"] == pytest.approx(1.0)
        assert [c for c in terms if c.startswith("v_invest_p[g3,")] == ["v_invest_p[g3,p2020]"]
