"""Investment and retirement are offered without ``invest_cost`` / ``salvage_value`` (#218).

An ``invest_method`` that allows investing (retiring) gives the entity its ``v_invest`` (``v_divest``) in every
investment period of the solve, also where ``invest_cost`` (``salvage_value``) is missing or 0.  Before, such a
pair needed a non-zero annuity or a user-constraint capacity coefficient, so an ``invest_method`` without a cost
silently offered nothing.  The capacity has no annuity; its ``fixed_cost`` still counts.

Vehicle: ``tests/fixtures/tests.json``, chain ``init, west`` (one period p2020, 48 one-hour steps of the ``2day``
timeset, ``years_represented`` 1, ``inflation_rate`` 0) plus one override alternative: the west demand is a
constant 100 MW, the solve invests in p2020.  Expected values are hand-derived (``penalty_up`` 900 CUR/MWh ≫
60 000 / 8760 CUR/MWh, so investing 100 MW is the unique optimum of (a)):

* (a) unit ``g`` (output west, ``invest_no_limit``, ``fixed_cost`` 60 CUR/kW/a, no ``invest_cost``, no lifetime):
  ``v_invest`` exists, 100 MW invested, investment cost 0, fixed cost of invested capacity
  100 MW · 1000 kW/MW · 60 = 6.0 M CUR.
* (b) the same with ``lifetime_method reinvest_choice``: the validator asks for a lifetime naming ``unit 'g'``.
* (c) unit ``r`` (``existing`` 100, ``retire_no_limit``, ``fixed_cost`` 50, no salvage) next to a free 100 MW
  source ``s``: ``r`` is not needed, so it retires 100 MW; the fixed-cost removal is −100 · 1000 · 50 = −5.0 M CUR
  (the pre-existing fixed cost +5.0 M CUR, net 0).
* (d) the method lists: an ``invest_period_total`` entity is not retirable, a ``retire_period_total`` entity
  cannot invest.
* (e) an entity with a user-constraint capacity coefficient is still offered (the coefficient exception is gone,
  the cross product covers it).  No investment value is asserted: with neither ``invest_cost`` nor ``fixed_cost``
  the amount is arbitrary.
"""
from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import b64, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
OUTPUT_CONFIG = str(REPO / "flextool" / "schemas" / "default_plots.yaml")
CHAIN = ["init", "west"]
SOLVE = "y2020_2day_dispatch"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


BASE = [
    ["node", "west", "inflow", _f(-100.0)],
    ["solve", SOLVE, "invest_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
]


def _unit(name: str) -> list:
    return [["unit", name, None], ["unit__outputNode", [name, "west"], None]]


def _write_outputs(steps: dict, work: Path, monkeypatch) -> Path:
    """The csv outputs of a run (as ``tests/model/test_other_operational_cost_shapes._write_outputs``)."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    monkeypatch.chdir(work)
    last = next(reversed(steps.values()))
    provider = getattr(last, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(work)
    write_outputs(
        scenario_name="i218", output_location=str(work), subdir="i218", output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider())
    return work / "output_csv" / "i218"


def _summary_lines(csv_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_dir / "summary_solve.csv") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].endswith("(M CUR)"):
                out[row[0]] = float(row[1])
    return out


def _solve(tmp_path_factory, alt: str, values: list, entities: list) -> tuple[dict, Path]:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=BASE + values, base_chain=CHAIN, entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_w")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    assert steps and all(s.solution is not None for s in steps.values())
    return steps, work


def _var(step, name: str, entity: str) -> pl.DataFrame:
    return utf8(step.solution.value(name)).filter(pl.col("p") == entity)


def _unitsize(step, unit: str) -> float:
    return float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == unit)["value"][0])


def test_a_investment_without_invest_cost(tmp_path_factory, monkeypatch) -> None:
    steps, work = _solve(tmp_path_factory, "i218_a", [
        ["unit", "g", "invest_method", _s("invest_no_limit")],
        ["unit", "g", "fixed_cost", _f(60.0)],
    ], _unit("g"))
    step = next(iter(steps.values()))
    invest = _var(step, "v_invest_p", "g")
    assert invest["d"].to_list() == ["p2020"]
    assert invest["value"][0] * _unitsize(step, "g") == pytest.approx(100.0, rel=1e-6)
    lines = _summary_lines(_write_outputs(steps, work, monkeypatch))
    assert lines["Investment costs for realized periods (M CUR)"] == pytest.approx(0.0, abs=1e-9)
    assert lines["Fixed costs for invested entities (M CUR)"] == pytest.approx(6.0, rel=1e-6)


def test_b_reinvest_choice_without_lifetime_is_an_error(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="i218_b", base_chain=CHAIN, entities=_unit("g"), values=BASE + [
        ["unit", "g", "invest_method", _s("invest_no_limit")],
        ["unit", "g", "fixed_cost", _f(60.0)],
        ["unit", "g", "lifetime_method", _s("reinvest_choice")],
    ])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert ("unit 'g' has no lifetime, but its lifetime_method is reinvest_choice and it invests or retires"
            in str(err.value)), str(err.value)


def test_c_retirement_without_salvage_value(tmp_path_factory, monkeypatch) -> None:
    steps, work = _solve(tmp_path_factory, "i218_c", [
        ["unit", "r", "existing", _f(100.0)],
        ["unit", "r", "invest_method", _s("retire_no_limit")],
        ["unit", "r", "fixed_cost", _f(50.0)],
        ["unit", "s", "existing", _f(100.0)],
    ], _unit("r") + _unit("s"))
    step = next(iter(steps.values()))
    divest = _var(step, "v_divest_p", "r")
    assert divest["d"].to_list() == ["p2020"]
    assert divest["value"][0] * _unitsize(step, "r") == pytest.approx(100.0, rel=1e-6)
    lines = _summary_lines(_write_outputs(steps, work, monkeypatch))
    assert lines["Retirement costs / salvage revenue for realized periods (M CUR)"] == pytest.approx(0.0, abs=1e-9)
    assert lines["Fixed costs for pre-existing entities (M CUR)"] == pytest.approx(5.0, rel=1e-6)
    assert lines["Fixed cost removal due to divested entities (M CUR)"] == pytest.approx(-5.0, rel=1e-6)


def _sets(tmp_path_factory, alt: str, values: list, entities: list) -> tuple[set, set]:
    steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt=alt, base_chain=CHAIN, entities=entities,
                                           values=BASE + values))
    data = next(iter(steps.values())).flex_data

    def pairs(frame):
        return set() if frame is None else {(str(e), str(d)) for e, d in frame.select("e", "d").iter_rows()}
    return pairs(data.ed_invest_set), pairs(data.ed_divest_set)


def test_d_period_total_methods_are_one_sided(tmp_path_factory) -> None:
    invest, divest = _sets(tmp_path_factory, "i218_d", [
        ["unit", "ipt", "existing", _f(50.0)],
        ["unit", "ipt", "invest_method", _s("invest_period_total")],
        ["unit", "ipt", "salvage_value", _f(100.0)],
        ["unit", "ipt", "invest_cost", _f(1000.0)],
        ["unit", "ipt", "lifetime", _f(20.0)],
        ["unit", "rpt", "existing", _f(50.0)],
        ["unit", "rpt", "invest_method", _s("retire_period_total")],
        ["unit", "rpt", "invest_cost", _f(1000.0)],
        ["unit", "rpt", "salvage_value", _f(100.0)],
        ["unit", "rpt", "lifetime", _f(20.0)],
    ], _unit("ipt") + _unit("rpt"))
    assert ("ipt", "p2020") in invest and ("ipt", "p2020") not in divest
    assert ("rpt", "p2020") in divest and ("rpt", "p2020") not in invest


def test_e_capacity_coefficient_entity_still_offered(tmp_path_factory) -> None:
    coeff = b64({"index_type": "str", "data": [["cap_c", 1.0]]}, "map")
    invest, _divest = _sets(tmp_path_factory, "i218_e", [
        ["unit", "k", "invest_method", _s("invest_no_limit")],
        ["unit", "k", "constraint_capacity_coeff", coeff],
        ["constraint", "cap_c", "sense", _s("less_than")],
        ["constraint", "cap_c", "constant", _f(1000.0)],
    ], _unit("k") + [["constraint", "cap_c", None]])
    assert ("k", "p2020") in invest


def test_notice_names_the_newly_offered_pairs(tmp_path_factory, capsys) -> None:
    """With the database's ``invest_without_cost_offered`` marker, the run names the entities offered without a
    cost (``g``: no invest_cost; ``r``: no salvage_value), not those the old rule offered too (``c``: a non-zero
    invest_cost; ``k``: a user-constraint capacity coefficient)."""
    from spinedb_api import DatabaseMapping

    coeff = b64({"index_type": "str", "data": [["cap_c", 1.0]]}, "map")
    url, scen = make_db(tmp_path_factory, alt="i218_notice", base_chain=CHAIN, values=BASE + [
        ["unit", "g", "invest_method", _s("invest_no_limit")],
        ["unit", "c", "invest_method", _s("invest_no_limit")],
        ["unit", "c", "invest_cost", _f(1000.0)],
        ["unit", "c", "lifetime", _f(20.0)],
        ["unit", "k", "invest_method", _s("invest_no_limit")],
        ["unit", "k", "constraint_capacity_coeff", coeff],
        ["constraint", "cap_c", "sense", _s("less_than")],
        ["constraint", "cap_c", "constant", _f(1000.0)],
        ["unit", "r", "existing", _f(100.0)],
        ["unit", "r", "invest_method", _s("retire_no_limit")],
    ], entities=_unit("g") + _unit("c") + _unit("k") + _unit("r") + [["constraint", "cap_c", None]])
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name="flextool_behaviour_change:invest_without_cost_offered", value="{}")
        db.commit_session("marker")
    run(tmp_path_factory, url, scen)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "invest_without_cost_offered" in ln]
    assert len(lines) == 1, lines
    assert "Affected entities: g, r." in lines[0], lines[0]
    assert "Investment without invest_cost (entity, periods): g (p2020)." in lines[0], lines[0]
    assert "Retirement without salvage_value (entity, periods): r (p2020)." in lines[0], lines[0]
    assert "invest_max_period 0" in lines[0]


def test_f_non_storage_node_invest_method_gets_no_variable(tmp_path_factory, caplog) -> None:
    """Node investment is storage capacity: the balance node ``west`` with ``invest_method`` ``invest_no_limit``
    and no ``invest_cost`` gets no ``v_invest_n`` (before, a free degenerate column), and the validator names
    it.  Solved under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (no scale gap for a node outside ``nodeState``)."""
    import logging

    with caplog.at_level(logging.WARNING):
        steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt="i218_f", base_chain=CHAIN, values=BASE + [
            ["node", "west", "invest_method", _s("invest_no_limit")],
        ]))
    step = next(iter(steps.values()))
    nd = step.flex_data.nd_invest_set
    assert nd is None or "west" not in {str(n) for n in nd["n"].to_list()}
    try:
        assert utf8(step.solution.value("v_invest_n")).height == 0
    except KeyError:
        pass  # no v_invest_n variable at all
    assert any("Node 'west' has invest_method invest_no_limit but node_type balance" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize(("method", "invested"), [("invest_period", 0.0), ("invest_total", 100.0)])
def test_f_invest_max_period_zero_forbids_only_under_a_period_method(tmp_path_factory, method, invested) -> None:
    """``invest_max_period`` 0 forbids investment only under the *_period methods (``_INVEST_PERIOD_METHODS``):
    under invest_total the period limit is not used, so ``g`` still invests the 100 MW demand (invest_max_total
    1000 does not bind); under invest_period nothing is invested and the demand goes unserved."""
    steps, _work = _solve(tmp_path_factory, f"i218_f_{method}", [
        ["unit", "g", "invest_method", _s(method)],
        ["unit", "g", "fixed_cost", _f(60.0)],
        ["unit", "g", "invest_max_period", _f(0.0)],
        ["unit", "g", "invest_max_total", _f(1000.0)],
    ], _unit("g"))
    step = next(iter(steps.values()))
    invest = _var(step, "v_invest_p", "g")
    total = float(invest["value"].sum()) * _unitsize(step, "g") if invest.height else 0.0
    assert total == pytest.approx(invested, abs=1e-6)
