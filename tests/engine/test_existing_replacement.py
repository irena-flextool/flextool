"""The existing capacity of a ``reinvest_automatic`` entity reaches its end of life and is replaced; retiring it
avoids every replacement still to come (``_npv.existing_salvage`` / ``chain_retirement_costs``), on hand-derived
cases.

Model: ``tests.json``, the single solve over p2020 … p2035 (5-year periods, run years 0, 5, 10, 15, horizon end
20).  coal_plant: 500 MW existing (age 0), lifetime_method reinvest_automatic, lifetime_technical L = 10,
invest_method retire_no_limit (retirement only), discount_rate r = 5 %, social_discount_rate i = 5 %,
discount_offset_investment 1, discount_offset_operations 0.5.  The existing plant serves [0, 10); its replacement,
online in p2030 at the values of p2030, serves [10, 20).  Per MW, v = 1/(1+i), C = 1000·invest_cost,
S = 1000·net_salvage_value::

    A = C·r/(1 − (1+r)^−10),  S[a, b) = Σ_{y=a}^{b−1} v^(y+1),  D(t) = v^t
    a_x = S·(1+r)^−10·r/(1 − (1+r)^−10)  (the existing plant's level salvage credit over its 10 years)

The objective's retirement term is −v_divest·unitsize·(divest_salvage + lifetime fixed cost saved); the tests read
``ed_divest_salvage`` (the saving of a retired MW, fixed cost apart):

* invest_cost 400, no salvage: a retirement before p2030 (the existing plant in service) or at p2030 (the
  replacement's own start: never bought) avoids the replacement, A·S[10, 20) = C·v^10 = 245 565.30; one in p2035
  (the replacement at m = 5) pays its contract as it ran, A·S[10, 15) + D(15)·B(5), and gives its price back,
  A·S[10, 20): at r = i both are C·v^10, saving 0.
* no invest_cost, net_salvage_value 200 (the replacement is free and keeps the salvage value, the owner's rule):
  retiring in p2025 receives S·D(5), gives back the existing plant's credits a_x·S[0, 10) = S·v^10 and the free
  replacement's salvage credit −a_x·S[10, 20) = −S·v^20: saving S·(v^5 − v^10 − v^20) = −41 455.31 (a cost).
* no invest_cost, no salvage, fixed_cost 10: no settlement at all (the retirement saves its fixed cost only).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import SINGLE

R = 0.05
V = 1.0 / 1.05
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def ann(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** -n)


def span(a: int, b: int) -> float:
    return sum(V ** (y + 1) for y in range(a, b))


def _values(*, cost: float | None = None, salvage: float | None = None, fixed: float = 0.0,
            extra: list | None = None) -> list:
    values = [["model", "flexTool", "social_discount_rate", b64(R, "float")],
              ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
              ["unit", PLANT, "discount_rate", b64(R, "float")],
              ["unit", PLANT, "fixed_cost", b64(fixed, "float")],
              ["unit", PLANT, "invest_method", b64("retire_no_limit", "str")],
              ["unit", PLANT, "lifetime_method", b64("reinvest_automatic", "str")],
              ["unit", PLANT, "lifetime_technical", b64(10.0, "float")]]
    values.append(["unit", PLANT, "invest_cost", b64(cost if cost is not None else 0.0, "float")])
    if salvage is not None:
        values.append(["unit", PLANT, "net_salvage_value", b64(salvage, "float")])
    return values + (extra or [])


def _step(tmp_path_factory, alt: str, values: list):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    return list(steps.values())[-1]


def _salvage(step) -> dict[str, float]:
    param = step.flex_data.ed_divest_salvage
    if param is None:
        return {}
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
def test_retiring_existing_capacity_avoids_its_priced_replacement(tmp_path_factory) -> None:
    a = 400_000.0 * ann(R, 10.0)
    avoided = a * span(10, 20)
    assert avoided == pytest.approx(400_000.0 * V ** 10, rel=1e-12)
    assert avoided == pytest.approx(245_565.30, abs=0.01)
    got = _salvage(_step(tmp_path_factory, "er_priced", _values(cost=400.0)))
    assert got["p2020"] == pytest.approx(avoided, rel=1e-12)
    assert got["p2025"] == pytest.approx(avoided, rel=1e-12)
    assert got["p2030"] == pytest.approx(avoided, rel=1e-12)
    assert got.get("p2035", 0.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.solver
def test_free_replacement_keeps_its_salvage_value(tmp_path_factory) -> None:
    s = 200_000.0
    a_x = s * 1.05 ** -10 * ann(R, 10.0)
    assert a_x == pytest.approx(15_900.91, abs=0.01)
    assert a_x * span(0, 10) == pytest.approx(s * V ** 10, rel=1e-12)
    want = s * V ** 5 - a_x * span(0, 10) - a_x * span(10, 20)
    assert want == pytest.approx(s * (V ** 5 - V ** 10 - V ** 20), rel=1e-12)
    assert want == pytest.approx(-41_455.31, abs=0.01)
    got = _salvage(_step(tmp_path_factory, "er_free_salvage", _values(salvage=200.0)))
    assert got["p2025"] == pytest.approx(want, rel=1e-12)


@pytest.mark.solver
def test_free_replacement_without_salvage_has_no_settlement(tmp_path_factory) -> None:
    """The retirement saves the fixed cost of every period from its own on (10 · 1000 a year at the middle of
    the year): p2025 10 000·Σ_{y=5..19} v^(y+0.5) = 83 335.72, p2030 48 575.42; nothing else."""
    step = _step(tmp_path_factory, "er_free", _values(fixed=10.0))
    assert _salvage(step) == {}
    frame = utf8(step.flex_data.ed_lifetime_fixed_cost_divest.frame).filter(pl.col("e") == PLANT)
    saved = dict(zip(frame["d"].to_list(), frame["value"].to_list()))
    assert saved["p2025"] == pytest.approx(10_000.0 * sum(V ** (y + 0.5) for y in range(5, 20)), rel=1e-12)
    assert saved["p2025"] == pytest.approx(83_335.72, abs=0.01)
    assert saved["p2030"] == pytest.approx(48_575.42, abs=0.01)


@pytest.mark.solver
def test_replacement_constant_moves_the_objective_not_the_decisions(tmp_path_factory) -> None:
    """invest_cost 400: the replacement of the 500 MW is the objective constant 500 · A·S[10, 20) = 122 782 650
    with include_existing_fixed_cost (no fixed cost, no salvage, so nothing else is in it), and the retirements
    are the same with and without it: the retirement coefficient gives the replacement back either way."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    step = _step(tmp_path_factory, "er_priced", _values(cost=400.0))
    constant = 500.0 * 400_000.0 * ann(R, 10.0) * span(10, 20)
    assert constant == pytest.approx(122_782_650.0, abs=1.0)

    def solve(include: bool):
        pb = Problem()
        build_flextool(pb, step.flex_data, include_existing_fixed_cost=include)
        sol = pb.solve()
        assert sol.optimal
        retired = utf8(sol.value("v_divest_p")).filter(pl.col("p") == PLANT)
        return sol.obj, dict(zip(retired["d"].to_list(), retired["value"].to_list()))

    (with_obj, with_retired), (without_obj, without_retired) = solve(True), solve(False)
    assert with_retired == pytest.approx(without_retired, abs=1e-9)
    assert with_obj - without_obj == pytest.approx(constant, rel=1e-9)


@pytest.mark.solver
def test_replacement_is_reported_as_investment_pre_existing(tmp_path_factory) -> None:
    """invest_cost 400, the 500 MW kept (no retirement option): the replacement is reported in p2030, the period it
    comes online, as 500 · A·S[10, 20) (M CUR), and annualized as that event divided by p2030's investment factor
    S[10, 15) (nothing in p2035); the same in the chain of solves as in the single solve (the run's existing chain
    is placed once)."""
    from tests.engine.test_chain_costs import MEGA, _results, _row
    from tests.engine.test_user_constraint_capacity_terms import CHAIN
    a = 400_000.0 * ann(R, 10.0)
    values = [v for v in _values(cost=400.0) if v[2] != "invest_method"] + [
        ["unit", PLANT, "invest_method", b64("not_allowed", "str")]]
    want = {"p2020": 0.0, "p2025": 0.0, "p2030": 500.0 * a * span(10, 20), "p2035": 0.0}
    yearly = {"p2020": 0.0, "p2025": 0.0, "p2030": 500.0 * a * span(10, 20) / span(10, 15), "p2035": 0.0}
    for kind, chain in (("single", SINGLE), ("chain", CHAIN)):
        results = _results(tmp_path_factory, f"er_report_{kind}", values, _no_battery_invest(chain))
        for d in want:
            got = _row(results["cost_unit_discounted__d"], d)["investment pre-existing"] * MEGA
            assert got == pytest.approx(want[d], rel=1e-7, abs=1e-6), (kind, d)
            got = _row(results["cost_unit_annualized__d"], d)["investment pre-existing"] * MEGA
            assert got == pytest.approx(yearly[d], rel=1e-7, abs=1e-6), (kind, d)


@pytest.mark.solver
def test_free_replacement_with_salvage_is_named(tmp_path_factory, capsys) -> None:
    """No invest_cost, net_salvage_value 200: the replacement is free but keeps its salvage value; a warning names
    the entity once."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import make_db
    url, scen = make_db(tmp_path_factory, alt="er_free_warning", values=_values(salvage=200.0),
                        base_chain=_no_battery_invest(SINGLE))
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("er_free_warning_w"))
    out = capsys.readouterr().out
    assert out.count(f"Warning: {PLANT}: the existing capacity (lifetime_method reinvest_automatic) is replaced") == 1
    assert f"Info: the existing capacity of {PLANT} (from year 10 of the run)" in out


@pytest.mark.solver
@pytest.mark.parametrize("cost, salvage, named", [(None, 200.0, True), (100.0, 200.0, False)])
def test_only_a_free_replacement_with_salvage_is_named(tmp_path_factory, capsys, cost, salvage, named) -> None:
    """The warning names a replacement without an invest_cost that keeps a salvage value, not a priced one,
    even when its salvage value discounted at r over its life (S·1.05^−10 = 122 782.65 for S 200 000) exceeds its
    cost C 100 000 (the owner's rule: no cap and no warning on a priced replacement)."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import make_db
    alt = f"er_gain_warning_{cost}"
    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(cost=cost, salvage=salvage),
                        base_chain=_no_battery_invest(SINGLE))
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    warned = f"Warning: {PLANT}: the existing capacity (lifetime_method reinvest_automatic) is replaced"
    assert capsys.readouterr().out.count(warned) == (1 if named else 0)


@pytest.mark.solver
@pytest.mark.parametrize("method", ["next_seam", "closest_seam"])
def test_existing_plant_credit_of_a_rounded_life(tmp_path_factory, method) -> None:
    """lifetime 30, age 29, salvage 200: one year of the existing plant remains.  next_seam rounds its end to year 5:
    it serves 5 years and its credits sum to its salvage once, S·v^5 = 156 705.23 (D(5) of the salvage at its
    rounded end of life).  closest_seam rounds it to the run start: it serves no period and holds no credit; its
    replacement serves from the run start."""
    values = [v for v in _values(salvage=200.0) if v[2] != "lifetime_technical"] + [
        ["unit", PLANT, "lifetime_technical", b64(30.0, "float")],
        ["unit", PLANT, "age_at_model_start", b64(29.0, "float")],
        ["unit", PLANT, "end_of_life_method", b64(method, "str")]]
    step = _step(tmp_path_factory, f"er_rounded_{method}", values)
    credit = step.flex_data.ed_existing_salvage_credit
    if method == "closest_seam":
        assert credit is None or utf8(credit.frame).filter(pl.col("e") == PLANT).height == 0
        return
    frame = utf8(credit.frame).filter(pl.col("e") == PLANT)
    assert sum(frame["value"].to_list()) == pytest.approx(200_000.0 * V ** 5, rel=1e-12)
    assert 200_000.0 * V ** 5 == pytest.approx(156_705.23, abs=0.01)
