"""The cost lines of the reported costs add up to the objective's terms they split (the lifetime-cost line
frames, ``params._cost_lines``).

Per decision (row) of each solve, to 1e-12 relative:

* Σ_lines ``edl_invest_lines_run`` [e, d] = the reported investment lump ``ed_entity_annual_discounted_run`` [e, d]
  (= Σ of its starts ``edd_entity_annual_discounted_run``; in a single solve the objective's lump);
* Σ_lines ``edl_divest_lines`` [e, d] = −``ed_divest_salvage`` [e, d] (the cost of a MW of existing capacity
  retired in d);
* Σ_lines ``cdl_divest_built_lines`` [e, c, d] = ``cd_divest_built_cost`` [e, c, d];
* Σ_lines ``cdl_extend_lines`` [e, c, d] = ``cd_extend_cost`` [e, c, d];
* Σ_lines ``edl_existing_credit_lines`` [e, d] = −``ed_existing_salvage_credit`` [e, d] and, in one solve,
  Σ_lines ``edl_existing_replacement_lines`` [e, d] = ``ed_existing_replacement`` [e, d] (the objective constant's
  term).

Cases: reinvest_choice with a salvage value (single solve and chain), reinvest_automatic (chains of plants, the
existing capacity's replacements), retirement of existing and of built capacity, a life extension, and a
stochastic fan.  The lines exist wherever their terms do.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE, STOCH_CHAIN, STOCH_JSON

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _plant(method: str, invest: str = "invest_retire_no_limit", **extra) -> list:
    values = [["model", "flexTool", "social_discount_rate", b64(0.05, "float")],
              ["unit", PLANT, "invest_method", b64(invest, "str")],
              ["unit", PLANT, "lifetime_method", b64(method, "str")],
              ["unit", PLANT, "lifetime_technical", b64(10.0, "float")],
              ["unit", PLANT, "invest_cost", b64(400.0, "float")],
              ["unit", PLANT, "discount_rate", b64(0.08, "float")],
              ["unit", PLANT, "net_salvage_value", b64(50.0, "float")]]
    return values + [["unit", PLANT, k, v] for k, v in extra.items()]


CASES = {
    "choice_single": (_plant("reinvest_choice"), _no_battery_invest(SINGLE), None),
    "choice_chain": (_plant("reinvest_choice"), _no_battery_invest(CHAIN), None),
    "automatic_single": (_plant("reinvest_automatic"), _no_battery_invest(SINGLE), None),
    "automatic_chain": (_plant("reinvest_automatic"), _no_battery_invest(CHAIN), None),
    "extension": (_plant("reinvest_choice", life_extension_cost=b64(100.0, "float"),
                         life_extension_years=b64(10.0, "float")), _no_battery_invest(SINGLE), None),
    "stochastic": ([["unit", "peaker", "lifetime_method", b64("reinvest_choice", "str")],
                    ["unit", "peaker", "invest_method", b64("invest_retire_no_limit", "str")],
                    ["unit", "peaker", "lifetime_technical", b64(1.0, "float")],
                    ["unit", "peaker", "net_salvage_value", b64(0.1, "float")]], STOCH_CHAIN, STOCH_JSON),
}


def _steps(tmp_path_factory, case: str) -> dict:
    values, chain, fixture = CASES[case]
    kwargs = {} if fixture is None else {"fixture_json": fixture}
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"lines_{case}", values=values, base_chain=chain, **kwargs)
    return steps


def _summed(lines: pl.DataFrame | None, keys: tuple[str, ...]) -> dict:
    if lines is None:
        return {}
    return {tuple(r[:-1]): r[-1] for r in lines.group_by(*keys).agg(pl.col("value").sum()).iter_rows()}


def _param(param, keys: tuple[str, ...], sign: float = 1.0) -> dict:
    if param is None:
        return {}
    frame = utf8(param.frame) if hasattr(param, "frame") else param
    return {tuple(r[:-1]): sign * r[-1] for r in frame.select(*keys, "value").iter_rows()}


def _assert_same(lines: dict, terms: dict, what: str) -> None:
    for key, value in terms.items():
        assert lines.get(key, 0.0) == pytest.approx(value, rel=1e-12, abs=1e-6), (what, key)
    for key, value in lines.items():
        assert key in terms or value == pytest.approx(0.0, abs=1e-6), (what, key)


@pytest.mark.solver
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_lines_add_up_to_the_terms(tmp_path_factory, case) -> None:
    seen = set()
    for name, step in _steps(tmp_path_factory, case).items():
        fd = step.flex_data
        lump = fd.ed_entity_annual_discounted_run
        if lump is not None:
            _assert_same(_summed(fd.edl_invest_lines_run, ("e", "d")), _param(lump, ("e", "d")), f"{name} invest")
            seen.add("invest")
        if fd.ed_divest_salvage is not None:
            _assert_same(_summed(fd.edl_divest_lines, ("e", "d")), _param(fd.ed_divest_salvage, ("e", "d"), -1.0),
                         f"{name} divest")
            seen.add("divest")
        if fd.cd_divest_built_cost is not None:
            _assert_same(_summed(fd.cdl_divest_built_lines, ("e", "cohort", "d")),
                         _param(fd.cd_divest_built_cost, ("e", "cohort", "d")), f"{name} built")
            seen.add("built")
        if fd.cd_extend_cost is not None:
            _assert_same(_summed(fd.cdl_extend_lines, ("e", "cohort", "d")),
                         _param(fd.cd_extend_cost, ("e", "cohort", "d")), f"{name} extend")
            seen.add("extend")
        if fd.ed_existing_salvage_credit is not None:
            _assert_same(_summed(fd.edl_existing_credit_lines, ("e", "d")),
                         _param(fd.ed_existing_salvage_credit, ("e", "d"), -1.0), f"{name} credit")
            seen.add("credit")
        if fd.ed_existing_replacement is not None and len(_steps(tmp_path_factory, case)) == 1:
            # One solve: the run is the solve, and the replacement lines are the objective constant's terms.
            _assert_same(_summed(fd.edl_existing_replacement_lines, ("e", "d")),
                         _param(fd.ed_existing_replacement, ("e", "d")), f"{name} replacement")
            seen.add("replacement")
    want = {"choice_single": {"invest", "divest", "built", "credit"}, "automatic_single": {"invest", "replacement", "divest", "built"},
            "extension": {"invest", "extend"}, "stochastic": {"invest", "built"}}
    assert want.get(case, {"invest"}) <= seen, (case, seen)
