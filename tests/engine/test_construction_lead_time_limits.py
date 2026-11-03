"""Construction lead time: capacity invested (coming online), in construction and in service.

With a construction lead time ``invested`` is the capacity that starts to operate in a period, ``in construction``
the capacity decided but not yet operating; the per-period investment limits count capacity coming online, and
the user-constraint terms follow the same meaning, so capacity in service = pre-built + invested − retired.

Hand values from ``tests/fixtures/tests.json``: coal_plant existing 500 MW (unit size 500 MW), lifetime 30
(reinvest_automatic: the existing and every investment stay in service to the end of the horizon), periods
p2020 … p2035 five years apart, the model horizon ends at 2040.  ``invest_min_period = invest_max_period``
pins what comes online in each period.

* single solve ``y2020_2035_5week`` (p2020 … p2035, all realized);
* chain ``y2020_5week`` (p2020 + p2025, realizes p2020), ``y2025_5week`` (p2025 + p2030, realizes p2025),
  ``y2030_5week`` (p2030 + p2035, realizes p2030), ``y2035_5week`` (p2035).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8
from tests.engine.test_user_constraint_capacity_terms import CHAIN, GE, PLANT, SINGLE, _constraint, _map

PERIODS = ["p2020", "p2025", "p2030", "p2035"]


def _period_map(data: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")


def _values(lead: float, online: dict[str, float]) -> list:
    pins = {d: online.get(d, 0.0) for d in PERIODS}
    return ([["unit", PLANT, "construction_lead_time", b64(lead, "float")],
             ["unit", PLANT, "invest_method", b64("invest_period", "str")],
             ["unit", PLANT, "invest_min_period", _period_map(pins)],
             ["unit", PLANT, "invest_max_period", _period_map(pins)],
             ["unit", PLANT, "constraint_capacity_coeff", _map({"c_cap": 1.0})],
             ["unit", PLANT, "constraint_cumulative_pre_built_capacity_coeff", _map({"c_twin": 1.0})],
             ["unit", PLANT, "constraint_invested_capacity_coeff", _map({"c_twin": 1.0})],
             ["unit", PLANT, "constraint_in_construction_capacity_coeff", _map({"c_build": 1.0})]]
            + _constraint("c_cap") + _constraint("c_twin") + _constraint("c_build"))


ENTITIES = [["constraint", "c_cap", None], ["constraint", "c_twin", None], ["constraint", "c_build", None]]


def _steps(tmp_path_factory, alt: str, lead: float, online: dict[str, float], chain: list) -> list:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(lead, online), base_chain=chain,
                        entities=ENTITIES)
    return list(run(tmp_path_factory, url, scen).values())


def _us(step) -> float:
    return float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])


def _decisions(step) -> dict[str, float]:
    """``{decision period: MW}`` of coal_plant in ``step``."""
    if "v_invest_p" not in step.solution._vars:
        return {}
    v = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == PLANT)
    return dict(zip(v["d"].to_list(), (v["value"] * _us(step)).to_list()))


def _by_set(step, field: str, carried: str) -> dict[str, float]:
    """``{d: MW}``: Σ decisions over ``field`` (``[e, d_invest, d]``) plus the carried capacity, per period."""
    fd = step.flex_data
    dec = _decisions(step)
    out: dict[str, float] = {}
    rows = getattr(fd, field)
    if rows is not None:
        for o, d in utf8(rows).filter(pl.col("e") == PLANT).select("d_invest", "d").iter_rows():
            out[d] = out.get(d, 0.0) + dec.get(o, 0.0)
    param = getattr(fd, carried)
    if param is not None:
        for d, v in utf8(param.frame).filter(pl.col("e") == PLANT).select("d", "value").iter_rows():
            out[d] = out.get(d, 0.0) + v
    return out


def _total(step) -> dict[str, float]:
    """Capacity in service: existing (with what earlier solves built) + the solve's investments in service."""
    fd = step.flex_data
    dec = _decisions(step)
    out = dict(utf8(fd.p_entity_all_existing.frame).filter(pl.col("e") == PLANT).select("d", "value").iter_rows())
    if fd.edd_invest_set is not None:
        for o, d in utf8(fd.edd_invest_set).filter(pl.col("e") == PLANT).select("d_invest", "d").iter_rows():
            out[d] = out.get(d, 0.0) + dec.get(o, 0.0)
    return out


def _realized(steps: list, single: bool, fn) -> dict[str, float]:
    """``fn(step)`` per period, each period from the solve that realizes it."""
    if single:
        return {d: fn(steps[-1]).get(d, 0.0) for d in PERIODS}
    out = {}
    for step, d in zip(steps, PERIODS):
        out[d] = fn(step).get(d, 0.0)
    return out


def _twin_rows(pb) -> None:
    """Row by row: capacity in service == pre-built + invested (coal_plant cannot retire)."""
    cap = row_terms(pb, GE + "_period", "c_cap,")
    twin = row_terms(pb, GE + "_period", "c_twin,")
    assert sorted(cap) == sorted(r.replace("c_twin", "c_cap") for r in twin)
    for row, terms in cap.items():
        assert terms == pytest.approx(twin[row.replace("c_cap", "c_twin")], rel=1e-12), row


@pytest.mark.parametrize("single", [True, False], ids=["single_solve", "chain_of_solves"])
def test_five_year_lead_invested_in_construction_total(tmp_path_factory, single) -> None:
    """L = 5, coming online pinned to p2025: 100, p2030: 50, else 0.  A decision of o comes online in the next
    period, so the decisions are p2020: 100, p2025: 50 (p2030: 0).  invested {p2025: 100, p2030: 50},
    in construction {p2020: 100, p2025: 50, p2030: 0, p2035: 0}, capacity in service 500 + {0, 100, 150, 150};
    the same in a single solve and in a chain."""
    steps = _steps(tmp_path_factory, f"lt5_{single}", 5.0, {"p2025": 100.0, "p2030": 50.0},
                   SINGLE if single else CHAIN)
    if not single:
        steps = steps[:4]
    invested = _realized(steps, single, lambda s: _by_set(s, "edd_invest_commission", "p_entity_carried_commission"))
    building = _realized(steps, single,
                         lambda s: _by_set(s, "edd_invest_in_construction", "p_entity_carried_in_construction"))
    total = _realized(steps, single, _total)
    assert invested == pytest.approx({"p2020": 0.0, "p2025": 100.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6)
    assert building == pytest.approx({"p2020": 100.0, "p2025": 50.0, "p2030": 0.0, "p2035": 0.0}, abs=1e-6)
    assert total == pytest.approx({"p2020": 500.0, "p2025": 600.0, "p2030": 650.0, "p2035": 650.0}, abs=1e-6)
    for step in steps:
        _twin_rows(build(step.flex_data))


def test_ten_year_lead_single_solve(tmp_path_factory) -> None:
    """L = 10, coming online pinned to p2030: 100, p2035: 50.  Decisions p2020: 100, p2025: 50; the p2030 and
    p2035 decisions would start at the end of the horizon (2040) or later and are not offered.  invested
    {p2030: 100, p2035: 50}, in construction {p2020: 100, p2025: 150, p2030: 50, p2035: 0}, capacity in service
    500 + {0, 0, 100, 150}.  The in-construction row of p2025 carries v_invest[p2020] and v_invest[p2025]."""
    steps = _steps(tmp_path_factory, "lt10", 10.0, {"p2030": 100.0, "p2035": 50.0}, SINGLE)
    step = steps[-1]
    assert _decisions(step) == pytest.approx({"p2020": 100.0, "p2025": 50.0}, abs=1e-6)
    assert ({d: _by_set(step, "edd_invest_commission", "p_entity_carried_commission").get(d, 0.0) for d in PERIODS}
            == pytest.approx({"p2020": 0.0, "p2025": 0.0, "p2030": 100.0, "p2035": 50.0}, abs=1e-6))
    assert ({d: _by_set(step, "edd_invest_in_construction", "p_entity_carried_in_construction").get(d, 0.0)
             for d in PERIODS}
            == pytest.approx({"p2020": 100.0, "p2025": 150.0, "p2030": 50.0, "p2035": 0.0}, abs=1e-6))
    assert _total(step) == pytest.approx({"p2020": 500.0, "p2025": 500.0, "p2030": 600.0, "p2035": 650.0},
                                         abs=1e-6)
    pb = build(step.flex_data)
    _twin_rows(pb)
    us = _us(step)
    rows = row_terms(pb, GE + "_period", "c_build,")
    in_2025 = next(t for r, t in rows.items() if r.endswith(",p2025]"))
    assert in_2025 == pytest.approx({f"v_invest_p[{PLANT},p2020]": us, f"v_invest_p[{PLANT},p2025]": us},
                                    rel=1e-12)


def test_min_limit_nothing_can_come_online(tmp_path_factory) -> None:
    """invest_min_period 100 in p2020 with a 5-year lead time: no decision of the solve comes online in p2020,
    so the floor cannot be met — a configuration error naming the entity, the period and the lead time."""
    url, scen = make_db(tmp_path_factory, alt="lt_min_p2020", values=_values(5.0, {"p2020": 100.0}),
                        base_chain=SINGLE, entities=ENTITIES)
    with pytest.raises(Exception) as err:
        run(tmp_path_factory, url, scen)
    chain = [err.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    config = [e for e in chain if isinstance(e, FlexToolConfigError)]
    assert config, chain
    text = str(config[0])
    assert "coal_plant" in text and "p2020" in text and "5-year construction_lead_time" in text
