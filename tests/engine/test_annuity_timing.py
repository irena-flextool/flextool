"""The investment annuity is worth the investment cost at discount_rate for every ``inflation_offset_investment``.

The yearly payment of service year k falls at k + off years after the capacity comes online (off =
``inflation_offset_investment``: 0 the start of the year, 1 its end).  A loan of C taken when the capacity comes
online and repaid in n level payments at those moments is worth C at the interest r when each payment is

    A = C · r / (1 − (1+r)^−n) · (1+r)^(off − 1).

Hand derivation, C = 1000 CUR/kW = 1 000 000 CUR/MW, r = 5 %, n = 10:
    ann(5 %, 10) = 0.05 / (1 − 1.05^−10) = 0.129504575; C · ann = 129 504.575 CUR/MW/a (the year-end loan);
    off 1:   A = 129 504.575, payments at t = 1 … 10, PV = A · Σ_{t=1..10} 1.05^−t = A · 7.721735 = 1 000 000;
    off 0:   A = 129 504.575 / 1.05 = 123 337.690, payments at t = 0 … 9, PV = A · 8.107822 = 1 000 000;
    off 0.5: A = 129 504.575 / 1.05^0.5, payments at t = 0.5 … 9.5, PV = 1 000 000.
Before, A was the year-end annuity for every offset: at off 0 the loan was worth C · 1.05 = 1 050 000.

The present values below are summed payment by payment (not the closed form), so they check the engine's annuity
independently.  In the model (``tests.json`` single solve, 5-year periods p2020 … p2035, inflation_rate = r = 5 %,
lifetime_technical = payback_time = 10, invest_cost 400 CUR/kW) a p2020 investment is in service in years 0 … 9 and
each year's investment factor is 1.05^−(k + off): its objective coefficient is the PV of the loan, 400 000 CUR/MW,
at every offset (before: 420 000 at off 0).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import SINGLE

R = 0.05
N = 10


def _pv(payment: float, off: float) -> float:
    """Σ_{k=0..n−1} payment · (1+r)^−(k + off): the loan's payments discounted one by one."""
    return sum(payment * (1.0 + R) ** -(k + off) for k in range(N))


class _Lookup:
    """Duck-typed ``PdLookup`` over a ``{(e, param): value}`` dict (one value for every period)."""

    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self._v = values

    def get(self, e: str, param: str, d: str) -> float:
        return self._v.get((e, param), 0.0)

    def has(self, e: str, param: str, d: str) -> bool:
        return (e, param) in self._v

    def explicit_rows(self, param: str) -> list[tuple[str, float]]:
        return [(e, v) for (e, prm), v in self._v.items() if prm == param]


@pytest.mark.parametrize("off", [0.0, 0.5, 1.0])
def test_annuity_of_the_emitted_frames_is_worth_the_investment_cost(off) -> None:
    """The per-solve CSV path (``ed_entity_annual*.csv``, the retirement coefficients of a synthetic sub-solve):
    invest_cost 1000 CUR/kW, discount_rate 5 %, payback over lifetime_technical 10: PV 1 000 000 at every offset."""
    from flextool.engine.input._lifetime_roles import lookup_roles
    from flextool.engine.preprocess.emit._entity_annual import _annuities

    lookup = _Lookup({("u", "invest_cost"): 1000.0, ("u", "discount_rate"): R, ("u", "lifetime_technical"): 10.0})
    inp = {"pp": lookup, "roles": {"pp": lookup_roles(lookup)}, "offset_inv": off}
    payment = _annuities(inp, "invest_cost", [("u", "p2020", "pp", 1)])[0]
    assert _pv(payment, off) == pytest.approx(1_000_000.0, rel=1e-12)


_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _values(off: float) -> list:
    return [["model", "flexTool", "inflation_rate", b64(R, "float")],
            ["model", "flexTool", "inflation_offset_investment", b64(off, "float")],
            ["unit", PLANT, "invest_cost", b64(400.0, "float")],
            ["unit", PLANT, "discount_rate", b64(R, "float")],
            ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
            ["unit", PLANT, "invest_method", b64("invest_no_limit", "str")],
            ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
            ["unit", PLANT, "lifetime_technical", b64(10.0, "float")]]


def _plant(param) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
@pytest.mark.parametrize("off", [0.0, 0.5, 1.0])
def test_investment_is_worth_its_cost_at_every_offset(tmp_path_factory, off) -> None:
    """The objective coefficient of a p2020 investment (10 years in service, inflation_rate = discount_rate) is the
    PV of its loan, 400 CUR/kW · 1000 = 400 000 CUR/MW, at every offset; the reported yearly investment cost of
    that decision is the payment A, whose payments discounted one by one at discount_rate are worth 400 000."""
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"ann_off_{str(off).replace('.', '_')}", values=_values(off),
                            base_chain=_no_battery_invest(SINGLE))
    fd = list(steps.values())[-1].flex_data
    assert _plant(fd.ed_entity_annual_discounted)["p2020"] == pytest.approx(400_000.0, rel=1e-12)
    assert _pv(_plant(fd.ed_annuity)["p2020"], off) == pytest.approx(400_000.0, rel=1e-12)


@pytest.mark.solver
def test_notice_names_entities_whose_annuity_changes(tmp_path_factory, capsys) -> None:
    """With the upgrade's ``annuity_timing`` marker naming the plant, a run at offset 0 names it once; at offset 1
    its annuity is that of before and no notice is printed."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX
    from tests.engine._unit_db import make_db

    def lines(alt: str, off: float) -> list[str]:
        url, scen = make_db(tmp_path_factory, alt=alt, values=_values(off), base_chain=_no_battery_invest(SINGLE))
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=MARKER_PREFIX + "annuity_timing",
                        value=f'{{"since": 81, "entities": ["{PLANT}"]}}')
            db.commit_session("behaviour-change marker")
        capsys.readouterr()
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
        summary = BEHAVIOUR_CHANGES["annuity_timing"].summary
        return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]

    named = lines("ann_notice_off0", 0.0)
    assert len(named) == 1 and f"Affected entities: {PLANT}." in named[0], named
    assert lines("ann_notice_off1", 1.0) == []
