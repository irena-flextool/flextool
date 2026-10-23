"""CO2 caps measure emissions like the CO2 price and the reported emissions.

``co2_max_period`` (tonnes per year in each period) and ``co2_max_total`` count the fuel that processes draw
from the capped group's nodes times the commodity's ``co2_content``.  For a ``min_load_efficiency`` unit that
fuel is ``v_flow·slope + v_online·section`` (times unitsize), as in the CO2 price, the commodity cost and the
node balance; before, the caps left out the section fuel of the units online.

Fixtures: ``tests.json`` + override alternatives (built from JSON).  The ``coal_min_load`` alternative makes
``coal_plant`` (500 MW, eff 0.4, 0.3 at min load 0.5, linear startup) a min_load_efficiency unit, so slope =
1.667 and section = 0.833 (MWh fuel per MWh output, per MW online); ``co2_limit`` caps group ``co2_limit``
(node ``coal_market``, coal 0.34 t/MWh) at 3e6 t/a in p2020.
"""
from __future__ import annotations

import collections

import polars as pl
import pytest

from tests.engine._unit_db import COAL, PLANT, b64, build, flow, online, solve, uc, utf8

MINLOAD_CHAIN = ["init", "west", "coal", "co2_limit", "coal_min_load"]
GROUP = "co2_limit"


def _row_terms(pb, family: str) -> dict[str, dict[str, float]]:
    """``{row: {column: coefficient}}`` of the canonical rows of ``family``."""
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def _frame(param) -> pl.DataFrame:
    return utf8(param.frame if hasattr(param, "frame") else param)


def _value(param, **keys) -> float:
    """The single value of ``param`` at ``keys`` (keys absent from the Param's columns are ignored)."""
    f = _frame(param)
    for col, val in keys.items():
        if col in f.columns:
            f = f.filter(pl.col(col) == val)
    assert f.height == 1, (keys, f)
    return float(f["value"][0])


def _weight(fd, d: str, t: str) -> float:
    """w[d,t] = step_duration · timestep_weight / period_share."""
    return (_value(fd.p_step_duration, d=d, t=t) * _value(fd.p_timestep_weight, d=d, t=t)
            / _value(fd.p_period_share, d=d))


def _annual_emissions(step, kind: str, d: str) -> float:
    """Annualised CO2 of coal_plant in period ``d`` (t/a), section fuel included, from the solution."""
    fd = step.flex_data
    times = utf8(fd.dt).filter(pl.col("d") == d)["t"].to_list()
    out = flow(step, PLANT, COAL, "west").to_list()
    on = online(step, PLANT, kind).to_list()
    co2 = _value(fd.p_co2_content, c="coal")
    dts = utf8(fd.dt).sort("d", "t").select("d", "t").rows()
    total = 0.0
    for (dd, t), f_mw, on_mw in zip(dts, out, on, strict=True):
        if dd != d or t not in times:
            continue
        fuel = (f_mw * _value(fd.p_slope, p=PLANT, d=d, t=t)
                + on_mw * _value(fd.p_section, p=PLANT, d=d, t=t))
        total += fuel * co2 * _weight(fd, d, t)
    return total


def _minload_step(tpf, alt: str, method: str, values: list):
    return solve(tpf, alt=alt, base_chain=MINLOAD_CHAIN, values=uc(PLANT, 0.5, method) + values)


def _assert_section_coefficients(step, terms: dict[str, float], kind: str, d: str, years: float = 1.0) -> None:
    """``v_online_<kind>[coal_plant,d,t]`` carries section·unitsize·co2_content·w·years, and its ratio to the
    ``v_flow`` coefficient of the same step is section/slope."""
    fd = step.flex_data
    co2 = _value(fd.p_co2_content, c="coal")
    unitsize = _value(fd.p_unitsize, p=PLANT)
    seen = 0
    for t in utf8(fd.dt).filter(pl.col("d") == d)["t"].to_list():
        section = _value(fd.p_section, p=PLANT, d=d, t=t)
        slope = _value(fd.p_slope, p=PLANT, d=d, t=t)
        on_col = f"v_online_{kind}[{PLANT},{d},{t}]"
        flow_col = f"v_flow[{PLANT},{COAL},west,{d},{t}]"
        assert on_col in terms, on_col
        assert terms[on_col] == pytest.approx(section * unitsize * co2 * _weight(fd, d, t) * years, rel=1e-9)
        assert terms[on_col] / terms[flow_col] == pytest.approx(section / slope, rel=1e-9)
        seen += 1
    assert seen > 0


@pytest.mark.parametrize("method,kind", [("linear", "linear"), ("binary", "integer")])
def test_co2_max_period_counts_minload_section(tmp_path_factory, method, kind) -> None:
    """The period cap carries the section fuel of the units online, and the solved emissions (section fuel
    included) stay within the cap.  Before, ``v_online`` was absent from the row and the emissions exceeded
    the cap by the section share."""
    step = _minload_step(tmp_path_factory, f"co2c_period_{kind}", method, [])
    fd = step.flex_data
    rows = _row_terms(build(fd), "co2_max_period")
    terms = rows[f"co2_max_period[{GROUP},p2020]"]
    _assert_section_coefficients(step, terms, kind, "p2020")

    cap = _value(fd.p_co2_max_period, g=GROUP, d="p2020")
    emitted = _annual_emissions(step, kind, "p2020")
    assert emitted <= cap * (1 + 1e-6)
    assert emitted >= cap * (1 - 1e-3)  # the cap binds (the MIP stops within its gap)


def test_co2_max_total_counts_minload_section(tmp_path_factory) -> None:
    """The total cap carries the same section term (one period with 1 year: coefficients as in the period
    cap)."""
    step = _minload_step(tmp_path_factory, "co2c_total", "linear", [
        ["group", GROUP, "co2_method", b64("total", "str")],
        ["group", GROUP, "co2_max_total", b64(2.5e6, "float")]])
    fd = step.flex_data
    rows = _row_terms(build(fd), "co2_max_total")
    terms = rows[f"co2_max_total[{GROUP}]"]
    _assert_section_coefficients(step, terms, "linear", "p2020")

    emitted = _annual_emissions(step, "linear", "p2020")
    assert emitted <= 2.5e6 * (1 + 1e-6)
    assert emitted >= 2.5e6 * (1 - 1e-3)
