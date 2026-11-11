"""The operating-cost weight covers every period and time step of the solve (item 107).

Every operating cost term is weighted by

    op_factor[d, t] = step_duration[d, t] · timestep_weight[d, t] · inflation_op[d] / period_share[d]

and the product is an inner join: a period or (period, time step) that one of the four Params lacks would drop
out of the objective without a word (a partial ``solve_data`` CSV of a reloaded run, a hand-built FlexData).
``build_flextool`` raises a ``FlexToolConfigError`` naming the parameter and the gap instead.

Vehicle (oracle 107-T1): two one-hour periods p2020 / p2025 (one step t0001 each), weights and shares 1, node n
with inflow -5 MW and penalties 1e4, one unit u feeding n with a commodity of price 10 at efficiency 1 and
10 MW of capacity.  Each period dispatches 5 MW for 1 h at 10 = 50, so with inflation_op {p2020: 1, p2025: 2}
the objective is 50·1 + 50·2 = 150 (the bug signature without the checks: 50).
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine import build_flextool
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.flex_data import FlexData

from .conftest import solver_options

PERIODS = ["p2020", "p2025"]


def _data(*, inflation: dict[str, float], drop_step: str | None = None, drop_share: bool = False) -> FlexData:
    """The 107-T1 vehicle; ``drop_step`` removes the (p2025, t0001) row of that (d, t) Param, ``drop_share``
    the p2025 row of ``p_period_share``."""
    shares = {d: 1.0 for d in PERIODS if not (drop_share and d == "p2025")}
    dt = pl.DataFrame({"d": PERIODS, "t": ["t0001", "t0001"]})

    def dt_param(name: str, value: float) -> Param:
        frame = dt.with_columns(value=pl.lit(value))
        if drop_step == name:
            frame = frame.filter(pl.col("d") != "p2025")
        return Param(("d", "t"), frame)

    nodeBalance = pl.DataFrame({"n": ["n"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    pss = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_dt = pss.join(dt, how="cross")
    return FlexData(
        dt=dt,
        p_step_duration=dt_param("p_step_duration", 1.0),
        p_timestep_weight=dt_param("p_timestep_weight", 1.0),
        p_inflation_op=Param(("d",), pl.DataFrame({"d": list(inflation), "value": list(inflation.values())})),
        p_period_share=Param(("d",), pl.DataFrame({"d": list(shares), "value": list(shares.values())})),
        nodeBalance=nodeBalance, nodeBalance_dt=nodeBalance_dt,
        p_inflow=Param(("n", "d", "t"), nodeBalance_dt.with_columns(value=pl.lit(-5.0))),
        p_penalty_up=Param(("n", "d", "t"), nodeBalance_dt.with_columns(value=pl.lit(1e4))),
        p_penalty_down=Param(("n", "d", "t"), nodeBalance_dt.with_columns(value=pl.lit(1e4))),
        process_source_sink=pss,
        process_source_sink_eff=pss.clone(),
        process_source_sink_noEff=pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8}),
        pss_dt=pss_dt,
        flow_to_n=pss.with_columns(n=pl.col("sink")),
        flow_from_commodity_eff=pss.with_columns(c=pl.lit("FUEL")),
        flow_from_commodity_noEff=pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
                                                       "c": pl.Utf8}),
        p_unitsize=Param(("p",), pl.DataFrame({"p": ["u"], "value": [1.0]})),
        p_flow_upper=Param(("p", "source", "sink", "d", "t"), pss_dt.with_columns(value=pl.lit(10.0))),
        p_slope=Param(("p", "d", "t"), dt.with_columns(p=pl.lit("u"), value=pl.lit(1.0))
                      .select("p", "d", "t", "value")),
        p_commodity_price=Param(("c", "d", "t"), dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(10.0))
                                .select("c", "d", "t", "value")),
    )


def _build(data: FlexData) -> Problem:
    pb = Problem()
    build_flextool(pb, data)
    return pb


def test_objective_weights_each_period_by_its_inflation_factor() -> None:
    sol = _build(_data(inflation={"p2020": 1.0, "p2025": 2.0})).solve(options=solver_options())
    assert sol.optimal
    assert float(sol.obj) == pytest.approx(150.0, rel=1e-9)


@pytest.mark.parametrize("field, name, csv", [
    ("p_inflation_op", "p_inflation_factor_operations_yearly", "p_inflation_factor_operations_yearly.csv"),
    ("p_period_share", "complete_period_share_of_year", "complete_period_share_of_year_calc.csv"),
])
def test_period_param_missing_a_period_is_an_error(field: str, name: str, csv: str) -> None:
    """The (d,) checks name the parameter, the missing period (not the covered one) and its solve_data file."""
    if field == "p_inflation_op":
        data = _data(inflation={"p2020": 1.0})
    else:
        data = _data(inflation={"p2020": 1.0, "p2025": 2.0}, drop_share=True)
    with pytest.raises(FlexToolConfigError) as err:
        _build(data)
    msg = str(err.value)
    assert f"({name})" in msg
    assert "'p2025'" in msg and "'p2020'" not in msg
    assert f"solve_data/{csv}" in msg


@pytest.mark.parametrize("field", ["p_timestep_weight", "p_step_duration"])
def test_step_param_missing_a_period_time_step_is_an_error(field: str) -> None:
    """The (d, t) check names the missing (period, time step), not only the period."""
    with pytest.raises(FlexToolConfigError) as err:
        _build(_data(inflation={"p2020": 1.0, "p2025": 2.0}, drop_step=field))
    msg = str(err.value)
    assert field in msg
    assert "(p2025, t0001)" in msg and "p2020" not in msg
