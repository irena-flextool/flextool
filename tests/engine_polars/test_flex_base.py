"""Stage-2: flexpy ``base`` model on real flextool-generated data
(``tests/data/work_base/``) — parity vs flextool's recorded ``v_obj``,
plus parameter perturbations validated against a closed-form
re-evaluation."""

from dataclasses import replace
from pathlib import Path

import pytest
import polars as pl

from polar_high import Problem, Param
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_base"


def _scale(p: Param, factor: float) -> Param:
    return Param(p.dims, p.frame.with_columns(value=pl.col("value") * factor))


def _closed_form(d) -> float:
    df = (d.p_inflow.frame.rename({"value": "inflow"})
            .join(d.p_penalty_up.frame.rename({"value": "pen_up"}), on=["n","d","t"])
            .join(d.p_penalty_down.frame.rename({"value": "pen_dn"}), on=["n","d","t"])
            .join(d.p_step_duration.frame.rename({"value": "dur"}), on=["d","t"])
            .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}), on=["d","t"])
            .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
            .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
         )
    df = df.with_columns(
        slack_up   = pl.max_horizontal(-pl.col("inflow"), pl.lit(0.0)),
        slack_down = pl.max_horizontal( pl.col("inflow"), pl.lit(0.0)),
    )
    return float((
        (df["slack_up"] * df["pen_up"] + df["slack_down"] * df["pen_dn"])
        * df["dur"] * df["rpcw"] * df["infl"] / df["psh"]
    ).sum())


def _solve(data) -> "Solution":
    pb = Problem()
    build_flextool(pb, data)
    return pb.solve()


@pytest.fixture(scope="module")
def base_data():
    return load_flextool(WORK)


def test_base_parity(base_data):
    """flexpy's obj must match flextool's recorded v_obj within 1e-6 rel."""
    sol = _solve(base_data)
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6


@pytest.mark.parametrize("perturbation", [
    pytest.param(("p_penalty_up",   2.0),  id="penalty_up_x2"),
    pytest.param(("p_inflow",      10.0),  id="inflow_x10"),
    pytest.param(("p_inflow",       0.0),  id="inflow_zero"),
    pytest.param(("p_penalty_down", 0.5),  id="penalty_down_half"),
])
def test_base_closed_form(base_data, perturbation):
    """flexpy must agree with the closed-form re-evaluation for any
    multiplicative perturbation of the input parameters."""
    field, factor = perturbation
    data = replace(base_data, **{field: _scale(getattr(base_data, field), factor)})
    sol = _solve(data)
    expected = _closed_form(data)
    assert sol.optimal
    assert abs(sol.obj - expected) / max(1.0, abs(expected) + 1.0) < 1e-6
