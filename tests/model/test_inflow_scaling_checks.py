"""Inflow scaling methods refuse inputs they cannot honour (#231).

``scale_to_annual_flow`` / ``scale_in_proportion`` / ``scale_to_annual_and_peak_flow`` need ``annual_flow`` (and
``peak_inflow`` for the peak method) for every period in use, and an ``inflow`` profile to scale
(docs/how_to.md "How to set the demand in a node").  Before the check a missing or zero ingredient made the node's
inflow silently 0, and the peak method without an inflow profile turned the demand into a constant supply of
``annual_flow / 8760`` (+1027 MW for 9e6 MWh).  ``check_inflow_scaling_inputs`` now stops the solve with a
``FlexToolConfigError`` naming node, method, period and the missing input, and warns once per node where the
result is well defined but likely unintended.

Databases are ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal``: ``west`` is the
demand node, ``p2020`` the period; the solve uses 48 one-hour steps (period share of year 48/8760), over which
west's negative profile sums to -29103 MWh with peak -872 and smallest demand -423.
"""
from __future__ import annotations

import base64
import json

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db, run

NULL = ["bnVsbA==", None]  # explicit null: clears the base alternative's value
PEAK = "scale_to_annual_and_peak_flow"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _west(**params) -> list:
    out = []
    for name, val in params.items():
        if val is None:
            out.append(["node", "west", name, NULL])
        elif isinstance(val, str):
            out.append(["node", "west", name, b64(val, "str")])
        elif isinstance(val, dict):
            out.append(["node", "west", name, b64({"index_type": "str", "index_name": "period",
                                                   "data": [[k, v] for k, v in val.items()]}, "map")])
        else:
            out.append(["node", "west", name, b64(float(val), "float")])
    return out


def _run(tmp_path_factory, alt: str, values: list):
    url, scenario = make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN)
    return list(run(tmp_path_factory, url, scenario).values())[-1]


def _west_inflow(step) -> pl.Series:
    f = step.flex_data.p_inflow.frame
    return f.filter(pl.col("n").cast(pl.Utf8) == "west").sort("t")["value"]


@pytest.mark.parametrize("alt, values, needle", [
    ("is_annual_no_af", _west(inflow_method="scale_to_annual_flow"), "annual_flow is not given"),
    ("is_annual_no_inflow", _west(inflow_method="scale_to_annual_flow", annual_flow=9e6, inflow=None),
     "inflow sums to 0"),
    ("is_annual_af_map_misses_period", _west(inflow_method="scale_to_annual_flow", annual_flow={"p2025": 9e6}),
     "annual_flow is not given"),
    ("is_annual_af_negative", _west(inflow_method="scale_to_annual_flow", annual_flow=-9e6), "annual_flow is negative"),
    ("is_prop_no_af", _west(inflow_method="scale_in_proportion"), "annual_flow is not given"),
    ("is_prop_no_inflow", _west(inflow_method="scale_in_proportion", annual_flow=9e6, inflow=None),
     "inflow sums to 0"),
    ("is_peak_no_af", _west(inflow_method=PEAK, peak_inflow=-1100), "annual_flow is not given"),
    ("is_peak_no_peak", _west(inflow_method=PEAK, annual_flow=9e6), "peak_inflow is not given"),
    ("is_peak_no_inflow", _west(inflow_method=PEAK, annual_flow=9e6, peak_inflow=-1100, inflow=None),
     "no inflow profile"),
    ("is_peak_sign", _west(inflow_method=PEAK, annual_flow=9e6, peak_inflow=1100), "demand into supply"),
])
def test_unusable_scaling_input_is_rejected(tmp_path_factory, alt, values, needle) -> None:
    method = next(json.loads(base64.b64decode(v[0])) for _c, _e, p, v in values if p == "inflow_method")
    with pytest.raises(FlexToolConfigError, match="Inflow scaling cannot be applied") as err:
        _run(tmp_path_factory, alt, values)
    msg = str(err.value)
    assert needle in msg and f"node 'west', inflow_method '{method}', period 'p2020'" in msg, msg


def test_zero_annual_flow_is_zero_inflow(tmp_path_factory) -> None:
    """An explicit annual_flow of 0 asks for no flow in the period: accepted, inflow 0 (docs: inflow is scaled to
    sum to annual_flow)."""
    step = _run(tmp_path_factory, "is_annual_af_zero", _west(inflow_method="scale_to_annual_flow", annual_flow=0))
    assert _west_inflow(step).abs().max() == 0.0


def test_peak_scaling_hits_peak_and_annual_flow(tmp_path_factory, capsys) -> None:
    """Valid peak scaling (no message): the scaled peak is peak_inflow and the period's inflow sums to
    annual_flow x period share of year = 9e6 x 48/8760 = 49315.068... MWh (both by the method's definition)."""
    step = _run(tmp_path_factory, "is_peak_ok", _west(inflow_method=PEAK, annual_flow=9e6, peak_inflow=-1100))
    inflow = _west_inflow(step)
    assert inflow.min() == pytest.approx(-1100.0, rel=1e-9)
    assert inflow.sum() == pytest.approx(-9e6 * 48 / 8760, rel=1e-9)
    assert "Warning: node 'west'" not in capsys.readouterr().out


def test_peak_scaling_beyond_reach_warns(tmp_path_factory, capsys) -> None:
    """annual_flow 1e7 > |peak_inflow| x 8760 = 9.636e6: no profile with peak 1100 reaches it; the scaled inflow
    exceeds peak_inflow off-peak (warning, result still matches annual_flow)."""
    step = _run(tmp_path_factory, "is_peak_reach", _west(inflow_method=PEAK, annual_flow=1e7, peak_inflow=-1100))
    out = capsys.readouterr().out
    assert "node 'west'" in out and "above |peak_inflow| x 8760" in out, out
    assert _west_inflow(step).min() < -1100.0


def test_peak_scaling_direction_flip_warns(tmp_path_factory, capsys) -> None:
    """Hand derivation (docs/how_to.md formula): r = peak/peak_old = 1100/872, the profile scaled to the peak
    annualises to S = r x 29103 x 8760/48 = 6.70e6 MWh, N = 1100 x 8760 = 9.636e6, so for annual_flow 1e6
    c = (S - A)/(N - S) = 1.94 and the smallest-demand step becomes r(1+c) x 423 - 1100c = 1569 - 2135 < 0
    in demand terms: the node supplies there (warning)."""
    step = _run(tmp_path_factory, "is_peak_flip", _west(inflow_method=PEAK, annual_flow=1e6, peak_inflow=-1100))
    out = capsys.readouterr().out
    assert "node 'west'" in out and "changes direction" in out, out
    assert _west_inflow(step).max() > 0.0
