"""The capacity-margin penalty of a stochastic fan is weighted by the forecast's probability.

    + Σ_{g ∈ groupCapacityMargin, d ∈ D_inv} vq_capacity_margin[g,d] · gcs[g,d] · penalty_capacity_margin[g,d]
                                                          · inflation_op[d] · bw[d] · 1000
      bw = pd_branch_weight[d], 1 where it has no row

``stoch_two_period_invest.json`` ``recourse``: the fan reveals at p2035 into ``high`` (weight 1) and ``low``
(weight 3), and p2040 continues each branch.  Sibling-normalised: bw[p2035_high] = bw[p2040_high] = 1 / (1 + 3)
= 0.25 and bw[p2035_low] = bw[p2040_low] = 3 / 4 = 0.75, so each low slack costs three times its high twin.
Before the fix every forecast's slack was priced in full (equal coefficients).  The deterministic twin (the
fixture without the fan) prices p2035 / p2040 with bw = 1.

Coefficients are read from the unscaled LP (``build_flextool`` on a fresh Problem; autoscale runs at solve
time only).  Databases are built from JSON (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine._unit_db import b64, build, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
RECOURSE_JSON = REPO / "tests" / "fixtures" / "stoch_two_period_invest.json"
CHAIN = ["init", "base", "stoch", "recourse"]
CM_G = [["group", "cm_g", None], ["group__node", ["cm_g", "elec"], None]]
MARGIN = [["group", "cm_g", "capacity_margin", b64(1.0e5, "float")],
          ["group", "cm_g", "capacity_margin_method", b64("manual", "str")],
          ["group", "cm_g", "penalty_capacity_margin", b64(1.0, "float")]]
FAN = ("p2035_high", "p2035_low", "p2040_high", "p2040_low")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _deterministic_json(tmp_path_factory) -> Path:
    """The fixture without its stochastic fan (branches, group flag and invest method removed)."""
    data = json.loads(RECOURSE_JSON.read_text())
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if pv[2] not in ("stochastic_branches", "include_stochastics",
                                                 "stochastic_invest_method")]
    out = tmp_path_factory.mktemp("cm_det_json") / "stoch_two_period_invest_det.json"
    out.write_text(json.dumps(data))
    return out


def _flex_data(tmp_path_factory, fixture_json: Path, alt: str):
    url, scen = make_db(tmp_path_factory, alt=alt, values=MARGIN, base_chain=CHAIN, fixture_json=fixture_json,
                        entities=CM_G)
    steps = run(tmp_path_factory, url, scen)
    (step,) = steps.values()
    return step.flex_data


def _obj_coefs(fd) -> dict[str, float]:
    """``{period: objective coefficient}`` of ``vq_capacity_margin[cm_g, period]`` in the unscaled LP."""
    m = build(fd).canonicalise()
    out = {}
    for i, nm in enumerate(m.col_names):
        if nm.startswith("vq_capacity_margin["):
            g, d = nm[nm.index("[") + 1:-1].split(",")
            assert g == "cm_g", nm
            out[d] = float(m.col_obj[i])
    return out


def _by_d(param, keys: tuple[str, ...] = ("d",)) -> dict:
    df = utf8(param.frame)
    if keys == ("d",):
        return {r["d"]: float(r["value"]) for r in df.iter_rows(named=True)}
    return {r["d"]: float(r["value"]) for r in df.iter_rows(named=True) if r["g"] == "cm_g"}


def _unweighted(fd) -> dict[str, float]:
    """gcs · penalty · inflation_op · 1000 per period, from FlexData."""
    gcs = _by_d(fd.p_group_capacity_for_scaling, ("g", "d"))
    pen = _by_d(fd.pdGroup_penalty_capacity_margin, ("g", "d"))
    infl = _by_d(fd.p_inflation_op)
    return {d: gcs[d] * pen[d] * infl[d] * 1000.0 for d in gcs if d in pen and d in infl}


def test_fan_capacity_margin_penalty_weighted_by_branch_probability(tmp_path_factory) -> None:
    fd = _flex_data(tmp_path_factory, RECOURSE_JSON, "cm_stoch_fan")
    coefs = _obj_coefs(fd)
    assert sorted(coefs) == list(FAN)
    bw = {"p2035_high": 0.25, "p2035_low": 0.75, "p2040_high": 0.25, "p2040_low": 0.75}
    base = _unweighted(fd)
    for d in FAN:
        assert coefs[d] == pytest.approx(base[d] * bw[d], rel=1e-12), d
    for p in ("p2035", "p2040"):
        assert coefs[f"{p}_low"] / coefs[f"{p}_high"] == pytest.approx(3.0, rel=1e-12), p


def test_deterministic_capacity_margin_penalty_unweighted(tmp_path_factory) -> None:
    fd = _flex_data(tmp_path_factory, _deterministic_json(tmp_path_factory), "cm_stoch_det")
    coefs = _obj_coefs(fd)
    assert sorted(coefs) == ["p2035", "p2040"]
    base = _unweighted(fd)
    for d, c in coefs.items():
        assert c == pytest.approx(base[d], rel=1e-12), d
