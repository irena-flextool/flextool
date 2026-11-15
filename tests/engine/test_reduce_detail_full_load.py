"""``reduce_unit_detail_method online`` runs every ``min_load_efficiency`` and ``piecewise_linear`` unit as a
``constant_efficiency`` unit at its full-load efficiency (``preprocess._reduce_detail``, "Full-load efficiency"):
the fuel is the weighted output Y times slope' = (full model's fuel of one unit online at full output) / W, W the
weighted output that unit then gives, and the input limits are the full model's.

The case (``tests/model/_piecewise_oracle_db.py``, built from the schema of ``tests.json``): unit ``CHP``, gas
(20 CUR/MWh, input ``conversion_flow_coeff`` 1) -> elec (``conversion_flow_coeff`` c) + heat (0.2), one 100-MW unit
(``virtual_unitsize`` 100, ``existing`` 1), binary start-ups at no cost, ``min_load`` 0.3; elec demand 100 MW in t01
and 50 MW in t02, heat demand 0, one period p2030 (weight 8760 / 2 = 4380 per step, so the objective is
87600 × the fuel MW summed over the steps).  ``W`` fills the outputs in descending ``conversion_flow_coeff`` order up
to the sum cap 1: W = c for c ∈ {2, 0.5, 1} (heat 0.2 is below each).  The weighted output is Y = c × elec: 100 c
and 50 c MW on the 1 unit online (load share ℓ = Y / 100).

Hand values (exact fractions; F = fuel MW).

``min_load_efficiency`` (``mle``) with ``efficiency`` 0.4 and ``efficiency_at_min_load`` 0.25, and the single-line
``piecewise_linear`` curve (0.3, 0.25), (1.0, 0.40) (``pw1``: one line, handled as min_load_efficiency): section
s = 0.3 × (1/0.25 − 1/0.4) / (1 − 0.3) = 9/14, slope σ = 1/0.4 − 9/14 = 13/7.  Full: F = σ·Y + s·100.  Reduced:
slope' = σ + s/W, F = slope'·Y.  Input limit per unit of capacity (the full model's): σ·W + s.

* W = 2: full 3050/7, 1300/7 + 450/7 = 250; slope' = 13/7 + 9/28 = 61/28, reduced 3050/7, 1525/7; limit 61/14.
* W = 1/2: full 650/7 + 450/7 = 1100/7, 325/7 + 450/7 = 775/7; slope' = 13/7 + 9/7 = 22/7, reduced 1100/7, 550/7;
  limit 13/14 + 9/14 = 11/7.
* W = 1: full 250, 650/7 + 450/7 = 1100/7; slope' = 5/2 = 1/0.4, reduced 250, 125; limit 5/2.

Multi-line ``piecewise_linear`` curve (``pw``) (0.3, 0.25), (0.6, 0.35), (1.0, 0.40): F(ℓ)/100 = max(12/7·ℓ + 24/35,
55/28·ℓ + 15/28) (online form; the efficiency rises, no origin line).  Reduced: slope' = 1/η(1) = 5/2 for W = 1, else
F(W)/(100·W); limit F(W)/100.

* W = 2: full ℓ = 2: 100 × max(144/35, 125/28) = 3125/7; ℓ = 1: 250.  slope' = (125/28)/2 = 125/56: reduced
  200 × 125/56 = 3125/7, 100 × 125/56 = 3125/14; limit 125/28.
* W = 1/2: full ℓ = 1/2: 100 × max(54/35, 85/56) = 1080/7; ℓ = 1/4: 100 × max(39/35, 115/112) = 780/7.
  slope' = (54/35)/(1/2) = 108/35: reduced 50 × 108/35 = 1080/7, 25 × 108/35 = 540/7; limit 54/35.
* W = 1: full 250, ℓ = 1/2: 1080/7.  slope' = 5/2: reduced 250, 125; limit 5/2.

More efficient at part load (the full-load efficiency is the poorer one, and the reduced unit burns more than the
full model at part load):

* ``mle_neg``: ``efficiency`` 0.25, ``efficiency_at_min_load`` 0.4: s = 0.3 × (1/0.4 − 1/0.25) / 0.7 = −9/14,
  σ = 4 + 9/14 = 65/14.  Full (one unit online): 400 at 100 MW, 3250/14 − 900/14 = 1175/7 at 50 MW.  W = 1 (elec 1
  fills the sum cap; with or without heat): slope' = σ + s = 4 = 1/0.25, reduced 400 and 200.
* ``pw_peak``: (0.3, 0.30), (0.7, 0.40), (1.0, 0.38), best at load 0.7.  Fuel per 100 MW: 100 at ℓ 0.3, 175 at 0.7,
  5000/19 at 1.  Full: 5000/19 at 100 MW; at 50 MW (ℓ = 1/2, the first segment) 100 + 75/2 = 275/2.  W = 1:
  slope' = 1/0.38 = 50/19, reduced 5000/19 and 2500/19.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import build, utf8
from tests.model._piecewise_oracle_db import (
    balance_node, commodity_node, curve, make_oracle_db, merge, scalar, text, time_settings, unit,
)

pytestmark = pytest.mark.solver

RTOL = 1e-9
#: Objective per MW of fuel in one step: 4380 h weight × 20 CUR/MWh.
COST_PER_MW = 87600

MLE = {"conversion_method": text("min_load_efficiency"), "efficiency": scalar(0.4),
       "efficiency_at_min_load": scalar(0.25)}
CONVERSION = {
    "mle": MLE,
    "pw1": {"conversion_method": text("piecewise_linear"),
            "piecewise_linear_efficiency": curve([(0.3, 0.25), (1.0, 0.40)])},
    "mle_neg": {"conversion_method": text("min_load_efficiency"), "efficiency": scalar(0.25),
                "efficiency_at_min_load": scalar(0.4)},
    "pw": {"conversion_method": text("piecewise_linear"),
           "piecewise_linear_efficiency": curve([(0.3, 0.25), (0.6, 0.35), (1.0, 0.40)])},
    "pw_peak": {"conversion_method": text("piecewise_linear"),
                "piecewise_linear_efficiency": curve([(0.3, 0.30), (0.7, 0.40), (1.0, 0.38)])},
}

#: (curve family, W) -> (full fuel per step, reduced fuel per step, input limit per unit of capacity); module docstring.
EXPECT = {
    ("mle", 2.0): ([Fr(3050, 7), Fr(250)], [Fr(3050, 7), Fr(1525, 7)], Fr(61, 14)),
    ("mle", 0.5): ([Fr(1100, 7), Fr(775, 7)], [Fr(1100, 7), Fr(550, 7)], Fr(11, 7)),
    ("mle", 1.0): ([Fr(250), Fr(1100, 7)], [Fr(250), Fr(125)], Fr(5, 2)),
    ("pw", 2.0): ([Fr(3125, 7), Fr(250)], [Fr(3125, 7), Fr(3125, 14)], Fr(125, 28)),
    ("pw", 0.5): ([Fr(1080, 7), Fr(780, 7)], [Fr(1080, 7), Fr(540, 7)], Fr(54, 35)),
    ("pw", 1.0): ([Fr(250), Fr(1080, 7)], [Fr(250), Fr(125)], Fr(5, 2)),
    ("mle_neg", 1.0): ([Fr(400), Fr(1175, 7)], [Fr(400), Fr(200)], Fr(4)),
    ("pw_peak", 1.0): ([Fr(5000, 19), Fr(275, 2)], [Fr(5000, 19), Fr(2500, 19)], Fr(50, 19)),
}
FAMILY = {"mle": "mle", "pw1": "mle", "pw": "pw", "mle_neg": "mle_neg", "pw_peak": "pw_peak"}
DEMAND = (100.0, 50.0)

_RUNS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _case(conversion: dict, w: float, level: str | None, n_steps: int, rolling: bool, heat: bool,
          min_load: float) -> tuple[list, list]:
    extra = [["solve", "s1", "reduce_unit_detail_method", text(level)]] if level else []
    names = [f"t{i:02d}" for i in range(1, n_steps + 1)]
    elec = {t: -DEMAND[i % 2] for i, t in enumerate(names)}
    ents, vals = merge(
        time_settings(n_steps, ["p2030"], extra_solve=extra, rolling=(2.0, 2.0) if rolling else None),
        commodity_node("gas", "gas_node", 20.0), balance_node("elec", elec),
        *([balance_node("heat", {t: 0.0 for t in names})] if heat else []),
        unit("CHP", ["gas_node"], ["elec", "heat"] if heat else ["elec"], {
            **conversion, "virtual_unitsize": scalar(100.0), "existing": scalar(1.0),
            "startup_method": text("binary"), "startup_cost": scalar(0.0), "min_load": scalar(min_load)}))
    vals += [["unit__outputNode", ["CHP", "elec"], "conversion_flow_coeff", scalar(w)]]
    if heat:
        vals += [["unit__outputNode", ["CHP", "heat"], "conversion_flow_coeff", scalar(0.2)]]
    return ents, vals


def _run(tmp_path_factory, name: str, conversion: dict, w: float, level: str | None, *, rolling: bool = False,
         heat: bool = True, min_load: float = 0.3) -> dict:
    from flextool.engine import run_chain_from_db

    name = f"{name}_w{str(w).replace('.', 'p')}_{level or 'none'}{'_roll' if rolling else ''}{'' if heat else '_1out'}"
    if name not in _RUNS:
        url, scen = make_oracle_db(tmp_path_factory, name,
                                   *_case(conversion, w, level, 4 if rolling else 2, rolling, heat, min_load))
        _RUNS[name] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{name}_w"),
                                        keep_solutions=True)
    steps = _RUNS[name]
    assert all(s.solution is not None for s in steps.values()), list(steps)
    return steps


def _first(steps: dict):
    return next(iter(steps.values()))


def _mw(step, source: str, sink: str) -> list[float]:
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == "CHP") & (pl.col("source") == source) & (pl.col("sink") == sink)).sort("d", "t"))
    return [v * 100.0 for v in rows["value"].to_list()]


def _slack(step) -> list[float]:
    return utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == "elec")["value"].to_list()


def _cells(param) -> pl.DataFrame:
    return utf8(param.frame).filter(pl.col("p") == "CHP").sort(pl.all())


def _check_fuel(step, fuel: list, heat: bool) -> None:
    """Demand served without slack, the fuel and the objective of the hand values (the cost is fuel only).  With one
    output CHP is a direct unit: its flow is the arc gas_node -> elec."""
    if heat:
        assert _mw(step, "CHP", "elec") == pytest.approx(list(DEMAND), rel=RTOL)
        assert _mw(step, "gas_node", "CHP") == pytest.approx([float(f) for f in fuel], rel=RTOL)
    else:
        assert _mw(step, "gas_node", "elec") == pytest.approx(list(DEMAND), rel=RTOL)
    assert _slack(step) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert float(step.solution.obj) == pytest.approx(float(COST_PER_MW * sum(fuel)), rel=RTOL)


@pytest.mark.parametrize("kind,w", [(k, w) for k in ("mle", "pw1", "pw") for w in (2.0, 0.5, 1.0)]
                         + [("mle_neg", 1.0), ("pw_peak", 1.0)])
def test_reduced_unit_runs_at_full_load_efficiency(tmp_path_factory, kind, w) -> None:
    """Full and reduced serve the demand; the fuel is the hand value of each (module docstring), the same at full
    output (t01); the reduced unit has no online variables, no part-load section and no curve lines; the input
    limits (and the arc bounds built from them) are the full model's."""
    full_fuel, red_fuel, cap = EXPECT[(FAMILY[kind], w)]
    full = _first(_run(tmp_path_factory, kind, CONVERSION[kind], w, None))
    red = _first(_run(tmp_path_factory, kind, CONVERSION[kind], w, "online"))
    _check_fuel(full, full_fuel, True)
    _check_fuel(red, red_fuel, True)
    assert red_fuel[0] == full_fuel[0]
    fd = red.flex_data
    assert fd.process_online is None and fd.process_min_load_eff is None and fd.process_piecewise is None
    assert fd.p_piecewise_slope is None
    assert _cells(fd.p_indirect_input_cap).equals(_cells(full.flex_data.p_indirect_input_cap))
    assert _cells(fd.p_flow_upper).equals(_cells(full.flex_data.p_flow_upper))
    assert _cells(fd.p_indirect_input_cap)["value"].to_list() == pytest.approx([float(cap)] * 2, rel=RTOL)


def test_falling_efficiency_one_output(tmp_path_factory) -> None:
    """``mle_neg`` with elec only (a direct unit): runs at 1/0.25, reduced 400 and 200 (full 400 and 1175/7)."""
    full_fuel, red_fuel, _cap = EXPECT[("mle_neg", 1.0)]
    _check_fuel(_first(_run(tmp_path_factory, "mle_neg", CONVERSION["mle_neg"], 1.0, None, heat=False)),
                full_fuel, False)
    red = _first(_run(tmp_path_factory, "mle_neg", CONVERSION["mle_neg"], 1.0, "online", heat=False))
    _check_fuel(red, red_fuel, False)
    # slope + section (σ = 1/0.25 − s, then + s), one ulp from 1/0.25 for these values.
    assert _cells(red.flex_data.p_slope)["value"].to_list() == pytest.approx([4.0, 4.0], rel=1e-15)


@pytest.mark.parametrize("kind", ["mle", "pw"])
def test_rolling_reduced_unit_runs_at_full_load_efficiency(tmp_path_factory, kind) -> None:
    """A rolling solve (4 steps, horizon = jump = 2 h; demand 100, 50, 100, 50) gives every roll the W = 2 hand
    values: per roll the fuel of the single solve and the full model's input limits."""
    full_fuel, red_fuel, _cap = EXPECT[(FAMILY[kind], 2.0)]
    full = _run(tmp_path_factory, kind, CONVERSION[kind], 2.0, None, rolling=True)
    red = _run(tmp_path_factory, kind, CONVERSION[kind], 2.0, "online", rolling=True)
    assert list(full) == list(red) and len(full) == 2, list(full)
    for name in full:
        f, r = full[name], red[name]
        assert r.flex_data.process_online is None, name
        for s, fuel in ((f, full_fuel), (r, red_fuel)):
            assert _mw(s, "CHP", "elec") == pytest.approx(list(DEMAND), rel=RTOL), name
            assert _slack(s) == pytest.approx([0.0, 0.0], abs=1e-9), name
            assert _mw(s, "gas_node", "CHP") == pytest.approx([float(x) for x in fuel], rel=RTOL), name
        assert _cells(r.flex_data.p_indirect_input_cap).equals(_cells(f.flex_data.p_indirect_input_cap)), name


# ---------------------------------------------------------------------------
# The reduced unit is the constant_efficiency unit at its full-load efficiency, LP byte for byte.
#
# Dyadic curves so that the full-load slope and the input limit are the same floats on both sides: min_load 0.5
# (pw: the first load share), efficiency 0.5, efficiency_at_min_load 0.4: s = 0.5 × (2.5 − 2) × 2 = 1/2, σ = 3/2.
# W = 1: slope' = 2 = 1/0.5; W = 2 (elec coefficient 2): slope' = 3/2 + 1/4 = 7/4 = 1/(4/7), limit 3/2·2 + 1/2 =
# 7/2 = 2·7/4.  Falling (efficiency 0.4, at min load 0.5): s = −1/2, σ = 3, slope' = 5/2 = 1/0.4 (W = 1, also with
# heat).  Curves (W = 1, slope' = 1/η(1)): one line {0.5: 0.4, 1: 0.5} (the mle W = 1 case); rising
# {0.25: 0.25, 0.5: 0.4, 1: 0.5}: lines 1·ℓ + 3/4 and 3/2·ℓ + 1/2, F(1) = 2 = 1/0.5; peaking below full load
# {0.25: 0.4, 0.5: 0.5, 1: 0.4}: lines 3/2·ℓ + 1/4 and 3·ℓ − 1/2, F(1) = 5/2 = 1/0.4.

def _mle(eff: float, eff_min: float) -> dict:
    return {"conversion_method": text("min_load_efficiency"), "efficiency": scalar(eff),
            "efficiency_at_min_load": scalar(eff_min)}


def _pw(points: list[tuple[float, float]]) -> dict:
    return {"conversion_method": text("piecewise_linear"), "piecewise_linear_efficiency": curve(points)}


#: name -> (conversion, W, heat output, min_load, the constant_efficiency twin's efficiency, slope' = 1/efficiency).
TWINS = {
    "mle_w1": (_mle(0.5, 0.4), 1.0, True, 0.5, 0.5, 2.0),
    "mle_w2": (_mle(0.5, 0.4), 2.0, True, 0.5, 4.0 / 7.0, 1.75),
    "mle_falling_one_output": (_mle(0.4, 0.5), 1.0, False, 0.5, 0.4, 2.5),
    "mle_falling_two_outputs": (_mle(0.4, 0.5), 1.0, True, 0.5, 0.4, 2.5),
    "pw_single_line": (_pw([(0.5, 0.4), (1.0, 0.5)]), 1.0, True, 0.5, 0.5, 2.0),
    "pw_rising": (_pw([(0.25, 0.25), (0.5, 0.4), (1.0, 0.5)]), 1.0, True, 0.25, 0.5, 2.0),
    "pw_peak_below_full_load": (_pw([(0.25, 0.4), (0.5, 0.5), (1.0, 0.4)]), 1.0, True, 0.25, 0.4, 2.5),
}


def _mps(pb, path: Path) -> bytes:
    pb.write_mps(path)
    return path.read_bytes()


@pytest.mark.parametrize("rolling", [False, True], ids=["single", "rolling"])
@pytest.mark.parametrize("twin", list(TWINS))
def test_reduced_unit_is_the_constant_efficiency_unit(tmp_path_factory, tmp_path, twin, rolling) -> None:
    """Under ``online`` the unit gives the LP of the same unit authored as ``constant_efficiency`` at its full-load
    efficiency (the written MPS byte for byte, every sub-solve), the same objective, and its slope is the hand value."""
    conversion, w, heat, min_load, eff, slope = TWINS[twin]
    red = _run(tmp_path_factory, twin, conversion, w, "online", rolling=rolling, heat=heat, min_load=min_load)
    ref = _run(tmp_path_factory, f"{twin}_const", {"conversion_method": text("constant_efficiency"),
                                                   "efficiency": scalar(eff)},
               w, "online", rolling=rolling, heat=heat, min_load=min_load)
    assert list(red) == list(ref) and len(red) == (2 if rolling else 1), list(red)
    for name in red:
        fd = red[name].flex_data
        assert set(_cells(fd.p_slope)["value"].to_list()) == {slope}, name
        assert fd.process_piecewise is None and fd.process_min_load_eff is None, name
        safe = name.replace("/", "_")
        assert _mps(build(fd), tmp_path / f"{safe}_a.mps") == _mps(build(ref[name].flex_data),
                                                                    tmp_path / f"{safe}_b.mps"), name
        assert float(red[name].solution.obj) == float(ref[name].solution.obj), name
