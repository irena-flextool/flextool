"""``reduce_unit_detail_method online`` folds the part-load curve of an indirect unit into a relaxation for every
weighted output per unit online ``W`` (``preprocess._reduce_detail``, "Why the folds are relaxations"): never more
fuel than the full model, the same at full output, the full model's input limits kept.

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
slope' = σ + s/W, F = slope'·Y.  Input limit per unit of capacity (kept): σ·W + s.

* W = 2: full 3050/7, 1300/7 + 450/7 = 250; slope' = 13/7 + 9/28 = 61/28, reduced 3050/7, 1525/7; limit 61/14.
* W = 1/2: full 650/7 + 450/7 = 1100/7, 325/7 + 450/7 = 775/7; slope' = 13/7 + 9/7 = 22/7, reduced 1100/7, 550/7;
  limit 13/14 + 9/14 = 11/7.
* W = 1: full 250, 650/7 + 450/7 = 1100/7; slope' = 5/2 (= σ + s, the earlier fold), reduced 250, 125; limit 5/2.

Multi-line ``piecewise_linear`` curve (``pw``) (0.3, 0.25), (0.6, 0.35), (1.0, 0.40): F(ℓ)/100 = max(12/7·ℓ + 24/35,
55/28·ℓ + 15/28) (online form; the efficiency rises, no origin line, no segment with a negative intercept).
Reduced: the origin line α₀ = min(1/η_i for ℓ_i ≤ W, F(W)/(100·W)); limit max_g(α_g·W + β_g) (kept).

* W = 2: full ℓ = 2: 100 × max(144/35, 125/28) = 3125/7; ℓ = 1: 250.  α₀ = min(4, 20/7, 5/2, (125/28)/2 = 125/56)
  = 125/56: reduced 200 × 125/56 = 3125/7, 100 × 125/56 = 3125/14; limit 125/28.
* W = 1/2: full ℓ = 1/2: 100 × max(54/35, 85/56) = 1080/7; ℓ = 1/4: 100 × max(39/35, 115/112) = 780/7.
  α₀ = min(4, (54/35)/(1/2) = 108/35) = 108/35: reduced 50 × 108/35 = 1080/7, 25 × 108/35 = 540/7; limit 54/35.
* W = 1: full 250, ℓ = 1/2: 1080/7.  α₀ = 1/max η = 5/2: reduced 250, 125; limit 5/2.

The earlier fold (slope + section, the best efficiency up to ℓ = 1) gave for W = 2 the reduced fuel 500 at full
output (more than the full model's 3050/7 or 3125/7: a tighter model) and for W = 1/2 125 (less than needed).
"""
from __future__ import annotations

from fractions import Fraction as Fr

import polars as pl
import pytest

from tests.engine._unit_db import utf8
from tests.model._piecewise_oracle_db import (
    balance_node, commodity_node, curve, make_oracle_db, merge, scalar, text, time_settings, unit,
)

pytestmark = pytest.mark.solver

RTOL = 1e-9
#: Objective per MW of fuel in one step: 4380 h weight × 20 CUR/MWh.
COST_PER_MW = 87600

CONVERSION = {
    "mle": {"conversion_method": text("min_load_efficiency"), "efficiency": scalar(0.4),
            "efficiency_at_min_load": scalar(0.25)},
    "pw1": {"conversion_method": text("piecewise_linear"),
            "piecewise_linear_efficiency": curve([(0.3, 0.25), (1.0, 0.40)])},
    "mle_neg": {"conversion_method": text("min_load_efficiency"), "efficiency": scalar(0.25),
                "efficiency_at_min_load": scalar(0.4)},
    "pw": {"conversion_method": text("piecewise_linear"),
           "piecewise_linear_efficiency": curve([(0.3, 0.25), (0.6, 0.35), (1.0, 0.40)])},
}

#: (curve family, W) -> (full fuel per step, reduced fuel per step, input limit per unit of capacity); module docstring.
EXPECT = {
    ("mle", 2.0): ([Fr(3050, 7), Fr(250)], [Fr(3050, 7), Fr(1525, 7)], Fr(61, 14)),
    ("mle", 0.5): ([Fr(1100, 7), Fr(775, 7)], [Fr(1100, 7), Fr(550, 7)], Fr(11, 7)),
    ("mle", 1.0): ([Fr(250), Fr(1100, 7)], [Fr(250), Fr(125)], Fr(5, 2)),
    ("pw", 2.0): ([Fr(3125, 7), Fr(250)], [Fr(3125, 7), Fr(3125, 14)], Fr(125, 28)),
    ("pw", 0.5): ([Fr(1080, 7), Fr(780, 7)], [Fr(1080, 7), Fr(540, 7)], Fr(54, 35)),
    ("pw", 1.0): ([Fr(250), Fr(1080, 7)], [Fr(250), Fr(125)], Fr(5, 2)),
}
FAMILY = {"mle": "mle", "pw1": "mle", "pw": "pw"}
DEMAND = (100.0, 50.0)

_RUNS: dict = {}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _case(kind: str, w: float, level: str | None, n_steps: int, rolling: bool,
          heat: bool = True) -> tuple[list, list]:
    extra = [["solve", "s1", "reduce_unit_detail_method", text(level)]] if level else []
    names = [f"t{i:02d}" for i in range(1, n_steps + 1)]
    elec = {t: -DEMAND[i % 2] for i, t in enumerate(names)}
    ents, vals = merge(
        time_settings(n_steps, ["p2030"], extra_solve=extra, rolling=(2.0, 2.0) if rolling else None),
        commodity_node("gas", "gas_node", 20.0), balance_node("elec", elec),
        *([balance_node("heat", {t: 0.0 for t in names})] if heat else []),
        unit("CHP", ["gas_node"], ["elec", "heat"] if heat else ["elec"], {
            **CONVERSION[kind], "virtual_unitsize": scalar(100.0), "existing": scalar(1.0),
            "startup_method": text("binary"), "startup_cost": scalar(0.0), "min_load": scalar(0.3)}))
    vals += [["unit__outputNode", ["CHP", "elec"], "conversion_flow_coeff", scalar(w)]]
    if heat:
        vals += [["unit__outputNode", ["CHP", "heat"], "conversion_flow_coeff", scalar(0.2)]]
    return ents, vals


def _run(tmp_path_factory, kind: str, w: float, level: str | None, rolling: bool = False,
         heat: bool = True) -> dict:
    from flextool.engine import run_chain_from_db

    name = (f"{kind}_w{str(w).replace('.', 'p')}_{level or 'none'}{'_roll' if rolling else ''}"
            f"{'' if heat else '_noheat'}")
    if name not in _RUNS:
        url, scen = make_oracle_db(tmp_path_factory, name,
                                   *_case(kind, w, level, 4 if rolling else 2, rolling, heat))
        _RUNS[name] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{name}_w"),
                                        keep_solutions=True)
    steps = _RUNS[name]
    assert all(s.solution is not None for s in steps.values()), list(steps)
    return steps


def _mw(step, source: str, sink: str) -> list[float]:
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == "CHP") & (pl.col("source") == source) & (pl.col("sink") == sink)).sort("d", "t"))
    return [v * 100.0 for v in rows["value"].to_list()]


def _slack(step) -> list[float]:
    return utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == "elec")["value"].to_list()


def _cells(param) -> pl.DataFrame:
    return utf8(param.frame).filter(pl.col("p") == "CHP").sort(pl.all())


@pytest.mark.parametrize("w", [2.0, 0.5, 1.0], ids=["W2", "W0.5", "W1"])
@pytest.mark.parametrize("kind", ["mle", "pw1", "pw"])
def test_reduced_fold_is_a_relaxation(tmp_path_factory, kind, w) -> None:
    """Full and reduced serve the demand; the fuel is the hand value of each (module docstring), the reduced never
    above the full and equal at full output (t01); the reduced objective is at most the full one; the input limits
    (and the arc bounds built from them) are the full model's."""
    full_fuel, red_fuel, cap = EXPECT[(FAMILY[kind], w)]
    full = next(iter(_run(tmp_path_factory, kind, w, None).values()))
    red = next(iter(_run(tmp_path_factory, kind, w, "online").values()))
    for s, fuel in ((full, full_fuel), (red, red_fuel)):
        assert _mw(s, "CHP", "elec") == pytest.approx(list(DEMAND), rel=RTOL)
        assert _slack(s) == pytest.approx([0.0, 0.0], abs=1e-9)
        assert _mw(s, "gas_node", "CHP") == pytest.approx([float(f) for f in fuel], rel=RTOL)
        assert float(s.solution.obj) == pytest.approx(float(COST_PER_MW * sum(fuel)), rel=RTOL)
    assert all(r <= f for r, f in zip(red_fuel, full_fuel)) and red_fuel[0] == full_fuel[0]
    assert float(red.solution.obj) <= float(full.solution.obj) * (1 + RTOL)
    assert red.flex_data.process_online is None
    assert _cells(red.flex_data.p_indirect_input_cap).equals(_cells(full.flex_data.p_indirect_input_cap))
    assert _cells(red.flex_data.p_flow_upper).equals(_cells(full.flex_data.p_flow_upper))
    assert _cells(red.flex_data.p_indirect_input_cap)["value"].to_list() == pytest.approx([float(cap)] * 2, rel=RTOL)


@pytest.mark.parametrize("kind", ["mle", "pw"])
def test_w1_fold_is_the_earlier_fold(tmp_path_factory, kind) -> None:
    """W = 1 (availability 1): the folded slope is slope + section bit for bit (min_load_efficiency); the origin line
    is 1/max η = 5/2 exactly (piecewise_linear)."""
    full = next(iter(_run(tmp_path_factory, kind, 1.0, None).values())).flex_data
    red = next(iter(_run(tmp_path_factory, kind, 1.0, "online").values())).flex_data
    if kind == "mle":
        both = _cells(full.p_slope).join(_cells(full.p_section), on=["p", "d", "t"], suffix="_sec")
        expected = (both["value"] + both["value_sec"]).to_list()
        assert _cells(red.p_slope)["value"].to_list() == expected
        assert _cells(red.p_section)["value"].to_list() == [0.0, 0.0]
    else:
        lines = _cells(red.p_piecewise_slope).join(_cells(red.p_piecewise_intercept), on=["p", "seg", "d"],
                                                   suffix="_icpt")
        assert lines.select("seg", "value", "value_icpt").rows() == [("0", 2.5, 0.0)]


@pytest.mark.parametrize("kind", ["mle", "pw"])
def test_rolling_reduced_fold_is_a_relaxation(tmp_path_factory, kind) -> None:
    """A rolling solve (4 steps, horizon = jump = 2 h; demand 100, 50, 100, 50) folds every roll as the single
    solve with W = 2: per roll the fuel is the hand value of the W = 2 case, the reduced at most the full one, and
    the reduced / full objective ratio of a roll is its fuel ratio (the cost is fuel only)."""
    full_fuel, red_fuel, _cap = EXPECT[(FAMILY[kind], 2.0)]
    full = _run(tmp_path_factory, kind, 2.0, None, rolling=True)
    red = _run(tmp_path_factory, kind, 2.0, "online", rolling=True)
    assert list(full) == list(red) and len(full) == 2, list(full)
    for name in full:
        f, r = full[name], red[name]
        assert r.flex_data.process_online is None
        for s, fuel in ((f, full_fuel), (r, red_fuel)):
            assert _mw(s, "CHP", "elec") == pytest.approx(list(DEMAND), rel=RTOL), name
            assert _slack(s) == pytest.approx([0.0, 0.0], abs=1e-9), name
            assert _mw(s, "gas_node", "CHP") == pytest.approx([float(x) for x in fuel], rel=RTOL), name
        assert float(r.solution.obj) / float(f.solution.obj) == pytest.approx(
            float(sum(red_fuel) / sum(full_fuel)), rel=1e-7), name
        assert _cells(r.flex_data.p_indirect_input_cap).equals(_cells(f.flex_data.p_indirect_input_cap)), name


@pytest.mark.parametrize("heat, red_fuel", [(False, [Fr(250), Fr(125)]), (True, [Fr(0), Fr(0)])],
                         ids=["one_output", "two_outputs"])
def test_section_below_zero_folds_from_the_min_load(tmp_path_factory, heat, red_fuel) -> None:
    """``efficiency_at_min_load`` 0.4 above ``efficiency`` 0.25 (min_load 0.3): s = 0.3 × (1/0.4 − 1/0.25) / 0.7
    = −9/14 < 0, σ = 4 + 9/14 = 65/14.  The full model burns σ·Y + s·100: 400 at 100 MW, 3250/14 − 900/14 = 1175/7
    at 50 MW (elec weight 1, W = 1).  More units online burn less, so the bound is the minimum load:
    u ≤ Σ out / 0.3 ≤ Y / (0.3 · conv_min).  One output (a direct unit, conv_min 1):
    slope' = 65/14 − (9/14)/0.3 = 5/2 = 1/0.4, reduced 250 and 125.  With heat (weight 0.2, demand 0) conv_min = 0.2: 65/14 − (9/14)/0.06 = −85/14 < 0,
    clamped to 0: no fuel.  The earlier fold (σ + s = 4: 400 and 200) burnt more than the full model at 50 MW."""
    full_fuel = [Fr(400), Fr(1175, 7)]
    full = next(iter(_run(tmp_path_factory, "mle_neg", 1.0, None, heat=heat).values()))
    red = next(iter(_run(tmp_path_factory, "mle_neg", 1.0, "online", heat=heat).values()))
    # One output makes CHP a direct unit (its flow is the arc gas_node -> elec); the cost is fuel only, so the
    # objective / COST_PER_MW is the fuel summed over the steps.
    src = "CHP" if heat else "gas_node"
    for s, fuel in ((full, full_fuel), (red, red_fuel)):
        assert _mw(s, src, "elec") == pytest.approx(list(DEMAND), rel=RTOL)
        assert _slack(s) == pytest.approx([0.0, 0.0], abs=1e-9)
        assert float(s.solution.obj) == pytest.approx(float(COST_PER_MW * sum(fuel)), rel=RTOL, abs=1e-6)
    assert float(red.solution.obj) <= float(full.solution.obj)
