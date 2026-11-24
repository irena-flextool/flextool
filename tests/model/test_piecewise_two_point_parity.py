"""A one-line ``piecewise_linear`` unit is a ``min_load_efficiency`` unit (``specs/piecewise_efficiency_final.md``
§2.2, §3.2; tests T1, T1-R, T4 overwrite part, T8 part).

A two-point curve ``{ℓ₀: η₀, 1: η₁}`` with ``η₀ < η₁`` and ``min_load = ℓ₀`` is the ``min_load_efficiency`` curve
through ``(min_load, efficiency_at_min_load) = (ℓ₀, η₀)`` and ``(1, efficiency) = (1, η₁)``: one line, input =
α·output + β·online with

    β = ℓ₀ · (1/η₀ − 1/η₁) · (1/(1 − ℓ₀)),   α = 1/η₁ − β

the same float operations as ``_derived._mle_section`` / ``_mle_slope``.  So the two units give the same model,
bit for bit: the same ``p_slope`` / ``p_section`` / ``process_min_load_eff``, the same capacity parameters
derived from them (``p_indirect_input_cap``, ``p_flow_upper``) and the same LP (every row, column, bound and
coefficient; the written MPS byte for byte), in

* a single solve: a direct unit with linear and with binary start-ups, a direct unit with
  ``conversion_flow_coeff`` 2 on its input (the slope · k_sink / k_src and section / k_src fold), an indirect unit
  with two outputs (``conversion_flow_coeff`` 1 and 2; its start-ups are the automatic linear ones of a rising
  curve), and the direct unit under ``reduce_unit_detail_method online``;
* a rolling solve (the rolls are synthetic sub-solves whose slope and ``p_flow_max`` come from the per-roll emit,
  overwritten at the load boundary) and a stochastic solve (branch periods), each with a direct and an indirect
  unit.

A curve given once and the same curve per period (the stochastic branch periods inherit the anchor's) give the
same LP in a single, a rolling and a stochastic solve (CLAUDE.md, a parameter with several shapes).  A solve
period without a curve stops the run (E11).  A curve with two or more lines gets its lines and capacity
parameters at the load boundary (checked here against the hand-derived O1 lines).

Databases are built from JSON fixtures (``tests/fixtures/tests.json``, ``stochastics.json``), never a checked-in
``.sqlite``.  Every run has ``FLEXTOOL_AUTOSCALE_STRICT=1``.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import RunCache, b64, build, make_db, utf8

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
SOLVE = "y2020_2day_dispatch"
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Values


def scalar(v: float) -> list:
    return b64(float(v), "float")


def text(v: str) -> list:
    return b64(v, "str")


def curve(points: list[tuple[float, float]]) -> list:
    """A curve (map over load shares, float keys)."""
    return b64({"index_type": "float", "data": [[k, v] for k, v in points]}, "map")


def curve_per_period(per: dict[str, list[tuple[float, float]]]) -> list:
    """A curve per period (map over periods of maps over load shares)."""
    return b64({"index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "float", "data": [[k, v] for k, v in pts]}] for d, pts in per.items()]},
        "map")


def mle(unit: str, l0: float, e0: float, e1: float, *, startup: str | None = None) -> list:
    """``unit`` as a min_load_efficiency unit through (l0, e0) and (1, e1), min_load l0."""
    out = [["unit", unit, "conversion_method", text("min_load_efficiency")],
           ["unit", unit, "efficiency", scalar(e1)], ["unit", unit, "efficiency_at_min_load", scalar(e0)],
           ["unit", unit, "min_load", scalar(l0)]]
    return out + ([["unit", unit, "startup_method", text(startup)]] if startup else [])


def pw(unit: str, l0: float, e0: float, e1: float, *, startup: str | None = None, value: list | None = None) -> list:
    """``unit`` as a piecewise_linear unit with the curve {l0: e0, 1: e1} (or ``value``), min_load l0."""
    out = [["unit", unit, "conversion_method", text("piecewise_linear")],
           ["unit", unit, "piecewise_linear_efficiency", value or curve([(l0, e0), (1.0, e1)])],
           ["unit", unit, "min_load", scalar(l0)]]
    return out + ([["unit", unit, "startup_method", text(startup)]] if startup else [])


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text(encoding="utf-8"))["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


# Direct: coal_plant (coal_market -> west) with the coal_min_load alternative's linear start-ups and start-up cost.
PLANT = ["init", "west", "coal", "coal_min_load"]
PLANT_CURVE = (0.4, 0.30, 0.42)
# Indirect: coal_chp (coal_market -> west + heat), outputs weighted 1 and 2.
CHP = ["init", "west", "coal_chp", "heat"]
CHP_CURVE = (0.3, 0.6, 0.9)
CHP_OUTPUTS = [["unit__outputNode", ["coal_chp", "west"], "conversion_flow_coeff", scalar(1.0)],
               ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", scalar(2.0)]]
ROLLING = [["solve", SOLVE, "solve_mode", text("rolling_window")],
           ["solve", SOLVE, "rolling_solve_horizon", scalar(24.0)],
           ["solve", SOLVE, "rolling_solve_jump", scalar(12.0)]]
# Stochastic: 2_day_stochastic_dispatch; coal_plant direct, gas_plant given a second output (heat_s, weight 2).
STOCH = _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch")
STOCH_CURVE = (0.4, 0.5, 0.9)
STOCH_ENTITIES = [["node", "heat_s", None], ["unit__outputNode", ["gas_plant", "heat_s"], None]]
STOCH_HEAT = [["node", "heat_s", "node_type", text("balance")], ["node", "heat_s", "inflow", scalar(-100.0)],
              ["node", "heat_s", "penalty_up", scalar(5000.0)], ["node", "heat_s", "penalty_down", scalar(5000.0)],
              ["unit__outputNode", ["gas_plant", "heat_s"], "conversion_flow_coeff", scalar(2.0)]]

CASES: dict[str, dict] = {
    "direct_linear": dict(chain=PLANT, unit="coal_plant", curve=PLANT_CURVE, extra=[]),
    "direct_binary": dict(chain=PLANT, unit="coal_plant", curve=PLANT_CURVE, startup="integer_fix_in_uc", extra=[]),
    "direct_input_coeff_2": dict(chain=PLANT, unit="coal_plant", curve=PLANT_CURVE, extra=[
        ["unit__inputNode", ["coal_plant", "coal_market"], "conversion_flow_coeff", scalar(2.0)]]),
    # mle needs start-ups; the rising curve gets automatic linear start-ups (none authored on the piecewise twin).
    "indirect_two_outputs": dict(chain=CHP, unit="coal_chp", curve=CHP_CURVE, mle_startup="linear",
                                 extra=CHP_OUTPUTS),
    "direct_reduce_online": dict(chain=PLANT, unit="coal_plant", curve=PLANT_CURVE, extra=[
        ["solve", SOLVE, "reduce_unit_detail_method", text("online")]]),
}


def _twin_values(case: dict, kind: str, extra: list = ()) -> list:
    l0, e0, e1 = case["curve"]
    startup = case.get("startup") or (case.get("mle_startup") if kind == "mle" else None)
    unit = (mle if kind == "mle" else pw)(case["unit"], l0, e0, e1, startup=startup)
    return unit + case["extra"] + list(extra)


def _run(tmp_path_factory, alt: str, chain: list[str], values: list, *, fixture_json: Path = TESTS_JSON,
         entities: list | None = None) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities,
                            fixture_json=fixture_json)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


# ---------------------------------------------------------------------------
# LP and parameter views


def _lp(pb) -> dict:
    """The built LP keyed by names: rows (bounds and coefficients) and columns (objective and bounds)."""
    m = pb.canonicalise()
    rows: dict = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    cols: dict = {}
    for c in range(m.n_cols):
        name = m.col_names[c]
        cols[name] = (float(m.col_obj[c]), float(m.col_lb[c]), float(m.col_ub[c]))
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][name] = float(m.val[k])
    return {"rows": rows, "cols": cols}


def _mps(pb, path: Path) -> bytes:
    pb.write_mps(path)
    return path.read_bytes()


def _frame(value) -> pl.DataFrame | None:
    if value is None:
        return None
    f = value.frame if hasattr(value, "frame") else value
    return utf8(f.collect() if isinstance(f, pl.LazyFrame) else f)


FIELDS = ("p_slope", "p_section", "process_min_load_eff", "p_indirect_input_cap", "p_flow_upper", "p_min_load",
          "process_online", "process_minload")


def _assert_same_model(steps: dict, ref: dict, label: str, tmp_path: Path, *, fields=FIELDS) -> None:
    """Same steps; per step the same ``fields`` (values bit for bit), the same LP and the same written MPS."""
    assert list(steps) == list(ref), f"{label}: steps {list(steps)} vs {list(ref)}"
    for name in ref:
        fd, fd_ref = steps[name].flex_data, ref[name].flex_data
        for field in fields:
            got, want = _frame(getattr(fd, field)), _frame(getattr(fd_ref, field))
            assert (got is None) == (want is None), f"{label} {name}: {field} {got} vs {want}"
            if got is not None:
                assert got.equals(want), f"{label} {name}: {field} differs\n{got}\n{want}"
        pb, pb_ref = build(fd), build(fd_ref)
        assert _lp(pb) == _lp(pb_ref), f"{label} {name}: the LP differs"
        safe = name.replace("/", "_")
        assert _mps(pb, tmp_path / f"{safe}_a.mps") == _mps(pb_ref, tmp_path / f"{safe}_b.mps"), \
            f"{label} {name}: the written MPS differs"
        assert float(steps[name].solution.obj) == float(ref[name].solution.obj), f"{label} {name}: objective"


# ---------------------------------------------------------------------------
# T1: single solve


@pytest.mark.parametrize("case", list(CASES))
def test_two_point_curve_is_min_load_efficiency(tmp_path_factory, tmp_path, case) -> None:
    c = CASES[case]
    ref = _run(tmp_path_factory, f"t1_{case}_mle", c["chain"], _twin_values(c, "mle"))
    steps = _run(tmp_path_factory, f"t1_{case}_pw", c["chain"], _twin_values(c, "pw"))
    fd = list(steps.values())[-1].flex_data
    unit = c["unit"]
    if case == "direct_reduce_online":
        assert fd.process_min_load_eff is None or unit not in set(utf8(fd.process_min_load_eff)["p"].to_list())
    else:
        assert set(utf8(fd.process_min_load_eff)["p"].to_list()) == {unit}
        # the section of the curve, folded by the input's conversion_flow_coeff (2) on the direct unit
        l0, e0, e1 = c["curve"]
        beta = l0 * (1.0 / e0 - 1.0 / e1) * (1.0 / (1.0 - l0))
        k_src = 2.0 if case == "direct_input_coeff_2" else 1.0
        sec = utf8(fd.p_section.frame).filter(pl.col("p") == unit)
        assert set(sec["value"].to_list()) == {beta / k_src}
        assert sec.height == utf8(fd.dt).height
    _assert_same_model(steps, ref, case, tmp_path)


def test_hand_derived_slope_and_section(tmp_path_factory) -> None:
    """coal_plant {0.4: 0.30, 1: 0.42}: β = 0.4·(1/0.3 − 1/0.42)/0.6 = 0.4·(20/21)/0.6 = 40/63, α = 1/0.42 − β =
    50/21 − 40/63 = 110/63 (the oracle's O1 two-point curve; slope + section = 1/0.42)."""
    c = CASES["direct_linear"]
    fd = list(_run(tmp_path_factory, "t1_direct_linear_pw", c["chain"], _twin_values(c, "pw")).values())[-1].flex_data
    slope = set(utf8(fd.p_slope.frame).filter(pl.col("p") == "coal_plant")["value"].to_list())
    section = set(utf8(fd.p_section.frame).filter(pl.col("p") == "coal_plant")["value"].to_list())
    assert len(slope) == 1 and len(section) == 1
    assert slope.pop() == pytest.approx(float(Fraction(110, 63)), rel=1e-15)
    assert section.pop() == pytest.approx(float(Fraction(40, 63)), rel=1e-15)


# ---------------------------------------------------------------------------
# T1-R: rolling and stochastic


@pytest.mark.parametrize("case", ["direct_linear", "indirect_two_outputs"])
def test_rolling_two_point_curve_is_min_load_efficiency(tmp_path_factory, tmp_path, case) -> None:
    c = CASES[case]
    ref = _run(tmp_path_factory, f"t1r_{case}_mle", c["chain"], _twin_values(c, "mle", ROLLING))
    assert len([k for k in ref if "_roll_" in k]) >= 3, f"expected roll sub-solves, got {list(ref)}"
    steps = _run(tmp_path_factory, f"t1r_{case}_pw", c["chain"], _twin_values(c, "pw", ROLLING))
    _assert_same_model(steps, ref, f"rolling {case}", tmp_path)


STOCH_CASES = {
    "direct_linear": dict(unit="coal_plant", curve=STOCH_CURVE, startup="linear", extra=[]),
    "indirect_two_outputs": dict(unit="gas_plant", curve=STOCH_CURVE, mle_startup="linear", extra=STOCH_HEAT),
}


@pytest.mark.parametrize("case", list(STOCH_CASES))
def test_stochastic_two_point_curve_is_min_load_efficiency(tmp_path_factory, tmp_path, case) -> None:
    c = STOCH_CASES[case]
    kw = dict(fixture_json=STOCHASTICS_JSON, entities=STOCH_ENTITIES)
    ref = _run(tmp_path_factory, f"t1s_{case}_mle", STOCH, _twin_values(c, "mle"), **kw)
    steps = _run(tmp_path_factory, f"t1s_{case}_pw", STOCH, _twin_values(c, "pw"), **kw)
    fd = [s for k, s in steps.items() if "__realized_dispatch" not in k][-1].flex_data
    assert len(set(utf8(fd.dt)["d"].to_list())) > 1, "expected the branch periods in the stochastic solve"
    _assert_same_model(steps, ref, f"stochastic {case}", tmp_path)


# ---------------------------------------------------------------------------
# An investment solve handing its capacity to a dispatch solve; the written outputs; a flat curve

INVEST = "invest_1year_5weeks"


def _chain_values(existing: float, unitsize: float | None) -> list:
    """``invest_1year_5weeks`` (p2020 on the 2day timeset, coal_chp invests) before the dispatch solve."""
    def arr(xs: list[str]) -> list:
        return b64({"value_type": "str", "data": xs}, "array")

    out = [["model", "flexTool", "solves", arr([INVEST, SOLVE])],
           ["solve", INVEST, "invest_periods", arr(["p2020"])],
           ["solve", INVEST, "realized_invest_periods", arr(["p2020"])],
           ["solve", INVEST, "period_timeset", b64({"index_type": "str", "data": [["p2020", "2day"]]}, "map")],
           ["solve", INVEST, "solve_mode", text("single_solve")],
           ["unit", "coal_chp", "existing", scalar(existing)],
           ["unit", "coal_chp", "invest_method", text("invest_total")],
           ["unit", "coal_chp", "invest_cost", scalar(50.0)],
           ["unit", "coal_chp", "invest_max_total", scalar(3000.0)],
           ["unit", "coal_chp", "lifetime_technical", scalar(30.0)],
           ["unit", "coal_chp", "discount_rate", scalar(0.05)],
           ["unit", "coal_chp", "fixed_cost", scalar(1.0)]]
    return out + ([["unit", "coal_chp", "virtual_unitsize", scalar(unitsize)]] if unitsize else [])


@pytest.mark.parametrize("existing, unitsize", [(0.0, None), (200.0, 100.0)])
def test_invest_then_dispatch_chain(tmp_path_factory, tmp_path, existing, unitsize) -> None:
    """The indirect two-output unit invested in by an investment solve and then dispatched: in both solves the
    curve unit has the min_load_efficiency twin's parameters and LP (the dispatch solve's capacity comes from the
    investment solve's handoff)."""
    c = CASES["indirect_two_outputs"]
    extra = _chain_values(existing, unitsize)
    ref = _run(tmp_path_factory, f"chain_mle_{int(existing)}", c["chain"], _twin_values(c, "mle", extra))
    steps = _run(tmp_path_factory, f"chain_pw_{int(existing)}", c["chain"], _twin_values(c, "pw", extra))
    assert list(steps) == [INVEST, SOLVE]
    _assert_same_model(steps, ref, f"chain {existing}", tmp_path)


@pytest.mark.parametrize("case", ["direct_linear", "indirect_two_outputs"])
def test_written_outputs_are_those_of_min_load_efficiency(tmp_path_factory, case) -> None:
    """Every output file the runs write (parquet) is the same for the curve unit and its min_load_efficiency twin
    (the reported input of a direct unit adds the section on the units online, process_min_load_eff)."""
    c = CASES[case]
    _, wa, _ = _RUNS.get(tmp_path_factory, alt=f"t1_{case}_mle", values=_twin_values(c, "mle"),
                         base_chain=c["chain"], fixture_json=TESTS_JSON)
    _, wb, _ = _RUNS.get(tmp_path_factory, alt=f"t1_{case}_pw", values=_twin_values(c, "pw"),
                         base_chain=c["chain"], fixture_json=TESTS_JSON)
    fa = {p.relative_to(wa): p for p in wa.rglob("*.parquet")}
    fb = {p.relative_to(wb): p for p in wb.rglob("*.parquet")}
    assert fa and set(fa) == set(fb), sorted(map(str, set(fa) ^ set(fb)))
    diffs = [str(k) for k in sorted(fa) if not pl.read_parquet(fa[k]).equals(pl.read_parquet(fb[k]))]
    assert not diffs, diffs


@pytest.mark.parametrize("rolling", [False, True], ids=["single", "rolling"])
def test_flat_multi_point_curve_is_constant_efficiency(tmp_path_factory, tmp_path, rolling) -> None:
    """A three-point curve with one efficiency {0.3: 0.9, 0.6: 0.9, 1: 0.9} (merged to one line, intercept 0) is
    coal_chp with constant_efficiency 0.9: the same LP, single and rolling."""
    ex = ROLLING if rolling else []
    ref = _run(tmp_path_factory, f"flat_ce_{rolling}", CHP, [
        ["unit", "coal_chp", "conversion_method", text("constant_efficiency")],
        ["unit", "coal_chp", "efficiency", scalar(0.9)]] + CHP_OUTPUTS + ex)
    steps = _run(tmp_path_factory, f"flat_pw_{rolling}", CHP, [
        ["unit", "coal_chp", "conversion_method", text("piecewise_linear")],
        ["unit", "coal_chp", "piecewise_linear_efficiency", curve([(0.3, 0.9), (0.6, 0.9), (1.0, 0.9)])]]
        + CHP_OUTPUTS + ex)
    fd = list(steps.values())[-1].flex_data
    assert fd.process_piecewise is None and (fd.process_min_load_eff is None
                                             or "coal_chp" not in set(utf8(fd.process_min_load_eff)["p"].to_list()))
    _assert_same_model(steps, ref, f"flat {rolling}", tmp_path)


# ---------------------------------------------------------------------------
# A curve given once vs the same curve per period


def _const_vs_map(tmp_path_factory, tmp_path, label: str, chain: list[str], period: str, extra: list, **kw) -> dict:
    l0, e0, e1 = PLANT_CURVE if "entities" not in kw else STOCH_CURVE
    unit = "coal_plant"
    startup = "linear" if "entities" in kw else None
    pts = [(l0, e0), (1.0, e1)]
    ref = _run(tmp_path_factory, f"{label}_const", chain,
               pw(unit, l0, e0, e1, startup=startup, value=curve(pts)) + extra, **kw)
    steps = _run(tmp_path_factory, f"{label}_map", chain,
                 pw(unit, l0, e0, e1, startup=startup, value=curve_per_period({period: pts})) + extra, **kw)
    _assert_same_model(steps, ref, label, tmp_path)
    return steps


def test_constant_curve_equals_period_map_single(tmp_path_factory, tmp_path) -> None:
    _const_vs_map(tmp_path_factory, tmp_path, "shape_single", PLANT, "p2020", [])


def test_constant_curve_equals_period_map_rolling(tmp_path_factory, tmp_path) -> None:
    steps = _const_vs_map(tmp_path_factory, tmp_path, "shape_rolling", PLANT, "p2020", ROLLING)
    assert len([k for k in steps if "_roll_" in k]) >= 3


def test_constant_curve_equals_period_map_stochastic(tmp_path_factory, tmp_path) -> None:
    """The map has the anchor period1 only; every branch period takes its curve (same slope and section)."""
    steps = _const_vs_map(tmp_path_factory, tmp_path, "shape_stoch", STOCH, "period1", [],
                          fixture_json=STOCHASTICS_JSON, entities=STOCH_ENTITIES)
    fd = [s for k, s in steps.items() if "__realized_dispatch" not in k][-1].flex_data
    sec = utf8(fd.p_section.frame).filter(pl.col("p") == "coal_plant")
    periods = set(utf8(fd.dt)["d"].to_list())
    assert len(periods) > 1 and set(sec["d"].to_list()) == periods
    assert sec["value"].n_unique() == 1


# ---------------------------------------------------------------------------
# E11 and the multi-line parameters


def test_E11_no_curve_for_a_solve_period(tmp_path_factory) -> None:
    from flextool.engine import run_chain_from_db

    l0, e0, e1 = PLANT_CURVE
    # p2025 is a period of the database (model.periods_available) that the solve does not use.
    url, scen = make_db(tmp_path_factory, alt="e11", base_chain=PLANT, values=pw(
        "coal_plant", l0, e0, e1, value=curve_per_period({"p2025": [(l0, e0), (1.0, e1)]})) + [
        ["model", "flexTool", "periods_available", b64({"value_type": "str", "data": ["p2020", "p2025"]},
                                                         "array")]])
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("e11_w"))
    assert ("unit 'coal_plant': piecewise_linear_efficiency has no curve for period p2020 of solve "
            "y2020_2day_dispatch") in str(err.value)


def test_E11_rolling_names_the_user_solve_at_the_first_roll(tmp_path_factory, monkeypatch) -> None:
    """A rolling solve over p2030 and p2035 whose curve per period has p2030 only: the run stops at the load of its
    first roll (the periods of the user's solve are checked once), and the message names the user's solve s1,
    not the roll."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess.params import _piecewise
    from tests.model._piecewise_oracle_db import (
        balance_node, commodity_node, make_oracle_db, merge, scalar, time_settings, unit,
    )

    from pathlib import Path as _Path

    from flextool.engine.preprocess.params import _derived

    loads: list[str] = []
    original = _piecewise.apply_piecewise_curves

    def spy(flex_data, db_reader, provider=None):
        loads.append(str(_derived._read_active_solve(_Path("."), provider=provider)))
        return original(flex_data, db_reader, provider)

    monkeypatch.setattr(_piecewise, "apply_piecewise_curves", spy)
    ents, vals = merge(time_settings(3, ["p2030", "p2035"], rolling=(2.0, 1.0)),
                       commodity_node("fuel", "fuel_node", 10.0), balance_node("elec", -40.0),
                       unit("G", ["fuel_node"], ["elec"], {
                           "conversion_method": text("piecewise_linear"), "existing": scalar(100.0),
                           "piecewise_linear_efficiency": curve_per_period({"p2030": [(0.5, 0.4), (1.0, 0.5)]})}))
    url, scen = make_oracle_db(tmp_path_factory, "e11_roll", ents, vals)
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("e11_roll_w"))
    assert str(err.value) == "unit 'G': piecewise_linear_efficiency has no curve for period p2035 of solve s1."
    assert len(loads) == 1 and "_roll_" in loads[0], f"the run did not stop at the first roll: {loads}"


def test_multi_line_curve_parameters(tmp_path_factory) -> None:
    """The O1 curve {0.4: 0.30, 0.7: 0.40, 1: 0.42} on coal_plant (online, linear start-ups): the lines and the
    capacity parameters at the load boundary.  Hand-derived (oracle §2): seg 1 α = 25/18, β = 7/9; seg 2 α =
    265/126, β = 5/18 (no origin line, the efficiency rises); p_slope = 1/η(1) = 50/21; the input-arc limit is the
    full-load fuel max_g(α_g·W + β_g) with W = 1, i.e. 265/126 + 5/18 = 50/21 (one input, conversion_flow_coeff 1,
    input_share_max 1); a one-input one-output unit is remapped.  The solve runs (conversion_piecewise)."""
    c = CASES["direct_linear"]
    steps = _run(tmp_path_factory, "multi", c["chain"], pw(
        "coal_plant", 0.4, 0.30, 0.42, value=curve([(0.4, 0.30), (0.7, 0.40), (1.0, 0.42)])))
    fd = list(steps.values())[-1].flex_data
    lines = {(seg, d): (a, b) for (seg, d, a), b in zip(
        utf8(fd.p_piecewise_slope.frame).select("seg", "d", "value").iter_rows(),
        utf8(fd.p_piecewise_intercept.frame)["value"].to_list())}
    want = {"1": (Fraction(25, 18), Fraction(7, 9)), "2": (Fraction(265, 126), Fraction(5, 18))}
    assert {seg for seg, _ in lines} == set(want) and {d for _, d in lines} == {"p2020"}
    for (seg, _d), (a, b) in lines.items():
        assert a == pytest.approx(float(want[seg][0]), rel=1e-14)
        assert b == pytest.approx(float(want[seg][1]), rel=1e-14)
    for field in ("process_piecewise", "process_piecewise_needs_online", "process_piecewise_remap"):
        assert utf8(getattr(fd, field))["p"].to_list() == ["coal_plant"], field
    assert set(utf8(fd.p_slope.frame).filter(pl.col("p") == "coal_plant")["value"].to_list()) == {1.0 / 0.42}
    cap = utf8(fd.p_indirect_input_cap.frame).filter(pl.col("p") == "coal_plant")
    assert cap.height > 0 and set(cap.select("source", "sink").unique().rows()) == {("coal_market", "coal_plant")}
    assert cap["value"].to_list() == pytest.approx([float(Fraction(50, 21))] * cap.height, rel=1e-14)
    pts = fd.p_piecewise_points.with_columns(pl.col("d").cast(pl.Utf8)).sort("load")
    assert pts.select("d", "load", "eff").rows() == [("p2020", 0.4, 0.30), ("p2020", 0.7, 0.40),
                                                     ("p2020", 1.0, 0.42)]
