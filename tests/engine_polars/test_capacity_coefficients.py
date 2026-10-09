"""Per-edge capacity coefficients and capacity multipliers act on the
WHOLE built capacity (existing + invested − divested).

FlexTool semantics:

* ``capacity_max_coeff`` / ``capacity_min_coeff`` — fraction of the unit's
  capacity available to (imposed on) the edge.  ``0`` is a ZERO cap.
* ``conversion_flow_coeff = 0`` — the edge leaves the conversion equation
  and every per-edge capacity / ramp / min-load constraint (the
  hydro-pass-through pattern): it is uncapped.
* Capacity multipliers (``capacity_max_coeff``, ``availability``, ramp
  speed) scale ``existing + Σ invest − Σ divest`` — the .mod's
  ``v_flow ≤ coef · availability · (existing + invest − divest)``.  They
  used to multiply only the existing part, so an output with
  ``capacity_max_coeff = 0.2`` on a unit investing 900 MW could deliver
  900 MW instead of 180 MW.
* ``invest_retire_no_limit`` lifts every investment limit, exactly like
  ``invest_no_limit``.

Every scenario is ``tests.json`` + one override alternative, built from
JSON (never a checked-in DB).  ``virtual_unitsize = 1`` keeps the LP's
unit-count values equal to MW.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"

CHP = "coal_chp"            # indirect: coal_market → coal_chp → {west, heat}
PLANT = "coal_plant"        # direct:   coal_market → coal_plant → west
CHP_CHAIN = ["init", "west", "coal_chp", "heat"]
PLANT_CHAIN = ["init", "west", "coal"]
SOLVE = "y2020_2day_dispatch"
TIMESTEPS = [f"t{i:04d}" for i in range(1, 73)]


def _b64(x, kind: str) -> list:
    """Spine import-format packed value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


def _time_map(values: dict[str, float], default: float) -> list:
    return _b64({"index_type": "str", "index_name": "time",
                 "data": [[t, values.get(t, default)] for t in TIMESTEPS]},
                "map")


def _make_db(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
             values: list, base_chain: list[str],
             entities: list | None = None) -> tuple[str, str]:
    """``tests.json`` + override alternative ``alt`` appended to
    ``base_chain``; returns ``(url, scenario)``."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    scenario = f"{alt}_test"
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([alt, ""])
    for ent in entities or []:
        data["entities"].append(ent)
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, val, alt])
    chain = list(base_chain) + [alt]
    data["scenarios"].append([scenario, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([scenario, a, before])
    work = tmp_path_factory.mktemp(f"{alt}_db")
    json_path = work / f"{alt}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{alt}.sqlite")
    migrate_database(url)
    return url, scenario


def _solve(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
           values: list, base_chain: list[str],
           entities: list | None = None):
    from flextool.engine_polars import run_chain_from_db

    url, scenario = _make_db(tmp_path_factory, alt=alt, values=values,
                             base_chain=base_chain, entities=entities)
    work_folder = tmp_path_factory.mktemp(f"{alt}_work")
    steps = run_chain_from_db(url, scenario, work_folder=work_folder,
                              keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    step = list(steps.values())[-1]
    assert step.solution is not None, "solve produced no solution"
    return step


def _demand(node: str, mw: float) -> list:
    return ["node", node, "inflow", _b64(-mw, "float")]


def _free_chp() -> list:
    """Switch off the ratio-fixing user constraint of coal_chp."""
    return [["constraint", "coal_chp_fix", "is_enabled", _b64("no", "str")]]


def _greenfield(unit: str, method: str = "invest_total") -> list:
    """``unit`` starts from zero capacity and may invest cheaply in p2020."""
    return [
        ["unit", unit, "existing", _b64(0.0, "float")],
        ["unit", unit, "virtual_unitsize", _b64(1.0, "float")],
        ["unit", unit, "invest_method", _b64(method, "str")],
        ["unit", unit, "invest_max_total", _b64(5000.0, "float")],
        ["unit", unit, "invest_cost", _b64(1.0, "float")],
        ["unit", unit, "lifetime", _b64(20.0, "float")],
        ["unit", unit, "discount_rate", _b64(0.05, "float")],
        ["solve", SOLVE, "invest_periods",
         _b64({"type": "array", "value_type": "str", "data": ["p2020"]},
              "array")],
    ]


def _invested(step, unit: str) -> float:
    inv = (step.solution.value("v_invest_p")
           .with_columns(pl.col("p").cast(pl.Utf8))
           .filter(pl.col("p") == unit))
    assert inv.height > 0, f"{unit} has no investment variable"
    return float(inv["value"].sum())


def _flow(step, p: str, source: str, sink: str) -> pl.Series:
    us = float(step.flex_data.p_unitsize.frame
               .with_columns(pl.col("p").cast(pl.Utf8))
               .filter(pl.col("p") == p)["value"][0])
    f = (step.solution.value("v_flow")
         .with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink"))
         .filter((pl.col("p") == p) & (pl.col("source") == source)
                 & (pl.col("sink") == sink))
         .sort("d", "t"))
    assert f.height > 0, f"no v_flow rows for {p}: {source}->{sink}"
    return f["value"] * us


def _const(series: pl.Series, value: float, **kw) -> None:
    assert series.to_list() == pytest.approx([value] * series.len(), **kw)


# ── capacity_max_coeff with investment ──────────────────────────────────
WEST = 600.0
HEAT = 300.0


def test_indirect_output_coeff_scales_invested_capacity(tmp_path_factory):
    """Greenfield CHP, ``capacity_max_coeff = 0.2`` on the heat output:
    300 MW of heat needs 300 / 0.2 = 1500 MW BUILT (the west output and
    the output sum fit inside it).  With the coefficient applied only to
    the existing part, 900 MW of investment delivered 900 MW of heat."""
    step = _solve(tmp_path_factory, alt="capcoef_chp_invest",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + _greenfield(CHP) + [
                      _demand("west", WEST), _demand("heat", HEAT),
                      ["unit__outputNode", [CHP, "heat"],
                       "capacity_max_coeff", _b64(0.2, "float")]])
    assert _invested(step, CHP) == pytest.approx(HEAT / 0.2, rel=1e-6)
    _const(_flow(step, CHP, CHP, "heat"), HEAT, rel=1e-6)
    _const(_flow(step, CHP, CHP, "west"), WEST, rel=1e-6)


def test_indirect_output_zero_coeff_is_zero_cap_with_investment(
        tmp_path_factory):
    """``capacity_max_coeff = 0`` is a ZERO cap, also on invested
    capacity: no heat, investment sized for west only."""
    step = _solve(tmp_path_factory, alt="capcoef_chp_zero_invest",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + _greenfield(CHP) + [
                      _demand("west", WEST), _demand("heat", HEAT),
                      ["unit__outputNode", [CHP, "heat"],
                       "capacity_max_coeff", _b64(0.0, "float")]])
    assert _invested(step, CHP) == pytest.approx(WEST, rel=1e-6)
    _const(_flow(step, CHP, CHP, "heat"), 0.0, abs=1e-6)
    _const(_flow(step, CHP, CHP, "west"), WEST, rel=1e-6)


def test_indirect_output_zero_coeff_is_zero_cap_in_dispatch(
        tmp_path_factory):
    """Existing 1000 MW CHP, ``capacity_max_coeff = 0`` on heat: the heat
    output is pinned to zero (it used to be treated as uncapped)."""
    step = _solve(tmp_path_factory, alt="capcoef_chp_zero_dispatch",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + [
                      _demand("west", 700.0), _demand("heat", 1000.0),
                      ["unit__outputNode", [CHP, "heat"],
                       "capacity_max_coeff", _b64(0.0, "float")]])
    _const(_flow(step, CHP, CHP, "heat"), 0.0, abs=1e-6)
    _const(_flow(step, CHP, CHP, "west"), 700.0, rel=1e-6)


def test_direct_output_coeff_scales_invested_capacity(tmp_path_factory):
    """Direct unit (``coal_plant``), ``capacity_max_coeff = 0.2`` on its
    output: 600 MW needs 3000 MW built."""
    step = _solve(tmp_path_factory, alt="capcoef_direct_invest",
                  base_chain=PLANT_CHAIN,
                  values=_greenfield(PLANT) + [
                      _demand("west", WEST),
                      ["unit__outputNode", [PLANT, "west"],
                       "capacity_max_coeff", _b64(0.2, "float")]])
    assert _invested(step, PLANT) == pytest.approx(WEST / 0.2, rel=1e-6)
    _const(_flow(step, PLANT, "coal_market", "west"), WEST, rel=1e-6)


# ── availability with investment ────────────────────────────────────────
def test_direct_availability_scales_invested_capacity(tmp_path_factory):
    """Direct unit with availability 0.5: 600 MW needs 1200 MW built."""
    step = _solve(tmp_path_factory, alt="avail_direct_invest",
                  base_chain=PLANT_CHAIN,
                  values=_greenfield(PLANT) + [
                      _demand("west", WEST),
                      ["unit", PLANT, "availability", _b64(0.5, "float")]])
    assert _invested(step, PLANT) == pytest.approx(WEST / 0.5, rel=1e-6)
    _const(_flow(step, PLANT, "coal_market", "west"), WEST, rel=1e-6)


def test_indirect_availability_scales_invested_capacity(tmp_path_factory):
    """Indirect CHP with availability 0.5: the output SUM (900 MW) needs
    1800 MW built (``maxOutputSum``'s invest term carries availability)."""
    step = _solve(tmp_path_factory, alt="avail_chp_invest",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + _greenfield(CHP) + [
                      _demand("west", WEST), _demand("heat", HEAT),
                      ["unit", CHP, "availability", _b64(0.5, "float")]])
    assert _invested(step, CHP) == pytest.approx((WEST + HEAT) / 0.5,
                                                 rel=1e-6)
    _const(_flow(step, CHP, CHP, "west"), WEST, rel=1e-6)
    _const(_flow(step, CHP, CHP, "heat"), HEAT, rel=1e-6)


# ── conversion_flow_coeff = 0 → uncapped edge ───────────────────────────
def test_zero_conversion_coeff_edge_is_uncapped(tmp_path_factory):
    """``conversion_flow_coeff = 0`` on the heat output takes it out of the
    conversion equation and of every capacity constraint: 1500 MW of heat
    flows from a 1000 MW unit that also serves 700 MW of west, and the
    fuel covers west only."""
    step = _solve(tmp_path_factory, alt="convcoef_zero_uncapped",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + [
                      _demand("west", 700.0), _demand("heat", 1500.0),
                      ["unit__outputNode", [CHP, "heat"],
                       "conversion_flow_coeff", _b64(0.0, "float")]])
    fd = step.flex_data
    unc = (fd.process_source_sink_uncapped
           .with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink")))
    assert unc.rows() == [(CHP, CHP, "heat")]
    _const(_flow(step, CHP, CHP, "heat"), 1500.0, rel=1e-6)
    _const(_flow(step, CHP, CHP, "west"), 700.0, rel=1e-6)
    _const(_flow(step, CHP, "coal_market", CHP), 700.0 / 0.9, rel=1e-6)


def test_zero_conversion_coeff_edge_needs_no_investment(tmp_path_factory):
    """Greenfield CHP whose heat output has ``conversion_flow_coeff = 0``:
    only west needs built capacity."""
    step = _solve(tmp_path_factory, alt="convcoef_zero_invest",
                  base_chain=CHP_CHAIN,
                  values=_free_chp() + _greenfield(CHP) + [
                      _demand("west", WEST), _demand("heat", HEAT),
                      ["unit__outputNode", [CHP, "heat"],
                       "conversion_flow_coeff", _b64(0.0, "float")]])
    assert _invested(step, CHP) == pytest.approx(WEST, rel=1e-6)
    _const(_flow(step, CHP, CHP, "heat"), HEAT, rel=1e-6)


# ── no-limit investment methods ─────────────────────────────────────────
@pytest.mark.parametrize("method", ["invest_no_limit",
                                    "invest_retire_no_limit"])
def test_no_limit_methods_lift_invest_limits(tmp_path_factory, method):
    """Both no-limit methods ignore ``invest_max_period`` (800 MW here) and
    build the 900 MW the output sum needs.  ``invest_retire_no_limit`` used
    to be unrecognised: ``p_entity_max_units`` stayed empty, so
    ``maxInvest_var_bound`` pinned ``v_invest ≤ 0`` (and the fuel arc's
    structural bound collapsed to 0)."""
    period_cap = _b64({"index_type": "str", "index_name": "period",
                       "data": [["p2020", 800.0]]}, "map")
    values = _free_chp() + _greenfield(CHP, method) + [
        _demand("west", WEST), _demand("heat", HEAT),
        ["unit", CHP, "invest_max_period", period_cap]]
    step = _solve(tmp_path_factory, alt=f"nolimit_{method}",
                  base_chain=CHP_CHAIN, values=values)
    max_units = (step.flex_data.p_entity_max_units.frame
                 .with_columns(pl.col("e").cast(pl.Utf8))
                 .filter(pl.col("e") == CHP))
    assert max_units.height > 0
    assert float(max_units["value"].min()) >= 1_000_000.0
    assert _invested(step, CHP) == pytest.approx(WEST + HEAT, rel=1e-6)
    _const(_flow(step, CHP, CHP, "west"), WEST, rel=1e-6)
    _const(_flow(step, CHP, CHP, "heat"), HEAT, rel=1e-6)


# ── capacity_min_coeff: the min-load floor scales with built capacity ───
def test_min_coeff_floor_scales_with_invested_capacity(tmp_path_factory):
    """Greenfield linear-online ``coal_plant`` (min_load 0.5,
    ``capacity_min_coeff = 0.8``) serving a flat 600 MW with one 100 MW
    dip.  A prohibitive startup cost keeps the whole built capacity
    online, so in the dip the floor ``v_online · min_load · coef`` binds at
    0.5 · 0.8 · 600 = 240 MW — the floor follows the INVESTED capacity."""
    dip = "t0010"
    step = _solve(tmp_path_factory, alt="mincoef_invest",
                  base_chain=PLANT_CHAIN + ["coal_min_load"],
                  values=_greenfield(PLANT) + [
                      ["node", "west", "inflow",
                       _time_map({dip: -100.0}, -WEST)],
                      ["unit", PLANT, "startup_cost", _b64(1.0e6, "float")],
                      ["unit__outputNode", [PLANT, "west"],
                       "capacity_min_coeff", _b64(0.8, "float")]])
    built = _invested(step, PLANT)
    assert built == pytest.approx(WEST, rel=1e-6)
    t = (step.solution.value("v_flow")
         .with_columns(pl.col(c).cast(pl.Utf8)
                       for c in ("p", "source", "sink", "t"))
         .filter((pl.col("p") == PLANT) & (pl.col("sink") == "west")))
    dip_flow = float(t.filter(pl.col("t") == dip)["value"][0])
    assert dip_flow == pytest.approx(0.5 * 0.8 * built, rel=1e-6)


# ── ramp limits scale with coefficient and invested capacity ────────────
def test_ramp_limit_follows_invested_capacity(tmp_path_factory):
    """Greenfield ``coal_plant`` with a (non-binding for built capacity)
    ramp limit follows the west demand series.  Before the fix the ramp
    RHS used the existing count only (0), freezing the flow."""
    step = _solve(tmp_path_factory, alt="ramp_invest",
                  base_chain=PLANT_CHAIN + ["coal_ramp"],
                  values=_greenfield(PLANT) + [
                      ["unit__outputNode", [PLANT, "west"],
                       "ramp_speed_up", _b64(0.5, "float")],
                      ["unit__outputNode", [PLANT, "west"],
                       "ramp_speed_down", _b64(0.5, "float")]])
    flow = _flow(step, PLANT, "coal_market", "west")
    assert flow.n_unique() > 1, "ramp limit froze the invested unit"
    assert _invested(step, PLANT) == pytest.approx(float(flow.max()),
                                                   rel=1e-6)


def test_ramp_limit_scales_with_capacity_max_coeff(tmp_path_factory):
    """Existing 1000 MW ``coal_plant`` with ``ramp_speed_up = 0.001``
    (0.06 of capacity per hour) and ``capacity_max_coeff = 0.5``: the
    hourly increase is capped at 0.001 · 60 · 0.5 · 1000 = 30 MW and the
    demand series (rising by up to 44 MW/h) makes it bind."""
    step = _solve(tmp_path_factory, alt="ramp_coef",
                  base_chain=PLANT_CHAIN + ["coal_ramp"],
                  values=[
                      ["unit", PLANT, "existing", _b64(1000.0, "float")],
                      ["unit__outputNode", [PLANT, "west"],
                       "capacity_max_coeff", _b64(0.5, "float")]])
    flow = _flow(step, PLANT, "coal_market", "west")
    steps = flow.to_list()
    # cyclic within the timeset: t0001 follows the last step.
    rises = [b - a for a, b in zip([steps[-1]] + steps[:-1], steps)]
    assert max(rises) == pytest.approx(30.0, rel=1e-6)


# ── 2-way reverse direction uses invested capacity ──────────────────────
def test_reverse_flow_uses_invested_capacity(tmp_path_factory):
    """Lossless 2-way connection ``ll`` built from scratch: it must carry
    100 MW forward at t0001 and 100 MW backwards at t0002 on the 100 MW
    it invests (``maxFlow_back`` used to cap the reverse direction at the
    existing capacity only)."""
    solve = "lossless_2way_solve"
    step = _solve(tmp_path_factory, alt="ll_invest",
                  base_chain=["Base", "lossless_2way"],
                  values=[
                      ["connection", "ll", "existing", _b64(0.0, "float")],
                      ["connection", "ll", "virtual_unitsize",
                       _b64(1.0, "float")],
                      ["connection", "ll", "invest_method",
                       _b64("invest_total", "str")],
                      ["connection", "ll", "invest_max_total",
                       _b64(1000.0, "float")],
                      ["connection", "ll", "invest_cost", _b64(1.0, "float")],
                      ["connection", "ll", "lifetime", _b64(20.0, "float")],
                      ["connection", "ll", "discount_rate",
                       _b64(0.05, "float")],
                      ["solve", solve, "invest_periods",
                       _b64({"type": "array", "value_type": "str",
                             "data": ["p2020"]}, "array")]])
    assert _invested(step, "ll") == pytest.approx(100.0, rel=1e-6)
    back = (step.solution.value("v_flow_back")
            .with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "t"))
            .filter(pl.col("p") == "ll").sort("t"))
    assert back["value"].to_list() == pytest.approx([0.0, 100.0], abs=1e-6)
