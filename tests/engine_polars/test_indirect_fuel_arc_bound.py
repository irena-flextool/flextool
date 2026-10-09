"""Indirect (multi-flow) unit capacity: the SUM of outputs is capped, and
the loose fuel-arc bound must never be the binding limit.

Unit capacity (``existing`` + invest − divest) is "the maximum sum of
output flows".  For an indirect (``nvar``) unit that is ``maxOutputSum``::

    Σ_k out_k (+ Σ_k reserve_up_k) − invest + divest ≤ existing · availability

(``maxOutputSum_online_<kind>``: ``Σ_k out_k ≤ v_online · availability``),
on top of the per-arc caps ``out_k ≤ capacity · capacity_max_coeff_k``
(``maxFlow``).  The input (fuel) arcs carry only a structural "loose"
bound from ``p_flow_upper`` that exists to give the solver a bounded
region.  The fuel is tied to the outputs by ``conversion_indirect``::

    Σ_s src_conv_s · in_s = slope · Σ_k sink_conv_k · out_k

so the largest fuel any single input arc can ever be asked for is::

    in_s ≤ capacity · slope · Σ_k (sink_conv_k · capacity_max_coeff_k)
                                                          / src_conv_s

The bound used to be ``slope · capacity`` (enough fuel for ONE output at
full capacity).  With several outputs — or with a ``conversion_flow_coeff``
above 1 on an output — the loose bound became the binding constraint and
silently capped the unit (and in investment solves, where the fuel bound
is loose, the sum of outputs was not capped at all).

The fixtures use the ``coal_chp`` unit from ``tests.json`` (existing
1000 MW, efficiency 0.9, outputs ``west`` + ``heat``, input
``coal_market``) with its ratio-fixing user constraint switched off.
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

UNIT = "coal_chp"
FUEL_NODE = "coal_market"
CAPACITY = 1000.0
EFFICIENCY = 0.9
SINK_CONV = {"west": 2.0, "heat": 0.5}
SRC_CONV = 0.8
# Σ_k sink_conv_k · capacity_max_coeff_k / src_conv  (capacity coefs = 1)
FUEL_FACTOR = sum(SINK_CONV.values()) / SRC_CONV
# Demands of the main fixture: west (penalty 900 €/MWh) is served first,
# heat (penalty 100 €/MWh, demand = capacity) gets what is left of the
# SUM cap.  Per-arc caps alone would serve both in full.
WEST_DEMAND = 700.0
HEAT_DEMAND = CAPACITY

_ALT = "indirect_fuel_bound"
_SCENARIO = "indirect_fuel_bound_test"
_BASE_CHAIN = ["init", "west", "coal_chp", "heat"]


def _b64(x, kind: str) -> list:
    """Spine import-format packed scalar value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


def _make_db(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
             scenario: str, values: list, entities: list | None = None,
             base_chain: list[str] | None = None) -> str:
    """``tests.json`` + one override alternative ``alt`` (parameter rows
    ``[class, entity, param, packed_value]``) appended to ``base_chain``
    as scenario ``scenario``.  Built from JSON — never a checked-in DB."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([alt, ""])
    for ent in entities or []:
        data["entities"].append(ent)
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, val, alt])
    chain = list(base_chain or _BASE_CHAIN) + [alt]
    data["scenarios"].append([scenario, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([scenario, a, before])
    work = tmp_path_factory.mktemp(f"{alt}_db")
    json_path = work / f"{alt}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{alt}.sqlite")
    migrate_database(url)
    return url


def _solve(url: str, scenario: str,
           tmp_path_factory: pytest.TempPathFactory):
    from flextool.engine_polars import run_chain_from_db

    work_folder = tmp_path_factory.mktemp(f"{scenario}_work")
    steps = run_chain_from_db(url, scenario, work_folder=work_folder,
                              keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    step = list(steps.values())[-1]
    assert step.solution is not None, "solve produced no solution"
    return step


def _free_chp() -> list:
    """Switch off the ratio-fixing user constraint of coal_chp."""
    return [["constraint", "coal_chp_fix", "is_enabled", _b64("no", "str")]]


def _demand(node: str, mw: float) -> list:
    return ["node", node, "inflow", _b64(-mw, "float")]


@pytest.fixture(scope="module")
def fuel_bound_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """``tests.json`` + an override alternative making the coal_chp unit
    a 2-output indirect unit with non-unit conversion coefficients whose
    combined demand exceeds its capacity.
    """
    values = _free_chp() + [_demand("west", WEST_DEMAND),
                            _demand("heat", HEAT_DEMAND)]
    for sink, conv in SINK_CONV.items():
        values.append(["unit__outputNode", [UNIT, sink],
                       "conversion_flow_coeff", _b64(conv, "float")])
    values.append(["unit__inputNode", [UNIT, FUEL_NODE],
                   "conversion_flow_coeff", _b64(SRC_CONV, "float")])
    return _make_db(tmp_path_factory, alt=_ALT, scenario=_SCENARIO,
                    values=values)


@pytest.fixture(scope="module")
def solved_step(fuel_bound_db: str, tmp_path_factory: pytest.TempPathFactory):
    return _solve(fuel_bound_db, _SCENARIO, tmp_path_factory)


def _unitsize(fd) -> float:
    return float(fd.p_unitsize.frame
                 .filter(pl.col("p") == UNIT)["value"][0])


def test_unit_is_indirect(solved_step) -> None:
    """Guard: the scenario really exercises the indirect (nvar) path."""
    fd = solved_step.flex_data
    assert fd.process_indirect is not None
    assert UNIT in fd.process_indirect["p"].cast(pl.Utf8).to_list()


def test_fuel_arc_bound_covers_conversion_equation(solved_step) -> None:
    """``p_flow_upper`` on the fuel arc equals the conversion-implied
    maximum ``slope · cap · Σ(sink_conv·capcoef) / src_conv`` (per unit
    of unitsize) — not ``slope · cap`` and not ``slope · cap · n_outputs``.
    """
    fd = solved_step.flex_data
    us = _unitsize(fd)
    rows = (fd.p_flow_upper.frame
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink"))
            .filter((pl.col("p") == UNIT) & (pl.col("source") == FUEL_NODE)
                    & (pl.col("sink") == UNIT)))
    assert rows.height > 0, "no p_flow_upper rows for the fuel arc"
    expected = CAPACITY / us / EFFICIENCY * FUEL_FACTOR
    assert rows["value"].to_list() == pytest.approx(
        [expected] * rows.height, rel=1e-9)


def _flows(step) -> pl.DataFrame:
    return (step.solution.value("v_flow")
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink")))


def _out(flow: pl.DataFrame, sink: str) -> pl.DataFrame:
    out = flow.filter((pl.col("p") == UNIT) & (pl.col("source") == UNIT)
                      & (pl.col("sink") == sink)).sort("d", "t")
    assert out.height > 0, f"no v_flow rows for {UNIT}->{sink}"
    return out


def test_sum_of_outputs_reaches_capacity(solved_step) -> None:
    """Combined demand (700 + 1000) exceeds the capacity: the SUM of the
    outputs must equal the capacity in every timestep — west (high
    penalty) at its demand, heat taking the remaining 300 MW — and the
    fuel arc must deliver exactly what conversion_indirect asks for (it
    must not be the binding limit)."""
    us = _unitsize(solved_step.flex_data)
    flow = _flows(solved_step)
    west = _out(flow, "west")["value"] * us
    heat = _out(flow, "heat")["value"] * us
    n = west.len()
    assert west.to_list() == pytest.approx([WEST_DEMAND] * n, rel=1e-6)
    assert heat.to_list() == pytest.approx(
        [CAPACITY - WEST_DEMAND] * n, rel=1e-6), (
        "heat output exceeds what the SUM capacity cap leaves after west")
    assert (west + heat).to_list() == pytest.approx([CAPACITY] * n,
                                                    rel=1e-6)
    fuel = flow.filter((pl.col("p") == UNIT)
                       & (pl.col("source") == FUEL_NODE)
                       & (pl.col("sink") == UNIT))
    expected_fuel = (SINK_CONV["west"] * WEST_DEMAND
                     + SINK_CONV["heat"] * (CAPACITY - WEST_DEMAND)
                     ) / EFFICIENCY / SRC_CONV
    assert (fuel["value"] * us).to_list() == pytest.approx(
        [expected_fuel] * fuel.height, rel=1e-6)


def test_max_output_sum_rows_emitted(solved_step) -> None:
    """Structural: one ``maxOutputSum`` row per (unit, d, t) — not one
    per output arc — with RHS = existing count (availability 1)."""
    from polar_high import Problem

    from flextool.engine_polars import build_flextool

    fd = solved_step.flex_data
    pb = Problem()
    build_flextool(pb, fd)
    assert pb.cstr_row_count("maxOutputSum") == fd.dt.height
    assert "maxOutputSum_negCap" not in set(pb.cstr_names())


# ── Uncapped output → unbounded fuel need ───────────────────────────────
_ALT_UNCAPPED = "indirect_fuel_uncapped"
_SCENARIO_UNCAPPED = "indirect_fuel_uncapped_test"


def test_uncapped_output_leaves_fuel_arc_unconstrained(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """An output with ``capacity_max_coeff = 0`` has no cap (coeff_zero),
    so the fuel it can demand is unbounded — the fuel arc must take the
    unconstrained value rather than a capacity-derived one."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.engine_polars import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([_ALT_UNCAPPED, ""])
    pv = data["parameter_values"]
    pv.append(["constraint", "coal_chp_fix", "is_enabled",
               _b64("no", "str"), _ALT_UNCAPPED])
    pv.append(["unit__outputNode", [UNIT, "heat"], "capacity_max_coeff",
               _b64(0.0, "float"), _ALT_UNCAPPED])
    chain = ["init", "west", "coal_chp", "heat", _ALT_UNCAPPED]
    data["scenarios"].append([_SCENARIO_UNCAPPED, False, ""])
    for alt, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append(
            [_SCENARIO_UNCAPPED, alt, before])
    work = tmp_path_factory.mktemp("indirect_fuel_uncapped")
    json_path = work / "uncapped.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "uncapped.sqlite")
    migrate_database(url)

    steps = run_chain_from_db(url, _SCENARIO_UNCAPPED,
                              work_folder=work, keep_solutions=True)
    fd = list(steps.values())[-1].flex_data
    rows = (fd.p_flow_upper.frame
            .with_columns(pl.col(c).cast(pl.Utf8)
                          for c in ("p", "source", "sink"))
            .filter((pl.col("p") == UNIT) & (pl.col("source") == FUEL_NODE)
                    & (pl.col("sink") == UNIT)))
    assert rows.height > 0
    assert rows["value"].min() >= 1_000_000.0


# ── Delay: one input step can feed a heavier-weighted output step ───────
class _StubSource:
    """Minimal InputSource exposing only ``unit.delay``."""

    def __init__(self, delay: pl.DataFrame) -> None:
        self._delay = delay

    def parameter(self, entity_class: str, parameter_name: str):
        if (entity_class, parameter_name) == ("unit", "delay"):
            return self._delay
        raise KeyError((entity_class, parameter_name))


def test_delay_fuel_factor() -> None:
    """``delay_factor = 1 / max(weight)``: a 0.25/0.75 split lets one input
    step carry 1/0.75 of the steady-state fuel; a scalar delay (single
    duration, weight 1) leaves the bound unchanged."""
    from flextool.engine_polars._derived_params import _delay_fuel_factor_lf

    weighted = _StubSource(pl.DataFrame({
        "name": ["u_split", "u_split"],
        "delay_duration": ["1.0", "2.0"],
        "value": [0.25, 0.75],
    }))
    got = dict(_delay_fuel_factor_lf(weighted)
               .with_columns(pl.col("p").cast(pl.Utf8)).collect().iter_rows())
    assert got == pytest.approx({"u_split": 1.0 / 0.75})

    single = _StubSource(pl.DataFrame({"name": ["u_single"], "value": [3.0]}))
    got = dict(_delay_fuel_factor_lf(single)
               .with_columns(pl.col("p").cast(pl.Utf8)).collect().iter_rows())
    assert got == pytest.approx({"u_single": 1.0})


# ── Indirect output arcs honour capacity_max_coeff ──────────────────────
def test_indirect_output_capacity_max_coeff_is_enforced(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``coal_chp_extraction`` sets ``capacity_max_coeff = 0.8`` on the
    electricity output (west), whose demand peaks at 870 MW.  With heat
    demand switched off the SUM cap (1000 MW) is slack, so the per-arc
    cap is the only limit: west must stop at 0.8 · 1000 = 800 MW
    (maxToSink) — and must actually reach it, i.e. neither the per-output
    coefficient may be dropped nor may the fuel arc cap the unit below it
    (it used to pin west at 470 MW)."""
    url = _make_db(
        tmp_path_factory, alt="indirect_capcoef_no_heat",
        scenario="indirect_capcoef_no_heat_test",
        values=[_demand("heat", 0.0)],
        base_chain=_BASE_CHAIN + ["coal_chp_extraction"])
    step = _solve(url, "indirect_capcoef_no_heat_test", tmp_path_factory)
    us = _unitsize(step.flex_data)
    flow = _flows(step)
    assert float((_out(flow, "heat")["value"] * us).max()) == pytest.approx(
        0.0, abs=1e-6)
    west = _out(flow, "west")
    assert float(west["value"].max()) * us == pytest.approx(0.8 * CAPACITY,
                                                            rel=1e-6)


# ── Sum cap against BUILT capacity in an investment solve ───────────────
INV_WEST = 600.0
INV_HEAT = 300.0


def test_sum_cap_binds_against_invested_capacity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Greenfield coal_chp (existing 0, invest allowed, cheap).  Serving
    west 600 MW + heat 300 MW needs 900 MW of BUILT capacity — the sum
    of the outputs — not max(600, 300) = 600 MW as per-arc caps alone
    would allow (and the loose fuel bound does not cap investment
    solves)."""
    values = _free_chp() + [
        _demand("west", INV_WEST), _demand("heat", INV_HEAT),
        ["unit", UNIT, "existing", _b64(0.0, "float")],
        ["unit", UNIT, "virtual_unitsize", _b64(1.0, "float")],
        ["unit", UNIT, "invest_method", _b64("invest_total", "str")],
        ["unit", UNIT, "invest_max_total", _b64(5000.0, "float")],
        ["unit", UNIT, "invest_cost", _b64(1.0, "float")],
        ["unit", UNIT, "lifetime", _b64(20.0, "float")],
        ["unit", UNIT, "discount_rate", _b64(0.05, "float")],
        ["solve", "y2020_2day_dispatch", "invest_periods",
         _b64({"type": "array", "value_type": "str",
               "data": ["p2020"]}, "array")],
    ]
    url = _make_db(tmp_path_factory, alt="indirect_sum_invest",
                   scenario="indirect_sum_invest_test", values=values)
    step = _solve(url, "indirect_sum_invest_test", tmp_path_factory)
    us = _unitsize(step.flex_data)
    assert us == pytest.approx(1.0)
    inv = (step.solution.value("v_invest_p")
           .with_columns(pl.col("p").cast(pl.Utf8))
           .filter(pl.col("p") == UNIT))
    assert inv.height > 0, "coal_chp has no investment variable"
    assert float(inv["value"].sum()) * us == pytest.approx(
        INV_WEST + INV_HEAT, rel=1e-6)
    flow = _flows(step)
    west = _out(flow, "west")["value"] * us
    heat = _out(flow, "heat")["value"] * us
    assert west.to_list() == pytest.approx([INV_WEST] * west.len(), rel=1e-6)
    assert heat.to_list() == pytest.approx([INV_HEAT] * heat.len(), rel=1e-6)


# ── Online (unit-commitment) variant ────────────────────────────────────
ON_WEST = 500.0
ON_HEAT = 300.0


def test_sum_cap_online_variant(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Linear-online coal_chp with a min-load efficiency section: fuel
    grows with ``v_online``, so the LP keeps ``v_online`` as low as the
    capacity constraints allow.  ``maxOutputSum_online_linear`` forces
    ``v_online · unitsize ≥ west + heat = 800`` MW; per-arc
    ``maxFlow_online`` alone would let it drop to max(500, 300) = 500."""
    values = _free_chp() + [
        _demand("west", ON_WEST), _demand("heat", ON_HEAT),
        ["unit", UNIT, "startup_method", _b64("linear", "str")],
        ["unit", UNIT, "conversion_method",
         _b64("min_load_efficiency", "str")],
        ["unit", UNIT, "min_load", _b64(0.2, "float")],
        ["unit", UNIT, "efficiency_at_min_load", _b64(0.4, "float")],
    ]
    url = _make_db(tmp_path_factory, alt="indirect_sum_online",
                   scenario="indirect_sum_online_test", values=values)
    step = _solve(url, "indirect_sum_online_test", tmp_path_factory)
    us = _unitsize(step.flex_data)
    flow = _flows(step)
    west = _out(flow, "west")["value"] * us
    heat = _out(flow, "heat")["value"] * us
    n = west.len()
    assert west.to_list() == pytest.approx([ON_WEST] * n, rel=1e-6)
    assert heat.to_list() == pytest.approx([ON_HEAT] * n, rel=1e-6)
    online = (step.solution.value("v_online_linear")
              .with_columns(pl.col("p").cast(pl.Utf8))
              .filter(pl.col("p") == UNIT).sort("d", "t"))
    assert online.height == n
    assert (online["value"] * us).to_list() == pytest.approx(
        [ON_WEST + ON_HEAT] * n, rel=1e-6)

    from polar_high import Problem

    from flextool.engine_polars import build_flextool

    pb = Problem()
    build_flextool(pb, step.flex_data)
    assert pb.cstr_row_count("maxOutputSum_online_linear") == n


# ── Upward reserve on an output counts against the sum ──────────────────
RES_WEST = 500.0
RES_UP = 200.0


def test_sum_cap_counts_upward_reserve(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """coal_chp provides 200 MW of upward reserve on its west output
    (reserve penalty ≫ heat penalty, so the reserve is always held).
    The capacity held for the reserve is unavailable to any output:
    heat = 1000 − 500 (west) − 200 (reserve) = 300 MW, not 500."""
    res = ["primary", "up", UNIT, "west"]
    values = _free_chp() + [
        _demand("west", RES_WEST), _demand("heat", CAPACITY),
        ["reserve__upDown__group", ["primary", "up", "electricity"],
         "reserve_method", _b64("timeseries_only", "str")],
        ["reserve__upDown__group", ["primary", "up", "electricity"],
         "reservation", _b64(RES_UP, "float")],
        ["reserve__upDown__group", ["primary", "up", "electricity"],
         "penalty_reserve", _b64(20000.0, "float")],
        ["reserve__upDown__unit__node", res, "is_enabled",
         _b64("yes", "str")],
        ["reserve__upDown__unit__node", res, "max_share",
         _b64(1.0, "float")],
        ["reserve__upDown__unit__node", res, "reliability",
         _b64(1.0, "float")],
    ]
    url = _make_db(tmp_path_factory, alt="indirect_sum_reserve",
                   scenario="indirect_sum_reserve_test", values=values,
                   entities=[["reserve__upDown__unit__node", res, None]])
    step = _solve(url, "indirect_sum_reserve_test", tmp_path_factory)
    us = _unitsize(step.flex_data)
    reserve = (step.solution.value("v_reserve")
               .with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "n"))
               .filter((pl.col("p") == UNIT) & (pl.col("n") == "west")))
    assert reserve.height > 0, "coal_chp carries no reserve variable"
    n = reserve.height
    assert (reserve["value"] * us).to_list() == pytest.approx(
        [RES_UP] * n, rel=1e-6)
    flow = _flows(step)
    west = _out(flow, "west")["value"] * us
    heat = _out(flow, "heat")["value"] * us
    assert west.to_list() == pytest.approx([RES_WEST] * west.len(), rel=1e-6)
    assert heat.to_list() == pytest.approx(
        [CAPACITY - RES_WEST - RES_UP] * heat.len(), rel=1e-6)
