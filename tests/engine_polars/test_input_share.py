"""Input-arc limits of FlexTool units: ``unit__inputNode.input_share_max``.

Unit capacity is the maximum SUM of outputs (``maxOutputSum``).  The input
(fuel) arcs of an indirect unit need no capacity cap for correctness — the
fuel follows the outputs through ``conversion_indirect``::

    Σ_s src_conv_s · in_s = slope · Σ_k conv_k · out_k (+ section · online)

Each input arc ``s`` carries the limit::

    in_s ≤ input_share_max_s · L_s · availability · capacity
    L_s  = (slope · W + section) · delay_factor / src_conv_s

``W`` is the most output-side fuel energy per unit of capacity the outputs
can demand under ``Σ out ≤ capacity`` and ``out_k ≤ capacity_max_coeff_k ·
capacity`` (greedy: outputs by descending ``conv_k``, each filled up to
``min(capacity_max_coeff_k, remaining)``).  ``input_share_max`` (default 1)
is the largest share of the full-load fuel the input may supply: with 1 the
input alone runs the unit at full output (no effective limit).

How the limit is applied:

* unit WITHOUT invest / divest variables in the solve (constant capacity):
  a per-element ``v_flow`` upper bound, and NO ``maxFlow`` row on the arc;
* unit WITH them: a ``maxFlow`` row
  ``v_flow − M·Σv_invest + M·Σv_divest ≤ M·existing/unitsize``,
  ``M = input_share_max · L · availability`` (the invest term used to carry
  multiplier 1 — fuel needs ``L`` per unit of new capacity).

Fixtures: ``tests.json`` + an override alternative, built from JSON (never
a checked-in DB).  ``coal_chp`` (existing 1000 MW, efficiency 0.9,
outputs west + heat, input coal_market) has its ratio-fixing user
constraint switched off.
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
COAL = "coal_market"
GAS = "gas_market"
CAPACITY = 1000.0
EFFICIENCY = 0.9
CHP_CHAIN = ["init", "west", "coal_chp", "heat"]
GAS_ENTITIES = [
    ["commodity", "gas", None],
    ["node", GAS, None],
    ["commodity__node", ["gas", GAS], None],
    ["unit__inputNode", [UNIT, GAS], None],
]


def _b64(x, kind: str) -> list:
    """Spine import-format packed value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


def _make_db(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
             values: list, base_chain: list[str] | None = None,
             entities: list | None = None) -> tuple[str, str]:
    """``tests.json`` + override alternative ``alt`` appended to
    ``base_chain``; returns ``(url, scenario)``."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([alt, ""])
    for ent in entities or []:
        data["entities"].append(ent)
        if ent[0] == "node":
            # ``node`` is not active by default: activate it in ``alt``.
            data["entity_alternatives"].append(["node", [ent[1]], alt, True])
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, val, alt])
    scenario = f"{alt}_test"
    chain = list(base_chain or CHP_CHAIN) + [alt]
    data["scenarios"].append([scenario, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([scenario, a, before])
    work = tmp_path_factory.mktemp(f"{alt}_db")
    json_path = work / f"{alt}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{alt}.sqlite")
    migrate_database(url)
    return url, scenario


def _run(tmp_path_factory, url: str, scenario: str) -> dict:
    from flextool.engine_polars import run_chain_from_db

    steps = run_chain_from_db(
        url, scenario, work_folder=tmp_path_factory.mktemp(f"{scenario}_w"),
        keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


def _solve(tmp_path_factory, url: str, scenario: str):
    return list(_run(tmp_path_factory, url, scenario).values())[-1]


def _utf8(df: pl.DataFrame, cols=("p", "source", "sink")) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Utf8) for c in cols
                           if c in df.columns)


def _flow(step, source: str, sink: str) -> pl.Series:
    """Flow in MW (v_flow × unitsize), ordered by (d, t)."""
    us = float(_utf8(step.flex_data.p_unitsize.frame, ("p",))
               .filter(pl.col("p") == UNIT)["value"][0])
    rows = (_utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == UNIT) & (pl.col("source") == source)
                    & (pl.col("sink") == sink))
            .sort("d", "t"))
    assert rows.height > 0, f"no v_flow rows for {UNIT} {source}->{sink}"
    return rows["value"] * us


def _demand(node: str, mw: float) -> list:
    return ["node", node, "inflow", _b64(-mw, "float")]


def _no_ratio_constraint() -> list:
    return [["constraint", "coal_chp_fix", "is_enabled", _b64("no", "str")]]


def _gas(price: float, share: float | None = None) -> list:
    vals = [["node", GAS, "node_type", _b64("commodity", "str")],
            ["commodity", "gas", "price", _b64(price, "float")]]
    if share is not None:
        vals.append(["unit__inputNode", [UNIT, GAS], "input_share_max",
                     _b64(share, "float")])
    return vals


def _share(node: str, share: float) -> list:
    return ["unit__inputNode", [UNIT, node], "input_share_max",
            _b64(share, "float")]


def _build(fd):
    from polar_high import Problem

    from flextool.engine_polars import build_flextool

    pb = Problem()
    build_flextool(pb, fd)
    return pb


def _max_flow_arcs(pb) -> set[tuple[str, str, str]]:
    arcs: set[tuple[str, str, str]] = set()
    for rec in pb.cstrs_named("maxFlow"):
        if rec.name != "maxFlow":
            continue
        arcs |= {tuple(r) for r in _utf8(rec.over)
                 .select("p", "source", "sink").unique().iter_rows()}
    return arcs


# ── W: the greedy full-load fuel width ──────────────────────────────────
class _StubSource:
    """Minimal InputSource for the output-side coefficients."""

    def __init__(self, conv: dict, capcoef: dict) -> None:
        self._outs = pl.DataFrame({"unit": ["chp"] * len(conv),
                                   "node": list(conv)})
        self._params = {
            ("unit__outputNode", "conversion_flow_coeff"): pl.DataFrame(
                {"unit": ["chp"] * len(conv), "node": list(conv),
                 "value": list(conv.values())}),
            ("unit__outputNode", "capacity_max_coeff"): pl.DataFrame(
                {"unit": ["chp"] * len(capcoef), "node": list(capcoef),
                 "value": list(capcoef.values())}),
        }

    def entities(self, ec):
        if ec == "unit__outputNode":
            return self._outs
        raise KeyError(ec)

    def parameter(self, ec, pn):
        if (ec, pn) in self._params:
            return self._params[(ec, pn)]
        raise KeyError((ec, pn))


@pytest.mark.parametrize("conv, capcoef, expected", [
    # Extraction CHP: elec 2 (cap 0.8) filled first, heat 0.2 takes the
    # remaining 0.2 → 2·0.8 + 0.2·0.2 = 1.64 (the old Σ conv·capcoef
    # gave 1.8, ignoring the sum cap).
    ({"elec": 2.0, "heat": 0.2}, {"elec": 0.8, "heat": 1.0}, 1.64),
    # Default coefficients: one output at full capacity.
    ({"a": 1.0, "b": 1.0}, {}, 1.0),
    # conversion_flow_coeff = 0 outputs draw no fuel.
    ({"a": 0.5, "pass": 0.0}, {}, 0.5),
    # Caps summing below 1: every output at its own cap.
    ({"a": 3.0, "b": 1.0}, {"a": 0.2, "b": 0.3}, 3.0 * 0.2 + 1.0 * 0.3),
])
def test_fuel_width_greedy(conv, capcoef, expected) -> None:
    from flextool.engine_polars._derived_params import _indirect_fuel_width_lf

    got = dict(_indirect_fuel_width_lf(_StubSource(conv, capcoef))
               .with_columns(pl.col("p").cast(pl.Utf8)).collect()
               .iter_rows())
    assert got == pytest.approx({"chp": expected})


# ── (1) A fuel's conversion_flow_coeff never limits full output ─────────
@pytest.mark.parametrize("src_conv", [1.5, 0.5])
def test_single_fuel_reaches_full_output_with_default_share(
        tmp_path_factory, src_conv: float) -> None:
    """``conversion_flow_coeff`` on the input is the fuel energy per flow
    unit; 0.5 needs twice the flow, 1.5 two thirds of it.  With the default
    ``input_share_max = 1`` the input alone runs the unit at full output
    either way, and supplies exactly the conversion-equation fuel."""
    tag = str(src_conv).replace(".", "_")
    url, scen = _make_db(
        tmp_path_factory, alt=f"share_conv_{tag}",
        values=_no_ratio_constraint() + [
            _demand("west", CAPACITY), _demand("heat", 0.0),
            ["unit__inputNode", [UNIT, COAL], "conversion_flow_coeff",
             _b64(src_conv, "float")]])
    step = _solve(tmp_path_factory, url, scen)
    west = _flow(step, UNIT, "west")
    assert west.to_list() == pytest.approx([CAPACITY] * west.len(), rel=1e-6)
    fuel = _flow(step, COAL, UNIT)
    assert fuel.to_list() == pytest.approx(
        [CAPACITY / EFFICIENCY / src_conv] * fuel.len(), rel=1e-6)


# ── (2) Two fuels at 0.6 each: full output only with both ───────────────
def test_two_fuels_share_reach_full_output_together(tmp_path_factory) -> None:
    """Each fuel may supply at most 60 % of the full-load fuel.  With both
    available the unit reaches full output; the cheaper coal is pinned at
    its share and gas supplies the remaining 40 %."""
    url, scen = _make_db(
        tmp_path_factory, alt="two_fuels_both",
        entities=GAS_ENTITIES,
        values=_no_ratio_constraint() + _gas(30.0, share=0.6) + [
            _share(COAL, 0.6), _demand("west", CAPACITY),
            _demand("heat", 0.0)])
    step = _solve(tmp_path_factory, url, scen)
    west = _flow(step, UNIT, "west")
    n = west.len()
    assert west.to_list() == pytest.approx([CAPACITY] * n, rel=1e-6)
    full_fuel = CAPACITY / EFFICIENCY
    assert _flow(step, COAL, UNIT).to_list() == pytest.approx(
        [0.6 * full_fuel] * n, rel=1e-6)
    assert _flow(step, GAS, UNIT).to_list() == pytest.approx(
        [0.4 * full_fuel] * n, rel=1e-6)


def test_two_fuels_share_one_fuel_cannot_reach_full_output(
        tmp_path_factory) -> None:
    """Gas is priced out (dearer than the west slack penalty): coal alone
    may supply only 60 % of the full-load fuel, so west stops at 600 MW
    and the rest is unserved."""
    url, scen = _make_db(
        tmp_path_factory, alt="two_fuels_coal_only",
        entities=GAS_ENTITIES,
        values=_no_ratio_constraint() + _gas(1.0e5, share=0.6) + [
            _share(COAL, 0.6), _demand("west", CAPACITY),
            _demand("heat", 0.0)])
    step = _solve(tmp_path_factory, url, scen)
    west = _flow(step, UNIT, "west")
    n = west.len()
    assert west.to_list() == pytest.approx([0.6 * CAPACITY] * n, rel=1e-6)
    assert _flow(step, COAL, UNIT).to_list() == pytest.approx(
        [0.6 * CAPACITY / EFFICIENCY] * n, rel=1e-6)
    assert float(_flow(step, GAS, UNIT).abs().max()) == pytest.approx(
        0.0, abs=1e-6)


# ── Share 0.3 without investment: a variable bound, no maxFlow row ──────
def test_share_binds_without_investment_as_variable_bound(
        tmp_path_factory) -> None:
    """Constant capacity: the coal limit 0.3 · L · availability · capacity
    is the ``v_flow`` column upper bound and the fuel arc has no
    ``maxFlow`` row.  West (700 MW demand) is limited to 0.3 · 1000."""
    url, scen = _make_db(
        tmp_path_factory, alt="share_bound",
        values=_no_ratio_constraint() + [
            _share(COAL, 0.3), _demand("west", 700.0),
            _demand("heat", 0.0)])
    step = _solve(tmp_path_factory, url, scen)
    west = _flow(step, UNIT, "west")
    assert west.to_list() == pytest.approx([0.3 * CAPACITY] * west.len(),
                                           rel=1e-6)

    fd = step.flex_data
    us = float(_utf8(fd.p_unitsize.frame, ("p",))
               .filter(pl.col("p") == UNIT)["value"][0])
    slope = (_utf8(fd.p_slope.frame, ("p",)).filter(pl.col("p") == UNIT)
             ["value"].unique().to_list())
    assert slope == pytest.approx([1.0 / EFFICIENCY])
    pb = _build(fd)
    arcs = _max_flow_arcs(pb)
    assert (UNIT, COAL, UNIT) not in arcs, "bound-path fuel arc kept a row"
    assert (UNIT, UNIT, "west") in arcs

    v_flow = pb._vars["v_flow"]
    assert v_flow.has_elementwise_bounds
    upper = (_utf8(v_flow.frame).with_columns(ub=pl.Series(
        v_flow.col_upper())))
    fuel_ub = upper.filter((pl.col("p") == UNIT) & (pl.col("source") == COAL)
                           & (pl.col("sink") == UNIT))["ub"]
    # input_share_max · slope · W / src_conv · existing / unitsize
    # (W = 1: default output coefficients; availability 1).
    expected = 0.3 * (1.0 / EFFICIENCY) * 1.0 * CAPACITY / us
    assert fuel_ub.len() == fd.dt.height
    assert fuel_ub.to_list() == pytest.approx([expected] * fuel_ub.len(),
                                              rel=1e-12)
    out_ub = upper.filter((pl.col("p") == UNIT)
                          & (pl.col("source") == UNIT))["ub"]
    assert out_ub.is_infinite().all(), "output arcs must keep +inf bounds"


# ── Share 0.3 with investment: a maxFlow row scaling with built capacity ─
INV_WEST = 600.0


def test_share_binds_with_investment_and_scales_with_built_capacity(
        tmp_path_factory) -> None:
    """Greenfield CHP, coal (cheap) capped at 30 % of the full-load fuel,
    gas (dear) unrestricted.  The coal limit scales with BUILT capacity:
    the LP builds 600 / 0.3 = 2000 MW so that coal alone can fuel the
    600 MW west demand.  The fuel arc keeps its maxFlow row (invest
    variables present) and no variable bound."""
    url, scen = _make_db(
        tmp_path_factory, alt="share_row_invest",
        entities=GAS_ENTITIES,
        values=_no_ratio_constraint() + _gas(600.0) + [
            _share(COAL, 0.3), _demand("west", INV_WEST),
            _demand("heat", 0.0),
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
        ])
    step = _solve(tmp_path_factory, url, scen)
    inv = (_utf8(step.solution.value("v_invest_p"), ("p",))
           .filter(pl.col("p") == UNIT))
    assert float(inv["value"].sum()) == pytest.approx(INV_WEST / 0.3,
                                                      rel=1e-6)
    west = _flow(step, UNIT, "west")
    n = west.len()
    assert west.to_list() == pytest.approx([INV_WEST] * n, rel=1e-6)
    assert _flow(step, COAL, UNIT).to_list() == pytest.approx(
        [INV_WEST / EFFICIENCY] * n, rel=1e-6)
    assert float(_flow(step, GAS, UNIT).abs().max()) == pytest.approx(
        0.0, abs=1e-6)

    pb = _build(step.flex_data)
    arcs = _max_flow_arcs(pb)
    assert (UNIT, COAL, UNIT) in arcs and (UNIT, GAS, UNIT) in arcs
    v_flow = pb._vars["v_flow"]
    assert not v_flow.has_elementwise_bounds


# ── Multi-period investment: the fuel row follows the built capacity ────
def test_multi_period_investment_fuel_not_limited_below_need(
        tmp_path_factory) -> None:
    """``invest_period`` CHP over two periods (max 800 MW in p2020, 200 MW
    in p2025; demand west 600 + heat 300 = 900 MW).  The fuel row is
    ``fuel − L·Σinvest ≤ L·existing`` with ``L = slope · W``: fuel must
    follow the cumulative built capacity in every period.  (The row used
    to read ``fuel − Σinvest ≤ p_flow_upper``, whose RHS is the period's
    own investment limit — 200 MW · L in p2025.)"""
    period_cap = {"p2020": 800.0, "p2025": 200.0}
    url, scen = _make_db(
        tmp_path_factory, alt="share_multi_period",
        base_chain=["init", "west", "coal_chp", "heat", "2x5y"],
        values=_no_ratio_constraint() + [
            _demand("west", 600.0), _demand("heat", 300.0),
            ["unit", UNIT, "existing", _b64(0.0, "float")],
            ["unit", UNIT, "virtual_unitsize", _b64(1.0, "float")],
            ["unit", UNIT, "invest_method", _b64("invest_period", "str")],
            ["unit", UNIT, "invest_max_period",
             _b64({"index_type": "str", "index_name": "period",
                   "data": [[k, v] for k, v in period_cap.items()]},
                  "map")],
            ["unit", UNIT, "invest_cost", _b64(1.0, "float")],
            ["unit", UNIT, "lifetime", _b64(20.0, "float")],
            ["unit", UNIT, "discount_rate", _b64(0.05, "float")],
        ])
    step = _solve(tmp_path_factory, url, scen)
    inv = dict(_utf8(step.solution.value("v_invest_p"), ("p", "d"))
               .filter(pl.col("p") == UNIT).select("d", "value").iter_rows())
    assert sum(inv.values()) == pytest.approx(900.0, rel=1e-6)

    flows = _utf8(step.solution.value("v_flow"), ("p", "source", "sink", "d"))

    def per_period(source, sink):
        return dict(flows.filter((pl.col("p") == UNIT)
                                 & (pl.col("source") == source)
                                 & (pl.col("sink") == sink))
                    .group_by("d").agg(pl.col("value").min().alias("lo"),
                                       pl.col("value").max().alias("hi"))
                    .select("d", pl.struct("lo", "hi")).iter_rows())

    fuel = per_period(COAL, UNIT)
    heat = per_period(UNIT, "heat")
    west = per_period(UNIT, "west")
    assert fuel["p2025"]["lo"] == pytest.approx(900.0 / EFFICIENCY, rel=1e-6)
    assert heat["p2025"]["lo"] == pytest.approx(300.0, rel=1e-6)
    assert west["p2025"]["lo"] == pytest.approx(600.0, rel=1e-6)
    for d, v in fuel.items():
        assert v["hi"] == pytest.approx(
            (west[d]["hi"] + heat[d]["hi"]) / EFFICIENCY, rel=1e-6)

    pb = _build(step.flex_data)
    assert (UNIT, COAL, UNIT) in _max_flow_arcs(pb)


# ── Direct unit: an authored share caps the single flow variable ────────
def test_direct_unit_input_share_caps_flow(tmp_path_factory) -> None:
    """``coal_plant`` (direct, coal_market → west, existing 500 MW): an
    authored ``input_share_max = 0.5`` caps its flow at 0.5 · capacity
    (input ≤ 0.5 × full-load input ⇔ output ≤ 0.5 × capacity)."""
    url, scen = _make_db(
        tmp_path_factory, alt="share_direct",
        base_chain=["init", "west", "coal"],
        values=[["unit__inputNode", ["coal_plant", COAL], "input_share_max",
                 _b64(0.5, "float")]])
    step = _solve(tmp_path_factory, url, scen)
    fd = step.flex_data
    us = float(_utf8(fd.p_unitsize.frame, ("p",))
               .filter(pl.col("p") == "coal_plant")["value"][0])
    flow = (_utf8(step.solution.value("v_flow"))
            .filter(pl.col("p") == "coal_plant"))
    assert flow.height > 0
    assert float(flow["value"].max()) * us == pytest.approx(250.0, rel=1e-6)


# ── Rolling: roll sub-solves carry the same limits ───────────────────────
def test_rolling_sub_solves_apply_input_share_and_output_coeffs(
        tmp_path_factory) -> None:
    """Roll sub-solves (``<solve>_roll_<n>``) take the synthetic-solve
    loader path.  It must carry ``p_indirect_input_cap`` (extraction CHP:
    W = 2·0.8 + 0.2·0.2 = 1.64) and the output ``capacity_max_coeff``:
    with ``input_share_max = 0.5`` the coal flow is bounded by
    0.5 · 1.64 / 0.9 · 1000 MW in every roll and binds."""
    url, scen = _make_db(
        tmp_path_factory, alt="share_rolling",
        base_chain=CHP_CHAIN + ["coal_chp_extraction", "fullYear",
                                "dispatch_fullYear_roll"],
        values=_no_ratio_constraint() + [
            _share(COAL, 0.5),
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon",
             _b64(36.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump",
             _b64(24.0, "float")]])
    steps = _run(tmp_path_factory, url, scen)
    limit = 0.5 * (2.0 * 0.8 + 0.2 * 0.2) / EFFICIENCY
    rolls = [k for k in steps if "_roll_" in k]
    assert len(rolls) >= 2, f"expected roll sub-solves, got {list(steps)}"
    for name in rolls:
        step = steps[name]
        fd = step.flex_data
        assert fd.p_arc_max_cap_coef is not None, name
        cap = (_utf8(fd.p_indirect_input_cap.frame)
               .filter(pl.col("p") == UNIT))
        # rel 1e-6: the roll sub-solves read ``p_slope`` from the
        # solve_data CSV seed, written with 8 significant digits.
        assert cap["value"].to_list() == pytest.approx(
            [limit] * cap.height, rel=1e-6), name
        us = float(_utf8(fd.p_unitsize.frame, ("p",))
                   .filter(pl.col("p") == UNIT)["value"][0])
        # The roll sub-solves' p_flow_upper is the CSV-chain seed
        # (``derive_p_flow_max``): its fuel arc mirrors the native formula.
        pfu = (_utf8(fd.p_flow_upper.frame)
               .filter((pl.col("p") == UNIT) & (pl.col("source") == COAL)))
        assert pfu.height > 0, name
        assert pfu["value"].to_list() == pytest.approx(
            [limit * CAPACITY / us] * pfu.height, rel=1e-6), name
        fuel = _flow(step, COAL, UNIT)
        assert float(fuel.max()) == pytest.approx(limit * CAPACITY, rel=1e-6)
        west = _flow(step, UNIT, "west")
        assert float(west.max()) <= 0.8 * CAPACITY + 1e-6
