"""The non-synchronous limit and the capacity margin read the real demand, not the energy-margined one.

The invest-solve energy margin (``energy_margin_multiplier`` / ``energy_margin_adder``) is an adequacy
device: it adds demand to the node balance only.  The group rows measure the group with the real demand
(owner decision 2026-10-25):

    non_sync_constraint   positive_inflow / negative_inflow = max(inflow_real, 0) / min(inflow_real, 0)
    capacityMargin        … ≥ capacity_margin · icap − Σ_n inflow_real[n,d,t] / Δt[d,t] · icap − …

so a model with a margin has the same group rows as the model without it, and a different node balance.
Checked in a single invest solve, a nested invest + dispatch chain, a stochastic invest solve and after a
``--csv-dump`` dump (the reload is in ``tests/csv_reload/``).  Every database is a JSON fixture plus an override
alternative.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INVEST_CHAIN = ["init", "west", "wind", "battery", "battery_invest", "5weeks", "coal"]
GROUP_FAMILIES = ("non_sync_constraint", "capacityMargin")


def _group(g: str, nodes: list[str], *, flagged: list[tuple[str, str]], capacity_margin: bool = True,
           extra_nodes: tuple[str, ...] = ()) -> tuple[list, list]:
    """Group ``g`` with a non-synchronous limit 0.5 (and a manual capacity margin 0) on ``nodes``."""
    ents = [["node", n, None] for n in extra_nodes]
    ents += [["group", g, None]] + [["group__node", [g, n], None] for n in nodes]
    vals = [["node", n, "node_type", b64("balance", "str")] for n in extra_nodes]
    vals += [["group", g, "has_non_synchronous", b64("yes", "str")],
             ["group", g, "non_synchronous_limit", b64(0.5, "float")]]
    if capacity_margin:
        vals += [["group", g, "capacity_margin_method", b64("manual", "str")],
                 ["group", g, "capacity_margin", b64(0.0, "float")]]
    vals += [["unit__outputNode", [u, n], "is_non_synchronous", b64("yes", "str")] for u, n in flagged]
    return ents, vals


def _multiplier(node: str, factor: float) -> list:
    return [["node", node, "energy_margin_method", b64("inflow_multiplier", "str")],
            ["node", node, "energy_margin_multiplier", b64(factor, "float")]]


def _adder(node: str, mw: float) -> list:
    return [["node", node, "energy_margin_method", b64("inflow_adder", "str")],
            ["node", node, "energy_margin_adder", b64(mw, "float")]]


def _steps(tmp_path_factory, alt: str, ents: list, vals: list, chain: list[str],
           fixture_json: Path | None = None) -> dict:
    kw = {"fixture_json": fixture_json} if fixture_json is not None else {}
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=chain, entities=ents, **kw)
    return run(tmp_path_factory, url, scen)


def _rows(fd) -> dict[str, dict]:
    pb = build(fd)
    out = {}
    for fam in GROUP_FAMILIES + ("nodeBalance_eq",):
        out[fam] = {"bounds": row_bounds(pb, fam), "terms": row_terms(pb, fam)}
    return out


def _assert_group_rows_equal(with_margin: dict, without: dict, solve: str) -> None:
    for fam in GROUP_FAMILIES:
        a, b = with_margin[fam], without[fam]
        assert set(a["bounds"]) == set(b["bounds"]), (solve, fam)
        for row, (lo, hi) in a["bounds"].items():
            assert lo == pytest.approx(b["bounds"][row][0], rel=1e-12, abs=1e-9), (solve, row)
            assert hi == pytest.approx(b["bounds"][row][1], rel=1e-12, abs=1e-9), (solve, row)
        for row, terms in a["terms"].items():
            assert terms.keys() == b["terms"][row].keys(), (solve, row)
            for col, v in terms.items():
                assert v == pytest.approx(b["terms"][row][col], rel=1e-12, abs=1e-12), (solve, row, col)


def _balance_differs(with_margin: dict, without: dict, node: str) -> bool:
    a, b = with_margin["nodeBalance_eq"]["bounds"], without["nodeBalance_eq"]["bounds"]
    return any(abs(a[r][0] - b[r][0]) > 1e-6 or abs(a[r][1] - b[r][1]) > 1e-6
               for r in a if r.startswith(f"nodeBalance_eq[{node},") and r in b)


# ---------------------------------------------------------------------------
# single invest solve (tests.json west + wind + battery invest + coal, 5 weeks)


def test_multiplier_not_in_group_rows(tmp_path_factory) -> None:
    """energy_margin_multiplier 1.1 on west: the non-sync and capacity-margin rows of {west} equal those of the
    model without the margin (t0001: non-sync ub 0.5 · 589 = 294.5, capacity-margin lb −291.0), the node
    balance of west is margined."""
    ents, vals = _group("gm", ["west"], flagged=[("wind_plant", "west")])
    base = list(_steps(tmp_path_factory, "rd_mult0", ents, vals, INVEST_CHAIN).values())[-1]
    marg = list(_steps(tmp_path_factory, "rd_mult", ents, vals + _multiplier("west", 1.1),
                       INVEST_CHAIN).values())[-1]
    assert marg.flex_data.p_inflow_unmargined is not None
    a, b = _rows(marg.flex_data), _rows(base.flex_data)
    _assert_group_rows_equal(a, b, marg.solve_name)
    assert _balance_differs(a, b, "west"), "the margin must reach the node balance"
    assert a["non_sync_constraint"]["bounds"]["non_sync_constraint[gm,p2020,t0001]"][1] == pytest.approx(294.5)
    assert a["capacityMargin"]["bounds"]["capacityMargin[gm,p2020,t0001]"][0] == pytest.approx(-291.0)


def test_adder_not_in_group_rows(tmp_path_factory) -> None:
    """energy_margin_adder 50 on spare (no inflow) and on spare2 (inflow +30, margined −20), both in the group:
    the group rows equal those without the adders (spare2's +30 counts as non-synchronous infeed, its added
    demand nowhere), the node balances of spare and spare2 are margined."""
    ents, vals = _group("gm", ["west", "spare", "spare2"], flagged=[("wind_plant", "west")],
                        extra_nodes=("spare", "spare2"))
    vals = vals + [["node", "spare2", "inflow", b64(30.0, "float")],
                   ["node", "spare", "penalty_up", b64(10000.0, "float")],
                   ["node", "spare2", "penalty_up", b64(10000.0, "float")]]
    base = list(_steps(tmp_path_factory, "rd_add0", ents, vals, INVEST_CHAIN).values())[-1]
    marg = list(_steps(tmp_path_factory, "rd_add", ents, vals + _adder("spare", 50.0) + _adder("spare2", 50.0),
                       INVEST_CHAIN).values())[-1]
    a, b = _rows(marg.flex_data), _rows(base.flex_data)
    _assert_group_rows_equal(a, b, marg.solve_name)
    assert _balance_differs(a, b, "spare") and _balance_differs(a, b, "spare2")
    # spare2's real inflow +30 MWh per step is infeed (LHS constant 30 · icap → ub lowered by it)
    icap = {(g, d): v for g, d, v in utf8(marg.flex_data.p_inv_group_cap.frame).select("g", "d", "value").iter_rows()}
    real = {(n, d, t): v for n, d, t, v in utf8(marg.flex_data.p_inflow_unmargined.frame)
            .select("n", "d", "t", "value").iter_rows()}
    ub = a["non_sync_constraint"]["bounds"]["non_sync_constraint[gm,p2020,t0001]"][1]
    expected = icap[("gm", "p2020")] * (0.5 * -min(real[("west", "p2020", "t0001")], 0.0) - 30.0)
    assert ub == pytest.approx(expected, rel=1e-9)


def test_no_margin_frames_from_p_inflow(tmp_path_factory) -> None:
    """Without a margin there is no unmargined record and the three group frames are the helpers applied
    to p_inflow (the default is unchanged)."""
    from flextool.engine.preprocess.params import _derived as drv

    ents, vals = _group("gm", ["west"], flagged=[("wind_plant", "west")])
    fd = list(_steps(tmp_path_factory, "rd_none", ents, vals, INVEST_CHAIN).values())[-1].flex_data
    assert fd.p_inflow_unmargined is None
    assert fd.p_positive_inflow.frame.equals(drv.p_positive_inflow_from_inflow(fd.p_inflow).frame)
    assert fd.p_negative_inflow.frame.equals(drv.p_negative_inflow_from_inflow(fd.p_inflow).frame)
    assert fd.pdtNodeInflow_per_step.frame.equals(
        drv.pdtNodeInflow_per_step_from_inflow(fd.p_inflow, fd.p_step_duration).frame)


# ---------------------------------------------------------------------------
# nested invest + storage + dispatch rolls; stochastic invest


@pytest.mark.slow
def test_nested_chain_group_rows(tmp_path_factory) -> None:
    """``multi_fullYear_battery_nested_24h_invest_one_solve`` with multiplier 1.5 on west and a non-sync group
    {west}: every sub-solve (invest, storage, dispatch rolls — the latter on the synthetic-solve path) has the
    non-sync rows of the chain without the margin; the invest sub-solve's node balance is margined."""
    data_chain = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest",
                  "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
                  "storage_fullYear_6h", "invest_24h", "bind_battery_forward", "unlimited_investment"]
    ents, vals = _group("gw", ["west"], flagged=[("wind_plant", "west")], capacity_margin=False)
    base = _steps(tmp_path_factory, "rd_nest0", ents, vals, data_chain)
    marg = _steps(tmp_path_factory, "rd_nest", ents, vals + _multiplier("west", 1.5), data_chain)
    assert list(base) == list(marg) and len(marg) >= 3, list(marg)
    margined = 0
    for name, step in marg.items():
        a, b = _rows(step.flex_data), _rows(base[name].flex_data)
        assert a["non_sync_constraint"]["bounds"], name
        _assert_group_rows_equal(a, b, name)
        margined += _balance_differs(a, b, "west")
    assert margined >= 1, "the invest sub-solve must see the margin"
    assert margined < len(marg), "the dispatch sub-solves must not see the margin"


def test_stochastic_invest_group_rows(tmp_path_factory) -> None:
    """``stoch_two_period_invest.json`` ``recourse`` with multiplier 1.2 on elec and non-sync + capacity
    margin on {elec}: the group rows of every (branch period, t) equal those without the margin; the node
    balance is margined."""
    fx = FIXTURES / "stoch_two_period_invest.json"
    ents, vals = _group("ge", ["elec"], flagged=[("wind", "elec")])
    chain = ["init", "base", "stoch", "recourse"]
    base = _steps(tmp_path_factory, "rd_sto0", ents, vals, chain, fixture_json=fx)
    marg = _steps(tmp_path_factory, "rd_sto", ents, vals + _multiplier("elec", 1.2), chain, fixture_json=fx)
    assert list(base) == list(marg)
    seen = 0
    for name, step in marg.items():
        a, b = _rows(step.flex_data), _rows(base[name].flex_data)
        _assert_group_rows_equal(a, b, name)
        seen += _balance_differs(a, b, "elec")
        periods = {r[r.index("[") + 1:-1].split(",")[-2] for r in a["non_sync_constraint"]["bounds"]}
        assert len(periods) > 1, (name, periods)
    assert seen >= 1, "the invest solve must see the margin"


# ---------------------------------------------------------------------------
# --csv-dump reload


def test_csv_dump_writes_real_demand(tmp_path_factory) -> None:
    """With a margin, ``dump_csvs`` of the solved FlexData writes ``pdtNodeInflow_unmargined.csv``, holding the
    real inflow.  The reload half is ``tests/csv_reload/test_group_limits_real_demand.py`` (c24 S1-8)."""
    import polars as pl

    from flextool.engine import run_chain_from_db

    ents, vals = _group("gm", ["west"], flagged=[("wind_plant", "west")])
    url, scen = make_db(tmp_path_factory, alt="rd_dump", values=vals + _multiplier("west", 1.1),
                        base_chain=INVEST_CHAIN, entities=ents)
    wf = tmp_path_factory.mktemp("rd_dump_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    original = list(steps.values())[-1].flex_data
    assert original.p_inflow_unmargined is not None
    out = tmp_path_factory.mktemp("rd_dumped")
    original.dump_csvs(out, include_heavy=True)      # the FlexData fields only
    path = out / "solve_data" / "pdtNodeInflow_unmargined.csv"
    assert path.exists(), "dump_csvs must write the real inflow"
    assert pl.read_csv(path).height == original.p_inflow_unmargined.frame.height

