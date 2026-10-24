"""Dynamic and n-1 reserve requirements are reserve demand by themselves (#40).

A reserve's relationships ``(p, r, ud, n)`` get ``v_reserve`` (``prundt``) when ``(r, ud)`` has a requirement
(``_reserve_methods.demanded_reserves``): a non-zero ``reservation`` on some group, a dynamic group driven by an
``increase_reserve_ratio`` at one of its nodes, or an n-1 group with a ``large_failure_ratio`` at one of its
nodes.  Before, only a ``reservation`` activated a reserve, so a ``large_failure_only`` or ``dynamic_only``
reserve without a timeseries sibling had no ``v_reserve`` columns and empty requirement rows.

Fixtures: ``tests.json`` (``coal`` 500 MW, ``wind`` 1000 MW, ``reserve_n_1``: coal ``large_failure_ratio`` 0.1,
wind provider with reliability 0.6) plus override alternatives, built from JSON.  No timeseries "activator".
"""
from __future__ import annotations

import collections
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import (
    COAL, PLANT, PLANT_CHAIN, b64, build, flow, reserve, run, make_db, solve, utf8,
)

pytestmark = pytest.mark.solver

REL = "reserve__upDown__unit__node"
WIND = "wind_plant"


def _row_terms(pb, family: str) -> dict[str, dict[str, float]]:
    """``{row: {column: coefficient}}`` of the canonical rows of ``family``."""
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def _prundt_keys(fd, r: str | None = None) -> set[tuple[str, ...]]:
    pr = utf8(fd.prundt.select("p", "r", "ud", "n")).unique()
    if r is not None:
        pr = pr.filter(pl.col("r") == r)
    return set(pr.iter_rows())


def _group(r: str, ud: str, method: str, penalty: float = 1000.0) -> tuple[list, list]:
    grp = [r, ud, "electricity"]
    return ([["reserve", r, None], ["reserve__upDown__group", grp, None]],
            [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
             ["reserve__upDown__group", grp, "penalty_reserve", b64(penalty, "float")]])


def _rel(r: str, ud: str, unit: str, **params: float) -> tuple[list, list]:
    key = [r, ud, unit, "west"]
    vals = [[REL, key, "is_enabled", b64("yes", "str")]]
    vals += [[REL, key, name, b64(v, "float")] for name, v in params.items()]
    return [[REL, key, None]], vals


def _merge(*parts: tuple[list, list]) -> tuple[list, list]:
    ents: list = []
    vals: list = []
    for e, v in parts:
        ents += e
        vals += v
    return ents, vals


def _dynamic_only(r: str = "dyn", wind_ratio: float | None = 0.5, method: str = "dynamic_only"):
    """``r`` (up) on electricity: wind's output drives it (``increase_reserve_ratio`` ``wind_ratio``, None = no
    ratio); coal provides (``max_share`` 0.1 = 50 MW)."""
    wind_params = {} if wind_ratio is None else {"increase_reserve_ratio": wind_ratio}
    return _merge(_group(r, "up", method),
                  _rel(r, "up", WIND, **wind_params),
                  _rel(r, "up", PLANT, max_share=0.1, reliability=1.0))


# ---------------------------------------------------------------------------
# n-1 (large_failure_only) without reservation


@pytest.fixture(scope="module")
def n_1_only(tmp_path_factory):
    return solve(tmp_path_factory, alt="act_n1", base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"], values=[])


def test_large_failure_only_without_reservation_is_active(n_1_only) -> None:
    """The ``reserve_n_1`` alternative has no reservation: coal's failure (ratio 0.1) is the requirement and
    both relationships get v_reserve; the n-1 rows carry coal's flow; wind's reserve (reliability 0.6) plus the
    shortfall covers 10 % of coal's output at every step."""
    step = n_1_only
    fd = step.flex_data
    assert _prundt_keys(fd, "n_1") == {(PLANT, "n_1", "up", "west"), (WIND, "n_1", "up", "west")}
    lfr = set(utf8(fd.process_reserve_upDown_node_large_failure_ratio).iter_rows())
    assert (PLANT, "n_1", "up", "west") in lfr
    rows = _row_terms(build(fd), "reserveBalance_up_n_1_eq")
    assert len(rows) == fd.dt.height
    for row, terms in rows.items():
        assert row.startswith(f"reserveBalance_up_n_1_eq[n_1,up,electricity,{PLANT},"), row
        assert [c for c in terms if c.startswith(f"v_flow[{PLANT},{COAL},west,")], row
    from flextool.engine.model._reserve import reserve_shortfall_scale

    scale = utf8(reserve_shortfall_scale(fd).frame).filter(pl.col("r") == "n_1").sort("d", "t")["value"]
    vq = utf8(step.solution.value("vq_reserve")).filter(pl.col("r") == "n_1").sort("d", "t")["value"]
    covered = reserve(step, WIND, "n_1", "west") * 0.6 + vq * scale
    requirement = 0.1 * flow(step, PLANT, COAL, "west")
    assert requirement.max() > 1.0
    assert ((covered - requirement) >= -1e-6).all()


# ---------------------------------------------------------------------------
# dynamic_only without reservation


@pytest.fixture(scope="module")
def dynamic_only(tmp_path_factory):
    """(last step, work folder) of the dynamic_only model."""
    from flextool.engine import run_chain_from_db

    ents, vals = _dynamic_only()
    url, scen = make_db(tmp_path_factory, alt="act_dyn", base_chain=PLANT_CHAIN + ["wind"], entities=ents,
                        values=vals)
    wf = tmp_path_factory.mktemp("act_dyn_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    return list(steps.values())[-1], wf


def test_dynamic_only_without_reservation_is_active(dynamic_only) -> None:
    """Wind's output (ratio 0.5) drives a dynamic_only requirement with no reservation anywhere: the reserve's
    relationships get v_reserve and every reserveBalance_dynamic_eq row has wind's v_flow and coal's
    v_reserve."""
    fd = dynamic_only[0].flex_data
    assert _prundt_keys(fd, "dyn") == {(PLANT, "dyn", "up", "west"), (WIND, "dyn", "up", "west")}
    rows = _row_terms(build(fd), "reserveBalance_dynamic_eq")
    assert len(rows) == fd.dt.height
    for row, terms in rows.items():
        assert [c for c in terms if c.startswith(f"v_flow[{WIND},{WIND},west,")], row
        assert [c for c in terms if c.startswith(f"v_reserve[{PLANT},dyn,up,west,")], row


def test_dynamic_only_reports_reserve_price(dynamic_only) -> None:
    """The dynamic_only group's reserve_price_t is the dual of its reserveBalance_dynamic_eq row (one row per
    group and step), after the period scaling the timeseries price gets."""
    from flextool.lean_parquet import read_lean_parquet

    step, wf = dynamic_only
    from flextool.process_outputs.read_highs_solution import (
        _load_complete_period_share_of_year, _load_inflation_factor,
    )

    duals = step.solution.constraint_dual("reserveBalance_dynamic_eq")
    duals = (duals.with_columns(pl.col("key").str.split(",").alias("__k"))
             .with_columns(pl.col("__k").list.get(0).alias("r"), pl.col("__k").list.get(3).alias("d"),
                           pl.col("__k").list.get(4).alias("t"))
             .filter(pl.col("r") == "dyn").sort("d", "t"))
    assert duals.height == step.flex_data.dt.height
    assert (duals["dual"].abs() > 1e-9).any(), "the requirement must bind somewhere for the test to mean something"
    files = list(Path(wf).rglob(f"v_dual_reserve__upDown__group__period__t__{step.solve_name}.parquet"))
    assert files, "v_dual_reserve parquet not written"
    price = read_lean_parquet(files[0])
    cols = [c for c in price.columns if tuple(str(x) for x in c) == ("dyn", "up", "electricity")]
    assert len(cols) == 1, price.columns
    share = _load_complete_period_share_of_year(Path(wf), flex_data=step.flex_data)
    infl = _load_inflation_factor(Path(wf), flex_data=step.flex_data)
    got = {(str(idx[1]), str(idx[2])): float(v) for idx, v in price[cols[0]].items()}
    for d, t, dual in duals.select("d", "t", "dual").iter_rows():
        expected = dual * share.get(d, 1.0) / infl.get(d, 1.0)
        assert got[(d, t)] == pytest.approx(expected, rel=1e-9, abs=1e-12), (d, t)


# ---------------------------------------------------------------------------
# no driver → inactive


def _no_driver(case: str) -> tuple[list, list]:
    if case == "n_1_without_ratio":
        return _merge(_group("nx", "up", "large_failure_only"),
                      _rel("nx", "up", PLANT, max_share=0.1, reliability=1.0),
                      _rel("nx", "up", WIND, max_share=0.1, reliability=1.0))
    if case == "dynamic_without_ratio":
        return _dynamic_only("nx", wind_ratio=None)
    assert case == "no_reserve_method"
    return _dynamic_only("nx", method="no_reserve")


@pytest.mark.parametrize("case", ["n_1_without_ratio", "dynamic_without_ratio", "no_reserve_method"])
def test_requirement_without_driver_stays_inactive(tmp_path_factory, case: str) -> None:
    """A large_failure_only group without a large_failure_ratio, a dynamic_only group without an
    increase_reserve_ratio, and a no_reserve group (whose relationship has a ratio) have no requirement: no
    v_reserve for the reserve and no requirement rows that use it."""
    ents, vals = _no_driver(case)
    step = solve(tmp_path_factory, alt=f"act_none_{case}", base_chain=PLANT_CHAIN + ["wind"], entities=ents,
                 values=vals)
    fd = step.flex_data
    assert not _prundt_keys(fd, "nx")
    pb = build(fd)
    for family in ("reserveBalance_up_n_1_eq", "reserveBalance_dynamic_eq"):
        for row, terms in _row_terms(pb, family).items():
            assert not [c for c in terms if c.startswith("v_reserve[")], row


# ---------------------------------------------------------------------------
# the solve_data producer and the FlexData producer agree


def test_csv_and_flexdata_activation_agree(tmp_path_factory) -> None:
    """One n-1 reserve driven by coal's large_failure_ratio and one dynamic_only reserve without a driver, no
    reservation anywhere: the solve_data ``prundt.csv`` (which filters the ratio sets the model reads) and
    ``flex_data.prundt`` activate the same relationships."""
    ents, vals = _no_driver("dynamic_without_ratio")
    url, scen = make_db(tmp_path_factory, alt="act_agree", base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"],
                        entities=ents, values=vals)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    provider = step.flex_data_provider
    assert provider is not None
    csv = provider.get("solve_data/prundt")
    assert csv is not None
    csv_keys = set(csv.select(pl.col(c).cast(pl.Utf8) for c in ("process", "reserve", "upDown", "node"))
                   .unique().iter_rows())
    fd_keys = _prundt_keys(step.flex_data)
    assert csv_keys == fd_keys
    assert fd_keys == {(PLANT, "n_1", "up", "west"), (WIND, "n_1", "up", "west")}
