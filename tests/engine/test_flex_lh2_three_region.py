"""polar_high lh2_three_region parity — JSON-fixture scenario imported from
flextool's tests/fixtures/lh2_three_region.json (regen via
tests/_gen_lh2_three_region.py).

The h2_* / lh2_* nodes are in the 24-h ``daily_group``: each is balanced
once per day by nodeBalanceCoarse_eq, the sum of its hourly balance terms
(every flow lives on the hourly steps).
"""
from pathlib import Path

import polars as pl

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import load_scenario
import pytest

pytestmark = pytest.mark.solver


def _flextool_obj(work: Path) -> float:
    sc_file = work / "solve_data" / "solve_current.csv"
    if sc_file.exists():
        solve = pl.read_csv(sc_file)["solve"][0]
        parq = work / "output_raw" / f"v_obj__{solve}.parquet"
        if parq.exists():
            return pl.read_parquet(parq)["objective"][0]
    parq = list(work.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(sorted(parq)[-1])["objective"][0]
    return pl.read_csv(work / "output_raw" / "v_obj.csv")["objective"][0]


def test_lh2_three_region_parity(scenario_run):
    run = scenario_run("lh2_three_region", db_fixture="lh2")
    work = run.work_folder
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    # Γ.4: prefer golden_obj.json if present, fall back to parquet.
    from _golden import has_golden, assert_obj_within
    if has_golden(work):
        assert_obj_within(sol.obj, work)
        return
    flextool_obj = _flextool_obj(work)
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"obj mismatch: polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


def test_coarse_node_dual_broadcast(scenario_run):
    """``write_v_dual_node_balance`` reports per-(d,t) prices for the
    coarse-group nodes (``nodeCoarse``: h2_A/B/C, lh2_A/B/C here).

    These nodes are balanced by ``nodeBalanceCoarse_eq`` (one row per
    coarse step), NOT the per-(d,t) ``nodeBalance_eq``.  The writer must
    extract that dual (same orientation as ``nodeBalance_eq``) and
    broadcast it to every timestep of the node's coarse step, else the
    downstream ``node_prices_dt_e`` output (out_node) raises a KeyError
    indexing the missing columns.

    Scale-independent checks (the harness builds the LP unscaled):
      * every ``nodeCoarse`` node has a dual column;
      * each coarse node's price is constant within each of its coarse steps;
      * the writer output equals an independent re-derivation that maps
        each coarse row's dual to its timesteps via ``node_coarse_step``;
      * every node ``node_prices_dt_e`` requests
        (``node_balance \\ node_state``) has a dual column.
    """
    import numpy as np
    import pandas as pd

    run = scenario_run("lh2_three_region", db_fixture="lh2")
    work = run.work_folder
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(keep_solver=True)
    assert sol.optimal
    h = sol.highs

    from flextool.process_outputs.read_highs_solution import (
        extract_variable, write_v_dual_node_balance,
        _resolve_inv_scale_the_objective,
    )
    from flextool.engine.output._prices import price_weight_dt
    from flextool.process_outputs.read_sets import read_sets

    out = work / "output_raw"
    sd = work / "solve_data"
    sc = sd / "solve_current.csv"
    solve = pl.read_csv(sc)["solve"][0] if sc.exists() else "solve"
    rd = sd / "realized_dispatch.csv"

    # Item 163: the writer and the oracle's extractor no longer read
    # ``solve_data/scale_the_objective.csv`` off disk — ``build_flextool(pb,
    # data)`` built this LP at the default ``scale_the_objective=1.0``
    # (``core.py`` l.336), so that is the scale to pass explicitly, not the
    # (removed) work-folder read, which only happened to cancel out before.
    p = write_v_dual_node_balance(
        h, solve_name=solve, output_dir=out,
        realized_dispatch_csv=rd, flex_data=data,
        scale_the_objective=1.0,
    )
    dual = pd.read_parquet(p).set_index(["solve", "period", "time"])
    cols = set(dual.columns)

    coarse_nodes = sorted(data.nodeCoarse["n"].cast(pl.Utf8).to_list())
    assert coarse_nodes == ["h2_A", "h2_B", "h2_C", "lh2_A", "lh2_B", "lh2_C"]
    missing = [n for n in coarse_nodes if n not in cols]
    assert not missing, f"coarse nodes missing from dual: {missing}"

    # Independent re-derivation: map each coarse row's dual to its t's, per node, then ÷ the step weight
    # tw · inflation / period_share (+dual: the coarse row has the orientation of nodeBalance_eq).
    inv = _resolve_inv_scale_the_objective(scale_the_objective=1.0)
    raw = extract_variable(
        h, "nodeBalanceCoarse_eq", ("node",), solve_name=solve,
        has_time=True, source="row_dual", value_scale=inv,
        realized_dispatch_csv=rd, flex_data=data,
    )
    ncs = data.node_coarse_step.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "b_first", "t")).to_pandas()
    long = raw.stack(future_stack=True).rename("v").reset_index()
    exp = (
        long.merge(ncs, left_on=["node", "period", "time"],
                   right_on=["n", "d", "b_first"], how="inner")
        .pivot_table(index=["solve", "period", "t"], columns="node",
                     values="v", aggfunc="first")
    )
    exp.index = exp.index.set_names(["solve", "period", "time"])
    w = {(d, t): v for d, t, v in price_weight_dt(data).iter_rows()}
    exp = exp.div(pd.Series([w[(d, t)] for _s, d, t in exp.index], index=exp.index), axis=0)

    a = dual[coarse_nodes].sort_index()
    b = exp[coarse_nodes].reindex(a.index)
    assert np.nanmax(np.abs(a.values - b.values)) < 1e-9, (
        "writer coarse prices disagree with node_coarse_step re-derivation"
    )

    # Constant within each coarse step.
    for (n, d, _bf), grp in ncs.groupby(["n", "d", "b_first"]):
        sub = dual.loc[(slice(None), d, grp["t"].tolist()), n]
        assert sub.max() - sub.min() < 1e-9

    # Every node node_prices_dt_e will request has a dual column.
    s = read_sets(data, sol, solve_name=solve)
    price_nodes = s.node_balance.difference(s.node_state)
    assert set(price_nodes) <= cols, (
        f"node_prices nodes without a dual column: "
        f"{sorted(set(price_nodes) - cols)}"
    )
    # The non-storage coarse nodes (h2_*) are among them.
    assert {"h2_A", "h2_B", "h2_C"} <= set(price_nodes)
