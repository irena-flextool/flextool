"""polar_high lh2_three_region parity — JSON-fixture scenario imported from
flextool's tests/fixtures/lh2_three_region.json (regen via
tests/_gen_lh2_three_region.py).

Closes at machine epsilon via:
  * arc-side block-aware aggregation in nodeBalanceBlock_eq (per-arc
    weight = block_step_duration of the relevant side block).
  * arc-block-compatibility filter on flow_to_n / flow_from_n /
    flow_from_nodeBalance_*: arcs whose side-block doesn't connect to
    a node's block via overlap_set are dropped from that node's
    nodeBalance_eq (the .mod's process_side_block restriction).
"""
from pathlib import Path

import polars as pl

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
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


def test_lh2_three_region_parity(scenario_workdir):
    work = scenario_workdir("lh2_three_region", db_fixture="lh2")
    data = load_flextool(work)
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


def test_nodebalanceblock_dual_broadcast(scenario_workdir):
    """``write_v_dual_node_balance`` reports per-(d,t) prices for the
    coarse-resolution ``nodeStateBlock`` nodes (h2_A/B/C, lh2_A/B/C here).

    These nodes are balanced by ``nodeBalanceBlock_eq`` (one row per
    block), NOT the per-(d,t) ``nodeBalance_eq``.  The writer must extract
    that block dual and broadcast it to every timestep in the block, else
    the downstream ``node_prices_dt_e`` output (out_node) raises a
    KeyError indexing the missing columns.

    Scale-independent checks (the harness builds the LP unscaled, so
    absolute magnitudes are not validated here — the scaling convention is
    shared verbatim with the already-trusted ``nodeBalance_eq`` path):
      * every ``nodeStateBlock`` node has a dual column;
      * each block node's price is constant within each block;
      * the writer output equals an independent re-derivation that maps
        each block's ``b_first`` dual to its timesteps via
        ``period_block_time`` (i.e. each block keeps its own value — no
        collapse/cross-block bleed);
      * every node ``node_prices_dt_e`` requests
        (``node_balance \\ node_state``) now has a dual column.
    """
    import numpy as np
    import pandas as pd

    work = scenario_workdir("lh2_three_region", db_fixture="lh2")
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(keep_solver=True)
    assert sol.optimal
    h = sol.highs

    from flextool.process_outputs.read_highs_solution import (
        extract_variable, write_v_dual_node_balance,
        _resolve_inv_scale_the_objective, _divide_by_inflation_and_row_scaler,
    )
    from flextool.process_outputs.read_sets import read_sets

    out = work / "output_raw"
    sd = work / "solve_data"
    sc = sd / "solve_current.csv"
    solve = pl.read_csv(sc)["solve"][0] if sc.exists() else "solve"
    rd = sd / "realized_dispatch.csv"

    p = write_v_dual_node_balance(
        h, solve_name=solve, output_dir=out,
        realized_dispatch_csv=rd, flex_data=data,
    )
    dual = pd.read_parquet(p).set_index(["solve", "period", "time"])
    cols = set(dual.columns)

    block_nodes = sorted(data.nodeStateBlock["n"].to_list())
    assert block_nodes, "fixture must have nodeStateBlock nodes"
    missing = [n for n in block_nodes if n not in cols]
    assert not missing, f"block nodes missing from dual: {missing}"

    # Independent re-derivation: map each block's b_first dual to its t's.
    inv = _resolve_inv_scale_the_objective(sd)
    blk = extract_variable(
        h, "nodeBalanceBlock_eq", ("node",), solve_name=solve,
        has_time=True, source="row_dual", value_scale=-inv,
        realized_dispatch_csv=rd, flex_data=data,
    )
    blk = _divide_by_inflation_and_row_scaler(
        blk, wf=sd, solve_name=solve, flex_data=data,
    )
    pbt = data.period_block_time.select(["d", "b_first", "t"]).to_pandas()
    blk_long = blk.stack(future_stack=True).rename("v").reset_index()
    exp = (
        blk_long.merge(pbt, left_on=["period", "time"],
                       right_on=["d", "b_first"], how="inner")
        .pivot_table(index=["solve", "period", "t"], columns="node",
                     values="v", aggfunc="first")
    )
    exp.index = exp.index.set_names(["solve", "period", "time"])

    a = dual[block_nodes].sort_index()
    b = exp[block_nodes].reindex(a.index)
    assert np.nanmax(np.abs(a.values - b.values)) < 1e-9, (
        "writer block prices disagree with period_block_time re-derivation"
    )

    # Constant within each block.
    for (d, _bf), grp in pbt.groupby(["d", "b_first"]):
        sub = dual.loc[(slice(None), d, grp["t"].tolist()), block_nodes]
        assert (sub.max() - sub.min()).abs().max() < 1e-9

    # Every node node_prices_dt_e will request now has a dual column.
    s = read_sets(data, sol, solve_name=solve)
    price_nodes = s.node_balance.difference(s.node_state)
    assert set(price_nodes) <= cols, (
        f"node_prices nodes without a dual column: "
        f"{sorted(set(price_nodes) - cols)}"
    )
    # The non-storage block nodes (h2_*) are among them.
    assert {"h2_A", "h2_B", "h2_C"} <= set(price_nodes)
