"""energy_margin — invest-only, negative-inflow demand margin emitter.

Runs per-solve from ``_emit_solve_time.run`` immediately AFTER
``emit_pdtNodeInflow`` (batch 54) and BEFORE the positive/negative inflow
split (batch 58), so that split inherits the margin.

FlexTool sign convention: DEMAND is NEGATIVE inflow (the exogenous
outflow / ``p_negative_inflow`` term); SUPPLY / generation is POSITIVE
inflow.  The energy_margin feature inflates demand in the invest solve, so
it scales the negative (demand) rows.

What it does
------------
For a node with ``energy_margin_method == "inflow_multiplier"`` and an effective
``energy_margin`` factor ``!= 1.0``, multiply that node's ``pdtNodeInflow``
rows by the factor — but ONLY:

* in the solve that carries investment periods
  (``bool(state.solve.invest_periods.get(solve_name))``); and
* on NEGATIVE (demand) inflow rows (``value < 0``) — a ``> 1`` factor makes
  the demand row MORE negative (larger demand) and must never scale a
  positive/zero (supply / net-zero) row.

The dispatch solves see the true (un-margined) demand.

Early-return byte-parity contract
----------------------------------
If the solve is NOT an invest solve, OR no node has
``energy_margin_method == "inflow_multiplier"`` with an effective factor ``!= 1.0``,
the emitter returns WITHOUT touching the provider — the batch-54
``pdtNodeInflow`` frame stands untouched, so the default (nobody sets a
margin) is byte-identical to today.

Invariant #5 (byte-parity)
--------------------------
The re-emitted ``value`` column is rendered with the ``repr()``-based
:func:`_render_value_column`, NEVER ``.cast(Utf8)`` — a cast diverges from
``repr`` on sci-notation padding and ``NaN``.  Only the scaled rows are
re-rendered; every other row keeps its original ``value`` string verbatim,
so unchanged rows are byte-identical by construction.
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._emit_inflow_scaling import (
    _read_keyed_float,
    _read_pairs,
)
from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars._vectorize import _render_value_column


def emit_energy_margin_inflow(
    state,
    solve_name,
    input_dir,
    solve_data_dir,
    *,
    provider,
) -> None:
    """Apply the invest-only, demand-only energy_margin to pdtNodeInflow.

    See the module docstring for the full contract.  A missing/empty
    method or value frame yields no manual nodes ⇒ early return.
    """
    # 1. node → method and node → margin factor.
    method_pairs = _read_pairs(
        input_dir / "node__energy_margin_method.csv", provider=provider,
    )
    method_for_node = {n: m for (n, m) in method_pairs}
    margin_for_node = _read_keyed_float(
        input_dir / "energy_margin.csv", provider=provider,
    )

    # 2. manual = {node: margin}; margin = value if present else 1.0; drop
    #    entries whose margin == 1.0 (a no-op).
    manual: dict[str, float] = {}
    for node, method in method_for_node.items():
        if method != "inflow_multiplier":
            continue
        margin = margin_for_node.get(node, 1.0)
        if margin == 1.0:
            continue
        manual[node] = margin

    # 3. Invest-only gate + no-manual-node gate → early return (byte-parity).
    is_invest = bool(state.solve.invest_periods.get(solve_name))
    if not is_invest or not manual:
        return

    # 4. Read the pdtNodeInflow frame from the SAME provider key
    #    emit_pdtNodeInflow wrote it under.  Columns: node, period, time,
    #    value — value is Utf8 (repr-rendered floats).
    key = _provider_key(solve_data_dir / "pdtNodeInflow.csv")
    df = provider.get(key)
    if df is None or df.height == 0:
        return

    # 5. Multiply matching nodes' NEGATIVE-value (demand) rows by their
    #    margin; leave every other row (other nodes, and positive/zero supply
    #    rows) verbatim.  Parse value → float, join the per-node margin (1.0
    #    for non-manual), render the scaled float via repr, and select it only
    #    where the row is a manual node AND value < 0.
    work = df.with_columns(
        pl.col("value").cast(pl.Float64).alias("__vf"),
        pl.col("node")
        .replace_strict(manual, default=1.0, return_dtype=pl.Float64)
        .alias("__margin"),
    )
    scaled_str = _render_value_column(work["__vf"] * work["__margin"])
    work = work.with_columns(scaled_str.alias("__scaled"))
    new_frame = work.with_columns(
        pl.when((pl.col("__margin") != 1.0) & (pl.col("__vf") < 0.0))
        .then(pl.col("__scaled"))
        .otherwise(pl.col("value"))
        .alias("value"),
    ).select(df.columns)

    # 6. Re-emit under the SAME key (repr-rendered value → invariant #5).
    _emit(provider, "solve_data/pdtNodeInflow.csv", new_frame)
