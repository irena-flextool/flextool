"""energy_margin_adder — invest-only, additive negative-inflow demand margin.

Runs per-solve from ``_emit_solve_time.run`` immediately AFTER the
multiplier (``emit_energy_margin_inflow``, batch 54) and BEFORE the
positive/negative inflow split (batch 58), so that split inherits the
added demand.  Ordering is multiply-then-add.

FlexTool sign convention: DEMAND is NEGATIVE inflow (the exogenous
outflow / ``p_negative_inflow`` term); SUPPLY / generation is POSITIVE
inflow.  This feature adds ``energy_margin_adder`` MWh of EXTRA DEMAND at
a node in the invest solve, which — because demand is negative inflow —
means the node's inflow value becomes MORE NEGATIVE:

    value_new = value_old - adder            (adder > 0 ⇒ deeper demand)

Adding a positive number would REDUCE demand — the inverted-sign no-op.
The subtraction above is the ONLY correct direction.

What it does vs. the multiplier
-------------------------------
The multiplier (``_emit_energy_margin.py``) SCALES a node's existing
NEGATIVE inflow rows — a no-op on a node with zero native inflow.  The
adder targets exactly those zero-inflow slack nodes: it must CREATE
negative demand rows for a node that has NO ``pdtNodeInflow`` row at all
(value ``-adder`` at every invest ``(d, t)``), as well as deepen existing
rows.

For a node with ``energy_margin_method == "inflow_adder"`` and an
effective ``energy_margin_adder`` value ``!= 0.0``, over the invest
solve's ``(d, t)`` grid (``steps_in_use.csv``):

* an EXISTING ``(node, d, t)`` inflow row → ``value - adder``;
* a MISSING ``(node, d, t)`` row → a new row with value ``-adder``.

Applied ONLY:

* in the solve that carries investment periods
  (``bool(state.solve.invest_periods.get(solve_name))``) — dispatch
  solves see the true, un-margined demand; and
* to nodes whose method is ``inflow_adder`` with a non-zero adder.

Shape support & the (d, t) broadcast
------------------------------------
A SCALAR adder is a constant per-timestep addition broadcast across every
invest ``(d, t)`` (the "spread uniformly" the calibrator relies on); a
period (or period,time) Map places per-``(d, t)`` values.  Both are
supported: the authored shape is read from the ingested frame's explicit
index columns (``period`` / ``time``) — safe here because the frame has
already been normalised by ingestion — and broadcast over the invest grid
through :func:`._param_shapes.promote_param_to_dt` (NEVER by a hand-rolled
column-name cross-join; invariant #2).

Early-return byte-parity contract
----------------------------------
If the solve is NOT an invest solve, OR no node has
``energy_margin_method == "inflow_adder"`` with an effective adder
``!= 0.0``, the emitter returns WITHOUT touching the provider — the
batch-54 ``pdtNodeInflow`` frame stands untouched, so the default (nobody
sets an adder) is byte-identical to today.

RHS-only (protects warm-start)
------------------------------
This emitter only mutates the ``pdtNodeInflow`` provider frame — a
nodeBalance RHS constant.  It introduces NO new variable, constraint,
column or row into the LP: the node already carries its balance rows;
creating an inflow row for an existing node only changes the inflow
constant, never adds an LP row.

Invariant #5 (byte-parity)
--------------------------
Every touched (deepened) or created row's ``value`` is rendered with the
``repr()``-based :func:`_render_value_column`, NEVER ``.cast(Utf8)``.
Untouched rows (other nodes, non-invest (d,t) — none here since the grid
IS the invest grid — and any row of a node that isn't an ``inflow_adder``
node) keep their original ``value`` string verbatim.

Known limitation — ``no_inflow`` nodes and group accounting
-----------------------------------------------------------
The core ``nodeBalance_eq`` reads ``p_inflow`` DIRECTLY from
``pdtNodeInflow.csv`` (``model.py`` ~L1640), so the added demand IS
served for every node type, including one set ``inflow_method="no_inflow"``.
BUT the ``p_positive_inflow`` / ``p_negative_inflow`` split
(``_emit_period_params._derive_positive_negative_inflow``) forces
``no_inflow`` nodes to ``0.0`` — and those two frames feed the group
energy-slack / capacity-margin / reserve auxiliary constraints
(``_group_slack.py``).  So an adder on a node that is BOTH explicitly
``no_inflow`` AND a member of such a group is served by the balance yet
invisible to that group's accounting.  This does NOT affect the
calibrator (its targets are default ``use_original`` nodes, which flow
through both channels consistently); fix it when ``capacity_margin``
integration lands, where the split can be made adder-aware with full
context.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param

from flextool.engine_polars._emit_inflow_scaling import (
    _read_pairs,
)
from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars._param_shapes import promote_param_to_dt
from flextool.engine_polars._vectorize import _render_value_column

# The pdtNodeInflow frame is all-Utf8: node, period, time, value.
_PDT_COLS = ("node", "period", "time", "value")

# The authored adder ingests through three specs (see _specs.py): a
# scalar float, a period Map (1d), and a period-time Map (2d).  Each lands
# in its own CSV / Provider key; the emitter reads all three and unions
# the broadcast frames.  A node has a scalar OR a map adder, never both.
_ADDER_FILES = (
    "energy_margin_adder.csv",       # scalar float → shape (node,)
    "pd_energy_margin_adder.csv",    # 1d Map       → shape (node, d)
    "pdt_energy_margin_adder.csv",   # 2d Map       → shape (node, d, t)
)


def _broadcast_authored_adder(adder_df, manual_nodes, dt_grid):
    """Broadcast one authored adder frame over the invest (d, t) grid.

    *adder_df* is an all-Utf8 ingested frame whose FIRST column is the
    node and whose LAST column is the value; any middle columns are the
    authored Map index axes (``period`` and/or ``time``).  Returns a frame
    with columns ``node, period, time, __adder`` (Float64 adder) restricted
    to *manual_nodes*, or ``None`` when there is nothing to apply (empty
    frame, an unrecognised index axis, or all-zero / null adders).
    """
    if adder_df is None or adder_df.height == 0:
        return None
    cols = adder_df.columns
    if len(cols) < 2:
        return None

    # Resolve the authored shape from the frame's explicit index columns
    # and build the rename → canonical (node, d, t) Param dims.  Any index
    # column that is neither ``period`` nor ``time`` is an unrecognised
    # axis for this parameter — skip this frame rather than guessing a
    # broadcast (byte-parity).
    node_col, value_col = cols[0], cols[-1]
    rename: dict[str, str] = {node_col: "node"}
    dims: list[str] = ["node"]
    for c in cols[1:-1]:
        cl = c.strip().lower()
        if cl == "period":
            rename[c] = "d"
            dims.append("d")
        elif cl in ("time", "t"):
            rename[c] = "t"
            dims.append("t")
        else:
            return None

    # Restrict to the manual (inflow_adder) nodes, cast the value to
    # Float64, and drop null / zero adders (a zero adder is a no-op).
    work = (
        adder_df.rename(rename)
        .with_columns(
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        )
        .filter(pl.col("node").is_in(list(manual_nodes)))
        .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
        .select([*dims, "value"])
    )
    if work.height == 0:
        return None

    # Broadcast the authored shape over the invest solve's (d, t) grid.
    # Route through _param_shapes' promote_param_to_dt (invariant #2 —
    # never a hand-rolled column-name cross-join): a SCALAR (node,) Param
    # cross-joins the whole grid; a (node, d) / (node, d, t) Param joins on
    # its authored axis.
    adder_dt = (
        promote_param_to_dt(Param(tuple(dims), work), dt_grid)
        .select("node", "d", "t", "value")
        .collect()
        .rename({"d": "period", "t": "time", "value": "__adder"})
    )
    if adder_dt.height == 0:
        return None
    return adder_dt


def emit_energy_margin_adder(
    state,
    solve_name,
    input_dir,
    solve_data_dir,
    *,
    provider,
) -> None:
    """Apply the invest-only additive energy_margin_adder to pdtNodeInflow.

    See the module docstring for the full contract.  A non-invest solve,
    no ``inflow_adder`` node, or an all-zero / missing adder frame yields
    an early return (byte-parity).
    """
    # 1. node → method; manual = the inflow_adder nodes.
    method_pairs = _read_pairs(
        input_dir / "node__energy_margin_method.csv", provider=provider,
    )
    manual_nodes = {n for (n, m) in method_pairs if m == "inflow_adder"}

    # 2. Invest-only gate + no-manual-node gate → early return (byte-parity).
    is_invest = bool(state.solve.invest_periods.get(solve_name))
    if not is_invest or not manual_nodes:
        return

    # 3. Build the invest solve's (d, t) grid once.  ``steps_in_use.csv``
    #    is the current (invest) solve's active (period, time) grid — the
    #    same grid every pdtX emitter expands over.
    dt_pairs = _read_pairs(
        solve_data_dir / "steps_in_use.csv", provider=provider,
    )
    if not dt_pairs:
        return
    dt_grid = pl.DataFrame(
        {"d": [d for d, _t in dt_pairs], "t": [t for _d, t in dt_pairs]},
        schema={"d": pl.Utf8, "t": pl.Utf8},
    )

    # 4. Read every authored adder file (scalar float, period Map, and
    #    period-time Map — each in its own Provider key) and broadcast each
    #    over the (d, t) grid via _broadcast_authored_adder.  A node carries
    #    a scalar OR a map adder, never both, so the per-file frames are
    #    disjoint by node; union them into one (node, period, time, __adder)
    #    frame.  No authored value anywhere → early return (byte-parity).
    parts = []
    for fname in _ADDER_FILES:
        adder_df = provider.get(_provider_key(input_dir / fname))
        part = _broadcast_authored_adder(adder_df, manual_nodes, dt_grid)
        if part is not None:
            parts.append(part)
    if not parts:
        return
    adder_dt = pl.concat(parts) if len(parts) > 1 else parts[0]
    if adder_dt.height == 0:
        return

    # 7. Merge into pdtNodeInflow.  Existing rows of an inflow_adder node
    #    are deepened (value - adder); grid cells with no existing row are
    #    CREATED with value -adder.  Every other row is verbatim.
    df = provider.get(_provider_key(solve_data_dir / "pdtNodeInflow.csv"))
    if df is None:
        df = pl.DataFrame(schema={c: pl.Utf8 for c in _PDT_COLS})
    out_cols = df.columns

    # 7a. Deepen existing rows.  Left-join the per-(node, d, t) adder; where
    #     present, re-render (value - adder) via repr, else keep the original
    #     value string verbatim (byte-parity for untouched rows).
    existing = df.with_columns(
        pl.col("value").cast(pl.Float64).alias("__vf"),
    ).join(adder_dt, on=["node", "period", "time"], how="left")
    deepened = _render_value_column(
        existing["__vf"] - existing["__adder"].fill_null(0.0),
    )
    df_mod = (
        existing.with_columns(deepened.alias("__deep"))
        .with_columns(
            pl.when(pl.col("__adder").is_not_null())
            .then(pl.col("__deep"))
            .otherwise(pl.col("value"))
            .alias("value"),
        )
        .select(out_cols)
    )

    # 7b. Create rows for grid cells the node had no inflow for.  value =
    #     -adder (adding demand ⇒ more negative).  Deterministic order.
    created = adder_dt.join(
        df.select("node", "period", "time"),
        on=["node", "period", "time"],
        how="anti",
    ).sort(["node", "period", "time"])
    created_frame = (
        created.with_columns(
            _render_value_column(-created["__adder"]).alias("value"),
        )
        .select(out_cols)
    )

    new_frame = pl.concat([df_mod, created_frame])

    # 8. Re-emit under the SAME key (repr-rendered value → invariant #5).
    _emit(provider, "solve_data/pdtNodeInflow.csv", new_frame)
