"""Block-layout consumers: the ``BlockBundle`` wrapper, the node-arc frames and ``nodeState_last_dt``.

* ``flow_to_n`` / ``flow_from_n`` — ``(p, source, sink, n)`` with ``n`` the arc's sink / source: every arc
  enters the balance of both its nodes.  A node of a coarse new_stepduration group sums its fine flows over
  each coarse step (``nodeBalanceCoarse_eq``, frames from ``params._coarse``), so no arc is filtered by block.
* ``flow_from_nodeBalance_eff`` / ``flow_from_nodeBalance_noEff`` — source-side arcs of balance nodes.
* ``nodeState_last_dt`` — ``(n, d, t)`` the last step of each period for every storage node.
* ``dtttdt_block_interior`` — interior-of-block dtttdt rows (bind_intraperiod_blocks).

The single ``BlockLayout`` of a solve is built once via ``BlockLayout.load_from_solve_data``; ``BlockBundle``
wraps it with renamed lazy join frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    cast_dim,
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.preprocess.params._block_layout import (
    DEFAULT_BLOCK,
    BlockLayout,
)


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:  # pragma: no cover — typing only
    pass


# ---------------------------------------------------------------------------
# BlockBundle — BlockLayout + cached lazy join frames
# ---------------------------------------------------------------------------


@dataclass
class BlockBundle:
    """Lazy-frame surface over a :class:`BlockLayout`.

    Attributes
    ----------
    layout : BlockLayout
        The underlying per-solve block layout.
    entity_block_lf : pl.LazyFrame
        Renamed ``(n, bk)`` lazy frame for node-side block lookups.
    block_period_time_first_lf / block_period_time_last_lf : pl.LazyFrame
        Renamed ``(bk, d, t)`` lazy frames of the first / last step of each period per block.
    """

    layout: BlockLayout

    @property
    def entity_block_lf(self) -> pl.LazyFrame:
        """Lazy ``(n, bk)`` frame, or empty when no block data.

        The block axis column is named ``bk`` (not ``b``) to disambiguate
        from the branch axis — see the b_collision review note in
        ``schemas/flextool_axis_contract.json``.
        """
        f = self.layout.entity_block_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "n": schema_dtype(_enums, "n"),
                "bk": schema_dtype(_enums, "bk"),
            })
        return f.lazy().pipe(rename_to_axis, {"entity": "n", "block": "bk"})

    @property
    def block_period_time_first_lf(self) -> pl.LazyFrame:
        f = self.layout.block_period_time_first_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "bk": schema_dtype(_enums, "bk"),
                "d": schema_dtype(_enums, "d"),
                "t": schema_dtype(_enums, "t"),
            })
        return f.lazy().pipe(rename_to_axis, {"block": "bk", "period": "d", "step": "t"})

    @property
    def block_period_time_last_lf(self) -> pl.LazyFrame:
        f = self.layout.block_period_time_last_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "bk": schema_dtype(_enums, "bk"),
                "d": schema_dtype(_enums, "d"),
                "t": schema_dtype(_enums, "t"),
            })
        return f.lazy().pipe(rename_to_axis, {"block": "bk", "period": "d", "step": "t"})

    def has_block_data(self) -> bool:
        """Return ``True`` when any block frame has data."""
        return (
            self.layout.entity_block_frame.height > 0
            or self.layout.block_step_duration_frame.height > 0
        )


# ---------------------------------------------------------------------------
# Provider load — BlockBundle from the per-solve block frames
# ---------------------------------------------------------------------------


def load_block_bundle(
    workdir: Path | None,
    *,
    block_layout: "BlockLayout | None" = None,
    provider: "object | None" = None,
) -> BlockBundle | None:
    """Load a :class:`BlockBundle` from an in-memory layout or a Provider.

    When ``block_layout`` is supplied, wrap that layout directly.  Otherwise route through
    :meth:`BlockLayout.load_from_solve_data` against *provider*; ``provider=None`` raises
    ``ValueError`` (no ``solve_data/`` disk read).  *workdir* is retained for call
    compatibility and is not read.

    Returns ``None`` if the layout is empty.
    """
    if block_layout is not None:
        if block_layout.is_empty():
            return None
        return BlockBundle(layout=block_layout)
    if provider is None:
        raise ValueError(
            "load_block_bundle requires a FlexDataProvider (or a block_layout); "
            "it never reads solve_data/ on disk."
        )
    sd = (
        Path(workdir) / "solve_data" if workdir is not None
        else Path(".")
    )
    layout = BlockLayout.load_from_solve_data(sd, provider=provider)
    if layout.is_empty():
        return None
    return BlockBundle(layout=layout)


# ---------------------------------------------------------------------------
# §3.3.1 — flow_to_n / flow_from_n
# ---------------------------------------------------------------------------


def _flow_n_frame(pss: pl.DataFrame | None, side: str) -> pl.DataFrame:
    """``(p, source, sink, n)`` with ``n`` = the arc's ``side`` node.

    ``source`` / ``sink`` carry e-axis tokens (node + process names for indirect units' arcs);
    ``alias_to_axis(.., "e")`` keeps every token under axis-enum activation."""
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "source": schema_dtype(_enums, "source"),
            "sink": schema_dtype(_enums, "sink"),
            "n": schema_dtype(_enums, "e"),
        })
    return (
        pss.lazy()
        .with_columns(alias_to_axis(pl.col(side), "e").alias("n"))
        .select("p", "source", "sink", "n")
        .sort("p", "source", "sink", "n")
        .collect()
    )


def flow_to_n_frame(pss: pl.DataFrame | None) -> pl.DataFrame:
    """``flow_to_n``: ``(p, source, sink, n = sink)`` for every arc."""
    return _flow_n_frame(pss, "sink")


def flow_from_n_frame(pss: pl.DataFrame | None) -> pl.DataFrame:
    """``flow_from_n``: ``(p, source, sink, n = source)`` for every arc."""
    return _flow_n_frame(pss, "source")


# ---------------------------------------------------------------------------
# Δ.27 — flow_from_nodeBalance_{eff,noEff} source-driven seed
# ---------------------------------------------------------------------------


def flow_from_nodeBalance_seed(
    pss_partition: pl.DataFrame | None,
    nodeBalance: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Source-driven seed for ``flow_from_nodeBalance_eff`` / ``flow_from_nodeBalance_noEff``::

        flow_from_nb_<part> = pss_<part>.filter(source ∈ nodeBalance.n).with_columns(n=source)

    ``pss_partition``: ``process_source_sink_eff`` or ``_noEff`` ``[p, source, sink]``; ``nodeBalance``
    ``[n]``.  Returns ``[p, source, sink, n]``, or ``None`` when the inputs cannot produce a row.
    """
    if pss_partition is None or pss_partition.height == 0:
        return None
    if nodeBalance is None or nodeBalance.height == 0:
        return None
    if "n" not in nodeBalance.columns:
        return None
    # Cross-axis is_in: nodeBalance.n is n-Enum (n ⊂ e), pss.source is e-Enum.  Up-cast nodeBalance.n to
    # e-Enum so the membership filter composes natively in Enum; project ``n`` as e-Enum.
    nb_nodes_e = nodeBalance.lazy().select(cast_dim(pl.col("n"), None, "e"))
    seed = (
        pss_partition.lazy()
        .filter(pl.col("source").is_in(nb_nodes_e.collect()["n"]))
        .with_columns(alias_to_axis(pl.col("source"), "e").alias("n"))
        .select("p", "source", "sink", "n")
        .unique()
        .sort("p", "source", "sink", "n")
        .collect()
    )
    if seed.height == 0:
        return None
    return seed


# ---------------------------------------------------------------------------
# §3.9 — nodeState_last_dt
# ---------------------------------------------------------------------------


def nodeState_last_dt_lf(
    nodeState: pl.DataFrame | None,
    bundle: BlockBundle | None,
) -> pl.LazyFrame:
    """``nodeState_last_dt`` ``(n, d, t)``: the default block's last step of each period for every storage
    node.  Every node lives on the solve's steps (a coarse storage node's state is constant within its last
    coarse step), so the default block's boundary serves all of them."""
    empty = pl.LazyFrame(schema={
        "n": schema_dtype(_enums, "n"),
        "d": schema_dtype(_enums, "d"),
        "t": schema_dtype(_enums, "t"),
    })
    if nodeState is None or nodeState.height == 0 or bundle is None:
        return empty
    if bundle.layout.block_period_time_last_frame.height == 0:
        return empty
    _live = get_global_axis_enums()
    if _live is not None:
        nodeState = cast_frame_axes(nodeState, _live)
    last = (bundle.block_period_time_last_lf
            .filter(pl.col("bk").cast(pl.Utf8) == DEFAULT_BLOCK)
            .select("d", "t"))
    return nodeState.lazy().select("n").unique(maintain_order=True).join(last, how="cross").select("n", "d", "t")


# ---------------------------------------------------------------------------
# §3.9 — dtttdt_block_interior
# ---------------------------------------------------------------------------


def dtttdt_block_interior_lf(
    dtttdt: pl.DataFrame | None,
    period_block_time: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Interior-of-block dtttdt rows of the ``bind_intraperiod_blocks`` timeset blocks.

    1. **One block per period** (or no *period_block_time*): keep dtttdt rows
       where ``t_previous_within_timeset == t_previous`` (jump=1 interior).
    2. **Several timeset blocks per period**: per (d, b_first) of
       *period_block_time*, consecutive sorted steps give the interior
       predecessor pairs (the same set as branch 1, built from the blocks).

    The caller passes *period_block_time*; this helper picks the branch.
    """
    empty = pl.LazyFrame(schema={
        "d": schema_dtype(_enums, "d"),
        "t": schema_dtype(_enums, "t"),
        "t_previous": schema_dtype(_enums, "t_previous"),
    })
    if dtttdt is None or dtttdt.height == 0:
        return empty
    multi_res = False
    if period_block_time is not None and period_block_time.height > 0:
        nb = (
            period_block_time
            .group_by("d")
            .agg(pl.col("b_first").n_unique().alias("nb"))
            ["nb"].max()
        )
        if nb is not None and nb > 1:
            multi_res = True
    if multi_res and period_block_time is not None:
        rows: list[tuple[str, str, str]] = []
        pbt_sorted = period_block_time.sort("d", "b_first", "t")
        for (dval, _bf), grp in pbt_sorted.group_by(
            ["d", "b_first"], maintain_order=True
        ):
            ts = grp["t"].to_list()
            for i in range(1, len(ts)):
                rows.append((dval, ts[i], ts[i - 1]))
        if not rows:
            return empty
        return (
            pl.DataFrame(
                rows,
                schema=["d", "t", "t_previous"],
                orient="row",
            )
            .with_columns(
                alias_to_axis("d", "d"),
                alias_to_axis("t", "t"),
                alias_to_axis("t_previous", "t_previous"),
            )
            .unique()
            .lazy()
        )
    if "t_previous_within_timeset" not in dtttdt.columns:
        return empty
    # Defensive re-cast: align d / t / t_previous to canonical Enum so
    # the returned schema matches the empty-fallback and multi_res
    # branches above even when ``dtttdt`` arrives with Utf8 axis columns.
    return (
        dtttdt.lazy()
        .filter(pl.col("t_previous_within_timeset")
                == pl.col("t_previous"))
        .select(alias_to_axis("d", "d"),
                alias_to_axis("t", "t"),
                alias_to_axis("t_previous", "t_previous"))
    )


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


__all__ = [
    "BlockBundle",
    "load_block_bundle",
    "flow_to_n_frame",
    "flow_from_n_frame",
    "flow_from_nodeBalance_seed",
    "nodeState_last_dt_lf",
    "dtttdt_block_interior_lf",
]
