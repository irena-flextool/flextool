"""Cluster E — block-layout consumers (Δ.9).

Lazy-polars port of flextool's block-aware derived helpers.  Cluster E is
the fifth of six derived-helper port phases per
``audit/native_data_path_design_derived_clusters.md``.

Cluster E fields (per the schematic):

* ``nodeStateBlock`` — set: nodes pulling daily-aggregation balance.
* ``period_block`` / ``period_block_succ`` / ``period_block_time``
  — multi-resolution block decomposition for storage state.
* ``arc_sink_block_dt`` / ``arc_source_block_dt`` — per-arc daily-block
  aggregation index ``(p, source, sink, d, b_first, t, weight)``.
* ``p_arc_sink_weight`` / ``p_arc_source_weight`` —
  ``Param[(p, source, sink, d, t), weight]`` projected from the above.
* ``dtttdt_block_interior`` — interior-of-block dtttdt rows.
* ``nodeState_last_dt`` — ``(n, d, t)`` last fine-step of last block.
* ``flow_to_n`` / ``flow_from_n`` — block-compatibility filtered.
* ``flow_from_nodeBalance_eff`` / ``flow_from_nodeBalance_noEff`` —
  block-compatibility filtered source-side nodeBalance arcs.

All consumers read ``BlockLayout``'s in-memory frames; no helper
re-reads ``solve_data/{entity_block,process_side_block,
block_step_duration,overlap_set,block_period_time_*}.csv``.

The single ``BlockLayout`` is built once per solve via
``BlockLayout.load_from_solve_data`` (or, post-Γ.8, natively from
``BlockLayout.build``).  ``BlockBundle`` wraps the layout with cached
derived frames (``block_compat_frame``, ``process_side_block_lf`` etc.)
that the cluster E helpers join against.

Design decisions:

* **One BlockLayout per solve.** Repeated CSV reads were the Δ.2 carry-
  over; cluster E folds every consumer onto one shared layout.
* **Cache ``block_compat`` and the rename'd join helpers.** Lazy frames
  share the cached materialisation; downstream `.collect()` once at
  the rim.
* **Identity-trivial fast path.** Single-block fixtures (``work_base``
  etc.) collapse the filter joins to no-ops because
  ``block_compat`` carries only ``(default, default)``.
* **Closes the Δ.3 `flow_to_n` / `flow_from_n` gap.** The block-aware
  filter previously lived only in the CSV path
  (``input.py::_load_process_topology``); the helper here mirrors it
  on the source-driven path so multi-block fixtures' `db_direct_parity`
  test stops being a known gap.

Reference: ``flextool/engine_polars/input.py::_load_process_topology``
(lines 703-783) and ``input.py::_load_storage`` (lines 1647-1699,
2233-2253).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    cast_dim,
    get_global_axis_enums,
    lit_axis,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine_polars._block_layout import (
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
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# BlockBundle — BlockLayout + cached lazy join frames
# ---------------------------------------------------------------------------


@dataclass
class BlockBundle:
    """Cached lazy-frame surface over a :class:`BlockLayout`.

    The bundle wraps a single per-solve :class:`BlockLayout` and exposes
    pre-renamed lazy frames keyed for cluster E consumers' joins.  The
    rename + cache pattern matches the schematic's "cache ``overlap_set``
    per (b_coarse, b_fine) pair" recommendation — every helper that
    joins on ``block_compat`` shares the same materialised frame.

    Attributes
    ----------
    layout : BlockLayout
        The underlying per-solve block layout.
    process_side_block_lf : pl.LazyFrame
        Renamed ``(p, side, b_f)`` lazy frame for arc-side block lookups.
    entity_block_lf : pl.LazyFrame
        Renamed ``(n, b)`` lazy frame for node-side block lookups.
    block_compat_frame : pl.DataFrame
        Cached compatibility set ``(b, b_f)`` derived from
        ``overlap_set`` — populated lazily on first access.
    block_step_duration_arc_lf : pl.LazyFrame
        Renamed ``(b_f, d, t, weight)`` lazy frame for per-arc weight
        materialisation.
    block_period_time_first_lf : pl.LazyFrame
        Renamed ``(b, d, t)`` lazy frame for first-step boundaries.
    block_period_time_last_lf : pl.LazyFrame
        Renamed ``(b, d, t)`` lazy frame for last-step boundaries.
    """

    layout: BlockLayout
    _block_compat_cached: pl.DataFrame | None = field(default=None, repr=False)
    _coarse_blocks_cached: list[str] | None = field(default=None, repr=False)
    _is_multi_block_cached: bool | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lazy-frame surface (joins consume these)
    # ------------------------------------------------------------------

    @property
    def process_side_block_lf(self) -> pl.LazyFrame:
        """Lazy ``(p, side, b_f)`` frame, or empty when no block data."""
        f = self.layout.process_side_block_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "p": schema_dtype(_enums, "p"),
                "side": pl.Utf8,
                "b_f": schema_dtype(_enums, "b_f"),
            })
        return f.lazy().pipe(rename_to_axis, {"process": "p", "block": "b_f"})

    @property
    def entity_block_lf(self) -> pl.LazyFrame:
        """Lazy ``(n, bk)`` frame, or empty when no block data.

        The block axis column is named ``bk`` (not ``b``) to disambiguate
        from the branch axis — see the b_collision review note in
        ``version/flextool_axis_contract.json``.
        """
        f = self.layout.entity_block_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "n": schema_dtype(_enums, "n"),
                "bk": schema_dtype(_enums, "bk"),
            })
        return f.lazy().pipe(rename_to_axis, {"entity": "n", "block": "bk"})

    @property
    def block_step_duration_arc_lf(self) -> pl.LazyFrame:
        """Lazy ``(b_f, d, t, weight)`` arc-keyed step duration."""
        f = self.layout.block_step_duration_frame
        if f.height == 0:
            return pl.LazyFrame(schema={
                "b_f": pl.Utf8,
                "d": schema_dtype(_enums, "d"),
                "t": schema_dtype(_enums, "t"),
                "weight": pl.Float64,
            })
        return f.lazy().pipe(rename_to_axis, {
            "block": "b_f", "period": "d",
            "step": "t", "step_duration": "weight",
        })

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

    @property
    def block_compat_frame(self) -> pl.DataFrame:
        """Cached ``(b, b_f)`` overlap-derived compatibility set.

        Computed on first access from ``layout.overlap_set_frame``;
        materialised once per :class:`BlockBundle` instance.
        """
        if self._block_compat_cached is None:
            self._block_compat_cached = self.layout.block_compat()
        return self._block_compat_cached

    def is_multi_block(self) -> bool:
        """Return ``True`` when the layout exercises >1 distinct block.

        Cached: layout is immutable for the bundle's lifetime.
        """
        if self._is_multi_block_cached is None:
            bsd = self.layout.block_step_duration_frame
            if bsd.height == 0:
                self._is_multi_block_cached = False
            else:
                self._is_multi_block_cached = bsd["block"].n_unique() >= 2
        return self._is_multi_block_cached

    def coarse_blocks(self, threshold: float = 1.0) -> list[str]:
        """Return blocks with at least one row of ``step_duration >
        threshold``.

        Cached for the canonical ``threshold=1.0`` (the schematic's
        rule for "coarse" blocks).  Other thresholds bypass the cache.
        """
        if threshold == 1.0:
            if self._coarse_blocks_cached is None:
                self._coarse_blocks_cached = self.layout.coarse_blocks(
                    threshold=1.0,
                )
            return self._coarse_blocks_cached
        return self.layout.coarse_blocks(threshold=threshold)

    def has_block_data(self) -> bool:
        """Return ``True`` when any block frame has data."""
        return (
            self.layout.process_side_block_frame.height > 0
            or self.layout.entity_block_frame.height > 0
            or self.layout.block_step_duration_frame.height > 0
        )


# ---------------------------------------------------------------------------
# Workdir bridge — load BlockBundle from solve_data/ CSVs
# ---------------------------------------------------------------------------


def load_block_bundle(
    workdir: Path | None,
    *,
    block_layout: "BlockLayout | None" = None,
    provider: "object | None" = None,
) -> BlockBundle | None:
    """Load a :class:`BlockBundle` from a Provider (preferred) or an
    in-memory layout.

    When ``block_layout`` is supplied (Phase 2 multi-block fast-path),
    wrap that layout directly.  Otherwise route through
    :meth:`BlockLayout.load_from_solve_data` against *provider*.

    The legacy ``workdir``-only signature is honoured by seeding an
    ephemeral Provider from ``workdir/solve_data`` via
    :func:`_input_source.seed_provider_from_dir`.  This keeps off-
    cascade tests (and the few callers that still pass a workdir
    without a Provider) working while the cascade itself never
    touches disk.

    Returns ``None`` if neither input yields a non-empty layout.
    """
    if block_layout is not None:
        if block_layout.is_empty():
            return None
        return BlockBundle(layout=block_layout)
    if provider is None and workdir is None:
        return None
    if provider is None and workdir is not None:
        sd = Path(workdir) / "solve_data"
        if not sd.exists():
            return None
    sd = (
        Path(workdir) / "solve_data" if workdir is not None
        else Path(".")
    )
    layout = BlockLayout.load_from_solve_data(sd, provider=provider)
    if layout.is_empty():
        return None
    return BlockBundle(layout=layout)


# ---------------------------------------------------------------------------
# §3.3.1 — flow_to_n / flow_from_n block-aware filter (Δ.3 gap closure)
# ---------------------------------------------------------------------------


def filter_flow_n_by_block(
    flow_n: pl.DataFrame,
    bundle: BlockBundle | None,
    *,
    side: str,
) -> pl.DataFrame:
    """Apply block-compatibility filter to ``flow_to_n`` / ``flow_from_n``.

    Mirror of ``input.py::_load_process_topology`` lines 728-782.  An
    arc ``(p, source, sink)`` contributes to node ``n``'s nodeBalance
    iff ``(b_n, b_f)`` exists in :py:attr:`BlockBundle.block_compat_frame`,
    where:

    * ``b_n`` is the entity-block of the destination node ``n``
      (default = ``DEFAULT_BLOCK`` when missing).
    * ``b_f`` is the process-side block on *side* (default =
      ``DEFAULT_BLOCK`` when missing).

    Parameters
    ----------
    flow_n : pl.DataFrame
        Schema ``[p, source, sink, n]``.
    bundle : BlockBundle or None
        When ``None`` or empty / single-block, filter is a no-op
        (returns ``flow_n`` unchanged).
    side : str
        ``"sink"`` (for ``flow_to_n``) or ``"source"``
        (for ``flow_from_n``).

    Returns
    -------
    pl.DataFrame
        The filtered frame.  Mirrors the reference's "only replace if
        the filter actually drops rows" guard so empty-overlap fixtures
        keep their pre-filter shape.
    """
    if flow_n is None or flow_n.height == 0:
        return flow_n
    if bundle is None:
        return flow_n
    psb_f = bundle.layout.process_side_block_frame
    eb_f = bundle.layout.entity_block_frame
    compat = bundle.block_compat_frame
    if psb_f.height == 0 or eb_f.height == 0 or compat.height == 0:
        return flow_n

    psb_side = (
        bundle.process_side_block_lf
        .filter(pl.col("side") == side)
        .select("p", "b_f")
    )
    eb_lf = bundle.entity_block_lf

    # Align ``n`` dtype across both sides.  Under activation
    # ``flow_n.n`` carries e-vocabulary (node + process tokens for
    # indirect units' arcs); entity_block_lf.n carries n-vocabulary
    # (nodes only).  Per contract n ⊂ e, so up-cast entity_block_lf.n
    # to e-Enum and the join composes natively in Enum.  Process
    # tokens in flow_n.n won't match any entity_block_lf row (left
    # join produces null block info) and the subsequent inner join
    # with ``compat`` drops them — same semantics as the prior
    # Utf8-roundtrip but without the materialisation.
    #
    # Block join keys (bk, b_f) use the cast_dim fill: substrate
    # produces them as block-Enum under activation, Utf8 otherwise.
    # The ``DEFAULT_BLOCK`` fill must use lit_axis so the literal
    # matches the column dtype.
    eb_lf_e = eb_lf.with_columns(cast_dim(pl.col("n"), None, "e"))
    flow_n_e = flow_n.with_columns(cast_dim(pl.col("n"), None, "e"))
    with_blocks = (
        flow_n_e.lazy()
        .join(psb_side, on="p", how="left")
        .join(eb_lf_e, on="n", how="left")
        .with_columns(
            b_f=pl.col("b_f").fill_null(lit_axis(DEFAULT_BLOCK, "block")),
            bk=pl.col("bk").fill_null(lit_axis(DEFAULT_BLOCK, "block")),
        )
    )
    filtered = (
        with_blocks
        .join(compat.lazy(), on=["bk", "b_f"], how="inner")
        .select("p", "source", "sink", "n")
        .unique()
        .collect()
    )
    # Match reference: only replace if filter dropped rows.  The
    # ``filtered.height > 0`` guard preserves the pre-filter frame for
    # fixtures whose overlap_set is degenerate-but-non-empty.
    if 0 < filtered.height < flow_n.height:
        return filtered
    return flow_n


def flow_to_n_block_filtered(
    pss: pl.DataFrame,
    bundle: BlockBundle | None,
) -> pl.DataFrame:
    """Build ``flow_to_n`` (``n = sink``) and apply the block-aware filter.

    Schema: ``[p, source, sink, n]``.  The block-aware filter is the
    Δ.3 gap closure — flextool's CSV path filters in
    ``_load_process_topology``; the source-driven path now mirrors the
    same filter.
    """
    if pss is None or pss.height == 0:
        # Empty-fallback dtype: ``n`` carries e-vocabulary (sink/source
        # values are e-typed under activation; the column legitimately
        # mixes node + process tokens for indirect units' arcs).
        # Declare it ``e`` so empty and populated branches agree.
        return pl.DataFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "source": schema_dtype(_enums, "source"),
            "sink": schema_dtype(_enums, "sink"),
            "n": schema_dtype(_enums, "e"),
        })
    # Cross-axis projection: ``sink`` carries e-axis tokens (mix of
    # node + process names).  ``alias_to_axis("sink", "e")`` casts
    # to e-Enum under activation, preserving every token.  The
    # downstream block-filter join in ``filter_flow_n_by_block``
    # also up-casts entity_block_lf.n to e-Enum (n ⊂ e), so the
    # join composes natively without Utf8 materialisation.
    base = (
        pss.lazy()
        .with_columns(alias_to_axis(pl.col("sink"), "e").alias("n"))
        .select("p", "source", "sink", "n")
        .sort("p", "source", "sink", "n")
        .collect()
    )
    return filter_flow_n_by_block(base, bundle, side="sink")


def flow_from_n_block_filtered(
    pss: pl.DataFrame,
    bundle: BlockBundle | None,
) -> pl.DataFrame:
    """Build ``flow_from_n`` (``n = source``) with block-aware filter."""
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "source": schema_dtype(_enums, "source"),
            "sink": schema_dtype(_enums, "sink"),
            "n": schema_dtype(_enums, "e"),
        })
    base = (
        pss.lazy()
        # Cross-axis projection: ``source`` is e-axis (node + process
        # union).  Cast to e-Enum (preserves every token) so the
        # downstream block-filter join on ``n`` composes natively in
        # Enum after entity_block_lf.n is up-cast to e as well.
        .with_columns(alias_to_axis(pl.col("source"), "e").alias("n"))
        .select("p", "source", "sink", "n")
        .sort("p", "source", "sink", "n")
        .collect()
    )
    return filter_flow_n_by_block(base, bundle, side="source")


# ---------------------------------------------------------------------------
# §3.9 — flow_from_nodeBalance block filter
# ---------------------------------------------------------------------------


def flow_from_nodeBalance_block_filtered(
    flow_from_nb: pl.DataFrame | None,
    bundle: BlockBundle | None,
) -> pl.DataFrame | None:
    """Apply the block-compatibility filter to source-side nodeBalance arcs.

    Mirror of ``input.py::_load_storage`` lines 1664-1699.  The
    ``flow_from_nodeBalance_eff`` / ``flow_from_nodeBalance_noEff``
    frames carry ``(p, source, sink, n=source)``; the filter drops arcs
    whose source-block doesn't overlap the destination node's block.

    Returns the filtered frame (possibly identical to input when no
    filter applies).  ``None`` in → ``None`` out.
    """
    if flow_from_nb is None or flow_from_nb.height == 0:
        return flow_from_nb
    if bundle is None:
        return flow_from_nb
    return filter_flow_n_by_block(flow_from_nb, bundle, side="source")


# ---------------------------------------------------------------------------
# Δ.27 — flow_from_nodeBalance_{eff,noEff} source-driven seed
# ---------------------------------------------------------------------------


def flow_from_nodeBalance_seed(
    pss_partition: pl.DataFrame | None,
    nodeBalance: pl.DataFrame | None,
    bundle: BlockBundle | None = None,
) -> pl.DataFrame | None:
    """Source-driven seed for ``flow_from_nodeBalance_eff`` /
    ``flow_from_nodeBalance_noEff``.

    Mirror of the inline derivation in
    ``input.py::_load_storage`` lines 1658-1705 (the slow path's
    storage loader).  Per the dispatch:

    ::

        flow_from_nb_<part> = pss_<part>
            .filter(source ∈ nodeBalance.n)
            .with_columns(n=source)
            .select(p, source, sink, n)

    Then the block-compatibility filter (when *bundle* carries an
    ``overlap_set``) drops arc rows whose source-block doesn't overlap
    the destination node's block.

    The two partitions (``eff`` / ``noEff``) share this logic; the
    caller passes the appropriate ``process_source_sink_eff`` /
    ``_noEff`` partition to produce the matching field.

    Parameters
    ----------
    pss_partition : pl.DataFrame or None
        ``process_source_sink_eff`` or ``_noEff`` — schema
        ``[p, source, sink]``.  ``None`` → returns ``None``.
    nodeBalance : pl.DataFrame or None
        ``[n]`` set frame.  Empty / ``None`` → returns ``None`` (no
        nodeBalance nodes means there are no source-side flows to
        gather).
    bundle : BlockBundle or None
        When supplied, applies the source-side block-compat filter
        (mirrors :func:`flow_from_nodeBalance_block_filtered`).
        ``None`` keeps the unfiltered seed.

    Returns
    -------
    pl.DataFrame or None
        ``[p, source, sink, n]`` with ``n = source`` and the filter
        applied.  ``None`` when the inputs cannot produce a non-empty
        frame.
    """
    if pss_partition is None or pss_partition.height == 0:
        return None
    if nodeBalance is None or nodeBalance.height == 0:
        return None
    if "n" not in nodeBalance.columns:
        return None
    # Cross-axis is_in: nodeBalance.n is n-Enum (n ⊂ e), pss.source is
    # e-Enum.  Up-cast nodeBalance.n to e-Enum (Pattern 2) so the
    # membership filter composes natively in Enum.  Project ``n`` as
    # e-Enum via alias_to_axis — matches the downstream block-filter
    # join in :func:`filter_flow_n_by_block`, which also up-casts
    # entity_block_lf.n to e-Enum.
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
    if bundle is not None:
        return filter_flow_n_by_block(seed, bundle, side="source")
    return seed


# ---------------------------------------------------------------------------
# §3.9.2 — nodeStateBlock multi-resolution synthesis
# ---------------------------------------------------------------------------


def nodeStateBlock_lf(
    bundle: BlockBundle | None,
    explicit_intraperiod: pl.LazyFrame | None,
    node_set: set[str] | None = None,
    *,
    coarse_threshold: float = 1.0,
) -> pl.LazyFrame:
    """Synthesise ``nodeStateBlock`` per audit §3.9.2.

    Two contributing branches:

    1. **Explicit method**: nodes whose
       ``storage_binding_method == 'bind_intraperiod_blocks'`` enter
       the set verbatim.
    2. **Multi-resolution synthesis**: when ``bundle`` carries >=2
       distinct blocks AND a node entity is assigned a coarse block
       (``step_duration > coarse_threshold``), that node is folded
       into the set so the daily-aggregation balance fires.

    Returns a lazy ``[n]`` frame, possibly empty.
    """
    parts: list[pl.LazyFrame] = []
    if explicit_intraperiod is not None:
        parts.append(
            explicit_intraperiod.select("n").unique()
        )
    if bundle is not None and bundle.is_multi_block():
        coarse = bundle.coarse_blocks(threshold=coarse_threshold)
        if coarse:
            eb_lf = bundle.entity_block_lf
            picked = (
                eb_lf
                .filter(pl.col("bk").is_in(coarse))
                .select(pl.col("n"))
            )
            if node_set is not None:
                picked = picked.filter(
                    pl.col("n").is_in(list(node_set)))
            parts.append(picked.unique())
    if not parts:
        return pl.LazyFrame(schema={"n": schema_dtype(_enums, "n")})
    out = pl.concat(parts).unique().sort("n")
    return out


# ---------------------------------------------------------------------------
# §3.9.3 — period_block / period_block_succ / period_block_time
# ---------------------------------------------------------------------------


def period_block_multi_resolution_lf(
    bundle: BlockBundle | None,
    *,
    coarse_threshold: float = 1.0,
) -> dict[str, pl.LazyFrame] | None:
    """Build the multi-resolution synthesis branch of period_block_*.

    Returns a dict ``{"period_block": LF, "period_block_succ": LF,
    "period_block_time": LF}`` when *bundle* exercises multiple blocks
    AND at least one block is coarse (per *coarse_threshold*).  Returns
    ``None`` when the synthesis branch doesn't fire (caller falls back
    to the timeset-based default branch).

    Mirror of ``input.py:1985-2126`` and the multi-resolution path of
    ``period_block_family_from_source`` in ``_derived_params.py``.
    """
    if bundle is None or not bundle.is_multi_block():
        return None
    coarse = bundle.coarse_blocks(threshold=coarse_threshold)
    if not coarse:
        return None
    eb = bundle.layout.entity_block_frame
    coarse_use_set = set(
        eb.filter(pl.col("block").is_in(coarse))["block"]
        .unique().to_list()
    )
    if not coarse_use_set:
        return None
    coarse_use = sorted(coarse_use_set)

    bsd = bundle.layout.block_step_duration_frame
    bsd_c = bsd.filter(pl.col("block").is_in(coarse_use))
    if bsd_c.height == 0:
        return None

    # period_block: (d, b_first) — coarse block step list.
    new_pb = (
        bsd_c
        .pipe(rename_to_axis, {"period": "d", "step": "b_first"})
        .select("d", "b_first")
        .unique()
    )

    # period_block_succ: cyclic per (block, period).
    succ_rows: list[tuple[str, str, str]] = []
    bsd_sorted = (
        bsd_c
        .pipe(rename_to_axis, {"period": "d", "step": "b_first"})
        .sort("block", "d", "b_first")
    )
    for (_blk, dval), grp in bsd_sorted.group_by(
        ["block", "d"], maintain_order=True
    ):
        bfs = grp["b_first"].to_list()
        n = len(bfs)
        for i in range(n):
            succ_rows.append((dval, bfs[i], bfs[(i + 1) % n]))
    if succ_rows:
        new_pbs = pl.DataFrame(
            succ_rows,
            schema=["d", "b_first", "b_next"],
            orient="row",
        ).with_columns(
            alias_to_axis("d", "d"),
            alias_to_axis("b_first", "b_first"),
            alias_to_axis("b_next", "b_next"),
        )
    else:
        new_pbs = pl.DataFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "b_first": schema_dtype(_enums, "b_first"),
            "b_next": schema_dtype(_enums, "b_next"),
        })

    # period_block_time: (d, b_first, t) — overlap_set rows where
    # b_coarse=coarse, b_fine=default.
    ov = bundle.layout.overlap_set_frame
    if ov.height == 0:
        return None
    ov_renamed = ov.pipe(rename_to_axis, {
        "period": "d",
        "block_coarse": "bk",
        "step_coarse": "b_first",
        "block_fine": "b_fine",
        "step_fine": "t",
    })
    ov_keep = ov_renamed.filter(
        pl.col("bk").is_in(coarse_use)
        & (pl.col("b_fine") == DEFAULT_BLOCK)
    )
    if ov_keep.height == 0:
        new_pbt = pl.DataFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "b_first": schema_dtype(_enums, "b_first"),
            "t": schema_dtype(_enums, "t"),
        })
    else:
        new_pbt = ov_keep.select("d", "b_first", "t").unique()

    return {
        "period_block": new_pb.lazy(),
        "period_block_succ": new_pbs.lazy(),
        "period_block_time": new_pbt.lazy(),
    }


# ---------------------------------------------------------------------------
# §3.9.4 — arc_sink_block_dt / arc_source_block_dt + weights
# ---------------------------------------------------------------------------


@dataclass
class ArcBlockFrames:
    """Container for the cluster E arc-block aggregation frames."""

    arc_sink_block_dt: pl.DataFrame | None = None
    arc_source_block_dt: pl.DataFrame | None = None
    p_arc_sink_weight: Param | None = None
    p_arc_source_weight: Param | None = None


def arc_block_dt(
    pss: pl.DataFrame | None,
    nodeStateBlock: pl.DataFrame | None,
    period_block_time: pl.DataFrame | None,
    bundle: BlockBundle | None,
) -> ArcBlockFrames:
    """Build per-arc daily-block aggregation frames.

    For each arc ``(p, source, sink)`` whose nodeStateBlock side participates
    in ``nodeStateBlock``, project to ``(p, source, sink, d, b_first, t,
    weight)`` with ``weight = block_step_duration`` of the arc-side block
    at fine ``(d, t)``.

    Returns an :class:`ArcBlockFrames` with up to four populated fields;
    any may be ``None`` when the corresponding side has no rows.
    """
    out = ArcBlockFrames()
    if (bundle is None
            or not bundle.has_block_data()
            or pss is None or pss.height == 0
            or nodeStateBlock is None or nodeStateBlock.height == 0
            or period_block_time is None or period_block_time.height == 0):
        return out

    psb_f = bundle.layout.process_side_block_frame
    bsd_f = bundle.layout.block_step_duration_frame
    if psb_f.height == 0 or bsd_f.height == 0:
        return out

    psb = bundle.process_side_block_lf
    bsd_arc = bundle.block_step_duration_arc_lf
    nsb_set = nodeStateBlock["n"].unique().to_list()
    pbt = period_block_time

    psb_sink = psb.filter(pl.col("side") == "sink").select("p", "b_f")
    sink_arcs = (
        pss.lazy()
        .filter(pl.col("sink").is_in(nsb_set))
        .join(psb_sink, on="p", how="inner")
    )
    sink_ab = (
        sink_arcs
        .join(bsd_arc, on="b_f", how="inner")
        .join(pbt.lazy(), on=["d", "t"], how="inner")
        .select("p", "source", "sink", "d", "b_first", "t", "weight")
        .unique()
        .collect()
    )
    if sink_ab.height > 0:
        out.arc_sink_block_dt = sink_ab
        wf = (
            sink_ab.select("p", "source", "sink", "d", "t", "weight")
            .unique()
            .rename({"weight": "value"})
        )
        out.p_arc_sink_weight = Param(
            ("p", "source", "sink", "d", "t"), wf,
        )

    psb_src = psb.filter(pl.col("side") == "source").select("p", "b_f")
    src_arcs = (
        pss.lazy()
        .filter(pl.col("source").is_in(nsb_set))
        .join(psb_src, on="p", how="inner")
    )
    src_ab = (
        src_arcs
        .join(bsd_arc, on="b_f", how="inner")
        .join(pbt.lazy(), on=["d", "t"], how="inner")
        .select("p", "source", "sink", "d", "b_first", "t", "weight")
        .unique()
        .collect()
    )
    if src_ab.height > 0:
        out.arc_source_block_dt = src_ab
        wf = (
            src_ab.select("p", "source", "sink", "d", "t", "weight")
            .unique()
            .rename({"weight": "value"})
        )
        out.p_arc_source_weight = Param(
            ("p", "source", "sink", "d", "t"), wf,
        )
    return out


# ---------------------------------------------------------------------------
# §3.9 — nodeState_last_dt
# ---------------------------------------------------------------------------


def nodeState_last_dt_lf(
    nodeState: pl.DataFrame | None,
    bundle: BlockBundle | None,
) -> pl.LazyFrame:
    """Build ``nodeState_last_dt`` ``(n, d, t)``.

    Last-fine-step-of-last-block per node.  Built from
    ``block_period_time_last`` × ``entity_block`` × ``nodeState``.
    Mirror of ``input.py:2233-2253``.
    """
    empty = pl.LazyFrame(schema={
        "n": schema_dtype(_enums, "n"),
        "d": schema_dtype(_enums, "d"),
        "t": schema_dtype(_enums, "t"),
    })
    if nodeState is None or nodeState.height == 0:
        return empty
    if bundle is None:
        return empty
    bptl_f = bundle.layout.block_period_time_last_frame
    eb_f = bundle.layout.entity_block_frame
    if bptl_f.height == 0 or eb_f.height == 0:
        return empty
    return (
        nodeState.lazy().select("n")
        .join(bundle.entity_block_lf, on="n", how="inner")
        .join(bundle.block_period_time_last_lf, on="bk", how="inner")
        .select("n", "d", "t")
        .unique()
    )


# ---------------------------------------------------------------------------
# §3.9 — dtttdt_block_interior
# ---------------------------------------------------------------------------


def dtttdt_block_interior_lf(
    dtttdt: pl.DataFrame | None,
    period_block_time: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Interior-of-block dtttdt rows.

    Two paths matching ``input.py``'s branching:

    1. **Default** (timeset-block decomposition): keep dtttdt rows where
       ``t_previous_within_timeset == t_previous`` (jump=1 interior).
    2. **Synthesised (multi-resolution)**: when *period_block_time* has
       multiple ``b_first`` per period, rebuild interior pairs from the
       coarse block decomposition: per (d, b_first), consecutive sorted
       fine t's give intra-day predecessor pairs.

    The caller distinguishes via *period_block_time*'s shape; this
    helper detects automatically.
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
    return (
        dtttdt.lazy()
        .filter(pl.col("t_previous_within_timeset")
                == pl.col("t_previous"))
        .select("d", "t", "t_previous")
    )


# ---------------------------------------------------------------------------
# Public — apply_block_cluster: single-pass entry for apply_derived_e.
# ---------------------------------------------------------------------------


__all__ = [
    "BlockBundle",
    "load_block_bundle",
    "filter_flow_n_by_block",
    "flow_to_n_block_filtered",
    "flow_from_n_block_filtered",
    "flow_from_nodeBalance_block_filtered",
    "flow_from_nodeBalance_seed",
    "nodeStateBlock_lf",
    "period_block_multi_resolution_lf",
    "arc_block_dt",
    "ArcBlockFrames",
    "nodeState_last_dt_lf",
    "dtttdt_block_interior_lf",
]
