"""Δ.25 — surgical source-only FlexData loader.

Replaces ``flextool/flextoolrunner/input_writer.write_input`` (~2400 LOC
of CSV writers) for the **single-solve fast path**.  Reads directly
from a :class:`flextool.engine_polars._spinedb_reader.SpineDbReader`
and constructs a :class:`flextool.engine_polars.input.FlexData`
without touching the workdir CSVs.

Status
------

**Experimental / non-production.**  This is the user-flagged fast path
for simple single-solve fixtures (``test_24h_shipping`` was the
motivating workload).  No feature detection, no fallback to the slow
path: any helper that demands a workdir CSV raises ``FastLoadError``
with the field + helper name, the user fixes the helper or the
fixture, repeat.

The slow path (``run_chain_from_db`` → ``_native_input_writer``) is
unchanged and remains the canonical path until the preprocessing port
(Δ.20-redo) is complete.

What this module does
---------------------

1. Builds an empty :class:`FlexData` stub (required positional fields
   are sentinels).
2. Calls :func:`flextool.engine_polars.input._apply_db_overrides` —
   passes 1-9 of the override chain populate ~80% of the FlexData
   fields directly from the source.
3. Patches in the few topology fields the override chain doesn't yet
   own (``process_source_sink``, ``pss_dt``, ``flow_to_n`` /
   ``flow_from_n``, ``nodeBalance_dt``, the commodity-flow joins, …)
   using the projection helpers directly.
4. Returns the constructed FlexData.

What it does NOT do
-------------------

* No support for multi-solve / rolling / nested cascades — this is the
  *single-solve* fast path.  The slow path
  (:func:`flextool.engine_polars._orchestration.run_chain_from_db`)
  remains the canonical multi-solve driver.
* No handoff plumbing between solves — a single solve has no prior
  handoff to consume.
* No warm-LP — unrelated to the input-loader question.
* No region filter / Lagrangian — outside the dispatch scope.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from ._axis_enums import (
    alias_to_axis,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)


# Phase 4.6 — proxy over the live cascade-wide axis enum dict.
class _EnumsProxy:
    def __bool__(self) -> bool:
        return get_global_axis_enums() is not None

    def get(self, key, default=None):
        live = get_global_axis_enums()
        if live is None:
            return default
        return live.get(key, default)

    def __iter__(self):
        live = get_global_axis_enums()
        return iter(live) if live is not None else iter(())


_enums = _EnumsProxy()

if TYPE_CHECKING:
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars.input import FlexData


__all__ = [
    "FastLoadError",
    "load_flextool_source_only",
]


class FastLoadError(RuntimeError):
    """Raised when the fast path can't materialise a FlexData field
    that downstream consumers (model.py, output writer) require.

    Per the Δ.25 design (non-production, raise loudly), the fast path
    does not silently degrade — the operator either fixes the helper
    that produced ``None`` for a required field, or falls back to the
    slow path.
    """


# ---------------------------------------------------------------------------
# Empty-FlexData factory
# ---------------------------------------------------------------------------


def _empty_flex_data() -> "FlexData":
    """Construct a :class:`FlexData` with sentinel values on the
    required positional fields.

    The sentinels are immediately overwritten by the override chain;
    they exist only because ``FlexData.__init__`` requires them.  All
    optional fields default to ``None`` per the dataclass definition.
    """
    from polar_high import Param

    from flextool.engine_polars.input import FlexData

    empty_dt = pl.DataFrame(schema={
        "d": schema_dtype(_enums, "d"),
        "t": schema_dtype(_enums, "t")})
    empty_2 = pl.DataFrame(schema={
        "d": schema_dtype(_enums, "d"),
        "t": schema_dtype(_enums, "t"),
        "value": pl.Float64})
    empty_d = pl.DataFrame(schema={
        "d": schema_dtype(_enums, "d"),
        "value": pl.Float64})
    empty_node = pl.DataFrame(schema={"n": schema_dtype(_enums, "n")})
    empty_node_dt = pl.DataFrame(
        schema={"n": schema_dtype(_enums, "n"),
                "d": schema_dtype(_enums, "d"),
                "t": schema_dtype(_enums, "t")}
    )
    empty_ndt = pl.DataFrame(
        schema={"n": schema_dtype(_enums, "n"),
                "d": schema_dtype(_enums, "d"),
                "t": schema_dtype(_enums, "t"),
                "value": pl.Float64}
    )

    return FlexData(
        dt=empty_dt,
        p_step_duration=Param(("d", "t"), empty_2),
        p_rp_cost_weight=Param(("d", "t"), empty_2),
        p_inflation_op=Param(("d",), empty_d),
        p_period_share=Param(("d",), empty_d),
        nodeBalance=empty_node,
        nodeBalance_dt=empty_node_dt,
        p_inflow=Param(("n", "d", "t"), empty_ndt),
        p_penalty_up=Param(("n", "d", "t"), empty_ndt),
        p_penalty_down=Param(("n", "d", "t"), empty_ndt),
    )


# ---------------------------------------------------------------------------
# Topology — fields the override chain doesn't (yet) populate.
# ---------------------------------------------------------------------------


def _populate_topology(flex_data: "FlexData",
                        source: "SpineDbReader") -> None:
    """Build ``process_source_sink`` family + commodity-flow joins from
    the source.

    Mirrors the source-driven branch of
    :func:`flextool.engine_polars.input._load_process_topology` but with
    no workdir CSV reads — the four CSV-only sub-fields
    (``p_unitsize`` seed, ``p_slope`` seed, ``p_flow_upper`` from
    ``p_flow_max.csv``, ``commodity__node`` join) are skipped or built
    from the source-equivalent.

    The override chain populates ``p_unitsize`` (Δ.4 / Δ.4b) and
    ``p_slope`` (Δ.4b) via :mod:`_derived_params`; ``p_flow_upper`` is
    computed natively below from existing capacity + max_invest_cum
    when available, otherwise left ``None``.
    """
    from polar_high import Param

    from flextool.engine_polars._projection_params import (
        process_source_sink_canonical,
        _try_entities,
    )

    canonical = process_source_sink_canonical(source)
    if canonical.height == 0:
        # No processes / connections — leave topology empty.
        return

    pss = canonical.select("p", "source", "sink").unique()
    pss_eff = (canonical.filter(pl.col("method") == "eff")
                .select("p", "source", "sink").unique())
    pss_noEff = (canonical.filter(pl.col("method") == "noEff")
                  .select("p", "source", "sink").unique())

    flex_data.process_source_sink = pss
    flex_data.process_source_sink_eff = pss_eff
    flex_data.process_source_sink_noEff = pss_noEff
    flex_data.flow_to_n = pss.with_columns(n=pl.col("sink"))
    flex_data.flow_from_n = pss.with_columns(n=pl.col("source"))

    # Canonical (p, source) / (p, sink) — one row per unit input/output node
    # and one row per connection using the original connection__node__node
    # direction (not the added reverse arc).
    src_parts: list[pl.DataFrame] = []
    snk_parts: list[pl.DataFrame] = []
    _uin = _try_entities(source, "unit__inputNode")
    if _uin is not None and _uin.height > 0:
        src_parts.append(_uin.select(
            alias_to_axis("unit", "p"), alias_to_axis("node", "source")))
    _uout = _try_entities(source, "unit__outputNode")
    if _uout is not None and _uout.height > 0:
        snk_parts.append(_uout.select(
            alias_to_axis("unit", "p"), alias_to_axis("node", "sink")))
    _cnn = _try_entities(source, "connection__node__node")
    if _cnn is not None and _cnn.height > 0:
        src_parts.append(_cnn.select(
            alias_to_axis("connection", "p"), alias_to_axis("node_1", "source")))
        snk_parts.append(_cnn.select(
            alias_to_axis("connection", "p"), alias_to_axis("node_2", "sink")))
    if src_parts:
        flex_data.process_source_canonical = (
            pl.concat(src_parts).unique().sort("p", "source")
        )
    if snk_parts:
        flex_data.process_sink_canonical = (
            pl.concat(snk_parts).unique().sort("p", "sink")
        )

    # commodity__node join — read from source.
    cn = source.get_entities("commodity__node") if hasattr(source,
                                                              "get_entities") else None
    if cn is None:
        # Try via _try_entities helper (shared with projection_params).
        from flextool.engine_polars._projection_params import _try_entities
        cn = _try_entities(source, "commodity__node")
    if cn is not None and cn.height > 0:
        # commodity__node carries (commodity, node) columns; align with
        # the canonical-branch shape used by _load_process_topology.
        if "commodity" in cn.columns and "node" in cn.columns:
            flex_data.flow_from_commodity_eff = (
                pss_eff
                .join(cn, left_on="source", right_on="node", how="inner")
                .pipe(rename_to_axis, {"commodity": "c"})
                .select("p", "source", "sink", "c")
            )
            flex_data.flow_from_commodity_noEff = (
                pss_noEff
                .join(cn, left_on="source", right_on="node", how="inner")
                .pipe(rename_to_axis, {"commodity": "c"})
                .select("p", "source", "sink", "c")
            )
            flex_data.flow_to_commodity = (
                pss
                .join(cn, left_on="sink", right_on="node", how="inner")
                .pipe(rename_to_axis, {"commodity": "c"})
                .select("p", "source", "sink", "c")
            )

    # ``p_commodity_price`` — placeholder empty Param (model.py invariant
    # requires non-None when topology is non-empty).  apply_direct_params
    # may have already overlaid the real value; respect that.
    if flex_data.p_commodity_price is None:
        flex_data.p_commodity_price = Param(
            ("c", "d", "t"),
            pl.DataFrame(schema={
                "c": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                "value": pl.Float64,
            }),
        )

    # Δ.26: ``p_flow_upper`` is now produced natively by
    # :func:`flextool.engine_polars._derived_params.p_flow_upper_from_source`
    # in :func:`apply_derived_c`.  We seed an empty Param here purely so
    # the model.py ``PROCESSES`` invariant (non-None when topology is
    # non-empty) is satisfied even when the override chain skips this
    # field — e.g. the rare degenerate fixture with no explicit
    # ``existing`` and no invest method.  The override chain overwrites
    # this seed when it has data; downstream consumers always join on
    # the (p, source, sink, d, t) key so an empty seed is a no-op.
    flex_data.p_flow_upper = Param(
        ("p", "source", "sink", "d", "t"),
        pl.DataFrame(schema={
            "p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
            "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        }),
    )


def _populate_pss_dt_and_balance_dt(flex_data: "FlexData") -> None:
    """Build ``pss_dt`` (process_source_sink × dt) and ``nodeBalance_dt``
    (nodeBalance × dt) from the now-populated ``dt`` + topology fields.

    ``pss_dt`` is required for ``v_flow``'s domain;
    ``nodeBalance_dt`` for ``v_state_up`` / ``v_state_down`` slack
    domains.  Both are simple cross-joins, mirroring ``input.py:898``
    (``pss.join(dt, how="cross")``) and the cross-join inside
    ``_load_node`` for ``nodeBalance_dt``.
    """
    dt = getattr(flex_data, "dt", None)
    if dt is None or dt.height == 0:
        raise FastLoadError(
            "fast path requires a non-empty `dt` after the override "
            "chain.  Helper `dt_and_step_duration_from_source` "
            "(apply_derived_a step 1) returned no rows for the active "
            "solve — check that `solve.realized_periods` and "
            "`solve.period_timeset` are populated for this scenario."
        )

    pss = getattr(flex_data, "process_source_sink", None)
    if pss is not None and pss.height > 0:
        flex_data.pss_dt = pss.join(dt, how="cross")

    nb = getattr(flex_data, "nodeBalance", None)
    if nb is not None and nb.height > 0:
        flex_data.nodeBalance_dt = nb.join(dt, how="cross")

    # ``nodeState_dt`` — required by the storage feature.  Cross-join
    # of nodeState (set by apply_projection_params) × dt.  Mirrors
    # ``input.py:_load_storage`` lines that build it from the seed.
    ns = getattr(flex_data, "nodeState", None)
    if ns is not None and ns.height > 0:
        flex_data.nodeState_dt = ns.join(dt, how="cross")
        # ``nodeState_first_dt`` — first (d, t) per (n) for the
        # storage-fix-start equality.  Lexicographically smallest
        # period × smallest timestep within that period.  Mirrors
        # ``input.py:_load_storage`` lines 1773-1783.
        first_period = (dt.select("d").unique().sort("d").head(1))
        flex_data.nodeState_first_dt = (flex_data.nodeState_dt
            .join(first_period, on="d", how="inner")
            .group_by("n", "d")
            .agg(pl.col("t").min().alias("t"))
            .select("n", "d", "t"))


# ---------------------------------------------------------------------------
# Active-solve resolution from the source.
# ---------------------------------------------------------------------------


def _resolve_single_solve_name(reader: "SpineDbReader") -> str:
    """Pick the active solve name from the source for fast mode.

    Mirrors the slow path's ``orchestration.run_model`` logic for the
    single-solve case: take ``model.solves`` (a list) for the only
    model, return its first element.  Multi-solve / multi-model
    fixtures are out of fast-path scope.
    """
    from flextool.engine_polars._projection_params import _try_param

    # ``model.solves`` is an Array param.  source returns it as
    # ``[name, value]`` rows where ``value`` is the solve name string.
    solves_param = _try_param(reader, "model", "solves")
    if solves_param is None or solves_param.height == 0:
        # Try via solve entity class — single-solve fixtures may store
        # the solve directly there without a model.solves Array.
        try:
            solve_ents = reader.entities("solve")
        except KeyError:
            solve_ents = None
        if solve_ents is None or solve_ents.height == 0:
            raise FastLoadError(
                "fast path: no `model.solves` Array found in the source "
                "and the `solve` entity class is empty.  Cannot resolve "
                "the active solve name."
            )
        # Pick the first solve.
        first_col = next(c for c in solve_ents.columns
                         if c != "id" and c != "elements")
        return solve_ents[first_col][0]
    # ``solves_param.value`` is the Spine Array element.  Take the
    # first row.
    val_col = "value" if "value" in solves_param.columns else solves_param.columns[-1]
    return str(solves_param[val_col][0])


# ---------------------------------------------------------------------------
# Synthetic source object for _apply_db_overrides.
# ---------------------------------------------------------------------------


@dataclass
class _SourceShim:
    """Adapter the override chain's ``_apply_db_overrides`` expects.

    The override chain reads ``source.workdir`` to derive the workdir
    for the per-solve passes (apply_derived_a..g).  We feed it the
    fast path's work_folder so those helpers' ``ctx.read_csv`` calls
    can fall through (the workdir's solve_data dir is intentionally
    empty in fast mode — helpers that need a file return None and the
    chain skips that override).
    """

    workdir: Path
    input_dir: Path

    @property
    def solve_data_dir(self) -> Path:
        return self.workdir / "solve_data"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def load_flextool_source_only(
    reader: "SpineDbReader",
    work_folder: Path,
    *,
    logger: logging.Logger | None = None,
) -> "FlexData":
    """Build a :class:`FlexData` directly from *reader*, skipping all
    workdir CSV reads.

    Steps:
      1. Construct empty FlexData with sentinel required fields.
      2. Apply Direct + Projection Params (passes 1-2; source-only).
      3. Apply Derived A-G + existing chain (passes 3-9; may consult
         a ctx-cached workdir CSV but tolerates absence).
      4. Patch in topology fields not covered by the override chain
         (``process_source_sink``, ``pss_dt``, ``nodeBalance_dt``, …).
      5. Validate that the LP-required fields are populated; raise
         loudly otherwise.

    Parameters
    ----------
    reader : SpineDbReader
        Pre-constructed Spine reader.  Its ``db_url`` + ``scenario``
        identify the data source.
    work_folder : Path
        Workdir for the solve.  Must already exist; the function
        creates ``solve_data/`` and ``output_raw/`` subfolders if
        absent so the output writer adapter has somewhere to land
        artefacts.  No CSVs are written into ``solve_data/`` by this
        function — the per-solve preprocessing chain is bypassed
        entirely.
    logger : logging.Logger, optional
        Logger.  Defaults to a module-named logger.

    Returns
    -------
    FlexData
        Fully constructed.  Caller passes it to
        :func:`flextool.engine_polars.model.build_flextool` to build
        the LP.

    Raises
    ------
    FastLoadError
        When a required FlexData field can't be populated — the fast
        path does not silently fall back.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_folder = Path(work_folder)
    (work_folder / "solve_data").mkdir(parents=True, exist_ok=True)
    (work_folder / "output_raw").mkdir(parents=True, exist_ok=True)
    (work_folder / "input").mkdir(parents=True, exist_ok=True)

    # The override chain's per-solve helpers (apply_derived_a..g) read
    # ``solve_data/solve_current.csv`` to determine the active solve.
    # In fast mode we don't run flextool's preprocessing, so we
    # synthesize a minimal ``solve_current.csv`` from the source's
    # ``model.solves`` entry: pick the first solve in
    # ``model_solve[<the_one_model>]``.  This is the same logic the
    # slow path's preprocessing arrives at on a single-solve fixture.
    active_solve = _resolve_single_solve_name(reader)
    (work_folder / "solve_data" / "solve_current.csv").write_text(
        f"solve\n{active_solve}\n"
    )
    # Also a minimal p_model.csv so _read_solve_first defaults to True.
    (work_folder / "solve_data" / "p_model.csv").write_text(
        "modelParam,p_model\nsolveFirst,1\n"
    )

    # 1. Empty FlexData stub.
    flex_data = _empty_flex_data()

    # 2. Topology not covered by the override chain — populate FIRST
    # so apply_derived_b's ``p_unitsize_from_source`` /
    # ``p_flow_constraint_coef_from_source`` see ``process_source_sink``.
    # The override chain reads but never writes ``process_source_sink``;
    # if we don't seed it pre-chain, downstream Params keyed on the pss
    # tuple stay None.
    _populate_topology(flex_data, reader)

    # Build a shim source object for _apply_db_overrides.
    source_shim = _SourceShim(
        workdir=work_folder,
        input_dir=work_folder / "input",
    )

    # Phase 2 multi-block fast-path: build BlockLayout source-only
    # (Phase 1's ``BlockLayout.from_source``) so the override chain's
    # block-aware helpers (``nodeStateBlock_from_source`` Branch 2,
    # ``period_block_family_from_source`` multi-resolution synthesis,
    # ``arc_block_dt_from_source``, ``load_block_bundle``) can consume
    # the in-memory frames instead of looking for ``solve_data/`` CSVs
    # that won't exist on the fast path (no preprocessing has run).
    try:
        from flextool.engine_polars._block_layout import BlockLayout
        from flextool.engine_polars._solve_config import SolveConfig
        from flextool.engine_polars._timeline import TimelineConfig

        sc = SolveConfig.load_from_source(reader)
        tc = TimelineConfig.load_from_source(reader)
        tc.create_assumptive_parts(sc)
        tc.create_timeline_from_timestep_duration(sc)
        flex_data.block_layout = BlockLayout.from_source(
            reader, sc, tc, active_solve=active_solve,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "fast path: failed to build in-memory BlockLayout: %s; "
            "multi-block helpers will see no block data and the LP may "
            "be over-constrained on fixtures that use coarse blocks.",
            exc,
        )

    # 3-4. Override chain.  Late-import to avoid cycles.
    from flextool.engine_polars.input import _apply_db_overrides
    _apply_db_overrides(flex_data, reader, source_shim, ctx=None)

    # 5. pss_dt / nodeBalance_dt — depend on dt (which the chain
    # populates in apply_derived_a) AND on topology (which we seeded
    # above).  Both must run before this.
    _populate_pss_dt_and_balance_dt(flex_data)

    # 5. Validate the absolute minimum.
    _validate_required_fields(flex_data)

    # Δ.25: assign Param names so the LP-build's reflection (``Param.name``)
    # matches what the slow path produces.
    from flextool.engine_polars.input import _assign_param_names
    return _assign_param_names(flex_data)


def _validate_required_fields(flex_data: "FlexData") -> None:
    """Hard-check the FlexData fields that ``model.build_flextool``
    treats as invariants.

    Per the Δ.25 design, this raises :class:`FastLoadError` rather
    than silently proceeding.  The error message names the exact
    field that's empty so the operator can identify which override
    helper has a coverage gap on this fixture.
    """
    required: list[tuple[str, str]] = [
        ("dt", "DataFrame"),
        ("p_step_duration", "Param"),
        ("nodeBalance", "DataFrame"),
        ("p_inflow", "Param"),
    ]
    # If BlockLayout carries non-default (intraperiod-block) storage,
    # require the storage block sets.  This guards future regressions:
    # a helper that silently returns None for nodeStateBlock would
    # currently produce a wrong-but-non-zero obj (the +21.6 % bug
    # Phase 2 fixed); the dense vs block-relaxed LP substitution is
    # invisible without an explicit invariant.  Note that every fixture
    # has at least the trivial ``default`` block in the layout, so
    # ``is_empty()`` is not a useful trigger here — we gate on the
    # presence of non-default block names.
    bl = getattr(flex_data, "block_layout", None)
    if bl is not None and not bl.is_empty():
        block_names = set(
            bl.block_step_duration_frame["block"].unique().to_list()
        )
        if block_names - {"default"}:
            required.extend([
                ("nodeStateBlock", "DataFrame"),
                ("period_block", "DataFrame"),
                ("period_block_succ", "DataFrame"),
                ("period_block_time", "DataFrame"),
            ])
    for field, kind in required:
        v = getattr(flex_data, field, None)
        if v is None:
            raise FastLoadError(
                f"fast path: required field `{field}` is None after the "
                f"override chain.  Check the Spine DB has the data and "
                f"the corresponding helper is wired in apply_derived_*."
            )
        # Heuristic non-empty check.
        frame = v.frame if hasattr(v, "frame") else v
        if frame is not None and hasattr(frame, "height") and frame.height == 0:
            raise FastLoadError(
                f"fast path: required field `{field}` is empty after the "
                f"override chain — the helper produced no rows for this "
                f"scenario.  Field kind: {kind}.  Likely root cause: "
                f"missing entity / parameter rows in the source DB."
            )
