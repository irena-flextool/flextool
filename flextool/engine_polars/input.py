"""Read flextool ``input/`` + ``solve_data/`` into a single
``FlexData`` bag.

All fields are optional — None / empty when the scenario doesn't
exercise that feature.  ``build_flextool(p, d)`` switches on field
presence to decide which constraints / variables / objective terms
to add.

Pipeline shape today:

    Spine DB → flextool preprocess → input/ + solve_data/ CSVs → load_flextool

Once flextool's preprocessing migration to Python is complete, this
module gains a parallel entry point that consumes the in-memory
preprocessing state directly, skipping the CSV roundtrip.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl
from polar_high import Param

if TYPE_CHECKING:
    from ._input_source import FlexInputSource, InputSource


# ---------------------------------------------------------------------------
# Diagnostic gate for `__setattr__` on FlexData.  Production code never
# arms this — it's a tool for ``tests/engine_polars/test_native_cascade_parity.py``
# (and ad-hoc investigation) to compare the cascade's native-override
# objective against the CSV-seed-preserved objective on the same
# fixture.  See ``specs/native_cascade_parity.md`` for the bug class
# this diagnoses.
#
# When ``_CASCADE_GATE_ACTIVE`` is True (set via ``_cascade_gate()``),
# ``FlexData.__setattr__`` drops any reassignment of a field that
# already carries a non-None value.  Construction, handoff overlay,
# and ``apply_existing_chain`` are unaffected — only direct
# assignments inside the gated context window are filtered.

_CASCADE_GATE_ACTIVE = False


@contextmanager
def _cascade_gate():
    """Test-only: while active, ``FlexData.__setattr__`` drops
    reassignments of fields that already have a seeded value.  Used
    by the cascade parity test to A/B the cascade override path
    against the CSV-seed preservation path.
    """
    global _CASCADE_GATE_ACTIVE
    prev = _CASCADE_GATE_ACTIVE
    _CASCADE_GATE_ACTIVE = True
    try:
        yield
    finally:
        _CASCADE_GATE_ACTIVE = prev

from . import _group_slack
from . import _reserve
from . import _cumulative_invest
from . import _delay
from . import _dc_power_flow
from . import _commodity_ladder
from ._block_layout import BlockLayout
from ._input_source import read_csv_fallback
from ._emit_provider_io import _provider_key


# ---------------------------------------------------------------------------
# Provider-aware access helpers.
#
# Post-Step-2 the cascade prefers the live :class:`FlexDataProvider` as
# the single source of truth for preprocessing artefacts.  The Provider
# carries every frame emitted by writers in ``_PATCH_MODULES`` (and any
# downstream Step 1-g writer that wired ``put(name, frame)`` into its
# emit path).  A disk arm is retained for the *raw* fixture inputs under
# ``input/`` which are produced by the legacy ``input_writer``
# preprocessing pass and are NOT captured into the Provider — they live
# on disk for the cascade to consume.  Once that legacy preprocessing
# is folded into the Provider (a separate, larger refactor), the disk
# arm can collapse to ``None``.

def _provider_has(provider: "object | None", name: str,
                   path: "Path | str") -> bool:
    """True iff *provider* has *name* OR *path* exists on disk."""
    if provider is not None and provider.has(name):
        return True
    return Path(path).exists()


def _provider_read(provider: "object | None", name: str,
                    path: "Path | str") -> pl.DataFrame:
    """Return the frame for *name* from *provider*.

    Cascade callers always thread a Provider seeded by the cascade-input
    emitters (``input_derivation.run`` and the per-solve preprocessing
    chain).  Off-cascade callers (the loader-unit tests in
    ``tests/engine_polars/loaders``) pass ``provider=None``; for those
    the residual disk-read goes through
    :func:`flextool.engine_polars._input_source.read_csv_fallback` so
    Rule 1 of ``tests/engine_polars/test_meta_provider_invariants.py``
    can confirm input.py never calls ``_read_csv_file`` /
    ``pl.read_csv`` directly.
    """
    if provider is not None and provider.has(name):
        return provider.get(name)
    return read_csv_fallback(path)


def _provider_pick(provider: "object | None",
                    *candidates: "tuple[str, Path]") -> "Path | None":
    """Return the first ``Path`` for which the Provider has the matching
    *name* OR the disk file exists; ``None`` if none.
    """
    for name, path in candidates:
        if (provider is not None and provider.has(name)) or Path(path).exists():
            return Path(path)
    return None


def _provider_open(provider: "object | None", name: str,
                    path: "Path | str"):
    """Open a file-like handle for *name* sourced from the Provider or
    from disk; return ``None`` when neither has the file.
    """
    if provider is not None and provider.has(name):
        import io
        df = provider.get(name)
        buf = io.StringIO()
        df.write_csv(buf)
        buf.seek(0)
        return buf
    p = Path(path)
    if p.exists():
        return p.open()
    return None
from ._axis_enums import (  # substrate retained for Path B — see handoff
    alias_to_axis,
    cast_dim,
    cast_frame_axes,
    cast_value_axes,
    cast_flexdata_axes,
    get_global_axis_enums,
    rename_to_axis,
    lit_axis,
    set_global_axis_enums,
)


# ---------------------------------------------------------------------------
# CSV-shape helpers (same three shapes as before)

def _read_long(path: Path, *, drop=("solve",), rename=None,
               cast_value: bool = True,
               provider: "object | None" = None) -> pl.DataFrame:
    df = _provider_read(provider, _provider_key(path), path)
    df = df.drop([c for c in drop if c in df.columns])
    if rename: df = df.pipe(rename_to_axis, rename)
    if cast_value and "value" in df.columns:
        df = df.with_columns(value=pl.col("value").cast(pl.Float64,
                                                        strict=False))
    return df


def _read_wide_per_entity(path: Path, value_col: str = "value",
                          rename=None,
                          *,
                          provider: "object | None" = None) -> pl.DataFrame:
    """Reads a wide-per-entity CSV (header: solve, period, time, e1, e2,…)
    OR a long CSV (header: <entity_col>, period, time, value) — the new
    Python-preprocessed format.  In the long case the entity column is
    whatever flextool's preprocessor wrote (node / process / commodity
    /…); the caller's ``rename={'entity': X}`` is applied either way."""
    df = _provider_read(provider, _provider_key(path), path)
    if "value" in df.columns and "solve" not in df.columns:
        # Long format from new Python preprocessing.  Entity column is
        # the first; rename to "entity" to keep the downstream contract.
        # Use a plain ``.rename`` (not ``rename_to_axis``) for the
        # entity column: the entity_col may carry values that belong to
        # a non-``e`` axis (e.g. ``profile`` → ``f``-axis names like
        # ``wind_profile``), and casting via ``rename_to_axis`` to the
        # synonym ``entity`` would coerce the column to the e-axis Enum
        # — nulling every value that isn't in the entity-union vocab.
        # The caller's ``rename={"entity": <target>}`` below performs the
        # correct axis cast in one step.
        entity_col = df.columns[0]
        out = (df.rename({entity_col: "entity"})
                 .pipe(rename_to_axis, {"period": "d", "time": "t"})
                 .with_columns(value=pl.col(value_col).cast(pl.Float64,
                                                            strict=False))
                 .select("entity", "d", "t", "value"))
        out = out.with_columns(value=pl.col("value").fill_null(0.0))
        if rename: out = out.pipe(rename_to_axis, rename)
        return out
    # Legacy wide-per-entity format.
    df = df.drop("solve")
    id_cols = ["period", "time"]
    value_cols = [c for c in df.columns if c not in id_cols]
    out = (df.unpivot(on=value_cols, index=id_cols, variable_name="entity",
                      value_name=value_col)
             .pipe(rename_to_axis, {"period": "d", "time": "t"}))
    if rename: out = out.pipe(rename_to_axis, rename)
    return out


def _read_unitsize(path: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    """Read ``p_entity_unitsize.csv``.  The canonical Python-preprocessing
    output is long-format ``(entity, value)`` in ``solve_data/``.  The
    ``.mod`` also printf's a wide-format twin to ``input/`` (one row,
    columns are entity names) — supported as a fallback for legacy
    fixtures."""
    df = _provider_read(provider, _provider_key(path), path)
    if {"entity", "value"}.issubset(df.columns):
        return (df.pipe(rename_to_axis, {"entity": "e"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False))
                  .select("e", "value"))
    # legacy wide format: drop the first column (label "entity"/"value"),
    # then transpose so column names become rows.
    df = df.drop(df.columns[0])
    return (df.transpose(include_header=True, header_name="e",
                         column_names=["value"])
              .with_columns(value=pl.col("value").cast(pl.Float64)))


def _read_capacity(path: Path,
                    previously_invested_path: Path | None = None,
                    all_existing_path: Path | None = None,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    # ``p_entity_all_existing`` is the cumulative existing capacity per
    # period (reflecting lifetime, carried over across periods within a
    # solve), which is what the .mod's ``p_entity_dispatch_capacity_max``
    # formula uses.  ``_emit_chain_params.emit_p_entity_existing_chain``
    # (called per-iter by ``_emit_solve_time.run`` batch 59,
    # unconditionally) populates ``solve_data/p_entity_all_existing`` in
    # the Provider, so the cascade always has this key available.
    df = _provider_read(
        provider, "solve_data/p_entity_all_existing", all_existing_path)
    if "solve" in df.columns: df = df.drop("solve")
    # Long-format variant: columns are (entity, period, value).
    if {"entity", "period", "value"}.issubset(df.columns):
        return (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                  .with_columns(value=pl.col("value")
                                        .cast(pl.Float64, strict=False)
                                        .fill_null(0.0))
                  .select("e", "d", "value"))
    # Wide-format variant: columns are (period, entity1, entity2, …).
    val_cols = [c for c in df.columns if c != "period"]
    if df.height == 0 or not val_cols:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    return (df.unpivot(on=val_cols, index=["period"], variable_name="e",
                       value_name="value")
              .pipe(rename_to_axis, {"period": "d"})
              .with_columns(value=pl.col("value")
                                    .cast(pl.Float64, strict=False)
                                    .fill_null(0.0))
              .select("e", "d", "value"))


def _read_p_flow_max(
    path: Path,
    provider: "object | None" = None,
) -> pl.DataFrame | None:
    """Read flextool's canonical ``solve_data/p_flow_max.csv`` long-format
    file ``[process, source, sink, period, time, value]`` (the same file
    flextool.mod consumes via ``table data IN``).

    Step 1-b — pilot reader migrated to :class:`FlexDataProvider`.  When
    *provider* is supplied, the existence guard + frame fetch go through
    ``provider.has`` / ``provider.get``; the *path* argument is then
    redundant (kept for the dual-write window — it is removed in a
    cleanup commit once every reader is migrated).  The lookup key
    ``"solve_data/p_flow_max"`` is the parent-qualified, suffix-stripped
    form of *path* — it matches the emitter's dual-key registration
    (basename + parent/basename); using the qualified form keeps the
    read disambiguated even if a same-basename frame appears under
    another parent dir later.

    When *provider* is ``None`` the legacy seed funnel path runs
    unchanged.
    """
    name = "solve_data/p_flow_max"
    if not _provider_has(provider, name, path):
        return None
    df = _provider_read(provider, name, path)
    if df is None or df.height == 0:
        return None
    return df.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"}) \
             .select("p", "source", "sink", "d", "t", "value")


def _slice_param(path: Path, entity_col: str, param_value: str,
                 has_time: bool = True,
                 rename_entity_to: str | None = None,
                 *,
                 provider: "object | None" = None) -> pl.DataFrame | None:
    """Slice a generic param-bearing canonical input
    (``pdtNode.csv``/``pdtCommodity.csv``/``pdtProcess.csv``/``pdProcess.csv``/``pdtGroup.csv``)
    by a literal ``param`` string — the same operation .mod does inline
    via e.g. ``pdtNode[n, 'penalty_up', d, t]``.

    Returns ``(entity, d, t, value)`` (or ``(entity, d, value)`` when
    ``has_time=False``) or ``None`` if the file is missing or the slice
    is empty.  ``rename_entity_to`` renames the entity column for
    downstream consumers (e.g. ``"node" -> "n"``)."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return None
    df = _provider_read(provider, name, path)
    if df.height == 0:
        return None
    sliced = df.filter(pl.col("param") == param_value).drop("param")
    if sliced.height == 0:
        return None
    rename = {"period": "d"}
    if has_time:
        rename["time"] = "t"
    if rename_entity_to is not None:
        rename[entity_col] = rename_entity_to
    out = sliced.pipe(rename_to_axis, rename)
    # ``pdtProcess.csv`` / ``pdtNode.csv`` / ``pdtCommodity.csv`` / ``pdtGroup.csv``
    # / ``pdProcess.csv`` are emitted by the writer-port with every column
    # (including ``value``) as ``Utf8`` — values round-trip through
    # ``repr(v)`` so heterogeneous param leaf types (int methods + float
    # data) survive byte-identically (see
    # :func:`flextool.engine_polars._emit_pdt_params.derive_pdtProcess`).
    # When a numeric param (e.g. ``availability``) is sliced out for use
    # as a polars-numeric Param ``value`` column, the Utf8 must be cast
    # back to ``Float64`` at the producer — otherwise downstream
    # ``Param × Param`` multiplications (e.g. ``flow_upper_rhs *
    # p_process_availability`` at ``model.py:1348``) raise
    # ``InvalidOperationError: arithmetic on string and numeric``.
    # ``strict=False`` mirrors the casts in ``_read_long`` / pdtNode
    # path (input.py:958) — non-parseable values become null and the
    # downstream consumer's ``fill_null(0.0)`` handles them.
    if "value" in out.columns and out.schema["value"] == pl.Utf8:
        out = out.with_columns(
            value=pl.col("value").cast(pl.Float64, strict=False))
    cols = [rename.get(entity_col, entity_col), "d"] + (["t"] if has_time else []) + ["value"]
    return out.select(cols)


def _read_step_previous(path: Path,
                         *,
                         provider: "object | None" = None) -> pl.DataFrame | None:
    """Read flextool's canonical ``solve_data/step_previous.csv`` (the
    same file .mod reads as the ``dtttdt`` set, see flextool.mod:786).
    Renames columns to the names the downstream ``Lag`` call sites
    expect (``t_previous``, ``t_previous_within_timeset``, ``d_previous``,
    ``t_previous_within_solve``)."""
    name = "solve_data/step_previous"
    if not _provider_has(provider, name, path):
        return None
    df = _provider_read(provider, name, path)
    rename = {
        "period": "d", "time": "t",
        "previous": "t_previous",
        "previous_within_timeset": "t_previous_within_timeset",
        "previous_period": "d_previous",
        "previous_within_solve": "t_previous_within_solve",
    }
    out = df.pipe(rename_to_axis, {k: v for k, v in rename.items() if k in df.columns})
    keep = [v for v in rename.values() if v in out.columns]
    return out.select(keep)


# ---------------------------------------------------------------------------
# Single FlexData container

@dataclass
class FlexData:
    """Naming convention:

    * **Sets** (index frames, ``pl.DataFrame``) — no prefix.
      Examples: ``nodeBalance``, ``process_source_sink``, ``flow_to_n``,
      ``cdt_eq``, ``nodeState``.
    * **Parameters** (numeric ``Param``) — ``p_`` prefix.
      Examples: ``p_inflow``, ``p_unitsize``, ``p_commodity_price``.

    Variables created by ``build_flextool`` use ``v_`` for primal,
    ``vq_`` for slack — same convention as flextool.mod.
    """

    # ─── Time / weighting (always present) ────────────────────────────────
    dt: pl.DataFrame                         # set: (d, t)
    p_step_duration: Param                   # (d, t)
    p_rp_cost_weight: Param                  # (d, t)
    p_inflation_op: Param                    # (d,)
    p_period_share: Param                    # (d,)

    # ─── Nodes (always present in tested scenarios) ───────────────────────
    nodeBalance: pl.DataFrame                # set: (n,)
    p_inflow: Param                          # (n, d, t)
    p_penalty_up: Param                      # (n, d, t)
    p_penalty_down: Param                    # (n, d, t)

    # Per-period years-represented R (e.g. 5.0 for a 5-year invest period).
    # Built from ``solve.years_represented`` via
    # ``p_years_represented_d_from_source``.  None when the source
    # carries no ``solve.years_represented`` rows (single-year fixtures
    # default each period to width 1).
    p_years_represented_d: Param | None = None  # (d,)

    # ─── Process topology  ───────────────────────────────────────────────
    process_source_sink: pl.DataFrame | None = None
    process_source_sink_eff: pl.DataFrame | None = None
    process_source_sink_noEff: pl.DataFrame | None = None
    # Phase E.3: ``pss_dt`` / ``nodeBalance_dt`` / ``nodeState_dt`` /
    # ``nodeState_first_dt`` / ``process_indirect_dt`` are no longer
    # eager-built up-front.  Consumers call the on-demand helpers in
    # :mod:`flextool.engine_polars._pdt_join`.  Fields stay declared
    # (defaulting to ``None``) so callers that defensively read them via
    # ``getattr(d, ..., None)`` continue to work and the warm-update
    # over_field plumbing can fall through to cold-rebuild cleanly.
    pss_dt: pl.DataFrame | None = None
    nodeBalance_dt: pl.DataFrame | None = None
    # Canonical (p, source) / (p, sink) per process — one row per unit input
    # node / output node, and one row per connection using the original
    # connection__node__node direction (not the added reverse arc).
    process_source_canonical: pl.DataFrame | None = None
    process_sink_canonical: pl.DataFrame | None = None
    flow_to_n: pl.DataFrame | None = None
    flow_from_n: pl.DataFrame | None = None
    flow_from_commodity_eff: pl.DataFrame | None = None
    flow_from_commodity_noEff: pl.DataFrame | None = None
    flow_to_commodity: pl.DataFrame | None = None  # §2.4 sell into priced commodity node
    p_unitsize: Param | None = None              # (p,)
    p_all_entity_unitsize: Param | None = None  # (e,) — all entities (processes + connections + nodes); used by scaling
    p_flow_upper: Param | None = None            # (p, source, sink, d, t) — preprocessed structural max (existing + max_invest_cum)
    p_flow_upper_existing: Param | None = None   # (p, source, sink, d) — existing/unitsize only; used by maxToSink
    p_slope: Param | None = None                 # (p, d, t)
    p_commodity_price: Param | None = None       # (c, d, t)
    pd_neg_cap: pl.DataFrame | None = None       # set: (p, d) where existing<0 AND unitsize<0
                                                  # (anti-energy semantics: forces v_flow ≥ |existing|/|unitsize|)

    # ─── CO2 price ────────────────────────────────────────────────────────
    flow_from_co2_priced: pl.DataFrame | None = None
    flow_from_co2_priced_noEff: pl.DataFrame | None = None
    p_co2_content: Param | None = None           # (c,)
    p_co2_price: Param | None = None             # (g, d, t)

    # ─── CO2 cap (period) ─────────────────────────────────────────────────
    group_co2_max_period: pl.DataFrame | None = None
    flow_from_co2_capped: pl.DataFrame | None = None        # eff partition (slope)
    flow_from_co2_capped_noEff: pl.DataFrame | None = None  # noEff partition (no slope)
    p_co2_max_period: Param | None = None        # (g, d)
    group_d_co2_capped: pl.DataFrame | None = None

    # ─── CO2 cap (multi-period total) — port of v3.32.0 co2_max_total ────
    # Mirrors co2_max_period but the cap is a single tonnes value per group
    # spanning the whole horizon.  Active only for groups whose
    # ``group__co2_method`` is ``total`` / ``price_total`` / ``period_total``
    # (set materialised in ``solve_data/group_co2_max_total.csv``).  LHS sums
    # across (d, t); RHS is the per-group ``p_group[g, 'co2_max_total']``
    # scalar (in tonnes).  See .mod:4019-4055 for the legacy formulation.
    group_co2_max_total: pl.DataFrame | None = None              # (g,)
    flow_from_co2_capped_total: pl.DataFrame | None = None       # eff partition
    flow_from_co2_capped_total_noEff: pl.DataFrame | None = None # noEff partition
    p_co2_max_total: Param | None = None         # (g,)

    # ─── Indirect-conversion (CHP) ────────────────────────────────────────
    process_indirect: pl.DataFrame | None = None
    process_input_flows: pl.DataFrame | None = None
    process_output_flows: pl.DataFrame | None = None
    process_indirect_dt: pl.DataFrame | None = None
    # Per-arc multipliers on the source / sink side of the
    # ``conversion_indirect`` equation (.mod:2557-2580).  Both default to
    # 1.0 (the absent-Param convention) and are only populated when at
    # least one row in the corresponding
    # ``input/p_process_*_conversion_flow_coeff.csv`` has a non-default,
    # non-zero value.  When populated, the Param covers *all* relevant
    # (p, source) / (p, sink) rows of the indirect inputs / outputs
    # (filled to 1.0 where the CSV is silent), so multiplying
    # ``v_flow * unitsize * Param`` won't drop any flows.
    p_process_source_conversion_flow_coeff: Param | None = None  # (p, source)
    p_process_sink_conversion_flow_coeff: Param | None = None    # (p, sink)

    # ─── User-defined flow constraints ────────────────────────────────────
    flow_constraint_idx: pl.DataFrame | None = None  # (p, source, sink, cn)
    p_flow_constraint_coef: Param | None = None  # (p, source, sink, cn)
    p_constraint_constant: Param | None = None   # (cn,)
    cdt_eq: pl.DataFrame | None = None  # (cn, d, t)
    cdt_le: pl.DataFrame | None = None  # (cn, d, t)
    cdt_ge: pl.DataFrame | None = None  # (cn, d, t)
    p_node_constraint_invested_capacity_coeff: Param | None = None  # (n, cn)
    p_process_constraint_invested_capacity_coeff: Param | None = None  # (p, cn)
    p_node_constraint_state_coeff: Param | None = None  # (n, cn) — user-cstr v_state coefficient
    p_node_constraint_prebuilt_capacity_coeff: Param | None = None  # (n, cn)
    p_process_constraint_prebuilt_capacity_coeff: Param | None = None  # (p, cn)

    # ─── Profiles ─────────────────────────────────────────────────────────
    process_profile_upper: pl.DataFrame | None = None    # (p, source, sink, f)
    process_profile_lower: pl.DataFrame | None = None
    process_profile_fixed: pl.DataFrame | None = None
    p_profile_value: Param | None = None         # (f, d, t)
    p_process_existing_count: Param | None = None  # (p, d) = cap / unitsize
    p_process_availability: Param | None = None  # (p, d, t)

    # ─── Invest / divest ──────────────────────────────────────────────────
    ed_invest_set: pl.DataFrame | None = None        # (e, d) — invest var index
    ed_divest_set: pl.DataFrame | None = None        # (e, d) — divest var index
    pd_invest_set: pl.DataFrame | None = None        # (p, d) — process-side
    pd_divest_set: pl.DataFrame | None = None        # (p, d)
    nd_invest_set: pl.DataFrame | None = None        # (n, d) — node-side
    nd_divest_set: pl.DataFrame | None = None        # (n, d)
    edd_invest_set: pl.DataFrame | None = None       # (e, d_invest, d)
    edd_invest_lookback_set: pl.DataFrame | None = None  # (e, d_invest, d) strict d_invest<d
    edd_divest_active: pl.DataFrame | None = None    # (p, d_divest, d) where d_divest ≤ d
    p_entity_max_units: Param | None = None          # (e, d)
    ed_lifetime_fixed_cost: Param | None = None      # (e, d)
    ed_lifetime_fixed_cost_divest: Param | None = None
    ed_entity_annual_discounted: Param | None = None
    ed_entity_annual_divest_discounted: Param | None = None
    e_invest_total: pl.DataFrame | None = None       # (e,)
    e_divest_total: pl.DataFrame | None = None
    e_invest_max_total: Param | None = None          # (e,)
    e_divest_max_total: Param | None = None
    ed_invest_period_set: pl.DataFrame | None = None  # (e, d) — entities with per-period invest cap
    ed_divest_period_set: pl.DataFrame | None = None  # (e, d)
    ed_invest_max_period: Param | None = None        # (e, d)
    ed_divest_max_period: Param | None = None        # (e, d)

    # Multi-solve handoff state — populated when running a sub-solve of
    # a chain, tells the LP what investment/divestment was realized in
    # prior sub-solves so that cumulative caps stay tight.  See
    # flextool.mod:3597-3623 (maxInvest_entity_total / maxDivest_entity_total
    # / minInvest_entity_total / minDivest_entity_total).
    p_entity_previously_invested_capacity: Param | None = None  # (e, d)
    p_entity_invested: Param | None = None      # (e,)  — cumulative prior-solve invest, used by min/max divest variants when not solveFirst
    p_entity_divested: Param | None = None      # (e,)  — cumulative prior-solve divest, used by max/min divest variants when not solveFirst

    # ─── Ramp limits ──────────────────────────────────────────────────────
    process_source_sink_ramp_limit_sink_up:   pl.DataFrame | None = None
    process_source_sink_ramp_limit_sink_down: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_up: pl.DataFrame | None = None
    process_source_sink_ramp_limit_source_down: pl.DataFrame | None = None
    # Δ.17c Gap D — process_source_sink_ramp_cost (mod L1115-1119): (p, src,
    # sink) rows whose source-side OR sink-side ramp_method ∈
    # RAMP_COST_METHOD.  Populated by ``apply_projection_params``; not yet
    # consumed by model.py (the LP doesn't carry a per-arc ramp-cost
    # objective term in the current scope) but kept for parity with
    # flextool's preprocessing set family.
    process_source_sink_ramp_cost: pl.DataFrame | None = None
    p_ramp_speed_up_sink:   Param | None = None    # (p, sink)
    p_ramp_speed_down_sink: Param | None = None
    p_ramp_speed_up_source:   Param | None = None  # (p, source)
    p_ramp_speed_down_source: Param | None = None

    # ─── Online / min_load (unit commitment) ──────────────────────────────
    process_online: pl.DataFrame | None = None              # set: (p,)
    process_online_linear: pl.DataFrame | None = None        # set: (p,)
    process_online_integer: pl.DataFrame | None = None       # set: (p,)
    process_minload: pl.DataFrame | None = None              # set: (p,)
    process_min_load_eff: pl.DataFrame | None = None  # (p,) where ct_method=min_load_efficiency
    p_online_dt: pl.DataFrame | None = None                  # set: (p, d, t) — UC var domain
    pdt_online_linear: pl.DataFrame | None = None  # (p, d, t) — startup-cost obj index, linear
    pdt_online_integer: pl.DataFrame | None = None # (p, d, t) — startup-cost obj index, integer
    p_min_load: Param | None = None                          # (p,)
    p_startup_cost: Param | None = None                      # (p, d)
    p_section: Param | None = None                           # (p, d, t)
    pdt_uptime_set: pl.DataFrame | None = None               # (p, d, t) — minimum_uptime constraint domain
    pdt_downtime_set: pl.DataFrame | None = None             # (p, d, t) — minimum_downtime constraint domain
    uptime_lookback: pl.DataFrame | None = None              # (p, d, t, d_back, t_back) — startup lookback window
    downtime_lookback: pl.DataFrame | None = None            # (p, d, t, d_back, t_back) — shutdown lookback window

    # ─── Storage ─────────────────────────────────────────────────────────
    nodeState: pl.DataFrame | None = None
    nodeState_dt: pl.DataFrame | None = None
    nodeState_first_dt: pl.DataFrame | None = None
    storage_bind_within_timeblock: pl.DataFrame | None = None
    storage_bind_forward_only: pl.DataFrame | None = None    # set: (n,)
    storage_bind_within_solve: pl.DataFrame | None = None    # set: (n,)
    storage_bind_within_solve_blended_weights: pl.DataFrame | None = None  # set: (n,)
    # Phase C — recognised by the v55 value_list but the constraint
    # implementations land in Phases D / E.  Loaded here so the
    # ``nodeBalance_eq`` guard in :mod:`flextool.engine_polars.model`
    # can raise a precise "not yet implemented" error instead of
    # silently emitting zero state-change residuals.
    storage_bind_within_period_blended_weights: pl.DataFrame | None = None  # set: (n,)
    storage_bind_forward_only_blended_weights: pl.DataFrame | None = None   # set: (n,)
    storage_fix_start: pl.DataFrame | None = None
    dtttdt: pl.DataFrame | None = None           # (d, t, t_previous_*, ...)
    dtttdt_forward_only: pl.DataFrame | None = None  # dtttdt with first (d,t) per solve dropped
    # Rolling-horizon (nested-solve) framework — flextool.mod:2196 + 2760.
    # ``p_nested_solve_first``: tri-state.  None → no p_nested_model.csv,
    # treat as single-solve (== solveFirst).  True / False — read from
    # ``solve_data/p_nested_model.csv``'s ``solveFirst`` row.
    # When False, the nodeBalance ``fwd_fix_*`` block is *replaced* with a
    # ``roll_continue`` term that pins
    # ``v_state[n, d_first, t_first] * unitsize == p_roll_continue_state[n]``.
    p_nested_solve_first: bool | None = None
    p_roll_continue_state: Param | None = None        # (n,)
    n_fix_storage_quantity: pl.DataFrame | None = None  # (n,)
    ndt_fix_storage_quantity: pl.DataFrame | None = None  # (n, d_upper, t_upper)
    p_fix_storage_quantity: Param | None = None       # (n, d_upper, t_upper)
    # Phase B4-pre — fix_storage_usage loader (constraint added in B4).
    # Mirrors the fix_storage_quantity triplet above; populated from the
    # canonical handoff key ``handoff/fix_storage_usage`` (schema
    # ``[node, period, step, p_fix_storage_usage]``).
    n_fix_storage_usage: pl.DataFrame | None = None     # (n,)
    ndt_fix_storage_usage: pl.DataFrame | None = None   # (n, d, t)
    p_fix_storage_usage: Param | None = None            # (n, d, t)
    dtt_timeline_matching: pl.DataFrame | None = None  # (d, t, t_upper) — lower→upper step map
    period_branch: pl.DataFrame | None = None         # (d_upper, d) — period→branch map
    period_last: pl.DataFrame | None = None           # (d,)
    nodeState_last_dt: pl.DataFrame | None = None     # (n, d, t) — block_period_time_last × node__block × nodeState
    # In-memory BlockLayout shared between slow and fast paths.  Populated
    # by load_flextool (slow) and load_flextool_source_only (fast); consumed
    # by nodeStateBlock_from_source and period_block_family_from_source's
    # multi-resolution synthesis branches, plus arc_block_dt_from_source and
    # load_block_bundle, so the fast path doesn't have to look for the
    # solve_data/ block CSVs that won't exist when preprocessing is skipped.
    block_layout: "BlockLayout | None" = None
    # ─── Intraperiod-block storage (bind_intraperiod_blocks) ─────────────
    nodeStateBlock: pl.DataFrame | None = None             # set: (n,)
    period_block: pl.DataFrame | None = None               # set: (d, b_first)
    period_block_succ: pl.DataFrame | None = None          # set: (d, b_first, b_next)
    period_block_time: pl.DataFrame | None = None          # set: (d, b_first, t)
    dtttdt_block_interior: pl.DataFrame | None = None      # dtttdt rows where t_previous_within_timeset == t_previous (interior-of-block jump=1)
    # ─── RP-blended-weights storage (bind_within_solve_blended_weights) ─────────
    # Eight per-solve sets / params that drive the intra-period state-change
    # branch for ``nodeState_rp`` plus the three ``rp_inter_period_*``
    # constraints (.mod:2197-2200, .mod:2965-2997).  Populated by
    # ``_load_storage`` from the Phase-1 Provider keys (see
    # ``_provider_keys.SOLVE_DATA_NODE_STATE_RP`` + siblings).  Loader-only
    # at this commit — model.py wiring lands in Phase 5+.  When
    # ``nodeState_rp`` is non-empty the loader enforces that the four
    # tightly-coupled fields (``rp_base_period_set``, ``rp_base__rep``,
    # ``rp_block_first``, ``p_rp_last_step``) are also non-empty.
    nodeState_rp: pl.DataFrame | None = None               # set: (n,)
    rp_base_period_set: pl.DataFrame | None = None         # set: (b,)
    rp_base_chain: pl.DataFrame | None = None              # set: (b, b_prev)
    rp_base_first: pl.DataFrame | None = None              # set: (b,)
    rp_base_last: pl.DataFrame | None = None               # set: (b,)
    rp_block_first: pl.DataFrame | None = None             # set: (d, t)
    # Relation r → last_step (DataFrame, not Param — see audit §6 Risk #1:
    # the .mod's ``p_rp_last_step`` is a symbolic-Param-keyed-by-value
    # pattern; Phase 7 implements ``v_state[n, d, p_rp_last_step[r]]`` as
    # a relational join on ``r → last_step``).
    p_rp_last_step: pl.DataFrame | None = None             # set: (r, last_step)
    rp_base__rep: Param | None = None                      # (b, r) → weight
    # ─── Per-arc effective block step durations (M-matrix collapsed) ──
    # Indexed (p, source, sink, d, t) with value = block_step_duration of
    # the arc's relevant side block at fine step (d, t).  Drives the daily
    # flow-aggregation in nodeBalanceBlock_eq when coarse blocks are
    # active.  None for fixtures without process_side_block.csv.
    p_arc_step_duration_sink: Param | None = None
    p_arc_step_duration_source: Param | None = None
    # ─── Per-arc-side block aggregation index for nodeBalanceBlock_eq ──
    # (p, source, sink, d, b_first, t, weight): for each (n=sink, d, b_first)
    # in nodeStateBlock, the fine timesteps t (and weights) at which v_flow
    # contributes to the daily nodeBalance via the .mod's overlap × block_
    # step_duration aggregation.  weight = block_step_duration[b_f, d, t]
    # where b_f is the arc's sink-side block.  For coarse-side arcs only
    # the coarse step (t=b_first) appears, with weight=24 (or whatever sd).
    # For fine-side arcs (e.g., electrolyser source on hourly_group when
    # h2 is sink on daily_group), all 24 fine steps appear with weight=1.
    arc_sink_block_dt: pl.DataFrame | None = None    # (p, source, sink, d, b_first, t, weight)
    arc_source_block_dt: pl.DataFrame | None = None  # (p, source, sink, d, b_first, t, weight)
    p_arc_sink_weight: Param | None = None     # (p, source, sink, d, t) → weight
    p_arc_source_weight: Param | None = None   # (p, source, sink, d, t) → weight
    flow_from_nodeBalance_eff: pl.DataFrame | None = None
    flow_from_nodeBalance_noEff: pl.DataFrame | None = None
    p_state_upper: Param | None = None           # (n, d) — capacity / unitsize
    p_state_unitsize: Param | None = None        # (n,)
    p_state_self_discharge: Param | None = None  # (n,)
    p_state_start: Param | None = None           # (n,)
    p_state_existing_capacity: Param | None = None  # (n, d)
    # ─── Storage end-state binding (use_reference_value) ─────────────────
    # mod:2802-2822 — pins v_state at the last timestep of period_last to
    # ``reference_value × existing/unitsize`` for nodes with
    # ``storage_solve_horizon_method=use_reference_value`` and no
    # competing fix_end / fix_start_end / bind_within_solve method.
    storage_use_reference_value: pl.DataFrame | None = None  # (n,)
    p_storage_state_reference_value: Param | None = None     # (n, d, t)
    # ─── Storage end-state binding (use_reference_price) ─────────────────
    # B1a — per-(node, period) reference price emitted by
    # ``_emit_arc_unions.emit_p_storage_state_reference_price`` into
    # ``solve_data/p_storage_state_reference_price.csv``.  Consumed by
    # the use_reference_price objective term (B1b) for nodes whose
    # ``storage_solve_horizon_method=use_reference_price``.  Loaded but
    # unused at the B1a commit; B1b wires it into model.py.
    p_storage_state_reference_price: Param | None = None     # (n, d)
    # ─── State-profile bounds (node__profile__profile_method) ────────────
    # (n, f) tuples for nodes with a profile-method state bound.  Mirrors
    # ``process_profile_*`` (process side) but for ``v_state``.
    node_profile_upper: pl.DataFrame | None = None  # (n, f)
    node_profile_lower: pl.DataFrame | None = None  # (n, f)
    node_profile_fixed: pl.DataFrame | None = None  # (n, f)
    p_node_availability: Param | None = None     # (n, d, t) — slice of pdtNode availability

    # ─── Process variable cost (other_operational_cost) ──────────────────
    pssdt_varCost_noEff: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_source: pl.DataFrame | None = None
    pssdt_varCost_eff_unit_sink: pl.DataFrame | None = None
    pssdt_varCost_eff_connection: pl.DataFrame | None = None
    p_pssdt_varCost: Param | None = None     # (p, source, sink, d, t)
    p_pdt_varCost_source: Param | None = None  # (p, source, d, t) — eff source O&M
    p_pdt_varCost_sink: Param | None = None    # (p, sink, d, t) — eff sink O&M
    p_pdt_varCost_process: Param | None = None # (p, d, t) — connection O&M

    # ─── Existing-entity fixed cost (constant; reported in objective) ─────
    p_ed_fixed_cost: Param | None = None         # (e, d)
    p_entity_all_existing: Param | None = None   # (e, d)

    # ─── Slack penalty scaling ────────────────────────────────────────────
    p_node_capacity_for_scaling: Param | None = None  # (n, d)

    # ─── Group-level slack (capacity_margin / inertia / non_sync) ─────────
    groupCapacityMargin: pl.DataFrame | None = None      # (g,)
    groupInertia: pl.DataFrame | None = None             # (g,)
    groupNonSync: pl.DataFrame | None = None             # (g,)
    group_node: pl.DataFrame | None = None               # (g, n)
    process_unit: pl.DataFrame | None = None             # (p,)  set of unit-typed processes (mod's process_unit set)
    process_sink_inertia: pl.DataFrame | None = None     # (p, sink)
    process_source_inertia: pl.DataFrame | None = None   # (p, source)
    process_sink_nonSync: pl.DataFrame | None = None     # (p, sink)
    process_group_inside_nonSync: pl.DataFrame | None = None  # (p, g)
    p_inv_group_cap: Param | None = None                 # (g, d)
    p_group_capacity_for_scaling: Param | None = None    # (g, d)
    pdGroup_capacity_margin: Param | None = None         # (g, d)
    pdGroup_penalty_capacity_margin: Param | None = None # (g, d)
    pdGroup_inertia_limit: Param | None = None           # (g, d)
    pdGroup_penalty_inertia: Param | None = None         # (g, d)
    pdGroup_non_synchronous_limit: Param | None = None   # (g, d)
    pdGroup_penalty_non_synchronous: Param | None = None # (g, d)
    p_process_sink_inertia_constant: Param | None = None    # (p, sink)
    p_process_source_inertia_constant: Param | None = None  # (p, source)
    p_positive_inflow: Param | None = None               # (n, d, t)
    p_negative_inflow: Param | None = None               # (n, d, t)
    pdtNodeInflow_per_step: Param | None = None          # (n, d, t)

    # ─── Reserves (timeseries / dynamic / n-1, plus per-process upper) ────
    reserve_upDown_group: pl.DataFrame | None = None                  # (r, ud, g) — gate
    reserve_upDown_group_method_timeseries: pl.DataFrame | None = None  # (r, ud, g, method)
    reserve_upDown_group_method_dynamic: pl.DataFrame | None = None     # (r, ud, g, method)
    reserve_upDown_group_method_n_1: pl.DataFrame | None = None         # (r, ud, g, method)
    prundt: pl.DataFrame | None = None                                  # (p, r, ud, n, d, t) — v_reserve domain
    process_reserve_upDown_node_active: pl.DataFrame | None = None      # (p, r, ud, n)
    process_reserve_upDown_node_increase_reserve_ratio: pl.DataFrame | None = None  # (p, r, ud, n)
    process_reserve_upDown_node_large_failure_ratio: pl.DataFrame | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_reliability: Param | None = None      # (p, r, ud, n)
    pdtReserve_upDown_group_reservation: Param | None = None            # (r, ud, g, d, t)
    p_reserve_upDown_group_penalty_reserve: Param | None = None         # (r, ud, g)
    p_process_reserve_upDown_node_max_share: Param | None = None        # (p, r, ud, n)
    p_process_reserve_upDown_node_large_failure_ratio_value: Param | None = None     # (p, r, ud, n)
    p_process_reserve_upDown_node_increase_reserve_ratio_value: Param | None = None  # (p, r, ud, n)

    # ─── Cumulative / group-invest / min-invest (read by _cumulative_invest) ─
    # Sets
    ed_invest_forbidden_no_investment: pl.DataFrame | None = None  # (e, d) — pin v_invest = 0
    ed_invest_cumulative: pl.DataFrame | None = None               # (e, d) — cumulative-cap rows
    group_entity: pl.DataFrame | None = None                       # (g, e)
    g_invest_total: pl.DataFrame | None = None                     # (g,)
    g_divest_total: pl.DataFrame | None = None                     # (g,)
    g_invest_cumulative: pl.DataFrame | None = None                # (g,)
    gd_invest_period: pl.DataFrame | None = None                   # (g, d)
    gd_divest_period: pl.DataFrame | None = None                   # (g, d)
    gdt_maxInstantFlow: pl.DataFrame | None = None                 # (g, d, t)
    gdt_minInstantFlow: pl.DataFrame | None = None                 # (g, d, t)
    group_process_node: pl.DataFrame | None = None                 # (g, p, n)
    # Parameters
    ed_invest_min_period: Param | None = None             # (e, d)
    ed_divest_min_period: Param | None = None             # (e, d)
    e_invest_min_total: Param | None = None               # (e,)
    e_divest_min_total: Param | None = None               # (e,)
    ed_cumulative_max_capacity: Param | None = None       # (e, d)
    ed_cumulative_min_capacity: Param | None = None       # (e, d)
    p_group_invest_max_period: Param | None = None        # (g, d)
    p_group_invest_min_period: Param | None = None        # (g, d)
    p_group_retire_max_period: Param | None = None        # (g, d)
    p_group_retire_min_period: Param | None = None        # (g, d)
    p_group_invest_max_total: Param | None = None         # (g,)
    p_group_invest_min_total: Param | None = None         # (g,)
    p_group_retire_max_total: Param | None = None         # (g,)
    p_group_retire_min_total: Param | None = None         # (g,)
    p_group_invest_max_cumulative: Param | None = None    # (g,)
    p_group_invest_min_cumulative: Param | None = None    # (g,)
    p_group_max_cumulative_flow: Param | None = None      # (g,)
    p_group_min_cumulative_flow: Param | None = None      # (g,)
    pd_max_cumulative_flow: Param | None = None           # (g, d)
    pd_min_cumulative_flow: Param | None = None           # (g, d)
    pdt_max_instant_flow: Param | None = None             # (g, d, t)
    pdt_min_instant_flow: Param | None = None             # (g, d, t)

    # ─── Delayed processes (read by _delay) ───────────────────────────
    process_delayed: pl.DataFrame | None = None                  # (p,)
    process_delayed__duration: pl.DataFrame | None = None        # (p, td)
    process_source_delayed: pl.DataFrame | None = None           # (p, source)
    process_source_undelayed: pl.DataFrame | None = None         # (p, source)
    process_source_sink_delayed: pl.DataFrame | None = None      # (p, source, sink)
    process_source_sink_undelayed: pl.DataFrame | None = None    # (p, source, sink)
    dtt__delay_duration: pl.DataFrame | None = None              # (d, t_source, t_sink, td)
    p_process_delay_weight: Param | None = None                  # (p, td)

    # ─── DC power flow (read by _dc_power_flow) ──────────────────────────
    # Populated only when ``input/node_dc_power_flow.csv`` and
    # ``connection_dc_power_flow.csv`` carry rows.  See
    # :mod:`flextool._dc_power_flow` for the constraint emission.
    node_dc_power_flow: pl.DataFrame | None = None               # (n,)
    connection_dc_power_flow: pl.DataFrame | None = None         # (p,)
    node_reference_angle: pl.DataFrame | None = None             # (n,)
    p_connection_susceptance: Param | None = None                # (p,)
    # Forward-direction (p, source, sink) mirror of process_source_toSink,
    # used to dedupe DC PF arcs.  process_source_sink doubles up 2-way
    # connections; the .mod's dc_flow_eq indexes on process_source_toSink
    # (one direction per arc).
    process_source_toSink_dc: pl.DataFrame | None = None         # (p, source, sink)

    # ─── Commodity price ladder (read by _commodity_ladder) ─────────────
    # Populated only when at least one commodity has
    # ``price_method = price_ladder_*``.  See
    # :mod:`flextool._commodity_ladder` for the constraint emission.
    commodity_with_ladder: pl.DataFrame | None = None            # (c,)
    commodity_with_ladder_annual: pl.DataFrame | None = None     # (c,)
    commodity_with_ladder_cumulative: pl.DataFrame | None = None # (c,)
    cnd_ladder: pl.DataFrame | None = None                       # (c, n, d)
    cndi_ladder: pl.DataFrame | None = None                      # (c, n, d, i)
    cndi_ladder_ann: pl.DataFrame | None = None                  # (c, n, d, i)
    cndi_ladder_cum: pl.DataFrame | None = None                  # (c, n, d, i)
    ci_ladder_cumulative: pl.DataFrame | None = None             # (c, i)
    commodity__tier_ann: pl.DataFrame | None = None              # (c, i)
    commodity__tier_cum: pl.DataFrame | None = None              # (c, i)
    p_ladder_ann_price: Param | None = None                      # (c, i, d)
    p_ladder_ann_quantity: Param | None = None                   # (c, i, d)
    p_ladder_cum_price: Param | None = None                      # (c, i)
    p_ladder_cum_quantity: Param | None = None                   # (c, i)
    p_commodity_unitsize: Param | None = None                    # (c,)
    p_f_d_k: Param | None = None                                 # (d,)
    p_ladder_cum_realized_mwh: Param | None = None               # (c, i, d)

    # ─── Stochastic / multi-branch operational data (A6) ─────────────────
    # All fields populated only when the active solve actually runs a
    # multi-branch stochastic dispatch (signalled by ``solve_data/
    # pdt_branch_weight.csv`` containing rows where the cohort
    # (anchor period d) has multiple sibling periods b).  When stochastics
    # is inactive every (d, t) carries weight 1.0 and these fields stay
    # ``None`` (the model layer falls back to the deterministic path).
    #
    # ``period_branch_full`` is the unfiltered ``period__branch.csv``
    # (anchor d → sibling b).  Distinct from the existing
    # ``period_branch`` field which is the rolling-handoff helper
    # (renamed columns).  Both share the same source CSV; we keep them
    # separate to avoid disturbing the rolling-handoff consumer.
    pdt_branch_weight: Param | None = None        # (d, t) — operational weight (defaults 1.0)
    pd_branch_weight: Param | None = None         # (d,) — period-level weight (defaults 1.0)
    period_branch_full: pl.DataFrame | None = None  # (d, b) — full anchor→sibling map
    dt_non_anticipativity: pl.DataFrame | None = None  # (d, t) — realised dispatch + fix-storage timesteps
    groupStochastic: pl.DataFrame | None = None   # (g,) — groups enabling storage non-anticipativity
    period_in_use_set: pl.DataFrame | None = None  # (d,) — periods active this solve (filters branches)

    # ─── Gap F final — handoff-path auxiliaries ───────────────────────────
    # Per-solve in-memory carriers for fields that ``build_handoff_from_solution``
    # would otherwise re-read from ``solve_data/`` to capture the post-solve
    # handoff.  Populated by :func:`load_flextool` from the corresponding
    # CSVs when present; ``None`` falls through to the disk-read fallback in
    # the handoff extractor (preserves test paths that construct FlexData by
    # hand).
    realized_dispatch: pl.DataFrame | None = None         # (period, step)
    period__time_last: pl.DataFrame | None = None         # (period, step)
    node__storage_nested_fix_method: pl.DataFrame | None = None  # (node, method)

    # ─── HiGHS solver options (read from input/solve_mode.csv) ───────────
    # Maps HiGHS option name → value (str / int / float / bool).  flextool
    # writes ``highs_method``, ``highs_parallel``, ``highs_presolve`` rows
    # keyed on ``solve``; load_flextool picks the row for the active solve
    # (solve_data/solve_current.csv) and renames keys to HiGHS canonical
    # option names (``solver``, ``parallel``, ``presolve``).  Applied in
    # ``Problem.solve()`` via ``Highs.setOptionValue``.  ``None`` means no
    # CSV / no rows for the active solve → HiGHS defaults.
    solver_options: dict | None = None

    def dump_csvs(self,
                   workdir: "Path | str",
                   *,
                   copy_meta_from: "Path | str | None" = None,
                   include_heavy: bool | None = None,
                   ) -> "Path":
        """Materialise this FlexData to flextool's CSV layout under ``workdir``.

        See :mod:`flextool._dump_csvs` for the full mapping.  Round-trip
        contract: ``load_flextool(dump_csvs(out))`` reproduces every
        populated FlexData field frame-for-frame (modulo row order).

        ``copy_meta_from`` is the original workdir whose per-solve
        metadata (``solve_current.csv``, timeline reference files,
        period-first markers, …) we copy through verbatim — these are
        runner state, not FlexData fields, but the CSV reader needs
        them.  When the round-trip caller has access to the original
        workdir, pass it here.

        ``include_heavy`` (default ``None``) controls whether the seven
        gigabyte-scale CSVs (``p_flow_max.csv`` and friends) are
        written.  ``None`` honours the ``FLEXTOOL_DUMP_CSVS`` env var
        (off by default); pass ``True`` to force-write them (e.g. for
        the round-trip regression test).
        """
        # Local import — avoids a circular import at module-load.
        from flextool.engine_polars._dump_csvs import dump_csvs as _impl
        return _impl(self, workdir, copy_meta_from=copy_meta_from,
                     include_heavy=include_heavy)

    def __setattr__(self, name: str, value):
        # Diagnostic gate: only armed by ``_cascade_gate()`` (used by
        # the cascade parity test).  Production code paths never see
        # this branch — the module-level flag stays False end-to-end.
        if _CASCADE_GATE_ACTIVE:
            current = self.__dict__.get(name)
            if current is not None and value is not current:
                return
        object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Time + node helpers (always loaded)

def _load_time(sd: Path,
                *,
                provider: "object | None" = None):
    # ``steps_in_use.csv`` is the canonical source for both the dt set
    # and step_duration (.mod reads them together at flextool.mod:781).
    # ``dt.csv`` and ``p_step_duration.csv`` are .mod printf debug-exports
    # that only cover dispatch periods — using them silently drops the
    # invest-period (d, t) rows in multi-period scenarios.
    siu = _provider_read(provider, "solve_data/steps_in_use",
                          sd / "steps_in_use.csv").pipe(
        rename_to_axis, {"period": "d", "step": "t", "step_duration": "value"})
    # Phase E-d — the in-memory accumulator returns Utf8-typed frames
    # (writers funnel through ``_to_utf8_frame``); cast ``value`` to
    # Float64 so downstream Param arithmetic doesn't hit a
    # ``arithmetic on string and numeric not allowed`` error.
    if "value" in siu.columns and siu.schema["value"] != pl.Float64:
        siu = siu.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
    dt = siu.select("d", "t")
    step_dur = Param(("d","t"), siu.select("d", "t", "value"))
    # rp_cost_weight: canonical ``rp_cost_weight.csv``
    # (.mod's ``p_rp_cost_weight.csv`` is a printf debug-export).
    # Defaults to 1.0 per (d, t) when the canonical file is empty
    # (matches .mod's ``param p_rp_cost_weight ... default 1`` clause).
    rp_default = dt.with_columns(value=pl.lit(1.0))
    rp_cw_path = sd / "rp_cost_weight.csv"
    if _provider_has(provider, "solve_data/rp_cost_weight", rp_cw_path):
        rp_df = _provider_read(provider, "solve_data/rp_cost_weight", rp_cw_path)
        if rp_df.height > 0:
            # canonical column is named ``weight`` per .mod's ``table data IN``.
            value_col = "weight" if "weight" in rp_df.columns else "value"
            rp_df = (rp_df.pipe(rename_to_axis, {"period": "d", "time": "t",
                                    value_col: "value"})
                          .with_columns(value=pl.col("value")
                                                 .cast(pl.Float64, strict=False))
                          .select("d", "t", "value"))
            # Left-join the default with explicit overrides.
            rp_default = (rp_default.join(rp_df, on=["d","t"], how="left",
                                            suffix="__r")
                                     .with_columns(value=pl.coalesce(
                                          pl.col("value__r"), pl.col("value")))
                                     .select("d","t","value"))
    rp_cw = Param(("d","t"), rp_default)
    infl = Param(("d",),
        _read_long(sd / "p_inflation_factor_operations_yearly.csv",
                    rename={"period": "d"}, provider=provider))
    # complete_period_share_of_year: canonical
    # ``complete_period_share_of_year_calc.csv``.
    psh = Param(("d",),
        _read_long(sd / "complete_period_share_of_year_calc.csv",
                    rename={"period": "d"}, provider=provider))
    return dt, step_dur, rp_cw, infl, psh


def _load_node(sd: Path, dt: pl.DataFrame,
                *,
                provider: "object | None" = None):
    nb = _provider_read(provider, "solve_data/nodeBalance",
                         sd / "nodeBalance.csv").pipe(rename_to_axis, {"node": "n"})
    # pdtNodeInflow.csv is canonical (.mod reads it via `table data IN`).
    # TODO(Δ.18+): retire pdtNodeInflow.csv read when ``apply_derived_a``
    # extends ``p_inflow_from_source`` to cover ``inflow_method ∈ {scale_to_*}``
    # and stochastic 3d_map shapes.
    inflow_long = _read_wide_per_entity(sd / "pdtNodeInflow.csv",
                                          rename={"entity":"n"},
                                          provider=provider)

    # Δ.18 — CSV-fallback seed for ``p_penalty_up`` / ``p_penalty_down``
    # from the wide-by-param ``pdtNode.csv`` slice.  Override chain
    # (``apply_derived_a`` via ``p_penalty_up_from_source`` /
    # ``p_penalty_down_from_source``) overlays these when active; for
    # synthetic per-sub-solve fixtures the snapshot CSV is the only source.
    empty_n_d_t = pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8,
                                         "t": pl.Utf8, "value": pl.Float64})
    pen_up_df = empty_n_d_t
    pen_dn_df = empty_n_d_t
    pdtnode_path = sd / "pdtNode.csv"
    if _provider_has(provider, "solve_data/pdtNode", pdtnode_path):
        df_pn = _provider_read(provider, "solve_data/pdtNode", pdtnode_path)
        if df_pn.height > 0 and {"node", "param", "period", "time", "value"}.issubset(df_pn.columns):
            df_pn = (df_pn.pipe(rename_to_axis, {"node": "n", "period": "d", "time": "t"})
                          .with_columns(value=pl.col("value")
                                                  .cast(pl.Float64, strict=False)
                                                  .fill_null(0.0)))
            up = df_pn.filter(pl.col("param") == "penalty_up").select("n", "d", "t", "value")
            if up.height > 0:
                pen_up_df = up
            dn = df_pn.filter(pl.col("param") == "penalty_down").select("n", "d", "t", "value")
            if dn.height > 0:
                pen_dn_df = dn

    # Phase E.3: ``nodeBalance_dt`` no longer materialised; consumers
    # call ``_pdt_join.compute_nodeBalance_dt`` on demand.
    return (nb, None,
            Param(("n","d","t"), inflow_long.select("n","d","t","value")),
            Param(("n","d","t"), pen_up_df),
            Param(("n","d","t"), pen_dn_df))


# ---------------------------------------------------------------------------
# Process-topology helpers (skipped if no processes)

def _load_process_topology(inp: Path, sd: Path, dt: pl.DataFrame,
                            block_layout: "BlockLayout | None" = None,
                            *, source: "InputSource | None" = None,
                            provider: "object | None" = None):
    # Δ.17b Gap B: ``process_source_sink_canonical`` produces flextool's
    # preprocessing-side collapsed shape directly from Spine (DIRECT methods
    # cross-joined; INDIRECT methods kept as 2-arc form; 2way reverse arcs
    # added; noConversion fallbacks handled).
    empty_return = {k: None for k in ("pss","pss_eff","pss_noEff","pss_dt",
                                       "flow_to_n","flow_from_n",
                                       "flow_from_commodity_eff",
                                       "flow_from_commodity_noEff",
                                       "unitsize","flow_upper","slope","commodity_price",
                                       "pss_source_canonical","pss_sink_canonical")}

    # Δ.18 — CSV-fallback for the pss family.  When ``source`` is None
    # (dump_csvs roundtrip workdirs without a ``tests.sqlite``), read
    # the canonical preprocessed CSVs (``process_source_sink.csv`` /
    # ``_eff.csv`` / ``_noEff.csv``) directly.  These mirror the
    # source-driven helper's output; the round-trip test depends on it.
    if source is None:
        pss_path     = sd / "process_source_sink.csv"
        pss_eff_path = sd / "process_source_sink_eff.csv"
        pss_noe_path = sd / "process_source_sink_noEff.csv"
        if not _provider_has(provider, "solve_data/process_source_sink", pss_path):
            return empty_return
        from ._axis_enums import schema_dtype
        empty_pss = pl.DataFrame(
            schema={
                "p": schema_dtype(None, "p"),
                "source": schema_dtype(None, "source"),
                "sink": schema_dtype(None, "sink"),
            })

        def _read_pss(p: Path) -> pl.DataFrame:
            df = _provider_read(provider, f"solve_data/{p.stem}", p)
            if df.height == 0 or "process" not in df.columns:
                return empty_pss
            return (df.pipe(rename_to_axis, {"process": "p"})
                      .with_columns(
                          cast_dim(pl.col("source"), None, "source"),
                          cast_dim(pl.col("sink"), None, "sink"),
                      )
                      .select("p", "source", "sink")
                      .unique()
                      .sort("p", "source", "sink"))
        pss = _read_pss(pss_path)
        if pss.height == 0:
            return empty_return
        pss_eff = (_read_pss(pss_eff_path)
                   if _provider_has(provider,
                                    "solve_data/process_source_sink_eff",
                                    pss_eff_path)
                   else empty_pss)
        pss_noEff = (_read_pss(pss_noe_path)
                     if _provider_has(provider,
                                      "solve_data/process_source_sink_noEff",
                                      pss_noe_path)
                     else empty_pss)

        # Canonical source/sink per process from the preprocessed CSV sets.
        # process_source.csv and process_sink.csv (written by flextool's
        # preprocessing) contain exactly the canonical node per process:
        # unit__inputNode for source, unit__outputNode for sink, and the
        # original connection__node__node direction for connections.
        def _read_canonical_set(p: Path, side: str) -> pl.DataFrame | None:
            name = f"solve_data/{p.stem}"
            if not _provider_has(provider, name, p):
                return None
            df = _provider_read(provider, name, p)
            if df.height == 0 or "process" not in df.columns or side not in df.columns:
                return None
            return (df.pipe(rename_to_axis, {"process": "p"})
                      .with_columns(cast_dim(pl.col(side), None, side))
                      .select("p", side)
                      .unique()
                      .sort("p", side))
        pss_source_canonical = _read_canonical_set(sd / "process_source.csv", "source")
        pss_sink_canonical   = _read_canonical_set(sd / "process_sink.csv",   "sink")
    else:
        from ._projection_params import process_source_sink_canonical, _try_entities
        canonical = process_source_sink_canonical(source)
        if canonical.height == 0:
            return empty_return
        pss = (canonical.select("p", "source", "sink")
                          .unique()
                          .sort("p", "source", "sink"))
        pss_eff = (canonical
            .filter(pl.col("method") == "eff")
            .select("p", "source", "sink").unique()
            .sort("p", "source", "sink"))
        pss_noEff = (canonical
            .filter(pl.col("method") == "noEff")
            .select("p", "source", "sink").unique()
            .sort("p", "source", "sink"))

        # Canonical source/sink per process from the original entity tables:
        # unit__inputNode → (p, source), unit__outputNode → (p, sink),
        # connection__node__node → (p, source=node_1) and (p, sink=node_2).
        src_parts: list[pl.DataFrame] = []
        snk_parts: list[pl.DataFrame] = []
        _uin = _try_entities(source, "unit__inputNode")
        if _uin is not None and _uin.height > 0:
            src_parts.append(
                _uin.pipe(rename_to_axis, {"unit": "p", "node": "source"})
                    .select("p", "source"))
        _uout = _try_entities(source, "unit__outputNode")
        if _uout is not None and _uout.height > 0:
            snk_parts.append(
                _uout.pipe(rename_to_axis, {"unit": "p", "node": "sink"})
                     .select("p", "sink"))
        _cnn = _try_entities(source, "connection__node__node")
        if _cnn is not None and _cnn.height > 0:
            src_parts.append(
                _cnn.pipe(rename_to_axis,
                          {"connection": "p", "node_1": "source"})
                    .select("p", "source"))
            snk_parts.append(
                _cnn.pipe(rename_to_axis,
                          {"connection": "p", "node_2": "sink"})
                    .select("p", "sink"))
        pss_source_canonical = (
            pl.concat(src_parts).unique().sort("p", "source") if src_parts else None
        )
        pss_sink_canonical = (
            pl.concat(snk_parts).unique().sort("p", "sink") if snk_parts else None
        )

    # ``n`` derives from sink/source which are entity-union (``e``)
    # axis values — they may carry node OR process tokens (indirect
    # units' arcs).  Cast against the ``e`` axis (union) rather than
    # the narrower ``n`` (node-only) axis so process-vocab tokens are
    # preserved instead of silently nulled.  Downstream eb_local.n is
    # also up-cast to ``e`` below so the join composes natively in
    # Enum.  Mirrors the source-driven cascade pattern in
    # ``_derived_block.flow_to_n_block_filtered`` /
    # ``filter_flow_n_by_block``.
    _enums = get_global_axis_enums()
    flow_to_n   = pss.with_columns(
        n=cast_dim(pl.col("sink"), _enums, "e"))
    flow_from_n = pss.with_columns(
        n=cast_dim(pl.col("source"), _enums, "e"))

    # ─── Filter arcs by block compatibility (mod's process_side_block) ──
    # In the .mod, an arc contributes to a node's nodeBalance_eq iff the
    # overlap set has a row connecting (b_n, t) ↔ (b_f, t_f) where
    # (p, side, b_f) ∈ process_side_block.  In particular, a daily-side
    # arc (e.g. electrolyser_A's sink on daily_group) does NOT contribute
    # to a fine-grid (hourly/default) node's hourly nodeBalance because
    # the overlap (hourly_group, t, daily_group, t_f) doesn't exist.
    # We replicate this restriction by filtering ``flow_to_n``/
    # ``flow_from_n`` to drop (p, source, sink) rows whose relevant
    # side-block doesn't connect via overlap to the node's own block.
    #
    # Δ.2: block frames consumed via in-memory ``BlockLayout`` when one
    # is provided; fall back to the legacy on-disk reads when not.
    if (block_layout is not None
            and block_layout.process_side_block_frame.height > 0
            and block_layout.entity_block_frame.height > 0
            and block_layout.overlap_set_frame.height > 0):
        psb_local = block_layout.process_side_block_frame.pipe(
            rename_to_axis, {"process": "p", "block": "b_f"})
        # Match the flow_to_n / flow_from_n e-Enum ``n`` dtype: rename
        # entity → e (cast against the union axis, no data loss), then
        # alias to "n" while keeping the e-Enum dtype.  Downstream join
        # on "n" then composes natively in Enum because both sides are
        # e-typed (n ⊂ e — node-only tokens still match).
        eb_local = (block_layout.entity_block_frame.pipe(
                rename_to_axis, {"entity": "e", "block": "bk"})
            .with_columns(n=cast_dim(pl.col("e"), None, "e"))
            .drop("e"))
        block_compat = block_layout.block_compat()
        if (psb_local.height > 0 and eb_local.height > 0
                and block_compat.height > 0):
            psb_sink = psb_local.filter(pl.col("side") == "sink").select("p", "b_f")
            psb_source = psb_local.filter(pl.col("side") == "source").select("p", "b_f")
            # flow_to_n is keyed by sink-as-n; the relevant side is 'sink'.
            ftn_with_blocks = (flow_to_n
                .join(psb_sink, on="p", how="left")
                .join(eb_local, on="n", how="left"))
            # If b_f or bk is null, treat as 'default' (compatibility default).
            ftn_with_blocks = ftn_with_blocks.with_columns(
                b_f=pl.col("b_f").fill_null(lit_axis("default", "block")),
                bk=pl.col("bk").fill_null(lit_axis("default", "block")),
            )
            # Inner-join with block_compat to keep compatible rows.
            ftn_filtered = (ftn_with_blocks
                .join(block_compat, on=["bk", "b_f"], how="inner")
                .select("p", "source", "sink", "n").unique())
            # Replace flow_to_n if filter actually drops rows.
            if ftn_filtered.height > 0 and ftn_filtered.height < flow_to_n.height:
                flow_to_n = ftn_filtered
            # flow_from_n: source-as-n, side='source'.
            ffn_with_blocks = (flow_from_n
                .join(psb_source, on="p", how="left")
                .join(eb_local, on="n", how="left"))
            ffn_with_blocks = ffn_with_blocks.with_columns(
                b_f=pl.col("b_f").fill_null(lit_axis("default", "block")),
                bk=pl.col("bk").fill_null(lit_axis("default", "block")),
            )
            ffn_filtered = (ffn_with_blocks
                .join(block_compat, on=["bk", "b_f"], how="inner")
                .select("p", "source", "sink", "n").unique())
            if ffn_filtered.height > 0 and ffn_filtered.height < flow_from_n.height:
                flow_from_n = ffn_filtered

    cn = _provider_read(provider, "input/commodity__node",
                          inp / "commodity__node.csv")
    # CSV read produces Utf8 columns; project to the canonical axis
    # column names so the joins against pss_eff/pss_noEff line up on
    # Enum-typed source/sink columns (the union ``e`` axis).  The
    # ``node → source`` / ``node → sink`` renames cast against ``e``
    # via rename_to_axis.
    cn_as_source = cn.pipe(rename_to_axis,
                            {"node": "source", "commodity": "c"})
    cn_as_sink = cn.pipe(rename_to_axis,
                          {"node": "sink", "commodity": "c"})
    flow_from_commodity_eff = (pss_eff
        .join(cn_as_source, on="source", how="inner")
        .select("p","source","sink","c"))
    flow_from_commodity_noEff = (pss_noEff
        .join(cn_as_source, on="source", how="inner")
        .select("p","source","sink","c"))
    # §2.4 commodity sell: sink-side flow into a commodity-priced node.
    # No slope correction — straight v_flow * unitsize * commodity_price.
    flow_to_commodity = (pss
        .join(cn_as_sink, on="sink", how="inner")
        .select("p","source","sink","c"))

    # ``p_unitsize`` is overwritten by ``apply_derived_b.p_unitsize_from_source``
    # but the returned ``unitsize`` Param is also consumed inline by
    # ``_load_profiles`` (None → blank profile dict) and ``_load_storage``
    # (used in cap_pd / state_unitsize cascades).  Keep the seed.
    # TODO(Δ.12c+): when ``_load_profiles`` / ``_load_storage`` consume the
    # source-driven p_unitsize via flex_data, drop this seed read.
    unitsize_long = _read_unitsize(_provider_pick(
        provider,
        ("solve_data/p_entity_unitsize", sd / "p_entity_unitsize.csv"),
        ("input/p_entity_unitsize", inp / "p_entity_unitsize.csv"),
    ) or (inp / "p_entity_unitsize.csv"), provider=provider)
    unitsize_p = (unitsize_long.pipe(rename_to_axis, {"e": "p"})
                       .filter(pl.col("p").is_in(pss["p"].unique())))

    # ``p_slope`` is produced by ``apply_derived_b.p_slope_from_source``
    # but the 7 mismatch fixtures (see Δ.12-drop close stanza in
    # progress.md) skip auto-resolution and rely on the seed.  Keep
    # CSV read.
    # TODO(Δ.12c+): retire when ``_find_scenario`` covers underscore-
    # variant fixtures or all fixtures explicitly pass db_reader=.
    slope_long = _read_wide_per_entity(sd / "pdtProcess_slope.csv",
                                         rename={"entity":"p"},
                                         provider=provider)
    # Δ.17c Gap C: ``p_commodity_price`` produced authoritatively by
    # ``apply_direct_params`` via ``p_commodity_price_from_source``
    # (uses the ``_param_shapes`` resolver — scalar / 1d_map[period] /
    # 1d_map[time] cascade with explicit allow-list).  Local
    # pdtCommodity.csv slice dropped.
    cp_long = None

    # flow_upper is the canonical ``p_flow_max.csv`` long-format file
    # the .mod reads via ``table data IN`` (`[process, source, sink,
    # period, time], p_flow_max~value`).
    # TODO(Δ.12c+): no override-chain helper covers ``p_flow_upper`` yet —
    # the preprocessed p_flow_max.csv bakes in invest_max_cum etc. that
    # the source-driven path would have to recompute.
    flow_upper_psskdt = _read_p_flow_max(sd / "p_flow_max.csv", provider=provider)

    return dict(
        pss = pss,
        pss_eff = pss_eff,
        pss_noEff = pss_noEff,
        # Phase E.3: ``pss_dt`` is no longer materialised here; consumers
        # call ``_pdt_join.compute_pss_dt`` on demand.
        pss_dt = None,
        flow_to_n = flow_to_n,
        flow_from_n = flow_from_n,
        flow_from_commodity_eff = flow_from_commodity_eff,
        flow_from_commodity_noEff = flow_from_commodity_noEff,
        flow_to_commodity = flow_to_commodity,
        pss_source_canonical = pss_source_canonical,
        pss_sink_canonical   = pss_sink_canonical,
        unitsize = Param(("p",), unitsize_p.select("p","value")),
        flow_upper = Param(("p","source","sink","d","t"), flow_upper_psskdt),
        slope = Param(("p","d","t"), slope_long.select("p","d","t","value")),
        # Δ.17c Gap C: ``p_commodity_price`` populated authoritatively by
        # ``apply_direct_params``.  The seed used to materialise a fully-
        # zero-filled (c, d, t) Param for fixtures with no explicit price
        # (preprocessed pdtCommodity.csv emits 0.0 rows for every cell);
        # the model.py ``PROCESSES`` invariant requires the field non-None.
        # We satisfy it with an empty Param as a placeholder; ``Sum(Where(
        # ..., ...) * d.p_commodity_price * ...)`` joins yield no rows
        # which is the same behaviour as zero-filled rows in the LP.
        commodity_price = (Param(("c","d","t"), cp_long)
                            if cp_long is not None
                            else Param(("c","d","t"),
                                        pl.DataFrame(schema={
                                            "c": pl.Utf8, "d": pl.Utf8,
                                            "t": pl.Utf8,
                                            "value": pl.Float64}))),
    )


# ---------------------------------------------------------------------------
# Optional features (CO2 price, CO2 cap, indirect, user-defined, profiles)

def _load_co2_price(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                     pss_noEff: pl.DataFrame | None = None,
                     *,
                     provider: "object | None" = None):
    if pss_eff is None: return (None, None, None, None)
    files = ["group_co2_price.csv", "commodity_node_co2.csv", "pdtGroup.csv"]
    if not all(_provider_has(provider, f"solve_data/{Path(f).stem}", sd / f)
                for f in files):
        return (None, None, None, None)
    g_price = _provider_read(provider, "solve_data/group_co2_price",
                              sd / "group_co2_price.csv").pipe(rename_to_axis, {"group": "g"})
    if g_price.height == 0: return (None, None, None, None)
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2",
                              sd / "commodity_node_co2.csv").pipe(rename_to_axis, {"commodity":"c","node":"n"})
    g_node = _provider_read(provider, "input/group__node",
                              inp / "group__node.csv").pipe(rename_to_axis, {"group":"g","node":"n"})
    gcn = (g_price.join(g_node, on="g", how="inner")
                  .join(cn_co2, on="n", how="inner")
                  .select("g","c","n"))
    # Cross-axis join (Pattern 2): pss_eff.source is the entity-union
    # (``e``) axis; gcn.n is the node (``n``) axis.  Up-cast the
    # narrower side to ``e`` so the join is a native Enum match under
    # Phase 4 activation.
    gcn_e = gcn.with_columns(cast_dim(pl.col("n"), None, "e"))
    flow_from_co2_priced = (pss_eff
        .join(gcn_e, left_on="source", right_on="n", how="inner")
        .select("p","source","sink","c","g"))
    # noEff variant: source flow into a CO2-priced commodity node where the
    # process is on the noEff side.  Rare but used for "cheap simplified"
    # gas/coal models that don't model efficiency curves.
    flow_from_co2_priced_noEff = None
    if pss_noEff is not None:
        flow_from_co2_priced_noEff = (pss_noEff
            .join(gcn_e, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if flow_from_co2_priced_noEff.height == 0:
            flow_from_co2_priced_noEff = None
    if flow_from_co2_priced.height == 0 and flow_from_co2_priced_noEff is None:
        return (None, None, None, None)
    # ``p_co2_content`` is produced by ``apply_direct_params`` BUT some
    # callers exercise the pure-CSV path (e.g. ``run_chain``-style
    # tempdir without tests.sqlite — see
    # test_orchestration_parity::test_build_handoff_from_solution_covers_eight_carriers).
    # Keep the seed.
    # TODO(Δ.12c+): retire when all callers either pass an explicit
    # db_reader= or a workdir whose tests.sqlite + scenario auto-resolve.
    p_comm = _provider_read(provider, "input/p_commodity",
                              inp / "p_commodity.csv")
    co2_content = Param(("c",),
        p_comm.filter(pl.col("commodityParam")=="co2_content")
              .pipe(rename_to_axis, {"commodity":"c","p_commodity":"value"})
              .select("c","value"))
    # Δ.17c Gap C: ``p_co2_price`` produced authoritatively by
    # ``apply_direct_params`` via ``p_co2_price_from_source`` (uses the
    # ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade with explicit allow-list).
    # Local CSV slice dropped.
    co2_price = None
    return (flow_from_co2_priced, flow_from_co2_priced_noEff,
            co2_content, co2_price)


def _load_co2_cap(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                   dt: pl.DataFrame,
                   pss_noEff: pl.DataFrame | None = None,
                   *,
                   provider: "object | None" = None):
    if pss_eff is None and pss_noEff is None:
        return (None, None, None, None, None)
    p = sd / "group_co2_max_period.csv"
    if not _provider_has(provider, "solve_data/group_co2_max_period", p):
        return (None, None, None, None, None)
    g_max = _provider_read(provider, "solve_data/group_co2_max_period", p).pipe(rename_to_axis, {"group":"g"})
    if g_max.height == 0: return (None, None, None, None, None)
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2",
                              sd / "commodity_node_co2.csv").pipe(rename_to_axis, {"commodity":"c","node":"n"})
    g_node = _provider_read(provider, "input/group__node",
                              inp / "group__node.csv").pipe(rename_to_axis, {"group":"g","node":"n"})
    gcn = (g_max.join(g_node, on="g", how="inner")
                .join(cn_co2, on="n", how="inner")
                .select("g","c","n"))
    if gcn.height == 0: return (None, None, None, None, None)
    # Cross-axis join setup (Pattern 2): up-cast gcn.n (n axis) to ``e``
    # so the join against pss.source (e axis) is a native Enum match.
    gcn_e = gcn.with_columns(cast_dim(pl.col("n"), None, "e"))
    # The .mod's co2_max_period sums emissions over (p, source, sink)
    # for processes whose source is a CO2-priced node — but with
    # different formulae for eff vs noEff.  eff is multiplied by
    # ``pdtProcess_slope[p, d, t]`` (the conversion-efficiency factor);
    # noEff is just ``v_flow * unitsize`` with no slope.  We must
    # therefore split the set into two and handle each leg separately
    # — using a single combined set with the eff-style slope multiplier
    # would over-count noEff processes' emissions (e.g. coal_chp's
    # slope=1.111 inflates its CO2 by ~11%, breaking co2_max_period
    # parity on multi-period fixtures with non-trivial CHP shares).
    flow_from_co2_capped_eff = None
    flow_from_co2_capped_noEff = None
    if pss_eff is not None and pss_eff.height > 0:
        eff = (pss_eff.select("p","source","sink")
            .join(gcn_e, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if eff.height > 0:
            flow_from_co2_capped_eff = eff
    if pss_noEff is not None and pss_noEff.height > 0:
        noeff = (pss_noEff.select("p","source","sink")
            .join(gcn_e, left_on="source", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if noeff.height > 0:
            flow_from_co2_capped_noEff = noeff
    if flow_from_co2_capped_eff is None and flow_from_co2_capped_noEff is None:
        return (None, None, None, None, None)
    # Δ.17c Gap C: ``p_co2_max_period`` produced authoritatively by
    # ``apply_direct_params`` via ``p_co2_max_period_from_source`` (uses
    # the ``_param_shapes`` resolver — scalar / 1d_map[period] cascade
    # with explicit allow-list).  Local pd_group.csv slice dropped.
    co2_max_period = None
    period = dt.select("d").unique()
    return (g_max, flow_from_co2_capped_eff, flow_from_co2_capped_noEff,
            co2_max_period, g_max.join(period, how="cross"))


def _load_co2_cap_total(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                          pss_noEff: pl.DataFrame | None = None,
                          *,
                          provider: "object | None" = None):
    """Sibling of :func:`_load_co2_cap` for the multi-period total cap.

    Mirrors the period-cap topology (``group_co2_max_period.csv`` → eff /
    noEff (p, source, sink, c, g) frames) but the gate set comes from
    ``solve_data/group_co2_max_total.csv`` (groups whose ``group__co2_method``
    is ``total`` / ``price_total`` / ``period_total`` — already projected
    by :func:`_emit_leaf_sets.write_co2_method_sets`).  The cap value
    is read from the canonical ``solve_data/pdGroup.csv`` slice
    (``param == 'co2_max_total'``); flextool preprocessing broadcasts the
    Spine scalar across periods, so we collapse to one row per group by
    taking the maximum (all rows carry the same value when authored as
    a scalar).

    Returns a 4-tuple (g_max_total, flow_eff, flow_noEff, p_co2_max_total).
    """
    if pss_eff is None and pss_noEff is None:
        return (None, None, None, None)
    p = sd / "group_co2_max_total.csv"
    if not _provider_has(provider, "solve_data/group_co2_max_total", p):
        return (None, None, None, None)
    g_max = _provider_read(provider, "solve_data/group_co2_max_total", p).pipe(rename_to_axis, {"group": "g"})
    if g_max.height == 0:
        return (None, None, None, None)
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2",
                              sd / "commodity_node_co2.csv").pipe(
        rename_to_axis, {"commodity": "c", "node": "n"})
    g_node = _provider_read(provider, "input/group__node",
                              inp / "group__node.csv").pipe(
        rename_to_axis, {"group": "g", "node": "n"})
    gcn = (g_max.join(g_node, on="g", how="inner")
                .join(cn_co2, on="n", how="inner")
                .select("g", "c", "n"))
    if gcn.height == 0:
        return (None, None, None, None)
    flow_eff = None
    flow_noEff = None
    if pss_eff is not None and pss_eff.height > 0:
        eff = (pss_eff.select("p", "source", "sink")
            .join(gcn, left_on="source", right_on="n", how="inner")
            .select("p", "source", "sink", "c", "g"))
        if eff.height > 0:
            flow_eff = eff
    if pss_noEff is not None and pss_noEff.height > 0:
        noeff = (pss_noEff.select("p", "source", "sink")
            .join(gcn, left_on="source", right_on="n", how="inner")
            .select("p", "source", "sink", "c", "g"))
        if noeff.height > 0:
            flow_noEff = noeff
    if flow_eff is None and flow_noEff is None:
        return (None, None, None, None)
    # Read the cap value from pdGroup.csv (param='co2_max_total').  Legacy
    # preprocessing writes one row per (group, period) by broadcasting the
    # Spine scalar; we collapse via max() per group (all per-period rows
    # share the same value when authored as a scalar).  Filter zero rows
    # so the constraint set is naturally pruned to the active caps.
    p_cap = None
    pdg = sd / "pdGroup.csv"
    if _provider_has(provider, "solve_data/pdGroup", pdg):
        df = _provider_read(provider, "solve_data/pdGroup", pdg)
        if df.height > 0 and {"group", "param", "value"}.issubset(df.columns):
            sliced = (df.filter(pl.col("param") == "co2_max_total")
                        .pipe(rename_to_axis, {"group": "g"})
                        .with_columns(pl.col("value").cast(pl.Float64,
                                                            strict=False))
                        .filter(pl.col("value").is_not_null())
                        .filter(pl.col("value") != 0.0)
                        .group_by("g", maintain_order=True)
                        .agg(pl.col("value").max())
                        .join(g_max, on="g", how="inner"))
            if sliced.height > 0:
                p_cap = Param(("g",), sliced.select("g", "value"))
    if p_cap is None:
        # Without a non-zero cap there's nothing to bind — bail out so
        # the consumer leaves ``has_co2_cap_total`` false.
        return (None, None, None, None)
    # Restrict the gate set to groups that actually carry a cap value.
    cap_groups = p_cap.frame.select("g")
    g_max = g_max.join(cap_groups, on="g", how="inner")
    if flow_eff is not None:
        flow_eff = flow_eff.join(cap_groups, on="g", how="inner")
        if flow_eff.height == 0:
            flow_eff = None
    if flow_noEff is not None:
        flow_noEff = flow_noEff.join(cap_groups, on="g", how="inner")
        if flow_noEff.height == 0:
            flow_noEff = None
    if flow_eff is None and flow_noEff is None:
        return (None, None, None, None)
    return (g_max, flow_eff, flow_noEff, p_cap)


def _load_indirect(sd: Path, pss: pl.DataFrame | None, dt: pl.DataFrame,
                    inp: Path | None = None,
                    *,
                    provider: "object | None" = None):
    if pss is None: return (None, None, None, None, None, None)
    p = sd / "process__method_indirect.csv"
    if not _provider_has(provider, "solve_data/process__method_indirect", p):
        return (None, None, None, None, None, None)
    raw = _provider_read(provider, "solve_data/process__method_indirect", p).pipe(rename_to_axis, {"process":"p"})
    if raw.height == 0: return (None, None, None, None, None, None)
    indirect = raw.select("p").unique()
    # Cross-axis compares: sink/source (e-axis) vs p (p-axis).  Per
    # contract p ⊂ e; up-cast p to e so the compare runs in Enum
    # natively without Utf8 materialisation.
    inputs  = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("sink") == cast_dim(pl.col("p"), None, "e")))
    outputs = pss.filter((pl.col("p").is_in(indirect["p"])) & (pl.col("source") == cast_dim(pl.col("p"), None, "e")))

    # The .mod's conversion_indirect LHS multiplies each source-side
    # v_flow by ``p_process_source_conversion_flow_coeff[p, source]`` and
    # the RHS sum by ``p_process_sink_conversion_flow_coeff[p, sink]``
    # (.mod:2557-2580).  Most scenarios have all coefs = 1 (the default);
    # a zero coefficient effectively drops that flow from the conversion
    # equation; the ``coal_chp_extraction`` scenario uses non-default
    # sink coefficients ({heat: 0.2, west: 2.0}) to encode the iso-fuel
    # relationship via the source-side capacity bound.  Build optional
    # Params restricted to the indirect inputs / outputs sets — only
    # when non-default coefficients are present — and let model.py
    # multiply them into the conversion equation.  Zero-coefficient
    # rows are still anti-joined out (so they don't survive into
    # ``inputs`` / ``outputs``).
    p_source_flow_coef = None
    p_sink_flow_coef = None
    if inp is not None:
        src_path = inp / "p_process_source_conversion_flow_coeff.csv"
        if _provider_has(provider,
                          "input/p_process_source_conversion_flow_coeff",
                          src_path):
            srcdf = _provider_read(provider,
                                     "input/p_process_source_conversion_flow_coeff",
                                     src_path)
            if srcdf.height > 0 and "p_process_source_conversion_flow_coeff" in srcdf.columns:
                src_long = (srcdf
                    .pipe(rename_to_axis, {"process": "p", "source": "source",
                             "p_process_source_conversion_flow_coeff": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "source", "coef"))
                zero_src = src_long.filter(pl.col("coef") == 0.0).select("p", "source")
                if zero_src.height > 0:
                    inputs = inputs.join(zero_src, on=["p", "source"], how="anti")
                # If any non-default, non-zero coefficient applies to a
                # surviving (p, source) row, build a Param covering ALL
                # surviving (p, source) pairs (defaulted to 1.0 where not
                # listed) so the inner-join in v_flow * Param doesn't drop
                # rows.  Zero-coef rows have already been removed.
                nonzero_nondefault = src_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    in_pair = inputs.select("p", "source").unique()
                    if in_pair.height > 0:
                        merged = (in_pair.join(src_long, on=["p", "source"], how="left")
                                          .with_columns(pl.col("coef").fill_null(1.0))
                                          .rename({"coef": "value"}))
                        p_source_flow_coef = Param(("p", "source"), merged)
        sink_path = inp / "p_process_sink_conversion_flow_coeff.csv"
        if _provider_has(provider,
                          "input/p_process_sink_conversion_flow_coeff",
                          sink_path):
            sinkdf = _provider_read(provider,
                                      "input/p_process_sink_conversion_flow_coeff",
                                      sink_path)
            if sinkdf.height > 0 and "p_process_sink_conversion_flow_coeff" in sinkdf.columns:
                sink_long = (sinkdf
                    .pipe(rename_to_axis, {"process": "p", "sink": "sink",
                             "p_process_sink_conversion_flow_coeff": "coef"})
                    .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                    .select("p", "sink", "coef"))
                zero_sink = sink_long.filter(pl.col("coef") == 0.0).select("p", "sink")
                if zero_sink.height > 0:
                    outputs = outputs.join(zero_sink, on=["p", "sink"], how="anti")
                nonzero_nondefault = sink_long.filter(
                    (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
                if nonzero_nondefault.height > 0:
                    out_pair = outputs.select("p", "sink").unique()
                    if out_pair.height > 0:
                        merged = (out_pair.join(sink_long, on=["p", "sink"], how="left")
                                           .with_columns(pl.col("coef").fill_null(1.0))
                                           .rename({"coef": "value"}))
                        p_sink_flow_coef = Param(("p", "sink"), merged)

    # Phase E.3: ``process_indirect_dt`` no longer materialised;
    # consumers call ``_pdt_join.compute_process_indirect_dt`` on demand.
    return (indirect, inputs, outputs, None,
            p_source_flow_coef, p_sink_flow_coef)


def _load_user_constraints(inp: Path, pss: pl.DataFrame | None, dt: pl.DataFrame,
                            *,
                            provider: "object | None" = None):
    """Returns 12 items:
    flow_cstr_idx, flow_cstr_coef, constraint_constant, cdt_eq, cdt_le, cdt_ge,
    n_inv_cstr_coef, p_inv_cstr_coef, n_state_cstr_coef,
    n_prebuilt_cstr_coef, p_prebuilt_cstr_coef, has_user_cstr.

    The ``*_inv_cstr_coef`` Params carry
    ``p_<entity>_constraint_invested_capacity_coeff`` data;
    ``n_state_cstr_coef`` carries ``p_node_constraint_state_coeff``
    (user-cstr v_state contribution); the ``*_prebuilt_cstr_coef`` Params
    carry ``p_<entity>_constraint_prebuilt_capacity_coeff``
    (existing + prior-period invest)."""
    if pss is None: return [None]*12
    cs_path = inp / "constraint__sense.csv"
    if not _provider_has(provider, "input/constraint__sense", cs_path):
        return [None]*12
    cs = _provider_read(provider, "input/constraint__sense", cs_path).pipe(rename_to_axis, {"constraint":"cn"})
    if cs.height == 0: return [None]*12
    coef_path = inp / "p_process_node_constraint_flow_coeff.csv"
    flow_cstr_idx = None
    # Δ.12-drop: ``p_flow_constraint_coef`` produced authoritatively by
    # ``apply_derived_b.p_flow_constraint_coef_from_source`` when pss is
    # non-empty.  We retain the CSV read because ``flow_constraint_idx``
    # (the index frame, not a Param) is still needed by downstream
    # constraint emission and isn't produced by an override-chain helper.
    # The constraint axis column is ``cn`` (not ``c``) to disambiguate
    # from the commodity axis — see the c_collision review note in
    # ``schemas/flextool_axis_contract.json``.
    flow_cstr_coef = None
    if _provider_has(provider,
                      "input/p_process_node_constraint_flow_coeff",
                      coef_path):
        coef_long = (_provider_read(
                provider,
                "input/p_process_node_constraint_flow_coeff",
                coef_path)
            .pipe(rename_to_axis, {"process":"p","node":"n","constraint":"cn",
                     "p_process_node_constraint_flow_coeff":"coef"})
            .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
            .select("p","n","cn","coef"))
        # Cross-axis join: pss carries ``source``/``sink`` (e-axis) joined
        # against ``coef_long.n`` (n-axis).  Per the contract n ⊂ e, so
        # up-cast coef_long.n to e-Enum once; the join then composes
        # natively in Enum without any Utf8 materialisation at the join
        # boundary.
        coef_long_e = coef_long.with_columns(
            cast_dim(pl.col("n"), None, "e"))
        src_match = (pss.join(
                            coef_long_e,
                            left_on=["p","source"], right_on=["p","n"],
                            how="inner")
                       .select("p", "source", "sink", "cn", "coef"))
        sink_match = (pss.join(
                            coef_long_e,
                            left_on=["p","sink"], right_on=["p","n"],
                            how="inner")
                        .select("p", "source", "sink", "cn", "coef"))
        if src_match.height + sink_match.height > 0:
            joined = (pl.concat([src_match, sink_match], how="vertical")
                        .group_by(["p","source","sink","cn"])
                        .agg(pl.col("coef").sum()))
            flow_cstr_idx  = joined.select("p","source","sink","cn")
    # Δ.12-drop: ``p_node_constraint_invested_capacity_coeff`` /
    # ``p_process_constraint_invested_capacity_coeff`` /
    # ``p_node_constraint_state_coeff`` /
    # ``p_node_constraint_prebuilt_capacity_coeff`` /
    # ``p_process_constraint_prebuilt_capacity_coeff`` and
    # ``p_constraint_constant`` produced authoritatively by
    # ``apply_direct_params``.  Seeds dropped.
    n_inv_cstr_coef = None
    p_inv_cstr_coef = None
    n_state_cstr_coef = None
    n_prebuilt_cstr_coef = None
    p_prebuilt_cstr_coef = None
    constraint_constant = None
    cdt_eq = cdt_le = cdt_ge = None
    for s, slot in [("equal","eq"), ("less_than","le"), ("greater_than","ge")]:
        cs_s = cs.filter(pl.col("sense")==s).select("cn")
        if cs_s.height > 0:
            axes = cs_s.join(dt, how="cross")
            if   slot=="eq": cdt_eq = axes
            elif slot=="le": cdt_le = axes
            else:            cdt_ge = axes
    return (flow_cstr_idx, flow_cstr_coef, constraint_constant,
            cdt_eq, cdt_le, cdt_ge, n_inv_cstr_coef, p_inv_cstr_coef,
            n_state_cstr_coef, n_prebuilt_cstr_coef, p_prebuilt_cstr_coef,
            True)


def _read_wide_e_d(path: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    """Read a CSV in either long-format (``entity, period, value``) or
    wide-format (``solve, period, e1, e2, …``) and return long form
    ``(e, d, value)``."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    df = _provider_read(provider, name, path)
    if "solve" in df.columns:
        df = df.drop("solve")
    if df.height == 0 or "period" not in df.columns:
        return pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64})
    # Long-format detection: explicit (entity, period, value) columns.
    if {"entity", "period", "value"}.issubset(df.columns):
        return (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                  .with_columns(value=pl.col("value")
                                        .cast(pl.Float64, strict=False)
                                        .fill_null(0.0))
                  .select("e", "d", "value"))
    val_cols = [c for c in df.columns if c != "period"]
    return (df.unpivot(on=val_cols, index=["period"], variable_name="e",
                       value_name="value")
              .pipe(rename_to_axis, {"period": "d"})
              .with_columns(value=pl.col("value")
                                  .cast(pl.Float64, strict=False)
                                  .fill_null(0.0)))


def _load_invest(sd: Path, dt: pl.DataFrame, inp: Path,
                  pss: pl.DataFrame | None,
                  *,
                  db_reader: "object | None" = None,
                  provider: "object | None" = None) -> dict:
    """Load invest/divest sets and per-(e, d) cost params.  Empty when
    neither ed_invest nor ed_divest has any row.

    Δ.19 — when ``db_reader`` is supplied AND the active solve is a
    recognised synthetic ``<base>_<anchor>`` (see
    :func:`_derived_params._resolve_synthetic_solve`), the 8 invest-set
    seed reads (``ed_invest``, ``ed_divest``, ``ed_invest_forbidden``,
    ``pd_invest``, ``pd_divest``, ``nd_invest``, ``nd_divest``,
    ``edd_invest``) are skipped — :func:`apply_synthetic_invest_sets`
    populates these fields from Spine during ``_apply_db_overrides``.
    The cost-param seeds (``ed_lifetime_fixed_cost`` etc.) and per-
    period cap seeds remain on the CSV path.
    """
    blank = dict(
        ed_invest_set=None, ed_divest_set=None,
        pd_invest_set=None, pd_divest_set=None,
        nd_invest_set=None, nd_divest_set=None,
        edd_invest_set=None, edd_invest_lookback_set=None,
        edd_divest_active=None,
        p_entity_max_units=None,
        ed_lifetime_fixed_cost=None,
        ed_lifetime_fixed_cost_divest=None,
        ed_entity_annual_discounted=None,
        ed_entity_annual_divest_discounted=None,
        e_invest_total=None, e_divest_total=None,
        e_invest_max_total=None, e_divest_max_total=None,
        ed_invest_period_set=None, ed_divest_period_set=None,
        ed_invest_max_period=None, ed_divest_max_period=None,
    )
    # Δ.19 — detect synthetic ``<base>_<anchor>`` solve.  When matched,
    # the apply_synthetic_invest_sets path covers the 8 set frames, so
    # we skip those reads entirely (set frames default to None and the
    # override populates them after this loader returns).
    skip_set_seeds = False
    if db_reader is not None:
        try:
            scc = sd / "solve_current.csv"
            if _provider_has(provider, "solve_data/solve_current", scc):
                from ._derived_params import (_read_active_solve,
                                                _resolve_synthetic_solve,
                                                _solve_in_spine)
                # workdir = sd.parent (sd is solve_data/)
                active_solve = _read_active_solve(sd.parent, provider=provider)
                if (active_solve is not None
                        and not _solve_in_spine(db_reader, active_solve)
                        and _resolve_synthetic_solve(db_reader, active_solve)
                            is not None):
                    skip_set_seeds = True
        except Exception:  # pragma: no cover — defensive
            skip_set_seeds = False

    # Workdir-CSV seeds for the invest/divest cascade live in
    # ``_invest_seeds.py`` — keeps the synthetic-solve fallback I/O off
    # ``input.py`` so the override chain stays the dominant data path.
    from ._invest_seeds import (
        read_invest_set as _seed_invest_set,
        read_forbidden_no_investment as _seed_forbidden_ni,
        read_set_seed as _seed_set,
        read_edd_invest as _seed_edd_invest,
    )

    if skip_set_seeds:
        # Set frames are populated by apply_synthetic_invest_sets after
        # this loader returns; the dispatch-only empty case is detected
        # from Spine directly (``solve.invest_periods`` for ``<base>``
        # at the anchor key is empty → no invest activity).  No disk
        # reads needed for the gate.
        from ._derived_params import _solve_periods as _sp
        invest_periods = _sp(db_reader, active_solve, "invest_periods")
        if not invest_periods:
            return blank
        # Cost-cascade seeds + per-period caps continue below; skip the
        # set-frame seeds (ed_inv/ed_div/forbid/pd/nd/edd).
        ed_inv = None  # type: ignore[assignment]
        ed_div = None  # type: ignore[assignment]
        pd_inv = None  # type: ignore[assignment]
        pd_div = None  # type: ignore[assignment]
        nd_inv = None  # type: ignore[assignment]
        nd_div = None  # type: ignore[assignment]
        edd_inv = None  # type: ignore[assignment]
    else:
        ed_inv = _seed_invest_set(sd, "ed_invest", "e", provider=provider)
        ed_div = _seed_invest_set(sd, "ed_divest", "e", provider=provider)
        if ed_inv.height == 0 and ed_div.height == 0:
            return blank

        forbid = _seed_forbidden_ni(sd, provider=provider)
        if forbid.height > 0:
            ed_inv = ed_inv.join(forbid, on=["e", "d"], how="anti")

        # Δ.18 — CSV-fallback seeds for pd/nd_invest_set, pd/nd_divest_set,
        # edd_invest_set.  The override chain (``apply_derived_c`` via the
        # lazy LFs in ``_derived_existing.py``) overlays these when active.
        # For synthetic per-sub-solve fixtures the snapshot CSV is the only
        # source.
        pd_inv = _seed_set(sd, "pd_invest", "p", provider=provider)
        pd_div = _seed_set(sd, "pd_divest", "p", provider=provider)
        nd_inv = _seed_set(sd, "nd_invest", "n", provider=provider)
        nd_div = _seed_set(sd, "nd_divest", "n", provider=provider)

        edd_inv = _seed_edd_invest(sd, provider=provider)

    edd_div = pl.DataFrame(
        schema={"p": pl.Utf8, "d_divest": pl.Utf8, "d": pl.Utf8})
    edd_inv_lookback = pl.DataFrame(
        schema={"e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8})

    # ``p_entity_max_units`` is produced by apply_derived_c BUT some
    # callers exercise the pure-CSV path (e.g. tempdir-symlink-based
    # tests in test_orchestration_parity.py).  Keep the seed.
    # TODO(Δ.12c+): retire when all callers either pass an explicit
    # db_reader= or a workdir whose tests.sqlite + scenario auto-resolve.
    p_max_units = Param(("e", "d"),
        _read_wide_e_d(sd / "p_entity_max_units.csv", provider=provider)
            .filter(pl.col("value") > 0)
            .select("e", "d", "value")) if _provider_has(
                provider, "solve_data/p_entity_max_units",
                sd / "p_entity_max_units.csv") else None

    # Δ.18 — CSV-fallback seeds for the lifetime / NPV / total / max-period
    # cost cascade.  These were dropped in Δ.12-drop / Δ.17 because the
    # override chain (``apply_derived_f`` / ``apply_direct_params`` /
    # ``apply_projection_params``) is authoritative.  But for synthetic
    # per-sub-solve fixtures (e.g. ``invest_5weeks_p2020`` — an artefact of
    # flextool's per-period sub-solving that doesn't exist as a row in
    # Spine), the override returns None for every field that depends on
    # ``solve.invest_periods`` / ``solve.realized_periods`` lookups.  The
    # seed CSV is the snapshot canonical and the only available source.
    # The override chain runs after ``_load_invest`` and overlays its
    # values when it has data; when it returns None, the seed persists.
    def _read_e_d(name: str) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _read_wide_e_d(f, provider=provider)
        if df.height == 0:
            return None
        return Param(("e", "d"), df.select("e", "d", "value"))

    def _read_e(name: str) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0:
            return None
        if "entity" not in df.columns or "value" not in df.columns:
            return None
        return Param(("e",),
                     df.pipe(rename_to_axis, {"entity": "e"})
                       .with_columns(value=pl.col("value")
                                             .cast(pl.Float64, strict=False)
                                             .fill_null(0.0))
                       .select("e", "value"))

    ed_lifetime_fc_seed       = _read_e_d("ed_lifetime_fixed_cost")
    ed_lifetime_fc_div_seed   = _read_e_d("ed_lifetime_fixed_cost_divest")
    ed_annual_disc_seed       = _read_e_d("ed_entity_annual_discounted")
    ed_annual_div_disc_seed   = _read_e_d("ed_entity_annual_divest_discounted")
    e_invest_max_total_seed   = _read_e("e_invest_max_total")
    e_divest_max_total_seed   = _read_e("e_divest_max_total")
    ed_invest_max_period_seed = _read_e_d("ed_invest_max_period")
    ed_divest_max_period_seed = _read_e_d("ed_divest_max_period")

    # ``ed_invest_period_set`` / ``ed_divest_period_set`` (set frames
    # of (e, d) pairs with per-period invest / divest caps) —
    # ``apply_derived_c`` populates these via
    # ``ed_invest_period_set_from_source`` /
    # ``ed_divest_period_set_from_source`` when active_solve is in
    # Spine; the workdir-CSV seeds below cover the synthetic per-sub-
    # solve case.
    from ._invest_seeds import read_period_set as _seed_period_set

    # Multi-solve handoff state.  These files are written between
    # sub-solves; the .mod uses them as constants on the
    # max/min Invest/Divest_entity_total + cumulative-group
    # constraints.  Empty / missing → no prior-solve activity.
    # Δ.12c — ``_read_handoff_e_d`` and ``_read_handoff_e`` retired:
    # the handoff carriers (``p_entity_previously_invested_capacity`` /
    # ``p_entity_invested`` / ``p_entity_divested``) are now produced by
    # ``apply_derived_f`` BEFORE ``apply_existing_chain`` consumes them,
    # so the CSV seed is no longer load-bearing.

    # Δ.12-drop: CSV seeds for fields whose override-chain helpers are
    # authoritative are set to None below.  The override chain repopulates
    # each field unconditionally:
    #   * ``e_invest_max_total`` / ``e_divest_max_total`` / ``e_invest_min_total``
    #     / ``e_divest_min_total``  ← ``apply_direct_params._e_total_param``.
    #   * ``ed_lifetime_fixed_cost`` / ``ed_lifetime_fixed_cost_divest`` /
    #     ``ed_entity_annual_discounted`` /
    #     ``ed_entity_annual_divest_discounted`` ← ``apply_derived_f`` (npv).
    #   * ``ed_invest_max_period`` / ``ed_divest_max_period``
    #     ← ``apply_direct_params`` via ``ed_*_max_period_from_source``.
    #
    # Δ.12c — handoff carriers (``p_entity_previously_invested_capacity`` /
    # ``p_entity_invested`` / ``p_entity_divested``) dropped: now produced
    # authoritatively by ``apply_derived_f``.  Since Δ.12c moved
    # ``apply_existing_chain`` to run AFTER ``apply_derived_f`` (was inside
    # ``apply_derived_d``), the chain summation sees the carriers populated
    # by the override helper without the seed.
    def _hnz(x):  # height-non-zero predicate that tolerates None
        return x if (x is not None and x.height > 0) else None

    return dict(
        ed_invest_set=_hnz(ed_inv),
        ed_divest_set=_hnz(ed_div),
        pd_invest_set=_hnz(pd_inv),
        pd_divest_set=_hnz(pd_div),
        nd_invest_set=_hnz(nd_inv),
        nd_divest_set=_hnz(nd_div),
        edd_invest_set=_hnz(edd_inv),
        edd_invest_lookback_set=edd_inv_lookback if edd_inv_lookback.height > 0 else None,
        edd_divest_active=edd_div if edd_div.height > 0 else None,
        p_entity_max_units=p_max_units,
        # Δ.18 — CSV-fallback seeds (override chain overlays when it has
        # data; for synthetic per-sub-solve fixtures the snapshot CSV is
        # the only source).
        ed_lifetime_fixed_cost=ed_lifetime_fc_seed,
        ed_lifetime_fixed_cost_divest=ed_lifetime_fc_div_seed,
        ed_entity_annual_discounted=ed_annual_disc_seed,
        ed_entity_annual_divest_discounted=ed_annual_div_disc_seed,
        e_invest_total=None,
        e_divest_total=None,
        e_invest_max_total=e_invest_max_total_seed,
        e_divest_max_total=e_divest_max_total_seed,
        ed_invest_period_set=_seed_period_set(sd, "ed_invest_period",
                                                provider=provider),
        ed_divest_period_set=_seed_period_set(sd, "ed_divest_period",
                                                provider=provider),
        ed_invest_max_period=ed_invest_max_period_seed,
        ed_divest_max_period=ed_divest_max_period_seed,
        p_entity_previously_invested_capacity=None,
        p_entity_invested=None,
        p_entity_divested=None,
    )


def _read_p_process_side(path: Path, side_col: str,
                          *,
                          provider: "object | None" = None) -> dict[str, pl.DataFrame]:
    """Parse ``input/p_process_sink.csv`` or ``input/p_process_source.csv``.

    Canonical (Python-preprocessing-input) format: long, columns
    ``[process, sink_or_source, sourceSinkParam, p_process_sink_or_source]``.
    The .mod also printf's a wide debug-export to ``solve_data/`` with
    a 2-row hierarchical header (process row, side row, then
    param/value rows) — supported as a fallback.  Returns
    ``{param_name: DataFrame(p, side, value)}``."""
    out: dict[str, pl.DataFrame] = {}
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return out
    df = _provider_read(provider, name, path)
    # canonical long: ``process, <side>, sourceSinkParam, p_process_<side>``
    if {"process", "sourceSinkParam"}.issubset(df.columns):
        if df.height == 0:
            return out
        # value column is last; rename for uniformity
        value_col = df.columns[-1]
        out_df = (df.filter(pl.col(value_col) != 0)
                    .pipe(rename_to_axis, {"process": "p",
                             "sourceSinkParam": "param",
                             value_col: "value"}))
        for param, sub in out_df.group_by("param", maintain_order=True):
            param_str = param[0] if isinstance(param, tuple) else param
            out[param_str] = sub.select("p", side_col, "value")
        return out
    # The legacy 2-row-header printf-export format used to fall through
    # here; Step 2.5 removed the disk arm and the Provider only carries
    # the canonical long ``(process, <side>, sourceSinkParam, value)``
    # shape, so any other layout is a contract violation — surface it
    # rather than silently parsing a wide CSV that shouldn't exist.
    raise ValueError(
        f"_read_p_process_side: Provider key '{name}' returned a frame "
        f"with unexpected columns {df.columns!r}; expected canonical "
        f"long format with 'process'+'sourceSinkParam'."
    )


def _load_ramp(inp: Path, sd: Path, pss: pl.DataFrame | None,
                *,
                provider: "object | None" = None) -> dict:
    """Load ramp-limit sets and ramp_speed params.  Empty when no
    process_source_sink_ramp_limit_* row is populated."""
    blank = dict(
        process_source_sink_ramp_limit_sink_up=None,
        process_source_sink_ramp_limit_sink_down=None,
        process_source_sink_ramp_limit_source_up=None,
        process_source_sink_ramp_limit_source_down=None,
        p_ramp_speed_up_sink=None,
        p_ramp_speed_down_sink=None,
        p_ramp_speed_up_source=None,
        p_ramp_speed_down_source=None,
    )
    if pss is None:
        return blank

    def _read_set(name: str) -> pl.DataFrame | None:
        p = sd / f"process_source_sink_ramp_limit_{name}.csv"
        key = f"solve_data/process_source_sink_ramp_limit_{name}"
        if not _provider_has(provider, key, p):
            return None
        df = _provider_read(provider, key, p)
        if df.height == 0: return None
        return df.pipe(rename_to_axis, {"process": "p"}).select("p", "source", "sink")

    sets = {f"process_source_sink_ramp_limit_{name}": _read_set(name)
            for name in ("sink_up", "sink_down", "source_up", "source_down")}
    if not any(s is not None for s in sets.values()):
        return blank

    # Δ.12-drop: ``p_ramp_speed_*_{sink,source}`` Params produced
    # authoritatively by ``apply_direct_params`` via
    # ``p_ramp_speed_up_sink_from_source`` etc.  Seeds dropped (the
    # ``_read_p_process_side`` reads of ``p_process_sink.csv`` /
    # ``p_process_source.csv`` are removed).  The four ramp-limit set
    # frames above remain CSV-loaded — apply_projection_params doesn't
    # cover those set partitions today.
    return dict(
        **sets,
        p_ramp_speed_up_sink   = None,
        p_ramp_speed_down_sink = None,
        p_ramp_speed_up_source = None,
        p_ramp_speed_down_source = None,
    )


def _load_online(inp: Path, sd: Path, dt: pl.DataFrame,
                  pss: pl.DataFrame | None,
                  *, source: "InputSource | None" = None,
                  provider: "object | None" = None) -> dict:
    """Load online / min_load / startup data.  Empty dict-of-Nones when
    no process is online."""
    blank = dict(
        process_online=None, process_online_linear=None,
        process_online_integer=None, process_minload=None,
        process_min_load_eff=None,
        p_online_dt=None, pdt_online_linear=None, pdt_online_integer=None,
        p_min_load=None, p_startup_cost=None, p_section=None,
        pdt_uptime_set=None, pdt_downtime_set=None,
        uptime_lookback=None, downtime_lookback=None,
    )
    if pss is None:
        return blank
    online_path = sd / "process_online.csv"
    if not _provider_has(provider, "solve_data/process_online", online_path):
        return blank
    p_online = _provider_read(provider, "solve_data/process_online", online_path).pipe(rename_to_axis, {"process": "p"})
    if p_online.height == 0:
        return blank

    p_online_lin = _provider_read(
        provider, "solve_data/process_online_linear",
        sd / "process_online_linear.csv").pipe(rename_to_axis, {"process": "p"})
    p_online_int_path = sd / "process_online_integer.csv"
    p_online_int = (_provider_read(
                        provider, "solve_data/process_online_integer",
                        p_online_int_path).pipe(rename_to_axis, {"process": "p"})
                    if _provider_has(provider,
                                       "solve_data/process_online_integer",
                                       p_online_int_path) else
                    pl.DataFrame(schema={"p": pl.Utf8}))
    p_minload_path = sd / "process_minload.csv"
    p_minload = (_provider_read(provider, "solve_data/process_minload",
                                  p_minload_path).pipe(rename_to_axis, {"process": "p"})
                 if _provider_has(provider, "solve_data/process_minload",
                                   p_minload_path) else
                 pl.DataFrame(schema={"p": pl.Utf8}))

    # ct_method: min_load_efficiency rows.  Canonical input file is
    # ``input/process__ct_method.csv`` with columns (process, ct_method);
    # the .mod also printf's a debug-export to ``solve_data/`` with
    # column ``method`` — tolerate either schema/location.
    ctm_path = inp / "process__ct_method.csv"
    ctm_key = "input/process__ct_method"
    if not _provider_has(provider, ctm_key, ctm_path):
        ctm_path = sd / "process__ct_method.csv"
        ctm_key = "solve_data/process__ct_method"
    p_min_load_eff = pl.DataFrame(schema={"p": pl.Utf8})
    if _provider_has(provider, ctm_key, ctm_path):
        ctm = _provider_read(provider, ctm_key, ctm_path).pipe(rename_to_axis, {"process": "p"})
        method_col = "ct_method" if "ct_method" in ctm.columns else "method"
        p_min_load_eff = (ctm.filter(pl.col(method_col) == "min_load_efficiency")
                          .select("p").unique())

    # p_online_dt — block-aware variable indexing (process, period, step)
    p_odt = _provider_read(provider, "solve_data/p_online_dt_set",
                              sd / "p_online_dt_set.csv").pipe(rename_to_axis, {"process": "p", "step": "t"})
    p_odt = p_odt.select("p", "period", "t").pipe(rename_to_axis, {"period": "d"})

    # Δ.12-drop: ``p_min_load`` produced authoritatively by
    # ``apply_direct_params.p_min_load_from_source``.  Seed dropped.
    p_min_load = None

    # startup_cost is per (p, d) — produced by ``apply_direct_params``
    # via ``p_startup_cost_from_source`` with full scalar / 1d_map(period)
    # broadcast cascade.  Δ.17b Gap C: local seed dropped.
    # However the LOCAL pdt_online_lin / pdt_online_int sets DEPEND on
    # the same (p, d) keying.  We reconstruct sc_long from the override
    # if it has fired by reading flex_data.p_startup_cost — but
    # flex_data isn't available at this scope.  Defensive: rebuild
    # pdt_online_lin / pdt_online_int from the source-side helper too.
    p_startup_cost = None
    pdt_online_lin = pdt_online_int = None
    if source is not None:
        from ._direct_params import p_startup_cost_from_source
        sc_param = p_startup_cost_from_source(source, period_filter=dt)
        if sc_param is not None and sc_param.frame.height > 0:
            sc_frame = sc_param.frame.filter(pl.col("value") != 0)
            if sc_frame.height > 0:
                p_startup_cost = Param(("p", "d"),
                                          sc_frame.select("p", "d", "value"))
                sc_p = sc_frame.select("p", "d").unique()
                pdt_online_lin = (p_odt.join(p_online_lin, on="p", how="inner")
                                       .join(sc_p, on=["p", "d"], how="inner"))
            if p_online_int.height > 0:
                pdt_online_int = (p_odt.join(p_online_int, on="p", how="inner")
                                        .join(sc_p, on=["p", "d"], how="inner"))

    # Δ.12-drop: ``p_section`` produced authoritatively by
    # ``apply_derived_c.p_section_from_source`` when dt and classified
    # processes are non-empty.  Seed dropped.
    p_section = None

    # Δ.12-drop: ``pdt_uptime_set`` / ``pdt_downtime_set`` /
    # ``uptime_lookback`` / ``downtime_lookback`` produced authoritatively
    # by ``apply_derived_c`` (helpers ``uptime_lookback_from_source`` /
    # ``downtime_lookback_from_source`` + ``pdt_uptime_set_from_lookback``
    # / ``pdt_downtime_set_from_lookback``).  Seeds dropped.
    return dict(
        process_online=p_online,
        process_online_linear=p_online_lin,
        process_online_integer=p_online_int,
        process_minload=p_minload,
        process_min_load_eff=p_min_load_eff,
        p_online_dt=p_odt,
        pdt_online_linear=pdt_online_lin,
        pdt_online_integer=pdt_online_int,
        p_min_load=p_min_load,
        p_startup_cost=p_startup_cost,
        p_section=p_section,
        pdt_uptime_set=None,
        pdt_downtime_set=None,
        uptime_lookback=None,
        downtime_lookback=None,
    )


def _load_storage(inp: Path, sd: Path, dt: pl.DataFrame,
                   nb: pl.DataFrame,
                   pss_eff: pl.DataFrame | None,
                   pss_noEff: pl.DataFrame | None,
                   cap_pd: pl.DataFrame | None,
                   unitsize: Param | None,
                   block_layout: "BlockLayout | None" = None,
                   *,
                   provider: "object | None" = None) -> dict:
    """Load storage feature: nodeState set, capacity bounds, binding
    methods, dtttdt, and source-side nodeBalance topology.

    Returns dict with all storage-related fields.  Empty if no
    nodeState entries."""
    # Source-side nodeBalance flow mappings.  These describe processes
    # whose source is a balance node — needed for both transmission
    # (network scenarios with no storage) and storage discharge.  Compute
    # unconditionally so a network-without-storage fixture still has the
    # source flow contributions in nodeBalance.
    flow_from_nb_eff = flow_from_nb_noEff = None
    # ``source`` is the entity-union (``e``) enum; ``nb["n"]`` is the
    # narrower node (``n``) enum.  Use a semi-join keyed on the node
    # rename (cast against ``e`` via rename_to_axis) so the membership
    # filter survives Phase 4 activation without a cross-vocab is_in.
    # ``n`` is then aliased from the surviving e-typed source column
    # (all surviving values are nodes per the semi-join, but the dtype
    # stays e so the downstream block-compat join with eb_l.n (also
    # cast to e below) composes natively in Enum).
    _enums_local = get_global_axis_enums()
    nb_as_source = nb.lazy().pipe(rename_to_axis, {"n": "source"})
    if pss_eff is not None:
        flow_from_nb_eff = (pss_eff.lazy()
            .join(nb_as_source.select("source"), on="source", how="semi")
            .with_columns(n=cast_dim(pl.col("source"), _enums_local, "e"))
            .select("p","source","sink","n")
            .collect())
    if pss_noEff is not None:
        flow_from_nb_noEff = (pss_noEff.lazy()
            .join(nb_as_source.select("source"), on="source", how="semi")
            .with_columns(n=cast_dim(pl.col("source"), _enums_local, "e"))
            .select("p","source","sink","n")
            .collect())

    # Apply the same block-compatibility filter as in flow_from_n /
    # flow_to_n: arc contributes to node's nodeBalance only if (b_n, b_f)
    # has an overlap row.  Δ.2: consume frames from the in-memory
    # ``BlockLayout`` (loaded once at the top of ``load_flextool``) when
    # supplied; otherwise the CSVs would still be on disk but no caller
    # passes None today.
    if (block_layout is not None
            and block_layout.process_side_block_frame.height > 0
            and block_layout.entity_block_frame.height > 0
            and block_layout.overlap_set_frame.height > 0
            and (flow_from_nb_eff is not None or flow_from_nb_noEff is not None)):
        psb_l = block_layout.process_side_block_frame.pipe(
            rename_to_axis, {"process": "p", "block": "b_f"})
        # Match the e-Enum ``n`` dtype on flow_from_nb_{eff,noEff}: rename
        # entity → e (cast against the union axis), then alias to "n"
        # keeping the e-Enum dtype.  Downstream join on "n" composes
        # natively in Enum (n ⊂ e).
        eb_l = (block_layout.entity_block_frame.pipe(
                rename_to_axis, {"entity": "e", "block": "bk"})
            .with_columns(n=cast_dim(pl.col("e"), None, "e"))
            .drop("e"))
        block_compat_l = block_layout.block_compat()
        if psb_l.height > 0 and eb_l.height > 0 and block_compat_l.height > 0:
            psb_src_l = psb_l.filter(pl.col("side") == "source").select("p", "b_f")
            def _filter_by_compat(df: pl.DataFrame) -> pl.DataFrame:
                if df is None or df.height == 0:
                    return df
                with_blocks = (df
                    .join(psb_src_l, on="p", how="left")
                    .join(eb_l, on="n", how="left"))
                with_blocks = with_blocks.with_columns(
                    b_f=pl.col("b_f").fill_null(lit_axis("default", "block")),
                    bk=pl.col("bk").fill_null(lit_axis("default", "block")),
                )
                f = (with_blocks
                    .join(block_compat_l, on=["bk", "b_f"], how="inner")
                    .select("p", "source", "sink", "n").unique())
                if f.height < df.height and f.height > 0:
                    return f
                return df
            flow_from_nb_eff = _filter_by_compat(flow_from_nb_eff)
            flow_from_nb_noEff = _filter_by_compat(flow_from_nb_noEff)

    # dtttdt — needed for ramps and online dynamics regardless of storage.
    dtttdt = _read_step_previous(sd / "step_previous.csv", provider=provider)

    blank = dict(
        nodeState = None, nodeState_dt = None, nodeState_first_dt = None,
        p_state_upper = None, p_state_unitsize = None,
        p_state_self_discharge = None, p_state_start = None,
        p_state_existing_capacity = None,
        storage_bind_within_timeblock = None,
        storage_bind_forward_only = None,
        storage_fix_start = None,
        storage_use_reference_value = None,
        p_storage_state_reference_value = None,
        p_storage_state_reference_price = None,
        dtttdt = dtttdt,
        dtttdt_forward_only = None,
        nodeStateBlock = None,
        period_block = None,
        period_block_succ = None,
        period_block_time = None,
        dtttdt_block_interior = None,
        nodeState_rp = None,
        rp_base_period_set = None,
        rp_base_chain = None,
        rp_base_first = None,
        rp_base_last = None,
        rp_block_first = None,
        p_rp_last_step = None,
        rp_base__rep = None,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = None,
        p_roll_continue_state = None,
        n_fix_storage_quantity = None,
        ndt_fix_storage_quantity = None,
        p_fix_storage_quantity = None,
        n_fix_storage_usage = None,
        ndt_fix_storage_usage = None,
        p_fix_storage_usage = None,
        dtt_timeline_matching = None,
        period_branch = None,
        period_last = None,
        nodeState_last_dt = None,
        node_profile_upper = None,
        node_profile_lower = None,
        node_profile_fixed = None,
        p_node_availability = None,
    )
    ns_path = sd / "nodeState.csv"
    if not _provider_has(provider, "solve_data/nodeState", ns_path):
        return blank
    nodeState = _provider_read(provider, "solve_data/nodeState", ns_path).pipe(rename_to_axis, {"node": "n"})
    if nodeState.height == 0:
        return blank

    # Phase E.3: ``nodeState_dt`` is no longer materialised here.
    # Consumers call ``_pdt_join.compute_nodeState_dt`` on demand.
    # ``nodeState_first_dt`` below stays materialised — it's a small
    # one-row-per-node slice and the CSV-fallback resolution for
    # ``first_period`` needs the slow-path provider context.

    # First (d, t) per period — used for storage_state_start_binding.
    # The .mod uses ``period_first_of_solve`` for the boundary tests in
    # both the fwd_fix start binding (mod:2197) and the roll_continue
    # term (mod:2196).  ``period_first.csv`` is the legacy single-solve
    # source (often empty in nested / rolling-horizon fixtures), so we
    # prefer ``period_first_of_solve.csv`` when it has rows; otherwise
    # fall back to ``period_first.csv``; otherwise the first dt period.
    # TODO(Δ.18+): no canonical helper yet for ``nodeState_first_dt`` —
    # the override-chain produces ``period_branch`` / ``dtt_timeline_matching``
    # but not the simple per-(n, d) first timestep.  ``_read_period_first``
    # in ``_derived_params`` reads similar data but is workdir-only and
    # used by the existing-chain helper.
    fpos_path = sd / "period_first_of_solve.csv"
    fp_path = sd / "period_first.csv"
    first_period = None
    if _provider_has(provider, "solve_data/period_first_of_solve", fpos_path):
        df = _provider_read(provider, "solve_data/period_first_of_solve", fpos_path)
        if df.height > 0:
            first_period = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()
    if first_period is None and _provider_has(provider, "solve_data/period_first", fp_path):
        df = _provider_read(provider, "solve_data/period_first", fp_path)
        if df.height > 0:
            first_period = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()
    if first_period is None:
        # Fallback: take the lexicographically smallest period.
        first_period = (dt.select("d").unique()
                          .sort("d").head(1))
    # Phase E.3: build ``first_dt`` lazily from ``nodeState`` × ``dt``
    # without persisting the full cross-product on ``flex_data``.
    first_dt = (
        nodeState.lazy()
        .join(dt.lazy(), how="cross")
        .join(first_period.lazy(), on="d", how="inner")
        .group_by("n", "d")
        .agg(pl.col("t").min().alias("t"))
        .select("n", "d", "t")
        .collect()
    )

    # Δ.18 — restore CSV-fallback seeds for ``p_state_existing_capacity``
    # / ``p_state_unitsize`` / ``p_state_upper``.  They were dropped in
    # Δ.17 batch 1 because ``apply_derived_e`` was authoritative — but
    # for synthetic per-sub-solve fixtures the per-solve override chain
    # is skipped (workdir's ``solve_current.csv`` names a solve not in
    # Spine, see ``_apply_db_overrides`` Δ.18 gate) and the snapshot CSV
    # is the only source.  When the override does run, it overlays
    # these via ``setattr`` so the seed becomes inert.
    if unitsize is not None and cap_pd is not None and nodeState is not None:
        cap_long = _read_capacity(sd / "p_entity_period_existing_capacity.csv",
                                   sd / "p_entity_previously_invested_capacity.csv",
                                   sd / "p_entity_all_existing.csv",
                                   provider=provider)
        unitsize_long = _read_unitsize(_provider_pick(
            provider,
            ("solve_data/p_entity_unitsize", sd / "p_entity_unitsize.csv"),
            ("input/p_entity_unitsize", inp / "p_entity_unitsize.csv"),
        ) or (inp / "p_entity_unitsize.csv"), provider=provider)
        state_existing = (cap_long.pipe(rename_to_axis, {"e": "n", "value": "cap"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n", "d", "cap"))
        state_us_long = (unitsize_long.pipe(rename_to_axis, {"e": "n"})
            .filter(pl.col("n").is_in(nodeState["n"]))
            .select("n", "value"))
        if state_existing.height > 0 and state_us_long.height > 0:
            state_existing_capacity = Param(("n", "d"),
                state_existing.rename({"cap": "value"}))
            state_unitsize = Param(("n",), state_us_long)
            state_upper_long = (state_existing
                .join(state_us_long.rename({"value": "us"}), on="n", how="inner")
                .with_columns(value=pl.col("cap") / pl.col("us"))
                .select("n", "d", "value"))
            state_upper = Param(("n", "d"), state_upper_long)
        else:
            state_unitsize = state_existing_capacity = state_upper = None
    else:
        state_unitsize = state_existing_capacity = state_upper = None

    # Δ.12-drop: ``state_self_discharge`` (``p_state_self_discharge``) and
    # ``state_start`` (``p_state_start``) seeds dropped — both are now
    # produced authoritatively by ``apply_direct_params`` via
    # ``p_state_self_discharge_from_source`` / ``p_state_start_from_source``.
    state_self_discharge = None
    state_start = None

    # Binding methods (sd-level, per node).
    # NOTE: the .mod attaches a (v_state[t] - v_state[t-1]) term in
    # nodeBalance for several binding methods, with subtle differences:
    #   * ``bind_within_timeblock``  — fully cyclic (wraps via
    #     ``t_previous_within_timeset``).
    #   * ``bind_within_period``   — cyclic within period
    #     (``t_previous`` column).
    #   * ``bind_within_solve``    — cyclic within solve
    #     (``t_previous_within_solve``); equivalent to within_timeset for
    #     a single-block dispatch.
    #   * ``bind_forward_only``    — also uses
    #     ``t_previous_within_solve``, BUT the .mod *omits* the
    #     state-change term at the first timestep of the first period
    #     (line 2188 condition).  This makes the storage non-cyclic at
    #     the boundary.
    # In the current parity tests, every fixture that *does*
    # exercise a state node uses ``bind_within_timeblock`` (which is what
    # this loader picks up).  ``work_water_pump`` is the only fixture
    # that uses ``bind_forward_only``, and faithful parity there
    # requires modelling the first-timestep exemption — see
    # questions_for_user.md#water_pump.
    sbm_path = sd / "node__storage_binding_method.csv"
    binding_within_timeblock = None
    binding_forward_only = None
    binding_within_solve = None
    # Phase C — ``binding_within_solve_blended_weights`` is also derived
    # here from the per-solve provider so the partition reflects any
    # cascade-applied silent downgrade (see
    # :func:`._native_run_model._downgrade_rp_methods_for_non_rp_solve`).
    # apply_projection_params no longer touches this field — the
    # per-solve CSV is the single source of truth.
    binding_within_solve_blended_weights = None
    # Phase C — not-yet-implemented methods.  Loaded so the model.py
    # guard can surface a precise error instead of silently emitting
    # zero state-change residuals.  Constraint wiring lands in Phases
    # D / E of the storage-binding-methods restructure.
    binding_within_period_blended_weights = None
    binding_forward_only_blended_weights = None
    if _provider_has(provider, "solve_data/node__storage_binding_method", sbm_path):
        sbm = _provider_read(provider, "solve_data/node__storage_binding_method", sbm_path)
        # Column names in this file have varied — handle both schemas
        if "storage_binding_method" in sbm.columns:
            sbm = sbm.pipe(rename_to_axis, {"node":"n","storage_binding_method":"method"})
        elif "method" in sbm.columns:
            sbm = sbm.pipe(rename_to_axis, {"node":"n"})
        binding_within_timeblock = (sbm.filter(pl.col("method")=="bind_within_timeblock")
                                     .select("n").unique())
        fo = (sbm.filter(pl.col("method")=="bind_forward_only")
                 .select("n").unique())
        if fo.height > 0:
            binding_forward_only = fo
        ws = (sbm.filter(pl.col("method")=="bind_within_solve")
                 .select("n").unique())
        if ws.height > 0:
            binding_within_solve = ws
        wsbw = (sbm.filter(pl.col("method")=="bind_within_solve_blended_weights")
                   .select("n").unique())
        if wsbw.height > 0:
            binding_within_solve_blended_weights = wsbw
        wpbw = (sbm.filter(pl.col("method")=="bind_within_period_blended_weights")
                   .select("n").unique())
        if wpbw.height > 0:
            binding_within_period_blended_weights = wpbw
        fobw = (sbm.filter(pl.col("method")=="bind_forward_only_blended_weights")
                   .select("n").unique())
        if fobw.height > 0:
            binding_forward_only_blended_weights = fobw

    # ``bind_forward_only`` mirrors ``bind_within_solve`` (uses the
    # ``t_previous_within_solve`` lag column) BUT the .mod omits the
    # state-change term at the very first timestep of the first period
    # (flextool.mod:2188).  We model that exemption by dropping the
    # corresponding row from the lag frame — the wrap row whose
    # ``t_previous_within_solve`` jumps backwards.  Sorting by (d, t)
    # and dropping the first row is equivalent for single-solve fixtures
    # (the native engine is single-solve per build).
    dtttdt_forward_only_df = None
    if binding_forward_only is not None and dtttdt is not None and dtttdt.height > 0:
        dtttdt_forward_only_df = dtttdt.sort("d", "t").slice(1)
        if dtttdt_forward_only_df.height == 0:
            dtttdt_forward_only_df = None

    # node__storage_start_end_method is read by .mod from input/
    # (flextool.mod:662) — that's the canonical user-input source.
    # solve_data/ may have a .mod-printf debug-export with renamed
    # column "method"; tolerate either schema.
    sse_path = inp / "node__storage_start_end_method.csv"
    sse_key = "input/node__storage_start_end_method"
    if not _provider_has(provider, sse_key, sse_path):
        sse_path = sd / "node__storage_start_end_method.csv"
        sse_key = "solve_data/node__storage_start_end_method"
    fix_start = None
    fix_end = None
    fix_start_end = None
    if _provider_has(provider, sse_key, sse_path):
        sse = _provider_read(provider, sse_key, sse_path)
        if "storage_start_end_method" in sse.columns:
            sse = sse.pipe(rename_to_axis, {"node":"n","storage_start_end_method":"method"})
        elif "method" in sse.columns:
            sse = sse.pipe(rename_to_axis, {"node":"n"})
        fix_start = (sse.filter(pl.col("method")=="fix_start").select("n").unique())
        fix_end = (sse.filter(pl.col("method")=="fix_end").select("n").unique())
        fix_start_end = (sse.filter(pl.col("method")=="fix_start_end").select("n").unique())

    # node__storage_solve_horizon_method (.mod:663): nodes with method
    # ``use_reference_value`` get a v_state pin at the last timestep of
    # the last period, equal to ``reference_value × existing/unitsize``.
    sshm_path = inp / "node__storage_solve_horizon_method.csv"
    sshm_key = "input/node__storage_solve_horizon_method"
    if not _provider_has(provider, sshm_key, sshm_path):
        sshm_path = sd / "node__storage_solve_horizon_method.csv"
        sshm_key = "solve_data/node__storage_solve_horizon_method"
    use_reference_value = None
    if _provider_has(provider, sshm_key, sshm_path):
        sshm = _provider_read(provider, sshm_key, sshm_path)
        col = ("storage_solve_horizon_method"
               if "storage_solve_horizon_method" in sshm.columns
               else "method")
        sshm = sshm.pipe(rename_to_axis, {"node": "n", col: "method"})
        use_reference_value = (sshm
            .filter(pl.col("method") == "use_reference_value")
            .select("n").unique())
        # Filter out nodes with a competing storage method (mod:2806-2811):
        # fix_end / fix_start_end / bind_within_solve / bind_within_period /
        # bind_within_timeblock / bind_intraperiod_blocks.
        # ``nodeStateBlock`` is the set carrying bind_intraperiod_blocks
        # (loaded below; we look it up via the on-disk CSV here to keep
        # ordering simple).  bind_within_period not exercised yet.
        nsb_for_excl = None
        nsb_path_local = sd / "nodeStateBlock.csv"
        if _provider_has(provider, "solve_data/nodeStateBlock", nsb_path_local):
            nsb_df_local = _provider_read(provider, "solve_data/nodeStateBlock", nsb_path_local)
            if nsb_df_local.height > 0:
                nsb_for_excl = nsb_df_local.pipe(rename_to_axis, {"node": "n"}).select("n")
        for excl in (fix_end, fix_start_end,
                     binding_within_solve, binding_within_timeblock,
                     nsb_for_excl):
            if excl is not None and excl.height > 0:
                use_reference_value = use_reference_value.join(
                    excl, on="n", how="anti")
        if use_reference_value.height == 0:
            use_reference_value = None

    # Δ.17c Gap C: ``p_storage_state_reference_value`` produced
    # authoritatively by ``apply_direct_params`` via
    # ``p_storage_state_reference_value_from_source`` (uses the
    # ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade).  The
    # ``use_reference_value`` consumer-side filter runs in apply_direct_params
    # — kept here as None so the override path takes ownership.
    p_ssrv = None

    # B1a — load ``p_storage_state_reference_price`` (n, d) from the canonical
    # ``solve_data/p_storage_state_reference_price.csv`` emitted by
    # ``_emit_arc_unions.emit_p_storage_state_reference_price``.  Schema is
    # ``(node, period, value)``; per-(node, period) parameter consumed by
    # B1b's ``use_reference_price`` objective term.  Field is loaded but
    # unused at this commit; B1b adds the consumer.
    p_storage_state_reference_price = None
    psrp_path = sd / "p_storage_state_reference_price.csv"
    if _provider_has(provider, "solve_data/p_storage_state_reference_price",
                     psrp_path):
        df_psrp = _provider_read(
            provider, "solve_data/p_storage_state_reference_price", psrp_path,
        )
        if df_psrp.height > 0:
            value_col = ("p_storage_state_reference_price"
                         if "p_storage_state_reference_price" in df_psrp.columns
                         else "value")
            df_psrp = (df_psrp
                       .pipe(rename_to_axis,
                             {"node": "n", "period": "d",
                              value_col: "value"})
                       .with_columns(value=pl.col("value")
                                              .cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                       .select("n", "d", "value"))
            if df_psrp.height > 0:
                p_storage_state_reference_price = Param(("n", "d"), df_psrp)

    # ─── Intraperiod-block (bind_intraperiod_blocks) sets ────────────────
    # Used by ``stateConstantWithinBlock_eq`` and ``nodeBalanceBlock_eq``
    # in model.py for nodes whose binding method is ``bind_intraperiod_blocks``.
    # Δ.18 — CSV-fallback seed for ``nodeStateBlock``.  Override chain
    # (``apply_derived_e`` via ``nodeStateBlock_from_source``) overlays
    # this when active; for dump_csvs roundtrip workdirs (no DB) the
    # canonical CSV is the only source.
    nodeStateBlock = None
    nsb_path_seed = sd / "nodeStateBlock.csv"
    if _provider_has(provider, "solve_data/nodeStateBlock", nsb_path_seed):
        nsb_df_seed = _provider_read(provider, "solve_data/nodeStateBlock", nsb_path_seed)
        if nsb_df_seed.height > 0 and "node" in nsb_df_seed.columns:
            nodeStateBlock = nsb_df_seed.pipe(rename_to_axis, {"node": "n"}).select("n").unique()
    period_block = None
    period_block_succ = None
    period_block_time = None
    dtttdt_block_interior = None

    # ─── RP-blended-weights (bind_within_solve_blended_weights) sets / params ───
    # Eight per-solve frames driving the .mod's intra-period state-change
    # branch for ``nodeState_rp`` plus the three ``rp_inter_period_*``
    # constraints (.mod:2197-2200, .mod:2965-2997).  Emitted by
    # ``_emit_leaf_sets.emit_node_state_subsets`` (nodeState_rp),
    # ``_emit_per_solve`` (rp_base_period_set), and
    # ``_emit_solve_writers._compute_rp_frames`` (chain / first / last /
    # block_first / block_start_last / weights).  Loaded here; model.py
    # wiring lands in Phase 5+.
    from flextool.engine_polars import _provider_keys as _K_rp
    nodeState_rp = None
    nsrp_path = sd / "nodeState_rp.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_NODE_STATE_RP, nsrp_path):
        df_nsrp = _provider_read(
            provider, _K_rp.SOLVE_DATA_NODE_STATE_RP, nsrp_path,
        )
        if df_nsrp.height > 0 and "node" in df_nsrp.columns:
            nodeState_rp = (df_nsrp
                            .pipe(rename_to_axis, {"node": "n"})
                            .select("n").unique())
            if nodeState_rp.height == 0:
                nodeState_rp = None

    rp_base_period_set = None
    rpbps_path = sd / "rp_base_period_set.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BASE_PERIOD_SET,
                     rpbps_path):
        df_rpbps = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BASE_PERIOD_SET, rpbps_path,
        )
        if df_rpbps.height > 0:
            # Emitter writes a single column ``period`` (see
            # ``_emit_per_solve`` line ~169 via ``_emit_singles``).
            col = "period" if "period" in df_rpbps.columns else df_rpbps.columns[0]
            rp_base_period_set = (df_rpbps
                                  .rename({col: "b"})
                                  .select("b").unique())
            if rp_base_period_set.height == 0:
                rp_base_period_set = None

    rp_base_chain = None
    rpbc_path = sd / "rp_base_chain.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BASE_CHAIN, rpbc_path):
        df_rpbc = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BASE_CHAIN, rpbc_path,
        )
        if df_rpbc.height > 0:
            # Emitter columns: ``(base_start, prev_base_start)`` for the
            # within_solve variant; Phase E adds an optional ``period``
            # column for the within_period variant so the downstream
            # cyclic constraint can pair endpoints per FlexTool period.
            _renames = {"base_start": "b", "prev_base_start": "b_prev"}
            if "period" in df_rpbc.columns:
                _renames["period"] = "d"
            df_rpbc = df_rpbc.rename(_renames)
            _keep = ["b", "b_prev"] + (["d"] if "d" in df_rpbc.columns else [])
            rp_base_chain = df_rpbc.select(_keep).unique()
            if rp_base_chain.height == 0:
                rp_base_chain = None

    rp_base_first = None
    rpbf_path = sd / "rp_base_first.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BASE_FIRST, rpbf_path):
        df_rpbf = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BASE_FIRST, rpbf_path,
        )
        if df_rpbf.height > 0:
            # Phase E — preserve the optional ``period`` column (within
            # period variant emits one row per FlexTool period).
            _renames = {"base_start": "b"}
            if "period" in df_rpbf.columns:
                _renames["period"] = "d"
            df_rpbf = df_rpbf.rename(_renames)
            _keep = ["b"] + (["d"] if "d" in df_rpbf.columns else [])
            rp_base_first = df_rpbf.select(_keep).unique()
            if rp_base_first.height == 0:
                rp_base_first = None

    rp_base_last = None
    rpbl_path = sd / "rp_base_last.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BASE_LAST, rpbl_path):
        df_rpbl = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BASE_LAST, rpbl_path,
        )
        if df_rpbl.height > 0:
            # Phase E — preserve the optional ``period`` column.
            _renames = {"base_start": "b"}
            if "period" in df_rpbl.columns:
                _renames["period"] = "d"
            df_rpbl = df_rpbl.rename(_renames)
            _keep = ["b"] + (["d"] if "d" in df_rpbl.columns else [])
            rp_base_last = df_rpbl.select(_keep).unique()
            if rp_base_last.height == 0:
                rp_base_last = None

    rp_block_first = None
    rpblkf_path = sd / "rp_block_first.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BLOCK_FIRST,
                     rpblkf_path):
        df_rpblkf = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BLOCK_FIRST, rpblkf_path,
        )
        if df_rpblkf.height > 0:
            # Emitter columns: ``(period, step)``.
            rp_block_first = (df_rpblkf
                              .pipe(rename_to_axis,
                                    {"period": "d", "step": "t"})
                              .select("d", "t").unique())
            if rp_block_first.height == 0:
                rp_block_first = None

    # ``p_rp_last_step`` — stored as a DataFrame relation [r, last_step]
    # (NOT a numeric Param) — see audit §6 Risk #1.  Source basename is
    # ``rp_block_start_last.csv`` with header ``(rep_start, last_step)``;
    # we rename ``rep_start → r`` so the .mod's symbol ``r`` is the join
    # key.  Phase 7 implements ``v_state[n, d, p_rp_last_step[r]]`` as a
    # join on ``r → last_step``.
    p_rp_last_step = None
    prpls_path = sd / "rp_block_start_last.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_BLOCK_START_LAST,
                     prpls_path):
        df_prpls = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_BLOCK_START_LAST, prpls_path,
        )
        if df_prpls.height > 0:
            p_rp_last_step = (df_prpls
                              .rename({"rep_start": "r"})
                              .select("r", "last_step").unique())
            if p_rp_last_step.height == 0:
                p_rp_last_step = None

    # ``rp_base__rep`` — Param keyed by (b, r) with weight value.  Source
    # basename ``rp_weights.csv`` carries ``(base_start, rep_start,
    # weight)`` rows with weight as a stringified float (see
    # ``_compute_rp_frames``).
    rp_base__rep = None
    rpw_path = sd / "rp_weights.csv"
    if _provider_has(provider, _K_rp.SOLVE_DATA_RP_WEIGHTS, rpw_path):
        df_rpw = _provider_read(
            provider, _K_rp.SOLVE_DATA_RP_WEIGHTS, rpw_path,
        )
        if df_rpw.height > 0:
            df_rpw = (df_rpw
                      .rename({"base_start": "b",
                               "rep_start": "r",
                               "weight": "value"})
                      .with_columns(value=pl.col("value")
                                              .cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                      .select("b", "r", "value"))
            if df_rpw.height > 0:
                rp_base__rep = Param(("b", "r"), df_rpw)

    # ─── Invariant: when nodeState_rp is non-empty, the four tightly
    # coupled fields below must also be non-empty.  Silent absence has no
    # graceful degradation — the model just emits no inter-period state
    # constraint and the LP solves the wrong problem.  Mirrors the
    # ``_fast_load.py:686-690`` block invariant pattern.
    if nodeState_rp is not None and nodeState_rp.height > 0:
        _rp_required = (
            ("rp_base_period_set", rp_base_period_set),
            ("rp_base__rep", rp_base__rep),
            ("rp_block_first", rp_block_first),
            ("p_rp_last_step", p_rp_last_step),
        )
        for _name, _field in _rp_required:
            _frame = (_field.frame if hasattr(_field, "frame") else _field)
            if _frame is None or _frame.height == 0:
                raise ValueError(
                    f"FlexData loader (backstop check): nodeState_rp is "
                    f"non-empty ({nodeState_rp.height} node(s)) but the "
                    f"tightly-coupled field `{_name}` is missing or "
                    f"empty.  Under the storage-binding-methods-restructure "
                    f"(Phase C onwards), the per-solve downgrade in "
                    f"_native_run_model._downgrade_rp_methods_for_non_rp_solve "
                    f"normally strips bind_within_solve_blended_weights "
                    f"to bind_within_solve when a solve's active "
                    f"timeset has no representative_period_weights.  "
                    f"If you are seeing THIS error, the downgrade was "
                    f"bypassed — either load_flextool was called "
                    f"directly (not via _native_run_model) or there "
                    f"is an upstream emitter bug.  Please report this "
                    f"with the calling code.  Required RP set family: "
                    f"nodeState_rp, rp_base_period_set, rp_base__rep, "
                    f"rp_block_first, p_rp_last_step."
                )

    # ─── Multi-resolution block synthesis ───────────────────────────────
    # Δ.17b Gap A: synthesis is performed end-to-end by the override chain
    # (``period_block_family_from_source`` + ``nodeStateBlock_from_source``
    # + ``dtttdt_block_interior_from_dtttdt`` mirror the local logic).
    # Local synthesis dropped here.

    # ─── Rolling-horizon nested-solve framework (flextool.mod:2196 + 2760) ─
    # p_nested_model.csv: { modelParam, p_nested_model } with rows
    # solveFirst / solveLast.  Tri-state: missing → None (single-solve);
    # 0 → False; non-zero → True.
    p_nested_solve_first: bool | None = None
    nm_path = sd / "p_nested_model.csv"
    if _provider_has(provider, "solve_data/p_nested_model", nm_path):
        nm = _provider_read(provider, "solve_data/p_nested_model", nm_path)
        if nm.height > 0:
            # Column may be ``p_nested_model`` (canonical) or ``value``.
            value_col = "p_nested_model" if "p_nested_model" in nm.columns else "value"
            row = nm.filter(pl.col("modelParam") == "solveFirst")
            if row.height > 0:
                p_nested_solve_first = bool(int(row[value_col][0]))

    # Phase 4.2-0 — read the rolling end-state carrier from the canonical
    # handoff Provider key (``handoff/roll_end_state``, schema
    # ``[node, value]``).  The translator populates the key from
    # ``SolveHandoff.roll_end_state`` at iteration start; the legacy
    # ``solve_data/p_roll_continue_state`` CSV-fallback path is gone.
    #
    # Upward feedback (specs/feature_fixes.md §1): when a dispatch sub-
    # solve has populated ``handoff/upward_roll_end_state`` (carrying
    # the dispatch's realized end-of-horizon v_state for parent storage
    # consumption), prefer it over the sequential-prior roll_end_state.
    # Per user direction: always-on for any storage→dispatch nesting;
    # parent's next-roll initial state comes from the dispatch's
    # realized state, not the parent's own previous prediction.
    from flextool.engine_polars import _provider_keys as K
    from flextool.engine_polars._provider_translators import read_handoff_frame
    p_roll_continue_state = None
    df_rcs = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STATE)
    df_upward = read_handoff_frame(provider, K.HANDOFF_UPWARD_ROLL_END_STATE)
    if df_upward is not None and df_upward.height > 0:
        df_rcs = df_upward
    if df_rcs is not None and df_rcs.height > 0:
        df_rcs = (df_rcs
                  .pipe(rename_to_axis, {"node": "n"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "value"))
        if df_rcs.height > 0:
            p_roll_continue_state = Param(("n",), df_rcs)

    # Phase 4.1d — read the rolling-handoff fix_storage_quantity carrier
    # from the canonical handoff Provider key
    # (``handoff/fix_storage_quantity``, schema
    # ``[node, period, step, p_fix_storage_quantity]``).  The translator
    # populates the key from ``SolveHandoff.fix_storage_quantity`` at
    # iteration start; the legacy ``solve_data/fix_storage_quantity``
    # read path is gone.
    from flextool.engine_polars import _provider_keys as K
    from flextool.engine_polars._provider_translators import read_handoff_frame
    p_fix_storage_quantity = None
    df_fsq = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_QUANTITY)
    if df_fsq is not None and df_fsq.height > 0:
        df_fsq = (df_fsq
                  .pipe(rename_to_axis, {"node": "n", "period": "d",
                                         "step": "t",
                                         "p_fix_storage_quantity": "value"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "d", "t", "value"))
        if df_fsq.height > 0:
            p_fix_storage_quantity = Param(("n", "d", "t"), df_fsq)

    # ``n_fix_storage_quantity`` / ``ndt_fix_storage_quantity`` are
    # derived from ``p_fix_storage_quantity`` after the helper assigns it
    # (see ``_finalise_fix_storage_index_sets`` below).
    n_fix_storage_quantity = None
    ndt_fix_storage_quantity = None
    if p_fix_storage_quantity is not None:
        fsq_frame = p_fix_storage_quantity.frame
        n_fix_storage_quantity = fsq_frame.select("n").unique()
        ndt_fix_storage_quantity = fsq_frame.select("n", "d", "t").unique()

    # Phase B4-pre — read the rolling-handoff fix_storage_usage carrier
    # from the canonical handoff Provider key
    # (``handoff/fix_storage_usage``, schema
    # ``[node, period, step, p_fix_storage_usage]``).  Populated by B3's
    # extractor; consumed by the B4 constraint (added in the next commit).
    # Loader-only at this commit — fields exist on FlexData but are not
    # yet referenced by any LP block, so zero LP effect.
    p_fix_storage_usage = None
    df_fsu = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_USAGE)
    if df_fsu is not None and df_fsu.height > 0:
        df_fsu = (df_fsu
                  .pipe(rename_to_axis, {"node": "n", "period": "d",
                                         "step": "t",
                                         "p_fix_storage_usage": "value"})
                  .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0))
                  .select("n", "d", "t", "value"))
        if df_fsu.height > 0:
            p_fix_storage_usage = Param(("n", "d", "t"), df_fsu)

    n_fix_storage_usage = None
    ndt_fix_storage_usage = None
    if p_fix_storage_usage is not None:
        fsu_frame = p_fix_storage_usage.frame
        n_fix_storage_usage = fsu_frame.select("n").unique()
        ndt_fix_storage_usage = fsu_frame.select("n", "d", "t").unique()

    # ``dtt_timeline_matching`` (d, t, t_upper) and ``period_branch``
    # (d_upper, d) — seed from the snapshot CSVs.  ``apply_derived_e``
    # (the override-chain producer) is bypassed entirely for
    # *synthetic* per-sub-solve workdirs (their ``solve_current`` names a
    # solve that doesn't exist in Spine — the multi_fullYear_battery_nested
    # rolling-handoff snapshots are the canonical case), so without this
    # seed both fields stay ``None`` and ``node_balance_fix_quantity_eq_lower``
    # never fires for the upper-level anchor pin at the last timestep —
    # leaving the LP severely under-constrained at the rolling-handoff
    # boundary (vq_state_up slack lights up, objective explodes).
    dtt_timeline_matching = None
    tlm_path = sd / "timeline_matching_map.csv"
    if _provider_has(provider, "solve_data/timeline_matching_map", tlm_path):
        df_tlm = _provider_read(provider, "solve_data/timeline_matching_map", tlm_path)
        if df_tlm.height > 0:
            dtt_timeline_matching = (df_tlm
                .pipe(rename_to_axis,
                      {"period": "d", "step": "t", "upper_step": "t_upper"})
                .select("d", "t", "t_upper").unique())
            if dtt_timeline_matching.height == 0:
                dtt_timeline_matching = None

    period_branch = None
    pb_path = sd / "period__branch.csv"
    if _provider_has(provider, "solve_data/period__branch", pb_path):
        df_pb = _provider_read(provider, "solve_data/period__branch", pb_path)
        if df_pb.height > 0:
            period_branch = (df_pb
                .pipe(rename_to_axis, {"period": "d", "branch": "d_upper"})
                .select("d_upper", "d").unique())
            if period_branch.height == 0:
                period_branch = None

    # period_last: (d,).
    # TODO(Δ.18+): no canonical helper yet for ``period_last`` — this is
    # preprocessing-only data (flextool's per-solve last-period anchor for
    # storage/handoff binding).  The override-chain produces
    # ``dtt_timeline_matching`` / ``period_branch`` / handoff carriers but
    # not the simple ``period_last`` set frame.
    period_last_df = None
    pl_path = sd / "period_last.csv"
    if _provider_has(provider, "solve_data/period_last", pl_path):
        df = _provider_read(provider, "solve_data/period_last", pl_path)
        if df.height > 0:
            period_last_df = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()

    # nodeState_last_dt: (n, d, t) — last (d, t) per node, used as the index
    # for ``node_balance_fix_quantity_eq_lower``.  Built from
    # block_period_time_last (bk, d, t) × entity_block (e=n, bk) × nodeState.
    # The block-axis column is ``bk`` (not ``b``) to disambiguate from the
    # branch axis — see the b_collision review note in
    # ``schemas/flextool_axis_contract.json``.
    # Δ.2: consume frames from in-memory ``BlockLayout``.
    nodeState_last_dt = None
    if (nodeState is not None and nodeState.height > 0
            and block_layout is not None
            and block_layout.block_period_time_last_frame.height > 0
            and block_layout.entity_block_frame.height > 0):
        bptl = block_layout.block_period_time_last_frame.pipe(
            rename_to_axis, {"block": "bk", "period": "d", "step": "t"}).select("bk", "d", "t")
        eb = block_layout.entity_block_frame.pipe(
            rename_to_axis, {"entity": "n", "block": "bk"}).select("n", "bk")
        if bptl.height > 0 and eb.height > 0:
            nodeState_last_dt = (nodeState.select("n")
                .join(eb, on="n", how="inner")
                .join(bptl, on="bk", how="inner")
                .select("n", "d", "t").unique())
            if nodeState_last_dt.height == 0:
                nodeState_last_dt = None

    # Δ.17 — ``node_profile_upper`` / ``node_profile_lower`` /
    # ``node_profile_fixed`` produced authoritatively by
    # ``apply_projection_params`` (Γ.2 SIMPLE_PROJECTIONS).  Verified
    # row-by-row parity across all 72 work_* fixtures with
    # node__profile__profile_method.csv.  Seed dropped (1
    # ``_read_csv_file`` call retired).
    node_profile_upper_df = node_profile_lower_df = node_profile_fixed_df = None

    # Δ.17c Gap C: ``p_node_availability`` produced authoritatively by
    # ``apply_direct_params`` via ``p_node_availability_from_source``
    # (uses the ``_param_shapes`` resolver — full broadcast cascade with
    # explicit allow-list).  Local pdtNode.csv slice dropped.
    p_node_avail = None

    return dict(
        nodeState = nodeState,
        # Phase E.3: ``nodeState_dt`` no longer materialised; consumers
        # call ``_pdt_join.compute_nodeState_dt`` on demand.
        nodeState_dt = None,
        nodeState_first_dt = first_dt,
        p_state_upper = state_upper,
        p_state_unitsize = state_unitsize,
        p_state_self_discharge = state_self_discharge,
        p_state_start = state_start,
        p_state_existing_capacity = state_existing_capacity,
        storage_bind_within_timeblock = binding_within_timeblock,
        storage_bind_forward_only = binding_forward_only,
        storage_bind_within_solve = binding_within_solve,
        # Phase C — single source of truth for the RP-flavoured partition
        # is the per-solve CSV; apply_projection_params no longer
        # contributes here so the cascade's silent-downgrade rewrite
        # propagates correctly.
        storage_bind_within_solve_blended_weights = binding_within_solve_blended_weights,
        # Phase C — not-yet-implemented method partitions; consumed by
        # the model.py guard which raises FlexToolConfigError when
        # non-empty.
        storage_bind_within_period_blended_weights = binding_within_period_blended_weights,
        storage_bind_forward_only_blended_weights = binding_forward_only_blended_weights,
        storage_fix_start = fix_start,
        storage_use_reference_value = use_reference_value,
        p_storage_state_reference_value = p_ssrv,
        p_storage_state_reference_price = p_storage_state_reference_price,
        dtttdt = dtttdt,
        dtttdt_forward_only = dtttdt_forward_only_df,
        nodeStateBlock = nodeStateBlock,
        period_block = period_block,
        period_block_succ = period_block_succ,
        period_block_time = period_block_time,
        dtttdt_block_interior = dtttdt_block_interior,
        nodeState_rp = nodeState_rp,
        rp_base_period_set = rp_base_period_set,
        rp_base_chain = rp_base_chain,
        rp_base_first = rp_base_first,
        rp_base_last = rp_base_last,
        rp_block_first = rp_block_first,
        p_rp_last_step = p_rp_last_step,
        rp_base__rep = rp_base__rep,
        flow_from_nodeBalance_eff = flow_from_nb_eff,
        flow_from_nodeBalance_noEff = flow_from_nb_noEff,
        p_nested_solve_first = p_nested_solve_first,
        p_roll_continue_state = p_roll_continue_state,
        n_fix_storage_quantity = n_fix_storage_quantity,
        ndt_fix_storage_quantity = ndt_fix_storage_quantity,
        p_fix_storage_quantity = p_fix_storage_quantity,
        n_fix_storage_usage = n_fix_storage_usage,
        ndt_fix_storage_usage = ndt_fix_storage_usage,
        p_fix_storage_usage = p_fix_storage_usage,
        dtt_timeline_matching = dtt_timeline_matching,
        period_branch = period_branch,
        period_last = period_last_df,
        nodeState_last_dt = nodeState_last_dt,
        node_profile_upper = node_profile_upper_df,
        node_profile_lower = node_profile_lower_df,
        node_profile_fixed = node_profile_fixed_df,
        p_node_availability = p_node_avail,
    )


def _load_profiles(inp: Path, sd: Path, pss: pl.DataFrame | None,
                    unitsize: Param | None,
                    cap_pd: pl.DataFrame | None,
                    *,
                    provider: "object | None" = None):
    """Load profile_flow_upper/lower/fixed mappings.  ``cap_pd`` is the
    (p, d, base_cap) frame; combined with unitsize we get the
    ``existing_count`` term used on the RHS."""
    if pss is None or unitsize is None or cap_pd is None:
        return [None]*6
    pp_path = sd / "process__source__sink__profile__profile_method.csv"
    if not _provider_has(provider,
                          "solve_data/process__source__sink__profile__profile_method",
                          pp_path):
        return [None]*6
    pp = _provider_read(provider,
                          "solve_data/process__source__sink__profile__profile_method",
                          pp_path).pipe(rename_to_axis, {"process":"p"})
    if pp.height == 0:
        return [None]*6
    method_col = "method" if "method" in pp.columns else "profile_method"
    upper = pp.filter(pl.col(method_col)=="upper_limit").select("p","source","sink","profile")
    lower = pp.filter(pl.col(method_col)=="lower_limit").select("p","source","sink","profile")
    fixed = pp.filter(pl.col(method_col)=="fixed").select("p","source","sink","profile")

    # profile values - file is solve, period, time, p1, p2... — wide per profile.
    # TODO(Δ.12c+): retire pdtProfile.csv read when
    # ``apply_profile_cascade`` covers fixtures where the source carries
    # the profile data via a different shape (e.g. fixtures whose pdtProfile
    # rows arrive at flextool from preprocessing rather than from a Spine
    # ``profile.profile_data`` parameter).
    pdt_profile = sd / "pdtProfile.csv"
    profile_value = None
    if _provider_has(provider, "solve_data/pdtProfile", pdt_profile):
        prof_long = _read_wide_per_entity(pdt_profile, rename={"entity":"f"},
                                            provider=provider)
        if "profile" in upper.columns:
            upper = upper.pipe(rename_to_axis, {"profile": "f"})
            lower = lower.pipe(rename_to_axis, {"profile": "f"})
            fixed = fixed.pipe(rename_to_axis, {"profile": "f"})
        profile_value = Param(("f","d","t"), prof_long.select("f","d","t","value"))

    # existing_count = capacity / unitsize per (p, d).  For our scenarios
    # (no investment yet) this equals base_cap_pd.
    existing_count = Param(("p","d"), cap_pd.rename({"base":"value"}))

    # Δ.17c Gap C: ``p_process_availability`` produced authoritatively
    # by ``apply_direct_params`` via ``p_process_availability_from_source``
    # (uses the ``_param_shapes`` resolver — unions ``unit.availability``
    # + ``connection.availability`` with full broadcast cascade).  Local
    # pdtProcess.csv slice dropped.
    availability = None

    return upper, lower, fixed, profile_value, existing_count, availability


# ---------------------------------------------------------------------------
# The single loader

def _load_varcost(sd: Path, pss: pl.DataFrame | None,
                   *,
                   provider: "object | None" = None) -> dict:
    """Load process variable-cost (other_operational_cost) sets and Params.

    The .mod has 4 disjoint sets:
      pssdt_varCost_noEff       — uses pdtProcess__source__sink__dt_varCost
      pssdt_varCost_eff_unit_source — uses pdtProcess_source[…,'other_operational_cost']
      pssdt_varCost_eff_unit_sink   — uses pdtProcess_sink[…,'other_operational_cost']
      pssdt_varCost_eff_connection  — uses pdtProcess[…,'other_operational_cost']
    """
    blank = dict(
        pssdt_varCost_noEff=None,
        pssdt_varCost_eff_unit_source=None,
        pssdt_varCost_eff_unit_sink=None,
        pssdt_varCost_eff_connection=None,
        p_pssdt_varCost=None,
        p_pdt_varCost_source=None,
        p_pdt_varCost_sink=None,
        p_pdt_varCost_process=None,
    )
    if pss is None:
        return blank

    def _read_pssdt_set(name: str) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0:
            return None
        return df.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"}) \
                 .select("p", "source", "sink", "d", "t")

    pssdt_noEff = _read_pssdt_set("pssdt_varCost_noEff")
    pssdt_es = _read_pssdt_set("pssdt_varCost_eff_unit_source")
    pssdt_ek = _read_pssdt_set("pssdt_varCost_eff_unit_sink")
    pssdt_ec = _read_pssdt_set("pssdt_varCost_eff_connection")

    # Δ.18 — CSV-fallback seed for ``p_pssdt_varCost``.  Override chain
    # (``apply_derived_b.p_pssdt_varCost_from_source``) overlays this
    # when active; for synthetic per-sub-solve fixtures the snapshot CSV
    # is the only source.  Reads ``pdtProcess__source__sink__dt_varCost.csv``
    # (long: process, source, sink, period, time, value).  Filtering
    # zero-value rows mirrors the override's "drop zero coefficients" pass.
    p_pssdt_var = None
    pssdt_path = sd / "pdtProcess__source__sink__dt_varCost.csv"
    if _provider_has(provider, "solve_data/pdtProcess__source__sink__dt_varCost", pssdt_path):
        df = _provider_read(provider, "solve_data/pdtProcess__source__sink__dt_varCost", pssdt_path)
        if df.height > 0:
            sliced = (df.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
                        .with_columns(value=pl.col("value")
                                              .cast(pl.Float64, strict=False)
                                              .fill_null(0.0))
                        .filter(pl.col("value") != 0.0))
            if sliced.height > 0:
                p_pssdt_var = Param(
                    ("p", "source", "sink", "d", "t"),
                    sliced.select("p", "source", "sink", "d", "t", "value"))

    # pdtProcess_source[p,source,'other_operational_cost',d,t] — wide param file
    def _slice_pds(name: str, side_col: str) -> Param | None:
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0:
            return None
        sliced = df.filter(pl.col("param") == "other_operational_cost") \
                   .drop("param")
        if sliced.height == 0:
            return None
        sliced = (sliced.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
                          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                              .fill_null(0.0))
                          .filter(pl.col("value") != 0))
        if sliced.height == 0:
            return None
        return Param(("p", side_col, "d", "t"),
                     sliced.select("p", side_col, "d", "t", "value"))

    p_var_src  = _slice_pds("pdtProcess_source", "source")
    p_var_sink = _slice_pds("pdtProcess_sink",   "sink")

    # pdtProcess[p,'other_operational_cost',d,t] — process-level (no source/sink dim)
    p_var_proc = None
    pp_path = sd / "pdtProcess.csv"
    if _provider_has(provider, "solve_data/pdtProcess", pp_path):
        df = _provider_read(provider, "solve_data/pdtProcess", pp_path)
        if df.height > 0:
            sliced = df.filter(pl.col("param") == "other_operational_cost") \
                       .drop("param")
            if sliced.height > 0:
                sliced = (sliced.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
                                  .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                                                      .fill_null(0.0))
                                  .filter(pl.col("value") != 0))
                if sliced.height > 0:
                    p_var_proc = Param(("p", "d", "t"),
                                       sliced.select("p", "d", "t", "value"))

    return dict(
        pssdt_varCost_noEff=pssdt_noEff,
        pssdt_varCost_eff_unit_source=pssdt_es,
        pssdt_varCost_eff_unit_sink=pssdt_ek,
        pssdt_varCost_eff_connection=pssdt_ec,
        p_pssdt_varCost=p_pssdt_var,
        p_pdt_varCost_source=p_var_src,
        p_pdt_varCost_sink=p_var_sink,
        p_pdt_varCost_process=p_var_proc,
    )


def _load_fixed_cost(sd: Path,
                      *,
                      provider: "object | None" = None) -> dict:
    """Load (e, d) ed_fixed_cost and (e, d) p_entity_all_existing.

    Δ.18 — CSV-fallback seeds.  Override chain (``apply_derived_f`` /
    ``apply_existing_chain``) overlays these when active; for synthetic
    per-sub-solve fixtures the snapshot CSV is the only source.  Empty
    or missing CSV → None (override-only path).  All-zero rows are
    dropped from ``p_ed_fixed_cost`` to match the override's filter — the
    override's ``ed_lifetime_fixed_cost_*`` family filters zero rows; the
    ``p_ed_fixed_cost`` helper similarly skips zero-value rows so the
    "no fixed cost" semantic round-trips as None on both paths.
    """
    def _read_e_d_seed(name: str, drop_zero: bool = False) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _read_wide_e_d(f, provider=provider)
        if df.height == 0:
            return None
        if drop_zero:
            df = df.filter(pl.col("value") != 0.0)
        if df.height == 0:
            return None
        return Param(("e", "d"), df.select("e", "d", "value"))

    return dict(
        p_ed_fixed_cost=_read_e_d_seed("ed_fixed_cost", drop_zero=True),
        # ``p_entity_all_existing`` keeps zero rows — the chain consumer
        # uses them as the "no existing" sentinel, distinct from absent.
        p_entity_all_existing=_read_e_d_seed("p_entity_all_existing"),
    )


def _load_node_capacity_for_scaling(sd: Path,
                                     nb: pl.DataFrame,
                                     *,
                                     provider: "object | None" = None) -> dict:
    """Load node_capacity_for_scaling[n, d] for slack-penalty scaling."""
    blank = dict(p_node_capacity_for_scaling=None)
    f = sd / "node_capacity_for_scaling.csv"
    if not _provider_has(provider, "solve_data/node_capacity_for_scaling", f):
        return blank
    df = _provider_read(provider, "solve_data/node_capacity_for_scaling", f)
    if df.height == 0:
        return blank
    df = df.pipe(rename_to_axis, {"node": "n", "period": "d"}) \
           .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
    # Restrict to nodes in nodeBalance to avoid spurious rows
    if nb is not None and nb.height > 0:
        df = df.join(nb, on="n", how="inner")
    if df.height == 0:
        return blank
    return dict(p_node_capacity_for_scaling=Param(("n", "d"), df.select("n", "d", "value")))


def _load_cumulative_invest(inp: Path, sd: Path, dt: pl.DataFrame,
                              *,
                              provider: "object | None" = None) -> dict:
    """Load the new ``FlexData`` fields consumed by ``_cumulative_invest``.

    All fields are independently optional — missing CSV / empty file ⇒ None.
    Sets are filtered to keep only non-empty rows; per-period parameters
    drop all-zero rows so ``has_feature(d)`` won't fire on placeholder
    fixtures whose CSVs exist with all-zero placeholders.
    """
    out: dict = {}

    def _read_set(name: str, src_to_dst: dict[str, str]) -> pl.DataFrame | None:
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0: return None
        rename = {s: d for s, d in src_to_dst.items() if s in df.columns}
        out_df = df.pipe(rename_to_axis, rename).select(*src_to_dst.values()).unique()
        return out_df if out_df.height > 0 else None

    # Δ.17 — ``_read_set_drop_zeros`` / ``_read_e_d_param`` / ``_read_e_param``
    # / ``_slice_pdgroup`` / ``_slice_pgroup`` were dead-code helpers
    # retained from Δ.12-drop (their consumer Params were retired but
    # the inner-function definitions weren't cleaned up).  Removed; the
    # override chain produces the corresponding fields:
    #   * ``ed_invest_min_period`` etc. ← ``apply_direct_params``.
    #   * ``p_group_invest_*_period`` etc. ← ``apply_direct_params``.
    # 5 dead-code ``_read_csv_file`` calls retired.

    def _slice_pdtgroup(param_name: str) -> pl.DataFrame | None:
        """solve_data/pdtGroup.csv slice → (g, d, t, value), zero dropped."""
        f = sd / "pdtGroup.csv"
        if not _provider_has(provider, "solve_data/pdtGroup", f):
            return None
        df = _provider_read(provider, "solve_data/pdtGroup", f)
        if df.height == 0: return None
        sliced = (df.filter(pl.col("param") == param_name)
                    .pipe(rename_to_axis, {"group": "g", "period": "d", "time": "t"})
                    .with_columns(pl.col("value").cast(pl.Float64, strict=False))
                    .filter(pl.col("value").is_not_null() & (pl.col("value") != 0.0))
                    .select("g", "d", "t", "value"))
        return sliced if sliced.height > 0 else None

    # ── Sets (key-only frames) ────────────────────────────────────────────
    out["ed_invest_forbidden_no_investment"] = _read_set(
        "ed_invest_forbidden_no_investment",
        {"entity": "e", "period": "d"})
    out["ed_invest_cumulative"] = _read_set(
        "ed_invest_cumulative", {"entity": "e", "period": "d"})

    # group_entity: prefer solve_data, fallback input/group__entity.csv
    ge = None
    for cand, mapping, key in [
        (sd / "group_entity.csv",   {"group": "g", "entity": "e"}, "solve_data/group_entity"),
        (inp / "group__entity.csv", {"group": "g", "entity": "e"}, "input/group__entity"),
    ]:
        if _provider_has(provider, key, cand):
            df = _provider_read(provider, key, cand)
            if df.height > 0:
                ge = df.pipe(rename_to_axis, mapping).select("g", "e").unique()
                break
    out["group_entity"] = ge

    # group_process_node: solve_data/group_process_node.csv (preprocessed long)
    # or input/group__process__node.csv (raw long)
    gpn = None
    for cand, mapping, key in [
        (sd / "group_process_node.csv",  {"group": "g", "process": "p", "node": "n"}, "solve_data/group_process_node"),
        (inp / "group__process__node.csv", {"group": "g", "process": "p", "node": "n"}, "input/group__process__node"),
    ]:
        if _provider_has(provider, key, cand):
            df = _provider_read(provider, key, cand)
            if df.height > 0:
                gpn = df.pipe(rename_to_axis, mapping).select("g", "p", "n").unique()
                break
    out["group_process_node"] = gpn

    out["g_invest_total"]      = _read_set("g_invest_total", {"group": "g"})
    out["g_divest_total"]      = _read_set("g_divest_total", {"group": "g"})
    out["g_invest_cumulative"] = _read_set("g_invest_cumulative", {"group": "g"})
    out["gd_invest_period"]    = _read_set("gd_invest_period",
                                            {"group": "g", "period": "d"})
    out["gd_divest_period"]    = _read_set("gd_divest_period",
                                            {"group": "g", "period": "d"})

    # Δ.12-drop: ``ed_invest_min_period`` / ``ed_divest_min_period`` /
    # ``ed_cumulative_max_capacity`` / ``ed_cumulative_min_capacity`` /
    # ``e_invest_min_total`` / ``e_divest_min_total`` /
    # ``p_group_invest_max_period`` / ``p_group_invest_min_period`` /
    # ``p_group_retire_max_period`` / ``p_group_retire_min_period`` /
    # ``p_group_invest_max_total`` / ``p_group_invest_min_total`` /
    # ``p_group_retire_max_total`` / ``p_group_retire_min_total`` /
    # ``p_group_invest_max_cumulative`` / ``p_group_invest_min_cumulative`` /
    # ``p_group_max_cumulative_flow`` / ``p_group_min_cumulative_flow`` /
    # ``pd_max_cumulative_flow`` / ``pd_min_cumulative_flow``
    # all produced authoritatively by ``apply_direct_params`` (Δ.4b).
    # Seeds dropped.
    out["ed_invest_min_period"]       = None
    out["ed_divest_min_period"]       = None
    out["ed_cumulative_max_capacity"] = None
    out["ed_cumulative_min_capacity"] = None

    # Δ.18 — CSV-fallback seeds (override chain overlays when it has data;
    # for synthetic per-sub-solve fixtures the snapshot CSV is the only
    # source).
    def _read_e_seed(name: str) -> "Param | None":
        f = sd / f"{name}.csv"
        if not _provider_has(provider, f"solve_data/{name}", f):
            return None
        df = _provider_read(provider, f"solve_data/{name}", f)
        if df.height == 0 or "entity" not in df.columns or "value" not in df.columns:
            return None
        return Param(("e",),
                     df.pipe(rename_to_axis, {"entity": "e"})
                       .with_columns(value=pl.col("value")
                                             .cast(pl.Float64, strict=False)
                                             .fill_null(0.0))
                       .select("e", "value"))
    out["e_invest_min_total"]         = _read_e_seed("e_invest_min_total")
    out["e_divest_min_total"]         = _read_e_seed("e_divest_min_total")

    # Δ.17c Gap C: ``pdt_max_instant_flow`` / ``pdt_min_instant_flow``
    # produced authoritatively by ``apply_direct_params`` via
    # ``pdt_max_instant_flow_from_source`` / ``pdt_min_instant_flow_from_source``
    # (use the ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade).  Local pdtGroup.csv
    # slices dropped.
    pdt_max = None
    pdt_min = None
    out["p_group_invest_max_period"]      = None
    out["p_group_invest_min_period"]      = None
    out["p_group_retire_max_period"]      = None
    out["p_group_retire_min_period"]      = None
    out["p_group_invest_max_total"]       = None
    out["p_group_invest_min_total"]       = None
    out["p_group_retire_max_total"]       = None
    out["p_group_retire_min_total"]       = None
    out["p_group_invest_max_cumulative"]  = None
    out["p_group_invest_min_cumulative"]  = None
    out["p_group_max_cumulative_flow"]    = None
    out["p_group_min_cumulative_flow"]    = None
    out["pd_max_cumulative_flow"]         = None
    out["pd_min_cumulative_flow"]         = None
    out["pdt_max_instant_flow"]           = (Param(("g", "d", "t"), pdt_max)
                                              if pdt_max is not None else None)
    out["pdt_min_instant_flow"]           = (Param(("g", "d", "t"), pdt_min)
                                              if pdt_min is not None else None)
    # Support of pdt_*_instant_flow (rows where param is non-null/non-zero)
    out["gdt_maxInstantFlow"] = (pdt_max.select("g", "d", "t")
                                  if pdt_max is not None else None)
    out["gdt_minInstantFlow"] = (pdt_min.select("g", "d", "t")
                                  if pdt_min is not None else None)

    return out


# ---------------------------------------------------------------------------
# HiGHS solver options (input/solve_mode.csv)
#
# flextool's ``solve_mode.csv`` is keyed (param, solve, value).  As of
# Batches C.3-C.5 none of the three legacy ``highs_*`` string
# shortcuts (``highs_method`` / ``highs_parallel`` / ``highs_presolve``)
# exist any longer — those overrides are now authored on
# ``solver_arguments`` and routed through
# :func:`flextool.engine_polars._solver_dispatch._resolve_effective_highs_options`.
# Only a forward-compatibility set of numeric / boolean HiGHS option
# rows would still be picked up here (table below).  When the CSV
# carries none of those rows the loader returns ``None`` and HiGHS
# defaults stand.
# Other ``param`` rows (notably ``solve_mode``) describe the flextool
# solve framework, not HiGHS, and are ignored here.
#
# When the file lists multiple solves, we pick the row whose ``solve``
# matches ``solve_data/solve_current.csv``.  If ``solve_current`` is
# absent or the row is missing for that solve, we silently fall back to
# HiGHS defaults (``solver_options=None``) — current behavior.

# flextool param → HiGHS canonical option name + coercion
_HIGHS_PARAM_MAP: dict[str, tuple[str, type]] = {
    # Numeric / boolean HiGHS options that flextool *may* emit in the
    # future — wire them up so they Just Work when they appear.
    "highs_time_limit":                 ("time_limit",                 float),
    "highs_mip_rel_gap":                ("mip_rel_gap",                float),
    "highs_mip_abs_gap":                ("mip_abs_gap",                float),
    "highs_threads":                    ("threads",                    int),
    "highs_random_seed":                ("random_seed",                int),
    "highs_output_flag":                ("output_flag",                bool),
    "highs_primal_feasibility_tolerance":
        ("primal_feasibility_tolerance", float),
    "highs_dual_feasibility_tolerance":
        ("dual_feasibility_tolerance",   float),
}


def _coerce_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "yes", "on", "1"): return True
    if s in ("false", "no", "off", "0"): return False
    raise ValueError(f"cannot coerce {v!r} to bool")


def _load_solver_options(sd: Path,
                          *,
                          provider: "object | None" = None) -> dict | None:
    p = sd.parent / "input" / "solve_mode.csv"
    key = "input/solve_mode"
    if not _provider_has(provider, key, p):
        # Fixtures sometimes drop the CSV under solve_data/ instead of input/.
        p = sd / "solve_mode.csv"
        key = "solve_data/solve_mode"
        if not _provider_has(provider, key, p):
            return None
    df = _provider_read(provider, key, p)
    if df.height == 0 or "param" not in df.columns or "value" not in df.columns:
        return None

    # Pick the active solve.  ``solve_current.csv`` has a single ``solve``
    # column with one row.  If multiple HiGHS rows exist for a single
    # ``param`` across solves and we can't disambiguate, prefer the
    # solve_current match; otherwise (single-solve fixtures) take whatever's
    # there.
    cur_path = sd / "solve_current.csv"
    cur_solve: str | None = None
    if _provider_has(provider, "solve_data/solve_current", cur_path):
        cur_df = _provider_read(provider, "solve_data/solve_current", cur_path)
        if cur_df.height > 0 and "solve" in cur_df.columns:
            cur_solve = str(cur_df["solve"][0])

    if cur_solve is not None and "solve" in df.columns:
        df_active = df.filter(pl.col("solve") == cur_solve)
        if df_active.height == 0:
            df_active = df  # fall back: no rows for current solve
    else:
        df_active = df

    out: dict = {}
    for row in df_active.iter_rows(named=True):
        param = str(row["param"]).strip()
        if param not in _HIGHS_PARAM_MAP:
            continue
        opt_name, opt_type = _HIGHS_PARAM_MAP[param]
        raw = row["value"]
        try:
            if opt_type is bool:
                val = _coerce_bool(raw)
            elif opt_type is int:
                val = int(float(raw))   # tolerate "1.0"
            elif opt_type is float:
                val = float(raw)
            else:
                val = str(raw).strip()
        except (TypeError, ValueError):
            # Don't crash on a malformed cell — let HiGHS defaults stand.
            continue
        out[opt_name] = val
    return out or None


def _load_stochastics(inp: Path, sd: Path, dt: pl.DataFrame,
                       *,
                       provider: "object | None" = None) -> dict:
    """Load multi-branch stochastic operational data.

    Mirrors flextool.mod's stochastic feature (mod:38-41, :562-588,
    :873-895, :988, :1978-2142, :4173-4233).  Reads four CSVs:

    * ``solve_data/pdt_branch_weight.csv`` (period, time, value) →
      :class:`Param` keyed (d, t).  Per-branch operational probability
      that multiplies every dispatch-class objective term.  Defaults to
      1.0 per (d, t) when CSV is empty / missing.
    * ``solve_data/pd_branch_weight.csv`` (period, value) →
      :class:`Param` keyed (d,).  Per-branch period-level probability
      for investment-fixed-cost terms.  Defaults to 1.0.
    * ``solve_data/dt_non_anticipativity_set.csv`` (period, time) →
      ``pl.DataFrame``.  Realised-dispatch + fix-storage timesteps where
      the four ``non_anticipativity_*`` constraints fire.  Empty when
      stochastics inactive.
    * ``input/groupIncludeStochastics.csv`` (group,) → ``pl.DataFrame``.
      Groups whose ``group_node`` membership unlocks the storage
      non-anticipativity coupling (``non_anticipativity_storage_use``).

    Also loads the unfiltered ``period__branch.csv`` (anchor → sibling)
    distinct from the existing ``period_branch`` rolling-handoff field
    (which renames columns to ``d_upper``/``d``).  And the active
    ``period_in_use_set`` from ``solve_data/period_in_use_set.csv`` —
    used by the model layer to filter branch periods that exist in the
    metadata-only ``period_branch`` map but aren't part of the actual
    LP (e.g. ``period1_realized`` in the 2_day_stochastic_dispatch
    fixture).
    """
    # Δ.18 — CSV-fallback seeds for ``pdt_branch_weight`` /
    # ``pd_branch_weight`` / ``period_in_use_set``.  Override chain
    # (``apply_branch_cluster`` in ``apply_derived_g``) overlays these
    # when active; for synthetic per-sub-solve fixtures the snapshot
    # CSV is the only source.  pdt_branch_weight defaults to dense-dt
    # × 1.0 when CSV missing / empty (mirrors mod's
    # ``param pdt_branch_weight {(d,t) in dt}`` declaration).
    pdt_branch_weight = None
    pdt_bw_path = sd / "pdt_branch_weight.csv"
    if _provider_has(provider, "solve_data/pdt_branch_weight", pdt_bw_path):
        df = _provider_read(provider, "solve_data/pdt_branch_weight", pdt_bw_path)
        if df.height > 0:
            df = (df.pipe(rename_to_axis, {"period": "d", "time": "t"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "t", "value"))
            base = dt.with_columns(value=pl.lit(1.0)).select("d", "t", "value")
            base = (base
                    .join(df, on=["d", "t"], how="left", suffix="__r")
                    .with_columns(value=pl.coalesce(
                        pl.col("value__r"), pl.col("value")))
                    .select("d", "t", "value"))
            pdt_branch_weight = Param(("d", "t"), base)

    pd_branch_weight = None
    pd_bw_path = sd / "pd_branch_weight.csv"
    if _provider_has(provider, "solve_data/pd_branch_weight", pd_bw_path):
        df = _provider_read(provider, "solve_data/pd_branch_weight", pd_bw_path)
        if df.height > 0:
            df = (df.pipe(rename_to_axis, {"period": "d"})
                    .with_columns(value=pl.col("value")
                                            .cast(pl.Float64, strict=False)
                                            .fill_null(1.0))
                    .select("d", "value"))
            pd_branch_weight = Param(("d",), df)

    # Δ.12-drop: ``dt_non_anticipativity`` / ``period_branch_full``
    # produced authoritatively by ``apply_branch_cluster`` in
    # ``apply_derived_g``.  Seeds dropped.
    dt_non_anticipativity = None
    period_branch_full = None

    # Δ.18 — CSV-fallback seed for ``period_in_use_set``.
    period_in_use_set = None
    piu_path = sd / "period_in_use_set.csv"
    if _provider_has(provider, "solve_data/period_in_use_set", piu_path):
        df = _provider_read(provider, "solve_data/period_in_use_set", piu_path)
        if df.height > 0 and "period" in df.columns:
            period_in_use_set = df.pipe(rename_to_axis, {"period": "d"}).select("d").unique()

    # groupIncludeStochastics: (g,)
    gis_path = inp / "groupIncludeStochastics.csv"
    groupStochastic = None
    if _provider_has(provider, "input/groupIncludeStochastics", gis_path):
        df = _provider_read(provider, "input/groupIncludeStochastics", gis_path)
        if df.height > 0:
            # CSV column is named ``group``; rename to canonical ``g``.
            df = df.pipe(rename_to_axis, {df.columns[0]: "g"})
            groupStochastic = df.select("g").unique()

    return dict(
        pdt_branch_weight=pdt_branch_weight,
        pd_branch_weight=pd_branch_weight,
        dt_non_anticipativity=dt_non_anticipativity,
        groupStochastic=groupStochastic,
        period_branch_full=period_branch_full,
        period_in_use_set=period_in_use_set,
    )


def _assign_param_names(data: "FlexData") -> "FlexData":
    """Stamp the FlexData attribute name onto every :class:`Param` field.

    Enables :class:`polar_high.WarmProblem`'s Param-tracked auto-update by
    giving each Param a stable logical name (``"p_inflow"`` etc.) that
    flows through the algebra primitives' source-Param metadata.
    Anonymous (``name is None``) Params are not tracked.
    """
    from dataclasses import fields as _dc_fields
    for f in _dc_fields(data):
        v = getattr(data, f.name, None)
        if isinstance(v, Param) and v.name is None:
            v.name = f.name
    return data


# Δ.12c — explicit fixture → (sqlite_filename, scenario_name) overrides
# for fixtures whose workdir basename doesn't follow the
# ``work_<scenario>`` convention (or whose DB lives in a non-default
# sqlite file).  Each entry was validated against the fixture's
# ``_gen_*.py`` script.
_FIND_SCENARIO_OVERRIDES: "dict[str, tuple[str, str]]" = {
    "work_2day_stochastic_dispatch_full_storage":
        ("tests.sqlite", "2_day_stochastic_dispatch"),
    "work_2day_stochastic_dispatch_no_storage":
        ("tests.sqlite", "2_day_stochastic_dispatch_no_storage"),
    "work_commodity_ladder_annual":
        ("tests.sqlite", "coal_ladder_annual"),
    "work_commodity_ladder_cumulative":
        ("tests.sqlite", "coal_ladder_cumulative"),
    "work_dc_power_flow":
        ("case14.sqlite", "dc_opf_test"),
    "work_delay_source_coef":
        ("tests.sqlite", "water_pump_delayed"),
    "work_inflation_check":
        ("tests.sqlite", "wind_battery_invest_lifetime_renew"),
}


def _find_scenario(workdir: Path) -> str | None:
    """Best-effort scenario discovery for ``load_flextool``'s Γ.8.F Step 3
    auto-construction of a :class:`SpineDbReader`.

    Strategy (in order):

    1. Explicit override map (``_FIND_SCENARIO_OVERRIDES``) — fixtures
       whose workdir basename doesn't match the ``work_<scenario>``
       convention.  Each entry maps to ``(sqlite_filename, scenario_name)``.
    2. Strip a leading ``work_`` from the workdir's basename and check
       whether the resulting name appears as a scenario in
       ``<workdir>/tests.sqlite``.  If yes, return it.
    3. Δ.16 — when the workdir's ``input/`` is a symlink, recurse into
       the linked directory's parent (the canonical workdir).  This
       lets the per-sub-solve test pattern
       (``tempdir/{input,output_raw}`` symlinked to a fixture's
       canonical dirs, ``tempdir/solve_data`` symlinked to a sub-solve
       snapshot) auto-resolve to the original fixture's scenario.
    4. Otherwise return ``None`` — the caller should fall back to the
       CSV-only path.

    This covers every fixture in ``tests/engine_polars/data/`` without
    forcing a scenario name into workdirs that don't follow the
    convention.  Production callers that build their own workdirs
    without the ``work_`` prefix will continue to use the CSV-only
    path until they pass an explicit ``db_reader=``.
    """
    # Δ.16 — per-sub-solve test pattern: ``tempdir/input`` is a symlink
    # into the fixture's canonical ``work_<scenario>/input``.  Recurse
    # into that fixture before falling back to the basename heuristic.
    input_link = workdir / "input"
    if input_link.is_symlink():
        try:
            target = (workdir / "input").resolve()
            canonical = target.parent
            if canonical != workdir and canonical.is_dir():
                resolved = _find_scenario(canonical)
                if resolved is not None:
                    return resolved
        except Exception:  # noqa: BLE001 — best-effort
            pass
    # 1. Explicit override map for the seven mismatch fixtures.
    override = _FIND_SCENARIO_OVERRIDES.get(workdir.name)
    if override is not None:
        sqlite_filename, scenario_name = override
        sqlite_path = workdir / sqlite_filename
        if not sqlite_path.exists():
            return None
        # Verify the scenario actually exists in the DB before returning
        # it — this guards against stale fixtures with renamed scenarios.
        try:
            import sqlite3
            with sqlite3.connect(str(sqlite_path)) as con:
                cur = con.cursor()
                cur.execute("SELECT name FROM scenario")
                scenarios = {row[0] for row in cur.fetchall()}
        except Exception:  # noqa: BLE001 — best-effort discovery
            return None
        if scenario_name in scenarios:
            return scenario_name
        return None

    # 2. Default ``work_<scenario>`` convention.
    sqlite_path = workdir / "tests.sqlite"
    if not sqlite_path.exists():
        return None
    candidate = workdir.name
    if candidate.startswith("work_"):
        candidate = candidate[len("work_"):]
    if not candidate:
        return None
    # Cheap probe: list scenarios via spinedb_api.  The probe is best-
    # effort; any failure (DB locked, bad schema, missing scenario
    # table) returns None and we fall back to CSV.
    try:
        import sqlite3
        with sqlite3.connect(str(sqlite_path)) as con:
            cur = con.cursor()
            cur.execute("SELECT name FROM scenario")
            scenarios = {row[0] for row in cur.fetchall()}
    except Exception:  # noqa: BLE001 — best-effort discovery
        return None
    if candidate in scenarios:
        return candidate
    return None


def load_flextool(source: "Path | str | FlexInputSource",
                   *,
                   db_reader: "object | None" = None,
                   handoff: "object | None" = None,
                   provider: "object | None" = None) -> FlexData:
    """Load a :class:`FlexData` from either a workdir on disk or a
    :class:`flextool._input_source.FlexInputSource`.

    Backward-compatible: passing a ``Path`` (today's call style) wraps
    it as a :class:`flextool._input_source.CsvSource` internally and
    behaves identically.

    Γ.1 of the deeper DB-direct migration adds an optional ``db_reader``
    keyword: when supplied, the per-(entity_class, parameter_name)
    :class:`flextool._input_source.InputSource` (typically
    :class:`SpineDbReader`) overrides the chosen first-wave Direct
    Params with frames built directly from the DB.  Every other
    ``FlexData`` field is still loaded via the CSV path; the full
    sweep into ``input.py`` happens in Γ.2/Γ.3.

    Γ.8.F Step 3 — when ``source`` is a workdir-shaped path (Path / str)
    AND ``db_reader`` is not supplied, ``load_flextool`` auto-constructs
    a :class:`SpineDbReader` against ``<workdir>/tests.sqlite`` using
    the scenario name derived from the workdir basename
    (``work_<scenario>`` convention).  When the convention doesn't match
    or ``tests.sqlite`` is absent, the loader falls back to the
    CSV-only path (no override-chain).  Explicit ``db_reader=`` overrides
    the auto-construction.

    See ``audit/db_direct_param_map.md §7.1`` for the migration plan.

    Δ.11 — ``handoff`` (in-memory :class:`SolveHandoff`) overlay
    ------------------------------------------------------------

    When ``handoff`` is supplied, the loader populates the five
    handoff-derived FlexData fields directly during the build (replacing
    the previous post-load :func:`apply_handoff` call):

      * ``p_entity_previously_invested_capacity`` ← ``realized_invest``
        × ``edd_history`` (read from ``solve_data/edd_history.csv``).
      * ``p_entity_invested`` ← ``realized_invest`` summed over period.
      * ``p_entity_divested`` ← ``divest_cumulative``.
      * ``p_roll_continue_state`` ← ``roll_end_state``.
      * ``p_fix_storage_quantity`` ← ``fix_storage_quantity``.

    Construct-with-handoff replaces the old overlay-after-load pattern
    so the in-memory carriers flow into the build as inputs, no separate
    "apply" step.  Snapshot CSV state for these fields is overwritten
    by the in-memory handoff (the in-memory handoff is the source of
    truth, even when the workdir's ``solve_data/*.csv`` already carries
    a value).
    """
    # Late-import the Protocol + adapters to avoid a circular import
    # against the tests' fixture-loaders (which sometimes import this
    # module before flextool.__init__ finishes).
    from flextool.engine_polars._input_source import CsvSource, FlexInputSource, InputSource

    workdir_for_db: Path | None = None
    if isinstance(source, (str, Path)):
        workdir_for_db = Path(source)
        source = CsvSource(source)
    elif not isinstance(source, FlexInputSource):
        raise TypeError(
            f"load_flextool expects Path | str | FlexInputSource, "
            f"got {type(source).__name__}"
        )
    if db_reader is not None and not isinstance(db_reader, InputSource):
        raise TypeError(
            f"load_flextool db_reader must implement InputSource, "
            f"got {type(db_reader).__name__}"
        )
    # Γ.8.F Step 3 — defer SpineDbReader auto-construction until AFTER
    # the axis_enums vocabulary is built so the reader can be threaded
    # with it (cast-on-emit).  ``_auto_construct_db_reader`` records the
    # intent; the actual construction lives below the axis_enums build.
    _auto_construct_db_reader = (db_reader is None and workdir_for_db is not None)

    inp = source.input_dir
    sd  = source.solve_data_dir

    # Step 2.5 Phase B — every ``_provider_read`` call now requires the
    # Provider.  When the caller did not pass one (legacy workdir-only
    # entry point used by the loader-level tests in
    # ``tests/engine_polars/loaders/``), seed an ephemeral Provider from
    # the workdir's ``input/`` and ``solve_data/`` directories so the
    # cascade keeps reading from memory.  Cascade-level entry points
    # (``input_derivation.run`` → ``load_flextool``) always pass an
    # explicit Provider, so this branch is a no-op there.
    if provider is None:
        from flextool.engine_polars._flex_data_provider import (
            FlexDataProvider,
        )
        from flextool.engine_polars._input_source import (
            seed_provider_from_dir,
        )
        provider = FlexDataProvider()
        if Path(inp).exists():
            seed_provider_from_dir(provider, inp, "input")
        if Path(sd).exists():
            seed_provider_from_dir(provider, sd, "solve_data")

    from flextool.engine_polars._orchestration import get_phase_recorder
    _entry_rec = get_phase_recorder()
    _entry_logger = logging.getLogger("flextool.engine_polars.input")
    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_provider_seeded",
            _entry_logger,
            user_label="load_flextool: provider seeded",
        )

    # Δ.12a — build the per-solve in-memory state (typed fields +
    # process-level CSV cache) up front, then activate the cache so
    # every ``_read_csv_file`` call inside the loader (``_load_time``,
    # ``_load_node``, the helpers' ``load_data``, the apply_derived_*
    # passes) deduplicates by absolute path.  Falls back to ``None``
    # when no workdir is known (handoff-only callers).
    ctx = None
    ctx_workdir = workdir_for_db
    if ctx_workdir is None:
        ctx_workdir = (Path(source.work_folder)
                          if hasattr(source, "work_folder")
                          else None)
    if ctx_workdir is None and hasattr(source, "input_dir"):
        try:
            ctx_workdir = Path(source.input_dir).parent
        except Exception:  # pragma: no cover — defensive
            ctx_workdir = None
    if ctx_workdir is not None:
        from flextool.engine_polars._solve_context import SolveContext
        try:
            ctx = SolveContext.from_workdir(ctx_workdir, provider=provider)
        except Exception:  # pragma: no cover — defensive
            ctx = None
    if ctx is not None:
        ctx.activate()

    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_solve_context_ready",
            _entry_logger,
            user_label="load_flextool: SolveContext ready",
        )

    # ── Phase 4 — axis enum vocabulary activation ────────────────────
    # Production path (input_derivation.run) populates
    # ``provider.axis_enums`` + ``provider.contract`` up-front.  The
    # workdir-only path (loader-level tests, bare ``load_flextool``)
    # lazily builds them against the workdir sqlite below.  When neither
    # works we keep activation OFF (axis_enums = None), preserving
    # pre-Phase-4 behaviour for tests that have no DB at all.
    #
    # See ``specs/enum_dtype_refactor_plan.md §Phase 4``.
    axis_enums: "dict[str, pl.Enum] | None" = getattr(
        provider, "axis_enums", None,
    )
    contract = getattr(provider, "contract", None)
    if axis_enums is None:
        from flextool.spinedb_backend._axis_enums import (
            build_axis_enums,
            load_axis_contract,
        )
        sqlite_for_backend: Path | None = None
        if workdir_for_db is not None:
            db_workdir = workdir_for_db
            input_link = workdir_for_db / "input"
            if input_link.is_symlink():
                try:
                    db_workdir = input_link.resolve().parent
                except Exception:  # noqa: BLE001
                    db_workdir = workdir_for_db
            override = _FIND_SCENARIO_OVERRIDES.get(db_workdir.name)
            sqlite_filename = (
                override[0] if override is not None else "tests.sqlite"
            )
            sp = db_workdir / sqlite_filename
            if sp.exists():
                sqlite_for_backend = sp
        if sqlite_for_backend is not None:
            try:
                from flextool.spinedb_backend import SpineDBBackend
                contract = load_axis_contract()
                with SpineDBBackend(
                    f"sqlite:///{sqlite_for_backend}",
                    None,
                ) as _ab:
                    axis_enums = build_axis_enums(_ab, contract)
                if provider is not None:
                    provider.axis_enums = axis_enums
                    provider.contract = contract
            except Exception:  # noqa: BLE001
                axis_enums = None
                contract = None

    if _entry_rec is not None:
        _entry_rec.checkpoint(
            "load_flextool_axis_enums_done",
            _entry_logger,
            user_label="load_flextool: axis_enums built",
        )

    # Auto-construct a SpineDbReader for the workdir-only entry path
    # (no explicit ``db_reader=``).  Phase 4.6: thread ``axis_enums`` +
    # ``contract`` so cast-on-emit at the SpineDbReader boundary
    # delivers Enum-typed frames to the cascade — paired with the
    # alias sweep + cross-Enum compare fixes, downstream joins compose
    # without SchemaError.
    if _auto_construct_db_reader:
        scenario = _find_scenario(workdir_for_db)
        if scenario is not None:
            from flextool.engine_polars._spinedb_reader import SpineDbReader
            db_workdir = workdir_for_db
            input_link = workdir_for_db / "input"
            if input_link.is_symlink():
                try:
                    db_workdir = input_link.resolve().parent
                except Exception:  # noqa: BLE001
                    db_workdir = workdir_for_db
            override = _FIND_SCENARIO_OVERRIDES.get(db_workdir.name)
            sqlite_filename = (
                override[0] if override is not None else "tests.sqlite"
            )
            sqlite_path = db_workdir / sqlite_filename
            try:
                db_reader = SpineDbReader(
                    f"sqlite:///{sqlite_path}", scenario=scenario,
                    axis_enums=axis_enums, contract=contract,
                )
            except Exception:  # noqa: BLE001 — best-effort auto-construction
                db_reader = None
            if _entry_rec is not None:
                _entry_rec.checkpoint(
                    "load_flextool_spinedb_reader_done",
                    _entry_logger,
                    user_label="load_flextool: SpineDbReader constructed",
                )

    # Phase 4.6 — flip the cascade-wide global on.  Every cascade module
    # that uses ``rename_to_axis`` / ``alias_to_axis`` / ``lit_axis`` /
    # ``schema_dtype`` / ``cast_dim`` picks up the Enum dtypes for the
    # duration of this load.  The ``finally`` block at the end of the
    # try below resets to ``None`` so concurrent / nested loads see a
    # clean slate.
    #
    # The cross-axis alias sweep landed in clusters 4.0-4.5b + the
    # 4.6 fix-up commit (canonical self-references + ``time``/``d_h``
    # synonyms); the cross-Enum value comparisons that the activation
    # would otherwise expose were fixed in this dispatch's Step 2.
    if axis_enums is not None:
        set_global_axis_enums(axis_enums)

    try:
        # Reuse the recorder bound at the function entry above so the
        # sub-checkpoints (axis_enums / SpineDbReader / load_flextool
        # start) share a single timeline.
        _rec = _entry_rec
        _load_logger = _entry_logger

        def _load_mem(label: str, user_label: str) -> None:
            if _rec is not None:
                _rec.checkpoint(label, _load_logger, user_label=user_label)

        _load_mem("load_flextool_start", "load_flextool start")

        # Δ.2: build the per-solve BlockLayout once from flextool's
        # solve_data/ block CSVs (still produced by flextool's
        # ``write_block_data_for_solve``).  Downstream block-aware helpers
        # consume the in-memory frames instead of re-reading the same CSVs
        # at each call site.  When the orchestrator transitions to building
        # ``BlockLayout`` natively (Δ.3+), this load_from_solve_data call
        # becomes a no-op or is replaced by passing the live layout in.
        block_layout = BlockLayout.load_from_solve_data(sd, provider=provider)

        dt, step_dur, rp_cw, infl, psh = _load_time(sd, provider=provider)
        nb, nb_dt, inflow, pen_up, pen_dn = _load_node(sd, dt, provider=provider)
        _load_mem("load_node_end", "load_flextool: time + node loaded")

        proc = _load_process_topology(inp, sd, dt, block_layout=block_layout,
                                       source=db_reader,
                                       provider=provider)
        _load_mem("load_process_topology_end",
                  "load_flextool: process topology loaded")

        # base_cap_pd = (p, d, base) for profile RHS — recompute here; small.
        base_cap_pd = None
        p_flow_upper_existing = None
        pd_neg_cap = None
        all_entity_unitsize_param = None
        if proc["pss"] is not None:
            cap_long = _read_capacity(sd / "p_entity_period_existing_capacity.csv",
                                       sd / "p_entity_previously_invested_capacity.csv",
                                       sd / "p_entity_all_existing.csv",
                                       provider=provider)
            unitsize_long = _read_unitsize(_provider_pick(
                provider,
                ("solve_data/p_entity_unitsize", sd / "p_entity_unitsize.csv"),
                ("input/p_entity_unitsize", inp / "p_entity_unitsize.csv"),
            ) or (inp / "p_entity_unitsize.csv"), provider=provider)
            # p_all_entity_unitsize: unfiltered — covers processes, connections AND nodes.
            # Used by the scaling analyzer to compute the full entity-unitsize spread.
            if unitsize_long.height > 0:
                all_entity_unitsize_param = Param(
                    ("e",),
                    unitsize_long.pipe(rename_to_axis, {"e": "e"}).select("e", "value"),
                )
            cap_us_pd = (cap_long.pipe(rename_to_axis, {"e":"p","value":"cap"})
                .filter(pl.col("p").is_in(proc["pss"]["p"].unique()))
                .join(unitsize_long.pipe(rename_to_axis, {"e":"p","value":"us"}), on="p", how="inner"))
            base_cap_pd = (cap_us_pd
                .with_columns(base=pl.col("cap")/pl.col("us"))
                .select("p","d","base"))
            # pd_neg_cap = (p, d) where both existing and unitsize are negative.
            # In the .mod, maxToSink is ``v_flow * unitsize ≤ existing × ...``.
            # When both are negative (e.g. anti_energy_plant: us=-50, existing=-50)
            # dividing by unitsize FLIPS the inequality direction, yielding
            # ``v_flow ≥ existing/unitsize`` (a forced *minimum* output).
            # We therefore route these (p, d) rows out of the standard ``≤``
            # maxToSink and into a sign-flipped ``≥`` companion constraint.
            neg_pd = cap_us_pd.filter(
                (pl.col("cap") < 0.0) & (pl.col("us") < 0.0)
            ).select("p", "d")
            if neg_pd.height > 0:
                pd_neg_cap = neg_pd
            # p_flow_upper_existing = (existing/unitsize) per (p, source, sink, d).
            # This is the *true* structural existing-capacity upper bound on
            # v_flow.  It corresponds to the .mod's RHS without invest/divest
            # (assuming cap_coef=1).  flextool's preprocessed p_flow_max may
            # bake in max_invest_cum (for invest-method = invest_no_limit) and
            # is therefore looser; using p_flow_upper_existing + the explicit
            # invest tightening on the LHS gives the tight constraint.
            p_flow_upper_existing = Param(("p", "source", "sink", "d"),
                base_cap_pd.rename({"base": "value"})
                           .join(proc["pss"], on="p", how="inner")
                           .select("p", "source", "sink", "d", "value"))

        flow_co2_p, flow_co2_p_noEff, co2c, co2pr = _load_co2_price(
            inp, sd, proc["pss_eff"], proc.get("pss_noEff"),
            provider=provider)
        g_co2_max, flow_co2_cap, flow_co2_cap_noEff, co2_max_p, g_d_capped = _load_co2_cap(
            inp, sd, proc["pss_eff"], dt, pss_noEff=proc.get("pss_noEff"),
            provider=provider)
        (g_co2_max_total, flow_co2_cap_total, flow_co2_cap_total_noEff,
         co2_max_total_p) = _load_co2_cap_total(
            inp, sd, proc["pss_eff"], pss_noEff=proc.get("pss_noEff"),
            provider=provider)
        if (co2_max_p is not None or co2_max_total_p is not None) and co2c is None:
            p_comm = _provider_read(provider, "input/p_commodity",
                                      inp / "p_commodity.csv")
            co2c = Param(("c",),
                p_comm.filter(pl.col("commodityParam")=="co2_content")
                      .pipe(rename_to_axis, {"commodity":"c","p_commodity":"value"})
                      .select("c","value"))

        (indir_set, indir_in, indir_out, indir_dt,
         p_source_flow_coef, p_sink_flow_coef) = _load_indirect(
             sd, proc["pss"], dt, inp, provider=provider)
        (fc_idx, fc_coef, c_const, cdt_eq, cdt_le, cdt_ge,
         n_inv_coef, p_inv_coef,
         n_state_coef, n_prebuilt_coef, p_prebuilt_coef,
         _) = _load_user_constraints(inp, proc["pss"], dt, provider=provider)

        p_up, p_lo, p_fx, prof_v, exist_cnt, avail = _load_profiles(
            inp, sd, proc["pss"], proc["unitsize"], base_cap_pd,
            provider=provider)
        # existing_count is also needed by the online/UC feature even when
        # no profile features are active; fall back to base_cap_pd directly.
        if exist_cnt is None and base_cap_pd is not None:
            exist_cnt = Param(("p", "d"), base_cap_pd.rename({"base": "value"}))
        # availability: default to 1.0 from preprocessing — also used by UC
        # capacity bounds; if loader didn't populate (no profile data), try
        # to read pdtProcess_availability.csv standalone.
        if avail is None and proc["pss"] is not None:
            avail_long = _slice_param(sd / "pdtProcess.csv", "process", "availability",
                                       rename_entity_to="p",
                                       provider=provider)
            if avail_long is not None:
                avail = Param(("p","d","t"), avail_long)

        # dtttdt is needed by both storage and online features — always load
        # it when present (preprocessing always emits it for non-trivial
        # solves).  p_process_existing_count (= existing/unitsize per (p, d))
        # is needed by online + profile features — always load when processes
        # exist.
        dtttdt = _read_step_previous(sd / "step_previous.csv", provider=provider)

        online = _load_online(inp, sd, dt, proc["pss"], source=db_reader,
                              provider=provider)
        ramp = _load_ramp(inp, sd, proc["pss"], provider=provider)
        invest = _load_invest(sd, dt, inp, proc["pss"], db_reader=db_reader,
                              provider=provider)
        varcost = _load_varcost(sd, proc["pss"], provider=provider)
        _load_mem("load_varcost_end", "load_flextool: varcost loaded")
        fixed_cost = _load_fixed_cost(sd, provider=provider)
        capacity_for_scaling = _load_node_capacity_for_scaling(sd, nb,
                                                                provider=provider)

        # ─── Storage (nodeState + binding methods + dtttdt + node-balance source-side flows)
        storage = _load_storage(inp, sd, dt, nb,
                                 proc["pss_eff"], proc["pss_noEff"],
                                 base_cap_pd, proc["unitsize"],
                                 block_layout=block_layout,
                                 provider=provider)
        # _load_storage emits its own dtttdt; if storage is inactive it'll be
        # None there but we want the top-level read.
        if storage["dtttdt"] is None:
            storage["dtttdt"] = dtttdt

        # ─── Per-arc block step durations (reserved for future use) ──────────
        p_arc_step_duration_sink = None
        p_arc_step_duration_source = None

        # ─── Per-arc-side daily-block aggregation index ──────────────────────
        # Δ.17b Gap A: produced by ``apply_derived_e`` (which calls
        # ``arc_block_dt_from_source``) — see ``_derived_params.py:6049+``.
        # The local seed had been redundant since Γ.3.E but remained the
        # sole producer because of a typo (``getattr(flex_data, "pss", ...)``
        # — actual attribute is ``process_source_sink``).  Δ.17b fixed the
        # typo; seeds dropped here.
        arc_sink_block_dt = None
        arc_source_block_dt = None
        p_arc_sink_weight = None
        p_arc_source_weight = None

        # ─── Group-level slack (capacity_margin / inertia / non_sync) ────────
        group_slack = _group_slack.load_data(
            inp=inp, sd=sd, dt=dt,
            nb=nb,
            pss_eff=proc["pss_eff"],
            pss_noEff=proc["pss_noEff"],
            p_unitsize=proc["unitsize"],
            provider=provider,
        )

        # ─── Reserves (timeseries / dynamic / n_1 / per-process upper) ────────
        reserve_data = _reserve.load_data(inp=inp, sd=sd, dt=dt,
                                            provider=provider)
        # ``group_node`` is shared between _group_slack and _reserve (both
        # populate it from the canonical solve_data/group_node.csv).  Drop the
        # reserve copy to avoid duplicate-kwargs at the FlexData(...) call when
        # group_slack already provided it; reserve will read it back off d in
        # add_constraints.  If group_slack didn't populate it, hand the
        # reserve copy through.
        if "group_node" in reserve_data and group_slack.get("group_node") is not None:
            reserve_data = {k: v for k, v in reserve_data.items() if k != "group_node"}

        # ─── Cumulative / group-invest / min-invest data ─────────────────────
        # The module's ``load_data`` is a no-op stub; ``flextool/input.py`` is
        # the canonical loader.  Call it for symmetry, then populate the new
        # ``FlexData`` fields from the canonical helper below.
        _cumulative_invest.load_data(inp=inp, sd=sd, dt=dt)
        ci_data = _load_cumulative_invest(inp=inp, sd=sd, dt=dt, provider=provider)

        # ─── Delayed processes / DR data ─────────────────────────────────────
        delay_data = _delay.load_data(inp_dir=inp, sd_dir=sd,
                                         provider=provider)

        # ─── DC power flow data ──────────────────────────────────────────────
        # Step 2.5-F Phase B: Provider holds the four DC PF frames
        # under input/<key>; disk arm is reserved for off-cascade
        # fixture loaders without a Provider.
        dc_pf_data = _dc_power_flow.load_data(inp_dir=inp, provider=provider)

        # ─── Commodity price ladder data ─────────────────────────────────────
        # Step 2.5-F Phases D + E: Provider carries the two ladder
        # ``input/commodity_ladder_*`` frames; disk arm reserved for
        # off-cascade fixture loaders.
        ladder_data = _commodity_ladder.load_data(
            inp_dir=inp, sd_dir=sd, provider=provider,
        )

        # ─── Multi-branch stochastic data (A6) ───────────────────────────────
        stoch_data = _load_stochastics(inp=inp, sd=sd, dt=dt, provider=provider)
        _load_mem("load_stochastics_end",
                  "load_flextool: stochastics + remaining CSV loaders done")

        flex_data = FlexData(
            dt = dt,
            p_step_duration = step_dur,
            p_rp_cost_weight = rp_cw,
            p_inflation_op = infl,
            p_period_share = psh,

            nodeBalance = nb,
            nodeBalance_dt = nb_dt,
            p_inflow = inflow,
            p_penalty_up = pen_up,
            p_penalty_down = pen_dn,

            process_source_sink       = proc["pss"],
            process_source_sink_eff   = proc["pss_eff"],
            process_source_sink_noEff = proc["pss_noEff"],
            pss_dt                    = proc["pss_dt"],
            flow_to_n                 = proc["flow_to_n"],
            flow_from_n               = proc["flow_from_n"],
            flow_from_commodity_eff   = proc["flow_from_commodity_eff"],
            flow_from_commodity_noEff = proc["flow_from_commodity_noEff"],
            flow_to_commodity         = proc.get("flow_to_commodity"),
            process_source_canonical  = proc.get("pss_source_canonical"),
            process_sink_canonical    = proc.get("pss_sink_canonical"),
            p_unitsize                = proc["unitsize"],
            p_all_entity_unitsize     = all_entity_unitsize_param,
            p_flow_upper              = proc["flow_upper"],
            p_flow_upper_existing     = p_flow_upper_existing,
            p_slope                   = proc["slope"],
            p_commodity_price         = proc["commodity_price"],
            pd_neg_cap                = pd_neg_cap,

            flow_from_co2_priced = flow_co2_p,
            flow_from_co2_priced_noEff = flow_co2_p_noEff,
            p_co2_content = co2c,
            p_co2_price = co2pr,

            group_co2_max_period = g_co2_max,
            flow_from_co2_capped = flow_co2_cap,
            flow_from_co2_capped_noEff = flow_co2_cap_noEff,
            p_co2_max_period = co2_max_p,
            group_d_co2_capped = g_d_capped,

            group_co2_max_total = g_co2_max_total,
            flow_from_co2_capped_total = flow_co2_cap_total,
            flow_from_co2_capped_total_noEff = flow_co2_cap_total_noEff,
            p_co2_max_total = co2_max_total_p,

            process_indirect = indir_set,
            process_input_flows = indir_in,
            process_output_flows = indir_out,
            process_indirect_dt = indir_dt,
            p_process_source_conversion_flow_coeff = p_source_flow_coef,
            p_process_sink_conversion_flow_coeff = p_sink_flow_coef,

            flow_constraint_idx = fc_idx,
            p_flow_constraint_coef = fc_coef,
            p_constraint_constant = c_const,
            cdt_eq = cdt_eq,
            cdt_le = cdt_le,
            cdt_ge = cdt_ge,
            p_node_constraint_invested_capacity_coeff = n_inv_coef,
            p_process_constraint_invested_capacity_coeff = p_inv_coef,
            p_node_constraint_state_coeff = n_state_coef,
            p_node_constraint_prebuilt_capacity_coeff = n_prebuilt_coef,
            p_process_constraint_prebuilt_capacity_coeff = p_prebuilt_coef,

            process_profile_upper = p_up,
            process_profile_lower = p_lo,
            process_profile_fixed = p_fx,
            p_profile_value = prof_v,
            p_process_existing_count = exist_cnt,
            p_process_availability = avail,

            **online,
            **ramp,
            **invest,
            **storage,
            **varcost,
            **fixed_cost,
            **capacity_for_scaling,
            **group_slack,
            **reserve_data,
            **ci_data,
            **delay_data,
            **dc_pf_data,
            **ladder_data,
            **stoch_data,
            p_arc_step_duration_sink = p_arc_step_duration_sink,
            p_arc_step_duration_source = p_arc_step_duration_source,
            arc_sink_block_dt = arc_sink_block_dt,
            arc_source_block_dt = arc_source_block_dt,
            p_arc_sink_weight = p_arc_sink_weight,
            p_arc_source_weight = p_arc_source_weight,
            solver_options = _load_solver_options(sd, provider=provider),
            # Phase 2 multi-block fast-path: stash the per-solve BlockLayout
            # built above on the FlexData so the override chain helpers
            # (period_block_family_from_source, nodeStateBlock_from_source,
            # arc_block_dt_from_source, load_block_bundle) can consume the
            # in-memory frames instead of re-reading solve_data/ CSVs on the
            # fast path.  On the slow path this is the same layout that
            # was just used to load process topology a few lines above.
            block_layout = block_layout,
        )

        # Gap F final — handoff-path auxiliaries: surface the three CSVs
        # that ``build_handoff_from_solution`` would otherwise re-read.
        # Each is lenient: file missing or empty → field stays ``None``
        # and the handoff extractor's disk fallback kicks in.
        #
        # Stochastic / output_horizon: prefer ``dt_realize_dispatch_set.csv``
        # when present.  That's the canonical "rows to emit" set built by
        # ``_emit_per_solve.write_period_set_csvs`` — it includes all
        # forecast-branch (period, step) pairs for stochastic scenarios
        # (where ``realized_dispatch.csv`` is anchor-only by design).  For
        # non-stochastic / non-output_horizon solves it collapses to the
        # same rows as ``realized_dispatch.csv`` (verified on fullYear_roll).
        # Without this preference dispatch-time CSVs (costs__dt, node__dt,
        # …) drop branch rows silently because the parquet extractor's
        # canonical row order comes from this frame's downstream
        # ``dt_realize_dispatch`` MultiIndex.
        flex_data.realized_dispatch = _load_handoff_aux_pair(
            sd / "dt_realize_dispatch_set.csv", ("period", "time"),
            provider=provider)
        if flex_data.realized_dispatch is not None:
            flex_data.realized_dispatch = flex_data.realized_dispatch.pipe(
                rename_to_axis, {"time": "step"})
        if flex_data.realized_dispatch is None:
            flex_data.realized_dispatch = _load_handoff_aux_pair(
                sd / "realized_dispatch.csv", ("period", "step"),
                provider=provider)
        flex_data.period__time_last = _load_handoff_aux_pair(
            sd / "period__time_last.csv", ("period", "step"),
            provider=provider)
        # ``node__storage_nested_fix_method`` lives in solve_data/ for
        # cascade solves; fall back to input/ if not yet copied.  Explicit
        # ``is None`` chain (DataFrame is non-truthy in polars).
        nsfm = _load_handoff_aux_pair(
            sd / "node__storage_nested_fix_method.csv", ("node", "method"),
            provider=provider)
        if nsfm is None:
            nsfm = _load_handoff_aux_pair(
                inp / "node__storage_nested_fix_method.csv", ("node", "method"),
                provider=provider)
        flex_data.node__storage_nested_fix_method = nsfm

        # Δ.3/Δ.4 — DB-direct construction.  Replaces the previous 3-pass
        # override layering (CSV → first_wave_overrides → projection_overrides
        # → derived_overrides_a..g).  Δ.3 collapsed the dict-overlay
        # round-trip into linear ``apply_*`` mutations; Δ.4 deleted the nine
        # deprecated wrapper aliases.  Each FlexData field that has a
        # DB-direct helper is now built by exactly one helper that mutates
        # ``flex_data`` directly — no dict-overlay round-trip, no "override"
        # semantics.  See progress.md (Δ.3 / Δ.4 close stanzas).
        #
        # NOTE: the CSV path above still populates every field as the
        # initial seed.  In Δ.5+, when every FlexData field has a DB-direct
        # helper, the CSV path will retire and these apply_* calls become
        # the primary loader.
        # Δ.12a — ctx (with the process-level CSV-read cache activated)
        # was constructed at the top of ``load_flextool`` so the caching
        # benefit reaches every ``_read_csv_file`` call inside the loader
        # (``_load_*`` family + helpers' ``load_data`` + apply_derived_*).
        # ``deactivate`` happens in the outer ``finally`` block.
        # We deliberately DEFER the FlexData → Enum sweep until AFTER
        # ``_apply_db_overrides`` (the derived-cascade).  The cascade
        # builds many scratch frames with hard-coded ``pl.Utf8`` dim
        # column schemas in ``_derived_params.py`` /
        # ``_derived_block.py`` / etc., and joining Enum-cast FlexData
        # fields against String-dtype scratch frames raises
        # ``SchemaError``.  Casting after the cascade keeps the entire
        # CSV-read + cascade pipeline in String land and converts to
        # Enum once at the end — Var construction in ``model.py`` and
        # the model-build cross-joins then operate on Enum-typed
        # frames.
        if db_reader is not None:
            _apply_db_overrides(flex_data, db_reader, source, ctx=ctx,
                                 provider=provider)

        # Δ.11 — overlay in-memory handoff carriers onto the FlexData
        # built so far.  Replaces the previous post-load ``apply_handoff``
        # call: the handoff is now an input to the build, not a separate
        # overlay step.  ``solve_data_dir`` (if known) is consulted only for
        # ``edd_history.csv`` — used by the
        # ``p_entity_previously_invested_capacity`` derivation.  Cluster B
        # chained-handoff state (``p_entity_all_existing``) is rebuilt from
        # the now-populated carriers via :func:`apply_existing_chain` (called
        # below — db_reader is required for that path).
        if handoff is not None:
            sd_dir = workdir_for_db / "solve_data" if workdir_for_db is not None else None
            flex_data = _overlay_handoff(flex_data, handoff, sd_dir, ctx=ctx,
                                            provider=provider)
            # Re-apply the cluster B chained-existing helper so
            # ``p_entity_all_existing`` reflects the in-memory handoff
            # carriers (rather than the workdir's pre-handoff CSV value).
            if db_reader is not None and workdir_for_db is not None:
                from flextool.engine_polars import _derived_existing as _ex
                _ex.apply_existing_chain(flex_data, db_reader, workdir_for_db,
                                              handoff=handoff, ctx=ctx,
                                              provider=provider)

        # Phase 4.6 — end-of-load FlexData → Enum sweep.  With the global
        # flipped on above, the cascade emits Enum-typed dim columns
        # throughout, but ad-hoc CSV-fallback paths inside the cascade
        # can still produce Utf8 dim columns.  Sweep the FlexData
        # container once before return so every Param / DataFrame /
        # LazyFrame field has its dim columns cast against the canonical
        # axis enums.  Idempotent: columns already in the correct Enum
        # dtype are skipped by ``cast_frame_axes``.
        if axis_enums is not None:
            cast_flexdata_axes(flex_data, axis_enums)

        # Stash the resolved axis_enums on the returned FlexData so
        # downstream consumers (``build_flextool`` and the dumpers that
        # already read ``getattr(data, "_axis_enums", None)``) can rebind
        # the global ContextVar — substrate helpers ``cast_dim`` /
        # ``schema_dtype`` / ``lit_axis`` / ``alias_to_axis`` reach for
        # the live ContextVar when called with ``enums=None``, and
        # without this stash the ContextVar is reset before
        # ``build_flextool`` runs (see _delay.build_indirect_delayed_in_flow
        # which calls ``cast_dim(..., None, ...)``).
        flex_data._axis_enums = axis_enums

        result = _assign_param_names(flex_data)
        load_ok = True  # noqa: F841 — consumed by finally clause below
        return result
    finally:
        if ctx is not None:
            ctx.deactivate()
        # On the success path, leave the global ContextVar set to the
        # cascade's axis_enums — ``build_flextool`` runs AFTER
        # ``load_flextool`` returns and still needs the live vocabulary
        # for ``cast_dim(..., None, "e")`` call sites in _delay /
        # downstream model builders.  On cascade error reset so a
        # half-built state doesn't leak into the next cascade.  When
        # activation is off (``axis_enums is None``) the reset is a no-op
        # because ``set_global_axis_enums(None)`` matches the default.
        if not locals().get("load_ok", False):
            set_global_axis_enums(None)
        else:
            set_global_axis_enums(axis_enums)


def _apply_db_overrides(flex_data: "FlexData", db_reader: "InputSource",
                          source: "object",
                          ctx: "object | None" = None,
                          *,
                          provider: "object | None" = None) -> None:
    """Apply the DB-direct construction passes to ``flex_data`` in
    place.  Δ.3 consolidates the previous 9-wrapper override chain into
    a single linear sequence; each ``apply_*`` callee mutates
    ``flex_data`` directly via setattr (no dict round-trip).

    Pass order (preserved from the legacy chain — each pass may depend
    on fields written by earlier passes):
      1. Direct Params — scalar + relationship 1d_map.
      2. Projection Params — entity-instance + reserve-method partitions.
      3. Derived A — dt / step duration / weighting / inflow / profiles.
      4. Derived B — process topology + reclassified method-derived.
      5. Derived C — invest/divest + online/UC + group slack.
      6. Derived D — p_entity_all_existing, node_reference_angle, etc.
      7. Derived E — storage block algebra.
      8. Derived F — lifetime cascade + handoff state + full inflation.
      9. Derived G — commodity ladder, reserves, delay, multi-branch.

    Δ.12a — when *ctx* is supplied, helpers may consume the typed
    in-memory state (``ctx.solve_name``, ``ctx.realized_periods``, …)
    and the cached :meth:`SolveContext.read_csv` instead of issuing
    fresh ``_read_csv_file`` calls per invocation.  The ctx is built by
    :func:`load_flextool` from the workdir before this function runs;
    helpers that don't yet consume it fall through to their pre-Δ.12a
    direct-CSV path with no behavioural change.
    """
    from flextool.engine_polars import _direct_params as _dp
    from flextool.engine_polars import _projection_params as _pp
    from flextool.engine_polars import _derived_params as _drv

    from flextool.engine_polars._orchestration import get_phase_recorder
    _rec = get_phase_recorder()
    _logger = logging.getLogger("flextool.engine_polars.input")

    def _timed(label, fn, *args, **kwargs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if _rec is not None:
            # Emit via the phase recorder so the line carries RSS +
            # Δrss + Δpeak (when full diagnostics is on) in the same
            # format as the cascade/build checkpoints.
            _rec.checkpoint(
                f"input_pass_{label.split()[0]}",
                _logger,
                user_label=f"input pass {label}",
            )
        else:
            # Fallback: legacy plain timer line (used by unit tests
            # that don't go through run_orchestration).
            print(f"  input pass {label}: {elapsed:.3f}s")

    # Pass 1a-2: source-only Params (no workdir needed).
    # Δ.28 — pass 1 splits into 1a (dt-independent) and 1b (dt-dependent).
    # Pass 1b runs after ``apply_derived_a`` populates ``flex_data.dt``
    # so scalar / 1d_map[period] / 1d_map[time] values authored on the
    # source can broadcast across the active solve's (d, t) axis.  See
    # ``_direct_params.apply_direct_params_a/b`` docstrings for the
    # full Δ.28 rationale.
    _timed("1a direct_params_a", _dp.apply_direct_params_a, db_reader, flex_data)
    _timed("2  projection_params", _pp.apply_projection_params, db_reader, flex_data)

    # Pass 3-9: workdir-aware Params.  Resolve the workdir from the
    # source object (CsvSource exposes ``input_dir.parent``).
    try:
        workdir = source.workdir if hasattr(source, "workdir") \
                   else source.input_dir.parent
    except Exception:  # pragma: no cover — defensive
        workdir = None
    if workdir is None:
        # Δ.28 — no workdir means apply_derived_a won't run here.  In the
        # slow path ``flex_data.dt`` was populated by ``_load_*`` from
        # CSV before ``_apply_db_overrides``; run pass 1b with that dt.
        # In the fast path ``workdir`` is always non-None
        # (``_SourceShim`` carries ``work_folder``).
        _timed("1b direct_params_b", _dp.apply_direct_params_b, db_reader, flex_data)
        return
    workdir_path = Path(workdir)

    # Δ.18 — synthetic per-sub-solve detection.  When the workdir's
    # ``solve_data/solve_current.csv`` names a solve that doesn't exist
    # in Spine (e.g. nested-multi-invest fixtures whose orchestrator emits
    # per-period sub-solve names like ``invest_5weeks_p2020`` that are
    # synthesized at runtime, not declared in the data DB), the per-solve
    # override chain (``apply_derived_a..g`` + ``apply_existing_chain``)
    # would return None for every helper that takes ``active_solve`` as
    # a key — wiping out the snapshot CSV seeds those helpers would have
    # otherwise overlaid.  For these synthetic-solve workdirs, the
    # snapshot CSV is the canonical and the per-solve overrides must be
    # skipped entirely.  Direct + Projection Params (passes 1-2) are
    # solve-agnostic and remain authoritative.
    #
    # Δ.19 — for the synthetic ``<base>_<anchor>`` shape we additionally
    # produce the 8 invest-set frames from ``<base>``'s Spine entries
    # filtered to the anchor's period subset (see
    # :func:`_derived_params._resolve_synthetic_solve` and
    # :func:`_derived_params.apply_synthetic_invest_sets`).  This cuts the
    # 8 invest-set disk reads in ``_invest_seeds.py``.  The cost cascade
    # (``apply_derived_f`` NPV / fixed-cost) and per-period caps remain
    # on the CSV-seed path because those bake in multi-year discounting
    # that the per-sub-solve filter doesn't compose cleanly with —
    # deferred to a future dispatch.
    active_solve = _drv._read_active_solve(workdir_path, provider=provider)
    if active_solve is not None and not _drv._solve_in_spine(db_reader,
                                                                  active_solve):
        synth = _drv._resolve_synthetic_solve(db_reader, active_solve)
        if synth is not None:
            _timed("c.synth invest_sets",
                   _drv.apply_synthetic_invest_sets,
                   flex_data, db_reader, active_solve, synth, workdir_path,
                   provider=provider)
        # Δ.28 — synthetic solve: apply_derived_a is skipped, but in the
        # slow path ``flex_data.dt`` is already populated from CSV (see
        # ``_load_time``).  Run pass 1b so broadcast-needing Direct
        # Params still apply against the CSV-loaded dt.
        _timed("1b direct_params_b", _dp.apply_direct_params_b, db_reader, flex_data)

        # RESERVE-1 — the synthetic-solve early-return skips passes 3-10
        # entirely, but two reserve-feature fields are produced by those
        # passes and are required by ``_reserve.add_variables`` /
        # ``add_constraints`` when ``reserve_upDown_group`` is non-empty
        # (``has_feature(d)`` activates the subsystem from the CSV-seeded
        # ``reserve_upDown_group``).  Both producers are solve-agnostic
        # (no ``active_solve`` filter), so calling them on the synthetic
        # path is safe and mirrors what ``apply_derived_d`` / ``apply_derived_g``
        # do on the non-synthetic path.  See
        # ``specs/model_bugs.md::RESERVE-1`` for the full diagnosis.
        def _wire_reserve_for_synthetic_solve():
            flex_data.process_reserve_upDown_node_active = (
                _drv.process_reserve_upDown_node_active_from_source(db_reader))
            flex_data.prundt = _drv.prundt_from_source(
                db_reader, active_solve, getattr(flex_data, "dt", None))
        _timed("c.synth reserve", _wire_reserve_for_synthetic_solve)
        return

    _timed("3  derived_a", _drv.apply_derived_a, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)

    # Δ.28 — pass 1b: broadcast-needing Direct Params now have
    # ``flex_data.dt`` populated by ``apply_derived_a`` step 1.  The
    # broadcast helpers (``broadcast_to_period_time`` /
    # ``broadcast_to_period`` / ``_entity_period_scalar`` /
    # ``_entity_period_time_param``) require non-empty ``period_filter``
    # to fan scalar/1d_map values across the (d, t) axis; running this
    # pass before ``apply_derived_a`` (the legacy ordering) left
    # ``p_commodity_price`` / ``p_process_availability`` / similar
    # fields empty on the fast path even when Spine carried the data.
    _timed("1b direct_params_b", _dp.apply_direct_params_b, db_reader, flex_data)

    _timed("4  derived_b", _drv.apply_derived_b, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _timed("5  derived_c", _drv.apply_derived_c, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _timed("6  derived_d", _drv.apply_derived_d, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _timed("7  derived_e", _drv.apply_derived_e, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _timed("8  derived_f", _drv.apply_derived_f, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)
    _timed("9  derived_g", _drv.apply_derived_g, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)

    # Δ.12c — ``apply_existing_chain`` runs LAST (after ``apply_derived_f``)
    # so that the handoff carriers ``p_entity_previously_invested_capacity``
    # and ``p_entity_divested`` are populated before the chain summation
    # consumes them.  Previously this was inside ``apply_derived_d``, which
    # forced the call site to depend on a pre-seeded CSV value for those
    # carriers.  Now that ``apply_derived_f`` is the authoritative producer
    # of the carriers, the seed in ``_load_invest`` becomes redundant.
    from flextool.engine_polars import _derived_existing as _ex
    _timed("10 existing_chain", _ex.apply_existing_chain, flex_data, db_reader, workdir_path, ctx=ctx, provider=provider)


def _load_handoff_aux_pair(path: Path, expected: tuple[str, str],
                            *,
                            provider: "object | None" = None) -> "pl.DataFrame | None":
    """Gap F final — load a 2-col handoff-auxiliary CSV into a polars
    frame, tolerating empty / missing files.  ``expected`` lists the two
    canonical column names the caller wants; we select them and drop
    anything else.  Returns ``None`` when the file is missing, empty,
    or doesn't carry the expected columns.
    """
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return None
    try:
        df = _provider_read(provider, name, path)
    except pl.exceptions.NoDataError:
        return None
    if df.height == 0:
        return None
    a, b = expected
    if a not in df.columns or b not in df.columns:
        return None
    return df.select(a, b)


def _read_period_set(path: Path,
                       *,
                       provider: "object | None" = None) -> set[str]:
    """Read a single-column period CSV (header row, then one period per row)."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    fh = _provider_open(provider, name, path)
    if fh is None:
        return set()
    out: set[str] = set()
    with fh:
        reader = __import__("csv").reader(fh)
        next(reader, None)
        for r in reader:
            if r and r[0]:
                out.add(r[0])
    return out


def _read_realize_invest_periods(path: Path,
                                   *,
                                   provider: "object | None" = None) -> set[str]:
    """Read ``realized_invest_periods_of_current_solve.csv`` (single
    ``period`` column emitted per-solve by the periods emitter).

    Empty file or missing → empty set (treat as: nothing realized this solve).
    """
    return _read_period_set(path, provider=provider)


def _read_realized_dispatch_periods(path: Path,
                                       *,
                                       provider: "object | None" = None) -> set[str]:
    """Read distinct periods from ``realized_dispatch.csv`` (cols include ``period``)."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    fh = _provider_open(provider, name, path)
    if fh is None:
        return set()
    out: set[str] = set()
    csv = __import__("csv")
    with fh:
        reader = csv.reader(fh)
        header = next(reader, None) or []
        try:
            i = header.index("period")
        except ValueError:
            return set()
        for r in reader:
            if len(r) > i and r[i]:
                out.add(r[i])
    return out


def _read_solve_first(work_folder: Path,
                        *,
                        provider: "object | None" = None) -> bool:
    """Read ``p_model.csv``'s ``solveFirst`` flag.

    flextool's per-solve preprocessing writes ``solve_data/p_model.csv``
    with the chain-position flag (``solveFirst=1`` only for the first
    sub-solve in the multi-solve cascade, ``0`` for the rest).  The
    static ``input/p_model.csv`` does not exist in DB-driven fixtures —
    the file is purely a preprocessing-derived artifact.

    Resolution order:
    1. ``solve_data/p_model.csv`` — preferred (chain-aware).
    2. ``input/p_model.csv`` — legacy fallback when a fixture predates
       the preprocessing rewrite.
    3. Default ``True`` when neither exists.

    Bug-fix anchor: prior to Γ.8.E this only consulted ``input/`` which
    in the native cascade path produced ``solveFirst=True`` for every
    sub-solve, causing ``build_handoff_from_solution`` to add
    ``pre_existing`` to ``realized_existing`` on every iteration —
    inflating the chain's cumulative ``p_entity_period_existing_capacity``
    by ``Σ pre_existing`` per extra sub-solve and zeroing out demand on
    sub-solves 3+ of fixtures like ``wind_battery_invest_lifetime_renew_4solve``.
    """
    csv = __import__("csv")
    for cand in ("solve_data/p_model.csv", "input/p_model.csv"):
        path = work_folder / cand
        # cand already has the parent prefix and the suffix; pass as-is
        # (suffix stripped by Provider).
        name = cand
        fh = _provider_open(provider, name, path)
        if fh is None:
            continue
        with fh:
            reader = csv.reader(fh)
            header = next(reader, None) or []
            try:
                param_idx = header.index("modelParam")
                value_idx = header.index("p_model")
            except ValueError:
                return True
            for r in reader:
                if len(r) > max(param_idx, value_idx) and r[param_idx] == "solveFirst":
                    try:
                        return bool(int(r[value_idx]))
                    except (ValueError, TypeError):
                        return True
        # File existed but didn't contain the flag — treat as default.
        return True
    return True


def _read_unitsize_long(work_folder: Path,
                         *,
                         provider: "object | None" = None) -> dict[str, float]:
    """Read entity unitsizes as ``{entity: value}``.

    Layers two sources:

    * ``input/p_entity_unitsize.csv`` — fully-populated table (wide
      format from the .mod printf, or long-format from Python
      preprocessing).  Used as the base.
    * ``solve_data/p_entity_unitsize.csv`` — explicit overrides
      written by per-solve preprocessing (long format, only entities
      that diverge from the base).  Overlays on top.

    Reading only ``solve_data/`` silently drops every entity that
    doesn't carry an override — callers like ``build_handoff_from_solution``
    rely on ``unitsize.get(n, 1.0)``, so a missing entry collapses to a
    factor of 1.0 and produces handoff values off by the entity's
    unitsize.  Mirrors v3.32.0's ``_load_unitsize`` which reads
    ``input/p_entity_unitsize.csv`` (the populated copy).
    """
    out: dict[str, float] = {}
    for src, key in (
        (work_folder / "input" / "p_entity_unitsize.csv", "input/p_entity_unitsize"),
        (work_folder / "solve_data" / "p_entity_unitsize.csv", "solve_data/p_entity_unitsize"),
    ):
        if not _provider_has(provider, key, src):
            continue
        try:
            df = _read_unitsize(src, provider=provider)
        except Exception:  # noqa: BLE001 — best-effort fallback
            continue
        for r in df.iter_rows(named=True):
            try:
                out[str(r["e"])] = float(r["value"])
            except (TypeError, ValueError):
                continue
    return out


def _read_pre_existing_long(work_folder: Path,
                              *,
                              provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """Read ``solve_data/p_entity_pre_existing.csv`` (long: entity, period, value).

    Returns ``{(period, entity): value}`` to match
    flextool's ``_load_pre_existing`` key order (``[d, e]`` lookup).
    """
    path = work_folder / "solve_data" / "p_entity_pre_existing.csv"
    fh = _provider_open(provider, "solve_data/p_entity_pre_existing", path)
    if fh is None:
        return {}
    csv = __import__("csv")
    out: dict[tuple[str, str], float] = {}
    with fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[0] and r[1]:
                try:
                    out[(str(r[1]), str(r[0]))] = float(r[2])
                except ValueError:
                    continue
    return out


def _read_singles_csv(path: Path,
                       *,
                       provider: "object | None" = None) -> list[str]:
    """Read a single-column CSV (header row, then one value per row)."""
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    fh = _provider_open(provider, name, path)
    if fh is None:
        return []
    csv = __import__("csv")
    with fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _step_duration_frame(
    sd: Path, flex_data: "FlexData | None",
    *,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Return the (period/d, time/t, value) frame for p_step_duration.

    Phase 4 (Gap F) — prefer ``flex_data.p_step_duration`` (in-memory)
    over the workdir's ``p_step_duration.csv``.  ``flex_data`` Param
    frames use (d, t, value); the CSV uses (period, time, value).
    Callers tolerate either schema.
    """
    if flex_data is not None and getattr(flex_data, "p_step_duration", None) is not None:
        return flex_data.p_step_duration.frame
    p = sd / "p_step_duration.csv"
    if not _provider_has(provider, "solve_data/p_step_duration", p):
        return None
    try:
        return _provider_read(provider, "solve_data/p_step_duration", p)
    except pl.exceptions.NoDataError:
        return None


def _realized_dispatch_frame(
    sd: Path, flex_data: "FlexData | None",
    *,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Gap F final — prefer ``flex_data.realized_dispatch`` (in-memory)
    over the workdir's ``realized_dispatch.csv``.
    """
    if flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        return flex_data.realized_dispatch
    p = sd / "realized_dispatch.csv"
    if not _provider_has(provider, "solve_data/realized_dispatch", p):
        return None
    try:
        return _provider_read(provider, "solve_data/realized_dispatch", p)
    except pl.exceptions.NoDataError:
        return None


def _extract_cum_sim_hours(
    sd: Path, *, prior_handoff=None, flex_data: "FlexData | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Δ.11 — derive ``cum_sim_hours[period]`` from
    ``flex_data.p_step_duration`` (in-memory, when supplied) + the
    workdir's ``realized_dispatch.csv``.

    Algorithm:

      this_roll_hrs[d] = Σ_t step_duration[d, t]   for (d, t) ∈ realized_dispatch
      cum[d]           = prior[d] + this_roll[d]   ∀d ∈ keys(prior) ∪ keys(this_roll)

    Phase 4 (Gap F) — ``p_step_duration.csv`` read is replaced by the
    ``flex_data.p_step_duration`` Param when ``flex_data`` is supplied
    (the cascade always supplies it).  Test fixtures that call this
    helper directly without ``flex_data`` retain the disk fallback.

    Returns the wide ``[period, value]`` carrier frame, or ``None`` when
    neither prior nor the workdir's realized set contributes any rows.
    """
    rd_df = _realized_dispatch_frame(sd, flex_data, provider=provider)
    realized: set[tuple[str, str]] = set()
    if rd_df is not None and rd_df.height > 0 and {"period", "step"}.issubset(rd_df.columns):
        for r in rd_df.iter_rows(named=True):
            realized.add((str(r["period"]), str(r["step"])))
    this_roll_hrs: dict[str, float] = {}
    if realized:
        sd_df = _step_duration_frame(sd, flex_data, provider=provider)
        if sd_df is not None and sd_df.height > 0:
            d_col = "period" if "period" in sd_df.columns else (
                "d" if "d" in sd_df.columns else None)
            t_col = "time" if "time" in sd_df.columns else (
                "step" if "step" in sd_df.columns else (
                    "t" if "t" in sd_df.columns else None))
            v_col = "value" if "value" in sd_df.columns else (
                "p_step_duration" if "p_step_duration" in sd_df.columns else None
            )
            if d_col is not None and t_col is not None and v_col is not None:
                for r in sd_df.iter_rows(named=True):
                    key = (str(r[d_col]), str(r[t_col]))
                    if key in realized:
                        try:
                            this_roll_hrs[key[0]] = this_roll_hrs.get(key[0], 0.0) + float(r[v_col])
                        except (TypeError, ValueError):
                            continue
    prior_hrs: dict[str, float] = {}
    if prior_handoff is not None and prior_handoff.cum_sim_hours is not None:
        # Canonical schema is [period, p_ladder_cum_sim_hours] (Phase 4.1a).
        for r in prior_handoff.cum_sim_hours.iter_rows(named=True):
            prior_hrs[str(r["period"])] = float(r["p_ladder_cum_sim_hours"])
    if not this_roll_hrs and not prior_hrs:
        return None
    keys = sorted(set(prior_hrs) | set(this_roll_hrs))
    rows = [(d, prior_hrs.get(d, 0.0) + this_roll_hrs.get(d, 0.0)) for d in keys]
    return pl.DataFrame(
        rows, schema=["period", "p_ladder_cum_sim_hours"], orient="row",
    )


def _extract_cumulative_commodity(
    sol, sd: Path, *, prior_handoff=None, flex_data: "FlexData | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Δ.11 — derive ``cumulative_commodity[c, i, d]`` from ``v_trade``
    on the LP solution + workdir-side accumulator metadata.

    Algorithm:

      this_roll[c, i, d] = Σ_n v_trade[c, n, d, i] × unitsize[c]
                              × (realized_hours[d] / horizon_hours[d])

      cum[c, i, d] = prior[c, i, d] + this_roll[c, i, d]

    Restricted to ``(c, i)`` in ``ci_ladder_cumulative.csv`` (the finite
    tiers; non-finite tiers don't need carry-over).  Returns the wide
    ``[commodity, tier, period, mwh]`` carrier frame, or ``None`` when
    no v_trade variable exists on the solution AND no prior accumulator
    is supplied (caller falls back to the workdir CSV).
    """
    if sol is None or "v_trade" not in getattr(sol, "_vars", {}):
        # Without v_trade we can only propagate prior — same shape the
        # legacy file path uses.  Returning None lets the caller fall
        # through to the file-based propagation.
        return None

    # Finite ladder tiers — cumulative restriction set.  Phase 4 (Gap F):
    # prefer in-memory ``flex_data.ci_ladder_cumulative`` over the workdir
    # CSV when supplied.
    #
    # B1 — also filter by quantity to mirror the legacy
    # ``_load_finite_ladder_tiers``:
    # tiers with the ``1e30`` infinity sentinel (quantity >= 1e29) are
    # dropped because their cap never binds and a 0-valued accumulator
    # row carries no semantic content.  Without this filter
    # infinity-sentinel tiers like ``coal,2`` leak into the carrier with
    # 0.0 values and breach the test invariant in
    # ``tests/test_commodity_ladder_rolling.py``.
    #
    # B1 — additionally union annual-ladder tiers from
    # ``commodity__tier_ann``.  Annual fixtures may have an EMPTY
    # ``ci_ladder_cumulative`` (no cumulative ladder commodities) but
    # still need accumulator rows — the legacy
    # ``_load_finite_ladder_tiers`` reads BOTH
    # ``commodity_ladder_cumulative`` and ``commodity_ladder_annual``
    # for this same reason.
    finite_tiers: set[tuple[str, int]] = set()
    cilc_df = None
    if flex_data is not None and getattr(flex_data, "ci_ladder_cumulative", None) is not None:
        cilc_df = flex_data.ci_ladder_cumulative
    else:
        cilc_path = sd / "ci_ladder_cumulative.csv"
        if _provider_has(provider, "solve_data/ci_ladder_cumulative", cilc_path):
            try:
                cilc_df = _provider_read(provider, "solve_data/ci_ladder_cumulative", cilc_path)
            except pl.exceptions.NoDataError:
                cilc_df = None
    if cilc_df is not None and cilc_df.height > 0:
        # FlexData carries (c, i); the CSV carries (commodity, tier).
        c_col = "commodity" if "commodity" in cilc_df.columns else "c"
        i_col = "tier" if "tier" in cilc_df.columns else "i"
        if c_col in cilc_df.columns and i_col in cilc_df.columns:
            for r in cilc_df.iter_rows(named=True):
                try:
                    finite_tiers.add((str(r[c_col]), int(r[i_col])))
                except (TypeError, ValueError, KeyError):
                    continue

    # Annual-ladder tier set (commodity__tier_ann) — additive.
    cta_df = None
    if flex_data is not None and getattr(flex_data, "commodity__tier_ann", None) is not None:
        cta_df = flex_data.commodity__tier_ann
    else:
        cta_path = sd / "commodity__tier_ann.csv"
        if _provider_has(provider, "solve_data/commodity__tier_ann", cta_path):
            try:
                cta_df = _provider_read(provider, "solve_data/commodity__tier_ann", cta_path)
            except pl.exceptions.NoDataError:
                cta_df = None
    if cta_df is not None and cta_df.height > 0:
        c_col = "commodity" if "commodity" in cta_df.columns else "c"
        i_col = "tier" if "tier" in cta_df.columns else "i"
        if c_col in cta_df.columns and i_col in cta_df.columns:
            for r in cta_df.iter_rows(named=True):
                try:
                    finite_tiers.add((str(r[c_col]), int(r[i_col])))
                except (TypeError, ValueError, KeyError):
                    continue

    if not finite_tiers:
        return None

    # B1 — drop infinity-sentinel tiers (quantity >= 1e29).  The
    # quantity column lives on ``input/commodity_ladder_cumulative`` and
    # ``input/commodity_ladder_annual``; a tier is "finite" if either
    # source carries a finite quantity for it (the annual cap evaluates
    # per period, so any finite-cap period qualifies the tier).
    _INFINITE_TIER_THRESHOLD = 1e29
    qty_finite: set[tuple[str, int]] = set()
    for prov_key, csv_path in (
        ("input/commodity_ladder_cumulative",
         (Path(sd).parent / "input" / "commodity_ladder_cumulative.csv")),
        ("input/commodity_ladder_annual",
         (Path(sd).parent / "input" / "commodity_ladder_annual.csv")),
    ):
        qdf = None
        if _provider_has(provider, prov_key, csv_path):
            try:
                qdf = _provider_read(provider, prov_key, csv_path)
            except pl.exceptions.NoDataError:
                qdf = None
        if qdf is None or qdf.height == 0:
            continue
        if not {"commodity", "tier", "quantity"}.issubset(qdf.columns):
            continue
        for r in qdf.iter_rows(named=True):
            try:
                c = str(r["commodity"])
                i = int(r["tier"])
                q = float(r["quantity"])
            except (TypeError, ValueError, KeyError):
                continue
            if q != q or q >= _INFINITE_TIER_THRESHOLD:
                continue
            qty_finite.add((c, i))
    if qty_finite:
        finite_tiers = finite_tiers & qty_finite
    if not finite_tiers:
        return None

    # Per-period horizon vs realized hours (uniform-split fraction).
    # Phase 4 (Gap F) — ``p_step_duration`` sourced from FlexData when
    # supplied.  Gap F final — ``realized_dispatch`` also flows through
    # ``flex_data.realized_dispatch`` via ``_realized_dispatch_frame``.
    horizon_hrs: dict[str, float] = {}
    realized_hrs: dict[str, float] = {}
    sd_df = _step_duration_frame(sd, flex_data, provider=provider)
    if sd_df is not None and sd_df.height > 0:
        d_col = "period" if "period" in sd_df.columns else (
            "d" if "d" in sd_df.columns else None)
        t_col = "time" if "time" in sd_df.columns else (
            "step" if "step" in sd_df.columns else (
                "t" if "t" in sd_df.columns else None))
        v_col = "value" if "value" in sd_df.columns else (
            "p_step_duration" if "p_step_duration" in sd_df.columns else None
        )
        realized_set: set[tuple[str, str]] = set()
        rd_df = _realized_dispatch_frame(sd, flex_data, provider=provider)
        if rd_df is not None and rd_df.height > 0 and {"period", "step"}.issubset(rd_df.columns):
            for r in rd_df.iter_rows(named=True):
                realized_set.add((str(r["period"]), str(r["step"])))
        if d_col is not None and t_col is not None and v_col is not None:
            for r in sd_df.iter_rows(named=True):
                d_v = str(r[d_col])
                try:
                    dur = float(r[v_col])
                except (TypeError, ValueError):
                    continue
                horizon_hrs[d_v] = horizon_hrs.get(d_v, 0.0) + dur
                if (d_v, str(r[t_col])) in realized_set:
                    realized_hrs[d_v] = realized_hrs.get(d_v, 0.0) + dur

    # Commodity unitsize (defaults 1.0 when absent — flextool default).
    # Phase 4 (Gap F) — prefer in-memory ``flex_data.p_commodity_unitsize``.
    unitsize: dict[str, float] = {}
    cu_df = None
    if flex_data is not None and getattr(flex_data, "p_commodity_unitsize", None) is not None:
        cu_df = flex_data.p_commodity_unitsize.frame
    else:
        cu_path = sd / "p_commodity_unitsize.csv"
        if _provider_has(provider, "solve_data/p_commodity_unitsize", cu_path):
            try:
                cu_df = _provider_read(provider, "solve_data/p_commodity_unitsize", cu_path)
            except pl.exceptions.NoDataError:
                cu_df = None
    if cu_df is not None and cu_df.height > 0:
        c_col = "commodity" if "commodity" in cu_df.columns else (
            "c" if "c" in cu_df.columns else "name")
        v_col = "value" if "value" in cu_df.columns else "p_commodity_unitsize"
        if c_col in cu_df.columns and v_col in cu_df.columns:
            for r in cu_df.iter_rows(named=True):
                try:
                    unitsize[str(r[c_col])] = float(r[v_col])
                except (TypeError, ValueError):
                    continue

    # v_trade extraction — schema from _commodity_ladder.add_variables.
    v_trade_df = sol.value("v_trade")
    this_roll: dict[tuple[str, int, str], float] = {}
    if v_trade_df is not None and v_trade_df.height > 0:
        for r in v_trade_df.iter_rows(named=True):
            try:
                c = str(r["c"])
                i = int(r["i"])
                d = str(r["d"])
                v = float(r["value"])
            except (TypeError, ValueError, KeyError):
                continue
            if (c, i) not in finite_tiers:
                continue
            hz = horizon_hrs.get(d, 0.0)
            rz = realized_hrs.get(d, 0.0)
            if hz <= 0.0 or rz <= 0.0:
                continue
            us = unitsize.get(c, 1.0)
            key = (c, i, d)
            this_roll[key] = this_roll.get(key, 0.0) + v * us * (rz / hz)

    # Prior accumulator (carry across solves).
    # Canonical schema is [commodity, tier, period, p_ladder_cum_realized_mwh]
    # (Phase 4.1a).
    prior: dict[tuple[str, int, str], float] = {}
    if prior_handoff is not None and prior_handoff.cumulative_commodity is not None:
        for r in prior_handoff.cumulative_commodity.iter_rows(named=True):
            try:
                key = (str(r["commodity"]), int(r["tier"]), str(r["period"]))
                prior[key] = float(r["p_ladder_cum_realized_mwh"])
            except (TypeError, ValueError, KeyError):
                continue

    if not this_roll and not prior:
        return None

    keys = sorted(set(prior) | set(this_roll))
    rows = [(c, i, d, prior.get((c, i, d), 0.0) + this_roll.get((c, i, d), 0.0))
                for (c, i, d) in keys]
    return pl.DataFrame(
        rows,
        schema=["commodity", "tier", "period", "p_ladder_cum_realized_mwh"],
        orient="row",
    )


def build_handoff_from_solution(
    sol, work_folder: Path, solve_name: str,
    prior_handoff=None,
    *,
    flex_data: "FlexData | None" = None,
    parent_handoff=None,
    provider: "object | None" = None,
):
    """Build a ``SolveHandoff`` from a polar_high ``Solution`` + the work
    folder's per-solve metadata, mirroring flextool's post-solve
    ``write_p_entity_period_existing_capacity`` + ``write_p_entity_divested``
    logic but in-memory.

    Covers all 9 carriers (Γ.8.D extension — was 3 of 9 before the
    Γ.8.D port):

    * ``realized_invest`` — per-(entity, period) chain-cumulative invest.
    * ``realized_existing`` — per-(entity, period) chain-cumulative existing.
    * ``divest_cumulative`` — per-entity chain-cumulative divest.
    * ``roll_end_state`` — last-step v_state per nodeState node.
    * ``fix_storage`` — wide [node, period, time, quantity, price, usage]
      with NULL columns for inactive metrics.  ``quantity`` is populated
      from v_state at fix_storage_timesteps for fix_quantity nodes; the
      price (dual-based) and usage (flow-based) variants stay NULL until
      a fixture exercises them and the dual / flow extraction lands.
    * ``cumulative_co2`` — per-(group, period), summed from
      ``solve_data/co2_cum_realized_tonnes.csv`` if present.
    * ``cumulative_commodity`` — per-(commodity, tier, period),
      derived via :func:`_extract_cumulative_commodity` (this-roll
      v_trade + ``prior_handoff.cumulative_commodity``).
    * ``cum_sim_hours`` — per-period running sim-hour total, derived
      via :func:`_extract_cum_sim_hours` (this-roll realized hours +
      ``prior_handoff.cum_sim_hours``).
    (Δ.1 — ``periods_already_emitted`` was here; it moved to
    ``_output_writer.OutputWriterState`` since it's a writer-side
    emission gate, not a solver handoff carrier.  The on-disk source
    ``solve_data/period_capacity.csv`` is unchanged.)

    The work folder must already have completed flextool's per-solve
    preprocessing for ``solve_name`` (so ``solve_data/`` carries
    ``period_first.csv``, ``solve__ed_invest.csv``,
    ``realized_invest_periods_of_current_solve.csv``, etc.).
    """
    import polars as pl  # local — keep this helper's import surface narrow
    # Native import — Γ.8.D moved SolveHandoff into engine_polars.
    from flextool.engine_polars._solve_handoff import SolveHandoff

    sd = work_folder / "solve_data"
    first_solve = _read_solve_first(work_folder, provider=provider)
    unitsize = _read_unitsize_long(work_folder, provider=provider)
    pre_existing = _read_pre_existing_long(work_folder, provider=provider) if first_solve else {}

    # Prior solve's accumulators — sourced from the in-memory handoff
    # carriers when supplied, else empty (multi-solve cascade always
    # passes the parent handoff).
    prior_existing: dict[tuple[str, str], float] = {}
    prior_invested: dict[tuple[str, str], float] = {}
    prior_divested: dict[str, float] = {}
    if prior_handoff is not None:
        if prior_handoff.realized_existing is not None:
            for r in prior_handoff.realized_existing.iter_rows(named=True):
                prior_existing[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.realized_invest is not None:
            for r in prior_handoff.realized_invest.iter_rows(named=True):
                prior_invested[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.divest_cumulative is not None:
            for r in prior_handoff.divest_cumulative.iter_rows(named=True):
                prior_divested[str(r["entity"])] = float(r["value"])

    # ---- v_invest / v_divest from polar_high ----
    invest_by_ed: dict[tuple[str, str], float] = {}
    divest_by_e: dict[str, float] = {}
    for var_name, entity_col in (("v_invest_p", "p"), ("v_invest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                invest_by_ed[(str(r[entity_col]), str(r["d"]))] = v
    for var_name, entity_col in (("v_divest_p", "p"), ("v_divest_n", "n")):
        if var_name in sol._vars:
            df = sol.value(var_name)
            for r in df.iter_rows(named=True):
                v = float(r["value"])
                if v <= 1e-12:
                    continue
                e = str(r[entity_col])
                divest_by_e[e] = divest_by_e.get(e, 0.0) + v

    # ---- iteration set: prior keys ∪ entity × iteration_periods ----
    realize_invest = _read_realize_invest_periods(
        sd / "realized_invest_periods_of_current_solve.csv",
        provider=provider,
    )
    period_first = _read_period_set(sd / "period_first.csv", provider=provider)
    if first_solve:
        realized_periods = _read_realized_dispatch_periods(
            sd / "realized_dispatch.csv", provider=provider)
        fix_storage_periods = _read_realized_dispatch_periods(
            sd / "fix_storage_timesteps.csv", provider=provider)
        iter_periods = realize_invest | realized_periods | fix_storage_periods
    else:
        iter_periods = set(realize_invest)

    iter_keys: set[tuple[str, str]] = set(prior_existing.keys())
    entities = _read_singles_csv(sd / "entity.csv", provider=provider)
    if not entities:
        entities = _read_singles_csv(work_folder / "input" / "entity.csv",
                                       provider=provider)
    for e in entities:
        for d in iter_periods:
            iter_keys.add((e, d))

    # ---- compute realized_invest + realized_existing per (e, d) ----
    # The handoff is a CHAIN-CUMULATIVE record: every solve carries
    # forward prior solves' (e, d) contributions and ADDS its own.
    # ``first_solve`` (the .mod's solveFirst flag) means the SOLVE
    # treats itself as fresh on the LP side (no roll-state subtraction)
    # — but the OUTPUT handoff still has to cumulate, otherwise downstream
    # solves lose history.  So prior_existing / prior_invested are added
    # for every key, regardless of first_solve.
    inv_rows: list[tuple[str, str, float]] = []
    exist_rows: list[tuple[str, str, float]] = []
    for e, d in sorted(iter_keys):
        existing = 0.0
        invested = 0.0
        # Carry prior solves' contributions forward unconditionally.
        existing += prior_existing.get((e, d), 0.0)
        invested += prior_invested.get((e, d), 0.0)
        # First-solve seed: the user-defined pre-existing capacity becomes
        # part of ``realized_existing`` only on the first solve in the chain,
        # at periods belonging to that solve's period_first set.
        if first_solve and d in period_first:
            existing += pre_existing.get((d, e), 0.0)
        # This solve's invest contribution at (e, d).
        if (e, d) in invest_by_ed and d in realize_invest:
            v = invest_by_ed[(e, d)]
            us = unitsize.get(e, 1.0)
            existing += v * us
            invested += v * us
        inv_rows.append((e, d, invested))
        exist_rows.append((e, d, existing))

    # ---- divest_cumulative: prior + sum_d v_divest * unitsize ----
    entity_divest = set(_read_singles_csv(sd / "entityDivest.csv",
                                            provider=provider))
    div_rows: list[tuple[str, float]] = []
    for e in sorted(entity_divest):
        cum = prior_divested.get(e, 0.0) + divest_by_e.get(e, 0.0) * unitsize.get(e, 1.0)
        div_rows.append((e, cum))

    # ---- roll_end_state: v_state[n, last_t] * unitsize per nodeState node ----
    # Mirrors flextool's ``write_p_roll_continue_state``: takes v_state at
    # the LAST realized (period, time) pair (from ``realized_dispatch.csv``),
    # multiplies by p_entity_unitsize, and emits a (node, value) row per
    # nodeState node.  Skipped when the solve has no nodeState or no
    # realized dispatch.  See flextool/process_outputs/handoff_writers.py:
    # 250-271 (``_load_realized_period_time_last``) and 425-468.
    #
    # NOTE: source MUST be ``realized_dispatch`` (end-of-realized-commitment,
    # e.g. roll_7 jump=4 realizes t0029-t0032 → last = t0032), NOT
    # ``period__time_last`` (end-of-horizon, e.g. t0036 for the same roll).
    # Using the horizon-end variant breaks rolling storage handoff: the next
    # roll starts from a wrong initial v_state.
    roll_end_state_df = None
    nodes_state = _read_singles_csv(sd / "nodeState.csv", provider=provider)
    # Prefer ``flex_data.realized_dispatch`` (in-memory).
    last_pairs_df = None
    if flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        last_pairs_df = flex_data.realized_dispatch
    else:
        rd_path = sd / "realized_dispatch.csv"
        if _provider_has(provider, "solve_data/realized_dispatch", rd_path):
            try:
                last_pairs_df = _provider_read(provider, "solve_data/realized_dispatch", rd_path)
            except pl.exceptions.NoDataError:
                last_pairs_df = None
    if nodes_state and last_pairs_df is not None and "v_state" in sol._vars:
        # Schema: ``period, step``.  Take the last (period, step) in
        # dispatch order — equivalent to ``_load_realized_period_time_last``
        # in v3.32.0 followed by ``last_pairs[-1]`` in
        # ``write_p_roll_continue_state``.  Sorting by (d, t) and picking
        # the lexically-last row yields the end-of-realized step of the
        # last realized period.
        if last_pairs_df.height > 0:
            cols = last_pairs_df.columns
            d_col = "period" if "period" in cols else "d"
            t_col = "step" if "step" in cols else "t"
            last_pairs_df = (last_pairs_df
                .select(alias_to_axis(d_col, "d"),
                         alias_to_axis(t_col, "t"))
                .unique()
                .sort(["d", "t"]))
            if last_pairs_df.height > 0:
                last_d = last_pairs_df["d"][-1]
                last_t = last_pairs_df["t"][-1]
                v_state = sol.value("v_state")
                rcs_rows: list[tuple[str, float]] = []
                if v_state is not None and v_state.height > 0:
                    last_state = v_state.filter(
                        (pl.col("d") == last_d) & (pl.col("t") == last_t))
                    nodes_state_set = set(nodes_state)
                    for r in last_state.iter_rows(named=True):
                        n = str(r["n"])
                        if n not in nodes_state_set:
                            continue
                        v = float(r["value"]) * unitsize.get(n, 1.0)
                        rcs_rows.append((n, v))
                if rcs_rows:
                    roll_end_state_df = pl.DataFrame(
                        rcs_rows, schema=["node", "value"], orient="row")

    # ---- upward_roll_end_state: routed UPWARD to parent storage's next roll ----
    # Per specs/feature_fixes.md §1: dispatch sub-solves pass their
    # realized end-of-horizon v_state UPWARD to the parent storage
    # solve's next roll, replacing the parent's own previously-
    # predicted state.  Always-on for any storage→dispatch nesting;
    # no opt-in flag.
    #
    # Carrier value is the same as ``roll_end_state``: the LAST realized
    # (period, step) v_state per nodeState node.  The carrier exists as
    # a distinct SolveHandoff field so the parent storage's consumer
    # can prefer it explicitly over the sequential-prior fallback (see
    # input.py:2810-2819).  In the current sequential-prior orchestration,
    # the last completed solve before the parent's next roll IS the
    # child dispatch sub-solve, so sequential prior already fans the
    # dispatch's roll_end_state into the parent's provider key — but
    # adding the explicit upward carrier makes the intent visible and
    # leaves room for future routing logic (e.g. picking a non-last
    # child by topology rather than completion order).
    upward_roll_end_state_df = roll_end_state_df

    # ---- fix_storage_quantity: v_state at fix_quantity timesteps × unitsize ----
    # Mirrors flextool's ``write_fix_storage_quantity`` (handoff_writers.py
    # :380).  Restricted to nodes whose storage_nested_fix_method is
    # ``fix_quantity`` and (period, step) in fix_storage_timesteps.csv.
    # The fix_price (dual-based) and fix_usage (flow-based) variants are
    # left unfilled here — they require nodeBalance_eq dual extraction
    # / per-arc flow summation which is significantly more involved than
    # the quantity case and isn't exercised by the multi_invest fixture.
    # Phase 4.1l — the narrow per-metric carrier is built directly from
    # v_state × unitsize; the legacy wide intermediate frame has been
    # retired (its last consumer, _native_run_model._fan_out_fix_storage,
    # was deleted in 4.1l).
    fix_storage_quantity_df = None
    fq_nodes: set[str] = set()
    # Gap F final — prefer ``flex_data.node__storage_nested_fix_method``.
    nsfm_df = None
    if flex_data is not None and getattr(
            flex_data, "node__storage_nested_fix_method", None) is not None:
        nsfm_df = flex_data.node__storage_nested_fix_method
    else:
        nsfm_path = sd / "node__storage_nested_fix_method.csv"
        if _provider_has(provider, "solve_data/node__storage_nested_fix_method", nsfm_path):
            try:
                nsfm_df = _provider_read(provider, "solve_data/node__storage_nested_fix_method", nsfm_path)
            except pl.exceptions.NoDataError:
                nsfm_df = None
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fq_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_quantity")["node"]
            .cast(pl.Utf8).to_list()
        )
    # Gap F final — prefer the in-memory fix_storage_timesteps carriers:
    # ``parent_handoff.fix_storage_timesteps`` deposits the (period, step)
    # set for child solves; otherwise this solve emits its own set via
    # ``emit_fix_storage_timesteps`` and we read it back from the Provider.
    fs_steps_df = None
    if parent_handoff is not None and getattr(
            parent_handoff, "fix_storage_timesteps", None) is not None:
        fs_steps_df = parent_handoff.fix_storage_timesteps
    else:
        fix_steps_path = sd / "fix_storage_timesteps.csv"
        if _provider_has(provider, "solve_data/fix_storage_timesteps", fix_steps_path):
            try:
                fs_steps_df = _provider_read(provider, "solve_data/fix_storage_timesteps", fix_steps_path)
            except pl.exceptions.NoDataError:
                fs_steps_df = None
    if (fq_nodes
            and fs_steps_df is not None
            and "v_state" in sol._vars):
        if fs_steps_df.height > 0 and {"period", "step"}.issubset(
                fs_steps_df.columns):
            fs_steps = (fs_steps_df
                .select(alias_to_axis("period", "d"),
                         alias_to_axis("step", "t"))
                .unique())
            v_state = sol.value("v_state")
            if v_state is not None and v_state.height > 0:
                fq_rows = (v_state
                    .filter(pl.col("n").is_in(list(fq_nodes)))
                    .join(fs_steps, on=["d", "t"], how="inner"))
                if fq_rows.height > 0:
                    # Multiply by unitsize (per-node) and emit the
                    # canonical narrow schema
                    # ``[node, period, step, p_fix_storage_quantity]``.
                    us_rows = [(n, unitsize.get(n, 1.0))
                                 for n in sorted(fq_nodes)]
                    us_df = pl.DataFrame(
                        us_rows, schema=["n", "us"], orient="row")
                    fq_rows = (fq_rows
                        .join(us_df, on="n", how="inner")
                        .with_columns(
                            p_fix_storage_quantity=pl.col("value") * pl.col("us"),
                        )
                        .select(
                            alias_to_axis("n", "node"),
                            alias_to_axis("d", "period"),
                            alias_to_axis("t", "step"),
                            pl.col("p_fix_storage_quantity"),
                        ))
                    if fq_rows.height > 0:
                        fix_storage_quantity_df = fq_rows

    # ---- cumulative_co2: per-(group, period) running total ----
    # Gap F final close-out — native compute via
    # ``_emit_co2_accumulators.compute_co2_rolling_accumulator`` when
    # ``flex_data`` + ``sol`` are available (cascade path).  Falls back to
    # the disk read for legacy / test callers that only pass ``sol``.
    cumulative_co2_df = None
    if prior_handoff is not None and prior_handoff.cumulative_co2 is not None:
        cumulative_co2_df = prior_handoff.cumulative_co2
    used_native_co2 = False
    if flex_data is not None and sol is not None:
        from flextool.engine_polars._emit_co2_accumulators import (
            compute_co2_rolling_accumulator,
        )
        prior_df = (prior_handoff.cumulative_co2
                    if prior_handoff is not None else None)
        native_co2 = compute_co2_rolling_accumulator(
            flex_data, sol, work_folder=work_folder,
            prior_cumulative_co2=prior_df,
            provider=provider,
        )
        if native_co2.height > 0:
            cumulative_co2_df = (native_co2
                .with_columns(
                    value=pl.col("p_co2_cum_realized_tonnes")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0))
                .select("group", "period", "value"))
            used_native_co2 = True
    if not used_native_co2:
        co2_path = sd / "co2_cum_realized_tonnes.csv"
        if _provider_has(provider, "solve_data/co2_cum_realized_tonnes", co2_path):
            try:
                co2_df = _provider_read(provider, "solve_data/co2_cum_realized_tonnes", co2_path)
            except pl.exceptions.NoDataError:
                co2_df = None
            if co2_df is not None and co2_df.height > 0 and \
                    "p_co2_cum_realized_tonnes" in co2_df.columns:
                cumulative_co2_df = (
                    co2_df.with_columns(
                        value=pl.col("p_co2_cum_realized_tonnes")
                                .cast(pl.Float64, strict=False)
                                .fill_null(0.0))
                      .select("group", "period", "value"))

    # ---- cumulative_commodity: per-(commodity, tier, period) running mwh ----
    # Δ.11 — derive from sol when v_trade is in the LP and prior_handoff
    # carries finite-tier ladder commodities.  Algorithm:
    #
    #   this_roll_mwh[c, i, d] = Σ_n v_trade[c, n, d, i] × unitsize[c]
    #                              × (realized_hours[d] / horizon_hours[d])
    #
    # Cumulative across solves: prior_mwh + this_roll_mwh, restricted to
    # finite ladder tiers (``ci_ladder_cumulative.csv``) — non-finite
    # tiers don't need the carry-over since their cap is unbounded.
    #
    # Falls back to the workdir CSV when v_trade isn't in the solution
    # (the LP didn't expose the variable for this fixture) — preserves
    # the legacy propagation path.
    cumulative_commodity_df = _extract_cumulative_commodity(
        sol, sd, prior_handoff=prior_handoff, flex_data=flex_data,
        provider=provider)
    if cumulative_commodity_df is None:
        # Phase 4 (Gap F) — disk fallback retired.  When
        # ``_extract_cumulative_commodity`` returns None this solve had no
        # this-roll increment to add (no ``v_trade`` in the LP, or no
        # finite ladder tiers), so the carrier reduces to whatever the
        # prior solve deposited.  The legacy
        # ``solve_data/commodity_ladder_cumulative.csv`` fallback is
        # unreachable in the cascade path: ``write_ladder_rolling_accumulators``
        # writes ``ladder_cum_realized_mwh.csv`` (different name), and
        # ``commodity_ladder_cumulative.csv`` lives under ``input/`` not
        # ``solve_data/`` in production.
        if (prior_handoff is not None
                and prior_handoff.cumulative_commodity is not None):
            cumulative_commodity_df = prior_handoff.cumulative_commodity

    # ---- cum_sim_hours: per-period running sim-hour total ----
    # Δ.11 — derive from the workdir's ``p_step_duration.csv`` +
    # ``realized_dispatch.csv`` (uniform-split assumption):
    #
    #   this_roll_hrs[d] = Σ_t step_duration[d, t] for (d, t) ∈ realized_dispatch
    #
    # Cumulative: prior_hrs + this_roll_hrs.  ``v_trade`` not required —
    # the carrier exists for every chained fixture even when the ladder
    # itself is inactive (CO2-cap normalisation also consumes it).
    cum_sim_hours_df = _extract_cum_sim_hours(
        sd, prior_handoff=prior_handoff, flex_data=flex_data,
        provider=provider)
    if cum_sim_hours_df is None:
        # Disk fallback retired.  ``_extract_cum_sim_hours`` returns
        # None only when this solve has zero realized timesteps AND
        # ``prior_handoff`` carries no prior hours.  Propagate the
        # prior carrier as-is when present — the post-solve writer
        # hasn't run yet, so re-reading disk would just round-trip
        # data we already hold in memory.
        if (prior_handoff is not None
                and prior_handoff.cum_sim_hours is not None):
            cum_sim_hours_df = prior_handoff.cum_sim_hours

    # Δ.1 — ``periods_already_emitted`` extraction removed.  The carrier
    # moved to ``_output_writer.OutputWriterState`` (writer-side state).
    # ``solve_data/period_capacity.csv`` is unchanged (handoff_writers
    # still bumps it post-solve).

    # Phase 4.1h / 4.1l — parent's narrow fix_storage_usage field passes
    # straight through to this solve's narrow field when nested.  The
    # fix_storage_quantity carrier is produced above from v_state ×
    # unitsize; deferred-B Phase B2 (below) produces fix_storage_price
    # from ``nodeBalance_eq`` row duals at this solve's fix_price
    # storage timesteps.  The parent's narrow price carrier remains the
    # fallback when this solve declares no fix_price-method nodes.
    fix_storage_price_df = None
    fix_storage_usage_df = None
    if parent_handoff is not None:
        if parent_handoff.fix_storage_price is not None:
            fix_storage_price_df = parent_handoff.fix_storage_price
        if parent_handoff.fix_storage_usage is not None:
            fix_storage_usage_df = parent_handoff.fix_storage_usage

    # ── deferred-B Phase B2: fix_storage_price from nodeBalance_eq duals ─
    # For each storage node n with
    #   node__storage_nested_fix_method[n] == 'fix_price'
    # and each (period, step) in fix_storage_timesteps, take the row
    # dual of ``nodeBalance_eq[n, d, t]`` and normalize:
    #
    #   p_fix_storage_price[n, d, t]
    #     = -dual / p_inflation_op[d]
    #              * p_period_share[d]
    #              / scale_the_objective
    #
    # Mirrors v3.32.0 ``write_fix_storage_price`` (handoff_writers.py
    # :594-691) — same minus sign / same factor product.
    # ``scale_the_objective`` comes from
    # ``solve_data/scale_the_objective.csv`` (emitted by the native
    # input writer; defaults to 1.0 when absent/unreadable).  This
    # overrides the parent passthrough above whenever this solve has
    # any fix_price node × fix-step pair with extractable duals.
    fp_nodes: set[str] = set()
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fp_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_price")["node"]
            .cast(pl.Utf8).to_list()
        )
    if (fp_nodes
            and fs_steps_df is not None
            and fs_steps_df.height > 0
            and {"period", "step"}.issubset(fs_steps_df.columns)
            and sol is not None
            and flex_data is not None):
        try:
            duals_df = sol.constraint_dual("nodeBalance_eq")
        except KeyError:
            duals_df = None
        if duals_df is not None and duals_df.height > 0 and "key" in duals_df.columns:
            # Row names are formatted as ``"nodeBalance_eq[n,d,t]"`` —
            # ``constraint_dual`` strips the prefix/brackets and exposes
            # the comma-joined dims as a single ``key`` string column.
            # Split into the over=(n, d, t) axis triple.
            parsed = duals_df.with_columns(
                pl.col("key").str.split_exact(",", 2).alias("_parts"),
            )
            parsed = parsed.with_columns(
                n=pl.col("_parts").struct.field("field_0"),
                d=pl.col("_parts").struct.field("field_1"),
                t=pl.col("_parts").struct.field("field_2"),
            ).drop(["_parts", "key"])

            fs_steps_fp = (fs_steps_df
                .select(alias_to_axis("period", "d"),
                         alias_to_axis("step", "t"))
                .unique())
            fp_node_df = pl.DataFrame(
                {"n": sorted(fp_nodes)},
                schema={"n": pl.Utf8},
            )

            picked = (parsed
                .with_columns(
                    pl.col("n").cast(pl.Utf8),
                    pl.col("d").cast(pl.Utf8),
                    pl.col("t").cast(pl.Utf8),
                )
                .join(fp_node_df, on="n", how="inner")
                .join(fs_steps_fp.with_columns(
                          pl.col("d").cast(pl.Utf8),
                          pl.col("t").cast(pl.Utf8),
                      ),
                      on=["d", "t"], how="inner"))

            if picked.height > 0:
                # Per-period inflation_op / period_share dictionaries.
                infl_frame = flex_data.p_inflation_op.frame.select(
                    "d", pl.col("value").alias("infl"),
                ).with_columns(pl.col("d").cast(pl.Utf8))
                share_frame = flex_data.p_period_share.frame.select(
                    "d", pl.col("value").alias("share"),
                ).with_columns(pl.col("d").cast(pl.Utf8))

                scale_val = 1.0
                scale_path = sd / "scale_the_objective.csv"
                if _provider_has(provider, _provider_key(scale_path), scale_path):
                    try:
                        scale_df = _provider_read(
                            provider, _provider_key(scale_path), scale_path,
                        )
                        if scale_df.height > 0 and "value" in scale_df.columns:
                            v0 = scale_df["value"][0]
                            if v0 is not None and float(v0) > 0:
                                scale_val = float(v0)
                    except Exception:
                        scale_val = 1.0

                normalized = (picked
                    .join(infl_frame, on="d", how="left")
                    .join(share_frame, on="d", how="left")
                    .with_columns(
                        infl=pl.col("infl").fill_null(1.0),
                        share=pl.col("share").fill_null(1.0),
                    )
                    .with_columns(
                        p_fix_storage_price=(
                            -pl.col("dual") / pl.col("infl")
                            * pl.col("share") / pl.lit(scale_val)
                        ),
                    )
                    .select(
                        alias_to_axis("n", "node"),
                        alias_to_axis("d", "period"),
                        alias_to_axis("t", "step"),
                        pl.col("p_fix_storage_price"),
                    ))
                if normalized.height > 0:
                    fix_storage_price_df = normalized

    # ── deferred-B Phase B3: fix_storage_usage from v_flow primals ──────
    # For each storage node n with
    #   node__storage_nested_fix_method[n] == 'fix_usage'
    # and each (period, step) in fix_storage_timesteps, sum v_flow primals
    # of arcs touching n, weighted by p_entity_unitsize[process] and
    # p_step_duration[d, t].  The formula mirrors the LP constraint
    # ``node_storage_usage_fix_le`` (model.py:1401-1545) so the
    # producer ↔ constraint round-trip closes exactly:
    #
    #   p_fix_storage_usage[n, d, t]
    #     = ( - Σ_{(p, n, sink) ∈ pss}                                # n-as-sink (raw)
    #             v_flow[p, n, sink, d, t] * unitsize[p]
    #         + Σ_{(p, source, n) ∈ pss_eff, p_slope present}         # n-as-source (eff, slope)
    #             v_flow[p, source, n, d, t] * unitsize[p] * slope[p, d, t]
    #         + Σ_{(p, source, n) ∈ pss_eff ∩ min_load_eff}           # n-as-source (eff, min_load_eff section)
    #             v_online[p, d, t] * section[p, d, t] * unitsize[p]
    #         + Σ_{(p, source, n) ∈ pss_noEff}                        # n-as-source (noEff, raw)
    #             v_flow[p, source, n, d, t] * unitsize[p]
    #       ) × step_duration[d, t]
    #
    # Diverges from the legacy producer (handoff_writers.py:694-766)
    # which always used the simplified formula (sink and source both
    # raw, no slope/section).  The new engine prefers round-trip
    # consistency with its own constraint over legacy fidelity — see
    # specs/feature_fixes.md §3 and the user decision recorded there
    # (option a).  Pairing fix_usage with min_load_efficiency and
    # non-unit slope is now supported.  Per-process flow-coefficient
    # ratio is still deferred (matches model.py:1424-1429).
    #
    # Overrides the parent passthrough whenever this solve has any
    # fix_usage node × fix-step pair with extractable flows.
    fu_nodes: set[str] = set()
    if nsfm_df is not None and nsfm_df.height > 0 and "method" in nsfm_df.columns:
        fu_nodes = set(
            nsfm_df.filter(pl.col("method") == "fix_usage")["node"]
            .cast(pl.Utf8).to_list()
        )
    if (fu_nodes
            and fs_steps_df is not None
            and fs_steps_df.height > 0
            and {"period", "step"}.issubset(fs_steps_df.columns)
            and sol is not None
            and flex_data is not None
            and "v_flow" in getattr(sol, "_vars", {})):
        v_flow_long = sol.value("v_flow")
        if v_flow_long is not None and v_flow_long.height > 0:
            fs_steps_fu = (fs_steps_df
                .select(alias_to_axis("period", "d"),
                         alias_to_axis("step", "t"))
                .unique()
                .with_columns(pl.col("d").cast(pl.Utf8),
                              pl.col("t").cast(pl.Utf8)))

            us_param = (getattr(flex_data, "p_all_entity_unitsize", None)
                        or getattr(flex_data, "p_unitsize", None))
            if us_param is not None:
                unitsize_frame = (us_param.frame
                    .select("p", pl.col("value").alias("us"))
                    .with_columns(pl.col("p").cast(pl.Utf8)))
            else:
                unitsize_frame = pl.DataFrame(
                    schema={"p": pl.Utf8, "us": pl.Float64})

            step_dur_param = getattr(flex_data, "p_step_duration", None)
            if step_dur_param is not None:
                step_dur_frame = (step_dur_param.frame
                    .select("d", "t", pl.col("value").alias("dur"))
                    .with_columns(pl.col("d").cast(pl.Utf8),
                                  pl.col("t").cast(pl.Utf8)))
            else:
                step_dur_frame = pl.DataFrame(
                    schema={"d": pl.Utf8, "t": pl.Utf8, "dur": pl.Float64})

            fu_node_df = pl.DataFrame(
                {"n": sorted(fu_nodes)}, schema={"n": pl.Utf8},
            )

            flows = (v_flow_long
                .with_columns(
                    pl.col("p").cast(pl.Utf8),
                    pl.col("source").cast(pl.Utf8),
                    pl.col("sink").cast(pl.Utf8),
                    pl.col("d").cast(pl.Utf8),
                    pl.col("t").cast(pl.Utf8),
                )
                .join(fs_steps_fu, on=["d", "t"], how="inner")
                .join(unitsize_frame, on="p", how="left")
                .with_columns(pl.col("us").fill_null(1.0)))

            # n-as-sink: inflow contribution (negative, raw v_flow * unitsize).
            # Mirrors constraint LHS sink_flow term (model.py:1500-1504).
            # Applied to all sink arcs touching fu_nodes regardless of
            # eff vs noEff partition.
            sink_side = (flows
                .join(fu_node_df, left_on="sink", right_on="n", how="inner")
                .with_columns(
                    contrib=-pl.col("value") * pl.col("us"),
                    n=pl.col("sink"),
                )
                .select("n", "d", "t", "contrib"))

            # n-as-source contributions partitioned to match constraint:
            #   - eff partition: v_flow * unitsize * slope (+ section term
            #     for min_load_efficiency processes, see below).
            #   - noEff partition: v_flow * unitsize (raw).
            contrib_sides: list[pl.DataFrame] = [sink_side]

            pss_eff_fu = None
            if (flex_data.process_source_sink_eff is not None
                    and flex_data.process_source_sink_eff.height > 0):
                pss_eff_fu = (flex_data.process_source_sink_eff
                    .with_columns(
                        pl.col("p").cast(pl.Utf8),
                        pl.col("source").cast(pl.Utf8),
                        pl.col("sink").cast(pl.Utf8),
                    )
                    .join(fu_node_df, left_on="source", right_on="n",
                          how="inner"))
                if pss_eff_fu.height > 0:
                    # source_eff with slope multiplier (model.py:1505-1511
                    # gates on p_slope is not None — mirror that gate).
                    if flex_data.p_slope is not None:
                        slope_frame = (flex_data.p_slope.frame
                            .select("p", "d", "t",
                                    pl.col("value").alias("slope"))
                            .with_columns(
                                pl.col("p").cast(pl.Utf8),
                                pl.col("d").cast(pl.Utf8),
                                pl.col("t").cast(pl.Utf8),
                            ))
                        source_eff_side = (flows
                            .join(pss_eff_fu.select("p", "source", "sink"),
                                  on=["p", "source", "sink"], how="inner")
                            .join(slope_frame, on=["p", "d", "t"], how="left")
                            .with_columns(pl.col("slope").fill_null(1.0))
                            .with_columns(
                                contrib=pl.col("value") * pl.col("us")
                                        * pl.col("slope"),
                                n=pl.col("source"),
                            )
                            .select("n", "d", "t", "contrib"))
                        contrib_sides.append(source_eff_side)

                    # source_eff min_load_efficiency section term
                    # (model.py:1512-1531).  Uses v_online_linear /
                    # v_online_integer * section * unitsize.  No slope
                    # multiplier (per the constraint).
                    if (flex_data.process_min_load_eff is not None
                            and flex_data.process_min_load_eff.height > 0
                            and flex_data.p_section is not None):
                        mle_p = (flex_data.process_min_load_eff
                            .select("p")
                            .with_columns(pl.col("p").cast(pl.Utf8))
                            .unique())
                        section_arcs = (pss_eff_fu
                            .select("p", "source", "sink")
                            .join(mle_p, on="p", how="inner"))
                        if section_arcs.height > 0:
                            section_frame = (flex_data.p_section.frame
                                .select("p", "d", "t",
                                        pl.col("value").alias("section"))
                                .with_columns(
                                    pl.col("p").cast(pl.Utf8),
                                    pl.col("d").cast(pl.Utf8),
                                    pl.col("t").cast(pl.Utf8),
                                ))
                            # Combine v_online_linear + v_online_integer
                            # (both contribute to the section term per
                            # model.py:1518-1531; only one is typically
                            # present per process, but the sum is safe).
                            online_long = None
                            sol_vars = getattr(sol, "_vars", {}) or {}
                            for var_name in (
                                "v_online_linear", "v_online_integer",
                            ):
                                if var_name in sol_vars:
                                    vd = sol.value(var_name)
                                    if vd is not None and vd.height > 0:
                                        vd = vd.with_columns(
                                            pl.col("p").cast(pl.Utf8),
                                            pl.col("d").cast(pl.Utf8),
                                            pl.col("t").cast(pl.Utf8),
                                        ).select("p", "d", "t", "value")
                                        online_long = (vd if online_long is None
                                                       else pl.concat(
                                                           [online_long, vd],
                                                           how="vertical"))
                            if (online_long is not None
                                    and online_long.height > 0):
                                section_side = (online_long
                                    .join(fs_steps_fu, on=["d", "t"],
                                          how="inner")
                                    .join(section_arcs, on="p", how="inner")
                                    .join(section_frame,
                                          on=["p", "d", "t"], how="left")
                                    .with_columns(
                                        pl.col("section").fill_null(0.0))
                                    .join(unitsize_frame, on="p", how="left")
                                    .with_columns(
                                        pl.col("us").fill_null(1.0))
                                    .with_columns(
                                        contrib=pl.col("value")
                                                * pl.col("section")
                                                * pl.col("us"),
                                        n=pl.col("source"),
                                    )
                                    .select("n", "d", "t", "contrib"))
                                contrib_sides.append(section_side)

            if (flex_data.process_source_sink_noEff is not None
                    and flex_data.process_source_sink_noEff.height > 0):
                pss_noEff_fu = (flex_data.process_source_sink_noEff
                    .with_columns(
                        pl.col("p").cast(pl.Utf8),
                        pl.col("source").cast(pl.Utf8),
                        pl.col("sink").cast(pl.Utf8),
                    )
                    .join(fu_node_df, left_on="source", right_on="n",
                          how="inner"))
                if pss_noEff_fu.height > 0:
                    source_noEff_side = (flows
                        .join(pss_noEff_fu.select("p", "source", "sink"),
                              on=["p", "source", "sink"], how="inner")
                        .with_columns(
                            contrib=pl.col("value") * pl.col("us"),
                            n=pl.col("source"),
                        )
                        .select("n", "d", "t", "contrib"))
                    contrib_sides.append(source_noEff_side)

            # Fallback when neither partition is populated: legacy
            # simplified formula on the full arc set (covers fixtures
            # whose eff/noEff partition is missing, e.g. the no-process
            # toy fixtures used by wiring tests; the new formula must
            # not regress those).
            if len(contrib_sides) == 1:  # sink_side only — no source partitions
                source_side = (flows
                    .join(fu_node_df, left_on="source", right_on="n",
                          how="inner")
                    .with_columns(
                        contrib=pl.col("value") * pl.col("us"),
                        n=pl.col("source"),
                    )
                    .select("n", "d", "t", "contrib"))
                contrib_sides.append(source_side)

            net = (pl.concat(contrib_sides, how="vertical")
                .group_by(["n", "d", "t"])
                .agg(pl.col("contrib").sum().alias("net")))

            if net.height > 0:
                weighted = (net
                    .join(step_dur_frame, on=["d", "t"], how="left")
                    .with_columns(pl.col("dur").fill_null(1.0))
                    .with_columns(
                        p_fix_storage_usage=pl.col("net") * pl.col("dur"),
                    )
                    .filter(pl.col("p_fix_storage_usage") != 0.0)
                    .select(
                        alias_to_axis("n", "node"),
                        alias_to_axis("d", "period"),
                        alias_to_axis("t", "step"),
                        pl.col("p_fix_storage_usage"),
                    ))
                if weighted.height > 0:
                    fix_storage_usage_df = weighted

    return SolveHandoff(
        realized_invest=pl.DataFrame(
            inv_rows, schema=["entity", "period", "value"], orient="row",
        ) if inv_rows else None,
        realized_existing=pl.DataFrame(
            exist_rows, schema=["entity", "period", "value"], orient="row",
        ) if exist_rows else None,
        divest_cumulative=pl.DataFrame(
            div_rows, schema=["entity", "value"], orient="row",
        ) if div_rows else None,
        roll_end_state=roll_end_state_df,
        upward_roll_end_state=upward_roll_end_state_df,
        fix_storage_quantity=fix_storage_quantity_df,
        fix_storage_price=fix_storage_price_df,
        fix_storage_usage=fix_storage_usage_df,
        cumulative_co2=cumulative_co2_df,
        cumulative_commodity=cumulative_commodity_df,
        cum_sim_hours=cum_sim_hours_df,
    )


def _overlay_handoff(flex_data: "FlexData", handoff,
                       solve_data_dir: Path | None = None,
                       *,
                       ctx: "object | None" = None,
                       provider: "object | None" = None) -> "FlexData":
    """Δ.11 — internal helper used by :func:`load_flextool` to overlay an
    in-memory :class:`SolveHandoff` onto the FlexData built from disk.

    Returns a NEW FlexData with the 5 carrier-derived fields replaced
    (uses :func:`dataclasses.replace`, original untouched).  Called from
    inside :func:`load_flextool` when ``handoff`` is supplied — there is
    no longer a public ``apply_handoff`` entry point; the construct-with-
    handoff path is the only supported way to pipe an in-memory
    :class:`SolveHandoff` into a fresh :class:`FlexData`.

    Carriers overlaid (target FlexData fields):

    * ``p_entity_previously_invested_capacity (e, d)``  ← derived from
      ``realized_invest`` summed over historical periods using
      ``solve_data/edd_history.csv``.  This mirrors flextool's
      ``write_p_entity_previously_invested_capacity`` (see
      ``preprocessing/entity_period_calc_params.py:1584``):
      ``v[e, d] = Σ_{(e, d_h, d) ∈ edd_history ∧ (e, d_h) realized}  realized_invest[(e, d_h)]``.
    * ``p_entity_invested (e,)``  ← ``realized_invest`` summed over period.
    * ``p_entity_divested (e,)``  ← ``divest_cumulative``.
    * ``p_roll_continue_state (n,)``  ← ``roll_end_state``.
    * ``p_fix_storage_quantity (n, d, t)``  ← ``fix_storage_quantity``.

    For each carrier, ``None`` on the handoff side leaves the FlexData
    field untouched (snapshot wins).  Non-None replaces the entire
    field — the handoff is the source of truth.  Rows with value=0.0
    are filtered to match the canonical loader's behaviour (see
    ``_read_handoff_e_d`` at L1215).

    Parameters
    ----------
    flex_data : FlexData
        Base FlexData (typically from ``load_flextool``) carrying the
        sub-solve's structure (sets, profiles, methods).
    handoff : SolveHandoff
        Carrier set built by :func:`build_handoff_from_solution` from the
        prior sub-solve's polar_high solution.
    solve_data_dir : Path, optional
        Path to the current sub-solve's ``solve_data/`` directory.
        Required for the ``p_entity_previously_invested_capacity``
        overlay (it reads ``edd_history.csv`` to know which prior
        invest periods feed each current period).  ``None`` skips that
        carrier; the snapshot's pre-written value is then used as-is.
    """
    from dataclasses import replace
    overrides: dict = {}

    # --- p_entity_previously_invested_capacity (e, d): realized_invest
    # summed over the historical d_h that feed each current d, per
    # solve_data/edd_history.csv ∩ ed_history_realized.
    # Mirrors flextool/preprocessing/entity_period_calc_params.py:1525-1543.
    # Phase E-g — seed-aware existence check so the overlay still fires
    # when the cascade runs under ``the in-memory cascade`` (the per-
    # sub-solve accumulator carries ``edd_history.csv`` in memory even
    # when the disk file is suppressed).
    if (handoff.realized_invest is not None
            and solve_data_dir is not None
            and _provider_has(provider, "solve_data/edd_history",
                                solve_data_dir / "edd_history.csv")):
        # Build the (e, d_h) → realized_invest dict.
        ppic: dict[tuple[str, str], float] = {}
        for r in handoff.realized_invest.iter_rows(named=True):
            ppic[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        # ed_history_realized = keys(ppic) ∪ ed_history_realized_first.csv.
        ed_realized: set[tuple[str, str]] = set(ppic.keys())
        ehrf_path = solve_data_dir / "ed_history_realized_first.csv"
        if _provider_has(provider, "solve_data/ed_history_realized_first", ehrf_path):
            ehrf = _provider_read(provider, "solve_data/ed_history_realized_first", ehrf_path)
            if ehrf.height > 0:
                for r in ehrf.iter_rows(named=True):
                    ed_realized.add((str(r["entity"]), str(r["period"])))
        # Sum realized_invest over historical d_h per (e, d).
        edd_hist = _provider_read(provider, "solve_data/edd_history",
                                    solve_data_dir / "edd_history.csv")
        prev_inv: dict[tuple[str, str], float] = {}
        if edd_hist.height > 0:
            for r in edd_hist.iter_rows(named=True):
                e = str(r["entity"]); d_h = str(r["period_history"])
                d = str(r["period"])
                if (e, d_h) in ed_realized:
                    prev_inv[(e, d)] = prev_inv.get((e, d), 0.0) \
                                       + ppic.get((e, d_h), 0.0)
        if prev_inv:
            rows = [(e, d, v) for (e, d), v in prev_inv.items() if v != 0.0]
            if rows:
                df = pl.DataFrame(rows,
                                    schema=["e", "d", "value"], orient="row")
                overrides["p_entity_previously_invested_capacity"] = \
                    Param(("e", "d"), df)
            else:
                overrides["p_entity_previously_invested_capacity"] = None
        else:
            overrides["p_entity_previously_invested_capacity"] = None

    # --- realized_invest → p_entity_invested (e,)  (sum over period) ---
    # ``p_entity_invested`` is a per-entity scalar (cumulative prior-solve
    # invest), the sum of ``realized_invest`` rows for that entity.
    if handoff.realized_invest is not None:
        df = (handoff.realized_invest
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                 .fill_null(0.0))
            .group_by("entity").agg(pl.col("value").sum())
            .pipe(rename_to_axis, {"entity": "e"})
            .filter(pl.col("value") != 0.0)
            .select("e", "value"))
        overrides["p_entity_invested"] = (
            Param(("e",), df) if df.height > 0 else None)

    # --- divest_cumulative → p_entity_divested (e,) ---
    if handoff.divest_cumulative is not None:
        df = (handoff.divest_cumulative
            .pipe(rename_to_axis, {"entity": "e"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                 .fill_null(0.0))
            .filter(pl.col("value") != 0.0)
            .select("e", "value"))
        overrides["p_entity_divested"] = (
            Param(("e",), df) if df.height > 0 else None)

    # --- roll_end_state → p_roll_continue_state (n,) ---
    if handoff.roll_end_state is not None:
        df = (handoff.roll_end_state
            .pipe(rename_to_axis, {"node": "n"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
            .select("n", "value"))
        # The loader does NOT filter zero rows for this carrier (see
        # input.py:1996-1999) — it keeps them.  Match that behaviour.
        overrides["p_roll_continue_state"] = (
            Param(("n",), df) if df.height > 0 else None)

    # --- fix_storage_quantity → p_fix_storage_quantity (n, d, t) ---
    # Phase 4.1k — read the canonical narrow field
    # ``SolveHandoff.fix_storage_quantity`` (schema
    # ``[node, period, step, p_fix_storage_quantity]``) populated by
    # ``build_handoff_from_solution`` (4.1c).  Mirrors the Provider-key
    # consumer at L2620-2641.
    if handoff.fix_storage_quantity is not None:
        df = (handoff.fix_storage_quantity
            .pipe(rename_to_axis, {"node": "n", "period": "d", "step": "t",
                     "p_fix_storage_quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
            .select("n", "d", "t", "value"))
        overrides["p_fix_storage_quantity"] = (
            Param(("n", "d", "t"), df) if df.height > 0 else None)

    if not overrides:
        return flex_data
    return _assign_param_names(replace(flex_data, **overrides))
