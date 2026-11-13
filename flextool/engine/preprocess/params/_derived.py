"""Γ.3.A — Foundational Derived Param helpers.

This module covers the first batch of Derived Params per the
``audit/db_direct_param_map.md`` §3 deep-dive:

* ``§3.1`` Time / weighting:
  ``dt`` / ``p_step_duration``, ``p_timestep_weight``,
  ``p_inflation_op``, ``p_period_share``.
* ``§3.2`` Nodes:
  ``p_inflow`` (scaling step on top of node.inflow Map).
* ``§3.6`` Profiles:
  ``p_profile_value`` (multi-tier alternative cascade),
  ``p_process_existing_count``.
* ``§3.18`` Stochastic:
  ``pdt_branch_weight`` / ``pd_branch_weight``.

Each helper takes an :class:`flextool._input_source.InputSource` plus
any already-computed sibling frames it depends on (foundational order
established in §7.3 Phase Γ.3a) and returns the eager
``Param`` / ``DataFrame`` shape consumed by downstream model-build.

Lazy-evaluation pattern
-----------------------
``InputSource.parameter()`` collects at the source boundary.  Helpers
compose multi-call results via :class:`pl.LazyFrame` and ``.collect()``
once per public function.

Per-solve scope
---------------
Several Params (``dt``, ``p_step_duration``, ``p_timestep_weight``,
``pdt_branch_weight``, ``pd_branch_weight``) are per-active-solve.  The
helpers accept an explicit ``active_solve`` argument; the
:func:`apply_derived_a` integration entrypoint reads
``solve_data/solve_current.csv`` from the workdir to pick the active
solve, mirroring the established hand-off used by the CSV path.
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    align_join_dtypes,
    axis_lazyframe,
    cast_dim,
    cast_frame_axes,
    get_global_axis_enums,
    lit_axis,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.params._branch import (
    active_branch_anchor_pairs,
    branch_periods_inherit_carry,
    fan_period_labels,
    inherit_active_branch_periods,
    period_branch_pairs_lf,
    period_in_use_set_lf,
)
from flextool.engine.preprocess.params._direct import (
    _e_period_param_union,
    pdtReserve_upDown_group_reservation_from_source,
)
from flextool.engine.preprocess.params._per_solve_sets import (
    solve_period_timeset_frame,
    timeset_duration_lf,
    timeset_timeline_lf,
)
from flextool.engine.input._solve_state import CommissioningLagError, FlexToolConfigError, LineageFilterError
from flextool.engine.input._solve_step_aggregation import aggregate_if_active
from flextool.engine.input._param_shapes import (
    broadcast_to_period_time,
    parameter_has_values,
    promote_param_to_dt,
    resolve_param_shape,
    solve_years_represented,
)
from flextool.engine.preprocess.params._existing import (
    DIVEST_NOT_ALLOWED as _DIVEST_NOT_ALLOWED,
    INVEST_METHODS_ALL as _INVEST_METHODS_ALL,
    INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED,
)


def _provider_has_key(provider, path: "Path | str") -> bool:
    """Provider-only existence check.  Returns ``True`` iff the live
    :class:`FlexDataProvider` carries the artefact for *path* (under
    its canonical name); ``False`` when *provider* is ``None`` or the
    key is missing.  The disk-fallback arm was removed in Step 2.5 —
    cascade modules consume frames exclusively from the Provider.
    """
    if provider is None:
        return False
    return provider.has(_provider_key(path))


def _provider_read(provider, path: "Path | str") -> "pl.DataFrame":
    """Provider-only read.  Returns the frame for *path* (under its
    canonical key) from the Provider.  Caller must guard with
    :func:`_provider_has_key` first; calling with ``provider=None``
    raises :class:`ValueError`.
    """
    if provider is None:
        raise ValueError(
            "_provider_read requires a FlexDataProvider; guard with "
            "_provider_has_key first."
        )
    return provider.get(_provider_key(path))


def _recourse_invest_active(
    workdir: "Path | None",
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> bool:
    """True iff ``solve_data/stochastic_invest_method.csv`` == ``'recourse'``.

    Slice D's single gate.  False when the CSV is absent (deterministic /
    flag-off / pre-Slice-D dumps) — the byte-parity-safe default.

    Provider-only read (canonical key), matching every other reader in
    this module (``_build_recourse_anchor_pairs`` et al.) and the post-
    Step-2 cascade invariant (``test_meta_provider_invariants``): the
    flag CSV reaches the cascade exclusively through the Provider, never
    from disk.  On the snapshot-reload path (``load_flextool`` off a
    dumped workdir) the Provider is seeded from ``solve_data/`` before
    the cascade runs (``_csv_reload.load_from_dump``), so it carries
    this key by construction;
    cascade entry points always thread an explicit Provider.  Tests that
    exercise the flag seed the Provider from the workdir the same way.
    ``workdir`` is retained in the signature only to key the canonical
    Provider path (``_provider_key`` ignores the prefix, so the key is
    stable whether or not a workdir is supplied).
    """
    path = (
        Path(workdir) / "solve_data" / "stochastic_invest_method.csv"
        if workdir is not None
        else Path("solve_data") / "stochastic_invest_method.csv"
    )
    if not _provider_has_key(provider, path):
        return False
    df = _provider_read(provider, path)
    if df is None or df.height == 0 or "method" not in df.columns:
        return False
    val = df["method"].cast(pl.Utf8, strict=False).to_list()[0]
    return str(val).strip().lower() == "recourse" if val is not None else False


# ─── Slice D (B, §4) — per-period parameter anchor-mapping ─────────────
# Module-level holder for the (anchor, br) synthetic-branch pairs of the
# ACTIVE solve.  Set fresh at the top of every Layer-4 apply boundary
# (``apply_derived_c`` / ``apply_npv`` / ``apply_synthetic_invest_sets``)
# to the recourse pairs, or ``None`` when the flag is off / deterministic.
# Read by the explicit-per-period producers (_resolve_per_period_lf and
# _ed_period_param_lf, which serves the zero-cost investment notice and
# p_entity_max_units_from_source) to inherit each
# anchor's per-(e,d) value onto its branch periods (§4.1).  ``None`` ==
# no-op == flag-off byte-parity.  The set-at-boundary-top invariant means
# a stale value can never reach a producer: every boundary overwrites it
# before any producer runs, and producers run nowhere else.
_RECOURSE_ANCHOR_PAIRS: "pl.DataFrame | None" = None

# Boundary-scoped Provider (+ its workdir, which the canonical readers need
# to key the Provider path) for the period walks' year / factor arms.  Set at
# each Layer-4 boundary alongside the anchor pairs whenever the solve has a
# stochastic fan (branch periods), under every ``stochastic_invest_method``,
# so ``period_walk_iterator`` and ``_existing._solve_walk_years`` read the
# canonical ``p_years_d.csv`` / ``p_years_represented.csv`` arms, which give
# each branch period the years of its period label, WITHOUT threading
# ``provider``/``workdir`` through the NPV/edd walker-caller signatures.
# ``None`` (a solve without branch periods) → the walks read the solve's own
# ``years_represented`` (its periods are all labels, so the years agree).
_FAN_WALK_PROVIDER: "object | None" = None
_FAN_WALK_WORKDIR: "Path | None" = None


def _fan_walk_provider() -> "object | None":
    """The active boundary's walker Provider of a solve with branch periods, or ``None``."""
    return _FAN_WALK_PROVIDER


def _fan_walk_workdir() -> "Path | None":
    """The active boundary's walker workdir of a solve with branch periods, or ``None``."""
    return _FAN_WALK_WORKDIR


def _recourse_branch_axis_present() -> bool:
    """True iff the active boundary is recourse-active with a genuine branch
    invest axis — i.e. the anchor-pairs holder was armed non-empty.  The
    capability conjunct (§2), computed from the provider (branch-cluster
    frames on ``flex_data`` are not populated until derived_g)."""
    return _RECOURSE_ANCHOR_PAIRS is not None


def _build_recourse_anchor_pairs(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """(anchor, br) pairs for in-use synthetic branches under recourse.

    Returns ``None`` when the recourse flag is off (byte-parity default);
    the pairs themselves are :func:`_fan_branch_pairs`.
    """
    if not _recourse_invest_active(workdir, ctx=ctx, provider=provider):
        return None
    return _fan_branch_pairs(workdir, ctx=ctx, provider=provider)


def _fan_branch_pairs(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """(anchor, br) pairs of the solve's in-use synthetic branch periods,
    under every ``stochastic_invest_method``; ``None`` without any.

    Mirrors :func:`_expand_branch_periods`' acquisition (ctx-first, then
    provider/workdir ``period__branch`` + ``period_in_use``); keeps only
    synthetic (``anchor != br``) branches that carry LP variables (``br``
    in ``period_in_use``).
    """
    pb_df: "pl.DataFrame | None" = None
    in_use: set[str] = set()
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            pb_df = pb_ctx.rename({"d_anchor": "anchor", "b": "br"})
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
    if pb_df is None:
        if workdir is None:
            return None
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if not _provider_has_key(provider, p):
            return None
        pb_df = _provider_read(provider, p)
        if pb_df.height == 0:
            return None
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):
                piu = _provider_read(provider, piu_path)
                if piu.height > 0:
                    in_use = set(piu["period"].to_list())
        pb_df = pb_df.rename({"period": "anchor", "branch": "br"})
    out = (pb_df.select(
                pl.col("anchor").cast(pl.Utf8),
                pl.col("br").cast(pl.Utf8))
               .filter(pl.col("anchor") != pl.col("br")))
    if in_use:
        out = out.filter(pl.col("br").is_in(list(in_use)))
    out = out.unique()
    return out if out.height > 0 else None


def _fan_walk_scope_provider(workdir: Path | None, *, provider: "object | None" = None) -> "object | None":
    """``provider`` when the solve has branch periods or is under recourse (the walker Provider
    :func:`_enter_recourse_anchor_scope` arms), else ``None``: for a reader outside a Layer-4 boundary."""
    if _recourse_invest_active(workdir, provider=provider) or _fan_branch_pairs(workdir, provider=provider) is not None:
        return provider
    return None


def _reset_recourse_scope() -> None:
    """Clear all three recourse process-globals to ``None``.

    Defensive per-solve reset: called once at the top of the cascade
    entry (:func:`flex_data._apply_db_overrides`) BEFORE any pass runs, so a
    future per-period producer wired into an EARLIER cascade step cannot
    inherit a stale anchor-pairs holder / walker Provider from a prior
    flag-on solve in a chained flag-on→flag-off multi-solve.  The three
    flag-aware Layer-4 boundaries still overwrite these unconditionally
    (to ``None`` when flag-off), so this reset is belt-and-suspenders;
    the invariant it guards is "no producer sees a global from a
    different solve".
    """
    global _RECOURSE_ANCHOR_PAIRS, _FAN_WALK_PROVIDER, _FAN_WALK_WORKDIR
    _RECOURSE_ANCHOR_PAIRS = None
    _FAN_WALK_PROVIDER = None
    _FAN_WALK_WORKDIR = None


def _enter_recourse_anchor_scope(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> None:
    """Set :data:`_RECOURSE_ANCHOR_PAIRS` for the active solve.

    Called at the top of each Layer-4 apply boundary BEFORE any explicit-
    per-period producer OR walker runs.  Overwrites unconditionally (to
    ``None`` when flag-off) so no stale value from a prior solve can leak.
    Also arms the boundary-scoped walker Provider whenever the solve has
    branch periods (any ``stochastic_invest_method``) or is under recourse,
    so every period walk gives a branch period its label's years; the
    anchor pairs stay recourse-only.
    """
    global _RECOURSE_ANCHOR_PAIRS, _FAN_WALK_PROVIDER, _FAN_WALK_WORKDIR
    active = _recourse_invest_active(workdir, ctx=ctx, provider=provider)
    pairs = _fan_branch_pairs(workdir, ctx=ctx, provider=provider)
    walk = active or pairs is not None
    _FAN_WALK_PROVIDER = provider if walk else None
    _FAN_WALK_WORKDIR = (
        Path(workdir) if walk and workdir is not None else None)
    _RECOURSE_ANCHOR_PAIRS = pairs if active else None


def _anchor_expand_explicit(explicit_lf: pl.LazyFrame) -> pl.LazyFrame:
    """§4.1 — add branch rows inheriting each anchor's per-(e, d) value.

    ``explicit_lf`` is an ``(e, d, <value…>)`` lazy frame keyed by REAL
    anchor periods (any value-column name(s); all are preserved).  When
    the recourse anchor-pairs holder is set, each
    ``(e, anchor, v)`` row spawns ``(e, br, v)`` for every in-use branch
    ``br`` of ``anchor``; a branch that already carries its OWN explicit
    row keeps it (branch-native wins, via order-preserving dedupe).
    Holder ``None`` (flag-off / deterministic) → returns the frame
    unchanged (byte-parity).  Operates in Utf8 on ``d`` then restores the
    original dtype.  The Utf8→Enum round-trip is lossless for anchor rows
    by construction (their ``d`` arrived already typed as ``d_dtype``, so
    each token is in-vocab), and every inherited branch (``br``) token is
    covered by the 4.0.4 vocabulary splice; a nulled token therefore
    signals a wiring bug and is hard-raised (mirroring
    :func:`._branch.d_leaf_lf`) rather than silently dropped.
    """
    pairs = _RECOURSE_ANCHOR_PAIRS
    if pairs is None or pairs.height == 0:
        return explicit_lf
    d_dtype = explicit_lf.collect_schema().get("d")
    base = explicit_lf.with_columns(pl.col("d").cast(pl.Utf8))
    inherited = (base.join(pairs.lazy(),
                              left_on="d", right_on="anchor", how="inner")
                     .drop("d")
                     .rename({"br": "d"})
                     .select(base.collect_schema().names()))
    combined = pl.concat([base, inherited], how="vertical_relaxed")
    out = combined.unique(subset=["e", "d"], keep="first", maintain_order=True)
    if d_dtype is not None and d_dtype != pl.Utf8:
        # Restore the original (non-Utf8, i.e. Enum) ``d`` dtype.  A
        # ``strict=False`` cast would SILENTLY null any token absent from
        # the live ``d``-axis vocabulary; hard-raise instead so a splice
        # gap surfaces loudly (never fires in valid runs — see docstring).
        out_df = out.collect()
        casted = out_df.with_columns(pl.col("d").cast(d_dtype, strict=False))
        if casted["d"].null_count():
            bad = sorted({v for v, c in zip(out_df["d"].to_list(),
                                            casted["d"].to_list())
                          if c is None})
            raise ValueError(
                f"_anchor_expand_explicit: axis-enum cast nulled period "
                f"token(s) {bad} — the token is missing from the live "
                "d-axis vocabulary (wiring bug; the 4.0.4 vocabulary splice "
                "includes every continuation branch-period token)."
            )
        return casted.lazy()
    return out


def _has_branch_invest_axis(flex_data: "object | None") -> bool:
    """Capability conjunct (Slice B §8, design §2): True iff the invest
    axis actually carries synthetic branch members.

    Reuses :func:`_stochastic_detect.is_genuinely_stochastic` — a branch
    period enters ``period_invest`` iff it is a synthetic ``period__branch``
    token that is also in ``period_in_use_set``.  Keeps the lineage filter
    provably inert on any solve whose invest axis carries no branch member
    even if the flag was authored (defense-in-depth behind Guard 1).
    """
    if flex_data is None:
        return False
    from flextool.engine.preprocess._stochastic_detect import (
        is_genuinely_stochastic as _igs,
    )
    return _igs(flex_data)


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

# Canonical unitsize fallback for the ``virtual_unitsize OR existing OR
# <fallback>`` cascade.  NOTE: this is a FlexTool *modeling* default, not a
# schema ``default_value`` — the SpineDB schema declares ``virtual_unitsize``
# with default ``null`` (see ``flextool/schemas/spinedb_schema.json``); the
# 1000.0 fallback lives only in flextool's preprocessing
# (``entity_period_calc_params.py``).  Defined here so the cascade tails
# (:func:`_entity_unitsize_lf` / :func:`_node_unitsize_lf`) and the
# structural-completeness projection (``_arithmetic._unitsize_complete``)
# share ONE source of truth and cannot drift.
UNITSIZE_DEFAULT = 1000.0

# Memoization caches for the solve-INVARIANT unitsize cascades.  Both
# ``_entity_unitsize_lf`` and ``_node_unitsize_lf`` compute a pure
# coalesce (``virtual_unitsize OR existing OR 1000``) purely from
# ``source``; the whole-model ``SpineDbReader`` is reused across all
# rolls, so caching the collected cascade keyed on the source object
# collapses N per-roll recomputes to one.  WeakKeyDictionary so the
# entry is dropped when the source is garbage-collected.
_ENTITY_UNITSIZE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_NODE_UNITSIZE_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_param(source: "InputSource", entity_class: str,
                parameter_name: str) -> pl.DataFrame | None:
    """Return ``source.parameter(...)`` or ``None`` if class unknown /
    parameter unknown / empty.  Mirrors ``_projection._try_param``.
    """
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_param_explicit(source: "InputSource", entity_class: str,
                         parameter_name: str) -> pl.DataFrame | None:
    """Like :func:`_try_param` but suppresses default-broadcast rows.

    Mirrors flextool's ``p_unit.get(name, None)`` semantic: returns only
    entities with a parameter_value row in the active scenario.  Falls
    back to the public :meth:`InputSource.parameter` when the source
    has no ``parameter_explicit`` method (legacy custom InputSource
    implementations).
    """
    fn = getattr(source, "parameter_explicit", None) or source.parameter
    try:
        df = fn(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _try_entities(source: "InputSource", entity_class: str) -> pl.DataFrame | None:
    try:
        df = source.entities(entity_class)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _scalar_default(source: "InputSource", entity_class: str,
                     parameter_name: str, fallback: Any) -> Any:
    """Look up the scalar Spine default; fall back to *fallback* when
    the source returns ``None`` (None-default policy).
    """
    try:
        d = source.parameter_default(entity_class, parameter_name)
    except KeyError:
        return fallback
    return fallback if d is None else d


def _read_active_solve(workdir: Path, *, provider) -> str | None:
    """Read ``solve_data/solve_current.csv`` and return the active solve
    name, or ``None`` when the Provider doesn't carry it / it's empty.
    """
    if provider is None:
        raise ValueError(
            "_read_active_solve requires a FlexDataProvider; the disk-"
            "fallback arm was removed in Step 2.5."
        )
    p = Path(workdir) / "solve_data" / "solve_current.csv"
    if not provider.has(_provider_key(p)):
        return None
    df = provider.get(_provider_key(p))
    if df is None or df.height == 0:
        return None
    col = df.columns[0]
    return df[col][0]


def _solve_in_spine(source: "InputSource",
                      active_solve: str | None) -> bool:
    """Return True iff ``active_solve`` appears as a row in Spine's
    ``solve`` entity class.

    Δ.18 — synthetic per-sub-solve names (e.g. ``invest_5weeks_p2020``
    for nested-multi-invest fixtures) don't exist in Spine; the per-solve
    override chain (``apply_derived_a`` etc.) returns None for every
    helper that takes ``active_solve`` as a key.  Detecting this upfront
    lets the loader keep the snapshot-CSV-loaded values as authoritative.

    Probes the ``solve`` entity table directly — relying on a per-solve
    parameter (e.g. ``invest_periods``) was incorrect because some solves
    have empty ``invest_periods`` (dispatch-only) yet are valid solves.
    """
    if active_solve is None:
        return False
    try:
        ents = source.entities("solve")
    except KeyError:
        return False
    if ents is None or ents.height == 0:
        return False
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    return ents.filter(pl.col(name_col) == active_solve).height > 0


def _ctx_read(
    ctx: "SolveContext | None",
    workdir: Path | str | None,
    name: str,
    *,
    kind: str = "solve_data",
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Read the per-solve frame *name* (``<kind>/<stem>``) from *provider*.

    *provider* is required: ``None`` raises ``ValueError`` (no work-folder
    read).  Key present → the frame (an empty body → ``pl.DataFrame()``);
    key absent, or no *workdir* to form the key from → ``None`` (the
    callers' "no frame" branch).  *ctx* is unused; it stays in the
    signature for call-site stability.
    """
    if provider is None:
        raise ValueError(
            f"_ctx_read({name!r}) requires a FlexDataProvider; it never reads the work folder on disk."
        )
    if workdir is None:
        return None
    if kind == "solve_data":
        path = Path(workdir) / "solve_data" / name
    elif kind == "input":
        path = Path(workdir) / "input" / name
    else:
        path = Path(workdir) / name
    if not _provider_has_key(provider, path):
        return None
    try:
        return _provider_read(provider, path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


# ---------------------------------------------------------------------------
# §3.1.1 — dt + p_step_duration
# ---------------------------------------------------------------------------


def dt_and_step_duration_from_source(
    source: "InputSource",
    active_solve: str,
    workdir: Path | None = None,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> tuple[pl.DataFrame, Param] | None:
    """Compute the per-(d, t) ``dt`` set and ``p_step_duration`` Param
    for the active solve.

    Algorithm (audit §3.1.1):

      1. ``solve.realized_periods`` → list of periods active in this solve.
      2. ``solve.period_timeset[active_solve, period]`` → timeset for each
         (or the Rule 1 / 5b default, :func:`solve_period_timeset_frame`).
      3. ``timeset.timeline[ts]`` → which timeline supplies step_duration.
      4. ``timeset.timeset_duration[ts]`` is a 1d_map ``(start_step → count)``
         describing one or more contiguous timestep blocks.
      5. For each (start_step, count) emit *count* consecutive timesteps
         starting at ``start_step`` in the timeline's step ordering.
      6. ``step_duration = timeline.timestep_duration[timeline_name, step]``.

    Per-solve Params; ``active_solve`` is required.

    Returns ``None`` if the active solve has no realized periods (e.g.
    the solve doesn't exist in the source) — caller falls back to CSV.
    """
    # Build the period_in_use set.  When the workdir is supplied AND
    # ``solve_data/period_in_use_set.csv`` exists, prefer that — it's the
    # authoritative set INCLUDING any stochastic-branch periods (mirrors
    # flextool's per_solve_sets.py:95-101 which writes branch periods on
    # top of realised + invest periods).  Otherwise fall back to the
    # union of realized_periods + invest_periods.
    realized_p = None
    if ctx is not None and ctx.period_in_use.height > 0:
        realized_p = ctx.period_in_use.lazy().select("d").unique()
    elif workdir is not None:
        piu = _ctx_read(ctx, workdir, "period_in_use_set.csv", provider=provider)
        if piu is not None and piu.height > 0 and "period" in piu.columns:
            realized_p = (piu.lazy()
                            .select(alias_to_axis("period", "d"))
                            .unique())
    if realized_p is None:
        parts: list[pl.LazyFrame] = []
        _live = get_global_axis_enums()
        for ec, par in (("solve", "realized_periods"),
                         ("solve", "invest_periods")):
            df = _try_param(source, ec, par)
            if df is None:
                continue
            parts.append(df.lazy()
                           .filter(pl.col("name") == active_solve)
                           .select(alias_to_axis("value", "d")))
        if not parts:
            return None
        realized_p = pl.concat(parts).unique()

    # period → timeset of the active solve: ``solve.period_timeset``, or the
    # Rule 1 / 5b default of ``create_assumptive_parts`` (item #175).
    pt_frame = solve_period_timeset_frame(source, active_solve)
    if pt_frame is None:
        return None
    pt = pt_frame.lazy().select(alias_to_axis("d", "d"), pl.col("ts"))
    # Broadcast timesets to stochastic-branch periods: when a period
    # appears in period_in_use_set without a direct period_timeset row,
    # use the anchor's timeset via ``period__branch.csv`` (anchor → branch
    # map; flextool's per_solve_sets.py:65-95 does this implicitly by
    # writing per-branch steps_in_use rows under the anchor's timeset).
    pb_raw = _ctx_read(ctx, workdir, "period__branch.csv", provider=provider) if workdir is not None or ctx is not None else None
    if pb_raw is not None and pb_raw.height > 0:
        pb_lf = (pb_raw.lazy()
                    .pipe(rename_to_axis, {"period": "anchor", "branch": "d"})
                    # Cross-axis-by-value compare: "anchor" and "d"
                    # both resolve to the canonical d (period) enum, so
                    # the Enum vocabularies match.  Cast to Utf8 here
                    # explicitly so the equality test stays robust if
                    # the vocabularies ever diverge (e.g. branch axis
                    # gains a stochastic vocabulary distinct from d).
                    .filter(pl.col("anchor").cast(pl.Utf8)
                            != pl.col("d").cast(pl.Utf8)))
        # Map branch d → anchor's timeset.  Pre-snap both branches'
        # dim-column dtypes to the canonical Enum so the concat below
        # sees identical schemas (Polars's UNION schema-resolution
        # otherwise drops one side to Utf8, triggering a SchemaError
        # at the downstream join).
        anchor_ts = (pt.rename({"d": "anchor"})
                        .join(pb_lf, on="anchor", how="inner")
                        .select("d", "ts"))
        _live = get_global_axis_enums()
        # Materialise both halves eagerly before concat so Polars's
        # lazy union schema-resolver doesn't see two large optimiser
        # contexts at once — that path triggers an internal
        # str-vs-Enum mis-resolution on the ``d`` column when both
        # halves transitively reference ``period_timeset`` with cast
        # nodes.  Eager concat + re-cast yields the canonical Enum
        # dtype reliably; the upstream cost is negligible (pt + anchor
        # are O(num_periods) rows).
        pt_e = pt.collect()
        at_e = anchor_ts.collect()
        if _live is not None:
            pt_e = cast_frame_axes(pt_e, _live)
            at_e = cast_frame_axes(at_e, _live)
        pt = pl.concat([pt_e, at_e]).unique().lazy()
        # A period label (the original name of a period that opens a fan) is in period_in_use_set for the
        # period-level bookkeeping but has no dispatch: it gets no (d, t) rows.
        labels = _labels_of_pairs(pb_raw["period"].to_list(), pb_raw["branch"].to_list())
        if labels:
            realized_p = realized_p.filter(~pl.col("d").cast(pl.Utf8).is_in(sorted(labels)))

    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    # timeline.timestep_duration: name=timeline, <step_col>, value=duration.
    tl_cols = tl_dur.columns
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_cols and c != "name" and c != "value"),
                        None)
    if tl_step_col is None:
        return None
    tl = (tl_dur.lazy()
                .select(pl.col("name").alias("timeline"),
                        alias_to_axis(tl_step_col, "t"),
                        pl.col("value").cast(pl.Float64).alias("step_duration")))
    # Determine step rank within each timeline (sort by t lex order, which
    # matches t0001 < t0002 < ... in flextool's canonical naming).
    tl = tl.sort(["timeline", "t"]) \
            .with_columns(rank=pl.col("t").cum_count().over("timeline"))

    # ``timeset.timeline`` / ``timeset.timeset_duration``, through the
    # shared single-timeline defaults (item #171, Rules 2/3 — see
    # ``_per_solve_sets.timeset_timeline_lf`` / ``timeset_duration_lf``).
    # ``timesets`` is the set of timesets the active solve's (branch-
    # broadcast) ``period_timeset`` actually names.
    timesets = pt.select("ts").unique()
    ttl = timeset_timeline_lf(source, timesets, tl)
    if ttl is None:
        return None

    ts_dur_lf = timeset_duration_lf(source, timesets, tl)
    if ts_dur_lf is None:
        return None
    # ``start_step`` joins against ``tl.t`` which is t-Enum under
    # activation.  The shared helper returns Utf8; cast explicitly to
    # t-Enum and alias to "start_step" (caller-side cast, per design).
    _live_enums = get_global_axis_enums()
    blocks = ts_dur_lf.select(
        pl.col("ts"),
        cast_dim(pl.col("start_step"), _live_enums, "t").alias("start_step"),
        pl.col("count"),
    )

    # Build (period, ts, timeline, start_step, count) frame, then expand.
    realized_p_with_ts = realized_p.join(pt, on="d", how="inner")
    pst = (realized_p_with_ts
            .join(ttl, on="ts", how="inner")
            .join(blocks, on="ts", how="inner"))

    # For each (period, timeline, start_step, count), emit the timeline
    # rows whose rank ∈ [start_rank, start_rank + count - 1].  Resolve
    # start_rank by joining timeline ranks on (timeline, t=start_step).
    start_ranks = (pst.join(
        tl.select(pl.col("timeline"),
                  pl.col("t").alias("start_step"),
                  pl.col("rank").alias("start_rank")),
        on=["timeline", "start_step"],
        how="left",
    ))
    # Drop blocks whose start_step isn't in the timeline (mis-spec).
    start_ranks = start_ranks.filter(pl.col("start_rank").is_not_null())

    # Cross-join with tl on timeline, then filter rank-in-window.
    expanded = (start_ranks
                  .join(tl, on="timeline", how="inner")
                  .filter((pl.col("rank") >= pl.col("start_rank"))
                          & (pl.col("rank") < pl.col("start_rank")
                              + pl.col("count").cast(pl.Int64))))
    out = (expanded
            .select("d", pl.col("t"),
                    pl.col("step_duration").alias("value"))
            .unique()
            .sort("d", "t")
            .collect())
    if out.height == 0:
        return None
    dt = out.select("d", "t")
    step_dur = Param(("d", "t"), out)
    return dt, step_dur


# ---------------------------------------------------------------------------
# §3.1.4 — p_period_share
# ---------------------------------------------------------------------------


def p_period_share_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    step_dur: Param,
) -> Param | None:
    """``p_period_share[d] = sum_{t in dt[d]} step_duration[d,t] / 8760`` — the
    fraction of a full year covered by this period in the active solve.

    Audit §3.1.4: built from ``dt`` and ``p_step_duration``, over the steps of
    ``dt`` only (``step_dur`` rows outside ``dt`` are ignored; the keys are
    matched as strings so Enum and Utf8 key columns align).
    """
    if dt is None or step_dur is None:
        return None
    keys = dt.select(pl.col("d").cast(pl.Utf8).alias("_d"),
                     pl.col("t").cast(pl.Utf8).alias("_t")).unique()
    sd = (step_dur.frame
          .with_columns(pl.col("d").cast(pl.Utf8).alias("_d"),
                        pl.col("t").cast(pl.Utf8).alias("_t"))
          .join(keys, on=["_d", "_t"], how="semi", maintain_order="left")
          .drop("_d", "_t"))
    out = (sd.lazy()
              .group_by("d")
              .agg((pl.col("value").cast(pl.Float64).sum() / HOURS_PER_YEAR)
                   .alias("value"))
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


def p_window_fraction_from_shares(
    dt: pl.DataFrame,
    step_dur: Param,
    period_share: Param,
) -> Param | None:
    """``p_window_fraction[d] = (Σ_{t in dt[d]} step_duration[d,t] / 8760) / p_period_share[d]``.

    The share of each period's complete-solve hours that the sub-solve's own ``dt`` covers (the
    window sum runs over the steps of ``dt`` only, see :func:`p_period_share_from_source`).  In a
    roll of a rolling solve ``p_period_share`` is the complete (parent) solve's share, so the
    fraction is the roll's window over the complete solve, per period.  A value within 1e-9
    (relative) of 1 is snapped to exactly 1.0, so a sub-solve that covers its periods whole keeps
    the ``co2_max_period`` RHS bit for bit.  A dt period without a ``p_period_share`` row raises: the
    share is inconsistent with dt (a loader bug, or a reloaded workdir whose share file lacks the
    period), and the operational costs (``op_factor``) and the cap's RHS of that period would be lost.
    """
    window = p_period_share_from_source(None, dt, step_dur)
    if window is None:
        return None
    share = period_share.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("value").cast(pl.Float64).alias("_share"))
    out = (window.frame.with_columns(pl.col("d").cast(pl.Utf8).alias("_d"))
           .join(share, on="_d", how="left"))
    missing = out.filter(pl.col("_share").is_null())["_d"].to_list()
    if missing:
        raise ValueError(
            f"p_window_fraction: p_period_share has no row for dt periods {missing}; the period share (the "
            f"complete solve's share in a sub-solve, complete_period_share_of_year_calc.csv in a reloaded workdir) "
            f"must cover every dt period, else that period's operational costs and co2_max_period cap lose "
            f"their weight.")
    out = out.with_columns(
        value=pl.when((pl.col("value") - pl.col("_share")).abs() <= 1e-9 * pl.col("value").abs())
        .then(pl.lit(1.0, dtype=pl.Float64))
        .otherwise(pl.col("value") / pl.col("_share")))
    return Param(("d",), out.select("d", "value").sort("d"))


# ---------------------------------------------------------------------------
# p_years_represented_d — per-period R width sum
# ---------------------------------------------------------------------------


def periods_of_dt(dt: "pl.DataFrame | None") -> list[str]:
    """The distinct periods of ``dt`` (column ``d``) as strings, in dt order."""
    if dt is None or dt.height == 0:
        return []
    return [str(x) for x in dt["d"].unique(maintain_order=True).to_list()]


def _labels_of_pairs(periods: list, branches: list) -> set[str]:
    """The period labels (:func:`._branch.fan_period_labels`) of ``period__branch`` given as two columns."""
    return fan_period_labels({(str(a), str(b)) for a, b in zip(periods, branches)})


def bookkeeping_periods(dt: "pl.DataFrame | None", workdir, *, ctx=None,
                        provider: "object | None" = None) -> list[str]:
    """The periods of ``dt`` (in dt order) followed by the solve's period labels (sorted).

    A period label (the original name of a period that opens a stochastic fan) has no (d, t) rows but is a period
    of the solve for period-level bookkeeping (years, investment costs).
    """
    dt_periods = periods_of_dt(dt)
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb["d"].to_list(), pb["b"].to_list())
    if not labels:
        return dt_periods
    piu = {str(d) for d in period_in_use_set_lf(workdir, ctx=ctx, provider=provider).collect()["d"].to_list()}
    return dt_periods + sorted(d for d in labels if d in piu and d not in dt_periods)


def p_years_represented_d_from_source(
    source: "InputSource",
    active_solve: str | None,
    period_set: list[str],
    *,
    provider: "object | None" = None,
) -> Param | None:
    """Per-period ``years_represented_d``: sum of per-year widths
    bound to each period (the ``R`` value flextool's
    ``solve.years_represented`` Map carries — e.g. 5.0 for a 5-year
    investment period).

    ``period_set`` is the periods of the solve's ``dt``, in dt order.
    With ``provider``, the widths come from the canonical
    ``solve_data/p_years_represented.csv``, which gives every stochastic
    branch period its anchor period's years and, on a rolling or
    synthetic sub-solve, the parent solve's years; the source arm is the
    fallback (see ``_years_for_period_from_source``).

    Returns ``None`` when no period set is available; otherwise emits
    one row per period.
    """
    if not period_set:
        return None
    yfp = _years_for_period_from_source(source, active_solve, period_set,
                                        provider=provider)
    rows = []
    for d in period_set:
        widths = yfp.get(d, [])
        # Empty period — emit 0.0 (matches legacy ``sum(())``); upstream
        # consumers (out_costs) multiply by emissions so a 0 period
        # contributes nothing.  This hits periods explicitly set to R=0
        # (and periods the canonical CSV omits for that reason); the
        # source arm gives a period absent from ``years_represented`` a
        # single (label, 1.0) row.
        s = sum(w for _y, w in widths)
        rows.append({"d": str(d), "value": float(s)})
    out = pl.DataFrame(rows, schema={"d": pl.Utf8, "value": pl.Float64})
    if out.height == 0:
        return None
    return Param(("d",), out)


# ---------------------------------------------------------------------------
# §3.1.3 — p_inflation_op
# ---------------------------------------------------------------------------


def p_inflation_op_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Period-level inflation factor for operations.

    Audit §3.1.3 says the formula compounds ``(1 + inflation_rate)^...``;
    the actual flextool implementation
    (``period_calculated_params.py:280-322``) computes
    ``sum_y p_years_represented[d,y] * (1+rate)^until_op[d,y]``
    where ``until_op[d, y] = base[d,y] + pyr[d,y] * offset_op``,
    ``base[d, y] = sum_{y'<y} pyr[d, y']``.

    Γ.8.E coverage:
      * ``model.inflation_rate ∈ {None, 0}`` (the default) — formula
        collapses to ``ops_factor[d] = Σ_y pyr[d, y] * 1 =
        years_represented[solve, d]`` since each period has a single
        flat scalar in the 1D-Map shape exposed by InputSource (each
        scalar IS the total summed over the period's year-set).
      * Multi-year-per-period (``years_represented`` set per solve,
        rate=0) — handled here, mirrors the regression that
        ``work_fullYear_roll`` exposes when CSV retirement Step 3 lands.

    Non-trivial-rate (``rate != 0``) and ``inflation_offset`` cascade
    are handled by :func:`p_inflation_op_multi_year_from_source`
    (Γ.3.C, applied in :func:`apply_derived_c`).

    Returns ``None`` only when no solve / no dt is available; the
    helper otherwise produces the canonical frame so CSV retirement
    won't reintroduce the simple-1.0 regression.
    """
    if dt is None:
        return None
    rate = _try_param(source, "model", "inflation_rate")
    rate_v = 0.0
    if rate is not None and rate.height > 0:
        rate_v = float(rate["value"][0])
    if rate_v != 0.0:
        # Non-trivial rate handled in the multi-year cascade helper
        # (which runs later in apply_derived_c).  Returning None
        # here lets the simple-default 1.0 stay in flex_data; the
        # later helper overlays the correct factor.
        return None

    # Rate = 0.  ``ops_factor[d] = Σ_y pyr[d, y]``.  The 1D Map shape
    # InputSource exposes for ``solve.years_represented`` already
    # collapses the inner year axis into a per-period scalar — so
    # that scalar IS the sum we need.
    #
    # Rolling solves: ``active_solve`` in the workdir is the per-roll
    # name (e.g. ``dispatch_fullYear_roll_roll_71``) but the DB has
    # ``years_represented`` keyed by the parent solve
    # (``dispatch_fullYear_roll``).  flextool's preprocessing copies
    # the parent's value down to every roll; we mirror that here
    # by trying the roll name first, then the parent name (strip
    # ``_roll_<N>`` suffix).
    yr = None
    if active_solve is not None:
        yr_raw = solve_years_represented(source)
        if yr_raw is not None:
            candidate_names = [active_solve]
            import re
            parent = re.sub(r"_roll_\d+$", "", active_solve)
            if parent != active_solve:
                candidate_names.append(parent)
            yr_lf = None
            for cand in candidate_names:
                lf = (yr_raw.lazy()
                            .filter(pl.col("name") == cand)
                            .select(alias_to_axis("period", "d"),
                                    pl.col("value").cast(pl.Float64).alias("yr_total")))
                if lf.collect().height > 0:
                    yr_lf = lf
                    break
            if yr_lf is not None:
                d_unique_lf = dt.lazy().select("d").unique()
                from flextool.engine.input._axis_enums import align_join_dtypes
                d_unique_lf, yr_lf = align_join_dtypes(
                    d_unique_lf, yr_lf, ["d"])
                yr_joined = (d_unique_lf.join(yr_lf, on="d", how="left")
                                         .with_columns(
                                             value=pl.col("yr_total").fill_null(1.0)
                                         )
                                         .select("d", "value")
                                         .sort("d"))
                yr_collected = yr_joined.collect()
                if yr_collected.height > 0:
                    yr = yr_collected

    if yr is None:
        # Fallback: trivial 1.0 per realized period (no
        # ``years_represented`` row → defaults to 1 year per period).
        # Defensive re-cast: ensure d is canonical Enum on the Param.
        yr = (dt.lazy()
                .select(alias_to_axis("d", "d")).unique()
                .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                .sort("d")
                .collect())
    if yr.height == 0:
        return None
    return Param(("d",), yr)


# ---------------------------------------------------------------------------
# §3.1.2 — p_timestep_weight
# ---------------------------------------------------------------------------


def solve_has_rp_timeset(source: "InputSource", active_solve: str | None) -> bool:
    """True when a timeset in ``active_solve``'s ``period_timeset`` has ``representative_period_weights``.

    Such a solve's per-step weights come from the writer's ``solve_data/timestep_weight`` frame
    (``_solve_writers._compute_rp_frames``): the folded representative-period weights of its rp periods and
    the ``timeset_weights`` rows of its other periods, so no derived value may replace them.
    """
    if active_solve is None:
        return False
    rpw = _try_param_explicit(source, "timeset", "representative_period_weights")
    if rpw is None or rpw.height == 0 or "name" not in rpw.columns:
        return False
    pt = solve_period_timeset_frame(source, active_solve)
    if pt is None:
        return False
    active_ts = set(pt.get_column("ts").cast(pl.Utf8).to_list())
    rp_ts = {str(n) for n in rpw.get_column("name").to_list()}
    return bool(active_ts & rp_ts)


def p_timestep_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Per-(d, t) representative-period cost weight.

    Audit §3.1.2: defaults to 1.0 per (d, t); explicit overrides come
    from solve-level ``timeset_weights`` (flat per-step weighting,
    non-RP) and ``representative_period_weights`` (nested base→rep
    Map, RP scenarios) in flextool.

    Algorithm (mirrors
    :func:`._solve_writers.emit_timeset_cost_weight` for the non-RP
    path and :func:`._solve_writers.emit_rp_data` for the RP path):

    Non-RP path (``timeset_weights`` set on a timeset):
      1. For each ``(period, timeset)`` pair from
         ``solve.period_timeset`` filtered to *active_solve*, look up
         ``timeset.timeset_weights[timeset]`` — a flat
         ``{timestep: weight}`` mapping.
      2. For each active timestep ``t`` in the period, emit
         ``raw[t] = weights.get(t, 0.0)``.
      3. Normalise to sum-to-n_active over the period's active steps:
         ``scale = n_active / sum(raw)``, then per-(d, t)
         ``value = raw[t] * scale``.  This makes a uniform input
         reproduce the trivial default 1.0 per step.
      4. Periods without a ``timeset_weights`` row keep the trivial
         default 1.0.

    RP path (``representative_period_weights`` set):
      Representative-period weights are folded into
      ``solve_data/timestep_weight.csv`` by the writer
      (:func:`._solve_writers._compute_rp_frames`,
      ``w_r[rep]=Σ_base·n_rp/n_base``) and loaded into
      ``flex_data.p_timestep_weight`` by the CSV loader
      (``flex_data.py`` ``_load_time``).  When the ACTIVE solve's
      ``period_timeset`` references a timeset carrying
      ``representative_period_weights``, this helper returns ``None`` so
      the call site (guarded) PRESERVES that folded CSV value instead of
      clobbering it with the dense 1.0 default.  This keeps a single
      source of truth (``_compute_rp_frames``) shared by every consumer
      — objective, outputs, inflow scaling — rather than re-folding the
      weights against ``source`` here.

    The ``active_solve`` argument was added in Γ.8.E so the helper can
    filter ``solve.period_timeset`` to the right solve when running in
    multi-solve cascades.  Backward-compat: ``None`` falls back to a
    no-op (returns the dense 1.0 default) for any caller still on the
    Γ.3.A signature.
    """
    if dt is None:
        return None

    # Build the dense default as the baseline; any per-(d, t) override
    # from ``timeset_weights`` will replace it via a left-join.
    # Defensive re-cast: ensure d/t are canonical Enum on the Param.
    default_lf = (dt.lazy()
                    .select(alias_to_axis("d", "d"), alias_to_axis("t", "t"))
                    .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                    .sort("d", "t"))

    # Discover ``solve.period_timeset`` and ``timeset.timeset_weights``;
    # if either is unavailable / empty, fall through to the trivial
    # default.
    if active_solve is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None

    # period → timeset of the active solve: ``solve.period_timeset``, or the Rule 1 / 5b default
    # (item #175) -- so the sole timeset's ``timeset_weights`` apply without an authored mapping.
    pt = solve_period_timeset_frame(source, active_solve)
    if pt is None:
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    # A stochastic branch period is a copy of its anchor period: it takes the anchor's timeset (and so
    # its timeset_weights) through the active branch-anchor pairs, like every period-keyed input.
    period_timeset = inherit_active_branch_periods(
        pt.select(pl.col("d").cast(pl.Utf8), pl.col("ts")),
        group_keys=(), period_col="d")
    period_timeset_lf = period_timeset.lazy().select(alias_to_axis("d", "d"), pl.col("ts"))

    # ``timeset.timeset_weights``: a ``time`` Map (index_name ``time`` / ``t``, or the silent ``x``),
    # resolved and validated by the shape registry; any other shape raises.
    resolved_tw = resolve_param_shape(source, "timeset", "timeset_weights")
    tw = resolved_tw.frame if resolved_tw is not None else None
    if tw is not None and tw.height > 0:
        # A solve with new_stepduration: one weight per solve step, the step mean (_solve_step_aggregation).
        tw = aggregate_if_active(
            tw.select("name", resolved_tw.time_index_column, pl.col("value").cast(pl.Float64)),
            ("timeset", "timeset_weights"), time_col=resolved_tw.time_index_column, steps=dt.get_column("t"))
    if tw is None or tw.height == 0:
        # No flat ``timeset_weights`` — but this may be an RP solve.
        # ``representative_period_weights`` are folded into
        # ``solve_data/timestep_weight.csv`` by the writer
        # (``_solve_writers._compute_rp_frames``) and already loaded
        # into ``flex_data.p_timestep_weight`` by the CSV loader.  If the
        # ACTIVE solve's ``period_timeset`` references a timeset carrying
        # RP weights, return None so the guarded call site preserves that
        # folded value instead of clobbering it with the dense 1.0
        # default.  Filter to the active timesets so a stray RP param on
        # an UNUSED timeset cannot suppress the legitimate dense default
        # for a genuinely non-RP solve.
        rpw = _try_param_explicit(
            source, "timeset", "representative_period_weights")
        if rpw is not None and rpw.height > 0 and "name" in rpw.columns:
            active_ts = (period_timeset_lf
                         .select(pl.col("ts"))
                         .unique()
                         .collect()
                         .get_column("ts")
                         .to_list())
            rp_ts = set(rpw.get_column("name").to_list())
            if any(ts in rp_ts for ts in active_ts):
                return None
        out = default_lf.collect()
        return Param(("d", "t"), out) if out.height > 0 else None
    # Resolved ``timeset_weights`` frame: name (timeset), t (timestep), value (weight).
    weights_lf = (tw.lazy()
                    .select(pl.col("name").alias("ts"),
                            alias_to_axis(resolved_tw.time_index_column, "t"),
                            pl.col("value").cast(pl.Float64).alias("w_raw")))

    # Per-(d, t) raw weight: lookup via period_timeset → timeset_weights.
    # Periods whose timeset has weights produce non-null ``w_raw``;
    # periods without keep null and fall back to the trivial default 1.0.
    dt_lf = dt.lazy().select("d", "t")
    # Align the CSV-fresh helper frames to dt's dim dtypes (the loader
    # cast dt to Enum end-of-load).
    from flextool.engine.input._axis_enums import align_join_dtypes
    dt_lf, period_timeset_lf = align_join_dtypes(
        dt_lf, period_timeset_lf, ["d"])
    dt_lf, weights_lf = align_join_dtypes(dt_lf, weights_lf, ["t"])
    joined = (dt_lf
              .join(period_timeset_lf, on="d", how="left")
              .join(weights_lf, on=["ts", "t"], how="left"))

    # Per-period sum of raw weights and active-step count for scaling.
    # A period without any ``timeset_weights`` row sums to null/0 — the
    # downstream branch leaves ``value=1.0`` for those rows.
    by_period = (joined
                 .group_by("d")
                 .agg(pl.col("w_raw").fill_null(0.0).sum().alias("w_sum"),
                      pl.len().alias("n_active")))

    # Periods with weights: scale = n_active / w_sum, value = w_raw * scale.
    # Periods without weights: value = 1.0 (default).
    out_lf = (joined
              .join(by_period, on="d", how="left")
              .with_columns(
                  value=pl.when(
                      pl.col("w_sum").is_not_null() & (pl.col("w_sum") > 0)
                  ).then(
                      pl.col("w_raw").fill_null(0.0)
                      * pl.col("n_active").cast(pl.Float64)
                      / pl.col("w_sum")
                  ).otherwise(pl.lit(1.0))
              )
              .select("d", "t", "value")
              .sort("d", "t"))

    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("d", "t"), out)


# ---------------------------------------------------------------------------
# §3.18.1 — pdt_branch_weight / pd_branch_weight
# ---------------------------------------------------------------------------


def pd_branch_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
) -> Param | None:
    """Per-period branch weight — dense 1.0 baseline (superseded later).

    Returns the trivial 1.0 default per realized period.  This is the
    baseline assigned in ``apply_derived_a``; it is the final value only
    for deterministic / non-stochastic fixtures (where it matches the
    .mod's ``param pd_branch_weight`` default).

    For stochastic fixtures it is SUPERSEDED downstream: ``apply_derived_g``
    calls ``_branch.apply_branch_cluster``, whose
    ``pd_branch_weight_lf`` reads ``solve_branch_weight`` /
    ``period__branch`` and writes the real sibling-normalised weights
    (each branch's input weight divided by the sum across siblings that
    share the same first-step).  That later pass overwrites the dense 1.0
    set here whenever stochastic branches are active.
    """
    if dt is None:
        return None
    # Defensive re-cast: ensure d is canonical Enum on the Param.
    out = (dt.lazy()
             .select(alias_to_axis("d", "d")).unique()
             .with_columns(value=pl.lit(1.0))
             .sort("d")
             .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


def pdt_branch_weight_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    pd_bw: Param | None = None,
) -> Param | None:
    """Per-(d, t) branch weight — dense 1.0 baseline (superseded later).

    Returns the trivial 1.0 default per (d, t).  Like
    :func:`pd_branch_weight_from_source`, this is the ``apply_derived_a``
    baseline; it is the final value only for deterministic / non-stochastic
    fixtures.  For stochastic fixtures it is SUPERSEDED by the real
    sibling-normalised weights computed in ``apply_derived_g``
    (``_branch.apply_branch_cluster`` → ``pdt_branch_weight_lf``).
    """
    if dt is None:
        return None
    # Defensive re-cast: ensure d/t are canonical Enum on the Param.
    out = (dt.lazy()
             .with_columns(value=pl.lit(1.0))
             .select(alias_to_axis("d", "d"),
                     alias_to_axis("t", "t"),
                     "value")
             .sort("d", "t")
             .collect())
    if out.height == 0:
        return None
    return Param(("d", "t"), out)


# ---------------------------------------------------------------------------
# §3.6.2 — p_process_existing_count
# ---------------------------------------------------------------------------


def _read_p_entity_all_existing_csv(workdir: Path | None,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> pl.DataFrame | None:
    """Γ.6.D — read ``solve_data/p_entity_all_existing`` from *provider*.

    Returns ``[e, d, value]``, or None when *workdir* is None or the
    provider does not carry the frame.  Used by helpers that need the
    chained existing capacity (incl. multi-solve handoff) without
    recomputing it from raw ``entity.existing``.

    ``emit_p_entity_existing_chain`` (``emit/_chain_params.py``) puts the
    frame into the provider; *workdir* only forms the key, nothing is
    read from disk.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
    if not _provider_has_key(provider, p):
        return None
    try:
        df = _provider_read(provider, p)
    except Exception:
        return None
    if (df.height == 0 or "entity" not in df.columns
            or "period" not in df.columns or "value" not in df.columns):
        return None
    return (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .select("e", "d", "value"))


def _process_capacity_per_unit_lf(cap_lf: pl.LazyFrame, source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[p, d, value]``: the capacity ``[e, d, value]`` (MW / MWh) of each entity divided by its unitsize
    (:func:`_entity_unitsize_lf`; ``UNITSIZE_DEFAULT`` where absent).

    capacity = existing × virtual_unitsize when virtual_unitsize is above 0, else existing, so the result is the
    existing unit count: ``existing`` itself with virtual_unitsize, 1 without it (unitsize = existing; a
    virtual_unitsize of 0 or below is ignored).
    """
    us_lf = _entity_unitsize_lf(source)
    return (cap_lf.pipe(rename_to_axis, {"e": "p"})
                  .join(us_lf.pipe(rename_to_axis, {"e": "p"}), on="p", how="left")
                  .with_columns(us=pl.col("us").fill_null(UNITSIZE_DEFAULT))
                  .with_columns(value=pl.col("value") / pl.col("us"))
                  .select("p", "d", "value"))


def _explicit_process_existing(source: "InputSource") -> list[str]:
    """Units and connections with an explicit ``existing`` row (Spine's default-broadcast rows excluded)."""
    names: list[str] = []
    for cls in ("unit", "connection"):
        ex = _try_param_explicit(source, cls, "existing")
        if ex is not None and ex.height > 0:
            names.extend(str(x) for x in ex.get_column("name").unique().to_list())
    return sorted(set(names))


def _pre_existing_process_capacity_lf(source: "InputSource",
                                       active_solve: str | None,
                                       periods: list[str],
                                       workdir: Path | None,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> pl.LazyFrame | None:
    """Lazy ``[e, d, value]``: the lifetime-gated existing capacity (``p_entity_pre_existing``: existing ×
    virtual_unitsize when it is set, else existing) of each unit / connection with an explicit ``existing``, over
    ``periods``.  ``None`` when no process has an explicit ``existing`` or ``periods`` is empty."""
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    names = _explicit_process_existing(source)
    if not names or not periods:
        return None
    return (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
              .filter(pl.col("e").cast(pl.Utf8).is_in(names)))


def p_process_existing_count_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
    workdir: Path | None = None,
    *,
    provider: "object | None" = None,
) -> Param | None:
    """Per-(p, d) existing unit count = capacity / unitsize (= ``existing`` when virtual_unitsize is above 0).

    capacity[p, d] = existing[p, d] × virtual_unitsize[p] when virtual_unitsize is above 0, else existing[p, d];
    unitsize per the canonical cascade (:func:`_entity_unitsize_lf`)::

        unitsize = virtual_unitsize (if explicitly set and above 0)
                  OR existing       (if non-zero)
                  OR 1000.0

    The capacity comes from the canonical chained ``p_entity_all_existing`` (multi-solve handoff state and
    lifetime gate integrated) when the run has emitted it; otherwise from ``p_entity_pre_existing`` (the
    lifetime-gated existing × virtual_unitsize cascade over the periods of ``dt``) for the units / connections
    with an explicit ``existing``.

    Returns ``None`` when no process has an explicit existing capacity.
    """
    if dt is None:
        return None

    pae_csv = _read_p_entity_all_existing_csv(workdir, provider=provider)
    if pae_csv is not None and pae_csv.height > 0:
        # Restrict to process-side entities (unit ∪ connection).
        proc_set, _, _ = _entity_classes_lookup(source)
        per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
        if per_proc.height > 0:
            df = (_process_capacity_per_unit_lf(per_proc.lazy(), source)
                     .sort("p", "d")
                     .collect())
            if df.height > 0:
                return Param(("p", "d"), df)

    periods = [str(d) for d in dt.get_column("d").unique(maintain_order=True).to_list()]
    cap_lf = _pre_existing_process_capacity_lf(source, active_solve, periods, workdir, provider=provider)
    if cap_lf is None:
        return None
    out = (_process_capacity_per_unit_lf(cap_lf, source)
              .sort("p", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "d"), out)


# ---------------------------------------------------------------------------
# Catalog (used by parity tests)
# ---------------------------------------------------------------------------

DERIVED_A_FIELDS = (
    "dt",
    "p_step_duration",
    "p_period_share",
    "p_inflation_op",
    "p_timestep_weight",
    "pd_branch_weight",
    "pdt_branch_weight",
    "p_inflow",
    "p_process_existing_count",
    "p_profile_value",
)


# ---------------------------------------------------------------------------
# Integration entrypoint
# ---------------------------------------------------------------------------


def apply_derived_a(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.A foundational Derived Params, mutating ``flex_data``
    in place.

    Δ.12b — assignment is unconditional for the helpers that are
    authoritative producers (``p_inflation_op``, ``p_timestep_weight``,
    ``pd_branch_weight``, ``pdt_branch_weight``, ``p_penalty_up``,
    ``p_penalty_down``).  ``p_inflow`` and ``p_process_existing_count``
    retain a conditional assignment because the helpers have known
    incomplete coverage (``inflow_method=scale_to_*`` is deferred to
    Batch B; ``p_process_existing_count`` skips when no processes have
    explicit existing capacity in the source) — see TODO at each call
    site for the helper-extension scope.

    Dependency order:
        dt / p_step_duration → p_period_share, p_inflation_op,
        p_timestep_weight, pd_branch_weight, pdt_branch_weight, p_inflow
        → p_process_existing_count, p_profile_value (high-risk, last).
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    if active_solve is None:
        # Single-solve fixtures may omit solve_current.csv; fall back to
        # the FlexData's already-loaded dt (no override possible).
        return

    # 1. dt + p_step_duration -------------------------------------------
    # When ``solve.new_stepduration`` is set for the active solve, the
    # legacy ``TimelineConfig.create_timeline_from_timestep_duration``
    # has already rewritten the timeline (e.g. collapsing 8760 hourly
    # rows into one 8760 h row) and the legacy per-solve preprocessing
    # has materialised that rewrite into ``solve_data/steps_in_use.csv``.
    # The DB-direct ``dt_and_step_duration_from_source`` reads the raw
    # ``timeline.timestep_duration`` rows and would re-introduce the
    # full hourly grid, undoing the aggregation.  Skip the DB override
    # in that case and trust the CSV-loaded values.
    dt_csv = getattr(flex_data, "dt", None)
    new_sd_df = _try_param(source, "solve", "new_stepduration")
    skip_dt_override = False
    if new_sd_df is not None and "name" in new_sd_df.columns:
        sel = new_sd_df.filter(pl.col("name") == active_solve)
        if sel.height > 0 and "value" in sel.columns:
            v = sel["value"][0]
            if v is not None:
                skip_dt_override = True
    if skip_dt_override:
        usable_dt = dt_csv
        sd_for_share = getattr(flex_data, "p_step_duration", None)
    else:
        dt_step = dt_and_step_duration_from_source(source, active_solve,
                                                       workdir, ctx=ctx,
                                                       provider=provider)
        if dt_step is None:
            # Without dt we can't derive the dependent Params either; bail.
            return
        dt_db, step_dur_db = dt_step
        if dt_db is not None:
            flex_data.dt = dt_db
            flex_data.p_step_duration = step_dur_db
            usable_dt = dt_db
            sd_for_share = step_dur_db
        else:
            usable_dt = dt_csv
            sd_for_share = getattr(flex_data, "p_step_duration", None)

    if usable_dt is None:
        return

    # 2. p_period_share — None when sd_for_share is None (no step duration).
    if sd_for_share is not None:
        flex_data.p_period_share = p_period_share_from_source(
            source, usable_dt, sd_for_share)
        # The share was just built from this solve's own dt, so the solve covers all of it.
        if flex_data.p_period_share is not None:
            flex_data.p_window_fraction = Param(
                ("d",), flex_data.p_period_share.frame.with_columns(value=pl.lit(1.0, dtype=pl.Float64)))

    # 2b. p_years_represented_d — per-period R width sum (e.g. 5.0 for a
    # 5-year invest period) over the periods of the solve's dt and its
    # period labels (a label has no dt rows but keeps its years for the
    # period-level costs); consumed by ``process_outputs.read_parameters``
    # to scale annualised per-period costs / CO2 to absolute totals.  The
    # provider arm reads the canonical CSV, which carries the anchor's
    # years on every stochastic branch period.
    flex_data.p_years_represented_d = p_years_represented_d_from_source(
        source, active_solve, bookkeeping_periods(usable_dt, workdir, ctx=ctx, provider=provider),
        provider=provider)

    # 3. p_inflation_op — Δ.12b: unconditional.  None == no inflation
    #    declared; the multi-year cascade in apply_derived_c overlays
    #    this with the full per-period inflation later.
    flex_data.p_inflation_op = p_inflation_op_from_source(
        source, usable_dt, active_solve)

    # 4. p_timestep_weight — GUARDED (narrow inversion of Δ.12b for THIS
    #    helper).  None now means "no override — keep the CSV-loaded
    #    value".  On RP solves the helper returns None so the folded
    #    ``solve_data/timestep_weight.csv`` weights survive (the loader
    #    always populates a non-None Param, at minimum the dense 1.0
    #    default, so arithmetic stays safe).  Non-RP ``timeset_weights``
    #    solves still take the helper's computed Param unchanged.
    #    An rp solve keeps the writer's frame even when timeset_weights exist: it holds the rp periods'
    #    folded weights and the other periods' timeset_weights rows (solve_has_rp_timeset).
    tsw_new = p_timestep_weight_from_source(
        source, usable_dt, active_solve)
    if tsw_new is not None and not solve_has_rp_timeset(source, active_solve):
        flex_data.p_timestep_weight = tsw_new

    # 5. pd_branch_weight + pdt_branch_weight — Δ.12b: unconditional.
    # Γ.3.A's simple-1.0 helpers; Γ.3.G's full multi-branch cascade
    # overlays them later via apply_branch_cluster.
    pd_bw = pd_branch_weight_from_source(source, usable_dt)
    flex_data.pd_branch_weight = pd_bw
    flex_data.pdt_branch_weight = pdt_branch_weight_from_source(
        source, usable_dt, pd_bw)

    # 6. p_inflow -------------------------------------------------------
    # Native derivation pathways are DISABLED.  ``apply_p_inflow_with_scaling``
    # (and the ``p_inflow_from_source`` fallback) overrode the seed value
    # loaded by ``_load_node`` from ``solve_data/pdtNodeInflow.csv``, but
    # for aggregated-timestep solves (e.g. ``lt_rp`` with
    # ``step_duration = 8760``) the fallback joined raw per-hour source
    # values against ``dt`` and picked the per-hour rate as if it were
    # MWh-per-timestep — collapsing annual demand by a factor of
    # ``step_duration`` and yielding a structurally-wrong LP.
    #
    # The seed value produced by flextool's
    # ``preprocessing/node_inflow_scaling_params.py`` +
    # ``entity_period_calc_params.write_pdtNodeInflow`` is the known-good
    # input.  Until the native cascade is proven correct across all
    # ``inflow_method`` × ``step_duration`` combinations, the seed
    # survives unchanged.  Re-enable per-pathway only after parity is
    # established (see commit message for the disable for context).

    # 7. p_process_existing_count ---------------------------------------
    # Δ.12c-fix gap #4: the helper handles two paths:
    # (1) ``p_entity_all_existing.csv`` (workdir-derived, carries
    #     multi-solve handoff state and lifetime gate);
    # (2) ``p_entity_pre_existing`` (existing × virtual_unitsize) for the
    #     explicit ``unit.existing`` / ``connection.existing``; count = capacity / unitsize per (p, d).
    # Returns ``None`` for fixtures with neither path active (pure-invest
    # without initial capacity).  Keep conditional so the seed Param
    # (which always emits a row per process for ``base_cap_pd`` from the
    # CSV preprocessing) survives in that degenerate case.
    ec = p_process_existing_count_from_source(
        source, usable_dt, active_solve, workdir, provider=provider)
    if ec is not None:
        flex_data.p_process_existing_count = ec

    # 8. p_profile_value (Δ.7 cluster C — profile cascade) ------------
    # Δ.7 lifts the helper into ``_profiles.apply_profile_cascade``
    # — the new module ports flextool's full 5-branch cascade lazily and
    # drops the Δ.5/Δ.6 canonical-keys gate.  We delegate here so the
    # apply_derived_a entry stays the single integration point.
    # Δ.12b: defensive try/except removed — apply_profile_cascade is
    # parity-bound (cluster C tests gate it).
    from flextool.engine.preprocess.params._profiles import apply_profile_cascade
    apply_profile_cascade(flex_data, source, workdir, provider=provider)

    # 9. p_penalty_up / p_penalty_down (Δ.10 cluster F) ----------------
    # Dense over (nodeBalance ∪ nodeBalancePeriod) × usable_dt: the authored value (scalar, period,
    # time or period × time map, mixed across nodes), else the schema default.  A None result
    # (unclassifiable map, or an empty node set / dt) keeps the seed (emit pdtNode); a stochastic-time
    # map is resolved by the producer's own per-solve fold, not left to the seed.
    from flextool.engine.preprocess.params._arithmetic import apply_penalty_params
    apply_penalty_params(flex_data, source, usable_dt)


# ---------------------------------------------------------------------------
# Internal: frame-equal guard for safe overlay
# ---------------------------------------------------------------------------


def _frame_equal_sorted(a, b, keys: tuple[str, ...]) -> bool:
    """Return True iff frames *a* and *b* are equal after sort by *keys*
    and ``Float64`` cast on the value column where present.  Lenient
    on dtype mismatches — the bar is *value parity*, not metadata
    parity.
    """
    fa = a.frame if hasattr(a, "frame") else a
    fb = b.frame if hasattr(b, "frame") else b
    if fa is None or fb is None:
        return False
    cols_a = set(fa.columns)
    cols_b = set(fb.columns)
    if cols_a != cols_b:
        return False
    sort_cols = [c for c in keys if c in cols_a]
    extra = [c for c in fa.columns if c not in sort_cols]
    fa = fa.sort(sort_cols + extra) if sort_cols else fa
    fb = fb.sort(sort_cols + extra) if sort_cols else fb
    if "value" in cols_a:
        fa = fa.with_columns(value=pl.col("value").cast(pl.Float64,
                                                          strict=False))
        fb = fb.with_columns(value=pl.col("value").cast(pl.Float64,
                                                          strict=False))
    return fa.equals(fb)


def _param_matches(csv_val: object | None, db_val: object | None,
                    tol: float = 1e-9) -> bool:
    """Return True iff CSV and DB Params have matching frames within
    *tol* on the value column.  Conservative: requires both non-None
    and same column set.  Used as the safety gate before overlaying.
    """
    if csv_val is None or db_val is None:
        return False
    fa = csv_val.frame if hasattr(csv_val, "frame") else csv_val
    fb = db_val.frame if hasattr(db_val, "frame") else db_val
    cols_a = set(fa.columns)
    cols_b = set(fb.columns)
    if cols_a != cols_b:
        return False
    if fa.height != fb.height:
        return False
    keys = [c for c in fa.columns if c != "value"]
    fa = fa.sort(keys) if keys else fa
    fb = fb.sort(keys) if keys else fb
    if "value" in cols_a:
        a_v = fa["value"].cast(pl.Float64, strict=False).fill_null(0.0)
        b_v = fb["value"].cast(pl.Float64, strict=False).fill_null(0.0)
        diff = (a_v - b_v).abs().max()
        if diff is None or diff > tol:
            return False
        # Compare keys row-by-row.
        for k in keys:
            if not fa[k].equals(fb[k]):
                return False
        return True
    # No value column — pure set equality.
    return fa.equals(fb)


# ===========================================================================
# Γ.3.B — Process topology + reclassified method-derived helpers
# ===========================================================================
#
# Per the audit doc §3.3 / §3.5 / §3.10 (and the Γ.2 reclassification list),
# every Param built on top of the "internal method" classification
# must derive that classification from the (conversion_method,
# startup_method, fork_method) triplet.  This batch implements that
# mapping plus the dependent topology / cap / slope / varCost helpers.
#
# Each helper is gated on a frame-equal precheck against the CSV-loaded
# value before being written to ``flex_data`` (``apply_derived_b``).
# A mismatch leaves the CSV value in place — never an LP corruption surface.

# (ct_method, startup_method, fork_method) → internal method.
# See METHODS_MAPPING in :mod:`flextool.input_derivation._specs`.
# (removed in the Tier 4 retirement, 2026-05-23).
_METHODS_MAPPING: dict[tuple[str, str, str], str] = {
    ("constant_efficiency", "no_startup", "fork_no"): "method_1way_1var_off",
    ("constant_efficiency", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("constant_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("constant_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("constant_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("constant_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("no_losses_no_variable_cost", "no_startup", "fork_no"): "method_2way_1var_off",
    ("no_losses_no_variable_cost", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("variable_cost_only", "no_startup", "fork_no"): "method_2way_2var_off",
    ("variable_cost_only", "no_startup", "fork_yes"): "method_2way_nvar_off",
    ("regular", "no_startup", "fork_no"): "method_2way_2var_exclude",
    ("regular", "no_startup", "fork_yes"): "not_applicable",
    ("exact", "no_startup", "fork_no"): "method_2way_2var_MIP_exclude",
    ("exact", "no_startup", "fork_yes"): "not_applicable",
    # Two-way connections with a startup method: the flow shape of the no-startup entry (the online
    # variables come from the startup method, not from the internal method).
    ("no_losses_no_variable_cost", "linear", "fork_no"): "method_2way_1var_off",
    ("no_losses_no_variable_cost", "linear", "fork_yes"): "method_2way_nvar_off",
    ("no_losses_no_variable_cost", "binary", "fork_no"): "method_2way_1var_off",
    ("no_losses_no_variable_cost", "binary", "fork_yes"): "method_2way_nvar_off",
    ("variable_cost_only", "linear", "fork_no"): "method_2way_2var_off",
    ("variable_cost_only", "linear", "fork_yes"): "method_2way_nvar_off",
    ("variable_cost_only", "binary", "fork_no"): "method_2way_2var_off",
    ("variable_cost_only", "binary", "fork_yes"): "method_2way_nvar_off",
    ("regular", "linear", "fork_no"): "method_2way_2var_exclude",
    ("regular", "linear", "fork_yes"): "not_applicable",
    ("regular", "binary", "fork_no"): "method_2way_2var_exclude",
    ("regular", "binary", "fork_yes"): "not_applicable",
    ("exact", "linear", "fork_no"): "method_2way_2var_MIP_exclude",
    ("exact", "linear", "fork_yes"): "not_applicable",
    ("exact", "binary", "fork_no"): "method_2way_2var_MIP_exclude",
    ("exact", "binary", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_no"): "not_applicable",
    ("min_load_efficiency", "no_startup", "fork_yes"): "not_applicable",
    ("min_load_efficiency", "linear", "fork_no"): "method_1way_1var_LP",
    ("min_load_efficiency", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("min_load_efficiency", "binary", "fork_no"): "method_1way_1var_MIP",
    ("min_load_efficiency", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("none", "no_startup", "fork_no"): "method_1way_1var_off",
    ("none", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("none", "linear", "fork_no"): "method_1way_1var_LP",
    ("none", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("none", "binary", "fork_no"): "method_1way_1var_MIP",
    ("none", "binary", "fork_yes"): "method_1way_nvar_MIP",
    ("unidirectional", "no_startup", "fork_no"): "method_1way_1var_off",
    ("unidirectional", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("unidirectional", "linear", "fork_no"): "method_1way_1var_LP",
    ("unidirectional", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("unidirectional", "binary", "fork_no"): "method_1way_1var_MIP",
    ("unidirectional", "binary", "fork_yes"): "method_1way_nvar_MIP",
}


# Method-category subsets — mirror of
# :mod:`flextool.input_derivation._method_constants`.
_METHOD_DIRECT = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_2way_1var_off", "method_2way_2var_off",
    "method_2way_2var_exclude", "method_2way_2var_MIP_exclude",
))
_METHOD_INDIRECT = frozenset((
    "method_1way_nvar_off", "method_1way_nvar_LP", "method_1way_nvar_MIP",
    "method_2way_nvar_off",
))
_METHOD_2WAY_2VAR = frozenset((
    "method_2way_2var_off", "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))
_METHOD_1WAY_1VAR = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
))
_METHOD_LP = frozenset(("method_1way_1var_LP", "method_1way_nvar_LP"))
_METHOD_MIP = frozenset((
    "method_1way_1var_MIP", "method_1way_nvar_MIP",
))


def _group_transfer_method_overrides(source: "InputSource") -> "pl.LazyFrame | None":
    """``(p, ct_override)``: the transfer method a group's ``transfer_method`` imposes on its connections.

    Mirrors :func:`flextool.input_derivation._dc_power_flow.derive_dc_power_flow`: every group whose
    ``transfer_method`` is not ``use_connection_transfer_methods`` overrides the method of each connection with
    both endpoints (``connection__node__node``) among the group's nodes; ``dc_power_flow_with_angles`` becomes
    ``no_losses_no_variable_cost`` (the DC lines carry one signed flow) and skips the ``is_DC = yes``
    connections.  A connection matched by several groups with different overrides raises (as the input
    derivation does).  The process key is the Utf8 column ``_pk``.  None when no group overrides anything.
    """
    gtm = _try_param(source, "group", "transfer_method")
    gn = _try_entities(source, "group__node")
    cnn = _try_entities(source, "connection__node__node")
    if gtm is None or gn is None or cnn is None:
        return None
    groups = (gtm.lazy()
              .select(pl.col("name").cast(pl.Utf8).alias("g"), pl.col("value").cast(pl.Utf8).alias("method"))
              .filter(pl.col("method").is_not_null() & (pl.col("method") != "use_connection_transfer_methods")))
    nodes = gn.lazy().select(pl.col("group").cast(pl.Utf8).alias("g"), pl.col("node").cast(pl.Utf8).alias("n"))
    ends = cnn.lazy().select(pl.col("connection").cast(pl.Utf8).alias("p"),
                             pl.col("node_1").cast(pl.Utf8).alias("n1"), pl.col("node_2").cast(pl.Utf8).alias("n2"))
    matched = (groups.join(nodes, on="g", how="inner").rename({"n": "n1"})
               .join(ends, on="n1", how="inner")
               .join(nodes.rename({"n": "n2"}), on=["g", "n2"], how="semi")
               .select("p", "g", "method"))
    is_dc = _try_param(source, "connection", "is_DC")
    if is_dc is not None:
        dc = is_dc.lazy().filter(pl.col("value") == "yes").select(pl.col("name").cast(pl.Utf8).alias("p"))
        matched = pl.concat([
            matched.filter(pl.col("method") != "dc_power_flow_with_angles"),
            matched.filter(pl.col("method") == "dc_power_flow_with_angles").join(dc, on="p", how="anti"),
        ])
    matched = matched.collect()
    n_methods = matched.group_by("p").agg(pl.col("method").n_unique().alias("n"))
    bad = n_methods.filter(pl.col("n") > 1)["p"].to_list()
    if bad:
        from flextool.input_derivation._dc_power_flow import (
            conflicting_overrides_message, group_override_conflicts,
        )
        sources: dict[str, dict[str, str]] = {}
        for p, g, m in matched.filter(pl.col("p").is_in(bad)).select("p", "g", "method").rows():
            sources.setdefault(p, {})[g] = m
        raise FlexToolConfigError(conflicting_overrides_message(group_override_conflicts(sources)))
    out = (matched.lazy()
           .with_columns(ct_override=pl.when(pl.col("method") == "dc_power_flow_with_angles")
                         .then(pl.lit("no_losses_no_variable_cost")).otherwise(pl.col("method")))
           .group_by("p", maintain_order=True).agg(pl.col("ct_override").first())
           .collect())
    if out.height == 0:
        return None
    return out.lazy().select(pl.col("p").alias("_pk"), "ct_override")


def _classify_process_method(source: "InputSource") -> pl.DataFrame:
    """Build the per-process internal-method classification.

    Returns a DataFrame with schema ``[p, ct, startup, fork, method,
    klass]``:

      * ``ct`` — user-facing ``conversion_method`` / ``transfer_method``.
      * ``startup`` — user-facing ``startup_method`` (default ``no_startup``,
        promoted to ``linear`` when ``minimum_time_method ∈ {min_uptime,
        min_downtime, both}``; mirrors input_writer.py:1294-1304).
      * ``fork`` — ``fork_yes`` when |sources|>1 OR |sinks|>1 OR delayed,
        else ``fork_no`` (input_writer.py:1350-1354).
      * ``method`` — internal method per :data:`_METHODS_MAPPING`; rows
        whose triplet maps to ``not_applicable`` or has no mapping are
        dropped.
      * ``klass`` — ``"unit"`` or ``"connection"`` (the entity class of
        the row).

    Algorithm (single-collect lazy chain):

      1. Pull all units + connections.
      2. Pull ``conversion_method`` (units), ``transfer_method``
         (connections), ``startup_method`` (units + connections),
         ``minimum_time_method`` (units), ``delay`` (units only — a
         connection ``delay`` is ignored).
      3. Count input/output arcs per process.
      4. Apply defaults (``constant_efficiency`` for units,
         ``regular`` for connections; ``no_startup``), then the group
         ``transfer_method`` overrides
         (:func:`_group_transfer_method_overrides`, as the input
         derivation's ``derived/ct_method_overrides``).
      5. Apply minimum_time → linear-startup override.
      6. Resolve fork from arc counts + delayed-set membership.
      7. Map ``(ct, startup, fork) → method`` via the polars when/then
         chain built from :data:`_METHODS_MAPPING`.
      8. Drop ``not_applicable`` rows; ``not_applicable`` is also returned
         when the triplet has no mapping (defensive).
    """
    # Step 1 — process universe.
    units = _try_entities(source, "unit")
    conns = _try_entities(source, "connection")
    parts: list[pl.LazyFrame] = []
    if units is not None:
        parts.append(
            units.lazy()
                 .pipe(rename_to_axis, {"name": "p"})
                 .with_columns(klass=lit_axis("unit", "klass"))
                 .select("p", "klass"),
        )
    if conns is not None:
        parts.append(
            conns.lazy()
                 .pipe(rename_to_axis, {"name": "p"})
                 .with_columns(klass=lit_axis("connection", "klass"))
                 .select("p", "klass"),
        )
    if not parts:
        return pl.DataFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "ct": pl.Utf8, "startup": pl.Utf8,
            "fork": pl.Utf8, "method": pl.Utf8, "klass": pl.Utf8,
        })
    base = pl.concat(parts).unique()

    # Step 2 — pull method-shaping params.
    cm_unit = _try_param(source, "unit", "conversion_method")
    cm_conn = _try_param(source, "connection", "transfer_method")
    cm_parts: list[pl.LazyFrame] = []
    if cm_unit is not None:
        cm_parts.append(cm_unit.lazy().select(
            alias_to_axis("name", "p"),
            pl.col("value").alias("ct"),
        ))
    if cm_conn is not None:
        cm_parts.append(cm_conn.lazy().select(
            alias_to_axis("name", "p"),
            pl.col("value").alias("ct"),
        ))
    cm_lazy = pl.concat(cm_parts) if cm_parts else None

    sm_unit = _try_param(source, "unit", "startup_method")
    sm_conn = _try_param(source, "connection", "startup_method")
    sm_parts: list[pl.LazyFrame] = []
    if sm_unit is not None:
        sm_parts.append(sm_unit.lazy().select(
            alias_to_axis("name", "p"),
            pl.col("value").alias("startup_raw"),
        ))
    if sm_conn is not None:
        sm_parts.append(sm_conn.lazy().select(
            alias_to_axis("name", "p"),
            pl.col("value").alias("startup_raw"),
        ))
    sm_lazy = pl.concat(sm_parts) if sm_parts else None

    mtm = _try_param(source, "unit", "minimum_time_method")
    mtm_lazy = (mtm.lazy().select(
        alias_to_axis("name", "p"),
        pl.col("value").alias("mtm"),
    )) if mtm is not None else None

    # Only units are delayed: a connection ``delay`` is not implemented
    # and is ignored (input_derivation warns about it), so it must not
    # mark the connection ``fork_yes`` (that changed its method, dropped
    # its efficiency losses and made a ``regular`` connection one-way).
    delay_unit = _try_param(source, "unit", "delay")
    delay_parts: list[pl.LazyFrame] = []
    if delay_unit is not None:
        delay_parts.append(delay_unit.lazy().select(
            alias_to_axis("name", "p"),
        ).unique())
    delay_lazy = (pl.concat(delay_parts).unique()
                    .with_columns(is_delayed=pl.lit(True))) \
                  if delay_parts else None

    # Step 3 — input/output arc counts per process.
    src_acc: list[pl.LazyFrame] = []
    sink_acc: list[pl.LazyFrame] = []
    uin = _try_entities(source, "unit__inputNode")
    if uin is not None:
        src_acc.append(uin.lazy().select(alias_to_axis("unit", "p")))
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None:
        sink_acc.append(uout.lazy().select(alias_to_axis("unit", "p")))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None:
        src_acc.append(cnn.lazy().select(alias_to_axis("connection", "p")))
        sink_acc.append(cnn.lazy().select(alias_to_axis("connection", "p")))

    src_counts = (pl.concat(src_acc).group_by("p").agg(
                    pl.len().alias("n_src"))) if src_acc else None
    sink_counts = (pl.concat(sink_acc).group_by("p").agg(
                    pl.len().alias("n_sink"))) if sink_acc else None

    # Step 4 — assemble lazy frame with defaults.
    lf = base.lazy()
    if cm_lazy is not None:
        lf = lf.join(cm_lazy, on="p", how="left")
    else:
        lf = lf.with_columns(ct=pl.lit(None).cast(pl.Utf8))
    # Defaults: unit → constant_efficiency, connection → regular.
    lf = lf.with_columns(
        ct=pl.when(pl.col("ct").is_not_null()).then(pl.col("ct"))
            .when(pl.col("klass") == "unit").then(pl.lit("constant_efficiency"))
            .otherwise(pl.lit("regular"))
    )
    # Step 4b — group transfer_method overrides (connections with both endpoints in the group).
    overrides = _group_transfer_method_overrides(source)
    if overrides is not None:
        lf = (lf.with_columns(_pk=pl.col("p").cast(pl.Utf8))
                .join(overrides, on="_pk", how="left")
                .with_columns(ct=pl.when((pl.col("klass") == "connection") & pl.col("ct_override").is_not_null())
                                .then(pl.col("ct_override")).otherwise(pl.col("ct")))
                .drop("ct_override", "_pk"))
    if sm_lazy is not None:
        lf = lf.join(sm_lazy, on="p", how="left")
    else:
        lf = lf.with_columns(startup_raw=pl.lit(None).cast(pl.Utf8))
    lf = lf.with_columns(
        startup=pl.col("startup_raw").fill_null("no_startup")
    ).drop("startup_raw")
    # Step 5 — minimum_time_method overrides startup → linear when active.
    if mtm_lazy is not None:
        lf = lf.join(mtm_lazy, on="p", how="left")
        lf = lf.with_columns(
            startup=pl.when(
                (pl.col("startup") == "no_startup")
                & pl.col("mtm").is_in(["min_uptime", "min_downtime", "both"])
            ).then(pl.lit("linear"))
             .otherwise(pl.col("startup"))
        ).drop("mtm")
    # Step 6 — fork resolution.
    if src_counts is not None:
        lf = lf.join(src_counts, on="p", how="left")
    else:
        lf = lf.with_columns(n_src=pl.lit(0))
    if sink_counts is not None:
        lf = lf.join(sink_counts, on="p", how="left")
    else:
        lf = lf.with_columns(n_sink=pl.lit(0))
    lf = lf.with_columns(
        n_src=pl.col("n_src").fill_null(0),
        n_sink=pl.col("n_sink").fill_null(0),
    )
    if delay_lazy is not None:
        lf = lf.join(delay_lazy, on="p", how="left")
        lf = lf.with_columns(
            is_delayed=pl.col("is_delayed").fill_null(False))
    else:
        lf = lf.with_columns(is_delayed=pl.lit(False))
    lf = lf.with_columns(
        fork=pl.when((pl.col("n_src") > 1) | (pl.col("n_sink") > 1)
                      | pl.col("is_delayed"))
              .then(pl.lit("fork_yes"))
              .otherwise(pl.lit("fork_no"))
    )

    # Step 7 — map (ct, startup, fork) → method via when/then chain.
    expr = pl.lit("not_applicable")
    for (ct, st, fk), method in _METHODS_MAPPING.items():
        expr = (pl.when((pl.col("ct") == ct)
                        & (pl.col("startup") == st)
                        & (pl.col("fork") == fk))
                  .then(pl.lit(method))
                  .otherwise(expr))
    lf = lf.with_columns(method=expr)

    # Step 8 — drop not_applicable; project final columns.
    out = (lf.filter(pl.col("method") != "not_applicable")
              .select("p", "ct", "startup", "fork", "method", "klass")
              .sort("p")
              .collect())
    return out


# ---------------------------------------------------------------------------
# Method-classifier-keyed Param sets (reclassified Projections, §1.3, §1.5)
# ---------------------------------------------------------------------------


def process_indirect_set(source: "InputSource",
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """Units with internal method ∈ INDIRECT family.  Schema ``[p]``.

    Reclassified-Derived equivalent of the §1.5 ``process_indirect``
    Projection: filter classified processes (units only) to the
    ``method_*_nvar_*`` family (CHP / extraction patterns).  flextool's
    preprocessing only emits unit names here; connections are skipped.
    """
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p")})
    return (classified.lazy()
              .filter((pl.col("klass") == "unit")
                      & pl.col("method").is_in(list(_METHOD_INDIRECT)))
              .select("p")
              .sort("p")
              .collect())


def _zero_flow_coef_pairs(source: "InputSource",
                              kind: str
                              ) -> pl.DataFrame:
    """Read ``unit__inputNode.conversion_flow_coeff`` (kind='source') or
    ``unit__outputNode.conversion_flow_coeff`` (kind='sink') and return
    the ``(p, source)`` / ``(p, sink)`` pairs whose coefficient is zero.
    flextool drops these from the conversion_indirect LHS / RHS
    (``flex_data.py:_load_indirect`` lines 942-944 / 968-970).
    """
    if kind == "source":
        ec = "unit__inputNode"
        col = "source"
    else:
        ec = "unit__outputNode"
        col = "sink"
    df = _try_param(source, ec, "conversion_flow_coeff")
    if df is None or df.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                     col: schema_dtype(_enums, col)})
    cols = df.columns
    rename: dict[str, str] = {}
    for c in cols:
        if c == "unit":
            rename[c] = "p"
        elif c == "node":
            rename[c] = col
    out = (df.lazy().pipe(rename_to_axis, rename)
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .filter(pl.col("value") == 0.0)
              .select("p", col))
    return out.unique().collect()


def process_input_flows(source: "InputSource",
                          pss: pl.DataFrame,
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """Indirect-process input arcs.

    pss filtered to ``sink == p AND p ∈ process_indirect``.  Drops arcs
    whose ``conversion_flow_coeff == 0`` (mirrors
    ``flex_data.py:_load_indirect:942-944``).  Schema ``[p, source, sink]``.
    """
    pi = process_indirect_set(source, classified)
    if pss is None or pss.height == 0 or pi.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink")})
    # Cross-axis compare: sink (e) vs p (p).  Per contract p ⊂ e;
    # up-cast p to e so the compare runs in Enum.
    out_lf = (pss.lazy()
                .filter(pl.col("sink") == cast_dim(pl.col("p"), None, "e"))
                .join(pi.lazy(), on="p", how="inner"))
    zero_src = _zero_flow_coef_pairs(source, "source")
    if zero_src.height > 0:
        out_lf = out_lf.join(zero_src.lazy(), on=["p", "source"], how="anti")
    return out_lf.sort("p", "source", "sink").collect()


def process_output_flows(source: "InputSource",
                           pss: pl.DataFrame,
                           classified: pl.DataFrame | None = None
                           ) -> pl.DataFrame:
    """Indirect-process output arcs (mirror of input).

    Drops arcs whose ``conversion_flow_coeff == 0`` (mirror of
    ``flex_data.py:_load_indirect:968-970``).
    """
    pi = process_indirect_set(source, classified)
    if pss is None or pss.height == 0 or pi.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink")})
    # Cross-axis compare: source (e) vs p (p).  Per contract p ⊂ e;
    # up-cast p to e so the compare runs in Enum.
    out_lf = (pss.lazy()
                .filter(pl.col("source") == cast_dim(pl.col("p"), None, "e"))
                .join(pi.lazy(), on="p", how="inner"))
    zero_sink = _zero_flow_coef_pairs(source, "sink")
    if zero_sink.height > 0:
        out_lf = out_lf.join(zero_sink.lazy(), on=["p", "sink"], how="anti")
    return out_lf.sort("p", "source", "sink").collect()


def process_indirect_dt(source: "InputSource",
                          dt: pl.DataFrame,
                          classified: pl.DataFrame | None = None
                          ) -> pl.DataFrame:
    """``process_indirect × dt`` cross-product.  Schema ``[p, d, t]``."""
    pi = process_indirect_set(source, classified)
    if pi.height == 0 or dt is None or dt.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "d": schema_dtype(_enums, "d"),
                                      "t": schema_dtype(_enums, "t")})
    # Defensive re-cast: ensure d/t are canonical Enum so the cross-join
    # against pi (Enum p) keeps Enum dtype on the output (p, d, t) Param.
    return (pi.lazy()
              .join(dt.lazy()
                       .with_columns(alias_to_axis(pl.col("d"), "d"),
                                     alias_to_axis(pl.col("t"), "t")),
                    how="cross")
              .sort("p", "d", "t")
              .collect())


# ---------------------------------------------------------------------------
# §3.3.1 — flow_to_n / flow_from_n (block-aware filter on top of pss)
# ---------------------------------------------------------------------------


def flow_to_n(source: "InputSource",
              pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = sink``.  Schema ``[p, source, sink, n]``.

    Every arc enters the balance of its sink node (a node of a coarse
    new_stepduration group sums its fine flows, nodeBalanceCoarse_eq).
    """
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink"),
                                      "n": schema_dtype(_enums, "n")})
    return (pss.lazy()
              .with_columns(n=pl.col("sink"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


def flow_from_n(source: "InputSource",
                pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = source``.  Schema ``[p, source, sink, n]``."""
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink"),
                                      "n": schema_dtype(_enums, "n")})
    return (pss.lazy()
              .with_columns(n=pl.col("source"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


# ---------------------------------------------------------------------------
# §3.3.3 — p_flow_upper_existing (existing_count = capacity / unitsize per arc)
# ---------------------------------------------------------------------------


def _flow_upper_existing_from_chained_csv(source: "InputSource",
                                              pss: pl.DataFrame,
                                              workdir: Path | None,
                                              *,
                                              provider: "object | None" = None,
                                              ) -> "Param | None":
    """Γ.6.D — build ``p_flow_upper_existing`` from the canonical
    ``solve_data/p_entity_all_existing.csv`` when it exists, divided by
    the per-entity unitsize cascade and cross-joined with ``pss``.

    Returns None when the CSV is absent (caller falls back to the
    raw-existing path).
    """
    pae_csv = _read_p_entity_all_existing_csv(workdir, provider=provider)
    if pae_csv is None or pae_csv.height == 0:
        return None
    proc_set, _, _ = _entity_classes_lookup(source)
    per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
    if per_proc.height == 0:
        return None
    base = _process_capacity_per_unit_lf(per_proc.lazy(), source)
    if pss is None or pss.height == 0:
        return None
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d"), out)


def p_flow_upper_existing_from_source(source: "InputSource",
                                         pss: pl.DataFrame,
                                         active_solve: str | None = None,
                                         workdir: Path | None = None,
                                         *,
                                         ctx: "SolveContext | None" = None,
                                         provider: "object | None" = None,
                                         ) -> "Param | None":
    """Per-arc structural existing bound: the existing unit count (capacity / unitsize per (p, d), =
    ``existing`` when virtual_unitsize is above 0) cross-joined with ``pss``.

    capacity[p, d] = existing[p, d] × virtual_unitsize[p] when virtual_unitsize is above 0, else existing[p, d];
    unitsize = virtual_unitsize above 0, else existing, else 1000 (:func:`_entity_unitsize_lf`).  The capacity comes from
    the canonical chained ``p_entity_all_existing`` (multi-solve handoff and lifetime gate integrated) when the run
    has emitted it; otherwise from ``p_entity_pre_existing`` (the lifetime-gated existing × virtual_unitsize
    cascade) over ``period_in_use`` (:func:`_fallback_period_in_use`) for the units / connections with an explicit
    ``existing``.

    Schema: ``Param[(p, source, sink, d), value]``.

    Returns ``None`` when the source has no usable ``existing`` data —
    caller falls back to CSV value.
    """
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None and pss is not None:
        pss = cast_frame_axes(pss, _enums)
    chained = _flow_upper_existing_from_chained_csv(
        source, pss, workdir, provider=provider)
    if chained is not None:
        return chained

    if pss is None or pss.height == 0:
        return None
    periods = _fallback_period_in_use(source, workdir, ctx=ctx, provider=provider)
    cap_lf = _pre_existing_process_capacity_lf(source, active_solve, periods, workdir, provider=provider)
    if cap_lf is None:
        return None
    base = _process_capacity_per_unit_lf(cap_lf, source)
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d"), out)


def _fallback_period_in_use(source: "InputSource",
                            workdir: Path | None = None,
                            *,
                            ctx: "SolveContext | None" = None,
                            provider: "object | None" = None,
                            ) -> list[str]:
    """``period_in_use`` for the provider-less existing fallback.

    flextool's per_solve_sets.py:95-101 defines ``period_in_use`` as the union of realised + invest periods plus
    stochastic-branch siblings.  The in-memory :pyattr:`SolveContext.period_in_use` frame is consulted first, then
    the canonical ``period_in_use_set.csv`` (includes stochastic branches) when ``workdir`` is provided; otherwise
    the realised + invest union of the source's solves.
    """
    if ctx is not None:
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            return [str(d) for d in piu_ctx.get_column("d").unique(maintain_order=True).to_list()]
    if workdir is not None:
        piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if _provider_has_key(provider, piu_path):  # Step 1-g-5 — Provider-first
            piu_df = _provider_read(provider, piu_path)
            if piu_df.height > 0 and "period" in piu_df.columns:
                return [str(d) for d in piu_df.get_column("period").unique(maintain_order=True).to_list()]
    periods: list[str] = []
    for par in ("realized_periods", "invest_periods"):
        rp = _try_param(source, "solve", par)
        if rp is None:
            continue
        periods.extend(str(d) for d in rp.get_column("value").to_list())
    return list(dict.fromkeys(periods))


# ---------------------------------------------------------------------------
# §3.5.1 — p_flow_constraint_coef
# ---------------------------------------------------------------------------


def p_flow_constraint_coef_from_source(source: "InputSource",
                                          pss: pl.DataFrame,
                                          ) -> "Param | None":
    """User-defined per-arc (p, source, sink, cn) ``constraint_flow_coeff``.

    The constraint axis column is ``cn`` (not ``c``) to avoid collision
    with the commodity axis — see the c_collision review note in
    ``schemas/flextool_axis_contract.json``.

    Algorithm (audit §3.5.1, flex_data.py:1005-1019):
      1. Pull unit__inputNode / unit__outputNode /
         connection__node.constraint_flow_coeff.  Each row carries
         ``(process, node, constraint, coef)``.
      2. Inner-join with pss on (p, n=source) → source-leg rows.
      3. Inner-join with pss on (p, n=sink) → sink-leg rows.
      4. Concat + group_by(p, source, sink, cn).sum.

    Returns ``None`` when no ``constraint_flow_coeff`` rows exist.
    """
    if pss is None or pss.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None:
        pss = cast_frame_axes(pss, _enums)
    parts: list[pl.LazyFrame] = []
    for entity_class in ("unit__inputNode", "unit__outputNode",
                          "connection__node"):
        df = _try_param(source, entity_class, "constraint_flow_coeff")
        if df is None:
            continue
        # Schema: <process_col>, <node_col>, <constraint_index_col>, value.
        # constraint_flow_coeff is a 1d_map[constraint] → coef.
        cols = df.columns
        # Disambiguate process / node columns by entity_class layout.
        p_col = ("unit" if entity_class.startswith("unit")
                  else "connection")
        n_col = "node"
        c_col = next((c for c in cols
                       if c not in (p_col, n_col, "value")), None)
        if c_col is None:
            continue
        parts.append(df.lazy().select(
            alias_to_axis(p_col, "p"),
            alias_to_axis(n_col, "n"),
            alias_to_axis(c_col, "cn"),
            pl.col("value").cast(pl.Float64).alias("coef"),
        ))
    if not parts:
        return None
    coef = pl.concat(parts)

    # Cross-axis joins: pss carries ``source``/``sink`` (e-axis) joined
    # against ``coef.n`` (n-axis).  Per the contract n ⊂ e; up-cast
    # coef.n to e-Enum once and the join composes natively in Enum
    # without Utf8 materialisation at the join boundary.
    coef_e = coef.with_columns(cast_dim(pl.col("n"), None, "e"))
    src_match = (pss.lazy()
        .join(coef_e, left_on=["p", "source"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "cn", "coef"))
    sink_match = (pss.lazy()
        .join(coef_e, left_on=["p", "sink"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "cn", "coef"))
    joined = (pl.concat([src_match, sink_match], how="vertical")
                .group_by(["p", "source", "sink", "cn"])
                .agg(pl.col("coef").sum().alias("value"))
                .sort("p", "source", "sink", "cn")
                .collect())
    if joined.height == 0:
        return None
    return Param(("p", "source", "sink", "cn"), joined)


# ---------------------------------------------------------------------------
# §3.3.4 — p_slope (efficiency-curve slope)
# ---------------------------------------------------------------------------


def _eff_param_to_pdt(source: "InputSource",
                       entity_class: str,
                       parameter_name: str,
                       dt: pl.DataFrame,
                       value_col_alias: str,
                       ) -> "pl.LazyFrame | None":
    """Resolve an efficiency-family time parameter to a
    ``[p, d, t, <value_col_alias>]`` LazyFrame via the authoritative
    shape resolver (:mod:`._param_shapes`).

    Replaces the column-name-based ``_broadcast_param_to_dt``: the Spine
    Map index label is user-set (``"x"`` is the spinedb_api default when
    unset), so name matching mis-classified time-series parameters as
    scalars and broadcast them with a ``how="cross"`` cartesian product
    (process × full (d, t) grid) — which both overflowed polars' 2^32
    row-index on long timelines and, on short ones, silently flattened
    the time variation via the downstream ``unique(["p","d","t"])``.

    The resolver reads the parameter's nesting depth + per-level
    ``index_name`` from the DB and (for silent-default labels) probes the
    index values against ``dt``'s period / timestep domains, so a
    ``Map(time → value)`` resolves to ``Shape.MAP_TIME`` regardless of
    label.  :func:`broadcast_to_period_time` then keeps the Param at its
    authored shape (no cross-product) and :func:`promote_param_to_dt`
    widens it to ``(p, d, t)`` with a *narrowing* inner-join on the axis
    the source carries.

    A unit authoring a stochastic-time value takes the solve's emit-folded values
    (:func:`._param_shapes.broadcast_to_period_time`); the other units keep their own shapes.

    Returns ``None`` when the parameter has no usable rows (mirrors the
    old helper's ``None`` contract so callers' gating is unchanged).
    """
    resolved = resolve_param_shape(
        source, entity_class, parameter_name, period_filter=dt)
    param = broadcast_to_period_time(resolved, "p", dt)
    if param is None:
        return None
    return (promote_param_to_dt(param, dt)
                .select("p", "d", "t",
                        pl.col("value").cast(pl.Float64).alias(value_col_alias)))


def _mle_eta_min(eta_min: pl.Expr, eta: pl.Expr) -> pl.Expr:
    """Efficiency at minimum load of a min_load_efficiency cell: ``efficiency_at_min_load`` where given, else
    ``efficiency`` (a flat curve), cell by cell.  Shared by ``p_slope_from_source`` and
    ``p_section_from_source`` (the rolling emit seed ``_period_params._derive_conversion_trio`` applies the same
    fallback)."""
    return pl.coalesce(eta_min, eta)


def _mle_section(eta: pl.Expr, eta_min: pl.Expr, min_load: pl.Expr) -> pl.Expr:
    """Section (input per online unit) of a min_load_efficiency cell, the curve through (min_load, eta_min) and
    (1, eta):

        section = min_load · (1/eta_min − 1/eta) · (1 / (1 − min_load))    (min_load 0 → 0 whatever eta_min is)

    Unrounded.  Algebraically 1/eta − (1/eta − min_load/eta_min)/(1 − min_load), but written as a product of
    the efficiency difference so that a flat curve (eta_min = eta) gives exactly 0: the difference form left a
    float residue (−2.2e-16 for eta 0.9, min_load 0.3) on v_online that reached the objective as a ~1e-17 cost,
    a 16-decade cost range that autoscale centred with a 2^25 objective scale and HiGHS' dual simplex then failed
    on (excessive dual values, no model status).  The rolling emit seed
    ``_period_params._derive_conversion_trio`` evaluates the same float operations in the same order.

    The division by ``1 − min_load`` is a multiplication by its reciprocal.  Polars divides a column by a
    scalar-valued (broadcast) column as a multiplication by the reciprocal, but a column by a column as a true
    division, and whether ``min_load`` stays scalar-valued depends on how it was authored (a constant or a
    period map) and on the polars version; the two differ by an ulp.  With only multiplications and
    ``1.0 / x`` divisions the bits do not depend on the representation, so a constant and its equivalent map
    give the same section on every path."""
    return (pl.when(min_load == 0.0).then(0.0)
            .otherwise(min_load * (1.0 / eta_min - 1.0 / eta) * (1.0 / (1.0 - min_load))))


def _mle_slope(eta: pl.Expr, eta_min: pl.Expr, min_load: pl.Expr) -> pl.Expr:
    """Slope of input over output of a min_load_efficiency cell:

        slope = 1/eta − section      (= (1/eta − min_load/eta_min) / (1 − min_load);  section 0 → 1/eta)

    Shared by ``p_slope_from_source`` and ``p_section_from_source`` (via ``_mle_section``) so that
    slope + section = 1/eta on every cell and every path (single, rolling, nested, stochastic) reads identical
    values."""
    return 1.0 / eta - _mle_section(eta, eta_min, min_load)


def _mle_curve_inputs(source: "InputSource", dt: pl.DataFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """``base`` (p, d, t, …, eta) with ``eta_min`` and ``min_load`` added for the min_load_efficiency curve.

        min_load[p,d,t] = authored value, 0 where not authored (no minimum load)
        eta_min[p,d,t]  = efficiency_at_min_load where authored, else efficiency  (_mle_eta_min)
    """
    if parameter_has_values(source, "unit", "efficiency_at_min_load"):
        eta_min_lf = _eff_param_to_pdt(source, "unit", "efficiency_at_min_load", dt, "eta_min")
    else:
        eta_min_lf = None
    ml_lf = (_eff_param_to_pdt(source, "unit", "min_load", dt, "min_load")
             if parameter_has_values(source, "unit", "min_load") else None)
    base = (base.join(eta_min_lf, on=["p", "d", "t"], how="left") if eta_min_lf is not None
            else base.with_columns(eta_min=pl.lit(None).cast(pl.Float64)))
    base = (base.join(ml_lf, on=["p", "d", "t"], how="left") if ml_lf is not None
            else base.with_columns(min_load=pl.lit(None).cast(pl.Float64)))
    return base.with_columns(min_load=pl.col("min_load").fill_null(0.0),
                             eta_min=_mle_eta_min(pl.col("eta_min"), pl.col("eta")))


def _raise_on_null_mle_cells(df: pl.DataFrame, what: str) -> None:
    """A min_load_efficiency cell without a value would be a unit without fuel use there: refuse it."""
    if df.height:
        cells = ", ".join(f"({p}, {d}, {t})" for p, d, t in df.select(
            pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).head(5).iter_rows())
        raise ValueError(
            f"{what}: min_load_efficiency cells without a value (the unit would use no fuel there): {cells}; "
            f"this is a loader bug (min_load, efficiency_at_min_load and efficiency all have defaults).")


def p_slope_from_source(source: "InputSource",
                          dt: pl.DataFrame,
                          classified: pl.DataFrame | None = None,
                          ) -> "Param | None":
    """Conversion-curve slope per (p, d, t).

    For ``constant_efficiency`` ct_method: slope = 1 / efficiency.
    For ``min_load_efficiency`` ct_method: slope is linearised between
    (min_load, efficiency_at_min_load) and (1.0, efficiency); per
    audit §3.3.4 / entity_period_calc_params.py:1209-1306.

    On a direct (one-input, one-output) unit, the slope above is further scaled by
    ``k_sink/k_src``, the unit's output/input ``conversion_flow_coeff`` (default 1 each; #10 / Q12),
    folded in by :func:`_arithmetic._direct_flow_coef_ratio` — the same ratio the rolling emit seed
    (``emit._period_params._derive_conversion_trio``) applies.  A zero coefficient keeps its
    uncapped-edge meaning instead (ratio left at 1, with a warning naming the unit).

    Time-varying efficiency Maps are broadcast to the (d, t) grid;
    scalar efficiency is broadcast to the full grid.

    Returns ``None`` when no usable efficiency data is available.
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
        classified = cast_frame_axes(classified, _enums)
    # Pull efficiency / efficiency_at_min_load / min_load from units +
    # transfer connections (efficiency lives on both unit and connection
    # though connection.efficiency may be implicit via transfer_method).
    # Presence checks only (shape-agnostic: a stochastic-time unit value is not classified here).
    eff_unit = parameter_has_values(source, "unit", "efficiency")
    eff_conn = parameter_has_values(source, "connection", "efficiency")

    # Build a per-(p, d, t) efficiency lazyframe — union unit + connection.
    # Routed through the authoritative shape resolver so time-series
    # efficiency (e.g. connection.efficiency authored as Map(time→value)
    # with the spinedb_api silent-default ``"x"`` index label) is narrowed
    # to the active timeline instead of cross-joined; see
    # :func:`_eff_param_to_pdt`.
    eff_lfs: list[pl.LazyFrame] = []
    if eff_unit:
        e = _eff_param_to_pdt(source, "unit", "efficiency", dt, "eta")
        if e is not None:
            eff_lfs.append(e)
    if eff_conn:
        e = _eff_param_to_pdt(source, "connection", "efficiency", dt, "eta")
        if e is not None:
            eff_lfs.append(e)
    # Default-fill for processes in the classifier but missing from the
    # explicit efficiency rows.  flextool's default is 1.0 (constant_eff
    # ct_method baseline) — broadcast to the full classified × dt grid
    # then left-join with explicit rows so explicit overrides win.
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (classified carries Enum p / ct / klass).
    classified_dt = (classified.lazy()
        .select("p", "ct", "klass")
        .join(dt.lazy()
                 .with_columns(alias_to_axis(pl.col("d"), "d"),
                               alias_to_axis(pl.col("t"), "t")),
              how="cross")
        .select("p", "d", "t", "ct", "klass"))
    if eff_lfs:
        explicit = pl.concat(eff_lfs).unique(subset=["p", "d", "t"])
        eff_lf = (classified_dt
                    .join(explicit, on=["p", "d", "t"], how="left")
                    .with_columns(eta=pl.col("eta").fill_null(1.0)))
    else:
        eff_lf = classified_dt.with_columns(eta=pl.lit(1.0))

    # constant_efficiency branch: slope = 1 / eta.
    base = eff_lf  # already has p, d, t, ct, klass, eta

    # min_load_efficiency rows: the slope of input over output between (min_load, eta_min) and (1, eta):
    #   slope[p,d,t] = 1/eta − section,  section = min_load·(1/eta_min − 1/eta) / (1 − min_load)  (_mle_slope)
    #   eta = efficiency (default 1.0);  min_load = 0 where not authored;
    #   eta_min = efficiency_at_min_load, else efficiency (_mle_eta_min: a flat curve, slope = 1/eta)
    # A min_load of 0 gives the constant-efficiency slope 1/eta whatever eta_min is.
    if classified.filter(pl.col("ct") == "min_load_efficiency").height > 0:
        base = _mle_curve_inputs(source, dt, base)
    else:
        base = base.with_columns(eta_min=pl.lit(None).cast(pl.Float64),
                                 min_load=pl.lit(0.0))

    # slope expression: branch on ct.
    base = base.with_columns(
        slope=pl.when(pl.col("ct") == "min_load_efficiency")
                .then(_mle_slope(pl.col("eta"), pl.col("eta_min"), pl.col("min_load")))
              .otherwise(1.0 / pl.col("eta"))
    )
    # #10 / Q12: fold a direct unit's output/input conversion_flow_coeff ratio into its slope —
    # slope = slope⁰·k_sink/k_src (Q-A; _arithmetic._direct_flow_coef_ratio is the shared direct-set
    # definition with the rolling emit seed, _period_params._derive_conversion_trio).
    from flextool.engine.preprocess.params._arithmetic import _direct_flow_coef_ratio
    ratio = _direct_flow_coef_ratio(source)
    if ratio.height > 0:
        base = (base.join(ratio.lazy(), on="p", how="left")
                     .with_columns(
                         slope=pl.when(pl.col("k_sink").is_not_null())
                                 .then(pl.col("slope") * pl.col("k_sink") / pl.col("k_src"))
                                 .otherwise(pl.col("slope"))))
    out = (base.select("p", "d", "t", pl.col("slope").alias("value"))
                .sort("p", "d", "t")
                .collect())
    _raise_on_null_mle_cells(out.filter(pl.col("value").is_null()), "p_slope")
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.3.4b — p_min_load (minFlow_minload floor), shape-correct (p, d, t)
# ---------------------------------------------------------------------------


def p_min_load_pdt_from_source(source: "InputSource",
                                 dt: pl.DataFrame,
                                 ) -> "Param | None":
    """``unit.min_load`` → ``Param(("p", "d", "t"), [p, d, t, value])``.

    Routed through the authoritative shape resolver (:func:`_eff_param_to_pdt`,
    the same path the efficiency-curve ``p_slope`` / ``p_section`` already use)
    so a period / time / period-time ``Map`` keeps its per-(d, t) semantics.

    This replaces the old ``(p,)`` direct param (``_direct``
    ``p_min_load_from_source`` → ``_entity_scalar_explicit``) whose
    ``.select(dim, "value")`` dropped the Map index column: a Map min_load
    then yielded one ``(unit, value)`` row per key, and polar_high **summed**
    the duplicate ``(p)`` keys in the ``minFlow_minload`` constraint
    (``model/core.py``) — a period Map ``{p1: 0.4, p2: 0.3}`` produced a floor of
    0.7 in *every* period.  See ``specs/fix_map_reading.md`` §5.

    A **scalar** min_load is broadcast to the full ``(d, t)`` grid, so the
    constraint RHS (``v_online · min_load``) is byte-identical to the old
    ``(p,)`` broadcast — only Map authoring changes behaviour.

    Returns ``None`` when no unit authors ``min_load`` (no value = 0, no
    floor).  The online feature does not need this field; ``model/core.py``
    checks it (``MIN_LOAD``) only when some unit has ``min_load > 0``
    (``process_minload``).
    """
    if dt is None or dt.height == 0:
        return None
    # Defend axis-aware join keys on the incoming frame param, mirroring
    # ``p_slope_from_source``'s treatment of ``dt`` before ``_eff_param_to_pdt``.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
    lf = _eff_param_to_pdt(source, "unit", "min_load", dt, "value")
    if lf is None:
        return None
    out = lf.sort("p", "d", "t").collect()
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


def process_minload_from_p_min_load(p_min_load: "Param | None") -> pl.DataFrame | None:
    """``process_minload`` ``(p,)``: the units whose ``min_load`` is above 0 on some (d, t) of this solve.

    Read from the solve-resolved ``p_min_load`` (not the raw parameter), so a unit whose map has values only in
    other periods of the model gets no ``minFlow_minload`` floor, and the floor's units always have a
    ``p_min_load`` row.  None when no unit qualifies.
    """
    if p_min_load is None:
        return None
    out = (p_min_load.frame.lazy()
           .filter(pl.col("value") > 0)
           .select(pl.col("p"))
           .unique()
           .sort("p")
           .collect())
    return out if out.height > 0 else None


# ===========================================================================
# Γ.3.B integration entrypoint
# ===========================================================================


DERIVED_B_FIELDS = (
    "p_flow_upper_existing",
    "p_flow_constraint_coef",
    "p_slope",
    "p_min_load",
)


def apply_derived_b(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.B Derived Params, mutating ``flex_data`` in place.

    The classifier (``_classify_process_method``) is the single
    dependency for the reclassified ``process_indirect`` / ``flow_to_n``
    family; it's computed once per call and threaded through the
    per-Param helpers as a kwarg.

    Δ.3 replaced the previous ``derived_overrides_b`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    # Active solve (used for existing-period broadcast in some helpers).
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)

    # Build the classifier once.  Δ.12b: hard errors here would
    # indicate a malformed source — let them propagate instead of
    # silently producing an empty schema.
    classified = _classify_process_method(source)

    # Topology — already-computed CSV-side process_source_sink (a plain
    # DataFrame on FlexData; see flex_data.py:252).
    pss_frame = getattr(flex_data, "process_source_sink", None)
    dt_csv = getattr(flex_data, "dt", None)

    # process_indirect / process_input_flows / process_output_flows /
    # process_indirect_dt.  Δ.12b: helpers produce or raise.  ``None``
    # / empty frame == "no indirect units" — same semantic the seed
    # produces.  We keep the ``height > 0`` filter on the assignment
    # so the SET-frame contract (None or non-empty) is preserved.
    if classified.height > 0:
        pi_db = process_indirect_set(source, classified)
        if pi_db is not None and pi_db.height > 0:
            flex_data.process_indirect = pi_db
        else:
            flex_data.process_indirect = None

        if pss_frame is not None and pss_frame.height > 0:
            pif_db = process_input_flows(source, pss_frame, classified)
            flex_data.process_input_flows = (
                pif_db if pif_db is not None and pif_db.height > 0
                else None)

            pof_db = process_output_flows(source, pss_frame, classified)
            flex_data.process_output_flows = (
                pof_db if pof_db is not None and pof_db.height > 0
                else None)

        # Phase E.3: ``process_indirect_dt`` is no longer materialised
        # eagerly; consumers call
        # ``_pdt_join.compute_process_indirect_dt`` on demand.  The
        # field stays declared on ``FlexData`` for ``getattr`` defenders
        # but defaults to ``None``.

    # flow_to_n / flow_from_n: every arc enters the balance of both its nodes (a coarse node sums its fine
    # flows over each coarse step, nodeBalanceCoarse_eq).
    if pss_frame is not None and pss_frame.height > 0:
        from flextool.engine.preprocess.params._block import (
            flow_from_n_frame,
            flow_from_nodeBalance_seed,
            flow_to_n_frame,
        )
        ftn_db = flow_to_n_frame(pss_frame)
        ffn_db = flow_from_n_frame(pss_frame)
        if ftn_db is not None and ftn_db.height > 0:
            flex_data.flow_to_n = ftn_db
        if ffn_db is not None and ffn_db.height > 0:
            flex_data.flow_from_n = ffn_db

        # ─── Δ.27 flow_from_nodeBalance_{eff,noEff} ─────────────────
        # Source-side nodeBalance topology — native port of the
        # inline derivation in ``flex_data.py::_load_storage`` lines
        # 1658-1705.  Without these, source-side flows from
        # nodeBalance nodes (e.g. transmission discharge into a
        # balance node) don't appear in nodeBalance_eq and the LP
        # decouples on multi-arc fixtures
        # (``work_lh2_three_region``).
        # Source-side flows must be gathered for BOTH plain ``nodeBalance``
        # nodes and ``balance_within_period`` (``nodeBalancePeriod``) nodes —
        # a period node that is a *source* on an arc (e.g. GAS → gas plant)
        # must contribute its source-side flow term to ``nodeBalancePeriod_eq``,
        # else the period balance is vacuous and the source draws for free.
        # Pass the dtype-safe union to the seed (which filters ``pss.source ∈
        # n``).  Both frames come from the same ``rename_to_axis({"node":"n"})``
        # in ``flex_data.py:_load_node`` so the ``n`` Enum aligns.  When the period
        # set is empty/None the union collapses to exactly ``nb_for_ffnb``
        # (no concat/unique/reorder) → byte-identical legacy behavior.
        nb_for_ffnb = getattr(flex_data, "nodeBalance", None)
        _nbp_for_ffnb = getattr(flex_data, "nodeBalancePeriod", None)
        if (nb_for_ffnb is not None and _nbp_for_ffnb is not None
                and _nbp_for_ffnb.height > 0):
            nb_for_ffnb = (pl.concat([nb_for_ffnb.select("n"),
                                      _nbp_for_ffnb.select("n")])
                             .unique()
                             .select("n"))
        pss_eff_frame = getattr(flex_data, "process_source_sink_eff", None)
        pss_noEff_frame = getattr(flex_data, "process_source_sink_noEff",
                                    None)
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_eff_frame is not None
                and pss_eff_frame.height > 0):
            ffnb_eff = flow_from_nodeBalance_seed(pss_eff_frame, nb_for_ffnb)
            if ffnb_eff is not None and ffnb_eff.height > 0:
                flex_data.flow_from_nodeBalance_eff = ffnb_eff
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_noEff_frame is not None
                and pss_noEff_frame.height > 0):
            ffnb_noEff = flow_from_nodeBalance_seed(pss_noEff_frame, nb_for_ffnb)
            if ffnb_noEff is not None and ffnb_noEff.height > 0:
                flex_data.flow_from_nodeBalance_noEff = ffnb_noEff

    # ─── §3.3.3 p_flow_upper_existing ──────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.  Helper returns None
    # for fixtures without entity-existing-capacity.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_upper_existing = p_flow_upper_existing_from_source(
            source, pss_frame, active_solve, workdir,
            ctx=ctx, provider=provider)

    # ─── §3.3.3b p_arc_max_cap_coef (per-arc capacity_max_coeff) ───────
    # ``p_flow_upper_existing`` is existing_count only and drops the
    # per-arc ``capacity_max_coeff`` that the .mod's maxToSink /
    # maxFromSource apply.  Carry the factor here so the model/core.py maxFlow
    # consumer can fold it onto every arc bound by the existing-only RHS
    # (direct arcs + indirect output arcs; indirect input arcs carry it
    # via ``p_flow_upper``).
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_arc_max_cap_coef = p_arc_max_cap_coef_from_source(
            source, pss_frame, classified)
        flex_data.process_source_sink_uncapped = (
            process_source_sink_uncapped_from_source(source, pss_frame))

    # ─── §3.5.1 p_flow_constraint_coef ─────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_constraint_coef = p_flow_constraint_coef_from_source(
            source, pss_frame)

    # ─── §3.3.4 p_slope ────────────────────────────────────────────────
    # Δ.12b: unconditional when (dt, classified) are non-empty.
    if dt_csv is not None and dt_csv.height > 0 and classified.height > 0:
        flex_data.p_slope = p_slope_from_source(source, dt_csv, classified)

    # ─── §3.3.4b p_min_load (minFlow_minload floor, (p, d, t)) ──────────
    # Shape-correct (p, d, t) min_load routed through the efficiency
    # resolver so a Map isn't summed in the minFlow_minload constraint.
    # Gated on dt only (min_load applies to any online+minload unit,
    # independent of the ct_method classifier).  See
    # ``p_min_load_pdt_from_source`` / ``specs/fix_map_reading.md`` §5.
    if dt_csv is not None and dt_csv.height > 0:
        flex_data.p_min_load = p_min_load_pdt_from_source(source, dt_csv)

    # ─── §F.1 p_unitsize  (Δ.10 cluster F) ─────────────────────────────
    # Per-process unitsize cascade restricted to processes appearing in
    # ``pss``.  Mirrors ``flex_data.py:800-825``'s
    # ``unitsize_long.filter(p ∈ pss["p"].unique())``.
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        from flextool.engine.preprocess.params._arithmetic import p_unitsize_from_source
        flex_data.p_unitsize = p_unitsize_from_source(source, pss_frame)

    # ─── §F.1b p_all_entity_unitsize  (scaling family — all entities) ────
    # Unfiltered entity unitsize covering processes + connections + nodes.
    # Used by the scaling analyzer (scaling.py:analyze_solve) to compute the
    # full entity-unitsize spread including node entries, and by the output
    # reader (``read_parameters._build_entity_unitsize_series``) as the
    # complete entity→unitsize carrier.
    # Note: _entity_unitsize_lf returns all entities (unit ∪ node ∪ connection)
    # with the cascade: virtual_unitsize OR existing OR 1000.0.
    #
    # This value is solve-invariant (computed against the whole-model,
    # unfiltered-scenario source).  In the DB cascade path it is computed
    # ONCE before the per-solve loop and seeded into the Provider under
    # ``input/p_all_entity_unitsize`` (see ``_orchestration._drive_cascade``),
    # so here we read it from the Provider rather than recompute it per roll.
    # The recompute is kept as the fallback for the CSV/fixture path and any
    # path that lacks the seeded Provider.
    from polar_high import Param as _Param
    _all_us_df = None
    if provider is not None:
        _seeded = provider.get("input/p_all_entity_unitsize")
        if _seeded is not None and _seeded.height > 0:
            _all_us_df = _seeded
    if _all_us_df is None:
        _all_us_lf = _entity_unitsize_lf(source)
        _all_us_df = _all_us_lf.rename({"us": "value"}).collect()
    if _all_us_df.height > 0:
        flex_data.p_all_entity_unitsize = _Param(("e",), _all_us_df)

    # ─── §F.4 p_process_source_conversion_flow_coeff /
    # p_process_sink_conversion_flow_coeff ─────────────────────────────
    # (Δ.10 cluster F).  Mirrors ``flex_data.py:_load_indirect`` lines
    # 950-1002.  Anti-joins zero-coef rows out of the indirect-process
    # input/output flow sets and emits a Param keyed on (p, source) /
    # (p, sink) when any non-default, non-zero coef remains.
    # Δ.12b: helper exceptions propagate; the anti-join inner gate
    # (only when zero-rows exist) is preserved as a structural filter.
    from flextool.engine.preprocess.params._arithmetic import (
        p_process_source_conversion_flow_coeff_from_source,
        p_process_sink_conversion_flow_coeff_from_source,
    )
    pif = getattr(flex_data, "process_input_flows", None)
    pof = getattr(flex_data, "process_output_flows", None)
    z_src, p_src_coef = p_process_source_conversion_flow_coeff_from_source(
        source, pif)
    if z_src is not None and z_src.height > 0 and pif is not None \
            and pif.height > 0:
        new_pif = pif.join(z_src, on=["p", "source"], how="anti")
        if new_pif.height < pif.height:
            flex_data.process_input_flows = new_pif
    # TODO(Δ.12b helper-fix):
    # p_process_source_conversion_flow_coeff_from_source returns None when
    # no zero-coef rows exist in the source — but the seed-side
    # _load_indirect builds the coefficient Param from a different code
    # path (process__commodity__node_conversion_flow_coeff.csv).  Keep
    # the conditional assignment until the helpers converge on a single
    # producer.
    if p_src_coef is not None:
        flex_data.p_process_source_conversion_flow_coeff = p_src_coef

    z_sink, p_sink_coef = p_process_sink_conversion_flow_coeff_from_source(
        source, pof)
    if z_sink is not None and z_sink.height > 0 and pof is not None \
            and pof.height > 0:
        new_pof = pof.join(z_sink, on=["p", "sink"], how="anti")
        if new_pof.height < pof.height:
            flex_data.process_output_flows = new_pof
    if p_sink_coef is not None:
        flex_data.p_process_sink_conversion_flow_coeff = p_sink_coef


# ===========================================================================
# Γ.3.C — invest/divest + online/UC + group slack + existing fixed cost
# ===========================================================================
#
# Per the audit doc §3.7 / §3.8 / §3.11 / §3.12.  Each helper composes a
# lazy chain over ``InputSource`` reads and ``.collect()``s once at the
# boundary.  Each overlay is gated on a frame-equal pre-check against the
# CSV-loaded value (``apply_derived_c``) so multi-year invest cascades
# / scaling-active fixtures fall back to the CSV value instead of being
# corrupted by a simple-path approximation.
#
# Integration: ``apply_derived_c`` runs after Γ.3.A + Γ.3.B; see
# ``flex_data.py::load_flextool``.

# ---------------------------------------------------------------------------
# §3.7.0 — Helpers (active-solve scope, period_invest, p_years_d, etc.)
# ---------------------------------------------------------------------------


def _resolve_synthetic_solve(source: "InputSource",
                                  active_solve: str | None,
                                  ) -> tuple[str, str] | None:
    """Recognise a synthetic per-sub-solve name of the form
    ``<base>_<anchor>`` and return ``(base, anchor)``.

    Δ.19 — flextool's orchestrator synthesises per-period sub-solve
    names at runtime by joining a Spine ``solve`` name with one of its
    2-D period map's outer keys (``SolveConfig._split_2d_period_maps``
    makes ``<base>_<anchor>`` with ``derive_solve(<base>, ...)``).  These synthetic
    names don't exist as rows in Spine; the per-solve override chain
    would otherwise return None for every ``solve``-keyed parameter.

    Algorithm: try every suffix split (rightmost underscore first) and
    return the first ``(base, anchor)`` where ``base`` is a Spine
    ``solve`` entity.  ``base`` may itself contain underscores
    (``invest_5weeks_p2020`` splits as ``base='invest_5weeks'``,
    ``anchor='p2020'``).

    Returns ``None`` when no split matches a Spine solve.
    """
    if active_solve is None or "_" not in active_solve:
        return None
    try:
        ents = source.entities("solve")
    except KeyError:
        return None
    if ents is None or ents.height == 0:
        return None
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    spine_solves = set(ents[name_col].to_list())
    parts = active_solve.split("_")
    for i in range(len(parts) - 1, 0, -1):
        base = "_".join(parts[:i])
        anchor = "_".join(parts[i:])
        if base in spine_solves:
            return base, anchor
    return None


def _solve_periods(source: "InputSource", active_solve: str,
                    parameter_name: str) -> list[str] | None:
    """Read the period-list-valued ``solve.<parameter>`` for the active solve.

    ``solve.realized_periods`` is stored as a Spine Array
    (``[name, i (int), value=period]``); ``solve.invest_periods`` is
    stored either as an Array (single-anchor solves) or as a 2D Map
    (nested multi-invest solves: ``[name, x=anchor, i=period,
    value="yes"]``).  This helper returns the period list regardless of
    shape.

    The inner Map level's column name is taken from the Spine
    ``index_name`` and is not guaranteed to be ``"i"`` — some
    hand-authored fixtures (e.g. ``examples.sqlite::invest_5weeks``)
    leave both Map levels named ``"x"``, producing the post-unroll
    shape ``[name, x, value]``.  We therefore compute the period
    column as ``df.columns[-2]`` (the last non-value column) instead
    of hardcoding ``"i"``.

    Δ.19 — when ``active_solve`` doesn't appear in the param table,
    fall back to the synthetic ``<base>_<anchor>`` recognition: for
    Map-shaped params filter ``name==base AND x==anchor`` and return
    the inner-period values; for Array-shaped params return
    ``[anchor]`` (the single realised period of the synthetic
    sub-solve).
    """
    if active_solve is None:
        return None
    df = _try_param(source, "solve", parameter_name)
    if df is None:
        return None
    is_map = "x" in df.columns
    # Inner-period column = deepest non-entity, non-value column.
    # For Map-shaped params this is ``df.columns[-2]`` (covers both
    # the canonical ``"i"`` shape and the hand-authored ``"x"``-named
    # inner-index shape from ``examples.sqlite``).  For Array shapes
    # the value column itself carries the period.
    period_col = df.columns[-2] if is_map else "value"
    sub = df.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        # Synthetic fallback — split active_solve as <base>_<anchor>.
        resolved = _resolve_synthetic_solve(source, active_solve)
        if resolved is None:
            return None
        base, anchor = resolved
        if is_map:
            sub = df.filter((pl.col("name") == base)
                              & (pl.col("x") == anchor))
            if sub.height == 0:
                return None
            return sub[period_col].cast(pl.Utf8, strict=False).to_list()
        # Array param: synthetic sub-solve realises only its anchor.
        # Confirm the anchor is in base's array; if not, return None.
        base_sub = df.filter(pl.col("name") == base)
        if base_sub.height == 0:
            return None
        base_periods = set(base_sub["value"].cast(pl.Utf8, strict=False).to_list())
        return [anchor] if anchor in base_periods else None
    # Non-synthetic path.  For Map params filter to the (most-common)
    # anchor's row set; in practice the non-synthetic Map case doesn't
    # surface (orchestrator always synthesises sub-solves), but if it
    # ever did we'd return the union of inner periods deterministically.
    if "i" in sub.columns and not is_map:
        sub = sub.sort("i")
    if is_map:
        return sub[period_col].cast(pl.Utf8, strict=False).unique(maintain_order=True).to_list()
    return sub["value"].to_list()


def _entityInvest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities (units + nodes + connections) whose
    ``invest_method`` ∉ ``_INVEST_NOT_ALLOWED``.  Returns a lazy frame
    with column ``e``.  Mirrors flextool's ``entityInvest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_INVEST_ALLOWED)


def _entityDivest_set(source: "InputSource") -> pl.LazyFrame:
    """Entities whose ``invest_method`` ∉ ``_DIVEST_NOT_ALLOWED``.
    Mirrors flextool's ``entityDivest`` set.
    """
    return _entity_invest_filter(source, _INVEST_METHODS_DIVEST_ALLOWED)


# The invest/retire method lists (_INVEST_METHODS_ALL / _INVEST_NOT_ALLOWED /
# _DIVEST_NOT_ALLOWED) are imported from params/_existing.py, their one definition.
_INVEST_METHODS_INVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _INVEST_NOT_ALLOWED
)
_INVEST_METHODS_DIVEST_ALLOWED: frozenset[str] = (
    _INVEST_METHODS_ALL - _DIVEST_NOT_ALLOWED
)

# Invest methods that lift every investment limit ("no_limit removes all
# limits even if there are limiting values set").  Both the invest-only
# and the invest+retire flavours qualify; the max-capacity cascade adds
# ``max_flow_for_unconstrained_variables`` for them.
INVEST_METHODS_INVEST_NO_LIMIT: frozenset[str] = frozenset((
    "invest_no_limit", "invest_retire_no_limit",
))
# Mirror ``invest_total_sets.py:25-32``.
_INVEST_METHODS_INVEST_TOTAL: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_INVEST_METHODS_DIVEST_TOTAL: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
# Mirror ``invest_divest_sets.py:33-40``.
_INVEST_METHODS_INVEST_PERIOD: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_INVEST_METHODS_DIVEST_PERIOD: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))


def _entity_invest_method(source: "InputSource") -> pl.LazyFrame | None:
    """Pull the per-entity invest_method as a (e, method) lazy frame.

    Spine source: ``unit.invest_method`` / ``node.invest_method`` /
    ``connection.invest_method``.  Returns None when no entity has an
    explicit method (default is ``not_allowed``).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").alias("method"),
        ))
    if not parts:
        return None
    return pl.concat(parts).unique()


def _entity_invest_filter(source: "InputSource",
                           allowed: frozenset[str]) -> pl.LazyFrame:
    """Lazy frame ``[e]`` of entities whose ``invest_method ∈ allowed``.
    """
    eim = _entity_invest_method(source)
    if eim is None:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e")}).lazy()
    return (eim.filter(pl.col("method").is_in(list(allowed)))
                .select("e").unique())


def _ed_period_param_lf(source: "InputSource",
                        parameter_name: str,
                        periods: list[str]) -> pl.LazyFrame | None:
    """``unit/node/connection.<parameter_name>`` as ``(e, d, value)`` over ``periods``.

    Read through the shape resolver (:func:`._direct._e_period_param_union` →
    ``resolve_param_shape`` → ``broadcast_to_period``): a constant is broadcast over the
    periods, a period map keeps its own per-period values (also when authored with the
    silent ``x`` index, and also next to a constant on another entity of the same class),
    and map keys outside ``periods`` are dropped.  Every anchor period's explicit value is
    inherited by its stochastic branch periods (:func:`._branch.inherit_active_branch_periods`
    — unconditionally, not only under recourse, #63) before the final restriction to
    ``periods``.

    Zero values are kept (``filter_zero=False``) so that a branch-native explicit 0 wins over
    its anchor's non-zero value in the expansion (the value only: whether an investment or
    retirement variable exists does not depend on the cost).  Anchor periods are added to the period
    filter so that their explicit rows survive the resolver's inner join until the expansion
    has used them.  ``None`` when no entity authors the parameter.
    """
    pairs = active_branch_anchor_pairs()
    anchors = ([] if pairs is None
               else [a for a in pairs["anchor"].unique(maintain_order=True).to_list()
                     if a not in periods])
    period_filter = axis_lazyframe({"d": list(periods) + anchors}).collect()
    param = _e_period_param_union(source, parameter_name, period_filter=period_filter)
    if param is None:
        return None
    frame = param.frame if isinstance(param.frame, pl.LazyFrame) else param.frame.lazy()
    lf = frame.select("e", "d", pl.col("value").cast(pl.Float64))
    lf = inherit_active_branch_periods(lf.collect(), group_keys=("e",), period_col="d").lazy()
    # Restrict to ``periods``: semi-join in the frame's own d dtype.
    periods_d = axis_lazyframe({"d": list(periods)})
    d_dtype = lf.collect_schema().get("d")
    if periods_d.collect_schema().get("d") != d_dtype:
        periods_d = periods_d.with_columns(pl.col("d").cast(d_dtype))
    return lf.join(periods_d, on="d", how="semi")


# ---------------------------------------------------------------------------
# §3.7.1 — ed_invest_set / ed_divest_set
# ---------------------------------------------------------------------------


def _read_period_first(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None,
                          *,
                          provider: "object | None" = None,
                          ) -> list[str]:
    """Return ``period_first`` for the active solve (Γ.6.D).

    The first period of the first timeset on the active solve, plus any
    stochastic-branch siblings (matching the legacy ``write_periods``
    output).

    Resolution order:
      1. ``solve_data/period_first.csv`` if present (canonical).
      2. First entry of ``solve.realized_periods`` (non-stochastic
         fallback for tests that don't have the CSV).

    Returns ``[]`` when no usable source.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_first.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p)
                if df.height > 0 and "period" in df.columns:
                    return df["period"].cast(pl.Utf8, strict=False).to_list()
            except Exception:
                pass
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seq = realized or invest
    return [seq[0]] if seq else []


def run_start_period(years: "dict[str, float]") -> str | None:
    """The period the model run starts in, among a solve's period years ``years`` (``p_years_d``: the period
    history first, each period's years from the start of the run, a branch period with its label's year): the
    first period with the smallest year.  ``None`` without periods.

    The end of life of pre-existing capacity counts from this period in every solve of the run (a later solve of
    a chain holds the earlier periods as its history), so a chain and a single solve age it alike, and a fan of
    the first period gives it one start, not one per branch."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    start: str | None = None
    for d, y in years.items():
        if start is None or y < years[start] - YEAR_TOL:
            start = d
    return start


def _lifetime_years(source: "InputSource",
                    active_solve: str | None,
                    workdir: Path | None,
                    *,
                    provider: "object | None" = None,
                    ) -> dict[str, float]:
    """``{period: year}`` the lifetime gate compares: :func:`_p_years_d_lf` (with the Provider, the solve's
    ``p_years_d``: years from the start of the run over its period history; without it, the solve's own
    ``years_represented``, the solve being the run)."""
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return {}
    pyd_df = pyd_lf.collect()
    out: dict[str, float] = {}
    for d, y in zip(pyd_df["d"].cast(pl.Utf8).to_list(), pyd_df["yr"].to_list()):
        out.setdefault(str(d), float(y if y is not None else 0.0))
    return out


def _lifetime_gate_sums(source: "InputSource",
                            active_solve: str | None,
                            workdir: Path | None,
                            *,
                            provider: "object | None" = None,
                            years: "dict[str, float] | None" = None,
                            ) -> dict[str, float]:
    """Per-entity end of life ``life_sum`` (a year of :func:`_lifetime_years`) of the pre-existing capacity of
    the ``reinvest_choice`` / ``no_investment`` entities::

        life_sum(e) = p_years_d[d0] + lifetime[e, d0],   d0 = run_start_period(p_years_d)

    One value per entity, counted from the start of the run (not from the solve's own first period, and not
    once per branch of a fan in the first period).  Other methods are not returned (no expiry gate); a
    connection without lifetime (not a process / node) gets 0.
    """
    pyd = years if years is not None else _lifetime_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(pyd)
    if start is None:
        return {}
    per_p = _per_entity_period_value(source, "lifetime") or {}
    scalar = _per_entity_scalar(source, "lifetime")
    process_set, node_set, _ = _entity_classes_lookup(source)
    all_e = _all_entities(source)
    elm = _entity_lifetime_methods(source, all_e)

    out: dict[str, float] = {}
    for e, methods in elm.items():
        if not (("reinvest_choice" in methods)
                or ("no_investment" in methods)):
            continue
        if e not in process_set and e not in node_set:
            # Connection without is_DC / non-process — no lifetime.
            out[e] = 0.0
            continue
        out[e] = pyd[start] + _resolve_pdX(per_p, scalar, e, start)
    return out


def _lifetime_expired_pairs(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None,
                                methods: tuple[str, ...] = (
                                    "reinvest_choice", "no_investment"),
                                *,
                                provider: "object | None" = None,
                                periods: "list[str] | None" = None,
                                ) -> set[tuple[str, str]]:
    """Set of (e, d) pairs whose pre-existing capacity has expired.

    For each entity whose ``lifetime_method`` is in ``methods``, the periods d of ``periods`` (default the
    solve's ``period_in_use``) with ``p_years_d[d] >= life_sum(e)`` (:func:`_lifetime_gate_sums`, one end of
    life per entity from the start of the run).  These are the rows where:

      * ``p_entity_pre_existing[e, d]`` should evaluate to 0.
      * ``ed_invest_forbidden_no_investment`` adds (e, d) to the
        forbidden set (only ``no_investment`` is forbidden — but the
        gate is shared with ``reinvest_choice`` for the existing-cap
        cascade).

    Only ``no_investment`` entries actually pin ``v_invest = 0`` —
    callers filtering ``ed_invest_set`` should pass ``methods=("no_investment",)``.
    """
    pyd = _lifetime_years(source, active_solve, workdir, provider=provider)
    sums = _lifetime_gate_sums(source, active_solve, workdir, provider=provider, years=pyd)
    if not sums:
        return set()
    # Restrict to entities whose method is in the requested filter.
    all_e = _all_entities(source)
    elm = _entity_lifetime_methods(source, all_e)
    keep_e = {e for e, ms in elm.items() if any(m in ms for m in methods)}
    sums = {e: v for e, v in sums.items() if e in keep_e}
    if not sums:
        return set()
    piu = (list(periods) if periods is not None
           else _period_in_use_set(source, active_solve, workdir, provider=provider))
    if not piu:
        return set()
    out: set[tuple[str, str]] = set()
    for e, s in sums.items():
        for d in piu:
            if pyd.get(str(d), 0.0) >= s:
                out.add((e, str(d)))
    return out


def ed_invest_forbidden_no_investment_from_source(
        source: "InputSource",
        active_solve: str | None,
        workdir: Path | None,
        ed_invest: pl.DataFrame | None = None,  # noqa: ARG001 — kept for back-compat
        *,
        ctx: "SolveContext | None" = None,
        provider: "object | None" = None,
        ) -> pl.DataFrame | None:
    """Audit §3.7 — ``ed_invest`` rows whose ``no_investment`` lifetime
    window has already ended.

    Mirrors :func:`flextool/preprocessing/invest_divest_sets.py
    :write_ed_invest_forbidden_no_investment` — used in
    ``fix_v_invest_no_investment_eq`` (mod L3930) to pin ``v_invest = 0``
    after the lifetime expires.

    Computed against the *unfiltered* ed_invest cross-product (so the
    output matches flextool's CSV regardless of whether
    :func:`ed_invest_set_from_source` has already anti-joined the
    forbidden tuples).  We rebuild the universe locally:
    ``entityInvest × period_invest`` then keep only the (e, d) pairs
    where:

      * ``e`` has ``lifetime_method == 'no_investment'``,
      * ``yr[d] >= life_sum(e)``.
    """
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, ctx=ctx, provider=provider)
    if not period_invest:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    forbidden = _lifetime_expired_pairs(
        source, active_solve, workdir, methods=("no_investment",), provider=provider)
    if not forbidden:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    # Universe = entityInvest × period_invest (canonical's source set,
    # before the forbidden filter is applied).
    ei_lf = _entityInvest_set(source)
    ei_df = ei_lf.collect() if ei_lf is not None else pl.DataFrame(
        schema={"e": schema_dtype(_enums, "e")})
    if ei_df.height == 0:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    ei = set(ei_df["e"].to_list())
    rows = [(e, d) for (e, d) in forbidden
              if e in ei and d in period_invest]
    if not rows:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    return pl.DataFrame(rows, schema=["e", "d"], orient="row").sort("e", "d")


def ed_invest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None = None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                ) -> pl.DataFrame | None:
    """Compute the (entity, period) pairs where invest is allowed.

    ::

        ed_invest = { (e, d) : e ∈ entityInvest, d ∈ period_invest_of_solve }

    Every entity whose ``invest_method`` allows investing gets its investment variable in every
    investment period of the solve, also without an ``invest_cost`` (or with a 0 in a period):
    such capacity has no annuity, its ``fixed_cost`` still counts.  (Before, a pair needed a
    non-zero annuity or a user-constraint capacity coefficient; :func:`zero_cost_invest_notice`
    names the pairs that rule left out.)

    Γ.6.D: ``ed_invest_forbidden_no_investment`` rows (entities with
    ``lifetime_method = no_investment`` whose lifetime window has
    expired) are anti-joined out.  Mirrors
    ``invest_divest_sets.py:write_ed_invest_forbidden_no_investment``
    consumed by ``fix_v_invest_no_investment_eq`` (mod L3930).
    """
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, ctx=ctx, provider=provider)
    if not period_invest:
        # Empty invest_periods → empty ed_invest (the dispatch-only solve case).
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})

    out = (_entityInvest_set(source)
              .join(axis_lazyframe({"d": period_invest}), how="cross")
              .select("e", "d")
              .unique()
              .sort("e", "d")
              .collect())
    # Γ.6.D: drop pairs forbidden by the no_investment lifetime gate.
    forbidden = _lifetime_expired_pairs(
        source, active_solve, workdir, methods=("no_investment",), provider=provider)
    if forbidden and out.height > 0:
        keep = [(e, d) for (e, d) in out.iter_rows()
                  if (e, d) not in forbidden]
        if not keep:
            out = pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                        "d": schema_dtype(_enums, "d")})
        else:
            # Phase 4: preserve Enum dtypes on the surviving (e, d) rows
            # so downstream partitions / joins see the axis-typed columns
            # rather than Utf8 from the Python-list rebuild.
            out = axis_lazyframe({
                "e": [e for (e, _d) in keep],
                "d": [d for (_e, d) in keep],
            }).collect()
    return out


def ed_divest_set_from_source(source: "InputSource",
                                active_solve: str | None,
                                workdir: Path | None = None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                existing_only: bool = True,
                                ) -> pl.DataFrame | None:
    """Mirror of ``ed_invest_set`` for divest::

        ed_divest = { (e, d) :
                       e ∈ entityDivest, d ∈ period_invest_of_solve,
                       pre_existing[e, d] > 0 }

    Also without a ``salvage_value``: retiring still saves the ``fixed_cost`` of the capacity.
    Only capacity that existed before the run can be retired: a period where the entity has no pre-existing
    capacity (none given, or past its end of life) has no retirement (:func:`retirable_existing_pairs`).
    ``existing_only=False`` gives the retirement options before that filter.
    """
    period_invest = _expand_invest_branch_periods(
        _solve_periods(source, active_solve, "invest_periods"),
        workdir, ctx=ctx, provider=provider)
    if not period_invest:
        return pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "d": schema_dtype(_enums, "d")})
    out = (_entityDivest_set(source)
              .join(axis_lazyframe({"d": period_invest}), how="cross")
              .select("e", "d")
              .unique()
              .sort("e", "d")
              .collect())
    return retirable_existing_pairs(out, source, active_solve, workdir, provider=provider) if existing_only else out


def retirable_existing_pairs(pairs: pl.DataFrame,
                             source: "InputSource",
                             active_solve: str | None,
                             workdir: Path | None,
                             *,
                             provider: "object | None" = None,
                             ) -> pl.DataFrame:
    """The ``[e, d]`` rows of ``pairs`` where the entity has pre-existing capacity (``p_entity_pre_existing`` > 0:
    the user's ``existing`` before the run, until its end of life), the only capacity a retirement can remove."""
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    if pairs.height == 0:
        return pairs
    periods = pairs["d"].cast(pl.Utf8).unique().sort().to_list()
    retirable = (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
                   .filter(pl.col("value") > 0.0)
                   .select("e", "d")
                   .unique()
                   .collect())
    pairs, retirable = align_join_dtypes(pairs, retirable, ["e", "d"])
    return pairs.join(retirable, on=["e", "d"], how="semi").sort("e", "d")


# ---------------------------------------------------------------------------
# §3.7.2 / §3.7.3 — edd_invest_set / edd_invest_lookback / edd_divest_active
# ---------------------------------------------------------------------------


def _p_years_d_lf(source: "InputSource",
                    active_solve: str,
                    workdir: Path | None = None,
                    *,
                    provider: "object | None" = None,
                    ) -> pl.LazyFrame | None:
    """Build per-period ``p_years_d`` for the active solve.

    Resolution order (Γ.6.D):
      1. ``solve_data/p_years_d.csv`` from workdir, if present — the
         canonical post-preprocessing CSV that already encodes the
         cumulative year offset.
      2. ``solve_data/period_with_history.csv`` (column ``param`` holds
         the cumulative year value).
      3. ``solve.years_from_start`` 1d_map(period) on the active solve
         (rare; not in current SpineDB schema but kept as a safety net).
      4. ``solve.years_represented`` — cumulative-sum the per-period
         year-count to derive the offset (matches
         ``_p_discount_years_from_source``).
      5. Integer-indexed fallback ``[0, 1, 2, ...]`` over period_in_use.

    Returns lazy frame ``[d, yr]`` or None when nothing usable exists.
    """
    # 1 — workdir CSV (authoritative).
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "p_years_d.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("value" if "value" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        alias_to_axis("period", "d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
        p2 = Path(workdir) / "solve_data" / "period_with_history.csv"
        if _provider_has_key(provider, p2):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p2)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("param" if "param" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        alias_to_axis("period", "d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
    # 2 — explicit years_from_start (rare; not present in current schema).
    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is not None and "period" in yfs.columns:
        sub = yfs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            return sub.lazy().select(
                alias_to_axis("period", "d"),
                pl.col("value").cast(pl.Float64).alias("yr"),
            )
    # 3 — cumulative-sum from years_represented.
    yrp = solve_years_represented(source)
    if yrp is not None and active_solve is not None:
        sub = yrp.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            # Map iteration order — the SpineDbReader returns rows in DB
            # storage order which mirrors the canonical Map index order.
            cum = 0.0
            rows: list[tuple[str, float]] = []
            for r in sub.iter_rows(named=True):
                d = str(r["period"])
                rows.append((d, cum))
                try:
                    cum += float(r["value"])
                except Exception:
                    pass
            if rows:
                return pl.LazyFrame({
                    "d": [r[0] for r in rows],
                    "yr": [r[1] for r in rows],
                }).with_columns(
                    pl.col("d").cast(schema_dtype(_enums, "d"),
                                       strict=False),
                ).lazy()
    # 4 — fallback to integer-indexed period_in_use.
    periods = _period_in_use_set(source, active_solve, provider=provider)
    if periods:
        return pl.LazyFrame({
            "d": periods,
            "yr": [float(i) for i in range(len(periods))],
        }).with_columns(
            pl.col("d").cast(schema_dtype(_enums, "d"), strict=False),
        ).lazy()
    return None


def _period_in_use_set(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          *,
                          provider: "object | None" = None,
                          ctx: "SolveContext | None" = None) -> list[str]:
    """Compute the canonical ``period_in_use`` set for the active solve.

    Mirrors flextool's ``preprocessing/per_solve_sets.py:95-101``:
    distinct periods in ``solve_data/steps_in_use.csv`` (a.k.a. the
    realised ``dt`` set).  When ``ctx`` is supplied, the typed
    :pyattr:`SolveContext.period_in_use` frame is consulted first — same
    semantics as the workdir CSV path but served from cache.  When
    ``workdir`` is provided AND the file is present (no ctx), the
    authoritative CSV is read directly.

    Falls back to ``solve.realized_periods ∪ solve.invest_periods`` when
    no in-memory / workdir source is available — sufficient for non-
    stochastic fixtures.
    """
    if ctx is not None:
        piu = ctx.period_in_use
        if piu.height > 0:
            return piu["d"].cast(pl.Utf8, strict=False).to_list()
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            df = _provider_read(provider, p)
            if df.height > 0 and "period" in df.columns:
                return df["period"].cast(pl.Utf8, strict=False).to_list()
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seen: dict[str, None] = {}
    for d in realized + invest:
        seen.setdefault(d, None)
    return list(seen.keys())


def edd_invest_lookback_set_from_source(source: "InputSource",
                                            active_solve: str | None,
                                            ed_invest: pl.DataFrame | None,
                                            workdir: Path | None = None,
                                            *,
                                            provider: "object | None" = None,
                                            lineage: pl.DataFrame | None = None,
                                            commission_lf: "pl.LazyFrame | None" = None,
                                            ) -> pl.DataFrame | None:
    """Build the strict-lookback (e, d_invest, d) tuples used by the
    user-constraint LHS prebuilt-capacity term (mod L2885-2898).

    Algorithm (audit §3.7.3, flex_data.py:1248-1256 + canonical
    ``invest_divest_sets.py:227-270`` lifetime filter):

      lookback = { (e, d_invest, d) :
                    (e, d_invest) ∈ ed_invest,
                    d in dt periods,
                    yr[d_invest] < yr[d],
                    yr[d] < yr[d_invest] + lifetime[e, d_invest]   (Γ.6.D)
                                            (only when entity's
                                             lifetime_method limits the
                                             window — reinvest_choice /
                                             no_investment) }

    Δ.7 consolidation: delegates to the lazy port
    :func:`._existing.edd_invest_lookback_set_lf` which
    consumes :func:`._walks.period_walk_iterator` with the
    new ``STRICT_LOOKBACK_*`` modes.  The previous eager
    ``for r in out.iter_rows`` lifetime gate is replaced with a fully
    lazy join + filter on the shared walker.

    ``lineage`` — see :func:`._walks.period_walk_iterator`;
    always ``None`` until the recourse flag lands (Slice C/D).
    """
    if ed_invest is None or ed_invest.height == 0:
        return pl.DataFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d_invest": schema_dtype(_enums, "d_invest"),
            "d": schema_dtype(_enums, "d"),
        })
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        ed_invest = cast_frame_axes(ed_invest, _live)
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return None
    periods = _period_in_use_set(source, active_solve, workdir, provider=provider)
    if not periods:
        return pl.DataFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d_invest": schema_dtype(_enums, "d_invest"),
            "d": schema_dtype(_enums, "d"),
        })
    # Δ.7 — delegate to the lazy port consuming the shared walker.
    from flextool.engine.preprocess.params._existing import edd_invest_lookback_set_lf
    ed_invest_lf = ed_invest.lazy().select("e", "d")
    return edd_invest_lookback_set_lf(
        source, active_solve, ed_invest_lf, periods, workdir,
        lineage=lineage, commission_lf=commission_lf).collect()


def edd_divest_active_from_source(source: "InputSource",
                                      active_solve: str | None,
                                      pd_divest: pl.DataFrame | None,
                                      *,
                                      entity_col: str = "p",
                                      provider: "object | None" = None,
                                      lineage: pl.DataFrame | None = None,
                                      workdir: Path | None = None,
                                      ctx: "SolveContext | None" = None,
                                      ) -> pl.DataFrame | None:
    """Build the active-divest (entity_col, d_divest, d) tuples — see audit §3.7.3.

    edd_divest_active = {(e, d_divest, d) : (e, d_divest) ∈ pd_divest, yr[d_divest] ≤ yr[d] < EOL[e]}

    EOL[e] is the end of life of the entity's pre-existing capacity (:func:`_lifetime_gate_sums`, counted from
    the start of the run; none for ``reinvest_automatic``, whose existing capacity is renewed): a retirement
    removes pre-existing capacity only while that capacity would exist.

    With *workdir* (or *ctx*) the period domain is the solve's ``period_in_use_set``, which also holds the
    forecast branch periods of a stochastic fan; the years are the solve's ``p_years_d``, where a branch period
    has the year of its period label.  A divest at a period then reduces the capacity of the branch periods it
    reaches (a here-and-now divest every branch, a recourse branch divest its own branch through the lineage
    filter) — the branch periods have the dispatch, the period label of a fan has none.

    pd_divest ⊆ ed_divest with ``entity_col ∈ {process, node}``. ``entity_col``
    selects which entity vocabulary the incoming/outgoing frame uses — the
    process set calls with ``entity_col="p"``, the node set with
    ``entity_col="n"`` — so the two never cross-contaminate each other's
    Enum casts.

    ``lineage`` — the one non-walker cross-period pair-former
    (Slice B design §7.0, inventory #18): when not ``None``, the
    shared :func:`._walks._apply_lineage_filter` is applied to
    the ``(d_divest, d)`` pairs with ``anchor_col="d_divest"``,
    ``dall_col="d"``, under the same Slice A semi-join contract.  The
    ``period_in_use`` list the filter tests membership against is the
    function-local source-derived ``periods`` list below (NOT the
    workdir-CSV PIU — this helper resolves its period domain from the
    source only).  Always ``None`` until the recourse flag lands
    (Slice C/D).
    """
    if pd_divest is None or pd_divest.height == 0:
        return pl.DataFrame(schema={
            entity_col: schema_dtype(_enums, entity_col),
            "d_divest": schema_dtype(_enums, "d_divest"),
            "d": schema_dtype(_enums, "d"),
        })
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        pd_divest = cast_frame_axes(pd_divest, _live)
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return None
    periods = _period_in_use_set(source, active_solve, workdir, provider=provider, ctx=ctx)
    if not periods:
        return pl.DataFrame(schema={
            entity_col: schema_dtype(_enums, entity_col),
            "d_divest": schema_dtype(_enums, "d_divest"),
            "d": schema_dtype(_enums, "d"),
        })
    period_lf = axis_lazyframe({"d": periods})
    pdd_lf = pd_divest.lazy().pipe(rename_to_axis, {"d": "d_divest"})
    yr_div = pyd_lf.pipe(rename_to_axis, {"d": "d_divest", "yr": "yr_divest"})
    yr_d = pyd_lf.rename({"yr": "yr"})
    chain = (pdd_lf
               .join(period_lf, how="cross")
               .join(yr_div, on="d_divest", how="inner")
               .join(yr_d, on="d", how="inner")
               .filter(pl.col("yr_divest") <= pl.col("yr")))
    # A retirement removes pre-existing capacity until that capacity's end of life (d < EOL[e]): from then on
    # the existing capacity is gone anyway, so the retirement no longer counts.
    expired = _lifetime_expired_pairs(source, active_solve, workdir, provider=provider, periods=periods)
    if expired:
        expired_lf = (pl.LazyFrame({entity_col: [e for e, _ in expired], "d": [d for _, d in expired]},
                                   schema={entity_col: pl.Utf8, "d": pl.Utf8})
                        .with_columns(pl.col(entity_col).cast(chain.collect_schema()[entity_col], strict=False),
                                      pl.col("d").cast(chain.collect_schema()["d"], strict=False)))
        chain = chain.join(expired_lf, on=[entity_col, "d"], how="anti")
    if lineage is not None:
        from flextool.engine.preprocess.params._walks import (
            _apply_lineage_filter,
            _assert_lineage_castable,
        )
        d_dtype = chain.collect_schema().get("d", pl.Utf8)
        _assert_lineage_castable(lineage, d_dtype)
        chain = _apply_lineage_filter(chain, lineage, periods, d_dtype,
                                      anchor_col="d_divest",
                                      dall_col="d")
    out = (chain
              .select(entity_col, "d_divest", "d")
              .sort(entity_col, "d_divest", "d")
              .collect())
    return out


# ---------------------------------------------------------------------------
# §3.7.4 — p_entity_max_units
# ---------------------------------------------------------------------------


def unitsize_cascade(base: pl.LazyFrame,
                     vu: "pl.LazyFrame | None",
                     ex: "pl.LazyFrame | None",
                     *, key: str) -> pl.LazyFrame:
    """The unitsize cascade, ``[key, us]`` for every row of ``base``::

        unitsize = virtual_unitsize (if explicitly set above 0)
                  OR existing       (if explicitly set non-zero)
                  OR UNITSIZE_DEFAULT (1000.0)

    A ``virtual_unitsize`` of 0 or below is ignored, as if it were not set
    (``validate_basic_sanity`` warns about it).

    ``base`` is ``[key]`` (one row per entity); ``vu`` is ``[key, vu]``
    and ``ex`` is ``[key, ex]`` with only the explicitly set values
    (``None`` when there are none).  ``existing`` is a constant (one row
    per entity) or a period Map (one row per period): it is reduced to
    the per-entity MAX, which is non-zero iff any period is — the cascade
    only checks "non-zero".  ``virtual_unitsize`` is a constant in the
    schema; it is reduced the same way so a stray row cannot duplicate
    an entity.

    The ONE implementation of the cascade: the DB-side
    :func:`_entity_unitsize_lf` / :func:`_node_unitsize_lf` and the
    per-solve ``solve_data/p_entity_unitsize.csv`` emitter
    (``emit._dispatchers._compute_p_entity_unitsize``, which reads
    ``p_process`` / ``pd_process`` and ``p_node`` / ``pd_node``) all call
    it, so the per-solve CSV seed and the DB rebuild cannot disagree.
    """
    out = base
    for name, frame in (("vu", vu), ("ex", ex)):
        if frame is None:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
            continue
        reduced = (frame.select(key, pl.col(name).cast(pl.Float64))
                   .group_by(key).agg(pl.col(name).max()))
        out = out.join(reduced, on=key, how="left")
    return out.with_columns(
        us=pl.when(pl.col("vu").fill_null(0.0) > 0.0)
              .then(pl.col("vu"))
              .when(pl.col("ex").fill_null(0.0) != 0.0)
              .then(pl.col("ex"))
              .otherwise(pl.lit(UNITSIZE_DEFAULT))
    ).select(key, "us")


def _entity_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-entity unitsize cascade across unit + node + connection,
    mirroring ``entity_period_calc_params.py:159-202``::

        unitsize = virtual_unitsize (if explicitly set above 0)
                  OR existing       (if explicitly set non-zero)
                  OR 1000.0

    Returns a lazy frame ``[e, us]`` covering every entity in
    ``unit ∪ node ∪ connection``.  Entities with neither
    ``virtual_unitsize`` nor ``existing`` get the canonical fallback
    ``1000.0``.

    For ``existing`` 1d_map(period) values, the per-period MAX is used
    as the cascade input (matches flextool's flat ``p_unit/p_node`` Map
    -agnostic read; the cascade only checks "non-zero", and the per-
    period MAX is non-zero iff any period is).

    Solve-invariant: the cascade depends only on ``source`` (entities +
    per-entity params), so the collected frame is memoized per source
    object (see ``_ENTITY_UNITSIZE_CACHE``).
    """
    try:
        cached = _ENTITY_UNITSIZE_CACHE.get(source)
    except TypeError:
        cached = None
    if cached is not None:
        return cached.lazy()
    base_parts: list[pl.LazyFrame] = []
    us_parts: list[pl.LazyFrame] = []
    ex_parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        base_parts.append(ents.lazy().select(
            alias_to_axis("name", "e")))
        us = _try_param_explicit(source, ec, "virtual_unitsize")
        if us is not None and us.height > 0:
            us_parts.append(us.lazy().select(
                alias_to_axis("name", "e"),
                pl.col("value").cast(pl.Float64).alias("vu"),
            ))
        ex = _try_param_explicit(source, ec, "existing")
        if ex is not None and ex.height > 0:
            # ``existing`` may be a scalar (one row per entity) or a
            # 1d_map(period).  The period axis carries Spine's silent-
            # default name ``"x"`` — NOT ``"period"`` — so a column-name
            # check misfires and, via the downstream ``keep="last"``
            # dedup, takes the *last* period's value.  For a retiring
            # unit whose map ends at 0 (e.g. ``220→0`` by expiry) that
            # collapses the cascade input to 0, defaults unitsize to
            # ``1000`` and turns ``existing_count = capacity/unitsize``
            # into a spurious fraction (caps continuous online below 1
            # and forbids integer online entirely).  Take the per-entity
            # MAX over whatever period rows exist — index-name-agnostic
            # per CLAUDE.md Invariant #2; a scalar's single row makes the
            # group_by a no-op.  The cascade only checks "non-zero", and
            # the per-period MAX is non-zero iff any period is.
            # (The per-entity MAX is taken in :func:`unitsize_cascade`.)
            ex_parts.append(ex.lazy().select(
                alias_to_axis("name", "e"),
                pl.col("value").cast(pl.Float64).alias("ex")))
    if not base_parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "us": pl.Float64})
    result = unitsize_cascade(
        pl.concat(base_parts).unique(),
        pl.concat(us_parts) if us_parts else None,
        pl.concat(ex_parts) if ex_parts else None,
        key="e",
    ).collect()
    try:
        _ENTITY_UNITSIZE_CACHE[source] = result
    except TypeError:
        pass
    return result.lazy()


def _entity_class_membership(source: "InputSource") -> dict[str, set[str]]:
    """Return ``{"unit": {...}, "node": {...}, "connection": {...}}``."""
    out: dict[str, set[str]] = {}
    for ec in ("unit", "node", "connection"):
        ents = _try_entities(source, ec)
        out[ec] = set() if ents is None else set(ents["name"].to_list())
    return out


def _entity_method_pairs(source: "InputSource") -> set[tuple[str, str]]:
    """Mirror flextool's ``entity__invest_method`` 2-col CSV — return
    ``{(e, method)}``.  Uses ``parameter`` (not _explicit) since
    ``invest_method`` has a non-broadcasting "not_allowed" default that
    SpineDbReader doesn't synthesise.
    """
    out: set[tuple[str, str]] = set()
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for e, m in df.select("name", "value").iter_rows():
            out.add((str(e), str(m)))
    return out


def _e_explicit_param(source: "InputSource", parameter_name: str
                       ) -> pl.LazyFrame:
    """Union explicit per-entity scalar parameter across unit + node +
    connection — returns ``[e, value]`` lazy frame (empty if no rows).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        df = _try_param_explicit(source, ec, parameter_name)
        if df is None or df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").cast(pl.Float64),
        ))
    if not parts:
        return pl.LazyFrame(schema={"e": schema_dtype(_enums, "e"),
                                     "value": pl.Float64})
    return pl.concat(parts).unique(subset=["e"], keep="last")


def _ed_period_value_map(source: "InputSource", parameter_name: str,
                         periods: list[str]) -> dict[tuple[str, str], float]:
    """:func:`_ed_period_param_lf` as a ``{(e, d): value}`` dict (``{}`` when not authored)."""
    lf = _ed_period_param_lf(source, parameter_name, periods)
    if lf is None:
        return {}
    return {(e, d): v for e, d, v in lf.collect().iter_rows()}


def p_entity_max_units_from_source(source: "InputSource",
                                       ed_invest: pl.DataFrame | None,
                                       active_solve: str | None = None,
                                       workdir: Path | None = None,
                                       p_entity_all_existing: "Param | None" = None,
                                       *,
                                       ctx: "SolveContext | None" = None,
                                       provider: "object | None" = None,
                                       commission: "pl.DataFrame | None" = None,
                                       ) -> "Param | None":
    """Per-(e, d) maximum new-unit count = ``max_capacity / unitsize``.

    ``commission`` (``[e, d, d_c]``, the decisions with a construction lead time) reads the per-period
    invest_max_period of a decision at the period ``d_c`` its capacity starts to operate in: the per-period
    limits count capacity coming online.

    Mirrors flextool's ``entity_period_calc_params.py:1718-1761``:
      ``p_entity_max_capacity[e, d]`` is the per-(entity, period)
      capacity ceiling factoring in existing capacity, per-period and
      total invest caps, and the no-limit blanket cap
      (``invest_no_limit`` / ``invest_retire_no_limit``).
      ``p_entity_max_units[e, d] = max_capacity / unitsize`` (or 0 when
      unitsize is 0).  Caller filters value > 0.

    The returned Param mirrors the CSV-loader: only rows where
    ``value > 0`` after the divide.

    The ``p_entity_all_existing`` argument lets the caller pass the
    CSV-loaded lifetime cascade frame (which factors in
    ``previously_invested`` and lifetime-cumulative chains for
    multi-period invest fixtures).  When ``None`` we fall back to
    :func:`p_entity_all_existing_from_source` — sufficient for the
    no-prior-handoff single-year case.
    """
    # Gate: both ed_invest AND ed_divest must be empty for the
    # corresponding CSV ``_load_invest`` to blank-out.  We get
    # ``ed_invest`` here; the caller checks ed_divest analogously.
    if ed_invest is None or ed_invest.height == 0:
        return None

    piu = _period_in_use_set(source, active_solve, workdir, ctx=ctx, provider=provider)
    if not piu:
        return None
    period_set = set(piu)

    # Entities universe ∪ classes.
    cls_members = _entity_class_membership(source)
    entities = sorted(cls_members["unit"] | cls_members["node"]
                      | cls_members["connection"])
    if not entities:
        return None

    # Per-(e, method) pairs.
    method_pairs = _entity_method_pairs(source)
    has_no_limit_e = {e for (e, m) in method_pairs
                      if m in INVEST_METHODS_INVEST_NO_LIMIT}
    methods_by_e: dict[str, set[str]] = {}
    for (e, m) in method_pairs:
        methods_by_e.setdefault(e, set()).add(m)

    # entityInvest = entities with any allowed-invest method.
    entityInvest = {e for e, ms in methods_by_e.items()
                    if ms - _INVEST_NOT_ALLOWED}
    # e_invest_total = subset filtered by INVEST_TOTAL methods.
    e_invest_total = {e for e in entityInvest
                       if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_TOTAL}

    # Unitsize cascade per entity.
    us_lf = _entity_unitsize_lf(source)
    us_df = us_lf.collect()
    us_map: dict[str, float] = dict(us_df.rows())

    # Existing capacity per (e, d).  Prefer the supplied lifetime cascade
    # frame (CSV-loaded ``p_entity_all_existing``), fall back to the
    # raw-existing helper.
    if p_entity_all_existing is not None:
        existing_map: dict[tuple[str, str], float] = {
            (e, d): v
            for e, d, v in p_entity_all_existing.frame.iter_rows()
        }
    else:
        pae_param = p_entity_all_existing_from_source(source, active_solve,
                                                         workdir=workdir,
                                                         ctx=ctx, provider=provider)
        if pae_param is None:
            existing_map = {}
        else:
            existing_map = {(e, d): v
                            for e, d, v in pae_param.frame.iter_rows()}

    # ─── PARAM ─── p_entity_max_units ───────────────────────────────────────────
    # p_entity_max_units[e,d] = max_capacity[e,d] / unitsize[e]       ∀ d ∈ period_in_use, kept where > 0
    # max_capacity[e,d] = cmc[e,d]                                      if (e,d) ∈ ed_invest_cumulative
    #                   = existing[e,d] + imp[e,d]                      if (e,d) ∈ ed_invest_period, e ∉ e_invest_total
    #                                   + eim[e]                        if e ∈ e_invest_total, (e,d) ∉ ed_invest_period
    #                                   + max(imp[e,d], eim[e])         if both
    #                                   + max_flow_for_unconstrained    if invest_method ∈ *_no_limit
    # imp[e,d] / cmc[e,d] = authored[e,d]  (period map)
    #                     | authored[e]    (constant)
    #                     | authored[e, anchor(d)]  (recourse branch period without its own value)
    #                     | 0
    # imp / cmc are resolved for every d ∈ period_in_use (not only invest periods).
    # With a construction lead time imp[e,d] is imp[e,c(d)], at the period c(d) the decision's capacity starts to
    # operate in.
    #
    # invest_max_period[e, d]: constant or period map via the shape resolver, default 0.
    imp_map = _ed_period_value_map(source, "invest_max_period", piu)
    if commission is not None and commission.height > 0:
        imp_map = dict(imp_map)
        for _e, _o, _dc in commission.select(
                pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("d_c").cast(pl.Utf8)).iter_rows():
            imp_map[(_e, _o)] = imp_map.get((_e, _dc), 0.0)
    # ed_invest_period: rows in ed_invest_pairs whose method ∈ INVEST_PERIOD.
    ed_invest_period = {(e, d) for (e, d) in ed_invest.iter_rows()
                         if methods_by_e.get(e, set()) & _INVEST_METHODS_INVEST_PERIOD}

    # invest_max_total[e]: explicit values, default 0.
    eim_df = _e_explicit_param(source, "invest_max_total").collect()
    eim_map: dict[str, float] = dict(eim_df.iter_rows())

    # cumulative_max_capacity[e, d]: constant or period map via the shape resolver, default 0.
    cmc_map = _ed_period_value_map(source, "cumulative_max_capacity", piu)
    # Entities that author cumulative_max_capacity at all (any period) — an entity
    # absent here never set the parameter, as opposed to an authored-but-zero value.
    cmc_authors = {e for (e, _d) in cmc_map}
    # ed_invest_cumulative: ed_invest pairs with cumulative_limits method.
    ed_invest_cumulative = {(e, d) for (e, d) in ed_invest.iter_rows()
                             if "cumulative_limits" in methods_by_e.get(e, set())}

    # p_max_flow_for_unconstrained_variables: the largest value over the
    # model entities, used as given (also below 1e6); 1e6 when none is set.
    # DB stores the model param as ``max_flow_for_unconstrained_variables``
    # (no ``p_`` prefix — that prefix is only the CSV-emit alias).
    p_unc = 1000000.0  # default per ``entity_period_calc_params.py:1689``.
    df_unc = _try_param(source, "model", "max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = float(df_unc["value"].max())
        except Exception:
            pass

    # Compute max_capacity per (e, d in period_in_use) and divide by unitsize.
    rows: list[tuple[str, str, float]] = []
    for e in entities:
        us = us_map.get(e, UNITSIZE_DEFAULT)
        in_total = e in e_invest_total
        has_no_limit = e in has_no_limit_e
        for d in piu:
            if d not in period_set:
                continue  # defensive; piu is the period_in_use list
            if (e, d) in ed_invest_cumulative:
                if e in cmc_authors:
                    # A period map that leaves out an investment period of the solve never gets here:
                    # pass 1b refuses it (``require_period_coverage``).  The 0.0 is only reached for
                    # a period the solve does not invest in.
                    v = cmc_map.get((e, d), 0.0)
                else:
                    # A cumulative_limits entity that never authors
                    # cumulative_max_capacity (e.g. only cumulative_min_capacity, a floor with no
                    # ceiling) is unconstrained above, not pinned to v_invest = 0 — the floor would
                    # otherwise be unmeetable.  The real ceiling (if any) is still enforced by
                    # {max,min}Cumulative_capacity_{p,n} (_cumulative_invest.py); this value only
                    # bounds the variable itself.
                    v = existing_map.get((e, d), 0.0) + p_unc
            else:
                v = existing_map.get((e, d), 0.0)
                in_period = (e, d) in ed_invest_period
                imp = imp_map.get((e, d), 0.0)
                eim = eim_map.get(e, 0.0)
                if in_period and not in_total:
                    v += imp
                if in_total and not in_period:
                    v += eim
                if in_period and in_total:
                    v += max(imp, eim)
                if has_no_limit:
                    v += p_unc
            mu = (v / us) if us != 0.0 else 0.0
            if mu > 0.0:
                rows.append((e, d, float(mu)))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema=["e", "d", "value"], orient="row").sort("e", "d")
    return Param(("e", "d"), out)


# ---------------------------------------------------------------------------
# §3.7.5/6 — Multi-year inflation cascade  (default-rate path only)
# ---------------------------------------------------------------------------
#
# Replaces the simple Γ.3.A ``p_inflation_op`` when ``model.inflation_rate
# != 0`` AND the active solve carries non-trivial ``years_represented`` /
# ``period_timeset``.  The full cascade is documented at
# ``period_calculated_params.py:280-322``: for each period d, sum across
# (d, y) ∈ years_for_period, weighted by ``years_represented[d, y]`` of
# ``(1+r)^{-(base[d, y] + pyr * offset_op)}`` where base counts cumulative
# years across the GLOBAL year universe.
#
# Γ.3.C scope: produce the cascade for the simplest non-trivial case —
# every period has exactly one represented year and ``years_from_start``
# is supplied as an absolute year offset.  This collapses to
# ``inflation_op[d] = (1+r)^{-(yr[d] + offset_op)}``.  More general shapes
# (multi-year-per-period, missing offset) gate-fail and the CSV value
# survives.


def p_inflation_op_multi_year_from_source(source: "InputSource",
                                             active_solve: str,
                                             dt: pl.DataFrame,
                                             ) -> "Param | None":
    """Multi-year inflation factor for the default 1-year-per-period case.

    Returns None when:
      * ``model.inflation_rate == 0`` (Γ.3.A's simple path covers it).
      * The active solve has no ``years_from_start`` parameter.
      * Any period has > 1 represented year (defer to CSV).
    """
    rate_df = _try_param(source, "model", "inflation_rate")
    rate_v = 0.0
    if rate_df is not None and rate_df.height > 0:
        rate_v = float(rate_df["value"][0])
    if rate_v == 0.0:
        return None  # Γ.3.A's trivial path applies.
    offset_df = _try_param(source, "model", "inflation_offset_operations")
    offset_v = 0.5
    if offset_df is not None and offset_df.height > 0:
        offset_v = float(offset_df["value"][0])

    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is None or "period" not in yfs.columns:
        return None
    yfs_sub = yfs.filter(pl.col("name") == active_solve)
    if yfs_sub.height == 0:
        return None
    # Multi-year-per-period guard: if ``years_represented`` Map has
    # multiple (period, year) rows for any period, defer.
    yr = solve_years_represented(source)
    if yr is not None:
        yr_sub = yr.filter(pl.col("name") == active_solve)
        if yr_sub.height > 0:
            counts = (yr_sub.group_by("period").agg(pl.len().alias("n")))
            if counts["n"].max() and counts["n"].max() > 1:
                return None

    factor = 1.0 + rate_v
    if dt is None or dt.height == 0:
        return None
    yfs_lf = yfs_sub.lazy().select(
        alias_to_axis("period", "d"),
        pl.col("value").cast(pl.Float64).alias("yr"),
    )
    # Defensive re-cast: align d to canonical Enum so the left-join
    # against yfs_lf (Enum d) composes cleanly.
    out = (dt.lazy().select(alias_to_axis("d", "d")).unique()
              .join(yfs_lf, on="d", how="left")
              .with_columns(yr=pl.col("yr").fill_null(0.0))
              .with_columns(
                  value=pl.lit(factor).pow(-(pl.col("yr") + offset_v))
              )
              .select("d", "value")
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


# ---------------------------------------------------------------------------
# §3.8.1 — p_section
# ---------------------------------------------------------------------------


def p_section_from_source(source: "InputSource",
                            dt: pl.DataFrame,
                            classified: pl.DataFrame | None = None,
                            ) -> "Param | None":
    """Section term (y-intercept) of the ``min_load_efficiency`` linearisation, one row per
    min_load_efficiency unit × (d, t):

        section = min_load · (1/eta_min − 1/eta) / (1 − min_load)   (_mle_section)
        eta = efficiency (default 1.0);  min_load = 0 where not authored (section 0);
        eta_min = efficiency_at_min_load, else efficiency (_mle_eta_min: a flat curve, section exactly 0)

    i.e. 1/eta − slope.  Unrounded, and the term ``p_slope_from_source`` subtracts from 1/eta, so the full-load
    input slope⁰ + section⁰ is 1/eta (the old flextool rounding of 1/eta and of the slope to 6 decimals left
    slope⁰ + section⁰ ≠ 1/eta, #140).  Other ct_methods → no section row; None
    when there is no min_load_efficiency unit.

    On a direct (one-input, one-output) unit, the section above is further divided by ``k_src``, the
    unit's input ``conversion_flow_coeff`` (default 1; #10 / Q12, Q-A — the section is per unit of
    native output capacity, not weighted by the sink coefficient).  With ``k_src`` ≠ 1 the folded
    slope + section no longer sums to 1/eta — the identity above is for the unfolded slope⁰/section⁰
    only.  Folded by :func:`_arithmetic._direct_flow_coef_ratio`, shared with
    ``p_slope_from_source`` and the rolling emit seed; a zero ``k_src``/``k_sink`` keeps its
    uncapped-edge meaning instead (ratio left at 1).
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
        classified = cast_frame_axes(classified, _enums)
    mle = classified.filter(pl.col("ct") == "min_load_efficiency").select("p").unique()
    if mle.height == 0:
        return None
    base = mle.lazy().join(dt.lazy().with_columns(alias_to_axis(pl.col("d"), "d"),
                                                  alias_to_axis(pl.col("t"), "t")).select("d", "t"),
                           how="cross")
    eta_lf = (_eff_param_to_pdt(source, "unit", "efficiency", dt, "eta")
              if parameter_has_values(source, "unit", "efficiency") else None)
    base = (base.join(eta_lf, on=["p", "d", "t"], how="left") if eta_lf is not None
            else base.with_columns(eta=pl.lit(None).cast(pl.Float64)))
    base = _mle_curve_inputs(source, dt, base.with_columns(eta=pl.col("eta").fill_null(1.0)))
    base = base.with_columns(section=_mle_section(pl.col("eta"), pl.col("eta_min"), pl.col("min_load")))
    # #10 / Q12: fold a direct unit's output/input conversion_flow_coeff ratio into its section —
    # section = section⁰/k_src (Q-A default; see p_slope_from_source).  No warning here — the DB-direct
    # zero-coefficient warning is emitted once, by p_slope_from_source.
    from flextool.engine.preprocess.params._arithmetic import _direct_flow_coef_ratio
    ratio = _direct_flow_coef_ratio(source, warn=False)
    if ratio.height > 0:
        base = (base.join(ratio.lazy().select("p", "k_src"), on="p", how="left")
                     .with_columns(
                         section=pl.when(pl.col("k_src").is_not_null())
                                   .then(pl.col("section") / pl.col("k_src"))
                                   .otherwise(pl.col("section"))))
    out = (base
                .select("p", "d", "t", pl.col("section").alias("value"))
                .sort("p", "d", "t")
                .collect())
    _raise_on_null_mle_cells(out.filter(pl.col("value").is_null()), "p_section")
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.3.5 — p_flow_upper  (Δ.26)
# ---------------------------------------------------------------------------
#
# Native port of flextool's
# ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
# `process_arc_unions.py:1469-1624` for `write_p_flow_max`).  Mirrors
# flextool.mod L1661-1677:
#
#     p_flow_max{(p, source, sink, d, t) in peedt} :=
#       if (p, source, sink) in process_source_sink_coeff_zero
#       then p_unconstrained_flow_cap
#       else (
#         if exists{(p, m) in process__method_indirect} 1
#            AND (p, source) in process_source
#         then ( if (p, 'min_load_efficiency') in process__ct_method
#                then pdtProcess_slope[p,d,t] + pdtProcess_section[p,d,t]
#                else pdtProcess_slope[p,d,t]
#              )
#              * (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#              / p_process_source_capacity_max_coeff[p, source]
#         else (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#       )
#       * (if (p, sink) in process_sink
#          then p_process_sink_capacity_max_coeff[p, sink] else 1)
#
# The indirect input-arc branch no longer follows the .mod (whose fuel
# arc was slope × capacity / source coefficient — too tight for
# multi-output units and inverted in the coefficient): it is the input
# limit of :func:`_indirect_input_cap_lf` (``input_share_max`` × the
# full-load fuel per unit of capacity) times the capacity count.
#
# Slow-path consumer: ``flex_data.py::_load_process_topology`` reads
# ``solve_data/p_flow_max.csv`` (preprocessed CSV).  Δ.26 ports the
# derivation natively as a polars-lazy helper, wired into
# :func:`apply_derived_c` so the fast single-solve path produces the
# correct ``p_flow_upper`` Param without any preprocessing.


def _arc_max_capacity_coef_lf(source: "InputSource",
                                  side: str,
                                  *,
                                  explicit: bool = False,
                                  ) -> pl.LazyFrame:
    """Per-(p, node) capacity coefficient of a unit edge.

    ``side='sink'`` → ``unit__outputNode.capacity_max_coeff`` as
    ``[p, sink, coef]``; ``side='source'`` → ``unit__inputNode.
    input_share_max`` (schema v70; was ``capacity_max_coeff``) as
    ``[p, source, coef]``.  ``explicit=True`` returns only authored rows
    (no schema-default broadcast).

    Connections don't carry these parameters (the per-arc coefficient is
    unit-only; connections always default to 1 — flextool.mod L686-687),
    so only unit-arc rows are emitted; callers fill the default 1.0 via
    left-join.
    """
    if side == "source":
        ec = "unit__inputNode"
        pname = "input_share_max"
        node_alias = "source"
    else:
        ec = "unit__outputNode"
        pname = "capacity_max_coeff"
        node_alias = "sink"
    df = (_try_param_explicit(source, ec, pname) if explicit
          else _try_param(source, ec, pname))
    if df is None or df.height == 0:
        return pl.LazyFrame(
            schema={"p": schema_dtype(_enums, "p"),
                    node_alias: schema_dtype(_enums, node_alias),
                    "coef": pl.Float64})
    cols = df.columns
    rename: dict[str, str] = {}
    for c in cols:
        if c == "unit":
            rename[c] = "p"
        elif c == "node":
            rename[c] = node_alias
    return (df.lazy().pipe(rename_to_axis, rename)
              .with_columns(pl.col("value").cast(pl.Float64).alias("coef"))
              .select("p", node_alias, "coef")
              .unique(subset=["p", node_alias], keep="last"))


def _arc_conversion_coef_lf(source: "InputSource",
                               side: str,
                               ) -> pl.LazyFrame:
    """Per-(p, node) ``conversion_flow_coeff`` from ``unit__inputNode``
    (``side='source'`` → ``[p, source, coef]``) or ``unit__outputNode``
    (``side='sink'`` → ``[p, sink, coef]``).  Only authored rows are
    returned; callers fill the default 1.0 via left-join.
    """
    ec = "unit__inputNode" if side == "source" else "unit__outputNode"
    df = _try_param(source, ec, "conversion_flow_coeff")
    if df is None or df.height == 0:
        return pl.LazyFrame(
            schema={"p": schema_dtype(_enums, "p"),
                    side: schema_dtype(_enums, side),
                    "coef": pl.Float64})
    rename = {c: ("p" if c == "unit" else side)
              for c in df.columns if c in ("unit", "node")}
    return (df.lazy().pipe(rename_to_axis, rename)
              .with_columns(pl.col("value").cast(pl.Float64).alias("coef"))
              .select("p", side, "coef")
              .unique(subset=["p", side], keep="last"))


def _delay_fuel_factor_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-process ``[p, delay_factor]`` = ``1 / max_td(delay_weight)``
    for delayed processes (a scalar ``delay`` has weight 1 → factor 1).

    One input step of a delayed process feeds several output steps with
    weights ``w_td``; with every output at its cap, a single input step
    can carry up to ``slope · Σ out / max(w_td)``.  Processes without a
    delay are absent (callers default to 1).
    """
    empty = pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                 "delay_factor": pl.Float64})
    dist = _delay_distributions_from_source(source)
    if dist is None:
        return empty
    weighted, single, processes = dist
    max_w: dict[str, float] = {p: 1.0 for p, _ in single}
    for (p, _), w in weighted.items():
        max_w[p] = max(max_w.get(p, 0.0), w)
    factors = {p: 1.0 / w for p, w in max_w.items()
               if p in processes and w > 0.0}
    if not factors:
        return empty
    return (pl.LazyFrame({"p": list(factors),
                          "delay_factor": list(factors.values())})
              .with_columns(alias_to_axis(pl.col("p"), "p")))


def _process_source_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_source`` set ``[p, source]`` — union of unit input arcs
    and connection's first node.  Mirrors flextool.mod L686 + the
    expansion in ``input_writer.write_topology``."""
    parts: list[pl.LazyFrame] = []
    uin = _try_entities(source, "unit__inputNode")
    if uin is not None and uin.height > 0:
        parts.append(uin.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("node", "source"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("node_1", "source"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                     "source": schema_dtype(_enums, "source")})
    return pl.concat(parts).unique()


def _process_sink_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_sink`` set ``[p, sink]`` — union of unit output arcs
    and connection's second node.  Mirrors flextool.mod L687."""
    parts: list[pl.LazyFrame] = []
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None and uout.height > 0:
        parts.append(uout.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("node", "sink"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("node_2", "sink"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                     "sink": schema_dtype(_enums, "sink")})
    return pl.concat(parts).unique()


def _process_source_sink_coeff_zero_lf(source: "InputSource",
                                            pss: pl.DataFrame,
                                            ) -> pl.LazyFrame:
    """``process_source_sink_coeff_zero`` set: rows of pss whose source-
    or sink-side ``conversion_flow_coeff`` is explicitly zero.

    A zero conversion coefficient takes the edge out of the conversion
    equation and out of every per-edge capacity / ramp / min-load
    constraint (the hydro-pass-through pattern); the edge's flow is
    bounded only by ``max_flow_for_unconstrained_variables``.  The set
    used to be keyed on ``capacity_max_coeff == 0`` — an artefact of the
    v35 rename of the old ``coefficient`` (which was the flow
    coefficient).  ``capacity_max_coeff = 0`` now means what its name
    says: a zero cap on the edge.
    """
    if pss is None or pss.height == 0:
        return pl.LazyFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "source": schema_dtype(_enums, "source"),
            "sink": schema_dtype(_enums, "sink")})
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
    src_zero = (_arc_conversion_coef_lf(source, "source")
                  .filter(pl.col("coef") == 0.0)
                  .select("p", "source"))
    sink_zero = (_arc_conversion_coef_lf(source, "sink")
                   .filter(pl.col("coef") == 0.0)
                   .select("p", "sink"))
    src_match = (pss.lazy()
                   .join(src_zero, on=["p", "source"], how="inner")
                   .select("p", "source", "sink"))
    sink_match = (pss.lazy()
                    .join(sink_zero, on=["p", "sink"], how="inner")
                    .select("p", "source", "sink"))
    return pl.concat([src_match, sink_match]).unique()


def process_source_sink_uncapped_from_source(source: "InputSource",
                                              pss: pl.DataFrame | None,
                                              ) -> pl.DataFrame | None:
    """``(p, source, sink)`` arcs with ``conversion_flow_coeff = 0`` on
    either side — the uncapped (hydro-pass-through) edges.

    The model removes these arcs from every per-edge capacity / ramp /
    min-load constraint; their ``p_flow_upper`` is the unconstrained
    ``max_flow_for_unconstrained_variables`` value.  ``None`` when there
    are none.
    """
    if pss is None or pss.height == 0:
        return None
    out = (_process_source_sink_coeff_zero_lf(source, pss)
           .sort("p", "source", "sink")
           .collect())
    return out if out.height > 0 else None


def p_arc_max_cap_coef_from_source(source: "InputSource",
                                      pss: pl.DataFrame,
                                      classified: pl.DataFrame | None = None,
                                      ) -> "Param | None":
    """Per-(p, source, sink) ``capacity_max_coeff`` factor for **direct**
    arcs and indirect **output** arcs — the multiplier the ``maxFlow`` RHS drops when it uses
    ``p_flow_upper_existing`` (which is ``existing_count`` = capacity / unitsize only).

    Mirrors flextool.mod ``maxToSink`` / ``maxFromSource``:

    * ``maxToSink`` (``process__source__sinkIsNode``, the normal output
      arc) multiplies the existing-capacity RHS by
      ``p_process_sink_max_capacity_coefficient[p, sink]`` whenever
      ``(p, sink) ∈ process_sink``.  We carry that as ``sink_coef`` =
      ``unit__outputNode.capacity_max_coeff``.
    * ``maxFromSource`` (``process__sourceIsNode__sink_1way_noSink…``, a
      1-way unit whose node is the *source* and that has no output node)
      multiplies by ``p_process_source_max_capacity_coefficient[p,
      source]``.  The engine emits such an arc as a synthetic
      ``(p, source, p)`` row whose ``sink`` is the process itself, so
      ``(p, sink) ∉ process_sink``; we carry that as ``src_coef`` =
      ``unit__inputNode.input_share_max`` (schema v70; was
      ``capacity_max_coeff`` — same numbers).
    * A direct unit with an input AND an output node has a single flow
      variable (output units; its input is ``slope × v_flow``).  An
      AUTHORED ``input_share_max`` on its input caps that input at
      ``share × full-load input``, i.e. ``v_flow ≤ share × capacity``;
      combined with the output's ``capacity_max_coeff`` the arc factor is
      ``min(capacity_max_coeff_out, input_share_max_in)``.  The schema
      default (1, "no effective limit") is not applied, so an output
      coefficient above 1 is left untouched when the share is unset.

    Indirect (CHP / multi-flow) processes contribute only their OUTPUT
    arcs (``process_output_flows``): ``model/core.py`` binds those with the
    existing-only ``maxToSink`` RHS (``p_flow_upper_existing``), so their
    ``capacity_max_coeff`` must be folded here or it is silently lost.
    Indirect input arcs stay excluded — their limit is
    :func:`p_indirect_input_cap_from_source` (``input_share_max`` × the
    full-load fuel per unit of capacity).  ``conversion_flow_coeff = 0`` arcs (direct or
    indirect, :func:`process_source_sink_uncapped_from_source`) are
    uncapped and excluded too.

    Returns a Param keyed ``(p, source, sink)`` carrying only the direct
    arcs whose effective coefficient differs from the schema default
    (1.0); absent arcs default to 1.0 at the consumer's densify.  Returns
    ``None`` when there is nothing to carry (no authored coefficient, or
    every authored row resolves to 1.0) so the consumer skips the fold.
    """
    if pss is None or pss.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)

    # Direct arcs + indirect OUTPUT arcs (see docstring).
    indirect_p = process_indirect_set(source, classified)
    direct_pss = (pss.lazy()
                    .select("p", "source", "sink")
                    .join(indirect_p.lazy().with_columns(_ind=pl.lit(True)),
                          on="p", how="left")
                    .filter(pl.col("_ind").is_null())
                    .select("p", "source", "sink"))
    out_flows = process_output_flows(source, pss, classified)
    if out_flows.height > 0:
        direct_pss = pl.concat([
            direct_pss,
            out_flows.lazy().select("p", "source", "sink"),
        ])
    # conversion_flow_coeff = 0 edges are uncapped (coeff_zero): model/core.py
    # binds them with the unconstrained ``p_flow_upper`` value, so no
    # capacity coefficient applies to them.
    direct_pss = direct_pss.join(
        _process_source_sink_coeff_zero_lf(source, pss),
        on=["p", "source", "sink"], how="anti")

    sink_pairs = _process_sink_pairs_lf(source)
    src_coef_lf = _arc_max_capacity_coef_lf(source, "source")
    src_share_explicit_lf = _arc_max_capacity_coef_lf(
        source, "source", explicit=True)
    sink_coef_lf = _arc_max_capacity_coef_lf(source, "sink")

    out = (direct_pss
        # Tag (p, sink) ∈ process_sink — selects maxToSink vs maxFromSource.
        .join(sink_pairs.with_columns(_has_sink=pl.lit(True)),
              on=["p", "sink"], how="left")
        .with_columns(_has_sink=pl.col("_has_sink").fill_null(False))
        # Source-side input_share_max (default 1.0).
        .join(src_coef_lf.rename({"coef": "src_coef"}),
              on=["p", "source"], how="left")
        .with_columns(src_coef=pl.col("src_coef").fill_null(1.0))
        # Authored input_share_max only (direct units with an output).
        .join(src_share_explicit_lf.rename({"coef": "src_share"}),
              on=["p", "source"], how="left")
        # Sink-side capacity_max_coeff (default 1.0).
        .join(sink_coef_lf.rename({"coef": "sink_coef"}),
              on=["p", "sink"], how="left")
        .with_columns(sink_coef=pl.col("sink_coef").fill_null(1.0))
        .with_columns(
            value=pl.when(pl.col("_has_sink")
                          & pl.col("src_share").is_not_null())
                     # maxToSink of a direct unit with an authored
                     # input share: both caps apply to the one variable.
                     .then(pl.min_horizontal(pl.col("sink_coef"),
                                             pl.col("src_share")))
                     .when(pl.col("_has_sink"))
                     .then(pl.col("sink_coef"))     # maxToSink
                     .otherwise(pl.col("src_coef")))  # maxFromSource
        # Only carry rows that actually deviate from the 1.0 default; the
        # consumer fills absent (p, source, sink) with 1.0.
        .filter(pl.col("value") != 1.0)
        .select("p", "source", "sink", "value")
        .sort("p", "source", "sink")
        .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink"), out)


def wire_arc_capacity_params(flex_data, source: "InputSource") -> None:
    """Set the per-arc capacity Params ``model/core.py``'s ``maxFlow`` needs:
    ``p_arc_max_cap_coef``, ``process_source_sink_uncapped`` and
    ``p_indirect_input_cap``.

    For the synthetic sub-solve path (rolling rolls ``<solve>_roll_<n>``,
    nested per-period sub-solves) that skips ``apply_derived_b..g``: all
    three producers are solve-agnostic (they read unit / edge parameters,
    the process topology, ``dt`` and the CSV-seeded ``p_slope`` /
    ``p_section``), so the result equals the full-cascade path.  Without
    this the roll sub-solves dropped the output ``capacity_max_coeff``,
    treated ``conversion_flow_coeff = 0`` edges as capped, and left the
    indirect input arcs without their limit.
    """
    pss = getattr(flex_data, "process_source_sink", None)
    if pss is None or pss.height == 0:
        return
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
    classified = _classify_process_method(source)
    flex_data.p_arc_max_cap_coef = p_arc_max_cap_coef_from_source(
        source, pss, classified)
    flex_data.process_source_sink_uncapped = (
        process_source_sink_uncapped_from_source(source, pss))
    flex_data.p_indirect_input_cap = p_indirect_input_cap_from_source(
        source, pss, getattr(flex_data, "dt", None),
        p_slope=getattr(flex_data, "p_slope", None),
        p_section=getattr(flex_data, "p_section", None),
        classified=classified)


def p_flow_upper_from_source(source: "InputSource",
                                pss: pl.DataFrame,
                                dt: pl.DataFrame,
                                p_slope: "Param | None",
                                p_section: "Param | None",
                                p_unitsize: "Param | None",
                                p_entity_max_units: "Param | None",
                                p_entity_all_existing: "Param | None",
                                classified: pl.DataFrame | None = None,
                                ) -> "Param | None":
    """§3.3.5 — per-(p, source, sink, d, t) structural max flow.

    Native port of flextool's
    ``preprocessing/process_arc_unions.py:write_p_flow_max``
    (file:line `process_arc_unions.py:1469-1624`).  Returns a Param
    keyed on ``(p, source, sink, d, t)`` whose values match
    ``solve_data/p_flow_max.csv`` written by the slow path.

    Inputs:

    ``pss`` — ``process_source_sink`` set; the (p, source, sink) index
    of the output Param.  Cross-joined with ``dt`` to form ``peedt``.

    ``p_slope`` / ``p_section`` — per-(p, d, t) slope / section terms
    from :func:`p_slope_from_source` / :func:`p_section_from_source`.
    Used in the indirect-method branch.

    ``p_unitsize`` — per-(p,) unitsize cascade from
    :func:`_arithmetic.p_unitsize_from_source`.  Caller may
    also pass the per-entity cascade keyed on ``(e,)``; we accept either
    column name.

    ``p_entity_max_units`` — per-(e, d) ``max_capacity / unitsize`` from
    :func:`p_entity_max_units_from_source`.  Provides
    ``p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p]``
    directly, so we don't need to recompute the dispatch cap inline.

    ``p_entity_all_existing`` — fallback when ``p_entity_max_units`` is
    absent (no invest configured); when both invest sets are empty
    flextool's preprocessing still emits ``p_flow_max`` rows valued at
    the existing unit count (capacity/unitsize) per arc.  This matches the slow path's
    behaviour in the no-invest case.

    Returns ``None`` when no usable inputs exist (empty pss or dt) —
    caller leaves the Param at its empty placeholder.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None

    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
        dt = cast_frame_axes(dt, _live)
        if classified is not None:
            classified = cast_frame_axes(classified, _live)

    # ── 1. Resolve the (p, d) cap-per-unitsize term ─────────────────
    # Prefer p_entity_max_units (= dispatch_capacity_max / unitsize per
    # (e, d)).  When absent (no invest in fixture), derive it from
    # p_entity_all_existing / p_unitsize.
    if p_entity_max_units is not None and p_entity_max_units.frame.height > 0:
        cap_per_unit_lf = (p_entity_max_units.frame.lazy()
            .pipe(rename_to_axis, {"e": "p"})
            .select("p", alias_to_axis("d", "d"), pl.col("value").alias("cap_per_unit")))
    else:
        # Derive from existing capacity / unitsize.  We only need rows that join
        # against pss, so we filter inline.
        if p_entity_all_existing is None or p_entity_all_existing.frame.height == 0:
            return None
        # p_unitsize may come keyed on (p,) (post-Δ.4b override) or (e,)
        # (raw cascade); accept both.
        if p_unitsize is None or p_unitsize.frame.height == 0:
            us_df = _entity_unitsize_lf(source).pipe(rename_to_axis, {"e": "p", "us": "us"}).collect()
        else:
            us_cols = p_unitsize.frame.columns
            us_key = "p" if "p" in us_cols else "e"
            us_df = p_unitsize.frame.pipe(rename_to_axis, {us_key: "p", "value": "us"})
        cap_per_unit_lf = (
            p_entity_all_existing.frame.lazy()
                .pipe(rename_to_axis, {"e": "p", "value": "cap"})
                .join(us_df.lazy(), on="p", how="left")
                .with_columns(
                    cap_per_unit=pl.when((pl.col("us").is_not_null())
                                          & (pl.col("us") != 0.0))
                                    .then(pl.col("cap") / pl.col("us"))
                                    .otherwise(0.0))
                .select("p", alias_to_axis("d", "d"), "cap_per_unit")
        )

    # ── 2. Indirect input (fuel) arc limit per unit of capacity ─────
    # (p, source, sink, d, t) for the input arcs of indirect units —
    # ``input_share_max × full-load fuel / src_conv``; see
    # :func:`_indirect_input_cap_lf`.  p_flow_upper on those arcs is that
    # limit times the structural capacity count.
    if classified is None:
        classified = _classify_process_method(source)
    in_cap_lf = _indirect_input_cap_lf(source, pss, dt, p_slope, p_section,
                                       classified)
    process_sink_lf = _process_sink_pairs_lf(source)

    # ── 3. Coefficient frames ───────────────────────────────────────
    sink_coef_lf = _arc_max_capacity_coef_lf(source, "sink")

    # ── 4. p_unconstrained_flow_cap ────────────────────────────────
    # DB stores the model param as ``max_flow_for_unconstrained_variables``
    # (no ``p_`` prefix — that prefix is only the CSV-emit alias).
    # The largest value over the model entities, used as given (also below
    # 1e6); 1e6 when none is set.
    p_unc = 1_000_000.0
    df_unc = _try_param(source, "model", "max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = float(df_unc["value"].max())
        except Exception:
            pass

    # ── 5. coeff_zero set — rows that get the unconstrained value ─
    coeff_zero_lf = _process_source_sink_coeff_zero_lf(source, pss)

    # ── 6. Build peedt = pss × dt with cap-per-unit and arc tags ────
    pss_lf = pss.lazy().select("p", "source", "sink")
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # against pss (Enum p / source / sink) — keeps the (p, source, sink,
    # d, t) base frame Enum-typed across all axes.
    base = (pss_lf
        .join(dt.lazy()
                 .with_columns(alias_to_axis(pl.col("d"), "d"),
                               alias_to_axis(pl.col("t"), "t")),
              how="cross")  # adds d, t
        .join(cap_per_unit_lf, on=["p", "d"], how="left")
        .with_columns(
            cap_per_unit=pl.col("cap_per_unit").fill_null(0.0))
        # Indirect input arcs: fuel limit per unit of capacity.
        .join(in_cap_lf.rename({"value": "_in_cap"}),
              on=["p", "source", "sink", "d", "t"], how="left")
        # Sink-side capacity_max_coeff (default 1.0); also tag
        # (p, sink) ∈ process_sink (the .mod multiplies by sink_coef
        # only when (p, sink) ∈ process_sink, defaulting to 1 outside).
        .join(process_sink_lf.with_columns(_has_sink=pl.lit(True)),
                on=["p", "sink"], how="left")
        .with_columns(
            _has_sink=pl.col("_has_sink").fill_null(False))
        .join(sink_coef_lf.rename({"coef": "sink_coef"}),
                on=["p", "sink"], how="left")
        .with_columns(
            sink_coef=pl.col("sink_coef").fill_null(1.0))
        # Mark coeff_zero rows.
        .join(coeff_zero_lf.with_columns(_coeff_zero=pl.lit(True)),
                on=["p", "source", "sink"], how="left")
        .with_columns(
            _coeff_zero=pl.col("_coeff_zero").fill_null(False))
    )

    # ── 7. Compute the formula. ─────────────────────────────────────
    # indirect input arc: cap_per_unit · input-arc limit per capacity;
    # everything else:    cap_per_unit.
    # Multiply by sink_coef when (p, sink) ∈ process_sink, else *1.
    base = base.with_columns(
        _branch_value=pl.when(pl.col("_in_cap").is_not_null())
                          .then(pl.col("cap_per_unit") * pl.col("_in_cap"))
                          .otherwise(pl.col("cap_per_unit")),
    ).with_columns(
        # sink-coef multiplier — only when (p, sink) ∈ process_sink.
        _sink_factor=pl.when(pl.col("_has_sink"))
                          .then(pl.col("sink_coef"))
                          .otherwise(1.0),
    ).with_columns(
        value=pl.when(pl.col("_coeff_zero"))
                  .then(pl.lit(p_unc))
                  .otherwise(pl.col("_branch_value") * pl.col("_sink_factor")),
    )

    out = (base
        .select("p", "source", "sink", "d", "t", "value")
        .sort("p", "source", "sink", "d", "t")
        .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), out)


def _indirect_fuel_width_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-process ``[p, fuel_width]`` = ``W``: the most output-side fuel
    energy per unit of capacity that the outputs can demand.

    ``conversion_indirect`` reads ``Σ_s src_conv_s·in_s = slope ·
    Σ_k conv_k·out_k (+ section · online)``.  Capacity caps the SUM of
    the outputs (``Σ_k out_k ≤ cap``, ``maxOutputSum``) and each output
    individually (``out_k ≤ capacity_max_coeff_k · cap``), so per unit of
    capacity ``W = max Σ_k conv_k·x_k`` s.t. ``Σ_k x_k ≤ 1``,
    ``0 ≤ x_k ≤ capacity_max_coeff_k`` — solved greedily: fill the
    outputs in descending ``conv_k`` order, each up to
    ``min(capacity_max_coeff_k, remaining)``.

    Outputs with ``conversion_flow_coeff ≤ 0`` draw no fuel (the
    ``= 0`` ones are the uncapped pass-through edges, outside the
    conversion equation and the sum cap) and are left out.  Only
    unit output edges (``unit__outputNode``) contribute; processes
    without any are absent (callers treat them as ``W = 0``).
    """
    empty = pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                 "fuel_width": pl.Float64})
    outs = _try_entities(source, "unit__outputNode")
    if outs is None or outs.height == 0:
        return empty
    arcs = outs.lazy().select(alias_to_axis("unit", "p"),
                              alias_to_axis("node", "sink"))
    return (arcs
        .join(_arc_max_capacity_coef_lf(source, "sink"),
              on=["p", "sink"], how="left")
        .join(_arc_conversion_coef_lf(source, "sink")
                .rename({"coef": "conv"}),
              on=["p", "sink"], how="left")
        .with_columns(cap=pl.col("coef").fill_null(1.0).clip(lower_bound=0.0),
                      conv=pl.col("conv").fill_null(1.0))
        .filter(pl.col("conv") > 0.0)
        .sort(["p", "conv", "sink"], descending=[False, True, False])
        .with_columns(
            _prev=(pl.col("cap").cum_sum().over("p") - pl.col("cap")))
        .with_columns(
            _fill=pl.min_horizontal(
                pl.col("cap"),
                (1.0 - pl.col("_prev")).clip(lower_bound=0.0)))
        .group_by("p")
        .agg(fuel_width=(pl.col("conv") * pl.col("_fill")).sum()))


def _indirect_input_cap_lf(source: "InputSource",
                           pss: pl.DataFrame,
                           dt: pl.DataFrame,
                           p_slope: "Param | None",
                           p_section: "Param | None",
                           classified: pl.DataFrame,
                           ) -> pl.LazyFrame:
    """Input-arc limit per unit of capacity for indirect (multi-flow) units.

    One row per ``(p, source, sink, d, t)`` input arc of an indirect unit
    (``p ∈ process__method_indirect``, ``(p, source) ∈ process_source``,
    ``conversion_flow_coeff ≠ 0`` on the input — the zero-coefficient
    pass-through inputs are uncapped and absent)::

        value = input_share_max_s · L_s
        L_s   = (slope · W + section) · delay_factor / src_div_s

    ``L_s`` is the input flow (native units, per unit of unit capacity)
    that runs the unit at full output from input ``s`` alone: ``W`` is
    :func:`_indirect_fuel_width_lf`; ``section`` (the min-load-efficiency
    term, per unit online ≤ capacity) only for ``min_load_efficiency``
    processes; ``delay_factor = 1/max(delay weight)`` for delayed
    processes (one input step can feed a heavier-weighted output step);
    ``src_div_s`` is the input's ``conversion_flow_coeff`` (``1`` when
    ``≤ 0``), for delayed processes too: their delayed conversion term
    multiplies the input by ``conversion_flow_coeff`` exactly like the
    undelayed term (``_delay.delayed_input_expr``).  The full-load fuel is
    measured in fuel energy (flow × conversion_flow_coeff), so a poorer
    fuel automatically gets a larger flow limit.

    ``input_share_max`` (schema default 1: this input alone can run the
    unit at full output) is the largest share of the full-load fuel the
    input may supply.  Multiplied by availability × capacity (existing +
    invested − retired) in ``model/core.py``'s ``maxFlow`` handling.
    """
    keys = ["p", "source", "sink", "d", "t"]
    indirect_p = (classified.lazy()
                    .filter((pl.col("klass") == "unit")
                            & pl.col("method").is_in(list(_METHOD_INDIRECT)))
                    .select("p")
                    .unique())
    min_load_p = (classified.lazy()
                    .filter(pl.col("ct") == "min_load_efficiency")
                    .select("p")
                    .unique()
                    .with_columns(_has_min_load=pl.lit(True)))
    arcs = (pss.lazy().select("p", "source", "sink")
            .join(indirect_p, on="p", how="semi")
            .join(_process_source_pairs_lf(source), on=["p", "source"],
                  how="semi")
            .join(_process_source_sink_coeff_zero_lf(source, pss),
                  on=["p", "source", "sink"], how="anti"))
    base = (arcs
        .join(dt.lazy()
                 .select(alias_to_axis(pl.col("d"), "d"),
                         alias_to_axis(pl.col("t"), "t")),
              how="cross")
        .join(_arc_max_capacity_coef_lf(source, "source")
                .rename({"coef": "share"}),
              on=["p", "source"], how="left")
        .with_columns(share=pl.col("share").fill_null(1.0))
        .join(_arc_conversion_coef_lf(source, "source")
                .rename({"coef": "src_conv"}),
              on=["p", "source"], how="left")
        .with_columns(src_conv=pl.col("src_conv").fill_null(1.0))
        .join(_indirect_fuel_width_lf(source), on="p", how="left")
        .with_columns(fuel_width=pl.col("fuel_width").fill_null(0.0))
        .join(_delay_fuel_factor_lf(source), on="p", how="left")
        .with_columns(delay_factor=pl.col("delay_factor").fill_null(1.0))
        .join(min_load_p, on="p", how="left")
        .with_columns(
            _has_min_load=pl.col("_has_min_load").fill_null(False)))
    if p_slope is not None and p_slope.frame.height > 0:
        base = base.join(
            p_slope.frame.lazy().select(
                "p", "d", "t",
                pl.col("value").cast(pl.Float64).alias("slope")),
            on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(slope=pl.lit(None, dtype=pl.Float64))
    if p_section is not None and p_section.frame.height > 0:
        base = base.join(
            p_section.frame.lazy().select(
                "p", "d", "t",
                pl.col("value").cast(pl.Float64).alias("section")),
            on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(section=pl.lit(None, dtype=pl.Float64))
    return (base
        .with_columns(
            _fuel=(pl.col("slope").fill_null(0.0) * pl.col("fuel_width")
                   + pl.when(pl.col("_has_min_load"))
                       .then(pl.col("section").fill_null(0.0))
                       .otherwise(0.0)),
            # src_conv ≤ 0 inputs would feed the conversion negatively
            # (not tied to the fuel need): keep a finite limit with a
            # unit divisor.  Delayed and undelayed inputs alike enter the
            # conversion as flow × src_conv.
            _src_div=pl.when(pl.col("src_conv") <= 0.0)
                        .then(1.0)
                        .otherwise(pl.col("src_conv")))
        .select(*keys,
                value=(pl.col("share") * pl.col("_fuel")
                       * pl.col("delay_factor") / pl.col("_src_div"))))


def p_indirect_input_cap_from_source(source: "InputSource",
                                     pss: "pl.DataFrame | None",
                                     dt: "pl.DataFrame | None",
                                     p_slope: "Param | None",
                                     p_section: "Param | None",
                                     classified: pl.DataFrame | None = None,
                                     ) -> "Param | None":
    """``Param((p, source, sink, d, t))`` — the input-arc limit of indirect
    units per unit of capacity (:func:`_indirect_input_cap_lf`):
    ``input_share_max × full-load input flow per unit of capacity``.

    ``model/core.py`` multiplies it by availability and the unit's capacity:
    a per-element ``v_flow`` upper bound when the unit has no invest /
    divest variables in the solve, else a ``maxFlow`` row whose invest /
    divest terms carry the same multiplier.  ``None`` when there are no
    such arcs.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
        dt = cast_frame_axes(dt, _live)
        if classified is not None:
            classified = cast_frame_axes(classified, _live)
    if classified is None:
        classified = _classify_process_method(source)
    out = (_indirect_input_cap_lf(source, pss, dt, p_slope, p_section,
                                  classified)
           .sort("p", "source", "sink", "d", "t")
           .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.8.2 / §3.8.3 — pdt_uptime_set / pdt_downtime_set / lookbacks
# ---------------------------------------------------------------------------


def _process_min_time(source: "InputSource",
                        parameter_name: str) -> list[tuple[str, float]]:
    """Pull (entity, value) for ``unit.<parameter_name>`` (and connection
    when applicable) — value > 0 only.  Returns a list of (proc, min_time)
    pairs.
    """
    out: dict[str, float] = {}
    for ec in ("unit", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            if v > 0:
                out[r["name"]] = v
    return list(out.items())


def uptime_lookback_from_source(source: "InputSource",
                                    dt: pl.DataFrame | None,
                                    p_step_duration: object | None = None,
                                    link: pl.DataFrame | None = None,
                                    ) -> pl.DataFrame | None:
    """Build the ``uptime_lookback`` set ``(p, d, t, d_back, t_back)``: each step and its ancestors along the
    ramp predecessor links *link* (``dt_ramp_link``) within ``min_uptime`` (``params._mintime.mintime_lookback``)."""
    from flextool.engine.preprocess.params._mintime import mintime_lookback
    return mintime_lookback(dt, p_step_duration, link, _process_min_time(source, "min_uptime"))


def downtime_lookback_from_source(source: "InputSource",
                                       dt: pl.DataFrame | None,
                                       p_step_duration: object | None = None,
                                       link: pl.DataFrame | None = None,
                                       ) -> pl.DataFrame | None:
    """Build the ``downtime_lookback`` set (as :func:`uptime_lookback_from_source`, with ``min_downtime``)."""
    from flextool.engine.preprocess.params._mintime import mintime_lookback
    return mintime_lookback(dt, p_step_duration, link, _process_min_time(source, "min_downtime"))


def apply_mintime_lookbacks(flex_data: object, source: "InputSource",
                            first_step_link: pl.DataFrame | None, solve_name: str) -> None:
    """Set ``uptime_lookback`` / ``downtime_lookback`` and the projected ``pdt_*_set`` of ``flex_data`` from its
    ``dt``, ``p_step_duration`` and ``dt_ramp_link``, the roll carries ``uptime_carry`` / ``downtime_carry``
    (:func:`online_mintime_carry_from_source`), and print a warning for each process whose look-back the
    solve boundary cuts short (``params._mintime.cut_short_lookback_warnings``; ``first_step_link`` is the
    complete solve's ``step_previous`` of this solve's period first steps, ``None`` when not written)."""
    from flextool.engine.preprocess.params._mintime import report_cut_short_lookback
    dt = getattr(flex_data, "dt", None)
    sd = getattr(flex_data, "p_step_duration", None)
    link = getattr(flex_data, "dt_ramp_link", None)
    ulb = uptime_lookback_from_source(source, dt, sd, link)
    if ulb is not None and ulb.height > 0:
        flex_data.uptime_lookback = ulb
        flex_data.pdt_uptime_set = pdt_uptime_set_from_lookback(ulb)
    dlb = downtime_lookback_from_source(source, dt, sd, link)
    if dlb is not None and dlb.height > 0:
        flex_data.downtime_lookback = dlb
        flex_data.pdt_downtime_set = pdt_downtime_set_from_lookback(dlb)
    if ulb is None and dlb is None:
        return
    flex_data.uptime_carry = (online_mintime_carry_from_source(source, flex_data, "min_uptime")
                              if ulb is not None else None)
    flex_data.downtime_carry = (online_mintime_carry_from_source(source, flex_data, "min_downtime")
                                if dlb is not None else None)
    min_times = {"min_uptime": _process_min_time(source, "min_uptime"),
                 "min_downtime": _process_min_time(source, "min_downtime")}
    report_cut_short_lookback(min_times, dt, first_step_link, getattr(flex_data, "dt_ramp_carry", None),
                              solve_name)


def pdt_uptime_set_from_lookback(uptime_lookback: pl.DataFrame | None
                                     ) -> pl.DataFrame | None:
    """Project ``(p, d, t, d_back, t_back) → (p, d, t)`` distinct."""
    if uptime_lookback is None or uptime_lookback.height == 0:
        return None
    return (uptime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


def pdt_downtime_set_from_lookback(downtime_lookback: pl.DataFrame | None
                                       ) -> pl.DataFrame | None:
    if downtime_lookback is None or downtime_lookback.height == 0:
        return None
    return (downtime_lookback.lazy()
              .select("p", "d", "t").unique()
              .sort("p", "d", "t")
              .collect())


def online_mintime_carry_from_source(source: "InputSource", flex_data: object, parameter_name: str
                                     ) -> pl.DataFrame | None:
    """The roll carry of the minimum up / down time ``(p, d, t, value)`` (Utf8) of *flex_data*.

    ``parameter_name`` ``"min_uptime"`` carries the realized ``v_startup`` (``uptime_carry``, on the
    ``minimum_uptime`` RHS), ``"min_downtime"`` the realized ``v_shutdown`` (``downtime_carry``, on the
    ``minimum_downtime`` LHS).  The previous solve's realized on-line history (``online_startup_history``, each step
    with its ``hours_to_end``) reaches the steps that follow the roots ``dt_ramp_carry`` along ``dt_ramp_link``
    (``params._mintime.mintime_roll_carry``): the first steps that continue from the previous solve's last realized
    step on the timeline — a period that continues the previous one and the forecast branches that start with such a
    step included, a period that does not continue it (``jump != 1``) and a step after a gap excluded.  ``L`` is the
    authored parameter.  ``None`` when nothing is carried."""
    from flextool.engine.preprocess.params._mintime import mintime_roll_carry
    history = getattr(flex_data, "online_startup_history", None)
    if history is None or history.height == 0:
        return None
    return mintime_roll_carry(getattr(flex_data, "dt", None), getattr(flex_data, "p_step_duration", None),
                              getattr(flex_data, "dt_ramp_link", None), getattr(flex_data, "dt_ramp_carry", None),
                              history, _process_min_time(source, parameter_name),
                              "startup" if parameter_name == "min_uptime" else "shutdown",
                              getattr(flex_data, "period_branch_full", None))


# ---------------------------------------------------------------------------
# §3.12.1 — process_group_inside_nonSync
# ---------------------------------------------------------------------------


def process_group_inside_nonSync_from_flex(fd: object) -> pl.DataFrame | None:
    """``(p, g)``: the processes whose flows stay inside the non-synchronous group g (internal transfers).

    inside = {(p,g) : g ∈ groupNonSync, ∃ s ∈ src(p) ∩ N(g), k ∈ snk(p) ∩ N(g),
                      not (|src(p) ∩ N(g)| = |snk(p) ∩ N(g)| = 1 ∧ s = k)}
      src(p) = sources of the arcs of p in process_source_sink (other than p itself),
      snk(p) = sinks of the arcs of p (other than p itself),  N(g) = nodes of g (group_node)
    (flextool.mod l.4065/4075, ``emit/_mid_sets.derive_process__group_inside_group_nonSync``).  A process
    with a source and a different sink node in g moves energy between the group's nodes: the non-synchronous
    limit counts neither its infeed nor its draw.  A regular connection with one end in g has the same single
    node on both sides, so it is not inside.  One rule for the single-solve and the synthetic-solve paths.
    Sorted by (p, g); None when no process is inside a group.
    """
    pss = getattr(fd, "process_source_sink", None)
    gn = getattr(fd, "group_node", None)
    gns = getattr(fd, "groupNonSync", None)
    if pss is None or gn is None or gns is None or pss.height == 0 or gn.height == 0 or gns.height == 0:
        return None
    gns_g, gn_g = align_join_dtypes(gns.select("g"), gn.select("g", "n"), ("g",))
    gn_ns = gn_g.join(gns_g, on="g", how="semi")
    if gn_ns.height == 0:
        return None
    ends = []
    for side in ("source", "sink"):
        arcs = (pss.filter(pl.col(side).cast(pl.Utf8) != pl.col("p").cast(pl.Utf8))
                .select("p", pl.col(side).alias("n")).unique())
        arcs, gnj = align_join_dtypes(arcs, gn_ns, ("n",))
        ends.append(arcs.join(gnj, on="n", how="inner")
                    .group_by("p", "g")
                    .agg(pl.col("n").n_unique().alias(f"{side}_count"),
                         pl.col("n").cast(pl.Utf8).min().alias(f"{side}_node")))
    both = ends[0].join(ends[1], on=["p", "g"], how="inner")
    single_node = ((pl.col("source_count") == 1) & (pl.col("sink_count") == 1)
                   & (pl.col("source_node") == pl.col("sink_node")))
    out = both.filter(~single_node).select("p", "g").sort("p", "g")
    return out if out.height > 0 else None


# ---------------------------------------------------------------------------
# §3.12.2 — p_inv_group_cap / p_group_capacity_for_scaling
# ---------------------------------------------------------------------------


def p_group_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  *,
                                                  ctx: "SolveContext | None" = None,
                                                  provider: "object | None" = None,
                                                  ) -> "Param | None":
    """Per-(g, d) row scaling factor.  Algorithm (audit §3.12.2,
    ``lp_scaling_params.py:91-244``): pow10-clamped per-group capacity
    proxy when scaling is active, else 1.0 across all (g, d).

    For Γ.3.C we cover the **scaling-inactive default**: emit 1.0 across
    all (group, period) pairs (including stochastic-branch siblings when
    workdir-provided ``period_in_use_set.csv`` is read).  Active scaling
    (rare; activated via ``solve.use_row_scaling``) returns None and the
    CSV value survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir,
                                        ctx=ctx, provider=provider)
    if not period_in_use:
        return None
    # A4-followup: broadcast to stochastic-branch siblings — same
    # rationale as p_node_capacity_for_scaling.
    period_in_use = _expand_branch_periods(period_in_use, workdir,
                                            ctx=ctx, provider=provider)
    groups_df = _try_entities(source, "group")
    if groups_df is None or groups_df.height == 0:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    out = (groups_df.lazy().select(alias_to_axis("name", "g"))
              .join(axis_lazyframe({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("g", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("g", "d"), out)


def p_inv_group_cap_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    workdir: Path | None = None,
                                    *,
                                    ctx: "SolveContext | None" = None,
                                    provider: "object | None" = None,
                                    ) -> "Param | None":
    """Reciprocal of ``p_group_capacity_for_scaling``.  When scaling is
    inactive (default), this is 1.0 per (g, d).
    """
    gcs = p_group_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if gcs is None:
        return None
    out = (gcs.frame.lazy()
              .with_columns(value=1.0 / pl.col("value"))
              .sort("g", "d")
              .collect())
    return Param(("g", "d"), out)


def p_node_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  *,
                                                  ctx: "SolveContext | None" = None,
                                                  provider: "object | None" = None,
                                                  ) -> "Param | None":
    """Per-(n, d) row scaling factor for slack-penalty terms in the
    objective (``vq_up``/``vq_down`` × ``p_penalty`` × this factor).

    Δ.29 — tactical default-1.0 path matching
    :func:`p_group_capacity_for_scaling_from_source`.  Algorithm
    (audit §3.12.2, ``lp_scaling_params.py:91-244``): pow10-clamped
    per-node capacity proxy when ``solve.use_row_scaling`` is active,
    else 1.0 across all (nodeBalance, period_in_use) pairs.

    For Δ.29 we cover the **scaling-inactive default**: emit 1.0 across
    nodeBalance × period_in_use.  Active scaling (rare; gated on
    ``solve.use_row_scaling``) returns None and the CSV value survives.

    The slow path's :func:`_load_node_capacity_for_scaling` reads
    ``solve_data/node_capacity_for_scaling.csv`` and filters to
    nodeBalance via ``df.join(nb, on='n', how='inner')``; we replicate
    that filter inline here.

    TODO(future): port the full pow10 cascade in
    ``preprocessing/lp_scaling_params.py`` for fixtures with
    ``solve.use_row_scaling=1``.  For now the helper bails (returns
    None) when scaling is active so the seed CSV survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir,
                                        ctx=ctx, provider=provider)
    if not period_in_use:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    # nodeBalance projection — same set as the slow path's filter.  Slack
    # vars span ``nodeBalance ∪ nodeBalancePeriod`` (period nodes also
    # carry ``vq_state_up/down``), so union the period nodes in before the
    # cross-join.  Dtype-safe; when the period set is empty the union
    # collapses to exactly ``nb`` → byte-identical legacy behavior.
    from flextool.engine.preprocess.params._projection import (
        nodeBalance as _nodeBalance_proj,
        nodeBalancePeriod as _nodeBalancePeriod_proj,
    )
    nb = _nodeBalance_proj(source)
    if nb is None or nb.height == 0:
        return None
    nbp = _nodeBalancePeriod_proj(source)
    if nbp is not None and nbp.height > 0:
        nb = (pl.concat([nb.select("n"), nbp.select("n")])
                .unique()
                .select("n"))
    out = (nb.lazy()
              .join(axis_lazyframe({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("n", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d"), out)


# ---------------------------------------------------------------------------
# §3.12.3 — p_positive_inflow / p_negative_inflow
# ---------------------------------------------------------------------------


def p_positive_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Positive component (clip-low at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") > 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


def p_negative_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Negative component (clip-high at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") < 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


def real_inflow(p_inflow: object | None, p_inflow_unmargined: object | None) -> object | None:
    """The node inflow before the invest-solve energy margin: the real demand.

    ``p_inflow_unmargined`` is ``solve_data/pdtNodeInflow_unmargined``, written by the energy-margin
    emitters whenever a solve on the Provider applies a margin and refreshed by every later solve; without
    it no solve has a margin and the real inflow is ``p_inflow`` itself.  The group rows (non-synchronous
    limit, capacity margin, the group demand of the dynamic reserve requirement) read the real demand: the
    energy margin is an adequacy device and not load.
    """
    return p_inflow if p_inflow_unmargined is None else p_inflow_unmargined


# ---------------------------------------------------------------------------
# §3.12.4 — pdtNodeInflow_per_step
# ---------------------------------------------------------------------------


def pdtNodeInflow_per_step_from_inflow(p_inflow: object | None,
                                          p_step_duration: object | None,
                                          ) -> "Param | None":
    """``p_inflow / p_step_duration`` per (n, d, t).  Used by
    capacity_margin RHS.  Mirrors the .mod's
    ``pdtNodeInflow / step_duration`` divider.

    Note: ``p_inflow`` here is flextool's ``pdtNodeInflow`` shape — a
    long-format (n, d, t, value) frame already integrated over the step.
    Dividing by ``step_duration`` returns the per-hour rate.
    """
    if p_inflow is None or p_step_duration is None:
        return None
    inflow_fr = (p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow)
    sd_fr = (p_step_duration.frame if hasattr(p_step_duration, "frame")
              else p_step_duration)
    if (inflow_fr is None or inflow_fr.height == 0
            or sd_fr is None or sd_fr.height == 0):
        return None
    sd_lf = sd_fr.lazy().select(
        "d", "t", pl.col("value").cast(pl.Float64).alias("dur"))
    out = (inflow_fr.lazy()
              .join(sd_lf, on=["d", "t"], how="inner")
              .with_columns(value=pl.col("value") / pl.col("dur"))
              .select("n", "d", "t", "value")
              .sort("n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)


# ===========================================================================
# Γ.3.C integration entrypoint
# ===========================================================================


DERIVED_C_FIELDS = (
    # §3.7
    "ed_invest_set", "ed_divest_set",
    "edd_invest_lookback_set", "edd_divest_active", "ndd_divest_active",
    "p_entity_max_units",
    # §3.8
    "p_section",
    "uptime_lookback", "downtime_lookback",
    "pdt_uptime_set", "pdt_downtime_set",
    # §3.12
    "process_group_inside_nonSync",
    "p_group_capacity_for_scaling", "p_inv_group_cap",
    "p_node_capacity_for_scaling",
    "p_positive_inflow", "p_negative_inflow",
    "pdtNodeInflow_per_step",
    # §3.1.3 multi-year (extends Γ.3.A)
    "p_inflation_op",
)


def _set_eq_sorted(a: pl.DataFrame | None,
                     b: pl.DataFrame | None,
                     keys: tuple[str, ...]) -> bool:
    """Frame equality on a *set-shaped* DataFrame (no value column).

    Sort by ``keys`` then call ``.equals``.  Either being None or empty
    → equal iff the other is None / empty.
    """
    a_empty = a is None or a.height == 0
    b_empty = b is None or b.height == 0
    if a_empty and b_empty:
        return True
    if a_empty or b_empty:
        return False
    cols_a = set(a.columns)
    cols_b = set(b.columns)
    if cols_a != cols_b:
        return False
    sk = [k for k in keys if k in cols_a]
    return a.sort(sk).equals(b.sort(sk))


def _build_commission_lf(source: "InputSource",
                             active_solve: str | None,
                             ed_lf: "pl.LazyFrame",
                             period_in_use,
                             workdir: "Path | None",
                             *,
                             provider: "object | None" = None,
                             ) -> "pl.LazyFrame | None":
    """The start-of-operation frame of the investment decisions (construction lead time), failures surfaced.

    Shared by :func:`apply_derived_c` and :func:`apply_synthetic_invest_sets`.  Returns the frame of
    :func:`._existing.commissioning_year_lf` on the run calendar the sub-solve ``provider`` carries, or ``None``
    when there are no in-use periods or no decision has a lead time.  ``provider`` gives only the run calendar;
    the solve's year calendar is resolved as the availability walks resolve it.

    A :class:`CommissioningLagError` from the helper propagates unchanged.  Any other helper failure is
    swallowed to ``None`` only when the model configures no construction lead time (byte-parity for models
    without one); on a lag-configured model it is re-raised as :class:`CommissioningLagError`, so a
    misconfiguration never turns into a silently unlagged solve.
    """
    if not period_in_use:
        return None
    from flextool.engine.preprocess.params._existing import (
        _construction_lag_configured as _lag_configured,
        commissioning_year_lf as _commissioning_year_lf,
    )
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    try:
        return _commissioning_year_lf(
            source, active_solve, ed_lf, list(period_in_use),
            workdir=workdir, run_calendar=read_run_calendar(provider))
    except CommissioningLagError:
        raise
    except Exception as exc:
        if _lag_configured(source):
            raise CommissioningLagError(
                "Finding the start of operation of investments with a construction_lead_time failed; refusing "
                "to solve the model without the lead time."
            ) from exc
        return None


def _offer_invest_set(flex_data: object,
                         source: "InputSource",
                         active_solve: str | None,
                         ed_inv: "pl.DataFrame | None",
                         period_in_use,
                         workdir: "Path | None",
                         *,
                         provider: "object | None" = None,
                         ) -> tuple["pl.DataFrame | None", "pl.LazyFrame | None"]:
    """``(ed_invest offered, commission frame)``: drop the investment decisions whose capacity would start to
    operate after the solve or the model horizon (with a warning), and keep the start of operation of the rest
    on ``flex_data.ed_invest_commission`` (``[e, d, d_c, yr_c]``, the decisions with a lead time: the start
    period and its year on the solve calendar; ``None`` without).

    When decisions are dropped, the invest-set fields of ``flex_data`` are cleared, so no earlier value (seed)
    brings them back where the offered set leaves one empty."""
    flex_data.ed_invest_commission = None
    if ed_inv is None or ed_inv.height == 0 or not period_in_use:
        return ed_inv, None
    from flextool.engine.preprocess.params._existing import (
        offered_invest_set as _offered,
        warn_investments_not_offered as _warn,
    )
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    commission_lf = _build_commission_lf(
        source, active_solve, ed_inv.lazy().select("e", "d"), period_in_use, workdir, provider=provider)
    if commission_lf is None:
        return ed_inv, None
    from flextool.engine import _behaviour_changes
    lagged = commission_lf.select(pl.col("e").cast(pl.Utf8)).unique().sort("e").collect()["e"].to_list()
    _behaviour_changes.notice("construction_lead_time_commissioning", "entities", lagged)
    offered, late = _offered(ed_inv, commission_lf)
    _warn(source, active_solve, late, list(period_in_use), read_run_calendar(provider))
    kept = (commission_lf.filter(pl.col("yr_c").is_finite())
                         .select("e", "d", "d_c", "yr_c").collect())
    flex_data.ed_invest_commission = kept if kept.height > 0 else None
    if late is not None:
        # The sets built from ed_invest_set are set below only where they have rows; clear the earlier values
        # (seeds) so a set left empty by the dropped decisions does not keep them.
        for name in ("ed_invest_set", "pd_invest_set", "nd_invest_set", "ed_invest_period_set",
                     "edd_invest_set", "edd_invest_lookback_set"):
            setattr(flex_data, name, None)
    return offered, commission_lf


def _apply_lead_time_sets(flex_data: object,
                              source: "InputSource",
                              active_solve: str | None,
                              ed_inv: "pl.DataFrame | None",
                              commission_lf: "pl.LazyFrame | None",
                              period_in_use,
                              workdir: "Path | None",
                              *,
                              lineage: "pl.DataFrame | None" = None,
                              ) -> None:
    """``flex_data.edd_invest_commission`` / ``edd_invest_in_construction`` / ``e_construction_lead_time`` (``[e, L]``,
    the longest lead time of the entity's decisions) of a solve where some investment decision has a construction
    lead time (offered or not); all ``None`` otherwise, so the model keeps the formulation without lead times.

    Then checks the period maps of the per-period investment limits against the periods in which the solve's
    investments start to operate (:func:`._direct.require_invest_period_coverage`)."""
    _set_lead_time_sets(flex_data, source, active_solve, ed_inv, commission_lf, period_in_use, workdir,
                        lineage=lineage)
    from flextool.engine.preprocess.params._direct import require_invest_period_coverage
    require_invest_period_coverage(source, flex_data, active_solve)


def _set_lead_time_sets(flex_data: object,
                        source: "InputSource",
                        active_solve: str | None,
                        ed_inv: "pl.DataFrame | None",
                        commission_lf: "pl.LazyFrame | None",
                        period_in_use,
                        workdir: "Path | None",
                        *,
                        lineage: "pl.DataFrame | None" = None,
                        ) -> None:
    """The lead-time sets of :func:`_apply_lead_time_sets`."""
    flex_data.edd_invest_commission = None
    flex_data.edd_invest_in_construction = None
    flex_data.e_construction_lead_time = None
    if commission_lf is None or ed_inv is None or ed_inv.height == 0:
        return
    from flextool.engine.preprocess.params._existing import lead_time_sets as _lead_time_sets
    commission, in_construction = _lead_time_sets(
        source, active_solve, ed_inv, getattr(flex_data, "edd_invest_set", None), commission_lf,
        list(period_in_use), workdir, lineage=lineage)
    flex_data.edd_invest_commission = commission if commission.height > 0 else None
    flex_data.edd_invest_in_construction = in_construction if in_construction.height > 0 else None
    lagged = (commission_lf.select(pl.col("e").cast(pl.Utf8), "L").group_by("e").agg(pl.col("L").max())
                           .sort("e").collect())
    flex_data.e_construction_lead_time = lagged if lagged.height > 0 else None


def _fan_branch_years_notice(source: "InputSource",
                             flex_data: object,
                             ed_inv: "pl.DataFrame | None",
                             ed_div: "pl.DataFrame | None") -> None:
    """Behaviour-change notice of a solve with a stochastic fan without recourse: its branch periods now have the
    years of their period labels in the investment, retirement and lifetime windows.  Names the entities with
    investment or retirement options and those with a ``reinvest_choice`` / ``no_investment`` lifetime."""
    if _fan_walk_provider() is None or getattr(flex_data, "recourse_invest", False):
        return
    names: dict[str, None] = {}
    for frame in (ed_inv, ed_div):
        if frame is not None and frame.height > 0:
            for e in frame["e"].cast(pl.Utf8).unique().sort().to_list():
                names.setdefault(str(e), None)
    for e, methods in _entity_lifetime_methods(source, _all_entities(source)).items():
        if "reinvest_choice" in methods or "no_investment" in methods:
            names.setdefault(e, None)
    if names:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("stochastic_branch_period_years", "entities", sorted(names))


def _retirement_without_existing_warning(source: "InputSource", offered: pl.DataFrame) -> None:
    """Warn about the entities that have a retirement option but no ``existing`` capacity in any period (none
    given, or 0 everywhere): only capacity that existed before the run can be retired, so the option has no
    effect — no retirement variable, and their retired-capacity terms (user constraints, group retirement
    limits) are 0.  Each entity once per run.  An entity with existing capacity whose retirement periods are past
    its end of life is a valid model and is not warned about (the end of life is in the retirement notice)."""
    from flextool.engine import _behaviour_changes
    from flextool.engine.preprocess.params._npv import _per_entity_param_lf
    with_existing = (_per_entity_param_lf(source, "existing")
                       .filter(pl.col("value").cast(pl.Float64, strict=False) > 0.0)
                       .select(pl.col("e").cast(pl.Utf8)).unique().collect())
    lost = (offered.select(pl.col("e").cast(pl.Utf8)).unique()
                   .join(with_existing, on="e", how="anti").sort("e")["e"].to_list())
    if not lost:
        return
    _behaviour_changes.warn_once_each(
        "retirement_without_existing", lost,
        lambda names: "Warning: " + ", ".join(names) + (" have" if len(names) > 1 else " has")
        + " a retirement option (invest_method) but no existing capacity. Only existing capacity can be retired, "
        "so there is no retirement variable; retired-capacity terms of user constraints and group retirement "
        "limits are 0.")


def _check_retirement_minimums(flex_data: object,
                               source: "InputSource",
                               active_solve: str | None,
                               workdir: Path | None,
                               offered: "pl.DataFrame | None",
                               kept: "pl.DataFrame | None",
                               active_frames: "list[pl.DataFrame | None]",
                               period_in_use: list[str],
                               *,
                               provider: "object | None" = None) -> None:
    """:func:`._retire_existing.check_retirement_minimums` on the solve's retirement options."""
    if offered is None or offered.height == 0:
        return
    from flextool.engine.preprocess.params._existing import (
        e_divest_total_lf,
        ed_divest_period_set_lf,
        p_entity_pre_existing_lf,
        p_entity_retired_carried_from_handoff,
    )
    from flextool.engine.preprocess.params._retire_existing import (
        active_windows,
        check_retirement_minimums,
        ed_values,
    )
    pre_df = (p_entity_pre_existing_lf(source, active_solve, [str(d) for d in period_in_use],
                                       workdir=workdir, provider=provider)
                .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "value").collect())
    pre = {(e, d): float(v) for e, d, v in pre_df.iter_rows()}
    years = _lifetime_years(source, active_solve, workdir, provider=provider)
    check_retirement_minimums(
        offered=offered, kept=kept,
        period_method_pairs=ed_divest_period_set_lf(source, offered.lazy()).collect(),
        total_method_entities=e_divest_total_lf(source).collect()["e"].cast(pl.Utf8).to_list(),
        windows=active_windows(active_frames), pre=pre,
        min_period=getattr(flex_data, "ed_divest_min_period", None),
        min_total=getattr(flex_data, "e_divest_min_total", None),
        prior=p_entity_divested_from_handoff(provider=provider),
        carried=ed_values(p_entity_retired_carried_from_handoff(source, active_solve, list(period_in_use), workdir,
                                                                provider=provider)),
        eol=_lifetime_gate_sums(source, active_solve, workdir, provider=provider, years=years))


def _end_of_life_notice(source: "InputSource",
                        active_solve: str | None,
                        workdir: Path | None,
                        *,
                        provider: "object | None" = None) -> None:
    """Behaviour-change notice ``retirement_of_existing_capacity_only`` for the end of life of existing capacity:
    names the reinvest_choice / no_investment entities of a solve whose end of life changed, a later solve of a
    chain (its period history starts before its own first period) or a solve whose first period branches."""
    from flextool.engine import _behaviour_changes
    if not _behaviour_changes.has_marker("retirement_of_existing_capacity_only"):
        return
    first = _read_period_first(source, active_solve, workdir, provider=provider)
    years = _lifetime_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(years)
    later_in_chain = bool(first) and start is not None and years.get(first[0], 0.0) > years.get(start, 0.0)
    if not (later_in_chain or len(first) > 1):
        return
    names = sorted(e for e, life_sum in _lifetime_gate_sums(source, active_solve, workdir, provider=provider,
                                                            years=years).items() if life_sum > 0.0)
    if names:
        _behaviour_changes.notice("retirement_of_existing_capacity_only", "entities", names)


def _retirement_notice(offered: pl.DataFrame, kept: pl.DataFrame) -> None:
    """Behaviour-change notice ``retirement_of_existing_capacity_only``: names the entities with a retirement
    option (and, in the detail, those left without one for lack of existing capacity)."""
    from flextool.engine import _behaviour_changes
    names = sorted(set(offered["e"].cast(pl.Utf8).to_list()))
    lost = sorted(set(names) - set(kept["e"].cast(pl.Utf8).to_list()))
    detail = ("Without existing capacity in the retirement periods (no retirement variable): "
              + ", ".join(lost) + ".") if lost else ""
    _behaviour_changes.notice("retirement_of_existing_capacity_only", "entities", names, detail)


#: User-constraint coefficients that offered an entity's investment (retirement) variable before
#: ``invest_without_cost_offered`` whatever its cost: such entities are not named by the notice.
_OLD_RULE_INVEST_COEFFS: tuple[str, ...] = (
    "constraint_invested_capacity_coeff",
    "constraint_cumulative_pre_built_capacity_coeff",
    "constraint_capacity_coeff",
    "constraint_in_construction_capacity_coeff",
)
_OLD_RULE_DIVEST_COEFFS: tuple[str, ...] = (*_OLD_RULE_INVEST_COEFFS, "constraint_retired_capacity_coeff")


def _zero_cost_pairs(source: "InputSource", pairs: "pl.DataFrame | None", cost_param: str,
                     coeffs: tuple[str, ...]) -> dict[str, list[str]]:
    """``{entity: [periods]}`` of the ``pairs`` (``[e, d]``) whose ``cost_param`` is missing or 0 in the period,
    leaving out the entities a user-constraint capacity coefficient names (their variable existed before)."""
    if pairs is None or pairs.height == 0:
        return {}
    rows = [(str(e), str(d)) for e, d in pairs.select("e", "d").iter_rows()]
    coeff_entities: set[str] = set()
    for cls in ("unit", "node", "connection"):
        for pname in coeffs:
            df = _try_param(source, cls, pname)
            if df is not None and "name" in df.columns:
                coeff_entities.update(df.get_column("name").cast(pl.Utf8).to_list())
    periods = list(dict.fromkeys(d for _e, d in rows))
    cost = _ed_period_value_map(source, cost_param, periods)
    out: dict[str, list[str]] = {}
    for e, d in rows:
        if e not in coeff_entities and cost.get((e, d), 0.0) == 0.0:
            out.setdefault(e, []).append(d)
    return out


def zero_cost_invest_notice(source: "InputSource", ed_invest: "pl.DataFrame | None",
                            ed_divest: "pl.DataFrame | None") -> None:
    """Behaviour-change notice ``invest_without_cost_offered``: names the entities that now get an investment
    variable where their ``invest_cost`` is missing or 0, or a retirement variable where their ``salvage_value``
    is (before, such a period had no variable unless a user-constraint capacity coefficient named the entity)."""
    from flextool.engine import _behaviour_changes
    if not _behaviour_changes.has_marker("invest_without_cost_offered"):
        return
    invest = _zero_cost_pairs(source, ed_invest, "invest_cost", _OLD_RULE_INVEST_COEFFS)
    divest = _zero_cost_pairs(source, ed_divest, "salvage_value", _OLD_RULE_DIVEST_COEFFS)
    # A later solve of a multi-solve / rolling run lists only the entities not yet announced.
    fresh = set(_behaviour_changes.unannounced("invest_without_cost_offered", sorted(set(invest) | set(divest))))
    invest = {e: ds for e, ds in invest.items() if e in fresh}
    divest = {e: ds for e, ds in divest.items() if e in fresh}

    def _listed(found: dict[str, list[str]]) -> str:
        return ", ".join(f"{e} ({', '.join(ds)})" for e, ds in sorted(found.items()))

    detail = []
    if invest:
        detail.append("Investment without invest_cost (entity, periods): " + _listed(invest) + ".")
    if divest:
        detail.append("Retirement without salvage_value (entity, periods): " + _listed(divest) + ".")
    _behaviour_changes.notice("invest_without_cost_offered", "entities", sorted(set(invest) | set(divest)),
                              " ".join(detail))


def apply_derived_c(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.C Derived Params, mutating ``flex_data`` in place.

    Order:
      1. Multi-year inflation (extends Γ.3.A's default-rate path).
      2. §3.12 group slack (smallest, foundational for cap_margin).
      3. §3.8 online/UC.
      4. §3.7 invest/divest cascade.
      5. §3.11 (capped by §3.7.5/6 — no separate Params).

    Δ.3 replaced the previous ``derived_overrides_c`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    dt_csv = getattr(flex_data, "dt", None)
    sd_csv = getattr(flex_data, "p_step_duration", None)

    # Slice D — resolve the per-scenario stochastic-invest flag once and
    # stamp it on FlexData (survives the region-filter dataclasses.replace)
    # so the model layer reads ``d.recourse_invest`` without a workdir touch.
    flex_data.recourse_invest = _recourse_invest_active(
        workdir, ctx=ctx, provider=provider)
    # Slice D §4 — arm the per-period anchor-map scope for the invest/
    # divest set predicates + cap readers built below (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, ctx=ctx, provider=provider)
    # Slice D §8 (E) — the scenario-lineage frame for the edd builders.
    # Active only under recourse AND a genuine branch invest axis — the
    # anchor-pairs holder just armed above is exactly that conjunct
    # (non-None ⟺ recourse-active AND synthetic branches in PIU).  Build
    # the lineage frame ON-DEMAND from the provider (``dd_same_scenario_df``
    # is a standalone builder): ``flex_data.dd_same_scenario`` is not
    # populated until ``apply_branch_cluster`` in derived_g, which runs
    # AFTER this boundary.  Hoist the precondition check ABOVE the edd
    # try/except blocks (§6) so a LineageFilterError is loud, not swallowed.
    # ``None`` flag-off → edd builders keep today's unfiltered behaviour.
    if _RECOURSE_ANCHOR_PAIRS is not None:
        from flextool.engine.preprocess.params._branch import (
            assert_recourse_npv_preconditions as _assert_recourse,
            dd_same_scenario_df as _dd_same_scenario_df,
        )
        _c_lineage = _dd_same_scenario_df(
            workdir, source, active_solve, ctx=ctx, provider=provider)
        _assert_recourse(workdir, source, active_solve,
                         ctx=ctx, provider=provider)
    else:
        _c_lineage = None

    # Δ.12b — assignment is unconditional except for fields with
    # documented helper-coverage gaps (multi-year cascade extends
    # apply_derived_a's path; fall-throughs noted inline).

    # ─── Multi-year inflation cascade ─────────────────────────────────
    # TODO(Δ.12b helper-fix): the multi-year helper EXTENDS rather than
    # replaces apply_derived_a's p_inflation_op (single-year path) — when
    # the multi-year cascade isn't applicable (single-year fixture) the
    # helper returns None and we want to retain apply_derived_a's value.
    # Keep the conditional assignment.
    if active_solve is not None and dt_csv is not None:
        infl_my = p_inflation_op_multi_year_from_source(
            source, active_solve, dt_csv)
        if infl_my is not None:
            flex_data.p_inflation_op = infl_my

    # ─── §3.12 group slack ────────────────────────────────────────────
    # TODO(Δ.12b helper-fix): the helpers cover the scaling-inactive
    # default path; scaling-active fixtures (``solve.use_row_scaling``)
    # require the pow10 cascade in ``lp_scaling_params.py:91-244`` which
    # the seed-side ``_group_slack`` reads directly from
    # ``solve_data/group_capacity_for_scaling.csv`` /
    # ``solve_data/inv_group_cap.csv``.  Keep the conditional assignment.
    gcs_db = p_group_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if gcs_db is not None:
        flex_data.p_group_capacity_for_scaling = gcs_db
    igc_db = p_inv_group_cap_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if igc_db is not None:
        flex_data.p_inv_group_cap = igc_db
    # Δ.29 — tactical default-1.0 path for p_node_capacity_for_scaling.
    # The proper fix is the lp_scaling_params pow10 cascade
    # (``preprocessing/lp_scaling_params.py:91-244``); until that lands,
    # default to 1.0 broadcast across (nodeBalance, period_in_use).
    # Without this default, the fast path leaves the field None for
    # fixtures with no preprocessed CSV → the slack penalty becomes
    # the only sink for unmet demand without the row-scaling factor,
    # but more critically when the pow10 cascade is active CSV-side and
    # the fast path emits None, the LP misses the scaling on penalty
    # terms (penalty becomes a free source of energy).  Default 1.0
    # produces the same numeric coefficient as the None-branch in
    # ``model._obj_balance_slack`` but exposes the field uniformly so future
    # consumers (e.g. region filter, warm cascade) see a populated
    # frame.
    ncs_db = p_node_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if ncs_db is not None:
        flex_data.p_node_capacity_for_scaling = ncs_db

    # Group-accounting inflow frames — from the real inflow (before the invest-solve energy margin; the
    # CSV-loaded or override-set p_inflow when no solve has a margin).  Helpers return None when it is None.
    inflow_real = real_inflow(getattr(flex_data, "p_inflow", None), getattr(flex_data, "p_inflow_unmargined", None))
    flex_data.p_positive_inflow = p_positive_inflow_from_inflow(inflow_real)
    flex_data.p_negative_inflow = p_negative_inflow_from_inflow(inflow_real)
    flex_data.pdtNodeInflow_per_step = pdtNodeInflow_per_step_from_inflow(inflow_real, sd_csv)

    # process_group_inside_nonSync — internal transfers of the non-synchronous groups.
    flex_data.process_group_inside_nonSync = process_group_inside_nonSync_from_flex(flex_data)

    # ─── §3.8 online / UC ─────────────────────────────────────────────
    classified = _classify_process_method(source)
    if dt_csv is not None and classified is not None and classified.height > 0:
        flex_data.p_section = p_section_from_source(source, dt_csv, classified)

    # The uptime / downtime look-backs follow the ramp predecessor links, which apply_derived_e builds from the
    # source timeline; they are set there.

    # ─── §3.7 invest / divest ─────────────────────────────────────────
    # Δ.12b — set frames (None or non-empty); keep height>0 as a
    # structural filter to preserve the SET-frame contract.
    ed_inv_db = ed_invest_set_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    # Construction lead time: the start of operation of each investment decision, on the run calendar.  The
    # decisions whose capacity would start to operate after the solve or the model horizon are not offered.
    # The frame is threaded into the availability walks below (edd_invest_lookback_set + edd_invest_set);
    # ``None`` when no decision has a lead time (byte-parity).
    _c_period_in_use = _period_in_use_set(source, active_solve, workdir,
                                             ctx=ctx, provider=provider)
    ed_inv_db, commission_lf = _offer_invest_set(
        flex_data, source, active_solve,
        ed_inv_db if ed_inv_db is not None else getattr(flex_data, "ed_invest_set", None),
        _c_period_in_use, workdir, provider=provider)
    if ed_inv_db is not None and ed_inv_db.height > 0:
        flex_data.ed_invest_set = ed_inv_db

    offered_div = ed_divest_set_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider, existing_only=False)
    ed_div_db = offered_div
    if offered_div is not None and offered_div.height > 0:
        # Only pre-existing capacity can be retired: the retirement options of periods without it are dropped.
        ed_div_db = retirable_existing_pairs(offered_div, source, active_solve, workdir, provider=provider)
        _retirement_without_existing_warning(source, offered_div)
        _retirement_notice(offered_div, ed_div_db)
        flex_data.ed_divest_set = ed_div_db if ed_div_db.height > 0 else None
        if ed_div_db.height == 0:
            # The seeds of the retirement partitions describe the options before the filter.
            flex_data.pd_divest_set = flex_data.nd_divest_set = flex_data.ed_divest_period_set = None
    zero_cost_invest_notice(source, ed_inv_db, ed_div_db)

    # Δ.17c Gap D — pd/nd_invest_set, pd/nd_divest_set, edd_invest_set:
    # partition the (e, d) frames to (process, d) / (node, d) by
    # entity-class membership.  Mirrors the user's MathProg snippets
    # for ``e_invest_total`` / ``e_divest_total`` (already wired in
    # apply_projection_params) and ``invest_divest_sets.py:215-217``.
    #
    # We use the ed_*_set values just produced (or, when this run
    # contributes nothing, fall back to the seed-loaded ed_*_set on
    # flex_data).  The partition helpers are pure projections; they're
    # cheap so re-running on every invocation is fine.
    from flextool.engine.preprocess.params._existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
        ed_invest_period_set_lf as _ed_invest_period_lf,
        ed_divest_period_set_lf as _ed_divest_period_lf,
    )
    ed_inv_for_partition = (
        ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0)
        else getattr(flex_data, "ed_invest_set", None))
    ed_div_for_partition = (
        ed_div_db if (ed_div_db is not None and ed_div_db.height > 0)
        else getattr(flex_data, "ed_divest_set", None))
    if ed_inv_for_partition is not None and ed_inv_for_partition.height > 0:
        ed_inv_lf = ed_inv_for_partition.lazy()
        pd_inv_db = _pd_invest_lf(source, ed_inv_lf).collect()
        if pd_inv_db.height > 0:
            flex_data.pd_invest_set = pd_inv_db
        nd_inv_db = _nd_invest_lf(source, ed_inv_lf).collect()
        if nd_inv_db.height > 0:
            flex_data.nd_invest_set = nd_inv_db
        # Per-period invest cap subset — entities whose invest_method
        # carries a period cap (``maxInvest_entity_period`` index).
        ed_inv_period_db = _ed_invest_period_lf(source, ed_inv_lf).collect()
        if ed_inv_period_db.height > 0:
            flex_data.ed_invest_period_set = ed_inv_period_db
    if ed_div_for_partition is not None and ed_div_for_partition.height > 0:
        ed_div_lf = ed_div_for_partition.lazy()
        pd_div_db = _pd_divest_lf(source, ed_div_lf).collect()
        if pd_div_db.height > 0:
            flex_data.pd_divest_set = pd_div_db
        nd_div_db = _nd_divest_lf(source, ed_div_lf).collect()
        if nd_div_db.height > 0:
            flex_data.nd_divest_set = nd_div_db
        ed_div_period_db = _ed_divest_period_lf(source, ed_div_lf).collect()
        if ed_div_period_db.height > 0:
            flex_data.ed_divest_period_set = ed_div_period_db

    # Γ.6.D — ed_invest_forbidden_no_investment.  Built off the
    # (possibly-overridden) ed_invest_set so the helper sees the same
    # frame the LP downstream consumes.
    ed_invest_for_forbidden = (
        ed_inv_db if ed_inv_db is not None
        else getattr(flex_data, "ed_invest_set", None))
    forbidden_db = ed_invest_forbidden_no_investment_from_source(
        source, active_solve, workdir, ed_invest_for_forbidden,
        ctx=ctx, provider=provider)
    if forbidden_db is not None:
        flex_data.ed_invest_forbidden_no_investment = (
            forbidden_db if forbidden_db.height > 0 else None)

    _fan_branch_years_notice(source, flex_data, ed_inv_for_partition, ed_div_for_partition)
    _end_of_life_notice(source, active_solve, workdir, provider=provider)

    # Dispatch-only gate: when neither ed_invest nor ed_divest carries
    # any (e, d) row, mirror ``flex_data.py::_load_invest`` blank-out by
    # nulling every invest-cascade Param.
    #
    # Δ.18 — gate must consult the *effective* invest/divest sets (the
    # seed-loaded value persists when the override returned empty, e.g.
    # for synthetic per-sub-solve names that don't exist in Spine).  Using
    # ed_inv_db / ed_div_db alone fired the blank-out spuriously on
    # per-sub-solve snapshot fixtures, nulling p_entity_max_units etc.
    # even though the snapshot CSV had genuine invest activity.
    inv_empty = ed_inv_for_partition is None or ed_inv_for_partition.height == 0
    div_empty = ed_div_for_partition is None or ed_div_for_partition.height == 0
    if inv_empty and div_empty:
        _check_retirement_minimums(flex_data, source, active_solve, workdir, offered_div, ed_div_db, [],
                                   _c_period_in_use, provider=provider)
        for blank_field in (
            "e_invest_total", "e_divest_total",
            "e_invest_max_total", "e_divest_max_total",
            "e_invest_min_total", "e_divest_min_total",
            "p_entity_max_units",
            "ed_invest_period_set", "ed_divest_period_set",
            "ed_invest_max_period", "ed_divest_max_period",
        ):
            setattr(flex_data, blank_field, None)
        return

    # edd_invest_lookback uses the (possibly-overlaid) ed_invest_set.
    ed_inv_used = ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0) \
                   else getattr(flex_data, "ed_invest_set", None)

    try:
        eil_db = edd_invest_lookback_set_from_source(
            source, active_solve, ed_inv_used, workdir,
            provider=provider, lineage=_c_lineage,
            commission_lf=commission_lf)
    except (LineageFilterError, CommissioningLagError):
        raise
    except Exception:
        eil_db = None
    if eil_db is not None and eil_db.height > 0:
        flex_data.edd_invest_lookback_set = eil_db

    # Δ.17c Gap D — edd_invest_set: union of edd_history walks intersected
    # with ed_invest_set on (e, d_history).  Mirror of
    # ``invest_divest_sets.py:267-270``.
    if ed_inv_used is not None and ed_inv_used.height > 0:
        from flextool.engine.preprocess.params._existing import (
            edd_invest_set_lf as _edd_invest_lf,
        )
        period_in_use = _c_period_in_use
        period_with_history = (_read_period_with_history(workdir, provider=provider)
                                  or list(period_in_use))
        # A failure here propagates: a missing edd_invest_set would drop every investment from the capacity.
        edd_inv_db = _edd_invest_lf(
            source, active_solve, ed_inv_used.lazy(),
            period_with_history, period_in_use, workdir,
            lineage=_c_lineage, commission_lf=commission_lf).collect()
        if edd_inv_db is not None and edd_inv_db.height > 0:
            flex_data.edd_invest_set = edd_inv_db
    # Construction lead time: the period each decision starts to operate in and the periods it is in construction.
    _apply_lead_time_sets(flex_data, source, active_solve, ed_inv_used, commission_lf, _c_period_in_use, workdir,
                          lineage=_c_lineage)

    pd_div_used = getattr(flex_data, "pd_divest_set", None)
    # A failure here (or in the node call below) propagates: a missing active-divest set would leave the
    # retirements out of the capacity.
    edda_db = edd_divest_active_from_source(
        source, active_solve, pd_div_used,
        provider=provider, lineage=_c_lineage, workdir=workdir, ctx=ctx)
    if edda_db is not None and edda_db.height > 0:
        flex_data.edd_divest_active = edda_db

    nd_div_used = getattr(flex_data, "nd_divest_set", None)
    ndda_db = edd_divest_active_from_source(
        source, active_solve, nd_div_used,
        entity_col="n", provider=provider, lineage=_c_lineage, workdir=workdir, ctx=ctx)
    if ndda_db is not None and ndda_db.height > 0:
        flex_data.ndd_divest_active = ndda_db
    _check_retirement_minimums(flex_data, source, active_solve, workdir, offered_div, ed_div_db,
                               [edda_db, ndda_db], _c_period_in_use, provider=provider)

    # p_entity_max_units uses ed_invest plus the per-period-in-use grid
    # and the unitsize cascade.
    try:
        pae_for_max_units = getattr(flex_data, "p_entity_all_existing", None)
        pemu_db = p_entity_max_units_from_source(
            source, ed_inv_used, active_solve, workdir=workdir,
            p_entity_all_existing=pae_for_max_units, ctx=ctx,
            commission=getattr(flex_data, "ed_invest_commission", None))
    except FlexToolConfigError:
        # A shape error from the resolver: keeping the legacy CSV seed would hide it
        # behind a wrong p_entity_max_units (the seed misses recourse branch periods).
        raise
    except Exception:
        pemu_db = None
    if pemu_db is not None:
        flex_data.p_entity_max_units = pemu_db




# =============================================================================
# Γ.3.D — final derived-Param batch.  Narrow, conservative scope:
#
#   * §3.11 existing-entity fixed cost — ``p_entity_all_existing``
#     (sum of pre_existing + previously_invested per (e, d)).
#   * §3.13 reserve cleanup — narrow gate-on-equality overlay for the
#     `_increase_reserve_ratio` / `_large_failure_ratio` Projection sets
#     when the underlying SpineDB reserve relationships are present.
#
# Storage / lifetime cascade / handoff state / ladder / delay / multi-
# branch Params are deferred to Γ.3.E; their algorithms either depend
# on per-solve state not in Spine, or require a fully-faithful port of
# flextool preprocessing's coarse-block synthesis (flex_data.py:1985-2126)
# whose risk profile exceeds the time budget for this batch.  See
# ``progress.md`` for the deferred-list bookkeeping.
# =============================================================================


def p_entity_all_existing_from_source(source: "InputSource",
                                        active_solve: str | None,
                                        workdir: Path | None = None,
                                        *,
                                        ctx: "SolveContext | None" = None,
                                        provider: "object | None" = None,
                                        ) -> "Param | None":
    """§3.11 — sum of pre-existing capacity per (entity, period).

    First-solve (or single-solve) collapses to:

        p_entity_all_existing[e, d] = pre_existing[e, d]

    where ``pre_existing[e, d]`` is the lifetime-gated existing
    capacity (:func:`._existing.p_entity_pre_existing_lf`), with
    capacity[e, d] = existing[e, d] × virtual_unitsize[e] when
    virtual_unitsize is above 0, else existing[e, d]:

      * ``reinvest_automatic`` → capacity[e, d].
      * ``reinvest_choice`` / ``no_investment`` → capacity[e, d]
        only while ``p_years_d[d] < life_sum(e)``; 0 thereafter.
      * Any other ``lifetime_method`` → 0.

    Chain-run handoff (Γ.6.D): when ``solve_data/p_entity_period_existing_capacity.csv``
    is present in ``workdir`` it carries the chained existing capacity
    summed across the entity's history (see
    ``entity_period_calc_params.py:1463-1543``).  We prefer that CSV
    when it has explicit non-zero rows for ``period_in_use`` periods
    that aren't covered by ``entity.existing`` — captures the
    multi-solve state passed via ``apply_handoff``.
    """
    piu = _period_in_use_set(source, active_solve, workdir, ctx=ctx, provider=provider)
    if not piu:
        return None

    # Γ.6.D — chain-run handoff: prefer the canonical per-solve
    # ``solve_data/p_entity_all_existing.csv`` when present.  This CSV
    # carries the lifetime-cumulative + previously-invested chain
    # already integrated by the chain runner / flextool's preprocessing
    # for sub-solves N>1 (see
    # ``entity_period_calc_params.write_p_entity_existing_chain``).
    # When the CSV is absent (single-solve fixtures, or DB-direct sweep
    # tooling without preprocessing), fall back to deriving from
    # ``entity.existing`` + lifetime gate.
    #
    # Path B Cat B: when ``ctx`` is provided, the in-memory
    # :pyattr:`SolveContext.p_entity_all_existing` frame is consulted
    # first.  Falls back to the CSV-read path when ``ctx`` is None /
    # the typed accessor is empty.
    if ctx is not None:
        pae_ctx = ctx.p_entity_all_existing
        if pae_ctx.height > 0 \
                and "entity" in pae_ctx.columns \
                and "period" in pae_ctx.columns \
                and "value" in pae_ctx.columns:
            df2 = (pae_ctx.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                          .with_columns(value=pl.col("value")
                                                  .cast(pl.Float64,
                                                          strict=False)
                                                  .fill_null(0.0))
                          .select("e", "d", "value")
                          .sort("e", "d"))
            if df2.height > 0:
                return Param(("e", "d"), df2)
    elif workdir is not None:
        pae_path = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
        if _provider_has_key(provider, pae_path):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, pae_path)
                if df.height > 0 and "entity" in df.columns \
                        and "period" in df.columns and "value" in df.columns:
                    df2 = (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                              .with_columns(value=pl.col("value")
                                                       .cast(pl.Float64,
                                                                strict=False)
                                                       .fill_null(0.0))
                              .select("e", "d", "value")
                              .sort("e", "d"))
                    if df2.height > 0:
                        return Param(("e", "d"), df2)
            except Exception:
                pass
    # No chained frame: the lifetime-gated existing capacity of every unit / node / connection
    # (p_entity_pre_existing: existing × virtual_unitsize when it is set, else existing; 0 without existing).
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    out = (p_entity_pre_existing_lf(source, active_solve, piu, workdir=workdir, provider=provider)
             .sort("e", "d")
             .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


# ---------------------------------------------------------------------------
# §3.13 reserve helpers (narrow gate-on-equality overlays)
# ---------------------------------------------------------------------------


def _reserve_active_relationships(source: "InputSource"
                                       ) -> pl.DataFrame | None:
    """Union the unit-side and connection-side reserve relationships
    into a single ``(p, r, ud, n)`` frame.  Returns None when no reserve
    relationships exist on the source.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        ents = _try_entities(source, ec)
        if ents is None or ents.height == 0:
            continue
        cols = ents.columns
        # Schema: [reserve, upDown, unit_or_connection, node]
        # Column names from SpineDbReader use the dim class names.
        rename = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        if "r" not in [rename.get(c, c) for c in cols] \
                or "ud" not in [rename.get(c, c) for c in cols]:
            continue
        parts.append(ents.lazy().pipe(rename_to_axis, rename).select("p", "r", "ud", "n"))
    if not parts:
        return None
    return pl.concat(parts).unique().sort("p", "r", "ud", "n").collect()


def process_reserve_upDown_node_active_from_source(source: "InputSource",
                                                       ) -> pl.DataFrame | None:
    """§3.13.x — Projection of (p, r, ud, n) tuples that are active.

    The simple case is the union of the unit-side / connection-side
    reserve relationships filtered to those with non-zero
    ``reliability``.  When ``reliability`` is missing on the source,
    fall back to the relationship membership (default reliability=1.0
    per the Spine schema).
    """
    # Build a typed empty frame for the "reserve feature active but no
    # active (p, r, ud, n) tuples" fall-through.  ``_reserve._check``
    # requires a DataFrame (not None) when ``reserve_upDown_group`` is
    # populated; downstream uses guard on ``height > 0`` before joining.
    def _empty_pruna() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "p": pl.Utf8, "r": pl.Utf8, "ud": pl.Utf8, "n": pl.Utf8,
        })

    rel = _reserve_active_relationships(source)
    if rel is None or rel.height == 0:
        return _empty_pruna()
    # If reliability is parametrised, keep rows with reliability > 0.
    rel_dfs: list[pl.LazyFrame] = []
    for ec in ("reserve__upDown__unit__node",
               "reserve__upDown__connection__node"):
        df = _try_param(source, ec, "reliability")
        if df is None or df.height == 0:
            continue
        cols = df.columns
        rename: dict[str, str] = {}
        for c in cols:
            if c == "reserve":
                rename[c] = "r"
            elif c == "upDown":
                rename[c] = "ud"
            elif c == "node":
                rename[c] = "n"
            elif c in ("unit", "connection"):
                rename[c] = "p"
        rel_dfs.append(df.lazy().pipe(rename_to_axis, rename).select(
            "p", "r", "ud", "n",
            pl.col("value").cast(pl.Float64).alias("rel"),
        ))
    if not rel_dfs:
        return rel  # Defaults to "all relationships active".
    rel_lf = (pl.concat(rel_dfs)
                  .filter(pl.col("rel") > 0)
                  .select("p", "r", "ud", "n"))
    out = rel_lf.unique().sort("p", "r", "ud", "n").collect()
    return out if out.height > 0 else _empty_pruna()


# ---------------------------------------------------------------------------
# Public — Γ.3.D field list (for selective scope-checking).
# ---------------------------------------------------------------------------


D_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_entity_all_existing",
    "process_reserve_upDown_node_active",
)


def apply_derived_d(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.D Derived Params, mutating ``flex_data`` in place.

    Scope (Tier-1 selected items):
      * §3.11 ``p_entity_all_existing`` (simple existing-only path).
      * §3.13 ``process_reserve_upDown_node_active`` (reliability>0 set).

    Δ.12b — assignment is unconditional; helpers are authoritative
    producers.  ``None`` is the explicit "feature inactive" signal
    (no reserve relationships /
    no chained existing capacity).  Hard exceptions propagate.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)

    # ─── §3.11 p_entity_all_existing — Δ.12b: unconditional ──────────
    flex_data.p_entity_all_existing = p_entity_all_existing_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)

    # Δ.12c — ``apply_existing_chain`` was previously invoked here, but
    # it consumes the handoff carriers (``p_entity_previously_invested_capacity``
    # / ``p_entity_divested``) which are populated by ``apply_derived_f``.
    # Calling it here meant the chain-summation read stale (or seed-only)
    # carriers on the first iteration.  Moved to ``_apply_db_overrides``
    # so it runs strictly after ``apply_derived_f``.

    # ─── §3.13 process_reserve_upDown_node_active — Δ.12b: unconditional
    flex_data.process_reserve_upDown_node_active = (
        process_reserve_upDown_node_active_from_source(source))

    # ─── §3.3.5 p_flow_upper (Δ.26) ───────────────────────────────────
    # Per-(p, source, sink, d, t) structural max flow.  Native port of
    # ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
    # `process_arc_unions.py:1469-1624`).  Wired in apply_derived_d so
    # ``p_entity_all_existing`` (set immediately above) is available;
    # ``p_entity_max_units`` from apply_derived_c is also picked up
    # when the fixture has invest activity.
    pss_for_upper = getattr(flex_data, "process_source_sink", None)
    dt_for_upper = getattr(flex_data, "dt", None)
    if (pss_for_upper is not None and pss_for_upper.height > 0
            and dt_for_upper is not None and dt_for_upper.height > 0):
        slope_for_upper = getattr(flex_data, "p_slope", None)
        section_for_upper = getattr(flex_data, "p_section", None)
        unitsize_for_upper = getattr(flex_data, "p_unitsize", None)
        max_units_for_upper = getattr(flex_data, "p_entity_max_units", None)
        existing_for_upper = getattr(flex_data, "p_entity_all_existing", None)
        pfu_db = p_flow_upper_from_source(
            source,
            pss_for_upper,
            dt_for_upper,
            p_slope=slope_for_upper,
            p_section=section_for_upper,
            p_unitsize=unitsize_for_upper,
            p_entity_max_units=max_units_for_upper,
            p_entity_all_existing=existing_for_upper,
        )
        if pfu_db is not None:
            flex_data.p_flow_upper = pfu_db
        # Input-arc limit of indirect units per unit of capacity
        # (``input_share_max`` × full-load fuel); model/core.py requires it
        # whenever indirect input arcs exist.
        flex_data.p_indirect_input_cap = p_indirect_input_cap_from_source(
            source, pss_for_upper, dt_for_upper,
            p_slope=slope_for_upper, p_section=section_for_upper)




# ===========================================================================
# Γ.3.E — Storage block algebra (audit §3.9)
#
# Eight sub-sections covering the multi-resolution timeline algebra used by
# storage state, intra-period blocks, arc weighting, and rolling-handoff.
# No defensive gating — these helpers either produce the canonical frame or
# the parity test fails loudly.  See task spec for the architectural shift.
# ===========================================================================


def _expand_branch_periods(period_order: list[str],
                              workdir: Path | None,
                              *,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None,
                              ) -> list[str]:
    """Given a period_order coming from solve.realized_periods +
    period_timeset, add stochastic-branch periods read from
    ``solve_data/period__branch.csv`` (anchor → branch map), filtered
    to those that actually appear in the canonical ``period_in_use_set``
    or ``steps_in_use.csv`` (excludes realized-output-only branches).

    For each anchor in period_order, append all unique in-use branches
    whose anchor is the anchor — preserving anchor's relative position.

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` and
    :pyattr:`SolveContext.period_in_use` frames are preferred over the
    workdir CSVs.  Falls back to reading the CSVs when ``ctx`` is None
    or the typed accessors are empty (and the workdir path exists).
    """
    # Try ctx-side first: canonical in-memory carriers.
    pb_df: pl.DataFrame | None = None
    in_use: set[str] = set()
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            # Canonical schema is [d_anchor, b]; remap to legacy ('period', 'branch')
            # only for the local iteration below.
            pb_df = pb_ctx.rename({"d_anchor": "anchor", "b": "br"})
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
    if pb_df is None:
        if workdir is None:
            return list(period_order)
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
            return list(period_order)
        try:
            df = _provider_read(provider, p)
        except Exception:
            return list(period_order)
        if df.height == 0:
            return list(period_order)
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu = _provider_read(provider, piu_path)
                    if piu.height > 0:
                        in_use = set(piu["period"].to_list())
                except Exception:
                    in_use = set()
        pb_df = df.rename({"period": "anchor", "branch": "br"})
    anchor_to_branches: dict[str, list[str]] = {}
    for row in pb_df.iter_rows(named=True):
        anchor_to_branches.setdefault(row["anchor"], []).append(row["br"])
    out: list[str] = []
    seen: set[str] = set()
    for d in period_order:
        if d not in seen:
            out.append(d)
            seen.add(d)
        for br in anchor_to_branches.get(d, []):
            if br in seen:
                continue
            if in_use and br not in in_use:
                continue
            out.append(br)
            seen.add(br)
    return out


def _expand_invest_branch_periods(period_invest: "list[str] | None",
                                     workdir: Path | None,
                                     *,
                                     ctx: "SolveContext | None" = None,
                                     provider: "object | None" = None,
                                     ) -> "list[str] | None":
    """Slice D (A, §3.1) — invest-side mirror of
    :func:`_expand_branch_periods`.

    When the recourse flag is active, append each real invest period's
    synthetic branch members (filtered to ``period_in_use``) after the
    anchor, so ``period_invest`` carries ``{p_k, p_k_b, …}`` — the axis
    on which ``pd/nd_invest_set`` (and thus ``v_invest``) get their
    columns.  Flag-off (or empty axis) → returns ``period_invest``
    unchanged (byte-parity).  Reuses — does not fork — the dispatch-side
    helper so invest and dispatch axes stay consistent.

    Under recourse a period label (the original name of a period that
    opens a fan, :func:`._branch.fan_period_labels`) has no investment
    column: every forecast branch invests on its own after the reveal,
    so nothing is invested here-and-now at the label.
    """
    if not period_invest:
        return period_invest
    if not _recourse_invest_active(workdir, ctx=ctx, provider=provider):
        return period_invest
    expanded = _expand_branch_periods(list(period_invest), workdir,
                                      ctx=ctx, provider=provider)
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb["d"].to_list(), pb["b"].to_list())
    return [d for d in expanded if str(d) not in labels]


def _dt_period_active_steps_from_workdir(
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> dict | None:
    """Δ.12c-fix2 gap #5 — preprocessing-only-timeline fallback.

    Build ``per_period`` / ``period_order`` directly from the workdir's
    preprocessing artefacts (``solve_data/steps_in_use.csv`` +
    ``solve_data/period_in_use_set.csv``) when the source's
    ``solve.period_timeset`` filter for the active solve is empty.

    This handles **rolling-horizon sub-solves** and **nested-invest
    sub-solves** whose synthetic solve names (e.g.
    ``dispatch_fullYear_roll_roll_71``) aren't in the Spine DB — only
    the parent solve is.  flextool's preprocessing pipeline writes
    ``steps_in_use.csv`` per sub-solve; we leverage that CSV plus the
    Spine ``timeline.timestep_duration`` for the rank/timeline lookup.

    Returns ``None`` when the workdir doesn't have the required
    preprocessing artefacts (caller stays on the seed-CSV path).

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.steps_in_use` and
    :pyattr:`SolveContext.period_in_use` frames are consulted first.
    """
    siu: pl.DataFrame | None = None
    if ctx is not None:
        siu_ctx = ctx.steps_in_use
        if siu_ctx.height > 0:
            # Canonical schema [d, t, step_duration]; select the (d, t)
            # subset to match the legacy local frame.
            siu = siu_ctx.select("d", "t").unique()
    if siu is None:
        sd = workdir / "solve_data"
        siu_path = sd / "steps_in_use.csv"
        if not _provider_has_key(provider, siu_path):  # Phase E-j — seed-aware
            return None
        try:
            siu_raw = _provider_read(provider, siu_path)
        except Exception:
            return None
        if siu_raw.height == 0:
            return None
        period_col = next((c for c in ("period", "d") if c in siu_raw.columns),
                            None)
        step_col = next((c for c in ("step", "t", "time")
                           if c in siu_raw.columns), None)
        if period_col is None or step_col is None:
            return None
        siu = siu_raw.select(alias_to_axis(period_col, "d"),
                                alias_to_axis(step_col, "t")).unique()
    if siu.height == 0:
        return None
    sd = workdir / "solve_data"
    # Period order: prefer per_solve_sets.py's ``period_in_use_set.csv``
    # (canonical insertion order); fall back to insertion order in
    # steps_in_use.csv.
    period_order: list[str] = []
    seen: set[str] = set()
    piu_ctx_df = ctx.period_in_use if ctx is not None else None
    if piu_ctx_df is not None and piu_ctx_df.height > 0:
        for d in piu_ctx_df["d"].to_list():
            if d and d not in seen:
                period_order.append(d)
                seen.add(d)
    else:
        piu_path = sd / "period_in_use_set.csv"
        if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
            try:
                piu_df = _provider_read(provider, piu_path)
                if piu_df.height > 0 and "period" in piu_df.columns:
                    for d in piu_df["period"].to_list():
                        if d and d not in seen:
                            period_order.append(d)
                            seen.add(d)
            except Exception:
                pass
    for d in siu["d"].to_list():
        if d and d not in seen:
            period_order.append(d)
            seen.add(d)
    # A period label is in period_in_use_set for the period-level bookkeeping but has no active steps.
    pb_wd = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb_wd["d"].to_list(), pb_wd["b"].to_list())
    period_order = [d for d in period_order if str(d) not in labels]
    if not period_order:
        return None
    # Timeline lookup: per period, the timeline is whichever timeline
    # contains the period's first step (per the Spine
    # ``timeline.timestep_duration`` rank).  When multiple timelines
    # match (rare; only for ambiguous step naming) pick the first.
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_dur.columns
                          and c not in ("name", "value")),
                         None)
    if tl_step_col is None:
        return None
    tl_lf = (tl_dur.lazy()
                    .select(pl.col("name").alias("timeline"),
                            alias_to_axis(tl_step_col, "t"),
                            pl.col("value").cast(pl.Float64)
                                            .alias("step_duration"))
                    .sort("timeline", "t")
                    .with_columns(rank=pl.col("t").cum_count()
                                                  .over("timeline")
                                                  .cast(pl.Int64)))
    # Pick a single timeline per period via the first step of that period.
    # Mirrors flextool's preprocessing: a period maps to exactly one
    # timeset, which maps to exactly one timeline.  When the timeline
    # is ambiguous (multiple timelines contain the same step name —
    # shouldn't happen with t0001..t8760 naming), prefer the timeline
    # whose first matching step has the smallest rank (i.e. earliest
    # step in the timeline).
    siu_e = siu.lazy().join(tl_lf, on="t", how="inner").collect()
    if siu_e.height == 0:
        return None
    timeline_per_period: dict[str, str] = {}
    per_period: dict[str, list[tuple[str, int]]] = {}
    block_starts_per_period: dict[str, list[str]] = {}
    for d in period_order:
        rows = siu_e.filter(pl.col("d") == d)
        if rows.height == 0:
            per_period[d] = []
            block_starts_per_period[d] = []
            continue
        # When multiple timelines match, pick the timeline with the
        # smallest sum-of-ranks for this period (== smallest first step).
        if rows["timeline"].n_unique() > 1:
            best_tl = (rows.group_by("timeline")
                           .agg(pl.col("rank").min().alias("min_rk"))
                           .sort("min_rk")
                           .select("timeline").to_series()[0])
            rows = rows.filter(pl.col("timeline") == best_tl)
        rows_sorted = rows.sort("rank")
        timeline_per_period[d] = rows_sorted["timeline"][0]
        per_period[d] = list(zip(rows_sorted["t"].to_list(),
                                    rows_sorted["rank"].to_list()))
    step_space = _to_solve_step_space(per_period, timeline_per_period, workdir, provider)
    for d, steps in per_period.items():
        if not steps:
            continue
        # Block starts: indices where the rank gap to the previous step
        # is >1.  Mirrors ``make_period_block``'s detection rule.
        block_firsts: list[str] = [steps[0][0]]
        for j in range(1, len(steps)):
            if steps[j][1] - steps[j - 1][1] > 1:
                block_firsts.append(steps[j][0])
        block_starts_per_period[d] = block_firsts
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
        step_space=step_space,
    )


def _solve_step_positions(timeline_for_period: dict[str, str], workdir: "Path | None",
                          provider: "object | None") -> "dict[tuple[str, str], int] | None":
    """``{(timeline, t): 1-based position}`` of every step in the solve's own timeline of each original timeline.

    Read from ``solve_data/steps_in_timeline`` (every step of each period's timeline, in timeline order; under a
    solve-level ``new_stepduration`` the aggregated timeline, whose gap chunks keep the gaps between timeset
    blocks).  The longest list per timeline wins (stochastic branch rows list only their active steps).  ``None``
    when the frame is absent."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "steps_in_timeline.csv"
    if not _provider_has_key(provider, path):
        return None
    try:
        df = _provider_read(provider, path)
    except Exception:
        return None
    if df.height == 0 or len(df.columns) < 2:
        return None
    steps_of: dict[str, list[str]] = {}
    for d, t in df.select(pl.col(df.columns[0]).cast(pl.Utf8), pl.col(df.columns[1]).cast(pl.Utf8)).iter_rows():
        steps_of.setdefault(d, []).append(t)
    best: dict[str, list[str]] = {}
    for d, steps in steps_of.items():
        tl = timeline_for_period.get(d)
        if tl is not None and len(steps) > len(best.get(tl, ())):
            best[tl] = steps
    return {(tl, t): i + 1 for tl, steps in best.items() for i, t in enumerate(steps)}


def _to_solve_step_space(per_period: dict[str, list[tuple[str, int]]], timeline_for_period: dict[str, str],
                         workdir: "Path | None", provider: "object | None") -> bool:
    """Re-rank ``per_period`` in place by position in the solve's own timeline (:func:`_solve_step_positions`).

    The rank difference of two steps is then ``make_step_jump``'s ``jump``: 1 exactly for steps adjacent on the
    timeline, also for aggregated ``new_stepduration`` steps, and > 1 across a gap.  Without the solve's timeline
    (or when a step is missing from it) the raw timeline ranks stay; returns whether the re-rank was applied."""
    positions = _solve_step_positions(timeline_for_period, workdir, provider)
    if not positions:
        return False
    reranked: dict[str, list[tuple[str, int]]] = {}
    for d, steps in per_period.items():
        tl = timeline_for_period.get(d)
        new = [(t, positions.get((tl, str(t)))) for t, _rk in steps]
        if any(rk is None for _t, rk in new):
            return False
        reranked[d] = sorted(new, key=lambda r: r[1])
    per_period.update(reranked)
    return True


def _dt_period_active_steps(source: "InputSource",
                              active_solve: str,
                              workdir: Path | None = None,
                              *,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None,
                              ) -> dict | None:
    """Build the per-period active-step structure used by every storage
    block-algebra helper.

    Returns a dict ``{"per_period": {d -> [(t, global_rank), ...]},
    "timeline_for_period": {d -> timeline_name},
    "period_order": [d in dispatch order],
    "block_starts_per_period": {d -> [t_block_first ordered by rank]},
    "step_space": bool}``, where ``global_rank`` is the step's position in
    the solve's own timeline (:func:`_to_solve_step_space`; ``step_space``
    True), else the raw timeline rank (cum_count over the timeline).  The
    block-first list is derived from
    ``timeset.timeset_duration`` block start_steps cross-joined with
    ``period_timeset`` for the active solve.

    Returns ``None`` when the active solve has no realized periods or any
    of the required Spine classes are absent.

    Δ.12c-fix2 gap #5 — when the source-side path returns no rows for the
    active solve (e.g. rolling-horizon / nested-invest synthetic solves
    whose names aren't in Spine), falls back to
    :func:`_dt_period_active_steps_from_workdir` which reads
    ``steps_in_use.csv`` directly.  The fallback uses the Spine
    ``timeline.timestep_duration`` for rank/timeline assignment.
    """
    # Prefer the workdir fallback when ``steps_in_use.csv`` (or
    # ``ctx.steps_in_use``) is available: that frame reflects the
    # post-``new_stepduration`` reality (e.g. invest_24h's 3 timesteps/
    # period after 24h aggregation), whereas the source path expands the
    # timeset over the raw hourly timeline.  When ``new_stepduration`` is
    # set, the source path would otherwise return the wrong (hourly)
    # active-step list, breaking storage state-coupling rows in the LP.
    steps_in_use_available = False
    if ctx is not None:
        try:
            steps_in_use_available = ctx.steps_in_use.height > 0
        except Exception:
            steps_in_use_available = False
    if not steps_in_use_available and workdir is not None:
        steps_in_use_available = (
            _provider_has_key(provider,   # Phase E-j — seed-aware
                Path(workdir) / "solve_data" / "steps_in_use.csv")
        )
    if steps_in_use_available and workdir is not None:
        wd_result = _dt_period_active_steps_from_workdir(
            source, workdir, ctx=ctx, provider=provider)
        if wd_result is not None:
            return wd_result

    # period → timeset of the active solve: ``solve.period_timeset``, or the
    # Rule 1 / 5b default (item #175).
    pt_frame = solve_period_timeset_frame(source, active_solve)
    if pt_frame is None:
        # No mapping for the active solve (synthetic rolling-horizon /
        # nested-invest sub-solve): preprocessing-only fallback.
        if workdir is not None:
            return _dt_period_active_steps_from_workdir(
                source, workdir, ctx=ctx, provider=provider)
        return None
    pt_eager_full = pt_frame.pipe(rename_to_axis, {"d": "d"})
    # Anchor → timeset map for the active solve.
    anchor_to_ts: dict[str, str] = dict(
        zip(pt_eager_full["d"].to_list(),
              pt_eager_full["ts"].to_list()))
    # Expand to include branch periods (workdir-aware — period__branch
    # is preprocessing-only data, no Spine class in v25).  Filter on
    # period_in_use_set.csv when present to exclude realized-output-only
    # branches (e.g. period1_realized).
    in_use: set[str] = set()
    pbf: pl.DataFrame | None = None
    # Path B Cat B: prefer typed ctx accessors over CSV reads.
    if ctx is not None:
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            # Match the legacy iteration shape via a local rename.
            pbf = pb_ctx.pipe(rename_to_axis, {"d_anchor": "period", "b": "branch"})
    if workdir is not None:
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu_df = _provider_read(provider, piu_path)
                    if piu_df.height > 0:
                        in_use = set(piu_df["period"].to_list())
                except Exception:
                    in_use = set()
        if pbf is None:
            pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
            if _provider_has_key(provider, pb_path):  # Phase E-j — seed-aware
                try:
                    pbf = _provider_read(provider, pb_path)
                except Exception:
                    pbf = None
    if pbf is not None and pbf.height > 0:
        for row in pbf.iter_rows(named=True):
            anchor = row["period"]
            branch = row["branch"]
            if in_use and branch not in in_use:
                continue
            if (anchor in anchor_to_ts
                    and branch not in anchor_to_ts):
                anchor_to_ts[branch] = anchor_to_ts[anchor]
        # A period label has no dispatch: it has no active steps.
        for label in _labels_of_pairs(pbf["period"].to_list(), pbf["branch"].to_list()):
            anchor_to_ts.pop(label, None)
    if not anchor_to_ts:
        return None
    # Use axis_lazyframe so ``d`` carries the global Enum vocabulary
    # when active — otherwise the downstream join against
    # ``realized_clean`` (which IS axis_lazyframe-built) trips on
    # ``SchemaError: d: enum on left does not match d: str on right``.
    pt_lf = axis_lazyframe({
        "d": list(anchor_to_ts.keys()),
        "ts": list(anchor_to_ts.values()),
    })
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    tl_step_col = next((c for c in ("t", "step", "timestep", "x")
                          if c in tl_dur.columns and c not in ("name", "value")),
                         None)
    if tl_step_col is None:
        return None
    tl_lf = (tl_dur.lazy()
                    .select(pl.col("name").alias("timeline"),
                            alias_to_axis(tl_step_col, "t"),
                            pl.col("value").cast(pl.Float64)
                                            .alias("step_duration"))
                    .sort("timeline", "t")
                    .with_columns(rank=pl.col("t").cum_count()
                                                  .over("timeline")
                                                  .cast(pl.Int64)))

    # ``timeset.timeline`` / ``timeset.timeset_duration``, through the
    # shared single-timeline defaults (item #171, Rules 2/3 — see
    # ``_per_solve_sets.timeset_timeline_lf`` / ``timeset_duration_lf``).
    timesets_lf = pt_lf.select("ts").unique()
    ttl_lf = timeset_timeline_lf(source, timesets_lf, tl_lf)
    if ttl_lf is None:
        return None

    ts_dur_lf = timeset_duration_lf(source, timesets_lf, tl_lf)
    if ts_dur_lf is None:
        return None
    # ``start_step`` lives on the ``t`` axis; cast to the canonical Enum
    # so the downstream join against ``tl_lf`` (whose ``t``/``start_step``
    # is cast to Enum at line 6204+) doesn't trip on Utf8-vs-Enum.  The
    # shared helper returns Utf8/Float64; this caller casts both itself.
    blocks_lf = ts_dur_lf.select(
        pl.col("ts"),
        cast_dim(pl.col("start_step"), None, "t").alias("start_step"),
        pl.col("count").cast(pl.Int64),
    )
    # Period order: prefer realized_periods.i ordering when present, else
    # fall back to period_timeset's natural order.  ``period_in_use_set``
    # lists ALL solve periods (realized + non-realized + branch siblings).
    realized = _try_param(source, "solve", "realized_periods")
    explicit_order: list[str] = []
    if realized is not None:
        # ``realized_periods`` is Array-shaped; the index column name is
        # the Spine ``index_name`` when set, else ``_array_index`` (see
        # ``_spinedb_reader._unroll_index_columns_from_value`` — Array
        # types use ``_array_index`` to avoid collision with the tier
        # axis ``i``).  Use the second-to-last column (between ``name``
        # and ``value``) rather than hardcoding ``"i"`` — same shape-
        # tolerant rule as ``_solve_periods``.
        order_col = (realized.columns[-2]
                       if len(realized.columns) >= 2 else "i")
        realized_p = (realized
                       .filter(pl.col("name") == active_solve)
                       .sort(order_col))
        explicit_order = realized_p["value"].to_list()
    pt_eager = pt_lf.select("d").unique().collect()
    pt_periods = pt_eager["d"].to_list()
    seen_o: set[str] = set(explicit_order)
    period_order: list[str] = list(explicit_order)
    for d in pt_periods:
        if d not in seen_o:
            period_order.append(d)
            seen_o.add(d)
    # Workdir-aware branch expansion + reorder by period_in_use_set
    # insertion order (canonical flextool active_time_list ordering).
    if workdir is not None or ctx is not None:
        period_order = _expand_branch_periods(period_order, workdir, ctx=ctx, provider=provider)
        # Path B Cat B: prefer ctx.period_in_use over CSV read.
        canonical: list[str] | None = None
        if ctx is not None:
            piu_ctx = ctx.period_in_use
            if piu_ctx.height > 0:
                canonical = piu_ctx["d"].to_list()
        if canonical is None and workdir is not None:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu_df = _provider_read(provider, piu_path)
                    if piu_df.height > 0:
                        canonical = piu_df["period"].to_list()
                except Exception:
                    canonical = None
        if canonical is not None:
            # Re-order period_order to match canonical, keeping
            # only periods present in BOTH (and any extra
            # period_order entries appended at the end for
            # robustness — doesn't matter when canonical covers).
            in_canonical = [p for p in canonical
                               if p in seen_o
                               or p in set(period_order)]
            extras = [p for p in period_order
                        if p not in set(in_canonical)]
            period_order = in_canonical + extras
    if not period_order:
        return None
    realized_clean = axis_lazyframe({"d": period_order})
    # Tie everything together to find each (d, t, rank).
    realized_with_ts = realized_clean.join(pt_lf, on="d", how="inner")
    pst = (realized_with_ts
            .join(ttl_lf, on="ts", how="inner")
            .join(blocks_lf, on="ts", how="inner"))
    starts = pst.join(
        tl_lf.select("timeline",
                       pl.col("t").alias("start_step"),
                       pl.col("rank").alias("start_rank")),
        on=["timeline", "start_step"], how="inner",
    )
    # Expand: for each (d, ts, timeline, start_step, count), emit each
    # rank in [start_rank, start_rank + count - 1].
    expanded = (starts
                  .join(tl_lf, on="timeline", how="inner")
                  .filter((pl.col("rank") >= pl.col("start_rank"))
                          & (pl.col("rank") < pl.col("start_rank")
                              + pl.col("count")))
                  .select("d", "t", "rank", "timeline", "start_step")
                  .unique())
    expanded_e = expanded.collect()
    if expanded_e.height == 0:
        return None
    timeline_per_period: dict[str, str] = {}
    per_period: dict[str, list[tuple[str, int]]] = {}
    block_starts_per_period: dict[str, list[str]] = {}
    # Group by period; within period, sort by rank to get (t, rank) order.
    for d in period_order:
        rows = expanded_e.filter(pl.col("d") == d)
        if rows.height == 0:
            per_period[d] = []
            block_starts_per_period[d] = []
            continue
        rows_sorted = rows.sort("rank")
        per_period[d] = list(zip(rows_sorted["t"].to_list(),
                                    rows_sorted["rank"].to_list()))
        timelines = rows_sorted["timeline"].unique().to_list()
        # Single timeline per period under the single-timeset-per-(solve,
        # period) rule of period_timeset.  Pick the first.
        timeline_per_period[d] = timelines[0]
        # Block starts: distinct start_step values, sorted by their rank
        # in the timeline (== rank of the first-occurrence row).
        first_occ = (rows_sorted
                       .group_by("start_step")
                       .agg(pl.col("rank").min().alias("rk"))
                       .sort("rk"))
        block_starts_per_period[d] = first_occ["start_step"].to_list()
    step_space = _to_solve_step_space(per_period, timeline_per_period, workdir, provider)
    return dict(
        per_period=per_period,
        timeline_for_period=timeline_per_period,
        period_order=period_order,
        block_starts_per_period=block_starts_per_period,
        step_space=step_space,
    )


# ---------------------------------------------------------------------------
# §3.9.1 — dtttdt
# ---------------------------------------------------------------------------


def _roll_first_step_link(workdir: "Path | None", provider: "object | None") -> "pl.DataFrame | None":
    """``solve_data/roll_first_step_link`` (the complete solve's ``step_previous`` of each period's first step);
    ``None`` when absent."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "roll_first_step_link.csv"
    if not _provider_has_key(provider, path):
        return None
    return _provider_read(provider, path)


def _rp_block_first_utf8(workdir: "Path | None", provider: "object | None") -> "pl.DataFrame | None":
    """``solve_data/rp_block_first`` as Utf8 ``(d, t)``; ``None`` when absent or empty."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "rp_block_first.csv"
    if not _provider_has_key(provider, path):
        return None
    try:
        df = _provider_read(provider, path)
    except Exception:
        return None
    if df.height == 0:
        return None
    return df.select(pl.col("period").cast(pl.Utf8).alias("d"), pl.col("step").cast(pl.Utf8).alias("t"))


def dtttdt_from_source(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          *,
                          ctx: "SolveContext | None" = None,
                          provider: "object | None" = None,
                          ) -> pl.DataFrame | None:
    """The ``dtttdt`` lag frame of :func:`dtttdt_with_jump_from_source` without its ``jump`` column."""
    df = dtttdt_with_jump_from_source(source, active_solve, workdir, ctx=ctx, provider=provider)
    return None if df is None else df.drop("jump")


def _timeline_start_hours(source: "InputSource") -> "dict[tuple[str, str], float]":
    """``{(timeline, t): start hour}`` of every raw timeline step (cumulative ``timestep_duration`` in the rank
    order of :func:`_dt_period_active_steps`, i.e. sorted by step name within the timeline)."""
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return {}
    step_col = next((c for c in ("t", "step", "timestep", "x") if c in tl_dur.columns and c not in ("name", "value")),
                    None)
    if step_col is None:
        return {}
    df = (tl_dur.select(pl.col("name").cast(pl.Utf8).alias("tl"), pl.col(step_col).cast(pl.Utf8).alias("t"),
                        pl.col("value").cast(pl.Float64).alias("dur"))
          .sort("tl", "t")
          .with_columns(start=(pl.col("dur").cum_sum().over("tl") - pl.col("dur"))))
    return {(tl, t): h for tl, t, h in df.select("tl", "t", "start").iter_rows()}


def dtttdt_with_jump_from_source(source: "InputSource",
                                 active_solve: str | None,
                                 workdir: Path | None = None,
                                 *,
                                 ctx: "SolveContext | None" = None,
                                 provider: "object | None" = None,
                                 step_duration: "Param | None" = None,
                                 ) -> pl.DataFrame | None:
    """Build the canonical ``(d, t, t_previous, t_previous_within_timeset,
    d_previous, t_previous_within_solve)`` dispatch-step lag frame.

    Mirrors ``timeline_config.make_step_jump`` (default-block path):

      * Cyclic within solve: first step of the first realized period
        wraps to the last step of the last realized period.
      * Cyclic within timeset: the ``t_previous_within_timeset`` column
        is the block's last step for first-of-period rows, else
        identical to ``t_previous`` for jump=1 interior steps; for
        jump>1 transitions (block boundaries) it pins on the prior
        block's last step.
      * Within-period predecessor: previous step in the same period
        cyclically (first wraps to block_last of the period).

    The extra ``jump`` column is ``make_step_jump``'s: the rank difference to the predecessor
    ``(d_previous, t_previous_within_solve)`` in the solve's own timeline (1 for adjacent steps, also aggregated
    ``new_stepduration`` steps; ≤ 0 at a cyclic wrap, > 1 across a gap).  When the solve's timeline is not
    available (raw ranks), 1 where the predecessor ends where the step starts on the timeline (with
    ``step_duration``, the solve's step lengths), otherwise the raw rank difference.
    """
    info = _dt_period_active_steps(source, active_solve, workdir, ctx=ctx, provider=provider)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    if not period_order:
        return None
    timeline_of = info.get("timeline_for_period") or {}
    hours = _timeline_start_hours(source) if step_duration is not None else {}
    durs: dict[tuple[str, str], float] = {}
    if hours and step_duration is not None:
        durs = {(str(d), str(t)): float(v) for d, t, v in
                step_duration.frame.select("d", "t", "value").iter_rows()}

    step_space = bool(info.get("step_space"))

    def _jump(d_cur: str, t_cur: str, rk_cur: int, d_prev: str, t_prev: str, rk_prev: int) -> int:
        raw = rk_cur - rk_prev
        if step_space or not durs:
            return raw
        tl_c, tl_p = timeline_of.get(d_cur), timeline_of.get(d_prev)
        h_c, h_p = hours.get((tl_c, t_cur)), hours.get((tl_p, t_prev))
        dur_p = durs.get((d_prev, t_prev))
        if tl_c is None or tl_c != tl_p or h_c is None or h_p is None or dur_p is None:
            return raw
        if abs(h_c - (h_p + dur_p)) < 1e-9:
            return 1
        return raw if raw != 1 else 2

    # Per-period anchor lookup: branches whose anchor != self use a
    # per-branch self-loop for the period-wrap predecessor.  Read
    # period__branch.csv (workdir-aware) to discover which periods are
    # branches off a non-self anchor.
    branch_anchor: dict[str, str] = {}
    anchors_with_self: set[str] = set()
    anchor_order: list[str] = []
    pbf: pl.DataFrame | None = None
    period_col, branch_col = "period", "branch"
    # Path B Cat B: prefer ctx.period_branch (schema [d_anchor, b]); it is
    # null-free (under axis Enums its loader raises on a label outside the
    # ``d_anchor`` / ``branch`` vocabulary).  Without a ctx frame, read the
    # provider/workdir CSV.
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            pbf = pb_ctx
            period_col, branch_col = "d_anchor", "b"
    if pbf is None and workdir is not None:
        pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
        if _provider_has_key(provider, pb_path):  # Phase E-j — seed-aware
            try:
                pbf = _provider_read(provider, pb_path)
            except Exception:
                pbf = None
    if pbf is not None and pbf.height > 0:
        pb_rows = list(zip(pbf[period_col].to_list(),
                              pbf[branch_col].to_list()))
        for a, b in pb_rows:
            if a == b:
                anchors_with_self.add(a)
                anchor_order.append(a)
        for a, b in pb_rows:
            if b != a and a in anchors_with_self:
                branch_anchor[b] = a
    # Continuation fan-out: the time-branch map (period -> time_branch,
    # from solve_branch__time_branch.csv) lets a continuation branch
    # period wrap to the nearest earlier period of the SAME time-branch
    # instead of self-wrapping, mirroring ``make_step_jump``.  Ctx-first
    # (typed accessor), provider/workdir fallback.
    time_branch_of: dict[str, str] = {}
    if ctx is not None:
        sbtb_ctx = ctx.solve_branch_time_branch
        if sbtb_ctx.height > 0:
            time_branch_of = dict(zip(sbtb_ctx["d"].to_list(),
                                        sbtb_ctx["time_branch"].to_list()))
    if not time_branch_of and workdir is not None:
        sbtb_path = (Path(workdir) / "solve_data"
                     / "solve_branch__time_branch.csv")
        if _provider_has_key(provider, sbtb_path):  # Phase E-j — seed-aware
            try:
                sbtb_df = _provider_read(provider, sbtb_path)
            except Exception:
                sbtb_df = None
            if (sbtb_df is not None and sbtb_df.height > 0
                    and "period" in sbtb_df.columns
                    and "branch" in sbtb_df.columns):
                time_branch_of = dict(zip(sbtb_df["period"].to_list(),
                                            sbtb_df["branch"].to_list()))
    if branch_anchor and not time_branch_of:
        # Failure semantics (do NOT fall back to self-wrap): a
        # period__branch with branch rows but no time-branch map means
        # the per-solve preprocessing artefacts are inconsistent.  A
        # silent self-wrap fallback would sever continuation branch
        # periods from their own branch chain — the exact silent
        # truncation bug this walk repairs — so raise instead.
        raise ValueError(
            "dtttdt_from_source: period__branch declares branch periods "
            f"({sorted(branch_anchor)}) but solve_branch__time_branch is "
            "missing or empty — cannot resolve cross-period wrap "
            "predecessors for branch periods.  The per-solve emitter "
            "(emit_branch_weights_and_map) writes this frame for every "
            "solve; its absence indicates an inconsistent workdir/"
            "Provider state."
        )
    out_rows: list[tuple[str, str, str, str, str, str]] = []
    len(period_order)
    # Find within-solve predecessor for first-of-period rows.
    for pi, period in enumerate(period_order):
        steps = per_period.get(period) or []
        if not steps:
            continue
        # Determine the predecessor period for the j=0 wrap row.
        # Mirrors ``timeline_config.make_step_jump``:
        #   * branch (non-anchor) period → nearest earlier period with
        #     the SAME time-branch (continuation linkage); self-wrap
        #     when none exists (the branching period's own fan);
        #   * first anchor period (== first_period_name in flextool's
        #     reversed walk) → wrap to last_period_name (last in
        #     active_time_list iteration order, == period_order[-1] —
        #     retained quirk);
        #   * non-first anchor period → nearest earlier ANCHOR period
        #     (skips branch periods so a realized continuation period
        #     wraps to the previous realized period); positional
        #     ``period_order[pi - 1]`` fallback when there is no
        #     period__branch data (deterministic path, where
        #     positional ≡ previous-anchor anyway).
        if period in branch_anchor:
            wrap_period = period
            tb = time_branch_of.get(period)
            if tb is not None:
                for qi in range(pi - 1, -1, -1):
                    if time_branch_of.get(period_order[qi]) == tb:
                        wrap_period = period_order[qi]
                        break
            if wrap_period == period:
                # No earlier same-time-branch period exists.  MID-HORIZON
                # reveal: this branch copy's own anchor period is not the
                # first period, so link its first step to the nearest
                # anchor STRICTLY BEFORE the copy's own anchor — the copy
                # inherits the shared pre-reveal trunk end-state (two-stage
                # storage-state non-anticipativity, Slice E design §5.2).
                # FIRST-PERIOD fan: the copy's anchor IS the first period
                # → empty range → wrap_period stays == period (self-wrap,
                # byte-parity with the pre-Slice-E behaviour).  This mirror
                # of ``make_step_jump`` is pinned by the Step-0.5 parity
                # guard.
                p_anchor = branch_anchor.get(period)
                if p_anchor is not None and p_anchor in period_order:
                    pos_anchor = period_order.index(p_anchor)
                    for qi in range(pos_anchor - 1, -1, -1):
                        if period_order[qi] in anchors_with_self:
                            wrap_period = period_order[qi]
                            break
                elif p_anchor is not None and p_anchor in anchor_order:
                    # The anchor is a period label (no steps, not in period_order): walk the anchors in
                    # period__branch order back from it to the nearest one with steps.
                    for a in reversed(anchor_order[:anchor_order.index(p_anchor)]):
                        if a in period_order:
                            wrap_period = a
                            break
        else:
            if pi == 0:
                wrap_period = period_order[-1]
            else:
                wrap_period = period_order[pi - 1]
                for qi in range(pi - 1, -1, -1):
                    if period_order[qi] in anchors_with_self:
                        wrap_period = period_order[qi]
                        break
        prev_solve_steps = per_period.get(wrap_period) or []
        prev_solve_last_t = prev_solve_steps[-1][0] if prev_solve_steps \
                            else steps[-1][0]
        # Within-period: cyclic over current period.
        steps[-1][0]
        n = len(steps)
        # Track running block_last_t as we walk forward from j=0 to j=n-1.
        # In flextool's reversed loop, block_last initialises to len-1 and
        # decreases on jump>1 transitions.  Equivalent forward pass: the
        # block_last for index j is the index of the next jump>1 boundary
        # to the right (or the last index n-1 if none).
        # Pre-compute for each j, the block_last index (forward sweep).
        block_last_idx_for: list[int] = [n - 1] * n
        # Find boundaries: indices where ranks[j] - ranks[j-1] > 1.
        # block_last_idx_for[k] = block_last for indices in current block.
        # We iterate from the right, tracking the current block's last index.
        cur_last = n - 1
        for k in range(n - 1, 0, -1):
            t_k, rk_k = steps[k]
            t_prev, rk_prev = steps[k - 1]
            block_last_idx_for[k] = cur_last
            if rk_k - rk_prev > 1:
                # Boundary — next k's block_last is k-1.
                cur_last = k - 1
        block_last_idx_for[0] = cur_last
        for j in range(n):
            t_cur, rk_cur = steps[j]
            if j == 0:
                # First step of the period — cross-period wrap.  Use
                # wrap_period computed above (handles branch self-wrap
                # vs anchor cyclic-chain).
                prev_p = wrap_period
                prev_p_steps = per_period.get(prev_p) or []
                prev_t = prev_p_steps[-1][0] if prev_p_steps else t_cur
                # Block's last step in current period — specifically the
                # last step of the block CONTAINING the current step
                # (block_last_idx_for[0]).
                bli0 = block_last_idx_for[0]
                prev_within_ts = steps[bli0][0]
                prev_within_solve = prev_solve_last_t
                d_prev = prev_p
                jump = (_jump(period, t_cur, rk_cur, prev_p, prev_p_steps[-1][0], prev_p_steps[-1][1])
                        if prev_p_steps else 0)
            else:
                t_prev, rk_prev = steps[j - 1]
                prev_t = t_prev
                jump = rk_cur - rk_prev
                if jump > 1:
                    # Block boundary — pin within_ts to the last step of
                    # the block CONTAINING the current step (j), per
                    # ``make_step_jump``'s reverse-walk semantics.  The
                    # outgoing-block last index would be (j-1) but the
                    # canonical algorithm pins it to active_time[block_last]
                    # tracked in reverse-iteration order which equals the
                    # last index of the block containing j.
                    bli = block_last_idx_for[j]
                    prev_within_ts = steps[bli][0]
                else:
                    prev_within_ts = t_prev
                d_prev = period
                prev_within_solve = t_prev
                jump = _jump(period, t_cur, rk_cur, period, t_prev, rk_prev)
            out_rows.append((period, t_cur, prev_t, prev_within_ts,
                              d_prev, prev_within_solve, jump))
    if not out_rows:
        return None
    df = pl.DataFrame(
        out_rows,
        schema=["d", "t", "t_previous", "t_previous_within_timeset",
                  "d_previous", "t_previous_within_solve", "jump"],
        orient="row",
    )
    return df.sort("d", "t")


# ---------------------------------------------------------------------------
# §3.9.3 — period_block / period_block_succ / period_block_time
# ---------------------------------------------------------------------------


def period_block_family_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       workdir: Path | None = None,
                                       *,
                                       ctx: "SolveContext | None" = None,
                                       provider: "object | None" = None,
                                       realized: "pl.DataFrame | None" = None,
                                       ) -> dict | None:
    """``period_block`` ``(d, b_first)``, ``period_block_succ`` ``(d, b_first, b_next)`` and
    ``period_block_time`` ``(d, b_first, t)``: the timeset blocks of each period (mirrors
    ``timeline_config.make_period_block``) for the ``bind_intraperiod_blocks`` nodes.  Per period the active
    steps are taken in rank order; a new block starts where the rank gap to the previous step exceeds 1.

    *realized* ``(d, t)`` (a rolling solve's realized steps): a block also ends at the last realized step of a
    period that continues past it, so the state the roll hands over is the end state of a block (the same cut as
    the coarse steps, ``params._coarse.coarse_node_frames``).

    Returns the three frames, or None when the source is insufficient.
    """
    info = _dt_period_active_steps(source, active_solve, workdir, ctx=ctx, provider=provider)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    realized_set: set[tuple[str, str]] = set()
    if realized is not None and realized.height > 0:
        realized_set = set(realized.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).iter_rows())
    pb_rows: list[tuple[str, str]] = []         # (d, b_first)
    pbs_rows: list[tuple[str, str, str]] = []   # (d, b_first, b_next)
    pbt_rows: list[tuple[str, str, str]] = []   # (d, b_first, t)
    for period in period_order:
        steps = per_period.get(period) or []
        if not steps:
            continue
        # Index of the period's last realized step when the period continues past it (rolling solves only).
        cut_after = None
        if realized_set:
            last = max((j for j, (t, _rk) in enumerate(steps) if (str(period), str(t)) in realized_set), default=None)
            if last is not None and last < len(steps) - 1:
                cut_after = last
        block_firsts: list[str] = [steps[0][0]]
        cur_b = steps[0][0]
        for j, (t, rk) in enumerate(steps):
            if j > 0 and (rk - steps[j - 1][1] > 1 or j - 1 == cut_after):
                cur_b = t
                block_firsts.append(cur_b)
            pbt_rows.append((period, cur_b, t))
        for i, b in enumerate(block_firsts):
            pb_rows.append((period, b))
            nxt = block_firsts[(i + 1) % len(block_firsts)]
            pbs_rows.append((period, b, nxt))
    period_block = (pl.DataFrame(pb_rows, schema=["d", "b_first"],
                                    orient="row").unique(maintain_order=True)
                    if pb_rows else None)
    period_block_succ = (pl.DataFrame(pbs_rows,
                                          schema=["d", "b_first", "b_next"],
                                          orient="row")
                          if pbs_rows else None)
    period_block_time = (pl.DataFrame(pbt_rows,
                                          schema=["d", "b_first", "t"],
                                          orient="row").unique(maintain_order=True)
                          if pbt_rows else None)

    return dict(
        period_block=period_block,
        period_block_succ=period_block_succ,
        period_block_time=period_block_time,
    )


# ---------------------------------------------------------------------------
# §3.9.2 — nodeStateBlock (bind_intraperiod_blocks nodes)
# ---------------------------------------------------------------------------


def _node_storage_binding_method_with_fallback(source: "InputSource"
                                                       ) -> pl.DataFrame | None:
    """Per-node storage binding method with the
    ``method_with_fallback_sets.write_node_storage_binding_method``
    fallback rule applied.

    Mirrors flextool's preprocessing: explicit rows are kept verbatim;
    every node lacking an explicit row gets the default
    ``bind_forward_only``.  Returns ``[n, method]`` frame.
    """
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return None
    explicit = _try_param(source, "node", "storage_binding_method")
    if explicit is None:
        explicit_rows = pl.DataFrame(schema={"n": schema_dtype(_enums, "n"),
                                              "method": pl.Utf8})
    else:
        explicit_rows = (explicit
            .pipe(rename_to_axis, {"name": "n", "value": "method"})
            .select("n", "method"))
    explicit_n = set(explicit_rows["n"].to_list())
    fallback_n = [n for n in nodes["name"].to_list() if n not in explicit_n]
    if fallback_n:
        fb = (pl.DataFrame({"n": fallback_n,
                              "method": ["bind_forward_only"] * len(fallback_n)})
                .with_columns(pl.col("n").cast(schema_dtype(_enums, "n"),
                                                  strict=False)))
    else:
        fb = pl.DataFrame(
            schema={"n": schema_dtype(_enums, "n"),
                    "method": pl.Utf8})
    out = pl.concat([explicit_rows, fb]).unique().sort("n", "method")
    return out


def nodeStateBlock_from_source(source: "InputSource") -> pl.DataFrame | None:
    """The ``nodeStateBlock`` set: nodes with ``storage_binding_method == 'bind_intraperiod_blocks'``, balanced
    once per timeset block (``nodeBalanceBlock_eq``).  ``None`` when there are none.  (Nodes of a coarse
    new_stepduration group are ``nodeCoarse``, see ``params._coarse``.)"""
    sbm = _try_param(source, "node", "storage_binding_method")
    if sbm is None:
        return None
    intraperiod = (sbm.lazy()
                   .filter(pl.col("value") == "bind_intraperiod_blocks")
                   .select(alias_to_axis("name", "n"))
                   .collect())
    if intraperiod.height == 0:
        return None
    return pl.DataFrame({"n": sorted(set(intraperiod["n"].to_list()))})


# ---------------------------------------------------------------------------
# §3.9.5 — p_state_existing_capacity / p_state_upper
# ---------------------------------------------------------------------------


def p_state_existing_capacity_from_source(source: "InputSource",
                                                active_solve: str | None,
                                                nodeState_df: pl.DataFrame
                                                  | None,
                                                workdir: Path | None = None,
                                                *,
                                                ctx: "SolveContext | None"
                                                  = None,
                                                provider: "object | None" = None,
                                                ) -> "Param | None":
    """``p_state_existing_capacity`` per (n, d) — node existing capacity
    restricted to nodes carrying state (``nodeState`` set).

    Algorithm: take ``p_entity_all_existing`` for nodes only, restrict to
    nodes in ``nodeState_df``.  Falls back to the lifetime-gated existing
    capacity (existing × virtual_unitsize when it is set) of the state
    nodes when the chained Param isn't available.  Broadcasts onto the full
    period set including stochastic-branch siblings (workdir-aware).
    """
    if nodeState_df is None or nodeState_df.height == 0:
        return None
    state_n = set(nodeState_df["n"].to_list())
    # Full period set (realized + invest + stochastic-branch siblings).
    piu_base = _period_in_use_set(source, active_solve, ctx=ctx, provider=provider) if active_solve \
                else []
    full_periods = _expand_branch_periods(piu_base, workdir, ctx=ctx, provider=provider)
    pae = p_entity_all_existing_from_source(source, active_solve, workdir, ctx=ctx, provider=provider)
    if pae is not None and pae.frame.height > 0:
        base = (pae.frame.lazy()
                  .pipe(rename_to_axis, {"e": "n"})
                  .filter(pl.col("n").is_in(list(state_n)))
                  .select("n", "d", "value")
                  .collect())
        # Broadcast each (n, d) to its branch siblings sharing the
        # anchor's existing value.
        if full_periods and len(full_periods) > base["d"].n_unique():
            anchor_branches = _expand_branch_periods(
                base["d"].unique().to_list(), workdir, ctx=ctx, provider=provider)
            # Build (anchor → list of d) mapping
            extra_rows: list[tuple[str, str, float]] = []
            covered = set(base["d"].to_list())
            for d in anchor_branches:
                if d in covered:
                    continue
                # Find anchor whose row to copy: any anchor it expanded
                # from.  Walk period__branch.csv to find anchor mapping.
                anchor_for_d = _branch_anchor(d, workdir, ctx=ctx, provider=provider)
                if anchor_for_d is None:
                    continue
                rows_for_anchor = base.filter(
                    pl.col("d") == anchor_for_d)
                for r in rows_for_anchor.iter_rows(named=True):
                    extra_rows.append((r["n"], d, r["value"]))
            if extra_rows:
                add = pl.DataFrame(
                    extra_rows, schema=["n", "d", "value"], orient="row")
                base = pl.concat([base, add])
        df = base.sort("n", "d")
        if df.height > 0:
            return Param(("n", "d"), df)
    # Fallback (no period_in_use for p_entity_all_existing): the lifetime-gated existing capacity
    # (existing × virtual_unitsize when it is set) of the state nodes with an explicit existing, over the full
    # period set or, without one, the periods of an existing period map.
    ex = _try_param(source, "node", "existing")
    if ex is None or ex.height == 0:
        return None
    ex_nodes = set(ex["name"].cast(pl.Utf8).to_list()) & state_n
    periods = full_periods
    if not periods and "period" in ex.columns:
        periods = sorted({str(d) for d in ex["period"].drop_nulls().to_list()})
    if not ex_nodes or not periods:
        return None
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    df = (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
            .filter(pl.col("e").cast(pl.Utf8).is_in(sorted(ex_nodes)))
            .pipe(rename_to_axis, {"e": "n"})
            .select("n", "d", "value")
            .sort("n", "d")
            .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


def _branch_anchor(branch: str, workdir: Path | None,
                      *, ctx: "SolveContext | None" = None,
                      provider: "object | None" = None,
                      ) -> str | None:
    """Return the anchor period whose ``period__branch.csv`` row maps
    to *branch*.  None if no mapping found.

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` frame is consulted first.
    Falls back to the CSV-read path when ``ctx`` is None or the typed
    accessor is empty.
    """
    if ctx is not None:
        pb = ctx.period_branch
        if pb.height > 0:
            cand = pb.filter(pl.col("b") == branch)
            if cand.height > 0:
                return cand["d_anchor"][0]
            # Typed accessor is authoritative when populated; don't fall
            # through to the CSV path even if branch isn't in it.
            return None
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    try:
        df = _provider_read(provider, p)
    except Exception:
        return None
    if df.height == 0:
        return None
    cand = df.filter(pl.col("branch") == branch)
    if cand.height > 0:
        return cand["period"][0]
    return None


def _node_unitsize_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-node unitsize lazy frame following the flextool preprocess
    formula (entity_period_calc_params.py:186-202): use
    ``virtual_unitsize`` if explicitly set non-zero, else ``existing``
    if explicitly set non-zero, else 1000.0.

    Uses :func:`_try_param_explicit` so default-broadcast rows from
    SpineDbReader are suppressed — only entities with a parameter_value
    row count as "explicitly set".  This mirrors flextool's
    ``p_node.get(name, None)`` semantic: an absent row is *not*
    equivalent to a row with the schema default value.

    Returns a lazy frame with ``[n, us]`` for every node entity.

    Solve-invariant: the cascade depends only on ``source`` (node
    entities + per-node params), so the collected frame is memoized per
    source object (see ``_NODE_UNITSIZE_CACHE``).
    """
    try:
        cached = _NODE_UNITSIZE_CACHE.get(source)
    except TypeError:
        cached = None
    if cached is not None:
        return cached.lazy()
    us = _try_param_explicit(source, "node", "virtual_unitsize")
    ex = _try_param_explicit(source, "node", "existing")
    nodes = _try_entities(source, "node")
    if nodes is None or nodes.height == 0:
        return pl.LazyFrame(schema={"n": schema_dtype(_enums, "n"),
                                     "us": pl.Float64})
    # virtual_unitsize / existing may carry a period dim authored as a Spine
    # Map; the index column inherits the Map's ``index_name`` and may be
    # named ``period``, ``x``, etc.  :func:`unitsize_cascade` collapses any
    # number of rows per node to one (per-node MAX), so the index column
    # is not read.  Existing has no default (schema-None policy); rows
    # present == explicit.
    def _rows(frame, name):
        if frame is None or frame.height == 0:
            return None
        return frame.lazy().select(alias_to_axis("name", "n"),
                                   pl.col("value").cast(pl.Float64).alias(name))

    result = unitsize_cascade(
        nodes.lazy().select(alias_to_axis("name", "n")),
        _rows(us, "vu"), _rows(ex, "ex"), key="n",
    ).collect()
    try:
        _NODE_UNITSIZE_CACHE[source] = result
    except TypeError:
        pass
    return result.lazy()


def p_state_upper_from_source(source: "InputSource",
                                  state_existing_capacity: "Param | None",
                                  nodeState_df: pl.DataFrame | None,
                                  ) -> "Param | None":
    """``p_state_upper[n, d] = state_existing_capacity[n, d] /
    p_entity_unitsize[n]``, with the unitsize fallback chain
    ``virtual_unitsize OR existing OR 1000``.  Mirrors
    ``entity_period_calc_params.py:186-202`` and ``flex_data.py:1755-1772``.
    """
    if (state_existing_capacity is None
            or state_existing_capacity.frame.height == 0
            or nodeState_df is None or nodeState_df.height == 0):
        return None
    cap = state_existing_capacity.frame
    us_lf = _node_unitsize_lf(source)
    df = (cap.lazy()
              .join(us_lf, on="n", how="left")
              .with_columns(us=pl.col("us").fill_null(UNITSIZE_DEFAULT))
              .with_columns(value=pl.col("value").cast(pl.Float64)
                                                   / pl.col("us"))
              .select("n", "d", "value")
              .sort("n", "d")
              .collect())
    if df.height == 0:
        return None
    return Param(("n", "d"), df)


# ---------------------------------------------------------------------------
# §3.9.6 — storage_use_reference_value (multi-method exclusion chain)
# ---------------------------------------------------------------------------


def storage_use_reference_value_from_source(source: "InputSource",
                                                  nodeStateBlock_df: pl.DataFrame
                                                    | None,
                                                  *,
                                                  rolling: bool = False,
                                                  ) -> pl.DataFrame | None:
    """``storage_use_reference_value`` — nodes with horizon method
    ``use_reference_value`` after multi-method exclusion.

    Mirror of flex_data.py:1880-1899.  Filter
    ``node.storage_solve_horizon_method == 'use_reference_value'``, then
    anti-join against:

      * ``storage_start_end_method ∈ {fix_end, fix_start_end}``
      * ``storage_binding_method ∈ {bind_within_solve, bind_within_period,
                                      bind_within_timeblock, bind_intraperiod_blocks}``
        (the end state is tied to the start); the caller also drops the
        nodes of the solve's own plain cyclic sets, which hold the
        blended-weights nodes a non-RP solve downgrades to those methods
      * ``nodeStateBlock`` (synthesised + explicit).

    With *rolling* (a rolling_window sub-solve) the bind_intraperiod_blocks nodes keep the pin: it applies where the
    roll's last period chains forward (``model.core._eq_storage_reference_value``).
    """
    sshm = _try_param(source, "node", "storage_solve_horizon_method")
    if sshm is None:
        return None
    base = (sshm.lazy()
                 .filter(pl.col("value") == "use_reference_value")
                 .select(alias_to_axis("name", "n"))
                 .unique()
                 .collect())
    if base.height == 0:
        return None
    # Exclusion sets.
    sse = _try_param(source, "node", "storage_start_end_method")
    excl_lists: list[list[str]] = []
    if sse is not None:
        e = (sse.lazy()
                 .filter(pl.col("value").is_in(["fix_end", "fix_start_end"]))
                 .select(alias_to_axis("name", "n"))
                 .collect())
        if e.height > 0:
            excl_lists.append(e["n"].to_list())
    sbm_full = _node_storage_binding_method_with_fallback(source)
    if sbm_full is not None:
        b = (sbm_full.lazy()
                  .filter(pl.col("method").is_in([
                      "bind_within_solve", "bind_within_period", "bind_within_timeblock"]
                      + ([] if rolling else ["bind_intraperiod_blocks"])))
                  .select("n")
                  .collect())
        if b.height > 0:
            excl_lists.append(b["n"].to_list())
    if not rolling and nodeStateBlock_df is not None and nodeStateBlock_df.height > 0:
        excl_lists.append(nodeStateBlock_df["n"].to_list())
    excl = set()
    for lst in excl_lists:
        excl.update(lst)
    out = base.filter(~pl.col("n").is_in(list(excl)))
    if out.height == 0:
        return None
    return out.unique().sort("n")


# ---------------------------------------------------------------------------
# §3.9.7 — p_roll_continue_state / p_fix_storage_quantity (rolling handoff)
# ---------------------------------------------------------------------------


def p_roll_continue_state_from_handoff(
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Read the rolling-handoff ``p_roll_continue_state`` from the
    canonical ``handoff/roll_end_state`` Provider key
    (schema ``[node, value]``).

    Phase 4.2-1b — the legacy ``solve_data/p_roll_continue_state``
    CSV read path is gone; the typed handoff translator pipeline
    (extended in Phase 4.2-0) populates the handoff key from
    :pyattr:`SolveHandoff.roll_end_state` at iteration start.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_ROLL_END_STATE)
    if df is None or df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"node": "n"})
            .with_columns(value=pl.col("value")
                                    .cast(pl.Float64, strict=False)
                                    .fill_null(0.0))
            .select("n", "value"))
    if df.height == 0:
        return None
    return Param(("n",), df)


def p_fix_storage_quantity_from_handoff(
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Read the rolling-handoff ``p_fix_storage_quantity`` from the
    canonical ``handoff/fix_storage_quantity`` Provider key
    (schema ``[node, period, step, p_fix_storage_quantity]``).

    Phase 4.1d — the legacy ``solve_data/fix_storage_quantity`` read
    path is gone; the translator populates the handoff key from
    :pyattr:`SolveHandoff.fix_storage_quantity` at iteration start.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_QUANTITY)
    if df is None or df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"period": "d", "step": "t", "node": "n",
                       "p_fix_storage_quantity": "value"})
            .with_columns(value=pl.col("value").cast(pl.Float64))
            .select("n", "d", "t", "value"))
    if df.height == 0:
        return None
    return Param(("n", "d", "t"), df)


# ---------------------------------------------------------------------------
# §3.9.8 — dtt_timeline_matching / period_branch
# ---------------------------------------------------------------------------


def dtt_timeline_matching_from_workdir(workdir: Path | None,
                                              *,
                                              provider: "object | None" = None,
                                              ) -> pl.DataFrame | None:
    """Per-solve timeline matching map ``(d, t, t_upper)`` between a
    sub-solve's fine timesteps and the upper-level coarse timesteps.
    Read from ``solve_data/timeline_matching_map.csv``.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "timeline_matching_map.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    return (df
            .pipe(rename_to_axis, {"period": "d", "step": "t", "upper_step": "t_upper"})
            .select("d", "t", "t_upper")
            .unique(maintain_order=True))


def period_branch_from_source(source: "InputSource",
                                workdir: Path | None,
                                *,
                                ctx: "SolveContext | None" = None,
                                provider: "object | None" = None,
                                ) -> pl.DataFrame | None:
    """``period_branch`` rolling-handoff helper — (d_upper, d) anchor →
    branch map per audit §3.9.8.

    Spine source: ``period__branch`` 1-dim entity class is currently a
    flat name list with no values; the canonical per-solve mapping is
    written by ``solve_writers`` to ``solve_data/period__branch.csv``
    with columns ``(period, branch)``.  Read that file when present
    (workdir-aware path).

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` frame is consulted first
    (canonical schema ``[d_anchor, b]`` → remapped to the legacy
    ``[d_upper, d]`` shape this helper exposes).
    """
    if ctx is not None:
        pb = ctx.period_branch
        if pb.height > 0:
            # ctx.period_branch schema: [d_anchor, b]; downstream consumer
            # expects (d_upper, d) where d_upper == branch, d == anchor.
            return (pb
                    .pipe(rename_to_axis, {"d_anchor": "d", "b": "d_upper"})
                    .select("d_upper", "d")
                    .unique(maintain_order=True))
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if _provider_has_key(provider, p):  # Phase E-h — seed-aware
            df = _provider_read(provider, p)
            if df.height > 0:
                return (df
                        .pipe(rename_to_axis, {"period": "d", "branch": "d_upper"})
                        .select("d_upper", "d")
                        .unique(maintain_order=True))
    return None


# ---------------------------------------------------------------------------
# §3.9 helper compose: dtttdt-derived auxiliary frames
# ---------------------------------------------------------------------------


def dtttdt_forward_only_from_dtttdt(dtttdt: pl.DataFrame | None,
                                    first_periods: "list[str] | None" = None,
                                    ) -> pl.DataFrame | None:
    """``dtttdt`` sorted by (d, t), .slice(1) — drops the first wrap row
    so bind_forward_only nodes don't carry a state-change term across
    the very first step of the first period (mod:2188).

    *first_periods* are the periods of ``period_first_of_solve``.  When a
    stochastic fan opens in the solve's first period there are several
    (its forecast branch periods; the period label has no steps): every
    one of them starts the solve, so the first step of each is dropped.
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    out = dtttdt.sort("d", "t").slice(1)
    if first_periods is not None and len(first_periods) > 1:
        firsts = (dtttdt.filter(pl.col("d").cast(pl.Utf8).is_in([str(p) for p in first_periods]))
                  .group_by("d").agg(pl.col("t").min()))
        out = out.join(firsts, on=["d", "t"], how="anti")
    return out if out.height > 0 else None


def dtttdt_within_solve_from_dtttdt(dtttdt: pl.DataFrame | None,
                                    first_periods: "list[str] | None" = None,
                                    ) -> pl.DataFrame | None:
    """``dtttdt`` with the bind_within_solve wrap of a stochastic fan in the solve's first period; None when the
    plain ``dtttdt`` wrap is already the bind_within_solve one.

    *first_periods* are the periods of ``period_first_of_solve``: with a fan in the solve's first period, its
    forecast branch periods (the roots; the period label has no steps).  ``dtttdt`` (``make_step_jump``) wraps the
    first step of each root to the root's own last step.  bind_within_solve closes one loop per path instead: the
    first step of each root continues from the last step of the last period on its own path (the leaf the root's
    continuation branch periods lead to; with several, the last in name order).  A root that is its own leaf (a
    single-period solve) keeps its wrap, so only a fan in the first period of a multi-period solve gives a frame.
    The tree is read from the first-step rows of ``dtttdt``: a period whose first step continues from another
    period is its child.  A period's first and last steps are its smallest and largest time steps (as
    :func:`dtttdt_forward_only_from_dtttdt` takes the first).
    """
    if dtttdt is None or dtttdt.height == 0 or first_periods is None or len(first_periods) < 2:
        return None
    roots = [str(p) for p in first_periods]
    u = dtttdt.select(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8),
                      _dp=pl.col("d_previous").cast(pl.Utf8))
    span = u.group_by("_d").agg(_first=pl.col("_t").min(), _last=pl.col("_t").max())
    last = dict(span.select("_d", "_last").iter_rows())
    first = dict(span.select("_d", "_first").iter_rows())
    children: dict[str, list[str]] = {}
    for d_name, d_prev in (u.join(span.select("_d", _t=pl.col("_first")), on=["_d", "_t"], how="semi")
                           .select("_d", "_dp").unique().iter_rows()):
        if d_prev != d_name and d_name not in roots:
            children.setdefault(d_prev, []).append(d_name)

    def leaf_of(root: str) -> str:
        leaves, seen, stack = [], set(), [root]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            kids = children.get(node, [])
            if kids:
                stack.extend(kids)
            else:
                leaves.append(node)
        # Convention: with several leaves, the last in sorted name order (as core._eq_storage_within_solve_path_end
        # picks d*).
        return max(leaves)

    relink = {r: leaf_of(r) for r in roots if r in last}
    relink = {r: leaf for r, leaf in relink.items() if leaf != r}
    if not relink:
        return None
    wrap = pl.DataFrame({"_d": list(relink), "_t": [first[r] for r in relink],
                         "_dp_new": list(relink.values()), "_tp_new": [last[leaf] for leaf in relink.values()]},
                        schema={"_d": pl.Utf8, "_t": pl.Utf8, "_dp_new": pl.Utf8, "_tp_new": pl.Utf8})
    dp_dtype, tp_dtype = dtttdt.schema["d_previous"], dtttdt.schema["t_previous_within_solve"]
    return (dtttdt.with_columns(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(wrap, on=["_d", "_t"], how="left")
            .with_columns(
                d_previous=pl.coalesce(pl.col("_dp_new"), pl.col("d_previous").cast(pl.Utf8)).cast(dp_dtype),
                t_previous_within_solve=pl.coalesce(pl.col("_tp_new"),
                                                    pl.col("t_previous_within_solve").cast(pl.Utf8)).cast(tp_dtype))
            .select(dtttdt.columns))


def read_period_first_of_solve(sd: Path, *, provider: "object | None" = None) -> "list[str] | None":
    """The periods of ``solve_data/period_first_of_solve.csv``; None when absent."""
    p = Path(sd) / "period_first_of_solve.csv"
    if not _provider_has_key(provider, p):
        return None
    df = _provider_read(provider, p)
    if df.height == 0 or "period" not in df.columns:
        return None
    return [str(x) for x in df["period"].to_list()]


def dtttdt_block_interior_from_dtttdt(dtttdt: pl.DataFrame | None,
                                          period_block_time: pl.DataFrame
                                              | None = None,
                                          ) -> pl.DataFrame | None:
    """Interior-of-block dtttdt rows of the ``bind_intraperiod_blocks`` timeset blocks.

      1. **One block per period** (or no *period_block_time*): keep dtttdt rows where
         ``t_previous_within_timeset == t_previous`` (jump=1 interior, not a boundary).
      2. **Several timeset blocks per period**: for each (d, b_first) of *period_block_time*, consecutive
         sorted t's give (d, t, t_previous) — the same set as branch 1, built from the blocks.
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    multi_res = False
    if period_block_time is not None and period_block_time.height > 0:
        # Several timeset blocks in a period: more than one distinct b_first per d.
        nb = (period_block_time
              .group_by("d")
              .agg(pl.col("b_first").n_unique().alias("nb"))
              ["nb"].max())
        if nb is not None and nb > 1:
            multi_res = True
    if multi_res and period_block_time is not None:
        rows: list[tuple[str, str, str]] = []
        pbt_sorted = period_block_time.sort("d", "b_first", "t")
        for (dval, bf), grp in pbt_sorted.group_by(
                ["d", "b_first"], maintain_order=True):
            ts = grp["t"].to_list()
            for i in range(1, len(ts)):
                rows.append((dval, ts[i], ts[i - 1]))
        if not rows:
            return None
        return (pl.DataFrame(rows, schema=["d", "t", "t_previous"],
                                orient="row").unique(maintain_order=True))
    if "t_previous_within_timeset" not in dtttdt.columns:
        return None
    out = (dtttdt
            .filter(pl.col("t_previous_within_timeset")
                    == pl.col("t_previous"))
            .select("d", "t", "t_previous"))
    return out if out.height > 0 else None


def dtttdt_block_lag_from_dtttdt(dtttdt: pl.DataFrame | None,
                                 period_block_time: pl.DataFrame | None,
                                 chained: pl.DataFrame | None = None,
                                 ) -> pl.DataFrame | None:
    """``(d, t, d_previous, t_previous_within_solve)``: the step each step of a ``bind_intraperiod_blocks`` block
    continues from (the state change of ``nodeBalanceBlock_eq``).

    The blocks of a period form a loop: inside the period the previous step (across block boundaries too), and the
    period's first step continues from the period's own last step (the within-period lag of
    ``flex_data._within_period_lag``).  The periods of *chained* ``(d)`` (in a rolling solve, the periods the roll
    holds only in part) chain forward instead: their rows are the bind_forward_only lag
    (:func:`dtttdt_forward_only_from_dtttdt`), so a period's first step continues from the previous step of the
    solve and the solve's first step has no row (its start is the fix_start state or the roll carry).  Restricted to
    the steps of *period_block_time*.
    """
    if dtttdt is None or dtttdt.height == 0 or period_block_time is None or period_block_time.height == 0:
        return None
    from flextool.engine.preprocess.flex_data import _within_period_lag

    lag = _within_period_lag(dtttdt)
    if lag is None:
        return None
    loop = lag.select("d", "t", pl.col("d").alias("d_previous"),
                      pl.col("t_previous_within_period").alias("t_previous_within_solve"))
    if chained is not None and chained.height > 0:
        names = [str(x) for x in chained["d"].to_list()]
        loop = loop.filter(~pl.col("d").cast(pl.Utf8).is_in(names))
        fwd = dtttdt_forward_only_from_dtttdt(dtttdt)
        if fwd is not None:
            chain = (fwd.filter(pl.col("d").cast(pl.Utf8).is_in(names))
                     .select([pl.col(c).cast(loop.schema[c], strict=False) for c in loop.columns]))
            loop = pl.concat([loop, chain], how="vertical")
    steps = period_block_time.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
    out = (loop.with_columns(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
           .join(steps.rename({"d": "_d", "t": "_t"}), on=["_d", "_t"], how="semi")
           .select("d", "t", "d_previous", "t_previous_within_solve")
           .sort("d", "t"))
    return out if out.height > 0 else None


def _rolling_realized_steps(flex_data: object, workdir: Path | None,
                            provider: "object | None") -> pl.DataFrame | None:
    """``(d, t)`` the solve realizes when it is a rolling sub-solve (``flex_data.solve_is_rolling``), else None:
    where the bind_intraperiod_blocks blocks are cut (:func:`period_block_family_from_source`)."""
    if not getattr(flex_data, "solve_is_rolling", False) or workdir is None:
        return None
    from flextool.engine.preprocess.flex_data import _realized_steps

    return _realized_steps(Path(workdir) / "solve_data", flex_data, provider=provider)


# ---------------------------------------------------------------------------
# Public — Γ.3.E field list (selective scope-checking).
# ---------------------------------------------------------------------------


E_PUBLIC_FIELDS: tuple[str, ...] = (
    "dtttdt",
    "dtttdt_forward_only",
    "dtttdt_within_solve",
    "dtttdt_block_interior",
    "dtttdt_block_lag",
    "nodeStateBlock",
    "period_block",
    "period_block_succ",
    "period_block_time",
    "p_state_upper",
    "p_state_existing_capacity",
    "storage_use_reference_value",
    "p_roll_continue_state",
    "p_fix_storage_quantity",
    "dtt_timeline_matching",
    "period_branch",
)


def apply_derived_e(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.E storage block algebra (§3.9), mutating ``flex_data``
    in place.

    Order — dependency-driven:
      1. dtttdt + auxiliary lag frames (§3.9.1).
      2. period_block / _succ / _time: the timeset blocks of bind_intraperiod_blocks nodes (§3.9.3).
      3. nodeStateBlock: the bind_intraperiod_blocks nodes (§3.9.2).
      4. nodeState_last_dt.
      5. p_state_existing_capacity / p_state_upper (§3.9.5).
      6. storage_use_reference_value (§3.9.6).
      7. p_roll_continue_state / p_fix_storage_quantity (§3.9.7).
      8. dtt_timeline_matching / period_branch (§3.9.8).

    The coarse new_stepduration groups are not handled here: ``load_flextool`` derives their frames for every
    solve (``params._coarse.coarse_node_frames``).
    """
    from flextool.engine.preprocess.params._block import load_block_bundle, nodeState_last_dt_lf

    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    nodeState_df = getattr(flex_data, "nodeState", None)
    has_state = (nodeState_df is not None
                  and getattr(nodeState_df, "height", 0) > 0)
    # Δ.9 — single BlockBundle per solve (replaces 4× CSV reads).
    # No block frames (fixtures without explicit blocks) → ``None``; any
    # other failure (including a missing provider) propagates.
    # Phase 2 multi-block fast-path: thread the in-memory layout (built
    # source-only by the fast loader and stashed on ``FlexData``) so
    # ``load_block_bundle`` returns a populated bundle even when the
    # workdir's solve_data/ is empty.
    fd_layout = getattr(flex_data, "block_layout", None)
    block_bundle = load_block_bundle(
        workdir, block_layout=fd_layout, provider=provider,
    )

    # Δ.12b — assignment is unconditional for storage-only Params and
    # the rolling-handoff carriers.  ``None`` is the explicit
    # "feature inactive / no handoff state" signal.

    # 1. dtttdt -------------------------------------------------------
    # Δ.12c-fix2 gap #5 close — ``_dt_period_active_steps`` now falls
    # back to reading ``solve_data/steps_in_use.csv`` for synthetic
    # rolling-horizon / nested-invest sub-solves whose names aren't in
    # the Spine DB.  The fallback uses ``timeline.timestep_duration``
    # for rank/timeline assignment.  Conditional assignment is retained
    # because fixtures genuinely lacking timeline data on the source AND
    # the workdir CSV (very rare; not present in the engine
    # corpus) still need to fall through to the seed-loaded value.
    dtttdt_jump = dtttdt_with_jump_from_source(source, active_solve, workdir, ctx=ctx, provider=provider,
                                               step_duration=getattr(flex_data, "p_step_duration", None))
    dtttdt_db = None if dtttdt_jump is None else dtttdt_jump.drop("jump")
    if dtttdt_db is not None:
        flex_data.dtttdt = dtttdt_db
        # Ramp predecessors: the adjacent (jump == 1) rows, without representative-block starts; the online
        # stretch wrap is derived from them (ramp_link_fields sets both).
        from flextool.engine.preprocess.params._ramp_links import ramp_link_fields, ramp_link_from_jump
        for _name, _value in ramp_link_fields(
                flex_data.dt, ramp_link_from_jump(dtttdt_jump, _rp_block_first_utf8(workdir, provider))).items():
            setattr(flex_data, _name, _value)
    # Minimum up/down-time look-backs along the final ramp predecessor links (the seed links when the source
    # timeline walk gives none).
    apply_mintime_lookbacks(flex_data, source, _roll_first_step_link(workdir, provider), active_solve)

    # 2. period_block family (storage-only). --------------------------
    pb_family = None
    if has_state:
        pb_family = period_block_family_from_source(
            source, active_solve, workdir, ctx=ctx, provider=provider,
            realized=_rolling_realized_steps(flex_data, workdir, provider))
    period_block_time_for_arc: pl.DataFrame | None = None
    if pb_family is not None:
        flex_data.period_block = pb_family["period_block"]
        flex_data.period_block_succ = pb_family["period_block_succ"]
        flex_data.period_block_time = pb_family["period_block_time"]
        period_block_time_for_arc = pb_family["period_block_time"]

    # 1b. dtttdt-derived lag frames (storage-only). ------------------
    if has_state:
        fwd = dtttdt_forward_only_from_dtttdt(
            dtttdt_db, read_period_first_of_solve(Path(workdir) / "solve_data", provider=provider)
            if workdir is not None else None)
        if fwd is not None:
            # The per-solve forward-only set counts too: it holds the fix_start_end nodes whose cyclic method the
            # solve ignores (_run_model._fix_start_end_chains_forward), which the DB does not call forward-only.
            sbm_full = _node_storage_binding_method_with_fallback(source)
            per_solve_fo = getattr(flex_data, "storage_bind_forward_only", None)
            if ((sbm_full is not None
                    and sbm_full.filter(
                        pl.col("method") == "bind_forward_only").height > 0)
                    or (per_solve_fo is not None and per_solve_fo.height > 0)):
                flex_data.dtttdt_forward_only = fwd
        # bind_within_solve: the first step of each root of a fan in the solve's first period continues from the
        # last step of its own path (the state-change lag of the bind_within_solve nodes).  This value is
        # authoritative: it overwrites the seed value of flex_data._load_storage, which uses the same function and
        # must agree with it.
        ws_nodes = getattr(flex_data, "storage_bind_within_solve", None)
        flex_data.dtttdt_within_solve = (
            dtttdt_within_solve_from_dtttdt(
                getattr(flex_data, "dtttdt", None),
                read_period_first_of_solve(Path(workdir) / "solve_data", provider=provider)
                if workdir is not None else None)
            if ws_nodes is not None and ws_nodes.height > 0 else None)
        flex_data.dtttdt_block_interior = dtttdt_block_interior_from_dtttdt(
            dtttdt_db, period_block_time_for_arc)
        flex_data.dtttdt_block_lag = dtttdt_block_lag_from_dtttdt(
            getattr(flex_data, "dtttdt", None), period_block_time_for_arc,
            getattr(flex_data, "period_block_chain", None))

    # 3. nodeStateBlock = the bind_intraperiod_blocks nodes (storage-only) -
    nsb_db = None
    if has_state:
        nsb_db = nodeStateBlock_from_source(source)
        flex_data.nodeStateBlock = nsb_db

    # 4. nodeState_last_dt: the default block's last step of each period (every node lives on the solve's
    # steps; a coarse node's state is constant within its last coarse step).
    if has_state and block_bundle is not None:
        nsld = nodeState_last_dt_lf(nodeState_df, block_bundle).collect()
        if nsld.height > 0:
            flex_data.nodeState_last_dt = nsld

    # 5. p_state_existing_capacity / p_state_upper --------------------
    if has_state:
        # A4-followup (BUG fix): pass ``provider`` so the helper can
        # read the workdir ``period__branch.csv`` / ``period_in_use_set.csv``
        # for stochastic-branch broadcast.  Without the provider the
        # Provider-only ``_provider_has_key`` returns False (Step 2.5
        # removed the disk-fallback), ``ctx`` is typically empty at this
        # cascade stage, and the helper falls through to the
        # ``solve.realized_periods`` set — which does NOT include
        # stochastic-branch siblings.  Result: ``p_state_existing_capacity``
        # carries only the realized-period row → downstream constraints
        # that join on (n, d) lose branch rows, e.g.
        # ``storage_state_solve_horizon_reference_value`` becomes an
        # empty Param for branches and the LP leaves their v_state at 0
        # while the realized branch is properly pinned.
        pse = p_state_existing_capacity_from_source(
            source, active_solve, nodeState_df, workdir,
            ctx=ctx, provider=provider)
        flex_data.p_state_existing_capacity = pse
        flex_data.p_state_upper = p_state_upper_from_source(
            source, pse, nodeState_df)
        # §F.2 p_state_unitsize (Δ.10 cluster F) — per-node unitsize
        # restricted to nodeState.
        from flextool.engine.preprocess.params._arithmetic import p_state_unitsize_from_source
        flex_data.p_state_unitsize = p_state_unitsize_from_source(
            source, nodeState_df)

    # 6. storage_use_reference_value (storage-only) -------------------
    if has_state:
        use_ref = storage_use_reference_value_from_source(
            source, nsb_db, rolling=bool(getattr(flex_data, "solve_is_rolling", False)))
        # The solve's plain cyclic sets come from the per-solve frame, so they include the blended-weights
        # nodes this solve downgraded (no representative periods): those cycle too and get no end pin.
        for cyclic in (flex_data.storage_bind_within_solve, flex_data.storage_bind_within_period):
            if use_ref is not None and cyclic is not None and cyclic.height > 0:
                names = [str(n) for n in cyclic["n"].to_list()]
                use_ref = use_ref.filter(~pl.col("n").cast(pl.Utf8).is_in(names))
        flex_data.storage_use_reference_value = use_ref if use_ref is not None and use_ref.height > 0 else None

    # 7. rolling-handoff (storage-only — read-from-disk handoff) ------
    # Path B Cat A: ``_load_storage`` (flex_data.py:2083-2104, 2106-2133)
    # already reads ``p_roll_continue_state.csv`` and
    # ``fix_storage_quantity.csv`` and assigns the resulting Params to
    # ``flex_data.p_roll_continue_state`` / ``p_fix_storage_quantity``
    # at load time.  Re-reading the same CSV here duplicates the IO and
    # the polars allocation.  Audit `specs/in_cascade_csv_audit.md`
    # §Category A entries `_derived.py:6562` and `:6582`.
    # Prefer the already-populated FlexData fields when available; fall
    # back to the workdir reader for legacy/test callers that bypass
    # ``_load_storage`` or when the seed produced None (e.g. CSV absent
    # at seed time but produced later by a prior solve).
    if has_state:
        if getattr(flex_data, "p_roll_continue_state", None) is None:
            flex_data.p_roll_continue_state = p_roll_continue_state_from_handoff(provider=provider)
        if getattr(flex_data, "p_fix_storage_quantity", None) is None:
            flex_data.p_fix_storage_quantity = p_fix_storage_quantity_from_handoff(provider=provider)
        flex_data.dtt_timeline_matching = dtt_timeline_matching_from_workdir(workdir, provider=provider)
        flex_data.period_branch = period_branch_from_source(source, workdir, ctx=ctx, provider=provider)




# ===========================================================================
# Γ.3.F — Lifetime cascade + handoff state + multi-year inflation cascade
# ===========================================================================
#
# This batch lands the §3.1.3 multi-year inflation cascade in full
# (replacing Γ.3.A's simple-1-year-per-period subcase), the §3.7.5/6
# discounted lifetime-fixed-cost / annual-discounted-invest annuity
# families, and the §3.7.7/8 rolling-handoff Param read side.  It is
# the dependency-final piece of the §3.7 + §3.11 cohort.
#
# Per the architectural shift (no defensive feature gating), helpers
# return ``None`` only when the underlying source is missing or the
# fixture's invest/lifetime configuration is degenerate (no
# entityInvest, no lifetime_method, etc.).  When data is present, the
# helper produces the canonical frame; the parity test catches any
# discrepancy.
#
# Authoritative algorithm references:
#   * §3.1.3 (multi-year cascade): preprocessing/period_calculated_params.py:200-322.
#   * §3.7.5/6 (lifetime + annual_discounted): preprocessing/entity_annual_calc_params.py:105-348.
#   * §3.7.7/8 (handoff state): preprocessing/entity_period_calc_params.py:1525-1600.


# ---------------------------------------------------------------------------
# §3.1.3 — multi-year inflation cascade (full version)
# ---------------------------------------------------------------------------


_LIFETIME_METHOD_DEFAULT_FALLBACK = "reinvest_automatic"


def _solve_inflation_inputs(source: "InputSource") -> tuple[float, float, float]:
    """Read the three scalar inflation inputs from ``model``.

    Returns ``(rate, offset_invest, offset_ops)``.  Mirrors flextool's
    ``_scalar_max`` over ``p_inflation_*.csv``:
        * rate              → 0.0 when no explicit Spine value
        * offset_invest     → 0.0 when no explicit Spine value
        * offset_operations → 0.5 when no explicit Spine value

    flextool's input_writer skips Spine *defaults* (only writes
    explicit parameter rows to ``input/p_inflation_*.csv``), so the
    downstream cascade reads ``_scalar_max(... default)`` and falls back
    to the constant defaults documented above.  We mirror that: when
    the SpineDbReader returns only the broadcast-default row (no
    explicit per-entity row), we fall back to flextool's defaults.

    We can't tell from a returned frame whether a value came from
    explicit storage or default broadcast; we use ``parameter_default``
    plus row count to detect the all-default case.  When the default
    matches every row's value AND no explicit per-class row exists,
    treat as "no explicit" (use flextool's CSV default).
    """
    def _explicit_max(par: str, default: float) -> float:
        df = _try_param(source, "model", par)
        if df is None or df.height == 0:
            return default
        # Detect "all rows = Spine default broadcast" case.  When the
        # Spine reader's default-fill kicked in, every row's value
        # equals the parameter_default and the CSV writer would have
        # produced an empty file.  Fall back to flextool's constant.
        try:
            spine_default = source.parameter_default("model", par)
        except KeyError:
            spine_default = None
        if spine_default is not None:
            try:
                spine_default_f = float(spine_default)
            except (ValueError, TypeError):
                spine_default_f = None
            if spine_default_f is not None:
                # If every row equals the default, treat as no explicit data.
                vals = df["value"].cast(pl.Float64, strict=False).to_list()
                if vals and all(
                        (v is not None and abs(v - spine_default_f) < 1e-12)
                        for v in vals):
                    return default
        try:
            return float(df["value"].cast(pl.Float64).max())
        except Exception:
            return default

    rate = _explicit_max("inflation_rate", 0.0)
    off_inv = _explicit_max("inflation_offset_investment", 0.0)
    off_ops = _explicit_max("inflation_offset_operations", 0.5)
    return rate, off_inv, off_ops


def _years_for_period_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    period_set: list[str],
                                    *,
                                    provider: "object | None" = None,
                                    ) -> dict[str, list[tuple[str, float]]]:
    """Derive the per-period ``(year_label, width)`` list emitted into
    ``solve_data/p_years_represented.csv`` (matches the
    ``emit_years_represented`` output line-for-line).

    Algorithm:
      For each (period, R) in solve.years_represented[active_solve],
      with R > 0:
        rows = ceil(R), remaining = R, year_count starts at solve-global 0.
        For i in range(rows):
          width = min(1.0, remaining)
          emit (period, str(year_count), width)
          year_count += width; remaining -= width

    Two arms.  With ``provider`` (canonical): read
    ``solve_data/p_years_represented.csv``, which also copies the
    anchor period's rows to its stochastic branch periods
    (``period__branch``).  Without it (or when the CSV is absent): the
    source arm below, which does NOT expand to branch periods (a branch
    period gets no rows).  ``p_years_represented_d`` and the recourse
    walker pass ``provider``; the inflation and NPV callers still use the
    source arm, so a branch period gets no years there.

    Returns a dict ``{period: [(year_label, width), ...]}``.  When the
    parameter is absent or active_solve has no rows, default to one row
    per period in the order supplied (width=1, year_label = "0", "1", …).
    """
    # The emitted ``p_years_represented.csv`` already byte-copies
    # fan-member year rows (``derive_years_represented`` branch loop), so
    # read it directly to give branch periods their anchor's
    # (year_label, width) rows.  Callers that pass ``provider=None`` get
    # the source path below.
    if provider is not None:
        p = Path("solve_data") / "p_years_represented.csv"
        if _provider_has_key(provider, p):
            df = _provider_read(provider, p)
            need = {"period", "years_from_solve", "p_years_represented"}
            if df.height > 0 and need.issubset(df.columns):
                canon: dict[str, list[tuple[str, float]]] = {}
                for r in df.iter_rows(named=True):
                    canon.setdefault(str(r["period"]), []).append(
                        (str(r["years_from_solve"]),
                         float(r["p_years_represented"])))
                if canon:
                    return canon

    yr_p = solve_years_represented(source)
    out: dict[str, list[tuple[str, float]]] = {}
    if active_solve is None or yr_p is None:
        # Default fallback — one year per period, width 1, integer-indexed.
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Rolling solves: ``active_solve`` is the per-roll name
    # (``<parent>_roll_<N>``) but the DB has the parameter on the parent
    # solve.  flextool's preprocessing copies the parent value down to
    # each roll.  Try the active_solve first, then the parent name.
    sub = yr_p.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        import re
        parent = re.sub(r"_roll_\d+$", "", active_solve)
        if parent != active_solve:
            sub = yr_p.filter(pl.col("name") == parent)
    if sub.height == 0:
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Iterate in CSV order — flextool walks the Spine Map order, which
    # the Spine reader presents row-by-row.  We sort by period to match
    # the canonical ordering when periods aren't already sorted in the
    # source — flextool's solve_writers iterates the dict in insertion
    # order, which mirrors the periodAll set ordering.  For our purposes
    # the period_set arg gives the canonical order.
    rows_by_d: dict[str, float] = {}
    for r in sub.iter_rows(named=True):
        rows_by_d[str(r["period"])] = float(r["value"])

    import math
    year_count = 0.0
    for d in period_set:
        R = rows_by_d.get(d, 0.0)
        if R <= 0:
            out[d] = []
            continue
        rows = int(math.ceil(R))
        remaining = R
        emitted: list[tuple[str, float]] = []
        for _i in range(rows):
            width = min(1.0, remaining)
            # Mirror solve_writers' int-cast path: when year_count is an
            # integer (R = whole numbers) we emit ``str(int(yc))``
            # literals; with fractional widths flextool writes the
            # float-stringified value.  Emulate the same:
            if year_count == float(int(year_count)):
                lbl = str(int(year_count))
            else:
                lbl = str(year_count)
            emitted.append((lbl, width))
            year_count += width
            remaining -= width
        out[d] = emitted
    return out


def _periodAll_from_source(source: "InputSource",
                              active_solve: str | None,
                              *,
                              provider: "object | None" = None,
                              workdir: Path | None = None) -> list[str]:
    """Compute the periodAll set for the active solve.

    Mirrors flextool's ``periodAll_set.csv`` shape.  When ``workdir``
    is provided AND the per-solve preprocessing has already written
    the file, prefer that (it's the authoritative cumulative-history
    union for chained solves).  Falls back to the simple
    realized ∪ invest union from the source when the file is absent.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "periodAll_set.csv"
        df = None
        if provider is not None and provider.has(_provider_key(p)):
            df = provider.get(_provider_key(p))
        if df is not None and df.height > 0 and "period" in df.columns:
            return df["period"].cast(pl.Utf8, strict=False).to_list()
    seen: dict[str, None] = {}
    if active_solve is None:
        return []
    for par in ("realized_periods", "invest_periods"):
        ps = _solve_periods(source, active_solve, par) or []
        for d in ps:
            seen.setdefault(d, None)
    return list(seen.keys())


def _inflation_yearly_from_source(source: "InputSource",
                                     active_solve: str | None,
                                     workdir: Path | None = None,
                                     *,
                                     provider: "object | None" = None,
                                     ) -> tuple[dict[str, float],
                                                 dict[str, float]] | None:
    """Compute ``p_inflation_factor_investment_yearly[d]`` and
    ``p_inflation_factor_operations_yearly[d]`` per
    ``period_calculated_params.py:200-322``.

    Returns ``(invest_factor, ops_factor)`` dicts keyed on period.
    Returns ``None`` only when the periodAll set is empty.

    Algorithm:
      1. ``inflation_rate``, ``inflation_offset_invest/ops`` from model.
      2. For each period ``d`` ∈ periodAll, build ``years_for_period[d]``
         via :func:`_years_for_period_from_source`.
      3. Build the GLOBAL year set = union of all year labels across
         all periods, sort numerically (or lexically when non-numeric).
      4. For each period ``d``:
         a. Walk the global year set computing
            ``global_pos[y2] = sum_{y' < y2 in global order} pyr[d, y']``
            (inner accumulator restricted to the period's row).
         b. For each y in years_for_period[d]:
              base = global_pos[y]
              until_inv = base + pyr[d, y] * offset_invest
              until_ops = base + pyr[d, y] * offset_ops
         c. ``inv_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_inv)``
            ``ops_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_ops)``
            When ``sum_y pyr[d, y] == 0`` → factor = 1.0.
    """
    period_universe = _periodAll_from_source(source, active_solve, workdir=workdir, provider=provider)
    if not period_universe:
        return None
    rate, off_inv, off_ops = _solve_inflation_inputs(source)
    one_plus_inflation_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    yfp = _years_for_period_from_source(source, active_solve,
                                          period_universe)

    # Global year set — union of all year labels.
    global_year_set: dict[str, None] = {}
    for years_list in yfp.values():
        for y, _w in years_list:
            global_year_set.setdefault(y, None)
    try:
        sorted_global_years = sorted(global_year_set.keys(),
                                       key=lambda y: float(y))
    except ValueError:
        sorted_global_years = sorted(global_year_set.keys())

    inflation_invest: dict[str, float] = {}
    inflation_ops: dict[str, float] = {}

    # Build a fast lookup: pyr_lookup[(d, y)] = width.
    pyr_lookup: dict[tuple[str, str], float] = {}
    for d, years in yfp.items():
        for y, w in years:
            pyr_lookup[(d, y)] = w

    for d in period_universe:
        years_for_d = yfp.get(d, [])
        try:
            sorted_d_years = sorted(years_for_d, key=lambda yw: float(yw[0]))
        except ValueError:
            sorted_d_years = sorted(years_for_d, key=lambda yw: yw[0])

        # global_pos[y2] = sum_{y' in global, y' < y2} pyr[d, y'] (default 1)
        cumulative = 0.0
        global_pos: dict[str, float] = {}
        for y2 in sorted_global_years:
            global_pos[y2] = cumulative
            cumulative += pyr_lookup.get((d, y2), 1.0)

        sum_p_years_for_d = sum(w for _y, w in sorted_d_years)
        if sum_p_years_for_d > 0:
            inv_factor = 0.0
            ops_factor = 0.0
            for y, w in sorted_d_years:
                base = global_pos.get(y, 0.0)
                until_inv = base + w * off_inv
                until_ops = base + w * off_ops
                inv_factor += w * (one_plus_inflation_inv ** until_inv)
                ops_factor += w * (one_plus_inflation_inv ** until_ops)
            inflation_invest[d] = inv_factor
            inflation_ops[d] = ops_factor
        else:
            inflation_invest[d] = 1.0
            inflation_ops[d] = 1.0

    return inflation_invest, inflation_ops


# ---------------------------------------------------------------------------
# §3.7.7/8 — handoff state (rolling solve-chain feeds)
# ---------------------------------------------------------------------------


def _read_handoff_e_d_from_workdir(workdir: Path,
                                       name: str,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> "Param | None":
    """Read a (entity, period, value) handoff CSV; mirror the loader's
    zero-filter (flex_data.py:1316-1328).  Returns None when the file is
    absent / empty / all-zero.
    """
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "d", "value"))
    if df.height == 0:
        return None
    return Param(("e", "d"), df)


def _read_handoff_e_from_workdir(workdir: Path, name: str,
                                    value_col: str | None = None,
                                    *,
                                    provider: "object | None" = None,
                                    ) -> "Param | None":
    """Wide-format (entity, p_entity_invested) → Param ((e,), value)."""
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    if value_col is None:
        non = [c for c in df.columns if c != "entity"]
        if not non:
            return None
        value_col = non[0]
    df = (df.pipe(rename_to_axis, {"entity": "e", value_col: "value"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


def p_entity_previously_invested_capacity_from_workdir(
        workdir: Path,
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-solve handoff carrier (§3.7.7).  Read-side mirror of the
    chain runner's write-side: ``solve_data/p_entity_previously_invested_capacity.csv``
    holds the cumulative prior-solve invest summed across the entity's
    historical periods.  In single-solve fixtures this CSV is
    header-only (or all-zero), and the Param collapses to ``None``.

    For chain runs, the chain runner is responsible for writing the
    fresh CSV between sub-solves (or for calling :func:`apply_handoff`
    with the in-memory carrier).  This helper only reads from disk —
    it doesn't manufacture state.
    """
    return _read_handoff_e_d_from_workdir(
        workdir, "p_entity_previously_invested_capacity", provider=provider)


def p_entity_invested_from_handoff(
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve invest (§3.7.8).

    Phase 4.2-1a — read the realized-invest carrier from the canonical
    ``handoff/realized_invest`` Provider key (schema
    ``[entity, period, value]``) and sum over period to produce the
    per-entity scalar ``Param(("e",), [(e, value)])``.  The translator
    populates this key from :pyattr:`SolveHandoff.realized_invest` at
    iteration start; absent / header-only / all-zero collapses to
    ``None``.  Mirrors the overlay logic at ``flex_data.py:5464-5476``.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_REALIZED_INVEST)
    if df is None or df.height == 0:
        return None
    df = (df
          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                .fill_null(0.0))
          .group_by("entity").agg(pl.col("value").sum())
          .pipe(rename_to_axis, {"entity": "e"})
          .filter(pl.col("value") != 0.0)
          .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


def p_entity_divested_from_handoff(
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve divest (§3.7.8).

    Phase 4.2-1a — read the divest-cumulative carrier from the canonical
    ``handoff/divest_cumulative`` Provider key (schema
    ``[entity, value]``).  The translator populates this key from
    :pyattr:`SolveHandoff.divest_cumulative` at iteration start;
    absent / header-only / all-zero collapses to ``None``.  Mirrors
    the overlay logic at ``flex_data.py:5478-5487``.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_DIVEST_CUMULATIVE)
    if df is None or df.height == 0:
        return None
    df = (df
          .pipe(rename_to_axis, {"entity": "e"})
          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                .fill_null(0.0))
          .filter(pl.col("value") != 0.0)
          .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


# ---------------------------------------------------------------------------
# §3.7.5/6 — Lifetime fixed cost + entity annual discounted (annuity)
# ---------------------------------------------------------------------------


def _entity_classes_lookup(source: "InputSource"
                              ) -> tuple[set[str], set[str], set[str]]:
    """Return ``(process_set, node_set, connection_set)`` of entity names.

    Mirrors flextool's input/process.csv = unit ∪ connection (see
    ``input_writer.py:42``); the distinct ``unit`` and ``connection``
    classes both feed into the shared ``process_set`` consumed by
    ``entity_period_calc_params._ed_value`` and the annuity sums.
    The ``connection_set`` is kept separate so callers can resolve
    connection-only attributes (e.g. ``is_DC``) when needed.
    """
    def _set(ec: str) -> set[str]:
        df = _try_entities(source, ec)
        if df is None:
            return set()
        return set(df["name"].to_list())
    unit_s = _set("unit")
    conn_s = _set("connection")
    process_s = unit_s | conn_s
    return process_s, _set("node"), conn_s


def _per_entity_period_value(source: "InputSource",
                                  parameter_name: str,
                                  ) -> dict[tuple[str, str], float] | None:
    """Read ``unit/node/connection.<parameter_name>`` and return a
    ``(entity, period) → value`` dict.  Scalar shapes broadcast to all
    periods downstream; this helper itself only emits explicit rows.

    A separate ``_get_value(e, d)`` helper layers the broadcast.  A
    stochastic branch period inherits its anchor's explicit row here
    (unconditionally, not only under recourse — #63); without it
    ``_lifetime_gate_sums`` sees a ``lifetime`` of 0 (not the anchor's)
    in every branch period.
    """
    out: dict[tuple[str, str], float] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "period" in cols:
            df = inherit_active_branch_periods(
                df, group_keys=("name",), period_col="period")
            for r in df.iter_rows(named=True):
                try:
                    out[(str(r["name"]), str(r["period"]))] = \
                        float(r["value"])
                except Exception:
                    continue
    return out if out else None


def _per_entity_scalar(source: "InputSource",
                          parameter_name: str,
                          ) -> dict[str, float]:
    """Per-entity scalar (no period dim) from
    unit/node/connection.<parameter_name>.  Returns ``{}`` when none.
    """
    out: dict[str, float] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, parameter_name)
        if df is None:
            continue
        cols = df.columns
        if "period" in cols:
            continue
        for r in df.iter_rows(named=True):
            try:
                out[str(r["name"])] = float(r["value"])
            except Exception:
                continue
    return out


def _entity_invest_methods(source: "InputSource"
                              ) -> dict[str, list[str]]:
    """Return ``entity → [invest_method, ...]`` from
    ``entity__invest_method.csv``.  Spine source: per-entity-class
    ``invest_method`` Map, scalar-valued; flextool keeps the per-entity
    list (flextool.mod allows multi-method but every concrete model
    has 1).
    """
    out: dict[str, list[str]] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            out.setdefault(str(r["name"]), []).append(str(r["value"]))
    return out


def _entity_lifetime_methods(source: "InputSource",
                                  all_entities: list[str],
                                  ) -> dict[str, list[str]]:
    """Return ``entity → [lifetime_method, ...]`` for every entity.

    Default (when an entity has no explicit lifetime_method row) is
    ``["reinvest_automatic"]`` per
    ``preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT``.
    """
    explicit: dict[str, list[str]] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "lifetime_method")
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            explicit.setdefault(str(r["name"]), []).append(str(r["value"]))

    out: dict[str, list[str]] = {}
    for e in all_entities:
        if e in explicit:
            out[e] = list(explicit[e])
        else:
            out[e] = [_LIFETIME_METHOD_DEFAULT_FALLBACK]
    return out


def _all_entities(source: "InputSource") -> list[str]:
    """Union of unit + node + connection entity names, in flextool's
    canonical iteration order (concatenation, dedup with first-wins).
    """
    seen: dict[str, None] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_entities(source, ec)
        if df is None:
            continue
        for n in df["name"].to_list():
            seen.setdefault(str(n), None)
    return list(seen.keys())


def _resolve_pdX(per_period: dict[tuple[str, str], float] | None,
                    scalar: dict[str, float],
                    e: str,
                    d: str) -> float:
    """Resolve `pdX[e, param, d]` mirroring flextool's PdLookup:
       1. (e, d) explicit row → that value.
       2. else (e, ·) scalar  → broadcast.
       3. else                → 0.0 (mod default).
    """
    if per_period is not None:
        v = per_period.get((e, d))
        if v is not None:
            return v
    return scalar.get(e, 0.0)


def _p_discount_years_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       ) -> dict[str, float]:
    """Derive ``solve_data/p_discount_years.csv`` (matches the
    ``emit_period_years`` output).

    Algorithm: for each period in solve.years_represented[active_solve]
    (iteration order = Spine Map order = canonical period_set order),
    emit ``(period, year_count)`` then ``year_count += R[d]``.

    Periods absent from years_represented (or with R=0) get value 0.0
    via the consumer's ``.get(d, 0.0)`` default.

    Returns ``{period: cumulative_years}`` dict.
    """
    out: dict[str, float] = {}
    if active_solve is None:
        return out
    yrp = solve_years_represented(source)
    if yrp is None:
        return out
    sub = yrp.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        return out
    # Spine Map iteration order — preserve the row order from the source.
    # The Spine reader returns rows in DB-storage order which mirrors the
    # canonical Map index order.
    year_count = 0.0
    for r in sub.iter_rows(named=True):
        d = str(r["period"])
        out[d] = year_count
        try:
            year_count += float(r["value"])
        except Exception:
            pass
    return out


def _read_period_with_history(workdir: Path | None, *, provider: "object | None" = None) -> list[str]:
    """Read ``solve_data/period_with_history.csv`` from disk if present.

    For multi-solve chain runs flextool's per-solve preprocessing
    (``orchestration.py::main`` block at L262) writes this CSV with the
    accumulated period universe from all prior solves.  When the file
    is absent (e.g. tests using only the input/ side), the caller's
    fallback to ``period_in_use`` is used.

    Returns ``[period, ...]`` in CSV row order.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period_with_history.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return []
    df = _provider_read(provider, p)
    if df.height == 0 or "period" not in df.columns:
        return []
    return df["period"].cast(pl.Utf8, strict=False).to_list()


# ---------------------------------------------------------------------------
# Integration: apply_derived_f
# ---------------------------------------------------------------------------


def apply_derived_f(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.F lifetime cascade family + handoff state + multi-year
    inflation cascade, mutating ``flex_data`` in place.

    Order:
      1. Cluster A (Δ.5): ``p_inflation_op``, ``p_ed_fixed_cost``,
         ``ed_entity_annual_discounted``, ``ed_entity_annual_divest_discounted``,
         ``ed_lifetime_fixed_cost``, ``ed_lifetime_fixed_cost_divest``
         — delegated to :func:`._npv.apply_npv`, the lazy-polars
         port.  Replaces the eager ``ed_entity_annual_family_from_source``
         and ``p_inflation_op_full_cascade_from_source`` paths.
      2. ``p_entity_previously_invested_capacity`` / ``p_entity_invested``
         / ``p_entity_divested`` (handoff state).

    Δ.3 replaced the previous ``derived_overrides_f`` dict-return;
    Δ.4 deleted the deprecated wrapper alias; Δ.5 ports the lifetime
    cascade family to lazy polars.
    """
    from flextool.engine.preprocess.params import _npv as _derived_npv

    # 1. Cluster A (lazy NPV / inflation / fixed cost cascade).
    _derived_npv.apply_npv(flex_data, source, workdir, provider=provider)

    # 2. Handoff state -----------------------------------------------
    # Δ.12b — unconditional assignment.  Each helper returns ``None``
    # when the corresponding handoff CSV is missing / header-only
    # (single-solve fixtures, or first-of-chain) — ``None`` is the
    # explicit "no prior-solve state" signal.
    flex_data.p_entity_previously_invested_capacity = (
        p_entity_previously_invested_capacity_from_workdir(
            workdir, provider=provider))
    flex_data.p_entity_invested = p_entity_invested_from_handoff(
        provider=provider)
    flex_data.p_entity_divested = p_entity_divested_from_handoff(
        provider=provider)




# ===========================================================================
# Γ.3.G — Residual Derived Params (audit §3.13/3.15/3.17/3.18 + §3.6.1 cascade)
#
# Final batch.  Closes the Derived class:
#
#   * §3.17 commodity ladder (`p_f_d_k`, `p_ladder_cum_realized_mwh`)
#   * §3.13 reserves remainder (`prundt`)
#   * §3.15 delay (`dtt__delay_duration`, `p_process_delay_weight`)
#   * §3.18 multi-branch normalisation (full cascade for both
#     ``pdt_branch_weight`` and ``pd_branch_weight``)
#   * §3.6.1 stochastic 3d_map profile cascade (branch-keyed alternative
#     resolution for ``p_profile_value``)
#
# No defensive gating — helpers either produce the canonical frame or
# return None when the structural source data is genuinely absent (e.g.
# a fixture without commodity-ladder rows).  Architectural shift per the
# task spec: the CSV path is the debug oracle, not a runtime fallback;
# parity test failures are the right signal.
# ===========================================================================


# ---------------------------------------------------------------------------
# §3.17 — Commodity ladder
# ---------------------------------------------------------------------------


def p_f_d_k_from_source(
    source: "InputSource",
    active_solve: str | None,
    workdir: Path | None,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> "Param | None":
    """Per-period "fraction realised this solve" for the price-ladder
    feature (audit §3.17.1).

    Algorithm — mirror of
    ``preprocessing/period_calculated_params.py:324-344``::

        f_d_k[d] = (p_ladder_cum_sim_hours[α(d)] + sum_t step_duration[d, t])
                 / (complete_period_share_of_year[d] * 8760)

    α(d) is the period label of a forecast branch period d (d itself otherwise, or when d has a
    carry row of its own): the realization solve carries the realized hours under the label, and every
    forecast continues from its period's realized history (``branch_periods_inherit_carry``).

    Domain: ``period_in_use``, plus one row per period that the chain has already finished and
    dropped from ``period_in_use`` (value 1.0 — a period only leaves ``period_in_use`` once it is
    fully realised, so its f_d_k is exactly 1.0 from then on).  Carrying those rows forward instead
    of letting the period vanish from the domain is what lets ``_eq_tier_cap_cumulative``'s
    ``Σ_d f_d_k`` accumulate monotonically across period boundaries, rather than resetting to the
    new period's own local progress each time the chain advances (#143).  Inputs:

    * ``solve_data/steps_in_use.csv`` — per-(d, t) step duration sum.
    * ``solve_data/ladder_cum_sim_hours.csv`` — handoff carrier; default
      0.0 when row absent.
    * ``solve_data/complete_period_share_of_year_calc.csv`` — per-period
      hour share of the year (already-computed §3.1.4 frame).

    Returns ``None`` unless the price-ladder feature is active (i.e. at
    least one commodity has ``price_method ∈ {price_ladder_annual,
    price_ladder_cumulative}`` — flextool emits ``f_d_k.csv`` always
    but ``engine/model/_commodity_ladder.py:load`` only consumes it when
    ladder commodities exist; the parity test mirrors that semantic).
    """
    if workdir is None:
        return None
    # Gate on ladder activity — mirrors the CSV-side behaviour
    # where the loader only emits p_f_d_k when there are ladder
    # commodities.
    sd = Path(workdir) / "solve_data"
    cwl_path = sd / "commodity_with_ladder.csv"
    if _provider_has_key(provider, cwl_path):  # Phase E-j — seed-aware (no-op: writer not in expected_basenames)
        cwl = _provider_read(provider, cwl_path)
        if cwl.height == 0:
            return None
    else:
        # Fall back to source data: any commodity with a non-default
        # price_method counts as ladder-active.
        method = _try_param(source, "commodity", "price_method")
        if method is None or method.height == 0:
            return None
        ladder = method.filter(pl.col("value") != "price")
        if ladder.height == 0:
            return None
    # Path B Cat B (WriterSnapshot top-7): prefer SolveContext's typed
    # ``steps_in_use`` and ``period_share_of_year`` accessors when
    # available — already in canonical ``d`` / ``t`` / ``step_duration``
    # form, so we skip the per-call rename + select allocation.  Falls
    # back to the CSV-read path for legacy callers without ``ctx``.
    if ctx is not None and ctx.steps_in_use.height > 0:
        siu_df = ctx.steps_in_use
        if "d" not in siu_df.columns or "step_duration" not in siu_df.columns:
            return None
        sum_step = (siu_df.lazy()
                      .group_by("d")
                      .agg(pl.col("step_duration").sum().alias("sum_step")))
    else:
        siu = sd / "steps_in_use.csv"
        if not _provider_has_key(provider, siu):  # Phase E-j — seed-aware
            return None
        df = _provider_read(provider, siu)
        if df.height == 0 or "period" not in df.columns:
            return None
        sum_step = (df.lazy()
                      .with_columns(pl.col("step_duration")
                                      .cast(pl.Float64, strict=False))
                      .group_by("period")
                      .agg(pl.col("step_duration").sum().alias("sum_step"))
                      .pipe(rename_to_axis, {"period": "d"}))
    # complete_period_share_of_year[d]
    if ctx is not None and ctx.period_share_of_year.height > 0:
        csy = (ctx.period_share_of_year.lazy()
                  .with_columns(pl.col("value").alias("share")))
    else:
        csy_path = sd / "complete_period_share_of_year_calc.csv"
        if not _provider_has_key(provider, csy_path):  # Phase E-j — seed-aware
            # Try the non-_calc variant (some fixtures only emit the latter).
            csy_path = sd / "complete_period_share_of_year.csv"
        if not _provider_has_key(provider, csy_path):  # Phase E-j — seed-aware (no-op for non-_calc variant)
            return None
        csy = (_provider_read(provider, csy_path).lazy()
                  .pipe(rename_to_axis, {"period": "d"})
                  .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                      .alias("share")))
    # ladder_cum_sim_hours — default 0.0 when absent.
    # Phase 4.1a / 4.2-1g — the cross-roll cumulative-sim-hours accumulator
    # rides on ``handoff/cum_sim_hours`` (canonical schema
    # ``[period, p_ladder_cum_sim_hours]``).  The legacy
    # ``solve_data/ladder_cum_sim_hours`` CSV fallback was dropped in
    # 4.2-1g — the translator pipeline always populates the handoff key.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    cum_raw = read_handoff_frame(provider, K.HANDOFF_CUM_SIM_HOURS)
    if cum_raw is not None and cum_raw.height > 0 and \
            "p_ladder_cum_sim_hours" in cum_raw.columns:
        cum_lf = (cum_raw.lazy()
                   .pipe(rename_to_axis, {"period": "d",
                            "p_ladder_cum_sim_hours": "cum"})
                   .with_columns(pl.col("cum")
                                   .cast(pl.Float64, strict=False)))
        # A forecast branch period continues from its period's realized hours, written under the period
        # label (emit_period_calculated_params, the emitted twin): the numerator joins the inherited
        # carry; the fold of finished periods below keeps the raw keys.
        cum_join_lf = branch_periods_inherit_carry(
            cum_lf.collect(), _read_period_branch_pairs(workdir, provider=provider)).lazy()
    else:
        cum_lf = None
        cum_join_lf = None
    # period_in_use as the output domain — Path B Cat B: prefer
    # SolveContext's typed accessor (already canonical ``[d]``).
    if ctx is not None and ctx.period_in_use.height > 0:
        piu_lf = ctx.period_in_use.lazy().select("d")
    else:
        piu_path = sd / "period_in_use_set.csv"
        if not _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
            return None
        piu_lf = (_provider_read(provider, piu_path).lazy()
                    .pipe(rename_to_axis, {"period": "d"})
                    .select("d"))
    # Phase 4.8f: ctx-side (period_in_use, period_share_of_year,
    # steps_in_use) and CSV-side ``d`` columns may arrive in different
    # dtype regimes (Enum from SolveContext, Utf8 from raw CSV).  Defend
    # the cascade join keys by casting every contributing LazyFrame to
    # the canonical axis Enum vocabulary when active.
    _enums = get_global_axis_enums()
    if _enums is not None:
        piu_lf = cast_frame_axes(piu_lf, _enums)
        sum_step = cast_frame_axes(sum_step, _enums)
        csy = cast_frame_axes(csy, _enums)
        if cum_lf is not None:
            cum_lf = cast_frame_axes(cum_lf, _enums)
            cum_join_lf = cast_frame_axes(cum_join_lf, _enums)
    out_lf = (piu_lf
                .join(sum_step.select("d", "sum_step"), on="d", how="left")
                .join(csy.select("d", "share"), on="d", how="left"))
    if cum_lf is not None:
        out_lf = out_lf.join(cum_join_lf.select("d", "cum"), on="d", how="left")
        out_lf = out_lf.with_columns(pl.col("cum").fill_null(0.0))
    else:
        out_lf = out_lf.with_columns(cum=pl.lit(0.0))
    out_lf = (out_lf
                .with_columns(pl.col("sum_step").fill_null(0.0))
                .with_columns(value=(pl.col("cum") + pl.col("sum_step"))
                                       / (pl.col("share") * HOURS_PER_YEAR))
                .select("d", "value"))
    if cum_lf is not None:
        # A period leaves `period_in_use` only once the rolling chain has moved past it, which
        # (periods are visited in order, never revisited) only happens once it is fully realised —
        # at that point its frozen `cum_sim_hours` already equals its own complete hours, i.e. its
        # true f_d_k is exactly 1.0.  Folding those periods back in (instead of letting them drop
        # out of the domain) is what keeps Σ_d f_d_k monotone across period boundaries for
        # ladder_tier_cap_cumulative_roll, whose Σ_d f_d_k must accumulate over every period the
        # chain has ever touched, not reset to the current period's own local progress each time
        # the chain advances to a new one (#143's root cause).
        completed_lf = (cum_lf.select("d").unique()
                         .join(piu_lf.select("d"), on="d", how="anti")
                         .with_columns(value=pl.lit(1.0, dtype=pl.Float64)))
        out_lf = pl.concat([out_lf, completed_lf], how="vertical_relaxed")
    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("d",), out)


def p_ladder_cum_realized_mwh_from_workdir(
    workdir: Path | None,
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Per-(c, i, d) cumulative realized MWh handoff carrier
    (audit §3.17.2).

    This is a pure rolling-handoff state — written by the chain runner
    after each solve summing ``v_trade`` per tier into the next solve's
    starting accumulator.  Single-solve fixtures emit a header-only CSV
    (returns None).
    """
    if workdir is None:
        return None
    # Phase 4.1a / 4.2-1g — prior-roll accumulator now rides on
    # ``handoff/cumulative_commodity`` (canonical schema
    # ``[commodity, tier, period, p_ladder_cum_realized_mwh]``).  The
    # legacy ``solve_data/ladder_cum_realized_mwh`` CSV fallback was
    # dropped in 4.2-1g — the translator pipeline is the only source.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    raw = read_handoff_frame(provider, K.HANDOFF_CUMULATIVE_COMMODITY)
    if raw is None:
        return None
    if raw.height == 0:
        return None
    value_col = ("p_ladder_cum_realized_mwh"
                 if "p_ladder_cum_realized_mwh" in raw.columns else "value")
    if value_col not in raw.columns:
        return None
    out = (raw.pipe(rename_to_axis, {"commodity": "c", "tier": "i", "period": "d",
                         value_col: "value"})
              .with_columns(pl.col("i").cast(pl.Utf8))
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .select("c", "i", "d", "value"))
    if out.height == 0:
        return None
    return Param(("c", "i", "d"), out)


# ---------------------------------------------------------------------------
# §3.13 — prundt (residual reserve coverage)
# ---------------------------------------------------------------------------


def prundt_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """``prundt`` (p, r, ud, n, d, t): the reserve relationships of the reserves with a requirement × dt.

    A relationship ``(p, r, ud, n)`` (``process_reserve_upDown_node_active_from_source``) is kept when its
    ``(r, ud)`` has a requirement (:func:`~flextool.engine.preprocess._reserve_methods.demanded_reserves`):
    a ``reservation`` whose sum over the groups of ``(r, ud)`` and the solve's (d, t) is not 0, or a
    dynamic / n-1 group driven by an ``increase_reserve_ratio`` (of a relationship at a node of the group, or of the group itself) /
    ``large_failure_ratio``.  The solve_data producer
    (``emit/_reserve_sets._compute_process_reserve_active``) applies the same rule, so ``prundt`` and the
    ratio sets the model reads agree.  A reserve without a requirement gets no ``v_reserve`` columns
    (no all-zero columns in ``process__reserve__upDown__node__dt``).

    Returns an empty typed frame when dt, the relationships or every requirement are absent.
    """
    from flextool.engine.preprocess._reserve_methods import demanded_reserves
    from flextool.engine.preprocess.params import _projection as _proj

    # Empty typed prundt for the "feature active but no requirement" fall-through: ``_reserve._check``
    # expects a DataFrame (not None) when the reserve feature is active.  Downstream guards on
    # ``height == 0`` before touching values.
    def _empty_prundt() -> pl.DataFrame:
        return pl.DataFrame(schema={
            "p": pl.Utf8,
            "r": pl.Utf8,
            "ud": pl.Utf8,
            "n": pl.Utf8,
            "d": pl.Utf8,
            "t": pl.Utf8,
        })

    if dt is None or dt.height == 0:
        return _empty_prundt()
    pruna = process_reserve_upDown_node_active_from_source(source)
    if pruna is None or pruna.height == 0:
        return _empty_prundt()

    def _labels(df: pl.DataFrame, cols: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [tuple(str(x) for x in row)
                for row in df.select(pl.col(c).cast(pl.Utf8) for c in cols).iter_rows()]

    # 1. Reservation: (r, ud) whose reservation, summed over its groups and this solve's (d, t), is not 0
    #    (the solve-resolved values; a stochastic-time reservation contributes its fold) — the emit's rule.
    reservation_pairs: set[tuple[str, str]] = set()
    res = pdtReserve_upDown_group_reservation_from_source(source, period_filter=dt)
    if res is not None and res.frame.height > 0:
        # A value without a d / t axis (a constant) holds on every (d, t) of the solve.
        axes = [c for c in ("d", "t") if c in res.frame.columns]
        dt_keys = dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique().lazy()
        vals = res.frame.lazy().select(*(pl.col(c).cast(pl.Utf8) for c in ("r", "ud", *axes)),
                                       pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
        on_dt = vals.join(dt_keys, on=axes, how="inner") if axes else vals.join(dt_keys, how="cross")
        sums = (on_dt
                .group_by("r", "ud").agg(pl.col("value").sum())
                .filter(pl.col("value") != 0.0)
                .collect())
        reservation_pairs = set(_labels(sums, ("r", "ud")))
    # 2./3. Dynamic and n-1 groups and the ratios that drive them.
    group_methods: list[tuple[str, ...]] = []
    for part in ("dynamic", "n_1"):
        group_methods += _labels(_proj.reserve_upDown_group_method(source, part), ("r", "ud", "g", "method"))
    group_nodes = _labels(_proj.group_node(source), ("g", "n"))
    rel = _reserve_active_relationships(source)
    rel_keys = set(_labels(rel, ("p", "r", "ud", "n"))) if rel is not None else set()
    ratios: dict[tuple[str, ...], dict[str, float]] = {}
    for ec, ent in (("reserve__upDown__unit__node", "unit"), ("reserve__upDown__connection__node", "connection")):
        for name in ("increase_reserve_ratio", "large_failure_ratio"):
            df = _try_param(source, ec, name)
            if df is None:
                continue
            rows = df.select(*(pl.col(c).cast(pl.Utf8) for c in (ent, "reserve", "upDown", "node")),
                             pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
            for p, r, ud, n, v in rows.iter_rows():
                if (p, r, ud, n) in rel_keys:
                    ratios.setdefault((p, r, ud, n), {})[name] = float(v)
    group_ratios: dict[tuple[str, ...], float] = {}
    gr = _try_param(source, "reserve__upDown__group", "increase_reserve_ratio")
    if gr is not None:
        rows = gr.select(*(pl.col(c).cast(pl.Utf8) for c in ("reserve", "upDown", "group")),
                         pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
        group_ratios = {(r, ud, g): float(v) for r, ud, g, v in rows.iter_rows()}
    demanded = demanded_reserves(
        reservation_pairs=reservation_pairs,
        group_methods=group_methods,
        group_nodes=group_nodes,
        relationship_ratios=ratios,
        group_ratios=group_ratios,
    )
    if not demanded:
        return _empty_prundt()
    active_ru = pl.DataFrame({"__r": [r for r, _ in sorted(demanded)], "__ud": [u for _, u in sorted(demanded)]},
                             schema={"__r": pl.Utf8, "__ud": pl.Utf8})
    pruna_lf = (pruna.lazy()
                .with_columns(pl.col("r").cast(pl.Utf8).alias("__r"), pl.col("ud").cast(pl.Utf8).alias("__ud"))
                .join(active_ru.lazy(), on=["__r", "__ud"], how="semi")
                .drop("__r", "__ud"))
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (pruna_lf carries Enum p / r / ud / n).
    out = (pruna_lf
              .join(dt.lazy()
                       .select(alias_to_axis("d", "d"),
                               alias_to_axis("t", "t")),
                    how="cross")
              .select("p", "r", "ud", "n", "d", "t")
              .sort("p", "r", "ud", "n", "d", "t")
              .collect())
    if out.height == 0:
        return _empty_prundt()
    return out


# ---------------------------------------------------------------------------
# §3.15 — Delay
# ---------------------------------------------------------------------------


def _delay_distributions_from_source(
    source: "InputSource",
) -> tuple[dict[tuple[str, str], float],
            set[tuple[str, str]],
            set[str]] | None:
    """Read ``unit.delay`` from the source and classify into:

    * ``weighted`` — (p, td) → weight when value is a 1d_map.  These are
      multi-duration distributions.
    * ``single`` — (p, td) when value is a scalar.  Single delay duration.
    * ``processes`` — set of all delayed processes (union of both).

    Returns ``None`` if no delays are defined.

    ``connection.delay`` is not read: connection delay is not implemented
    and is ignored (``input_derivation`` warns about it), so a connection
    is modelled exactly as without a delay.

    flextool's input writer (input_writer.py:638-648) emits the same
    distinction via ``filter_in_type``: 1d_map → ``p_process_delay_weighted``,
    scalar → ``process_delay_single``.
    """
    weighted: dict[tuple[str, str], float] = {}
    single: set[tuple[str, str]] = set()
    processes: set[str] = set()
    found_any = False
    for ec in ("unit",):
        df = _try_param(source, ec, "delay")
        if df is None or df.height == 0:
            continue
        found_any = True
        cols = df.columns
        # Distinguish 1d_map shape (entity, key, value — three cols) from
        # scalar shape (entity, value — two cols).  The SpineDbReader
        # uses generic key names for 1d_map (typically `constraint` or
        # similar from the shape registry); we detect by column count.
        if len(cols) >= 3:
            # 1d_map: name, <key>, value.
            key_col = next((c for c in cols
                              if c not in ("name", "value")),
                            None)
            if key_col is None:
                continue
            for row in df.iter_rows(named=True):
                p = row["name"]
                kv = row[key_col]
                v = row["value"]
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # The CSV writer formats the delay duration via
                # repr(float(x)), so the string key matches when we
                # round-trip via float.
                try:
                    td = repr(float(kv))
                except (TypeError, ValueError):
                    td = str(kv)
                weighted[(p, td)] = fv
                processes.add(p)
        else:
            # Scalar entry — value IS the delay duration.
            for row in df.iter_rows(named=True):
                p = row["name"]
                v = row["value"]
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # 0 and negative durations don't activate delay.
                if fv <= 0.0:
                    continue
                td = repr(fv)
                single.add((p, td))
                processes.add(p)
    if not found_any:
        return None
    return weighted, single, processes


def p_process_delay_weight_from_source(
    source: "InputSource",
) -> "Param | None":
    """Per-(p, td) delay weight — audit §3.15.2.

    Algorithm — mirror of
    ``preprocessing/process_arc_unions.py:write_p_process_delay_weight:2105-2141``::

        p_process_delay_weight[p, td] = 1.0 if (p, td) ∈ process_delay_single
                                         else p_process_delay_weighted[p, td]

    The set ``process_delayed__duration`` is the union of the two sources;
    every row in that set produces an output value.

    Returns ``None`` when no delays are defined (no fixture rows on
    either source).
    """
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    rows: list[tuple[str, float, float]] = []
    # Order: weighted entries first, then single-only (mirrors flextool's
    # union which preserves first-occurrence order across the two sources).
    seen: set[tuple[str, str]] = set()
    for (p, td), v in weighted.items():
        rows.append((p, float(td), 1.0 if (p, td) in single else v))
        seen.add((p, td))
    for (p, td) in single:
        if (p, td) not in seen:
            rows.append((p, float(td), 1.0))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"p": schema_dtype(_enums, "p"),
                                       "td": pl.Float64,
                                       "value": pl.Float64},
                       orient="row")
    return Param(("p", "td"), out)


def process_delayed__duration_from_weight(
    p_process_delay_weight: "Param | None",
) -> pl.DataFrame | None:
    """``process_delayed__duration`` ``(p, td)`` — the delay durations of
    each delayed process (flextool.mod ``process_delay_weighted ∪
    process_delay_single``).

    Built from the keys of ``p_process_delay_weight`` (non-zero weights),
    i.e. from the same ``_delay_distributions_from_source`` read as the
    weights and ``dtt__delay_duration``: a scalar ``delay`` gives
    ``(p, delay)`` and a map gives one row per duration.  ``td`` therefore
    has exactly the dtype and values of ``p_process_delay_weight.td`` and
    ``dtt__delay_duration.td`` (Float64), which ``_delay.delayed_input_expr``
    joins on.  (A separate map-only reader used to produce this set: a
    scalar ``delay`` got no rows, the delayed unit lost all its input
    terms and was forced to zero output.)

    Returns ``None`` when no process is delayed.
    """
    if p_process_delay_weight is None or p_process_delay_weight.frame.height == 0:
        return None
    out = (p_process_delay_weight.frame
           .filter(pl.col("value") != 0.0)
           .select("p", "td")
           .sort("p", "td"))
    return out if out.height > 0 else None


def apply_delay_params(flex_data: object, source: "InputSource") -> None:
    """Set the three delay frames — ``dtt__delay_duration``,
    ``p_process_delay_weight`` and ``process_delayed__duration`` — from one
    read of ``unit.delay`` over ``flex_data.dt`` (a connection ``delay``
    is ignored).

    Solve-agnostic: :func:`apply_derived_g` calls it, and so does the
    synthetic (rolling) sub-solve path in ``flex_data._apply_db_overrides``,
    which skips the derived passes.  Producing all three together keeps
    the ``td`` labels identical across them.
    """
    flex_data.dtt__delay_duration = dtt__delay_duration_from_source(
        source, getattr(flex_data, "dt", None))
    flex_data.p_process_delay_weight = p_process_delay_weight_from_source(
        source)
    flex_data.process_delayed__duration = (
        process_delayed__duration_from_weight(flex_data.p_process_delay_weight))


def dtt__delay_duration_from_source(
    source: "InputSource",
    dt: pl.DataFrame | None,
) -> pl.DataFrame | None:
    """Per-(d, t_source, t_sink, td) delay shift map — audit §3.15.1.

    For each delay duration ``td`` and each (d, t_source) in dt, produce
    one row (d, t_source, t_sink, td) where ``t_sink = t_source + td``
    timesteps within the period.  Wraps at period boundaries::

        if k + td_int < len(time_steps):
            t_sink = time_steps[k + td_int]
        else:
            t_sink = time_steps[k - len(time_steps) + td_int]

    flextool casts ``td`` through ``int(float(delay_duration))`` so the
    csv-string "4.0" becomes shift index 4.

    Inputs: ``dt`` (period-time index) + the union of all delay durations
    from ``unit.delay`` (a connection ``delay`` is ignored).

    Returns ``None`` when no delays are defined or dt is empty.
    """
    if dt is None or dt.height == 0:
        return None
    info = _delay_distributions_from_source(source)
    if info is None:
        return None
    weighted, single, _ = info
    # Distinct delay duration set (string keys, as written by the CSV).
    durations: set[str] = set()
    for (_, td) in weighted:
        durations.add(td)
    for (_, td) in single:
        durations.add(td)
    if not durations:
        return None
    # Build per-period ordered timestep lists from dt.
    dt_sorted = (dt.lazy().sort("d", "t")
                    .group_by("d", maintain_order=True)
                    .agg(pl.col("t").alias("steps"))
                    .collect())
    rows: list[tuple[str, str, str, float]] = []
    for row in dt_sorted.iter_rows(named=True):
        d = row["d"]
        steps: list[str] = list(row["steps"])
        n = len(steps)
        if n == 0:
            continue
        for td in sorted(durations):  # a set: its order varies with the hash seed
            try:
                td_int = int(float(td))
                td_float = float(td)
            except (TypeError, ValueError):
                continue
            for k, t_src in enumerate(steps):
                if k + td_int < n:
                    t_sink = steps[k + td_int]
                else:
                    t_sink = steps[k - n + td_int]
                rows.append((d, t_src, t_sink, td_float))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"d": schema_dtype(_enums, "d"),
                                       "t_source": schema_dtype(_enums, "t_source"),
                                       "t_sink": schema_dtype(_enums, "t_sink"),
                                       "td": pl.Float64},
                       orient="row")
    return out


# ---------------------------------------------------------------------------
# §3.18 — Multi-branch normalisation (full cascade)
# ---------------------------------------------------------------------------


def _read_period_branch_pairs(workdir: Path | None,
                                  *,
                                  provider: "object | None" = None,
                                  ) -> list[tuple[str, str]]:
    """Read ``solve_data/period__branch.csv`` as a list of (d2, b)
    pairs — d2 is the parent period, b is the sibling branch.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return []
    df = _provider_read(provider, p)
    if df.height == 0:
        return []
    cols = df.columns
    if "period" in cols and "branch" in cols:
        d_col, b_col = "period", "branch"
    elif len(cols) >= 2:
        d_col, b_col = cols[0], cols[1]
    else:
        return []
    return [(str(r[d_col]), str(r[b_col])) for r in df.iter_rows(named=True)]


def _read_solve_branch_weights(workdir: Path | None,
                                   *,
                                   provider: "object | None" = None,
                                   ) -> dict[str, float]:
    """Read ``solve_data/solve_branch_weight.csv`` as branch → weight."""
    out: dict[str, float] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "solve_branch_weight.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return out
    df = _provider_read(provider, p)
    if df.height == 0:
        return out
    cols = df.columns
    b_col = "branch" if "branch" in cols else cols[0]
    v_col = "value" if "value" in cols else (
        cols[1] if len(cols) > 1 else None)
    if v_col is None:
        return out
    for r in df.iter_rows(named=True):
        b = str(r[b_col])
        try:
            out[b] = float(r[v_col])
        except (TypeError, ValueError):
            continue
    return out


def _read_first_timesteps(workdir: Path | None,
                              *,
                              provider: "object | None" = None,
                              ) -> dict[str, str]:
    """Read ``solve_data/first_timesteps.csv`` as period → first step."""
    out: dict[str, str] = {}
    if workdir is None:
        return out
    p = Path(workdir) / "solve_data" / "first_timesteps.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return out
    df = _provider_read(provider, p)
    if df.height == 0:
        return out
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    s_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    for r in df.iter_rows(named=True):
        out[str(r[d_col])] = str(r[s_col])
    return out


def pd_branch_weight_full_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
    workdir: Path | None,
) -> "Param | None":
    """Per-period branch weight — full multi-branch cascade.

    Δ.8 consolidation: delegates to the lazy port
    :func:`._branch.pd_branch_weight_param`.  Behaviour
    preserved: defaults to 1.0 when no ``period__branch`` rows exist
    (deterministic fixtures), normalises across siblings otherwise.

    See :mod:`._branch` for the algorithm reference.
    """
    from flextool.engine.preprocess.params._branch import pd_branch_weight_param
    out = pd_branch_weight_param(workdir, source, active_solve)
    if out is not None:
        return out
    # Match the historical "no period_in_use, but dt present →
    # deterministic 1.0 per realised period" fallback path.  This
    # keeps parity with the eager helper for chain-runner-less single
    # solves where the workdir CSVs may be absent but dt is built.
    if dt is None or dt.height == 0:
        return None
    # Defensive re-cast: ensure d is canonical Enum on the Param.
    df = (dt.lazy()
              .select(alias_to_axis("d", "d")).unique()
              .with_columns(value=pl.lit(1.0))
              .sort("d")
              .collect())
    if df.height == 0:
        return None
    return Param(("d",), df)


def pdt_branch_weight_full_from_source(
    source: "InputSource",
    active_solve: str | None,
    dt: pl.DataFrame | None,
    workdir: Path | None,
) -> "Param | None":
    """Per-(d, t) branch weight — full multi-branch cascade.

    Δ.8 consolidation: delegates to the lazy port
    :func:`._branch.pdt_branch_weight_param`.  Behaviour
    preserved: dense over ``dt`` (every (d, t) gets a value;
    denominator-zero rows fall through to 1.0).

    See :mod:`._branch` for the algorithm reference.
    """
    from flextool.engine.preprocess.params._branch import pdt_branch_weight_param
    return pdt_branch_weight_param(workdir, source, active_solve, dt)


def apply_synthetic_invest_sets(flex_data: object,
                                     source: "InputSource",
                                     active_solve: str,
                                     synthetic: tuple[str, str],
                                     workdir: "Path | None" = None,
                                     *,
                                     provider: "object | None" = None) -> None:
    """Δ.19 — populate the invest-set FlexData fields for a synthetic
    ``<base>_<anchor>`` solve, replacing the matching disk reads in
    ``_invest_seeds.py``.

    The synthetic-aware :func:`_solve_periods` returns the right period
    list for the anchor (Map-shaped ``invest_periods`` filtered to the
    synthetic anchor); the eager helpers in this module and the lazy
    partition helpers in :mod:`._existing` are pure derivations
    on top of that period list so they work as-is.  Fields populated
    (mirrors the seed names in ``_invest_seeds.py``):

      * ``ed_invest_set`` / ``ed_divest_set``
      * ``pd_invest_set`` / ``nd_invest_set`` (process / node partition
        of ``ed_invest_set``)
      * ``pd_divest_set`` / ``nd_divest_set`` (process / node partition
        of ``ed_divest_set``)
      * ``edd_invest_set`` (history-extended triple; requires workdir
        for ``period_with_history.csv`` + ``period_in_use_set.csv``).
      * ``ed_invest_forbidden_no_investment`` (no-investment lifetime
        gate; needed by ``apply_existing_chain``).

    The per-period cap subsets (``ed_invest_period_set`` /
    ``ed_divest_period_set``) and the NPV cost cascade
    (``ed_lifetime_fixed_cost`` etc.) are NOT touched — they stay on
    the ``_invest_seeds.py`` / ``_load_invest`` CSV-seed path because
    those values bake in multi-year discounting and the per-sub-solve
    filter doesn't compose cleanly with the existing derivation
    (deferred per the Gap D dispatch).
    """
    base, anchor = synthetic
    # Slice D §4 — arm the per-period anchor-map scope (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, provider=provider)
    # ed_invest_set / ed_divest_set — eager helpers use _solve_periods
    # which is synthetic-aware (Δ.19); pass the synthetic name through.
    # Γ.6.D forbidden filter is applied internally by the helper.
    ed_inv = ed_invest_set_from_source(
        source, active_solve, workdir=workdir, provider=provider)
    # Construction lead time: drop the decisions whose capacity would start to operate after the solve or the
    # model horizon (with a warning) before every set below is built from ed_invest_set.
    synth_commission_lf = None
    # A synthetic sub-solve (a roll, a nested solve renamed for its period; the solves of a 2-D period map are
    # real solves of the reader and never come here) counts one year per period on its own calendar and takes its
    # costs from the emitted seeds, so the start of operation and the cost window of a construction lead time
    # cannot be placed in it.
    from flextool.engine.preprocess.params._existing import (
        _construction_lag_configured as _lag_configured,
        lead_time_decisions as _lead_time_decisions,
    )
    if ed_inv is not None and ed_inv.height > 0 and _lag_configured(source):
        lagged = _lead_time_decisions(source, ed_inv.lazy().select("e", "d"))
        if lagged.height > 0:
            longest = lagged.group_by("e").agg(pl.col("L").max()).sort("e")
            entities = ", ".join(f"{e} ({lead:g} years)" for e, lead in longest.iter_rows())
            raise FlexToolConfigError(
                f"Solve '{active_solve}' is a sub-solve of '{base}' (a roll, or a nested solve renamed for its "
                "period) that invests, and FlexTool cannot place the start of operation of a construction lead "
                f"time in such a sub-solve. Entities with a lead time: {entities}. Make the investments in a "
                "single_solve (a solve of its own in model.solves), or remove the lead time.")
    if workdir is not None:
        ed_inv, synth_commission_lf = _offer_invest_set(
            flex_data, source, active_solve,
            ed_inv if ed_inv is not None else getattr(flex_data, "ed_invest_set", None),
            _period_in_use_set(source, active_solve, workdir, provider=provider), workdir, provider=provider)
    if ed_inv is not None and ed_inv.height > 0:
        flex_data.ed_invest_set = ed_inv
    ed_div = ed_divest_set_from_source(
        source, active_solve, workdir, provider=provider)
    if ed_div is not None and ed_div.height > 0:
        flex_data.ed_divest_set = ed_div
    zero_cost_invest_notice(source, ed_inv, ed_div)

    # ed_invest_forbidden_no_investment — derived from the lifetime
    # gate; consumed by apply_existing_chain.
    try:
        forbidden = ed_invest_forbidden_no_investment_from_source(
            source, active_solve, workdir, ed_inv, provider=provider)
    except Exception:  # pragma: no cover — defensive
        forbidden = None
    if forbidden is not None and forbidden.height > 0:
        flex_data.ed_invest_forbidden_no_investment = forbidden

    # pd/nd_invest_set, pd/nd_divest_set partitions — pure entity-class
    # projections, no solve dependency beyond the input ed_*_set frame.
    from flextool.engine.preprocess.params._existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
        edd_invest_set_lf as _edd_invest_lf,
    )
    if ed_inv is not None and ed_inv.height > 0:
        ed_inv_lf = ed_inv.lazy()
        pd_inv = _pd_invest_lf(source, ed_inv_lf).collect()
        if pd_inv.height > 0:
            flex_data.pd_invest_set = pd_inv
        nd_inv = _nd_invest_lf(source, ed_inv_lf).collect()
        if nd_inv.height > 0:
            flex_data.nd_invest_set = nd_inv
    if ed_div is not None and ed_div.height > 0:
        ed_div_lf = ed_div.lazy()
        pd_div = _pd_divest_lf(source, ed_div_lf).collect()
        if pd_div.height > 0:
            flex_data.pd_divest_set = pd_div
        nd_div = _nd_divest_lf(source, ed_div_lf).collect()
        if nd_div.height > 0:
            flex_data.nd_divest_set = nd_div

    # edd_invest_set — history-extended triple.  Uses workdir CSV for
    # period_with_history + period_in_use_set (both anchor-specific in
    # the synthetic-solve snapshot).
    if (ed_inv is not None and ed_inv.height > 0
            and workdir is not None):
        period_in_use = _period_in_use_set(source, active_solve, workdir, provider=provider)
        period_with_history = (_read_period_with_history(workdir, provider=provider)
                                  or list(period_in_use))
        # The start of operation of the decisions with a construction lead time was found (and the decisions
        # starting after the solve dropped) where ed_invest_set was built above.
        commission_lf = synth_commission_lf
        # Slice D §8 — synthetic edd lineage filter (gated + hoisted check).
        # Build on-demand from the provider (branch-cluster runs later); the
        # anchor-pairs holder armed at the top of this fn is the conjunct.
        if _RECOURSE_ANCHOR_PAIRS is not None:
            from flextool.engine.preprocess.params._branch import (
                assert_recourse_npv_preconditions as _assert_recourse,
                dd_same_scenario_df as _dd_same_scenario_df,
            )
            _s_lineage = _dd_same_scenario_df(
                workdir, source, active_solve, provider=provider)
            _assert_recourse(workdir, source, active_solve, provider=provider)
        else:
            _s_lineage = None
        try:
            edd_inv = _edd_invest_lf(
                source, active_solve, ed_inv.lazy(),
                period_with_history, period_in_use, workdir,
                lineage=_s_lineage, commission_lf=commission_lf).collect()
        except (LineageFilterError, CommissioningLagError):
            raise
        except Exception:  # pragma: no cover — defensive
            edd_inv = None
        if edd_inv is not None and edd_inv.height > 0:
            flex_data.edd_invest_set = edd_inv
        _apply_lead_time_sets(flex_data, source, active_solve, ed_inv, commission_lf, period_in_use, workdir,
                              lineage=_s_lineage)


# ---------------------------------------------------------------------------
# Public — Γ.3.G field list (for selective scope-checking).
# ---------------------------------------------------------------------------


G_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_f_d_k",
    "p_ladder_cum_realized_mwh",
    "prundt",
    "dtt__delay_duration",
    "p_process_delay_weight",
    "process_delayed__duration",
    "pd_branch_weight",
    "pdt_branch_weight",
)


def apply_derived_g(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.G residual Derived Params, mutating ``flex_data``
    in place.

    Scope:
      * §3.17 commodity ladder — ``p_f_d_k``,
        ``p_ladder_cum_realized_mwh``.
      * §3.13 reserves — ``prundt``.
      * §3.15 delay — ``dtt__delay_duration``, ``p_process_delay_weight``,
        ``process_delayed__duration``.
      * §3.18 multi-branch normalisation — full cascade for
        ``pd_branch_weight`` and ``pdt_branch_weight``.

    Δ.3 replaced the previous ``derived_overrides_g`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    dt = getattr(flex_data, "dt", None)

    # Δ.12b — assignment is unconditional; helpers are authoritative
    # producers.  ``None`` is the explicit "feature inactive" signal
    # (no commodity-ladder commodities, no reserve relationships,
    # no delayed processes, etc.).  Hard exceptions propagate.

    # ─── §3.17.1 p_f_d_k ───────────────────────────────────────────
    flex_data.p_f_d_k = p_f_d_k_from_source(source, active_solve, workdir,
                                                  ctx=ctx, provider=provider)

    # ─── §3.17.2 p_ladder_cum_realized_mwh ─────────────────────────
    flex_data.p_ladder_cum_realized_mwh = (
        p_ladder_cum_realized_mwh_from_workdir(workdir, provider=provider))

    # ─── §3.13.1 prundt ────────────────────────────────────────────
    flex_data.prundt = prundt_from_source(source, active_solve, dt)

    # ─── §3.15 dtt__delay_duration, p_process_delay_weight and
    #     process_delayed__duration (one source read, same td labels) ──
    apply_delay_params(flex_data, source)

    # ─── §3.18 cluster D — multi-branch normalisation + non-anticipativity ──
    # Δ.8 consolidation: delegate the full cluster D port to
    # ``_branch.apply_branch_cluster``.  The helper covers
    # ``pd_branch_weight``, ``pdt_branch_weight``, and (already
    # CSV-loaded by ``flex_data.py``) ``period_branch_full``,
    # ``period_in_use_set``, ``dt_non_anticipativity`` — the latter
    # three are kept as overrides so the lazy port becomes the single
    # producer once the CSV cascade retires (Δ.12).  R-O6: the helper
    # never touches ``invest_periods`` / ``v_invest``.
    # Δ.12b: defensive try/except removed; apply_branch_cluster is
    # parity-bound (cluster D tests gate it).
    from flextool.engine.preprocess.params._branch import apply_branch_cluster
    apply_branch_cluster(flex_data, source, workdir, active_solve,
                          ctx=ctx, provider=provider)


