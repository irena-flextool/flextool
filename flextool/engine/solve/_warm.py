"""Warm-LP primitives for the native cascade in ``_orchestration``.

Contract: a built LP is reused only when the next sub-solve's FlexData and
solver configuration are identical (labels included), so the solution, the
handoff and the output writer, which all read the LP's labels, see the right
rows and columns; the warm path pushes no updates.

* :func:`_fingerprint` is a cheap pre-filter (set heights and the time-axis
  structure).  A mismatch means "build a new LP" without a closer look.
* :func:`_reuse_blocker` is the decision: it compares every field of the two
  FlexData objects by content and names the first one that differs.
* :func:`_apply_warm_updates` keeps the historical entry point: it raises
  :class:`_IncompatibleUpdate` naming the blocker, or returns 0.

Why labels matter: a reused :class:`polar_high.WarmProblem` keeps the ``Var``
frames and the HiGHS column / row names of the sub-solve it was built for.
The output writer and ``build_handoff_from_solution`` select the realized
``(d, t)`` of the current sub-solve by those labels, so a reused LP whose
labels belong to another window reports the wrong hours and loses the
storage / online / invest carry-over.  Rolls shift ``dt`` on every roll, so
they are always rebuilt.  The solver configuration (HiGHS options, autoscale
mode, ``user_bound_scale``, objective scale) is compared by the orchestrator,
which owns it.

Consumed by ``_orchestration.py::_drive_cascade``.  Re-exported as module
attributes on ``chain.py`` for backward compatibility.
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param, Problem, WarmProblem

if TYPE_CHECKING:
    from flextool.engine.preprocess.flex_data import FlexData


__all__ = [
    "_STRUCTURAL_FIELDS",
    "_IncompatibleUpdate",
    "_fingerprint",
    "_param_frame_equal",
    "_param_values_position_equal",
    "_reuse_blocker",
    "_values_equal",
    "_apply_warm_updates",
    "_build_warm_problem",
]


# ---------------------------------------------------------------------------
# Structural-fingerprint fields (the cheap pre-filter).
#
# The fingerprint is a tuple of (field_name, height) pairs over the set
# fields that shape the LP, plus the time-axis structure.  Two sub-solves
# whose fingerprints differ cannot build the same LP, so the orchestrator
# skips the full comparison and builds a new one.  The list does not need to
# be complete: a field missing here is still compared by content in
# :func:`_reuse_blocker`, which decides.
#
# Heights alone do not capture how the time axis is partitioned: two rolls
# with the same number of steps can split them differently between the
# periods (p2020 18 h + p2025 12 h, then 12 h + 18 h) or have a timeblock
# start at a different step.  :func:`_fingerprint` therefore also carries the
# time-axis structure (:func:`_time_axis_signature`).

_STRUCTURAL_FIELDS: tuple[str, ...] = (
    # Time + node sets (the foundation of every LP).
    "dt", "nodeBalance", "nodeBalance_dt",
    # Process topology.
    "process_source_sink", "process_source_sink_eff",
    "process_source_sink_noEff", "pss_dt",
    "flow_to_n", "flow_from_n",
    "flow_from_commodity_eff", "flow_from_commodity_noEff",
    "flow_to_commodity",
    "pd_neg_cap",
    # CO2.
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "group_d_co2_capped",
    "group_co2_max_total", "flow_from_co2_capped_total",
    "flow_from_co2_capped_total_noEff",
    # Indirect (CHP).
    "process_indirect", "process_input_flows",
    "process_output_flows", "process_indirect_dt",
    # User constraints.
    "flow_constraint_idx", "cdt_eq", "cdt_le", "cdt_ge",
    # Profiles.
    "process_profile_upper", "process_profile_lower",
    "process_profile_fixed",
    # Invest / divest.
    "ed_invest_set", "ed_divest_set",
    "pd_invest_set", "pd_divest_set",
    "nd_invest_set", "nd_divest_set",
    "edd_invest_set", "edd_invest_lookback_set", "edd_divest_active",
    "e_invest_total", "e_divest_total",
    "ed_invest_period_set", "ed_divest_period_set",
    # Ramp.
    "process_source_sink_ramp_limit_sink_up",
    "process_source_sink_ramp_limit_sink_down",
    "process_source_sink_ramp_limit_source_up",
    "process_source_sink_ramp_limit_source_down",
    # Online / UC.
    "process_online", "process_online_linear", "process_online_integer", "process_exact",
    "process_minload", "process_min_load_eff",
    "p_online_dt", "pdt_online_linear", "pdt_online_integer",
    "pdt_uptime_set", "pdt_downtime_set",
    "uptime_lookback", "downtime_lookback",
    # Storage.
    "nodeState", "nodeState_dt", "nodeState_first_dt",
    "storage_bind_within_timeblock", "storage_bind_forward_only",
    "storage_bind_within_solve",
    "storage_bind_within_solve_blended_weights",
    # Phase D — landed with the new variant's constraint wiring.
    "storage_bind_forward_only_blended_weights",
    # Phase E — landed with the per-period cyclic-closure variant.
    "storage_bind_within_period_blended_weights",
    "storage_fix_start",
    "dtttdt", "dtttdt_forward_only",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "dtt_timeline_matching", "period_branch", "period_last",
    "nodeState_last_dt",
    "nodeStateBlock", "period_block", "period_block_succ",
    "period_block_time", "dtttdt_block_interior",
    # Coarse new_stepduration group nodes.
    "nodeCoarse", "node_coarse_step", "node_coarse_interior",
    # RP-blended-weights (bind_within_solve_blended_weights) sets / params —
    # per-solve constants that don't change across iterations within a
    # level; same carry-over kind as the intraperiod-block family above.
    "nodeState_rp", "rp_base_period_set",
    "rp_base_chain", "rp_base_first", "rp_base_last",
    "rp_block_first", "p_rp_last_step", "rp_base__rep",
    "flow_from_nodeBalance_eff", "flow_from_nodeBalance_noEff",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    # Variable cost partitions.
    "pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source",
    "pssdt_varCost_eff_unit_sink", "pssdt_varCost_eff_connection",
    # Group slack.
    "groupCapacityMargin", "groupInertia", "groupNonSync",
    # Capacity-margin periods: constant within one solve's rolls; a height change forces a cold rebuild.
    "period_invest",
    "group_node", "process_unit",
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
    # Reserves.
    "reserve_upDown_group",
    "reserve_upDown_group_method_timeseries",
    "reserve_upDown_group_method_dynamic",
    "reserve_upDown_group_method_n_1",
    "prundt", "process_reserve_upDown_node_active",
    "process_reserve_upDown_node_increase_reserve_ratio",
    "process_reserve_upDown_node_large_failure_ratio",
    # Cumulative invest / group invest.
    "ed_invest_forbidden_no_investment", "ed_invest_cumulative",
    "group_entity", "g_invest_total", "g_divest_total",
    "g_invest_cumulative", "gd_invest_period", "gd_divest_period",
    "group_process_node",
    # Delays.
    "process_delayed", "process_delayed__duration",
    "process_source_delayed", "process_source_undelayed",
    "process_source_sink_delayed", "process_source_sink_undelayed",
    "dtt__delay_duration",
)



class _IncompatibleUpdate(Exception):
    """Raised by :func:`_apply_warm_updates` when the next sub-solve would
    not build the live LP; the message names the first differing field.
    The orchestrator then builds a new LP for the next sub-solve."""


def _fingerprint(data: "FlexData") -> tuple:
    """Compute a structural fingerprint of a FlexData snapshot.

    Returns a tuple of ``(field_name, height_or_None)`` pairs covering
    every field listed in :data:`_STRUCTURAL_FIELDS`, plus the time-axis
    structure of :func:`_time_axis_signature`.  Two FlexData snapshots
    with equal fingerprints emit identically-shaped LPs (set of vars and
    cstrs match by row count and dim signature, and the same positions of
    the time axis belong to the same period and link to the same
    positions) under the current ``build_flextool`` rules.

    ``height_or_None`` is ``None`` when the field is unset and the
    integer ``height`` when it's a polars DataFrame.  Boolean / scalar
    fields contribute their value directly.  Heights alone do not see a
    different split of the same number of steps between the periods,
    nor a timeblock boundary that moves inside the window.
    """
    out = []
    for name in _STRUCTURAL_FIELDS:
        v = getattr(data, name, None)
        if v is None:
            out.append((name, None))
        elif isinstance(v, pl.DataFrame):
            out.append((name, int(v.height)))
        elif isinstance(v, bool):
            out.append((name, bool(v)))
        else:
            # Unexpected — be conservative and force-mismatch.
            out.append((name, repr(type(v).__name__)))
    # ``p_nested_solve_first`` is a tri-state flag that swaps a whole
    # constraint family in/out — track it explicitly.
    out.append(("p_nested_solve_first",
                getattr(data, "p_nested_solve_first", None)))
    out.extend(_time_axis_signature(data))
    return tuple(out)


def _time_axis_signature(data: "FlexData") -> tuple:
    """How the sub-solve's time axis is partitioned and linked, without its labels.

    * ``dt_period_runs``: the number of steps of each period, in the order
      of ``dt`` sorted by its columns (the order in which every (d, t)
      domain frame is sorted before its columns / rows get ids).  The
      period labels are left out: this is a cheap pre-filter of the shape
      only.  A label change with the same run lengths (``{p2020: 30}`` ->
      ``{p2025: 30}``) passes it and is caught by :func:`_reuse_blocker`
      (``dt`` differs).
    * ``dtttdt_link_offsets``: for each ``dtttdt`` row (sorted by d, t),
      the offset from the row's own position in the sorted ``dt`` to the
      position of each linked step (``t_previous``,
      ``t_previous_within_timeset``, ``(d_previous,
      t_previous_within_solve)``); -1 inside a block, other values at block
      and period starts, and a marker when the linked step is not in
      ``dt``.  Hashed to a short digest.
    """
    import hashlib

    dt = getattr(data, "dt", None)
    if not isinstance(dt, pl.DataFrame) or "d" not in dt.columns or "t" not in dt.columns:
        return (("dt_period_runs", None), ("dtttdt_link_offsets", None))
    ordered = dt.select("d", "t").sort("d", "t")
    runs = (ordered.select(pl.col("d").cast(pl.Utf8))
            .with_columns(run=pl.col("d").rle_id())
            .group_by("run", maintain_order=True).len())
    out: list = [("dt_period_runs", tuple(int(x) for x in runs["len"].to_list()))]

    lk = getattr(data, "dtttdt", None)
    if not isinstance(lk, pl.DataFrame) or not {"d", "t"} <= set(lk.columns):
        out.append(("dtttdt_link_offsets", None))
        return tuple(out)
    pos = ordered.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        pl.int_range(pl.len(), dtype=pl.Int64).alias("_pos"))
    links = [("t_previous", "d"), ("t_previous_within_timeset", "d"),
             ("t_previous_within_solve", "d_previous")]
    links = [(tc, dc) for tc, dc in links if tc in lk.columns and dc in lk.columns]
    rows = lk.sort("d", "t").select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        *[pl.col(dc).cast(pl.Utf8).alias(f"_ld{i}") for i, (_, dc) in enumerate(links)],
        *[pl.col(tc).cast(pl.Utf8).alias(f"_lt{i}") for i, (tc, _) in enumerate(links)])
    rows = rows.with_row_index("_row").join(pos, on=["_d", "_t"], how="left")
    for i in range(len(links)):
        rows = rows.join(
            pos.rename({"_d": f"_ld{i}", "_t": f"_lt{i}", "_pos": f"_lpos{i}"}),
            on=[f"_ld{i}", f"_lt{i}"], how="left")
    rows = rows.sort("_row")
    # Offsets of linked steps outside dt (or rows outside dt) are null -> a marker value.
    marker = -(1 << 40)
    offsets = rows.select(
        pl.col("_pos").fill_null(marker).alias("_own"),
        *[(pl.col(f"_lpos{i}") - pl.col("_pos")).fill_null(marker).alias(f"_off{i}")
          for i in range(len(links))])
    digest = hashlib.sha1(
        repr([c for c, _ in links]).encode()
        + offsets.to_numpy().astype("<i8").tobytes()).hexdigest()
    out.append(("dtttdt_link_offsets", (rows.height, digest)))
    return tuple(out)


def _param_frame_equal(a, b) -> bool:
    """Return True if two polar_high Params have identical frames.

    Compares dim signature and value column row-by-row.  Robust to
    polars row-order differences via a sort on the dim columns.
    """
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if a.dims != b.dims:
        return False
    af = a.frame
    bf = b.frame
    if af.height != bf.height:
        return False
    if af.height == 0:
        return True
    if a.dims:
        cols = list(a.dims)
        af = af.sort(cols)
        bf = bf.sort(cols)
    return af.equals(bf)


def _param_values_position_equal(a, b) -> bool:
    """Return True if two Params have value columns that are equal at
    matching positions (after sorting each by its full set of dim
    columns).

    Captures the rolling-horizon case where dim labels (e.g. ``t``)
    shift between sub-solves but the per-position values (e.g. constant
    ``3000.0`` slack penalties for every (n,d,t)) stay identical.

    Returns ``False`` if dim signatures differ or row counts differ.
    Returns ``True`` if both Params are ``None``.  Otherwise compares
    the sorted-by-dims value column element-wise within float64
    precision.
    """
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        return False
    if a.dims != b.dims:
        return False
    af = a.frame
    bf = b.frame
    if af.height != bf.height:
        return False
    if af.height == 0:
        return True
    if a.dims:
        # Sort each frame by its dim columns positionally — the t-labels
        # in `a` and `b` differ for rolling-horizon snapshots, so we
        # can't just compare frames as-is.  Sorting brings the value
        # columns into 1-to-1 positional correspondence assuming the
        # sort orders agree (which they do when both sub-solves have
        # the same number of dim-tuples).  This is a fast O(n log n)
        # numeric check rather than a full frame equality.
        cols = list(a.dims)
        av = af.sort(cols)["value"].to_numpy()
        bv = bf.sort(cols)["value"].to_numpy()
    else:
        av = af["value"].to_numpy()
        bv = bf["value"].to_numpy()
    import numpy as np
    return bool(np.array_equal(av, bv))




# ---------------------------------------------------------------------------
# Reuse decision: every field identical, by content.

_MISSING = object()


def _field_names(obj) -> list[str]:
    """Field names of a FlexData-like object in a deterministic order:
    dataclass field order, else the sorted attribute names (the
    ``SimpleNamespace`` stand-ins of the unit tests)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclasses.fields(obj)]
    return sorted(vars(obj))


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    """Same columns with the same dtypes and the same rows, in any row and
    column order.

    Fast path: identical schemas (column order and dtypes) and ``.equals``
    as the frames are; a producer that is deterministic gives this for
    identical sub-solves.  Otherwise the columns are selected in sorted name
    order and both frames are sorted by all columns before ``.equals``.  A
    dtype difference (Enum categories included) counts as unequal: the LP's
    column / row order follows the Enum order, so the safe answer is a new
    build.
    """
    if a.height != b.height or a.width != b.width:
        return False
    if a.schema == b.schema and a.equals(b):
        return True
    sa, sb = a.schema, b.schema
    if set(sa) != set(sb) or any(sa[c] != sb[c] for c in sa):
        return False
    if a.height == 0:
        return True
    cols = sorted(sa)
    return a.select(cols).sort(cols).equals(b.select(cols).sort(cols))


def _values_equal(a, b) -> bool:
    """Generic content equality of two FlexData field values.

    ``None`` equals only ``None``; frames by :func:`_frames_equal`;
    :class:`polar_high.Param` by dims and frame; dicts, lists, tuples and
    dataclass instances element by element; floats by ``==`` or both NaN;
    anything else by identity or ``==``.  A comparison that raises counts as
    unequal (the safe answer: build a new LP).
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        if isinstance(a, pl.DataFrame) or isinstance(b, pl.DataFrame):
            return (isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame)
                    and _frames_equal(a, b))
        if isinstance(a, Param) or isinstance(b, Param):
            return (isinstance(a, Param) and isinstance(b, Param)
                    and tuple(a.dims) == tuple(b.dims)
                    and _frames_equal(a.frame, b.frame))
        if isinstance(a, pl.Series) or isinstance(b, pl.Series):
            return (isinstance(a, pl.Series) and isinstance(b, pl.Series)
                    and a.dtype == b.dtype and a.equals(b))
        if isinstance(a, dict) or isinstance(b, dict):
            return (isinstance(a, dict) and isinstance(b, dict)
                    and a.keys() == b.keys()
                    and all(_values_equal(a[k], b[k]) for k in a))
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (type(a) is type(b) and len(a) == len(b)
                    and all(_values_equal(x, y) for x, y in zip(a, b)))
        if dataclasses.is_dataclass(a) and not isinstance(a, type):
            return (type(a) is type(b)
                    and all(_values_equal(getattr(a, f.name), getattr(b, f.name))
                            for f in dataclasses.fields(a)))
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (math.isnan(a) and math.isnan(b))
        if type(a) is not type(b):
            return False
        return bool(a == b)
    except Exception:  # noqa: BLE001 — unequal is the safe answer
        return False


def _reuse_blocker(prior, nxt) -> str | None:
    """Name of the first field in which ``nxt`` differs from ``prior``, or
    ``None`` when every field is identical (labels included).

    Every field is compared, with no exceptions: ``dt`` first (cheap, and the
    field that differs on every roll), then the dataclass fields in
    declaration order, then any extra attribute names in sorted order.  The
    order only affects which name is reported, never the outcome.
    """
    names: list[str] = []
    seen: set[str] = set()
    for name in ["dt", *_field_names(prior), *_field_names(nxt)]:
        if name not in seen:
            seen.add(name)
            names.append(name)
    for name in names:
        a = getattr(prior, name, _MISSING)
        b = getattr(nxt, name, _MISSING)
        if a is _MISSING and b is _MISSING:
            continue
        if a is _MISSING or b is _MISSING or not _values_equal(a, b):
            return name
    return None


def _apply_warm_updates(warm: WarmProblem,
                        prior: "FlexData", nxt: "FlexData") -> int:
    """Decide whether the live LP in ``warm`` can serve ``nxt``.

    Returns 0 (nothing to push: the LP is the one ``nxt`` would build).
    Raises :class:`_IncompatibleUpdate` naming the first differing field
    (:func:`_reuse_blocker`) otherwise.  ``warm`` is not touched; the
    argument is kept for the call contract of the orchestrator and tests.
    """
    blocker = _reuse_blocker(prior, nxt)
    if blocker is not None:
        raise _IncompatibleUpdate(
            f"FlexData field {blocker!r} differs between sub-solves; the "
            f"live LP is reused only for an identical sub-solve")
    return 0


def _build_warm_problem(
    data: "FlexData",
    *,
    scale_the_objective: float = 1.0,
    solver_options: dict | None = None,
) -> WarmProblem:
    """Build a fresh :class:`polar_high.WarmProblem` from a FlexData
    snapshot.

    Parameters
    ----------
    data
        The :class:`FlexData` bundle for this iteration.
    scale_the_objective
        Multiplier applied to the objective inside ``build_flextool``.
    solver_options
        HiGHS options to apply via ``pb.set_solver_options`` BEFORE
        wrapping in :class:`WarmProblem`.  Typically the dict returned
        by :func:`flextool.engine.scaling.recommended_highs_options`.

    No Param is declared mutable: the warm path pushes no updates, so the
    build skips polar-high's per-cell tracking pass.

    Late-imports ``build_flextool`` to avoid a build-time cycle between
    this module and ``model/core.py``.
    """
    from flextool.engine.solve._solver_dispatch import apply_data_solver_options
    from flextool.engine.model.core import build_flextool

    pb = Problem()
    build_flextool(pb, data, scale_the_objective=scale_the_objective)
    apply_data_solver_options(pb, data)
    if solver_options:
        pb.set_solver_options(solver_options)
    return WarmProblem(pb)
