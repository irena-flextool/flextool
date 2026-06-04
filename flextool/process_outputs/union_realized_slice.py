"""Union per-roll realized param/set slices at output time (stage 3).

Multi-sub-solve runs persist each roll's realized param/set slice to
``output_raw/<attr>__<solve>.parquet`` while the Solution is still live
(:mod:`flextool.process_outputs.persist_realized_slice`, stage 2).  This
module is the read side: it reconstructs the full ``par`` / ``s``
namespaces by UNIONING those per-roll parquets, reproducing exactly what
the proven-correct in-memory oracle
(:func:`...read_parameters.read_parameters_multi` /
:func:`...read_sets.read_sets_multi`) produces — but sourcing the
per-roll frames from disk instead of holding every roll's
``flex_data`` / ``solution`` in memory (which the default
``keep_solutions=False`` flow nulls on all but the last step).

Contract (identical to the in-memory multi readers):

* **Solve-keyed attrs** (carry a ``solve`` index level) — the union over
  rolls.  The persisted slices are ALREADY realized-filtered and
  post-hack (``persist_realized_slice`` applies the three
  ``entity_lifetime_fixed_cost`` / ``entity_all_existing`` /
  ``entity_annual_*`` hacks + the realized intersect at persist time),
  so the union is a plain row concat — exactly the ``pd.concat`` in
  ``read_parameters_multi`` / the per-level MultiIndex concat in
  ``read_sets_multi``.  No fillna, no dedup (each roll carries a distinct
  ``solve`` value); ``drop_levels`` does the realized-intersect + dedup +
  solve-drop downstream, just as it does for the in-memory path.
* **Static (solve-invariant) attrs** — taken once from the last step's
  in-memory namespace (``read_parameters`` / ``read_sets`` on the last
  roll's ``flex_data`` / ``solution``).  These carry no ``solve`` level
  and are identical across rolls; they are NOT persisted per-roll.
* **Output-dead solve-keyed params** (``flow_min``, ``flow_max``,
  ``years_from_start_d``, ``entity_max_units``, ``node_annual_flow``,
  ``group_capacity_margin``) are not persisted (``persist_realized_slice
  ._DEAD_PARAMS``) and have no consumer; the last step's value is kept so
  the namespace stays shape-complete.

Per-roll parquets are unioned in solve CREATION order
(:func:`...solve_order.load_solve_order`) — the same order the oracle's
``solve_steps`` (``steps.items()``) carries and the same order the
variable reader uses, so the assembled row order matches.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pandas as pd

from flextool.lean_parquet import read_lean_parquet
from flextool.process_outputs.read_parameters import _has_solve_level
from flextool.process_outputs.read_sets import (
    _multi_index_has_solve,
    _series_or_df_has_solve,
)
from flextool.process_outputs.solve_order import load_solve_order

# Sentinel prefixes that are persisted ONLY by the per-roll multi-solve
# writer (a solve-keyed core set + a solve-keyed core param).  Their
# presence in ``output_raw/`` means the stage-3 union path is live.
_SENTINEL_PREFIXES: tuple[str, ...] = ("dt", "step_duration")


def has_persisted_slices(output_dir: Path | str) -> bool:
    """True iff MULTI-roll param/set slices were persisted to ``output_dir``.

    Detection mirrors the variable reader's ``v_flow__*.parquet`` probe:
    look for the per-roll ``<sentinel>__*.parquet`` files the stage-2
    writer emits.  Both sentinels must be present (one set, one param),
    AND there must be MORE THAN ONE distinct solve label — a genuine
    multi-sub-solve cascade.  A single-solve scenario also persists these
    parquets, but its outputs are produced by the in-memory single-
    ``flex_data`` path (which is NOT realized-filtered); activating the
    union path only for true multi-roll runs keeps single-solve byte-
    identical.
    """
    out = Path(output_dir)
    if not all(
        any(out.glob(f"{prefix}__*.parquet")) for prefix in _SENTINEL_PREFIXES
    ):
        return False
    sentinel = _SENTINEL_PREFIXES[0]
    solves = {
        p.name[len(sentinel) + 2 : -len(".parquet")]
        for p in out.glob(f"{sentinel}__*.parquet")
    }
    return len(solves) > 1


def _ordered_parts(output_dir: Path, attr: str, solve_order: dict[str, int]):
    """Per-roll parquet paths for ``attr`` in solve creation order.

    ``<attr>__<solve>.parquet`` — the solve label is everything after the
    first ``__``.  Parts whose solve is unknown to ``solve_order`` sort
    first (defensive; shouldn't happen).
    """
    parts = list(output_dir.glob(f"{attr}__*.parquet"))

    def _solve(path: Path) -> str:
        return path.name[len(attr) + 2 : -len(".parquet")]

    parts.sort(key=lambda p: solve_order.get(_solve(p), -1))
    return parts


def _union_param_attr(parts: Sequence[Path], template) -> object:
    """Union one solve-keyed param attr from its per-roll parquets.

    ``template`` is the last step's value for this attr (a ``pd.Series``
    or ``pd.DataFrame``) — used only to decide the output shape and to
    restore the ``columns.name`` / Series name.  Mirrors the
    ``read_parameters_multi`` per-attr concat: drop empty pieces, concat
    rows, restore the column metadata; if every piece is empty fall back
    to the (empty) template so the namespace stays shape-complete.
    """
    frames = [read_lean_parquet(p) for p in parts]
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return template
    merged = pd.concat(non_empty, axis=0)
    if isinstance(template, pd.Series):
        # Series were persisted as a one-column ``value`` frame.
        series = merged.iloc[:, 0]
        series.name = template.name
        return series
    # DataFrame — restore the columns name (concat keeps it, but be
    # explicit to match ``read_parameters_multi``).
    if hasattr(template, "columns"):
        merged.columns.name = template.columns.name
    return merged


def union_params(
    last_par: SimpleNamespace, output_dir: Path | str
) -> SimpleNamespace:
    """Reconstruct the full ``par`` namespace by unioning per-roll slices.

    For every attr carrying a ``solve`` level in ``last_par`` whose
    per-roll parquets exist, replace it with the union; static attrs (and
    the output-dead solve-keyed params that are never persisted) keep the
    last step's value.  Byte-equivalent to ``read_parameters_multi`` for
    the consumed attrs.
    """
    output_dir = Path(output_dir)
    solve_order = load_solve_order(output_dir.parent)

    out = SimpleNamespace()
    for attr, value in vars(last_par).items():
        if not _has_solve_level(value):
            # Static / invariant — last-step value (identical per roll).
            setattr(out, attr, value)
            continue
        parts = _ordered_parts(output_dir, attr, solve_order)
        if not parts:
            # Output-dead solve-keyed param (never persisted) — keep the
            # last step's value; it has no downstream consumer.
            setattr(out, attr, value)
            continue
        setattr(out, attr, _union_param_attr(parts, value))
    return out


def _union_set_multiindex(parts: Sequence[Path], template: pd.MultiIndex):
    """Union one solve-keyed set's per-roll MultiIndex parquets.

    Each set parquet stores the (Multi)Index as the row index of a
    one-column ``_SET_MARKER_COL`` frame.  Drop the marker column and
    rebuild the MultiIndex via the same per-level concat
    ``read_sets_multi`` uses (robust to empty-frame dtype quirks).  If
    every roll is empty, fall back to the template (empty) MultiIndex.
    """
    indices: list[pd.MultiIndex] = []
    for p in parts:
        df = read_lean_parquet(p)
        idx = df.index
        if len(idx) > 0:
            indices.append(idx)
    if not indices:
        return template
    names = indices[0].names
    level_lists: list[list] = [[] for _ in names]
    for idx in indices:
        for i in range(len(names)):
            level_lists[i].extend(list(idx.get_level_values(i)))
    return pd.MultiIndex.from_arrays(level_lists, names=names)


def union_sets(
    last_s: SimpleNamespace, output_dir: Path | str
) -> SimpleNamespace:
    """Reconstruct the full ``s`` namespace by unioning per-roll slices.

    Solve-keyed sets (``period``, ``dt``, ``dtt``, ``dtttdt``, the
    ``ed_*`` invest sets, …) are unioned across rolls; static topology
    sets (``node``, ``process``, ``upDown``, …) keep the last step's
    value.  Byte-equivalent to ``read_sets_multi``.
    """
    output_dir = Path(output_dir)
    solve_order = load_solve_order(output_dir.parent)

    out = SimpleNamespace()
    for attr, value in vars(last_s).items():
        if isinstance(value, pd.MultiIndex) and _multi_index_has_solve(value):
            parts = _ordered_parts(output_dir, attr, solve_order)
            if not parts:
                setattr(out, attr, value)
                continue
            setattr(out, attr, _union_set_multiindex(parts, value))
        elif _series_or_df_has_solve(value):
            # Defensive: no varying set is a DataFrame/Series with a solve
            # level today, but the structural test admits one.  Reuse the
            # param-side concat.
            parts = _ordered_parts(output_dir, attr, solve_order)
            if not parts:
                setattr(out, attr, value)
                continue
            setattr(out, attr, _union_param_attr(parts, value))
        else:
            # Static topology set — last-step value (invariant per roll).
            setattr(out, attr, value)
    return out


__all__ = ["has_persisted_slices", "union_params", "union_sets"]
