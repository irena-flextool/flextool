"""Provider access and the CSV-shape readers the loaders share (long, wide per entity, unit size,
capacity, p_flow_max, sliced params, step predecessors), and the load profiler."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.preprocess._provider_io import _provider_key


def _loadflex_prof(label: str) -> None:
    """Env-gated (FLEXTOOL_LOADFLEX_PROFILE=1) RSS print to stderr. No-op otherwise."""
    import os
    import sys
    if os.environ.get("FLEXTOOL_LOADFLEX_PROFILE") != "1":
        return
    try:
        with open("/proc/self/status", encoding="utf-8") as _f:
            for _ln in _f:
                if _ln.startswith("VmRSS:"):
                    _rss = int(_ln.split()[1]) / (1024 * 1024)
                    break
            else:
                _rss = -1.0
        sys.stderr.write(f"[loadflex profile] step={label}\trss_gb={_rss:.3f}\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provider access helpers.
#
# The loader reads only the sub-solve's in-memory :class:`FlexDataProvider`; it never reads ``input/`` or
# ``solve_data/`` on disk, so CSV files left in a reused work folder (a ``--csv-dump``, an earlier run) cannot
# change the model.  The ``path`` arguments only name the key (``_provider_key(path)``) and appear in error
# messages.  A ``provider`` of ``None`` is a programming error and raises.

def _require_provider(provider: "object | None", helper: str) -> None:
    """Raise ``ValueError`` when *provider* is ``None``."""
    if provider is None:
        raise ValueError(
            f"{helper} requires a FlexDataProvider; the loader never reads the work folder "
            "(the disk fallback was removed)"
        )


def _provider_has(provider: "object | None", name: str,
                   path: "Path | str") -> bool:
    """True iff *provider* has *name*.  A file at *path* is ignored."""
    _require_provider(provider, "_provider_has")
    return provider.has(name)


def _provider_read(provider: "object | None", name: str,
                    path: "Path | str") -> pl.DataFrame:
    """Return the frame for *name* from *provider*.

    Raises ``FileNotFoundError`` when the provider has no *name* (the error a read of a missing CSV gave
    before, so the callers' ``except`` clauses behave as they did); a file at *path* is ignored.
    """
    _require_provider(provider, "_provider_read")
    if provider.has(name):
        return provider.get(name)
    raise FileNotFoundError(f"provider has no {name!r} (the loader does not read {path})")


def _provider_pick(provider: "object | None",
                    *candidates: "tuple[str, Path]") -> "Path | None":
    """Return the ``Path`` of the first ``(name, path)`` candidate whose *name* the provider has; ``None`` if
    none.  Files on disk are ignored.
    """
    _require_provider(provider, "_provider_pick")
    for name, path in candidates:
        if provider.has(name):
            return Path(path)
    return None


# ---------------------------------------------------------------------------
# CSV-shape helpers (same three shapes as before)

def _read_long(path: Path, *, drop=("solve",), rename=None,
               cast_value: bool = True,
               provider: "object | None" = None) -> pl.DataFrame:
    df = _provider_read(provider, _provider_key(path), path)
    df = df.drop([c for c in drop if c in df.columns])
    if rename:
        df = df.pipe(rename_to_axis, rename)
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
        if rename:
            out = out.pipe(rename_to_axis, rename)
        return out
    # Legacy wide-per-entity format.
    df = df.drop("solve")
    id_cols = ["period", "time"]
    value_cols = [c for c in df.columns if c not in id_cols]
    out = (df.unpivot(on=value_cols, index=id_cols, variable_name="entity",
                      value_name=value_col)
             .pipe(rename_to_axis, {"period": "d", "time": "t"}))
    if rename:
        out = out.pipe(rename_to_axis, rename)
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


def _read_capacity(all_existing_path: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    # ``p_entity_all_existing`` is the capacity in service at the start of the solve per period (the existing
    # cohort and the investments earlier solves realized, minus the carried retirements), which is what the
    # ``p_entity_dispatch_capacity_max`` formula uses.  ``_chain_params.emit_existing_capacity`` (called for
    # every sub-solve by ``_solve_time.run`` batch 26) populates ``solve_data/p_entity_all_existing`` in the
    # Provider, so the cascade always has this key available.
    df = _provider_read(
        provider, "solve_data/p_entity_all_existing", all_existing_path)
    if "solve" in df.columns:
        df = df.drop("solve")
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
    out = (df.pipe(rename_to_axis, {"process": "p", "period": "d", "time": "t"})
             .select("p", "source", "sink", "d", "t", "value"))
    # ``p_flow_max.csv`` is emitted with ``value`` as ``repr(float)`` — a
    # Utf8 column — so heterogeneous CSV leaves round-trip byte-identically
    # (see :func:`flextool.engine.preprocess.emit._arc_unions.derive_p_flow_max`,
    # whose ``_rows_to_frame`` types every column ``pl.Utf8``).  This frame is
    # the authoritative seed for ``p_flow_upper`` in the native cascade.  Cast
    # ``value`` back to Float64 here: ``p_flow_upper`` feeds numeric
    # ``pl.concat`` (``model/core/`` flow_upper_rhs, indirect_pss ⧺ direct_pss_dt)
    # and ``Param × Param`` paths that raise ``SchemaError`` /
    # ``InvalidOperationError`` on a String ``value`` (the concat fires once
    # the per-(p,source,sink,d,t) frame carries indirect-process rows — e.g.
    # the mt_3h solve).  ``strict=False`` mirrors the cast in
    # :func:`_slice_param` above.
    if out.schema["value"] == pl.Utf8:
        out = out.with_columns(
            value=pl.col("value").cast(pl.Float64, strict=False))
    return out


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
    # :func:`flextool.engine.preprocess.emit._pdt_params.derive_pdtProcess`).
    # When a numeric param (e.g. ``availability``) is sliced out for use
    # as a polars-numeric Param ``value`` column, the Utf8 must be cast
    # back to ``Float64`` at the producer — otherwise downstream
    # ``Param × Param`` multiplications (e.g. ``flow_upper_rhs *
    # p_process_availability`` in ``model._max_flow_rhs``) raise
    # ``InvalidOperationError: arithmetic on string and numeric``.
    # ``strict=False`` mirrors the casts in ``_read_long`` / pdtNode
    # path (flex_data.py:958) — non-parseable values become null and the
    # downstream consumer's ``fill_null(0.0)`` handles them.
    if "value" in out.columns and out.schema["value"] == pl.Utf8:
        out = out.with_columns(
            value=pl.col("value").cast(pl.Float64, strict=False))
    cols = [rename.get(entity_col, entity_col), "d"] + (["t"] if has_time else []) + ["value"]
    return out.select(cols)


# The storage_roll_end kinds that tie a node's horizon (or first-scope) end to its start: such a node gets no
# use_reference_value pin in the roll.  The nested-fix kinds (fix_quantity, usage_cap) only restate the upper
# solve's target and leave the node's own horizon method as it was; scope_close (a bind_intraperiod_blocks period
# that a roll closes) drops the pin only at the closed step (model.core._storage_state._eq_storage_reference_value).
ROLL_END_TIED_KINDS = ("anchor", "close", "anchor_share", "close_share", "wrap_end", "scope_carry", "scope_anchor")


def _roll_end_tied_nodes(roll_end: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """The nodes (n) of ``storage_roll_end`` with a kind of :data:`ROLL_END_TIED_KINDS`; None when none."""
    if roll_end is None or roll_end.height == 0:
        return None
    out = roll_end.filter(pl.col("kind").is_in(list(ROLL_END_TIED_KINDS))).select("n").unique()
    return out if out.height > 0 else None


def _within_period_lag(dtttdt: pl.DataFrame,
                       period_branch: "pl.DataFrame | None" = None) -> "pl.DataFrame | None":
    """``(d, t, d_previous, t_previous_within_period)``: the bind_within_period predecessor of every step of ``dtttdt``.

    Inside a period it is the step's ``t_previous`` (across block boundaries too: blocks chain inside the
    period).  The first step of a period wraps to the period's OWN last step — the state cycles within each
    period.  ``dtttdt``'s first step of a period continues from another period (``d_previous`` ≠ ``d``) or, for
    a period that wraps onto itself, already points at its own last step.  A period's last step is the step
    no same-period step follows.  ``d_previous`` is ``d`` on these rows.

    A split realization roll (``period_branch`` (d, b) with a label that has steps, ``_step_relations``) splits a
    period into paths: the label's jump steps and one branch period's tail each.  The period's part in the roll is
    one path, so the lag crosses the seam: each branch period's first step continues from the label's last jump
    step, and the label's first step wraps to the last step of its last branch period in name order, d* (the
    model ties every other branch period's end to d*'s, ``core._storage_state._eq_storage_split_path_end``).
    """
    t_dtype = dtttdt.schema["t"]
    frame = (dtttdt.select("d", "t", "t_previous", "d_previous")
             .with_columns(pl.col("t_previous").cast(t_dtype, strict=False)))
    same = pl.col("d_previous").cast(pl.Utf8) == pl.col("d").cast(pl.Utf8)
    inner = frame.filter(same)
    first = frame.filter(~same)
    followed = inner.select("d", pl.col("t_previous").alias("t"))
    last = (frame.select("d", "t").unique()
            .join(followed, on=["d", "t"], how="anti")
            .rename({"t": "t_previous_within_period"}))
    multi = last.group_by("d").len().filter(pl.col("len") > 1)
    if multi.height > 0:
        raise ValueError(
            "bind_within_period: the steps of period(s) "
            f"{sorted(str(x) for x in multi['d'].to_list())} do not form one chain in step_previous "
            "(more than one last step)")
    first = first.select("d", "t").join(last, on="d", how="inner").with_columns(d_previous=pl.col("d"))
    from flextool.engine.preprocess._step_relations import _split_labels
    lab = {} if period_branch is None or period_branch.height == 0 else _split_labels(period_branch, frame)
    if lab:
        ends = dict(last.select(pl.col("d").cast(pl.Utf8), pl.col("t_previous_within_period").cast(pl.Utf8)).rows())
        seam = {**{f: p for p, fs in lab.items() for f in fs}, **{p: max(fs) for p, fs in lab.items() if fs}}
        src = pl.col("d").cast(pl.Utf8).replace_strict(seam, default=None)
        first = first.with_columns(
            d_previous=pl.coalesce(src, pl.col("d").cast(pl.Utf8)).cast(frame.schema["d"]),
            t_previous_within_period=pl.coalesce(src.replace_strict(ends, default=None),
                                                 pl.col("t_previous_within_period").cast(pl.Utf8)).cast(t_dtype))
    out = pl.concat([
        inner.select("d", "t", pl.col("d").alias("d_previous"), pl.col("t_previous").alias("t_previous_within_period")),
        first.select("d", "t", "d_previous", "t_previous_within_period"),
    ], how="vertical")
    return out.sort("d", "t") if out.height > 0 else None


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
    from flextool.engine.preprocess._step_relations import null_split_dtttdt
    return null_split_dtttdt(out.select(keep), _split_period_branch(path.parent, provider))


def _active_solve_name(sd: Path, provider: "object | None") -> str | None:
    """The active solve (``solve_data/solve_current``), for messages; None without a provider or the frame."""
    if provider is None:
        return None
    from flextool.engine.preprocess.params._derived import _read_active_solve
    return _read_active_solve(sd.parent, provider=provider)


def _split_period_branch(sd: Path, provider: "object | None") -> pl.DataFrame | None:
    """``solve_data/period__branch`` as ``(d, b)``, for the split-roll placeholders; None without it."""
    from flextool.engine.preprocess.params._branch import period_branch_full_df
    return period_branch_full_df(sd.parent, provider=provider)
