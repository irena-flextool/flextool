"""Γ.3.B — process method classification: the method mapping, the per-process internal method,
the indirect sets and arcs, and the one-variable efficiency of two-way connections."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, cast_dim, lit_axis, rename_to_axis, schema_dtype
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.input_derivation._method_constants import MIN_TIME_ENABLES, STARTUP_CLASSES

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import (
    _enums,
    _min_time_methods,
    _read_active_solve,
    _try_entities,
    _try_param,
    _try_param_explicit,
)


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
    ("piecewise_linear", "no_startup", "fork_no"): "method_1way_1var_off",
    ("piecewise_linear", "no_startup", "fork_yes"): "method_1way_nvar_off",
    ("piecewise_linear", "linear", "fork_no"): "method_1way_1var_LP",
    ("piecewise_linear", "linear", "fork_yes"): "method_1way_nvar_LP",
    ("piecewise_linear", "binary", "fork_no"): "method_1way_1var_MIP",
    ("piecewise_linear", "binary", "fork_yes"): "method_1way_nvar_MIP",
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
    derivation does).  The process key is the Utf8 column ``_pk``; ``dc_line`` marks the lines of a
    ``dc_power_flow_with_angles`` group.  None when no group overrides anything.
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
           .group_by("p", maintain_order=True).agg(
               pl.col("ct_override").first(),
               (pl.col("method") == "dc_power_flow_with_angles").any().alias("dc_line"))
           .collect())
    if out.height == 0:
        return None
    return out.lazy().select(pl.col("p").alias("_pk"), "ct_override", "dc_line")


def _piecewise_flags(source: "InputSource") -> pl.DataFrame:
    """``(p, multi_line, needs_online, single_line_rising, single_line_flat, auto_startup)`` of the units with a
    ``piecewise_linear_efficiency`` curve, from the whole authored curve (every period;
    :func:`flextool.engine.input._piecewise_curve.unit_flags_frame` -- the same function the input derivation's
    classifier calls).  Memoised on the run's reader (data only)."""
    from flextool.engine.input._param_shapes import _run_memo, piecewise_curve_frame
    from flextool.engine.input._piecewise_curve import unit_flags_frame
    return _run_memo(source, ("piecewise_unit_flags",),
                     lambda: unit_flags_frame(piecewise_curve_frame(source)))


def _classify_process_method(source: "InputSource") -> pl.DataFrame:
    """Build the per-process internal-method classification.

    Returns a DataFrame with schema ``[p, ct, startup, fork, method,
    klass, fix_in_uc]``:

      * ``ct`` — user-facing ``conversion_method`` / ``transfer_method``.
      * ``startup`` — the startup class (``no_startup`` / ``linear`` /
        ``binary``, :func:`~flextool.input_derivation._method_constants.startup_class`)
        of ``startup_method`` (default ``no_startup``, promoted to
        ``linear`` when ``minimum_time_method ∈ {min_uptime,
        min_downtime, both}``; mirrors input_writer.py:1294-1304).
      * ``fix_in_uc`` — the ``startup_method`` is a ``_fix_in_uc`` member
        (``False`` for the promotions and the default).
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
      5. Apply minimum_time → linear-startup override, then the
         piecewise_linear one: a ``piecewise_linear`` unit whose curve
         burns fuel at zero output (the efficiency rises from the first
         point to the second in some period) and has no start-ups gets
         ``linear`` (:func:`_piecewise_flags`).
      6. Resolve fork from arc counts + delayed-set membership; a
         ``piecewise_linear`` unit with two or more lines in some period
         is ``fork_yes`` (as a delayed unit).
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
            "fork": pl.Utf8, "method": pl.Utf8, "klass": pl.Utf8, "fix_in_uc": pl.Boolean,
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

    enabling = sorted(u for u, m in _min_time_methods(source).items()
                      if m in MIN_TIME_ENABLES["min_uptime"] | MIN_TIME_ENABLES["min_downtime"])

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
                .drop("ct_override", "dc_line", "_pk"))
    if sm_lazy is not None:
        lf = lf.join(sm_lazy, on="p", how="left")
    else:
        lf = lf.with_columns(startup_raw=pl.lit(None).cast(pl.Utf8))
    raw = pl.col("startup_raw").cast(pl.Utf8)
    lf = lf.with_columns(
        startup=raw.replace({k: cls for k, (cls, _fix) in STARTUP_CLASSES.items()}).fill_null("no_startup"),
        fix_in_uc=raw.replace_strict({k: fix for k, (_cls, fix) in STARTUP_CLASSES.items()}, default=False,
                                     return_dtype=pl.Boolean).fill_null(False),
    ).drop("startup_raw")
    # Step 5 — minimum_time_method overrides startup → linear when active.
    if enabling:
        lf = lf.with_columns(
            startup=pl.when(
                (pl.col("startup") == "no_startup")
                & (pl.col("klass") == "unit") & pl.col("p").cast(pl.Utf8).is_in(enabling)
            ).then(pl.lit("linear"))
             .otherwise(pl.col("startup"))
        )
    # Step 5b — piecewise_linear: auto linear start-ups and the forced fork of multi-line curves.
    pw_flags = _piecewise_flags(source) if cm_unit is not None and (
        cm_unit.get_column("value").cast(pl.Utf8) == "piecewise_linear").any() else None
    if pw_flags is not None:
        lf = (lf.with_columns(_pk=pl.col("p").cast(pl.Utf8))
                .join(pw_flags.lazy().select(pl.col("p").alias("_pk"), "auto_startup", "multi_line"),
                      on="_pk", how="left")
                .with_columns(pw=(pl.col("ct") == "piecewise_linear") & (pl.col("klass") == "unit"))
                .with_columns(
                    startup=pl.when(pl.col("pw") & (pl.col("startup") == "no_startup")
                                    & pl.col("auto_startup").fill_null(False))
                              .then(pl.lit("linear")).otherwise(pl.col("startup")),
                    pw_multi=pl.col("pw") & pl.col("multi_line").fill_null(False))
                .drop("_pk", "auto_startup", "multi_line", "pw"))
    else:
        lf = lf.with_columns(pw_multi=pl.lit(False))
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
                      | pl.col("is_delayed") | pl.col("pw_multi"))
              .then(pl.lit("fork_yes"))
              .otherwise(pl.lit("fork_no"))
    ).drop("pw_multi")

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
              .select("p", "ct", "startup", "fork", "method", "klass", "fix_in_uc")
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
    (``flex_data/_load_features.py:_load_indirect``).
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
    ``flex_data/_load_features.py:_load_indirect``).  Schema ``[p, source, sink]``.
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
    ``flex_data/_load_features.py:_load_indirect``).
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


def one_variable_efficiency_ignored(source: "InputSource", classified: pl.DataFrame | None = None) -> set[str]:
    """The one-variable connections (``method_2way_1var_off``) whose ``efficiency`` is ignored (W-0591).

    One signed flow cannot model losses: a loss one way would be a gain the other way.  So the efficiency is
    ignored on a line of a ``dc_power_flow_with_angles`` group, and on a ``no_losses_no_variable_cost``
    connection unless both its nodes have an explicit ``unit_of_measure`` and the two differ (then it is a unit
    conversion, the same both ways: W-0589).
    """
    if classified is None:
        classified = _classify_process_method(source)
    one_var = {str(p) for p in classified.filter((pl.col("method") == "method_2way_1var_off")
                                                 & (pl.col("klass") == "connection")).get_column("p").to_list()}
    cnn = _try_entities(source, "connection__node__node")
    if not one_var or cnn is None:
        return set()
    from flextool import units_of_measure as uom
    overrides = _group_transfer_method_overrides(source)
    dc = set() if overrides is None else set(overrides.filter(pl.col("dc_line")).collect().get_column("_pk").to_list())
    uom_rows = _try_param_explicit(source, "node", "unit_of_measure")
    units = {} if uom_rows is None else {
        str(n): uom.normalise(v) for n, v in uom_rows.select(pl.col("name").cast(pl.Utf8), "value").iter_rows()
        if v is not None and str(v).strip()}
    ignored = set()
    for p, n1, n2 in cnn.select(*(pl.col(c).cast(pl.Utf8) for c in ("connection", "node_1", "node_2"))).iter_rows():
        if p in one_var and (p in dc or units.get(n1) is None or units.get(n1) == units.get(n2)
                             or units.get(n2) is None):
            ignored.add(p)
    return ignored


def _one_variable_slope_in_provider(provider, ignored: set[str]) -> None:
    """The provider's ``solve_data/pdtProcess_slope`` (the emit seed, raw efficiency) gets slope 1 on *ignored*
    too, so a CSV reload of the run's provider snapshot (or of a light ``dump_csvs`` that takes the slope from it)
    has the slope of the solved model."""
    if provider is None:
        return
    from flextool.engine.preprocess._flex_data_provider import EvictedFrameError
    key = "solve_data/pdtProcess_slope"
    try:
        seed = provider.get(key)
    except EvictedFrameError:
        return
    if seed is None or seed.height == 0 or "process" not in seed.columns:
        return
    hit = pl.col("process").cast(pl.Utf8).is_in(sorted(ignored))
    one = pl.lit(repr(1.0)) if seed.schema["value"] == pl.Utf8 else pl.lit(1.0)
    provider.put(key, seed.with_columns(value=pl.when(hit).then(one).otherwise(pl.col("value"))))


def apply_one_variable_efficiency(flex_data: object, source: "InputSource", provider=None) -> None:
    """The load-boundary step of K-1: ``p_slope`` is 1 on the one-variable connections whose efficiency is
    ignored (:func:`one_variable_efficiency_ignored`), so the LP and the results are those of efficiency 1 on
    every path (single, rolling, synthetic sub-solves); W-0591 once per solve names those with an efficiency ≠ 1.
    """
    slope = getattr(flex_data, "p_slope", None)
    if slope is None:
        return
    ignored = one_variable_efficiency_ignored(source)
    if not ignored:
        return
    _one_variable_slope_in_provider(provider, ignored)
    frame = slope.frame.collect() if isinstance(slope.frame, pl.LazyFrame) else slope.frame
    hit = pl.col("p").cast(pl.Utf8).is_in(sorted(ignored))
    changed = frame.filter(hit & (pl.col("value") != 1.0))
    if changed.height == 0:
        return
    flex_data.p_slope = Param(slope.dims, frame.with_columns(
        value=pl.when(hit).then(pl.lit(1.0)).otherwise(pl.col("value"))), name=slope.name)
    details = {str(p): f"efficiency {lo:g}" if lo == hi else f"efficiency {lo:g} to {hi:g}"
               for p, lo, hi in changed.group_by(pl.col("p").cast(pl.Utf8))
               .agg((1.0 / pl.col("value")).min().alias("lo"), (1.0 / pl.col("value")).max().alias("hi"))
               .sort("p").iter_rows()}
    from flextool import run_warnings
    solve = _read_active_solve(Path("."), provider=provider) if provider is not None else None
    run_warnings.warn_grouped("W-0591", list(details), solve=solve, details=details)
