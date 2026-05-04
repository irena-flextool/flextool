"""Delayed-process feature (canonical use case: hydro / river chains).

This module covers the .mod's ``process_delayed`` family — processes
whose sink-side delivery at time ``t`` is fed by source-side flows at
*earlier* times ``t_``, weighted by a per-process delay profile.  The
canonical use case is a **hydro / river network**: water released
upstream takes a known time to arrive downstream, so the
``conversion_indirect`` balance for the downstream unit must aggregate
upstream inflows over a delay window rather than read instantaneous
flow.  Other plausible uses include thermal-storage charging chains
and freight-style logistics.  ``water_pump`` / ``water_pump_delayed``
are the test fixtures that currently exercise this.

**Note on demand response.**  An earlier draft of this module also
handled DR (``dr_decrease_demand`` / ``dr_increase_demand`` /
``dr_shift_demand``).  DR was promoted out: those scenarios do not use
the delay tables — DR is just a regular storage + process pattern with
``bind_within_solve`` storage and a sign-bearing inflow at the demand
node, already handled by the storage and process blocks in
``flextool.model.build_flextool``.  The DR fixtures' delay CSVs are
empty, so this module's ``has_feature`` returns False on them.

================================================================
What this module implements
================================================================

The .mod's ``conversion_indirect`` constraint (flextool.mod:2343)
splits the source-side flow term into an *undelayed* part (current-
time) and a *delayed* part (time-shifted via ``dtt__delay_duration``,
weighted by ``p_process_delay_weight``)::

    sum {source : (p, source) in process_source_undelayed}
        + v_flow[p, source, p, d, t] * unitsize * source_coef
    + sum {source : (p, source) in process_source_delayed}
        + sum {(d, t_, t, td) in dtt__delay_duration
                : (p, td) in process_delayed__duration}
            + v_flow[p, source, p, d, t_]
                * unitsize * source_coef * delay_weight[p, td]

That is, for delayed (p, source), the inflow to the conversion
balance at sink-side time ``t`` is the *weighted sum* of source-side
flows at times ``t_`` paired with ``t`` through ``dtt__delay_duration``.

The .mod also has a *commented-out* analogous shift in
``nodeBalance_eq`` (flextool.mod:2148-2154) — kept commented in
upstream .mod, so we do **not** mirror it here.  If a future scenario
needs it, the same machinery applies to ``flow_from_nodeBalance_*``.

================================================================
Module API
================================================================

    has_feature(d) -> bool
        True iff ``d.process_delayed`` is non-empty.

    load_data(inp_dir, sd_dir) -> dict[str, ...]
        Reads the delay CSVs.  Returns a dict of new FlexData fields:
            process_delayed, process_source_delayed,
            process_source_undelayed,
            dtt__delay_duration, p_process_delay_weight.
        All values are ``None`` (or empty) when the feature is inactive.

    delayed_input_expr(d, v_flow) -> Expr | None
        Returns the delay-shifted source-side LHS aggregate that the
        ``conversion_indirect`` emission in ``model.py`` threads into
        ``lhs_terms["input_delayed"]`` alongside the undelayed term.

    add_constraints(m, d, vars) -> None
        Currently a no-op (the delay term is woven into the existing
        ``conversion_indirect`` via ``delayed_input_expr`` rather than
        emitting a separate constraint).

    add_objective_terms(m, d, vars, op_factor) -> None
        No-op.  Delayed processes do not add objective terms; their
        costs propagate through commodity prices and slack penalties
        already wired into ``build_flextool``.

================================================================
Integration with ``conversion_indirect`` in ``model.py``
================================================================

``model.py``'s ``conversion_indirect`` builds the source-side input
term from ``d.process_input_flows``.  When the delay feature is
active, that block anti-joins ``process_input_flows`` against
``d.process_delayed`` (so the undelayed term skips delayed rows) and
adds the delay-shifted contribution by reading
``delayed_input_expr(d, v_flow)`` from this module — woven into the
same ``add_cstr`` call as the named ``lhs_terms["input_delayed"]``
entry, so a single constraint per (p, d, t) is preserved.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

from polar_high_opt import Sum, Where, Param
# Engine imports kept light — we don't introduce new variable types.

from ._input_source import _read_csv_file


# ---------------------------------------------------------------------------
# Feature detection

def has_feature(d) -> bool:
    """True iff ``d`` carries non-empty delayed-flow data."""
    pd_set = getattr(d, "process_delayed", None)
    return pd_set is not None and pd_set.height > 0


# ---------------------------------------------------------------------------
# Data loading

def load_data(inp_dir: str | Path, sd_dir: str | Path) -> dict:
    """Read the delay-related solve_data CSVs.

    Reads from ``solve_data/`` (where flextool.mod's preprocessing emits
    the delay sets).  ``inp_dir`` is accepted for API symmetry with the
    other ``_load_*`` helpers in ``input.py`` but is not currently used —
    the canonical sources are all in ``solve_data/`` (see flextool.mod
    lines 525-527).

    Returns a dict whose keys correspond to (proposed) ``FlexData`` field
    names.  When the feature is inactive (every scenario with no delayed
    processes) every value is ``None``.

    Proposed FlexData fields::

        process_delayed              pl.DataFrame | None  # cols: (p,)
        process_delayed__duration    pl.DataFrame | None  # cols: (p, td)
        process_source_delayed       pl.DataFrame | None  # cols: (p, source)
        process_source_undelayed     pl.DataFrame | None  # cols: (p, source)
        process_source_sink_delayed  pl.DataFrame | None  # cols: (p, source, sink)
        process_source_sink_undelayed pl.DataFrame | None # cols: (p, source, sink)
        dtt__delay_duration          pl.DataFrame | None  # cols: (d, t_source, t_sink, td)
        p_process_delay_weight       Param | None         # dims: (p, td)
    """
    sd = Path(sd_dir)

    blank = dict(
        process_delayed              = None,
        process_delayed__duration    = None,
        process_source_delayed       = None,
        process_source_undelayed     = None,
        process_source_sink_delayed  = None,
        process_source_sink_undelayed = None,
        dtt__delay_duration          = None,
        p_process_delay_weight       = None,
    )

    pd_path = sd / "process_delayed.csv"
    if not pd_path.exists():
        return blank
    pd_df = _read_csv_file(pd_path)
    if pd_df.height == 0:
        # Header-only file — flextool emits these even when no process is
        # delayed (e.g. on DR scenarios that don't use the delay feature).
        # Treat as inactive.
        return blank

    pd_df = pd_df.rename({"process": "p"}) if "process" in pd_df.columns else pd_df

    # process_delayed__duration: (process, delay_duration)
    pdd_path = sd / "process_delayed__duration.csv"
    pdd_df = None
    if pdd_path.exists():
        raw = _read_csv_file(pdd_path)
        if raw.height > 0:
            pdd_df = raw.rename(
                {c: r for c, r in [("process", "p"), ("delay_duration", "td")]
                 if c in raw.columns}
            ).select("p", "td")

    # process_source_(un)delayed: (process, source) frames
    def _read_pse(name: str) -> pl.DataFrame | None:
        p = sd / f"{name}.csv"
        if not p.exists(): return None
        df = _read_csv_file(p)
        if df.height == 0: return None
        if "process" in df.columns: df = df.rename({"process": "p"})
        return df.select("p", "source")

    pse_delayed   = _read_pse("process_source_delayed")
    pse_undelayed = _read_pse("process_source_undelayed")

    # process_source_sink_(un)delayed: (process, source, sink) frames
    def _read_psse(name: str) -> pl.DataFrame | None:
        p = sd / f"{name}.csv"
        if not p.exists(): return None
        df = _read_csv_file(p)
        if df.height == 0: return None
        if "process" in df.columns: df = df.rename({"process": "p"})
        return df.select("p", "source", "sink")

    psse_delayed   = _read_psse("process_source_sink_delayed")
    psse_undelayed = _read_psse("process_source_sink_undelayed")

    # Filter out (p, source) pairs where p_process_source_flow_coefficient == 0:
    # the .mod's conversion_indirect LHS multiplies each source-side flow by
    # this coefficient, so a zero coef effectively drops the row from the
    # input balance.  Mirror that filter on the delayed side — _load_indirect
    # already does the same on the undelayed side.  Without this, water_pump's
    # west→water_pump delayed input is double-counted in conversion_indirect,
    # forcing the LP to dispatch extra battery/coal to compensate (visible
    # as a ~0.099% gap on test_a_lot's multi-period parity).
    inp = Path(inp_dir)
    src_path = inp / "p_process_source_flow_coefficient.csv"
    if src_path.exists():
        srcdf = _read_csv_file(src_path)
        if srcdf.height > 0 and "p_process_source_flow_coefficient" in srcdf.columns:
            zero_src = (srcdf
                .rename({"process": "p",
                         "p_process_source_flow_coefficient": "coef"})
                .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
                .filter(pl.col("coef") == 0.0)
                .select("p", "source"))
            if zero_src.height > 0:
                if pse_delayed is not None:
                    pse_delayed = pse_delayed.join(
                        zero_src, on=["p", "source"], how="anti")
                if psse_delayed is not None:
                    psse_delayed = psse_delayed.join(
                        zero_src, on=["p", "source"], how="anti")

    # dtt__delay_duration: (period, time_source, time_sink, delay_duration)
    dtt_path = sd / "dtt__delay_duration.csv"
    dtt_df = None
    if dtt_path.exists():
        raw = _read_csv_file(dtt_path)
        if raw.height > 0:
            rename_map = {}
            if "period" in raw.columns:      rename_map["period"] = "d"
            if "time_source" in raw.columns: rename_map["time_source"] = "t_source"
            if "time_sink" in raw.columns:   rename_map["time_sink"] = "t_sink"
            if "delay_duration" in raw.columns: rename_map["delay_duration"] = "td"
            dtt_df = raw.rename(rename_map).select("d", "t_source", "t_sink", "td")

    # p_process_delay_weight: (process, delay_duration, value)
    pw_path = sd / "p_process_delay_weight.csv"
    pw_param = None
    if pw_path.exists():
        raw = _read_csv_file(pw_path)
        if raw.height > 0:
            rename_map = {}
            if "process" in raw.columns: rename_map["process"] = "p"
            if "delay_duration" in raw.columns: rename_map["delay_duration"] = "td"
            pw_long = raw.rename(rename_map).select("p", "td", "value")
            pw_param = Param(("p", "td"), pw_long)

    return dict(
        process_delayed              = pd_df.select("p"),
        process_delayed__duration    = pdd_df,
        process_source_delayed       = pse_delayed,
        process_source_undelayed     = pse_undelayed,
        process_source_sink_delayed  = psse_delayed,
        process_source_sink_undelayed = psse_undelayed,
        dtt__delay_duration          = dtt_df,
        p_process_delay_weight       = pw_param,
    )


# ---------------------------------------------------------------------------
# Constraint contribution

def delayed_input_expr(d, v_flow):
    """Return an Expr representing the delay-shifted source-side input
    contribution to ``conversion_indirect``.

    Shape: an Expr with open dims ``(p, d, t)`` (where ``t`` is the
    *sink-side* time, matching ``conversion_indirect``'s axes).

    Definition (mirrors flextool.mod:2348-2353)::

        Σ_{source : (p, source) ∈ process_source_delayed}
        Σ_{(d, t_source, t_sink, td) ∈ dtt__delay_duration
                 : (p, td) ∈ process_delayed__duration}
            v_flow[p, source, p, d, t_source]
            · unitsize[p] · delay_weight[p, td]

    Implementation:

      * Rename v_flow's ``t`` → ``t_source`` so the index aligns with
        ``dtt__delay_duration``.
      * Rename v_flow's ``sink`` → ``p_sink`` and inner-join against the
        delayed input flows (where sink = p, the process's own indirect
        balance node).
      * Inner-join with ``dtt__delay_duration`` on (d, t_source) and with
        ``process_delayed__duration`` on (p, td) — this attaches the
        sink-side ``t`` (renamed back to ``t``) and the delay weight.
      * Multiply by ``p_unitsize[p]`` and ``p_process_delay_weight[p, td]``.
      * ``Sum`` over (source, td) leaves dims (p, d, t).

    Returns ``None`` when there's no delayed-process data (caller should
    use a no-op).
    """
    if not has_feature(d):
        return None
    if d.dtt__delay_duration is None or d.p_process_delay_weight is None:
        return None
    if d.process_source_delayed is None or d.process_delayed__duration is None:
        return None
    if d.p_unitsize is None:
        return None

    # Delayed input flows: (p, source, sink=p) — restrict process_input_flows
    # (or build directly from process_source_sink_delayed where sink == p).
    psse_delayed = d.process_source_sink_delayed
    if psse_delayed is None or psse_delayed.height == 0:
        return None
    indirect_inputs_delayed = psse_delayed.filter(
        pl.col("sink") == pl.col("p")
    ).select("p", "source", "sink")
    if indirect_inputs_delayed.height == 0:
        return None

    # Build the delay-mapping table: (p, source, sink, d, t_source, t, td, weight)
    # Step 1: cross-product of (p, source, sink) ∈ delayed inputs with
    #         (p, td) ∈ process_delayed__duration  →  (p, source, sink, td)
    pdd = d.process_delayed__duration
    pst_td = indirect_inputs_delayed.join(pdd, on="p", how="inner")
    if pst_td.height == 0:
        return None

    # Step 2: cross-join with dtt__delay_duration on (d, t_source, t_sink, td)
    # via td (and the per-period mapping).  dtt has columns
    # (d, t_source, t_sink, td); join on td, get all (d, t_source, t_sink).
    dtt = d.dtt__delay_duration
    full_map = pst_td.join(dtt, on="td", how="inner")
    if full_map.height == 0:
        return None

    # Step 3: rename t_sink → t for the constraint axes; keep t_source
    # as the v_flow time index, td for the weight lookup.
    full_map = full_map.rename({"t_sink": "t"})
    # Cols now: (p, source, sink, td, d, t_source, t)

    # The Where filter against v_flow needs a frame whose columns are
    # exactly the dims of v_flow (with t renamed to t_source on the
    # variable side).  We use Sum with an explicit ``where`` argument
    # in two steps: rename + Where.
    #
    # v_flow has dims (p, source, sink, d, t).  We first build a
    # virtual variable with dims (p, source, sink, d, t_source) by
    # renaming the v_flow frame, then Where it against full_map (which
    # has all those plus t and td).  The Where adds t and td as new
    # open dims.
    from polar_high_opt.engine import Var as _Var
    v_flow_at_source = _Var(
        name=v_flow.name + "__at_t_source",
        dims=("p", "source", "sink", "d", "t_source"),
        frame=v_flow.frame.rename({"t": "t_source"}),
        lower=v_flow.lower, upper=v_flow.upper, integer=v_flow.integer,
    )

    # Where(v_flow_at_source, full_map) joins on (p, source, sink, d, t_source)
    # and adds (td, t) as new open dims — matching what we need.
    expr = Where(v_flow_at_source, full_map)
    # Multiply by unitsize[p] and delay weight[p, td].
    expr = expr * d.p_unitsize * d.p_process_delay_weight
    # Per flextool.mod:2573, the delayed source-side term also carries the
    # ``p_process_source_flow_coefficient[p, source]`` multiplier — same
    # factor that the undelayed source-side term in ``conversion_indirect``
    # applies (model.py:1424-1425).  When the Param is None (every coef=1
    # default) the multiplication is skipped to keep the Expr's open dims
    # unchanged; when present it covers all surviving (p, source) pairs
    # (defaulted to 1.0 by the loader where the CSV is silent).  Without
    # this, fixtures combining a delay with a non-default source coefficient
    # diverge from flextool — see ``tests/test_flex_delay_source_coef.py``.
    if getattr(d, "p_process_source_flow_coef", None) is not None:
        expr = expr * d.p_process_source_flow_coef

    # Sum over source, sink, td, t_source → leaves (p, d, t).
    # NOTE: original version listed only ("source", "sink", "td") — but
    # t_source is also an open dim of v_flow_at_source after the Where
    # join.  Without summing it, the constraint engine refuses the term
    # because t_source is not in the constraint axes (p, d, t).  See
    # ``audit/integration_manifest.md`` "## merge step 4 issues".
    expr = Sum(expr, over=("source", "sink", "td", "t_source"))
    return expr


def add_constraints(m, d, vars: dict) -> None:
    """Add the delayed-flow constraint contribution.

    *PENDING DOWNSTREAM PATCH (see module docstring):*  The merge agent
    must wire the Expr returned by :func:`delayed_input_expr` into
    ``model.py``'s ``conversion_indirect`` ``add_cstr`` call as an
    additional ``lhs_terms`` entry, **and** filter
    ``d.process_input_flows`` to exclude delayed processes.  Until that
    patch lands, this function is a no-op (delayed processes will not
    solve correctly — their source-side input will be summed at the
    *current* time without the delay shift).

    ``vars`` is the dict of decision-variable handles passed in by the
    caller; it should contain ``v_flow`` for the Expr construction.  When
    the merge wiring is complete, this function will simply call
    :func:`delayed_input_expr` and feed the result into the existing
    ``conversion_indirect`` constraint via the engine's named-term API.
    """
    if not has_feature(d):
        return
    # Self-contained: nothing to emit until model.py is patched.
    # The Expr is exposed via `delayed_input_expr` for the merge agent.
    return


def add_objective_terms(m, d, vars: dict, op_factor):
    """No delay-specific objective terms.

    Costs are propagated through commodity prices on the source flows
    (the ``v_flow`` referenced by the delay-shifted aggregate is the
    same variable priced in the existing commodity-buy obj term) and
    through any storage-state slacks the surrounding scenario already
    emits.  Returns ``None`` to signal "no contribution".
    """
    return None
