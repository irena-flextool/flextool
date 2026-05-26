"""Δ.31 — In-memory replacement for the CSV-based ``read_sets``.

Translates the polars ``FlexData`` set-frames into the pandas
:class:`pd.Index` / :class:`pd.MultiIndex` shapes the downstream
``out_*`` modules consume.  CSV reads are gone entirely — every
attribute on the returned :class:`SimpleNamespace` maps to a FlexData
field (or, for a small handful of derived sets, a simple in-memory
projection of one or more FlexData fields).

Failure mode: every helper raises loudly when a FlexData field is
absent or has an unexpected schema.  Empty sets are returned as
empty :class:`pd.Index` / :class:`pd.MultiIndex` with the correct
``names`` / ``dtype`` so downstream consumers can call
``.intersection`` / ``.get_level_values`` etc. without crashing.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Sequence

import pandas as pd
import polars as pl

from flextool.process_outputs._inmemory_helpers import (
    empty_index,
    empty_multi_index,
    long_dim,
    to_index,
    to_multi_index,
)

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Per-set helpers
# ---------------------------------------------------------------------------


def _index_or_empty(
    frame_pl: "pl.DataFrame | None",
    *,
    dim: str,
    name: str | None = None,
) -> pd.Index:
    """Return an :class:`pd.Index` for ``frame_pl[dim]`` or an empty Index."""
    name = name if name is not None else long_dim(dim)
    if frame_pl is None or frame_pl.height == 0 or dim not in frame_pl.columns:
        return empty_index(name=name)
    return to_index(frame_pl, dim=dim, name=name)


def _multi_index_or_empty(
    frame_pl: "pl.DataFrame | None",
    *,
    dims: Sequence[str],
    names: Sequence[str] | None = None,
) -> pd.MultiIndex:
    """Return a :class:`pd.MultiIndex` for ``frame_pl[*dims]`` or an
    empty MultiIndex with the given names."""
    if names is None:
        names = [long_dim(d) for d in dims]
    if frame_pl is None or frame_pl.height == 0:
        return empty_multi_index(list(names))
    if not all(d in frame_pl.columns for d in dims):
        return empty_multi_index(list(names))
    return to_multi_index(frame_pl, dims=dims, names=names)


def _multi_index_with_solve(
    frame_pl: "pl.DataFrame | None",
    *,
    solve_name: str,
    dims: Sequence[str],
    names: Sequence[str],
) -> pd.MultiIndex:
    """Like :func:`_multi_index_or_empty` but prepend a constant
    ``solve`` level.  Used for sets whose legacy CSV layout was
    ``solve, period, …`` but FlexData drops the solve column.
    """
    if frame_pl is None or frame_pl.height == 0:
        return empty_multi_index(["solve"] + list(names))
    if not all(d in frame_pl.columns for d in dims):
        return empty_multi_index(["solve"] + list(names))
    pdf = frame_pl.with_columns(pl.lit(solve_name).alias("solve")).select(
        "solve", *list(dims)
    ).to_pandas()
    return pd.MultiIndex.from_frame(pdf, names=["solve"] + list(names))


def _commodity_node_from_flex(
    flex_data: "FlexData",
    *,
    attrs: Sequence[tuple[str, str]],
) -> pd.MultiIndex:
    """Derive a ``(commodity, node)`` MultiIndex from FlexData flow_* frames.

    Each entry of ``attrs`` is a ``(field_name, node_column)`` tuple where
    ``field_name`` is a ``flex_data`` attribute that — when present — is a
    polars frame with at least columns ``c`` and ``node_column``.  The
    union of ``(c, node_column)`` across all listed frames is returned
    as a deterministic, deduplicated ``pd.MultiIndex`` with names
    ``("commodity", "node")``.  Returns an empty MultiIndex when no
    listed field is populated.

    All Enum columns are cast to Utf8 before crossing the pandas
    boundary to avoid Enum-vs-Utf8 join-key mismatches downstream
    (cf. ``specs/enum_dtype_refactor_handoff.md``).
    """
    pieces: list[pl.DataFrame] = []
    for field_name, node_col in attrs:
        fr = getattr(flex_data, field_name, None)
        if fr is None or fr.height == 0 or node_col not in fr.columns:
            continue
        c_col = "c" if "c" in fr.columns else "commodity"
        if c_col not in fr.columns:
            continue
        pieces.append(
            fr.select(
                pl.col(c_col).cast(pl.Utf8).alias("commodity"),
                pl.col(node_col).cast(pl.Utf8).alias("node"),
            )
        )
    if not pieces:
        return empty_multi_index(["commodity", "node"])
    df = (
        pl.concat(pieces, how="vertical_relaxed")
          .unique()
          .sort("commodity", "node")
          .to_pandas()
    )
    if df.empty:
        return empty_multi_index(["commodity", "node"])
    return pd.MultiIndex.from_frame(df, names=["commodity", "node"])


def _process_commodity_node_from_flex(
    flex_data: "FlexData",
    *,
    attrs: Sequence[tuple[str, str]],
) -> pd.MultiIndex:
    """Derive a ``(process, commodity, node)`` MultiIndex from FlexData
    flow_* frames.

    Each entry of ``attrs`` is a ``(field_name, node_column)`` tuple.  The
    union of ``(p, c, node_column)`` rows across all listed frames is
    returned as a deterministic, deduplicated ``pd.MultiIndex`` with
    names ``("process", "commodity", "node")``.
    """
    pieces: list[pl.DataFrame] = []
    for field_name, node_col in attrs:
        fr = getattr(flex_data, field_name, None)
        if fr is None or fr.height == 0 or node_col not in fr.columns:
            continue
        c_col = "c" if "c" in fr.columns else "commodity"
        p_col = "p" if "p" in fr.columns else "process"
        if c_col not in fr.columns or p_col not in fr.columns:
            continue
        pieces.append(
            fr.select(
                pl.col(p_col).cast(pl.Utf8).alias("process"),
                pl.col(c_col).cast(pl.Utf8).alias("commodity"),
                pl.col(node_col).cast(pl.Utf8).alias("node"),
            )
        )
    if not pieces:
        return empty_multi_index(["process", "commodity", "node"])
    df = (
        pl.concat(pieces, how="vertical_relaxed")
          .unique()
          .sort("process", "commodity", "node")
          .to_pandas()
    )
    if df.empty:
        return empty_multi_index(["process", "commodity", "node"])
    return pd.MultiIndex.from_frame(
        df, names=["process", "commodity", "node"],
    )


def _node_types(flex_data: "FlexData") -> dict[str, str]:
    r"""Return ``{node: p_node_type}`` for every node, defaulting to
    ``balance``.

    FlexData carries the type-discrimination through dedicated set
    frames (``nodeBalance``, ``nodeState``).  We synthesize the
    ``p_node_type`` semantics from those:

    * ``storage``               — node in nodeState
    * ``balance``               — node in nodeBalance \ nodeState
    * ``balance_within_period`` — node in nodeBalance with the
      ``balance_within_period`` flag (no FlexData field carries this
      flag standalone; treat as default for now)
    * ``commodity``             — node attached to a commodity (via
      ``commodity_node``)
    """
    nodes = []
    if (flex_data.nodeBalance is not None
            and flex_data.nodeBalance.height > 0):
        nodes = flex_data.nodeBalance["n"].to_list()
    state_nodes = set()
    if (flex_data.nodeState is not None
            and flex_data.nodeState.height > 0):
        state_nodes = set(flex_data.nodeState["n"].to_list())

    types: dict[str, str] = {}
    for n in nodes:
        if n in state_nodes:
            types[n] = "storage"
        else:
            types[n] = "balance"
    return types


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def read_sets(
    flex_data: "FlexData",
    solution: "Solution",
    *,
    solve_name: str = "solve",
) -> SimpleNamespace:
    """Translate ``FlexData`` set frames into the legacy ``s`` namespace.

    Parameters
    ----------
    flex_data : FlexData
        The polars input bundle the LP was built from.
    solution : polar_high.Solution
        Currently unused — kept for signature symmetry with
        :func:`read_parameters` and to allow future post-solve sets.
    solve_name : str, optional
        The active solve identifier; injected as the leading level on
        sets whose legacy layout was ``(solve, period, …)``.

    Returns
    -------
    SimpleNamespace
        Sets as :class:`pd.Index` / :class:`pd.MultiIndex` /
        :class:`pd.Series` matching the legacy CSV-path signature.
    """
    s = SimpleNamespace()

    # ─── Process / entity sets ────────────────────────────────────────────
    # entity = nodes ∪ processes
    nodes = []
    if (flex_data.nodeBalance is not None
            and flex_data.nodeBalance.height > 0):
        nodes = flex_data.nodeBalance["n"].to_list()

    processes = []
    if (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        processes = (flex_data.process_source_sink.select("p")
                     .unique().to_pandas()["p"].tolist())

    s.entity = pd.Index(nodes + processes, name="entity")
    s.node = pd.Index(nodes, name="node")
    s.process = pd.Index(processes, name="process")

    # entityInvest / entityDivest — entities with a non-empty invest/divest set.
    invest_entities = []
    if (flex_data.ed_invest_set is not None
            and flex_data.ed_invest_set.height > 0):
        invest_entities = (flex_data.ed_invest_set.select("e")
                           .unique().to_pandas()["e"].tolist())
    s.entityInvest = pd.Index(invest_entities, name="entity")

    divest_entities = []
    if (flex_data.ed_divest_set is not None
            and flex_data.ed_divest_set.height > 0):
        divest_entities = (flex_data.ed_divest_set.select("e")
                           .unique().to_pandas()["e"].tolist())
    s.entityDivest = pd.Index(divest_entities, name="entity")

    # process_unit / process_connection — partition of processes.
    unit_set = set()
    if (flex_data.process_unit is not None
            and flex_data.process_unit.height > 0):
        unit_set = set(flex_data.process_unit["p"].to_list())
    s.process_unit = pd.Index(sorted(unit_set), name="process")
    s.process_connection = pd.Index(
        [p for p in processes if p not in unit_set],
        name="process",
    )

    # process_profile — processes with a profile method (any of the
    # profile_upper / lower / fixed sets).
    profile_processes = set()
    for fld in (flex_data.process_profile_upper,
                flex_data.process_profile_lower,
                flex_data.process_profile_fixed):
        if fld is not None and fld.height > 0 and "p" in fld.columns:
            profile_processes.update(fld["p"].to_list())
    s.process_profile = pd.Index(sorted(profile_processes), name="process")

    # process_online / process_online_integer / process_online_linear —
    # processes with unit commitment.
    s.process_online = _index_or_empty(
        flex_data.process_online, dim="p", name="process",
    )
    s.process_online_integer = _index_or_empty(
        flex_data.process_online_integer, dim="p", name="process",
    )
    s.process_online_linear = _index_or_empty(
        flex_data.process_online_linear, dim="p", name="process",
    )

    # process_VRE — processes whose source is via a profile_upper
    # method on the source side.  Flextool emits this set distinctly;
    # in FlexData it's encoded via process_profile_upper rows where
    # source ∉ sink (i.e., upstream profile drives the flow).  We
    # leave this as an approximation: union of profile-upper processes
    # constrained to (process, node) pairs.
    if (flex_data.process_profile_upper is not None
            and flex_data.process_profile_upper.height > 0):
        # process_VRE: (process, node) pairs where the process has a
        # profile_upper on its sink (output) arc.  ``calc_storage_vre``
        # filters via ``process_VRE.isin(process_sink)`` and indexes
        # flow_dt / potentialVREgen_dt columns by (process, sink) — so
        # the node
        # half must be the OUTPUT side (sink), matching the convention
        # used by ``process__node__profile__profile_method`` below.
        cols = flex_data.process_profile_upper.columns
        if "sink" in cols:
            ent_dim = "sink"
        elif "source" in cols:
            ent_dim = "source"
        else:
            ent_dim = cols[1] if len(cols) > 1 else cols[0]
        pdf = (flex_data.process_profile_upper.select("p", ent_dim)
               .unique().to_pandas())
        s.process_VRE = pd.MultiIndex.from_frame(
            pdf, names=["process", "node"],
        )
    else:
        s.process_VRE = empty_multi_index(["process", "node"])

    # process_source_sink: (process, source, sink) tuples.
    s.process_source_sink = _multi_index_or_empty(
        flex_data.process_source_sink, dims=("p", "source", "sink"),
        names=["process", "source", "sink"],
    )

    # process_method_sources_sinks — derived (method/orig_source/orig_sink/
    # always_source/always_sink).  FlexData doesn't carry these explicit
    # fields, so we run a 3-way join.
    #
    # The ``always_*`` form keeps the process as one endpoint for every
    # arc (so each direct-method unit has one source-side row pointing
    # to ``(source → p)`` and one sink-side row pointing to ``(p → sink)``).
    # The downstream slope branch in calc_capacity_flows expects this
    # split — without it, the source-side fuel-flow column is missing and
    # commodity costs are computed against the output-side flow (the
    # e89bf53e regression).
    #
    # Method label: any value in ``s.method_1var_per_way`` (defined below)
    # triggers the slope multiplier in calc_capacity_flows.  Connections
    # are 1var-per-way by default in flextool; indirect units have already
    # been split into ``(p, source, p)`` / ``(p, p, sink)`` rows by
    # ``process_source_sink_canonical`` upstream, so always=orig is correct
    # for those rows.
    if (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        pss_pdf = flex_data.process_source_sink.select(
            "p", "source", "sink",
        ).to_pandas()

        rows: list[tuple[str, str, str, str, str, str]] = []
        for p, src, snk in zip(
            pss_pdf["p"], pss_pdf["source"], pss_pdf["sink"],
        ):
            if src == p or snk == p:
                # Indirect / noConversion form — already split, always=orig.
                rows.append((p, "method_1way_1var_LP", src, snk, src, snk))
            else:
                # Direct-method arc — emit BOTH always-process variants so
                # downstream consumers get a source-side and a sink-side
                # column in r.flow_dt (with slope applied on the source
                # side).  See process_arc_unions write_process_arc_unions
                # at L248-263 for the legacy equivalent.
                rows.append((p, "method_1way_1var_LP", src, snk, src, p))
                rows.append((p, "method_1way_1var_LP", src, snk, p, snk))
        rows = list(dict.fromkeys(rows))
        cols = ["process", "method", "orig_source", "orig_sink",
                "always_source", "always_sink"]
        s.process_method_sources_sinks = pd.MultiIndex.from_frame(
            pd.DataFrame(rows, columns=cols),
            names=cols,
        )
    else:
        s.process_method_sources_sinks = empty_multi_index(
            ["process", "method", "orig_source", "orig_sink",
             "always_source", "always_sink"],
        )

    # process_method — variable-style classification (1var_LP, 2var_LP,
    # …).  Currently approximated as ``method_1way_1var_LP`` for every
    # process that appears in ``process_source_sink``; this is the
    # legacy seed the rest of the reader has always carried.
    if (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        pdf = flex_data.process_source_sink.select("p").unique().to_pandas()
        pdf["method"] = "method_1way_1var_LP"
        s.process_method = pd.MultiIndex.from_frame(
            pdf, names=["process", "method"],
        )
    else:
        s.process_method = empty_multi_index(["process", "method"])

    # process__ct_method — conversion-time classification, orthogonal
    # to ``process_method``.  Values that the downstream calculators
    # check for include ``"min_load_efficiency"``,
    # ``"constant_efficiency"``, ``"min_load"`` etc.  The only field
    # FlexData carries on the conversion-time axis is
    # ``process_min_load_eff`` (the subset of processes with
    # ct_method=min_load_efficiency); we materialise those rows so the
    # membership check ``(p, 'min_load_efficiency') in
    # s.process__ct_method`` in calc_capacity_flows resolves to True
    # for the right processes.  Processes that fall outside that
    # subset simply don't appear in the index — which is correct: no
    # downstream consumer keys on any other ct_method value via this
    # set.
    pmle = getattr(flex_data, "process_min_load_eff", None)
    if pmle is not None and pmle.height > 0 and "p" in pmle.columns:
        pdf = pmle.select("p").unique().to_pandas()
        pdf["method"] = "min_load_efficiency"
        s.process__ct_method = pd.MultiIndex.from_frame(
            pdf, names=["process", "method"],
        )
    else:
        s.process__ct_method = empty_multi_index(["process", "method"])

    # method_1var_per_way / method_nvar — flextool's known-method names.
    s.method_1var_per_way = pd.Index(
        ["method_1way_1var_off", "method_1way_1var_LP",
         "method_1way_1var_MIP", "method_2way_1var_off",
         "method_2way_1var_LP", "method_2way_1var_MIP"],
        name="method",
    )
    s.method_nvar = pd.Index(
        ["method_1way_nvar_off", "method_1way_nvar_LP",
         "method_1way_nvar_MIP", "method_2way_nvar_off",
         "method_2way_nvar_LP", "method_2way_nvar_MIP"],
        name="method",
    )

    # ─── Time-related sets (per-solve) ────────────────────────────────────
    # period — (solve, period).  Derived from dt's distinct d values.
    if flex_data.dt is not None and flex_data.dt.height > 0:
        periods_pl = flex_data.dt.select("d").unique()
        period_list = periods_pl.to_pandas()["d"].tolist()
        s.period = pd.MultiIndex.from_arrays(
            [[solve_name] * len(period_list), period_list],
            names=["solve", "period"],
        )
        s.dt = pd.MultiIndex.from_frame(
            flex_data.dt.with_columns(pl.lit(solve_name).alias("solve"))
                .select("solve", "d", "t").to_pandas(),
            names=["solve", "period", "time"],
        )
    else:
        s.period = empty_multi_index(["solve", "period"])
        s.dt = empty_multi_index(["solve", "period", "time"])

    # d_realized_period / d_realize_invest / dt_realize_dispatch /
    # d_realize_dispatch_or_invest — these are subsets of period / dt.
    # Prefer the explicit ``flex_data.realized_dispatch`` (period, step)
    # frame when supplied — this restricts dispatch-time CSVs to the
    # realized window and excludes foresight rows from rolling solves.
    # Falls back to ``period`` / ``dt`` when no realized_dispatch is
    # available (single-period fixtures where every period is realized).
    if (flex_data.realized_dispatch is not None
            and flex_data.realized_dispatch.height > 0):
        rd_pl = flex_data.realized_dispatch
        s.dt_realize_dispatch = pd.MultiIndex.from_frame(
            rd_pl.with_columns(pl.lit(solve_name).alias("solve"))
                .select("solve", "period", "step").to_pandas(),
            names=["solve", "period", "time"],
        )
        # Preserve the canonical period order from ``flex_data.dt`` —
        # ``realized_dispatch`` uniqueness can reorder periods, which
        # breaks period-indexed CSV row order (e.g. ``unit_capacity__d``).
        rd_periods_unordered = set(
            rd_pl.select("period").unique().to_pandas()["period"].tolist()
        )
        if flex_data.dt is not None and flex_data.dt.height > 0:
            dt_period_order = (flex_data.dt.select("d")
                               .unique(maintain_order=True)
                               .to_pandas()["d"].tolist())
            rd_periods = [p for p in dt_period_order if p in rd_periods_unordered]
        else:
            rd_periods = list(rd_periods_unordered)
        s.d_realized_period = pd.MultiIndex.from_arrays(
            [[solve_name] * len(rd_periods), rd_periods],
            names=["solve", "period"],
        )
    else:
        # Without explicit ``realized_dispatch`` we cannot tell which
        # rows of ``dt`` are realized vs foresight.  Emit empty sets so
        # this solve contributes no realized rows to ``read_sets_multi``
        # unions.  Single-solve modern fixtures always populate
        # ``flex_data.realized_dispatch``; legacy/empty cases will
        # surface as empty output which is still cleaner than leaking
        # the full per-solve ``dt`` (including foresight horizons).
        s.d_realized_period = empty_multi_index(["solve", "period"])
        s.dt_realize_dispatch = empty_multi_index(
            ["solve", "period", "time"],
        )
    s.dt_fix_storage_timesteps = empty_multi_index(
        ["solve", "period", "time"],
    )

    # d_realize_invest — periods where invest *or divest* decisions are
    # realized.  Use the union of ed_invest_set and ed_divest_set "d"
    # columns: scenarios that only divest (e.g. ``coal_retire``) have
    # empty ed_invest_set but non-empty ed_divest_set, and we still need
    # those periods in d_realize_invest so the drop_levels inner-join
    # against ed_divest doesn't collapse to empty (which was leaving
    # ``divested`` as NaN in ``unit_capacity__d.csv``).
    #
    # Intersect with ``realized_dispatch`` periods: ed_invest_set covers
    # CANDIDATE (e, d) pairs which include lookahead periods (e.g.
    # test_a_lot_but_not_multi_year has p2020 realized + p2025 lookahead;
    # ed_invest_set spans both, but only p2020 is a REALIZED invest
    # period).  Without this filter, ``unit_capacity__d`` emitted both
    # p2020 and p2025 rows instead of just p2020.
    invest_periods_list: list = []
    seen: set = set()
    realized_period_set: set = set(
        s.d_realized_period.get_level_values("period").tolist()
    ) if len(s.d_realized_period) > 0 else set()
    for src in (flex_data.ed_invest_set, flex_data.ed_divest_set):
        if (src is not None and src.height > 0
                and "d" in src.columns):
            for d in src.select("d").unique().to_pandas()["d"].tolist():
                if d in seen:
                    continue
                if realized_period_set and d not in realized_period_set:
                    continue
                seen.add(d)
                invest_periods_list.append(d)
    invest_periods = invest_periods_list
    s.d_realize_invest = pd.MultiIndex.from_arrays(
        [[solve_name] * len(invest_periods), invest_periods],
        names=["solve", "period"],
    )
    # d_realize_dispatch_or_invest = d_realized_period ∪ d_realize_invest.
    # Use realized-dispatch periods (not all dt periods) so foresight
    # periods don't leak into invest/period-aggregate outputs.  Reuse
    # the canonically ordered ``d_realized_period`` we just built.
    realized_periods: list = list(
        s.d_realized_period.get_level_values("period")
    ) if len(s.d_realized_period) > 0 else []
    union_periods = list(dict.fromkeys(realized_periods + invest_periods))
    s.d_realize_dispatch_or_invest = pd.MultiIndex.from_arrays(
        [[solve_name] * len(union_periods), union_periods],
        names=["solve", "period"],
    )

    # ed_invest / ed_divest / edd_invest — (entity, period) /
    # (entity, period_invest, period).
    s.ed_invest = _multi_index_with_solve(
        flex_data.ed_invest_set, solve_name=solve_name,
        dims=("e", "d"), names=["entity", "period"],
    )
    s.ed_divest = _multi_index_with_solve(
        flex_data.ed_divest_set, solve_name=solve_name,
        dims=("e", "d"), names=["entity", "period"],
    )

    # edd_invest may have either (e, d_invest, d) or (e, d) dims.
    if (flex_data.edd_invest_set is not None
            and flex_data.edd_invest_set.height > 0):
        cols = flex_data.edd_invest_set.columns
        if len(cols) >= 3:
            pdf = flex_data.edd_invest_set.with_columns(
                pl.lit(solve_name).alias("solve")
            ).select("solve", cols[0], cols[1], cols[2]).to_pandas()
            s.edd_invest = pd.MultiIndex.from_frame(
                pdf, names=["solve", "entity", "period_invest", "period"],
            )
        else:
            s.edd_invest = empty_multi_index(
                ["solve", "entity", "period_invest", "period"]
            )
    else:
        s.edd_invest = empty_multi_index(
            ["solve", "entity", "period_invest", "period"]
        )

    # process__node__profile__profile_method — DataFrame from FlexData
    # process_profile_upper / lower / fixed.
    pieces: list[pd.DataFrame] = []
    for fld, method in (
        (flex_data.process_profile_upper, "upper_limit"),
        (flex_data.process_profile_lower, "lower_limit"),
        (flex_data.process_profile_fixed, "equality"),
    ):
        if fld is not None and fld.height > 0:
            pdf = fld.to_pandas().copy()
            cols = pdf.columns.tolist()
            # canonical: (p, source, sink, f) → (process, node, profile, method)
            if "f" in cols:
                pdf = pdf.rename(columns={"p": "process", "f": "profile"})
                # node = sink (output) by convention
                if "sink" in pdf.columns:
                    pdf["node"] = pdf["sink"]
                elif "source" in pdf.columns:
                    pdf["node"] = pdf["source"]
                else:
                    pdf["node"] = ""
                pdf["profile_method"] = method
                pieces.append(pdf[["process", "node", "profile", "profile_method"]])
    if pieces:
        all_df = pd.concat(pieces, ignore_index=True)
        s.process__node__profile__profile_method = pd.MultiIndex.from_frame(
            all_df,
            names=["process", "node", "profile", "profile_method"],
        )
    else:
        s.process__node__profile__profile_method = empty_multi_index(
            ["process", "node", "profile", "profile_method"]
        )

    # ─── Time helpers ─────────────────────────────────────────────────────
    if (flex_data.dtttdt is not None and flex_data.dtttdt.height > 0):
        cols = flex_data.dtttdt.columns
        # Expected: (d, t, t_previous, t_previous_within_timeset, d_previous, t_previous_within_solve)
        pdf = flex_data.dtttdt.with_columns(
            pl.lit(solve_name).alias("solve")
        ).to_pandas()
        # Map flex names → legacy names
        rename = {"d": "period", "t": "time"}
        pdf = pdf.rename(columns=rename)
        # Order: solve, period, time, t_previous, t_previous_within_timeset,
        #        d_previous, t_previous_within_solve
        legacy_names = ["solve", "period", "time", "t_previous",
                        "t_previous_within_timeset", "d_previous",
                        "t_previous_within_solve"]
        # Only keep columns we have
        use_cols = ["solve", "period", "time"] + [
            c for c in legacy_names[3:] if c in pdf.columns
        ]
        s.dtttdt = pd.MultiIndex.from_frame(
            pdf[use_cols], names=use_cols,
        )
    else:
        s.dtttdt = empty_multi_index(
            ["solve", "period", "time", "t_previous"]
        )

    # dtt — typically (d, t, t_previous). Derive from dtttdt if available.
    # Sort by (solve, period, time, t_previous) so the resulting
    # MultiIndex follows the canonical iteration order — without
    # this, downstream droplevel('t_previous') retains polars'
    # ``unique()`` hash-order and writers like ``unit_ramps`` emit time
    # columns out of order (e.g. coal_ramp_limit).
    if (flex_data.dtttdt is not None and flex_data.dtttdt.height > 0):
        pdf = (flex_data.dtttdt.with_columns(
                  pl.lit(solve_name).alias("solve"),
              )
              .select("solve", "d", "t", "t_previous")
              .unique()
              .sort(["solve", "d", "t", "t_previous"])
              .to_pandas())
        s.dtt = pd.MultiIndex.from_frame(
            pdf, names=["solve", "period", "time", "t_previous"],
        )
    else:
        s.dtt = empty_multi_index(
            ["solve", "period", "time", "t_previous"]
        )

    s.period__time_first = empty_multi_index(["solve", "period", "time"])
    s.period_first_of_solve = empty_multi_index(["solve", "period"])
    s.period_in_use = (
        _multi_index_with_solve(
            flex_data.period_in_use_set, solve_name=solve_name,
            dims=("d",), names=["period"],
        )
        if flex_data.period_in_use_set is not None
        else s.period
    )

    # ─── Node-related sets ────────────────────────────────────────────────
    node_types = _node_types(flex_data)
    s.node_state = pd.Index(
        [n for n, t in node_types.items() if t == "storage"], name="node",
    )
    s.node_balance = pd.Index(
        [n for n, t in node_types.items() if t in ("balance", "storage")],
        name="node",
    )
    s.node_balance_period = pd.Index(
        [n for n, t in node_types.items() if t == "balance_within_period"],
        name="node",
    )
    s.node_commodity = pd.Index([], name="node", dtype="object")
    if (flex_data.flow_to_commodity is not None
            and flex_data.flow_to_commodity.height > 0):
        # flow_to_commodity rows imply the sink node is a commodity node
        cols = flex_data.flow_to_commodity.columns
        sink_col = "sink" if "sink" in cols else None
        if sink_col is not None:
            s.node_commodity = pd.Index(
                flex_data.flow_to_commodity.select(sink_col).unique()
                    .to_pandas()[sink_col].tolist(),
                name="node",
            )

    # node_self_discharge — nodes with non-zero self-discharge.
    if (flex_data.p_state_self_discharge is not None
            and flex_data.p_state_self_discharge.frame.height > 0):
        s.node_self_discharge = pd.Index(
            (flex_data.p_state_self_discharge.frame
                .filter(pl.col("value") != 0.0)
                .select("n").unique().to_pandas()["n"].tolist()),
            name="node",
        )
    else:
        s.node_self_discharge = pd.Index([], name="node", dtype="object")

    # node__storage_binding_method — (node, method).
    #
    # Walk every storage_bind_* projection exposed by FlexData.  The four
    # frames below mirror the entries in ``_projection_params.py:1858-1861``
    # and the FlexData declarations in ``input.py:579-582``.  Pre-fix this
    # block walked only ``storage_bind_within_solve`` and
    # ``storage_bind_forward_only`` (audit §5.2), which silently dropped the
    # ``bind_within_timeblock`` and ``bind_within_solve_blended_weights`` methods and
    # broke ``calc_storage_vre.py`` for nodes carrying them.
    #
    # Orthogonal gap (per audit §4): the v54 value list also accepts
    # ``bind_within_period`` and ``bind_intraperiod_blocks``, but
    # ``nodeBalance_eq`` (model.py:615) has no constraint branch for either,
    # and FlexData carries no projection attribute for them.  Those methods
    # are accepted at ingestion but currently produce no constraint term —
    # tracked separately, not in scope here.
    binding_pieces: list[tuple[str, str]] = []
    bind_projections: tuple[tuple[str, str], ...] = (
        ("storage_bind_within_solve", "bind_within_solve"),
        ("storage_bind_forward_only", "bind_forward_only"),
        ("storage_bind_within_timeblock", "bind_within_timeblock"),
        ("storage_bind_within_solve_blended_weights", "bind_within_solve_blended_weights"),
    )
    for attr_name, method_string in bind_projections:
        frame = getattr(flex_data, attr_name, None)
        if frame is not None and frame.height > 0:
            for n in frame["n"].to_list():
                binding_pieces.append((n, method_string))
    if binding_pieces:
        s.node__storage_binding_method = pd.MultiIndex.from_tuples(
            binding_pieces, names=["node", "method"],
        )
    else:
        s.node__storage_binding_method = empty_multi_index(["node", "method"])

    s.node__storage_start_end_method = empty_multi_index(["node", "method"])
    s.node__inflow_method = empty_multi_index(["node", "method"])
    s.node__storage_nested_fix_method = empty_multi_index(["node", "method"])

    # process_source / process_sink — (process, source) / (process, sink).
    #
    # Must contain exactly one canonical (process, node) per input/output arc:
    #   • units  : unit__inputNode nodes for source, unit__outputNode for sink
    #   • connections : node_1 of connection__node__node for source, node_2 for sink
    #
    # The collapsed process_source_sink includes reverse arcs for 2-way
    # connections AND intermediate (p, p) arcs for indirect units, so it
    # cannot be projected directly.  Use the pre-computed canonical fields
    # (populated in input.py from the raw entity tables) when available; fall
    # back to a pss-based derivation that filters out process-as-node arcs.
    if (flex_data.process_source_canonical is not None
            and flex_data.process_source_canonical.height > 0):
        s.process_source = pd.MultiIndex.from_frame(
            flex_data.process_source_canonical.to_pandas(),
            names=["process", "source"],
        )
    elif (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        pss_pl = flex_data.process_source_sink
        src_df = (pss_pl.filter(pl.col("source") != pl.col("p"))
                  .select("p", "source").unique().to_pandas())
        s.process_source = pd.MultiIndex.from_frame(src_df, names=["process", "source"])
    else:
        s.process_source = empty_multi_index(["process", "source"])

    if (flex_data.process_sink_canonical is not None
            and flex_data.process_sink_canonical.height > 0):
        s.process_sink = pd.MultiIndex.from_frame(
            flex_data.process_sink_canonical.to_pandas(),
            names=["process", "sink"],
        )
    elif (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        pss_pl = flex_data.process_source_sink
        snk_df = (pss_pl.filter(pl.col("sink") != pl.col("p"))
                  .select("p", "sink").unique().to_pandas())
        s.process_sink = pd.MultiIndex.from_frame(snk_df, names=["process", "sink"])
    else:
        s.process_sink = empty_multi_index(["process", "sink"])

    # process__source__sink__profile__profile_method — DataFrame for VRE etc.
    s.process__source__sink__profile__profile_method = pd.DataFrame(
        columns=["process", "source", "sink", "profile", "profile_method"],
    )

    # ─── Commodity-related sets ───────────────────────────────────────────
    # Recover the four sets that the legacy CSV path obtained from
    # ``solve_data/commodity_node.csv`` / ``commodity_node_co2.csv`` /
    # ``process__commodity__node.csv`` / ``process__commodity__node_co2.csv``
    # by deriving from the FlexData flow_* frames the model already builds.
    # Each flow_* frame is a join of (p, source, sink) arcs onto either
    # ``commodity__node.csv`` (commodity-priced) or a CO2-priced/capped
    # (g, c, n) projection, so the union of source/sink-side commodity
    # nodes across the frames covers exactly the (c, n) pairs the LP
    # objective and downstream cost calculation reference.
    # Regression introduced by e89bf53e (Δ.31 piece 2, in-memory readers).
    s.commodity_node = _commodity_node_from_flex(
        flex_data,
        attrs=(
            ("flow_from_commodity_eff", "source"),
            ("flow_from_commodity_noEff", "source"),
            ("flow_to_commodity", "sink"),
        ),
    )
    s.commodity_node_co2 = _commodity_node_from_flex(
        flex_data,
        attrs=(
            ("flow_from_co2_priced", "source"),
            ("flow_from_co2_priced_noEff", "source"),
            ("flow_from_co2_capped", "source"),
            ("flow_from_co2_capped_noEff", "source"),
        ),
    )
    s.process__commodity__node = _process_commodity_node_from_flex(
        flex_data,
        attrs=(
            ("flow_from_commodity_eff", "source"),
            ("flow_from_commodity_noEff", "source"),
            ("flow_to_commodity", "sink"),
        ),
    )
    s.process__commodity__node_co2 = _process_commodity_node_from_flex(
        flex_data,
        attrs=(
            ("flow_from_co2_priced", "source"),
            ("flow_from_co2_priced_noEff", "source"),
            ("flow_from_co2_capped", "source"),
            ("flow_from_co2_capped_noEff", "source"),
        ),
    )
    s.group_co2_price = pd.Index([], name="group", dtype="object")
    s.group_co2_limit = pd.Index([], name="group", dtype="object")
    if (flex_data.p_co2_price is not None
            and flex_data.p_co2_price.frame.height > 0):
        s.group_co2_price = pd.Index(
            flex_data.p_co2_price.frame.select("g").unique()
                .to_pandas()["g"].tolist(),
            name="group",
        )
    if (flex_data.group_co2_max_period is not None
            and flex_data.group_co2_max_period.height > 0):
        gcol = flex_data.group_co2_max_period.columns[0]
        s.group_co2_limit = pd.Index(
            flex_data.group_co2_max_period.select(gcol).unique()
                .to_pandas()[gcol].tolist(),
            name="group",
        )

    # ─── Group-related sets ───────────────────────────────────────────────
    s.groupInertia = _index_or_empty(flex_data.groupInertia, dim="g", name="group")
    s.groupNonSync = _index_or_empty(flex_data.groupNonSync, dim="g", name="group")
    s.groupCapacityMargin = _index_or_empty(
        flex_data.groupCapacityMargin, dim="g", name="group",
    )

    # Aggregator / dispatch group sets — empty by default; populated
    # when nodeGroupDispatch features are configured.  No FlexData
    # field carries these directly today.
    s.nodeGroupDispatch = pd.Index([], name="group", dtype="object")
    s.nodeGroupIndicators = pd.Index([], name="group", dtype="object")
    s.flowGroupIndicators = pd.Index([], name="group", dtype="object")
    s.nodeGroupDispatch__connection_Not_in_aggregate = empty_multi_index(
        ["group", "connection"]
    )
    s.nodeGroupDispatch__process__unit__to_node_Not_in_aggregate = empty_multi_index(
        ["group", "process", "unit", "node"]
    )
    s.nodeGroupDispatch__process__node__to_unit_Not_in_aggregate = empty_multi_index(
        ["group", "process", "node", "unit"]
    )
    s.nodeGroupDispatch__process__connection__to_node_Not_in_aggregate = empty_multi_index(
        ["group", "process", "connection", "node"]
    )
    s.nodeGroupDispatch__process__node__to_connection_Not_in_aggregate = empty_multi_index(
        ["group", "process", "node", "connection"]
    )
    s.nodeGroupDispatch__processGroup_Unit_to_group = empty_multi_index(
        ["group", "group_aggregate"]
    )
    s.nodeGroupDispatch__processGroup__process__unit__to_node = empty_multi_index(
        ["group", "group_aggregate", "process", "unit", "node"]
    )
    s.nodeGroupDispatch__processGroup_Group_to_unit = empty_multi_index(
        ["group", "group_aggregate"]
    )
    s.nodeGroupDispatch__processGroup__process__node__to_unit = empty_multi_index(
        ["group", "group_aggregate", "process", "node", "unit"]
    )
    s.nodeGroupDispatch__processGroup_Connection = empty_multi_index(
        ["group", "group_aggregate"]
    )
    s.nodeGroupDispatch__processGroup__process__connection__to_node = empty_multi_index(
        ["group", "group_aggregate", "process", "connection", "node"]
    )
    s.nodeGroupDispatch__processGroup__process__node__to_connection = empty_multi_index(
        ["group", "group_aggregate", "process", "node", "connection"]
    )
    s.nodeGroupDispatch__process_fully_inside = empty_multi_index(
        ["group", "process"]
    )

    # group_node — (group, node) from group_node FlexData field.
    if (flex_data.group_node is not None
            and flex_data.group_node.height > 0):
        cols = flex_data.group_node.columns
        s.group_node = pd.MultiIndex.from_frame(
            flex_data.group_node.to_pandas(),
            names=[long_dim(c) for c in cols],
        )
    else:
        s.group_node = empty_multi_index(["group", "node"])
    s.group_process = empty_multi_index(["group", "process"])
    s.group_process_node = empty_multi_index(["group", "process", "node"])

    # upDown — flextool's known up/down set.
    s.upDown = pd.Index(["up", "down"], name="updown")

    # enable_optional_outputs — read from FlexData if available.  Empty set OK.
    s.enable_optional_outputs = set()

    # ─── DC power flow sets ───────────────────────────────────────────────
    s.node_dc_power_flow = _index_or_empty(
        flex_data.node_dc_power_flow, dim="n", name="node",
    )
    s.connection_dc_power_flow = _index_or_empty(
        flex_data.connection_dc_power_flow, dim="p", name="connection",
    )
    if (flex_data.p_connection_susceptance is not None
            and flex_data.p_connection_susceptance.frame.height > 0):
        df = flex_data.p_connection_susceptance.frame.to_pandas()
        s.connection_susceptance = df.set_index("p")["value"].astype(float)
        s.connection_susceptance.index.name = "connection"
    else:
        s.connection_susceptance = pd.Series(dtype=float)

    return s


# ---------------------------------------------------------------------------
# Multi-solve wrapper
# ---------------------------------------------------------------------------


def _multi_index_has_solve(obj) -> bool:
    if isinstance(obj, pd.MultiIndex):
        return "solve" in (obj.names or ())
    return False


def _series_or_df_has_solve(obj) -> bool:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        idx = obj.index
        if isinstance(idx, pd.MultiIndex):
            return "solve" in (idx.names or ())
    return False


def read_sets_multi(
    steps: "list[tuple[str, FlexData, Solution]] | list[tuple[str, FlexData]]",
    solution: "Solution | None" = None,
) -> SimpleNamespace:
    """Multi-solve variant of :func:`read_sets`.

    See :func:`flextool.process_outputs.read_parameters.read_parameters_multi`
    for the rationale.  Per-sub-solve ``(solve, period, …)``-keyed sets
    (``period``, ``dt``, ``d_realized_period``, ``dt_realize_dispatch``,
    ``d_realize_invest``, ``ed_invest``, ``dtttdt``, etc.) are unioned
    across sub-solves; static set attributes (``node``, ``process``,
    ``upDown``, …) are taken from the last step (invariant).
    """
    if not steps:
        raise ValueError("read_sets_multi: steps must be non-empty")

    def _step_solution(s):
        if len(s) >= 3:
            return s[2]
        if solution is None:
            raise ValueError(
                "read_sets_multi: step is a 2-tuple but no fallback "
                "solution was supplied"
            )
        return solution

    per_step = [
        (s[0], read_sets(s[1], _step_solution(s), solve_name=s[0]))
        for s in steps
    ]
    last_ns = per_step[-1][1]
    if len(per_step) == 1:
        return last_ns

    out = SimpleNamespace()
    attr_names = list(vars(last_ns).keys())
    for attr in attr_names:
        pieces = [getattr(ns, attr) for _, ns in per_step]
        first = pieces[0]
        if isinstance(first, pd.MultiIndex):
            if _multi_index_has_solve(first):
                # Concat MultiIndex rows from each sub-solve.  Non-empty
                # pieces only — empty MultiIndex with mismatching dtypes
                # can perturb the result's dtype.
                non_empty = [p for p in pieces if len(p) > 0]
                if not non_empty:
                    setattr(out, attr, first)
                    continue
                # Build per-level Python lists and reassemble (works
                # regardless of dtype quirks at the empty-frame
                # boundary).
                names = non_empty[0].names
                # All pieces share the same names by construction.
                level_lists = [[] for _ in names]
                for p in non_empty:
                    for i in range(len(names)):
                        level_lists[i].extend(list(p.get_level_values(i)))
                setattr(out, attr, pd.MultiIndex.from_arrays(
                    level_lists, names=names,
                ))
            else:
                setattr(out, attr, getattr(last_ns, attr))
        elif _series_or_df_has_solve(first):
            non_empty = [p for p in pieces if len(p) > 0]
            if not non_empty:
                setattr(out, attr, pieces[-1])
                continue
            merged = pd.concat(non_empty, axis=0)
            if hasattr(pieces[-1], "columns") and hasattr(merged, "columns"):
                merged.columns.name = pieces[-1].columns.name
            setattr(out, attr, merged)
        else:
            setattr(out, attr, getattr(last_ns, attr))
    return out
