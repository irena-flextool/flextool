"""Loaders of the start-up input nodes and the CO2 price and cap features."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl
from polar_high import Param

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.input._axis_enums import cast_dim, join_aligned, rename_to_axis
# The FlexData fields of the start-up input (cleared together by reduce_unit_detail_method online).
from flextool.engine.preprocess._reduce_detail import STARTUP_FIELDS
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read


# ---------------------------------------------------------------------------
# Start-up input nodes (unit__inputNode.startup_input_method / startup_energy)



def _load_startup_inputs(inp: Path, sd: Path, dt: pl.DataFrame, pss: "pl.DataFrame | None",
                         nb: "pl.DataFrame | None", nbp: "pl.DataFrame | None", *,
                         source: "InputSource | None" = None,
                         provider: "object | None" = None) -> dict:
    """The start-up input nodes of the solve's units (Utf8 frames; the end-of-load sweep casts them).

    ``process_startup_node`` (p, n): the arcs with ``startup_input_method`` ``startup_and_input`` or
    ``startup_only`` (``source.startup_arcs()``: the readers hide the startup_only arcs from every arc set, so
    their start-up values are read through it) of the units in the solve, with a ``startup_energy`` other than
    0 in a period of the solve.  ``p_startup_energy`` (p, n, d) [MWh per MW started] via
    :func:`params._direct.p_startup_energy_from_source` (constant or period).  ``startup_to_balance``: the pairs
    whose node has an energy balance (nodeBalance or nodeBalancePeriod); ``startup_from_commodity`` (p, n, c):
    the pairs at a commodity node.  The CO2 sets come from :func:`_startup_co2_sets`.  Every field is None
    without a start-up input.  A reader-less reload (``flextool.engine._csv_reload``) reads the dumped
    ``p_startup_energy`` (``solve_data/process_node_startup_energy``, ``output._dump_csvs``) instead."""
    blank = {f: None for f in STARTUP_FIELDS}
    if pss is None or pss.height == 0:
        return blank
    in_solve = pss.select(pl.col("p").cast(pl.Utf8)).unique()
    arcs_fn = getattr(source, "startup_arcs", None)
    if arcs_fn is not None:
        arcs = arcs_fn()
        if arcs is None or arcs.height == 0:
            return blank
        from flextool.engine.preprocess.params._direct import p_startup_energy_from_source
        se = p_startup_energy_from_source(source, period_filter=dt)
        if se is None:
            return blank
        pn = (arcs.select(pl.col("unit").cast(pl.Utf8).alias("p"), pl.col("node").cast(pl.Utf8).alias("n"))
              .join(in_solve, on="p", how="semi").unique().sort("p", "n"))
        se_frame = se.frame.collect() if isinstance(se.frame, pl.LazyFrame) else se.frame
        se_frame = se_frame.select(pl.col("p").cast(pl.Utf8), pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                   pl.col("value").cast(pl.Float64))
    elif source is None and _provider_has(provider, "solve_data/process_node_startup_energy",
                                          sd / "process_node_startup_energy.csv"):
        # A reader-less reload of a dump: the dumped p_startup_energy (output._dump_csvs).
        se_frame = _provider_read(provider, "solve_data/process_node_startup_energy",
                                  sd / "process_node_startup_energy.csv")
        se_frame = se_frame.select(pl.col("process").cast(pl.Utf8).alias("p"),
                                   pl.col("node").cast(pl.Utf8).alias("n"),
                                   pl.col("period").cast(pl.Utf8).alias("d"), pl.col("value").cast(pl.Float64))
        pn = se_frame.select("p", "n").join(in_solve, on="p", how="semi").unique().sort("p", "n")
    else:
        return blank
    se_frame = (se_frame.join(pn, on=["p", "n"], how="semi")
                .join(dt.select(pl.col("d").cast(pl.Utf8)).unique(), on="d", how="semi"))
    pn = pn.join(se_frame.filter(pl.col("value") != 0.0).select("p", "n").unique(), on=["p", "n"], how="semi")
    if pn.height == 0:
        return blank
    out = dict(blank)
    out["process_startup_node"] = pn
    out["p_startup_energy"] = Param(("p", "n", "d"),
                                    se_frame.join(pn, on=["p", "n"], how="semi").sort("p", "n", "d"))
    balance = [f.select(pl.col("n").cast(pl.Utf8)) for f in (nb, nbp) if f is not None and f.height > 0]
    if balance:
        to_balance = pn.join(pl.concat(balance).unique(), on="n", how="semi")
        out["startup_to_balance"] = to_balance if to_balance.height > 0 else None
    if _provider_has(provider, "input/commodity__node", inp / "commodity__node.csv"):
        cn = _provider_read(provider, "input/commodity__node", inp / "commodity__node.csv")
        cn = cn.select(pl.col("commodity").cast(pl.Utf8).alias("c"), pl.col("node").cast(pl.Utf8).alias("n"))
        from_c = pn.join(cn, on="n", how="inner").select("p", "n", "c").sort("p", "n", "c")
        out["startup_from_commodity"] = from_c if from_c.height > 0 else None
    return out


def _startup_co2_sets(inp: Path, sd: Path, startup_pn: "pl.DataFrame | None", *,
                      provider: "object | None" = None) -> dict:
    """``startup_from_co2_priced`` / ``_capped`` / ``_capped_total`` (p, n, c, g), Utf8: the start-up input nodes
    ``startup_pn`` (p, n) at a node of a CO2-priced / co2_max_period / co2_max_total group that carries a
    CO2-bearing commodity c (the gcn sets of :func:`_load_co2_price` / :func:`_load_co2_cap` /
    :func:`_load_co2_cap_total`).  Built apart from those loaders' arc gates, so a group whose only emitter is a
    start-up draw is still priced and capped.  The total-cap set is filtered to the groups with a cap value by
    the caller."""
    out = {"startup_from_co2_priced": None, "startup_from_co2_capped": None,
           "startup_from_co2_capped_total": None}
    if startup_pn is None or startup_pn.height == 0:
        return out
    if not (_provider_has(provider, "solve_data/commodity_node_co2", sd / "commodity_node_co2.csv")
            and _provider_has(provider, "input/group__node", inp / "group__node.csv")):
        return out
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2", sd / "commodity_node_co2.csv")
    cn_co2 = cn_co2.select(pl.col("commodity").cast(pl.Utf8).alias("c"), pl.col("node").cast(pl.Utf8).alias("n"))
    g_node = _provider_read(provider, "input/group__node", inp / "group__node.csv")
    g_node = g_node.select(pl.col("group").cast(pl.Utf8).alias("g"), pl.col("node").cast(pl.Utf8).alias("n"))
    pn = startup_pn.select(pl.col("p").cast(pl.Utf8), pl.col("n").cast(pl.Utf8))
    for field, key in (("startup_from_co2_priced", "group_co2_price"),
                       ("startup_from_co2_capped", "group_co2_max_period"),
                       ("startup_from_co2_capped_total", "group_co2_max_total")):
        if not _provider_has(provider, f"solve_data/{key}", sd / f"{key}.csv"):
            continue
        groups = _provider_read(provider, f"solve_data/{key}", sd / f"{key}.csv")
        if groups.height == 0:
            continue
        groups = groups.select(pl.col("group").cast(pl.Utf8).alias("g")).unique()
        rows = (pn.join(g_node.join(groups, on="g", how="semi"), on="n", how="inner")
                .join(cn_co2, on="n", how="inner")
                .select("p", "n", "c", "g").unique().sort("p", "n", "c", "g"))
        out[field] = rows if rows.height > 0 else None
    return out


# ---------------------------------------------------------------------------
# Optional features (CO2 price, CO2 cap, indirect, user-defined, profiles)

def _load_co2_price(
inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                     pss_noEff: pl.DataFrame | None = None,
                     *,
                     pss: pl.DataFrame | None = None,
                     provider: "object | None" = None,
                     startup_emitters: bool = False):
    """``startup_emitters``: a start-up draw emits in a priced group (:func:`_startup_co2_sets`), so the price
    term stays on without an emitting arc."""
    if pss_eff is None and not startup_emitters:
        return (None, None, None, None, None)
    files = ["group_co2_price.csv", "commodity_node_co2.csv", "pdtGroup.csv"]
    if not all(_provider_has(provider, f"solve_data/{Path(f).stem}", sd / f)
                for f in files):
        return (None, None, None, None, None)
    g_price = _provider_read(provider, "solve_data/group_co2_price",
                              sd / "group_co2_price.csv").pipe(rename_to_axis, {"group": "g"})
    if g_price.height == 0:
        return (None, None, None, None, None)
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
    flow_from_co2_priced = None if pss_eff is None else (pss_eff
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
    # (#19) Removal credit: sink-side flow into the SAME priced node/group —
    # unsplit (eff + noEff together, like flow_to_commodity), no slope: the
    # delivered flow is the physical quantity removed, not a fuel input.
    flow_to_co2_priced = None
    if pss is not None and pss.height > 0:
        flow_to_co2_priced = (pss
            .join(gcn_e, left_on="sink", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if flow_to_co2_priced.height == 0:
            flow_to_co2_priced = None
    # A pure-removal group (no emitting flow at all, only a removal credit)
    # still needs the price term active — don't drop it here.
    if ((flow_from_co2_priced is None or flow_from_co2_priced.height == 0) and flow_from_co2_priced_noEff is None
            and flow_to_co2_priced is None and not startup_emitters):
        return (None, None, None, None, None)
    # ``p_co2_content`` is written by ``apply_direct_params_a`` on every DB path (a reader-less reload seeds it in
    # ``_csv_reload._field_seeds``), so this slot is None.
    co2_content = None
    # Δ.17c Gap C: ``p_co2_price`` produced authoritatively by
    # ``apply_direct_params`` via ``p_co2_price_from_source`` (uses the
    # ``_param_shapes`` resolver — full scalar / 1d_map[period] /
    # 1d_map[time] / 2d_map[period,time] cascade with explicit allow-list).
    # Local CSV slice dropped.
    co2_price = None
    return (flow_from_co2_priced, flow_from_co2_priced_noEff,
            co2_content, co2_price, flow_to_co2_priced)


def _load_co2_cap(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                   dt: pl.DataFrame,
                   pss_noEff: pl.DataFrame | None = None,
                   *,
                   pss: pl.DataFrame | None = None,
                   provider: "object | None" = None,
                   startup_emitters: bool = False):
    """``startup_emitters``: a start-up draw emits in a capped group (:func:`_startup_co2_sets`), so the cap
    rows stay on without an emitting arc."""
    if pss_eff is None and pss_noEff is None and not startup_emitters:
        return (None, None, None, None, None, None)
    p = sd / "group_co2_max_period.csv"
    if not _provider_has(provider, "solve_data/group_co2_max_period", p):
        return (None, None, None, None, None, None)
    g_max = _provider_read(provider, "solve_data/group_co2_max_period", p).pipe(rename_to_axis, {"group":"g"})
    if g_max.height == 0:
        return (None, None, None, None, None, None)
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2",
                              sd / "commodity_node_co2.csv").pipe(rename_to_axis, {"commodity":"c","node":"n"})
    g_node = _provider_read(provider, "input/group__node",
                              inp / "group__node.csv").pipe(rename_to_axis, {"group":"g","node":"n"})
    gcn = (g_max.join(g_node, on="g", how="inner")
                .join(cn_co2, on="n", how="inner")
                .select("g","c","n"))
    if gcn.height == 0:
        return (None, None, None, None, None, None)
    # Cross-axis join setup (Pattern 2): up-cast gcn.n (n axis) to ``e``
    # so the join against pss.source (e axis) is a native Enum match.
    gcn_e = gcn.with_columns(cast_dim(pl.col("n"), None, "e"))
    # (#19) Removal credit: sink-side flow into the SAME capped node/group —
    # unsplit (no slope), mirrors flow_to_co2_priced.  Reduces the cap's LHS.
    flow_to_co2_capped = None
    if pss is not None and pss.height > 0:
        flow_to_co2_capped = (pss.select("p","source","sink")
            .join(gcn_e, left_on="sink", right_on="n", how="inner")
            .select("p","source","sink","c","g"))
        if flow_to_co2_capped.height == 0:
            flow_to_co2_capped = None
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
    # Unlike the price term, a cap with no emitting flow at all (pure
    # removal) never binds — removal only makes the LHS more negative
    # against a positive cap — so there's no need to keep the constraint
    # active for a removal-only group; the original emitting-flow gate is
    # kept as-is.
    if flow_from_co2_capped_eff is None and flow_from_co2_capped_noEff is None and not startup_emitters:
        return (None, None, None, None, None, None)
    # Δ.17c Gap C: ``p_co2_max_period`` produced authoritatively by
    # ``apply_direct_params`` via ``p_co2_max_period_from_source`` (uses
    # the ``_param_shapes`` resolver — scalar / 1d_map[period] cascade
    # with explicit allow-list).  Local pd_group.csv slice dropped.
    co2_max_period = None
    period = dt.select("d").unique()
    return (g_max, flow_from_co2_capped_eff, flow_from_co2_capped_noEff,
            co2_max_period, g_max.join(period, how="cross"), flow_to_co2_capped)


def co2_capped_emitter_arcs(groups: pl.DataFrame, inp: Path, sd: Path, pss_eff: "pl.DataFrame | None",
                            pss_noEff: "pl.DataFrame | None", *, provider: "object | None" = None):
    """``(gcn, flow_eff, flow_noEff)`` of the capped *groups* (column ``g``): gcn (g, c, n) the group's nodes that
    carry a CO2-bearing commodity c, and the eff / noEff arcs (p, source, sink, c, g) drawing from them (the
    emitters of the cap row; None where there are none).  gcn is None when no group has such a node.  Shared by
    :func:`_load_co2_cap_total` and the run's check that a total cap is seen whole (``solve._co2_total_scope``)."""
    cn_co2 = _provider_read(provider, "solve_data/commodity_node_co2",
                              sd / "commodity_node_co2.csv").pipe(
        rename_to_axis, {"commodity": "c", "node": "n"})
    g_node = _provider_read(provider, "input/group__node",
                              inp / "group__node.csv").pipe(
        rename_to_axis, {"group": "g", "node": "n"})
    gcn = (groups.join(g_node, on="g", how="inner")
                 .join(cn_co2, on="n", how="inner")
                 .select("g", "c", "n"))
    if gcn.height == 0:
        return None, None, None
    flows = []
    for arcs in (pss_eff, pss_noEff):
        flow = None
        if arcs is not None and arcs.height > 0:
            # source is the entity-union axis, gcn.n the node axis.
            joined = (join_aligned(arcs.select("p", "source", "sink"), gcn, left_on="source", right_on="n")
                      .select("p", "source", "sink", "c", "g"))
            flow = joined if joined.height > 0 else None
        flows.append(flow)
    return gcn, flows[0], flows[1]


def _load_co2_cap_total(inp: Path, sd: Path, pss_eff: pl.DataFrame | None,
                          pss_noEff: pl.DataFrame | None = None,
                          *,
                          pss: pl.DataFrame | None = None,
                          provider: "object | None" = None,
                          startup_emitters: bool = False):
    """Sibling of :func:`_load_co2_cap` for the multi-period total cap.

    Mirrors the period-cap topology (``group_co2_max_period.csv`` → eff /
    noEff (p, source, sink, c, g) frames) but the gate set comes from
    ``solve_data/group_co2_max_total.csv`` (groups whose ``group__co2_method``
    is ``total`` / ``price_total`` / ``period_total`` — already projected
    by :func:`_leaf_sets.write_co2_method_sets`).  The cap value
    is read from the canonical ``solve_data/pdGroup.csv`` slice
    (``param == 'co2_max_total'``); flextool preprocessing broadcasts the
    Spine scalar across periods, so we collapse to one row per group by
    taking the maximum (all rows carry the same value when authored as
    a scalar).

    Returns a 5-tuple (g_max_total, flow_eff, flow_noEff, p_co2_max_total,
    flow_to_co2_capped_total) — the last is the (#19) removal credit:
    sink-side flow into the same capped group's node(s), unsplit (no slope).
    ``startup_emitters``: a start-up draw emits in a total-capped group (:func:`_startup_co2_sets`), so the cap
    row stays on without an emitting arc.
    """
    if pss_eff is None and pss_noEff is None and not startup_emitters:
        return (None, None, None, None, None)
    p = sd / "group_co2_max_total.csv"
    if not _provider_has(provider, "solve_data/group_co2_max_total", p):
        return (None, None, None, None, None)
    g_max = _provider_read(provider, "solve_data/group_co2_max_total", p).pipe(rename_to_axis, {"group": "g"})
    if g_max.height == 0:
        return (None, None, None, None, None)
    gcn, flow_eff, flow_noEff = co2_capped_emitter_arcs(g_max, inp, sd, pss_eff, pss_noEff, provider=provider)
    if gcn is None:
        return (None, None, None, None, None)
    # (#19) Removal credit — same cap group's node(s), unsplit, no slope
    # (mirrors flow_to_co2_capped).  A pure-removal group never binds the
    # cap (removal only loosens it), so the original emitting-flow gate
    # below is left as-is.
    flow_to_co2_capped_total = None
    if pss is not None and pss.height > 0:
        rem = (join_aligned(pss.select("p", "source", "sink"), gcn, left_on="sink", right_on="n")
            .select("p", "source", "sink", "c", "g"))
        if rem.height > 0:
            flow_to_co2_capped_total = rem
    if flow_eff is None and flow_noEff is None and not startup_emitters:
        return (None, None, None, None, None)
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
        return (None, None, None, None, None)
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
    if flow_to_co2_capped_total is not None:
        flow_to_co2_capped_total = flow_to_co2_capped_total.join(
            cap_groups, on="g", how="inner")
        if flow_to_co2_capped_total.height == 0:
            flow_to_co2_capped_total = None
    if flow_eff is None and flow_noEff is None and not startup_emitters:
        return (None, None, None, None, None)
    return (g_max, flow_eff, flow_noEff, p_cap, flow_to_co2_capped_total)
