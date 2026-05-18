"""Reserve subsystem: per-(reserve, upDown, group) reservation balance and
per-(process, reserve, upDown, node) reserve allocation upper bounds.

Self-contained module that mirrors the four reserve constraints in
``flextool.mod`` and the ``vq_reserve`` slack term in the objective:

  * ``reserveBalance_timeseries_eq``  (mod 2408-2431)
  * ``reserveBalance_dynamic_eq``     (mod 2433-2464)
  * ``reserveBalance_up_n_1_eq``      (mod 2466-2496)
  * ``reserveBalance_down_n_1_eq``    (mod 2498-2532)
  * ``reserve_process_upward``        (mod 3451-3472)
  * ``reserve_process_downward``      (mod 3474-3486)
  * ``vq_reserve`` slack penalty in the objective (mod 2100-2101 / objective_audit.md §9.4)

Variables introduced by ``add_variables``/``add_constraints``:

    v_reserve [(p, r, ud, n, d, t)]   in prundt   (>= 0)
    vq_reserve[(r, ud, ng, d, t)]     in reserve__upDown__group × dt   (>= 0, <= 1)

Module API (mirrors what ``_group_slack`` is expected to look like):

    has_feature(d)              -> bool
    load_data(inp, sd, ...)     -> dict of new FlexData fields (caller merges)
    add_variables(m, d)         -> dict of new vars (v_reserve, vq_reserve)
    add_constraints(m, d, vars) -> emits the six reserve constraints
    add_objective_terms(m, d, vars, op_factor) -> Expr (+vq_reserve penalty term)

Pending downstream patches (FOR A4 / merge agent — listed in
``audit/integration_manifest.md``): the .mod also adds
``+ Σ v_reserve * unitsize`` LHS terms to the existing ``maxToSink``,
``ramp_*``, and ``profile_flow_*`` constraints.  Wiring those couplings
requires *editing those constraints in flextool/model.py*, which this
module deliberately does not do.  See the manifest for the explicit list
of LHS terms to add.
"""

from __future__ import annotations

import csv
from pathlib import Path
import polars as pl

from polar_high import Sum, Where, Param
from polar_high.engine import Var

from ._axis_enums import get_global_axis_enums, rename_to_axis, schema_dtype
from ._writer_provider_io import _provider_key


def _provider_get(provider, path: "Path") -> "pl.DataFrame | None":
    """Provider-only fetch.  Returns ``None`` when the Provider is
    missing or doesn't carry *path*'s canonical key.
    """
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)

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


# ---------------------------------------------------------------------------
# Field requirements
# These are the names this module will look up on FlexData (``d``).  The
# caller's loader is expected to populate them via load_data() (then merge
# the returned dict into FlexData).

RESERVE_FIELDS: tuple[str, ...] = (
    # core sets
    "reserve_upDown_group",                       # (r, ud, g) — present iff feature active
    "prundt",                                     # (p, r, ud, n, d, t) — v_reserve domain
    "process_reserve_upDown_node_active",         # (p, r, ud, n)
    "group_node",                                 # (g, n)
    # method partitions of reserve__upDown__group
    "reserve_upDown_group_method_timeseries",     # (r, ud, g, method) — may be empty
    "reserve_upDown_group_method_dynamic",        # (r, ud, g, method)
    "reserve_upDown_group_method_n_1",            # (r, ud, g, method)
    # parameters
    "p_process_reserve_upDown_node_reliability",  # (p, r, ud, n) — already coalesced
    "pdtReserve_upDown_group_reservation",        # (r, ud, g, d, t) — RHS demand
    "p_reserve_upDown_group_penalty_reserve",     # (r, ud, g) — objective penalty
)


def has_feature(d) -> bool:
    """Reserve subsystem is active iff at least one (r, ud, g) tuple exists
    in ``reserve__upDown__group``.  An empty file (column headers only) is
    treated as inactive.

    Mirrors the convention used by every other feature gate in
    ``flextool.model.build_flextool``."""
    rug = getattr(d, "reserve_upDown_group", None)
    return rug is not None and rug.height > 0


def _check(d, fields: tuple[str, ...]) -> None:
    """Same fail-fast contract as ``flextool.model._check``."""
    missing = [f for f in fields if getattr(d, f, None) is None]
    if missing:
        raise ValueError(
            f"_reserve: feature is active but data fields are not populated "
            f"(None): {missing}.  Populate them via _reserve.load_data()."
        )


# ---------------------------------------------------------------------------
# Loader

def load_data(inp: Path | str, sd: Path | str,
              dt: pl.DataFrame | None = None,
              *, provider=None) -> dict:
    """Read all reserve CSVs from ``inp/`` (input) and ``sd/`` (solve_data),
    return a plain dict of ``field_name -> Param-or-DataFrame``.

    The caller is expected to merge this dict into FlexData (e.g. via
    ``setattr(d, k, v)``).  Returns an empty dict if reserves are not
    active in this scenario.

    Parameters
    ----------
    inp : Path
        ``input/`` directory (raw CSVs from preprocessing).
    sd : Path
        ``solve_data/`` directory (preprocessed / Python-emitted CSVs).
    dt : DataFrame, optional
        Time index ``(d, t)``.  Used to clip pdtReserve_upDown_group to
        rows actually present in the dispatch horizon.  If None, the
        loader keeps every row.
    """
    inp = Path(inp)
    sd  = Path(sd)
    out: dict = {}

    # ── Core (r, ud, g) set ─────────────────────────────────────────────
    rug_path = sd / "reserve__upDown__group.csv"
    rug = _provider_get(provider, rug_path)
    if rug is None or rug.height == 0:
        return out
    rug = rug.pipe(rename_to_axis, {"reserve": "r", "upDown": "ud", "group": "g"}) \
              .select("r", "ud", "g")
    out["reserve_upDown_group"] = rug

    # ── Method partitions ──────────────────────────────────────────────
    for method, attr in [
        ("timeseries", "reserve_upDown_group_method_timeseries"),
        ("dynamic",    "reserve_upDown_group_method_dynamic"),
        ("n_1",        "reserve_upDown_group_method_n_1"),
    ]:
        path = sd / f"reserve__upDown__group__method_{method}.csv"
        df = _provider_get(provider, path)
        if df is not None and df.height > 0:
            df = df.pipe(rename_to_axis, {"reserve": "r", "upDown": "ud", "group": "g"}) \
                   .select("r", "ud", "g", "method")
        else:
            df = pl.DataFrame(schema={
                "r": schema_dtype(_enums, "r"),
                "ud": schema_dtype(_enums, "ud"),
                "g": schema_dtype(_enums, "g"),
                "method": pl.Utf8})
        out[attr] = df

    # Δ.12-drop: ``prundt`` produced authoritatively by
    # ``apply_derived_g.prundt_from_source``.  Seed dropped.
    # Δ.12-drop: ``process_reserve_upDown_node_active`` produced
    # authoritatively by ``apply_derived_d``
    # (``process_reserve_upDown_node_active_from_source``).  Seed dropped.

    # ── process_reserve_upDown_node_increase_reserve_ratio (dynamic RHS) ─
    irr_path = sd / "process_reserve_upDown_node_increase_reserve_ratio.csv"
    irr = _provider_get(provider, irr_path)
    if irr is not None:
        if irr.height > 0:
            irr = irr.pipe(rename_to_axis, {"process": "p", "reserve": "r",
                               "upDown": "ud", "node": "n"}) \
                      .select("p", "r", "ud", "n")
        else:
            irr = pl.DataFrame(schema={
                "p": schema_dtype(_enums, "p"),
                "r": schema_dtype(_enums, "r"),
                "ud": schema_dtype(_enums, "ud"),
                "n": schema_dtype(_enums, "n")})
        out["process_reserve_upDown_node_increase_reserve_ratio"] = irr

    # ── process_reserve_upDown_node_large_failure_ratio (n-1 RHS) ──────
    lfr_path = sd / "process_reserve_upDown_node_large_failure_ratio.csv"
    lfr = _provider_get(provider, lfr_path)
    if lfr is not None:
        if lfr.height > 0:
            lfr = lfr.pipe(rename_to_axis, {"process": "p", "reserve": "r",
                               "upDown": "ud", "node": "n"}) \
                      .select("p", "r", "ud", "n")
        else:
            lfr = pl.DataFrame(schema={
                "p": schema_dtype(_enums, "p"),
                "r": schema_dtype(_enums, "r"),
                "ud": schema_dtype(_enums, "ud"),
                "n": schema_dtype(_enums, "n")})
        out["process_reserve_upDown_node_large_failure_ratio"] = lfr

    # ── group_node ────────────────────────────────────────────────────
    # Canonical preprocessing target: solve_data/group_node.csv.
    # Fallback: input/group__node.csv (raw user input).  Mirrors the
    # defensive read in _group_slack.py:389.  Single-source non-defensive
    # reads are vulnerable to the bug class fixed upstream in flextool
    # 042fae23 (preprocessing path typo silently dropped group data).
    gn = None
    gn_paths = (sd / "group_node.csv", inp / "group__node.csv")
    seen_any = False
    for path in gn_paths:
        df = _provider_get(provider, path)
        if df is None:
            continue
        seen_any = True
        if df.height > 0:
            gn = df.pipe(rename_to_axis, {"group": "g", "node": "n"}).select("g", "n").unique()
            break
    if gn is not None:
        out["group_node"] = gn
    elif seen_any:
        out["group_node"] = pl.DataFrame(schema={
            "g": schema_dtype(_enums, "g"),
            "n": schema_dtype(_enums, "n")})

    # Δ.12-drop: ``p_process_reserve_upDown_node_reliability`` produced
    # authoritatively by ``apply_direct_params`` (Δ.4b).  Seed dropped.

    # ── pdtReserve_upDown_group: long-format reservation timeseries ───
    pdtR_path = sd / "pdtReserve_upDown_group.csv"
    pdtR = _provider_get(provider, pdtR_path)
    if pdtR is not None:
        if pdtR.height > 0:
            pdtR = pdtR.pipe(rename_to_axis, {"reserve": "r", "upDown": "ud",
                                 "group": "g", "period": "d", "time": "t"})
            res_only = (pdtR.filter(pl.col("param") == "reservation")
                            .select("r", "ud", "g", "d", "t", "value")
                            .with_columns(value=pl.col("value")
                                          .cast(pl.Float64, strict=False)
                                          .fill_null(0.0)))
            if dt is not None and dt.height > 0:
                res_only = res_only.join(dt, on=["d", "t"], how="inner")
            out["pdtReserve_upDown_group_reservation"] = Param(
                ("r", "ud", "g", "d", "t"), res_only)

    # Δ.12-drop: ``p_reserve_upDown_group_penalty_reserve`` /
    # ``p_process_reserve_upDown_node_max_share`` /
    # ``p_process_reserve_upDown_node_large_failure_ratio_value`` /
    # ``p_process_reserve_upDown_node_increase_reserve_ratio_value``
    # produced authoritatively by ``apply_direct_params`` (Δ.4b).
    # Seeds dropped.

    return out


# ---------------------------------------------------------------------------
# Variables

def add_variables(m, d) -> dict:
    """Declare ``v_reserve`` and ``vq_reserve`` on the problem ``m``.

    Returns the two Var objects keyed by name so the constraint emitter
    (and the merge agent's downstream LHS patches) can reference them
    without re-declaring."""
    if not has_feature(d):
        return {}
    _check(d, ("reserve_upDown_group", "prundt", "dt"))

    # v_reserve domain = prundt (only entries that the preprocessor flagged)
    v_reserve = m.add_var("v_reserve",
                          ("p", "r", "ud", "n", "d", "t"),
                          d.prundt, lower=0.0)

    # vq_reserve: domain = reserve__upDown__group × dt.  Cap at 1 to match
    # the .mod's ``var vq_reserve … <= 1`` declaration.
    rug_dt = d.reserve_upDown_group.join(d.dt, how="cross")
    vq_reserve = m.add_var("vq_reserve",
                           ("r", "ud", "g", "d", "t"),
                           rug_dt, lower=0.0, upper=1.0)

    return {"v_reserve": v_reserve, "vq_reserve": vq_reserve}


# ---------------------------------------------------------------------------
# Constraints

def add_constraints(m, d, vars: dict) -> None:
    """Emit reserve constraints.  ``vars`` is the dict returned by
    ``add_variables`` (keys ``v_reserve``, ``vq_reserve``).  ``m``,
    ``d`` are the ``Problem`` and ``FlexData``.

    Caller-provided dependencies on FlexData:
      * ``v_flow``  — read off ``vars["v_flow"]`` if present (needed for
        the dynamic and n-1 RHS terms).  If ``v_flow`` is missing, those
        constraint families are emitted with a constant-zero RHS, which
        matches the .mod's ``sum {} … = 0`` behaviour when the underlying
        sets are empty.
      * ``p_unitsize`` — read off ``d.p_unitsize`` (Param keyed on ``p``).
      * ``p_slope``    — read off ``d.p_slope`` (Param keyed on (p, d, t)).
    """
    if not has_feature(d):
        return
    _check(d, ("reserve_upDown_group", "prundt", "dt",
               "process_reserve_upDown_node_active", "group_node"))

    v_reserve  = vars["v_reserve"]
    vq_reserve = vars["vq_reserve"]
    p_unitsize = d.p_unitsize
    rug        = d.reserve_upDown_group                # (r, ud, g)
    rug_dt     = rug.join(d.dt, how="cross")           # (r, ud, g, d, t)
    pruna      = d.process_reserve_upDown_node_active  # (p, r, ud, n)
    gn         = d.group_node.pipe(rename_to_axis, {"n": "n"})       # (g, n)

    # ── LHS (shared across all four reserveBalance variants) ────────────
    #
    #   + Σ_{(p,r,ud,n) ∈ pruna : (g,n) ∈ group_node, (r,ud,g) ∈ rug}
    #         v_reserve[p,r,ud,n,d,t] · unitsize[p] · reliability[p,r,ud,n]
    #   + Σ … (1var-per-way + source case, multiplied by pdtProcess_slope)
    #   + vq_reserve · pdtReserve_reservation
    #
    # Without the slope distinction (V1 simplification — see manifest TODO
    # for the eventual 1var-per-way + source split), we collapse both
    # process partitions into a single sum without the slope factor.  This
    # is exact whenever no reserve-providing processes are 1var_per_way
    # (the common case in the test fixtures).  The slope-bearing partition
    # can be added by joining ``pruna`` against ``method_1var_per_way`` and
    # ``process_source`` to split the sum and applying ``d.p_slope`` to
    # the source-side rows only.

    # Restrict pruna to (p, r, ud, n) with (g, n) ∈ group_node and
    # (r, ud, g) ∈ rug (=> bring g into the dim set).
    pruna_g = (pruna.join(gn, on="n", how="inner")     # adds g
                    .join(rug, on=["r", "ud", "g"], how="inner")
                    .select("p", "r", "ud", "n", "g"))

    if (d.p_process_reserve_upDown_node_reliability is not None
            and pruna_g.height > 0):
        rel = d.p_process_reserve_upDown_node_reliability
        # v_reserve has dims (p, r, ud, n, d, t); multiply by unitsize and
        # reliability (both keyed inside (p, r, ud, n)), then Where-join to
        # pruna_g to add the ``g`` dim, and Sum out (p, n) to leave (r, ud, g, d, t).
        lhs_reserve_core = Sum(
            Where(v_reserve * p_unitsize * rel, pruna_g),
            over=("p", "n"),
        )
    else:
        lhs_reserve_core = None

    # vq_reserve · pdtReserve_reservation, dims = (r, ud, g, d, t) already
    if d.pdtReserve_upDown_group_reservation is not None:
        lhs_vq = vq_reserve * d.pdtReserve_upDown_group_reservation
    else:
        lhs_vq = vq_reserve

    # ── reserveBalance_timeseries_eq ─────────────────────────────────────
    method_ts = d.reserve_upDown_group_method_timeseries
    if method_ts is not None and method_ts.height > 0:
        # Restrict the constraint domain to (r, ud, g, d, t) where
        # (r, ud, g) ∈ method_ts.
        ts_rug = method_ts.select("r", "ud", "g").unique()
        ts_dt  = ts_rug.join(d.dt, how="cross")
        rhs = (d.pdtReserve_upDown_group_reservation
               if d.pdtReserve_upDown_group_reservation is not None
               else 0.0)
        lhs_terms = {"vq_term": lhs_vq}
        if lhs_reserve_core is not None:
            lhs_terms["reserve_sum"] = Where(lhs_reserve_core, ts_rug)
        m.add_cstr(
            "reserveBalance_timeseries_eq",
            over      = ts_dt,
            sense     = ">=",
            lhs_terms = lhs_terms,
            rhs_terms = {"demand": rhs} if not isinstance(rhs, float) else {"demand": rhs},
        )

    # ── reserveBalance_dynamic_eq ────────────────────────────────────────
    method_dyn = d.reserve_upDown_group_method_dynamic
    if method_dyn is not None and method_dyn.height > 0:
        dyn_rug = method_dyn.select("r", "ud", "g").unique()
        dyn_dt  = dyn_rug.join(d.dt, how="cross")
        # RHS: Σ v_flow · unitsize · increase_reserve_ratio (over flows
        # touching n in (g, n)) + Σ pdtNodeInflow · ratio / step_duration.
        # V1: emit *only* the LHS + vq slack vs. RHS = 0 if v_flow / the
        # ratio set are absent.  When v_flow is present and the
        # increase_reserve_ratio param is loaded, build the dynamic RHS.
        rhs_terms: dict = {}
        v_flow      = vars.get("v_flow")
        irr_set     = getattr(d, "process_reserve_upDown_node_increase_reserve_ratio", None)
        irr_param   = getattr(d, "p_process_reserve_upDown_node_increase_reserve_ratio_value", None)
        if (v_flow is not None and irr_set is not None and irr_set.height > 0
                and irr_param is not None and d.process_source_sink is not None):
            # join irr_set to group_node to add g, restrict to dyn_rug
            irr_g = (irr_set.join(gn, on="n", how="inner")
                            .join(dyn_rug, on=["r", "ud", "g"], how="inner")
                            .select("p", "r", "ud", "n", "g"))
            if irr_g.height > 0:
                # All flow rows whose (p, n) (either as source or sink) is
                # listed in process_source_sink contribute.  Simplification:
                # use process_source_sink (the union) and let the join
                # over (p, n) pick up both source-side and sink-side rows.
                # The .mod splits noEff/eff for the slope factor; we keep
                # the no-slope variant here and let A4 patch in slope where
                # applicable.
                #
                # rename source/sink so the join landing column is "n"
                pss = d.process_source_sink
                # Sum pieces: source-side (n appears as 'sink') + sink-side
                # (n appears as 'source').  We unify by inner-joining both
                # ways and summing the contributions.
                # NOTE: this is a structural simplification.  The .mod's
                # full RHS additionally splits noEff/eff and applies slope
                # to the eff sink-side rows.  Listed in manifest TODO.
                pieces = []
                # n on the "sink" side: pss has columns (p, source, sink)
                pss_as_sink = pss.pipe(rename_to_axis, {"sink": "n"}).select("p", "source", "n")
                pieces.append(Sum(
                    Where(v_flow * p_unitsize * irr_param,
                          pss_as_sink.join(irr_g, on=["p", "n"], how="inner")),
                    over=("p", "source", "n"),
                ))
                pss_as_source = pss.pipe(rename_to_axis, {"source": "n"}).select("p", "n", "sink")
                pieces.append(Sum(
                    Where(v_flow * p_unitsize * irr_param,
                          pss_as_source.join(irr_g, on=["p", "n"], how="inner")),
                    over=("p", "sink", "n"),
                ))
                rhs_dyn = pieces[0]
                for piece in pieces[1:]:
                    rhs_dyn = rhs_dyn + piece
                rhs_terms["dynamic_flow"] = rhs_dyn

        # demand term: pdtReserve_reservation is the reservation timeseries.
        # The .mod's dynamic_eq RHS does *not* include pdtReserve_reservation
        # directly — the demand is dynamic.  However the LHS keeps the
        # vq_reserve · pdtReserve_reservation term.  When the dynamic RHS
        # is unavailable in V1 we fall back to a 0 RHS, matching the .mod
        # in the empty-set degenerate case.
        if not rhs_terms:
            rhs_terms = {"zero": 0.0}

        lhs_terms = {"vq_term": Where(lhs_vq, dyn_rug)}
        if lhs_reserve_core is not None:
            lhs_terms["reserve_sum"] = Where(lhs_reserve_core, dyn_rug)
        m.add_cstr(
            "reserveBalance_dynamic_eq",
            over      = dyn_dt,
            sense     = ">=",
            lhs_terms = lhs_terms,
            rhs_terms = rhs_terms,
        )

    # ── reserveBalance_up_n_1_eq / reserveBalance_down_n_1_eq ──────────
    # The n-1 family adds an extra index ``p_n_1 in process_large_failure``
    # to the constraint domain — one row per (failing process × group ×
    # d × t).  V1 simplification: emit per-(r, ud, g, d, t) only, with the
    # large-failure RHS summed across all p_n_1 candidates.  The .mod
    # imposes the constraint *for each* p_n_1 separately, which is tighter
    # in pathological cases.  Listed in manifest TODO.
    method_n1 = d.reserve_upDown_group_method_n_1
    if method_n1 is not None and method_n1.height > 0:
        for ud_filter, name in [
            ("up",   "reserveBalance_up_n_1_eq"),
            ("down", "reserveBalance_down_n_1_eq"),
        ]:
            n1_rug = method_n1.filter(pl.col("ud") == ud_filter) \
                              .select("r", "ud", "g").unique()
            if n1_rug.height == 0:
                continue
            n1_dt = n1_rug.join(d.dt, how="cross")
            rhs_terms: dict = {}
            v_flow    = vars.get("v_flow")
            lfr_set   = getattr(d, "process_reserve_upDown_node_large_failure_ratio", None)
            lfr_param = getattr(d, "p_process_reserve_upDown_node_large_failure_ratio_value", None)
            if (v_flow is not None and lfr_set is not None and lfr_set.height > 0
                    and lfr_param is not None and d.process_source_sink is not None):
                lfr_g = (lfr_set.join(gn, on="n", how="inner")
                                 .join(n1_rug, on=["r", "ud", "g"], how="inner")
                                 .select("p", "r", "ud", "n", "g"))
                if lfr_g.height > 0:
                    pss = d.process_source_sink
                    if ud_filter == "up":
                        # n is the failing-process sink (delivering side)
                        pss_idx = pss.pipe(rename_to_axis, {"sink": "n"}).select("p", "source", "n")
                        rhs_terms["n_1_failure"] = Sum(
                            Where(v_flow * p_unitsize * lfr_param,
                                  pss_idx.join(lfr_g, on=["p", "n"], how="inner")),
                            over=("p", "source", "n"),
                        )
                    else:
                        # n is the failing-process source (consuming side)
                        pss_idx = pss.pipe(rename_to_axis, {"source": "n"}).select("p", "n", "sink")
                        rhs_terms["n_1_failure"] = Sum(
                            Where(v_flow * p_unitsize * lfr_param,
                                  pss_idx.join(lfr_g, on=["p", "n"], how="inner")),
                            over=("p", "sink", "n"),
                        )
            if not rhs_terms:
                rhs_terms = {"zero": 0.0}

            lhs_terms = {"vq_term": Where(lhs_vq, n1_rug)}
            if lhs_reserve_core is not None:
                lhs_terms["reserve_sum"] = Where(lhs_reserve_core, n1_rug)
            m.add_cstr(
                name,
                over      = n1_dt,
                sense     = ">=",
                lhs_terms = lhs_terms,
                rhs_terms = rhs_terms,
            )

    # ── reserve_process_upward / reserve_process_downward ──────────────
    # Per-(p, r, ud, n, d, t):
    #   v_reserve · unitsize  ≤  max_share · (online · unitsize) [if online]
    #   v_reserve · unitsize  ≤  max_share · existing-capacity   [otherwise]
    #
    # V1: online-not-aware variant.  RHS uses max_share · ( existing_count +
    # Σ_{d_inv} v_invest_p[d_inv] - Σ_{d_div} v_divest_p[d_div] ) (the
    # invest/divest terms are moved to the LHS as -invest / +divest).
    # The .mod's online variant requires the online_set + v_online from
    # the online module — that wiring is left to A4.
    max_share = getattr(d, "p_process_reserve_upDown_node_max_share", None)
    if max_share is not None and d.prundt.height > 0:
        existing = getattr(d, "p_process_existing_count", None)
        if existing is not None:
            v_invest_p = vars.get("v_invest_p")
            v_divest_p = vars.get("v_divest_p")
            edd_inv_set  = getattr(d, "edd_invest_set", None)
            edd_div_act  = getattr(d, "edd_divest_active", None)
            for ud_filter, name in [
                ("up",   "reserve_process_upward"),
                ("down", "reserve_process_downward"),
            ]:
                idx = d.prundt.filter(pl.col("ud") == ud_filter)
                if idx.height == 0:
                    continue
                # max_share is keyed on (p, r, ud, n).  existing_count
                # keyed on (p, d).  Together these give RHS dims
                # (p, r, ud, n, d) — broadcast-with-Sum across t at row
                # binding.  We emit on idx (which has all six dims).
                rhs_param = max_share * existing
                lhs: dict = {"reserve": v_reserve}
                # Restrict invest-tightening to processes appearing in idx.
                p_in_idx = idx.select("p").unique()
                # invest tightening: -Σ_{d_inv} v_invest_p[d_inv] · max_share
                if (v_invest_p is not None and edd_inv_set is not None
                        and edd_inv_set.height > 0):
                    edd_p = (edd_inv_set.pipe(rename_to_axis, {"e": "p"})
                             .join(p_in_idx, on="p", how="inner"))
                    if edd_p.height > 0:
                        v_inv_at = Var(
                            name=v_invest_p.name + f"__at_{name}",
                            dims=("p", "d_invest"),
                            frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                            lower=v_invest_p.lower, upper=v_invest_p.upper,
                        )
                        inv_sum = Sum(Where(v_inv_at, edd_p), over=("d_invest",))
                        lhs["invest_neg"] = -(inv_sum * max_share)
                # divest tightening: +Σ_{d_div} v_divest_p[d_div] · max_share
                if (v_divest_p is not None and edd_div_act is not None
                        and edd_div_act.height > 0):
                    edd_p_div = edd_div_act.join(p_in_idx, on="p", how="inner")
                    if edd_p_div.height > 0:
                        v_div_at = Var(
                            name=v_divest_p.name + f"__at_{name}",
                            dims=("p", "d_divest"),
                            frame=v_divest_p.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                            lower=v_divest_p.lower, upper=v_divest_p.upper,
                        )
                        div_sum = Sum(Where(v_div_at, edd_p_div), over=("d_divest",))
                        lhs["divest"] = div_sum * max_share
                m.add_cstr(
                    name,
                    over      = idx,
                    sense     = "<=",
                    lhs_terms = lhs,
                    rhs_terms = {"max":     rhs_param},
                )


# ---------------------------------------------------------------------------
# Objective

def add_objective_terms(m, d, vars: dict, op_factor):
    """Return the ``vq_reserve`` slack penalty Expr to be added to the
    objective.  Mirrors objective_audit.md §9.4 / flextool.mod 2100-2101.

      + Σ vq_reserve · pdtReserve_reservation · penalty_reserve · op_factor

    where ``op_factor = step_duration · rp_cost_weight · inflation_op /
    period_share`` (the same factor used elsewhere in the objective).

    Returns ``None`` (zero contribution) if the reserve subsystem is not
    active or if any dependency is missing — same pattern as the rest of
    flexpy's optional objective terms.
    """
    if not has_feature(d):
        return None
    if "vq_reserve" not in vars:
        return None

    vq_reserve = vars["vq_reserve"]
    res_param = d.pdtReserve_upDown_group_reservation
    pen       = d.p_reserve_upDown_group_penalty_reserve
    if res_param is None or pen is None:
        return None

    # ``op_factor`` carries ``pdt_branch_weight`` when stochastics is
    # active (folded in by the model.py caller — see A6 close).  In
    # deterministic single-branch runs ``pdt_branch_weight`` is None and
    # ``op_factor`` is the four-Param product the .mod uses.
    return Sum(vq_reserve * res_param * pen * op_factor)
