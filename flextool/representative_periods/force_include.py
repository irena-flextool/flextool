"""Net-load force-include scoring for representative-period selection.

Pure numpy scoring functions that identify adequacy-critical *base periods*
to force into the representative set, alongside the hull picks. No database
access — every input is passed in by the caller (``preprocess.py``).

Background (see ``specs/repperiod_forceinclude_design.md`` §2 and
``specs/rp_force_include_build_decisions.md``). The greedy convex-hull
clustering optimises a whole-year L2 approximation of *shape* and is blind to
adequacy: a sustained low-VRE / high-demand trough can have an unremarkable
within-period shape and never get selected, so the investment solve never
sees the stress week as a hard balance constraint. Force-include injects the
worst base period(s) under an explicit system-coincident net-load signal.

Scope (per the build-decisions file, which overrides the design where they
disagree):

* **Single ``vg_weight`` knob** — the convex blend between the VG-shortfall
  term and the inflow-demand term. Inflow gets the remainder
  ``1 - vg_weight``. There is no per-category weight and no storage/non-storage
  node split.
* **System scope only** — no ``region_of()`` / name parsing / per-region code.
  The signal is a single system-coincident aggregate.

FlexTool sign convention (the gotcha): **demand is NEGATIVE inflow**, supply is
positive inflow. A demand node's ``inflow`` value is negative; its demand
magnitude is ``|value|``. So ``-inflow_h`` summed over nodes is the system net
demand at hour ``h``, and a scalar (constant) demand node contributes
``+|value|`` as a fixed demand level.
"""

from __future__ import annotations

import numpy as np

# Series whose scale (mean absolute value) is below this are treated as
# carrying no information and normalise to all-zeros, rather than dividing by
# a near-zero denominator.
_SCALE_EPS = 1e-12


def _series_matrix(
    series: dict[str, list[tuple[str, float]]],
    timestep_keys: list[str],
) -> np.ndarray | None:
    """Stack fully-covering time series into a ``(n_series, n_hours)`` matrix.

    Each entry of ``series`` maps a name to ``[(timestep_key, value), ...]``.
    A series is included only if it has a value at *every* key in
    ``timestep_keys`` (matching ``preprocess._build_clustering_matrix``, which
    skips partial-coverage series). Returns ``None`` when no series qualifies.
    """
    rows: list[list[float]] = []
    for data in series.values():
        lookup = dict(data)
        if all(k in lookup for k in timestep_keys):
            rows.append([float(lookup[k]) for k in timestep_keys])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _normalize(term: np.ndarray) -> np.ndarray:
    """Scale a signal by its own mean-absolute value.

    Dividing each of the two net-load terms by its mean-absolute magnitude puts
    them on a comparable, dimensionless scale before the convex blend, so
    neither dominates purely by physical units (a VG shortfall lives in [0, 1],
    a demand level can be in the hundreds).

    Mean-absolute (rather than max) is chosen because it is robust to a single
    outlier hour — one freak spike would otherwise shrink the rest of the
    signal toward zero and flatten period-to-period differences.

    Key property this preserves: a **constant** term (e.g. the inflow term for
    a scalar-only system) normalises to a constant ``1.0`` everywhere, so it
    shifts ``netload`` uniformly and cannot change the ``argmax`` over periods.
    That is what makes the scalar-inflow result invariant to ``vg_weight``.
    A near-zero-scale term (no information) normalises to all zeros.
    """
    scale = float(np.mean(np.abs(term)))
    if scale < _SCALE_EPS:
        return np.zeros_like(term)
    return term / scale


def _weighted_vg_term(
    profiles: dict[str, list[tuple[str, float]]],
    timestep_keys: list[str],
    region_profiles: dict[str, list[str]],
    region_demand: dict[str, float],
    n_hours: int,
) -> np.ndarray | None:
    """Node-group demand-weighted VG-shortfall term (design §2.0).

    Implements ``vg_term_h = Σ_r D_r · mean_{p ∈ region r}(1 - avail_{p,h})`` —
    the mean shortfall *within* each region, then a ``D_r``-weighted *sum*
    across regions (NOT a flat weighted mean over all profiles). This up-weights
    the shortfall of high-demand regions, which is what makes a coincident
    trough in the big importers dominate the argmax.

    Only profiles that (a) map to one of the named regions and (b) fully cover
    the horizon contribute. Returns ``None`` (caller falls back to the
    unweighted term with a warning) when no profile maps to a positive-``D_r``
    region — i.e. the weighting carries no information.
    """
    vg_term = np.zeros(n_hours, dtype=np.float64)
    used_profiles = 0
    total_mapped = 0
    for region, profile_names in region_profiles.items():
        d_r = float(region_demand.get(region, 0.0))
        # Stack the fully-covering profiles of this region.
        region_series = {
            name: profiles[name] for name in profile_names if name in profiles
        }
        total_mapped += len(region_series)
        avail = _series_matrix(region_series, timestep_keys)
        if avail is None:
            continue
        if d_r <= 0.0:
            # A named region that owns profiles but has no scalar demand: warn,
            # it contributes nothing to the weighted sum.
            print(
                f"  Force-include: region '{region}' owns "
                f"{avail.shape[0]} profile(s) but D_r == 0 — it does not "
                f"weight the net-load signal."
            )
            continue
        vg_term += d_r * (1.0 - avail.mean(axis=0))
        used_profiles += avail.shape[0]

    if used_profiles == 0:
        return None

    n_profiles = sum(1 for _ in profiles)
    unmapped = n_profiles - total_mapped
    if unmapped > 0:
        print(
            f"  Force-include: {unmapped} profile(s) not mapped to any named "
            f"region group — excluded from the demand-weighted VG term."
        )
    return vg_term


def build_netload_hourly(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    *,
    vg_weight: float,
    region_profiles: dict[str, list[str]] | None = None,
    region_demand: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the system-coincident net-load signal, one value per timestep.

    Implements the single-knob system aggregate of the net-load formula in
    ``rp_force_include_build_decisions.md``. There are two weighting modes for
    the VG-shortfall term, selected by whether ``region_profiles`` /
    ``region_demand`` are supplied:

    * **Unweighted (default)** — the mean over *all* profile series of
      ``1 - availability_h`` (availability is 0-1). High when VRE is low
      system-wide. This is the byte-parity path.
    * **Node-group demand-weighted** — when both ``region_profiles`` and
      ``region_demand`` are given, ``vg_term_h = Σ_r D_r · mean_{p ∈ r}(1 -
      avail_{p,h})`` (see :func:`_weighted_vg_term`): the within-region mean
      shortfall, ``D_r``-weighted and *summed* into one coincident signal. This
      up-weights high-demand regions so a coincident trough in the big
      importers dominates. Falls back to the unweighted term (with a warning)
      when no profile maps to a positive-``D_r`` region.

    The inflow-demand term is unchanged in both modes: system net demand at
    hour ``h``, ``Σ|demand_scalars| - Σ_nodes inflow_h`` over the time-varying
    inflow nodes. The scalar sum is a constant demand *level* (demand is
    negative inflow, so each scalar contributes ``+|value|``); the time-varying
    part enters with a minus sign so that a more-negative (larger-demand)
    inflow raises the term and a positive (supply) inflow lowers it. It is
    already in demand/energy units, so only the dimensionless VG availability
    term needs ``D_r`` scaling.

    Each term is normalised by its own mean-absolute value (see
    :func:`_normalize`) so they are comparably scaled, then blended:

        ``netload_h = vg_weight * vg_norm_h + (1 - vg_weight) * inflow_norm_h``

    Invariance guarantee: for a system whose inflow is purely scalar (the
    H2_trade case), the inflow term is constant, so ``inflow_norm_h`` is a
    constant ``1.0``. Adding a constant to every hour cannot move the ``argmax``
    over periods, and scaling the VG contribution by a positive ``vg_weight``
    cannot either — so the forced period is invariant to ``vg_weight`` for any
    ``vg_weight > 0``. (A unit test pins this.)

    Args:
        profiles: VRE availability series (name -> [(key, value), ...]), 0-1.
        inflows: Time-varying node inflow series (name -> [(key, value), ...]).
        demand_scalars: Constant node inflows (name -> scalar value), the
            dropped-by-clustering scalars collected as demand levels.
        timestep_keys: Ordered timestep keys defining the horizon.
        vg_weight: Convex blend weight in [0, 1] on the VG term; the inflow
            term gets ``1 - vg_weight``.
        region_profiles: Optional region -> profile names mapping. When given
            together with ``region_demand``, enables demand-weighting.
        region_demand: Optional region -> demand level ``D_r`` mapping.

    Returns:
        1-D array of length ``len(timestep_keys)``.
    """
    n_hours = len(timestep_keys)

    # VG-shortfall term: unweighted mean, or node-group demand-weighted sum.
    vg_term: np.ndarray | None = None
    if region_profiles is not None and region_demand is not None:
        vg_term = _weighted_vg_term(
            profiles, timestep_keys, region_profiles, region_demand, n_hours
        )
        if vg_term is None:
            print(
                "  Force-include: no profile mapped to a positive-demand "
                "region group — falling back to unweighted VG term."
            )
    if vg_term is None:
        avail = _series_matrix(profiles, timestep_keys)
        if avail is not None:
            vg_term = 1.0 - avail.mean(axis=0)
        else:
            vg_term = np.zeros(n_hours, dtype=np.float64)

    # Inflow-demand term: constant scalar demand level minus time-varying inflow.
    scalar_demand = sum(abs(float(v)) for v in demand_scalars.values())
    tv_inflow = _series_matrix(inflows, timestep_keys)
    if tv_inflow is not None:
        inflow_term = scalar_demand - tv_inflow.sum(axis=0)
    else:
        inflow_term = np.full(n_hours, scalar_demand, dtype=np.float64)

    vg_norm = _normalize(vg_term)
    inflow_norm = _normalize(inflow_term)
    return vg_weight * vg_norm + (1.0 - vg_weight) * inflow_norm


def score_peak(
    netload: np.ndarray,
    period_length: int,
    n_base: int,
) -> np.ndarray:
    """Per-period peak net load (§2.1, Flag A).

    ``score_peak[d] = max_{h in period d} netload_h`` — the single worst hour
    in each aligned base period. Capacity-adequacy driver; noisy for
    energy-constrained systems.

    Args:
        netload: Hourly net-load signal.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods (``len(netload) // period_length``
            or fewer; the horizon is truncated to ``n_base * period_length``).

    Returns:
        1-D array of length ``n_base``.
    """
    n_used = n_base * period_length
    grid = np.asarray(netload[:n_used], dtype=np.float64).reshape(n_base, period_length)
    return grid.max(axis=1)


def score_net(
    netload: np.ndarray,
    period_length: int,
    n_base: int,
    window: int | None = None,
) -> np.ndarray:
    """Per-period worst sustained net load (§2.2, Flag B — the energy fix).

    ``score_net[d] = max_{h0} mean_{h in [h0, h0+window)} netload_h`` within
    period ``d`` — the worst sustained sub-window mean. Rewards *sustained*
    troughs rather than a single spiky hour.

    Args:
        netload: Hourly net-load signal.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods.
        window: Sub-window length in timesteps. ``None`` (the default) means the
            whole-period mean (``window == period_length``). Clamped to
            ``[1, period_length]``.

    Returns:
        1-D array of length ``n_base``.
    """
    if window is None:
        window = period_length
    window = max(1, min(int(window), period_length))

    n_used = n_base * period_length
    grid = np.asarray(netload[:n_used], dtype=np.float64).reshape(n_base, period_length)

    if window == period_length:
        return grid.mean(axis=1)

    # Sliding-window means over each period, take the worst (max) start position.
    windows = np.lib.stride_tricks.sliding_window_view(grid, window, axis=1)
    return windows.mean(axis=2).max(axis=1)


def _greedy_region_cover(
    region_candidates: dict[str, list[int]],
    region_scores: dict[str, np.ndarray],
    budget: int,
) -> list[int]:
    """Greedy budgeted max-coverage of regions by forced base periods.

    A generic weighted set-cover: each region contributes a small candidate set
    of base periods (its worst-lull period(s)); a base period *covers* every
    region whose candidate set contains it. Repeatedly pick the not-yet-selected
    period covering the most still-uncovered regions, adding it to the forced
    set, until the ``budget`` cap is spent or every region is covered.

    A single period that is the coincident worst-lull of several regions covers
    all of them at once — that is the dedup the design calls for (it is never
    forced twice, and covering many regions makes it win the greedy pick).

    Determinism: ties on coverage are broken by the greater summed score over
    the newly-covered regions (prefer the deeper lull), then by the lower period
    index. No hidden dependence on dict iteration order.

    Args:
        region_candidates: region -> list of covering base-period indices.
        region_scores: region -> per-period score array (for tie-breaking).
        budget: maximum number of forced periods (cap on the returned set).

    Returns:
        Sorted list of forced base-period indices (length <= ``budget``).
    """
    uncovered: set[str] = set(region_candidates)
    period_regions: dict[int, set[str]] = {}
    for region, cands in region_candidates.items():
        for p in cands:
            period_regions.setdefault(p, set()).add(region)

    selected: list[int] = []
    selected_set: set[int] = set()
    while uncovered and len(selected) < budget:
        best_p: int | None = None
        best_key: tuple[int, float, int] | None = None
        for p, regs in period_regions.items():
            if p in selected_set:
                continue
            newly = regs & uncovered
            if not newly:
                continue
            score_sum = sum(float(region_scores[r][p]) for r in newly)
            # maximize coverage, then summed score, then prefer lower index.
            key = (len(newly), score_sum, -p)
            if best_key is None or key > best_key:
                best_key = key
                best_p = p
        if best_p is None:
            break
        selected.append(best_p)
        selected_set.add(best_p)
        uncovered -= period_regions[best_p]
    return sorted(selected)


def _region_scope_forced_indices(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    period_length: int,
    n_base: int,
    *,
    force_peak_load: bool,
    force_highest_net_load: bool,
    force_window: int | None,
    vg_weight: float,
    region_profiles: dict[str, list[str]],
    region_nodes: dict[str, list[str]],
    budget: int | None,
    region_top_k: int = 1,
) -> list[int] | None:
    """Per-region budgeted force-include (generic, opt-in).

    Scores net load *independently per region* — each region's own profiles
    (VG-shortfall term), time-varying inflows, and scalar demand — via the same
    :func:`build_netload_hourly` used system-wide, then :func:`score_net` (when
    ``force_highest_net_load``) or :func:`score_peak`. Each region's
    ``region_top_k`` worst base periods become its candidate cover set, and
    :func:`_greedy_region_cover` selects the forced periods under ``budget``.

    This is fully generic: no region names, no period indices, no hemisphere or
    period-length assumptions — the picks emerge from each region's own
    net-load data. It works identically for 3, 18, or 35 regions and any
    ``period_length`` / ``n_base``.

    Args:
        profiles/inflows/demand_scalars/timestep_keys: system-wide series;
            filtered to each region by membership below.
        period_length/n_base: base-period geometry.
        force_peak_load/force_highest_net_load: which per-period scorer to use
            (sustained :func:`score_net` preferred when both set).
        force_window: sub-window for :func:`score_net`.
        vg_weight: convex blend weight on the VG term (per region).
        region_profiles: region -> profile names attached to that region.
        region_nodes: region -> member node names (filters inflow / demand).
        budget: max forced periods; ``None`` covers every region (no cap).
        region_top_k: candidate periods per region (default 1 = its worst lull).

    Returns:
        Sorted forced base-period indices, or ``None`` when no region carries a
        usable signal (caller then falls back to the system-coincident path).
    """
    use_net = bool(force_highest_net_load)
    region_scores: dict[str, np.ndarray] = {}
    region_candidates: dict[str, list[int]] = {}
    top_k = max(1, int(region_top_k))

    for region, prof_names in region_profiles.items():
        node_set = set(region_nodes.get(region, []))
        prof_r = {n: profiles[n] for n in prof_names if n in profiles}
        inflow_r = {n: v for n, v in inflows.items() if n in node_set}
        demand_r = {n: v for n, v in demand_scalars.items() if n in node_set}
        # A region with no profile, no inflow and no scalar demand carries no
        # net-load information — skip it (it cannot be scored or covered).
        if not prof_r and not inflow_r and not demand_r:
            continue
        netload_r = build_netload_hourly(
            prof_r,
            inflow_r,
            demand_r,
            timestep_keys,
            vg_weight=vg_weight,
        )
        if use_net:
            scores_r = score_net(netload_r, period_length, n_base, force_window)
        else:
            scores_r = score_peak(netload_r, period_length, n_base)
        # A flat (all-equal) score carries no lull to force — skip so it does
        # not consume budget covering a meaningless argmax.
        if float(scores_r.max() - scores_r.min()) < _SCALE_EPS:
            continue
        region_scores[region] = scores_r
        # Top-k worst periods (descending score); stable for ties via argsort.
        order = np.argsort(scores_r)[::-1]
        region_candidates[region] = [int(order[i]) for i in range(min(top_k, order.size))]

    if not region_candidates:
        return None

    effective_budget = (
        len(region_candidates) if budget is None else max(0, int(budget))
    )
    return _greedy_region_cover(region_candidates, region_scores, effective_budget)


def compute_forced_indices(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    period_length: int,
    n_base: int,
    *,
    force_peak_load: bool,
    force_highest_net_load: bool,
    force_window: int | None,
    vg_weight: float,
    region_profiles: dict[str, list[str]] | None = None,
    region_demand: dict[str, float] | None = None,
    force_region_scope: bool = False,
    force_region_budget: int | None = None,
    region_nodes: dict[str, list[str]] | None = None,
) -> list[int]:
    """Orchestrate net-load scoring and return the forced base-period indices.

    Two modes:

    * **System-coincident (default, ``force_region_scope=False``)** — builds one
      system net-load signal (optionally node-group demand-weighted via
      ``region_profiles`` / ``region_demand``) and takes the ``argmax`` of every
      enabled flag. This is the byte-parity path; its result is unchanged by the
      region-scope parameters when they are left at their defaults.
    * **Per-region budgeted (``force_region_scope=True``)** — scores net load
      independently per region (:func:`_region_scope_forced_indices`) and
      greedily selects forced periods to cover the most regions' worst lulls
      under ``force_region_budget`` (:func:`_greedy_region_cover`). Requires
      ``region_profiles`` + ``region_nodes``; falls back to the system path when
      no region carries a usable signal.

    Returns the deduplicated, sorted list of base-period indices to
    force-include. Empty list when no flag is set.

    Args:
        profiles: VRE availability series.
        inflows: Time-varying node inflow series.
        demand_scalars: Constant node inflows collected as demand levels.
        timestep_keys: Ordered timestep keys defining the horizon.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods.
        force_peak_load: Enable Flag A (peak net load).
        force_highest_net_load: Enable Flag B (sustained net load).
        force_window: Sub-window length for Flag B (``None`` = whole period).
        vg_weight: Convex blend weight on the VG term.
        region_profiles: Optional region -> profile names mapping (demand-weighted
            system term, or per-region VG term under region scope).
        region_demand: Optional region -> demand level ``D_r`` mapping (system
            demand-weighting only).
        force_region_scope: Opt in to the per-region budgeted mode.
        force_region_budget: Max forced periods under region scope (``None`` =
            cover every region).
        region_nodes: Optional region -> member node names mapping; required for
            region scope to filter inflow / demand per region.

    Returns:
        Sorted, deduplicated list of forced base-period indices.
    """
    if not (force_peak_load or force_highest_net_load):
        return []

    if force_region_scope:
        if region_profiles and region_nodes:
            region_forced = _region_scope_forced_indices(
                profiles,
                inflows,
                demand_scalars,
                timestep_keys,
                period_length,
                n_base,
                force_peak_load=force_peak_load,
                force_highest_net_load=force_highest_net_load,
                force_window=force_window,
                vg_weight=vg_weight,
                region_profiles=region_profiles,
                region_nodes=region_nodes,
                budget=force_region_budget,
            )
            if region_forced is not None:
                return region_forced
        # No usable region maps → fall through to the system-coincident path
        # (fail safe, never crash) rather than returning nothing.
        print(
            "  Force-include: region scope requested but no region carried a "
            "usable per-region signal — falling back to the system aggregate."
        )

    netload = build_netload_hourly(
        profiles,
        inflows,
        demand_scalars,
        timestep_keys,
        vg_weight=vg_weight,
        region_profiles=region_profiles,
        region_demand=region_demand,
    )

    forced: set[int] = set()
    if force_peak_load:
        forced.add(int(np.argmax(score_peak(netload, period_length, n_base))))
    if force_highest_net_load:
        forced.add(
            int(np.argmax(score_net(netload, period_length, n_base, force_window)))
        )
    return sorted(forced)
