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


def score_low(
    netload: np.ndarray,
    period_length: int,
    n_base: int,
) -> np.ndarray:
    """Per-period minimum net load (net-load lows).

    ``score_low[d] = min_{h in period d} netload_h`` — the single best (lowest)
    hour in each aligned base period. Net-load lows are surplus/curtailment
    periods (high VRE / low demand); forcing the most-negative periods keeps the
    representative set covering the down-flexibility / spill extremes, not just
    the scarcity extremes.

    Same ``(netload, period_length, n_base)`` signature and reshape convention
    as :func:`score_peak`.

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
    return grid.min(axis=1)


def score_gradient(
    netload: np.ndarray,
    period_length: int,
    n_base: int,
) -> np.ndarray:
    """Per-period worst within-period net-load ramp.

    ``score_gradient[d] = max_{h in period d} |netload_{h+1} - netload_h|`` over
    the consecutive hours *inside* aligned base period ``d`` — the steepest
    within-period first difference. Forcing the worst-ramp periods keeps the
    representative set covering the hardest flexibility/ramping stress, which a
    pure L2 shape hull can miss.

    A period of length 1 has no internal consecutive pair, so there is no
    within-period difference: this returns ``0.0`` for that period (and for
    every period when ``period_length == 1``).

    Same ``(netload, period_length, n_base)`` signature and reshape convention
    as :func:`score_peak`.

    Args:
        netload: Hourly net-load signal.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods (``len(netload) // period_length``
            or fewer; the horizon is truncated to ``n_base * period_length``).

    Returns:
        1-D array of length ``n_base``.
    """
    if period_length <= 1:
        # A single-timestep period has no internal ramp.
        return np.zeros(n_base, dtype=np.float64)
    n_used = n_base * period_length
    grid = np.asarray(netload[:n_used], dtype=np.float64).reshape(n_base, period_length)
    return np.abs(np.diff(grid, axis=1)).max(axis=1)


def _top_n_indices(
    scores: np.ndarray,
    n: int,
    *,
    descending: bool,
) -> list[int]:
    """Return the ``n`` extreme base periods, deterministic and tie-stable.

    Selects the top-``n`` indices of ``scores`` — the largest when
    ``descending`` (peaks / gradients / sustained deficit), the smallest when
    not (net-load lows). Uses a *stable* argsort so equal scores break the tie
    to the LOWER period index (mirrors the intent of the region-scope
    :func:`_region_scope_forced_indices` idiom, with no hash/set-ordering
    dependence). Returns ``[]`` when ``n <= 0`` or the score is flat
    (``max - min < _SCALE_EPS``) — nothing meaningful to force.
    """
    if n <= 0:
        return []
    if float(scores.max() - scores.min()) < _SCALE_EPS:
        return []
    keys = -scores if descending else scores
    order = np.argsort(keys, kind="stable")
    return [int(order[i]) for i in range(min(int(n), order.size))]


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


def compute_shape_season_forced(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    period_length: int,
    n_base: int,
    *,
    n_high: int,
    n_low: int,
    n_grad: int,
    n_season: int,
    force_window: int | None,
    vg_weight: float,
    region_profiles: dict[str, list[str]] | None = None,
    region_demand: dict[str, float] | None = None,
) -> list[int]:
    """Force a controllable count of net-load extreme + seasonal-stress periods.

    An additive enrichment of :func:`compute_forced_indices`: instead of the
    single ``argmax`` per flag, it forces the top-``n`` base periods in each of
    four categories, all measured on ONE system-coincident net-load signal
    (:func:`build_netload_hourly`, with the same optional node-group
    demand-weighting via ``region_profiles`` / ``region_demand``):

    * ``n_high`` — highest per-period peak net load (:func:`score_peak`,
      descending): capacity-adequacy extremes.
    * ``n_low`` — lowest per-period min net load (:func:`score_low`, ASCENDING,
      most-negative first): surplus / curtailment extremes.
    * ``n_grad`` — worst within-period net-load ramp (:func:`score_gradient`,
      descending): flexibility / ramping extremes.
    * ``n_season`` — worst sustained net-load deficit (:func:`score_net` with
      ``force_window``, descending): seasonal-stress periods.

    The returned set is the deduplicated, ascending UNION of the four category
    picks — a period that qualifies in two categories (e.g. both a peak and a
    worst-gradient period) is forced exactly once. This shape matches
    :func:`compute_forced_indices`, so it feeds the same greedy-hull seed. Each
    ``n_*`` defaults to 0; with all four zero the function returns ``[]`` and has
    no effect (the byte-parity contract).

    A category whose count is 0 is skipped; a category whose score is flat
    (``max - min < _SCALE_EPS``) is skipped too — there is nothing meaningful to
    force. Selection is deterministic: :func:`_top_n_indices` uses a stable
    argsort that breaks ties to the lower period index, with no dependence on
    hash / set iteration order.

    Args:
        profiles: VRE availability series.
        inflows: Time-varying node inflow series.
        demand_scalars: Constant node inflows collected as demand levels.
        timestep_keys: Ordered timestep keys defining the horizon.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods.
        n_high: Number of highest-peak-net-load periods to force.
        n_low: Number of lowest-net-load periods to force.
        n_grad: Number of worst-within-period-gradient periods to force.
        n_season: Number of worst-sustained-net-load-deficit periods to force.
        force_window: Sub-window length for the sustained :func:`score_net`
            (``None`` = whole period).
        vg_weight: Convex blend weight on the VG term.
        region_profiles: Optional region -> profile names mapping (demand-weighted
            system term).
        region_demand: Optional region -> demand level ``D_r`` mapping.

    Returns:
        Sorted, deduplicated list of forced base-period indices.
    """
    if not (n_high or n_low or n_grad or n_season):
        return []

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
    forced.update(
        _top_n_indices(
            score_peak(netload, period_length, n_base), n_high, descending=True
        )
    )
    forced.update(
        _top_n_indices(
            score_low(netload, period_length, n_base), n_low, descending=False
        )
    )
    forced.update(
        _top_n_indices(
            score_gradient(netload, period_length, n_base), n_grad, descending=True
        )
    )
    forced.update(
        _top_n_indices(
            score_net(netload, period_length, n_base, force_window),
            n_season,
            descending=True,
        )
    )
    return sorted(forced)


# Fixed priority order in which critical weeks are placed, independent of the
# order tokens appear in ``which``. ``median2`` is last so it takes the
# next-nearest-median slot after the mids are placed.
_CRITICAL_WEEK_ORDER: tuple[str, ...] = (
    "drought",
    "low",
    "median",
    "highmid",
    "lowmid",
    "median2",
)


def _find_critical_weeks(
    netload: np.ndarray,
    period_length: int,
    n_base: int,
    *,
    which: "set[str] | list[str]",
) -> dict[str, tuple[int, int]]:
    """Locate up to six non-overlapping whole-week windows of base periods.

    Reduces the hourly ``netload`` to a per-period mean array (``score_peak``'s
    reshape convention, then ``.mean(axis=1)``), then a uniform sliding mean of
    length ``periods_per_week = 7 * (24 // period_length)`` over that per-period
    array — the sustained weekly net-load level at each possible week start
    (``0 .. n_base - periods_per_week``).

    Let ``mx = max(window_means)``, ``mn = min(window_means)`` and
    ``md = median(window_means)``, all computed once over ALL window means
    before any exclusion. The requested weeks are placed greedily in the FIXED
    priority order ``drought`` → ``low`` → ``median`` → ``highmid`` →
    ``lowmid`` → ``median2`` (independent of the iteration order of ``which``):

    * ``drought`` — the start with the **highest** window mean ``mx`` (the worst
      sustained high-net-load / drought week).
    * ``low`` — the start with the **lowest** window mean ``mn`` (the deepest
      surplus / low-net-load week).
    * ``median`` — the start whose window mean is closest to ``md`` (a
      representative middle week).
    * ``highmid`` — a mix of median and drought: the start whose window mean is
      closest to the target ``(mx + md) / 2``.
    * ``lowmid`` — a mix of median and low: the start whose window mean is
      closest to the target ``(mn + md) / 2``.
    * ``median2`` — a SECOND median week: the start whose window mean is closest
      to ``md`` among those still available. Placed LAST (after the mids) so it
      takes the next-nearest-median slot once the primary ``median`` week has
      been excluded. If ``median`` was not also requested, ``median2`` is simply
      the closest-to-``md`` available window.

    After each pick, every start within ``periods_per_week`` of the chosen start
    is excluded, so the returned windows can never overlap. Ties (equal window
    mean, or equal distance to a target) break to the LOWEST start index.
    A week that cannot be placed without overlapping an already-chosen one is
    skipped.

    Args:
        netload: Hourly net-load signal.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods.
        which: Subset of ``{"drought", "low", "median", "highmid", "lowmid",
            "median2"}`` to place.

    Returns:
        ``{name: (start, start + periods_per_week)}`` for each requested and
        successfully-placed week — half-open ranges of base-period indices.
        Empty dict when ``which`` is empty or the horizon is shorter than one
        week (``n_base < periods_per_week``).
    """
    periods_per_week = 7 * (24 // period_length)
    if n_base < periods_per_week:
        return {}

    n_used = n_base * period_length
    grid = np.asarray(netload[:n_used], dtype=np.float64).reshape(
        n_base, period_length
    )
    per_period = grid.mean(axis=1)

    kernel = np.ones(periods_per_week, dtype=np.float64) / periods_per_week
    window_mean = np.convolve(per_period, kernel, mode="valid")
    n_starts = int(window_mean.size)  # = n_base - periods_per_week + 1
    # Reference levels over ALL window means, fixed before any exclusion.
    all_max = float(window_mean.max())
    all_min = float(window_mean.min())
    all_median = float(np.median(window_mean))

    available = np.ones(n_starts, dtype=bool)
    result: dict[str, tuple[int, int]] = {}
    for name in _CRITICAL_WEEK_ORDER:
        if name not in which:
            continue
        avail_idx = np.nonzero(available)[0]
        if avail_idx.size == 0:
            # No non-overlapping window left for this (lower-priority) week.
            continue
        vals = window_mean[avail_idx]
        if name == "drought":
            local = int(np.argmax(vals))
        elif name == "low":
            local = int(np.argmin(vals))
        elif name == "highmid":
            target = (all_max + all_median) / 2.0
            local = int(np.argmin(np.abs(vals - target)))
        elif name == "lowmid":
            target = (all_min + all_median) / 2.0
            local = int(np.argmin(np.abs(vals - target)))
        else:  # "median" or "median2" — both target the overall median
            local = int(np.argmin(np.abs(vals - all_median)))
        # ``avail_idx`` is ascending and ``argmax``/``argmin`` return the FIRST
        # extremum, so ties resolve to the lowest start index.
        chosen = int(avail_idx[local])
        result[name] = (chosen, chosen + periods_per_week)
        # Exclude every start within ``periods_per_week`` (overlap guard):
        # a start ``s`` overlaps ``chosen`` iff ``|s - chosen| < periods_per_week``.
        lo = max(0, chosen - periods_per_week + 1)
        hi = min(n_starts, chosen + periods_per_week)
        available[lo:hi] = False

    return result


def _turning_points(week_vals: np.ndarray, prom_threshold: float) -> list[int]:
    """Significant net-load turning points within one week (ZigZag / swing filter).

    Walks the per-period series ``week_vals`` tracking a running extreme in the
    current trend direction. When the series reverses away from that running
    extreme by at least ``prom_threshold``, the running extreme is confirmed as
    a pivot (a local peak or trough) and the trend flips. Peaks and troughs
    therefore strictly alternate, and only reversals of at least
    ``prom_threshold`` in magnitude count — small wiggles below the threshold
    are filtered out (they neither confirm a pivot nor flip the trend).

    Endpoints are not forced: the first index is recorded only once the series
    swings away from it by ``prom_threshold`` (which confirms it as a genuine
    leading pivot), and the trailing running extreme is left unconfirmed unless
    a later reversal turns it into a pivot. This is fully deterministic.

    Args:
        week_vals: 1-D per-period net-load values for one week.
        prom_threshold: Minimum reversal magnitude (in net-load units) that
            confirms a turning point.

    Returns:
        Sorted 0-based offsets (within the week) of the confirmed turning
        points. Empty list when the series range is below ``_SCALE_EPS`` (a flat
        week has no turning point).
    """
    v = np.asarray(week_vals, dtype=np.float64)
    n = int(v.size)
    if n == 0 or float(v.max() - v.min()) < _SCALE_EPS:
        return []

    pivots: list[int] = []
    # Running extremes since the last confirmed pivot: the highest point is the
    # candidate peak, the lowest is the candidate trough. In an unknown trend
    # both are live; whichever the series first reverses away from by
    # ``prom_threshold`` is confirmed, which also sets the trend direction.
    max_i = 0
    min_i = 0
    trend = 0  # 0 unknown, +1 up (looking for a peak), -1 down (looking for trough)
    for i in range(1, n):
        if v[i] > v[max_i]:
            max_i = i
        if v[i] < v[min_i]:
            min_i = i
        if trend >= 0 and v[max_i] - v[i] >= prom_threshold:
            pivots.append(max_i)  # confirmed peak
            trend = -1
            min_i = i  # start tracking the next trough from here
            max_i = i
        elif trend <= 0 and v[i] - v[min_i] >= prom_threshold:
            pivots.append(min_i)  # confirmed trough
            trend = 1
            max_i = i  # start tracking the next peak from here
            min_i = i

    return sorted(pivots)


def compute_week_turn_forced(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    period_length: int,
    n_base: int,
    *,
    which: "set[str] | list[str]",
    turn_prominence: float,
    vg_weight: float,
    region_profiles: dict[str, list[str]] | None = None,
    region_demand: dict[str, float] | None = None,
) -> list[int]:
    """Force the net-load turning points of up to six critical weeks.

    Builds one system-coincident net-load signal (:func:`build_netload_hourly`,
    with the same optional node-group demand-weighting via ``region_profiles`` /
    ``region_demand``), locates the requested critical weeks
    (:func:`_find_critical_weeks`), and within each week forces ONLY the net-load
    *turning points* — the base periods where the sustained (per-period-mean)
    net load reverses direction by at least ``turn_prominence`` of that week's
    net-load range (:func:`_turning_points`). The monotonic stretches between
    turning points are left for the greedy hull to fill.

    Each week's threshold is ``turn_prominence * (week_vals.max() -
    week_vals.min())`` so the same fraction adapts to weeks of different
    amplitude. Turning-point offsets are converted to absolute base-period
    indices (``start + offset``) and unioned across weeks.

    Args:
        profiles: VRE availability series.
        inflows: Time-varying node inflow series.
        demand_scalars: Constant node inflows collected as demand levels.
        timestep_keys: Ordered timestep keys defining the horizon.
        period_length: Timesteps per aligned base period.
        n_base: Number of aligned base periods.
        which: Subset of ``{"drought", "low", "median", "highmid", "lowmid",
            "median2"}`` to force.
        turn_prominence: Minimum net-load reversal to count a turning point, as
            a fraction of the week's net-load range.
        vg_weight: Convex blend weight on the VG term.
        region_profiles: Optional region -> profile names mapping (demand-weighted
            system term).
        region_demand: Optional region -> demand level ``D_r`` mapping.

    Returns:
        Sorted, deduplicated list of forced base-period indices. Empty list when
        ``which`` is empty or no requested week can be placed.
    """
    if not which:
        return []

    netload = build_netload_hourly(
        profiles,
        inflows,
        demand_scalars,
        timestep_keys,
        vg_weight=vg_weight,
        region_profiles=region_profiles,
        region_demand=region_demand,
    )

    weeks = _find_critical_weeks(netload, period_length, n_base, which=which)
    if not weeks:
        return []

    n_used = n_base * period_length
    grid = np.asarray(netload[:n_used], dtype=np.float64).reshape(
        n_base, period_length
    )
    per_period = grid.mean(axis=1)

    forced: set[int] = set()
    for name in _CRITICAL_WEEK_ORDER:
        if name not in weeks:
            continue
        start, end = weeks[name]
        week_vals = per_period[start:end]
        rng = float(week_vals.max() - week_vals.min())
        prom_threshold = turn_prominence * rng
        offsets = _turning_points(week_vals, prom_threshold)
        forced.update(start + off for off in offsets)
        day_of_year = start * period_length / 24
        print(
            f"  Force week-turns: {name} week starts ~day {day_of_year:.1f} "
            f"(base period {start}), contributed {len(offsets)} turning "
            f"point(s)."
        )

    return sorted(forced)
