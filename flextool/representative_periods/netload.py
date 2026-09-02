"""Pure net-load clustering-matrix math (no database access).

Every function here operates on a :class:`~flextool.representative_periods.
netload_inputs.NetloadInputs` value (built by ``netload_inputs.read_netload_inputs``)
plus caller-supplied ``timestep_keys`` — mirroring the pure caller-passes-data
split used by :mod:`flextool.representative_periods.force_include`. This keeps
each step independently unit-testable.

The signal is a real-MW **net load** per aggregation unit ``g``:

    ``net_load[g, h] = Σ_{demand node in g} demand_h
                       − Σ_{VRE unit in g} cap[u] · avail[profile(u)][h]``

with ``demand_h = −inflow_h (time-varying) + |scalar level|`` (positive-demand
convention — a demand node's inflow is negative in FlexTool, so ``−inflow_h`` is
positive demand and a scalar demand level enters as ``+|value|``; this matches
``force_include.build_netload_hourly``). Each aggregation unit's series is then
min-max normalized to ``[0, 1]`` exactly as
``preprocess._build_clustering_matrix`` does per feature.

The iteration-0 VRE capacities come from a **demand-match** default: size the
investable VRE of each aggregation unit so its energy plus the existing VRE
energy just covers the unit's demand energy (pure energy balance, curtailment
ignored).
"""

from __future__ import annotations

import numpy as np

from flextool.representative_periods.netload_inputs import NetloadInputs

# Series / availability whose scale is below this are treated as carrying no
# usable capacity-per-energy information (avoids dividing by a near-zero mean
# availability).
_EPS = 1e-12


def _series_lookup(series: list[tuple[str, float]]) -> dict[str, float]:
    """A ``{timestep_key: value}`` lookup from a ``[(key, value), ...]`` list."""
    return {k: float(v) for k, v in series}


def _covers(lookup: dict[str, float], timestep_keys: list[str]) -> bool:
    """Whether ``lookup`` has a value at *every* key in ``timestep_keys``.

    Matches the partial-coverage policy of ``preprocess._build_clustering_matrix``
    and ``force_include._series_matrix``: a series that does not cover every used
    timestep is skipped by the caller (never zero-filled).
    """
    return all(k in lookup for k in timestep_keys)


def _duration_array(
    step_durations: dict[str, float], timestep_keys: list[str]
) -> np.ndarray:
    """Duration per key over ``timestep_keys`` (missing key → duration 1.0)."""
    return np.array(
        [step_durations.get(k, 1.0) for k in timestep_keys], dtype=np.float64
    )


def demand_match_default_caps(
    inputs: NetloadInputs,
    timestep_keys: list[str],
    vre_penetration: float = 1.0,
) -> dict[str, float]:
    """Iteration-0 demand-match capacities for every investable VRE unit.

    For each aggregation unit ``g`` (``inputs.units_by_group``):

    * ``E_demand = Σ_{node in g}( |scalar| + Σ_h |min(inflow_h, 0)| · dur_h )`` —
      the scalar demand level (magnitude) plus the DURATION-WEIGHTED energy of the
      time-varying *demand* (negative-inflow) part. A time-varying inflow series
      that does not cover every key in ``timestep_keys`` is SKIPPED with a warning
      (never zero-filled), matching ``preprocess._build_clustering_matrix``.
    * ``E_existing_VRE = Σ_{VRE unit in g} existing_cap · Σ_h avail_h · dur_h``
      over ALL of ``g``'s VRE units (investable and not) — the true
      duration-weighted profiled energy, correct for UNEQUAL step durations. A VRE
      unit whose availability profile does not cover every used timestep is
      SKIPPED with a warning (its contribution is not subtracted).
    * ``target = max(0, vre_penetration · E_demand − E_existing_VRE)`` — the
      (scaled) demand energy the existing VRE fleet does not already cover.
      ``vre_penetration`` (default ``1.0`` → 100% energy match) scales the demand
      target: e.g. ``0.5`` sizes the investable fleet to a half-energy VRE share.
    * ``target`` is split as EQUAL ENERGY SHARES across ``g``'s ``k`` investable
      VRE units that have a usable profile energy
      (``w_u = Σ_h avail_h · dur_h ≥ eps``). Each such unit's invested capacity is
      ``(target/k) / w_u``, i.e. the capacity whose profiled energy
      ``cap · Σ_h avail_h · dur_h`` equals its ``target/k`` share EXACTLY.

    Contract of the returned dict (documented, and pinned by the capacity-contract
    test):

    * Keys are **investable VRE units only**; existing-only (non-investable)
      units are absent (their capacity is their existing cap, applied by
      :func:`build_group_capacities`).
    * Each value is the unit's **TOTAL** iteration-0 capacity =
      ``existing_cap + invested_share``. Because ``target`` already subtracts the
      existing VRE energy, adding it back per unit keeps the group's total VRE
      energy equal to ``vre_penetration · E_demand`` (when ``target > 0`` and
      every investable unit is usable) — a clean, energy-consistent total the
      net-load builder can multiply by availability directly. The invested VRE
      energy ``Σ_u invested_u · w_u`` equals ``target`` to machine precision, for
      equal AND unequal step durations alike.
    * An investable unit with ``w_u < eps`` (no usable profile), or a group whose
      ``target`` is 0 or has no usable investable unit, gets its ``existing_cap``
      (no useful invest possible) — never a divide-by-near-zero blow-up.

    Curtailment is ignored (pure energy balance).
    """
    dur = _duration_array(inputs.step_durations, timestep_keys)
    caps: dict[str, float] = {}

    for group in sorted(inputs.units_by_group):
        node_set = set(inputs.units_by_group[group])

        # Demand energy of the group (duration-weighted).
        e_demand = 0.0
        for node in sorted(node_set):
            e_demand += abs(inputs.demand_scalar.get(node, 0.0))
            ts = inputs.demand_ts.get(node)
            if not ts:
                continue
            lookup = _series_lookup(ts)
            if not _covers(lookup, timestep_keys):
                print(
                    f"  Net-load: skipping inflow of node '{node}' "
                    f"(demand-match): does not cover all used timesteps."
                )
                continue
            inflow = np.array(
                [lookup[k] for k in timestep_keys], dtype=np.float64
            )
            demand_part = np.where(inflow < 0.0, -inflow, 0.0)
            e_demand += float(np.dot(demand_part, dur))

        # VRE units of the group, with their duration-weighted profile energy.
        vre_units = sorted(
            u for u, vu in inputs.vre.items() if vu.node in node_set
        )
        weight: dict[str, float] = {}
        e_existing = 0.0
        for u in vre_units:
            vu = inputs.vre[u]
            lookup = _series_lookup(inputs.profiles.get(vu.profile, []))
            if not _covers(lookup, timestep_keys):
                print(
                    f"  Net-load: skipping VRE unit '{u}' (demand-match): "
                    f"profile '{vu.profile}' does not cover all used timesteps."
                )
                weight[u] = 0.0
                continue
            avail = np.array(
                [lookup[k] for k in timestep_keys], dtype=np.float64
            )
            w_u = float(np.dot(avail, dur))
            weight[u] = w_u
            e_existing += vu.existing_cap * w_u

        target = max(0.0, vre_penetration * e_demand - e_existing)

        # Investable units with usable profile energy share the target energy.
        usable = [
            u
            for u in vre_units
            if inputs.vre[u].investable and weight[u] >= _EPS
        ]
        k = len(usable)

        for u in vre_units:
            vu = inputs.vre[u]
            if not vu.investable:
                continue  # existing-only units are not in this dict.
            w_u = weight[u]
            if k == 0 or target <= 0.0 or w_u < _EPS:
                # No useful invest possible: total cap is just the existing cap.
                caps[u] = vu.existing_cap
            else:
                invested = (target / k) / w_u
                caps[u] = vu.existing_cap + invested

    return {u: caps[u] for u in sorted(caps)}


def build_group_capacities(
    inputs: NetloadInputs,
    default_caps: dict[str, float],
    solved_caps: dict[str, float] | None,
) -> dict[str, float]:
    """Resolve the per-VRE-unit capacity used to build the net-load signal.

    Contract (existing vs default vs solved), one rule per VRE unit:

    * **Non-investable (existing-only)** unit → always its ``existing_cap``. Its
      capacity is fixed data; neither the demand-match default nor a solve can
      change it.
    * **Investable** unit → the *total* capacity for this iteration:
      ``solved_caps[unit]`` when ``solved_caps`` is provided and contains the
      unit (a later iteration feeding back a solve's realized capacity), else
      ``default_caps[unit]`` (the iteration-0 demand-match total from
      :func:`demand_match_default_caps`). Both sources are totals, so no existing
      capacity is added here. A unit absent from the chosen source falls back to
      its ``existing_cap`` (defensive; the demand-match default always emits
      every investable unit).

    Returns a capacity for every VRE unit in ``inputs.vre`` (sorted).
    """
    caps: dict[str, float] = {}
    for u in sorted(inputs.vre):
        vu = inputs.vre[u]
        if not vu.investable:
            caps[u] = vu.existing_cap
            continue
        if solved_caps is not None and u in solved_caps:
            caps[u] = float(solved_caps[u])
        elif u in default_caps:
            caps[u] = float(default_caps[u])
        else:
            caps[u] = vu.existing_cap
    return caps


def build_netload_matrix(
    inputs: NetloadInputs,
    caps: dict[str, float],
    timestep_keys: list[str],
    period_length: int,
) -> tuple[np.ndarray, int, list[str]]:
    """Build the net-load clustering matrix ``C`` from per-unit capacities.

    For each aggregation unit ``g`` (sorted): compute the hourly net load
    ``demand_h − Σ VRE cap·avail_h``, min-max normalize the series to ``[0, 1]``
    (same convention as ``preprocess._build_clustering_matrix``; a constant /
    all-zero series normalizes to all zeros and is KEPT — it simply carries no
    information), reshape to ``(n_base_periods, period_length)``, and stack.

    Period geometry copies ``preprocess._build_clustering_matrix``: drop the
    tail so ``n_base_periods = len(timestep_keys) // period_length`` whole
    periods remain (a positive drop prints a warning; ``n_base_periods == 0``
    raises).

    Args:
        inputs: The net-load inputs (aggregation units, demand, VRE, profiles).
        caps: Per-VRE-unit capacity (from :func:`build_group_capacities`); a unit
            absent from the map is treated as 0 capacity.
        timestep_keys: Ordered timestep keys defining the horizon.
        period_length: Timesteps per aligned base period.

    Returns:
        ``(C, n_base_periods, agg_unit_names)`` where ``C`` has shape
        ``(n_agg · period_length, n_base_periods)`` and ``agg_unit_names`` is the
        sorted list of aggregation-unit names (block order along ``C``'s rows).

    Raises:
        ValueError: If ``n_base_periods == 0``, or if there are no aggregation
            units at all (mirrors ``_build_clustering_matrix``'s "no valid time
            series" guard).
    """
    n_total = len(timestep_keys)
    n_base_periods = n_total // period_length
    n_dropped = n_total - n_base_periods * period_length

    if n_base_periods == 0:
        raise ValueError(
            f"Timeline has {n_total} timesteps but period_length is "
            f"{period_length}. Need at least {period_length} timesteps."
        )
    if n_dropped > 0:
        print(f"Warning: Dropping {n_dropped} timesteps from end of timeline")

    n_used = n_base_periods * period_length
    used_keys = timestep_keys[:n_used]

    agg_names = sorted(inputs.units_by_group)
    if not agg_names:
        raise ValueError("No aggregation units found for net-load clustering.")

    # Pre-build availability arrays (over the used keys) for every profile
    # referenced by a VRE unit. A profile that does not cover all used timesteps
    # is cached as ``None`` (skip policy of ``preprocess._build_clustering_matrix``
    # / ``force_include._series_matrix`` — never zero-filled).
    avail_cache: dict[str, np.ndarray | None] = {}

    def _avail(profile: str) -> np.ndarray | None:
        if profile not in avail_cache:
            lookup = _series_lookup(inputs.profiles.get(profile, []))
            if _covers(lookup, used_keys):
                avail_cache[profile] = np.array(
                    [lookup[k] for k in used_keys], dtype=np.float64
                )
            else:
                avail_cache[profile] = None
        return avail_cache[profile]

    feature_blocks: list[np.ndarray] = []
    for group in agg_names:
        node_set = set(inputs.units_by_group[group])

        # Demand term: Σ nodes ( −inflow_h(time-varying) + |scalar| ).
        demand = np.zeros(n_used, dtype=np.float64)
        has_demand = False
        for node in sorted(node_set):
            scalar = inputs.demand_scalar.get(node, 0.0)
            if scalar:
                demand += abs(scalar)
                has_demand = True
            ts = inputs.demand_ts.get(node)
            if ts:
                lookup = _series_lookup(ts)
                if not _covers(lookup, used_keys):
                    print(
                        f"  Net-load: skipping inflow of node '{node}': "
                        f"does not cover all used timesteps."
                    )
                    continue
                demand -= np.array(
                    [lookup[k] for k in used_keys], dtype=np.float64
                )
                has_demand = True

        # VRE term: Σ VRE units cap · avail_h.
        vre_supply = np.zeros(n_used, dtype=np.float64)
        has_vre = False
        for u in sorted(u for u, vu in inputs.vre.items() if vu.node in node_set):
            has_vre = True
            avail = _avail(inputs.vre[u].profile)
            if avail is None:
                print(
                    f"  Net-load: skipping VRE unit '{u}': profile "
                    f"'{inputs.vre[u].profile}' does not cover all used timesteps."
                )
                continue
            cap = float(caps.get(u, 0.0))
            if cap:
                vre_supply += cap * avail

        # Co-location check: a group with VRE capacity but NO demand (neither
        # time-varying inflow nor a scalar level) yields a pure-negative "net
        # load" with nothing to net against — usually the region's load lives on
        # a different node/group. Warn and suggest a co-locating region-group.
        # (Demand-but-no-VRE, a pure load region, is normal and NOT warned.)
        if has_vre and not has_demand:
            print(
                f"  Net-load: aggregation unit '{group}' has VRE capacity but "
                f"no demand (time-varying or scalar) — its net load is pure "
                f"negative VRE. Define a region-group "
                f"(use_for_representative_periods) to co-locate demand and VRE."
            )

        series = demand - vre_supply

        # Min-max normalize to [0, 1]; a constant series → all zeros (kept).
        s_min = series.min()
        s_max = series.max()
        if s_max > s_min:
            series = (series - s_min) / (s_max - s_min)
        else:
            series = np.zeros_like(series)

        feature_blocks.append(series.reshape(n_base_periods, period_length))

    feature_matrix = np.hstack(feature_blocks)  # (n_base, n_agg * PL)
    C = feature_matrix.T  # (n_agg * PL, n_base)
    return C, n_base_periods, agg_names
