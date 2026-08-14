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


def _profile_array(
    profile: str,
    profiles: dict[str, list[tuple[str, float]]],
    timestep_keys: list[str],
) -> np.ndarray:
    """Availability array over ``timestep_keys`` (missing key → 0.0 available)."""
    lookup = _series_lookup(profiles.get(profile, []))
    return np.array(
        [lookup.get(k, 0.0) for k in timestep_keys], dtype=np.float64
    )


def _total_duration(
    step_durations: dict[str, float], timestep_keys: list[str]
) -> float:
    """Σ duration over ``timestep_keys`` (missing key → duration 1.0)."""
    return float(sum(step_durations.get(k, 1.0) for k in timestep_keys))


def demand_match_default_caps(
    inputs: NetloadInputs, timestep_keys: list[str]
) -> dict[str, float]:
    """Iteration-0 demand-match capacities for every investable VRE unit.

    For each aggregation unit ``g`` (``inputs.units_by_group``):

    * ``E_demand = Σ_{node in g}( |scalar| + Σ_h |min(inflow_h, 0)| · dur_h )`` —
      the scalar demand level (magnitude) plus the energy of the time-varying
      *demand* (negative-inflow) part.
    * ``E_existing_VRE = Σ_{VRE unit in g} existing_cap · mean(avail) · Σ_h dur_h``
      over ALL of ``g``'s VRE units (investable and not).
    * ``target = max(0, E_demand − E_existing_VRE)`` — the demand energy the
      existing VRE fleet does not already cover.
    * ``target`` is split as EQUAL ENERGY SHARES across ``g``'s ``k`` investable
      VRE units that have usable mean availability (``mean(avail) ≥ eps``). Each
      such unit's invested capacity is ``(target/k) / (mean(avail) · Σ_h dur_h)``,
      i.e. the capacity whose profiled energy equals its ``target/k`` share.

    Contract of the returned dict (documented, and pinned by the capacity-contract
    test):

    * Keys are **investable VRE units only**; existing-only (non-investable)
      units are absent (their capacity is their existing cap, applied by
      :func:`build_group_capacities`).
    * Each value is the unit's **TOTAL** iteration-0 capacity =
      ``existing_cap + invested_share``. Because ``target`` already subtracts the
      existing VRE energy, adding it back per unit keeps the group's total VRE
      energy equal to ``E_demand`` (when ``target > 0`` and every investable unit
      is usable) — a clean, energy-consistent total the net-load builder can
      multiply by availability directly.
    * An investable unit with ``mean(avail) < eps`` (no usable profile), or a
      group whose ``target`` is 0 or has no usable investable unit, gets its
      ``existing_cap`` (no useful invest possible) — never a divide-by-near-zero
      blow-up.

    Curtailment is ignored (pure energy balance).
    """
    total_dur = _total_duration(inputs.step_durations, timestep_keys)
    caps: dict[str, float] = {}

    for group in sorted(inputs.units_by_group):
        node_set = set(inputs.units_by_group[group])

        # Demand energy of the group.
        e_demand = 0.0
        for node in sorted(node_set):
            e_demand += abs(inputs.demand_scalar.get(node, 0.0))
            ts = inputs.demand_ts.get(node)
            if ts:
                for key, value in ts:
                    v = float(value)
                    if v < 0.0:
                        e_demand += abs(v) * inputs.step_durations.get(key, 1.0)

        # VRE units of the group, with their mean availability.
        vre_units = sorted(
            u for u, vu in inputs.vre.items() if vu.node in node_set
        )
        mean_avail: dict[str, float] = {}
        e_existing = 0.0
        for u in vre_units:
            vu = inputs.vre[u]
            avail = _profile_array(vu.profile, inputs.profiles, timestep_keys)
            ma = float(avail.mean()) if avail.size else 0.0
            mean_avail[u] = ma
            e_existing += vu.existing_cap * ma * total_dur

        target = max(0.0, e_demand - e_existing)

        # Investable units with usable availability share the target energy.
        usable = [
            u
            for u in vre_units
            if inputs.vre[u].investable and mean_avail[u] >= _EPS
        ]
        k = len(usable)

        for u in vre_units:
            vu = inputs.vre[u]
            if not vu.investable:
                continue  # existing-only units are not in this dict.
            ma = mean_avail[u]
            if k == 0 or target <= 0.0 or ma < _EPS:
                # No useful invest possible: total cap is just the existing cap.
                caps[u] = vu.existing_cap
            else:
                invested = (target / k) / (ma * total_dur)
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

    # Pre-build availability arrays for every profile referenced by a VRE unit.
    avail_cache: dict[str, np.ndarray] = {}

    def _avail(profile: str) -> np.ndarray:
        arr = avail_cache.get(profile)
        if arr is None:
            arr = _profile_array(profile, inputs.profiles, used_keys)
            avail_cache[profile] = arr
        return arr

    feature_blocks: list[np.ndarray] = []
    for group in agg_names:
        node_set = set(inputs.units_by_group[group])

        # Demand term: Σ nodes ( −inflow_h(time-varying) + |scalar| ).
        demand = np.zeros(n_used, dtype=np.float64)
        for node in sorted(node_set):
            scalar = inputs.demand_scalar.get(node, 0.0)
            if scalar:
                demand += abs(scalar)
            ts = inputs.demand_ts.get(node)
            if ts:
                lookup = _series_lookup(ts)
                demand -= np.array(
                    [lookup.get(k, 0.0) for k in used_keys], dtype=np.float64
                )

        # VRE term: Σ VRE units cap · avail_h.
        vre_supply = np.zeros(n_used, dtype=np.float64)
        for u in sorted(u for u, vu in inputs.vre.items() if vu.node in node_set):
            cap = float(caps.get(u, 0.0))
            if cap:
                vre_supply += cap * _avail(inputs.vre[u].profile)

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
