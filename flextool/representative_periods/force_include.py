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


def build_netload_hourly(
    profiles: dict[str, list[tuple[str, float]]],
    inflows: dict[str, list[tuple[str, float]]],
    demand_scalars: dict[str, float],
    timestep_keys: list[str],
    *,
    vg_weight: float,
) -> np.ndarray:
    """Build the system-coincident net-load signal, one value per timestep.

    Implements the single-knob system aggregate of the net-load formula in
    ``rp_force_include_build_decisions.md``. Because we ship system scope only
    (no per-region demand weights ``D_r``, which would need forbidden name
    parsing), the two per-region terms collapse to system aggregates:

    * **VG-shortfall term** (time-varying): the mean over *all* profile series
      of ``1 - availability_h`` (availability is 0-1). High when VRE is low
      system-wide.
    * **Inflow-demand term**: system net demand at hour ``h``,
      ``Σ|demand_scalars| - Σ_nodes inflow_h`` over the time-varying inflow
      nodes. The scalar sum is a constant demand *level* (demand is negative
      inflow, so each scalar contributes ``+|value|``); the time-varying part
      enters with a minus sign so that a more-negative (larger-demand) inflow
      raises the term and a positive (supply) inflow lowers it.

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

    Returns:
        1-D array of length ``len(timestep_keys)``.
    """
    n_hours = len(timestep_keys)

    # VG-shortfall term: 1 - mean availability over all profile series.
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
) -> list[int]:
    """Orchestrate net-load scoring and return the forced base-period indices.

    Builds the net-load signal once, computes each requested score, and takes
    the ``argmax`` of every enabled flag. Returns the deduplicated, sorted list
    of base-period indices to force-include. Empty list when no flag is set
    (the default byte-parity path).

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

    Returns:
        Sorted, deduplicated list of forced base-period indices.
    """
    if not (force_peak_load or force_highest_net_load):
        return []

    netload = build_netload_hourly(
        profiles, inflows, demand_scalars, timestep_keys, vg_weight=vg_weight
    )

    forced: set[int] = set()
    if force_peak_load:
        forced.add(int(np.argmax(score_peak(netload, period_length, n_base))))
    if force_highest_net_load:
        forced.add(
            int(np.argmax(score_net(netload, period_length, n_base, force_window)))
        )
    return sorted(forced)
