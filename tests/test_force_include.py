"""Unit tests for net-load force-include scoring.

Synthetic, no solver, no DB — mirrors the style of
``tests/test_representative_periods.py`` ``TestGreedyClustering``. Covers the
score metrics (planted trough / peak, sub-window), the ``vg_weight`` invariance
for scalar-only inflow, and dedup/sort of the orchestrator, per design §8.2 and
``specs/rp_force_include_build_decisions.md``.
"""

from __future__ import annotations

import numpy as np

from flextool.representative_periods.force_include import (
    build_netload_hourly,
    compute_forced_indices,
    score_net,
    score_peak,
)


# ---------------------------------------------------------------------------
# Helpers to build tiny synthetic inputs
# ---------------------------------------------------------------------------

def _keys(n: int) -> list[str]:
    """Ordered string timestep keys ``t0000 .. t{n-1}``."""
    return [f"t{h:04d}" for h in range(n)]


def _profile_from_array(values: np.ndarray, keys: list[str]):
    """A single-series profile dict from an hourly availability array."""
    return {"p0": list(zip(keys, [float(v) for v in values]))}


# ---------------------------------------------------------------------------
# score_net / score_peak: planted trough and peak
# ---------------------------------------------------------------------------

class TestScores:
    def test_planted_trough_score_net(self):
        """A low-availability period is the highest net load (worst deficit)."""
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        # Availability high everywhere (0.9) except period 2, which is a trough.
        avail = np.full(n_base * period_length, 0.9)
        trough = 2
        avail[trough * period_length:(trough + 1) * period_length] = 0.1
        profiles = _profile_from_array(avail, keys)

        netload = build_netload_hourly(
            profiles, {}, {}, keys, vg_weight=1.0
        )
        scores = score_net(netload, period_length, n_base)
        assert int(np.argmax(scores)) == trough

    def test_planted_peak_score_peak(self):
        """A single very-low-availability hour makes that period the peak."""
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        avail = np.full(n_base * period_length, 0.8)
        peak = 1
        # One hour of near-zero availability -> a spike in (1 - avail).
        avail[peak * period_length + 3] = 0.0
        profiles = _profile_from_array(avail, keys)

        netload = build_netload_hourly(profiles, {}, {}, keys, vg_weight=1.0)
        scores = score_peak(netload, period_length, n_base)
        assert int(np.argmax(scores)) == peak

    def test_subwindow_finds_partial_trough(self):
        """A short trough is found by a sub-window but diluted by whole-period mean."""
        n_base, period_length = 4, 12
        keys = _keys(n_base * period_length)
        avail = np.full(n_base * period_length, 0.9)

        # Period 3: a short deep trough (3 h) inside an otherwise-high period.
        short_period = 3
        s = short_period * period_length
        avail[s + 4:s + 7] = 0.0

        # Period 1: a shallow but period-long dip, higher whole-period mean deficit.
        long_period = 1
        lo = long_period * period_length
        avail[lo:lo + period_length] = 0.6

        profiles = _profile_from_array(avail, keys)
        netload = build_netload_hourly(profiles, {}, {}, keys, vg_weight=1.0)

        # Whole-period mean: the shallow period-long dip wins over the short trough.
        whole = score_net(netload, period_length, n_base, window=None)
        assert int(np.argmax(whole)) == long_period

        # Short sub-window: the deep 3 h trough wins.
        sub = score_net(netload, period_length, n_base, window=3)
        assert int(np.argmax(sub)) == short_period


# ---------------------------------------------------------------------------
# vg_weight invariance for scalar-only inflow
# ---------------------------------------------------------------------------

class TestVgWeightInvariance:
    def test_scalar_only_inflow_invariant(self):
        """With no time-varying inflow, the forced index is the same for any vg_weight."""
        n_base, period_length = 5, 8
        keys = _keys(n_base * period_length)
        avail = np.full(n_base * period_length, 0.85)
        trough = 3
        avail[trough * period_length:(trough + 1) * period_length] = 0.05
        profiles = _profile_from_array(avail, keys)

        # Only scalar demand nodes (negative inflow = demand), no time-varying inflow.
        demand_scalars = {"d_h2": -190.0, "d_nh3": -70.8}

        results = set()
        for vg_weight in (0.3, 0.5, 1.0):
            idx = compute_forced_indices(
                profiles,
                {},
                demand_scalars,
                keys,
                period_length,
                n_base,
                force_peak_load=False,
                force_highest_net_load=True,
                force_window=None,
                vg_weight=vg_weight,
            )
            results.add(tuple(idx))

        # Exactly one distinct result, and it is the planted trough.
        assert len(results) == 1
        assert results == {(trough,)}

    def test_scalar_inflow_term_is_constant(self):
        """Scalar-only inflow makes the inflow term constant across the horizon."""
        n_base, period_length = 3, 4
        keys = _keys(n_base * period_length)
        profiles = _profile_from_array(
            np.linspace(0.2, 0.9, n_base * period_length), keys
        )
        # vg_weight=0 isolates the (normalised) inflow term.
        netload = build_netload_hourly(
            profiles, {}, {"d": -100.0}, keys, vg_weight=0.0
        )
        assert np.allclose(netload, netload[0])


# ---------------------------------------------------------------------------
# Time-varying inflow actually moves the signal
# ---------------------------------------------------------------------------

class TestTimeVaryingInflow:
    def test_demand_spike_drives_net_load(self):
        """A time-varying demand spike (more-negative inflow) raises net load there."""
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        # Flat availability so the VG term carries no period-to-period signal.
        profiles = _profile_from_array(
            np.full(n_base * period_length, 0.7), keys
        )
        # Demand node: constant small demand except a big spike in period 2.
        spike = 2
        inflow_vals = np.full(n_base * period_length, -10.0)
        inflow_vals[spike * period_length:(spike + 1) * period_length] = -500.0
        inflows = {"n0": list(zip(keys, [float(v) for v in inflow_vals]))}

        idx = compute_forced_indices(
            profiles,
            inflows,
            {},
            keys,
            period_length,
            n_base,
            force_peak_load=False,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=0.0,
        )
        assert idx == [spike]


# ---------------------------------------------------------------------------
# Node-group demand-weighting of the VG term
# ---------------------------------------------------------------------------

class TestDemandWeighting:
    def test_weighting_shifts_argmax_to_high_demand_region(self):
        """Unweighted argmax follows the many-profile region's trough; the
        node-group demand-weighted signal shifts it to the high-demand region.

        Region A: 4 profiles, small demand, trough in period 1.
        Region B: 1 profile, large demand, trough in period 3.

        Unweighted (mean over ALL 5 profiles) is dominated by A's 4 profiles →
        argmax = period 1. Demand-weighted (Σ_r D_r·mean_within_r(shortfall))
        is dominated by B's huge D_r → argmax = period 3.
        """
        n_base, period_length = 5, 6
        keys = _keys(n_base * period_length)
        n = n_base * period_length

        a_trough, b_trough = 1, 3

        def _flat_with_trough(trough: int) -> list[float]:
            arr = np.full(n, 1.0)
            arr[trough * period_length:(trough + 1) * period_length] = 0.0
            return [float(v) for v in arr]

        profiles = {}
        a_names = [f"pA{i}" for i in range(4)]
        for name in a_names:
            profiles[name] = list(zip(keys, _flat_with_trough(a_trough)))
        b_names = ["pB0"]
        profiles["pB0"] = list(zip(keys, _flat_with_trough(b_trough)))

        region_profiles = {"A": a_names, "B": b_names}
        region_demand = {"A": 10.0, "B": 1000.0}

        # Unweighted: A's trough (more profiles) wins.
        nl_unw = build_netload_hourly(profiles, {}, {}, keys, vg_weight=1.0)
        assert int(np.argmax(score_net(nl_unw, period_length, n_base))) == a_trough

        # Demand-weighted: B's trough (huge D_r) wins.
        nl_w = build_netload_hourly(
            profiles,
            {},
            {},
            keys,
            vg_weight=1.0,
            region_profiles=region_profiles,
            region_demand=region_demand,
        )
        assert int(np.argmax(score_net(nl_w, period_length, n_base))) == b_trough

        # Same through the orchestrator.
        forced_w = compute_forced_indices(
            profiles,
            {},
            {},
            keys,
            period_length,
            n_base,
            force_peak_load=False,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=1.0,
            region_profiles=region_profiles,
            region_demand=region_demand,
        )
        assert forced_w == [b_trough]

    def test_all_zero_demand_falls_back_to_unweighted(self):
        """When every named region has D_r == 0, the signal falls back to the
        unweighted VG term (identical to passing no region maps)."""
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        avail = np.full(n_base * period_length, 0.9)
        trough = 2
        avail[trough * period_length:(trough + 1) * period_length] = 0.1
        profiles = {"pA0": list(zip(keys, [float(v) for v in avail]))}

        nl_fallback = build_netload_hourly(
            profiles,
            {},
            {},
            keys,
            vg_weight=1.0,
            region_profiles={"A": ["pA0"]},
            region_demand={"A": 0.0},
        )
        nl_unweighted = build_netload_hourly(
            profiles, {}, {}, keys, vg_weight=1.0
        )
        assert np.allclose(nl_fallback, nl_unweighted)


# ---------------------------------------------------------------------------
# Orchestrator: dedup, sort, empty
# ---------------------------------------------------------------------------

class TestComputeForcedIndices:
    def test_no_flags_returns_empty(self):
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        profiles = _profile_from_array(
            np.linspace(0.1, 0.9, n_base * period_length), keys
        )
        idx = compute_forced_indices(
            profiles,
            {},
            {},
            keys,
            period_length,
            n_base,
            force_peak_load=False,
            force_highest_net_load=False,
            force_window=None,
            vg_weight=0.5,
        )
        assert idx == []

    def test_two_flags_same_period_collapse(self):
        """Both flags resolving to one period yield a single deduped index."""
        n_base, period_length = 4, 6
        keys = _keys(n_base * period_length)
        # One period is both the deepest sustained trough AND the single worst
        # hour, so peak and net argmax coincide.
        avail = np.full(n_base * period_length, 0.9)
        worst = 2
        avail[worst * period_length:(worst + 1) * period_length] = 0.05
        profiles = _profile_from_array(avail, keys)

        idx = compute_forced_indices(
            profiles,
            {},
            {},
            keys,
            period_length,
            n_base,
            force_peak_load=True,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=1.0,
        )
        assert idx == [worst]

    def test_two_flags_distinct_periods_sorted(self):
        """Distinct peak and net periods return sorted, deduplicated."""
        n_base, period_length = 5, 10
        keys = _keys(n_base * period_length)
        avail = np.full(n_base * period_length, 0.9)

        # Period 3: shallow period-long dip -> highest sustained (net) deficit.
        net_period = 3
        lo = net_period * period_length
        avail[lo:lo + period_length] = 0.55

        # Period 1: a single near-zero hour -> highest instantaneous peak,
        # but its whole-period mean deficit stays below period 3's.
        peak_period = 1
        avail[peak_period * period_length + 5] = 0.0

        profiles = _profile_from_array(avail, keys)
        idx = compute_forced_indices(
            profiles,
            {},
            {},
            keys,
            period_length,
            n_base,
            force_peak_load=True,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=1.0,
        )
        assert idx == sorted([peak_period, net_period])
        assert idx == [peak_period, net_period]
