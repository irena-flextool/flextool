"""Pure (no-DB) tests for the net-load clustering-matrix math.

Constructs :class:`NetloadInputs` values by hand and exercises the pure
functions in :mod:`flextool.representative_periods.netload`:

* (a) demand-match energy balance to machine precision,
* (b) sign + per-group min-max normalization (demand shape; VRE reduces it),
* (c) determinism (two builds byte-identical),
* (d) the existing-vs-default-vs-solved capacity contract.

The VRE ``upper_limit`` DEFAULT case (item (e)) needs the DB reader's
default-fill and is covered in ``test_netload_reader.py``.
"""

from __future__ import annotations

import numpy as np

from flextool.representative_periods.netload import (
    build_group_capacities,
    build_netload_matrix,
    demand_match_default_caps,
)
from flextool.representative_periods.netload_inputs import NetloadInputs, VreUnit


def _keys(n: int) -> list[str]:
    return [f"t{h:04d}" for h in range(n)]


def _flat_profile(value: float, keys: list[str]) -> list[tuple[str, float]]:
    return [(k, float(value)) for k in keys]


# ---------------------------------------------------------------------------
# (a) demand-match energy balance
# ---------------------------------------------------------------------------

class TestDemandMatchEnergyBalance:
    def test_investable_energy_equals_target(self):
        """Σ investable-VRE energy == target (demand) to machine precision.

        One group, scalar demand 100, two investable VRE units with existing
        capacity 0 and distinct mean availabilities. The demand-match default
        must size them so their combined profiled energy equals the demand.
        """
        keys = _keys(4)
        step_durations = {k: 1.0 for k in keys}  # total_dur = 4
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={},
            demand_scalar={"n1": -100.0},  # |scalar| = 100 demand level
            vre={
                "u1": VreUnit("n1", "p_half", 0.0, 1.0, investable=True),
                "u2": VreUnit("n1", "p_quarter", 0.0, 1.0, investable=True),
            },
            profiles={
                "p_half": _flat_profile(0.5, keys),
                "p_quarter": _flat_profile(0.25, keys),
            },
            step_durations=step_durations,
            granularity="node",
        )
        caps = demand_match_default_caps(inputs, keys)

        total_dur = 4.0
        e_u1 = caps["u1"] * 0.5 * total_dur
        e_u2 = caps["u2"] * 0.25 * total_dur
        # Existing caps are 0 here, so total cap == invested cap.
        assert e_u1 == 100.0 / 2
        assert e_u2 == 100.0 / 2
        assert abs((e_u1 + e_u2) - 100.0) < 1e-12
        # Per-unit inverse-availability sizing.
        assert abs(caps["u1"] - (50.0 / (0.5 * total_dur))) < 1e-12
        assert abs(caps["u2"] - (50.0 / (0.25 * total_dur))) < 1e-12

    def test_existing_vre_reduces_target(self):
        """Existing VRE energy is netted off before sizing the investable unit."""
        keys = _keys(2)
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -100.0},
            vre={
                # Existing-only unit: mean avail 0.5, cap 40 → energy 40*0.5*2=40.
                "ex": VreUnit("n1", "p_half", 40.0, 1.0, investable=False),
                # Investable unit must cover the remaining 60.
                "inv": VreUnit("n1", "p_half", 0.0, 1.0, investable=True),
            },
            profiles={"p_half": _flat_profile(0.5, keys)},
            step_durations={k: 1.0 for k in keys},  # total_dur = 2
        )
        caps = demand_match_default_caps(inputs, keys)
        assert "ex" not in caps  # existing-only units are absent
        # target = 100 - 40 = 60; cap = 60 / (0.5 * 2) = 60.
        assert abs(caps["inv"] - 60.0) < 1e-12

    def test_target_zero_when_existing_covers_demand(self):
        keys = _keys(2)
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -10.0},
            vre={
                "ex": VreUnit("n1", "p1", 100.0, 1.0, investable=False),
                "inv": VreUnit("n1", "p1", 0.0, 1.0, investable=True),
            },
            profiles={"p1": _flat_profile(0.5, keys)},
            step_durations={k: 1.0 for k in keys},
        )
        caps = demand_match_default_caps(inputs, keys)
        # Existing energy 100*0.5*2 = 100 >> demand 10 → target 0 → invest 0.
        assert caps["inv"] == 0.0

    def test_zero_availability_investable_gets_existing_only(self):
        keys = _keys(3)
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -50.0},
            vre={
                "dead": VreUnit("n1", "p_zero", 7.0, 1.0, investable=True),
            },
            profiles={"p_zero": _flat_profile(0.0, keys)},
            step_durations={k: 1.0 for k in keys},
        )
        caps = demand_match_default_caps(inputs, keys)
        # Σ avail·dur == 0 → no usable invest → total cap == existing cap.
        assert caps["dead"] == 7.0

    def test_unequal_durations_energy_balance(self):
        """Σ investable-VRE energy == target with UNEQUAL step durations.

        The VRE energy must be ``Σ_h avail_h · dur_h`` — NOT ``mean(avail) · Σ
        dur_h`` (which diverges once the durations are not all equal). Two
        investable units with distinct non-flat profiles split the demand energy
        equally; their duration-weighted profiled energy must sum to the target
        to machine precision.
        """
        keys = _keys(3)
        dur = {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 3.0}
        avail_a = np.array([0.2, 0.4, 0.6])
        avail_b = np.array([0.5, 0.0, 1.0])
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -100.0},  # E_demand = 100
            vre={
                "ua": VreUnit("n1", "pa", 0.0, 1.0, investable=True),
                "ub": VreUnit("n1", "pb", 0.0, 1.0, investable=True),
            },
            profiles={
                "pa": list(zip(keys, avail_a)),
                "pb": list(zip(keys, avail_b)),
            },
            step_durations=dur,
        )
        caps = demand_match_default_caps(inputs, keys)

        dur_arr = np.array([dur[k] for k in keys])
        w_a = float(np.dot(avail_a, dur_arr))  # 2.8
        w_b = float(np.dot(avail_b, dur_arr))  # 3.5
        # A mean(avail)·total_dur denominator would use 0.4·6 and 0.5·6 instead.
        assert w_a != avail_a.mean() * dur_arr.sum()
        e_a = caps["ua"] * w_a
        e_b = caps["ub"] * w_b
        # Equal energy shares of the 100 demand.
        assert abs(e_a - 50.0) < 1e-12
        assert abs(e_b - 50.0) < 1e-12
        assert abs((e_a + e_b) - 100.0) < 1e-12

    def test_vre_penetration_scales_target(self):
        """vre_penetration=0.5 halves the sized investable energy (linear)."""
        keys = _keys(3)
        dur = {keys[0]: 1.0, keys[1]: 2.0, keys[2]: 3.0}
        avail = np.array([0.2, 0.4, 0.6])
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -100.0},  # E_demand = 100
            vre={"u": VreUnit("n1", "p", 0.0, 1.0, investable=True)},
            profiles={"p": list(zip(keys, avail))},
            step_durations=dur,
        )
        dur_arr = np.array([dur[k] for k in keys])
        w = float(np.dot(avail, dur_arr))

        full = demand_match_default_caps(inputs, keys)  # penetration 1.0 default
        half = demand_match_default_caps(inputs, keys, vre_penetration=0.5)
        # target 1.0 → 100 energy; target 0.5 → 50 energy.
        assert abs(full["u"] * w - 100.0) < 1e-12
        assert abs(half["u"] * w - 50.0) < 1e-12
        assert abs(half["u"] - full["u"] * 0.5) < 1e-12

    def test_vre_penetration_default_is_full_match(self):
        keys = _keys(2)
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -100.0},
            vre={"u": VreUnit("n1", "p", 0.0, 1.0, investable=True)},
            profiles={"p": _flat_profile(0.5, keys)},
            step_durations={k: 1.0 for k in keys},
        )
        assert demand_match_default_caps(inputs, keys) == demand_match_default_caps(
            inputs, keys, vre_penetration=1.0
        )


# ---------------------------------------------------------------------------
# (b) sign + normalization
# ---------------------------------------------------------------------------

class TestSignAndNormalization:
    def test_demand_only_matches_demand_shape(self):
        """A demand-only group's normalized net load == normalized demand.

        Demand is negative inflow, so a more-negative inflow is a larger demand
        and must map to a HIGHER net load. The min-max normalized series must
        therefore be the min-max normalization of ``-inflow``.
        """
        keys = _keys(4)
        # inflow = [-10, -40, -20, -30] → demand = [10, 40, 20, 30].
        inflow = [-10.0, -40.0, -20.0, -30.0]
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            profiles={},
            step_durations={k: 1.0 for k in keys},
        )
        C, n_base, names = build_netload_matrix(inputs, {}, keys, period_length=4)
        assert names == ["g"]
        assert n_base == 1
        demand = -np.array(inflow)
        expected = (demand - demand.min()) / (demand.max() - demand.min())
        # C is (n_agg*PL, n_base) = (4, 1); one period, column 0.
        assert np.allclose(C[:, 0], expected)
        # Argmax hour is the largest-demand hour (index 1).
        assert int(np.argmax(C[:, 0])) == 1

    def test_vre_reduces_net_load(self):
        """Adding VRE supply lowers net load where the profile is high."""
        keys = _keys(4)
        inflow = [-30.0, -30.0, -30.0, -30.0]  # flat demand 30
        avail = [0.0, 1.0, 0.0, 1.0]  # VRE only in hours 1, 3
        base = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            vre={"w": VreUnit("n1", "pf", 0.0, 1.0, investable=True)},
            profiles={"pf": list(zip(keys, avail))},
            step_durations={k: 1.0 for k in keys},
        )
        # With cap 0 the demand is flat → constant series → all zeros.
        C0, _, _ = build_netload_matrix(base, {"w": 0.0}, keys, period_length=4)
        assert np.allclose(C0[:, 0], 0.0)
        # With capacity, hours 1 & 3 drop below hours 0 & 2 → not constant.
        C1, _, _ = build_netload_matrix(base, {"w": 20.0}, keys, period_length=4)
        raw = np.array([30.0, 30.0, 30.0, 30.0]) - 20.0 * np.array(avail)
        expected = (raw - raw.min()) / (raw.max() - raw.min())
        assert np.allclose(C1[:, 0], expected)
        # The VRE hours are now the low-net-load hours.
        assert C1[1, 0] < C1[0, 0]
        assert C1[3, 0] < C1[2, 0]

    def test_each_group_normalized_independently(self):
        """Every aggregation unit's (non-constant) series spans [0, 1]."""
        keys = _keys(4)
        inputs = NetloadInputs(
            units_by_group={"a": ["na"], "b": ["nb"]},
            demand_ts={
                "na": list(zip(keys, [-10.0, -20.0, -30.0, -40.0])),
                # Different scale entirely for group b.
                "nb": list(zip(keys, [-1000.0, -2000.0, -1500.0, -1250.0])),
            },
            profiles={},
            step_durations={k: 1.0 for k in keys},
        )
        C, n_base, names = build_netload_matrix(inputs, {}, keys, period_length=4)
        assert names == ["a", "b"]
        # Rows 0-3 = group a, rows 4-7 = group b (block order = sorted names).
        block_a = C[0:4, 0]
        block_b = C[4:8, 0]
        for block in (block_a, block_b):
            assert abs(block.min() - 0.0) < 1e-12
            assert abs(block.max() - 1.0) < 1e-12

    def test_scalar_demand_positive_convention(self):
        """A scalar demand level enters as +|value| (positive demand)."""
        keys = _keys(4)
        # Time-varying supply (positive inflow) + a constant scalar demand.
        inflow = [5.0, 10.0, 15.0, 20.0]  # positive = supply → lowers net load
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            demand_scalar={"n1": -100.0},  # |scalar| = 100 constant demand
            profiles={},
            step_durations={k: 1.0 for k in keys},
        )
        C, _, _ = build_netload_matrix(inputs, {}, keys, period_length=4)
        raw = 100.0 - np.array(inflow)  # net demand per hour
        expected = (raw - raw.min()) / (raw.max() - raw.min())
        assert np.allclose(C[:, 0], expected)


# ---------------------------------------------------------------------------
# (c) determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def _inputs(self) -> NetloadInputs:
        keys = _keys(8)
        return NetloadInputs(
            units_by_group={"g2": ["n2"], "g1": ["n1"]},  # unsorted on purpose
            demand_ts={
                "n1": list(zip(keys, [-10.0, -12.0, -8.0, -20.0, -30.0, -5.0, -7.0, -9.0])),
                "n2": list(zip(keys, [-100.0, -90.0, -80.0, -70.0, -60.0, -50.0, -40.0, -30.0])),
            },
            vre={
                "wb": VreUnit("n2", "pb", 0.0, 1.0, investable=True),
                "wa": VreUnit("n1", "pa", 0.0, 1.0, investable=True),
            },
            profiles={
                "pa": list(zip(keys, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])),
                "pb": list(zip(keys, [0.5, 0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8])),
            },
            step_durations={k: 1.0 for k in keys},
        )

    def test_two_builds_byte_identical(self):
        inputs = self._inputs()
        keys = _keys(8)
        caps = build_group_capacities(
            inputs, demand_match_default_caps(inputs, keys), None
        )
        C1, n1, names1 = build_netload_matrix(inputs, caps, keys, period_length=4)
        C2, n2, names2 = build_netload_matrix(inputs, caps, keys, period_length=4)
        assert np.array_equal(C1, C2)
        assert n1 == n2
        assert names1 == names2 == ["g1", "g2"]  # sorted block order

    def test_default_caps_deterministic(self):
        inputs = self._inputs()
        keys = _keys(8)
        c1 = demand_match_default_caps(inputs, keys)
        c2 = demand_match_default_caps(inputs, keys)
        assert list(c1.items()) == list(c2.items())  # same order + values


# ---------------------------------------------------------------------------
# (d) existing-vs-default-vs-solved capacity contract
# ---------------------------------------------------------------------------

class TestCapacityContract:
    def _inputs(self) -> NetloadInputs:
        return NetloadInputs(
            units_by_group={"g": ["n1"]},
            vre={
                "ex": VreUnit("n1", "p1", 12.5, 1.0, investable=False),
                "inv": VreUnit("n1", "p1", 4.0, 1.0, investable=True),
            },
            profiles={"p1": _flat_profile(0.5, _keys(2))},
            step_durations={k: 1.0 for k in _keys(2)},
        )

    def test_default_path(self):
        inputs = self._inputs()
        default_caps = {"inv": 99.0}
        caps = build_group_capacities(inputs, default_caps, solved_caps=None)
        assert caps["ex"] == 12.5  # non-investable → existing_cap
        assert caps["inv"] == 99.0  # investable → default

    def test_solved_path_overrides_default(self):
        inputs = self._inputs()
        default_caps = {"inv": 99.0}
        caps = build_group_capacities(
            inputs, default_caps, solved_caps={"inv": 42.0}
        )
        assert caps["ex"] == 12.5  # existing still fixed
        assert caps["inv"] == 42.0  # solved overrides default

    def test_investable_absent_falls_back_to_existing(self):
        inputs = self._inputs()
        # Neither default nor solved carry the investable unit.
        caps = build_group_capacities(inputs, {}, solved_caps=None)
        assert caps["inv"] == 4.0  # falls back to existing_cap

    def test_solved_provided_but_unit_missing_uses_default(self):
        inputs = self._inputs()
        caps = build_group_capacities(
            inputs, {"inv": 99.0}, solved_caps={"other": 1.0}
        )
        # solved_caps provided but does not contain 'inv' → default used.
        assert caps["inv"] == 99.0


# ---------------------------------------------------------------------------
# (e) partial-coverage series are SKIPPED, never zero-filled
# ---------------------------------------------------------------------------

class TestPartialCoverageSkipped:
    def test_partial_profile_vre_skipped(self, capsys):
        """A VRE profile missing a used timestep is skipped (with warning).

        The unit must NOT contribute a zero-filled fake net-load; the group's
        net load stays the pure demand shape.
        """
        keys = _keys(4)
        inflow = [-10.0, -40.0, -20.0, -30.0]
        # Profile covers only 3 of the 4 used keys → partial coverage.
        partial = list(zip(keys[:3], [1.0, 1.0, 1.0]))
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            vre={"w": VreUnit("n1", "pf", 0.0, 1.0, investable=True)},
            profiles={"pf": partial},
            step_durations={k: 1.0 for k in keys},
        )
        C, _, _ = build_netload_matrix(inputs, {"w": 50.0}, keys, period_length=4)
        out = capsys.readouterr().out
        assert "skipping VRE unit 'w'" in out
        # VRE not subtracted → net load is the normalized demand shape.
        demand = -np.array(inflow)
        expected = (demand - demand.min()) / (demand.max() - demand.min())
        assert np.allclose(C[:, 0], expected)

    def test_partial_inflow_demand_skipped(self, capsys):
        """A demand inflow series missing used timesteps is skipped (warning)."""
        keys = _keys(4)
        full = [-10.0, -40.0, -20.0, -30.0]
        partial = list(zip(keys[:2], [-5.0, -5.0]))  # covers 2 of 4 keys
        inputs = NetloadInputs(
            units_by_group={"g": ["n1", "n2"]},
            demand_ts={"n1": list(zip(keys, full)), "n2": partial},
            profiles={},
            step_durations={k: 1.0 for k in keys},
        )
        C, _, _ = build_netload_matrix(inputs, {}, keys, period_length=4)
        out = capsys.readouterr().out
        assert "skipping inflow of node 'n2'" in out
        # Only n1's demand shapes the net load; n2 is dropped, not zero-filled.
        demand = -np.array(full)
        expected = (demand - demand.min()) / (demand.max() - demand.min())
        assert np.allclose(C[:, 0], expected)

    def test_partial_profile_skipped_in_demand_match(self, capsys):
        """demand_match also skips a partial VRE profile (no zero-fill energy)."""
        keys = _keys(4)
        partial = list(zip(keys[:2], [0.5, 0.5]))
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_scalar={"n1": -100.0},
            vre={
                "ex": VreUnit("n1", "pf", 40.0, 1.0, investable=False),
                "inv": VreUnit("n1", "pg", 0.0, 1.0, investable=True),
            },
            profiles={
                "pf": partial,  # existing unit's profile is partial → skipped
                "pg": _flat_profile(0.5, keys),
            },
            step_durations={k: 1.0 for k in keys},
        )
        caps = demand_match_default_caps(inputs, keys)
        out = capsys.readouterr().out
        assert "skipping VRE unit 'ex'" in out
        # ex's energy is NOT subtracted → target stays the full 100 demand.
        # cap = 100 / (0.5 · 4) = 50.
        assert abs(caps["inv"] - 50.0) < 1e-12


# ---------------------------------------------------------------------------
# (f) co-location warning: VRE capacity but no demand on the aggregation unit
# ---------------------------------------------------------------------------

class TestColocationWarning:
    def _keys_avail(self):
        keys = _keys(4)
        return keys, [0.2, 0.4, 0.6, 0.8]

    def test_vre_only_group_warns(self, capsys):
        keys, avail = self._keys_avail()
        inputs = NetloadInputs(
            units_by_group={"g_vre": ["n_vre"]},
            demand_ts={},
            vre={"w": VreUnit("n_vre", "pf", 10.0, 1.0, investable=False)},
            profiles={"pf": list(zip(keys, avail))},
            step_durations={k: 1.0 for k in keys},
        )
        build_netload_matrix(inputs, {"w": 10.0}, keys, period_length=4)
        out = capsys.readouterr().out
        assert "has VRE capacity but" in out
        assert "g_vre" in out

    def test_demand_and_vre_group_no_warn(self, capsys):
        keys, avail = self._keys_avail()
        inflow = [-10.0, -40.0, -20.0, -30.0]
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            vre={"w": VreUnit("n1", "pf", 10.0, 1.0, investable=False)},
            profiles={"pf": list(zip(keys, avail))},
            step_durations={k: 1.0 for k in keys},
        )
        build_netload_matrix(inputs, {"w": 10.0}, keys, period_length=4)
        out = capsys.readouterr().out
        assert "has VRE capacity but" not in out

    def test_demand_only_group_no_warn(self, capsys):
        keys = _keys(4)
        inflow = [-10.0, -40.0, -20.0, -30.0]
        inputs = NetloadInputs(
            units_by_group={"g": ["n1"]},
            demand_ts={"n1": list(zip(keys, inflow))},
            profiles={},
            step_durations={k: 1.0 for k in keys},
        )
        build_netload_matrix(inputs, {}, keys, period_length=4)
        out = capsys.readouterr().out
        assert "has VRE capacity but" not in out
