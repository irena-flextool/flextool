"""Wiring tests for force-include representative-period selection.

Drives ``preprocess_representative_periods`` (and its internal pieces) on a
schema-complete database built from the FlexTool JSON schema (CLAUDE.md
invariant #3 — never read a checked-in ``.sqlite`` for building). Covers the
opt-in contract of ``specs/repperiod_forceinclude_design.md`` §8:

1. byte-parity of the default (no-force) path,
2. weight conservation (partition-of-unity) after augmentation,
3. seeded selection pinned at n_rp + dedup,
4. the ``+f{n}`` naming rule.

The synthetic fixture is a controlled 6-week (period_length=24) horizon with a
single VRE ``profile`` whose availability collapses to a sustained trough in
base period 5 while periods 0-4 carry two distinct extreme square-wave shapes.
Consequences that make the assertions deterministic:

* The hull (which scores *shape*) fills its picks from the two distinct
  square-wave families and never selects the flat trough period 5.
* The sustained-net-load score (which scores *level*) peaks in period 5.

So ``force_highest_net_load`` forces base period 5 — a period the hull leaves
out — giving a genuine (non-dedup) augmentation.
"""

from __future__ import annotations

import numpy as np
import pytest
import spinedb_api as api
from spinedb_api import DatabaseMapping, Map, import_data

from flextool._resources import package_data_path
from flextool.representative_periods import force_include
from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.preprocess import (
    _build_clustering_matrix,
    preprocess_representative_periods,
)
from flextool.representative_periods.weights import compute_weight_matrix
from flextool.update_flextool import initialize_database

PERIOD_LENGTH = 24
N_PERIODS = 6
SCENARIO = "wiring_scen"
ALT = "wiring_base"


def _timestep_keys() -> list[str]:
    return [f"t{i}" for i in range(N_PERIODS * PERIOD_LENGTH)]


def _wind_availability() -> list[float]:
    """Per-hour VRE availability over the whole horizon.

    Periods 0/2/4: high first half, low second half (pattern A).
    Periods 1/3:   low first half, high second half (pattern B).
    Period 5:      flat sustained trough (0.05) — the forced week.
    """
    values: list[float] = []
    for p in range(N_PERIODS):
        for h in range(PERIOD_LENGTH):
            if p == 5:
                values.append(0.05)
            elif p % 2 == 0:  # pattern A
                values.append(1.0 if h < PERIOD_LENGTH // 2 else 0.0)
            else:  # pattern B
                values.append(0.0 if h < PERIOD_LENGTH // 2 else 1.0)
    return values


def _synthetic_series() -> tuple[
    dict[str, list[tuple[str, float]]],
    dict[str, list[tuple[str, float]]],
    dict[str, float],
    list[str],
]:
    """The same data as plain dicts (no DB) for the pure-numpy piece tests."""
    keys = _timestep_keys()
    profiles = {"wind": list(zip(keys, _wind_availability()))}
    inflows: dict[str, list[tuple[str, float]]] = {}
    demand_scalars = {"demand": -1000.0}
    return profiles, inflows, demand_scalars, keys


def _build_controlled_db(db_path: str) -> str:
    """Initialise a schema-complete DB and import the controlled scenario."""
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"

    keys = _timestep_keys()
    wind = _wind_availability()
    timeline_map = Map(keys, [1.0] * len(keys))
    wind_map = Map(keys, wind)
    period_timeset_map = Map(["y2050"], ["placeholder_ts"])

    with DatabaseMapping(url) as db:
        count, errors = import_data(
            db,
            alternatives=[(ALT, "controlled wiring fixture")],
            scenarios=[(SCENARIO, True, "wiring scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=[
                ("timeline", "tl", None),
                ("profile", "wind", None),
                ("node", "demand", None),
                ("solve", "s1", None),
            ],
            parameter_values=[
                ("timeline", "tl", "timestep_duration", timeline_map, ALT),
                ("profile", "wind", "profile", wind_map, ALT),
                ("node", "demand", "inflow", -1000.0, ALT),
                ("solve", "s1", "period_timeset", period_timeset_map, ALT),
            ],
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("controlled wiring fixture")
    return url


def _read_timeset_map(
    url: str, timeset_name: str, param: str
) -> tuple[Map, tuple]:
    """Return (Map object, (raw_value_bytes, raw_type)) for a timeset param."""
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        matches = [
            pv
            for pv in db.find_parameter_values(
                entity_class_name="timeset", parameter_definition_name=param
            )
            if pv["entity_name"] == timeset_name
        ]
    assert len(matches) == 1, (
        f"expected exactly one '{param}' on timeset '{timeset_name}', "
        f"got {len(matches)}"
    )
    pv = matches[0]
    value_obj = api.from_database(pv["value"], pv["type"])
    return value_obj, (pv["value"], pv["type"])


@pytest.fixture
def controlled_db(tmp_path):
    db_path = str(tmp_path / "controlled.sqlite")
    return _build_controlled_db(db_path)


# ---------------------------------------------------------------------------
# 1. Byte-parity of the default (no-force) path.
# ---------------------------------------------------------------------------

class TestByteParityDefault:
    def test_default_path_is_byte_identical_and_unsuffixed(self, tmp_path):
        url_a = _build_controlled_db(str(tmp_path / "a.sqlite"))
        url_b = _build_controlled_db(str(tmp_path / "b.sqlite"))

        name_a = preprocess_representative_periods(
            url_a, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH
        )
        name_b = preprocess_representative_periods(
            url_b, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH
        )

        # No force flags → no +f suffix; the base name only.
        assert name_a == "hull_2rp_24h"
        assert name_b == name_a
        assert "+f" not in name_a

        _, dur_a = _read_timeset_map(url_a, name_a, "timeset_duration")
        _, dur_b = _read_timeset_map(url_b, name_b, "timeset_duration")
        _, w_a = _read_timeset_map(url_a, name_a, "representative_period_weights")
        _, w_b = _read_timeset_map(url_b, name_b, "representative_period_weights")

        # Serialised (value, type) bytes identical across two default runs.
        assert dur_a == dur_b
        assert w_a == w_b


# ---------------------------------------------------------------------------
# 2. Weight conservation (partition of unity) after augmentation.
# ---------------------------------------------------------------------------

class TestWeightConservation:
    def test_conserved_over_augmented_rep_set(self):
        profiles, inflows, demand_scalars, keys = _synthetic_series()
        C, n_base = _build_clustering_matrix(
            profiles, inflows, keys, PERIOD_LENGTH
        )
        hull = sorted(greedy_convex_hull_clustering(C, 2))
        forced = force_include.compute_forced_indices(
            profiles,
            inflows,
            demand_scalars,
            keys,
            PERIOD_LENGTH,
            n_base,
            force_peak_load=False,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=0.5,
        )
        assert forced == [5], f"expected forced=[5], got {forced}"
        assert 5 not in hull, f"trough period leaked into hull {hull}"

        rep_indices = sorted(set(hull) | set(forced))
        W = compute_weight_matrix(C, rep_indices)

        # Partition of unity: every base-period row sums to 1, and the total
        # mass equals the number of base periods (annual mass conserved).
        assert abs(W.sum() - n_base) < 1e-9
        row_sums = W.sum(axis=1)
        assert float(np.max(np.abs(row_sums - 1.0))) < 1e-9


# ---------------------------------------------------------------------------
# 3. Seeded selection: total pinned at n_rp + dedup.
# ---------------------------------------------------------------------------

class TestCountModes:
    def _rep_count(self, url, name) -> int:
        dur_map, _ = _read_timeset_map(url, name, "timeset_duration")
        return len(dur_map.indexes)

    def _rep_start_keys(self, url, name) -> list[str]:
        dur_map, _ = _read_timeset_map(url, name, "timeset_duration")
        return [str(k) for k in dur_map.indexes]

    def test_forced_period_seeded_total_pinned_at_n_rp(self, tmp_path):
        # Forcing SEEDS the greedy hull with base period 5 and keeps the total
        # at n_rp: the forced period consumes one of the n_rp slots, it never
        # grows the set beyond n_rp (the single, seeded behaviour).
        url_default = _build_controlled_db(str(tmp_path / "d.sqlite"))
        url_forced = _build_controlled_db(str(tmp_path / "f.sqlite"))

        default_name = preprocess_representative_periods(
            url_default, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH
        )
        forced_name = preprocess_representative_periods(
            url_forced,
            SCENARIO,
            n_rp=2,
            period_length=PERIOD_LENGTH,
            force_highest_net_load=True,
        )

        # Forced period 5 is not a pure-hull pick → it displaces one → +f1, but
        # the total stays pinned at n_rp for both the default and forced runs.
        assert default_name == "hull_2rp_24h"
        assert forced_name == "hull_2rp_24h+f1"
        assert self._rep_count(url_default, default_name) == 2
        assert self._rep_count(url_forced, forced_name) == 2
        # The forced trough (base period 5, start key t120) is in the seeded set.
        forced_key = _timestep_keys()[5 * PERIOD_LENGTH]
        assert forced_key in self._rep_start_keys(url_forced, forced_name)

    def test_dedup_forced_already_in_hull_adds_nothing(self, tmp_path):
        # With n_rp == n_base every base period is already a hull pick, so the
        # forced period is a dedup no-op: no +f suffix, count unchanged.
        url = _build_controlled_db(str(tmp_path / "dd.sqlite"))
        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_PERIODS,
            period_length=PERIOD_LENGTH,
            force_highest_net_load=True,
        )
        assert name == f"hull_{N_PERIODS}rp_24h"
        assert "+f" not in name
        assert self._rep_count(url, name) == N_PERIODS


# ---------------------------------------------------------------------------
# 4. Naming rule.
# ---------------------------------------------------------------------------

class TestNaming:
    def test_default_unsuffixed_forced_suffixed(self, tmp_path):
        url_default = _build_controlled_db(str(tmp_path / "n0.sqlite"))
        url_forced = _build_controlled_db(str(tmp_path / "n1.sqlite"))

        default_name = preprocess_representative_periods(
            url_default, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH
        )
        forced_name = preprocess_representative_periods(
            url_forced,
            SCENARIO,
            n_rp=2,
            period_length=PERIOD_LENGTH,
            force_highest_net_load=True,
        )
        assert default_name == "hull_2rp_24h"
        assert forced_name == "hull_2rp_24h+f1"
