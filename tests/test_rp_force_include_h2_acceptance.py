"""H2_trade acceptance tests for force-include (design §8.5, §7).

Two tests on the ``H2_trade`` / ``y2050_rp`` worked example:

1. **Default (unweighted) documents the no-op limitation.** With no region
   groups the system-scope unweighted net-load argmax is base index **22**,
   which is *already* a hull pick, so nothing is appended and the timeset name
   stays ``hull_5rp_168h`` (no ``+f``). This pins the documented limitation of
   the unweighted aggregate — it does not surface the binding autumn week.

2. **Region-weighted acceptance (the real fix).** With
   ``region_groups=["decomp_AUS","decomp_JAP","decomp_KOR"]`` the node-group
   demand-weighted signal forces base index **41** (start ``2050-10-15``), the
   binding coincident autumn week: the timeset becomes ``hull_5rp_168h+f1`` and
   the 5 pure-hull picks ``{3,8,14,22,49}`` are all still present (grow mode).

``H2_trade.sqlite`` is a real INPUT-source database; reading it as the tool's
input is exactly the tool's job (CLAUDE.md invariant #3 is about test-BUILT
model DBs, not the tool's real input). The checked-in file is COPIED to
``tmp_path`` before running so it is never mutated.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import spinedb_api as api
from spinedb_api import DatabaseMapping

from flextool.representative_periods import force_include
from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.preprocess import (
    _build_clustering_matrix,
    _get_timeline_keys,
    _read_region_maps,
    _read_time_series,
    preprocess_representative_periods,
)

H2_DB = (
    Path(__file__).resolve().parent.parent
    / "projects"
    / "test-engine"
    / "input_sources"
    / "H2_trade.sqlite"
)
SCENARIO = "y2050_rp"
N_RP = 5
PERIOD_LENGTH = 168
REGION_GROUPS = ["decomp_AUS", "decomp_JAP", "decomp_KOR"]

EXPECTED_HULL = [3, 8, 14, 22, 49]
EXPECTED_UNWEIGHTED = 22  # already a hull pick -> no-op
EXPECTED_WEIGHTED = 41
EXPECTED_WEIGHTED_KEY = "2050-10-15T00:00:00"


def _copy_db(tmp_path) -> str:
    db_copy = tmp_path / "H2_trade.sqlite"
    shutil.copy(H2_DB, db_copy)
    return f"sqlite:///{db_copy}"


def _read_inputs(url: str):
    """Replicate the tool's scenario-filtered read, incl. region maps."""
    scen_config = api.filters.scenario_filter.scenario_filter_config(SCENARIO)
    with DatabaseMapping(url) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        db.fetch_all("parameter_value")
        profiles, inflows, demand_scalars = _read_time_series(db)
        timestep_keys = _get_timeline_keys(db)
        region_profiles, region_demand, _region_nodes = _read_region_maps(
            db, REGION_GROUPS, demand_scalars
        )
    return (
        profiles,
        inflows,
        demand_scalars,
        timestep_keys,
        region_profiles,
        region_demand,
    )


def _read_timeset_map(url: str, timeset_name: str, param: str):
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        matches = [
            pv
            for pv in db.find_parameter_values(
                entity_class_name="timeset", parameter_definition_name=param
            )
            if pv["entity_name"] == timeset_name
        ]
    assert len(matches) == 1
    pv = matches[0]
    return api.from_database(pv["value"], pv["type"])


@pytest.mark.slow
class TestH2ForceIncludeAcceptance:
    def test_unweighted_is_documented_no_op(self, tmp_path):
        if not H2_DB.exists():
            pytest.skip(f"H2_trade fixture not present at {H2_DB}")
        url = _copy_db(tmp_path)

        profiles, inflows, demand_scalars, timestep_keys, _, _ = _read_inputs(url)
        n_base = len(timestep_keys) // PERIOD_LENGTH

        C, _ = _build_clustering_matrix(
            profiles, inflows, timestep_keys, PERIOD_LENGTH
        )
        hull = sorted(greedy_convex_hull_clustering(C, N_RP))
        assert hull == EXPECTED_HULL

        forced = force_include.compute_forced_indices(
            profiles,
            inflows,
            demand_scalars,
            timestep_keys,
            PERIOD_LENGTH,
            n_base,
            force_peak_load=False,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=0.5,
        )
        # Unweighted argmax is idx 22, which is already a hull pick.
        assert forced == [EXPECTED_UNWEIGHTED]
        assert EXPECTED_UNWEIGHTED in hull

        # End-to-end: forcing an already-selected period is a dedup no-op, so
        # the name stays unsuffixed and the count is unchanged.
        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_RP,
            period_length=PERIOD_LENGTH,
            force_highest_net_load=True,
        )
        assert name == "hull_5rp_168h"
        dur_map = _read_timeset_map(url, name, "timeset_duration")
        assert len(dur_map.indexes) == N_RP

    def test_region_weighted_forces_autumn_week(self, tmp_path):
        if not H2_DB.exists():
            pytest.skip(f"H2_trade fixture not present at {H2_DB}")
        url = _copy_db(tmp_path)

        (
            profiles,
            inflows,
            demand_scalars,
            timestep_keys,
            region_profiles,
            region_demand,
        ) = _read_inputs(url)
        n_base = len(timestep_keys) // PERIOD_LENGTH

        C, _ = _build_clustering_matrix(
            profiles, inflows, timestep_keys, PERIOD_LENGTH
        )
        hull = sorted(greedy_convex_hull_clustering(C, N_RP))
        assert hull == EXPECTED_HULL

        forced = force_include.compute_forced_indices(
            profiles,
            inflows,
            demand_scalars,
            timestep_keys,
            PERIOD_LENGTH,
            n_base,
            force_peak_load=False,
            force_highest_net_load=True,
            force_window=None,
            vg_weight=0.5,
            region_profiles=region_profiles,
            region_demand=region_demand,
        )
        # If this is NOT [41] the demand-weighting is wrong; report the actual
        # argmax and the top-5 net-load ranking rather than fudging.
        if forced != [EXPECTED_WEIGHTED]:
            netload = force_include.build_netload_hourly(
                profiles,
                inflows,
                demand_scalars,
                timestep_keys,
                vg_weight=0.5,
                region_profiles=region_profiles,
                region_demand=region_demand,
            )
            scores = force_include.score_net(netload, PERIOD_LENGTH, n_base, None)
            ranking = np.argsort(scores)[::-1][:5]
            pytest.fail(
                "region-weighted force_highest_net_load did not select idx 41.\n"
                f"  forced (argmax) = {forced}\n"
                f"  top-5 score_net base indices = {ranking.tolist()}\n"
                f"  their scores = {scores[ranking].tolist()}"
            )

        assert timestep_keys[EXPECTED_WEIGHTED * PERIOD_LENGTH] == EXPECTED_WEIGHTED_KEY

        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_RP,
            period_length=PERIOD_LENGTH,
            force_highest_net_load=True,
            force_count_mode="grow",
            region_groups=REGION_GROUPS,
        )
        assert name == "hull_5rp_168h+f1"

        dur_map = _read_timeset_map(url, name, "timeset_duration")
        rep_keys = set(dur_map.indexes)

        # Forced autumn week present.
        assert EXPECTED_WEIGHTED_KEY in rep_keys

        # All pure-hull picks still present (grow mode adds, never drops).
        for idx in EXPECTED_HULL:
            hull_key = timestep_keys[idx * PERIOD_LENGTH]
            assert hull_key in rep_keys, (
                f"hull pick idx {idx} (start {hull_key}) missing from rep set"
            )

        assert len(rep_keys) == N_RP + 1
