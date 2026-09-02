"""End-to-end tests for the ``--netload-clustering`` preprocess mode (Phase 3).

Every database is built from the live schema (``initialize_database`` off
``schemas/spinedb_schema.json``) — never a checked-in ``.sqlite`` (invariant #3).
The public entry point ``preprocess_representative_periods`` is driven against a
temp DB and the written ``timeset`` Maps are read back for assertions.

Coverage:

* **Byte-parity default path** — ``netload_clustering=False`` still routes
  through ``_build_clustering_matrix`` and writes the exact ``hull_*`` name and
  the exact ``timeset_duration`` / ``representative_period_weights`` Maps the
  pure-hull pipeline produces (guard against default-path drift).
* **Net-load mode** — ``netload_clustering=True`` runs end-to-end and writes a
  distinct ``netload_*`` alternative with a valid non-empty RP set.
* **Scenario-filter parity** — the net-load reader runs inside the scenario
  filter, so it sees scenario-specific ``existing`` capacity.
* **vre_penetration threading** — the value reaches
  ``demand_match_default_caps``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spinedb_api as api
from spinedb_api import DatabaseMapping, Map, import_data

from flextool.representative_periods import preprocess as pp
from flextool.representative_periods.clustering import (
    greedy_convex_hull_clustering,
)
from flextool.representative_periods.preprocess import (
    _build_clustering_matrix,
    _build_timeset_duration_map,
    _build_weights_map,
    _get_timeline_keys,
    _read_time_series,
    preprocess_representative_periods,
)
from flextool.representative_periods.weights import compute_weight_matrix
from flextool.update_flextool.initialize_database import initialize_database

_SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "flextool"
    / "schemas"
    / "spinedb_schema.json"
)

# 8 timesteps → with period_length 2 gives 4 base periods.
_KEYS = [f"t{i:02d}" for i in range(1, 9)]
_N_RP = 2
_PERIOD_LENGTH = 2


def _build_db(
    db_path: Path,
    *,
    wind_existing: float = 1.0,
    scenario_name: str = "base",
) -> str:
    """A one-node model: demand + a co-located investable wind unit.

    The wind unit outputs to the same node that carries the (negative) demand
    inflow, so under per-node granularity the node's net load is
    ``demand − cap·avail`` — a well-formed net-load signal. Both the profile and
    the inflow are time-varying, so the default profile/inflow stack also has
    two usable features.
    """
    initialize_database(str(_SCHEMA), str(db_path))
    url = f"sqlite:///{db_path}"

    # Demand and availability chosen so periods are clearly distinguishable.
    demand = Map(_KEYS, [-10.0, -40.0, -80.0, -20.0, -15.0, -90.0, -30.0, -25.0])
    wind_pf = Map(_KEYS, [0.9, 0.2, 0.1, 0.8, 0.7, 0.05, 0.6, 0.5])
    timeline = Map(_KEYS, [1.0] * len(_KEYS))

    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            alternatives=[[scenario_name, ""]],
            scenarios=[[scenario_name, True, ""]],
            scenario_alternatives=[[scenario_name, scenario_name]],
            entities=[
                ["node", ["n1"], None],
                ["profile", ["wind_profile"], None],
                ["unit", ["wind"], None],
                ["timeline", ["main"], None],
                ["unit__outputNode", ["wind", "n1"], None],
                ["unit__node__profile", ["wind", "n1", "wind_profile"], None],
            ],
            # Activate every entity in the scenario's alternative — the scenario
            # filter drops parameter values of entities that are not active, so
            # without this the reader (running inside the filter) sees no inflow
            # / existing / VRE.
            entity_alternatives=[
                ["node", ["n1"], scenario_name, True],
                ["profile", ["wind_profile"], scenario_name, True],
                ["unit", ["wind"], scenario_name, True],
                ["timeline", ["main"], scenario_name, True],
                ["unit__outputNode", ["wind", "n1"], scenario_name, True],
                [
                    "unit__node__profile",
                    ["wind", "n1", "wind_profile"],
                    scenario_name,
                    True,
                ],
            ],
            parameter_values=[
                ["node", "n1", "inflow", demand, scenario_name],
                ["profile", "wind_profile", "profile", wind_pf, scenario_name],
                ["timeline", "main", "timestep_duration", timeline, scenario_name],
                # NO profile_method → schema default upper_limit → VRE.
                ["unit", "wind", "existing", wind_existing, scenario_name],
                ["unit", "wind", "invest_method", "invest_no_limit", scenario_name],
            ],
        )
        assert not errors, errors
        db.commit_session("netload preprocess fixture")
    return url


def _read_timeset_maps(url: str, alt: str, timeset_name: str) -> tuple[Map, Map]:
    """Read back ``timeset_duration`` and ``representative_period_weights``."""
    dur: Map | None = None
    wts: Map | None = None
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        for param, target in (
            ("timeset_duration", "dur"),
            ("representative_period_weights", "wts"),
        ):
            for pv in db.find_parameter_values(
                entity_class_name="timeset",
                parameter_definition_name=param,
            ):
                if (
                    pv["entity_name"] == timeset_name
                    and pv["alternative_name"] == alt
                ):
                    value = api.from_database(pv["value"], pv["type"])
                    if target == "dur":
                        dur = value
                    else:
                        wts = value
    assert dur is not None and wts is not None, "timeset Maps not written"
    return dur, wts


def _map_pairs(m: Map) -> list[tuple[str, float]]:
    return list(zip([str(i) for i in m.indexes], [float(v) for v in m.values]))


def _weights_nested(m: Map) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for idx, val in zip(m.indexes, m.values):
        out[str(idx)] = _map_pairs(val)
    return out


def test_default_path_byte_parity(tmp_path: Path):
    """netload_clustering=False writes the exact pure-hull name and Maps.

    The expected Maps are recomputed from the SAME pipeline the default path
    uses (``_build_clustering_matrix`` → greedy hull → ``compute_weight_matrix``
    → the ``_build_*_map`` helpers); the DB round-trip must match byte-for-byte,
    proving the netload branch did not perturb the default path.
    """
    url = _build_db(tmp_path / "default.sqlite")

    timeset_name = preprocess_representative_periods(
        url,
        scenario_name="base",
        n_rp=_N_RP,
        period_length=_PERIOD_LENGTH,
    )

    # Name is the pure-hull default (no netload_ prefix, no force suffix).
    assert timeset_name == f"hull_{_N_RP}rp_{_PERIOD_LENGTH}h"

    # Recompute the expected selection + Maps via the default pipeline.
    with DatabaseMapping(url) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config("base")
        )
        db.fetch_all("parameter_value")
        profiles, inflows, _demand = _read_time_series(db)
        timestep_keys = _get_timeline_keys(db)
    C, n_base = _build_clustering_matrix(
        profiles, inflows, timestep_keys, _PERIOD_LENGTH
    )
    rep_indices = sorted(greedy_convex_hull_clustering(C, _N_RP))
    W = compute_weight_matrix(C, rep_indices)
    exp_dur = _build_timeset_duration_map(
        rep_indices, timestep_keys, _PERIOD_LENGTH
    )
    exp_wts = _build_weights_map(
        W, rep_indices, timestep_keys, _PERIOD_LENGTH, n_base
    )

    dur, wts = _read_timeset_maps(url, timeset_name, timeset_name)
    assert _map_pairs(dur) == _map_pairs(exp_dur)
    assert _weights_nested(wts) == _weights_nested(exp_wts)


def test_netload_mode_writes_distinct_alternative(tmp_path: Path):
    """netload_clustering=True writes a distinct netload_* alternative + RP set."""
    url = _build_db(tmp_path / "netload.sqlite")

    timeset_name = preprocess_representative_periods(
        url,
        scenario_name="base",
        n_rp=_N_RP,
        period_length=_PERIOD_LENGTH,
        netload_clustering=True,
    )

    assert timeset_name == f"netload_{_N_RP}rp_{_PERIOD_LENGTH}h"
    assert timeset_name.startswith("netload_")

    dur, wts = _read_timeset_maps(url, timeset_name, timeset_name)
    dur_pairs = _map_pairs(dur)
    # Exactly n_rp representative periods, each with the period-length duration.
    assert len(dur_pairs) == _N_RP
    assert all(v == float(_PERIOD_LENGTH) for _k, v in dur_pairs)

    # Weights: every base period's row sums to ~1 (convex).
    nested = _weights_nested(wts)
    assert nested, "weights map is empty"
    for _base, inner in nested.items():
        assert abs(sum(v for _r, v in inner) - 1.0) < 1e-6
        assert all(v >= -1e-8 for _r, v in inner)

    # The alternative exists and is named netload_*.
    with DatabaseMapping(url) as db:
        alt_names = {a["name"] for a in db.get_alternative_items()}
    assert timeset_name in alt_names


def test_scenario_filter_sees_scenario_caps(tmp_path: Path, monkeypatch):
    """The net-load reader runs INSIDE the scenario filter.

    Two databases differ only in the wind unit's scenario-specific ``existing``
    capacity. The reader (captured via ``demand_match_default_caps``) must see
    the scenario-correct value, proving it reads under the scenario filter.
    """
    captured: dict[str, float] = {}
    real = pp.demand_match_default_caps

    def _spy(inputs, timestep_keys, vre_penetration=1.0):
        captured["existing"] = inputs.vre["wind"].existing_cap
        return real(inputs, timestep_keys, vre_penetration)

    monkeypatch.setattr(pp, "demand_match_default_caps", _spy)

    url_lo = _build_db(tmp_path / "lo.sqlite", wind_existing=1.0)
    preprocess_representative_periods(
        url_lo, "base", _N_RP, _PERIOD_LENGTH, netload_clustering=True
    )
    assert captured["existing"] == 1.0

    url_hi = _build_db(tmp_path / "hi.sqlite", wind_existing=777.0)
    preprocess_representative_periods(
        url_hi, "base", _N_RP, _PERIOD_LENGTH, netload_clustering=True
    )
    assert captured["existing"] == 777.0


def test_vre_penetration_threads_through(tmp_path: Path, monkeypatch):
    """vre_penetration reaches demand_match_default_caps and moves the caps."""
    seen: list[float] = []
    caps_seen: list[dict] = []
    real = pp.demand_match_default_caps

    def _spy(inputs, timestep_keys, vre_penetration=1.0):
        seen.append(vre_penetration)
        result = real(inputs, timestep_keys, vre_penetration)
        caps_seen.append(dict(result))
        return result

    monkeypatch.setattr(pp, "demand_match_default_caps", _spy)

    url = _build_db(tmp_path / "pen.sqlite")
    preprocess_representative_periods(
        url, "base", _N_RP, _PERIOD_LENGTH,
        netload_clustering=True, vre_penetration=0.25,
    )
    url2 = _build_db(tmp_path / "pen2.sqlite")
    preprocess_representative_periods(
        url2, "base", _N_RP, _PERIOD_LENGTH,
        netload_clustering=True, vre_penetration=1.0,
    )

    assert seen == [0.25, 1.0]
    # Different penetration → different demand-match caps for the wind unit.
    assert not np.isclose(
        caps_seen[0]["wind"], caps_seen[1]["wind"]
    ), "vre_penetration did not change the demand-match caps"
