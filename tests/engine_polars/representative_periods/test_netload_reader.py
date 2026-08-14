"""DB-reader tests for ``read_netload_inputs`` (invariant #3: build from schema).

Builds a minimal model on top of a fresh schema DB (``initialize_database`` off
``schemas/spinedb_schema.json`` — never a checked-in .sqlite), then asserts the
reader returns the right aggregation units, VRE set (including the default
``upper_limit`` case — item (e)), demand series, profiles, and durations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, Map, import_data

from flextool.representative_periods.netload_inputs import read_netload_inputs
from flextool.update_flextool.initialize_database import initialize_database

_SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "flextool"
    / "schemas"
    / "spinedb_schema.json"
)

_KEYS = ["t01", "t02", "t03", "t04"]


@pytest.fixture
def netload_db(tmp_path: Path) -> str:
    """A minimal two-node model with a demand node and two VRE units.

    * ``n_load`` — a demand node (time-varying negative inflow).
    * ``n_vre``  — the VRE injection node.
    * ``wind``   — a VRE unit whose ``unit__node__profile`` sets NO
      ``profile_method`` (schema-default ``upper_limit`` → must read as VRE);
      investable (``invest_no_limit``), existing 3.0, virtual_unitsize 1.5.
    * ``solar``  — a VRE unit with an EXPLICIT ``profile_method == upper_limit``;
      non-investable (default ``not_allowed``), existing 2.0.
    * ``gas``    — a NON-VRE unit whose profile arc is ``profile_method ==
      fixed`` → must be EXCLUDED from the VRE set.
    """
    db_path = tmp_path / "netload.sqlite"
    initialize_database(str(_SCHEMA), str(db_path))
    url = f"sqlite:///{db_path}"

    demand = Map(_KEYS, [-10.0, -40.0, -20.0, -30.0])
    wind_pf = Map(_KEYS, [0.2, 0.4, 0.6, 0.8])
    solar_pf = Map(_KEYS, [0.1, 0.5, 0.9, 0.3])
    gas_pf = Map(_KEYS, [1.0, 1.0, 1.0, 1.0])
    timeline = Map(_KEYS, [1.0, 1.0, 1.0, 1.0])

    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            alternatives=[["base", ""]],
            entities=[
                ["node", ["n_load"], None],
                ["node", ["n_vre"], None],
                ["profile", ["wind_profile"], None],
                ["profile", ["solar_profile"], None],
                ["profile", ["gas_profile"], None],
                ["unit", ["wind"], None],
                ["unit", ["solar"], None],
                ["unit", ["gas"], None],
                ["timeline", ["main"], None],
                ["unit__outputNode", ["wind", "n_vre"], None],
                ["unit__outputNode", ["solar", "n_vre"], None],
                ["unit__outputNode", ["gas", "n_vre"], None],
                ["unit__node__profile", ["wind", "n_vre", "wind_profile"], None],
                ["unit__node__profile", ["solar", "n_vre", "solar_profile"], None],
                ["unit__node__profile", ["gas", "n_vre", "gas_profile"], None],
            ],
            parameter_values=[
                ["node", "n_load", "inflow", demand, "base"],
                ["profile", "wind_profile", "profile", wind_pf, "base"],
                ["profile", "solar_profile", "profile", solar_pf, "base"],
                ["profile", "gas_profile", "profile", gas_pf, "base"],
                ["timeline", "main", "timestep_duration", timeline, "base"],
                # wind: NO profile_method (schema default upper_limit → VRE).
                ["unit", "wind", "existing", 3.0, "base"],
                ["unit", "wind", "virtual_unitsize", 1.5, "base"],
                ["unit", "wind", "invest_method", "invest_no_limit", "base"],
                # solar: explicit upper_limit, non-investable (default), existing.
                [
                    "unit__node__profile",
                    ["solar", "n_vre", "solar_profile"],
                    "profile_method",
                    "upper_limit",
                    "base",
                ],
                ["unit", "solar", "existing", 2.0, "base"],
                # gas: profile_method fixed → NOT VRE.
                [
                    "unit__node__profile",
                    ["gas", "n_vre", "gas_profile"],
                    "profile_method",
                    "fixed",
                    "base",
                ],
            ],
        )
        assert not errors, errors
        db.commit_session("minimal net-load fixture")
    return url


def test_reader_node_granularity_fallback(netload_db: str):
    """No group flag exists yet → per-node granularity, each node its own unit."""
    with DatabaseMapping(netload_db) as db:
        inputs = read_netload_inputs(db)

    assert inputs.granularity == "node"
    # Every node is its own aggregation unit (sorted).
    assert inputs.units_by_group == {
        "n_load": ["n_load"],
        "n_vre": ["n_vre"],
    }


def test_reader_vre_set_and_default_upper_limit(netload_db: str):
    """wind (default upper_limit) and solar (explicit) are VRE; gas is not."""
    with DatabaseMapping(netload_db) as db:
        inputs = read_netload_inputs(db)

    assert set(inputs.vre) == {"wind", "solar"}  # gas (fixed) excluded

    wind = inputs.vre["wind"]
    assert wind.node == "n_vre"
    assert wind.profile == "wind_profile"
    assert wind.existing_cap == 3.0
    assert wind.unitsize == 1.5
    assert wind.investable is True  # invest_no_limit

    solar = inputs.vre["solar"]
    assert solar.node == "n_vre"
    assert solar.profile == "solar_profile"
    assert solar.existing_cap == 2.0
    assert solar.unitsize == 1.0  # virtual_unitsize unset → 1.0 placeholder
    assert solar.investable is False  # default not_allowed


def test_reader_demand_and_profiles(netload_db: str):
    with DatabaseMapping(netload_db) as db:
        inputs = read_netload_inputs(db)

    # Demand: time-varying inflow on n_load, no scalar demand.
    assert inputs.demand_ts["n_load"] == [
        ("t01", -10.0),
        ("t02", -40.0),
        ("t03", -20.0),
        ("t04", -30.0),
    ]
    assert inputs.demand_scalar == {}

    # Profiles: all three time-varying series present (VRE + gas).
    assert set(inputs.profiles) == {"wind_profile", "solar_profile", "gas_profile"}
    assert inputs.profiles["wind_profile"] == [
        ("t01", 0.2),
        ("t02", 0.4),
        ("t03", 0.6),
        ("t04", 0.8),
    ]

    # Step durations from the single timeline.
    assert inputs.step_durations == {k: 1.0 for k in _KEYS}


def test_reader_end_to_end_builds_matrix(netload_db: str):
    """Reader output drives the pure builders without a shape error."""
    from flextool.representative_periods.netload import (
        build_group_capacities,
        build_netload_matrix,
        demand_match_default_caps,
    )

    with DatabaseMapping(netload_db) as db:
        inputs = read_netload_inputs(db)

    default_caps = demand_match_default_caps(inputs, _KEYS)
    # Only the investable VRE unit (wind) is sized by the demand-match default.
    assert set(default_caps) == {"wind"}

    caps = build_group_capacities(inputs, default_caps, solved_caps=None)
    assert caps["solar"] == 2.0  # non-investable → existing
    assert caps["wind"] == default_caps["wind"]

    C, n_base, names = build_netload_matrix(inputs, caps, _KEYS, period_length=2)
    # 2 aggregation units (n_load, n_vre) × period_length 2 = 4 rows;
    # 4 timesteps / period_length 2 = 2 base periods.
    assert names == ["n_load", "n_vre"]
    assert n_base == 2
    assert C.shape == (4, 2)


def test_reader_array_inflow_skipped(tmp_path, capsys):
    """An Array-typed inflow (keyless value list) is skipped, not crashed on.

    ``params_to_dict`` returns an ``Array`` as a bare list of values (no keys),
    which the downstream ``for key, value in ts`` unpack cannot consume. The
    reader must detect the non-``(key, value)`` shape and skip it with a warning.
    """
    db_path = tmp_path / "array_inflow.sqlite"
    initialize_database(str(_SCHEMA), str(db_path))
    url = f"sqlite:///{db_path}"

    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            alternatives=[["base", ""]],
            entities=[
                ["node", ["n_map"], None],
                ["node", ["n_array"], None],
            ],
            parameter_values=[
                # Map-typed inflow → normal (key, value) series.
                ["node", "n_map", "inflow", Map(_KEYS, [-1.0, -2.0, -3.0, -4.0]), "base"],
                # Array-typed inflow → keyless value list → must be skipped.
                ["node", "n_array", "inflow", Array([-1.0, -2.0, -3.0, -4.0]), "base"],
            ],
        )
        assert not errors, errors
        db.commit_session("array inflow fixture")

    with DatabaseMapping(url) as db:
        inputs = read_netload_inputs(db)  # must not raise

    out = capsys.readouterr().out
    assert "skipping inflow of node 'n_array'" in out
    # The Map series survives; the Array node is absent from both maps.
    assert "n_map" in inputs.demand_ts
    assert "n_array" not in inputs.demand_ts
    assert "n_array" not in inputs.demand_scalar
