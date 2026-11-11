"""The input checks of #104 / #213 / #170, items 46 / 65 and audit bug 8 raise nothing on the shipped data.

Every scenario of ``tests/fixtures/tests.json``, of ``h2_trade_parity.json`` (representative periods), of
``stoch_two_period.json`` and ``stochastics.json`` (stochastic branch periods) and of each canonical JSON
(``flextool/schemas/canonical_databases``) is read as ``input_derivation.run`` reads it (the profile alternatives from the whole database, then the scenario
filter) and goes through ``validate_profile_links``, ``validate_basic_sanity``,
``validate_zero_conversion_coeff``, ``validate_period_keys`` (with the whole database's periods) and
``validate_timeline_timestep_duration`` (no solve), then ``warn_node_connections`` (#276).  ``case14_dc_power_flow.json``
and ``null_override_clears_map.json`` are read too (they hold nodes the #276 check names).  None may raise, and none
may warn except as listed in ``EXPECTED``: every zero ``conversion_flow_coeff`` of these files is tied by a user
constraint, and the maps keyed by constraint names (``constraint_flow_coeff``) are not period maps.  The expected
warnings: the deliberate negative ``existing`` of the anti_energy_plant example (scenario test_a_lot of
``tests.json`` and of the ``templates_examples.json`` generated from it); the demand node ``west`` without units in
the five scenarios that test the inflow alone; ``ARG_HBI`` of h2_trade_parity's no-carrier-storage scenario; the
bare nodes of null_override_clears_map (they hold the parameter under test); case14's fuel nodes commodity_gen1/2
(node_type balance, only drained by the generators).
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import spinedb_api as api

from flextool._resources import package_data_path

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = sorted(Path(str(package_data_path("schemas/canonical_databases"))).glob("*.json"))
# h2_trade_parity: representative periods; stoch_two_period / stochastics: stochastic branch periods.
SOURCES = [FLEXTOOL_ROOT / "tests" / "fixtures" / name
           for name in ("tests.json", "h2_trade_parity.json", "stoch_two_period.json", "stochastics.json",
                        "case14_dc_power_flow.json", "null_override_clears_map.json")] + CANONICAL


_ANTI_ENERGY = ("Warning: unit 'anti_energy_plant' existing = -50 (alternative 'anti_energy_plant'): a negative "
                "capacity makes an anti-energy unit (its flows run the other way). If you did not mean that, use 0 "
                "or more. Combined with a profile, min_load or invest / retire limits it can make the problem "
                "infeasible.\n")


def _empty_with_inflow(node: str, alt: str) -> str:
    return (f"Warning: node '{node}' (alternative '{alt}') has no units or connections; its inflow can only be met by "
            "the penalty slack (penalty_up / penalty_down).\n")


def _empty(*nodes: str) -> str:
    return "".join(f"Warning: node '{n}' has no units or connections and no inflow; it does nothing in this "
                   "scenario.\n" for n in nodes)


def _only_drained(*nodes: str) -> str:
    return "".join(f"Warning: node '{n}' (node_type balance) has no inflow and its units / connections only take "
                   "energy out of it; nothing brings energy in, so those flows can only be met by the penalty slack "
                   "(penalty_up). Add a unit or connection that supplies it, give it an inflow, or make it a "
                   "commodity node.\n" for n in nodes)


_WEST = {s: _empty_with_inflow("west", "west")
         for s in ("base", "base_weighted", "capacity_margin", "fullYear", "scale_to_peak_flow")}
# {source stem: {scenario: the whole printed text}}: the warnings the shipped data is meant to raise.
EXPECTED = {
    "tests": {"test_a_lot": _ANTI_ENERGY, **_WEST},
    "templates_examples": {"test_a_lot": _ANTI_ENERGY, **_WEST},
    "h2_trade_parity": {"scenario_test_6h_no_carrier_storage": _empty_with_inflow("ARG_HBI", "shared")},
    "null_override_clears_map": {"nulled_mixed": _empty("node_cleared", "node_keep", "node_partial"),
                                 "nulled_only": _empty("node_cleared"),
                                 "with_map": _empty("node_cleared", "node_keep", "node_partial")},
    "case14_dc_power_flow": {"dc_opf_test": _only_drained("commodity_gen1", "commodity_gen2")},
}


def _build(json_path: Path, work: Path) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.initialize_database import initialize_database
    from tests.db_utils import json_to_db

    sqlite = (work / f"{json_path.stem}.sqlite").resolve()
    if json_path.parent.name == "canonical_databases":
        initialize_database(str(json_path), str(sqlite))
        url = f"sqlite:///{sqlite}"
    else:
        url = json_to_db(json_path, sqlite)
    migrate_database(url)
    return url


def _check(url: str, scenario: str) -> str:
    from flextool.input_derivation import _validators as v
    from flextool.spinedb_backend import SpineDBBackend

    with api.DatabaseMapping(url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        alternatives = v.value_alternatives(db, "profile", "profile")
        whole = v.database_time_facts(db)
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config(scenario))
        backend = SpineDBBackend.__new__(SpineDBBackend)
        backend._db = db
        backend._api = api
        backend._scenario_name = scenario
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v.validate_profile_links(db, alternatives)
            v.validate_basic_sanity(backend)
            v.validate_zero_conversion_coeff(db)
            v.validate_timeline_timestep_duration(db)
            v.validate_period_keys(backend, whole)
            v.warn_node_connections(db)
    return out.getvalue()


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_shipped_scenarios_pass_the_input_checks(tmp_path_factory, source: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        url = _build(source, tmp_path_factory.mktemp(source.stem))
    with api.DatabaseMapping(url) as db:
        scenarios = sorted(s["name"] for s in db.find_scenarios())
    assert scenarios, source
    messages = {}
    for scenario in scenarios:
        text = _check(url, scenario)
        if text:
            messages[scenario] = text
    assert messages == EXPECTED.get(source.stem, {}), messages
