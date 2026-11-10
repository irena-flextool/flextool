"""The input checks of #104 / #213 / #170 raise nothing on the shipped data.

Every scenario of ``tests/fixtures/tests.json`` and of each canonical JSON (``flextool/schemas/canonical_databases``)
is read as ``input_derivation.run`` reads it (the profile alternatives from the whole database, then the scenario
filter) and goes through ``validate_profile_links``, ``validate_basic_sanity`` and
``validate_zero_conversion_coeff`` (no solve).  None may raise, and none may warn: every zero
``conversion_flow_coeff`` of these files is tied by a user constraint, and the negative ``existing`` of the
anti_energy_plant example is not reported.
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
SOURCES = [FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json", *CANONICAL]


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
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config(scenario))
        backend = SpineDBBackend.__new__(SpineDBBackend)
        backend._db = db
        backend._api = api
        backend._scenario_name = scenario
        _periods, timelines = v._scenario_time_vocab(backend)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v.validate_profile_links(db, alternatives)
            v.validate_basic_sanity(db, {step for steps in timelines.values() for step in steps})
            v.validate_zero_conversion_coeff(db)
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
    assert messages == {}, messages
