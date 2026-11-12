"""Profile links without profile data (#104).

``unit__node__profile.profile_method`` defaults to ``upper_limit`` and a profile without a value is 0, so a link to
a profile that has no value in the scenario held the flow at 0 without a word (``coal`` with such a link on
``coal_plant`` solved to the coal-off objective).  ``validate_profile_links`` (``input_derivation.run``, before the
parameter read) now stops the run; ``lower_limit`` / ``none`` stay allowed, and a ``node__profile`` /
``connection__profile`` link without a ``profile_method`` (default ``none``: not applied) gets a warning.

Databases: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (base objective
1 144 037 750, pinned in the existing suite).  ``wind_profile`` has its value only in alternative ``wind``, which
this chain does not include.
"""
from __future__ import annotations

import pytest
import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db, run

COAL_OBJECTIVE = 1_144_037_750.0


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _db(tmp_path_factory, alt: str, values: list, entities: list) -> tuple[str, str]:
    return make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN, entities=entities)


def _validate(url: str, scenario: str) -> None:
    """The check as ``input_derivation.run`` calls it: the profile alternatives from the whole database."""
    from flextool.input_derivation._validators import validate_profile_links, value_alternatives
    from flextool.spinedb_backend import SpineDBBackend

    with api.DatabaseMapping(url) as whole:
        alternatives = value_alternatives(whole, "profile", "profile")
    with SpineDBBackend(url, scenario) as backend:
        validate_profile_links(backend._db, alternatives)


GHOST = [["profile", "ghost", None], ["unit__node__profile", ["coal_plant", "west", "ghost"], None]]


def test_link_to_a_profile_without_data_stops_the_run(tmp_path_factory) -> None:
    url, scen = _db(tmp_path_factory, "b104_ghost", [], GHOST)
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == (
        "Profile links without profile data:\n  unit__node__profile (coal_plant, west, ghost): profile 'ghost' has "
        "no value in any alternative, so it holds the flow at 0 (profile_method is upper_limit (the default)). "
        "Give the profile a value in an alternative of the scenario, set profile_method lower_limit or none, or "
        "remove the link.")


def test_error_names_the_alternative_that_holds_the_profile(tmp_path_factory) -> None:
    url, scen = _db(tmp_path_factory, "b104_wind", [],
                    [["unit__node__profile", ["coal_plant", "west", "wind_profile"], None]])
    with pytest.raises(FlexToolConfigError) as err:
        _validate(url, scen)
    msg = str(err.value)
    assert ("unit__node__profile (coal_plant, west, wind_profile): profile 'wind_profile' has no value in this "
            "scenario: its value is only in alternative 'wind', which the scenario does not include, so it holds "
            "the flow at 0") in msg, msg
    assert "Add alternative 'wind' to the scenario" in msg, msg


def test_fixed_method_without_data_is_an_error(tmp_path_factory) -> None:
    url, scen = _db(tmp_path_factory, "b104_fixed", [
        ["unit__node__profile", ["coal_plant", "west", "ghost"], "profile_method", b64("fixed", "str")]], GHOST)
    with pytest.raises(FlexToolConfigError) as err:
        _validate(url, scen)
    assert "holds the flow at 0 (profile_method is fixed)" in str(err.value), str(err.value)


def test_lower_limit_without_data_runs_unchanged(tmp_path_factory, capsys) -> None:
    """A floor of 0 does nothing: the coal objective is the base one."""
    steps = run(tmp_path_factory, *_db(tmp_path_factory, "b104_lower", [
        ["unit__node__profile", ["coal_plant", "west", "ghost"], "profile_method", b64("lower_limit", "str")]],
        GHOST))
    assert list(steps.values())[-1].solution.obj == pytest.approx(COAL_OBJECTIVE, rel=1e-9)
    assert "unit__node__profile" not in capsys.readouterr().out


def test_node_profile_without_method_warns(tmp_path_factory, capsys) -> None:
    _validate(*_db(tmp_path_factory, "b104_node", [],
                   [["profile", "ghost", None], ["node__profile", ["west", "ghost"], None]]))
    assert ("Warning: node__profile (west, ghost) has no profile_method, so the profile is not applied (the "
            "default is none). Set profile_method upper_limit, lower_limit or fixed, or remove the link."
            ) in capsys.readouterr().out


def test_explicit_none_and_profiles_with_data_are_silent(tmp_path_factory, capsys) -> None:
    _validate(*_db(tmp_path_factory, "b104_quiet", [
        ["node__profile", ["west", "ghost"], "profile_method", b64("none", "str")],
        ["profile", "ghost", "profile", b64(0.5, "float")]],
        [["profile", "ghost", None], ["node__profile", ["west", "ghost"], None],
         ["unit__node__profile", ["coal_plant", "west", "ghost"], None]]))
    assert "Warning" not in capsys.readouterr().out
