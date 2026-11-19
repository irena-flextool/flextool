"""Unit flows with ``conversion_flow_coeff = 0`` that nothing limits (#213).

A zero coefficient takes the flow out of the conversion equation and of every capacity, ramp and minimum-load
limit; only ``max_flow_for_unconstrained_variables`` (default 1 000 000) bounds it.  ``coal`` with
``conversion_flow_coeff 0`` on ``coal_plant``'s only output solved to free power without a word.
``validate_zero_conversion_coeff`` (``input_derivation.run``, once per run) now warns; a flow tied by a user
constraint (non-zero ``constraint_flow_coeff``), a flowGroup maximum or an upper_limit / fixed profile is silent.

Databases: ``tests/fixtures/tests.json`` + one override alternative.  ``coal_chp`` (chain ``init, west, coal_chp,
heat``) has outputs ``west`` and ``heat``, both in user constraint ``coal_chp_fix`` (``constraint_flow_coeff`` in
alternative ``coal_chp``).
"""
from __future__ import annotations

import pytest

from tests._support.coded_capture import capture_coded
from tests.engine._unit_db import CHP_CHAIN, PLANT_CHAIN, b64, make_db, run


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


ZERO = b64(0.0, "float")
LIMIT = "up to max_flow_for_unconstrained_variables (1000000)"


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_zero_conversion_coeff
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_zero_conversion_coeff(backend._db)


def test_unit_with_all_outputs_zero_warns_once_per_run(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b213_w1", base_chain=PLANT_CHAIN,
                        values=[["unit__outputNode", ["coal_plant", "west"], "conversion_flow_coeff", ZERO]])
    with capture_coded() as got:
        run(tmp_path_factory, url, scen)
    assert [n for call in got.calls if call.code == "W-0216" for n in call.names] == ["coal_plant"]
    assert got.detail("W-0216", "coal_plant") == f"free outputs 'west', {LIMIT}"
    assert "coal_plant, west" not in got.names("W-0217")


def test_untied_zero_output_warns(tmp_path_factory) -> None:
    """A constraint coefficient of 0 does not tie the flow."""
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b213_w2", base_chain=CHP_CHAIN, values=[
            ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO],
            ["unit__outputNode", ["coal_chp", "heat"], "constraint_flow_coeff",
             b64({"index_type": "str", "data": [["coal_chp_fix", 0.0]]}, "map")]]))
    assert got.names("W-0217", side="output") == ["coal_chp, heat"]
    assert got.detail("W-0217", "coal_chp, heat") == f"a free source on the node, {LIMIT}"
    assert "W-0216" not in got.codes()


def test_untied_zero_input_warns(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b213_w2_in", base_chain=PLANT_CHAIN, values=[
            ["unit__inputNode", ["coal_plant", "coal_market"], "conversion_flow_coeff", ZERO]]))
    assert got.names("W-0217", side="input") == ["coal_plant, coal_market"]
    assert got.detail("W-0217", "coal_plant, coal_market") == f"a dump load on the node, {LIMIT}"


def test_zero_output_tied_by_a_user_constraint_is_silent(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b213_tied", base_chain=CHP_CHAIN, values=[
            ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO]]))
    assert got.codes() == []


def test_capacity_coeff_on_a_zero_output_warns(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b213_w3", base_chain=CHP_CHAIN, values=[
            ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO],
            ["unit__outputNode", ["coal_chp", "heat"], "capacity_max_coeff", b64(0.5, "float")]]))
    assert got.names("W-0218") == ["coal_chp, heat"]
    assert got.detail("W-0218", "coal_chp, heat") == "capacity_max_coeff"
    assert "W-0217" not in got.codes()


def test_zero_output_with_an_upper_limit_profile_is_silent(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b213_profile", base_chain=PLANT_CHAIN, values=[
            ["unit__outputNode", ["coal_plant", "west"], "conversion_flow_coeff", ZERO],
            ["profile", "cap", "profile", b64(0.5, "float")]],
            entities=[["profile", "cap", None], ["unit__node__profile", ["coal_plant", "west", "cap"], None]]))
    assert got.codes() == []
