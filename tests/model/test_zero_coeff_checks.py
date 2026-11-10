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

from tests.engine._unit_db import CHP_CHAIN, PLANT_CHAIN, b64, make_db, run


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


ZERO = b64(0.0, "float")
W1 = ("Warning: unit 'coal_plant' uses no input: all its outputs have conversion_flow_coeff 0, so its output to "
      "'west' is free up to max_flow_for_unconstrained_variables (1000000). Give an output a non-zero "
      "conversion_flow_coeff, or tie the flow with a user constraint.")
W2_HEAT = ("Warning: unit__outputNode (coal_chp, heat): conversion_flow_coeff is 0 and the output flow is in no user "
           "constraint (constraint_flow_coeff), flowGroup maximum or profile limit, so it has no capacity, ramp, "
           "minimum-load or fuel limit; only max_flow_for_unconstrained_variables (1000000) bounds it. (A flowGroup "
           "min_instant_flow / min_cumulative_flow bounds it only from below.)")
W3_HEAT = ("Warning: unit__outputNode (coal_chp, heat): capacity_max_coeff has no effect on an output with "
           "conversion_flow_coeff 0 (the flow is outside the capacity limits).")


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_zero_conversion_coeff
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_zero_conversion_coeff(backend._db)


def test_unit_with_all_outputs_zero_warns_once_per_run(tmp_path_factory, capsys) -> None:
    url, scen = make_db(tmp_path_factory, alt="b213_w1", base_chain=PLANT_CHAIN,
                        values=[["unit__outputNode", ["coal_plant", "west"], "conversion_flow_coeff", ZERO]])
    run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    assert out.count(W1) == 1, out
    assert "unit__outputNode (coal_plant, west): conversion_flow_coeff is 0" not in out, out


def test_untied_zero_output_warns(tmp_path_factory, capsys) -> None:
    """A constraint coefficient of 0 does not tie the flow."""
    _validate(*make_db(tmp_path_factory, alt="b213_w2", base_chain=CHP_CHAIN, values=[
        ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO],
        ["unit__outputNode", ["coal_chp", "heat"], "constraint_flow_coeff",
         b64({"index_type": "str", "data": [["coal_chp_fix", 0.0]]}, "map")]]))
    out = capsys.readouterr().out
    assert W2_HEAT in out and "uses no input" not in out, out


def test_zero_output_tied_by_a_user_constraint_is_silent(tmp_path_factory, capsys) -> None:
    _validate(*make_db(tmp_path_factory, alt="b213_tied", base_chain=CHP_CHAIN, values=[
        ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO]]))
    assert "Warning" not in capsys.readouterr().out


def test_capacity_coeff_on_a_zero_output_warns(tmp_path_factory, capsys) -> None:
    _validate(*make_db(tmp_path_factory, alt="b213_w3", base_chain=CHP_CHAIN, values=[
        ["unit__outputNode", ["coal_chp", "heat"], "conversion_flow_coeff", ZERO],
        ["unit__outputNode", ["coal_chp", "heat"], "capacity_max_coeff", b64(0.5, "float")]]))
    out = capsys.readouterr().out
    assert W3_HEAT in out and "no user constraint" not in out, out


def test_zero_output_with_an_upper_limit_profile_is_silent(tmp_path_factory, capsys) -> None:
    _validate(*make_db(tmp_path_factory, alt="b213_profile", base_chain=PLANT_CHAIN, values=[
        ["unit__outputNode", ["coal_plant", "west"], "conversion_flow_coeff", ZERO],
        ["profile", "cap", "profile", b64(0.5, "float")]],
        entities=[["profile", "cap", None], ["unit__node__profile", ["coal_plant", "west", "cap"], None]]))
    assert "Warning" not in capsys.readouterr().out
