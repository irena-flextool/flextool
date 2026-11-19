"""The input checks of a life extension (``life_extension_cost`` / ``life_extension_years``).

``validate_basic_sanity`` stops the run on a life extension the model cannot price and warns about one it never
offers.  Errors: only one of the two parameters on an entity (an extension needs both: the cost it pays and the
years it is kept), ``life_extension_years`` ≤ 0 (charged for no years), ``life_extension_cost`` < 0 (paid to
be extended) and either on a ``reinvest_automatic`` entity (explicit or the default: its capacity is replaced at its
end of life, never extended).  Warning: both given on an entity without ``lifetime_technical`` (its capacity never
reaches an end of life).  Both given with a lifetime: silent.

Database: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (coal_plant: existing
500 MW, no lifetime_technical, no lifetime_method: the default reinvest_automatic); the cases set lifetime_method
reinvest_choice unless they check the reinvest_automatic error.  Messages are the expected text, written by hand.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests._support.coded_warnings import capture_coded
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db

PREFIX = "Input values out of range:\n  "
CHOICE = ["unit", "coal_plant", "lifetime_method", b64("reinvest_choice", "str")]


def _f(x: float) -> list:
    return b64(x, "float")


def _map(data: list) -> list:
    return b64({"index_type": "str", "data": data}, "map")


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_basic_sanity
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_basic_sanity(backend)


def _error(tmp_path_factory, alt: str, values: list) -> str:
    with pytest.raises(FlexToolConfigError) as err:
        _validate(*make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN))
    return str(err.value)


@pytest.mark.parametrize("alt, given, missing, value", [
    ("le_cost_only", "life_extension_cost", "life_extension_years", _f(100.0)),
    ("le_years_only", "life_extension_years", "life_extension_cost", _map([["p2020", 10.0]])),
])
def test_one_of_the_two_is_an_error(tmp_path_factory, alt, given, missing, value) -> None:
    msg = _error(tmp_path_factory, alt, [["unit", "coal_plant", given, value],
                                         ["unit", "coal_plant", "lifetime_technical", _f(10.0)], CHOICE])
    assert msg == PREFIX + (f"unit 'coal_plant' has a {given} but no {missing}: a life extension needs both. Give "
                            f"the {missing}, or remove the {given}.")


def test_zero_extension_years_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "le_years0", [["unit", "coal_plant", "life_extension_cost", _f(100.0)],
                                                 ["unit", "coal_plant", "life_extension_years", _f(0.0)],
                                                 ["unit", "coal_plant", "lifetime_technical", _f(10.0)], CHOICE])
    assert msg == PREFIX + (
        "unit 'coal_plant' life_extension_years = 0 (alternative 'le_years0'): life_extension_years must be above 0; "
        "with 0 the extension is charged for no years. Give the life_extension_years in years.")


def test_negative_extension_cost_names_the_period(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "le_cost_neg", [
        ["unit", "coal_plant", "life_extension_cost", _map([["p2020", 100.0], ["p2025", -5.0]])],
        ["unit", "coal_plant", "life_extension_years", _f(10.0)],
        ["unit", "coal_plant", "lifetime_technical", _f(10.0)], CHOICE])
    assert msg == PREFIX + (
        "unit 'coal_plant' life_extension_cost = -5 (alternative 'le_cost_neg', at p2025): life_extension_cost must "
        "be 0 or above (a negative cost pays the capacity to be extended).")


def test_extension_without_a_lifetime_warns(tmp_path_factory) -> None:
    """No existing capacity and no invest_method (so reinvest_choice needs no lifetime_technical): W-0225."""
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="le_lifeless", base_chain=PLANT_CHAIN, values=[
            ["unit", "coal_plant", "life_extension_cost", _f(100.0)],
            ["unit", "coal_plant", "life_extension_years", _f(10.0)], CHOICE,
            ["unit", "coal_plant", "existing", _f(0.0)]]))
    assert got.names("W-0225", cls="unit") == ["coal_plant"]


def test_extension_with_a_lifetime_is_silent(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="le_ok", base_chain=PLANT_CHAIN, values=[
            ["unit", "coal_plant", "life_extension_cost", _f(0.0)],
            ["unit", "coal_plant", "life_extension_years", _map([["p2020", 10.0]])],
            ["unit", "coal_plant", "lifetime_technical", _f(10.0)], CHOICE]))
    assert got.codes() == []


@pytest.mark.parametrize("alt, method", [("le_auto_default", None), ("le_auto_explicit", "reinvest_automatic")])
def test_extension_of_reinvest_automatic_capacity_is_an_error(tmp_path_factory, alt, method) -> None:
    values = [["unit", "coal_plant", "life_extension_cost", _f(100.0)],
              ["unit", "coal_plant", "life_extension_years", _f(10.0)],
              ["unit", "coal_plant", "lifetime_technical", _f(10.0)]]
    if method is not None:
        values.append(["unit", "coal_plant", "lifetime_method", b64(method, "str")])
    msg = _error(tmp_path_factory, alt, values)
    default = "" if method is not None else " (the default)"
    assert msg == PREFIX + (
        "unit 'coal_plant' has life_extension_cost and life_extension_years but lifetime_method reinvest_automatic"
        f"{default}: its capacity is replaced at the end of its life, so it is never extended. Set lifetime_method "
        "reinvest_choice or no_reinvestment to extend it, or remove the life extension.")
