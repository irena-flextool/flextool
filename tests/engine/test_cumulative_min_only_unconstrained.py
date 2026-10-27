"""Regression: a ``cumulative_limits`` entity with only ``cumulative_min_capacity`` (no
``cumulative_max_capacity`` at all) is unconstrained above, not pinned to ``v_invest = 0``.

``p_entity_max_units_from_source`` (``flextool/engine/preprocess/params/_derived.py``) computes a
per-(entity, period) ceiling on invested units: for a ``cumulative_limits`` entity it used the
authored ``cumulative_max_capacity`` directly as the ceiling, defaulting to 0 when the entity
authors no value at all.  A ``cumulative_limits`` entity that only sets a floor
(``cumulative_min_capacity``, no ceiling) therefore got ``p_entity_max_units = 0`` for every
period — forbidding ANY investment, including the investment needed to reach its own floor.  The
real binding limit (if any) is still enforced separately by
``{max,min}Cumulative_capacity_{p,n}`` (``_cumulative_invest.py``); this ceiling only bounds the
variable itself and must not invent a restriction nobody authored.

This test builds a DB from the JSON single-source-of-truth fixture (never a checked-in
``.sqlite``), overrides ``wind_plant`` to ``invest_method = cumulative_limits`` with only
``cumulative_min_capacity`` set, and asserts the resulting ``p_entity_max_units`` ceiling is the
unconstrained one (``existing + max_flow_for_unconstrained_variables``), not 0.  It FAILS against
the old code (ceiling pinned at 0, so ``wind_plant`` is missing from the Param entirely — ``mu >
0.0`` never holds) and PASSES after the fix.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from flextool.engine.preprocess.params import _derived as dp
from flextool.engine.input._spinedb_reader import SpineDbReader

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"

# From tests.json / test_max_flow_unconstrained_override.py: wind_plant has existing = 1000 and
# the default unitsize (1000.0).
WIND_EXISTING = 1000.0
UNITSIZE = dp.UNITSIZE_DEFAULT  # 1000.0
MIN_CAPACITY = 1200.0  # above existing, so it is a real floor
ACTIVE_SOLVE = "invest_1year_5weeks"  # invest_periods == ['p2020']
INVEST_PERIOD = "p2020"

_OVERRIDE_ALT = "cum_min_only_override"
_SCENARIO = "cum_min_only_test"
# Same chain as test_max_flow_unconstrained_override.py (known-good entity/solve activation),
# with our override appended as the highest-priority layer.
_CHAIN = [
    "init", "west", "coal", "coal_invest", "wind", "wind_invest",
    "fullYear", "5weeks", "5weeks_only_invest", _OVERRIDE_ALT,
]


def _b64_float(x: float) -> list:
    return [base64.b64encode(json.dumps(x).encode()).decode(), "float"]


def _b64_str(x: str) -> list:
    return [base64.b64encode(json.dumps(x).encode()).decode(), "str"]


@pytest.fixture(scope="module")
def min_only_source(tmp_path_factory: pytest.TempPathFactory) -> SpineDbReader:
    """``tests.json`` + an override alternative: ``wind_plant`` becomes a ``cumulative_limits``
    entity with only ``cumulative_min_capacity`` (no ``cumulative_max_capacity``)."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([_OVERRIDE_ALT, ""])
    data["parameter_values"].append([
        "unit", "wind_plant", "invest_method", _b64_str("cumulative_limits"), _OVERRIDE_ALT,
    ])
    data["parameter_values"].append([
        "unit", "wind_plant", "cumulative_min_capacity", _b64_float(MIN_CAPACITY), _OVERRIDE_ALT,
    ])
    data["scenarios"].append([_SCENARIO, False, ""])
    for alt, before in zip(_CHAIN, _CHAIN[1:] + [None]):
        data["scenario_alternatives"].append([_SCENARIO, alt, before])

    work = tmp_path_factory.mktemp("cum_min_only_db")
    json_path = work / "cum_min_only.json"
    json_path.write_text(json.dumps(data))
    db_path = work / "cum_min_only.sqlite"
    url = json_to_db(json_path, db_path)
    return SpineDbReader(url, _SCENARIO)


def test_wind_plant_is_cumulative_with_no_max_authored(min_only_source: SpineDbReader) -> None:
    """Root-cause pin: the override lands, and no ``cumulative_max_capacity`` is set anywhere."""
    assert dp._try_param(min_only_source, "unit", "cumulative_max_capacity") is None
    got = dp._try_param(min_only_source, "unit", "cumulative_min_capacity")
    assert got is not None and got.height == 1
    assert float(got["value"][0]) == pytest.approx(MIN_CAPACITY)


def test_cumulative_min_only_ceiling_is_unconstrained(min_only_source: SpineDbReader) -> None:
    """``p_entity_max_units`` for a cumulative_limits entity with no ``cumulative_max_capacity``
    is ``(existing + max_flow_for_unconstrained_variables) / unitsize``, not 0.
    """
    ed_invest = dp.ed_invest_set_from_source(min_only_source, ACTIVE_SOLVE)
    assert ed_invest is not None
    ed_pairs = set(ed_invest.iter_rows())
    assert ("wind_plant", INVEST_PERIOD) in ed_pairs

    pmu = dp.p_entity_max_units_from_source(
        min_only_source, ed_invest, active_solve=ACTIVE_SOLVE,
        workdir=None, provider=None,
    )
    assert pmu is not None
    ceiling_by_e = {(e, d): v for e, d, v in pmu.frame.iter_rows()}

    # Before the fix: wind_plant is absent entirely (mu == 0 is filtered out by "mu > 0.0").
    assert ("wind_plant", INVEST_PERIOD) in ceiling_by_e, (
        "wind_plant must still get a p_entity_max_units row: a cumulative_limits entity with "
        "only a floor must not be pinned to v_invest <= 0"
    )
    wind_units = ceiling_by_e[("wind_plant", INVEST_PERIOD)]
    expected_units = (WIND_EXISTING + 1_000_000.0) / UNITSIZE  # the 1e6 default p_unc
    assert wind_units == pytest.approx(expected_units)
    # Unmistakably above the authored floor converted to units — proves it is the unconstrained
    # ceiling, not the (small) floor value misread as a ceiling.
    assert wind_units > (MIN_CAPACITY / UNITSIZE) * 10
