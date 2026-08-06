"""Regression: model ``max_flow_for_unconstrained_variables`` override is
honored by the invest-ceiling derived params.

The model parameter ``max_flow_for_unconstrained_variables`` (default
1,000,000) sets the invested-capacity ceiling for every
``invest_no_limit`` entity: the ceiling is ``existing + this value``,
enforced through ``p_entity_max_units`` (→ ``maxInvest_var_bound_n``).

The derived-params code used to look the parameter up under a spurious
``p_``-prefixed name (``p_max_flow_for_unconstrained_variables``).  The
DB / schema stores it under its raw name ``max_flow_for_unconstrained_
variables`` (no prefix — that prefix is only the CSV-emit alias), so
``source.parameter(...)`` never matched and the code silently fell back
to the hard-coded 1e6 default.  Result: every ``invest_no_limit`` entity
was capped at 1e6 regardless of the DB override.

This test builds a DB from the JSON single-source-of-truth fixture (never
a checked-in ``.sqlite``), overrides the model parameter to a value well
above 1e6, and asserts the resulting ``p_entity_max_units`` ceiling for
an ``invest_no_limit`` unit reflects the override rather than being pinned
at the 1e6 default.  It FAILS against the old ``p_``-prefixed lookup
(ceiling pinned at ``(existing + 1e6) / unitsize``) and PASSES after the
fix (ceiling ``(existing + override) / unitsize``).
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from flextool.engine_polars import _derived_params as dp
from flextool.engine_polars._spinedb_reader import SpineDbReader

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"

# Distinctive override, well above the 1e6 default so the two code paths
# land on clearly separated ceilings.
OVERRIDE_VALUE = 7_500_000.0
# From the ``tests.json`` fixture: ``wind_plant`` is the invest_no_limit
# unit, with ``existing = 1000`` and a default unitsize of 1000.
WIND_EXISTING = 1000.0
UNITSIZE = dp.UNITSIZE_DEFAULT  # 1000.0
ACTIVE_SOLVE = "invest_1year_5weeks"  # invest_periods == ['p2020']
INVEST_PERIOD = "p2020"

_OVERRIDE_ALT = "mfuv_override"
_SCENARIO = "mfuv_no_limit_test"
# Mirror the known-good ``5weeks_invest_fullYear_dispatch_coal_wind``
# alternative chain so entity/solve activation is identical, then append
# the override alternative as the highest-priority layer.
_CHAIN = [
    "init", "west", "coal", "coal_invest", "wind", "wind_invest",
    "fullYear", "5weeks", "5weeks_only_invest", _OVERRIDE_ALT,
]


def _b64_float(x: float) -> list:
    """Spine import-format packed scalar float value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), "float"]


@pytest.fixture(scope="module")
def override_source(tmp_path_factory: pytest.TempPathFactory) -> SpineDbReader:
    """Build a DB from ``tests.json`` with an additive override alternative
    that sets ``model.max_flow_for_unconstrained_variables`` above 1e6, and
    return a ``SpineDbReader`` bound to the override scenario.
    """
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([_OVERRIDE_ALT, ""])
    data["parameter_values"].append([
        "model", "flexTool", "max_flow_for_unconstrained_variables",
        _b64_float(OVERRIDE_VALUE), _OVERRIDE_ALT,
    ])
    data["scenarios"].append([_SCENARIO, False, ""])
    for alt, before in zip(_CHAIN, _CHAIN[1:] + [None]):
        data["scenario_alternatives"].append([_SCENARIO, alt, before])

    work = tmp_path_factory.mktemp("mfuv_override_db")
    json_path = work / "mfuv.json"
    json_path.write_text(json.dumps(data))
    db_path = work / "mfuv.sqlite"
    url = json_to_db(json_path, db_path)
    return SpineDbReader(url, _SCENARIO)


def test_override_is_readable_under_raw_db_name(override_source: SpineDbReader) -> None:
    """The model param is stored under its raw name, NOT the ``p_``-prefixed
    CSV-emit alias — pins the root cause of the bug.
    """
    # The (wrongly) prefixed name the buggy lookup used never matches.
    assert dp._try_param(
        override_source, "model", "p_max_flow_for_unconstrained_variables",
    ) is None
    # The correct raw name carries the override.
    got = dp._try_param(
        override_source, "model", "max_flow_for_unconstrained_variables",
    )
    assert got is not None and got.height == 1
    assert float(got["value"][0]) == pytest.approx(OVERRIDE_VALUE)


def test_invest_no_limit_ceiling_honors_override(
    override_source: SpineDbReader,
) -> None:
    """``p_entity_max_units`` for the ``invest_no_limit`` unit reflects the
    DB override, not the hard-coded 1e6 default.

    ceiling(capacity) = existing + max_flow_for_unconstrained_variables
    p_entity_max_units = ceiling(capacity) / unitsize
    """
    ed_invest = dp.ed_invest_set_from_source(override_source, ACTIVE_SOLVE)
    assert ed_invest is not None
    ed_pairs = set(ed_invest.iter_rows())
    assert ("wind_plant", INVEST_PERIOD) in ed_pairs

    pmu = dp.p_entity_max_units_from_source(
        override_source, ed_invest, active_solve=ACTIVE_SOLVE,
        workdir=None, provider=None,
    )
    assert pmu is not None
    ceiling_by_e = {
        (e, d): v for e, d, v in pmu.frame.iter_rows()
    }
    wind_units = ceiling_by_e[("wind_plant", INVEST_PERIOD)]

    expected_units = (WIND_EXISTING + OVERRIDE_VALUE) / UNITSIZE
    buggy_units = (WIND_EXISTING + 1_000_000.0) / UNITSIZE

    # The fixed lookup lands on the override-derived ceiling ...
    assert wind_units == pytest.approx(expected_units)
    # ... which is unmistakably above the 1e6-default ceiling the buggy
    # ``p_``-prefixed lookup would silently produce.
    assert wind_units > buggy_units * 2
