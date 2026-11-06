"""Δ.4 — second-wave Direct Param migration parity tests.

Δ.4 extends ``apply_direct_params`` (formerly ``first_wave_overrides``)
with helpers for scalar Direct Params previously read by
``flex_data.py``'s CSV loaders only:

* ``p_state_self_discharge`` — ``node.self_discharge_loss``
* ``p_state_start`` — ``node.storage_state_start``
* ``p_connection_susceptance`` — ``connection.susceptance``
* ``p_commodity_unitsize`` — ``commodity.unitsize``

``p_min_load`` (``unit.min_load``) was formerly a Δ.4 scalar Direct Param
here, but is now produced as a shape-correct (p, d, t) Param by
``_derived.apply_derived_b`` so a period/time Map isn't summed in the
minFlow_minload constraint (see ``specs/fix_map_reading.md`` §5).  Its
coverage moved to ``test_min_load_pdt.py`` and the derived-producer unit test
below.

Each helper keeps "explicit rows only" semantics via
:meth:`InputSource.parameter_explicit`.

The tests below are deliberately narrow: they exercise the helpers
on representative fixtures using :class:`InMemoryReader`, and check on
real fixtures that a rebuild from the post-solve provider (the
production loader call) gives the same fields as the FlexData the
cascade itself built and solved.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine import InMemoryReader
from flextool.engine.preprocess.params import _direct as dp
from polar_high import Param
from tests._support.fields import field_diff
from tests._support.load import load_scenario

# ---------------------------------------------------------------------------
# In-memory unit tests — exercise the helpers without sqlite.

def _node_reader(name: str, value: float, parameter_name: str
                  ) -> InMemoryReader:
    return InMemoryReader(
        entities={"node": pl.DataFrame({"name": [name]})},
        parameters={
            ("node", parameter_name): pl.DataFrame({
                "name": [name], "value": [value],
            }),
        },
    )


# The solve's active (d, t): self_discharge_loss goes through the _param_shapes resolver (#128), which produces
# nothing without an active period.
_DT = pl.DataFrame({"d": ["p1", "p1"], "t": ["t1", "t2"]})


def test_p_state_self_discharge_inmemory_explicit():
    """Explicit scalar ``self_discharge_loss`` row → ``Param(("n",), [n, value])`` (a scalar keeps the entity
    axis only; the model broadcasts it over (d, t)); no active period → None."""
    src = _node_reader("nA", 0.05, "self_discharge_loss")
    assert dp.p_state_self_discharge_from_source(src) is None
    p = dp.p_state_self_discharge_from_source(src, period_filter=_DT)
    assert isinstance(p, Param)
    assert p.dims == ("n",)
    assert p.frame.sort("n")["value"].to_list() == pytest.approx([0.05])


def test_p_state_self_discharge_inmemory_no_rows():
    """Empty parameter frame → helper returns None."""
    src = InMemoryReader(
        entities={"node": pl.DataFrame({"name": ["nA"]})},
        parameters={
            ("node", "self_discharge_loss"): pl.DataFrame(
                schema={"name": pl.Utf8, "value": pl.Float64},
            ),
        },
    )
    assert dp.p_state_self_discharge_from_source(src, period_filter=_DT) is None


def test_p_state_start_inmemory_explicit():
    src = _node_reader("battery", 0.5, "storage_state_start")
    p = dp.p_state_start_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("n",)
    assert p.frame.sort("n")["value"].to_list() == pytest.approx([0.5])


def test_p_min_load_pdt_inmemory_explicit_scalar():
    """``p_min_load`` is now a derived (p, d, t) Param (not a Δ.4 scalar).

    A scalar min_load broadcasts to the full (d, t) grid — byte-identical
    floor to the old (p,) broadcast.  See ``specs/fix_map_reading.md`` §5.
    """
    from flextool.engine.preprocess.params import _derived as drv

    dt = pl.DataFrame({
        "d": ["2025", "2025", "2030", "2030"],
        "t": ["t01", "t02", "t01", "t02"],
    })
    src = InMemoryReader(
        entities={"unit": pl.DataFrame({"name": ["coal_chp"]})},
        parameters={
            ("unit", "min_load"): pl.DataFrame({
                "name": ["coal_chp"], "value": [0.4],
            }),
        },
    )
    p = drv.p_min_load_pdt_from_source(src, dt)
    assert isinstance(p, Param)
    assert p.dims == ("p", "d", "t")
    frame = p.frame.collect() if hasattr(p.frame, "collect") else p.frame
    # Scalar broadcast → 0.4 for every (d, t).
    assert frame["value"].to_list() == pytest.approx([0.4] * 4)


def test_p_connection_susceptance_inmemory_explicit():
    src = InMemoryReader(
        entities={"connection": pl.DataFrame({"name": ["line_AB"]})},
        parameters={
            ("connection", "susceptance"): pl.DataFrame({
                "name": ["line_AB"], "value": [10.0],
            }),
        },
    )
    p = dp.p_connection_susceptance_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("p",)
    assert p.frame.sort("p")["value"].to_list() == pytest.approx([10.0])


def test_p_connection_susceptance_inmemory_unknown_param():
    """Older fixtures may lack ``susceptance`` on the schema — helper
    returns None instead of crashing.
    """
    src = InMemoryReader(
        entities={"connection": pl.DataFrame({"name": ["line_AB"]})},
        parameters={},
    )
    assert dp.p_connection_susceptance_from_source(src) is None


def test_p_commodity_unitsize_inmemory_explicit():
    src = InMemoryReader(
        entities={"commodity": pl.DataFrame({"name": ["gas"]})},
        parameters={
            ("commodity", "unitsize"): pl.DataFrame({
                "name": ["gas"], "value": [2.0],
            }),
        },
    )
    p = dp.p_commodity_unitsize_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("c",)
    assert p.frame.sort("c")["value"].to_list() == pytest.approx([2.0])


def test_p_reserve_duration_inmemory_explicit():
    """Explicit ``reserve_duration`` row →
    ``Param(("r", "ud", "g"), [r, ud, g, value])``."""
    src = InMemoryReader(
        entities={"reserve__upDown__group": pl.DataFrame({
            "reserve": ["primary"], "upDown": ["up"], "group": ["elecG"],
        })},
        parameters={
            ("reserve__upDown__group", "reserve_duration"): pl.DataFrame({
                "reserve": ["primary"], "upDown": ["up"], "group": ["elecG"],
                "value": [0.5],
            }),
        },
    )
    p = dp.p_reserve_upDown_group_reserve_duration_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("r", "ud", "g")
    got = p.frame.sort("r", "ud", "g")
    assert got["r"].to_list() == ["primary"]
    assert got["ud"].to_list() == ["up"]
    assert got["g"].to_list() == ["elecG"]
    assert got["value"].to_list() == pytest.approx([0.5])


def test_p_reserve_duration_inmemory_absent():
    """No ``reserve_duration`` parameter at all → helper returns None
    (byte-identical LP: the storage-coupling constraints are skipped)."""
    src = InMemoryReader(
        entities={"reserve__upDown__group": pl.DataFrame({
            "reserve": ["primary"], "upDown": ["up"], "group": ["elecG"],
        })},
        parameters={},
    )
    assert dp.p_reserve_upDown_group_reserve_duration_from_source(src) is None


def test_p_reserve_duration_inmemory_empty_rows():
    """Header-only ``reserve_duration`` frame → helper returns None."""
    src = InMemoryReader(
        entities={"reserve__upDown__group": pl.DataFrame({
            "reserve": ["primary"], "upDown": ["up"], "group": ["elecG"],
        })},
        parameters={
            ("reserve__upDown__group", "reserve_duration"): pl.DataFrame(
                schema={"reserve": pl.Utf8, "upDown": pl.Utf8,
                        "group": pl.Utf8, "value": pl.Float64},
            ),
        },
    )
    assert dp.p_reserve_upDown_group_reserve_duration_from_source(src) is None


# ---------------------------------------------------------------------------
# Fixture-driven rebuild check: the field of a rebuild from the post-solve provider (the production
# loader call, ``load_scenario``) == the field of the FlexData the cascade itself built and solved
# (``run.last_flex_data``).  These used to compare a CSV load with a DB-direct load of the same,
# DB-derived snapshot, which could no longer fail.

PARITY_SCENARIOS = ["coal", "test_a_lot", "coal_min_load"]


@pytest.fixture(scope="module")
def rebuilt(scenario_run):
    """``scenario -> (rebuilt FlexData, cascade FlexData)``, one rebuild per scenario."""
    cache: dict[str, tuple[object, object]] = {}

    def get(scenario: str) -> tuple[object, object]:
        if scenario not in cache:
            run = scenario_run(scenario)
            assert run.last_flex_data is not None, f"{scenario}: the cascade kept no FlexData"
            cache[scenario] = (load_scenario(run), run.last_flex_data)
        return cache[scenario]

    return get


@pytest.mark.parametrize("scenario", PARITY_SCENARIOS)
@pytest.mark.parametrize("field", [
    "p_state_self_discharge",
    "p_state_start",
    # ``p_min_load`` moved to the derived (p, d, t) path — see
    # ``test_min_load_pdt.py`` (specs/fix_map_reading.md §5).
    "p_commodity_unitsize",
    # Δ.4b additions:
    "p_startup_cost",
    "p_co2_price",
    "p_co2_max_period",
    "p_node_availability",
    "p_storage_state_reference_value",
    "pdGroup_capacity_margin",
    "pdGroup_penalty_capacity_margin",
    "pdGroup_inertia_limit",
    "pdGroup_penalty_inertia",
    "pdGroup_non_synchronous_limit",
    "pdGroup_penalty_non_synchronous",
    "p_group_invest_max_period",
    "p_group_invest_min_period",
    "p_group_retire_max_period",
    "p_group_retire_min_period",
    "p_group_invest_max_total",
    "p_group_invest_min_total",
    "p_group_retire_max_total",
    "p_group_retire_min_total",
    "pd_group_cumulative_max_capacity",
    "pd_group_cumulative_min_capacity",
    "p_group_max_cumulative_flow",
    "p_group_min_cumulative_flow",
    "pd_max_cumulative_flow",
    "pd_min_cumulative_flow",
    "pdt_max_instant_flow",
    "pdt_min_instant_flow",
    "ed_invest_max_period",
    "ed_divest_max_period",
    "ed_invest_min_period",
    "ed_divest_min_period",
    "ed_cumulative_max_capacity",
    "ed_cumulative_min_capacity",
    "p_ramp_speed_up_sink",
    "p_ramp_speed_down_sink",
    "p_ramp_speed_up_source",
    "p_ramp_speed_down_source",
    "p_process_sink_inertia_constant",
    "p_process_source_inertia_constant",
    "p_pdt_varCost_source",
    "p_pdt_varCost_sink",
    "p_pdt_varCost_process",
    "pdtReserve_upDown_group_reservation",
    "p_reserve_upDown_group_penalty_reserve",
    "p_process_reserve_upDown_node_reliability",
    "p_process_reserve_upDown_node_max_share",
    "p_process_reserve_upDown_node_large_failure_ratio_value",
    "p_process_reserve_upDown_node_increase_reserve_ratio_value",
    "process_delayed__duration",
    "p_process_availability",
    "p_commodity_price",
    "p_ladder_ann_price",
    "p_ladder_ann_quantity",
    "p_ladder_cum_price",
    "p_ladder_cum_quantity",
])
def test_delta4_field_rebuild_vs_cascade(scenario: str, field: str, rebuilt):
    """For every (fixture × field), the rebuild from the post-solve provider and the cascade's own
    FlexData agree frame-for-frame (row sets; both ``None`` when the field is unused in that fixture).
    """
    data, cascade = rebuilt(scenario)
    diff = field_diff(getattr(data, field), getattr(cascade, field))
    assert diff is None, f"{field}: rebuild != cascade on {scenario}: {diff}"
