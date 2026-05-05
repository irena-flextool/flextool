"""Δ.4 — second-wave Direct Param migration parity tests.

Δ.4 extends ``apply_direct_params`` (formerly ``first_wave_overrides``)
with helpers for scalar Direct Params previously read by
``input.py``'s CSV loaders only:

* ``p_state_self_discharge`` — ``node.self_discharge_loss``
* ``p_state_start`` — ``node.storage_state_start``
* ``p_min_load`` — ``unit.min_load``
* ``p_connection_susceptance`` — ``connection.susceptance``
* ``p_commodity_unitsize`` — ``commodity.unitsize``

Each helper mirrors the CSV path's "explicit rows only" semantics
via :meth:`InputSource.parameter_explicit`.  The DB-direct overlay
runs after the CSV seed in ``load_flextool``; equality of the two
paths is checked frame-for-frame on the existing fixture corpus.

The tests below are deliberately narrow: they exercise the helpers
on representative fixtures using :class:`InMemoryReader` and on a
real-fixture overlay through ``load_flextool(workdir, db_reader=...)``.
The wider parity sweep (CSV vs DB on every fixture) is already
covered by ``test_override_chain_step3_parity.py``'s end-to-end
LP-objective test.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import (
    InMemoryReader,
    SpineDbReader,
    load_flextool,
)
from flextool.engine_polars import _direct_params as dp
from polar_high import Param

DATA = Path(__file__).resolve().parent / "data"


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


def test_p_state_self_discharge_inmemory_explicit():
    """Explicit ``self_discharge_loss`` row → ``Param(("n",), [n, value])``."""
    src = _node_reader("nA", 0.05, "self_discharge_loss")
    p = dp.p_state_self_discharge_from_source(src)
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
    assert dp.p_state_self_discharge_from_source(src) is None


def test_p_state_start_inmemory_explicit():
    src = _node_reader("battery", 0.5, "storage_state_start")
    p = dp.p_state_start_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("n",)
    assert p.frame.sort("n")["value"].to_list() == pytest.approx([0.5])


def test_p_min_load_inmemory_explicit():
    src = InMemoryReader(
        entities={"unit": pl.DataFrame({"name": ["coal_chp"]})},
        parameters={
            ("unit", "min_load"): pl.DataFrame({
                "name": ["coal_chp"], "value": [0.4],
            }),
        },
    )
    p = dp.p_min_load_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("p",)
    assert p.frame.sort("p")["value"].to_list() == pytest.approx([0.4])


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


# ---------------------------------------------------------------------------
# Fixture-driven CSV vs DB-direct frame parity.

# (work_dirname, scenario)
PARITY_FIXTURES = [
    ("work_coal", "coal"),
    ("work_test_a_lot", "test_a_lot"),
    ("work_coal_min_load", "coal_min_load"),
]


def _load_csv_db(work: Path, scenario: str):
    csv = load_flextool(work)
    reader = SpineDbReader(work / "tests.sqlite", scenario)
    db = load_flextool(work, db_reader=reader)
    return csv, db


def _frame(x):
    if x is None:
        return None
    if hasattr(x, "frame"):
        return x.frame
    return x


@pytest.mark.parametrize("work_name,scenario", PARITY_FIXTURES,
                          ids=[f[0] for f in PARITY_FIXTURES])
@pytest.mark.parametrize("field", [
    "p_state_self_discharge",
    "p_state_start",
    "p_min_load",
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
    "p_group_invest_max_cumulative",
    "p_group_invest_min_cumulative",
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
def test_delta4_field_csv_vs_db_parity(work_name: str, scenario: str,
                                          field: str):
    """For every (fixture × field), CSV and DB-direct paths agree
    frame-for-frame.  Skip when both paths produce ``None`` (the field
    is unused in that fixture).
    """
    work = DATA / work_name
    if not (work / "tests.sqlite").exists():
        pytest.skip(f"{work_name} has no tests.sqlite")
    csv, db = _load_csv_db(work, scenario)
    a = _frame(getattr(csv, field, None))
    b = _frame(getattr(db, field, None))
    if a is None and b is None:
        pytest.skip(f"{field} is None on both CSV and DB for {work_name}")
    assert a is not None, f"{field}: CSV is None, DB is not on {work_name}"
    assert b is not None, f"{field}: DB is None, CSV is not on {work_name}"
    keep = sorted(a.columns)
    aa = a.sort(keep) if keep else a
    bb = b.sort(keep) if keep else b
    if "value" in aa.columns and "value" in bb.columns:
        aa = aa.with_columns(value=pl.col("value").cast(pl.Float64,
                                                            strict=False))
        bb = bb.with_columns(value=pl.col("value").cast(pl.Float64,
                                                            strict=False))
    assert aa.equals(bb), (
        f"{field} mismatch on {work_name}:\nCSV:\n{aa}\nDB:\n{bb}"
    )
