"""The node ``unit_of_measure`` derivation rule (``flextool.units_of_measure``) and the output unit-source registry.

Oracle values are written by hand from the rule: amount A, rate R = A per hour (``^(.*W)h(.*)$`` -> ``\\1\\2``,
else ``A/h``), kilo forms for the per-kW / per-kWh costs, ``Mixed units of measure!`` for a sum over different
units, and the capacity label of owner decision S1 (``MW + MW_th`` shared, ``MW; MW_th`` per arc).
"""
from __future__ import annotations

import re

import polars as pl
import pytest

from flextool import units_of_measure as uom
from flextool.input_derivation._method_constants import (
    METHOD_1WAY,
    METHOD_2WAY_1VAR,
    METHOD_2WAY_2VAR,
    METHOD_DIRECT,
    METHOD_INDIRECT,
)
from flextool.input_derivation._specs import METHODS_MAPPING
from flextool.process_outputs import _output_meta as meta

MIXED = "Mixed units of measure!"


def test_mixed_label_text() -> None:
    assert uom.MIXED_UOM == MIXED
    assert uom.DEFAULT_A == "MWh"


@pytest.mark.parametrize("value,amount", [
    (None, "MWh"), ("", "MWh"), ("   ", "MWh"), (" m3 ", "m3"), ("MWh_th", "MWh_th"), ("t", "t")])
def test_normalise(value, amount: str) -> None:
    assert uom.normalise(value) == amount


@pytest.mark.parametrize("amount,rate", [
    ("MWh", "MW"), ("MWh_th", "MW_th"), ("kWh", "kW"), ("GWh", "GW"), ("m3", "m3/h"), ("t", "t/h"),
    ("Nm3", "Nm3/h"), ("mwh", "mwh/h"), (MIXED, MIXED)])
def test_rate_of(amount: str, rate: str) -> None:
    assert uom.rate_of(amount) == rate


@pytest.mark.parametrize("amount,k_rate,k_amount", [
    ("MWh", "kW", "kWh"), ("MWh_th", "kW_th", "kWh_th"), ("m3", "(0.001 m3/h)", "(0.001 m3)"),
    ("t", "(0.001 t/h)", "(0.001 t)"), ("GWh", "(0.001 GW)", "(0.001 GWh)"), (MIXED, MIXED, MIXED)])
def test_kilo_forms(amount: str, k_rate: str, k_amount: str) -> None:
    assert uom.kilo_rate(amount) == k_rate
    assert uom.kilo_amount(amount) == k_amount


@pytest.mark.parametrize("template,amount,label", [
    ("{R}", "MWh", "MW"), ("{A}/a", "MWh", "MWh/a"), ("{A}", "MWh", "MWh"), ("CUR/{A}", "MWh", "CUR/MWh"),
    ("CUR/{R}", "MWh", "CUR/MW"), ("CUR/{kR}", "MWh", "CUR/kW"), ("CUR/{kA}", "MWh", "CUR/kWh"),
    ("{A}/h", "MWh", "MWh/h"), ("{A} (annualized)", "MWh", "MWh (annualized)"),
    ("{R}", "m3", "m3/h"), ("{A}/a", "m3", "m3/a"), ("CUR/{A}", "m3", "CUR/m3"),
    ("CUR/{kR}", "m3", "CUR/(0.001 m3/h)"), ("CUR/{kA}", "m3", "CUR/(0.001 m3)"), ("{R}", "MWh_th", "MW_th"),
    ("{A}/a", MIXED, MIXED), ("CUR/{kR}", MIXED, MIXED), ("{R}", MIXED, MIXED),
    ("MW", "m3", "MW"), ("MW", MIXED, "MW"), ("", "m3", ""),
    # A compound unit after "/" is parenthesised (WP1 review S1): per m3/h, not per m3 per hour.
    ("CUR/{R}", "m3", "CUR/(m3/h)"), ("CUR/{R}", "t", "CUR/(t/h)"), ("CUR/{R}", "MWh_th", "CUR/MW_th"),
    ("{R}/a", "m3", "m3/h/a")])
def test_fill(template: str, amount: str, label: str) -> None:
    assert uom.fill(template, amount) == label


def test_units_of() -> None:
    units = {"water": "m3", "heat": "MWh_th"}
    assert uom.units_of([], units) == "MWh"
    assert uom.units_of(["east", "west"], units) == "MWh"
    assert uom.units_of(["water"], units) == "m3"
    assert uom.units_of(["water", "water"], units) == "m3"
    assert uom.units_of(["east", "water"], units) == MIXED
    assert uom.units_of(["heat", "water"], units) == MIXED


@pytest.mark.parametrize("template,nodes,shared,label", [
    ("{R}", ["east"], True, "MW"),
    ("{R}", ["east", "west"], True, "MW"),
    ("{R}", ["east", "west"], False, "MW"),
    ("{R}", ["east", "heat"], True, "MW + MW_th"),
    ("{R}", ["heat", "east"], True, "MW_th + MW"),
    ("{R}", ["east", "heat", "west"], True, "MW + MW_th"),
    ("{R}", ["east", "heat"], False, "MW; MW_th"),
    ("CUR/{R}", ["east", "heat"], True, "CUR/(MW + MW_th)"),
    ("CUR/{kR}", ["east", "heat"], True, "CUR/(kW + kW_th)"),
    ("CUR/{kR}", ["east", "heat"], False, "CUR/kW; CUR/kW_th"),
    ("CUR/{kR}", ["water"], True, "CUR/(0.001 m3/h)"),
    ("CUR/{kR}", ["east", "water"], True, "CUR/(kW + 0.001 m3/h)"),
    ("CUR/{R}", ["east", "water"], True, "CUR/(MW + m3/h)"),
    ("CUR/{R}", ["east", "water"], False, "CUR/MW; CUR/(m3/h)"),
    ("{R}", [], True, "MW"),
    ("MW", ["east", "heat"], True, "MW"),
])
def test_capacity_units(template: str, nodes: list[str], shared: bool, label: str) -> None:
    """Owner decision S1: shared capacity -> rates joined with ' + ' (output-arc order), per-arc caps -> '; ',
    one unit when all outputs agree; never the mixed label."""
    units = {"water": "m3", "heat": "MWh_th"}
    assert uom.capacity_units(template, nodes, units, shared=shared) == label


def test_shares_capacity_is_the_indirect_methods() -> None:
    """maxOutputSum (shared capacity) is built for process_indirect = the METHOD_INDIRECT methods."""
    methods = set(METHODS_MAPPING.values()) | METHOD_DIRECT | METHOD_INDIRECT | METHOD_1WAY | METHOD_2WAY_1VAR \
        | METHOD_2WAY_2VAR
    assert {m for m in methods if uom.shares_capacity(m)} == set(METHOD_INDIRECT)


@pytest.mark.parametrize("value,rate_like", [
    ("m3/h", True), ("kg/s", True), ("MW", True), ("kW", True), ("MW_th", True), ("W", True),
    ("m3", False), ("MWh", False), ("MWh_th", False), ("kWh", False), ("t", False), ("", False),
    ("Wood_t", False), ("Water", False), ("Water_m3", False), ("H2_LHV_MWh_W", False)])
def test_looks_like_rate(value: str, rate_like: bool) -> None:
    assert uom.looks_like_rate(value) is rate_like


@pytest.mark.parametrize("value,amount", [
    ("m3/h", "m3"), ("kg/s", "kg"), ("MW", "MWh"), ("MW_th", "MWh_th"), ("kW", "kWh"), ("m3", "m3")])
def test_amount_of_rate(value: str, amount: str) -> None:
    assert uom.amount_of_rate(value) == amount


class _Provider:
    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        self.frames = frames

    def has(self, name: str) -> bool:
        return name in self.frames

    def get(self, name: str) -> pl.DataFrame:
        return self.frames[name]


def test_read_node_units() -> None:
    frame = pl.DataFrame({"node": ["water", "east", "heat", "blank"],
                          "unit_of_measure": ["m3", "MWh", " MWh_th ", ""]})
    assert uom.read_node_units(_Provider({uom.NODE_UNITS_FRAME: frame})) == {"water": "m3", "heat": "MWh_th"}
    assert uom.read_node_units(None) == {}
    assert uom.read_node_units(_Provider({})) == {}
    assert uom.read_node_units(_Provider({uom.NODE_UNITS_FRAME: frame.head(0)})) == {}


def test_spec_writes_the_node_units_frame() -> None:
    from flextool.input_derivation._specs import _PARAMETER_SPECS

    spec = next(s for s in _PARAMETER_SPECS if ("node", "unit_of_measure") in s["cl_pars"])
    assert spec["header"] == "node,unit_of_measure"
    assert spec["filename"] == uom.NODE_UNITS_FRAME + ".csv"


# ─── the unit-source registry ────────────────────────────────────────────

#: A unit of power or energy (and the prices per them): the units a node's unit_of_measure can replace.
_CARRIER_UNIT = re.compile(r"^(CUR/)?[kM]W(h|·s)?(/a)?$")


def _units(output_key: str) -> set[str]:
    spec = meta.OUTPUT_TRANSFORM.get(output_key)
    if spec is None:
        return set()
    transforms = spec.values() if isinstance(spec, dict) else [spec]
    return {tf.unit for tf in transforms if tf is not None and tf.unit}


def test_every_power_or_energy_table_has_a_unit_source() -> None:
    """A table with an MW / MWh / CUR-per-MW(h) unit follows its nodes (UNIT_SOURCE) or stays a power feature
    (DEFAULT_UNITS_ONLY): nothing is forgotten silently."""
    carrier = {k for k in meta.OUTPUT_DIMS if any(_CARRIER_UNIT.search(u) for u in _units(k))}
    assert not set(meta.UNIT_SOURCE) & meta.DEFAULT_UNITS_ONLY
    assert carrier - set(meta.UNIT_SOURCE) - meta.DEFAULT_UNITS_ONLY == set()
    assert set(meta.UNIT_SOURCE) | meta.DEFAULT_UNITS_ONLY <= carrier


def test_unit_source_levels_are_table_levels() -> None:
    for key, source in meta.UNIT_SOURCE.items():
        dims = meta.OUTPUT_DIMS[key]
        levels = dims.rows if source.row else dims.columns
        assert source.level in levels, (key, source, dims)


def test_power_features_are_default_units_only() -> None:
    """Critique M1: the reserve / non-synchronous / inertia tables with a node or group level stay MW."""
    for key in ("process_reserve_upDown_node_dt_eppe", "nodeGroup_slack_reserve_dt_eeg",
                "nodeGroup_slack_reserve_d_eeg", "nodeGroup_slack_nonsync_dt_g", "nodeGroup_slack_nonsync_d_g",
                "nodeGroup_inertia_largest_flow_dt_g"):
        assert key in meta.DEFAULT_UNITS_ONLY


def test_node_level_beats_connection_ends() -> None:
    assert meta.UNIT_SOURCE["connection_leftward_dt_eee"].kind is uom.UnitSourceKind.NODE
    assert meta.UNIT_SOURCE["connection_dt_eee"].kind is uom.UnitSourceKind.CONNECTION_ENDS


def test_empty_context_is_the_default() -> None:
    ctx = meta.UnitContext()
    assert ctx.node_units == {} and ctx.capacity_nodes == {} and ctx.group_nodes == {}
    assert ctx.shared_capacity == frozenset()
