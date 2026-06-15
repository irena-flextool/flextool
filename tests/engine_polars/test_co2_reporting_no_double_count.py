"""CO2 emissions-reporting set: total-cap wiring + no group double-count.

The system-wide ``CO2 [Mt]`` total (``calc_costs.emissions_co2_d``) is a
*per-process* sum over ``read_sets.process__commodity__node_co2`` — NOT a
sum of per-group totals.  That set is the deduplicated union of the
CO2-priced / period-capped / total-capped emitting-flow frames
(``_CO2_EMISSION_FLOW_ATTRS``).

Two invariants are pinned here:

1. **Total-cap is wired in** — a ``co2_method = total`` model (only the
   ``*_capped_total`` frames populated) must still contribute emitting
   flows, else the CO2 reports read 0 (the SouthAfrica regression).
2. **No double-count across overlapping groups** — a process whose flow
   appears in several CO2 frames (because it belongs to several CO2
   groups) collapses to a single ``(process, commodity, node)`` row, so
   the per-process total counts it exactly once.  This would break if a
   future refactor switched the total to a sum over per-group emissions.
"""
from __future__ import annotations

import types

import polars as pl

from flextool.process_outputs.read_sets import (
    _CO2_EMISSION_FLOW_ATTRS,
    _process_commodity_node_from_flex,
)


def _flow(p: str, c: str, src: str) -> pl.DataFrame:
    return pl.DataFrame({"p": [p], "c": [c], "source": [src]})


def test_total_cap_flow_sets_are_wired_in() -> None:
    """The total-cap frames must be part of the emissions set (regression
    for ``co2_method = total`` reporting 0)."""
    fields = {f for f, _ in _CO2_EMISSION_FLOW_ATTRS}
    assert "flow_from_co2_capped_total" in fields
    assert "flow_from_co2_capped_total_noEff" in fields


def test_total_only_model_emitting_flows_present() -> None:
    """A pure total-cap model (only ``*_capped_total`` populated) yields a
    non-empty emissions set — previously empty → 0 CO2."""
    fd = types.SimpleNamespace(
        flow_from_co2_priced=None,
        flow_from_co2_priced_noEff=None,
        flow_from_co2_capped=None,
        flow_from_co2_capped_noEff=None,
        flow_from_co2_capped_total=_flow("Medupi", "Coal_Medupi", "Coal_Medupi_node"),
        flow_from_co2_capped_total_noEff=None,
    )
    idx = _process_commodity_node_from_flex(fd, attrs=_CO2_EMISSION_FLOW_ATTRS)
    assert list(map(tuple, idx)) == [("Medupi", "Coal_Medupi", "Coal_Medupi_node")]


def test_overlapping_co2_groups_counted_once() -> None:
    """A process in two overlapping CO2 groups (its flow in both a
    period-cap and a total-cap frame) appears exactly ONCE, so the
    per-process system total counts its emissions once, not twice."""
    fd = types.SimpleNamespace(
        flow_from_co2_priced=_flow("P_priced", "gas", "gas_node"),
        flow_from_co2_priced_noEff=None,
        # P_overlap's emitting flow is identical in both cap frames.
        flow_from_co2_capped=_flow("P_overlap", "coal", "coal_node"),
        flow_from_co2_capped_noEff=None,
        flow_from_co2_capped_total=pl.concat(
            [
                _flow("P_overlap", "coal", "coal_node"),  # same (p, c, node)
                _flow("P_total", "diesel", "diesel_node"),
            ]
        ),
        flow_from_co2_capped_total_noEff=None,
    )
    rows = sorted(map(tuple, _process_commodity_node_from_flex(
        fd, attrs=_CO2_EMISSION_FLOW_ATTRS)))

    # The overlapping process collapses to a single row -> counted once.
    assert rows.count(("P_overlap", "coal", "coal_node")) == 1
    # Nothing is dropped: every distinct emitting flow is present.
    assert ("P_total", "diesel", "diesel_node") in rows
    assert ("P_priced", "gas", "gas_node") in rows
    assert len(rows) == 3
