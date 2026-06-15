"""Regression: ``compute_co2_rolling_accumulator`` must read the
unitsize param off whichever entity axis it carries.

``p_all_entity_unitsize`` is keyed on the entity-union axis ``e``
(unit ∪ node ∪ connection); ``p_unitsize`` is keyed on the process
axis ``p``.  The accumulator prefers ``p_all_entity_unitsize`` and
used to ``.select(pl.col("p"))`` unconditionally — which raised
``ColumnNotFoundError: unable to find column "p"; valid columns:
["e","value"]`` for every model that actually reaches this code path
(an active ``co2_method=total`` cap on a rolling solve).  No shipped
fixture used ``co2_method=total``, so the bug was latent from the v4
root until the SouthAfrica model exercised it.

This test crafts the minimal frames that drive the emission branch and
hand-checks the resulting cumulative-CO2 tonnage, with the unitsize
delivered on the ``e`` axis (fails before the fix, passes after).
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from flextool.engine_polars._emit_co2_accumulators import (
    compute_co2_rolling_accumulator,
)


class _Provider:
    """Minimal Provider stub: ``has`` / ``get`` over a dict of frames."""

    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        self._frames = frames

    def has(self, key: str) -> bool:
        return key in self._frames

    def get(self, key: str):
        return self._frames.get(key)


class _Sol:
    """Minimal solution stub exposing ``_vars`` and ``value``."""

    def __init__(self, v_flow: pl.DataFrame) -> None:
        self._vars = {"v_flow": object()}
        self._v_flow = v_flow

    def value(self, name: str):
        return self._v_flow if name == "v_flow" else None


def _flex_data(*, unitsize_axis: str) -> SimpleNamespace:
    """FlexData stub carrying exactly the attributes the emission branch
    of the accumulator reads.

    ``unitsize_axis`` selects which carrier (and hence which axis column)
    delivers the per-process unitsize:
      * ``"e"`` → ``p_all_entity_unitsize`` (entity-union axis) — the
        path that crashed.
      * ``"p"`` → ``p_unitsize`` (process axis).
    """
    fd = SimpleNamespace()
    # Realized (d, t) — uniform-weighted single step.
    fd.realized_dispatch = pl.DataFrame(
        {"period": ["p1"], "step": ["t1"]}
    )
    fd.p_step_duration = Param(
        ("d", "t"),
        pl.DataFrame({"d": ["p1"], "t": ["t1"], "value": [2.0]}),
    )
    fd.p_timestep_weight = Param(
        ("d", "t"),
        pl.DataFrame({"d": ["p1"], "t": ["t1"], "value": [1.0]}),
    )
    # noEff process u1: source=coal_node (a CO2 node), sink=elec.
    fd.process_source_sink_noEff = pl.DataFrame(
        {"p": ["u1"], "source": ["coal_node"], "sink": ["elec"]}
    )
    fd.process_source_sink_eff = None
    fd.process_unit = pl.DataFrame({"p": ["u1"]})
    fd.process_min_load_eff = None
    fd.p_co2_content = Param(
        ("c",), pl.DataFrame({"c": ["coal"], "value": [100.0]})
    )
    fd.p_slope = None
    fd.p_process_source_conversion_flow_coeff = None
    fd.p_process_sink_conversion_flow_coeff = None
    fd.group_node = pl.DataFrame({"g": ["cap_grp"], "n": ["coal_node"]})

    if unitsize_axis == "e":
        # The carrier that crashed: keyed on the entity-union axis ``e``.
        fd.p_all_entity_unitsize = Param(
            ("e",), pl.DataFrame({"e": ["u1"], "value": [5.0]})
        )
        fd.p_unitsize = None
    else:
        fd.p_all_entity_unitsize = None
        fd.p_unitsize = Param(
            ("p",), pl.DataFrame({"p": ["u1"], "value": [5.0]})
        )
    return fd


def _provider() -> _Provider:
    return _Provider({
        "input/group__co2_method": pl.DataFrame(
            {"group": ["cap_grp"], "co2_method": ["total"]}
        ),
        "solve_data/commodity_node_co2": pl.DataFrame(
            {"commodity": ["coal"], "node": ["coal_node"]}
        ),
    })


def _v_flow() -> pl.DataFrame:
    return pl.DataFrame({
        "p": ["u1"],
        "source": ["coal_node"],
        "sink": ["elec"],
        "d": ["p1"],
        "t": ["t1"],
        "value": [10.0],
    })


# Hand calc (emission branch, noEff → slope=coeff=1):
#   content/1000 * value * us * dur * rpw
#   = 100/1000 * 10 * 5 * 2 * 1 = 0.1 * 100 = 10.0 tonnes
_EXPECTED_TONNES = 10.0


@pytest.mark.parametrize("unitsize_axis", ["e", "p"])
def test_accumulator_reads_unitsize_off_either_axis(
    unitsize_axis: str, tmp_path
) -> None:
    out = compute_co2_rolling_accumulator(
        _flex_data(unitsize_axis=unitsize_axis),
        _Sol(_v_flow()),
        work_folder=tmp_path,
        prior_cumulative_co2=None,
        provider=_provider(),
    )
    assert out.columns == [
        "group", "period", "p_co2_cum_realized_tonnes",
    ]
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["group"] == "cap_grp"
    assert row["period"] == "p1"
    assert row["p_co2_cum_realized_tonnes"] == pytest.approx(
        _EXPECTED_TONNES, rel=1e-12
    )


def test_accumulator_carries_prior_cumulative(tmp_path) -> None:
    """The ``e``-axis unitsize path also sums the prior-roll carrier."""
    prior = pl.DataFrame({
        "group": ["cap_grp"], "period": ["p1"], "value": [3.5],
    })
    out = compute_co2_rolling_accumulator(
        _flex_data(unitsize_axis="e"),
        _Sol(_v_flow()),
        work_folder=tmp_path,
        prior_cumulative_co2=prior,
        provider=_provider(),
    )
    row = out.row(0, named=True)
    assert row["p_co2_cum_realized_tonnes"] == pytest.approx(
        _EXPECTED_TONNES + 3.5, rel=1e-12
    )
