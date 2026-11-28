"""The supply bound of ``check_pinned_realization_feasible`` follows the LP: a pinned unit online has its full
capacity (maxFlow_online: v_flow ≤ κ · v_online) and availability derates the built capacity (maxFlow: v_flow ≤
κ · availability · existing).

Hand-built data: node ``elec`` with ``penalty_method = off`` fed only by unit ``u``, 4 units of 125 MW (existing 4
units), availability 0.75, one 1-h step.  Pinned to 3 units online it can give min(3, 0.75 · 4) · 125 = 375 MWh;
pinned to 4, min(4, 3) · 125 = 375 MWh as well; pinned to 2, 250 MWh.  Before, the bound was the pinned count ×
availability, 3 · 0.75 · 125 = 281.25 MWh, so a realized need of 375 MWh raised although the LP can meet it.
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._realization import check_pinned_realization_feasible


def _data(pinned: "float | None", need: float) -> SimpleNamespace:
    one = {"d": ["p"], "t": ["t1"]}
    return SimpleNamespace(
        nodeBalance_penalty_off=pl.DataFrame({"n": ["elec"]}),
        p_fix_online=(None if pinned is None
                      else Param(("p", "d", "t"), pl.DataFrame({"p": ["u"], **one, "value": [pinned]}))),
        flow_to_n=pl.DataFrame({"p": ["u"], "source": ["fuel"], "sink": ["elec"], "n": ["elec"]}),
        flow_from_n=None,
        p_inflow=Param(("n", "d", "t"), pl.DataFrame({"n": ["elec"], **one, "value": [-need]})),
        nodeState=None, nodeStateBlock=None, nodeCoarse=None,
        process_unit=pl.DataFrame({"p": ["u"]}),
        process_source_sink_uncapped=None,
        dt=pl.DataFrame(one),
        p_unitsize=Param(("p",), pl.DataFrame({"p": ["u"], "value": [125.0]})),
        p_process_availability=Param(("p",), pl.DataFrame({"p": ["u"], "value": [0.75]})),
        p_step_duration=Param(("d", "t"), pl.DataFrame({**one, "value": [1.0]})),
        p_process_sink_max_capacity_coef=None,
        p_arc_max_cap_coef=None,
        p_flow_upper_existing=Param(("p", "source", "sink", "d"), pl.DataFrame(
            {"p": ["u"], "source": ["fuel"], "sink": ["elec"], "d": ["p"], "value": [4.0]})),
        p_profile_value=None, process_profile_upper=None, process_profile_fixed=None,
    )


@pytest.mark.parametrize("pinned", [3.0, 4.0, None])
def test_pinned_units_at_full_capacity_within_the_derated_capacity(pinned) -> None:
    check_pinned_realization_feasible(_data(pinned, 375.0), "rlz")
    with pytest.raises(FlexToolConfigError, match=r"cannot balance node 'elec'.*at most 375 MWh"):
        check_pinned_realization_feasible(_data(pinned, 376.0), "rlz")


def test_fewer_pinned_units_than_available_bound_the_supply() -> None:
    check_pinned_realization_feasible(_data(2.0, 250.0), "rlz")
    with pytest.raises(FlexToolConfigError, match=r"at most 250 MWh"):
        check_pinned_realization_feasible(_data(2.0, 251.0), "rlz")
