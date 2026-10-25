"""Benders split guards for connections between regions (cycle c14b critique change 4).

A connection between two regions (or a region and the master) is replaced by half-flows in each region, so its
own rows are gone: online variables or an exact direction switch would be free columns without flow rows, and
its reserve (limited by its flows, ``connection_reserve_*``) would have no channel left.  ``split`` raises for
them.  A region subproblem with integer columns (binary online status, exact switch) gets a warning: Benders cuts
from MIP duals are not valid.

Fixture: ``lh2_three_region_trade_invest`` (regions A / B / C, pipes pipe_AB and pipe_BC between them), built
from JSON by the ``scenario_workdir`` fixture; online / exact / reserve membership set on the loaded FlexData.
"""
from __future__ import annotations

import dataclasses
import logging

import polars as pl
import pytest

from flextool.engine import load_flextool
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._region_filter import split

REGIONS = ["region_A", "region_B", "region_C"]


@pytest.fixture(scope="module")
def data(scenario_workdir):
    return load_flextool(scenario_workdir("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))


def _replace(data, **fields):
    """dataclasses.replace keeping the loader's axis-enum snapshot (not a dataclass field)."""
    new = dataclasses.replace(data, **fields)
    new._axis_enums = getattr(data, "_axis_enums", None)
    return new


def _procs(data, names: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"p": names}).with_columns(pl.col("p").cast(data.process_source_sink.schema["p"]))


@pytest.mark.parametrize("field", ["process_online_linear", "process_online_integer", "process_exact"])
def test_cross_region_connection_online_or_exact_raises(data, field) -> None:
    bad = _replace(data, **{field: _procs(data, ["pipe_AB"])})
    with pytest.raises(FlexToolConfigError, match="pipe_AB"):
        split(bad, regions=REGIONS, benders_uncap_cross_region=True)


def test_cross_region_connection_reserve_raises(data) -> None:
    pss = data.process_source_sink
    pr = pl.DataFrame({"p": ["pipe_AB"], "r": ["r1"], "ud": ["up"], "n": ["lh2_B"]}).with_columns(
        pl.col("p").cast(pss.schema["p"]), pl.col("n").cast(pss.schema["sink"]))
    bad = _replace(data, process_reserve_upDown_node_active=pr)
    with pytest.raises(FlexToolConfigError, match="pipe_AB.*reserve"):
        split(bad, regions=REGIONS, benders_uncap_cross_region=True)


def test_in_region_integer_connection_warns(data, caplog) -> None:
    """An in-region connection with integer online variables splits, with a warning naming it in its region
    only."""
    ok = _replace(data, process_online_integer=_procs(data, ["electrolyser_A"]))
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._region_filter"):
        splits = split(ok, regions=REGIONS, benders_uncap_cross_region=True)
    assert len(splits) == 3
    msgs = [r.getMessage() for r in caplog.records if "integer variables" in r.getMessage()]
    assert len(msgs) == 1 and "region_A" in msgs[0] and "electrolyser_A" in msgs[0]


def test_plain_split_has_no_guard_hit(data, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._region_filter"):
        assert len(split(data, regions=REGIONS, benders_uncap_cross_region=True)) == 3
    assert not [r for r in caplog.records if "integer variables" in r.getMessage()]
