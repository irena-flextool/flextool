"""Benders split guards for connections between regions.

A connection between two regions (or a region and the master) is replaced by half-flows in each region, so its
own rows are gone: online variables or an exact direction switch would be free columns without flow rows, and
its reserve (limited by its flows, ``connection_reserve_*``) would have no channel left.  ``split`` raises for
them.  A region subproblem with integer columns (binary online status, exact switch) gets a warning: Benders cuts
from MIP duals are not valid.

Fixture: ``lh2_three_region_trade_invest`` (regions A / B / C, pipes pipe_AB and pipe_BC between them), built
from JSON by the ``scenario_run`` fixture; online / exact / reserve membership set on the loaded FlexData.
"""
from __future__ import annotations

import dataclasses
import logging

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve._region_filter import split
from tests._support.load import load_scenario

REGIONS = ["region_A", "region_B", "region_C"]


@pytest.fixture(scope="module")
def data(scenario_run):
    return load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"))


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


# ---------------------------------------------------------------------------
# reduce_unit_detail_method / reduce_system_detail_method lift the guards (the split runs on the reduced data)


@pytest.mark.parametrize("field, unit, system", [
    ("process_reserve_upDown_node_active", "none", "reserves"),
    ("process_online_linear", "online", "none"),
    ("process_online_integer", "online", "none"),
    ("process_exact", "integers", "none"),
])
def test_cross_region_connection_splits_when_the_detail_is_left_out(data, field, unit, system) -> None:
    """A cross-region connection with reserve, online variables or exact is refused at none and splits once the
    solve's level leaves that detail out."""
    from flextool.engine.preprocess._reduce_detail import apply_reduce_detail

    pss = data.process_source_sink
    if field == "process_reserve_upDown_node_active":
        frame = pl.DataFrame({"p": ["pipe_AB"], "r": ["r1"], "ud": ["up"], "n": ["lh2_B"]}).with_columns(
            pl.col("p").cast(pss.schema["p"]), pl.col("n").cast(pss.schema["sink"]))
    else:
        frame = _procs(data, ["pipe_AB"])
    bad = _replace(data, **{field: frame})
    with pytest.raises(FlexToolConfigError, match="pipe_AB"):
        split(bad, regions=REGIONS, benders_uncap_cross_region=True)
    reduced, report = apply_reduce_detail(bad, unit, system)
    assert getattr(reduced, field) is None
    assert len(split(reduced, regions=REGIONS, benders_uncap_cross_region=True)) == 3


def test_integer_region_warning_is_quiet_when_integers_are_relaxed(data, caplog) -> None:
    """A binary unit in a region warns at none, naming reduce_unit_detail_method; under integers its online
    columns are continuous and the region gets no warning.

    electrolyser_A gets a start-up cost (``pdt_online_integer``), a use of its online variables, so the integers
    level keeps them (relaxed) instead of pruning them as unused."""
    from flextool.engine.preprocess._reduce_detail import apply_reduce_detail

    procs = _procs(data, ["electrolyser_A"])
    ok = _replace(data, process_online_integer=procs, pdt_online_integer=procs.join(data.dt, how="cross"))
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._region_filter"):
        split(ok, regions=REGIONS, benders_uncap_cross_region=True)
    msgs = [r.getMessage() for r in caplog.records if "integer variables" in r.getMessage()]
    assert len(msgs) == 1 and "reduce_unit_detail_method = integers" in msgs[0]
    caplog.clear()
    relaxed, report = apply_reduce_detail(ok, "integers")
    assert relaxed.relax_integrality is True
    assert relaxed.process_online_integer is not None and "electrolyser_A" in _names(relaxed.process_online_integer)
    assert "online_unused" not in report.removed
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._region_filter"):
        assert len(split(relaxed, regions=REGIONS, benders_uncap_cross_region=True)) == 3
    assert not [r for r in caplog.records if "integer variables" in r.getMessage()]


def test_unused_integer_online_is_pruned_under_integers(data, caplog) -> None:
    """A binary unit whose online variables nothing uses (no start-up cost, min load, reserve, ...) loses them
    under integers: nothing integer is left to relax, and the region gets no warning."""
    from flextool.engine.preprocess._reduce_detail import apply_reduce_detail

    ok = _replace(data, process_online_integer=_procs(data, ["electrolyser_A"]))
    reduced, report = apply_reduce_detail(ok, "integers")
    assert report.removed.get("online_unused") == ["electrolyser_A"]
    assert reduced.process_online_integer is None
    assert not reduced.relax_integrality
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._region_filter"):
        assert len(split(reduced, regions=REGIONS, benders_uncap_cross_region=True)) == 3
    assert not [r for r in caplog.records if "integer variables" in r.getMessage()]


def _names(frame) -> set[str]:
    return set(frame.get_column("p").cast(pl.Utf8).to_list())
