"""Surface A.14 (group-level slack), A.15 (reserves) and A.19 (commodity
ladder) loader tests.

All three modules expose a ``load_data(...)`` helper that walks
hand-built ``input/`` and ``solve_data/`` directories.  Tests build the
minimal CSVs needed to exercise one transformation per spec and call
``load_data`` directly — no ``load_flextool``/``Problem`` wiring.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import _group_slack, _reserve, _commodity_ladder
from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._input_source import seed_provider_from_dir


def _provider_for_workdir(inp: Path, sd: Path) -> FlexDataProvider:
    """Seed an in-memory provider from a test ``input/`` + ``solve_data/``
    for *.load_data (Step 2.5 disk-fallback removal).
    """
    provider = FlexDataProvider()
    if inp.exists():
        seed_provider_from_dir(provider, inp, "input")
    if sd.exists():
        seed_provider_from_dir(provider, sd, "solve_data")
    return provider


# --- helpers --------------------------------------------------------

def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir()
    sd.mkdir()
    return inp, sd


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


# ====================================================================
# A.14 — _group_slack.load_data
# ====================================================================

def test_pdgroup_topfile_wide_unpivot_and_long_form_path(tmp_path: Path):
    """Covers A14-pdGroup_topfile_wide_unpivot + A14-pdGroup_long_form_path.

    Wide-per-group ``pdGroup_capacity_margin.csv`` → 2-row unpivot for
    capacity_margin; ``pdGroup.csv`` long form provides the
    inertia_limit slice via the fallback path.
    """
    inp, sd = _make_dirs(tmp_path)
    # Wide-per-group form: header (period, g1, g2) — first col is "period",
    # not "group", so the unpivot branch fires.
    _write(sd / "pdGroup_capacity_margin.csv",
           "period,g1,g2\nd1,0.10,0.20\n")
    # Long form for inertia_limit; cap_margin still served by the wide file.
    _write(sd / "pdGroup.csv",
           "group,param,period,value\n"
           "g1,inertia_limit,d1,5.0\n")

    out = _group_slack.load_data(inp, sd, dt=pl.DataFrame(), provider=_provider_for_workdir(inp, sd),
                                  nb=None, pss_eff=None, pss_noEff=None,
                                  p_unitsize=None)
    # Hand-calc: wide unpivot yields 2 (g, d) rows; groupCapacityMargin = {g1, g2}.
    assert out["groupCapacityMargin"].sort("g")["g"].to_list() == ["g1", "g2"]
    # Long-form fallback for inertia: groupInertia = {g1}.
    assert out["groupInertia"]["g"].to_list() == ["g1"]


def test_zero_drop_and_nonsync_override_and_inertia_only_filter(tmp_path: Path):
    """Covers A14-zero_value_rows_dropped_to_inactive_group +
    A14-groupNonSync_input_overrides_pdgroup_slice +
    A14-inertia_constants_inertia_only_filter.

    One frame asserts: zero rows in capacity_margin drop the group;
    ``input/groupNonSync.csv`` wins over a non_synchronous_limit pdGroup
    slice; ``p_process_sink.csv`` rows get filtered to ``inertia_constant``
    (and zero values dropped).
    """
    inp, sd = _make_dirs(tmp_path)
    # Capacity margin: g1 zero (drop), g2 non-zero (keep).
    _write(sd / "pdGroup_capacity_margin.csv",
           "period,g1,g2\nd1,0.0,0.5\n")
    # nonSync: input file has g_sync_only; pdGroup slice has g_other.
    _write(inp / "groupNonSync.csv", "group\ng_sync_only\n")
    _write(sd / "pdGroup.csv",
           "group,param,period,value\n"
           "g_other,non_synchronous_limit,d1,0.3\n")
    # Inertia constants: filter to "inertia_constant" + drop zeros.
    _write(inp / "p_process_sink.csv",
           "process,sink,sourceSinkParam,p_process_sink\n"
           "p1,n1,inertia_constant,3.0\n"
           "p1,n1,efficiency,0.95\n"
           "p2,n1,inertia_constant,0.0\n")

    out = _group_slack.load_data(inp, sd, dt=pl.DataFrame(), provider=_provider_for_workdir(inp, sd),
                                  nb=None, pss_eff=None, pss_noEff=None,
                                  p_unitsize=None)
    # Hand-calc: zero-row drop ⇒ g1 absent, only g2.
    assert out["groupCapacityMargin"]["g"].to_list() == ["g2"]
    # Hand-calc: input override ⇒ groupNonSync = {g_sync_only}.
    assert out["groupNonSync"]["g"].to_list() == ["g_sync_only"]
    # Hand-calc: only inertia_constant rows with value!=0 ⇒ {(p1,n1)}.
    psi = out["process_sink_inertia"].sort(["p", "sink"])
    assert psi.height == 1
    assert psi.row(0) == ("p1", "n1")


def test_process_unit_filter_excludes_connections(tmp_path: Path):
    """Covers A14-process_unit_filter_excludes_connections.

    ``solve_data/process_unit.csv`` carries only ``p_unit``; the loader
    rename-selects to a 1-row ``(p,)`` frame that downstream
    capacityMargin LHS uses to filter out the connection process.
    """
    inp, sd = _make_dirs(tmp_path)
    _write(sd / "process_unit.csv", "process_unit\np_unit\n")
    out = _group_slack.load_data(inp, sd, dt=pl.DataFrame(), provider=_provider_for_workdir(inp, sd),
                                  nb=None, pss_eff=None, pss_noEff=None,
                                  p_unitsize=None)
    # Hand-calc: 1 row, column renamed to "p", value "p_unit".
    pu = out["process_unit"]
    assert pu is not None and pu.columns == ["p"]
    assert pu["p"].to_list() == ["p_unit"]


# ====================================================================
# A.15 — _reserve.load_data
# ====================================================================

def test_feature_gate_empty_rug_returns_blank(tmp_path: Path):
    """Covers A15-feature_gate_empty_rug_returns_blank.

    Header-only ``reserve__upDown__group.csv`` short-circuits load_data
    to an empty dict (no FlexData fields populated)."""
    inp, sd = _make_dirs(tmp_path)
    _write(sd / "reserve__upDown__group.csv", "reserve,upDown,group\n")
    out = _reserve.load_data(inp, sd, dt=None, provider=_provider_for_workdir(inp, sd))
    # Hand-calc: empty CSV ⇒ early return ⇒ {}.
    assert out == {}


def test_method_partition_three_way(tmp_path: Path):
    """Covers A15-method_partition_three_way.

    Each of three method CSVs holds a single distinct (r, ud, g) row;
    loader returns three named partitions, each with exactly its row.
    """
    inp, sd = _make_dirs(tmp_path)
    _write(sd / "reserve__upDown__group.csv",
           "reserve,upDown,group\nr1,up,g1\nr1,up,g2\nr1,up,g3\n")
    _write(sd / "reserve__upDown__group__method_timeseries.csv",
           "reserve,upDown,group,method\nr1,up,g1,timeseries\n")
    _write(sd / "reserve__upDown__group__method_dynamic.csv",
           "reserve,upDown,group,method\nr1,up,g2,dynamic\n")
    _write(sd / "reserve__upDown__group__method_n_1.csv",
           "reserve,upDown,group,method\nr1,up,g3,n_1\n")

    out = _reserve.load_data(inp, sd, dt=None, provider=_provider_for_workdir(inp, sd))
    # Hand-calc: each partition exactly one row, with its own group.
    assert out["reserve_upDown_group_method_timeseries"]["g"].to_list() == ["g1"]
    assert out["reserve_upDown_group_method_dynamic"]["g"].to_list() == ["g2"]
    assert out["reserve_upDown_group_method_n_1"]["g"].to_list() == ["g3"]


def test_pdtReserve_param_slice_and_dt_clip(tmp_path: Path):
    """Covers A15-pdtReserve_reservation_param_slice + A15-dt_clip_drops_out_of_horizon_rows.

    Two reservation rows on different (d, t) plus one ``other_param`` row;
    ``dt`` covers only one timestep.  Loader filters by param and inner-
    joins on dt.
    """
    inp, sd = _make_dirs(tmp_path)
    _write(sd / "reserve__upDown__group.csv",
           "reserve,upDown,group\nr1,up,g1\n")
    _write(sd / "pdtReserve_upDown_group.csv",
           "reserve,upDown,group,period,time,param,value\n"
           "r1,up,g1,d1,t1,reservation,50.0\n"
           "r1,up,g1,d1,t1,other_param,9.0\n"
           "r1,up,g1,d2,t1,reservation,77.0\n")
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})

    out = _reserve.load_data(inp, sd, dt=dt, provider=_provider_for_workdir(inp, sd))
    # Hand-calc: param=="reservation" ⇒ 2 rows; dt-join ⇒ keep only (d1,t1)
    # ⇒ 1 row, value=50.0.  other_param dropped, (d2,t1) clipped.
    frame = out["pdtReserve_upDown_group_reservation"].frame
    assert frame.height == 1
    assert frame["value"].to_list() == pytest.approx([50.0], rel=1e-7)
    assert frame.row(0)[:5] == ("r1", "up", "g1", "d1", "t1")


# ====================================================================
# A.19 — _commodity_ladder.load_data
# ====================================================================

def test_feature_gate_no_commodity_with_ladder_returns_blank(tmp_path: Path):
    """Covers A19-feature_gate_no_commodity_with_ladder_returns_blank.

    No ``commodity_with_ladder.csv`` on disk ⇒ load_data returns the
    all-None blank dict (every key present, every value None).
    """
    inp, sd = _make_dirs(tmp_path)
    out = _commodity_ladder.load_data(inp, sd, provider=_provider_for_workdir(inp, sd))
    # Hand-calc: missing CSV ⇒ blank dict (all values None).
    assert out["commodity_with_ladder"] is None
    assert all(v is None for v in out.values())
    # Spot-check that all expected keys are present (loader contract).
    for k in ("p_ladder_ann_price", "p_ladder_cum_quantity", "cndi_ladder",
              "ci_ladder_cumulative", "p_f_d_k", "p_ladder_cum_realized_mwh"):
        assert k in out


def test_tier_string_dtype_preservation(tmp_path: Path):
    """Covers A19-tier_string_dtype_preservation.

    Integer tier column in ``cndi_ladder_set.csv`` is cast to Utf8 so
    Param-table joins (also Utf8) line up.
    """
    inp, sd = _make_dirs(tmp_path)
    _write(sd / "commodity_with_ladder.csv", "commodity\nc1\n")
    _write(sd / "cndi_ladder_set.csv",
           "commodity,node,period,tier\nc1,n1,d1,1\nc1,n1,d1,2\n")
    out = _commodity_ladder.load_data(inp, sd, provider=_provider_for_workdir(inp, sd))
    cndi = out["cndi_ladder"]
    # Hand-calc: read int tier ⇒ cast to Utf8 ⇒ values ["1","2"].
    assert cndi["i"].dtype == pl.Utf8
    assert sorted(cndi["i"].to_list()) == ["1", "2"]
