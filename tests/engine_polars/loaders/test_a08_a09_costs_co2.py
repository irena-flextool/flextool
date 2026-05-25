"""Surface A.8 (variable / fixed operating-cost loader) and A.9 (CO2 cap
loader) tests.

A.8 covers ``flextool.engine_polars.input._load_varcost`` — the helper
that reads up to four ``pssdt_varCost_*.csv`` set files plus three
wide ``pdt*`` Param files and returns the eight-key dict consumed by
the cost-emit pass.  Each helper returns ``None`` independently per
field so a single source-only / sink-only / process-only fixture
suffices to verify the disjoint-routing contract.

A.9 covers ``flextool.engine_polars.input._load_co2_cap`` — the loader
behind the per-period CO2-cap constraint family.  Tests focus on the
short-circuit gates (no-pss, missing CSV, empty CSV, no priced-node
intersection) that decide whether the constraint family fires at all;
the live-path joins are exercised by adjacent tests / goldens already.

Both helpers are pure: hand-built ``Path`` directories + small CSV
overlays isolate one transformation per test (mirroring the A.6 / A.7
patterns).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars.input import _load_varcost, _load_co2_cap


# --- helpers --------------------------------------------------------

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    return inp, sd


def _pss_one_row() -> pl.DataFrame:
    """Minimal pss frame; _load_varcost only checks `is not None`."""
    return pl.DataFrame({"p": ["p1"], "source": ["n_src"], "sink": ["n_snk"]})


# ====================================================================
# A.8 — _load_varcost
# ====================================================================

def test_pssdt_varCost_eff_partitions_disjoint(tmp_path: Path):
    """Covers A8-pssdt_varCost_eff_unit_source_disjoint
    + A8-pssdt_varCost_eff_unit_sink_routing
    + A8-pssdt_varCost_eff_connection_routing.

    Populate all three eff-side ``pssdt_varCost_eff_*`` set files with
    one row each; leave ``pssdt_varCost_noEff.csv`` absent.  Each
    ``_read_pssdt_set`` call reads its own file independently so the
    three sets land on disjoint keys without cross-contamination.
    """
    _, sd = _make_dirs(tmp_path)
    _write(sd / "pssdt_varCost_eff_unit_source.csv",
           "process,source,sink,period,time\np1,n_src,p1,d1,t1\n")
    _write(sd / "pssdt_varCost_eff_unit_sink.csv",
           "process,source,sink,period,time\np1,p1,n_snk,d1,t1\n")
    _write(sd / "pssdt_varCost_eff_connection.csv",
           "process,source,sink,period,time\nc1,n1,n2,d1,t1\n")

    out = _load_varcost(sd, _pss_one_row())
    # Hand-calc: each _read_pssdt_set hits its own file -> height=1 each;
    # noEff file absent -> None (no cross-fill).
    assert out["pssdt_varCost_noEff"] is None
    for k in ("pssdt_varCost_eff_unit_source",
              "pssdt_varCost_eff_unit_sink",
              "pssdt_varCost_eff_connection"):
        df = out[k]
        assert df is not None and df.height == 1, f"{k} not height=1"
        assert df.columns == ["p", "source", "sink", "d", "t"]
    # Spot-check the source-side row keys to confirm no cross-leakage.
    assert out["pssdt_varCost_eff_unit_source"].row(0) == (
        "p1", "n_src", "p1", "d1", "t1")
    assert out["pssdt_varCost_eff_connection"].row(0) == (
        "c1", "n1", "n2", "d1", "t1")


def test_p_pssdt_varCost_drops_zeros_and_returns_none_when_all_zero(
        tmp_path: Path):
    """Covers A8-p_pssdt_varCost_drops_zero_rows
    + A8-p_pssdt_varCost_returns_none_when_all_zero.

    Two passes share the same minimal ``pdtProcess__source__sink__dt_varCost.csv``
    schema: first with one nonzero + one zero row (zero dropped, Param
    height=1); second with a single zero row (post-filter height=0 -> None).
    """
    _, sd = _make_dirs(tmp_path)
    # (1) mixed: 2.5 at t1, 0.0 at t2 -> filter drops zero, height=1.
    _write(sd / "pdtProcess__source__sink__dt_varCost.csv",
           "process,source,sink,period,time,value\n"
           "p1,n_src,n_snk,d1,t1,2.5\n"
           "p1,n_src,n_snk,d1,t2,0.0\n")
    out1 = _load_varcost(sd, _pss_one_row())
    # Hand-calc: cast Float64, fill_null(0), filter !=0 -> 1 row at t1=2.5.
    p = out1["p_pssdt_varCost"]
    assert p is not None
    fr = p.frame
    assert fr.height == 1
    assert fr["t"][0] == "t1"
    assert fr["value"][0] == pytest.approx(2.5, rel=1e-7)

    # (2) all-zero: post-filter height=0 -> None (no degenerate Param).
    _write(sd / "pdtProcess__source__sink__dt_varCost.csv",
           "process,source,sink,period,time,value\n"
           "p1,n_src,n_snk,d1,t1,0.0\n")
    out2 = _load_varcost(sd, _pss_one_row())
    # Hand-calc: filter !=0 -> height=0 -> the `if sliced.height > 0` gate
    # never wraps a Param.
    assert out2["p_pssdt_varCost"] is None


def test_pdt_varCost_three_legs_independent(tmp_path: Path):
    """Covers A8-pdt_varCost_source_filters_param_value
    + A8-pdt_varCost_sink_routing_independent
    + A8-pdt_varCost_process_no_side_dim.

    Populate all three wide param files in one pass:
      * ``pdtProcess_source.csv`` mixes ``other_operational_cost`` + a
        sibling ``inertia_constant`` row to verify the param-name filter.
      * ``pdtProcess_sink.csv`` carries a single sink-side cost.
      * ``pdtProcess.csv`` carries a single process-level (no source/sink
        dim) cost.
    Each ``_slice_pds`` / ``pdtProcess`` block emits its own Param with
    the documented key arity; sibling param rows must NOT bleed across.
    """
    _, sd = _make_dirs(tmp_path)
    _write(sd / "pdtProcess_source.csv",
           "process,source,period,time,param,value\n"
           "p1,n_src,d1,t1,other_operational_cost,4.0\n"
           "p1,n_src,d1,t1,inertia_constant,2.0\n")
    _write(sd / "pdtProcess_sink.csv",
           "process,sink,period,time,param,value\n"
           "p1,n_snk,d1,t1,other_operational_cost,3.0\n")
    _write(sd / "pdtProcess.csv",
           "process,period,time,param,value\n"
           "c1,d1,t1,other_operational_cost,7.0\n")

    out = _load_varcost(sd, _pss_one_row())
    # Hand-calc: source slice keeps only param=='other_operational_cost'
    # -> 1 row, value=4.0; key arity 4 (p,source,d,t).
    src = out["p_pdt_varCost_source"]
    assert src is not None and src.frame.height == 1
    assert src.dims == ("p", "source", "d", "t")
    assert src.frame["value"][0] == pytest.approx(4.0, rel=1e-7)
    # Hand-calc: sink slice -> 1 row, value=3.0; key arity 4 (p,sink,d,t).
    snk = out["p_pdt_varCost_sink"]
    assert snk is not None and snk.frame.height == 1
    assert snk.dims == ("p", "sink", "d", "t")
    assert snk.frame["value"][0] == pytest.approx(3.0, rel=1e-7)
    # Hand-calc: process-level slice -> 1 row, value=7.0; key arity 3.
    proc = out["p_pdt_varCost_process"]
    assert proc is not None and proc.frame.height == 1
    assert proc.dims == ("p", "d", "t")
    assert proc.frame["value"][0] == pytest.approx(7.0, rel=1e-7)


def test_blank_when_pss_none(tmp_path: Path):
    """Covers A8-blank_when_pss_none.

    With ``pss=None`` the helper must return the all-None blank dict
    BEFORE any CSV read — even when every CSV exists on disk.  Demonstrates
    the upstream-empty gate that prevents seeding a varCost universe
    when no process source/sink universe exists.
    """
    _, sd = _make_dirs(tmp_path)
    # Populate every CSV the live path would consume; the gate must skip
    # ALL of them.
    _write(sd / "pssdt_varCost_noEff.csv",
           "process,source,sink,period,time\np1,n_src,n_snk,d1,t1\n")
    _write(sd / "pdtProcess__source__sink__dt_varCost.csv",
           "process,source,sink,period,time,value\np1,n_src,n_snk,d1,t1,9.0\n")
    _write(sd / "pdtProcess.csv",
           "process,period,time,param,value\np1,d1,t1,other_operational_cost,9.0\n")

    out = _load_varcost(sd, pss=None)
    # Hand-calc: pss is None -> blank dict; all 8 keys present and None.
    expected_keys = {
        "pssdt_varCost_noEff",
        "pssdt_varCost_eff_unit_source",
        "pssdt_varCost_eff_unit_sink",
        "pssdt_varCost_eff_connection",
        "p_pssdt_varCost",
        "p_pdt_varCost_source",
        "p_pdt_varCost_sink",
        "p_pdt_varCost_process",
    }
    assert set(out.keys()) == expected_keys
    assert all(v is None for v in out.values())


# ====================================================================
# A.9 — _load_co2_cap
# ====================================================================

def test_co2_cap_blank_shortcuts(tmp_path: Path):
    """Covers A9-co2_cap_blank_when_no_pss
    + A9-co2_cap_missing_group_max_period_csv
    + A9-co2_cap_empty_group_max_period_csv.

    Three blank-shortcut paths sharing one fixture; each returns the
    canonical 5-tuple of Nones.

    1. ``pss_eff=None`` AND ``pss_noEff=None`` -> early gate (no disk).
    2. ``pss_eff`` populated but ``group_co2_max_period.csv`` absent.
    3. ``group_co2_max_period.csv`` exists but height=0.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss_eff = pl.DataFrame({"p": ["p1"], "source": ["n1"], "sink": ["n2"]})

    # (1) both pss are None -> earliest gate.
    out1 = _load_co2_cap(inp, sd, None, dt, pss_noEff=None)
    # Hand-calc: pss_eff is None and pss_noEff is None -> 5 Nones.
    assert out1 == (None, None, None, None, None)

    # (2) pss_eff populated but no group_co2_max_period.csv on disk.
    out2 = _load_co2_cap(inp, sd, pss_eff, dt)
    # Hand-calc: file absent -> 5 Nones.
    assert out2 == (None, None, None, None, None)

    # (3) header-only group_co2_max_period.csv -> g_max.height==0 gate.
    _write(sd / "group_co2_max_period.csv", "group\n")
    out3 = _load_co2_cap(inp, sd, pss_eff, dt)
    # Hand-calc: g_max.height==0 -> 5 Nones (distinct code path from (2)).
    assert out3 == (None, None, None, None, None)


def test_co2_cap_no_priced_node_intersection_returns_none(tmp_path: Path):
    """Covers A9-co2_cap_no_priced_node_intersection_returns_none.

    ``group_co2_max_period.csv`` names group g1; ``group__node.csv``
    maps g1 -> n2; ``commodity_node_co2.csv`` covers n1 only.  The
    3-way join produces ``gcn`` height=0 -> the helper must return
    the 5-tuple of Nones rather than emitting an empty constraint.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss_eff = pl.DataFrame({"p": ["p1"], "source": ["n1"], "sink": ["n2"]})
    _write(sd / "group_co2_max_period.csv", "group\ng1\n")
    _write(inp / "group__node.csv", "group,node\ng1,n2\n")
    _write(sd / "commodity_node_co2.csv", "commodity,node\nc1,n1\n")

    out = _load_co2_cap(inp, sd, pss_eff, dt)
    # Hand-calc: g1 -> n2 (group__node), CO2 priced n=n1 only;
    # inner join on n -> gcn height=0 -> 5 Nones.
    assert out == (None, None, None, None, None)
