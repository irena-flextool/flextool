"""Surface A.8 (variable / fixed operating-cost loader) and A.9 (CO2 cap
loader) tests.

A.8 covers ``flextool.engine.preprocess.flex_data._load_varcost`` — the helper
that reads the three wide ``pdt*`` Param files and returns the three per-side
variable-cost Params (``p_pdt_varCost_{source,sink,process}``).  Each leg
returns ``None`` independently, so a single source-only / sink-only /
process-only fixture suffices to verify the disjoint-routing contract.  The
four ``pssdt_varCost_*`` index sets and ``p_pssdt_varCost`` are not seeded by
the loader: the DB overlay builds them (``_direct.apply_varcost_params``,
``tests/engine/test_varcost_index_sets.py``), and a reader-less CSV reload
seeds them in ``flextool.engine._csv_reload._field_seeds``; the A8 checks of
those reads are with the reload hook (``tests/csv_reload/test_dump_seeds.py``).

A.9 covers ``flextool.engine.preprocess.flex_data._load_co2_cap`` — the loader
behind the per-period CO2-cap constraint family.  Tests focus on the
short-circuit gates (no-pss, missing CSV, empty CSV, no priced-node
intersection) that decide whether the constraint family fires at all;
the live-path joins are exercised by adjacent tests / goldens already.

Both helpers are pure: hand-built ``Path`` directories + small CSV
overlays, passed as a provider (``tests._support.provider.provider_from_csv_dirs``),
isolate one transformation per test (mirroring the A.6 / A.7 patterns).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.flex_data import _load_varcost, _load_co2_cap
from tests._support.provider import provider_from_csv_dirs


# --- helpers --------------------------------------------------------

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir()
    sd.mkdir()
    return inp, sd


def _pss_one_row() -> pl.DataFrame:
    """Minimal pss frame; _load_varcost only checks `is not None`."""
    return pl.DataFrame({"p": ["p1"], "source": ["n_src"], "sink": ["n_snk"]})


# ====================================================================
# A.8 — _load_varcost
# ====================================================================

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

    out = _load_varcost(sd, _pss_one_row(), provider=provider_from_csv_dirs(solve_data_dir=sd))
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
    _write(sd / "pdtProcess.csv",
           "process,period,time,param,value\np1,d1,t1,other_operational_cost,9.0\n")

    out = _load_varcost(sd, pss=None, provider=provider_from_csv_dirs(solve_data_dir=sd))
    # Hand-calc: pss is None -> blank dict; all 3 keys present and None.
    expected_keys = {
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
    canonical 6-tuple of Nones (c18a added the removal-credit frame,
    ``flow_to_co2_capped``, as the 6th element).

    1. ``pss_eff=None`` AND ``pss_noEff=None`` -> early gate (no disk).
    2. ``pss_eff`` populated but ``group_co2_max_period.csv`` absent.
    3. ``group_co2_max_period.csv`` exists but height=0.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss_eff = pl.DataFrame({"p": ["p1"], "source": ["n1"], "sink": ["n2"]})

    # (1) both pss are None -> earliest gate.
    out1 = _load_co2_cap(inp, sd, None, dt, pss_noEff=None,
                         provider=provider_from_csv_dirs(inp, sd))
    # Hand-calc: pss_eff is None and pss_noEff is None -> 6 Nones.
    assert out1 == (None, None, None, None, None, None)

    # (2) pss_eff populated but no group_co2_max_period.csv on disk.
    out2 = _load_co2_cap(inp, sd, pss_eff, dt, provider=provider_from_csv_dirs(inp, sd))
    # Hand-calc: file absent -> 6 Nones.
    assert out2 == (None, None, None, None, None, None)

    # (3) header-only group_co2_max_period.csv -> g_max.height==0 gate.
    _write(sd / "group_co2_max_period.csv", "group\n")
    out3 = _load_co2_cap(inp, sd, pss_eff, dt, provider=provider_from_csv_dirs(inp, sd))
    # Hand-calc: g_max.height==0 -> 6 Nones (distinct code path from (2)).
    assert out3 == (None, None, None, None, None, None)


def test_co2_cap_no_priced_node_intersection_returns_none(tmp_path: Path):
    """Covers A9-co2_cap_no_priced_node_intersection_returns_none.

    ``group_co2_max_period.csv`` names group g1; ``group__node.csv``
    maps g1 -> n2; ``commodity_node_co2.csv`` covers n1 only.  The
    3-way join produces ``gcn`` height=0 -> the helper must return
    the 6-tuple of Nones (c18a's ``flow_to_co2_capped`` 6th element included)
    rather than emitting an empty constraint.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss_eff = pl.DataFrame({"p": ["p1"], "source": ["n1"], "sink": ["n2"]})
    _write(sd / "group_co2_max_period.csv", "group\ng1\n")
    _write(inp / "group__node.csv", "group,node\ng1,n2\n")
    _write(sd / "commodity_node_co2.csv", "commodity,node\nc1,n1\n")

    out = _load_co2_cap(inp, sd, pss_eff, dt, provider=provider_from_csv_dirs(inp, sd))
    # Hand-calc: g1 -> n2 (group__node), CO2 priced n=n1 only;
    # inner join on n -> gcn height=0 -> 6 Nones.
    assert out == (None, None, None, None, None, None)
