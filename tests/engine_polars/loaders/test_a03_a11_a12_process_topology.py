"""Surface A.3 (Process Topology & Source-Sink Flow Routes), A.11
(Flow Profile Bounds) and A.12 (Multi-Flow Conversion / CHP) loader
tests.

Entry point: ``flextool.engine_polars.load_flextool(workdir)``.  These
sections share the ``process`` + ``process__source__sink`` foundation;
two consolidated tests cover the weakest indirect spots flagged in
``02_coverage_map.md``:

  * a 2-input/2-output indirect (CHP) overlay exercises A.12's
    ``process_indirect`` set, the input/output filters, the zero-coef
    anti-join, the non-default sink-coef Param path AND A.3's
    ``flow_to_n`` / ``flow_from_n`` keying + ``flow_to_commodity``
    sink-side join — one minimal CSV bundle pins all of it.
  * a 3-row ``process__source__sink__profile__profile_method.csv``
    paired with the legacy wide-format ``pdtProfile.csv`` exercises
    A.11's three-way method partition AND the wide-to-long unpivot in
    ``_read_wide_per_entity`` AND the ``existing_count`` rename
    contract.

The base seed (``data/work_base``) is dispatch-only with no processes;
to drive the CSV-only branch of ``_load_process_topology`` (so the
overlays are honoured rather than overridden by ``SpineDbReader``) the
seed's ``tests.sqlite`` is removed from the tmp_path copy first.
"""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flextool.engine_polars import load_flextool


def _strip_db(workdir):
    """Force ``_load_process_topology``'s CSV-only branch by removing the
    auto-resolved ``tests.sqlite`` from the tmp copy.  Without this the
    loader silently constructs a ``SpineDbReader`` from the seed DB and
    bypasses the overlays."""
    (workdir / "tests.sqlite").unlink()


def _write(workdir, rel: str, content: str) -> None:
    p = workdir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


# --- A.3 + A.12 -----------------------------------------------------

def test_chp_2in_2out_indirect_with_zero_coef_and_nondefault_sink_coef(tiny_workdir):
    """Covers A12-process_indirect_set_unique (direct) +
    A12-inputs_arcs_filter_sink_equals_p (direct) +
    A12-outputs_arcs_filter_source_equals_p (consolidated) +
    A12-source_flow_coef_zero_anti_join_drops_arc (direct) +
    A12-sink_flow_coef_nondefault_param_with_default_fill (direct) +
    A3-flow_to_n_and_flow_from_n_keying (consolidated) +
    A3-flow_to_commodity_join_sink_side (consolidated).

    Single CHP-style overlay: 2 input arcs (coal->u1 with non-default
    coef 1.5; gas->u1 with zero coef -> dropped), 2 output arcs
    (u1->heat with sink-coef 0.2; u1->west with sink-coef 2.0;
    ``west`` carries commodity ``elec`` so flow_to_commodity must pick
    up exactly the (u1,u1,west,elec) row).
    """
    _strip_db(tiny_workdir)
    _write(tiny_workdir, "solve_data/process_source_sink.csv",
        "process,source,sink\n"
        "u1,coal,u1\nu1,gas,u1\nu1,u1,heat\nu1,u1,west\n")
    _write(tiny_workdir, "solve_data/process_source_sink_eff.csv",
        "process,source,sink\n")
    _write(tiny_workdir, "solve_data/process_source_sink_noEff.csv",
        "process,source,sink\n"
        "u1,coal,u1\nu1,gas,u1\nu1,u1,heat\nu1,u1,west\n")
    _write(tiny_workdir, "solve_data/process_source.csv",
        "process,source\nu1,coal\nu1,gas\n")
    _write(tiny_workdir, "solve_data/process_sink.csv",
        "process,sink\nu1,heat\nu1,west\n")
    _write(tiny_workdir, "solve_data/process__method_indirect.csv",
        "process,method\nu1,method_indirect\n")
    _write(tiny_workdir, "input/commodity__node.csv",
        "commodity,node\nelec,west\n")
    _write(tiny_workdir, "input/p_process_source_flow_coefficient.csv",
        "process,source,p_process_source_flow_coefficient\n"
        "u1,gas,0.0\nu1,coal,1.5\n")
    _write(tiny_workdir, "input/p_process_sink_flow_coefficient.csv",
        "process,sink,p_process_sink_flow_coefficient\n"
        "u1,heat,0.2\nu1,west,2.0\n")
    # p_flow_max is empty header-only in the seed; provide ≥1 row so the
    # downstream Param init doesn't trip on a None frame.
    _write(tiny_workdir, "solve_data/p_flow_max.csv",
        "process,source,sink,period,time,value\n"
        "u1,coal,u1,p2020,t0001,5.0\n")

    d = load_flextool(tiny_workdir)

    # A.12 process_indirect: unique on process, single row.
    # Hand-calc: process__method_indirect has one process u1.
    assert d.process_indirect.sort("p")["p"].to_list() == ["u1"]
    # A.12 inputs filter (sink == p) with zero-coef anti-join on gas.
    # Hand-calc: pss inputs = {(u1,coal,u1),(u1,gas,u1)}; gas zero-coef
    # is anti-joined out -> single row.
    assert_frame_equal(
        d.process_input_flows.sort(["p", "source", "sink"]),
        pl.DataFrame({"p": ["u1"], "source": ["coal"], "sink": ["u1"]}))
    # A.12 outputs filter (source == p), no zero sink-coefs.
    # Hand-calc: pss outputs = {(u1,u1,heat),(u1,u1,west)} both kept.
    assert_frame_equal(
        d.process_output_flows.sort(["p", "source", "sink"]),
        pl.DataFrame({"p": ["u1", "u1"], "source": ["u1", "u1"],
                      "sink": ["heat", "west"]}))
    # A.12 source-coef Param: only coal survives (gas dropped); coef 1.5.
    src_coef = d.p_process_source_flow_coef.frame.sort("source")
    assert src_coef["value"].to_list() == pytest.approx([1.5], rel=1e-7)
    assert src_coef["source"].to_list() == ["coal"]
    # A.12 sink-coef Param: both surviving outputs carry their listed coef.
    # Hand-calc: sink_long has both heat=0.2 and west=2.0 (no fill_null).
    sink_coef = d.p_process_sink_flow_coef.frame.sort("sink")
    assert sink_coef["value"].to_list() == pytest.approx([0.2, 2.0], rel=1e-7)
    assert sink_coef["sink"].to_list() == ["heat", "west"]
    # A.3 flow_to_n keys n=sink; flow_from_n keys n=source — pure projection.
    # Hand-calc: from the 4 pss rows, flow_to_n's n is exactly the sink
    # column, flow_from_n's n is exactly the source column.
    ftn = d.flow_to_n.sort(["p", "source", "sink"])
    assert ftn["n"].to_list() == ftn["sink"].to_list()
    ffn = d.flow_from_n.sort(["p", "source", "sink"])
    assert ffn["n"].to_list() == ffn["source"].to_list()
    # A.3 flow_to_commodity: inner-join pss on sink==node with
    # commodity__node.  Hand-calc: only (u1,u1,west) matches elec/west.
    assert_frame_equal(
        d.flow_to_commodity.sort(["p", "source", "sink", "c"]),
        pl.DataFrame({"p": ["u1"], "source": ["u1"],
                      "sink": ["west"], "c": ["elec"]}))


# --- A.11 -----------------------------------------------------------

def test_profile_method_partition_with_legacy_wide_pdtProfile(tiny_workdir):
    """Covers A11-profile_lower_and_fixed_partition (direct) +
    A11-profile_upper_method_extracts_upper_only (consolidated) +
    A11-profile_value_long_via_wide_pdt (direct) +
    A11-existing_count_equals_cap_pd_when_no_invest (direct).

    Three rows in ``process__source__sink__profile__profile_method.csv``
    with disjoint methods + profiles drive the partition; a single-(d,t)
    legacy-wide ``pdtProfile.csv`` (with the ``solve`` column header)
    forces the unpivot branch in ``_read_wide_per_entity``; u1's
    existing capacity (200) divided by unitsize (10) pins
    ``p_process_existing_count`` at 20.
    """
    _strip_db(tiny_workdir)
    _write(tiny_workdir, "solve_data/process_source_sink.csv",
        "process,source,sink\nu1,west,heat\n")
    _write(tiny_workdir, "solve_data/process_source_sink_eff.csv",
        "process,source,sink\nu1,west,heat\n")
    _write(tiny_workdir, "solve_data/process_source_sink_noEff.csv",
        "process,source,sink\n")
    _write(tiny_workdir, "solve_data/process_source.csv",
        "process,source\nu1,west\n")
    _write(tiny_workdir, "solve_data/process_sink.csv",
        "process,sink\nu1,heat\n")
    _write(tiny_workdir, "solve_data/process__source__sink__profile__profile_method.csv",
        "process,source,sink,profile,method\n"
        "u1,west,heat,prof_a,upper_limit\n"
        "u1,west,heat,prof_b,lower_limit\n"
        "u1,west,heat,prof_c,fixed\n")
    # Legacy wide-format pdtProfile (header has ``solve`` column);
    # _read_wide_per_entity must drop ``solve`` and unpivot the three
    # profile columns.
    _write(tiny_workdir, "solve_data/pdtProfile.csv",
        "solve,period,time,prof_a,prof_b,prof_c\n"
        "s1,p2020,t0001,0.5,0.7,0.3\n")
    _write(tiny_workdir, "solve_data/p_entity_unitsize.csv",
        "entity,value\nwest,1000.0\nu1,10.0\n")
    _write(tiny_workdir, "solve_data/p_entity_all_existing.csv",
        "entity,period,value\nwest,p2020,0.0\nu1,p2020,200.0\n")
    _write(tiny_workdir, "solve_data/p_flow_max.csv",
        "process,source,sink,period,time,value\n"
        "u1,west,heat,p2020,t0001,5.0\n")

    d = load_flextool(tiny_workdir)

    # A.11 partition: each method extracts exactly its single row.
    # Hand-calc: filter on method == upper_limit yields just (u1,west,heat,prof_a).
    assert_frame_equal(
        d.process_profile_upper.sort(["p", "f"]),
        pl.DataFrame({"p": ["u1"], "source": ["west"],
                      "sink": ["heat"], "f": ["prof_a"]}))
    assert_frame_equal(
        d.process_profile_lower.sort(["p", "f"]),
        pl.DataFrame({"p": ["u1"], "source": ["west"],
                      "sink": ["heat"], "f": ["prof_b"]}))
    assert_frame_equal(
        d.process_profile_fixed.sort(["p", "f"]),
        pl.DataFrame({"p": ["u1"], "source": ["west"],
                      "sink": ["heat"], "f": ["prof_c"]}))
    # A.11 wide-to-long unpivot: 1 (d,t) x 3 profile cols -> 3 rows.
    # Hand-calc: drop solve, unpivot {prof_a:0.5,prof_b:0.7,prof_c:0.3}.
    pv = d.p_profile_value.frame.sort("f")
    assert pv["f"].to_list() == ["prof_a", "prof_b", "prof_c"]
    assert pv["value"].to_list() == pytest.approx([0.5, 0.7, 0.3], rel=1e-7)
    # A.11 existing_count rename of cap_pd: cap=200, unitsize=10 -> 20.
    ec = d.p_process_existing_count.frame.sort(["p", "d"])
    assert ec["value"].to_list() == pytest.approx([20.0], rel=1e-7)
    assert ec["p"].to_list() == ["u1"]
