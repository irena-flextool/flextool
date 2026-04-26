"""Unit tests for the rolling CO2 accumulator handoff writer.

Mirrors ``test_cumulative_handoffs.py``'s structure for the ladder
side: the writer's correctness hinges on several pieces (group
detection, noEff vs eff discrimination, per-period realized-tonnes
direct aggregation, prior-accumulator carryover) so the tests exercise
each with a mocked HiGHS instance and hand-crafted on-disk parameter
CSVs.

End-to-end validation (real rolling solve, within-period rolling
against ``co2_max_total``) is out of scope for this unit-test file
and should be covered by the scenario-level regression suite.

Test matrix:
    * Group detection via ``group__co2_method.csv``.
    * Prior accumulator round-trip (first solve → empty; subsequent
      → dict).
    * First-solve empty-prior path: direct aggregation of noEff flow
      over realized (d, t) pairs.
    * Realized-subset filtering: lookahead (d, t) contributes zero.
    * Prior + this-roll carryover: second-solve sum.
    * Removals: sink-side flow subtracts from the accumulator.
    * No-CO2-groups fast path: header-only CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from flextool.process_outputs.cumulative_handoffs import (
    _attribute_groups,
    _co2_tonnes_this_roll,
    _load_co2_max_total_groups,
    _load_commodity_co2_content,
    _load_commodity_node_co2,
    _load_entity_unitsize,
    _load_group_node,
    _load_prior_co2_cum_realized_tonnes,
    _load_process_source_sink_partition,
    _load_process_unit_set,
    write_co2_rolling_accumulators,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_highs(
    variable_names: list[str], col_values: list[float],
) -> MagicMock:
    h = MagicMock()
    h.allVariableNames.return_value = list(variable_names)
    sol = SimpleNamespace(col_value=list(col_values), col_dual=[], row_dual=[])
    h.getSolution.return_value = sol
    h.getLp.return_value = SimpleNamespace(row_names_=[])
    return h


def _make_workfolder(tmp_path: Path, *, first_solve: bool = True) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "solve_data").mkdir()
    (tmp_path / "input" / "p_model.csv").write_text(
        f"modelParam,p_model\nsolveFirst,{1 if first_solve else 0}\nsolveLast,1\n"
    )
    return tmp_path


def _write_group_co2_method(work: Path, rows: list[tuple[str, str]]) -> None:
    with open(work / "input" / "group__co2_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "co2_method"])
        for g, m in rows:
            w.writerow([g, m])


def _write_commodity_co2_content(work: Path, content: dict[str, float]) -> None:
    cols = list(content.keys())
    with open(work / "input" / "p_commodity_co2_content.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity"] + cols)
        w.writerow(["value"] + [content[c] for c in cols])


def _write_set_commodity_node_co2(
    work: Path, pairs: list[tuple[str, str]],
) -> None:
    with open(work / "solve_data" / "commodity_node_co2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity", "node"])
        for c, n in pairs:
            w.writerow([c, n])


def _write_set_group_node(
    work: Path, pairs: list[tuple[str, str]],
) -> None:
    with open(work / "solve_data" / "group_node.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "node"])
        for g, n in pairs:
            w.writerow([g, n])


def _write_process_source_sink_partition(
    work: Path,
    *,
    noeff: list[tuple[str, str, str]],
    eff: list[tuple[str, str, str]] | None = None,
) -> None:
    with open(
        work / "solve_data" / "process_source_sink_noEff.csv", "w", newline="",
    ) as f:
        w = csv.writer(f)
        w.writerow(["process", "source", "sink"])
        for r in noeff:
            w.writerow(r)
    with open(
        work / "solve_data" / "process_source_sink_eff.csv", "w", newline="",
    ) as f:
        w = csv.writer(f)
        w.writerow(["process", "source", "sink"])
        for r in (eff or []):
            w.writerow(r)


def _write_entity_unitsize(work: Path, us: dict[str, float]) -> None:
    cols = list(us.keys())
    with open(work / "input" / "p_entity_unitsize.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity"] + cols)
        w.writerow(["value"] + [us[c] for c in cols])


def _write_realized_dispatch(
    work: Path, pairs: list[tuple[str, str]],
) -> None:
    with open(work / "solve_data" / "realized_dispatch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "step"])
        for d, t in pairs:
            w.writerow([d, t])


def _write_steps_in_use(
    work: Path, rows: list[tuple[str, str, float]],
) -> None:
    with open(work / "solve_data" / "steps_in_use.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "step", "step_duration"])
        for d, t, dur in rows:
            w.writerow([d, t, dur])


def _write_rp_cost_weight(
    work: Path, rows: list[tuple[str, str, str, float]],
) -> None:
    with open(work / "solve_data" / "p_rp_cost_weight.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solve", "period", "time", "value"])
        for s, d, t, v in rows:
            w.writerow([s, d, t, v])


def _read_co2(path: Path) -> dict[tuple[str, str], float]:
    rows = list(csv.DictReader(open(path)))
    return {
        (r["group"], r["period"]): float(r["p_co2_cum_realized_tonnes"])
        for r in rows
    }


def _standard_co2_setup(
    work: Path,
    *,
    groups: list[str] = ("co2_system",),
    co2_content: dict[str, float] | None = None,
    commodity_node: list[tuple[str, str]] | None = None,
    group_node: list[tuple[str, str]] | None = None,
    noeff: list[tuple[str, str, str]] | None = None,
    eff: list[tuple[str, str, str]] | None = None,
    unitsize: dict[str, float] | None = None,
    realized_pairs: list[tuple[str, str]] | None = None,
    horizon_pairs: list[tuple[str, str, float]] | None = None,
    rp_weights: list[tuple[str, str, str, float]] | None = None,
    solve: str = "s1",
) -> None:
    """Seed every CSV the writer needs."""
    _write_group_co2_method(
        work, [(g, "total") for g in groups],
    )
    _write_commodity_co2_content(
        work, co2_content or {"Coal": 0.34},
    )
    _write_set_commodity_node_co2(
        work, commodity_node or [("Coal", "coal_market")],
    )
    _write_set_group_node(
        work, group_node or [(groups[0], "coal_market")],
    )
    _write_process_source_sink_partition(
        work, noeff=noeff or [], eff=eff or [],
    )
    _write_entity_unitsize(work, unitsize or {"coal_plant": 1.0})
    _write_realized_dispatch(work, realized_pairs or [])
    _write_steps_in_use(work, horizon_pairs or [])
    if rp_weights is None:
        # Default weight 1.0 for every horizon step.
        rp_weights = [
            (solve, d, t, 1.0) for d, t, _dur in (horizon_pairs or [])
        ]
    _write_rp_cost_weight(work, rp_weights)
    # Minimal process_unit.csv (empty — no process_unit correction).
    with open(work / "solve_data" / "process_unit.csv", "w", newline="") as f:
        csv.writer(f).writerow(["process"])


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------


def test_load_co2_max_total_groups_picks_total_methods(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_group_co2_method(
        work,
        [
            ("co2_system", "total"),          # included
            ("co2_eu", "price_total"),        # included
            ("co2_period", "period"),         # excluded (period only)
            ("co2_priced", "price"),          # excluded
            ("co2_both", "period_total"),     # included
        ],
    )
    assert _load_co2_max_total_groups(work) == {
        "co2_system", "co2_eu", "co2_both",
    }


def test_load_co2_max_total_groups_missing_file(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    assert _load_co2_max_total_groups(work) == set()


def test_load_commodity_co2_content_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_commodity_co2_content(work, {"Coal": 0.34, "Gas": 0.2})
    assert _load_commodity_co2_content(work) == {"Coal": 0.34, "Gas": 0.2}


def test_load_commodity_node_co2_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_set_commodity_node_co2(
        work, [("Coal", "coal_market"), ("Gas", "gas_market")],
    )
    assert _load_commodity_node_co2(work) == {
        ("Coal", "coal_market"), ("Gas", "gas_market"),
    }


def test_load_group_node_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_set_group_node(
        work,
        [
            ("co2_system", "coal_market"),
            ("co2_system", "gas_market"),
            ("co2_eu", "coal_market"),
        ],
    )
    got = _load_group_node(work)
    assert got == {
        "co2_system": {"coal_market", "gas_market"},
        "co2_eu": {"coal_market"},
    }


def test_load_process_source_sink_partition_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_process_source_sink_partition(
        work,
        noeff=[("p1", "n1", "n2"), ("p2", "n2", "n3")],
        eff=[("p3", "n3", "n4")],
    )
    noe, eff = _load_process_source_sink_partition(work)
    assert noe == {("p1", "n1", "n2"), ("p2", "n2", "n3")}
    assert eff == {("p3", "n3", "n4")}


def test_load_entity_unitsize_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_entity_unitsize(work, {"coal_plant": 400.0, "wind_plant": 100.0})
    assert _load_entity_unitsize(work) == {
        "coal_plant": 400.0, "wind_plant": 100.0,
    }


def test_load_process_unit_set_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    with open(work / "solve_data" / "process_unit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["process"])
        w.writerow(["coal_plant"])
        w.writerow(["wind_plant"])
    assert _load_process_unit_set(work) == {"coal_plant", "wind_plant"}


def test_load_prior_co2_cum_realized_tonnes_header_only(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "co2_cum_realized_tonnes.csv"
    path.write_text("group,period,p_co2_cum_realized_tonnes\n")
    assert _load_prior_co2_cum_realized_tonnes(path) == {}


def test_load_prior_co2_cum_realized_tonnes_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "co2_cum_realized_tonnes.csv"
    path.write_text(
        "group,period,p_co2_cum_realized_tonnes\n"
        "co2_system,p2020,123.4\nco2_system,p2025,99.5\n"
    )
    assert _load_prior_co2_cum_realized_tonnes(path) == {
        ("co2_system", "p2020"): 123.4,
        ("co2_system", "p2025"): 99.5,
    }


# ---------------------------------------------------------------------------
# _attribute_groups
# ---------------------------------------------------------------------------


def test_attribute_groups_uses_group_node_intersection() -> None:
    groups = {"g1", "g2", "g3"}
    group_node = {
        "g1": {"n1", "n2"},
        "g2": {"n2"},
        "g3": {"n3"},
    }
    # n2 → g1 and g2 (but not g3)
    assert _attribute_groups("n2", groups, group_node) == {"g1", "g2"}
    # n3 → only g3
    assert _attribute_groups("n3", groups, group_node) == {"g3"}


def test_attribute_groups_fallback_when_group_node_empty() -> None:
    groups = {"g1", "g2"}
    # Empty group_node → attribute to every CO2 group (conservative).
    assert _attribute_groups("n1", groups, {}) == groups


# ---------------------------------------------------------------------------
# _co2_tonnes_this_roll
# ---------------------------------------------------------------------------


def _v_flow_frame(
    rows: list[tuple[str, str, str, str, str, float]],
    solve: str = "s1",
) -> pd.DataFrame:
    """Build a v_flow-style wide DataFrame.

    ``rows`` = [(period, time, process, source, sink, value), ...].
    Returns a DF with row index (solve, period, time) and column
    MultiIndex (process, source, sink) — matches ``extract_variable``
    output for v_flow.
    """
    seen_cols: list[tuple[str, str, str]] = []
    seen_col_set: set[tuple[str, str, str]] = set()
    seen_rows: list[tuple[str, str, str]] = []
    seen_row_set: set[tuple[str, str, str]] = set()
    data: dict[tuple[str, str, str], dict[tuple[str, str, str], float]] = {}
    for period, time, p, s, k, v in rows:
        col = (p, s, k)
        row = (solve, period, time)
        if col not in seen_col_set:
            seen_cols.append(col)
            seen_col_set.add(col)
        if row not in seen_row_set:
            seen_rows.append(row)
            seen_row_set.add(row)
        data.setdefault(row, {})[col] = v
    if not seen_rows or not seen_cols:
        return pd.DataFrame()
    matrix = [
        [data.get(r, {}).get(c, 0.0) for c in seen_cols]
        for r in seen_rows
    ]
    return pd.DataFrame(
        matrix,
        index=pd.MultiIndex.from_tuples(
            seen_rows, names=["solve", "period", "time"],
        ),
        columns=pd.MultiIndex.from_tuples(
            seen_cols, names=["process", "source", "sink"],
        ),
    )


def test_co2_tonnes_this_roll_noeff_branch_direct() -> None:
    """v_flow on a noEff (p, commodity_node, sink) triple with 1 MWh/h
    for 2 realized hours → 2 MWh → 0.68 tonnes (Coal co2_content=0.34,
    already ÷1000)."""
    df = _v_flow_frame(
        [
            ("p2020", "t0001", "coal_plant", "coal_market", "el", 1.0),
            ("p2020", "t0002", "coal_plant", "coal_market", "el", 1.0),
        ],
    )
    # Scale by unitsize = 1.0 — already in place.
    got = _co2_tonnes_this_roll(
        df,
        realized_set={("p2020", "t0001"), ("p2020", "t0002")},
        co2_groups={"co2_system"},
        co2_content={"Coal": 0.34},
        commodity_node_co2={("Coal", "coal_market")},
        group_node={"co2_system": {"coal_market"}},
        noeff_set={("coal_plant", "coal_market", "el")},
        eff_set=set(),
        process_unit_set=set(),
        process_source_flow_coeff={},
        process_sink_flow_coeff={},
        step_duration={("p2020", "t0001"): 1.0, ("p2020", "t0002"): 1.0},
        rp_weight={("p2020", "t0001"): 1.0, ("p2020", "t0002"): 1.0},
        slope={},
    )
    # 2 MWh × 0.34 / 1000 = 0.00068 tonnes
    assert got == pytest.approx({("co2_system", "p2020"): 0.00068})


def test_co2_tonnes_this_roll_skips_lookahead() -> None:
    """v_flow on a lookahead (d, t) pair (not in realized_set) contributes
    zero — matches the writer's 'realized-only' attribution."""
    df = _v_flow_frame(
        [
            ("p2020", "t0001", "coal_plant", "coal_market", "el", 1.0),
            ("p2020", "t0002", "coal_plant", "coal_market", "el", 1.0),
        ],
    )
    got = _co2_tonnes_this_roll(
        df,
        realized_set={("p2020", "t0001")},  # t0002 is lookahead
        co2_groups={"co2_system"},
        co2_content={"Coal": 0.34},
        commodity_node_co2={("Coal", "coal_market")},
        group_node={"co2_system": {"coal_market"}},
        noeff_set={("coal_plant", "coal_market", "el")},
        eff_set=set(),
        process_unit_set=set(),
        process_source_flow_coeff={},
        process_sink_flow_coeff={},
        step_duration={("p2020", "t0001"): 1.0, ("p2020", "t0002"): 1.0},
        rp_weight={("p2020", "t0001"): 1.0, ("p2020", "t0002"): 1.0},
        slope={},
    )
    # Only t0001 realized → 1 MWh × 0.34 / 1000 = 0.00034
    assert got == pytest.approx({("co2_system", "p2020"): 0.00034})


def test_co2_tonnes_this_roll_rp_weight_applied() -> None:
    """Non-trivial rp_cost_weight scales the contribution accordingly."""
    df = _v_flow_frame(
        [
            ("p2020", "t0001", "coal_plant", "coal_market", "el", 2.0),
        ],
    )
    got = _co2_tonnes_this_roll(
        df,
        realized_set={("p2020", "t0001")},
        co2_groups={"co2_system"},
        co2_content={"Coal": 0.34},
        commodity_node_co2={("Coal", "coal_market")},
        group_node={"co2_system": {"coal_market"}},
        noeff_set={("coal_plant", "coal_market", "el")},
        eff_set=set(),
        process_unit_set=set(),
        process_source_flow_coeff={},
        process_sink_flow_coeff={},
        step_duration={("p2020", "t0001"): 3.0},
        rp_weight={("p2020", "t0001"): 5.0},
        slope={},
    )
    # 2 (flow) * 3 (dur) * 5 (rpw) = 30 MWh physical window.
    # 30 * 0.34 / 1000 = 0.0102
    assert got == pytest.approx({("co2_system", "p2020"): 0.0102})


def test_co2_tonnes_this_roll_removal_branch_subtracts() -> None:
    """A flow INTO a CO2 commodity_node (sink side) is a removal — the
    contribution is negative."""
    df = _v_flow_frame(
        [
            # Removal: source=scrubber, sink=coal_market (CO2 node)
            ("p2020", "t0001", "scrubber", "el", "coal_market", 1.0),
        ],
    )
    got = _co2_tonnes_this_roll(
        df,
        realized_set={("p2020", "t0001")},
        co2_groups={"co2_system"},
        co2_content={"Coal": 0.34},
        commodity_node_co2={("Coal", "coal_market")},
        group_node={"co2_system": {"coal_market"}},
        noeff_set=set(),
        eff_set=set(),
        process_unit_set=set(),
        process_source_flow_coeff={},
        process_sink_flow_coeff={},
        step_duration={("p2020", "t0001"): 1.0},
        rp_weight={("p2020", "t0001"): 1.0},
        slope={},
    )
    # Removal: -1 MWh × 0.34 / 1000 = -0.00034
    assert got == pytest.approx({("co2_system", "p2020"): -0.00034})


def test_co2_tonnes_this_roll_no_co2_groups_returns_empty() -> None:
    """Fast-path: no CO2 groups → empty dict."""
    df = _v_flow_frame(
        [
            ("p2020", "t0001", "coal_plant", "coal_market", "el", 1.0),
        ],
    )
    got = _co2_tonnes_this_roll(
        df,
        realized_set={("p2020", "t0001")},
        co2_groups=set(),
        co2_content={"Coal": 0.34},
        commodity_node_co2={("Coal", "coal_market")},
        group_node={},
        noeff_set={("coal_plant", "coal_market", "el")},
        eff_set=set(),
        process_unit_set=set(),
        process_source_flow_coeff={},
        process_sink_flow_coeff={},
        step_duration={("p2020", "t0001"): 1.0},
        rp_weight={("p2020", "t0001"): 1.0},
        slope={},
    )
    assert got == {}


# ---------------------------------------------------------------------------
# write_co2_rolling_accumulators (end-to-end unit, no HiGHS run)
# ---------------------------------------------------------------------------


def test_writer_no_co2_groups_emits_header_only(tmp_path: Path) -> None:
    """Every group uses period-only CO2 method → header-only CSV."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_group_co2_method(work, [("co2_period", "period")])
    h = _fake_highs(variable_names=[], col_values=[])
    write_co2_rolling_accumulators(h, solve_name="s1", work_folder=work)

    text = (work / "solve_data" / "co2_cum_realized_tonnes.csv").read_text()
    assert text.strip() == "group,period,p_co2_cum_realized_tonnes"


def test_writer_first_solve_records_realized_tonnes(tmp_path: Path) -> None:
    """Full no-eff flow, full realized horizon, single group → single
    accumulator row with the hand-computed tonnes value."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _standard_co2_setup(
        work,
        groups=["co2_system"],
        co2_content={"Coal": 0.34},
        commodity_node=[("Coal", "coal_market")],
        group_node=[("co2_system", "coal_market")],
        noeff=[("coal_plant", "coal_market", "el")],
        unitsize={"coal_plant": 1.0},
        realized_pairs=[("p2020", "t0001"), ("p2020", "t0002")],
        horizon_pairs=[("p2020", "t0001", 1.0), ("p2020", "t0002", 1.0)],
    )
    # Two 1 MWh flows (t0001, t0002) → 2 MWh × 0.34/1000 = 0.00068
    h = _fake_highs(
        variable_names=[
            "v_flow[coal_plant,coal_market,el,p2020,t0001]",
            "v_flow[coal_plant,coal_market,el,p2020,t0002]",
        ],
        col_values=[1.0, 1.0],
    )
    write_co2_rolling_accumulators(h, solve_name="s1", work_folder=work)

    got = _read_co2(work / "solve_data" / "co2_cum_realized_tonnes.csv")
    assert got == pytest.approx({("co2_system", "p2020"): 0.00068})


def test_writer_prior_plus_this_roll(tmp_path: Path) -> None:
    """Second solve with non-empty prior → prior + this-roll = new total."""
    work = _make_workfolder(tmp_path, first_solve=False)
    # Seed prior accumulator.
    (work / "solve_data" / "co2_cum_realized_tonnes.csv").write_text(
        "group,period,p_co2_cum_realized_tonnes\n"
        "co2_system,p2020,0.001\n"
    )
    _standard_co2_setup(
        work,
        groups=["co2_system"],
        co2_content={"Coal": 0.34},
        commodity_node=[("Coal", "coal_market")],
        group_node=[("co2_system", "coal_market")],
        noeff=[("coal_plant", "coal_market", "el")],
        unitsize={"coal_plant": 1.0},
        realized_pairs=[("p2020", "t0003")],
        horizon_pairs=[("p2020", "t0003", 1.0)],
    )
    # 1 MWh × 0.34/1000 = 0.00034  →  new total = 0.00134
    h = _fake_highs(
        variable_names=["v_flow[coal_plant,coal_market,el,p2020,t0003]"],
        col_values=[1.0],
    )
    write_co2_rolling_accumulators(h, solve_name="s2", work_folder=work)

    got = _read_co2(work / "solve_data" / "co2_cum_realized_tonnes.csv")
    assert got == pytest.approx({("co2_system", "p2020"): 0.00134})


def test_writer_entity_unitsize_scales_flow(tmp_path: Path) -> None:
    """The writer multiplies each column by p_entity_unitsize (wide
    format) before aggregating — 400 MW unitsize + 0.5 flow = 200 MWh."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _standard_co2_setup(
        work,
        groups=["co2_system"],
        co2_content={"Coal": 0.34},
        commodity_node=[("Coal", "coal_market")],
        group_node=[("co2_system", "coal_market")],
        noeff=[("coal_plant", "coal_market", "el")],
        unitsize={"coal_plant": 400.0},
        realized_pairs=[("p2020", "t0001")],
        horizon_pairs=[("p2020", "t0001", 1.0)],
    )
    h = _fake_highs(
        variable_names=["v_flow[coal_plant,coal_market,el,p2020,t0001]"],
        col_values=[0.5],
    )
    write_co2_rolling_accumulators(h, solve_name="s1", work_folder=work)

    # 0.5 * 400 * 1 (dur) * 1 (rpw) = 200 MWh
    # 200 * 0.34 / 1000 = 0.068 tonnes
    got = _read_co2(work / "solve_data" / "co2_cum_realized_tonnes.csv")
    assert got == pytest.approx({("co2_system", "p2020"): 0.068})
