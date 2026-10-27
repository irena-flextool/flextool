"""Tests for the solve-to-solve handoff CSV writers (debug mirrors; item 163, c24).

Item 163: these writers no longer read the work folder.  Every input comes from a ``flex_data`` stub
(:func:`_flex_data_stub`), an explicit ``is_first_solve`` and an :class:`OutputWriterState`; the five
handoff-mirror CSVs are written only when ``csv_dump=True`` (design_163.md §11 Q1).
``period_capacity.csv`` is unaffected by that gate (tested separately in
``tests/engine/test_output_writer.py``).
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl

from flextool.engine.output._writer import OutputWriterState
from flextool.process_outputs.handoff_writers import (
    write_p_entity_divested,
    write_p_entity_period_existing_capacity,
    write_p_roll_continue_state,
    write_fix_storage_quantity,
    write_fix_storage_usage,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_highs(
    variable_names: list[str],
    col_values: list[float],
    row_names: list[str] | None = None,
    row_duals: list[float] | None = None,
) -> MagicMock:
    h = MagicMock()
    h.allVariableNames.return_value = list(variable_names)
    sol = SimpleNamespace(
        col_value=list(col_values),
        row_dual=list(row_duals or []),
    )
    h.getSolution.return_value = sol
    h.getLp.return_value = SimpleNamespace(row_names_=list(row_names or []))
    return h


def _flex_data_stub(
    *,
    unitsize: dict[str, float] | None = None,
    entity_divest: list[str] | None = None,
    fix_methods: dict[str, str] | None = None,
    node_state: list[str] | None = None,
    realized_dispatch: list[tuple[str, str]] | None = None,
    step_duration: dict[tuple[str, str], float] | None = None,
) -> SimpleNamespace:
    """A ``flex_data`` stand-in carrying only the fields the handoff writers read.

    ``unitsize``: ``{entity: unitsize}`` -> ``p_all_entity_unitsize`` (also the entity set source).
    ``entity_divest``: entities allowed to divest -> ``ed_divest_set``.
    ``fix_methods``: ``{node: method}`` -> ``node__storage_nested_fix_method``.
    ``node_state``: nodes carrying a state variable -> ``nodeState``.
    ``realized_dispatch``: ``[(period, step), ...]`` -> ``realized_dispatch``.
    ``step_duration``: ``{(period, step): duration}`` -> ``p_step_duration``.
    """
    p_all_entity_unitsize = None
    if unitsize:
        p_all_entity_unitsize = SimpleNamespace(
            frame=pl.DataFrame({"entity": list(unitsize), "value": list(unitsize.values())})
        )
    ed_divest_set = pl.DataFrame({"e": entity_divest}) if entity_divest else None
    nsfm = (
        pl.DataFrame({"node": list(fix_methods), "method": list(fix_methods.values())})
        if fix_methods else None
    )
    node_state_df = pl.DataFrame({"n": node_state}) if node_state else None
    rd = (
        pl.DataFrame({"period": [p for p, _t in realized_dispatch],
                      "step": [t for _p, t in realized_dispatch]})
        if realized_dispatch else None
    )
    p_step_duration = None
    if step_duration:
        p_step_duration = SimpleNamespace(frame=pl.DataFrame({
            "period": [k[0] for k in step_duration],
            "time": [k[1] for k in step_duration],
            "value": list(step_duration.values()),
        }))
    return SimpleNamespace(
        p_all_entity_unitsize=p_all_entity_unitsize,
        ed_divest_set=ed_divest_set,
        node__storage_nested_fix_method=nsfm,
        nodeState=node_state_df,
        realized_dispatch=rd,
        p_step_duration=p_step_duration,
    )


# ---------------------------------------------------------------------------
# write_p_entity_divested
# ---------------------------------------------------------------------------


def test_p_entity_divested_first_solve(tmp_path: Path) -> None:
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"unit_a": 100.0, "unit_b": 1.0}, entity_divest=["unit_a", "unit_b"])

    # v_divest[unit_a, p2025]=2 -> cumulative 2*100 = 200.  unit_b has none.
    h = _fake_highs(
        variable_names=["v_divest[unit_a,p2025]"], col_values=[2.0],
    )

    write_p_entity_divested(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert len(rows) == 2
    by_entity = {r["entity"]: float(r["p_entity_divested"]) for r in rows}
    assert by_entity == {"unit_a": 200.0, "unit_b": 0.0}


def test_p_entity_divested_csv_dump_false_writes_nothing(tmp_path: Path) -> None:
    """Item 163 (design_163.md §11 Q1): without ``csv_dump=True`` the mirror is never written."""
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"unit_a": 100.0}, entity_divest=["unit_a"])
    h = _fake_highs(variable_names=["v_divest[unit_a,p2025]"], col_values=[2.0])

    path = write_p_entity_divested(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), csv_dump=False,
    )
    assert not path.exists()


def test_p_entity_divested_second_solve_accumulates(tmp_path: Path) -> None:
    """Non-first solve adds the writer_state prior (standing in for prior_handoff when it's absent) +
    this solve's v_divest.  u=50, prior=100 (d1=2 from an earlier call), this call d2=3: 100 + 3*50=150
    -> 250."""
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"unit_a": 50.0}, entity_divest=["unit_a"])
    writer_state = OutputWriterState()
    writer_state.divested = {"unit_a": 100.0}

    h = _fake_highs(variable_names=["v_divest[unit_a,p2030]"], col_values=[3.0])
    write_p_entity_divested(
        h, solve_name="s2", work_folder=work, flex_data=fd,
        is_first_solve=False, writer_state=writer_state, csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert {r["entity"]: float(r["p_entity_divested"]) for r in rows} == {"unit_a": 250.0}


def test_p_entity_divested_prior_from_writer_state(tmp_path: Path) -> None:
    """Item 163 T6: with no ``prior_handoff`` (e.g. the direct-writer call shape this test uses), the
    SAME ``writer_state`` carries the previous call's cumulative across two calls, in place of the
    removed disk read.  u=50, d1=2 -> 100; delete the file; d2=3 -> 100 + 150 = 250."""
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"unit_a": 50.0}, entity_divest=["unit_a"])
    writer_state = OutputWriterState()

    h1 = _fake_highs(variable_names=["v_divest[unit_a,p2025]"], col_values=[2.0])
    write_p_entity_divested(
        h1, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=writer_state, csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert {r["entity"]: float(r["p_entity_divested"]) for r in rows} == {"unit_a": 100.0}

    (work / "solve_data" / "p_entity_divested.csv").unlink()

    h2 = _fake_highs(variable_names=["v_divest[unit_a,p2030]"], col_values=[3.0])
    write_p_entity_divested(
        h2, solve_name="s2", work_folder=work, flex_data=fd,
        is_first_solve=False, prior_handoff=None, writer_state=writer_state, csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert {r["entity"]: float(r["p_entity_divested"]) for r in rows} == {"unit_a": 250.0}


# ---------------------------------------------------------------------------
# write_fix_storage_quantity
# ---------------------------------------------------------------------------


def test_fix_storage_quantity_writes_only_for_target_method(tmp_path: Path) -> None:
    """Item 163 drops the ``dt_fix_storage_timesteps`` filter (it was never provider-wired, so it was
    already a no-op in every production run) — every ``v_state`` step of a ``fix_quantity`` node is
    emitted now, not just the ones a (never-populated) filter happened to keep."""
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(
        unitsize={"battery": 10.0, "tank": 1.0},
        fix_methods={"battery": "fix_quantity", "tank": "fix_price"},
    )
    # v_state for both nodes; only battery is fix_quantity -> only it's emitted.
    h = _fake_highs(
        variable_names=[
            "v_state[battery,p2020,t0001]",
            "v_state[battery,p2020,t0024]",
            "v_state[battery,p2020,t0048]",
            "v_state[tank,p2020,t0001]",
        ],
        col_values=[5.0, 7.0, 9.0, 99.0],
    )

    write_fix_storage_quantity(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "fix_storage_quantity.csv")))
    assert {(r["period"], r["step"], r["node"], r["p_fix_storage_quantity"]) for r in rows} == {
        ("p2020", "t0001", "battery", "50"),  # 5 x 10
        ("p2020", "t0024", "battery", "70"),  # 7 x 10
        ("p2020", "t0048", "battery", "90"),  # 9 x 10
    }


def test_fix_storage_quantity_preserves_prior_when_empty(tmp_path: Path) -> None:
    """Non-first solve with no fix_quantity entries must not clobber prior content."""
    work = tmp_path
    (work / "solve_data").mkdir()
    prior = "period,step,node,p_fix_storage_quantity\np2020,t0001,battery,50\n"
    (work / "solve_data" / "fix_storage_quantity.csv").write_text(prior)
    fd = _flex_data_stub(unitsize={"battery": 1.0})  # no fix_methods -> target_nodes empty

    h = _fake_highs(variable_names=[], col_values=[])
    write_fix_storage_quantity(
        h, solve_name="s2", work_folder=work, flex_data=fd,
        is_first_solve=False, writer_state=OutputWriterState(), csv_dump=True,
    )
    assert (work / "solve_data" / "fix_storage_quantity.csv").read_text() == prior


def test_fix_storage_mirror_ignores_a_leftover_file(tmp_path: Path) -> None:
    """Item 163 T7: a stale ``fix_storage_quantity.csv`` left by an earlier run (or an earlier roll's
    reused work folder) must not be mistaken for "already written this cascade" — ``writer_state
    .mirrors_written`` (fresh/empty here) replaces the old ``out_path.exists()`` check, so a first-solve
    call with no rows truncates it back to header-only instead of preserving the stale row."""
    work = tmp_path
    (work / "solve_data").mkdir()
    (work / "solve_data" / "fix_storage_quantity.csv").write_text(
        "period,step,node,p_fix_storage_quantity\np2020,t0001,battery,50\n"
    )
    fd = _flex_data_stub(unitsize={"battery": 1.0})  # no fix_methods -> target_nodes empty
    h = _fake_highs(variable_names=[], col_values=[])

    write_fix_storage_quantity(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), csv_dump=True,
    )
    assert (work / "solve_data" / "fix_storage_quantity.csv").read_text() == (
        "period,step,node,p_fix_storage_quantity\n"
    )


# ---------------------------------------------------------------------------
# write_p_roll_continue_state
# ---------------------------------------------------------------------------


def test_p_roll_continue_state_uses_last_realized_step(tmp_path: Path) -> None:
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(
        unitsize={"battery": 2.0, "non_state_node": 1.0},
        node_state=["battery"],
        # Two periods, multiple steps each -- last entry per period is the "last realized" boundary.
        realized_dispatch=[("p2020", "t0001"), ("p2020", "t0002"), ("p2025", "t0001"), ("p2025", "t0048")],
    )
    h = _fake_highs(
        variable_names=[
            "v_state[battery,p2020,t0001]",
            "v_state[battery,p2020,t0002]",
            "v_state[battery,p2025,t0001]",
            "v_state[battery,p2025,t0048]",
        ],
        col_values=[1.0, 2.0, 3.0, 7.5],
    )
    write_p_roll_continue_state(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        writer_state=OutputWriterState(), csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "p_roll_continue_state.csv")))
    # Last (period, time) pair is (p2025, t0048); v_state=7.5 x 2 = 15.0.
    assert rows == [{"node": "battery", "p_roll_continue_state": "15"}]


# ---------------------------------------------------------------------------
# write_p_entity_period_existing_capacity
# ---------------------------------------------------------------------------


def test_p_entity_period_existing_capacity_first_solve(tmp_path: Path) -> None:
    """existing = pre_existing[d] on the first solve.  Item 163 drops the ``ed_invest``/``v_invest``
    term entirely (its source CSV has had no producer since v4, so it was always empty in a fresh
    run -- see the function's docstring)."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"battery": 1.0, "wind_plant": 1000.0})
    provider = FlexDataProvider()
    provider.put("solve_data/p_entity_pre_existing", pl.DataFrame({
        "entity": ["battery", "wind_plant"], "period": ["p2020", "p2020"], "value": [50.0, 1000.0],
    }))
    provider.put("solve_data/realized_invest_periods_of_current_solve", pl.DataFrame({"period": ["p2020"]}))
    provider.put("solve_data/period_first", pl.DataFrame({"period": ["p2020"]}))
    h = _fake_highs(variable_names=[], col_values=[])

    # ``csv_dump=True`` is the public opt-in to disk materialisation (mirrors the CLI ``--csv-dump``
    # flag in ``cmd_run_flextool.py``).  This unit test asserts on the CSV contract (column order,
    # ``%.8g`` rounding, on-disk layout).
    write_p_entity_period_existing_capacity(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), provider=provider, csv_dump=True,
    )
    rows = list(csv.DictReader(
        open(work / "solve_data" / "p_entity_period_existing_capacity.csv")
    ))
    by_key = {(r["entity"], r["period"]): r for r in rows}
    assert float(by_key[("battery", "p2020")]["p_entity_period_existing_capacity"]) == 50.0
    assert float(by_key[("battery", "p2020")]["p_entity_period_invested_capacity"]) == 0.0
    assert float(by_key[("wind_plant", "p2020")]["p_entity_period_existing_capacity"]) == 1000.0
    assert float(by_key[("wind_plant", "p2020")]["p_entity_period_invested_capacity"]) == 0.0


def test_p_entity_period_existing_capacity_csv_dump_false_still_updates_writer_state(tmp_path: Path) -> None:
    """Item 163: unlike the other four mirrors, this writer always computes + updates ``writer_state
    .existing``/``.invested`` (the next solve's fallback prior), even when ``csv_dump=False`` skips the
    disk write."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(unitsize={"battery": 1.0})
    provider = FlexDataProvider()
    provider.put("solve_data/p_entity_pre_existing", pl.DataFrame({
        "entity": ["battery"], "period": ["p2020"], "value": [50.0],
    }))
    provider.put("solve_data/realized_invest_periods_of_current_solve", pl.DataFrame({"period": ["p2020"]}))
    provider.put("solve_data/period_first", pl.DataFrame({"period": ["p2020"]}))
    h = _fake_highs(variable_names=[], col_values=[])
    writer_state = OutputWriterState()

    path = write_p_entity_period_existing_capacity(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=writer_state, provider=provider, csv_dump=False,
    )
    assert not path.exists()
    assert writer_state.existing.get(("battery", "p2020")) == 50.0
    assert writer_state.invested.get(("battery", "p2020")) == 0.0


# ---------------------------------------------------------------------------
# write_fix_storage_usage
# ---------------------------------------------------------------------------


def test_fix_storage_usage_net_flow_through_storage_node(tmp_path: Path) -> None:
    """Exact formula for method_nvar / simple 1var_per_way:
    ``(outflow - inflow) × step_duration`` where flow is
    ``v_flow[p, n, *]`` or ``v_flow[p, *, n]`` times ``unitsize[p]``."""
    work = tmp_path
    (work / "solve_data").mkdir()
    fd = _flex_data_stub(
        unitsize={"inverter": 2.0, "battery_node": 1.0, "west": 1.0},
        fix_methods={"battery_node": "fix_usage"},
        step_duration={("p2020", "t0024"): 1.5},
    )
    # Discharge: battery_node -> west (outflow 3 x unitsize 2 = 6)
    # Charge:    west -> battery_node (inflow 1 x unitsize 2 = 2)
    # Net: (6 - 2) x 1.5 = 6
    h = _fake_highs(
        variable_names=[
            "v_flow[inverter,battery_node,west,p2020,t0024]",
            "v_flow[inverter,west,battery_node,p2020,t0024]",
        ],
        col_values=[3.0, 1.0],
    )
    write_fix_storage_usage(
        h, solve_name="s1", work_folder=work, flex_data=fd,
        is_first_solve=True, writer_state=OutputWriterState(), csv_dump=True,
    )
    rows = list(csv.DictReader(open(work / "solve_data" / "fix_storage_usage.csv")))
    assert rows == [{
        "period": "p2020", "step": "t0024",
        "node": "battery_node", "p_fix_storage_usage": "6",
    }]


def test_fix_storage_usage_preserves_prior_when_no_matching_node(tmp_path: Path) -> None:
    work = tmp_path
    (work / "solve_data").mkdir()
    prior = "period,step,node,p_fix_storage_usage\np2020,t0012,battery,7.5\n"
    (work / "solve_data" / "fix_storage_usage.csv").write_text(prior)
    # No node has fix_usage method -> early return, must not clobber.
    fd = _flex_data_stub(unitsize={"any": 1.0})

    h = _fake_highs(variable_names=[], col_values=[])
    write_fix_storage_usage(
        h, solve_name="s2", work_folder=work, flex_data=fd,
        is_first_solve=False, writer_state=OutputWriterState(), csv_dump=True,
    )
    assert (work / "solve_data" / "fix_storage_usage.csv").read_text() == prior
