"""``timeline_breaks.parquet``: the realized steps at which a time-series plot starts a new line, written by
``write_outputs`` from the realized step adjacency.

Each case solves a small scenario (``tests.json`` / ``stochastics.json`` plus one override alternative, built under
the pytest tmp dir), writes the parquet outputs the way ``flextool run`` does (the last step's FlexData and
solution, per-roll realized slices unioned from ``output_raw``) and reads the break file back.  Expected sets are
derived from the timeline of each fixture: a break is the first realized step after a gap, a representative-block
start (also when the blocks are adjacent), or the first step of a period that does not continue the previous one; a
roll seam that continues the timeline is not a break.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.engine._unit_db import BASE_FIXTURE_JSON, FLEXTOOL_ROOT, PLANT_CHAIN, b64, make_db

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"


def _pmap(data: dict, index_name: str = "period") -> dict:
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in data.items()]}


def _timeset(blocks: dict[str, float], name: str = "2day") -> list:
    return ["timeset", name, "timeset_duration", b64(_pmap(blocks, "time"), "map")]


def _rolling(horizon: float, jump: float, solve: str = SOLVE) -> list:
    return [["solve", solve, "solve_mode", b64("rolling_window", "str")],
            ["solve", solve, "rolling_solve_horizon", b64(horizon, "float")],
            ["solve", solve, "rolling_solve_jump", b64(jump, "float")]]


def _two_periods(start_a: str, start_b: str, *, shared: bool = False) -> tuple[list, list]:
    """Periods p2020 and p2025 of one solve on 24-h blocks starting at ``start_a`` / ``start_b`` of one timeline,
    each on its own timeset, or (``shared``) both on timeset ``ta`` (the years reuse one timeset)."""
    entities = [["timeset", "ta", None], ["timeset", "tb", None]]
    values = [["timeset", "ta", "timeline", b64("y2020", "str")],
              ["timeset", "tb", "timeline", b64("y2020", "str")],
              _timeset({start_a: 24.0}, "ta"), _timeset({start_b: 24.0}, "tb"),
              ["solve", SOLVE, "period_timeset", b64(_pmap({"p2020": "ta", "p2025": "ta" if shared else "tb"}), "map")],
              ["solve", SOLVE, "realized_periods", b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")]]
    return values, entities


def _adjacent_rp_blocks() -> tuple[list, list]:
    """A representative-period timeset of two 24-step blocks t0001 and t0025, adjacent on the timeline."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.5], ["t0025", 0.5]]}]]}
    return ([["timeset", "rp2", "timeline", b64("y2020", "str")],
             _timeset({"t0001": 24.0, "t0025": 24.0}, "rp2"),
             ["timeset", "rp2", "representative_period_weights", b64(rpw, "map")],
             ["solve", SOLVE, "period_timeset", b64(_pmap({"p2020": "rp2"}), "map")]],
            [["timeset", "rp2", None]])


def _write(tmp_path_factory, alt: str, values: list, *, entities=None, chain=PLANT_CHAIN,
           fixture_json: Path = BASE_FIXTURE_JSON) -> tuple[dict, Path]:
    """Solve, write the parquet outputs as ``flextool run`` does; return ``(steps, scenario parquet dir)``."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities,
                        fixture_json=fixture_json)
    work = tmp_path_factory.mktemp(f"{alt}_out")
    steps = run_chain_from_db(url, scen, work_folder=work)
    last = list(steps.values())[-1]
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["parquet"],
                  fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=last.flex_data_provider)
    return steps, work / "output_parquet" / scen


def _breaks(parquet_dir: Path) -> set[tuple[str, str]]:
    table = pq.read_table(parquet_dir / "timeline_breaks.parquet")
    return set(zip(table.column("period").to_pylist(), table.column("time").to_pylist()))


GAP = {"t0001": 24.0, "t0049": 24.0}         # t0001-t0024 and t0049-t0072: a gap of t0025-t0048


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    out["one_block"] = _write(tmp_path_factory, "tb_one_block", [_timeset({"t0001": 24.0})])
    out["gap"] = _write(tmp_path_factory, "tb_gap", [_timeset(GAP)])
    out["gap_rolling"] = _write(tmp_path_factory, "tb_gap_roll", [_timeset(GAP)] + _rolling(24.0, 24.0))
    out["gap_rolling_12"] = _write(tmp_path_factory, "tb_gap_roll12", [_timeset(GAP)] + _rolling(18.0, 12.0))
    values, ents = _two_periods("t0001", "t0049")
    out["periods_gap"] = _write(tmp_path_factory, "tb_periods_gap", values, entities=ents)
    values, ents = _two_periods("t0001", "t0025")
    out["periods_continue"] = _write(tmp_path_factory, "tb_periods_cont", values, entities=ents)
    values, ents = _two_periods("t0001", "t0001", shared=True)
    out["periods_shared"] = _write(tmp_path_factory, "tb_periods_shared", values, entities=ents)
    values, ents = _adjacent_rp_blocks()
    out["rp_adjacent"] = _write(tmp_path_factory, "tb_rp_adjacent", values, entities=ents)
    return out


def test_file_schema_and_present_when_empty(runs):
    """Two Utf8 columns ``period`` and ``time``; a single block has no breaks and the file is still written."""
    import pyarrow as pa

    _, pdir = runs["one_block"]
    schema = pq.read_schema(pdir / "timeline_breaks.parquet")
    assert schema.names == ["period", "time"]
    assert schema.field("period").type == pa.string() and schema.field("time").type == pa.string()
    assert _breaks(pdir) == set()


def _single_period(steps: dict) -> str:
    periods = {p for s in steps.values() for p in s.flex_data.realized_dispatch["period"].cast(str).to_list()}
    assert len(periods) == 1, periods
    return periods.pop()


def test_gap_in_the_timeline_breaks_at_the_step_after_it(runs):
    """Blocks t0001-t0024 and t0049-t0072 of one period: the line starts again at t0049 only."""
    steps, pdir = runs["gap"]
    assert _breaks(pdir) == {(_single_period(steps), "t0049")}


def test_rolling_twin_breaks_only_at_the_gap(runs):
    """The gap model rolled one block per roll (seam at the gap) and in 12-h jumps (seams inside the blocks that
    continue the timeline, carried from the previous roll): the same single break as the single solve."""
    steps, pdir = runs["gap"]
    expected = {(_single_period(steps), "t0049")}
    for name in ("gap_rolling", "gap_rolling_12"):
        roll_steps, roll_dir = runs[name]
        assert len(roll_steps) > 1, name
        assert _breaks(roll_dir) == expected, name


def test_period_that_does_not_continue_breaks(runs):
    """p2025 starts at t0049 after p2020 ended at t0024 (a gap): break at p2025's first step."""
    assert _breaks(runs["periods_gap"][1]) == {("p2025", "t0049")}


def test_continuing_period_is_one_line(runs):
    """p2025 starts at t0025 right after p2020's t0024 on the same timeline: no break."""
    assert _breaks(runs["periods_continue"][1]) == set()


def test_years_reusing_one_timeset_break(runs):
    """p2020 and p2025 both on timeset t0001-t0024: p2025 repeats the hours, it does not continue p2020 — break at
    (p2025, t0001)."""
    assert _breaks(runs["periods_shared"][1]) == {("p2025", "t0001")}


def test_adjacent_representative_blocks_break(runs):
    """Representative blocks t0001-t0024 and t0025-t0048 are adjacent on the timeline but represent different base
    periods: break at the second block's start."""
    steps, pdir = runs["rp_adjacent"]
    assert _breaks(pdir) == {(_single_period(steps), "t0025")}


def test_rolling_stochastic_realization_seams_are_one_line(tmp_path_factory):
    """Rolling stochastic 1-week fixture with a realized branch (an engine-added realization solve per roll): the
    realized rolls follow each other on the timeline, so the realized dispatch is one line — no break."""
    from tests.engine._unit_db import is_realization

    steps, pdir = _write(tmp_path_factory, "tb_stoch_roll", [], chain=["init", "base", "system", "1week_rolling"],
                         fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
    assert sum(is_realization(k) for k in steps) > 1, list(steps)
    assert _breaks(pdir) == set()
