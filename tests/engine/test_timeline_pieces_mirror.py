"""``flextool.timeline_pieces.solve_step_chunks`` gives the steps the engine builds for a solve with ``new_stepduration``.

The helper (used outside the engine, by the assistant's model facts) must mirror the engine's chunk grouping: the
solves that share a timeset share its aggregated timeline, cut at the blocks of every timeset of the group on that
timeline and at the representative base-period starts.  Each case loads the solve and timeline configuration from a
database built from JSON as the engine's loader does, gives the helper the original timelines and blocks, runs the
engine's ``create_timeline_from_timestep_duration`` and compares the steps.

Cases: ``tests.json`` scenario ``multi_fullYear_battery_nested_24h_invest_one_solve`` (24 h and 6 h steps on the
8760-h timeline, nested solves); ``5weeks`` with a 3-h step (4-h blocks end with a 1-h step; the t0009 and t0013
blocks touch); a representative timeset (rep blocks t0001, t0049 of 24 h, base periods t0001, t0025, t0049) with a
5-h step (each rep block ends with a 4-h step; the base start t0025 cuts the gap).
"""
from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys

from flextool.timeline_pieces import aggregate_timeline_pieces, quiet_logger, solve_step_chunks
from tests.engine._unit_db import BASE_FIXTURE_JSON, FLEXTOOL_ROOT, b64, make_db


def _pmap(values: dict, index_name: str) -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}, "map")


def _rp_values(step: float) -> tuple[list, list]:
    def rep(w: float) -> dict:
        return {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                "data": [["t0001", w], ["t0049", 1.0 - w]]}
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", rep(1.0)], ["t0025", rep(0.5)], ["t0049", rep(0.0)]]}
    values = [["timeset", "rpg", "timeline", b64("y2020", "str")],
              ["timeset", "rpg", "timeset_duration", _pmap({"t0001": 24.0, "t0049": 24.0}, "time")],
              ["timeset", "rpg", "representative_period_weights", b64(rpw, "map")],
              ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "rpg"}, "period")],
              ["solve", "y2020_2day_dispatch", "new_stepduration", b64(step, "float")]]
    return values, [["timeset", "rpg", None]]


def _tests_db(tmp_path_factory) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp("tp_scen_db")
    url = json_to_db(BASE_FIXTURE_JSON, work / "tests.sqlite")
    migrate_database(url)
    return url


def _configs(url: str, scenario: str):
    """SolveConfig and TimelineConfig as the engine's loader has them right before the ``new_stepduration``
    aggregation (``_db_loader``: load, ``create_assumptive_parts``, ``validate_timeset_blocks``)."""
    import spinedb_api as api

    from flextool.engine.input._solve_config import SolveConfig
    from flextool.engine.input._timeline import TimelineConfig
    from flextool.update_flextool.db_handle import open_database_mapping

    log = logging.getLogger("test_timeline_pieces_mirror")
    with open_database_mapping(url) as db:
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config(scenario))
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        solve = SolveConfig.load_from_db(db=db, logger=log)
        timeline = TimelineConfig.load_from_db(db=db, logger=log)
    timeline.create_assumptive_parts(solve)
    timeline.validate_timeset_blocks(solve, set())
    return solve, timeline


def _compare(solve, timeline) -> list:
    """Run the helper on the original inputs for every solve with new_stepduration, then the engine; assert the
    same steps.  Returns the helper's timelines (for the case's own checks)."""
    inputs = dict(
        new_step_durations=dict(timeline.new_step_durations),
        timesets_of_solve={s: [ts for _d, ts in pairs] for s, pairs in solve.timesets_used_by_solves.items()},
        timeset_blocks=copy.deepcopy(dict(timeline.timeset_durations)),
        timeline_of_timeset=dict(timeline.timesets__timeline),
        timelines=copy.deepcopy(dict(timeline.timelines)),
        rp_weights=copy.deepcopy(dict(timeline.rp_weights)),
    )
    step_solves = [s for s, raw in inputs["new_step_durations"].items() if raw is not None]
    assert step_solves
    helper = {s: solve_step_chunks(s, **inputs) for s in step_solves}
    timeline.create_timeline_from_timestep_duration(solve)
    seen = []
    for s, by_timeline in helper.items():
        own = [ts for ts in dict.fromkeys(inputs["timesets_of_solve"].get(s, ())) if inputs["timeset_blocks"].get(ts)]
        # Every timeset of the solve is on one of the helper's aggregated timelines, as in the engine.
        assert {timeline.timesets__timeline[ts] for ts in own} == {st.name for st in by_timeline.values()}, s
        for original, st in by_timeline.items():
            assert timeline.original_timeline[st.name] == original
            assert timeline.timelines[st.name] == st.chunks, (s, original)
            for ts, blocks in st.block_chunks.items():
                assert timeline.timeset_durations[ts] == blocks, (s, ts)
            seen.append(st)
    return seen


def _hours(chunks) -> set[float]:
    return {float(h) for _label, h in chunks}


def test_nested_full_year_scenario(tmp_path_factory) -> None:
    scenario = "multi_fullYear_battery_nested_24h_invest_one_solve"
    solve, timeline = _configs(_tests_db(tmp_path_factory), scenario)
    seen = _compare(solve, timeline)
    assert {24.0, 6.0} <= set().union(*(_hours(st.chunks) for st in seen))


def test_non_multiple_block_ends_with_a_shorter_step(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="tp_5w_3h", base_chain=["init", "west", "wind", "battery", "5weeks"],
                        values=[["solve", "y2020_5week", "new_stepduration", b64(3.0, "float")]])
    seen = _compare(*_configs(url, scen))
    (st,) = seen
    assert st.chunks[:3] == [("t0001", "3.0"), ("t0004", "1.0"), ("t0005", "3.0")]
    assert st.block_chunks["5weeks"] == [("t0001", 2), ("t0009", 2), ("t0013", 2), ("t0025", 2), ("t0037", 2)]


def test_representative_periods(tmp_path_factory) -> None:
    values, entities = _rp_values(5.0)
    url, scen = make_db(tmp_path_factory, alt="tp_rp_5h", base_chain=["init", "west", "coal"], values=values,
                        entities=entities)
    seen = _compare(*_configs(url, scen))
    (st,) = seen
    labels = [label for label, _h in st.chunks]
    assert {"t0001", "t0025", "t0049"} <= set(labels)          # the base start t0025 cuts the gap
    assert dict(st.chunks)["t0021"] == "4.0"                    # each rep block ends with a 4-h step
    assert st.block_chunks["rpg"] == [("t0001", 5), ("t0049", 5)]


def test_quiet_logger_prints_nothing(capsys, caplog) -> None:
    """The shorter-step warning goes nowhere with the quiet logger (no propagation, no last-resort print)."""
    steps = [(f"t{i:04d}", "1") for i in range(1, 9)]
    with caplog.at_level(logging.WARNING):
        chunks, blocks = aggregate_timeline_pieces(steps, {"ts": [("t0001", 5)]}, 3.0, None,
                                                   solve_name="s", timeline_name="T")
    assert chunks == [("t0001", "3.0"), ("t0004", "2.0"), ("t0006", "3.0")]
    assert blocks == {"ts": [("t0001", 2)]}
    assert caplog.records == []
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    log = quiet_logger()
    assert log.propagate is False and len(log.handlers) == 1


def test_light_module_does_not_import_the_engine() -> None:
    code = ("import sys, flextool.timeline_pieces; "
            "print(sorted(m for m in sys.modules if m == 'flextool.engine' or m.startswith('flextool.engine.')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=FLEXTOOL_ROOT, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1].replace("'", '"')) == []


def test_merged_groups() -> None:
    """Solves s1 (ta) and s2 (tb) start two groups; s3 (ta, tb) joins them into s1's: every solve gets the one
    timeline ``T_s1`` cut at both timesets' blocks (the overlap cuts ta's step at t0004)."""
    from collections import defaultdict
    from types import SimpleNamespace

    from flextool.engine.input._timeline import TimelineConfig

    blocks = {"ta": [("t0001", 8)], "tb": [("t0004", 4)]}
    solves = {"s1": [("p1", "ta")], "s2": [("p2", "tb")], "s3": [("p1", "ta"), ("p2", "tb")]}
    tl = TimelineConfig(
        timelines=defaultdict(list, {"T": [(f"t{i:04d}", "1") for i in range(1, 13)]}),
        timesets=list(blocks), timesets__timeline=defaultdict(str, {ts: "T" for ts in blocks}),
        timeset_durations=defaultdict(list, blocks), new_step_durations={s: 2.0 for s in solves})
    sc = SimpleNamespace(timesets_used_by_solves=solves, model_solve={"m": list(solves)}, contains_solves={})
    seen = _compare(sc, tl)
    assert {st.name for st in seen} == {"T_s1"}
    assert [label for label, _h in seen[0].chunks] == ["t0001", "t0003", "t0004", "t0006", "t0008", "t0009",
                                                       "t0011"]


def test_solve_without_new_stepduration() -> None:
    assert solve_step_chunks("s", new_step_durations={"s": None}, timesets_of_solve={"s": ["ts"]},
                             timeset_blocks={"ts": [("t0001", 2)]}, timeline_of_timeset={"ts": "T"},
                             timelines={"T": [("t0001", "1"), ("t0002", "1")]}) == {}
