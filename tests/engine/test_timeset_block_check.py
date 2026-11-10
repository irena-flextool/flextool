"""The timeset_duration block check against the timeline (#222).

``TimelineConfig.validate_timeset_blocks`` checks the original blocks of the timesets the model's solves use,
before any expansion (the ``new_stepduration`` aggregation included):
- a block that starts off the timeline -> error (before: the block was skipped without a word);
- a block that runs past the end of the timeline -> one warning per run; the block is cut at the end of the
  timeline, as before (on a representative-period timeset -> error, the block cannot be cut);
- a timeset that no solve of the model uses is not checked.

Databases are ``tests/fixtures/tests.json`` plus one override alternative (never a checked-in ``.sqlite``).
Chain ``init, west, coal``: model ``flexTool`` runs ``y2020_2day_dispatch`` with timeset ``2day`` on timeline
``y2020`` = t0001..t0072 (72 steps). Expected values worked out by hand:
- ``{t0030: 48}``: t0030 is step 30, so 72 - 30 + 1 = 43 steps are left (t0030..t0072), 48 - 43 = 5 are cut;
- ``{t0025: 48}``: 72 - 25 + 1 = 48 steps left, the block fits exactly.
"""
from __future__ import annotations

import json
import logging

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import BASE_FIXTURE_JSON, PLANT_CHAIN, b64, make_db, run

LATE_WARNING = ("Warning: timeset '2day' (solve 'y2020_2day_dispatch'): the block starting 't0030' has 48 time "
                "steps, but timeline 'y2020' has only 43 from 't0030' (to 't0072'); the last 5 are not used.")


def _blocks(blocks: dict[str, float], timeset: str = "2day") -> list:
    return ["timeset", timeset, "timeset_duration",
            b64({"index_type": "str", "index_name": "time", "data": [[k, v] for k, v in blocks.items()]}, "map")]


def _check(url: str, scenario: str) -> None:
    """The check alone, as at the run's construction sites (after create_assumptive_parts)."""
    from flextool.engine.input._solve_config import SolveConfig
    from flextool.engine.input._timeline import TimelineConfig

    logger = logging.getLogger("test_timeset_block_check")
    sc = SolveConfig.load_from_db_url(url, scenario, logger=logger)
    tc = TimelineConfig.load_from_db_url(url, scenario, logger=logger)
    tc.create_assumptive_parts(sc)
    tc.validate_timeset_blocks(sc)


def _db(tmp_path_factory, alt: str, values: list) -> tuple[str, str]:
    return make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN)


def test_block_past_the_end_warns_and_is_cut(tmp_path_factory, capsys) -> None:
    url, scen = _db(tmp_path_factory, "b222_late", [_blocks({"t0030": 48.0})])
    steps = run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    assert out.count(LATE_WARNING) == 1, out
    st = list(steps.values())[-1]
    assert st.flex_data.dt.height == 43


def test_block_that_fits_exactly_passes(tmp_path_factory, capsys) -> None:
    _check(*_db(tmp_path_factory, "b222_exact", [_blocks({"t0025": 48.0})]))
    assert "Warning: timeset" not in capsys.readouterr().out


def test_block_start_in_wrong_case_is_an_error(tmp_path_factory) -> None:
    url, scen = _db(tmp_path_factory, "b222_case", [_blocks({"T0001": 48.0})])
    with pytest.raises(FlexToolConfigError) as err:
        _check(url, scen)
    msg = str(err.value)
    assert "'2day'" in msg and "'T0001'" in msg and "Did you mean 't0001'? Time step names are case sensitive." \
        in msg, msg


def test_only_the_block_off_the_timeline_is_named(tmp_path_factory) -> None:
    url, scen = _db(tmp_path_factory, "b222_two", [_blocks({"t0001": 24.0, "x9999": 24.0})])
    with pytest.raises(FlexToolConfigError) as err:
        _check(url, scen)
    msg = str(err.value)
    assert "the block starting 'x9999'" in msg and msg.count("the block starting") == 1, msg
    # no case-insensitive match: no "did you mean", no case sentence
    assert "Did you mean" not in msg and "case sensitive" not in msg, msg


def test_fractional_block_length_warns(tmp_path_factory, capsys) -> None:
    _check(*_db(tmp_path_factory, "b222_frac", [_blocks({"t0001": 47.5})]))
    assert ("Warning: timeset '2day' (solve 'y2020_2day_dispatch'): the block starting 't0001' has length 47.5, "
            "which is not a whole number of time steps; 47 time steps are used.") in capsys.readouterr().out


def test_each_run_in_one_process_warns_once(tmp_path_factory, capsys) -> None:
    """The warn-once state belongs to a run: a second run in the same process prints the warning again."""
    url, scen = _db(tmp_path_factory, "b222_twice", [_blocks({"t0030": 48.0})])
    for _ in range(2):
        run(tmp_path_factory, url, scen)
        assert capsys.readouterr().out.count(LATE_WARNING) == 1


def test_timeset_no_solve_uses_is_not_checked(tmp_path_factory, capsys) -> None:
    # ``5weeks`` is not a timeset of y2020_2day_dispatch
    _check(*_db(tmp_path_factory, "b222_unused", [_blocks({"x9999": 4.0, "t0070": 10.0}, "5weeks")]))
    assert "Warning: timeset" not in capsys.readouterr().out


def test_new_stepduration_warns_once_before_the_aggregation(tmp_path_factory, capsys) -> None:
    url, scen = _db(tmp_path_factory, "b222_nsd", [
        _blocks({"t0030": 48.0}),
        ["solve", "y2020_2day_dispatch", "new_stepduration", b64(3.0, "float")],
    ])
    steps = run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    assert out.count(LATE_WARNING) == 1, out
    # 43 steps of 1 h from t0030 -> 14 chunks of 3 h + one of 1 h
    st = list(steps.values())[-1]
    assert st.flex_data.dt.height == 15


def test_overlong_representative_period_block_is_an_error(tmp_path_factory) -> None:
    rp = b64({"index_type": "str", "index_name": "base_period",
              "data": [["t0030", {"type": "map", "index_type": "str", "index_name": "representative_period",
                                  "data": [["t0030", 1.0]]}]]}, "map")
    url, scen = _db(tmp_path_factory, "b222_rp", [
        _blocks({"t0030": 48.0}), ["timeset", "2day", "representative_period_weights", rp]])
    with pytest.raises(FlexToolConfigError) as err:
        _check(url, scen)
    msg = str(err.value)
    assert "'2day'" in msg and "'t0030'" in msg and "only 43 time steps" in msg and "cannot be cut" in msg, msg


def test_coal_wind_min_uptime_fixture_warns(tmp_path_factory, capsys) -> None:
    """``coal_min_updowntime`` keeps ``2day = {t0001: 168}`` on the 72-step timeline: 96 steps are cut."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp("b222_fixture")
    (work / "tests.json").write_text(json.dumps(json.loads(BASE_FIXTURE_JSON.read_text())))
    url = json_to_db(work / "tests.json", work / "tests.sqlite")
    migrate_database(url)
    _check(url, "coal_wind_min_uptime")
    out = capsys.readouterr().out
    assert out.count(
        "Warning: timeset '2day' (solve 'y2020_2day_dispatch'): the block starting 't0001' has 168 time steps, "
        "but timeline 'y2020' has only 72 from 't0001' (to 't0072'); the last 96 are not used.") == 1, out
