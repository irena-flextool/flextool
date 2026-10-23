"""The reserve method partitions follow the combined method strings.

``reserve__upDown__group.reserve_method`` is ``timeseries_only``, ``dynamic_only``, ``large_failure_only`` or a
combination (``timeseries_and_dynamic``, ``timeseries_and_large_failure``, ``dynamic_and_large_failure``,
``all``); a group belongs to every requirement whose name its method contains.  Two producers build the three
partitions (``reserve_upDown_group_method_{timeseries,dynamic,n_1}``): the solve_data CSVs
(``emit/_mid_sets.derive_reserve_method_partition``) and the DB projection
(``params/_projection.reserve_upDown_group_method``), which overwrites the CSV result.  The projection used to
keep only one exact string per partition (``timeseries_only``), so a combined-method group lost its timeseries
requirement whenever another group used ``timeseries_only``.

Fixture: ``tests.json`` (``coal``, ``wind``) plus an override alternative, built from JSON.  Every reserve has
a non-zero ``reservation`` on some group, so its relationships are active.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, build, make_db, run, utf8

REL = "reserve__upDown__unit__node"
WIND = "wind_plant"

# (reserve, group method, reservation); each reserve r has one group (r, up, electricity).
GROUPS = [
    ("ts", "timeseries_only", 10.0),
    ("comb", "timeseries_and_dynamic", 7.0),
    ("tlf", "timeseries_and_large_failure", 6.0),
    ("dl", "dynamic_and_large_failure", 5.0),
    ("allm", "all", 4.0),
    ("nr", "no_reserve", 3.0),
]
EXPECTED = {
    "timeseries": {"ts", "comb", "tlf", "allm"},
    "dynamic": {"comb", "dl", "allm"},
    "n_1": {"tlf", "dl", "allm"},
}
METHOD = {r: m for r, m, _ in GROUPS}


def _db_values() -> tuple[list, list]:
    ents: list = []
    vals: list = []
    for r, method, reservation in GROUPS:
        grp = [r, "up", "electricity"]
        ents += [["reserve", r, None], ["reserve__upDown__group", grp, None],
                 [REL, [r, "up", PLANT, "west"], None], [REL, [r, "up", WIND, "west"], None]]
        vals += [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
                 ["reserve__upDown__group", grp, "reservation", b64(reservation, "float")],
                 ["reserve__upDown__group", grp, "penalty_reserve", b64(1000.0, "float")],
                 [REL, [r, "up", PLANT, "west"], "is_enabled", b64("yes", "str")],
                 [REL, [r, "up", PLANT, "west"], "max_share", b64(1.0, "float")],
                 [REL, [r, "up", PLANT, "west"], "reliability", b64(1.0, "float")],
                 [REL, [r, "up", PLANT, "west"], "large_failure_ratio", b64(0.1, "float")],
                 [REL, [r, "up", WIND, "west"], "is_enabled", b64("yes", "str")],
                 [REL, [r, "up", WIND, "west"], "max_share", b64(0.0, "float")],
                 [REL, [r, "up", WIND, "west"], "increase_reserve_ratio", b64(0.1, "float")]]
    return ents, vals


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    ents, vals = _db_values()
    url, scenario = make_db(tmp_path_factory, alt="rmp", values=vals,
                            base_chain=PLANT_CHAIN + ["wind"], entities=ents)
    step = list(run(tmp_path_factory, url, scenario).values())[-1]
    return url, scenario, step


def _quads(frame: pl.DataFrame) -> set[tuple[str, str, str, str]]:
    return set(utf8(frame).select("r", "ud", "g", "method").iter_rows())


def _expected_quads(part: str) -> set[tuple[str, str, str, str]]:
    return {(r, "up", "electricity", METHOD[r]) for r in EXPECTED[part]}


@pytest.mark.parametrize("part", ["timeseries", "dynamic", "n_1"])
def test_partition_matches_the_method_strings(solved, part) -> None:
    """The model's partition, the DB projection and the solve_data CSV agree as sets of (r, ud, g, method),
    with the method as authored, and no_reserve is in none of them."""
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.model import _reserve
    from flextool.engine.preprocess.params._projection import reserve_upDown_group_method

    url, scenario, step = solved
    field = f"reserve_upDown_group_method_{part}"
    model_frame = getattr(step.flex_data, field)
    csv_frame = _reserve.load_data(Path("input"), Path("solve_data"), provider=step.flex_data_provider)[field]
    projected = reserve_upDown_group_method(SpineDbReader(url, scenario), part)
    expected = _expected_quads(part)
    assert _quads(model_frame) == expected
    assert _quads(csv_frame) == expected
    assert _quads(projected) == expected
    assert "nr" not in set(utf8(model_frame)["r"].to_list())
    # The partition joins the model's (r, ud, g) sets: same key dtypes as reserve_upDown_group and the CSV.
    rug = step.flex_data.reserve_upDown_group
    for col in ("r", "ud", "g"):
        assert model_frame.schema[col] == rug.schema[col], col
        assert model_frame.schema[col] == csv_frame.schema[col], col


def test_every_requirement_row_is_built(solved) -> None:
    """Each group gets the rows of every requirement its method names: the combined-method groups keep their
    timeseries row next to the timeseries_only group."""
    _url, _scenario, step = solved
    pb = build(step.flex_data)
    m = pb.canonicalise()

    def groups(family: str) -> set[str]:
        return {nm[len(family) + 1:].split(",")[0] for nm in m.row_names if nm.startswith(family + "[")}

    assert groups("reserveBalance_timeseries_eq") == EXPECTED["timeseries"]
    assert groups("reserveBalance_dynamic_eq") == EXPECTED["dynamic"]
    assert groups("reserveBalance_up_n_1_eq") == EXPECTED["n_1"]
