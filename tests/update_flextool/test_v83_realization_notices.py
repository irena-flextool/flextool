"""Part 3 of v83: the behaviour-change markers W-0635 (``rolling_within_solve_one_solve``) and W-0636
(``separate_dispatch_realization_looks_ahead``) list the solves whose realization changes (plan §2a "Notices", MR
design rev 7.1 §6.2).

The rule, derived from those two texts (not from the code): per solve and scenario (the scenario's ranked
alternatives, the highest-ranked row wins, alternatives no scenario uses never count), *stochastic* = the winning
``stochastic_branches`` is non-empty (one forecast counts), *rolling* = the winning explicit ``solve_mode`` is
``rolling_window``, *method* = the winning explicit ``dispatch_realization_method`` (part 5's writes count; no
definition default is read).  W-0635 lists a solve that is rolling, stochastic and ``within_solve`` in one scenario;
W-0636 one that is stochastic and ``separate_dispatch`` in one scenario, rolling or single.  A method that resolves
only in a scenario where the solve is not stochastic lists nothing (DET-m review N6).  An undecided part-5 solve has
no method and gets nothing.

Pure planner rows first, then the batch on a database made from the v82 schema (``v82_schema_db_url``, copied into
``tmp_path``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, to_database
from spinedb_api.exception import NothingToCommit

from flextool.update_flextool import db_migration as m

W635 = "flextool_behaviour_change:rolling_within_solve_one_solve"
W636 = "flextool_behaviour_change:separate_dispatch_realization_looks_ahead"
R, S = "rolling_window", "single_solve"
WS, SD = "within_solve", "separate_dispatch"

# Each row: branches {(solve, alternative): non-empty}, modes, methods {(solve, alternative): value}, scenarios,
# expected W-0635 solves, expected W-0636 solves.  The derivation of each row is its comment.
ROWS = {
    # S1 = [base]: rolling, stochastic, within_solve → W-0635.
    "rolling_within": ({("v", "base"): True}, {("v", "base"): R}, {("v", "base"): WS}, {"S1": ["base"]},
                       ("v",), ()),
    # Rolling, stochastic, separate_dispatch → W-0636.
    "rolling_separate": ({("v", "base"): True}, {("v", "base"): R}, {("v", "base"): SD}, {"S1": ["base"]},
                         (), ("v",)),
    # No solve_mode (a single solve), separate_dispatch → W-0636 (single solves change too).
    "single_separate": ({("v", "base"): True}, {}, {("v", "base"): SD}, {"S1": ["base"]}, (), ("v",)),
    # An explicit single_solve with within_solve keeps its separate realization → nothing.
    "single_within": ({("v", "base"): True}, {("v", "base"): S}, {("v", "base"): WS}, {"S1": ["base"]}, (), ()),
    # deterministic → nothing.
    "deterministic": ({("v", "base"): True}, {("v", "base"): R}, {("v", "base"): "deterministic"},
                      {"S1": ["base"]}, (), ()),
    # No method resolves (an undecided part-5 solve) → nothing.
    "no_method": ({("v", "base"): True}, {("v", "base"): R}, {}, {"S1": ["base"]}, (), ()),
    # A winning null method row reads as no method → nothing.
    "null_method": ({("v", "base"): True}, {("v", "base"): R}, {("v", "base"): WS, ("v", "H"): None},
                    {"S1": ["base", "H"]}, (), ()),
    # No stochastic_branches → nothing, whatever the method.
    "no_branches": ({}, {("v", "base"): R}, {("v", "base"): SD}, {"S1": ["base"]}, (), ()),
    # The winning stochastic_branches is an empty map (a1 above base) → not stochastic in S1 → nothing.
    "empty_branches_win": ({("v", "base"): True, ("v", "a1"): False}, {("v", "base"): R}, {("v", "base"): SD},
                           {"S1": ["base", "a1"]}, (), ()),
    # N6: branches only in a1; the method resolves in S1 = [base] where v is not stochastic, and in S2 = [base, a1]
    # where it is → listed (from S2).
    "stochastic_in_one_scenario": ({("v", "a1"): True}, {("v", "base"): R}, {("v", "base"): WS},
                                   {"S1": ["base"], "S2": ["base", "a1"]}, ("v",), ()),
    # N6: the same with no scenario using a1 → stochastic nowhere → nothing.
    "stochastic_only_in_unused": ({("v", "a1"): True}, {("v", "base"): R}, {("v", "base"): WS},
                                  {"S1": ["base"]}, (), ()),
    # rolling_window only in R: S1 = [base] is single (within_solve keeps its realization), S2 = [base, R] is rolling
    # → listed (from S2).
    "rolling_in_one_scenario": ({("v", "base"): True}, {("v", "R"): R}, {("v", "base"): WS},
                                {"S1": ["base"], "S2": ["base", "R"]}, ("v",), ()),
    # The three conditions must hold in the same scenario: rolling only in S1 = [base, R] (no branches there: a1 is
    # not in S1), stochastic only in S2 = [base, a1] (single) → nothing.
    "conditions_in_different_scenarios": ({("v", "a1"): True}, {("v", "R"): R}, {("v", "base"): WS},
                                          {"S1": ["base", "R"], "S2": ["base", "a1"]}, (), ()),
    # A winning null solve_mode reads as no mode (single) → within_solve lists nothing.
    "null_mode": ({("v", "base"): True}, {("v", "base"): R, ("v", "H"): None}, {("v", "base"): WS},
                  {"S1": ["base", "H"]}, (), ()),
    # within_solve in S1, separate_dispatch (E) in S2, rolling in both → listed in both markers.
    "both_methods": ({("v", "base"): True}, {("v", "base"): R}, {("v", "base"): WS, ("v", "E"): SD},
                     {"S1": ["base"], "S2": ["base", "E"]}, ("v",), ("v",)),
    # Two solves, sorted: w (rolling within_solve) and v (single separate_dispatch).
    "two_solves": ({("w", "base"): True, ("v", "base"): True}, {("w", "base"): R},
                   {("w", "base"): WS, ("v", "base"): SD}, {"S1": ["base"]}, ("w",), ("v",)),
}


@pytest.mark.parametrize("row", sorted(ROWS))
def test_planner(row: str) -> None:
    branches, modes, methods, scenarios, within, separate = ROWS[row]
    assert m.plan_realization_notices(branches, modes, methods, scenarios) == (within, separate)


# --- on a database --------------------------------------------------------------------------------------------
def _copy(v82_schema_db_url: str, folder: Path) -> str:
    path = folder / "db.sqlite"
    shutil.copy2(v82_schema_db_url.removeprefix("sqlite:///"), path)
    return f"sqlite:///{path.resolve().as_posix()}"


def _one_forecast() -> Map:
    """One forecast and the realized branch in p2020 (period → branch → analysis_time → realized → weight)."""
    def fan(realized: str) -> Map:
        return Map(["t0001"], [Map([realized], [1.0])])
    return Map(["p2020"], [Map(["f1", "real"], [fan("no"), fan("yes")])])


def _add(db, solve: str, name: str, alternative: str, value) -> None:
    raw, typ = to_database(value)
    _item, error = db.add_item("parameter_value", entity_class_name="solve", entity_byname=(solve,),
                               parameter_definition_name=name, alternative_name=alternative, value=raw, type=typ)
    assert not error, error


def _markers(url: str) -> dict[str, dict]:
    with DatabaseMapping(url) as db:
        return {md["name"]: json.loads(md["value"]) for md in db.get_items("metadata")
                if md["name"] in (W635, W636)}


def _seed(url: str) -> None:
    """Scenarios S1 = [base], S2 = [E, base] (base wins); solves:

    - ``roll_ws``: rolling, one forecast, no method → part 5 writes within_solve into base → W-0635.
    - ``roll_sd``: rolling, explicit separate_dispatch → W-0636.
    - ``single_sd``: single (no solve_mode), separate_dispatch → W-0636.
    - ``single_ws``: single, no method → part 5 writes within_solve → nothing (a single within_solve keeps its
      realization).
    - ``undecided``: rolling, branches in base, separate_dispatch in E only.  S1 has no method (a part-5 candidate,
      target base); in S2 base outranks E, so a within_solve in base would replace E's method there → undecided,
      nothing written.  S1 then lists nothing (no method); in S2 E's separate_dispatch wins (base has no method
      row) and the solve is stochastic → W-0636, from S2 only.
    - ``default_rolling``: no solve_mode of its own under a ``rolling_window`` definition default, within_solve →
      not rolling (the default is not read) → nothing.
    """
    with DatabaseMapping(url) as db:
        db.add_item("alternative", name="E")
        for scenario, alts in {"S1": ["base"], "S2": ["E", "base"]}.items():
            db.add_item("scenario", name=scenario)
            for rank, alternative in enumerate(alts, start=1):
                db.add_item("scenario_alternative", scenario_name=scenario, alternative_name=alternative, rank=rank)
        for solve in ("roll_ws", "roll_sd", "single_sd", "single_ws", "undecided", "default_rolling"):
            db.add_item("entity", entity_class_name="solve", name=solve)
            _add(db, solve, "stochastic_branches", "base", _one_forecast())
        for solve in ("roll_ws", "roll_sd", "undecided"):
            _add(db, solve, "solve_mode", "base", R)
        _add(db, "roll_sd", "dispatch_realization_method", "base", SD)
        _add(db, "single_sd", "dispatch_realization_method", "base", SD)
        _add(db, "undecided", "dispatch_realization_method", "E", SD)
        _add(db, "default_rolling", "dispatch_realization_method", "base", WS)
        pdef = db.get_item("parameter_definition", entity_class_name="solve", name="solve_mode")
        raw, typ = to_database(R)
        db.update_item("parameter_definition", id=pdef["id"], default_value=raw, default_type=typ)
        db.commit_session("seed")


def test_batch_lists_the_solves(v82_schema_db_url: str, tmp_path: Path) -> None:
    """The batch (part 5, then part 3) on :func:`_seed`; the rerun on its own result changes no marker."""
    url = _copy(v82_schema_db_url, tmp_path)
    _seed(url)
    with DatabaseMapping(url) as db:
        m._migrate_v83_schema_batch(db)
    with DatabaseMapping(url) as db:  # part 5 wrote within_solve on roll_ws and single_ws; default_rolling's is seeded
        written = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name="solve", parameter_definition_name="dispatch_realization_method")
            if pv["parsed_value"] == WS and pv["alternative_name"] == "base"}
    assert written == {"roll_ws", "single_ws", "default_rolling"}
    expected = {W635: {"since": 83, "solves": ["roll_ws"]},
                W636: {"since": 83, "solves": ["roll_sd", "single_sd", "undecided"]}}
    assert _markers(url) == expected
    with DatabaseMapping(url) as db, pytest.raises(NothingToCommit):
        m._v83_realization_notices(db)
        db.commit_session("rerun")
    assert _markers(url) == expected


def test_no_marker_without_a_listed_solve(v82_schema_db_url: str, tmp_path: Path) -> None:
    """A single within_solve solve and a rolling within_solve one without stochastic_branches: nothing listed, no
    marker written."""
    url = _copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        for solve in ("a", "b"):
            db.add_item("entity", entity_class_name="solve", name=solve)
        _add(db, "a", "stochastic_branches", "base", _one_forecast())
        _add(db, "a", "dispatch_realization_method", "base", WS)
        _add(db, "b", "solve_mode", "base", R)
        _add(db, "b", "dispatch_realization_method", "base", WS)
        db.commit_session("seed")
    with DatabaseMapping(url) as db:
        m._migrate_v83_schema_batch(db)
    assert _markers(url) == {}


def test_a_stored_list_is_kept(v82_schema_db_url: str, tmp_path: Path) -> None:
    """A marker already holding another solve keeps it; the new solve is appended."""
    url = _copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        db.add_item("entity", entity_class_name="solve", name="v")
        _add(db, "v", "stochastic_branches", "base", _one_forecast())
        _add(db, "v", "dispatch_realization_method", "base", SD)
        db.add_item("metadata", name=W636, value=json.dumps({"since": 83, "solves": ["old"]}))
        db.commit_session("seed")
    with DatabaseMapping(url) as db:
        m._migrate_v83_schema_batch(db)
    assert _markers(url) == {W636: {"since": 83, "solves": ["old", "v"]}}
