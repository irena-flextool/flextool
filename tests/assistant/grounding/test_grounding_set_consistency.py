"""The question set, the answer key and the grounding inputs fit together (P3 test 7).

The set is checked always.  The key is checked in two forms: a placeholder key made here (one entry per question,
values that only exercise the checks, never answers) and, once the independent oracle has written it,
``tools/assistant_eval_data/answer_key.yaml``.  Names are checked against the database ``build_inputs`` makes
(not solved); a number fact read from a result table must carry that table's ``_output_meta`` unit (critique B1);
a lookup question's expected tool path must fit the basic level's six calls (M3; three until 2026-11-08).
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.assistant import grounding_projects as gp

#: The classes of the first round (P3 §6.3, P3-O18).
CLASS_TOTALS = {"lookup": 9, "interpret": 2, "diagnose": 2}


@pytest.fixture(scope="module")
def names(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    from flextool.update_flextool.db_handle import open_database_mapping

    db = gp.build_inputs(tmp_path_factory.mktemp("grounding_inputs"))
    with open_database_mapping(f"sqlite:///{db.as_posix()}") as db_map:
        return {"scenarios": {s["name"] for s in db_map.get_items("scenario")},
                "entities": {e["name"] for e in db_map.get_items("entity")}}


def placeholder_key(qset) -> dict:
    """A key in the oracle's format whose values only exercise the checks (``placeholder: true``)."""
    note = "placeholder: the oracle's key replaces it"
    entries = {qid: {"facts": [{"id": "subject", "kind": "token", "value": "west", "derivation": note}],
                     "tools_any": [["list_runs"], ["query_result"]], "manual": ["facts_vs_assumptions"]}
               for qid in qset.questions}
    entries["G04"]["facts"].append({"id": "per_year", "kind": "number", "value": 1.0, "table": "node_slack_up_d_e",
                                    "unit": "MWh/a", "scale_units": ["MWh/a", "GWh/a", "TWh/a"],
                                    "tolerance": {"rel": 0.001}, "critical": True, "derivation": note})
    entries["G04"]["facts"].append({"id": "node", "kind": "entity", "value": "west", "derivation": note})
    entries["G05"]["facts"].append({"id": "cost", "kind": "number", "value": 1.0, "table": "costs_dt_p",
                                    "unit": "CUR", "critical": True, "derivation": note})
    entries["G07"]["successful_run_first"] = {"folder": "demand_update_test_1", "reason_any": [["period"]]}
    entries["G10"].update(expect_unknown=True, unknown_units=["CUR/MWh"])
    entries["G08"]["facts"].append({"id": "chain", "kind": "statement", "criterion": note, "derivation": note})
    entries["S01"]["facts"].append({"id": "turn8", "kind": "entity", "value": "coal", "turn": 8, "derivation": note})
    return {"format": 1, "set_version": qset.set_version, "key_version": 0, "placeholder": True,
            "derived_by": note, "sources": [], "questions": entries}


def test_question_set(scoring, names) -> None:
    qset = scoring.load_set()
    assert qset.set_version == gp.SET_VERSION
    assert qset.project["builder"] == "tests.assistant.grounding_projects:build_grounding_project"
    assert qset.project["input_source"] == Path(gp.SOURCE_NAME).stem
    classes = [q.cls for q in qset.questions.values()]
    assert {c: classes.count(c) for c in CLASS_TOTALS} == CLASS_TOTALS
    assert [q.id for q in qset.questions.values() if q.automatic_level_only] == ["S01"]
    assert len(qset.questions["S01"].turns) == 8
    assert scoring.consistency_problems(qset, None, **names) == []


def test_placeholder_key_fits(scoring, names) -> None:
    qset = scoring.load_set()
    key = scoring.parse_key(placeholder_key(qset))
    assert key.placeholder
    assert scoring.consistency_problems(qset, key, **names) == []


@pytest.mark.parametrize("change, problem", [
    (lambda k: k["questions"]["G04"]["facts"][1].update(unit="MWh", scale_units=["MWh"]),
     "G04.per_year: unit 'MWh', node_slack_up_d_e is in 'MWh/a'"),
    (lambda k: k["questions"].pop("G09"), "G09 has no key entry"),
    (lambda k: k["questions"].update(G99=k["questions"]["G01"]), "the key has G99, which is not a question"),
    (lambda k: k["questions"]["G04"]["facts"][2].update(value="east_village"),
     "G04.node: entity east_village is not in the grounding inputs"),
    (lambda k: k["questions"]["G01"].update(tools_any=[["list_runs"], ["get_entity"], ["get_parameter"],
                                                       ["query_result"], ["read_doc"], ["search_docs"],
                                                       ["list_sources"]]),
     "G01: the expected path has 7 calls, the basic level allows 6"),
    (lambda k: k["questions"]["G03"].update(tools_any=[["get_slack_summary"]]),
     "G03: no basic tool in the expected call ('get_slack_summary',)"),
    (lambda k: k["questions"]["G05"]["facts"][1].update(table="no_such_table"),
     "G05.cost: table no_such_table is not a FlexTool result table"),
    (lambda k: k["questions"]["S01"]["facts"][1].update(turn=9), "S01.turn8: turn 9, the question has 8"),
])
def test_key_problems_are_found(scoring, names, change, problem) -> None:
    qset = scoring.load_set()
    data = copy.deepcopy(placeholder_key(qset))
    change(data)
    assert problem in scoring.consistency_problems(qset, scoring.parse_key(data), **names)


@pytest.mark.parametrize("fact, message", [
    ({"id": "x", "kind": "number", "value": 1.0}, "every fact needs a derivation"),
    ({"id": "x", "kind": "number", "value": "many", "derivation": "d"}, "value must be a finite number"),
    ({"id": "x", "kind": "statement", "derivation": "d"}, "a statement fact needs a criterion"),
    ({"id": "x", "kind": "guess", "value": 1, "derivation": "d"}, "kind 'guess'"),
])
def test_key_format_errors(scoring, fact, message) -> None:
    data = {"format": 1, "set_version": 3, "key_version": 1, "questions": {"G01": {"facts": [fact]}}}
    with pytest.raises(scoring.FormatError, match=message):
        scoring.parse_key(data)


def test_oracle_key(scoring, names) -> None:
    if not scoring.KEY_PATH.is_file():
        pytest.skip("answer_key.yaml is written by the independent oracle (P3 commit 9)")
    qset = scoring.load_set()
    key = scoring.load_key()
    assert not key.placeholder
    assert key.derived_by and key.sources
    assert scoring.consistency_problems(qset, key, **names) == []
