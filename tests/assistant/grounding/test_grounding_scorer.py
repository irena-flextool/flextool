"""The automatic checks of a grounding cell (P3 test 6): number formats, numbers without a tool source, what the
source rule leaves out, forbidden patterns, "the data does not answer", both escalation cases, the failed-run
case (G07) and the cell outcome.  Synthetic answers and keys; no model, no project."""
from __future__ import annotations

import pytest

ROWS = '{"rows": [["p2020", "west", 25000.0], ["t0043", "west", 372.0], ["p2020", "coal_1", 1033497.5]]}'


def turn(scoring, answer: str, *, results: tuple[str, ...] = (ROWS,), user: str = "How much?", **extra):
    calls = tuple({"name": "query_result", "arguments": {}, "ok": True, "result": r} for r in results)
    return scoring.TurnView(user=user, answer=answer, tool_calls=calls, **extra)


def entry(scoring, **fields):
    return scoring.parse_key({"format": 1, "set_version": 3, "key_version": 1,
                              "questions": {"Q": fields}}).questions["Q"]


def question(scoring, cls: str = "lookup"):
    return scoring.Question(id="Q", cls=cls, category="test", context={}, turns=("How much?",))


def by_id(items):
    return {i.id: i for i in items}


@pytest.mark.parametrize("text", ["25000", "25 000", "25 000", "25 000", "25,000", "25000.0", "2.5e4",
                                  "2.5 × 10^4", "2.5×10⁴", "25k", "25 thousand"])
def test_number_formats(scoring, text: str) -> None:
    found = scoring.find_numbers(f"The cost was {text} CUR.")
    assert [n.scaled for n in found] == [25000.0]


def test_number_fact_with_scale_units(scoring) -> None:
    key = entry(scoring, facts=[{"id": "slack", "kind": "number", "value": 1033497.5, "unit": "MWh/a",
                                 "scale_units": ["MWh/a", "GWh/a", "TWh/a"], "tolerance": {"rel": 0.001},
                                 "critical": True, "derivation": "test"}])
    for answer in ("1 033 497.5 MWh/a", "1033.4975 GWh/a", "1.0334975 TWh per year", "1.0335 million MWh/a"):
        items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, answer)], max_calls=3))
        assert items["fact:slack"].state == "pass", answer
        assert items["units"].state == "pass", answer
    items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, "1033497.5 in total")],
                                     max_calls=3))
    assert items["fact:slack"].state == "pass" and items["units"].state == "fail"
    items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, "1.2 TWh/a")], max_calls=3))
    assert items["fact:slack"].state == "fail" and items["units"].state == "n/a"


def test_unsourced_number_goes_to_review_with_candidates(scoring) -> None:
    items = scoring.check_cell(question(scoring), entry(scoring), "small",
                               [turn(scoring, "The slack peaks at 372 MW and costs 26000 CUR.")], max_calls=3)
    review = [i for i in items if i.state == "review"]
    assert [i.check for i in review] == ["tool_sourced"]
    assert review[0].excerpt.startswith("26000")
    assert review[0].candidates[0] == 25000.0
    assert review[0].critical


def test_identifiers_years_constants_question_numbers_and_list_markers_are_left_out(scoring) -> None:
    answer = ("1. In t0017 of p2020 (folder coal_1, node h2_tank) in 2030, with 8 760 hours a year, 1000 kW per MW "
              "and 24 h a day, the share is 100 %. You asked about 48 hours.")
    items = scoring.check_cell(question(scoring), entry(scoring), "small",
                               [turn(scoring, answer, user="What about the 48 hours?")], max_calls=3)
    assert by_id(items)["tool_sourced"].state == "pass"


def test_rounded_and_scaled_tool_numbers_are_sourced(scoring) -> None:
    items = scoring.check_cell(question(scoring), entry(scoring), "small",
                               [turn(scoring, "About 1.03 million MWh/a, or 1 033 GWh/a; 25 thousand CUR.")],
                               max_calls=3)
    assert by_id(items)["tool_sourced"].state == "pass"


def test_allowed_derived_value_is_sourced(scoring) -> None:
    key = entry(scoring, allowed_derived=[{"value": 5663.0, "note": "sample total"}])
    items = scoring.check_cell(question(scoring), key, "small", [turn(scoring, "In the sample: 5 663 MWh.")],
                               max_calls=3)
    assert by_id(items)["tool_sourced"].state == "pass"


def test_earlier_turn_tool_results_count_later(scoring) -> None:
    turns = [turn(scoring, "Looked it up."), turn(scoring, "It was 372 MW.", results=())]
    items = scoring.check_cell(question(scoring), entry(scoring), "small", turns, max_calls=3)
    assert by_id(items)["tool_sourced"].state == "pass"


def test_forbidden_pattern(scoring) -> None:
    key = entry(scoring, forbidden=[{"pattern": r"(?i)\bbinding\b.*\bprov", "note": "binding as proven",
                                     "critical": True}])
    items = by_id(scoring.check_cell(question(scoring), key, "large",
                                     [turn(scoring, "The capacity limit is binding, which proves it.")], max_calls=30))
    assert items["forbidden:0"].state == "fail" and items["forbidden:0"].critical
    items = by_id(scoring.check_cell(question(scoring), key, "large",
                                     [turn(scoring, "This is consistent with a binding limit.")], max_calls=30))
    assert items["forbidden:0"].state == "pass"


def test_unknown(scoring) -> None:
    key = entry(scoring, expect_unknown=True, unknown_units=["CUR/MWh"])
    cases = {"The data does not cover 2030; only period p2020 exists.": "pass",
             "The price in 2030 was high.": "fail",
             "The data does not answer this for 2030; the price was 50 CUR/MWh.": "review"}
    for answer, state in cases.items():
        items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, answer, results=())],
                                         max_calls=3))
        assert items["unknown"].state == state, answer


def _graded(scoring, cls: str, key, answer: str, level: str = "small"):
    """The cell's grade items and its outcome with every judged item given *verdicts* (the item's first allowed
    verdict unless named)."""
    items, _ = scoring.grading_items(question(scoring, cls), key, level, [turn(scoring, answer)], max_calls=3)

    def outcome(**verdicts):
        return scoring.outcome(items, {i.id: verdicts.get(i.id.replace(":", "_"), i.allowed[0])
                                       for i in items if not i.auto})
    return {i.id: i for i in items}, outcome


def test_escalation_above_the_level_is_clean(scoring) -> None:
    answer = ("This needs a deeper analysis than the basic level does. A higher assistant level or a more capable "
              "model can trace it. So far: the slack peaks at 372 MW.")
    key = entry(scoring, facts=[{"id": "cause", "kind": "token", "value": "coal_plant", "critical": True,
                                 "derivation": "test"}])
    items, outcome = _graded(scoring, "diagnose", key, answer)
    assert items["escalation"].auto
    assert outcome(fact_cause="missing") == "escalated"                    # what it left out is ignored
    items, outcome = _graded(scoring, "diagnose", key, answer.replace("372", "999"))
    assert outcome(fact_cause="missing", number_1_0="right") == "escalated"
    assert outcome(fact_cause="missing", number_1_0="wrong") == "failed"   # an invented number
    assert scoring.verdict_counts(list(items.values()), {"number:1:0": "wrong", "fact:cause": "missing"}
                                  )["invented_numbers"] == 1


def test_escalation_within_the_level_is_judged_and_never_answered(scoring) -> None:
    answer = "This needs a deeper analysis than the basic level does. A higher assistant level can trace it."
    items, outcome = _graded(scoring, "lookup", entry(scoring), answer)
    assert items["escalation"].allowed == ("right", "wrong")
    assert outcome(escalation="right") == "escalated"
    assert outcome(escalation="wrong") == "failed"


def test_failed_run_case(scoring) -> None:
    key = entry(scoring, successful_run_first={"folder": "demand_update_test_1",
                                               "reason_any": [["period"], ["annual_flow"]],
                                               "result_units": ["MWh"]})
    cases = {"The latest run failed: the inflow is keyed by period; use annual_flow and read the run log.": "pass",
             "The latest run failed, so there are no results.": "fail",
             "Unserved energy was 1 033 497.5 MWh/a.": "fail",
             "The run failed (inflow keyed by period, use annual_flow); an older run showed 1 033 497.5 MWh.": "review"}
    for answer, state in cases.items():
        items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, answer)], max_calls=3))
        assert items["successful_run_first"].state == state, answer


def test_tools_and_health(scoring) -> None:
    key = entry(scoring, tools_any=[["list_runs"], ["query_result"]])
    items = by_id(scoring.check_cell(question(scoring), key, "small", [turn(scoring, "25000 CUR.")], max_calls=3))
    assert items["tools"].state == "fail"           # list_runs not called
    broken = turn(scoring, "25000 CUR.", error={"code": "context", "message": "still too long"})
    items = by_id(scoring.check_cell(question(scoring), entry(scoring), "small", [broken], max_calls=3))
    assert items["health"].state == "fail" and not items["health"].critical
    many = scoring.TurnView(user="q", answer="25000", tool_calls=tuple(
        {"name": "query_result", "arguments": {}, "ok": True, "result": ROWS} for _ in range(4)))
    items = by_id(scoring.check_cell(question(scoring), entry(scoring), "small", [many], max_calls=3))
    assert items["tools"].state == "fail"


def test_cell_outcome(scoring) -> None:
    key = entry(scoring, facts=[
        {"id": "value", "kind": "number", "value": 25000, "unit": "CUR", "critical": True, "derivation": "test"},
        {"id": "node", "kind": "entity", "value": "west", "critical": False, "derivation": "test"},
        {"id": "why", "kind": "statement", "value": None, "criterion": "names the fuel price", "critical": False,
         "derivation": "test"}], manual=["facts_vs_assumptions"])
    items, outcome = _graded(scoring, "lookup", key,
                             "The fuel cost at west was 25 000 CUR in t0001 (fuel price times flow).")
    assert {i for i, x in items.items() if not x.auto} == {"fact:why", "facts_vs_assumptions"}
    assert scoring.outcome(list(items.values()), {"fact:why": "right"}) is None          # one still waits
    assert outcome() == "answered"
    assert outcome(fact_why="missing") == "incomplete"
    assert outcome(facts_vs_assumptions="missing") == "answered"           # form only (form_missing)
    items, outcome = _graded(scoring, "lookup", key, "The fuel cost at west was 1033497.5 CUR.")
    assert not items["fact:value"].auto and "1.0335e+06 CUR" in items["fact:value"].hint
    assert outcome(fact_value="wrong") == "failed"
    assert outcome(fact_value="missing") == "incomplete"
    # tools and health are diagnostics: a rate-limit error does not change the outcome
    weak = turn(scoring, "25 000 CUR at west; fuel price.", error={"code": "rpm", "message": "limited"})
    got, diagnostics = scoring.grading_items(question(scoring), key, "small", [weak], max_calls=3)
    assert diagnostics["health"] == "fail"
    assert scoring.outcome(got, {i.id: "right" for i in got if not i.auto}) == "answered"


def test_blinding(scoring) -> None:
    from flextool.assistant.session import SESSION_MESSAGES as M

    text = " ".join(["As an AI model, Claude says:", M["not_at_level"].format(name="get_slack_summary", names="a, b"),
                     M["turn_limit_call"].format(n=3), M["limit"], M["free_plan"].format(rounds=3),
                     "I am gpt-oss:20b, not Gemini or qwen2.5:14b; the slack was 372 MW."])
    blinded = scoring.blind(text)
    for word in ("Claude", "gpt-oss", "Gemini", "qwen", "assistant level", "Free plan", "limit of 3", "steps"):
        assert word.lower() not in blinded.lower(), word
    assert blinded.count("[tool limit reached]") == 2
    assert "[a tool call was refused]" in blinded and "[notice removed]" in blinded
    assert "[assistant]" in blinded and "the slack was 372 MW." in blinded
