"""A file the repair tool calls ``unclear`` (exit 4) gets a repair record after the W-0490 checklist (P7 follow-up).

The W-0490 body sends an ``unclear`` file to its manual checklist; the checklist's step 7 is "Run the repair tool on
the fixed copy to record the repair".  Each test builds an ``unclear`` file (``_value_list_cases.make_hit``), does
the hand fix the checklist describes with spinedb_api (as the Spine DB editor does: a new member goes at the end,
step 3; the three class defaults set to ``not_allowed``, step 4; the values decided one by one, step 5), then runs
the tool again.

Oracles (from the checklist text and the design's state rules, never from the tool's output):

- after step 3 the list holds every member of the parameter reference (the 13 shipped members and
  ``cumulative_limits``), so the record's list check holds;
- a list whose first row R is the shipped ``not_allowed`` row from the template (older than the session), with the
  list now complete and ``cumulative_limits`` present: design state ``not-hit`` → no suspect → the record alone;
- R still reads ``cumulative_limits`` with a hand-restored shipped member at the top (added after the session):
  a known lost member and older siblings → ``hit-unanchored`` → the tool asks only about the values that still
  refer to R (the defaults are the shipped default after step 4, so they are not asked); one ``keep`` → the record
  with ``kept`` 1;
- the record ends W-0490 on the repaired copy (``possible_value_list_overwrite_ro`` → ``[]``) and the tool still
  sees the session (``session_findings_ro`` → the one finding, step 22, 21 → 22, 2024-09-19).

The ``unclear`` cases with no such path (several sessions, no list, R reading ``cumulative_limits`` with every
shipped member present, or with no sibling older than the overwrite) are described in the P7 follow-up handover.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.cli import cmd_repair_value_list as cli
from flextool.update_flextool import value_list_history as vlh
from tests.update_flextool._value_list_cases import HIT_DATE, add_commit, make_hit, member_id, sql

FINDING = vlh.Finding(22, ("invest_methods",), 21, 22, HIT_DATE)
AFTER_SESSION = "2025-01-10"


def _checklist(path: Path, add: tuple[str, ...], values: dict[str, str] | None = None) -> None:
    """Steps 3-5 of the W-0490 checklist with spinedb_api: *add* the missing members at the end of
    ``invest_methods``, set the ``invest_method`` default of unit, connection and node to ``not_allowed`` and the
    unit values *values* (``{entity: member}``)."""
    from spinedb_api import to_database
    from spinedb_api.exception import NothingToCommit

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{path}") as db:
        top = max(lv["index"] for lv in db.get_items("list_value", parameter_value_list_name="invest_methods"))
        for k, member in enumerate(add, start=1):
            value, typ = to_database(member)
            db.add_list_value(parameter_value_list_name="invest_methods", value=value, type=typ, index=top + k)
        value, typ = to_database("not_allowed")
        for cls in ("unit", "connection", "node"):
            d = db.get_item("parameter_definition", entity_class_name=cls, name="invest_method")
            db.update_parameter_definition(id=d["id"], default_value=value, default_type=typ)
        for entity, member in (values or {}).items():
            pv = db.get_item("parameter_value", entity_class_name="unit", entity_byname=(entity,),
                             parameter_definition_name="invest_method", alternative_name="base")
            value, typ = to_database(member)
            db.update_parameter_value(id=pv["id"], value=value, type=typ)
        try:
            db.commit_session("hand fix")
        except NothingToCommit:
            pass


class _Ask:
    def __init__(self, *replies: str) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise EOFError
        return self.replies.pop(0)


def _ends_after_one_run(path: Path, ask=None) -> Path:
    """Run the tool on the hand-fixed *path* and return the repaired copy, checked to end W-0490."""
    assert cli.main([str(path)], ask=ask) == cli.EXIT_OK
    output = path.with_name(f"{path.stem}_repaired.sqlite")
    assert vlh.possible_value_list_overwrite_ro(output) == []
    assert vlh.session_findings_ro(output) == [FINDING]
    assert vlh.possible_value_list_overwrite_ro(path) == [FINDING], "the original keeps the warning"
    return output


@pytest.mark.parametrize("missing", ["cumulative_limits", "retire_total"])
def test_a_list_with_a_missing_member(tmp_path, schema_db_url, missing) -> None:
    """The bug did not fire (R is the template's ``not_allowed``), but one member was deleted later: no known
    pattern.  Steps 2-3 add the missing member back → ``not-hit`` → the record alone, nothing asked."""
    case = make_hit(tmp_path, schema_db_url, overwrite=False)
    sql(case.path, "DELETE FROM list_value WHERE id = ?", (member_id(case.path, "invest_methods", missing),))
    assert cli.main([str(case.path)]) == cli.EXIT_UNCLEAR
    _checklist(case.path, (missing,))
    _ends_after_one_run(case.path)


def test_an_unanchored_overwrite_with_two_members_missing(tmp_path, schema_db_url) -> None:
    """R reads ``cumulative_limits`` but was rewritten after the session (no anchor), and two shipped members are
    missing (no single lost member).  Steps 2-5: add ``not_allowed`` and ``retire_total`` at the end, set the
    defaults, U1 decided ``not_allowed``, U2 left at ``cumulative_limits`` (meant).  The rerun asks only about U2
    (still a reference to R): ``k`` then ``y`` at the confirm → the record, ``kept`` 1."""
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")],
                    after=[("unit", "U2", "base", "cumulative_limits")])
    sql(case.path, "UPDATE list_value SET commit_id = ? WHERE id = ?",
        (add_commit(case.path, "Updated", AFTER_SESSION), case.r_id))
    sql(case.path, "DELETE FROM list_value WHERE id = ?", (member_id(case.path, "invest_methods", "retire_total"),))
    assert cli.main([str(case.path)]) == cli.EXIT_UNCLEAR
    _checklist(case.path, ("not_allowed", "retire_total"), {"U1": "not_allowed"})
    ask = _Ask("k", "y")
    output = _ends_after_one_run(case.path, ask)
    assert len(ask.prompts) == 2, "one question (U2), then the confirm"
    record = vlh.read_record(output)
    assert record is not None and record.value["kept"] == 1
