"""Construction lead time behaviour-change notice: an upgraded database (carrying the marker) with a lead time
prints one notice per run naming the entities with a lead time, also over a chain of solves; without the marker,
or without a lead time, nothing is printed."""
from __future__ import annotations

import pytest
from spinedb_api import DatabaseMapping

from tests.engine._unit_db import PLANT, make_db
from tests.engine.test_construction_lead_time_limits import ENTITIES, _values
from tests.engine.test_user_constraint_capacity_terms import CHAIN

pytestmark = pytest.mark.solver

MARKER = "flextool_behaviour_change:construction_lead_time_commissioning"
NOTICE = "With a construction lead time, 'invested' and the per-period investment limits count capacity coming online"


def _run_output(tmp_path_factory, capsys, alt: str, lead: float, marker: bool) -> str:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=_values(lead, {"p2025": 100.0, "p2030": 50.0}),
                        base_chain=CHAIN, entities=ENTITIES)
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=MARKER, value="77")
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    return capsys.readouterr().out


def test_notice_printed_once_per_run(tmp_path_factory, capsys) -> None:
    out = _run_output(tmp_path_factory, capsys, "lt_notice", 5.0, marker=True)
    lines = [ln for ln in out.splitlines() if NOTICE in ln]
    assert len(lines) == 1, out
    assert f"Affected entities: {PLANT}." in lines[0] and MARKER in lines[0]


@pytest.mark.parametrize("lead, marker", [(5.0, False), (0.0, True)], ids=["no_marker", "no_lead_time"])
def test_no_notice(tmp_path_factory, capsys, lead, marker) -> None:
    out = _run_output(tmp_path_factory, capsys, f"lt_notice_{lead}_{marker}", lead, marker=marker)
    assert NOTICE not in out
