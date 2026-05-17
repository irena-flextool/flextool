"""Tests for the validation warnings emitted when a group-level output
flag is ``yes`` but the group has no members of the required entity
class.

The warnings are emitted by
``flextool.input_derivation._validators.validate_group_output_memberships``
and cover four silent-no-op cases:

* ``output_nodeGroup_dispatch: yes`` with no ``group__node`` row
* ``output_nodeGroup_indicators: yes`` with no ``group__node`` row
* ``output_flowGroup_indicators: yes`` with no ``group__unit__node`` or
  ``group__connection__node`` row
* ``flow_aggregator: yes`` with no ``group__unit__node`` or
  ``group__connection__node`` row
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping

from flextool.input_derivation._validators import validate_group_output_memberships


def _build_db(tmp_path: Path, parameter_values: list[tuple[str, str]],
              memberships: dict[str, list[tuple[str, ...]]]) -> str:
    """Create a fresh SQLite DB with the four group-output parameter
    definitions plus the supplied parameter values and memberships.

    ``parameter_values`` is a list of ``(group_name, parameter_name)``
    pairs, all set to ``"yes"``.

    ``memberships`` maps entity-class name (``group__node``,
    ``group__unit__node``, ``group__connection__node``) to a list of
    byname tuples.
    """
    db_path = tmp_path / "warnings_test.sqlite"
    url = f"sqlite:///{db_path}"

    # Unique set of group names across params + memberships.
    group_names: set[str] = {g for g, _ in parameter_values}
    for rows in memberships.values():
        for row in rows:
            group_names.add(row[0])

    with DatabaseMapping(url, create=True) as db:
        # Minimal entity classes.
        db.add_update_item("entity_class", name="group")
        db.add_update_item("entity_class", name="node")
        db.add_update_item("entity_class", name="unit")
        db.add_update_item("entity_class", name="connection")
        db.add_update_item("entity_class", name="group__node",
                           dimension_name_list=("group", "node"))
        db.add_update_item("entity_class", name="group__unit__node",
                           dimension_name_list=("group", "unit", "node"))
        db.add_update_item("entity_class", name="group__connection__node",
                           dimension_name_list=("group", "connection", "node"))

        # Value list for the yes/no flags.
        db.add_update_item("parameter_value_list", name="yes_no")
        db.add_update_item("list_value", parameter_value_list_name="yes_no",
                           value=b'"yes"', type="str", index=0)
        db.add_update_item("list_value", parameter_value_list_name="yes_no",
                           value=b'"no"', type="str", index=1)

        # Parameter definitions.
        for pname in (
            "output_nodeGroup_dispatch",
            "output_nodeGroup_indicators",
            "output_flowGroup_indicators",
            "flow_aggregator",
        ):
            db.add_update_item("parameter_definition",
                               entity_class_name="group",
                               name=pname,
                               parameter_value_list_name="yes_no")

        # Entities — ensure every referenced group, node, unit, connection
        # exists before adding relationships.
        for g in group_names:
            db.add_update_item("entity", entity_class_name="group",
                               name=g, entity_byname=(g,))
        for cls_name, rows in memberships.items():
            dims = {"group__node": ("group", "node"),
                    "group__unit__node": ("group", "unit", "node"),
                    "group__connection__node": ("group", "connection", "node")}[cls_name]
            for row in rows:
                for dim_cls, dim_name in zip(dims, row):
                    if dim_cls == "group":
                        continue  # already added
                    db.add_update_item("entity", entity_class_name=dim_cls,
                                       name=dim_name, entity_byname=(dim_name,))
                db.add_update_item("entity",
                                   entity_class_name=cls_name,
                                   entity_byname=row)

        # Parameter values (all ``"yes"``).
        for group_name, param_name in parameter_values:
            db.add_update_item("parameter_value",
                               entity_class_name="group",
                               entity_byname=(group_name,),
                               parameter_definition_name=param_name,
                               alternative_name="Base",
                               value=b'"yes"', type="str")

        db.commit_session("warnings test fixture")

    return url


def _run_validation(url: str, caplog) -> list[logging.LogRecord]:
    logger = logging.getLogger("test_group_output_warnings")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        validate_group_output_memberships(db, logger)
    return [r for r in caplog.records if r.name == logger.name]


def test_output_nodeGroup_dispatch_missing_group_node(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoNodes", "output_nodeGroup_dispatch")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoNodes" in m and "output_nodeGroup_dispatch" in m
               and "group__node" in m for m in msgs), msgs


def test_output_nodeGroup_indicators_missing_group_node(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoNodes", "output_nodeGroup_indicators")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoNodes" in m and "output_nodeGroup_indicators" in m
               and "group__node" in m for m in msgs), msgs


def test_output_flowGroup_indicators_missing_flow_members(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoFlows", "output_flowGroup_indicators")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoFlows" in m and "output_flowGroup_indicators" in m
               and "group__unit__node" in m for m in msgs), msgs


def test_flow_aggregator_missing_flow_members(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoFlows", "flow_aggregator")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoFlows" in m and "flow_aggregator" in m
               and "group__unit__node" in m for m in msgs), msgs


def test_no_warnings_when_memberships_match(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Setting the flag ``yes`` with the right membership must be silent."""
    url = _build_db(
        tmp_path,
        parameter_values=[
            ("gNode", "output_nodeGroup_dispatch"),
            ("gNode", "output_nodeGroup_indicators"),
            ("gFlowUnit", "output_flowGroup_indicators"),
            ("gFlowConn", "flow_aggregator"),
        ],
        memberships={
            "group__node": [("gNode", "nA")],
            "group__unit__node": [("gFlowUnit", "u1", "nA")],
            "group__connection__node": [("gFlowConn", "c1", "nA")],
        },
    )
    records = _run_validation(url, caplog)
    assert records == [], [r.getMessage() for r in records]


def test_flow_flag_with_only_node_members_still_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A flow-output flag on a pure node group must still warn: the
    presence of ``group__node`` rows does not satisfy the flow output
    requirement."""
    url = _build_db(
        tmp_path,
        parameter_values=[("gNode", "output_flowGroup_indicators")],
        memberships={
            "group__node": [("gNode", "nA")],
        },
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNode" in m and "output_flowGroup_indicators" in m
               for m in msgs), msgs
