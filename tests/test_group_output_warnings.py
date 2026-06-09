"""Tests for the validation warnings emitted when a group-level output
flag is set but the group has no members of the required entity class.

The warnings are emitted by
``flextool.input_derivation._validators.validate_group_output_memberships``
and cover three silent-no-op cases (v58 vocabulary):

* ``group.print_dispatch: yes`` with no ``group__node`` row
* ``group.print_indicators: yes`` with no ``group__node`` row
* ``flowGroup.flow_aggregator`` set to a non-``none`` method with no
  ``flowGroup__unit__node`` or ``flowGroup__connection__node`` row
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping

from flextool.input_derivation._validators import validate_group_output_memberships

# Parameter names whose class is ``flowGroup`` (the v58 carve-out); every
# other parameter checked by the validator lives on ``group``.
_FLOWGROUP_PARAMS = {"flow_aggregator"}


def _build_db(tmp_path: Path, parameter_values: list[tuple[str, str, str]],
              memberships: dict[str, list[tuple[str, ...]]]) -> str:
    """Create a fresh SQLite DB with the group/flowGroup output parameter
    definitions plus the supplied parameter values and memberships.

    ``parameter_values`` is a list of ``(entity_name, parameter_name,
    value)`` triples.  The owning class is inferred from
    :data:`_FLOWGROUP_PARAMS` (``flowGroup`` for ``flow_aggregator``,
    ``group`` otherwise).

    ``memberships`` maps entity-class name (``group__node``,
    ``flowGroup__unit__node``, ``flowGroup__connection__node``) to a list
    of byname tuples.
    """
    db_path = tmp_path / "warnings_test.sqlite"
    url = f"sqlite:///{db_path}"

    # Unique set of group / flowGroup names across params + memberships.
    group_names: set[str] = set()
    flowgroup_names: set[str] = set()
    for ent, pname, _ in parameter_values:
        if pname in _FLOWGROUP_PARAMS:
            flowgroup_names.add(ent)
        else:
            group_names.add(ent)
    for cls_name, rows in memberships.items():
        for row in rows:
            if cls_name.startswith("flowGroup"):
                flowgroup_names.add(row[0])
            else:
                group_names.add(row[0])

    with DatabaseMapping(url, create=True) as db:
        # Minimal entity classes.
        db.add_update_item("entity_class", name="group")
        db.add_update_item("entity_class", name="flowGroup")
        db.add_update_item("entity_class", name="node")
        db.add_update_item("entity_class", name="unit")
        db.add_update_item("entity_class", name="connection")
        db.add_update_item("entity_class", name="group__node",
                           dimension_name_list=("group", "node"))
        db.add_update_item("entity_class", name="flowGroup__unit__node",
                           dimension_name_list=("flowGroup", "unit", "node"))
        db.add_update_item("entity_class", name="flowGroup__connection__node",
                           dimension_name_list=("flowGroup", "connection", "node"))

        # Value list for the yes/no flags.
        db.add_update_item("parameter_value_list", name="yes_no")
        db.add_update_item("list_value", parameter_value_list_name="yes_no",
                           value=b'"yes"', type="str", index=0)
        db.add_update_item("list_value", parameter_value_list_name="yes_no",
                           value=b'"no"', type="str", index=1)

        # Value list for the flow_aggregator method enum.
        db.add_update_item("parameter_value_list", name="flow_aggregator_methods")
        for idx, v in enumerate(("none", "dispatch_plots_only",
                                 "standalone_aggregator_only", "both")):
            db.add_update_item("list_value",
                               parameter_value_list_name="flow_aggregator_methods",
                               value=f'"{v}"'.encode(), type="str", index=idx)

        # Parameter definitions on ``group``.
        for pname in ("print_dispatch", "print_indicators"):
            db.add_update_item("parameter_definition",
                               entity_class_name="group",
                               name=pname,
                               parameter_value_list_name="yes_no")
        # flow_aggregator lives on ``flowGroup``.
        db.add_update_item("parameter_definition",
                           entity_class_name="flowGroup",
                           name="flow_aggregator",
                           parameter_value_list_name="flow_aggregator_methods")

        # Entities — ensure every referenced entity exists before adding
        # relationships.
        for g in group_names:
            db.add_update_item("entity", entity_class_name="group",
                               name=g, entity_byname=(g,))
        for fg in flowgroup_names:
            db.add_update_item("entity", entity_class_name="flowGroup",
                               name=fg, entity_byname=(fg,))
        for cls_name, rows in memberships.items():
            dims = {"group__node": ("group", "node"),
                    "flowGroup__unit__node": ("flowGroup", "unit", "node"),
                    "flowGroup__connection__node":
                        ("flowGroup", "connection", "node")}[cls_name]
            for row in rows:
                for dim_cls, dim_name in zip(dims, row):
                    if dim_cls in ("group", "flowGroup"):
                        continue  # already added
                    db.add_update_item("entity", entity_class_name=dim_cls,
                                       name=dim_name, entity_byname=(dim_name,))
                db.add_update_item("entity",
                                   entity_class_name=cls_name,
                                   entity_byname=row)

        # Parameter values.
        for ent, pname, value in parameter_values:
            cls = "flowGroup" if pname in _FLOWGROUP_PARAMS else "group"
            db.add_update_item("parameter_value",
                               entity_class_name=cls,
                               entity_byname=(ent,),
                               parameter_definition_name=pname,
                               alternative_name="Base",
                               value=f'"{value}"'.encode(), type="str")

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


def test_print_dispatch_missing_group_node(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoNodes", "print_dispatch", "yes")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoNodes" in m and "print_dispatch" in m
               and "group__node" in m for m in msgs), msgs


def test_print_indicators_missing_group_node(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoNodes", "print_indicators", "yes")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoNodes" in m and "print_indicators" in m
               and "group__node" in m for m in msgs), msgs


def test_flow_aggregator_missing_flow_members(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoFlows", "flow_aggregator", "both")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("gNoFlows" in m and "flow_aggregator" in m
               and "flowGroup__unit__node" in m for m in msgs), msgs


def test_flow_aggregator_none_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``flow_aggregator: none`` requests no output — never warns even
    without flow members."""
    url = _build_db(
        tmp_path,
        parameter_values=[("gNoFlows", "flow_aggregator", "none")],
        memberships={},
    )
    records = _run_validation(url, caplog)
    assert records == [], [r.getMessage() for r in records]


def test_no_warnings_when_memberships_match(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Setting the flag with the right membership must be silent."""
    url = _build_db(
        tmp_path,
        parameter_values=[
            ("gNode", "print_dispatch", "yes"),
            ("gNode", "print_indicators", "yes"),
            ("fgUnit", "flow_aggregator", "dispatch_plots_only"),
            ("fgConn", "flow_aggregator", "both"),
        ],
        memberships={
            "group__node": [("gNode", "nA")],
            "flowGroup__unit__node": [("fgUnit", "u1", "nA")],
            "flowGroup__connection__node": [("fgConn", "c1", "nA")],
        },
    )
    records = _run_validation(url, caplog)
    assert records == [], [r.getMessage() for r in records]
