"""The parameter catalogue: the web-only docs page and the Excel data dictionary.

Every parameter of the shipped schema appears once; a record agrees with the assistant's ``describe_parameter``;
table cells are escaped; ``--xlsx`` writes the columns and marks the parameters a database uses (a database built
from ``templates_examples.json``) without needing the documentation folder.  Fixture facts (from
``templates_examples.json``): ``unit.efficiency`` has values for 9 units, of which ``coal_plant`` only in
alternative ``coal``; scenario ``coal`` = init, west, coal; no ``solve.benders_tolerance`` value.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.assistant import docs_store
from flextool.docs_build import render_pages, scope
from flextool.docs_build.__main__ import main as docs_build_main
from flextool.docs_build.parameters import escape_cell, parameter_record

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
SCHEMA = REPO / "flextool" / "schemas" / "spinedb_schema.json"


def _schema_parameters() -> list[tuple[str, str]]:
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return [(row[0], row[1]) for row in data["parameter_definitions"]]


def test_the_page_has_every_parameter_once_and_is_web_only() -> None:
    page = render_pages(DOCS)[scope.PARAMETER_CATALOGUE_PAGE]
    section = None
    seen: list[tuple[str, str]] = []
    for line in page.splitlines():
        if line.startswith("## "):
            section = line[3:]
        elif line.startswith("| `"):
            seen.append((section, line.split("`")[1]))
    assert sorted(seen) == sorted(_schema_parameters())
    assert len(seen) == len(set(seen))
    # The Markdown cell of solve.benders_tolerance: its description holds "|best_UB|".
    assert r"/\|best_UB\| for convergence" in page
    assert scope.PARAMETER_CATALOGUE_PAGE not in scope.GENERATED_PAGES     # not in the assistant's index


def test_cells_are_escaped() -> None:
    assert escape_cell("a | b\nc\r\n d") == r"a \| b c d"


@pytest.mark.parametrize(("entity_class", "parameter"), [
    ("unit", "efficiency"),        # a PARAM_ALLOWED_SHAPES entry
    ("commodity", "price_method"),  # a value list and a default
    ("node", "inflow"),            # no allow-list entry: the map ranks
    ("unit", "existing"),          # [MW or count]
])
def test_parameter_record_matches_describe_parameter(entity_class, parameter, tmp_path, monkeypatch) -> None:
    from flextool.assistant.caps import DESCRIPTION_MAX, cap_text
    from flextool.assistant.context import make_context
    from flextool.assistant.tools import inputs
    from flextool.assistant.unit_wording import shown_unit

    monkeypatch.setattr(inputs, "_reference_sections", lambda parameter: [])
    payload = inputs.describe_parameter(make_context(tmp_path), {"entity_class": entity_class,
                                                                 "parameter": parameter})
    record = parameter_record(entity_class, parameter)
    assert cap_text(record.description, DESCRIPTION_MAX) == payload["description"]
    assert record.default == payload["default"]
    assert (list(record.value_list) if record.value_list is not None else None) == payload["value_list"]
    assert record.declared_shapes == payload["allowed_shapes"]
    assert shown_unit(record.unit) == payload["unit"]
    assert record.unit_reason == payload["unit_reason"]
    assert record.group == payload["group"]


@pytest.fixture(scope="module")
def examples_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    from flextool._resources import package_data_path
    from flextool.update_flextool.initialize_database import initialize_database

    path = tmp_path_factory.mktemp("catalogue_db") / "examples.sqlite"
    initialize_database(str(package_data_path("schemas/canonical_databases/templates_examples.json")), str(path))
    return f"sqlite:///{path}"


def _sheet(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, object]], object]:
    from openpyxl import load_workbook

    ws = load_workbook(path)["parameters"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    return head, {(r[0], r[1]): dict(zip(head, r)) for r in rows[1:]}, ws


def test_xlsx_catalogue_with_input_db(examples_db_url, tmp_path, monkeypatch) -> None:
    # No documentation in this installation: --xlsx does not need it.
    monkeypatch.setattr(docs_store, "markdown_root", lambda package_dir=docs_store.PACKAGE_DIR: None)
    out = tmp_path / "catalogue.xlsx"
    assert docs_build_main(["--xlsx", str(out), "--input-db", examples_db_url]) == 0
    head, rows, ws = _sheet(out)
    assert head == ["class", "parameter", "definition", "unit", "allowed shapes", "default", "value list",
                    "parameter group", "used in DB", "entities", "alternatives", "Source", "How to replace",
                    "Notes"]
    assert sorted(rows) == sorted(_schema_parameters())
    efficiency = rows[("unit", "efficiency")]
    assert (efficiency["used in DB"], efficiency["entities"]) == ("yes", 9)
    assert efficiency["unit"] == "dimensionless" and efficiency["default"] == "1"
    assert not rows[("solve", "benders_tolerance")]["used in DB"]
    assert all(not r[c] for r in rows.values() for c in ("Source", "How to replace", "Notes"))
    assert ws.freeze_panes == "C2" and ws.auto_filter.ref == f"A1:N{len(rows) + 1}"
    assert ws["A1"].font.bold and ws["C2"].alignment.wrap_text

    scenario_out = tmp_path / "coal.xlsx"
    assert docs_build_main(["--xlsx", str(scenario_out), "--input-db", examples_db_url, "--scenario", "coal"]) == 0
    efficiency = _sheet(scenario_out)[1][("unit", "efficiency")]
    assert (efficiency["used in DB"], efficiency["entities"], efficiency["alternatives"]) == ("yes", 1, "coal")

    plain = tmp_path / "plain.xlsx"
    assert docs_build_main(["--xlsx", str(plain)]) == 0
    head, rows, _ = _sheet(plain)
    assert "used in DB" not in head and head[-3:] == ["Source", "How to replace", "Notes"]
