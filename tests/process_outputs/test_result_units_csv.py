"""The CSV result folder's sidecars (#306): ``units.csv`` covers every value column of every written file and
``datapackage.json`` describes each file by the header it was written with."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from flextool.process_outputs._csv_sidecars import (
    reserved_csv_names,
    summary_solve_row,
    write_datapackage,
    write_units_csv,
)

TEST_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TEST_DIR.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

SCENARIO = "unidirectional_connection"


def _units(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_single_result_write_keeps_the_other_files_entries(tmp_path: Path):
    a = {"file": "a.csv", "position": 2, "header": ("x",), "unit": "MW", "semantics": "instantaneous"}
    b = {"file": "b.csv", "position": 1, "header": ("y", "east"), "unit": "MWh/a", "semantics": "annualized"}
    write_units_csv(str(tmp_path), [a, b, summary_solve_row()])
    write_datapackage(str(tmp_path), [{"path": "a.csv", "name": "a"}, {"path": "b.csv", "name": "b"}])
    write_units_csv(str(tmp_path), [{**a, "unit": "GW"}], merge=True)
    write_datapackage(str(tmp_path), [{"path": "a.csv", "name": "a2"}], merge=True)
    rows = _units(tmp_path / "units.csv")
    assert [(r["file"], r["unit"]) for r in rows] == [("a.csv", "GW"), ("b.csv", "MWh/a"),
                                                      ("summary_solve.csv", "M CUR")]
    assert rows[1]["header_2"] == "east" and rows[1]["header_3"] == ""
    dp = json.loads((tmp_path / "datapackage.json").read_text())
    assert [r["name"] for r in dp["resources"]] == ["a2", "b"]
    assert dp["flextool:units"] == "units.csv"


def test_sidecar_names_are_reserved():
    from flextool.process_outputs.write_outputs import _csv_filename
    used = reserved_csv_names(with_summary=False)
    with pytest.raises(ValueError):
        _csv_filename("node_units", {"node_units": ["units", True]}, used)


def _header_rows(path: Path, n: int) -> list[tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [next(csv.reader([line])) for line, _ in zip(fh, range(n))]
    return list(zip(*rows))


@pytest.mark.solver
def test_scenario_csv_sidecars(test_db_url: str, test_solver_config_dir: Path, tmp_path: Path):
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    steps = run_chain_from_db(input_db_url=test_db_url, scenario_name=SCENARIO, work_folder=tmp_path,
                              solver_config_dir=test_solver_config_dir, warm=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    write_outputs(
        scenario_name=SCENARIO, output_location=str(tmp_path), subdir=SCENARIO,
        output_config_path=str(REPO_ROOT / "templates" / "default_plots.yaml"),
        fallback_output_location=str(tmp_path), raw_output_dir=str(tmp_path / "output_raw"),
        write_methods=["csv"], solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=last.flex_data_provider,
    )
    csv_dir = tmp_path / "output_csv" / SCENARIO
    dp = json.loads((csv_dir / "datapackage.json").read_text())
    units = _units(csv_dir / "units.csv")
    by_pos = {(r["file"], int(r["position"])): r for r in units if r["position"]}
    written = {p.name for p in csv_dir.glob("*.csv")} - {"units.csv", "summary_solve.csv"}
    assert {r["path"] for r in dp["resources"]} == written
    assert any(r["file"] == "summary_solve.csv" and r["unit"] == "M CUR" for r in units)

    wide = None
    n_measures = 0
    for res in dp["resources"]:
        fields = res["schema"]["fields"]
        n_rows = len(res.get("dialect", {}).get("headerRows", [1]))
        header = _header_rows(csv_dir / res["path"], n_rows)
        # The field names are the file's header, the labels of a column joined with '|'.
        assert [f["name"] for f in fields] == ["|".join(h) for h in header], res["path"]
        for pos, field in enumerate(fields):
            if field["flextool:semantics"] == "dimension":
                assert (res["path"], pos) not in by_pos
                continue
            n_measures += 1
            row = by_pos[(res["path"], pos)]
            assert row["unit"] == (field.get("unit") or "per unit"), (res["path"], pos)
            assert tuple(row[f"header_{i + 1}"] for i in range(n_rows)) == header[pos]
        if n_rows > 1 and wide is None:
            wide = res
    assert n_measures == len(by_pos)
    assert wide is not None

    # A Frictionless reader joins the header rows into the same field names and finds the file valid.
    frictionless = pytest.importorskip("frictionless")
    report = frictionless.Resource(dict(wide), basepath=str(csv_dir)).validate()
    assert report.valid, report.flatten(["type", "note"])[:5]
