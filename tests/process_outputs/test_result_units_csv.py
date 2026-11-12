"""The CSV result folder's sidecars (#306): ``units.csv`` covers every value column of every written file and
``datapackage.json`` describes each file by the header it was written with."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
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


def _header_rows(path: Path, dialect: dict) -> list[tuple[str, ...]]:
    """The labels of every column in the file's header rows (``dialect.headerRows``, default ``[1]``); the
    ``commentRows`` among the first lines must be blank."""
    header_rows = dialect.get("headerRows", [1])
    comment_rows = dialect.get("commentRows", [])
    n = max([*header_rows, *comment_rows])
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [next(csv.reader([line])) for line, _ in zip(fh, range(n))]
    for r in comment_rows:
        assert not any(rows[r - 1]), (path.name, r)
    assert sorted([*header_rows, *comment_rows]) == list(range(1, n + 1)), path.name
    return list(zip(*(rows[r - 1] for r in header_rows)))


def _frictionless_name(labels: tuple[str, ...]) -> str:
    """A column's name as Frictionless reads a multi-row header: the labels joined with '|', a label equal to
    the one above it left out (a merged cell)."""
    parts: list[str] = []
    for label in labels:
        if not parts or label != parts[-1]:
            parts.append(label)
    return "|".join(parts)


def _check_resource(csv_dir: Path, res: dict) -> list[tuple[str, ...]]:
    """The resource's field names are the file's header as Frictionless joins it."""
    header = _header_rows(csv_dir / res["path"], res.get("dialect", {}))
    assert [f["name"] for f in res["schema"]["fields"]] == [_frictionless_name(h) for h in header], res["path"]
    return header


def _write_like_the_writer(df: pd.DataFrame, path: Path) -> list:
    """Write *df* as the CSV writer does and return the written header."""
    out = df.reset_index()
    out.columns.names = [None] * out.columns.nlevels
    out.to_csv(path, index=False, float_format="%.8g")
    return list(out.columns)


@pytest.mark.parametrize("n_levels", [2, 3])
@pytest.mark.parametrize("with_values", [False, True])
def test_wide_csv_resource_validates(tmp_path: Path, with_values: bool, n_levels: int):
    """A wide table validates with and without value columns: with none, the CSV's lower header rows are
    blank and are described as comment rows; with values, an index column's blank cells below its name
    read as one merged cell (``period|``)."""
    from flextool.process_outputs._output_meta import datapackage_resource
    index = pd.MultiIndex.from_tuples([("p1", "t1"), ("p1", "t2")], names=["period", "time"])
    label = ("u1", "n1", "c1")[:n_levels]
    cols = pd.MultiIndex.from_tuples([label] if with_values else [], names=["unit", "node", "x"][:n_levels])
    df = pd.DataFrame([[1.0], [2.5]] if with_values else [[], []], index=index, columns=cols)
    header = _write_like_the_writer(df, tmp_path / "t.csv")
    res = datapackage_resource("unit_outputNode_dt_ee", "t.csv", list(index.names), list(df.columns),
                               header=header)
    dialect = res.get("dialect", {})
    assert dialect.get("commentRows", []) == ([] if with_values else list(range(2, n_levels + 1)))
    assert dialect.get("headerRows", [1]) == (list(range(1, n_levels + 1)) if with_values else [1])
    names = [f["name"] for f in res["schema"]["fields"]]
    assert names[:2] == (["period|", "time|"] if with_values else ["period", "time"])
    _check_resource(tmp_path, res)
    _validate(tmp_path, [res])


def _validate(csv_dir: Path, resources: list[dict]) -> None:
    """Frictionless validation of the whole package when Frictionless is installed (the in-test checks of
    :func:`_check_resource` run either way)."""
    try:
        import frictionless
    except ImportError:
        return
    write_datapackage(str(csv_dir), resources)
    report = frictionless.Package(str(csv_dir / "datapackage.json")).validate()
    assert report.valid, report.flatten(["type", "note"])[:5]


@pytest.mark.solver
def test_scenario_csv_sidecars(test_db_url: str, test_solver_config_dir: Path, tmp_path: Path):
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    steps = run_chain_from_db(input_db_url=test_db_url, scenario_name=SCENARIO, work_folder=tmp_path,
                              solver_config_dir=test_solver_config_dir, warm=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    write_outputs(
        scenario_name=SCENARIO, output_location=str(tmp_path), subdir=SCENARIO,
        output_config_path=str(REPO_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
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

    n_measures = 0
    n_wide = 0
    for res in dp["resources"]:
        header = _check_resource(csv_dir, res)
        n_wide += len(res.get("dialect", {}).get("headerRows", [1])) > 1
        for pos, field in enumerate(res["schema"]["fields"]):
            if field["flextool:semantics"] == "dimension":
                assert (res["path"], pos) not in by_pos
                continue
            n_measures += 1
            row = by_pos[(res["path"], pos)]
            assert row["unit"] == (field.get("unit") or "per unit"), (res["path"], pos)
            assert tuple(row[f"header_{i + 1}"] for i in range(len(header[pos]))) == header[pos]
    assert n_measures == len(by_pos)
    assert n_wide > 0

    # A Frictionless reader finds every file valid against its resource (the whole package).
    frictionless = pytest.importorskip("frictionless")
    report = frictionless.Package(str(csv_dir / "datapackage.json")).validate()
    assert report.valid, report.flatten(["type", "note"])[:5]
