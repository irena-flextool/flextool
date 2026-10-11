"""Solves write no HiGHS solution file unless one is explicitly requested.

Older ``highs.opt.template`` versions carried the GLPK-era lines
``write_solution_to_file=true`` / ``solution_file=flextool.sol`` /
``write_solution_style=2`` (glpsol format), so every solve wrote an unused
``flextool.sol`` and HiGHS warned "The cost row for the Glpsol solution file
is reported last".  Projects created earlier still have that ``highs.opt``;
the engine drops those keys from the ``highs.opt`` layer.  A solution file
requested through the solve's ``solver_arguments`` is still honoured, and
then HiGHS' writer mangles spaces in the live LP's names — so the
``_restore_space_mangled_names`` repair must keep entity names with spaces
intact in the outputs.

Fixture: ``tests.json`` (built from JSON, never a checked-in DB) with the
unit ``coal_chp`` renamed to ``coal chp`` (a space in an entity name).
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import _solver_dispatch
from flextool.engine_polars._solver_dispatch import (
    _resolve_effective_highs_options,
)

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"
TEMPLATE = FLEXTOOL_ROOT / "flextool" / "solver_config" / "highs.opt.template"
OLD_NAME, NEW_NAME = "coal_chp", "coal chp"
SCENARIO = "space_name_dispatch"
CHAIN = ["init", "west", "coal_chp", "heat"]
SOLVE = "y2020_2day_dispatch"
LEGACY_LINES = (
    "write_solution_to_file=true\n"
    "solution_file=flextool.sol\n"
    "write_solution_style=2\n"
)


def _b64(x, kind: str) -> list:
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


def _rename(item, old: str, new: str):
    if item == old:
        return new
    if isinstance(item, list):
        return [new if e == old else e for e in item]
    return item


def _build_db(tmp_path_factory, name: str, extra_values: list) -> str:
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    # Rename the unit entity (not the alternative of the same name).
    for key in ("entities", "entity_alternatives", "parameter_values"):
        for row in data[key]:
            row[1] = _rename(row[1], OLD_NAME, NEW_NAME)
    data["parameter_values"].extend(extra_values)
    data["scenarios"].append([SCENARIO, False, ""])
    for a, before in zip(CHAIN, CHAIN[1:] + [None]):
        data["scenario_alternatives"].append([SCENARIO, a, before])
    work = tmp_path_factory.mktemp(name)
    json_path = work / f"{name}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{name}.sqlite")
    migrate_database(url)
    return url


def _legacy_solver_config(tmp_path: Path) -> Path:
    """A project ``solver_config`` holding a pre-cleanup ``highs.opt``."""
    cfg = tmp_path / "project_solver_config"
    cfg.mkdir()
    (cfg / "highs.opt").write_text(LEGACY_LINES + TEMPLATE.read_text())
    return cfg


def _flow_columns(work_folder: Path) -> str:
    """Column labels of the solver-derived ``v_flow`` output (wide layout:
    one ``('process', 'source', 'sink')`` column per flow)."""
    files = sorted((work_folder / "output_raw").glob("v_flow__*.parquet"))
    assert files, f"no v_flow output under {work_folder}"
    return " ".join(c for f in files for c in pl.read_parquet(f).columns)


# ---------------------------------------------------------------------------
# Resolver: the highs.opt layer drops the solution-file keys
# ---------------------------------------------------------------------------


def test_legacy_highs_opt_lines_dropped_silently(tmp_path, caplog) -> None:
    opt = tmp_path / "highs.opt"
    opt.write_text(LEGACY_LINES + "mip_rel_gap=0.002\n")
    with caplog.at_level(logging.WARNING):
        opts = _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=opt, cli_overrides=None,
        )
    assert opts == {"mip_rel_gap": "0.002"}
    assert not caplog.records


def test_customised_solution_file_lines_dropped_with_one_warning(
        tmp_path, caplog, monkeypatch) -> None:
    monkeypatch.setattr(_solver_dispatch, "_warned_ignored_highs_opt_keys", False)
    opt = tmp_path / "highs.opt"
    opt.write_text("write_solution_to_file = true\nwrite_solution_style=0\n")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            opts = _resolve_effective_highs_options(
                solver_arguments_map=None, highs_opt_path=opt,
                cli_overrides=None,
            )
            assert opts == {}
    warnings = [r for r in caplog.records if "solution-file" in r.getMessage()]
    assert len(warnings) == 1
    assert "write_solution_style=0" in warnings[0].getMessage()


def test_solution_file_via_solver_arguments_is_kept(tmp_path) -> None:
    opt = tmp_path / "highs.opt"
    opt.write_text(LEGACY_LINES)
    opts = _resolve_effective_highs_options(
        solver_arguments_map={"write_solution_to_file": "true",
                              "solution_file": "mine.sol"},
        highs_opt_path=opt, cli_overrides=None,
    )
    assert opts == {"write_solution_to_file": "true", "solution_file": "mine.sol"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_legacy_highs_opt_solve_writes_no_solution_file(
        tmp_path, tmp_path_factory, monkeypatch, capfd) -> None:
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(tmp_path_factory, "legacy_opt", [])
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    steps = run_chain_from_db(
        url, SCENARIO, work_folder=work,
        solver_config_dir=_legacy_solver_config(tmp_path),
    )
    assert steps
    out = capfd.readouterr()
    assert not list(tmp_path.rglob("*.sol")), list(tmp_path.rglob("*.sol"))
    assert "Glpsol" not in out.out + out.err
    assert f"'{NEW_NAME}'" in _flow_columns(work)


@pytest.mark.solver
def test_requested_solution_file_keeps_space_names(
        tmp_path, tmp_path_factory, monkeypatch) -> None:
    """A deliberately requested solution file is written, and HiGHS'
    space→underscore mangling of the live LP names is repaired, so the
    outputs still carry ``coal chp``."""
    from flextool.engine_polars import run_chain_from_db

    sol_file = tmp_path / "requested.sol"
    args_map = {
        "type": "map", "index_type": "str",
        "data": [["write_solution_to_file", "true"],
                 ["solution_file", str(sol_file)],
                 ["write_solution_style", "0"]],
    }
    url = _build_db(tmp_path_factory, "requested_sol", [
        ["solve", SOLVE, "solver_arguments", _b64(args_map, "map"), "init"],
    ])
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    run_chain_from_db(
        url, SCENARIO, work_folder=work,
        solver_config_dir=_legacy_solver_config(tmp_path),
    )
    assert sol_file.is_file()
    # The file itself carries HiGHS' mangled names ...
    assert "coal_chp" in sol_file.read_text()
    # ... but the outputs keep the real entity name.
    columns = _flow_columns(work)
    assert f"'{NEW_NAME}'" in columns
    assert "coal_chp" not in columns
