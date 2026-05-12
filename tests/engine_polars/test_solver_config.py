"""Tests for v52 multi-solver dispatch in SolveConfig + _solver_dispatch.

Phase 2 of the polar-high multi-solver handoff (see
``specs/flextool-multi-solver-handoff.md``) wires the seven new
``solver_*`` parameters introduced by the v52 schema migration into
:class:`flextool.engine_polars._solve_config.SolverConfig` and provides
the per-solver option-translation helper
:func:`flextool.engine_polars._solver_dispatch.build_solver_options`.

These tests cover:

* Defaults — no ``solver_*`` parameter authored → empty
  ``solver_configs`` dict, so callers fall back to
  ``SolverConfig()`` field defaults.
* Explicit-HiGHS round-trip parity.
* ``solver_options`` Map parameter unpacking.
* Convenience knob round-trip (time_limit / mip_gap / threads).
* Per-solver option-name translation (table-driven across all 5
  solvers + raw-takes-precedence collision case).
* Unknown-solver error contains the supported list.

Fixtures
--------
Each test starts from ``tests/fixtures/stochastics.json`` (a v25
snapshot), migrates it through to v52, and uses
:func:`spinedb_api.import_data` to add the per-test
``solver_*`` parameter values onto an existing solve.  Mirrors the
authoring style of :mod:`tests.test_v52_migration`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, import_data
from spinedb_api.filters.scenario_filter import (
    apply_scenario_filter_to_subqueries,
)

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars._solve_config import (  # noqa: E402
    SolveConfig,
    SolverConfig,
)
from flextool.engine_polars._solver_dispatch import (  # noqa: E402
    _PARAM_MAP,
    build_solver_options,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURE = TESTS_DIR / "fixtures" / "stochastics.json"
SCENARIO = "1_week_rolling_wind"
SOLVE_NAME = "1week_rolling"  # the active solve under SCENARIO
# The fixture's "Base" alternative is NOT actually part of the scenario
# alternative chain (the scenarios cover ``init``/``base``/``system`` +
# a per-scenario override).  We append our injected solver_* values
# onto the per-scenario override alternative so the scenario filter
# picks them up.
INJECT_ALTERNATIVE = "1week_rolling"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_migrated_db(tmp_path: Path) -> str:
    db_path = tmp_path / "v52.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    return url


def _inject_param_values(
    url: str,
    values: list[tuple[str, str, str, object]],
) -> None:
    """Append parameter_value rows to the migrated DB.

    *values* is a list of ``(entity_class, entity_name, param_name,
    python_value)`` tuples.  The python_value is passed through
    untouched to :func:`spinedb_api.import_data`, which calls
    :func:`spinedb_api.to_database` internally.  Map / float / int /
    str values are all handled correctly that way.
    """
    rows = [
        (ec, en, pn, py_val, INJECT_ALTERNATIVE)
        for ec, en, pn, py_val in values
    ]
    with DatabaseMapping(url) as db:
        count, errors = import_data(
            db,
            parameter_values=rows,
        )
        if errors:
            raise RuntimeError(f"import errors: {errors[:5]}")
        db.commit_session("inject test solver params")


def _load_solve_config(url: str) -> SolveConfig:
    with DatabaseMapping(url) as db:
        apply_scenario_filter_to_subqueries(db, SCENARIO)
        return SolveConfig.load_from_db(
            db, logging.getLogger("test.solver_config")
        )


# ---------------------------------------------------------------------------
# DB-load tests
# ---------------------------------------------------------------------------


def test_solve_config_solver_defaults(tmp_path: Path):
    """No solver_* params authored → empty solver_configs.

    Callers fall back to ``SolverConfig()`` defaults
    (highs/direct/no overrides), which is exactly what the v52 parameter
    definitions default to anyway.
    """
    url = _make_migrated_db(tmp_path)
    sc = _load_solve_config(url)
    assert sc.solver_configs == {}
    # The dataclass defaults must match the v52 schema defaults so that
    # absent-from-DB == default-from-DB semantically.
    default = SolverConfig()
    assert default.name == "highs"
    assert default.io_api == "direct"
    assert default.options == {}
    assert default.time_limit is None
    assert default.mip_gap is None
    assert default.threads is None
    assert default.log_level == "normal"


def test_solve_config_solver_explicit_highs(tmp_path: Path):
    """solver="highs" set explicitly → SolverConfig.name=='highs'."""
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [("solve", SOLVE_NAME, "solver", "highs")],
    )
    sc = _load_solve_config(url)
    assert SOLVE_NAME in sc.solver_configs
    cfg = sc.solver_configs[SOLVE_NAME]
    assert cfg.name == "highs"
    # Other fields still default — only ``solver`` was authored.
    assert cfg.io_api == "direct"
    assert cfg.options == {}
    assert cfg.time_limit is None
    assert cfg.log_level == "normal"


def test_solve_config_solver_options_map(tmp_path: Path):
    """solver_options Map round-trips into SolverConfig.options dict."""
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            (
                "solve",
                SOLVE_NAME,
                "solver_options",
                Map(["presolve", "log_to_console"], ["off", "no"]),
            ),
        ],
    )
    sc = _load_solve_config(url)
    cfg = sc.solver_configs[SOLVE_NAME]
    assert cfg.options == {"presolve": "off", "log_to_console": "no"}


def test_solve_config_solver_convenience_knobs(tmp_path: Path):
    """Time-limit / mip-gap / threads round-trip onto SolverConfig."""
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            ("solve", SOLVE_NAME, "solver_time_limit", 60.0),
            ("solve", SOLVE_NAME, "solver_mip_gap", 0.01),
            ("solve", SOLVE_NAME, "solver_threads", 4),
        ],
    )
    sc = _load_solve_config(url)
    cfg = sc.solver_configs[SOLVE_NAME]
    assert cfg.time_limit == 60.0
    assert cfg.mip_gap == pytest.approx(0.01)
    assert cfg.threads == 4
    # Sanity: name still default — convenience knobs don't override solver
    assert cfg.name == "highs"


def test_solve_config_load_idempotent(tmp_path: Path):
    """Two consecutive load_from_db calls produce the same SolverConfigs."""
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            ("solve", SOLVE_NAME, "solver", "gurobi"),
            ("solve", SOLVE_NAME, "solver_time_limit", 120.0),
            (
                "solve",
                SOLVE_NAME,
                "solver_options",
                Map(["MIPFocus"], ["1"]),
            ),
        ],
    )
    sc1 = _load_solve_config(url)
    sc2 = _load_solve_config(url)
    assert sc1.solver_configs == sc2.solver_configs


# ---------------------------------------------------------------------------
# build_solver_options translation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_name, expected_keys",
    [
        ("highs",  {"time_limit": "time_limit",   "mip_gap": "mip_rel_gap",            "threads": "threads"}),
        ("gurobi", {"time_limit": "TimeLimit",    "mip_gap": "MIPGap",                  "threads": "Threads"}),
        ("cplex",  {"time_limit": "timelimit",    "mip_gap": "mip.tolerances.mipgap",   "threads": "threads"}),
        ("xpress", {"time_limit": "maxtime",      "mip_gap": "miprelstop",              "threads": "threads"}),
        ("copt",   {"time_limit": "TimeLimit",    "mip_gap": "RelGap",                  "threads": "Threads"}),
    ],
)
def test_build_solver_options_translation_per_solver(
    solver_name: str, expected_keys: dict[str, str]
):
    """Convenience knobs land under each solver's native parameter name."""
    cfg = SolverConfig(
        name=solver_name,
        time_limit=42.0,
        mip_gap=0.005,
        threads=8,
    )
    opts = build_solver_options(cfg)
    assert opts == {
        expected_keys["time_limit"]: 42.0,
        expected_keys["mip_gap"]: 0.005,
        expected_keys["threads"]: 8,
    }


def test_build_solver_options_raw_overrides_translation():
    """Raw solver_options entries override the translated convenience knobs."""
    cfg = SolverConfig(
        name="gurobi",
        time_limit=60.0,
        # User hand-wrote a Gurobi-native option whose key collides with
        # the translation of solver_time_limit -> "TimeLimit".  Raw wins.
        options={"TimeLimit": 5.0, "my_raw_key": "raw_value"},
    )
    opts = build_solver_options(cfg)
    assert opts["TimeLimit"] == 5.0  # raw wins, not the translated 60.0
    assert opts["my_raw_key"] == "raw_value"


def test_build_solver_options_empty_when_nothing_set():
    """No convenience knobs and no raw options → empty dict."""
    cfg = SolverConfig(name="highs")
    assert build_solver_options(cfg) == {}


def test_build_solver_options_raw_only_unknown_solver_ok():
    """An unknown solver with raw-options-only does NOT raise.

    Users may want to plumb a future solver via ``solver_options``
    before _PARAM_MAP is updated; we keep that escape hatch open.
    The convenience knobs are the only thing that needs translation.
    """
    cfg = SolverConfig(name="mosek", options={"MSK_DPAR_MIO_MAX_TIME": 30})
    opts = build_solver_options(cfg)
    assert opts == {"MSK_DPAR_MIO_MAX_TIME": 30}


def test_build_solver_options_unknown_solver_raises():
    """Unknown solver + any convenience knob → ValueError listing solvers."""
    cfg = SolverConfig(name="bogus", time_limit=10.0)
    with pytest.raises(ValueError) as ei:
        build_solver_options(cfg)
    msg = str(ei.value)
    assert "bogus" in msg
    # All five supported solvers must appear in the error so users see
    # the menu inline.
    for solver in _PARAM_MAP:
        assert solver in msg
