"""Tests for v52 multi-solver dispatch in SolveConfig + _solver_dispatch.

Phase 2 of the polar-high multi-solver handoff (see
``specs/flextool-multi-solver-handoff.md``) wires the seven new
``solver_*`` parameters introduced by the v52 schema migration into
:class:`flextool.engine.input._solve_config.SolverConfig` and provides
the solve's convenience knobs into the commercial option stack,
:func:`flextool.engine.solve._solver_dispatch.resolve_commercial_options`.

These tests cover:

* Defaults — no ``solver_*`` parameter authored → empty
  ``solver_configs`` dict, so callers fall back to
  ``SolverConfig()`` field defaults.
* Explicit-HiGHS round-trip parity.
* ``solver_options`` Map parameter unpacking.
* Convenience knob round-trip (time_limit / mip_gap / threads).
* Per-solver option-name translation (table-driven across the 4
  commercial solvers + solver_arguments-take-precedence collision case).

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

from flextool.engine.input._solve_config import (  # noqa: E402
    SolveConfig,
    SolverConfig,
)
from flextool.engine.solve._solver_dispatch import (  # noqa: E402
    resolve_commercial_options,
)
from flextool.run_options import RunOptions  # noqa: E402
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
    assert cfg.options == {}
    assert cfg.time_limit is None
    assert cfg.log_level == "normal"


def test_solve_config_solver_options_map(tmp_path: Path):
    """solver_arguments 1d-map round-trips into SolverConfig.options dict.

    v56 Batch C.2 folded the legacy ``solver_options`` Map into
    ``solver_arguments`` (retyped to a 1d-map by Batch C.1).  The
    engine-side resolver now sources ``SolverConfig.options`` from
    ``solver_arguments`` (see ``_solve_config.load_from_db``).
    """
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            (
                "solve",
                SOLVE_NAME,
                "solver_arguments",
                Map(["presolve", "log_to_console"], ["off", "no"]),
            ),
        ],
    )
    sc = _load_solve_config(url)
    cfg = sc.solver_configs[SOLVE_NAME]
    assert cfg.options == {"presolve": "off", "log_to_console": "no"}


def test_solve_config_solver_convenience_knobs(tmp_path: Path):
    """``solver_mip_gap`` round-trips onto SolverConfig.mip_gap.

    v56 Batches C.6 and C.8 removed the ``solver_threads`` and
    ``solver_time_limit`` DB axes (replaced by the ``--highs-threads``
    and ``--solver-time-limit`` CLI flags).  Only ``solver_mip_gap``
    survives at v56 as a per-solve convenience knob.
    """
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            ("solve", SOLVE_NAME, "solver_mip_gap", 0.01),
        ],
    )
    sc = _load_solve_config(url)
    cfg = sc.solver_configs[SOLVE_NAME]
    assert cfg.mip_gap == pytest.approx(0.01)
    # The retired DB axes always resolve to None (CLI-flag-driven now).
    assert cfg.time_limit is None
    assert cfg.threads is None
    # Sanity: name still default — convenience knobs don't override solver
    assert cfg.name == "highs"


def test_solve_config_load_idempotent(tmp_path: Path):
    """Two consecutive load_from_db calls produce the same SolverConfigs.

    v56 Batch C.1/C.2: the legacy ``solver_options`` Map was folded
    into ``solver_arguments`` (now a 1d-map).  Batch C.8 removed
    ``solver_time_limit``.  This test now exercises the surviving
    per-solve solver params: ``solver`` + ``solver_arguments`` +
    ``solver_mip_gap``.
    """
    url = _make_migrated_db(tmp_path)
    _inject_param_values(
        url,
        [
            ("solve", SOLVE_NAME, "solver", "gurobi"),
            ("solve", SOLVE_NAME, "solver_mip_gap", 0.005),
            (
                "solve",
                SOLVE_NAME,
                "solver_arguments",
                Map(["MIPFocus"], ["1"]),
            ),
        ],
    )
    sc1 = _load_solve_config(url)
    sc2 = _load_solve_config(url)
    assert sc1.solver_configs == sc2.solver_configs


# ---------------------------------------------------------------------------
# The solve's convenience knobs in the commercial option stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_name, time_native, time_value, gap_native, threads_native",
    [
        ("gurobi", "TimeLimit", "59.5", "MIPGap", "Threads"),
        ("cplex", "timelimit", "59.5", "mip tolerances mipgap", "threads"),
        # MAXTIME is an integer control: rounded up to whole seconds.
        ("xpress", "MAXTIME", "60", "MIPRELSTOP", "THREADS"),
        ("copt", "TimeLimit", "59.5", "RelGap", "Threads"),
    ],
)
def test_convenience_knobs_reach_each_commercial_solver_under_native_names(
    solver_name: str, time_native: str, time_value: str, gap_native: str, threads_native: str,
):
    """The solve's ``time_limit`` / ``mip_gap`` / ``threads`` (the ``solver_mip_gap`` layer) are passed under
    each solver's native parameter name, in that order, when no other layer sets them."""
    cfg = SolverConfig(name=solver_name, time_limit=59.5, mip_gap=0.01, threads=4)
    resolved = resolve_commercial_options(
        solver_name, opt_file=None, solver_config=cfg, run_options=RunOptions.from_env({}),
    )
    assert [(o.name, o.value) for o in resolved] == [
        (time_native, time_value), (gap_native, "0.01"), (threads_native, "4"),
    ]
    assert {o.layer for o in resolved} == {"solver_mip_gap"}


def test_solver_arguments_beat_the_time_limit_knob_case_insensitively():
    """A ``solver_arguments`` entry for the same Gurobi parameter, in another spelling, wins over the
    ``time_limit`` knob: one ``TimeLimit`` entry is passed, with the solver_arguments value and spelling."""
    cfg = SolverConfig(name="gurobi", time_limit=60.0, options={"timelimit": 999})
    resolved = resolve_commercial_options(
        "gurobi", opt_file=None, solver_config=cfg, run_options=RunOptions.from_env({}),
    )
    assert [(o.name, o.value, o.layer) for o in resolved] == [("timelimit", "999", "solver_arguments")]


def test_a_copy_of_a_solve_keeps_its_solver_configuration(tmp_path: Path):
    """``derive_solve`` (2-D period maps, nested sub-solves) and ``derive_realization_solve`` (the realization
    solve FlexTool adds) give the new solve the parent's solver, ``solver_arguments`` and ``solver_mip_gap``.
    Before item 240's fix the copy had the solver name but no configuration, so it ran on HiGHS;
    ``solver_config_for`` now raises on such a mismatch instead."""
    url = _make_migrated_db(tmp_path)
    _inject_param_values(url, [
        ("solve", SOLVE_NAME, "solver", "gurobi"),
        ("solve", SOLVE_NAME, "solver_mip_gap", 0.2),
        ("solve", SOLVE_NAME, "solver_arguments", Map(["TimeLimit"], ["800"])),
    ])
    sc = _load_solve_config(url)
    expected = SolverConfig(name="gurobi", options={"TimeLimit": "800"}, mip_gap=0.2)
    sc.derive_solve(SOLVE_NAME, f"{SOLVE_NAME}_p1")
    sc.derive_realization_solve(SOLVE_NAME, f"{SOLVE_NAME}__realized_dispatch")
    for copy_name in (f"{SOLVE_NAME}_p1", f"{SOLVE_NAME}__realized_dispatch"):
        assert sc.solver_config_for(copy_name) == expected
        assert sc.solver_settings.arguments[copy_name] == {"TimeLimit": "800"}
    del sc.solver_configs[f"{SOLVE_NAME}_p1"]
    with pytest.raises(RuntimeError, match="its solver is 'gurobi', but its solver configuration says 'highs'"):
        sc.solver_config_for(f"{SOLVE_NAME}_p1")
