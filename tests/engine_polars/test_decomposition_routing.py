"""Unit tests for the v60 per-solve decomposition plumbing.

Solver-free coverage of:

* :meth:`SolveConfig.decomposition_for` / ``lagrangian_config_for`` —
  the per-solve scheme + knob resolvers (defaults + normalisation).
* :func:`_orchestration._lagrangian_consume_guard_message` — the loud
  guard that fires when a downstream solve would consume a Lagrangian
  solve's (deferred, absent) handoff.

The solver-backed end-to-end routing check lives in
``test_cli_smoke.py::test_decomposition_lagrangian_db_driven``.
"""
from __future__ import annotations

from flextool.engine_polars._orchestration import (
    _lagrangian_consume_guard_message,
)
from flextool.engine_polars._solve_config import SolveConfig


def _bare_solve_config(**dicts) -> SolveConfig:
    """A :class:`SolveConfig` with only the decomposition dicts set.

    The resolver methods read nothing else, so bypass ``__init__`` (which
    needs a full DB-shaped argument set) and attach just the four dicts.
    """
    sc = SolveConfig.__new__(SolveConfig)
    sc.decomposition = dicts.get("decomposition", {})
    sc.lagrangian_alpha = dicts.get("lagrangian_alpha", {})
    sc.lagrangian_max_iter = dicts.get("lagrangian_max_iter", {})
    sc.lagrangian_tolerance = dicts.get("lagrangian_tolerance", {})
    return sc


# ---------------------------------------------------------------------------
# decomposition_for
# ---------------------------------------------------------------------------


def test_decomposition_for_defaults_to_none_when_unset() -> None:
    sc = _bare_solve_config()
    assert sc.decomposition_for("any_solve") == "none"


def test_decomposition_for_recognises_lagrangian() -> None:
    sc = _bare_solve_config(decomposition={"invest": "lagrangian"})
    assert sc.decomposition_for("invest") == "lagrangian"


def test_decomposition_for_normalises_case_and_whitespace() -> None:
    sc = _bare_solve_config(
        decomposition={"a": "LAGRANGIAN", "b": "  lagrangian  "}
    )
    assert sc.decomposition_for("a") == "lagrangian"
    assert sc.decomposition_for("b") == "lagrangian"


def test_decomposition_for_unknown_value_resolves_to_none() -> None:
    # An explicit "none", a blank string, and an unrecognised value all
    # mean monolithic.
    sc = _bare_solve_config(
        decomposition={"x": "none", "y": "", "z": "benders"}
    )
    assert sc.decomposition_for("x") == "none"
    assert sc.decomposition_for("y") == "none"
    assert sc.decomposition_for("z") == "none"


# ---------------------------------------------------------------------------
# lagrangian_config_for
# ---------------------------------------------------------------------------


def test_lagrangian_config_defaults() -> None:
    sc = _bare_solve_config()
    assert sc.lagrangian_config_for("solve") == (0.1, 80, 1.0)


def test_lagrangian_config_reads_authored_knobs() -> None:
    # params_to_dict stores scalar floats as str(float); the resolver
    # coerces them back (and rounds max_iter to int).
    sc = _bare_solve_config(
        lagrangian_alpha={"s": "5.0"},
        lagrangian_max_iter={"s": "50.0"},
        lagrangian_tolerance={"s": "0.25"},
    )
    assert sc.lagrangian_config_for("s") == (5.0, 50, 0.25)


def test_lagrangian_config_partial_authoring_fills_defaults() -> None:
    sc = _bare_solve_config(lagrangian_alpha={"s": "2.0"})
    assert sc.lagrangian_config_for("s") == (2.0, 80, 1.0)


# ---------------------------------------------------------------------------
# consume-side guard
# ---------------------------------------------------------------------------


def test_guard_silent_when_no_predecessor() -> None:
    assert _lagrangian_consume_guard_message("disp", None, set()) is None


def test_guard_silent_when_predecessor_not_lagrangian() -> None:
    assert _lagrangian_consume_guard_message("disp", "invest", set()) is None


def test_guard_fires_for_downstream_consumer() -> None:
    msg = _lagrangian_consume_guard_message("disp", "invest", {"invest"})
    assert msg is not None
    assert "disp" in msg and "invest" in msg
    assert "lagrangian" in msg.lower()


def test_guard_silent_for_roll_of_same_lagrangian_solve() -> None:
    # A later roll of the same Lagrangian solve shares its base name and
    # is not a cross-scheme consume.
    assert _lagrangian_consume_guard_message(
        "invest", "invest_roll_3", {"invest_roll_3"}
    ) is None
