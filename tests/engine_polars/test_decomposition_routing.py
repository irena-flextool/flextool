"""Unit tests for the v60/v62 per-solve decomposition plumbing.

Solver-free coverage of:

* :meth:`SolveConfig.decomposition_for` / ``benders_config_for`` —
  the per-solve scheme + knob resolvers (defaults + normalisation).
* :func:`_orchestration._benders_consume_guard_message` — the loud
  guard that (post TIER 1) fires only when a downstream solve would
  consume a Benders predecessor that deposited NO usable investment
  handoff; it is silent when the predecessor handed forward invested
  capacity (the supported invest→dispatch path).

The solver-backed end-to-end routing check lives in
``test_cli_smoke.py::test_decomposition_benders_db_driven``.
"""
from __future__ import annotations

from flextool.engine_polars._orchestration import (
    _benders_consume_guard_message,
)
from flextool.engine_polars._solve_config import SolveConfig


def _bare_solve_config(**dicts) -> SolveConfig:
    """A :class:`SolveConfig` with only the decomposition dicts set.

    The resolver methods read nothing else, so bypass ``__init__`` (which
    needs a full DB-shaped argument set) and attach just the three dicts.
    """
    sc = SolveConfig.__new__(SolveConfig)
    sc.decomposition = dicts.get("decomposition", {})
    sc.benders_max_iter = dicts.get("benders_max_iter", {})
    sc.benders_tolerance = dicts.get("benders_tolerance", {})
    return sc


# ---------------------------------------------------------------------------
# decomposition_for
# ---------------------------------------------------------------------------


def test_decomposition_for_defaults_to_none_when_unset() -> None:
    sc = _bare_solve_config()
    assert sc.decomposition_for("any_solve") == "none"


def test_decomposition_for_recognises_benders() -> None:
    sc = _bare_solve_config(decomposition={"invest": "benders"})
    assert sc.decomposition_for("invest") == "benders"


def test_decomposition_for_normalises_case_and_whitespace() -> None:
    sc = _bare_solve_config(
        decomposition={"a": "BENDERS", "b": "  benders  "}
    )
    assert sc.decomposition_for("a") == "benders"
    assert sc.decomposition_for("b") == "benders"


def test_decomposition_for_unknown_value_resolves_to_none() -> None:
    # An explicit "none", a blank string, and the retired "lagrangian"
    # value all mean monolithic.
    sc = _bare_solve_config(
        decomposition={"x": "none", "y": "", "z": "lagrangian"}
    )
    assert sc.decomposition_for("x") == "none"
    assert sc.decomposition_for("y") == "none"
    assert sc.decomposition_for("z") == "none"


# ---------------------------------------------------------------------------
# benders_config_for
# ---------------------------------------------------------------------------


def test_benders_config_defaults() -> None:
    sc = _bare_solve_config()
    assert sc.benders_config_for("solve") == (50, 1e-3)


def test_benders_config_reads_authored_knobs() -> None:
    # params_to_dict stores scalar floats as str(float); the resolver
    # coerces them back (and rounds max_iter to int).
    sc = _bare_solve_config(
        benders_max_iter={"s": "30.0"},
        benders_tolerance={"s": "0.005"},
    )
    assert sc.benders_config_for("s") == (30, 0.005)


def test_benders_config_partial_authoring_fills_defaults() -> None:
    sc = _bare_solve_config(benders_max_iter={"s": "20.0"})
    assert sc.benders_config_for("s") == (20, 1e-3)


# ---------------------------------------------------------------------------
# consume-side guard
# ---------------------------------------------------------------------------


def test_guard_silent_when_no_predecessor() -> None:
    assert _benders_consume_guard_message(
        "disp", None, set(), set()
    ) is None


def test_guard_silent_when_predecessor_not_benders() -> None:
    assert _benders_consume_guard_message(
        "disp", "invest", set(), set()
    ) is None


def test_guard_fires_when_predecessor_has_no_invest_handoff() -> None:
    # A Benders predecessor that deposited NO usable investment handoff
    # (not in the invest-handoff set) → consuming it loads nothing investy.
    msg = _benders_consume_guard_message(
        "disp", "invest", {"invest"}, set()
    )
    assert msg is not None
    assert "disp" in msg and "invest" in msg
    assert "benders" in msg.lower()


def test_guard_silent_when_predecessor_deposited_invest_handoff() -> None:
    # TIER 1 — a Benders invest solve that handed forward usable
    # capacity (in the invest-handoff set) is a SUPPORTED consume; silent.
    assert _benders_consume_guard_message(
        "disp", "invest", {"invest"}, {"invest"}
    ) is None


def test_guard_silent_for_roll_of_same_benders_solve() -> None:
    # A later roll of the same Benders solve shares its base name and
    # is not a cross-scheme consume.
    assert _benders_consume_guard_message(
        "invest", "invest_roll_3", {"invest_roll_3"}, set()
    ) is None
