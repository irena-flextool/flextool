"""W4 Slice 1 (W4a) — dispatch_realization_method hard-reject guard.

``solve.dispatch_realization_method`` is the 2-stage seed of a future
multi-stage market structure.  Only ``'within_solve'`` (the schema
default, today's behaviour) is implemented; ``'separate_dispatch'`` — a
dedicated downstream dispatch stage — has no engine machinery yet and is
HARD-REJECTED until a later slice.

The guard (:func:`_dispatch_realization_raw_guard`) reads the RAW authored
value and its allowed set is EXPLICITLY ``{"", "within_solve"}`` — it
raises :class:`FlexToolConfigError` for ``'separate_dispatch'`` AND for any
unrecognised value.  The normalizing accessor
(:meth:`SolveConfig.dispatch_realization_method_for`) would silently
collapse those to ``'within_solve'``, which is why the guard inspects the
raw dict.

The guard is wired at BOTH v70-style call sites:

* ``_validate_model_solve`` — a fail-fast pass over the model's top-level
  solves.
* the authoritative ``run(complete_solve_name)`` site, which strips the
  ``_roll_\\d+$`` suffix and fires for every solve that RUNS (including
  each roll of a rolling solve and each contained/chained sub-solve).  The
  roll-suffix-strip composition is exercised here directly (mirrors the
  v70 ``test_g1_rolling_suffix_strip_composition``).
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._solve_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# A bare SolveConfig carrying only the two fields the guard/accessor read —
# ``model_solve`` and ``dispatch_realization_method``.
# ---------------------------------------------------------------------------


def _solve_config_with_method(model_solve, method_map) -> SolveConfig:
    sc = object.__new__(SolveConfig)
    sc.model_solve = model_solve
    sc.dispatch_realization_method = method_map
    # ``_validate_model_solve`` also runs the sibling stochastic_invest
    # fail-fast pass, which reads this dict; keep it empty (all-default) so
    # that guard is inert and only the dispatch guard is under test.
    sc.stochastic_invest_method = {}
    return sc


# ===========================================================================
# Accessor normalisation — absent / blank / typo / separate_dispatch
# ===========================================================================


def test_resolver_normalisation():
    sc = _solve_config_with_method(
        {"m": ["s"]},
        {
            "s": "within_solve",
            "loud": "WITHIN_SOLVE",
            "sep": "separate_dispatch",
            "blank": "",
            "junk": "xyz",
        },
    )
    assert sc.dispatch_realization_method_for("s") == "within_solve"
    # case-fold
    assert sc.dispatch_realization_method_for("loud") == "within_solve"
    # the non-default listed value normalises cleanly (resolver recognises
    # it; the GUARD is what rejects it — see below)
    assert (
        sc.dispatch_realization_method_for("sep") == "separate_dispatch"
    )
    # blank / typo / absent collapse to the default
    assert sc.dispatch_realization_method_for("blank") == "within_solve"
    assert sc.dispatch_realization_method_for("junk") == "within_solve"
    assert sc.dispatch_realization_method_for("absent") == "within_solve"


# ===========================================================================
# Guard — hard-reject separate_dispatch + any unrecognised value
# ===========================================================================


def test_raw_guard_rejects_separate_dispatch_and_typos():
    """The guard inspects the RAW authored value (the resolver would
    silently collapse a typo to 'within_solve' and happily pass
    'separate_dispatch' to downstream machinery that does not exist yet),
    so 'separate_dispatch' AND any unrecognised value raise — while
    'within_solve' / blank / absent do not."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    for bad in ("separate_dispatch", "SEPARATE_DISPATCH", "wihtin_solve",
                "expected_value"):
        sc = _solve_config_with_method({"m": ["s"]}, {"s": bad})
        with pytest.raises(FlexToolConfigError) as exc:
            _dispatch_realization_raw_guard(sc, "s")
        assert bad in str(exc.value)

    # Accepted / tolerated raw values → no raise.
    for ok in ("within_solve", "WITHIN_SOLVE", "", "  within_solve  "):
        sc = _solve_config_with_method({"m": ["s"]}, {"s": ok})
        _dispatch_realization_raw_guard(sc, "s")  # no raise
    # Absent → no raise.
    _dispatch_realization_raw_guard(
        _solve_config_with_method({"m": ["s"]}, {}), "s")


def test_separate_dispatch_message_names_solve_and_within_solve():
    """The reject message names the solve and points at 'within_solve'."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method({"m": ["my_solve"]},
                                   {"my_solve": "separate_dispatch"})
    with pytest.raises(FlexToolConfigError) as exc:
        _dispatch_realization_raw_guard(sc, "my_solve")
    msg = str(exc.value)
    assert "my_solve" in msg
    assert "not yet implemented" in msg
    assert "within_solve" in msg


# --- Call site 1: _validate_model_solve fail-fast pass ------------------


def test_validate_model_solve_rejects_separate_dispatch():
    """The fail-fast pass over the model's top-level solves raises on
    'separate_dispatch' before any solve runs."""
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_method({"m": ["s_ok", "s_sep"]},
                                   {"s_sep": "separate_dispatch"})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    with pytest.raises(FlexToolConfigError):
        _validate_model_solve(state)


def test_validate_model_solve_passes_within_solve_and_absent():
    """'within_solve' and absent (all-default) pass and return the solve
    list unchanged."""
    from flextool.engine_polars._orchestration import _validate_model_solve

    # Explicit within_solve on one solve, the other unset.
    sc = _solve_config_with_method({"m": ["s1", "s2"]},
                                   {"s1": "within_solve"})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s1", "s2"]

    # All default (absent).
    sc2 = _solve_config_with_method({"m": ["s1", "s2"]}, {})
    state2 = dataclasses.make_dataclass("S", ["solve"])(solve=sc2)
    assert _validate_model_solve(state2) == ["s1", "s2"]


# --- Call site 2: authoritative run() site (roll-suffix strip) ----------


def test_run_site_rejects_rolling_child_of_separate_dispatch():
    """The authoritative ``run()`` guard strips the ``_roll_N`` suffix
    before the raw-dict lookup, so a rolling / contained child of a
    separate_dispatch solve is rejected — this mirrors the exact code the
    run() site executes (``re.sub`` strip -> ``_dispatch_realization_raw_guard``),
    so a contained/chained/rolling sub-solve cannot silently normalize to
    'within_solve'."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method({"m": ["my_solve"]},
                                   {"my_solve": "separate_dispatch"})
    # Rolling child name as it reaches run(): "<base>_roll_<N>".
    complete_solve_name = "my_solve_roll_2"
    guard_base = re.sub(r"_roll_\d+$", "", complete_solve_name)
    assert guard_base == "my_solve"
    with pytest.raises(FlexToolConfigError):
        _dispatch_realization_raw_guard(sc, guard_base)


def test_run_site_passes_within_solve_rolling_child():
    """A rolling child of a within_solve (or unset) solve passes the
    run()-site guard."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method({"m": ["my_solve"]},
                                   {"my_solve": "within_solve"})
    guard_base = re.sub(r"_roll_\d+$", "", "my_solve_roll_7")
    _dispatch_realization_raw_guard(sc, guard_base)  # no raise
    # Unset base → no raise.
    sc2 = _solve_config_with_method({"m": ["my_solve"]}, {})
    _dispatch_realization_raw_guard(sc2, guard_base)  # no raise
