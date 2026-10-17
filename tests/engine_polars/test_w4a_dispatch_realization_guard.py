"""W4 — ``dispatch_realization_method`` guard (W4a schema + W4b narrowing).

``solve.dispatch_realization_method`` selects how realized operations are
produced.  ``'within_solve'`` (the schema default, today's behaviour) is
always allowed.  ``'separate_dispatch'`` is ACCEPTED for a TOP-LEVEL
``single_solve`` (W4b) OR a TOP-LEVEL ``rolling_window`` solve (OC1 rolling
separate_dispatch) — for which the engine automatically registers a
deterministic realized-dispatch sibling ``<solve>__realized_dispatch``
(single-shot monolithic, or interleaved per-roll for rolling).  These
shapes are still rejected:

  * a solve that itself ``contains_solves`` children;
  * a solve nested inside another solve's ``contains_solves``;
  * any typo / unrecognised value (unchanged hard-reject).

The guard (:func:`_dispatch_realization_raw_guard`) reads the RAW authored
value because the normalizing accessor
(:meth:`SolveConfig.dispatch_realization_method_for`) would silently
collapse a typo to ``'within_solve'``.  It is wired at BOTH v70-style call
sites: ``_validate_model_solve`` (fail-fast over top-level solves) and the
authoritative ``run(complete_solve_name)`` site (which strips the
``_roll_\\d+$`` suffix and fires for every solve that RUNS).
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._solve_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# A bare SolveConfig carrying only the fields the guard/accessor read.
# ---------------------------------------------------------------------------


def _solve_config_with_method(
    model_solve,
    method_map,
    *,
    solve_modes=None,
    contains_solves=None,
) -> SolveConfig:
    sc = object.__new__(SolveConfig)
    sc.model_solve = model_solve
    sc.dispatch_realization_method = method_map
    # The W4b eligibility predicate reads these (via getattr/.get defaults,
    # so partial stubs are tolerated); set them explicitly where a test
    # exercises the non-single_solve / nested shapes.
    sc.solve_modes = solve_modes if solve_modes is not None else {}
    sc.contains_solves = contains_solves if contains_solves is not None else {}
    # ``_validate_model_solve`` also runs the sibling stochastic_invest
    # fail-fast pass; keep it inert.
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
    assert sc.dispatch_realization_method_for("loud") == "within_solve"
    assert sc.dispatch_realization_method_for("sep") == "separate_dispatch"
    assert sc.dispatch_realization_method_for("blank") == "within_solve"
    assert sc.dispatch_realization_method_for("junk") == "within_solve"
    assert sc.dispatch_realization_method_for("absent") == "within_solve"


# ===========================================================================
# Guard — accept an eligible single top-level separate_dispatch solve
# ===========================================================================


def test_raw_guard_accepts_eligible_separate_dispatch():
    """A single top-level ``single_solve`` authoring ``separate_dispatch``
    passes the guard (the engine will auto-spawn its realized-dispatch
    sibling).  ``within_solve`` / blank / absent also pass."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["s"]}, {"s": "separate_dispatch"},
        solve_modes={"s": "single_solve"},
    )
    _dispatch_realization_raw_guard(sc, "s")  # no raise

    # Case-folded value also accepted.
    sc = _solve_config_with_method(
        {"m": ["s"]}, {"s": "SEPARATE_DISPATCH"},
        solve_modes={"s": "single_solve"},
    )
    _dispatch_realization_raw_guard(sc, "s")  # no raise

    # Accepted / tolerated raw values → no raise.
    for ok in ("within_solve", "WITHIN_SOLVE", "", "  within_solve  "):
        sc = _solve_config_with_method({"m": ["s"]}, {"s": ok})
        _dispatch_realization_raw_guard(sc, "s")  # no raise
    # Absent → no raise.
    _dispatch_realization_raw_guard(
        _solve_config_with_method({"m": ["s"]}, {}), "s")


def test_raw_guard_rejects_typos():
    """An unrecognised value (typo) is still a hard-reject — the resolver
    would silently collapse it to 'within_solve'."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    for bad in ("wihtin_solve", "expected_value", "sep_dispatch"):
        sc = _solve_config_with_method(
            {"m": ["s"]}, {"s": bad}, solve_modes={"s": "single_solve"},
        )
        with pytest.raises(FlexToolConfigError) as exc:
            _dispatch_realization_raw_guard(sc, "s")
        assert bad in str(exc.value)


# ===========================================================================
# Guard narrowing — separate_dispatch rejected for non-single-top-level
# ===========================================================================


def test_raw_guard_accepts_rolling_separate_dispatch():
    """A rolling_window separate_dispatch solve is now ACCEPTED (OC1 rolling
    separate_dispatch): the sibling machinery interleaves a realized-dispatch
    roll with each UC roll.  The contains_solves / nesting bans remain (see
    the dedicated reject tests)."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["s"]}, {"s": "separate_dispatch"},
        solve_modes={"s": "rolling_window"},
    )
    _dispatch_realization_raw_guard(sc, "s")  # no raise — rolling is allowed


def test_raw_guard_rejects_separate_dispatch_with_children():
    """A separate_dispatch solve that itself ``contains_solves`` children is
    rejected."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["parent"]}, {"parent": "separate_dispatch"},
        solve_modes={"parent": "single_solve"},
        contains_solves={"parent": ["child_a", "child_b"]},
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _dispatch_realization_raw_guard(sc, "parent")
    assert "contained solves" in str(exc.value)


def test_raw_guard_rejects_nested_separate_dispatch():
    """A separate_dispatch solve nested inside another's ``contains_solves``
    is rejected (it is not a top-level solve)."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["outer"]}, {"nested": "separate_dispatch"},
        solve_modes={"nested": "single_solve"},
        contains_solves={"outer": ["nested"]},
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _dispatch_realization_raw_guard(sc, "nested")
    msg = str(exc.value)
    assert "top-level" in msg or "nested" in msg


def test_separate_dispatch_reject_message_names_solve():
    """A still-rejected shape (a separate_dispatch solve that itself
    ``contains_solves``) names the offending solve in the error."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["my_solve"]}, {"my_solve": "separate_dispatch"},
        solve_modes={"my_solve": "single_solve"},
        contains_solves={"my_solve": ["child"]},
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _dispatch_realization_raw_guard(sc, "my_solve")
    assert "my_solve" in str(exc.value)


# ===========================================================================
# Call site 1: _validate_model_solve fail-fast pass
# ===========================================================================


def test_validate_model_solve_accepts_eligible_separate_dispatch():
    """The fail-fast pass ACCEPTS an eligible single top-level
    separate_dispatch solve (alongside other top-level single_solves)."""
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_method(
        {"m": ["s_ok", "s_sep"]}, {"s_sep": "separate_dispatch"},
        solve_modes={"s_ok": "single_solve", "s_sep": "single_solve"},
    )
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s_ok", "s_sep"]


def test_validate_model_solve_accepts_rolling_separate_dispatch():
    """The fail-fast pass now ACCEPTS a rolling_window separate_dispatch solve
    (OC1 rolling separate_dispatch)."""
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_method(
        {"m": ["s_sep"]}, {"s_sep": "separate_dispatch"},
        solve_modes={"s_sep": "rolling_window"},
    )
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s_sep"]


def test_validate_model_solve_passes_within_solve_and_absent():
    from flextool.engine_polars._orchestration import _validate_model_solve

    sc = _solve_config_with_method({"m": ["s1", "s2"]},
                                   {"s1": "within_solve"})
    state = dataclasses.make_dataclass("S", ["solve"])(solve=sc)
    assert _validate_model_solve(state) == ["s1", "s2"]

    sc2 = _solve_config_with_method({"m": ["s1", "s2"]}, {})
    state2 = dataclasses.make_dataclass("S", ["solve"])(solve=sc2)
    assert _validate_model_solve(state2) == ["s1", "s2"]


# ===========================================================================
# Call site 2: authoritative run() site (roll-suffix strip)
# ===========================================================================


def test_run_site_accepts_rolling_separate_dispatch():
    """A rolling separate_dispatch solve reaches run() as ``<base>_roll_<N>``;
    the guard strips the suffix to the base and now ACCEPTS it — rolling
    separate_dispatch is supported (OC1)."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["my_solve"]}, {"my_solve": "separate_dispatch"},
        solve_modes={"my_solve": "rolling_window"},
    )
    guard_base = re.sub(r"_roll_\d+$", "", "my_solve_roll_2")
    assert guard_base == "my_solve"
    _dispatch_realization_raw_guard(sc, guard_base)  # no raise


def test_run_site_passes_within_solve_rolling_child():
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method({"m": ["my_solve"]},
                                   {"my_solve": "within_solve"})
    guard_base = re.sub(r"_roll_\d+$", "", "my_solve_roll_7")
    _dispatch_realization_raw_guard(sc, guard_base)  # no raise
    sc2 = _solve_config_with_method({"m": ["my_solve"]}, {})
    _dispatch_realization_raw_guard(sc2, guard_base)  # no raise


def test_run_site_sibling_name_no_ops():
    """The auto-spawned sibling authors no dispatch_realization_method (it
    lives only in ``runner.state.solve``, not the guard's config), so the
    run()-site guard no-ops for it — no infinite re-spawn."""
    from flextool.engine_polars._orchestration import (
        _dispatch_realization_raw_guard,
    )

    sc = _solve_config_with_method(
        {"m": ["my_solve"]}, {"my_solve": "separate_dispatch"},
        solve_modes={"my_solve": "single_solve"},
    )
    # sibling absent from the dispatch dict → no raise.
    _dispatch_realization_raw_guard(sc, "my_solve__realized_dispatch")
