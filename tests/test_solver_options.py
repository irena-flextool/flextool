"""Unit tests for the HiGHS solver-option resolver helpers.

After Δ.22 collapsed ``flextool/flextoolrunner/solver_runner.py`` to
its native-cascade shell + the resolver helpers, the only directly
testable surface in this module is the pair of pure functions:

* ``resolve_relax_feasibility(cli_value) -> float | None``
* ``resolve_ipm(cli_flag) -> bool``

Both map a CLI / env-var input to a HiGHS option value.  The CLI
flags (``--ipm`` / ``--relax-feasibility``) were removed from the
production CLI in Δ.22 phase C, but the env-var fallbacks
(``FLEXTOOL_IPM`` / ``FLEXTOOL_RELAX_FEASIBILITY``) survive and the
resolvers are still importable for direct callers.

The legacy ``_run_highs`` plumbing tests (which exercised the GMPL
HiGHS bridge that Δ.22 deleted) were dropped along with the helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make flextool importable when running the tests in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from flextool.flextoolrunner.solver_runner import (
    DEFAULT_RELAX_FEASIBILITY,
    IPM_ENV_VAR,
    RELAX_FEASIBILITY_ENV_VAR,
    resolve_ipm,
    resolve_relax_feasibility,
)


# ---------------------------------------------------------------------------
# resolve_relax_feasibility
# ---------------------------------------------------------------------------


def test_relax_feasibility_cli_none_and_no_env_returns_none(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility(None) is None


def test_relax_feasibility_cli_default_sentinel_returns_default(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("default") == DEFAULT_RELAX_FEASIBILITY


def test_relax_feasibility_cli_explicit_numeric_string(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("1e-4") == 1e-4
    assert resolve_relax_feasibility("0.0001") == 1e-4


def test_relax_feasibility_cli_float(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility(1e-6) == 1e-6


def test_relax_feasibility_cli_invalid_returns_none(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("abc") is None
    # Non-positive is invalid.
    assert resolve_relax_feasibility("0") is None
    assert resolve_relax_feasibility("-1") is None


def test_relax_feasibility_env_truthy_returns_default(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1")
    assert resolve_relax_feasibility(None) == DEFAULT_RELAX_FEASIBILITY
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "yes")
    assert resolve_relax_feasibility(None) == DEFAULT_RELAX_FEASIBILITY


def test_relax_feasibility_env_numeric(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1e-4")
    assert resolve_relax_feasibility(None) == 1e-4


def test_relax_feasibility_env_invalid_returns_none(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "not-a-number-or-bool")
    assert resolve_relax_feasibility(None) is None


def test_relax_feasibility_cli_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1e-3")
    # Explicit CLI float takes precedence.
    assert resolve_relax_feasibility(1e-6) == 1e-6
    # Default sentinel from CLI also wins over env.
    assert resolve_relax_feasibility("default") == DEFAULT_RELAX_FEASIBILITY


# ---------------------------------------------------------------------------
# resolve_ipm
# ---------------------------------------------------------------------------


def test_ipm_cli_false_no_env(monkeypatch) -> None:
    monkeypatch.delenv(IPM_ENV_VAR, raising=False)
    assert resolve_ipm(False) is False


def test_ipm_cli_true(monkeypatch) -> None:
    monkeypatch.delenv(IPM_ENV_VAR, raising=False)
    assert resolve_ipm(True) is True


def test_ipm_env_truthy_variants(monkeypatch) -> None:
    for raw in ("1", "true", "yes", "on", "TRUE", "On"):
        monkeypatch.setenv(IPM_ENV_VAR, raw)
        assert resolve_ipm(False) is True, f"expected True for {raw!r}"


def test_ipm_env_falsy_variants(monkeypatch) -> None:
    for raw in ("0", "false", "no", "off", "", "bogus"):
        monkeypatch.setenv(IPM_ENV_VAR, raw)
        assert resolve_ipm(False) is False, f"expected False for {raw!r}"


def test_ipm_cli_wins_over_falsy_env(monkeypatch) -> None:
    monkeypatch.setenv(IPM_ENV_VAR, "0")
    assert resolve_ipm(True) is True
