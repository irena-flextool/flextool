"""Unit tests for ``write_branch_weights`` (period_calculated_params.py:364).

The function normalises ``p_branch_weight_input`` so that sibling
branches sharing a parent period sum to 1.0 — once at the period level
(``pd_branch_weight``) and once at the (period, time) level
(``pdt_branch_weight``).

These tests handcraft the five input CSVs the writer reads
(``period__branch``, ``solve_branch_weight``, ``first_timesteps``,
``steps_in_use``, ``period_in_use_set``) and assert the normalisation
arithmetic on three shapes:

  * Single-branch self-loop — the degenerate non-stochastic case used by
    every parity baseline today.  Weight must be 1.0.
  * Two equal-weight sibling branches sharing a parent period — the
    canonical 2-branch stochastic case.  Weights must be 0.5 / 0.5.
  * Two unequal-weight branches (1:3) — weights must be 0.25 / 0.75.

The writer reads the CSVs positionally with header skip, so a column
order regression in any of the upstream writers would surface here as
either a numeric mismatch or an empty output file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars._emit_period_calc import (
    write_branch_weights,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_solve_data(
    solve_data_dir: Path,
    *,
    period__branch: list[tuple[str, str]],
    solve_branch_weight: list[tuple[str, float]],
    first_timesteps: list[tuple[str, str]],
    steps_in_use: list[tuple[str, str, float]],
    period_in_use: list[str],
) -> None:
    """Write the five CSVs ``write_branch_weights`` consumes."""
    solve_data_dir.mkdir(parents=True, exist_ok=True)
    (solve_data_dir / "period__branch.csv").write_text(
        "period,branch\n" + "".join(f"{d},{b}\n" for d, b in period__branch)
    )
    (solve_data_dir / "solve_branch_weight.csv").write_text(
        "branch,p_branch_weight_input\n"
        + "".join(f"{b},{w}\n" for b, w in solve_branch_weight)
    )
    (solve_data_dir / "first_timesteps.csv").write_text(
        "period,step\n" + "".join(f"{p},{s}\n" for p, s in first_timesteps)
    )
    (solve_data_dir / "steps_in_use.csv").write_text(
        "period,step,step_duration\n"
        + "".join(f"{p},{s},{d}\n" for p, s, d in steps_in_use)
    )
    (solve_data_dir / "period_in_use_set.csv").write_text(
        "period\n" + "".join(f"{p}\n" for p in period_in_use)
    )


def _read_pd_weights(solve_data_dir: Path) -> dict[str, float]:
    text = (solve_data_dir / "pd_branch_weight.csv").read_text().splitlines()
    out: dict[str, float] = {}
    for line in text[1:]:
        period, value = line.split(",")
        out[period] = float(value)
    return out


def _read_pdt_weights(
    solve_data_dir: Path,
) -> dict[tuple[str, str], float]:
    text = (solve_data_dir / "pdt_branch_weight.csv").read_text().splitlines()
    out: dict[tuple[str, str], float] = {}
    for line in text[1:]:
        period, time, value = line.split(",")
        out[(period, time)] = float(value)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_branch_self_loop_weight_is_1(tmp_path: Path) -> None:
    """The non-stochastic case: one period, self-loop in period__branch.

    All 5 MPS-parity baselines look like this today — sanity check that
    the writer collapses to 1.0 weights and doesn't divide-by-zero or
    misbehave on the trivial topology.
    """
    solve_data = tmp_path / "solve_data"
    _write_solve_data(
        solve_data,
        period__branch=[("p2020", "p2020")],
        solve_branch_weight=[("p2020", 1.0)],
        first_timesteps=[("p2020", "t01")],
        steps_in_use=[
            ("p2020", "t01", 1.0),
            ("p2020", "t02", 1.0),
        ],
        period_in_use=["p2020"],
    )

    write_branch_weights(input_dir=tmp_path, solve_data_dir=solve_data)

    pd_w = _read_pd_weights(solve_data)
    assert pd_w == {"p2020": pytest.approx(1.0)}

    pdt_w = _read_pdt_weights(solve_data)
    assert pdt_w == {
        ("p2020", "t01"): pytest.approx(1.0),
        ("p2020", "t02"): pytest.approx(1.0),
    }


def test_two_equal_weight_branches_normalise_to_half(tmp_path: Path) -> None:
    """Two sibling branches with equal input weights → 0.5 / 0.5.

    The realised period (``p2020``) and an alternative branch
    (``p2020_alt``) share parent ``p2020`` (FlexTool models this with
    ``period__branch`` rows ``(p2020, p2020)`` and ``(p2020,
    p2020_alt)``).  Both branches share the same first timestep and the
    same dt, so the denominator is the sum of both input weights.
    """
    solve_data = tmp_path / "solve_data"
    _write_solve_data(
        solve_data,
        period__branch=[
            ("p2020", "p2020"),
            ("p2020", "p2020_alt"),
        ],
        solve_branch_weight=[
            ("p2020", 1.0),
            ("p2020_alt", 1.0),
        ],
        first_timesteps=[
            ("p2020", "t01"),
            ("p2020_alt", "t01"),
        ],
        steps_in_use=[
            ("p2020", "t01", 1.0),
            ("p2020", "t02", 1.0),
            ("p2020_alt", "t01", 1.0),
            ("p2020_alt", "t02", 1.0),
        ],
        period_in_use=["p2020", "p2020_alt"],
    )

    write_branch_weights(input_dir=tmp_path, solve_data_dir=solve_data)

    pd_w = _read_pd_weights(solve_data)
    assert pd_w == {
        "p2020": pytest.approx(0.5),
        "p2020_alt": pytest.approx(0.5),
    }

    pdt_w = _read_pdt_weights(solve_data)
    assert pdt_w == {
        ("p2020", "t01"): pytest.approx(0.5),
        ("p2020", "t02"): pytest.approx(0.5),
        ("p2020_alt", "t01"): pytest.approx(0.5),
        ("p2020_alt", "t02"): pytest.approx(0.5),
    }


def test_two_unequal_weight_branches_normalise_proportionally(
    tmp_path: Path,
) -> None:
    """Input weights 1 : 3 → 0.25 / 0.75 after normalisation.

    Verifies that the normaliser actually divides by the *sum* of input
    weights and not, e.g., by the count of sibling branches (which would
    produce 0.5 / 1.5).
    """
    solve_data = tmp_path / "solve_data"
    _write_solve_data(
        solve_data,
        period__branch=[
            ("p2020", "p2020"),
            ("p2020", "p2020_alt"),
        ],
        solve_branch_weight=[
            ("p2020", 1.0),
            ("p2020_alt", 3.0),
        ],
        first_timesteps=[
            ("p2020", "t01"),
            ("p2020_alt", "t01"),
        ],
        steps_in_use=[
            ("p2020", "t01", 1.0),
            ("p2020_alt", "t01", 1.0),
        ],
        period_in_use=["p2020", "p2020_alt"],
    )

    write_branch_weights(input_dir=tmp_path, solve_data_dir=solve_data)

    pd_w = _read_pd_weights(solve_data)
    assert pd_w == {
        "p2020": pytest.approx(0.25),
        "p2020_alt": pytest.approx(0.75),
    }

    pdt_w = _read_pdt_weights(solve_data)
    assert pdt_w == {
        ("p2020", "t01"): pytest.approx(0.25),
        ("p2020_alt", "t01"): pytest.approx(0.75),
    }
