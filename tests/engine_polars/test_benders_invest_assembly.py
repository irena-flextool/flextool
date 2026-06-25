"""Unit tests for the Benders-side region invest/divest assembly.

Exercises the two pure helpers in
:mod:`flextool.engine_polars._benders` (formerly in the deleted
subgradient ``_lagrangian`` module; the owner-selection / normalisation
logic was ported into Benders and is now its sole home):

* :func:`_resolve_entity_owner` — entity -> owning-region resolution from
  the exclusive per-region membership, with the deterministic
  shared-owner tie-break + warning.
* :func:`_assemble_region_invest_vars` — materialize each region's
  recovered primal invest/divest frame the way ``Solution.value`` does,
  owner-select the rows, and concatenate across regions into one
  whole-system frame per var with disjoint entity keys.

These use SYNTHETIC :class:`polar_high.Problem` objects (a small
invest-like ``Var`` with known ``col_value``s) — no DB / model run.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from polar_high import Problem

from flextool.engine_polars._benders import (
    _assemble_region_invest_vars as _assemble_invest_vars,
    _resolve_entity_owner,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic per-region Problems + col_value arrays
# ---------------------------------------------------------------------------


def _make_invest_problem(
    var_name: str,
    entity_col: str,
    rows: list[tuple[str, str]],
) -> Problem:
    """Build a tiny :class:`Problem` carrying one invest-like ``Var``.

    *rows* is a list of ``(entity, period)`` tuples; the Var is declared
    over ``(entity_col, "d")`` exactly like ``v_invest_p`` / ``v_invest_n``.
    ``Problem.add_var`` assigns the canonical ``col_id``s itself, so a
    matching ``col_value`` array must be sized to ``pb._next_col`` and
    scattered by the Var frame's ``col_id``.
    """
    pb = Problem()
    index = pl.DataFrame(
        {entity_col: [r[0] for r in rows], "d": [r[1] for r in rows]}
    )
    pb.add_var(var_name, (entity_col, "d"), index)
    return pb


def _col_values_for(pb: Problem, var_name: str, values: list[float]) -> np.ndarray:
    """Build a ``col_value`` array for *pb* placing *values* at the Var's
    canonical ``col_id``s (in frame row order)."""
    cv = np.zeros(pb._next_col, dtype=np.float64)
    ids = pb._vars[var_name].frame["col_id"].to_numpy()
    cv[ids] = np.asarray(values, dtype=np.float64)
    return cv


# ---------------------------------------------------------------------------
# _resolve_entity_owner
# ---------------------------------------------------------------------------


class TestResolveEntityOwner:
    def test_disjoint_membership(self) -> None:
        membership = {
            "region_A": {"nodes": {"nA"}, "processes": {"pA"}},
            "region_B": {"nodes": {"nB"}, "processes": {"pB"}},
        }
        owner = _resolve_entity_owner(membership, ["region_A", "region_B"])
        assert owner == {
            "nA": "region_A",
            "pA": "region_A",
            "nB": "region_B",
            "pB": "region_B",
        }

    def test_shared_entity_deterministic_owner_and_warning(
        self, caplog
    ) -> None:
        # ``shared`` is claimed by BOTH regions -> deterministic owner =
        # first in sorted region order = "region_A", plus a warning.
        membership = {
            "region_B": {"nodes": {"nB", "shared"}, "processes": set()},
            "region_A": {"nodes": {"nA", "shared"}, "processes": set()},
        }
        with caplog.at_level(logging.WARNING):
            owner = _resolve_entity_owner(
                membership, ["region_B", "region_A"]
            )
        assert owner["shared"] == "region_A"  # sorted order tie-break
        assert owner["nA"] == "region_A"
        assert owner["nB"] == "region_B"
        assert any(
            "shared across regions" in rec.message
            and "shared" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_all_disjoint(self, caplog) -> None:
        membership = {
            "region_A": {"nodes": {"nA"}, "processes": set()},
            "region_B": {"nodes": {"nB"}, "processes": set()},
        }
        with caplog.at_level(logging.WARNING):
            _resolve_entity_owner(membership, ["region_A", "region_B"])
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# _assemble_invest_vars
# ---------------------------------------------------------------------------


def _owner_pred(owner: dict[str, str], region_of_index: list[str]):
    def pred(region_idx: int, entity: str) -> bool:
        return owner.get(entity) == region_of_index[region_idx]

    return pred


class TestAssembleInvestVars:
    def test_owner_select_and_concat_disjoint(self) -> None:
        # Both regions' invest Var ranges over the WHOLE-system entity set
        # {pA, pB} (invest sets are not re-filtered per region).  region_0
        # owns pA, region_1 owns pB.  Out-of-region values collapse to 0.
        rows = [("pA", "2025"), ("pB", "2025")]
        pb0 = _make_invest_problem("v_invest_p", "p", rows)
        pb1 = _make_invest_problem("v_invest_p", "p", rows)
        cv0 = _col_values_for(pb0, "v_invest_p", [5.0, 0.0])  # pA=5 owned
        cv1 = _col_values_for(pb1, "v_invest_p", [0.0, 7.0])  # pB=7 owned

        owner = {"pA": "r0", "pB": "r1"}
        pred = _owner_pred(owner, ["r0", "r1"])
        out = _assemble_invest_vars([pb0, pb1], [cv0, cv1], pred)

        assert set(out) == {"v_invest_p"}
        frame = out["v_invest_p"]
        assert frame.columns == ["p", "d", "value"]
        got = {(r["p"], r["d"]): r["value"] for r in frame.iter_rows(named=True)}
        # Only owner rows survive -> disjoint (entity, d) keys.
        assert got == {("pA", "2025"): 5.0, ("pB", "2025"): 7.0}
        # Disjointness: one row per (entity, d).
        assert frame.height == frame.select("p", "d").unique().height

    def test_value_matches_solution_value_shape(self) -> None:
        # Columns must be exactly ``(*dims, "value")`` so a SnapshotSolution
        # can serve them via ``.value(name)``.  Node var uses entity col "n".
        rows = [("nA", "2025"), ("nA", "2030")]
        pb = _make_invest_problem("v_invest_n", "n", rows)
        cv = _col_values_for(pb, "v_invest_n", [3.0, 4.0])
        owner = {"nA": "r0"}
        pred = _owner_pred(owner, ["r0"])
        out = _assemble_invest_vars([pb], [cv], pred)
        assert out["v_invest_n"].columns == ["n", "d", "value"]
        vals = {
            r["d"]: r["value"]
            for r in out["v_invest_n"].iter_rows(named=True)
        }
        assert vals == {"2025": 3.0, "2030": 4.0}

    def test_multiple_var_names_and_divest(self) -> None:
        pb0 = Problem()
        pb0.add_var(
            "v_invest_p", ("p", "d"),
            pl.DataFrame({"p": ["pA", "pB"], "d": ["2025", "2025"]}),
        )
        pb0.add_var(
            "v_divest_p", ("p", "d"),
            pl.DataFrame({"p": ["pA", "pB"], "d": ["2025", "2025"]}),
        )
        pb0.add_var(
            "v_invest_n", ("n", "d"),
            pl.DataFrame({"n": ["nA"], "d": ["2025"]}),
        )
        cv = np.zeros(pb0._next_col, dtype=np.float64)
        cv[pb0._vars["v_invest_p"].frame["col_id"].to_numpy()] = [9.0, 0.0]
        cv[pb0._vars["v_divest_p"].frame["col_id"].to_numpy()] = [0.0, 2.0]
        cv[pb0._vars["v_invest_n"].frame["col_id"].to_numpy()] = [1.0]

        owner = {"pA": "r0", "pB": "r0", "nA": "r0"}
        pred = _owner_pred(owner, ["r0"])
        out = _assemble_invest_vars([pb0], [cv], pred)
        # v_divest_n absent from the Problem -> not in output.
        assert set(out) == {"v_invest_p", "v_divest_p", "v_invest_n"}
        assert out["v_invest_p"].filter(pl.col("p") == "pA")["value"][0] == 9.0
        assert out["v_divest_p"].filter(pl.col("p") == "pB")["value"][0] == 2.0
        assert out["v_invest_n"]["value"][0] == 1.0

    def test_nonowner_nonzero_canary_warns(self, caplog) -> None:
        # region_1 (NOT the owner of pA) carries a materially non-zero pA
        # invest value -> canary warning, but owner's value is still the
        # one kept.
        rows = [("pA", "2025")]
        pb0 = _make_invest_problem("v_invest_p", "p", rows)
        pb1 = _make_invest_problem("v_invest_p", "p", rows)
        cv0 = _col_values_for(pb0, "v_invest_p", [5.0])  # owner r0
        cv1 = _col_values_for(pb1, "v_invest_p", [3.3])  # non-owner non-zero
        owner = {"pA": "r0"}
        pred = _owner_pred(owner, ["r0", "r1"])
        with caplog.at_level(logging.WARNING):
            out = _assemble_invest_vars([pb0, pb1], [cv0, cv1], pred)
        # Owner's value wins; non-owner row dropped.
        assert out["v_invest_p"].height == 1
        assert out["v_invest_p"]["value"][0] == 5.0
        assert any(
            "non-owner region" in rec.message and "v_invest_p" in rec.message
            for rec in caplog.records
        )

    def test_nonowner_zero_does_not_warn(self, caplog) -> None:
        rows = [("pA", "2025")]
        pb0 = _make_invest_problem("v_invest_p", "p", rows)
        pb1 = _make_invest_problem("v_invest_p", "p", rows)
        cv0 = _col_values_for(pb0, "v_invest_p", [5.0])
        cv1 = _col_values_for(pb1, "v_invest_p", [0.0])  # collapsed to 0
        owner = {"pA": "r0"}
        pred = _owner_pred(owner, ["r0", "r1"])
        with caplog.at_level(logging.WARNING):
            _assemble_invest_vars([pb0, pb1], [cv0, cv1], pred)
        assert not [
            r for r in caplog.records
            if "non-owner region" in r.message
        ]

    def test_no_invest_vars_returns_empty(self) -> None:
        # A Problem with no invest/divest var -> empty dict.
        pb = Problem()
        pb.add_var(
            "v_flow", ("p", "source", "sink", "d", "t"),
            pl.DataFrame({
                "p": ["x"], "source": ["a"], "sink": ["b"],
                "d": ["2025"], "t": ["t0"],
            }),
        )
        cv = np.zeros(pb._next_col, dtype=np.float64)
        out = _assemble_invest_vars([pb], [cv], lambda i, e: True)
        assert out == {}

    def test_missing_col_values_skips_region(self) -> None:
        # An empty / missing col_value array (older polar_high) skips that
        # region rather than crashing.
        rows = [("pA", "2025")]
        pb0 = _make_invest_problem("v_invest_p", "p", rows)
        pb1 = _make_invest_problem("v_invest_p", "p", rows)
        cv0 = _col_values_for(pb0, "v_invest_p", [5.0])
        owner = {"pA": "r0"}
        pred = _owner_pred(owner, ["r0", "r1"])
        # region 1 has an empty array -> skipped; region 0 still contributes.
        out = _assemble_invest_vars(
            [pb0, pb1], [cv0, np.zeros(0)], pred
        )
        assert out["v_invest_p"].height == 1
        assert out["v_invest_p"]["value"][0] == 5.0
