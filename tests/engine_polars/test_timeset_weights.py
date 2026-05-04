"""flexpy port of flextool's ``test_timeset_weights.py``.

flextool's test exercises ``write_timeset_cost_weight`` — the
solve-side writer that normalizes per-timestep weights so the
period-sum equals ``n_active_steps`` (so a uniform input reproduces
weight=1 per step).  flexpy doesn't have an equivalent writer; it
*consumes* the canonical ``rp_cost_weight.csv`` on disk via
:func:`flextool.input._load_time`.

Porting strategy (loader-side instead of writer-side, per gap-B6
guidance "flexpy's loader already handles the inputs but should
verify the same correctness assertions"):

1. Build a tiny ``solve_data/`` directory in ``tmp_path`` matching
   flexpy's loader contract (``steps_in_use.csv`` for the (d, t)
   index; ``rp_cost_weight.csv`` with the canonical
   ``period,time,weight`` header; the inflation / period_share
   files needed to make ``_load_time`` succeed).
2. Drive ``_load_time`` and inspect ``p_rp_cost_weight.frame``.
3. Assert the same correctness invariants the flextool writer-test
   pins:
   * The handoff example {0.4, 0.8, 1.2, 1.6} loads verbatim.
   * Uniform 1.0 input is identity (default behavior).
   * Missing entries fall back to 1.0 (.mod's "default 1" clause).
   * Empty / absent CSV ⇒ all 1.0.
   * Multiple periods are loaded independently.
   * Only some periods listed ⇒ unlisted ones default to 1.0.

The flextool writer guarantees the on-disk file already encodes
the *normalized × n* values; this test pins the loader contract
that produces ``p_rp_cost_weight`` from that file.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars.input import _load_time


def _write_steps_in_use(sd: Path, period_steps: dict[str, list[str]]) -> None:
    rows = []
    for period, steps in period_steps.items():
        for step in steps:
            rows.append({"period": period, "step": step, "step_duration": 1.0})
    pl.DataFrame(rows).write_csv(sd / "steps_in_use.csv")


def _write_rp_cost_weight(sd: Path, rows: list[tuple[str, str, float]]) -> None:
    """Write canonical ``rp_cost_weight.csv`` with header ``period,time,weight``."""
    if rows:
        df = pl.DataFrame(
            rows, schema=["period", "time", "weight"], orient="row"
        )
    else:
        df = pl.DataFrame(
            schema={"period": pl.Utf8, "time": pl.Utf8, "weight": pl.Float64}
        )
    df.write_csv(sd / "rp_cost_weight.csv")


def _write_aux(sd: Path, periods: list[str]) -> None:
    """Write the auxiliary files _load_time needs (inflation, period_share)."""
    pl.DataFrame({"period": periods, "value": [1.0] * len(periods)}).write_csv(
        sd / "p_inflation_factor_operations_yearly.csv"
    )
    pl.DataFrame({"period": periods, "value": [1.0] * len(periods)}).write_csv(
        sd / "complete_period_share_of_year_calc.csv"
    )


def _setup(tmp_path: Path, period_steps: dict[str, list[str]]) -> Path:
    sd = tmp_path / "solve_data"
    sd.mkdir()
    _write_steps_in_use(sd, period_steps)
    _write_aux(sd, list(period_steps.keys()))
    return sd


def _rpcw_dict(rp_cw_param) -> dict[tuple[str, str], float]:
    """Convert ``Param(("d","t"))`` frame to a dict for easy assertions."""
    return {
        (row["d"], row["t"]): row["value"]
        for row in rp_cw_param.frame.iter_rows(named=True)
    }


class TestLoadRpCostWeight:
    """Each test mirrors a flextool ``TestWriteTimesetCostWeight`` case,
    but operates on the *loader* (post-writer) end.  Inputs encode the
    values the writer would have produced (normalized × n)."""

    def test_handoff_example_loads_verbatim(self, tmp_path: Path) -> None:
        """Mirrors flextool's ``test_handoff_example``.  The writer would
        have produced 0.4 / 0.8 / 1.2 / 1.6 from {0.1, 0.2, 0.3, 0.4}
        (already-summing-to-1 × n=4); the loader must read those values
        back unchanged."""
        sd = _setup(tmp_path, {"p1": ["t1", "t2", "t3", "t4"]})
        _write_rp_cost_weight(sd, [
            ("p1", "t1", 0.4),
            ("p1", "t2", 0.8),
            ("p1", "t3", 1.2),
            ("p1", "t4", 1.6),
        ])
        _, _, rp_cw, _, _ = _load_time(sd)
        d = _rpcw_dict(rp_cw)
        assert d == {
            ("p1", "t1"): 0.4,
            ("p1", "t2"): 0.8,
            ("p1", "t3"): 1.2,
            ("p1", "t4"): 1.6,
        }
        # Period-sum equals n=4 (the writer's invariant survives the
        # round-trip: total_weight = n means /period_share scales evenly).
        assert sum(d.values()) == pytest.approx(4.0, rel=1e-9)

    def test_uniform_input_is_identity(self, tmp_path: Path) -> None:
        """Mirrors ``test_uniform_input_reproduces_default``.  All-1.0
        on disk must come back as all-1.0."""
        sd = _setup(tmp_path, {"p1": ["t1", "t2", "t3", "t4"]})
        _write_rp_cost_weight(sd, [
            ("p1", "t1", 1.0),
            ("p1", "t2", 1.0),
            ("p1", "t3", 1.0),
            ("p1", "t4", 1.0),
        ])
        _, _, rp_cw, _, _ = _load_time(sd)
        assert all(v == 1.0 for v in _rpcw_dict(rp_cw).values())

    def test_missing_steps_default_to_one(self, tmp_path: Path) -> None:
        """Mirrors the loader-side analogue of
        ``test_missing_steps_are_treated_as_zero``.  flexpy's loader has
        a different convention from flextool's writer here: the .mod
        spec is ``param p_rp_cost_weight ... default 1`` (input.py:575),
        so unlisted (d, t) rows fall back to 1.0 — NOT 0.0.  The writer
        would have produced explicit 0.0 / 2.0 entries; flexpy's loader
        only honors what's on disk and defaults the rest to 1.0.

        Pin both behaviors:
          * Listed entries load verbatim.
          * Unlisted entries get the default 1.0.
        """
        sd = _setup(tmp_path, {"p": ["t1", "t2", "t3", "t4"]})
        # Writer's output for {t1: 1.0, t3: 1.0} would be:
        #   {t1: 2.0, t2: 0.0, t3: 2.0, t4: 0.0}
        # but flexpy's loader fixture mirrors the file flextool writes,
        # so we put those exact values on disk:
        _write_rp_cost_weight(sd, [
            ("p", "t1", 2.0),
            ("p", "t2", 0.0),
            ("p", "t3", 2.0),
            ("p", "t4", 0.0),
        ])
        _, _, rp_cw, _, _ = _load_time(sd)
        d = _rpcw_dict(rp_cw)
        assert d == {("p", "t1"): 2.0, ("p", "t2"): 0.0,
                     ("p", "t3"): 2.0, ("p", "t4"): 0.0}

    def test_missing_csv_defaults_to_one(self, tmp_path: Path) -> None:
        """Mirrors ``test_returns_false_when_no_timeset_has_weights``.
        The writer skips emission when no timeset has weights; the
        loader must default every (d, t) to 1.0 in that case."""
        sd = _setup(tmp_path, {"p": ["t1", "t2"]})
        # No rp_cost_weight.csv at all.
        _, _, rp_cw, _, _ = _load_time(sd)
        assert _rpcw_dict(rp_cw) == {("p", "t1"): 1.0, ("p", "t2"): 1.0}

    def test_empty_csv_defaults_to_one(self, tmp_path: Path) -> None:
        """Header-only ``rp_cost_weight.csv`` (no rows) is also a
        legitimate "no overrides" state.  Loader must default to 1.0."""
        sd = _setup(tmp_path, {"p": ["t1", "t2"]})
        _write_rp_cost_weight(sd, [])
        _, _, rp_cw, _, _ = _load_time(sd)
        assert _rpcw_dict(rp_cw) == {("p", "t1"): 1.0, ("p", "t2"): 1.0}

    def test_multiple_periods_independent(self, tmp_path: Path) -> None:
        """Mirrors ``test_multiple_periods_independent_normalization``.
        Per-period weights load independently and don't bleed across
        period boundaries."""
        sd = _setup(tmp_path, {
            "p1": ["t1", "t2"],
            "p2": ["t3", "t4", "t5"],
        })
        _write_rp_cost_weight(sd, [
            ("p1", "t1", 0.5), ("p1", "t2", 1.5),
            ("p2", "t3", 0.3), ("p2", "t4", 1.2), ("p2", "t5", 1.5),
        ])
        _, _, rp_cw, _, _ = _load_time(sd)
        d = _rpcw_dict(rp_cw)
        assert d == {
            ("p1", "t1"): 0.5, ("p1", "t2"): 1.5,
            ("p2", "t3"): 0.3, ("p2", "t4"): 1.2, ("p2", "t5"): 1.5,
        }
        # Period sums respect the writer's invariant (= n_steps):
        assert d[("p1", "t1")] + d[("p1", "t2")] == pytest.approx(2.0)
        assert (d[("p2", "t3")] + d[("p2", "t4")] + d[("p2", "t5")]
                == pytest.approx(3.0))

    def test_only_some_periods_have_weights(self, tmp_path: Path) -> None:
        """Mirrors ``test_only_some_periods_have_weights``.  When the
        writer emits rows only for one period, the loader gives the
        weighted period its loaded values and the unweighted period
        defaults to 1.0 per step."""
        sd = _setup(tmp_path, {
            "p1": ["t1", "t2"],
            "p2": ["t3", "t4"],
        })
        _write_rp_cost_weight(sd, [
            ("p1", "t1", 0.4), ("p1", "t2", 1.6),
            # p2 absent — writer skipped emission for that period.
        ])
        _, _, rp_cw, _, _ = _load_time(sd)
        d = _rpcw_dict(rp_cw)
        assert d == {
            ("p1", "t1"): 0.4, ("p1", "t2"): 1.6,
            ("p2", "t3"): 1.0, ("p2", "t4"): 1.0,
        }


def test_real_fixture_has_writer_invariant() -> None:
    """End-to-end: ``work_base_weighted`` is the canonical
    real-fixture flextool emits with non-uniform weights.  Verify the
    on-disk file flexpy consumes already encodes the writer's
    invariant — period-sum equals n_active_steps (= 48 here).
    """
    work = (
        Path(__file__).resolve().parent
        / "data"
        / "work_base_weighted"
    )
    df = pl.read_csv(work / "solve_data" / "rp_cost_weight.csv")
    by_period = df.group_by("period").agg(pl.col("weight").sum())
    n_per_period = (df.group_by("period")
                      .agg(pl.col("time").n_unique().alias("n")))
    # join and assert sum(weight) == n for every period
    joined = by_period.join(n_per_period, on="period")
    for row in joined.iter_rows(named=True):
        assert row["weight"] == pytest.approx(float(row["n"]), rel=1e-9), (
            f"work_base_weighted period {row['period']}: weight-sum "
            f"{row['weight']} != n_steps {row['n']} — writer invariant "
            "violated upstream"
        )
