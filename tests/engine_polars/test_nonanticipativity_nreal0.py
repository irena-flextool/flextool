"""R-NA — n_real=0 forecast-to-forecast non-anticipativity witness.

Corrected stochastic model: the stochastic fan is FORECASTS ONLY and
non-anticipativity ties the forecasts to EACH OTHER (via a deterministic
reference forecast), never to a realised anchor.  This is the capability
witness that the NA machinery works with NO realised branch (``n_real = 0``).

Fixture ``stoch_two_period_nreal0.json`` (built via
``build_stoch_two_period_nreal0.py``): the Slice G ``free_ops`` topology —
two 3-step periods ``p2035`` / ``p2040``, branch-asymmetric wind timing, a
stochastic-group storage node ``resv`` — but with all three branches
``mid`` / ``up`` / ``low`` as FORECASTS (``realized: no``, weight 1.0 each,
``n_real = 0``) and a curated ``shared_operation_periods = [p2035, p2040]``
window so ``dt_non_anticipativity`` populates without any realised_dispatch.

The storage NA family must:

* re-ground each anchor group to a per-group FORECAST star keyed on the
  lexicographically-smallest forecast — ``p2035_low`` for the p2035 group,
  ``p2040_low`` for the p2040 group;
* tie NO realised/trunk/base period (``p2035`` / ``p2040`` ∉ over.d ∪ over.b);
* hold ``v_state[resv]`` equal across the forecast copies at the tied steps;
* reach an Optimal solve.
"""
from __future__ import annotations

import os

import pytest

# The base (phantom) period names that must never appear in an NA tie.
BASE_PERIODS = {"p2035", "p2040"}
# Expected per-group reference forecasts (lexicographically smallest).
EXPECTED_D_REF = {"p2035_low", "p2040_low"}
# Expected tied siblings (the other forecasts of each group).
EXPECTED_B = {"p2035_mid", "p2035_up", "p2040_mid", "p2040_up"}


@pytest.fixture(scope="module")
def nreal0_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("nreal0",
                                db_fixture="stoch_two_period_nreal0")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def nreal0_resolved(nreal0_wf):
    from polar_high import Problem

    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(nreal0_wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, pb, sol


def _na_over(pb):
    """Concatenated ``over`` frame of the storage NA family (d, b, t as str)."""
    import polars as pl

    recs = pb.cstrs_named("non_anticipativity_storage_use")
    assert recs, "non_anticipativity_storage_use family did not fire"
    over = pl.concat([r.over for r in recs])
    return over.with_columns(
        pl.col("d").cast(pl.Utf8),
        pl.col("b").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    )


@pytest.mark.solver
def test_nreal0_solve_optimal(nreal0_resolved):
    """The n_real=0 fan solves to a strict optimum (NA fires with no
    realised branch)."""
    _data, _pb, sol = nreal0_resolved
    assert sol.optimal


@pytest.mark.solver
def test_nreal0_no_realized_designation(nreal0_resolved):
    """The fan carries ZERO realised members — ``n_real = 0``."""
    data, _pb, _sol = nreal0_resolved
    # dt_non_anticipativity still populates (from the curated window).
    assert data.dt_non_anticipativity is not None
    assert data.dt_non_anticipativity.height > 0


@pytest.mark.solver
def test_nreal0_db_pairs_reground_forecast_star(nreal0_resolved):
    """The storage NA family re-grounds to a per-anchor-group forecast star.

    Each group ties its forecasts to the lexicographically-smallest one
    (``p2035_low`` / ``p2040_low``); no row couples distinct period groups.
    """
    _data, pb, _sol = nreal0_resolved
    over = _na_over(pb)

    d_vals = set(over["d"].unique().to_list())
    b_vals = set(over["b"].unique().to_list())
    assert d_vals == EXPECTED_D_REF, d_vals
    assert b_vals == EXPECTED_B, b_vals

    # Star shape per group: d_ref ties the OTHER two forecasts of its own
    # period group, and no cross-period pair exists.
    pairs = {(r["d"], r["b"]) for r in over.select("d", "b").unique()
             .iter_rows(named=True)}
    assert pairs == {
        ("p2035_low", "p2035_mid"), ("p2035_low", "p2035_up"),
        ("p2040_low", "p2040_mid"), ("p2040_low", "p2040_up"),
    }, pairs
    for dd, bb in pairs:
        assert dd.split("_")[0] == bb.split("_")[0], (dd, bb)


@pytest.mark.solver
def test_nreal0_no_trunk_or_base_period_tied(nreal0_resolved):
    """No realised/trunk/base period appears as an NA hub or sibling."""
    _data, pb, _sol = nreal0_resolved
    over = _na_over(pb)
    d_vals = set(over["d"].unique().to_list())
    b_vals = set(over["b"].unique().to_list())
    assert not (BASE_PERIODS & d_vals), (BASE_PERIODS & d_vals)
    assert not (BASE_PERIODS & b_vals), (BASE_PERIODS & b_vals)


@pytest.mark.solver
def test_nreal0_forecasts_v_state_equal_at_tied_steps(nreal0_resolved):
    """At the tied steps the forecast copies' ``v_state[resv]`` are equal.

    The committed storage decision is the same across the forecast branches
    (its non-anticipative realisation), while the untied base/phantom period
    is excluded.  A positive number of tied cells is checked so the witness
    cannot be silently vacuous.
    """
    import polars as pl

    _data, pb, sol = nreal0_resolved
    assert "v_state" in sol._vars, "fixture must have a storage v_state"
    over = _na_over(pb)

    # Tied forecast periods (d_ref + siblings) and tied timesteps, per group.
    tied_periods = set(over["d"].unique().to_list()) | set(
        over["b"].unique().to_list())
    assert tied_periods and not (BASE_PERIODS & tied_periods)

    vs = sol.value("v_state")
    vs = vs.with_columns(
        pl.col("n").cast(pl.Utf8).alias("__n"),
        pl.col("d").cast(pl.Utf8).alias("__d"),
        pl.col("t").cast(pl.Utf8).alias("__t"),
    ).filter((pl.col("__n") == "resv") & pl.col("__d").is_in(tied_periods))

    tol = 1e-6
    checked = 0
    # Group by the period prefix (p2035 / p2040) so forecasts of the same
    # group are compared; within each, assert equal v_state at each tied t.
    vs = vs.with_columns(
        pl.col("__d").str.split("_").list.first().alias("__grp"))
    spread = (vs.group_by(["__grp", "__t"])
                .agg(n_fc=pl.col("value").len(),
                     vmax=pl.col("value").max(),
                     vmin=pl.col("value").min())
                .filter(pl.col("n_fc") > 1))
    for row in spread.iter_rows(named=True):
        assert row["vmax"] - row["vmin"] <= tol + 1e-6 * max(
            1.0, abs(row["vmax"]), abs(row["vmin"])), (
            f"forecast v_state spread {row['vmax'] - row['vmin']} at "
            f"(grp={row['__grp']}, t={row['__t']}) exceeds tol")
        checked += 1
    assert checked > 0, "no tied (grp, t) cells with >1 forecast were checked"
