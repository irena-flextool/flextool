"""Tier-7: ``non_anticipativity_storage_use`` is emitted with the right shape.

Fixture: ``work_2day_stochastic_dispatch_full_storage`` — a genuine
4-forecast fan (period1_realized, period1_upper, period1_lower,
period1_mid) × 48 timesteps, 1 stochastic-group node (hydro_reservoir),
with the .mod's stochastic flag set via ``groupIncludeStochastics``.
(``realized`` was retagged from a designation to a plain forecast, so all
four named branches are forecast members; the real-named ``period1`` is a
weight-0 phantom trunk.)

The constraint pins the per-(n, t) net storage charge equal FORECAST-TO-
FORECAST: the raw ``{(d, b) ∈ period__branch : d != b ∧ b ∈ period_in_use}``
cohort is re-grounded (per anchor group) to a forecast star keyed on the
deterministic reference forecast ``d_ref`` (the lexicographically smallest
forecast — ``period1_lower`` here), tying the OTHER forecasts to it and
NEVER to the phantom trunk ``period1``.  For this fixture: 1 stoch node ×
3 sibling pairs (``period1_lower`` → ``period1_mid`` / ``period1_realized``
/ ``period1_upper``, with the trunk ``period1`` dropped as a weight-0
phantom) × 48 realised-dispatch timesteps = 144 LP rows.

Also asserts the three other non-anticipativity families are absent
(no v_online / v_reserve in this fixture).
"""

from __future__ import annotations

import pytest

from tests.engine_polars.emission._helpers import (
    assert_cstr_absent,
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.emission
def test_non_anticipativity_storage_use_emits_full_shape(
    scenario_workdir,
) -> None:
    pb, data = build(
        scenario_workdir(
            "2_day_stochastic_dispatch", db_fixture="stochastic"
        )
    )

    # Fixture invariants — stochastic data is populated.
    assert data.dt_non_anticipativity is not None
    assert data.period_branch_full is not None
    assert data.groupStochastic is not None
    assert data.period_in_use_set is not None

    # Domain shape.  stoch_nodes via groupStochastic ∩ group_node:
    # add_stochastics group → hydro_reservoir.  The 2day fixture is now a
    # genuine 4-forecast fan (realized retagged to a forecast), so the
    # forecast-to-forecast star is d_ref=period1_lower → {period1_mid,
    # period1_realized, period1_upper}: 3 sibling pairs (the trunk period1
    # is a weight-0 phantom, never an anchor).  dtna: 48 rows re-keyed onto
    # d_ref.  Total: 1 × 3 × 48 = 144 LP rows.
    n_stoch_nodes = 1   # hydro_reservoir
    n_db_pairs = 3      # period1_lower → {period1_mid, period1_realized, period1_upper}
    n_dtna = data.dt_non_anticipativity.height
    assert n_dtna == 48
    expected = n_stoch_nodes * n_db_pairs * n_dtna
    assert_cstr_row_count(pb, "non_anticipativity_storage_use", expected)
    assert_cstr_present(pb, "non_anticipativity_storage_use")

    # The constraint family must carry (n, d, b, t) keys, and the tie is
    # forecast-to-forecast: the hub d is the reference forecast and the
    # trunk period1 appears in NEITHER the d nor the b axis.
    import polars as pl

    recs = pb.cstrs_named("non_anticipativity_storage_use")
    assert len(recs) == 1
    over = recs[0].over
    assert set(over.columns) >= {"n", "d", "b", "t"}
    over_u = over.with_columns(
        pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
    assert set(over_u["d"].unique().to_list()) == {"period1_lower"}
    assert set(over_u["b"].unique().to_list()) == {
        "period1_mid", "period1_realized", "period1_upper"}
    assert "period1" not in over_u["d"].unique().to_list()
    assert "period1" not in over_u["b"].unique().to_list()

    # The other three non-anticipativity families are vacuous on this
    # fixture: no online / no reserve providers.
    assert_cstr_absent(pb, "non_anticipativity_online_integer")
    assert_cstr_absent(pb, "non_anticipativity_online_linear")
    assert_cstr_absent(pb, "non_anticipativity_reserve")
