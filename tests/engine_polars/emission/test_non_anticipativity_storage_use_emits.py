"""Tier-7: ``non_anticipativity_storage_use`` is emitted with the right shape.

Fixture: ``work_2day_stochastic_dispatch_full_storage`` — 4 branches
(period1, period1_upper, period1_lower, period1_mid) × 48 timesteps,
1 stochastic-group node (hydro_reservoir), with the .mod's stochastic
flag set via ``groupIncludeStochastics``.

The .mod constraint (mod:4173-4217) has domain
``(n, d, b, t) ∈ stoch_nodes × {(d, b) ∈ period__branch : d != b ∧ b ∈ period_in_use}
× dt_non_anticipativity``.  For this fixture: 1 stoch node × 3 sibling
branches (after dropping the metadata-only ``period1_realized``) × 48
realised-dispatch timesteps = 144 LP rows.

Also asserts the three other non-anticipativity families are absent
(no v_online / v_reserve in this fixture).
"""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from tests.emission._helpers import (
    assert_cstr_absent,
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.emission
def test_non_anticipativity_storage_use_emits_full_shape() -> None:
    pb, data = build(
        DATA_DIR / "work_2day_stochastic_dispatch_full_storage")

    # Fixture invariants — stochastic data is populated.
    assert data.dt_non_anticipativity is not None
    assert data.period_branch_full is not None
    assert data.groupStochastic is not None
    assert data.period_in_use_set is not None

    # Domain shape.  stoch_nodes via groupStochastic ∩ group_node:
    # add_stochastics group → hydro_reservoir.  db_pairs: d ≠ b AND b ∈
    # period_in_use → 3 (period1 → upper/lower/mid).  dtna: 48 rows for
    # period1.  Total: 1 × 3 × 48 = 144 LP rows.
    n_stoch_nodes = 1   # hydro_reservoir
    n_db_pairs = 3      # period1 → {upper, lower, mid}
    n_dtna = data.dt_non_anticipativity.height
    assert n_dtna == 48
    expected = n_stoch_nodes * n_db_pairs * n_dtna
    assert_cstr_row_count(pb, "non_anticipativity_storage_use", expected)
    assert_cstr_present(pb, "non_anticipativity_storage_use")

    # The constraint family must carry (n, d, b, t) keys.
    recs = pb.cstrs_named("non_anticipativity_storage_use")
    assert len(recs) == 1
    assert set(recs[0].over.columns) >= {"n", "d", "b", "t"}

    # The other three non-anticipativity families are vacuous on this
    # fixture: no online / no reserve providers.
    assert_cstr_absent(pb, "non_anticipativity_online_integer")
    assert_cstr_absent(pb, "non_anticipativity_online_linear")
    assert_cstr_absent(pb, "non_anticipativity_reserve")
