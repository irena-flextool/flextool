"""``2_day_stochastic_dispatch`` (full topology) — A6 parity test.

Full fixture: 4 branches × hydro_reservoir storage + hydro_plant +
wind_plant + gas_plant + coal_plant + demand_node.  Exercises BOTH
features added in A6:

* ``pdt_branch_weight`` factor on every dispatch-class objective term
  (~0.25 per branch on a 4-branch fixture)
* ``non_anticipativity_storage_use`` constraint pinning the per-timestep
  net storage charge LHS across branch siblings (mod:4173-4217),
  preventing storage from drifting independently across branches at
  the cheaper-objective expense.

Plus the ``storage_state_solve_horizon_reference_value`` end-state
constraint (mod:2802-2822) which the fixture's
``storage_solve_horizon_method=use_reference_value`` activates — pins
v_state at the last (d, t) of period_last to ``reference_value ×
existing/unitsize``.

Pre-A6: flexpy was 3.85× higher than flextool (combination of missing
4× pdt_branch_weight and cheaper-LP storage drift).
Post-A6: machine-epsilon parity.
"""

from pathlib import Path

import polars as pl

from polar_high_opt import Problem
from flextool.engine_polars import build_flextool, load_flextool
import pytest

pytestmark = pytest.mark.solver


WORK = (Path(__file__).resolve().parent
        / "data" / "work_2day_stochastic_dispatch_full_storage")


def test_2day_stochastic_dispatch_full_storage_parity():
    data = load_flextool(WORK)

    # Fixture sanity — stochastic data + storage data populated.
    assert data.pdt_branch_weight is not None
    assert data.dt_non_anticipativity is not None
    assert data.period_branch_full is not None
    assert data.groupStochastic is not None
    assert data.nodeState is not None
    assert data.nodeState.height == 1   # hydro_reservoir
    assert data.storage_use_reference_value is not None
    assert data.p_storage_state_reference_value is not None

    pb = Problem()
    build_flextool(pb, data)

    # Constraint families are wired.
    cstr_names = set(pb.cstr_names())
    assert "non_anticipativity_storage_use" in cstr_names
    assert "storage_state_solve_horizon_reference_value" in cstr_names

    sol = pb.solve()
    assert sol.optimal

    # Γ.4: prefer golden_obj.json if present, fall back to parquet.
    from _golden import assert_obj_within
    assert_obj_within(sol.obj, WORK,
                       parquet_glob="v_obj__2day_dispatch.parquet")

    # v_state must be equal across branches at every dt_non_anticipativity
    # timestep (anchor period == realised period1 → siblings).
    v_state_df = sol.value("v_state")
    spread = (v_state_df
        .group_by(["n", "t"])
        .agg([pl.col("value").min().alias("mn"),
              pl.col("value").max().alias("mx")])
        .with_columns(spread=pl.col("mx") - pl.col("mn")))
    max_spread = float(spread["spread"].max())
    assert max_spread < 1e-9, (
        f"v_state varies across branches by {max_spread} — non-anticipativity "
        f"failed to enforce equal storage state.")
