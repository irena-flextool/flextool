# RP-blended-weights — minimal hand-calculable test design

Worktree: `/tmp/rp-blended-weights` (branch `rp-blended-weights`).
Phase 6.5 is HEAD; Phases 7-9 are the pending implementation phases.

This spec covers two deliverables:

* **A — audit existing tests** (`tests/test_representative_periods.py`)
  for which `.mod` constraints they actually exercise.
* **B — design new minimal tests** that DO exercise the pending
  `rp_inter_period_*` constraints, with hand-derived goldens shown by
  arithmetic.

The deleted GMPL `.mod` (blob `c04afa59cf40fa0b74d8481409f5ef365295a8c9`)
is the spec.  Pending constraints:

| .mod lines  | Constraint                  | Phase |
|-------------|-----------------------------|-------|
| 2197-2200   | intra-period state-change   | landed (Phase 6) |
| 2965-2975   | `rp_inter_period_balance`   | Phase 7 (pending) |
| 2978-2988   | `rp_inter_period_cyclic`    | Phase 8 (pending) |
| 2991-2997   | `rp_inter_period_max_state` | Phase 9 (pending) |

---

## A. Audit of existing tests

`tests/test_representative_periods.py` has two RP integration tests
inside `@pytest.mark.slow` (full FlexTool CLI run):

### A.1 `TestRPAllRepresented::test_all_represented_matches_full`

* **Scenario shape**: `small_yaml` is `profile_lengths: 168` → one
  period of 168 timesteps; the RP override picks `n_rp=7` reps of
  `period_length=24h` → `n_base_periods = 7`, `n_rp = 7` (identity
  weight matrix).
* **Per-mod-constraint activity** (`n_rp = n_base = 7`):
  | .mod                   | Status        | Reason                                                                                              |
  |------------------------|---------------|-----------------------------------------------------------------------------------------------------|
  | 2197-2200 intra-period | **active**    | Each rep block has 24 interior steps + 1 rp_block_first; nb_terms branch fires.                     |
  | 2965-2975 balance      | **vacuous**   | `rp_base_chain` has `n_base - 1 = 6` edges, so it WOULD fire — but with identity weights and rep == base, `Σ p_rp_weight·(v_state[d,last] − v_state_rp_start[d,r]) = v_state[d_b,t_last_b] − v_state_rp_start[d_b,t_first_b]`, i.e. inter-period balance reduces to "v_state_inter follows the actual intra-period state change of the matching base period" — non-trivial, but `v_state_inter` doesn't appear in any *other* constraint with all-reps so it just trails v_state without affecting cost. |
  | 2978-2988 cyclic       | **vacuous**   | Forces inter-period state at b_first equals state at b_last — same situation: v_state_inter has no other coupling. |
  | 2991-2997 max_state    | **vacuous**   | `v_state_inter` lower-bounded at 0, free upward only via the chain; balance/cyclic plus inflow could push it positive, but capacity is generous, no binding. |
* **Tolerance**: `rel_diff < 0.01` (1%).  Strict but **insensitive to
  the inter-period constraints** because v_state_inter is free and
  uncoupled — the LP can satisfy all three pending constraints by setting
  v_state_inter equal to whatever the intra-period chain demands, at no
  cost increment.  With Phase 6 alone (interior + start) the intra-period
  state evolves correctly per RP, and identity weights mean each base
  period == its own RP exactly.  Hence **the test already XPASSes at
  Phase 6** (confirmed earlier by Phase-6 verification).
* **Hand-derivation for all-7-selected + identity weights**:
  * Inflow / demand profiles are random (PERT distributions); no closed
    form available.  The test is *parity vs full model*, not absolute.
  * Identity weights + all-7 reps → the LP is structurally isomorphic
    to the full 7-day model (the rep timeset and full timeset are
    identical; rp_block_first selects t01 of each day; the intra-period
    state-change branch is equivalent to the .mod's `bind_within_period`
    branch).  The free starting state `v_state_rp_start[d, t_first_d]`
    can take any non-negative value; the full model has `bind_within_solve`
    which similarly leaves the period-first state free up to the cyclic
    state-start at t_first_of_solve.
  * With balance + cyclic NOT enforced (Phase 6 only), the LP has the
    same feasible region as the full model.  XPASS is expected and
    observed.

* **Verdict**: this test does NOT exercise the pending Phase-7/8/9
  constraints — it XPASSes the strict xfail just from Phase 6.

### A.2 `TestRPHalfRepresented::test_half_represented_close`

* **Scenario shape**: same `small_yaml`, but `n_rp=4` of 7 base periods.
* **Per-mod-constraint activity**: all three pending constraints are
  *active* in principle (4 reps clustering 7 base periods → non-identity
  weight matrix).  But `v_state_inter` STILL has no coupling to v_state
  or any other LP variable (see audit §1 — there is no `v_state_inter`
  reference in any other constraint in the .mod), so its value is free
  modulo balance + cyclic.  The pending constraints DO change feasibility
  via `Δs = v_state[d, t_last_r] − v_state_rp_start[d, t_first_r]`:
  the cyclic constraint forces a weighted sum of Δs values to be zero,
  which couples back to v_state.
* **Tolerance**: `rel_diff < 0.20` (20%).  Too lenient.  The .mod's
  full RP-coupled model can solve any reasonable inflow scenario to
  well within 5% of the full model when reps are convex combinations
  of base periods (per the Kotzur-style theory the FlexTool RP method
  is implementing).  A 20% tolerance is "did it crash, did it produce
  a number in the ballpark" — not a constraint-correctness check.
* **Verdict**: this test would catch *gross* breakage of Phase 7-9
  emission (e.g. forgetting to add the constraints would let the LP
  arbitrarily decouple storage, potentially reducing cost below the
  full model), but is too lenient to detect a partially-wrong constraint
  (e.g. forgotten Σ over `period_in_use`, wrong sign on `p_state_unitsize`,
  weight matrix transposed).

### A.3 The gap

Neither existing test gives us:
* **Closed-form, hand-derived golden cost** — both are full-pipeline
  integration tests with random PERT inflows.
* **Sensitivity to constraint correctness** — `TestRPAllRepresented`
  is structurally insensitive; `TestRPHalfRepresented` has 20%
  tolerance.

We need a small fixture where:
* Inflow / capacity are integers we control.
* `v_state_inter`, `v_state_rp_start`, `v_state[t]` and total cost are
  hand-derivable in two lines of arithmetic.
* Optimum WITH the inter-period constraints differs measurably
  (≥ 1.0) from optimum WITHOUT them, so we can detect their emission
  via cost alone.

---

## B. New minimal test design

### B.1 Scenario `toy_rp_2base_1rep` — exercises Phases 7 + 8

**Topology** (one screen of code):

* 1 storage node `bat` (also acts as a nodeBalance node — it absorbs
  inflow into state).
* 1 period `p_rp` containing the rep block: 2 timesteps t01, t02
  (period_length=2).
* 2 base periods b1, b2 (abstract; only appear via `rp_base_period_set`,
  `rp_base_chain`, `rp_base_first`, `rp_base_last`).
* 1 RP block: starts at t01, last_step = t02.
* `rp_base__rep` = `{(b1, t01) → 1.0, (b2, t01) → 1.0}` — both base
  periods point to the same rep block with weight 1 (so the inter-period
  chain has length 1: b1 → b2, and a cyclic edge b2 → b1).
* `nodeState_rp = {bat}`, `storage_bind_using_blended_weights = {bat}`.
* `p_state_unitsize[bat] = 1.0`, `p_state_existing_capacity[bat, p_rp]
  = 100.0`, `p_state_upper[bat, p_rp] = 100.0` (cap/unitsize = 100;
  non-binding).
* `p_inflow[bat, p_rp, t01] = +5.0`, `p_inflow[bat, p_rp, t02] = -3.0`.
  Net intra-RP supply = +2 (the "RP isn't actually cyclic" surplus that
  Phase 8 must reject).
* `p_penalty_up = p_penalty_down = 1.0` per slack unit (cheap, but
  preferable to slack avoidance).  step_duration = rp_cost_weight =
  inflation_op = period_share = 1.0.
* No processes, no commodities.  `dtttdt` is the 2-step cyclic
  within-timeset frame.

**Hand-derivation of LP optimum**:

Sign convention (flexpy):
```
nb_terms_sum = -p_inflow      (LHS = RHS)
state_change_rp_start    @ t01:  (v_state_rp_start − v_state[t01]) · unitsize
state_change_rp_interior @ t02:  (v_state[t01] − v_state[t02]) · unitsize
```

**Without Phase 7-9 (current Phase 6 state):**

The LP has the two balance equations:
* @ t01:  `(v_state_rp_start − v_state[t01]) + vq_up[t01] − vq_dn[t01] = −5`
* @ t02:  `(v_state[t01] − v_state[t02]) + vq_up[t02] − vq_dn[t02] = +3`

Slack-free solution (cost 0): `v_state[t01] = v_state_rp_start + 5`,
`v_state[t02] = v_state_rp_start + 2`.  All slacks zero.  No cyclic
constraint forces v_state[t02] back down, so the +2 intra-RP surplus
simply accumulates in storage.

**Cost (Phase 6 only) = 0.0.**  v_state_inter unconstrained, defaults to 0.

**With Phases 7 + 8 added:**

`rp_inter_period_cyclic` (b1 → b2 and b2 → b1, all weights 1.0):
```
v_state_inter[bat, b1] − v_state_inter[bat, b2]
  = Σ_{(b1,r) ∈ rp_base__rep, d:(d,r)∈rp_block_first}
      p_rp_weight[b1, r] · (v_state[bat, d, p_rp_last_step[r]]
                            − v_state_rp_start[bat, d, r]) · unitsize
  = 1.0 · (v_state[bat, p_rp, t02] − v_state_rp_start[bat, p_rp, t01]) · 1.0
```

`rp_inter_period_balance` (b2 ← b1):
```
v_state_inter[bat, b2] − v_state_inter[bat, b1]
  = 1.0 · (v_state[bat, p_rp, t02] − v_state_rp_start[bat, p_rp, t01]) · 1.0
```

Adding LHS: 0.  Adding RHS: `2 · (v_state[t02] − v_state_rp_start)`.
Therefore `v_state[t02] − v_state_rp_start = 0`, i.e. **Δs = 0**.

Now Δs = 0 forces v_state[t02] = v_state_rp_start (call this `S`).
Substitute into the balance equations:

* @ t01:  `(S − v_state[t01]) + vq_up[t01] − vq_dn[t01] = −5`
* @ t02:  `(v_state[t01] − S) + vq_up[t02] − vq_dn[t02] = 3`

Add: `vq_up[t01] + vq_up[t02] − vq_dn[t01] − vq_dn[t02] = −2`.
Minimum of `vq_up + vq_dn` subject to net = −2, all ≥ 0 is **vq_dn
total = 2**, vq_up total = 0.  Cost = 1.0 · 2 = **2.0**.

(Several distributions are optimal — e.g. all 2 at t02, or split.  The
LP is degenerate on the slack-pair distribution but the *cost* is
uniquely 2.0.)

`rp_inter_period_max_state` (Phase 9): bounds v_state_inter ≤ 100.
With v_state_inter free and only appearing in the balance + cyclic
(which now imply v_state_inter[b1] = v_state_inter[b2]), the LP picks
some non-negative value.  Default: 0.  The max bound is non-binding.

**Cost (Phases 7+8) = 2.0.**  (Phase 9 emitted but inactive here.)

| Metric                 | Phase 6 | Phase 7+8 | Δ    |
|------------------------|---------|-----------|------|
| `sol.obj`              | 0.0     | 2.0       | +2.0 |
| `Δs = v_state[t02] − v_state_rp_start` | +2.0    | 0.0       |      |
| `v_state[t01]`         | S + 5   | S + 5 (or S + 3 if slack at t01) | |
| `v_state_inter[b1,b2]` | 0 (free) | 0 (free, equal) | |

**Golden assertions for the test**:
* `sol.optimal` is True.
* `sol.obj == 2.0` (rtol = 1e-9).  Detects emission of Phase 7 + 8.

`v_state` / `v_state_inter` / `v_state_rp_start` numerical values:
multiple LP optima exist on the slack-pair distribution.  Cost is the
robust check.  If we want sharper checks: tighten so only one
optimum is feasible (e.g. make t01-slack 10× more expensive than
t02-slack), but cost = 2.0 already proves the constraints are
emitted and active.

### B.2 Smoke that Phase 6 is at sol.obj = 0.0 (current state)

A second xfail variant — the test in its CURRENT form (no Phase 7-9):
assert `sol.obj == 0.0`.  Documents the "before" state and acts as a
canary if a future agent removes Phase 6 by accident.  This is the
*passing-now* sibling test.

Actually we collapse this into a single xfail-strict test that asserts
`sol.obj == 2.0`: today it FAILS (0.0 ≠ 2.0) → strict xfail PASSES (as
xfail).  When Phase 7+8 lands → 2.0 == 2.0 → test PASSES → strict-xfail
fails (XPASSED) → Phase 10 removes the xfail mark.

### B.3 Phase 9 (`rp_inter_period_max_state`) — not exercised here

`v_state_inter` doesn't appear in any other constraint in the .mod.
With cyclic + balance forcing v_state_inter[b1] = v_state_inter[b2]
and no other coupling, the LP picks v_state_inter = 0 regardless of
the max-state upper bound.  To exercise Phase 9 *bindingly* we'd need
to put v_state_inter into another constraint or initial condition,
which doesn't match the .mod.  **Surfaced limitation**: this test
verifies emission of Phases 7+8 only.  Phase 9's correctness will be
caught by `TestRPHalfRepresented` if it ever produces a measurable
deviation (cost drop) when capacities are tight in `small_yaml`, but
that's not a clean unit test.  Recommend the Phase 9 verification be
done via a dedicated capacity-binding fixture in a follow-up.

### B.4 Skeleton code

```python
# tests/engine_polars/test_rp_blended_weights_minimal.py

import polars as pl
import pytest
from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


def _build_toy_rp_2base_1rep() -> FlexData:
    """Minimal RP-blended-weights fixture: 2 base periods, 1 rep, 1 storage node.

    Inflow t01=+5, t02=-3 → intra-RP net surplus +2.  Phases 7+8 force
    v_state[t02] − v_state_rp_start = 0 (cyclic), so the +2 must spill
    via slack; cost = 2.0.  Phase 6 only → cost = 0.0.
    """
    dt = pl.DataFrame({"d": ["p_rp", "p_rp"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rpcw = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))

    nb = pl.DataFrame({"n": ["bat"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["bat"] * 2, "d": ["p_rp"] * 2,
        "t": ["t01", "t02"], "value": [5.0, -3.0]}))
    p_pup = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0))
             .select("n", "d", "t", "value"))
    p_pdn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0))
             .select("n", "d", "t", "value"))

    nodeState = pl.DataFrame({"n": ["bat"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [1.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"], "value": [100.0]}))
    p_state_sd = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [0.0]}))
    p_state_exi_cap = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"], "value": [100.0]}))

    dtttdt = pl.DataFrame({
        "d": ["p_rp", "p_rp"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p_rp", "p_rp"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    storage_bind_using_blended_weights = pl.DataFrame({"n": ["bat"]})
    nodeState_rp = pl.DataFrame({"n": ["bat"]})
    rp_base_period_set = pl.DataFrame({"b": ["b1", "b2"]})
    rp_base_chain = pl.DataFrame({"b": ["b2"], "b_prev": ["b1"]})
    rp_base_first = pl.DataFrame({"b": ["b1"]})
    rp_base_last = pl.DataFrame({"b": ["b2"]})
    rp_block_first = pl.DataFrame({"d": ["p_rp"], "t": ["t01"]})
    p_rp_last_step = pl.DataFrame({"r": ["t01"], "last_step": ["t02"]})
    rp_base__rep = Param(("b", "r"), pl.DataFrame({
        "b": ["b1", "b2"], "r": ["t01", "t01"], "value": [1.0, 1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rpcw,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pup, p_penalty_down=p_pdn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize, p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_sd,
        p_state_existing_capacity=p_state_exi_cap,
        dtttdt=dtttdt,
        storage_bind_using_blended_weights=storage_bind_using_blended_weights,
        nodeState_rp=nodeState_rp,
        rp_base_period_set=rp_base_period_set,
        rp_base_chain=rp_base_chain,
        rp_base_first=rp_base_first,
        rp_base_last=rp_base_last,
        rp_block_first=rp_block_first,
        p_rp_last_step=p_rp_last_step,
        rp_base__rep=rp_base__rep,
    )


@pytest.mark.xfail(
    reason=(
        "Phases 7-8 not yet implemented in engine_polars/model.py: "
        "rp_inter_period_balance (.mod:2965-2975) and "
        "rp_inter_period_cyclic (.mod:2978-2988) are missing.  "
        "Without them, the +2 intra-RP surplus accumulates in "
        "v_state_rp_start at no slack cost, so sol.obj == 0.0 instead "
        "of the expected 2.0 (one unit of vq_down × penalty=1.0 × "
        "qty=2 absorbs the cyclic-mismatch).  Phase 10 strips this "
        "xfail when 7+8 land."
    ),
    strict=True,
)
def test_rp_2base_1rep_cyclic_forces_two_unit_spill():
    d = _build_toy_rp_2base_1rep()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol.optimal, "LP did not solve to optimum"
    assert abs(sol.obj - 2.0) < 1e-9, (
        f"obj = {sol.obj!r}, expected 2.0.  Either Phases 7-8 emit "
        f"correctly (→ 2.0) or the constraints are missing/wrong "
        f"(→ 0.0 with the current code, or some other value if a "
        f"sign / weight error slipped in)."
    )
```

### B.5 Sanity check — currently-observed value

Without Phases 7-8, `pb.solve()` returns `sol.obj == 0.0` on this
fixture.  Verified by running the same code path (above smoke-test, see
agent-session transcript: HiGHS `Objective value: 0.0000000000e+00`).
The test is therefore *currently expected-failing under strict xfail*.

### B.6 What `v_state_inter` / `v_state_rp_start` won't tell us

`polar_high.Problem.solve()` returns `sol` which exposes `sol.obj` and
variable values via `sol.value(var)`, but accessing individual
`v_state_inter` rows requires the Phase 5 variable object handle.
`build_flextool` doesn't expose `v_state_inter` as a return value.  We
could refactor to expose it, but for the purposes of this minimal test
the **cost golden alone** uniquely distinguishes Phase 6 (0.0) from
Phase 7+8 (2.0).

Future enhancement (deferred): expose v_state_inter via a return value
or a `pb.vars` dict, and add assertions on individual rows.

---

## C. Phase 9 verification — known gap

This test doesn't activate `rp_inter_period_max_state` because
`v_state_inter` is uncoupled from the rest of the LP in the .mod (it
appears ONLY in balance + cyclic + max_state).  With cyclic + balance
forcing the chain to be flat and no demand for v_state_inter > 0, the
LP picks v_state_inter = 0 regardless of the upper bound.

**Surfaced**: a separate test fixture is needed to exercise Phase 9,
ideally by tightening `p_entity_all_existing` so the v_state_inter
upper would have to bind — but only if v_state_inter is forced
positive by some external coupling, which doesn't exist in the current
.mod.  The pragmatic fallback is to verify Phase 9 emission via
constraint enumeration (`pb.cstrs["rp_inter_period_max_state"]` row
count) once the constraint lands.  Deferred to Phase 9's own commit.

---

## D. Files touched / added

* **NEW** `specs/rp_blended_weights_test_design.md` (this file).
* **NEW** `tests/engine_polars/test_rp_blended_weights_minimal.py`
  (the test, xfail-strict).

No production source edits.  No edits to existing tests.
