# Unit floors, input shares and reserve shortfall

Engine notes for `model.py::_add_unit_floors`, `_add_min_input_share` and
`_reserve.reserve_shortfall_scale`. User-facing semantics are in
`docs/reference.md` (`min_load`, `capacity_min_coeff`, `input_share_min`,
Reserves).

## Output measure

`unit_measured_arcs(d)` — the arcs that define a unit's capacity:
the `conversion_flow_coeff ≠ 0` output arcs of an indirect unit (the
`maxOutputSum` arc set, `_indirect_sum_arcs`), the single capped arc of a
direct unit (including the `(p, src, p)` arc of a sink-less unit). Inputs of
indirect units are never in it.

## Rows

| name | over | row |
|---|---|---|
| `minFlow_minload_<kind>` | `(p,d,t) ∈ p_idx`, online ∩ min_load units | `Σ_A v_flow ≥ v_online·min_load + Rdn_p` |
| `minFlow_output_floor_<kind>` | `(p,p,k,d,t)`, units with ≥ 2 measured outputs, `capacity_min_coeff_k > 0` | `v_flow ≥ v_online·min_load·cmin_k + Rdn_k` |
| `minFlow_reserve` | `(p,d,t)`, units with a reducing reserve on A, not covered above | `Σ_A v_flow ≥ Rdn_p` |
| `minFlow_reserve_arc` | `(p,s,k,d,t)`, reducing reserve on an arc of a multi-output unit or outside A | `v_flow ≥ Rdn_a` |
| `minInputShare` | `(p,s,p,d,t)`, indirect units with ≥ 2 conv≠0 inputs, `share_s > 0` | `conv_s·in_s ≥ share_s·Σ conv·in` |

`Rdn` = reserve that reduces the arc: `down` at the sink node, `up` at the
source node. Factor 1 when `v_flow` is the flow at that node (sink side; an
indirect unit's input; a sink-less unit); `1/slope` for a source-side reserve
on a direct input→output arc. All rows are dimensionless (unit-count flow
units) in the autoscale registry.

Decisions: the floor is not scaled by availability; connections are excluded;
an indirect unit's input-side `up` reserve gets only the zero floor on that
input arc (expressing "the input cut takes the outputs below min load" needs
the conversion mix), so combined with output-side `down` reserve such a unit
can end up below `min_load`.

## Reserve shortfall multiplier

`vq_reserve ∈ [0, 1]` (bounded for numerics) is multiplied by
`reserve_shortfall_scale(d)` in every `reserveBalance_*` row, in the
objective penalty and in `process_outputs` (`calc_slacks`):

* timeseries: `reservation` — returned as the very same Param when no
  dynamic / n-1 group exists (byte-identical LP);
* dynamic: `Σ p_flow_upper·unitsize·increase_reserve_ratio` over the flows
  of the dynamic RHS (sink and source side);
* n-1: `Σ p_flow_upper·unitsize·large_failure_ratio` over the failing flows —
  a SUM over failing processes because the V1 n-1 RHS sums them (the .mod
  has one row per failing process; switch to a max if that is ported);
* authored `reservation` is added for any group; methods add up.

`p_flow_upper` is the structural bound (existing + max invest by that
period, per unit size); unlimited invest and `conversion_flow_coeff = 0`
arcs carry `max_flow_for_unconstrained_variables`, so the multiplier stays
finite. Rows with a zero multiplier are dropped (no coefficient).

Activation (unchanged, .mod parity): a reserve's relationships are active
only if some group of the same (reserve, up/down) authors a non-zero
`reservation` (`prundt_from_source`, `_emit_reserve`). A dynamic / n-1-only
reserve without any `reservation` is therefore inactive.

## Follow-ups (not implemented)

1. Upward reserve is not online-aware: an online-capable unit that is off
   can offer upward reserve (`reserve_process_upward` is capacity-based and
   `maxFlow_online` / `maxOutputSum_online` carry no reserve-up term). This
   touches existing fixtures (reserves on `coal_plant`).
2. Increasing reserves on the source side (a consumer offering `down` by
   consuming more) are not bounded by input headroom.
3. Connections: a net-position rule (`down at n ≤ f_into_n + (cap·avail −
   f_out_of_n)`), interacting with `v_flow_back` and the other end.
4. `reserveBalance` lacks the .mod's slope factor for source-side reserves
   of direct units; `_inverse_slope` in `model.py` follows the current
   convention and must change with it (tripwire:
   `test_down_reserve_floor.py::test_direct_unit_input_side_up_reserve`).
5. Connection `delay` is not implemented. Connections are never in
   `process_indirect`, and the delayed input term exists only in
   `conversion_indirect`; the .mod had the same gap (its nodeBalance delay
   term is commented out). The `delay` used to mark the connection delayed
   anyway, which made it `fork_yes`, moved its arcs from
   `process_source_sink_eff` to `_noEff` (efficiency losses dropped) and
   turned a `regular` connection one-way. Now every reader of `delay`
   reads `unit.delay` only (`input_derivation/_specs.py` delay tables,
   `input_derivation/_process_method.py`, `_derived_params`
   `_classify_process_method` / `_delay_distributions_from_source`,
   `_projection_params.process_delayed`), so a connection is modelled as
   if no delay were set, and `_validators.validate_connection_delay` warns
   once per connection with a non-zero delay
   (`tests/engine_polars/test_connection_delay_ignored.py`). Applying a
   connection delay needs a delayed term in the node balance; when it is
   added, drop the filters and the warning.
6. Dynamic and n-1 reserves activate only if some group of the same
   (reserve, up/down) has a non-zero `reservation` (`prundt_from_source`,
   `_emit_reserve`; see "Activation" above). A dynamic / n-1-only reserve
   is silently inactive — e.g. `network_coal_wind_reserve_n_1` does not
   test any n-1 requirement. Activation should follow the group's
   `reserve_method` (dynamic: any `increase_reserve_ratio`; n-1: any
   `large_failure_ratio`), not the presence of a timeseries reservation.
7. The n-1 requirement is one `reserveBalance` row per (r, ud, g, d, t)
   whose RHS sums `large_failure_ratio × flow` over ALL large-failure
   processes of the group; the .mod had one constraint per failing process
   (the largest single failure sets the requirement). The sum over-procures
   reserve whenever two or more processes can fail. When split per failing
   process, the n-1 component of `reserve_shortfall_scale` must become the
   max over the failing processes instead of the sum.
