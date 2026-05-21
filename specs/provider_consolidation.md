# Provider consolidation — one read interface, layered seeding

Companion to `architecture_provider_and_future_duckdb.md` (which is
authoritative for *why*). This spec is the *what* and *how*: the
concrete extensions and migrations that turn today's brittle
multi-carrier arrangement into the destination shape — one
`provider.get(name)` interface, all sophistication living above it in
the orchestrator's seeding step.

Sequenced as commit boundaries; each phase ships independently and is
gated by `tests/engine_polars/...` (~3 min).

---

## Status (2026-05-22)

| Phase | Status | Commits | Notes |
|-------|--------|---------|-------|
| 0a — drop dual-key registration | **DONE** | `0fb1b1ab` | `_emit` registers under one canonical key; consumer-side bidirectional Provider lookup keeps bare-form queries working. |
| 0b — `_provider_keys.py` constants | **DONE** | `88f99608` | Constants for cross-solve set + high-traffic keys.  Migration of single-use literal strings deferred (mechanical, can land in batches). |
| 1a — pilot `invest_periods_of_current_solve` | **DONE** | `d0cea323` | Orchestrator builds the frame via `derive_periods` and `provider.put(K.X, frame)` directly; the `emit_periods` call for this file is gone. |
| 1b — drop empty cumulative-carrier seeds | **DONE** | `9d7fb5f0` | `emit_empty_cumulative_files` deleted entirely.  Consumers' `_read_csv` empty-frame fallback handles first-roll Provider miss naturally. |
| 2.1 — handoff translator | **DONE** | `26d89e3c` | `_provider_translators.translate_handoff_to_provider` + iteration-start call.  Six `K.HANDOFF_*` constants. |
| 2.2 — migrate `_emit_co2_accumulators` | **DONE** | `b9b12e67` | `prior_handoff` dropped from `derive_co2_cum_realized_tonnes` + `emit_co2_rolling_accumulator`. |
| 2.3 — migrate `_emit_chain_params`, drop `prior_handoff` from cascade | **DONE** | `e8ae0bbd` | `_emit_solve_time.run` no longer takes `prior_handoff`.  Preprocessing-cascade depth was just one module (chain_params + co2_accumulators); the original spec audit had over-counted by including post-solve sites. |
| 3a — delete `capture_post_solve` | **DONE** | `90a932b0` | Dead code (no live callers) + 3 orphan `SolveHandoff` fields (`fix_storage_timesteps`, `ed_history_realized_first`, `edd_history`) + their tests. `build_handoff_from_flexpy` is the canonical post-solve constructor. |
| 3b — drop `.csv` from key constants | **DONE** | `7454ae74` | Purely documentary; Provider strips `.csv` at lookup.  Per user note: the suffix confused agents into thinking keys correspond to files. |
| 4 — retire `CROSS_SOLVE_KEYS` + drop dual-form lookup | **DONE** | `aab8132c`–`30efa968` | Landed as Phase 4.1 (fan-out → handoff carriers; `_fan_out_*` deletions) and 4.2 (`CROSS_SOLVE_KEYS` plumbing removed, bare↔qualified fallback dropped).  LayeredProvider did not land — the single mutable `FlexDataProvider` with translator-built handoff entries proved sufficient. |
| 5a — override translator infrastructure | **DONE** | `df6c4274` | Ten `K.OVERRIDE_*` constants; `translate_overrides_to_provider` writes `override/<field>` keys and raises `ValueError` on unknown keys; `read_handoff_frame` checks override slot first, falls back to handoff. |
| 5b — wire override translator into orchestrator | **DONE** | `f5224217` | Third translator call in `native_run_model` iteration setup; `state.override_provider: Callable[[], dict] \| None` is the orchestrator hook; INFO log on apply, DEBUG lists keys. |
| 5c — end-to-end override test | **DONE** | `4ef17539` | `tests/engine_polars/test_override_provider.py` asserts an override of `K.HANDOFF_REALIZED_INVEST` reaches `p_entity_previously_invested_capacity`; `run_chain_from_db` gains `override_provider=` kwarg threaded onto `RunnerState`. |
| 5d — document precedence + transport | **DONE** | (this commit) | External Overrides section in `architecture_provider_and_future_duckdb.md`; spec status table + Phase 5 narrative reflect what shipped. |
| 6a — source tag on `provider.put` | **DONE** | `64d8697f` | `provider.put(key, frame, *, source=None)` records a free-form tag; `provider.get_source(key)` reads it.  Override translator tags writes with `source="external_override"`. |
| 6b — env-gated audit dump | **DONE** | (this commit) | `dump_provider_sources(provider, path, solve_name)` in `_provider_translators.py`; orchestrator calls it at end of per-iteration preprocessing when `FLEXTOOL_AUDIT_SOURCES=1`.  Log path `<work_folder>/audit_sources.log`; tab-separated `solve_name\tkey\tsource` per line, append mode. |

The dead-key cleanup that preceded this spec landed in `752dff3f`
(`rp_base/rp_block` writers) and `3c51404d` (`p_years_represented_d_calc`
writer) — these were the starting point for the broader effort.

### What changed vs. the original plan

* **Phase 2's scope shrank.** The original spec audit counted 78
  `prior_handoff` references in `engine_polars/` and listed many
  consumer sites in `input.py` / `_orchestration.py`.  Closer
  inspection: those sites were all inside **post-solve** handoff
  construction (`build_handoff_from_flexpy`,
  `write_outputs_for_solve`) — not preprocessing.  The cascade
  preprocessing path threaded `prior_handoff` through exactly two
  modules (`_emit_chain_params`, `_emit_co2_accumulators`); migrating
  those was the entire scope.
* **Phase 3's scope shrank.** The original spec described rewriting
  `capture_post_solve` to build from solver output.  Reality:
  `capture_post_solve` was already dead code — the cascade switched to
  `build_handoff_from_flexpy` long ago.  Phase 3 became deletion, not
  rewriting.
* **Phase 3b (`.csv` suffix retirement) was added late** in response
  to the user's observation that the suffix in key constants
  reinforces a misleading "this is a file" mental model.  Now folded
  into the spec.
* **Three `SolveHandoff` fields retired.**  `fix_storage_timesteps`,
  `ed_history_realized_first`, and `edd_history` were populated only
  by the deleted `capture_post_solve` and had Provider-fallback
  consumers (or no consumer at all).  Dataclass shrinks from 11
  fields to 8.
* **Phase 4 dropped LayeredProvider.**  The original plan called for a
  typed `LayeredProvider` with an explicit `parents` chain.  Reality:
  once Phase 4.1 routed every cross-solve carrier through
  `translate_handoff_to_provider`, the orchestrator no longer needed an
  iteration-end extraction loop or a `CROSS_SOLVE_KEYS` tuple at all —
  the single mutable `FlexDataProvider` populated by the translator at
  iteration start covered every cross-solve path.  Phase 4.2 deleted
  `CROSS_SOLVE_KEYS` plumbing (`4cdb39a9`) and dropped the
  bare↔qualified `_split_qualified` fallback in
  `FlexDataProvider.get` (`30efa968`).  No layer abstraction landed
  because no layer abstraction was needed.
* **Phase 5 followed Phase 4's shape, not the original plan.**  With
  `LayeredProvider` gone, "external overrides as a layer" became
  "external overrides as a third translator call writing into a
  parallel `override/*` Provider key namespace, with
  `read_handoff_frame` enforcing the precedence at read time."  This is
  simpler than the original layer chain and has the same precedence
  semantics.  See Phase 5 narrative below.

---

## TL;DR

Today (post-Phase-5): the preprocessing cascade has one read interface
(`provider.get(K.X)`); the orchestrator translates cross-solve state
into Provider entries at iteration start via
`_provider_translators.translate_handoff_to_provider`; the post-solve
handoff is constructed directly from the flexpy `Solution` via
`build_handoff_from_flexpy` (no disk round-trip); Provider key
constants live in `_provider_keys.py` without `.csv` suffix; the
hand-maintained `CROSS_SOLVE_KEYS` extraction loop is gone (Phase 4);
external overrides reach preprocessing through a third translator
writing to a parallel `override/*` namespace, with `read_handoff_frame`
enforcing `override > handoff > base` precedence at read time (Phase 5).

Remaining (Phase 6): opt-in source-tagging for write-time provenance,
with override entries tagged `external_override:<id>` so the optional
end-of-preprocessing audit dump distinguishes them from natural handoff
carriers.

---

## Destination shape

```
Pre-sub-solve (orchestrator, _native_run_model.native_run_model):
  sub_solve_provider = FlexDataProvider()           # single mutable Provider
  seed from cascade_input_provider                  # static base
  translate_handoff_to_provider(prior_handoff, sub_solve_provider)
                                                    # writes handoff/<field> keys
  if state.override_provider is not None:           # Phase 5
      overrides = state.override_provider()
      translate_overrides_to_provider(overrides, sub_solve_provider)
                                                    # writes override/<field> keys
  preprocessing.solve_time.run(provider=sub_solve_provider)
                                                    # fills solve_data/* keys
  cascade reads via provider.get(K.<NAME>)          # one interface
  handoff-class reads route through read_handoff_frame, which checks
    override/<field> first, falls back to handoff/<field>.

Solver runs.

Post-sub-solve:
  build_handoff_from_flexpy(solution) constructs the next SolveHandoff
  next iteration's translate_handoff_to_provider consumes it.
```

The reusable building blocks that landed:

1. **`flextool/engine_polars/_provider_keys.py`** — Python constants
   for every Provider key. Imported by both producers (`provider.put(K.P_FLOW_MAX, df)`)
   and consumers (`provider.get(K.P_FLOW_MAX)`).  Includes the ten
   `K.HANDOFF_*` carriers and the parallel ten `K.OVERRIDE_*` slots.
2. **`flextool/engine_polars/_flex_data_provider.FlexDataProvider`** —
   single mutable in-memory Provider.  Post-Phase 4.2 the
   `_split_qualified` bare↔qualified fallback in `get` is gone;
   consumers all use qualified keys.
3. **`flextool/engine_polars/_provider_translators.py`** — pure
   translators that write into a `FlexDataProvider`:
   `translate_handoff_to_provider(handoff, provider)` populates
   `handoff/*`, `translate_overrides_to_provider(overrides, provider)`
   populates `override/*`, `read_handoff_frame(provider, key)` is the
   consumer-side helper enforcing the override→handoff precedence.

---

## Implementation plan

### Phases 0–3 — landed (see Status table above)

The four landed phases produced:

* `flextool/engine_polars/_provider_keys.py` — named constants for the
  cross-solve set, handoff carriers, and other high-traffic keys.
  ``.csv`` suffix retired from constant values.
* `flextool/engine_polars/_provider_translators.py` —
  ``translate_handoff_to_provider(handoff, provider)`` writes each
  consumed ``SolveHandoff`` field to a dedicated ``handoff/<field>``
  Provider key at iteration start.  ``read_handoff_frame(provider,
  key)`` is the consumer-side helper (returns ``None`` for empty
  handoff frames so existing ``if x is not None`` guards keep
  working).
* `_emit` registers under one canonical key (parent/basename, no
  ``.csv``).  ``_native_run_model._CROSS_SOLVE_KEYS`` (renamed from
  ``_CROSS_SOLVE_BASENAMES``) imports the tuple from
  ``_provider_keys.CROSS_SOLVE_KEYS``.
* Preprocessing cascade no longer threads ``prior_handoff``;
  ``_emit_solve_time.run`` signature is ``(state, solve_name, *,
  provider)``.
* ``capture_post_solve`` and its three orphan ``SolveHandoff`` fields
  are deleted; ``build_handoff_from_flexpy`` is the canonical
  post-solve constructor.

**Incremental follow-ups not blocking Phase 4** (mechanical, can land
anytime in batches):

* Many call sites still use literal ``"solve_data/foo.csv"`` strings
  rather than ``K.SOLVE_DATA_FOO``.  Both resolve to the same Provider
  key (the Provider's ``_strip_csv`` collapses them) — the literal
  form just doesn't get IDE rename / Find-Usages.  Migrate as the
  surrounding code is touched.
* ``_emit_provider_io._emit_path`` docstring still mentions "dual-key
  registration"; harmless but stale (the helper does single-key
  registration since Phase 0a).  Trim when next editing that file.

---

### Phase 4 — retire `CROSS_SOLVE_KEYS` (landed)

**What shipped (`aab8132c`–`30efa968`):**

* Phase 4.1 migrated every cross-solve carrier off
  ``state.cross_solve_carriers`` and onto
  ``translate_handoff_to_provider``-written ``handoff/<field>`` Provider
  keys (`fix_storage_quantity/price/usage`, `roll_end_state`, ladder
  accumulators, parent-overlay reads).  The ``_fan_out_*`` helpers were
  deleted (`d6fd636d`, prior commits) once their consumers read from
  the handoff translator instead.
* Phase 4.2 deleted the ``CROSS_SOLVE_KEYS`` plumbing entirely
  (`4cdb39a9`): the iteration-end extraction loop, the
  hand-maintained tuple in ``_provider_keys.py``, and
  ``state.cross_solve_carriers`` itself.  The previous-iteration
  carrier set now lives exclusively in the post-solve ``SolveHandoff``
  and is reapplied via ``translate_handoff_to_provider`` at the next
  iteration start.
* Phase 4.2 also dropped the ``_split_qualified`` bare↔qualified
  fallback in ``FlexDataProvider.get`` (`30efa968`).  All consumers use
  qualified keys.

**What did not ship — and why:** the original plan proposed a typed
``LayeredProvider`` with ``ProviderLayer`` parents.  After Phase 4.1
routed every cross-solve carrier through the handoff translator, no
caller needed the layered semantics — the single mutable
``FlexDataProvider``, populated once per iteration by the translator,
covered every case.  ``LayeredProvider`` was dropped as YAGNI; the
precedence semantics it would have enforced are instead enforced at
read time by ``read_handoff_frame`` (Phase 5).

---

### Phase 5 — External overrides through a parallel namespace (landed)

**What shipped (`df6c4274`, `f5224217`, `4ef17539`):**

* **Phase 5a (`df6c4274`).** Ten ``K.OVERRIDE_*`` constants parallel
  the ten ``K.HANDOFF_*`` carriers in ``_provider_keys.py``.
  ``translate_overrides_to_provider(overrides, provider)`` writes each
  frame under ``override/<field>``; the translator raises
  ``ValueError`` on any key not in the whitelist
  (``K.HANDOFF_REALIZED_INVEST``, ``HANDOFF_REALIZED_EXISTING``,
  ``HANDOFF_DIVEST_CUMULATIVE``, ``HANDOFF_ROLL_END_STATE``,
  ``HANDOFF_CUMULATIVE_CO2``, ``HANDOFF_CUMULATIVE_COMMODITY``,
  ``HANDOFF_CUM_SIM_HOURS``, ``HANDOFF_FIX_STORAGE_QUANTITY``,
  ``HANDOFF_FIX_STORAGE_PRICE``, ``HANDOFF_FIX_STORAGE_USAGE``).
  ``read_handoff_frame(provider, key)`` checks ``override/<field>``
  first, falls back to ``handoff/<field>``, and collapses
  ``height == 0`` frames to ``None`` in either slot so consumers keep
  their ``if frame is not None`` guard pattern.

* **Phase 5b (`f5224217`).** Third translator call in
  ``_native_run_model.native_run_model`` iteration setup, after the
  handoff translator: if ``state.override_provider`` is set, invoke it
  and pass the returned dict to ``translate_overrides_to_provider``.
  ``state.override_provider`` is typed
  ``Callable[[], dict[str, pl.DataFrame]] | None`` and defaults to
  ``None``.  When the call returns a non-empty dict, the orchestrator
  logs ``[override] applied N keys at iter=<i> solve=<solve_name>`` at
  INFO and the sorted key list at DEBUG.

* **Phase 5c (`4ef17539`).** End-to-end test
  ``tests/engine_polars/test_override_provider.py``: an override of
  ``K.HANDOFF_REALIZED_INVEST`` carrying a scaled copy of solve 1's
  realized_invest reaches solve 2's preprocessing and shows up
  multiplied in ``p_entity_previously_invested_capacity``.
  ``run_chain_from_db`` gained an ``override_provider=`` kwarg threaded
  onto the engine_polars ``RunnerState``; ``_drive_cascade`` forwards
  it onto ``runner.state`` so the per-sub-solve hook in
  ``_native_run_model`` picks it up.

* **Phase 5d (this commit).** Architecture spec
  (`architecture_provider_and_future_duckdb.md`) gains an
  ``External overrides`` section covering precedence chain
  (``override > handoff > base``), Python-callable transport, the
  whitelist, the logging surface, and a forward-looking note about
  Phase 6 source-tagging.

**Precedence (enforced by ``read_handoff_frame``):**
``override/<field>`` shadows ``handoff/<field>`` shadows the
cascade_input_provider base.  Preprocessing-emitted ``solve_data/*``
keys are independent; they do not pass through ``read_handoff_frame``.

**Transport.** A Python callable on ``state.override_provider`` is the
contract.  Future transports (file watch, ZeroMQ subscriber, REST
callback) wrap that callable; the orchestrator never sees anything
other than a function returning a dict.

**Whitelist.** Only ``K.HANDOFF_*`` keys are overridable.  Unknown keys
raise ``ValueError`` at translator time.  This keeps the override
surface explicit and grep-able and prevents external code from
trampling preprocessing-emitted state.

**Out of scope for Phase 5:** the external program itself, any
non-Python transport, the operator UI.  Those wrap the existing
callable contract when they land.

**Verification:** end-to-end test simulating an external override of
`p_node_state_initial` for one node/period; assert the solver sees
the overridden value and the audit log records the application.

**Risk:** low once Phase 4 is in. The mechanism is "one more layer";
the work is in defining the transport contract and the audit format.

**Out of scope:** the external program itself, the transport protocol,
the override-permitted-fields whitelist (decided when the external
program's design lands).

---

### Phase 6 — Source-tagging (opt-in provenance, landed)

**Goal:** make non-default writes traceable end-to-end.

**Phase 6a (`64d8697f`).** `provider.put(name, frame, *, source=None)`
records a free-form tag alongside the frame; `provider.get_source(name)`
surfaces it.  Entries with `source=None` (the natural-cascade default)
leave no record so memory overhead stays zero.  Eviction by
`release_unused` clears the matching source entry alongside the frame.
The override translator (Phase 5) passes `source="external_override"`
for every override write; all other callers leave the kwarg unset.

**Phase 6b (this commit).** `dump_provider_sources(provider, path,
solve_name)` in `_provider_translators.py` iterates Provider keys (via
`provider.keys()`) and appends one tab-separated line per
source-tagged key: `<solve_name>\t<key>\t<source>\n`.  The orchestrator
invokes the helper at the end of per-iteration preprocessing — after
the override translator + per-iter emits, before `solver.run` — when
`FLEXTOOL_AUDIT_SOURCES=1` is set in the environment.  The log path is
`<work_folder>/audit_sources.log`; the file is opened in append mode
so multiple sub-solves accumulate.  Sorted iteration keeps the line
order deterministic.

**Verification:** `tests/engine_polars/test_override_provider.py::test_audit_sources_dump_records_external_override`
sets the env var + an override provider, runs the cascade, and asserts
the override key appears in the log tagged `external_override`.

**Risk:** trivial.  Additive — env var off by default, no behaviour
change for production runs.

---

## Verification gate

The targeted gate (commit-readiness signal, ~3 min) from
`writer_to_emitter.md §2.1` applies to every phase:

```bash
python -m pytest tests/engine_polars/test_region_filter.py \
  tests/engine_polars/test_lagrangian.py \
  tests/engine_polars/test_cli_engine_dispatch.py \
  tests/engine_polars/test_cost_aggregation_semantics.py \
  tests/engine_polars/test_dump_csvs_roundtrip.py \
  tests/engine_polars/test_orchestration_parity.py \
  tests/engine_polars/test_override_chain_step3_parity.py \
  tests/engine_polars/test_fast_single_solve.py \
  tests/engine_polars/test_pbt_node_inflow.py \
  tests/engine_polars/test_rivendell_bug1_5_6_user_bound_scale.py \
  tests/engine_polars/test_rivendell_bug3_s07_co2_price.py \
  tests/engine_polars/test_output_writer.py \
  tests/engine_polars/test_phase_e_g_multi_roll_parity.py \
  tests/engine_polars/scaling/ tests/engine_polars/loaders/ \
  tests/engine_polars/synthetic/ tests/engine_polars/emission/ \
  tests/engine_polars/perturbation/ tests/engine_polars/constraints/ \
  --tb=line -q -p no:cacheprovider --continue-on-collection-errors
```

Phase 1 and Phase 4 additionally require the rolling-handoff tests:

```bash
python -m pytest tests/test_commodity_ladder_rolling.py \
  tests/test_cumulative_handoffs.py \
  tests/engine_polars/test_chain_cumulative_handoffs.py \
  --tb=line -q -p no:cacheprovider
```

---

## Deferred design

**Form B typed keys with bound `pl.Schema` and put-time validation.**
Catches schema drift at the writer rather than at a downstream
consumer. Costs runtime work; gate behind a debug flag. Layer in if
schema drift becomes a recurring bug source after Phase 0 ships.

**Warm-update diff support — `provider.diff(other_provider)`.**
Returns the changed (key, rows) set between two Providers. Used by the
polar_high.Problem coefficient-update path when that API lands.
Speculative now; the iteration-by-iteration translator-write pattern
gives us the building block (retain roll-N's Provider snapshot, build
roll-N+1's, diff against the corresponding entries). Concrete API
shape depends on what polar_high.Problem accepts; design alongside the
warm-update feature.

**DuckDB-backed Provider.** Out of scope per
`architecture_provider_and_future_duckdb.md §"When the DuckDB question
gets answered"`. The constants module translates naturally — `K.X` is
a SQL view/table name in the DuckDB case; the override/handoff
precedence collapses to a `COALESCE` chain across precedence-ordered
views.  Worth verifying with a thought-experiment when DuckDB
prototype starts, but no design accommodation needed in Phase 0-6.

---

## Phasing recommendation

Status as of 2026-05-22:

| Phase | Effort | Risk | Dependencies | Status |
|-------|--------|------|--------------|--------|
| 0 — constants module + dual-key cleanup | small | low | none | DONE |
| 1 — pilot two small-data files | small + medium | low–medium | Phase 0 | DONE |
| 2 — eliminate `prior_handoff` threading | medium (shrunk to small) | medium | Phases 0–1 | DONE |
| 3 — delete `capture_post_solve` + `.csv` retirement | small (shrunk; was rewrite) | low | independent | DONE |
| 4 — retire `CROSS_SOLVE_KEYS` (LayeredProvider dropped) | medium–high (shipped without LayeredProvider) | medium | Phases 0–3 | DONE |
| 5 — external overrides (override/* namespace, not a layer) | small | low | Phase 4 | DONE |
| 6 — source-tagging (put-side tag + env-gated audit dump) | trivial | trivial | Phase 5 (or independent) | DONE |

---

## Tester-facing changelog (cumulative through Phase 5)

One paragraph for release notes covering everything landed so far:

> Internal refactor of the per-sub-solve preprocessing pipeline.
> Provider keys are now imported as Python constants from
> `flextool/engine_polars/_provider_keys.py` (no behaviour change at
> runtime).  The dual-key registration (``foo.csv`` and
> ``solve_data/foo.csv``) was retired in favour of a single canonical
> ``solve_data/foo`` form.  Cross-solve state is fanned out into
> ``handoff/<field>`` Provider keys at iteration start; the cascade no
> longer takes a ``prior_handoff`` parameter, and the
> ``CROSS_SOLVE_KEYS`` extraction loop / fan-out helpers are gone.  The
> dead disk-reading ``capture_post_solve`` constructor was deleted along
> with three ``SolveHandoff`` fields that only it populated.  External
> programs may now inject overrides for the ten handoff carriers via
> ``state.override_provider`` (a Python callable returning a dict);
> overrides land in a parallel ``override/<field>`` Provider namespace
> and shadow the natural handoff carriers at read time.  Report any
> ``KeyError`` / ``no provider entry for ...`` traceback — it indicates
> a missed call-site migration.

---

## Notes for the implementing agent

- **No destructive git** (`rebase`, `reset --hard`, `checkout -- .`,
  `clean -f`, `branch -D`, force-push). No hook-skipping (`--no-verify`).
- **No tweaking tests / goldens / tolerances** to make a failure go
  away. Tests are parity oracles; if they diverge, fix the producer.
  **Exception**: tests that assert behaviour explicitly retired by
  this spec (e.g. dual-key registration, ``capture_post_solve``)
  should be replaced with an inverse-regression guard OR deleted
  outright — see the Phase 0a / Phase 3a commits for precedent and
  the user's `feedback_keep_diagnostic_tests` memo for the tension.
- **No invented helpers / duplicated helpers** — grep the substrate
  first.  Phase 6 in particular: any audit-dump format should reuse
  existing logging infrastructure rather than introduce a parallel
  serialization layer.
- **Surface blockers**, do not work around them.
- **Per-phase commit boundaries** matter. Each phase is one or a few
  commits with the targeted gate green. Don't bundle phases; the goal
  is bisectable history.  Where a phase splits into sub-commits
  naturally (Phase 0 = 0a + 0b; Phase 1 = 1a + 1b; Phase 2 = 2.1–2.3;
  Phase 3 = 3a + 3b; Phase 4 = 4.1a–l + 4.2-{0..2}; Phase 5 = 5a + 5b
  + 5c + 5d), commit each sub-phase separately.
- **Sequential subagents only.** Per
  `feedback_sequential_agents.md` — never dispatch agents in parallel
  in this repo.
- **Targeted tests only between commits.** Run the full sweep at
  most once per spec-phase boundary (after a clean commit lands),
  not after every edit.  Per `feedback_targeted_tests.md`.
