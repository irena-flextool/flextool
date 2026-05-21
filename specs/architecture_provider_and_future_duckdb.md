# Architecture: FlexDataProvider and the future DuckDB switch

Strategic context document. Read this **alongside** `flex_data_provider_migration_handoff.md` — the operational handoff is the work; this document is why that work has the shape it does, and what comes after.

---

## TL;DR

The destination architecture has one input interface (`FlexDataProvider.get(name)`) and one solver (HiGHS, via polar_high). Loaders, writers, and post-processing all consume the Provider. The Provider's implementation is replaceable: in-memory polars today, possibly DuckDB later. The LP builder feeds HiGHS one item-group at a time so peak memory is bounded by the HiGHS matrix itself.

This document explains:
1. Why this shape, given where we've been.
2. What the Provider interface buys us.
3. The future DuckDB question — when it might be answered and what would tip the decision.
4. The deliberately-separate Tier-1 SpineDB replacement (DB editor rewrite).
5. What this architecture deliberately does NOT do.

---

## How we got here

The engine_polars work over the past several weeks went through Phases A through E-g. The arc was:
- **Phases A-B**: Lift writers out of the monolith into 17 dedicated modules.
- **Phase C**: Introduce a per-sub-solve `FlexDataAccumulator` that captures writer frames in memory.
- **Phase D**: Wire `load_flextool(seed=accum)` so the loader can read from the accumulator.
- **Phase E-c**: Add `csv_emission_disabled()` context manager; cascade default flips to in-memory.
- **Phases E-d/e**: Migrate writer-side `csv.reader` sites to consult the in-memory seed first.
- **Phase E-f**: Migrate ~60 loader-side `path.exists()` checks to be seed-aware.
- **Phase E-g**: Install the seed around `build_handoff_from_flexpy` (the last cross-boundary call into path-based code).
- **Phases F-H**: Parquet bundle manifest, process_outputs in-memory kwargs, e2e verification.

After Phase E-g, the cascade runs end-to-end in-memory with byte-parity against v3.32.0 goldens on multi-roll fixtures. **What it does NOT have is a clean interface.** The data flow goes through a path-based shim (`_seed_lookup`, `_seed_or_exists`, etc.) that intercepts `path.exists()` calls and serves them from memory. This works but has known costs:

1. **Bug class**: Every call site that crosses from `_drive_cascade` into code that takes `Path` arguments needs the seed installed in scope. Phase E-g existed because one such site was missed. Future call sites carry the same risk.
2. **Cognitive overhead**: Three sets of helpers (`_seed_lookup`, `_seed_or_exists`, `_seed_open`), an `_active_seed` global, install/restore try/finally boilerplate. Onboarding cost is real.
3. **Locks in path-based loaders**: As long as the interface is "I take a Path and read it," changing the data source means another shim layer.

The Provider migration retires this scaffolding and replaces it with a single explicit interface.

---

## Why a Provider interface is the right destination

Three properties matter:

### 1. It bounds memory in a measurable way

Today, `load_flextool` materialises FlexData eagerly: every preprocessing artefact becomes a polars frame in RAM at the start of the sub-solve, and stays there until the sub-solve ends. Peak memory = sum of all FlexData frames + HiGHS matrix.

Under the Provider with item-group streaming (Step 3 of the handoff), peak memory = HiGHS matrix + one item-group's frames + the Provider's small per-name cache. This is the smallest possible bound: HiGHS itself defines the floor (it has to hold the LP), and the Provider holds essentially nothing extra.

**Memory goal stated by the user**: preprocessing memory ≤ HiGHS memory. The Provider with eviction is the only way to make this true by construction. Any other approach (lazy polars chains, careful frame dropping in loader code, etc.) is best-effort and will regress over time.

### 2. It decouples data source from data consumer

Today, the loader code knows where its data lives — it constructs Paths against `work_folder/solve_data/`, calls `path.exists()`, reads parquet/CSV. The seed funnel is a band-aid because the source DID change (memory replaced disk for the preprocessing layer) but the interface didn't.

Under the Provider, the loader doesn't know where data lives. It says `provider.get("p_flow_max")` and gets a frame. The Provider knows whether that frame comes from:
- an in-memory dict (current)
- a parquet file in `output_raw/` (process_outputs use case)
- a DuckDB query (possible future)
- a SQL view over a remote database (hypothetical, but not impossible)

Crucially, **the loader code doesn't change when the source changes.** That's the property that keeps the future DuckDB switch optional. We can prototype a DuckDB-backed Provider without touching one line of loader code, benchmark, and decide.

### 3. It eliminates an entire bug class

The seed-funnel-class bug is: "control flow crossed into path-based code, the seed wasn't installed in this scope, the path-based code fell through to disk, disk was empty, data was silently missing, results were wrong." Phase E-g was one instance. There will be more as long as the funnel exists.

Once loaders take a Provider explicitly (passed as an argument), there is no scope. The Provider is either passed or it isn't. If it isn't, you get a `TypeError` immediately, not a silent numeric divergence three rolls later.

---

## The Provider interface contract

```python
class FlexDataProvider(Protocol):
    # Step 1
    def get(self, name: str) -> pl.DataFrame | None: ...
    def has(self, name: str) -> bool: ...
    def put(self, name: str, frame: pl.DataFrame) -> None: ...

    # Step 2
    def snapshot_raw_inputs(self, work_folder: Path) -> None: ...
    def snapshot_processed_inputs(self, work_folder: Path) -> None: ...

    # Step 3
    def get_for_group(self, name: str, group: ItemGroup) -> pl.DataFrame | None: ...
    def release_unused(self, after: ItemGroup) -> None: ...
    def precompute_lifetimes(self, lp_topology: LpTopology) -> None: ...
```

That's the entire contract. Anything else is implementation detail.

This contract is **deliberately minimal**. It does not include:
- Schema introspection (`get_columns`, `get_dtypes`) — consumers should know what they're asking for
- Async / future-based APIs — preprocessing is synchronous, don't add complexity
- Change notifications / observability hooks — not needed
- Multi-version / multi-scenario support — the Provider is per-sub-solve
- A query language — `get(name)` is sufficient; complex selections happen in loader code

If a future need arises that justifies extending this contract, that's a deliberate decision — but every extension is a tax on every Provider implementation. Keep it small.

---

## When the DuckDB question gets answered

After Step 3 of the handoff lands, the Provider interface is stable and the cascade has a measured memory profile. At that point the question becomes empirical:

> Does an in-memory polars Provider hit the memory ceiling on scenarios the user actually runs, or doesn't it?

The decision tree:

```
                        Provider+streaming stable
                                 │
                                 ▼
              ┌─── Memory OK on real workloads? ───┐
              │                                    │
             Yes                                  No
              │                                    │
        Stay in-memory.              ┌── Spill-to-disk acceptable on user environments?
        DuckDB never                 │   (measure on slow disk / network mount)
        becomes urgent.              │
                                    Yes              No
                                     │                │
                              Prototype           Re-examine
                              DuckDBProvider.     in-memory:
                              Benchmark on        lazy materialisation,
                              fast and slow       per-frame eviction,
                              disk. Decide.       row-level eviction.
                                                  DuckDB still on table
                                                  but not the obvious answer.
```

**The Provider interface keeps both branches open with zero cost.** That's the whole point of doing the interface refactor first. Whatever the answer ends up being — in-memory wins, DuckDB wins, or "depends on user environment" with a runtime toggle — the LP builder, loader code, and post-processing don't care.

### What a DuckDBProvider would look like

If the decision goes that way, the implementation is roughly:

```python
class DuckDBProvider:
    def __init__(self, conn: duckdb.DuckDBPyConnection, schema: dict[str, str]):
        self._conn = conn
        # schema maps frame name → SQL view / table name in the DB

    def get(self, name: str) -> pl.DataFrame | None:
        sql = self._schema.get(name)
        if sql is None: return None
        return self._conn.execute(sql).pl()

    def get_for_group(self, name: str, group: ItemGroup) -> pl.DataFrame | None:
        sql = self._schema.get(name)
        if sql is None: return None
        # apply group predicate as a parameterised query
        return self._conn.execute(
            f"SELECT * FROM ({sql}) WHERE entity_class = ? AND period = ?",
            [group.entity_class, group.period],
        ).pl()

    def release_unused(self, after: ItemGroup) -> None:
        # No-op — DuckDB manages its own buffer pool.
        pass
```

This is illustrative, not specified. The actual prototype happens after Step 3, with real benchmarks driving the design.

### What would tip the decision toward DuckDB

- Memory pressure on real user scenarios that can't be fixed with eviction granularity.
- A clear use case for SQL queries against preprocessing intermediates beyond what loaders need (e.g. an external tool that wants to inspect FlexData).
- Concrete benchmarks showing DuckDB streaming + spill is faster than polars + OOM-prevention manual chunking at the scales users hit.

### What would tip the decision against DuckDB

- The in-memory Provider with streaming hits the memory target on all observed scenarios.
- Benchmarks show DuckDB's per-query overhead dominates on small frames (which most preprocessing artefacts are).
- The schema-migration burden of DuckDB outweighs the wins.
- Slow disks / antivirus on common user environments make spill prohibitively expensive in practice.

The honest version: we don't know which way this goes until we measure. The Provider interface is the option-preserving move.

---

## The separately-tracked Tier-1: SpineDB → DuckDB

This is a different piece of work. It is **not** part of the engine refactor and is **not** in scope for the FlexDataProvider migration.

### Motivation

SpineDB uses EAV (Entity-Attribute-Value) storage. For FlexTool's data — dimensional facts with strong schema — EAV is fundamentally slower than relational columnar storage. The slowness manifests as:
- GUI editing latency on large databases (the DB editor pulls every parameter as a join across class/object/parameter/value tables).
- Long `load_flextool` startup time as it walks the EAV graph.
- Memory bloat from EAV row overhead.

The mid-term plan is to rebuild the input layer as full relational tables in DuckDB, with parquet backing for large time-series tables. The DB editor UI gets rewritten to match.

### Why this is separate from the Provider work

- Different scope: this is a user-facing feature change (new DB editor UI, new import/export, migration story for existing user projects), not an internal refactor.
- Different timeline: probably 6+ months of work including UI rewrite.
- Different risk profile: directly touches user data and workflows.

### How it fits with the Provider

When Tier-1 lands, it becomes another possible Provider source — `provider.get("p_inflow")` could query the Tier-1 DuckDB directly instead of going through the current preprocessing-writers chain. But that integration is decided when Tier-1 ships; it doesn't shape the Provider design today.

The Provider should NOT pre-empt this:
- Don't add Spine-specific methods to the Provider.
- Don't add DuckDB-specific methods.
- Don't model "scenario alternatives" in the Provider interface — that lives in whatever fills the data source role.

### Concretely, the architecture once Tier-1 lands

```
[ User-edited Tier-1 DuckDB ]   (with parquet for time series)
                │
                │  preprocessing reads, derives, writes to Provider
                ▼
        ┌───────────────────────┐
        │ FlexDataProvider      │  ◄── same interface as today
        │                       │
        │ Implementation:       │
        │  in-memory polars     │  (or DuckDB if the Tier-2-style
        │  by default           │   decision went that way)
        └───────────────────────┘
                │
                │  one item-group at a time
                ▼
            polar_high → HiGHS
```

Tier-1 replaces the source (Spine → DuckDB). The Provider stays put. The cascade doesn't notice.

---

## External overrides

External programs (an outer optimisation loop, a regression test driver, an
operator dashboard) sometimes need to inject values into the cascade that
would otherwise come from the prior sub-solve's handoff. Phase 5 of
`provider_consolidation.md` landed the mechanism; the design choices are:

**Precedence chain.** Within a sub-solve Provider, reads of a handoff-class
key resolve in the order `override/<field> > handoff/<field> >
cascade_input_provider`. The override layer shadows the natural handoff
carrier, which in turn shadows the static cascade base. `read_handoff_frame`
in `_provider_translators.py` is the single chokepoint that enforces this
ordering: it checks the `override/*` slot first, falls back to `handoff/*`,
and collapses `height == 0` frames to `None` in either layer so consumers
keep their `if frame is not None` guard pattern.

**Transport.** A Python callable lives on `state.override_provider:
Callable[[], dict[str, pl.DataFrame]] | None`. The orchestrator invokes it
at iteration start, after sequential and parent handoff translation, and
hands the returned dict to `translate_overrides_to_provider`. Future
transports (file watcher, ZeroMQ subscriber, REST callback) wrap that
callable rather than replacing the contract — the orchestrator only ever
sees a Python function returning a dict.

**Whitelist.** External overrides may target only the existing handoff
carriers — the ten `K.HANDOFF_*` constants in `_provider_keys.py`. The
translator raises `ValueError` on unknown keys; arbitrary Provider-key
writes through the override path are rejected by design so the override
surface stays explicit and grep-able.

**Logging.** When the callable returns a non-empty dict, the orchestrator
emits an INFO record `[override] applied N keys at iter=<i> solve=<name>`
followed by a DEBUG line listing the keys. This is the audit signal users
follow to confirm an override reached the iteration they expected.

**Source-tagging audit dump (Phase 6).** Override entries are tagged with
`source="external_override"` at write time (Phase 6a — `provider.put`
accepts an optional `source` kwarg; `provider.get_source(key)` surfaces it).
When the environment variable `FLEXTOOL_AUDIT_SOURCES=1` is set, the
orchestrator dumps every source-tagged Provider key at the end of each
sub-solve's preprocessing (Phase 6b) to `<work_folder>/audit_sources.log`.
The format is tab-separated lines `<solve_name>\t<key>\t<source>` and the
file is opened in append mode so a multi-roll cascade accumulates one
record per overridden key per sub-solve.  Natural cascade writes leave
`source=None` and are skipped, so the log is a minimal audit trail of
externally-injected entries — not a Provider snapshot.  This is additive
and does not change the precedence rules above.

---

## What this architecture deliberately does NOT do

These are choices, not omissions. Pushing the architecture to do these things would be regression toward earlier rejected designs.

### Not: a two-tier preprocessing pipeline

An earlier proposal had Tier-1 (scenario-input DuckDB) and Tier-2 (processed-FlexData DuckDB) as two separate persistence layers. Rejected because the Tier-2 layer adds disk I/O that the in-memory cascade already removed, and the change-tracking machinery to make Tier-2 worth its complexity (incremental re-derive on parameter edits) is far larger than the win.

If a user-facing "edit one parameter, only re-derive what changed" feature is wanted, that's a separate design problem solved by a dependency-tracking layer above the Provider — not by adding a second persistence tier.

### Not: change tracking inside the Provider

The Provider has no concept of "what changed since last time." Every sub-solve gets a fresh Provider populated from scratch. Incremental update is out of scope for this layer. If wanted, build it above the Provider (e.g. cache invalidation logic in the cascade orchestrator), not inside it.

### Not: a SQL query layer for loaders

The Provider's `get(name)` is a name lookup, not a query. Loaders that want to filter / join / aggregate do that in polars expressions on the returned frame. This is deliberate — it keeps the Provider interface trivially mockable, makes loaders testable in isolation, and avoids the "every loader is a SQL string" maintenance trap.

The DuckDB-backed Provider (if it happens) translates `get(name)` to a `SELECT * FROM <view>` and returns a polars frame. The loader doesn't know SQL was involved.

### Not: parallelism

Preprocessing is single-threaded, deliberately. Adding parallelism here would mostly fight HiGHS for cores and complicate determinism. If preprocessing is the bottleneck after the Provider lands, the answer is "make HiGHS use those cores," not "parallelise preprocessing."

### Not: an extensible plugin system for Providers

There are at most three Provider implementations ever expected:
- In-memory polars (Step 1+)
- DuckDB (possibly, after Step 3)
- A mock for unit tests

That's it. Don't build registration / discovery / configuration / plugin loading. Hard-code the choice in `cmd_run_flextool.py` and accept a flag if needed.

### Not: backward-compatible aliases for the old API

After Step 2 deletes the seed funnel, do not keep deprecation shims. The seed funnel is gone, not "deprecated." Anything that called `_seed_lookup` either now calls `provider.get` or has been deleted.

---

## The user-experience principle behind all of this

The user's stated principle, paraphrased: **FlexTool should never be a bigger memory or speed bottleneck than HiGHS.** HiGHS is doing the work that justifies the run; everything else is overhead.

The architecture supports this by:
- Making the memory ceiling explicit (Provider + item-group streaming = HiGHS matrix + ε).
- Keeping the source-of-data swap cheap (Provider interface = same loader code regardless of source).
- Deleting bridging code aggressively (Step 2 of the migration = pure deletion).
- Avoiding speculative complexity (no two-tier, no change tracking, no plugins).

Every deviation from these principles needs a strong argument grounded in user-visible behaviour. "It might be useful later" is not enough.

---

## Reading guide for the next agent

Order to read these documents:

1. **This document** (`architecture_provider_and_future_duckdb.md`) — for the why.
2. **`flex_data_provider_migration_handoff.md`** — for the what and how.
3. The Phase E-g commit message (`git show 988efb2e`) — for the most recent on-the-ground state.
4. `flextool/engine_polars/_input_source.py` (current seed-funnel implementation) — for what's being deleted.
5. `flextool/engine_polars/_flex_data_accumulator.py` (current in-memory carrier) — for what's being absorbed into the Provider.
6. `flextool/engine_polars/_orchestration.py` — for the cascade control flow that the Provider plugs into.

If something in this document seems to conflict with the migration handoff, the migration handoff is authoritative for *what to do*; this document is authoritative for *why*. Conflicts in either direction should be surfaced to the user, not silently resolved.
