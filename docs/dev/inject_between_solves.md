# Injecting parameters between solves

Reference for the programmatic `override_provider` hook on
[`run_chain_from_db`][rc]. This is the supported mechanism for feeding
data from an external tool — another optimisation model, a market
simulator, a Python data step — into a FlexTool cascade *between*
sub-solves of the same run.

[rc]: https://github.com/irena-flextool/flextool/blob/main/flextool/engine/solve/_orchestration.py

## Purpose

A FlexTool cascade runs several sub-solves in sequence; each one passes
a typed `SolveHandoff` (realised investments, end-of-horizon storage
state, cumulative emissions, …) to the next. The override hook lets a
Python wrapper substitute values for any of those handoff carriers
*before* the next sub-solve's preprocessing reads them.

Typical wrappers:

- An outer model in a bi-level setup writes a Parquet file with target
  capacities; the wrapper reads it and overrides `realized_invest`.
- A market simulator runs between two operational solves; the wrapper
  feeds an updated `cumulative_co2` or `cumulative_commodity` ladder
  into the second solve.
- A Python data step rewrites end-of-horizon storage state from an
  observed (rather than optimised) trajectory to validate roll-forward
  behaviour.

The mechanism is programmatic only — there is no CLI flag. The wrapper
script imports `run_chain_from_db` and passes a callable.

## API

```python
from flextool.engine import run_chain_from_db

run_chain_from_db(
    input_db_url,
    scenario_name=None,
    work_folder=None,
    *,
    override_provider=None,  # Callable[[], dict[str, pl.DataFrame]] | None
    ...,
)
```

The `override_provider` callable takes no arguments and returns a dict
mapping a whitelisted `K.HANDOFF_*` constant to a `polars.DataFrame`.
It is invoked **once per sub-solve iteration**, after the natural
handoff translators and before preprocessing. Returning `{}` (or
`None`) means "no override this iteration".

Full source: `flextool/engine/solve/_orchestration.py` (entry point),
`flextool/engine/solve/_run_model.py` (invocation site),
`flextool/engine/preprocess/_provider_translators.py`
(`translate_overrides_to_provider`, `read_handoff_frame`,
`dump_provider_sources`).

## Whitelisted handoff keys

Only the 11 keys below are overridable. Supplying any other key raises
`ValueError` in `translate_overrides_to_provider`. The constants live
in `flextool.engine.preprocess._provider_keys`; import as
`from flextool.engine.preprocess import _provider_keys as K`.

| Key constant | What it overrides | Required columns |
|---|---|---|
| `K.HANDOFF_REALIZED_INVEST` | Realised investment decisions carried into the next solve's "previously invested capacity" | `entity, period, value` |
| `K.HANDOFF_REALIZED_EXISTING` | Realised existing (non-decision) capacity adjustments | `entity, period, value` |
| `K.HANDOFF_DIVEST_CUMULATIVE` | Cumulative divestments carried forward | `entity, value` |
| `K.HANDOFF_ROLL_END_STATE` | End-of-horizon storage state (downward roll) | `node, value` |
| `K.HANDOFF_UPWARD_ROLL_END_STATE` | End-of-horizon storage state (upward roll) | `node, value` |
| `K.HANDOFF_CUMULATIVE_CO2` | Cumulative CO2 per group/period for the next emission cap | `group, period, value` |
| `K.HANDOFF_CUMULATIVE_COMMODITY` | Cumulative commodity-ladder realised MWh | `commodity, tier, period, p_ladder_cum_realized_mwh` |
| `K.HANDOFF_CUM_SIM_HOURS` | Cumulative simulated hours per period (commodity ladder denominator) | `period, p_ladder_cum_sim_hours` |
| `K.HANDOFF_FIX_STORAGE_QUANTITY` | Fixed storage quantity at specific steps | `node, period, step, p_fix_storage_quantity` |
| `K.HANDOFF_FIX_STORAGE_PRICE` | Fixed storage price at specific steps | `node, period, step, p_fix_storage_price` |
| `K.HANDOFF_FIX_STORAGE_USAGE` | Fixed storage usage at specific steps | `node, period, step, p_fix_storage_usage` |

The column shapes match the empty-frame schemas in
`_provider_translators._HANDOFF_EMPTY_SCHEMAS`; the `value` columns are
read as `Utf8` and cast on consumption (the legacy CSV-roundtrip
contract). Building the frame as numeric `Float64` also works — polars
casts on the consumer side. Match the natural handoff carrier's schema
if in doubt; the easiest pattern is to start from a captured
`SolveHandoff.<field>` frame and transform it (see the example below).

## Lifecycle

At the start of each sub-solve iteration, the orchestrator runs the
following steps in order:

1. **Sequential handoff translation.** The previous sub-solve's
   `SolveHandoff` is fanned into `handoff/*` Provider keys
   (`translate_handoff_to_provider`). A realised-dispatch roll takes
   only `PIN_FIELDS` from its UC roll and `ROLL_FORWARD_FIELDS` from the
   previous dispatch roll.
2. **Parent handoff translation.** In nested cascades, the parent's
   handoff is translated next, `PIN_FIELDS` only (realised investment /
   existing / divest, fixed online, the nested storage fixes).
3. **Routed carriers.** The carriers tied to the realised end — the
   storage state (`handoff/roll_end_state`, `handoff/upward_roll_end_state`)
   and the step carriers (online state, last flow, last step, online
   history) — are written by the carry router (`solve/_carry_routing.py`):
   the latest state realised before this sub-solve at its nesting level or
   deeper, or empty when the sub-solve is the first of its level. The
   running totals (`cumulative_*`, `cum_sim_hours`) are not routed: every
   sub-solve passes them on, so they come from the sequential predecessor.
4. **Override callable.** If `state.override_provider` is non-`None`,
   it is invoked. The returned dict is fanned into the parallel
   `override/*` Provider keys
   (`translate_overrides_to_provider`).
5. **Preprocessing.** Per-solve preprocessing reads handoff values via
   `read_handoff_frame`, which checks the `override/*` slot first and
   falls back to `handoff/*`. Consumers stay key-stable.
6. **Solve.** The LP is built (cold) or updated (warm) and dispatched
   to the solver.
7. **Produced handoff.** After the solve, the sub-solve's own
   `SolveHandoff` is written under `produced_handoff/*`
   (`translate_produced_handoff_to_provider`). The `handoff/*` keys keep
   the values the solve consumed, so a LP rebuilt from the post-solve
   provider is the LP that was solved; read `produced_handoff/*` for what
   the solve passes on (e.g. a roll's final running totals).

The callable is invoked **on every iteration** including the first.
Return `{}` whenever no override applies — that iteration falls through
to the natural handoff carrier with no Provider writes and no log
output. If the callable raises, the exception propagates and the
cascade aborts; clean error handling is the wrapper's responsibility.

Each invocation that returns a non-empty dict logs at INFO::

    [override] applied N keys at iter=I solve=SOLVE_NAME

with the sorted key list at DEBUG.

## Warm-path interaction

Under `warm=True` (the default) an LP is reused for iteration N only when
iteration N's `FlexData` — computed after the override callable has fired
and preprocessing has read the overridden handoff — is identical to
iteration N-1's in every field, labels included, and the solver
configuration is the same (`_reuse_blocker` in `solve/_warm.py`).
Concretely:

- An override that changes anything preprocessing computes (for example
  `K.HANDOFF_REALIZED_INVEST` at iteration N changes
  `p_entity_previously_invested_capacity`) makes iteration N's `FlexData`
  differ, so iteration N builds its own LP: the override always reaches
  the LP matrix, the objective and the right-hand sides.
- An override that changes nothing in `FlexData` leaves the LP identical,
  and reusing it is equivalent to rebuilding it.

No `warm=False` run is needed for overrides to take effect.

## Audit

Set `FLEXTOOL_AUDIT_SOURCES=1` (see [Environment variables](env_vars.md))
to enable the per-sub-solve audit dump. Every Provider key written with
a non-`None` `source` tag is appended to
`<work_folder>/audit_sources.log` as one tab-separated line per write::

    <solve_name>\t<provider_key>\t<source_tag>

Override writes carry the tag `external_override` and the key prefix
`override/`. Useful for confirming the wrapper fired on the iterations
it was meant to (and didn't fire on the ones it wasn't).

## Example wrapper

A wrapper script that reads override frames from a directory of
Parquet files. Each file is named `<scenario>__<solve_name>.parquet`
and contains a single override frame for `realized_invest`. Files that
don't exist are treated as "no override for that iteration".

```python
"""Run a FlexTool cascade with realized_invest overrides supplied
by an external tool that drops Parquet files into a watch directory.

Usage:
    python run_with_overrides.py path/to/input.sqlite scenario_name
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from flextool.engine import run_chain_from_db
from flextool.engine.preprocess import _provider_keys as K


WATCH_DIR = Path("overrides_in")
WORK_FOLDER = Path("work_overrides")


def _make_override_callback(scenario: str, solve_order: list[str]):
    """Return a callable suitable for ``override_provider=``.

    Captures the cascade's expected solve order so the callback knows
    which iteration index maps to which solve name (the orchestrator
    does not pass that into the callable; the wrapper tracks it).
    """
    iter_state = {"i": 0}

    def fetch_override() -> dict[str, pl.DataFrame]:
        i = iter_state["i"]
        iter_state["i"] += 1

        # First sub-solve has no prior carrier — nothing to override.
        if i == 0:
            return {}

        solve_name = solve_order[i] if i < len(solve_order) else None
        if solve_name is None:
            return {}

        parquet = WATCH_DIR / f"{scenario}__{solve_name}.parquet"
        if not parquet.exists():
            # No file from the external tool for this iteration —
            # fall through to the natural handoff carrier.
            return {}

        frame = pl.read_parquet(parquet)
        # Ensure the schema matches what the handoff translator wrote
        # for an empty realized_invest carrier:
        #     (entity, period, value).  Cast value to Utf8 to match
        # the CSV-roundtrip contract (polars also accepts Float64).
        frame = frame.select(
            pl.col("entity").cast(pl.Utf8),
            pl.col("period").cast(pl.Utf8),
            pl.col("value").cast(pl.Utf8),
        )
        return {K.HANDOFF_REALIZED_INVEST: frame}

    return fetch_override


def main() -> None:
    db_path = Path(sys.argv[1])
    scenario = sys.argv[2]

    # The wrapper must know the solve order ahead of time to map
    # iteration index → solve name.  Alternatives:
    #   - hard-code the chain when it is fixed (as below);
    #   - read it from a prior dry-run and persist it;
    #   - gate purely on file existence (drop the solve_order list and
    #     name the Parquet files by iteration index instead).
    solve_order = [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ]

    WORK_FOLDER.mkdir(parents=True, exist_ok=True)
    results = run_chain_from_db(
        db_path,
        scenario_name=scenario,
        work_folder=WORK_FOLDER,
        override_provider=_make_override_callback(scenario, solve_order),
    )

    for name, step in results.items():
        print(f"{name}: optimal={step.optimal} obj={step.obj}")


if __name__ == "__main__":
    main()
```

Set `FLEXTOOL_AUDIT_SOURCES=1` when running this to verify the
override fired on the expected iterations.

## Security

The override callable is arbitrary Python supplied by the wrapper
author. Whoever invokes `run_chain_from_db` already has full Python
execution privileges in the same process, so the override hook
introduces no new injection surface — anything the callable could do,
the wrapper script itself could already do. The 11-key whitelist
constrains *what cascade state* the returned dict can substitute, not
*what code can run inside the callable*. No `eval` / `exec` /
`importlib` machinery is involved on the FlexTool side; the callable
is just invoked with `()` and its return value is consumed.

If you are integrating FlexTool into a service that accepts untrusted
input, the trust boundary is the wrapper around `run_chain_from_db`
itself, not this hook.

## See also

- [`tests/engine/test_override_provider.py`](https://github.com/irena-flextool/flextool/blob/main/tests/engine/test_override_provider.py) —
  end-to-end test of the override path plus the audit-dump variant.
  Useful as a reference for additional patterns (capturing a real
  handoff frame, scaling it, asserting the propagated effect).
- `specs/architecture_provider_and_future_duckdb.md` (internal design notes,
  not in the public repository)
  (sections on Phases 5 and 6) — design rationale for the
  `override/*` namespace, the override-aware `read_handoff_frame`
  lookup, and the audit-dump format.
- [Environment variables](env_vars.md) — `FLEXTOOL_AUDIT_SOURCES` for
  inspecting which iterations were overridden.
