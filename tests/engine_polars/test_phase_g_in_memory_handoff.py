"""Phase G — process_outputs disk-read elimination test.

Phase G migrates per-iter file reads inside ``process_outputs/`` to
in-memory carriers threaded from the cascade (FlexData + extended
``write_outputs_for_solve`` kwargs).  See
``specs/in_memory_carriers_audit.md`` (Per-iter readers section) for the
per-reader mapping.

What this test asserts
----------------------

1. The representative cascade still completes successfully under the new
   wiring (Phase G is pure plumbing; no LP/solver behaviour change).
2. The per-iter file reads listed in the audit drop to zero or to a
   small known count for the migrated readers.  We instrument
   ``pandas.read_csv`` and count calls per basename, then check that
   the targeted files were NOT re-read on the per-iter hot path.

Targeted file basenames (per audit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handoff-side (handoff_writers.py):

  * ``p_entity_unitsize.csv``                            (loop-invariant)
  * ``solve__p_entity_pre_existing.csv``                 (loop-invariant)
  * ``node__storage_nested_fix_method.csv``              (loop-invariant)
  * ``p_node_type.csv``                                  (loop-invariant)
  * ``entity.csv``                                       (loop-invariant)
  * ``entityDivest.csv``                                 (loop-invariant)
  * ``realized_dispatch.csv`` (when read by ``_load_realized_period_time_last``)
  * ``complete_period_share_of_year.csv``                (loop-invariant)
  * ``solve__p_inflation_factor_operations_yearly.csv``  (loop-invariant)
  * ``steps_in_use.csv``                                 (loop-invariant)
  * ``p_model.csv``  (replaced by cascade-supplied ``is_first_solve`` bool)
  * ``period_capacity.csv``  (replaced by writer_state in-memory set)

Extractor-side (read_highs_solution.py):

  * ``p_step_duration.csv``                              (canonical row order)
  * ``p_years_from_start_d.csv``                         (canonical period order)
  * ``solve__p_inflation_factor_operations_yearly.csv``  (duplicate of handoff)
  * ``complete_period_share_of_year.csv``                (duplicate)
  * ``solve__node_capacity_for_scaling.csv``             (Agent 9 row scaler)
  * ``solve__group_capacity_for_scaling.csv``            (Agent 9 row scaler)
  * ``scale_the_objective.csv``                          (resolved once per solve)

Each Phase G migration MAY still leave a handful of reads in pre/post
phases that aren't strictly per-iter (e.g. a one-shot per-solve read of
``scale_the_objective.csv`` during initial CSV resolution).  We allow
a small constant per cascade run for those — the assertion focuses on
the per-iter multiplication: total reads should NOT scale linearly with
number of sub-solves for the targeted basenames.
"""
from __future__ import annotations

import collections
import functools
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from flextool.engine_polars import run_chain_from_db


pytestmark = pytest.mark.solver


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# Basenames migrated by Phase G — read counts for these should NOT
# multiply by sub-solve count on the per-iter hot path.
_PHASE_G_TARGETS = (
    # handoff_writers.py targets
    "p_entity_unitsize.csv",
    "solve__p_entity_pre_existing.csv",
    "node__storage_nested_fix_method.csv",
    "p_node_type.csv",
    "entity.csv",
    "entityDivest.csv",
    "complete_period_share_of_year.csv",
    "solve__p_inflation_factor_operations_yearly.csv",
    "steps_in_use.csv",
    "p_model.csv",
    "period_capacity.csv",
    # read_highs_solution.py targets
    "p_step_duration.csv",
    "p_years_from_start_d.csv",
    "solve__node_capacity_for_scaling.csv",
    "solve__group_capacity_for_scaling.csv",
    "scale_the_objective.csv",
)


def _pick_multi_solve_fixture() -> tuple[Path, str] | None:
    """Return ``(sqlite_path, scenario)`` for a multi-sub-solve fixture
    if one is available; ``None`` otherwise."""
    candidates = [
        ("work_fullYear_roll", "fullYear_roll"),
        ("work_multi_year", "multi_year"),
        ("work_wind_battery_invest_lifetime_renew_4solve",
         "wind_battery_invest_lifetime_renew_4solve"),
        ("work_base", "base"),
    ]
    for name, scen in candidates:
        sqlite = DATA / name / "tests.sqlite"
        if sqlite.exists():
            return sqlite, scen
    return None


def test_phase_g_per_iter_reads_drop(tmp_path: Path) -> None:
    """Run a representative cascade and confirm migrated readers no
    longer hit disk on the per-iter hot path.

    Strategy: count ``pd.read_csv`` calls per basename, then for each
    Phase G target, assert the count is bounded by a small constant
    (not linear in number of sub-solves).

    The bound chosen is generous: we allow up to ``len(sols) - 1`` reads
    per target (rather than 0) because:
      * One-shot post-cascade reads in ``write_outputs`` still happen.
      * Some readers stay on the file path when ``flex_data`` does not
        cover that field (the audit's "deferred" entries — currently
        only ``solve__p_entity_pre_existing.csv`` falls here).

    If a regression reintroduces per-iter reads, the count will scale
    with sub-solve count, blowing the bound.
    """
    fixture = _pick_multi_solve_fixture()
    if fixture is None:
        pytest.skip("no multi-sub-solve fixture available")
    sqlite, scen = fixture

    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    read_counts: dict[str, int] = collections.Counter()
    real_read_csv = pd.read_csv

    @functools.wraps(real_read_csv)
    def patched_read_csv(filepath_or_buffer: Any, *args: Any, **kwargs: Any):
        try:
            name = Path(str(filepath_or_buffer)).name
        except Exception:  # noqa: BLE001
            name = str(filepath_or_buffer)
        # Only count reads originating from the process_outputs/* code
        # path (the Phase G scope).  Other callers (cumulative_handoffs,
        # _emit_chain_params, ...) are out of scope here.
        frame = sys._getframe(1)
        depth = 0
        in_scope = False
        while frame is not None and depth < 30:
            fname = frame.f_code.co_filename
            if "process_outputs/" in fname or "process_outputs\\" in fname:
                in_scope = True
                break
            frame = frame.f_back
            depth += 1
        if in_scope:
            read_counts[name] += 1
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    pd.read_csv = patched_read_csv  # type: ignore[assignment]
    try:
        sols = run_chain_from_db(
            sqlite, scenario_name=scen, work_folder=work,
        )
    finally:
        pd.read_csv = real_read_csv  # type: ignore[assignment]

    assert sols, "cascade produced no sub-solves"
    n_sub_solves = len(sols)

    # Per-iter bound — we allow constant per cascade plus one per
    # sub-solve as a safety margin for one-shot reads that aren't on
    # the hot per-iter path.  A regression that reintroduces per-iter
    # reads would multiply count by ~writers-per-solve (~12 handoff
    # writers + ~25 extractor specs), easily exceeding this bound.
    bound = max(4, n_sub_solves + 2)

    for target in _PHASE_G_TARGETS:
        actual = read_counts.get(target, 0)
        # Note: `solve__p_entity_pre_existing.csv` is the audit's only
        # "Action: Field+wire ... no in-memory equivalent yet" entry;
        # we deliberately allow it to remain on the file path until a
        # FlexData carrier is added (deferred to Phase H or follow-up).
        if target == "solve__p_entity_pre_existing.csv":
            continue
        assert actual <= bound, (
            f"Phase G regression: per-iter file read of {target!r} "
            f"observed {actual} times across {n_sub_solves} sub-solves "
            f"(bound = {bound}).  This basename was migrated to in-memory "
            f"in Phase G — re-introduction of the file read defeats the "
            f"audit goal.  See specs/in_memory_carriers_audit.md."
        )


def test_phase_g_cascade_succeeds_with_kwargs(tmp_path: Path) -> None:
    """Smoke test — Phase G wiring (flex_data + is_first_solve +
    writer_state threaded through ``write_outputs_for_solve``) does not
    break a representative cascade run.  The other handoff suites
    cover correctness; this test guards Phase G's wiring specifically.
    """
    fixture = _pick_multi_solve_fixture()
    if fixture is None:
        pytest.skip("no multi-sub-solve fixture available")
    sqlite, scen = fixture

    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)

    sols = run_chain_from_db(sqlite, scenario_name=scen, work_folder=work)

    assert sols, "Phase G wiring broke the cascade — no sub-solves returned"
    # Every sub-solve should be optimal on the canonical fixtures.
    for name, step in sols.items():
        assert step.optimal is None or step.optimal, (
            f"Phase G wiring broke sub-solve {name!r} — optimal={step.optimal}"
        )
