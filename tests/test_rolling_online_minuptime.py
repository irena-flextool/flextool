"""Regression: min_uptime / min_downtime enforcement in ROLLING solves
(MINTIME-1, the prerequisite layer of OC1 Carrier 2).

On a rolling / synthetic sub-solve the online min_uptime / min_downtime
lookback (``uptime_lookback`` / ``pdt_uptime_set`` and the downtime twins)
was **never built** — the synthetic-solve early-return in
``input._apply_db_overrides`` skips ``apply_derived_c`` (§3.8), the sole
producer of those frames, and returned without re-wiring them (unlike the
documented RESERVE-1 / SECTION-1 / MINLOAD-1 / INFLOW-1 / NA-1 gaps).  With
the frames ``None`` the ``minimum_uptime`` / ``minimum_downtime`` constraints
(``model.py`` ``_add_online_block``) were silently dropped **even within a
single roll**, so a unit authored with ``min_uptime`` / ``min_downtime`` was
free to collapse below its own recent startup obligation inside a roll.

The fix wires ``uptime_lookback_from_source`` /
``downtime_lookback_from_source`` (+ the ``pdt_*_set`` projections) into the
synthetic early-return, exactly mirroring ``apply_derived_c`` §3.8.  Both
helpers return ``None`` when no process authors min-up/down, so the wire is a
no-op (byte-identical) on every rolling fixture without min-up/down.

Scope: MINTIME-1 is **within-roll** enforcement only.  Cross-boundary
obligation continuity (an obligation opened in roll N constraining roll N+1's
early steps) is a SEPARATE following slice (OC1 Carrier 2) and is NOT asserted
here — boundary violations may still remain after this slice, by design.

Driven witness: ``coal_min_updown_roll`` — a ``min_load_efficiency`` coal
unit with ``min_uptime = min_downtime = 5`` and ``minimum_time_method="both"``
on the fullYear rolling timeline (``rolling_solve_horizon=8``,
``rolling_solve_jump=4`` → 72 rolls, ``W=5 > jump=4``).  Pre-fix the realized
roll-0 ``v_online`` collapsed to ``[0.418, 0.274, 0.152, 0.044]`` below its own
startup obligation; with the wire it holds at ``[0.418, 0.274, 0.152, 0.152]``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

pytestmark = [pytest.mark.solver]

SCENARIO = "coal_min_updown_roll"
UNIT = "coal_plant"
MIN_TIME = 5          # min_uptime = min_downtime (hourly steps → W = 5 steps)
EXISTING_COUNT = 1.0  # existing=500, unitsize=500 → one unit, v_online∈[0,1]


def _add_scenario(db_url: str) -> None:
    """Insert the witness scenario — the Carrier-1 chain with the
    ``coal_min_updowntime`` alternative inserted (sets ``min_uptime=5`` /
    ``min_downtime=5`` / ``minimum_time_method="both"`` on ``coal_plant``).
    Built from JSON per CLAUDE.md invariant #3 (never a checked-in .sqlite).
    """
    from spinedb_api import DatabaseMapping, import_data

    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            scenarios=[(SCENARIO, False, "")],
            scenario_alternatives=[
                (SCENARIO, "init", "west"),
                (SCENARIO, "west", "coal"),
                (SCENARIO, "coal", "coal_min_load"),
                (SCENARIO, "coal_min_load", "coal_min_updowntime"),
                (SCENARIO, "coal_min_updowntime", "wind"),
                (SCENARIO, "wind", "fullYear"),
                (SCENARIO, "fullYear", "dispatch_fullYear_roll"),
                (SCENARIO, "dispatch_fullYear_roll", None),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add rolling min-up/down witness scenario")


@pytest.fixture(scope="module")
def rolling_uc(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Drive the witness ONCE with ``keep_solutions=True`` so each per-roll
    ``flex_data`` (the MINTIME-1 lookback consumer) and the realized
    ``output_raw`` parquets are both available."""
    db_path = tmp_path_factory.mktemp("db_mintime") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _add_scenario(url)
    workdir = tmp_path_factory.mktemp("mintime_run")
    steps = run_chain_from_db(
        url, SCENARIO, work_folder=workdir, keep_solutions=True,
    )
    return {"steps": steps, "workdir": workdir}


def _roll_index(name: str) -> int:
    return int(name.split("roll_")[-1])


def _per_roll_frame(workdir: Path, key: str) -> dict[int, pl.DataFrame]:
    raw = workdir / "output_raw"
    out: dict[int, pl.DataFrame] = {}
    for f in raw.glob(f"{key}__*.parquet"):
        out[_roll_index(f.stem)] = pl.read_parquet(f).sort(["period", "time"])
    return out


def test_all_rolls_optimal(rolling_uc):
    """Every roll of the rolling min-up/down UC solve reaches optimality."""
    steps = rolling_uc["steps"]
    assert len(steps) == 72, f"expected 72 rolls, got {len(steps)}"
    assert all(s.optimal for s in steps.values()), {
        n: s.optimal for n, s in steps.items() if not s.optimal
    }


def test_mintime_lookback_built_each_roll(rolling_uc):
    """MINTIME-1 emission: every rolling sub-solve builds the min_uptime /
    min_downtime lookback + projected pdt set (HEAD-failing — all ``None``
    on the synthetic path, so the constraints were silently absent)."""
    for name, step in rolling_uc["steps"].items():
        fd = step.flex_data
        ulb = getattr(fd, "uptime_lookback", None)
        pds = getattr(fd, "pdt_uptime_set", None)
        dlb = getattr(fd, "downtime_lookback", None)
        pdd = getattr(fd, "pdt_downtime_set", None)
        assert ulb is not None and ulb.height > 0, (
            f"{name}: uptime_lookback not built on rolling sub-solve"
        )
        assert pds is not None and pds.height > 0, (
            f"{name}: pdt_uptime_set not built on rolling sub-solve"
        )
        assert dlb is not None and dlb.height > 0, (
            f"{name}: downtime_lookback not built on rolling sub-solve"
        )
        assert pdd is not None and pdd.height > 0, (
            f"{name}: pdt_downtime_set not built on rolling sub-solve"
        )


def test_within_roll_min_uptime_enforced(rolling_uc):
    """Within each roll, ``v_online[k] >= Σ v_startup`` over the within-roll
    min_uptime window (the ``MIN_TIME`` consecutive steps ending at ``k``).
    HEAD-failing: roll 0 collapses to ``online[3]=0.044`` under its own
    ``startup[0]=0.152``.  Cross-boundary rows are NOT checked (Carrier 2)."""
    on = _per_roll_frame(rolling_uc["workdir"], "v_online_linear")
    su = _per_roll_frame(rolling_uc["workdir"], "v_startup_linear")

    violations = []
    for r in sorted(on):
        o = on[r][UNIT].to_list()
        s = su[r][UNIT].to_list()
        for k in range(len(o)):
            lo = max(0, k - MIN_TIME + 1)
            startup_sum = sum(s[lo:k + 1])
            if o[k] < startup_sum - 1e-6:
                violations.append((r, k, o[k], startup_sum))
    assert not violations, (
        f"within-roll min_uptime violated at {violations[:5]} "
        f"({len(violations)} total)"
    )


def test_within_roll_min_downtime_enforced(rolling_uc):
    """Within each roll, ``v_online[k] + Σ v_shutdown <= existing_count`` over
    the within-roll min_downtime window.  Cross-boundary rows are NOT checked
    (Carrier 2)."""
    on = _per_roll_frame(rolling_uc["workdir"], "v_online_linear")
    sd = _per_roll_frame(rolling_uc["workdir"], "v_shutdown_linear")

    violations = []
    for r in sorted(on):
        o = on[r][UNIT].to_list()
        s = sd[r][UNIT].to_list()
        for k in range(len(o)):
            lo = max(0, k - MIN_TIME + 1)
            shutdown_sum = sum(s[lo:k + 1])
            if o[k] + shutdown_sum > EXISTING_COUNT + 1e-6:
                violations.append((r, k, o[k], shutdown_sum))
    assert not violations, (
        f"within-roll min_downtime violated at {violations[:5]} "
        f"({len(violations)} total)"
    )


def test_roll0_online_holds_not_collapses(rolling_uc):
    """Pinned driven witness: roll 0's realized ``v_online`` holds at its
    startup floor (``[0.418, 0.274, 0.152, 0.152]``) instead of the HEAD
    collapse ``[..., 0.044]`` below the ``startup[0]`` obligation."""
    on = _per_roll_frame(rolling_uc["workdir"], "v_online_linear")
    online0 = on[0][UNIT].to_list()[:4]
    assert online0 == pytest.approx([0.418, 0.274, 0.152, 0.152], abs=5e-3), (
        f"roll-0 online {online0} — expected the held floor, not a collapse"
    )
    # The HEAD collapse was online[3]≈0.044 < startup[0]≈0.152; now it holds.
    assert online0[3] >= online0[2] - 1e-6, (
        f"roll-0 online collapsed at step 3 ({online0}) — min_uptime not held"
    )
