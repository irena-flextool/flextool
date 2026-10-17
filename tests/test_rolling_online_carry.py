"""Regression: rolling online-commitment carry (OC1, Carrier 1).

A rolling-window unit-commitment solve used to re-charge a unit's startup
cost at EVERY roll boundary, even when the unit stayed committed across
the boundary.  The online dynamics reference the predecessor commitment
at ``t_previous_within_solve``, which at a roll's first step cyclically
wraps to the roll's OWN last look-ahead step (``_block_layout.py``) — a
physically meaningless value on a continuation roll, with zero memory of
the prior roll's realized commitment.

The fix is the online analog of the storage ``roll_end_state →
p_roll_continue_state`` carry (purely internal / derived — no schema, no
migration): the prior roll's committed-jump-end ``v_online`` is carried as
``SolveHandoff.roll_end_online`` and overlaid as
``FlexData.p_roll_continue_online``; at the continuation roll's first step
the spurious cyclic predecessor is sliced out of the online Lag frame and
the carried value is injected as the predecessor constant, so::

    v_startup[first] >= v_online[first] - carried

(no startup when the unit stays committed; a genuine OFF→ON start still
pays — ``carried ≈ 0`` leaves the full charge).

Driven witness: ``coal_min_load_fullYear_roll`` (a ``min_load_efficiency``
coal unit on the fullYear timeline, ``rolling_solve_horizon=8``,
``rolling_solve_jump=4`` → 72 rolls spanning p2020/p2025/p2030/p2035, so
multiple-period roll windows are exercised).  Pre-fix the realized coal
``v_startup`` summed to 7.284 (startup cost 364 200 €); with the carry the
boundary over-charges vanish and it drops to 5.854 (292 700 €).
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

SCENARIO = "coal_min_load_fullYear_roll"
UNIT = "coal_plant"
STARTUP_COST = 100.0  # €/unit-start
UNITSIZE = 500.0      # MW  → objective coef = STARTUP_COST * UNITSIZE

# Driven baselines (see module docstring).  Pre-fix the cyclic-wrap
# re-charge inflated the yearly realized startup to 7.284; the carry
# removes exactly the boundary over-charges.
CORRECTED_TOTAL_STARTUP = 5.854
PREFIX_TOTAL_STARTUP = 7.284


def _add_scenario(db_url: str) -> None:
    from spinedb_api import DatabaseMapping, import_data

    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            scenarios=[(SCENARIO, False, "")],
            scenario_alternatives=[
                (SCENARIO, "init", "west"),
                (SCENARIO, "west", "coal"),
                (SCENARIO, "coal", "coal_min_load"),
                (SCENARIO, "coal_min_load", "wind"),
                (SCENARIO, "wind", "fullYear"),
                (SCENARIO, "fullYear", "dispatch_fullYear_roll"),
                (SCENARIO, "dispatch_fullYear_roll", None),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add rolling online-carry witness scenario")


@pytest.fixture(scope="module")
def rolling_uc(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Drive the witness ONCE with ``keep_solutions=True`` so the per-roll
    ``OrchestrationStep`` (``handoff`` producer + ``flex_data`` consumer)
    and the realized ``output_raw`` parquets are both available."""
    db_path = tmp_path_factory.mktemp("db_oc1") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _add_scenario(url)
    workdir = tmp_path_factory.mktemp("oc1_run")
    steps = run_chain_from_db(
        url, SCENARIO, work_folder=workdir, keep_solutions=True,
    )
    return {"steps": steps, "workdir": workdir}


def _roll_index(name: str) -> int:
    return int(name.split("roll_")[-1])


def _ordered_steps(result: dict):
    return sorted(result["steps"].items(), key=lambda kv: _roll_index(kv[0]))


def _per_roll_frame(workdir: Path, key: str) -> dict[int, pl.DataFrame]:
    raw = workdir / "output_raw"
    out: dict[int, pl.DataFrame] = {}
    for f in raw.glob(f"{key}__*.parquet"):
        out[_roll_index(f.stem)] = pl.read_parquet(f).sort(["period", "time"])
    return out


def _carried_value(step, unit: str) -> float | None:
    """The committed value this roll CONSUMED (overlaid carrier), or None."""
    p = getattr(step.flex_data, "p_roll_continue_online", None)
    if p is None:
        return None
    row = p.frame.filter(pl.col("p").cast(pl.Utf8) == unit)
    return float(row["value"][0]) if row.height else 0.0


def _produced_value(step, unit: str) -> float | None:
    """The committed jump-end value this roll PRODUCED, or None."""
    ho = getattr(step, "handoff", None)
    frame = None if ho is None else getattr(ho, "roll_end_online", None)
    if frame is None:
        return None
    row = frame.filter(pl.col("process").cast(pl.Utf8) == unit)
    return float(row["value"][0]) if row.height else 0.0


def test_all_rolls_optimal(rolling_uc):
    """Every roll of the rolling UC solve reaches optimality with the carry."""
    steps = rolling_uc["steps"]
    assert len(steps) == 72, f"expected 72 rolls, got {len(steps)}"
    assert all(s.optimal for s in steps.values()), {
        n: s.optimal for n, s in steps.items() if not s.optimal
    }


def test_carrier_produced_and_consumed(rolling_uc):
    """The online carrier is produced by each roll and consumed by the next.

    Producer-level (HEAD has no ``roll_end_online`` / ``p_roll_continue_online``
    field).  The first roll consumes NO carrier (byte-identical baseline);
    every continuation roll consumes exactly the prior roll's committed
    jump-end ``v_online``.
    """
    ordered = _ordered_steps(rolling_uc)

    # Roll 0 consumes nothing (no prior handoff) and still produces a carrier.
    first_step = ordered[0][1]
    assert _carried_value(first_step, UNIT) is None, (
        "first roll must consume no online carrier (byte-identical path)"
    )
    assert _produced_value(first_step, UNIT) is not None, (
        "first roll must still PRODUCE a roll_end_online carrier for roll 1"
    )

    # Each continuation roll consumes exactly the prior roll's produced value.
    for i in range(1, len(ordered)):
        prior = _produced_value(ordered[i - 1][1], UNIT)
        consumed = _carried_value(ordered[i][1], UNIT)
        assert consumed is not None, (
            f"roll {_roll_index(ordered[i][0])} consumed no carrier"
        )
        assert consumed == pytest.approx(prior, abs=1e-9), (
            f"roll {_roll_index(ordered[i][0])}: consumed carrier {consumed} "
            f"!= prior roll's committed jump-end {prior}"
        )


def test_total_startup_drops_to_corrected_value(rolling_uc):
    """The yearly realized startup drops to the corrected (no boundary
    re-charge) value — strictly below the pre-fix over-charged total."""
    su = _per_roll_frame(rolling_uc["workdir"], "v_startup_linear")
    total = sum(float(df[UNIT].sum()) for df in su.values())
    assert total == pytest.approx(CORRECTED_TOTAL_STARTUP, abs=5e-3), (
        f"realized yearly v_startup={total} (expected {CORRECTED_TOTAL_STARTUP})"
    )
    assert total < PREFIX_TOTAL_STARTUP - 1.0, (
        f"startup {total} not meaningfully below pre-fix {PREFIX_TOTAL_STARTUP}"
    )
    # Objective-coef translation (startup_cost × unitsize): the saved cost.
    saved = (PREFIX_TOTAL_STARTUP - total) * STARTUP_COST * UNITSIZE
    assert saved == pytest.approx(71_500.0, abs=500.0), (
        f"saved startup cost {saved} € off expected ≈71 500 €"
    )


def test_boundary_startup_equals_increment(rolling_uc):
    """At every continuation-roll first step the startup equals the honest
    increment ``max(0, v_online_first - carried)`` — NOT the full online
    (which the cyclic-wrap bug charged).  This single invariant encodes
    both suppression (committed → carried>0 → little/no startup) and the
    cold-start charge (carried≈0 → full startup)."""
    ordered = _ordered_steps(rolling_uc)
    on = _per_roll_frame(rolling_uc["workdir"], "v_online_linear")
    su = _per_roll_frame(rolling_uc["workdir"], "v_startup_linear")

    committed_suppressed = 0
    for i in range(1, len(ordered)):
        r = _roll_index(ordered[i][0])
        online_first = float(on[r][UNIT][0])
        startup_first = float(su[r][UNIT][0])
        carried = _carried_value(ordered[i][1], UNIT)
        expected = max(0.0, online_first - carried)
        assert startup_first == pytest.approx(expected, abs=1e-2), (
            f"roll {r}: startup_first={startup_first} != "
            f"max(0, online_first={online_first} - carried={carried})={expected}"
        )
        # Suppression witness: a unit committed into this roll (carried>0)
        # whose first step keeps it on must NOT be re-charged the full online.
        if carried > 0.1 and online_first > carried + 1e-6:
            assert startup_first < online_first - 1e-3, (
                f"roll {r}: committed unit re-charged a full startup "
                f"({startup_first} ≈ online {online_first})"
            )
            committed_suppressed += 1
    assert committed_suppressed > 0, (
        "no committed-across-boundary roll witnessed — witness fixture changed?"
    )


def test_genuine_cold_start_still_charged(rolling_uc):
    """The fix is surgical to the roll boundary: a genuine OFF→ON start
    (predecessor ≈ 0) still pays its full startup.  Witnessed by a mid-roll
    off→on step, whose cyclic lag the carry never touches."""
    on = _per_roll_frame(rolling_uc["workdir"], "v_online_linear")
    su = _per_roll_frame(rolling_uc["workdir"], "v_startup_linear")

    found = False
    for r in sorted(on):
        o = on[r][UNIT].to_list()
        s = su[r][UNIT].to_list()
        for k in range(1, len(o)):  # mid-roll steps only
            if o[k - 1] < 1e-3 and o[k] > 0.05:
                # off→on within the roll → full startup charged.
                assert s[k] == pytest.approx(o[k], abs=1e-2), (
                    f"roll {r} step {k}: genuine cold start not fully charged "
                    f"(startup {s[k]} vs online {o[k]})"
                )
                found = True
    assert found, "no mid-roll cold start witnessed — witness fixture changed?"
