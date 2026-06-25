"""Constants for the three-region LH2 test fixture.

The fixture itself lives in :mod:`tests/fixtures/lh2_three_region.json`
(committed to git) and is materialised by tests via
:func:`tests.db_utils.json_to_db` — same pattern as the baseline
``tests.json`` fixture loaded in :mod:`tests.conftest`.

Layout (recap)
--------------
* Three regions A/B/C, each with hourly elec node + daily H2/LH2 nodes.
* Per region: wind, coal, battery (charge+discharge), electrolyser
  (regular connection — straddles hourly elec → daily h2),
  liquefier (h2→lh2, both daily).
* Inter-region H2 pipelines pipe_AB and pipe_BC (daily) between LH2
  storage nodes.
* Single ``coal_market`` commodity node (hourly) feeding every coal
  plant.
* Single solve ``lh2_week`` over 168 hourly timesteps.

Regenerating the JSON
---------------------
The imperative builder that originally produced this fixture lives at
:mod:`tests/fixtures/regen_lh2_three_region.py`.  Run it as a script
to rebuild the SQLite, export to JSON, and overwrite the committed
fixture::

    python tests/fixtures/regen_lh2_three_region.py

The script is byte-deterministic — repeated runs produce identical
JSON.  Tests do NOT invoke the imperative builder; they only consume
the committed JSON.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants used by tests/test_lh2_three_region.py and friends.
# ---------------------------------------------------------------------------

REGIONS: tuple[str, str, str] = ("A", "B", "C")
SCENARIO: str = "lh2_three_region"
ALT: str = "lh2_three_region"

N_HOURS: int = 168
N_DAYS: int = 7
HOURLY_STEPS: list[str] = [f"t{i:04d}" for i in range(1, N_HOURS + 1)]
DAILY_STEPS: list[str] = [HOURLY_STEPS[d * 24] for d in range(N_DAYS)]


# ---------------------------------------------------------------------------
# Horizon descriptor — lets the imperative builder emit fixtures at a
# different (n_hours, n_days) horizon WITHOUT mutating the module-level
# constants above.  The DEFAULT horizon reproduces the committed
# ``lh2_three_region.json`` byte-for-byte; a shorter horizon (e.g. 48h /
# 2 days) is used by the Benders Phase-0 ``lh2_three_region_trade_invest``
# sibling fixture.  Every time-synthesis / payload helper in
# ``regen_lh2_three_region.py`` accepts a :class:`Horizon`; passing
# ``Horizon.default()`` keeps the legacy emit unchanged.
# ---------------------------------------------------------------------------


class Horizon:
    """An (n_hours, n_days) timeline descriptor.

    ``hourly_steps`` are ``t0001 … t{n_hours:04d}``; ``daily_steps`` are
    the per-day block step labels ``hourly_steps[d*24]`` for each day —
    exactly the derivation the legacy module-level constants used.
    """

    def __init__(self, n_hours: int, n_days: int) -> None:
        if n_days * 24 > n_hours:
            raise ValueError(
                f"n_days*24 ({n_days * 24}) exceeds n_hours ({n_hours}); "
                "the daily-block step labels would index past the timeline."
            )
        self.n_hours = n_hours
        self.n_days = n_days
        self.hourly_steps: list[str] = [
            f"t{i:04d}" for i in range(1, n_hours + 1)
        ]
        self.daily_steps: list[str] = [
            self.hourly_steps[d * 24] for d in range(n_days)
        ]

    @classmethod
    def default(cls) -> "Horizon":
        """The committed-fixture horizon (168 hours / 7 days)."""
        return cls(N_HOURS, N_DAYS)
