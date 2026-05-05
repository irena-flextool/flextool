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
