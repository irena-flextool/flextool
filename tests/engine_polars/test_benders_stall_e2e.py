"""End-to-end reproduction of the Benders stall guard on the real H2-trade
model — the only test that exercises the true master/region + LB-pinning
interaction that drives a stall (§5 of the plan).

At ``TRADE_INVEST_COST_SCALE = 0.001`` (near-free trade) the N=3 H2-trade model
does NOT converge: one node group (``decomp_KOR``) is near-infeasible standalone
(autarky cost ~3000x the next), so the master keeps proposing coupling flows
that force the penalty/slack regime. The best feasible cost freezes at ~1.03e11
(vs a true monolith optimum of ~6.68e6) and the loop would otherwise burn all 50
iterations returning that garbage. The stall guard must fire and RAISE the
plain-English diagnostic instead.

RESOLVED (recorded here so the guard's assumptions stay documented): N=3 at
0.001x STALLS CLEANLY to the iteration cap — it does NOT crash (the benchmark's
rc=1 crash was N=6-specific, at iter 41). The guard therefore pre-empts a
50-iteration silent stall, firing at iter (frozen-start + K); with the default
K=8 that is ~iter 34.

PROVISIONING. This drives the full CLI cascade on the generated H2-trade DB,
which is large and slow (~45 s, ~1.3 GB) and is NOT a checked-in fixture — it is
produced by the plan's benchmark driver (``plexos-to-flextool`` +
``run_benchmark.py::scale_trade_invest_cost``). The test therefore SKIPS unless
the DB is present, located via ``FLEXTOOL_STALL_E2E_DB`` (an explicit sqlite
path) or the default benchmark scratch location. It is opt-in by design so the
routine engine_polars suite stays fast; run it explicitly when validating the
guard against the real model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The scaled (0.001x trade cost) N=3 H2-trade DB produced by the benchmark
# driver. Overridable so a fresh generation can point the test at its own DB.
_DEFAULT_DB = Path("/tmp/benders_bench/dbs/h2_3reg.sqlite")
_DB = Path(os.environ.get("FLEXTOOL_STALL_E2E_DB", str(_DEFAULT_DB)))
_SCENARIO = "lt_rp_only_lagrangian"

_pytestmark_reason = (
    f"stall e2e DB not present at {_DB} (set FLEXTOOL_STALL_E2E_DB to the "
    f"scaled N=3 H2-trade sqlite produced by the benchmark driver)"
)

pytestmark = pytest.mark.skipif(not _DB.exists(), reason=_pytestmark_reason)


def _scale_marker_ok(db: Path) -> bool:
    """The benchmark driver stamps a ``<db>.tradecost_scaled`` sidecar whose
    first line is ``scale=<factor>``; require the 0.001 near-free-trade factor
    so the test only runs against a DB actually in the stalling regime."""
    marker = db.with_suffix(db.suffix + ".tradecost_scaled")
    if not marker.exists():
        return False
    try:
        first = marker.read_text().splitlines()[0]
    except OSError:
        return False
    return first.strip() == "scale=0.001"


def test_benders_stall_guard_fires_on_real_h2_trade(tmp_path):
    """The 0.001x-trade N=3 H2-trade solve must RAISE the stall diagnostic
    (naming the worst-offender node group) rather than silently exhaust the
    iteration cap."""
    if not _scale_marker_ok(_DB):
        pytest.skip(
            f"{_DB} is not stamped scale=0.001 (near-free trade) — the "
            "stalling regime; regenerate it with the benchmark driver."
        )

    outdir = tmp_path / "out"
    outdir.mkdir()
    env = dict(os.environ)
    # Fire the guard promptly so the test is as short as the real dynamics
    # allow; the incumbent freezes ~iter 27, so K=3 fires ~iter 30. This does
    # NOT change WHETHER it stalls, only how early the guard reports it.
    env["FLEXTOOL_BENDERS_MAX_STALL"] = "3"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "flextool.cli.cmd_run_flextool",
            f"sqlite:///{_DB.resolve()}",
            "--scenario-name",
            _SCENARIO,
            "--output-location",
            str(outdir),
            "--highs-threads",
            "1",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    combined = proc.stdout + proc.stderr

    # The solve must FAIL (non-zero exit) with the stall diagnostic — not
    # converge, and not run to the 50-iteration cap.
    assert proc.returncode != 0, (
        "expected the stalled solve to fail, but it exited 0\n" + combined[-3000:]
    )
    assert "Benders stalled at iteration" in combined, (
        "stall guard did not fire; tail of run output:\n" + combined[-3000:]
    )
    # Three-section plain-English diagnostic naming the worst-offender node
    # group (the near-infeasible KOR-like region).
    assert "What this means:" in combined
    assert "How to avoid it:" in combined
    # The near-infeasible KOR-like node group is named as the root cause.
    assert "Node group 'decomp_KOR' is the likely cause" in combined
    assert "cannot meet its own demand without imports" in combined
    # It must NOT have silently reached the iteration cap.
    assert "DID NOT CONVERGE after 50/50" not in combined
