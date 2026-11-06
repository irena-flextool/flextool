"""Axis-Enum mode builds the byte-identical LP of the Utf8 run, fixture by fixture (opt-in, ``-m axis_enums``).

``tools/lp_snapshot.py snapshot`` runs the fixtures below twice into one temporary folder, plain and with
``--axis-enums`` (``FLEXTOOL_AXIS_ENUMS=1`` in every worker), each run fanned out over ``-j 8`` worker processes.
One test per fixture then requires that neither run crashed and that ``lp_snapshot.py compare`` finds every
MPS file and the manifest identical (Tier A).

The fixtures cover every site family where Enum mode once compared axis columns of different vocabularies:
entity / group invest limits and cumulative capacity, non-synchronous limits, the CO2 caps and the rolling CO2
accumulator, the fix-quantity / fix-price handoffs of nested solves, the stochastic and recourse fans
(non-anticipativity, branch periods), rolling dispatch with online variables, and flowGroup limits.

Both snapshots are built once per module (``--dist loadfile`` puts the whole module on one xdist worker; the
lp_snapshot pool runs the fixtures in parallel).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests._support.lp_parity import compare, fixture_exit, snapshot

pytestmark = pytest.mark.solver

FIXTURES = (
    "base", "invest_limits", "invest_total_no_var", "cumulative_capacity",
    "howto_non_sync_curtailment", "nonsync_internal_dc", "nonsync_1var_dc",
    "co2_period_total", "co2_stoch_total", "coal_co2_limit", "co2_period_rolling", "test_a_lot",
    "nested_fix_quantity", "coarse_nested_fix_price",
    "stoch_recourse", "stoch_recourse_na", "stoch_invest_storage", "stoch_t0_commit", "uc_midhorizon",
    "rolling_separate_dispatch", "conn_online_roll",
    "flowgroup_minload_weighted", "flowgroup_stochastic_path",
)
_JOBS = 8
_TIMEOUT = 3600


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("enum_mode_lp_parity")
    plain, enum = root / "plain", root / "enum"
    snapshot(plain, FIXTURES, axis_enums=False, jobs=_JOBS, timeout=_TIMEOUT)
    snapshot(enum, FIXTURES, axis_enums=True, jobs=_JOBS, timeout=_TIMEOUT)
    return plain, enum


@pytest.mark.timeout(2 * _TIMEOUT + 600)
@pytest.mark.parametrize("fixture", FIXTURES)
def test_enum_mode_lp_equals_utf8_lp(snapshots: tuple[Path, Path], fixture: str) -> None:
    plain, enum = snapshots
    for label, out_dir in (("plain", plain), ("axis-enums", enum)):
        rc, log_tail = fixture_exit(out_dir, fixture)
        assert rc == 0, f"{label} run of {fixture} exited {rc}:\n{log_tail}"
    result = compare(plain, enum, fixture)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
