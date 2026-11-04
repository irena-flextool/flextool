"""Axis-Enum mode builds the Utf8 LP at every site that compared two axis vocabularies.

One lp_snapshot fixture per fixed site family runs twice, plain and with ``FLEXTOOL_AXIS_ENUMS=1``
(``tools/lp_snapshot.py snapshot --axis-enums``), and every MPS file and manifest must be byte-identical.
Each fixture crashed in Enum mode before its site was fixed:

* ``howto_non_sync_curtailment`` — non_sync_constraint arc / group-node joins (entity-union vs node Enum);
* ``invest_limits`` — minInvest_entity_period and the group per-period invest rows (entity-union vs process);
* ``stoch_t0_commit`` — non_anticipativity_online: the forecast star's period labels on the branch axis;
* ``ramp_roll_span`` — maxOnline's carried-step anti join (a Utf8 cast against the Enum steps);
* ``co2_period_total`` — the co2_max_total cap's arc / CO2-node join;
* ``coal_co2_limit`` — the rolling CO2 accumulator's concat of producers (node vs entity-union Enum);
* ``test_a_lot`` — the rolling CO2 accumulator's attribution of arc endpoints to groups;
* ``nested_fix_quantity`` — the fix-quantity handoff's unitsize frame (Utf8 literal vs node Enum).

The wider comparison is the opt-in parity run (``tests/engine/axis_enums/``, ``pytest -m axis_enums``); this
subset gates every family in the normal suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests._support.lp_parity import snapshot

pytestmark = pytest.mark.solver

FIXTURES = (
    "howto_non_sync_curtailment", "invest_limits", "stoch_t0_commit", "ramp_roll_span",
    "co2_period_total", "coal_co2_limit", "test_a_lot", "nested_fix_quantity",
)


def _snapshot(out_dir: Path, *, axis_enums: bool):
    return snapshot(out_dir, FIXTURES, axis_enums=axis_enums, jobs=4, timeout=1200)


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for pattern in ("*/*.mps", "*/manifest.json") for p in sorted(root.glob(pattern))}


def test_axis_enum_lp_equals_utf8_lp(tmp_path) -> None:
    plain = _snapshot(tmp_path / "plain", axis_enums=False)
    assert plain.returncode == 0, plain.stdout[-3000:] + plain.stderr[-3000:]
    enum = _snapshot(tmp_path / "enum", axis_enums=True)
    assert enum.returncode == 0, enum.stdout[-3000:] + enum.stderr[-3000:]
    plain_files, enum_files = _files(tmp_path / "plain"), _files(tmp_path / "enum")
    assert {Path(f).parts[0] for f in plain_files} == set(FIXTURES)
    assert sorted(enum_files) == sorted(plain_files)
    different = [f for f in plain_files if plain_files[f] != enum_files[f]]
    assert different == []
