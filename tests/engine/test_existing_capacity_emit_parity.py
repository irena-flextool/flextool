"""Tier A parity of the emitted existing-capacity frames (``_chain_params.emit_existing_capacity``).

The frames every solve emits — ``p_entity_pre_existing``, ``p_entity_all_existing``,
``p_entity_previously_invested_capacity``, ``p_entity_retired_carried``, ``p_entity_invested`` and ``edd_invest`` —
are rendered value by value (``repr``), so two runs that hold the same numbers emit ``.equals`` frames:

* a constant ``existing`` and the equivalent period map, on a chain that invests (carried cohorts in every later
  solve);
* the same run with and without ``--csv-dump``.
"""
from __future__ import annotations

import pytest

from flextool.engine import run_chain_from_db
from tests.engine._unit_db import b64, make_db
from tests.engine.test_existing_map_chain import CHAIN_INVEST, PLANT, _existing

pytestmark = pytest.mark.solver

KEYS = ("p_entity_pre_existing", "p_entity_all_existing", "p_entity_previously_invested_capacity",
        "p_entity_retired_carried", "p_entity_invested", "edd_invest")


def _frames(tmp_path_factory, alt: str, existing: list, *, csv_dump: bool = False) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=[existing], base_chain=CHAIN_INVEST)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True,
                              csv_dump=csv_dump)
    return {name: {k: step.flex_data_provider.get(f"solve_data/{k}") for k in KEYS} for name, step in steps.items()}


def _assert_equal(a: dict, b: dict) -> None:
    assert list(a) == list(b)
    for name in a:
        for key in KEYS:
            assert a[name][key] is not None, (name, key)
            assert a[name][key].equals(b[name][key]), (name, key)


def test_constant_and_equivalent_map_emit_the_same_frames(tmp_path_factory) -> None:
    const = _frames(tmp_path_factory, "emit_const", ["unit", PLANT, "existing", b64(500.0, "float")])
    as_map = _frames(tmp_path_factory, "emit_map", _existing([500.0] * 4))
    assert any(f["p_entity_invested"].height > 0 for f in const.values()), "fixture invariant: the chain invests"
    _assert_equal(const, as_map)


def test_csv_dump_emits_the_same_frames(tmp_path_factory) -> None:
    falling = _existing([500.0, 300.0, 100.0, 0.0])
    _assert_equal(_frames(tmp_path_factory, "emit_plain", falling),
                  _frames(tmp_path_factory, "emit_dump", falling, csv_dump=True))
