"""Every fixture builder script still reproduces the data of its committed fixture.

A builder takes its schema sections from ``spinedb_schema.json`` and writes
through ``migrate_fixture``; its data sections are compared, as sets, with
the committed fixture brought through the same ``migrate_fixture`` (a round
trip may reorder rows).  A difference means the builder or the fixture
changed alone: fix the builder, or rebuild the fixture on purpose.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import pytest

from flextool.update_flextool.fixture_schema import normalise_data, section_diffs
from flextool.update_flextool.test_fixtures import migrate_fixture, write_built_fixture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# builder module (in tests/fixtures) -> committed fixture it writes
SPEC_BUILDERS: dict[str, str] = {
    "build_stoch_two_period": "stoch_two_period.json",
    "build_stoch_two_period_invest": "stoch_two_period_invest.json",
    "build_stoch_two_period_invest_na": "stoch_two_period_invest_na.json",
    "build_stoch_two_period_hedge": "stoch_two_period_hedge.json",
    "build_stoch_two_period_free_ops": "stoch_two_period_free_ops.json",
    "build_stoch_two_period_nreal0": "stoch_two_period_nreal0.json",
    "build_stoch_two_period_midhorizon_nreal0": "stoch_two_period_midhorizon_nreal0.json",
    "build_stoch_realized_dispatch": "stoch_realized_dispatch.json",
    "build_w4b_auto_dispatch": "w4b_auto_dispatch.json",
    "build_w4b_reserve_dispatch": "w4b_reserve_dispatch.json",
    "build_w4c_divergent_dispatch": "w4c_divergent_dispatch.json",
}


def _module(name: str):
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    return importlib.import_module(name)


def _spec_builder(module: str) -> Callable[[Path], None]:
    return lambda out: write_built_fixture(_module(module).build(), out)


def _lh2(function: str) -> Callable[[Path], None]:
    return lambda out: getattr(_module("regen_lh2_three_region"), function)(out)


def _augment(module: str, source: str) -> Callable[[Path], None]:
    return lambda out: _module(module).build(FIXTURES / source, out)


BUILDERS: dict[str, tuple[Callable[[Path], None], str]] = {
    **{module: (_spec_builder(module), fixture) for module, fixture in SPEC_BUILDERS.items()},
    "regen_lh2_three_region": (_lh2("regenerate_json"), "lh2_three_region.json"),
    "regen_lh2_three_region:trade_invest": (_lh2("regenerate_trade_invest_json"),
                                            "lh2_three_region_trade_invest.json"),
    "regen_lh2_three_region:rp_invest": (_lh2("regenerate_rp_invest_json"), "lh2_three_region_rp_invest.json"),
    "_augment_stochastics_pbt_inflow": (_augment("_augment_stochastics_pbt_inflow", "stochastics.json"),
                                        "stochastics_pbt_inflow.json"),
    "_augment_branch2_parent_period": (_augment("_augment_branch2_parent_period", "stochastics_pbt_inflow.json"),
                                       "branch2_parent_period.json"),
}


def _diff(built: dict, committed: dict) -> list[str]:
    lines = []
    for key in sorted(set(built) | set(committed)):
        b, c = built.get(key, set()), committed.get(key, set())
        lines += [f"{key}: builder only: {str(row)[:200]}" for row in sorted(b - c, key=repr)[:5]]
        lines += [f"{key}: fixture only: {str(row)[:200]}" for row in sorted(c - b, key=repr)[:5]]
    return lines


@pytest.mark.parametrize("name", list(BUILDERS))
def test_builder_reproduces_committed_fixture_data(name: str, tmp_path: Path) -> None:
    build, fixture = BUILDERS[name]
    out = tmp_path / fixture
    build(out)
    with open(out) as f:
        built = json.load(f)
    assert section_diffs(built, "packed") == [], "a builder writes the schema's sections"
    committed_path = tmp_path / ("committed_" + fixture)
    migrate_fixture(FIXTURES / fixture, committed_path)
    with open(committed_path) as f:
        committed = json.load(f)
    diff = _diff(normalise_data(built, "packed"), normalise_data(committed, "packed"))
    assert not diff, f"{name} no longer reproduces tests/fixtures/{fixture}:\n" + "\n".join(diff)


def test_every_builder_script_is_checked() -> None:
    """A new ``build_*.py`` is added to the check.  Not checked: ``build_h2_trade_parity`` (reads a
    private source database) and ``build_lh2_three_region`` (constants of ``regen_lh2_three_region``)."""
    scripts = {p.stem for p in FIXTURES.glob("build_*.py")}
    assert scripts - set(SPEC_BUILDERS) == {"build_h2_trade_parity", "build_lh2_three_region"}
