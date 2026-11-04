"""``FLEXTOOL_AXIS_ENUMS=1`` switches a whole ``run_chain_from_db`` run to axis-Enum mode.

The switch sets the scenario's vocabulary on the cascade-input provider; every sub-solve provider inherits it,
so every sub-solve's FlexData is loaded with the Enums.  The run logs one EXPERIMENTAL warning.  Unset (or any
other value) leaves every provider and FlexData without a vocabulary — the production Utf8 mode.
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

pytestmark = pytest.mark.solver

SCENARIO = "multi_year"  # a deterministic chain of several solves


def _run(test_db_url, tmp_path, caplog, monkeypatch, env_value):
    from flextool.engine import run_chain_from_db

    if env_value is None:
        monkeypatch.delenv("FLEXTOOL_AXIS_ENUMS", raising=False)
    else:
        monkeypatch.setenv("FLEXTOOL_AXIS_ENUMS", env_value)
    with caplog.at_level(logging.WARNING):
        steps = run_chain_from_db(test_db_url, SCENARIO, work_folder=tmp_path / "work", keep_solutions=True)
    warnings = [r for r in caplog.records if "FLEXTOOL_AXIS_ENUMS=1" in r.getMessage()]
    return list(steps.values()), warnings


def test_env_switch_sets_enums_on_every_sub_solve(test_db_url, tmp_path, caplog, monkeypatch) -> None:
    steps, warnings = _run(test_db_url, tmp_path, caplog, monkeypatch, "1")
    assert len(steps) > 1
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "EXPERIMENTAL" in warnings[0].getMessage()
    for step in steps:
        assert step.flex_data._axis_enums is not None, step.solve_name

    from flextool.spinedb_backend._axis_enums import build_scenario_axis_enums

    expected, _ = build_scenario_axis_enums(test_db_url, SCENARIO)
    for step in steps:
        enums = step.flex_data._axis_enums
        assert step.flex_data_provider.axis_enums is enums, step.solve_name
        assert step.flex_data_provider.contract is not None, step.solve_name
        assert {k: v.categories.to_list() for k, v in enums.items()} == {
            k: v.categories.to_list() for k, v in expected.items()}
        # the loaded frames really carry the Enums (not only the attribute)
        assert isinstance(step.flex_data.dt.schema["d"], pl.Enum), step.solve_name


@pytest.mark.parametrize("env_value", [None, "0"], ids=["unset", "zero"])
def test_without_switch_every_sub_solve_is_utf8(test_db_url, tmp_path, caplog, monkeypatch, env_value) -> None:
    steps, warnings = _run(test_db_url, tmp_path, caplog, monkeypatch, env_value)
    assert warnings == []
    for step in steps:
        assert step.flex_data._axis_enums is None, step.solve_name
        assert step.flex_data_provider.axis_enums is None, step.solve_name
