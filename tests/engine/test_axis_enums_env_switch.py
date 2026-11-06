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


def _as_utf8_rows(frame: pl.DataFrame) -> list[tuple]:
    out = frame.with_columns(pl.col(c).cast(pl.Utf8) for c, t in frame.schema.items() if isinstance(t, pl.Enum))
    return sorted(out.rows(), key=lambda r: tuple(str(v) for v in r))


def test_env_switch_keeps_the_derived_carriers_of_the_utf8_run(test_db_url, tmp_path, caplog, monkeypatch) -> None:
    """Axis-Enum mode builds the same structural flow caps and seeds the same entity unitsize carrier.

    Both used to fail in Enum mode inside a ``try`` that swallowed the error: the flow caps were left unset
    and the solve-invariant unitsize carrier (``input/p_all_entity_unitsize``) was never seeded.  The Utf8
    run comes second in the same process: the Enum run's vocabulary must not stay live and type its frames
    (it did, and the Utf8 run crashed on two different Enums).
    """
    enum_steps, _ = _run(test_db_url, tmp_path / "enum", caplog, monkeypatch, "1")
    utf8_steps, _ = _run(test_db_url, tmp_path / "utf8", caplog, monkeypatch, None)
    assert [s.solve_name for s in enum_steps] == [s.solve_name for s in utf8_steps]
    for enum_step, utf8_step in zip(enum_steps, utf8_steps):
        name = enum_step.solve_name
        assert utf8_step.flex_data.dt.schema["d"] == pl.Utf8, name
        utf8_cap = utf8_step.flex_data.p_flow_upper
        assert utf8_cap is not None and utf8_cap.frame.height > 0, name
        enum_cap = enum_step.flex_data.p_flow_upper
        assert enum_cap is not None, name
        assert isinstance(enum_cap.frame.schema["d"], pl.Enum), name
        assert _as_utf8_rows(enum_cap.frame) == _as_utf8_rows(utf8_cap.frame), name

        utf8_us = utf8_step.flex_data_provider.get("input/p_all_entity_unitsize")
        enum_us = enum_step.flex_data_provider.get("input/p_all_entity_unitsize")
        assert utf8_us is not None and utf8_us.height > 0, name
        assert enum_us is not None, name
        assert isinstance(enum_us.schema["e"], pl.Enum), name
        assert _as_utf8_rows(enum_us) == _as_utf8_rows(utf8_us), name
