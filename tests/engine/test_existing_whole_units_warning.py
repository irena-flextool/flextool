"""A unit with integer online variables whose ``existing`` is not a whole number of units gets a warning.

With ``virtual_unitsize`` set, ``existing`` is the number of units (capacity = existing × virtual_unitsize).
coal_plant gets ``startup_method = binary``, virtual_unitsize 200 MW and existing 2.5 in the override alternative:
2.5 units of 200 MW (500 MW, the coal alternative's capacity), not whole, so one warning naming the unit, the count
2.5 and the unit size.  existing 2 (two units, 400 MW) is whole: no warning.  A linear startup has no integer
variable: no warning for 2.5 either.  Fixtures are built from ``tests.json`` under the pytest tmp dir.
"""
from __future__ import annotations

import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, make_db, run, uc

pytestmark = pytest.mark.solver


def _warnings(tmp_path_factory, capsys, alt: str, method: str, existing: float) -> list[str]:
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN,
                        values=uc(PLANT, 0.0, method) + [["unit", PLANT, "virtual_unitsize", b64(200.0, "float")],
                                                         ["unit", PLANT, "existing", b64(existing, "float")]])
    capsys.readouterr()
    run(tmp_path_factory, url, scen)
    return [ln for ln in capsys.readouterr().out.splitlines() if "not a whole number" in ln]


def test_fractional_unit_count_with_integer_online_warns_once(tmp_path_factory, capsys):
    lines = _warnings(tmp_path_factory, capsys, "binary_2_5", "binary", 2.5)
    assert len(lines) == 1, lines
    assert f"unit '{PLANT}'" in lines[0]
    assert "existing gives 2.5 units" in lines[0]
    assert "(200 MW each)" in lines[0]


@pytest.mark.parametrize("alt, method, existing", [("binary_2", "binary", 2.0), ("linear_2_5", "linear", 2.5)])
def test_whole_unit_count_or_linear_online_does_not_warn(tmp_path_factory, capsys, alt, method, existing):
    assert _warnings(tmp_path_factory, capsys, alt, method, existing) == []


# ---------------------------------------------------------------------------
# The solve-side check leaves a unit with a *positive* virtual_unitsize to the input check (existing is the count
# there); a virtual_unitsize <= 0 is ignored everywhere (the unit size is the largest existing), so such a unit is
# checked here like one without virtual_unitsize.  Hand-built frames: coal_plant has integer online variables and a
# count of 0.5 units of 400 MW in p2025 (existing 400 -> 200 MW over the periods).


def _fractional_lines(capsys, vu: float | None) -> list[str]:
    from types import SimpleNamespace

    import polars as pl

    from flextool.engine.input._inmemory_reader import InMemoryReader
    from flextool.engine.preprocess.flex_data import _warn_fractional_integer_unit_counts

    flex_data = SimpleNamespace(
        p_process_existing_count=SimpleNamespace(frame=pl.DataFrame(
            {"p": [PLANT, PLANT], "d": ["p2020", "p2025"], "value": [1.0, 0.5]})),
        process_online_integer=pl.DataFrame({"p": [PLANT]}),
        process_unit=pl.DataFrame({"p": [PLANT]}),
        p_unitsize=SimpleNamespace(frame=pl.DataFrame({"p": [PLANT], "value": [400.0]})))
    parameters = {} if vu is None else {
        ("unit", "virtual_unitsize"): pl.DataFrame({"name": [PLANT], "value": [vu]})}
    reader = InMemoryReader(entities={"unit": pl.DataFrame({"name": [PLANT]})}, parameters=parameters)
    capsys.readouterr()
    _warn_fractional_integer_unit_counts(flex_data, reader)
    return [ln for ln in capsys.readouterr().out.splitlines() if "not a whole number" in ln]


@pytest.mark.parametrize("vu", [None, -5.0, 0.0], ids=["absent", "negative", "zero"])
def test_ignored_virtual_unitsize_is_checked_at_the_solve(capsys, vu):
    lines = _fractional_lines(capsys, vu)
    assert len(lines) == 1, lines
    assert f"unit '{PLANT}'" in lines[0] and "makes 0.5 in p2025 units of 400 MW," in lines[0]


def test_positive_virtual_unitsize_is_left_to_the_input_check(capsys):
    assert _fractional_lines(capsys, 200.0) == []
