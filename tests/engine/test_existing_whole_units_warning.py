"""A unit with integer online variables whose existing capacity is not a whole number of units gets a warning.

``existing`` is capacity in MW; with ``virtual_unitsize`` the number of units is ``existing / virtual_unitsize``.
coal_plant (``coal`` alternative) has existing = 500 MW.  With ``startup_method = binary`` and virtual_unitsize
200 MW that is 500 / 200 = 2.5 units, not whole: one warning naming the unit, the count 2.5 and the unit size.
With 250 MW it is 2 units (no warning), and a linear startup has no integer variable (no warning).  Fixtures are
built from ``tests.json`` under the pytest tmp dir.
"""
from __future__ import annotations

import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, make_db, run, uc

pytestmark = pytest.mark.solver


def _warnings(tmp_path_factory, capsys, alt: str, method: str, unitsize: float) -> list[str]:
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN,
                        values=uc(PLANT, 0.0, method) + [["unit", PLANT, "virtual_unitsize", b64(unitsize, "float")]])
    capsys.readouterr()
    run(tmp_path_factory, url, scen)
    return [ln for ln in capsys.readouterr().out.splitlines() if "not a whole number" in ln]


def test_fractional_unit_count_with_integer_online_warns_once(tmp_path_factory, capsys):
    lines = _warnings(tmp_path_factory, capsys, "binary_200", "binary", 200.0)
    assert len(lines) == 1, lines
    assert f"unit '{PLANT}'" in lines[0]
    assert "2.5 (500 MW / 200 MW) units" in lines[0]
    assert "existing is the capacity in MW (MWh for storage)" in lines[0]


@pytest.mark.parametrize("alt, method, unitsize", [("binary_250", "binary", 250.0), ("linear_200", "linear", 200.0)])
def test_whole_unit_count_or_linear_online_does_not_warn(tmp_path_factory, capsys, alt, method, unitsize):
    assert _warnings(tmp_path_factory, capsys, alt, method, unitsize) == []
