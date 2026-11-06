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
