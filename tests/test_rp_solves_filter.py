"""Tests for the ``--solves`` subset filter of the RP preprocess.

``preprocess_representative_periods(..., solves=...)`` restricts which solves
get their ``period_timeset`` repointed at the freshly-built RP timeset:

* ``solves=None`` (default) updates every solve carrying a ``period_timeset`` —
  the pre-existing behaviour, asserted here to be byte-identical to a run that
  passes the full solve list explicitly.
* ``solves=["s1"]`` updates only that solve; the others are left untouched (no
  ``period_timeset`` value lands for them in the new alternative).
* A solve name absent from the database raises ``ValueError``.

The fixture is built from the FlexTool JSON schema (CLAUDE.md invariant #3 —
never read a checked-in ``.sqlite``) and carries three solves so the subset
filter has something to discriminate.
"""

from __future__ import annotations

import pytest
from spinedb_api import DatabaseMapping, Map, import_data

from flextool._resources import package_data_path
from flextool.representative_periods.preprocess import (
    preprocess_representative_periods,
)
from flextool.update_flextool import initialize_database

PERIOD_LENGTH = 24
N_PERIODS = 6
SCENARIO = "solves_scen"
ALT = "solves_base"
SOLVES = ["s1", "s2", "s3"]


def _timestep_keys() -> list[str]:
    return [f"t{i}" for i in range(N_PERIODS * PERIOD_LENGTH)]


def _wind_availability() -> list[float]:
    """Per-hour VRE availability with enough shape variation to cluster."""
    values: list[float] = []
    for p in range(N_PERIODS):
        for h in range(PERIOD_LENGTH):
            if p % 2 == 0:
                values.append(1.0 if h < PERIOD_LENGTH // 2 else 0.0)
            else:
                values.append(0.0 if h < PERIOD_LENGTH // 2 else 1.0)
    return values


def _build_controlled_db(db_path: str) -> str:
    """Initialise a schema-complete DB with three solves, each period_timeset."""
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"

    keys = _timestep_keys()
    timeline_map = Map(keys, [1.0] * len(keys))
    wind_map = Map(keys, _wind_availability())
    period_timeset_map = Map(["y2050"], ["placeholder_ts"])

    entities = [
        ("timeline", "tl", None),
        ("profile", "wind", None),
        ("node", "demand", None),
    ]
    parameter_values = [
        ("timeline", "tl", "timestep_duration", timeline_map, ALT),
        ("profile", "wind", "profile", wind_map, ALT),
        ("node", "demand", "inflow", -1000.0, ALT),
    ]
    for solve_name in SOLVES:
        entities.append(("solve", solve_name, None))
        parameter_values.append(
            ("solve", solve_name, "period_timeset", period_timeset_map, ALT)
        )

    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            alternatives=[(ALT, "solves-filter fixture")],
            scenarios=[(SCENARIO, True, "solves scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=entities,
            parameter_values=parameter_values,
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("solves-filter fixture")
    return url


def _solve_timesets_in_alt(url: str, alt: str) -> dict[str, bytes]:
    """Map solve name -> serialised period_timeset bytes written in ``alt``.

    A solve absent from the returned dict had *no* period_timeset value written
    in that alternative (i.e. it was not updated by the preprocess).
    """
    out: dict[str, bytes] = {}
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        for pv in db.find_parameter_values(
            entity_class_name="solve", parameter_definition_name="period_timeset"
        ):
            if pv["alternative_name"] == alt:
                out[pv["entity_name"]] = pv["value"]
    return out


# ---------------------------------------------------------------------------
# 1. Subset: only the named solve is repointed.
# ---------------------------------------------------------------------------

def test_solves_subset_updates_only_named_solve(tmp_path):
    url = _build_controlled_db(str(tmp_path / "subset.sqlite"))
    name = preprocess_representative_periods(
        url, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH, solves=["s1"]
    )
    updated = _solve_timesets_in_alt(url, name)
    assert set(updated) == {"s1"}, (
        f"expected only 's1' repointed in alt '{name}', got {sorted(updated)}"
    )


# ---------------------------------------------------------------------------
# 2. Default (None) == explicit full list, byte-for-byte.
# ---------------------------------------------------------------------------

def test_default_none_matches_explicit_full_list(tmp_path):
    url_none = _build_controlled_db(str(tmp_path / "none.sqlite"))
    url_all = _build_controlled_db(str(tmp_path / "all.sqlite"))

    name_none = preprocess_representative_periods(
        url_none, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH, solves=None
    )
    name_all = preprocess_representative_periods(
        url_all, SCENARIO, n_rp=2, period_length=PERIOD_LENGTH, solves=SOLVES
    )
    assert name_none == name_all

    ts_none = _solve_timesets_in_alt(url_none, name_none)
    ts_all = _solve_timesets_in_alt(url_all, name_all)

    # Default updates every solve, byte-identical to naming them all explicitly.
    assert set(ts_none) == set(SOLVES)
    assert ts_none == ts_all


# ---------------------------------------------------------------------------
# 3. Bogus solve name is a loud error.
# ---------------------------------------------------------------------------

def test_bogus_solve_name_raises(tmp_path):
    url = _build_controlled_db(str(tmp_path / "bogus.sqlite"))
    with pytest.raises(ValueError, match="no_such_solve"):
        preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=2,
            period_length=PERIOD_LENGTH,
            solves=["s1", "no_such_solve"],
        )
