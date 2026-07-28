"""Tests for the ``--alternative-name`` override of the RP preprocess.

``preprocess_representative_periods(..., alternative_name=...)`` overrides the
name of both the output alternative and the ``timeset`` entity it creates:

* ``alternative_name=None`` (default) derives both names from
  ``n_rp``/``period_length`` (plus any force suffix) — asserted byte-identical
  to ``f"hull_{n_rp}rp_{period_length}h{suffix}"``.
* Two runs with DISTINCT ``alternative_name`` values produce two distinct
  alternatives AND two distinct timeset entities that coexist — neither the
  alternative nor the timeset entity (which is NOT alternative-scoped)
  collides, so both scenarios' ``period_timeset`` updates survive.

The fixture is built from the FlexTool JSON schema (CLAUDE.md invariant #3 —
never read a checked-in ``.sqlite``).
"""

from __future__ import annotations

from spinedb_api import DatabaseMapping, Map, import_data

from flextool._resources import package_data_path
from flextool.representative_periods.preprocess import (
    preprocess_representative_periods,
)
from flextool.update_flextool import initialize_database

PERIOD_LENGTH = 24
N_PERIODS = 6
N_RP = 2
SCENARIO = "altname_scen"
ALT = "altname_base"
SOLVES = ["s1", "s2"]


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
    """Initialise a schema-complete DB with two solves, each period_timeset."""
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
            alternatives=[(ALT, "altname fixture")],
            scenarios=[(SCENARIO, True, "altname scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=entities,
            parameter_values=parameter_values,
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("altname fixture")
    return url


def _alternative_names(url: str) -> set[str]:
    with DatabaseMapping(url) as db:
        return {a["name"] for a in db.get_items("alternative")}


def _timeset_entity_names(url: str) -> set[str]:
    with DatabaseMapping(url) as db:
        return {
            e["name"]
            for e in db.get_items("entity")
            if e["entity_class_name"] == "timeset"
        }


def _solve_timesets_in_alt(url: str, alt: str) -> dict[str, bytes]:
    """Map solve name -> serialised period_timeset bytes written in ``alt``."""
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
# 1. Default (None) yields the current derived name (byte-parity).
# ---------------------------------------------------------------------------

def test_default_none_derives_hull_name(tmp_path):
    url = _build_controlled_db(str(tmp_path / "default.sqlite"))
    name = preprocess_representative_periods(
        url, SCENARIO, n_rp=N_RP, period_length=PERIOD_LENGTH
    )
    # No force flags → no suffix.
    expected = f"hull_{N_RP}rp_{PERIOD_LENGTH}h"
    assert name == expected
    assert expected in _alternative_names(url)
    assert expected in _timeset_entity_names(url)


# ---------------------------------------------------------------------------
# 2. Distinct overrides → distinct alternatives AND timeset entities coexist.
# ---------------------------------------------------------------------------

def test_distinct_overrides_do_not_collide(tmp_path):
    # Same DB, two runs with the SAME n_rp/period_length but distinct names.
    url = _build_controlled_db(str(tmp_path / "distinct.sqlite"))

    name_a = preprocess_representative_periods(
        url,
        SCENARIO,
        n_rp=N_RP,
        period_length=PERIOD_LENGTH,
        alternative_name="rp_scenA",
    )
    name_b = preprocess_representative_periods(
        url,
        SCENARIO,
        n_rp=N_RP,
        period_length=PERIOD_LENGTH,
        alternative_name="rp_scenB",
    )

    assert name_a == "rp_scenA"
    assert name_b == "rp_scenB"

    # Both alternatives exist.
    alts = _alternative_names(url)
    assert {"rp_scenA", "rp_scenB"} <= alts

    # Both timeset entities exist (entity is NOT alternative-scoped, so this is
    # the collision that a scenario-independent name would silently overwrite).
    timesets = _timeset_entity_names(url)
    assert {"rp_scenA", "rp_scenB"} <= timesets

    # Both alternatives carry their own full set of solve period_timeset updates.
    ts_a = _solve_timesets_in_alt(url, "rp_scenA")
    ts_b = _solve_timesets_in_alt(url, "rp_scenB")
    assert set(ts_a) == set(SOLVES)
    assert set(ts_b) == set(SOLVES)

    # Each scenario's period_timeset points at its OWN timeset entity.
    for solve in SOLVES:
        assert b"rp_scenA" in ts_a[solve]
        assert b"rp_scenB" in ts_b[solve]
