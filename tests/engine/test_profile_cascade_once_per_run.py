"""The profile cascade reads the scalar / period / period x time profiles once per run (c8 review R1).

The synthetic path (rolling rolls, nested sub-solves) runs ``apply_profile_cascade`` in every roll with the run's
``SpineDbReader``.  The time tier comes from the per-roll ``pt_profile.csv``; the other tiers read the DB.  They must
unroll only their own profiles (never the long time-series ones) and only once per reader:

    rows unrolled per run = rows of the scalar + period + period x time profiles      (independent of #rolls)

The database is built in the pytest tmp dir with ``spinedb_api`` (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import polars as pl
import pytest

import spinedb_api as api
from spinedb_api import DatabaseMapping

from flextool.engine import SpineDbReader
from flextool.engine.preprocess.params._profiles import p_profile_value_lf

T = [f"t{i:04d}" for i in range(1, 25)]
P = ["p2020", "p2025"]
N_TS = 5


def _add(db, name: str, value) -> None:
    db.add_entity(entity_class_name="profile", name=name)
    v, t = api.to_database(value)
    db.add_parameter_value(entity_class_name="profile", entity_byname=(name,), parameter_definition_name="profile",
                           alternative_name="Base", value=v, type=t)


@pytest.fixture
def reader(tmp_path) -> SpineDbReader:
    url = f"sqlite:///{tmp_path / 'profiles.sqlite'}"
    db = DatabaseMapping(url, create=True)
    db.add_entity_class(name="profile")
    db.add_parameter_definition(entity_class_name="profile", name="profile")
    db.add_scenario(name="s")
    db.add_scenario_alternative(scenario_name="s", alternative_name="Base", rank=1)
    for n in range(N_TS):  # time series first: they decided the index columns of the old whole-table read
        _add(db, f"ts{n}", api.Map(T, [0.5] * len(T), index_name="time"))
    _add(db, "per", api.Map(P, [0.2, 0.4], index_name="period"))
    _add(db, "pt", api.Map(P, [api.Map(T, [0.3] * len(T), index_name="time") for _ in P], index_name="period"))
    _add(db, "const", 0.7)
    db.commit_session("profiles")
    db.close()
    return SpineDbReader(url, "s")


def _dt() -> pl.DataFrame:
    return pl.DataFrame({"d": [d for d in P for _ in T], "t": [t for _ in P for t in T]})


def _values(lf: pl.LazyFrame) -> dict:
    out = lf.collect().with_columns(pl.col("f", "d", "t").cast(pl.Utf8))
    return {(f, d, t): v for f, d, t, v in out.iter_rows()}


def test_non_time_tiers_unroll_only_their_profiles_once(reader, monkeypatch) -> None:
    unrolled: list[int] = []
    orig = SpineDbReader._unroll_rows

    def spy(self, rows, *args, **kwargs):
        unrolled.append(len(rows))
        return orig(self, rows, *args, **kwargs)

    monkeypatch.setattr(SpineDbReader, "_unroll_rows", spy)
    first = _values(p_profile_value_lf(reader, _dt()))
    # scalar, period and period x time tiers: one profile each; the time tier falls back to the DB (no workdir)
    # and reads the N_TS time series, not memoised
    assert sorted(unrolled) == [1, 1, 1, N_TS], unrolled
    unrolled.clear()
    for _roll in range(3):
        assert _values(p_profile_value_lf(reader, _dt())) == first
    assert unrolled == [N_TS] * 3, unrolled  # only the (uncached) time fallback; never the non-time profiles again

    for d, share in zip(P, [0.2, 0.4]):
        for t in T:
            assert first[("per", d, t)] == pytest.approx(share)
            assert first[("pt", d, t)] == pytest.approx(0.3)
            assert first[("const", d, t)] == pytest.approx(0.7)
            assert first[("ts0", d, t)] == pytest.approx(0.5)


def test_period_profile_next_to_time_series_without_2d(tmp_path) -> None:
    """A period profile next to time-series profiles (no period x time one) reads its values: the period tier no
    longer depends on which entity's shape the whole-table read discovered first."""
    url = f"sqlite:///{tmp_path / 'p.sqlite'}"
    db = DatabaseMapping(url, create=True)
    db.add_entity_class(name="profile")
    db.add_parameter_definition(entity_class_name="profile", name="profile")
    db.add_scenario(name="s")
    db.add_scenario_alternative(scenario_name="s", alternative_name="Base", rank=1)
    _add(db, "ts0", api.Map(T, [0.5] * len(T), index_name="time"))
    _add(db, "per", api.Map(P, [0.2, 0.4], index_name="period"))
    db.commit_session("profiles")
    db.close()
    got = _values(p_profile_value_lf(SpineDbReader(url, "s"), _dt()))
    assert got[("per", "p2020", "t0001")] == pytest.approx(0.2)
    assert got[("per", "p2025", "t0024")] == pytest.approx(0.4)
    assert got[("ts0", "p2025", "t0024")] == pytest.approx(0.5)


def test_parameter_for_entities_matches_parameter(reader) -> None:
    names = ["per"]
    sub = reader.parameter_for_entities("profile", "profile", names)
    assert sub.columns == ["name", "period", "value"]
    assert sub.sort("period")["value"].to_list() == [0.2, 0.4]
    cached = reader.parameter_for_entities("profile", "profile", names, cache=True)
    assert reader.parameter_for_entities("profile", "profile", set(names), cache=True) is cached
    whole = reader.parameter("profile", "profile").filter(pl.col("name") == "const")
    assert reader.parameter_for_entities("profile", "profile", ["const"])["value"].to_list() == \
        whole["value"].to_list()
