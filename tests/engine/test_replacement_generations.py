"""The generations of a ``reinvest_automatic`` chain in the cohort table (``_cohorts.chain_rows``), on hand-derived
cases.

Model: ``tests.json``, the single solve ``y2020_2035_5week`` over p2020 … p2035 of 5 years each: years 0, 5, 10,
15, horizon end 20, seams {0, 5, 10, 15, 20}.  coal_plant has 500 MW existing and the default end_of_life_method
closest_seam (a tie goes to the later seam).  A generation ``gen:<root>:<k>`` comes online at ``y_on`` (g1: the
root's start; ``g_{k+1}``: ``y_eol`` of ``g_k``) and ends at ``y_eol = closest_seam(y_on + L)``, the next seam when
that is at or before ``y_on`` (every plant serves a period), until the end of the horizon (past the last seam:
the horizon end, 20).

* invested in p2020, L 12: 12 → 10 (2 < 3): [0, 10); 10 + 12 = 22 → 20: [10, 20).
* L 2: 0 + 2 → 0 (2 < 3), at its own start → the next seam 5: [0, 5), [5, 10), [10, 15), [15, 20).
* invested in p2025, L 10: [5, 15), [15, 20) (15 + 10 = 25 → 20).
* no lifetime_technical: one generation, never ending (+inf).
* existing, L 10, the plant in service at the run start is the existing capacity less the replacements before the
  run (k = ⌊age / L⌋, age_eff = age − k·L, y_on = −age_eff); a replacement the end_of_life_method dates at the run
  start (the seams before it continue by the first period's length: …, −5, 0) is in the model, at 0:
  - age 23: k 2, age_eff 3, y_on −3; −3 rounds to −5 (2 < 3), before the run; ends 7 → 5: [−3, 5), [5, 15),
    [15, 20);
  - age 9.9: k 0, ends 0.1 → 0 (it serves no period): g1 [−9.9, 0), g2 [0, 10), g3 [10, 20);
  - age 10: k 1, age_eff 0: the replacement at 0 is in the model: g1 [0, 0), g2 [0, 10), g3 [10, 20);
  - age 10.1: k 1, age_eff 0.1, −0.1 rounds to 0 (0.1 < 4.9): g1 [−0.1, 0), g2 [0, 10), g3 [10, 20);
  - age 12.6: k 1, age_eff 2.6, −2.6 rounds to −5 (2.4 < 2.6): sunk, ends 7.4 → 5: [−2.6, 5), [5, 15), [15, 20).
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine.preprocess.params._cohort_handoff import existing_capacity, realized_cohorts
from flextool.engine.preprocess.params._cohorts import (
    KIND_GENERATION,
    _cohort_frame,
    build_cohort_frames,
    cohort_frame_from_utf8,
)
from tests.engine._unit_db import PLANT, b64, make_db
from tests.engine.test_user_constraint_capacity_terms import SINGLE

SOLVE = "y2020_2035_5week"
PERIODS = ["p2020", "p2025", "p2030", "p2035"]
INF = math.inf

_READERS: dict[str, object] = {}


#: The single-solve chain without coal_invest, which gives coal_plant a lifetime_technical of 30.
_WITHOUT_LIFETIME = [a for a in SINGLE if a != "coal_invest"]


def _reader(tmp_path_factory, alt: str, values: list):
    """A reader of ``tests.json`` + ``values`` (one DB per case); the ``*_nolife`` cases leave out the alternative
    that sets coal_plant's lifetime."""
    from flextool.engine import SpineDbReader
    if alt not in _READERS:
        chain = _WITHOUT_LIFETIME if alt.endswith("_nolife") else SINGLE
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
        _READERS[alt] = SpineDbReader(url, scen)
    return _READERS[alt]


@pytest.fixture(scope="module", autouse=True)
def _free_readers():
    yield
    _READERS.clear()


def _automatic(life: float | None, age: float | None = None) -> list:
    values = [["unit", PLANT, "lifetime_method", b64("reinvest_automatic", "str")]]
    if life is not None:
        values.append(["unit", PLANT, "lifetime_technical", b64(life, "float")])
    if age is not None:
        values.append(["unit", PLANT, "age_at_model_start", b64(age, "float")])
    return values


def _generations(frames, root: str) -> list[tuple]:
    rows = (frames.cohort.filter((pl.col("e").cast(pl.Utf8) == PLANT) & (pl.col("kind") == KIND_GENERATION)
                                 & (pl.col("root") == root))
                         .sort("y_on"))
    return rows.select("cohort", "d_on", "y_on", "y_eol", "d_eol").rows()


def _decided(d: str) -> pl.DataFrame:
    return pl.DataFrame({"e": [PLANT], "d": [d]})


@pytest.mark.parametrize("alt,life,decided,want", [
    ("gen_l12", 12.0, "p2020", [("gen:invest:p2020:1", "p2020", 0.0, 10.0, "p2030"),
                                ("gen:invest:p2020:2", "p2030", 10.0, 20.0, None)]),
    ("gen_l2", 2.0, "p2020", [("gen:invest:p2020:1", "p2020", 0.0, 5.0, "p2025"),
                              ("gen:invest:p2020:2", "p2025", 5.0, 10.0, "p2030"),
                              ("gen:invest:p2020:3", "p2030", 10.0, 15.0, "p2035"),
                              ("gen:invest:p2020:4", "p2035", 15.0, 20.0, None)]),
    ("gen_l10", 10.0, "p2025", [("gen:invest:p2025:1", "p2025", 5.0, 15.0, "p2035"),
                                ("gen:invest:p2025:2", "p2035", 15.0, 20.0, None)]),
    ("gen_nolife", None, "p2020", [("gen:invest:p2020:1", "p2020", 0.0, INF, None)]),
])
def test_investment_chain(tmp_path_factory, alt, life, decided, want) -> None:
    reader = _reader(tmp_path_factory, alt, _automatic(life))
    frames = build_cohort_frames(reader, SOLVE, _decided(decided), PERIODS, None)
    assert _generations(frames, f"invest:{decided}") == want
    # The root keeps the whole chain's window, and only the root is in service.
    root = frames.cohort.filter((pl.col("e").cast(pl.Utf8) == PLANT) & (pl.col("kind") == "invest"))
    assert root.select("cohort", "y_eol", "root").rows() == [(f"invest:{decided}", INF, f"invest:{decided}")]
    assert set(frames.cohort_alive["kind"].unique().to_list()) <= {"existing", "invest"}


@pytest.mark.parametrize("alt,age,want", [
    ("gen_age23", 23.0, [(-3.0, 5.0), (5.0, 15.0), (15.0, 20.0)]),
    ("gen_age9_9", 9.9, [(-9.9, 0.0), (0.0, 10.0), (10.0, 20.0)]),
    ("gen_age10", 10.0, [(0.0, 0.0), (0.0, 10.0), (10.0, 20.0)]),
    ("gen_age10_1", 10.1, [(-0.1, 0.0), (0.0, 10.0), (10.0, 20.0)]),
    ("gen_age12_6", 12.6, [(-2.6, 5.0), (5.0, 15.0), (15.0, 20.0)]),
])
def test_existing_chain_starts_from_the_plant_in_service(tmp_path_factory, alt, age, want) -> None:
    reader = _reader(tmp_path_factory, alt, _automatic(10.0, age))
    frames = build_cohort_frames(reader, SOLVE, None, PERIODS, None)
    got = [y for _c, _d, y_on, y_eol, _de in _generations(frames, "existing:p2020") for y in (y_on, y_eol)]
    assert got == pytest.approx([y for pair in want for y in pair], abs=1e-9)
    # The existing root is unchanged: in service in every period.
    alive = frames.cohort_alive.filter((pl.col("e").cast(pl.Utf8) == PLANT) & (pl.col("kind") == "existing"))
    assert [str(d) for d in alive["d"].to_list()] == PERIODS


def test_existing_chain_without_lifetime_is_one_generation(tmp_path_factory) -> None:
    reader = _reader(tmp_path_factory, "gen_ex_nolife", _automatic(None))
    frames = build_cohort_frames(reader, SOLVE, None, PERIODS, None)
    assert _generations(frames, "existing:p2020") == [("gen:existing:p2020:1", "p2020", 0.0, INF, None)]


def _table(rows: list[tuple]) -> pl.DataFrame:
    """A cohort table of ``rows`` = (cohort, kind, d_build, d_on, y_on, y_life, y_eol, d_eol, a_cap, root)."""
    full = [(c, "plant", kind, "reinvest_automatic", "closest_seam", d_build, d_on, y_on, 0.0, life,
             None if life is None else y_on + life, None, y_eol, d_eol, a_cap, None, None, None, 0.0, None, root,
             None)
            for c, kind, d_build, d_on, y_on, life, y_eol, d_eol, a_cap, root in rows]
    return _cohort_frame(full)


CHAIN = _table([
    ("invest:p2020", "invest", "p2020", "p2020", 0.0, INF, INF, None, 52_000.0, "invest:p2020"),
    ("gen:invest:p2020:1", KIND_GENERATION, "p2020", "p2020", 0.0, 12.0, 10.0, "p2030", 45_130.0, "invest:p2020"),
    ("gen:invest:p2020:2", KIND_GENERATION, "p2020", "p2030", 10.0, 12.0, 20.0, None, 40_000.0, "invest:p2020"),
])


def test_realized_chain_carries_its_generations_by_name() -> None:
    """Solve 1 realizes 2 units (unitsize 50 MW) of the p2020 decision: the carrier holds the root (100 MW) and its
    two generations as solve 1 placed and priced them, size null; solve 2, which has none of these rows in its own
    table, carries them unchanged.  The capacity a later solve starts from counts the root only: 100 MW in p2030,
    where generation 2 holds it."""
    c1, _ = realized_cohorts(None, None, CHAIN, {("plant", "p2020"): 2.0}, {}, {"plant": 50.0})
    assert c1.select("cohort", "kind", "period_online", "y_on", "y_eol", "a_cap", "root", "size").rows() == [
        ("invest:p2020", "invest", "p2020", 0.0, INF, 52_000.0, "invest:p2020", 100.0),
        ("gen:invest:p2020:1", KIND_GENERATION, "p2020", 0.0, 10.0, 45_130.0, "invest:p2020", None),
        ("gen:invest:p2020:2", KIND_GENERATION, "p2030", 10.0, 20.0, 40_000.0, "invest:p2020", None)]
    c2, _ = realized_cohorts(c1, None, _table([]), {}, {}, {"plant": 50.0})
    assert c2.rows() == c1.rows()
    existing = _table([("existing:p2020", "existing", None, "p2020", 0.0, INF, INF, None, None, "existing:p2020")])
    cap = existing_capacity(["plant"], ["p2025", "p2030"], {"p2025": 5.0, "p2030": 10.0}, existing,
                            {("plant", "p2025"): 0.0, ("plant", "p2030"): 0.0}, c2, None, [])
    assert cap.carried == {("plant", "p2025"): 100.0, ("plant", "p2030"): 100.0}
    assert cap.invested == {"plant": 100.0}


def test_cohort_table_with_generations_reads_back_from_its_csv_dump(tmp_path) -> None:
    """The cohort table written to a csv (the Provider dump) and read back (column types inferred) is the table:
    +inf ends, null periods and prices, and the root column survive."""
    path = tmp_path / "cohort.csv"
    CHAIN.with_columns(pl.col("e").cast(pl.Utf8), pl.col("d_build").cast(pl.Utf8)).write_csv(path)
    back = cohort_frame_from_utf8(pl.read_csv(path))
    assert back.schema == CHAIN.schema
    assert back.rows() == CHAIN.rows()
