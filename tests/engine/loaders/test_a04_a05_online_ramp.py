"""Surface A.4 (Online / Unit-Commitment / Startup Constraints) and A.5
(Ramp-Rate Constraints) loader tests.

Entry point: ``flextool.engine.load_flextool`` (through ``load_scenario``).  These sections are
gated by ``pss`` (process_source_sink) presence, so each test starts from
a real scenario that wires up online / ramp data (``coal_min_load``,
``coal_ramp_limit``) or from ``base`` (no process, pss absent).

Production-path tests (provider clone + DB reader, ``load_scenario``):

* A5-empty_when_pss_none on ``base`` (``tiny_run``).
* A5-all_four_directions_independent: the ramp direction sets are
  ``SIMPLE_PROJECTIONS`` of the DB (``params/_projection.py``) that keep
  the provider seed when the DB has no member, so an overlay on a
  direction the DB leaves empty is what the model sees.
* ``pdt_online_zero_startup_cost_excluded`` calls ``_load_online``
  directly with a stub source returning ``startup_cost=0`` and the
  ``coal_min_load`` provider (the ``filter(value != 0)`` branch is
  reachable only with a source).

The reload-only online tests and the all-empty / ramp-speed parts of the
ramp blank-path test (the DB re-derives those sets on every load) are in
``tests/csv_reload/test_a04_a05_online_ramp.py`` (c24 S1-8).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests._support.fields import field_diff
from tests._support.load import ScenarioRun, clone_provider, load_scenario

from .conftest import drop_frame, put_frame


@pytest.fixture(scope="function")
def online_run(scenario_run) -> tuple[ScenarioRun, object]:
    """The ``coal_min_load`` run (1 online process with min_load_eff) and a fresh clone of its provider."""
    run = scenario_run("coal_min_load")
    return run, clone_provider(run.last.provider)


@pytest.fixture(scope="function")
def ramp_run(scenario_run) -> tuple[ScenarioRun, object]:
    """The ``coal_ramp_limit`` run (one ``sink_up`` ramp arc) and a fresh clone of its provider."""
    run = scenario_run("coal_ramp_limit")
    return run, clone_provider(run.last.provider)


# --- A.4 ------------------------------------------------------------


def test_pdt_online_zero_startup_cost_excluded(online_run):
    """Covers A4-pdt_online_zero_startup_cost_excluded (direct).

    Stub ``InputSource`` returns ``startup_cost = 0`` for the lone
    process; ``_load_online`` must filter that row before constructing
    the ``pdt_online_*`` join sets, leaving them and ``p_startup_cost``
    as ``None``.  Called directly because the CSV path skips the
    source helper entirely (``source=None``).  The online sets come from
    the ``coal_min_load`` provider; ``inp`` / ``sd`` point into the empty
    load folder, so no file is read.
    """
    from flextool.engine.preprocess.flex_data import _load_online

    run, P = online_run
    inp = run.load_folder / "input"
    sd = run.load_folder / "solve_data"
    pss = pl.DataFrame({"p": ["coal_plant"],
                        "source": ["coal_market"], "sink": ["west"]})
    dt = pl.DataFrame({"d": ["p2020"], "t": ["t0001"]})

    class _ZeroStartupSource:
        # ``parameter`` is the core InputSource read; ``parameter_explicit``
        # the optional refinement.  Both expose only the lone zero-startup
        # row — every other ``(ec, name)`` raises ``KeyError`` so callers
        # (e.g. Carrier 2's ``_process_min_time`` probe for min_uptime /
        # min_downtime) see "no such parameter" and contribute nothing.
        def parameter(self, ec: str, name: str) -> pl.DataFrame:
            if (ec, name) == ("unit", "startup_cost"):
                # Hand-calc: explicit zero row triggers filter(value != 0).
                return pl.DataFrame({"name": ["coal_plant"],
                                     "period": ["p2020"],
                                     "value": [0.0]})
            raise KeyError((ec, name))

        def parameter_explicit(self, ec: str, name: str) -> pl.DataFrame:
            return self.parameter(ec, name)

    out = _load_online(inp, sd, dt, pss, source=_ZeroStartupSource(), provider=P)
    # The provider's online set is read (not the blank early return), so the Nones below come from
    # the zero-startup filter.
    assert out["process_online"]["p"].to_list() == ["coal_plant"]
    assert out["process_online_linear"]["p"].to_list() == ["coal_plant"]
    # Hand-calc: filter removes the only sc row -> sc_frame.height == 0
    # -> pdt_online_lin / p_startup_cost left at their initial None.
    assert out["pdt_online_linear"] is None
    assert out["pdt_online_integer"] is None
    assert out["p_startup_cost"] is None


# --- A.5 ------------------------------------------------------------


def test_ramp_blank_when_pss_none(tiny_run):
    """Covers A5-empty_when_pss_none (direct) on the production path.

    ``base`` has no process, so pss is None and ``_load_ramp`` returns the
    blank dict: every ramp set and every ``p_ramp_speed_*`` Param is None.
    No overlay; the ``coal_ramp_limit`` load below shows the same fields
    are populated when a ramp arc exists.
    """
    run, P = tiny_run
    data = load_scenario(run, provider=P)
    assert data.process_source_sink is None
    for key in ("process_source_sink_ramp_limit_sink_up",
                "process_source_sink_ramp_limit_sink_down",
                "process_source_sink_ramp_limit_source_up",
                "process_source_sink_ramp_limit_source_down",
                "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
                "p_ramp_speed_up_source", "p_ramp_speed_down_source"):
        assert getattr(data, key) is None, f"{key} not None on the base seed"


def test_ramp_directions_independent(ramp_run):
    """Covers A5-all_four_directions_independent (direct).

    The ``coal_ramp_limit`` provider holds the four direction sets
    (``sink_up`` = the one ramp arc, the other three empty).  The overlay
    removes all four keys and puts back *only* ``source_down`` populated.
    Per ``_read_set``, a missing key -> None; this asserts no cross-fill
    between directions.

    Production path: the direction sets are DB projections that keep the
    provider seed when the DB has no member.  The DB has no ``source_down``
    arc, so the overlay is what the model sees; ``sink_up`` is re-derived
    from the DB (the scenario's ramp arc) although its key was removed.
    """
    run, P = ramp_run
    for dirn in ("sink_up", "sink_down", "source_up", "source_down"):
        drop_frame(P, f"process_source_sink_ramp_limit_{dirn}.csv")
    put_frame(P, "process_source_sink_ramp_limit_source_down.csv",
              "process,source,sink\ncoal_plant,coal_market,west\n")
    data = load_scenario(run, provider=P)
    # Hand-calc: only source_down populated by the overlay; sink_down and
    # source_up stay None — no accidental fill from a populated direction.
    assert data.process_source_sink_ramp_limit_source_down.height == 1
    assert data.process_source_sink_ramp_limit_source_down["p"].to_list() == [
        "coal_plant"]
    assert data.process_source_sink_ramp_limit_sink_down is None
    assert data.process_source_sink_ramp_limit_source_up is None
    # The three directions the overlay did not populate equal the un-overlaid
    # load whole (every column, every row): sink_up is the DB's own ramp arc,
    # so a cross-fill of the source_down row into sink_up (same p, other
    # source/sink) fails here.
    for dirn in ("sink_up", "sink_down", "source_up"):
        field = f"process_source_sink_ramp_limit_{dirn}"
        assert field_diff(getattr(data, field), getattr(run.last_flex_data, field)) is None, field
    assert data.process_source_sink_ramp_limit_sink_up.height == 1
    # The overlay is consumed: the un-overlaid load has no source_down set.
    assert run.last_flex_data.process_source_sink_ramp_limit_source_down is None
