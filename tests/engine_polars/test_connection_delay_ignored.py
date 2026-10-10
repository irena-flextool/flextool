"""Connection ``delay`` is not implemented and is ignored, with a warning.

No connection term shifts a flow in time (the delayed input term exists
only in the indirect conversion of units).  The ``delay`` used to mark the
connection delayed anyway: it became ``fork_yes`` (another method), its
arcs moved from ``process_source_sink_eff`` to ``_noEff`` (efficiency
losses dropped) and a ``regular`` connection became one-way — on the
network chain with a 3 h delay on ``west_east`` the objective fell from
585.7 M to 507.5 M.  Now the delay is filtered out where it is read (the
input_derivation delay tables and process-method classifier, the engine's
``_classify_process_method``, delay distributions and ``process_delayed``),
so the connection is modelled exactly as without a delay, and
``validate_connection_delay`` warns once per connection.

Fixture: ``tests.json`` network chain (``init, west, coal, wind,
network``) + an override alternative, built from JSON.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine_polars._unit_db import b64, run, make_db, utf8

pytestmark = pytest.mark.solver

CHAIN = ["init", "west", "coal", "wind", "network"]
CONN = "west_east"
WARNING = ("Connection 'west_east' has a delay: connection delay is not "
           "implemented yet and is ignored")


def _solve(tpf, alt: str, method: str, delay: list | None):
    values = [["connection", CONN, "transfer_method", b64(method, "str")]]
    if delay is not None:
        values.append(["connection", CONN, "delay", delay])
    url, scen = make_db(tpf, alt=alt, values=values, base_chain=CHAIN)
    return list(run(tpf, url, scen).values())[-1]


def _conn_frame(df: pl.DataFrame | None) -> pl.DataFrame:
    if df is None:
        return pl.DataFrame()
    df = utf8(df)
    return df.filter(pl.col("p") == CONN).sort(df.columns)


def _flows(step) -> pl.DataFrame:
    return (utf8(step.solution.value("v_flow"))
            .filter(pl.col("p") == CONN)
            .sort("source", "sink", "d", "t"))


_MAP_3H = b64({"index_type": "float", "rank": 1,
               "data": [[2.0, 0.5], [3.0, 0.5]],
               "index_name": "constraint", "type": "map"}, "map")


@pytest.mark.parametrize("method,delay", [
    ("regular", b64(3.0, "float")),
    ("unidirectional", b64(3.0, "float")),
    ("no_losses_no_variable_cost", b64(3.0, "float")),
    ("regular", _MAP_3H),
], ids=["regular", "unidirectional", "no_losses_no_variable_cost",
        "regular_map"])
def test_connection_delay_is_ignored(tmp_path_factory, caplog,
                                     method: str, delay: list) -> None:
    """The delayed connection solves exactly like the undelayed one (same
    objective, flows, arc classification) and is not in any delay set;
    one warning names it."""
    tag = f"{method}_{'map' if delay is _MAP_3H else 'c'}"
    base = _solve(tmp_path_factory, f"cd_no_{tag}", method, None)
    with caplog.at_level("WARNING"):
        delayed = _solve(tmp_path_factory, f"cd_yes_{tag}", method, delay)

    assert delayed.solution.obj == pytest.approx(base.solution.obj, rel=1e-9)
    fb, fd = _flows(base), _flows(delayed)
    assert fd.height == fb.height > 0
    assert fd.select("source", "sink", "d", "t").equals(
        fb.select("source", "sink", "d", "t"))
    assert fd["value"].to_list() == pytest.approx(
        fb["value"].to_list(), rel=1e-9, abs=1e-9)

    for field in ("process_source_sink_eff", "process_source_sink_noEff",
                  "process_source_sink_2way_1var"):
        assert _conn_frame(getattr(delayed.flex_data, field)).equals(
            _conn_frame(getattr(base.flex_data, field))), field
    for field in ("process_delayed", "process_delayed__duration",
                  "process_source_delayed", "process_source_sink_delayed"):
        assert _conn_frame(getattr(delayed.flex_data, field)).height == 0, \
            field
    pdw = delayed.flex_data.p_process_delay_weight
    if pdw is not None:
        assert _conn_frame(pdw.frame).height == 0

    msgs = [r.getMessage() for r in caplog.records
            if WARNING in r.getMessage()]
    assert len(msgs) == 1, msgs
