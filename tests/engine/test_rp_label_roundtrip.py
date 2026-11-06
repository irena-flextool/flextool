"""Regression: representative-period label columns must survive the
``load_flextool`` axis-enum sweep on the production load.

Root cause (fixed in ``_axis_enums._FLEXDATA_SWEEP_SKIP_COLUMNS``): the RP
base-period / rep-start relations name their label columns ``"b"`` (base
period start) and ``"r"`` (rep-block start) so ``model/core.py``'s inter-period
closure wiring can join on the ``.mod`` symbols — but those labels are
TIMESTEP tokens (``t0001`` …), NOT branch / rep-axis tokens.  The
end-of-load ``cast_flexdata_axes`` sweep resolves ``"b" → branch`` and
treats ``"r"`` as the (rep) axis via the synonym table, so a blind cast
nulled every value (branch vocab is e.g. ``{eff, noEff}``; the rep axis
enum is empty for a non-stochastic solve).  A null ``rp_base_period_set.b``
collapses ``v_state_inter``'s index → the seasonal-closure constraints
(``rp_inter_period_cyclic`` / ``_balance``) are silently dropped → the LP
solves an under-constrained, too-cheap problem.

This test loads the last sub-solve with the production call (provider +
DB reader, :func:`tests._support.load.load_scenario`) with the axis-Enum
vocabulary activated (``axis_enums=True``: the sweep is what collided;
production builds no Enums, c24 design_stage1 F2), and pins that the RP
labels equal the emitted ones and that the seasonal-closure constraint
fires with a non-empty index.  The disk-CSV reload and the disk-vs-provider
agreement check are in ``tests/csv_reload/test_rp_label_roundtrip.py``
(c24 S1-8).
"""
from __future__ import annotations

import polars as pl

from polar_high import Problem

from flextool.engine import build_flextool
from tests._support.load import load_scenario


def _closure_rows(data) -> int:
    pb = Problem()
    build_flextool(pb, data)
    total = 0
    for n in pb.cstr_names():
        if n.startswith("rp_inter_period_cyclic") or n.startswith(
            "rp_inter_period_balance"
        ):
            total += pb.cstr_row_count(n)
    return total


def _emitted_labels(provider, key: str, col: str) -> set[str]:
    frame = provider.get(key)
    assert frame is not None, f"{key} not in the post-solve provider"
    return set(frame[col].cast(pl.Utf8).to_list())


def test_rp_labels_survive_production_load(scenario_run):
    run = scenario_run("lh2_three_region_rp_invest", db_fixture="lh2_rp_invest")

    # The emitted frames DO carry the labels (so any nulling is a
    # read/cast bug, not an emit bug).
    emitted_base = _emitted_labels(run.last.provider, "solve_data/rp_base_first", "base_start")
    emitted_rep = _emitted_labels(run.last.provider, "solve_data/rp_weights", "rep_start")
    assert emitted_base and all(b is not None for b in emitted_base)
    assert emitted_rep and all(r is not None for r in emitted_rep)

    data = load_scenario(run, axis_enums=True)

    # --- The RP label columns round-trip NON-NULL and equal to the emit.
    def labels(field, col):
        v = getattr(data, field)
        frame = getattr(v, "frame", v)
        assert frame is not None and frame.height > 0, f"{field} empty"
        s = frame[col]
        assert s.null_count() == 0, (
            f"{field}.{col} has {s.null_count()} null label(s) after load "
            f"(the branch/rep-axis cast collision regressed)"
        )
        return set(s.cast(pl.Utf8).to_list())

    assert labels("rp_base_first", "b") == emitted_base
    assert labels("rp_base_last", "b") == emitted_base
    assert labels("rp_base_period_set", "b")  # non-null, non-empty
    rep_labels = labels("rp_base__rep", "r")
    assert rep_labels == emitted_rep
    assert labels("rp_base__rep", "b") == emitted_base
    assert labels("p_rp_last_step", "r") == emitted_rep

    # --- Because the labels survive, the seasonal-closure constraint
    # fires with a non-empty index (it would be dropped entirely if any
    # of the tightly-coupled RP frames nulled out).
    assert _closure_rows(data) > 0, (
        "rp_inter_period seasonal-closure constraint did not fire — a "
        "nulled RP label collapsed v_state_inter's index"
    )
    # The Enum build went onto the clone, not the session-cached provider.
    assert run.last.provider.axis_enums is None
