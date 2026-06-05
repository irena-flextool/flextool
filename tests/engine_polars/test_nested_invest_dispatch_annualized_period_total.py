"""Pin the annualized per-period DISPATCH output of a nested invest→dispatch
run against hand-calculated numbers from an extremely simple, fully-forced
model.

Why this test exists (the bug it catches)
------------------------------------------
In a nested run an ``invest`` sub-solve samples a *representative* timeline
window for a period (here 4 of 8 hours) and a sibling/contained ``dispatch``
sub-solve realizes the SAME period over its *full* timeline (here 8 hours).
Per-period DISPATCH outputs are annualized as::

    period_total = Σ_t(power · step_duration) ÷ complete_period_share_of_year[period]
    complete_period_share_of_year[d] = complete_hours_in_period[d] / 8760

Both sub-solves persist a realized-slice row for the period, and
``process_outputs/drop_levels.py`` deduplicates the unioned
``complete_period_share_of_year`` by period.  The dispatch VARIABLES are
deduped ``keep='last'`` (the dispatch sub-solve unions after the invest
sub-solve), so the annualization PARAMETER must be deduped ``keep='last'``
too — sourced from the SAME (realized-dispatch) sub-solve.  The bug was that
``complete_period_share_of_year`` lived in the ``keep='first'`` bucket, so the
dispatch flows got annualized with the invest window's tiny share, inflating
every period total by ``8760 / (sampled hours)`` (a 121.7× over-count on the
3-day invest fixture that originally surfaced this).

The existing nested-invest goldens only pinned ``*__dt`` (hourly,
un-annualized) + ``unit_capacity__d`` + objective parity — never the
annualized ``*__d`` energy totals — so the bug shipped.  This test closes
that gap with a model whose every number is a single multiplication.

Hand calculation
----------------
* One balance node ``demand`` with a constant inelastic demand of 100 MW
  every step (inflow = -100).
* One generator ``gen`` (pure source, efficiency 1, existing capacity 200 MW,
  no profile) feeding ``demand`` at ``other_operational_cost`` = 5 CUR/MWh.
  Generation is forced to exactly meet demand at every step → deterministic.
* One period ``p2020``.
* ``mini_dispatch`` realizes ``p2020`` over the FULL 8-step timeline
  (8 × 1 h = 8 complete hours).
* ``mini_invest`` samples only a 4-step window (4 × 1 h = 4 hours) and
  realizes invest for ``p2020``.  The two windows DIFFER (8 vs 4) so the two
  ``complete_period_share_of_year`` values differ and the bug is exercised.

  Dispatch energy in the realized 8 h window  = 100 MW · 8 h        = 800 MWh
  Dispatch complete_period_share_of_year      = 8 / 8760
  CORRECT annualized period total             = 800 ÷ (8/8760)
                                              = 100 · 8760           = 876 000 MWh
  Annual operational cost                     = 876 000 · 5          = 4 380 000 CUR
                                              = 4.38 M CUR
  "Time in use in years"                      = 8 / 8760             ≈ 0.000913242

  Bug value (invest window 4 h, share 4/8760) = 800 ÷ (4/8760)
                                              = 100 · 2 · 8760       = 1 752 000 MWh (2×)

The model is built from the FlexTool schema JSON + ``import_data`` at test
time (CLAUDE.md invariant #3 — never read a checked-in ``.sqlite``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.solver


# ── Hand-calculated expectations ─────────────────────────────────────────
DEMAND_MW = 100.0
DISPATCH_HOURS = 8          # mini_dispatch realizes 8 full steps of p2020
INVEST_HOURS = 4            # mini_invest samples a 4-step window of p2020
OPERATIONAL_COST = 5.0      # CUR / MWh

# CORRECT: annualize the dispatch window energy with the DISPATCH share.
EXPECTED_PERIOD_ENERGY = DEMAND_MW * 8760.0           # 876 000 MWh
EXPECTED_ANNUAL_COST = EXPECTED_PERIOD_ENERGY * OPERATIONAL_COST  # 4 380 000 CUR
EXPECTED_TIME_IN_USE_YEARS = DISPATCH_HOURS / 8760.0  # 8/8760 ≈ 0.000913242

# BUG: annualizing with the INVEST window share over-counts by 8760/INVEST_HOURS
# on the raw window energy, i.e. DISPATCH_HOURS/INVEST_HOURS = 2× the correct
# period total.  Asserted-against as a guard so a regression is unambiguous.
BUG_PERIOD_ENERGY = (DEMAND_MW * DISPATCH_HOURS) / (INVEST_HOURS / 8760.0)  # 1 752 000


def _build_mini_nested_db(db_path: Path) -> str:
    """Build the minimal nested invest→dispatch scenario DB from schema JSON.

    Uses ``initialize_database`` (empty DB with the full FlexTool schema)
    then ``import_data`` for the handful of scenario entities/params — no
    checked-in ``.sqlite`` is read (CLAUDE.md invariant #3).
    """
    from spinedb_api import Array, DatabaseMapping, Map, import_data

    from flextool.update_flextool.initialize_database import initialize_database

    repo_root = Path(__file__).resolve().parents[2]
    schema_json = repo_root / "flextool" / "schemas" / "spinedb_schema.json"
    initialize_database(str(schema_json), str(db_path))
    url = f"sqlite:///{db_path.resolve()}"

    steps = [f"t{i:04d}" for i in range(1, DISPATCH_HOURS + 1)]
    raw_values = [
        # 8 hourly timesteps, 1 h each.
        ("timeline", "tl_mini", "timestep_duration",
         Map(steps, [1.0] * DISPATCH_HOURS, index_name="timestep")),
        # ts_full: one block of 8 steps from t0001 → full window (8 h).
        ("timeset", "ts_full", "timeline", "tl_mini"),
        ("timeset", "ts_full", "timeset_duration",
         Map(["t0001"], [float(DISPATCH_HOURS)], index_name="timestep")),
        # ts_sample: one block of 4 steps from t0001 → sampled window (4 h).
        ("timeset", "ts_sample", "timeline", "tl_mini"),
        ("timeset", "ts_sample", "timeset_duration",
         Map(["t0001"], [float(INVEST_HOURS)], index_name="timestep")),
        # demand node: balance node, constant -100 MW inflow (forced demand).
        ("node", "demand", "node_type", "balance"),
        ("node", "demand", "inflow",
         Map(steps, [-DEMAND_MW] * DISPATCH_HOURS, index_name="timestep")),
        ("node", "demand", "penalty_up", 1000.0),
        ("node", "demand", "penalty_down", 1000.0),
        # gen unit: pure source, capacity 200 MW (≥ demand), eff 1, cost 5/MWh.
        ("unit", "gen", "conversion_method", "none"),
        ("unit", "gen", "efficiency", 1.0),
        ("unit", "gen", "existing", 2.0 * DEMAND_MW),
        ("unit__outputNode", ("gen", "demand"), "other_operational_cost",
         OPERATIONAL_COST),
        # cascade: two sibling sub-solves, invest then dispatch.
        ("model", "flexTool", "solves",
         Array(["mini_invest", "mini_dispatch"], value_type=str,
               index_name="sequence_index")),
        # invest sub-solve: samples the 4 h window, realizes invest for p2020.
        ("solve", "mini_invest", "solve_mode", "single_solve"),
        ("solve", "mini_invest", "period_timeset",
         Map(["p2020"], ["ts_sample"], index_name="period")),
        ("solve", "mini_invest", "invest_periods",
         Array(["p2020"], value_type=str, index_name="period")),
        ("solve", "mini_invest", "realized_invest_periods",
         Array(["p2020"], value_type=str, index_name="period")),
        # dispatch sub-solve: realizes p2020 over the full 8 h window.
        ("solve", "mini_dispatch", "solve_mode", "single_solve"),
        ("solve", "mini_dispatch", "period_timeset",
         Map(["p2020"], ["ts_full"], index_name="period")),
        ("solve", "mini_dispatch", "realized_periods",
         Array(["p2020"], value_type=str, index_name="period")),
    ]
    data = {
        "entities": [
            ("timeline", "tl_mini"),
            ("timeset", "ts_full"),
            ("timeset", "ts_sample"),
            ("node", "demand"),
            ("unit", "gen"),
            ("unit__outputNode", ("gen", "demand")),
            ("model", "flexTool"),
            ("solve", "mini_invest"),
            ("solve", "mini_dispatch"),
        ],
        "alternatives": [("mini",)],
        "entity_alternatives": [
            ("node", ("demand",), "mini", True),
            ("unit", ("gen",), "mini", True),
            ("unit__outputNode", ("gen", "demand"), "mini", True),
        ],
        "scenarios": [("mini_nested",)],
        "scenario_alternatives": [("mini_nested", "mini")],
        # Every parameter value lives on the single ``mini`` alternative.
        "parameter_values": [(c, e, p, v, "mini") for (c, e, p, v) in raw_values],
    }

    with DatabaseMapping(url, create=False) as db_map:
        count, errors = import_data(db_map, **data)
        if errors:
            raise RuntimeError(f"mini_nested import errors: {errors}")
        db_map.commit_session("mini nested invest->dispatch scenario")
    return url


def _run_and_read_outputs(url: str, work: Path, out: Path):
    """Run the cascade end-to-end and return the per-scenario output CSV dir.

    Mirrors the CLI ``cmd_run_flextool`` path: ``run_chain_from_db`` (which
    persists per-roll realized slices to ``output_raw/``) followed by
    ``write_outputs`` (which unions those slices and produces the annualized
    ``*__d`` frames).  ``csv_dump=True`` keeps ``output_raw/`` on disk so the
    multi-sub-solve union path activates.
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    steps = run_chain_from_db(
        url, scenario_name="mini_nested", work_folder=work, csv_dump=True,
    )
    # Genuine multi-sub-solve cascade: invest + dispatch both ran.
    assert list(steps.keys()) == ["mini_invest", "mini_dispatch"], (
        f"expected a 2-solve cascade, got {list(steps.keys())}"
    )
    assert all(s.optimal for s in steps.values()), (
        "every sub-solve must be optimal: "
        f"{[(n, s.optimal) for n, s in steps.items()]}"
    )

    last = next(reversed(list(steps.values())))
    write_outputs(
        scenario_name="mini_nested",
        output_location=str(out),
        subdir="mini_nested",
        write_methods=["csv"],
        flex_data=last.flex_data,
        solution=last.solution,
        solve_name=last.solve_name,
        flex_data_provider=getattr(last, "flex_data_provider", None),
        raw_output_dir=str(work / "output_raw"),
    )
    csv_dir = out / "output_csv" / "mini_nested"
    assert csv_dir.is_dir(), f"output CSV dir not produced: {csv_dir}"
    return csv_dir


def test_nested_invest_dispatch_annualized_period_total(tmp_path: Path) -> None:
    """The realized-dispatch per-period energy total annualizes with the
    DISPATCH window share (8/8760), not the invest sample window (4/8760).
    """
    url = _build_mini_nested_db(tmp_path / "mini.sqlite")
    csv_dir = _run_and_read_outputs(url, tmp_path / "work", tmp_path / "out")

    # ── unit__outputNode__d: the gen→demand period energy total ──────────
    # First row is the header-detail row (units/labels); the data row has
    # the realized solve label.  Read with the header-detail row dropped.
    flows = pd.read_csv(csv_dir / "unit__outputNode__d.csv")
    data_rows = flows[flows["solve"] == "mini_dispatch"]
    assert len(data_rows) == 1, f"expected one realized dispatch row, got:\n{flows}"
    gen_energy = float(data_rows["gen"].iloc[0])
    # 800 MWh window ÷ (8/8760) = 100 · 8760 = 876 000 MWh.
    assert gen_energy == pytest.approx(EXPECTED_PERIOD_ENERGY, rel=1e-9), (
        f"unit__outputNode__d gen={gen_energy}, expected "
        f"{EXPECTED_PERIOD_ENERGY} (=100 MW · 8760 h).  The bug inflates "
        f"this to {BUG_PERIOD_ENERGY} (annualizing with the 4 h invest "
        f"window's share instead of the 8 h dispatch window's)."
    )
    # Guard: must NOT equal the bug value.
    assert gen_energy != pytest.approx(BUG_PERIOD_ENERGY, rel=1e-9), (
        f"unit__outputNode__d gen equals the BUG value {BUG_PERIOD_ENERGY} — "
        f"complete_period_share_of_year was sourced from the invest sub-solve"
    )

    # ── node__d: "From units" / "Inflow" mirror the same period total ────
    node_d = pd.read_csv(csv_dir / "node__d.csv")
    node_row = node_d[node_d["solve"] == "mini_dispatch"]
    assert len(node_row) == 1
    # The first 'demand' column is "From units" (the header-detail row 0
    # carries the column labels).
    from_units = float(node_row["demand"].iloc[0])
    assert from_units == pytest.approx(EXPECTED_PERIOD_ENERGY, rel=1e-9), (
        f"node__d 'From units'={from_units}, expected {EXPECTED_PERIOD_ENERGY}"
    )

    # ── annualized_costs__d: other operational = energy · 5 = 4.38 M ─────
    costs_d = pd.read_csv(csv_dir / "annualized_costs__d.csv")
    cost_row = costs_d[costs_d["solve"] == "mini_dispatch"]
    assert len(cost_row) == 1
    other_op = float(cost_row["other operational"].iloc[0])
    # annualized_costs are reported in M CUR.
    assert other_op == pytest.approx(EXPECTED_ANNUAL_COST / 1e6, rel=1e-9), (
        f"annualized 'other operational'={other_op} M CUR, expected "
        f"{EXPECTED_ANNUAL_COST / 1e6} M CUR (=876 000 MWh · 5 CUR/MWh)"
    )

    # ── summary_solve: "Time in use in years" == dispatch share 8/8760 ───
    # summary_solve.csv has no real header row (its first cell is a run
    # timestamp) and ragged rows, so parse it line-by-line: the row label
    # lives in column 0 and the realized-period value in column 1.
    import csv

    tiu_values: list[str] = []
    with open(csv_dir / "summary_solve.csv", newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip() == "Time in use in years":
                tiu_values = row[1:]
    assert tiu_values, "expected a 'Time in use in years' row in summary_solve.csv"
    time_in_use = float(tiu_values[0])
    assert time_in_use == pytest.approx(EXPECTED_TIME_IN_USE_YEARS, rel=1e-9), (
        f"'Time in use in years'={time_in_use}, expected "
        f"{EXPECTED_TIME_IN_USE_YEARS} (=8/8760, the dispatch window share)"
    )
