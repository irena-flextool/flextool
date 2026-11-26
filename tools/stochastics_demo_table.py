"""Run the example scenarios of ``docs/how_to/stochastic_examples.md`` and print the how-to's result tables.

The scenarios are in ``tests/fixtures/stochastics.json`` (``tests/fixtures/build_stochastics_demo.py`` describes
them).  The script builds a database from that file in a temporary folder, runs each example scenario with the
shipped solver settings, reads FlexTool's own result tables and prints two markdown tables: the rolling dispatch
scenarios and the investment scenarios.  The rolling scenarios take about a minute each.

Usage: ``python tools/stochastics_demo_table.py [--only rolling|invest]``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIXTURE = REPO / "tests" / "fixtures" / "stochastics.json"


def _rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def _table(path: Path, header_rows: int = 1) -> tuple[list[str], list[list[str]]]:
    rows = _rows(path)
    return rows[0], rows[header_rows:]


def rolling_summary(csv_dir: Path, steps: dict) -> dict[str, float]:
    """Realized results of a rolling run over its simulated hours, from the result tables in *csv_dir*."""
    from tests.fixtures.build_stochastics_demo import PERIOD_HOURS

    share = PERIOD_HOURS / 8760          # the __d tables are scaled to a full year
    _header, rows = _table(csv_dir / "costs__dt.csv")
    cost = sum(float(v) for r in rows for v in r[3:] if v)
    header, rows = _table(csv_dir / "unit_startup__d.csv")
    col = {name: i for i, name in enumerate(header)}
    coal_starts = sum(float(r[col["coal_units"]]) for r in rows) * share
    peaker_started = sum(float(r[col["gas_peaker"]]) for r in rows) * share
    header, rows = _table(csv_dir / "connection_leftward__dt.csv", header_rows=2)
    into_grid = [float(r[header.index("battery_inverter")]) for r in rows]
    header, rows = _table(csv_dir / "unit__outputNode__d.csv", header_rows=2)
    gen = {u: sum(float(r[header.index(u)]) for r in rows) * share for u in ("coal_units", "gas_peaker", "wind_farm")}
    header, rows = _table(csv_dir / "slack__upward__node_state__d.csv")
    unserved = sum(float(r[header.index("elec")]) for r in rows) * share
    return {
        "solves": len(steps), "cost": cost, "coal_starts": coal_starts, "peaker_started": peaker_started,
        "discharged": sum(v for v in into_grid if v > 0), "charged": -sum(v for v in into_grid if v < 0),
        "coal": gen["coal_units"], "peaker": gen["gas_peaker"], "wind": gen["wind_farm"], "unserved": unserved,
    }


def invest_summary(csv_dir: Path, steps: dict) -> dict[str, float]:
    """The investment solve's objective, the realized y2030 investment and its cost, from *csv_dir* and *steps*."""
    first = next(iter(steps.values()))
    header, rows = _table(csv_dir / "unit_cohort_capacity__d.csv")
    invested = sum(float(r[header.index("capacity")]) for r in rows if r[header.index("kind")] == "invested")
    header, rows = _table(csv_dir / "annualized_costs__d.csv")
    yearly = sum(float(r[header.index("investment")]) for r in rows)
    header, rows = _table(csv_dir / "unit__outputNode__dt.csv", header_rows=2)
    energy = sum(float(r[header.index("new_plant")]) for r in rows)
    return {"solves": ", ".join(steps), "objective": first.solution.obj / 1e6, "invested": invested,
            "yearly": yearly, "energy": energy}


def run_scenario(url: str, scenario: str, work: Path) -> tuple[Path, dict]:
    """Run *scenario* in *work* and write its CSV results; returns their folder and the run's steps."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    work.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(work)
    try:
        steps = run_chain_from_db(url, scenario, work_folder=work, keep_solutions=True, csv_dump=True)
        last = next(reversed(steps.values()))
        write_outputs(
            scenario_name=scenario, output_location=str(work), subdir=scenario,
            output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"),
            write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
            solution=last.solution, solve_name=last.solve_name,
            solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
            flex_data_provider=FlexDataProvider())
    finally:
        os.chdir(cwd)
    return work / "output_csv" / scenario, steps


def rolling_table(results: dict[str, dict]) -> list[str]:
    lines = ["| scenario | solves | cost (k EUR) | coal starts | peaker starts (350 MW each) | battery discharged (MWh) | "
             "battery charged (MWh) | coal (GWh) | peaker (GWh) | wind (GWh) | unserved (MWh) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['solves']} | {r['cost'] / 1e3:,.1f} | {r['coal_starts']:.0f} | {r['peaker_started']:.1f} | "
            f"{r['discharged']:,.0f} | {r['charged']:,.0f} | {r['coal'] / 1e3:.2f} | {r['peaker'] / 1e3:.2f} | "
            f"{r['wind'] / 1e3:.2f} | {r['unserved']:,.1f} |")
    return lines


def invest_table(results: dict[str, dict]) -> list[str]:
    lines = ["| scenario | solves | objective of the investment solve (M EUR) | y2030 investment (MW) | "
             "its yearly cost (M EUR) | y2030 dispatch (MWh) |",
             "|---|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['solves']} | {r['objective']:.1f} | {r['invested']:.0f} | {r['yearly']:.1f} | "
                     f"{r['energy']:.0f} |")
    return lines


def main(argv: list[str] | None = None) -> None:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.fixtures.build_stochastics_demo import INVEST_SCENARIOS, ROLL_SCENARIOS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("rolling", "invest"), help="run only one of the two groups")
    args = parser.parse_args(argv)
    groups = {"rolling": (ROLL_SCENARIOS, rolling_summary, rolling_table),
              "invest": (INVEST_SCENARIOS, invest_summary, invest_table)}
    with tempfile.TemporaryDirectory() as tmp:
        url = json_to_db(FIXTURE, Path(tmp) / "stochastics.sqlite")
        migrate_database(url)
        for group, (scenarios, summary, table) in groups.items():
            if args.only not in (None, group):
                continue
            results = {}
            for scenario in scenarios:
                csv_dir, steps = run_scenario(url, scenario, Path(tmp) / scenario)
                results[scenario] = summary(csv_dir, steps)
            print("\n".join(table(results)) + "\n")


if __name__ == "__main__":
    main()
