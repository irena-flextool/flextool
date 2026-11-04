"""Every output table gets its own CSV file.

The CSV writer names each table ``<display_name>.csv`` from the ``rename``
section of ``default_plots.yaml``.  Two tables with the same display name
would overwrite each other, so:

* the default settings give every output a distinct file name (also when
  compared without case — Windows and macOS file systems are case-blind);
* the writer refuses a repeated file name instead of overwriting;
* both discounted-cost tables reach disk: the per-period table
  ``costs_discounted__d.csv`` and the per-category horizon total
  ``costs_discounted.csv``.  The horizon total is the per-period table
  summed over the periods (``out_costs.cost_summaries`` builds the total
  as ``summary_inflation_years.sum(axis=0)`` of the very frame written per
  period), so the two files must agree category by category.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from flextool._resources import package_data_path
from flextool.engine.output._parquet_bundle import REGISTRY
from flextool.process_outputs.write_outputs import _csv_filename, _parse_rename_entry

from tests.test_cost_aggregation_semantics import _run_scenario


def _rename_map() -> dict:
    with open(package_data_path("schemas/default_plots.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["rename"]


def test_default_display_names_are_distinct() -> None:
    rename = _rename_map()
    keys = set(rename) | {k for k, spec in REGISTRY.items() if spec.category == "processed"}
    names = [_parse_rename_entry(rename.get(k, k))[0].lower() for k in sorted(keys)]
    repeated = sorted(n for n, c in Counter(names + ["summary_solve"]).items() if c > 1)
    assert repeated == []


def test_writer_refuses_a_repeated_file_name() -> None:
    rename = {"table_a": ["same", True], "table_b": ["Same", True]}
    used: dict[str, str] = {}
    assert _csv_filename("table_a", rename, used) == "same.csv"
    with pytest.raises(ValueError, match="'table_a' and 'table_b'"):
        _csv_filename("table_b", rename, used)


@pytest.fixture(scope="module")
def multi_year_csv(test_db_url: str, test_solver_config_dir: Path,
                   tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("csv_names_multi_year")
    return _run_scenario("multi_year", test_db_url, test_solver_config_dir, workdir)


def test_both_discounted_cost_tables_are_written(multi_year_csv: Path) -> None:
    per_period = pd.read_csv(multi_year_csv / "costs_discounted__d.csv")
    total = pd.read_csv(multi_year_csv / "costs_discounted.csv").set_index("category").iloc[:, 0]
    assert per_period["period"].nunique() > 1, "fixture must span several periods"
    summed = per_period.drop(columns=["solve", "period"]).sum(axis=0)
    pd.testing.assert_series_equal(summed.reindex(total.index), total, check_names=False,
                                   rtol=1e-6, atol=1e-9)
