"""Regression tests for the folder-name vs in-data scenario-tag mismatch.

Output folders may carry a GUI run-index suffix (``S2_Dry_1``) while the
dispatch data inside is tagged with the model scenario name (``S2_Dry``).
The load boundary (``combine_parquet_files`` /
``combine_dispatch_mappings``) re-tags each scenario to its folder identity
so two folders holding the *same* model scenario stay distinct and all
downstream slicing keys off the folder name.  ``data_scenario_tags`` /
``resolve_data_scenario_tag`` read those identities back.  See
specs/dispatch_scenario_identity_retag.md.
"""

import pandas as pd

from flextool.lean_parquet import write_lean_parquet
from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.db_reader import combine_parquet_files
from flextool.scenario_comparison.dispatch_mappings import (
    combine_dispatch_mappings,
    data_scenario_tags,
    resolve_data_scenario_tag,
)


def _mappings_with_tag(tag: str) -> DispatchMappings:
    """DispatchMappings whose dispatch_groups is indexed by *tag* (as loaded)."""
    df = pd.DataFrame({"scenario": [tag], "group": ["elec"]}).set_index("scenario")
    return DispatchMappings(dispatch_groups=df)


def test_resolves_in_data_tag_over_folder_fallback():
    # Folder is "S2_Dry_1" but the data is tagged "S2_Dry": the data wins.
    mappings = _mappings_with_tag("S2_Dry")
    assert resolve_data_scenario_tag(mappings, fallback="S2_Dry_1") == "S2_Dry"


def test_resolves_from_scenario_column_when_not_indexed():
    df = pd.DataFrame({"scenario": ["S2_Dry"], "group": ["elec"]})
    mappings = DispatchMappings(dispatch_groups=df)
    assert resolve_data_scenario_tag(mappings, fallback="S2_Dry_1") == "S2_Dry"


def test_falls_back_when_no_dispatch_groups():
    assert resolve_data_scenario_tag(DispatchMappings(), fallback="folder") == "folder"


def test_falls_back_on_empty_frame():
    empty = pd.DataFrame({"scenario": [], "group": []}).set_index("scenario")
    mappings = DispatchMappings(dispatch_groups=empty)
    assert resolve_data_scenario_tag(mappings, fallback="folder") == "folder"


def test_falls_back_when_multiple_tags_ambiguous():
    # Combined multi-scenario mappings: no single tag to pick — keep fallback.
    df = pd.DataFrame(
        {"scenario": ["S2_Dry", "S3_Dry"], "group": ["elec", "elec"]}
    ).set_index("scenario")
    mappings = DispatchMappings(dispatch_groups=df)
    assert resolve_data_scenario_tag(mappings, fallback="folder") == "folder"


def _multi_mappings(tags: list[str]) -> DispatchMappings:
    df = pd.DataFrame(
        {"scenario": tags, "group": ["elec"] * len(tags)}
    ).set_index("scenario")
    return DispatchMappings(dispatch_groups=df)


def test_data_scenario_tags_preserves_folder_order_and_dedups():
    # Comparison: folders [base_1, S2_Dry_1, S2_Normal_1] hold tags
    # [base, S2_Dry, S2_Normal]; order (= folder concat order) is preserved.
    mappings = _multi_mappings(["base", "S2_Dry", "S2_Normal"])
    assert data_scenario_tags(mappings) == ["base", "S2_Dry", "S2_Normal"]


def test_data_scenario_tags_dedups_repeated_rows():
    # dispatch_groups may carry several rows per scenario (one per group);
    # tags collapse to first-seen order.
    mappings = _multi_mappings(["S2_Dry", "S2_Dry", "S3_Dry"])
    assert data_scenario_tags(mappings) == ["S2_Dry", "S3_Dry"]


def test_data_scenario_tags_empty_when_no_data():
    assert data_scenario_tags(DispatchMappings()) == []


# --- Load-boundary re-tag: two folders, same model scenario, stay distinct ---

def _ts_frame(tag: str) -> pd.DataFrame:
    """A time-series frame tagged *tag* on the column scenario level."""
    cols = pd.MultiIndex.from_tuples(
        [(tag, "unitA", "nodeA")], names=["scenario", "unit", "node"],
    )
    idx = pd.MultiIndex.from_tuples(
        [("p1", 0), ("p1", 1)], names=["period", "time"],
    )
    return pd.DataFrame([[1.0], [2.0]], index=idx, columns=cols)


def test_combine_parquet_files_retags_columns_to_folder(tmp_path):
    # Two folders A_1 and A_2 both hold data tagged with the model name 'A'.
    files = []
    for folder in ("A_1", "A_2"):
        d = tmp_path / folder
        d.mkdir()
        path = d / "unit_outputNode_dt_ee.parquet"
        write_lean_parquet(_ts_frame("A"), path)
        files.append((folder, path))

    combined = combine_parquet_files(
        {"unit_outputNode_dt_ee.parquet": files}, num_scenarios=2,
    )
    df = combined["unit_outputNode_dt_ee"]
    # The two folders survive as DISTINCT scenario identities (not merged
    # under the shared model tag 'A'), keyed by folder name.
    assert set(df.columns.get_level_values("scenario")) == {"A_1", "A_2"}
    assert "A" not in df.columns.get_level_values("scenario")


def test_combine_parquet_files_noop_when_tag_equals_folder(tmp_path):
    # DB/CLI path: folder name already equals the in-data tag → re-tag is a
    # no-op and the scenario level is preserved verbatim.
    files = []
    for folder in ("A", "B"):
        d = tmp_path / folder
        d.mkdir()
        path = d / "unit_outputNode_dt_ee.parquet"
        write_lean_parquet(_ts_frame(folder), path)
        files.append((folder, path))
    combined = combine_parquet_files(
        {"unit_outputNode_dt_ee.parquet": files}, num_scenarios=2,
    )
    assert set(
        combined["unit_outputNode_dt_ee"].columns.get_level_values("scenario")
    ) == {"A", "B"}


def test_combine_dispatch_mappings_retags_rows_to_folder(tmp_path):
    # Two folders both tagged model scenario 'A'; nodeGroupDispatch carries
    # scenario as a row column.
    for folder in ("A_1", "A_2"):
        d = tmp_path / folder
        d.mkdir()
        df = pd.DataFrame({"scenario": ["A"], "group": ["elec"]})
        write_lean_parquet(df, d / "nodeGroupDispatch.parquet", index=False)

    scenario_folders = {"A_1": str(tmp_path), "A_2": str(tmp_path)}
    mappings = combine_dispatch_mappings(scenario_folders, "")

    # Both folders present as distinct identities; each slices independently.
    assert data_scenario_tags(mappings) == ["A_1", "A_2"]
    assert mappings.get_for_scenario("dispatch_groups", "A_1") is not None
    assert mappings.get_for_scenario("dispatch_groups", "A_2") is not None
    assert mappings.get_for_scenario("dispatch_groups", "A") is None
