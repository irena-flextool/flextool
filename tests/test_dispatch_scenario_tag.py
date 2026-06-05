"""Regression tests for the folder-name vs in-data scenario-tag mismatch.

Output folders may carry a GUI run-index suffix (``S2_Dry_1``) while the
dispatch data inside is tagged with the model scenario name (``S2_Dry``).
Dispatch slicing keys off the in-data tag; feeding the folder name silently
drops every nodeGroup (empty ``_dispatch_metadata.json`` + "No dispatch
data" in the viewer).  ``resolve_data_scenario_tag`` reconciles the two.
"""

import pandas as pd

from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.dispatch_mappings import resolve_data_scenario_tag


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
