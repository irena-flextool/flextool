"""Release guard for the behaviour-change registry (``flextool/engine/_behaviour_changes.py``).

A notice names the FlexTool version that changed the behaviour and its publication date.  Both are filled in by
the release that ships the change, from its CHANGELOG heading ``## Release X (D.M.YYYY)``.  ``publish.yml`` runs
this file before it builds a release:

* once the first CHANGELOG heading is a release (no ``## Unreleased`` on top), no entry may be left unfilled;
* every filled entry matches a release heading with that version and date.
"""
from __future__ import annotations

import re
from pathlib import Path

from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX, BehaviourChange
from flextool.update_flextool import db_migration

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
RELEASE = re.compile(r"^## Release (\S+) \((\d{1,2}\.\d{1,2}\.\d{4})\)")


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("## ")]


def registry_problems(changelog_text: str, changes) -> list[str]:
    """What stops a release: unfilled entries under a released CHANGELOG, filled ones without their heading."""
    headings = _headings(changelog_text)
    problems = []
    if headings and headings[0].startswith("## Release"):
        problems += [f"{c.id}: since / published not filled (newest heading {headings[0]!r})"
                     for c in changes if c.since is None or c.published is None]
    releases = {m.groups() for m in (RELEASE.match(h) for h in headings) if m}
    problems += [f"{c.id}: no CHANGELOG heading '## Release {c.since} ({c.published})'"
                 for c in changes if (c.since, c.published) != (None, None) and (c.since, c.published) not in releases]
    return problems


def test_registry_matches_the_changelog() -> None:
    problems = registry_problems(CHANGELOG.read_text(encoding="utf-8"), BEHAVIOUR_CHANGES.values())
    assert not problems, "Fill since / published in flextool/engine/_behaviour_changes.py: " + "; ".join(problems)


def test_guard_catches_a_release_without_filled_entries() -> None:
    change = BehaviourChange(id="x", summary="s", silence="q")
    filled = BehaviourChange(id="x", summary="s", silence="q", since="4.2.0", published="1.12.2026")
    unreleased = "## Unreleased\n\n## Release 4.1.1 (21.10.2026) - t\n"
    released = "## Release 4.2.0 (1.12.2026) - t\n\n## Release 4.1.1 (21.10.2026) - t\n"
    assert registry_problems(unreleased, [change]) == []
    assert registry_problems(released, [change])
    assert registry_problems(released, [filled]) == []
    wrong_date = BehaviourChange(id="x", summary="s", silence="q", since="4.2.0", published="2.12.2026")
    assert registry_problems(released, [wrong_date])


def test_migration_markers_are_registered() -> None:
    """Every marker the database upgrade writes has a registry entry (else its notice could never print)."""
    assert db_migration.BEHAVIOUR_CHANGE_MARKER_PREFIX == MARKER_PREFIX
    written = set(db_migration._V77_UNCONDITIONAL_MARKERS) | {"storage_default_fix_nothing",
                                                              "storage_fix_method_rewritten",
                                                              "ladder_tier_labels_as_written",
                                                              "unnamed_maps_read_per_depth",
                                                              "solve_step_series_combined",
                                                              "energy_margin_adder_per_original_step",
                                                              "discount_rate_zero_is_zero",
                                                              "invest_without_cost_offered",
                                                              "stochastic_storage_paths_tied",
                                                              "chained_solve_discount_from_run_start"}
    assert written <= set(BEHAVIOUR_CHANGES)
