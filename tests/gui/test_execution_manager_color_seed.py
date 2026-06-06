"""Stage 5: ExecutionManager.add_jobs upfront input-DB color seeding hook.

Verifies the GUI-execution hook:

* an sqlite-source scenario batch (input DB exists on disk) seeds entity
  colors into the project ``plot_settings.yaml`` once;
* an xlsx-source batch (intermediate/<stem>.sqlite not yet created) is
  skipped without error (known limitation);
* the seeding is best-effort and never breaks job queuing.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping

from flextool.gui.data_models import ProjectSettings, ScenarioInfo
from flextool.gui.execution_manager import ExecutionManager


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for sub in ("input_sources", "intermediate"):
        (project / sub).mkdir(parents=True, exist_ok=True)
    # Minimal project plot_settings.yaml (so seeding writes the project copy).
    (project / "plot_settings.yaml").write_text(
        "scenarios:\n\nentities:\n  group:\n  unit:\n  connection:\n  node:\n",
        encoding="utf-8",
    )
    return project


def _build_input_db(path: Path, units: list[str]) -> None:
    url = "sqlite:///" + str(path)
    with DatabaseMapping(url, create=True) as db:
        db.add_update_item("entity_class", name="unit")
        for u in units:
            db.add_update_item(
                "entity", entity_class_name="unit", name=u, entity_byname=(u,)
            )
        db.commit_session("color-seed hook test fixture")


def test_add_jobs_seeds_sqlite_source(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    db_path = project / "input_sources" / "in.sqlite"
    _build_input_db(db_path, ["coal_plant", "battery"])

    mgr = ExecutionManager(project_path=project, settings=ProjectSettings())
    jobs = mgr.add_jobs([
        ScenarioInfo(name="base", source_number=1, source_name="in.sqlite"),
    ])
    assert len(jobs) == 1

    settings_text = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    assert "coal_plant:" in settings_text
    assert "battery:" in settings_text


def test_add_jobs_skips_xlsx_intermediate_not_built(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    # xlsx source present, but intermediate/<stem>.sqlite NOT yet created.
    (project / "input_sources" / "data.xlsx").write_bytes(b"not a real xlsx")
    before = (project / "plot_settings.yaml").read_text(encoding="utf-8")

    mgr = ExecutionManager(project_path=project, settings=ProjectSettings())
    jobs = mgr.add_jobs([
        ScenarioInfo(name="base", source_number=1, source_name="data.xlsx"),
    ])
    # Job is still queued (best-effort: skipping the seed never blocks it).
    assert len(jobs) == 1
    after = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    assert before == after
