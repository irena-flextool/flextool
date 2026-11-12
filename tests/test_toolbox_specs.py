"""Every file a Spine Toolbox tool specification of the shipped project includes exists (item 237).

Toolbox copies a Tool's ``includes`` into its work folder and skips a missing one without a word, so a
module that moved or a deleted file (``bin/highs.opt``) stays in the specification unnoticed.  The
paths are relative to the specification's ``includes_main_path``, which is relative to the
specification file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_JSON = REPO_ROOT / ".spinetoolbox" / "project.json"

pytestmark = pytest.mark.skipif(not PROJECT_JSON.is_file(), reason="no .spinetoolbox/ (sdist)")


def _tool_specs() -> list[Path]:
    project = json.loads(PROJECT_JSON.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in project["project"]["specifications"].get("Tool", [])]
    return [REPO_ROOT / p for p in paths]


def test_project_lists_tool_specifications() -> None:
    names = {p.name for p in _tool_specs()}
    assert {"flextool3.json", "process_outputs.json", "scenario_comparison.json"} <= names


@pytest.mark.parametrize("spec_path", _tool_specs() if PROJECT_JSON.is_file() else [], ids=lambda p: p.name)
def test_every_include_exists(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    main_dir = (spec_path.parent / spec.get("includes_main_path", ".")).resolve()
    missing = [inc for inc in spec.get("includes", []) if not (main_dir / inc).exists()]
    assert missing == [], f"{spec_path.name} includes files that do not exist under {main_dir}: {missing}"
