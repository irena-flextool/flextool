"""``migrate_project`` drops the Open_summary item of an old Spine Toolbox project: it ran ``open_summary.py``, which
no longer exists, and its specification file is no longer shipped."""
from __future__ import annotations

import json
from pathlib import Path

from flextool.update_flextool.self_update import migrate_project

_SHIPPED = Path(__file__).resolve().parents[2] / ".spinetoolbox" / "project.json"
_SPEC = ".spinetoolbox/specifications/Tool/open_summary.json"


def test_old_open_summary_item_is_dropped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".spinetoolbox").mkdir()
    old = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    assert "Open_summary" not in old["items"]
    old["items"]["Open_summary"] = {"type": "Tool", "specification": "open_summary", "x": 0, "y": 0}
    old["project"]["connections"].append(
        {"name": "from FlexTool3 to Open_summary", "from": ["FlexTool3", "bottom"], "to": ["Open_summary", "right"]})
    old["project"]["specifications"]["Tool"].append({"type": "path", "relative": True, "path": _SPEC})
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old), encoding="utf-8")

    new_path = tmp_path / "project.json"  # migrate_project writes the result over the new project.json
    new_path.write_text(_SHIPPED.read_text(encoding="utf-8"), encoding="utf-8")

    migrate_project(str(old_path), str(new_path))

    migrated = json.loads(new_path.read_text(encoding="utf-8"))
    shipped = json.loads(_SHIPPED.read_text(encoding="utf-8"))
    assert "Open_summary" not in migrated["items"]
    assert all("Open_summary" not in (c["from"][0], c["to"][0]) for c in migrated["project"]["connections"])
    assert _SPEC not in [s["path"] for s in migrated["project"]["specifications"]["Tool"]]
    assert len(migrated["project"]["connections"]) == len(shipped["project"]["connections"])
