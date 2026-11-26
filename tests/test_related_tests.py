"""tools/related_tests.py: a module of a package that its facade imports relates the tests that import the package."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("related_tests", REPO / "tools" / "related_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_module_of_a_split_package_relates_the_importers_of_the_package() -> None:
    """test_coarse_row_rule imports only ``flextool.engine.model.core`` (the facade); _ramp is imported by the
    facade itself, _storage_links only through other modules of the package: both relate it."""
    tool = _tool()
    for changed in ("flextool/engine/model/core/_ramp.py", "flextool/engine/model/core/_storage_links.py"):
        assert "tests/engine/test_coarse_row_rule.py" in tool.related([changed])


def test_importing_a_plain_package_still_does_not_count() -> None:
    """``flextool.engine.model`` is a package whose __init__ imports none of its modules: a change of _online
    relates only the tests that import _online by name."""
    tool = _tool()
    assert "tests/engine/test_coarse_row_rule.py" not in tool.related(["flextool/engine/model/_online.py"])
