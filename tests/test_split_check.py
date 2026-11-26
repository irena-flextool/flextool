"""``tools/split_check.py``: a split that only moves code passes; each kind of change is caught."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "split_check.py"
_spec = importlib.util.spec_from_file_location("split_check", _SCRIPT)
split_check = importlib.util.module_from_spec(_spec)
sys.modules["split_check"] = split_check
_spec.loader.exec_module(split_check)

OLD = '''import json

LIMIT = 3
# the doubled limit
DOUBLE = LIMIT * 2


def helper(x):
    """Help."""
    from json import dumps
    return dumps(x)


def run(db, n):
    total = 0
    while n < LIMIT:
        if n == 0:
            pass
        elif n == 1:
            # step one
            seen = [helper(db)]
            for s in seen:
                if not s:
                    continue
                json.loads(s)
        else:
            total += 1
        n += 1
    return total
'''

COMMON = '''"""Shared."""
LIMIT = 3
# the doubled limit
DOUBLE = LIMIT * 2


def helper(x):
    """Help."""
    import json as _j
    from json import dumps
    return dumps(x)
'''

STEP = '''import json


def _step1(db):
    # step one
    seen = [helper(db)]
    for s in seen:
        if not s:
            continue
        json.loads(s)
'''

RUNNER = '''from pkg.common import LIMIT
from pkg.step import _step1


def run(db, n):
    total = 0
    while n < LIMIT:
        if n == 0:
            pass
        elif n == 1:
            _step1(db)
        else:
            total += 1
        n += 1
    return total
'''


def _check(tmp_path: Path, **modules: str) -> list[str]:
    texts = {tmp_path / "pkg" / f"{name}.py": text for name, text in modules.items()}
    return split_check.check_texts({"old.py": OLD}, texts)


def _ok(tmp_path: Path, **override: str) -> list[str]:
    modules = {"common": COMMON.replace("    import json as _j\n", ""), "step": STEP, "runner": RUNNER}
    modules.update(override)
    return _check(tmp_path, **modules)


def test_moves_and_an_extracted_body_pass(tmp_path: Path) -> None:
    assert _ok(tmp_path) == []


def test_an_import_inside_a_body_is_compared_by_the_names_it_binds(tmp_path: Path) -> None:
    moved = COMMON.replace("    import json as _j\n", "").replace("from json import dumps", "from json.encoder import dumps")
    assert _ok(tmp_path, common=moved) == []
    errors = _ok(tmp_path, common=COMMON)
    assert any(e.startswith("changed: helper") for e in errors)


def test_a_changed_definition_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, common=COMMON.replace("    import json as _j\n", "").replace("* 2", "* 3"))
    assert errors == ["missing from the new modules: old.py statement 1 (Assign, old line 5)",
                      f"new statement: {tmp_path / 'pkg' / 'common.py'}:4 DOUBLE = LIMIT * 3"]


def test_a_new_statement_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, step=STEP + "\nEXTRA = 1\n")
    assert errors == ["new statement: " + str(tmp_path / "pkg" / "step.py") + ":12 EXTRA = 1"]


def test_statements_out_of_their_old_order_fail(tmp_path: Path) -> None:
    common = '''LIMIT = 3


def helper(x):
    """Help."""
    from json import dumps
    return dumps(x)
# the doubled limit
DOUBLE = LIMIT * 2
'''
    errors = _ok(tmp_path, common=common)
    assert errors == [f"{tmp_path / 'pkg' / 'common.py'}: statements from old.py out of their old order"]


def test_a_lost_comment_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, step=STEP.replace("    # step one\n", ""))
    assert errors == ["comment lost (1x): # step one"]


def test_a_comment_reworded_on_purpose_is_allowed(tmp_path: Path) -> None:
    step = STEP.replace("    # step one\n", "    # the first step\n")
    texts = {tmp_path / "pkg" / f"{k}.py": v for k, v in
             {"common": COMMON.replace("    import json as _j\n", ""), "step": step, "runner": RUNNER}.items()}
    assert split_check.check_texts({"old.py": OLD}, texts) == ["comment lost (1x): # step one"]
    assert split_check.check_texts({"old.py": OLD}, texts, allow_lost=(r"^# step one$",)) == []


def test_an_extracted_body_that_reads_a_local_fails(tmp_path: Path) -> None:
    step = STEP.replace("seen = [helper(db)]", "seen = [helper(total)]")
    errors = _ok(tmp_path, step=step)
    assert any(e.startswith("changed: run") for e in errors)   # the old body read db: not the same statements
    old = OLD.replace("seen = [helper(db)]", "seen = [helper(total)]")
    texts = {tmp_path / "pkg" / f"{k}.py": v for k, v in
             {"common": COMMON.replace("    import json as _j\n", ""), "step": step, "runner": RUNNER}.items()}
    errors = split_check.check_texts({"old.py": old}, texts)
    assert errors == ["_step1: reads locals of run: ['total']"]


def test_an_extracted_body_that_continues_the_outer_loop_fails(tmp_path: Path) -> None:
    old = OLD.replace("            seen = [helper(db)]\n", "            seen = [helper(db)]\n            if not seen:\n"
                      "                continue\n")
    step = STEP.replace("    seen = [helper(db)]\n", "    seen = [helper(db)]\n    if not seen:\n        continue\n")
    texts = {tmp_path / "pkg" / f"{k}.py": v for k, v in
             {"common": COMMON.replace("    import json as _j\n", ""), "step": step, "runner": RUNNER}.items()}
    errors = split_check.check_texts({"old.py": old}, texts)
    assert errors == ["_step1: Continue at line 8 would act on run"]


def test_an_extracted_body_that_assigns_a_name_the_rest_reads_fails(tmp_path: Path) -> None:
    old = OLD.replace("            seen = [helper(db)]\n", "            seen = [helper(db)]\n            total = 5\n")
    step = STEP.replace("    seen = [helper(db)]\n", "    seen = [helper(db)]\n    total = 5\n")
    texts = {tmp_path / "pkg" / f"{k}.py": v for k, v in
             {"common": COMMON.replace("    import json as _j\n", ""), "step": step, "runner": RUNNER}.items()}
    errors = split_check.check_texts({"old.py": old}, texts)
    assert errors == ["_step1: assigns names the rest of run uses: ['total']"]


def test_an_import_cycle_fails(tmp_path: Path) -> None:
    common = COMMON.replace("    import json as _j\n", "").replace('"""Shared."""\n', '"""Shared."""\nfrom pkg.runner import run\n')
    errors = _ok(tmp_path, common=common)
    assert len(errors) == 1 and errors[0].startswith("import cycle: ")
