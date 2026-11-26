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

from pkg.common import helper


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


def test_an_import_inside_a_body_is_compared_by_name_and_target(tmp_path: Path) -> None:
    common = COMMON.replace("    import json as _j\n", "")
    for changed in (COMMON, common.replace("from json import dumps", "from pickle import dumps"),
                    common.replace("from json import dumps", "from json.encoder import dumps"),
                    common.replace("    from json import dumps\n", "")):
        errors = _ok(tmp_path, common=changed)
        assert any(e.startswith("changed: helper") for e in errors), errors


def test_a_changed_definition_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, common=COMMON.replace("    import json as _j\n", "").replace("* 2", "* 3"))
    assert errors == ["missing from the new modules: old.py statement 1 (Assign, old line 5)",
                      f"new statement: {tmp_path / 'pkg' / 'common.py'}:4 DOUBLE = LIMIT * 3"]


def test_a_new_statement_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, step=STEP + "\nEXTRA = 1\n")
    assert errors == ["new statement: " + str(tmp_path / "pkg" / "step.py") + ":14 EXTRA = 1"]


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
    assert errors == ["_step1: Continue at line 10 would act on run"]


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
    assert errors == ["import cycle: pkg.common -> pkg.runner -> pkg.common",
                      "import cycle: pkg.common -> pkg.runner -> pkg.step -> pkg.common"]


# --- what a moved statement's global names resolve to ---------------------------------------------------------

def _texts(tmp_path: Path, **modules: str) -> dict[Path, str]:
    """``{tmp_path / "pkg" / "a.py": text}`` from ``pkg__a=text`` (``__init__`` / ``__main__`` spelled ``init`` /
    ``main``)."""
    out = {}
    for name, text in modules.items():
        parts = [{"init": "__init__", "main": "__main__"}.get(p, p) for p in name.split("__")]
        out[tmp_path.joinpath(*parts[:-1], f"{parts[-1]}.py")] = text
    return out


def test_a_missing_import_of_a_moved_helper_fails(tmp_path: Path) -> None:
    errors = _ok(tmp_path, step=STEP.replace("from pkg.common import helper\n", ""))
    assert errors == [f"{tmp_path / 'pkg' / 'step.py'}:5 _step1: helper resolves to missing helper, "
                      "was old.py statement 2"]


def test_a_shadowing_top_level_import_fails(tmp_path: Path) -> None:
    common = COMMON.replace("    import json as _j\n", "").replace("LIMIT = 3\n", "LIMIT = 3\nfrom os import sep as LIMIT\n")
    errors = _ok(tmp_path, common=common)
    assert errors == [f"{tmp_path / 'pkg' / 'common.py'}:5 DOUBLE = LIMIT * 2: LIMIT resolves to os.sep, was old.py statement 0",
                      f"{tmp_path / 'pkg' / 'runner.py'}:5 run: LIMIT resolves to os.sep, was old.py statement 0"]


def test_a_shadowed_default_value_fails(tmp_path: Path) -> None:
    old = "A = 1\n\n\ndef f(x=A):\n    return x\n"
    ok = {"pkg__a": "A = 1\n", "pkg__b": "from pkg.a import A\n\n\ndef f(x=A):\n    return x\n"}
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **ok)) == []
    bad = dict(ok, pkg__b="from os import sep as A\n\n\ndef f(x=A):\n    return x\n")
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **bad)) == [
        f"{tmp_path / 'pkg' / 'b.py'}:4 f: A resolves to os.sep, was old.py statement 0"]


def test_an_external_name_compares_by_identity(tmp_path: Path) -> None:
    old = "from json import JSONDecodeError\n\n\ndef f(s):\n    raise JSONDecodeError(s, s, 0)\n"
    same = old.replace("from json import", "from json.decoder import")
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, pkg__a=same)) == []
    other = old.replace("from json import JSONDecodeError", "from pickle import UnpicklingError as JSONDecodeError")
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, pkg__a=other)) == [
        f"{tmp_path / 'pkg' / 'a.py'}:4 f: JSONDecodeError resolves to pickle.UnpicklingError, "
        "was json.JSONDecodeError",
        "dropped import: old.py: from json import JSONDecodeError"]


def test_a_relative_import_one_level_off_fails(tmp_path: Path) -> None:
    old = "def f():\n    from .util import x\n    return x\n"
    deeper = _texts(tmp_path, pkg__mod__part=old.replace("from .util", "from ..util"))
    assert split_check.check_texts({"pkg/mod.py": old}, deeper, root=tmp_path) == []
    errors = split_check.check_texts({"pkg/mod.py": old}, _texts(tmp_path, pkg__mod__part=old), root=tmp_path)
    part = tmp_path / "pkg" / "mod" / "part.py"
    assert errors == [f"changed: f (old line 1, now {part}:1)", f"new statement: {part}:1 f"]


def test_the_main_block_moves_to_the_package_main_module(tmp_path: Path) -> None:
    old = "import sys\n\nif __name__ == '__main__':\n    sys.exit(0)\n"
    assert split_check.check_texts({"pkg.py": old}, _texts(tmp_path, pkg__init='"""P."""\n', pkg__main=old)) == []


def test_a_module_dunder_read_by_a_moved_function_fails(tmp_path: Path) -> None:
    old = "def who():\n    return __name__\n"
    errors = split_check.check_texts({"pkg.py": old}, _texts(tmp_path, pkg__a=old))
    assert errors == [f"{tmp_path / 'pkg' / 'a.py'}:1 who: __name__ resolves to __name__ of pkg.a, "
                      "was __name__ of pkg"]
    kept = split_check.check_texts({"pkg.py": old}, _texts(tmp_path, pkg__init=old))
    assert kept == []


def test_a_dropped_top_level_import_fails(tmp_path: Path) -> None:
    old = OLD.replace("import json\n", "import json\nimport os\n")
    texts = _texts(tmp_path, pkg__common=COMMON.replace("    import json as _j\n", ""), pkg__step=STEP,
                   pkg__runner=RUNNER)
    assert split_check.check_texts({"old.py": old}, texts) == ["dropped import: old.py: import os"]
    errors = _ok(tmp_path, step=STEP.replace("import json\n", ""))
    assert errors == [f"{tmp_path / 'pkg' / 'step.py'}:5 _step1: json resolves to missing json, was json",
                      "dropped import: old.py: import json"]


def test_import_time_effects_keep_their_order_across_modules(tmp_path: Path) -> None:
    old = "REG = []\nREG.append(1)\nREG.append(2)\n\n\ndef f():\n    pass\n\n\ndef g():\n    pass\n"
    imp = "from pkg.a import REG\n"
    good = {"pkg__a": "REG = []\n", "pkg__b": imp + "REG.append(1)\n\n\ndef g():\n    pass\n",
            "pkg__c": imp + "REG.append(2)\n\n\ndef f():\n    pass\n"}
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **good)) == []
    bad = dict(good, pkg__b=imp + "REG.append(2)\n", pkg__c=imp + "REG.append(1)\n\n\ndef f():\n    pass\n\n\n"
               "def g():\n    pass\n")
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **bad)) == [
        "statements with import-time effects of old.py run out of their old order: statement 2 before 1"]


def test_an_extracted_body_reading_a_local_beside_a_comprehension_of_that_name_fails(tmp_path: Path) -> None:
    old = OLD.replace("seen = [helper(db)]", "seen = [total for total in [helper(db)]] + [total]")
    step = STEP.replace("seen = [helper(db)]", "seen = [total for total in [helper(db)]] + [total]")
    texts = _texts(tmp_path, pkg__common=COMMON.replace("    import json as _j\n", ""), pkg__step=step,
                   pkg__runner=RUNNER)
    assert split_check.check_texts({"old.py": old}, texts) == ["_step1: reads locals of run: ['total']"]


def test_a_global_rebound_name_split_over_modules_fails(tmp_path: Path) -> None:
    old = "COUNT = 0\n\n\ndef bump():\n    global COUNT\n    COUNT += 1\n\n\ndef get():\n    return COUNT\n"
    one = {"pkg__a": old}
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **one)) == []
    two = {"pkg__a": "COUNT = 0\n\n\ndef bump():\n    global COUNT\n    COUNT += 1\n",
           "pkg__b": "from pkg.a import COUNT\n\n\ndef get():\n    return COUNT\n"}
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, **two)) == [
        "COUNT is rebound by `global` but used in several new modules: pkg.a, pkg.b"]


def test_a_name_callers_take_from_the_old_module_stays_there(tmp_path: Path) -> None:
    old = "def run():\n    return _x()\n\n\ndef _x():\n    return 1\n"
    good = {"pkg__init": "from pkg.impl import run\n\n__all__ = ['run']\n", "pkg__impl": old}
    callers = {"pkg": {"run", "impl", "__name__"}}    # the submodule and a dunder are not names of the old module
    assert split_check.check_texts({"pkg.py": old}, _texts(tmp_path, **good), callers=callers) == []
    bad = dict(good, pkg__init='"""Package."""\n')
    assert split_check.check_texts({"pkg.py": old}, _texts(tmp_path, **bad), callers=callers) == [
        "kept name run of pkg resolves to missing run, was pkg.py statement 0"]
    listed = old + "\n__all__ = ['run']\n"     # a name in the old __all__ is kept without callers
    bad = dict(good, pkg__init="__all__ = ['run']\n")
    assert split_check.check_texts({"pkg.py": listed}, _texts(tmp_path, **bad)) == [
        "kept name run of pkg resolves to missing run, was pkg.py statement 0"]


def test_the_callers_of_the_old_module_are_found(tmp_path: Path) -> None:
    found = split_check._callers({"flextool.update_flextool.db_migration"}, [])
    assert {"migrate_database", "MigrationCancelled"} <= found["flextool.update_flextool.db_migration"]


def test_a_lost_future_import_fails(tmp_path: Path) -> None:
    old = "from __future__ import annotations\n\n\ndef f(x: Thing) -> Thing:\n    return x\n"
    assert split_check.check_texts({"old.py": old}, _texts(tmp_path, pkg__a=old)) == []
    errors = split_check.check_texts({"old.py": old}, _texts(tmp_path, pkg__a=old.split("\n", 1)[1]))
    assert errors == [f"{tmp_path / 'pkg' / 'a.py'}: __future__ features [] differ from old.py's ['annotations']"]


def test_an_import_inside_a_body_of_a_module_moved_whole_follows_it(tmp_path: Path) -> None:
    olds = {"pkg/app.py": "def f():\n    from pkg import helper\n    return helper.g()\n",
            "pkg/helper.py": "def g():\n    return 1\n\n\ndef h():\n    return 2\n"}
    app = "def f():\n    from pkg.app import helper\n    return helper.g()\n"
    whole = _texts(tmp_path, pkg__app__init=app, pkg__app__helper=olds["pkg/helper.py"])
    assert split_check.check_texts(olds, whole, root=tmp_path) == []
    part = _texts(tmp_path, pkg__app__init=app, pkg__app__helper="def g():\n    return 1\n",
                  pkg__app__rest="def h():\n    return 2\n")
    assert split_check.check_texts(olds, part, root=tmp_path) == [
        f"{tmp_path / 'pkg' / 'app' / '__init__.py'}:1 f: imports inside it bind [('helper', 'module "
        "pkg.app.helper')], was [('helper', 'module pkg.helper')]"]


def test_effect_free_statements_may_change_order_across_modules(tmp_path: Path) -> None:
    old = "A = 'a'\nB: tuple[str, ...] = ('b', A)\n"
    texts = _texts(tmp_path, pkg__a="from pkg.b import A\n\nB: tuple[str, ...] = ('b', A)\n", pkg__b="A = 'a'\n")
    assert split_check.check_texts({"old.py": old}, texts) == []
