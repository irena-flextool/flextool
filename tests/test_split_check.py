"""``tools/split_check.py``: a split that only moves code passes; each kind of change is caught."""
from __future__ import annotations

import importlib.util
import os
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
    found, patched = split_check._callers({"flextool.update_flextool.db_migration"}, [])
    assert {"migrate_database", "MigrationCancelled"} <= found["flextool.update_flextool.db_migration"]
    assert "migrate_database" in patched["flextool.update_flextool.db_migration"]   # a monkeypatch seam


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


# --- review cases: each split changes behaviour and must fail -----------------------------------------------------

def _run(tmp_path: Path, old: str | dict[str, str], new: dict[str, str], **kw) -> list[str]:
    """``check_texts`` of *old* (``pkg.py`` unless named) against ``{"pkg/a.py": text}`` under *tmp_path*."""
    olds = old if isinstance(old, dict) else {"pkg.py": old}
    errors = split_check.check_texts(olds, {tmp_path.joinpath(*k.split("/")): v for k, v in new.items()},
                                     root=tmp_path, **kw)
    return [e.replace(str(tmp_path) + os.sep, "").replace(os.sep, "/") for e in errors]


def test_a_star_import_binds_only_public_names_and_rebinds(tmp_path: Path) -> None:
    old = "_X = 1\n\n\ndef f():\n    return _X\n"
    new = {"pkg/__init__.py": "from pkg.b import f\n", "pkg/a.py": "_X = 1\n",
           "pkg/b.py": "from pkg.a import *\n\n\ndef f():\n    return _X\n"}
    assert _run(tmp_path, old, new) == ["pkg/b.py:4 f: _X resolves to missing _X, was pkg.py statement 0"]
    old = "X = 1\n\n\ndef f():\n    return X\n"
    new = {"pkg/__init__.py": "from pkg.b import f, X\n", "pkg/a.py": "from os import sep as X\n",
           "pkg/b.py": "X = 1\nfrom pkg.a import *\n\n\ndef f():\n    return X\n"}
    assert "pkg/b.py:5 f: X resolves to os.sep, was pkg.py statement 0" in _run(tmp_path, old, new)


def test_a_dotted_import_is_not_its_parent(tmp_path: Path) -> None:
    old = "import importlib.util\n\n\ndef f(p):\n    return importlib.util.find_spec(p)\n"
    new = {"pkg/__init__.py": "import importlib\n\n\ndef f(p):\n    return importlib.util.find_spec(p)\n"}
    assert _run(tmp_path, old, new) == [
        "pkg/__init__.py:4 f: importlib resolves to importlib, was importlib (import importlib.util)",
        "dropped import: pkg.py: import importlib.util"]


def test_an_extracted_body_that_updates_or_deletes_an_old_local_fails(tmp_path: Path) -> None:
    for stmt in ("count += 1", "del count"):
        old = f"def run(db):\n    count = db\n    if db:\n        {stmt}\n    return db\n"
        new = {"pkg/__init__.py": f"def _g():\n    {stmt}\n\n\ndef run(db):\n    count = db\n    if db:\n"
                                  "        _g()\n    return db\n"}
        assert _run(tmp_path, old, new) == ["_g: reads locals of run: ['count']"]


def test_an_extracted_body_binding_a_name_before_reading_it_passes(tmp_path: Path) -> None:
    def texts(bind: list[str]) -> tuple[str, dict[str, str]]:
        body = ["for item in db:", "    try:", *bind, "    except ValueError:", "        continue",
                "    print(v, {item for item in db})"]
        old = ("def run(db):\n    v = 0\n    for x in db:\n        if x:\n"
               + "".join(" " * 12 + line + "\n" for line in body) + "    return db\n")
        new = ("def _g(db):\n" + "".join(" " * 4 + line + "\n" for line in body)
               + "\n\ndef run(db):\n    v = 0\n    for x in db:\n        if x:\n            _g(db)\n    return db\n")
        return old, {"pkg/__init__.py": new}
    assert _run(tmp_path, *texts(["        v = float(item)"])) == []     # v and item bound in _g before read
    assert _run(tmp_path, *texts(["        if item:", "            v = float(item)"])) == [
        "_g: reads locals of run: ['v']"]                                 # v bound on one path only


def test_an_extracted_bodys_relative_import_reads_its_own_package(tmp_path: Path) -> None:
    old = {"pkg/old.py": "def run(db):\n    if db:\n        from .util import x\n        x(db)\n    return db\n"}
    runner = "from pkg.old.sub.step import _g\n\n\ndef run(db):\n    if db:\n        _g(db)\n    return db\n"
    good = {"pkg/old/__init__.py": runner, "pkg/old/sub/__init__.py": "",
            "pkg/old/sub/step.py": "def _g(db):\n    from ...util import x\n    x(db)\n"}
    assert _run(tmp_path, old, good) == []
    bad = dict(good, **{"pkg/old/sub/step.py": "def _g(db):\n    from ..util import x\n    x(db)\n"})
    assert _run(tmp_path, old, bad)[0] == "changed: run (old line 1, now pkg/old/__init__.py:4)"


def test_a_name_rebound_by_global_through_an_import_or_a_re_export_fails(tmp_path: Path) -> None:
    old = "def init():\n    global np\n    import numpy as np\n\n\ndef use():\n    return np.zeros(3)\n"
    new = {"pkg/__init__.py": "from pkg.a import init\nfrom pkg.b import use\n",
           "pkg/a.py": "def init():\n    global np\n    import numpy as np\n",
           "pkg/b.py": "def use():\n    return np.zeros(3)\n"}
    assert _run(tmp_path, old, new) == ["np is rebound by `global` but used in several new modules: pkg.a, pkg.b"]
    old = "COUNT = 0\n\n\ndef bump():\n    global COUNT\n    COUNT += 1\n"
    new = {"pkg/__init__.py": "from pkg.a import COUNT, bump\n", "pkg/a.py": old}
    assert _run(tmp_path, old, new, callers={"pkg": {"COUNT", "bump"}}) == [
        "COUNT is rebound by `global` but used in several new modules: pkg, pkg.a"]


def test_a_patched_name_read_outside_the_old_module_fails(tmp_path: Path) -> None:
    old = "def load():\n    return 1\n\n\ndef run():\n    return load()\n"
    new = {"pkg/__init__.py": "from pkg.a import load, run\n", "pkg/a.py": old}
    assert _run(tmp_path, old, new, callers={"pkg": {"load"}}, patched={"pkg": {"load"}}) == [
        "callers patch pkg.load, but moved code reading load in pkg.a no longer sees the patch"]


def test_names_callers_take_need_a_module_of_the_old_name(tmp_path: Path) -> None:
    new = {"pkg/impl.py": "def run():\n    return 1\n"}
    assert _run(tmp_path, "def run():\n    return 1\n", new, callers={"pkg": {"run"}}) == [
        "no new module is named pkg, but callers or its __all__ take names from it: run"]


def test_import_time_effects_in_an_unimported_module_fail(tmp_path: Path) -> None:
    old = "REG = []\nREG.append(1)\n\n\ndef f():\n    return REG\n"
    new = {"pkg/__init__.py": "from pkg.a import REG, f\n", "pkg/a.py": "REG = []\n\n\ndef f():\n    return REG\n",
           "pkg/effects.py": "from pkg.a import REG\nREG.append(1)\n"}
    assert _run(tmp_path, old, new) == [
        "pkg.effects holds import-time effects of pkg.py (statement 1) but an import of pkg does not run it"]


def test_an_effect_moved_before_a_side_effect_import_fails(tmp_path: Path) -> None:
    old = "import plugin_a\nimport registry\nX = registry.get('a')\n"
    new = {"pkg/__init__.py": "import registry\nX = registry.get('a')\nimport plugin_a\n"}
    assert _run(tmp_path, old, new) == [
        "pkg.py statement 0 now runs before the import of plugin_a (an import only for its effects)"]


def test_a_copy_of_mutable_state_keeps_its_order(tmp_path: Path) -> None:
    old = "REG = []\nREG.append(1)\nSNAP = tuple(REG)\n"
    new = {"pkg/__init__.py": "from pkg.a import REG, SNAP\nfrom pkg import b\n",
           "pkg/a.py": "REG = []\nSNAP = tuple(REG)\n", "pkg/b.py": "from pkg.a import REG\nREG.append(1)\n"}
    assert _run(tmp_path, old, new) == [
        "statements with import-time effects of pkg.py run out of their old order: statement 2 before 1"]
    old = "A = ('a',)\nB = ('b',)\nREG = []\nREG.append(1)\nAB = A + B\n"     # immutable constants: order free
    new = {"pkg/__init__.py": "from pkg.a import A, B, AB, REG\nfrom pkg import b\n",
           "pkg/a.py": "A = ('a',)\nB = ('b',)\nREG = []\nAB = A + B\n",
           "pkg/b.py": "from pkg.a import REG\nREG.append(1)\n"}
    assert _run(tmp_path, old, new) == []


def test_a_walrus_in_a_comprehension_binds_outside_it(tmp_path: Path) -> None:
    old = "ok = any((hit := x) > 2 for x in [1, 5])\n\n\ndef f():\n    return hit\n"
    new = {"pkg/__init__.py": "from pkg.a import ok\nfrom pkg.b import f\n",
           "pkg/a.py": "ok = any((hit := x) > 2 for x in [1, 5])\n", "pkg/b.py": "def f():\n    return hit\n"}
    assert _run(tmp_path, old, new) == ["pkg/b.py:1 f: hit resolves to missing hit, was pkg.py statement 0"]
    old = "def run(xs):\n    seen = [(y := v) for v in xs]\n    if seen:\n        print(y)\n    return seen\n"
    new = {"pkg/__init__.py": "def _g():\n    print(y)\n\n\ndef run(xs):\n    seen = [(y := v) for v in xs]\n"
                              "    if seen:\n        _g()\n    return seen\n"}
    assert _run(tmp_path, old, new) == ["_g: reads locals of run: ['y']"]


def test_a_class_body_reads_a_name_it_binds_later_from_the_module(tmp_path: Path) -> None:
    old = "Y = 1\n\n\nclass C:\n    x = Y\n    Y = 2\n"
    new = {"pkg/__init__.py": "from pkg.a import Y\nfrom pkg.b import C\n", "pkg/a.py": "Y = 1\n",
           "pkg/b.py": "class C:\n    x = Y\n    Y = 2\n"}
    assert _run(tmp_path, old, new) == ["pkg/b.py:1 C: Y resolves to missing Y, was pkg.py statement 0"]


def test_an_extracted_class_body_reads_the_class_locals(tmp_path: Path) -> None:
    old = "def make():\n    class K:\n        y = 1\n        if y:\n            print(y)\n    return K\n"
    new = {"pkg/__init__.py": "def _g():\n    print(y)\n\n\ndef make():\n    class K:\n        y = 1\n        if y:\n"
                              "            _g()\n    return K\n"}
    assert _run(tmp_path, old, new) == ["_g: reads locals of make: ['y']"]


def test_annotations_resolve_and_type_checking_imports_may_repeat(tmp_path: Path) -> None:
    head = "from __future__ import annotations\n\n"
    old = (head + "from dataclasses import dataclass\nfrom typing import ClassVar\n\n\n@dataclass\nclass C:\n"
           "    x: ClassVar[int] = 1\n\n\ndef g(v: ClassVar):\n    return v\n")
    new = {"pkg/__init__.py": head + "from typing import ClassVar\n\nfrom pkg.b import C\n\n\n"
                                     "def g(v: ClassVar):\n    return v\n",
           "pkg/b.py": head + "from dataclasses import dataclass\n\n\n@dataclass\nclass C:\n    x: ClassVar[int] = 1\n"}
    assert _run(tmp_path, old, new) == ["pkg/b.py:7 C: ClassVar resolves to missing ClassVar, was typing.ClassVar"]
    block = head + "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pathlib import Path\n\n\n"
    old = block + "def f(p: Path) -> None:\n    pass\n\n\ndef g(p: Path) -> None:\n    pass\n"
    new = {"pkg/__init__.py": head + "from pkg.a import f\nfrom pkg.b import g\n",
           "pkg/a.py": block + "def f(p: Path) -> None:\n    pass\n",
           "pkg/b.py": block + "def g(p: Path) -> None:\n    pass\n"}
    assert _run(tmp_path, old, new) == []


def test_a_failure_accepted_on_purpose_is_listed_not_counted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(split_check, "check", lambda *a, **k: ["changed: f", "callers patch pkg.x"])
    assert split_check.main(["--old", "R:pkg.py", "pkg", "--accept", r"^callers patch pkg\.x$"]) == 1
    out = capsys.readouterr().out
    assert "FAIL changed: f" in out and "accepted on purpose: callers patch pkg.x" in out
    assert "split check: 1 failure(s)" in out
    assert split_check.main(["--old", "R:pkg.py", "pkg", "--accept", "^changed", "--accept", "^callers"]) == 0
