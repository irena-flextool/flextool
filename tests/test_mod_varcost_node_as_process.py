"""Regression test for flextool.mod — ``process_sink_toProcess`` ordering.

MathProg ``set A := { sink in node, p in process, p2 in process ... }``
emits positional tuples ``(sink, p, p2)`` — a **node** in column 1.  The
sibling ``process_sink_toProcess_direct`` and every other ``*toProcess`` /
``process_to*`` set put a process in column 1, and all downstream
consumers (``process_source_sink``, ``process_source_sink_noEff``,
``process_source_sink_alwaysProcess``, each iterated as ``(p, source,
sink) in X`` with ``p`` assumed to be a process) rely on that convention.

The user-reported crash was ``pdtProcess[ARGH2, other_operational_cost,
y_test, '2050-01-01T00:00:00'] out of domain`` at ``total_cost``
generation, triggered by a bidirectional H2 pipeline (method
``method_2way_nvar_off``, node ``ARGH2`` as both source and sink).  The
pre-fix ordering emitted ``(ARGH2, BRA_ARG_H2, BRA_ARG_H2)`` into
``process_source_sink_noEff``; ``pdtProcess__source__sink__dt_varCost``
(declared over ``process_source_sink``) then called ``pdtProcess[p='ARGH2',
'other_operational_cost', d, t]`` and ``ARGH2 ∉ process`` → out of
domain.

The test **extracts the live ``process_sink_toProcess`` block from
flextool.mod** and runs it against a minimal MathProg harness that
reproduces the downstream chain.  A regression on line 871 flips the
extracted block back to the node-first ordering and the test fails
with the original ``out of domain`` message — or with the col-1
orientation assertion below.

Δ.21 status — GMPL/glpsol retired
---------------------------------

The native cascade (``flextool.engine_polars``) does NOT consume
``flextool.mod``; the LP is built directly from FlexData via
``model.build_flextool``.  ``process_sink_toProcess`` regressions are
therefore caught by the engine_polars LP-builder tests instead.

Both the static text assertion and the live-glpsol harness in this
file remain pinned to the ``flextool.mod`` source.  Δ.22 will delete
``flextool.mod`` and ``bin/glpsol``; both tests will be removed at
that point.  Until then the live-glpsol harness is skip-marked when
the binary or the .mod file is missing (the static test still runs as
long as ``flextool.mod`` is on disk).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GLPSOL = REPO_ROOT / "bin" / "glpsol"
FLEXTOOL_MOD = REPO_ROOT / "flextool" / "flextool.mod"


def _extract_block(text: str, set_name: str) -> str:
    """Return the full ``set <name> := { ... };`` block from flextool.mod.

    MathProg ``set`` declarations are terminated by ``;`` at the end of
    the block — we scan from ``set <name> :=`` to the first line
    ending in ``};``.  Braces are not nested in the flextool.mod
    definitions we target.
    """
    start = re.search(rf"^\s*set\s+{re.escape(set_name)}\s*:=", text, re.MULTILINE)
    assert start is not None, f"set {set_name} not found in flextool.mod"
    tail = text[start.start():]
    end = re.search(r"\n\s*}\s*;", tail)
    assert end is not None, f"end of set {set_name} block not found"
    return tail[: end.end()]


def _extract_param_block(text: str, name: str) -> str:
    """Return the full ``param <name> ... := ... ;`` declaration."""
    start = re.search(rf"^\s*param\s+{re.escape(name)}\b", text, re.MULTILINE)
    assert start is not None, f"param {name} not found"
    tail = text[start.start():]
    end = re.search(r";\s*\n", tail)
    assert end is not None
    return tail[: end.end()]


@pytest.fixture(scope="module")
def _mod_source() -> str:
    if not FLEXTOOL_MOD.exists():
        pytest.skip(
            f"flextool.mod not present at {FLEXTOOL_MOD}; the GMPL "
            "model file is being retired in Δ.22.  See progress.md "
            "Δ.21 close stanza."
        )
    text = FLEXTOOL_MOD.read_text()
    # ``process_sink_toProcess`` migrated to Python preprocessing
    # (``flextool/flextoolrunner/preprocessing/process_method_sets.py``)
    # — flextool.mod now reads it from a CSV via ``table data IN ...``
    # instead of defining the set inline.  The harness here used to
    # extract the inline ``set process_sink_toProcess := { ... }``
    # block; once the inline definition was removed, the regression
    # this test guarded moved upstream into the Python preprocessing
    # layer (which has its own coverage).  Skip until Δ.22 deletes the
    # whole file.
    if not re.search(r"^\s*set\s+process_sink_toProcess\s*:=",
                     text, re.MULTILINE):
        pytest.skip(
            "process_sink_toProcess no longer defined inline in "
            "flextool.mod (migrated to "
            "flextool/flextoolrunner/preprocessing/process_method_sets.py "
            "in Γ.* sweeps).  The Python equivalent is exercised by "
            "tests/engine_polars/ — this glpsol-binary harness is now "
            "redundant and will be deleted with flextool.mod in Δ.22."
        )
    return text


@pytest.fixture
def _requires_glpsol() -> None:
    # Δ.21: GMPL/glpsol path retired from the production CLI.  The
    # binary may still ship for one release cycle, but skip-mark when
    # absent so this regression test stays green on glpsol-free dev
    # boxes.  Δ.22 will delete this test along with flextool.mod.
    if not GLPSOL.exists():
        pytest.skip(f"glpsol binary not present at {GLPSOL}")


# ---------------------------------------------------------------------------
# Static text assertion — cheap, catches the literal regression.
# ---------------------------------------------------------------------------


def test_process_sink_toProcess_definition_places_process_first(
    _mod_source: str,
) -> None:
    """Guard the exact structural convention: the first quantifier
    must draw ``p`` from a process-bearing set, not a node-only one.
    """
    block = _extract_block(_mod_source, "process_sink_toProcess")
    # The fix: first quantifier is (p, sink) in process_sink, so the
    # positional tuple is (p, sink, p2).  The bug: `sink in node` as
    # the first quantifier, making the tuple (sink, p, p2).
    header = block.splitlines()[1].strip()  # first line after `set := `
    assert "sink in node" not in header, (
        "Regression: process_sink_toProcess first quantifier is "
        f"`sink in node` — puts a node in column 1.  Block:\n{block}"
    )
    assert re.search(r"\(\s*p\s*,\s*sink\s*\)\s+in\s+process_sink", header), (
        "process_sink_toProcess must start with "
        "`(p, sink) in process_sink, p2 in process` so column 1 is a "
        f"process.  Block:\n{block}"
    )


# ---------------------------------------------------------------------------
# Dynamic glpsol assertion — embeds the LIVE flextool.mod definitions
# of process_sink_toProcess + pdtProcess__source__sink__dt_varCost into
# a minimal harness, so any regression propagates to glpsol failure.
# ---------------------------------------------------------------------------


_HARNESS_HEADER = r"""
set entity dimen 1;
set node dimen 1 within entity;
set process dimen 1 within entity;
set process_source dimen 2 within {process, entity};
set process_sink   dimen 2 within {process, entity};
set processMethod dimen 1;
set process_method dimen 2 within {process, processMethod};
set processTimeParam dimen 1;
set processTimeParamRequired within processTimeParam;
set period dimen 1;
set time dimen 1;
set dt dimen 2 within {period, time};
set process__ct_method dimen 2;

set method_2way_nvar := {"method_2way_nvar_off"};
"""

_HARNESS_PRE_PARAM = r"""
set process_source_sink := process_sink_toProcess;
set process_source_sink_noEff := process_sink_toProcess;
set process_source_sink_alwaysProcess := process_sink_toProcess;

set process_TimeParam_in_use :=
  { p in process, param in processTimeParam:
    param in processTimeParamRequired ||
    ((p, 'min_load_efficiency') in process__ct_method
     && ((param == 'min_load') || (param == 'efficiency_at_min_load')))
  };

param p_process_fallback {p in process, param in processTimeParam} default 0;
param pdtProcess {(p, param) in process_TimeParam_in_use, (d, t) in dt} :=
      p_process_fallback[p, param];
param pdtProcess_source {p in process, source in entity,
                         param in processTimeParam, (d, t) in dt} default 0;
param pdtProcess_sink   {p in process, sink in entity,
                         param in processTimeParam, (d, t) in dt} default 0;
"""

_HARNESS_FOOTER = r"""
set pssdt_varCost_noEff :=
  { (p, source, sink) in process_source_sink_noEff, (d, t) in dt :
    pdtProcess__source__sink__dt_varCost[p, source, sink, d, t] };

printf "sink_toProcess_col1:";
printf {(p, s, k) in process_sink_toProcess} " %s", p;
printf "\n";

minimize trivial:
  sum {(p, source, sink, d, t) in pssdt_varCost_noEff} 1;

solve;

data;
set entity := N P;
set node := N;
set process := P;
set process_source := P N;
set process_sink   := P N;
set processMethod := method_2way_nvar_off method_1way_1var_off;
set process_method := P method_2way_nvar_off;
set processTimeParam := other_operational_cost;
set processTimeParamRequired := other_operational_cost;
set period := d1;
set time := t1;
set dt := (d1, t1);
end;
"""


def test_live_process_sink_toProcess_does_not_crash_glpsol(
    _requires_glpsol, _mod_source: str, tmp_path: Path,
) -> None:
    """Embed the **live** flextool.mod definitions of
    ``process_sink_toProcess`` and ``pdtProcess__source__sink__dt_varCost``
    into a minimal harness, then run glpsol.  If either regresses to
    put a node in the process column, the harness fires the same
    ``pdtProcess[...] out of domain`` the user hit.
    """
    sink_to_process = _extract_block(_mod_source, "process_sink_toProcess")
    dt_varCost = _extract_param_block(_mod_source, "pdtProcess__source__sink__dt_varCost")

    mod_text = "\n".join([
        _HARNESS_HEADER,
        sink_to_process,
        _HARNESS_PRE_PARAM,
        dt_varCost,
        _HARNESS_FOOTER,
    ])
    mod_path = tmp_path / "regression.mod"
    mod_path.write_text(mod_text)

    result = subprocess.run(
        [str(GLPSOL), "--check", "-m", str(mod_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = result.stdout + result.stderr

    assert "out of domain" not in out, (
        "Regression: pdtProcess[<node>, ...] out of domain.  "
        "process_sink_toProcess (line ~871) has likely reverted to the "
        "(sink, p, p2) ordering; every downstream consumer of "
        "process_source_sink* that iterates as (p, source, sink) then "
        f"sees a node in column 1.\nglpsol output:\n{out}"
    )
    assert result.returncode == 0, (
        f"glpsol exited {result.returncode}; output:\n{out}"
    )
    # Positive orientation check: col 1 must be the process.
    col1_line = next(
        (line for line in out.splitlines()
         if line.startswith("sink_toProcess_col1:")),
        None,
    )
    assert col1_line is not None, f"orientation marker missing in:\n{out}"
    col1_values = col1_line.split(":", 1)[1].split()
    assert col1_values == ["P"], (
        f"process_sink_toProcess col 1 = {col1_values!r} (expected ['P']). "
        "Every caller iterating process_source_sink* as (p, source, "
        "sink) assumes col 1 is a process."
    )
