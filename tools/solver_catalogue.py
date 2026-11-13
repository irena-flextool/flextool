"""Generate FlexTool's solver parameter catalogue from the solver packages (item 240 B1).

Usage (in a virtual environment with the solver packages installed; none needs a licence)::

    python tools/solver_catalogue.py --check            # every installed solver against its shipped file
    python tools/solver_catalogue.py --write            # regenerate (merge into) the shipped files
    python tools/solver_catalogue.py --write --solver gurobi

How each solver's parameters are listed (no solver is started):

* HiGHS (highspy): every identifier in ``libhighs`` and each of its ``_``-suffixes (the linker merges
  ``solution_file`` into ``read_solution_file``), kept where ``Highs().getOptionType`` answers ``kOk``; this also
  finds the advanced options that ``writeOptions`` leaves out.
* Gurobi (gurobipy): the C library's ``GRBgetnumparams`` / ``GRBgetparamname`` / ``GRBgetparamtype`` (ctypes on
  the ``libgurobi`` inside gurobipy, an empty environment, no licence), plus the parameters of
  ``gurobipy._paramdetails`` that the library lacks (``gurobi_cl``'s own: ``InputFile``, the setting-file lists).
* CPLEX (cplex): the ``Parameter`` leaves of ``Cplex().parameters``; the path joined by spaces is the Interactive
  Optimizer's ``set`` name.  Plus the Interactive Optimizer's own names from IBM's documentation
  (:data:`DOCUMENTED`).
* Xpress (xpress): the keys of ``problem().getControl()``; the type is the value's Python type.
* COPT (coptpy): the attributes of ``COPT.Param``; the type from ``Model.getParamInfo`` (``unknown`` for the
  per-objective parameters it refuses).

Every name needs a reviewed class in ``tools/solver_catalogue_review.json`` (see
:mod:`flextool.engine.solve._solver_catalogue` for the classes).  Every text-valued (or unknown-typed) name and
every class other than the one the name rules give needs a note there that says why.  ``--write`` stops on a name
without a review and prints a proposed entry; nothing is written until every name is reviewed.

The new listing is merged into the shipped file: a name is never removed, each entry records the versions it was
seen in, and ``generated_from`` lists every package version merged.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.metadata
import json
import os
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from flextool.engine.solve._solver_catalogue import (  # noqa: E402
    CATALOGUE_FOLDER,
    CLASSES,
    FORMAT,
    REASONS,
    SOLVERS,
    catalogue_key,
    rule_class,
)

CATALOGUE_DIR = REPO / "flextool" / CATALOGUE_FOLDER
REVIEW_FILE = REPO / "tools" / "solver_catalogue_review.json"

#: Names a solver's console reads that its Python package does not list, with their source.
DOCUMENTED: dict[str, dict[str, dict[str, str]]] = {
    "cplex": {
        "logfile": {"name": "logfile", "type": "string",
                    "source": "IBM Documentation, CPLEX Interactive Optimizer, 'Table of the commands of the "
                              "Interactive Optimizer': set logfile <filename>"},
        "defaults": {"name": "defaults", "type": "none",
                     "source": "IBM Documentation, CPLEX Interactive Optimizer, 'Table of the commands of the "
                               "Interactive Optimizer': set defaults (resets every parameter; takes no value)"},
        "distmip config": {"name": "distmip config", "type": "string",
                           "source": "IBM Documentation, CPLEX Interactive Optimizer, 'Table of the commands of "
                                     "the Interactive Optimizer': set distmip config (distributed MIP: the "
                                     "machines to solve on)"},
    },
    "copt": {
        "logfile": {"name": "LogFile", "type": "string",
                    "source": "FlexTool 4.3 path list: the copt_cmd console's log setting (not a parameter of "
                              "coptpy 8.0.4's COPT.Param), listed in case"},
    },
}
#: Where the documented names come from, kept in the catalogue file.
SOURCES = {
    "cplex": "https://www.ibm.com/docs/SSSA5P_20.1.0/ilog.odms.cplex.help/CPLEX/InteractiveOptimizer/topics/"
             "commands.html",
}


# ---------------------------------------------------------------------------
# Listing each solver's parameters: (versions, {key: (name, type)})
# ---------------------------------------------------------------------------

Listing = tuple[dict[str, str], dict[str, tuple[str, str]]]


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def list_highs() -> Listing:
    import highspy

    h = highspy.Highs()
    h.silent()
    folder = Path(highspy.__file__).parent
    libs = sorted(glob.glob(str(folder / "libhighs*")) + glob.glob(str(folder / "highs*.dll")))
    if not libs:
        raise SystemExit(f"no libhighs in {folder}")
    candidates: set[str] = set()
    for lib in libs:
        for match in re.finditer(rb"[a-z][a-z0-9_]{2,63}", Path(lib).read_bytes()):
            word = match.group().decode()
            candidates.add(word)
            candidates.update(word[i + 1:] for i, ch in enumerate(word) if ch == "_" and word[i + 1:i + 2].isalpha())
    types = {highspy.HighsOptionType.kBool: "bool", highspy.HighsOptionType.kInt: "int",
             highspy.HighsOptionType.kDouble: "double", highspy.HighsOptionType.kString: "string"}
    params: dict[str, tuple[str, str]] = {}
    for name in sorted(candidates):
        status, kind = h.getOptionType(name)
        if status == highspy.HighsStatus.kOk:
            params[name] = (name, types[kind])
    version = h.version() if hasattr(h, "version") else _version("highspy")
    return {"package": "highspy", "version": _version("highspy"), "library": str(version)}, params


def _gurobi_library():
    import gurobipy

    folder = Path(gurobipy.__file__).parent
    patterns = ("libgurobi*.so*", "libgurobi*.dylib", "gurobi*.dll")
    libs = sorted(p for base in (folder / ".libs", folder) for pat in patterns for p in base.glob(pat))
    if not libs:
        raise SystemExit(f"no Gurobi library in {folder}")
    return ctypes.CDLL(str(libs[0]))


def list_gurobi() -> Listing:
    from gurobipy._paramdetails import param_details

    lib = _gurobi_library()
    lib.GRBemptyenv.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.GRBgetnumparams.argtypes = [ctypes.c_void_p]
    lib.GRBgetparamname.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.GRBgetparamtype.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.GRBfreeenv.argtypes = [ctypes.c_void_p]
    lib.GRBversion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
    env = ctypes.c_void_p()
    if lib.GRBemptyenv(ctypes.byref(env)) != 0:
        raise SystemExit("GRBemptyenv failed")
    names: dict[str, str] = {}
    try:
        for i in range(lib.GRBgetnumparams(env)):
            p = ctypes.c_char_p()
            if lib.GRBgetparamname(env, i, ctypes.byref(p)) != 0:
                raise SystemExit(f"GRBgetparamname({i}) failed")
            names[p.value.decode()] = {1: "int", 2: "double", 3: "string"}[lib.GRBgetparamtype(env, p.value)]
    finally:
        lib.GRBfreeenv(env)
    for name, details in param_details.items():  # gurobi_cl's own parameters
        names.setdefault(name, details["values"]["type"])
    major, minor, tech = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    lib.GRBversion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(tech))
    params = {catalogue_key("gurobi", n): (n, t) for n, t in sorted(names.items())}
    if len(params) != len(names):
        raise SystemExit("two Gurobi parameter names share one key")
    library = f"{major.value}.{minor.value}.{tech.value}"
    return {"package": "gurobipy", "version": _version("gurobipy"), "library": library}, params


def list_cplex() -> Listing:
    import cplex
    from cplex._internal._parameter_classes import Parameter, ParameterGroup

    model = cplex.Cplex()  # the parameter objects hold only a weak reference to it
    for stream in (model.set_log_stream, model.set_results_stream, model.set_warning_stream):
        stream(None)
    params: dict[str, tuple[str, str]] = {}

    def walk(group, path: list[str]) -> None:
        for attr in (a for a in dir(group) if not a.startswith("_")):
            child = getattr(group, attr, None)
            if isinstance(child, Parameter):
                name = " ".join([*path, attr])
                params[catalogue_key("cplex", name)] = (name, {int: "int", float: "double", str: "string"}[
                    type(child.default())])
            elif isinstance(child, ParameterGroup):
                walk(child, [*path, attr])

    walk(model.parameters, [])
    return {"package": "cplex", "version": _version("cplex"), "library": model.get_version()}, params


def list_xpress() -> Listing:
    warnings.simplefilter("ignore")
    import xpress as xp

    params: dict[str, tuple[str, str]] = {}
    for key, value in xp.problem().getControl().items():
        kind = "string" if isinstance(value, str) else "double" if isinstance(value, float) else "int"
        params[catalogue_key("xpress", key)] = (key.upper(), kind)
    return {"package": "xpress", "version": _version("xpress"), "library": str(xp.getversion())}, params


def list_copt() -> Listing:
    import coptpy as cp

    model = cp.Envr().createModel("catalogue")
    model.setParam("Logging", 0)
    params: dict[str, tuple[str, str]] = {}
    for name in (n for n in dir(cp.COPT.Param) if not n.startswith("_")):
        try:
            info = model.getParamInfo(getattr(cp.COPT.Param, name))
            kind = {int: "int", float: "double", str: "string"}[type(info[1])]
        except AttributeError:  # the multi-objective parameters are per objective, not model parameters
            kind = "unknown"
        params[catalogue_key("copt", name)] = (name, kind)
    library = f"{cp.COPT.VERSION_MAJOR}.{cp.COPT.VERSION_MINOR}.{cp.COPT.VERSION_TECHNICAL}"
    return {"package": "coptpy", "version": _version("coptpy"), "library": library}, params


LISTERS = {"highs": list_highs, "gurobi": list_gurobi, "cplex": list_cplex, "xpress": list_xpress,
           "copt": list_copt}


# ---------------------------------------------------------------------------
# Review and merge
# ---------------------------------------------------------------------------


def load_review(path: Path = REVIEW_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed_class(review: dict, solver: str, key: str) -> tuple[str, str | None] | None:
    """``(class, reason)`` the review gives *key*, else ``None``."""
    section = review.get(solver, {})
    for cls in ("plain", "file", "flextool"):
        if key in section.get(cls, []):
            return cls, None
    if key in section.get("local_only", {}):
        return "local_only", section["local_only"][key]
    return None


def review_problems(review: dict, solver: str, params: dict[str, tuple[str, str]]) -> list[str]:
    """What is wrong with *review* for the catalogue *params* (``{key: (name, type)}``): unreviewed names, names
    listed twice, unknown reasons, and text / unknown-typed names or rule overrides without a note."""
    section = review.get(solver, {})
    problems: list[str] = []
    seen: dict[str, str] = {}
    for cls in CLASSES:
        keys = section.get(cls, {} if cls == "local_only" else [])
        for key in keys:
            if key in seen:
                problems.append(f"{solver}: {key!r} is reviewed as both {seen[key]} and {cls}")
            seen[key] = cls
    for key, reason in section.get("local_only", {}).items():
        if reason not in REASONS:
            problems.append(f"{solver}: {key!r} has the reason {reason!r} (one of {', '.join(REASONS)})")
    notes = section.get("notes", {})
    for key, (_name, kind) in sorted(params.items()):
        got = reviewed_class(review, solver, key)
        if got is None:
            problems.append(f"{solver}: {key!r} ({kind}) is not reviewed")
            continue
        rule = rule_class(solver, key)
        if kind in ("string", "unknown", "none") and key not in notes:
            problems.append(f"{solver}: {key!r} is a {kind} parameter, reviewed as {got[0]} without a note")
        if rule is not None and got != rule and key not in notes:
            problems.append(f"{solver}: {key!r} matches the name rule {rule} but is reviewed as {got} without a note")
        if rule is None and got[0] != "plain" and key not in notes:
            problems.append(f"{solver}: {key!r} is reviewed as {got} without a note")
    for key in sorted(set(seen) - set(params)):
        problems.append(f"{solver}: the review lists {key!r}, which is not a catalogue name")
    return problems


def proposal(solver: str, key: str, kind: str) -> str:
    rule = rule_class(solver, key)
    cls = rule if rule is not None else ("plain", None)
    return f"  {solver}: {key!r} ({kind}) -> {cls[0]}{'/' + cls[1] if cls[1] else ''}"


def merge(solver: str, shipped: dict | None, versions: dict[str, str] | None,
          listed: dict[str, tuple[str, str]], review: dict) -> dict:
    """The catalogue file of *solver*: *shipped* plus the new listing (names never removed; the reviewed class
    for every name)."""
    params: dict[str, dict] = {k: dict(v) for k, v in (shipped or {}).get("parameters", {}).items()}
    generated = list((shipped or {}).get("generated_from", []))
    if versions is not None and versions not in generated:
        generated.append(versions)
    for key, (name, kind) in listed.items():
        item = params.setdefault(key, {"name": name, "seen_in": []})
        item["type"] = kind
        if versions is not None and versions["library"] not in item["seen_in"]:
            item["seen_in"].append(versions["library"])
    for key, doc in DOCUMENTED.get(solver, {}).items():
        item = params.setdefault(key, {"name": doc["name"], "seen_in": []})
        item.update(type=doc["type"], source=doc["source"])
    for key, item in params.items():
        cls, reason = reviewed_class(review, solver, key) or ("?", None)
        item["class"] = cls
        item.pop("reason", None)
        if reason:
            item["reason"] = reason
    out = {"format": FORMAT, "solver": solver, "generated_from": generated}
    if solver in SOURCES:
        out["sources"] = [SOURCES[solver]]
    out["parameters"] = {k: {f: params[k][f] for f in ("name", "type", "class", "reason", "seen_in", "source")
                             if f in params[k]} for k in sorted(params)}
    return out


def dump(data: dict) -> str:
    """The file text: one parameter per line, so a regeneration diffs by name."""
    head = {k: v for k, v in data.items() if k != "parameters"}
    lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
    lines.append('  "parameters": {')
    items = list(data["parameters"].items())
    for i, (key, item) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"    {json.dumps(key)}: {json.dumps(item, ensure_ascii=False)}{comma}")
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def shipped(solver: str) -> dict | None:
    path = CATALOGUE_DIR / f"{solver}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def check(solver: str, listed: dict[str, tuple[str, str]], review: dict) -> list[str]:
    """Differences between the installed package's listing and the shipped file: missing names, other types or
    classes (not the version text, so a patch release with the same parameters passes)."""
    current = shipped(solver)
    if current is None:
        return [f"{solver}: no shipped catalogue"]
    problems = review_problems(review, solver, {k: (v["name"], v["type"]) for k, v in current["parameters"].items()})
    for key, (_name, kind) in sorted(listed.items()):
        item = current["parameters"].get(key)
        if item is None:
            problems.append(f"{solver}: {key!r} ({kind}) is not in the shipped catalogue")
            continue
        if item["type"] != kind:
            problems.append(f"{solver}: {key!r} is {kind} in the package, {item['type']} in the catalogue")
        expected = reviewed_class(review, solver, key)
        if expected is not None and (item["class"], item.get("reason")) != expected:
            problems.append(f"{solver}: {key!r} is {item['class']} in the catalogue, {expected} in the review")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--solver", action="append", choices=SOLVERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    review = load_review()
    failed = False
    for solver in args.solver or SOLVERS:
        try:
            versions, listed = LISTERS[solver]()
        except ImportError as exc:
            print(f"{solver}: package not installed ({exc}); skipped")
            if args.solver:
                failed = True
            continue
        if args.check:
            problems = check(solver, listed, review)
            for line in problems:
                print(line)
            failed |= bool(problems)
            print(f"{solver} {versions['library']}: {'differs' if problems else 'matches'} ({len(listed)} names)")
            continue
        data = merge(solver, shipped(solver), versions, listed, review)
        all_params = {k: (v["name"], v["type"]) for k, v in data["parameters"].items()}
        problems = review_problems(review, solver, all_params)
        if problems:
            failed = True
            print(f"{solver}: not written; review {REVIEW_FILE.relative_to(REPO)} first:")
            for line in problems:
                print("  " + line)
            for key in sorted(k for k in all_params if reviewed_class(review, solver, k) is None):
                print(proposal(solver, key, all_params[key][1]))
            continue
        CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        (CATALOGUE_DIR / f"{solver}.json").write_text(dump(data), encoding="utf-8")
        print(f"{solver} {versions['library']}: {len(data['parameters'])} names written")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    sys.exit(main())
