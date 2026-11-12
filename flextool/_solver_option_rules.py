"""Solver option rules that need no engine: reading the option lines of a ``<solver>.opt`` text, the solvers'
name forms, which options name a file or folder or another computer, and the screen of an option file that came
from someone else (a project package).

The engine (:mod:`flextool.engine.solve._solver_dispatch`, :mod:`flextool.engine.solve._subprocess_solve`) uses
the same rules for the input database's options; they live here so that project import (the command line and the
FlexTool window) can screen a file without loading the engine and polars.
"""
from __future__ import annotations

import re

#: The UTF-8 byte order mark as a character: some Windows editors write it at the start of a text file.
_BOM = "\ufeff"


def _without_bom(text: str) -> str:
    """*text* without a leading byte order mark, which would otherwise become part of the first option's name."""
    return text[1:] if text.startswith(_BOM) else text


def highs_opt_lines(text: str) -> list[tuple[int, str, str]]:
    """``(line number, key, value)`` of every option line of a ``highs.opt`` text, in file order (a leading
    UTF-8 byte order mark ignored)."""
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(_without_bom(text).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        out.append((lineno, key.strip(), val.strip()))
    return out



def native_opt_lines(text: str) -> list[tuple[int, str, str]]:
    """``(line number, name, value)`` of every parameter line of a native ``.opt`` text (Gurobi, CPLEX, Xpress,
    COPT: ``name value``, the value the last word), in file order; blank lines and ``#`` comments skipped, a
    leading UTF-8 byte order mark ignored."""
    out: list[tuple[int, str, str]] = []
    for lineno, raw in enumerate(_without_bom(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line and "\t" not in line:
            continue
        name, value = line.rsplit(maxsplit=1)
        out.append((lineno, name.strip(), value.strip()))
    return out


def commercial_option_key(solver: str, name: str) -> str:
    """The comparison key of a commercial parameter name: one key for every spelling the solver reads as the
    same parameter, so that the file-path filter and the merge of the layers match what the solver does.

    Every solver: case-folded, runs of whitespace collapsed.  CPLEX: ``.`` read as a space.  Gurobi: ``_``
    after the first character removed (the Gurobi library (``GRBsetparam``, 13.0.2) reads ``Log_File``,
    ``LOG_FILE`` and ``M_I_P_Gap`` as ``LogFile`` / ``MIPGap``, but refuses a leading ``_``; it ignores no
    other character).  Xpress: one leading ``xprs_`` removed (the library's control lookup, xpress 46.01.02,
    reads ``XPRS_MAXTIME`` / ``xprs_maxtime`` as ``MAXTIME``, but not ``XPRS_XPRS_MAXTIME`` or ``XPRSMAXTIME``).
    COPT (coptpy 8.0.4) only ignores case; HiGHS names are exact.
    """
    key = str(name).strip()
    if solver == "cplex":
        key = key.replace(".", " ")
    key = " ".join(key.split()).casefold()
    if solver == "gurobi":
        key = key[:1] + key[1:].replace("_", "")
    elif solver == "xpress" and key.startswith("xprs_"):
        key = key[len("xprs_"):]
    return key


#: Words in an option name that mark a file, folder or path option (``LogFile``, ``workdir``, ``TUNEROUTPUTPATH``).
_PATH_NAME_WORDS = ("file", "dir", "path")

#: The file, folder and path options of each solver, as :func:`commercial_option_key` forms (HiGHS: lower case).
#: Many are also found by the name rule above; the list adds the rest and is the set FlexTool places in the
#: solve's work folder for HiGHS (the in-process HiGHS solve runs in the run's current folder).  A block list: a
#: parameter a later solver version adds is caught only by the name rule (item 240 has the allow-list).
_PATH_OPTIONS: dict[str, frozenset[str]] = {
    # highspy 1.15.1: every string option of ``Highs().writeOptions(<file>)`` whose description is a file
    # ("Log file", "Read solution file", "Read basis file", "Write model file", "Write solution file", "Write
    # basis file", "Write presolved model file", "Write IIS model file", "File for reporting improving MIP
    # solutions").  The other string options (presolve, solver, parallel, run_crossover, ranging, ...) are words.
    "highs": frozenset({
        "log_file", "solution_file", "write_model_file", "write_presolved_model_file", "write_iis_model_file",
        "write_basis_file", "read_basis_file", "read_solution_file", "mip_improving_solution_file",
    }),
    # gurobipy 13.0.2 ``_paramdetails.py`` (Gurobi's parameter reference): the parameters whose value is a file
    # or folder name (ResultFile, LogFile, SolFiles, NodefileDir, InputFile; the .prm / .json file lists
    # ConcurrentSettings, MultiObjSettings, TuneBaseSettings, TuneIgnoreSettings, TuneParams) and Record (writes
    # API-call recording files; an integer, so the database may set it: the files land in the work folder).
    # InputFile and the five setting-file lists are ``gurobi_cl`` parameters, not library ones.
    "gurobi": frozenset({
        "resultfile", "logfile", "solfiles", "nodefiledir", "inputfile", "concurrentsettings",
        "multiobjsettings", "tunebasesettings", "tuneignoresettings", "tuneparams", "record",
    }),
    # cplex 22.1.2.0: the string parameters of ``Cplex().parameters`` that name a file or folder (``workdir``,
    # ``output.intsolfileprefix``; ``read.fileencoding`` is a code page), plus the Interactive Optimizer's own
    # ``set logfile <file>`` (IBM CPLEX Interactive Optimizer commands reference; not in the Python API).  The
    # Interactive Optimizer takes any unique abbreviation of each word, so an abbreviation is matched too
    # (:func:`is_path_option`); a name whose first word is ``read`` / ``write`` is one as well.
    "cplex": frozenset({"workdir", "output intsolfileprefix", "logfile"}),
    # xpress 46.01.02: the string controls of ``problem().getControl()`` that name a file or folder
    # (TUNERMETHODFILE read, TUNEROUTPUTPATH written, TUNERSESSIONNAME the name of the tuner's output under it).
    # The others are MPS section names, masks and formats.  Compared without case or an ``XPRS_`` prefix.
    "xpress": frozenset({"tunermethodfile", "tuneroutputpath", "tunersessionname"}),
    # coptpy 8.0.4 (COPT 8.0.4): every parameter of ``COPT.Param`` is an integer or a float
    # (``Model.getParamInfo``); ``LogFile`` (the ``copt_cmd`` console's log setting) is listed in case.
    "copt": frozenset({"logfile"}),
}
#: CPLEX: the Interactive Optimizer command FlexTool puts before every ``cplex.opt`` line (``set <name> <value>``).
_CPLEX_SET_WORD = "set"
#: CPLEX: a name whose first word is one of these is a path option too (``read`` / ``write`` commands).
_CPLEX_PATH_FIRST_WORDS = frozenset({"read", "write"})

#: Gurobi options that connect to another machine or carry its credentials: Compute Server, Cluster Manager,
#: Instant Cloud, Web License Service, token server and distributed workers (Gurobi 12 parameter list), as
#: :func:`commercial_option_key` forms (so ``Compute_Server`` is ``ComputeServer``, as Gurobi reads it).
_SERVER_OPTIONS: dict[str, frozenset[str]] = {
    "gurobi": frozenset({
        "computeserver", "serverpassword", "servertimeout",
        "csmanager", "csapiaccessid", "csapisecret", "csappname", "csauthtoken", "csbatchmode", "csclientlog",
        "csgroup", "csidletimeout", "cspriority", "csqueuetimeout", "csrouter", "cstlsinsecure",
        "cloudaccessid", "cloudsecretkey", "cloudpool", "cloudhost",
        "wlsaccessid", "wlssecret", "wlstoken", "wlstokenduration", "wlstokenrefresh", "wlsproxy", "wlsconfig",
        "licenseid", "username", "tokenserver", "tokenport", "tsport",
        "workerpool", "workerpassword", "workerport",
    }),
}
#: Gurobi: a name with one of these prefixes is a server option too (a later Gurobi version's new ones).
_SERVER_OPTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "gurobi": ("computeserver", "server", "cs", "cloud", "wls", "token", "worker"),
}

#: A plain file name: ASCII letters, digits, ``_ . + -``, starting with a letter, digit or ``_``.
_PLAIN_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*")


def _option_key(solver: str, name: str) -> str:
    """The name as *solver* reads it: :func:`commercial_option_key`; HiGHS lower case (a superset of its exact
    names, so that no spelling slips past the database check)."""
    return str(name).strip().casefold() if solver == "highs" else commercial_option_key(solver, name)


def is_path_option(solver: str, name: object) -> bool:
    """Whether *name* is an option of *solver* that takes a file, folder or path.

    Names are compared as the solver reads them (:func:`_option_key`).  CPLEX: a name each of whose words
    abbreviates the listed name's word (``set log`` for ``set logfile``) is one too, since the Interactive
    Optimizer accepts any unique abbreviation.
    """
    key = _option_key(solver, str(name))
    listed = _PATH_OPTIONS.get(solver, frozenset())
    if key in listed or any(word in key for word in _PATH_NAME_WORDS):
        return True
    if solver != "cplex" or not key:
        return False
    words = key.split(" ")
    if words[0] == _CPLEX_SET_WORD and len(words) > 1:
        # ``set lo /tmp/x``: FlexTool writes ``set <name> <value>`` itself; judge the name the user meant.
        words = words[1:]
        key = " ".join(words)
        if key in listed or any(word in key for word in _PATH_NAME_WORDS):
            return True
    if words[0] in _CPLEX_PATH_FIRST_WORDS:
        return True
    return any(
        len(words) == len(full.split(" ")) and all(f.startswith(w) for w, f in zip(words, full.split(" ")))
        for full in listed
    )


def cplex_set_word_problem(solver: str, name: object) -> str | None:
    """Why a CPLEX option *name* that starts with the word ``set`` (``set lpmethod``, ``set lo``) cannot be used,
    else ``None``.  FlexTool writes each option as ``set <name> <value>``, so the line would reach CPLEX as
    ``set set ...``, which CPLEX refuses; no CPLEX parameter name starts with ``set``."""
    if solver != "cplex":
        return None
    words = " ".join(str(name).replace(".", " ").split()).casefold().split(" ")
    if words[0] != _CPLEX_SET_WORD:
        return None
    return ("starts with 'set', which FlexTool writes itself (the line would reach CPLEX as 'set set ...'); "
            "write the parameter name without it")


def is_server_option(solver: str, name: object) -> bool:
    """Whether *name* is an option of *solver* that connects to another machine or carries its credentials."""
    key = _option_key(solver, str(name))
    return key in _SERVER_OPTIONS.get(solver, frozenset()) or key.startswith(_SERVER_OPTION_PREFIXES.get(solver, ()))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True




def is_plain_file_name(value: str) -> bool:
    """Whether *value* is a plain file name: ASCII letters, digits and ``_ . + -``, starting with a letter, digit
    or ``_``, without ``..`` (no folder part, so the file lands in the solve's work folder)."""
    return value.isascii() and _PLAIN_FILE_NAME.fullmatch(value) is not None and ".." not in value


def screen_solver_option_file(solver: str, text: str) -> list[str]:
    """The lines of *solver*'s option file *text* that a file from someone else must not set unseen.

    The rules of the input database's options (:func:`is_server_option`, :func:`is_path_option` with
    :func:`is_plain_file_name`): an option that connects the solver to another computer, or one that names a file
    or folder by more than a plain file name; and a CPLEX name that starts with the word ``set``
    (:func:`cplex_set_word_problem`), which a run refuses.  One message per such line, ``line N: <name> <value>
    ...`` (a server option's value shown as ``<hidden>``); an empty list when the file is clean.  A project import or the first open of an imported project keeps a file
    with a message as ``<solver>.opt.review`` (never read by a run) until the user has checked it.
    """
    lines = highs_opt_lines(text) if solver == "highs" else native_opt_lines(text)
    problems: list[str] = []
    for lineno, name, value in lines:
        shown = f"{name} = {value}" if solver == "highs" else f"{name} {value}"
        if is_server_option(solver, name):
            # The value can be a licence key or a password: the message names the option, never its value.
            shown = f"{name} = <hidden>" if solver == "highs" else f"{name} <hidden>"
            problems.append(f"line {lineno}: {shown} connects the solver to another computer (a server, cloud or "
                            "licence service) or carries its credentials")
        elif value and is_path_option(solver, name) and not _is_number(value) and not is_plain_file_name(value):
            problems.append(f"line {lineno}: {shown} names a file or folder outside the solve's work folder")
        elif (reason := cplex_set_word_problem(solver, name)) is not None:
            problems.append(f"line {lineno}: {shown} {reason}")
    return problems


__all__ = [
    "commercial_option_key",
    "cplex_set_word_problem",
    "highs_opt_lines",
    "is_path_option",
    "is_plain_file_name",
    "is_server_option",
    "native_opt_lines",
    "screen_solver_option_file",
]
