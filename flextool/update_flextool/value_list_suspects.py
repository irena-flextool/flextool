"""The value-list repair tool's suspects (W-0490): what may read the wrong member, the evidence, the suggestion.

Built from an :class:`~flextool.update_flextool.value_list_repair.Analysis` of a snapshot, with ``sqlite3`` only
(read-only).  Each suspect carries its own list and the member its reference reads now, so it is checked against
that list (a carried group value is in ``group_invest_methods``, not R's list).

- **Values**: ``list_value_ref`` references to R on definitions bound to R's list.  In ``renamed-back`` only those
  written in (C, H), while R read the appended member.
- **Method parameter defaults**: every ``invest_method`` definition bound to R's list whose default references R
  (in ``hit`` and ``hit-unanchored``; R reads ``cumulative_limits`` there), also a text default the item-279 pre-pass
  turned into a reference to R.  A default set to another member is the user's and never asked; an empty one is
  part B's (``list_default_restore``).
- **Carried group values** (data version 77 and later): group values that read ``cumulative_limits`` and were
  written by the v77 step's commit, which rewrote every group value by name.

Evidence and suggestion (``label`` / ``suggestion``):

- a value written before C (or with a NULL commit) in state ``hit`` with the position proof: ``certain``,
  ``set:<lost member>``;
- in ``hit-unanchored``, or without the proof: ``probably <lost>``; ``lost member not known (hand-restored X)``
  when the lost member is only inferred from a hand-restored row, ``lost member not known (one of X, Y)`` with two
  or more shipped members missing; ``legacy member`` when no shipped member is missing or hand-restored.  No
  suggestion;
- a value written after the overwrite: no suggestion (the user decides);
- a method parameter default: FlexTool's default (``set:not_allowed``), recommended with :data:`DEFAULT_REASON`;
  ``keep`` (:data:`HAND_DEFAULT`) when it was changed by hand after the last upgrade
  (``list_default_restore.changed_by_hand``);
- a carried group value or a ``renamed-back`` value in (C, H): no suggestion.

:func:`groups` puts suspects with the same kind, class, label, suggestion, archive member and context signature
(``invest_cost``, cumulative cap, inert alternative) in one similarity group.  The defaults and the storage
nodes are asked first, in one block (:func:`defaults_block`, :func:`results_change`,
``list_default_restore.block_lines``).

TODO(v5.0.0): remove with W-0490 (see ``value_list_history``).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr

VALUE, DEFAULT, CARRIED = "value", "default", "carried"
CERTAIN = "certain"
LEGACY = "legacy member"
UNKNOWN = "lost member not known"  # + " (hand-restored X)" or " (one of X, Y)": :func:`lost_unknown`
AFTER = "written after the overwrite"
SHIPPED = "reads cumulative_limits since the upgrade"
HAND_DEFAULT = "changed by hand on {date}"
CARRIED_LABEL = "carried by v77, inert either way"
RENAMED = "written while R read cumulative_limits"
CERTAIN_ARCHIVE = "certain (archive)"  # the same row in a same-lineage archive (``value_list_archive``)
#: The labels ``--accept-certain`` and a group-apply may pre-answer.
CERTAIN_LABELS: tuple[str, ...] = (CERTAIN, CERTAIN_ARCHIVE)

DEFAULT_REASON = ("Recommended: FlexTool's default. A method parameter default that reads cumulative_limits can "
                  "lead to unwanted behaviour (here: entities that rely on it invest and retire up to 1e6 MW).")
HAND_REASON = ("Suggested: keep, because it was changed by hand after the last upgrade; set:not_allowed restores "
               "FlexTool's default.")
#: FlexTool's default of ``invest_method`` in R's list (``invest_methods``).
FLEXTOOL_DEFAULT = "not_allowed"
#: How a prompt line names a method parameter default (no entity).
DEFAULT_WHERE = " (method parameter default)"
_INVEST_CLASSES = ("unit", "connection", "node", "group")
_CAPS = ("cumulative_max_capacity", "cumulative_min_capacity")
_LIMITS = ("invest_max_period", "invest_max_total", "retire_max_period", "retire_max_total")
_UNCAPPED = "invests and retires up to 1e6 MW"
_COMMENT_CHARS = 60


@dataclass(frozen=True)
class Context:
    """What the user needs to judge a suspect.  For a value: the entity's parameters (any alternative), the
    scenarios that use its alternative; for a default: the entities that rely on it (:func:`relying_entities`), how
    many of them have ``invest_cost`` / a cumulative cap, and their names with the cap (``relying_names``)."""

    invest_cost: bool = False
    cap: bool = False
    limits: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    date: str = ""
    comment: str = ""
    relying: int | None = None
    relying_cost: int = 0
    relying_cap: int = 0
    uncapped: bool = False  # the reference reads cumulative_limits
    invest: bool = True  # False for a part B suspect: no invest_cost / cap fields in the text
    relying_names: tuple[tuple[str, bool], ...] = ()  # (entity name, has a cumulative cap), by name

    @property
    def inert(self) -> bool:
        """A value in an alternative no scenario uses: it cannot change a result."""
        return self.relying is None and not self.scenarios

    def text(self) -> str:
        if self.relying is not None and not self.invest:
            out = f"{rely(self.relying)} on it"
        elif self.relying is not None:
            out = (f"{rely(self.relying)} on it: {self.relying_cost} with invest_cost, {self.relying_cap} "
                   "with a cumulative cap")
            if self.uncapped and self.relying > self.relying_cap:
                out += f" (without one: {_UNCAPPED})"
        elif not self.invest:
            out = f"scenarios: {', '.join(self.scenarios)}" if self.scenarios else "scenarios: none (inert)"
        else:
            cap = "cumulative cap yes" if self.cap else (
                f"no cumulative cap: {_UNCAPPED}" if self.uncapped else "cumulative cap no")
            out = f"invest_cost {'yes' if self.invest_cost else 'no'}, {cap}"
            if self.limits:
                out += f"; {', '.join(self.limits)}"
            out += f"; scenarios: {', '.join(self.scenarios)}" if self.scenarios else "; scenarios: none (inert)"
        if self.date:
            out += f"; written {self.date}" + (f" ({self.comment})" if self.comment else "")
        return out


@dataclass(frozen=True)
class Suspect:
    """One item the user answers.  *item* is ``pv:<parameter_value id>``, ``def:<definition id>`` or (part B)
    ``sn:<node entity id>:<alternative id>``; *current* the
    member its reference reads now in its own list *list_id* (``""``: no member); *suggestion* the pre-selected
    answer (``set:<member>``; ``keep`` for a part B default kept in an earlier repair or a part A default changed
    by hand) or ``None``; *reason* why (a method parameter default); *keep_text* what ``keep`` means."""

    item: str
    kind: str
    cls: str
    parameter: str
    entity: str
    alternative: str
    list_id: int
    current: str
    label: str
    evidence: str
    suggestion: str | None
    context: Context
    commit: int | None = None
    reason: str = ""
    keep_text: str = ""
    archive: str | None = None  # the archive's member, translated (archive mode)
    archive_note: str = ""  # what the archive shows, for the prompt and the CSV ``archive`` column
    part_b: bool = False  # an empty method parameter default or a storage node (``list_default_restore``)

    @property
    def certain(self) -> bool:
        return self.label in CERTAIN_LABELS


@dataclass(frozen=True)
class Group:
    key: tuple
    suspects: tuple[Suspect, ...]


def rely(n: int) -> str:
    """``"1 entity relies"`` / ``"N entities rely"``."""
    return f"{n} entity relies" if n == 1 else f"{n} entities rely"


def defaults_block(suspects) -> tuple[Suspect, ...]:
    """The suspects of the one block question: every method parameter default and every storage node."""
    return tuple(s for s in suspects if s.kind in (DEFAULT, ldr.STORAGE_KIND))


def results_change(block) -> bool:
    """Whether Enter on the block changes results: a storage node is in it, or a part A default that Enter sets has
    an entity relying on it.  Part B restores never change results (the engine falls back to FlexTool's default)."""
    return any(s.kind == ldr.STORAGE_KIND for s in block) or any(
        s.kind == DEFAULT and not s.part_b and s.suggestion != "keep" and (s.context.relying or 0) > 0 for s in block)


def relying_entities(con, class_id: int, def_id: int) -> list[tuple[int, str, bool]]:
    """``(id, name, capped)`` of the entities of class *class_id* that rely on the default of definition *def_id*,
    by name: with scenarios, an entity relies when at least one scenario has no value for it in any of its
    alternatives; without scenarios, when it has no value in any alternative.  A scenario with no alternatives is
    ignored (it runs nothing).  Entity activity is ignored (it can only over-count).  *capped*: the entity has a
    cumulative cap value in some alternative."""
    scenarios: dict[int, set[int]] = {}
    for scenario, alt in con.execute("SELECT scenario_id, alternative_id FROM scenario_alternative"):
        scenarios.setdefault(scenario, set()).add(alt)
    own: dict[int, set[int]] = {}
    for entity, alt in con.execute("SELECT entity_id, alternative_id FROM parameter_value WHERE "
                                   "parameter_definition_id = ?", (def_id,)):
        own.setdefault(entity, set()).add(alt)
    capped = {e for (e,) in con.execute(
        "SELECT pv.entity_id FROM parameter_value pv JOIN parameter_definition pd ON pd.id = "
        f"pv.parameter_definition_id WHERE pd.entity_class_id = ? AND pd.name IN ({','.join('?' * len(_CAPS))})",
        (class_id, *_CAPS))}
    out = []
    for entity, name in con.execute("SELECT id, name FROM entity WHERE class_id = ? ORDER BY name", (class_id,)):
        alts = own.get(entity, set())
        if any(not alts & s for s in scenarios.values()) if scenarios else not alts:
            out.append((entity, name, entity in capped))
    return out


def similarity_key(s: Suspect) -> tuple:
    c = s.context
    return (s.kind, s.cls, s.label, s.suggestion, s.archive, c.invest_cost, c.cap, c.inert)


def groups(suspects) -> tuple[Group, ...]:
    """Similarity groups, in the order of their first suspect."""
    found: dict[tuple, list[Suspect]] = {}
    for s in suspects:
        found.setdefault(similarity_key(s), []).append(s)
    return tuple(Group(k, tuple(v)) for k, v in found.items())


def _ref(payload) -> int | None:
    """The member id a ``list_value_ref`` payload holds (ASCII digits only, as spinedb_api reads it)."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return int(raw) if isinstance(raw, bytes) and raw.isdigit() else None


def lost_unknown(a: vr.Analysis) -> str | None:
    """What the label says when the lost member is not known: ``"hand-restored X"`` (one shipped member added back
    by hand, without the position proof: it may not be the lost one) or ``"one of X, Y"`` (two or more shipped
    members missing); ``None`` when the lost member is known or no shipped member is missing or hand-restored
    (then R was a legacy member)."""
    if a.lost_row is not None and not a.position_proof:
        return f"hand-restored {a.lost}"
    if a.lost is None and len(a.restore) > 1:
        return f"one of {', '.join(a.restore)}"
    return None


def overwrite_commit(a: vr.Analysis) -> int:
    """The commit that separates "before" from "after the overwrite": C in ``renamed-back`` and in
    ``hit-unanchored`` (there R's commit may be a later rewrite of R, after the overwrite), else R's commit (the
    moment R got the appended text; C in ``hit``), else C, else the session's closing version commit."""
    commit = a.c if a.state in (vr.RENAMED_BACK, vr.HIT_UNANCHORED) or a.r.commit is None else a.r.commit
    return commit if commit is not None else a.finding.end_commit


def suspects(a: vr.Analysis) -> tuple[Suspect, ...]:
    """The suspects of the analysed snapshot: none unless the state is ``hit``, ``hit-unanchored`` or
    ``renamed-back``.  Defaults first, then values, then carried group values."""
    if a.state not in (vr.HIT, vr.HIT_UNANCHORED, vr.RENAMED_BACK):
        return ()
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(a.path)) as con:
        return _Reader(con, a).suspects()


class _Reader:
    def __init__(self, con, a: vr.Analysis) -> None:
        self.con, self.a = con, a
        self.commits = ldr._commits(con)
        self.members = {i: (lid, vr._text(v)) for i, lid, v in con.execute(
            "SELECT id, parameter_value_list_id, value FROM list_value")}
        self.lists = dict(con.execute("SELECT id, name FROM parameter_value_list"))
        self.params: dict[int, set[str]] = {}
        names = ("invest_cost", "invest_method", *_CAPS, *_LIMITS)
        for entity, name in con.execute(
                "SELECT pv.entity_id, pd.name FROM parameter_value pv JOIN parameter_definition pd ON pd.id = "
                f"pv.parameter_definition_id WHERE pd.name IN ({','.join('?' * len(names))})", names):
            self.params.setdefault(entity, set()).add(name)
        self.scenarios: dict[int, list[str]] = {}
        for alt, name in con.execute("SELECT sa.alternative_id, s.name FROM scenario_alternative sa JOIN scenario s "
                                     "ON s.id = sa.scenario_id ORDER BY s.name"):
            self.scenarios.setdefault(alt, []).append(name)
        self.overwrite = overwrite_commit(a)
        self.overwrite_date = self.commits.get(self.overwrite, ("?", ""))[0]

    def _member(self, payload, list_id: int) -> str:
        lid, text = self.members.get(_ref(payload), (None, None))
        return (text or "") if lid == list_id else ""

    def _when(self, commit) -> tuple[str, str]:
        date, comment = self.commits.get(commit, ("", ""))
        comment = comment or ""
        return date, comment if len(comment) <= _COMMENT_CHARS else comment[:_COMMENT_CHARS - 1] + "…"

    def _context(self, entity: int, alternative: int, commit, current: str) -> Context:
        names = self.params.get(entity, set())
        date, comment = self._when(commit)
        return Context(invest_cost="invest_cost" in names, cap=bool(names & set(_CAPS)),
                       limits=tuple(n for n in _LIMITS if n in names),
                       scenarios=tuple(self.scenarios.get(alternative, ())), date=date, comment=comment,
                       uncapped=current == vlh.APPENDED[22])

    def _definitions(self):
        """``(id, class, class id, list id, default payload, commit)`` of the ``invest_method``
        definitions of the investment classes."""
        return self.con.execute(
            "SELECT pd.id, ec.name, ec.id, pd.parameter_value_list_id, pd.default_value, pd.commit_id "
            "FROM parameter_definition pd JOIN entity_class ec ON ec.id = pd.entity_class_id "
            f"WHERE pd.name = 'invest_method' AND ec.name IN ({','.join('?' * len(_INVEST_CLASSES))}) ORDER BY pd.id",
            _INVEST_CLASSES).fetchall()

    def suspects(self) -> tuple[Suspect, ...]:
        defs = self._definitions()
        out = [s for d in defs if (s := self._default(*d)) is not None]
        own = [d[0] for d in defs if d[3] == self.a.list_id]
        out += [s for row in self._values(own) if (s := self._value(*row)) is not None]
        group_list = [d for d in defs if self.lists.get(d[3]) == "group_invest_methods"]
        if self.a.version >= 77 and group_list:
            out += self._carried(group_list[0])
        return tuple(out)

    def _default(self, def_id, cls, class_id, list_id, payload, commit) -> Suspect | None:
        """A suspect only for a default that references R while R reads ``cumulative_limits`` (design §1a): what
        the bug may have changed.  Another member is the user's choice; an empty default is part B's."""
        a = self.a
        if a.state not in (vr.HIT, vr.HIT_UNANCHORED) or list_id != a.list_id or payload is None or \
                _ref(payload) != a.r.id:
            return None
        current = self._member(payload, list_id)
        relying = relying_entities(self.con, class_id, def_id)
        date, comment = self._when(commit)
        ctx = Context(relying=len(relying), date=date, comment=comment, uncapped=current == vlh.APPENDED[22],
                      relying_cost=sum("invest_cost" in self.params.get(e, ()) for e, _, _ in relying),
                      relying_cap=sum(c for _, _, c in relying), relying_names=tuple((n, c) for _, n, c in relying))
        evidence = f"the method parameter default reads {current}; FlexTool's default is {FLEXTOOL_DEFAULT}"
        if ldr.changed_by_hand(commit, self.commits):
            label = HAND_DEFAULT.format(date=date)
            return Suspect(f"def:{def_id}", DEFAULT, cls, "invest_method", "", "", list_id, current, label,
                           f"{evidence}; {label}" + (f" ({comment})" if comment else ""), "keep", ctx, commit,
                           reason=HAND_REASON, keep_text=self._keep(current))
        return Suspect(f"def:{def_id}", DEFAULT, cls, "invest_method", "", "", list_id, current, SHIPPED, evidence,
                       f"set:{FLEXTOOL_DEFAULT}", ctx, commit, reason=DEFAULT_REASON, keep_text=self._keep(current))

    def _values(self, definition_ids):
        if not definition_ids:
            return []
        return self.con.execute(
            "SELECT pv.id, ec.name, pd.name, e.name, al.name, pv.entity_id, pv.alternative_id, pv.value, "
            "pv.commit_id FROM parameter_value pv JOIN parameter_definition pd ON pd.id = pv.parameter_definition_id "
            "JOIN entity_class ec ON ec.id = pd.entity_class_id JOIN entity e ON e.id = pv.entity_id "
            "JOIN alternative al ON al.id = pv.alternative_id "
            f"WHERE pv.parameter_definition_id IN ({','.join('?' * len(definition_ids))}) "
            "AND pv.type = 'list_value_ref' ORDER BY ec.name, e.name, al.name, pv.id", definition_ids).fetchall()

    def _value(self, pv_id, cls, param, entity, alt, entity_id, alt_id, payload, commit) -> Suspect | None:
        a, r = self.a, self.a.r
        if _ref(payload) != r.id:
            return None
        before = vr._older(commit, self.overwrite)
        if a.state == vr.RENAMED_BACK:
            if before or commit is None or commit >= a.h:
                return None
            label, suggestion = RENAMED, None
            evidence = f"written {self._when(commit)[0]}, while R read cumulative_limits (before the rename back)"
        elif not before:
            label, suggestion = AFTER, None
            evidence = f"written {self._when(commit)[0]}, after the overwrite on {self.overwrite_date}"
        elif a.state == vr.HIT and a.position_proof:
            label, suggestion = CERTAIN, f"set:{a.lost}"
            evidence = f"written before the overwrite on {self.overwrite_date}: it was {a.lost} then"
        elif (unknown := lost_unknown(a)) is not None:
            label, suggestion = f"{UNKNOWN} ({unknown})", None
            evidence = (f"written before the overwrite on {self.overwrite_date}: then the lost member, which is "
                        + (f"{unknown}" if a.lost_row is None else f"not proven to be the {unknown} member"))
        elif a.lost is not None:
            label, suggestion = f"probably {a.lost}", None
            evidence = f"written before the overwrite on {self.overwrite_date}: probably {a.lost} then (not proven)"
        else:
            label, suggestion = LEGACY, None
            evidence = (f"written before the overwrite on {self.overwrite_date}: then a legacy member FlexTool no "
                        "longer has")
        current = r.value or ""
        return Suspect(f"pv:{pv_id}", VALUE, cls, param, entity, alt, a.list_id, current, label, evidence,
                       suggestion, self._context(entity_id, alt_id, commit, current), commit,
                       keep_text=self._keep(current))

    def _carried(self, definition) -> list[Suspect]:
        def_id, list_id = definition[0], definition[3]
        v77 = {i for i, (_, comment) in self.commits.items()
               if (comment or "").startswith("v77:") and i > self.overwrite}
        if self.a.state == vr.RENAMED_BACK:  # carried only if v77 ran while R read cumulative_limits
            v77 = {i for i in v77 if i < self.a.h}
        out = []
        for pv_id, cls, param, entity, alt, entity_id, alt_id, payload, commit in self._values([def_id]):
            current = self._member(payload, list_id)
            if commit in v77 and current == vlh.APPENDED[22]:
                out.append(Suspect(
                    f"pv:{pv_id}", CARRIED, cls, param, entity, alt, list_id, current, CARRIED_LABEL,
                    "rewritten by name by the v77 upgrade; what it meant before is gone", None,
                    self._context(entity_id, alt_id, commit, current), commit, keep_text=self._keep(current)))
        return out

    def _keep(self, current: str) -> str:
        if self.a.state == vr.RENAMED_BACK and current:
            return f"keep (stays {current}, the current text)"
        return f"keep {current}" if current else "keep (no default)"
