"""The assistant level of a model (P3 §3): from its name, its size and the context it runs with.

A model is placed at a level (internal ``small`` / ``medium`` / ``large``; users see basic / standard / advanced)
by :data:`TIER_RULES`, tried in order on the normalised model id (:func:`normalise_model_id`); the first rule
that matches wins:

1. a small-model word (``mini``, ``nano``, ``haiku``, …) → basic;
2. the size: the parameter count the local server reports, else a size tag in the id (``14b``, ``8x7b``) — up to
   34 billion basic, up to 200 billion standard, more advanced;
3. a mid-tier word (``fast``, ``maverick``, ``nova``, …) → standard;
4. a large-model family (``opus``, ``claude``, ``large``, ``o3``, …) → advanced;
5. ``gpt`` with a major version of 5 or more → advanced;
6. another known family (``flash``, ``gpt``, ``gemini``, ``mistral``, …) → standard;
7. not recognised: basic on this computer or the local network, standard for a remote service (a guess).

The context window then sets the highest level (:data:`WINDOW_LEVELS`); below :data:`MIN_WINDOW` tokens no
request is sent.  The **automatic level** is the lower of the two.  A model may be used at its automatic level or
lower (P3-O7); one FlexTool does not recognise also higher, up to the window's level, at the user's risk
(:data:`ALLOW_HIGHER_WHEN_UNRECOGNISED`, P3-O12).  A level chosen above the allowed ones is lowered to the
automatic level with a notice.

The rules are data, reviewed by the owner when a release is prepared and when a user reports a model at the wrong
level (P3-O8); nothing tracks model names automatically.  Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from flextool.assistant.llm_adapter import Route
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.providers import Preset

#: The internal level names and the names users see, lowest first.
LEVEL_NAMES: dict[str, str] = {"small": "basic", "medium": "standard", "large": "advanced"}
LEVEL_ORDER: tuple[str, ...] = ("small", "medium", "large")
#: A model whose name FlexTool does not recognise may be used above its guessed level, at the user's risk
#: (P3-O12, final).
ALLOW_HIGHER_WHEN_UNRECOGNISED = True

#: Below this many tokens of context no request is sent (P3 §3.3).
MIN_WINDOW = 12_288
#: (lowest window, highest level): a window from that many tokens allows that level at most.
WINDOW_LEVELS: tuple[tuple[int, str], ...] = ((65_536, "large"), (32_768, "medium"), (MIN_WINDOW, "small"))
#: The window assumed when nothing reports one: a remote service, and a server on this computer or the network.
ASSUMED_REMOTE_WINDOW = 128_000
ASSUMED_LOCAL_WINDOW = 16_384

#: Size bands, in billions of parameters (total, not active).
SMALL_SIZE_MAX = 34.0
MEDIUM_SIZE_MAX = 200.0

#: Ollama tags that say nothing about the model, dropped before matching.
_DROPPED_TAGS = frozenset({"free", "latest", "cloud", "beta", "nitro", "floor", "online", "extended", "thinking"})
_DATE_STAMPS = re.compile(r"-\d{4}-\d{2}-\d{2}(?![0-9])|-\d{8}(?![0-9])|@\d{8}(?![0-9])")
_QUANTISATION = re.compile(r"[-_:](?:q\d(?:_\d)?(?:_k)?(?:_[msl0-9])?|fp16|bf16|int[48]|mxfp4|awq|gptq|gguf|mlx)"
                           r"(?=$|[-_:])")
_TOKEN_SPLIT = re.compile(r"[-_:/ ]+")
_SIZE_TOKEN = re.compile(r"(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?b")
_LETTERS_SIZE_TOKEN = re.compile(r"([a-z]+)(\d+(?:\.\d+)?)b")
_EXPERT_COUNT = re.compile(r"\d+e")
_JOINED_SMALL = re.compile(r"(small|mini|lite)\d+(?:\.\d+)*")

Tier = Literal["small", "medium", "large"]
Source = Literal["name", "size", "reported_size", "fallback"]
WindowSource = Literal["loaded", "model", "model_list", "assumed"]


@dataclass(frozen=True)
class TierRule:
    """One row of the level table: *kind* says how *match* is read; *note* is the release-review note."""

    kind: Literal["token", "size", "family", "gpt_major", "fallback"]
    match: str
    tier: str
    note: str


#: The level table (P3 §3.2), first match wins.  ``match`` lists whole tokens separated by spaces; ``a+b`` needs
#: both tokens, ``a+(b|c)`` *a* with one of the others; ``^…$`` is a token pattern; ``v3*`` a token prefix.
TIER_RULES: tuple[TierRule, ...] = (
    TierRule("token", "mini nano micro tiny lite small haiku ministral", "small",
             "small-model words, also small/mini/lite joined to a version (small3.2)"),
    TierRule("size", "34 200", "by size",
             "the reported size, else the id's size tag: up to 34 B basic, up to 200 B standard, more advanced"),
    TierRule("family", "fast maverick scout nova sonar deepseek+coder", "medium",
             "fast or mid-tier variants of large families"),
    TierRule("family", "opus fable sonnet claude ultra large max grok kimi ^o\\d+$ pro+(gemini|gpt) "
             "deepseek+(r1|v3*|chat|reasoner)", "large", "the providers' top families"),
    TierRule("gpt_major", "5", "large", "gpt with a major version of 5 or more"),
    TierRule("family", "flash medium gpt gemini codestral devstral magistral mistral command glm", "medium",
             "other known families"),
    TierRule("fallback", "", "small or medium", "not recognised: small on this computer or the network, else medium"),
)

# --- texts ----------------------------------------------------------------------------------------------------
REASON_SIZE = "a model of about {n} billion parameters"
REASON_SMALL_FAMILY = "a small model of its family ({token})"
REASON_MEDIUM_FAMILY = "a mid-range model family ({token})"
REASON_LARGE_FAMILY = "a large model family ({token})"
REASON_FALLBACK = "not recognised; guessed from where it runs"
PLACED_BY_NAME = "This model is placed at the {level} level from its name ({reason})."
PLACED_BY_SIZE = "This model is placed at the {level} level from its size ({reason})."
NO_HIGHER = "Higher levels are not available for it."
WINDOW_CEILING = ("This model runs with a {window}-token context, so the highest level is {level}. A larger context "
                  "allows a higher level.")
WINDOW_CEILING_LOCAL = "Set OLLAMA_CONTEXT_LENGTH or the model's context length in its load settings."
UNRECOGNISED = ("FlexTool does not recognise this model, so its level is a guess. A higher level may give "
                "unreliable answers.")
NAME_ONLY = ("The level comes from the model name; it may be lowered when FlexTool sees the model's context "
             "size.")
LOWERED = "The level chosen before ({level}) is above this model's level, so the automatic level is used."


@dataclass(frozen=True)
class TierChoice:
    """The level of one model: the automatic one and why, the window's ceiling, the levels the user may choose
    and the one in use (internal names)."""

    automatic: str
    source: Source
    reason: str
    window_ceiling: str | None          # None: no window known yet (the level comes from the name only)
    window_known: bool                  # the window came from the model or its server, not assumed
    notice: str | None                  # a chosen level was lowered
    allowed: tuple[str, ...]
    chosen: str
    chosen_note: str | None             # the chosen level is above the automatic one (unrecognised model)
    name_level: str = "medium"          # the level the name (or size) alone gives
    window: int | None = None
    window_source: WindowSource | None = None
    local: bool = False                 # the model runs on this computer or the local network

    @property
    def level_name(self) -> str:
        """The level in use, as users see it."""
        return LEVEL_NAMES[self.chosen]

    @property
    def automatic_name(self) -> str:
        return LEVEL_NAMES[self.automatic]

    def placement(self) -> str:
        """Why the model is at its automatic level, in one or two sentences (P3 §3.3)."""
        if self.source == "fallback":
            return UNRECOGNISED
        level = LEVEL_NAMES[self.name_level]
        template = PLACED_BY_NAME if self.source == "name" else PLACED_BY_SIZE
        text = template.format(level=level, reason=self.reason)
        if self.name_level == self.automatic and self.automatic != "large":
            text += " " + NO_HIGHER
        return text

    def window_line(self) -> str | None:
        """The sentence that says the context lowered the level, ``None`` when it did not."""
        if self.window_ceiling is None or self.window is None or \
                _rank(self.window_ceiling) >= _rank(self.name_level):
            return None
        text = WINDOW_CEILING.format(window=self.window, level=LEVEL_NAMES[self.window_ceiling])
        return f"{text} {WINDOW_CEILING_LOCAL}" if self.local else text

    def lines(self) -> list[str]:
        """Every sentence the settings and the terminal show about the level."""
        out = [self.placement()]
        window = self.window_line()
        if window is not None:
            out.append(window)
        if self.window_ceiling is None:
            out.append(NAME_ONLY)
        if self.notice:
            out.append(self.notice)
        if self.chosen_note and self.chosen_note not in out:
            out.append(self.chosen_note)
        return out


def _rank(tier: str) -> int:
    return LEVEL_ORDER.index(tier)


def internal_level(level: str) -> str:
    """The internal name of a level given as either name (``basic`` or ``small`` → ``small``)."""
    for internal, user in LEVEL_NAMES.items():
        if level in (internal, user):
            return internal
    raise ValueError(f"unknown level {level!r}")


# --- the model id -------------------------------------------------------------------------------------------
def is_free_variant(model: str) -> bool:
    """Whether *model* is an OpenRouter free variant (its id ends ``:free``), read before normalising."""
    return model.strip().lower().endswith(":free")


def normalise_model_id(model: str) -> tuple[str, tuple[str, ...]]:
    """The model id as the rules read it, and its tokens (P3 §3.1): lower case; no vendor path; an Ollama tag that
    says nothing dropped; no ``-cloud``, date stamps or quantisation suffixes; tokens split on ``-``, ``_``,
    ``:``, ``/`` and space (not on ``.``)."""
    text = model.strip().lower()
    text = text.rsplit("/", 1)[-1]
    if ":" in text:
        name, tag = text.split(":", 1)
        text = name if tag in _DROPPED_TAGS else f"{name}:{tag}"
    text = _DATE_STAMPS.sub("", text)
    if text.endswith("-cloud"):
        text = text[: -len("-cloud")]
    while True:
        shorter = _QUANTISATION.sub("", text)
        if shorter == text:
            break
        text = shorter
    tokens = [t for t in _TOKEN_SPLIT.split(text) if t]
    return text, tuple(tokens)


def _token_sizes(tokens: tuple[str, ...]) -> list[float]:
    sizes: list[float] = []
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        match = _SIZE_TOKEN.fullmatch(token)
        if match:
            if _EXPERT_COUNT.fullmatch(following):
                continue                # 17b-128e: the active parameters of a mixture of experts
            size = float(match.group(1))
            if match.group(2):
                size *= float(match.group(2))
            sizes.append(size)
            continue
        match = _LETTERS_SIZE_TOKEN.fullmatch(token)
        if match and match.group(1) not in ("a", "e") and not _EXPERT_COUNT.fullmatch(following):
            sizes.append(float(match.group(2)))       # command-r7b; never a3b (active) or e4b (effective)
    return sizes


def size_tag(normalised: str) -> float | None:
    """The size in the id in billions of parameters, the largest when there are several; ``None`` when there is
    none or only an active or effective count."""
    _, tokens = normalise_model_id(normalised)
    sizes = _token_sizes(tokens)
    return max(sizes) if sizes else None


def _size_tier(size: float) -> str:
    if size <= SMALL_SIZE_MAX:
        return "small"
    if size <= MEDIUM_SIZE_MAX:
        return "medium"
    return "large"


def _size_text(size: float) -> str:
    return f"{round(size):d}" if size >= 10 else f"{round(size, 1):g}"


def _family_match(term: str, tokens: tuple[str, ...]) -> str | None:
    """The token text a family term matches in *tokens*, or ``None``."""
    if term.startswith("^"):
        pattern = re.compile(term)
        return next((t for t in tokens if pattern.match(t)), None)
    if "+" in term:
        head, rest = term.split("+", 1)
        if head not in tokens:
            return None
        options = rest.strip("()").split("|")
        for option in options:
            for token in tokens:
                if (option.endswith("*") and token.startswith(option[:-1])) or token == option:
                    pair = (head, token) if tokens.index(head) < tokens.index(token) else (token, head)
                    return "-".join(pair)
        return None
    return term if term in tokens else None


def _gpt_major(tokens: tuple[str, ...], minimum: int) -> str | None:
    for i, token in enumerate(tokens):
        joined = re.fullmatch(r"gpt(\d+)(?:\.\d+)?", token)
        if joined and int(joined.group(1)) >= minimum:
            return token
        if token == "gpt" and i + 1 < len(tokens):
            major = re.match(r"(\d+)", tokens[i + 1])
            if major and int(major.group(1)) >= minimum:
                return f"gpt-{tokens[i + 1]}"
    return None


def name_level(model: str, *, reported_size: float | None = None,
               local: bool = False) -> tuple[str, Source, str]:
    """The level the rules give *model*: ``(tier, source, reason)``.  *reported_size* is the size the local
    server reports, in billions; *local* says where an unrecognised model runs (its guess)."""
    _, tokens = normalise_model_id(model)
    for rule in TIER_RULES:
        if rule.kind == "token":
            words = rule.match.split()
            hit = next((t for t in tokens if t in words), None) or next(
                (t for t in tokens if _JOINED_SMALL.fullmatch(t)), None)
            if hit is not None:
                return "small", "name", REASON_SMALL_FAMILY.format(token=hit)
        elif rule.kind == "size":
            if reported_size is not None:
                return _size_tier(reported_size), "reported_size", REASON_SIZE.format(n=_size_text(reported_size))
            sizes = _token_sizes(tokens)
            if sizes:
                size = max(sizes)
                return _size_tier(size), "size", REASON_SIZE.format(n=_size_text(size))
        elif rule.kind == "family":
            for term in rule.match.split():
                hit = _family_match(term, tokens)
                if hit is not None:
                    reason = REASON_LARGE_FAMILY if rule.tier == "large" else REASON_MEDIUM_FAMILY
                    return rule.tier, "name", reason.format(token=hit)
        elif rule.kind == "gpt_major":
            hit = _gpt_major(tokens, int(rule.match))
            if hit is not None:
                return "large", "name", REASON_LARGE_FAMILY.format(token=hit)
        else:
            return ("small" if local else "medium"), "fallback", REASON_FALLBACK
    raise AssertionError("the level table ends with its fallback rule")


# --- the context window -------------------------------------------------------------------------------------
def window_level(window: int) -> str | None:
    """The highest level a context of *window* tokens allows; ``None`` below :data:`MIN_WINDOW`."""
    for lowest, tier in WINDOW_LEVELS:
        if window >= lowest:
            return tier
    return None


def _route_is_local(route: Route | str | None, preset: Preset | None) -> bool:
    if route is None:
        return bool(preset is not None and preset.local)
    if isinstance(route, str):
        return route in ("loopback", "local_network", "local")
    return route.kind in ("loopback", "local_network") and route.forwarded_to is None


def context_window(preset: Preset | None, facts: ModelFacts | None, *,
                   local: bool) -> tuple[int, WindowSource]:
    """The context window the requests have (P3 §3.3): the loaded context, else the model's, else the one the
    model list gives, else the assumed one."""
    if facts is not None:
        if facts.loaded_context is not None:
            return facts.loaded_context, "loaded"
        if facts.context_window is not None:
            return facts.context_window, "model"
        listed = getattr(facts, "listed_context", None)
        if listed is not None:
            return listed, "model_list"
    return (ASSUMED_LOCAL_WINDOW if local else ASSUMED_REMOTE_WINDOW), "assumed"


def allowed_levels(automatic: str, source: str, window_ceiling: str | None) -> tuple[str, ...]:
    """The levels a model may be used at: the automatic one and lower; for a model FlexTool does not recognise
    (``source == "fallback"``) also the higher ones up to the window's level, while
    :data:`ALLOW_HIGHER_WHEN_UNRECOGNISED` holds."""
    top = automatic
    if source == "fallback" and ALLOW_HIGHER_WHEN_UNRECOGNISED:
        top = window_ceiling or "large"
    return LEVEL_ORDER[: _rank(top) + 1]


class WindowTooSmall(Exception):
    """The model's context is below :data:`MIN_WINDOW` tokens: no request is sent."""

    def __init__(self, window: int, local: bool) -> None:
        super().__init__(window)
        self.window = window
        self.local = local


def resolve_tier(preset: Preset | None, model: str, facts: ModelFacts | None, user_choice: str | None,
                 route: Route | str | None = None) -> TierChoice:
    """The level of *model* (P3 §3).  *facts* are the probe's (or cached) facts; ``None`` before any probe, when
    the level comes from the name only.  *user_choice* is the stored or requested level (either name; ``None``:
    automatic); above the allowed levels it is lowered to the automatic level with a notice.  Raises
    :class:`WindowTooSmall` for a known window below :data:`MIN_WINDOW` tokens."""
    local = _route_is_local(route, preset)
    reported = getattr(facts, "parameter_count", None) if facts is not None else None
    reported_size = reported / 1e9 if isinstance(reported, (int, float)) and reported > 0 else None
    tier, source, reason = name_level(model, reported_size=reported_size, local=local)
    window: int | None = None
    window_source: WindowSource | None = None
    ceiling: str | None = None
    if facts is not None:
        window, window_source = context_window(preset, facts, local=local)
        ceiling = window_level(window)
        if ceiling is None:
            raise WindowTooSmall(window, local)
    automatic = tier if ceiling is None or _rank(ceiling) >= _rank(tier) else ceiling
    allowed = allowed_levels(automatic, source, ceiling)
    notice = None
    chosen = automatic
    if user_choice:
        wanted = internal_level(user_choice)
        if wanted in allowed:
            chosen = wanted
        else:
            notice = LOWERED.format(level=LEVEL_NAMES[wanted])
    chosen_note = UNRECOGNISED if _rank(chosen) > _rank(automatic) else None
    return TierChoice(automatic=automatic, source=source, reason=reason, window_ceiling=ceiling,
                      window_known=window_source not in (None, "assumed"), notice=notice, allowed=allowed,
                      chosen=chosen, chosen_note=chosen_note, name_level=tier, window=window,
                      window_source=window_source, local=local)
