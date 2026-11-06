"""Keyword search for the assistant: a tokenizer that knows FlexTool identifiers, and BM25 ranking.

Identifiers stay whole and are also split at ``_`` and at camelCase, so ``penalty_up`` matches the
parameter itself first and the words "penalty" and "up" after it, and ``maxOnline_linear`` also matches
"online".  Standard library only; the documentation index (``flextool.docs_build.index``) stores the term
counts this module computes, so a search after a cache hit tokenizes only the query.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

K1 = 1.2
B = 0.75
HEADING_WEIGHT = 2          # a term in a heading counts this many times

# Runs of letters and digits joined by "_" (an identifier stays one word).
_WORD_RE = re.compile(r"[^\W_]+(?:_[^\W_]+)*")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

STOP_WORDS = frozenset("""
a an and are as at be been by can do does for from has have how if in into is it its no not of on or so
such than that the then there these they this those to was were what when where which while will with
""".split())


def tokens(text: str) -> list[str]:
    """Lower-case terms of ``text``: every word, and for an identifier also its ``_`` and (ASCII) camelCase
    parts.  Stop words are dropped (an identifier is never one)."""
    out: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        whole = word.lower()
        if whole not in STOP_WORDS:
            out.append(whole)
        parts = word.split("_")
        pieces: list[str] = []
        for part in parts:
            sub = _CAMEL_RE.findall(part) if part.isascii() else []
            if len(parts) > 1:
                pieces.append(part.lower())
            if len(sub) > 1:
                pieces.extend(s.lower() for s in sub)
        for piece in pieces:
            if piece != whole and piece not in STOP_WORDS:
                out.append(piece)
    return out


def term_counts(text: str, heading: str = "") -> dict[str, int]:
    """Term frequencies of a passage; the terms of its heading count :data:`HEADING_WEIGHT` times."""
    counts = Counter(tokens(text))
    for term in tokens(heading):
        counts[term] += HEADING_WEIGHT
    return dict(counts)


class BM25:
    """Okapi BM25 over passages given as term counts (``k1`` = 1.2, ``b`` = 0.75)."""

    def __init__(self, counts: Iterable[Mapping[str, int]], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, passage in enumerate(counts):
            self._lengths.append(sum(passage.values()))
            for term, tf in passage.items():
                self._postings.setdefault(term, []).append((i, tf))
        n = len(self._lengths)
        self._avg = (sum(self._lengths) / n) if n else 0.0
        self._idf = {term: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
                     for term, p in self._postings.items()}

    def __len__(self) -> int:
        return len(self._lengths)

    def scores(self, query: str) -> dict[int, float]:
        """``{passage number: score}`` of the passages that contain a term of ``query``."""
        out: dict[int, float] = {}
        if not self._avg:
            return out
        for term in dict.fromkeys(tokens(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for i, tf in postings:
                norm = tf + self.k1 * (1.0 - self.b + self.b * self._lengths[i] / self._avg)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / norm
        return out

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """The best ``limit`` passages, ``[(passage number, score)]``, best first (ties by passage order)."""
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
