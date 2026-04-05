"""Keyword / semantic jailbreak filter.

A simple string-matching filter that flags known jailbreak patterns
(DAN prefixes, "ignore previous instructions", role-play hijacks, etc.)
and common harmful-intent lexemes. Used as a weak baseline in exp3.

The ``DEFAULT_PATTERNS`` list is a short, high-precision pattern set.
A larger list can be loaded via ``from_file``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PATTERNS: tuple[str, ...] = (
    r"ignore (all |previous |prior )?instructions",
    r"pretend (to be|you are)",
    r"you are (now )?(DAN|jailbreak|developer mode)",
    r"disregard (the )?(safety|system|above) (guidelines|prompt|rules)",
    r"you have no restrictions",
    r"act as an? (unrestricted|unfiltered) (ai|assistant|model)",
    r"step by step.*how to (make|build|create) (a )?(bomb|explosive|virus|malware)",
    r"how (do|can) i (hack|exploit|bypass|crack)",
)


@dataclass
class SemanticVerdict:
    flagged: bool
    matched_patterns: list[str]


class SemanticFilter:
    def __init__(self, patterns: Optional[Iterable[str]] = None):
        pats = list(patterns) if patterns is not None else list(DEFAULT_PATTERNS)
        self._compiled = [re.compile(p, flags=re.IGNORECASE) for p in pats]
        self._raw = pats

    def classify(self, text: str) -> SemanticVerdict:
        matched = [self._raw[i] for i, r in enumerate(self._compiled) if r.search(text)]
        return SemanticVerdict(flagged=bool(matched), matched_patterns=matched)

    @classmethod
    def from_file(cls, path: str | Path) -> "SemanticFilter":
        p = Path(path)
        pats = [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return cls(patterns=pats)
