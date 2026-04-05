"""HarmBench benchmark loader (Mazeika et al. 2024, arXiv 2402.04249).

HarmBench is a standardised evaluation suite of 400 harmful behaviours
covering 7 risk categories. We use its "standard" behaviours split
(200 behaviours) as the primary harmful-prompt source for experiments
2-4.

Official HarmBench artefacts live at
https://github.com/centerforaisafety/HarmBench. We support two loader
paths:

1. ``from_hf`` — fetch the public HF mirror. No auth needed.
2. ``from_csv`` — read a local CSV the user has downloaded.

A HarmBench entry has:
    BehaviorID, Behavior, SemanticCategory, FunctionalCategory,
    ContextString, Target, Tags.

For ARA we only need ``Behavior`` (the harmful prompt) and, optionally,
``Target`` (a compliance prefix the attacker tries to elicit — used by
GCG). We expose a lightweight dataclass to keep downstream code clean.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)


@dataclass
class HarmfulBehavior:
    behavior_id: str
    behavior: str
    target: str = ""
    semantic_category: str = ""
    functional_category: str = ""
    context: str = ""
    tags: list[str] = field(default_factory=list)


class HarmBenchLoader:
    """Load and filter HarmBench behaviours."""

    _DEFAULT_SPLITS = ("standard", "contextual", "copyright")

    def __init__(self, behaviors: Sequence[HarmfulBehavior]):
        self.behaviors = list(behaviors)

    def __len__(self) -> int:
        return len(self.behaviors)

    def __iter__(self):
        return iter(self.behaviors)

    def filter(
        self,
        *,
        semantic_category: Optional[str] = None,
        functional_category: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> "HarmBenchLoader":
        out = self.behaviors
        if semantic_category is not None:
            out = [b for b in out if b.semantic_category == semantic_category]
        if functional_category is not None:
            out = [b for b in out if b.functional_category == functional_category]
        if tag is not None:
            out = [b for b in out if tag in b.tags]
        return HarmBenchLoader(out)

    def sample(self, n: int, seed: int = 0) -> "HarmBenchLoader":
        import random
        rng = random.Random(seed)
        pool = list(self.behaviors)
        rng.shuffle(pool)
        return HarmBenchLoader(pool[: min(n, len(pool))])

    @classmethod
    def from_csv(cls, path: str | Path) -> "HarmBenchLoader":
        p = Path(path)
        out: list[HarmfulBehavior] = []
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(HarmfulBehavior(
                    behavior_id=row.get("BehaviorID", row.get("behavior_id", "")),
                    behavior=row.get("Behavior", row.get("behavior", "")),
                    target=row.get("Target", row.get("target", "")),
                    semantic_category=row.get("SemanticCategory", ""),
                    functional_category=row.get("FunctionalCategory", ""),
                    context=row.get("ContextString", ""),
                    tags=_split_tags(row.get("Tags", "")),
                ))
        log.info("loaded %d HarmBench behaviours from %s", len(out), p)
        return cls(out)

    @classmethod
    def from_hf(
        cls,
        repo_id: str = "walledai/HarmBench",
        split: str = "standard",
        hf_token: Optional[str] = None,
    ) -> "HarmBenchLoader":
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "`datasets` package required for from_hf; install requirements.txt"
            ) from e
        ds = load_dataset(repo_id, split=split, token=hf_token)
        out = [
            HarmfulBehavior(
                behavior_id=str(row.get("BehaviorID", row.get("id", i))),
                behavior=row["Behavior"] if "Behavior" in row else row["prompt"],
                target=row.get("Target", ""),
                semantic_category=row.get("SemanticCategory", row.get("category", "")),
                functional_category=row.get("FunctionalCategory", ""),
                context=row.get("ContextString", ""),
                tags=_split_tags(row.get("Tags", "")),
            )
            for i, row in enumerate(ds)
        ]
        log.info("loaded %d HarmBench behaviours from HF repo %s", len(out), repo_id)
        return cls(out)


def load_harmbench(
    *,
    path: str | Path | None = None,
    split: str = "standard",
    repo_id: str = "walledai/HarmBench",
    hf_token: Optional[str] = None,
) -> HarmBenchLoader:
    """Convenience entry point.

    If ``path`` is provided, load from CSV. Otherwise fetch ``repo_id``
    (default ``walledai/HarmBench``) from HuggingFace.
    """
    if path is not None:
        return HarmBenchLoader.from_csv(path)
    return HarmBenchLoader.from_hf(repo_id=repo_id, split=split, hf_token=hf_token)


def _split_tags(raw) -> list[str]:
    if isinstance(raw, list):
        return [t for t in raw if t]
    if isinstance(raw, str):
        return [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    return []
