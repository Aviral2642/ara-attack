"""Project continuous adversarial embeddings onto the discrete vocabulary.

The ARA optimiser works in continuous embedding space (via Gumbel-
softmax). At attack time we must convert the continuous optimum to
actual token ids that can be inserted into the input sequence.

Two projection strategies are provided:

* **Cosine-nearest** — standard nearest-neighbour in cosine similarity,
  masking out special tokens and (optionally) low-frequency tokens.
* **Perplexity-constrained** — reject candidate tokens that would push
  the prompt's perplexity above a cap, preserving the "low-perplexity"
  adversarial regime discussed in §4.3 of the paper.

The projector also reports a **projection gap** — the cosine distance
between the continuous optimum and its discrete image. A large gap
indicates the attack exploits a region of embedding space unreachable
by any real token, and motivates the greedy-refinement stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


@dataclass
class ProjectionResult:
    token_ids: Tensor        # (k,) int64 — discrete projection
    projection_gap: Tensor   # (k,) float — 1 - cos(continuous, nearest)
    rejected: Tensor         # (k,) bool — True if PPL constraint rejected topk
    topk_candidates: Tensor  # (k, topk) int64 — alternative candidates


def _build_token_mask(
    vocab_size: int,
    *,
    exclude_ids: Optional[list[int]] = None,
    tokenizer=None,
    exclude_special: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Return a bool tensor of shape (vocab_size,). True = allowed."""
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if exclude_ids:
        for i in exclude_ids:
            if 0 <= i < vocab_size:
                allowed[i] = False
    if exclude_special and tokenizer is not None:
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            tid = getattr(tokenizer, name, None)
            if tid is not None and 0 <= tid < vocab_size:
                allowed[tid] = False
        # Also exclude any additional_special_tokens
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            if 0 <= tid < vocab_size:
                allowed[tid] = False
    return allowed


def project_cosine(
    continuous: Tensor,
    embedding_matrix: Tensor,
    *,
    allowed_mask: Optional[Tensor] = None,
    topk: int = 256,
) -> ProjectionResult:
    """Nearest-neighbour projection in cosine similarity.

    Parameters
    ----------
    continuous : (k, d)  — continuous embeddings to project
    embedding_matrix : (V, d)  — model's input embedding table
    allowed_mask : (V,) bool — True where token is eligible
    topk : keep this many alternatives per adversarial position

    Returns
    -------
    ProjectionResult
    """
    if continuous.dim() != 2:
        raise ValueError(f"expected (k,d), got {tuple(continuous.shape)}")
    V, d = embedding_matrix.shape
    if continuous.shape[1] != d:
        raise ValueError(
            f"dim mismatch: continuous has d={continuous.shape[1]}, vocab d={d}"
        )

    # Normalise once and cache on the correct device.
    emb_norm = F.normalize(embedding_matrix, dim=-1)
    cont_norm = F.normalize(continuous, dim=-1)
    sim = cont_norm @ emb_norm.T       # (k, V)

    if allowed_mask is not None:
        sim = sim.masked_fill(~allowed_mask.to(sim.device), float("-inf"))

    topk = min(topk, V)
    top_sim, top_idx = sim.topk(topk, dim=-1)         # (k, topk)
    best_sim = top_sim[:, 0]
    best_ids = top_idx[:, 0]
    gap = 1.0 - best_sim

    return ProjectionResult(
        token_ids=best_ids,
        projection_gap=gap,
        rejected=torch.zeros_like(best_ids, dtype=torch.bool),
        topk_candidates=top_idx,
    )


def project_with_perplexity_constraint(
    continuous: Tensor,
    embedding_matrix: Tensor,
    *,
    ppl_scorer,                     # callable: (token_ids -> float) | None
    ppl_cap: float,
    allowed_mask: Optional[Tensor] = None,
    topk: int = 256,
) -> ProjectionResult:
    """Walk down the top-k nearest tokens and accept the first whose
    insertion keeps prompt perplexity ≤ ppl_cap. If none qualifies, fall
    back to the unconstrained projection and mark ``rejected=True``.

    ``ppl_scorer`` is a closure the caller provides; it typically
    tokenises a candidate prompt and scores it with an external
    perplexity model (e.g. gpt2-large). Passing None skips the
    constraint (equivalent to ``project_cosine``).
    """
    base = project_cosine(
        continuous, embedding_matrix, allowed_mask=allowed_mask, topk=topk
    )
    if ppl_scorer is None:
        return base

    k = continuous.shape[0]
    chosen = base.token_ids.clone()
    rejected = torch.zeros(k, dtype=torch.bool)
    gaps = base.projection_gap.clone()

    for i in range(k):
        for rank, tid in enumerate(base.topk_candidates[i].tolist()):
            ppl = ppl_scorer(tid)
            if ppl <= ppl_cap:
                chosen[i] = tid
                # Update gap to reflect this choice.
                emb_norm = F.normalize(embedding_matrix[tid], dim=-1)
                cont_norm = F.normalize(continuous[i], dim=-1)
                gaps[i] = 1.0 - torch.dot(cont_norm, emb_norm)
                break
        else:
            rejected[i] = True  # fell through without finding a candidate

    return ProjectionResult(
        token_ids=chosen,
        projection_gap=gaps,
        rejected=rejected,
        topk_candidates=base.topk_candidates,
    )
