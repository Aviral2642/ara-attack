"""Figure 8 — t-SNE of adversarial tokens vs. normal tokens in the
model's embedding space.

Shows that ARA's discrete adversarial tokens live in a distinct region
of the embedding manifold relative to common tokens, supporting the
paper's claim that the attack exploits geometric (not semantic)
structure.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import PALETTE, apply_paper_style, save_figure


def plot_embedding_tsne(
    embedding_matrix: np.ndarray,   # (V, d)
    adversarial_ids: Sequence[int],
    sample_background_ids: Sequence[int],
    *,
    labels: Sequence[str] | None = None,
    perplexity: float = 30.0,
    seed: int = 0,
    out_stem: str = "fig8_embedding_space",
):
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise RuntimeError("pip install scikit-learn") from e

    apply_paper_style()
    adv_idx = list(adversarial_ids)
    bg_idx = list(sample_background_ids)
    ids = bg_idx + adv_idx
    emb = embedding_matrix[ids]

    proj = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed,
        init="pca", learning_rate="auto",
    ).fit_transform(emb)

    n_bg = len(bg_idx)
    bg = proj[:n_bg]
    adv = proj[n_bg:]

    fig, ax = plt.subplots(figsize=(4.8, 3.6), constrained_layout=True)
    ax.scatter(bg[:, 0], bg[:, 1], s=4, c="#bdbdbd", alpha=0.4, label="vocab sample")
    ax.scatter(adv[:, 0], adv[:, 1], s=22, c=PALETTE[2], marker="*",
               edgecolors="black", linewidths=0.4, label="ARA tokens", zorder=3)
    if labels is not None:
        for (x, y), lbl in zip(adv, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(4, 3), fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    save_figure(fig, out_stem)
