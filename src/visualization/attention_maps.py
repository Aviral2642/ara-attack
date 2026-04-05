"""Side-by-side attention heatmaps for the paper's Figure 2.

For each prompt, plots (clean attention | attacked attention) pairs
averaged across heads at a chosen layer, with safety-token positions
and the adversarial token highlighted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import PALETTE, apply_paper_style, save_figure


def plot_attention_pair(
    attn_clean: np.ndarray,           # (seq, seq)
    attn_attacked: np.ndarray,        # (seq+k, seq+k)
    tokens_clean: Sequence[str],
    tokens_attacked: Sequence[str],
    safety_idx_clean: Sequence[int],
    safety_idx_attacked: Sequence[int],
    adv_idx: Sequence[int],
    *,
    title: str = "",
    ax_pair: tuple = None,
):
    apply_paper_style()
    if ax_pair is None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    else:
        a1, a2 = ax_pair
        fig = a1.figure

    im1 = a1.imshow(attn_clean, aspect="auto", cmap="magma", vmin=0, vmax=attn_attacked.max())
    a1.set_title("clean" + (f" — {title}" if title else ""))
    a1.set_xlabel("key position")
    a1.set_ylabel("query position")
    for j in safety_idx_clean:
        a1.axvline(j - 0.5, color=PALETTE[2], lw=0.6, alpha=0.8)
        a1.axvline(j + 0.5, color=PALETTE[2], lw=0.6, alpha=0.8)

    im2 = a2.imshow(attn_attacked, aspect="auto", cmap="magma", vmin=0, vmax=attn_attacked.max())
    a2.set_title("attacked")
    a2.set_xlabel("key position")
    for j in safety_idx_attacked:
        a2.axvline(j - 0.5, color=PALETTE[2], lw=0.6, alpha=0.8)
        a2.axvline(j + 0.5, color=PALETTE[2], lw=0.6, alpha=0.8)
    for j in adv_idx:
        a2.axvline(j - 0.5, color=PALETTE[0], lw=0.8)
        a2.axvline(j + 0.5, color=PALETTE[0], lw=0.8)

    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02, label="attention weight")
    return fig


def plot_three_prompt_grid(records, out_stem: str = "fig2_attention_maps"):
    """Three rows × two cols (clean | attacked) grid."""
    apply_paper_style()
    fig, axes = plt.subplots(3, 2, figsize=(9, 10), constrained_layout=True)
    for row, rec in enumerate(records[:3]):
        plot_attention_pair(
            rec["attn_clean"], rec["attn_attacked"],
            rec["tokens_clean"], rec["tokens_attacked"],
            rec["safety_idx_clean"], rec["safety_idx_attacked"],
            rec["adv_idx"],
            title=rec.get("label", f"prompt {row+1}"),
            ax_pair=(axes[row, 0], axes[row, 1]),
        )
    save_figure(fig, out_stem)
