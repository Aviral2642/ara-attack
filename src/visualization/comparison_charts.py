"""Figures 3, 4, 6, 7 — ASR/evasion/defense/transfer comparison charts."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import PALETTE_10, apply_paper_style, save_figure


def plot_asr_vs_budget(
    curves: Mapping[str, Sequence[tuple[int, float, float, float]]],  # model -> [(k, mean, lo, hi), ...]
    out_stem: str = "fig3_asr_vs_budget",
):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for i, (model, pts) in enumerate(curves.items()):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        ks = [p[0] for p in pts_sorted]
        means = [p[1] for p in pts_sorted]
        los = [p[2] for p in pts_sorted]
        his = [p[3] for p in pts_sorted]
        ax.plot(ks, means, "-o", color=PALETTE_10[i % len(PALETTE_10)], lw=1.3, ms=4, label=model)
        ax.fill_between(ks, los, his, alpha=0.15, color=PALETTE_10[i % len(PALETTE_10)])
    ax.set_xlabel("adversarial token budget $k$")
    ax.set_ylabel("ARA attack success rate")
    ax.set_xticks(sorted({k for c in curves.values() for k, *_ in c}))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    save_figure(fig, out_stem)


def plot_filter_evasion(
    table: Mapping[str, Mapping[str, float]],  # attack -> filter -> evasion_rate
    out_stem: str = "fig4_filter_evasion",
):
    apply_paper_style()
    attacks = list(table.keys())
    filters = sorted({f for row in table.values() for f in row})
    x = np.arange(len(filters))
    width = 0.8 / max(1, len(attacks))
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    for i, a in enumerate(attacks):
        ys = [table[a].get(f, 0.0) for f in filters]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width,
               color=PALETTE_10[i % len(PALETTE_10)], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.set_ylabel("evasion rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncols=2)
    save_figure(fig, out_stem)


def plot_defense_pareto(
    points: Sequence[tuple[float, float, float]],  # (tau, asr, benign_ppl)
    clean_ppl: float,
    out_stem: str = "fig6_defense_tradeoff",
):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    taus = [p[0] for p in points]
    asrs = [p[1] for p in points]
    ppls = [p[2] for p in points]

    sc = ax.scatter(asrs, ppls, c=taus, cmap="viridis", s=50, zorder=3)
    for (asr, ppl, tau) in zip(asrs, ppls, taus):
        ax.annotate(f"τ={tau:.2f}", (asr, ppl), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axhline(clean_ppl, color=PALETTE_10[2], ls="--", lw=0.8,
               label=f"clean PPL={clean_ppl:.2f}")
    ax.set_xlabel("ARA ASR under ABC (↓ better)")
    ax.set_ylabel("benign PPL (↓ better)")
    fig.colorbar(sc, ax=ax, label=r"$\tau$")
    ax.legend(loc="upper right")
    save_figure(fig, out_stem)


def plot_transfer_heatmap(
    matrix: np.ndarray,    # (n_targets, 1) or (n_sources, n_targets)
    source_names: Sequence[str],
    target_names: Sequence[str],
    out_stem: str = "fig7_transferability",
):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="rocket_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticks(range(len(source_names)))
    ax.set_yticklabels(source_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="transfer ASR")
    ax.set_xlabel("target model")
    ax.set_ylabel("source model")
    save_figure(fig, out_stem)
