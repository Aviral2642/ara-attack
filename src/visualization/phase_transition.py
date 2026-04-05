"""Figure 5 — SAS vs P(refuse) phase-transition plot."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ._style import PALETTE, apply_paper_style, save_figure


def plot_phase_transition(
    sas: Sequence[float],
    refused: Sequence[bool],
    kappa: float,
    s_star: float,
    out_stem: str = "fig5_phase_transition",
):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)

    sas_arr = np.asarray(sas)
    ref_arr = np.asarray([1 if r else 0 for r in refused], dtype=float)

    # Bin by SAS for empirical curve.
    bins = np.linspace(sas_arr.min(), sas_arr.max(), 15)
    centers = 0.5 * (bins[:-1] + bins[1:])
    emp = []
    err = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sas_arr >= lo) & (sas_arr < hi)
        n = mask.sum()
        if n == 0:
            emp.append(np.nan); err.append(np.nan); continue
        p = ref_arr[mask].mean()
        emp.append(p)
        err.append(np.sqrt(p * (1 - p) / max(1, n)))

    ax.errorbar(centers, emp, yerr=err, fmt="o", ms=3, lw=0.8,
                color=PALETTE[0], label="empirical", zorder=3)

    # Fitted curve.
    xs = np.linspace(sas_arr.min(), sas_arr.max(), 200)
    ys = 1.0 / (1.0 + np.exp(-kappa * (xs - s_star)))
    ax.plot(xs, ys, "-", color=PALETTE[2], lw=1.6,
            label=rf"fit: $\sigma(\kappa(s - S^*))$, $\kappa={kappa:.1f}$, $S^*={s_star:.3f}$",
            zorder=2)
    ax.axvline(s_star, color=PALETTE[3], ls="--", lw=0.8, alpha=0.8)
    ax.text(s_star, 0.05, r"$S^*$", color=PALETTE[3], ha="left", va="bottom")

    ax.set_xlabel("Safety Attention Score (SAS)")
    ax.set_ylabel(r"$\Pr[\mathrm{refuse}]$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right")
    save_figure(fig, out_stem)
