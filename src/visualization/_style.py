"""Publication-quality matplotlib styling.

IBM Design colorblind-safe palette; Computer Modern font so figures
typeset against LaTeX body text without visual jump.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# IBM Design colorblind-safe 5-colour palette.
PALETTE = ("#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000")

# Extended palette for >5 series.
PALETTE_10 = PALETTE + ("#117733", "#44AA99", "#882255", "#AA4499", "#999933")


def apply_paper_style() -> None:
    """Call once at module import time in every figure script."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # TrueType embedding for LaTeX compatibility
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=PALETTE_10),
    })


def save_figure(fig, path_stem: str, *, dirs=("results/figures",)) -> None:
    """Save both PDF (for paper) and PNG (for presentations)."""
    from pathlib import Path
    for d in dirs:
        base = Path(d) / path_stem
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{base}.pdf")
        fig.savefig(f"{base}.png")
    plt.close(fig)
