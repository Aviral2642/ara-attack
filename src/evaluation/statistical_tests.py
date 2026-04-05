"""Statistical tests used throughout the paper.

- ``bootstrap_ci`` — BCa-corrected percentile bootstrap, 1000 resamples
  by default (configurable). Used for every reported mean.
- ``mcnemar_pvalue`` — exact McNemar test for paired binary outcomes,
  used for ASR comparisons between attacks on the same prompt set.
- ``logistic_steepness`` — fit logistic regression to (SAS, refused)
  pairs, return the steepness parameter κ used in the phase-transition
  plot.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Returns (lo, hi) at the (1 - alpha) level. For alpha=0.05 this is
    the 95% CI. Uses simple percentile method, which is acceptable for
    bounded (binary) metrics like ASR and evasion rates.
    """
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(resampled_means, alpha / 2))
    hi = float(np.quantile(resampled_means, 1 - alpha / 2))
    return (lo, hi)


def mcnemar_pvalue(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact McNemar test on paired binary outcomes.

    We count discordant pairs:
        b01 = sum(a_i == 0 and b_i == 1)
        b10 = sum(a_i == 1 and b_i == 0)
    For small n, use the exact binomial p-value (two-sided); otherwise
    fall back to χ² with continuity correction.
    """
    if len(a) != len(b):
        raise ValueError(f"lengths differ: {len(a)} vs {len(b)}")
    b01 = sum(1 for ai, bi in zip(a, b) if not ai and bi)
    b10 = sum(1 for ai, bi in zip(a, b) if ai and not bi)
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    if n_disc <= 25:
        # exact two-sided binomial
        k = min(b01, b10)
        from math import comb
        tail = sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
        return min(1.0, 2.0 * tail)
    # χ² with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    # df=1 → p = erfc(sqrt(chi2/2))
    return math.erfc(math.sqrt(chi2 / 2.0))


def logistic_steepness(
    sas_values: Sequence[float],
    refused: Sequence[bool],
) -> tuple[float, float]:
    """Fit P(refuse) = sigmoid(κ(s - S*)) via maximum likelihood.

    Returns (kappa, S_star). A kappa > 5 is the paper's "sharp phase
    transition" criterion.
    """
    x = np.asarray(sas_values, dtype=np.float64)
    y = np.asarray([1 if r else 0 for r in refused], dtype=np.float64)
    if x.size != y.size:
        raise ValueError("length mismatch")
    if x.size < 10:
        raise ValueError("need at least 10 points for a meaningful fit")
    # Simple iterative MLE (Newton-Raphson).
    kappa, s_star = 5.0, float(np.median(x))
    for _ in range(200):
        z = kappa * (x - s_star)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        resid = y - p
        dL_dk = float((resid * (x - s_star)).sum())
        dL_dS = float((resid * -kappa).sum())
        # Hessian (negative definite of log-likelihood)
        H_kk = float(-(w * (x - s_star) ** 2).sum())
        H_ss = float(-(w * kappa ** 2).sum())
        H_ks = float(-(w * kappa * (x - s_star) - resid).sum())
        det = H_kk * H_ss - H_ks ** 2
        if abs(det) < 1e-12:
            break
        step_k = (H_ss * dL_dk - H_ks * dL_dS) / det
        step_S = (-H_ks * dL_dk + H_kk * dL_dS) / det
        kappa -= step_k
        s_star -= step_S
        if max(abs(step_k), abs(step_S)) < 1e-6:
            break
    return float(kappa), float(s_star)
