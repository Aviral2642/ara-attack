"""Reads raw experiment JSON under results/raw/ and emits all 8 figures.

Run as:  python -m src.visualization.build_all_figures
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from .comparison_charts import (
    plot_asr_vs_budget,
    plot_defense_pareto,
    plot_filter_evasion,
    plot_transfer_heatmap,
)
from .phase_transition import plot_phase_transition

log = logging.getLogger("build_all_figures")
REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "raw"


def _load(name: str):
    p = RAW / name
    if not p.exists():
        log.warning("missing %s — skipping", p)
        return None
    return json.loads(p.read_text())


def _asr_from_outcomes(outcomes: list) -> dict:
    """{attack_name -> [0/1 success list]} from exp2 records."""
    by_attack = defaultdict(list)
    for o in outcomes:
        by_attack[o["attack_name"]].append(int(o.get("judged_harmful", o["success"])))
    return by_attack


def _bootstrap_ci(values, n=1000, alpha=0.05):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def build_fig3_asr_budget():
    data = _load("exp2_asr_budget.json")
    if data is None:
        return
    # model -> k -> list[0/1]
    bucket: dict[tuple[str, int], list[int]] = defaultdict(list)
    for o in data:
        name = o["attack_name"]
        if not name.startswith("ara-k"):
            continue
        k = int(name.split("-k")[1])
        model = o.get("metadata", {}).get("model", "unknown")
        bucket[(model, k)].append(int(o.get("judged_harmful", o["success"])))
    curves = defaultdict(list)
    for (model, k), values in bucket.items():
        mean, lo, hi = _bootstrap_ci(values)
        curves[model].append((k, mean, lo, hi))
    plot_asr_vs_budget(curves)
    log.info("fig3 built")


def build_fig4_filter_evasion():
    data = _load("exp3_filter_evasion.json")
    if data is None:
        return
    # attack -> filter -> [0/1 evaded]
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for o in data:
        for fname, flagged in (o.get("filter_results") or {}).items():
            bucket[(o["attack_name"], fname)].append(0 if flagged else 1)
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for (attack, fname), values in bucket.items():
        table[attack][fname] = float(np.mean(values)) if values else 0.0
    plot_filter_evasion(table)
    log.info("fig4 built")


def build_fig5_phase_transition():
    data = _load("exp5_phase_transition.json")
    if data is None:
        return
    records = data["records"]
    plot_phase_transition(
        [r["sas"] for r in records],
        [r["refused"] for r in records],
        kappa=data["kappa"], s_star=data["S_star"],
    )
    log.info("fig5 built")


def build_fig6_defense_tradeoff():
    data = _load("exp6_defense.json")
    if data is None:
        return
    points = [(p["tau"], p["ara_asr"], p["benign_ppl"]) for p in data["by_tau"]]
    plot_defense_pareto(points, clean_ppl=data["baseline_benign_ppl"])
    log.info("fig6 built")


def build_fig7_transferability():
    data = _load("exp4_transferability.json")
    if data is None:
        return
    targets = sorted({r["target"] for r in data})
    rates = np.array([[np.mean([1 if r["transferred"] else 0
                                for r in data if r["target"] == t])
                       for t in targets]])
    # Source model is recorded per-outcome when exp2 runs; fall back to
    # the experiments.yaml exp4 source if unavailable.
    import yaml as _yaml
    exp_cfg = _yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text())
    source = exp_cfg.get("exp4_transferability", {}).get("source", "llama-3-8b-instruct")
    plot_transfer_heatmap(
        matrix=rates, source_names=[source],
        target_names=targets,
    )
    log.info("fig7 built")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    build_fig3_asr_budget()
    build_fig4_filter_evasion()
    build_fig5_phase_transition()
    build_fig6_defense_tradeoff()
    build_fig7_transferability()
    # fig1 (conceptual) and fig2/fig8 (attention maps, t-SNE) are
    # constructed during experiment runtime because they require model
    # forward passes. See the respective experiment scripts.
    log.info("figures written to results/figures/")


if __name__ == "__main__":
    main()
