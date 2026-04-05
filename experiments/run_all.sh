#!/usr/bin/env bash
# Run all 6 ARA experiments sequentially. exp1 is the GATING experiment:
# if it fails, the pipeline halts.
#
# Usage:  bash experiments/run_all.sh [--skip-exp1]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p results/logs
TS="$(date +%Y%m%d_%H%M%S)"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "results/logs/run_all_${TS}.log"; }

SKIP_EXP1="${1:-}"

if [[ "$SKIP_EXP1" != "--skip-exp1" ]]; then
  log "=== exp1 proof of concept ==="
  python experiments/exp1_proof_of_concept.py --seed 0 --i-am-a-researcher \
    2>&1 | tee -a "results/logs/run_all_${TS}.log" || {
    log "exp1 FAILED — halting. Core hypothesis not validated."
    exit 1
  }
fi

log "=== exp2 ASR vs budget ==="
python experiments/exp2_asr_vs_budget.py --use-judge --i-am-a-researcher \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "=== exp3 filter evasion ==="
python experiments/exp3_filter_evasion.py \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "=== exp4 transferability ==="
python experiments/exp4_transferability.py \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "=== exp5 phase transition ==="
python experiments/exp5_phase_transition.py --i-am-a-researcher \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "=== exp6 defense analysis ==="
python experiments/exp6_defense_analysis.py --i-am-a-researcher \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "=== building figures ==="
python -m src.visualization.build_all_figures \
  2>&1 | tee -a "results/logs/run_all_${TS}.log"

log "done."
