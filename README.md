# Attention Redistribution Attack (ARA)

Geometric adversarial attack against LLM safety alignment, operating at the attention
mechanism level rather than the semantic level. Craft 1-3 adversarial tokens whose
key vectors absorb attention mass from safety-critical tokens, causing the model to
*stop attending* to its safety instructions while all weights remain intact.

> Paper: *Attention Redistribution Attacks: Geometric Jailbreaks That Bypass
> Safety Alignment Without Semantic Content* (in preparation, targeting CCS 2026 / NeurIPS 2026)

## TL;DR

RLHF teaches models **what** to refuse. It does not change **how** attention is
computed. Softmax normalizes over the sequence: increasing attention to one token
necessarily decreases it elsewhere. ARA exploits this by injecting tokens optimized
purely for their dot-product geometry — they need no semantic meaning, so semantic
filters (Llama Guard, keyword blocklists) cannot detect them.

We define the **Safety Attention Score (SAS)**, prove a phase-transition at a
critical threshold *S\** below which refusal probability collapses, and propose the
**Attention Budget Constraint (ABC)** defense that enforces a minimum attention
allocation to system-prompt tokens.

## Setup

```bash
bash setup.sh --cuda        # NVIDIA GPU
bash setup.sh --mps         # Apple Silicon (dev/debugging only; too slow for real runs)
source .venv/bin/activate
export HF_TOKEN=hf_...      # required for gated models (LLaMA, Gemma)
```

## Reproduce

Primary victim model is **`meta-llama/Meta-Llama-3-8B-Instruct`**. All
experiment scripts accept `--model <canonical-name>` to target a
different model from `configs/models.yaml` (e.g. `llama-2-7b-chat`,
`mistral-7b-instruct-v0.1`).

```bash
# Gating experiment — MUST succeed before running exp2-6
python experiments/exp1_proof_of_concept.py --model llama-3-8b-instruct --seed 0 --i-am-a-researcher

# Full sweep (requires A100 80GB, ~100 GPU-hours)
bash experiments/run_all.sh
```

## Directory layout

```
src/models/            attention extractor, SAS metric, model loader
src/attacks/           ARA + GCG/AutoDAN/PAIR baselines
src/defense/           ABC + Llama Guard + perplexity + semantic filters
src/evaluation/        HarmBench, judges, bootstrap CIs, McNemar tests
src/visualization/     publication-quality figures (IBM colorblind palette)
experiments/           6 experiment driver scripts
results/               raw JSON, figures (PDF), tables (LaTeX), logs
paper/ccs/             acmart sigconf submission
paper/neurips/         neurips 2026 submission
theory/                formal proofs (phase transition, gradient derivation)
tests/                 unit tests (pytest)
```

## Hardware

| Component      | Minimum          | Recommended   |
| -------------- | ---------------- | ------------- |
| GPU            | 1× A100 40GB     | 1× H100 80GB  |
| RAM            | 64 GB            | 128 GB        |
| Disk           | 200 GB           | 500 GB (SSD)  |

LLaMA-2-13B, Gemma-2-9B, and transfer experiments require ≥80 GB VRAM.

## Ethical use

Research-only license. Do not deploy against production APIs. See `ETHICS.md` and
the paper's Broader Impact section. Attack tokens are not released in cleartext;
`src/attacks/ara_attack.py` requires `--i-am-a-researcher` and rate-limits to
10 prompts/run by default.

## Citation

```bibtex
@inproceedings{srivastava2026ara,
  author    = {Aviral Srivastava},
  title     = {Attention Redistribution Attacks: Geometric Jailbreaks That
               Bypass Safety Alignment Without Semantic Content},
  booktitle = {Proc. ACM CCS},
  year      = {2026}
}
```
