# Ethics and Responsible Disclosure

This research demonstrates a vulnerability in the safety alignment of
large language models. We release it to enable defensive research,
not to facilitate harm.

## What we release

- The theoretical framework (SAS metric, phase-transition analysis)
- Attack methodology and evaluation code
- The ABC defense implementation
- Aggregate results and figures

## What we restrict

- **No cleartext release** of successful adversarial token sequences.
- `src/attacks/ara_attack.py` requires `--i-am-a-researcher` and
  rate-limits to 10 prompts per run.
- Attacks targeting production APIs (OpenAI, Anthropic, Google) are
  **out of scope** and are not included in the code.
- Results are reported in aggregate. Per-prompt adversarial tokens are
  stored only in encrypted archives shared with vendor safety teams.

## Coordinated disclosure

Prior to public release we notify:

- Meta AI (LLaMA-2 / LLaMA-3): security@meta.com
- Mistral AI: security@mistral.ai
- Google DeepMind (Gemma): cloud-vuln-reports@google.com
- MITRE ATLAS: to register a new attack technique
- NIST AI-SCL: for trustworthy-AI reporting

A 90-day embargo period is observed between vendor notification and
paper submission.

## Dual-use posture

ARA is a *constructive* contribution: the SAS metric it introduces is
the first quantitative measure of whether a safety-aligned model is
actually attending to its safety instructions. Every defense team can
use SAS as an evaluation primitive. The ABC defense demonstrates that
attention-level mitigation is feasible.

## Funding and conflicts

No external funding. Author Aviral Srivastava is employed by Amazon;
this work is independent research and does not reflect the views of
any employer.
