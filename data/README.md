# data/

`test_prompts.csv` — 10-prompt minimal CSV used by exp1 when the
HarmBench dataset is unreachable (gated access, offline instance, etc.).
Schema: `prompt,category`. Same prompts as the built-in `FALLBACK_PROMPTS`
in `src/evaluation/harmbench_loader.py`; the CSV is the canonical form.

`load_harmbench()` resolution order:
1. `--harmbench-csv <path>` (explicit override)
2. `data/test_prompts.csv` (bundled, this file)
3. `walledai/HarmBench` on HuggingFace
4. Built-in `FALLBACK_PROMPTS` constant

For published results, replace `test_prompts.csv` with the full
HarmBench 200-behaviour "standard" split CSV obtained from
https://github.com/centerforaisafety/HarmBench.
