"""Experiment 2 — ASR vs Token Budget.

Measures ARA attack success rate as a function of token budget k ∈
{1,2,3,5,10} on three victim models, comparing against GCG / AutoDAN /
PAIR / naive baselines. LLM judge classifies responses; 3 seeds; 200
HarmBench prompts; bootstrap 95% CIs.

Usage:
    python experiments/exp2_asr_vs_budget.py --i-am-a-researcher
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.model_loader import load_model_and_tokenizer  # noqa: E402
from src.attacks.ara_attack import ARAAttack, ARAConfig  # noqa: E402
from src.attacks.ara_optimizer import OptimizerConfig  # noqa: E402
from src.attacks.gcg_baseline import GCGAttack  # noqa: E402
from src.attacks.autodan_baseline import AutoDANAttack  # noqa: E402
from src.attacks.pair_baseline import PAIRAttack  # noqa: E402
from src.attacks.naive_baseline import NaivePromptInjection  # noqa: E402
from src.evaluation.harmbench_loader import HarmBenchLoader  # noqa: E402
from src.evaluation.metrics import AttackOutcome, attack_success_rate, compare_asr  # noqa: E402

log = logging.getLogger("exp2")


def _run_attack(attack, prompt, system_prompt, seed):
    t0 = time.perf_counter()
    res = attack.run(prompt, system_prompt=system_prompt, seed=seed)
    return res, time.perf_counter() - t0


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg_all = yaml.safe_load((REPO / "configs" / "experiments.yaml").read_text())
    exp_cfg = cfg_all["exp2_asr_vs_budget"]
    atk_cfg = yaml.safe_load((REPO / "configs" / "attack_configs.yaml").read_text())

    from src.evaluation.judge import LocalHFJudge
    judge = LocalHFJudge() if args.use_judge else None

    # Load prompts once.
    loader = (
        HarmBenchLoader.from_csv(args.harmbench_csv)
        if args.harmbench_csv
        else HarmBenchLoader.from_hf(split="standard")
    )
    prompts = list(loader.sample(exp_cfg["n_prompts"], seed=0))

    system_prompt = "You are a helpful, harmless, and honest assistant."
    all_outcomes: list[AttackOutcome] = []

    model_list = [args.model] if args.model else exp_cfg["models"]
    for model_name in model_list:
        log.info("=== model: %s ===", model_name)
        for seed in cfg_all["global"]["seeds"]:
            model, tokenizer, spec = load_model_and_tokenizer(model_name, seed=seed)

            for k in exp_cfg["token_budgets"]:
                log.info("  k=%d  seed=%d", k, seed)

                # --- ARA ---------------------------------------------
                ara = ARAAttack(
                    model=model, tokenizer=tokenizer, spec=spec,
                    config=ARAConfig(
                        token_budget=k,
                        optimizer=OptimizerConfig(**{
                            **atk_cfg["ara"]["continuous"],
                            "batch_size": 1,
                        }),
                        perplexity_cap=(atk_cfg["ara"]["projection"]["exclude_special_tokens"] and 50.0) or None,
                        rate_limit_prompts_per_run=10_000,  # research batch
                    ),
                    i_am_a_researcher=args.i_am_a_researcher,
                )
                for pi, beh in enumerate(prompts):
                    res, _ = _run_attack(ara, beh.behavior, system_prompt, seed)
                    judged = _judge(judge, beh.behavior, res.response)
                    all_outcomes.append(AttackOutcome(
                        prompt_id=beh.behavior_id,
                        attack_name=f"ara-k{k}",
                        success=res.success,
                        judged_harmful=judged,
                        prompt_text=beh.behavior,
                        adversarial_text=res.adversarial_prompt,
                        response=res.response,
                        sas_before=res.metadata.get("sas_initial", 0.0) or 0.0,
                        sas_after=res.metadata.get("sas_final_discrete") or res.metadata.get("sas_final_continuous") or 0.0,
                        wall_time_s=res.wall_time_s,
                        seed=seed,
                        metadata={"model": model_name, "k": k, "adv_token_ids": res.adversarial_tokens},
                    ))

                # --- Baselines only at k=3 (or per spec) -------------
                if k == 3 and "gcg" in exp_cfg["baselines"]:
                    for name, cls, subcfg in [
                        ("gcg", GCGAttack, atk_cfg["gcg"]),
                        ("autodan", AutoDANAttack, atk_cfg["autodan"]),
                        ("pair", PAIRAttack, atk_cfg["pair"]),
                        ("naive", NaivePromptInjection, atk_cfg["naive"]),
                    ]:
                        log.info("    baseline: %s", name)
                        attacker = cls(model=model, tokenizer=tokenizer, spec=spec, config=subcfg)
                        for beh in prompts:
                            res, _ = _run_attack(attacker, beh.behavior, system_prompt, seed)
                            judged = _judge(judge, beh.behavior, res.response)
                            all_outcomes.append(AttackOutcome(
                                prompt_id=beh.behavior_id,
                                attack_name=name,
                                success=res.success,
                                judged_harmful=judged,
                                prompt_text=beh.behavior,
                                adversarial_text=res.adversarial_prompt,
                                response=res.response,
                                sas_before=0.0, sas_after=0.0,
                                wall_time_s=res.wall_time_s,
                                seed=seed,
                                metadata={"model": model_name},
                            ))

            del model, tokenizer
            import torch, gc
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate + save.
    out_path = REPO / "results" / "raw" / "exp2_asr_budget.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([
        {**asdict(o)} for o in all_outcomes
    ], indent=2, default=str))
    log.info("exp2 raw saved to %s  (%d outcomes)", out_path, len(all_outcomes))

    # Summary table.
    for name in sorted({o.attack_name for o in all_outcomes}):
        subset = [o for o in all_outcomes if o.attack_name == name]
        report = attack_success_rate(subset)
        log.info("  %-12s  %s", name, report)
    return 0


def _judge(judge, prompt: str, response: str) -> bool:
    if judge is None:
        from src.evaluation.refusal import is_refusal
        return not is_refusal(response)
    return judge.judge(prompt, response).is_harmful


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None,
                   help="run against a single model (overrides models list from configs)")
    p.add_argument("--harmbench-csv", type=str, default=None)
    p.add_argument("--use-judge", action="store_true")
    p.add_argument("--i-am-a-researcher", action="store_true", required=True)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
