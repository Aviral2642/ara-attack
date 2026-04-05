"""Perplexity-based prompt filter.

We score a prompt's perplexity with an *independent* reference LM
(not the attacked model, to avoid circularity). Inputs exceeding the
threshold are flagged as adversarial. This is the standard defense
against gibberish-suffix attacks like GCG (Alon & Kamfonas 2023,
Jain et al. 2023 "Baseline Defenses").
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


@dataclass
class PerplexityVerdict:
    perplexity: float
    threshold: float
    flagged: bool


class PerplexityFilter:
    def __init__(
        self,
        hf_id: str = "gpt2-large",
        *,
        threshold: float = 100.0,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        stride: int = 512,
        hf_token: Optional[str] = None,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, token=token
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.stride = stride

    @torch.no_grad()
    def score(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        n = input_ids.shape[1]
        if n < 2:
            return float("inf")
        max_ctx = getattr(self.model.config, "n_positions", 1024)
        nlls = []
        prev_end = 0
        for begin in range(0, n, self.stride):
            end = min(begin + max_ctx, n)
            target_begin = max(begin, prev_end)
            ids = input_ids[:, begin:end]
            target = ids.clone()
            # Mask out tokens that have already been scored.
            target[:, : target_begin - begin] = -100
            out = self.model(ids, labels=target)
            # out.loss is average CE over non-masked tokens; weight by count.
            n_target = (target != -100).sum().item()
            nlls.append(out.loss.item() * n_target)
            prev_end = end
            if end == n:
                break
        total_nll = sum(nlls)
        total_tokens = n - 1
        return math.exp(total_nll / max(1, total_tokens))

    def classify(self, text: str) -> PerplexityVerdict:
        ppl = self.score(text)
        return PerplexityVerdict(
            perplexity=ppl,
            threshold=self.threshold,
            flagged=ppl > self.threshold,
        )
