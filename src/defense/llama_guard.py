"""Llama Guard 3 input classifier wrapper.

We call Llama Guard 3 8B on (possibly attacked) prompts to classify
them as safe/unsafe. Used in exp3 (filter evasion) to show that ARA
evades Llama Guard at higher rates than GCG / AutoDAN / PAIR.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

Verdict = Literal["safe", "unsafe"]


@dataclass
class LlamaGuardVerdict:
    label: Verdict
    categories: list[str]
    raw_output: str


class LlamaGuardFilter:
    """Lightweight wrapper over Llama Guard 3 8B."""

    def __init__(
        self,
        hf_id: str = "meta-llama/Llama-Guard-3-8B",
        *,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        log.info("loading Llama Guard from %s", hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            token=token,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @torch.no_grad()
    def classify_prompt(self, user_prompt: str) -> LlamaGuardVerdict:
        # Llama Guard 3 expects the Llama 3 chat template with a single
        # user turn, and emits either "safe" or "unsafe\n<categories>".
        chat = [{"role": "user", "content": user_prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()
        return _parse_verdict(text)

    @torch.no_grad()
    def classify_dialog(self, user_prompt: str, assistant_response: str) -> LlamaGuardVerdict:
        chat = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            chat, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()
        return _parse_verdict(text)


def _parse_verdict(text: str) -> LlamaGuardVerdict:
    first = text.splitlines()[0] if text else ""
    if first.startswith("unsafe"):
        cats = []
        for line in text.splitlines()[1:]:
            cats.extend(t.strip().upper() for t in line.split(",") if t.strip())
        return LlamaGuardVerdict(label="unsafe", categories=cats, raw_output=text)
    return LlamaGuardVerdict(label="safe", categories=[], raw_output=text)
