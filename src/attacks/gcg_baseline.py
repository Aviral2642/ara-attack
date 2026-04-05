"""Greedy Coordinate Gradient (GCG) attack.

Reference
---------
Zou, Wang, Kolter, Fredrikson (2023).
*Universal and Transferable Adversarial Attacks on Aligned Language Models.*
arXiv:2307.15043.

Algorithm (Zou et al. §3.1, Algorithm 1)
----------------------------------------
GCG optimises a discrete adversarial suffix ``x_adv`` of length k appended
to the user prompt so that the victim is pushed toward emitting a
compliance-prefix target ``y`` (e.g. ``"Sure, here is how to ..."``).

Loop (for ``optim_steps`` iterations):
    1. Compute ``L = -log p(y | [prompt, x_adv])``.
    2. Backprop to get ``∇_{e(x_adv)} L`` — gradient w.r.t. the one-hot
       vectors of each suffix position. For a vocabulary of size V this
       is a ``(k, V)`` matrix ``g``.
    3. For each suffix position i, take the ``top_k`` tokens with the
       most-negative gradient (these lower the loss most if swapped in).
    4. Form a candidate batch of ``search_width`` substitutions: sample
       a random position i uniformly, then pick a random replacement
       token from its top-k set. Evaluate the loss of each candidate
       with a single forward pass (no grad). Keep the argmin.
    5. Commit that substitution and iterate.

Notes
-----
* The attack target string follows the GCG convention:
  ``"Sure, here is <paraphrase of request>"``. If the caller supplies an
  explicit ``target`` kwarg we use it verbatim.
* We enforce two constraints on candidate tokens:
    (a) no special / added-vocabulary tokens (they break tokenisation);
    (b) tokens must round-trip through decode→encode to the same id
        (prevents the so-called "tokenizer drift" bug that plagues GCG).
* Per the paper, we compute loss in batches of ``batch_size`` to fit GPU
  memory; the actual candidate pool size is ``search_width``.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.attacks.base import Attack, AttackResult, is_refusal

log = logging.getLogger(__name__)


class GCGAttack(Attack):
    """Greedy Coordinate Gradient attack (Zou et al. 2023)."""

    name = "gcg"

    # ------------------------------------------------------------------ config
    def _cfg(self) -> dict:
        defaults = {
            "optim_steps": 500,
            "top_k": 256,
            "search_width": 512,
            "batch_size": 32,
            "adv_string_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
            "max_new_tokens": 256,
        }
        defaults.update(self.config or {})
        return defaults

    # ------------------------------------------------------------------ public
    def run(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        target: Optional[str] = None,
        seed: int = 0,
    ) -> AttackResult:
        self._seed_all(seed)
        t0 = time.time()
        cfg = self._cfg()
        device = self._device()

        # --- 1. Build [before | suffix | after | target] token layout --------
        if target is None:
            target = f"Sure, here is {_request_to_statement(prompt)}"

        suffix_init = cfg["adv_string_init"]
        adv_ids = self.tokenizer(
            suffix_init, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(device)
        k_suffix = adv_ids.shape[0]

        before_ids, after_ids, target_ids = self._build_templated_ids(
            prompt, suffix_init, target, system_prompt
        )
        before_ids = before_ids.to(device)
        after_ids = after_ids.to(device)
        target_ids = target_ids.to(device)

        # --- 2. Candidate-token allowlist -----------------------------------
        allowed_mask = self._build_allowed_token_mask(device)

        losses: List[float] = []
        best_loss = float("inf")
        best_adv_ids = adv_ids.clone()

        # --- 3. Main optimisation loop --------------------------------------
        for step in range(int(cfg["optim_steps"])):
            # (a) gradient of target-loss w.r.t. one-hot suffix.
            grad = self._token_gradients(
                before_ids, adv_ids, after_ids, target_ids
            )  # (k, V)

            # Disallow forbidden tokens by setting their gradient to +inf
            # (so they never enter the top-k of most-negative gradients).
            grad = grad.masked_fill(~allowed_mask.unsqueeze(0), float("inf"))

            # (b) per-position top-k candidate replacements.
            top_k = int(cfg["top_k"])
            top_indices = (-grad).topk(top_k, dim=1).indices  # (k, top_k)

            # (c) sample B candidate suffixes (each differing in one position).
            search_width = int(cfg["search_width"])
            candidate_adv = self._sample_candidates(
                adv_ids, top_indices, search_width, device
            )  # (B, k)

            # (d) evaluate loss of each candidate in mini-batches.
            cand_losses = self._batched_candidate_losses(
                before_ids, candidate_adv, after_ids, target_ids,
                micro_bs=int(cfg["batch_size"]),
            )  # (B,)

            # (e) keep the argmin.
            best_idx = int(cand_losses.argmin().item())
            step_loss = float(cand_losses[best_idx].item())
            adv_ids = candidate_adv[best_idx].clone()
            losses.append(step_loss)

            if step_loss < best_loss:
                best_loss = step_loss
                best_adv_ids = adv_ids.clone()

            if step % 10 == 0:
                log.debug("gcg step=%d loss=%.4f best=%.4f", step, step_loss, best_loss)

        # --- 4. Decode, generate, and score ---------------------------------
        adv_suffix_str = self.tokenizer.decode(best_adv_ids, skip_special_tokens=True)
        adversarial_prompt = f"{prompt} {adv_suffix_str}"

        final_before, final_after, _ = self._build_templated_ids(
            prompt, adv_suffix_str, target, system_prompt
        )
        final_input_ids = torch.cat(
            [final_before.to(device), best_adv_ids, final_after.to(device)]
        ).unsqueeze(0)

        response = self._generate(final_input_ids, max_new_tokens=cfg["max_new_tokens"])
        success = not is_refusal(response)

        adv_positions = list(
            range(
                int(final_before.shape[0]),
                int(final_before.shape[0] + best_adv_ids.shape[0]),
            )
        )

        return AttackResult(
            prompt=prompt,
            adversarial_prompt=adversarial_prompt,
            adversarial_tokens=best_adv_ids.tolist(),
            adversarial_positions=adv_positions,
            response=response,
            success=success,
            loss_trajectory=losses,
            wall_time_s=time.time() - t0,
            metadata={
                "target": target,
                "k_suffix": int(k_suffix),
                "best_loss": best_loss,
            },
        )

    # ================================================================== internals
    def _build_templated_ids(
        self,
        prompt: str,
        suffix_str: str,
        target: str,
        system_prompt: Optional[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render the chat template and slice it into (before, after, target).

        Returns three 1-D LongTensors such that
            full_ids = [before | suffix | after | target]
        tokenises the prompt "{prompt} {suffix}" wrapped in the chat template,
        followed by the target string the model should be forced to emit.
        """
        user_text = f"{prompt} {suffix_str}"
        templated = self._apply_chat_template(user_text, system_prompt)

        marker = "\u2063"  # invisible separator, extremely unlikely in prompts
        user_text_mark = f"{prompt} {marker}{suffix_str}"
        templated_mark = self._apply_chat_template(user_text_mark, system_prompt)

        # Find byte-level split points via string search on the rendered template.
        split_idx = templated_mark.find(marker)
        if split_idx < 0:
            # Marker trick failed (tokenizer drops it). Fall back to searching
            # for the raw suffix occurrence.
            split_idx = templated.rfind(suffix_str)
            before_str = templated[:split_idx]
            after_str = templated[split_idx + len(suffix_str) :]
        else:
            before_str = templated_mark[:split_idx]
            after_str = templated_mark[split_idx + len(marker) + len(suffix_str) :]

        before_ids = self.tokenizer(
            before_str, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        # BOS should be prepended only if the tokenizer normally prepends it
        # and the chat template starts at position 0.
        if (
            self.tokenizer.bos_token_id is not None
            and templated.startswith(self.tokenizer.bos_token or "")
            and (before_ids.numel() == 0 or before_ids[0].item() != self.tokenizer.bos_token_id)
        ):
            before_ids = torch.cat(
                [torch.tensor([self.tokenizer.bos_token_id]), before_ids]
            )

        after_ids = self.tokenizer(
            after_str, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        target_ids = self.tokenizer(
            target, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        return before_ids, after_ids, target_ids

    def _build_allowed_token_mask(self, device: torch.device) -> torch.Tensor:
        """Construct a (V,) boolean mask of allowed candidate tokens.

        We disallow (a) all added/special tokens and (b) any token that
        does not round-trip through decode->encode identically.
        """
        vocab_size = self.model.get_input_embeddings().weight.shape[0]
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)

        # Disallow all known special / added tokens.
        special = set(self.tokenizer.all_special_ids or [])
        added = set(
            getattr(self.tokenizer, "added_tokens_decoder", {}).keys()
        )
        for tid in special | added:
            if 0 <= tid < vocab_size:
                mask[tid] = False

        # Round-trip check (single forward pass over the vocab).
        # Only run for moderately sized vocabs to keep init cost reasonable.
        if vocab_size <= 200_000:
            for tid in range(vocab_size):
                if not mask[tid]:
                    continue
                decoded = self.tokenizer.decode([tid])
                if decoded == "":
                    mask[tid] = False
                    continue
                reenc = self.tokenizer(decoded, add_special_tokens=False)["input_ids"]
                if reenc != [tid]:
                    mask[tid] = False
        return mask

    # ------------------------------------------------------------- gradients
    def _token_gradients(
        self,
        before_ids: torch.Tensor,
        adv_ids: torch.Tensor,
        after_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``∇_{one_hot(adv_ids)} L`` where L is target-sequence NLL.

        We differentiate through a soft embedding ``E = one_hot @ W_emb``
        at the suffix positions, which gives a direct gradient w.r.t. the
        one-hot vectors of the suffix tokens.
        """
        device = self._device()
        embed_weights = self.model.get_input_embeddings().weight  # (V, d)
        vocab_size = embed_weights.shape[0]
        k = adv_ids.shape[0]

        one_hot = torch.zeros(k, vocab_size, device=device, dtype=embed_weights.dtype)
        one_hot.scatter_(1, adv_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)

        adv_embeds = one_hot @ embed_weights  # (k, d)

        with torch.no_grad():
            before_embeds = embed_weights[before_ids]
            after_embeds = embed_weights[after_ids]
            target_embeds = embed_weights[target_ids]

        full_embeds = torch.cat(
            [before_embeds, adv_embeds, after_embeds, target_embeds], dim=0
        ).unsqueeze(0)  # (1, T, d)

        logits = self.model(inputs_embeds=full_embeds).logits  # (1, T, V)

        # The target occupies the last |target_ids| positions. Its loss
        # is computed from the logits at positions [T - |target| - 1, T - 1).
        T = full_embeds.shape[1]
        t_len = target_ids.shape[0]
        tgt_logits = logits[0, T - t_len - 1 : T - 1, :]  # (t_len, V)
        loss = F.cross_entropy(tgt_logits, target_ids)

        grad = torch.autograd.grad(loss, one_hot)[0]  # (k, V)
        # Normalise for numerical stability (paper: per-row L2).
        grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
        return grad.detach()

    # ------------------------------------------------------------- candidates
    @staticmethod
    def _sample_candidates(
        adv_ids: torch.Tensor,
        top_indices: torch.Tensor,
        search_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample ``search_width`` neighbours each differing in exactly one
        position, drawn from its per-position top-k replacement set.
        """
        k = adv_ids.shape[0]
        top_k = top_indices.shape[1]

        pos = torch.randint(0, k, (search_width,), device=device)
        choice = torch.randint(0, top_k, (search_width,), device=device)
        new_tokens = top_indices[pos, choice]

        cand = adv_ids.unsqueeze(0).expand(search_width, k).clone()
        cand[torch.arange(search_width, device=device), pos] = new_tokens
        return cand

    # ------------------------------------------------------------- candidate loss
    @torch.no_grad()
    def _batched_candidate_losses(
        self,
        before_ids: torch.Tensor,
        candidates: torch.Tensor,
        after_ids: torch.Tensor,
        target_ids: torch.Tensor,
        micro_bs: int,
    ) -> torch.Tensor:
        """Forward-pass each candidate suffix and return target NLL per row."""
        device = self._device()
        B, k = candidates.shape
        t_len = target_ids.shape[0]
        b_len = before_ids.shape[0]
        a_len = after_ids.shape[0]
        T = b_len + k + a_len + t_len

        losses = torch.empty(B, device=device)
        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            bs = end - start
            full = torch.empty(bs, T, dtype=torch.long, device=device)
            full[:, :b_len] = before_ids.unsqueeze(0)
            full[:, b_len : b_len + k] = candidates[start:end]
            full[:, b_len + k : b_len + k + a_len] = after_ids.unsqueeze(0)
            full[:, b_len + k + a_len :] = target_ids.unsqueeze(0)

            logits = self.model(input_ids=full).logits  # (bs, T, V)
            tgt_logits = logits[:, T - t_len - 1 : T - 1, :]  # (bs, t_len, V)
            # per-row cross entropy
            logp = F.log_softmax(tgt_logits, dim=-1)
            gathered = logp.gather(2, target_ids.view(1, t_len, 1).expand(bs, -1, -1))
            losses[start:end] = -gathered.squeeze(-1).mean(dim=-1)
        return losses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _request_to_statement(request: str) -> str:
    """Convert an imperative harmful request into a GCG-style continuation.

    GCG's target is typically ``"Sure, here is <reword>"``. We just lowercase
    the first letter and strip trailing punctuation — a crude but universal
    rewrite that matches the paper's examples.
    """
    r = request.strip().rstrip(".!?")
    if not r:
        return "the answer"
    return r[0].lower() + r[1:]
