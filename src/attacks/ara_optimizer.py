"""Continuous optimisation of adversarial token embeddings for ARA.

The optimiser minimises the Safety Attention Score (SAS) of a chat
prompt by adjusting the embeddings of ``k`` adversarial tokens inserted
at caller-specified positions. Optimisation is performed in a
differentiable relaxation of the vocabulary simplex (Gumbel-softmax;
Jang et al. 2017, Maddison et al. 2017), followed by a discrete
projection (see ``token_projector.py``).

Mathematical form
-----------------
Let ``θ ∈ ℝ^{k×V}`` be learnable logits (one row per adversarial
position). At each step we draw Gumbel noise g, form

    p_t^{(m)} = softmax( (θ_m + g_m) / τ )       m = 1..k

and feed soft embeddings  ``e_m^{adv} = p_m E``  (where E is the model's
input embedding matrix) into the forward pass in place of real token
embeddings at positions ``P_adv``. As τ → 0, p_m becomes one-hot; as
τ → ∞ it becomes uniform. We anneal τ exponentially from 1.0 to 0.1.

The loss is SAS(x + e^{adv}) (streaming through attention hooks). Grads
flow: SAS → attention weights → key/query projections → e^{adv} → p_m
→ θ_m. Adam updates θ.

Memory
------
We never materialise the full ``L × H × S × S`` attention tensor. A
streaming accumulator reduces each layer's attention to a scalar SAS
contribution before the tensor is released (see
``src.models.safety_attention_score``).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from src.models.attention_extractor import AttentionExtractor, TokenSpan
from src.models.safety_attention_score import compute_sas

log = logging.getLogger(__name__)


@dataclasses.dataclass
class OptimizerConfig:
    optim_steps: int = 500
    learning_rate: float = 0.1
    lr_schedule: str = "cosine"          # "constant" | "cosine"
    gumbel_temp_start: float = 1.0
    gumbel_temp_end: float = 0.1
    gumbel_anneal: str = "exponential"   # "linear" | "exponential"
    early_stop_sas: float = 0.02
    grad_clip_norm: float = 1.0
    batch_size: int = 1
    logit_init_std: float = 0.01


@dataclasses.dataclass
class OptimizerTrace:
    sas_history: list[float]
    lr_history: list[float]
    temp_history: list[float]
    wall_time_s: float
    converged: bool


class ARAOptimizer:
    """Continuous optimiser for adversarial token embeddings.

    Parameters
    ----------
    extractor : loaded ``AttentionExtractor`` (wraps model + tokenizer)
    config : hyperparameters
    adv_positions : token positions where adversarial embeddings are
        inserted. Must be a strictly increasing sequence within
        ``[0, seq_len + k)`` where k = len(adv_positions).
    """

    def __init__(
        self,
        extractor: AttentionExtractor,
        config: OptimizerConfig,
    ):
        self.extractor = extractor
        self.config = config
        self.model: PreTrainedModel = extractor.model
        self.device = next(self.model.parameters()).device
        self.embedding_layer = self.model.get_input_embeddings()
        # Embedding matrix is tied to the vocabulary.
        self.embedding_matrix: Tensor = self.embedding_layer.weight  # (V, d)
        self.vocab_size, self.d_model = self.embedding_matrix.shape

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    def _lr(self, step: int) -> float:
        t = step / max(1, self.config.optim_steps - 1)
        if self.config.lr_schedule == "constant":
            return self.config.learning_rate
        # cosine
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * t))

    def _temp(self, step: int) -> float:
        t = step / max(1, self.config.optim_steps - 1)
        t0, t1 = self.config.gumbel_temp_start, self.config.gumbel_temp_end
        if self.config.gumbel_anneal == "linear":
            return t0 + (t1 - t0) * t
        # exponential
        return t0 * (t1 / t0) ** t

    # ------------------------------------------------------------------
    # Core optimisation loop
    # ------------------------------------------------------------------
    def optimize(
        self,
        *,
        input_ids: Tensor,
        adv_positions: Sequence[int],
        output_span: TokenSpan,
        safety_spans: Sequence[TokenSpan],
        attention_mask: Optional[Tensor] = None,
        allowed_mask: Optional[Tensor] = None,
        seed: int = 0,
    ) -> tuple[Tensor, OptimizerTrace]:
        """Run continuous optimisation.

        Returns
        -------
        (logits, trace)
            logits : (k, V) final optimised logits. Pass through
                ``token_projector.project_cosine`` to obtain discrete
                token ids.
            trace : per-step SAS / lr / τ history.
        """
        torch.manual_seed(seed)
        k = len(adv_positions)
        if k == 0:
            raise ValueError("need at least one adversarial position")

        # Build the base sequence: insert k placeholder tokens at
        # ``adv_positions`` so that the total sequence has the right
        # length. The placeholder ids are irrelevant — their embeddings
        # will be overwritten by the Gumbel mixture.
        extended_ids, adv_idx_in_extended = self._insert_placeholders(
            input_ids, adv_positions
        )
        extended_mask = self._extend_mask(attention_mask, extended_ids, input_ids)

        # Base embeddings (no grad) for non-adversarial positions.
        with torch.no_grad():
            base_embeds = self.embedding_layer(extended_ids)  # (B, S+k, d)

        # Optimisation variable.
        logits = torch.randn(
            k, self.vocab_size, device=self.device, dtype=torch.float32
        ) * self.config.logit_init_std
        if allowed_mask is not None:
            # Pre-mask disallowed tokens with a large negative bias so
            # they are effectively excluded throughout optimisation.
            # The mask is sized against ``tokenizer.vocab_size`` upstream;
            # however, some families (e.g. LLaMA-3) expose an embedding
            # matrix wider than ``vocab_size`` (added special-token rows).
            # Pad/trim the mask to match the logit width exactly.
            bias = torch.zeros_like(logits)
            vocab_size = bias.shape[1]
            if allowed_mask.shape[0] < vocab_size:
                allowed_mask = torch.nn.functional.pad(
                    allowed_mask,
                    (0, vocab_size - allowed_mask.shape[0]),
                    value=False,
                )
            elif allowed_mask.shape[0] > vocab_size:
                allowed_mask = allowed_mask[:vocab_size]
            bias[:, ~allowed_mask.to(self.device)] = -1e4
            logits = logits + bias
        logits = logits.detach().requires_grad_(True)

        optimiser = torch.optim.Adam([logits], lr=self.config.learning_rate)

        # Work in fp32 for the embedding mixture to avoid drift, then
        # cast to the model's compute dtype on injection.
        emb_dtype = base_embeds.dtype

        trace = OptimizerTrace(
            sas_history=[], lr_history=[], temp_history=[],
            wall_time_s=0.0, converged=False,
        )
        t0 = time.perf_counter()
        E = self.embedding_matrix.to(torch.float32)

        # Shift output/safety spans since we inserted tokens.
        output_span_shifted = self._shift_span(output_span, adv_positions)
        safety_spans_shifted = [self._shift_span(s, adv_positions) for s in safety_spans]

        for step in range(self.config.optim_steps):
            lr = self._lr(step)
            for g in optimiser.param_groups:
                g["lr"] = lr
            tau = self._temp(step)

            # Sample Gumbel noise (with straight-through trick).
            g_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-20) + 1e-20
            )
            soft_probs = F.softmax((logits + g_noise) / tau, dim=-1)  # (k, V)
            adv_embeds = (soft_probs @ E).to(emb_dtype)               # (k, d)

            # Inject adversarial embeddings into base_embeds.
            embeds = base_embeds.clone()
            embeds[:, adv_idx_in_extended, :] = adv_embeds.unsqueeze(0).expand(
                base_embeds.shape[0], -1, -1
            )

            inputs = {"inputs_embeds": embeds}
            if extended_mask is not None:
                inputs["attention_mask"] = extended_mask

            sas = compute_sas(
                self.extractor,
                inputs,
                output_span=output_span_shifted,
                safety_spans=safety_spans_shifted,
            )

            optimiser.zero_grad(set_to_none=True)
            sas.backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([logits], self.config.grad_clip_norm)
            optimiser.step()

            sas_val = float(sas.detach().item())
            trace.sas_history.append(sas_val)
            trace.lr_history.append(lr)
            trace.temp_history.append(tau)

            if sas_val <= self.config.early_stop_sas:
                trace.converged = True
                log.info("ARA early stop at step %d: SAS=%.4f", step, sas_val)
                break
            if step % 50 == 0:
                log.info(
                    "step %4d  SAS=%.4f  lr=%.4g  τ=%.3f",
                    step, sas_val, lr, tau,
                )

        trace.wall_time_s = time.perf_counter() - t0
        return logits.detach(), trace

    # ------------------------------------------------------------------
    # Generic optimisation with user-supplied loss
    # ------------------------------------------------------------------
    def optimize_with_loss_fn(
        self,
        *,
        input_ids: Tensor,
        adv_positions: Sequence[int],
        loss_fn,                       # callable(extended_ids, combined_embeds) -> scalar Tensor
        attention_mask: Optional[Tensor] = None,
        allowed_mask: Optional[Tensor] = None,
        seed: int = 0,
        early_stop_loss: Optional[float] = None,
    ) -> tuple[Tensor, "OptimizerTrace"]:
        """Generalised Gumbel-softmax optimisation. ``loss_fn`` takes
        the extended input_ids and the combined embedding tensor
        (B, S+k, d) and returns a differentiable scalar to minimise.

        Used by variants 2–5 of the aggressive sweep:
          * layer-/head-targeted SAS
          * -log P(target token | context)
          * combined SAS + output loss
        """
        torch.manual_seed(seed)
        k = len(adv_positions)
        if k == 0:
            raise ValueError("need at least one adversarial position")

        extended_ids, adv_idx_in_extended = self._insert_placeholders(
            input_ids, adv_positions
        )
        extended_mask = self._extend_mask(attention_mask, extended_ids, input_ids)

        with torch.no_grad():
            base_embeds = self.embedding_layer(extended_ids)

        logits = torch.randn(
            k, self.vocab_size, device=self.device, dtype=torch.float32
        ) * self.config.logit_init_std
        if allowed_mask is not None:
            bias = torch.zeros_like(logits)
            vocab_size = bias.shape[1]
            if allowed_mask.shape[0] < vocab_size:
                allowed_mask = torch.nn.functional.pad(
                    allowed_mask,
                    (0, vocab_size - allowed_mask.shape[0]),
                    value=False,
                )
            elif allowed_mask.shape[0] > vocab_size:
                allowed_mask = allowed_mask[:vocab_size]
            bias[:, ~allowed_mask.to(self.device)] = -1e4
            logits = logits + bias
        logits = logits.detach().requires_grad_(True)

        optimiser = torch.optim.Adam([logits], lr=self.config.learning_rate)
        emb_dtype = base_embeds.dtype

        trace = OptimizerTrace(
            sas_history=[], lr_history=[], temp_history=[],
            wall_time_s=0.0, converged=False,
        )
        t0 = time.perf_counter()
        E = self.embedding_matrix.to(torch.float32)

        stop = early_stop_loss if early_stop_loss is not None else self.config.early_stop_sas

        for step in range(self.config.optim_steps):
            lr = self._lr(step)
            for g in optimiser.param_groups:
                g["lr"] = lr
            tau = self._temp(step)

            g_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-20) + 1e-20
            )
            soft_probs = F.softmax((logits + g_noise) / tau, dim=-1)
            adv_embeds = (soft_probs @ E).to(emb_dtype)

            embeds = base_embeds.clone()
            embeds[:, adv_idx_in_extended, :] = adv_embeds.unsqueeze(0).expand(
                base_embeds.shape[0], -1, -1
            )

            loss = loss_fn(extended_ids, embeds, extended_mask, adv_idx_in_extended)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([logits], self.config.grad_clip_norm)
            optimiser.step()

            loss_val = float(loss.detach().item())
            trace.sas_history.append(loss_val)   # field reused as generic loss
            trace.lr_history.append(lr)
            trace.temp_history.append(tau)

            if loss_val <= stop:
                trace.converged = True
                log.info("optimize_with_loss_fn early stop at %d: loss=%.4f", step, loss_val)
                break
            if step % 50 == 0:
                log.info(
                    "step %4d  loss=%.4f  lr=%.4g  τ=%.3f",
                    step, loss_val, lr, tau,
                )

        trace.wall_time_s = time.perf_counter() - t0
        return logits.detach(), trace

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _insert_placeholders(
        self,
        input_ids: Tensor,
        adv_positions: Sequence[int],
    ) -> tuple[Tensor, list[int]]:
        """Insert pad-token placeholders at ``adv_positions``, translated
        to indices in the extended sequence.

        Positions in ``adv_positions`` are interpreted as insertion
        points in the ORIGINAL sequence: position p means "insert
        immediately before input_ids[p]". Use p = len(input_ids) to
        append. Multiple positions are processed in order, each
        accounting for prior insertions.
        """
        B, S = input_ids.shape
        assert B == 1, "batch size >1 not supported for single-prompt attack"
        pad_id = self.extractor.tokenizer.pad_token_id or 0

        # Normalise positions (support negative = from end of original seq).
        positions = [
            p if p >= 0 else S + 1 + p for p in adv_positions
        ]
        if any(p < 0 or p > S for p in positions):
            raise ValueError(f"positions out of range [0, {S}]: {positions}")

        # Sort and assign final extended indices.
        order = sorted(range(len(positions)), key=lambda i: positions[i])
        sorted_positions = [positions[i] for i in order]

        extended = input_ids[0].tolist()
        adv_extended_indices: list[Optional[int]] = [None] * len(positions)
        for rank, orig_i in enumerate(order):
            insert_at = sorted_positions[rank] + rank  # shift by prior insertions
            extended.insert(insert_at, pad_id)
            adv_extended_indices[orig_i] = insert_at

        return (
            torch.tensor([extended], device=self.device, dtype=input_ids.dtype),
            [i for i in adv_extended_indices if i is not None],
        )

    def _extend_mask(
        self,
        mask: Optional[Tensor],
        extended_ids: Tensor,
        original_ids: Tensor,
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        # Adversarial tokens are always real content — mask = 1.
        B, S_new = extended_ids.shape
        ext = torch.ones(B, S_new, dtype=mask.dtype, device=mask.device)
        # Preserve original mask for pre-existing positions where they
        # remain aligned (padding is expected to be on the right; we do
        # not attempt to realign mid-sequence pads).
        return ext

    def _shift_span(
        self,
        span: TokenSpan,
        adv_positions: Sequence[int],
    ) -> TokenSpan:
        """Adjust a span's indices given that ``k`` tokens were inserted
        at the given positions (in the original sequence's coordinate)."""
        def shift(p: int) -> int:
            # count insertions at positions <= p
            return p + sum(1 for ap in adv_positions if ap <= p)
        return TokenSpan(shift(span.start), shift(span.end), label=span.label)
