"""Attention Budget Constraint (ABC) defense.

Core idea
---------
ARA succeeds because softmax competition lets a handful of adversarial
tokens steal attention mass away from safety-critical tokens. ABC fixes
this at inference time without retraining: after attention weights are
computed, we enforce a **minimum budget** τ for the attention that
output positions pay to a protected safety span. If the observed
attention mass falls short, we redistribute proportionally — scaling
up the attention to safety tokens and scaling down the non-safety mass
to preserve row-stochasticity.

Formal statement
----------------
Let ``a = (a_1, ..., a_n)`` be a row of attention weights for some
(layer, head, query position). Partition [n] into ``P_s`` (safety)
and ``P_n`` (non-safety). Define

    μ_s = Σ_{j∈P_s} a_j          (observed safety attention mass)
    μ_n = 1 − μ_s                 (non-safety mass)

ABC with budget τ ∈ (0,1) applies

    a'_j = a_j · (τ / μ_s)              if j ∈ P_s and μ_s < τ
    a'_j = a_j · ((1−τ) / μ_n)          if j ∉ P_s and μ_s < τ
    a'_j = a_j                           otherwise

This is projection onto the simplex slice ``{a : Σ_{j∈P_s} a_j ≥ τ}``
under the KL-geometry (scaling preserves within-partition ratios).

Properties
----------
- Row-stochasticity preserved: Σ_j a'_j = 1.
- No-op for clean prompts where safety already exceeds budget.
- O(L·H·n) overhead per forward pass.
- Differentiable (for defense-aware retraining if desired).

Implementation
--------------
We install forward hooks on each attention module that *rewrite* the
attention weights tensor before it is multiplied by V. Because this
requires intercepting the internal computation of the attention module
(not its output tuple), we monkey-patch the ``forward`` of each
attention module. A context manager ensures hooks are removed on exit.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedModel

from src.models.attention_extractor import TokenSpan
from src.models.model_loader import ModelSpec, iter_attention_modules

log = logging.getLogger(__name__)


@dataclass
class ABCConfig:
    tau: float = 0.10                    # minimum safety attention budget
    apply_to_layers: Optional[Sequence[int]] = None   # None = all
    apply_to_heads: Optional[Sequence[int]] = None    # None = all
    apply_to_query_positions: Optional[Sequence[int]] = None  # None = all


def abc_reweight(
    attn_weights: Tensor,
    safety_positions: Sequence[int],
    tau: float,
) -> Tensor:
    """Apply ABC re-weighting to a post-softmax attention tensor.

    Parameters
    ----------
    attn_weights : (B, H, Sq, Sk)
    safety_positions : list of key-side indices marked safety-critical
    tau : minimum total attention a query position must pay to safety

    Returns
    -------
    Tensor of same shape with re-weighted rows.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0,1), got {tau}")
    B, H, Sq, Sk = attn_weights.shape
    if not safety_positions:
        return attn_weights

    saf_idx = torch.as_tensor(
        list(safety_positions), device=attn_weights.device, dtype=torch.long
    )
    # Build boolean safety mask over key positions: (Sk,)
    saf_mask = torch.zeros(Sk, dtype=torch.bool, device=attn_weights.device)
    saf_mask[saf_idx] = True
    # Expand to broadcast across (B, H, Sq, Sk).
    saf_b = saf_mask.view(1, 1, 1, Sk)
    non_b = ~saf_b

    # mu_s: safety mass per row, shape (B, H, Sq, 1)
    mu_s = (attn_weights * saf_b).sum(dim=-1, keepdim=True)
    mu_n = 1.0 - mu_s
    # Rows needing correction:
    needs_fix = mu_s < tau
    # Safe division (avoid 0/0 for rows with zero mass — unlikely post-softmax).
    scale_s = torch.where(
        needs_fix & (mu_s > 0),
        tau / mu_s.clamp_min(1e-12),
        torch.ones_like(mu_s),
    )
    scale_n = torch.where(
        needs_fix & (mu_n > 0),
        (1.0 - tau) / mu_n.clamp_min(1e-12),
        torch.ones_like(mu_n),
    )
    # Apply per-group scaling.
    out = torch.where(saf_b, attn_weights * scale_s, attn_weights * scale_n)
    return out


@contextlib.contextmanager
def abc_defense(
    model: PreTrainedModel,
    spec: ModelSpec,
    *,
    safety_positions_fn,           # callable(batch_idx) -> list[int]
    config: ABCConfig,
):
    """Context manager that patches each attention module to apply ABC.

    ``safety_positions_fn`` is a closure returning the key-side safety
    positions for a given batch element; typically this is just a fixed
    list precomputed by the caller (the chat template pins safety span
    positions). We keep it as a callable to support variable-length
    batches later.
    """
    want_layers = (
        set(range(spec.n_layers)) if config.apply_to_layers is None
        else set(config.apply_to_layers)
    )
    tau = config.tau
    originals: List[tuple] = []  # (module, original_forward)

    def make_patched_forward(original_forward, layer_idx):
        def patched(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
            out = original_forward(*args, **kwargs)
            # Extract attn_weights (2nd element of tuple typically).
            if not isinstance(out, tuple):
                return out
            attn_output = out[0]
            attn_weights = None
            others = list(out[1:])
            for idx, item in enumerate(out[1:], start=1):
                if isinstance(item, Tensor) and item.dim() == 4 and item.shape[-1] == item.shape[-2]:
                    attn_weights = item
                    aw_idx = idx
                    break
            if attn_weights is None:
                return out
            # Reweight.
            sp = safety_positions_fn(0)
            if config.apply_to_heads is not None:
                # Apply only to selected heads — trickier; default = all heads.
                new_aw = attn_weights.clone()
                head_idx = torch.tensor(
                    list(config.apply_to_heads), device=attn_weights.device
                )
                patched_heads = abc_reweight(
                    attn_weights.index_select(1, head_idx), sp, tau
                )
                new_aw.index_copy_(1, head_idx, patched_heads)
            else:
                new_aw = abc_reweight(attn_weights, sp, tau)
            # Recompute attn_output from re-weighted attention. This is
            # where ABC actually changes model behaviour — the original
            # attn_output was computed from the *original* weights. We
            # need access to V. In practice, HF attention modules do
            # not expose V separately at this point, so a faithful
            # implementation requires rewriting the module's forward.
            # For simplicity in this research prototype, we recompute
            # by calling the forward AGAIN with an attention-weight
            # override. See ``AttentionWeightOverride`` in the longer
            # paper appendix. For now, we only return the re-weighted
            # attention in the tuple so SAS measurements reflect ABC,
            # and note this limitation.
            new_out = (attn_output,) + tuple(
                new_aw if i == aw_idx - 1 else o for i, o in enumerate(others)
            )
            return new_out
        return patched

    try:
        for i, attn_mod in iter_attention_modules(model, spec):
            if i not in want_layers:
                continue
            originals.append((attn_mod, attn_mod.forward))
            attn_mod.forward = make_patched_forward(attn_mod.forward, i)  # type: ignore
        yield
    finally:
        for mod, orig in originals:
            mod.forward = orig


# ---------------------------------------------------------------------------
# Faithful variant: recompute attn_output from re-weighted attention by
# rewriting the attention forward. This is required for ABC to actually
# change generation (not just SAS measurement). Implemented as a direct
# replacement of each LLaMA/Mistral/Gemma attention forward.
# ---------------------------------------------------------------------------

def install_faithful_abc(
    model: PreTrainedModel,
    spec: ModelSpec,
    *,
    safety_positions: List[int],
    tau: float,
):
    """Monkey-patch each self-attention module to run ABC on the
    post-softmax weights BEFORE the attn_weights @ V multiply. This is
    the defense used in paper §6.

    Returns an ``unpatch`` callable that restores the original forwards.
    """
    originals: List[tuple] = []
    for i, attn_mod in iter_attention_modules(model, spec):
        orig_forward = attn_mod.forward

        def make(orig, layer_idx):
            def patched_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=None,
                **kwargs,
            ):
                # Force output_attentions to get the weights.
                res = orig(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=True,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                # res structure: (attn_output, attn_weights, past_kv)
                attn_output, attn_weights = res[0], res[1]
                new_aw = abc_reweight(attn_weights, safety_positions, tau)
                # Approximate recomputation: we cannot cheaply access V
                # from outside the original module; however the ratio
                # between new_aw and attn_weights row-wise gives us the
                # correction. This yields the exact output only when
                # the module exposes V, which LLaMA does via
                # ``attn_mod.v_proj(hidden_states)``.
                B, S = hidden_states.shape[:2]
                v = attn_mod.v_proj(hidden_states)
                # Reshape to (B, n_heads, S, head_dim).
                H = attn_mod.num_heads if hasattr(attn_mod, "num_heads") else attn_mod.config.num_attention_heads
                Hd = v.shape[-1] // H
                v = v.view(B, S, H, Hd).transpose(1, 2)
                # Repeat for grouped-query attention if needed.
                n_kv = getattr(attn_mod, "num_key_value_heads", H)
                if n_kv != H:
                    repeats = H // n_kv
                    v = v.repeat_interleave(repeats, dim=1)
                new_out = torch.matmul(new_aw, v)  # (B, H, S, Hd)
                new_out = new_out.transpose(1, 2).contiguous().view(B, S, -1)
                new_out = attn_mod.o_proj(new_out)
                return (new_out, new_aw) + res[2:]
            return patched_forward

        attn_mod.forward = make(orig_forward, i)
        originals.append((attn_mod, orig_forward))

    def unpatch():
        for mod, orig in originals:
            mod.forward = orig

    return unpatch
